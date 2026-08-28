//! Atomic semantic resource-lifecycle transactions and native retirement handoff.
//!
//! Creation and namespace mutation live in [`ResourceGraph`]. Native objects
//! live in [`ManagedBackingOwner`]. This owner is the seam between them: every
//! backing created in the graph is registered once with the native epoch, and
//! graph retirement hands the same canonical identity to timeline retirement.

use crate::{
    BackingRegion, BackingView, GpuWriteBatchError, GpuWriteRequest, GpuWriteReservation,
    GraphError, HostIngressKey, HostIngressTransfer, HostLandingKey, ManagedBackingError,
    ManagedBackingOwner, ManagedBackingProgress, ManagedRepresentationFailure, MappingNode,
    QueueTimelinePoint, RegionVersion, RepresentationRoute, RepresentationUse, ResourceGraph,
    ResourceValidity, StorageBacking, StorageNode, TransferBatchError, TransferKey,
    GUEST_REPRESENTATION, HOST_REPRESENTATION,
};
use reims_vgpu_protocol::{
    BackingGeneration, BackingId, ByteLength, ByteOffset, HeapObject, MappingId, ObjectKind,
    ObjectTableRef, RepresentationId, ResourceDescriptor, ResourceId, ResourceObject,
    ResourceValidityOps, SubmissionId, TaskId, TransactionId, VulkanDeviceEpochId,
};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedValidityTarget {
    pub backing: BackingId,
    pub regions: Box<[BackingRegion]>,
    /// Native representation that receives a set-host GPU write.
    pub host_representation: Option<RepresentationId>,
    /// Fixed host-staging destination for guest-authored bytes. Imported guest
    /// storage remains authoritative directly and therefore has no ingress.
    pub host_ingress_destination: Option<RepresentationId>,
    /// Execution representation that must receive the guest-authored version
    /// before ordinary GPU work may read it.
    pub guest_upload_destination: Option<RepresentationId>,
    /// Current source used when clear-guest requests guest visibility.
    pub guest_visibility_source: Option<RepresentationId>,
    /// Native destination of the visibility transfer. Host staging uses the
    /// reserved host identity and owes a later CPU landing into guest bytes.
    pub guest_visibility_destination: RepresentationId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedValidityTransition {
    pub ops: ResourceValidityOps,
    /// The EXEC whose eventual GPU completion makes set-host content current.
    pub write: Option<crate::GpuWriteId>,
    /// The operation whose statement the clear-host guest write is, so
    /// re-preparing it repeats that statement rather than making a new one.
    /// See [`crate::GuestWriteId`].
    pub guest_write: Option<crate::GuestWriteId>,
    pub targets: Box<[ResolvedValidityTarget]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidityRepresentations {
    pub host_write: Option<RepresentationId>,
    pub host_ingress_destination: Option<RepresentationId>,
    pub guest_upload_destination: Option<RepresentationId>,
    pub guest_visibility_source: Option<RepresentationId>,
    pub guest_visibility_destination: RepresentationId,
}

impl Default for ValidityRepresentations {
    fn default() -> Self {
        Self {
            host_write: None,
            host_ingress_destination: None,
            guest_upload_destination: None,
            guest_visibility_source: None,
            guest_visibility_destination: GUEST_REPRESENTATION,
        }
    }
}

impl ResolvedValidityTransition {
    /// Bind backend representation identities only after the semantic command
    /// has resolved its canonical backing coverage.
    pub fn bind(
        state: &crate::ResolvedResourceState,
        write: Option<crate::GpuWriteId>,
        guest_write: Option<crate::GuestWriteId>,
        mut representations: impl FnMut(BackingId) -> ValidityRepresentations,
    ) -> Self {
        Self {
            ops: state.ops,
            write,
            guest_write,
            targets: state
                .targets
                .iter()
                .map(|target| {
                    let resolved = representations(target.backing);
                    ResolvedValidityTarget {
                        backing: target.backing,
                        regions: target.regions.clone(),
                        host_representation: resolved.host_write,
                        host_ingress_destination: resolved.host_ingress_destination,
                        guest_upload_destination: resolved.guest_upload_destination,
                        guest_visibility_source: resolved.guest_visibility_source,
                        guest_visibility_destination: resolved.guest_visibility_destination,
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedResourceLifecycle {
    CreateBacking {
        backing: StorageBacking,
        regions: Box<[BackingRegion]>,
    },
    CreateResource {
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        kind: ObjectKind,
        descriptor: Arc<ResourceDescriptor>,
        storage: Option<BackingId>,
        parents: Box<[ResourceId<ResourceObject>]>,
    },
    CreateBufferTexture {
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        descriptor: Arc<reims_vgpu_protocol::BufferTextureDescriptor>,
        buffer: ResourceId<ResourceObject>,
    },
    CreateHeapTexture {
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        descriptor: Arc<reims_vgpu_protocol::HeapTextureDescriptor>,
        heap: ResourceId<HeapObject>,
        size: ByteLength,
        alignment: ByteLength,
    },
    CreateMapperTexture {
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        descriptor: Arc<reims_vgpu_protocol::MapperIOSurfaceTextureView>,
    },
    ReleaseResource {
        resource: ResourceId<ResourceObject>,
    },
    /// Release a whole resource subtree and retire every backing it owned.
    ///
    /// A registered IOSurface owns no backing itself and its plane views own
    /// one each, so a tree release retires a *set*: one backing for a single
    /// plane surface and one per plane for a biplanar one.
    ///
    /// The set is derived from `root` when the command applies, not carried
    /// from where it was admitted. The guest means "this surface and whatever
    /// hangs off it", and what hangs off it can change between admission and
    /// the head of the channel: a plane view declared in between belongs to
    /// the tree the guest is deleting.
    ReleaseResourceTree {
        root: ResourceId<ResourceObject>,
    },
    /// Release every resource the task still owns, derived at apply for the
    /// same reason as [`Self::ReleaseResourceTree`].
    ReleaseTask {
        task: TaskId,
    },
    CreateMapping(MappingNode),
    ReleaseMapping {
        mapping: MappingId,
    },
    ReplaceBacking {
        resource: ResourceId<ResourceObject>,
        backing: BackingId,
    },
    /// The packet advances the resource's physical incarnation. It does not
    /// replace the logical backing identity or its content authority.
    ReplacePhysical {
        resource: ResourceId<ResourceObject>,
    },
    ReplacePhysicalBatch {
        resources: Box<[ResourceId<ResourceObject>]>,
    },
    RetireBacking {
        backing: BackingId,
    },
    GuestWrite {
        backing: BackingId,
        region: BackingRegion,
    },
    Discard {
        backing: BackingId,
        region: BackingRegion,
    },
    Synchronize {
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        regions: Box<[BackingRegion]>,
    },
    ApplyValidity(ResolvedValidityTransition),
    ApplyValidityBatch(Box<[ResolvedValidityTransition]>),
}

#[derive(Debug)]
pub struct RetiredBacking<T> {
    pub storage: StorageNode,
    pub native: ManagedBackingProgress<T>,
}

#[derive(Debug)]
pub enum ResourceLifecycleEffect<T> {
    BackingCreated(BackingId),
    ResourceCreated(ResourceId<ResourceObject>),
    ResourceReleased {
        resource: ResourceId<ResourceObject>,
        automatically_retired: Vec<RetiredBacking<T>>,
    },
    ResourceTreeReleased {
        root: ResourceId<ResourceObject>,
        resources: Box<[ResourceId<ResourceObject>]>,
        retired: Box<[RetiredBacking<T>]>,
    },
    TaskReleased {
        task: TaskId,
        resources: Box<[ResourceId<ResourceObject>]>,
        automatically_retired: Vec<RetiredBacking<T>>,
    },
    MappingCreated(MappingId),
    MappingReleased(MappingNode),
    BackingReplaced(BackingGeneration),
    PhysicalReplaced {
        backing: Option<BackingId>,
        generation: BackingGeneration,
        native: Option<ManagedBackingProgress<T>>,
    },
    PhysicalBatchReplaced {
        resources: Box<[(ResourceId<ResourceObject>, BackingGeneration)]>,
        native: Box<[(BackingId, ManagedBackingProgress<T>)]>,
    },
    BackingRetired(RetiredBacking<T>),
    GuestWrite(RegionVersion),
    Discarded,
    TransfersPlanned(Box<[TransferKey]>),
    ValidityApplied {
        guest_writes: Box<[(BackingId, RegionVersion)]>,
        host_ingresses: Box<[HostIngressKey]>,
        deferred_host_ingress_transfers: Box<[HostIngressTransfer]>,
        gpu_reservations: Box<[GpuWriteReservation]>,
        gpu_completions: Box<[ResolvedResourceCompletion]>,
        transfers: Box<[TransferKey]>,
        host_landings: Box<[HostLandingKey]>,
        states: Box<[(BackingId, ResourceValidity)]>,
    },
    ValidityBatchApplied(Box<[ResourceLifecycleEffect<T>]>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLifecycleError {
    Graph(GraphError),
    Native(ManagedBackingError),
    BufferTextureRelationMismatch,
    HeapTextureRelationMismatch,
    InvalidHeapTextureRequirements,
    HeapTextureOffsetMisaligned,
    MapperTextureRelationMismatch,
    EmptyValidityBatch,
    DuplicateValidityBatchBacking(BackingId),
    DuplicatePhysicalReplacement(ResourceId<ResourceObject>),
    Validity(ValidityTransitionError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidityTransitionError {
    EmptyTargets,
    DuplicateBacking(BackingId),
    EmptyRegions(BackingId),
    TooManyRegions(BackingId),
    MissingSubmission,
    UnexpectedSubmission,
    MissingHostRepresentation(BackingId),
    UnexpectedHostRepresentation(BackingId),
    UnexpectedHostIngressDestination(BackingId),
    InvalidHostIngressDestination {
        backing: BackingId,
        destination: RepresentationId,
    },
    UnexpectedGuestUploadDestination(BackingId),
    /// No route this backing's bytes could be uploaded through.
    ///
    /// The backing alone does not say which of the four ways this fails, and
    /// a boot has already been spent on one of these reporting nothing but
    /// the name: whether a destination was chosen at all, what route it
    /// carries, and what route the guest representation carries are three
    /// independent facts and the refusal is legible only with all three. A
    /// `destination` of `None` beside a host ingress is "nothing designated";
    /// a `HostStagingTransfer` destination means the host ingress did not
    /// pair with it; an `ImportedGuestTransfer` destination means the backing
    /// carries no direct guest alias, which is what that route reads through.
    /// `GUEST_REPRESENTATION` is reserved to `DirectGuestAlias` at creation,
    /// so `guest_route` reads `None` when the alias is absent and never some
    /// third route --- it is there because "absent" is the answer, not a
    /// missing one.
    InvalidGuestUploadRoute {
        backing: BackingId,
        destination: Option<RepresentationId>,
        destination_route: Option<RepresentationRoute>,
        guest_route: Option<RepresentationRoute>,
    },
    MissingGuestVisibilitySource(BackingId),
    UnexpectedGuestVisibilitySource(BackingId),
    InvalidGuestVisibilityDestination {
        backing: BackingId,
        destination: RepresentationId,
    },
    Backing {
        backing: BackingId,
        reason: ManagedBackingError,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceUseBatchError {
    DuplicateBacking(BackingId),
    Backing {
        backing: BackingId,
        reason: ManagedBackingError,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ResolvedResourceCompletion {
    GpuWrite {
        backing: BackingId,
        write: crate::GpuWriteId,
        representation: RepresentationId,
    },
    ValidityHostWrite {
        backing: BackingId,
        write: crate::GpuWriteId,
        representation: RepresentationId,
    },
    Transfer(TransferKey),
    Discard {
        backing: BackingId,
        region: BackingRegion,
    },
}

impl ResolvedResourceCompletion {
    pub const fn backing(self) -> BackingId {
        match self {
            Self::GpuWrite { backing, .. }
            | Self::ValidityHostWrite { backing, .. }
            | Self::Discard { backing, .. } => backing,
            Self::Transfer(transfer) => transfer.backing,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceCompletionBatchError {
    Duplicate(ResolvedResourceCompletion),
    Completion {
        completion: ResolvedResourceCompletion,
        reason: ManagedBackingError,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostLandingBatchError {
    Duplicate(HostLandingKey),
    StagedTransferAbsent(HostLandingKey),
    Landing {
        landing: HostLandingKey,
        reason: ManagedBackingError,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostIngressBatchError {
    Duplicate(HostIngressKey),
    DuplicateTransfer(HostIngressTransfer),
    Ingress {
        ingress: HostIngressKey,
        reason: ManagedBackingError,
    },
    Transfer {
        transfer: HostIngressTransfer,
        reason: ManagedBackingError,
    },
}

#[derive(Debug, Eq, PartialEq)]
pub enum ResourceCompletionEffect {
    GpuWrite {
        backing: BackingId,
        submission: SubmissionId,
        regions: Box<[RegionVersion]>,
    },
    ValidityHostWrite {
        backing: BackingId,
        submission: SubmissionId,
        regions: Box<[RegionVersion]>,
        state: ResourceValidity,
    },
    Transfer(TransferKey),
    Discard {
        backing: BackingId,
        region: BackingRegion,
    },
}

impl From<GraphError> for ResourceLifecycleError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<ManagedBackingError> for ResourceLifecycleError {
    fn from(error: ManagedBackingError) -> Self {
        Self::Native(error)
    }
}

pub struct ResourceLifecycleOwner<T> {
    graph: ResourceGraph,
    native: ManagedBackingOwner<T>,
}

impl<T> ResourceLifecycleOwner<T> {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            graph: ResourceGraph::default(),
            native: ManagedBackingOwner::new(epoch),
        }
    }

    pub const fn graph(&self) -> &ResourceGraph {
        &self.graph
    }

    /// See [`crate::ManagedBackingOwner::representation_census`].
    pub fn representation_census(&self, backing: BackingId) -> Vec<crate::RepresentationCensus> {
        self.native.representation_census(backing)
    }

    pub fn backing_validity(&self, backing: BackingId) -> Option<ResourceValidity> {
        self.native.validity(backing)
    }

    pub fn validate_validity_transition(
        &self,
        transition: &ResolvedValidityTransition,
    ) -> Result<(), ValidityTransitionError> {
        self.validate_validity(transition)
    }

    pub fn representation_matches(
        &self,
        backing: BackingId,
        representation: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<bool, ManagedBackingError> {
        self.native
            .representation_matches(backing, representation, snapshot)
    }

    pub fn current_native_representation_for_snapshot(
        &self,
        backing: BackingId,
        excluded: &[RepresentationId],
        snapshot: &[RegionVersion],
    ) -> Result<Option<RepresentationId>, ManagedBackingError> {
        self.native
            .current_native_representation_for_snapshot(backing, excluded, snapshot)
    }

    pub fn apply(
        &mut self,
        command: ResolvedResourceLifecycle,
    ) -> Result<ResourceLifecycleEffect<T>, ResourceLifecycleError> {
        match command {
            ResolvedResourceLifecycle::CreateBacking { backing, regions } => {
                let id = self.graph.create_storage_with_regions(backing, regions)?;
                let authority = self.graph.storage(id).unwrap().content.clone();
                if let Err(error) = self.native.register_backing(id, authority) {
                    let _ = self.graph.retire_storage(id);
                    return Err(error.into());
                }
                Ok(ResourceLifecycleEffect::BackingCreated(id))
            }
            ResolvedResourceLifecycle::CreateResource {
                task,
                object,
                kind,
                descriptor,
                storage,
                parents,
            } => self
                .graph
                .create_resource_with_descriptor(
                    task,
                    object,
                    kind,
                    Some(descriptor),
                    storage,
                    parents,
                )
                .map(ResourceLifecycleEffect::ResourceCreated)
                .map_err(Into::into),
            ResolvedResourceLifecycle::CreateBufferTexture {
                task,
                object,
                descriptor,
                buffer,
            } => self.create_buffer_texture(task, object, descriptor, buffer),
            ResolvedResourceLifecycle::CreateHeapTexture {
                task,
                object,
                descriptor,
                heap,
                size,
                alignment,
            } => self.create_heap_texture(task, object, descriptor, heap, size, alignment),
            ResolvedResourceLifecycle::CreateMapperTexture {
                task,
                object,
                descriptor,
            } => self.create_mapper_texture(task, object, descriptor),
            ResolvedResourceLifecycle::ReleaseResource { resource } => {
                self.graph.release_resource(resource)?;
                let automatically_retired = self.retire_automatic_storage()?;
                Ok(ResourceLifecycleEffect::ResourceReleased {
                    resource,
                    automatically_retired,
                })
            }
            ResolvedResourceLifecycle::ReleaseResourceTree { root } => {
                let resources = self
                    .graph
                    .live_resource_tree_child_first(root)
                    .ok_or(GraphError::ResourceAbsent)?;
                let backings = self.graph.resource_tree_backings(&resources);
                for &backing in backings.iter() {
                    self.graph
                        .validate_storage_retirement_after_resources(backing, &resources)?;
                    self.native.validate_begin_retirement(backing)?;
                }
                for &resource in resources.iter() {
                    self.graph.release_resource(resource)?;
                }
                let retired = backings
                    .iter()
                    .map(|&backing| {
                        let storage = self
                            .graph
                            .retire_storage(backing)
                            .expect("tree release was prevalidated to leave its backing unowned");
                        let native = self
                            .native
                            .begin_retirement(backing)
                            .expect("tree release native retirement was prevalidated");
                        RetiredBacking { storage, native }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                Ok(ResourceLifecycleEffect::ResourceTreeReleased {
                    root,
                    resources,
                    retired,
                })
            }
            ResolvedResourceLifecycle::ReleaseTask { task } => {
                let declared = self.graph.live_resources_for_task(task).to_vec();
                for resource in &declared {
                    self.graph.release_resource(*resource)?;
                }
                let automatically_retired = self.retire_automatic_storage()?;
                Ok(ResourceLifecycleEffect::TaskReleased {
                    task,
                    resources: declared.into_boxed_slice(),
                    automatically_retired,
                })
            }
            ResolvedResourceLifecycle::CreateMapping(mapping) => {
                let id = mapping.id;
                self.graph.create_mapping(mapping)?;
                Ok(ResourceLifecycleEffect::MappingCreated(id))
            }
            ResolvedResourceLifecycle::ReleaseMapping { mapping } => self
                .graph
                .release_mapping(mapping)
                .map(ResourceLifecycleEffect::MappingReleased)
                .map_err(Into::into),
            ResolvedResourceLifecycle::ReplaceBacking { resource, backing } => self
                .graph
                .replace_backing(resource, backing)
                .map(ResourceLifecycleEffect::BackingReplaced)
                .map_err(Into::into),
            ResolvedResourceLifecycle::ReplacePhysical { resource } => {
                let (backing, generation) = self.graph.replace_physical(resource)?;
                let native = backing
                    .map(|backing| self.native.replace_execution_representation(backing))
                    .transpose()?;
                Ok(ResourceLifecycleEffect::PhysicalReplaced {
                    backing,
                    generation,
                    native,
                })
            }
            ResolvedResourceLifecycle::ReplacePhysicalBatch { resources } => {
                let mut unique_resources = std::collections::BTreeSet::new();
                let mut backings = std::collections::BTreeSet::new();
                for &resource in resources.iter() {
                    if !unique_resources.insert(resource) {
                        return Err(ResourceLifecycleError::DuplicatePhysicalReplacement(
                            resource,
                        ));
                    }
                    let node = self
                        .graph
                        .resource(resource)
                        .ok_or(GraphError::ResourceAbsent)?;
                    if let Some(backing) = node.storage {
                        backings.insert(backing);
                    }
                }
                for &backing in &backings {
                    self.native
                        .validate_replace_execution_representation(backing)?;
                }
                let replaced = resources
                    .iter()
                    .copied()
                    .map(|resource| {
                        self.graph
                            .replace_physical(resource)
                            .map(|(_, generation)| (resource, generation))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let native = backings
                    .into_iter()
                    .map(|backing| {
                        self.native
                            .replace_execution_representation(backing)
                            .map(|progress| (backing, progress))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| {
                        unreachable!("the complete physical replacement batch was prevalidated")
                    });
                Ok(ResourceLifecycleEffect::PhysicalBatchReplaced {
                    resources: replaced.into_boxed_slice(),
                    native: native.into_boxed_slice(),
                })
            }
            ResolvedResourceLifecycle::RetireBacking { backing } => {
                self.native.validate_begin_retirement(backing)?;
                let storage = self.graph.retire_storage(backing)?;
                let native = self.native.begin_retirement(backing).unwrap_or_else(|_| {
                    unreachable!("retirement was validated before graph ownership moved")
                });
                Ok(ResourceLifecycleEffect::BackingRetired(RetiredBacking {
                    storage,
                    native,
                }))
            }
            ResolvedResourceLifecycle::GuestWrite { backing, region } => self
                .native
                .guest_write(backing, None, region)
                .map(ResourceLifecycleEffect::GuestWrite)
                .map_err(Into::into),
            ResolvedResourceLifecycle::Discard { backing, region } => {
                self.native.discard(backing, region)?;
                Ok(ResourceLifecycleEffect::Discarded)
            }
            ResolvedResourceLifecycle::Synchronize {
                backing,
                source,
                destination,
                regions,
            } => {
                let snapshot = self.native.snapshot_content(backing, &regions)?;
                self.native
                    .plan_transfers(backing, source, destination, &snapshot)
                    .map(ResourceLifecycleEffect::TransfersPlanned)
                    .map_err(Into::into)
            }
            ResolvedResourceLifecycle::ApplyValidity(transition) => self.apply_validity(transition),
            ResolvedResourceLifecycle::ApplyValidityBatch(transitions) => {
                self.apply_validity_batch(transitions)
            }
        }
    }

    fn create_buffer_texture(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        descriptor: Arc<reims_vgpu_protocol::BufferTextureDescriptor>,
        buffer: ResourceId<ResourceObject>,
    ) -> Result<ResourceLifecycleEffect<T>, ResourceLifecycleError> {
        let parent = self
            .graph
            .resource(buffer)
            .ok_or(ResourceLifecycleError::Graph(GraphError::ParentAbsent))?;
        if parent.task != task
            || parent.object.get() != descriptor.buffer_ref
            || parent.kind != reims_vgpu_protocol::ObjectKind::Buffer
            || descriptor.new_texture_ref != object.get()
        {
            return Err(ResourceLifecycleError::BufferTextureRelationMismatch);
        }
        let offset = reims_vgpu_protocol::ByteOffset::new(descriptor.offset);
        let bytes_per_row = reims_vgpu_protocol::ByteLength::new(descriptor.bytes_per_row);
        let existing = self
            .graph
            .find_buffer_range_storage(buffer, offset, bytes_per_row);
        let (backing, newly_created) = match existing {
            Some(backing) => (backing, false),
            None => {
                let backing = self.graph.create_storage_with_regions(
                    StorageBacking::BufferRange {
                        buffer,
                        offset,
                        bytes_per_row,
                    },
                    [BackingRegion::Whole],
                )?;
                let authority = self.graph.storage(backing).unwrap().content.clone();
                if let Err(error) = self.native.register_backing(backing, authority) {
                    self.graph
                        .retire_storage(backing)
                        .expect("a just-created buffer range has no owners");
                    return Err(error.into());
                }
                (backing, true)
            }
        };
        let created = self.graph.create_resource_with_descriptor(
            task,
            object,
            reims_vgpu_protocol::ObjectKind::TextureView,
            Some(Arc::new(ResourceDescriptor::BufferTexture(*descriptor))),
            Some(backing),
            [buffer],
        );
        match created {
            Ok(resource) => Ok(ResourceLifecycleEffect::ResourceCreated(resource)),
            Err(error) => {
                if newly_created {
                    self.native
                        .validate_begin_retirement(backing)
                        .expect("an unused just-created buffer range can retire");
                    self.graph
                        .retire_storage(backing)
                        .expect("a failed resource creation left no backing owner");
                    self.native
                        .begin_retirement(backing)
                        .expect("retirement was validated before ownership moved");
                }
                Err(error.into())
            }
        }
    }

    fn create_heap_texture(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        descriptor: Arc<reims_vgpu_protocol::HeapTextureDescriptor>,
        heap: ResourceId<HeapObject>,
        size: ByteLength,
        alignment: ByteLength,
    ) -> Result<ResourceLifecycleEffect<T>, ResourceLifecycleError> {
        if descriptor.object.get() != object.get() {
            return Err(ResourceLifecycleError::HeapTextureRelationMismatch);
        }
        if size.get() == 0 || !alignment.get().is_power_of_two() {
            return Err(ResourceLifecycleError::InvalidHeapTextureRequirements);
        }
        if descriptor.use_offset && !descriptor.offset.is_multiple_of(alignment.get()) {
            return Err(ResourceLifecycleError::HeapTextureOffsetMisaligned);
        }

        let resource = self.graph.create_resource_with_descriptor(
            task,
            object,
            ObjectKind::TextureView,
            Some(Arc::new(ResourceDescriptor::HeapTexture(*descriptor))),
            None,
            [],
        )?;
        let explicit = descriptor
            .use_offset
            .then_some((ByteOffset::new(descriptor.offset), size));
        if let Err(error) = self.graph.link_heap_texture(resource, heap, explicit) {
            self.graph
                .release_resource(resource)
                .expect("a just-created unlinked heap texture can always retire");
            return Err(error.into());
        }
        let backing = self
            .graph
            .resource(resource)
            .and_then(|node| node.storage)
            .expect("a linked heap texture owns its backing");
        if self.native.validity(backing).is_none() {
            let authority = self.graph.storage(backing).unwrap().content.clone();
            if let Err(error) = self.native.register_backing(backing, authority) {
                self.graph
                    .release_resource(resource)
                    .expect("a just-created heap texture can roll back");
                let retired = self.graph.take_automatically_retired_storage();
                debug_assert_eq!(retired.len(), 1);
                return Err(error.into());
            }
        }
        Ok(ResourceLifecycleEffect::ResourceCreated(resource))
    }

    fn create_mapper_texture(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        descriptor: Arc<reims_vgpu_protocol::MapperIOSurfaceTextureView>,
    ) -> Result<ResourceLifecycleEffect<T>, ResourceLifecycleError> {
        if descriptor.object.get() != object.get() || descriptor.mapper_surface.get() == 0 {
            return Err(ResourceLifecycleError::MapperTextureRelationMismatch);
        }
        let existing = self
            .graph
            .find_mapper_plane_storage(descriptor.mapper_surface, descriptor.plane);
        let (backing, newly_created) = match existing {
            Some(backing) => (backing, false),
            None => {
                let backing = self
                    .graph
                    .mapper_storage(descriptor.mapper_surface, descriptor.plane)?;
                let authority = self.graph.storage(backing).unwrap().content.clone();
                if let Err(error) = self.native.register_backing(backing, authority) {
                    self.graph
                        .retire_storage(backing)
                        .expect("a just-created mapper plane has no owners");
                    return Err(error.into());
                }
                (backing, true)
            }
        };
        let created = self.graph.create_resource_with_descriptor(
            task,
            object,
            ObjectKind::IOSurfaceTexture,
            Some(Arc::new(ResourceDescriptor::MapperIOSurfaceTextureView(
                *descriptor,
            ))),
            Some(backing),
            [],
        );
        match created {
            Ok(resource) => Ok(ResourceLifecycleEffect::ResourceCreated(resource)),
            Err(error) => {
                if newly_created {
                    self.native
                        .validate_begin_retirement(backing)
                        .expect("an unused just-created mapper plane can retire");
                    self.graph
                        .retire_storage(backing)
                        .expect("failed resource creation left no mapper-plane owner");
                    self.native
                        .begin_retirement(backing)
                        .expect("retirement was validated before ownership moved");
                }
                Err(error.into())
            }
        }
    }

    fn validate_validity(
        &self,
        transition: &ResolvedValidityTransition,
    ) -> Result<(), ValidityTransitionError> {
        if transition.targets.is_empty() {
            return Err(ValidityTransitionError::EmptyTargets);
        }
        if transition.ops.set_host_valid != 0 && transition.write.is_none() {
            return Err(ValidityTransitionError::MissingSubmission);
        }
        if transition.ops.set_host_valid == 0 && transition.write.is_some() {
            return Err(ValidityTransitionError::UnexpectedSubmission);
        }
        let mut backings = std::collections::BTreeSet::new();
        for target in transition.targets.iter() {
            if !backings.insert(target.backing) {
                return Err(ValidityTransitionError::DuplicateBacking(target.backing));
            }
            if target.regions.is_empty() {
                return Err(ValidityTransitionError::EmptyRegions(target.backing));
            }
            if transition.ops.set_host_valid != 0 && target.host_representation.is_none() {
                return Err(ValidityTransitionError::MissingHostRepresentation(
                    target.backing,
                ));
            }
            if transition.ops.set_host_valid == 0 && target.host_representation.is_some() {
                return Err(ValidityTransitionError::UnexpectedHostRepresentation(
                    target.backing,
                ));
            }
            if transition.ops.clear_host_valid == 0 && target.host_ingress_destination.is_some() {
                return Err(ValidityTransitionError::UnexpectedHostIngressDestination(
                    target.backing,
                ));
            }
            if let Some(destination) = target.host_ingress_destination {
                if destination != HOST_REPRESENTATION
                    || self
                        .native
                        .representation_route(target.backing, destination)
                        != Some(RepresentationRoute::HostStagingEndpoint)
                {
                    return Err(ValidityTransitionError::InvalidHostIngressDestination {
                        backing: target.backing,
                        destination,
                    });
                }
            }
            if transition.ops.clear_host_valid == 0 && target.guest_upload_destination.is_some() {
                return Err(ValidityTransitionError::UnexpectedGuestUploadDestination(
                    target.backing,
                ));
            }
            let guest_route = self
                .native
                .representation_route(target.backing, GUEST_REPRESENTATION);
            match target.guest_upload_destination {
                None if target.host_ingress_destination.is_some() => {
                    return Err(ValidityTransitionError::InvalidGuestUploadRoute {
                        backing: target.backing,
                        destination: None,
                        destination_route: None,
                        guest_route,
                    });
                }
                None => {}
                Some(destination) => {
                    let destination_route = self
                        .native
                        .representation_route(target.backing, destination);
                    // The same question the transition asked when it chose
                    // this destination, asked of the same function, so the
                    // two cannot disagree about which routes carry an upload.
                    // They were two hand-written matches over one enum, each
                    // with its own catch-all.
                    let route_is_valid = match destination_route
                        .map(crate::managed_resource::RepresentationRoute::guest_write_staging)
                    {
                        Some(crate::GuestWriteStaging::Transfer) => {
                            target.host_ingress_destination.is_none()
                                && guest_route == Some(RepresentationRoute::DirectGuestAlias)
                        }
                        Some(crate::GuestWriteStaging::StageThenTransfer) => {
                            target.host_ingress_destination == Some(HOST_REPRESENTATION)
                        }
                        Some(
                            crate::GuestWriteStaging::AlreadyHeld
                            | crate::GuestWriteStaging::NoUploadRoute,
                        )
                        | None => false,
                    };
                    if !route_is_valid {
                        return Err(ValidityTransitionError::InvalidGuestUploadRoute {
                            backing: target.backing,
                            destination: Some(destination),
                            destination_route,
                            guest_route,
                        });
                    }
                }
            }
            let reservation_passes = usize::from(transition.ops.clear_host_valid != 0)
                + usize::from(transition.ops.set_host_valid != 0);
            let reservations = target
                .regions
                .len()
                .checked_mul(reservation_passes)
                .ok_or(ValidityTransitionError::TooManyRegions(target.backing))?;
            let source_is_used =
                transition.ops.clear_guest_valid != 0 && transition.ops.clear_host_valid == 0;
            if source_is_used
                && !matches!(
                    target.guest_visibility_destination,
                    GUEST_REPRESENTATION | HOST_REPRESENTATION
                )
            {
                return Err(ValidityTransitionError::InvalidGuestVisibilityDestination {
                    backing: target.backing,
                    destination: target.guest_visibility_destination,
                });
            }
            if !source_is_used && target.guest_visibility_source.is_some() {
                return Err(ValidityTransitionError::UnexpectedGuestVisibilitySource(
                    target.backing,
                ));
            }
            self.native
                .validate_reservations(target.backing, reservations)
                .map_err(|reason| ValidityTransitionError::Backing {
                    backing: target.backing,
                    reason,
                })?;
            if transition.ops.clear_host_valid != 0 {
                self.native
                    .validate_guest_write(target.backing)
                    .map_err(|reason| ValidityTransitionError::Backing {
                        backing: target.backing,
                        reason,
                    })?;
            }
            if let (Some(write), Some(representation)) =
                (transition.write, target.host_representation)
            {
                self.native
                    .validate_plan_gpu_write(
                        target.backing,
                        write,
                        representation,
                        target.regions.len(),
                    )
                    .map_err(|reason| ValidityTransitionError::Backing {
                        backing: target.backing,
                        reason,
                    })?;
            }
            if transition.ops.clear_guest_valid != 0 && transition.ops.clear_host_valid == 0 {
                let snapshot = self
                    .native
                    .snapshot_content(target.backing, &target.regions)
                    .map_err(|reason| ValidityTransitionError::Backing {
                        backing: target.backing,
                        reason,
                    })?;
                let guest_current = self
                    .native
                    .representation_matches(target.backing, GUEST_REPRESENTATION, &snapshot)
                    .map_err(|reason| ValidityTransitionError::Backing {
                        backing: target.backing,
                        reason,
                    })?;
                if guest_current {
                    if target.guest_visibility_source.is_some() {
                        return Err(ValidityTransitionError::UnexpectedGuestVisibilitySource(
                            target.backing,
                        ));
                    }
                } else {
                    let source = target.guest_visibility_source.ok_or(
                        ValidityTransitionError::MissingGuestVisibilitySource(target.backing),
                    )?;
                    self.native
                        .validate_plan_transfer_demands(
                            target.backing,
                            source,
                            target.guest_visibility_destination,
                            &snapshot,
                        )
                        .map_err(|reason| ValidityTransitionError::Backing {
                            backing: target.backing,
                            reason,
                        })?;
                }
            }
        }
        Ok(())
    }

    fn apply_validity(
        &mut self,
        transition: ResolvedValidityTransition,
    ) -> Result<ResourceLifecycleEffect<T>, ResourceLifecycleError> {
        self.validate_validity(&transition)
            .map_err(ResourceLifecycleError::Validity)?;
        Ok(self.apply_prevalidated_validity(transition))
    }

    fn apply_validity_batch(
        &mut self,
        transitions: Box<[ResolvedValidityTransition]>,
    ) -> Result<ResourceLifecycleEffect<T>, ResourceLifecycleError> {
        if transitions.is_empty() {
            return Err(ResourceLifecycleError::EmptyValidityBatch);
        }
        let mut backings = std::collections::BTreeSet::new();
        for transition in transitions.iter() {
            self.validate_validity(transition)
                .map_err(ResourceLifecycleError::Validity)?;
            for target in transition.targets.iter() {
                if !backings.insert(target.backing) {
                    return Err(ResourceLifecycleError::DuplicateValidityBatchBacking(
                        target.backing,
                    ));
                }
            }
        }
        Ok(ResourceLifecycleEffect::ValidityBatchApplied(
            transitions
                .into_vec()
                .into_iter()
                .map(|transition| self.apply_prevalidated_validity(transition))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ))
    }

    fn apply_prevalidated_validity(
        &mut self,
        transition: ResolvedValidityTransition,
    ) -> ResourceLifecycleEffect<T> {
        let mut guest_writes = Vec::new();
        let mut host_ingresses = Vec::new();
        let mut deferred_host_ingress_transfers = Vec::new();
        let mut gpu_reservations = Vec::new();
        let mut gpu_completions = Vec::new();
        let mut transfers = Vec::new();
        let mut host_landings = Vec::new();
        let mut states = Vec::new();
        for target in transition.targets {
            if transition.ops.clear_host_valid != 0 {
                for region in target.regions.iter().copied() {
                    let version = self
                        .native
                        .guest_write(target.backing, transition.guest_write, region)
                        .unwrap_or_else(|_| unreachable!("validity guest write was prevalidated"));
                    guest_writes.push((target.backing, version));
                    if let Some(destination) = target.guest_upload_destination {
                        if target.host_ingress_destination.is_some() {
                            let ingress = self
                                .native
                                .plan_host_ingress(target.backing, version)
                                .unwrap_or_else(|_| {
                                    unreachable!(
                                        "host ingress follows its prevalidated guest write"
                                    )
                                });
                            host_ingresses.push(ingress);
                            deferred_host_ingress_transfers.push(HostIngressTransfer {
                                ingress,
                                destination,
                            });
                        } else {
                            transfers.extend(
                                self.native
                                    .plan_transfer_demands(
                                        target.backing,
                                        GUEST_REPRESENTATION,
                                        destination,
                                        &[version],
                                    )
                                    .unwrap_or_else(|_| {
                                        unreachable!(
                                            "imported upload follows its prevalidated guest write"
                                        )
                                    }),
                            );
                        }
                    }
                }
            }
            if let (Some(write), Some(representation)) =
                (transition.write, target.host_representation)
            {
                let regions = self
                    .native
                    .plan_gpu_write(
                        target.backing,
                        write,
                        representation,
                        target.regions.clone(),
                    )
                    .unwrap_or_else(|_| unreachable!("validity GPU write was prevalidated"));
                gpu_reservations.push(GpuWriteReservation {
                    backing: target.backing,
                    write,
                    representation,
                    regions,
                });
                gpu_completions.push(ResolvedResourceCompletion::ValidityHostWrite {
                    backing: target.backing,
                    write,
                    representation,
                });
            }
            if transition.ops.clear_guest_valid != 0 && transition.ops.clear_host_valid == 0 {
                let snapshot = self
                    .native
                    .snapshot_content(target.backing, &target.regions)
                    .expect("validity snapshot was prevalidated");
                if !self
                    .native
                    .representation_matches(target.backing, GUEST_REPRESENTATION, &snapshot)
                    .expect("validity guest representation was prevalidated")
                {
                    let source = target
                        .guest_visibility_source
                        .expect("validity transfer source was prevalidated");
                    let planned = self
                        .native
                        .plan_transfer_demands(
                            target.backing,
                            source,
                            target.guest_visibility_destination,
                            &snapshot,
                        )
                        .unwrap_or_else(|_| unreachable!("validity transfer was prevalidated"));
                    if target.guest_visibility_destination == HOST_REPRESENTATION {
                        host_landings.extend(planned.iter().copied().map(|transfer| {
                            self.native.plan_host_landing(transfer).unwrap_or_else(|_| {
                                unreachable!("host landing follows its prevalidated transfer")
                            })
                        }));
                    }
                    transfers.extend(planned);
                }
            }
            let immediate_ops = ResourceValidityOps {
                set_host_valid: 0,
                ..transition.ops
            };
            let state = self
                .native
                .apply_validity(target.backing, immediate_ops)
                .unwrap_or_else(|_| unreachable!("validity state was prevalidated"));
            states.push((target.backing, state));
        }
        ResourceLifecycleEffect::ValidityApplied {
            guest_writes: guest_writes.into_boxed_slice(),
            host_ingresses: host_ingresses.into_boxed_slice(),
            deferred_host_ingress_transfers: deferred_host_ingress_transfers.into_boxed_slice(),
            gpu_reservations: gpu_reservations.into_boxed_slice(),
            gpu_completions: gpu_completions.into_boxed_slice(),
            transfers: transfers.into_boxed_slice(),
            host_landings: host_landings.into_boxed_slice(),
            states: states.into_boxed_slice(),
        }
    }

    fn retire_automatic_storage(
        &mut self,
    ) -> Result<Vec<RetiredBacking<T>>, ResourceLifecycleError> {
        let storage = self.graph.take_automatically_retired_storage();
        let mut retired = Vec::with_capacity(storage.len());
        for storage in storage {
            self.native.validate_begin_retirement(storage.id)?;
            let native = self
                .native
                .begin_retirement(storage.id)
                .unwrap_or_else(|_| {
                    unreachable!("automatic retirement was validated before ownership moved")
                });
            retired.push(RetiredBacking { storage, native });
        }
        Ok(retired)
    }

    pub fn create_representation(
        &mut self,
        backing: BackingId,
        route: RepresentationRoute,
        native: T,
    ) -> Result<RepresentationId, ManagedRepresentationFailure<T>> {
        self.native.create_representation(backing, route, native)
    }

    pub fn create_execution_representation(
        &mut self,
        backing: BackingId,
        route: RepresentationRoute,
        view: BackingView,
        native: T,
    ) -> Result<RepresentationId, ManagedRepresentationFailure<T>> {
        self.native
            .create_execution_representation(backing, route, view, native)
    }

    /// The representation serving one view of a backing's bytes, when it holds
    /// the content snapshot the read requires.
    pub fn view_representation_for_snapshot(
        &self,
        backing: BackingId,
        view: BackingView,
        snapshot: &[RegionVersion],
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.native
            .view_representation_for_snapshot(backing, view, snapshot)
    }

    /// The representation serving one view of a backing's bytes.
    pub fn view_representation(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.native.view_representation(backing, view)
    }

    pub fn execution_representation_for_snapshot(
        &self,
        backing: BackingId,
        view: BackingView,
        snapshot: &[RegionVersion],
    ) -> Result<(RepresentationId, &T), ManagedBackingError> {
        self.native
            .execution_representation_for_snapshot(backing, view, snapshot)
    }

    /// See [`crate::ManagedBackingOwner::any_designated_representation`].
    pub fn any_designated_representation(
        &self,
        backing: BackingId,
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.native.any_designated_representation(backing)
    }

    /// See
    /// [`crate::ManagedBackingOwner::designated_representation_for_snapshot`].
    pub fn designated_representation_for_snapshot(
        &self,
        backing: BackingId,
        snapshot: &[RegionVersion],
    ) -> Result<(RepresentationId, &T), ManagedBackingError> {
        self.native
            .designated_representation_for_snapshot(backing, snapshot)
    }

    /// See [`crate::ManagedBackingOwner::is_designated`].
    pub fn is_designated(&self, backing: BackingId, representation: RepresentationId) -> bool {
        self.native.is_designated(backing, representation)
    }

    /// See
    /// [`crate::ManagedBackingOwner::stale_designated_representations`].
    pub fn stale_designated_representations(
        &self,
        backing: BackingId,
        snapshot: &[RegionVersion],
    ) -> Result<Vec<(BackingView, RepresentationId)>, ManagedBackingError> {
        self.native
            .stale_designated_representations(backing, snapshot)
    }

    pub fn stale_read_representations(
        &self,
        backing: BackingId,
        views: &[BackingView],
        snapshot: &[RegionVersion],
    ) -> Result<Vec<(BackingView, RepresentationId)>, ManagedBackingError> {
        self.native
            .stale_read_representations(backing, views, snapshot)
    }

    /// See
    /// [`crate::ManagedBackingOwner::host_write_reaches_every_designated_view`].
    pub fn host_write_reaches_every_designated_view(
        &self,
        backing: BackingId,
    ) -> Result<bool, ManagedBackingError> {
        self.native
            .host_write_reaches_every_designated_view(backing)
    }

    /// See [`crate::ManagedBackingOwner::designated_views`].
    pub fn designated_views(
        &self,
        backing: BackingId,
    ) -> Result<Vec<(BackingView, RepresentationId)>, ManagedBackingError> {
        self.native.designated_views(backing)
    }

    pub fn execution_representation_id(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.native.execution_representation_id(backing, view)
    }

    /// See [`crate::ManagedBackingOwner::execution_representation_coverage`].
    pub fn execution_representation_coverage(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Option<(RepresentationId, Vec<crate::RegionVersion>)> {
        self.native.execution_representation_coverage(backing, view)
    }

    pub fn execution_representation(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Option<(RepresentationId, &T)> {
        self.native.execution_representation(backing, view)
    }

    pub fn representation(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<&T> {
        self.native.representation(backing, representation)
    }

    pub fn representation_mut(
        &mut self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<&mut T> {
        self.native.representation_mut(backing, representation)
    }

    pub fn accepted_representations(
        &self,
        transaction: TransactionId,
        backings: &[BackingId],
    ) -> Result<Box<[crate::AcceptedRepresentation]>, ResourceUseBatchError> {
        self.native.accepted_representations(transaction, backings)
    }

    pub fn representation_route(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<RepresentationRoute> {
        self.native.representation_route(backing, representation)
    }

    pub fn accept_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
        representations: impl IntoIterator<Item = RepresentationId>,
    ) -> Result<(), ManagedBackingError> {
        self.native
            .accept_use(backing, transaction, representations)
    }

    pub fn accept_uses(
        &mut self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<(), ResourceUseBatchError> {
        self.native.accept_uses(transaction, uses)
    }

    pub fn validate_accept_uses(
        &self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<(), ResourceUseBatchError> {
        self.native.validate_accept_uses(transaction, uses)
    }

    pub fn submit_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.native.submit_use(backing, transaction, point)
    }

    /// Validate an exact multi-backing acceptance transition before native
    /// enqueue. With exclusive ownership held, every subsequent commit is
    /// infallible unless the validated set is changed first.
    pub fn validate_submit_uses(
        &self,
        backings: &[BackingId],
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<(), ResourceUseBatchError> {
        validate_unique_backings(backings)?;
        for &backing in backings {
            self.native
                .validate_submit_use(backing, transaction, point)
                .map_err(|reason| ResourceUseBatchError::Backing { backing, reason })?;
        }
        Ok(())
    }

    pub fn submit_uses(
        &mut self,
        backings: &[BackingId],
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<Vec<(BackingId, ManagedBackingProgress<T>)>, ResourceUseBatchError> {
        self.validate_submit_uses(backings, transaction, point)?;
        Ok(backings
            .iter()
            .copied()
            .map(|backing| {
                let progress = self
                    .native
                    .submit_use(backing, transaction, point)
                    .unwrap_or_else(|_| {
                        unreachable!("the complete backing-use batch was prevalidated")
                    });
                (backing, progress)
            })
            .collect())
    }

    pub fn cancel_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.native.cancel_use(backing, transaction)
    }

    pub fn validate_cancel_uses(
        &self,
        backings: &[BackingId],
        transaction: TransactionId,
    ) -> Result<(), ResourceUseBatchError> {
        validate_unique_backings(backings)?;
        for &backing in backings {
            self.native
                .validate_cancel_use(backing, transaction)
                .map_err(|reason| ResourceUseBatchError::Backing { backing, reason })?;
        }
        Ok(())
    }

    pub fn cancel_uses(
        &mut self,
        backings: &[BackingId],
        transaction: TransactionId,
    ) -> Result<Vec<(BackingId, ManagedBackingProgress<T>)>, ResourceUseBatchError> {
        self.validate_cancel_uses(backings, transaction)?;
        Ok(backings
            .iter()
            .copied()
            .map(|backing| {
                let progress = self
                    .native
                    .cancel_use(backing, transaction)
                    .unwrap_or_else(|_| {
                        unreachable!("the complete backing-use batch was prevalidated")
                    });
                (backing, progress)
            })
            .collect())
    }

    pub fn validate_cancel_representation_uses(
        &self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<(), ResourceUseBatchError> {
        self.native
            .validate_cancel_representation_uses(transaction, uses)
    }

    pub fn cancel_representation_uses(
        &mut self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<Vec<(BackingId, ManagedBackingProgress<T>)>, ResourceUseBatchError> {
        self.native.cancel_representation_uses(transaction, uses)
    }

    pub fn plan_gpu_write(
        &mut self,
        backing: BackingId,
        write: impl Into<crate::GpuWriteId>,
        representation: RepresentationId,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        self.native
            .plan_gpu_write(backing, write, representation, regions)
    }

    pub fn validate_plan_gpu_writes(
        &self,
        write: impl Into<crate::GpuWriteId> + Copy,
        requests: &[crate::GpuWriteRequest],
    ) -> Result<(), crate::GpuWriteBatchError> {
        self.native.validate_plan_gpu_writes(write, requests)
    }

    pub fn complete_gpu_write(
        &mut self,
        backing: BackingId,
        write: impl Into<crate::GpuWriteId> + Copy,
        representation: RepresentationId,
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        self.native
            .complete_gpu_write(backing, write, representation)
    }

    pub fn cancel_gpu_write(
        &mut self,
        backing: BackingId,
        write: impl Into<crate::GpuWriteId>,
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        self.native.cancel_gpu_write(backing, write)
    }

    pub fn plan_gpu_writes(
        &mut self,
        write: impl Into<crate::GpuWriteId> + Copy,
        requests: impl Into<Box<[GpuWriteRequest]>>,
    ) -> Result<Box<[GpuWriteReservation]>, GpuWriteBatchError> {
        self.native.plan_gpu_writes(write, requests)
    }

    pub fn cancel_gpu_writes(
        &mut self,
        reservations: &[GpuWriteReservation],
    ) -> Result<(), GpuWriteBatchError> {
        self.native.cancel_gpu_writes(reservations)
    }

    pub fn validate_cancel_gpu_writes(
        &self,
        reservations: &[GpuWriteReservation],
    ) -> Result<(), GpuWriteBatchError> {
        self.native.validate_cancel_gpu_writes(reservations)
    }

    pub fn snapshot_content(
        &self,
        backing: BackingId,
        regions: &[BackingRegion],
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        self.native.snapshot_content(backing, regions)
    }

    pub fn current_native_regions_for_version(
        &self,
        backing: BackingId,
        excluded: &[reims_vgpu_protocol::RepresentationId],
        required: crate::RegionVersion,
    ) -> Result<crate::RepresentationRegionCoverage, ManagedBackingError> {
        self.native
            .current_native_regions_for_version(backing, excluded, required)
    }

    pub fn current_regions_in_representation(
        &self,
        backing: BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
        required: crate::RegionVersion,
    ) -> Result<Box<[crate::BackingRegion]>, ManagedBackingError> {
        self.native
            .current_regions_in_representation(backing, representation, required)
    }

    pub fn transferable_regions_in_representation(
        &self,
        backing: BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
        required: crate::RegionVersion,
    ) -> Result<Box<[crate::BackingRegion]>, ManagedBackingError> {
        self.native
            .transferable_regions_in_representation(backing, representation, required)
    }

    pub fn pending_gpu_writes_overlapping(
        &self,
        backing: BackingId,
        representation: RepresentationId,
        regions: &[BackingRegion],
    ) -> Result<Box<[crate::GpuWriteId]>, ManagedBackingError> {
        self.native
            .pending_gpu_writes_overlapping(backing, representation, regions)
    }

    pub fn plan_host_ingress(
        &mut self,
        backing: BackingId,
        write: RegionVersion,
    ) -> Result<HostIngressKey, ManagedBackingError> {
        self.native.plan_host_ingress(backing, write)
    }

    pub fn complete_transfer(&mut self, key: TransferKey) -> Result<(), ManagedBackingError> {
        self.native.complete_transfer(key)
    }

    pub fn cancel_transfers(
        &mut self,
        transfers: &[TransferKey],
    ) -> Result<(), TransferBatchError> {
        self.native.cancel_transfers(transfers)
    }

    pub fn validate_cancel_transfers(
        &self,
        transfers: &[TransferKey],
    ) -> Result<(), TransferBatchError> {
        self.native.validate_cancel_transfers(transfers)
    }

    pub fn plan_transfers(
        &mut self,
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ManagedBackingError> {
        self.native
            .plan_transfers(backing, source, destination, snapshot)
    }

    pub fn plan_transfer_demands(
        &mut self,
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ManagedBackingError> {
        self.native
            .plan_transfer_demands(backing, source, destination, snapshot)
    }

    pub fn validate_resource_completions(
        &self,
        completions: &[ResolvedResourceCompletion],
    ) -> Result<(), ResourceCompletionBatchError> {
        let mut unique = std::collections::BTreeSet::new();
        for &completion in completions {
            if !unique.insert(completion) {
                return Err(ResourceCompletionBatchError::Duplicate(completion));
            }
            let result = match completion {
                ResolvedResourceCompletion::GpuWrite {
                    backing,
                    write,
                    representation,
                } => self
                    .native
                    .validate_complete_gpu_write(backing, write, representation),
                ResolvedResourceCompletion::ValidityHostWrite {
                    backing,
                    write,
                    representation,
                } => self
                    .native
                    .validate_complete_gpu_write(backing, write, representation),
                ResolvedResourceCompletion::Transfer(key) => {
                    self.native.validate_complete_transfer(key)
                }
                ResolvedResourceCompletion::Discard { backing, .. } => {
                    self.native.validate_discard(backing)
                }
            };
            result.map_err(|reason| ResourceCompletionBatchError::Completion {
                completion,
                reason,
            })?;
        }
        Ok(())
    }

    pub fn complete_resources(
        &mut self,
        completions: &[ResolvedResourceCompletion],
    ) -> Result<Vec<ResourceCompletionEffect>, ResourceCompletionBatchError> {
        self.validate_resource_completions(completions)?;
        Ok(completions
            .iter()
            .copied()
            .map(|completion| match completion {
                ResolvedResourceCompletion::GpuWrite {
                    backing,
                    write,
                    representation,
                } => ResourceCompletionEffect::GpuWrite {
                    backing,
                    submission: write.submission(),
                    regions: self
                        .native
                        .complete_gpu_write(backing, write, representation)
                        .unwrap_or_else(|_| {
                            unreachable!("the complete resource batch was prevalidated")
                        }),
                },
                ResolvedResourceCompletion::ValidityHostWrite {
                    backing,
                    write,
                    representation,
                } => {
                    let regions = self
                        .native
                        .complete_gpu_write(backing, write, representation)
                        .unwrap_or_else(|_| {
                            unreachable!("the complete validity batch was prevalidated")
                        });
                    let state = self
                        .native
                        .complete_validity(
                            backing,
                            ResourceValidityOps {
                                clear_host_valid: 0,
                                set_host_valid: 1,
                                clear_guest_valid: 0,
                                set_guest_valid: 0,
                            },
                        )
                        .unwrap_or_else(|_| {
                            unreachable!("the complete validity batch was prevalidated")
                        });
                    ResourceCompletionEffect::ValidityHostWrite {
                        backing,
                        submission: write.submission(),
                        regions,
                        state,
                    }
                }
                ResolvedResourceCompletion::Transfer(key) => {
                    self.native.complete_transfer(key).unwrap_or_else(|_| {
                        unreachable!("the complete resource batch was prevalidated")
                    });
                    ResourceCompletionEffect::Transfer(key)
                }
                ResolvedResourceCompletion::Discard { backing, region } => {
                    self.native.discard(backing, region).unwrap_or_else(|_| {
                        unreachable!("the complete resource batch was prevalidated")
                    });
                    ResourceCompletionEffect::Discard { backing, region }
                }
            })
            .collect())
    }

    pub fn validate_complete_host_landings(
        &self,
        landings: &[HostLandingKey],
    ) -> Result<(), HostLandingBatchError> {
        let mut unique = std::collections::BTreeSet::new();
        for &landing in landings {
            if !unique.insert(landing) {
                return Err(HostLandingBatchError::Duplicate(landing));
            }
            self.native
                .validate_complete_host_landing(landing)
                .map_err(|reason| HostLandingBatchError::Landing { landing, reason })?;
        }
        Ok(())
    }

    pub fn validate_host_landings_after_resource_completions(
        &self,
        completions: &[ResolvedResourceCompletion],
        landings: &[HostLandingKey],
    ) -> Result<(), HostLandingBatchError> {
        let mut unique = std::collections::BTreeSet::new();
        for &landing in landings {
            if !unique.insert(landing) {
                return Err(HostLandingBatchError::Duplicate(landing));
            }
            let staged = completions.iter().any(|completion| {
                matches!(
                    completion,
                    ResolvedResourceCompletion::Transfer(transfer)
                        if transfer.backing == landing.backing
                            && transfer.region == landing.region
                            && transfer.version == landing.version
                            && transfer.destination == HOST_REPRESENTATION
                )
            });
            if !staged {
                return Err(HostLandingBatchError::StagedTransferAbsent(landing));
            }
            self.native
                .validate_host_landing_pending(landing)
                .map_err(|reason| HostLandingBatchError::Landing { landing, reason })?;
        }
        Ok(())
    }

    pub fn complete_host_landings(
        &mut self,
        landings: &[HostLandingKey],
    ) -> Result<(), HostLandingBatchError> {
        self.validate_complete_host_landings(landings)?;
        for &landing in landings {
            self.native
                .complete_host_landing(landing)
                .unwrap_or_else(|_| unreachable!("host landing batch was prevalidated"));
        }
        Ok(())
    }

    pub fn validate_cancel_host_landings(
        &self,
        landings: &[HostLandingKey],
    ) -> Result<(), HostLandingBatchError> {
        let mut demands = std::collections::BTreeMap::new();
        for &landing in landings {
            let count = demands.entry(landing).or_insert(0usize);
            *count = count.checked_add(1).ok_or(HostLandingBatchError::Landing {
                landing,
                reason: ManagedBackingError::Content(
                    crate::ContentAuthorityError::TransferDemandCountExhausted,
                ),
            })?;
        }
        for (landing, count) in demands {
            self.native
                .validate_cancel_host_landing_demands(landing, count)
                .map_err(|reason| HostLandingBatchError::Landing { landing, reason })?;
        }
        Ok(())
    }

    pub fn cancel_host_landings(
        &mut self,
        landings: &[HostLandingKey],
    ) -> Result<(), HostLandingBatchError> {
        self.validate_cancel_host_landings(landings)?;
        for &landing in landings {
            self.native
                .cancel_host_landing(landing)
                .unwrap_or_else(|_| unreachable!("host landing cancellation was prevalidated"));
        }
        Ok(())
    }

    pub fn validate_complete_host_ingresses(
        &self,
        ingresses: &[HostIngressKey],
    ) -> Result<(), HostIngressBatchError> {
        let mut unique = std::collections::BTreeSet::new();
        for &ingress in ingresses {
            if !unique.insert(ingress) {
                return Err(HostIngressBatchError::Duplicate(ingress));
            }
            self.native
                .validate_complete_host_ingress(ingress)
                .map_err(|reason| HostIngressBatchError::Ingress { ingress, reason })?;
        }
        Ok(())
    }

    pub fn complete_host_ingresses(
        &mut self,
        ingresses: &[HostIngressKey],
    ) -> Result<(), HostIngressBatchError> {
        self.validate_complete_host_ingresses(ingresses)?;
        for &ingress in ingresses {
            self.native
                .complete_host_ingress(ingress)
                .unwrap_or_else(|_| unreachable!("host ingress batch was prevalidated"));
        }
        Ok(())
    }

    pub fn complete_host_ingress_transfers(
        &mut self,
        transfers: &[HostIngressTransfer],
    ) -> Result<Box<[TransferKey]>, HostIngressBatchError> {
        let mut unique = std::collections::BTreeSet::new();
        for &transfer in transfers {
            if !unique.insert(transfer) {
                return Err(HostIngressBatchError::DuplicateTransfer(transfer));
            }
            self.native
                .validate_host_ingress_transfer(transfer)
                .map_err(|reason| HostIngressBatchError::Transfer { transfer, reason })?;
        }
        let mut planned = Vec::new();
        for &transfer in transfers {
            self.native
                .complete_host_ingress(transfer.ingress)
                .unwrap_or_else(|_| unreachable!("host ingress transfer was prevalidated"));
            planned.extend(
                self.native
                    .plan_transfer_demands(
                        transfer.ingress.backing,
                        HOST_REPRESENTATION,
                        transfer.destination,
                        &[RegionVersion {
                            region: transfer.ingress.region,
                            version: transfer.ingress.version,
                        }],
                    )
                    .unwrap_or_else(|_| {
                        unreachable!("completed ingress transfer was prevalidated")
                    }),
            );
        }
        Ok(planned.into_boxed_slice())
    }

    pub fn validate_cancel_host_ingresses(
        &self,
        ingresses: &[HostIngressKey],
    ) -> Result<(), HostIngressBatchError> {
        let mut demands = std::collections::BTreeMap::new();
        for &ingress in ingresses {
            let count = demands.entry(ingress).or_insert(0usize);
            *count = count.checked_add(1).ok_or(HostIngressBatchError::Ingress {
                ingress,
                reason: ManagedBackingError::Content(
                    crate::ContentAuthorityError::TransferDemandCountExhausted,
                ),
            })?;
        }
        for (ingress, count) in demands {
            self.native
                .validate_cancel_host_ingress_demands(ingress, count)
                .map_err(|reason| HostIngressBatchError::Ingress { ingress, reason })?;
        }
        Ok(())
    }

    pub fn cancel_host_ingresses(
        &mut self,
        ingresses: &[HostIngressKey],
    ) -> Result<(), HostIngressBatchError> {
        self.validate_cancel_host_ingresses(ingresses)?;
        for &ingress in ingresses {
            self.native
                .cancel_host_ingress(ingress)
                .unwrap_or_else(|_| unreachable!("host ingress cancellation was prevalidated"));
        }
        Ok(())
    }

    pub fn advance_native_retirement(
        &mut self,
        queue: reims_vgpu_protocol::QueueOwnerId,
        completed: reims_vgpu_protocol::QueueTimelineValue,
    ) -> Result<Vec<T>, ManagedBackingError> {
        self.native.advance(queue, completed)
    }

    pub fn validate_native_retirement_advance(
        &self,
        queue: reims_vgpu_protocol::QueueOwnerId,
        completed: reims_vgpu_protocol::QueueTimelineValue,
    ) -> Result<(), ManagedBackingError> {
        self.native.validate_advance(queue, completed)
    }

    pub fn abandon(self) -> Vec<T> {
        self.native.abandon()
    }
}

fn validate_unique_backings(backings: &[BackingId]) -> Result<(), ResourceUseBatchError> {
    let mut unique = std::collections::BTreeSet::new();
    for &backing in backings {
        if !unique.insert(backing) {
            return Err(ResourceUseBatchError::DuplicateBacking(backing));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearRange, GUEST_REPRESENTATION};
    use reims_vgpu_protocol::{ByteLength, GuestVirtualAddress, QueueOwnerId, QueueTimelineValue};

    fn create_backing<T>(owner: &mut ResourceLifecycleOwner<T>) -> BackingId {
        let ResourceLifecycleEffect::BackingCreated(backing) = owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::TaskAddress {
                    task: TaskId::new(1),
                    address: GuestVirtualAddress::new(0x4000),
                    length: ByteLength::new(0x1000),
                },
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 0x1000).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        backing
    }

    #[test]
    fn buffer_texture_relation_refuses_a_non_buffer_before_backing_creation() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = create_backing(&mut owner);
        let ResourceLifecycleEffect::ResourceCreated(parent) = owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(7),
                kind: ObjectKind::Texture,
                descriptor: Arc::new(ResourceDescriptor::Texture(Default::default())),
                storage: Some(backing),
                parents: Box::new([]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let descriptor = reims_vgpu_protocol::BufferTextureDescriptor {
            new_texture_ref: 8,
            buffer_ref: 7,
            offset: 64,
            bytes_per_row: 256,
            desc: reims_vgpu_protocol::TextureDeclaration {
                texture_type: reims_vgpu_protocol::TextureType::Buffer,
                framebuffer_only: false,
                is_drawable: false,
                write_swizzle_enabled: None,
                allow_gpu_optimized_contents: false,
                usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                pixel_format: 80,
                width: 16,
                height: 1,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                resource_options: 0,
                protection_options: 0,
                swizzle: None,
            },
        };
        assert!(matches!(
            owner.apply(ResolvedResourceLifecycle::CreateBufferTexture {
                task: TaskId::new(1),
                object: ObjectTableRef::new(8),
                descriptor: Arc::new(descriptor),
                buffer: parent,
            }),
            Err(ResourceLifecycleError::BufferTextureRelationMismatch)
        ));
        assert_eq!(
            owner
                .graph()
                .resolve(TaskId::new(1), ObjectTableRef::new(8)),
            None
        );
        assert_eq!(
            owner.graph().find_buffer_range_storage(
                parent,
                reims_vgpu_protocol::ByteOffset::new(64),
                ByteLength::new(256),
            ),
            None
        );
    }

    #[test]
    fn heap_texture_creation_is_atomic_and_exact_placements_share_authority() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let heap = ResourceId::<HeapObject>::new(4, 2);
        let descriptor = |object| {
            Arc::new(reims_vgpu_protocol::HeapTextureDescriptor {
                object: reims_vgpu_protocol::SerializerRef::new(object),
                heap: reims_vgpu_protocol::SerializerRef::new(9),
                declaration: reims_vgpu_protocol::TextureDeclaration {
                    texture_type: reims_vgpu_protocol::TextureType::D2,
                    framebuffer_only: false,
                    is_drawable: false,
                    write_swizzle_enabled: None,
                    allow_gpu_optimized_contents: false,
                    usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                    pixel_format: 80,
                    width: 16,
                    height: 16,
                    depth: 1,
                    mipmap_level_count: 1,
                    sample_count: 1,
                    array_length: 1,
                    resource_options: 0,
                    protection_options: 0,
                    swizzle: None,
                },
                use_offset: true,
                offset: 0x400,
            })
        };
        let create = |object| ResolvedResourceLifecycle::CreateHeapTexture {
            task: TaskId::new(1),
            object: ObjectTableRef::new(object),
            descriptor: descriptor(object),
            heap,
            size: ByteLength::new(0x800),
            alignment: ByteLength::new(0x100),
        };
        let ResourceLifecycleEffect::ResourceCreated(first) = owner.apply(create(7)).unwrap()
        else {
            unreachable!()
        };
        let ResourceLifecycleEffect::ResourceCreated(second) = owner.apply(create(8)).unwrap()
        else {
            unreachable!()
        };
        let first_backing = owner.graph().resource(first).unwrap().storage.unwrap();
        let second_backing = owner.graph().resource(second).unwrap().storage.unwrap();
        assert_eq!(first_backing, second_backing);
        assert!(owner
            .graph()
            .resource(first)
            .unwrap()
            .content
            .same_authority(&owner.graph().resource(second).unwrap().content));

        let mut misaligned = *descriptor(10);
        misaligned.offset = 0x480;
        assert!(matches!(
            owner.apply(ResolvedResourceLifecycle::CreateHeapTexture {
                task: TaskId::new(1),
                object: ObjectTableRef::new(10),
                descriptor: Arc::new(misaligned),
                heap,
                size: ByteLength::new(0x800),
                alignment: ByteLength::new(0x100),
            }),
            Err(ResourceLifecycleError::HeapTextureOffsetMisaligned)
        ));
        assert!(owner
            .graph()
            .resolve(TaskId::new(1), ObjectTableRef::new(10))
            .is_none());
    }

    #[test]
    fn semantic_release_and_native_retirement_share_one_backing_identity() {
        let epoch = VulkanDeviceEpochId::new(3);
        let mut owner = ResourceLifecycleOwner::new(epoch);
        let backing = create_backing(&mut owner);
        let ResourceLifecycleEffect::ResourceCreated(resource) = owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(7),
                kind: ObjectKind::Buffer,
                descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                storage: Some(backing),
                parents: Box::new([]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert!(matches!(
            owner
                .graph()
                .resource(resource)
                .unwrap()
                .descriptor
                .as_deref(),
            Some(ResourceDescriptor::Buffer(_))
        ));
        let representation = owner
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, "buffer")
            .unwrap();
        assert_eq!(representation, GUEST_REPRESENTATION);
        owner
            .accept_use(backing, TransactionId::new(9), [representation])
            .unwrap();
        owner
            .apply(ResolvedResourceLifecycle::ReleaseResource { resource })
            .unwrap();
        let ResourceLifecycleEffect::BackingRetired(retired) = owner
            .apply(ResolvedResourceLifecycle::RetireBacking { backing })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(retired.storage.id, backing);
        assert!(matches!(
            retired.native,
            ManagedBackingProgress::WaitingForAcceptedUses
        ));
        assert!(matches!(
            owner
                .submit_use(
                    backing,
                    TransactionId::new(9),
                    QueueTimelinePoint {
                        epoch,
                        queue: QueueOwnerId::new(1),
                        value: QueueTimelineValue::new(4),
                    },
                )
                .unwrap(),
            ManagedBackingProgress::RetirementStarted { deferred: 1, .. }
        ));
        assert_eq!(
            owner
                .advance_native_retirement(QueueOwnerId::new(1), QueueTimelineValue::new(4))
                .unwrap(),
            vec!["buffer"]
        );
    }

    #[test]
    fn mapping_and_resource_lifetimes_both_gate_explicit_backing_retirement() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = {
            let ResourceLifecycleEffect::BackingCreated(backing) = owner
                .apply(ResolvedResourceLifecycle::CreateBacking {
                    backing: StorageBacking::TaskAddress {
                        task: TaskId::new(1),
                        address: GuestVirtualAddress::new(0x4000),
                        length: ByteLength::new(0x1000),
                    },
                    regions: Box::new([BackingRegion::Whole]),
                })
                .unwrap()
            else {
                unreachable!()
            };
            backing
        };
        owner
            .apply(ResolvedResourceLifecycle::CreateMapping(MappingNode {
                id: MappingId::new(2),
                task: TaskId::new(1),
                address: GuestVirtualAddress::new(0x4000),
                length: ByteLength::new(0x1000),
                storage: Some(backing),
                committed: true,
            }))
            .unwrap();
        assert!(matches!(
            owner.apply(ResolvedResourceLifecycle::RetireBacking { backing }),
            Err(ResourceLifecycleError::Graph(GraphError::StorageInUse))
        ));
        owner
            .apply(ResolvedResourceLifecycle::ReleaseMapping {
                mapping: MappingId::new(2),
            })
            .unwrap();
        assert!(matches!(
            owner
                .apply(ResolvedResourceLifecycle::RetireBacking { backing })
                .unwrap(),
            ResourceLifecycleEffect::BackingRetired(_)
        ));
    }

    #[test]
    fn a_task_release_takes_the_resources_the_task_still_owns() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let first = match owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(1),
                kind: ObjectKind::Buffer,
                descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                storage: None,
                parents: Box::new([]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::ResourceCreated(resource) => resource,
            _ => unreachable!(),
        };
        let second = match owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(2),
                kind: ObjectKind::Texture,
                descriptor: Arc::new(ResourceDescriptor::Texture(Default::default())),
                storage: None,
                parents: Box::new([]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::ResourceCreated(resource) => resource,
            _ => unreachable!(),
        };
        let ResourceLifecycleEffect::TaskReleased { resources, .. } = owner
            .apply(ResolvedResourceLifecycle::ReleaseTask {
                task: TaskId::new(1),
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            resources.as_ref(),
            [first, second],
            "a task release takes whatever the task still owns when it applies"
        );
        assert!(owner.graph().resources_for_task(TaskId::new(1)).is_empty());
    }

    #[test]
    fn physical_replacement_preserves_logical_backing_and_content_authority() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = create_backing(&mut owner);
        let resource = match owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(1),
                kind: ObjectKind::Buffer,
                descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                storage: Some(backing),
                parents: Box::new([]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::ResourceCreated(resource) => resource,
            _ => unreachable!(),
        };
        let before = owner.graph().resource(resource).unwrap().content.clone();
        assert!(matches!(
            owner
                .apply(ResolvedResourceLifecycle::ReplacePhysical { resource })
                .unwrap(),
            ResourceLifecycleEffect::PhysicalReplaced {
                backing: Some(found),
                generation,
                ..
            } if found == backing && generation == BackingGeneration::new(2)
        ));
        let after = owner.graph().resource(resource).unwrap();
        assert_eq!(after.storage, Some(backing));
        assert!(before.same_authority(&after.content));
    }

    #[test]
    fn physical_replacement_batch_revokes_one_shared_backing_representation_once() {
        let mut owner = ResourceLifecycleOwner::<&'static str>::new(VulkanDeviceEpochId::new(3));
        let backing = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::TaskAddress {
                    task: TaskId::new(1),
                    address: reims_vgpu_protocol::GuestVirtualAddress::new(0x4000),
                    length: ByteLength::new(0x2000),
                },
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let mut resources = Vec::new();
        for object in [1, 2] {
            let ResourceLifecycleEffect::ResourceCreated(resource) = owner
                .apply(ResolvedResourceLifecycle::CreateResource {
                    task: TaskId::new(1),
                    object: ObjectTableRef::new(object),
                    kind: ObjectKind::Buffer,
                    descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                    storage: Some(backing),
                    parents: Box::new([]),
                })
                .unwrap()
            else {
                unreachable!()
            };
            resources.push(resource);
        }
        owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                "old",
            )
            .unwrap();

        let ResourceLifecycleEffect::PhysicalBatchReplaced {
            resources: replaced,
            native,
        } = owner
            .apply(ResolvedResourceLifecycle::ReplacePhysicalBatch {
                resources: resources.clone().into_boxed_slice(),
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(replaced.len(), 2);
        assert_eq!(native.len(), 1);
        assert!(matches!(
            &native[0],
            (found, ManagedBackingProgress::RepresentationsRetired { ready, deferred: 0 })
                if *found == backing && ready == &["old"]
        ));
        assert_eq!(
            owner.any_designated_representation(backing),
            Err(ManagedBackingError::MissingExecutionRepresentation)
        );
    }

    #[test]
    fn resource_tree_release_retires_descendants_before_the_shared_backing() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = create_backing(&mut owner);
        let ResourceLifecycleEffect::ResourceCreated(root) = owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(1),
                kind: ObjectKind::SurfaceBacking,
                descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                storage: Some(backing),
                parents: Box::new([]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let ResourceLifecycleEffect::ResourceCreated(child) = owner
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(2),
                kind: ObjectKind::IOSurfacePlaneView,
                descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                storage: Some(backing),
                parents: Box::new([root]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let ResourceLifecycleEffect::ResourceTreeReleased {
            root: found,
            resources: released,
            retired,
        } = owner
            .apply(ResolvedResourceLifecycle::ReleaseResourceTree { root })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(found, root);
        assert_eq!(released.as_ref(), [child, root]);
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].storage.id, backing);
    }

    /// A refused upload route says which of its three facts was wrong.
    ///
    /// The backing on its own does not distinguish "nothing was designated"
    /// from "the designated route does not pair with the host ingress" from
    /// "the imported route has no direct guest alias under it", and those are
    /// three different repairs in three different places. A driven boot has
    /// already parked on this refusal reporting nothing but a backing id.
    #[test]
    fn a_refused_guest_upload_route_names_the_routes_it_refused() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let imported = owner
            .create_representation(
                backing,
                RepresentationRoute::ImportedGuestTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                (),
            )
            .unwrap();
        let transition = |target| {
            ResolvedResourceLifecycle::ApplyValidity(ResolvedValidityTransition {
                guest_write: None,
                ops: ResourceValidityOps {
                    clear_host_valid: 1,
                    set_host_valid: 0,
                    clear_guest_valid: 0,
                    set_guest_valid: 0,
                },
                write: None,
                targets: Box::new([target]),
            })
        };
        let target = ResolvedValidityTarget {
            backing,
            regions: Box::new([BackingRegion::Whole]),
            host_representation: None,
            host_ingress_destination: None,
            guest_upload_destination: Some(imported),
            guest_visibility_source: None,
            guest_visibility_destination: GUEST_REPRESENTATION,
        };

        // An imported-guest destination with no direct guest alias beneath it.
        // `guest_route` is the fact that says so, and it is `None` here
        // because nothing created that alias at all.
        assert_eq!(
            owner.apply(transition(target.clone())).err(),
            Some(ResourceLifecycleError::Validity(
                ValidityTransitionError::InvalidGuestUploadRoute {
                    backing,
                    destination: Some(imported),
                    destination_route: Some(RepresentationRoute::ImportedGuestTransfer {
                        working: crate::WorkingMemoryClass::DeviceLocal,
                    }),
                    guest_route: None,
                },
            )),
        );

        // Nothing designated at all, beside a host ingress that needs one.
        // The ingress endpoint is real, so what this refuses is the missing
        // upload destination and not the ingress.
        assert_eq!(
            owner
                .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
                .unwrap(),
            HOST_REPRESENTATION,
        );
        assert_eq!(
            owner
                .apply(transition(ResolvedValidityTarget {
                    host_ingress_destination: Some(HOST_REPRESENTATION),
                    guest_upload_destination: None,
                    ..target.clone()
                }))
                .err(),
            Some(ResourceLifecycleError::Validity(
                ValidityTransitionError::InvalidGuestUploadRoute {
                    backing,
                    destination: None,
                    destination_route: None,
                    guest_route: None,
                },
            )),
        );

        // The alias the imported route requires, so the same transition passes
        // -- which is what makes the two readings above facts about the route
        // and not about the transition.
        assert_eq!(
            owner
                .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
                .unwrap(),
            GUEST_REPRESENTATION,
        );
        assert!(owner.apply(transition(target)).is_ok());
    }

    #[test]
    fn validity_clear_then_set_splits_guest_write_from_gpu_completion() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let host = owner
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let submission = SubmissionId::new(7);
        let effect = owner
            .apply(ResolvedResourceLifecycle::ApplyValidity(
                ResolvedValidityTransition {
                    guest_write: None,
                    ops: ResourceValidityOps {
                        clear_host_valid: 1,
                        set_host_valid: 1,
                        clear_guest_valid: 0,
                        set_guest_valid: 0,
                    },
                    write: Some(submission.into()),
                    targets: Box::new([ResolvedValidityTarget {
                        backing,
                        regions: Box::new([BackingRegion::Whole]),
                        host_representation: Some(host),
                        host_ingress_destination: None,
                        guest_upload_destination: None,
                        guest_visibility_source: None,
                        guest_visibility_destination: GUEST_REPRESENTATION,
                    }]),
                },
            ))
            .unwrap();
        let ResourceLifecycleEffect::ValidityApplied {
            guest_writes,
            host_ingresses: _,
            deferred_host_ingress_transfers: _,
            gpu_reservations,
            gpu_completions,
            transfers,
            host_landings: _,
            states,
        } = effect
        else {
            unreachable!()
        };
        assert_eq!(guest_writes.len(), 1);
        assert_eq!(guest_writes[0].1.version.get(), 2);
        assert_eq!(transfers.as_ref(), []);
        assert_eq!(gpu_reservations.len(), 1);
        assert_eq!(gpu_reservations[0].backing, backing);
        assert_eq!(gpu_reservations[0].write, submission.into());
        assert_eq!(gpu_reservations[0].representation, host);
        assert!(!states[0].1.host_valid);
        assert!(states[0].1.host_stated);
        assert!(matches!(
            gpu_completions.as_ref(),
            [ResolvedResourceCompletion::ValidityHostWrite {
                backing: found,
                write: found_write,
                representation,
            }] if *found == backing && *found_write == submission.into() && *representation == host
        ));
        assert_eq!(
            owner
                .graph()
                .storage(backing)
                .unwrap()
                .content
                .snapshot_regions(&[BackingRegion::Whole])[0]
                .version
                .get(),
            2
        );
        let invalid = ResolvedResourceCompletion::GpuWrite {
            backing,
            write: SubmissionId::new(99).into(),
            representation: host,
        };
        let mut refused = gpu_completions.to_vec();
        refused.push(invalid);
        assert!(matches!(
            owner.complete_resources(&refused),
            Err(ResourceCompletionBatchError::Completion {
                completion,
                reason: ManagedBackingError::Content(
                    crate::ContentAuthorityError::SubmissionDidNotPlanWrite
                ),
            }) if completion == invalid
        ));
        assert!(!owner.backing_validity(backing).unwrap().host_valid);
        assert_eq!(
            owner
                .graph()
                .storage(backing)
                .unwrap()
                .content
                .snapshot_regions(&[BackingRegion::Whole])[0]
                .version
                .get(),
            2
        );
        owner.complete_resources(&gpu_completions).unwrap();
        assert!(owner.backing_validity(backing).unwrap().host_valid);
        assert_eq!(
            owner
                .graph()
                .storage(backing)
                .unwrap()
                .content
                .snapshot_regions(&[BackingRegion::Whole])[0]
                .version
                .get(),
            3
        );
    }

    #[test]
    fn invalid_validity_batch_changes_no_earlier_backing() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = create_backing(&mut owner);
        let missing = BackingId::new(backing.get() + 100);
        let region = BackingRegion::Linear(LinearRange::new(0, 0x1000).unwrap());
        let transition = ResolvedValidityTransition {
            ops: ResourceValidityOps::PAGE_ON,
            guest_write: None,
            write: None,
            targets: Box::new([
                ResolvedValidityTarget {
                    backing,
                    regions: Box::new([region]),
                    host_representation: None,
                    host_ingress_destination: None,
                    guest_upload_destination: None,
                    guest_visibility_source: None,
                    guest_visibility_destination: GUEST_REPRESENTATION,
                },
                ResolvedValidityTarget {
                    backing: missing,
                    regions: Box::new([region]),
                    host_representation: None,
                    host_ingress_destination: None,
                    guest_upload_destination: None,
                    guest_visibility_source: None,
                    guest_visibility_destination: GUEST_REPRESENTATION,
                },
            ]),
        };
        assert!(matches!(
            owner.apply(ResolvedResourceLifecycle::ApplyValidity(transition)),
            Err(ResourceLifecycleError::Validity(
                ValidityTransitionError::Backing {
                    backing: found,
                    reason: ManagedBackingError::UnknownBacking,
                }
            )) if found == missing
        ));
        assert_eq!(
            owner
                .graph()
                .storage(backing)
                .unwrap()
                .content
                .snapshot_regions(&[region])[0]
                .version
                .get(),
            1
        );
        assert_eq!(
            owner.backing_validity(backing),
            Some(ResourceValidity::default())
        );
    }

    #[test]
    fn heterogeneous_validity_batch_is_atomic_only_for_disjoint_backings() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let first = create_backing(&mut owner);
        let second = create_backing(&mut owner);
        let region = BackingRegion::Linear(LinearRange::new(0, 0x1000).unwrap());
        let transition = |backing, ops| ResolvedValidityTransition {
            guest_write: None,
            ops,
            write: None,
            targets: Box::new([ResolvedValidityTarget {
                backing,
                regions: Box::new([region]),
                host_representation: None,
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            }]),
        };
        let effect = owner
            .apply(ResolvedResourceLifecycle::ApplyValidityBatch(Box::new([
                transition(first, ResourceValidityOps::PAGE_ON),
                transition(second, ResourceValidityOps::default()),
            ])))
            .unwrap();
        assert!(matches!(
            effect,
            ResourceLifecycleEffect::ValidityBatchApplied(effects)
                if effects.len() == 2
                    && matches!(effects[0], ResourceLifecycleEffect::ValidityApplied { .. })
                    && matches!(effects[1], ResourceLifecycleEffect::ValidityApplied { .. })
        ));
        assert!(matches!(
            owner.apply(ResolvedResourceLifecycle::ApplyValidityBatch(Box::new([
                transition(first, ResourceValidityOps::default()),
                transition(first, ResourceValidityOps::default()),
            ]))),
            Err(ResourceLifecycleError::DuplicateValidityBatchBacking(found)) if found == first
        ));
    }

    #[test]
    fn guest_valid_statement_does_not_replace_the_required_transfer() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let host = owner
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        owner
            .plan_gpu_write(backing, SubmissionId::new(1), host, [BackingRegion::Whole])
            .unwrap();
        owner
            .complete_gpu_write(backing, SubmissionId::new(1), host)
            .unwrap();
        let snapshot = owner
            .graph()
            .storage(backing)
            .unwrap()
            .content
            .snapshot_regions(&[BackingRegion::Whole]);
        assert!(!owner
            .graph()
            .storage(backing)
            .unwrap()
            .content
            .representation_matches(GUEST_REPRESENTATION, &snapshot));

        let effect = owner
            .apply(ResolvedResourceLifecycle::ApplyValidity(
                ResolvedValidityTransition {
                    guest_write: None,
                    ops: ResourceValidityOps {
                        clear_host_valid: 0,
                        set_host_valid: 0,
                        clear_guest_valid: 1,
                        set_guest_valid: 1,
                    },
                    write: None,
                    targets: Box::new([ResolvedValidityTarget {
                        backing,
                        regions: Box::new([BackingRegion::Whole]),
                        host_representation: None,
                        host_ingress_destination: None,
                        guest_upload_destination: None,
                        guest_visibility_source: Some(host),
                        guest_visibility_destination: GUEST_REPRESENTATION,
                    }]),
                },
            ))
            .unwrap();
        let ResourceLifecycleEffect::ValidityApplied {
            transfers, states, ..
        } = effect
        else {
            unreachable!()
        };
        assert_eq!(transfers.len(), 1);
        assert_eq!(transfers[0].source, host);
        assert_eq!(transfers[0].destination, GUEST_REPRESENTATION);
        assert!(states[0].1.guest_valid);
        assert!(states[0].1.guest_stated);
        assert!(!owner
            .graph()
            .storage(backing)
            .unwrap()
            .content
            .representation_matches(GUEST_REPRESENTATION, &snapshot));
        owner.complete_transfer(transfers[0]).unwrap();
        assert!(owner
            .graph()
            .storage(backing)
            .unwrap()
            .content
            .representation_matches(GUEST_REPRESENTATION, &snapshot));
    }

    #[test]
    fn multi_backing_submit_prevalidates_the_complete_set_before_mutation() {
        let epoch = VulkanDeviceEpochId::new(3);
        let mut owner = ResourceLifecycleOwner::<()>::new(epoch);
        let first = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let second = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let first_representation = owner
            .create_representation(first, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        let second_representation = owner
            .create_representation(second, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        let transaction = TransactionId::new(11);
        owner
            .accept_use(first, transaction, [first_representation])
            .unwrap();
        let point = QueueTimelinePoint {
            epoch,
            queue: QueueOwnerId::new(1),
            value: QueueTimelineValue::new(4),
        };
        assert_eq!(
            owner.submit_uses(&[first, second], transaction, point),
            Err(ResourceUseBatchError::Backing {
                backing: second,
                reason: ManagedBackingError::UnknownAcceptedUse,
            })
        );
        owner
            .accept_use(second, transaction, [second_representation])
            .unwrap();
        assert!(owner
            .submit_uses(&[first, second], transaction, point)
            .is_ok());
        assert_eq!(
            owner.validate_submit_uses(&[first, first], transaction, point),
            Err(ResourceUseBatchError::DuplicateBacking(first))
        );
    }

    #[test]
    fn resource_completion_batch_refusal_preserves_every_pending_effect() {
        let mut owner = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(3));
        let backing = match owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let representation = owner
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        let submission = SubmissionId::new(5);
        owner
            .plan_gpu_write(backing, submission, representation, [BackingRegion::Whole])
            .unwrap();
        let valid = ResolvedResourceCompletion::GpuWrite {
            backing,
            write: submission.into(),
            representation,
        };
        let invalid = ResolvedResourceCompletion::GpuWrite {
            backing,
            write: SubmissionId::new(6).into(),
            representation,
        };
        assert!(matches!(
            owner.complete_resources(&[valid, invalid]),
            Err(ResourceCompletionBatchError::Completion {
                completion,
                reason: ManagedBackingError::Content(
                    crate::ContentAuthorityError::SubmissionDidNotPlanWrite
                ),
            }) if completion == invalid
        ));
        assert!(matches!(
            owner.complete_resources(&[valid]).unwrap().as_slice(),
            [ResourceCompletionEffect::GpuWrite {
                backing: found,
                submission: completed,
                ..
            }] if *found == backing && *completed == submission
        ));
    }
}
