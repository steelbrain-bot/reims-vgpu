//! Canonical task-owned resource, storage, and mapping graph.

use crate::content_authority::{
    BackingRegion, ContentAuthority, ContentAuthorityError, GPU_REPRESENTATION,
    GUEST_REPRESENTATION, HOST_REPRESENTATION,
};
use reims_vgpu_protocol::{
    BackingGeneration, BackingId, ByteLength, ByteOffset, ContentVersion, GuestVirtualAddress,
    HeapObject, MapperSurfaceRef, MappingId, ObjectKind, ObjectTableRef, PlaneIndex,
    ResourceDescriptor, ResourceId, ResourceObject, SubmissionId, SwizzlePlan, TaskId, TextureType,
    TextureViewForm,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::{atomic::AtomicU64, Weak};

type AnyResourceId = ResourceId<ResourceObject>;

/// The contract-defined maximum number of texture-view hops followed while
/// resolving one view. It bounds malformed guest cycles; it is not a cache or
/// a capacity limit on live resources.
pub const MAX_TEXTURE_VIEW_CHAIN: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedTextureViewRange {
    pub level_base: u64,
    pub level_count: u64,
    pub slice_base: u64,
    pub slice_count: u64,
}

impl ResolvedTextureViewRange {
    fn compose_over(self, inner: Self) -> Result<Self, TextureViewResolveError> {
        let level_end = self
            .level_base
            .checked_add(self.level_count)
            .ok_or(TextureViewResolveError::LevelOverflow)?;
        if level_end > inner.level_count {
            return Err(TextureViewResolveError::LevelOutOfRange);
        }
        let slice_end = self
            .slice_base
            .checked_add(self.slice_count)
            .ok_or(TextureViewResolveError::SliceOverflow)?;
        if slice_end > inner.slice_count {
            return Err(TextureViewResolveError::SliceOutOfRange);
        }
        Ok(Self {
            level_base: inner
                .level_base
                .checked_add(self.level_base)
                .ok_or(TextureViewResolveError::LevelOverflow)?,
            level_count: self.level_count,
            slice_base: inner
                .slice_base
                .checked_add(self.slice_base)
                .ok_or(TextureViewResolveError::SliceOverflow)?,
            slice_count: self.slice_count,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedTextureView {
    pub view: AnyResourceId,
    pub base: AnyResourceId,
    pub range: Option<ResolvedTextureViewRange>,
    pub texture_type: Option<TextureType>,
    pub pixel_format: Option<u16>,
    pub swizzle: Option<SwizzlePlan>,
}

/// Complete shader-visible view over one canonical texture resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedTextureBindingView {
    pub resource: AnyResourceId,
    pub base: AnyResourceId,
    /// The resource that owns the native image this binding reads through.
    ///
    /// Not always `base`: a view aliasing its base's storage reads the base's
    /// image, but an IOSurface plane view owns the plane it names and so owns
    /// its own. See [`ResourceGraph::image_owner`].
    pub image_owner: AnyResourceId,
    pub range: ResolvedTextureViewRange,
    pub texture_type: TextureType,
    pub pixel_format: u16,
    pub swizzle: SwizzlePlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureViewResolveError {
    ResourceAbsent(AnyResourceId),
    NotTextureView(AnyResourceId),
    DescriptorAbsent(AnyResourceId),
    DescriptorKindMismatch(AnyResourceId),
    /// A registered-surface plane view whose nested record carried no view
    /// geometry. The surface relation is known and the view is not, which is
    /// a different fact from a descriptor of the wrong kind and waits on a
    /// different thing.
    PlaneGeometryAbsent(AnyResourceId),
    ParentCount(AnyResourceId),
    EmptyRange(AnyResourceId),
    UnknownTextureType(u16),
    UnsupportedTextureType(TextureType),
    InvalidSwizzle(AnyResourceId),
    LevelOverflow,
    LevelOutOfRange,
    SliceOverflow,
    SliceOutOfRange,
    ChainOverflow(AnyResourceId),
    BaseDescriptorAbsent(AnyResourceId),
    UnsupportedBaseDescriptor(AnyResourceId),
    BaseDeclarationAbsent(AnyResourceId),
    EmptyBaseGeometry(AnyResourceId),
    ViewRangeOutsideBase(AnyResourceId),
}

fn texture_declaration_view_facts(
    declaration: reims_vgpu_protocol::TextureDeclaration,
    base: AnyResourceId,
) -> Result<(TextureType, u64, u64, u16), TextureViewResolveError> {
    let texture_type = declaration.texture_type;
    if let TextureType::Unknown(raw) = texture_type {
        return Err(TextureViewResolveError::UnknownTextureType(u16::from(raw)));
    }
    if texture_type == TextureType::Buffer {
        return Err(TextureViewResolveError::UnsupportedTextureType(
            texture_type,
        ));
    }
    let declared_layers = u64::from(declaration.array_length);
    let layers = if matches!(texture_type, TextureType::Cube | TextureType::CubeArray) {
        declared_layers
            .checked_mul(u64::from(reims_vgpu_protocol::CUBE_FACES))
            .ok_or(TextureViewResolveError::EmptyBaseGeometry(base))?
    } else {
        declared_layers
    };
    Ok((
        texture_type,
        u64::from(declaration.mipmap_level_count),
        layers,
        declaration.pixel_format,
    ))
}

static NEXT_RESOURCE_LIFETIME: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct ResourceLifetimeToken {
    id: u64,
}

/// Strong ownership token for one constructed semantic resource lifetime.
///
/// Backend caches receive only [`ResourceLifetimeRef`]. Entries therefore die
/// with the guest-owned resource rather than an invented capacity or timer.
#[derive(Clone, Debug)]
pub struct ResourceLifetime(Arc<ResourceLifetimeToken>);

impl Default for ResourceLifetime {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceLifetime {
    pub fn new() -> Self {
        let id = NEXT_RESOURCE_LIFETIME
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |id| id.checked_add(1),
            )
            .expect("resource lifetime identity exhausted");
        Self(Arc::new(ResourceLifetimeToken { id }))
    }

    pub fn reference(&self) -> ResourceLifetimeRef {
        ResourceLifetimeRef {
            id: self.0.id,
            live: Arc::downgrade(&self.0),
        }
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }
}

/// Weak executor-facing proof that one semantic resource still exists.
#[derive(Clone, Debug)]
pub struct ResourceLifetimeRef {
    id: u64,
    live: Weak<ResourceLifetimeToken>,
}

impl ResourceLifetimeRef {
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn is_live(&self) -> bool {
        self.live.strong_count() != 0
    }
}

/// Exact semantic content represented by one executor operand or effect.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ContentStamp {
    pub resource: ResourceId<ResourceObject>,
    pub version: ContentVersion,
}

/// Where a resource sits with respect to submissions that reference it.
///
/// There is no `Prepared` between [`Self::Created`] and [`Self::InFlight`].
/// There was one, and nothing ever read it: the only writer assigned it and the
/// next line of the same function overwrote it with `InFlight`, both under the
/// one lock that guards this graph, so no observer anywhere could sample a
/// resource in it. It is gone with the `prepare`/`submit` split that produced
/// it — see [`ResourceGraph::enter_submission`], which performs that pair as
/// the single transition its caller always wanted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleState {
    Created,
    InFlight,
    Released,
}

/// What the alias walk costs, as three running totals this crate owns and the
/// composition crate reports once a second on the `OFF` channel.
///
/// [`ResourceGraph::guest_wrote_aliases`] runs once per resource descriptor that
/// declares a guest write — around a million times a minute on a driven
/// fullscreen Maps boot — so a term in it that is not proportional to the
/// resource being written is paid a million times. The address-overlap search
/// was such a term: it scanned every storage node in the device, and counting
/// it is what turned "this looks quadratic" into 3 947.6 nodes examined per
/// walk against 1.00 resource actually visited. Arithmetic would not have
/// settled it — two arithmetic guesses in this same neighbourhood measured at
/// nothing.
///
/// `scan_iters / walks` is now the standing guard rather than a diagnosis:
/// `ResourceGraph::storage_overlapping` holds it near 1, and anything that
/// reintroduces a scan moves it by three orders of magnitude in a counter that
/// is already being printed. Three relaxed adds per walk is what that costs.
pub mod alias_walk_census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    static WALKS: AtomicU64 = AtomicU64::new(0);
    static VISITED: AtomicU64 = AtomicU64::new(0);
    static SCAN_ITERS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn note(walks: u64, visited: u64, scan_iters: u64) {
        WALKS.fetch_add(walks, Relaxed);
        VISITED.fetch_add(visited, Relaxed);
        SCAN_ITERS.fetch_add(scan_iters, Relaxed);
    }

    /// Read and reset the three totals: `(walks, visited, scan_iters)`.
    pub fn take() -> (u64, u64, u64) {
        (
            WALKS.swap(0, Relaxed),
            VISITED.swap(0, Relaxed),
            SCAN_ITERS.swap(0, Relaxed),
        )
    }
}

/// Versions held by the three places resource bytes can reside.
///
/// `None` means that replica does not contain bytes from this resource's
/// current lifetime. Currency is always derived by comparing a replica with
/// [`ContentState::current`]; there is no independent "GPU only" latch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicaVersions {
    pub guest: Option<ContentVersion>,
    pub gpu: Option<ContentVersion>,
    pub host: Option<ContentVersion>,
}

/// A version reserved for a submitted GPU write which has not completed yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingContentWrite {
    pub submission: SubmissionId,
    pub version: ContentVersion,
}

/// The sole authority for which replica contains a resource's newest bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentState {
    pub current: ContentVersion,
    pub replicas: ReplicaVersions,
    pending_gpu_writes: BTreeMap<SubmissionId, ContentVersion>,
    next_version: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentError {
    SubmissionAlreadyWrites,
    SubmissionDidNotPlanWrite,
    StaleSource,
    WouldLoseCurrentContent,
    VersionSpaceExhausted,
}

impl Default for ContentState {
    fn default() -> Self {
        let initial = ContentVersion::new(1);
        Self {
            current: initial,
            replicas: ReplicaVersions {
                guest: Some(initial),
                gpu: None,
                host: None,
            },
            pending_gpu_writes: BTreeMap::new(),
            next_version: 2,
        }
    }
}

impl ContentState {
    fn reserve_version(&mut self) -> Result<ContentVersion, ContentError> {
        let version = ContentVersion::new(self.next_version);
        self.next_version = self
            .next_version
            .checked_add(1)
            .ok_or(ContentError::VersionSpaceExhausted)?;
        Ok(version)
    }

    pub fn guest_wrote(&mut self) -> Result<ContentVersion, ContentError> {
        let version = self.reserve_version()?;
        self.current = version;
        self.replicas.guest = Some(version);
        Ok(version)
    }

    pub fn gpu_store_planned(
        &mut self,
        submission: SubmissionId,
    ) -> Result<PendingContentWrite, ContentError> {
        if self.pending_gpu_writes.contains_key(&submission) {
            return Err(ContentError::SubmissionAlreadyWrites);
        }
        let version = self.reserve_version()?;
        self.pending_gpu_writes.insert(submission, version);
        Ok(PendingContentWrite {
            submission,
            version,
        })
    }

    pub fn gpu_store_completed(
        &mut self,
        submission: SubmissionId,
    ) -> Result<ContentVersion, ContentError> {
        let version = self
            .pending_gpu_writes
            .remove(&submission)
            .ok_or(ContentError::SubmissionDidNotPlanWrite)?;
        self.replicas.gpu = Some(version);
        if version > self.current {
            self.current = version;
        }
        Ok(version)
    }

    pub fn copy_gpu_to_guest_completed(
        &mut self,
        version: ContentVersion,
    ) -> Result<(), ContentError> {
        if self.replicas.gpu != Some(version) || self.current != version {
            return Err(ContentError::StaleSource);
        }
        self.replicas.guest = Some(version);
        Ok(())
    }

    pub fn copy_guest_to_gpu_completed(
        &mut self,
        version: ContentVersion,
    ) -> Result<(), ContentError> {
        if self.replicas.guest != Some(version) || self.current != version {
            return Err(ContentError::StaleSource);
        }
        self.replicas.gpu = Some(version);
        Ok(())
    }

    /// Record that an executor materialized this exact current version.
    ///
    /// Unlike [`Self::copy_guest_to_gpu_completed`], the source replica is
    /// already established by the stamped operand.  The version check is the
    /// authority: a completion for content superseded since submission cannot
    /// make the stale GPU replica current.
    pub fn gpu_materialized(&mut self, version: ContentVersion) -> Result<(), ContentError> {
        if self.current != version {
            return Err(ContentError::StaleSource);
        }
        self.replicas.gpu = Some(version);
        Ok(())
    }

    pub fn current_in_guest(&self) -> bool {
        self.replicas.guest == Some(self.current)
    }

    pub fn current_in_gpu(&self) -> bool {
        self.replicas.gpu == Some(self.current)
    }

    pub fn replace_guest_backing(&mut self) -> Result<(), ContentError> {
        if self.current_in_guest()
            && self.replicas.gpu != Some(self.current)
            && self.replicas.host != Some(self.current)
        {
            return Err(ContentError::WouldLoseCurrentContent);
        }
        self.replicas.guest = None;
        Ok(())
    }
}

impl ContentAuthority {
    /// Compatibility projection for callers which still consume the entire
    /// backing as one region. Canonical ownership remains the regional state.
    pub fn snapshot(&self) -> ContentState {
        let current = self.whole_current();
        ContentState {
            current,
            replicas: ReplicaVersions {
                guest: self.whole_matches(GUEST_REPRESENTATION).then_some(current),
                gpu: self.whole_matches(GPU_REPRESENTATION).then_some(current),
                host: self.whole_matches(HOST_REPRESENTATION).then_some(current),
            },
            pending_gpu_writes: BTreeMap::new(),
            next_version: current.get().saturating_add(1),
        }
    }

    pub fn current(&self) -> ContentVersion {
        self.whole_current()
    }

    pub fn guest_wrote(&self) -> Result<ContentVersion, ContentError> {
        self.guest_write_region(None, GUEST_REPRESENTATION, BackingRegion::Whole)
            .map(|write| write.version)
            .map_err(content_error)
    }

    pub fn gpu_store_planned(
        &self,
        submission: SubmissionId,
    ) -> Result<PendingContentWrite, ContentError> {
        self.ensure_representation(GPU_REPRESENTATION);
        let writes = self
            .plan_gpu_write_regions(submission, GPU_REPRESENTATION, [BackingRegion::Whole])
            .map_err(content_error)?;
        Ok(PendingContentWrite {
            submission,
            version: writes[0].version,
        })
    }

    pub fn gpu_store_completed(
        &self,
        submission: SubmissionId,
    ) -> Result<ContentVersion, ContentError> {
        self.ensure_representation(GPU_REPRESENTATION);
        let writes = self
            .complete_gpu_write_regions(submission, GPU_REPRESENTATION)
            .map_err(content_error)?;
        Ok(writes[0].version)
    }

    pub fn copy_gpu_to_guest_completed(&self, version: ContentVersion) -> Result<(), ContentError> {
        if !self.whole_matches(GPU_REPRESENTATION) {
            return Err(ContentError::StaleSource);
        }
        self.materialize_whole(GUEST_REPRESENTATION, version)
            .map_err(content_error)
    }

    pub fn gpu_materialized(&self, version: ContentVersion) -> Result<(), ContentError> {
        self.materialize_whole(GPU_REPRESENTATION, version)
            .map_err(content_error)
    }

    pub fn replace_guest_backing(&self) -> Result<(), ContentError> {
        if self.whole_matches(GUEST_REPRESENTATION)
            && !self.whole_matches(GPU_REPRESENTATION)
            && !self.whole_matches(HOST_REPRESENTATION)
        {
            return Err(ContentError::WouldLoseCurrentContent);
        }
        self.remove_whole_representation(GUEST_REPRESENTATION);
        Ok(())
    }
}

fn content_error(error: ContentAuthorityError) -> ContentError {
    match error {
        ContentAuthorityError::GpuWriteAlreadyPlanned => ContentError::SubmissionAlreadyWrites,
        ContentAuthorityError::SubmissionDidNotPlanWrite => ContentError::SubmissionDidNotPlanWrite,
        ContentAuthorityError::StaleSource
        | ContentAuthorityError::GpuWriteReservationMismatch
        | ContentAuthorityError::GpuWriteRepresentationMismatch
        | ContentAuthorityError::UnknownRepresentation
        | ContentAuthorityError::TransferNotPlanned
        | ContentAuthorityError::InsufficientTransferDemand
        | ContentAuthorityError::HostLandingSourceMismatch
        | ContentAuthorityError::HostLandingNotPlanned
        | ContentAuthorityError::HostLandingSourceNotCurrent
        | ContentAuthorityError::HostIngressNotPlanned
        | ContentAuthorityError::HostIngressSourceNotCurrent
        | ContentAuthorityError::UnboundBacking
        | ContentAuthorityError::EmptyBacking => ContentError::StaleSource,
        ContentAuthorityError::VersionSpaceExhausted
        | ContentAuthorityError::TransferDemandCountExhausted => {
            ContentError::VersionSpaceExhausted
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StorageBacking {
    Dedicated,
    TaskAddress {
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    },
    BufferRange {
        buffer: AnyResourceId,
        offset: ByteOffset,
        bytes_per_row: ByteLength,
    },
    /// One declared plane of a registered IOSurface.
    ///
    /// The surface itself is an allocation and owns no backing: it may declare
    /// several planes at their own offsets with their own extent, row pitch
    /// and pixel format, and a backing carries one execution representation
    /// and one layout. The plane's owning surface is named by its generational
    /// resource identity, so a reused object slot cannot redirect a live
    /// plane.
    IOSurfacePlane {
        surface: AnyResourceId,
        plane: PlaneIndex,
    },
    /// Mapper-path surface storage whose shared-backing identity is not yet
    /// established independently of the mapping object.
    MapperSurface {
        mapper: MapperSurfaceRef,
        plane: PlaneIndex,
    },
    HeapPlacement {
        heap: ResourceId<HeapObject>,
        offset: ByteOffset,
        length: ByteLength,
    },
    /// A texture whose offset is selected by the heap allocator rather than by
    /// the guest. The allocation's resource lifetime is its identity; the wire
    /// offset is deliberately absent because `use_offset == false` says it has
    /// no meaning.
    HeapAllocation {
        heap: ResourceId<HeapObject>,
        allocation: AnyResourceId,
    },
}

/// Which class of storage a backing is, without the fields that distinguish
/// two backings of the same class.
///
/// A materializer serves exactly one class and must refuse the rest by name:
/// only [`StorageBacking::TaskAddress`] carries the guest address and length a
/// fresh materialization needs, and every other class has its own materializer.
/// A refusal that says only "this backing is unavailable" cannot tell a backing
/// this device has never heard of from one whose class is simply somebody
/// else's job, and those want opposite repairs. Carrying the class rather than
/// the whole [`StorageBacking`] keeps the refusal `Copy` and keeps a guest
/// address out of a diagnostic.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StorageClass {
    Dedicated,
    TaskAddress,
    BufferRange,
    IOSurfacePlane,
    MapperSurface,
    HeapPlacement,
    HeapAllocation,
}

impl StorageBacking {
    pub const fn class(&self) -> StorageClass {
        match self {
            Self::Dedicated => StorageClass::Dedicated,
            Self::TaskAddress { .. } => StorageClass::TaskAddress,
            Self::BufferRange { .. } => StorageClass::BufferRange,
            Self::IOSurfacePlane { .. } => StorageClass::IOSurfacePlane,
            Self::MapperSurface { .. } => StorageClass::MapperSurface,
            Self::HeapPlacement { .. } => StorageClass::HeapPlacement,
            Self::HeapAllocation { .. } => StorageClass::HeapAllocation,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageNode {
    pub id: BackingId,
    pub backing: StorageBacking,
    pub owners: BTreeSet<AnyResourceId>,
    pub content: ContentAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingNode {
    pub id: MappingId,
    pub task: TaskId,
    pub address: GuestVirtualAddress,
    pub length: ByteLength,
    pub storage: Option<BackingId>,
    pub committed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceNode {
    pub id: AnyResourceId,
    pub task: TaskId,
    pub object: ObjectTableRef<ResourceObject>,
    pub kind: ObjectKind,
    /// Complete semantic construction input decoded once at the namespace
    /// boundary. Legacy graph-only callers may omit it; replacement lifecycle
    /// admission always supplies it before native representation planning.
    pub descriptor: Option<Arc<ResourceDescriptor>>,
    pub lifecycle: LifecycleState,
    pub storage: Option<BackingId>,
    pub backing_generation: BackingGeneration,
    pub content: ContentAuthority,
    pub parents: BTreeSet<AnyResourceId>,
    pub children: BTreeSet<AnyResourceId>,
    pub in_flight: BTreeSet<SubmissionId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphError {
    ReferenceAlreadyBound,
    ReferenceUnbound,
    ResourceAbsent,
    ParentAbsent,
    ParentCycle,
    StorageAbsent,
    StorageInUse,
    StorageConflict,
    HeapPlacementOverlap,
    MappingAlreadyExists,
    MappingAbsent,
    SubmissionNotPrepared,
    Content(ContentAuthorityError),
    DescriptorConflict,
    IdentitySpaceExhausted,
}

impl From<ContentAuthorityError> for GraphError {
    fn from(error: ContentAuthorityError) -> Self {
        Self::Content(error)
    }
}

#[derive(Debug)]
struct NamespaceSlot {
    index: u32,
    next_generation: u32,
    current: Option<AnyResourceId>,
    /// The generation this name last held, kept after release.
    ///
    /// A refusal that says only "released" cannot be acted on: it does not say
    /// whether the name was retired long ago or emptied a moment before by a
    /// rebind whose replacement never landed, and those are opposite defects.
    released: Option<AnyResourceId>,
}

/// What [`ResourceGraph::slot_state`] found for one object-table name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectSlotState {
    /// No declaration this device admitted has ever named this slot.
    Undeclared,
    /// The slot was bound and its object has since been released, so the name
    /// is free for the guest to reuse under a new generation. Carries the
    /// generation it last held, which is what says *which* release emptied it.
    Released(Option<AnyResourceId>),
    /// The slot names a live resource.
    Bound(AnyResourceId),
}

/// One authority for object-name reuse and resource/storage/mapping lifetime.
#[derive(Debug)]
pub struct ResourceGraph {
    slots: BTreeMap<(TaskId, ObjectTableRef<ResourceObject>), NamespaceSlot>,
    resources: BTreeMap<AnyResourceId, ResourceNode>,
    /// Every [`StorageBacking::TaskAddress`] node, ordered by the guest range it
    /// backs: `(task, start, storage id) -> end`.
    ///
    /// Asking which storage nodes overlap a guest range used to be a walk of
    /// **every** node in the device, inside a walk run once per resource
    /// descriptor that declares a guest write. One driven fullscreen Maps boot
    /// counted 3 876 645 110 such examinations in 45 seconds — 3 947.6 nodes per
    /// walk over 982 038 walks — against 1.00 resource actually visited.
    ///
    /// Removing it is worth, over three interleaved boots an arm with the arms
    /// disjoint on every axis: the walk's own scan 3 801 -> 1.2 nodes examined,
    /// the `open` phase 4.19 -> 1.00 us per draw (-76 %), and the whole drain
    /// worker 27.22 -> 23.51 us per draw (-13.6 %).
    ///
    /// This is an **index, not a memo**: it is derived from the same nodes on
    /// every mutation and holds no computed answer, so it has no hit rate, no
    /// eviction and no invalidation. It is maintained at `create_storage`, which
    /// is the only place a node or its backing is ever written — nothing
    /// reassigns `StorageNode::backing` and nothing removes a node — so it
    /// cannot drift from the map it indexes.
    ///
    /// The storage id is in the key so two nodes may back the same range.
    task_address_index: BTreeMap<(TaskId, u64, u64), u64>,
    /// The longest `TaskAddress` extent recorded for a task, which is what
    /// bounds a query: any range overlapping `[start, end)` must begin at or
    /// after `start - longest`. It only grows, so it can widen the scanned
    /// window but can never narrow it past a real overlap.
    longest_task_extent: BTreeMap<TaskId, u64>,
    storage: BTreeMap<BackingId, StorageNode>,
    automatically_retired_storage: Vec<StorageNode>,
    mappings: BTreeMap<MappingId, MappingNode>,
    next_resource_index: u32,
    next_storage_id: u64,
}

impl Default for ResourceGraph {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
            resources: BTreeMap::new(),
            task_address_index: BTreeMap::new(),
            longest_task_extent: BTreeMap::new(),
            storage: BTreeMap::new(),
            automatically_retired_storage: Vec::new(),
            mappings: BTreeMap::new(),
            next_resource_index: 1,
            next_storage_id: 1,
        }
    }
}

impl ResourceGraph {
    /// Record bytes changing through one resource and every declared alias of
    /// those bytes.
    ///
    /// Task-address storage is range-shaped: a buffer may describe an entire
    /// allocation while a texture describes only an overlapping view. Exact
    /// `(address, length)` interning therefore cannot be the content boundary.
    /// Parent/view edges and overlapping task-address ranges together define
    /// the alias set whose content authorities must advance.
    pub fn guest_wrote_aliases(&self, id: AnyResourceId) -> Option<ContentVersion> {
        let source = self.resources.get(&id)?.content.clone();
        let (resources, _, scan_iters) = self.alias_closure(id);
        let mut authorities = Vec::<ContentAuthority>::new();
        for resource_id in &resources {
            let resource = self
                .resources
                .get(resource_id)
                .expect("alias closure contains only live resources");
            if !authorities
                .iter()
                .any(|known| known.same_authority(&resource.content))
            {
                authorities.push(resource.content.clone());
            }
        }

        alias_walk_census::note(1, resources.len() as u64, scan_iters);
        let version = source.guest_wrote().ok()?;
        for authority in authorities {
            if !authority.same_authority(&source) {
                authority.guest_wrote().ok()?;
            }
        }
        Some(version)
    }

    /// Every canonical backing connected to one resource by declared
    /// parent/view, shared-storage, heap-alias, or overlapping task-address
    /// relations. Dependency compilation uses these identities directly; it
    /// never invents an alias key from an object name or host address.
    pub fn alias_backings(&self, id: AnyResourceId) -> Option<Box<[BackingId]>> {
        self.resources.get(&id)?;
        let (_, backings, _) = self.alias_closure(id);
        Some(backings.into_iter().collect::<Vec<_>>().into_boxed_slice())
    }

    /// Resolve the one storage backing inherited by an operation endpoint
    /// through explicit parent/view relations. The wider alias closure is not
    /// eligible: overlapping allocations may legitimately contribute several
    /// hazard identities, while an endpoint must name one storage object.
    /// The resource that owns the image one binding reads through.
    ///
    /// A texture view is declared carrying its base's storage, so "owns
    /// storage" does not tell the two apart -- both do. What does is whether a
    /// parent owns the *same* storage: a texture view's base does, so the view
    /// reads the base's image and owns none, while an IOSurface plane view
    /// owns the plane it names and its parent surface owns the whole
    /// allocation and no backing at all. So the owner is the outermost
    /// ancestor sharing this resource's storage, which is the resource itself
    /// whenever no parent shares it.
    ///
    /// The view chain's `base` answers the first shape and inverts the second,
    /// which is why this is not that. `None` where the resource owns no
    /// storage -- a registered surface, which is an allocation and not an
    /// image -- or where two parents share the storage and neither is the
    /// outermost, which is a graph this device does not build.
    pub fn image_owner(&self, id: AnyResourceId) -> Option<AnyResourceId> {
        let storage = self.resources.get(&id)?.storage?;
        let mut owner = id;
        let mut visited = BTreeSet::new();
        while visited.insert(owner) {
            let node = self.resources.get(&owner)?;
            let mut sharing = node.parents.iter().copied().filter(|parent| {
                self.resources
                    .get(parent)
                    .is_some_and(|node| node.storage == Some(storage))
            });
            let (Some(next), None) = (sharing.next(), sharing.next()) else {
                return Some(owner);
            };
            owner = next;
        }
        Some(owner)
    }

    pub fn resolved_backing(&self, id: AnyResourceId) -> Option<BackingId> {
        self.resources.get(&id)?;
        let mut pending = vec![id];
        let mut visited = BTreeSet::new();
        let mut resolved = None;
        while let Some(resource_id) = pending.pop() {
            let resource = self.resources.get(&resource_id)?;
            if !visited.insert(resource_id) {
                continue;
            }
            if let Some(storage) = resource.storage {
                match resolved {
                    None => resolved = Some(storage),
                    Some(existing) if existing == storage => {}
                    Some(_) => return None,
                }
                // A resource owning storage *is* its backing. An IOSurface
                // plane view owns the plane it names while its parent surface
                // owns the whole allocation, so climbing past the first owner
                // would read two disagreeing backings off a graph that is
                // perfectly well formed.
                continue;
            }
            pending.extend(resource.parents.iter().copied());
        }
        resolved
    }

    fn alias_closure(
        &self,
        id: AnyResourceId,
    ) -> (BTreeSet<AnyResourceId>, BTreeSet<BackingId>, u64) {
        let mut scan_iters = 0u64;
        let mut pending = vec![id];
        let mut visited = BTreeSet::new();
        let mut backings = BTreeSet::new();
        let mut ranges = Vec::<(TaskId, u64, u64)>::new();

        while let Some(resource_id) = pending.pop() {
            let Some(resource) = self.resources.get(&resource_id) else {
                continue;
            };
            if !visited.insert(resource_id) {
                continue;
            }
            pending.extend(resource.parents.iter().copied());
            pending.extend(resource.children.iter().copied());
            let Some(storage_id) = resource.storage else {
                continue;
            };
            let Some(storage) = self.storage.get(&storage_id) else {
                continue;
            };
            backings.insert(storage_id);
            pending.extend(storage.owners.iter().copied());
            if let StorageBacking::TaskAddress {
                task,
                address,
                length,
            } = storage.backing
            {
                let start = address.get();
                let end = start.saturating_add(length.get());
                if start < end && !ranges.contains(&(task, start, end)) {
                    ranges.push((task, start, end));
                    let overlapping = self.storage_overlapping(task, start, end);
                    let mut examined = 0u64;
                    let owners = overlapping
                        .inspect(|_| examined = examined.saturating_add(1))
                        .filter_map(|id| self.storage.get(&id))
                        .flat_map(|candidate| candidate.owners.iter().copied())
                        .collect::<Vec<_>>();
                    scan_iters = scan_iters.saturating_add(examined);
                    pending.extend(owners);
                }
            }
        }
        (visited, backings, scan_iters)
    }

    fn parent_edge_would_cycle(&self, parent: AnyResourceId, child: AnyResourceId) -> bool {
        if parent == child {
            return true;
        }
        let mut pending = vec![child];
        let mut visited = BTreeSet::new();
        while let Some(candidate) = pending.pop() {
            if !visited.insert(candidate) {
                continue;
            }
            if candidate == parent {
                return true;
            }
            if let Some(resource) = self.resources.get(&candidate) {
                pending.extend(resource.children.iter().copied());
            }
        }
        false
    }

    pub fn create_resource(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        kind: ObjectKind,
        storage: Option<BackingId>,
        parents: impl IntoIterator<Item = AnyResourceId>,
    ) -> Result<AnyResourceId, GraphError> {
        self.create_resource_with_descriptor(task, object, kind, None, storage, parents)
    }

    pub fn create_resource_with_descriptor(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        kind: ObjectKind,
        descriptor: Option<Arc<ResourceDescriptor>>,
        storage: Option<BackingId>,
        parents: impl IntoIterator<Item = AnyResourceId>,
    ) -> Result<AnyResourceId, GraphError> {
        if storage.is_some_and(|id| !self.storage.contains_key(&id)) {
            return Err(GraphError::StorageAbsent);
        }
        let parents: BTreeSet<_> = parents.into_iter().collect();
        if parents.iter().any(|id| !self.resources.contains_key(id)) {
            return Err(GraphError::ParentAbsent);
        }
        let key = (task, object);
        let slot = if let Some(slot) = self.slots.get_mut(&key) {
            if slot.current.is_some() {
                return Err(GraphError::ReferenceAlreadyBound);
            }
            slot
        } else {
            let index = self.next_resource_index;
            self.next_resource_index = self
                .next_resource_index
                .checked_add(1)
                .ok_or(GraphError::IdentitySpaceExhausted)?;
            self.slots.entry(key).or_insert(NamespaceSlot {
                index,
                next_generation: 1,
                current: None,
                released: None,
            })
        };
        let id = ResourceId::new(slot.index, slot.next_generation);
        slot.next_generation = slot
            .next_generation
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        slot.current = Some(id);
        for parent in &parents {
            self.resources
                .get_mut(parent)
                .expect("parents validated")
                .children
                .insert(id);
        }
        if let Some(storage) = storage {
            self.storage
                .get_mut(&storage)
                .expect("storage validated")
                .owners
                .insert(id);
        }
        self.resources.insert(
            id,
            ResourceNode {
                id,
                task,
                object,
                kind,
                descriptor,
                lifecycle: LifecycleState::Created,
                storage,
                backing_generation: BackingGeneration::new(1),
                content: storage
                    .and_then(|storage| self.storage.get(&storage))
                    .map(|storage| storage.content.clone())
                    .unwrap_or_default(),
                parents,
                children: BTreeSet::new(),
                in_flight: BTreeSet::new(),
            },
        );
        Ok(id)
    }

    pub fn resolve(
        &self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
    ) -> Option<AnyResourceId> {
        self.slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
    }

    /// What this device knows about one object-table slot.
    ///
    /// [`Self::resolve`] collapses two states a caller usually has to tell
    /// apart. A slot with no record has never been named by a declaration this
    /// device admitted, so a later packet can still bind it. A slot whose
    /// object has been released is a name the guest has already finished with,
    /// and the resource a reference to it wanted no longer exists -- a wait on
    /// that is a wait on nothing, because the next declaration for the name
    /// creates a *different* object under a new generation.
    pub fn slot_state(
        &self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
    ) -> ObjectSlotState {
        match self.slots.get(&(task, object)) {
            None => ObjectSlotState::Undeclared,
            Some(slot) => match slot.current {
                Some(resource) => ObjectSlotState::Bound(resource),
                None => ObjectSlotState::Released(slot.released),
            },
        }
    }

    pub fn resource(&self, id: AnyResourceId) -> Option<&ResourceNode> {
        self.resources.get(&id)
    }

    pub fn resource_mut(&mut self, id: AnyResourceId) -> Option<&mut ResourceNode> {
        self.resources.get_mut(&id)
    }

    pub fn publish_resource_descriptor(
        &mut self,
        id: AnyResourceId,
        descriptor: Arc<ResourceDescriptor>,
    ) -> Result<(), GraphError> {
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        match node.descriptor.as_ref() {
            Some(existing) if existing.as_ref() == descriptor.as_ref() => Ok(()),
            Some(_) => Err(GraphError::DescriptorConflict),
            None => {
                node.descriptor = Some(descriptor);
                Ok(())
            }
        }
    }

    /// Resolve a texture-view chain entirely from retained semantic
    /// descriptors and generational parent edges. No task-local name is read
    /// after construction, so slot reuse cannot redirect a live view.
    pub fn resolve_texture_view(
        &self,
        view: AnyResourceId,
    ) -> Result<ResolvedTextureView, TextureViewResolveError> {
        let mut current = view;
        let mut range: Option<ResolvedTextureViewRange> = None;
        let mut texture_type = None;
        let mut pixel_format = None;
        let mut swizzle = None;

        for _ in 0..MAX_TEXTURE_VIEW_CHAIN {
            let node = self
                .resources
                .get(&current)
                .ok_or(TextureViewResolveError::ResourceAbsent(current))?;
            if node.kind == ObjectKind::IOSurfacePlaneView {
                let descriptor = node
                    .descriptor
                    .as_deref()
                    .ok_or(TextureViewResolveError::DescriptorAbsent(current))?;
                let ResourceDescriptor::IOSurfacePlaneView(descriptor) = descriptor else {
                    return Err(TextureViewResolveError::DescriptorKindMismatch(current));
                };
                let plane = descriptor
                    .view
                    .ok_or(TextureViewResolveError::PlaneGeometryAbsent(current))?;
                let plane_range = ResolvedTextureViewRange {
                    level_base: 0,
                    level_count: 1,
                    slice_base: 0,
                    slice_count: 1,
                };
                range = Some(match range {
                    Some(outer) => outer.compose_over(plane_range)?,
                    None => plane_range,
                });
                texture_type.get_or_insert(TextureType::D2);
                pixel_format.get_or_insert(plane.pixel_format);
                let mut parents = node.parents.iter().copied();
                let Some(parent) = parents.next() else {
                    return Err(TextureViewResolveError::ParentCount(current));
                };
                if parents.next().is_some() {
                    return Err(TextureViewResolveError::ParentCount(current));
                }
                return Ok(ResolvedTextureView {
                    view,
                    base: parent,
                    range,
                    texture_type,
                    pixel_format,
                    swizzle,
                });
            }
            if node.kind != ObjectKind::TextureView {
                if current == view {
                    return Err(TextureViewResolveError::NotTextureView(view));
                }
                return Ok(ResolvedTextureView {
                    view,
                    base: current,
                    range,
                    texture_type,
                    pixel_format,
                    swizzle,
                });
            }
            let descriptor = node
                .descriptor
                .as_deref()
                .ok_or(TextureViewResolveError::DescriptorAbsent(current))?;
            // The wire registers several things under the texture-view family
            // that are not view hops: a texture placed over a buffer's storage
            // and one placed in a heap are base textures, declaring their own
            // geometry and composing over no parent. What makes a node a hop
            // is the view descriptor it retains, not the family it was
            // registered under --- reading the family instead refuses a
            // buffer-backed texture the guest bound directly, and the refusal
            // sits on a submission head.
            let ResourceDescriptor::TextureView(descriptor) = descriptor else {
                return Ok(ResolvedTextureView {
                    view,
                    base: current,
                    range,
                    texture_type,
                    pixel_format,
                    swizzle,
                });
            };
            let hop_range = match descriptor.form {
                TextureViewForm::Simple => None,
                TextureViewForm::Ranged | TextureViewForm::Swizzled => {
                    if descriptor.level_count == 0 || descriptor.slice_count == 0 {
                        return Err(TextureViewResolveError::EmptyRange(current));
                    }
                    Some(ResolvedTextureViewRange {
                        level_base: descriptor.level_base,
                        level_count: descriptor.level_count,
                        slice_base: descriptor.slice_base,
                        slice_count: descriptor.slice_count,
                    })
                }
            };
            range = match (range, hop_range) {
                (Some(outer), Some(inner)) => Some(outer.compose_over(inner)?),
                (outer @ Some(_), None) => outer,
                (None, inner) => inner,
            };
            if texture_type.is_none() && descriptor.carries_range() {
                let raw = descriptor.texture_type;
                let narrowed = u8::try_from(raw)
                    .map_err(|_| TextureViewResolveError::UnknownTextureType(raw))?;
                let decoded = TextureType::from_raw(narrowed);
                if let TextureType::Unknown(_) = decoded {
                    return Err(TextureViewResolveError::UnknownTextureType(raw));
                }
                texture_type = Some(decoded);
            }
            if pixel_format.is_none() {
                pixel_format = descriptor.declared_pixel_format();
            }
            let hop_swizzle = if descriptor.carries_swizzle() {
                Some(
                    reims_vgpu_protocol::swizzle_plan(&descriptor.swizzle)
                        .ok_or(TextureViewResolveError::InvalidSwizzle(current))?,
                )
            } else {
                None
            };
            swizzle = match (swizzle, hop_swizzle) {
                (Some(outer), Some(inner)) => Some(outer.after(&inner)),
                (outer @ Some(_), None) => outer,
                (None, inner) => inner,
            };
            let mut parents = node.parents.iter().copied();
            let Some(parent) = parents.next() else {
                return Err(TextureViewResolveError::ParentCount(current));
            };
            if parents.next().is_some() {
                return Err(TextureViewResolveError::ParentCount(current));
            }
            current = parent;
        }
        let node = self
            .resources
            .get(&current)
            .ok_or(TextureViewResolveError::ResourceAbsent(current))?;
        if node.kind == ObjectKind::TextureView {
            Err(TextureViewResolveError::ChainOverflow(current))
        } else {
            Ok(ResolvedTextureView {
                view,
                base: current,
                range,
                texture_type,
                pixel_format,
                swizzle,
            })
        }
    }

    /// Resolve the complete shader-visible image view for either a base
    /// texture or a generational texture-view resource.
    pub fn resolve_texture_binding_view(
        &self,
        resource: AnyResourceId,
    ) -> Result<ResolvedTextureBindingView, TextureViewResolveError> {
        let resource_node = self
            .resources
            .get(&resource)
            .ok_or(TextureViewResolveError::ResourceAbsent(resource))?;
        let resolved = if matches!(
            resource_node.kind,
            ObjectKind::TextureView | ObjectKind::IOSurfacePlaneView
        ) {
            self.resolve_texture_view(resource)?
        } else {
            ResolvedTextureView {
                view: resource,
                base: resource,
                range: None,
                texture_type: None,
                pixel_format: None,
                swizzle: None,
            }
        };
        let base = self
            .resources
            .get(&resolved.base)
            .ok_or(TextureViewResolveError::ResourceAbsent(resolved.base))?;
        let descriptor = base
            .descriptor
            .as_deref()
            .ok_or(TextureViewResolveError::BaseDescriptorAbsent(resolved.base))?;
        let (base_type, mip_levels, array_layers, base_pixel_format) =
            match descriptor {
                ResourceDescriptor::Texture(descriptor) => {
                    let declaration = descriptor.declaration.ok_or(
                        TextureViewResolveError::BaseDeclarationAbsent(resolved.base),
                    )?;
                    texture_declaration_view_facts(declaration, resolved.base)?
                }
                ResourceDescriptor::HeapTexture(descriptor) => {
                    texture_declaration_view_facts(descriptor.declaration, resolved.base)?
                }
                ResourceDescriptor::BufferTexture(descriptor) => {
                    texture_declaration_view_facts(descriptor.desc, resolved.base)?
                }
                ResourceDescriptor::MapperIOSurfaceTextureView(descriptor) => {
                    texture_declaration_view_facts(descriptor.declaration, resolved.base)?
                }
                ResourceDescriptor::SurfaceBacking(_)
                    if resolved.view != resolved.base
                        && resolved.texture_type == Some(TextureType::D2)
                        && resolved.range
                            == Some(ResolvedTextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 0,
                                slice_count: 1,
                            }) =>
                {
                    (
                        TextureType::D2,
                        1,
                        1,
                        resolved.pixel_format.ok_or(
                            TextureViewResolveError::BaseDeclarationAbsent(resolved.base),
                        )?,
                    )
                }
                _ => {
                    return Err(TextureViewResolveError::UnsupportedBaseDescriptor(
                        resolved.base,
                    ));
                }
            };
        if mip_levels == 0 || array_layers == 0 {
            return Err(TextureViewResolveError::EmptyBaseGeometry(resolved.base));
        }
        let range = resolved.range.unwrap_or(ResolvedTextureViewRange {
            level_base: 0,
            level_count: mip_levels,
            slice_base: 0,
            slice_count: array_layers,
        });
        let level_end = range
            .level_base
            .checked_add(range.level_count)
            .ok_or(TextureViewResolveError::LevelOverflow)?;
        let slice_end = range
            .slice_base
            .checked_add(range.slice_count)
            .ok_or(TextureViewResolveError::SliceOverflow)?;
        if level_end > mip_levels || slice_end > array_layers {
            return Err(TextureViewResolveError::ViewRangeOutsideBase(resource));
        }
        let texture_type = resolved.texture_type.unwrap_or(base_type);
        if texture_type == TextureType::Buffer {
            return Err(TextureViewResolveError::UnsupportedTextureType(
                texture_type,
            ));
        }
        Ok(ResolvedTextureBindingView {
            resource,
            base: resolved.base,
            // Storage ownership, not the view chain: a plane view owns its
            // plane's image while its parent surface owns the allocation.
            image_owner: self.image_owner(resource).unwrap_or(resolved.base),
            range,
            texture_type,
            pixel_format: resolved.pixel_format.unwrap_or(base_pixel_format),
            swizzle: resolved
                .swizzle
                .unwrap_or_else(reims_vgpu_protocol::swizzle_identity),
        })
    }

    /// Resolve every live shader-visible view that belongs on the image
    /// `owner` owns. The returned identities are stable even when a task-local
    /// object slot is subsequently reused.
    ///
    /// Ownership, not the view chain. A view resolves its `base` by walking
    /// parents to the end of the chain, and that terminus is not always the
    /// resource holding the image: a view of an IOSurface plane resolves its
    /// base to the *surface*, which owns the allocation and no image at all,
    /// while the plane owns both the storage and the image the view has to be
    /// installed onto. Selecting by base drops exactly those views --- the
    /// plane materializes carrying none of them, no later pass installs them
    /// because the image now exists and nothing revisits declared views, and
    /// every draw that binds one refuses with no view present and parks its
    /// channel for the life of the device.
    pub fn texture_binding_views_for_image_owner(
        &self,
        owner: AnyResourceId,
    ) -> Result<Box<[ResolvedTextureBindingView]>, TextureViewResolveError> {
        self.resources
            .get(&owner)
            .ok_or(TextureViewResolveError::ResourceAbsent(owner))?;
        let mut pending = self
            .resources
            .get(&owner)
            .expect("owner presence was validated")
            .children
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let mut visited = BTreeSet::new();
        let mut views = Vec::new();
        while let Some(resource) = pending.pop() {
            if !visited.insert(resource) {
                continue;
            }
            let node = self
                .resources
                .get(&resource)
                .ok_or(TextureViewResolveError::ResourceAbsent(resource))?;
            pending.extend(node.children.iter().copied());
            if !matches!(
                node.kind,
                ObjectKind::TextureView | ObjectKind::IOSurfacePlaneView
            ) || node.lifecycle == LifecycleState::Released
            {
                continue;
            }
            let view = self.resolve_texture_binding_view(resource)?;
            if view.image_owner == owner {
                views.push(view);
            }
        }
        views.sort_by_key(|view| view.resource);
        Ok(views.into_boxed_slice())
    }
    pub fn resources_for_task(&self, task: TaskId) -> Box<[AnyResourceId]> {
        self.resources
            .values()
            .filter_map(|resource| (resource.task == task).then_some(resource.id))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn live_resources_for_task(&self, task: TaskId) -> Box<[AnyResourceId]> {
        let mut resources = self
            .slots
            .iter()
            .filter_map(|(&(owner, _), slot)| (owner == task).then_some(slot.current).flatten())
            .collect::<Vec<_>>();
        resources.sort();
        resources.into_boxed_slice()
    }

    /// The exact live child closure rooted at one resource, ordered with every
    /// descendant before its parent so lifecycle release cannot leave a child
    /// retaining a root that the command declared dead.
    pub fn live_resource_tree_child_first(
        &self,
        root: AnyResourceId,
    ) -> Option<Box<[AnyResourceId]>> {
        self.resources.get(&root)?;
        let mut pending = vec![(root, false)];
        let mut visited = BTreeSet::new();
        let mut ordered = Vec::new();
        while let Some((resource, expanded)) = pending.pop() {
            let node = self.resources.get(&resource)?;
            if expanded {
                ordered.push(resource);
                continue;
            }
            if !visited.insert(resource) {
                continue;
            }
            pending.push((resource, true));
            pending.extend(node.children.iter().rev().map(|child| (*child, false)));
        }
        Some(ordered.into_boxed_slice())
    }

    /// Resolve every live generational resource backed by a task-address span
    /// intersecting the decoded half-open range. Alias storage is expanded to
    /// its exact owners and the result is stable and duplicate-free.
    pub fn live_resources_overlapping_task_range(
        &self,
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    ) -> Option<Box<[AnyResourceId]>> {
        let end = address.get().checked_add(length.get())?;
        if address.get() == end {
            return Some(Box::new([]));
        }
        let mut resources = self
            .storage_overlapping(task, address.get(), end)
            .filter_map(|backing| self.storage.get(&backing))
            .flat_map(|storage| storage.owners.iter().copied())
            .filter(|resource| {
                self.resources
                    .get(resource)
                    .is_some_and(|node| node.lifecycle != LifecycleState::Released)
            })
            .collect::<Vec<_>>();
        resources.sort_unstable();
        resources.dedup();
        Some(resources.into_boxed_slice())
    }

    pub fn create_storage(&mut self, backing: StorageBacking) -> Result<BackingId, GraphError> {
        self.create_storage_with_regions(backing, [BackingRegion::Whole])
    }

    /// Create one canonical backing with its complete declared coordinate
    /// coverage. Replacement lifecycle decoding supplies exact linear or image
    /// regions when the contract establishes their translation; callers use
    /// [`BackingRegion::Whole`] otherwise.
    pub fn create_storage_with_regions(
        &mut self,
        backing: StorageBacking,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<BackingId, GraphError> {
        let id = BackingId::new(self.next_storage_id);
        let content = ContentAuthority::for_backing_regions(id, regions)?;
        self.next_storage_id = self
            .next_storage_id
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        self.storage.insert(
            id,
            StorageNode {
                id,
                backing,
                owners: BTreeSet::new(),
                content,
            },
        );
        if let StorageBacking::TaskAddress {
            task,
            address,
            length,
        } = self.storage[&id].backing
        {
            let start = address.get();
            let end = start.saturating_add(length.get());
            if start < end {
                self.task_address_index.insert((task, start, id.get()), end);
                let longest = self.longest_task_extent.entry(task).or_insert(0);
                *longest = (*longest).max(end - start);
            }
        }
        Ok(id)
    }

    /// Every storage node whose guest range intersects `[start, end)` on `task`.
    ///
    /// The bound is [`Self::longest_task_extent`]: a range starting before
    /// `start - longest` cannot reach `start`, so the scan begins there, and a
    /// range starting at or after `end` cannot overlap, so it stops there. What
    /// remains inside that window still has to be tested, because a shorter
    /// range inside it may stop before `start`.
    fn storage_overlapping(
        &self,
        task: TaskId,
        start: u64,
        end: u64,
    ) -> impl Iterator<Item = BackingId> + '_ {
        use std::ops::Bound;
        let longest = self.longest_task_extent.get(&task).copied().unwrap_or(0);
        let low = start.saturating_sub(longest);
        self.task_address_index
            .range((
                Bound::Included((task, low, 0)),
                Bound::Excluded((task, end, 0)),
            ))
            .filter(move |(_, &other_end)| start < other_end)
            .map(|(&(_, _, id), _)| BackingId::new(id))
    }

    pub fn storage(&self, id: BackingId) -> Option<&StorageNode> {
        self.storage.get(&id)
    }

    /// Remove an explicitly destroyed backing after every resource and mapping
    /// lifetime which names it has ended. Native ownership remains outside the
    /// graph and uses the returned canonical identity to begin timeline-gated
    /// retirement.
    pub fn retire_storage(&mut self, id: BackingId) -> Result<StorageNode, GraphError> {
        let storage = self.storage.get(&id).ok_or(GraphError::StorageAbsent)?;
        if !storage.owners.is_empty()
            || self
                .mappings
                .values()
                .any(|mapping| mapping.storage == Some(id))
        {
            return Err(GraphError::StorageInUse);
        }
        Ok(self.remove_storage_node(id))
    }

    /// Every distinct backing owned by a released resource subtree, in
    /// canonical order.
    ///
    /// Derived rather than passed in: a caller that assembled its own set could
    /// miss a plane, and a tree release that retires all but one backing leaks
    /// it with nothing to say so.
    pub fn resource_tree_backings(&self, resources: &[AnyResourceId]) -> Box<[BackingId]> {
        resources
            .iter()
            .filter_map(|resource| self.resources.get(resource)?.storage)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn validate_storage_retirement_after_resources(
        &self,
        id: BackingId,
        resources: &[AnyResourceId],
    ) -> Result<(), GraphError> {
        let storage = self.storage.get(&id).ok_or(GraphError::StorageAbsent)?;
        let resources = resources.iter().copied().collect::<BTreeSet<_>>();
        if storage
            .owners
            .iter()
            .any(|owner| !resources.contains(owner))
            || self
                .mappings
                .values()
                .any(|mapping| mapping.storage == Some(id))
        {
            return Err(GraphError::StorageInUse);
        }
        Ok(())
    }

    /// Drain storage whose contract lifetime ended automatically with its last
    /// heap allocation. The queue is unbounded and lossless; callers consume it
    /// after any graph mutation that can collect a released resource.
    pub fn take_automatically_retired_storage(&mut self) -> Vec<StorageNode> {
        std::mem::take(&mut self.automatically_retired_storage)
    }

    pub fn mapper_storage(
        &mut self,
        mapper: MapperSurfaceRef,
        plane: PlaneIndex,
    ) -> Result<BackingId, GraphError> {
        if let Some(id) = self.storage.values().find_map(|storage| {
            (storage.backing == StorageBacking::MapperSurface { mapper, plane })
                .then_some(storage.id)
        }) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::MapperSurface { mapper, plane })
    }

    pub fn find_mapper_plane_storage(
        &self,
        mapper: MapperSurfaceRef,
        plane: PlaneIndex,
    ) -> Option<BackingId> {
        self.storage.values().find_map(|storage| {
            (storage.backing == StorageBacking::MapperSurface { mapper, plane })
                .then_some(storage.id)
        })
    }

    /// Resolve every already established plane backing for one mapper-service
    /// surface identity without creating storage.
    pub fn find_mapper_storage(&self, mapper: MapperSurfaceRef) -> Box<[BackingId]> {
        self.storage
            .values()
            .filter_map(|storage| match storage.backing {
                StorageBacking::MapperSurface { mapper: found, .. } if found == mapper => {
                    Some(storage.id)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// One canonical backing per declared plane of an IOSurface.
    ///
    /// A multi-plane surface is one guest allocation carrying several textures
    /// at declared offsets with their own geometry, row pitch and pixel format,
    /// so each plane is its own backing rather than a view aliasing the whole
    /// allocation. Sharing one backing across planes leaves the surface with
    /// several plane views and no single layout, which is exactly the shape a
    /// biplanar guest surface arrives in.
    pub fn io_surface_plane_storage(
        &mut self,
        surface: AnyResourceId,
        plane: PlaneIndex,
    ) -> Result<BackingId, GraphError> {
        if let Some(id) = self.find_io_surface_plane_storage(surface, plane) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::IOSurfacePlane { surface, plane })
    }

    /// Resolve an already established plane backing without creating storage
    /// from a surface name and plane index alone.
    pub fn find_io_surface_plane_storage(
        &self,
        surface: AnyResourceId,
        plane: PlaneIndex,
    ) -> Option<BackingId> {
        self.storage.values().find_map(|storage| {
            (storage.backing == StorageBacking::IOSurfacePlane { surface, plane })
                .then_some(storage.id)
        })
    }

    pub fn task_address_storage(
        &mut self,
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    ) -> Result<BackingId, GraphError> {
        if let Some(id) = self.find_task_address_storage(task, address, length) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::TaskAddress {
            task,
            address,
            length,
        })
    }

    pub fn find_task_address_storage(
        &self,
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    ) -> Option<BackingId> {
        self.storage.values().find_map(|storage| {
            (storage.backing
                == StorageBacking::TaskAddress {
                    task,
                    address,
                    length,
                })
            .then_some(storage.id)
        })
    }

    pub fn find_buffer_range_storage(
        &self,
        buffer: AnyResourceId,
        offset: ByteOffset,
        bytes_per_row: ByteLength,
    ) -> Option<BackingId> {
        self.storage.values().find_map(|storage| {
            (storage.backing
                == StorageBacking::BufferRange {
                    buffer,
                    offset,
                    bytes_per_row,
                })
            .then_some(storage.id)
        })
    }

    pub fn link_parent(
        &mut self,
        child: AnyResourceId,
        parent: AnyResourceId,
    ) -> Result<(), GraphError> {
        let parent_storage = self
            .resources
            .get(&parent)
            .ok_or(GraphError::ParentAbsent)?
            .storage;
        let child_storage = self
            .resources
            .get(&child)
            .ok_or(GraphError::ResourceAbsent)?
            .storage;
        if self.parent_edge_would_cycle(parent, child) {
            return Err(GraphError::ParentCycle);
        }
        if matches!((child_storage, parent_storage), (Some(child), Some(parent)) if child != parent)
        {
            return Err(GraphError::StorageConflict);
        }
        self.resources
            .get_mut(&parent)
            .unwrap()
            .children
            .insert(child);
        let child_node = self.resources.get_mut(&child).unwrap();
        child_node.parents.insert(parent);
        if child_node.storage.is_none() {
            if let Some(storage) = parent_storage {
                child_node.storage = Some(storage);
                let storage_node = self
                    .storage
                    .get_mut(&storage)
                    .ok_or(GraphError::StorageAbsent)?;
                child_node.content = storage_node.content.clone();
                storage_node.owners.insert(child);
            }
        }
        Ok(())
    }

    pub fn link_buffer_range(
        &mut self,
        child: AnyResourceId,
        buffer: AnyResourceId,
        offset: ByteOffset,
        bytes_per_row: ByteLength,
    ) -> Result<(), GraphError> {
        if !self.resources.contains_key(&buffer) {
            return Err(GraphError::ParentAbsent);
        }
        let child_storage = self
            .resources
            .get(&child)
            .ok_or(GraphError::ResourceAbsent)?
            .storage;
        if self.parent_edge_would_cycle(buffer, child) {
            return Err(GraphError::ParentCycle);
        }
        let backing = StorageBacking::BufferRange {
            buffer,
            offset,
            bytes_per_row,
        };
        let existing = self
            .storage
            .values()
            .find_map(|storage| (storage.backing == backing).then_some(storage.id));
        if child_storage.is_some_and(|storage| Some(storage) != existing) {
            return Err(GraphError::StorageConflict);
        }
        let storage = match existing {
            Some(storage) => storage,
            None => self.create_storage(backing)?,
        };
        self.attach_initial_storage(child, storage)?;
        self.resources
            .get_mut(&buffer)
            .expect("parent validated")
            .children
            .insert(child);
        self.resources
            .get_mut(&child)
            .expect("child validated")
            .parents
            .insert(buffer);
        Ok(())
    }

    /// Attach one texture to storage owned by a generational heap lifetime.
    ///
    /// Equal explicit ranges are aliases and therefore intern one storage and
    /// one content authority. A partial overlap is not representable by the
    /// current whole-image authority and refuses instead of pretending the
    /// images are disjoint. Allocator-owned placements use the texture's own
    /// lifetime and never consult the meaningless serialized offset.
    pub fn link_heap_texture(
        &mut self,
        texture: AnyResourceId,
        heap: ResourceId<HeapObject>,
        explicit: Option<(ByteOffset, ByteLength)>,
    ) -> Result<(), GraphError> {
        if !self.resources.contains_key(&texture) {
            return Err(GraphError::ResourceAbsent);
        }
        let backing = match explicit {
            Some((offset, length)) => {
                let start = offset.get();
                let Some(end) = start.checked_add(length.get()) else {
                    return Err(GraphError::HeapPlacementOverlap);
                };
                if length.get() == 0 {
                    return Err(GraphError::HeapPlacementOverlap);
                }
                for storage in self.storage.values() {
                    let StorageBacking::HeapPlacement {
                        heap: other_heap,
                        offset: other_offset,
                        length: other_length,
                    } = storage.backing
                    else {
                        continue;
                    };
                    if other_heap != heap {
                        continue;
                    }
                    let other_start = other_offset.get();
                    let other_end = other_start.saturating_add(other_length.get());
                    let exact = start == other_start && end == other_end;
                    if !exact && start < other_end && other_start < end {
                        return Err(GraphError::HeapPlacementOverlap);
                    }
                }
                StorageBacking::HeapPlacement {
                    heap,
                    offset,
                    length,
                }
            }
            None => StorageBacking::HeapAllocation {
                heap,
                allocation: texture,
            },
        };
        let existing = self
            .storage
            .values()
            .find_map(|storage| (storage.backing == backing).then_some(storage.id));
        let storage = match existing {
            Some(storage) => storage,
            None => self.create_storage(backing)?,
        };
        self.attach_initial_storage(texture, storage)
    }

    pub fn attach_initial_storage(
        &mut self,
        id: AnyResourceId,
        storage: BackingId,
    ) -> Result<(), GraphError> {
        if !self.storage.contains_key(&storage) {
            return Err(GraphError::StorageAbsent);
        }
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if let Some(existing) = node.storage {
            return if existing == storage {
                Ok(())
            } else {
                Err(GraphError::StorageConflict)
            };
        }
        node.storage = Some(storage);
        let storage_node = self.storage.get_mut(&storage).unwrap();
        node.content = storage_node.content.clone();
        storage_node.owners.insert(id);
        Ok(())
    }

    pub fn replace_backing(
        &mut self,
        id: AnyResourceId,
        storage: BackingId,
    ) -> Result<BackingGeneration, GraphError> {
        if !self.storage.contains_key(&storage) {
            return Err(GraphError::StorageAbsent);
        }
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if let Some(old) = node.storage.replace(storage) {
            if let Some(old) = self.storage.get_mut(&old) {
                old.owners.remove(&id);
            }
        }
        self.storage.get_mut(&storage).unwrap().owners.insert(id);
        node.content = self.storage.get(&storage).unwrap().content.clone();
        let next = node
            .backing_generation
            .get()
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        node.backing_generation = BackingGeneration::new(next);
        Ok(node.backing_generation)
    }

    /// Advance the physical incarnation of one resource without changing its
    /// logical backing or content authority.
    ///
    /// The replacement notification names only the task-local resource. Its
    /// storage identity, guest address, geometry, and content remain the same;
    /// page-derived native state must instead be keyed by this generation and
    /// rebuilt after it advances.
    pub fn replace_physical(
        &mut self,
        id: AnyResourceId,
    ) -> Result<(Option<BackingId>, BackingGeneration), GraphError> {
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        let next = node
            .backing_generation
            .get()
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        node.backing_generation = BackingGeneration::new(next);
        Ok((node.storage, node.backing_generation))
    }

    /// Resolve one object-table reference and enter the resource it names into
    /// a submission, reporting the content version that submission must expect.
    ///
    /// # One transition, because it was never two
    ///
    /// This replaced a `resolve`/`resource`/`prepare`/`submit` quartet whose
    /// last three took the **same key**, so opening a submission cost four
    /// `BTreeMap` descents per resource descriptor where the work asks for two.
    /// `prepare` and `submit` had exactly one caller each, on adjacent lines,
    /// and `submit`'s `SubmissionNotPrepared` guard re-checked a set membership
    /// `prepare` had inserted one line earlier — a condition no caller could
    /// make false. Splitting them bought a state nobody could observe: both ran
    /// under one lock, so the intermediate was never visible outside this
    /// function, and the `Prepared` lifecycle it assigned was overwritten before
    /// the borrow ended and was read nowhere in the tree.
    ///
    /// The measurement that prompted it: `ExecPhase::Open` was 147 ms/s of a
    /// drain worker running at 0.95 duty on driven fullscreen Maps — 259 us of
    /// CPU to open one submission, and the second largest cost in this device
    /// after the draw encode itself.
    ///
    /// A resolved slot always names a live resource — the slot table and the
    /// resource table are written together under this same lock — which is what
    /// the old `expect` on `prepare` asserted and what this one asserts.
    pub fn enter_submission(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        submission: SubmissionId,
    ) -> Option<(AnyResourceId, ContentVersion)> {
        let id = self.reserve_submission(task, object, submission)?;
        let expected = self
            .begin_reserved_submission(id, submission)
            .expect("the just-reserved resource owns this submission");
        Some((id, expected))
    }

    /// Retain the exact resource generation named by an accepted submission.
    ///
    /// Reservation protects object lifetime while scheduler admission is
    /// parked, but deliberately does not snapshot content. A conflicting
    /// predecessor may still advance that content before this submission is
    /// admitted to resolution.
    pub fn reserve_submission(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
        submission: SubmissionId,
    ) -> Option<AnyResourceId> {
        let id = self
            .slots
            .get(&(task, object))
            .and_then(|slot| slot.current)?;
        let node = self
            .resources
            .get_mut(&id)
            .expect("a resolved slot names a live resource");
        node.in_flight.insert(submission);
        if node.lifecycle != LifecycleState::Released {
            node.lifecycle = LifecycleState::InFlight;
        }
        Some(id)
    }

    /// Snapshot content for an exact resource reservation at resolver admission.
    pub fn begin_reserved_submission(
        &self,
        id: AnyResourceId,
        submission: SubmissionId,
    ) -> Result<ContentVersion, GraphError> {
        let node = self.resources.get(&id).ok_or(GraphError::ResourceAbsent)?;
        if !node.in_flight.contains(&submission) {
            return Err(GraphError::SubmissionNotPrepared);
        }
        Ok(node.content.current())
    }

    pub fn complete(
        &mut self,
        id: AnyResourceId,
        submission: SubmissionId,
    ) -> Result<(), GraphError> {
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if !node.in_flight.remove(&submission) {
            return Err(GraphError::SubmissionNotPrepared);
        }
        if node.lifecycle != LifecycleState::Released {
            node.lifecycle = if node.in_flight.is_empty() {
                LifecycleState::Created
            } else {
                LifecycleState::InFlight
            };
        }
        self.collect_if_unowned(id);
        Ok(())
    }

    pub fn release_reference(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
    ) -> Result<AnyResourceId, GraphError> {
        let id = self
            .slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
            .ok_or(GraphError::ReferenceUnbound)?;
        self.release_resource(id)
    }

    /// Release the exact generational resource resolved at transaction
    /// admission. A stale generation cannot delete the replacement occupying
    /// the same task-local object slot.
    /// Release one generation, idempotently.
    ///
    /// Releasing an object that is already released is the guest's statement
    /// satisfied, not a failure. It is reachable whenever something else
    /// emptied the name first -- a rebind, a tree release, a task teardown --
    /// and the guest's own delete for that generation is still admitted and on
    /// its way. Refusing it there costs far more than the delete: the packet
    /// carrying it is refused at apply, its completion stamp is never
    /// published, and the guest waits on that stamp for the life of the boot.
    /// A driven macos-13 conformance boot died exactly that way, on
    /// `ReleaseResource { resource: ResourceId { index: 18, generation: 2 } }`
    /// against a name a rebind had emptied one millisecond earlier.
    ///
    /// Generations are what make this safe to wave through. The delete names
    /// the generation it was admitted against, so it can never reach whatever
    /// the name holds now.
    pub fn release_resource(&mut self, id: AnyResourceId) -> Result<AnyResourceId, GraphError> {
        let Some(node) = self.resources.get(&id) else {
            return Ok(id);
        };
        let key = (node.task, node.object);
        if let Some(slot) = self.slots.get_mut(&key) {
            if slot.current == Some(id) {
                slot.current = None;
                slot.released = Some(id);
            }
        }
        self.resources
            .get_mut(&id)
            .expect("the node was just found in the canonical graph")
            .lifecycle = LifecycleState::Released;
        self.collect_if_unowned(id);
        Ok(id)
    }

    pub fn release_task(&mut self, task: TaskId) -> usize {
        let refs: Vec<_> = self
            .slots
            .iter()
            .filter_map(|(&(owner, object), slot)| {
                (owner == task && slot.current.is_some()).then_some(object)
            })
            .collect();
        for object in &refs {
            let _ = self.release_reference(task, *object);
        }
        refs.len()
    }

    pub fn create_mapping(&mut self, mapping: MappingNode) -> Result<(), GraphError> {
        if mapping
            .storage
            .is_some_and(|id| !self.storage.contains_key(&id))
        {
            return Err(GraphError::StorageAbsent);
        }
        if self.mappings.insert(mapping.id, mapping).is_some() {
            return Err(GraphError::MappingAlreadyExists);
        }
        Ok(())
    }

    pub fn release_mapping(&mut self, id: MappingId) -> Result<MappingNode, GraphError> {
        self.mappings.remove(&id).ok_or(GraphError::MappingAbsent)
    }

    pub fn mapping(&self, id: MappingId) -> Option<&MappingNode> {
        self.mappings.get(&id)
    }

    fn collect_if_unowned(&mut self, id: AnyResourceId) {
        let collect = self.resources.get(&id).is_some_and(|node| {
            node.lifecycle == LifecycleState::Released
                && node.in_flight.is_empty()
                && node.children.is_empty()
        });
        if !collect {
            return;
        }
        let node = self
            .resources
            .remove(&id)
            .expect("collection candidate exists");
        if let Some(storage_id) = node.storage {
            let retire_heap_storage = self.storage.get_mut(&storage_id).is_some_and(|storage| {
                storage.owners.remove(&id);
                storage.owners.is_empty()
                    && matches!(
                        storage.backing,
                        StorageBacking::HeapPlacement { .. }
                            | StorageBacking::HeapAllocation { .. }
                    )
            });
            if retire_heap_storage {
                let retired = self.remove_storage_node(storage_id);
                self.automatically_retired_storage.push(retired);
            }
        }
        for parent in node.parents {
            if let Some(parent_node) = self.resources.get_mut(&parent) {
                parent_node.children.remove(&id);
            }
            self.collect_if_unowned(parent);
        }
    }

    fn remove_storage_node(&mut self, id: BackingId) -> StorageNode {
        let storage = self
            .storage
            .remove(&id)
            .expect("storage removal follows an exact presence check");
        if let StorageBacking::TaskAddress {
            task,
            address,
            length,
        } = storage.backing
        {
            let start = address.get();
            let end = start.saturating_add(length.get());
            self.task_address_index.remove(&(task, start, id.get()));
            debug_assert!(
                start >= end
                    || !self
                        .task_address_index
                        .contains_key(&(task, start, id.get()))
            );
        }
        storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> TaskId {
        TaskId::new(3)
    }

    fn object(value: u32) -> ObjectTableRef<ResourceObject> {
        ObjectTableRef::new(value)
    }

    /// A name nobody has declared and a name whose object is gone are two
    /// different answers, and [`ResourceGraph::resolve`] gives them the same
    /// one.
    ///
    /// The difference decides whether a reference to the slot is waiting for a
    /// packet that can still arrive or referring to an object that no longer
    /// exists, so a refusal that cannot state it cannot be acted on.
    #[test]
    fn an_undeclared_slot_and_a_released_one_are_not_the_same_answer() {
        let mut graph = ResourceGraph::default();

        assert_eq!(
            graph.slot_state(task(), object(47)),
            ObjectSlotState::Undeclared
        );

        let resource = graph
            .create_resource(task(), object(47), ObjectKind::Buffer, None, [])
            .unwrap();
        assert_eq!(
            graph.slot_state(task(), object(47)),
            ObjectSlotState::Bound(resource)
        );
        assert_eq!(graph.resolve(task(), object(47)), Some(resource));

        graph.release_resource(resource).unwrap();
        assert_eq!(
            graph.slot_state(task(), object(47)),
            ObjectSlotState::Released(Some(resource)),
            "an emptied name says which generation emptied it"
        );
        assert_eq!(graph.resolve(task(), object(47)), None);

        // The name stays reusable, and the reuse is a different object.
        let reused = graph
            .create_resource(task(), object(47), ObjectKind::Buffer, None, [])
            .unwrap();
        assert_ne!(reused, resource);
        assert_eq!(
            graph.slot_state(task(), object(47)),
            ObjectSlotState::Bound(reused)
        );

        // A different task's identically-numbered name is still its own.
        assert_eq!(
            graph.slot_state(TaskId::new(9), object(47)),
            ObjectSlotState::Undeclared
        );
    }

    /// A delete the guest already sent for a generation something else has
    /// already released must complete, not refuse.
    ///
    /// The two happen together whenever a name is rebound: the declaration that
    /// rebinds releases the live generation, and the guest's own delete for
    /// that generation is still admitted and on its way. Refusing it costs far
    /// more than the delete -- the packet carrying it is refused at apply, its
    /// completion stamp is never published, and the guest waits on that stamp
    /// for the rest of the boot with nothing in any census to say why.
    #[test]
    fn releasing_a_generation_something_else_already_released_completes() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(43), ObjectKind::IOSurfacePlaneView, None, [])
            .unwrap();

        // The rebind: the live generation goes and the name takes a new one.
        graph.release_resource(first).unwrap();
        let second = graph
            .create_resource(task(), object(43), ObjectKind::Buffer, None, [])
            .unwrap();
        assert_ne!(second, first);

        // The guest's delete, admitted against the generation it resolved.
        assert_eq!(graph.release_resource(first), Ok(first));
        assert_eq!(
            graph.slot_state(task(), object(43)),
            ObjectSlotState::Bound(second),
            "a delete names the generation it was admitted against and cannot \
             reach whatever the name holds now"
        );

        // And releasing the live one still empties the name, once.
        graph.release_resource(second).unwrap();
        assert_eq!(
            graph.slot_state(task(), object(43)),
            ObjectSlotState::Released(Some(second))
        );
        assert_eq!(graph.release_resource(second), Ok(second));
        assert_eq!(
            graph.slot_state(task(), object(43)),
            ObjectSlotState::Released(Some(second))
        );
    }

    /// The index must return exactly what walking every storage node returned.
    ///
    /// This replaced a scan of the whole storage map with a bounded range scan,
    /// and a bound that is too tight loses an alias silently: the resource that
    /// shares those guest bytes keeps a content version the guest has already
    /// invalidated, which is stale pixels and not a crash. So the property is
    /// checked against the predicate it replaced rather than against a handful
    /// of cases someone thought of — the shapes that break a range bound are
    /// exactly the ones that are easy not to think of. Ranges are drawn from a
    /// 4 KiB address space with lengths up to a quarter of it, so nodes overlap
    /// each other densely and the long-range-reaching-forward shape the bound
    /// exists for is generated rather than hoped for.
    #[test]
    fn the_address_index_agrees_with_walking_every_storage_node() {
        let mut graph = ResourceGraph::default();
        // A fixed sequence, so a failure is reproducible: xorshift, seeded.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let tasks = [TaskId::new(1), TaskId::new(2)];
        for _ in 0..400 {
            let task = tasks[(next() % 2) as usize];
            let address = next() % 4096;
            // Zero lengths are legal to construct and are deliberately included:
            // they must be indexed by neither arm.
            let length = next() % 1024;
            graph
                .create_storage(StorageBacking::TaskAddress {
                    task,
                    address: GuestVirtualAddress::new(address),
                    length: ByteLength::new(length),
                })
                .unwrap();
        }
        // Backings the index does not hold at all, to prove it excludes them.
        for plane in 0..8 {
            graph
                .mapper_storage(MapperSurfaceRef::new(plane), PlaneIndex::new(0))
                .unwrap();
        }

        let brute = |task: TaskId, start: u64, end: u64| -> BTreeSet<BackingId> {
            graph
                .storage
                .values()
                .filter_map(|candidate| {
                    let StorageBacking::TaskAddress {
                        task: other_task,
                        address: other_address,
                        length: other_length,
                    } = candidate.backing
                    else {
                        return None;
                    };
                    let other_start = other_address.get();
                    let other_end = other_start.saturating_add(other_length.get());
                    (task == other_task && start < other_end && other_start < end)
                        .then_some(candidate.id)
                })
                .collect()
        };

        let mut queries = 0;
        let mut nonempty = 0;
        for _ in 0..2000 {
            let task = tasks[(next() % 2) as usize];
            let start = next() % 4096;
            let end = start.saturating_add(next() % 1024);
            if start >= end {
                continue;
            }
            queries += 1;
            let indexed: BTreeSet<BackingId> =
                graph.storage_overlapping(task, start, end).collect();
            let expected = brute(task, start, end);
            assert_eq!(
                indexed, expected,
                "task={task:?} range=[{start}, {end}) disagreed"
            );
            nonempty += usize::from(!expected.is_empty());
        }
        // A test that only ever compared two empty sets would pass while the
        // index returned nothing at all, so the coverage is asserted too.
        assert!(queries > 1000, "too few usable queries: {queries}");
        assert!(nonempty > 500, "too few overlapping queries: {nonempty}");
    }

    /// A range that begins before the query and reaches into it must be found.
    ///
    /// The bound is `start - longest_task_extent`, so this is the shape that a
    /// too-tight bound drops, and it is worth a case of its own that names it.
    #[test]
    fn an_overlap_beginning_far_before_the_query_is_still_found() {
        let mut graph = ResourceGraph::default();
        let long = graph
            .create_storage(StorageBacking::TaskAddress {
                task: task(),
                address: GuestVirtualAddress::new(0),
                length: ByteLength::new(10_000),
            })
            .unwrap();
        let short = graph
            .create_storage(StorageBacking::TaskAddress {
                task: task(),
                address: GuestVirtualAddress::new(9_990),
                length: ByteLength::new(10),
            })
            .unwrap();
        // Starts 9 990 bytes before the query and is the only reason the bound
        // has to reach back at all.
        let found: BTreeSet<BackingId> = graph.storage_overlapping(task(), 9_995, 9_999).collect();
        assert_eq!(found, BTreeSet::from([long, short]));

        // Another task's identical range is not this task's alias.
        let found: BTreeSet<BackingId> = graph
            .storage_overlapping(TaskId::new(99), 9_995, 9_999)
            .collect();
        assert!(found.is_empty());

        // Touching end-to-start is not an overlap: [0, 10 000) and [10 000, ..).
        let found: BTreeSet<BackingId> =
            graph.storage_overlapping(task(), 10_000, 10_001).collect();
        assert!(found.is_empty(), "half-open ranges must not touch-overlap");
    }

    #[test]
    fn mapper_references_do_not_become_page_table_mapping_identities() {
        let mut graph = ResourceGraph::default();
        let mapper = MapperSurfaceRef::new(12);
        let storage = graph.mapper_storage(mapper, PlaneIndex::new(0)).unwrap();

        assert_eq!(
            graph.storage(storage).unwrap().backing,
            StorageBacking::MapperSurface {
                mapper,
                plane: PlaneIndex::new(0),
            }
        );
        assert!(graph.mapping(MappingId::new(12)).is_none());
    }

    #[test]
    fn delayed_gpu_completion_cannot_overwrite_a_newer_guest_write() {
        let mut content = ContentState::default();
        let planned = content.gpu_store_planned(SubmissionId::new(4)).unwrap();
        let guest = content.guest_wrote().unwrap();

        assert!(guest > planned.version);
        assert_eq!(
            content.gpu_store_completed(SubmissionId::new(4)),
            Ok(planned.version)
        );
        assert_eq!(content.current, guest);
        assert!(content.current_in_guest());
        assert!(!content.current_in_gpu());
        assert_eq!(
            content.copy_gpu_to_guest_completed(planned.version),
            Err(ContentError::StaleSource)
        );
    }

    #[test]
    fn gpu_only_content_is_derived_from_replica_versions() {
        let mut content = ContentState::default();
        let planned = content.gpu_store_planned(SubmissionId::new(9)).unwrap();
        content.gpu_store_completed(SubmissionId::new(9)).unwrap();

        assert_eq!(content.current, planned.version);
        assert!(content.current_in_gpu());
        assert!(!content.current_in_guest());
        content
            .copy_gpu_to_guest_completed(planned.version)
            .unwrap();
        assert!(content.current_in_guest());
    }

    #[test]
    fn a_stamped_materialization_cannot_revive_superseded_content() {
        let mut content = ContentState::default();
        let submitted = content.current;
        let newer = content.guest_wrote().unwrap();

        assert_eq!(
            content.gpu_materialized(submitted),
            Err(ContentError::StaleSource)
        );
        assert_eq!(content.current, newer);
        assert!(!content.current_in_gpu());
        assert_eq!(content.gpu_materialized(newer), Ok(()));
        assert!(content.current_in_gpu());
    }

    #[test]
    fn replacing_the_only_current_replica_is_refused() {
        let mut content = ContentState::default();
        assert_eq!(
            content.replace_guest_backing(),
            Err(ContentError::WouldLoseCurrentContent)
        );

        let current = content.current;
        content.copy_guest_to_gpu_completed(current).unwrap();
        content.replace_guest_backing().unwrap();
        assert!(!content.current_in_guest());
        assert!(content.current_in_gpu());
    }

    #[test]
    fn released_reference_reuse_gets_a_new_generation_and_rejects_stale_ids() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(7), ObjectKind::Buffer, None, [])
            .unwrap();
        graph.release_reference(task(), object(7)).unwrap();
        let second = graph
            .create_resource(task(), object(7), ObjectKind::Buffer, None, [])
            .unwrap();

        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());
        assert!(graph.resource(first).is_none());
        assert_eq!(graph.resolve(task(), object(7)), Some(second));
    }

    #[test]
    fn deleting_a_parent_keeps_it_until_its_child_view_is_released() {
        let mut graph = ResourceGraph::default();
        let parent = graph
            .create_resource(task(), object(1), ObjectKind::Texture, None, [])
            .unwrap();
        let child = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [parent])
            .unwrap();

        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(parent).is_some());
        assert_eq!(graph.resolve(task(), object(2)), Some(child));

        graph.release_reference(task(), object(2)).unwrap();
        assert!(graph.resource(child).is_none());
        assert!(graph.resource(parent).is_none());
    }

    #[test]
    fn delete_while_in_flight_retires_only_after_matching_completion() {
        let mut graph = ResourceGraph::default();
        let id = graph
            .create_resource(task(), object(1), ObjectKind::Texture, None, [])
            .unwrap();
        let submission = SubmissionId::new(9);
        let (entered, _) = graph
            .enter_submission(task(), object(1), submission)
            .expect("the resource this test just created resolves");
        assert_eq!(entered, id);
        assert_eq!(
            graph.resource(id).unwrap().lifecycle,
            LifecycleState::InFlight,
            "entering a submission is one transition and lands in InFlight"
        );
        graph.release_reference(task(), object(1)).unwrap();

        assert_eq!(
            graph.resource(id).unwrap().lifecycle,
            LifecycleState::Released
        );
        graph.complete(id, submission).unwrap();
        assert!(graph.resource(id).is_none());
    }

    #[test]
    fn reservation_retains_identity_but_snapshots_content_only_at_admission() {
        let mut graph = ResourceGraph::default();
        let id = graph
            .create_resource(task(), object(1), ObjectKind::Buffer, None, [])
            .unwrap();
        let submission = SubmissionId::new(10);

        assert_eq!(
            graph.reserve_submission(task(), object(1), submission),
            Some(id)
        );
        let version_after_reservation = graph.guest_wrote_aliases(id).unwrap();
        assert_eq!(
            graph.begin_reserved_submission(id, submission).unwrap(),
            version_after_reservation,
            "content is observed when the reserved submission is admitted"
        );

        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(id).is_some());
        graph.complete(id, submission).unwrap();
        assert!(graph.resource(id).is_none());
    }

    #[test]
    fn backing_replacement_preserves_resource_identity_and_mapping_release_is_independent() {
        let mut graph = ResourceGraph::default();
        let first_storage = graph.create_storage(StorageBacking::Dedicated).unwrap();
        let second_storage = graph.create_storage(StorageBacking::Dedicated).unwrap();
        let id = graph
            .create_resource(
                task(),
                object(1),
                ObjectKind::IOSurfaceTexture,
                Some(first_storage),
                [],
            )
            .unwrap();
        let mapping = MappingId::new(4);
        graph
            .create_mapping(MappingNode {
                id: mapping,
                task: task(),
                address: GuestVirtualAddress::new(0x4000),
                length: ByteLength::new(0x1000),
                storage: Some(first_storage),
                committed: true,
            })
            .unwrap();

        assert_eq!(graph.replace_backing(id, second_storage).unwrap().get(), 2);
        assert_eq!(graph.resource(id).unwrap().id, id);
        graph.release_mapping(mapping).unwrap();
        assert!(graph.resource(id).is_some());
    }

    #[test]
    fn a_registered_surface_owns_no_backing_and_its_planes_own_one_each() {
        let mut graph = ResourceGraph::default();
        let surface = graph
            .create_resource(task(), object(1), ObjectKind::SurfaceBacking, None, [])
            .unwrap();
        let first = graph
            .io_surface_plane_storage(surface, PlaneIndex::new(0))
            .unwrap();
        let second = graph
            .io_surface_plane_storage(surface, PlaneIndex::new(1))
            .unwrap();
        assert_ne!(first, second);
        // Asking twice for one plane is the same backing: the key is the
        // surface's generational identity and the declared plane, so nothing
        // has to be remembered anywhere else.
        assert_eq!(
            graph
                .io_surface_plane_storage(surface, PlaneIndex::new(0))
                .unwrap(),
            first
        );

        let views = [first, second].map(|storage| {
            graph
                .create_resource(
                    task(),
                    object(if storage == first { 2 } else { 3 }),
                    ObjectKind::IOSurfacePlaneView,
                    Some(storage),
                    [surface],
                )
                .unwrap()
        });
        assert_eq!(graph.resource(surface).unwrap().storage, None);
        assert_eq!(graph.resolved_backing(views[0]), Some(first));
        assert_eq!(graph.resolved_backing(views[1]), Some(second));
        // The tree's backings are both planes and nothing else, so a release
        // cannot retire one and leak the other.
        let resources = graph.live_resource_tree_child_first(surface).unwrap();
        assert_eq!(
            graph.resource_tree_backings(&resources).as_ref(),
            [first, second]
        );

        // Each plane view owns the image over its own plane. The view chain's
        // base is the surface, which owns no storage and therefore no image,
        // so an owner read off the chain would send both planes' bindings to
        // one image that was never built.
        for view in views {
            assert_eq!(graph.image_owner(view), Some(view));
        }
        assert_eq!(graph.image_owner(surface), None);

        // A guest write to the allocation reaches both planes, because the
        // alias closure follows the surface's children whether or not they
        // share its authority.
        let before = views.map(|view| graph.resource(view).unwrap().content.current());
        graph.guest_wrote_aliases(surface).unwrap();
        for (view, was) in views.into_iter().zip(before) {
            assert_ne!(graph.resource(view).unwrap().content.current(), was);
        }
    }

    /// A texture placed over a buffer's storage is a base texture, not a view
    /// hop, and binding it directly resolves.
    ///
    /// The wire registers it under the texture-view family, and the chain
    /// walker used to read that family as "this node is a view" and demand a
    /// view descriptor of it. What it retains is its own declaration, so the
    /// walk refused `DescriptorKindMismatch` --- on a compute binding, on a
    /// submission head, for the life of the boot. What makes a node a hop is
    /// the view descriptor it retains.
    #[test]
    fn a_texture_over_a_buffer_is_the_base_of_its_own_binding() {
        let mut graph = ResourceGraph::default();
        let storage = graph
            .create_storage_with_regions(StorageBacking::Dedicated, [BackingRegion::Whole])
            .unwrap();
        let buffer = graph
            .create_resource(task(), object(7), ObjectKind::Buffer, Some(storage), [])
            .unwrap();
        let declaration = reims_vgpu_protocol::TextureDeclaration {
            texture_type: TextureType::D2,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
            pixel_format: 80,
            width: 128,
            height: 64,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        };
        let texture = graph
            .create_resource_with_descriptor(
                task(),
                object(8),
                // The family the wire registers it under.
                ObjectKind::TextureView,
                Some(Arc::new(ResourceDescriptor::BufferTexture(
                    reims_vgpu_protocol::BufferTextureDescriptor {
                        new_texture_ref: 8,
                        buffer_ref: 7,
                        offset: 0,
                        bytes_per_row: 512,
                        desc: declaration,
                    },
                ))),
                Some(storage),
                [buffer],
            )
            .unwrap();

        let resolved = graph.resolve_texture_binding_view(texture).unwrap();
        assert_eq!(resolved.resource, texture);
        // Its own declaration answers for it: the chain ends here rather than
        // composing over the buffer it is placed on.
        assert_eq!(resolved.base, texture);
        assert_eq!(resolved.texture_type, TextureType::D2);
        assert_eq!(resolved.pixel_format, 80);
        assert_eq!(resolved.range.level_count, 1);
        assert_eq!(resolved.range.slice_count, 1);
    }

    #[test]
    fn declared_backing_regions_are_owned_once_and_shared_by_every_view() {
        let mut graph = ResourceGraph::default();
        let left = BackingRegion::Linear(crate::LinearRange::new(0, 64).unwrap());
        let right = BackingRegion::Linear(crate::LinearRange::new(64, 64).unwrap());
        let storage = graph
            .create_storage_with_regions(StorageBacking::Dedicated, [left, right])
            .unwrap();
        let first = graph
            .create_resource(task(), object(70), ObjectKind::Buffer, Some(storage), [])
            .unwrap();
        let second = graph
            .create_resource(
                task(),
                object(71),
                ObjectKind::TextureView,
                Some(storage),
                [first],
            )
            .unwrap();

        let first_authority = graph.resource(first).unwrap().content.clone();
        let second_authority = graph.resource(second).unwrap().content.clone();
        let write = first_authority
            .guest_write_region(None, GUEST_REPRESENTATION, right)
            .unwrap();
        assert!(first_authority.same_authority(&second_authority));
        assert_eq!(
            second_authority.snapshot_regions(&[right]).as_ref(),
            [write]
        );
    }

    #[test]
    fn invalid_region_declaration_does_not_consume_a_backing_identity() {
        let mut graph = ResourceGraph::default();
        assert_eq!(
            graph.create_storage_with_regions(
                StorageBacking::Dedicated,
                Box::<[BackingRegion]>::default(),
            ),
            Err(GraphError::Content(ContentAuthorityError::EmptyBacking))
        );
        assert_eq!(
            graph.create_storage(StorageBacking::Dedicated).unwrap(),
            BackingId::new(1)
        );
    }

    #[test]
    fn resources_over_the_same_task_allocation_share_storage_identity() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x2000),
            )
            .unwrap();
        let second = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x2000),
            )
            .unwrap();
        let other_task = graph
            .task_address_storage(
                TaskId::new(8),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x2000),
            )
            .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, other_task);
    }

    #[test]
    fn a_write_advances_every_overlapping_task_address_view() {
        let mut graph = ResourceGraph::default();
        let buffer_storage = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x4000),
            )
            .unwrap();
        let texture_storage = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x5000),
                ByteLength::new(0x1000),
            )
            .unwrap();
        let disjoint_storage = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x9000),
                ByteLength::new(0x1000),
            )
            .unwrap();
        let other_task_storage = graph
            .task_address_storage(
                TaskId::new(8),
                GuestVirtualAddress::new(0x5000),
                ByteLength::new(0x1000),
            )
            .unwrap();
        let buffer = graph
            .create_resource(
                task(),
                object(1),
                ObjectKind::Buffer,
                Some(buffer_storage),
                [],
            )
            .unwrap();
        let texture = graph
            .create_resource(
                task(),
                object(2),
                ObjectKind::Texture,
                Some(texture_storage),
                [],
            )
            .unwrap();
        let disjoint = graph
            .create_resource(
                task(),
                object(3),
                ObjectKind::Texture,
                Some(disjoint_storage),
                [],
            )
            .unwrap();
        let other_task = graph
            .create_resource(
                TaskId::new(8),
                object(1),
                ObjectKind::Texture,
                Some(other_task_storage),
                [],
            )
            .unwrap();
        let texture_before = graph.resource(texture).unwrap().content.current();
        let disjoint_before = graph.resource(disjoint).unwrap().content.current();
        let other_task_before = graph.resource(other_task).unwrap().content.current();

        graph.guest_wrote_aliases(buffer).unwrap();

        assert_ne!(
            graph.resource(texture).unwrap().content.current(),
            texture_before
        );
        assert_eq!(
            graph.resource(disjoint).unwrap().content.current(),
            disjoint_before
        );
        assert_eq!(
            graph.resource(other_task).unwrap().content.current(),
            other_task_before
        );
    }

    #[test]
    fn a_buffer_texture_owns_a_typed_range_and_retains_its_buffer() {
        let mut graph = ResourceGraph::default();
        let buffer_backing = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x1000),
                ByteLength::new(0x1000),
            )
            .unwrap();
        let buffer = graph
            .create_resource(
                task(),
                object(1),
                ObjectKind::Buffer,
                Some(buffer_backing),
                [],
            )
            .unwrap();
        let texture = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();

        graph
            .link_buffer_range(texture, buffer, ByteOffset::new(96), ByteLength::new(512))
            .unwrap();

        let texture_node = graph.resource(texture).unwrap();
        assert!(texture_node.parents.contains(&buffer));
        assert_eq!(
            graph
                .storage(texture_node.storage.unwrap())
                .unwrap()
                .backing,
            StorageBacking::BufferRange {
                buffer,
                offset: ByteOffset::new(96),
                bytes_per_row: ByteLength::new(512),
            }
        );
        assert_eq!(
            graph.alias_backings(texture).unwrap().as_ref(),
            [buffer_backing, texture_node.storage.unwrap()]
        );
        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(buffer).is_some());
        graph.release_reference(task(), object(2)).unwrap();
        assert!(graph.resource(buffer).is_none());
    }

    #[test]
    fn equal_explicit_heap_placements_share_storage_and_content_authority() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(1), ObjectKind::TextureView, None, [])
            .unwrap();
        let second = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();
        let heap = ResourceId::<HeapObject>::new(7, 3);

        graph
            .link_heap_texture(
                first,
                heap,
                Some((ByteOffset::new(0x200), ByteLength::new(0x800))),
            )
            .unwrap();
        graph
            .link_heap_texture(
                second,
                heap,
                Some((ByteOffset::new(0x200), ByteLength::new(0x800))),
            )
            .unwrap();

        let first_node = graph.resource(first).unwrap();
        let second_node = graph.resource(second).unwrap();
        assert_eq!(first_node.storage, second_node.storage);
        assert!(first_node.content.same_authority(&second_node.content));
    }

    #[test]
    fn a_heap_range_dies_with_its_last_texture_and_can_be_reused() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(1), ObjectKind::TextureView, None, [])
            .unwrap();
        let alias = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();
        let heap = ResourceId::<HeapObject>::new(7, 3);
        let placement = Some((ByteOffset::new(0x200), ByteLength::new(0x800)));

        graph.link_heap_texture(first, heap, placement).unwrap();
        graph.link_heap_texture(alias, heap, placement).unwrap();
        let storage = graph.resource(first).unwrap().storage.unwrap();

        graph.release_reference(task(), object(1)).unwrap();
        assert!(
            graph.storage(storage).is_some(),
            "the alias still owns the range"
        );
        graph.release_reference(task(), object(2)).unwrap();
        assert!(
            graph.storage(storage).is_none(),
            "the last release frees the range"
        );
        let retired = graph.take_automatically_retired_storage();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].id, storage);
        assert!(graph.take_automatically_retired_storage().is_empty());

        let replacement = graph
            .create_resource(task(), object(3), ObjectKind::TextureView, None, [])
            .unwrap();
        graph
            .link_heap_texture(
                replacement,
                heap,
                Some((ByteOffset::new(0x600), ByteLength::new(0x800))),
            )
            .expect("a placement may overlap a range whose lifetime ended");
    }

    #[test]
    fn explicit_storage_retirement_refuses_live_resource_and_mapping_owners() {
        let mut graph = ResourceGraph::default();
        let storage = graph
            .create_storage(StorageBacking::TaskAddress {
                task: task(),
                address: GuestVirtualAddress::new(0x4000),
                length: ByteLength::new(0x1000),
            })
            .unwrap();
        let resource = graph
            .create_resource(task(), object(1), ObjectKind::Buffer, Some(storage), [])
            .unwrap();
        assert_eq!(graph.retire_storage(storage), Err(GraphError::StorageInUse));
        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(resource).is_none());
        graph
            .create_mapping(MappingNode {
                id: MappingId::new(9),
                task: task(),
                address: GuestVirtualAddress::new(0x4000),
                length: ByteLength::new(0x1000),
                storage: Some(storage),
                committed: true,
            })
            .unwrap();
        assert_eq!(graph.retire_storage(storage), Err(GraphError::StorageInUse));
        graph.release_mapping(MappingId::new(9)).unwrap();
        assert_eq!(graph.retire_storage(storage).unwrap().id, storage);
        assert!(graph.storage(storage).is_none());
        assert!(graph
            .storage_overlapping(task(), 0x4000, 0x5000)
            .next()
            .is_none());
    }

    #[test]
    fn partial_heap_overlap_refuses_and_different_heap_generations_are_disjoint() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(1), ObjectKind::TextureView, None, [])
            .unwrap();
        let overlap = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();
        let reused_heap = graph
            .create_resource(task(), object(3), ObjectKind::TextureView, None, [])
            .unwrap();
        let heap = ResourceId::<HeapObject>::new(7, 3);

        graph
            .link_heap_texture(
                first,
                heap,
                Some((ByteOffset::new(0x200), ByteLength::new(0x800))),
            )
            .unwrap();
        assert_eq!(
            graph.link_heap_texture(
                overlap,
                heap,
                Some((ByteOffset::new(0x600), ByteLength::new(0x800))),
            ),
            Err(GraphError::HeapPlacementOverlap)
        );
        graph
            .link_heap_texture(
                reused_heap,
                ResourceId::<HeapObject>::new(7, 4),
                Some((ByteOffset::new(0x600), ByteLength::new(0x800))),
            )
            .unwrap();
        assert_ne!(
            graph.resource(first).unwrap().storage,
            graph.resource(reused_heap).unwrap().storage
        );
    }

    #[test]
    fn allocator_owned_heap_textures_ignore_wire_offset_by_having_distinct_allocations() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(1), ObjectKind::TextureView, None, [])
            .unwrap();
        let second = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();
        let heap = ResourceId::<HeapObject>::new(7, 3);

        graph.link_heap_texture(first, heap, None).unwrap();
        graph.link_heap_texture(second, heap, None).unwrap();

        assert_ne!(
            graph.resource(first).unwrap().storage,
            graph.resource(second).unwrap().storage
        );
        assert!(!graph
            .resource(first)
            .unwrap()
            .content
            .same_authority(&graph.resource(second).unwrap().content));
    }

    #[test]
    fn parent_relations_cannot_form_a_retention_cycle() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(1), ObjectKind::TextureView, None, [])
            .unwrap();
        let second = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();

        graph.link_parent(second, first).unwrap();
        assert_eq!(
            graph.link_parent(first, second),
            Err(GraphError::ParentCycle)
        );
        assert!(!graph.resource(first).unwrap().parents.contains(&second));
        assert!(!graph.resource(second).unwrap().children.contains(&first));
    }

    #[test]
    fn endpoint_backing_follows_only_an_unambiguous_parent_chain() {
        let mut graph = ResourceGraph::default();
        let first_parent = graph
            .create_resource(task(), object(1), ObjectKind::Texture, None, [])
            .unwrap();
        let second_parent = graph
            .create_resource(task(), object(2), ObjectKind::Texture, None, [])
            .unwrap();
        let view = graph
            .create_resource(task(), object(3), ObjectKind::TextureView, None, [])
            .unwrap();
        graph.link_parent(view, first_parent).unwrap();
        let first_storage = graph.create_storage(StorageBacking::Dedicated).unwrap();
        graph
            .attach_initial_storage(first_parent, first_storage)
            .unwrap();
        assert_eq!(graph.resolved_backing(view), Some(first_storage));

        graph.link_parent(view, second_parent).unwrap();
        let second_storage = graph.create_storage(StorageBacking::Dedicated).unwrap();
        graph
            .attach_initial_storage(second_parent, second_storage)
            .unwrap();
        assert_eq!(graph.resolved_backing(view), None);
    }

    #[test]
    fn cache_lifetime_refs_expire_only_with_the_owning_resource() {
        let lifetime = ResourceLifetime::new();
        let reference = lifetime.reference();
        let sibling = ResourceLifetime::new();

        assert!(reference.is_live());
        assert_ne!(reference.id(), sibling.reference().id());
        drop(sibling);
        assert!(
            reference.is_live(),
            "an unrelated resource cannot retire it"
        );
        drop(lifetime);
        assert!(!reference.is_live());
    }

    #[test]
    fn generational_texture_view_resolution_composes_ranges_swizzles_and_overrides() {
        let mut graph = ResourceGraph::default();
        let base = graph
            .create_resource_with_descriptor(
                task(),
                object(1),
                ObjectKind::Texture,
                Some(Arc::new(ResourceDescriptor::Texture(
                    reims_vgpu_protocol::LinearTextureDescriptor {
                        declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                            texture_type: TextureType::D2Array,
                            framebuffer_only: false,
                            is_drawable: false,
                            write_swizzle_enabled: None,
                            allow_gpu_optimized_contents: false,
                            usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                            pixel_format: 70,
                            width: 64,
                            height: 64,
                            depth: 1,
                            mipmap_level_count: 8,
                            sample_count: 1,
                            array_length: 10,
                            resource_options: 0,
                            protection_options: 0,
                            swizzle: None,
                        }),
                        ..Default::default()
                    },
                ))),
                None,
                [],
            )
            .unwrap();
        let inner_swizzle = [4, 3, 2, 5];
        let inner = graph
            .create_resource_with_descriptor(
                task(),
                object(2),
                ObjectKind::TextureView,
                Some(Arc::new(ResourceDescriptor::TextureView(
                    reims_vgpu_protocol::TextureViewDescriptor {
                        form: TextureViewForm::Swizzled,
                        view_texture_ref: 2,
                        base_texture_ref: 1,
                        pixel_format: 70,
                        texture_type: 3,
                        level_base: 2,
                        level_count: 4,
                        slice_base: 3,
                        slice_count: 5,
                        swizzle: inner_swizzle,
                    },
                ))),
                None,
                [base],
            )
            .unwrap();
        let outer_swizzle = [2, 4, 3, 5];
        let outer = graph
            .create_resource_with_descriptor(
                task(),
                object(3),
                ObjectKind::TextureView,
                Some(Arc::new(ResourceDescriptor::TextureView(
                    reims_vgpu_protocol::TextureViewDescriptor {
                        form: TextureViewForm::Swizzled,
                        view_texture_ref: 3,
                        base_texture_ref: 2,
                        pixel_format: 80,
                        texture_type: 3,
                        level_base: 1,
                        level_count: 2,
                        slice_base: 2,
                        slice_count: 2,
                        swizzle: outer_swizzle,
                    },
                ))),
                None,
                [inner],
            )
            .unwrap();

        let resolved = graph.resolve_texture_view(outer).unwrap();
        assert_eq!(resolved.view, outer);
        assert_eq!(resolved.base, base);
        assert_eq!(
            resolved.range,
            Some(ResolvedTextureViewRange {
                level_base: 3,
                level_count: 2,
                slice_base: 5,
                slice_count: 2,
            })
        );
        assert_eq!(resolved.texture_type, Some(TextureType::D2Array));
        assert_eq!(resolved.pixel_format, Some(80));
        assert_eq!(
            resolved.swizzle,
            Some(
                reims_vgpu_protocol::swizzle_plan(&outer_swizzle)
                    .unwrap()
                    .after(&reims_vgpu_protocol::swizzle_plan(&inner_swizzle).unwrap())
            )
        );
        assert_eq!(
            graph.resolve_texture_binding_view(outer).unwrap(),
            ResolvedTextureBindingView {
                resource: outer,
                base,
                image_owner: base,
                range: ResolvedTextureViewRange {
                    level_base: 3,
                    level_count: 2,
                    slice_base: 5,
                    slice_count: 2,
                },
                texture_type: TextureType::D2Array,
                pixel_format: 80,
                swizzle: reims_vgpu_protocol::swizzle_plan(&outer_swizzle)
                    .unwrap()
                    .after(&reims_vgpu_protocol::swizzle_plan(&inner_swizzle).unwrap()),
            }
        );
        let unrelated_base = graph
            .create_resource(task(), object(4), ObjectKind::Texture, None, [])
            .unwrap();
        graph
            .create_resource(
                task(),
                object(5),
                ObjectKind::TextureView,
                None,
                [unrelated_base],
            )
            .unwrap();
        assert_eq!(
            graph.texture_binding_views_for_image_owner(base).unwrap(),
            vec![
                graph.resolve_texture_binding_view(inner).unwrap(),
                graph.resolve_texture_binding_view(outer).unwrap(),
            ]
            .into_boxed_slice()
        );

        let backing = graph.create_storage(StorageBacking::Dedicated).unwrap();
        graph.attach_initial_storage(base, backing).unwrap();
        let aliased_base = graph
            .create_resource_with_descriptor(
                task(),
                object(6),
                ObjectKind::Texture,
                graph.resource(base).unwrap().descriptor.clone(),
                None,
                [],
            )
            .unwrap();
        graph.attach_initial_storage(aliased_base, backing).unwrap();
        let aliased_view = graph
            .create_resource_with_descriptor(
                task(),
                object(7),
                ObjectKind::TextureView,
                graph.resource(inner).unwrap().descriptor.clone(),
                None,
                [aliased_base],
            )
            .unwrap();
        // The alias shares the backing and owns its own view. Views belong to
        // the texture whose image they name and never to the range it happens
        // to sit on: the alias's view carries the alias's format, and no image
        // serving this base could express it.
        assert_eq!(
            graph.texture_binding_views_for_image_owner(base).unwrap(),
            vec![
                graph.resolve_texture_binding_view(inner).unwrap(),
                graph.resolve_texture_binding_view(outer).unwrap(),
            ]
            .into_boxed_slice()
        );
        assert_eq!(
            graph
                .texture_binding_views_for_image_owner(aliased_base)
                .unwrap(),
            vec![graph.resolve_texture_binding_view(aliased_view).unwrap()].into_boxed_slice()
        );
    }

    #[test]
    fn descriptor_publication_is_idempotent_and_conflict_preserving() {
        let mut graph = ResourceGraph::default();
        let resource = graph
            .create_resource(task(), object(1), ObjectKind::Buffer, None, [])
            .unwrap();
        let first = Arc::new(ResourceDescriptor::Buffer(
            reims_vgpu_protocol::BufferDescriptor {
                allocation_size: 64,
                ..Default::default()
            },
        ));
        graph
            .publish_resource_descriptor(resource, Arc::clone(&first))
            .unwrap();
        graph
            .publish_resource_descriptor(resource, Arc::new(first.as_ref().clone()))
            .unwrap();
        assert_eq!(
            graph.publish_resource_descriptor(
                resource,
                Arc::new(ResourceDescriptor::Buffer(
                    reims_vgpu_protocol::BufferDescriptor {
                        allocation_size: 128,
                        ..Default::default()
                    },
                )),
            ),
            Err(GraphError::DescriptorConflict)
        );
        assert_eq!(graph.resource(resource).unwrap().descriptor, Some(first));
    }
}
