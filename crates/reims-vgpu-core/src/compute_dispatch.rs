//! Prepared replacement compute dispatches.
//!
//! The immutable operation names only generational semantic resources. Native
//! preparation selects the construction-designated execution representation,
//! proves every read region is current there, reserves every write at the
//! operation's flattened position, and retains the ready pipeline lease.

use crate::{
    AccessIntent, AccessMode, AccessScope, AccessTarget, BackingRegion, BackingView,
    GpuWriteBatchError, GpuWriteId, GpuWriteRequest, GpuWriteReservation, ImageOwner,
    ImageSubresourceRange, ManagedBackingError, ManagedBackingProgress, ReadyPipelineLease,
    RepresentationUse, ResolvedResourceCompletion, ResourceLifecycleOwner, ResourceUseBatchError,
    SamplerResource, StageScope, ViewRepresentation,
};
use reims_vgpu_protocol::{
    BackingId, ComputePipelineObject, HazardDomainId, RepresentationId, ResourceId, ResourceObject,
    SubmissionId, TransactionId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ComputeBindingClass {
    Buffer,
    SampledImage,
    StorageImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeBindingView {
    Buffer(crate::LinearRange),
    Image(crate::ResolvedTextureBindingView),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedComputeResourceBinding {
    pub class: ComputeBindingClass,
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub view: ComputeBindingView,
    pub regions: Box<[BackingRegion]>,
    pub mode: AccessMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedComputeSamplerBinding {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub sampler: SamplerResource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedComputeNullBinding {
    pub class: ComputeBindingClass,
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
}

pub const COMPUTE_INDIRECT_ARGUMENT_BYTES: u64 = std::mem::size_of::<[u32; 3]>() as u64;

/// Shader-side grid contract derived from one immutable launch form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeProgramDispatchContract {
    DynamicThreads,
    Workgroups,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedComputeLaunch {
    Direct(reims_vgpu_protocol::dispatch::WorkgroupPlan),
    IndirectThreadgroups {
        arguments: crate::ResolvedBufferRange,
        threads_per_threadgroup: [u32; 3],
    },
}

impl ResolvedComputeLaunch {
    pub const fn program_dispatch_contract(self) -> ComputeProgramDispatchContract {
        match self {
            Self::Direct(_) => ComputeProgramDispatchContract::DynamicThreads,
            Self::IndirectThreadgroups { .. } => ComputeProgramDispatchContract::Workgroups,
        }
    }

    pub const fn threads_per_threadgroup(self) -> [u32; 3] {
        match self {
            Self::Direct(plan) => plan.threads_per_threadgroup,
            Self::IndirectThreadgroups {
                threads_per_threadgroup,
                ..
            } => threads_per_threadgroup,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedComputeDispatch {
    pub pipeline: ResourceId<ComputePipelineObject>,
    pub launch: ResolvedComputeLaunch,
    /// This command carries Metal's indirect-command `setBarrier` contract:
    /// every command before it must complete before this dispatch begins.
    /// Direct encoder barriers remain separate ordered operations.
    pub wait_for_prior_commands: bool,
    pub resources: Box<[ResolvedComputeResourceBinding]>,
    pub samplers: Box<[ResolvedComputeSamplerBinding]>,
    pub null_bindings: Box<[ResolvedComputeNullBinding]>,
}

impl ResolvedComputeDispatch {
    /// Exact canonical content consumed by this dispatch before any of its
    /// operation-local writes become visible.
    pub fn content_synchronization_requests(&self) -> Box<[crate::ContentSynchronizationRequest]> {
        let mut grouped = BTreeMap::<BackingId, BTreeSet<BackingRegion>>::new();
        for resource in &self.resources {
            if matches!(
                resource.mode,
                AccessMode::Read | AccessMode::ReadWrite | AccessMode::Unknown
            ) {
                grouped
                    .entry(resource.backing)
                    .or_default()
                    .extend(resource.regions.iter().copied());
            }
        }
        if let ResolvedComputeLaunch::IndirectThreadgroups { arguments, .. } = self.launch {
            grouped
                .entry(arguments.storage)
                .or_default()
                .insert(BackingRegion::Linear(arguments.region));
        }
        grouped
            .into_iter()
            .map(|(backing, regions)| crate::ContentSynchronizationRequest {
                backing,
                regions: regions.into_iter().collect::<Vec<_>>().into_boxed_slice(),
                permitted_pending_writes: Box::new([]),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn accesses(&self, hazard_domain: HazardDomainId) -> Box<[AccessIntent]> {
        let mut accesses = self
            .resources
            .iter()
            .flat_map(|resource| {
                resource.regions.iter().copied().map(move |region| {
                    let scope = match region {
                        BackingRegion::Whole => AccessScope::WholeBacking,
                        BackingRegion::Linear(range) => AccessScope::Linear(range),
                        BackingRegion::Image(region) => AccessScope::Image(
                            ImageSubresourceRange::new(
                                region.aspect,
                                region.mip,
                                1,
                                region.layer,
                                1,
                                Some(region.texels),
                            )
                            .expect("one exact image mip and layer are nonempty"),
                        ),
                    };
                    AccessIntent {
                        hazard_domain,
                        target: Some(AccessTarget::Backing(resource.backing)),
                        resource: Some(resource.resource),
                        scope,
                        mode: resource.mode,
                        stages: StageScope::Compute,
                    }
                })
            })
            .collect::<Vec<_>>();
        if let ResolvedComputeLaunch::IndirectThreadgroups { arguments, .. } = self.launch {
            accesses.push(AccessIntent {
                hazard_domain,
                target: Some(AccessTarget::Backing(arguments.storage)),
                resource: Some(arguments.resource),
                scope: AccessScope::Linear(arguments.region),
                mode: AccessMode::Read,
                stages: StageScope::Indirect,
            });
        }
        accesses.into_boxed_slice()
    }
}

#[derive(Debug)]
pub struct PreparedComputeDispatch<NativePipeline, Operation = ResolvedComputeDispatch> {
    transaction: TransactionId,
    operation_index: usize,
    operation: Operation,
    pipeline: ReadyPipelineLease<ComputePipelineObject, NativePipeline>,
    uses: Box<[RepresentationUse]>,
    /// The native object each bound endpoint resolves to, keyed by backing and
    /// by the view of it the binding's descriptor class names. A backing bound
    /// once as a buffer and once as a texture appears twice.
    representations: Box<[ViewRepresentation]>,
    writes: Box<[GpuWriteReservation]>,
    completions: Box<[ResolvedResourceCompletion]>,
}

impl<NativePipeline, Operation> PreparedComputeDispatch<NativePipeline, Operation> {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn operation_index(&self) -> usize {
        self.operation_index
    }

    pub const fn operation(&self) -> &Operation {
        &self.operation
    }

    pub const fn pipeline(&self) -> &ReadyPipelineLease<ComputePipelineObject, NativePipeline> {
        &self.pipeline
    }

    pub const fn uses(&self) -> &[RepresentationUse] {
        &self.uses
    }

    pub const fn representations(&self) -> &[ViewRepresentation] {
        &self.representations
    }

    pub const fn writes(&self) -> &[GpuWriteReservation] {
        &self.writes
    }

    pub const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }

    pub fn into_operation(self) -> Operation {
        self.operation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeDispatchPreparationError {
    PipelineMismatch,
    EmptyWorkgroupDimension,
    InvalidIndirectArgumentRange,
    EmptyRegions {
        binding: u32,
        array_element: u32,
    },
    DuplicateBinding {
        class: ComputeBindingClass,
        binding: u32,
        array_element: u32,
    },
    BindingViewMismatch {
        class: ComputeBindingClass,
        binding: u32,
        array_element: u32,
    },
    DuplicateSampler {
        binding: u32,
        array_element: u32,
    },
    SamplerBindingMismatch {
        binding: u32,
        sampler_binding: u32,
    },
    EmptyDescriptorArray {
        binding: u32,
    },
    ArrayElementPastDescriptorCount {
        binding: u32,
        array_element: u32,
        descriptor_count: u32,
    },
    DescriptorCountMismatch {
        binding: u32,
    },
    DescriptorClassCollision {
        binding: u32,
    },
    Backing {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
}

impl ComputeDispatchPreparationError {
    /// The backing a `StaleExecutionRepresentation` names, if that is what this
    /// is.
    ///
    /// The refusal says only that the execution representation does not hold
    /// the content the operation needs. What it holds instead is a query on the
    /// backing, and this is what lets a diagnostic that has only the failure
    /// reach it.
    pub const fn stale_backing(&self) -> Option<BackingId> {
        match self.backing_fault() {
            Some((backing, ManagedBackingError::StaleExecutionRepresentation)) => Some(backing),
            _ => None,
        }
    }

    /// The backing this refusal is about and what was wrong with it.
    ///
    /// See [`RenderDispatchPreparationError::backing_fault`]; the two dispatch
    /// classes answer the same questions and must answer them the same way.
    pub const fn backing_fault(&self) -> Option<(BackingId, ManagedBackingError)> {
        match self {
            Self::Backing { backing, reason } => Some((*backing, *reason)),
            _ => None,
        }
    }

    /// The backing that has no execution representation yet, if that is why
    /// this refused.
    ///
    /// See [`RenderDispatchPreparationError::missing_representation_backing`].
    pub const fn missing_representation_backing(&self) -> Option<BackingId> {
        match self.backing_fault() {
            Some((backing, ManagedBackingError::MissingExecutionRepresentation)) => Some(backing),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct ComputeDispatchCancellationFailure<NativePipeline> {
    pub reason: ComputeDispatchPreparationError,
    pub prepared: PreparedComputeDispatch<NativePipeline>,
}

pub fn prepare_compute_dispatch<T, NativePipeline>(
    owner: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    submission: SubmissionId,
    operation_index: usize,
    operation: ResolvedComputeDispatch,
    pipeline: ReadyPipelineLease<ComputePipelineObject, NativePipeline>,
) -> Result<PreparedComputeDispatch<NativePipeline>, ComputeDispatchPreparationError> {
    if pipeline.pipeline != operation.pipeline {
        return Err(ComputeDispatchPreparationError::PipelineMismatch);
    }
    match operation.launch {
        ResolvedComputeLaunch::Direct(workgroups)
            if workgroups.counts.contains(&0) || workgroups.threads_per_grid.contains(&0) =>
        {
            return Err(ComputeDispatchPreparationError::EmptyWorkgroupDimension);
        }
        ResolvedComputeLaunch::IndirectThreadgroups {
            arguments,
            threads_per_threadgroup,
        } => {
            if threads_per_threadgroup.contains(&0) {
                return Err(ComputeDispatchPreparationError::EmptyWorkgroupDimension);
            }
            if arguments.region.end() - arguments.region.start() != COMPUTE_INDIRECT_ARGUMENT_BYTES
            {
                return Err(ComputeDispatchPreparationError::InvalidIndirectArgumentRange);
            }
        }
        ResolvedComputeLaunch::Direct(_) => {}
    }
    let mut slots = BTreeSet::new();
    let mut samplers = BTreeSet::new();
    let mut descriptor_counts = BTreeMap::<(Option<ComputeBindingClass>, u32), u32>::new();
    let mut descriptor_classes = BTreeMap::<u32, Option<ComputeBindingClass>>::new();
    let mut grouped =
        BTreeMap::<(BackingId, BackingView), (BTreeSet<BackingRegion>, bool, bool)>::new();
    for resource in operation.resources.iter() {
        if descriptor_classes
            .insert(resource.binding, Some(resource.class))
            .is_some_and(|found| found != Some(resource.class))
        {
            return Err(ComputeDispatchPreparationError::DescriptorClassCollision {
                binding: resource.binding,
            });
        }
        if resource.descriptor_count == 0 {
            return Err(ComputeDispatchPreparationError::EmptyDescriptorArray {
                binding: resource.binding,
            });
        }
        if resource.array_element >= resource.descriptor_count {
            return Err(
                ComputeDispatchPreparationError::ArrayElementPastDescriptorCount {
                    binding: resource.binding,
                    array_element: resource.array_element,
                    descriptor_count: resource.descriptor_count,
                },
            );
        }
        if descriptor_counts
            .insert(
                (Some(resource.class), resource.binding),
                resource.descriptor_count,
            )
            .is_some_and(|found| found != resource.descriptor_count)
        {
            return Err(ComputeDispatchPreparationError::DescriptorCountMismatch {
                binding: resource.binding,
            });
        }
        if resource.regions.is_empty() {
            return Err(ComputeDispatchPreparationError::EmptyRegions {
                binding: resource.binding,
                array_element: resource.array_element,
            });
        }
        if !slots.insert((resource.class, resource.binding, resource.array_element)) {
            return Err(ComputeDispatchPreparationError::DuplicateBinding {
                class: resource.class,
                binding: resource.binding,
                array_element: resource.array_element,
            });
        }
        if !matches!(
            (resource.class, resource.view),
            (ComputeBindingClass::Buffer, ComputeBindingView::Buffer(_))
                | (
                    ComputeBindingClass::SampledImage | ComputeBindingClass::StorageImage,
                    ComputeBindingView::Image(_)
                )
        ) {
            return Err(ComputeDispatchPreparationError::BindingViewMismatch {
                class: resource.class,
                binding: resource.binding,
                array_element: resource.array_element,
            });
        }
        // The descriptor class says which view of the backing this binding
        // addresses, and the check above has already proved the class and the
        // view agree.
        let view = match resource.view {
            ComputeBindingView::Buffer(_) => BackingView::Bytes,
            ComputeBindingView::Image(view) => BackingView::Image(ImageOwner::of_view(view)),
        };
        let grouped = grouped.entry((resource.backing, view)).or_default();
        grouped.0.extend(resource.regions.iter().copied());
        grouped.1 |= matches!(
            resource.mode,
            AccessMode::Read | AccessMode::ReadWrite | AccessMode::Unknown
        );
        grouped.2 |= matches!(
            resource.mode,
            AccessMode::Write | AccessMode::ReadWrite | AccessMode::Unknown
        );
    }
    if let ResolvedComputeLaunch::IndirectThreadgroups { arguments, .. } = operation.launch {
        // Indirect arguments are read as bytes whatever else the backing is
        // bound as.
        let grouped = grouped
            .entry((arguments.storage, BackingView::Bytes))
            .or_default();
        grouped.0.insert(BackingRegion::Linear(arguments.region));
        grouped.1 = true;
    }
    for sampler in operation.samplers.iter() {
        if sampler.sampler.binding != sampler.binding {
            return Err(ComputeDispatchPreparationError::SamplerBindingMismatch {
                binding: sampler.binding,
                sampler_binding: sampler.sampler.binding,
            });
        }
        if descriptor_classes
            .insert(sampler.binding, None)
            .is_some_and(|found| found.is_some())
        {
            return Err(ComputeDispatchPreparationError::DescriptorClassCollision {
                binding: sampler.binding,
            });
        }
        if sampler.descriptor_count == 0 {
            return Err(ComputeDispatchPreparationError::EmptyDescriptorArray {
                binding: sampler.binding,
            });
        }
        if sampler.array_element >= sampler.descriptor_count {
            return Err(
                ComputeDispatchPreparationError::ArrayElementPastDescriptorCount {
                    binding: sampler.binding,
                    array_element: sampler.array_element,
                    descriptor_count: sampler.descriptor_count,
                },
            );
        }
        if descriptor_counts
            .insert((None, sampler.binding), sampler.descriptor_count)
            .is_some_and(|found| found != sampler.descriptor_count)
        {
            return Err(ComputeDispatchPreparationError::DescriptorCountMismatch {
                binding: sampler.binding,
            });
        }
        if !samplers.insert((sampler.binding, sampler.array_element)) {
            return Err(ComputeDispatchPreparationError::DuplicateSampler {
                binding: sampler.binding,
                array_element: sampler.array_element,
            });
        }
    }
    for null in operation.null_bindings.iter() {
        if descriptor_classes
            .insert(null.binding, Some(null.class))
            .is_some_and(|found| found != Some(null.class))
        {
            return Err(ComputeDispatchPreparationError::DescriptorClassCollision {
                binding: null.binding,
            });
        }
        if null.descriptor_count == 0 {
            return Err(ComputeDispatchPreparationError::EmptyDescriptorArray {
                binding: null.binding,
            });
        }
        if null.array_element >= null.descriptor_count {
            return Err(
                ComputeDispatchPreparationError::ArrayElementPastDescriptorCount {
                    binding: null.binding,
                    array_element: null.array_element,
                    descriptor_count: null.descriptor_count,
                },
            );
        }
        if descriptor_counts
            .insert((Some(null.class), null.binding), null.descriptor_count)
            .is_some_and(|found| found != null.descriptor_count)
        {
            return Err(ComputeDispatchPreparationError::DescriptorCountMismatch {
                binding: null.binding,
            });
        }
        if !slots.insert((null.class, null.binding, null.array_element)) {
            return Err(ComputeDispatchPreparationError::DuplicateBinding {
                class: null.class,
                binding: null.binding,
                array_element: null.array_element,
            });
        }
    }

    let mut representations = Vec::with_capacity(grouped.len());
    let mut grouped_uses = BTreeMap::<BackingId, Vec<RepresentationId>>::new();
    let mut write_requests = Vec::new();
    for ((backing, view), (regions, reads, writes)) in grouped {
        let regions = regions.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let representation = owner
            .view_representation(backing, view)
            .map_err(|reason| ComputeDispatchPreparationError::Backing { backing, reason })?;
        if reads {
            let snapshot = owner
                .snapshot_content(backing, &regions)
                .map_err(|reason| ComputeDispatchPreparationError::Backing { backing, reason })?;
            owner
                .view_representation_for_snapshot(backing, view, &snapshot)
                .map_err(|reason| ComputeDispatchPreparationError::Backing { backing, reason })?;
        }
        representations.push(ViewRepresentation {
            backing,
            view,
            representation,
        });
        grouped_uses
            .entry(backing)
            .or_default()
            .push(representation);
        if writes {
            write_requests.push(GpuWriteRequest {
                backing,
                representation,
                regions,
            });
        }
    }
    // One use per backing, naming every view of it this dispatch binds: both
    // native objects have to outlive the transaction.
    let uses = grouped_uses
        .into_iter()
        .map(|(backing, mut representations)| {
            representations.sort();
            representations.dedup();
            RepresentationUse {
                backing,
                representations: representations.into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>();
    let representations = representations.into_boxed_slice();
    let write = GpuWriteId::operation(transaction, submission, operation_index);
    owner
        .validate_plan_gpu_writes(write, &write_requests)
        .map_err(ComputeDispatchPreparationError::Writes)?;
    owner
        .validate_accept_uses(transaction, &uses)
        .map_err(ComputeDispatchPreparationError::Uses)?;
    let writes = owner
        .plan_gpu_writes(write, write_requests.into_boxed_slice())
        .expect("compute writes were prevalidated");
    owner
        .accept_uses(transaction, &uses)
        .expect("compute representation uses were prevalidated");
    let completions = writes
        .iter()
        .map(|write| ResolvedResourceCompletion::GpuWrite {
            backing: write.backing,
            write: write.write,
            representation: write.representation,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(PreparedComputeDispatch {
        transaction,
        operation_index,
        operation,
        pipeline,
        uses: uses.into_boxed_slice(),
        representations,
        writes,
        completions,
    })
}

pub fn cancel_prepared_compute_dispatch<T, NativePipeline>(
    owner: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedComputeDispatch<NativePipeline>,
) -> Result<ResolvedComputeDispatch, Box<ComputeDispatchCancellationFailure<NativePipeline>>> {
    if let Err(reason) = owner
        .validate_cancel_gpu_writes(prepared.writes())
        .map_err(ComputeDispatchPreparationError::Writes)
        .and_then(|()| {
            owner
                .validate_cancel_representation_uses(prepared.transaction, prepared.uses())
                .map_err(ComputeDispatchPreparationError::Uses)
        })
    {
        return Err(Box::new(ComputeDispatchCancellationFailure {
            reason,
            prepared,
        }));
    }
    owner
        .cancel_gpu_writes(prepared.writes())
        .expect("compute write cancellation was prevalidated");
    let _: Vec<(BackingId, ManagedBackingProgress<T>)> = owner
        .cancel_representation_uses(prepared.transaction, prepared.uses())
        .expect("compute use cancellation was prevalidated");
    Ok(prepared.into_operation())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assemble_prepared_exec_resources, cancel_prepared_exec_resources, ExecTransaction,
        PipelineLifecycle, PipelineReadiness, PreparedExecResourceInputs, RepresentationRoute,
        ResolvedExecSegment, ResolvedExecStream, ResolvedInfoOperation, ResolvedOperation,
        ResolvedResourceLifecycle, ResolvedTextureBindingView, ResolvedTextureViewRange,
        ResourceLifecycleEffect, SessionGeneration, StorageBacking, VulkanDeviceEpoch,
        GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ContentVersion, SegmentBoundary, SegmentKind, SessionGenerationId, SubmissionIdentity,
        TaskId, VulkanDeviceEpochId,
    };

    const EPOCH: VulkanDeviceEpochId = VulkanDeviceEpochId::new(4);

    fn image_view(resource: ResourceId<ResourceObject>) -> ResolvedTextureBindingView {
        ResolvedTextureBindingView {
            resource,
            base: resource,
            image_owner: resource,
            range: ResolvedTextureViewRange {
                level_base: 0,
                level_count: 1,
                slice_base: 0,
                slice_count: 1,
            },
            texture_type: reims_vgpu_protocol::TextureType::D2,
            pixel_format: 80,
            swizzle: reims_vgpu_protocol::swizzle_identity(),
        }
    }

    /// A backing whose execution representation serves one named view. Which
    /// one it is has to match the descriptor class the fixture binds it
    /// through: a storage buffer needs a buffer, a sampled image needs an
    /// image.
    fn backing(owner: &mut ResourceLifecycleOwner<()>, current: bool) -> BackingId {
        view_backing(owner, current, BackingView::Bytes)
    }

    fn view_backing(
        owner: &mut ResourceLifecycleOwner<()>,
        current: bool,
        view: BackingView,
    ) -> BackingId {
        let ResourceLifecycleEffect::BackingCreated(backing) = owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let execution = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                view,
                (),
            )
            .unwrap();
        if current {
            let snapshot = owner
                .snapshot_content(backing, &[BackingRegion::Whole])
                .unwrap();
            let transfer = owner
                .plan_transfers(backing, GUEST_REPRESENTATION, execution, &snapshot)
                .unwrap();
            for transfer in transfer {
                owner.complete_transfer(transfer).unwrap();
            }
        }
        backing
    }

    fn pipeline() -> ReadyPipelineLease<ComputePipelineObject, &'static str> {
        let id = ResourceId::new(3, 1);
        let mut owner =
            PipelineLifecycle::<ComputePipelineObject, (), &'static str, &'static str>::default();
        owner.declare(id, ()).unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(
                compile,
                crate::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                "pipeline",
            )
            .unwrap();
        let PipelineReadiness::Ready(lease) = owner.readiness(id, TransactionId::new(7)).unwrap()
        else {
            unreachable!()
        };
        lease
    }

    fn dispatch(backing: BackingId, mode: AccessMode) -> ResolvedComputeDispatch {
        ResolvedComputeDispatch {
            pipeline: ResourceId::new(3, 1),
            launch: ResolvedComputeLaunch::Direct(reims_vgpu_protocol::dispatch::WorkgroupPlan {
                counts: [2, 1, 1],
                threads_per_grid: [13, 1, 1],
                threads_per_threadgroup: [8, 1, 1],
            }),
            wait_for_prior_commands: false,
            resources: Box::new([ResolvedComputeResourceBinding {
                class: ComputeBindingClass::Buffer,
                binding: 0,
                array_element: 0,
                descriptor_count: 1,
                resource: ResourceId::new(5, 1),
                backing,
                view: ComputeBindingView::Buffer(crate::LinearRange::new(0, 1).unwrap()),
                regions: Box::new([BackingRegion::Whole]),
                mode,
            }]),
            samplers: Box::new([]),
            null_bindings: Box::new([]),
        }
    }

    fn indirect_dispatch(backing: BackingId, length: u64) -> ResolvedComputeDispatch {
        ResolvedComputeDispatch {
            pipeline: ResourceId::new(3, 1),
            launch: ResolvedComputeLaunch::IndirectThreadgroups {
                arguments: crate::ResolvedBufferRange {
                    resource: ResourceId::new(8, 1),
                    storage: backing,
                    region: crate::LinearRange::new(24, length).unwrap(),
                    address: reims_vgpu_protocol::GuestVirtualAddress::new(0x1018),
                    length: reims_vgpu_protocol::ByteLength::new(length),
                },
                threads_per_threadgroup: [4, 2, 1],
            },
            wait_for_prior_commands: false,
            resources: Box::new([]),
            samplers: Box::new([]),
            null_bindings: Box::new([]),
        }
    }

    #[test]
    fn prepared_dispatch_owns_exact_use_write_completion_and_pipeline_lease() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let backing = backing(&mut owner, true);
        let operation = dispatch(backing, AccessMode::ReadWrite);
        assert_eq!(
            operation.launch.program_dispatch_contract(),
            ComputeProgramDispatchContract::DynamicThreads
        );
        assert_eq!(
            operation.accesses(HazardDomainId::new(9)),
            vec![AccessIntent {
                hazard_domain: HazardDomainId::new(9),
                target: Some(AccessTarget::Backing(backing)),
                resource: Some(ResourceId::new(5, 1)),
                scope: AccessScope::WholeBacking,
                mode: AccessMode::ReadWrite,
                stages: StageScope::Compute,
            }]
            .into_boxed_slice()
        );
        let prepared = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(7),
            SubmissionId::new(11),
            4,
            operation.clone(),
            pipeline(),
        )
        .unwrap();

        assert_eq!(prepared.transaction(), TransactionId::new(7));
        assert_eq!(prepared.operation_index(), 4);
        assert_eq!(prepared.operation(), &operation);
        assert_eq!(prepared.pipeline().pipeline, operation.pipeline);
        assert_eq!(prepared.uses().len(), 1);
        assert_eq!(prepared.writes().len(), 1);
        assert_eq!(
            prepared.writes()[0].write,
            GpuWriteId::operation(TransactionId::new(7), SubmissionId::new(11), 4)
        );
        assert_eq!(
            prepared.completions(),
            [ResolvedResourceCompletion::GpuWrite {
                backing,
                write: GpuWriteId::operation(TransactionId::new(7), SubmissionId::new(11), 4,),
                representation: prepared.writes()[0].representation,
            }]
        );
        assert!(prepared
            .pipeline()
            .native_object
            .native_handles_are_usable());

        let operation = cancel_prepared_compute_dispatch(&mut owner, prepared).unwrap();
        let retry = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(7),
            SubmissionId::new(11),
            4,
            operation,
            pipeline(),
        )
        .unwrap();
        cancel_prepared_compute_dispatch(&mut owner, retry).unwrap();
    }

    #[test]
    fn indirect_threadgroups_own_the_exact_argument_read_and_validate_shape() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let argument_backing = backing(&mut owner, true);
        let operation = indirect_dispatch(argument_backing, COMPUTE_INDIRECT_ARGUMENT_BYTES);
        assert_eq!(
            operation.launch.program_dispatch_contract(),
            ComputeProgramDispatchContract::Workgroups
        );
        assert_eq!(
            operation.accesses(HazardDomainId::new(9)).as_ref(),
            [AccessIntent {
                hazard_domain: HazardDomainId::new(9),
                target: Some(AccessTarget::Backing(argument_backing)),
                resource: Some(ResourceId::new(8, 1)),
                scope: AccessScope::Linear(crate::LinearRange::new(24, 12).unwrap()),
                mode: AccessMode::Read,
                stages: StageScope::Indirect,
            }]
        );
        let prepared = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(7),
            SubmissionId::new(11),
            4,
            operation,
            pipeline(),
        )
        .unwrap();
        assert_eq!(prepared.uses().len(), 1);
        assert!(prepared.writes().is_empty());
        cancel_prepared_compute_dispatch(&mut owner, prepared).unwrap();

        let mut invalid_owner = ResourceLifecycleOwner::new(EPOCH);
        let invalid_backing = backing(&mut invalid_owner, true);
        assert_eq!(
            prepare_compute_dispatch(
                &mut invalid_owner,
                TransactionId::new(7),
                SubmissionId::new(11),
                4,
                indirect_dispatch(invalid_backing, 8),
                pipeline(),
            )
            .unwrap_err(),
            ComputeDispatchPreparationError::InvalidIndirectArgumentRange
        );

        let mut empty_owner = ResourceLifecycleOwner::new(EPOCH);
        let empty_backing = backing(&mut empty_owner, true);
        let mut empty = indirect_dispatch(empty_backing, COMPUTE_INDIRECT_ARGUMENT_BYTES);
        let ResolvedComputeLaunch::IndirectThreadgroups {
            threads_per_threadgroup,
            ..
        } = &mut empty.launch
        else {
            unreachable!()
        };
        *threads_per_threadgroup = [4, 0, 1];
        assert_eq!(
            prepare_compute_dispatch(
                &mut empty_owner,
                TransactionId::new(7),
                SubmissionId::new(11),
                4,
                empty,
                pipeline(),
            )
            .unwrap_err(),
            ComputeDispatchPreparationError::EmptyWorkgroupDimension
        );
    }

    #[test]
    fn stale_read_in_a_later_backing_refuses_before_reserving_an_earlier_write() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let writable = backing(&mut owner, true);
        let stale = view_backing(
            &mut owner,
            false,
            BackingView::Image(ImageOwner::owning(ResourceId::new(6, 1))),
        );
        let mut operation = dispatch(writable, AccessMode::Write);
        let mut resources = operation.resources.into_vec();
        resources.push(ResolvedComputeResourceBinding {
            class: ComputeBindingClass::SampledImage,
            binding: 1,
            array_element: 0,
            descriptor_count: 1,
            resource: ResourceId::new(6, 1),
            backing: stale,
            view: ComputeBindingView::Image(image_view(ResourceId::new(6, 1))),
            regions: Box::new([BackingRegion::Whole]),
            mode: AccessMode::Read,
        });
        operation.resources = resources.into_boxed_slice();
        assert_eq!(
            prepare_compute_dispatch(
                &mut owner,
                TransactionId::new(8),
                SubmissionId::new(12),
                2,
                operation,
                pipeline(),
            )
            .unwrap_err(),
            ComputeDispatchPreparationError::Backing {
                backing: stale,
                reason: ManagedBackingError::StaleExecutionRepresentation,
            }
        );

        let retry = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(8),
            SubmissionId::new(12),
            2,
            dispatch(writable, AccessMode::Write),
            pipeline(),
        )
        .unwrap();
        assert_eq!(retry.writes()[0].regions[0].version, ContentVersion::new(2));
        cancel_prepared_compute_dispatch(&mut owner, retry).unwrap();
    }

    #[test]
    fn whole_exec_envelope_retains_compute_backing_completion_and_cancellation() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let backing = backing(&mut owner, true);
        let transaction = TransactionId::new(9);
        let submission = SubmissionId::new(13);
        let operation = dispatch(backing, AccessMode::ReadWrite);
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: submission,
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Compute,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::<
                        (),
                        ResolvedComputeDispatch,
                        ResolvedInfoOperation,
                        (),
                        (),
                    >::Compute(operation.clone())]),
                }]),
            }]),
            accesses: operation.accesses(HazardDomainId::new(2)),
        };
        let prepared = prepare_compute_dispatch(
            &mut owner,
            transaction,
            submission,
            0,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        let resources = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([]),
                compute_dispatches: Box::new([prepared]),
                render_dispatches: Box::<[crate::PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        assert_eq!(resources.backings(), [backing]);
        assert_eq!(resources.resource_completions().len(), 1);
        let cancelled = cancel_prepared_exec_resources(&mut owner, resources).unwrap();
        assert_eq!(cancelled.compute_dispatches.as_ref(), [operation]);
    }
}
