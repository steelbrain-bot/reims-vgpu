//! Worker-owned native command recording resources for one Vulkan epoch.
//!
//! A recording carries the worker and command pool that allocated it. Queue
//! submission borrows those handles, while timeline retirement returns the
//! complete value to this exact owner for destruction. No other worker can
//! free or reset the pool.

use ash::vk;
use reims_vgpu_core::{
    AdmittedCompletionEffects, AdmittedConditionOperation, AdmittedExecConditions,
    AdmittedIndirectCommands, AdmittedResourceStateOperation, AdmittedResourceStates,
    BarrierOperation, BlitKind, DescriptorTier, ExecTransaction, ExpandedIndirectOperationOrigin,
    FixedExecutor, FixedExecutorError, OperationKind, RecordingWorkerId, ResolvedIndirectCommand,
    ResolvedOperation, ResolvedResourceCompletion, TransferKey,
};
use reims_vgpu_protocol::BackingId;
use reims_vgpu_protocol::TransactionId;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::mpsc,
};

use crate::{
    replacement_barrier_record::{
        record_hazard_barriers, resolve_explicit_barrier, resolve_hazard_barriers,
        BarrierRecordError, ExplicitBarrierResolveError, NativeBarrierBatch,
        ReplacementBarrierResolver, ReplacementBarrierResourceResolver,
    },
    replacement_barriers::{plan_hazard_barriers, BarrierPlanError},
    replacement_buffer_blit::{record_buffer_blit, NativeBufferBlit, ReplacementBufferBlitProgram},
    replacement_compute::{
        record_compute_dispatch, NativeComputeDispatch, ReplacementComputeProgram,
    },
    replacement_image_blit::{
        record_native_image_copies, PreparedNativeImageBlit, ReplacementImageBlitProgram,
    },
    replacement_indirect_range::{record_indirect_range_readback, ReplacementIndirectRangeProgram},
    replacement_info_query::{record_info_query, NativeInfoQuery, ReplacementInfoQueryProgram},
    replacement_render::{record_render_dispatch, NativeRenderDispatch, ReplacementRenderProgram},
    replacement_representation::ReplacementNativeRepresentationLease,
    replacement_resource_state::{
        ReplacementContentSynchronizationProgram, ReplacementHostLandingProgram,
        ReplacementResourceStateBatchProgram, ReplacementResourceStateProgram,
    },
};

#[derive(Clone, Debug)]
pub struct ReplacementNativeRecording {
    pub worker: RecordingWorkerId,
    pub queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Box<[vk::CommandBuffer]>,
    pub fence: vk::Fence,
    pub descriptor_sets: Box<[ReplacementDescriptorAllocation]>,
    pub framebuffers: Box<[vk::Framebuffer]>,
    pub query_pools: Box<[vk::QueryPool]>,
    /// Render variants referenced by recorded command buffers. They retain
    /// their exact device context and native handles until timeline retirement
    /// returns this recording to its worker.
    pub render_pipeline_variants:
        Box<[std::sync::Arc<crate::replacement_render::ReplacementRenderPipelineVariant>]>,
    /// Timeline-readable staging allocations for GPU-produced ICB ranges.
    pub(crate) indirect_range_programs: Box<[ReplacementIndirectRangeProgram]>,
    pub(crate) host_landing_programs: Box<[ReplacementHostLandingProgram]>,
    pub recorded_operations: Box<[OperationKind]>,
    /// Canonical backings whose exact native representations were retained by
    /// lifecycle preparation for this recording.
    pub(crate) backings: Box<[BackingId]>,
    /// Exact native objects referenced by raw handles in this recording.
    pub(crate) representation_leases: Box<[ReplacementNativeRepresentationLease]>,
    /// Exact resource facts completed by this recording's timeline point.
    /// They are not semantic completion facts before then.
    pub resource_completions: Box<[ResolvedResourceCompletion]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementDescriptorAllocation {
    pub worker: RecordingWorkerId,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

impl ReplacementNativeRecording {
    pub const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    pub fn host_landings(&self) -> Box<[reims_vgpu_core::HostLandingKey]> {
        self.host_landing_programs
            .iter()
            .map(|program| program.landing())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub(crate) fn take_indirect_range_programs_for(
        &mut self,
        transaction: TransactionId,
    ) -> Box<[ReplacementIndirectRangeProgram]> {
        let (matching, retained): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.indirect_range_programs)
                .into_vec()
                .into_iter()
                .partition(|program| program.transaction() == transaction);
        self.indirect_range_programs = retained.into_boxed_slice();
        matching.into_boxed_slice()
    }
}

#[cfg(test)]
impl ReplacementNativeRecording {
    pub(crate) fn synthetic(
        worker: RecordingWorkerId,
        command_buffers: impl Into<Box<[vk::CommandBuffer]>>,
        fence: vk::Fence,
    ) -> Self {
        Self {
            worker,
            queue_family: 0,
            command_pool: vk::CommandPool::null(),
            command_buffers: command_buffers.into(),
            fence,
            descriptor_sets: Box::new([]),
            framebuffers: Box::new([]),
            query_pools: Box::new([]),
            render_pipeline_variants: Box::new([]),
            indirect_range_programs: Box::new([]),
            host_landing_programs: Box::new([]),
            recorded_operations: Box::new([]),
            backings: Box::new([]),
            representation_leases: Box::new([]),
            resource_completions: Box::new([]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementRecordingError {
    NoCommandBuffers,
    OperationPositionOverflow,
    OperationPositionCountMismatch {
        operations: usize,
        positions: usize,
    },
    DuplicateOperationPosition(usize),
    TooManyVisibilityQueries,
    InvalidDescriptorRequest,
    DescriptorPoolTierUnavailable,
    BarrierPlan(BarrierPlanError),
    BarrierResolution(BarrierRecordError),
    ExplicitBarrierResolution {
        index: usize,
        reason: ExplicitBarrierResolveError,
    },
    FenceBarrierResolution {
        index: usize,
        reason: ExplicitBarrierResolveError,
    },
    DuplicateBufferBlitProgram(usize),
    UnexpectedBufferBlitProgram(usize),
    BufferBlitProgramMismatch(usize),
    BufferBlitPreparationRequired {
        index: usize,
        kind: BlitKind,
    },
    DuplicateImageBlitProgram(usize),
    UnexpectedImageBlitProgram(usize),
    ConflictingBlitPrograms(usize),
    ImageBlitProgramMismatch(usize),
    ImageBlitTransactionMismatch(usize),
    ImageBlitQueueFamilyMismatch(usize),
    ImageQueueReleaseSubmissionRequired(usize),
    ConditionAdmissionTransactionMismatch,
    ConditionAdmissionRequired {
        index: usize,
        kind: OperationKind,
    },
    ConditionAdmissionMismatch(usize),
    UnexpectedConditionAdmission(usize),
    CompletionAdmissionTransactionMismatch,
    CompletionAdmissionRequired(usize),
    CompletionAdmissionMismatch(usize),
    UnexpectedCompletionAdmission(usize),
    IndirectAdmissionTransactionMismatch,
    IndirectAdmissionRequired(usize),
    IndirectAdmissionMismatch(usize),
    IndirectExecutionPreparationRequired(usize),
    DuplicateIndirectRangeProgram(usize),
    UnexpectedIndirectRangeProgram(usize),
    IndirectRangeProgramMismatch(usize),
    IndirectRangeProgramTransactionMismatch(usize),
    UnexpectedIndirectAdmission(usize),
    ResourceStateAdmissionTransactionMismatch,
    ResourceStateBatchTransactionMismatch,
    ResourceStateAdmissionRequired(usize),
    ResourceStateAdmissionMismatch(usize),
    ResourceStatePreparationRequired(usize),
    ResourceStateNativeTransferRequired(usize),
    UnexpectedResourceStateAdmission(usize),
    DuplicateResourceStateProgram(usize),
    UnexpectedResourceStateProgram(usize),
    ResourceStateProgramMismatch(usize),
    ResourceStateProgramTransactionMismatch(usize),
    DuplicateInfoQueryProgram(usize),
    UnexpectedInfoQueryProgram(usize),
    InfoQueryProgramMismatch(usize),
    InfoQueryTransactionMismatch(usize),
    InfoQueryPreparationRequired(usize),
    DuplicateComputeProgram(usize),
    UnexpectedComputeProgram(usize),
    ComputeProgramMismatch(usize),
    ComputeProgramTransactionMismatch(usize),
    ComputePreparationRequired(usize),
    DuplicateRenderProgram(usize),
    UnexpectedRenderProgram(usize),
    RenderProgramMismatch(usize),
    RenderProgramTransactionMismatch(usize),
    RenderPreparationRequired(usize),
    RenderPassMustBegin(usize),
    RenderPassNested(usize),
    RenderPassInterrupted(usize),
    RenderPassMismatch(usize),
    RenderPassUnterminated,
    EmptyImageReleaseBatch,
    Driver(vk::Result),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementRecordingRecycleError {
    WrongWorker,
    UnknownCommandPool,
    UnknownDescriptorPool,
}

#[derive(Debug)]
pub struct ReplacementRecordingRecycleFailure {
    pub reason: ReplacementRecordingRecycleError,
    pub recording: Box<ReplacementNativeRecording>,
}

/// Unresolved request ownership before operation-local native targets have
/// been projected from canonical semantic identities.
#[derive(Clone, Debug)]
pub struct ReplacementRecordingInput<Operation> {
    pub transaction: TransactionId,
    pub worker: RecordingWorkerId,
    pub queue_family: u32,
    pub exec: ExecTransaction<Operation>,
    pub barriers: NativeBarrierBatch,
}

#[derive(Clone, Debug)]
enum ReplacementRecordedOperation {
    Covered(OperationKind),
    Barrier {
        kind: OperationKind,
        native: NativeBarrierBatch,
    },
    BufferBlit {
        native: NativeBufferBlit,
        completions: Box<[ResolvedResourceCompletion]>,
    },
    ImageBlit {
        native: PreparedNativeImageBlit,
        completions: Box<[ResolvedResourceCompletion]>,
    },
    InfoQuery {
        native: NativeInfoQuery,
        completions: Box<[ResolvedResourceCompletion]>,
    },
    Compute {
        native: NativeComputeDispatch,
        completions: Box<[ResolvedResourceCompletion]>,
    },
    Render {
        native: Box<NativeRenderDispatch>,
        completions: Box<[ResolvedResourceCompletion]>,
    },
    ResourceState {
        completions: Box<[ResolvedResourceCompletion]>,
        commands: Box<[NativeBufferBlit]>,
        image_commands: Box<[crate::replacement_image_blit::NativeImageBlitCommand]>,
        image_state: Option<crate::replacement_image_transition::PreparedNativeImageState>,
        host_landings: Box<[ReplacementHostLandingProgram]>,
    },
    ContentSynchronization {
        completions: Box<[ResolvedResourceCompletion]>,
        commands: Box<[NativeBufferBlit]>,
        image_commands: Box<[crate::replacement_image_blit::NativeImageBlitCommand]>,
    },
    IndirectRange(ReplacementIndirectRangeProgram),
}

fn retain_unique_resource_transfers(
    program: &ReplacementResourceStateProgram,
    recorded: &mut BTreeSet<TransferKey>,
    completions: &mut Vec<ResolvedResourceCompletion>,
) -> (
    Box<[NativeBufferBlit]>,
    Box<[crate::replacement_image_blit::NativeImageBlitCommand]>,
) {
    let mut buffers = Vec::new();
    let mut images = Vec::new();
    let mut newly_recorded = BTreeSet::new();
    for (transfer, command) in program
        .transfers()
        .iter()
        .copied()
        .zip(program.native_transfers())
    {
        if !recorded.insert(transfer) {
            continue;
        }
        newly_recorded.insert(transfer);
        match command {
            crate::replacement_resource_state::NativeResourceStateTransfer::Buffer(command) => {
                buffers.push(*command);
            }
            crate::replacement_resource_state::NativeResourceStateTransfer::Image(commands) => {
                images.extend(commands.iter().copied());
            }
        }
    }
    completions.extend(program.completions().iter().copied().filter(|completion| {
        !matches!(completion, ResolvedResourceCompletion::Transfer(transfer) if !newly_recorded.contains(transfer))
    }));
    (buffers.into_boxed_slice(), images.into_boxed_slice())
}

/// One immutable recording job after semantic hazards and operation-local
/// native targets have both been resolved.
#[derive(Clone, Debug)]
pub struct ReplacementRecordingRequest<Operation> {
    pub transaction: TransactionId,
    pub worker: RecordingWorkerId,
    pub queue_family: u32,
    pub exec: ExecTransaction<Operation>,
    pub barriers: NativeBarrierBatch,
    operations: Box<[ReplacementRecordedOperation]>,
    backings: Box<[BackingId]>,
    representation_leases: Box<[ReplacementNativeRepresentationLease]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementRecordingLeaseError {
    Duplicate {
        backing: BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
    },
    UnexpectedBacking(BackingId),
    MissingBacking(BackingId),
}

#[derive(Debug)]
pub struct ReplacementRecordingLeaseFailure<Operation> {
    pub reason: ReplacementRecordingLeaseError,
    pub request: Box<ReplacementRecordingRequest<Operation>>,
}

impl<Operation> ReplacementRecordingRequest<Operation> {
    pub const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    /// Retain one ownership token per native representation this recording may
    /// name.
    ///
    /// A backing has many representations and one recording can resolve
    /// several of them at once — a content-state transfer names both the
    /// source and the destination — so coverage is checked per backing rather
    /// than per representation, and identity is checked per pair.
    pub fn attach_representation_leases(
        mut self,
        leases: impl Into<Box<[ReplacementNativeRepresentationLease]>>,
    ) -> Result<Self, Box<ReplacementRecordingLeaseFailure<Operation>>> {
        let leases = leases.into();
        let mut identities = BTreeSet::new();
        for lease in &leases {
            if !identities.insert((lease.backing, lease.representation)) {
                return Err(Box::new(ReplacementRecordingLeaseFailure {
                    reason: ReplacementRecordingLeaseError::Duplicate {
                        backing: lease.backing,
                        representation: lease.representation,
                    },
                    request: Box::new(self),
                }));
            }
            if self.backings.binary_search(&lease.backing).is_err() {
                return Err(Box::new(ReplacementRecordingLeaseFailure {
                    reason: ReplacementRecordingLeaseError::UnexpectedBacking(lease.backing),
                    request: Box::new(self),
                }));
            }
        }
        if let Some(&backing) = self
            .backings
            .iter()
            .find(|backing| !leases.iter().any(|lease| lease.backing == **backing))
        {
            return Err(Box::new(ReplacementRecordingLeaseFailure {
                reason: ReplacementRecordingLeaseError::MissingBacking(backing),
                request: Box::new(self),
            }));
        }
        self.representation_leases = leases;
        Ok(self)
    }

    pub fn attach_content_synchronization(
        mut self,
        program: &ReplacementContentSynchronizationProgram,
    ) -> Result<Self, Box<Self>> {
        if program.transaction() != self.transaction {
            return Err(Box::new(self));
        }
        let commands = program
            .native_transfers()
            .iter()
            .filter_map(|command| match command {
                crate::replacement_resource_state::NativeResourceStateTransfer::Buffer(command) => {
                    Some(*command)
                }
                crate::replacement_resource_state::NativeResourceStateTransfer::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let image_commands = program
            .native_transfers()
            .iter()
            .flat_map(|command| match command {
                crate::replacement_resource_state::NativeResourceStateTransfer::Buffer(_) => None,
                crate::replacement_resource_state::NativeResourceStateTransfer::Image(commands) => {
                    Some(commands.iter().copied())
                }
            })
            .flatten()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let completions = program
            .transfers()
            .iter()
            .copied()
            .map(ResolvedResourceCompletion::Transfer)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut operations = self.operations.into_vec();
        operations.insert(
            0,
            ReplacementRecordedOperation::ContentSynchronization {
                completions,
                commands,
                image_commands,
            },
        );
        self.operations = operations.into_boxed_slice();
        let mut backings = self.backings.into_vec();
        backings.extend_from_slice(program.backings());
        backings.sort_unstable();
        backings.dedup();
        self.backings = backings.into_boxed_slice();
        Ok(self)
    }

    pub fn required_queue_flags(&self) -> vk::QueueFlags {
        let mut required = vk::QueueFlags::empty();
        for operation in &self.operations {
            match operation {
                ReplacementRecordedOperation::BufferBlit {
                    native: NativeBufferBlit::ComputeFill { .. },
                    ..
                } => required |= vk::QueueFlags::COMPUTE,
                ReplacementRecordedOperation::ImageBlit { native, .. } => {
                    required |= native.required_queue_flags();
                }
                ReplacementRecordedOperation::ResourceState { image_commands, .. }
                | ReplacementRecordedOperation::ContentSynchronization { image_commands, .. }
                    if image_commands.iter().any(|command| {
                        matches!(
                            command,
                            crate::replacement_image_blit::NativeImageBlitCommand::BufferToImage(copy)
                                | crate::replacement_image_blit::NativeImageBlitCommand::ImageToBuffer(copy)
                                if copy.aspect.intersects(
                                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                                )
                        )
                    }) => required |= vk::QueueFlags::GRAPHICS,
                ReplacementRecordedOperation::Compute { .. } => {
                    required |= vk::QueueFlags::COMPUTE;
                }
                ReplacementRecordedOperation::Render { .. } => {
                    required |= vk::QueueFlags::GRAPHICS;
                }
                ReplacementRecordedOperation::Covered(_)
                | ReplacementRecordedOperation::Barrier { .. }
                | ReplacementRecordedOperation::BufferBlit { .. }
                | ReplacementRecordedOperation::InfoQuery { .. }
                | ReplacementRecordedOperation::IndirectRange(_)
                | ReplacementRecordedOperation::ResourceState { .. } => {}
                ReplacementRecordedOperation::ContentSynchronization { .. } => {}
            }
        }
        required
    }
}

#[derive(Debug)]
pub struct ReplacementRecordingResolutionFailure<Operation> {
    pub reason: ReplacementRecordingError,
    pub input: ReplacementRecordingInput<Operation>,
}

fn validate_render_passes(
    operations: &[ReplacementRecordedOperation],
) -> Result<(), ReplacementRecordingError> {
    let mut active: Option<(
        std::sync::Arc<crate::replacement_render::ReplacementRenderPipelineVariant>,
        Box<[vk::ImageView]>,
        vk::Extent2D,
    )> = None;
    for (index, operation) in operations.iter().enumerate() {
        let ReplacementRecordedOperation::Render { native, .. } = operation else {
            if active.is_some() {
                return Err(ReplacementRecordingError::RenderPassInterrupted(index));
            }
            continue;
        };
        if native.begins_native_pass {
            if active.is_some() {
                return Err(ReplacementRecordingError::RenderPassNested(index));
            }
            active = Some((
                std::sync::Arc::clone(&native.pipeline),
                native.attachment_views.clone(),
                native.extent,
            ));
        } else {
            let Some((pipeline, attachment_views, extent)) = active.as_ref() else {
                return Err(ReplacementRecordingError::RenderPassMustBegin(index));
            };
            if !crate::replacement_render::render_passes_compatible(
                pipeline.native(),
                native.pipeline.native(),
            ) || attachment_views.as_ref() != native.attachment_views.as_ref()
                || *extent != native.extent
            {
                return Err(ReplacementRecordingError::RenderPassMismatch(index));
            }
        }
        if native.ends_native_pass {
            active = None;
        }
    }
    if active.is_some() {
        Err(ReplacementRecordingError::RenderPassUnterminated)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ReplacementImageReleaseRecordingRequest {
    pub transaction: TransactionId,
    pub worker: RecordingWorkerId,
    pub release: crate::replacement_image_transition::NativeImageRelease,
}

#[derive(Debug)]
pub struct ReplacementImageReleaseRecordingFailure {
    pub reason: ReplacementRecordingDispatchError,
    pub request: ReplacementImageReleaseRecordingRequest,
}

#[must_use = "a source-queue release recording must be observed or recovered"]
pub struct PendingReplacementImageReleaseRecording {
    receiver: mpsc::Receiver<
        Result<ReplacementNativeRecording, Box<ReplacementImageReleaseRecordingFailure>>,
    >,
    recovery: ReplacementImageReleaseRecordingRequest,
}

impl PendingReplacementImageReleaseRecording {
    pub fn wait(
        self,
    ) -> Result<ReplacementNativeRecording, Box<ReplacementImageReleaseRecordingFailure>> {
        self.receiver.recv().unwrap_or_else(|_| {
            Err(Box::new(ReplacementImageReleaseRecordingFailure {
                reason: ReplacementRecordingDispatchError::WorkerUnavailable,
                request: self.recovery,
            }))
        })
    }
}

pub type ReplacementRecordingResolutionResult<Operation> = Result<
    ReplacementRecordingRequest<Operation>,
    Box<ReplacementRecordingResolutionFailure<Operation>>,
>;

#[derive(Clone, Copy, Debug, Default)]
pub struct ReplacementSemanticAdmissions<'a, Info, Indirect, Completion> {
    pub conditions: Option<&'a AdmittedExecConditions>,
    pub completion_effects: Option<&'a AdmittedCompletionEffects<Completion>>,
    pub indirect_commands: Option<&'a AdmittedIndirectCommands<Indirect>>,
    pub resource_states: Option<&'a AdmittedResourceStates>,
    pub info_queries: &'a [ReplacementInfoQueryProgram<Info>],
    pub indirect_range_programs: &'a [ReplacementIndirectRangeProgram],
}

/// How the native recorder discharges one semantic operation family. Every
/// family is discharged; the variants differ only in which owner emits the
/// native work, not in whether the family is supported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementOperationRecording {
    HazardPlan,
    ExplicitBarrier,
    SemanticNoNativeCommand(OperationKind),
    PreparedSemanticCondition(OperationKind),
    PreparedSemanticCompletion,
    PreparedSemanticIndirect,
    PreparedNativeResourceState,
    PreparedNativeEmitter(OperationKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementBlitRecording {
    NativeEmitterImplemented(BlitKind),
}

/// Closed native-recording disposition of each blit variant.
pub const fn replacement_blit_recording(kind: BlitKind) -> ReplacementBlitRecording {
    match kind {
        BlitKind::Fill
        | BlitKind::Copy
        | BlitKind::BufferToTexture
        | BlitKind::TextureToBuffer
        | BlitKind::TextureToTexture
        | BlitKind::TextureCopyBatch => ReplacementBlitRecording::NativeEmitterImplemented(kind),
    }
}

/// Replacement-recorder disposition for one closed semantic family. This match
/// is intentionally exhaustive: extending [`OperationKind`] requires choosing
/// which owner discharges the new family.
pub const fn replacement_operation_recording(kind: OperationKind) -> ReplacementOperationRecording {
    match kind {
        OperationKind::EncoderBoundary => {
            ReplacementOperationRecording::SemanticNoNativeCommand(kind)
        }
        OperationKind::Participation => ReplacementOperationRecording::HazardPlan,
        OperationKind::Barrier => ReplacementOperationRecording::ExplicitBarrier,
        OperationKind::Blit => ReplacementOperationRecording::PreparedNativeEmitter(kind),
        OperationKind::Event | OperationKind::Fence => {
            ReplacementOperationRecording::PreparedSemanticCondition(kind)
        }
        OperationKind::Render => ReplacementOperationRecording::PreparedNativeEmitter(kind),
        OperationKind::Compute => ReplacementOperationRecording::PreparedNativeEmitter(kind),
        OperationKind::InfoQuery => ReplacementOperationRecording::PreparedNativeEmitter(kind),
        OperationKind::ResourceState => ReplacementOperationRecording::PreparedNativeResourceState,
        OperationKind::IndirectCommand => ReplacementOperationRecording::PreparedSemanticIndirect,
        OperationKind::CompletionEffect => {
            ReplacementOperationRecording::PreparedSemanticCompletion
        }
    }
}

impl<
        Render: PartialEq,
        Compute: PartialEq,
        Info: PartialEq,
        Indirect: PartialEq + ReplacementIndirectOperation,
        Completion: PartialEq,
    > ReplacementRecordingRequest<ResolvedOperation<Render, Compute, Info, Indirect, Completion>>
{
    /// Resolve the operation program from the exact EXEC it will record.
    /// Keeping this constructor on the request prevents an adapter from pairing
    /// native barrier handles with another same-shaped semantic operation.
    pub fn resolve(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_blits_and_conditions(input, resources, native, &[], &[], None)
    }

    pub fn resolve_with_buffer_blits(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_blits_and_conditions(input, resources, native, buffer_blits, &[], None)
    }

    pub fn resolve_with_blits(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_blits_and_conditions(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            None,
        )
    }

    pub fn resolve_with_blits_and_conditions(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        conditions: Option<&AdmittedExecConditions>,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_semantics(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            conditions,
            None,
        )
    }

    pub fn resolve_with_semantics(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        conditions: Option<&AdmittedExecConditions>,
        completion_effects: Option<&AdmittedCompletionEffects<Completion>>,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_all_semantics(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            ReplacementSemanticAdmissions {
                conditions,
                completion_effects,
                indirect_commands: None,
                resource_states: None,
                info_queries: &[],
                indirect_range_programs: &[],
            },
        )
    }

    pub fn resolve_with_all_semantics(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_resource_state_programs(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            semantics,
            None,
        )
    }

    pub fn resolve_with_resource_state_programs(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
        resource_state_sidecars: Option<&ReplacementResourceStateBatchProgram>,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_compute_programs(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            semantics,
            resource_state_sidecars,
            &[],
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is one independently prepared semantic operation family"
    )]
    pub fn resolve_with_compute_programs(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
        resource_state_sidecars: Option<&ReplacementResourceStateBatchProgram>,
        compute_programs: &[ReplacementComputeProgram<Compute>],
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_native_programs(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            semantics,
            resource_state_sidecars,
            compute_programs,
            &[],
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is one independently prepared semantic operation family"
    )]
    pub fn resolve_with_native_programs(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
        resource_state_sidecars: Option<&ReplacementResourceStateBatchProgram>,
        compute_programs: &[ReplacementComputeProgram<Compute>],
        render_programs: &[ReplacementRenderProgram<Render>],
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_native_programs_at(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            semantics,
            resource_state_sidecars,
            compute_programs,
            render_programs,
            0,
            None,
        )
    }

    /// Resolve one asynchronous continuation phase whose scalar base remains
    /// valid because no literal expansion changed its operation cardinality.
    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is one independently prepared semantic operation family"
    )]
    pub fn resolve_continuation_with_native_programs(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
        resource_state_sidecars: Option<&ReplacementResourceStateBatchProgram>,
        compute_programs: &[ReplacementComputeProgram<Compute>],
        render_programs: &[ReplacementRenderProgram<Render>],
        operation_base: usize,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_native_programs_at(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            semantics,
            resource_state_sidecars,
            compute_programs,
            render_programs,
            operation_base,
            None,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is one independently prepared semantic operation family"
    )]
    pub fn resolve_continuation_with_native_programs_at_positions(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
        resource_state_sidecars: Option<&ReplacementResourceStateBatchProgram>,
        compute_programs: &[ReplacementComputeProgram<Compute>],
        render_programs: &[ReplacementRenderProgram<Render>],
        operation_origins: &[ExpandedIndirectOperationOrigin],
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        Self::resolve_with_native_programs_at(
            input,
            resources,
            native,
            buffer_blits,
            image_blits,
            semantics,
            resource_state_sidecars,
            compute_programs,
            render_programs,
            0,
            Some(operation_origins),
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is one independently prepared semantic operation family"
    )]
    fn resolve_with_native_programs_at(
        input: ReplacementRecordingInput<
            ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
        >,
        resources: &impl ReplacementBarrierResourceResolver,
        native: &impl ReplacementBarrierResolver,
        buffer_blits: &[ReplacementBufferBlitProgram],
        image_blits: &[ReplacementImageBlitProgram],
        semantics: ReplacementSemanticAdmissions<'_, Info, Indirect, Completion>,
        resource_state_sidecars: Option<&ReplacementResourceStateBatchProgram>,
        compute_programs: &[ReplacementComputeProgram<Compute>],
        render_programs: &[ReplacementRenderProgram<Render>],
        operation_base: usize,
        operation_origins: Option<&[ExpandedIndirectOperationOrigin]>,
    ) -> ReplacementRecordingResolutionResult<
        ResolvedOperation<Render, Compute, Info, Indirect, Completion>,
    > {
        if semantics
            .conditions
            .is_some_and(|conditions| conditions.transaction() != input.transaction)
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::ConditionAdmissionTransactionMismatch,
                input,
            }));
        }
        if semantics
            .completion_effects
            .is_some_and(|effects| effects.transaction() != input.transaction)
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::CompletionAdmissionTransactionMismatch,
                input,
            }));
        }
        if semantics
            .indirect_commands
            .is_some_and(|commands| commands.transaction() != input.transaction)
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::IndirectAdmissionTransactionMismatch,
                input,
            }));
        }
        if semantics
            .resource_states
            .is_some_and(|states| states.transaction() != input.transaction)
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::ResourceStateAdmissionTransactionMismatch,
                input,
            }));
        }
        if resource_state_sidecars
            .is_some_and(|programs| programs.transaction() != input.transaction)
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::ResourceStateBatchTransactionMismatch,
                input,
            }));
        }
        let condition_programs = semantics
            .conditions
            .map(AdmittedExecConditions::operations)
            .unwrap_or_default()
            .iter()
            .copied()
            .collect::<BTreeMap<_, _>>();
        let completion_programs = semantics
            .completion_effects
            .map(AdmittedCompletionEffects::operations)
            .unwrap_or_default()
            .iter()
            .map(|(index, effect)| (*index, effect))
            .collect::<BTreeMap<_, _>>();
        let indirect_programs = semantics
            .indirect_commands
            .map(AdmittedIndirectCommands::operations)
            .unwrap_or_default()
            .iter()
            .map(|(index, operation)| (*index, operation))
            .collect::<BTreeMap<_, _>>();
        let mut indirect_range_sidecars = BTreeMap::new();
        for program in semantics.indirect_range_programs {
            if indirect_range_sidecars
                .insert(program.index(), program)
                .is_some()
            {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateIndirectRangeProgram(
                        program.index(),
                    ),
                    input,
                }));
            }
        }
        let resource_state_programs = semantics
            .resource_states
            .map(AdmittedResourceStates::operations)
            .unwrap_or_default()
            .iter()
            .map(|(index, operation)| (*index, operation))
            .collect::<BTreeMap<_, _>>();
        let mut prepared_resource_states = BTreeMap::new();
        for program in resource_state_sidecars
            .map(ReplacementResourceStateBatchProgram::programs)
            .unwrap_or_default()
        {
            if prepared_resource_states
                .insert(program.index(), program)
                .is_some()
            {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateResourceStateProgram(
                        program.index(),
                    ),
                    input,
                }));
            }
        }
        let mut info_query_programs = BTreeMap::new();
        for program in semantics.info_queries {
            if info_query_programs
                .insert(program.index(), program)
                .is_some()
            {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateInfoQueryProgram(program.index()),
                    input,
                }));
            }
        }
        let mut compute_sidecars = BTreeMap::new();
        for program in compute_programs {
            if compute_sidecars.insert(program.index(), program).is_some() {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateComputeProgram(program.index()),
                    input,
                }));
            }
        }
        let mut render_sidecars = BTreeMap::new();
        for program in render_programs {
            if render_sidecars.insert(program.index(), program).is_some() {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateRenderProgram(program.index()),
                    input,
                }));
            }
        }
        let mut programs = BTreeMap::new();
        for program in buffer_blits {
            if programs.insert(program.index(), program).is_some() {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateBufferBlitProgram(program.index()),
                    input,
                }));
            }
        }
        let mut image_programs = BTreeMap::new();
        for program in image_blits {
            if image_programs.insert(program.index(), program).is_some() {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateImageBlitProgram(program.index()),
                    input,
                }));
            }
            if programs.contains_key(&program.index()) {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::ConflictingBlitPrograms(program.index()),
                    input,
                }));
            }
        }
        let mut matched_programs = BTreeSet::new();
        let mut matched_image_programs = BTreeSet::new();
        let mut matched_conditions = BTreeSet::new();
        let mut matched_completion_effects = BTreeSet::new();
        let mut matched_indirect_commands = BTreeSet::new();
        let mut matched_indirect_range_programs = BTreeSet::new();
        let mut matched_resource_states = BTreeSet::new();
        let mut matched_prepared_resource_states = BTreeSet::new();
        let mut matched_info_queries = BTreeSet::new();
        let mut matched_compute_programs = BTreeSet::new();
        let mut matched_render_programs = BTreeSet::new();
        let mut recorded_transfers = BTreeSet::new();
        let mut recording_backings = BTreeSet::new();
        let mut local_fence_updates = BTreeMap::new();
        let operation_count = input.exec.operations().count();
        if let Some(operation_origins) = operation_origins {
            if operation_origins.len() != operation_count {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::OperationPositionCountMismatch {
                        operations: operation_count,
                        positions: operation_origins.len(),
                    },
                    input,
                }));
            }
            let mut unique = BTreeSet::new();
            if let Some(position) = operation_origins
                .iter()
                .map(|origin| origin.expanded_position)
                .find(|position| !unique.insert(*position))
            {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason: ReplacementRecordingError::DuplicateOperationPosition(position),
                    input,
                }));
            }
        }
        let operations = input.exec.operations().enumerate()
            .map(|(local_index, operation)| {
                let (resource_index, semantic_index) = match operation_origins {
                    Some(operation_origins) => {
                        let origin = operation_origins[local_index];
                        (origin.expanded_position, origin.original_position)
                    }
                    None => {
                        let index = operation_base
                            .checked_add(local_index)
                            .ok_or(ReplacementRecordingError::OperationPositionOverflow)?;
                        (index, index)
                    }
                };
                match operation {
                ResolvedOperation::EncoderBoundary(_) => Ok(ReplacementRecordedOperation::Covered(
                    OperationKind::EncoderBoundary,
                )),
                ResolvedOperation::Participation(_) => Ok(ReplacementRecordedOperation::Covered(
                    OperationKind::Participation,
                )),
                ResolvedOperation::Compute(compute) => {
                    let index = resource_index;
                    match compute_sidecars.get(&index) {
                    Some(program) if program.operation() != compute => {
                        Err(ReplacementRecordingError::ComputeProgramMismatch(index))
                    }
                    Some(program) if program.transaction() != input.transaction => Err(
                        ReplacementRecordingError::ComputeProgramTransactionMismatch(index),
                    ),
                    Some(program) => {
                        matched_compute_programs.insert(index);
                        recording_backings.extend(program.backings().iter().copied());
                        Ok(ReplacementRecordedOperation::Compute {
                            native: program.native().clone(),
                            completions: program.completions().into(),
                        })
                    }
                    None => Err(ReplacementRecordingError::ComputePreparationRequired(index)),
                    }
                }
                ResolvedOperation::Render(render) => {
                    let index = resource_index;
                    match render_sidecars.get(&index) {
                    Some(program) if program.operation() != render => {
                        Err(ReplacementRecordingError::RenderProgramMismatch(index))
                    }
                    Some(program) if program.transaction() != input.transaction => Err(
                        ReplacementRecordingError::RenderProgramTransactionMismatch(index),
                    ),
                    Some(program) => {
                        matched_render_programs.insert(index);
                        recording_backings.extend(program.backings().iter().copied());
                        Ok(ReplacementRecordedOperation::Render {
                            native: Box::new(program.native().clone()),
                            completions: program.completions().into(),
                        })
                    }
                    None => Err(ReplacementRecordingError::RenderPreparationRequired(index)),
                    }
                }
                ResolvedOperation::Event(event) => {
                    let index = semantic_index;
                    match condition_programs.get(&index) {
                    Some(AdmittedConditionOperation::Event(admitted)) if admitted == event => {
                        matched_conditions.insert(index);
                        Ok(ReplacementRecordedOperation::Covered(OperationKind::Event))
                    }
                    Some(_) => Err(ReplacementRecordingError::ConditionAdmissionMismatch(index)),
                    None => Err(ReplacementRecordingError::ConditionAdmissionRequired {
                        index,
                        kind: OperationKind::Event,
                    }),
                    }
                }
                ResolvedOperation::Fence(fence) => {
                    let index = semantic_index;
                    match condition_programs.get(&index) {
                    Some(AdmittedConditionOperation::Fence(admitted)) if admitted == fence => {
                        matched_conditions.insert(index);
                        match fence.kind {
                            reims_vgpu_core::FenceOperationKind::Update => {
                                local_fence_updates
                                    .insert((fence.fence, fence.generation), fence.scope);
                                Ok(ReplacementRecordedOperation::Covered(OperationKind::Fence))
                            }
                            reims_vgpu_core::FenceOperationKind::Wait => {
                                let Some(producer) = local_fence_updates
                                    .get(&(fence.fence, fence.generation))
                                    .copied()
                                else {
                                    return Ok(ReplacementRecordedOperation::Covered(
                                        OperationKind::Fence,
                                    ));
                                };
                                resolve_explicit_barrier(
                                    &BarrierOperation::Scope {
                                        scope: reims_vgpu_core::MemoryBarrierScope::ALL,
                                        before: producer.stage_scope(),
                                        after: fence.scope.stage_scope(),
                                    },
                                    resources,
                                    native,
                                )
                                .map(|native| ReplacementRecordedOperation::Barrier {
                                    kind: OperationKind::Fence,
                                    native,
                                })
                                .map_err(|reason| {
                                    ReplacementRecordingError::FenceBarrierResolution {
                                        index,
                                        reason,
                                    }
                                })
                            }
                        }
                    }
                    Some(_) => Err(ReplacementRecordingError::ConditionAdmissionMismatch(index)),
                    None => Err(ReplacementRecordingError::ConditionAdmissionRequired {
                        index,
                        kind: OperationKind::Fence,
                    }),
                    }
                }
                ResolvedOperation::Barrier(barrier) => {
                    let index = semantic_index;
                    resolve_explicit_barrier(barrier, resources, native)
                        .map(|native| ReplacementRecordedOperation::Barrier {
                            kind: OperationKind::Barrier,
                            native,
                        })
                        .map_err(
                            |reason| ReplacementRecordingError::ExplicitBarrierResolution {
                                index,
                                reason,
                            },
                        )
                }
                ResolvedOperation::Blit(blit) => {
                    let index = resource_index;
                    match (programs.get(&index), image_programs.get(&index)) {
                        (Some(program), None) if program.operation() == blit.as_ref() => {
                            matched_programs.insert(index);
                            recording_backings.extend(program.backings().iter().copied());
                            Ok(ReplacementRecordedOperation::BufferBlit {
                                native: program.native(),
                                completions: program.completions().into(),
                            })
                        }
                        (Some(_), None) => {
                            Err(ReplacementRecordingError::BufferBlitProgramMismatch(index))
                        }
                        (None, Some(program)) if program.operation() != blit.as_ref() => {
                            Err(ReplacementRecordingError::ImageBlitProgramMismatch(index))
                        }
                        (None, Some(program))
                            if program.native().state.transaction != input.transaction =>
                        {
                            Err(ReplacementRecordingError::ImageBlitTransactionMismatch(
                                index,
                            ))
                        }
                        (None, Some(program))
                            if program.native().state.destination_queue_family
                                != input.queue_family =>
                        {
                            Err(ReplacementRecordingError::ImageBlitQueueFamilyMismatch(
                                index,
                            ))
                        }
                        (None, Some(program)) if !program.native().state.releases.is_empty() => {
                            Err(
                                ReplacementRecordingError::ImageQueueReleaseSubmissionRequired(
                                    index,
                                ),
                            )
                        }
                        (None, Some(program)) => {
                            matched_image_programs.insert(index);
                            recording_backings.extend(program.backings().iter().copied());
                            Ok(ReplacementRecordedOperation::ImageBlit {
                                native: program.native().clone(),
                                completions: program.completions().into(),
                            })
                        }
                        (None, None) => {
                            Err(ReplacementRecordingError::BufferBlitPreparationRequired {
                                index,
                                kind: blit.kind(),
                            })
                        }
                        (Some(_), Some(_)) => unreachable!("conflicting programs were rejected"),
                    }
                }
                ResolvedOperation::CompletionEffect(effect) => {
                    let index = semantic_index;
                    match completion_programs.get(&index) {
                        Some(admitted) if *admitted == effect => {
                            matched_completion_effects.insert(index);
                            Ok(ReplacementRecordedOperation::Covered(
                                OperationKind::CompletionEffect,
                            ))
                        }
                        Some(_) => Err(ReplacementRecordingError::CompletionAdmissionMismatch(
                            index,
                        )),
                        None => Err(ReplacementRecordingError::CompletionAdmissionRequired(
                            index,
                        )),
                    }
                }
                ResolvedOperation::IndirectCommand(command) => {
                    let index = semantic_index;
                    match indirect_programs.get(&index) {
                        Some(admitted) if *admitted == command => {
                            if command.requires_range_readback() {
                                let Some(program) = indirect_range_sidecars.get(&resource_index) else {
                                    return Err(
                                        ReplacementRecordingError::IndirectExecutionPreparationRequired(
                                            index,
                                        ),
                                    );
                                };
                                if !command.matches_range_readback(program.operation()) {
                                    return Err(
                                        ReplacementRecordingError::IndirectRangeProgramMismatch(
                                            index,
                                        ),
                                    );
                                }
                                if program.transaction() != input.transaction {
                                    return Err(
                                        ReplacementRecordingError::IndirectRangeProgramTransactionMismatch(
                                            index,
                                        ),
                                    );
                                }
                                matched_indirect_range_programs.insert(resource_index);
                                matched_indirect_commands.insert(index);
                                recording_backings.insert(program.backing());
                                return Ok(ReplacementRecordedOperation::IndirectRange(
                                    (*program).clone(),
                                ));
                            }
                            if command.requires_native_execution() {
                                return Err(
                                    ReplacementRecordingError::IndirectExecutionPreparationRequired(
                                        index,
                                    ),
                                );
                            }
                            matched_indirect_commands.insert(index);
                            Ok(ReplacementRecordedOperation::Covered(
                                OperationKind::IndirectCommand,
                            ))
                        }
                        Some(_) => Err(ReplacementRecordingError::IndirectAdmissionMismatch(index)),
                        None => Err(ReplacementRecordingError::IndirectAdmissionRequired(index)),
                    }
                }
                ResolvedOperation::ResourceState(state) => {
                    let index = semantic_index;
                    match resource_state_programs.get(&index) {
                        Some(AdmittedResourceStateOperation::Semantic(admitted))
                            if admitted == state =>
                        {
                            match prepared_resource_states.get(&resource_index) {
                                Some(program) if program.operation() != state => Err(
                                    ReplacementRecordingError::ResourceStateProgramMismatch(
                                        resource_index,
                                    ),
                                ),
                                Some(program) if program.transaction() != input.transaction => Err(
                                    ReplacementRecordingError::ResourceStateProgramTransactionMismatch(
                                        resource_index,
                                    ),
                                ),
                                Some(program) => {
                                    matched_resource_states.insert(index);
                                    matched_prepared_resource_states.insert(resource_index);
                                    recording_backings.extend(program.backings().iter().copied());
                                    let mut completions = Vec::new();
                                    let (commands, image_commands) = retain_unique_resource_transfers(
                                        program,
                                        &mut recorded_transfers,
                                        &mut completions,
                                    );
                                    Ok(ReplacementRecordedOperation::ResourceState {
                                        completions: completions.into_boxed_slice(),
                                        commands,
                                        image_commands,
                                        image_state: program.image_state().cloned(),
                                        host_landings: program.host_landings().into(),
                                    })
                                }
                                None => Err(
                                    ReplacementRecordingError::ResourceStatePreparationRequired(
                                        resource_index,
                                    ),
                                ),
                            }
                        }
                        Some(AdmittedResourceStateOperation::NativeTransferMayBeRequired(
                            admitted,
                        )) if admitted == state => match prepared_resource_states.get(&resource_index) {
                            Some(program) if program.operation() != state => Err(
                                ReplacementRecordingError::ResourceStateProgramMismatch(
                                    resource_index,
                                ),
                            ),
                            Some(program) if program.transaction() != input.transaction => Err(
                                ReplacementRecordingError::ResourceStateProgramTransactionMismatch(
                                    resource_index,
                                ),
                            ),
                            Some(program) => {
                                matched_resource_states.insert(index);
                                matched_prepared_resource_states.insert(resource_index);
                                recording_backings.extend(program.backings().iter().copied());
                                let mut completions = Vec::new();
                                let (commands, image_commands) = retain_unique_resource_transfers(
                                    program,
                                    &mut recorded_transfers,
                                    &mut completions,
                                );
                                Ok(ReplacementRecordedOperation::ResourceState {
                                    completions: completions.into_boxed_slice(),
                                    commands,
                                    image_commands,
                                    image_state: program.image_state().cloned(),
                                    host_landings: program.host_landings().into(),
                                })
                            }
                            None => Err(
                                ReplacementRecordingError::ResourceStateNativeTransferRequired(
                                    resource_index,
                                ),
                            ),
                        },
                        Some(_) => Err(ReplacementRecordingError::ResourceStateAdmissionMismatch(
                            index,
                        )),
                        None => Err(ReplacementRecordingError::ResourceStateAdmissionRequired(
                            index,
                        )),
                    }
                }
                ResolvedOperation::InfoQuery(query) => {
                    let index = resource_index;
                    match info_query_programs.get(&index) {
                    Some(program) if program.operation() != query => {
                        Err(ReplacementRecordingError::InfoQueryProgramMismatch(index))
                    }
                    Some(program) if program.transaction() != input.transaction => Err(
                        ReplacementRecordingError::InfoQueryTransactionMismatch(index),
                    ),
                    Some(program) => {
                        matched_info_queries.insert(index);
                        recording_backings.insert(program.backing());
                        Ok(ReplacementRecordedOperation::InfoQuery {
                            native: program.native().clone(),
                            completions: program.completions().into(),
                        })
                    }
                    None => Err(ReplacementRecordingError::InfoQueryPreparationRequired(
                        index,
                    )),
                    }
                }
                }
            })
            .collect::<Result<Vec<_>, _>>();
        let operations = match operations {
            Ok(operations) => operations,
            Err(reason) => {
                return Err(Box::new(ReplacementRecordingResolutionFailure {
                    reason,
                    input,
                }));
            }
        };
        if let Some(index) = programs
            .keys()
            .copied()
            .find(|index| !matched_programs.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedBufferBlitProgram(index),
                input,
            }));
        }
        if let Some(index) = image_programs
            .keys()
            .copied()
            .find(|index| !matched_image_programs.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedImageBlitProgram(index),
                input,
            }));
        }
        if let Some(index) = condition_programs
            .keys()
            .copied()
            .find(|index| !matched_conditions.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedConditionAdmission(index),
                input,
            }));
        }
        if let Some(index) = completion_programs
            .keys()
            .copied()
            .find(|index| !matched_completion_effects.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedCompletionAdmission(index),
                input,
            }));
        }
        if let Some(index) = indirect_programs
            .keys()
            .copied()
            .find(|index| !matched_indirect_commands.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedIndirectAdmission(index),
                input,
            }));
        }
        if let Some(index) = indirect_range_sidecars
            .keys()
            .copied()
            .find(|index| !matched_indirect_range_programs.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedIndirectRangeProgram(index),
                input,
            }));
        }
        if let Some(index) = resource_state_programs
            .keys()
            .copied()
            .find(|index| !matched_resource_states.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedResourceStateAdmission(index),
                input,
            }));
        }
        if let Some(index) = prepared_resource_states
            .keys()
            .copied()
            .find(|index| !matched_prepared_resource_states.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedResourceStateProgram(index),
                input,
            }));
        }
        if let Some(index) = info_query_programs
            .keys()
            .copied()
            .find(|index| !matched_info_queries.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedInfoQueryProgram(index),
                input,
            }));
        }
        if let Some(index) = compute_sidecars
            .keys()
            .copied()
            .find(|index| !matched_compute_programs.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedComputeProgram(index),
                input,
            }));
        }
        if let Some(index) = render_sidecars
            .keys()
            .copied()
            .find(|index| !matched_render_programs.contains(index))
        {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason: ReplacementRecordingError::UnexpectedRenderProgram(index),
                input,
            }));
        }
        if let Err(reason) = validate_render_passes(&operations) {
            return Err(Box::new(ReplacementRecordingResolutionFailure {
                reason,
                input,
            }));
        }
        Ok(Self {
            transaction: input.transaction,
            worker: input.worker,
            queue_family: input.queue_family,
            exec: input.exec,
            barriers: input.barriers,
            operations: operations.into_boxed_slice(),
            backings: recording_backings
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            representation_leases: Box::new([]),
        })
    }
}

mod indirect_operation_sealed {
    pub trait Sealed {}
    impl Sealed for () {}
    impl Sealed for reims_vgpu_core::ResolvedIndirectCommand {}
}

pub trait ReplacementIndirectOperation: indirect_operation_sealed::Sealed {
    fn requires_native_execution(&self) -> bool;
    fn requires_range_readback(&self) -> bool;
    fn matches_range_readback(&self, operation: ResolvedIndirectCommand) -> bool;
}

impl ReplacementIndirectOperation for () {
    fn requires_native_execution(&self) -> bool {
        false
    }

    fn requires_range_readback(&self) -> bool {
        false
    }

    fn matches_range_readback(&self, _: ResolvedIndirectCommand) -> bool {
        false
    }
}

impl ReplacementIndirectOperation for ResolvedIndirectCommand {
    fn requires_native_execution(&self) -> bool {
        matches!(
            self,
            ResolvedIndirectCommand::Execute { .. }
                | ResolvedIndirectCommand::ExecuteIndirectRange { .. }
        )
    }

    fn requires_range_readback(&self) -> bool {
        matches!(self, ResolvedIndirectCommand::ExecuteIndirectRange { .. })
    }

    fn matches_range_readback(&self, operation: ResolvedIndirectCommand) -> bool {
        *self == operation
    }
}

mod private {
    pub trait Sealed {}
}

/// Closed classification of operation families currently emitted by the
/// replacement worker. This trait is sealed so an adapter cannot claim that
/// an arbitrary payload was recorded.
pub trait ReplacementRecordingOperation: private::Sealed {
    fn kind(&self) -> OperationKind;

    fn recording(&self) -> ReplacementOperationRecording;
}

impl<Render, Compute, Info, Indirect, Completion> private::Sealed
    for ResolvedOperation<Render, Compute, Info, Indirect, Completion>
{
}

impl<Render, Compute, Info, Indirect, Completion> ReplacementRecordingOperation
    for ResolvedOperation<Render, Compute, Info, Indirect, Completion>
{
    fn kind(&self) -> OperationKind {
        ResolvedOperation::kind(self)
    }

    fn recording(&self) -> ReplacementOperationRecording {
        replacement_operation_recording(self.kind())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementRecordingDispatchError {
    Executor(FixedExecutorError),
    WorkerIdentityMismatch {
        expected: RecordingWorkerId,
        actual: RecordingWorkerId,
    },
    Native(ReplacementRecordingError),
    WorkerUnavailable,
}

#[derive(Debug)]
pub struct ReplacementRecordingDispatchFailure<Operation> {
    pub reason: ReplacementRecordingDispatchError,
    pub request: ReplacementRecordingRequest<Operation>,
}

#[must_use = "a dispatched recording must be observed or its native ownership cannot advance"]
#[derive(Debug)]
pub struct PendingReplacementRecording<Operation> {
    receiver: mpsc::Receiver<
        Result<ReplacementNativeRecording, Box<ReplacementRecordingDispatchFailure<Operation>>>,
    >,
    recovery: ReplacementRecordingRequest<Operation>,
}

pub enum ReplacementRecordingPoll<Operation> {
    Pending(PendingReplacementRecording<Operation>),
    Completed(
        Result<ReplacementNativeRecording, Box<ReplacementRecordingDispatchFailure<Operation>>>,
    ),
}

impl<Operation> PendingReplacementRecording<Operation> {
    pub fn try_complete(self) -> ReplacementRecordingPoll<Operation> {
        match self.receiver.try_recv() {
            Ok(result) => ReplacementRecordingPoll::Completed(result),
            Err(mpsc::TryRecvError::Empty) => ReplacementRecordingPoll::Pending(self),
            Err(mpsc::TryRecvError::Disconnected) => ReplacementRecordingPoll::Completed(Err(
                Box::new(ReplacementRecordingDispatchFailure {
                    reason: ReplacementRecordingDispatchError::WorkerUnavailable,
                    request: self.recovery,
                }),
            )),
        }
    }

    pub fn wait(
        self,
    ) -> Result<ReplacementNativeRecording, Box<ReplacementRecordingDispatchFailure<Operation>>>
    {
        self.receiver.recv().unwrap_or({
            Err(Box::new(ReplacementRecordingDispatchFailure {
                reason: ReplacementRecordingDispatchError::WorkerUnavailable,
                request: self.recovery,
            }))
        })
    }
}

/// Dispatch one immutable recording request to its assigned epoch worker.
/// Admission refusal and native recording failure both return the complete
/// request, so neither can silently consume the semantic transaction.
pub fn dispatch_replacement_recording<
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
>(
    executor: &FixedExecutor<ReplacementRecordingWorker>,
    request: ReplacementRecordingRequest<Operation>,
) -> Result<
    PendingReplacementRecording<Operation>,
    Box<ReplacementRecordingDispatchFailure<Operation>>,
> {
    let recovery = request.clone();
    let worker = request.worker;
    let transaction = request.transaction;
    let (sender, receiver) = mpsc::sync_channel(1);
    if let Err(reason) = executor.submit_transaction_to(worker, transaction, move |state| {
        let result = if state.id() != request.worker {
            Err(Box::new(ReplacementRecordingDispatchFailure {
                reason: ReplacementRecordingDispatchError::WorkerIdentityMismatch {
                    expected: request.worker,
                    actual: state.id(),
                },
                request,
            }))
        } else {
            let recorded_operations = request
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    ReplacementRecordedOperation::Covered(kind) => Some(*kind),
                    ReplacementRecordedOperation::Barrier { kind, .. } => Some(*kind),
                    ReplacementRecordedOperation::BufferBlit { .. } => Some(OperationKind::Blit),
                    ReplacementRecordedOperation::ImageBlit { .. } => Some(OperationKind::Blit),
                    ReplacementRecordedOperation::InfoQuery { .. } => {
                        Some(OperationKind::InfoQuery)
                    }
                    ReplacementRecordedOperation::Compute { .. } => Some(OperationKind::Compute),
                    ReplacementRecordedOperation::Render { .. } => Some(OperationKind::Render),
                    ReplacementRecordedOperation::ResourceState { .. } => {
                        Some(OperationKind::ResourceState)
                    }
                    ReplacementRecordedOperation::IndirectRange(_) => {
                        Some(OperationKind::IndirectCommand)
                    }
                    ReplacementRecordedOperation::ContentSynchronization { .. } => None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let mut unique_resource_completions = std::collections::BTreeSet::new();
            let resource_completions = request
                .operations
                .iter()
                .flat_map(|operation| match operation {
                    ReplacementRecordedOperation::BufferBlit { completions, .. }
                    | ReplacementRecordedOperation::ImageBlit { completions, .. }
                    | ReplacementRecordedOperation::InfoQuery { completions, .. }
                    | ReplacementRecordedOperation::Compute { completions, .. }
                    | ReplacementRecordedOperation::Render { completions, .. }
                    | ReplacementRecordedOperation::ResourceState { completions, .. }
                    | ReplacementRecordedOperation::ContentSynchronization {
                        completions, ..
                    } => completions.to_vec(),
                    ReplacementRecordedOperation::Covered(_)
                    | ReplacementRecordedOperation::Barrier { .. }
                    | ReplacementRecordedOperation::IndirectRange(_) => Vec::new(),
                })
                .filter(|completion| unique_resource_completions.insert(*completion))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let mut render_pipeline_variants = Vec::new();
            for native in request.operations.iter().filter_map(|operation| {
                let ReplacementRecordedOperation::Render { native, .. } = operation else {
                    return None;
                };
                Some(&native.pipeline)
            }) {
                if !render_pipeline_variants
                    .iter()
                    .any(|retained| std::sync::Arc::ptr_eq(retained, native))
                {
                    render_pipeline_variants.push(std::sync::Arc::clone(native));
                }
            }
            let render_pipeline_variants = render_pipeline_variants.into_boxed_slice();
            let indirect_range_programs = request
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    ReplacementRecordedOperation::IndirectRange(program) => Some(program.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let mut host_landing_programs = BTreeMap::new();
            for program in request
                .operations
                .iter()
                .flat_map(|operation| match operation {
                    ReplacementRecordedOperation::ResourceState { host_landings, .. } => {
                        host_landings.to_vec()
                    }
                    _ => Vec::new(),
                })
            {
                host_landing_programs
                    .entry(program.landing())
                    .or_insert(program);
            }
            let host_landing_programs = host_landing_programs
                .into_values()
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let backings = request.backings.clone();
            // Every raw handle a barrier batch holds keeps its exact native
            // object alive through the recording's timeline retirement. A
            // barrier names backings the operation sidecars never do, so its
            // leases are additional to the ones lifecycle preparation attached.
            let representation_leases = request
                .representation_leases
                .iter()
                .chain(request.barriers.leases())
                .chain(
                    request
                        .operations
                        .iter()
                        .filter_map(|operation| match operation {
                            ReplacementRecordedOperation::Barrier { native, .. } => {
                                Some(native.leases())
                            }
                            _ => None,
                        })
                        .flatten(),
                )
                .cloned()
                .collect::<Vec<_>>()
                .into_boxed_slice();
            state
                .record_barrier_program(
                    request.queue_family,
                    &request.barriers,
                    &request.operations,
                )
                .map(|mut recording| {
                    recording.recorded_operations = recorded_operations;
                    recording.resource_completions = resource_completions;
                    recording.render_pipeline_variants = render_pipeline_variants;
                    recording.indirect_range_programs = indirect_range_programs;
                    recording.host_landing_programs = host_landing_programs;
                    recording.backings = backings;
                    recording.representation_leases = representation_leases;
                    recording
                })
                .map_err(|reason| {
                    Box::new(ReplacementRecordingDispatchFailure {
                        reason: ReplacementRecordingDispatchError::Native(reason),
                        request,
                    })
                })
        };
        let _ = sender.send(result);
    }) {
        return Err(Box::new(ReplacementRecordingDispatchFailure {
            reason: ReplacementRecordingDispatchError::Executor(reason),
            request: recovery,
        }));
    }
    Ok(PendingReplacementRecording { receiver, recovery })
}

pub fn dispatch_image_release_recording(
    executor: &FixedExecutor<ReplacementRecordingWorker>,
    request: ReplacementImageReleaseRecordingRequest,
) -> Result<PendingReplacementImageReleaseRecording, Box<ReplacementImageReleaseRecordingFailure>> {
    if request.release.barriers.memory.is_empty()
        && request.release.barriers.buffers.is_empty()
        && request.release.barriers.images.is_empty()
    {
        return Err(Box::new(ReplacementImageReleaseRecordingFailure {
            reason: ReplacementRecordingDispatchError::Native(
                ReplacementRecordingError::EmptyImageReleaseBatch,
            ),
            request,
        }));
    }
    let recovery = request.clone();
    let worker = request.worker;
    let transaction = request.transaction;
    let (sender, receiver) = mpsc::sync_channel(1);
    if let Err(reason) = executor.submit_transaction_to(worker, transaction, move |state| {
        let result = if state.id() != request.worker {
            Err(Box::new(ReplacementImageReleaseRecordingFailure {
                reason: ReplacementRecordingDispatchError::WorkerIdentityMismatch {
                    expected: request.worker,
                    actual: state.id(),
                },
                request,
            }))
        } else {
            state
                .record_image_release(&request.release)
                .map_err(|reason| {
                    Box::new(ReplacementImageReleaseRecordingFailure {
                        reason: ReplacementRecordingDispatchError::Native(reason),
                        request,
                    })
                })
        };
        let _ = sender.send(result);
    }) {
        return Err(Box::new(ReplacementImageReleaseRecordingFailure {
            reason: ReplacementRecordingDispatchError::Executor(reason),
            request: recovery,
        }));
    }
    Ok(PendingReplacementImageReleaseRecording { receiver, recovery })
}

/// Mutable native allocation state owned by exactly one fixed executor worker.
pub struct ReplacementRecordingWorker {
    id: RecordingWorkerId,
    device: ash::Device,
    command_pools: BTreeMap<u32, vk::CommandPool>,
    descriptor_tier: DescriptorTier,
    descriptor_blocks: Vec<DescriptorPoolBlock>,
    fill_pipeline: Option<crate::replacement_fill::ReplacementFillPipeline>,
}

struct DescriptorPoolBlock {
    signature: Box<[(i32, u32)]>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    in_use: bool,
}

impl ReplacementRecordingWorker {
    pub fn new(
        id: RecordingWorkerId,
        device: &ash::Device,
        descriptor_tier: DescriptorTier,
    ) -> Self {
        Self {
            id,
            device: device.clone(),
            command_pools: BTreeMap::new(),
            descriptor_tier,
            descriptor_blocks: Vec::new(),
            fill_pipeline: None,
        }
    }

    pub const fn id(&self) -> RecordingWorkerId {
        self.id
    }

    /// Allocate one reusable descriptor set from this worker's fallback pool
    /// tier. Each block is sized from the exact request and owns one set, so no
    /// arbitrary block capacity can reject guest work. Concurrent demand grows
    /// the unbounded arena; retirement makes the same set available again.
    pub fn allocate_descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
        counts: &[(vk::DescriptorType, u32)],
    ) -> Result<ReplacementDescriptorAllocation, ReplacementRecordingError> {
        if self.descriptor_tier != DescriptorTier::WorkerDescriptorPool {
            return Err(ReplacementRecordingError::DescriptorPoolTierUnavailable);
        }
        self.allocate_descriptor_block(layout, counts)
    }

    fn allocate_descriptor_block(
        &mut self,
        layout: vk::DescriptorSetLayout,
        counts: &[(vk::DescriptorType, u32)],
    ) -> Result<ReplacementDescriptorAllocation, ReplacementRecordingError> {
        let signature = descriptor_signature(counts)?;
        if let Some(block) = self.descriptor_blocks.iter_mut().find(|block| {
            !block.in_use
                && block.layout == layout
                && block.signature.as_ref() == signature.as_ref()
        }) {
            block.in_use = true;
            return Ok(ReplacementDescriptorAllocation {
                worker: self.id,
                pool: block.pool,
                set: block.set,
            });
        }
        let pool_sizes = signature
            .iter()
            .map(|&(ty, descriptor_count)| {
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::from_raw(ty))
                    .descriptor_count(descriptor_count)
            })
            .collect::<Vec<_>>();
        let pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(ReplacementRecordingError::Driver)?;
        let layouts = [layout];
        let set = match unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe { self.device.destroy_descriptor_pool(pool, None) };
                return Err(ReplacementRecordingError::Driver(error));
            }
        };
        self.descriptor_blocks.push(DescriptorPoolBlock {
            signature,
            layout,
            pool,
            set,
            in_use: true,
        });
        Ok(ReplacementDescriptorAllocation {
            worker: self.id,
            pool,
            set,
        })
    }

    /// Allocate one native recording from this worker's pool for the selected
    /// queue family. The pool is created lazily on its owning worker.
    pub fn allocate(
        &mut self,
        queue_family: u32,
        command_buffer_count: u32,
    ) -> Result<ReplacementNativeRecording, ReplacementRecordingError> {
        if command_buffer_count == 0 {
            return Err(ReplacementRecordingError::NoCommandBuffers);
        }
        let command_pool = match self.command_pools.get(&queue_family).copied() {
            Some(pool) => pool,
            None => {
                let pool = unsafe {
                    self.device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(queue_family)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                }
                .map_err(ReplacementRecordingError::Driver)?;
                self.command_pools.insert(queue_family, pool);
                pool
            }
        };
        let command_buffers = match unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(command_buffer_count),
            )
        } {
            Ok(buffers) => buffers.into_boxed_slice(),
            Err(error) => return Err(ReplacementRecordingError::Driver(error)),
        };
        let fence = match unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                unsafe {
                    self.device
                        .free_command_buffers(command_pool, &command_buffers)
                };
                return Err(ReplacementRecordingError::Driver(error));
            }
        };
        Ok(ReplacementNativeRecording {
            worker: self.id,
            queue_family,
            command_pool,
            command_buffers,
            fence,
            descriptor_sets: Box::new([]),
            framebuffers: Box::new([]),
            query_pools: Box::new([]),
            render_pipeline_variants: Box::new([]),
            indirect_range_programs: Box::new([]),
            host_landing_programs: Box::new([]),
            recorded_operations: Box::new([]),
            resource_completions: Box::new([]),
            backings: Box::new([]),
            representation_leases: Box::new([]),
        })
    }

    /// Record the synchronization preamble for one immutable native
    /// submission. The returned primary command buffer is ended and ready for
    /// queue submission. Any refusal reclaims its pool allocation and fence
    /// before returning.
    pub fn record_hazard_preamble(
        &mut self,
        queue_family: u32,
        hazards: &[reims_vgpu_core::HazardRequirement],
        resolver: &impl ReplacementBarrierResolver,
    ) -> Result<ReplacementNativeRecording, ReplacementRecordingError> {
        let barriers =
            plan_hazard_barriers(hazards).map_err(ReplacementRecordingError::BarrierPlan)?;
        let batch = resolve_hazard_barriers(&barriers, resolver)
            .map_err(ReplacementRecordingError::BarrierResolution)?;
        self.record_barrier_batch(queue_family, &batch)
    }

    /// Record one already-resolved synchronization preamble. Resolution is
    /// deliberately outside the worker: only command-pool mutation and Vulkan
    /// command emission occupy the fixed recording lane.
    pub fn record_barrier_batch(
        &mut self,
        queue_family: u32,
        barriers: &NativeBarrierBatch,
    ) -> Result<ReplacementNativeRecording, ReplacementRecordingError> {
        self.record_barrier_program(queue_family, barriers, &[])
    }

    fn record_image_release(
        &mut self,
        release: &crate::replacement_image_transition::NativeImageRelease,
    ) -> Result<ReplacementNativeRecording, ReplacementRecordingError> {
        self.record_barrier_program(release.source_queue_family, &release.barriers, &[])
    }

    fn record_barrier_program(
        &mut self,
        queue_family: u32,
        preamble: &NativeBarrierBatch,
        operations: &[ReplacementRecordedOperation],
    ) -> Result<ReplacementNativeRecording, ReplacementRecordingError> {
        let mut recording = self.allocate(queue_family, 1)?;
        let compute_fill_count = operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    ReplacementRecordedOperation::BufferBlit {
                        native: NativeBufferBlit::ComputeFill { .. },
                        ..
                    }
                )
            })
            .count();
        let fill_pipeline = if compute_fill_count == 0 {
            None
        } else {
            match self.fill_pipeline {
                Some(pipeline) => Some(pipeline),
                None => match unsafe {
                    crate::replacement_fill::ReplacementFillPipeline::create(&self.device)
                } {
                    Ok(pipeline) => {
                        self.fill_pipeline = Some(pipeline);
                        Some(pipeline)
                    }
                    Err(error) => {
                        self.recycle(recording)
                            .expect("new recording belongs to this worker and pool");
                        return Err(ReplacementRecordingError::Driver(error));
                    }
                },
            }
        };
        if let Some(pipeline) = fill_pipeline {
            for _ in 0..compute_fill_count {
                match self.allocate_descriptor_block(
                    pipeline.descriptor_layout,
                    &[(vk::DescriptorType::STORAGE_BUFFER, 1)],
                ) {
                    Ok(allocation) => {
                        let mut allocations = recording.descriptor_sets.into_vec();
                        allocations.push(allocation);
                        recording.descriptor_sets = allocations.into_boxed_slice();
                    }
                    Err(error) => {
                        self.recycle(recording)
                            .expect("new recording belongs to this worker and pool");
                        return Err(error);
                    }
                }
            }
        }
        let compute_descriptor_count = operations
            .iter()
            .filter(|operation| matches!(operation, ReplacementRecordedOperation::Compute { native, .. } if !native.descriptor_counts.is_empty()))
            .count();
        for native in operations.iter().filter_map(|operation| match operation {
            ReplacementRecordedOperation::Compute { native, .. }
                if !native.descriptor_counts.is_empty() =>
            {
                Some(native)
            }
            _ => None,
        }) {
            match self.allocate_descriptor_block(
                native.pipeline.descriptor_set_layout,
                &native.descriptor_counts,
            ) {
                Ok(allocation) => {
                    let mut allocations = recording.descriptor_sets.into_vec();
                    allocations.push(allocation);
                    recording.descriptor_sets = allocations.into_boxed_slice();
                }
                Err(error) => {
                    self.recycle(recording)
                        .expect("new recording belongs to this worker and pool");
                    return Err(error);
                }
            }
        }
        let visibility_query_count = operations
            .iter()
            .filter(|operation| {
                matches!(operation, ReplacementRecordedOperation::Render { native, .. } if native.visibility.is_some())
            })
            .count();
        if visibility_query_count != 0 {
            let query_count = match u32::try_from(visibility_query_count) {
                Ok(count) => count,
                Err(_) => {
                    self.recycle(recording)
                        .expect("new recording belongs to this worker and pool");
                    return Err(ReplacementRecordingError::TooManyVisibilityQueries);
                }
            };
            match unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::OCCLUSION)
                        .query_count(query_count),
                    None,
                )
            } {
                Ok(pool) => recording.query_pools = Box::new([pool]),
                Err(error) => {
                    self.recycle(recording)
                        .expect("new recording belongs to this worker and pool");
                    return Err(ReplacementRecordingError::Driver(error));
                }
            }
        }
        for native in operations.iter().filter_map(|operation| match operation {
            ReplacementRecordedOperation::Render { native, .. }
                if !native.descriptor_counts.is_empty() =>
            {
                Some(native)
            }
            _ => None,
        }) {
            match self.allocate_descriptor_block(
                native.pipeline.descriptor_set_layout,
                &native.descriptor_counts,
            ) {
                Ok(allocation) => {
                    let mut allocations = recording.descriptor_sets.into_vec();
                    allocations.push(allocation);
                    recording.descriptor_sets = allocations.into_boxed_slice();
                }
                Err(error) => {
                    self.recycle(recording)
                        .expect("new recording belongs to this worker and pool");
                    return Err(error);
                }
            }
        }
        for native in operations.iter().filter_map(|operation| match operation {
            ReplacementRecordedOperation::Render { native, .. } if native.begins_native_pass => {
                Some(native)
            }
            _ => None,
        }) {
            let create = vk::FramebufferCreateInfo::default()
                .render_pass(native.pipeline.render_pass)
                .attachments(&native.attachment_views)
                .width(native.extent.width)
                .height(native.extent.height)
                .layers(1);
            match unsafe { self.device.create_framebuffer(&create, None) } {
                Ok(framebuffer) => {
                    let mut framebuffers = recording.framebuffers.into_vec();
                    framebuffers.push(framebuffer);
                    recording.framebuffers = framebuffers.into_boxed_slice();
                }
                Err(error) => {
                    self.recycle(recording)
                        .expect("new recording belongs to this worker and pool");
                    return Err(ReplacementRecordingError::Driver(error));
                }
            }
        }
        let command_buffer = recording.command_buffers[0];
        let (fill_descriptors, operation_descriptors) =
            recording.descriptor_sets.split_at(compute_fill_count);
        let (compute_descriptors, render_descriptors) =
            operation_descriptors.split_at(compute_descriptor_count);
        let mut fill_descriptors = fill_descriptors.iter();
        let mut compute_descriptors = compute_descriptors.iter();
        let mut render_descriptors = render_descriptors.iter();
        let mut framebuffers = recording.framebuffers.iter().copied();
        let mut active_framebuffer = None;
        let mut render_group_start = None;
        let visibility_pool = recording.query_pools.first().copied();
        let mut next_visibility_query = 0u32;
        let mut pending_visibility = Vec::new();
        let result = unsafe {
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(ReplacementRecordingError::Driver)
                .and_then(|()| {
                    record_hazard_barriers(&self.device, command_buffer, preamble);
                    if let Some(pool) = visibility_pool {
                        self.device.cmd_reset_query_pool(
                            command_buffer,
                            pool,
                            0,
                            u32::try_from(visibility_query_count)
                                .expect("the query pool count was prevalidated"),
                        );
                    }
                    for (operation_index, operation) in operations.iter().enumerate() {
                        match operation {
                            ReplacementRecordedOperation::Barrier { native, .. } => {
                                record_hazard_barriers(&self.device, command_buffer, native);
                            }
                            ReplacementRecordedOperation::BufferBlit { native, .. } => {
                                if matches!(native, NativeBufferBlit::ComputeFill { .. }) {
                                    fill_pipeline
                                        .expect("a compute fill created its fixed pipeline")
                                        .record(
                                            &self.device,
                                            command_buffer,
                                            fill_descriptors
                                                .next()
                                                .expect(
                                                    "one descriptor was allocated per compute fill",
                                                )
                                                .set,
                                            *native,
                                        );
                                } else {
                                    record_buffer_blit(&self.device, command_buffer, *native);
                                }
                            }
                            ReplacementRecordedOperation::ImageBlit { native, .. } => {
                                record_native_image_copies(&self.device, command_buffer, native);
                            }
                            ReplacementRecordedOperation::InfoQuery { native, .. } => {
                                record_info_query(&self.device, command_buffer, native);
                            }
                            ReplacementRecordedOperation::Compute { native, .. } => {
                                let descriptor_set =
                                    (!native.descriptor_counts.is_empty()).then(|| {
                                        compute_descriptors
                                            .next()
                                            .expect(
                                                "one descriptor was allocated per compute dispatch",
                                            )
                                            .set
                                    });
                                record_compute_dispatch(
                                    &self.device,
                                    command_buffer,
                                    descriptor_set,
                                    native,
                                );
                            }
                            ReplacementRecordedOperation::Render { native, .. } => {
                                if native.begins_native_pass {
                                    active_framebuffer = Some(framebuffers.next().expect(
                                        "one framebuffer was allocated per native render pass",
                                    ));
                                    render_group_start = Some(operation_index);
                                    for operation in &operations[operation_index..] {
                                        let ReplacementRecordedOperation::Render { native, .. } =
                                            operation
                                        else {
                                            unreachable!(
                                                "render-pass shape was validated before recording"
                                            )
                                        };
                                        record_hazard_barriers(
                                            &self.device,
                                            command_buffer,
                                            &native.image_state.transitions.before,
                                        );
                                        if native.ends_native_pass {
                                            break;
                                        }
                                    }
                                }
                                let descriptor_set =
                                    (!native.descriptor_counts.is_empty()).then(|| {
                                        render_descriptors
                                            .next()
                                            .expect("one descriptor was allocated per render draw")
                                            .set
                                    });
                                let visibility_query = native.visibility.map(|visibility| {
                                    let query = next_visibility_query;
                                    next_visibility_query += 1;
                                    let pool = visibility_pool.expect(
                                        "a visibility draw allocated one worker query pool",
                                    );
                                    pending_visibility.push((visibility, pool, query));
                                    (pool, query)
                                });
                                record_render_dispatch(
                                    &self.device,
                                    command_buffer,
                                    descriptor_set,
                                    active_framebuffer
                                        .expect("render-pass shape was validated before recording"),
                                    visibility_query,
                                    native,
                                );
                                if native.ends_native_pass {
                                    let start = render_group_start
                                        .take()
                                        .expect("render-pass shape was validated before recording");
                                    let mut group_visibility =
                                        std::mem::take(&mut pending_visibility).into_iter();
                                    for operation in &operations[start..=operation_index] {
                                        let ReplacementRecordedOperation::Render { native, .. } =
                                            operation
                                        else {
                                            unreachable!(
                                                "render-pass shape was validated before recording"
                                            )
                                        };
                                        if native.visibility.is_some() {
                                            let (visibility, pool, query) = group_visibility
                                                .next()
                                                .expect("each visibility draw owns one query");
                                            self.device.cmd_copy_query_pool_results(
                                                command_buffer,
                                                pool,
                                                query,
                                                1,
                                                visibility.buffer,
                                                visibility.offset,
                                                8,
                                                vk::QueryResultFlags::TYPE_64
                                                    | vk::QueryResultFlags::WAIT,
                                            );
                                        }
                                        record_hazard_barriers(
                                            &self.device,
                                            command_buffer,
                                            &native.image_state.transitions.after,
                                        );
                                    }
                                    debug_assert!(group_visibility.next().is_none());
                                    active_framebuffer = None;
                                }
                            }
                            ReplacementRecordedOperation::ResourceState {
                                commands,
                                image_commands,
                                image_state,
                                ..
                            } => {
                                for command in commands {
                                    record_buffer_blit(&self.device, command_buffer, *command);
                                }
                                if let Some(image_state) = image_state {
                                    record_hazard_barriers(
                                        &self.device,
                                        command_buffer,
                                        &image_state.transitions.before,
                                    );
                                }
                                crate::replacement_image_blit::record_native_image_commands(
                                    &self.device,
                                    command_buffer,
                                    image_commands,
                                );
                                if let Some(image_state) = image_state {
                                    record_hazard_barriers(
                                        &self.device,
                                        command_buffer,
                                        &image_state.transitions.after,
                                    );
                                }
                            }
                            ReplacementRecordedOperation::ContentSynchronization {
                                commands,
                                image_commands,
                                ..
                            } => {
                                for command in commands {
                                    record_buffer_blit(&self.device, command_buffer, *command);
                                }
                                crate::replacement_image_blit::record_native_image_commands(
                                    &self.device,
                                    command_buffer,
                                    image_commands,
                                );
                            }
                            ReplacementRecordedOperation::IndirectRange(program) => {
                                record_indirect_range_readback(
                                    &self.device,
                                    command_buffer,
                                    program,
                                );
                            }
                            ReplacementRecordedOperation::Covered(_) => {}
                        }
                    }
                    self.device
                        .end_command_buffer(command_buffer)
                        .map_err(ReplacementRecordingError::Driver)
                })
        };
        match result {
            Ok(()) => Ok(recording),
            Err(error) => {
                self.recycle(recording)
                    .expect("new recording belongs to this worker and pool");
                Err(error)
            }
        }
    }

    /// Reclaim a timeline-retired recording. Validation occurs before any
    /// Vulkan handle is touched, and refusal returns the complete value.
    pub fn recycle(
        &mut self,
        recording: ReplacementNativeRecording,
    ) -> Result<(), ReplacementRecordingRecycleFailure> {
        let mut descriptor_pools = BTreeSet::new();
        let reason = if recording.worker != self.id {
            Some(ReplacementRecordingRecycleError::WrongWorker)
        } else if self.command_pools.get(&recording.queue_family).copied()
            != Some(recording.command_pool)
        {
            Some(ReplacementRecordingRecycleError::UnknownCommandPool)
        } else {
            recording.descriptor_sets.iter().find_map(|allocation| {
                if allocation.worker != self.id
                    || !descriptor_pools.insert(allocation.pool)
                    || !self.descriptor_blocks.iter().any(|block| {
                        block.pool == allocation.pool && block.set == allocation.set && block.in_use
                    })
                {
                    Some(ReplacementRecordingRecycleError::UnknownDescriptorPool)
                } else {
                    None
                }
            })
        };
        if let Some(reason) = reason {
            return Err(ReplacementRecordingRecycleFailure {
                reason,
                recording: Box::new(recording),
            });
        }
        unsafe {
            for framebuffer in &recording.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            for query_pool in &recording.query_pools {
                self.device.destroy_query_pool(*query_pool, None);
            }
            self.device
                .free_command_buffers(recording.command_pool, &recording.command_buffers);
            self.device.destroy_fence(recording.fence, None);
        }
        for allocation in &recording.descriptor_sets {
            self.descriptor_blocks
                .iter_mut()
                .find(|block| block.pool == allocation.pool)
                .expect("descriptor ownership was prevalidated")
                .in_use = false;
        }
        Ok(())
    }
}

impl Drop for ReplacementRecordingWorker {
    fn drop(&mut self) {
        unsafe {
            for (_, pool) in std::mem::take(&mut self.command_pools) {
                self.device.destroy_command_pool(pool, None);
            }
            for block in self.descriptor_blocks.drain(..) {
                self.device.destroy_descriptor_pool(block.pool, None);
            }
            if let Some(pipeline) = self.fill_pipeline.take() {
                pipeline.destroy(&self.device);
            }
        }
    }
}

fn descriptor_signature(
    counts: &[(vk::DescriptorType, u32)],
) -> Result<Box<[(i32, u32)]>, ReplacementRecordingError> {
    if counts.is_empty() || counts.iter().any(|(_, count)| *count == 0) {
        return Err(ReplacementRecordingError::InvalidDescriptorRequest);
    }
    let mut signature = BTreeMap::<i32, u32>::new();
    for &(ty, count) in counts {
        let total = signature.entry(ty.as_raw()).or_default();
        *total = total
            .checked_add(count)
            .ok_or(ReplacementRecordingError::InvalidDescriptorRequest)?;
    }
    Ok(signature.into_iter().collect::<Vec<_>>().into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::context::DeviceContext;
    use ash::vk::Handle;
    use reims_vgpu_core::BackingView;
    use reims_vgpu_core::{
        admit_indirect_commands, cancel_prepared_buffer_blit, prepare_info_query, AccessIntent,
        AccessMode, BackingRegion, BarrierOperation, BufferFillPattern, ContractDisposition,
        EncoderBoundary, HazardCause, HazardEdge, HazardRequirement, LinearRange,
        MemoryBarrierScope, ParticipationOperation, ParticipationScope, RefusalClosureLedger,
        RepresentationRoute, ResolvedBlit, ResolvedBufferRange, ResolvedBufferToTextureBlit,
        ResolvedExecSegment, ResolvedExecStream, ResolvedIndirectCommand,
        ResolvedIndirectCommandRange, ResolvedInfoOperation, ResolvedInfoReplyTarget,
        ResolvedLinearTextureLevel, ResolvedResourceLifecycle, ResolvedResourceState,
        ResolvedTextureBacking, ResolvedTextureEndpoint, ResolvedTextureToTextureBlit,
        ResourceLifecycleEffect, ResourceLifecycleOwner, StageScope, StorageBacking, TextureExtent,
        TextureOrigin,
    };
    use reims_vgpu_protocol::{
        ByteLength, ChannelId, GuestVirtualAddress, HazardDomainId, HeapObject, IngressOrdinal,
        QueueOwnerId, QueueTimelineValue, RenderPipelineObject, ResourceId, SegmentBoundary,
        SegmentKind, SessionGenerationId, SubmissionId, SubmissionIdentity, TaskId, TransactionId,
        VulkanDeviceEpochId,
    };
    use std::convert::Infallible;

    type TestOperation = ResolvedOperation<(), (), (), (), ()>;
    type IndirectTestOperation = ResolvedOperation<(), (), (), ResolvedIndirectCommand, ()>;
    type InfoTestOperation = ResolvedOperation<(), (), ResolvedInfoOperation, (), ()>;

    #[test]
    fn every_resolved_operation_family_has_one_closed_recording_disposition() {
        // `Infallible` as the refusal type is the assertion: no semantic
        // operation family is refused by the native recorder, so no value can
        // be constructed to record one.
        let mut ledger: RefusalClosureLedger<OperationKind, Infallible> =
            RefusalClosureLedger::new(OperationKind::ALL).unwrap();
        for kind in OperationKind::ALL {
            let disposition = match replacement_operation_recording(kind) {
                ReplacementOperationRecording::HazardPlan
                | ReplacementOperationRecording::ExplicitBarrier => {
                    ContractDisposition::Implemented
                }
                ReplacementOperationRecording::SemanticNoNativeCommand(proven) => {
                    assert_eq!(proven, kind);
                    ContractDisposition::ProvenNoOp
                }
                ReplacementOperationRecording::PreparedNativeEmitter(implemented) => {
                    assert_eq!(implemented, kind);
                    ContractDisposition::Implemented
                }
                ReplacementOperationRecording::PreparedSemanticCondition(implemented) => {
                    assert_eq!(implemented, kind);
                    ContractDisposition::Implemented
                }
                ReplacementOperationRecording::PreparedSemanticCompletion => {
                    assert_eq!(kind, OperationKind::CompletionEffect);
                    ContractDisposition::Implemented
                }
                ReplacementOperationRecording::PreparedSemanticIndirect => {
                    assert_eq!(kind, OperationKind::IndirectCommand);
                    ContractDisposition::Implemented
                }
                ReplacementOperationRecording::PreparedNativeResourceState => {
                    assert_eq!(kind, OperationKind::ResourceState);
                    ContractDisposition::Implemented
                }
            };
            ledger.record(kind, disposition).unwrap();
        }

        let counts = ledger.audit().unwrap();
        assert_eq!(counts.implemented, OperationKind::ALL.len() - 1);
        assert_eq!(counts.proven_no_op, 1);
        assert_eq!(counts.unsupported, 0);
    }

    #[test]
    fn every_blit_variant_names_its_remaining_native_prerequisite() {
        let mut seen = BTreeSet::new();
        let mut implemented = 0;
        for kind in BlitKind::ALL {
            assert!(seen.insert(kind));
            match replacement_blit_recording(kind) {
                ReplacementBlitRecording::NativeEmitterImplemented(implemented_kind) => {
                    assert_eq!(implemented_kind, kind);
                    implemented += 1;
                }
            }
        }
        assert_eq!(seen.len(), BlitKind::ALL.len());
        assert_eq!(implemented, 6);
    }

    #[test]
    fn compute_recording_requires_the_exact_positioned_prepared_program() {
        type Operation = ResolvedOperation<(), u32, (), (), ()>;
        let exec = |compute| ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(4),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
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
                    operations: Box::new([Operation::Compute(compute)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let input = || ReplacementRecordingInput {
            transaction: TransactionId::new(7),
            worker: RecordingWorkerId::new(0),
            queue_family: 2,
            exec: exec(19),
            barriers: NativeBarrierBatch::default(),
        };
        let missing = ReplacementRecordingRequest::resolve_with_compute_programs(
            input(),
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions::default(),
            None,
            &[],
        )
        .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::ComputePreparationRequired(0)
        );

        let program = ReplacementComputeProgram::synthetic(
            0,
            TransactionId::new(7),
            19,
            NativeComputeDispatch {
                pipeline: std::sync::Arc::new(
                    crate::replacement_compute::ReplacementComputePipelineVariant::synthetic(
                        crate::replacement_compute::ReplacementComputePipeline {
                            pipeline: vk::Pipeline::from_raw(1),
                            layout: vk::PipelineLayout::from_raw(2),
                            descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                            thread_grid_push_offset: Some(0),
                        },
                    ),
                ),
                wait_for_prior_commands: false,
                descriptors: Box::new([]),
                sampler_leases: Box::new([]),
                descriptor_counts: Box::new([]),
                launch: crate::replacement_compute::NativeComputeLaunch::Direct {
                    thread_grid: [4, 2, 1],
                    workgroups: [1, 1, 1],
                },
                image_state: None,
            },
            vec![BackingId::new(5)].into_boxed_slice(),
            Box::<[ResolvedResourceCompletion]>::default(),
        );
        let request = ReplacementRecordingRequest::resolve_with_compute_programs(
            input(),
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions::default(),
            None,
            &[program],
        )
        .unwrap();
        assert_eq!(request.required_queue_flags(), vk::QueueFlags::COMPUTE);
        assert_eq!(request.backings.as_ref(), [BackingId::new(5)]);
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::Compute { native, .. }]
                if native.launch == crate::replacement_compute::NativeComputeLaunch::Direct {
                    thread_grid: [4, 2, 1],
                    workgroups: [1, 1, 1],
                }
        ));

        let mismatch = ReplacementComputeProgram::synthetic(
            0,
            TransactionId::new(7),
            20,
            NativeComputeDispatch {
                pipeline: std::sync::Arc::new(
                    crate::replacement_compute::ReplacementComputePipelineVariant::synthetic(
                        crate::replacement_compute::ReplacementComputePipeline {
                            pipeline: vk::Pipeline::from_raw(1),
                            layout: vk::PipelineLayout::from_raw(2),
                            descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                            thread_grid_push_offset: Some(0),
                        },
                    ),
                ),
                wait_for_prior_commands: false,
                descriptors: Box::new([]),
                sampler_leases: Box::new([]),
                descriptor_counts: Box::new([]),
                launch: crate::replacement_compute::NativeComputeLaunch::Direct {
                    thread_grid: [4, 2, 1],
                    workgroups: [1, 1, 1],
                },
                image_state: None,
            },
            Box::<[BackingId]>::default(),
            Box::<[ResolvedResourceCompletion]>::default(),
        );
        assert_eq!(
            ReplacementRecordingRequest::resolve_with_compute_programs(
                input(),
                &EmptyResolver,
                &EmptyResolver,
                &[],
                &[],
                ReplacementSemanticAdmissions::default(),
                None,
                &[mismatch],
            )
            .unwrap_err()
            .reason,
            ReplacementRecordingError::ComputeProgramMismatch(0)
        );
    }

    #[test]
    fn render_recording_requires_the_exact_positioned_prepared_program() {
        type Operation = ResolvedOperation<i32, (), (), (), ()>;
        let exec = |render| ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(23),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Render,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([Operation::Render(render)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let input = || ReplacementRecordingInput {
            transaction: TransactionId::new(9),
            worker: RecordingWorkerId::new(0),
            queue_family: 2,
            exec: exec(31),
            barriers: NativeBarrierBatch::default(),
        };
        let native = crate::replacement_render::NativeRenderDispatch {
            pipeline: std::sync::Arc::new(
                crate::replacement_render::ReplacementRenderPipelineVariant::synthetic(
                    crate::replacement_render::ReplacementRenderPipeline {
                        pipeline: vk::Pipeline::from_raw(1),
                        layout: vk::PipelineLayout::from_raw(2),
                        descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                        render_pass: vk::RenderPass::from_raw(4),
                        program: Default::default(),
                        vertex_buffers: Box::new([]),
                        color_attachments: Box::new([]),
                        depth_stencil_attachment: None,
                        feedback_loop_aspects: vk::ImageAspectFlags::empty(),
                        color_input: false,
                        sample_count: vk::SampleCountFlags::TYPE_1,
                        viewport_count: 1,
                        static_state:
                            crate::replacement_render::ReplacementRenderStaticState::default(),
                        dynamic_states: Default::default(),
                        depth_stencil: None,
                    },
                ),
            ),
            descriptors: Box::new([]),
            sampler_leases: Box::new([]),
            descriptor_counts: Box::new([]),
            vertex_buffers: Box::new([]),
            index_buffer: None,
            indirect_buffer: None,
            attachment_views: Box::new([]),
            clear_values: Box::new([]),
            extent: vk::Extent2D {
                width: 4,
                height: 4,
            },
            viewports: Box::new([vk::Viewport {
                x: 0.0,
                y: 4.0,
                width: 4.0,
                height: -4.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }]),
            scissors: Box::new([vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 4,
                    height: 4,
                },
            }]),
            depth_bias: None,
            blend_color: None,
            stencil_reference: [0, 0],
            visibility: None,
            begins_native_pass: true,
            ends_native_pass: true,
            draw: reims_vgpu_core::ResolvedRenderDraw::Direct {
                topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            image_state: crate::replacement_image_transition::PreparedNativeImageState {
                transaction: TransactionId::new(9),
                destination_queue_family: 2,
                releases: Box::new([]),
                transitions: crate::replacement_image_transition::NativeImageUseTransitions {
                    before: NativeBarrierBatch::default(),
                    after: NativeBarrierBatch::default(),
                },
            },
        };
        let render_op = |native| ReplacementRecordedOperation::Render {
            native: Box::new(native),
            completions: Box::new([]),
        };
        let mut first = native.clone();
        first.ends_native_pass = false;
        let mut last = native.clone();
        last.begins_native_pass = false;
        assert_eq!(
            validate_render_passes(&[render_op(first.clone()), render_op(last.clone())]),
            Ok(())
        );
        assert_eq!(
            validate_render_passes(&[render_op(last.clone())]),
            Err(ReplacementRecordingError::RenderPassMustBegin(0))
        );
        assert_eq!(
            validate_render_passes(&[
                render_op(first.clone()),
                ReplacementRecordedOperation::Covered(OperationKind::Barrier),
            ]),
            Err(ReplacementRecordingError::RenderPassInterrupted(1))
        );
        let mut mismatch = last;
        mismatch.extent.width += 1;
        assert_eq!(
            validate_render_passes(&[render_op(first.clone()), render_op(mismatch)]),
            Err(ReplacementRecordingError::RenderPassMismatch(1))
        );
        assert_eq!(
            validate_render_passes(&[render_op(first)]),
            Err(ReplacementRecordingError::RenderPassUnterminated)
        );
        let program = ReplacementRenderProgram::synthetic(
            0,
            TransactionId::new(9),
            31,
            native,
            vec![BackingId::new(8)].into_boxed_slice(),
            Box::<[ResolvedResourceCompletion]>::default(),
        );
        let request = ReplacementRecordingRequest::resolve_with_native_programs(
            input(),
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions::default(),
            None,
            &[],
            &[program],
        )
        .unwrap();
        assert_eq!(request.required_queue_flags(), vk::QueueFlags::GRAPHICS);
        assert_eq!(request.backings.as_ref(), [BackingId::new(8)]);
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::Render { native, .. }]
                if native.extent.width == 4
        ));
        assert_eq!(
            ReplacementRecordingRequest::resolve_with_native_programs(
                input(),
                &EmptyResolver,
                &EmptyResolver,
                &[],
                &[],
                ReplacementSemanticAdmissions::default(),
                None,
                &[],
                &[],
            )
            .unwrap_err()
            .reason,
            ReplacementRecordingError::RenderPreparationRequired(0)
        );
    }

    fn empty_exec(id: u64) -> ExecTransaction<TestOperation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        }
    }

    fn indirect_exec(
        id: u64,
        command: ResolvedIndirectCommand,
    ) -> ExecTransaction<IndirectTestOperation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::IndirectCommand(command)]),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    #[test]
    fn indirect_optimize_requires_the_exact_admission_proof() {
        let transaction = TransactionId::new(73);
        let command = ResolvedIndirectCommand::Optimize {
            icb: ResourceId::new(8, 2),
            range: ResolvedIndirectCommandRange {
                location: 3,
                length: 5,
            },
        };
        let exec = indirect_exec(73, command);
        let admitted = admit_indirect_commands(transaction, &exec).unwrap();
        let input = ReplacementRecordingInput {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: 0,
            exec,
            barriers: NativeBarrierBatch::default(),
        };
        let missing = ReplacementRecordingRequest::resolve(input, &EmptyResolver, &EmptyResolver)
            .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::IndirectAdmissionRequired(0)
        );

        let mut altered = missing.input.clone();
        let ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Optimize { range, .. }) =
            &mut altered.exec.streams[0].segments[0].operations[0]
        else {
            unreachable!()
        };
        range.length = 6;
        let mismatch = ReplacementRecordingRequest::resolve_with_all_semantics(
            altered,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: None,
                completion_effects: None,
                indirect_commands: Some(&admitted),
                resource_states: None,
                info_queries: &[],
                indirect_range_programs: &[],
            },
        )
        .unwrap_err();
        assert_eq!(
            mismatch.reason,
            ReplacementRecordingError::IndirectAdmissionMismatch(0)
        );

        let request = ReplacementRecordingRequest::resolve_with_all_semantics(
            missing.input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: None,
                completion_effects: None,
                indirect_commands: Some(&admitted),
                resource_states: None,
                info_queries: &[],
                indirect_range_programs: &[],
            },
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::Covered(
                OperationKind::IndirectCommand
            )]
        ));
    }

    #[test]
    fn indirect_execution_cannot_be_erased_by_a_semantic_mutation_proof() {
        let transaction = TransactionId::new(74);
        let icb = ResourceId::new(9, 2);
        let command = ResolvedIndirectCommand::Execute {
            icb,
            range: ResolvedIndirectCommandRange {
                location: 0,
                length: 1,
            },
            kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
        };
        let exec = indirect_exec(74, command);
        let mut owner = reims_vgpu_core::IndirectCommandSlotOwner::<()>::default();
        owner.register(icb, 1).unwrap();
        let admitted =
            reims_vgpu_core::admit_indirect_commands_with_owner(transaction, &exec, &owner)
                .unwrap();
        let input = ReplacementRecordingInput {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: 0,
            exec,
            barriers: NativeBarrierBatch::default(),
        };
        assert_eq!(
            ReplacementRecordingRequest::resolve_with_all_semantics(
                input,
                &EmptyResolver,
                &EmptyResolver,
                &[],
                &[],
                ReplacementSemanticAdmissions {
                    conditions: None,
                    completion_effects: None,
                    indirect_commands: Some(&admitted),
                    resource_states: None,
                    info_queries: &[],
                    indirect_range_programs: &[],
                },
            )
            .unwrap_err()
            .reason,
            ReplacementRecordingError::IndirectExecutionPreparationRequired(0)
        );
    }

    #[test]
    fn continuation_recording_matches_sidecars_at_the_original_operation_position() {
        let transaction = TransactionId::new(741);
        let command = ResolvedIndirectCommand::Optimize {
            icb: ResourceId::new(9, 2),
            range: ResolvedIndirectCommandRange {
                location: 0,
                length: 1,
            },
        };
        let full = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(741),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([
                        IndirectTestOperation::IndirectCommand(command),
                        IndirectTestOperation::IndirectCommand(command),
                        IndirectTestOperation::IndirectCommand(command),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let admitted = admit_indirect_commands(transaction, &full)
            .unwrap()
            .for_operation_range(2, 3);
        let input = ReplacementRecordingInput {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: 0,
            exec: indirect_exec(741, command),
            barriers: NativeBarrierBatch::default(),
        };
        let request = ReplacementRecordingRequest::resolve_continuation_with_native_programs(
            input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: None,
                completion_effects: None,
                indirect_commands: Some(&admitted),
                resource_states: None,
                info_queries: &[],
                indirect_range_programs: &[],
            },
            None,
            &[],
            &[],
            2,
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::Covered(
                OperationKind::IndirectCommand
            )]
        ));
    }

    #[test]
    fn expanded_slots_share_one_semantic_origin_but_keep_unique_recording_positions() {
        let transaction = TransactionId::new(742);
        let exec: ExecTransaction<TestOperation> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(742),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
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
                    operations: Box::new([
                        ResolvedOperation::EncoderBoundary(EncoderBoundary::Begin(
                            SegmentKind::Compute,
                        )),
                        ResolvedOperation::EncoderBoundary(EncoderBoundary::End(
                            SegmentKind::Compute,
                        )),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let request =
            ReplacementRecordingRequest::resolve_continuation_with_native_programs_at_positions(
                ReplacementRecordingInput {
                    transaction,
                    worker: RecordingWorkerId::new(0),
                    queue_family: 0,
                    exec,
                    barriers: NativeBarrierBatch::default(),
                },
                &EmptyResolver,
                &EmptyResolver,
                &[],
                &[],
                ReplacementSemanticAdmissions::default(),
                None,
                &[],
                &[],
                &[
                    ExpandedIndirectOperationOrigin {
                        expanded_position: 8,
                        original_position: 3,
                        indirect_slot: Some(0),
                    },
                    ExpandedIndirectOperationOrigin {
                        expanded_position: 9,
                        original_position: 3,
                        indirect_slot: Some(1),
                    },
                ],
            )
            .unwrap();
        assert_eq!(request.operations.len(), 2);
    }

    #[test]
    fn expanded_indirect_render_command_enters_the_exact_native_render_gate() {
        type Operation = ResolvedOperation<i32, (), (), ResolvedIndirectCommand, ()>;
        let transaction = TransactionId::new(75);
        let icb = ResourceId::new(10, 2);
        let command = ResolvedIndirectCommand::Execute {
            icb,
            range: ResolvedIndirectCommandRange {
                location: 0,
                length: 1,
            },
            kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
        };
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(75),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Render,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([Operation::IndirectCommand(command)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let mut owner = reims_vgpu_core::IndirectCommandSlotOwner::default();
        owner.register(icb, 1).unwrap();
        owner
            .set(
                icb,
                0,
                reims_vgpu_core::ResolvedIndirectCommandSlot::<i32, ()>::Render(31),
            )
            .unwrap();
        let admitted =
            reims_vgpu_core::admit_indirect_commands_with_owner(transaction, &exec, &owner)
                .unwrap();
        let expanded =
            reims_vgpu_core::prepare_indirect_execution(&mut owner, transaction, &exec, admitted)
                .unwrap()
                .into_exec();
        let failure = ReplacementRecordingRequest::resolve(
            ReplacementRecordingInput {
                transaction,
                worker: RecordingWorkerId::new(0),
                queue_family: 0,
                exec: expanded,
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementRecordingError::RenderPreparationRequired(0)
        );
    }

    #[test]
    fn resource_state_requires_exact_admission_and_keeps_possible_transfers_native() {
        use reims_vgpu_core::ResolvedResourceState;
        use reims_vgpu_protocol::ResourceValidityOps;

        let semantic = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([]),
            ops: ResourceValidityOps::PAGE_ON,
        };
        let exec: ExecTransaction<TestOperation> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(74),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::ResourceState(semantic.clone())]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let mut runtime = reims_vgpu_core::TransactionRuntime::<()>::new(
            reims_vgpu_core::SessionGeneration::new(SessionGenerationId::new(1)),
        );
        let channel = ChannelId::new(5);
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 1)),
                exec,
            )
            .unwrap();
        let (envelope, conditions, effects, resource_states, _) = admitted.into_parts();
        let reims_vgpu_core::DeviceTransactionPayload::Exec(exec) = envelope.payload else {
            unreachable!()
        };
        let input = ReplacementRecordingInput {
            transaction: envelope.id,
            worker: RecordingWorkerId::new(0),
            queue_family: 0,
            exec,
            barriers: NativeBarrierBatch::default(),
        };
        let missing = ReplacementRecordingRequest::resolve(input, &EmptyResolver, &EmptyResolver)
            .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::ResourceStateAdmissionRequired(0)
        );
        let semantic_unprepared = ReplacementRecordingRequest::resolve_with_all_semantics(
            missing.input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: Some(&conditions),
                completion_effects: Some(&effects),
                indirect_commands: None,
                resource_states: Some(&resource_states),
                info_queries: &[],
                indirect_range_programs: &[],
            },
        )
        .unwrap_err();
        assert_eq!(
            semantic_unprepared.reason,
            ReplacementRecordingError::ResourceStatePreparationRequired(0)
        );

        let transfer_region = BackingRegion::Linear(LinearRange::new(8, 16).unwrap());
        let mut transfer_resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(transfer_backing) = transfer_resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([transfer_region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = transfer_resources
            .create_representation(
                transfer_backing,
                RepresentationRoute::HostVisibleWorking,
                (),
            )
            .unwrap();
        assert_eq!(
            transfer_resources
                .create_representation(transfer_backing, RepresentationRoute::DirectGuestAlias, (),)
                .unwrap(),
            reims_vgpu_core::GUEST_REPRESENTATION
        );
        transfer_resources
            .plan_gpu_write(
                transfer_backing,
                SubmissionId::new(1),
                source,
                [transfer_region],
            )
            .unwrap();
        transfer_resources
            .complete_gpu_write(transfer_backing, SubmissionId::new(1), source)
            .unwrap();
        let transfer_state = ResolvedResourceState {
            ops: ResourceValidityOps {
                clear_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
            targets: Box::new([reims_vgpu_core::ResolvedResourceStateTarget {
                backing: transfer_backing,
                regions: Box::new([transfer_region]),
            }]),
            ..semantic
        };
        let transfer_exec: ExecTransaction<TestOperation> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(75),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::ResourceState(
                        transfer_state.clone(),
                    )]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 2)),
                transfer_exec,
            )
            .unwrap();
        let (envelope, conditions, effects, states, _) = admitted.into_parts();
        let reims_vgpu_core::DeviceTransactionPayload::Exec(exec) = envelope.payload else {
            unreachable!()
        };
        let failure = ReplacementRecordingRequest::resolve_with_all_semantics(
            ReplacementRecordingInput {
                transaction: envelope.id,
                worker: RecordingWorkerId::new(0),
                queue_family: 0,
                exec,
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: Some(&conditions),
                completion_effects: Some(&effects),
                indirect_commands: None,
                resource_states: Some(&states),
                info_queries: &[],
                indirect_range_programs: &[],
            },
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementRecordingError::ResourceStateNativeTransferRequired(0)
        );

        struct TransferResolver {
            backing: reims_vgpu_protocol::BackingId,
            source: reims_vgpu_protocol::RepresentationId,
            destination: reims_vgpu_protocol::RepresentationId,
        }
        impl crate::replacement_buffer_blit::ReplacementBufferResolver for TransferResolver {
            fn resolve_buffer(
                &self,
                backing: reims_vgpu_protocol::BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<crate::replacement_buffer_blit::NativeBufferTarget> {
                if backing != self.backing {
                    return None;
                }
                let (buffer, base_offset) = if representation == self.source {
                    (vk::Buffer::from_raw(31), 100)
                } else if representation == self.destination {
                    (vk::Buffer::from_raw(32), 200)
                } else {
                    return None;
                };
                Some(crate::replacement_buffer_blit::NativeBufferTarget {
                    buffer,
                    base_offset,
                    accessible_size: 64,
                    size: 64,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                })
            }
        }
        let destination = reims_vgpu_core::GUEST_REPRESENTATION;
        let prepared = reims_vgpu_core::prepare_resource_state(
            &mut transfer_resources,
            &states,
            0,
            SubmissionId::new(75),
            |_, _| reims_vgpu_core::ValidityRepresentations {
                host_write: None,
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: Some(source),
                guest_visibility_destination: reims_vgpu_core::GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let transfer = prepared.transfers()[0];
        let prepared_batch =
            reims_vgpu_core::assemble_prepared_resource_states(&states, vec![prepared]).unwrap();
        let programs = ReplacementResourceStateBatchProgram::resolve(
            &prepared_batch,
            &TransferResolver {
                backing: transfer_backing,
                source,
                destination,
            },
        )
        .unwrap();
        let program = &programs.programs()[0];
        let mut recorded = BTreeSet::new();
        let mut shared_completions = Vec::new();
        let first =
            retain_unique_resource_transfers(program, &mut recorded, &mut shared_completions);
        assert_eq!(first.0.len() + first.1.len(), 1);
        let repeated =
            retain_unique_resource_transfers(program, &mut recorded, &mut shared_completions);
        assert!(repeated.0.is_empty() && repeated.1.is_empty());
        assert_eq!(
            shared_completions,
            [ResolvedResourceCompletion::Transfer(transfer)]
        );
        let request = ReplacementRecordingRequest::resolve_with_resource_state_programs(
            failure.input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: Some(&conditions),
                completion_effects: Some(&effects),
                indirect_commands: None,
                resource_states: Some(&states),
                info_queries: &[],
                indirect_range_programs: &[],
            },
            Some(&programs),
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::ResourceState {
                completions,
                commands,
                ..
            }]
                if completions.as_ref() == [ResolvedResourceCompletion::Transfer(transfer)]
                    && matches!(commands.as_ref(), [NativeBufferBlit::Copy {
                        source: found_source,
                        destination: found_destination,
                        source_offset: 108,
                        destination_offset: 208,
                        size: 16,
                    }] if *found_source == vk::Buffer::from_raw(31)
                        && *found_destination == vk::Buffer::from_raw(32))
        ));
    }

    #[test]
    fn admitted_event_and_fence_operations_are_consumed_only_with_the_exact_proof() {
        let channel = ChannelId::new(3);
        let event = ResourceId::new(8, 2);
        let fence = ResourceId::new(9, 2);
        let exec: ExecTransaction<TestOperation> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(72),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Event,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([
                        ResolvedOperation::Event(reims_vgpu_core::EventOperation {
                            event,
                            kind: reims_vgpu_core::EventOperationKind::Signal,
                            value: 4,
                        }),
                        ResolvedOperation::Fence(reims_vgpu_core::FenceOperation {
                            fence,
                            kind: reims_vgpu_core::FenceOperationKind::Update,
                            generation: 1,
                            scope: reims_vgpu_core::FenceScope::Compute,
                        }),
                        ResolvedOperation::Fence(reims_vgpu_core::FenceOperation {
                            fence,
                            kind: reims_vgpu_core::FenceOperationKind::Wait,
                            generation: 1,
                            scope: reims_vgpu_core::FenceScope::Render(
                                reims_vgpu_core::RenderBarrierStages::FRAGMENT,
                            ),
                        }),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let mut runtime = reims_vgpu_core::TransactionRuntime::<()>::new(
            reims_vgpu_core::SessionGeneration::new(SessionGenerationId::new(1)),
        );
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 1)),
                exec,
            )
            .unwrap();
        let (transaction, conditions, _completion_effects, _resource_states, _) =
            admitted.into_parts();
        let transaction_id = transaction.id;
        let reims_vgpu_core::DeviceTransactionPayload::Exec(exec) = transaction.payload else {
            unreachable!()
        };
        let missing = ReplacementRecordingRequest::resolve(
            ReplacementRecordingInput {
                transaction: transaction_id,
                worker: RecordingWorkerId::new(0),
                queue_family: 0,
                exec,
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::ConditionAdmissionRequired {
                index: 0,
                kind: OperationKind::Event,
            }
        );
        let mut altered = missing.input.clone();
        let ResolvedOperation::Event(event) =
            &mut altered.exec.streams[0].segments[0].operations[0]
        else {
            unreachable!()
        };
        event.value = 5;
        let mismatch = ReplacementRecordingRequest::resolve_with_blits_and_conditions(
            altered,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
        )
        .unwrap_err();
        assert_eq!(
            mismatch.reason,
            ReplacementRecordingError::ConditionAdmissionMismatch(0)
        );
        let mut wrong_transaction = missing.input.clone();
        wrong_transaction.transaction = TransactionId::new(transaction_id.get() + 1);
        let mismatch = ReplacementRecordingRequest::resolve_with_blits_and_conditions(
            wrong_transaction,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
        )
        .unwrap_err();
        assert_eq!(
            mismatch.reason,
            ReplacementRecordingError::ConditionAdmissionTransactionMismatch
        );
        let mut removed = missing.input.clone();
        removed.exec.streams[0].segments[0].operations =
            Box::new([ResolvedOperation::EncoderBoundary(EncoderBoundary::Begin(
                SegmentKind::Event,
            ))]);
        let unexpected = ReplacementRecordingRequest::resolve_with_blits_and_conditions(
            removed,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
        )
        .unwrap_err();
        assert_eq!(
            unexpected.reason,
            ReplacementRecordingError::UnexpectedConditionAdmission(0)
        );
        let request = ReplacementRecordingRequest::resolve_with_blits_and_conditions(
            missing.input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [
                ReplacementRecordedOperation::Covered(OperationKind::Event),
                ReplacementRecordedOperation::Covered(OperationKind::Fence),
                ReplacementRecordedOperation::Barrier {
                    kind: OperationKind::Fence,
                    native,
                },
            ]
                if native.memory.len() == 1
                    && native.memory[0].src_stage_mask == vk::PipelineStageFlags2::COMPUTE_SHADER
                    && native.memory[0].dst_stage_mask
                        == vk::PipelineStageFlags2::FRAGMENT_SHADER
        ));
    }

    #[test]
    fn completion_effects_require_the_exact_admission_and_remain_in_the_receipt() {
        type EffectOperation = ResolvedOperation<(), (), (), (), u32>;
        let channel = ChannelId::new(4);
        let exec: ExecTransaction<EffectOperation> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(73),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([
                        ResolvedOperation::CompletionEffect(7),
                        ResolvedOperation::CompletionEffect(9),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let mut runtime = reims_vgpu_core::TransactionRuntime::<()>::new(
            reims_vgpu_core::SessionGeneration::new(SessionGenerationId::new(1)),
        );
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 1)),
                exec,
            )
            .unwrap();
        let (transaction, conditions, effects, _resource_states, _) = admitted.into_parts();
        let transaction_id = transaction.id;
        let reims_vgpu_core::DeviceTransactionPayload::Exec(exec) = transaction.payload else {
            unreachable!()
        };
        let input = ReplacementRecordingInput {
            transaction: transaction_id,
            worker: RecordingWorkerId::new(0),
            queue_family: 0,
            exec,
            barriers: NativeBarrierBatch::default(),
        };
        let missing = ReplacementRecordingRequest::resolve(input, &EmptyResolver, &EmptyResolver)
            .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::CompletionAdmissionRequired(0)
        );

        let mut altered = missing.input.clone();
        altered.exec.streams[0].segments[0].operations[0] = ResolvedOperation::CompletionEffect(8);
        let mismatch = ReplacementRecordingRequest::resolve_with_semantics(
            altered,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
            Some(&effects),
        )
        .unwrap_err();
        assert_eq!(
            mismatch.reason,
            ReplacementRecordingError::CompletionAdmissionMismatch(0)
        );

        let mut removed = missing.input.clone();
        removed.exec.streams[0].segments[0].operations = Box::new([]);
        let unexpected = ReplacementRecordingRequest::resolve_with_semantics(
            removed,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
            Some(&effects),
        )
        .unwrap_err();
        assert_eq!(
            unexpected.reason,
            ReplacementRecordingError::UnexpectedCompletionAdmission(0)
        );

        let mut wrong_transaction = missing.input.clone();
        wrong_transaction.transaction = TransactionId::new(transaction_id.get() + 1);
        let wrong = ReplacementRecordingRequest::resolve_with_semantics(
            wrong_transaction,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            None,
            Some(&effects),
        )
        .unwrap_err();
        assert_eq!(
            wrong.reason,
            ReplacementRecordingError::CompletionAdmissionTransactionMismatch
        );

        let request = ReplacementRecordingRequest::resolve_with_semantics(
            missing.input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            Some(&conditions),
            Some(&effects),
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [
                ReplacementRecordedOperation::Covered(OperationKind::CompletionEffect),
                ReplacementRecordedOperation::Covered(OperationKind::CompletionEffect),
            ]
        ));
        assert_eq!(effects.effects().copied().collect::<Vec<_>>(), [7, 9]);
    }

    #[test]
    fn info_query_program_requires_exact_operation_and_transaction() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 64).unwrap())]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let representation = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let operation = ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::<RenderPipelineObject>::new(2, 1),
            reply: ResolvedInfoReplyTarget {
                resource: ResourceId::new(3, 1),
                backing,
                range: LinearRange::new(8, 12).unwrap(),
                requested_alignment: 4,
            },
        };
        let evaluated = crate::replacement_info_query::tests::evaluated_render_query(
            operation,
            reims_vgpu_core::RenderPipelineStateInfo {
                max_total_threads_per_threadgroup: 0x3c3c_3c3c,
                imageblock_sample_length: 0x3c3c_3c3c,
                threadgroup_size_matches_tile_size: false,
                supports_indirect_command_buffers: false,
            },
        );
        let transaction = evaluated.transaction();
        let prepared =
            prepare_info_query(&mut resources, SubmissionId::new(76), evaluated).unwrap();
        let resolver = BufferResolver {
            backing,
            representation,
            target: crate::replacement_buffer_blit::NativeBufferTarget {
                buffer: vk::Buffer::null(),
                base_offset: 32,
                accessible_size: 64,
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
            },
            compute_fill_limits: None,
        };
        let program = ReplacementInfoQueryProgram::resolve(&prepared, &resolver).unwrap();
        let exec: ExecTransaction<InfoTestOperation> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(76),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Info,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::InfoQuery(operation)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let input = ReplacementRecordingInput {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: 0,
            exec,
            barriers: NativeBarrierBatch::default(),
        };
        let missing = ReplacementRecordingRequest::resolve(input, &EmptyResolver, &EmptyResolver)
            .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::InfoQueryPreparationRequired(0)
        );
        let mut altered = missing.input.clone();
        let ResolvedOperation::InfoQuery(ResolvedInfoOperation::RenderPipelineState {
            pipeline,
            ..
        }) = &mut altered.exec.streams[0].segments[0].operations[0]
        else {
            unreachable!()
        };
        *pipeline = ResourceId::new(9, 1);
        let mismatch = ReplacementRecordingRequest::resolve_with_all_semantics(
            altered,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: None,
                completion_effects: None,
                indirect_commands: None,
                resource_states: None,
                info_queries: std::slice::from_ref(&program),
                indirect_range_programs: &[],
            },
        )
        .unwrap_err();
        assert_eq!(
            mismatch.reason,
            ReplacementRecordingError::InfoQueryProgramMismatch(0)
        );
        let mut wrong_transaction = missing.input.clone();
        wrong_transaction.transaction = TransactionId::new(transaction.get() + 1);
        let mismatch = ReplacementRecordingRequest::resolve_with_all_semantics(
            wrong_transaction,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: None,
                completion_effects: None,
                indirect_commands: None,
                resource_states: None,
                info_queries: std::slice::from_ref(&program),
                indirect_range_programs: &[],
            },
        )
        .unwrap_err();
        assert_eq!(
            mismatch.reason,
            ReplacementRecordingError::InfoQueryTransactionMismatch(0)
        );
        let request = ReplacementRecordingRequest::resolve_with_all_semantics(
            missing.input,
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[],
            ReplacementSemanticAdmissions {
                conditions: None,
                completion_effects: None,
                indirect_commands: None,
                resource_states: None,
                info_queries: &[program],
                indirect_range_programs: &[],
            },
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::InfoQuery {
                native: NativeInfoQuery {
                    offset: 40,
                    bytes,
                    ..
                },
                completions,
            }] if bytes.as_ref() == [0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0, 0, 0, 0]
                && completions.as_ref() == prepared.resource_completions().as_ref()
        ));
    }

    #[test]
    fn info_reply_updates_the_exact_actual_buffer_range() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement info recording: no device ({error})");
                return;
            }
        };
        let buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(32)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let Some(memory_type) = context.memory_type_for(
            requirements.memory_type_bits,
            requirements.size,
            crate::memory::MemoryClass::Readback,
        ) else {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.destroy();
            }
            eprintln!("SKIP replacement info recording: no host-visible memory");
            return;
        };
        let memory = unsafe {
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .unwrap();
        unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        let mapped = unsafe {
            context
                .device
                .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
        }
        .unwrap()
        .cast::<u8>();
        unsafe { mapped.write_bytes(0x11, requirements.size as usize) };
        let mapped_kind = context.mapped_memory_kind(memory_type);
        if !mapped_kind.coherent {
            unsafe {
                context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }
        let operation = ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::<RenderPipelineObject>::new(2, 1),
            reply: ResolvedInfoReplyTarget {
                resource: ResourceId::new(3, 1),
                backing: reims_vgpu_protocol::BackingId::new(1),
                range: LinearRange::new(8, 16).unwrap(),
                requested_alignment: 4,
            },
        };
        let transaction = TransactionId::new(77);
        let request = ReplacementRecordingRequest {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: context.gq,
            exec: ExecTransaction::<InfoTestOperation> {
                identity: SubmissionIdentity {
                    id: SubmissionId::new(77),
                    task: TaskId::new(1),
                },
                prologue: reims_vgpu_core::ExecPrologue::default(),
                streams: Box::new([ResolvedExecStream {
                    stream_index: 0,
                    segments: Box::new([ResolvedExecSegment {
                        boundary: SegmentBoundary {
                            stream_index: 0,
                            index: 0,
                            kind: SegmentKind::Info,
                            continues_previous: false,
                            continues_next: false,
                        },
                        operations: Box::new([ResolvedOperation::InfoQuery(operation)]),
                    }]),
                }]),
                accesses: Box::new([]),
            },
            barriers: NativeBarrierBatch::default(),
            operations: Box::new([ReplacementRecordedOperation::InfoQuery {
                native: NativeInfoQuery {
                    buffer,
                    offset: 8,
                    bytes: Box::new([0xa5; 16]),
                },
                completions: Box::new([]),
            }]),
            backings: Box::new([reims_vgpu_protocol::BackingId::new(1)]),
            representation_leases: Box::new([]),
        };
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [OperationKind::InfoQuery]
        );
        assert_eq!(
            recording.backings.as_ref(),
            [reims_vgpu_protocol::BackingId::new(1)]
        );
        let queue = unsafe { context.device.get_device_queue(context.gq, 0) };
        let submit = vk::SubmitInfo::default().command_buffers(&recording.command_buffers);
        unsafe {
            context
                .device
                .queue_submit(queue, &[submit], recording.fence)
        }
        .unwrap();
        unsafe {
            context
                .device
                .wait_for_fences(&[recording.fence], true, u64::MAX)
        }
        .unwrap();
        if !mapped_kind.coherent {
            unsafe {
                context
                    .device
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }
        let bytes = unsafe { core::slice::from_raw_parts(mapped, 32) };
        assert!(bytes[..8].iter().all(|byte| *byte == 0x11));
        assert!(bytes[8..24].iter().all(|byte| *byte == 0xa5));
        assert!(bytes[24..].iter().all(|byte| *byte == 0x11));
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        unsafe {
            context.device.unmap_memory(memory);
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(memory, None);
            context.destroy();
        }
    }

    #[test]
    fn resource_state_transfer_copies_the_exact_actual_buffer_range() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement resource-state recording: no device ({error})");
                return;
            }
        };
        let buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let Some(memory_type) = context.memory_type_for(
            requirements.memory_type_bits,
            requirements.size,
            crate::memory::MemoryClass::Readback,
        ) else {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.destroy();
            }
            eprintln!("SKIP replacement resource-state recording: no host-visible memory");
            return;
        };
        let memory = unsafe {
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .unwrap();
        unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        let mapped = unsafe {
            context
                .device
                .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
        }
        .unwrap()
        .cast::<u8>();
        unsafe {
            mapped.write_bytes(0x11, requirements.size as usize);
            core::slice::from_raw_parts_mut(mapped.add(8), 16).fill(0xa5);
        }
        let mapped_kind = context.mapped_memory_kind(memory_type);
        if !mapped_kind.coherent {
            unsafe {
                context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }

        struct Resolver {
            buffer: vk::Buffer,
            backing: reims_vgpu_protocol::BackingId,
            source: reims_vgpu_protocol::RepresentationId,
            destination: reims_vgpu_protocol::RepresentationId,
        }
        impl crate::replacement_buffer_blit::ReplacementBufferResolver for Resolver {
            fn resolve_buffer(
                &self,
                backing: reims_vgpu_protocol::BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<crate::replacement_buffer_blit::NativeBufferTarget> {
                if backing != self.backing {
                    return None;
                }
                let base_offset = if representation == self.source {
                    0
                } else if representation == self.destination {
                    32
                } else {
                    return None;
                };
                Some(crate::replacement_buffer_blit::NativeBufferTarget {
                    buffer: self.buffer,
                    base_offset,
                    accessible_size: 32,
                    size: 32,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                })
            }
        }

        let region = BackingRegion::Linear(LinearRange::new(8, 16).unwrap());
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = resources
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let destination = reims_vgpu_core::GUEST_REPRESENTATION;
        assert_eq!(
            resources
                .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
                .unwrap(),
            destination
        );
        resources
            .plan_gpu_write(backing, SubmissionId::new(1), source, [region])
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let state = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([reims_vgpu_core::ResolvedResourceStateTarget {
                backing,
                regions: Box::new([region]),
            }]),
            ops: reims_vgpu_protocol::ResourceValidityOps {
                clear_guest_valid: 1,
                set_host_valid: 1,
                ..reims_vgpu_protocol::ResourceValidityOps::default()
            },
        };
        let channel = ChannelId::new(6);
        let mut runtime = reims_vgpu_core::TransactionRuntime::<()>::new(
            reims_vgpu_core::SessionGeneration::new(SessionGenerationId::new(1)),
        );
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 1)),
                ExecTransaction::<TestOperation> {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(78),
                        task: TaskId::new(1),
                    },
                    prologue: reims_vgpu_core::ExecPrologue::default(),
                    streams: Box::new([ResolvedExecStream {
                        stream_index: 0,
                        segments: Box::new([ResolvedExecSegment {
                            boundary: SegmentBoundary {
                                stream_index: 0,
                                index: 0,
                                kind: SegmentKind::Blit,
                                continues_previous: false,
                                continues_next: false,
                            },
                            operations: Box::new([ResolvedOperation::ResourceState(state)]),
                        }]),
                    }]),
                    accesses: Box::new([]),
                },
            )
            .unwrap();
        let (envelope, _, _, states, _) = admitted.into_parts();
        let reims_vgpu_core::DeviceTransactionPayload::Exec(exec) = envelope.payload else {
            unreachable!()
        };
        let transaction = envelope.id;
        let prepared = reims_vgpu_core::prepare_resource_state(
            &mut resources,
            &states,
            0,
            SubmissionId::new(78),
            |_, _| reims_vgpu_core::ValidityRepresentations {
                host_write: Some(source),
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: Some(source),
                guest_visibility_destination: reims_vgpu_core::GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let transfer = prepared.transfers()[0];
        let validity_completion = prepared.resource_completions()[0];
        let program = ReplacementResourceStateProgram::resolve(
            &prepared,
            &Resolver {
                buffer,
                backing,
                source,
                destination,
            },
        )
        .unwrap();
        let request = ReplacementRecordingRequest {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: context.gq,
            exec,
            barriers: NativeBarrierBatch::default(),
            operations: Box::new([ReplacementRecordedOperation::ResourceState {
                completions: program.completions().into(),
                commands: program
                    .native_transfers()
                    .iter()
                    .filter_map(|transfer| match transfer {
                        crate::replacement_resource_state::NativeResourceStateTransfer::Buffer(
                            command,
                        ) => Some(*command),
                        crate::replacement_resource_state::NativeResourceStateTransfer::Image(
                            _,
                        ) => None,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                image_commands: program
                    .native_transfers()
                    .iter()
                    .filter_map(|transfer| match transfer {
                        crate::replacement_resource_state::NativeResourceStateTransfer::Image(
                            commands,
                        ) => Some(commands.iter().copied()),
                        crate::replacement_resource_state::NativeResourceStateTransfer::Buffer(
                            _,
                        ) => None,
                    })
                    .flatten()
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                image_state: program.image_state().cloned(),
                host_landings: program.host_landings().into(),
            }]),
            backings: Box::new([backing]),
            representation_leases: Box::new([]),
        };
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recording.resource_completions.as_ref(),
            [
                ResolvedResourceCompletion::Transfer(transfer),
                validity_completion
            ]
        );
        assert_eq!(recording.backings.as_ref(), [backing]);
        let queue = unsafe { context.device.get_device_queue(context.gq, 0) };
        let submit = vk::SubmitInfo::default().command_buffers(&recording.command_buffers);
        unsafe {
            context
                .device
                .queue_submit(queue, &[submit], recording.fence)
        }
        .unwrap();
        unsafe {
            context
                .device
                .wait_for_fences(&[recording.fence], true, u64::MAX)
        }
        .unwrap();
        if !mapped_kind.coherent {
            unsafe {
                context
                    .device
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }
        let bytes = unsafe { core::slice::from_raw_parts(mapped, 64) };
        assert!(bytes[..8].iter().all(|byte| *byte == 0x11));
        assert!(bytes[8..24].iter().all(|byte| *byte == 0xa5));
        assert!(bytes[24..40].iter().all(|byte| *byte == 0x11));
        assert!(bytes[40..56].iter().all(|byte| *byte == 0xa5));
        assert!(bytes[56..].iter().all(|byte| *byte == 0x11));
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        unsafe {
            context.device.unmap_memory(memory);
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(memory, None);
            context.destroy();
        }
    }

    #[test]
    fn recording_request_unions_native_queue_capabilities() {
        let transaction = TransactionId::new(71);
        let request = ReplacementRecordingRequest {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: 3,
            exec: empty_exec(71),
            barriers: NativeBarrierBatch::default(),
            operations: Box::new([
                ReplacementRecordedOperation::BufferBlit {
                    native: NativeBufferBlit::ComputeFill {
                        buffer: vk::Buffer::null(),
                        binding_offset: 0,
                        binding_range: 4,
                        start: 0,
                        byte_count: 4,
                        pattern: 0,
                        pattern_width: 1,
                        word_count: 1,
                        dispatch_x: 1,
                    },
                    completions: Box::new([]),
                },
                ReplacementRecordedOperation::ImageBlit {
                    native: PreparedNativeImageBlit {
                        state: crate::replacement_image_transition::PreparedNativeImageState {
                            transaction,
                            destination_queue_family: 3,
                            releases: Box::new([]),
                            transitions:
                                crate::replacement_image_transition::NativeImageUseTransitions {
                                    before: NativeBarrierBatch::default(),
                                    after: NativeBarrierBatch::default(),
                                },
                        },
                        commands: Box::new([
                            crate::replacement_image_blit::NativeImageBlitCommand::ImageToBuffer(
                                crate::replacement_image_blit::NativeBufferImageCopy {
                                    buffer: vk::Buffer::null(),
                                    image: vk::Image::null(),
                                    image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                    buffer_offset: 0,
                                    buffer_row_length: 1,
                                    buffer_image_height: 1,
                                    aspect: vk::ImageAspectFlags::STENCIL,
                                    mip: 0,
                                    layer: 0,
                                    image_offset: [0, 0, 0],
                                    extent: [1, 1, 1],
                                },
                            ),
                        ]),
                    },
                    completions: Box::new([]),
                },
            ]),
            backings: Box::new([]),
            representation_leases: Box::new([]),
        };

        assert_eq!(
            request.required_queue_flags(),
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS
        );
    }

    fn mixed_operation_exec(id: u64) -> ExecTransaction<TestOperation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([reims_vgpu_core::ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([reims_vgpu_core::ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Render,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([
                        ResolvedOperation::Participation(ParticipationOperation::Heap {
                            heap: ResourceId::<HeapObject>::new(4, 2),
                            scope: ParticipationScope::Render { stages: None },
                        }),
                        ResolvedOperation::Render(()),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    fn participation_exec(id: u64) -> ExecTransaction<TestOperation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([reims_vgpu_core::ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([reims_vgpu_core::ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Compute,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::Participation(
                        ParticipationOperation::Heap {
                            heap: ResourceId::<HeapObject>::new(4, 2),
                            scope: ParticipationScope::Compute,
                        },
                    )]),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    fn participation_barrier_exec(id: u64) -> ExecTransaction<TestOperation> {
        let mut exec = participation_exec(id);
        exec.streams[0].segments[0].operations = Box::new([
            ResolvedOperation::EncoderBoundary(EncoderBoundary::Begin(SegmentKind::Compute)),
            ResolvedOperation::Participation(ParticipationOperation::Heap {
                heap: ResourceId::<HeapObject>::new(4, 2),
                scope: ParticipationScope::Compute,
            }),
            ResolvedOperation::Barrier(BarrierOperation::Scope {
                scope: MemoryBarrierScope::BUFFERS,
                before: StageScope::Compute,
                after: StageScope::Compute,
            }),
        ]);
        exec
    }

    fn buffer_blit_exec(id: u64, operation: ResolvedBlit) -> ExecTransaction<TestOperation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::Blit(Box::new(operation))]),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    fn image_endpoint(backing: u64, resource: u32) -> ResolvedTextureEndpoint {
        ResolvedTextureEndpoint {
            resource: ResourceId::new(resource, 1),
            image_owner: ResourceId::new(resource, 1),
            storage: reims_vgpu_protocol::BackingId::new(backing),
            level: 0,
            slice: 0,
            backing: ResolvedTextureBacking::Linear(ResolvedLinearTextureLevel {
                base_gva: backing << 12,
                alloc_size: 4096,
                level_offset: 0,
                row_stride: 32,
                slice_stride: 0,
                slice_index: 0,
                width: 8,
                height: 8,
                depth: 1,
                bpp: 4,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            }),
        }
    }

    fn image_operation(source: u64, destination: u64) -> ResolvedBlit {
        ResolvedBlit::TextureToTexture(ResolvedTextureToTextureBlit {
            source: image_endpoint(source, 1),
            source_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            destination: image_endpoint(destination, 2),
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 8,
                height: 8,
                depth: 1,
            },
            aspect: reims_vgpu_core::pixel_format::BlitAspect::Full,
        })
    }

    fn buffer_to_image_operation(source: u64, destination: u64) -> ResolvedBlit {
        ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: ResolvedBufferRange {
                resource: ResourceId::new(1, 1),
                storage: reims_vgpu_protocol::BackingId::new(source),
                region: LinearRange::new(0, 256).unwrap(),
                address: GuestVirtualAddress::new(0x1000),
                length: ByteLength::new(256),
            },
            source_bytes_per_row: 32,
            source_bytes_per_image: 256,
            destination: image_endpoint(destination, 2),
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 8,
                height: 8,
                depth: 1,
            },
            aspect: reims_vgpu_core::pixel_format::BlitAspect::Full,
        })
    }

    fn synthetic_image_program(
        operation: ResolvedBlit,
        transaction: u64,
        queue_family: u32,
        source_release: Option<u32>,
    ) -> crate::replacement_image_blit::ReplacementImageBlitProgram {
        let releases = source_release
            .into_iter()
            .map(
                |source_queue_family| crate::replacement_image_transition::NativeImageRelease {
                    source_queue_family,
                    source_queue: QueueOwnerId::new(source_queue_family),
                    predecessor: reims_vgpu_core::QueueTimelinePoint {
                        epoch: VulkanDeviceEpochId::new(1),
                        queue: QueueOwnerId::new(source_queue_family),
                        value: QueueTimelineValue::new(1),
                    },
                    barriers: NativeBarrierBatch::default(),
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();
        crate::replacement_image_blit::ReplacementImageBlitProgram::synthetic(
            0,
            operation,
            crate::replacement_image_blit::PreparedNativeImageBlit {
                state: crate::replacement_image_transition::PreparedNativeImageState {
                    transaction: TransactionId::new(transaction),
                    destination_queue_family: queue_family,
                    releases,
                    transitions: crate::replacement_image_transition::NativeImageUseTransitions {
                        before: NativeBarrierBatch::default(),
                        after: NativeBarrierBatch::default(),
                    },
                },
                commands: Box::new([]),
            },
        )
    }

    struct EmptyResolver;

    impl ReplacementBarrierResolver for EmptyResolver {
        fn resolve(
            &self,
            _backing: reims_vgpu_protocol::BackingId,
        ) -> Box<[crate::replacement_barrier_record::NativeBarrierResolution]> {
            Box::new([])
        }
    }

    impl ReplacementBarrierResourceResolver for EmptyResolver {
        fn alias_backings(
            &self,
            _resource: ResourceId<reims_vgpu_protocol::ResourceObject>,
        ) -> Option<Box<[reims_vgpu_protocol::BackingId]>> {
            None
        }
    }

    struct BufferResolver {
        backing: reims_vgpu_protocol::BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
        target: crate::replacement_buffer_blit::NativeBufferTarget,
        compute_fill_limits: Option<crate::replacement_buffer_blit::NativeComputeFillLimits>,
    }

    impl crate::replacement_buffer_blit::ReplacementBufferResolver for BufferResolver {
        fn resolve_buffer(
            &self,
            backing: reims_vgpu_protocol::BackingId,
            representation: reims_vgpu_protocol::RepresentationId,
        ) -> Option<crate::replacement_buffer_blit::NativeBufferTarget> {
            (backing == self.backing && representation == self.representation)
                .then_some(self.target)
        }

        fn compute_fill_limits(
            &self,
        ) -> Option<crate::replacement_buffer_blit::NativeComputeFillLimits> {
            self.compute_fill_limits
        }
    }

    fn resolved_request(
        transaction: u64,
        worker: RecordingWorkerId,
        queue_family: u32,
        exec: ExecTransaction<TestOperation>,
    ) -> ReplacementRecordingRequest<TestOperation> {
        ReplacementRecordingRequest::resolve(
            ReplacementRecordingInput {
                transaction: TransactionId::new(transaction),
                worker,
                queue_family,
                exec,
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap()
    }

    #[test]
    fn worker_allocates_and_recycles_its_actual_native_recording() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement recording worker: no device ({error})");
                return;
            }
        };
        let queue_family = context.gq;
        let mut worker = ReplacementRecordingWorker::new(
            RecordingWorkerId::new(0),
            &context.device,
            DescriptorTier::WorkerDescriptorPool,
        );
        let recording = worker.allocate(queue_family, 2).unwrap();
        assert_eq!(recording.worker, RecordingWorkerId::new(0));
        assert_eq!(recording.command_buffers.len(), 2);
        worker.recycle(recording).unwrap();
        drop(worker);
        unsafe { context.destroy() };
    }

    #[test]
    fn render_draw_records_and_executes_on_the_assigned_actual_worker() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement render recording: no device ({error})");
                return;
            }
        };
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/replacement_render_test.vert");
        let shader_path = std::env::temp_dir().join(format!(
            "reims-vgpu-replacement-render-{}.spv",
            std::process::id()
        ));
        let compiled = std::process::Command::new("glslc")
            .args(["-fshader-stage=vertex", "-O"])
            .arg(source)
            .arg("-o")
            .arg(&shader_path)
            .status();
        let Ok(status) = compiled else {
            unsafe { context.destroy() };
            eprintln!("SKIP replacement render recording: glslc unavailable");
            return;
        };
        assert!(
            status.success(),
            "fixed replacement render shader must compile"
        );
        let context =
            std::sync::Arc::new(crate::engine::context::SharedDeviceContext::new(context));
        let shader_bytes = std::fs::read(&shader_path).unwrap();
        let _ = std::fs::remove_file(&shader_path);
        let shader_words = shader_bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let shader = unsafe {
            context.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&shader_words),
                None,
            )
        }
        .unwrap();
        let descriptor_layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default(), None)
        }
        .unwrap();
        let pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_layout]),
                None,
            )
        }
        .unwrap();
        let subpass =
            vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
        let render_pass = unsafe {
            context.device.create_render_pass(
                &vk::RenderPassCreateInfo::default().subpasses(std::slice::from_ref(&subpass)),
                None,
            )
        }
        .unwrap();
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(entry);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .rasterizer_discard_enable(true)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(std::slice::from_ref(&stage))
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .viewport_state(&viewport_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipeline = unsafe {
            context.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .unwrap()[0];

        let visibility_buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(16)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
        }
        .unwrap();
        let visibility_requirements = unsafe {
            context
                .device
                .get_buffer_memory_requirements(visibility_buffer)
        };
        let visibility_memory_type = context
            .memory_type_for(
                visibility_requirements.memory_type_bits,
                visibility_requirements.size,
                crate::memory::MemoryClass::Readback,
            )
            .expect("an actual-device recording fixture requires readback memory");
        let visibility_memory = unsafe {
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(visibility_requirements.size)
                    .memory_type_index(visibility_memory_type),
                None,
            )
        }
        .unwrap();
        unsafe {
            context
                .device
                .bind_buffer_memory(visibility_buffer, visibility_memory, 0)
        }
        .unwrap();

        let transaction = TransactionId::new(91);
        let semantic = reims_vgpu_core::ResolvedRenderDispatch {
            pipeline: ResourceId::new(2, 1),
            program: Default::default(),
            vertex_buffers: Box::new([]),
            depth_stencil: None,
            render_extent: [1, 1],
            raster: reims_vgpu_core::ResolvedRenderRasterState::default(),
            visibility: None,
            begins_encoder: true,
            ends_encoder: true,
            draw: reims_vgpu_core::ResolvedRenderDraw::Direct {
                topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            attachments: Box::new([]),
            resources: Box::new([]),
            null_bindings: Box::new([]),
            samplers: Box::new([]),
        };
        let native = NativeRenderDispatch {
            pipeline: std::sync::Arc::new(
                crate::replacement_render::ReplacementRenderPipelineVariant::new(
                    context.clone(),
                    crate::replacement_render::ReplacementRenderPipeline {
                        pipeline,
                        layout: pipeline_layout,
                        descriptor_set_layout: descriptor_layout,
                        render_pass,
                        program: Default::default(),
                        vertex_buffers: Box::new([]),
                        color_attachments: Box::new([]),
                        depth_stencil_attachment: None,
                        feedback_loop_aspects: vk::ImageAspectFlags::empty(),
                        color_input: false,
                        sample_count: vk::SampleCountFlags::TYPE_1,
                        viewport_count: 1,
                        static_state:
                            crate::replacement_render::ReplacementRenderStaticState::default(),
                        dynamic_states: Default::default(),
                        depth_stencil: None,
                    },
                ),
            ),
            descriptors: Box::new([]),
            sampler_leases: Box::new([]),
            descriptor_counts: Box::new([]),
            vertex_buffers: Box::new([]),
            index_buffer: None,
            indirect_buffer: None,
            attachment_views: Box::new([]),
            clear_values: Box::new([]),
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            viewports: Box::new([vk::Viewport {
                x: 0.0,
                y: 1.0,
                width: 1.0,
                height: -1.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }]),
            scissors: Box::new([vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            }]),
            depth_bias: None,
            blend_color: None,
            stencil_reference: [0, 0],
            visibility: Some(crate::replacement_render::NativeRenderVisibility {
                mode: reims_vgpu_protocol::VisibilityResultMode::Boolean,
                buffer: visibility_buffer,
                offset: 0,
            }),
            begins_native_pass: true,
            ends_native_pass: true,
            draw: semantic.draw,
            image_state: crate::replacement_image_transition::PreparedNativeImageState {
                transaction,
                destination_queue_family: context.gq,
                releases: Box::new([]),
                transitions: crate::replacement_image_transition::NativeImageUseTransitions {
                    before: NativeBarrierBatch::default(),
                    after: NativeBarrierBatch::default(),
                },
            },
        };
        let mut semantic_first = semantic.clone();
        semantic_first.ends_encoder = false;
        let mut semantic_last = semantic;
        semantic_last.begins_encoder = false;
        let mut native_first = native.clone();
        native_first.ends_native_pass = false;
        let mut native_last = native;
        native_last.begins_native_pass = false;
        native_last.visibility.as_mut().unwrap().offset = 8;
        let request = ReplacementRecordingRequest {
            transaction,
            worker: RecordingWorkerId::new(0),
            queue_family: context.gq,
            exec: ExecTransaction {
                identity: SubmissionIdentity {
                    id: SubmissionId::new(91),
                    task: TaskId::new(1),
                },
                prologue: reims_vgpu_core::ExecPrologue::default(),
                streams: Box::new([ResolvedExecStream {
                    stream_index: 0,
                    segments: Box::new([ResolvedExecSegment {
                        boundary: SegmentBoundary {
                            stream_index: 0,
                            index: 0,
                            kind: SegmentKind::Render,
                            continues_previous: false,
                            continues_next: false,
                        },
                        operations: Box::new([
                            ResolvedOperation::<_, (), (), (), ()>::Render(semantic_first),
                            ResolvedOperation::<_, (), (), (), ()>::Render(semantic_last),
                        ]),
                    }]),
                }]),
                accesses: Box::new([]),
            },
            barriers: NativeBarrierBatch::default(),
            operations: Box::new([
                ReplacementRecordedOperation::Render {
                    native: Box::new(native_first),
                    completions: Box::new([]),
                },
                ReplacementRecordedOperation::Render {
                    native: Box::new(native_last),
                    completions: Box::new([]),
                },
            ]),
            backings: Box::new([]),
            representation_leases: Box::new([]),
        };
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [OperationKind::Render, OperationKind::Render]
        );
        assert_eq!(recording.framebuffers.len(), 1);
        assert_eq!(recording.query_pools.len(), 1);
        assert_eq!(recording.render_pipeline_variants.len(), 1);
        let queue = unsafe { context.device.get_device_queue(context.gq, 0) };
        let submit = vk::SubmitInfo::default().command_buffers(&recording.command_buffers);
        unsafe {
            context
                .device
                .queue_submit(queue, &[submit], recording.fence)
        }
        .unwrap();
        unsafe {
            context
                .device
                .wait_for_fences(&[recording.fence], true, u64::MAX)
        }
        .unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        unsafe {
            context.device.destroy_shader_module(shader, None);
            context.device.destroy_buffer(visibility_buffer, None);
            context.device.free_memory(visibility_memory, None);
        }
    }

    #[test]
    fn image_release_records_on_its_source_family_worker() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement image release recording: no device ({error})");
                return;
            }
        };
        let image = unsafe {
            context.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: 8,
                        height: 8,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .unwrap();
        let mut barriers = NativeBarrierBatch::default();
        barriers.images.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        );
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_image_release_recording(
            &workers,
            ReplacementImageReleaseRecordingRequest {
                transaction: TransactionId::new(6),
                worker: RecordingWorkerId::new(0),
                release: crate::replacement_image_transition::NativeImageRelease {
                    source_queue_family: context.gq,
                    source_queue: QueueOwnerId::new(0),
                    predecessor: reims_vgpu_core::QueueTimelinePoint {
                        epoch: VulkanDeviceEpochId::new(1),
                        queue: QueueOwnerId::new(0),
                        value: QueueTimelineValue::new(1),
                    },
                    barriers,
                },
            },
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(recording.queue_family, context.gq);
        assert!(recording.recorded_operations.is_empty());
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        unsafe {
            context.device.destroy_image(image, None);
            context.destroy();
        }
    }

    #[test]
    fn image_program_is_joined_to_exact_exec_queue_and_source_release_state() {
        let operation = image_operation(1, 2);
        let program = synthetic_image_program(operation.clone(), 7, 3, None);
        let request = ReplacementRecordingRequest::resolve_with_blits(
            ReplacementRecordingInput {
                transaction: TransactionId::new(7),
                worker: RecordingWorkerId::new(0),
                queue_family: 3,
                exec: buffer_blit_exec(7, operation.clone()),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[],
            std::slice::from_ref(&program),
        )
        .unwrap();
        assert!(matches!(
            request.operations.as_ref(),
            [ReplacementRecordedOperation::ImageBlit { .. }]
        ));

        let wrong_queue = ReplacementRecordingRequest::resolve_with_blits(
            ReplacementRecordingInput {
                transaction: TransactionId::new(7),
                worker: RecordingWorkerId::new(0),
                queue_family: 4,
                exec: buffer_blit_exec(7, operation.clone()),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[program],
        )
        .unwrap_err();
        assert_eq!(
            wrong_queue.reason,
            ReplacementRecordingError::ImageBlitQueueFamilyMismatch(0)
        );

        let release_program = synthetic_image_program(operation.clone(), 7, 3, Some(2));
        let release = ReplacementRecordingRequest::resolve_with_blits(
            ReplacementRecordingInput {
                transaction: TransactionId::new(7),
                worker: RecordingWorkerId::new(0),
                queue_family: 3,
                exec: buffer_blit_exec(7, operation),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[release_program],
        )
        .unwrap_err();
        assert_eq!(
            release.reason,
            ReplacementRecordingError::ImageQueueReleaseSubmissionRequired(0)
        );
    }

    #[test]
    fn unaligned_buffer_fill_records_with_the_fixed_compute_emitter() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement buffer recording: no device ({error})");
                return;
            }
        };
        let buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default().size(260).usage(
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let Some(memory_type) = context.memory_type_for(
            requirements.memory_type_bits,
            requirements.size,
            crate::memory::MemoryClass::Readback,
        ) else {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.destroy();
            }
            eprintln!("SKIP replacement compute fill: no host-visible storage memory");
            return;
        };
        let memory = unsafe {
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .unwrap();
        unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        let mapped = unsafe {
            context
                .device
                .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
        }
        .unwrap()
        .cast::<u8>();
        unsafe { mapped.write_bytes(0x11, requirements.size as usize) };
        let mapped_kind = context.mapped_memory_kind(memory_type);
        if !mapped_kind.coherent {
            unsafe {
                context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(9));
        let backing = match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 257).unwrap())]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let representation = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let operation = ResolvedBlit::Fill {
            destination: ResolvedBufferRange {
                resource: ResourceId::new(1, 1),
                storage: backing,
                region: LinearRange::new(1, 256).unwrap(),
                address: GuestVirtualAddress::new(0x1001),
                length: ByteLength::new(256),
            },
            pattern: BufferFillPattern::Byte(0xa5),
        };
        let transaction = TransactionId::new(15);
        let unpositioned = reims_vgpu_core::prepare_buffer_blit(
            &mut resources,
            transaction,
            SubmissionId::new(15),
            operation.clone(),
        )
        .unwrap();
        assert_eq!(
            crate::replacement_buffer_blit::ReplacementBufferBlitProgram::resolve(
                0,
                &unpositioned,
                &BufferResolver {
                    backing,
                    representation,
                    target: crate::replacement_buffer_blit::NativeBufferTarget {
                        buffer,
                        base_offset: 0,
                        accessible_size: 260,
                        size: 257,
                        usage: vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    },
                    compute_fill_limits: Some(
                        crate::replacement_buffer_blit::NativeComputeFillLimits {
                            min_storage_buffer_offset_alignment: context
                                .storage_buffer_offset_align,
                            max_storage_buffer_range: context.max_storage_buffer_range,
                            max_compute_work_group_count_x: 1,
                        },
                    ),
                },
            ),
            Err(crate::replacement_buffer_blit::BufferBlitRecordError::WriteIdentityMismatch)
        );
        cancel_prepared_buffer_blit(&mut resources, unpositioned).unwrap();
        let prepared = reims_vgpu_core::prepare_buffer_blit_with_write(
            &mut resources,
            transaction,
            reims_vgpu_core::GpuWriteId::operation(transaction, SubmissionId::new(15), 0),
            operation.clone(),
        )
        .unwrap();
        let program = crate::replacement_buffer_blit::ReplacementBufferBlitProgram::resolve(
            0,
            &prepared,
            &BufferResolver {
                backing,
                representation,
                target: crate::replacement_buffer_blit::NativeBufferTarget {
                    buffer,
                    base_offset: 0,
                    accessible_size: 260,
                    size: 257,
                    usage: vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                },
                compute_fill_limits: Some(
                    crate::replacement_buffer_blit::NativeComputeFillLimits {
                        min_storage_buffer_offset_alignment: context.storage_buffer_offset_align,
                        max_storage_buffer_range: context.max_storage_buffer_range,
                        max_compute_work_group_count_x: 1,
                    },
                ),
            },
        )
        .unwrap();
        let expected_completions = prepared.resource_completions();
        let missing = ReplacementRecordingRequest::resolve(
            ReplacementRecordingInput {
                transaction,
                worker: RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: buffer_blit_exec(15, operation.clone()),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap_err();
        assert_eq!(
            missing.reason,
            ReplacementRecordingError::BufferBlitPreparationRequired {
                index: 0,
                kind: BlitKind::Fill,
            }
        );
        let duplicate = ReplacementRecordingRequest::resolve_with_buffer_blits(
            missing.input,
            &EmptyResolver,
            &EmptyResolver,
            &[program.clone(), program.clone()],
        )
        .unwrap_err();
        assert_eq!(
            duplicate.reason,
            ReplacementRecordingError::DuplicateBufferBlitProgram(0)
        );
        let mismatched = ReplacementRecordingRequest::resolve_with_buffer_blits(
            ReplacementRecordingInput {
                transaction,
                worker: RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: buffer_blit_exec(
                    15,
                    ResolvedBlit::Fill {
                        destination: match &operation {
                            ResolvedBlit::Fill { destination, .. } => *destination,
                            _ => unreachable!(),
                        },
                        pattern: BufferFillPattern::Byte(0x5a),
                    },
                ),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            std::slice::from_ref(&program),
        )
        .unwrap_err();
        assert_eq!(
            mismatched.reason,
            ReplacementRecordingError::BufferBlitProgramMismatch(0)
        );
        let request = ReplacementRecordingRequest::resolve_with_buffer_blits(
            ReplacementRecordingInput {
                transaction,
                worker: RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: buffer_blit_exec(15, operation),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[program],
        )
        .unwrap();
        assert_eq!(request.required_queue_flags(), vk::QueueFlags::COMPUTE);
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [OperationKind::Blit]
        );
        assert_eq!(recording.backings.as_ref(), [backing]);
        assert_eq!(recording.resource_completions, expected_completions);
        assert_eq!(recording.descriptor_sets.len(), 1);
        let queue = unsafe { context.device.get_device_queue(context.gq, 0) };
        let submit = vk::SubmitInfo::default().command_buffers(&recording.command_buffers);
        unsafe {
            context
                .device
                .queue_submit(queue, &[submit], recording.fence)
        }
        .unwrap();
        unsafe {
            context
                .device
                .wait_for_fences(&[recording.fence], true, u64::MAX)
        }
        .unwrap();
        if !mapped_kind.coherent {
            unsafe {
                context
                    .device
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }
        let bytes = unsafe { core::slice::from_raw_parts(mapped, 260) };
        assert_eq!(bytes[0], 0x11);
        assert!(bytes[1..257].iter().all(|byte| *byte == 0xa5));
        assert!(bytes[257..260].iter().all(|byte| *byte == 0x11));
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        cancel_prepared_buffer_blit(&mut resources, prepared).unwrap();
        unsafe {
            context.device.unmap_memory(memory);
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(memory, None);
        }
        unsafe { context.destroy() };
    }

    #[test]
    fn prepared_image_copy_records_on_the_assigned_actual_worker() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement image recording: no device ({error})");
                return;
            }
        };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 8,
                height: 8,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let source = unsafe { context.device.create_image(&image_info, None) }.unwrap();
        let destination = unsafe { context.device.create_image(&image_info, None) }.unwrap();
        let operation = image_operation(1, 2);
        let transaction = TransactionId::new(19);
        let program = crate::replacement_image_blit::ReplacementImageBlitProgram::synthetic(
            0,
            operation.clone(),
            crate::replacement_image_blit::PreparedNativeImageBlit {
                state: crate::replacement_image_transition::PreparedNativeImageState {
                    transaction,
                    destination_queue_family: context.gq,
                    releases: Box::new([]),
                    transitions: crate::replacement_image_transition::NativeImageUseTransitions {
                        before: NativeBarrierBatch::default(),
                        after: NativeBarrierBatch::default(),
                    },
                },
                commands: Box::new([
                    crate::replacement_image_blit::NativeImageBlitCommand::ImageToImage(
                        crate::replacement_image_blit::NativeImageCopy {
                            source,
                            source_layout: vk::ImageLayout::GENERAL,
                            destination,
                            destination_layout: vk::ImageLayout::GENERAL,
                            aspect: vk::ImageAspectFlags::COLOR,
                            source_mip: 0,
                            source_layer: 0,
                            destination_mip: 0,
                            destination_layer: 0,
                            source_offset: [0, 0, 0],
                            destination_offset: [0, 0, 0],
                            extent: [8, 8, 1],
                        },
                    ),
                ]),
            },
        );
        let request = ReplacementRecordingRequest::resolve_with_blits(
            ReplacementRecordingInput {
                transaction,
                worker: RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: buffer_blit_exec(19, operation),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[program],
        )
        .unwrap();
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [OperationKind::Blit]
        );
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        unsafe {
            context.device.destroy_image(source, None);
            context.device.destroy_image(destination, None);
            context.destroy();
        }
    }

    #[test]
    fn prepared_buffer_to_image_copy_records_on_the_assigned_actual_worker() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement buffer/image recording: no device ({error})");
                return;
            }
        };
        let buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(256)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }
        .unwrap();
        let image = unsafe {
            context.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: 8,
                        height: 8,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .unwrap();
        let operation = buffer_to_image_operation(1, 2);
        let transaction = TransactionId::new(20);
        let program = crate::replacement_image_blit::ReplacementImageBlitProgram::synthetic(
            0,
            operation.clone(),
            crate::replacement_image_blit::PreparedNativeImageBlit {
                state: crate::replacement_image_transition::PreparedNativeImageState {
                    transaction,
                    destination_queue_family: context.gq,
                    releases: Box::new([]),
                    transitions: crate::replacement_image_transition::NativeImageUseTransitions {
                        before: NativeBarrierBatch::default(),
                        after: NativeBarrierBatch::default(),
                    },
                },
                commands: Box::new([
                    crate::replacement_image_blit::NativeImageBlitCommand::BufferToImage(
                        crate::replacement_image_blit::NativeBufferImageCopy {
                            buffer,
                            image,
                            image_layout: vk::ImageLayout::GENERAL,
                            buffer_offset: 0,
                            buffer_row_length: 8,
                            buffer_image_height: 8,
                            aspect: vk::ImageAspectFlags::COLOR,
                            mip: 0,
                            layer: 0,
                            image_offset: [0, 0, 0],
                            extent: [8, 8, 1],
                        },
                    ),
                ]),
            },
        );
        let request = ReplacementRecordingRequest::resolve_with_blits(
            ReplacementRecordingInput {
                transaction,
                worker: RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: buffer_blit_exec(20, operation),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
            &[],
            &[program],
        )
        .unwrap();
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let recording = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [OperationKind::Blit]
        );
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        drop(workers);
        unsafe {
            context.device.destroy_buffer(buffer, None);
            context.device.destroy_image(image, None);
            context.destroy();
        }
    }

    #[test]
    fn worker_records_an_encoder_free_submission_as_one_ended_primary() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement empty recording: no device ({error})");
                return;
            }
        };
        let mut worker = ReplacementRecordingWorker::new(
            RecordingWorkerId::new(0),
            &context.device,
            DescriptorTier::WorkerDescriptorPool,
        );
        let recording = worker
            .record_hazard_preamble(context.gq, &[], &EmptyResolver)
            .unwrap();
        assert_eq!(recording.command_buffers.len(), 1);
        worker.recycle(recording).unwrap();
        drop(worker);
        unsafe { context.destroy() };
    }

    #[test]
    fn worker_records_a_global_semantic_hazard_into_the_primary() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement hazard recording: no device ({error})");
                return;
            }
        };
        let domain = HazardDomainId::new(3);
        let hazard = HazardRequirement {
            edge: HazardEdge {
                newer: TransactionId::new(2),
                older: TransactionId::new(1),
                newer_ordinal: IngressOrdinal::new(2),
                older_ordinal: IngressOrdinal::new(1),
                cause: HazardCause::WholeDomain,
            },
            earlier: AccessIntent::whole_domain(domain, AccessMode::Write, StageScope::Compute),
            later: AccessIntent::whole_domain(domain, AccessMode::Read, StageScope::Fragment),
        };
        let mut worker = ReplacementRecordingWorker::new(
            RecordingWorkerId::new(0),
            &context.device,
            DescriptorTier::WorkerDescriptorPool,
        );
        let recording = worker
            .record_hazard_preamble(context.gq, &[hazard], &EmptyResolver)
            .unwrap();
        worker.recycle(recording).unwrap();
        drop(worker);
        unsafe { context.destroy() };
    }

    #[test]
    fn immutable_recording_request_runs_on_its_assigned_fixed_worker() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement recording dispatch: no device ({error})");
                return;
            }
        };
        let workers = FixedExecutor::from_workers(
            (0..2)
                .map(|index| {
                    ReplacementRecordingWorker::new(
                        RecordingWorkerId::new(index),
                        &context.device,
                        DescriptorTier::WorkerDescriptorPool,
                    )
                })
                .collect(),
        )
        .unwrap();
        let request = resolved_request(
            9,
            RecordingWorkerId::new(1),
            context.gq,
            participation_barrier_exec(9),
        );
        let (release, held) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(1), move |_| {
                held.recv().unwrap();
            })
            .unwrap();
        let pending = dispatch_replacement_recording(&workers, request).unwrap();
        let ReplacementRecordingPoll::Pending(pending) = pending.try_complete() else {
            panic!("the occupied assigned worker cannot complete the queued recording");
        };
        release.send(()).unwrap();
        let recording = pending.wait().unwrap();
        assert_eq!(recording.worker, RecordingWorkerId::new(1));
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [
                OperationKind::EncoderBoundary,
                OperationKind::Participation,
                OperationKind::Barrier
            ]
        );

        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(recording.worker, move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        assert!(workers.take_events().iter().any(|event| {
            event.transaction == TransactionId::new(9) && event.worker == RecordingWorkerId::new(1)
        }));
        drop(workers);
        unsafe { context.destroy() };
    }

    #[test]
    fn recording_dispatch_refusal_returns_the_complete_request() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement recording dispatch refusal: no device ({error})");
                return;
            }
        };
        let workers = FixedExecutor::new(1, |worker| {
            ReplacementRecordingWorker::new(
                worker,
                &context.device,
                DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let request = resolved_request(11, RecordingWorkerId::new(2), context.gq, empty_exec(11));
        let failure = dispatch_replacement_recording(&workers, request).unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementRecordingDispatchError::Executor(FixedExecutorError::UnknownWorker)
        );
        assert_eq!(failure.request.transaction, TransactionId::new(11));
        assert_eq!(failure.request.worker, RecordingWorkerId::new(2));
        assert_eq!(failure.request.queue_family, context.gq);
        assert!(failure.request.barriers.is_empty());

        let failure = ReplacementRecordingRequest::resolve(
            ReplacementRecordingInput {
                transaction: TransactionId::new(13),
                worker: RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: mixed_operation_exec(13),
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementRecordingError::RenderPreparationRequired(1)
        );
        assert_eq!(failure.input.exec.operations().count(), 2);
        drop(workers);

        let workers = FixedExecutor::from_workers(vec![ReplacementRecordingWorker::new(
            RecordingWorkerId::new(3),
            &context.device,
            DescriptorTier::WorkerDescriptorPool,
        )])
        .unwrap();
        let request = resolved_request(12, RecordingWorkerId::new(0), context.gq, empty_exec(12));
        let failure = dispatch_replacement_recording(&workers, request)
            .unwrap()
            .wait()
            .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementRecordingDispatchError::WorkerIdentityMismatch {
                expected: RecordingWorkerId::new(0),
                actual: RecordingWorkerId::new(3),
            }
        );
        assert_eq!(failure.request.transaction, TransactionId::new(12));
        drop(workers);
        unsafe { context.destroy() };
    }

    #[test]
    fn fallback_descriptor_set_returns_to_the_same_exact_request_block() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement descriptor arena: no device ({error})");
                return;
            }
        };
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                    None,
                )
                .unwrap()
        };
        let mut worker = ReplacementRecordingWorker::new(
            RecordingWorkerId::new(0),
            &context.device,
            DescriptorTier::WorkerDescriptorPool,
        );
        let request = [(vk::DescriptorType::STORAGE_BUFFER, 1)];
        let first = worker.allocate_descriptor_set(layout, &request).unwrap();
        let mut recording = worker
            .record_hazard_preamble(context.gq, &[], &EmptyResolver)
            .unwrap();
        recording.descriptor_sets = Box::new([first]);
        worker.recycle(recording).unwrap();

        let second = worker.allocate_descriptor_set(layout, &request).unwrap();
        assert_eq!(second.pool, first.pool);
        assert_eq!(second.set, first.set);
        let mut recording = worker
            .record_hazard_preamble(context.gq, &[], &EmptyResolver)
            .unwrap();
        recording.descriptor_sets = Box::new([second]);
        worker.recycle(recording).unwrap();
        drop(worker);
        unsafe {
            context.device.destroy_descriptor_set_layout(layout, None);
            context.destroy();
        }
    }

    #[test]
    fn capability_selected_push_tier_cannot_allocate_from_the_pool_fallback() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement descriptor tier: no device ({error})");
                return;
            }
        };
        let mut worker = ReplacementRecordingWorker::new(
            RecordingWorkerId::new(0),
            &context.device,
            DescriptorTier::PushDescriptor,
        );
        assert_eq!(
            worker.allocate_descriptor_set(
                vk::DescriptorSetLayout::null(),
                &[(vk::DescriptorType::STORAGE_BUFFER, 1)],
            ),
            Err(ReplacementRecordingError::DescriptorPoolTierUnavailable)
        );
        drop(worker);
        unsafe { context.destroy() };
    }
}
