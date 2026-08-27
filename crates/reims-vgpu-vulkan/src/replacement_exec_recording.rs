//! Whole-EXEC ownership across asynchronous native recording.

use crate::{
    replacement_compute::ReplacementComputeImageBindings,
    replacement_exec_image::{
        exec_has_image_uses, validate_exec_image_states, ExecImageStateError,
    },
    replacement_image_state::PreparedImageStateBatch,
    replacement_queue::PreparedReplacementQueueSubmission,
    replacement_recording::{ReplacementRecordingOperation, ReplacementRecordingRequest},
    replacement_recording_queue::{
        dispatch_prepared_replacement_recording_with_auxiliary_waits,
        PendingPreparedReplacementRecording, PreparedReplacementRecordingError,
        PreparedReplacementRecordingFailure, PreparedReplacementRecordingPoll,
    },
    replacement_render::ReplacementRenderImageBindings,
    replacement_submit::QueueTimelineSemaphores,
};
use reims_vgpu_core::{
    AdmittedCompletionEffects, AdmittedExecConditions, AdmittedIndirectCommands,
    AdmittedResourceStates, FixedExecutor, PreparedExecResources, PreparedNativeSubmission,
    ResolvedComputeDispatch, ResolvedIndirectCommand, ResolvedInfoOperation, ResolvedOperation,
    ResolvedRenderDispatch, ResolvedReplayCompletion,
};
use std::sync::Arc;

use crate::{
    replacement_barrier_record::{ReplacementBarrierResolver, ReplacementBarrierResourceResolver},
    replacement_buffer_blit::{BufferBlitRecordError, ReplacementBufferBlitProgram},
    replacement_compute::{
        resolve_exec_compute_programs, ComputeExecProgramError, ReplacementComputePipelineVariant,
        ReplacementComputeResolver,
    },
    replacement_image_blit::{
        resolve_exec_image_blit_programs, ImageBlitRecordError, ReplacementImageBlitProgram,
    },
    replacement_indirect_range::{
        ReplacementIndirectRangeDevice, ReplacementIndirectRangeError,
        ReplacementIndirectRangeProgram,
    },
    replacement_info_query::{InfoQueryRecordError, ReplacementInfoQueryProgram},
    replacement_recording::{
        ReplacementRecordingError, ReplacementRecordingInput, ReplacementSemanticAdmissions,
    },
    replacement_render::{
        resolve_exec_render_programs, RenderExecProgramError, ReplacementRenderPipelineVariant,
        ReplacementRenderResolver,
    },
    replacement_resource_state::{
        ReplacementContentSynchronizationProgram, ReplacementResourceStateBatchProgram,
        ResourceStateBatchResolveError,
    },
};

/// Semantic admission proofs that are owned outside operation-local resource
/// preparation. Native query and indirect-range programs are deliberately not
/// accepted here: they are derived below from the one resource envelope.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplacementExecSemanticAdmissions<'a, Completion> {
    pub conditions: Option<&'a AdmittedExecConditions>,
    pub completion_effects: Option<&'a AdmittedCompletionEffects<Completion>>,
    pub indirect_commands: Option<&'a AdmittedIndirectCommands<ResolvedIndirectCommand>>,
    pub resource_states: Option<&'a AdmittedResourceStates>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementExecProgramError {
    DeviceLifetimeClosed,
    TransactionMismatch,
    UnpositionedBufferBlit,
    BufferBlit {
        index: usize,
        reason: BufferBlitRecordError,
    },
    ImageStateBatchMissing,
    ImageBlit(ImageBlitRecordError),
    Compute(ComputeExecProgramError),
    Render(RenderExecProgramError),
    InfoQuery {
        index: usize,
        reason: InfoQueryRecordError,
    },
    IndirectRangeDeviceMissing(usize),
    IndirectRange {
        index: usize,
        reason: ReplacementIndirectRangeError,
    },
    ResourceState(ResourceStateBatchResolveError),
    ContentSynchronization(crate::replacement_resource_state::ResourceStateTransferRecordError),
    ContentSynchronizationTransactionMismatch,
    RepresentationUses(reims_vgpu_core::ResourceUseBatchError),
    RepresentationNativeAbsent {
        backing: reims_vgpu_protocol::BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
    },
    RepresentationLeases(crate::replacement_recording::ReplacementRecordingLeaseError),
    Recording(ReplacementRecordingError),
}

impl ReplacementExecProgramError {
    /// Whether re-offering this EXEC could ever produce a different answer.
    ///
    /// Only the unimplemented image-view arms are terminal today. Everything
    /// else here is either a device fault or a state a later packet supplies,
    /// and a wrong `true` throws away guest work that would have recorded.
    /// See
    /// [`crate::replacement_image_transition::TextureBindingViewDecline::is_unimplemented`].
    pub const fn is_terminal_refusal(&self) -> bool {
        match self {
            Self::Render(reason) => reason.is_terminal_refusal(),
            Self::Compute(reason) => reason.is_terminal_refusal(),
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct ReplacementExecProgramFailure<Operation> {
    pub reason: ReplacementExecProgramError,
    pub input: ReplacementRecordingInput<Operation>,
}

pub type CanonicalReplacementOperation<Completion> = ResolvedOperation<
    ResolvedRenderDispatch,
    ResolvedComputeDispatch,
    ResolvedInfoOperation,
    ResolvedIndirectCommand,
    Completion,
>;

/// Resolve every operation-local native program from one already-assembled
/// resource envelope, then bind that complete set to the immutable EXEC.
/// Failures consume no semantic/resource preparation ownership; the exact
/// recording input is returned for retry or cancellation.
pub fn resolve_prepared_exec_recording<Completion: Clone + PartialEq>(
    input: ReplacementRecordingInput<CanonicalReplacementOperation<Completion>>,
    resources: &PreparedExecResources<
        ResolvedComputeDispatch,
        ReplacementComputePipelineVariant,
        ResolvedRenderDispatch,
        ReplacementRenderPipelineVariant,
    >,
    image_states: Option<&PreparedImageStateBatch>,
    semantics: ReplacementExecSemanticAdmissions<'_, Completion>,
    indirect_device: Option<Arc<dyn ReplacementIndirectRangeDevice>>,
    resolver: &(impl ReplacementComputeResolver
          + ReplacementRenderResolver
          + ReplacementBarrierResourceResolver
          + ReplacementBarrierResolver),
) -> Result<
    ReplacementRecordingRequest<CanonicalReplacementOperation<Completion>>,
    Box<ReplacementExecProgramFailure<CanonicalReplacementOperation<Completion>>>,
> {
    resolve_prepared_exec_recording_at(
        input,
        resources,
        image_states,
        semantics,
        indirect_device,
        resolver,
        0,
        false,
        None,
    )
}

/// Assemble a resumed indirect phase while retaining both its semantic source
/// positions and its unique expanded resource positions.
#[allow(
    clippy::too_many_arguments,
    reason = "the final argument is the original EXEC position owned by continuation state"
)]
pub fn resolve_prepared_exec_continuation_recording<Completion: Clone + PartialEq>(
    input: ReplacementRecordingInput<CanonicalReplacementOperation<Completion>>,
    resources: &PreparedExecResources<
        ResolvedComputeDispatch,
        ReplacementComputePipelineVariant,
        ResolvedRenderDispatch,
        ReplacementRenderPipelineVariant,
    >,
    image_states: Option<&PreparedImageStateBatch>,
    semantics: ReplacementExecSemanticAdmissions<'_, Completion>,
    indirect_device: Option<Arc<dyn ReplacementIndirectRangeDevice>>,
    resolver: &(impl ReplacementComputeResolver
          + ReplacementRenderResolver
          + ReplacementBarrierResourceResolver
          + ReplacementBarrierResolver),
    operation_base: usize,
) -> Result<
    ReplacementRecordingRequest<CanonicalReplacementOperation<Completion>>,
    Box<ReplacementExecProgramFailure<CanonicalReplacementOperation<Completion>>>,
> {
    resolve_prepared_exec_recording_at(
        input,
        resources,
        image_states,
        semantics,
        indirect_device,
        resolver,
        operation_base,
        true,
        None,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the origin slice is owned by the exact expanded continuation"
)]
pub fn resolve_prepared_exec_continuation_recording_at_positions<Completion: Clone + PartialEq>(
    input: ReplacementRecordingInput<CanonicalReplacementOperation<Completion>>,
    resources: &PreparedExecResources<
        ResolvedComputeDispatch,
        ReplacementComputePipelineVariant,
        ResolvedRenderDispatch,
        ReplacementRenderPipelineVariant,
    >,
    image_states: Option<&PreparedImageStateBatch>,
    semantics: ReplacementExecSemanticAdmissions<'_, Completion>,
    indirect_device: Option<Arc<dyn ReplacementIndirectRangeDevice>>,
    resolver: &(impl ReplacementComputeResolver
          + ReplacementRenderResolver
          + ReplacementBarrierResourceResolver
          + ReplacementBarrierResolver),
    operation_origins: &[reims_vgpu_core::ExpandedIndirectOperationOrigin],
) -> Result<
    ReplacementRecordingRequest<CanonicalReplacementOperation<Completion>>,
    Box<ReplacementExecProgramFailure<CanonicalReplacementOperation<Completion>>>,
> {
    resolve_prepared_exec_recording_at(
        input,
        resources,
        image_states,
        semantics,
        indirect_device,
        resolver,
        0,
        true,
        Some(operation_origins),
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "each input is one independently owned whole-EXEC preparation family"
)]
fn resolve_prepared_exec_recording_at<Completion: Clone + PartialEq>(
    input: ReplacementRecordingInput<CanonicalReplacementOperation<Completion>>,
    resources: &PreparedExecResources<
        ResolvedComputeDispatch,
        ReplacementComputePipelineVariant,
        ResolvedRenderDispatch,
        ReplacementRenderPipelineVariant,
    >,
    image_states: Option<&PreparedImageStateBatch>,
    semantics: ReplacementExecSemanticAdmissions<'_, Completion>,
    indirect_device: Option<Arc<dyn ReplacementIndirectRangeDevice>>,
    resolver: &(impl ReplacementComputeResolver
          + ReplacementRenderResolver
          + ReplacementBarrierResourceResolver
          + ReplacementBarrierResolver),
    operation_base: usize,
    continuation: bool,
    operation_origins: Option<&[reims_vgpu_core::ExpandedIndirectOperationOrigin]>,
) -> Result<
    ReplacementRecordingRequest<CanonicalReplacementOperation<Completion>>,
    Box<ReplacementExecProgramFailure<CanonicalReplacementOperation<Completion>>>,
> {
    let fail = |reason, input| Box::new(ReplacementExecProgramFailure { reason, input });
    if input.transaction != resources.transaction() {
        return Err(fail(
            ReplacementExecProgramError::TransactionMismatch,
            input,
        ));
    }
    let mut buffer_blits = Vec::<ReplacementBufferBlitProgram>::new();
    for prepared in &resources.inputs().buffer_blits {
        let Some(index) = prepared.write().operation_index() else {
            return Err(fail(
                ReplacementExecProgramError::UnpositionedBufferBlit,
                input,
            ));
        };
        let program =
            ReplacementBufferBlitProgram::resolve(index, prepared, resolver).map_err(|reason| {
                fail(
                    ReplacementExecProgramError::BufferBlit { index, reason },
                    input.clone(),
                )
            })?;
        buffer_blits.push(program);
    }

    let image_blits: Box<[ReplacementImageBlitProgram]> =
        if resources.inputs().image_blits.is_empty() {
            Box::new([])
        } else {
            let states = image_states.ok_or_else(|| {
                fail(
                    ReplacementExecProgramError::ImageStateBatchMissing,
                    input.clone(),
                )
            })?;
            resolve_exec_image_blit_programs(resources, states, resolver).map_err(|reason| {
                fail(
                    ReplacementExecProgramError::ImageBlit(reason),
                    input.clone(),
                )
            })?
        };
    let compute_programs = resolve_exec_compute_programs(resources, image_states, resolver)
        .map_err(|reason| fail(ReplacementExecProgramError::Compute(reason), input.clone()))?;
    let render_programs = resolve_exec_render_programs(resources, image_states, resolver)
        .map_err(|reason| fail(ReplacementExecProgramError::Render(reason), input.clone()))?;

    let mut info_queries = Vec::<ReplacementInfoQueryProgram>::new();
    for prepared in &resources.inputs().info_queries {
        let index = prepared.index();
        let program =
            ReplacementInfoQueryProgram::resolve(prepared, resolver).map_err(|reason| {
                fail(
                    ReplacementExecProgramError::InfoQuery { index, reason },
                    input.clone(),
                )
            })?;
        info_queries.push(program);
    }

    let mut indirect_ranges = Vec::<ReplacementIndirectRangeProgram>::new();
    for prepared in &resources.inputs().indirect_range_readbacks {
        let index = prepared.operation_index();
        let context = indirect_device.clone().ok_or_else(|| {
            fail(
                ReplacementExecProgramError::IndirectRangeDeviceMissing(index),
                input.clone(),
            )
        })?;
        let program = ReplacementIndirectRangeProgram::resolve(context, prepared, resolver)
            .map_err(|reason| {
                fail(
                    ReplacementExecProgramError::IndirectRange { index, reason },
                    input.clone(),
                )
            })?;
        indirect_ranges.push(program);
    }

    let resource_states = resources
        .inputs()
        .resource_states
        .as_ref()
        .map(|states| {
            ReplacementResourceStateBatchProgram::resolve_with_image_states(
                states,
                image_states,
                resolver,
            )
        })
        .transpose()
        .map_err(|reason| {
            fail(
                ReplacementExecProgramError::ResourceState(reason),
                input.clone(),
            )
        })?;
    let content_synchronization = resources
        .inputs()
        .content_synchronization
        .as_ref()
        .map(|prepared| {
            ReplacementContentSynchronizationProgram::resolve(prepared, image_states, resolver)
        })
        .transpose()
        .map_err(|reason| {
            fail(
                ReplacementExecProgramError::ContentSynchronization(reason),
                input.clone(),
            )
        })?;

    let semantic_programs = ReplacementSemanticAdmissions {
        conditions: semantics.conditions,
        completion_effects: semantics.completion_effects,
        indirect_commands: semantics.indirect_commands,
        resource_states: semantics.resource_states,
        info_queries: &info_queries,
        indirect_range_programs: &indirect_ranges,
    };
    let resolved = if let Some(operation_origins) = operation_origins {
        ReplacementRecordingRequest::resolve_continuation_with_native_programs_at_positions(
            input,
            resolver,
            resolver,
            &buffer_blits,
            &image_blits,
            semantic_programs,
            resource_states.as_ref(),
            &compute_programs,
            &render_programs,
            operation_origins,
        )
    } else if continuation {
        ReplacementRecordingRequest::resolve_continuation_with_native_programs(
            input,
            resolver,
            resolver,
            &buffer_blits,
            &image_blits,
            semantic_programs,
            resource_states.as_ref(),
            &compute_programs,
            &render_programs,
            operation_base,
        )
    } else {
        ReplacementRecordingRequest::resolve_with_native_programs(
            input,
            resolver,
            resolver,
            &buffer_blits,
            &image_blits,
            semantic_programs,
            resource_states.as_ref(),
            &compute_programs,
            &render_programs,
        )
    };
    let resolved = resolved.map_err(|failure| {
        fail(
            ReplacementExecProgramError::Recording(failure.reason),
            failure.input,
        )
    })?;
    match content_synchronization.as_ref() {
        Some(program) => resolved
            .attach_content_synchronization(program)
            .map_err(|input| {
                let input = *input;
                fail(
                    ReplacementExecProgramError::ContentSynchronizationTransactionMismatch,
                    ReplacementRecordingInput {
                        transaction: input.transaction,
                        worker: input.worker,
                        queue_family: input.queue_family,
                        exec: input.exec,
                        barriers: input.barriers,
                    },
                )
            }),
        None => Ok(resolved),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementExecRecordingError {
    TransactionMismatch,
    ImageStateBatchMissing,
    UnexpectedImageStateBatch,
    ImageStateProgram(ExecImageStateError),
    Recording(PreparedReplacementRecordingError),
}

#[derive(Debug)]
pub enum ReplacementExecRecordingRecovery<Semantic, Operation> {
    Input {
        prepared: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
        request: Box<ReplacementRecordingRequest<Operation>>,
    },
    Recording(
        Box<PreparedReplacementRecordingFailure<ResolvedReplayCompletion<Semantic>, Operation>>,
    ),
}

#[derive(Debug)]
pub struct ReplacementExecRecordingFailure<
    Semantic,
    Operation,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ReplacementExecRecordingError,
    pub recovery: ReplacementExecRecordingRecovery<Semantic, Operation>,
    pub resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    pub image_states: Option<PreparedImageStateBatch>,
}

#[must_use = "whole-EXEC semantic, resource, image, and recording ownership must be observed"]
pub struct PendingReplacementExecRecording<
    Semantic,
    Operation,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pending: PendingPreparedReplacementRecording<ResolvedReplayCompletion<Semantic>, Operation>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
}

#[derive(Debug)]
pub struct RecordedReplacementExec<
    Semantic,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    pub image_states: Option<PreparedImageStateBatch>,
}

pub enum ReplacementExecRecordingPoll<
    Semantic,
    Operation,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    Pending(
        PendingReplacementExecRecording<
            Semantic,
            Operation,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
    ),
    Completed(
        ExecRecordedResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>,
    ),
}

type ExecRecordedResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender> = Result<
    RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    Box<
        ReplacementExecRecordingFailure<
            Semantic,
            Operation,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
    >,
>;

type ExecRecordingInputResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender> =
    Result<
        PreparedExecRecordingInput<
            Semantic,
            Operation,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
        Box<
            ReplacementExecRecordingFailure<
                Semantic,
                Operation,
                Compute,
                NativeCompute,
                Render,
                NativeRender,
            >,
        >,
    >;

type ExecRecordingDispatchResult<
    Semantic,
    Operation,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> = Result<
    PendingReplacementExecRecording<
        Semantic,
        Operation,
        Compute,
        NativeCompute,
        Render,
        NativeRender,
    >,
    Box<
        ReplacementExecRecordingFailure<
            Semantic,
            Operation,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
    >,
>;

type ExecRecordingResult<Semantic, Operation> = Result<
    PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    Box<PreparedReplacementRecordingFailure<ResolvedReplayCompletion<Semantic>, Operation>>,
>;

#[derive(Debug)]
pub struct PreparedExecRecordingInput<
    Semantic,
    Operation,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    prepared: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    request: ReplacementRecordingRequest<Operation>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
}

type PreparedExecRecordingParts<Semantic, Operation, Compute, NativeCompute, Render, NativeRender> = (
    PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    ReplacementRecordingRequest<Operation>,
    PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    Option<PreparedImageStateBatch>,
);

impl<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
    PreparedExecRecordingInput<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
{
    pub(crate) fn into_parts(
        self,
    ) -> PreparedExecRecordingParts<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
    {
        (
            self.prepared,
            self.request,
            self.resources,
            self.image_states,
        )
    }
}

pub fn prepare_exec_recording_input<
    Semantic,
    Operation,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    prepared: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    request: ReplacementRecordingRequest<Operation>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
) -> ExecRecordingInputResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let transaction = prepared.plan().transaction;
    let has_images = exec_has_image_uses(&resources);
    let reason = if transaction != resources.transaction() || transaction != request.transaction {
        Some(ReplacementExecRecordingError::TransactionMismatch)
    } else if has_images.as_ref().is_ok_and(|has_images| *has_images) && image_states.is_none() {
        Some(ReplacementExecRecordingError::ImageStateBatchMissing)
    } else if has_images.as_ref().is_ok_and(|has_images| !*has_images)
        && image_states
            .as_ref()
            .is_some_and(|states| validate_exec_image_states(&resources, states).is_err())
    {
        Some(ReplacementExecRecordingError::UnexpectedImageStateBatch)
    } else if let Err(reason) = has_images {
        Some(ReplacementExecRecordingError::ImageStateProgram(reason))
    } else {
        image_states
            .as_ref()
            .and_then(|states| validate_exec_image_states(&resources, states).err())
            .map(ReplacementExecRecordingError::ImageStateProgram)
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementExecRecordingFailure {
            reason,
            recovery: ReplacementExecRecordingRecovery::Input {
                prepared,
                request: Box::new(request),
            },
            resources,
            image_states,
        }));
    }
    Ok(PreparedExecRecordingInput {
        prepared,
        request,
        resources,
        image_states,
    })
}

pub fn dispatch_prepared_exec_recording<
    Semantic,
    Operation,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    executor: &FixedExecutor<crate::replacement_recording::ReplacementRecordingWorker>,
    timelines: &QueueTimelineSemaphores,
    prepared: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    request: ReplacementRecordingRequest<Operation>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
) -> ExecRecordingDispatchResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
where
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    dispatch_prepared_exec_recording_with_additional_waits(
        executor,
        timelines,
        prepared,
        request,
        resources,
        image_states,
        Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
    )
}

pub fn dispatch_prepared_exec_recording_with_additional_waits<
    Semantic,
    Operation,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    executor: &FixedExecutor<crate::replacement_recording::ReplacementRecordingWorker>,
    timelines: &QueueTimelineSemaphores,
    prepared: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    request: ReplacementRecordingRequest<Operation>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
    additional_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
) -> ExecRecordingDispatchResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
where
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let PreparedExecRecordingInput {
        prepared,
        request,
        resources,
        image_states,
    } = prepare_exec_recording_input(prepared, request, resources, image_states)?;
    let mut auxiliary_waits = image_states
        .as_ref()
        .map(PreparedImageStateBatch::release_points)
        .unwrap_or_default()
        .into_vec();
    auxiliary_waits.extend(additional_waits);
    auxiliary_waits.sort_unstable();
    auxiliary_waits.dedup();
    let pending = match dispatch_prepared_replacement_recording_with_auxiliary_waits(
        executor,
        timelines,
        prepared,
        request,
        auxiliary_waits,
    ) {
        Ok(pending) => pending,
        Err(failure) => {
            let reason = ReplacementExecRecordingError::Recording(failure.reason);
            return Err(Box::new(ReplacementExecRecordingFailure {
                reason,
                recovery: ReplacementExecRecordingRecovery::Recording(failure),
                resources,
                image_states,
            }));
        }
    };
    Ok(PendingReplacementExecRecording {
        pending,
        resources,
        image_states,
    })
}

impl<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>
    PendingReplacementExecRecording<
        Semantic,
        Operation,
        Compute,
        NativeCompute,
        Render,
        NativeRender,
    >
{
    pub(crate) fn from_parts(
        pending: PendingPreparedReplacementRecording<ResolvedReplayCompletion<Semantic>, Operation>,
        resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
        image_states: Option<PreparedImageStateBatch>,
    ) -> Self {
        Self {
            pending,
            resources,
            image_states,
        }
    }

    pub fn try_complete(
        self,
    ) -> ReplacementExecRecordingPoll<
        Semantic,
        Operation,
        Compute,
        NativeCompute,
        Render,
        NativeRender,
    > {
        let Self {
            pending,
            resources,
            image_states,
        } = self;
        match pending.try_complete() {
            PreparedReplacementRecordingPoll::Pending(pending) => {
                ReplacementExecRecordingPoll::Pending(Self {
                    pending,
                    resources,
                    image_states,
                })
            }
            PreparedReplacementRecordingPoll::Completed(result) => {
                ReplacementExecRecordingPoll::Completed(finish_exec_recording(
                    result,
                    resources,
                    image_states,
                ))
            }
        }
    }

    pub fn wait(
        self,
    ) -> ExecRecordedResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender> {
        let Self {
            pending,
            resources,
            image_states,
        } = self;
        finish_exec_recording(pending.wait(), resources, image_states)
    }
}

fn finish_exec_recording<Semantic, Operation, Compute, NativeCompute, Render, NativeRender>(
    result: ExecRecordingResult<Semantic, Operation>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
) -> ExecRecordedResult<Semantic, Operation, Compute, NativeCompute, Render, NativeRender> {
    match result {
        Ok(submission) => Ok(RecordedReplacementExec {
            submission,
            resources,
            image_states,
        }),
        Err(failure) => {
            let reason = ReplacementExecRecordingError::Recording(failure.reason);
            Err(Box::new(ReplacementExecRecordingFailure {
                reason,
                recovery: ReplacementExecRecordingRecovery::Recording(failure),
                resources,
                image_states,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        replacement_barrier_record::{
            NativeBarrierBatch, ReplacementBarrierResolver, ReplacementBarrierResourceResolver,
        },
        replacement_image_state::{
            ReplacementImageKey, ReplacementImageSharing, ReplacementImageState,
            ReplacementImageStateOwner, ReplacementImageUse,
        },
        replacement_recording::{ReplacementRecordingInput, ReplacementRecordingRequest},
    };
    use ash::vk;
    use reims_vgpu_core::{
        assemble_prepared_exec_resources, DirectReplayNativeOwner, ExecTransaction,
        PreparedExecResourceInputs, ResolvedOperation, ResolvedReplayCompletion,
        TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        BackingId, QueueOwnerId, RepresentationId, ResourceId, SessionGenerationId, SubmissionId,
        SubmissionIdentity, TaskId, TransactionId, VulkanDeviceEpochId,
    };

    struct EmptyResolver;

    impl ReplacementBarrierResolver for EmptyResolver {
        fn resolve(
            &self,
            _backing: BackingId,
        ) -> Option<crate::replacement_barrier_record::NativeBarrierResolution> {
            None
        }
    }

    impl ReplacementBarrierResourceResolver for EmptyResolver {
        fn alias_backings(
            &self,
            _resource: ResourceId<reims_vgpu_protocol::ResourceObject>,
        ) -> Option<Box<[BackingId]>> {
            None
        }
    }

    impl crate::replacement_buffer_blit::ReplacementBufferResolver for EmptyResolver {
        fn resolve_buffer(
            &self,
            _backing: BackingId,
            _representation: RepresentationId,
        ) -> Option<crate::replacement_buffer_blit::NativeBufferTarget> {
            None
        }
    }

    impl crate::replacement_image_transition::ReplacementImageResolver for EmptyResolver {
        fn resolve_image(
            &self,
            _image: ReplacementImageKey,
        ) -> Option<crate::replacement_image_transition::NativeImageTarget> {
            None
        }
    }

    impl crate::replacement_compute::ReplacementComputeResolver for EmptyResolver {
        fn resolve_sampler(
            &self,
            _pipeline: ResourceId<reims_vgpu_protocol::ComputePipelineObject>,
            _sampler: &reims_vgpu_core::SamplerResource,
        ) -> Option<crate::replacement_sampler::ReplacementSamplerLease> {
            None
        }

        fn max_storage_buffer_range(&self) -> u64 {
            u64::MAX
        }

        fn min_storage_buffer_offset_alignment(&self) -> u64 {
            1
        }

        fn null_descriptors(&self) -> bool {
            false
        }
    }

    impl crate::replacement_render::ReplacementRenderResolver for EmptyResolver {
        fn resolve_sampler(
            &self,
            _pipeline: ResourceId<reims_vgpu_protocol::RenderPipelineObject>,
            _sampler: &reims_vgpu_core::SamplerResource,
        ) -> Option<crate::replacement_sampler::ReplacementSamplerLease> {
            None
        }

        fn max_storage_buffer_range(&self) -> u64 {
            u64::MAX
        }

        fn min_storage_buffer_offset_alignment(&self) -> u64 {
            1
        }

        fn max_viewports(&self) -> u32 {
            1
        }

        fn precise_occlusion_queries(&self) -> bool {
            false
        }

        fn null_descriptors(&self) -> bool {
            false
        }
    }

    type CanonicalTestOperation = CanonicalReplacementOperation<()>;

    fn empty_canonical_exec(submission: u64) -> ExecTransaction<CanonicalTestOperation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(submission),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        }
    }

    #[test]
    fn whole_exec_program_assembly_binds_the_exact_resource_transaction() {
        let resource_transaction = TransactionId::new(41);
        let exec = empty_canonical_exec(41);
        let resources = assemble_prepared_exec_resources(
            resource_transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([]),
                compute_dispatches: Vec::<
                    reims_vgpu_core::PreparedComputeDispatch<
                        ReplacementComputePipelineVariant,
                        ResolvedComputeDispatch,
                    >,
                >::new()
                .into_boxed_slice(),
                render_dispatches: Vec::<
                    reims_vgpu_core::PreparedRenderDispatch<
                        ReplacementRenderPipelineVariant,
                        ResolvedRenderDispatch,
                    >,
                >::new()
                .into_boxed_slice(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        let failure = resolve_prepared_exec_recording(
            ReplacementRecordingInput {
                transaction: TransactionId::new(42),
                worker: reims_vgpu_core::RecordingWorkerId::new(0),
                queue_family: 0,
                exec: empty_canonical_exec(41),
                barriers: NativeBarrierBatch::default(),
            },
            &resources,
            None,
            ReplacementExecSemanticAdmissions::default(),
            None,
            &EmptyResolver,
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementExecProgramError::TransactionMismatch
        );
        assert_eq!(failure.input.transaction, TransactionId::new(42));

        let request = resolve_prepared_exec_recording(
            ReplacementRecordingInput {
                transaction: resource_transaction,
                worker: reims_vgpu_core::RecordingWorkerId::new(0),
                queue_family: 0,
                exec,
                barriers: NativeBarrierBatch::default(),
            },
            &resources,
            None,
            ReplacementExecSemanticAdmissions::default(),
            None,
            &EmptyResolver,
        )
        .unwrap();
        assert_eq!(request.transaction, resource_transaction);
        assert!(request.required_queue_flags().is_empty());
    }

    #[test]
    fn join_refusal_returns_semantic_request_resources_and_image_state() {
        let epoch = VulkanDeviceEpochId::new(1);
        let transaction = TransactionId::new(7);
        let submission = SubmissionId::new(8);
        let exec = ExecTransaction::<
            ResolvedOperation<(), (), reims_vgpu_core::ResolvedInfoOperation, (), ()>,
        > {
            identity: SubmissionIdentity {
                id: submission,
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        };
        let resources = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([]),
                compute_dispatches:
                    Box::<[reims_vgpu_core::PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches:
                    Box::<[reims_vgpu_core::PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        let request = ReplacementRecordingRequest::resolve(
            ReplacementRecordingInput {
                transaction,
                worker: reims_vgpu_core::RecordingWorkerId::new(0),
                queue_family: 0,
                exec,
                barriers: NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap();
        let generation = SessionGenerationId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: reims_vgpu_protocol::SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = native
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(
                plan,
                QueueOwnerId::new(1),
                generation,
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: Box::new([]),
                },
            )
            .unwrap();
        let image = ReplacementImageKey {
            backing: BackingId::new(20),
            representation: RepresentationId::new(21),
        };
        let mut images = ReplacementImageStateOwner::new(epoch);
        images
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let states = images
            .prepare_batch(
                transaction,
                0,
                vec![(
                    0,
                    Box::new([ReplacementImageUse {
                        image,
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    }]) as Box<[_]>,
                )]
                .into_boxed_slice(),
            )
            .unwrap();
        let failure =
            prepare_exec_recording_input(prepared, request, resources, Some(states)).unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementExecRecordingError::UnexpectedImageStateBatch
        );
        assert_eq!(failure.resources.transaction(), transaction);
        images
            .validate_batch(failure.image_states.as_ref().unwrap())
            .unwrap();
        let ReplacementExecRecordingRecovery::Input { prepared, request } = failure.recovery else {
            panic!("pre-dispatch refusal must return the original input")
        };
        assert_eq!(prepared.plan().transaction, transaction);
        assert_eq!(request.transaction, transaction);
        images.cancel_batch(failure.image_states.unwrap()).unwrap();
    }
}
