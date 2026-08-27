//! Queue-point ownership for asynchronous indirect-range EXEC phases.
//!
//! The parent semantic point is allocated only after every range readback has
//! retired and become literal. While one auxiliary phase is outstanding this
//! owner withholds the point-free native chain, so a caller cannot allocate or
//! submit a successor phase early or replay an already submitted prefix.

use crate::replacement_indirect_range::{
    resume_indirect_range_after_timeline, ReplacementIndirectRangePhaseFailure,
    RetiredReplacementIndirectRanges,
};
use crate::{
    replacement_compute::ReplacementComputeImageBindings,
    replacement_exec_acceptance::{
        commit_driver_accepted_auxiliary_exec, commit_driver_accepted_exec_with_additional_waits,
        AcceptedReplacementAuxiliaryExec, AcceptedReplacementExec,
        ReplacementAuxiliaryExecAcceptanceFailure, ReplacementExecAcceptanceFailure,
        ReplacementExecAcceptanceOwners,
    },
    replacement_exec_image::{exec_has_image_uses, validate_exec_image_states},
    replacement_exec_queue::{
        enqueue_recorded_exec, PendingReplacementExecSubmit, ReplacementExecDriverPoll,
        ReplacementExecEnqueueFailure,
    },
    replacement_exec_recording::{
        prepare_exec_recording_input, PendingReplacementExecRecording, RecordedReplacementExec,
        ReplacementExecRecordingError, ReplacementExecRecordingFailure,
        ReplacementExecRecordingRecovery,
    },
    replacement_image_state::PreparedImageStateBatch,
    replacement_queue::{
        PendingPreparedReplacementAuxiliaryQueueSubmit,
        PreparedReplacementAuxiliaryQueueEnqueueFailure,
        PreparedReplacementAuxiliaryQueueSubmission, ReplacementQueueError,
    },
    replacement_recording::{ReplacementRecordingOperation, ReplacementRecordingRequest},
    replacement_render::ReplacementRenderImageBindings,
};
use reims_vgpu_core::{
    DirectReplayError, DirectReplayNativeOwner, IndirectRangeExecutionContinuation,
    IndirectRangeExecutionPhase, NativeChainFinalPreparationFailure, NextIndirectRangeExecution,
    PreparedAuxiliaryNativeSubmission, PreparedIndirectRangeReadback, PreparedNativeExecutionChain,
    PreparedNativeSubmission, QueueTimelinePoint,
};
use reims_vgpu_protocol::{QueueOwnerId, TransactionId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementIndirectExecChainError {
    TransactionMismatch {
        chain: TransactionId,
        continuation: TransactionId,
    },
    Auxiliary(DirectReplayError),
}

#[derive(Debug)]
pub struct ReplacementIndirectExecChainFailure<Semantic, Render, Compute, Info, Completion> {
    pub reason: ReplacementIndirectExecChainError,
    pub chain: ReplacementIndirectExecChain<Semantic, Render, Compute, Info, Completion>,
}

#[derive(Debug)]
pub struct ReplacementIndirectExecChain<Semantic, Render, Compute, Info, Completion> {
    native: PreparedNativeExecutionChain<Semantic>,
    continuation: IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>,
    last_auxiliary: Option<QueueTimelinePoint>,
}

type ChainResult<Semantic, Render, Compute, Info, Completion> = Result<
    ReplacementIndirectExecChain<Semantic, Render, Compute, Info, Completion>,
    Box<ReplacementIndirectExecChainFailure<Semantic, Render, Compute, Info, Completion>>,
>;

type NextPhaseResult<Semantic, Render, Compute, Info, Completion> = Result<
    NextReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
    Box<ReplacementIndirectExecChainFailure<Semantic, Render, Compute, Info, Completion>>,
>;

type ResumeResult<Semantic, Render, Compute, Info, Completion> = Result<
    ReplacementIndirectExecChain<Semantic, Render, Compute, Info, Completion>,
    Box<ReplacementIndirectExecResumeFailure<Semantic, Render, Compute, Info, Completion>>,
>;

impl<Semantic, Render, Compute, Info, Completion>
    ReplacementIndirectExecChain<Semantic, Render, Compute, Info, Completion>
{
    pub fn new(
        native: PreparedNativeExecutionChain<Semantic>,
        continuation: IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>,
    ) -> ChainResult<Semantic, Render, Compute, Info, Completion> {
        if native.plan().transaction != continuation.transaction() {
            let reason = ReplacementIndirectExecChainError::TransactionMismatch {
                chain: native.plan().transaction,
                continuation: continuation.transaction(),
            };
            return Err(Box::new(ReplacementIndirectExecChainFailure {
                reason,
                chain: Self {
                    native,
                    continuation,
                    last_auxiliary: None,
                },
            }));
        }
        Ok(Self {
            native,
            continuation,
            last_auxiliary: None,
        })
    }

    pub const fn transaction(&self) -> TransactionId {
        self.native.plan().transaction
    }

    pub const fn recording_worker(&self) -> reims_vgpu_core::RecordingWorkerId {
        self.native.recording_worker()
    }

    pub const fn last_auxiliary(&self) -> Option<QueueTimelinePoint> {
        self.last_auxiliary
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementIndirectAuxiliaryRecordingError {
    ExecMismatch,
    TransactionMismatch,
    ImageStateMismatch,
    Recording(crate::replacement_recording_queue::PreparedReplacementRecordingError),
}

#[derive(Debug)]
pub enum ReplacementIndirectAuxiliaryRecordingRecovery<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    Input {
        phase: Box<
            PreparedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion>,
        >,
        request:
            Box<ReplacementRecordingRequest<IndirectOperation<Render, Compute, Info, Completion>>>,
        resources: Box<
            reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
        >,
        image_states: Option<PreparedImageStateBatch>,
    },
    Recording {
        failure: Box<
            crate::replacement_recording_queue::PreparedReplacementAuxiliaryRecordingFailure<
                IndirectOperation<Render, Compute, Info, Completion>,
            >,
        >,
        continuation:
            Box<SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>>,
        resources: Box<
            reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
        >,
        image_states: Option<PreparedImageStateBatch>,
    },
}

#[derive(Debug)]
pub struct ReplacementIndirectAuxiliaryRecordingFailure<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pub reason: ReplacementIndirectAuxiliaryRecordingError,
    pub recovery: ReplacementIndirectAuxiliaryRecordingRecovery<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
}

#[must_use = "the auxiliary phase recording result must retain its continuation"]
pub struct PendingReplacementIndirectAuxiliaryRecording<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pending: crate::replacement_recording_queue::PendingPreparedReplacementAuxiliaryRecording<
        IndirectOperation<Render, Compute, Info, Completion>,
    >,
    continuation:
        SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
    resources: reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
}

#[derive(Debug)]
pub struct RecordedReplacementIndirectAuxiliaryExec<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    submission: PreparedReplacementAuxiliaryQueueSubmission,
    continuation:
        SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
    resources: reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
}

type AuxiliaryRecordingDispatchResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> = Result<
    PendingReplacementIndirectAuxiliaryRecording<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
    Box<
        ReplacementIndirectAuxiliaryRecordingFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    >,
>;

type AuxiliaryRecordedResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> = Result<
    RecordedReplacementIndirectAuxiliaryExec<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
    Box<
        ReplacementIndirectAuxiliaryRecordingFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    >,
>;

impl<Semantic, Render, Compute, Info, Completion>
    PreparedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion>
where
    Render: Clone + PartialEq + ReplacementRenderImageBindings,
    Compute: Clone + PartialEq + ReplacementComputeImageBindings,
    Info: Clone + PartialEq,
    Completion: Clone + PartialEq,
    IndirectOperation<Render, Compute, Info, Completion>:
        Clone + Send + ReplacementRecordingOperation + 'static,
{
    pub fn dispatch_recording<NativeCompute, NativeRender>(
        self,
        queues: &crate::replacement_epoch::ReplacementQueueEpoch,
        request: ReplacementRecordingRequest<IndirectOperation<Render, Compute, Info, Completion>>,
        resources: reims_vgpu_core::PreparedExecResources<
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
        image_states: Option<PreparedImageStateBatch>,
    ) -> AuxiliaryRecordingDispatchResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    > {
        let has_images = exec_has_image_uses(&resources);
        let reason = if !same_exec(&request.exec, self.phase().exec()) {
            Some(ReplacementIndirectAuxiliaryRecordingError::ExecMismatch)
        } else if resources.transaction() != self.prepared.transaction() {
            Some(ReplacementIndirectAuxiliaryRecordingError::TransactionMismatch)
        } else if has_images.is_err()
            || (has_images.as_ref().is_ok_and(|has_images| *has_images) && image_states.is_none())
            || image_states
                .as_ref()
                .is_some_and(|states| validate_exec_image_states(&resources, states).is_err())
        {
            Some(ReplacementIndirectAuxiliaryRecordingError::ImageStateMismatch)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(Box::new(ReplacementIndirectAuxiliaryRecordingFailure {
                reason,
                recovery: ReplacementIndirectAuxiliaryRecordingRecovery::Input {
                    phase: Box::new(self),
                    request: Box::new(request),
                    resources: Box::new(resources),
                    image_states,
                },
            }));
        }
        let waits = image_states
            .as_ref()
            .map(PreparedImageStateBatch::release_points)
            .unwrap_or_default();
        match queues.record_prepared_auxiliary(self.prepared, request, waits) {
            Ok(pending) => Ok(PendingReplacementIndirectAuxiliaryRecording {
                pending,
                continuation: self.continuation,
                resources,
                image_states,
            }),
            Err(failure) => Err(Box::new(ReplacementIndirectAuxiliaryRecordingFailure {
                reason: ReplacementIndirectAuxiliaryRecordingError::Recording(failure.reason),
                recovery: ReplacementIndirectAuxiliaryRecordingRecovery::Recording {
                    failure,
                    continuation: Box::new(self.continuation),
                    resources: Box::new(resources),
                    image_states,
                },
            })),
        }
    }
}

impl<Semantic, Render, Compute, Info, Completion, NativeCompute, NativeRender>
    PendingReplacementIndirectAuxiliaryRecording<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >
{
    pub fn wait(
        self,
    ) -> AuxiliaryRecordedResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    > {
        match self.pending.wait() {
            Ok(submission) => Ok(RecordedReplacementIndirectAuxiliaryExec {
                submission,
                continuation: self.continuation,
                resources: self.resources,
                image_states: self.image_states,
            }),
            Err(failure) => Err(Box::new(ReplacementIndirectAuxiliaryRecordingFailure {
                reason: ReplacementIndirectAuxiliaryRecordingError::Recording(failure.reason),
                recovery: ReplacementIndirectAuxiliaryRecordingRecovery::Recording {
                    failure,
                    continuation: Box::new(self.continuation),
                    resources: Box::new(self.resources),
                    image_states: self.image_states,
                },
            })),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementIndirectAuxiliaryEnqueueError {
    QueueAbsent(QueueOwnerId),
    Queue(ReplacementQueueError),
}

#[derive(Debug)]
pub struct ReplacementIndirectAuxiliaryEnqueueFailure<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pub reason: ReplacementIndirectAuxiliaryEnqueueError,
    pub recorded: RecordedReplacementIndirectAuxiliaryExec<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
}

#[must_use = "the auxiliary driver result must retain its continuation"]
pub struct PendingReplacementIndirectAuxiliarySubmit<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pending: PendingPreparedReplacementAuxiliaryQueueSubmit,
    continuation:
        SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
    resources: reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
}

type AuxiliaryEnqueueResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> = Result<
    PendingReplacementIndirectAuxiliarySubmit<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
    Box<
        ReplacementIndirectAuxiliaryEnqueueFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    >,
>;

impl<Semantic, Render, Compute, Info, Completion, NativeCompute, NativeRender>
    RecordedReplacementIndirectAuxiliaryExec<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >
{
    pub fn enqueue(
        self,
        queues: &crate::replacement_epoch::ReplacementQueueEpoch,
    ) -> AuxiliaryEnqueueResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    > {
        let queue = self.submission.prepared.point().queue;
        let Some(lane) = queues.lane(queue) else {
            return Err(Box::new(ReplacementIndirectAuxiliaryEnqueueFailure {
                reason: ReplacementIndirectAuxiliaryEnqueueError::QueueAbsent(queue),
                recorded: self,
            }));
        };
        let Self {
            submission,
            continuation,
            resources,
            image_states,
        } = self;
        match lane.submit.submit_auxiliary(submission) {
            Ok(pending) => Ok(PendingReplacementIndirectAuxiliarySubmit {
                pending,
                continuation,
                resources,
                image_states,
            }),
            Err(failure) => {
                let PreparedReplacementAuxiliaryQueueEnqueueFailure { reason, submission } =
                    *failure;
                Err(Box::new(ReplacementIndirectAuxiliaryEnqueueFailure {
                    reason: ReplacementIndirectAuxiliaryEnqueueError::Queue(reason),
                    recorded: Self {
                        submission,
                        continuation,
                        resources,
                        image_states,
                    },
                }))
            }
        }
    }
}

#[derive(Debug)]
pub struct DriverAcceptedReplacementIndirectAuxiliaryExec<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    recorded: RecordedReplacementIndirectAuxiliaryExec<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
}

pub enum ReplacementIndirectAuxiliaryDriverResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    DriverAccepted(
        DriverAcceptedReplacementIndirectAuxiliaryExec<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    ),
    DriverRefused {
        reason: ReplacementQueueError,
        recorded: RecordedReplacementIndirectAuxiliaryExec<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    },
}

impl<Semantic, Render, Compute, Info, Completion, NativeCompute, NativeRender>
    PendingReplacementIndirectAuxiliarySubmit<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >
{
    pub fn wait(
        self,
    ) -> ReplacementIndirectAuxiliaryDriverResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    > {
        let Self {
            pending,
            continuation,
            resources,
            image_states,
        } = self;
        match pending.wait() {
            Ok(submission) => ReplacementIndirectAuxiliaryDriverResult::DriverAccepted(
                DriverAcceptedReplacementIndirectAuxiliaryExec {
                    recorded: RecordedReplacementIndirectAuxiliaryExec {
                        submission,
                        continuation,
                        resources,
                        image_states,
                    },
                },
            ),
            Err(failure) => {
                let (reason, submission) = *failure;
                ReplacementIndirectAuxiliaryDriverResult::DriverRefused {
                    reason,
                    recorded: RecordedReplacementIndirectAuxiliaryExec {
                        submission,
                        continuation,
                        resources,
                        image_states,
                    },
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct AcceptedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion, T>
{
    continuation:
        SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
    accepted: AcceptedReplacementAuxiliaryExec<T, Compute, Render>,
}

#[derive(Debug)]
pub struct ReplacementIndirectAuxiliaryAcceptanceFailure<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pub failure: Box<
        ReplacementAuxiliaryExecAcceptanceFailure<Compute, NativeCompute, Render, NativeRender>,
    >,
    pub continuation:
        SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
}

type AuxiliaryAcceptanceResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
    T,
> = Result<
    AcceptedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion, T>,
    Box<
        ReplacementIndirectAuxiliaryAcceptanceFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    >,
>;

impl<Semantic, Render, Compute, Info, Completion, NativeCompute, NativeRender>
    DriverAcceptedReplacementIndirectAuxiliaryExec<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    pub fn accept<T>(
        self,
        resources: &mut reims_vgpu_core::ResourceLifecycleOwner<T>,
        recordings: &mut crate::replacement_replay::ReplacementRecordingOwner,
        queues: &mut crate::replacement_epoch::ReplacementQueueEpoch,
        images: &mut crate::replacement_image_state::ReplacementImageStateOwner,
    ) -> AuxiliaryAcceptanceResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
        T,
    > {
        let RecordedReplacementIndirectAuxiliaryExec {
            submission,
            continuation,
            resources: prepared_resources,
            image_states,
        } = self.recorded;
        match commit_driver_accepted_auxiliary_exec(
            resources,
            recordings,
            queues,
            images,
            submission,
            prepared_resources,
            image_states,
        ) {
            Ok(accepted) => Ok(AcceptedReplacementIndirectAuxiliaryPhase {
                continuation,
                accepted,
            }),
            Err(failure) => Err(Box::new(ReplacementIndirectAuxiliaryAcceptanceFailure {
                failure,
                continuation,
            })),
        }
    }
}

impl<Semantic, Render, Compute, Info, Completion, T>
    AcceptedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion, T>
{
    pub fn resume_after_timeline<Command>(
        self,
        owner: &reims_vgpu_core::IndirectCommandSlotOwner<Command>,
        retired: RetiredReplacementIndirectRanges,
    ) -> ResumeResult<Semantic, Render, Compute, Info, Completion> {
        self.continuation.resume_after_timeline(
            owner,
            self.accepted.point,
            self.accepted.outcomes.indirect_range_readbacks,
            retired,
        )
    }
}

type IndirectOperation<Render, Compute, Info, Completion> = reims_vgpu_core::ResolvedOperation<
    Render,
    Compute,
    Info,
    reims_vgpu_core::ResolvedIndirectCommand,
    Completion,
>;

type IndirectExecRecordingFailure<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> = ReplacementExecRecordingFailure<
    Semantic,
    IndirectOperation<Render, Compute, Info, Completion>,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>;

#[derive(Debug)]
pub enum ReplacementIndirectFinalRecordingRecovery<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    Input {
        final_phase: Box<
            PreparedReplacementIndirectFinalPhase<
                reims_vgpu_core::ResolvedReplayCompletion<Semantic>,
                Render,
                Compute,
                Info,
                Completion,
            >,
        >,
        request:
            Box<ReplacementRecordingRequest<IndirectOperation<Render, Compute, Info, Completion>>>,
        resources: Box<
            reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
        >,
        image_states: Option<PreparedImageStateBatch>,
    },
    Recording {
        failure: Box<
            IndirectExecRecordingFailure<
                Semantic,
                Render,
                Compute,
                Info,
                Completion,
                NativeCompute,
                NativeRender,
            >,
        >,
        additional_waits: Box<[QueueTimelinePoint]>,
    },
}

#[derive(Debug)]
pub struct ReplacementIndirectFinalRecordingFailure<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pub recovery: ReplacementIndirectFinalRecordingRecovery<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
}

#[must_use = "the final indirect-range recording result must retain its chain proof"]
pub struct PendingReplacementIndirectFinalRecording<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> {
    pending: PendingReplacementExecRecording<
        Semantic,
        IndirectOperation<Render, Compute, Info, Completion>,
        Compute,
        NativeCompute,
        Render,
        NativeRender,
    >,
    additional_waits: Box<[QueueTimelinePoint]>,
}

#[derive(Debug)]
pub struct RecordedReplacementChainedFinalExec<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> {
    recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    additional_waits: Box<[QueueTimelinePoint]>,
}

type IndirectFinalRecordedResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> = Result<
    RecordedReplacementChainedFinalExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    Box<
        ReplacementIndirectFinalRecordingFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    >,
>;

type IndirectFinalDispatchResult<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
    NativeCompute,
    NativeRender,
> = Result<
    PendingReplacementIndirectFinalRecording<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >,
    Box<
        ReplacementIndirectFinalRecordingFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
            NativeCompute,
            NativeRender,
        >,
    >,
>;

impl<Semantic, Render, Compute, Info, Completion>
    PreparedReplacementIndirectFinalPhase<
        reims_vgpu_core::ResolvedReplayCompletion<Semantic>,
        Render,
        Compute,
        Info,
        Completion,
    >
where
    Render: Clone + PartialEq + ReplacementRenderImageBindings,
    Compute: Clone + PartialEq + ReplacementComputeImageBindings,
    Info: Clone + PartialEq,
    Completion: Clone + PartialEq,
    IndirectOperation<Render, Compute, Info, Completion>:
        Clone + Send + ReplacementRecordingOperation + 'static,
{
    pub fn dispatch_recording<NativeCompute, NativeRender>(
        self,
        queues: &crate::replacement_epoch::ReplacementQueueEpoch,
        request: ReplacementRecordingRequest<IndirectOperation<Render, Compute, Info, Completion>>,
        resources: reims_vgpu_core::PreparedExecResources<
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
        image_states: Option<PreparedImageStateBatch>,
    ) -> IndirectFinalDispatchResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    > {
        if !same_exec(&request.exec, self.phase.exec()) {
            return Err(Box::new(ReplacementIndirectFinalRecordingFailure {
                recovery: ReplacementIndirectFinalRecordingRecovery::Input {
                    final_phase: Box::new(self),
                    request: Box::new(request),
                    resources: Box::new(resources),
                    image_states,
                },
            }));
        }
        let additional_waits = self.auxiliary_waits();
        let input =
            match prepare_exec_recording_input(self.prepared, request, resources, image_states) {
                Ok(input) => input,
                Err(failure) => {
                    return Err(Box::new(ReplacementIndirectFinalRecordingFailure {
                        recovery: ReplacementIndirectFinalRecordingRecovery::Recording {
                            failure,
                            additional_waits,
                        },
                    }));
                }
            };
        let (prepared, request, resources, image_states) = input.into_parts();
        match queues.record_prepared_with_auxiliary_waits(
            prepared,
            request,
            combined_phase_waits(image_states.as_ref(), &additional_waits),
        ) {
            Ok(pending) => Ok(PendingReplacementIndirectFinalRecording {
                pending: PendingReplacementExecRecording::from_parts(
                    pending,
                    resources,
                    image_states,
                ),
                additional_waits,
            }),
            Err(failure) => Err(Box::new(ReplacementIndirectFinalRecordingFailure {
                recovery: ReplacementIndirectFinalRecordingRecovery::Recording {
                    failure: Box::new(ReplacementExecRecordingFailure {
                        reason: ReplacementExecRecordingError::Recording(failure.reason),
                        recovery: ReplacementExecRecordingRecovery::Recording(failure),
                        resources,
                        image_states,
                    }),
                    additional_waits,
                },
            })),
        }
    }
}

fn same_exec<Operation: PartialEq>(
    left: &reims_vgpu_core::ExecTransaction<Operation>,
    right: &reims_vgpu_core::ExecTransaction<Operation>,
) -> bool {
    left.identity == right.identity
        && left.accesses == right.accesses
        && left.streams.len() == right.streams.len()
        && left
            .streams
            .iter()
            .zip(right.streams.iter())
            .all(|(left, right)| {
                left.stream_index == right.stream_index
                    && left.segments.len() == right.segments.len()
                    && left
                        .segments
                        .iter()
                        .zip(right.segments.iter())
                        .all(|(left, right)| {
                            left.boundary == right.boundary && left.operations == right.operations
                        })
            })
}

fn combined_phase_waits(
    image_states: Option<&PreparedImageStateBatch>,
    additional_waits: &[QueueTimelinePoint],
) -> Box<[QueueTimelinePoint]> {
    let mut waits = image_states
        .map(PreparedImageStateBatch::release_points)
        .unwrap_or_default()
        .into_vec();
    waits.extend_from_slice(additional_waits);
    waits.sort_unstable();
    waits.dedup();
    waits.into_boxed_slice()
}

impl<Semantic, Render, Compute, Info, Completion, NativeCompute, NativeRender>
    PendingReplacementIndirectFinalRecording<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    >
{
    pub fn wait(
        self,
    ) -> IndirectFinalRecordedResult<
        Semantic,
        Render,
        Compute,
        Info,
        Completion,
        NativeCompute,
        NativeRender,
    > {
        match self.pending.wait() {
            Ok(recorded) => Ok(RecordedReplacementChainedFinalExec {
                recorded,
                additional_waits: self.additional_waits,
            }),
            Err(failure) => Err(Box::new(ReplacementIndirectFinalRecordingFailure {
                recovery: ReplacementIndirectFinalRecordingRecovery::Recording {
                    failure,
                    additional_waits: self.additional_waits,
                },
            })),
        }
    }
}

#[derive(Debug)]
pub struct ReplacementChainedFinalEnqueueFailure<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> {
    pub failure:
        Box<ReplacementExecEnqueueFailure<Semantic, Compute, NativeCompute, Render, NativeRender>>,
    pub additional_waits: Box<[QueueTimelinePoint]>,
}

#[must_use = "the final indirect-range driver result must retain its chain proof"]
pub struct PendingReplacementChainedFinalSubmit<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> {
    pending: PendingReplacementExecSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>,
    additional_waits: Box<[QueueTimelinePoint]>,
}

type IndirectFinalEnqueueResult<Semantic, Compute, NativeCompute, Render, NativeRender> = Result<
    PendingReplacementChainedFinalSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>,
    Box<
        ReplacementChainedFinalEnqueueFailure<
            Semantic,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
    >,
>;

type IndirectFinalAcceptanceResult<Semantic, T, Compute, NativeCompute, Render, NativeRender> =
    Result<
        AcceptedReplacementExec<T, Compute, Render>,
        Box<
            ReplacementChainedFinalAcceptanceFailure<
                Semantic,
                Compute,
                NativeCompute,
                Render,
                NativeRender,
            >,
        >,
    >;

pub enum ReplacementChainedFinalDriverPoll<Semantic, Compute, NativeCompute, Render, NativeRender> {
    Pending(
        PendingReplacementChainedFinalSubmit<
            Semantic,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
    ),
    DriverAccepted(
        RecordedReplacementChainedFinalExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    ),
    DriverRefused {
        reason: crate::replacement_queue::ReplacementQueueError,
        recorded: RecordedReplacementChainedFinalExec<
            Semantic,
            Compute,
            NativeCompute,
            Render,
            NativeRender,
        >,
    },
}

impl<Semantic, Compute, NativeCompute, Render, NativeRender>
    RecordedReplacementChainedFinalExec<Semantic, Compute, NativeCompute, Render, NativeRender>
{
    pub fn from_recorded(
        recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
        additional_waits: impl Into<Box<[QueueTimelinePoint]>>,
    ) -> Self {
        Self {
            recorded,
            additional_waits: additional_waits.into(),
        }
    }

    pub const fn submission(
        &self,
    ) -> &crate::replacement_queue::PreparedReplacementQueueSubmission<
        reims_vgpu_core::ResolvedReplayCompletion<Semantic>,
    > {
        &self.recorded.submission
    }

    pub const fn additional_waits(&self) -> &[QueueTimelinePoint] {
        &self.additional_waits
    }

    pub fn enqueue(
        self,
        queues: &crate::replacement_epoch::ReplacementQueueEpoch,
    ) -> IndirectFinalEnqueueResult<Semantic, Compute, NativeCompute, Render, NativeRender> {
        match enqueue_recorded_exec(queues, self.recorded) {
            Ok(pending) => Ok(PendingReplacementChainedFinalSubmit {
                pending,
                additional_waits: self.additional_waits,
            }),
            Err(failure) => Err(Box::new(ReplacementChainedFinalEnqueueFailure {
                failure,
                additional_waits: self.additional_waits,
            })),
        }
    }

    pub fn accept_driver<T>(
        self,
        owners: ReplacementExecAcceptanceOwners<'_, Semantic, T>,
    ) -> IndirectFinalAcceptanceResult<Semantic, T, Compute, NativeCompute, Render, NativeRender>
    where
        Semantic: Clone,
        Compute: ReplacementComputeImageBindings,
        Render: ReplacementRenderImageBindings,
    {
        let RecordedReplacementExec {
            submission,
            resources,
            image_states,
        } = self.recorded;
        match commit_driver_accepted_exec_with_additional_waits(
            owners,
            submission,
            resources,
            image_states,
            self.additional_waits.clone(),
        ) {
            Ok(accepted) => Ok(accepted),
            Err(failure) => Err(Box::new(ReplacementChainedFinalAcceptanceFailure {
                failure,
                additional_waits: self.additional_waits,
            })),
        }
    }
}

#[derive(Debug)]
pub struct ReplacementChainedFinalAcceptanceFailure<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> {
    pub failure: Box<
        ReplacementExecAcceptanceFailure<Semantic, Compute, NativeCompute, Render, NativeRender>,
    >,
    pub additional_waits: Box<[QueueTimelinePoint]>,
}

impl<Semantic, Compute, NativeCompute, Render, NativeRender>
    PendingReplacementChainedFinalSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>
where
    Semantic: Clone,
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    pub fn poll_driver(
        self,
    ) -> ReplacementChainedFinalDriverPoll<Semantic, Compute, NativeCompute, Render, NativeRender>
    {
        match self.pending.poll_driver() {
            ReplacementExecDriverPoll::Pending(pending) => {
                ReplacementChainedFinalDriverPoll::Pending(Self {
                    pending,
                    additional_waits: self.additional_waits,
                })
            }
            ReplacementExecDriverPoll::DriverAccepted(recorded) => {
                ReplacementChainedFinalDriverPoll::DriverAccepted(
                    RecordedReplacementChainedFinalExec {
                        recorded,
                        additional_waits: self.additional_waits,
                    },
                )
            }
            ReplacementExecDriverPoll::DriverRefused { reason, recorded } => {
                ReplacementChainedFinalDriverPoll::DriverRefused {
                    reason,
                    recorded: RecordedReplacementChainedFinalExec {
                        recorded,
                        additional_waits: self.additional_waits,
                    },
                }
            }
        }
    }
}

// Indirect-range execution was the first consumer of the general chained
// final-submit owner. Keep its vocabulary as aliases while other asynchronous
// EXEC prefixes use the owner under its contract-level name.
pub type RecordedReplacementIndirectFinalExec<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> = RecordedReplacementChainedFinalExec<Semantic, Compute, NativeCompute, Render, NativeRender>;
pub type PendingReplacementIndirectFinalSubmit<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> = PendingReplacementChainedFinalSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>;
pub type ReplacementIndirectFinalDriverPoll<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> = ReplacementChainedFinalDriverPoll<Semantic, Compute, NativeCompute, Render, NativeRender>;
pub type ReplacementIndirectFinalEnqueueFailure<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> = ReplacementChainedFinalEnqueueFailure<Semantic, Compute, NativeCompute, Render, NativeRender>;
pub type ReplacementIndirectFinalAcceptanceFailure<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
> = ReplacementChainedFinalAcceptanceFailure<
    Semantic,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>;

impl<Semantic, Render: Clone, Compute: Clone, Info: Clone, Completion: Clone>
    ReplacementIndirectExecChain<Semantic, Render, Compute, Info, Completion>
{
    pub fn prepare_next(
        self,
        native: &mut DirectReplayNativeOwner<Semantic>,
        queue: QueueOwnerId,
    ) -> NextPhaseResult<Semantic, Render, Compute, Info, Completion>
    where
        Semantic: Clone,
    {
        let Self {
            native: native_chain,
            continuation,
            last_auxiliary,
        } = self;
        let recovery_continuation = continuation.clone();
        match continuation.next() {
            NextIndirectRangeExecution::Readback(pending) => {
                let prepared = match last_auxiliary {
                    Some(predecessor) => {
                        native.prepare_execution_chain_auxiliary_after(&native_chain, predecessor)
                    }
                    None => native.prepare_execution_chain_auxiliary(&native_chain, queue),
                };
                let prepared = match prepared {
                    Ok(prepared) => prepared,
                    Err(reason) => {
                        return Err(Box::new(ReplacementIndirectExecChainFailure {
                            reason: ReplacementIndirectExecChainError::Auxiliary(reason),
                            chain: Self {
                                native: native_chain,
                                continuation: recovery_continuation,
                                last_auxiliary,
                            },
                        }));
                    }
                };
                let point = prepared.point();
                Ok(NextReplacementIndirectExecPhase::Auxiliary(Box::new(
                    PreparedReplacementIndirectAuxiliaryPhase {
                        prepared,
                        continuation: SubmittedReplacementIndirectExecPhase {
                            native: native_chain,
                            pending,
                            point,
                        },
                    },
                )))
            }
            NextIndirectRangeExecution::Final(phase) => Ok(
                NextReplacementIndirectExecPhase::Final(PendingReplacementIndirectFinalPhase {
                    native: native_chain,
                    phase,
                    last_auxiliary,
                }),
            ),
        }
    }
}

#[derive(Debug)]
pub enum NextReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion> {
    Auxiliary(
        Box<PreparedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion>>,
    ),
    Final(PendingReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion>),
}

#[derive(Debug)]
pub struct PreparedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion> {
    prepared: PreparedAuxiliaryNativeSubmission,
    continuation:
        SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>,
}

impl<Semantic, Render, Compute, Info, Completion>
    PreparedReplacementIndirectAuxiliaryPhase<Semantic, Render, Compute, Info, Completion>
{
    pub const fn phase(&self) -> &IndirectRangeExecutionPhase<Render, Compute, Info, Completion> {
        self.continuation.phase()
    }

    pub const fn point(&self) -> QueueTimelinePoint {
        self.continuation.point()
    }
}

#[derive(Debug)]
pub struct SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion> {
    native: PreparedNativeExecutionChain<Semantic>,
    pending: reims_vgpu_core::PendingIndirectRangeExecution<Render, Compute, Info, Completion>,
    point: QueueTimelinePoint,
}

impl<Semantic, Render, Compute, Info, Completion>
    SubmittedReplacementIndirectExecPhase<Semantic, Render, Compute, Info, Completion>
{
    pub const fn phase(&self) -> &IndirectRangeExecutionPhase<Render, Compute, Info, Completion> {
        self.pending.phase()
    }

    pub const fn point(&self) -> QueueTimelinePoint {
        self.point
    }

    pub fn resume_after_timeline<Command>(
        self,
        owner: &reims_vgpu_core::IndirectCommandSlotOwner<Command>,
        observed: QueueTimelinePoint,
        readbacks: Box<[PreparedIndirectRangeReadback]>,
        retired: RetiredReplacementIndirectRanges,
    ) -> ResumeResult<Semantic, Render, Compute, Info, Completion> {
        if observed != self.point {
            return Err(Box::new(ReplacementIndirectExecResumeFailure {
                native: self.native,
                expected: self.point,
                observed,
                reason: ReplacementIndirectExecResumeError::TimelinePointMismatch {
                    readbacks,
                    retired,
                },
            }));
        }
        match resume_indirect_range_after_timeline(owner, self.pending, readbacks, retired) {
            Ok(continuation) => Ok(ReplacementIndirectExecChain {
                native: self.native,
                continuation,
                last_auxiliary: Some(observed),
            }),
            Err(failure) => Err(Box::new(ReplacementIndirectExecResumeFailure {
                native: self.native,
                expected: self.point,
                observed,
                reason: ReplacementIndirectExecResumeError::Range(failure),
            })),
        }
    }
}

#[derive(Debug)]
pub enum ReplacementIndirectExecResumeError<Render, Compute, Info, Completion> {
    TimelinePointMismatch {
        readbacks: Box<[PreparedIndirectRangeReadback]>,
        retired: RetiredReplacementIndirectRanges,
    },
    Range(Box<ReplacementIndirectRangePhaseFailure<Render, Compute, Info, Completion>>),
}

#[derive(Debug)]
pub struct ReplacementIndirectExecResumeFailure<Semantic, Render, Compute, Info, Completion> {
    pub native: PreparedNativeExecutionChain<Semantic>,
    pub expected: QueueTimelinePoint,
    pub observed: QueueTimelinePoint,
    pub reason: ReplacementIndirectExecResumeError<Render, Compute, Info, Completion>,
}

#[derive(Debug)]
pub struct PreparedReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion> {
    pub prepared: PreparedNativeSubmission<Semantic>,
    pub phase: IndirectRangeExecutionPhase<Render, Compute, Info, Completion>,
    last_auxiliary: Option<QueueTimelinePoint>,
}

#[derive(Debug)]
pub struct PendingReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion> {
    native: PreparedNativeExecutionChain<Semantic>,
    pub phase: IndirectRangeExecutionPhase<Render, Compute, Info, Completion>,
    last_auxiliary: Option<QueueTimelinePoint>,
}

type FinalPreparationResult<Semantic, Render, Compute, Info, Completion> = Result<
    PreparedReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion>,
    Box<
        PendingReplacementIndirectFinalPreparationFailure<
            Semantic,
            Render,
            Compute,
            Info,
            Completion,
        >,
    >,
>;

impl<Semantic, Render, Compute, Info, Completion>
    PendingReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion>
{
    pub fn map_semantic(self, map: impl FnOnce(Semantic) -> Semantic) -> Self {
        PendingReplacementIndirectFinalPhase {
            native: self.native.map_semantic(map),
            phase: self.phase,
            last_auxiliary: self.last_auxiliary,
        }
    }

    pub fn prepare(
        self,
        native: &mut DirectReplayNativeOwner<Semantic>,
        queue: QueueOwnerId,
    ) -> FinalPreparationResult<Semantic, Render, Compute, Info, Completion>
    where
        Semantic: Clone,
    {
        match native.prepare_execution_chain_final(self.native, queue) {
            Ok(prepared) => Ok(PreparedReplacementIndirectFinalPhase {
                prepared,
                phase: self.phase,
                last_auxiliary: self.last_auxiliary,
            }),
            Err(NativeChainFinalPreparationFailure { reason, chain }) => Err(Box::new(
                PendingReplacementIndirectFinalPreparationFailure {
                    reason,
                    pending: Self {
                        native: chain,
                        phase: self.phase,
                        last_auxiliary: self.last_auxiliary,
                    },
                },
            )),
        }
    }
}

#[derive(Debug)]
pub struct PendingReplacementIndirectFinalPreparationFailure<
    Semantic,
    Render,
    Compute,
    Info,
    Completion,
> {
    pub reason: DirectReplayError,
    pub pending: PendingReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion>,
}

impl<Semantic, Render, Compute, Info, Completion>
    PreparedReplacementIndirectFinalPhase<Semantic, Render, Compute, Info, Completion>
{
    pub fn auxiliary_waits(&self) -> Box<[QueueTimelinePoint]> {
        self.last_auxiliary
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use reims_vgpu_core::{
        assemble_prepared_exec_resources, ExecTransaction, LinearRange, PreparedExecResourceInputs,
        ResolvedExecSegment, ResolvedExecStream, ResolvedIndirectCommand, ResolvedOperation,
        ResolvedReplayCompletion, TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        BackingId, IndirectCommandBufferObject, ResourceId, ResourceObject, SegmentBoundary,
        SegmentKind, SessionGenerationId, SubmissionDomainId, SubmissionId, SubmissionIdentity,
        TaskId, VulkanDeviceEpochId,
    };
    use std::sync::mpsc;

    type Operation = ResolvedOperation<
        (),
        (),
        reims_vgpu_core::ResolvedInfoOperation,
        ResolvedIndirectCommand,
        (),
    >;

    fn exec(operation: Option<Operation>) -> ExecTransaction<Operation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: operation.map_or_else(
                || Vec::<ResolvedExecStream<Operation>>::new().into_boxed_slice(),
                |operation| {
                    vec![ResolvedExecStream {
                        stream_index: 0,
                        segments: Box::new([ResolvedExecSegment {
                            boundary: SegmentBoundary {
                                stream_index: 0,
                                index: 0,
                                kind: SegmentKind::Blit,
                                continues_previous: false,
                                continues_next: false,
                            },
                            operations: Box::new([operation]),
                        }]),
                    }]
                    .into_boxed_slice()
                },
            ),
            accesses: Box::new([]),
        }
    }

    fn native_chain(
        owner: &mut DirectReplayNativeOwner<&'static str>,
        transaction: TransactionId,
    ) -> PreparedNativeExecutionChain<&'static str> {
        owner
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = owner
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        owner
            .prepare_execution_chain(plan, SessionGenerationId::new(1), "done")
            .unwrap()
    }

    struct EmptyResolver;

    impl crate::replacement_barrier_record::ReplacementBarrierResolver for EmptyResolver {
        fn resolve(
            &self,
            _: BackingId,
        ) -> Option<crate::replacement_barrier_record::NativeBarrierResolution> {
            None
        }
    }

    impl crate::replacement_barrier_record::ReplacementBarrierResourceResolver for EmptyResolver {
        fn alias_backings(&self, _: ResourceId<ResourceObject>) -> Option<Box<[BackingId]>> {
            None
        }
    }

    #[test]
    fn unresolved_range_allocates_an_auxiliary_before_any_semantic_point() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(2);
        let transaction = TransactionId::new(3);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        let operation =
            ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::ExecuteIndirectRange {
                icb: ResourceId::<IndirectCommandBufferObject>::new(4, 1),
                arguments_resource: ResourceId::<ResourceObject>::new(5, 1),
                arguments_backing: BackingId::new(6),
                arguments_range: LinearRange::new(0, 8).unwrap(),
                kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
            });
        let continuation =
            IndirectRangeExecutionContinuation::new(transaction, exec(Some(operation)));
        let chain =
            ReplacementIndirectExecChain::new(native_chain(&mut native, transaction), continuation)
                .unwrap();
        let NextReplacementIndirectExecPhase::Auxiliary(auxiliary) =
            chain.prepare_next(&mut native, queue).unwrap()
        else {
            panic!("an unresolved range cannot become the final semantic phase")
        };
        assert_eq!(auxiliary.point().value.get(), 1);
        assert_eq!(auxiliary.phase().operation_base(), 0);
        assert_eq!(auxiliary.phase().exec().operations().count(), 1);
        assert_eq!(native.pending_completions(), 0);
    }

    #[test]
    fn range_free_chain_allocates_its_first_point_as_the_semantic_final() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(2);
        let transaction = TransactionId::new(3);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        let continuation = IndirectRangeExecutionContinuation::new(transaction, exec(None));
        let chain =
            ReplacementIndirectExecChain::new(native_chain(&mut native, transaction), continuation)
                .unwrap();
        let NextReplacementIndirectExecPhase::Final(final_phase) =
            chain.prepare_next(&mut native, queue).unwrap()
        else {
            panic!("a range-free EXEC is final")
        };
        let final_phase = final_phase.prepare(&mut native, queue).unwrap();
        assert_eq!(final_phase.prepared.point().value.get(), 1);
        assert!(final_phase.auxiliary_waits().is_empty());
        assert_eq!(native.pending_completions(), 0);
    }

    #[test]
    fn final_recording_retains_the_last_range_phase_as_an_exact_wait() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { crate::engine::context::DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP indirect final recording: no device ({error})");
                return;
            }
        };
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(2);
        let transaction = TransactionId::new(3);
        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            context.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type),
                None,
            )
        }
        .unwrap();
        let features = crate::device_features::DeviceFeatures {
            timeline_semaphore: true,
            ..Default::default()
        };
        let capabilities = crate::replacement_capabilities::ReplacementCapabilities::require(
            &features,
            reims_vgpu_core::DescriptorCapabilities {
                descriptor_buffer: false,
                push_descriptor: false,
            },
        )
        .unwrap();
        let actual_queue = unsafe { context.device.get_device_queue(context.gq, 0) };
        let queues = crate::replacement_epoch::ReplacementQueueEpoch::start(
            epoch,
            capabilities,
            &context.device,
            1,
            [crate::replacement_epoch::ReplacementQueueBinding {
                id: queue,
                queue_family: context.gq,
                flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                queue: actual_queue,
                timeline,
            }],
        )
        .unwrap();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
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
        let prepared_chain = native
            .prepare_execution_chain(
                plan,
                SessionGenerationId::new(1),
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: Box::new([]),
                },
            )
            .unwrap();
        let auxiliary = native
            .prepare_execution_chain_auxiliary(&prepared_chain, queue)
            .unwrap();
        let chain = ReplacementIndirectExecChain {
            native: prepared_chain,
            continuation: IndirectRangeExecutionContinuation::new(transaction, exec(None)),
            last_auxiliary: Some(auxiliary.point()),
        };
        let NextReplacementIndirectExecPhase::Final(final_phase) =
            chain.prepare_next(&mut native, queue).unwrap()
        else {
            panic!("the range-free suffix must be final")
        };
        let empty_resources = Box::new([]);
        let final_phase = final_phase
            .map_semantic(|completion| ResolvedReplayCompletion {
                semantic: completion.semantic,
                resources: empty_resources,
            })
            .prepare(&mut native, queue)
            .unwrap();
        assert_eq!(final_phase.prepared.point().value.get(), 2);
        let phase_exec = final_phase.phase.exec().clone();
        let resources = assemble_prepared_exec_resources(
            transaction,
            &phase_exec,
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
        let request = crate::replacement_recording::ReplacementRecordingRequest::resolve(
            crate::replacement_recording::ReplacementRecordingInput {
                transaction,
                worker: reims_vgpu_core::RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: phase_exec,
                barriers: crate::replacement_barrier_record::NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap();
        let recorded = final_phase
            .dispatch_recording(&queues, request, resources, None)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            recorded.recorded.submission.auxiliary_waits(),
            [auxiliary.point()]
        );
        let (_, recording) = recorded.recorded.submission.into_parts();
        let (sender, receiver) = mpsc::sync_channel(1);
        queues
            .recording_workers()
            .submit_to(recording.worker, move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();

        let other_transaction = TransactionId::new(4);
        let mut other_native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        let operation =
            ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::ExecuteIndirectRange {
                icb: ResourceId::<IndirectCommandBufferObject>::new(4, 1),
                arguments_resource: ResourceId::<ResourceObject>::new(5, 1),
                arguments_backing: BackingId::new(6),
                arguments_range: LinearRange::new(0, 8).unwrap(),
                kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
            });
        let other_chain = ReplacementIndirectExecChain::new(
            native_chain(&mut other_native, other_transaction),
            IndirectRangeExecutionContinuation::new(other_transaction, exec(Some(operation))),
        )
        .unwrap();
        let NextReplacementIndirectExecPhase::Auxiliary(other_phase) =
            other_chain.prepare_next(&mut other_native, queue).unwrap()
        else {
            panic!("the unresolved range must require an auxiliary phase")
        };
        let wrong_exec = exec(None);
        let wrong_resources = assemble_prepared_exec_resources(
            other_transaction,
            &wrong_exec,
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
        let wrong_request = crate::replacement_recording::ReplacementRecordingRequest::resolve(
            crate::replacement_recording::ReplacementRecordingInput {
                transaction: other_transaction,
                worker: reims_vgpu_core::RecordingWorkerId::new(0),
                queue_family: context.gq,
                exec: wrong_exec,
                barriers: crate::replacement_barrier_record::NativeBarrierBatch::default(),
            },
            &EmptyResolver,
            &EmptyResolver,
        )
        .unwrap();
        let failure =
            match other_phase.dispatch_recording(&queues, wrong_request, wrong_resources, None) {
                Ok(_) => panic!("a mismatched phase EXEC must not reach a recording worker"),
                Err(failure) => failure,
            };
        assert_eq!(
            failure.reason,
            ReplacementIndirectAuxiliaryRecordingError::ExecMismatch
        );
        assert!(matches!(
            failure.recovery,
            ReplacementIndirectAuxiliaryRecordingRecovery::Input { phase, .. }
                if phase.point().value.get() == 1
        ));

        drop(queues);
        unsafe { context.device.destroy_semaphore(timeline, None) };
        unsafe { context.destroy() };
    }
}
