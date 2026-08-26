//! Whole-EXEC ownership through physical queue admission and driver receipt.

use crate::{
    replacement_compute::ReplacementComputeImageBindings,
    replacement_exec_acceptance::{
        commit_driver_accepted_exec, AcceptedReplacementExec, ReplacementExecAcceptanceFailure,
        ReplacementExecAcceptanceOwners,
    },
    replacement_exec_recording::RecordedReplacementExec,
    replacement_render::ReplacementRenderImageBindings,
    replacement_replay::{
        enqueue_replacement_replay, PendingReplacementReplaySubmit, ReplacementReplayDriverPoll,
        ReplacementReplayEnqueueError,
    },
};
use reims_vgpu_core::ResolvedReplayCompletion;

#[derive(Debug)]
pub struct ReplacementExecEnqueueFailure<
    Semantic,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ReplacementReplayEnqueueError,
    pub recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
}

type ExecEnqueueResult<Semantic, Compute, NativeCompute, Render, NativeRender> = Result<
    PendingReplacementExecSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>,
    Box<ReplacementExecEnqueueFailure<Semantic, Compute, NativeCompute, Render, NativeRender>>,
>;

#[must_use = "an enqueued whole EXEC must be observed through driver acceptance"]
pub struct PendingReplacementExecSubmit<
    Semantic,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pending: PendingReplacementReplaySubmit<ResolvedReplayCompletion<Semantic>>,
    resources: reims_vgpu_core::PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<crate::replacement_image_state::PreparedImageStateBatch>,
}

#[cfg(test)]
impl<Semantic, Compute, NativeCompute, Render, NativeRender>
    PendingReplacementExecSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>
{
    fn disconnected_for_test(
        recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    ) -> Self {
        Self {
            pending: PendingReplacementReplaySubmit::disconnected_for_test(recorded.submission),
            resources: recorded.resources,
            image_states: recorded.image_states,
        }
    }
}

pub enum ReplacementExecSubmitPoll<
    Semantic,
    T,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    Pending(PendingReplacementExecSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>),
    Accepted(AcceptedReplacementExec<T, Compute, Render>),
    DriverRefused {
        reason: crate::replacement_queue::ReplacementQueueError,
        recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    },
    AcceptanceRefused(
        Box<
            ReplacementExecAcceptanceFailure<
                Semantic,
                Compute,
                NativeCompute,
                Render,
                NativeRender,
            >,
        >,
    ),
}

pub enum ReplacementExecDriverPoll<
    Semantic,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    Pending(PendingReplacementExecSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>),
    DriverAccepted(RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>),
    DriverRefused {
        reason: crate::replacement_queue::ReplacementQueueError,
        recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
    },
}

pub fn enqueue_recorded_exec<Semantic, Compute, NativeCompute, Render, NativeRender>(
    queues: &crate::replacement_epoch::ReplacementQueueEpoch,
    recorded: RecordedReplacementExec<Semantic, Compute, NativeCompute, Render, NativeRender>,
) -> ExecEnqueueResult<Semantic, Compute, NativeCompute, Render, NativeRender> {
    let RecordedReplacementExec {
        submission,
        resources,
        image_states,
    } = recorded;
    match enqueue_replacement_replay(queues, submission) {
        Ok(pending) => Ok(PendingReplacementExecSubmit {
            pending,
            resources,
            image_states,
        }),
        Err(failure) => Err(Box::new(ReplacementExecEnqueueFailure {
            reason: failure.reason,
            recorded: RecordedReplacementExec {
                submission: failure.submission,
                resources,
                image_states,
            },
        })),
    }
}

impl<
        Semantic: Clone,
        Compute: ReplacementComputeImageBindings,
        NativeCompute,
        Render: ReplacementRenderImageBindings,
        NativeRender,
    > PendingReplacementExecSubmit<Semantic, Compute, NativeCompute, Render, NativeRender>
{
    pub fn poll_driver(
        self,
    ) -> ReplacementExecDriverPoll<Semantic, Compute, NativeCompute, Render, NativeRender> {
        let Self {
            pending,
            resources,
            image_states,
        } = self;
        match pending.poll_driver() {
            ReplacementReplayDriverPoll::Pending(pending) => {
                ReplacementExecDriverPoll::Pending(Self {
                    pending,
                    resources,
                    image_states,
                })
            }
            ReplacementReplayDriverPoll::DriverAccepted(accepted) => {
                ReplacementExecDriverPoll::DriverAccepted(RecordedReplacementExec {
                    submission: accepted.into_submission(),
                    resources,
                    image_states,
                })
            }
            ReplacementReplayDriverPoll::DriverRefused { reason, submission } => {
                ReplacementExecDriverPoll::DriverRefused {
                    reason,
                    recorded: RecordedReplacementExec {
                        submission,
                        resources,
                        image_states,
                    },
                }
            }
        }
    }

    pub fn try_complete<T>(
        self,
        owners: ReplacementExecAcceptanceOwners<'_, Semantic, T>,
    ) -> ReplacementExecSubmitPoll<Semantic, T, Compute, NativeCompute, Render, NativeRender> {
        match self.poll_driver() {
            ReplacementExecDriverPoll::Pending(pending) => {
                ReplacementExecSubmitPoll::Pending(pending)
            }
            ReplacementExecDriverPoll::DriverRefused { reason, recorded } => {
                ReplacementExecSubmitPoll::DriverRefused { reason, recorded }
            }
            ReplacementExecDriverPoll::DriverAccepted(recorded) => {
                match commit_driver_accepted_exec(
                    owners,
                    recorded.submission,
                    recorded.resources,
                    recorded.image_states,
                ) {
                    Ok(accepted) => ReplacementExecSubmitPoll::Accepted(accepted),
                    Err(failure) => ReplacementExecSubmitPoll::AcceptanceRefused(failure),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        replacement_queue::PreparedReplacementQueueSubmission,
        replacement_recording::ReplacementNativeRecording,
        replacement_submit::QueueTimelineSemaphores,
    };
    use ash::vk;
    use reims_vgpu_core::{
        assemble_prepared_exec_resources, DirectReplayNativeOwner, ExecTransaction,
        PreparedExecResourceInputs, ResolvedInfoOperation, ResolvedOperation,
        ResolvedReplayCompletion, TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        QueueOwnerId, SessionGenerationId, SubmissionId, SubmissionIdentity, TaskId, TransactionId,
        VulkanDeviceEpochId,
    };

    #[test]
    fn disconnected_driver_returns_the_whole_exec_envelope() {
        let epoch = VulkanDeviceEpochId::new(1);
        let transaction = TransactionId::new(5);
        let queue = QueueOwnerId::new(2);
        let exec = ExecTransaction::<ResolvedOperation<(), (), ResolvedInfoOperation, (), ()>> {
            identity: SubmissionIdentity {
                id: SubmissionId::new(8),
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
                queue,
                SessionGenerationId::new(1),
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: Box::new([]),
                },
            )
            .unwrap();
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let recording = ReplacementNativeRecording::synthetic(
            reims_vgpu_core::RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let pending =
            PendingReplacementExecSubmit::disconnected_for_test(RecordedReplacementExec {
                submission,
                resources,
                image_states: None,
            });
        let ReplacementExecDriverPoll::DriverRefused { reason, recorded } = pending.poll_driver()
        else {
            panic!("a disconnected driver must return the complete EXEC")
        };
        assert_eq!(
            reason,
            crate::replacement_queue::ReplacementQueueError::OwnerStopped
        );
        assert_eq!(recorded.resources.transaction(), transaction);
        assert_eq!(recorded.submission.prepared.plan().transaction, transaction);
    }
}
