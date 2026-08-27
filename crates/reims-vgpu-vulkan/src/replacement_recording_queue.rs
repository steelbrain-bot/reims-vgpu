//! Ownership bridge from asynchronous recording to prepared queue submission.
//!
//! The semantic preparation token remains joined to its exact recording job
//! while a fixed worker runs. Only a successful recording can become a queue
//! submission; every refusal returns both the semantic token and whichever
//! native ownership form exists at that point.

use crate::{
    replacement_queue::{
        PreparedReplacementAuxiliaryQueueSubmission, PreparedReplacementQueueSubmission,
        ReplacementAuxiliaryQueuePreparationFailure, ReplacementQueuePreparationFailure,
    },
    replacement_recording::{
        dispatch_replacement_recording, PendingReplacementRecording, ReplacementNativeRecording,
        ReplacementRecordingDispatchError, ReplacementRecordingDispatchFailure,
        ReplacementRecordingOperation, ReplacementRecordingPoll, ReplacementRecordingRequest,
        ReplacementRecordingWorker,
    },
    replacement_submit::{QueueTimelineSemaphores, TimelineSubmitPlanError},
};
use reims_vgpu_core::{FixedExecutor, PreparedAuxiliaryNativeSubmission, PreparedNativeSubmission};
use reims_vgpu_protocol::TransactionId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedReplacementRecordingError {
    QueueAbsent(reims_vgpu_protocol::QueueOwnerId),
    QueueCapabilitiesMissing {
        queue: reims_vgpu_protocol::QueueOwnerId,
        required: ash::vk::QueueFlags,
        available: ash::vk::QueueFlags,
    },
    TransactionMismatch {
        prepared: TransactionId,
        request: TransactionId,
    },
    WorkerMismatch {
        prepared: reims_vgpu_core::RecordingWorkerId,
        request: reims_vgpu_core::RecordingWorkerId,
    },
    QueueFamilyMismatch {
        prepared: u32,
        request: u32,
    },
    Recording(ReplacementRecordingDispatchError),
    Queue(TimelineSubmitPlanError),
}

#[derive(Debug)]
pub enum PreparedReplacementRecordingRecovery<Operation> {
    Request(ReplacementRecordingRequest<Operation>),
    Recording(ReplacementNativeRecording),
}

#[derive(Debug)]
pub struct PreparedReplacementRecordingFailure<Semantic, Operation> {
    pub reason: PreparedReplacementRecordingError,
    pub prepared: PreparedNativeSubmission<Semantic>,
    pub recovery: PreparedReplacementRecordingRecovery<Operation>,
}

#[derive(Debug)]
pub struct PreparedReplacementAuxiliaryRecordingFailure<Operation> {
    pub reason: PreparedReplacementRecordingError,
    pub prepared: PreparedAuxiliaryNativeSubmission,
    pub recovery: PreparedReplacementRecordingRecovery<Operation>,
}

#[must_use = "prepared auxiliary and native recording ownership must be observed together"]
#[derive(Debug)]
pub struct PendingPreparedReplacementAuxiliaryRecording<Operation> {
    prepared: PreparedAuxiliaryNativeSubmission,
    pending: PendingReplacementRecording<Operation>,
    timelines: QueueTimelineSemaphores,
    auxiliary_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
}

pub enum PreparedReplacementAuxiliaryRecordingPoll<Operation> {
    Pending(PendingPreparedReplacementAuxiliaryRecording<Operation>),
    Completed(
        Result<
            PreparedReplacementAuxiliaryQueueSubmission,
            Box<PreparedReplacementAuxiliaryRecordingFailure<Operation>>,
        >,
    ),
}

impl<Operation> PendingPreparedReplacementAuxiliaryRecording<Operation> {
    pub fn try_complete(self) -> PreparedReplacementAuxiliaryRecordingPoll<Operation> {
        let Self {
            prepared,
            pending,
            timelines,
            auxiliary_waits,
        } = self;
        match pending.try_complete() {
            ReplacementRecordingPoll::Pending(pending) => {
                PreparedReplacementAuxiliaryRecordingPoll::Pending(Self {
                    prepared,
                    pending,
                    timelines,
                    auxiliary_waits,
                })
            }
            ReplacementRecordingPoll::Completed(result) => {
                PreparedReplacementAuxiliaryRecordingPoll::Completed(finish_auxiliary_recording(
                    prepared,
                    timelines,
                    auxiliary_waits,
                    result,
                ))
            }
        }
    }

    pub fn wait(
        self,
    ) -> Result<
        PreparedReplacementAuxiliaryQueueSubmission,
        Box<PreparedReplacementAuxiliaryRecordingFailure<Operation>>,
    > {
        let Self {
            prepared,
            pending,
            timelines,
            auxiliary_waits,
        } = self;
        finish_auxiliary_recording(prepared, timelines, auxiliary_waits, pending.wait())
    }
}

pub fn dispatch_prepared_replacement_auxiliary_recording<
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
>(
    executor: &FixedExecutor<ReplacementRecordingWorker>,
    timelines: &QueueTimelineSemaphores,
    prepared: PreparedAuxiliaryNativeSubmission,
    request: ReplacementRecordingRequest<Operation>,
    auxiliary_waits: impl Into<Box<[reims_vgpu_core::QueueTimelinePoint]>>,
) -> Result<
    PendingPreparedReplacementAuxiliaryRecording<Operation>,
    Box<PreparedReplacementAuxiliaryRecordingFailure<Operation>>,
> {
    let prepared_transaction = prepared.transaction();
    if prepared_transaction != request.transaction {
        return Err(Box::new(PreparedReplacementAuxiliaryRecordingFailure {
            reason: PreparedReplacementRecordingError::TransactionMismatch {
                prepared: prepared_transaction,
                request: request.transaction,
            },
            prepared,
            recovery: PreparedReplacementRecordingRecovery::Request(request),
        }));
    }
    if prepared.recording_worker() != request.worker {
        return Err(Box::new(PreparedReplacementAuxiliaryRecordingFailure {
            reason: PreparedReplacementRecordingError::WorkerMismatch {
                prepared: prepared.recording_worker(),
                request: request.worker,
            },
            prepared,
            recovery: PreparedReplacementRecordingRecovery::Request(request),
        }));
    }
    let pending = match dispatch_replacement_recording(executor, request) {
        Ok(pending) => pending,
        Err(failure) => {
            return Err(Box::new(PreparedReplacementAuxiliaryRecordingFailure {
                reason: PreparedReplacementRecordingError::Recording(failure.reason),
                prepared,
                recovery: PreparedReplacementRecordingRecovery::Request(failure.request),
            }));
        }
    };
    Ok(PendingPreparedReplacementAuxiliaryRecording {
        prepared,
        pending,
        timelines: timelines.clone(),
        auxiliary_waits: auxiliary_waits.into(),
    })
}

fn finish_auxiliary_recording<Operation>(
    prepared: PreparedAuxiliaryNativeSubmission,
    timelines: QueueTimelineSemaphores,
    auxiliary_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
    result: Result<ReplacementNativeRecording, Box<ReplacementRecordingDispatchFailure<Operation>>>,
) -> Result<
    PreparedReplacementAuxiliaryQueueSubmission,
    Box<PreparedReplacementAuxiliaryRecordingFailure<Operation>>,
> {
    let recording = match result {
        Ok(recording) => recording,
        Err(failure) => {
            return Err(Box::new(PreparedReplacementAuxiliaryRecordingFailure {
                reason: PreparedReplacementRecordingError::Recording(failure.reason),
                prepared,
                recovery: PreparedReplacementRecordingRecovery::Request(failure.request),
            }));
        }
    };
    PreparedReplacementAuxiliaryQueueSubmission::new_with_auxiliary_waits(
        prepared,
        &timelines,
        recording,
        auxiliary_waits,
    )
    .map_err(
        |failure: Box<ReplacementAuxiliaryQueuePreparationFailure>| {
            Box::new(PreparedReplacementAuxiliaryRecordingFailure {
                reason: PreparedReplacementRecordingError::Queue(failure.reason),
                prepared: failure.prepared,
                recovery: PreparedReplacementRecordingRecovery::Recording(failure.recording),
            })
        },
    )
}

#[must_use = "prepared semantic and native recording ownership must be observed together"]
#[derive(Debug)]
pub struct PendingPreparedReplacementRecording<Semantic, Operation> {
    prepared: PreparedNativeSubmission<Semantic>,
    pending: PendingReplacementRecording<Operation>,
    timelines: QueueTimelineSemaphores,
    auxiliary_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
}

pub enum PreparedReplacementRecordingPoll<Semantic, Operation> {
    Pending(PendingPreparedReplacementRecording<Semantic, Operation>),
    Completed(
        Result<
            PreparedReplacementQueueSubmission<Semantic>,
            Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
        >,
    ),
}

impl<Semantic, Operation> PendingPreparedReplacementRecording<Semantic, Operation> {
    pub fn try_complete(self) -> PreparedReplacementRecordingPoll<Semantic, Operation> {
        let Self {
            prepared,
            pending,
            timelines,
            auxiliary_waits,
        } = self;
        match pending.try_complete() {
            ReplacementRecordingPoll::Pending(pending) => {
                PreparedReplacementRecordingPoll::Pending(Self {
                    prepared,
                    pending,
                    timelines,
                    auxiliary_waits,
                })
            }
            ReplacementRecordingPoll::Completed(result) => {
                PreparedReplacementRecordingPoll::Completed(finish_recording(
                    prepared,
                    timelines,
                    auxiliary_waits,
                    result,
                ))
            }
        }
    }

    pub fn wait(
        self,
    ) -> Result<
        PreparedReplacementQueueSubmission<Semantic>,
        Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
    > {
        let Self {
            prepared,
            pending,
            timelines,
            auxiliary_waits,
        } = self;
        finish_recording(prepared, timelines, auxiliary_waits, pending.wait())
    }
}

pub fn dispatch_prepared_replacement_recording<
    Semantic,
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
>(
    executor: &FixedExecutor<ReplacementRecordingWorker>,
    timelines: &QueueTimelineSemaphores,
    prepared: PreparedNativeSubmission<Semantic>,
    request: ReplacementRecordingRequest<Operation>,
) -> Result<
    PendingPreparedReplacementRecording<Semantic, Operation>,
    Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
> {
    dispatch_prepared_replacement_recording_with_auxiliary_waits(
        executor,
        timelines,
        prepared,
        request,
        Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
    )
}

pub fn dispatch_prepared_replacement_recording_with_auxiliary_waits<
    Semantic,
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
>(
    executor: &FixedExecutor<ReplacementRecordingWorker>,
    timelines: &QueueTimelineSemaphores,
    prepared: PreparedNativeSubmission<Semantic>,
    request: ReplacementRecordingRequest<Operation>,
    auxiliary_waits: impl Into<Box<[reims_vgpu_core::QueueTimelinePoint]>>,
) -> Result<
    PendingPreparedReplacementRecording<Semantic, Operation>,
    Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
> {
    let auxiliary_waits = auxiliary_waits.into();
    let prepared_transaction = prepared.plan().transaction;
    if prepared_transaction != request.transaction {
        return Err(Box::new(PreparedReplacementRecordingFailure {
            reason: PreparedReplacementRecordingError::TransactionMismatch {
                prepared: prepared_transaction,
                request: request.transaction,
            },
            prepared,
            recovery: PreparedReplacementRecordingRecovery::Request(request),
        }));
    }
    if prepared.recording_worker() != request.worker {
        return Err(Box::new(PreparedReplacementRecordingFailure {
            reason: PreparedReplacementRecordingError::WorkerMismatch {
                prepared: prepared.recording_worker(),
                request: request.worker,
            },
            prepared,
            recovery: PreparedReplacementRecordingRecovery::Request(request),
        }));
    }
    let pending = match dispatch_replacement_recording(executor, request) {
        Ok(pending) => pending,
        Err(failure) => {
            return Err(Box::new(PreparedReplacementRecordingFailure {
                reason: PreparedReplacementRecordingError::Recording(failure.reason),
                prepared,
                recovery: PreparedReplacementRecordingRecovery::Request(failure.request),
            }));
        }
    };
    Ok(PendingPreparedReplacementRecording {
        prepared,
        pending,
        timelines: timelines.clone(),
        auxiliary_waits,
    })
}

fn finish_recording<Semantic, Operation>(
    prepared: PreparedNativeSubmission<Semantic>,
    timelines: QueueTimelineSemaphores,
    auxiliary_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
    result: Result<ReplacementNativeRecording, Box<ReplacementRecordingDispatchFailure<Operation>>>,
) -> Result<
    PreparedReplacementQueueSubmission<Semantic>,
    Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
> {
    let recording = match result {
        Ok(recording) => recording,
        Err(failure) => {
            let failure = *failure;
            return Err(Box::new(PreparedReplacementRecordingFailure {
                reason: PreparedReplacementRecordingError::Recording(failure.reason),
                prepared,
                recovery: PreparedReplacementRecordingRecovery::Request(failure.request),
            }));
        }
    };
    PreparedReplacementQueueSubmission::new_with_auxiliary_waits(
        prepared,
        &timelines,
        recording,
        auxiliary_waits,
    )
    .map_err(
        |failure: Box<ReplacementQueuePreparationFailure<Semantic>>| {
            Box::new(PreparedReplacementRecordingFailure {
                reason: PreparedReplacementRecordingError::Queue(failure.reason),
                prepared: failure.prepared,
                recovery: PreparedReplacementRecordingRecovery::Recording(failure.recording),
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::context::DeviceContext,
        replacement_barrier_record::NativeBarrierBatch,
        replacement_recording::{ReplacementRecordingInput, ReplacementRecordingWorker},
    };
    use ash::vk;
    use reims_vgpu_core::{
        DescriptorTier, DirectReplayNativeOwner, OperationKind, ParticipationOperation,
        ParticipationScope, RecordingWorkerId, ResolvedExecSegment, ResolvedExecStream,
        ResolvedOperation, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        HeapObject, QueueOwnerId, ResourceId, SegmentBoundary, SegmentKind, SessionGenerationId,
        SubmissionId, SubmissionIdentity, TaskId, TransactionId, VulkanDeviceEpochId,
    };
    use std::sync::mpsc;

    fn prepared(
        epoch: VulkanDeviceEpochId,
        queue: QueueOwnerId,
        transaction: TransactionId,
        semantic: &'static str,
    ) -> PreparedNativeSubmission<&'static str> {
        let mut owner = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        owner
            .assign_recording(reims_vgpu_core::TransactionRecordingPlan {
                transaction,
                domain: reims_vgpu_protocol::SubmissionDomainId::new(1),
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
            .prepare(plan, queue, SessionGenerationId::new(1), semantic)
            .unwrap()
    }

    fn auxiliary_prepared(
        epoch: VulkanDeviceEpochId,
        queue: QueueOwnerId,
        transaction: TransactionId,
    ) -> PreparedAuxiliaryNativeSubmission {
        let mut owner = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        owner
            .assign_recording(reims_vgpu_core::TransactionRecordingPlan {
                transaction,
                domain: reims_vgpu_protocol::SubmissionDomainId::new(1),
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
        let chain = owner
            .prepare_execution_chain(plan, SessionGenerationId::new(1), ())
            .unwrap();
        owner
            .prepare_execution_chain_auxiliary(&chain, queue)
            .unwrap()
    }

    fn empty_exec(
        id: u64,
    ) -> reims_vgpu_core::ExecTransaction<reims_vgpu_core::ResolvedOperation<(), (), (), (), ()>>
    {
        reims_vgpu_core::ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        }
    }

    struct EmptyResolver;

    impl crate::replacement_barrier_record::ReplacementBarrierResolver for EmptyResolver {
        fn resolve(
            &self,
            _backing: reims_vgpu_protocol::BackingId,
        ) -> Option<crate::replacement_barrier_record::NativeBarrierResolution> {
            None
        }
    }

    impl crate::replacement_barrier_record::ReplacementBarrierResourceResolver for EmptyResolver {
        fn alias_backings(
            &self,
            _resource: ResourceId<reims_vgpu_protocol::ResourceObject>,
        ) -> Option<Box<[reims_vgpu_protocol::BackingId]>> {
            None
        }
    }

    fn resolved_request(
        transaction: u64,
        worker: RecordingWorkerId,
        queue_family: u32,
        exec: reims_vgpu_core::ExecTransaction<
            reims_vgpu_core::ResolvedOperation<(), (), (), (), ()>,
        >,
    ) -> ReplacementRecordingRequest<reims_vgpu_core::ResolvedOperation<(), (), (), (), ()>> {
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

    fn participation_exec(
        id: u64,
    ) -> reims_vgpu_core::ExecTransaction<reims_vgpu_core::ResolvedOperation<(), (), (), (), ()>>
    {
        reims_vgpu_core::ExecTransaction {
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
                        kind: SegmentKind::Compute,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::Participation(
                        ParticipationOperation::Heap {
                            heap: ResourceId::<HeapObject>::new(4, 1),
                            scope: ParticipationScope::Compute,
                        },
                    )]),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    fn recycle(
        workers: &FixedExecutor<ReplacementRecordingWorker>,
        recording: ReplacementNativeRecording,
    ) {
        let (sender, receiver) = mpsc::sync_channel(1);
        workers
            .submit_to(recording.worker, move |worker| {
                let _ = sender.send(worker.recycle(recording));
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
    }

    #[test]
    fn prepared_semantics_remain_joined_through_async_recording_and_queue_planning() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let mut context = match unsafe { DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP prepared replacement recording: no device ({error})");
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
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(2);
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);

        let (release, held) = mpsc::sync_channel(1);
        workers
            .submit_to(RecordingWorkerId::new(0), move |_| {
                held.recv().unwrap();
            })
            .unwrap();
        let pending = dispatch_prepared_replacement_recording(
            &workers,
            &timelines,
            prepared(epoch, queue, TransactionId::new(7), "semantic"),
            resolved_request(
                7,
                RecordingWorkerId::new(0),
                context.gq,
                participation_exec(7),
            ),
        )
        .unwrap();
        let PreparedReplacementRecordingPoll::Pending(pending) = pending.try_complete() else {
            panic!("the occupied recording worker cannot complete the prepared job");
        };
        release.send(()).unwrap();
        let submission = pending.wait().unwrap();
        let (prepared_token, recording) = submission.into_parts();
        assert_eq!(prepared_token.plan().transaction, TransactionId::new(7));
        assert_eq!(prepared_token.semantic(), &"semantic");
        assert_eq!(recording.worker, RecordingWorkerId::new(0));
        assert_eq!(
            recording.recorded_operations.as_ref(),
            [OperationKind::Participation]
        );
        recycle(&workers, recording);

        let auxiliary = dispatch_prepared_replacement_auxiliary_recording(
            &workers,
            &timelines,
            auxiliary_prepared(epoch, queue, TransactionId::new(10)),
            resolved_request(10, RecordingWorkerId::new(0), context.gq, empty_exec(10)),
            Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
        )
        .unwrap()
        .wait()
        .unwrap();
        let (prepared_auxiliary, recording) = auxiliary.into_parts();
        assert_eq!(prepared_auxiliary.transaction(), TransactionId::new(10));
        assert_eq!(recording.worker, RecordingWorkerId::new(0));
        recycle(&workers, recording);

        let failure = dispatch_prepared_replacement_recording(
            &workers,
            &timelines,
            prepared(epoch, queue, TransactionId::new(8), "mismatch"),
            resolved_request(9, RecordingWorkerId::new(0), context.gq, empty_exec(9)),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            PreparedReplacementRecordingError::TransactionMismatch {
                prepared: TransactionId::new(8),
                request: TransactionId::new(9),
            }
        );
        assert_eq!(failure.prepared.semantic(), &"mismatch");
        assert!(matches!(
            failure.recovery,
            PreparedReplacementRecordingRecovery::Request(request)
                if request.transaction == TransactionId::new(9)
        ));

        let failure = dispatch_prepared_replacement_recording(
            &workers,
            &timelines,
            prepared(epoch, queue, TransactionId::new(9), "worker-mismatch"),
            resolved_request(9, RecordingWorkerId::new(1), context.gq, empty_exec(9)),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            PreparedReplacementRecordingError::WorkerMismatch {
                prepared: RecordingWorkerId::new(0),
                request: RecordingWorkerId::new(1),
            }
        );
        assert_eq!(failure.prepared.semantic(), &"worker-mismatch");

        let wrong_epoch = VulkanDeviceEpochId::new(3);
        let failure = dispatch_prepared_replacement_recording(
            &workers,
            &timelines,
            prepared(wrong_epoch, queue, TransactionId::new(10), "queue-refusal"),
            resolved_request(10, RecordingWorkerId::new(0), context.gq, empty_exec(10)),
        )
        .unwrap()
        .wait()
        .unwrap_err();
        assert_eq!(
            failure.reason,
            PreparedReplacementRecordingError::Queue(TimelineSubmitPlanError::MixedEpochs)
        );
        assert_eq!(failure.prepared.semantic(), &"queue-refusal");
        let PreparedReplacementRecordingRecovery::Recording(recording) = failure.recovery else {
            panic!("queue planning refusal must return the ended native recording");
        };
        recycle(&workers, recording);
        drop(workers);
        unsafe { context.destroy() };
    }
}
