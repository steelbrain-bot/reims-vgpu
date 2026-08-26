//! Atomic retirement of one guest-published replacement transaction.

use crate::{
    DirectReplayError, DirectReplayNativeOwner, TransactionRuntime, TransactionRuntimeError,
};
use reims_vgpu_protocol::TransactionId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayRetirementError {
    Runtime(TransactionRuntimeError),
    Native(DirectReplayError),
}

/// Retire semantic publication/dependency state and its fixed-worker/native
/// dependency assignment together. A pending native completion or accepted
/// semantic dependent refuses before either owner changes.
pub fn commit_replay_retirement<RuntimeCompletion: Clone, NativeSemantic: Clone>(
    runtime: &mut TransactionRuntime<RuntimeCompletion>,
    native: &mut DirectReplayNativeOwner<NativeSemantic>,
    transaction: TransactionId,
) -> Result<(), ReplayRetirementError> {
    runtime
        .validate_retire_transaction(transaction)
        .map_err(ReplayRetirementError::Runtime)?;
    native
        .validate_retire(transaction)
        .map_err(ReplayRetirementError::Native)?;
    runtime
        .retire_transaction(transaction)
        .unwrap_or_else(|_| unreachable!("semantic retirement was prevalidated"));
    native
        .retire(transaction)
        .unwrap_or_else(|_| unreachable!("native retirement was prevalidated"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompletionStamp, DeviceTransactionPayload, ExecTransaction, ResolvedExecSegment,
        ResolvedExecStream, ResolvedTransactionPrerequisite, SessionGeneration,
        WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        ChannelId, QueueOwnerId, QueueTimelineValue, SegmentBoundary, SegmentKind,
        SessionGenerationId, SubmissionId, SubmissionIdentity, TaskId, VulkanDeviceEpochId,
    };

    #[test]
    fn pending_completion_refuses_without_retiring_either_owner() {
        let generation = SessionGenerationId::new(1);
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(3);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<(), (), (), (), ()>::Exec(ExecTransaction {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(1),
                        task: TaskId::new(1),
                    },
                    prologue: crate::ExecPrologue::default(),
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
                            operations: Box::new([]),
                        }]),
                    }]),
                    accesses: Box::new([]),
                }),
            )
            .unwrap();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(runtime.recording_plan(transaction.id).unwrap())
            .unwrap();
        runtime.recorded(transaction.id).unwrap();
        runtime.take_submission_ready();
        let plan = native
            .queue_candidate(
                transaction.id,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native.prepare(plan, queue, generation, "done").unwrap();
        runtime.submitted(transaction.id).unwrap();
        native.accepted(prepared).unwrap();

        assert_eq!(
            commit_replay_retirement(&mut runtime, &mut native, transaction.id),
            Err(ReplayRetirementError::Runtime(
                TransactionRuntimeError::TransactionNotPublished
            ))
        );
        assert!(runtime
            .validate_gpu_complete(transaction.id, "done")
            .is_ok());
        assert_eq!(native.pending_completions(), 1);

        runtime.semantic_complete(transaction.id, "done").unwrap();
        assert_eq!(
            commit_replay_retirement(&mut runtime, &mut native, transaction.id),
            Err(ReplayRetirementError::Native(
                DirectReplayError::Completion(crate::CompletionOwnerError::TransactionStillPending)
            ))
        );
        assert!(runtime.validate_retire_transaction(transaction.id).is_ok());

        assert_eq!(
            native
                .advance(queue, QueueTimelineValue::new(1))
                .unwrap()
                .len(),
            1
        );
        commit_replay_retirement(&mut runtime, &mut native, transaction.id).unwrap();
        assert_eq!(
            runtime.validate_retire_transaction(transaction.id),
            Err(TransactionRuntimeError::UnknownTransaction)
        );
        assert!(matches!(
            native.validate_retire(transaction.id),
            Err(DirectReplayError::Assignment(_))
        ));
    }
}
