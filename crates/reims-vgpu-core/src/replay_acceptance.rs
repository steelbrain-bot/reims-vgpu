//! Atomic commit of one driver-accepted replacement submission.
//!
//! Native queue acceptance changes three owners together: logical FIFO
//! submission order, producer timeline/completion ownership, and every native
//! backing representation retained by the transaction. The complete set is
//! validated before any owner changes, so a stale or incomplete backing set
//! cannot strand the other two owners in different lifecycle states.

use crate::{
    AcceptedNativeSubmission, DirectReplayError, DirectReplayNativeOwner, ManagedBackingProgress,
    PreparedNativeSubmission, ResourceLifecycleOwner, ResourceUseBatchError, TransactionRuntime,
    TransactionRuntimeError,
};
use reims_vgpu_protocol::{BackingId, TransactionId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayAcceptanceError {
    TransactionMismatch,
    SessionGenerationMismatch,
    Native(DirectReplayError),
    Runtime(TransactionRuntimeError),
    Resources(ResourceUseBatchError),
}

#[derive(Debug)]
pub struct ReplayAcceptanceFailure<Semantic> {
    pub reason: ReplayAcceptanceError,
    pub prepared: PreparedNativeSubmission<Semantic>,
    pub backings: Box<[BackingId]>,
}

#[derive(Debug)]
pub struct ReplayAcceptance<T> {
    pub native: AcceptedNativeSubmission,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

pub fn validate_replay_acceptance<RuntimeCompletion: Clone, NativeSemantic: Clone, T>(
    runtime: &TransactionRuntime<RuntimeCompletion>,
    native: &DirectReplayNativeOwner<NativeSemantic>,
    resources: &ResourceLifecycleOwner<T>,
    prepared: &PreparedNativeSubmission<NativeSemantic>,
    transaction: TransactionId,
    backings: &[BackingId],
) -> Result<(), ReplayAcceptanceError> {
    if prepared.plan().transaction != transaction {
        return Err(ReplayAcceptanceError::TransactionMismatch);
    }
    if prepared.session_generation() != runtime.session_generation() {
        return Err(ReplayAcceptanceError::SessionGenerationMismatch);
    }
    native
        .validate_acceptance(prepared)
        .map_err(ReplayAcceptanceError::Native)?;
    runtime
        .validate_submitted(transaction)
        .map_err(ReplayAcceptanceError::Runtime)?;
    resources
        .validate_submit_uses(backings, transaction, prepared.point())
        .map_err(ReplayAcceptanceError::Resources)?;
    Ok(())
}

/// Commit the exact token returned by a successful native driver receipt.
pub fn commit_replay_acceptance<RuntimeCompletion: Clone, NativeSemantic: Clone, T>(
    runtime: &mut TransactionRuntime<RuntimeCompletion>,
    native: &mut DirectReplayNativeOwner<NativeSemantic>,
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedNativeSubmission<NativeSemantic>,
    transaction: TransactionId,
    backings: impl Into<Box<[BackingId]>>,
) -> Result<ReplayAcceptance<T>, ReplayAcceptanceFailure<NativeSemantic>> {
    let backings = backings.into();
    if let Err(reason) = validate_replay_acceptance(
        runtime,
        native,
        resources,
        &prepared,
        transaction,
        &backings,
    ) {
        return Err(ReplayAcceptanceFailure {
            reason,
            prepared,
            backings,
        });
    }
    let point = prepared.point();
    let resource_progress = resources
        .submit_uses(&backings, transaction, point)
        .unwrap_or_else(|_| unreachable!("the complete replay acceptance was prevalidated"));
    runtime
        .submitted(transaction)
        .unwrap_or_else(|_| unreachable!("the complete replay acceptance was prevalidated"));
    let accepted = native
        .accepted(prepared)
        .unwrap_or_else(|_| unreachable!("the complete replay acceptance was prevalidated"));
    Ok(ReplayAcceptance {
        native: accepted,
        resources: resource_progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BackingRegion, CompletionStamp, DeviceTransactionPayload, ExecTransaction,
        RepresentationRoute, ResolvedExecSegment, ResolvedExecStream, ResolvedResourceLifecycle,
        ResourceLifecycleEffect, SessionGeneration, StorageBacking,
    };
    use reims_vgpu_protocol::{
        ChannelId, QueueOwnerId, SegmentBoundary, SegmentKind, SessionGenerationId, SubmissionId,
        SubmissionIdentity, TaskId, VulkanDeviceEpochId,
    };

    #[test]
    fn invalid_backing_use_changes_no_acceptance_owner_and_can_be_retried() {
        let epoch = VulkanDeviceEpochId::new(2);
        let mut runtime: TransactionRuntime<&'static str> =
            TransactionRuntime::new(SessionGeneration::new(SessionGenerationId::new(1)));
        let channel = ChannelId::new(3);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[crate::ResolvedTransactionPrerequisite]>::default(),
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
        assert_eq!(
            runtime.take_submission_ready()[0].transaction,
            transaction.id
        );
        let plan = native
            .queue_candidate(
                transaction.id,
                Box::<[(TransactionId, crate::WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(
                plan,
                QueueOwnerId::new(1),
                SessionGenerationId::new(1),
                "done",
            )
            .unwrap();

        let mut resources = ResourceLifecycleOwner::new(epoch);
        let backing = match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let representation = resources
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        let failure = commit_replay_acceptance(
            &mut runtime,
            &mut native,
            &mut resources,
            prepared,
            transaction.id,
            [backing],
        )
        .unwrap_err();
        assert!(matches!(
            failure.reason,
            ReplayAcceptanceError::Resources(ResourceUseBatchError::Backing {
                backing: found,
                reason: crate::ManagedBackingError::UnknownAcceptedUse,
            }) if found == backing
        ));
        assert!(runtime.validate_submitted(transaction.id).is_ok());
        assert!(native.validate_acceptance(&failure.prepared).is_ok());

        resources
            .accept_use(backing, transaction.id, [representation])
            .unwrap();
        let accepted = commit_replay_acceptance(
            &mut runtime,
            &mut native,
            &mut resources,
            failure.prepared,
            transaction.id,
            failure.backings,
        )
        .unwrap();
        assert_eq!(accepted.native.transaction, transaction.id);
        assert_eq!(accepted.resources.len(), 1);
    }
}
