//! Atomic driver-receipt acceptance for one prepared buffer blit.

use crate::{
    replacement_epoch::ReplacementQueueEpoch,
    replacement_queue::PreparedReplacementQueueSubmission,
    replacement_replay::{
        commit_driver_accepted_with_watch, AcceptedReplacementReplay, ReplacementRecordingOwner,
        ReplacementReplayAcceptanceError,
    },
};
use reims_vgpu_core::{
    DirectReplayNativeOwner, PreparedBufferBlit, ResolvedBlit, ResolvedReplayCompletion,
    ResolvedResourceCompletion, ResourceLifecycleOwner, TransactionRuntime,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementBufferBlitAcceptanceError {
    TransactionMismatch,
    CompletionSetMismatch,
    BackingSetMismatch,
    Replay(ReplacementReplayAcceptanceError),
}

#[derive(Debug)]
pub struct ReplacementBufferBlitAcceptanceFailure<Semantic> {
    pub reason: ReplacementBufferBlitAcceptanceError,
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub blit: PreparedBufferBlit,
}

#[derive(Debug)]
pub struct AcceptedReplacementBufferBlit<T> {
    pub replay: AcceptedReplacementReplay<T>,
    pub operation: ResolvedBlit,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

pub struct ReplacementBufferBlitAcceptanceOwners<'a, Semantic, T> {
    pub runtime: &'a mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    pub native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    pub resources: &'a mut ResourceLifecycleOwner<T>,
    pub recordings: &'a mut ReplacementRecordingOwner,
    pub queues: &'a mut ReplacementQueueEpoch,
}

/// Join a successful physical-queue receipt to the exact semantic buffer-blit
/// preparation and every replay owner. No validation refusal changes an owner.
pub fn commit_driver_accepted_buffer_blit<Semantic: Clone, T>(
    owners: ReplacementBufferBlitAcceptanceOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    blit: PreparedBufferBlit,
) -> Result<AcceptedReplacementBufferBlit<T>, Box<ReplacementBufferBlitAcceptanceFailure<Semantic>>>
{
    let queue = submission.prepared.point().queue;
    let Some(lane) = owners.queues.lane(queue) else {
        return Err(Box::new(ReplacementBufferBlitAcceptanceFailure {
            reason: ReplacementBufferBlitAcceptanceError::Replay(
                ReplacementReplayAcceptanceError::QueueAbsent(queue),
            ),
            submission,
            blit,
        }));
    };
    commit_driver_accepted_buffer_blit_with_watch(
        owners.runtime,
        owners.native,
        owners.resources,
        owners.recordings,
        submission,
        blit,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_buffer_blit_with_watch<Semantic: Clone, T>(
    runtime: &mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    native: &mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    blit: PreparedBufferBlit,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> Result<AcceptedReplacementBufferBlit<T>, Box<ReplacementBufferBlitAcceptanceFailure<Semantic>>>
{
    let transaction = submission.prepared.plan().transaction;
    let completions = blit.resource_completions();
    let reason = if transaction != blit.transaction() {
        Some(ReplacementBufferBlitAcceptanceError::TransactionMismatch)
    } else if submission.prepared.semantic().resources != completions {
        Some(ReplacementBufferBlitAcceptanceError::CompletionSetMismatch)
    } else if !same_backings(submission.recording().backings(), &blit.backings()) {
        Some(ReplacementBufferBlitAcceptanceError::BackingSetMismatch)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementBufferBlitAcceptanceFailure {
            reason,
            submission,
            blit,
        }));
    }
    let replay = match commit_driver_accepted_with_watch(
        runtime, native, resources, recordings, submission, watch,
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(ReplacementBufferBlitAcceptanceFailure {
                reason: ReplacementBufferBlitAcceptanceError::Replay(failure.reason.clone()),
                submission: failure.submission,
                blit,
            }));
        }
    };
    Ok(AcceptedReplacementBufferBlit {
        replay,
        operation: blit.operation().clone(),
        resources: completions,
    })
}

fn same_backings(
    left: &[reims_vgpu_protocol::BackingId],
    right: &[reims_vgpu_protocol::BackingId],
) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    left.dedup();
    right.sort_unstable();
    right.dedup();
    left == right
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
        prepare_buffer_blit, BackingRegion, BufferFillPattern, CompletionStamp,
        DeviceTransactionPayload, ExecTransaction, LinearRange, RepresentationRoute,
        ResolvedBufferRange, ResolvedResourceLifecycle, ResourceLifecycleEffect, SessionGeneration,
        StorageBacking, TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        ByteLength, ChannelId, GuestVirtualAddress, QueueOwnerId, ResourceId, SessionGenerationId,
        SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId, TransactionId,
        VulkanDeviceEpochId,
    };

    #[test]
    fn exact_recording_backing_set_gates_atomic_buffer_acceptance() {
        let epoch = VulkanDeviceEpochId::new(1);
        let generation = SessionGenerationId::new(1);
        let queue = QueueOwnerId::new(2);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<(), (), (), (), ()>::Exec(ExecTransaction {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(1),
                        task: TaskId::new(1),
                    },
                    prologue: reims_vgpu_core::ExecPrologue::default(),
                    streams: Box::new([]),
                    accesses: Box::new([]),
                }),
            )
            .unwrap();
        let mut resources = ResourceLifecycleOwner::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let blit = prepare_buffer_blit(
            &mut resources,
            transaction.id,
            SubmissionId::new(1),
            ResolvedBlit::Fill {
                destination: ResolvedBufferRange {
                    resource: ResourceId::new(1, 1),
                    storage: backing,
                    region: LinearRange::new(0, 16).unwrap(),
                    address: GuestVirtualAddress::new(0x1000),
                    length: ByteLength::new(16),
                },
                pattern: BufferFillPattern::Byte(0xa5),
            },
        )
        .unwrap();
        let completions = blit.resource_completions();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: transaction.id,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
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
        let prepared = native
            .prepare(
                plan,
                queue,
                generation,
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: completions.clone(),
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
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let failure = commit_driver_accepted_buffer_blit_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            blit,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementBufferBlitAcceptanceError::BackingSetMismatch
        );
        let (prepared, mut recording) = failure.submission.into_parts();
        recording.backings = failure.blit.backings();
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let accepted = commit_driver_accepted_buffer_blit_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            failure.blit,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.resources, completions);
        assert_eq!(accepted.replay.replay.native.transaction, transaction.id);
    }
}
