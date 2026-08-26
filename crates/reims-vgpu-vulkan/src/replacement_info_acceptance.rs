//! Atomic driver-receipt acceptance for one prepared Info reply.

use crate::{
    replacement_epoch::ReplacementQueueEpoch,
    replacement_queue::PreparedReplacementQueueSubmission,
    replacement_replay::{
        commit_driver_accepted_with_watch, AcceptedReplacementReplay, ReplacementRecordingOwner,
        ReplacementReplayAcceptanceError,
    },
};
use reims_vgpu_core::{
    DirectReplayNativeOwner, PreparedInfoQuery, ResolvedInfoOperation, ResolvedReplayCompletion,
    ResolvedResourceCompletion, ResourceLifecycleOwner, TransactionRuntime,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementInfoAcceptanceError {
    TransactionMismatch,
    CompletionSetMismatch,
    BackingMismatch,
    Replay(ReplacementReplayAcceptanceError),
}

#[derive(Debug)]
pub struct ReplacementInfoAcceptanceFailure<Semantic> {
    pub reason: ReplacementInfoAcceptanceError,
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub info: PreparedInfoQuery,
}

#[derive(Debug)]
pub struct AcceptedReplacementInfo<T> {
    pub replay: AcceptedReplacementReplay<T>,
    pub operation: ResolvedInfoOperation,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

pub struct ReplacementInfoAcceptanceOwners<'a, Semantic, T> {
    pub runtime: &'a mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    pub native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    pub resources: &'a mut ResourceLifecycleOwner<T>,
    pub recordings: &'a mut ReplacementRecordingOwner,
    pub queues: &'a mut ReplacementQueueEpoch,
}

pub fn commit_driver_accepted_info<Semantic: Clone, T>(
    owners: ReplacementInfoAcceptanceOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    info: PreparedInfoQuery,
) -> Result<AcceptedReplacementInfo<T>, Box<ReplacementInfoAcceptanceFailure<Semantic>>> {
    let queue = submission.prepared.point().queue;
    let Some(lane) = owners.queues.lane(queue) else {
        return Err(Box::new(ReplacementInfoAcceptanceFailure {
            reason: ReplacementInfoAcceptanceError::Replay(
                ReplacementReplayAcceptanceError::QueueAbsent(queue),
            ),
            submission,
            info,
        }));
    };
    commit_driver_accepted_info_with_watch(
        owners.runtime,
        owners.native,
        owners.resources,
        owners.recordings,
        submission,
        info,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_info_with_watch<Semantic: Clone, T>(
    runtime: &mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    native: &mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    info: PreparedInfoQuery,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> Result<AcceptedReplacementInfo<T>, Box<ReplacementInfoAcceptanceFailure<Semantic>>> {
    let transaction = submission.prepared.plan().transaction;
    let completions = info.resource_completions();
    let reason = if transaction != info.transaction() {
        Some(ReplacementInfoAcceptanceError::TransactionMismatch)
    } else if submission.prepared.semantic().resources != completions {
        Some(ReplacementInfoAcceptanceError::CompletionSetMismatch)
    } else if submission.recording().backings() != [info.destination().backing] {
        Some(ReplacementInfoAcceptanceError::BackingMismatch)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementInfoAcceptanceFailure {
            reason,
            submission,
            info,
        }));
    }
    let replay = match commit_driver_accepted_with_watch(
        runtime, native, resources, recordings, submission, watch,
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(ReplacementInfoAcceptanceFailure {
                reason: ReplacementInfoAcceptanceError::Replay(failure.reason.clone()),
                submission: failure.submission,
                info,
            }));
        }
    };
    Ok(AcceptedReplacementInfo {
        replay,
        operation: *info.operation(),
        resources: completions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        replacement_info_query::tests::evaluated_render_query,
        replacement_recording::ReplacementNativeRecording,
        replacement_submit::QueueTimelineSemaphores,
    };
    use ash::vk;
    use reims_vgpu_core::{
        prepare_info_query, BackingRegion, CompletionStamp, DeviceTransactionPayload,
        ExecTransaction, LinearRange, RepresentationRoute, ResolvedInfoReplyTarget,
        ResolvedResourceLifecycle, ResourceLifecycleEffect, SessionGeneration, StorageBacking,
        TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        ChannelId, QueueOwnerId, RenderPipelineObject, ResourceId, ResourceObject,
        SessionGenerationId, SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId,
        TransactionId, VulkanDeviceEpochId,
    };

    #[test]
    fn info_acceptance_rejects_an_omitted_destination_before_watch_registration() {
        let epoch = VulkanDeviceEpochId::new(1);
        let generation = SessionGenerationId::new(1);
        let queue = QueueOwnerId::new(2);
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
        let operation = ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::<RenderPipelineObject>::new(2, 1),
            reply: ResolvedInfoReplyTarget {
                resource: ResourceId::<ResourceObject>::new(3, 1),
                backing,
                range: LinearRange::new(0, 12).unwrap(),
                requested_alignment: 4,
            },
        };
        let evaluated = evaluated_render_query(
            operation,
            reims_vgpu_core::RenderPipelineStateInfo::default(),
        );
        let transaction_id = evaluated.transaction();
        let info = prepare_info_query(&mut resources, SubmissionId::new(1), evaluated).unwrap();

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
        assert_eq!(transaction.id, transaction_id);
        let completions = info.resource_completions();
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
        let failure = commit_driver_accepted_info_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            info,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementInfoAcceptanceError::BackingMismatch
        );
        let (prepared, mut recording) = failure.submission.into_parts();
        recording.backings = Box::new([backing]);
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let accepted = commit_driver_accepted_info_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            failure.info,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.resources, completions);
    }
}
