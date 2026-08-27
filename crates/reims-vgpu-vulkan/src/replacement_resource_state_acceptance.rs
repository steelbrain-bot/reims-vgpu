//! Atomic driver-receipt acceptance for one prepared resource-state batch.

use crate::{
    replacement_epoch::ReplacementQueueEpoch,
    replacement_queue::PreparedReplacementQueueSubmission,
    replacement_replay::{
        commit_driver_accepted_with_watch, AcceptedReplacementReplay, ReplacementRecordingOwner,
        ReplacementReplayAcceptanceError,
    },
};
use reims_vgpu_core::{
    DirectReplayNativeOwner, PreparedResourceStateBatch, ResolvedReplayCompletion,
    ResolvedResourceCompletion, ResourceLifecycleOwner, ResourceStateOutcome, TransactionRuntime,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementResourceStateAcceptanceError {
    TransactionMismatch,
    CompletionSetMismatch,
    HostLandingSetMismatch,
    BackingSetMismatch,
    Replay(ReplacementReplayAcceptanceError),
}

#[derive(Debug)]
pub struct ReplacementResourceStateAcceptanceFailure<Semantic> {
    pub reason: ReplacementResourceStateAcceptanceError,
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub states: PreparedResourceStateBatch,
}

#[derive(Debug)]
pub struct AcceptedReplacementResourceStates<T> {
    pub replay: AcceptedReplacementReplay<T>,
    pub outcomes: Box<[ResourceStateOutcome]>,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

pub struct ReplacementResourceStateAcceptanceOwners<'a, Semantic, T> {
    pub runtime: &'a mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    pub native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    pub resources: &'a mut ResourceLifecycleOwner<T>,
    pub recordings: &'a mut ReplacementRecordingOwner,
    pub queues: &'a mut ReplacementQueueEpoch,
}

pub fn commit_driver_accepted_resource_states<Semantic: Clone, T>(
    owners: ReplacementResourceStateAcceptanceOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    states: PreparedResourceStateBatch,
) -> Result<
    AcceptedReplacementResourceStates<T>,
    Box<ReplacementResourceStateAcceptanceFailure<Semantic>>,
> {
    let queue = submission.prepared.point().queue;
    let Some(lane) = owners.queues.lane(queue) else {
        return Err(Box::new(ReplacementResourceStateAcceptanceFailure {
            reason: ReplacementResourceStateAcceptanceError::Replay(
                ReplacementReplayAcceptanceError::QueueAbsent(queue),
            ),
            submission,
            states,
        }));
    };
    commit_driver_accepted_resource_states_with_watch(
        owners.runtime,
        owners.native,
        owners.resources,
        owners.recordings,
        submission,
        states,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_resource_states_with_watch<Semantic: Clone, T>(
    runtime: &mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    native: &mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    states: PreparedResourceStateBatch,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> Result<
    AcceptedReplacementResourceStates<T>,
    Box<ReplacementResourceStateAcceptanceFailure<Semantic>>,
> {
    let transaction = submission.prepared.plan().transaction;
    let completions = states.resource_completions();
    let reason = if transaction != states.transaction() {
        Some(ReplacementResourceStateAcceptanceError::TransactionMismatch)
    } else if submission.prepared.semantic().resources != completions {
        Some(ReplacementResourceStateAcceptanceError::CompletionSetMismatch)
    } else if submission.recording().host_landings().as_ref() != states.host_landings().as_ref() {
        Some(ReplacementResourceStateAcceptanceError::HostLandingSetMismatch)
    } else if !same_backings(submission.recording().backings(), &states.backings()) {
        Some(ReplacementResourceStateAcceptanceError::BackingSetMismatch)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementResourceStateAcceptanceFailure {
            reason,
            submission,
            states,
        }));
    }
    let replay = match commit_driver_accepted_with_watch(
        runtime, native, resources, recordings, submission, watch,
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(ReplacementResourceStateAcceptanceFailure {
                reason: ReplacementResourceStateAcceptanceError::Replay(failure.reason.clone()),
                submission: failure.submission,
                states,
            }));
        }
    };
    Ok(AcceptedReplacementResourceStates {
        replay,
        outcomes: states.into_outcomes(),
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
        replacement_recording::ReplacementNativeRecording,
        replacement_submit::QueueTimelineSemaphores,
    };
    use ash::vk;
    use reims_vgpu_core::BackingView;
    use reims_vgpu_core::{
        assemble_prepared_resource_states, prepare_resource_state, BackingRegion, CompletionStamp,
        ExecTransaction, RepresentationRoute, ResolvedExecSegment, ResolvedExecStream,
        ResolvedOperation, ResolvedResourceLifecycle, ResolvedResourceState,
        ResolvedResourceStateTarget, ResourceLifecycleEffect, SessionGeneration, StorageBacking,
        TransactionRecordingPlan, ValidityRepresentations, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        ChannelId, QueueOwnerId, ResourceValidityOps, SegmentBoundary, SegmentKind,
        SessionGenerationId, SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId,
        TransactionId, VulkanDeviceEpochId,
    };

    #[test]
    fn complete_resource_state_batch_is_required_for_driver_acceptance() {
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
        let representation = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                set_host_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                ExecTransaction::<ResolvedOperation<(), (), (), (), ()>> {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(1),
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
                            operations: Box::new([ResolvedOperation::ResourceState(operation)]),
                        }]),
                    }]),
                    accesses: Box::new([]),
                },
            )
            .unwrap();
        let (envelope, _, _, admitted_states, _) = admitted.into_parts();
        let transaction = envelope.id;
        let prepared_state = prepare_resource_state(
            &mut resources,
            &admitted_states,
            0,
            SubmissionId::new(1),
            |_, _| ValidityRepresentations {
                host_write: Some(representation),
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: None,
                guest_visibility_destination: reims_vgpu_core::GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let states =
            assemble_prepared_resource_states(&admitted_states, vec![prepared_state]).unwrap();
        let completions = states.resource_completions();

        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        runtime.recorded(transaction).unwrap();
        runtime.take_submission_ready();
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
        let failure = commit_driver_accepted_resource_states_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            states,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementResourceStateAcceptanceError::BackingSetMismatch
        );
        let (prepared, mut recording) = failure.submission.into_parts();
        recording.backings = failure.states.backings();
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let accepted = commit_driver_accepted_resource_states_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            failure.states,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.resources, completions);
        assert_eq!(accepted.outcomes.len(), 1);
    }
}
