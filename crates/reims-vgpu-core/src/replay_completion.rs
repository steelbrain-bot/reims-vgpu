//! Timeline-proven resource and semantic completion as one replay transition.

use crate::{
    CompletionEvidence, CompletionFact, DirectReplayError, DirectReplayNativeOwner,
    ManagedBackingError, PublishedFact, ResolvedResourceCompletion, ResourceCompletionBatchError,
    ResourceCompletionEffect, ResourceLifecycleOwner, TransactionRuntime, TransactionRuntimeError,
};
use reims_vgpu_protocol::{QueueOwnerId, QueueTimelineValue};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedReplayCompletion<Semantic> {
    pub semantic: Semantic,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayCompletionError {
    NotGpuCompletion,
    SessionGenerationMismatch,
    Runtime(TransactionRuntimeError),
    Resources(ResourceCompletionBatchError),
}

#[derive(Debug)]
pub struct ReplayCompletionFailure<Semantic> {
    pub reason: ReplayCompletionError,
    pub fact: CompletionFact<ResolvedReplayCompletion<Semantic>>,
}

#[derive(Debug)]
pub struct CommittedReplayCompletion<Semantic> {
    pub published: Vec<PublishedFact<Semantic>>,
    pub resources: Vec<ResourceCompletionEffect>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplaySemanticCompletionError {
    NotGpuCompletion,
    SessionGenerationMismatch,
    Runtime(TransactionRuntimeError),
}

#[derive(Debug)]
pub struct ReplaySemanticCompletionFailure<Semantic> {
    pub reason: ReplaySemanticCompletionError,
    pub fact: CompletionFact<Semantic>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayTimelineProgressError {
    Native(DirectReplayError),
    Resources(ManagedBackingError),
}

#[derive(Debug)]
pub struct ReplayTimelineProgress<Semantic, T> {
    pub completions: Vec<CompletionFact<Semantic>>,
    pub retired_native: Vec<T>,
}

/// Advance completion facts and deferred native retirement from the same
/// observed queue counter. Both owners validate monotonicity before either
/// consumes the observation.
pub fn commit_replay_timeline_progress<Semantic: Clone, T>(
    native: &mut DirectReplayNativeOwner<Semantic>,
    resources: &mut ResourceLifecycleOwner<T>,
    queue: QueueOwnerId,
    completed: QueueTimelineValue,
) -> Result<ReplayTimelineProgress<Semantic, T>, ReplayTimelineProgressError> {
    native
        .validate_advance(queue, completed)
        .map_err(ReplayTimelineProgressError::Native)?;
    resources
        .validate_native_retirement_advance(queue, completed)
        .map_err(ReplayTimelineProgressError::Resources)?;
    let completions = native
        .advance(queue, completed)
        .unwrap_or_else(|_| unreachable!("timeline completion progress was prevalidated"));
    let retired_native = resources
        .advance_native_retirement(queue, completed)
        .unwrap_or_else(|_| unreachable!("native retirement progress was prevalidated"));
    Ok(ReplayTimelineProgress {
        completions,
        retired_native,
    })
}

/// Commit one immutable fact obtained from actual queue-timeline progress.
/// The full resource-effect set and runtime transition are validated before
/// either owner changes.
pub fn commit_replay_completion<Semantic: Clone, T>(
    runtime: &mut TransactionRuntime<Semantic>,
    resources: &mut ResourceLifecycleOwner<T>,
    fact: CompletionFact<ResolvedReplayCompletion<Semantic>>,
) -> Result<CommittedReplayCompletion<Semantic>, Box<ReplayCompletionFailure<Semantic>>> {
    let CompletionEvidence::Gpu(_) = fact.evidence else {
        return Err(Box::new(ReplayCompletionFailure {
            reason: ReplayCompletionError::NotGpuCompletion,
            fact,
        }));
    };
    if fact.session_generation != runtime.session_generation() {
        return Err(Box::new(ReplayCompletionFailure {
            reason: ReplayCompletionError::SessionGenerationMismatch,
            fact,
        }));
    }
    if let Err(error) =
        runtime.validate_gpu_complete(fact.transaction, fact.semantic.semantic.clone())
    {
        return Err(Box::new(ReplayCompletionFailure {
            reason: ReplayCompletionError::Runtime(error),
            fact,
        }));
    }
    if let Err(error) = resources.validate_resource_completions(&fact.semantic.resources) {
        return Err(Box::new(ReplayCompletionFailure {
            reason: ReplayCompletionError::Resources(error),
            fact,
        }));
    }

    let resource_effects = resources
        .complete_resources(&fact.semantic.resources)
        .unwrap_or_else(|_| unreachable!("the complete replay completion was prevalidated"));
    let published = runtime
        .gpu_complete(fact.transaction, fact.semantic.semantic)
        .unwrap_or_else(|_| unreachable!("the complete replay completion was prevalidated"));
    Ok(CommittedReplayCompletion {
        published,
        resources: resource_effects,
    })
}

/// Publish a replay fact whose native resource effects were already applied
/// by the timeline-retirement owner. Accepting the inner semantic type rather
/// than [`ResolvedReplayCompletion`] makes a second resource application
/// unrepresentable at this boundary.
pub fn commit_replay_semantic_completion<Semantic: Clone>(
    runtime: &mut TransactionRuntime<Semantic>,
    fact: CompletionFact<Semantic>,
) -> Result<Vec<PublishedFact<Semantic>>, Box<ReplaySemanticCompletionFailure<Semantic>>> {
    let CompletionEvidence::Gpu(_) = fact.evidence else {
        return Err(Box::new(ReplaySemanticCompletionFailure {
            reason: ReplaySemanticCompletionError::NotGpuCompletion,
            fact,
        }));
    };
    if fact.session_generation != runtime.session_generation() {
        return Err(Box::new(ReplaySemanticCompletionFailure {
            reason: ReplaySemanticCompletionError::SessionGenerationMismatch,
            fact,
        }));
    }
    if let Err(error) = runtime.validate_gpu_complete(fact.transaction, fact.semantic.clone()) {
        return Err(Box::new(ReplaySemanticCompletionFailure {
            reason: ReplaySemanticCompletionError::Runtime(error),
            fact,
        }));
    }

    Ok(runtime
        .gpu_complete(fact.transaction, fact.semantic)
        .unwrap_or_else(|_| unreachable!("the semantic replay completion was prevalidated")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BackingRegion, CompletionStamp, DeviceTransactionPayload, DirectReplayNativeOwner,
        ExecTransaction, QueueTimelinePoint, RepresentationRoute, ResolvedExecSegment,
        ResolvedExecStream, ResolvedResourceLifecycle, ResourceLifecycleEffect, SessionGeneration,
        StorageBacking, TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        ChannelId, QueueOwnerId, QueueTimelineValue, SegmentBoundary, SegmentKind,
        SessionGenerationId, SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId,
        TransactionId, VulkanDeviceEpochId,
    };

    #[test]
    fn invalid_resource_effect_preserves_runtime_and_pending_content_for_retry() {
        let generation = SessionGenerationId::new(1);
        let epoch = VulkanDeviceEpochId::new(2);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
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
        runtime.recorded(transaction.id).unwrap();
        runtime.take_submission_ready();
        runtime.submitted(transaction.id).unwrap();

        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
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
        let submission = SubmissionId::new(4);
        resources
            .plan_gpu_write(backing, submission, representation, [BackingRegion::Whole])
            .unwrap();
        let valid = ResolvedResourceCompletion::GpuWrite {
            backing,
            write: submission.into(),
            representation,
        };
        let invalid = ResolvedResourceCompletion::GpuWrite {
            backing,
            write: SubmissionId::new(5).into(),
            representation,
        };
        let fact = CompletionFact {
            transaction: transaction.id,
            session_generation: generation,
            evidence: CompletionEvidence::Gpu(QueueTimelinePoint {
                epoch,
                queue: QueueOwnerId::new(1),
                value: QueueTimelineValue::new(1),
            }),
            semantic: ResolvedReplayCompletion {
                semantic: "done",
                resources: Box::new([valid, invalid]),
            },
        };
        let mut failure = commit_replay_completion(&mut runtime, &mut resources, fact).unwrap_err();
        assert!(matches!(
            failure.reason,
            ReplayCompletionError::Resources(ResourceCompletionBatchError::Completion {
                completion,
                ..
            }) if completion == invalid
        ));
        assert!(runtime
            .validate_gpu_complete(transaction.id, "done")
            .is_ok());
        failure.fact.semantic.resources = Box::new([valid]);
        let committed =
            commit_replay_completion(&mut runtime, &mut resources, failure.fact).unwrap();
        assert_eq!(committed.published[0].transaction, transaction.id);
        assert!(matches!(
            committed.resources.as_slice(),
            [ResourceCompletionEffect::GpuWrite { submission: found, .. }]
                if *found == submission
        ));
    }

    #[test]
    fn retirement_regression_consumes_no_completion_facts() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: TransactionId::new(1),
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = native
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(plan, queue, SessionGenerationId::new(1), "completed")
            .unwrap();
        native.accepted(prepared).unwrap();
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        resources
            .advance_native_retirement(queue, QueueTimelineValue::new(5))
            .unwrap();
        assert!(matches!(
            commit_replay_timeline_progress(
                &mut native,
                &mut resources,
                queue,
                QueueTimelineValue::new(4),
            ),
            Err(ReplayTimelineProgressError::Resources(
                ManagedBackingError::Retirement(crate::NativeRetirementError::TimelineRegressed)
            ))
        ));
        let progress = commit_replay_timeline_progress(
            &mut native,
            &mut resources,
            queue,
            QueueTimelineValue::new(5),
        )
        .unwrap();
        assert_eq!(progress.completions.len(), 1);
        assert_eq!(progress.completions[0].transaction, TransactionId::new(1));
    }
}
