//! Native-lifecycle owner for direct replay and conformance backends.
//!
//! This owner contains no Vulkan handles. It composes the invariants a native
//! queue adapter must obey: fixed-worker assignment, producer-to-timeline wait
//! translation, monotonic queue points, and timeline-proven completion facts.
//! Native queue acceptance and timeline completion remain distinct methods.

use crate::{
    AbandonedCompletion, CompletionFact, CompletionOwnerError, NativeDependencyError,
    NativeDependencyOwner, NativeSubmissionPlan, QueueTimelineError, QueueTimelineOwner,
    RecordingAssignmentError, RecordingAssignmentOwner, TimelineCompletionOwner,
    TransactionRecordingPlan, WaitDependencyCause,
};
use reims_vgpu_protocol::{
    QueueOwnerId, QueueTimelineValue, SessionGenerationId, TransactionId, VulkanDeviceEpochId,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectReplayError {
    Assignment(RecordingAssignmentError),
    Dependency(NativeDependencyError),
    Timeline(QueueTimelineError),
    Completion(CompletionOwnerError),
}

impl From<RecordingAssignmentError> for DirectReplayError {
    fn from(error: RecordingAssignmentError) -> Self {
        Self::Assignment(error)
    }
}

impl From<NativeDependencyError> for DirectReplayError {
    fn from(error: NativeDependencyError) -> Self {
        Self::Dependency(error)
    }
}

impl From<QueueTimelineError> for DirectReplayError {
    fn from(error: QueueTimelineError) -> Self {
        Self::Timeline(error)
    }
}

impl From<CompletionOwnerError> for DirectReplayError {
    fn from(error: CompletionOwnerError) -> Self {
        Self::Completion(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedNativeSubmission {
    pub transaction: TransactionId,
    pub point: crate::QueueTimelinePoint,
    /// Candidates whose last missing producer acquired a native point through
    /// this acceptance.
    pub newly_ready: Vec<NativeSubmissionPlan>,
}

/// Submission ownership after a queue timeline point has been reserved but
/// before the driver has accepted the native submit containing that signal.
#[derive(Debug, Eq, PartialEq)]
#[must_use = "a prepared submission must be accepted, explicitly canceled, or abandoned with its replay owner"]
pub struct PreparedNativeSubmission<Semantic> {
    plan: NativeSubmissionPlan,
    point: crate::QueueTimelinePoint,
    recording_worker: crate::RecordingWorkerId,
    session_generation: SessionGenerationId,
    semantic: Semantic,
}

/// A validated native plan whose semantic signal point is deliberately not
/// allocated yet. Asynchronous native phases may allocate earlier points on
/// the destination queue before this owner becomes the final submission.
#[derive(Debug, Eq, PartialEq)]
#[must_use = "a native execution chain must reach final preparation or explicit cancellation"]
pub struct PreparedNativeExecutionChain<Semantic> {
    plan: NativeSubmissionPlan,
    recording_worker: crate::RecordingWorkerId,
    session_generation: SessionGenerationId,
    semantic: Semantic,
}

impl<Semantic> PreparedNativeExecutionChain<Semantic> {
    pub const fn plan(&self) -> &NativeSubmissionPlan {
        &self.plan
    }

    pub const fn recording_worker(&self) -> crate::RecordingWorkerId {
        self.recording_worker
    }

    pub const fn session_generation(&self) -> SessionGenerationId {
        self.session_generation
    }

    pub const fn semantic(&self) -> &Semantic {
        &self.semantic
    }

    pub fn map_semantic(self, map: impl FnOnce(Semantic) -> Semantic) -> Self {
        PreparedNativeExecutionChain {
            plan: self.plan,
            recording_worker: self.recording_worker,
            session_generation: self.session_generation,
            semantic: map(self.semantic),
        }
    }
}

/// One queue submission required by a prepared semantic transaction but not
/// itself a semantic completion. The point shares the queue's sole allocator,
/// so auxiliary and semantic signals can never collide or reorder silently.
#[derive(Debug, Eq, PartialEq)]
#[must_use = "an auxiliary point must be submitted or explicitly skipped"]
pub struct PreparedAuxiliaryNativeSubmission {
    transaction: TransactionId,
    point: crate::QueueTimelinePoint,
    recording_worker: crate::RecordingWorkerId,
    prerequisite_plan: Option<NativeSubmissionPlan>,
}

/// One presentation queue point reserved from the same allocator as ordinary
/// execution and auxiliary submissions. It carries no recording-worker or
/// semantic-completion ownership; the Present transaction owns those facts.
#[derive(Debug, Eq, PartialEq)]
#[must_use = "a prepared present point must be submitted or explicitly skipped"]
pub struct PreparedPresentNativeSubmission {
    transaction: TransactionId,
    point: crate::QueueTimelinePoint,
}

impl PreparedPresentNativeSubmission {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn point(&self) -> crate::QueueTimelinePoint {
        self.point
    }
}

impl PreparedAuxiliaryNativeSubmission {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn point(&self) -> crate::QueueTimelinePoint {
        self.point
    }

    pub const fn recording_worker(&self) -> crate::RecordingWorkerId {
        self.recording_worker
    }

    /// Semantic waits and hazards the first native phase inherits from its
    /// still-unsubmitted parent transaction.
    pub const fn prerequisite_plan(&self) -> Option<&NativeSubmissionPlan> {
        self.prerequisite_plan.as_ref()
    }
}

impl<Semantic> PreparedNativeSubmission<Semantic> {
    pub const fn plan(&self) -> &NativeSubmissionPlan {
        &self.plan
    }

    pub const fn point(&self) -> crate::QueueTimelinePoint {
        self.point
    }

    pub const fn recording_worker(&self) -> crate::RecordingWorkerId {
        self.recording_worker
    }

    pub const fn session_generation(&self) -> SessionGenerationId {
        self.session_generation
    }

    pub const fn semantic(&self) -> &Semantic {
        &self.semantic
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePreparationFailure<Semantic> {
    pub reason: DirectReplayError,
    pub plan: NativeSubmissionPlan,
    pub session_generation: SessionGenerationId,
    pub semantic: Semantic,
}

#[derive(Debug, Eq, PartialEq)]
pub struct NativeChainFinalPreparationFailure<Semantic> {
    pub reason: DirectReplayError,
    pub chain: PreparedNativeExecutionChain<Semantic>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct NativeAcceptanceFailure<Semantic> {
    pub reason: DirectReplayError,
    pub prepared: PreparedNativeSubmission<Semantic>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct CanceledNativeSubmission<Semantic> {
    pub plan: NativeSubmissionPlan,
    pub point: crate::QueueTimelinePoint,
    pub session_generation: SessionGenerationId,
    pub semantic: Semantic,
}

#[derive(Debug, Eq, PartialEq)]
pub struct CanceledNativeExecutionChain<Semantic> {
    pub plan: NativeSubmissionPlan,
    pub session_generation: SessionGenerationId,
    pub semantic: Semantic,
}

#[derive(Debug, Eq, PartialEq)]
pub struct NativeChainCancellationFailure<Semantic> {
    pub reason: DirectReplayError,
    pub chain: PreparedNativeExecutionChain<Semantic>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct NativeCancellationFailure<Semantic> {
    pub reason: DirectReplayError,
    pub prepared: PreparedNativeSubmission<Semantic>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectReplayAbandonment<Semantic> {
    pub unsubmitted: Vec<TransactionId>,
    pub submitted: Vec<AbandonedCompletion<Semantic>>,
}

#[derive(Clone, Debug)]
pub struct DirectReplayNativeOwner<Semantic> {
    assignments: RecordingAssignmentOwner,
    dependencies: NativeDependencyOwner,
    timelines: QueueTimelineOwner,
    completions: TimelineCompletionOwner<Semantic>,
}

impl<Semantic: Clone> DirectReplayNativeOwner<Semantic> {
    pub fn new(
        epoch: VulkanDeviceEpochId,
        recording_workers: usize,
    ) -> Result<Self, DirectReplayError> {
        Ok(Self {
            assignments: RecordingAssignmentOwner::new(recording_workers)?,
            dependencies: NativeDependencyOwner::default(),
            timelines: QueueTimelineOwner::new(epoch),
            completions: TimelineCompletionOwner::new(epoch),
        })
    }

    pub fn assign_recording(
        &mut self,
        plan: TransactionRecordingPlan,
    ) -> Result<crate::RecordingWorkerId, DirectReplayError> {
        self.assignments
            .assign(plan.transaction, plan.domain, plan.continuation_predecessor)
            .map_err(Into::into)
    }

    pub fn recording_worker(&self, transaction: TransactionId) -> Option<crate::RecordingWorkerId> {
        self.assignments.worker(transaction)
    }

    /// Cancel every old-generation native obligation at a platform reset while
    /// preserving this Vulkan incarnation's queue timeline allocation.
    pub fn platform_reset(&mut self) -> DirectReplayAbandonment<Semantic> {
        let workers = self.assignments.worker_count();
        let epoch = self.timelines.epoch();
        self.assignments = RecordingAssignmentOwner::new(workers)
            .expect("a live replay owner has a nonempty recording population");
        let dependencies = std::mem::take(&mut self.dependencies);
        let completions =
            std::mem::replace(&mut self.completions, TimelineCompletionOwner::new(epoch));
        DirectReplayAbandonment {
            unsubmitted: dependencies.abandon(),
            submitted: completions.abandon(),
        }
    }

    /// Reserve a Present submission from the actual queue's sole signal-value
    /// sequence. Transaction readiness is validated by the semantic Present
    /// owner before this boundary; this owner contributes only native ordering.
    pub fn prepare_present(
        &mut self,
        transaction: TransactionId,
        queue: QueueOwnerId,
    ) -> Result<PreparedPresentNativeSubmission, DirectReplayError> {
        let point = self.timelines.allocate(queue)?;
        Ok(PreparedPresentNativeSubmission { transaction, point })
    }

    /// See [`NativeDependencyOwner::parked_candidates`].
    pub fn parked_candidates(&self) -> Vec<(TransactionId, Vec<TransactionId>)> {
        self.dependencies.parked_candidates()
    }

    /// See [`NativeDependencyOwner::has_submission_point`].
    #[must_use]
    pub fn has_submission_point(&self, transaction: TransactionId) -> bool {
        self.dependencies.has_submission_point(transaction)
    }

    pub fn queue_candidate(
        &mut self,
        transaction: TransactionId,
        prerequisites: impl Into<Box<[(TransactionId, WaitDependencyCause)]>>,
    ) -> Result<Vec<NativeSubmissionPlan>, DirectReplayError> {
        self.dependencies
            .queue_ready(transaction, prerequisites)
            .map_err(Into::into)
    }

    pub fn queue_candidate_with_hazards(
        &mut self,
        transaction: TransactionId,
        prerequisites: impl Into<Box<[(TransactionId, WaitDependencyCause)]>>,
        hazards: impl Into<Box<[crate::HazardRequirement]>>,
    ) -> Result<Vec<NativeSubmissionPlan>, DirectReplayError> {
        self.dependencies
            .queue_ready_with_hazards(transaction, prerequisites, hazards)
            .map_err(Into::into)
    }

    /// Reserve the signal point which the native submit will carry. A driver
    /// refusal may consume this value; no transaction or completion owns it
    /// until [`Self::accepted`] records successful queue acceptance.
    pub fn prepare(
        &mut self,
        plan: NativeSubmissionPlan,
        queue: QueueOwnerId,
        session_generation: SessionGenerationId,
        semantic: Semantic,
    ) -> Result<PreparedNativeSubmission<Semantic>, NativePreparationFailure<Semantic>> {
        let chain = self.prepare_execution_chain(plan, session_generation, semantic)?;
        self.prepare_execution_chain_final(chain, queue)
            .map_err(|failure| NativePreparationFailure {
                reason: failure.reason,
                plan: failure.chain.plan,
                session_generation: failure.chain.session_generation,
                semantic: failure.chain.semantic,
            })
    }

    /// Validate assignment and dependency ownership without allocating the
    /// final semantic timeline point.
    pub fn prepare_execution_chain(
        &self,
        plan: NativeSubmissionPlan,
        session_generation: SessionGenerationId,
        semantic: Semantic,
    ) -> Result<PreparedNativeExecutionChain<Semantic>, NativePreparationFailure<Semantic>> {
        let recording_worker = match self.assignments.worker(plan.transaction) {
            Some(worker) => worker,
            None => {
                return Err(NativePreparationFailure {
                    reason: DirectReplayError::Assignment(
                        RecordingAssignmentError::UnknownTransaction,
                    ),
                    plan,
                    session_generation,
                    semantic,
                });
            }
        };
        if let Err(error) = self.dependencies.validate_plan(&plan) {
            return Err(NativePreparationFailure {
                reason: error.into(),
                plan,
                session_generation,
                semantic,
            });
        }
        Ok(PreparedNativeExecutionChain {
            plan,
            recording_worker,
            session_generation,
            semantic,
        })
    }

    /// Allocate the chain's one semantic completion point after every native
    /// auxiliary phase has received its earlier point.
    pub fn prepare_execution_chain_final(
        &mut self,
        chain: PreparedNativeExecutionChain<Semantic>,
        queue: QueueOwnerId,
    ) -> Result<PreparedNativeSubmission<Semantic>, NativeChainFinalPreparationFailure<Semantic>>
    {
        let point = match self.timelines.allocate(queue) {
            Ok(point) => point,
            Err(error) => {
                return Err(NativeChainFinalPreparationFailure {
                    reason: error.into(),
                    chain,
                })
            }
        };
        Ok(PreparedNativeSubmission {
            plan: chain.plan,
            point,
            recording_worker: chain.recording_worker,
            session_generation: chain.session_generation,
            semantic: chain.semantic,
        })
    }

    pub fn prepare_execution_chain_auxiliary(
        &mut self,
        chain: &PreparedNativeExecutionChain<Semantic>,
        queue: QueueOwnerId,
    ) -> Result<PreparedAuxiliaryNativeSubmission, DirectReplayError> {
        self.dependencies.validate_plan(&chain.plan)?;
        let point = self.timelines.allocate(queue)?;
        Ok(PreparedAuxiliaryNativeSubmission {
            transaction: chain.plan.transaction,
            point,
            recording_worker: chain.recording_worker,
            prerequisite_plan: Some(chain.plan.clone()),
        })
    }

    pub fn prepare_execution_chain_auxiliary_after(
        &mut self,
        chain: &PreparedNativeExecutionChain<Semantic>,
        predecessor: crate::QueueTimelinePoint,
    ) -> Result<PreparedAuxiliaryNativeSubmission, DirectReplayError> {
        self.dependencies.validate_plan(&chain.plan)?;
        let point = self.timelines.allocate_after(predecessor)?;
        Ok(PreparedAuxiliaryNativeSubmission {
            transaction: chain.plan.transaction,
            point,
            recording_worker: chain.recording_worker,
            prerequisite_plan: None,
        })
    }

    /// Cancel a validated execution chain before its final semantic point is
    /// allocated. Auxiliary points already accepted by the driver retain
    /// their independent native/resource retirement obligations; this closes
    /// only the unsubmitted parent transaction's assignment and issued plan.
    pub fn cancel_execution_chain(
        &mut self,
        chain: PreparedNativeExecutionChain<Semantic>,
    ) -> Result<CanceledNativeExecutionChain<Semantic>, NativeChainCancellationFailure<Semantic>>
    {
        let mut assignments = self.assignments.clone();
        let mut dependencies = self.dependencies.clone();
        if let Err(reason) = assignments
            .retire(chain.plan.transaction)
            .map_err(DirectReplayError::from)
            .and_then(|()| {
                dependencies
                    .cancel_issued(&chain.plan)
                    .map_err(DirectReplayError::from)
            })
        {
            return Err(NativeChainCancellationFailure { reason, chain });
        }
        self.assignments = assignments;
        self.dependencies = dependencies;
        Ok(CanceledNativeExecutionChain {
            plan: chain.plan,
            session_generation: chain.session_generation,
            semantic: chain.semantic,
        })
    }

    /// Reserve a queue point for native prerequisite work belonging to an
    /// already prepared semantic submission. This creates no dependency or
    /// completion fact of its own; destination submission ordering carries the
    /// prerequisite into semantic completion.
    pub fn prepare_auxiliary(
        &mut self,
        parent: &PreparedNativeSubmission<Semantic>,
        queue: QueueOwnerId,
    ) -> Result<PreparedAuxiliaryNativeSubmission, DirectReplayError> {
        let worker = self.assignments.worker(parent.plan.transaction).ok_or(
            DirectReplayError::Assignment(RecordingAssignmentError::UnknownTransaction),
        )?;
        if worker != parent.recording_worker {
            return Err(DirectReplayError::Assignment(
                RecordingAssignmentError::UnknownTransaction,
            ));
        }
        self.dependencies.validate_plan(&parent.plan)?;
        let point = self.timelines.allocate(queue)?;
        Ok(PreparedAuxiliaryNativeSubmission {
            transaction: parent.plan.transaction,
            point,
            recording_worker: worker,
            prerequisite_plan: Some(parent.plan.clone()),
        })
    }

    /// Reserve an auxiliary point ordered after a previously accepted point on
    /// the same queue. Queue-family release submissions use this form so an
    /// unissued or cross-epoch predecessor cannot authorize ownership change.
    pub fn prepare_auxiliary_after(
        &mut self,
        parent: &PreparedNativeSubmission<Semantic>,
        predecessor: crate::QueueTimelinePoint,
    ) -> Result<PreparedAuxiliaryNativeSubmission, DirectReplayError> {
        let worker = self.assignments.worker(parent.plan.transaction).ok_or(
            DirectReplayError::Assignment(RecordingAssignmentError::UnknownTransaction),
        )?;
        if worker != parent.recording_worker {
            return Err(DirectReplayError::Assignment(
                RecordingAssignmentError::UnknownTransaction,
            ));
        }
        self.dependencies.validate_plan(&parent.plan)?;
        let point = self.timelines.allocate_after(predecessor)?;
        Ok(PreparedAuxiliaryNativeSubmission {
            transaction: parent.plan.transaction,
            point,
            recording_worker: worker,
            prerequisite_plan: None,
        })
    }

    /// Apply successful native queue acceptance atomically to dependency and
    /// completion ownership. Driver return is still not GPU completion.
    pub fn validate_acceptance(
        &self,
        prepared: &PreparedNativeSubmission<Semantic>,
    ) -> Result<(), DirectReplayError> {
        let mut completions = self.completions.clone();
        let mut dependencies = self.dependencies.clone();
        completions.register(
            prepared.plan.transaction,
            prepared.session_generation,
            prepared.point,
            prepared.semantic.clone(),
        )?;
        dependencies.submitted(prepared.plan.transaction, prepared.point)?;
        Ok(())
    }

    /// Apply successful native queue acceptance atomically to dependency and
    /// completion ownership. Driver return is still not GPU completion.
    pub fn accepted(
        &mut self,
        prepared: PreparedNativeSubmission<Semantic>,
    ) -> Result<AcceptedNativeSubmission, NativeAcceptanceFailure<Semantic>> {
        if let Err(reason) = self.validate_acceptance(&prepared) {
            return Err(NativeAcceptanceFailure { reason, prepared });
        }
        let mut completions = self.completions.clone();
        let mut dependencies = self.dependencies.clone();
        completions
            .register(
                prepared.plan.transaction,
                prepared.session_generation,
                prepared.point,
                prepared.semantic.clone(),
            )
            .expect("native acceptance was prevalidated");
        let newly_ready = dependencies
            .submitted(prepared.plan.transaction, prepared.point)
            .expect("native acceptance was prevalidated");
        self.completions = completions;
        self.dependencies = dependencies;
        Ok(AcceptedNativeSubmission {
            transaction: prepared.plan.transaction,
            point: prepared.point,
            newly_ready,
        })
    }

    /// Cancel a prepared token after its queue signal point has been explicitly
    /// skipped. Cancellation creates neither a producer point nor a completion
    /// fact and retires its recording assignment.
    pub fn validate_cancellation(
        &self,
        prepared: &PreparedNativeSubmission<Semantic>,
    ) -> Result<(), DirectReplayError> {
        let mut assignments = self.assignments.clone();
        let mut dependencies = self.dependencies.clone();
        assignments.retire(prepared.plan.transaction)?;
        dependencies.cancel_issued(&prepared.plan)?;
        Ok(())
    }

    /// Cancel a prepared token whose reserved signal point was either skipped
    /// or consumed by a native attempt that the driver refused. Cancellation
    /// creates neither a producer point nor a completion fact.
    pub fn cancel_prepared(
        &mut self,
        prepared: PreparedNativeSubmission<Semantic>,
    ) -> Result<CanceledNativeSubmission<Semantic>, NativeCancellationFailure<Semantic>> {
        if let Err(reason) = self.validate_cancellation(&prepared) {
            return Err(NativeCancellationFailure { reason, prepared });
        }
        let mut assignments = self.assignments.clone();
        let mut dependencies = self.dependencies.clone();
        assignments
            .retire(prepared.plan.transaction)
            .expect("native cancellation was prevalidated");
        dependencies
            .cancel_issued(&prepared.plan)
            .expect("native cancellation was prevalidated");
        self.assignments = assignments;
        self.dependencies = dependencies;
        Ok(CanceledNativeSubmission {
            plan: prepared.plan,
            point: prepared.point,
            session_generation: prepared.session_generation,
            semantic: prepared.semantic,
        })
    }

    pub fn advance(
        &mut self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Vec<CompletionFact<Semantic>>, DirectReplayError> {
        self.completions
            .advance(queue, completed)
            .map_err(Into::into)
    }

    pub fn validate_advance(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<(), DirectReplayError> {
        self.completions
            .validate_advance(queue, completed)
            .map_err(Into::into)
    }

    pub fn pending_completions(&self) -> usize {
        self.completions.pending()
    }

    /// See [`TimelineCompletionOwner::last_submitted`].
    #[must_use]
    pub fn last_submitted_point(&self, queue: QueueOwnerId) -> Option<QueueTimelineValue> {
        self.completions.last_submitted(queue)
    }

    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), DirectReplayError> {
        self.validate_retire(transaction)?;
        let mut assignments = self.assignments.clone();
        let mut dependencies = self.dependencies.clone();
        assignments
            .retire(transaction)
            .expect("native retirement was prevalidated");
        dependencies
            .retire(transaction)
            .expect("native retirement was prevalidated");
        self.assignments = assignments;
        self.dependencies = dependencies;
        Ok(())
    }

    pub fn validate_retire(&self, transaction: TransactionId) -> Result<(), DirectReplayError> {
        self.completions.validate_retired(transaction)?;
        let mut assignments = self.assignments.clone();
        let mut dependencies = self.dependencies.clone();
        assignments.retire(transaction)?;
        dependencies.retire(transaction)?;
        Ok(())
    }

    /// Device loss returns unsubmitted candidates and submitted native work as
    /// distinct terminal obligations. Neither is a successful completion.
    pub fn abandon(self) -> DirectReplayAbandonment<Semantic> {
        DirectReplayAbandonment {
            unsubmitted: self.dependencies.abandon(),
            submitted: self.completions.abandon(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExplicitWaitCause, HazardCause, NativeWait, QueueTimelinePoint};
    use reims_vgpu_protocol::{ChannelId, SubmissionDomainId};

    fn recording(
        transaction: u64,
        domain: u64,
        predecessor: Option<u64>,
    ) -> TransactionRecordingPlan {
        TransactionRecordingPlan {
            transaction: TransactionId::new(transaction),
            domain: SubmissionDomainId::new(domain),
            continuation_predecessor: predecessor.map(TransactionId::new),
        }
    }

    #[test]
    fn acceptance_is_not_completion_and_future_point_releases_exact_consumer() {
        let mut owner = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(3), 2).unwrap();
        let cause = WaitDependencyCause::Explicit(ExplicitWaitCause::Stamp {
            source_channel: ChannelId::new(2),
            value: 4,
        });
        assert!(owner
            .queue_candidate(TransactionId::new(1), [(TransactionId::new(2), cause)],)
            .unwrap()
            .is_empty());
        owner.assign_recording(recording(2, 1, None)).unwrap();
        let producer = owner
            .queue_candidate(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let producer = owner
            .prepare(
                producer,
                QueueOwnerId::new(1),
                SessionGenerationId::new(7),
                "producer",
            )
            .unwrap();
        assert_eq!(producer.point.value, QueueTimelineValue::new(1));
        let producer = owner.accepted(producer).unwrap();
        assert_eq!(owner.pending_completions(), 1);
        assert_eq!(producer.newly_ready.len(), 1);
        assert_eq!(
            producer.newly_ready[0].waits.as_ref(),
            [NativeWait {
                producer: TransactionId::new(2),
                point: QueueTimelinePoint {
                    epoch: VulkanDeviceEpochId::new(3),
                    queue: QueueOwnerId::new(1),
                    value: QueueTimelineValue::new(1),
                },
                cause,
            }]
        );
        assert!(owner
            .advance(QueueOwnerId::new(1), QueueTimelineValue::new(0))
            .unwrap()
            .is_empty());
        assert_eq!(
            owner
                .advance(QueueOwnerId::new(1), QueueTimelineValue::new(1))
                .unwrap()[0]
                .semantic,
            "producer"
        );
    }

    #[test]
    fn continuation_recording_assignment_stays_on_one_fixed_worker() {
        let mut owner = DirectReplayNativeOwner::<()>::new(VulkanDeviceEpochId::new(1), 2).unwrap();
        let first = owner.assign_recording(recording(1, 9, None)).unwrap();
        let second = owner.assign_recording(recording(2, 9, Some(1))).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn native_preparation_requires_and_carries_the_established_worker() {
        let mut owner = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(1), 1).unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let failure = owner
            .prepare(
                plan,
                QueueOwnerId::new(0),
                SessionGenerationId::new(1),
                "owned",
            )
            .unwrap_err();
        assert_eq!(
            failure.reason,
            DirectReplayError::Assignment(RecordingAssignmentError::UnknownTransaction)
        );
        assert_eq!(failure.semantic, "owned");

        let worker = owner.assign_recording(recording(1, 1, None)).unwrap();
        let prepared = owner
            .prepare(
                failure.plan,
                QueueOwnerId::new(0),
                failure.session_generation,
                failure.semantic,
            )
            .unwrap();
        assert_eq!(prepared.recording_worker(), worker);
        assert_eq!(prepared.point().value, QueueTimelineValue::new(1));
    }

    #[test]
    fn auxiliary_points_share_the_exact_queue_allocator_without_semantic_completion() {
        let epoch = VulkanDeviceEpochId::new(1);
        let mut owner = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        owner.assign_recording(recording(1, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let parent = owner
            .prepare(
                plan,
                QueueOwnerId::new(2),
                SessionGenerationId::new(1),
                "done",
            )
            .unwrap();
        let first = owner
            .prepare_auxiliary(&parent, QueueOwnerId::new(1))
            .unwrap();
        let second = owner
            .prepare_auxiliary(&parent, QueueOwnerId::new(1))
            .unwrap();
        assert_eq!(first.transaction(), TransactionId::new(1));
        assert_eq!(first.recording_worker(), parent.recording_worker());
        assert_eq!(first.point().value, QueueTimelineValue::new(1));
        assert_eq!(second.point().value, QueueTimelineValue::new(2));
        assert_eq!(owner.pending_completions(), 0);
        let successor = owner
            .prepare_auxiliary_after(&parent, second.point())
            .unwrap();
        assert_eq!(successor.point().value, QueueTimelineValue::new(3));
    }

    #[test]
    fn present_points_share_the_actual_queue_signal_sequence() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(2);
        let mut owner = DirectReplayNativeOwner::<()>::new(epoch, 1).unwrap();
        let first = owner.prepare_present(TransactionId::new(8), queue).unwrap();
        let second = owner.prepare_present(TransactionId::new(9), queue).unwrap();
        assert_eq!(first.transaction(), TransactionId::new(8));
        assert_eq!(first.point().value, QueueTimelineValue::new(1));
        assert_eq!(second.point().value, QueueTimelineValue::new(2));
        assert_eq!(owner.pending_completions(), 0);
    }

    #[test]
    fn execution_chain_allocates_its_semantic_point_after_same_queue_auxiliaries() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(1);
        let transaction = TransactionId::new(1);
        let mut owner = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        owner.assign_recording(recording(1, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let chain = owner
            .prepare_execution_chain(plan, SessionGenerationId::new(1), "done")
            .unwrap();
        let first = owner
            .prepare_execution_chain_auxiliary(&chain, queue)
            .unwrap();
        assert_eq!(first.point.value, QueueTimelineValue::new(1));
        assert!(first.prerequisite_plan().is_some());
        let second = owner
            .prepare_execution_chain_auxiliary_after(&chain, first.point)
            .unwrap();
        assert_eq!(second.point.value, QueueTimelineValue::new(2));
        assert!(second.prerequisite_plan().is_none());
        let final_submission = owner.prepare_execution_chain_final(chain, queue).unwrap();
        assert_eq!(final_submission.point.value, QueueTimelineValue::new(3));
        assert!(owner.validate_acceptance(&final_submission).is_ok());
        assert_eq!(owner.pending_completions(), 0);
    }

    #[test]
    fn point_free_execution_chain_cancels_without_allocating_a_signal() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(1);
        let transaction = TransactionId::new(1);
        let mut owner = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        owner.assign_recording(recording(1, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let chain = owner
            .prepare_execution_chain(plan, SessionGenerationId::new(1), "canceled")
            .unwrap();
        let canceled = owner.cancel_execution_chain(chain).unwrap();
        assert_eq!(canceled.semantic, "canceled");
        assert_eq!(owner.pending_completions(), 0);

        owner.assign_recording(recording(2, 1, None)).unwrap();
        let next_plan = owner
            .queue_candidate(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let next = owner
            .prepare(next_plan, queue, SessionGenerationId::new(1), "next")
            .unwrap();
        assert_eq!(next.point.value, QueueTimelineValue::new(1));
    }

    #[test]
    fn device_loss_does_not_turn_native_obligations_into_completion() {
        let mut owner = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(1), 1).unwrap();
        owner
            .queue_candidate(
                TransactionId::new(1),
                [(
                    TransactionId::new(9),
                    WaitDependencyCause::ResourceHazard(HazardCause::Buffer),
                )],
            )
            .unwrap();
        owner.assign_recording(recording(2, 1, None)).unwrap();
        let submitted = owner
            .queue_candidate(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let submitted = owner
            .prepare(
                submitted,
                QueueOwnerId::new(0),
                SessionGenerationId::new(1),
                "pending",
            )
            .unwrap();
        owner.accepted(submitted).unwrap();
        let abandoned = owner.abandon();
        assert_eq!(abandoned.unsubmitted, vec![TransactionId::new(1)]);
        assert_eq!(abandoned.submitted.len(), 1);
        assert_eq!(abandoned.submitted[0].transaction, TransactionId::new(2));
    }

    #[test]
    fn platform_reset_abandons_old_work_without_rewinding_the_queue_timeline() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(0);
        let mut owner = DirectReplayNativeOwner::new(epoch, 2).unwrap();
        owner
            .queue_candidate(
                TransactionId::new(1),
                [(
                    TransactionId::new(9),
                    WaitDependencyCause::ResourceHazard(HazardCause::Buffer),
                )],
            )
            .unwrap();
        owner.assign_recording(recording(2, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let submitted = owner
            .prepare(plan, queue, SessionGenerationId::new(1), "old")
            .unwrap();
        assert_eq!(submitted.point.value, QueueTimelineValue::new(1));
        owner.accepted(submitted).unwrap();

        let abandoned = owner.platform_reset();
        assert_eq!(abandoned.unsubmitted, vec![TransactionId::new(1)]);
        assert_eq!(abandoned.submitted.len(), 1);
        assert_eq!(abandoned.submitted[0].semantic, "old");

        owner.assign_recording(recording(3, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(3),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let next = owner
            .prepare(plan, queue, SessionGenerationId::new(2), "new")
            .unwrap();
        assert_eq!(next.point.value, QueueTimelineValue::new(2));
        assert_eq!(next.recording_worker().index(), 0);
    }

    #[test]
    fn refused_native_attempt_consumes_no_completion_but_does_not_reuse_its_point() {
        let mut owner = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(1), 1).unwrap();
        owner.assign_recording(recording(1, 1, None)).unwrap();
        let first = owner
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let refused = owner
            .prepare(
                first,
                QueueOwnerId::new(0),
                SessionGenerationId::new(1),
                "refused",
            )
            .unwrap();
        assert_eq!(refused.point.value, QueueTimelineValue::new(1));
        drop(refused);
        assert_eq!(owner.pending_completions(), 0);

        owner.assign_recording(recording(2, 1, None)).unwrap();
        let second = owner
            .queue_candidate(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let second = owner
            .prepare(
                second,
                QueueOwnerId::new(0),
                SessionGenerationId::new(1),
                "accepted",
            )
            .unwrap();
        assert_eq!(second.point.value, QueueTimelineValue::new(2));
        owner.accepted(second).unwrap();
        assert_eq!(owner.pending_completions(), 1);
    }

    #[test]
    fn preparation_accepts_only_the_exact_plan_issued_by_dependency_readiness() {
        let mut owner = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(1), 1).unwrap();
        owner.assign_recording(recording(1, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let mut forged = plan.clone();
        forged.waits = Box::new([NativeWait {
            producer: TransactionId::new(9),
            point: QueueTimelinePoint {
                epoch: VulkanDeviceEpochId::new(1),
                queue: QueueOwnerId::new(0),
                value: QueueTimelineValue::new(1),
            },
            cause: WaitDependencyCause::ResourceHazard(HazardCause::Buffer),
        }]);
        assert_eq!(
            owner
                .prepare(
                    forged,
                    QueueOwnerId::new(0),
                    SessionGenerationId::new(1),
                    "forged",
                )
                .unwrap_err()
                .reason,
            DirectReplayError::Dependency(NativeDependencyError::PlanMismatch)
        );
        let prepared = owner
            .prepare(
                plan,
                QueueOwnerId::new(0),
                SessionGenerationId::new(1),
                "exact",
            )
            .unwrap();
        assert_eq!(prepared.point.value, QueueTimelineValue::new(1));
    }

    #[test]
    fn canceled_preparation_is_neither_unsubmitted_work_nor_completion() {
        let mut owner = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(1), 1).unwrap();
        let worker = owner.assign_recording(recording(1, 1, None)).unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = owner
            .prepare(
                plan,
                QueueOwnerId::new(0),
                SessionGenerationId::new(1),
                "canceled",
            )
            .unwrap();
        assert_eq!(prepared.recording_worker(), worker);
        let canceled = owner.cancel_prepared(prepared).unwrap();
        assert_eq!(canceled.semantic, "canceled");
        assert_eq!(owner.pending_completions(), 0);
        let abandoned = owner.abandon();
        assert!(abandoned.unsubmitted.is_empty());
        assert!(abandoned.submitted.is_empty());
    }
}
