//! Backend-independent submission envelopes retained by device state.

use reims_vgpu_protocol::{
    ResourceId, ResourceObject, SegmentBoundary, SubmissionId, SubmissionIdentity,
    SubmissionResourceUse, TaskId,
};
use std::collections::VecDeque;
use std::sync::Arc;

/// Protocol context shared by every operation in one submitted command stream.
///
/// Each value is an immutable snapshot. Executors may retain it without
/// observing later movement of the device-owned submission cursor or mutation
/// of the decoder and its resource-list accumulator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionContext {
    pub identity: SubmissionIdentity,
    pub resources: Arc<[SubmissionResourceUse]>,
    /// Every admitted segment in serialized-stream order.
    pub segments: Arc<[SegmentBoundary]>,
    /// Segment containing the operation currently submitted to the executor.
    pub segment: Option<SegmentBoundary>,
}

impl SubmissionContext {
    /// Context for direct test and tool operations outside a decoded EXEC packet.
    pub fn standalone(task_id: u32) -> Self {
        Self {
            identity: SubmissionIdentity {
                id: SubmissionId::new(0),
                task: TaskId::new(task_id),
            },
            resources: Arc::from([]),
            segments: Arc::from([]),
            segment: None,
        }
    }
}

/// Contract-owned resource reach of one complete submission.
///
/// This is the admission key for packet-level parallelism.  A nonzero EXEC may
/// overlap another only when every participant resolved to a generational
/// resource identity and the two sets are disjoint.  Standalone operations and
/// unresolved resource-list entries remain exclusive: neither carries enough
/// contract information to prove independence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubmissionFootprint {
    resources: Box<[ResourceId<ResourceObject>]>,
    standalone: bool,
    unresolved: bool,
}

/// First contract fact that prevents two submissions from overlapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionConflict {
    Standalone,
    UnresolvedParticipant,
    SharedResource(ResourceId<ResourceObject>),
}

/// Why a whole EXEC could not acquire recording ownership yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionAdmissionRefusal {
    AlreadyActive(SubmissionIdentity),
    Conflict {
        active: SubmissionIdentity,
        reason: SubmissionConflict,
    },
}

/// Unbounded ordered ledger of EXECs that currently own recording resources.
///
/// Queueing policy lives above this type. This owner answers only whether a
/// candidate may overlap every already-admitted submission and retires that
/// ownership only by exact submission identity.
#[derive(Debug, Default)]
pub struct SubmissionAdmissions {
    active: Vec<(SubmissionIdentity, SubmissionFootprint)>,
}

/// Guest-ordered ownership between concurrent recording and queue acceptance.
///
/// Recording workers may finish in any order.  A result becomes removable only
/// when it is at the head of this unbounded arrival-order ledger, so a later
/// EXEC can never reach a FIFO queue, presentation, or semantic completion
/// before every earlier EXEC has produced its terminal recording result.
#[derive(Debug)]
pub struct SubmissionCommitOrder<T> {
    pending: VecDeque<PendingCommit<T>>,
}

#[derive(Debug)]
struct PendingCommit<T> {
    identity: SubmissionIdentity,
    recorded: Option<T>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionCommitOrderError {
    AlreadyRegistered(SubmissionIdentity),
    UnknownSubmission(SubmissionIdentity),
    AlreadyRecorded(SubmissionIdentity),
}

/// One nonblocking admission decision made at the packet boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubmissionWork<W> {
    pub context: SubmissionContext,
    pub work: W,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubmissionDispatch<W> {
    /// The complete EXEC footprint is disjoint from every active recorder.
    Record(SubmissionWork<W>),
    /// The commit position is reserved, but recording ownership is parked.
    Queued {
        identity: SubmissionIdentity,
        blocked_by: SubmissionIdentity,
        reason: SubmissionConflict,
    },
}

/// A packet identity cannot reserve recording and commit ownership twice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionAcceptError {
    AlreadyAccepted(SubmissionIdentity),
    Commit(SubmissionCommitOrderError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionRecordError {
    NotRecording(SubmissionIdentity),
    Commit(SubmissionCommitOrderError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionCompleteError {
    NotRecording(SubmissionIdentity),
    CommitNotTaken(SubmissionIdentity),
}

/// Queueing policy over disjoint recording and guest-ordered commit ownership.
///
/// This owner never waits. Conflicting EXECs retain their immutable context in
/// arrival order, while every waiter is reconsidered after a recorder retires
/// so an unrelated transaction cannot be trapped behind a conflicting head.
#[derive(Debug)]
pub struct SubmissionScheduler<W, T> {
    waiting: VecDeque<SubmissionWork<W>>,
    admissions: SubmissionAdmissions,
    commits: SubmissionCommitOrder<T>,
}

impl<W, T> Default for SubmissionScheduler<W, T> {
    fn default() -> Self {
        Self {
            waiting: VecDeque::new(),
            admissions: SubmissionAdmissions::default(),
            commits: SubmissionCommitOrder::default(),
        }
    }
}

impl<T> Default for SubmissionCommitOrder<T> {
    fn default() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }
}

impl<T> SubmissionCommitOrder<T> {
    /// Reserve this EXEC's place when its packet is accepted, before recording
    /// is allowed to fan out.
    pub fn register(
        &mut self,
        identity: SubmissionIdentity,
    ) -> Result<(), SubmissionCommitOrderError> {
        if self.pending.iter().any(|entry| entry.identity == identity) {
            return Err(SubmissionCommitOrderError::AlreadyRegistered(identity));
        }
        self.pending.push_back(PendingCommit {
            identity,
            recorded: None,
        });
        Ok(())
    }

    /// Publish one worker's terminal recording result without changing its
    /// guest-order position.
    pub fn record(
        &mut self,
        identity: SubmissionIdentity,
        result: T,
    ) -> Result<(), SubmissionCommitOrderError> {
        let Some(entry) = self
            .pending
            .iter_mut()
            .find(|entry| entry.identity == identity)
        else {
            return Err(SubmissionCommitOrderError::UnknownSubmission(identity));
        };
        if entry.recorded.is_some() {
            return Err(SubmissionCommitOrderError::AlreadyRecorded(identity));
        }
        entry.recorded = Some(result);
        Ok(())
    }

    /// Remove the next queue-acceptable recording, if the oldest EXEC has
    /// finished.  Later finished entries remain parked behind an unfinished
    /// predecessor.
    pub fn take_ready(&mut self) -> Option<(SubmissionIdentity, T)> {
        let ready = self.pending.front()?.recorded.is_some();
        ready.then(|| {
            let entry = self.pending.pop_front().expect("the ready head exists");
            (
                entry.identity,
                entry.recorded.expect("the removed head was recorded"),
            )
        })
    }

    /// Abort every registered EXEC in arrival order after device loss or
    /// session reset.  The caller owns the typed refusal attached to each one.
    pub fn abort_all(&mut self) -> impl Iterator<Item = SubmissionIdentity> + '_ {
        self.pending.drain(..).map(|entry| entry.identity)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn contains(&self, identity: SubmissionIdentity) -> bool {
        self.pending.iter().any(|entry| entry.identity == identity)
    }
}

impl<W, T> SubmissionScheduler<W, T> {
    /// Reserve one guest-order position and acquire recording ownership when
    /// the candidate is independent. A conflict is queued, never waited on.
    pub fn accept(
        &mut self,
        context: SubmissionContext,
        work: W,
    ) -> Result<SubmissionDispatch<W>, SubmissionAcceptError> {
        let identity = context.identity;
        if self.commits.contains(identity) {
            return Err(SubmissionAcceptError::AlreadyAccepted(identity));
        }
        self.commits
            .register(identity)
            .map_err(SubmissionAcceptError::Commit)?;
        match self.admissions.admit(&context) {
            Ok(()) => Ok(SubmissionDispatch::Record(SubmissionWork { context, work })),
            Err(SubmissionAdmissionRefusal::Conflict { active, reason }) => {
                self.waiting.push_back(SubmissionWork { context, work });
                Ok(SubmissionDispatch::Queued {
                    identity,
                    blocked_by: active,
                    reason,
                })
            }
            Err(SubmissionAdmissionRefusal::AlreadyActive(_)) => {
                unreachable!("commit ownership rejected the duplicate identity first")
            }
        }
    }

    /// Publish a worker result without releasing its semantic footprint.
    ///
    /// Recording completion is not semantic completion. A conflicting waiter
    /// must remain parked until the recorded result reaches guest-order commit
    /// and the caller applies its completion facts through [`Self::complete`].
    pub fn record(
        &mut self,
        identity: SubmissionIdentity,
        result: T,
    ) -> Result<(), SubmissionRecordError> {
        if !self.admissions.contains(identity) {
            return Err(SubmissionRecordError::NotRecording(identity));
        }
        self.commits
            .record(identity, result)
            .map_err(SubmissionRecordError::Commit)?;
        Ok(())
    }

    /// Retire one semantically committed footprint and admit newly independent
    /// waiters.
    ///
    /// The caller invokes this only after the matching value returned by
    /// [`Self::take_ready`] has updated content, resource lifetime, and backend
    /// submission state. Keeping that ordering explicit prevents a waiter from
    /// snapshotting the predecessor's pre-commit state.
    pub fn complete(
        &mut self,
        identity: SubmissionIdentity,
    ) -> Result<Vec<SubmissionWork<W>>, SubmissionCompleteError> {
        if self.commits.contains(identity) {
            return Err(SubmissionCompleteError::CommitNotTaken(identity));
        }
        if !self.admissions.retire(identity) {
            return Err(SubmissionCompleteError::NotRecording(identity));
        }
        Ok(self.dispatch_ready())
    }

    /// Admit every waiter now independent of active recordings and of waiters
    /// admitted earlier in this same pass.
    pub fn dispatch_ready(&mut self) -> Vec<SubmissionWork<W>> {
        let mut ready = Vec::new();
        let mut blocked = VecDeque::new();
        while let Some(submission) = self.waiting.pop_front() {
            match self.admissions.admit(&submission.context) {
                Ok(()) => ready.push(submission),
                Err(SubmissionAdmissionRefusal::Conflict { .. }) => blocked.push_back(submission),
                Err(SubmissionAdmissionRefusal::AlreadyActive(identity)) => {
                    unreachable!("a queued identity cannot already own recording: {identity:?}")
                }
            }
        }
        self.waiting = blocked;
        ready
    }

    /// Remove the oldest terminal worker result, if the guest-order head has
    /// finished recording.
    pub fn take_ready(&mut self) -> Option<(SubmissionIdentity, T)> {
        self.commits.take_ready()
    }

    /// Drop every queued and active ownership on reset or device loss.
    /// Returned identities are in guest arrival order and are never successes.
    pub fn abort_all(&mut self) -> impl Iterator<Item = SubmissionIdentity> + '_ {
        self.waiting.clear();
        self.admissions.clear();
        self.commits.abort_all()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn active_len(&self) -> usize {
        self.admissions.len()
    }

    pub fn commit_len(&self) -> usize {
        self.commits.len()
    }
}

impl SubmissionAdmissions {
    pub fn admit(&mut self, context: &SubmissionContext) -> Result<(), SubmissionAdmissionRefusal> {
        if self
            .active
            .iter()
            .any(|(identity, _)| *identity == context.identity)
        {
            return Err(SubmissionAdmissionRefusal::AlreadyActive(context.identity));
        }
        let footprint = SubmissionFootprint::from_context(context);
        if let Some((active, reason)) = self.active.iter().find_map(|(identity, active)| {
            active
                .first_conflict(&footprint)
                .map(|reason| (*identity, reason))
        }) {
            return Err(SubmissionAdmissionRefusal::Conflict { active, reason });
        }
        self.active.push((context.identity, footprint));
        Ok(())
    }

    pub fn retire(&mut self, identity: SubmissionIdentity) -> bool {
        let Some(index) = self
            .active
            .iter()
            .position(|(active, _)| *active == identity)
        else {
            return false;
        };
        self.active.remove(index);
        true
    }

    pub fn len(&self) -> usize {
        self.active.len()
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    fn clear(&mut self) {
        self.active.clear();
    }

    fn contains(&self, identity: SubmissionIdentity) -> bool {
        self.active.iter().any(|(active, _)| *active == identity)
    }
}

impl SubmissionFootprint {
    pub fn from_context(context: &SubmissionContext) -> Self {
        let mut resources = context
            .resources
            .iter()
            .filter_map(|use_| use_.resource)
            .collect::<Vec<_>>();
        resources.sort_unstable();
        resources.dedup();
        Self {
            resources: resources.into_boxed_slice(),
            standalone: context.identity.id.get() == 0,
            unresolved: context.resources.iter().any(|use_| use_.resource.is_none()),
        }
    }

    pub fn resources(&self) -> &[ResourceId<ResourceObject>] {
        &self.resources
    }

    pub fn first_conflict(&self, other: &Self) -> Option<SubmissionConflict> {
        if self.standalone || other.standalone {
            return Some(SubmissionConflict::Standalone);
        }
        if self.unresolved || other.unresolved {
            return Some(SubmissionConflict::UnresolvedParticipant);
        }
        let (mut left, mut right) = (0, 0);
        while left < self.resources.len() && right < other.resources.len() {
            match self.resources[left].cmp(&other.resources[right]) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    return Some(SubmissionConflict::SharedResource(self.resources[left]));
                }
            }
        }
        None
    }
}

/// Device-local source of nonzero submission identities.
///
/// Immutable envelopes are owned by scheduler entries and resolver workers;
/// keeping an active envelope here would make concurrent resolution
/// unrepresentable by giving the whole device one mutable segment cursor.
#[derive(Debug)]
pub struct SubmissionTracker {
    next_id: u64,
}

impl Default for SubmissionTracker {
    fn default() -> Self {
        Self { next_id: 1 }
    }
}

impl SubmissionTracker {
    /// Mint the next nonzero identity for `task`.
    pub fn next_identity(&mut self, task: TaskId) -> SubmissionIdentity {
        let identity = SubmissionIdentity {
            id: SubmissionId::new(self.next_id),
            task,
        };
        self.next_id = self.next_id.wrapping_add(1).max(1);
        identity
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SubmissionAcceptError, SubmissionAdmissionRefusal, SubmissionAdmissions,
        SubmissionCommitOrder, SubmissionCommitOrderError, SubmissionCompleteError,
        SubmissionConflict, SubmissionContext, SubmissionDispatch, SubmissionFootprint,
        SubmissionRecordError, SubmissionScheduler, SubmissionTracker, SubmissionWork,
    };
    use reims_vgpu_protocol::{
        ObjectTableRef, ResourceId, ResourceObject, ResourceValidity, SubmissionId,
        SubmissionIdentity, SubmissionResourceUse, TaskId,
    };

    fn context_with_resources(
        id: u64,
        resources: impl IntoIterator<Item = Option<ResourceId<ResourceObject>>>,
    ) -> SubmissionContext {
        SubmissionContext {
            identity: SubmissionIdentity {
                id: SubmissionId::new(id),
                task: TaskId::new(7),
            },
            resources: resources
                .into_iter()
                .enumerate()
                .map(|(object, resource)| SubmissionResourceUse {
                    object: ObjectTableRef::new(object as u32),
                    resource,
                    expected_content: None,
                    validity: ResourceValidity::default(),
                })
                .collect::<Vec<_>>()
                .into(),
            segments: std::sync::Arc::from([]),
            segment: None,
        }
    }

    #[test]
    fn recording_finish_order_cannot_overtake_guest_commit_order() {
        let first = context_with_resources(40, []).identity;
        let second = context_with_resources(41, []).identity;
        let mut order = SubmissionCommitOrder::default();
        order.register(first).unwrap();
        order.register(second).unwrap();

        order.record(second, "second").unwrap();
        assert_eq!(order.take_ready(), None);
        order.record(first, "first").unwrap();
        assert_eq!(order.take_ready(), Some((first, "first")));
        assert_eq!(order.take_ready(), Some((second, "second")));
        assert!(order.is_empty());
    }

    #[test]
    fn commit_positions_and_terminal_results_are_single_owner() {
        let identity = context_with_resources(42, []).identity;
        let unknown = context_with_resources(43, []).identity;
        let mut order = SubmissionCommitOrder::default();
        order.register(identity).unwrap();
        assert_eq!(
            order.register(identity),
            Err(SubmissionCommitOrderError::AlreadyRegistered(identity))
        );
        assert_eq!(
            order.record(unknown, 1),
            Err(SubmissionCommitOrderError::UnknownSubmission(unknown))
        );
        order.record(identity, 2).unwrap();
        assert_eq!(
            order.record(identity, 3),
            Err(SubmissionCommitOrderError::AlreadyRecorded(identity))
        );
    }

    #[test]
    fn device_loss_aborts_every_registered_position_in_guest_order() {
        let identities = [44, 45, 46].map(|id| context_with_resources(id, []).identity);
        let mut order = SubmissionCommitOrder::<()>::default();
        for identity in identities {
            order.register(identity).unwrap();
        }
        order.record(identities[1], ()).unwrap();

        assert_eq!(order.abort_all().collect::<Vec<_>>(), identities);
        assert!(order.is_empty());
    }

    #[test]
    fn a_conflicting_waiter_does_not_block_a_later_independent_exec() {
        let shared = ResourceId::new(3, 1);
        let first = context_with_resources(50, [Some(shared)]);
        let blocked = context_with_resources(51, [Some(shared)]);
        let independent = context_with_resources(52, [Some(ResourceId::new(4, 1))]);
        let mut scheduler = SubmissionScheduler::<(), &'static str>::default();

        assert_eq!(
            scheduler.accept(first.clone(), ()),
            Ok(SubmissionDispatch::Record(SubmissionWork {
                context: first.clone(),
                work: (),
            }))
        );
        assert_eq!(
            scheduler.accept(blocked.clone(), ()),
            Ok(SubmissionDispatch::Queued {
                identity: blocked.identity,
                blocked_by: first.identity,
                reason: SubmissionConflict::SharedResource(shared),
            })
        );
        assert_eq!(
            scheduler.accept(independent.clone(), ()),
            Ok(SubmissionDispatch::Record(SubmissionWork {
                context: independent,
                work: (),
            }))
        );
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(scheduler.waiting_len(), 1);

        scheduler.record(first.identity, "first").unwrap();
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.take_ready(), Some((first.identity, "first")));
        assert_eq!(
            scheduler.complete(first.identity).unwrap(),
            vec![SubmissionWork {
                context: blocked,
                work: (),
            }]
        );
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(scheduler.waiting_len(), 0);
    }

    #[test]
    fn newly_admitted_waiters_conflict_with_each_other_in_the_same_dispatch() {
        let shared = ResourceId::new(5, 1);
        let first = context_with_resources(53, [Some(shared)]);
        let second = context_with_resources(54, [Some(shared)]);
        let third = context_with_resources(55, [Some(shared)]);
        let mut scheduler = SubmissionScheduler::<(), ()>::default();

        assert!(matches!(
            scheduler.accept(first.clone(), ()),
            Ok(SubmissionDispatch::Record(_))
        ));
        assert!(matches!(
            scheduler.accept(second.clone(), ()),
            Ok(SubmissionDispatch::Queued { .. })
        ));
        assert!(matches!(
            scheduler.accept(third, ()),
            Ok(SubmissionDispatch::Queued { .. })
        ));

        scheduler.record(first.identity, ()).unwrap();
        assert_eq!(scheduler.waiting_len(), 2);
        assert_eq!(scheduler.take_ready(), Some((first.identity, ())));
        assert_eq!(
            scheduler.complete(first.identity).unwrap(),
            vec![SubmissionWork {
                context: second,
                work: (),
            }]
        );
        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(scheduler.waiting_len(), 1);
    }

    #[test]
    fn recording_retirement_and_guest_order_commit_are_independent() {
        let first = context_with_resources(56, [Some(ResourceId::new(6, 1))]);
        let second = context_with_resources(57, [Some(ResourceId::new(7, 1))]);
        let mut scheduler = SubmissionScheduler::default();
        scheduler.accept(first.clone(), ()).unwrap();
        scheduler.accept(second.clone(), ()).unwrap();

        scheduler.record(second.identity, "second").unwrap();
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(scheduler.take_ready(), None);
        scheduler.record(first.identity, "first").unwrap();
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(scheduler.take_ready(), Some((first.identity, "first")));
        assert!(scheduler.complete(first.identity).unwrap().is_empty());
        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(scheduler.take_ready(), Some((second.identity, "second")));
        assert!(scheduler.complete(second.identity).unwrap().is_empty());
        assert_eq!(scheduler.active_len(), 0);
    }

    #[test]
    fn a_recorded_result_cannot_retire_before_the_commit_owner_takes_it() {
        let context = context_with_resources(63, [Some(ResourceId::new(11, 1))]);
        let mut scheduler = SubmissionScheduler::<(), ()>::default();
        scheduler.accept(context.clone(), ()).unwrap();
        scheduler.record(context.identity, ()).unwrap();

        assert_eq!(
            scheduler.complete(context.identity),
            Err(SubmissionCompleteError::CommitNotTaken(context.identity))
        );
        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(scheduler.take_ready(), Some((context.identity, ())));
        assert!(scheduler.complete(context.identity).unwrap().is_empty());
    }

    #[test]
    fn a_finished_successor_keeps_its_conflicting_waiter_parked_until_commit() {
        let first = context_with_resources(60, [Some(ResourceId::new(9, 1))]);
        let successor = context_with_resources(61, [Some(ResourceId::new(10, 1))]);
        let waiter = context_with_resources(62, [Some(ResourceId::new(10, 1))]);
        let mut scheduler = SubmissionScheduler::<(), &'static str>::default();
        scheduler.accept(first.clone(), ()).unwrap();
        scheduler.accept(successor.clone(), ()).unwrap();
        assert!(matches!(
            scheduler.accept(waiter.clone(), ()),
            Ok(SubmissionDispatch::Queued { .. })
        ));

        scheduler.record(successor.identity, "successor").unwrap();
        assert_eq!(scheduler.take_ready(), None);
        assert_eq!(scheduler.waiting_len(), 1);

        scheduler.record(first.identity, "first").unwrap();
        assert_eq!(scheduler.take_ready(), Some((first.identity, "first")));
        assert!(scheduler.complete(first.identity).unwrap().is_empty());
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(
            scheduler.take_ready(),
            Some((successor.identity, "successor"))
        );
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(
            scheduler.complete(successor.identity).unwrap(),
            vec![SubmissionWork {
                context: waiter,
                work: (),
            }]
        );
    }

    #[test]
    fn duplicate_accept_is_typed_and_reset_aborts_in_guest_order() {
        let first = context_with_resources(58, [Some(ResourceId::new(8, 1))]);
        let second = context_with_resources(59, [Some(ResourceId::new(8, 1))]);
        let mut scheduler = SubmissionScheduler::<(), ()>::default();
        scheduler.accept(first.clone(), ()).unwrap();
        assert_eq!(
            scheduler.accept(first.clone(), ()),
            Err(SubmissionAcceptError::AlreadyAccepted(first.identity))
        );
        scheduler.accept(second.clone(), ()).unwrap();
        assert_eq!(
            scheduler.record(second.identity, ()),
            Err(SubmissionRecordError::NotRecording(second.identity))
        );

        assert_eq!(
            scheduler.abort_all().collect::<Vec<_>>(),
            vec![first.identity, second.identity]
        );
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(scheduler.waiting_len(), 0);
        assert_eq!(scheduler.commit_len(), 0);
    }

    #[test]
    fn standalone_context_has_no_invented_participation() {
        let context = SubmissionContext::standalone(7);
        assert_eq!(context.identity.task.get(), 7);
        assert_eq!(context.identity.id.get(), 0);
        assert!(context.resources.is_empty());
        assert!(context.segments.is_empty());
        assert_eq!(context.segment, None);
    }

    #[test]
    fn disjoint_resolved_execs_may_overlap() {
        let first = SubmissionFootprint::from_context(&context_with_resources(
            1,
            [Some(ResourceId::new(3, 1)), Some(ResourceId::new(4, 1))],
        ));
        let second = SubmissionFootprint::from_context(&context_with_resources(
            2,
            [Some(ResourceId::new(5, 1))],
        ));
        assert_eq!(first.first_conflict(&second), None);
    }

    #[test]
    fn a_shared_live_resource_serializes_execs() {
        let resource = ResourceId::new(3, 4);
        let first = SubmissionFootprint::from_context(&context_with_resources(1, [Some(resource)]));
        let second =
            SubmissionFootprint::from_context(&context_with_resources(2, [Some(resource)]));
        assert_eq!(
            first.first_conflict(&second),
            Some(SubmissionConflict::SharedResource(resource))
        );
    }

    #[test]
    fn stale_slot_reuse_is_not_a_resource_conflict() {
        let first = SubmissionFootprint::from_context(&context_with_resources(
            1,
            [Some(ResourceId::new(3, 4))],
        ));
        let second = SubmissionFootprint::from_context(&context_with_resources(
            2,
            [Some(ResourceId::new(3, 5))],
        ));
        assert_eq!(first.first_conflict(&second), None);
    }

    #[test]
    fn unresolved_and_standalone_work_fail_closed() {
        let resolved = SubmissionFootprint::from_context(&context_with_resources(
            1,
            [Some(ResourceId::new(3, 4))],
        ));
        let unresolved = SubmissionFootprint::from_context(&context_with_resources(
            2,
            [Some(ResourceId::new(8, 1)), None],
        ));
        assert_eq!(
            resolved.first_conflict(&unresolved),
            Some(SubmissionConflict::UnresolvedParticipant)
        );
        let standalone = SubmissionFootprint::from_context(&SubmissionContext::standalone(7));
        assert_eq!(
            resolved.first_conflict(&standalone),
            Some(SubmissionConflict::Standalone)
        );
    }

    #[test]
    fn admission_owns_disjoint_execs_and_releases_only_the_exact_identity() {
        let first = context_with_resources(1, [Some(ResourceId::new(3, 4))]);
        let disjoint = context_with_resources(2, [Some(ResourceId::new(8, 1))]);
        let blocked = context_with_resources(3, [Some(ResourceId::new(3, 4))]);
        let mut admissions = SubmissionAdmissions::default();

        admissions.admit(&first).unwrap();
        admissions.admit(&disjoint).unwrap();
        assert_eq!(admissions.len(), 2);
        assert_eq!(
            admissions.admit(&blocked),
            Err(SubmissionAdmissionRefusal::Conflict {
                active: first.identity,
                reason: SubmissionConflict::SharedResource(ResourceId::new(3, 4)),
            })
        );
        assert!(!admissions.retire(blocked.identity));
        assert!(admissions.retire(first.identity));
        admissions.admit(&blocked).unwrap();
    }

    #[test]
    fn duplicate_admission_is_a_typed_inconsistency() {
        let context = context_with_resources(7, [Some(ResourceId::new(3, 4))]);
        let mut admissions = SubmissionAdmissions::default();
        admissions.admit(&context).unwrap();
        assert_eq!(
            admissions.admit(&context),
            Err(SubmissionAdmissionRefusal::AlreadyActive(context.identity))
        );
    }

    #[test]
    fn tracker_mints_nonzero_identities_per_device() {
        let mut tracker = SubmissionTracker::default();
        let first = tracker.next_identity(TaskId::new(7));
        let second = tracker.next_identity(TaskId::new(7));
        assert_ne!(first.id, second.id);
        assert_ne!(first.id.get(), 0);
        assert_eq!(second.id.get(), first.id.get() + 1);
    }
}
