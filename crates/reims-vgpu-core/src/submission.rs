//! Backend-independent submission envelopes retained by device state.

use reims_vgpu_protocol::{
    ResourceId, ResourceObject, SegmentBoundary, SubmissionId, SubmissionIdentity,
    SubmissionResourceUse, TaskId,
};
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
    /// Every admitted segment in command-buffer order.
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

/// Device-local ownership of the currently decoded submission envelope.
///
/// Callers can obtain immutable [`SubmissionContext`] snapshots, but cannot
/// mutate participation or segment position independently of submission
/// identity. Reset drops this owner and therefore its active envelope.
#[derive(Debug)]
pub struct SubmissionTracker {
    next_id: u64,
    active: Option<SubmissionContext>,
}

impl Default for SubmissionTracker {
    fn default() -> Self {
        Self {
            next_id: 1,
            active: None,
        }
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

    /// Install one complete participation envelope before its first segment.
    pub fn begin(
        &mut self,
        identity: SubmissionIdentity,
        resources: Arc<[SubmissionResourceUse]>,
        segments: Arc<[SegmentBoundary]>,
    ) {
        assert!(
            self.active.is_none(),
            "a submission cannot begin while another remains active"
        );
        self.active = Some(SubmissionContext {
            identity,
            resources,
            segments,
            segment: None,
        });
    }

    /// Select the active submission segment, when this operation belongs to an
    /// EXEC envelope. Direct tools and focused walkers intentionally have no
    /// active submission and continue to use standalone executor context.
    pub fn enter_segment_if_active(&mut self, segment: Option<SegmentBoundary>) {
        if let Some(active) = self.active.as_mut() {
            active.segment = segment;
        }
    }

    /// Immutable executor snapshot, or a standalone context for direct tools.
    pub fn context_or_standalone(&self, task_id: u32) -> SubmissionContext {
        self.active
            .clone()
            .unwrap_or_else(|| SubmissionContext::standalone(task_id))
    }

    /// Consume the active envelope at its single completion boundary.
    pub fn finish(&mut self) -> Option<SubmissionContext> {
        self.active.take()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SubmissionAdmissionRefusal, SubmissionAdmissions, SubmissionConflict, SubmissionContext,
        SubmissionFootprint, SubmissionTracker,
    };
    use reims_vgpu_protocol::{
        ObjectTableRef, ResourceId, ResourceObject, ResourceValidity, SegmentBoundary, SegmentKind,
        SubmissionId, SubmissionIdentity, SubmissionResourceUse, TaskId,
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
    fn tracker_owns_identity_envelope_segment_and_completion_together() {
        let mut tracker = SubmissionTracker::default();
        let first = tracker.next_identity(TaskId::new(7));
        let second = tracker.next_identity(TaskId::new(7));
        assert_ne!(first.id, second.id);
        assert_ne!(first.id.get(), 0);

        let segment = SegmentBoundary {
            stream_index: 2,
            index: 3,
            kind: SegmentKind::Render,
            continues_previous: false,
            continues_next: true,
        };
        tracker.begin(
            first,
            std::sync::Arc::from([]),
            std::sync::Arc::from([segment]),
        );
        tracker.enter_segment_if_active(Some(segment));
        let snapshot = tracker.context_or_standalone(99);
        assert_eq!(snapshot.identity, first);
        assert_eq!(snapshot.segment, Some(segment));
        assert_eq!(snapshot.segments.as_ref(), &[segment]);

        let finished = tracker
            .finish()
            .expect("the active envelope completes once");
        assert_eq!(finished.identity, first);
        assert!(tracker.finish().is_none());
        assert_eq!(tracker.context_or_standalone(99).identity.id.get(), 0);
    }
}
