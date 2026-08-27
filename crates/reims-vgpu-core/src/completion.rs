//! Immutable terminal facts and queue-timeline completion ownership.
//!
//! Queue acceptance assigns a concrete [`QueueTimelinePoint`]. Recording and
//! driver return are not completion. The owner below transfers a pending
//! semantic result into an immutable fact only after the corresponding queue
//! timeline reaches that point. Device loss abandons pending work explicitly;
//! it never manufactures successful completion.

use reims_vgpu_protocol::{
    QueueOwnerId, QueueTimelineValue, SessionGenerationId, TransactionId, VulkanDeviceEpochId,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct QueueTimelinePoint {
    pub epoch: VulkanDeviceEpochId,
    pub queue: QueueOwnerId,
    pub value: QueueTimelineValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionEvidence {
    Gpu(QueueTimelinePoint),
    Effect,
    Present,
}

/// A completed semantic result. This type contains identities and values, not
/// backend handles, and remains valid after the native epoch is lost.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionFact<T> {
    pub transaction: TransactionId,
    pub session_generation: SessionGenerationId,
    pub evidence: CompletionEvidence,
    pub semantic: T,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AbandonedCompletion<T> {
    pub transaction: TransactionId,
    pub session_generation: SessionGenerationId,
    pub point: QueueTimelinePoint,
    pub semantic: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionOwnerError {
    WrongEpoch,
    DuplicateTransaction,
    /// A submission asked to register a point at or below the highest already
    /// submitted on its queue.
    ///
    /// The point is allocated when a submission is prepared and registered
    /// when the driver accepts it, so anything accepted between those two
    /// moments takes the higher value and leaves this one behind. The values
    /// are deliberately *not* carried here: this enum is `Copy` and sits
    /// inside `ReplayAcceptanceFailure`, which already holds a whole prepared
    /// submission, and widening it pushes that failure past the size clippy
    /// refuses. Ask [`TimelineCompletionOwner::last_submitted`] at the
    /// reporting site instead, which has the point in hand.
    TimelineDidNotIncrease,
    TimelineAlreadyCompleted,
    TimelineRegressed,
    TransactionStillPending,
}

#[derive(Clone, Debug)]
struct Pending<T> {
    transaction: TransactionId,
    session_generation: SessionGenerationId,
    semantic: T,
}

#[derive(Clone, Debug)]
struct QueueCompletions<T> {
    last_submitted: Option<QueueTimelineValue>,
    last_completed: Option<QueueTimelineValue>,
    pending: BTreeMap<QueueTimelineValue, Vec<Pending<T>>>,
}

impl<T> Default for QueueCompletions<T> {
    fn default() -> Self {
        Self {
            last_submitted: None,
            last_completed: None,
            pending: BTreeMap::new(),
        }
    }
}

/// Sole owner of completion progress for one Vulkan device epoch.
#[derive(Clone, Debug)]
pub struct TimelineCompletionOwner<T> {
    epoch: VulkanDeviceEpochId,
    queues: BTreeMap<QueueOwnerId, QueueCompletions<T>>,
    transactions: BTreeMap<TransactionId, QueueTimelinePoint>,
}

impl<T> TimelineCompletionOwner<T> {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            epoch,
            queues: BTreeMap::new(),
            transactions: BTreeMap::new(),
        }
    }

    /// The highest point already submitted on `queue`, if any.
    ///
    /// A `TimelineDidNotIncrease` refusal says a point was at or below this
    /// without saying by how much, and the gap is the reading: adjacent values
    /// mean two submissions raced, a whole queue apart means an acceptance ran
    /// far out of the order its points were allocated in.
    #[must_use]
    pub fn last_submitted(&self, queue: QueueOwnerId) -> Option<QueueTimelineValue> {
        self.queues
            .get(&queue)
            .and_then(|queue| queue.last_submitted)
    }

    pub fn register(
        &mut self,
        transaction: TransactionId,
        session_generation: SessionGenerationId,
        point: QueueTimelinePoint,
        semantic: T,
    ) -> Result<(), CompletionOwnerError> {
        if point.epoch != self.epoch {
            return Err(CompletionOwnerError::WrongEpoch);
        }
        if self.transactions.contains_key(&transaction) {
            return Err(CompletionOwnerError::DuplicateTransaction);
        }
        let queue = self.queues.entry(point.queue).or_default();
        if queue
            .last_completed
            .is_some_and(|completed| point.value <= completed)
        {
            return Err(CompletionOwnerError::TimelineAlreadyCompleted);
        }
        match queue.last_submitted {
            Some(last) if point.value <= last => {
                return Err(CompletionOwnerError::TimelineDidNotIncrease);
            }
            _ => queue.last_submitted = Some(point.value),
        }
        queue.pending.entry(point.value).or_default().push(Pending {
            transaction,
            session_generation,
            semantic,
        });
        self.transactions.insert(transaction, point);
        Ok(())
    }

    pub fn advance(
        &mut self,
        queue_id: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Vec<CompletionFact<T>>, CompletionOwnerError> {
        self.validate_advance(queue_id, completed)?;
        let queue = self.queues.entry(queue_id).or_default();
        queue.last_completed = Some(completed);

        let reached = queue
            .pending
            .range(..=completed)
            .map(|(point, _)| *point)
            .collect::<Vec<_>>();
        let mut ready = reached
            .into_iter()
            .flat_map(|point| queue.pending.remove(&point).unwrap())
            .collect::<Vec<_>>();
        ready.sort_by_key(|pending| self.transactions[&pending.transaction].value);
        Ok(ready
            .into_iter()
            .map(|pending| {
                let point = self.transactions.remove(&pending.transaction).unwrap();
                CompletionFact {
                    transaction: pending.transaction,
                    session_generation: pending.session_generation,
                    evidence: CompletionEvidence::Gpu(point),
                    semantic: pending.semantic,
                }
            })
            .collect())
    }

    pub fn validate_advance(
        &self,
        queue_id: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<(), CompletionOwnerError> {
        if self
            .queues
            .get(&queue_id)
            .and_then(|queue| queue.last_completed)
            .is_some_and(|last| completed < last)
        {
            return Err(CompletionOwnerError::TimelineRegressed);
        }
        Ok(())
    }

    pub fn validate_retired(&self, transaction: TransactionId) -> Result<(), CompletionOwnerError> {
        if self.transactions.contains_key(&transaction) {
            return Err(CompletionOwnerError::TransactionStillPending);
        }
        Ok(())
    }

    /// Drain pending native work as abandoned during device loss. These are
    /// not successful completion facts and cannot enter semantic publication.
    pub fn abandon(self) -> Vec<AbandonedCompletion<T>> {
        let points = self.transactions;
        let mut abandoned = self
            .queues
            .into_values()
            .flat_map(|queue| queue.pending.into_values().flatten())
            .map(|pending| AbandonedCompletion {
                point: points[&pending.transaction],
                transaction: pending.transaction,
                session_generation: pending.session_generation,
                semantic: pending.semantic,
            })
            .collect::<Vec<_>>();
        abandoned.sort_by_key(|fact| (fact.point.queue, fact.point.value));
        abandoned
    }

    pub fn pending(&self) -> usize {
        self.transactions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(epoch: u64, queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(epoch),
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    #[test]
    fn driver_return_does_not_complete_work_before_timeline_progress() {
        let mut owner = TimelineCompletionOwner::new(VulkanDeviceEpochId::new(4));
        owner
            .register(
                TransactionId::new(1),
                SessionGenerationId::new(7),
                point(4, 0, 3),
                "draw",
            )
            .unwrap();
        assert_eq!(owner.pending(), 1);
        assert!(owner
            .advance(QueueOwnerId::new(0), QueueTimelineValue::new(2))
            .unwrap()
            .is_empty());
        let facts = owner
            .advance(QueueOwnerId::new(0), QueueTimelineValue::new(3))
            .unwrap();
        assert_eq!(facts[0].semantic, "draw");
        assert_eq!(facts[0].evidence, CompletionEvidence::Gpu(point(4, 0, 3)));
    }

    #[test]
    fn queue_progress_is_independent_and_releases_every_reached_point() {
        let mut owner = TimelineCompletionOwner::new(VulkanDeviceEpochId::new(1));
        for (id, queue, value) in [(1, 0, 2), (2, 0, 5), (3, 1, 1)] {
            owner
                .register(
                    TransactionId::new(id),
                    SessionGenerationId::new(1),
                    point(1, queue, value),
                    id,
                )
                .unwrap();
        }
        let queue_one = owner
            .advance(QueueOwnerId::new(1), QueueTimelineValue::new(1))
            .unwrap();
        assert_eq!(queue_one[0].transaction, TransactionId::new(3));
        let queue_zero = owner
            .advance(QueueOwnerId::new(0), QueueTimelineValue::new(5))
            .unwrap();
        assert_eq!(
            queue_zero
                .iter()
                .map(|fact| fact.transaction)
                .collect::<Vec<_>>(),
            vec![TransactionId::new(1), TransactionId::new(2)]
        );
    }

    #[test]
    fn device_loss_abandons_pending_work_without_success_facts() {
        let mut owner = TimelineCompletionOwner::new(VulkanDeviceEpochId::new(8));
        owner
            .register(
                TransactionId::new(9),
                SessionGenerationId::new(3),
                point(8, 0, 11),
                "uncommitted semantic result",
            )
            .unwrap();
        let abandoned = owner.abandon();
        assert_eq!(abandoned.len(), 1);
        assert_eq!(abandoned[0].transaction, TransactionId::new(9));
    }

    #[test]
    fn completion_points_are_monotonic_per_queue_and_bound_to_one_epoch() {
        let mut owner = TimelineCompletionOwner::new(VulkanDeviceEpochId::new(2));
        owner
            .register(
                TransactionId::new(1),
                SessionGenerationId::new(1),
                point(2, 4, 10),
                (),
            )
            .unwrap();
        assert_eq!(
            owner.register(
                TransactionId::new(2),
                SessionGenerationId::new(1),
                point(2, 4, 10),
                (),
            ),
            Err(CompletionOwnerError::TimelineDidNotIncrease)
        );
        assert_eq!(
            owner.register(
                TransactionId::new(3),
                SessionGenerationId::new(1),
                point(3, 5, 1),
                (),
            ),
            Err(CompletionOwnerError::WrongEpoch)
        );
    }

    /// The refusal says a point did not increase; the accessor says by how
    /// much, which is what separates two submissions racing from an
    /// acceptance running far out of its allocation order.
    #[test]
    fn the_highest_submitted_point_is_readable_beside_the_refusal() {
        let mut owner = TimelineCompletionOwner::new(VulkanDeviceEpochId::new(2));
        assert_eq!(owner.last_submitted(QueueOwnerId::new(4)), None);
        owner
            .register(
                TransactionId::new(1),
                SessionGenerationId::new(1),
                point(2, 4, 10),
                (),
            )
            .unwrap();
        assert_eq!(
            owner.last_submitted(QueueOwnerId::new(4)),
            Some(QueueTimelineValue::new(10))
        );
        // A refused registration must not move it: the gap a later report
        // reads has to describe the queue, not the last thing that asked.
        assert_eq!(
            owner.register(
                TransactionId::new(2),
                SessionGenerationId::new(1),
                point(2, 4, 3),
                (),
            ),
            Err(CompletionOwnerError::TimelineDidNotIncrease)
        );
        assert_eq!(
            owner.last_submitted(QueueOwnerId::new(4)),
            Some(QueueTimelineValue::new(10))
        );
        assert_eq!(owner.last_submitted(QueueOwnerId::new(9)), None);
    }

    #[test]
    fn work_cannot_be_registered_behind_observed_completion() {
        let mut owner = TimelineCompletionOwner::new(VulkanDeviceEpochId::new(2));
        owner
            .advance(QueueOwnerId::new(4), QueueTimelineValue::new(10))
            .unwrap();
        assert_eq!(
            owner.register(
                TransactionId::new(1),
                SessionGenerationId::new(1),
                point(2, 4, 9),
                (),
            ),
            Err(CompletionOwnerError::TimelineAlreadyCompleted)
        );
    }
}
