//! Deferred native destruction from exact queue-timeline obligations.

use crate::QueueTimelinePoint;
use reims_vgpu_protocol::{QueueOwnerId, QueueTimelineValue, VulkanDeviceEpochId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeRetirementError {
    MixedEpochs,
    DuplicateTicket,
    TimelineRegressed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRetirementFailure<T> {
    pub reason: NativeRetirementError,
    pub value: T,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeRetirementDisposition<T> {
    Deferred,
    Ready(T),
}

#[derive(Clone, Debug)]
struct Deferred<T> {
    value: T,
    obligations: BTreeSet<QueueTimelinePoint>,
}

#[derive(Clone, Debug)]
pub struct NativeRetirement<K, T> {
    epoch: VulkanDeviceEpochId,
    completed: BTreeMap<QueueOwnerId, QueueTimelineValue>,
    deferred: BTreeMap<K, Deferred<T>>,
    waiting: BTreeMap<QueueOwnerId, BTreeMap<QueueTimelineValue, BTreeSet<K>>>,
}

impl<K: Clone + Ord, T> NativeRetirement<K, T> {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            epoch,
            completed: BTreeMap::new(),
            deferred: BTreeMap::new(),
            waiting: BTreeMap::new(),
        }
    }

    pub fn defer(
        &mut self,
        key: K,
        value: T,
        obligations: impl IntoIterator<Item = QueueTimelinePoint>,
    ) -> Result<NativeRetirementDisposition<T>, NativeRetirementFailure<T>> {
        let obligations = obligations.into_iter().collect::<BTreeSet<_>>();
        if let Err(reason) = self.validate_defer(&key, &obligations) {
            return Err(NativeRetirementFailure { reason, value });
        }
        let obligations = obligations
            .into_iter()
            .filter(|point| {
                self.completed
                    .get(&point.queue)
                    .is_none_or(|completed| *completed < point.value)
            })
            .collect::<BTreeSet<_>>();
        if obligations.is_empty() {
            return Ok(NativeRetirementDisposition::Ready(value));
        }
        for point in &obligations {
            self.waiting
                .entry(point.queue)
                .or_default()
                .entry(point.value)
                .or_default()
                .insert(key.clone());
        }
        self.deferred.insert(key, Deferred { value, obligations });
        Ok(NativeRetirementDisposition::Deferred)
    }

    pub fn validate_defer(
        &self,
        key: &K,
        obligations: &BTreeSet<QueueTimelinePoint>,
    ) -> Result<(), NativeRetirementError> {
        if self.deferred.contains_key(key) {
            return Err(NativeRetirementError::DuplicateTicket);
        }
        if obligations.iter().any(|point| point.epoch != self.epoch) {
            return Err(NativeRetirementError::MixedEpochs);
        }
        Ok(())
    }

    pub fn advance(
        &mut self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Vec<(K, T)>, NativeRetirementError> {
        self.validate_advance(queue, completed)?;
        self.completed.insert(queue, completed);
        let reached_values = self
            .waiting
            .get(&queue)
            .into_iter()
            .flat_map(|waiting| waiting.range(..=completed).map(|(value, _)| *value))
            .collect::<Vec<_>>();
        let mut reached = BTreeSet::new();
        for value in reached_values {
            reached.extend(
                self.waiting
                    .get_mut(&queue)
                    .unwrap()
                    .remove(&value)
                    .unwrap(),
            );
        }
        let mut ready = Vec::new();
        for key in reached {
            let deferred = self.deferred.get_mut(&key).unwrap();
            deferred
                .obligations
                .retain(|point| point.queue != queue || point.value > completed);
            if deferred.obligations.is_empty() {
                ready.push(key);
            }
        }
        Ok(ready
            .into_iter()
            .map(|key| {
                let deferred = self.deferred.remove(&key).unwrap();
                (key, deferred.value)
            })
            .collect())
    }

    pub fn validate_advance(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<(), NativeRetirementError> {
        if self
            .completed
            .get(&queue)
            .is_some_and(|previous| completed < *previous)
        {
            return Err(NativeRetirementError::TimelineRegressed);
        }
        Ok(())
    }

    /// Borrow every value whose complete obligation set would be satisfied by
    /// this observation. This lets a caller validate effects carried by the
    /// retiring values before advancing any joined owner.
    pub fn values_ready_after(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Vec<&T>, NativeRetirementError> {
        self.validate_advance(queue, completed)?;
        let mut ready = self
            .deferred
            .iter()
            .filter(|(_, deferred)| {
                deferred.obligations.iter().all(|point| {
                    if point.queue == queue {
                        point.value <= completed
                    } else {
                        self.completed
                            .get(&point.queue)
                            .is_some_and(|value| *value >= point.value)
                    }
                })
            })
            .collect::<Vec<_>>();
        // A single observation can retire semantic and auxiliary recordings
        // together. Their keys name ownership class, not execution order; all
        // completion effects must instead follow their actual point on this
        // queue. The BTreeMap iteration remains the deterministic tie-breaker
        // for values attached to the same point.
        ready.sort_by_key(|(_, deferred)| {
            deferred
                .obligations
                .iter()
                .filter(|point| point.queue == queue)
                .map(|point| point.value)
                .max()
        });
        Ok(ready
            .into_iter()
            .map(|(_, deferred)| &deferred.value)
            .collect())
    }

    pub fn obligations_are_complete(&self, obligations: &BTreeSet<QueueTimelinePoint>) -> bool {
        obligations.iter().all(|point| {
            self.completed
                .get(&point.queue)
                .is_some_and(|completed| *completed >= point.value)
        })
    }

    /// Device loss transfers every still-owned object to finite-return
    /// abandonment cleanup; it does not describe timeline completion.
    pub fn abandon(self) -> Vec<(K, T)> {
        self.deferred
            .into_iter()
            .map(|(key, deferred)| (key, deferred.value))
            .collect()
    }

    pub fn pending(&self) -> usize {
        self.deferred.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(1),
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    #[test]
    fn destruction_waits_for_every_exact_queue_use() {
        let mut owner = NativeRetirement::new(VulkanDeviceEpochId::new(1));
        assert_eq!(
            owner.defer(7, "image", [point(0, 3), point(1, 5)]),
            Ok(NativeRetirementDisposition::Deferred)
        );
        assert!(owner
            .advance(QueueOwnerId::new(0), QueueTimelineValue::new(3))
            .unwrap()
            .is_empty());
        assert!(owner
            .advance(QueueOwnerId::new(1), QueueTimelineValue::new(4))
            .unwrap()
            .is_empty());
        assert_eq!(
            owner
                .advance(QueueOwnerId::new(1), QueueTimelineValue::new(5))
                .unwrap(),
            vec![(7, "image")]
        );
    }

    #[test]
    fn already_reached_points_are_not_live_obligations() {
        let mut owner = NativeRetirement::<u64, &str>::new(VulkanDeviceEpochId::new(1));
        owner
            .advance(QueueOwnerId::new(0), QueueTimelineValue::new(4))
            .unwrap();
        assert_eq!(
            owner.defer(1, "buffer", [point(0, 4)]),
            Ok(NativeRetirementDisposition::Ready("buffer"))
        );
    }

    #[test]
    fn ready_values_are_borrowed_without_consuming_progress() {
        let mut owner = NativeRetirement::new(VulkanDeviceEpochId::new(1));
        owner
            .defer(1, "first", [point(0, 2)])
            .expect("first value is deferred");
        owner
            .defer(2, "second", [point(0, 3)])
            .expect("second value is deferred");
        assert_eq!(
            owner
                .values_ready_after(QueueOwnerId::new(0), QueueTimelineValue::new(2))
                .unwrap(),
            vec![&"first"]
        );
        assert_eq!(owner.pending(), 2);
        assert_eq!(
            owner
                .advance(QueueOwnerId::new(0), QueueTimelineValue::new(2))
                .unwrap(),
            vec![(1, "first")]
        );
    }

    #[test]
    fn ready_values_follow_queue_points_when_key_order_disagrees() {
        let mut owner = NativeRetirement::new(VulkanDeviceEpochId::new(1));
        owner.defer(1, "later", [point(0, 3)]).unwrap();
        owner.defer(2, "earlier", [point(0, 2)]).unwrap();

        assert_eq!(
            owner
                .values_ready_after(QueueOwnerId::new(0), QueueTimelineValue::new(3))
                .unwrap(),
            vec![&"earlier", &"later"]
        );
    }

    #[test]
    fn device_loss_abandonment_is_separate_from_completion() {
        let mut owner = NativeRetirement::new(VulkanDeviceEpochId::new(1));
        owner.defer(1, "pipeline", [point(0, 9)]).unwrap();
        assert_eq!(owner.abandon(), vec![(1, "pipeline")]);
    }

    #[test]
    fn a_refused_defer_returns_ownership_to_its_caller() {
        let mut owner = NativeRetirement::new(VulkanDeviceEpochId::new(1));
        owner.defer(1, "first", [point(0, 2)]).unwrap();
        assert_eq!(
            owner.defer(1, "second", [point(0, 3)]),
            Err(NativeRetirementFailure {
                reason: NativeRetirementError::DuplicateTicket,
                value: "second",
            })
        );
    }
}
