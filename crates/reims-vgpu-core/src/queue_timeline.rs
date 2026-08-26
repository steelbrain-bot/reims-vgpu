//! Monotonic native queue timeline allocation for one Vulkan epoch.
//!
//! A queue owner allocates a point immediately before submission and signals
//! that exact value in the same native call. A failed submission may consume a
//! value; later signals may skip it, while no accepted transaction or wait is
//! ever associated with the failed point.

use crate::QueueTimelinePoint;
use reims_vgpu_protocol::{QueueOwnerId, QueueTimelineValue, VulkanDeviceEpochId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueTimelineError {
    ValueExhausted,
    MixedEpoch,
    PredecessorNotAllocated,
}

#[derive(Clone, Debug)]
pub struct QueueTimelineOwner {
    epoch: VulkanDeviceEpochId,
    next: BTreeMap<QueueOwnerId, u64>,
}

impl QueueTimelineOwner {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            epoch,
            next: BTreeMap::new(),
        }
    }

    pub const fn epoch(&self) -> VulkanDeviceEpochId {
        self.epoch
    }

    pub fn allocate(
        &mut self,
        queue: QueueOwnerId,
    ) -> Result<QueueTimelinePoint, QueueTimelineError> {
        let next = self.next.entry(queue).or_insert(1);
        let value = *next;
        *next = next
            .checked_add(1)
            .ok_or(QueueTimelineError::ValueExhausted)?;
        Ok(QueueTimelinePoint {
            epoch: self.epoch,
            queue,
            value: QueueTimelineValue::new(value),
        })
    }

    pub fn allocate_after(
        &mut self,
        predecessor: QueueTimelinePoint,
    ) -> Result<QueueTimelinePoint, QueueTimelineError> {
        if predecessor.epoch != self.epoch {
            return Err(QueueTimelineError::MixedEpoch);
        }
        let next = self.next.get(&predecessor.queue).copied().unwrap_or(1);
        if predecessor.value.get() >= next {
            return Err(QueueTimelineError::PredecessorNotAllocated);
        }
        self.allocate(predecessor.queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_queue_advances_independently_inside_one_epoch() {
        let mut owner = QueueTimelineOwner::new(VulkanDeviceEpochId::new(7));
        assert_eq!(
            owner.allocate(QueueOwnerId::new(1)).unwrap().value,
            QueueTimelineValue::new(1)
        );
        assert_eq!(
            owner.allocate(QueueOwnerId::new(2)).unwrap().value,
            QueueTimelineValue::new(1)
        );
        let second = owner.allocate(QueueOwnerId::new(1)).unwrap();
        assert_eq!(second.epoch, VulkanDeviceEpochId::new(7));
        assert_eq!(second.value, QueueTimelineValue::new(2));
    }

    #[test]
    fn an_auxiliary_successor_requires_a_point_from_this_epoch_and_allocator() {
        let epoch = VulkanDeviceEpochId::new(7);
        let mut owner = QueueTimelineOwner::new(epoch);
        let predecessor = owner.allocate(QueueOwnerId::new(2)).unwrap();
        assert_eq!(
            owner.allocate_after(predecessor).unwrap().value,
            QueueTimelineValue::new(2)
        );
        assert_eq!(
            owner.allocate_after(QueueTimelinePoint {
                epoch,
                queue: QueueOwnerId::new(3),
                value: QueueTimelineValue::new(1),
            }),
            Err(QueueTimelineError::PredecessorNotAllocated)
        );
    }
}
