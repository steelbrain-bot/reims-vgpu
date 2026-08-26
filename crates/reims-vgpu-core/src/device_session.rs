//! Per-vGPU ownership of semantic generations and Vulkan device epochs.

use crate::{
    SessionGeneration, SessionGenerationLease, VulkanDeviceEpoch, VulkanDeviceEpochLease,
    VulkanDeviceEpochState,
};
use reims_vgpu_protocol::{SessionGenerationId, SessionId, VulkanDeviceEpochId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceSessionError {
    GenerationDidNotAdvance,
    EpochDidNotAdvance,
    HealthyEpochCannotBeReplaced,
}

#[derive(Clone, Debug)]
pub struct DeviceSession {
    id: SessionId,
    generation: SessionGeneration,
    epoch: VulkanDeviceEpoch,
}

impl DeviceSession {
    pub fn new(id: SessionId, generation: SessionGenerationId, epoch: VulkanDeviceEpochId) -> Self {
        Self {
            id,
            generation: SessionGeneration::new(generation),
            epoch: VulkanDeviceEpoch::new(epoch),
        }
    }

    pub const fn id(&self) -> SessionId {
        self.id
    }

    pub fn generation_lease(&self) -> Option<SessionGenerationLease> {
        self.generation.try_lease()
    }

    /// Current semantic lifetime shared with ingress and native-object owners.
    pub fn session_generation(&self) -> SessionGeneration {
        self.generation.clone()
    }

    pub fn epoch_lease(&self) -> Option<VulkanDeviceEpochLease> {
        self.epoch.try_lease()
    }

    /// Current Vulkan lifetime shared with the backend incarnation that owns
    /// the handles. Sharing this object, rather than copying its id into a new
    /// state cell, makes device-loss invalidation atomic across both layers.
    pub fn vulkan_epoch(&self) -> VulkanDeviceEpoch {
        self.epoch.clone()
    }

    /// Close semantic admission and open a new generation while retaining the
    /// healthy Vulkan epoch and all leases held by already accepted work.
    pub fn guest_reset(
        &mut self,
        next: SessionGenerationId,
    ) -> Result<SessionGeneration, DeviceSessionError> {
        if next <= self.generation.id() {
            return Err(DeviceSessionError::GenerationDidNotAdvance);
        }
        self.generation.close();
        Ok(std::mem::replace(
            &mut self.generation,
            SessionGeneration::new(next),
        ))
    }

    pub fn begin_device_loss(&self) {
        self.epoch.begin_loss();
    }

    pub fn retire_lost_epoch(&self) {
        self.epoch.begin_retirement();
        self.epoch.finish_retirement();
    }

    /// Install a replacement only after the old epoch is dead. Semantic
    /// generation identity is unchanged by host-device recreation.
    pub fn replace_device_epoch(
        &mut self,
        next: VulkanDeviceEpochId,
    ) -> Result<VulkanDeviceEpoch, DeviceSessionError> {
        if self.epoch.state() != VulkanDeviceEpochState::Dead {
            return Err(DeviceSessionError::HealthyEpochCannotBeReplaced);
        }
        if next <= self.epoch.id() {
            return Err(DeviceSessionError::EpochDidNotAdvance);
        }
        Ok(std::mem::replace(
            &mut self.epoch,
            VulkanDeviceEpoch::new(next),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_reset_changes_semantics_without_recreating_the_healthy_device() {
        let mut session = DeviceSession::new(
            SessionId::new(1),
            SessionGenerationId::new(1),
            VulkanDeviceEpochId::new(9),
        );
        let old_generation = session.generation_lease().unwrap();
        let old_epoch = session.epoch_lease().unwrap();
        let closed = session.guest_reset(SessionGenerationId::new(2)).unwrap();
        assert!(!closed.is_accepting());
        assert_eq!(old_generation.id(), SessionGenerationId::new(1));
        assert_eq!(session.epoch_lease().unwrap().id(), old_epoch.id());
        assert!(old_epoch.is_usable());
    }

    #[test]
    fn device_replacement_changes_epoch_without_renaming_semantics() {
        let mut session = DeviceSession::new(
            SessionId::new(1),
            SessionGenerationId::new(4),
            VulkanDeviceEpochId::new(2),
        );
        let generation = session.generation_lease().unwrap();
        let old_epoch = session.epoch_lease().unwrap();
        session.begin_device_loss();
        assert!(!old_epoch.is_usable());
        session.retire_lost_epoch();
        session
            .replace_device_epoch(VulkanDeviceEpochId::new(3))
            .unwrap();
        assert_eq!(session.generation_lease().unwrap().id(), generation.id());
        assert_eq!(
            session.epoch_lease().unwrap().id(),
            VulkanDeviceEpochId::new(3)
        );
    }

    #[test]
    fn two_sessions_cannot_transition_each_others_epoch() {
        let mut first = DeviceSession::new(
            SessionId::new(1),
            SessionGenerationId::new(1),
            VulkanDeviceEpochId::new(1),
        );
        let second = DeviceSession::new(
            SessionId::new(2),
            SessionGenerationId::new(1),
            VulkanDeviceEpochId::new(1),
        );
        first.begin_device_loss();
        first.retire_lost_epoch();
        first
            .replace_device_epoch(VulkanDeviceEpochId::new(2))
            .unwrap();
        assert!(second.epoch_lease().unwrap().is_usable());
        assert_eq!(
            second.epoch_lease().unwrap().id(),
            VulkanDeviceEpochId::new(1)
        );
    }
}
