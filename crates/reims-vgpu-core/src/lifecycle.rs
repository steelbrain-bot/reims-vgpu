//! Independent semantic-generation and Vulkan-device-epoch lifetimes.
//!
//! Guest reset and host device loss invalidate different things. Closing a
//! [`SessionGeneration`] prevents later resolution from entering that semantic
//! generation, while leases already accepted from it remain owned until their
//! transaction retires. Losing a [`VulkanDeviceEpoch`] immediately makes every
//! native lease from that epoch unusable for new Vulkan calls. Keeping both
//! identities in [`NativeObjectLease`] makes those rules impossible to merge
//! into one ambiguous "device generation" check.

use reims_vgpu_protocol::{SessionGenerationId, VulkanDeviceEpochId};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Debug)]
struct SessionGenerationState {
    id: SessionGenerationId,
    accepting: AtomicBool,
}

/// Semantic lifetime opened by attach or guest reset.
#[derive(Clone, Debug)]
pub struct SessionGeneration(Arc<SessionGenerationState>);

impl SessionGeneration {
    pub fn new(id: SessionGenerationId) -> Self {
        Self(Arc::new(SessionGenerationState {
            id,
            accepting: AtomicBool::new(true),
        }))
    }

    pub fn id(&self) -> SessionGenerationId {
        self.0.id
    }

    /// Acquire the lifetime carried by newly accepted work.
    ///
    /// A close racing this operation has one linearization point: work either
    /// obtains the lease and is already accepted, or observes the closed state
    /// and must resolve against a later generation.
    pub fn try_lease(&self) -> Option<SessionGenerationLease> {
        self.0
            .accepting
            .load(Ordering::Acquire)
            .then(|| SessionGenerationLease(Arc::clone(&self.0)))
            .filter(|_| self.0.accepting.load(Ordering::Acquire))
    }

    /// Stop new resolution into this generation.
    pub fn close(&self) {
        self.0.accepting.store(false, Ordering::Release);
    }

    pub fn is_accepting(&self) -> bool {
        self.0.accepting.load(Ordering::Acquire)
    }
}

/// Retention of a semantic generation by work accepted before it closed.
#[derive(Clone, Debug)]
pub struct SessionGenerationLease(Arc<SessionGenerationState>);

impl SessionGenerationLease {
    pub fn id(&self) -> SessionGenerationId {
        self.0.id
    }
}

/// Host-device lifetime state relevant to use of native handles.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VulkanDeviceEpochState {
    Active = 0,
    Losing = 1,
    Retiring = 2,
    Dead = 3,
}

#[derive(Debug)]
struct VulkanDeviceEpochInner {
    id: VulkanDeviceEpochId,
    state: AtomicU8,
    active_gate: RwLock<()>,
}

/// Lifetime of host objects invalidated together by Vulkan device loss.
#[derive(Clone, Debug)]
pub struct VulkanDeviceEpoch(Arc<VulkanDeviceEpochInner>);

impl VulkanDeviceEpoch {
    pub fn new(id: VulkanDeviceEpochId) -> Self {
        Self(Arc::new(VulkanDeviceEpochInner {
            id,
            state: AtomicU8::new(VulkanDeviceEpochState::Active as u8),
            active_gate: RwLock::new(()),
        }))
    }

    pub fn id(&self) -> VulkanDeviceEpochId {
        self.0.id
    }

    pub fn try_lease(&self) -> Option<VulkanDeviceEpochLease> {
        (self.state() == VulkanDeviceEpochState::Active)
            .then(|| VulkanDeviceEpochLease(Arc::clone(&self.0)))
            .filter(|lease| lease.is_usable())
    }

    pub fn begin_loss(&self) -> bool {
        let _gate = self
            .0
            .active_gate
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.0
            .state
            .compare_exchange(
                VulkanDeviceEpochState::Active as u8,
                VulkanDeviceEpochState::Losing as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn begin_retirement(&self) {
        let _gate = self
            .0
            .active_gate
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.advance_to(VulkanDeviceEpochState::Retiring);
    }

    pub fn finish_retirement(&self) {
        let _gate = self
            .0
            .active_gate
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.advance_to(VulkanDeviceEpochState::Dead);
    }

    pub fn state(&self) -> VulkanDeviceEpochState {
        decode_epoch_state(self.0.state.load(Ordering::Acquire))
    }

    /// Run native-object construction and publication only while this epoch is
    /// active. Concurrent constructors share the read side; loss/retirement
    /// takes the write side and therefore cannot return while a constructor is
    /// still able to publish into the old incarnation.
    pub fn with_active<Input, Output>(
        &self,
        input: Input,
        operation: impl FnOnce(Input) -> Output,
    ) -> Result<Output, Input> {
        let _gate = self
            .0
            .active_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.state() != VulkanDeviceEpochState::Active {
            return Err(input);
        }
        Ok(operation(input))
    }

    fn advance_to(&self, target: VulkanDeviceEpochState) {
        self.0.state.fetch_max(target as u8, Ordering::AcqRel);
    }
}

/// Retention and validity check for handles owned by one Vulkan epoch.
#[derive(Clone, Debug)]
pub struct VulkanDeviceEpochLease(Arc<VulkanDeviceEpochInner>);

impl VulkanDeviceEpochLease {
    pub fn id(&self) -> VulkanDeviceEpochId {
        self.0.id
    }

    pub fn state(&self) -> VulkanDeviceEpochState {
        decode_epoch_state(self.0.state.load(Ordering::Acquire))
    }

    pub fn is_usable(&self) -> bool {
        self.state() == VulkanDeviceEpochState::Active
    }
}

/// Dual lifetime carried by every accepted native object use.
#[derive(Clone, Debug)]
pub struct NativeObjectLease {
    pub session_generation: SessionGenerationLease,
    pub vulkan_epoch: VulkanDeviceEpochLease,
}

impl NativeObjectLease {
    pub fn acquire(generation: &SessionGeneration, epoch: &VulkanDeviceEpoch) -> Option<Self> {
        let session_generation = generation.try_lease()?;
        let vulkan_epoch = epoch.try_lease()?;
        Some(Self {
            session_generation,
            vulkan_epoch,
        })
    }

    /// Whether the native handles may be used by a Vulkan call now.
    ///
    /// Closing the semantic generation does not revoke accepted work. Device
    /// loss does, even if the semantic generation remains named.
    pub fn native_handles_are_usable(&self) -> bool {
        self.vulkan_epoch.is_usable()
    }
}

fn decode_epoch_state(raw: u8) -> VulkanDeviceEpochState {
    match raw {
        0 => VulkanDeviceEpochState::Active,
        1 => VulkanDeviceEpochState::Losing,
        2 => VulkanDeviceEpochState::Retiring,
        _ => VulkanDeviceEpochState::Dead,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_reset_closes_semantics_without_recreating_a_healthy_epoch() {
        let first = SessionGeneration::new(SessionGenerationId::new(1));
        let epoch = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(9));
        let accepted = NativeObjectLease::acquire(&first, &epoch).expect("active lifetimes");

        first.close();
        let second = SessionGeneration::new(SessionGenerationId::new(2));

        assert!(first.try_lease().is_none());
        assert_eq!(second.try_lease().expect("new generation").id().get(), 2);
        assert_eq!(accepted.session_generation.id().get(), 1);
        assert_eq!(accepted.vulkan_epoch.id(), epoch.id());
        assert!(accepted.native_handles_are_usable());
    }

    #[test]
    fn device_loss_invalidates_native_handles_without_renaming_semantics() {
        let generation = SessionGeneration::new(SessionGenerationId::new(4));
        let lost = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(7));
        let accepted = NativeObjectLease::acquire(&generation, &lost).expect("active lifetimes");

        assert!(lost.begin_loss());
        assert!(!accepted.native_handles_are_usable());
        assert_eq!(accepted.session_generation.id(), generation.id());
        assert!(generation.is_accepting());
        assert!(lost.try_lease().is_none());

        let replacement = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(8));
        let replacement_lease =
            NativeObjectLease::acquire(&generation, &replacement).expect("replacement epoch");
        assert!(replacement_lease.native_handles_are_usable());
        assert_ne!(
            accepted.vulkan_epoch.id(),
            replacement_lease.vulkan_epoch.id()
        );
    }

    #[test]
    fn two_sessions_share_no_mutable_lifecycle_state() {
        let generation_a = SessionGeneration::new(SessionGenerationId::new(1));
        let generation_b = SessionGeneration::new(SessionGenerationId::new(1));
        let epoch_a = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(1));
        let epoch_b = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(1));

        generation_a.close();
        assert!(!generation_a.is_accepting());
        assert!(generation_b.is_accepting());

        epoch_a.begin_loss();
        assert_eq!(epoch_a.state(), VulkanDeviceEpochState::Losing);
        assert_eq!(epoch_b.state(), VulkanDeviceEpochState::Active);
    }

    #[test]
    fn epoch_state_only_advances() {
        let epoch = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(3));
        epoch.begin_retirement();
        assert_eq!(epoch.state(), VulkanDeviceEpochState::Retiring);
        assert!(!epoch.begin_loss());
        epoch.finish_retirement();
        assert_eq!(epoch.state(), VulkanDeviceEpochState::Dead);
    }

    #[test]
    fn loss_cannot_return_before_an_active_publication_finishes() {
        let epoch = VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(11));
        let constructor_epoch = epoch.clone();
        let (entered_sender, entered_receiver) = std::sync::mpsc::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let constructor = std::thread::spawn(move || {
            constructor_epoch
                .with_active((), |()| {
                    entered_sender.send(()).unwrap();
                    release_receiver.recv().unwrap();
                })
                .unwrap();
        });
        entered_receiver.recv().unwrap();

        let loss_epoch = epoch.clone();
        let (lost_sender, lost_receiver) = std::sync::mpsc::channel();
        let loss = std::thread::spawn(move || {
            lost_sender.send(loss_epoch.begin_loss()).unwrap();
        });
        assert!(lost_receiver.try_recv().is_err());
        release_sender.send(()).unwrap();
        constructor.join().unwrap();
        assert!(lost_receiver.recv().unwrap());
        loss.join().unwrap();
        assert!(matches!(epoch.with_active(7, |_| ()), Err(7)));
    }
}
