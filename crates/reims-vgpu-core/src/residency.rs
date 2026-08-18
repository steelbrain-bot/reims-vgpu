//! Opaque ownership contracts for executor-local residents.

use crate::TargetIdentity;

/// Backend-independent classification of a retained target's current content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentContentBacking {
    NotReady,
    DeviceAllocation,
}

/// Opaque executor ownership of one backend resident for a guest resource.
///
/// Dropping the token performs the executor's fence-safe release. Core state
/// can check identity continuity and content availability, but cannot inspect
/// or operate the backend allocation.
pub trait ResidentLease: std::fmt::Debug + Send {
    fn matches(&self, identity: &TargetIdentity) -> bool;
    fn backing(&self) -> ResidentContentBacking;
}
