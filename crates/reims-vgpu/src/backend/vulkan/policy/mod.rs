//! Host-memory topology changes placement and transfer scheduling only.
//!
//! The semantic core decides resource identity, content versions, completion,
//! and guest-visible success before this policy is consulted. Host-pointer
//! import is deliberately absent from this interface: it is an orthogonal
//! measured capability and each placement policy must work both with and
//! without it.

mod discrete;
mod unified;

use super::caps::memory_topology::{MemoryClass, MemoryRequest, MemoryTopology};

pub(crate) const UNIFIED_DEFAULT_BATCH_DRAWS: u64 = 128;
pub(crate) const DISCRETE_DEFAULT_BATCH_DRAWS: u64 = 32;
pub(crate) const MAX_BATCH_DRAWS: u64 = UNIFIED_DEFAULT_BATCH_DRAWS;

/// Decisions a host-memory topology may make.
trait TopologyPolicy {
    fn request(&self, class: MemoryClass) -> MemoryRequest;
    fn default_batch_draws(&self) -> u64;
}

/// Selected topology policy for one physical Vulkan device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolicyKind {
    Unified(unified::UnifiedMemoryPolicy),
    Discrete(discrete::DiscreteMemoryPolicy),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemoryPlacementPolicy(PolicyKind);

impl MemoryPlacementPolicy {
    pub(crate) const fn new(topology: MemoryTopology) -> Self {
        match topology {
            MemoryTopology::Unified => Self(PolicyKind::Unified(unified::UnifiedMemoryPolicy)),
            MemoryTopology::Discrete => Self(PolicyKind::Discrete(discrete::DiscreteMemoryPolicy)),
        }
    }

    pub(crate) fn request(self, class: MemoryClass) -> MemoryRequest {
        match self.0 {
            PolicyKind::Unified(policy) => policy.request(class),
            PolicyKind::Discrete(policy) => policy.request(class),
        }
    }

    pub(crate) fn default_batch_draws(self) -> u64 {
        match self.0 {
            PolicyKind::Unified(policy) => policy.default_batch_draws(),
            PolicyKind::Discrete(policy) => policy.default_batch_draws(),
        }
    }
}

fn topology_independent_request(class: MemoryClass) -> Option<MemoryRequest> {
    use ash::vk::MemoryPropertyFlags as F;
    match class {
        MemoryClass::DeviceLocal => Some(MemoryRequest {
            required: F::DEVICE_LOCAL,
            preferred: Vec::new(),
        }),
        MemoryClass::DeviceLocalPreferred => Some(MemoryRequest {
            required: F::empty(),
            preferred: vec![F::DEVICE_LOCAL],
        }),
        MemoryClass::Upload | MemoryClass::Readback => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_cannot_change_device_local_semantics() {
        for class in [MemoryClass::DeviceLocal, MemoryClass::DeviceLocalPreferred] {
            assert_eq!(
                MemoryPlacementPolicy::new(MemoryTopology::Unified).request(class),
                MemoryPlacementPolicy::new(MemoryTopology::Discrete).request(class)
            );
        }
    }

    #[test]
    fn topology_policies_have_independent_submission_defaults() {
        assert_eq!(
            MemoryPlacementPolicy::new(MemoryTopology::Unified).default_batch_draws(),
            UNIFIED_DEFAULT_BATCH_DRAWS
        );
        assert_eq!(
            MemoryPlacementPolicy::new(MemoryTopology::Discrete).default_batch_draws(),
            DISCRETE_DEFAULT_BATCH_DRAWS
        );
    }
}
