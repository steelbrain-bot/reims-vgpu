//! Capability admission for the replacement Vulkan epoch.
//!
//! Timeline semaphores are structural to queue completion and cross-queue
//! waits. A host without the queried feature cannot create this backend; no
//! fence or blocking-drain compatibility path is installed.

use crate::device_features::DeviceFeatures;
use reims_vgpu_core::{select_descriptor_tier, DescriptorCapabilities, DescriptorTier};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementCapabilityError {
    TimelineSemaphoreUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementCapabilities {
    timeline_semaphore: bool,
    descriptor_tier: DescriptorTier,
}

impl ReplacementCapabilities {
    pub fn require(
        features: &DeviceFeatures,
        descriptors: DescriptorCapabilities,
    ) -> Result<Self, ReplacementCapabilityError> {
        if !features.timeline_semaphore {
            return Err(ReplacementCapabilityError::TimelineSemaphoreUnavailable);
        }
        Ok(Self {
            timeline_semaphore: true,
            descriptor_tier: select_descriptor_tier(descriptors),
        })
    }

    pub const fn timeline_semaphore(self) -> bool {
        self.timeline_semaphore
    }

    pub const fn descriptor_tier(self) -> DescriptorTier {
        self.descriptor_tier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_semaphore_is_a_support_floor_not_a_fallback_choice() {
        let mut features = DeviceFeatures::default();
        assert_eq!(
            ReplacementCapabilities::require(
                &features,
                DescriptorCapabilities {
                    descriptor_buffer: false,
                    push_descriptor: false,
                },
            ),
            Err(ReplacementCapabilityError::TimelineSemaphoreUnavailable)
        );
        features.timeline_semaphore = true;
        assert!(ReplacementCapabilities::require(
            &features,
            DescriptorCapabilities {
                descriptor_buffer: false,
                push_descriptor: true,
            },
        )
        .unwrap()
        .timeline_semaphore());
        assert_eq!(
            ReplacementCapabilities::require(
                &features,
                DescriptorCapabilities {
                    descriptor_buffer: false,
                    push_descriptor: true,
                },
            )
            .unwrap()
            .descriptor_tier(),
            DescriptorTier::PushDescriptor
        );
    }
}
