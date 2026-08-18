//! Compatibility surface for the extracted Vulkan memory subsystem.

pub use reims_vgpu_vulkan::memory::*;

/// Query the device-wide Vulkan allocation ceiling and report a missing value.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn max_allocation_size(instance: &ash::Instance, pd: ash::vk::PhysicalDevice) -> u64 {
    match unsafe { reims_vgpu_vulkan::memory::reported_max_allocation_size(instance, pd) } {
        Some(size) => size,
        None => {
            crate::observe::fail(
                "vk_max_allocation_unreported reason=vk_max_allocation_unreported (the device \
                 reported maxMemoryAllocationSize=0; allocations are bounded by their heap alone)",
            );
            u64::MAX
        }
    }
}
