//! Composition wrapper for the extracted push-descriptor capability.

pub use reims_vgpu_vulkan::push_descriptor::PushDescriptorCaps;

/// Resolve the extension and its mandatory limit, honoring the operator switch.
///
/// # Safety
///
/// `pd` must belong to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: ash::vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
) -> PushDescriptorCaps {
    let enabled = crate::env::switch(crate::env::PUSH_DESCRIPTORS) != crate::env::Switch::Off;
    unsafe { reims_vgpu_vulkan::push_descriptor::query(instance, pd, has_extension, enabled) }
}
