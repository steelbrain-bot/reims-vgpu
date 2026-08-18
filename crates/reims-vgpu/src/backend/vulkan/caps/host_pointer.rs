//! Composition wrapper for the extracted host-pointer import capability.

pub use reims_vgpu_vulkan::host_pointer::*;

/// Interpret the operator switch as a capability-narrowing override.
fn env_override() -> Option<HostPointerImport> {
    match crate::env::read(crate::env::GUEST_IMPORT) {
        (crate::env::Switch::Off, _) => Some(HostPointerImport::DisabledByEnv),
        (crate::env::Switch::Unrecognized, value) => {
            crate::observe::fail(format!(
                "vk_guest_import_env_unrecognized var={} value={:?} (expected on|off; the rail is \
                 left to the device)",
                crate::env::GUEST_IMPORT,
                value.unwrap_or_default()
            ));
            None
        }
        (crate::env::Switch::On | crate::env::Switch::Unset, _) => None,
    }
}

/// Resolve host-pointer importability, honoring the operator's narrowing switch.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: ash::vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
    max_allocation: u64,
) -> HostPointerCaps {
    unsafe {
        reims_vgpu_vulkan::host_pointer::query(
            instance,
            pd,
            has_extension,
            max_allocation,
            env_override(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{env_override, HostPointerImport};

    fn with_env(value: Option<&str>) -> Option<HostPointerImport> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
        unsafe {
            match value {
                Some(value) => std::env::set_var(crate::env::GUEST_IMPORT, value),
                None => std::env::remove_var(crate::env::GUEST_IMPORT),
            }
        }
        let answer = env_override();
        unsafe { std::env::remove_var(crate::env::GUEST_IMPORT) };
        answer
    }

    #[test]
    fn the_env_switch_only_narrows_the_capability() {
        assert_eq!(
            with_env(Some("off")),
            Some(HostPointerImport::DisabledByEnv)
        );
        for enabled in ["1", "on", "true", "yes"] {
            assert_eq!(with_env(Some(enabled)), None);
        }
        assert_eq!(with_env(None), None);
    }

    #[test]
    fn an_unrecognized_value_leaves_the_device_to_decide() {
        assert_eq!(with_env(Some("maybe")), None);
    }
}
