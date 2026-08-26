//! Reims vGPU host path — single crate.
//!
//! | Module | Role |
//! | --- | --- |
//! | [`model`] | Register-window and FIFO contract vocabulary |
//! | [`runtime`] | Replacement transaction composition and host adaptation |
//! | [`qemu`] | QEMU C ABI surface only |
//!
//! The product executor is the sibling `reims-vgpu-vulkan` crate.
//!
//! # The two supported pathways
//!
//! Both pathways use the same backend and differ only in the Vulkan loader:
//!
//! | Arm | `cfg` | Host GPU API |
//! | --- | --- | --- |
//! | Vulkan / MoltenVK | `target_os = "macos"` | MoltenVK |
//! | Vulkan / native | `target_os = "linux"` | native ICD |
//!
//! **Gate the host on `target_os` and nothing else.** `macos` and `linux` are
//! the only two values this crate names, so the pathways differ in one term
//! each and a reader greps one key to find every host gate.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rust_2018_idioms)]

// Vulkan reaches the GPU through MoltenVK on macOS and a native ICD on Linux.
// Any other host is untested rather than known-broken — name it here so a new
// port is a deliberate edit to this list, not an accident.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!(
    "the Vulkan backend is supported on target_os = \"macos\" (MoltenVK) and \
     target_os = \"linux\" (native ICD) only"
);

/// Every environment variable this device reads, and the rule that an override
/// may only narrow what it does — see the module doc.
pub mod env;
pub mod model;
/// Crate-wide observability: the always-on fail sink and the decline
/// vocabulary. Above `runtime/` because every subsystem owes the reader a
/// reason, and `translate/` + `caps/` must be able to name one without
/// depending on `runtime/`.
pub mod observe;
pub mod runtime;

pub mod qemu;

/// Host-owned presentation window (winit + VkSurfaceKHR) — see
/// [[host-window]]. The `host-window` feature adds the windowing adapter and is
/// enabled for every verification command the x86 pathway is checked with.
#[cfg(feature = "host-window")]
pub mod host_window;

/// The replacement device registry and the entry surface `qemu::abi` wraps.
mod device;
pub(crate) use device::{backend_name, unwind_safe, CursorGlyphInfo};
