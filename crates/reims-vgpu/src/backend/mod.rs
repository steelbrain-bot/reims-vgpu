//! Backend selection seam.
//!
//! - [`vulkan`] is the concrete backend, self-contained in this crate via
//!   `ash` + [`vulkan::engine`].
//! - Draws, compute and blits do **not** come through this module. The live
//!   seams are `runtime/draw::try_metal2vulkan_draw` → [`vulkan::engine`]
//!   and [`vulkan::engine`]; the runtime drives them directly.
//!
//! Metal indices/semantics are canonical (guest wire is serialized Metal).
//! Vulkan-only binding rewrites live only in [`vulkan`].

#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

/// Guest-lifetime teardown, the one thing a backend owns that the runtime
/// cannot do for it.
///
/// The trait is this small on purpose. It once declared the whole
/// Metal-semantic operation set — texture create/write/read, blit, compute,
/// render, present — and nothing ever called any of it: the runtime drives the
/// backends directly through their own seams, so every one of those methods
/// returned a refusal or a bare `Ok` without touching a GPU.
pub trait Backend {
    /// Drop all state derived from the current guest lifetime.
    ///
    /// Immutable, content-keyed shader/pipeline caches may survive. Guest object
    /// identities, resident images, and aliases of guest memory must not.
    fn reset(&mut self) {}
}

/// Null backend for protocol/device tests without a GPU.
#[derive(Default)]
pub struct NullBackend;

impl Backend for NullBackend {}
