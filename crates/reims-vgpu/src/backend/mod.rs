//! Host GPU implementations.
//!
//! Metal indices and semantics are canonical at the protocol boundary.
//! Vulkan-only binding rewrites live below [`vulkan`]. Device ownership and
//! guest-lifetime reset enter through [`crate::runtime::executor::Executor`].

#[cfg(feature = "backend-vulkan")]
pub mod vulkan;
