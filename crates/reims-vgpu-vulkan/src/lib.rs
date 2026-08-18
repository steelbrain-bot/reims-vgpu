//! Vulkan implementation policy for Reims vGPU.
//!
//! This crate is the only layer allowed to interpret host Vulkan capabilities
//! as placement and transfer choices. Guest-visible resource lifetime and
//! content authority remain in `reims-vgpu-core`.

pub mod api_floor;
pub mod capabilities;
pub mod device_features;
pub mod device_select;
pub mod format;
pub mod host_pointer;
pub mod memory;
pub mod policy;
pub mod push_descriptor;
