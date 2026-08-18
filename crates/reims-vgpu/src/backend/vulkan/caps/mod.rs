//! Host GPU capability facade.
//!
//! Vulkan-owned classification and feature decisions live in
//! `reims-vgpu-vulkan`. This module retains only composition wrappers for
//! operator switches and compatibility paths while the engine is extracted.

pub mod device_features;
pub mod host_pointer;
pub mod memory_topology;
pub mod push_descriptor;

pub use reims_vgpu_vulkan::capabilities::{DriverQuirk, HostGpuCaps};
pub use reims_vgpu_vulkan::{api_floor, device_select};
