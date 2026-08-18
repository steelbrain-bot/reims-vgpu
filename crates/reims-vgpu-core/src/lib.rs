//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod resource;

pub use resource::{
    GraphError, LifecycleState, MappingNode, ResourceGraph, ResourceNode, StorageBacking,
    StorageNode,
};
