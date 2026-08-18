//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod namespace;
pub mod resource;

pub use namespace::{NamespaceError, ReferenceNamespace};
pub use resource::{
    ContentAuthority, ContentError, ContentState, GraphError, LifecycleState, MappingNode,
    PendingContentWrite, ReplicaVersions, ResourceGraph, ResourceNode, StorageBacking, StorageNode,
};
