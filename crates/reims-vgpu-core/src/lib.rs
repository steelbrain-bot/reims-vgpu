//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod capabilities;
pub mod namespace;
pub mod resource;
pub mod submission;
pub mod target;

pub use capabilities::{DeviceInfoLimits, ExecutorCapabilities};
pub use namespace::{NamespaceError, ReferenceNamespace};
pub use resource::{
    ContentAuthority, ContentError, ContentState, GraphError, LifecycleState, MappingNode,
    PendingContentWrite, ReplicaVersions, ResourceGraph, ResourceNode, StorageBacking, StorageNode,
};
pub use submission::SubmissionContext;
pub use target::{TargetIdentity, TargetKeyDivergence};
