//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod capabilities;
pub mod execution;
pub mod namespace;
pub mod residency;
pub mod resource;
pub mod service;
pub mod submission;
pub mod target;

pub use capabilities::{CapabilityService, DeviceInfoLimits, ExecutorCapabilities};
pub use execution::{
    ExecutionCompletion, ExecutionKind, ExecutionOutput, ExecutionPort, ExecutionReceipt,
    ResolvedSubmission,
};
pub use namespace::{NamespaceError, ReferenceNamespace};
pub use residency::{
    ComputeResidencyService, ComputeStorageOrigin, ComputeStorageResidencyKey,
    ResidentContentBacking, ResidentLease,
};
pub use resource::{
    ContentAuthority, ContentError, ContentStamp, ContentState, GraphError, LifecycleState,
    MappingNode, PendingContentWrite, ReplicaVersions, ResourceGraph, ResourceNode, StorageBacking,
    StorageNode,
};
pub use service::{
    GuestWriteReach, GuestWriteService, PresentDecline, PresentationService, ReadbackLease,
    ReadbackService, ResidentContent, ResidentReclaim, ResidentService, TargetReadback,
};
pub use submission::SubmissionContext;
pub use target::{TargetIdentity, TargetKeyDivergence};
