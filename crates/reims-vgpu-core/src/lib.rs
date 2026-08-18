//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod blit;
pub mod capabilities;
pub mod execution;
pub mod namespace;
pub mod residency;
pub mod resource;
pub mod resource_state;
pub mod service;
pub mod submission;
pub mod target;
pub mod texel;

pub use blit::{BufferFillPattern, ResolvedBufferBlit, ResolvedBufferRange};
pub use capabilities::{CapabilityService, DeviceInfoLimits, ExecutorCapabilities, MAX_CHANNELS};
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
    MappingNode, PendingContentWrite, ReplicaVersions, ResourceGraph, ResourceLifetime,
    ResourceLifetimeRef, ResourceNode, StorageBacking, StorageNode,
};
pub use resource_state::ResolvedResourceState;
pub use service::{
    GuestWriteReach, GuestWriteService, PresentDecline, PresentationService, ReadbackLease,
    ReadbackService, ResidentContent, ResidentReclaim, ResidentService, TargetReadback,
};
pub use submission::SubmissionContext;
pub use target::{TargetIdentity, TargetKeyDivergence};
pub use texel::{
    expand_rgba8_to_texel, f16_to_f32, f16_to_unorm8, narrow_texel_to_rgba8, unorm8_to_f16,
};
