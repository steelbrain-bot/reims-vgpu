//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod blit;
pub mod capabilities;
pub mod compute;
pub mod draw;
pub mod endian;
pub mod execution;
pub mod namespace;
pub mod pixel_format;
pub mod render;
pub mod residency;
pub mod resource;
pub mod resource_state;
pub mod service;
pub mod submission;
pub mod target;
pub mod texel;
pub mod visibility;

pub use blit::{BufferFillPattern, ResolvedBufferBlit, ResolvedBufferRange};
pub use capabilities::{CapabilityService, DeviceInfoLimits, ExecutorCapabilities, MAX_CHANNELS};
pub use compute::{
    ComputeBufferBacking, ComputeBufferOutput, ComputeBufferResource, ComputeBufferResult,
    ComputeImageDestination, ComputeImageResult, ComputeOutput, ComputeRequest,
    ComputeResidentSampleBind, ComputeSampledImageResource, ComputeSampledImageSource,
    ComputeStorageImageResource, ComputeStorageImageSeed, ComputeStorageResidency,
    SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
    SamplerMipFilter, SamplerResource,
};
pub use execution::{
    execute_resolved_submission, BlitCompletion, CommandExecution, ExecutionCompletion,
    ExecutionKind, ExecutionOutput, ExecutionPort, ExecutionReceipt, ResolvedCommand,
    ResolvedCommandBuffer, ResolvedExecutionCompletion, ResolvedSubmission,
    ResourceStateCompletion,
};
pub use namespace::{NamespaceError, ReferenceNamespace};
pub use render::{
    viewport_slot_count, AttachmentInitial, AttachmentSlot, BlendFactor, BlendOp,
    BlendStateResource, BufferContent, CullMode, DepthClipMode, DepthState, DrawOutput,
    DrawRequest, FillMode, IndexType, IndexedDrawResource, PrimitiveTopology, SampledByteOrigin,
    SampledContentIdentity, SampledImageResource, SampledSource, ScissorResource,
    SecondaryColorTarget, SeedOrder, StencilFaceOps, StencilOp, StencilState,
    StorageBufferResource, VertexAttributeFormat, VertexAttributeResource, VertexStepFunction,
    ViewportResource, VisibilityResultMode,
};
pub use residency::{
    ComputeResidencyService, ComputeStorageOrigin, ComputeStorageResidencyKey, GatherVouch,
    ResidentContentBacking,
};
pub use resource::{
    ContentAuthority, ContentError, ContentStamp, ContentState, GraphError, LifecycleState,
    MappingNode, PendingContentWrite, ReplicaVersions, ResourceGraph, ResourceLifetime,
    ResourceLifetimeRef, ResourceNode, StorageBacking, StorageNode,
};
pub use resource_state::ResolvedResourceState;
pub use service::{
    GuestWriteReach, GuestWriteService, PresentDecline, PresentationService, PresentationSource,
    ReadbackLease, ReadbackService, ResidentContent, ResidentReclaim, ResidentService,
    TargetReadback,
};
pub use submission::SubmissionContext;
pub use target::{TargetIdentity, TargetKeyDivergence};
pub use texel::{
    expand_rgba8_to_texel, f16_to_f32, f16_to_unorm8, narrow_texel_to_rgba8, unorm8_to_f16,
};
