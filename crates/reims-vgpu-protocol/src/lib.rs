//! Semantic boundary for the paravirtualized GPU protocol.
//!
//! [`reims_vgpu_wire`] owns byte-accurate views. This crate is the first layer
//! allowed to assign meaning to their values. Raw tags are consumed here and
//! semantic state uses the resulting types.

#![no_std]

pub mod blit;
pub mod geometry;
pub mod identity;
pub mod pixel;
pub mod resource;
pub mod resource_state;
pub mod submission;
pub mod vertex_step;

pub use blit::{
    BlitCommand, BlitCopyKind, BlitFillSource, BlitKind, BlitPoint, BlitRefKind, BlitSize,
};
pub use geometry::{
    mip_extent, tight_image_bytes, tight_image_layout, tight_layered_image_bytes, Extent3,
};
pub use identity::{
    BackingGeneration, ByteLength, ByteOffset, ContentVersion, GuestPhysicalAddress,
    GuestVirtualAddress, MapperSurfaceRef, MappingId, ObjectTableRef, PlaneIndex, ResourceId,
    ResourceNamespaceId, SerializerRef, StorageId, SubmissionId, SurfaceBackingId, SurfaceId,
    TaskId,
};
pub use pixel::{
    apply_swizzle_rgba8, swizzle_identity, swizzle_is_identity, swizzle_plan, ImageFormat,
    StorageImageFormat, SwizzlePlan, SwizzleSource, TexelLayout, TransferFunction,
};
pub use resource::{
    decode_object_list_entry, ColorWriteMask, ComputePipelineObject, DepthStencilObject,
    EventObject, FenceObject, FunctionObject, ObjectKind, ObjectListDecodeError, ObjectListEntry,
    RenderPipelineObject, SamplerObject, MAX_COLOR_ATTACHMENTS, MTL_COLOR_WRITE_MASK_ALL,
    MTL_COLOR_WRITE_MASK_ALPHA, MTL_COLOR_WRITE_MASK_BLUE, MTL_COLOR_WRITE_MASK_GREEN,
    MTL_COLOR_WRITE_MASK_NONE, MTL_COLOR_WRITE_MASK_RED, OBJECT_LIST_ENTRY_LEN,
};
pub use resource_state::ResourceValidityOps;
pub use submission::{
    HeapObject, IndirectCommandBufferObject, ResourceObject, ResourceValidity, SegmentBoundary,
    SegmentKind, SubmissionIdentity, SubmissionResourceUse,
};
pub use vertex_step::{
    step_rate_in_contract, MTL_VERTEX_STEP_FUNCTION_CONSTANT,
    MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE, MTL_VERTEX_STEP_FUNCTION_PER_PATCH,
    MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT, MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
};
