//! Semantic boundary for the paravirtualized GPU protocol.
//!
//! [`reims_vgpu_wire`] owns byte-accurate views. This crate is the first layer
//! allowed to assign meaning to their values. Raw tags are consumed here and
//! semantic state uses the resulting types.

#![no_std]

pub mod identity;
pub mod pixel;
pub mod resource;
pub mod submission;

pub use identity::{
    BackingGeneration, ByteLength, ByteOffset, ContentVersion, GuestPhysicalAddress,
    GuestVirtualAddress, MappingId, ObjectRef, PlaneIndex, ResourceId, ResourceNamespaceId,
    StorageId, SubmissionId, SurfaceBackingId, SurfaceId, TaskId,
};
pub use pixel::{StorageImageFormat, TexelLayout};
pub use resource::{
    decode_object_list_entry, ComputePipelineObject, DepthStencilObject, EventObject, FenceObject,
    FunctionObject, ObjectKind, ObjectListDecodeError, ObjectListEntry, RenderPipelineObject,
    SamplerObject, OBJECT_LIST_ENTRY_LEN,
};
pub use submission::{
    HeapObject, IndirectCommandBufferObject, ResourceObject, ResourceValidity, SegmentBoundary,
    SegmentKind, SubmissionIdentity, SubmissionResourceUse,
};
