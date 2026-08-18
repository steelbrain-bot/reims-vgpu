//! Semantic boundary for the paravirtualized GPU protocol.
//!
//! [`reims_vgpu_wire`] owns byte-accurate views. This crate is the first layer
//! allowed to assign meaning to their values. Raw tags are consumed here and
//! semantic state uses the resulting types.

#![no_std]

pub mod identity;
pub mod resource;

pub use identity::{
    ByteLength, ByteOffset, GuestPhysicalAddress, GuestVirtualAddress, MappingId, ObjectRef,
    ResourceId, ResourceNamespaceId, StorageId, SurfaceBackingId, SurfaceId, TaskId,
};
pub use resource::{
    decode_object_list_entry, ObjectKind, ObjectListDecodeError, ObjectListEntry,
    OBJECT_LIST_ENTRY_LEN,
};
