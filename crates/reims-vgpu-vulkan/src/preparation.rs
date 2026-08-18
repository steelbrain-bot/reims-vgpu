//! Typed facts and refusals produced while preparing a Vulkan request.

use reims_vgpu_observe::Decline;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Unknown,
    Vertex,
    Fragment,
}

impl ShaderStage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindTableClass {
    Buffer,
    Texture,
    Sampler,
}

impl BindTableClass {
    pub const fn table(self) -> u32 {
        match self {
            Self::Buffer => reims_vgpu_wire::ops::bind_limit::BUFFER,
            Self::Texture => reims_vgpu_wire::ops::bind_limit::TEXTURE,
            Self::Sampler => {
                crate::spirv_bind::COLOR_INPUT_BINDING_BASE
                    - crate::spirv_bind::SAMPLER_BINDING_BASE
            }
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::Sampler => "sampler",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PastTableBind {
    pub class: BindTableClass,
    pub stage: ShaderStage,
    pub index: u32,
    pub resource_ref: u32,
}

impl PastTableBind {
    pub const fn stage_name(&self) -> &'static str {
        self.stage.name()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexLoadReason {
    TypeUnsupported,
    CountOverflow,
    CountZero,
    EntryMissing,
    ObjectType,
    DescRead,
    DescDecode,
    BackingMissing,
    OffsetOverflow,
    OutOfBounds,
    ReadFail,
    BaseVertexOutOfRange,
}

impl Decline for IndexLoadReason {
    fn slug(&self) -> &'static str {
        match self {
            Self::TypeUnsupported => "draw_index_type_unsupported",
            Self::CountOverflow => "draw_index_count_overflow",
            Self::CountZero => "draw_index_count_zero",
            Self::EntryMissing => "draw_index_no_list_entry",
            Self::ObjectType => "draw_index_wrong_type",
            Self::DescRead => "draw_index_desc_read",
            Self::DescDecode => "draw_index_desc_decode",
            Self::BackingMissing => "draw_index_backing_missing",
            Self::OffsetOverflow => "draw_index_offset_overflow",
            Self::OutOfBounds => "draw_index_out_of_bounds",
            Self::ReadFail => "draw_index_read_fail",
            Self::BaseVertexOutOfRange => "draw_index_base_vertex_out_of_range",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MrtDrop {
    NonContiguousSlot,
    GeometryMismatch,
    UnknownFormat,
    NoIdentity,
    AliasesPrimary,
}

impl Decline for MrtDrop {
    fn slug(&self) -> &'static str {
        match self {
            Self::NonContiguousSlot => "mrt_drop_non_contiguous_slot",
            Self::GeometryMismatch => "mrt_drop_geometry_mismatch",
            Self::UnknownFormat => "mrt_drop_unknown_format",
            Self::NoIdentity => "mrt_drop_no_identity",
            Self::AliasesPrimary => "mrt_drop_aliases_primary",
        }
    }
}

impl MrtDrop {
    pub const fn code(self) -> u8 {
        match self {
            Self::NonContiguousSlot => 1,
            Self::GeometryMismatch => 2,
            Self::UnknownFormat => 3,
            Self::NoIdentity => 4,
            Self::AliasesPrimary => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecondaryMrtRefusal {
    pub slot: u32,
    pub reason: MrtDrop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtlbDecline {
    WrappedAirMissing {
        data_len: usize,
    },
    WrapperHeaderTruncated {
        offset: usize,
        data_len: usize,
    },
    BlobOutOfBounds {
        offset: usize,
        blob_len: u64,
        data_len: usize,
    },
}

impl Decline for MtlbDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WrappedAirMissing { .. } => "mtlb_wrapped_air_missing",
            Self::WrapperHeaderTruncated { .. } => "mtlb_wrapper_header_truncated",
            Self::BlobOutOfBounds { .. } => "mtlb_blob_out_of_bounds",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::WrappedAirMissing { data_len } => vec![("data_len", data_len.to_string())],
            Self::WrapperHeaderTruncated { offset, data_len } => vec![
                ("offset", offset.to_string()),
                ("data_len", data_len.to_string()),
            ],
            Self::BlobOutOfBounds {
                offset,
                blob_len,
                data_len,
            } => vec![
                ("offset", offset.to_string()),
                ("blob_len", blob_len.to_string()),
                ("data_len", data_len.to_string()),
            ],
        }
    }
}

reims_vgpu_observe::decline_display!(MtlbDecline);
impl std::error::Error for MtlbDecline {}
