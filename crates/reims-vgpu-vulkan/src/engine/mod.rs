//! Native Vulkan vocabulary shared by the replacement backend.
//!
//! Device, queue, recording, resource, and presentation ownership live in the
//! replacement modules. This module contains only construction-time context,
//! host-memory import, translated descriptor vocabulary, and typed Vulkan
//! refusals; it has no process-global execution state.

#![allow(unsafe_op_in_unsafe_fn)]

pub(crate) mod context;
pub(crate) mod host_ram;
pub mod init_decline;

pub use crate::native_types::{
    BlendFactor, BlendOp, BlendStateResource, CullMode, DepthClipMode, FillMode, IndexType,
    PrimitiveTopology, SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction,
    SamplerFilter, SamplerMipFilter, StencilOp, VertexAttributeFormat, VertexStepFunction,
    VisibilityResultMode,
};

pub(crate) fn format_is_depth(format: ash::vk::Format) -> bool {
    matches!(
        format,
        ash::vk::Format::D16_UNORM
            | ash::vk::Format::X8_D24_UNORM_PACK32
            | ash::vk::Format::D32_SFLOAT
            | ash::vk::Format::D16_UNORM_S8_UINT
            | ash::vk::Format::D24_UNORM_S8_UINT
            | ash::vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub(crate) fn format_has_stencil(format: ash::vk::Format) -> bool {
    matches!(
        format,
        ash::vk::Format::D16_UNORM_S8_UINT
            | ash::vk::Format::D24_UNORM_S8_UINT
            | ash::vk::Format::D32_SFLOAT_S8_UINT
            | ash::vk::Format::S8_UINT
    )
}
