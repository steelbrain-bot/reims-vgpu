//! Backend-neutral decoded state consumed by Vulkan translation and native
//! object construction.

pub use reims_vgpu_core::{
    BlendFactor, BlendOp, BlendStateResource, CullMode, DepthClipMode, FillMode, IndexType,
    PrimitiveTopology, SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction,
    SamplerFilter, SamplerMipFilter, SamplerResource, StencilOp, VertexAttributeFormat,
    VertexStepFunction, VisibilityResultMode,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct SamplerStateKey {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    pub compare_function: SamplerCompareFunction,
    pub lod_min: u32,
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerStateDecline {
    PixelMixedFilters,
    PixelMipmapped,
    PixelAddressMode,
    PixelAnisotropy,
    UnnormalizedCompare,
}

impl std::fmt::Display for SamplerStateDecline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::PixelMixedFilters => "sampler_pixel_mixed_filters",
            Self::PixelMipmapped => "sampler_pixel_mipmapped",
            Self::PixelAddressMode => "sampler_pixel_address_mode",
            Self::PixelAnisotropy => "sampler_pixel_anisotropy",
            Self::UnnormalizedCompare => "sampler_unnormalized_compare",
        })
    }
}

pub(crate) fn effective_sampler_state(
    sampler: &SamplerResource,
) -> Result<SamplerStateKey, SamplerStateDecline> {
    effective_sampler_state_key(SamplerStateKey {
        min_filter: sampler.min_filter,
        mag_filter: sampler.mag_filter,
        mip_filter: sampler.mip_filter,
        address_mode_u: sampler.address_mode_u,
        address_mode_v: sampler.address_mode_v,
        address_mode_w: sampler.address_mode_w,
        border_color: sampler.border_color,
        compare_function: sampler.compare_function,
        lod_min: sampler.lod_min,
        lod_max: sampler.lod_max,
        max_anisotropy: sampler.max_anisotropy,
        unnormalized_coordinates: sampler.unnormalized_coordinates,
    })
}

pub(crate) fn effective_sampler_state_key(
    mut key: SamplerStateKey,
) -> Result<SamplerStateKey, SamplerStateDecline> {
    if !key.unnormalized_coordinates {
        return Ok(key);
    }
    if key.min_filter != key.mag_filter {
        return Err(SamplerStateDecline::PixelMixedFilters);
    }
    if key.mip_filter != SamplerMipFilter::NotMipmapped {
        return Err(SamplerStateDecline::PixelMipmapped);
    }
    if !matches!(
        key.address_mode_u,
        SamplerAddressMode::ClampToEdge
            | SamplerAddressMode::ClampToZero
            | SamplerAddressMode::ClampToBorderColor
    ) || !matches!(
        key.address_mode_v,
        SamplerAddressMode::ClampToEdge
            | SamplerAddressMode::ClampToZero
            | SamplerAddressMode::ClampToBorderColor
    ) {
        return Err(SamplerStateDecline::PixelAddressMode);
    }
    if key.max_anisotropy != 1 {
        return Err(SamplerStateDecline::PixelAnisotropy);
    }
    if key.compare_function != SamplerCompareFunction::Never {
        return Err(SamplerStateDecline::UnnormalizedCompare);
    }
    key.lod_min = 0.0f32.to_bits();
    key.lod_max = 0.0f32.to_bits();
    Ok(key)
}
