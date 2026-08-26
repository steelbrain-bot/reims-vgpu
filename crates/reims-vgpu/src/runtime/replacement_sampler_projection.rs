//! Total projection from decoded or reflected sampler state to core semantics.

use crate::runtime::replacement_services::RenderTranslationDecline;

pub(crate) fn decoded_sampler(
    sampler_ref: u32,
    binding: u32,
    identity: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::SamplerObject>,
    sampler: &crate::runtime::decode::resource::SamplerDescriptor,
) -> Result<reims_vgpu_core::SamplerResource, RenderTranslationDecline> {
    use reims_vgpu_core::SamplerResource;
    Ok(SamplerResource {
        binding,
        source: reims_vgpu_core::SamplerSource::State,
        identity: Some(identity),
        min_filter: reims_vgpu_protocol::sampler_filter(sampler.min_filter).map_err(|reason| {
            RenderTranslationDecline::SamplerMinFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        mag_filter: reims_vgpu_protocol::sampler_filter(sampler.mag_filter).map_err(|reason| {
            RenderTranslationDecline::SamplerMagFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        mip_filter: reims_vgpu_protocol::sampler_mip_filter(sampler.mip_filter).map_err(
            |reason| RenderTranslationDecline::SamplerMipFilterTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        address_mode_u: reims_vgpu_protocol::sampler_address_mode(sampler.s_address).map_err(
            |reason| RenderTranslationDecline::SamplerAddressSTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        address_mode_v: reims_vgpu_protocol::sampler_address_mode(sampler.t_address).map_err(
            |reason| RenderTranslationDecline::SamplerAddressTTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        address_mode_w: reims_vgpu_protocol::sampler_address_mode(sampler.r_address).map_err(
            |reason| RenderTranslationDecline::SamplerAddressRTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        border_color: reims_vgpu_protocol::sampler_border_color(sampler.border_color).map_err(
            |reason| RenderTranslationDecline::SamplerBorderColorTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        compare_function: reims_vgpu_protocol::compare_function(sampler.compare_function).map_err(
            |reason| RenderTranslationDecline::SamplerCompareFunctionTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        lod_min: sampler.lod_min_clamp.to_bits(),
        lod_max: sampler.lod_max_clamp.to_bits(),
        max_anisotropy: sampler.max_anisotropy,
        unnormalized_coordinates: !sampler.normalized_coordinates,
    })
}

pub(crate) fn reflected_sampler(
    stage: &'static str,
    binding: u32,
    sampler: reims_vgpu_core::ReflectedStaticSamplerState,
) -> Result<reims_vgpu_core::SamplerResource, RenderTranslationDecline> {
    use reims_vgpu_core::{
        ReflectedSamplerAddressMode as Address, ReflectedSamplerBorderColor as Border,
        ReflectedSamplerCompareFunction as Compare, ReflectedSamplerCoordinates as Coordinates,
        ReflectedSamplerFilter as Filter, ReflectedSamplerMipFilter as Mip,
        ReflectedSamplerReduction as Reduction, SamplerAddressMode, SamplerBorderColor,
        SamplerCompareFunction, SamplerFilter, SamplerMipFilter, SamplerResource,
    };
    if sampler.reduction != Reduction::WeightedAverage {
        return Err(
            RenderTranslationDecline::StaticSamplerReductionUnsupported {
                stage,
                binding,
                reduction: format!("{:?}", sampler.reduction),
                raw_words: sampler.raw_words,
            },
        );
    }
    if sampler.lod_bias != 0.0 {
        return Err(RenderTranslationDecline::StaticSamplerLodBiasUnsupported {
            stage,
            binding,
            lod_bias_bits: sampler.lod_bias.to_bits(),
            raw_words: sampler.raw_words,
        });
    }
    let filter = |value, minimum| match value {
        Filter::Nearest => Ok(SamplerFilter::Nearest),
        Filter::Linear => Ok(SamplerFilter::Linear),
        Filter::Bicubic if minimum => {
            Err(RenderTranslationDecline::StaticSamplerMinFilterUnsupported { stage, binding })
        }
        Filter::Bicubic => {
            Err(RenderTranslationDecline::StaticSamplerMagFilterUnsupported { stage, binding })
        }
    };
    let address = |value| match value {
        Address::ClampToZero => SamplerAddressMode::ClampToZero,
        Address::ClampToEdge => SamplerAddressMode::ClampToEdge,
        Address::Repeat => SamplerAddressMode::Repeat,
        Address::MirroredRepeat => SamplerAddressMode::MirrorRepeat,
        Address::ClampToBorder => SamplerAddressMode::ClampToBorderColor,
    };
    Ok(SamplerResource {
        binding,
        source: reims_vgpu_core::SamplerSource::Static,
        identity: None,
        min_filter: filter(sampler.min_filter, true)?,
        mag_filter: filter(sampler.mag_filter, false)?,
        mip_filter: match sampler.mip_filter {
            Mip::None => SamplerMipFilter::NotMipmapped,
            Mip::Nearest => SamplerMipFilter::Nearest,
            Mip::Linear => SamplerMipFilter::Linear,
        },
        address_mode_u: address(sampler.address_mode_s),
        address_mode_v: address(sampler.address_mode_t),
        address_mode_w: address(sampler.address_mode_r),
        border_color: match sampler.border_color {
            Border::TransparentBlack => SamplerBorderColor::TransparentBlack,
            Border::OpaqueBlack => SamplerBorderColor::OpaqueBlack,
            Border::OpaqueWhite => SamplerBorderColor::OpaqueWhite,
        },
        compare_function: match sampler.compare_function {
            Compare::None | Compare::Never => SamplerCompareFunction::Never,
            Compare::Less => SamplerCompareFunction::Less,
            Compare::LessEqual => SamplerCompareFunction::LessEqual,
            Compare::Greater => SamplerCompareFunction::Greater,
            Compare::GreaterEqual => SamplerCompareFunction::GreaterEqual,
            Compare::Equal => SamplerCompareFunction::Equal,
            Compare::NotEqual => SamplerCompareFunction::NotEqual,
            Compare::Always => SamplerCompareFunction::Always,
        },
        lod_min: sampler.lod_min_clamp.to_bits(),
        lod_max: sampler.lod_max_clamp.to_bits(),
        max_anisotropy: sampler.max_anisotropy,
        unnormalized_coordinates: sampler.coordinates == Coordinates::Pixel,
    })
}
