//! Vulkan objects retained by canonical managed-backing lifetimes.
//!
//! The semantic [`reims_vgpu_core::ResourceLifecycleOwner`] remains the sole
//! registry. Its native value is this owned representation, so projection does
//! not reconstruct a parallel backing-to-handle map and retirement cannot drop
//! a live Vulkan object before its accepted-use obligations complete.

use crate::{
    replacement_barrier_record::{NativeBarrierTarget, ReplacementBarrierResolver},
    replacement_buffer_blit::{
        NativeBufferTarget, NativeComputeFillLimits, ReplacementBufferResolver,
    },
    replacement_image_state::{ReplacementImageKey, ReplacementImageStateOwner},
    replacement_image_transition::{
        same_subresource_range, NativeImageTarget, ReplacementImageResolver,
    },
};
use ash::vk;
use reims_vgpu_core::ResourceLifecycleOwner;
use reims_vgpu_protocol::{BackingId, RepresentationId, ResourceId, ResourceObject, TextureType};
use std::{any::Any, collections::BTreeMap, fmt, sync::Arc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementBufferAllocationError {
    DeviceLifetimeClosed,
    EmptySize,
    SizeOverflow,
    Create(vk::Result),
    NoMemoryType,
    Allocate(vk::Result),
    Bind(vk::Result),
    Map(vk::Result),
    Invalidate(vk::Result),
    Flush(vk::Result),
    GuestLoad(reims_vgpu_memory::GuestLoadError),
    GuestRegionUnsupported(reims_vgpu_core::BackingRegion),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementAdoptionError {
    DeviceLifetimeClosed,
}

#[derive(Debug)]
pub struct ReplacementAdoptionFailure<T> {
    pub reason: ReplacementAdoptionError,
    pub input: T,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplacementAttachmentViewPlan {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub range: vk::ImageSubresourceRange,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplacementShaderViewPlan {
    pub semantic: reims_vgpu_core::ResolvedTextureBindingView,
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub range: vk::ImageSubresourceRange,
}

#[derive(Clone, Debug)]
pub struct ReplacementImageCreatePlan {
    pub flags: vk::ImageCreateFlags,
    pub image_type: vk::ImageType,
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub full_range: vk::ImageSubresourceRange,
    pub pixel_format: u16,
    pub memory_class: crate::memory::MemoryClass,
    pub attachment_views: Box<[ReplacementAttachmentViewPlan]>,
    pub shader_views: Box<[ReplacementShaderViewPlan]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementTexturePlanError {
    UnknownTextureType(u8),
    UnsupportedTextureType(reims_vgpu_protocol::TextureType),
    UnknownUsageBits(u32),
    EmptyExtent,
    EmptyMipLevels,
    EmptyArrayLayers,
    ArrayLayerOverflow,
    InvalidTypeGeometry(reims_vgpu_protocol::TextureType),
    UnsupportedSampleCount(u16),
    InvalidMultisampleShape,
    MultisampleStorageUnsupported,
    UnsupportedDeclarationSwizzle {
        write_enabled: Option<bool>,
        swizzle: Option<[u8; 4]>,
    },
    UnsupportedProtectionOptions(u64),
    UnsupportedAttachmentType(reims_vgpu_protocol::TextureType),
    Format(crate::translate::reason::TranslateReason),
    MissingFormatFeatures(vk::FormatFeatureFlags),
    ShaderViewFormat(crate::translate::reason::TranslateReason),
    ShaderViewRangeOverflow(ResourceId<ResourceObject>),
    ShaderViewRangeOutsideBase(ResourceId<ResourceObject>),
    ShaderViewTypeIncompatible {
        resource: ResourceId<ResourceObject>,
        texture_type: TextureType,
    },
    ShaderViewFormatRequiresMutable(ResourceId<ResourceObject>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementTextureAllocationError {
    Plan(ReplacementTexturePlanError),
    FormatQuery(vk::Result),
    Limits(ReplacementImageFormatLimitError),
    Allocation(ReplacementImageAllocationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementTextureRequirementsError {
    DeviceLifetimeClosed,
    Plan(ReplacementTexturePlanError),
    FormatLimits(ReplacementImageFormatLimitError),
    Query(vk::Result),
    Create(vk::Result),
    ZeroRequirement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementImageFormatLimitError {
    Extent {
        requested: [u32; 3],
        maximum: [u32; 3],
    },
    MipLevels {
        requested: u32,
        maximum: u32,
    },
    ArrayLayers {
        requested: u32,
        maximum: u32,
    },
    SampleCount(vk::SampleCountFlags),
}

pub(crate) fn validate_image_format_limits(
    plan: &ReplacementImageCreatePlan,
    limits: vk::ImageFormatProperties,
) -> Result<(), ReplacementImageFormatLimitError> {
    let requested_extent = [plan.extent.width, plan.extent.height, plan.extent.depth];
    let maximum_extent = [
        limits.max_extent.width,
        limits.max_extent.height,
        limits.max_extent.depth,
    ];
    if requested_extent
        .iter()
        .zip(maximum_extent)
        .any(|(requested, maximum)| *requested > maximum)
    {
        return Err(ReplacementImageFormatLimitError::Extent {
            requested: requested_extent,
            maximum: maximum_extent,
        });
    }
    if plan.mip_levels > limits.max_mip_levels {
        return Err(ReplacementImageFormatLimitError::MipLevels {
            requested: plan.mip_levels,
            maximum: limits.max_mip_levels,
        });
    }
    if plan.array_layers > limits.max_array_layers {
        return Err(ReplacementImageFormatLimitError::ArrayLayers {
            requested: plan.array_layers,
            maximum: limits.max_array_layers,
        });
    }
    if !limits.sample_counts.contains(plan.samples) {
        return Err(ReplacementImageFormatLimitError::SampleCount(plan.samples));
    }
    Ok(())
}

#[allow(
    clippy::boxed_local,
    reason = "the lifecycle graph already returns an owned exact view set and image planning consumes that ownership"
)]
pub(crate) fn plan_owned_texture(
    declaration: reims_vgpu_protocol::TextureDeclaration,
    memory_class: crate::memory::MemoryClass,
    available: vk::FormatFeatureFlags,
    attachment_views: Box<[ReplacementAttachmentViewPlan]>,
    shader_views: Box<[reims_vgpu_core::ResolvedTextureBindingView]>,
) -> Result<ReplacementImageCreatePlan, ReplacementTexturePlanError> {
    let texture_type = declaration.texture_type;
    if let reims_vgpu_protocol::TextureType::Unknown(raw) = texture_type {
        return Err(ReplacementTexturePlanError::UnknownTextureType(raw));
    }
    let format = crate::translate::pixel::translate(declaration.pixel_format)
        .map_err(ReplacementTexturePlanError::Format)?;
    if declaration.protection_options != 0 {
        return Err(ReplacementTexturePlanError::UnsupportedProtectionOptions(
            declaration.protection_options,
        ));
    }
    if declaration.write_swizzle_enabled == Some(true)
        || declaration.swizzle.is_some_and(|raw| {
            reims_vgpu_protocol::swizzle_plan(&raw)
                .is_none_or(|plan| !reims_vgpu_protocol::swizzle_is_identity(&plan))
        })
    {
        return Err(ReplacementTexturePlanError::UnsupportedDeclarationSwizzle {
            write_enabled: declaration.write_swizzle_enabled,
            swizzle: declaration.swizzle,
        });
    }
    if declaration.width == 0 || declaration.height == 0 || declaration.depth == 0 {
        return Err(ReplacementTexturePlanError::EmptyExtent);
    }
    let mip_levels = u32::from(declaration.mipmap_level_count);
    if mip_levels == 0 {
        return Err(ReplacementTexturePlanError::EmptyMipLevels);
    }
    let declared_layers = u32::from(declaration.array_length);
    if declared_layers == 0 {
        return Err(ReplacementTexturePlanError::EmptyArrayLayers);
    }
    let (image_type, view_type, array_layers, flags) = match texture_type {
        reims_vgpu_protocol::TextureType::D1 => {
            if declaration.height != 1 || declaration.depth != 1 || declared_layers != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            (
                vk::ImageType::TYPE_1D,
                vk::ImageViewType::TYPE_1D,
                1,
                vk::ImageCreateFlags::empty(),
            )
        }
        reims_vgpu_protocol::TextureType::D1Array => {
            if declaration.height != 1 || declaration.depth != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            (
                vk::ImageType::TYPE_1D,
                vk::ImageViewType::TYPE_1D_ARRAY,
                declared_layers,
                vk::ImageCreateFlags::empty(),
            )
        }
        reims_vgpu_protocol::TextureType::D2 | reims_vgpu_protocol::TextureType::D2Multisample => {
            if declaration.depth != 1 || declared_layers != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            (
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::TYPE_2D,
                1,
                vk::ImageCreateFlags::empty(),
            )
        }
        reims_vgpu_protocol::TextureType::D2Array => {
            if declaration.depth != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            (
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::TYPE_2D_ARRAY,
                declared_layers,
                vk::ImageCreateFlags::empty(),
            )
        }
        reims_vgpu_protocol::TextureType::D2MultisampleArray => {
            if declaration.depth != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            (
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::TYPE_2D_ARRAY,
                declared_layers,
                vk::ImageCreateFlags::empty(),
            )
        }
        reims_vgpu_protocol::TextureType::Cube | reims_vgpu_protocol::TextureType::CubeArray => {
            if declaration.width != declaration.height || declaration.depth != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            if texture_type == reims_vgpu_protocol::TextureType::Cube && declared_layers != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            let array_layers = declared_layers
                .checked_mul(reims_vgpu_protocol::CUBE_FACES)
                .ok_or(ReplacementTexturePlanError::ArrayLayerOverflow)?;
            (
                vk::ImageType::TYPE_2D,
                if texture_type == reims_vgpu_protocol::TextureType::Cube {
                    vk::ImageViewType::CUBE
                } else {
                    vk::ImageViewType::CUBE_ARRAY
                },
                array_layers,
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            )
        }
        reims_vgpu_protocol::TextureType::D3 => {
            if declared_layers != 1 {
                return Err(ReplacementTexturePlanError::InvalidTypeGeometry(
                    texture_type,
                ));
            }
            (
                vk::ImageType::TYPE_3D,
                vk::ImageViewType::TYPE_3D,
                1,
                vk::ImageCreateFlags::empty(),
            )
        }
        reims_vgpu_protocol::TextureType::Buffer => {
            return Err(ReplacementTexturePlanError::UnsupportedTextureType(
                texture_type,
            ));
        }
        reims_vgpu_protocol::TextureType::Unknown(raw) => {
            return Err(ReplacementTexturePlanError::UnknownTextureType(raw));
        }
    };
    let samples = match declaration.sample_count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        other => return Err(ReplacementTexturePlanError::UnsupportedSampleCount(other)),
    };
    if matches!(
        texture_type,
        reims_vgpu_protocol::TextureType::D2Multisample
            | reims_vgpu_protocol::TextureType::D2MultisampleArray
    ) {
        if samples == vk::SampleCountFlags::TYPE_1 || mip_levels != 1 {
            return Err(ReplacementTexturePlanError::InvalidMultisampleShape);
        }
    } else if samples != vk::SampleCountFlags::TYPE_1 {
        return Err(ReplacementTexturePlanError::InvalidMultisampleShape);
    }
    if samples != vk::SampleCountFlags::TYPE_1
        && declaration.usage
            & (reims_vgpu_protocol::TEXTURE_USAGE_SHADER_WRITE
                | reims_vgpu_protocol::TEXTURE_USAGE_SHADER_ATOMIC)
            != 0
    {
        return Err(ReplacementTexturePlanError::MultisampleStorageUnsupported);
    }
    let unknown_usage = declaration.usage & !reims_vgpu_protocol::TEXTURE_USAGE_KNOWN;
    if unknown_usage != 0 {
        return Err(ReplacementTexturePlanError::UnknownUsageBits(unknown_usage));
    }
    let aspect_mask = if crate::engine::format_is_depth(format.vk) {
        if crate::engine::format_has_stencil(format.vk) {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else if format.vk == vk::Format::S8_UINT {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    let mut required = vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::TRANSFER_DST;
    let declared_usage = declaration.usage;
    let usage_unknown = declared_usage == 0;
    if declared_usage & reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ != 0
        || usage_unknown && available.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
    {
        usage |= vk::ImageUsageFlags::SAMPLED;
        required |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
    }
    if declared_usage & reims_vgpu_protocol::TEXTURE_USAGE_SHADER_WRITE != 0
        || usage_unknown && available.contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
    {
        usage |= vk::ImageUsageFlags::STORAGE;
        required |= vk::FormatFeatureFlags::STORAGE_IMAGE;
    }
    if declared_usage & reims_vgpu_protocol::TEXTURE_USAGE_SHADER_ATOMIC != 0 {
        usage |= vk::ImageUsageFlags::STORAGE;
        required |=
            vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC;
    }
    let attachment_feature = if aspect_mask == vk::ImageAspectFlags::COLOR {
        vk::FormatFeatureFlags::COLOR_ATTACHMENT
    } else {
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
    };
    if declared_usage & reims_vgpu_protocol::TEXTURE_USAGE_RENDER_TARGET != 0
        || usage_unknown && available.contains(attachment_feature)
    {
        usage |= if aspect_mask == vk::ImageAspectFlags::COLOR {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        };
        required |= attachment_feature;
    }
    let missing = required & !available;
    if !missing.is_empty() {
        return Err(ReplacementTexturePlanError::MissingFormatFeatures(missing));
    }
    let mut flags = flags;
    if declared_usage & reims_vgpu_protocol::TEXTURE_USAGE_PIXEL_FORMAT_VIEW != 0 {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    let components = crate::translate::pixel::vk_component_mapping(&format.components);
    let full_range = vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: 0,
        level_count: mip_levels,
        base_array_layer: 0,
        layer_count: array_layers,
    };
    let mut attachment_views = Vec::from(attachment_views);
    if declared_usage & reims_vgpu_protocol::TEXTURE_USAGE_RENDER_TARGET != 0 {
        let attachment_view_type = match texture_type {
            reims_vgpu_protocol::TextureType::D1 | reims_vgpu_protocol::TextureType::D1Array => {
                vk::ImageViewType::TYPE_1D
            }
            reims_vgpu_protocol::TextureType::D2
            | reims_vgpu_protocol::TextureType::D2Array
            | reims_vgpu_protocol::TextureType::D2Multisample
            | reims_vgpu_protocol::TextureType::D2MultisampleArray
            | reims_vgpu_protocol::TextureType::Cube
            | reims_vgpu_protocol::TextureType::CubeArray => vk::ImageViewType::TYPE_2D,
            reims_vgpu_protocol::TextureType::D3 => {
                return Err(ReplacementTexturePlanError::UnsupportedAttachmentType(
                    texture_type,
                ));
            }
            reims_vgpu_protocol::TextureType::Buffer => {
                return Err(ReplacementTexturePlanError::UnsupportedTextureType(
                    texture_type,
                ));
            }
            reims_vgpu_protocol::TextureType::Unknown(raw) => {
                return Err(ReplacementTexturePlanError::UnknownTextureType(raw));
            }
        };
        let aspects: &[_] =
            if aspect_mask == (vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) {
                &[vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL]
            } else {
                &[aspect_mask]
            };
        for aspect in aspects {
            for mip in 0..mip_levels {
                for layer in 0..array_layers {
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: *aspect,
                        base_mip_level: mip,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    };
                    if !same_subresource_range(range, full_range) {
                        attachment_views.push(ReplacementAttachmentViewPlan {
                            view_type: attachment_view_type,
                            format: format.vk,
                            components,
                            range,
                        });
                    }
                }
            }
        }
    }
    let mut plan = ReplacementImageCreatePlan {
        flags,
        image_type,
        view_type,
        format: format.vk,
        components,
        extent: vk::Extent3D {
            width: declaration.width,
            height: declaration.height,
            depth: declaration.depth,
        },
        mip_levels,
        array_layers,
        samples,
        tiling: vk::ImageTiling::OPTIMAL,
        usage,
        full_range,
        pixel_format: declaration.pixel_format,
        memory_class,
        attachment_views: attachment_views.into_boxed_slice(),
        shader_views: Box::new([]),
    };
    plan.shader_views = shader_views
        .iter()
        .copied()
        .map(|view| plan_shader_texture_view(&plan, view))
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    Ok(plan)
}

fn image_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if crate::engine::format_is_depth(format) {
        if crate::engine::format_has_stencil(format) {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else if format == vk::Format::S8_UINT {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

pub fn plan_shader_texture_view(
    base: &ReplacementImageCreatePlan,
    semantic: reims_vgpu_core::ResolvedTextureBindingView,
) -> Result<ReplacementShaderViewPlan, ReplacementTexturePlanError> {
    plan_shader_texture_view_for_base(
        base.flags,
        base.image_type,
        base.format,
        base.mip_levels,
        base.array_layers,
        semantic,
    )
}

fn plan_shader_texture_view_for_base(
    base_flags: vk::ImageCreateFlags,
    base_image_type: vk::ImageType,
    base_format: vk::Format,
    base_mip_levels: u32,
    base_array_layers: u32,
    semantic: reims_vgpu_core::ResolvedTextureBindingView,
) -> Result<ReplacementShaderViewPlan, ReplacementTexturePlanError> {
    let (view_type, image_type, cube_scale) = match semantic.texture_type {
        TextureType::D1 => (vk::ImageViewType::TYPE_1D, vk::ImageType::TYPE_1D, false),
        TextureType::D1Array => (
            vk::ImageViewType::TYPE_1D_ARRAY,
            vk::ImageType::TYPE_1D,
            false,
        ),
        TextureType::D2 | TextureType::D2Multisample => {
            (vk::ImageViewType::TYPE_2D, vk::ImageType::TYPE_2D, false)
        }
        TextureType::D2Array | TextureType::D2MultisampleArray => (
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageType::TYPE_2D,
            false,
        ),
        TextureType::Cube => (vk::ImageViewType::CUBE, vk::ImageType::TYPE_2D, true),
        TextureType::CubeArray => (vk::ImageViewType::CUBE_ARRAY, vk::ImageType::TYPE_2D, true),
        TextureType::D3 => (vk::ImageViewType::TYPE_3D, vk::ImageType::TYPE_3D, false),
        TextureType::Buffer | TextureType::Unknown(_) => {
            return Err(ReplacementTexturePlanError::ShaderViewTypeIncompatible {
                resource: semantic.resource,
                texture_type: semantic.texture_type,
            });
        }
    };
    if image_type != base_image_type {
        return Err(ReplacementTexturePlanError::ShaderViewTypeIncompatible {
            resource: semantic.resource,
            texture_type: semantic.texture_type,
        });
    }
    let mut base_array_layer = u32::try_from(semantic.range.slice_base)
        .map_err(|_| ReplacementTexturePlanError::ShaderViewRangeOverflow(semantic.resource))?;
    let mut layer_count = u32::try_from(semantic.range.slice_count)
        .map_err(|_| ReplacementTexturePlanError::ShaderViewRangeOverflow(semantic.resource))?;
    if cube_scale {
        base_array_layer = base_array_layer
            .checked_mul(reims_vgpu_protocol::CUBE_FACES)
            .ok_or(ReplacementTexturePlanError::ShaderViewRangeOverflow(
                semantic.resource,
            ))?;
        layer_count = layer_count
            .checked_mul(reims_vgpu_protocol::CUBE_FACES)
            .ok_or(ReplacementTexturePlanError::ShaderViewRangeOverflow(
                semantic.resource,
            ))?;
    }
    let base_mip_level = u32::try_from(semantic.range.level_base)
        .map_err(|_| ReplacementTexturePlanError::ShaderViewRangeOverflow(semantic.resource))?;
    let level_count = u32::try_from(semantic.range.level_count)
        .map_err(|_| ReplacementTexturePlanError::ShaderViewRangeOverflow(semantic.resource))?;
    let level_end = base_mip_level.checked_add(level_count).ok_or(
        ReplacementTexturePlanError::ShaderViewRangeOverflow(semantic.resource),
    )?;
    let layer_end = base_array_layer.checked_add(layer_count).ok_or(
        ReplacementTexturePlanError::ShaderViewRangeOverflow(semantic.resource),
    )?;
    if level_count == 0
        || layer_count == 0
        || level_end > base_mip_levels
        || layer_end > base_array_layers
        || matches!(
            semantic.texture_type,
            TextureType::D1 | TextureType::D2 | TextureType::D2Multisample | TextureType::D3
        ) && layer_count != 1
        || semantic.texture_type == TextureType::Cube
            && layer_count != reims_vgpu_protocol::CUBE_FACES
    {
        return Err(ReplacementTexturePlanError::ShaderViewRangeOutsideBase(
            semantic.resource,
        ));
    }
    let translated = crate::translate::pixel::translate(semantic.pixel_format)
        .map_err(ReplacementTexturePlanError::ShaderViewFormat)?;
    if translated.vk != base_format && !base_flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT) {
        return Err(
            ReplacementTexturePlanError::ShaderViewFormatRequiresMutable(semantic.resource),
        );
    }
    let components = crate::translate::pixel::vk_component_mapping(
        &semantic.swizzle.after(&translated.components),
    );
    Ok(ReplacementShaderViewPlan {
        semantic,
        view_type,
        format: translated.vk,
        components,
        range: vk::ImageSubresourceRange {
            aspect_mask: image_aspect(translated.vk),
            base_mip_level,
            level_count,
            base_array_layer,
            layer_count,
        },
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementImageAllocationError {
    DeviceLifetimeClosed,
    EmptyExtent,
    EmptyMipLevels,
    EmptyArrayLayers,
    EmptyAspect,
    CreateImage(vk::Result),
    NoMemoryType,
    Allocate(vk::Result),
    Bind(vk::Result),
    CreateView(vk::Result),
    DuplicateAttachmentView,
    CreateAttachmentView(vk::Result),
    DuplicateShaderView(ResourceId<ResourceObject>),
    CreateShaderView {
        resource: ResourceId<ResourceObject>,
        reason: vk::Result,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementShaderViewInstallError {
    DeviceLifetimeClosed,
    NotImage,
    Plan(ReplacementTexturePlanError),
    Duplicate(ResourceId<ResourceObject>),
    Create(vk::Result),
}

pub(crate) trait ReplacementRepresentationDevice: Send + Sync {
    unsafe fn destroy_buffer(&self, buffer: vk::Buffer);
    unsafe fn destroy_image_view(&self, view: vk::ImageView);
    unsafe fn destroy_image(&self, image: vk::Image);
    unsafe fn free_memory(&self, memory: vk::DeviceMemory);
}

impl ReplacementRepresentationDevice for crate::engine::context::SharedDeviceContext {
    unsafe fn destroy_buffer(&self, buffer: vk::Buffer) {
        self.device.destroy_buffer(buffer, None);
    }

    unsafe fn destroy_image_view(&self, view: vk::ImageView) {
        self.device.destroy_image_view(view, None);
    }

    unsafe fn destroy_image(&self, image: vk::Image) {
        self.device.destroy_image(image, None);
    }

    unsafe fn free_memory(&self, memory: vk::DeviceMemory) {
        self.device.free_memory(memory, None);
    }
}

enum ReplacementAllocation {
    Owned(vk::DeviceMemory),
    External(Box<dyn Any + Send + Sync>),
    None,
}

impl ReplacementAllocation {
    unsafe fn retire(self, device: &dyn ReplacementRepresentationDevice) {
        match self {
            Self::Owned(memory) => device.free_memory(memory),
            Self::External(owner) => drop(owner),
            Self::None => {}
        }
    }
}

pub struct ReplacementBufferRepresentation {
    device: Arc<dyn ReplacementRepresentationDevice>,
    allocation: ReplacementAllocation,
    owns_buffer: bool,
    target: NativeBufferTarget,
    host_staging: Option<ReplacementHostStagingBuffer>,
    linear_texture: Option<Arc<reims_vgpu_protocol::LinearTextureDescriptor>>,
    queue_families: Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
}

struct ReplacementHostStagingAllocation {
    context: Arc<crate::engine::context::SharedDeviceContext>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: usize,
    coherent: bool,
    size: usize,
    guest: reims_vgpu_memory::GuestWindow,
}

impl Drop for ReplacementHostStagingAllocation {
    fn drop(&mut self) {
        unsafe {
            self.context.device.unmap_memory(self.memory);
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
pub struct ReplacementHostStagingBuffer(Arc<ReplacementHostStagingAllocation>);

impl PartialEq for ReplacementHostStagingBuffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ReplacementHostStagingBuffer {}

impl fmt::Debug for ReplacementHostStagingBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplacementHostStagingBuffer")
            .field("buffer", &self.0.buffer)
            .field("size", &self.0.size)
            .finish_non_exhaustive()
    }
}

impl ReplacementHostStagingBuffer {
    pub(crate) fn guest(&self) -> &reims_vgpu_memory::GuestWindow {
        &self.0.guest
    }

    pub fn read_after_timeline(&self) -> Result<Vec<u8>, ReplacementBufferAllocationError> {
        if !self.0.coherent {
            unsafe {
                self.0.context.device.invalidate_mapped_memory_ranges(&[
                    vk::MappedMemoryRange::default()
                        .memory(self.0.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE),
                ])
            }
            .map_err(ReplacementBufferAllocationError::Invalidate)?;
        }
        let mut bytes = vec![0; self.0.size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.0.mapped as *const u8,
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(bytes)
    }

    /// Copy one guest-authored byte interval into the fixed mapped endpoint
    /// before a queue recording consumes it as a transfer source.
    pub fn write_guest_range_before_recording(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<(), ReplacementBufferAllocationError> {
        self.write_guest_ranges_before_recording(&[(offset, length)])
    }

    /// Prevalidate and copy a complete guest-authored range set, flushing the
    /// mapped allocation once. A malformed suffix cannot partially update the
    /// staging endpoint.
    pub fn write_guest_ranges_before_recording(
        &self,
        ranges: &[(u64, u64)],
    ) -> Result<(), ReplacementBufferAllocationError> {
        let staging_size = u64::try_from(self.0.size)
            .map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
        let copies = ranges
            .iter()
            .copied()
            .map(|(offset, length)| {
                let length = usize::try_from(length)
                    .map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
                let end = offset
                    .checked_add(
                        u64::try_from(length)
                            .map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?,
                    )
                    .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
                if end > staging_size {
                    return Err(ReplacementBufferAllocationError::SizeOverflow);
                }
                let bytes = self
                    .0
                    .guest
                    .load_range(offset, length)
                    .map_err(ReplacementBufferAllocationError::GuestLoad)?;
                let offset = usize::try_from(offset)
                    .map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
                Ok((offset, bytes))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (offset, bytes) in copies {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    (self.0.mapped as *mut u8).add(offset),
                    bytes.len(),
                );
            }
        }
        if !self.0.coherent {
            unsafe {
                self.0
                    .context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(self.0.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .map_err(ReplacementBufferAllocationError::Flush)?;
        }
        Ok(())
    }

    /// Convert canonical backing regions without asking composition code to
    /// reconstruct this endpoint's allocation bounds.
    pub fn write_guest_regions_before_recording(
        &self,
        regions: &[reims_vgpu_core::BackingRegion],
    ) -> Result<(), ReplacementBufferAllocationError> {
        let size = u64::try_from(self.0.size)
            .map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
        let ranges = regions
            .iter()
            .copied()
            .map(|region| match region {
                reims_vgpu_core::BackingRegion::Whole => Ok((0, size)),
                reims_vgpu_core::BackingRegion::Linear(range) => {
                    Ok((range.start(), range.end() - range.start()))
                }
                reims_vgpu_core::BackingRegion::Image(_) => Err(
                    ReplacementBufferAllocationError::GuestRegionUnsupported(region),
                ),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.write_guest_ranges_before_recording(&ranges)
    }
}

impl fmt::Debug for ReplacementBufferRepresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplacementBufferRepresentation")
            .field("target", &self.target)
            .field("queue_families", &self.queue_families)
            .finish_non_exhaustive()
    }
}

impl Drop for ReplacementBufferRepresentation {
    fn drop(&mut self) {
        unsafe {
            if self.owns_buffer {
                self.device.destroy_buffer(self.target.buffer);
            }
            let allocation = std::mem::replace(&mut self.allocation, ReplacementAllocation::None);
            allocation.retire(&*self.device);
        }
    }
}

pub struct ReplacementImageRepresentation {
    device: Arc<dyn ReplacementRepresentationDevice>,
    allocation: ReplacementAllocation,
    target: NativeImageTarget,
    flags: vk::ImageCreateFlags,
    attachment_views: BTreeMap<ReplacementImageViewKey, vk::ImageView>,
    shader_views: BTreeMap<ResourceId<ResourceObject>, ReplacementShaderView>,
}

#[derive(Clone, Copy, Debug)]
struct ReplacementShaderView {
    semantic: reims_vgpu_core::ResolvedTextureBindingView,
    view: vk::ImageView,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReplacementImageViewKey {
    aspect_mask: u32,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
}

impl From<vk::ImageSubresourceRange> for ReplacementImageViewKey {
    fn from(range: vk::ImageSubresourceRange) -> Self {
        Self {
            aspect_mask: range.aspect_mask.as_raw(),
            base_mip_level: range.base_mip_level,
            level_count: range.level_count,
            base_array_layer: range.base_array_layer,
            layer_count: range.layer_count,
        }
    }
}

impl fmt::Debug for ReplacementImageRepresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplacementImageRepresentation")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

impl Drop for ReplacementImageRepresentation {
    fn drop(&mut self) {
        unsafe {
            for view in std::mem::take(&mut self.shader_views).into_values() {
                self.device.destroy_image_view(view.view);
            }
            for view in std::mem::take(&mut self.attachment_views).into_values() {
                self.device.destroy_image_view(view);
            }
            self.device.destroy_image_view(self.target.view);
            self.device.destroy_image(self.target.image);
            let allocation = std::mem::replace(&mut self.allocation, ReplacementAllocation::None);
            allocation.retire(&*self.device);
        }
    }
}

#[derive(Debug)]
pub enum ReplacementNativeRepresentation {
    Buffer(ReplacementBufferRepresentation),
    Image(ReplacementImageRepresentation),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementTextureEndpointError {
    NotBuffer,
    AllocationSizeMismatch { expected: u64, actual: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementPresentImageError {
    GeometryMismatch {
        expected: [u32; 2],
        actual: [u32; 3],
    },
    PixelFormatMismatch {
        expected: u16,
        actual: u16,
    },
    Multisampled,
    MissingTransferSourceUsage,
    UnsupportedSubresourceShape,
}

/// Validate the complete native source shape consumed by the presentation
/// blit. This is a Vulkan contract check; semantic source selection remains in
/// the composition owner.
pub fn validate_replacement_present_image(
    image: NativeImageTarget,
    width: u32,
    height: u32,
    pixel_format: u16,
) -> Result<NativeImageTarget, ReplacementPresentImageError> {
    if image.extent.width != width || image.extent.height != height || image.extent.depth != 1 {
        return Err(ReplacementPresentImageError::GeometryMismatch {
            expected: [width, height],
            actual: [image.extent.width, image.extent.height, image.extent.depth],
        });
    }
    if image.pixel_format != pixel_format {
        return Err(ReplacementPresentImageError::PixelFormatMismatch {
            expected: pixel_format,
            actual: image.pixel_format,
        });
    }
    if image.samples != vk::SampleCountFlags::TYPE_1 {
        return Err(ReplacementPresentImageError::Multisampled);
    }
    if !image.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        return Err(ReplacementPresentImageError::MissingTransferSourceUsage);
    }
    if image.image_type != vk::ImageType::TYPE_2D
        || image.full_range.aspect_mask != vk::ImageAspectFlags::COLOR
        || image.full_range.base_mip_level != 0
        || image.full_range.level_count != 1
        || image.full_range.base_array_layer != 0
        || image.full_range.layer_count != 1
    {
        return Err(ReplacementPresentImageError::UnsupportedSubresourceShape);
    }
    Ok(image)
}

impl ReplacementNativeRepresentation {
    pub const fn buffer(&self) -> Option<NativeBufferTarget> {
        match self {
            Self::Buffer(buffer) => Some(buffer.target),
            Self::Image(_) => None,
        }
    }

    pub fn host_staging(&self) -> Option<ReplacementHostStagingBuffer> {
        match self {
            Self::Buffer(buffer) => buffer.host_staging.clone(),
            Self::Image(_) => None,
        }
    }

    pub fn attach_linear_texture_layout(
        &mut self,
        descriptor: Arc<reims_vgpu_protocol::LinearTextureDescriptor>,
    ) -> Result<(), ReplacementTextureEndpointError> {
        let Self::Buffer(buffer) = self else {
            return Err(ReplacementTextureEndpointError::NotBuffer);
        };
        if descriptor.allocation_size != buffer.target.size {
            return Err(ReplacementTextureEndpointError::AllocationSizeMismatch {
                expected: buffer.target.size,
                actual: descriptor.allocation_size,
            });
        }
        buffer.linear_texture = Some(descriptor);
        Ok(())
    }

    pub fn linear_texture_layout(
        &self,
    ) -> Option<&Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
        match self {
            Self::Buffer(buffer) => buffer.linear_texture.as_ref(),
            Self::Image(_) => None,
        }
    }

    pub const fn image(&self) -> Option<NativeImageTarget> {
        match self {
            Self::Image(image) => Some(image.target),
            Self::Buffer(_) => None,
        }
    }

    pub fn image_view(&self, range: vk::ImageSubresourceRange) -> Option<NativeImageTarget> {
        let Self::Image(image) = self else {
            return None;
        };
        if same_subresource_range(image.target.full_range, range) {
            return Some(image.target);
        }
        let view = *image.attachment_views.get(&range.into())?;
        Some(NativeImageTarget {
            view,
            ..image.target
        })
    }

    pub fn shader_view(
        &self,
        semantic: reims_vgpu_core::ResolvedTextureBindingView,
    ) -> Option<NativeImageTarget> {
        let Self::Image(image) = self else {
            return None;
        };
        if semantic.resource == semantic.base {
            let base_mip_level = u32::try_from(semantic.range.level_base).ok()?;
            let level_count = u32::try_from(semantic.range.level_count).ok()?;
            let mut base_array_layer = u32::try_from(semantic.range.slice_base).ok()?;
            let mut layer_count = u32::try_from(semantic.range.slice_count).ok()?;
            if matches!(
                semantic.texture_type,
                TextureType::Cube | TextureType::CubeArray
            ) {
                base_array_layer = base_array_layer.checked_mul(reims_vgpu_protocol::CUBE_FACES)?;
                layer_count = layer_count.checked_mul(reims_vgpu_protocol::CUBE_FACES)?;
            }
            let range = vk::ImageSubresourceRange {
                aspect_mask: image.target.full_range.aspect_mask,
                base_mip_level,
                level_count,
                base_array_layer,
                layer_count,
            };
            return (semantic.swizzle.is_identity()
                && semantic.pixel_format == image.target.pixel_format
                && same_subresource_range(range, image.target.full_range))
            .then_some(image.target);
        }
        let shader_view = image.shader_views.get(&semantic.resource)?;
        (shader_view.semantic == semantic).then_some(NativeImageTarget {
            view: shader_view.view,
            pixel_format: semantic.pixel_format,
            ..image.target
        })
    }

    pub(crate) fn install_shader_view(
        &mut self,
        context: &crate::engine::context::SharedDeviceContext,
        semantic: reims_vgpu_core::ResolvedTextureBindingView,
    ) -> Result<(), ReplacementShaderViewInstallError> {
        let Self::Image(image) = self else {
            return Err(ReplacementShaderViewInstallError::NotImage);
        };
        if image.shader_views.contains_key(&semantic.resource) {
            return Err(ReplacementShaderViewInstallError::Duplicate(
                semantic.resource,
            ));
        }
        let format = crate::translate::pixel::translate(image.target.pixel_format)
            .map_err(ReplacementTexturePlanError::ShaderViewFormat)
            .map_err(ReplacementShaderViewInstallError::Plan)?;
        let plan = plan_shader_texture_view_for_base(
            image.flags,
            image.target.image_type,
            format.vk,
            image.target.full_range.level_count,
            image.target.full_range.layer_count,
            semantic,
        )
        .map_err(ReplacementShaderViewInstallError::Plan)?;
        let view = unsafe {
            context.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.target.image)
                    .view_type(plan.view_type)
                    .format(plan.format)
                    .components(plan.components)
                    .subresource_range(plan.range),
                None,
            )
        }
        .map_err(ReplacementShaderViewInstallError::Create)?;
        image
            .shader_views
            .insert(semantic.resource, ReplacementShaderView { semantic, view });
        Ok(())
    }

    pub fn has_image_view_for_region(&self, region: reims_vgpu_core::BackingRegion) -> bool {
        match region {
            reims_vgpu_core::BackingRegion::Whole => self.image().is_some(),
            region => crate::replacement_image_transition::exact_image_subresource_range(region)
                .and_then(|range| self.image_view(range))
                .is_some(),
        }
    }

    fn barrier(&self, layout: Option<vk::ImageLayout>) -> NativeBarrierTarget {
        match self {
            Self::Buffer(buffer) => NativeBarrierTarget::Buffer {
                buffer: buffer.target.buffer,
                base_offset: buffer.target.base_offset,
                size: buffer.target.size,
                queue_families: buffer.queue_families,
            },
            Self::Image(image) => NativeBarrierTarget::Image {
                image: image.target.image,
                layout: layout.unwrap_or(vk::ImageLayout::UNDEFINED),
                full_range: image.target.full_range,
                queue_families: None,
            },
        }
    }
}

/// Borrowed projection of the canonical resource and image-state owners.
pub struct ReplacementRepresentationResolver<'a> {
    resources: &'a ResourceLifecycleOwner<ReplacementNativeRepresentation>,
    images: &'a ReplacementImageStateOwner,
    compute_fill_limits: Option<NativeComputeFillLimits>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplacementExecutionLimits {
    pub compute_fill: Option<NativeComputeFillLimits>,
    pub max_storage_buffer_range: u64,
    pub storage_buffer_offset_alignment: u64,
    pub max_viewports: u32,
    pub precise_occlusion_queries: bool,
    pub null_descriptors: bool,
}

/// Complete immutable native resolver borrowed from the canonical backing,
/// image-state, sampler, and device-capability owners for one EXEC assembly.
pub struct ReplacementExecutionResolver<'a> {
    representations: ReplacementRepresentationResolver<'a>,
    samplers: &'a crate::replacement_sampler::ReplacementSamplerRegistry,
    limits: ReplacementExecutionLimits,
}

impl<'a> ReplacementExecutionResolver<'a> {
    pub(crate) const fn new(
        resources: &'a ResourceLifecycleOwner<ReplacementNativeRepresentation>,
        images: &'a ReplacementImageStateOwner,
        samplers: &'a crate::replacement_sampler::ReplacementSamplerRegistry,
        limits: ReplacementExecutionLimits,
    ) -> Self {
        Self {
            representations: ReplacementRepresentationResolver::new(
                resources,
                images,
                limits.compute_fill,
            ),
            samplers,
            limits,
        }
    }
}

impl<'a> ReplacementRepresentationResolver<'a> {
    pub const fn new(
        resources: &'a ResourceLifecycleOwner<ReplacementNativeRepresentation>,
        images: &'a ReplacementImageStateOwner,
        compute_fill_limits: Option<NativeComputeFillLimits>,
    ) -> Self {
        Self {
            resources,
            images,
            compute_fill_limits,
        }
    }

    fn representation(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<&ReplacementNativeRepresentation> {
        self.resources.representation(backing, representation)
    }
}

impl ReplacementBufferResolver for ReplacementRepresentationResolver<'_> {
    fn resolve_buffer(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<NativeBufferTarget> {
        self.representation(backing, representation)?.buffer()
    }

    fn compute_fill_limits(&self) -> Option<NativeComputeFillLimits> {
        self.compute_fill_limits
    }

    fn resolve_host_staging(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<ReplacementHostStagingBuffer> {
        self.representation(backing, representation)?.host_staging()
    }

    fn resolve_linear_texture_layout(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
        self.representation(backing, representation)?
            .linear_texture_layout()
            .cloned()
    }
}

impl ReplacementImageResolver for ReplacementRepresentationResolver<'_> {
    fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
        self.representation(image.backing, image.representation)?
            .image()
    }

    fn resolve_image_view(
        &self,
        image: ReplacementImageKey,
        range: vk::ImageSubresourceRange,
    ) -> Option<NativeImageTarget> {
        self.representation(image.backing, image.representation)?
            .image_view(range)
    }

    fn resolve_texture_binding_view(
        &self,
        image: ReplacementImageKey,
        view: reims_vgpu_core::ResolvedTextureBindingView,
    ) -> Option<NativeImageTarget> {
        self.representation(image.backing, image.representation)?
            .shader_view(view)
    }
}

impl ReplacementBarrierResolver for ReplacementRepresentationResolver<'_> {
    fn resolve(&self, backing: BackingId) -> Option<NativeBarrierTarget> {
        let (representation, native) = self.resources.execution_representation(backing)?;
        let layout = native.image().and_then(|_| {
            self.images
                .state(ReplacementImageKey {
                    backing,
                    representation,
                })
                .map(|state| state.layout)
        });
        Some(native.barrier(layout))
    }
}

impl ReplacementBufferResolver for ReplacementExecutionResolver<'_> {
    fn resolve_buffer(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<NativeBufferTarget> {
        self.representations.resolve_buffer(backing, representation)
    }

    fn compute_fill_limits(&self) -> Option<NativeComputeFillLimits> {
        self.limits.compute_fill
    }

    fn resolve_host_staging(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<ReplacementHostStagingBuffer> {
        self.representations
            .resolve_host_staging(backing, representation)
    }

    fn resolve_linear_texture_layout(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
        self.representations
            .resolve_linear_texture_layout(backing, representation)
    }
}

impl ReplacementImageResolver for ReplacementExecutionResolver<'_> {
    fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
        self.representations.resolve_image(image)
    }

    fn resolve_image_view(
        &self,
        image: ReplacementImageKey,
        range: vk::ImageSubresourceRange,
    ) -> Option<NativeImageTarget> {
        self.representations.resolve_image_view(image, range)
    }

    fn resolve_texture_binding_view(
        &self,
        image: ReplacementImageKey,
        view: reims_vgpu_core::ResolvedTextureBindingView,
    ) -> Option<NativeImageTarget> {
        self.representations
            .resolve_texture_binding_view(image, view)
    }
}

impl ReplacementBarrierResolver for ReplacementExecutionResolver<'_> {
    fn resolve(&self, backing: BackingId) -> Option<NativeBarrierTarget> {
        self.representations.resolve(backing)
    }
}

impl crate::replacement_barrier_record::ReplacementBarrierResourceResolver
    for ReplacementExecutionResolver<'_>
{
    fn alias_backings(&self, resource: ResourceId<ResourceObject>) -> Option<Box<[BackingId]>> {
        self.representations
            .resources
            .graph()
            .alias_backings(resource)
    }
}

impl crate::replacement_compute::ReplacementComputeResolver for ReplacementExecutionResolver<'_> {
    fn resolve_sampler(
        &self,
        pipeline: ResourceId<reims_vgpu_protocol::ComputePipelineObject>,
        sampler: &reims_vgpu_core::SamplerResource,
    ) -> Option<crate::replacement_sampler::ReplacementSamplerLease> {
        self.samplers.compute(pipeline, sampler).ok()
    }

    fn max_storage_buffer_range(&self) -> u64 {
        self.limits.max_storage_buffer_range
    }

    fn min_storage_buffer_offset_alignment(&self) -> u64 {
        self.limits.storage_buffer_offset_alignment
    }

    fn null_descriptors(&self) -> bool {
        self.limits.null_descriptors
    }
}

impl crate::replacement_render::ReplacementRenderResolver for ReplacementExecutionResolver<'_> {
    fn resolve_sampler(
        &self,
        pipeline: ResourceId<reims_vgpu_protocol::RenderPipelineObject>,
        sampler: &reims_vgpu_core::SamplerResource,
    ) -> Option<crate::replacement_sampler::ReplacementSamplerLease> {
        self.samplers.render(pipeline, sampler).ok()
    }

    fn max_storage_buffer_range(&self) -> u64 {
        self.limits.max_storage_buffer_range
    }

    fn min_storage_buffer_offset_alignment(&self) -> u64 {
        self.limits.storage_buffer_offset_alignment
    }

    fn max_viewports(&self) -> u32 {
        self.limits.max_viewports
    }

    fn precise_occlusion_queries(&self) -> bool {
        self.limits.precise_occlusion_queries
    }

    fn null_descriptors(&self) -> bool {
        self.limits.null_descriptors
    }
}

pub(crate) unsafe fn owned_buffer(
    device: Arc<dyn ReplacementRepresentationDevice>,
    target: NativeBufferTarget,
    memory: vk::DeviceMemory,
    queue_families: Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
) -> ReplacementNativeRepresentation {
    ReplacementNativeRepresentation::Buffer(ReplacementBufferRepresentation {
        device,
        allocation: ReplacementAllocation::Owned(memory),
        owns_buffer: true,
        target,
        host_staging: None,
        linear_texture: None,
        queue_families,
    })
}

pub(crate) unsafe fn imported_guest_buffer(
    device: Arc<dyn ReplacementRepresentationDevice>,
    target: NativeBufferTarget,
    allocation: Arc<dyn Any + Send + Sync>,
) -> ReplacementNativeRepresentation {
    ReplacementNativeRepresentation::Buffer(ReplacementBufferRepresentation {
        device,
        allocation: ReplacementAllocation::External(Box::new(allocation)),
        owns_buffer: false,
        target,
        host_staging: None,
        linear_texture: None,
        queue_families: None,
    })
}

pub(crate) fn allocate_owned_buffer(
    context: Arc<crate::engine::context::SharedDeviceContext>,
    size: u64,
    usage: vk::BufferUsageFlags,
    memory_class: crate::memory::MemoryClass,
) -> Result<ReplacementNativeRepresentation, ReplacementBufferAllocationError> {
    if size == 0 {
        return Err(ReplacementBufferAllocationError::EmptySize);
    }
    let buffer = unsafe {
        context.device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(usage),
            None,
        )
    }
    .map_err(ReplacementBufferAllocationError::Create)?;
    let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
    let Some(memory_type) = context.memory_type_for(
        requirements.memory_type_bits,
        requirements.size,
        memory_class,
    ) else {
        unsafe { context.device.destroy_buffer(buffer, None) };
        return Err(ReplacementBufferAllocationError::NoMemoryType);
    };
    let memory = match unsafe {
        context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(reason) => {
            unsafe { context.device.destroy_buffer(buffer, None) };
            return Err(ReplacementBufferAllocationError::Allocate(reason));
        }
    };
    if let Err(reason) = unsafe { context.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(memory, None);
        }
        return Err(ReplacementBufferAllocationError::Bind(reason));
    }
    let device: Arc<dyn ReplacementRepresentationDevice> = context;
    Ok(unsafe {
        owned_buffer(
            device,
            NativeBufferTarget {
                buffer,
                base_offset: 0,
                accessible_size: size,
                size,
                usage,
            },
            memory,
            None,
        )
    })
}

pub(crate) fn allocate_host_staging_buffer(
    context: Arc<crate::engine::context::SharedDeviceContext>,
    size: u64,
    usage: vk::BufferUsageFlags,
    guest: reims_vgpu_memory::GuestWindow,
) -> Result<ReplacementNativeRepresentation, ReplacementBufferAllocationError> {
    if size == 0 {
        return Err(ReplacementBufferAllocationError::EmptySize);
    }
    let host_size =
        usize::try_from(size).map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
    let buffer = unsafe {
        context.device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(usage),
            None,
        )
    }
    .map_err(ReplacementBufferAllocationError::Create)?;
    let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
    let Some(memory_type) = context.memory_type_for(
        requirements.memory_type_bits,
        requirements.size,
        crate::memory::MemoryClass::Readback,
    ) else {
        unsafe { context.device.destroy_buffer(buffer, None) };
        return Err(ReplacementBufferAllocationError::NoMemoryType);
    };
    let memory = match unsafe {
        context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(reason) => {
            unsafe { context.device.destroy_buffer(buffer, None) };
            return Err(ReplacementBufferAllocationError::Allocate(reason));
        }
    };
    if let Err(reason) = unsafe { context.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(memory, None);
        }
        return Err(ReplacementBufferAllocationError::Bind(reason));
    }
    let mapped = match unsafe {
        context
            .device
            .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
    } {
        Ok(mapped) => mapped as usize,
        Err(reason) => {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.device.free_memory(memory, None);
            }
            return Err(ReplacementBufferAllocationError::Map(reason));
        }
    };
    let coherent =
        crate::memory::MappedMemoryKind::of(&context.memory_properties, memory_type).coherent;
    let staging = ReplacementHostStagingBuffer(Arc::new(ReplacementHostStagingAllocation {
        context: Arc::clone(&context),
        buffer,
        memory,
        mapped,
        coherent,
        size: host_size,
        guest,
    }));
    let device: Arc<dyn ReplacementRepresentationDevice> = context;
    Ok(ReplacementNativeRepresentation::Buffer(
        ReplacementBufferRepresentation {
            device,
            allocation: ReplacementAllocation::External(Box::new(staging.clone())),
            owns_buffer: false,
            target: NativeBufferTarget {
                buffer,
                base_offset: 0,
                accessible_size: size,
                size,
                usage,
            },
            host_staging: Some(staging),
            linear_texture: None,
            queue_families: None,
        },
    ))
}

pub(crate) fn allocate_owned_image(
    context: Arc<crate::engine::context::SharedDeviceContext>,
    plan: ReplacementImageCreatePlan,
) -> Result<ReplacementNativeRepresentation, ReplacementImageAllocationError> {
    if plan.extent.width == 0 || plan.extent.height == 0 || plan.extent.depth == 0 {
        return Err(ReplacementImageAllocationError::EmptyExtent);
    }
    if plan.mip_levels == 0 {
        return Err(ReplacementImageAllocationError::EmptyMipLevels);
    }
    if plan.array_layers == 0 {
        return Err(ReplacementImageAllocationError::EmptyArrayLayers);
    }
    if plan.full_range.aspect_mask.is_empty() {
        return Err(ReplacementImageAllocationError::EmptyAspect);
    }
    let image = unsafe {
        context.device.create_image(
            &vk::ImageCreateInfo::default()
                .flags(plan.flags)
                .image_type(plan.image_type)
                .format(plan.format)
                .extent(plan.extent)
                .mip_levels(plan.mip_levels)
                .array_layers(plan.array_layers)
                .samples(plan.samples)
                .tiling(plan.tiling)
                .usage(plan.usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }
    .map_err(ReplacementImageAllocationError::CreateImage)?;
    let requirements = unsafe { context.device.get_image_memory_requirements(image) };
    let Some(memory_type) = context.memory_type_for(
        requirements.memory_type_bits,
        requirements.size,
        plan.memory_class,
    ) else {
        unsafe { context.device.destroy_image(image, None) };
        return Err(ReplacementImageAllocationError::NoMemoryType);
    };
    let memory = match unsafe {
        context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(reason) => {
            unsafe { context.device.destroy_image(image, None) };
            return Err(ReplacementImageAllocationError::Allocate(reason));
        }
    };
    if let Err(reason) = unsafe { context.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            context.device.destroy_image(image, None);
            context.device.free_memory(memory, None);
        }
        return Err(ReplacementImageAllocationError::Bind(reason));
    }
    let view = match unsafe {
        context.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(plan.view_type)
                .format(plan.format)
                .components(plan.components)
                .subresource_range(plan.full_range),
            None,
        )
    } {
        Ok(view) => view,
        Err(reason) => {
            unsafe {
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(ReplacementImageAllocationError::CreateView(reason));
        }
    };
    let mut attachment_views = BTreeMap::<ReplacementImageViewKey, vk::ImageView>::new();
    for attachment in plan.attachment_views.iter().copied() {
        let key = ReplacementImageViewKey::from(attachment.range);
        if attachment_views.contains_key(&key) {
            unsafe {
                for view in attachment_views.into_values() {
                    context.device.destroy_image_view(view, None);
                }
                context.device.destroy_image_view(view, None);
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(ReplacementImageAllocationError::DuplicateAttachmentView);
        }
        let attachment_view = match unsafe {
            context.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(attachment.view_type)
                    .format(attachment.format)
                    .components(attachment.components)
                    .subresource_range(attachment.range),
                None,
            )
        } {
            Ok(view) => view,
            Err(reason) => {
                unsafe {
                    for view in attachment_views.into_values() {
                        context.device.destroy_image_view(view, None);
                    }
                    context.device.destroy_image_view(view, None);
                    context.device.destroy_image(image, None);
                    context.device.free_memory(memory, None);
                }
                return Err(ReplacementImageAllocationError::CreateAttachmentView(
                    reason,
                ));
            }
        };
        attachment_views.insert(key, attachment_view);
    }
    let mut shader_views = BTreeMap::<ResourceId<ResourceObject>, ReplacementShaderView>::new();
    for shader in plan.shader_views.iter().copied() {
        if shader_views.contains_key(&shader.semantic.resource) {
            unsafe {
                for shader in shader_views.into_values() {
                    context.device.destroy_image_view(shader.view, None);
                }
                for view in attachment_views.into_values() {
                    context.device.destroy_image_view(view, None);
                }
                context.device.destroy_image_view(view, None);
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(ReplacementImageAllocationError::DuplicateShaderView(
                shader.semantic.resource,
            ));
        }
        let shader_view = match unsafe {
            context.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(shader.view_type)
                    .format(shader.format)
                    .components(shader.components)
                    .subresource_range(shader.range),
                None,
            )
        } {
            Ok(view) => view,
            Err(reason) => {
                unsafe {
                    for shader in shader_views.into_values() {
                        context.device.destroy_image_view(shader.view, None);
                    }
                    for view in attachment_views.into_values() {
                        context.device.destroy_image_view(view, None);
                    }
                    context.device.destroy_image_view(view, None);
                    context.device.destroy_image(image, None);
                    context.device.free_memory(memory, None);
                }
                return Err(ReplacementImageAllocationError::CreateShaderView {
                    resource: shader.semantic.resource,
                    reason,
                });
            }
        };
        shader_views.insert(
            shader.semantic.resource,
            ReplacementShaderView {
                semantic: shader.semantic,
                view: shader_view,
            },
        );
    }
    let device: Arc<dyn ReplacementRepresentationDevice> = context;
    Ok(ReplacementNativeRepresentation::Image(
        ReplacementImageRepresentation {
            device,
            allocation: ReplacementAllocation::Owned(memory),
            target: NativeImageTarget {
                image,
                view,
                image_type: plan.image_type,
                full_range: plan.full_range,
                usage: plan.usage,
                pixel_format: plan.pixel_format,
                extent: plan.extent,
                samples: plan.samples,
            },
            flags: plan.flags,
            attachment_views,
            shader_views,
        },
    ))
}

pub(crate) unsafe fn external_buffer(
    device: Arc<dyn ReplacementRepresentationDevice>,
    target: NativeBufferTarget,
    owner: Box<dyn Any + Send + Sync>,
    queue_families: Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
) -> ReplacementNativeRepresentation {
    ReplacementNativeRepresentation::Buffer(ReplacementBufferRepresentation {
        device,
        allocation: ReplacementAllocation::External(owner),
        owns_buffer: true,
        target,
        host_staging: None,
        linear_texture: None,
        queue_families,
    })
}

pub(crate) unsafe fn owned_image(
    device: Arc<dyn ReplacementRepresentationDevice>,
    target: NativeImageTarget,
    memory: vk::DeviceMemory,
) -> ReplacementNativeRepresentation {
    ReplacementNativeRepresentation::Image(ReplacementImageRepresentation {
        device,
        allocation: ReplacementAllocation::Owned(memory),
        target,
        flags: vk::ImageCreateFlags::empty(),
        attachment_views: BTreeMap::new(),
        shader_views: BTreeMap::new(),
    })
}

pub(crate) unsafe fn external_image(
    device: Arc<dyn ReplacementRepresentationDevice>,
    target: NativeImageTarget,
    owner: Box<dyn Any + Send + Sync>,
) -> ReplacementNativeRepresentation {
    ReplacementNativeRepresentation::Image(ReplacementImageRepresentation {
        device,
        allocation: ReplacementAllocation::External(owner),
        target,
        flags: vk::ImageCreateFlags::empty(),
        attachment_views: BTreeMap::new(),
        shader_views: BTreeMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use parking_lot::Mutex;
    use reims_vgpu_core::{
        BackingRegion, RepresentationRoute, ResolvedResourceLifecycle, ResourceLifecycleEffect,
        StorageBacking, WorkingMemoryClass,
    };
    use reims_vgpu_protocol::VulkanDeviceEpochId;

    #[derive(Default)]
    struct FakeDevice(Mutex<Vec<&'static str>>);

    impl ReplacementRepresentationDevice for FakeDevice {
        unsafe fn destroy_buffer(&self, _: vk::Buffer) {
            self.0.lock().push("buffer");
        }

        unsafe fn destroy_image_view(&self, _: vk::ImageView) {
            self.0.lock().push("view");
        }

        unsafe fn destroy_image(&self, _: vk::Image) {
            self.0.lock().push("image");
        }

        unsafe fn free_memory(&self, _: vk::DeviceMemory) {
            self.0.lock().push("memory");
        }
    }

    fn buffer_target() -> NativeBufferTarget {
        NativeBufferTarget {
            buffer: vk::Buffer::from_raw(3),
            base_offset: 16,
            accessible_size: 128,
            size: 96,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        }
    }

    fn present_image_target() -> NativeImageTarget {
        NativeImageTarget {
            image: vk::Image::from_raw(4),
            view: vk::ImageView::from_raw(5),
            image_type: vk::ImageType::TYPE_2D,
            full_range: vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
            usage: vk::ImageUsageFlags::TRANSFER_SRC,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            extent: vk::Extent3D {
                width: 64,
                height: 32,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    #[test]
    fn presentation_image_gate_keeps_exact_native_shape() {
        let image = present_image_target();
        assert!(validate_replacement_present_image(
            image,
            64,
            32,
            reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        )
        .is_ok());
        assert_eq!(
            validate_replacement_present_image(
                image,
                63,
                32,
                reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            )
            .unwrap_err(),
            ReplacementPresentImageError::GeometryMismatch {
                expected: [63, 32],
                actual: [64, 32, 1],
            }
        );
        let mut no_transfer = image;
        no_transfer.usage = vk::ImageUsageFlags::SAMPLED;
        assert_eq!(
            validate_replacement_present_image(
                no_transfer,
                64,
                32,
                reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            )
            .unwrap_err(),
            ReplacementPresentImageError::MissingTransferSourceUsage
        );
    }

    fn texture_declaration() -> reims_vgpu_protocol::TextureDeclaration {
        reims_vgpu_protocol::TextureDeclaration {
            texture_type: reims_vgpu_protocol::TextureType::D2Array,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ
                | reims_vgpu_protocol::TEXTURE_USAGE_RENDER_TARGET,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            width: 16,
            height: 8,
            depth: 1,
            mipmap_level_count: 3,
            sample_count: 1,
            array_length: 2,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }
    }

    fn texture_features() -> vk::FormatFeatureFlags {
        vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::STORAGE_IMAGE
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT
    }

    #[test]
    fn texture_plan_preserves_declared_array_geometry_format_and_usage() {
        let plan = plan_owned_texture(
            texture_declaration(),
            crate::memory::MemoryClass::DeviceLocal,
            texture_features(),
            Box::new([]),
            Box::new([]),
        )
        .unwrap();
        assert_eq!(plan.image_type, vk::ImageType::TYPE_2D);
        assert_eq!(plan.view_type, vk::ImageViewType::TYPE_2D_ARRAY);
        assert_eq!(plan.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            plan.extent,
            vk::Extent3D {
                width: 16,
                height: 8,
                depth: 1
            }
        );
        assert_eq!(plan.mip_levels, 3);
        assert_eq!(plan.array_layers, 2);
        assert_eq!(plan.components.r, vk::ComponentSwizzle::R);
        assert_eq!(plan.components.g, vk::ComponentSwizzle::G);
        assert_eq!(plan.components.b, vk::ComponentSwizzle::B);
        assert_eq!(plan.components.a, vk::ComponentSwizzle::A);
        assert!(plan.usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(plan.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(!plan.usage.contains(vk::ImageUsageFlags::STORAGE));
        assert_eq!(plan.full_range.level_count, 3);
        assert_eq!(plan.full_range.layer_count, 2);
        assert_eq!(plan.attachment_views.len(), 6);
        assert!(plan.attachment_views.iter().any(|view| {
            view.range.base_mip_level == 2
                && view.range.base_array_layer == 1
                && view.range.level_count == 1
                && view.range.layer_count == 1
        }));
    }

    #[test]
    fn shader_view_plan_preserves_exact_identity_range_type_and_swizzle() {
        let base = plan_owned_texture(
            texture_declaration(),
            crate::memory::MemoryClass::DeviceLocal,
            texture_features(),
            Box::new([]),
            Box::new([]),
        )
        .unwrap();
        let semantic = reims_vgpu_core::ResolvedTextureBindingView {
            resource: ResourceId::new(9, 3),
            base: ResourceId::new(2, 1),
            range: reims_vgpu_core::ResolvedTextureViewRange {
                level_base: 1,
                level_count: 2,
                slice_base: 1,
                slice_count: 1,
            },
            texture_type: TextureType::D2Array,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            swizzle: reims_vgpu_protocol::SwizzlePlan {
                source: [
                    reims_vgpu_protocol::SwizzleSource::B,
                    reims_vgpu_protocol::SwizzleSource::G,
                    reims_vgpu_protocol::SwizzleSource::R,
                    reims_vgpu_protocol::SwizzleSource::A,
                ],
            },
        };
        let plan = plan_shader_texture_view(&base, semantic).unwrap();
        assert_eq!(plan.semantic, semantic);
        assert_eq!(plan.view_type, vk::ImageViewType::TYPE_2D_ARRAY);
        assert_eq!(plan.range.base_mip_level, 1);
        assert_eq!(plan.range.level_count, 2);
        assert_eq!(plan.range.base_array_layer, 1);
        assert_eq!(plan.range.layer_count, 1);
        assert_eq!(plan.components.r, vk::ComponentSwizzle::B);
        assert_eq!(plan.components.b, vk::ComponentSwizzle::R);
    }

    #[test]
    fn texture_plan_expands_each_declared_cube_array_element_to_six_faces() {
        let mut declaration = texture_declaration();
        declaration.texture_type = reims_vgpu_protocol::TextureType::CubeArray;
        declaration.width = 8;
        declaration.height = 8;
        let plan = plan_owned_texture(
            declaration,
            crate::memory::MemoryClass::DeviceLocal,
            texture_features(),
            Box::new([]),
            Box::new([]),
        )
        .unwrap();
        assert_eq!(plan.view_type, vk::ImageViewType::CUBE_ARRAY);
        assert_eq!(plan.array_layers, 2 * reims_vgpu_protocol::CUBE_FACES);
        assert!(plan.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE));
    }

    #[test]
    fn texture_plan_refuses_a_declared_usage_without_host_format_support() {
        let mut declaration = texture_declaration();
        declaration.usage = reims_vgpu_protocol::TEXTURE_USAGE_SHADER_WRITE;
        assert_eq!(
            plan_owned_texture(
                declaration,
                crate::memory::MemoryClass::DeviceLocal,
                vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::TRANSFER_DST,
                Box::new([]),
                Box::new([]),
            )
            .unwrap_err(),
            ReplacementTexturePlanError::MissingFormatFeatures(
                vk::FormatFeatureFlags::STORAGE_IMAGE
            )
        );
    }

    #[test]
    fn texture_plan_refuses_a_multisample_type_without_multisample_shape() {
        let mut declaration = texture_declaration();
        declaration.texture_type = reims_vgpu_protocol::TextureType::D2Multisample;
        declaration.array_length = 1;
        declaration.mipmap_level_count = 1;
        declaration.sample_count = 1;
        assert_eq!(
            plan_owned_texture(
                declaration,
                crate::memory::MemoryClass::DeviceLocal,
                texture_features(),
                Box::new([]),
                Box::new([]),
            )
            .unwrap_err(),
            ReplacementTexturePlanError::InvalidMultisampleShape
        );
    }

    #[test]
    fn texture_plan_refuses_a_declaration_swizzle_it_cannot_publish() {
        let mut declaration = texture_declaration();
        declaration.write_swizzle_enabled = Some(true);
        declaration.swizzle = Some([4, 3, 2, 5]);
        assert_eq!(
            plan_owned_texture(
                declaration,
                crate::memory::MemoryClass::DeviceLocal,
                texture_features(),
                Box::new([]),
                Box::new([]),
            )
            .unwrap_err(),
            ReplacementTexturePlanError::UnsupportedDeclarationSwizzle {
                write_enabled: Some(true),
                swizzle: Some([4, 3, 2, 5]),
            }
        );
    }

    #[test]
    fn texture_plan_retains_unsupported_protection_options() {
        let mut declaration = texture_declaration();
        declaration.protection_options = 0x1234_5678_9abc_def0;
        assert_eq!(
            plan_owned_texture(
                declaration,
                crate::memory::MemoryClass::DeviceLocal,
                texture_features(),
                Box::new([]),
                Box::new([]),
            )
            .unwrap_err(),
            ReplacementTexturePlanError::UnsupportedProtectionOptions(0x1234_5678_9abc_def0)
        );
    }

    #[test]
    fn texture_plan_maps_multisample_arrays_and_requires_atomic_format_support() {
        let mut declaration = texture_declaration();
        declaration.texture_type = reims_vgpu_protocol::TextureType::D2MultisampleArray;
        declaration.mipmap_level_count = 1;
        declaration.sample_count = 4;
        declaration.usage = reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ;
        let plan = plan_owned_texture(
            declaration,
            crate::memory::MemoryClass::DeviceLocal,
            texture_features(),
            Box::new([]),
            Box::new([]),
        )
        .unwrap();
        assert_eq!(plan.view_type, vk::ImageViewType::TYPE_2D_ARRAY);
        assert_eq!(plan.samples, vk::SampleCountFlags::TYPE_4);
        assert_eq!(plan.array_layers, 2);

        declaration.texture_type = reims_vgpu_protocol::TextureType::D2Array;
        declaration.sample_count = 1;
        declaration.usage = reims_vgpu_protocol::TEXTURE_USAGE_SHADER_ATOMIC;
        assert_eq!(
            plan_owned_texture(
                declaration,
                crate::memory::MemoryClass::DeviceLocal,
                vk::FormatFeatureFlags::TRANSFER_SRC
                    | vk::FormatFeatureFlags::TRANSFER_DST
                    | vk::FormatFeatureFlags::STORAGE_IMAGE,
                Box::new([]),
                Box::new([]),
            )
            .unwrap_err(),
            ReplacementTexturePlanError::MissingFormatFeatures(
                vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC
            )
        );
        let plan = plan_owned_texture(
            declaration,
            crate::memory::MemoryClass::DeviceLocal,
            texture_features() | vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC,
            Box::new([]),
            Box::new([]),
        )
        .unwrap();
        assert!(plan.usage.contains(vk::ImageUsageFlags::STORAGE));
    }

    #[test]
    fn texture_image_limits_are_host_answers_not_create_time_guesses() {
        let plan = plan_owned_texture(
            texture_declaration(),
            crate::memory::MemoryClass::DeviceLocal,
            texture_features(),
            Box::new([]),
            Box::new([]),
        )
        .unwrap();
        let limits = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 16,
                height: 8,
                depth: 1,
            },
            max_mip_levels: 3,
            max_array_layers: 2,
            sample_counts: vk::SampleCountFlags::TYPE_1,
            max_resource_size: u64::MAX,
        };
        assert_eq!(validate_image_format_limits(&plan, limits), Ok(()));

        let mut too_wide = plan.clone();
        too_wide.extent.width = 17;
        assert_eq!(
            validate_image_format_limits(&too_wide, limits),
            Err(ReplacementImageFormatLimitError::Extent {
                requested: [17, 8, 1],
                maximum: [16, 8, 1],
            })
        );
        let mut too_many_mips = plan.clone();
        too_many_mips.mip_levels = 4;
        assert_eq!(
            validate_image_format_limits(&too_many_mips, limits),
            Err(ReplacementImageFormatLimitError::MipLevels {
                requested: 4,
                maximum: 3,
            })
        );
        let mut too_many_layers = plan.clone();
        too_many_layers.array_layers = 3;
        assert_eq!(
            validate_image_format_limits(&too_many_layers, limits),
            Err(ReplacementImageFormatLimitError::ArrayLayers {
                requested: 3,
                maximum: 2,
            })
        );
        let mut unsupported_samples = plan;
        unsupported_samples.samples = vk::SampleCountFlags::TYPE_4;
        assert_eq!(
            validate_image_format_limits(&unsupported_samples, limits),
            Err(ReplacementImageFormatLimitError::SampleCount(
                vk::SampleCountFlags::TYPE_4
            ))
        );
    }

    #[test]
    fn canonical_backing_resolves_and_retires_the_owned_buffer() {
        let fake = Arc::new(FakeDevice::default());
        let device: Arc<dyn ReplacementRepresentationDevice> = fake.clone();
        let native =
            unsafe { owned_buffer(device, buffer_target(), vk::DeviceMemory::from_raw(7), None) };
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::NativeWorking {
                    memory: WorkingMemoryClass::DeviceLocal,
                },
                native,
            )
            .unwrap();
        let images = ReplacementImageStateOwner::new(VulkanDeviceEpochId::new(1));
        {
            let resolver = ReplacementRepresentationResolver::new(&resources, &images, None);
            assert_eq!(
                resolver.resolve_buffer(backing, representation),
                Some(buffer_target())
            );
            assert!(matches!(
                resolver.resolve(backing),
                Some(NativeBarrierTarget::Buffer {
                    buffer,
                    base_offset: 16,
                    size: 96,
                    queue_families: None,
                }) if buffer == vk::Buffer::from_raw(3)
            ));
        }
        drop(resources);
        assert_eq!(&*fake.0.lock(), &["buffer", "memory"]);
    }

    #[test]
    fn image_view_image_and_memory_retire_in_dependency_order() {
        let fake = Arc::new(FakeDevice::default());
        let device: Arc<dyn ReplacementRepresentationDevice> = fake.clone();
        let target = NativeImageTarget {
            image: vk::Image::from_raw(11),
            view: vk::ImageView::from_raw(12),
            image_type: vk::ImageType::TYPE_2D,
            full_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            usage: vk::ImageUsageFlags::SAMPLED,
            pixel_format: 80,
            extent: vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        };
        let derived_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 1,
            layer_count: 1,
        };
        let native = ReplacementNativeRepresentation::Image(ReplacementImageRepresentation {
            device,
            allocation: ReplacementAllocation::Owned(vk::DeviceMemory::from_raw(13)),
            target,
            flags: vk::ImageCreateFlags::empty(),
            attachment_views: BTreeMap::from([(
                ReplacementImageViewKey::from(derived_range),
                vk::ImageView::from_raw(14),
            )]),
            shader_views: BTreeMap::new(),
        });
        assert_eq!(
            native.image_view(derived_range).unwrap().view,
            vk::ImageView::from_raw(14)
        );
        drop(native);
        assert_eq!(&*fake.0.lock(), &["view", "view", "image", "memory"]);
    }
}
