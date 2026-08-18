//! Vulkan representation of backend-independent texel storage layouts.

use ash::vk;
use reims_vgpu_protocol::TexelLayout;

/// Vulkan's linear format spelling for one guest texel layout.
pub fn vk_texel_layout(layout: TexelLayout) -> vk::Format {
    match layout {
        TexelLayout::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TexelLayout::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        TexelLayout::R8 => vk::Format::R8_UNORM,
        TexelLayout::Rg8 => vk::Format::R8G8_UNORM,
        TexelLayout::R16Float => vk::Format::R16_SFLOAT,
        TexelLayout::R32Float => vk::Format::R32_SFLOAT,
        TexelLayout::R16Unorm => vk::Format::R16_UNORM,
        TexelLayout::Rg16Unorm => vk::Format::R16G16_UNORM,
        TexelLayout::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        TexelLayout::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TexelLayout::Rg16Float => vk::Format::R16G16_SFLOAT,
        TexelLayout::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        TexelLayout::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        TexelLayout::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
    }
}

/// Vulkan's sRGB spelling for layouts that define one.
pub fn srgb_texel_layout(layout: TexelLayout) -> Option<vk::Format> {
    match layout {
        TexelLayout::Rgba8 => Some(vk::Format::R8G8B8A8_SRGB),
        TexelLayout::Bgra8 => Some(vk::Format::B8G8R8A8_SRGB),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{srgb_texel_layout, vk_texel_layout};
    use reims_vgpu_protocol::TexelLayout;

    #[test]
    fn every_semantic_layout_has_one_vulkan_storage_format() {
        for &layout in TexelLayout::ALL {
            assert_ne!(vk_texel_layout(layout), ash::vk::Format::UNDEFINED);
            assert_eq!(
                srgb_texel_layout(layout).is_some(),
                layout.has_srgb_encoding()
            );
        }
    }
}
