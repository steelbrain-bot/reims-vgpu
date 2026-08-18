//! Backend-independent texel storage vocabulary.

/// The byte layout of one guest texel, independent of any host graphics API.
///
/// This is a storage contract, not a rendering-backend format. Backends map it
/// into their own format vocabulary at the point where they create or access a
/// resident resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLayout {
    /// Four unorm8 channels in red, green, blue, alpha byte order.
    Rgba8,
    /// Four unorm8 channels in blue, green, red, alpha byte order.
    Bgra8,
    /// One unorm8 channel.
    R8,
    /// Two unorm8 channels.
    Rg8,
    /// One IEEE binary16 channel.
    R16Float,
    /// One IEEE binary32 channel.
    R32Float,
    /// One sixteen-bit normalized channel.
    R16Unorm,
    /// Two sixteen-bit normalized channels.
    Rg16Unorm,
    /// Four IEEE binary16 channels.
    Rgba16Float,
    /// Two IEEE binary16 channels.
    Rg16Float,
    /// Four sixteen-bit normalized channels.
    Rgba16Unorm,
    /// Packed 10-bit RGB and 2-bit alpha, red in the low bits.
    Rgb10a2Unorm,
    /// Packed 10-bit BGR and 2-bit alpha, blue in the low bits.
    Bgr10a2Unorm,
    /// Packed 11-bit red and green plus 10-bit blue floating-point channels.
    Rg11b10Float,
}

/// Typed texel format carried by semantic sampled and storage image requests.
///
/// Access and capability requirements are separate from this vocabulary: the
/// same stored format can be sampled on a host which cannot expose it for
/// storage writes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum StorageImageFormat {
    #[default]
    Rgba32Float,
    Rgba16Float,
    R16Float,
    Rgba16Uint,
    Rgba8Uint,
    Rgba8Sint,
    Rgba8Unorm,
    Bgra8Unorm,
    Rg16Float,
    R8Unorm,
    Rg8Unorm,
    Rgba32Uint,
    R32Uint,
    R32Sint,
    R32Float,
    Rgb9e5Ufloat,
    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,
    Rgb10a2Unorm,
    Bgr10a2Unorm,
    Rg11b10Float,
}

impl StorageImageFormat {
    /// Bytes occupied by one stored texel.
    pub const fn bytes_per_texel(self) -> usize {
        match self {
            Self::Rgba32Float | Self::Rgba32Uint => 16,
            Self::Rgba16Float | Self::Rgba16Uint | Self::Rgba16Unorm => 8,
            Self::Rg16Float | Self::Rg16Unorm => 4,
            Self::R16Float | Self::Rg8Unorm | Self::R16Unorm => 2,
            Self::R8Unorm => 1,
            Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rgba8Unorm
            | Self::Bgra8Unorm
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => 4,
        }
    }
}

impl TexelLayout {
    /// Every layout in stable table-index order.
    pub const ALL: &'static [Self] = &[
        Self::Rgba8,
        Self::Bgra8,
        Self::R8,
        Self::Rg8,
        Self::R16Float,
        Self::R32Float,
        Self::R16Unorm,
        Self::Rg16Unorm,
        Self::Rgba16Float,
        Self::Rg16Float,
        Self::Rgba16Unorm,
        Self::Rgb10a2Unorm,
        Self::Bgr10a2Unorm,
        Self::Rg11b10Float,
    ];

    /// This layout's position in [`Self::ALL`].
    pub fn index(self) -> usize {
        match self {
            Self::Rgba8 => 0,
            Self::Bgra8 => 1,
            Self::R8 => 2,
            Self::Rg8 => 3,
            Self::R16Float => 4,
            Self::R32Float => 5,
            Self::R16Unorm => 6,
            Self::Rg16Unorm => 7,
            Self::Rgba16Float => 8,
            Self::Rg16Float => 9,
            Self::Rgba16Unorm => 10,
            Self::Rgb10a2Unorm => 11,
            Self::Bgr10a2Unorm => 12,
            Self::Rg11b10Float => 13,
        }
    }

    /// Bytes occupied by one texel in guest linear storage.
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R8 => 1,
            Self::Rg8 | Self::R16Float | Self::R16Unorm => 2,
            Self::Rgba8
            | Self::Bgra8
            | Self::R32Float
            | Self::Rg16Unorm
            | Self::Rg16Float
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => 4,
            Self::Rgba16Float | Self::Rgba16Unorm => 8,
        }
    }

    /// Whether this is one of the two byte-addressable four-channel layouts.
    pub fn is_four_byte_color(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }

    /// Whether the shared CPU conversion contract defines an RGBA8 loader.
    pub fn has_cpu_loader_arm(self) -> bool {
        matches!(
            self,
            Self::Rgba8 | Self::Bgra8 | Self::R8 | Self::Rg8 | Self::Rgba16Float | Self::Rg16Float
        )
    }

    /// Whether that CPU loader necessarily loses guest-visible precision.
    pub fn cpu_loader_arm_is_lossy(self) -> bool {
        matches!(self, Self::Rgba16Float | Self::Rg16Float)
    }

    /// Whether this storage order also has an sRGB backend encoding.
    pub fn has_srgb_encoding(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }
}

#[cfg(test)]
mod tests {
    use super::{StorageImageFormat, TexelLayout};

    #[test]
    fn all_is_a_total_unique_index() {
        let mut seen = [false; TexelLayout::ALL.len()];
        for &layout in TexelLayout::ALL {
            let index = layout.index();
            assert!(index < seen.len());
            assert!(!seen[index], "duplicate index {index} for {layout:?}");
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|present| present));
    }

    #[test]
    fn lossy_loader_is_always_an_existing_loader() {
        for &layout in TexelLayout::ALL {
            assert!(!layout.cpu_loader_arm_is_lossy() || layout.has_cpu_loader_arm());
        }
    }

    #[test]
    fn semantic_image_formats_report_their_storage_width() {
        assert_eq!(StorageImageFormat::R8Unorm.bytes_per_texel(), 1);
        assert_eq!(StorageImageFormat::Rgba8Uint.bytes_per_texel(), 4);
        assert_eq!(StorageImageFormat::Rgba16Float.bytes_per_texel(), 8);
        assert_eq!(StorageImageFormat::Rgba32Float.bytes_per_texel(), 16);
    }
}
