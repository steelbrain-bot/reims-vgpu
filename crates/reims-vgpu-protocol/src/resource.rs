//! Semantic resource namespace entries.

use core::fmt;

/// Number of color attachments carried by one render pass and pipeline.
///
/// Derived from the wire array width so pass decoding, pipeline decoding, and
/// backend allocation cannot acquire independent bounds.
pub const MAX_COLOR_ATTACHMENTS: usize =
    reims_vgpu_wire::ops::render_pass::RENDER_PASS_COLOR_ATTACHMENTS;

// `MTLColorWriteMask` bits, in Metal's alpha-first ordering.
pub const MTL_COLOR_WRITE_MASK_NONE: u32 = 0;
pub const MTL_COLOR_WRITE_MASK_ALPHA: u32 = 1 << 0;
pub const MTL_COLOR_WRITE_MASK_BLUE: u32 = 1 << 1;
pub const MTL_COLOR_WRITE_MASK_GREEN: u32 = 1 << 2;
pub const MTL_COLOR_WRITE_MASK_RED: u32 = 1 << 3;
pub const MTL_COLOR_WRITE_MASK_ALL: u32 = 0xf;

/// Channels written by one render-pipeline color attachment.
///
/// Default is `all`, matching Metal descriptor semantics; a derived zero
/// default would instead suppress every color write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask {
    bits: u32,
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self {
            bits: MTL_COLOR_WRITE_MASK_ALL,
        }
    }
}

impl ColorWriteMask {
    pub fn new(bits: u32) -> Option<Self> {
        (bits <= MTL_COLOR_WRITE_MASK_ALL).then_some(Self { bits })
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }
}

/// One semantic color-attachment entry in a render-pipeline descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineColorAttachment {
    pub slot: u32,
    pub has_pixel_format: bool,
    pub pixel_format: u32,
    pub blending_enabled: bool,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
    /// Channels written independently of whether blending is enabled.
    pub write_mask: ColorWriteMask,
}

/// Marker for the sampler API's independent reference namespace.
pub enum SamplerObject {}
/// Marker for the depth-stencil API's independent reference namespace.
pub enum DepthStencilObject {}
/// Marker for the render-pipeline API's independent reference namespace.
pub enum RenderPipelineObject {}
/// Marker for the compute-pipeline API's independent reference namespace.
pub enum ComputePipelineObject {}
/// Marker for the function API's independent reference namespace.
pub enum FunctionObject {}
/// Marker for the fence API's independent reference namespace.
#[derive(Debug)]
pub enum FenceObject {}
/// Marker for the event API's independent reference namespace.
#[derive(Debug)]
pub enum EventObject {}

/// Bytes in one object-list entry.
pub const OBJECT_LIST_ENTRY_LEN: usize = 12;
const OBJECT_TYPE_MASK: u32 = 0xff;
const OBJECT_DESC_LEN_SHIFT: u32 = 8;

/// Semantic class selected by an object-list tag.
///
/// Two texture wire encodings normalize to [`Self::Texture`]. The raw tag is
/// retained privately by [`ObjectListEntry`] solely for boundary diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    Buffer,
    Texture,
    SurfaceBacking,
    IOSurfacePlaneView,
    Function,
    SerializerResource,
    TextureView,
    MemorylessTexture,
    IOSurfaceTexture,
    DualPlaneTexture,
    ResourceHandle,
    HeapBuffer,
    ExternalBuffer,
}

impl ObjectKind {
    pub const fn from_wire_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Buffer),
            2 | 3 => Some(Self::Texture),
            4 => Some(Self::SurfaceBacking),
            5 => Some(Self::IOSurfacePlaneView),
            6 => Some(Self::Function),
            7 => Some(Self::SerializerResource),
            8 => Some(Self::TextureView),
            9 => Some(Self::MemorylessTexture),
            11 => Some(Self::IOSurfaceTexture),
            12 => Some(Self::DualPlaneTexture),
            13 => Some(Self::ResourceHandle),
            14 => Some(Self::HeapBuffer),
            15 => Some(Self::ExternalBuffer),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::SurfaceBacking => "surface_backing",
            Self::IOSurfacePlaneView => "iosurface_plane_view",
            Self::Function => "function",
            Self::SerializerResource => "serializer_resource",
            Self::TextureView => "texture_view",
            Self::MemorylessTexture => "memoryless_texture",
            Self::IOSurfaceTexture => "iosurface_texture",
            Self::DualPlaneTexture => "dual_plane_texture",
            Self::ResourceHandle => "resource_handle",
            Self::HeapBuffer => "heap_buffer",
            Self::ExternalBuffer => "external_buffer",
        }
    }
}

impl fmt::Display for ObjectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One decoded task-local object namespace entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectListEntry {
    pub kind: ObjectKind,
    pub descriptor_length: u32,
    pub descriptor_gva: u64,
    wire_tag: u8,
}

impl ObjectListEntry {
    /// Construct a semantic entry outside the wire decoder.
    ///
    /// Production guest entries should use [`decode_object_list_entry`]. This
    /// constructor exists for already-semantic producers such as scripted
    /// executors and tests, and chooses the canonical encoding when a semantic
    /// kind has more than one wire representation.
    pub const fn new(kind: ObjectKind, descriptor_length: u32, descriptor_gva: u64) -> Self {
        let wire_tag = match kind {
            ObjectKind::Buffer => 1,
            ObjectKind::Texture => 2,
            ObjectKind::SurfaceBacking => 4,
            ObjectKind::IOSurfacePlaneView => 5,
            ObjectKind::Function => 6,
            ObjectKind::SerializerResource => 7,
            ObjectKind::TextureView => 8,
            ObjectKind::MemorylessTexture => 9,
            ObjectKind::IOSurfaceTexture => 11,
            ObjectKind::DualPlaneTexture => 12,
            ObjectKind::ResourceHandle => 13,
            ObjectKind::HeapBuffer => 14,
            ObjectKind::ExternalBuffer => 15,
        };
        Self {
            kind,
            descriptor_length,
            descriptor_gva,
            wire_tag,
        }
    }

    /// The original numeric tag, for boundary diagnostics and fixture parity.
    pub const fn wire_tag(self) -> u8 {
        self.wire_tag
    }
}

/// A typed refusal from the object-list boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectListDecodeError {
    Short { actual: usize },
    UnknownKind { wire_tag: u8 },
}

/// Parse one object-list entry and consume its numeric class tag.
pub fn decode_object_list_entry(bytes: &[u8]) -> Result<ObjectListEntry, ObjectListDecodeError> {
    if bytes.len() < OBJECT_LIST_ENTRY_LEN {
        return Err(ObjectListDecodeError::Short {
            actual: bytes.len(),
        });
    }
    let first = u32::from_le_bytes(bytes[0..4].try_into().expect("length checked"));
    let wire_tag = (first & OBJECT_TYPE_MASK) as u8;
    let kind = ObjectKind::from_wire_tag(wire_tag)
        .ok_or(ObjectListDecodeError::UnknownKind { wire_tag })?;
    Ok(ObjectListEntry {
        kind,
        descriptor_length: first >> OBJECT_DESC_LEN_SHIFT,
        descriptor_gva: u64::from_le_bytes(bytes[4..12].try_into().expect("length checked")),
        wire_tag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tag: u8, len: u32, gva: u64) -> [u8; OBJECT_LIST_ENTRY_LEN] {
        let mut bytes = [0u8; OBJECT_LIST_ENTRY_LEN];
        bytes[0..4].copy_from_slice(&(u32::from(tag) | (len << 8)).to_le_bytes());
        bytes[4..12].copy_from_slice(&gva.to_le_bytes());
        bytes
    }

    #[test]
    fn texture_encodings_normalize_at_the_boundary() {
        let primary = decode_object_list_entry(&entry(2, 32, 0x4000)).unwrap();
        let alternate = decode_object_list_entry(&entry(3, 32, 0x5000)).unwrap();
        assert_eq!(primary.kind, ObjectKind::Texture);
        assert_eq!(alternate.kind, ObjectKind::Texture);
        assert_eq!(primary.wire_tag(), 2);
        assert_eq!(alternate.wire_tag(), 3);
    }

    #[test]
    fn iosurface_texture_has_a_semantic_name() {
        let decoded = decode_object_list_entry(&entry(11, 0x38, 0x6000)).unwrap();
        assert_eq!(decoded.kind, ObjectKind::IOSurfaceTexture);
        assert_eq!(decoded.kind.name(), "iosurface_texture");
    }

    #[test]
    fn unknown_tags_are_refused_at_the_boundary() {
        assert_eq!(
            decode_object_list_entry(&entry(0xfe, 16, 0x7000)),
            Err(ObjectListDecodeError::UnknownKind { wire_tag: 0xfe })
        );
    }

    #[test]
    fn color_write_masks_are_total_over_the_contract_bits() {
        assert_eq!(ColorWriteMask::default().bits(), MTL_COLOR_WRITE_MASK_ALL);
        for bits in 0..=MTL_COLOR_WRITE_MASK_ALL {
            assert_eq!(ColorWriteMask::new(bits).unwrap().bits(), bits);
        }
        assert_eq!(ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALL + 1), None);
    }
}
