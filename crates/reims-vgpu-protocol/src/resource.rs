//! Semantic resource namespace entries.

use core::fmt;

/// Marker for the sampler API's independent reference namespace.
pub enum SamplerObject {}
/// Marker for the depth-stencil API's independent reference namespace.
pub enum DepthStencilObject {}
/// Marker for the render-pipeline API's independent reference namespace.
pub enum RenderPipelineObject {}

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
    StateDescriptor,
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
            7 => Some(Self::StateDescriptor),
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
            Self::StateDescriptor => "state_descriptor",
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
            ObjectKind::StateDescriptor => 7,
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
}
