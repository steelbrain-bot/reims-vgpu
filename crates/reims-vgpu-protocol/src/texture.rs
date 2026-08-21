//! Semantic texture declarations and mapper-backed IOSurface views.

use crate::{
    HeapObject, MapperSurfaceRef, ObjectTableRef, PlaneIndex, ResourceDecodeError, ResourceObject,
    SerializerRef, TextureRotation,
};
use reims_vgpu_wire::device_desc::{
    MapperIOSurfaceTextureError as WireMapperError, MapperIOSurfaceTextureOperation,
};
use reims_vgpu_wire::ops::texture::{TextureDescriptorBody, WideTextureDescriptorBody};
use reims_vgpu_wire::WireError;

/// Complete texture declaration after wire-format decoding.
///
/// Narrow and wide serializer forms converge here. Optional fields preserve
/// whether a form actually encoded the value; downstream code must not turn an
/// absent field into a false declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDeclaration {
    pub texture_type: u8,
    pub framebuffer_only: bool,
    pub is_drawable: bool,
    /// Present only in the wide declaration. The narrow form leaves the same
    /// bit unwritten, so it cannot be projected as `false` there.
    pub write_swizzle_enabled: Option<bool>,
    pub allow_gpu_optimized_contents: bool,
    pub usage: u32,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u16,
    pub sample_count: u16,
    pub array_length: u16,
    pub resource_options: u16,
    pub protection_options: u64,
    /// Absent from the narrow declaration; present even when the wide form
    /// carries the identity swizzle.
    pub swizzle: Option<[u8; 4]>,
}

/// Mapper-backed IOSurface texture view decoded from object-list wire tag 11.
///
/// The mapper lookup identity, serializer object reference, and selected plane
/// belong to separate namespaces. This value describes their relation without
/// claiming that the view owns the mapper mapping or IOSurface storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapperIOSurfaceTextureView {
    pub mapper_surface: MapperSurfaceRef,
    pub object: SerializerRef<ResourceObject>,
    pub declaration: TextureDeclaration,
    pub plane: PlaneIndex,
    pub rotation: Option<TextureRotation>,
}

/// A complete texture object placed in a guest heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapTextureDescriptor {
    pub object: SerializerRef<ResourceObject>,
    pub heap: ObjectTableRef<HeapObject>,
    pub declaration: TextureDeclaration,
    pub use_offset: bool,
    pub offset: u64,
}

/// Decode either declared heap-texture serializer variant.
pub fn decode_heap_texture_descriptor(
    bytes: &[u8],
) -> Result<HeapTextureDescriptor, ResourceDecodeError> {
    use reims_vgpu_wire::ops::heap_texture as wire;

    let op = reims_vgpu_wire::op(bytes, 0)
        .map_err(|_| ResourceDecodeError::ErrShort("res_heap_texture_len"))?;
    match op.opcode() {
        wire::OPCODE_NEW_HEAP_TEXTURE => {
            if bytes.len() != wire::NEW_HEAP_TEXTURE_TOTAL_LEN as usize
                || op.length() != wire::NEW_HEAP_TEXTURE_TOTAL_LEN
            {
                return Err(ResourceDecodeError::ErrShort("res_heap_texture_len"));
            }
            let body = wire::new_heap_texture(&op)
                .map_err(|_| ResourceDecodeError::ErrShort("res_heap_texture_len"))?;
            Ok(HeapTextureDescriptor {
                object: SerializerRef::new(body.object_ref.get()),
                heap: ObjectTableRef::new(body.heap_ref.get()),
                declaration: texture_declaration_from_narrow(&body.desc),
                use_offset: body.use_offset(),
                offset: body.offset.get(),
            })
        }
        wire::OPCODE_NEW_HEAP_TEXTURE_WIDE => {
            if bytes.len() != wire::NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN as usize
                || op.length() != wire::NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN
            {
                return Err(ResourceDecodeError::ErrShort("res_heap_texture_len"));
            }
            let body = wire::new_heap_texture_wide(&op)
                .map_err(|_| ResourceDecodeError::ErrShort("res_heap_texture_len"))?;
            Ok(HeapTextureDescriptor {
                object: SerializerRef::new(body.object_ref.get()),
                heap: ObjectTableRef::new(body.heap_ref.get()),
                declaration: texture_declaration_from_wide(&body.desc),
                use_offset: body.use_offset(),
                offset: body.offset.get(),
            })
        }
        _ => Err(ResourceDecodeError::ErrUnsupported(
            "res_heap_texture_opcode",
        )),
    }
}

/// Semantic refusal from decoding a mapper IOSurface texture-view envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperIOSurfaceTextureDecodeError {
    Short,
    BadLength,
    UnknownVariant,
}

/// Project a checked narrow serializer view into one semantic declaration.
pub fn texture_declaration_from_narrow(d: &TextureDescriptorBody) -> TextureDeclaration {
    TextureDeclaration {
        texture_type: d.texture_type(),
        framebuffer_only: d.framebuffer_only(),
        is_drawable: d.is_drawable(),
        write_swizzle_enabled: None,
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        usage: u32::from(d.usage()),
        pixel_format: d.pixel_format(),
        width: d.width.get(),
        height: d.height.get(),
        depth: d.depth.get(),
        mipmap_level_count: d.mipmap_level_count.get(),
        sample_count: d.sample_count.get(),
        array_length: d.array_length.get(),
        resource_options: d.resource_options.get(),
        protection_options: d.protection_options.get(),
        swizzle: None,
    }
}

/// Project a checked wide serializer view into one semantic declaration.
pub fn texture_declaration_from_wide(d: &WideTextureDescriptorBody) -> TextureDeclaration {
    TextureDeclaration {
        texture_type: d.texture_type(),
        framebuffer_only: d.framebuffer_only(),
        is_drawable: d.is_drawable(),
        write_swizzle_enabled: Some(d.write_swizzle_enabled()),
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        usage: d.usage.get(),
        pixel_format: d.pixel_format.get(),
        width: d.width.get(),
        height: d.height.get(),
        depth: d.depth.get(),
        mipmap_level_count: d.mipmap_level_count.get(),
        sample_count: d.sample_count.get(),
        array_length: d.array_length.get(),
        resource_options: d.resource_options.get(),
        protection_options: d.protection_options.get(),
        swizzle: Some([
            d.swizzle_red,
            d.swizzle_green,
            d.swizzle_blue,
            d.swizzle_alpha,
        ]),
    }
}

/// Decode wire tag 11 into the protocol-owned mapper IOSurface view.
pub fn decode_mapper_iosurface_texture_view(
    bytes: &[u8],
) -> Result<MapperIOSurfaceTextureView, MapperIOSurfaceTextureDecodeError> {
    let view =
        reims_vgpu_wire::device_desc::mapper_iosurface_texture(bytes).map_err(
            |error| match error {
                WireMapperError::Wire(WireError::Short { .. } | WireError::OutOfRange { .. }) => {
                    MapperIOSurfaceTextureDecodeError::Short
                }
                WireMapperError::Wire(
                    WireError::BadLength { .. } | WireError::CountOverflow { .. },
                )
                | WireMapperError::OuterLength { .. } => {
                    MapperIOSurfaceTextureDecodeError::BadLength
                }
                WireMapperError::UnknownVariant { .. } => {
                    MapperIOSurfaceTextureDecodeError::UnknownVariant
                }
            },
        )?;

    let (object_ref, declaration, plane, rotation) = match view.operation {
        MapperIOSurfaceTextureOperation::Legacy(body) => (
            body.object_ref.get(),
            texture_declaration_from_narrow(&body.desc),
            body.plane.get(),
            None,
        ),
        MapperIOSurfaceTextureOperation::Rotated(body) => (
            body.object_ref.get(),
            texture_declaration_from_narrow(&body.desc),
            body.plane.get(),
            Some(TextureRotation::new(body.rotation)),
        ),
        MapperIOSurfaceTextureOperation::Wide(body) => (
            body.object_ref.get(),
            texture_declaration_from_wide(&body.desc),
            body.plane.get(),
            Some(TextureRotation::new(body.rotation)),
        ),
    };

    Ok(MapperIOSurfaceTextureView {
        mapper_surface: MapperSurfaceRef::new(view.mapper_ref.get()),
        object: SerializerRef::new(object_ref),
        declaration,
        plane: PlaneIndex::new(u32::from(plane)),
        rotation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn narrow(opcode: u32, rotation: u8, unwritten: u8) -> [u8; 0x38] {
        let mut bytes = [0u8; 0x38];
        put_u64(&mut bytes, 0, 0x1122_3344_5566_7788);
        put_u32(&mut bytes, 8, opcode);
        put_u32(&mut bytes, 12, 0x30);
        put_u32(&mut bytes, 16, 0xaabb_ccdd);
        bytes[20] = 0x46;
        bytes[21] = 3;
        put_u16(&mut bytes, 22, 0x73);
        put_u32(&mut bytes, 24, 640);
        put_u32(&mut bytes, 28, 480);
        put_u32(&mut bytes, 32, 1);
        put_u16(&mut bytes, 36, 2);
        put_u16(&mut bytes, 38, 4);
        put_u16(&mut bytes, 40, 6);
        put_u16(&mut bytes, 42, 0x20);
        put_u64(&mut bytes, 44, 0x8877_6655_4433_2211);
        put_u16(&mut bytes, 52, 7);
        bytes[54] = rotation;
        bytes[55] = unwritten;
        bytes
    }

    #[test]
    fn decodes_full_width_identity_and_narrow_variants() {
        let legacy = decode_mapper_iosurface_texture_view(&narrow(0x0c, 0xa5, 0x5a)).unwrap();
        assert_eq!(legacy.mapper_surface.get(), 0x1122_3344_5566_7788);
        assert_eq!(legacy.object.get(), 0xaabb_ccdd);
        assert_eq!(legacy.plane.get(), 7);
        assert_eq!(legacy.rotation, None);
        assert_eq!(
            (legacy.declaration.width, legacy.declaration.height),
            (640, 480)
        );
        assert_eq!(legacy.declaration.write_swizzle_enabled, None);
        assert_eq!(legacy.declaration.protection_options, 0x8877_6655_4433_2211);

        let rotated = decode_mapper_iosurface_texture_view(&narrow(0x2f, 3, 0xee)).unwrap();
        assert_eq!(rotated.rotation, Some(TextureRotation::new(3)));
        assert_eq!(rotated.plane.get(), 7);
    }

    #[test]
    fn decodes_wide_plane_rotation_and_swizzle_without_stale_bytes() {
        let mut bytes = [0u8; 0x40];
        put_u64(&mut bytes, 0, 0xfedc_ba98_7654_3210);
        put_u32(&mut bytes, 8, 0x39);
        put_u32(&mut bytes, 12, 0x38);
        put_u32(&mut bytes, 16, 19);
        bytes[20] = 0xc2;
        put_u16(&mut bytes, 21, 0x73);
        put_u32(&mut bytes, 23, 0x102);
        put_u32(&mut bytes, 27, 1920);
        put_u32(&mut bytes, 31, 1080);
        put_u32(&mut bytes, 35, 1);
        put_u16(&mut bytes, 39, 1);
        put_u16(&mut bytes, 41, 1);
        put_u16(&mut bytes, 43, 1);
        put_u16(&mut bytes, 45, 0x20);
        put_u64(&mut bytes, 47, 0x1234_5678_9abc_def0);
        bytes[55..59].copy_from_slice(&[1, 2, 3, 4]);
        bytes[59] = 0xaa;
        put_u16(&mut bytes, 60, 0x1234);
        bytes[62] = 5;
        bytes[63] = 0xbb;

        let view = decode_mapper_iosurface_texture_view(&bytes).unwrap();
        assert_eq!(view.mapper_surface.get(), 0xfedc_ba98_7654_3210);
        assert_eq!(view.plane.get(), 0x1234);
        assert_eq!(view.rotation, Some(TextureRotation::new(5)));
        assert_eq!(view.declaration.swizzle, Some([1, 2, 3, 4]));
        assert_eq!(view.declaration.write_swizzle_enabled, Some(true));
        assert_eq!(view.declaration.protection_options, 0x1234_5678_9abc_def0);
    }

    #[test]
    fn refuses_unknown_or_inconsistent_nested_records() {
        let mut unknown = narrow(0x58, 0, 0);
        assert_eq!(
            decode_mapper_iosurface_texture_view(&unknown),
            Err(MapperIOSurfaceTextureDecodeError::UnknownVariant)
        );
        put_u32(&mut unknown, 8, 0x0c);
        put_u32(&mut unknown, 12, 0x28);
        assert_eq!(
            decode_mapper_iosurface_texture_view(&unknown),
            Err(MapperIOSurfaceTextureDecodeError::BadLength)
        );
        assert_eq!(
            decode_mapper_iosurface_texture_view(&unknown[..0x37]),
            Err(MapperIOSurfaceTextureDecodeError::BadLength)
        );
    }
}
