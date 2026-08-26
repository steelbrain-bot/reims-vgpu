//! Arm mapper-service identity and capture lifecycle.
//!
//! Mapper references are 64-bit service identities. They do not name task
//! objects, registered surface backings, or GPU page-table mappings.

use std::collections::BTreeMap;

use reims_vgpu_protocol::{MapperRequestKind, MapperResolvedSurfaceId, MapperSurfaceRef};

/// Complete mapper-backed storage plan proven from one MAP capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapperSurfaceBacking {
    pub surface: MapperResolvedSurfaceId,
    pub mapping_internal: u64,
    pub descriptor_kva: u64,
    pub descriptor: Box<[u8]>,
    pub geometry: reims_vgpu_protocol::DeviceSurfaceRecord,
    /// Whole-surface single-plane format. Biplanar allocations deliberately
    /// have none; each mapper texture view supplies its own plane format.
    pub metal_pixel_format: Option<u16>,
    pub pages: reims_vgpu_paging::mapper::PageTablePlan,
    pub page_shift: u32,
    pub footprint: reims_vgpu_memory::GuestPageFootprint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperSurfaceBackingError {
    NotMap(MapperRequestKind),
    InvalidPageShift(u32),
    Internal(reims_vgpu_paging::mapper::Status),
    DescriptorRead { address: u64 },
    DescriptorDecode,
    EmptyGeometry,
    UnsupportedPixelFormat(u32),
    InvalidSpan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapperTexturePlanePlan {
    pub mapper_surface: MapperSurfaceRef,
    pub surface: MapperResolvedSurfaceId,
    pub plane: reims_vgpu_protocol::PlaneIndex,
    pub format: u16,
    pub width: u32,
    pub height: u32,
    pub allocation_offset: u64,
    pub row_pitch: u64,
    pub visible_end: u64,
    pub footprint: reims_vgpu_memory::GuestPageFootprint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperTexturePlaneError {
    SurfaceUnbound(MapperSurfaceRef),
    BackingAbsent(MapperResolvedSurfaceId),
    UnsupportedTextureShape,
    PlaneOutOfBounds { plane: u32, count: u8 },
    GeometryMismatch,
    FormatWidthMismatch,
    RangeOverflow,
}

/// Resolve all KVA-owned mapper state before any service relation changes.
///
/// The replacement path deliberately requires the device descriptor and a
/// complete page plan. A missing descriptor, unknown single-plane projection,
/// or incomplete page table is a typed refusal rather than partially published
/// geometry or a retained prior plan.
pub fn resolve_mapper_surface_backing(
    mem: &dyn reims_vgpu_paging::mapper::PagesMemory,
    surface: MapperResolvedSurfaceId,
    capture: MapperCapture,
    page_shift: u32,
) -> Result<MapperSurfaceBacking, MapperSurfaceBackingError> {
    if capture.request_kind != MapperRequestKind::Map {
        return Err(MapperSurfaceBackingError::NotMap(capture.request_kind));
    }
    let page_size = 1u64
        .checked_shl(page_shift)
        .ok_or(MapperSurfaceBackingError::InvalidPageShift(page_shift))?;
    let fields = reims_vgpu_paging::mapper::read_mapper_internal(
        mem,
        capture.mapping_internal,
        capture.mapper_device_kva != 0,
        capture.mapper_device_kva,
    )
    .map_err(MapperSurfaceBackingError::Internal)?;
    let status = reims_vgpu_paging::mapper::validate_mapper_internal(mem, surface.get(), &fields);
    if status != reims_vgpu_paging::mapper::Status::Ok {
        return Err(MapperSurfaceBackingError::Internal(status));
    }
    let descriptor_kva =
        reims_vgpu_paging::mapper::read_internal_desc_ptr(mem, capture.mapping_internal)
            .map_err(MapperSurfaceBackingError::Internal)?;
    let mut descriptor = vec![0u8; reims_vgpu_protocol::DEVICE_DESC_LEN];
    if !mem.read(descriptor_kva, &mut descriptor) {
        return Err(MapperSurfaceBackingError::DescriptorRead {
            address: descriptor_kva,
        });
    }
    let geometry = reims_vgpu_protocol::decode_device_surface(&descriptor)
        .ok_or(MapperSurfaceBackingError::DescriptorDecode)?;
    if geometry.width == 0 || geometry.height == 0 {
        return Err(MapperSurfaceBackingError::EmptyGeometry);
    }
    let projected_format = reims_vgpu_protocol::device_desc_format_to_mtl(geometry.pixel_format);
    let biplanar = reims_vgpu_protocol::iosurface_fourcc_is_biplanar(geometry.pixel_format);
    if projected_format == 0 && !biplanar {
        return Err(MapperSurfaceBackingError::UnsupportedPixelFormat(
            geometry.pixel_format,
        ));
    }
    let metal_pixel_format = (projected_format != 0).then_some(projected_format);
    let span = match metal_pixel_format {
        Some(format) => reims_vgpu_protocol::mapping_span_bound(
            Some(&descriptor),
            format,
            geometry.width,
            geometry.height,
        )
        .ok_or(MapperSurfaceBackingError::InvalidSpan)?,
        None => u64::from(geometry.alloc_size),
    };
    if span == 0 {
        return Err(MapperSurfaceBackingError::InvalidSpan);
    }
    let min_size = span.max(u64::from(geometry.alloc_size)).max(page_size);
    let pages = reims_vgpu_paging::mapper::build_table_plan(
        mem,
        surface.get(),
        &fields,
        min_size,
        page_shift,
    )
    .map_err(MapperSurfaceBackingError::Internal)?;
    let physical_pages = pages
        .entries
        .iter()
        .map(|entry| {
            reims_vgpu_paging::geometry::mapper_entry_gpa(*entry, page_shift)
                .expect("the page-table plan validated every retained entry")
        })
        .collect::<Vec<_>>();
    let footprint = reims_vgpu_memory::GuestPageFootprint::new(physical_pages.into(), page_size)
        .expect("the validated nonempty page plan and power-of-two page size form a footprint");
    Ok(MapperSurfaceBacking {
        surface,
        mapping_internal: capture.mapping_internal,
        descriptor_kva,
        descriptor: descriptor.into_boxed_slice(),
        geometry,
        metal_pixel_format,
        pages,
        page_shift,
        footprint,
    })
}

/// Directed mapper capture published with one producer write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperCapture {
    /// Producer index that published this request (entry = producer - 1).
    pub producer: u32,
    pub mapper_device_kva: u64,
    pub request_kind: MapperRequestKind,
    /// Guest kernel address of the mapper-internal object.
    pub mapping_internal: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperCapturePublicationError {
    ConflictingPublication { producer: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MapperRequestReadPlan {
    pub ring_base: u64,
    pub producer: u32,
    pub gpa: u64,
    pub byte_len: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapperRequestReadError {
    NoPublishedEntry,
    AddressOverflow { ring_base: u64, producer: u32 },
}

pub fn resolve_mapper_request_read(
    ring_base: u64,
    producer: u32,
) -> Result<MapperRequestReadPlan, MapperRequestReadError> {
    let offset = reims_vgpu_protocol::mapper_request_published_entry_offset(producer)
        .ok_or(MapperRequestReadError::NoPublishedEntry)?;
    let gpa = ring_base
        .checked_add(offset)
        .ok_or(MapperRequestReadError::AddressOverflow {
            ring_base,
            producer,
        })?;
    Ok(MapperRequestReadPlan {
        ring_base,
        producer,
        gpa,
        byte_len: reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN as u64,
    })
}

/// Device-local state owned by the arm mapper service.
#[derive(Debug, Default)]
pub struct MapperService {
    pending_captures: BTreeMap<u32, MapperCapture>,
    device_kva: u64,
    surfaces: BTreeMap<MapperSurfaceRef, MapperResolvedSurfaceId>,
    backings: BTreeMap<MapperResolvedSurfaceId, MapperSurfaceBacking>,
}

impl MapperService {
    pub fn publish_capture(
        &mut self,
        capture: MapperCapture,
    ) -> Result<(), MapperCapturePublicationError> {
        match self.pending_captures.get(&capture.producer) {
            Some(existing) if *existing == capture => Ok(()),
            Some(_) => Err(MapperCapturePublicationError::ConflictingPublication {
                producer: capture.producer,
            }),
            None => {
                self.pending_captures.insert(capture.producer, capture);
                Ok(())
            }
        }
    }

    /// Consume a capture only when it belongs to this published ring entry.
    pub fn take_capture(&mut self, producer: u32) -> Option<MapperCapture> {
        self.pending_captures.remove(&producer)
    }

    pub fn restore_capture(
        &mut self,
        capture: MapperCapture,
    ) -> Result<(), MapperCapturePublicationError> {
        self.publish_capture(capture)
    }

    /// Zero cannot erase an already established mapper-device identity.
    pub fn observe_device(&mut self, device_kva: u64) {
        if device_kva != 0 {
            self.device_kva = device_kva;
        }
    }

    pub fn device_kva(&self) -> u64 {
        self.device_kva
    }

    pub fn map_surface(
        &mut self,
        mapper_surface: MapperSurfaceRef,
        surface: MapperResolvedSurfaceId,
    ) -> bool {
        if mapper_surface.get() == 0 {
            return false;
        }
        self.surfaces.insert(mapper_surface, surface);
        true
    }

    pub fn resolve_surface(
        &self,
        mapper_surface: MapperSurfaceRef,
    ) -> Option<MapperResolvedSurfaceId> {
        self.surfaces.get(&mapper_surface).copied()
    }

    /// Return every mapper-service identity explicitly related to one resolved
    /// surface. The ordered map makes the returned order deterministic.
    pub fn mapper_surfaces_for(&self, surface: MapperResolvedSurfaceId) -> Box<[MapperSurfaceRef]> {
        self.surfaces
            .iter()
            .filter_map(|(&mapper, &resolved)| (resolved == surface).then_some(mapper))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Publish one completely resolved plan, returning the exact superseded
    /// incarnation to its native-retirement owner.
    pub fn publish_backing(
        &mut self,
        backing: MapperSurfaceBacking,
    ) -> Option<MapperSurfaceBacking> {
        self.backings.insert(backing.surface, backing)
    }

    pub fn backing(&self, surface: MapperResolvedSurfaceId) -> Option<&MapperSurfaceBacking> {
        self.backings.get(&surface)
    }

    /// Project a mapper texture view onto the exact plane of its published
    /// allocation without consulting object ids, dimensions, or page content
    /// as a substitute for the mapper-service relation.
    pub fn texture_plane_plan(
        &self,
        descriptor: &reims_vgpu_protocol::MapperIOSurfaceTextureView,
    ) -> Result<MapperTexturePlanePlan, MapperTexturePlaneError> {
        let surface = self.resolve_surface(descriptor.mapper_surface).ok_or(
            MapperTexturePlaneError::SurfaceUnbound(descriptor.mapper_surface),
        )?;
        let backing = self
            .backing(surface)
            .ok_or(MapperTexturePlaneError::BackingAbsent(surface))?;
        let declaration = descriptor.declaration;
        if declaration.texture_type != reims_vgpu_protocol::TextureType::D2
            || declaration.depth != 1
            || declaration.mipmap_level_count != 1
            || declaration.sample_count != 1
            || declaration.array_length != 1
        {
            return Err(MapperTexturePlaneError::UnsupportedTextureShape);
        }
        let plane = descriptor.plane.get();
        let (offset, width, height, row_pitch, bytes_per_element) =
            if backing.geometry.plane_count == 0 {
                if plane != 0 {
                    return Err(MapperTexturePlaneError::PlaneOutOfBounds { plane, count: 1 });
                }
                (
                    u64::from(backing.geometry.base_offset),
                    backing.geometry.width,
                    backing.geometry.height,
                    u64::from(backing.geometry.bytes_per_row),
                    u32::from(backing.geometry.bytes_per_element),
                )
            } else {
                let (plane_record, _count) =
                    reims_vgpu_protocol::device_desc_plane(&backing.descriptor, plane).ok_or(
                        MapperTexturePlaneError::PlaneOutOfBounds {
                            plane,
                            count: backing.geometry.plane_count,
                        },
                    )?;
                (
                    u64::from(if plane_record.plane_offset == 0 {
                        plane_record.plane_base
                    } else {
                        plane_record.plane_offset
                    }),
                    plane_record.width,
                    plane_record.height,
                    u64::from(plane_record.bytes_per_row),
                    u32::from(plane_record.bytes_per_element),
                )
            };
        if declaration.width != width || declaration.height != height || width == 0 || height == 0 {
            return Err(MapperTexturePlaneError::GeometryMismatch);
        }
        let bytes_per_texel = reims_vgpu_protocol::format_bytes_per_pixel(declaration.pixel_format)
            .ok_or(MapperTexturePlaneError::FormatWidthMismatch)?;
        if bytes_per_element != 0 && bytes_per_element != bytes_per_texel {
            return Err(MapperTexturePlaneError::FormatWidthMismatch);
        }
        let tight_row = u64::from(width)
            .checked_mul(u64::from(bytes_per_texel))
            .ok_or(MapperTexturePlaneError::RangeOverflow)?;
        if row_pitch < tight_row {
            return Err(MapperTexturePlaneError::FormatWidthMismatch);
        }
        let visible_end = u64::from(height - 1)
            .checked_mul(row_pitch)
            .and_then(|rows| rows.checked_add(offset))
            .and_then(|start| start.checked_add(tight_row))
            .ok_or(MapperTexturePlaneError::RangeOverflow)?;
        let allocation_len = u64::try_from(backing.footprint.pages().len())
            .ok()
            .and_then(|pages| pages.checked_mul(backing.footprint.page_size()))
            .ok_or(MapperTexturePlaneError::RangeOverflow)?;
        if visible_end > allocation_len
            || (backing.geometry.alloc_size != 0
                && visible_end > u64::from(backing.geometry.alloc_size))
        {
            return Err(MapperTexturePlaneError::RangeOverflow);
        }
        Ok(MapperTexturePlanePlan {
            mapper_surface: descriptor.mapper_surface,
            surface,
            plane: descriptor.plane,
            format: declaration.pixel_format,
            width,
            height,
            allocation_offset: offset,
            row_pitch,
            visible_end,
            footprint: backing.footprint.clone(),
        })
    }

    pub fn retire_surface(
        &mut self,
        surface: MapperResolvedSurfaceId,
    ) -> Option<MapperSurfaceBacking> {
        self.surfaces.retain(|_, related| *related != surface);
        self.backings.remove(&surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Memory(std::collections::BTreeMap<u64, u8>);

    impl Memory {
        fn write(&mut self, address: u64, bytes: &[u8]) {
            self.0.extend(
                bytes
                    .iter()
                    .enumerate()
                    .map(|(offset, byte)| (address + u64::try_from(offset).unwrap(), *byte)),
            );
        }

        fn mapper_fixture(pixel_format: u32) -> (Self, MapperCapture) {
            use reims_vgpu_paging::mapper as paging;
            const BASE: u64 = paging::ARM_KERNEL_VA_BASE;
            const INTERNAL: u64 = BASE + 0x1000;
            const MAPPER: u64 = BASE + 0x2000;
            const DESCRIPTOR: u64 = BASE + 0x3000;
            const PAGE_OWNER: u64 = BASE + 0x4000;
            const PAGE_TABLE: u64 = BASE + 0x5000;

            let mut memory = Self::default();
            memory.write(
                INTERNAL + paging::MAPPING_INTERNAL_BACKPTR,
                &MAPPER.to_le_bytes(),
            );
            memory.write(INTERNAL + paging::MAPPING_INTERNAL_ID, &9u32.to_le_bytes());
            memory.write(
                INTERNAL + paging::MAPPING_INTERNAL_DESC_PTR,
                &DESCRIPTOR.to_le_bytes(),
            );
            memory.write(
                INTERNAL + paging::MAPPING_INTERNAL_SIZE,
                &paging::MAPPING_INTERNAL_EXPECTED_SIZE.to_le_bytes(),
            );
            memory.write(
                INTERNAL + paging::MAPPING_INTERNAL_PAGE_FIELD_48,
                &PAGE_OWNER.to_le_bytes(),
            );
            memory.write(
                INTERNAL + paging::MAPPING_INTERNAL_PAGE_FIELD_50,
                &(BASE + 0x6000).to_le_bytes(),
            );
            memory.write(
                INTERNAL + paging::MAPPING_INTERNAL_PAGE_COUNT,
                &2u64.to_le_bytes(),
            );
            memory.write(
                PAGE_OWNER + paging::MAPPING_PAGE_TABLE_FROM_F48,
                &PAGE_TABLE.to_le_bytes(),
            );
            memory.write(PAGE_TABLE, &5u32.to_le_bytes());
            memory.write(PAGE_TABLE + 4, &9u32.to_le_bytes());

            let mut descriptor = [0u8; reims_vgpu_protocol::DEVICE_DESC_LEN];
            descriptor[reims_vgpu_protocol::DEVICE_DESC_PIXEL_FORMAT..][..4]
                .copy_from_slice(&pixel_format.to_le_bytes());
            descriptor[reims_vgpu_protocol::DEVICE_DESC_ALLOC_SIZE..][..4]
                .copy_from_slice(&0x2000u32.to_le_bytes());
            let dims = (64u64 << 8) | (32u64 << 40);
            descriptor[reims_vgpu_protocol::DEVICE_DESC_DIMS..][..8]
                .copy_from_slice(&dims.to_le_bytes());
            descriptor[reims_vgpu_protocol::DEVICE_DESC_BPR..][..4]
                .copy_from_slice(&256u32.to_le_bytes());
            descriptor[reims_vgpu_protocol::DEVICE_DESC_BPE..][..2]
                .copy_from_slice(&4u16.to_le_bytes());
            if reims_vgpu_protocol::iosurface_fourcc_is_biplanar(pixel_format) {
                descriptor[reims_vgpu_protocol::DEVICE_DESC_PLANE_COUNT] = 2;
                let plane = |descriptor: &mut [u8],
                             index: usize,
                             offset: u32,
                             width: u32,
                             height: u32,
                             bpr: u32,
                             bpe: u16| {
                    let base = reims_vgpu_protocol::DEVICE_DESC_PLANES
                        + index * reims_vgpu_protocol::DEVICE_PLANE_DESC_LEN;
                    descriptor[base + reims_vgpu_protocol::DEVICE_PLANE_OFFSET..][..4]
                        .copy_from_slice(&offset.to_le_bytes());
                    descriptor[base + reims_vgpu_protocol::DEVICE_PLANE_SIZE..][..4]
                        .copy_from_slice(&(bpr * height).to_le_bytes());
                    descriptor[base + reims_vgpu_protocol::DEVICE_PLANE_DIMS..][..8]
                        .copy_from_slice(
                            &((u64::from(width) << 8) | (u64::from(height) << 40)).to_le_bytes(),
                        );
                    descriptor[base + reims_vgpu_protocol::DEVICE_PLANE_BPR..][..4]
                        .copy_from_slice(&bpr.to_le_bytes());
                    descriptor[base + reims_vgpu_protocol::DEVICE_PLANE_BPE..][..2]
                        .copy_from_slice(&bpe.to_le_bytes());
                };
                plane(&mut descriptor, 0, 0, 64, 32, 64, 1);
                plane(&mut descriptor, 1, 0x1000, 32, 16, 64, 2);
            }
            memory.write(DESCRIPTOR, &descriptor);
            (
                memory,
                MapperCapture {
                    producer: 7,
                    mapper_device_kva: MAPPER,
                    request_kind: MapperRequestKind::Map,
                    mapping_internal: INTERNAL,
                },
            )
        }
    }

    impl reims_vgpu_paging::mapper::PagesMemory for Memory {
        fn read(&self, address: u64, dst: &mut [u8]) -> bool {
            let Some(bytes) = (0..dst.len())
                .map(|offset| {
                    self.0
                        .get(&(address + u64::try_from(offset).unwrap()))
                        .copied()
                })
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            dst.copy_from_slice(&bytes);
            true
        }
    }

    #[test]
    fn mapper_request_read_is_the_exact_published_fixed_record() {
        assert_eq!(
            resolve_mapper_request_read(0x4000, 3),
            Ok(MapperRequestReadPlan {
                ring_base: 0x4000,
                producer: 3,
                gpa: 0x4020,
                byte_len: reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN as u64,
            })
        );
        assert_eq!(
            resolve_mapper_request_read(0x4000, 0),
            Err(MapperRequestReadError::NoPublishedEntry)
        );
        assert_eq!(
            resolve_mapper_request_read(u64::MAX - 15, 2),
            Err(MapperRequestReadError::AddressOverflow {
                ring_base: u64::MAX - 15,
                producer: 2,
            })
        );
    }

    #[test]
    fn capture_consumption_is_scoped_to_the_publishing_entry() {
        let mut service = MapperService::default();
        let capture = MapperCapture {
            producer: 7,
            mapper_device_kva: 0x1000,
            request_kind: MapperRequestKind::Map,
            mapping_internal: 0x2000,
        };
        service.publish_capture(capture).unwrap();
        assert_eq!(service.take_capture(6), None);
        assert_eq!(service.take_capture(7), Some(capture));
        assert_eq!(service.take_capture(7), None);
    }

    #[test]
    fn every_live_producer_keeps_its_capture_until_that_entry_consumes_it() {
        let mut service = MapperService::default();
        let first = MapperCapture {
            producer: 7,
            mapper_device_kva: 0x1000,
            request_kind: MapperRequestKind::Map,
            mapping_internal: 0x2000,
        };
        let second = MapperCapture {
            producer: 8,
            mapper_device_kva: 0x1000,
            request_kind: MapperRequestKind::Map,
            mapping_internal: 0x3000,
        };
        service.publish_capture(first).unwrap();
        service.publish_capture(second).unwrap();

        assert_eq!(service.take_capture(7), Some(first));
        assert_eq!(service.take_capture(8), Some(second));
    }

    #[test]
    fn mapper_identity_is_wide_and_retirement_follows_the_resolved_surface() {
        let mut service = MapperService::default();
        let wide = MapperSurfaceRef::new(0x1_0000_0001);
        let low = MapperSurfaceRef::new(1);
        let surface = MapperResolvedSurfaceId::new(9);
        assert!(service.map_surface(wide, surface));
        assert_eq!(service.resolve_surface(wide), Some(surface));
        assert_eq!(service.resolve_surface(low), None);
        assert_eq!(service.mapper_surfaces_for(surface).as_ref(), [wide]);
        service.retire_surface(surface);
        assert_eq!(service.resolve_surface(wide), None);
    }

    #[test]
    fn complete_mapper_backing_publishes_atomically_and_unmap_returns_it() {
        let (memory, capture) = Memory::mapper_fixture(u32::from(
            reims_vgpu_protocol::metal_pixel::MTL_FORMAT_BGRA8_UNORM,
        ));
        let surface = MapperResolvedSurfaceId::new(9);
        let backing = resolve_mapper_surface_backing(&memory, surface, capture, 12).unwrap();
        assert_eq!((backing.geometry.width, backing.geometry.height), (64, 32));
        assert_eq!(backing.pages.entries, [5, 9]);
        assert_eq!(backing.page_shift, 12);
        assert_eq!(backing.footprint.pages(), [0x1000, 0x2000]);
        assert_eq!(
            backing.metal_pixel_format,
            Some(reims_vgpu_protocol::metal_pixel::MTL_FORMAT_BGRA8_UNORM)
        );

        let mut service = MapperService::default();
        assert!(service.publish_backing(backing.clone()).is_none());
        assert_eq!(service.backing(surface), Some(&backing));

        let (biplanar, biplanar_capture) =
            Memory::mapper_fixture(reims_vgpu_protocol::IOSURFACE_FOURCC_420F);
        let biplanar =
            resolve_mapper_surface_backing(&biplanar, surface, biplanar_capture, 12).unwrap();
        assert_eq!(biplanar.metal_pixel_format, None);
        assert_eq!(biplanar.footprint.pages(), [0x1000, 0x2000]);
        let mapper_surface = MapperSurfaceRef::new(0x1_0000_0009);
        let mut plane_service = MapperService::default();
        assert!(plane_service.map_surface(mapper_surface, surface));
        assert!(plane_service.publish_backing(biplanar).is_none());
        let plane_view = reims_vgpu_protocol::MapperIOSurfaceTextureView {
            mapper_surface,
            object: reims_vgpu_protocol::SerializerRef::new(4),
            declaration: reims_vgpu_protocol::TextureDeclaration {
                texture_type: reims_vgpu_protocol::TextureType::D2,
                framebuffer_only: false,
                is_drawable: false,
                write_swizzle_enabled: None,
                allow_gpu_optimized_contents: false,
                usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                pixel_format: reims_vgpu_protocol::metal_pixel::MTL_FORMAT_RG8_UNORM,
                width: 32,
                height: 16,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                resource_options: 0,
                protection_options: 0,
                swizzle: None,
            },
            plane: reims_vgpu_protocol::PlaneIndex::new(1),
            rotation: None,
        };
        assert_eq!(
            plane_service.texture_plane_plan(&plane_view).unwrap(),
            MapperTexturePlanePlan {
                mapper_surface,
                surface,
                plane: reims_vgpu_protocol::PlaneIndex::new(1),
                format: reims_vgpu_protocol::metal_pixel::MTL_FORMAT_RG8_UNORM,
                width: 32,
                height: 16,
                allocation_offset: 0x1000,
                row_pitch: 64,
                visible_end: 0x1400,
                footprint: plane_service.backing(surface).unwrap().footprint.clone(),
            }
        );

        let (wrong, wrong_capture) = Memory::mapper_fixture(0x5a5a_5a5a);
        assert_eq!(
            resolve_mapper_surface_backing(&wrong, surface, wrong_capture, 12),
            Err(MapperSurfaceBackingError::UnsupportedPixelFormat(
                0x5a5a_5a5a
            ))
        );
        assert_eq!(service.backing(surface), Some(&backing));
        assert_eq!(service.retire_surface(surface), Some(backing));
        assert!(service.backing(surface).is_none());
    }
}
