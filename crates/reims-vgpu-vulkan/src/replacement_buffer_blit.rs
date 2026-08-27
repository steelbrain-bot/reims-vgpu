//! Project prepared semantic buffer blits to exact Vulkan buffer commands.

use ash::vk;
use reims_vgpu_core::{
    BufferFillPattern, GpuWriteId, PreparedBufferBlit, PreparedNativeBufferBlit,
    PreparedNativeBufferRange, ResolvedBlit, ResolvedResourceCompletion,
};
use reims_vgpu_protocol::{BackingId, RepresentationId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeBufferTarget {
    pub buffer: vk::Buffer,
    pub base_offset: u64,
    /// Bytes safely addressable from `base_offset`; may include native padding
    /// beyond the exact semantic `size` for aligned storage-buffer access.
    pub accessible_size: u64,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeComputeFillLimits {
    pub min_storage_buffer_offset_alignment: u64,
    pub max_storage_buffer_range: u64,
    pub max_compute_work_group_count_x: u32,
}

pub trait ReplacementBufferResolver {
    fn resolve_buffer(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<NativeBufferTarget>;

    fn resolve_host_staging(
        &self,
        _backing: BackingId,
        _representation: RepresentationId,
    ) -> Option<crate::replacement_representation::ReplacementHostStagingBuffer> {
        None
    }

    fn resolve_linear_texture_layout(
        &self,
        _backing: BackingId,
        _representation: RepresentationId,
    ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
        None
    }

    fn compute_fill_limits(&self) -> Option<NativeComputeFillLimits> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferBlitRecordError {
    WriteIdentityMismatch,
    UnknownRepresentation {
        backing: BackingId,
        representation: RepresentationId,
    },
    RangeOutOfBounds(BackingId),
    RangeAddressOverflow(BackingId),
    CopyLengthMismatch,
    CopyOverlap,
    ComputeFillUnavailable,
    MissingStorageDestination(BackingId),
    ComputeFillRangeUnrepresentable,
    MissingTransferSource(BackingId),
    MissingTransferDestination(BackingId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBufferBlit {
    Fill {
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
        data: u32,
    },
    ComputeFill {
        buffer: vk::Buffer,
        binding_offset: u64,
        binding_range: u64,
        start: u32,
        byte_count: u32,
        pattern: u32,
        pattern_width: u32,
        word_count: u32,
        dispatch_x: u32,
    },
    Copy {
        source: vk::Buffer,
        destination: vk::Buffer,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplacementBufferBlitProgram {
    index: usize,
    operation: ResolvedBlit,
    backings: Box<[BackingId]>,
    completions: Box<[ResolvedResourceCompletion]>,
    native: NativeBufferBlit,
}

impl ReplacementBufferBlitProgram {
    /// Resolve native handles from the exact semantic preparation token. The
    /// recorder later verifies both index and operation against its owned EXEC.
    pub fn resolve(
        index: usize,
        prepared: &PreparedBufferBlit,
        resolver: &impl ReplacementBufferResolver,
    ) -> Result<Self, BufferBlitRecordError> {
        if prepared.write()
            != GpuWriteId::operation(prepared.transaction(), prepared.submission(), index)
        {
            return Err(BufferBlitRecordError::WriteIdentityMismatch);
        }
        Ok(Self {
            index,
            operation: prepared.operation().clone(),
            backings: prepared.backings(),
            completions: prepared.resource_completions(),
            native: resolve_buffer_blit(prepared.native(), resolver)?,
        })
    }

    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    pub(crate) const fn operation(&self) -> &ResolvedBlit {
        &self.operation
    }

    pub(crate) const fn native(&self) -> NativeBufferBlit {
        self.native
    }

    pub(crate) const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    pub(crate) const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }
}

pub fn resolve_buffer_blit(
    operation: PreparedNativeBufferBlit,
    resolver: &impl ReplacementBufferResolver,
) -> Result<NativeBufferBlit, BufferBlitRecordError> {
    match operation {
        PreparedNativeBufferBlit::Fill {
            destination,
            pattern,
        } => {
            let (target, offset, size) = resolve_range(destination, resolver)?;
            if !target.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
                return Err(BufferBlitRecordError::MissingTransferDestination(
                    destination.backing,
                ));
            }
            let data = match pattern {
                BufferFillPattern::Byte(byte) => u32::from_le_bytes([byte; 4]),
                BufferFillPattern::Word(bytes) => u32::from_le_bytes(bytes),
            };
            if offset.is_multiple_of(4) && size.is_multiple_of(4) {
                return Ok(NativeBufferBlit::Fill {
                    buffer: target.buffer,
                    offset,
                    size,
                    data,
                });
            }
            resolve_compute_fill(
                destination.backing,
                target,
                offset,
                size,
                data,
                pattern,
                resolver,
            )
        }
        PreparedNativeBufferBlit::Copy {
            source,
            destination,
        } => {
            let source_length = source.region.end() - source.region.start();
            let destination_length = destination.region.end() - destination.region.start();
            if source_length != destination_length {
                return Err(BufferBlitRecordError::CopyLengthMismatch);
            }
            let (source_target, source_offset, size) = resolve_range(source, resolver)?;
            let (destination_target, destination_offset, _) = resolve_range(destination, resolver)?;
            if !source_target
                .usage
                .contains(vk::BufferUsageFlags::TRANSFER_SRC)
            {
                return Err(BufferBlitRecordError::MissingTransferSource(source.backing));
            }
            if !destination_target
                .usage
                .contains(vk::BufferUsageFlags::TRANSFER_DST)
            {
                return Err(BufferBlitRecordError::MissingTransferDestination(
                    destination.backing,
                ));
            }
            if source_target.buffer == destination_target.buffer
                && ranges_overlap(source_offset, size, destination_offset, size)
            {
                return Err(BufferBlitRecordError::CopyOverlap);
            }
            Ok(NativeBufferBlit::Copy {
                source: source_target.buffer,
                destination: destination_target.buffer,
                source_offset,
                destination_offset,
                size,
            })
        }
    }
}

fn resolve_compute_fill(
    backing: BackingId,
    target: NativeBufferTarget,
    offset: u64,
    size: u64,
    pattern: u32,
    semantic_pattern: BufferFillPattern,
    resolver: &impl ReplacementBufferResolver,
) -> Result<NativeBufferBlit, BufferBlitRecordError> {
    if !target.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        return Err(BufferBlitRecordError::MissingStorageDestination(backing));
    }
    let limits = resolver
        .compute_fill_limits()
        .ok_or(BufferBlitRecordError::ComputeFillUnavailable)?;
    let alignment = limits.min_storage_buffer_offset_alignment.max(4);
    if !alignment.is_power_of_two() || !target.base_offset.is_multiple_of(alignment) {
        return Err(BufferBlitRecordError::ComputeFillRangeUnrepresentable);
    }
    let relative_start = offset
        .checked_sub(target.base_offset)
        .ok_or(BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    let binding_relative = relative_start / alignment * alignment;
    let binding_offset = target
        .base_offset
        .checked_add(binding_relative)
        .ok_or(BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    let start = relative_start - binding_relative;
    let touched_end = start
        .checked_add(size)
        .and_then(|end| end.checked_add(3))
        .map(|end| end / 4 * 4)
        .ok_or(BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    let accessible_after_binding = target
        .accessible_size
        .checked_sub(binding_relative)
        .ok_or(BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    if touched_end > accessible_after_binding
        || touched_end > limits.max_storage_buffer_range
        || limits.max_compute_work_group_count_x == 0
    {
        return Err(BufferBlitRecordError::ComputeFillRangeUnrepresentable);
    }
    let start =
        u32::try_from(start).map_err(|_| BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    let byte_count =
        u32::try_from(size).map_err(|_| BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    let word_count = u32::try_from(touched_end / 4)
        .map_err(|_| BufferBlitRecordError::ComputeFillRangeUnrepresentable)?;
    let required_groups = word_count.div_ceil(64).max(1);
    Ok(NativeBufferBlit::ComputeFill {
        buffer: target.buffer,
        binding_offset,
        binding_range: touched_end,
        start,
        byte_count,
        pattern,
        pattern_width: u32::try_from(semantic_pattern.bytes().len())
            .expect("buffer fill patterns have a fixed one- or four-byte width"),
        word_count,
        dispatch_x: required_groups.min(limits.max_compute_work_group_count_x),
    })
}

fn resolve_range(
    range: PreparedNativeBufferRange,
    resolver: &impl ReplacementBufferResolver,
) -> Result<(NativeBufferTarget, u64, u64), BufferBlitRecordError> {
    let target = resolver
        .resolve_buffer(range.backing, range.representation)
        .ok_or(BufferBlitRecordError::UnknownRepresentation {
            backing: range.backing,
            representation: range.representation,
        })?;
    let relative_end = range.region.end();
    if relative_end > target.size {
        return Err(BufferBlitRecordError::RangeOutOfBounds(range.backing));
    }
    let offset = target
        .base_offset
        .checked_add(range.region.start())
        .ok_or(BufferBlitRecordError::RangeAddressOverflow(range.backing))?;
    target
        .base_offset
        .checked_add(relative_end)
        .ok_or(BufferBlitRecordError::RangeAddressOverflow(range.backing))?;
    Ok((target, offset, range.region.end() - range.region.start()))
}

fn ranges_overlap(left: u64, left_len: u64, right: u64, right_len: u64) -> bool {
    let Some(left_end) = left.checked_add(left_len) else {
        return true;
    };
    let Some(right_end) = right.checked_add(right_len) else {
        return true;
    };
    left < right_end && right < left_end
}

/// Emit one fully projected command into an already recording primary.
///
/// # Safety
///
/// `command_buffer` must be recording on `device`; every target is retained by
/// the prepared transaction until its queue timeline retires.
pub unsafe fn record_buffer_blit(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    operation: NativeBufferBlit,
) {
    match operation {
        NativeBufferBlit::Fill {
            buffer,
            offset,
            size,
            data,
        } => unsafe { device.cmd_fill_buffer(command_buffer, buffer, offset, size, data) },
        NativeBufferBlit::ComputeFill { .. } => {
            unreachable!("compute fills require the worker-owned fixed pipeline")
        }
        NativeBufferBlit::Copy {
            source,
            destination,
            source_offset,
            destination_offset,
            size,
        } => unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                source,
                destination,
                &[vk::BufferCopy {
                    src_offset: source_offset,
                    dst_offset: destination_offset,
                    size,
                }],
            );
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::LinearRange;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Resolver(BTreeMap<(BackingId, RepresentationId), NativeBufferTarget>);

    impl ReplacementBufferResolver for Resolver {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            self.0.get(&(backing, representation)).copied()
        }
    }

    fn range(
        backing: u64,
        representation: u64,
        start: u64,
        length: u64,
    ) -> PreparedNativeBufferRange {
        PreparedNativeBufferRange {
            backing: BackingId::new(backing),
            representation: RepresentationId::new(representation),
            region: LinearRange::new(start, length).unwrap(),
        }
    }

    #[test]
    fn fill_and_copy_use_native_relative_offsets_and_exact_patterns() {
        let mut resolver = Resolver::default();
        resolver.0.insert(
            (BackingId::new(1), RepresentationId::new(2)),
            NativeBufferTarget {
                buffer: vk::Buffer::from_raw(11),
                base_offset: 128,
                accessible_size: 256,
                size: 256,
                usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            },
        );
        resolver.0.insert(
            (BackingId::new(3), RepresentationId::new(4)),
            NativeBufferTarget {
                buffer: vk::Buffer::from_raw(22),
                base_offset: 512,
                accessible_size: 256,
                size: 256,
                usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            },
        );

        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Fill {
                    destination: range(1, 2, 16, 32),
                    pattern: BufferFillPattern::Word([1, 2, 3, 4]),
                },
                &resolver,
            ),
            Ok(NativeBufferBlit::Fill {
                buffer: vk::Buffer::from_raw(11),
                offset: 144,
                size: 32,
                data: u32::from_le_bytes([1, 2, 3, 4]),
            })
        );
        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Copy {
                    source: range(1, 2, 8, 40),
                    destination: range(3, 4, 24, 40),
                },
                &resolver,
            ),
            Ok(NativeBufferBlit::Copy {
                source: vk::Buffer::from_raw(11),
                destination: vk::Buffer::from_raw(22),
                source_offset: 136,
                destination_offset: 536,
                size: 40,
            })
        );
    }

    #[test]
    fn projection_refuses_missing_compute_fill_usage_overlap_and_out_of_bounds_ranges() {
        let mut resolver = Resolver::default();
        resolver.0.insert(
            (BackingId::new(1), RepresentationId::new(2)),
            NativeBufferTarget {
                buffer: vk::Buffer::from_raw(11),
                base_offset: 0,
                accessible_size: 64,
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            },
        );
        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Fill {
                    destination: range(1, 2, 1, 4),
                    pattern: BufferFillPattern::Byte(0),
                },
                &resolver,
            ),
            Err(BufferBlitRecordError::MissingStorageDestination(
                BackingId::new(1)
            ))
        );
        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Copy {
                    source: range(1, 2, 0, 32),
                    destination: range(1, 2, 16, 32),
                },
                &resolver,
            ),
            Err(BufferBlitRecordError::CopyOverlap)
        );
        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Fill {
                    destination: range(1, 2, 60, 8),
                    pattern: BufferFillPattern::Byte(0),
                },
                &resolver,
            ),
            Err(BufferBlitRecordError::RangeOutOfBounds(BackingId::new(1)))
        );
        resolver
            .0
            .get_mut(&(BackingId::new(1), RepresentationId::new(2)))
            .unwrap()
            .usage = vk::BufferUsageFlags::empty();
        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Fill {
                    destination: range(1, 2, 0, 4),
                    pattern: BufferFillPattern::Byte(0),
                },
                &resolver,
            ),
            Err(BufferBlitRecordError::MissingTransferDestination(
                BackingId::new(1)
            ))
        );
    }

    #[test]
    fn unaligned_fill_projects_one_exact_compute_window_and_pattern_phase() {
        struct ComputeResolver(Resolver);
        impl ReplacementBufferResolver for ComputeResolver {
            fn resolve_buffer(
                &self,
                backing: BackingId,
                representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                self.0.resolve_buffer(backing, representation)
            }

            fn compute_fill_limits(&self) -> Option<NativeComputeFillLimits> {
                Some(NativeComputeFillLimits {
                    min_storage_buffer_offset_alignment: 16,
                    max_storage_buffer_range: 256,
                    max_compute_work_group_count_x: 65_535,
                })
            }
        }

        let mut resolver = Resolver::default();
        resolver.0.insert(
            (BackingId::new(1), RepresentationId::new(2)),
            NativeBufferTarget {
                buffer: vk::Buffer::from_raw(11),
                base_offset: 128,
                accessible_size: 68,
                size: 65,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        );
        assert_eq!(
            resolve_buffer_blit(
                PreparedNativeBufferBlit::Fill {
                    destination: range(1, 2, 17, 7),
                    pattern: BufferFillPattern::Byte(0x5a),
                },
                &ComputeResolver(resolver),
            ),
            Ok(NativeBufferBlit::ComputeFill {
                buffer: vk::Buffer::from_raw(11),
                binding_offset: 144,
                binding_range: 8,
                start: 1,
                byte_count: 7,
                pattern: u32::from_le_bytes([0x5a; 4]),
                pattern_width: 1,
                word_count: 2,
                dispatch_x: 1,
            })
        );
    }
}
