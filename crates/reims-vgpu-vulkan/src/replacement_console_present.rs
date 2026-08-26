//! Queue-owned BGRA readback for the QEMU console presentation endpoint.
//!
//! Each live Present owns one allocation and one command recording. Nothing is
//! pooled or evicted: the owner moves from queue preparation to timeline
//! completion and is destroyed only after its mapped bytes have been copied.

use crate::{
    engine::context::SharedDeviceContext,
    replacement_image_transition::{NativeImageTarget, NativeImageUseTransitions},
    replacement_queue::ReplacementPresentRecording,
};
use ash::vk;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementConsolePresentPhase {
    CreateImage,
    AllocateImageMemory,
    BindImageMemory,
    CreateBuffer,
    AllocateBufferMemory,
    BindBufferMemory,
    MapBufferMemory,
    CreateCommandPool,
    AllocateCommandBuffer,
    CreateFence,
    ResetFence,
    ResetCommandBuffer,
    BeginCommandBuffer,
    EndCommandBuffer,
    InvalidateBufferMemory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementConsolePresentError {
    EmptyExtent,
    SizeOverflow,
    PixelFormat(u16),
    SourceBlitUnsupported(vk::FormatFeatureFlags),
    DestinationBlitUnsupported(vk::FormatFeatureFlags),
    NoImageMemoryType,
    NoBufferMemoryType,
    Driver {
        phase: ReplacementConsolePresentPhase,
        result: vk::Result,
    },
}

#[derive(Debug)]
pub struct ReplacementConsoleFrame {
    bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementConsoleFrameCopyError {
    Width { expected: u32, actual: u32 },
    Height { expected: u32, actual: u32 },
    StrideOverflow,
    DestinationStride { required: u32, actual: u32 },
    DestinationLength { required: usize, actual: usize },
}

impl ReplacementConsoleFrame {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Copy tightly packed BGRA rows into a host display surface. Validation
    /// precedes every write, so a refused copy leaves both buffers unchanged.
    pub fn copy_to_bgra8(
        &self,
        destination: &mut [u8],
        destination_stride: u32,
        width: u32,
        height: u32,
    ) -> Result<(), ReplacementConsoleFrameCopyError> {
        if width != self.width {
            return Err(ReplacementConsoleFrameCopyError::Width {
                expected: self.width,
                actual: width,
            });
        }
        if height != self.height {
            return Err(ReplacementConsoleFrameCopyError::Height {
                expected: self.height,
                actual: height,
            });
        }
        let row_len = width
            .checked_mul(4)
            .ok_or(ReplacementConsoleFrameCopyError::StrideOverflow)?;
        if destination_stride < row_len {
            return Err(ReplacementConsoleFrameCopyError::DestinationStride {
                required: row_len,
                actual: destination_stride,
            });
        }
        let required = if height == 0 {
            0
        } else {
            usize::try_from(
                u64::from(height - 1) * u64::from(destination_stride) + u64::from(row_len),
            )
            .map_err(|_| ReplacementConsoleFrameCopyError::StrideOverflow)?
        };
        if destination.len() < required {
            return Err(ReplacementConsoleFrameCopyError::DestinationLength {
                required,
                actual: destination.len(),
            });
        }
        let source_stride = usize::try_from(self.stride)
            .map_err(|_| ReplacementConsoleFrameCopyError::StrideOverflow)?;
        let destination_stride = usize::try_from(destination_stride)
            .map_err(|_| ReplacementConsoleFrameCopyError::StrideOverflow)?;
        let row_len = usize::try_from(row_len)
            .map_err(|_| ReplacementConsoleFrameCopyError::StrideOverflow)?;
        for row in 0..usize::try_from(height)
            .map_err(|_| ReplacementConsoleFrameCopyError::StrideOverflow)?
        {
            let source_start = row * source_stride;
            let destination_start = row * destination_stride;
            destination[destination_start..destination_start + row_len]
                .copy_from_slice(&self.bytes[source_start..source_start + row_len]);
        }
        Ok(())
    }
}

struct ReplacementConsolePresentResources {
    context: Arc<SharedDeviceContext>,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    mapped: *mut u8,
    coherent: bool,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

// Vulkan handles are externally synchronized by the replacement queue owner;
// the allocation moves between threads only as opaque submission ownership.
unsafe impl Send for ReplacementConsolePresentResources {}

impl Drop for ReplacementConsolePresentResources {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_null() {
                // No mapping was established.
            } else {
                self.context.device.unmap_memory(self.buffer_memory);
            }
            if self.fence != vk::Fence::null() {
                self.context.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.context
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
            if self.buffer != vk::Buffer::null() {
                self.context.device.destroy_buffer(self.buffer, None);
            }
            if self.buffer_memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.buffer_memory, None);
            }
            if self.image != vk::Image::null() {
                self.context.device.destroy_image(self.image, None);
            }
            if self.image_memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.image_memory, None);
            }
        }
    }
}

#[must_use = "a console Present allocation must reach queue completion or explicit teardown"]
pub struct ReplacementPreparedConsolePresent {
    resources: ReplacementConsolePresentResources,
    byte_len: usize,
    width: u32,
    height: u32,
    stride: u32,
}

#[derive(Debug)]
pub struct ReplacementConsolePresentCompletionFailure {
    pub reason: ReplacementConsolePresentError,
    pub prepared: ReplacementPreparedConsolePresent,
}

impl std::fmt::Debug for ReplacementPreparedConsolePresent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplacementPreparedConsolePresent")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl ReplacementPreparedConsolePresent {
    pub const fn recording(&self) -> ReplacementPresentRecording {
        ReplacementPresentRecording {
            slot: 0,
            command_pool: self.resources.command_pool,
            command_buffer: self.resources.command_buffer,
            fence: self.resources.fence,
        }
    }

    /// Read the mapped transfer destination only after the queue timeline has
    /// reached the recording's accepted signal point.
    pub fn complete(
        self,
    ) -> Result<ReplacementConsoleFrame, Box<ReplacementConsolePresentCompletionFailure>> {
        if !self.resources.coherent {
            let invalidated = unsafe {
                self.resources
                    .context
                    .device
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(self.resources.buffer_memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            };
            if let Err(result) = invalidated {
                return Err(Box::new(ReplacementConsolePresentCompletionFailure {
                    reason: ReplacementConsolePresentError::Driver {
                        phase: ReplacementConsolePresentPhase::InvalidateBufferMemory,
                        result,
                    },
                    prepared: self,
                }));
            }
        }
        let mut bytes = vec![0; self.byte_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.resources.mapped.cast_const(),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(ReplacementConsoleFrame {
            bytes,
            width: self.width,
            height: self.height,
            stride: self.stride,
        })
    }
}

pub(crate) unsafe fn prepare_console_present(
    context: Arc<SharedDeviceContext>,
    source: NativeImageTarget,
    transitions: &NativeImageUseTransitions,
) -> Result<ReplacementPreparedConsolePresent, ReplacementConsolePresentError> {
    if source.extent.width == 0 || source.extent.height == 0 || source.extent.depth != 1 {
        return Err(ReplacementConsolePresentError::EmptyExtent);
    }
    let stride = source
        .extent
        .width
        .checked_mul(4)
        .ok_or(ReplacementConsolePresentError::SizeOverflow)?;
    let byte_len_u64 = u64::from(stride)
        .checked_mul(u64::from(source.extent.height))
        .ok_or(ReplacementConsolePresentError::SizeOverflow)?;
    let byte_len =
        usize::try_from(byte_len_u64).map_err(|_| ReplacementConsolePresentError::SizeOverflow)?;
    let (source_format, _) = crate::translate::pixel::verbatim_texel(source.pixel_format).ok_or(
        ReplacementConsolePresentError::PixelFormat(source.pixel_format),
    )?;
    let source_features = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.pd, source_format)
            .optimal_tiling_features
    };
    if !source_features.contains(vk::FormatFeatureFlags::BLIT_SRC) {
        return Err(ReplacementConsolePresentError::SourceBlitUnsupported(
            source_features,
        ));
    }
    let destination_format = vk::Format::B8G8R8A8_UNORM;
    let destination_features = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.pd, destination_format)
            .optimal_tiling_features
    };
    if !destination_features
        .contains(vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::TRANSFER_SRC)
    {
        return Err(ReplacementConsolePresentError::DestinationBlitUnsupported(
            destination_features,
        ));
    }

    let mut resources = ReplacementConsolePresentResources {
        context,
        image: vk::Image::null(),
        image_memory: vk::DeviceMemory::null(),
        buffer: vk::Buffer::null(),
        buffer_memory: vk::DeviceMemory::null(),
        mapped: std::ptr::null_mut(),
        coherent: false,
        command_pool: vk::CommandPool::null(),
        command_buffer: vk::CommandBuffer::null(),
        fence: vk::Fence::null(),
    };
    let driver = |phase, result| ReplacementConsolePresentError::Driver { phase, result };
    resources.image = unsafe {
        resources.context.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(destination_format)
                .extent(source.extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::CreateImage, result))?;
    let image_requirements = unsafe {
        resources
            .context
            .device
            .get_image_memory_requirements(resources.image)
    };
    let image_memory_type = resources
        .context
        .memory_type_for(
            image_requirements.memory_type_bits,
            image_requirements.size,
            crate::memory::MemoryClass::DeviceLocalPreferred,
        )
        .ok_or(ReplacementConsolePresentError::NoImageMemoryType)?;
    resources.image_memory = unsafe {
        resources.context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(image_requirements.size)
                .memory_type_index(image_memory_type),
            None,
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::AllocateImageMemory, result))?;
    unsafe {
        resources
            .context
            .device
            .bind_image_memory(resources.image, resources.image_memory, 0)
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::BindImageMemory, result))?;

    resources.buffer = unsafe {
        resources.context.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(byte_len_u64)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::CreateBuffer, result))?;
    let buffer_requirements = unsafe {
        resources
            .context
            .device
            .get_buffer_memory_requirements(resources.buffer)
    };
    let buffer_memory_type = resources
        .context
        .memory_type_for(
            buffer_requirements.memory_type_bits,
            buffer_requirements.size,
            crate::memory::MemoryClass::Readback,
        )
        .ok_or(ReplacementConsolePresentError::NoBufferMemoryType)?;
    resources.buffer_memory = unsafe {
        resources.context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(buffer_requirements.size)
                .memory_type_index(buffer_memory_type),
            None,
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::AllocateBufferMemory, result))?;
    unsafe {
        resources
            .context
            .device
            .bind_buffer_memory(resources.buffer, resources.buffer_memory, 0)
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::BindBufferMemory, result))?;
    resources.mapped = unsafe {
        resources.context.device.map_memory(
            resources.buffer_memory,
            0,
            buffer_requirements.size,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::MapBufferMemory, result))?
    .cast();
    resources.coherent = resources
        .context
        .mapped_memory_kind(buffer_memory_type)
        .coherent;
    resources.command_pool = unsafe {
        resources.context.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(resources.context.gq)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::CreateCommandPool, result))?;
    resources.command_buffer = unsafe {
        resources.context.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(resources.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|result| {
        driver(
            ReplacementConsolePresentPhase::AllocateCommandBuffer,
            result,
        )
    })?[0];
    resources.fence = unsafe {
        resources
            .context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::CreateFence, result))?;
    unsafe { record_console_present(&resources, source, transitions) }?;
    Ok(ReplacementPreparedConsolePresent {
        resources,
        byte_len,
        width: source.extent.width,
        height: source.extent.height,
        stride,
    })
}

unsafe fn record_console_present(
    resources: &ReplacementConsolePresentResources,
    source: NativeImageTarget,
    transitions: &NativeImageUseTransitions,
) -> Result<(), ReplacementConsolePresentError> {
    let device = &resources.context.device;
    let driver = |phase, result| ReplacementConsolePresentError::Driver { phase, result };
    unsafe { device.reset_fences(&[resources.fence]) }
        .map_err(|result| driver(ReplacementConsolePresentPhase::ResetFence, result))?;
    unsafe {
        device.reset_command_buffer(
            resources.command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::ResetCommandBuffer, result))?;
    unsafe {
        device.begin_command_buffer(
            resources.command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::BeginCommandBuffer, result))?;
    unsafe {
        crate::replacement_barrier_record::record_hazard_barriers(
            device,
            resources.command_buffer,
            &transitions.before,
        );
    }
    let color_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    unsafe {
        device.cmd_pipeline_barrier(
            resources.command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(resources.image)
                .subresource_range(color_range)],
        );
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let right = i32::try_from(source.extent.width)
            .map_err(|_| ReplacementConsolePresentError::SizeOverflow)?;
        let bottom = i32::try_from(source.extent.height)
            .map_err(|_| ReplacementConsolePresentError::SizeOverflow)?;
        let offsets = [
            vk::Offset3D::default(),
            vk::Offset3D {
                x: right,
                y: bottom,
                z: 1,
            },
        ];
        device.cmd_blit_image(
            resources.command_buffer,
            source.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            resources.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(layers)
                .src_offsets(offsets)
                .dst_subresource(layers)
                .dst_offsets(offsets)],
            vk::Filter::NEAREST,
        );
        crate::replacement_barrier_record::record_hazard_barriers(
            device,
            resources.command_buffer,
            &transitions.after,
        );
        device.cmd_pipeline_barrier(
            resources.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(resources.image)
                .subresource_range(color_range)],
        );
        device.cmd_copy_image_to_buffer(
            resources.command_buffer,
            resources.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            resources.buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(layers)
                .image_extent(source.extent)],
        );
        device.cmd_pipeline_barrier(
            resources.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[],
        );
        device.end_command_buffer(resources.command_buffer)
    }
    .map_err(|result| driver(ReplacementConsolePresentPhase::EndCommandBuffer, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> ReplacementConsoleFrame {
        ReplacementConsoleFrame {
            bytes: (0..16).collect(),
            width: 2,
            height: 2,
            stride: 8,
        }
    }

    #[test]
    fn console_frame_copy_preserves_destination_padding() {
        let mut destination = [0xff; 20];
        frame().copy_to_bgra8(&mut destination, 12, 2, 2).unwrap();
        assert_eq!(&destination[..8], &(0..8).collect::<Vec<_>>());
        assert_eq!(&destination[8..12], &[0xff; 4]);
        assert_eq!(&destination[12..20], &(8..16).collect::<Vec<_>>());
    }

    #[test]
    fn refused_console_frame_copy_writes_nothing() {
        let mut destination = [0x5a; 16];
        assert_eq!(
            frame().copy_to_bgra8(&mut destination, 8, 1, 2),
            Err(ReplacementConsoleFrameCopyError::Width {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(destination, [0x5a; 16]);
        assert_eq!(
            frame().copy_to_bgra8(&mut destination[..15], 8, 2, 2),
            Err(ReplacementConsoleFrameCopyError::DestinationLength {
                required: 16,
                actual: 15,
            })
        );
        assert_eq!(destination, [0x5a; 16]);
    }
}
