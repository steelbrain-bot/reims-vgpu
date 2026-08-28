//! Host-originated scanout staging for the replacement window presenter.
//!
//! The guest publishes a linear BGRA scanout long before its Metal driver
//! attaches — the firmware and early-boot display are CPU-written framebuffer
//! bytes and nothing else. Those frames reach the window through the same
//! swapchain presenter a guest Present uses; what they need first is a Vulkan
//! image the CPU can write and the blit can read.
//!
//! Three rules the image's shape is fixed by, none of them negotiable:
//!
//! - **LINEAR tiling.** A host write through a persistent map into an OPTIMAL
//!   image has no defined layout, so the bytes that arrive are not the bytes
//!   that were written.
//! - **The driver's row pitch, never the width.** An implementation is free to
//!   pad each row; copying tightly into a padded image shears the picture by a
//!   growing offset per row.
//! - **`PREINITIALIZED` on first use, `GENERAL` after.** Those are the only two
//!   layouts that preserve contents across a transition, so the barrier ahead of
//!   the blit must name whichever one the image is actually in. `UNDEFINED`
//!   discards the frame that was just uploaded.

use crate::{
    engine::context::DeviceContext,
    replacement_barrier_record::NativeBarrierBatch,
    replacement_image_transition::{NativeImageTarget, NativeImageUseTransitions},
};
use ash::vk;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementHostScanoutPhase {
    CreateImage,
    AllocateMemory,
    BindMemory,
    MapMemory,
    FlushMemory,
    WaitPriorPresent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementHostScanoutError {
    EmptyExtent,
    SizeOverflow,
    /// The frame the caller published is shorter than `width * height * 4`.
    ShortFrame {
        required: usize,
        actual: usize,
    },
    /// This host cannot blit out of a LINEAR `B8G8R8A8_UNORM` image, so a
    /// CPU-written scanout has no route to the swapchain at all.
    LinearBlitUnsupported(vk::FormatFeatureFlags),
    NoMemoryType,
    Driver {
        phase: ReplacementHostScanoutPhase,
        result: vk::Result,
    },
}

/// The persistent host mapping of the staging image.
///
/// A raw pointer is not `Send` and the staging owner lives behind the device
/// epoch's mutex, which must be. The mapping is created with the image, lives
/// exactly as long as it does, and is dereferenced only by the thread holding
/// that mutex. Saying so in a wrapper keeps it a pointer in the type system;
/// laundering it through a `usize` would hide the same claim behind an integer.
struct MappedScanout(*mut u8);

// SAFETY: see the type's documentation — ownership is exclusive under the
// epoch mutex and the mapping outlives every dereference.
unsafe impl Send for MappedScanout {}

/// One host-visible LINEAR image the published scanout is copied into and the
/// present blit reads from.
pub struct ReplacementHostScanout {
    image: vk::Image,
    memory: vk::DeviceMemory,
    mapped: MappedScanout,
    allocated: u64,
    coherent: bool,
    width: u32,
    height: u32,
    row_pitch: u64,
    offset: u64,
    /// Whether the image has ever left `PREINITIALIZED`.
    transitioned: bool,
    /// The fence of the present that last read these bytes. The image is one
    /// allocation shared by every host frame, so the next CPU write must not
    /// begin until that present has completed.
    reading: Option<vk::Fence>,
}

impl ReplacementHostScanout {
    /// Copy one published BGRA frame in and describe it as a blit source.
    ///
    /// # Safety
    ///
    /// `context` must be the live epoch that owns this staging image, and the
    /// returned target and transitions must be recorded and submitted on that
    /// same epoch before another frame is staged.
    pub(crate) unsafe fn stage(
        &mut self,
        context: &DeviceContext,
        bgra: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(NativeImageTarget, NativeImageUseTransitions), ReplacementHostScanoutError> {
        let required = frame_len(width, height)?;
        if bgra.len() < required {
            return Err(ReplacementHostScanoutError::ShortFrame {
                required,
                actual: bgra.len(),
            });
        }
        unsafe { self.wait_for_prior_present(context) }?;
        unsafe { self.reshape(context, width, height) }?;
        let row_len = required / height as usize;
        for row in 0..height as usize {
            // SAFETY: the mapping covers `offset + height * row_pitch`, which
            // `reshape` took from the driver's own subresource layout.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bgra.as_ptr().add(row * row_len),
                    self.mapped
                        .0
                        .add(self.offset as usize + row * self.row_pitch as usize),
                    row_len,
                );
            }
        }
        if !self.coherent {
            unsafe {
                context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(self.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .map_err(|result| ReplacementHostScanoutError::Driver {
                phase: ReplacementHostScanoutPhase::FlushMemory,
                result,
            })?;
        }
        Ok((self.target(), self.transitions()))
    }

    /// Record which present now reads these bytes. Called only once the driver
    /// has accepted the submission that names them.
    pub fn read_by(&mut self, fence: vk::Fence) {
        self.transitioned = true;
        self.reading = Some(fence);
    }

    /// Release the recorded reader after a present that never reached the
    /// driver, so the next frame does not wait on a fence nothing will signal.
    pub fn released(&mut self) {
        self.reading = None;
    }

    unsafe fn wait_for_prior_present(
        &mut self,
        context: &DeviceContext,
    ) -> Result<(), ReplacementHostScanoutError> {
        let Some(fence) = self.reading.take() else {
            return Ok(());
        };
        unsafe { context.device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(|result| {
            ReplacementHostScanoutError::Driver {
                phase: ReplacementHostScanoutPhase::WaitPriorPresent,
                result,
            }
        })
    }

    /// Rebuild the image when the published geometry changes. Safe here and
    /// only here: the prior present has been waited out immediately above.
    unsafe fn reshape(
        &mut self,
        context: &DeviceContext,
        width: u32,
        height: u32,
    ) -> Result<(), ReplacementHostScanoutError> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        let replacement = unsafe { create_scanout(context, width, height) }?;
        let previous = std::mem::replace(self, replacement);
        unsafe { previous.destroy(context) };
        Ok(())
    }

    fn target(&self) -> NativeImageTarget {
        NativeImageTarget {
            image: self.image,
            view: vk::ImageView::null(),
            image_type: vk::ImageType::TYPE_2D,
            full_range: color_range(),
            usage: vk::ImageUsageFlags::TRANSFER_SRC,
            pixel_format: reims_vgpu_protocol::metal_pixel::MTL_FORMAT_BGRA8_UNORM,
            extent: vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    /// The barriers that make the host writes readable by the blit and leave
    /// the image writable again afterwards.
    fn transitions(&self) -> NativeImageUseTransitions {
        let held = if self.transitioned {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::PREINITIALIZED
        };
        let before = image_barrier_batch(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::HOST)
                .src_access_mask(vk::AccessFlags2::HOST_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(held)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(color_range()),
        );
        let after = image_barrier_batch(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .subresource_range(color_range()),
        );
        NativeImageUseTransitions { before, after }
    }

    /// # Safety
    ///
    /// No submission that reads this image may still be in flight.
    pub(crate) unsafe fn destroy(self, context: &DeviceContext) {
        unsafe {
            context.device.unmap_memory(self.memory);
            context.device.destroy_image(self.image, None);
            context.device.free_memory(self.memory, None);
        }
    }
}

fn image_barrier_batch(barrier: vk::ImageMemoryBarrier2<'static>) -> NativeBarrierBatch {
    let mut batch = NativeBarrierBatch::default();
    batch.images.push(barrier);
    batch
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

fn frame_len(width: u32, height: u32) -> Result<usize, ReplacementHostScanoutError> {
    if width == 0 || height == 0 {
        return Err(ReplacementHostScanoutError::EmptyExtent);
    }
    u64::from(width)
        .checked_mul(4)
        .and_then(|row| row.checked_mul(u64::from(height)))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(ReplacementHostScanoutError::SizeOverflow)
}

/// Build a host-visible LINEAR BGRA staging image at exactly `width`x`height`.
///
/// # Safety
///
/// `context` must be a live device epoch, and the returned image belongs to it
/// until [`ReplacementHostScanout::destroy`].
pub(crate) unsafe fn create_scanout(
    context: &DeviceContext,
    width: u32,
    height: u32,
) -> Result<ReplacementHostScanout, ReplacementHostScanoutError> {
    frame_len(width, height)?;
    let format = vk::Format::B8G8R8A8_UNORM;
    let features = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.pd, format)
            .linear_tiling_features
    };
    if !features.contains(vk::FormatFeatureFlags::BLIT_SRC) {
        return Err(ReplacementHostScanoutError::LinearBlitUnsupported(features));
    }
    let driver = |phase, result| ReplacementHostScanoutError::Driver { phase, result };
    let image = unsafe {
        context.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::PREINITIALIZED),
            None,
        )
    }
    .map_err(|result| driver(ReplacementHostScanoutPhase::CreateImage, result))?;
    let mut owned = ReplacementHostScanout {
        image,
        memory: vk::DeviceMemory::null(),
        mapped: MappedScanout(std::ptr::null_mut()),
        allocated: 0,
        coherent: false,
        width,
        height,
        row_pitch: 0,
        offset: 0,
        transitioned: false,
        reading: None,
    };
    let requirements = unsafe { context.device.get_image_memory_requirements(owned.image) };
    let Some(memory_type) = context.memory_type_for(
        requirements.memory_type_bits,
        requirements.size,
        crate::memory::MemoryClass::Upload,
    ) else {
        unsafe { context.device.destroy_image(owned.image, None) };
        return Err(ReplacementHostScanoutError::NoMemoryType);
    };
    owned.allocated = requirements.size;
    match unsafe {
        context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => owned.memory = memory,
        Err(result) => {
            unsafe { context.device.destroy_image(owned.image, None) };
            return Err(driver(ReplacementHostScanoutPhase::AllocateMemory, result));
        }
    }
    if let Err(result) = unsafe {
        context
            .device
            .bind_image_memory(owned.image, owned.memory, 0)
    } {
        unsafe { partial_destroy(context, &owned) };
        return Err(driver(ReplacementHostScanoutPhase::BindMemory, result));
    }
    match unsafe {
        context.device.map_memory(
            owned.memory,
            0,
            owned.allocated,
            vk::MemoryMapFlags::empty(),
        )
    } {
        Ok(mapped) => owned.mapped = MappedScanout(mapped.cast()),
        Err(result) => {
            unsafe { partial_destroy(context, &owned) };
            return Err(driver(ReplacementHostScanoutPhase::MapMemory, result));
        }
    }
    let layout = unsafe {
        context.device.get_image_subresource_layout(
            owned.image,
            vk::ImageSubresource::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .array_layer(0),
        )
    };
    owned.row_pitch = layout.row_pitch;
    owned.offset = layout.offset;
    owned.coherent = context.mapped_memory_kind(memory_type).coherent;
    Ok(owned)
}

unsafe fn partial_destroy(context: &DeviceContext, owned: &ReplacementHostScanout) {
    unsafe {
        context.device.destroy_image(owned.image, None);
        context.device.free_memory(owned.memory, None);
    }
}

/// Why a host-published scanout frame did not reach an acquired swapchain
/// image.
#[derive(Debug)]
pub enum ReplacementHostScanoutPresentError {
    /// No native window is attached to this device epoch.
    NotAttached,
    Scanout(ReplacementHostScanoutError),
    Window(crate::replacement_window_present::ReplacementWindowPresentPrepareError),
    Queue(crate::replacement_queue::ReplacementQueueError),
    Reservation(crate::replacement_window_present::ReplacementWindowPresentStateError),
}
