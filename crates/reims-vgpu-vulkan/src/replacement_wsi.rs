//! Swapchain ownership used only by replacement presentation.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    engine::context::DeviceContext,
    replacement_image_transition::{NativeImageTarget, NativeImageUseTransitions},
    replacement_queue::{ReplacementPresentRecording, ReplacementQueuePresent},
    replacement_window_present::{
        PreparedReplacementWindowPresent, ReplacementWindowPresentDispatch,
        ReplacementWindowPresentOutcome, ReplacementWindowPresentPrepareError,
        ReplacementWindowPresentStateError,
    },
};
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

const PRESENT_IN_FLIGHT: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementWindowNativeOperation {
    CreateSurface,
    SurfaceSupport,
    CreateCommandPool,
    AllocateCommandBuffers,
    CreateAcquireSemaphore,
    CreateFence,
    SurfaceCapabilities,
    SurfaceFormats,
    SurfacePresentModes,
    CreateSwapchain,
    GetSwapchainImages,
    CreateRenderSemaphore,
    FenceStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementWindowNativeError {
    SwapchainUnavailable,
    QueueCannotPresent {
        queue_family: u32,
    },
    SwapchainLacksTransferDestination,
    NoSurfaceFormat,
    NoCompositeAlpha,
    Vulkan {
        operation: ReplacementWindowNativeOperation,
        result: vk::Result,
    },
}

impl ReplacementWindowNativeError {
    const fn vulkan(operation: ReplacementWindowNativeOperation, result: vk::Result) -> Self {
        Self::Vulkan { operation, result }
    }
}

struct PresentFrame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    fence: vk::Fence,
    state: PresentFrameState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresentFrameState {
    Free,
    Reserved,
    Submitted,
}

impl PresentFrameState {
    fn reserve(&mut self) -> bool {
        if *self != Self::Free {
            return false;
        }
        *self = Self::Reserved;
        true
    }

    fn accept(&mut self) -> bool {
        if *self != Self::Reserved {
            return false;
        }
        *self = Self::Submitted;
        true
    }

    fn abandon(&mut self) -> bool {
        if *self != Self::Reserved {
            return false;
        }
        *self = Self::Free;
        true
    }
}

pub(crate) struct ReplacementSwapchainPresenter {
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    desired_extent: vk::Extent2D,
    recreate_pending: bool,
    command_pool: vk::CommandPool,
    frames: Vec<PresentFrame>,
    render_finished: Vec<vk::Semaphore>,
    retired_swapchains: Vec<(vk::SwapchainKHR, Vec<vk::Semaphore>)>,
    frame_index: usize,
}

impl ReplacementSwapchainPresenter {
    pub(crate) unsafe fn create(
        context: &DeviceContext,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, ReplacementWindowNativeError> {
        if !context.swapchain {
            return Err(ReplacementWindowNativeError::SwapchainUnavailable);
        }
        let surface = unsafe {
            ash_window::create_surface(&context._entry, &context.instance, display, window, None)
        }
        .map_err(|result| {
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::CreateSurface,
                result,
            )
        })?;
        let surface_loader = ash::khr::surface::Instance::new(&context._entry, &context.instance);
        let present_capable = unsafe {
            surface_loader.get_physical_device_surface_support(context.pd, context.gq, surface)
        }
        .map_err(|result| {
            unsafe { surface_loader.destroy_surface(surface, None) };
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::SurfaceSupport,
                result,
            )
        })?;
        if !present_capable {
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err(ReplacementWindowNativeError::QueueCannotPresent {
                queue_family: context.gq,
            });
        }
        let command_pool = unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.gq)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|result| {
            unsafe { surface_loader.destroy_surface(surface, None) };
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::CreateCommandPool,
                result,
            )
        })?;
        let commands = unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(PRESENT_IN_FLIGHT as u32),
            )
        }
        .map_err(|result| {
            unsafe {
                context.device.destroy_command_pool(command_pool, None);
                surface_loader.destroy_surface(surface, None);
            }
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::AllocateCommandBuffers,
                result,
            )
        })?;
        let mut frames = Vec::with_capacity(commands.len());
        for command_buffer in commands {
            let image_available = match unsafe {
                context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(value) => value,
                Err(result) => {
                    unsafe {
                        destroy_frames(&context.device, frames.drain(..));
                        context.device.destroy_command_pool(command_pool, None);
                        surface_loader.destroy_surface(surface, None);
                    }
                    return Err(ReplacementWindowNativeError::vulkan(
                        ReplacementWindowNativeOperation::CreateAcquireSemaphore,
                        result,
                    ));
                }
            };
            let fence = match unsafe {
                context.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            } {
                Ok(value) => value,
                Err(result) => {
                    unsafe {
                        context.device.destroy_semaphore(image_available, None);
                        destroy_frames(&context.device, frames.drain(..));
                        context.device.destroy_command_pool(command_pool, None);
                        surface_loader.destroy_surface(surface, None);
                    }
                    return Err(ReplacementWindowNativeError::vulkan(
                        ReplacementWindowNativeOperation::CreateFence,
                        result,
                    ));
                }
            };
            frames.push(PresentFrame {
                command_buffer,
                image_available,
                fence,
                state: PresentFrameState::Free,
            });
        }
        let mut presenter = Self {
            surface_loader,
            surface,
            swapchain_loader: ash::khr::swapchain::Device::new(&context.instance, &context.device),
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            desired_extent: vk::Extent2D {
                width: width.max(1),
                height: height.max(1),
            },
            recreate_pending: true,
            command_pool,
            frames,
            render_finished: Vec::new(),
            retired_swapchains: Vec::new(),
            frame_index: 0,
        };
        if let Err(error) = unsafe { presenter.recreate(context) } {
            unsafe { presenter.destroy_after_idle(context) };
            return Err(error);
        }
        Ok(presenter)
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let requested = vk::Extent2D {
            width: width.max(1),
            height: height.max(1),
        };
        if requested != self.desired_extent {
            self.recreate_pending = true;
        }
        self.desired_extent = requested;
    }

    pub(crate) fn recreate_pending(&self) -> bool {
        self.swapchain == vk::SwapchainKHR::null() || self.recreate_pending
    }

    pub(crate) fn image_count(&self) -> u32 {
        u32::try_from(self.images.len()).expect("Vulkan swapchain image count fits u32")
    }

    unsafe fn retire_completed(
        &mut self,
        context: &DeviceContext,
    ) -> Result<bool, ReplacementWindowNativeError> {
        for frame in &mut self.frames {
            if frame.state != PresentFrameState::Submitted {
                continue;
            }
            let signaled =
                unsafe { context.device.get_fence_status(frame.fence) }.map_err(|result| {
                    ReplacementWindowNativeError::vulkan(
                        ReplacementWindowNativeOperation::FenceStatus,
                        result,
                    )
                })?;
            if signaled {
                frame.state = PresentFrameState::Free;
            }
        }
        Ok(self.frames[self.frame_index].state == PresentFrameState::Free)
    }

    pub(crate) unsafe fn recreate_deferred(
        &mut self,
        context: &DeviceContext,
    ) -> Result<bool, ReplacementWindowNativeError> {
        unsafe { self.retire_completed(context) }?;
        if self
            .frames
            .iter()
            .any(|frame| frame.state != PresentFrameState::Free)
        {
            return Ok(false);
        }
        unsafe { self.recreate(context) }?;
        Ok(true)
    }

    unsafe fn recreate(
        &mut self,
        context: &DeviceContext,
    ) -> Result<(), ReplacementWindowNativeError> {
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(context.pd, self.surface)
        }
        .map_err(|result| {
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::SurfaceCapabilities,
                result,
            )
        })?;
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(ReplacementWindowNativeError::SwapchainLacksTransferDestination);
        }
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(context.pd, self.surface)
        }
        .map_err(|result| {
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::SurfaceFormats,
                result,
            )
        })?;
        let format = formats
            .iter()
            .find(|format| {
                format.format == crate::translate::pixel::SCANOUT_FORMAT
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or(ReplacementWindowNativeError::NoSurfaceFormat)?;
        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(context.pd, self.surface)
        }
        .map_err(|result| {
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::SurfacePresentModes,
                result,
            )
        })?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: self
                    .desired_extent
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: self
                    .desired_extent
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let image_count = swapchain_image_count(caps.min_image_count, caps.max_image_count);
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|flag| caps.supported_composite_alpha.contains(*flag))
        .ok_or(ReplacementWindowNativeError::NoCompositeAlpha)?;
        let previous = self.swapchain;
        let swapchain = unsafe {
            self.swapchain_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(image_count)
                    .image_format(format.format)
                    .image_color_space(format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(composite_alpha)
                    .present_mode(present_mode)
                    .clipped(true)
                    .old_swapchain(previous),
                None,
            )
        }
        .map_err(|result| {
            ReplacementWindowNativeError::vulkan(
                ReplacementWindowNativeOperation::CreateSwapchain,
                result,
            )
        })?;
        let images = match unsafe { self.swapchain_loader.get_swapchain_images(swapchain) } {
            Ok(images) => images,
            Err(result) => {
                unsafe { self.swapchain_loader.destroy_swapchain(swapchain, None) };
                return Err(ReplacementWindowNativeError::vulkan(
                    ReplacementWindowNativeOperation::GetSwapchainImages,
                    result,
                ));
            }
        };
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            match unsafe {
                context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(semaphore) => render_finished.push(semaphore),
                Err(result) => {
                    unsafe {
                        for semaphore in render_finished.drain(..) {
                            context.device.destroy_semaphore(semaphore, None);
                        }
                        self.swapchain_loader.destroy_swapchain(swapchain, None);
                    }
                    return Err(ReplacementWindowNativeError::vulkan(
                        ReplacementWindowNativeOperation::CreateRenderSemaphore,
                        result,
                    ));
                }
            }
        }
        if previous != vk::SwapchainKHR::null() {
            self.retired_swapchains
                .push((previous, std::mem::take(&mut self.render_finished)));
        }
        self.swapchain = swapchain;
        self.images = images;
        self.render_finished = render_finished;
        self.format = format.format;
        self.extent = extent;
        self.desired_extent = extent;
        self.recreate_pending = false;
        Ok(())
    }

    pub(crate) unsafe fn prepare(
        &mut self,
        context: &DeviceContext,
        source: NativeImageTarget,
        transitions: &NativeImageUseTransitions,
        generation: reims_vgpu_protocol::SwapchainGenerationId,
    ) -> Result<ReplacementWindowPresentDispatch, ReplacementWindowPresentPrepareError> {
        let (source_format, _) = crate::translate::pixel::verbatim_texel(source.pixel_format)
            .ok_or(ReplacementWindowPresentPrepareError::PixelFormat(
                source.pixel_format,
            ))?;
        let source_features = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.pd, source_format)
        }
        .optimal_tiling_features;
        let required_source =
            vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        if !source_features.contains(required_source) {
            return Err(ReplacementWindowPresentPrepareError::SourceBlitUnsupported(
                source_features,
            ));
        }
        let destination_features = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.pd, self.format)
        }
        .optimal_tiling_features;
        if !destination_features.contains(vk::FormatFeatureFlags::BLIT_DST) {
            return Err(
                ReplacementWindowPresentPrepareError::DestinationBlitUnsupported(
                    destination_features,
                ),
            );
        }
        if !unsafe { self.retire_completed(context) }
            .map_err(ReplacementWindowPresentPrepareError::Window)?
        {
            return Ok(ReplacementWindowPresentDispatch::Busy);
        }
        let slot = self.frame_index;
        let command_buffer = self.frames[slot].command_buffer;
        let image_available = self.frames[slot].image_available;
        let fence = self.frames[slot].fence;
        let (image_index, acquire_suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                0,
                image_available,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                return Ok(ReplacementWindowPresentDispatch::Busy);
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                return Ok(ReplacementWindowPresentDispatch::Busy);
            }
            Err(result) => {
                return Err(ReplacementWindowPresentPrepareError::Window(
                    ReplacementWindowNativeError::vulkan(
                        ReplacementWindowNativeOperation::CreateAcquireSemaphore,
                        result,
                    ),
                ));
            }
        };
        let recording = ReplacementPresentRecording {
            slot,
            command_pool: self.command_pool,
            command_buffer,
            fence,
        };
        let recording = unsafe {
            crate::replacement_present::record_present_blit(
                &context.device,
                recording,
                source,
                transitions,
                self.images[image_index as usize],
                self.extent,
            )
        }
        .map_err(ReplacementWindowPresentPrepareError::Record)?;
        assert!(self.frames[slot].state.reserve());
        Ok(ReplacementWindowPresentDispatch::Prepared(
            PreparedReplacementWindowPresent {
                recording,
                present: ReplacementQueuePresent {
                    loader: self.swapchain_loader.clone(),
                    acquire_wait: image_available,
                    render_finished: self.render_finished[image_index as usize],
                    swapchain: self.swapchain,
                    image_index,
                },
                acquire_suboptimal,
                swapchain: generation,
                swapchain_images: self.image_count(),
                image_index,
            },
        ))
    }

    pub(crate) fn accept(
        &mut self,
        slot: usize,
        acquire_suboptimal: bool,
        present_result: Result<bool, vk::Result>,
    ) -> Result<ReplacementWindowPresentOutcome, ReplacementWindowPresentStateError> {
        let Some(frame) = self.frames.get_mut(slot) else {
            return Err(ReplacementWindowPresentStateError::SlotAbsent);
        };
        if !frame.state.accept() {
            return Err(ReplacementWindowPresentStateError::SlotNotReserved);
        }
        self.frame_index = (slot + 1) % self.frames.len();
        match present_result {
            Ok(present_suboptimal) => {
                let suboptimal = acquire_suboptimal || present_suboptimal;
                self.recreate_pending |= suboptimal;
                Ok(ReplacementWindowPresentOutcome::Presented {
                    width: self.extent.width,
                    height: self.extent.height,
                    swapchain_images: self.images.len(),
                    suboptimal,
                })
            }
            Err(result) => {
                self.recreate_pending |= result == vk::Result::ERROR_OUT_OF_DATE_KHR;
                Ok(ReplacementWindowPresentOutcome::Refused(result))
            }
        }
    }

    pub(crate) fn abandon(
        &mut self,
        slot: usize,
    ) -> Result<(), ReplacementWindowPresentStateError> {
        let Some(frame) = self.frames.get_mut(slot) else {
            return Err(ReplacementWindowPresentStateError::SlotAbsent);
        };
        if !frame.state.abandon() {
            return Err(ReplacementWindowPresentStateError::SlotNotReserved);
        }
        self.recreate_pending = true;
        Ok(())
    }

    pub(crate) unsafe fn destroy_after_idle(&mut self, context: &DeviceContext) {
        unsafe { destroy_frames(&context.device, self.frames.drain(..)) };
        for semaphore in self.render_finished.drain(..) {
            unsafe { context.device.destroy_semaphore(semaphore, None) };
        }
        for (swapchain, semaphores) in self.retired_swapchains.drain(..) {
            for semaphore in semaphores {
                unsafe { context.device.destroy_semaphore(semaphore, None) };
            }
            unsafe { self.swapchain_loader.destroy_swapchain(swapchain, None) };
        }
        unsafe { context.device.destroy_command_pool(self.command_pool, None) };
        if self.swapchain != vk::SwapchainKHR::null() {
            unsafe {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None)
            };
            self.swapchain = vk::SwapchainKHR::null();
        }
        unsafe { self.surface_loader.destroy_surface(self.surface, None) };
    }
}

unsafe fn destroy_frames(device: &ash::Device, frames: impl Iterator<Item = PresentFrame>) {
    for frame in frames {
        unsafe {
            device.destroy_fence(frame.fence, None);
            device.destroy_semaphore(frame.image_available, None);
        }
    }
}

fn swapchain_image_count(minimum: u32, maximum: u32) -> u32 {
    let desired = minimum.saturating_add(1);
    if maximum == 0 {
        desired
    } else {
        desired.min(maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapchain_depth_stays_within_the_surface_contract() {
        assert_eq!(swapchain_image_count(2, 0), 3);
        assert_eq!(swapchain_image_count(2, 2), 2);
        assert_eq!(swapchain_image_count(u32::MAX, 0), u32::MAX);
    }
}
