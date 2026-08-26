//! Per-device-epoch host-window ownership for replacement presentation.
//!
//! Swapchain policy remains in the established window owner, while queue
//! synchronization and native source ownership stay in the replacement epoch.
//! Acquiring an image produces an explicit reservation that must be accepted
//! or abandoned after the replacement queue owner replies.

use crate::{
    engine::context::DeviceContext,
    replacement_image_transition::{NativeImageTarget, NativeImageUseTransitions},
    replacement_present::ReplacementPresentRecordError,
    replacement_queue::{
        ReplacementPresentRecording, ReplacementQueueError, ReplacementQueuePresent,
    },
    replacement_wsi::{ReplacementSwapchainPresenter, ReplacementWindowNativeError},
};
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::{atomic::AtomicU64, Arc};

#[derive(Debug)]
pub enum ReplacementWindowAttachError {
    AlreadyAttached,
    Window(ReplacementWindowNativeError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementWindowDetachError {
    Queue(ReplacementQueueError),
}

#[derive(Debug)]
pub enum ReplacementWindowPresentPrepareError {
    NotAttached,
    SwapchainGenerationExhausted,
    Window(ReplacementWindowNativeError),
    PixelFormat(u16),
    SourceBlitUnsupported(vk::FormatFeatureFlags),
    DestinationBlitUnsupported(vk::FormatFeatureFlags),
    Record(ReplacementPresentRecordError),
}

#[derive(Debug)]
pub enum ReplacementWindowPresentDispatch {
    Busy,
    Prepared(PreparedReplacementWindowPresent),
}

#[derive(Debug)]
#[must_use = "an acquired replacement window image must be submitted or abandoned"]
pub struct PreparedReplacementWindowPresent {
    pub recording: ReplacementPresentRecording,
    pub present: ReplacementQueuePresent,
    pub acquire_suboptimal: bool,
    pub swapchain: reims_vgpu_protocol::SwapchainGenerationId,
    pub swapchain_images: u32,
    pub image_index: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementWindowSnapshot {
    pub generation: reims_vgpu_protocol::SwapchainGenerationId,
    pub image_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementWindowPresentOutcome {
    Presented {
        width: u32,
        height: u32,
        swapchain_images: usize,
        suboptimal: bool,
    },
    Refused(vk::Result),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementWindowPresentStateError {
    SlotAbsent,
    SlotNotReserved,
}

pub struct ReplacementWindowPresenter {
    inner: ReplacementSwapchainPresenter,
    generation: reims_vgpu_protocol::SwapchainGenerationId,
    generations: Arc<AtomicU64>,
}

impl ReplacementWindowPresenter {
    pub(crate) unsafe fn create(
        context: &DeviceContext,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
        generations: Arc<AtomicU64>,
    ) -> Result<Self, ReplacementWindowNativeError> {
        let inner = unsafe {
            ReplacementSwapchainPresenter::create(context, display, window, width, height)
        }?;
        let generation = allocate_swapchain_generation(&generations)
            .expect("the first swapchain generation is always representable");
        Ok(Self {
            inner,
            generation,
            generations,
        })
    }

    pub(crate) fn snapshot(&self) -> ReplacementWindowSnapshot {
        ReplacementWindowSnapshot {
            generation: self.generation,
            image_count: self.inner.image_count(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.inner.resize(width, height);
    }

    pub(crate) unsafe fn prepare(
        &mut self,
        context: &DeviceContext,
        source: NativeImageTarget,
        transitions: &NativeImageUseTransitions,
    ) -> Result<ReplacementWindowPresentDispatch, ReplacementWindowPresentPrepareError> {
        if self.inner.recreate_pending() {
            if !unsafe { self.inner.recreate_deferred(context) }
                .map_err(ReplacementWindowPresentPrepareError::Window)?
            {
                return Ok(ReplacementWindowPresentDispatch::Busy);
            }
            self.generation = allocate_swapchain_generation(&self.generations)
                .ok_or(ReplacementWindowPresentPrepareError::SwapchainGenerationExhausted)?;
        }
        unsafe {
            self.inner
                .prepare(context, source, transitions, self.generation)
        }
    }

    pub(crate) fn accept(
        &mut self,
        slot: usize,
        acquire_suboptimal: bool,
        present_result: Result<bool, vk::Result>,
    ) -> Result<ReplacementWindowPresentOutcome, ReplacementWindowPresentStateError> {
        self.inner.accept(slot, acquire_suboptimal, present_result)
    }

    pub(crate) fn abandon(
        &mut self,
        slot: usize,
    ) -> Result<(), ReplacementWindowPresentStateError> {
        self.inner.abandon(slot)
    }

    pub(crate) unsafe fn destroy_after_idle(&mut self, context: &DeviceContext) {
        unsafe { self.inner.destroy_after_idle(context) };
    }
}

fn allocate_swapchain_generation(
    generations: &AtomicU64,
) -> Option<reims_vgpu_protocol::SwapchainGenerationId> {
    generations
        .fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |next| next.checked_add(1),
        )
        .ok()
        .map(reims_vgpu_protocol::SwapchainGenerationId::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn swapchain_generations_are_epoch_monotonic_and_never_wrap() {
        let generations = AtomicU64::new(1);
        assert_eq!(
            allocate_swapchain_generation(&generations),
            Some(reims_vgpu_protocol::SwapchainGenerationId::new(1))
        );
        assert_eq!(
            allocate_swapchain_generation(&generations),
            Some(reims_vgpu_protocol::SwapchainGenerationId::new(2))
        );
        generations.store(u64::MAX, Ordering::Release);
        assert_eq!(allocate_swapchain_generation(&generations), None);
        assert_eq!(generations.load(Ordering::Acquire), u64::MAX);
    }
}
