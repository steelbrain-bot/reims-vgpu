//! QEMU-facing entry surface for the replacement device owner.
//!
//! The registry below is the only process-wide device state. The exported C
//! ABI calls these adapters as one surface; protocol and execution decisions
//! remain in the replacement coordinator.

mod replacement_registry;

use crate::{
    qemu::host_ops::ReimsVgpuHostOps,
    runtime::{host::HostAction, replacement_scanout::ScanoutCopyResult},
};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleFeed {
    Firmware,
    Product,
}

impl ConsoleFeed {
    pub fn kind(&self) -> u32 {
        match self {
            Self::Firmware => 0,
            Self::Product => 2,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorGlyphInfo {
    pub width: u32,
    pub height: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    pub pixel_count: u32,
}

pub(crate) fn replacement_device_create(
    ops: Option<ReimsVgpuHostOps>,
    page_shift: u32,
) -> Option<u64> {
    replacement_registry::create(ops, page_shift)
}

pub(crate) fn replacement_device_reset(id: u64) -> bool {
    match replacement_registry::reset(id) {
        Ok(()) => true,
        Err(reason) => {
            crate::observe::fail(format!(
                "replacement_device_reset_refused reason={reason:?}"
            ));
            false
        }
    }
}

pub(crate) fn replacement_device_destroy(id: u64) -> bool {
    replacement_registry::destroy(id)
}

pub(crate) fn replacement_device_gfx_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    replacement_registry::gfx_read(id, offset, size)
}

pub(crate) fn replacement_device_gfx_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    replacement_registry::gfx_write(id, offset, data, size)
}

pub(crate) fn replacement_device_iosfc_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    replacement_registry::iosfc_read(id, offset, size)
}

pub(crate) fn replacement_device_iosfc_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    replacement_registry::iosfc_write(id, offset, data, size)
}

pub(crate) fn replacement_device_drain(id: u64) -> bool {
    replacement_registry::drain(id)
}

pub(crate) fn replacement_device_poll(id: u64) -> bool {
    replacement_registry::poll(id)
}

pub(crate) fn replacement_device_pop_action(id: u64) -> Option<HostAction> {
    replacement_registry::pop_action(id)
}

pub(crate) fn replacement_device_console_feed(id: u64) -> Option<ConsoleFeed> {
    Some(if replacement_registry::product_presented(id)? {
        ConsoleFeed::Product
    } else {
        ConsoleFeed::Firmware
    })
}

pub(crate) fn replacement_device_scanout_may_paint(id: u64, mapping_id: u32) -> Option<bool> {
    replacement_registry::console_frame_may_paint(id, mapping_id)
}

pub(crate) fn replacement_device_scanout_copy(
    id: u64,
    mapping_id: u32,
    destination: &mut [u8],
    destination_stride: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> ScanoutCopyResult {
    match replacement_registry::copy_console_frame(
        id,
        mapping_id,
        generation,
        destination,
        destination_stride,
        width,
        height,
    ) {
        Ok(()) => ScanoutCopyResult::Painted,
        Err(crate::runtime::replacement_coordinator::ReplacementConsoleFrameError::Missing {
            ..
        }) => ScanoutCopyResult::Unchanged,
        Err(reason) => {
            crate::observe::fail(format!(
                "replacement_console_copy_refused reason={reason:?}"
            ));
            ScanoutCopyResult::Failed
        }
    }
}

pub(crate) fn replacement_device_efi_console_copy(
    id: u64,
    destination: &mut [u8],
    destination_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    replacement_registry::copy_efi_console(id, destination, destination_stride, width, height)
}

pub(crate) fn replacement_device_cursor_glyph_info(id: u64) -> Option<CursorGlyphInfo> {
    let (width, height, hot_x, hot_y, pixels) = replacement_registry::cursor_glyph(id)?;
    Some(CursorGlyphInfo {
        width: u32::from(width),
        height: u32::from(height),
        hot_x: u32::from(hot_x),
        hot_y: u32::from(hot_y),
        pixel_count: u32::try_from(pixels.len()).ok()?,
    })
}

pub(crate) fn replacement_device_cursor_glyph_copy(
    id: u64,
    destination: &mut [u32],
) -> Option<usize> {
    let (_, _, _, _, pixels) = replacement_registry::cursor_glyph(id)?;
    let count = destination.len().min(pixels.len());
    destination[..count].copy_from_slice(&pixels[..count]);
    Some(count)
}

pub(crate) fn replacement_device_window_start(id: u64, width: u32, height: u32) -> bool {
    replacement_registry::window_start(id, width, height)
}

pub(crate) fn replacement_device_window_run_main(id: u64) -> bool {
    replacement_registry::window_run_main(id)
}

pub(crate) fn replacement_device_window_stop(id: u64) -> bool {
    replacement_registry::window_stop(id)
}

pub(crate) fn replacement_device_window_set_early_fb(
    id: u64,
    pointer: usize,
    stride: u32,
    width: u32,
    height: u32,
) -> bool {
    replacement_registry::window_set_early_fb(id, pointer, stride, width, height)
}

pub fn backend_name() -> &'static str {
    "vulkan"
}

pub fn unwind_safe<T, F>(entry: &'static str, f: F, on_panic: T) -> T
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    crate::observe::panic::arm();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            crate::observe::panic::report(entry, payload.as_ref());
            on_panic
        }
    }
}
