//! QEMU-facing ownership adapter for the replacement device coordinator.
//!
//! This module translates ABI-shaped MMIO, host-action, console, and cursor
//! calls. It owns no protocol or presentation policy; those answers come from
//! the replacement coordinator and its session-owned state.

use crate::{
    qemu::host_ops::{NullHost, QemuHost, ReimsVgpuHostOps},
    runtime::{
        host::HostMemory,
        replacement_coordinator::{
            ReplacementConsoleFrameError, ReplacementDeviceCoordinator, ReplacementDeviceResetError,
        },
        HostAction,
    },
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};

struct ReplacementDeviceInner {
    device: ReplacementDeviceCoordinator<()>,
    actions: VecDeque<HostAction>,
}

struct BoundReplacementDevice {
    inner: Mutex<ReplacementDeviceInner>,
    prompt_actions: Mutex<VecDeque<HostAction>>,
    ops: Option<ReimsVgpuHostOps>,
    generation: AtomicU64,
    loss_reported: AtomicBool,
    reported_failures: Mutex<BTreeSet<(u64, &'static str, String)>>,
    reported_pipeline_failures: Mutex<
        BTreeSet<(
            &'static str,
            String,
            reims_vgpu_core::PipelineFailureStage,
            String,
        )>,
    >,
    reported_pipeline_states:
        Mutex<BTreeSet<(&'static str, String, reims_vgpu_core::PipelineState)>>,
    reported_transaction_states: Mutex<BTreeSet<(u64, String)>>,
    reported_blocked_drains: Mutex<BTreeSet<String>>,
    #[cfg(feature = "host-window")]
    window: Mutex<Option<ReplacementWindowLink>>,
    #[cfg(feature = "host-window")]
    early_fb: Mutex<Option<ReplacementEarlyFramebuffer>>,
}

#[cfg(feature = "host-window")]
struct ReplacementWindowLink {
    frames: crate::host_window::present::FrameSlot,
    wake: crate::host_window::present::WindowWakeHandle,
    stop: crate::host_window::present::StopFlag,
    thread: Option<std::thread::JoinHandle<Result<(), crate::host_window::present::WindowError>>>,
    seq: u64,
    #[cfg(target_os = "macos")]
    exited: crate::host_window::present::ExitedFlag,
}

#[cfg(feature = "host-window")]
#[derive(Clone, Copy)]
struct ReplacementEarlyFramebuffer {
    pointer: usize,
    stride: u32,
    width: u32,
    height: u32,
}

#[cfg(feature = "host-window")]
struct ReplacementWindowService(std::sync::Weak<BoundReplacementDevice>);

#[cfg(feature = "host-window")]
impl std::fmt::Debug for ReplacementWindowService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ReplacementWindowService")
    }
}

#[cfg(feature = "host-window")]
impl crate::runtime::replacement_services::WindowPresentationService for ReplacementWindowService {
    fn attach_window_presenter(
        &self,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<(), crate::runtime::replacement_services::WindowPresentationError> {
        let slot = self.0.upgrade().ok_or_else(|| {
            crate::runtime::replacement_services::WindowPresentationError::replacement(
                "attach",
                "device was destroyed".to_string(),
            )
        })?;
        let inner = slot.inner.lock();
        unsafe { inner.device.attach_window(display, window, width, height) }.map_err(|reason| {
            crate::runtime::replacement_services::WindowPresentationError::replacement(
                "attach",
                format!("{reason:?}"),
            )
        })
    }

    fn resize_window_presenter(&self, width: u32, height: u32) {
        if let Some(slot) = self.0.upgrade() {
            slot.inner.lock().device.resize_window(width, height);
        }
    }

    fn present_window_frame(
        &self,
        frame: Option<crate::runtime::replacement_services::WindowPresentationFrame<'_>>,
    ) -> Result<
        crate::runtime::replacement_services::WindowPresentOutcome,
        crate::runtime::replacement_services::WindowPresentationError,
    > {
        use crate::runtime::replacement_services::{
            WindowPresentOutcome, WindowPresentationError, WindowPresentationPayload,
        };
        // A resident frame belongs to a guest Present transaction, which
        // submits itself through the replacement queue owner: the window loop
        // owns native handles and resize/input, and never manufactures a second
        // presentation transaction. A CPU frame has no such owner — it is the
        // host's own scanout publication, and it is the only thing the window
        // has to show before the guest's driver attaches.
        let Some(frame) = frame else {
            return Ok(WindowPresentOutcome::Busy);
        };
        let WindowPresentationPayload::CpuBgra(bgra) = frame.payload else {
            return Ok(WindowPresentOutcome::Busy);
        };
        let slot = self.0.upgrade().ok_or_else(|| {
            WindowPresentationError::replacement("present", "device was destroyed".to_string())
        })?;
        let inner = slot.inner.lock();
        let outcome = unsafe {
            inner
                .device
                .present_host_scanout_frame(bgra, frame.width, frame.height)
        }
        .map_err(|reason| WindowPresentationError::replacement("present", format!("{reason:?}")))?;
        Ok(match outcome {
            None => WindowPresentOutcome::Busy,
            Some(
                reims_vgpu_vulkan::replacement_window_present::ReplacementWindowPresentOutcome::Presented {
                    width,
                    height,
                    swapchain_images,
                    suboptimal,
                },
            ) => WindowPresentOutcome::Presented {
                route: reims_vgpu_core::PresentationRoute::CpuBgra,
                width,
                height,
                swapchain_images,
                suboptimal,
            },
            Some(
                reims_vgpu_vulkan::replacement_window_present::ReplacementWindowPresentOutcome::Refused(
                    result,
                ),
            ) => {
                return Err(WindowPresentationError::replacement(
                    "present",
                    format!("swapchain refused the frame: {result:?}"),
                ))
            }
        })
    }

    fn detach_window_presenter(&self) {
        if let Some(slot) = self.0.upgrade() {
            if let Err(reason) = unsafe { slot.inner.lock().device.detach_window() } {
                crate::observe::fail(format!(
                    "replacement_window_detach_refused reason={reason:?}"
                ));
            }
        }
    }
}

static REPLACEMENT_DEVICES: Lazy<Mutex<HashMap<u64, Arc<BoundReplacementDevice>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_REPLACEMENT_DEVICE: AtomicU64 = AtomicU64::new(1);

fn slot(id: u64) -> Option<Arc<BoundReplacementDevice>> {
    REPLACEMENT_DEVICES.lock().get(&id).cloned()
}

pub(super) fn create(ops: Option<ReimsVgpuHostOps>, page_shift: u32) -> Option<u64> {
    if !matches!(
        page_shift,
        crate::model::PAGE_SHIFT_X86 | crate::model::PAGE_SHIFT_ARM64E
    ) {
        crate::observe::fail(format!(
            "replacement_device_create_refused phase=page_geometry page_shift={page_shift}"
        ));
        return None;
    }
    let Some(id) = NEXT_REPLACEMENT_DEVICE
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(1)
        })
        .ok()
    else {
        crate::observe::fail(
            "replacement_device_create_refused phase=device_identity reason=exhausted",
        );
        return None;
    };
    let runtime = match crate::runtime::replacement_session::ReplacementRuntimeSession::create(
        reims_vgpu_protocol::SessionId::new(id),
        reims_vgpu_protocol::SessionGenerationId::new(1),
        reims_vgpu_protocol::VulkanDeviceEpochId::new(id),
        match u32::try_from(id) {
            Ok(id) => reims_vgpu_protocol::QueueOwnerId::new(id),
            Err(reason) => {
                crate::observe::fail(format!(
                    "replacement_device_create_refused phase=queue_identity reason={reason}"
                ));
                return None;
            }
        },
        2,
    ) {
        Ok(runtime) => runtime,
        Err(reason) => {
            crate::observe::fail(format!(
                "replacement_device_create_refused phase=runtime reason={reason:?}"
            ));
            return None;
        }
    };
    let device = match ReplacementDeviceCoordinator::new(runtime, page_shift) {
        Ok(device) => device,
        Err(reason) => {
            crate::observe::fail(format!(
                "replacement_device_create_refused phase=transport reason={reason:?}"
            ));
            return None;
        }
    };
    REPLACEMENT_DEVICES.lock().insert(
        id,
        Arc::new(BoundReplacementDevice {
            inner: Mutex::new(ReplacementDeviceInner {
                device,
                actions: VecDeque::new(),
            }),
            prompt_actions: Mutex::new(VecDeque::new()),
            ops,
            generation: AtomicU64::new(1),
            loss_reported: AtomicBool::new(false),
            reported_failures: Mutex::new(BTreeSet::new()),
            reported_pipeline_failures: Mutex::new(BTreeSet::new()),
            reported_pipeline_states: Mutex::new(BTreeSet::new()),
            reported_transaction_states: Mutex::new(BTreeSet::new()),
            reported_blocked_drains: Mutex::new(BTreeSet::new()),
            #[cfg(feature = "host-window")]
            window: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_fb: Mutex::new(None),
        }),
    );
    Some(id)
}

pub(super) fn reset(id: u64) -> Result<(), ReplacementDeviceResetError> {
    let slot = slot(id).ok_or(ReplacementDeviceResetError::DeviceAbsent)?;
    let next = slot
        .generation
        .load(Ordering::Acquire)
        .checked_add(1)
        .ok_or(ReplacementDeviceResetError::Generation(
            crate::runtime::replacement_session::ReplacementRuntimeResetError::Session(
                reims_vgpu_core::DeviceSessionError::GenerationDidNotAdvance,
            ),
        ))?;
    let mut inner = slot.inner.lock();
    let effect = inner
        .device
        .platform_reset(reims_vgpu_protocol::SessionGenerationId::new(next))?;
    crate::observe::off(format!(
        "replacement_platform_reset generation={} semantic_transactions={} unsubmitted={} submitted={} native_representations={} native_recordings={}",
        next,
        effect.abandonment.semantic_transactions,
        effect.abandonment.unsubmitted_transactions,
        effect.abandonment.submitted_transactions,
        effect.abandonment.native_representations,
        effect.abandonment.native_recordings,
    ));
    inner.actions.clear();
    slot.prompt_actions.lock().clear();
    slot.reported_failures.lock().clear();
    slot.reported_pipeline_failures.lock().clear();
    slot.reported_pipeline_states.lock().clear();
    slot.reported_transaction_states.lock().clear();
    slot.reported_blocked_drains.lock().clear();
    slot.generation.store(next, Ordering::Release);
    Ok(())
}

pub(super) fn destroy(id: u64) -> bool {
    #[cfg(feature = "host-window")]
    if slot(id).is_some() && !window_stop(id) {
        return false;
    }
    let removed = REPLACEMENT_DEVICES.lock().remove(&id);
    let present = removed.is_some();
    drop(removed);
    present
}

pub(super) fn gfx_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    let slot = slot(id)?;
    let mut inner = slot.inner.lock();
    let value = inner.device.gfx_read(offset, size);
    report_device_loss(&slot, &mut inner.device);
    Some(value)
}

pub(super) fn gfx_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    let mut inner = slot.inner.lock();
    let ReplacementDeviceInner { device, actions } = &mut *inner;
    match slot.ops {
        Some(ops) => device.gfx_write(
            &mut QemuHost::new(&ops, actions, &slot.prompt_actions),
            offset,
            data,
            size,
        ),
        None => device.gfx_write(&mut NullHost, offset, data, size),
    }
    report_device_loss(&slot, &mut inner.device);
    true
}

pub(super) fn iosfc_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    let slot = slot(id)?;
    let mut inner = slot.inner.lock();
    let value = inner.device.iosfc_read(offset, size);
    report_device_loss(&slot, &mut inner.device);
    Some(value)
}

pub(super) fn iosfc_write(id: u64, offset: u64, data: u64, _size: u32) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    let mut inner = slot.inner.lock();
    let ReplacementDeviceInner { device, actions } = &mut *inner;
    match slot.ops {
        Some(ops) => device.iosfc_write(
            &mut QemuHost::new(&ops, actions, &slot.prompt_actions),
            offset,
            data,
        ),
        None => device.iosfc_write(&mut NullHost, offset, data),
    }
    report_device_loss(&slot, &mut inner.device);
    true
}

pub(super) fn drain(id: u64) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    let mut inner = slot.inner.lock();
    let ReplacementDeviceInner { device, actions } = &mut *inner;
    match slot.ops {
        Some(ops) => {
            let _ = device.tick(&mut QemuHost::new(&ops, actions, &slot.prompt_actions));
        }
        None => {
            poll_display_refresh(device, &mut NullHost);
            let _ = device.tick(&mut NullHost);
        }
    }
    report_coordinator_failures(&slot, &inner.device);
    report_device_loss(&slot, &mut inner.device);
    crate::runtime::host_action_census::report();
    crate::runtime::contract_census::report();
    true
}

pub(super) fn poll(id: u64) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    let mut inner = slot.inner.lock();
    let ReplacementDeviceInner { device, actions } = &mut *inner;
    match slot.ops {
        Some(ops) => {
            let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
            if let Err(reason) = device.poll_display_online(&mut host) {
                if reason
                    != crate::runtime::replacement_session::ReplacementDisplayOnlineError::NativeLifetimeClosed
                {
                    crate::observe::fail(format!(
                        "replacement_display_online_refused reason={reason:?}"
                    ));
                }
            }
            poll_display_refresh(device, &mut host);
            let _ = device.tick(&mut host);
        }
        None => {
            if let Err(reason) = device.poll_display_online(&mut NullHost) {
                if reason
                    != crate::runtime::replacement_session::ReplacementDisplayOnlineError::NativeLifetimeClosed
                {
                    crate::observe::fail(format!(
                        "replacement_display_online_refused reason={reason:?}"
                    ));
                }
            }
            let _ = device.tick(&mut NullHost);
        }
    }
    report_coordinator_failures(&slot, &inner.device);
    report_device_loss(&slot, &mut inner.device);
    crate::runtime::host_action_census::report();
    crate::runtime::contract_census::report();
    #[cfg(feature = "host-window")]
    let product_presented = inner.device.product_presented();
    drop(inner);
    #[cfg(feature = "host-window")]
    publish_early_window_frame(&slot, product_presented);
    true
}

/// Raise the display refresh pulse for this poll and report a refusal by name.
///
/// Both poll arms call this rather than carrying the same two lines twice:
/// the lock-free arm and the locked one already diverged once on the display
/// handshake, and a copy is where the next divergence comes from.
fn poll_display_refresh(
    device: &mut ReplacementDeviceCoordinator<()>,
    host: &mut (impl crate::runtime::host::HostMemory + crate::runtime::host::HostControl),
) {
    if let Err(reason) = device.poll_display_refresh(host) {
        if reason
            != crate::runtime::replacement_session::ReplacementDisplayRefreshError::NativeLifetimeClosed
        {
            crate::observe::fail(format!(
                "replacement_display_refresh_refused reason={reason:?}"
            ));
        }
    }
}

fn report_coordinator_failures(
    slot: &BoundReplacementDevice,
    device: &ReplacementDeviceCoordinator<()>,
) {
    let mut reported = slot.reported_failures.lock();
    for (transaction, stage, reason) in device.cpu_failure_diagnostics() {
        let key = (transaction.get(), stage, reason);
        if reported.insert(key.clone()) {
            crate::observe::fail(format!(
                "replacement_cpu_transaction_refused transaction={} stage={} reason={}",
                key.0, key.1, key.2
            ));
        }
    }
    drop(reported);
    let mut reported = slot.reported_pipeline_failures.lock();
    for (kind, pipeline, stage, reason) in device.pipeline_failure_diagnostics() {
        let key = (kind, pipeline, stage, reason);
        if reported.insert(key.clone()) {
            crate::observe::fail(format!(
                "replacement_pipeline_refused kind={} pipeline={} stage={:?} reason={}",
                key.0, key.1, key.2, key.3
            ));
        }
    }
    drop(reported);
    let mut reported = slot.reported_pipeline_states.lock();
    for (kind, pipeline, state) in device.pipeline_state_diagnostics() {
        let key = (kind, pipeline, state);
        if reported.insert(key.clone()) {
            crate::observe::off(format!(
                "replacement_pipeline_state kind={} pipeline={} state={:?}",
                key.0, key.1, key.2
            ));
        }
    }
    drop(reported);
    let mut reported = slot.reported_transaction_states.lock();
    for (transaction, state) in device.transaction_state_diagnostics() {
        let key = (transaction, state);
        if reported.insert(key.clone()) {
            crate::observe::off(format!(
                "replacement_transaction_state transaction={} {}",
                key.0, key.1
            ));
        }
    }
    drop(reported);
    let mut reported = slot.reported_blocked_drains.lock();
    for reason in device.blocked_drain_diagnostics() {
        if reported.insert(reason.clone()) {
            crate::observe::fail(format!("replacement_transport_blocked {reason}"));
        }
    }
}

fn report_device_loss(
    slot: &BoundReplacementDevice,
    device: &mut ReplacementDeviceCoordinator<()>,
) {
    let state = device.vulkan_state();
    if state == reims_vgpu_core::VulkanDeviceEpochState::Active {
        return;
    }
    device.terminalize_device_loss();
    if !slot.loss_reported.swap(true, Ordering::AcqRel) {
        let effect = device
            .device_loss_effect()
            .expect("a closed replacement epoch is terminalized before reporting");
        crate::observe::fail(format!(
            "replacement_vulkan_epoch_closed state={state:?} semantic_transactions={} unsubmitted={} submitted={} native_representations={} native_recordings={} render_pipelines={} compute_pipelines={}",
            effect.abandonment.semantic_transactions,
            effect.abandonment.unsubmitted_transactions,
            effect.abandonment.submitted_transactions,
            effect.abandonment.native_representations,
            effect.abandonment.native_recordings,
            effect.session.render_pipelines,
            effect.session.compute_pipelines,
        ));
    }
}

pub(super) fn pop_action(id: u64) -> Option<HostAction> {
    let slot = slot(id)?;
    let mut prompt = slot.prompt_actions.lock();
    if let Some(action) = prompt.pop_front() {
        if prompt.is_empty() {
            crate::runtime::host_action_census::note_delivered();
        }
        return Some(action);
    }
    drop(prompt);
    let action = slot.inner.lock().actions.pop_front();
    action
}

pub(super) fn product_presented(id: u64) -> Option<bool> {
    Some(slot(id)?.inner.lock().device.product_presented())
}

pub(super) fn console_frame_may_paint(id: u64, mapping_id: u32) -> Option<bool> {
    Some(
        slot(id)?
            .inner
            .lock()
            .device
            .console_frame_may_paint(mapping_id),
    )
}

pub(super) fn copy_console_frame(
    id: u64,
    mapping_id: u32,
    generation: u32,
    destination: &mut [u8],
    destination_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), ReplacementConsoleFrameError> {
    let Some(slot) = slot(id) else {
        return Err(ReplacementConsoleFrameError::DeviceAbsent);
    };
    let result = slot.inner.lock().device.copy_console_frame(
        mapping_id,
        generation,
        destination,
        destination_stride,
        width,
        height,
    );
    result
}

/// Fill `destination` from the console framebuffer the guest programmed, if
/// that is where the console currently lives.
///
/// MMIO `efi_fb_start` carries the console's guest-physical base and
/// `efi_fb_stride` its row pitch. The geometry is the single mode this device
/// advertises -- [`crate::model::EFI_BOOT_WIDTH`] by
/// [`crate::model::EFI_BOOT_HEIGHT`] -- so a request for any other geometry is
/// not this framebuffer; the stride is the only part of the layout the guest
/// sets, and reading rows at `width * 4` instead shears the image by the
/// padding on every row.
///
/// This is what keeps the boot screen moving. macOS relocates its kernel video
/// console off the BAR1 GOP aperture into system RAM part-way through boot --
/// the guest serial says `console relocated to <gpa>` -- and BAR1 stops
/// changing from that moment. A console that follows only BAR1 therefore
/// freezes on whatever text was last written there and shows it for the rest of
/// the boot.
///
/// `false` means this framebuffer is not the source right now, and every way
/// that happens is ordinary rather than a fault: unprogrammed, a geometry that
/// is not the advertised mode, a stride narrower than one row, or a base that
/// is not guest RAM. For most of early boot the base *is* the BAR1 aperture,
/// which is device memory and reads closed by design. The caller holds BAR1 for
/// exactly this case.
///
/// A closed door costs exactly one refused read: the loop below starts at row
/// zero and returns on the first failure. That matters because each refused
/// read is reported on the failure channel, and this door is shut for most of
/// early boot -- reading the whole frame before giving up would turn one
/// expected decline into one per row, on the console's own repeat cadence.
fn copy_guest_efi_console(
    host: &impl HostMemory,
    start: u64,
    stride: u32,
    destination: &mut [u8],
    destination_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    if start == 0
        || width != crate::model::EFI_BOOT_WIDTH
        || height != crate::model::EFI_BOOT_HEIGHT
    {
        return false;
    }
    let row_len = match width.checked_mul(4) {
        Some(row_len) if destination_stride >= row_len => row_len,
        _ => return false,
    };
    let stride = if stride == 0 { row_len } else { stride };
    if stride < row_len {
        return false;
    }
    let required = match height.checked_sub(1).and_then(|rows| {
        u64::from(rows)
            .checked_mul(u64::from(destination_stride))
            .and_then(|offset| offset.checked_add(u64::from(row_len)))
    }) {
        Some(required) => match usize::try_from(required) {
            Ok(required) => required,
            Err(_) => return false,
        },
        None if height == 0 => 0,
        None => return false,
    };
    if destination.len() < required {
        return false;
    }
    for row in 0..height {
        let source = match u64::from(row)
            .checked_mul(u64::from(stride))
            .and_then(|offset| start.checked_add(offset))
        {
            Some(source) => source,
            None => return false,
        };
        let destination_start = usize::try_from(u64::from(row) * u64::from(destination_stride))
            .expect("the complete destination span was validated");
        let destination_end = destination_start + usize::try_from(row_len).unwrap();
        if host
            .read_gpa(source, &mut destination[destination_start..destination_end])
            .is_err()
        {
            return false;
        }
    }
    true
}

pub(super) fn copy_efi_console(
    id: u64,
    destination: &mut [u8],
    destination_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    let inner = slot.inner.lock();
    let gfx = &inner.device.transport_registers().gfx;
    let (start, stride) = (gfx.efi_fb_start, gfx.efi_fb_stride);
    let Some(ops) = slot.ops else {
        return false;
    };
    let mut scratch = VecDeque::new();
    let host = QemuHost::new(&ops, &mut scratch, &slot.prompt_actions);
    copy_guest_efi_console(
        &host,
        start,
        stride,
        destination,
        destination_stride,
        width,
        height,
    )
}

pub(super) fn cursor_glyph(id: u64) -> Option<(u16, u16, u16, u16, Vec<u32>)> {
    let slot = slot(id)?;
    let inner = slot.inner.lock();
    let glyph = inner.device.cursor().glyph()?;
    Some((
        glyph.width,
        glyph.height,
        glyph.hot_x,
        glyph.hot_y,
        glyph.pixels.to_vec(),
    ))
}

#[cfg(feature = "host-window")]
fn publish_early_window_frame(slot: &BoundReplacementDevice, product_presented: bool) {
    if product_presented {
        return;
    }
    let Some(framebuffer) = *slot.early_fb.lock() else {
        return;
    };
    let row_len = match framebuffer.width.checked_mul(4) {
        Some(row_len) if framebuffer.stride >= row_len => row_len,
        _ => return,
    };
    let source_len = match u64::from(framebuffer.stride)
        .checked_mul(u64::from(framebuffer.height))
        .and_then(|length| usize::try_from(length).ok())
    {
        Some(length) => length,
        None => return,
    };
    let output_len = match u64::from(row_len)
        .checked_mul(u64::from(framebuffer.height))
        .and_then(|length| usize::try_from(length).ok())
    {
        Some(length) => length,
        None => return,
    };
    // SAFETY: the QEMU shim registers a RAMBlock pointer valid for at least
    // stride * height bytes through the device lifetime.
    let source =
        unsafe { std::slice::from_raw_parts(framebuffer.pointer as *const u8, source_len) };
    let mut bgra = vec![0; output_len];
    let source_stride = usize::try_from(framebuffer.stride).unwrap();
    let row_len = usize::try_from(row_len).unwrap();
    for row in 0..usize::try_from(framebuffer.height).unwrap() {
        bgra[row * row_len..(row + 1) * row_len]
            .copy_from_slice(&source[row * source_stride..row * source_stride + row_len]);
    }
    let mut window = slot.window.lock();
    let Some(window) = window.as_mut() else {
        return;
    };
    // Republish only when the firmware framebuffer's bytes actually differ from
    // what the window is already showing.
    //
    // `Frame::seq` is the whole "is this a new picture" test the window loop
    // applies, so a fresh seq over identical bytes buys a full-screen present
    // that puts the same picture back up. This publisher runs once per device
    // poll rather than once per guest update -- about 205 times a second across
    // an x86 boot -- and the guest is drawing into this framebuffer while we
    // read it with no synchronization available on either side. Publishing
    // unconditionally therefore spent a full-screen copy and a present on every
    // one of those polls, and showed the boot console tearing at the poll rate
    // instead of at the rate the console actually changes.
    //
    // The comparison needs no state of its own: the frame slot already retains
    // the last frame published, which is exactly the picture on screen. The
    // slot handle is cloned rather than borrowed so the `seq` bump below can
    // still take `window` mutably.
    let frames = window.frames.clone();
    let Ok(mut current) = frames.lock() else {
        return;
    };
    if let Some(crate::host_window::present::FramePayload::CpuBgra {
        bgra: published,
        width,
        height,
    }) = current.as_deref().map(|frame| &frame.payload)
    {
        if *width == framebuffer.width && *height == framebuffer.height && published[..] == bgra[..]
        {
            return;
        }
    }
    window.seq = window.seq.wrapping_add(1);
    *current = Some(Arc::new(crate::host_window::present::Frame {
        seq: window.seq,
        payload: crate::host_window::present::FramePayload::CpuBgra {
            bgra,
            width: framebuffer.width,
            height: framebuffer.height,
        },
    }));
    window.wake.wake();
}

#[cfg(feature = "host-window")]
pub(super) fn window_start(id: u64, width: u32, height: u32) -> bool {
    use crate::host_window::present::{
        FrameSlot, InputSink, WindowConfig, WindowMode, WindowWaker,
    };
    let Some(slot) = slot(id) else {
        return false;
    };
    let mut window = slot.window.lock();
    if window.is_some() {
        return true;
    }
    let frames: FrameSlot = Arc::new(std::sync::Mutex::new(None));
    let wake = WindowWaker::new();
    let weak = Arc::downgrade(&slot);
    let input: InputSink = Arc::new(move |action| {
        let Some(slot) = weak.upgrade() else {
            return;
        };
        slot.prompt_actions.lock().push_back(action);
        if let Some(ops) = slot.ops {
            if let Some(notify) = ops.notify_actions {
                // SAFETY: QEMU owns the callback context for the bound device.
                unsafe { notify(ops.ctx) };
            }
        }
    });
    let presenter: Arc<dyn crate::runtime::replacement_services::WindowPresentationService> =
        Arc::new(ReplacementWindowService(Arc::downgrade(&slot)));
    let config = WindowConfig {
        title: "Reims vGPU".to_string(),
        width: if width == 0 {
            crate::model::EFI_BOOT_WIDTH
        } else {
            width
        },
        height: if height == 0 {
            crate::model::EFI_BOOT_HEIGHT
        } else {
            height
        },
        mode: WindowMode::requested(),
    };
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(target_os = "macos")]
    let (thread, exited) = {
        let exited = Arc::new(std::sync::atomic::AtomicBool::new(false));
        if let Err(reason) = crate::host_window::present::start_main_thread(
            crate::host_window::present::MainThreadWindowStart {
                id,
                presenter,
                config,
                on_input: input,
                frames: Arc::clone(&frames),
                stop: Arc::clone(&stop),
                exited: Arc::clone(&exited),
                wake: Arc::clone(&wake),
            },
        ) {
            crate::observe::fail(format!("replacement_window_start_refused reason={reason}"));
            return false;
        }
        (None, exited)
    };
    #[cfg(not(target_os = "macos"))]
    let thread = Some(crate::host_window::present::spawn(
        presenter,
        config,
        input,
        Arc::clone(&frames),
        Arc::clone(&stop),
        Arc::clone(&wake),
    ));
    *window = Some(ReplacementWindowLink {
        frames,
        wake,
        stop,
        thread,
        seq: 0,
        #[cfg(target_os = "macos")]
        exited,
    });
    true
}

#[cfg(not(feature = "host-window"))]
pub(super) fn window_start(_id: u64, _width: u32, _height: u32) -> bool {
    false
}

#[cfg(all(feature = "host-window", target_os = "macos"))]
pub(super) fn window_run_main(id: u64) -> bool {
    crate::host_window::present::run_main_thread(id).is_ok()
}

#[cfg(not(all(feature = "host-window", target_os = "macos")))]
pub(super) fn window_run_main(_id: u64) -> bool {
    false
}

#[cfg(feature = "host-window")]
pub(super) fn window_stop(id: u64) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    let Some(mut window) = slot.window.lock().take() else {
        return true;
    };
    window.stop.store(true, Ordering::Release);
    #[cfg(target_os = "macos")]
    {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !window.exited.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if !window.exited.load(Ordering::Acquire) {
            return false;
        }
    }
    match window.thread.take().map(std::thread::JoinHandle::join) {
        Some(Ok(Ok(()))) | None => true,
        Some(Ok(Err(reason))) => {
            crate::observe::fail(format!("replacement_window_run_refused reason={reason}"));
            false
        }
        Some(Err(_)) => false,
    }
}

#[cfg(not(feature = "host-window"))]
pub(super) fn window_stop(_id: u64) -> bool {
    false
}

#[cfg(feature = "host-window")]
pub(super) fn window_set_early_fb(
    id: u64,
    pointer: usize,
    stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(slot) = slot(id) else {
        return false;
    };
    if pointer == 0
        || width == 0
        || height == 0
        || width.checked_mul(4).is_none_or(|row| stride < row)
        || u64::from(stride).checked_mul(u64::from(height)).is_none()
    {
        return false;
    }
    *slot.early_fb.lock() = Some(ReplacementEarlyFramebuffer {
        pointer,
        stride,
        width,
        height,
    });
    true
}

#[cfg(not(feature = "host-window"))]
pub(super) fn window_set_early_fb(
    _id: u64,
    _pointer: usize,
    _stride: u32,
    _width: u32,
    _height: u32,
) -> bool {
    false
}

#[cfg(all(test, feature = "host-window"))]
mod tests {
    #[test]
    fn poll_releases_the_device_before_publishing_an_early_window_frame() {
        let handle = super::create(None, crate::model::PAGE_SHIFT_X86)
            .expect("the replacement device should start on the test Vulkan host");

        assert!(super::poll(handle));
        assert!(super::destroy(handle));
    }
}

#[cfg(test)]
mod console_tests {
    use super::copy_guest_efi_console;
    use crate::model::{EFI_BOOT_HEIGHT, EFI_BOOT_WIDTH};
    use crate::runtime::host::{FakeHost, HostMemory};

    const BASE: u64 = 0x8000_0000;
    const ROW_BYTES: u32 = EFI_BOOT_WIDTH * 4;

    /// A guest that pads its rows is read at the pitch it published.
    ///
    /// The failure this gates is a shear rather than a refusal: reading at
    /// `width * 4` when the guest published a wider stride walks the source
    /// short by the padding on every row, so row N arrives holding bytes from
    /// part-way through row N-1 and the console slides diagonally down the
    /// screen. Marking one byte per row and checking where it lands says which
    /// pitch the copy used.
    #[test]
    fn the_guest_console_is_read_at_the_stride_the_guest_published() {
        let padded = ROW_BYTES + 256;
        let mut host = FakeHost::new();
        let rows = [0u32, 1, 2, 17, EFI_BOOT_HEIGHT - 1];
        for (marker, row) in rows.iter().enumerate() {
            let byte = [marker as u8 + 1];
            host.write_gpa(BASE + u64::from(*row) * u64::from(padded), &byte)
                .unwrap();
        }
        let mut destination = vec![0u8; (ROW_BYTES as usize) * (EFI_BOOT_HEIGHT as usize)];
        assert!(copy_guest_efi_console(
            &host,
            BASE,
            padded,
            &mut destination,
            ROW_BYTES,
            EFI_BOOT_WIDTH,
            EFI_BOOT_HEIGHT,
        ));
        for (marker, row) in rows.iter().enumerate() {
            assert_eq!(
                destination[(*row as usize) * (ROW_BYTES as usize)],
                marker as u8 + 1,
                "row {row} did not come from the published stride"
            );
        }
    }

    /// A zero stride means the guest published none, so rows are contiguous.
    #[test]
    fn an_unpublished_stride_reads_contiguous_rows() {
        let mut host = FakeHost::new();
        host.write_gpa(BASE + u64::from(ROW_BYTES), &[9]).unwrap();
        let mut destination = vec![0u8; (ROW_BYTES as usize) * (EFI_BOOT_HEIGHT as usize)];
        assert!(copy_guest_efi_console(
            &host,
            BASE,
            0,
            &mut destination,
            ROW_BYTES,
            EFI_BOOT_WIDTH,
            EFI_BOOT_HEIGHT,
        ));
        assert_eq!(destination[ROW_BYTES as usize], 9);
    }

    /// The console is the one mode this device advertises. Any other geometry
    /// names a different framebuffer, and the caller has BAR1 for it.
    #[test]
    fn a_geometry_that_is_not_the_advertised_mode_is_refused() {
        let host = FakeHost::new();
        let mut destination = vec![0u8; 1280 * 720 * 4];
        assert!(!copy_guest_efi_console(
            &host,
            BASE,
            0,
            &mut destination,
            1280 * 4,
            1280,
            720,
        ));
    }

    /// A stride narrower than one row cannot describe this framebuffer.
    #[test]
    fn a_stride_narrower_than_a_row_is_refused() {
        let host = FakeHost::new();
        let mut destination = vec![0u8; (ROW_BYTES as usize) * (EFI_BOOT_HEIGHT as usize)];
        assert!(!copy_guest_efi_console(
            &host,
            BASE,
            ROW_BYTES - 4,
            &mut destination,
            ROW_BYTES,
            EFI_BOOT_WIDTH,
            EFI_BOOT_HEIGHT,
        ));
    }

    /// An unprogrammed base is the ordinary early-boot answer, not a fault.
    #[test]
    fn an_unprogrammed_console_base_is_refused() {
        let host = FakeHost::new();
        let mut destination = vec![0u8; (ROW_BYTES as usize) * (EFI_BOOT_HEIGHT as usize)];
        assert!(!copy_guest_efi_console(
            &host,
            0,
            0,
            &mut destination,
            ROW_BYTES,
            EFI_BOOT_WIDTH,
            EFI_BOOT_HEIGHT,
        ));
    }

    /// The base is the BAR1 aperture for most of early boot -- device memory,
    /// which reads closed -- and one refused read is the whole cost.
    #[test]
    fn a_console_base_that_is_not_guest_ram_is_refused() {
        let mut host = FakeHost::new();
        host.mark_non_ram(BASE, u64::from(ROW_BYTES) * u64::from(EFI_BOOT_HEIGHT));
        let mut destination = vec![0u8; (ROW_BYTES as usize) * (EFI_BOOT_HEIGHT as usize)];
        assert!(!copy_guest_efi_console(
            &host,
            BASE,
            0,
            &mut destination,
            ROW_BYTES,
            EFI_BOOT_WIDTH,
            EFI_BOOT_HEIGHT,
        ));
    }
}
