//! Self-contained Vulkan execution backend.
//!
//! All host GPU work lives under `backend/vulkan/`, driven by `ash`. Product draw encode uses the
//! internal [`engine`] (persistent ash context + content-keyed caches). This
//! crate has no external graphics-executor dependency; AIR translation comes
//! from the pinned public `metal2vulkan` crate.
//!
//! Product submission and guest-lifetime reset enter through the device-owned
//! [`crate::runtime::executor::Executor`] implementation.
//!
//! [`caps`] classifies the bound host GPU into the four-cell support matrix
//! (unified/discrete memory × has/has-no import) that every path here must keep
//! working. [`policy`] owns the independent unified and discrete placement and
//! scheduling choices; host-pointer import remains an orthogonal capability.
//!
//! [`translate`] is the matching seam for *state*: decoded Metal formats and
//! pipeline enums become Vulkan ones there and nowhere else, so the same
//! decision cannot be made twice with two different answers.

pub mod caps;
pub mod engine;
pub mod translate;
pub use reims_vgpu_vulkan::policy;

struct DeviceTelemetry;

impl reims_vgpu_vulkan::telemetry::BackendTelemetry for DeviceTelemetry {
    fn route(&self, name: &'static str) {
        crate::runtime::drain::note_store_route(name);
    }

    fn route_n(&self, name: &'static str, count: u64) {
        crate::runtime::drain::note_store_route_n(name, count);
    }

    fn route_us(&self, name: &'static str, micros: u64) {
        crate::runtime::drain::note_store_route_us(name, micros);
    }

    fn readback_phase(&self, phase: reims_vgpu_vulkan::telemetry::ReadbackPhase, micros: u64) {
        use crate::runtime::drain::ReadbackPhase as DevicePhase;
        use reims_vgpu_vulkan::telemetry::ReadbackPhase as BackendPhase;
        let phase = match phase {
            BackendPhase::Submit => DevicePhase::Submit,
            BackendPhase::Fence => DevicePhase::Fence,
            BackendPhase::Map => DevicePhase::Map,
            BackendPhase::Write => DevicePhase::Write,
            BackendPhase::Vouch => DevicePhase::Vouch,
            BackendPhase::Resolve => DevicePhase::Resolve,
        };
        crate::runtime::drain::note_readback_phase(phase, micros);
    }

    fn readback_gpu_us(&self, barrier: u64, copy: u64) {
        crate::runtime::drain::note_readback_gpu_us(barrier, copy);
    }

    fn guest_imports_invalidated(&self) {
        crate::runtime::guest_ram_map::reset();
    }
}

static DEVICE_TELEMETRY: DeviceTelemetry = DeviceTelemetry;

pub(crate) fn install_telemetry() {
    reims_vgpu_vulkan::telemetry::install(&DEVICE_TELEMETRY);
}
