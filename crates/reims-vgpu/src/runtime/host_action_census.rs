//! Delivery-hop census for prompt host actions.
//!
//! Prompt IRQs are coalesced before QEMU consumes them. This owner measures
//! that queue directly and does not depend on either command scheduler.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Default)]
struct HostActionCensus {
    armed_us: AtomicU64,
    wait_us: AtomicU64,
    waits: AtomicU64,
    wait_max_us: AtomicU64,
    coalesced_gfx: AtomicU64,
    coalesced_iosfc: AtomicU64,
    last_report_ms: AtomicU64,
}

impl HostActionCensus {
    fn arm(&self, now_us: u64) {
        let _ = self
            .armed_us
            .compare_exchange(0, now_us.max(1), Relaxed, Relaxed);
    }

    fn delivered(&self, now_us: u64) {
        let armed = self.armed_us.swap(0, Relaxed);
        if armed == 0 {
            return;
        }
        let wait = now_us.saturating_sub(armed);
        self.wait_us.fetch_add(wait, Relaxed);
        self.waits.fetch_add(1, Relaxed);
        self.wait_max_us.fetch_max(wait, Relaxed);
    }

    fn coalesced(&self, kind: crate::runtime::host::HostActionKind) {
        match kind {
            crate::runtime::host::HostActionKind::IrqGfxPulse => {
                self.coalesced_gfx.fetch_add(1, Relaxed);
            }
            crate::runtime::host::HostActionKind::IrqIosfcPulse => {
                self.coalesced_iosfc.fetch_add(1, Relaxed);
            }
            _ => {}
        }
    }

    fn report(&self, now_ms: u64) -> Option<String> {
        let last = self.last_report_ms.load(Relaxed);
        if now_ms.saturating_sub(last) < 1_000
            || self
                .last_report_ms
                .compare_exchange(last, now_ms, Relaxed, Relaxed)
                .is_err()
        {
            return None;
        }
        Some(format!(
            "replacement_host_action_census irq_wait_us={} irq_waits={} irq_wait_max_us={} irq_coalesced_gfx={} irq_coalesced_iosfc={}",
            self.wait_us.swap(0, Relaxed),
            self.waits.swap(0, Relaxed),
            self.wait_max_us.swap(0, Relaxed),
            self.coalesced_gfx.swap(0, Relaxed),
            self.coalesced_iosfc.swap(0, Relaxed),
        ))
    }
}

static CENSUS: std::sync::LazyLock<HostActionCensus> =
    std::sync::LazyLock::new(HostActionCensus::default);

pub(crate) fn note_armed() {
    CENSUS.arm(crate::observe::elapsed_us());
}

pub(crate) fn note_delivered() {
    CENSUS.delivered(crate::observe::elapsed_us());
}

pub(crate) fn note_coalesced(kind: crate::runtime::host::HostActionKind) {
    CENSUS.coalesced(kind);
}

pub(crate) fn report() {
    if let Some(line) = CENSUS.report(crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivery_report_banks_the_oldest_arm_and_each_irq_kind() {
        let census = HostActionCensus::default();
        census.arm(10);
        census.arm(20);
        census.delivered(35);
        census.coalesced(crate::runtime::host::HostActionKind::IrqGfxPulse);
        census.coalesced(crate::runtime::host::HostActionKind::IrqIosfcPulse);
        let report = census.report(1_000).unwrap();
        assert!(report.contains("irq_wait_us=25"));
        assert!(report.contains("irq_waits=1"));
        assert!(report.contains("irq_wait_max_us=25"));
        assert!(report.contains("irq_coalesced_gfx=1"));
        assert!(report.contains("irq_coalesced_iosfc=1"));
    }
}
