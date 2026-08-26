//! Unbounded counters for replacement decode and transport contract routes.

use std::{collections::BTreeMap, sync::atomic::AtomicU64};

static COUNTS: std::sync::LazyLock<parking_lot::Mutex<BTreeMap<&'static str, u64>>> =
    std::sync::LazyLock::new(|| parking_lot::Mutex::new(BTreeMap::new()));
static LAST_REPORT_MS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note(route: &'static str) {
    let mut counts = COUNTS.lock();
    *counts.entry(route).or_default() += 1;
}

pub(crate) fn report() {
    use std::sync::atomic::Ordering::Relaxed;
    let now = crate::observe::elapsed_ms() as u64;
    let last = LAST_REPORT_MS.load(Relaxed);
    if now.saturating_sub(last) < 1_000
        || LAST_REPORT_MS
            .compare_exchange(last, now, Relaxed, Relaxed)
            .is_err()
    {
        return;
    }
    let counts = std::mem::take(&mut *COUNTS.lock());
    if counts.is_empty() {
        return;
    }
    let fields = counts
        .into_iter()
        .map(|(route, count)| format!("{route}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    crate::observe::off(format!("replacement_contract_census {fields}"));
}
