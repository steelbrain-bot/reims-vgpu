//! The process's imports of guest RAM: built once from the shim's spans, and
//! the only place a guest physical address becomes a bindable reference.
//!
//! # What this replaces
//!
//! The dma-buf rail had a cache here, and it had to: `UDMABUF_CREATE_LIST`
//! walked every page, took a kernel reference on each, and cost enough that a
//! digest-bucketed LRU bounded by pinned bytes was worth its own module.
//!
//! Under the host-pointer model there is nothing left to cache.
//! [`crate::runtime::guest_ram::GuestRamImport::slice`] is a range check.
//! What *is* worth holding is the small thing the cache was built around: the
//! imports themselves, made once at first use and held for the VM's lifetime.
//! This module is that, and it is a sorted `Vec` of a dozen or so entries rather
//! than a cache with an eviction policy.
//!
//! A RAMBlock is imported in **chunks** when it is longer than the backend's
//! queried single-allocation limit. A window resolves against whichever import
//! backs its GPA, and one straddling two of them groups into two `VkBuffer`
//! sources, because a RAMBlock boundary could already split one.
//!
//! # Why the imports are built here and not at device create
//!
//! The backend measures the granularity; the runtime holds the
//! [`GuestRamProvider`](crate::runtime::host::GuestRamProvider) that can say where
//! guest RAM lives.
//! Neither side has both, and the device context deliberately does not take a
//! host — see the module doc on [`crate::qemu::host_ops`] for why the runtime
//! keeps it. So the granularity is published by the backend through
//! [`crate::runtime::guest_ram::latch_import_limits`] — together with the
//! largest import that backend's heaps could hold — and the spans are fetched
//! here, on the first guest-memory reference of a boot.
//!
//! Building lazily rather than eagerly also gets the ordering right for free:
//! the device exists before any guest command is decoded, so the granularity is
//! always published by the time the first reference is asked for.
//!
//! # What a refusal means
//!
//! Every refusal here puts the whole boot on the copying rails for the
//! addresses it covers, so none of them is a slow path and none may be silent.
//! The one *expected* refusal is [`MapRefusal::NoBackendImport`](crate::runtime::guest_ram_map::MapRefusal::NoBackendImport): a host without
//! the extension, or an operator who set
//! [`crate::env::GUEST_IMPORT`](crate::env::GUEST_IMPORT) off. That one is a
//! statement about the host rather than a loss, so it is reported once on the
//! off channel rather than as a failure per reference.

use crate::runtime::guest_ram::{
    granularity, import_budget, import_span_max, GuestRamError, GuestRamImport, GuestRamRegion,
    GuestRef,
};
use crate::runtime::host::{GuestRamProvider, GuestRamRegionsError};
use std::sync::Arc;

/// Why a guest physical address did not become a bindable reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRefusal {
    /// No backend published an import granularity: this host cannot import
    /// guest RAM, or an operator asked it not to. Expected, and the state every
    /// copying rail exists for.
    NoBackendImport,
    /// The shim could not say where guest RAM lives. Carries the check that
    /// refused.
    HostRefused(GuestRamRegionsError),
    /// The shim answered, but no span survived being bounded to the granularity
    /// — every region was empty, unmapped, malformed, or shorter than one
    /// granule. Distinct from [`Self::HostRefused`] because the host answered
    /// fine and it is our own bound that rejected every span.
    NoUsableRegion { spans: usize },
    /// The spans are importable and this guest is larger than the roomiest heap
    /// on the host's GPU, so nothing may import any of them.
    ///
    /// # Why the whole map refuses rather than the block that does not fit
    ///
    /// An import is a `VkDeviceMemory` charged to one heap, and a submission
    /// that names it makes the driver keep all of it resident. On a part whose
    /// heaps are a fraction of the guest — an APU with a few gigabytes of
    /// carve-out against a `-m 16G` guest — the kernel refuses to validate the
    /// allocation and the submission fails, which arrives as a **lost device**
    /// and a dead guest rather than as a slow rail. That has been reported from
    /// the field on `radv`/`amdgpu` (`Not enough memory for command submission`,
    /// then a lost context), and the reporter's own fix was to set
    /// [`crate::env::GUEST_IMPORT`] off by hand. This makes the device reach the
    /// same state without being told to.
    ///
    /// Refusing the whole map rather than the oversized block is what makes it
    /// safe: the copying rails are selected by a page having no [`GuestRef`], and
    /// a *partial* import would leave the writeback paths holding references
    /// into one RAMBlock and none into another, which is a hard error at those
    /// sites and not a fallback. All or nothing keeps the boot on the one arm
    /// that is tested end to end.
    ///
    /// The comparison is against the sum, because every import is live at once
    /// for the VM's lifetime and a submission may name any of them.
    ///
    /// # Its relationship to the per-import check, which is the exact one
    ///
    /// The backend publishes this budget as the roomiest heap an import can be
    /// *charged to* — the same population of memory types
    /// `reims-vgpu-vulkan`'s memory selector will choose from, since every
    /// import goes through it carrying one class's required flags. So a sum that
    /// passes here has a heap that each individual chunk fits, and the exact
    /// per-allocation check at the pick — which refuses rather than making a
    /// call Vulkan declares invalid — agrees with this one by construction
    /// rather than by coincidence. Publishing the maximum over *every* heap
    /// instead, which this once did, breaks that: a part whose device-local heap
    /// is twice its host-visible heap passes here with room to spare and then
    /// refuses at every pick, which is the partial import this refusal exists to
    /// prevent.
    ///
    /// This is a heap-*capacity* test and not a residency one, so it is a lower
    /// bound: a host that passes it can still be too full to import. It catches
    /// the direction that has been seen to kill a guest.
    ///
    /// # What would give such a host the fast rail back
    ///
    /// Not this refusal, which governs the optional whole-VM import. Resource-
    /// sized stable aliases are admitted independently by
    /// [`crate::runtime::guest_ram::host_allocation_import_align`], so a guest
    /// allocation that fits can still take the direct rail without making
    /// unrelated RAM resident.
    ImportExceedsHeap { needed: u64, budget: u64 },
    /// The address is not inside any imported span. Guest RAM the GPU can reach
    /// exists, and this address is not in it — a device MMIO address, a hole,
    /// or a page the guest named that this machine does not back.
    GpaNotInAnyImport { gpa: u64 },
    /// The address is in a span, and the length asked for leaves it. Carries the
    /// bound's own reason so the check that refused keeps its name.
    OutsideImport(GuestRamError),
    /// A page list that is not one GPA-contiguous stretch.
    ///
    /// Not a statement that the pages are un-importable — they are all inside
    /// one RAMBlock and each is nameable. It is a statement about the *bind*: a
    /// `VkBuffer` range and a Metal buffer offset are each one offset and one
    /// length, so a surface assembled from four stretches is four of them, and
    /// no consumer takes several yet. Named and counted because how often it
    /// fires is what says whether widening them is worth doing.
    ///
    /// `runs` is what says *how much* widening would cost, and it is the number
    /// to read before building it. "Scattered" is one word for both a window in
    /// two stretches — where a second bind is obviously worth it — and a window
    /// in five hundred, where each run is a couple of pages and the region list
    /// starts to rival the copy it replaces. `pages` alone cannot tell those
    /// apart, and a count of *refusals* tells them apart even less: both read as
    /// one line here. Sampled at the point of refusal, so it bands the reach
    /// actually requested rather than the reach some other rail asked for.
    ///
    /// # What it measured, on an x86 guest
    ///
    /// A driven boot — Safari window drag, 25 s, PCI attach — put **every**
    /// window at almost exactly four pages per run, across three orders of
    /// magnitude of size: 2025 pages in 507 runs for each 1920x1080 writeback,
    /// 813/204, 630/158, 588/147, 256/65, 128/32, 45/12. The ratio holds
    /// because it is not a ratio: the runs are 16 KiB each, and the guest backs
    /// a surface in 16 KiB physically-contiguous granules that are unrelated to
    /// each other. Four 4 KiB x86 pages is what one of those granules looks
    /// like from this side.
    ///
    /// Two consequences worth carrying, because both contradict the obvious
    /// guess. Scattering is **not** a fragmentation artifact that a longer
    /// uptime or a quieter guest would improve — it is the allocator's
    /// granularity, so it is the steady state. And the run count scales with
    /// the surface, so the widening this field exists to price is ~500 ranges
    /// for a full-screen flush and not the handful the word "scattered"
    /// suggests.
    Scattered {
        pages: usize,
        runs: usize,
        first: u64,
    },
}

impl crate::observe::Decline for MapRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoBackendImport => "guest_ram_map_no_backend_import",
            Self::HostRefused(_) => "guest_ram_map_host_refused",
            Self::NoUsableRegion { .. } => "guest_ram_map_no_usable_region",
            Self::ImportExceedsHeap { .. } => "guest_ram_map_import_exceeds_heap",
            Self::GpaNotInAnyImport { .. } => "guest_ram_map_gpa_not_in_any_import",
            Self::Scattered { .. } => "guest_ram_map_scattered",
            // The inner reason is the diagnosis; this wrapper only says where
            // it happened, so it forwards rather than adding a slug of its own.
            Self::OutsideImport(inner) => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoBackendImport => Vec::new(),
            Self::HostRefused(inner) => {
                let mut f = vec![("host_reason", inner.slug().to_string())];
                f.extend(crate::observe::Decline::fields(inner));
                f
            }
            Self::NoUsableRegion { spans } => vec![("spans", spans.to_string())],
            Self::ImportExceedsHeap { needed, budget } => vec![
                ("needed_mb", (needed >> 20).to_string()),
                ("budget_mb", (budget >> 20).to_string()),
            ],
            Self::GpaNotInAnyImport { gpa } => vec![("gpa", format!("{gpa:#x}"))],
            Self::Scattered { pages, runs, first } => vec![
                ("pages", pages.to_string()),
                ("runs", runs.to_string()),
                ("first", format!("{first:#x}")),
            ],
            Self::OutsideImport(inner) => inner.fields(),
        }
    }
}

crate::observe::decline_display!(MapRefusal);

/// The greppable event class for this module's refusals.
const EVENT: &str = "guest_ram_map";

/// The imports this process holds, or the refusal that stopped it building any.
///
/// Resolved once and then read. A `Mutex` rather than a `OnceLock` because a
/// device recreate must be able to drop the imports: the backend's handles die
/// with the device, and an import whose identity outlived them would let a
/// stale [`crate::runtime::guest_ram::GuestSlice`] resolve against a
/// `VkDeviceMemory` that no longer exists.
static MAP: std::sync::Mutex<Option<Resolved>> = std::sync::Mutex::new(None);

#[derive(Debug)]
struct Resolved {
    /// One per usable RAMBlock span, in the order the shim reported them.
    /// Ordinary machines have one or two.
    imports: Vec<Arc<GuestRamImport>>,
    /// Set when the resolution refused, so the next reference does not re-ask
    /// the shim for an answer that will not change. A refusal here is about the
    /// host and the granularity, both of which are fixed for the device's life.
    refusal: Option<MapRefusal>,
}

impl Resolved {
    /// Turn one guest physical address and length into a bindable reference.
    ///
    /// The single implementation, so the one-span and the whole-window entry
    /// points cannot disagree about which import owns a GPA or which refusal a
    /// miss earns. Takes no lock of its own — the caller is already inside
    /// [`with_map`], which is what lets a scattered window resolve every run
    /// under one acquisition.
    ///
    /// [`Self::refusal`] is **not** re-checked here: an entry point asks it once
    /// before walking, and asking again per run would emit the same standing
    /// refusal N times for one window.
    fn reference(&self, gpa: u64, len: u64) -> Result<GuestRef, MapRefusal> {
        // Binary search, not a linear scan. The imports are sorted by
        // `gpa_base`, so the last one whose base is at or below `gpa` is the
        // only one that can contain it — `partition_point` names that index and
        // `contains_gpa` still decides, so an address in a hole between two
        // imports is refused exactly as before.
        //
        // A scan was right while a machine had one or two imports. Chunking a
        // RAMBlock at the span ceiling makes it eight to a dozen on an ordinary
        // guest, and this runs once per run of a scattered window — 9 to 32 runs
        // per bind, thousands of binds a second — so the growth would land in
        // the hot path. This makes the count stop mattering instead of trading
        // one host's correctness for another's throughput.
        let import = self
            .imports
            .partition_point(|i| i.gpa_base().is_some_and(|base| base <= gpa))
            .checked_sub(1)
            .map(|last| &self.imports[last])
            .filter(|i| i.contains_gpa(gpa))
            .ok_or(MapRefusal::GpaNotInAnyImport { gpa })
            .map_err(report_once)?;
        // `slice_for_gpa` emits its own named refusal on the fail channel, so
        // the wrapper forwards the reason rather than adding a second line.
        let slice = import
            .slice_for_gpa(gpa, len)
            .map_err(MapRefusal::OutsideImport)?;
        GuestRef::new(Arc::clone(import), slice).map_err(MapRefusal::OutsideImport)
    }

    /// The exclusive end GPA of the import backing `gpa`, or `None` if nothing
    /// backs it.
    ///
    /// Exists so a caller can split a contiguous guest stretch at the seam
    /// between two imports instead of being refused at it — see
    /// [`references_for_runs`]. Deliberately not a public entry point: an import
    /// boundary is this module's own bookkeeping, and the only thing outside it
    /// may do with one is stop at it.
    fn import_end(&self, gpa: u64) -> Option<u64> {
        self.imports
            .partition_point(|i| i.gpa_base().is_some_and(|base| base <= gpa))
            .checked_sub(1)
            .map(|last| &self.imports[last])
            .filter(|i| i.contains_gpa(gpa))
            .and_then(|i| i.gpa_base().map(|base| base + i.len()))
    }
}

/// Forget every import.
///
/// Called when the backend tears its device down. The next reference rebuilds,
/// against fresh identities, so nothing made before the teardown resolves after
/// it — see [`crate::runtime::guest_ram::ImportId`] for why that matters.
pub fn reset() {
    *MAP.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// How many RAMBlock spans the shim reported, and how many bytes they cover.
///
/// The denominator for the backend's *imported* count. A backend imports a span
/// at its first reference and not before, so "one imported" means one of these
/// has been touched — which on a two-span machine is a workload fact, not a
/// defect. Reporting the count alone cannot tell those apart, which is why the
/// census line carries both.
///
/// Counts rather than clones: this runs once a census window, and cloning an
/// `Arc` per span to take a length would be a refcount touch per span per
/// second for a number that does not change after the first reference.
///
/// `(0, 0)` before the first reference of a boot and on a host that cannot
/// import, which is the same reading the census suppresses.
pub fn span_census() -> (usize, u64) {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| (r.imports.len(), r.imports.iter().map(|i| i.len()).sum()))
        .unwrap_or((0, 0))
}

/// Every import this process holds, for a backend that needs to create or
/// release its device-side handles.
///
/// Empty before the first reference of a boot and on a host that cannot import.
pub fn imports() -> Vec<Arc<GuestRamImport>> {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| r.imports.clone())
        .unwrap_or_default()
}

/// Take the whole guest-RAM import now, so the guest's first draw does not pay
/// for it.
///
/// # Two steps, and only the second one costs
///
/// Asking the host where its RAMBlocks are ([`resolve`]) is a handful of shim
/// calls. Handing each of those mappings to the GPU is `vkAllocateMemory` with
/// a host pointer chained, which is where a driver that pins takes a reference
/// on every page of guest RAM — seconds, proportional to the RAM the VM was
/// given, and measured per block by
/// [`reims_vgpu_vulkan::engine::warm_guest_ram_imports`].
///
/// Both were lazy and both landed on the guest's first `gather`, inside its
/// first draw, inside a display transaction the guest abandons after 1000 ms.
/// Moving only the first one bought nothing measurable, which is the finding
/// that located the second: `guest_ram_span` moved a second earlier and
/// `gather_us` did not move at all. Called from the guest driver's
/// protocol-version handshake, both now run before the guest has a display pipe
/// to arm a watchdog on, and every later caller finds the answer already there.
///
/// **It must never cache a negative.** [`resolve`] answers `NoBackendImport`
/// when no backend has published a granularity yet, and that answer is latched
/// in `MAP` for the rest of the boot — so warming before the backend is up
/// would turn a capable host into one that refuses every window, which is the
/// opposite of the intent and would look like a host that lacks the extension.
/// The guard is the same question `resolve` asks first, and asking it here
/// leaves the lazy path to handle a backend that is genuinely late.
///
/// Resolve on the first call of a boot, then run `body` against the result.
///
/// The one place the resolution is built, so no entry point can hold a second
/// copy of "have we asked the host yet".
fn with_map<H: GuestRamProvider + ?Sized, R>(host: &mut H, body: impl FnOnce(&Resolved) -> R) -> R {
    let mut guard = MAP.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(resolve(host));
    }
    body(guard.as_ref().expect("just resolved"))
}

/// The refusal the whole rail is standing on, if there is one.
///
/// An entry point that judges the *shape* of what it was given must ask this
/// first. The order is not cosmetic: on a host with no import every window
/// refuses, and one told it was too fragmented sends a reader hunting for a
/// contiguity problem that a contiguous window would not have fixed either —
/// which is exactly what a driven `REIMS_VGPU_GUEST_IMPORT=off` boot logged
/// before [`reference_for_pages`] asked.
///
/// Public because it is also the cheap early-out: a rail whose next step is an
/// `O(pages)` walk should ask this before paying for one it is going to throw
/// away. That caller must ask *this* rather than re-reading
/// [`crate::runtime::guest_ram::granularity`], which is the same answer for one
/// of the four refusals and silence for the other three.
pub fn standing_refusal<H: GuestRamProvider + ?Sized>(host: &mut H) -> Option<MapRefusal> {
    with_map(host, |resolved| resolved.refusal)
}

/// Turn a guest physical address and a length into a bindable reference.
///
/// The whole guest-memory rail goes through here. Building the imports on the
/// first call is why `host` is taken: after that it is not touched.
pub fn reference<H: GuestRamProvider + ?Sized>(
    host: &mut H,
    gpa: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    with_map(host, |resolved| {
        if let Some(refusal) = resolved.refusal {
            return Err(report_once(refusal));
        }
        resolved.reference(gpa, len)
    })
}

/// One stretch of a scattered window: where it starts in the window, and the
/// import reference covering it.
///
/// `window_offset` is a byte offset from the first byte the caller asked for,
/// not from the start of a page and not from the start of the import. It is
/// what a copy's source offset is measured in, which is the only thing a
/// consumer needs and the only thing it may not compute for itself.
pub use reims_vgpu_memory::GuestWindowRun;

/// [`reference_for_pages`] for a window that is *not* one contiguous stretch:
/// one reference per maximal GPA run, in window order.
///
/// # Why this exists as well as [`reference_for_pages`]
///
/// A driven x86 boot measured every guest surface at four 4 KiB pages per run —
/// the guest backs a surface in 16 KiB physically-contiguous granules — so a
/// 1920x1080 window is 2025 pages in 507 runs and *always* will be. See
/// [`MapRefusal::Scattered`] for the distribution. A rail that only takes one
/// contiguous stretch therefore never runs on a real workload, which is what
/// the boot found: the import was `supported` and bound 8 KiB in 25 seconds.
///
/// # What the caller owes
///
/// Every returned run is a separate bind. A consumer that issues one GPU copy
/// per run is correct; one that concatenates them is not, because nothing
/// relates two runs' import offsets. The runs tile the window exactly — no
/// gaps, no overlaps, ascending — and the tests below assert that rather than
/// leaving it to be re-derived.
///
/// Runs are **not** bounded here. A bound belongs where the cost is, which is
/// the consumer's region array, and a cap in this function would silently hand
/// back a partial window — the failure mode that loses guest work quietly. A
/// consumer that cannot issue N copies must refuse by name on the count it got.
///
/// # One lock for the whole window
///
/// The resolution is behind a mutex, and this runs on the draw-time buffer rail
/// at ~16 000 windows a second of ~16 runs each. Resolving each run through
/// [`reference`] would take and drop that mutex a quarter of a million times a
/// second for an answer that cannot change inside one call, so the walk happens
/// inside a single [`with_map`] instead. [`reference`] keeps its own lock for
/// the callers that resolve exactly one span.
pub fn references_for_runs<H: GuestRamProvider + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<Vec<GuestWindowRun>, MapRefusal> {
    if gpas.is_empty() || page_size == 0 || len == 0 {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            runs: 0,
            first: gpas.first().copied().unwrap_or(0),
        }));
    }
    // Absolute byte range this window occupies, measured from the first byte of
    // `gpas[0]` — the same frame `in_page` is stated in. Page indices and run
    // boundaries are both in this frame, so no step below re-derives it.
    let window_start = in_page;
    let window_end = in_page.checked_add(len).ok_or(MapRefusal::Scattered {
        pages: gpas.len(),
        runs: 0,
        first: gpas[0],
    })?;

    with_map(host, |resolved| {
        if let Some(refusal) = resolved.refusal {
            return Err(report_once(refusal));
        }
        let mut out = Vec::new();
        for run in reims_vgpu_paging::runs::contig_page_runs(gpas, page_size) {
            let run_start = (run.start as u64) * page_size;
            let run_end = (run.end as u64) * page_size;
            // Clip to the window: the first run usually starts before it (the
            // window begins `in_page` bytes in) and the last usually ends after
            // it.
            let start = run_start.max(window_start);
            let end = run_end.min(window_end);
            if start >= end {
                continue;
            }
            // GPA-contiguous is not import-contiguous. A RAMBlock is imported in
            // chunks, so a stretch the guest laid out as one run can cross a
            // seam between two of them — and a `GuestRef` is an offset into one
            // import, so it cannot describe both sides.
            //
            // Split at the seam rather than refuse at it. The consumers already
            // take a list and a RAMBlock boundary has always been able to
            // produce one, so two runs here are indistinguishable from two the
            // guest's own page plan produced. Refusing instead would drop the
            // *whole window* to the copying rail — a named, safe decline, but
            // one that fires on roughly one writeback in 250 for no reason the
            // guest could see, which is a chunk size leaking into throughput.
            //
            // Within a run the GPAs are contiguous by construction, so one add
            // reaches any byte of it.
            let mut piece = start;
            while piece < end {
                let gpa = gpas[run.start] + (piece - run_start);
                // `None` means nothing backs this address at all: hand the whole
                // remainder to `reference` so it names that refusal rather than
                // this loop inventing a second one.
                let piece_end = match resolved.import_end(gpa) {
                    Some(import_end) if import_end > gpa => end.min(piece + (import_end - gpa)),
                    _ => end,
                };
                out.push(GuestWindowRun {
                    window_offset: piece - window_start,
                    guest: resolved.reference(gpa, piece_end - piece)?,
                });
                piece = piece_end;
            }
        }
        if out.is_empty() {
            return Err(report_once(MapRefusal::Scattered {
                pages: gpas.len(),
                runs: 0,
                first: gpas[0],
            }));
        }
        Ok(out)
    })
}

/// [`reference`] for a decoded page list: `len` bytes starting `in_page` bytes
/// into `gpas[0]`.
///
/// The one implementation of the contiguity rule, so the sampled, buffer and
/// writeback rails cannot disagree about what a bindable page list is.
pub fn reference_for_pages<H: GuestRamProvider + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    if let Some(refusal) = standing_refusal(host) {
        return Err(report_once(refusal));
    }
    let Some(&first) = gpas.first() else {
        return Err(report_once(MapRefusal::Scattered {
            pages: 0,
            runs: 0,
            first: 0,
        }));
    };
    let contiguous = gpas
        .iter()
        .enumerate()
        .all(|(i, gpa)| *gpa == first + (i as u64) * page_size);
    if !contiguous {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            runs: reims_vgpu_paging::runs::contig_run_count(gpas, page_size),
            first,
        }));
    }
    reference(host, first + in_page, len)
}

/// Ask the host where guest RAM lives and bound every span to the backend's
/// granularity.
///
/// # This used to run on the guest's first draw, and it is NOT the two seconds
///
/// It was lazy — the first `reference_for_pages` triggered it — and it now runs
/// at the guest driver's protocol handshake instead ([`warm`]). **That move was
/// measured and it did not shift the stall**, which is what located the real
/// one: asking the host where its RAM is costs nothing, and handing those
/// mappings to the GPU costs everything. The evidence is one timestamp —
/// `guest_ram_span`, emitted once per boot by this function, moved from t=56453
/// to t=55342 while `gather_us` on the first frame stayed at 2 180 583 over the
/// same six gathers.
///
/// The seconds are `vkAllocateMemory` with the host pointer chained, measured
/// per RAMBlock at [`reims_vgpu_vulkan::engine::warm_guest_ram_imports`],
/// which is now also warmed from [`warm`]. The table below is the state before
/// that, kept because its second row is what ruled out a per-byte cost and sent
/// the search to one-time setup:
///
/// ```text
///                 draw_stall     stage_us     gather_us  gather_n  gather_b
/// macos-11 first   2 028 844    2 022 252     2 022 259         6   1 176 768
/// macos-11 later           —            —            75        61  13 545 376
/// macos-13 first   1 959 875    1 951 567     1 951 562         4     523 904
/// macos-13 later           —            —           105       104  15 318 048
/// ```
///
/// Six gathers of 1.1 MB taking two seconds and sixty-one gathers of 13 MB
/// taking 75 µs is not a gather cost; it is one-time setup charged to whoever
/// arrives first. The same boots report it as a `sync_exec_lock_hold` of
/// ~2 000 000 µs over one to three draws.
///
/// **The guest has a one-second watchdog behind this.** Its display pipe waits
/// on a submitted display transaction and gives up after 1000 ms, so a first
/// frame that takes two seconds blows it on every boot of every rail. Both
/// rails measured here do blow it; the macos-13 guest recovers and the macos-11
/// guest does not, and on macos-11 that is the whole visible failure — the
/// transaction stays pending, WindowServer stops answering, and the session
/// never starts.
///
/// The same driven boot then timed the two halves of the import separately and
/// read `probe_us=0` beside `alloc_us=2 493 029` for a 15 032 385 536-byte
/// RAMBlock and `alloc_us=309 796` for a 2 146 435 072-byte one — the whole
/// stall, in the one call the first gather was the first to reach. That is what
/// [`warm`] now takes at the handshake.
///
/// Timings above are wall clock on a shared host and are upper bounds; the
/// counts and byte totals are not.
/// Split one RAMBlock into consecutive regions of at most `span_max` bytes.
///
/// The regions tile the block exactly and in ascending order: no byte is
/// dropped, none is covered twice, and the last one is whatever remains. A block
/// already inside the ceiling comes back as itself, so a host that needs no
/// chunking pays one comparison and allocates the same one-element shape it
/// always had.
///
/// Alignment is deliberately *not* applied here. `GuestRamImport::new` trims
/// each region to the device's granularity and names its own refusal when a
/// region cannot survive that — doing it twice would be two spellings of one
/// rule, and this one has no granularity in hand. The consequence is that
/// `span_max` must itself be a multiple of the granularity, which is why the
/// backend masks it before publishing and why a ceiling below the granularity is
/// refused outright rather than clamped.
fn chunk_span(span: GuestRamRegion, span_max: u64) -> Vec<GuestRamRegion> {
    if span_max == 0 || span.len <= span_max {
        return vec![span];
    }
    let mut out = Vec::with_capacity((span.len / span_max) as usize + 1);
    let mut done = 0u64;
    while done < span.len {
        let len = span_max.min(span.len - done);
        out.push(GuestRamRegion {
            host_va: span.host_va + done,
            gpa_base: span.gpa_base + done,
            len,
        });
        done += len;
    }
    out
}

fn resolve<H: GuestRamProvider + ?Sized>(host: &mut H) -> Resolved {
    let Some(align) = granularity() else {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::NoBackendImport),
        };
    };
    let spans = match host.guest_ram_regions() {
        Ok(spans) => spans,
        Err(why) => {
            return Resolved {
                imports: Vec::new(),
                refusal: Some(MapRefusal::HostRefused(why)),
            }
        }
    };
    let count = spans.len();
    // A span this device cannot bound is skipped rather than fatal: a machine
    // with one ordinary RAMBlock and one odd sliver should import the RAMBlock.
    // `GuestRamImport::new` names the check that rejected each skipped one on
    // the fail channel, so a partial import is never silent.
    //
    // Each block is imported in chunks no larger than the API-derived span the
    // backend published. Nothing else has to change: a window already resolves
    // against whichever import backs its GPA, and one straddling two of them
    // already groups into two `VkBuffer` sources, because a RAMBlock boundary
    // has always been able to split one.
    let span_max = import_span_max().unwrap_or(u64::MAX);
    let mut imports: Vec<Arc<GuestRamImport>> = spans
        .into_iter()
        .flat_map(|span| chunk_span(span, span_max))
        .filter_map(|span| GuestRamImport::new(span, align).ok().map(Arc::new))
        .collect();
    // `reference` binary-searches this, so the order is load-bearing rather than
    // cosmetic. The shim reports blocks in ascending GPA and `chunk_span` keeps
    // that, so this is a no-op on every machine seen so far — it is here because
    // the search would silently answer `GpaNotInAnyImport` for a live address if
    // a future shim ever reported them out of order.
    imports.sort_by_key(|i| i.gpa_base());
    // Every import is live for the VM's lifetime and any submission may name any
    // of them, so what has to fit is the sum and not the largest block. A guest
    // that does not fit takes the copying rails whole rather than in part — see
    // [`MapRefusal::ImportExceedsHeap`] for why a partial import is the one
    // outcome that is worse than either.
    let needed: u64 = imports.iter().map(|i| i.len()).sum();
    let over_budget = import_budget().filter(|budget| needed > *budget);
    if let Some(budget) = over_budget {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::ImportExceedsHeap { needed, budget }),
        };
    }
    let refusal = imports
        .is_empty()
        .then_some(MapRefusal::NoUsableRegion { spans: count });
    // Once per boot, because this is what makes `guest_import_levels`'s
    // denominator interpretable. That line reports `imported/reported` and a
    // reader seeing `1/4` cannot tell which three went untouched, or whether the
    // untouched ones are guest RAM at all — on q35 the reported set is the two
    // halves of `-m` either side of the PCI hole plus whatever smaller writable
    // RAM regions the board exposes. Naming each span's base and length answers
    // that from the log instead of from a comment that would go stale when a
    // board changes.
    //
    // `resolve` runs once per boot (and again only after a device teardown), so
    // this is a handful of lines, not a cadence.
    for (n, import) in imports.iter().enumerate() {
        crate::observe::off(format!(
            "guest_ram_span n={n}/{count} gpa={:#x} len={} mib={}",
            import.gpa_base().expect("RAMBlock imports have a GPA base"),
            import.len(),
            import.len() / (1024 * 1024),
        ));
    }
    Resolved { imports, refusal }
}

/// Emit `refusal` and hand it back.
///
/// Deduped by slug: these are per-reference and a decode path that names an
/// unbacked address once will name it every frame.
/// [`MapRefusal::NoBackendImport`] goes to the off channel — it is the host
/// saying what it is, not a loss of guest work — and everything else to the
/// fail channel.
fn report_once(refusal: MapRefusal) -> MapRefusal {
    let line = crate::observe::Emit::decline(EVENT, &refusal);
    match refusal {
        MapRefusal::NoBackendImport => {
            if crate::observe::first_sight("guest_ram_map_no_backend_import", 0) {
                line.off();
            }
        }
        _ => line.fail_once(0),
    }
    refusal
}
