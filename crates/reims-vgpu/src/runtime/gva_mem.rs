//! Read task GPU-virtual addresses via the task page directory.
//!
//! The device's adapter over `reims_vgpu_paging`: the walk, the span cutting
//! and the geometry table live there, and what this module adds is the three
//! things that crate structurally cannot see — the device's [`TaskTable`], its
//! [`HostMemory`] (as the [`HostPhys`] seam), and the mapping of the walk's
//! typed refusals onto [`MemError`] and the failure channel.
//!
//! Geometry always requires an explicit create-time page_shift (12 = x86_64,
//! 14 = arm64e). There is no arm-default overload — callers must choose.

use crate::runtime::host::{HostMemory, MemError};
use reims_vgpu_core::{TaskEntry, TaskTable};
use reims_vgpu_paging::resolve::{
    geometry_for_page_shift, read_task_root, resolve_status_name, translate_root, ResolveStatus,
    Task,
};
use reims_vgpu_paging::span::{visit_span_chunks, walk_span, SpanRefusal};
use reims_vgpu_wire::mem::GuestMemory;

/// Minimal contract required to walk one task address space.
///
/// Both the legacy model and the replacement task owner expose these same
/// decoded fields. Keeping the walker parameterized by that contract avoids a
/// semantic adapter or a second page-table implementation at cutover.
pub(crate) trait TaskAddressSpace {
    fn active(&self) -> bool;
    fn directory_pfn(&self) -> u32;
}

impl TaskAddressSpace for TaskEntry {
    fn active(&self) -> bool {
        self.active
    }

    fn directory_pfn(&self) -> u32 {
        self.directory_pfn
    }
}

/// A span refusal as this device's memory error.
///
/// One spelling, because every rail that reads or writes bytes across a span
/// ends here and each would otherwise decide for itself which refusals mean
/// "the directory did not read", which mean "the walk refused", and which mean
/// "that address does not translate".
///
/// A [`SpanRefusal::Page`] is the last of those: the walk ran and the guest's
/// own table had no mapping, so the status is about the address. A
/// [`SpanRefusal::Setup`] is one of the first two, and which one is the whole
/// content of that arm — `ErrZeroRootPfn` and `ErrZeroDepth` are the walk's
/// answer about a directory it *could* read, and everything else there is a
/// failure to get as far as the directory at all.
pub(crate) fn span_refusal_error(refusal: SpanRefusal) -> MemError {
    match refusal {
        SpanRefusal::Setup(
            status @ (ResolveStatus::ErrZeroRootPfn | ResolveStatus::ErrZeroDepth),
        ) => MemError::Unresolved(status),
        SpanRefusal::Setup(_) => MemError::TaskRootRead,
        SpanRefusal::Page(status) => MemError::Unresolved(status),
    }
}

/// [`HostMemory`]'s guest-physical reads as the wire crate's guest-memory
/// seam. One address space — guest-physical — per that trait's hard rule.
///
/// The one spelling in the crate. There were two, and the second was declared
/// inside a function body in `gva_view`, which is how it stayed invisible: a
/// reader of either site saw a complete four-line adapter and no reason to look
/// for another. They agreed, but nothing made them — and the seam they
/// implement is the one place where "which address space is this" is decided,
/// so a copy that grew a second method or read a different accessor would put
/// two answers in the crate with no diff to catch it.
pub(crate) struct HostPhys<'a, M: HostMemory>(pub &'a M);

impl<M: HostMemory> GuestMemory for HostPhys<'_, M> {
    fn read_at(&self, gpa: u64, dst: &mut [u8]) -> bool {
        self.0.read_gpa(gpa, dst).is_ok()
    }
}

/// Translate `gva` under `task` and copy `buf.len()` bytes into `buf`.
///
/// `page_shift` must be the device create-time guest page shift (12 or 14).
pub(crate) fn read_task_gva<M: HostMemory, T: TaskAddressSpace>(
    host: &M,
    task: &T,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if !task.active() || task.directory_pfn() == 0 {
        return Err(MemError::NoTaskDirectory);
    }
    let geom = geometry_for_page_shift(page_shift).ok_or(MemError::UnsupportedPageShift)?;
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn(),
    };
    // Streams rather than collecting the chunks first: this sits one level
    // below per-row blit loops, and a read that resolves to a single page —
    // which most of them do — would otherwise allocate a one-element Vec per
    // row. The write path cannot do the same, because the walk holds the host
    // shared and the write needs it exclusively.
    let mut result: Result<(), MemError> = Ok(());
    visit_span_chunks(&reader, geom, &gr_task, gva, buf.len(), &mut |chunk| {
        match host.read_gpa(chunk.gpa, &mut buf[chunk.range()]) {
            Ok(()) => true,
            Err(e) => {
                // The host's own error, not a walk status: the address resolved
                // and the transaction is what failed, and which transaction it
                // was is the finding.
                result = Err(e);
                false
            }
        }
    })
    .map_err(span_refusal_error)?;
    result
}

/// Resolve one complete task-GVA reply span before writing any of its bytes.
///
/// This is the synchronous control/query publication path. It deliberately
/// collects the exact translated chunks once, before the first store, so a
/// multi-chunk reply cannot re-walk into a different guest mapping midway
/// through publication. Deferred and repeated data-plane writes retain their
/// stronger mapped-window ownership in `gva_view`.
pub(crate) fn write_task_gva_once<M: HostMemory, T: TaskAddressSpace>(
    host: &mut M,
    task: &T,
    gva: u64,
    buf: &[u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if !task.active() || task.directory_pfn() == 0 {
        return Err(MemError::NoTaskDirectory);
    }
    let geom = geometry_for_page_shift(page_shift).ok_or(MemError::UnsupportedPageShift)?;
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn(),
    };
    let chunks = {
        let reader = HostPhys(&*host);
        reims_vgpu_paging::span::span_chunks(&reader, geom, &gr_task, gva, buf.len())
            .map_err(span_refusal_error)?
    };
    for chunk in chunks {
        host.write_gpa(chunk.gpa, &buf[chunk.range()])?;
    }
    Ok(())
}

/// Read `[gva, gva+len)` under the task the guest named. **That task, or an
/// error.**
///
/// This used to fall back to walking `task_id >> 1`'s page table at the same
/// address, and it was the last of the three `>> 1` arms this crate improvised.
/// The other two were deleted after measuring zero. This one measured **9-11
/// substitutions per boot**, every boot, all from `objects::lookup_list_entry` —
/// and the contract says every one of them was wrong:
///
/// A GVA has no meaning apart from the page table it is resolved against.
/// `lookup_list_entry` builds its address from the **named** task's own
/// `object_list_pfn`, so the same number under a different task's table is a
/// different location that merely happens to be readable. And it always is:
/// tasks put their object lists in low pages, so the neighbour's table has
/// something mapped there on essentially every attempt. The fallback therefore
/// did not fail loudly when it was wrong — it succeeded, and returned the
/// neighbour's object-list entry as if it were this task's.
///
/// The failure mode is now a typed refusal the caller already handles
/// (`lookup_list_entry` returns `None`, which is its "the guest has not told us"
/// answer), carrying **which** of the walk's checks refused.
/// `#[track_caller]` names the site.
#[track_caller]
pub fn read_task_gva_by_id<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    let r = try_read_task_gva_by_id(host, tasks, task_id, gva, buf, page_shift);
    if let Err(named) = r {
        note_read_refusal(task_id, gva, named);
    }
    r
}

/// [`read_task_gva_by_id`] without the refusal line, for a caller whose miss is
/// an **answer** rather than a failure.
///
/// There is exactly one such shape in this device and it is worth naming,
/// because using the loud read for it put 18 lines per boot on the fail channel
/// that meant nothing. `objects::surface_backing_probe_order` walks the live tasks asking
/// "does this one own surface N?", and a task that does not own it has no entry
/// at that slot — so the walk *must* miss on every task before the owner. The
/// miss is how the search works.
///
/// This is not a way to quieten a noisy path. The caller has to be able to say
/// what the miss means, which is why it is a second function rather than a flag:
/// a read whose failure the caller cannot interpret must stay on the loud one.
pub fn try_read_task_gva_by_id<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    let Some(task) = tasks.get(task_id) else {
        return Err(MemError::NoSuchTask);
    };
    read_task_gva(host, task, gva, buf, page_shift)
}

/// Record a refused read, latched per `(reason, task, site)`.
///
/// The reason is the [`MemError`] the walk itself returned, so the line names
/// which of the walk's checks refused rather than a label chosen here.
///
/// The latch is taken before the line is built: `Emit::field` renders eagerly,
/// and this sits one level below per-row blit loops, so building and dropping
/// strings on every refused read would make the probe cost scale with the
/// traffic it is measuring.
#[track_caller]
fn note_read_refusal(task_id: u32, gva: u64, named: MemError) {
    use crate::observe::Decline;
    // Key off the raw location, not its rendering — a refused read can repeat
    // per row, and formatting before the latch would allocate on every one.
    let loc = std::panic::Location::caller();
    if !crate::observe::first_sight(named.slug(), latch_key(task_id, 0, loc)) {
        return;
    }
    let via = via_caller();
    crate::observe::Emit::decline("gva_read_refused", &named)
        .field("task", task_id)
        .field("gva", format!("{gva:#x}"))
        .field("via", via)
        .fail();
}

/// Fixture write at the arm64e page shift, panicking if it does not land.
///
/// The page shift is fixed in the name, per the crate rule that portable code
/// takes `page_shift` and arch-fixed helpers say so. Every unit-test fixture in
/// this crate writes arm64e and treats a failed write as a broken fixture
/// rather than a result, which is why the assertion lives here instead of at
/// each call site.
///
/// # The `#[cfg(test)]` is the enforcement — do not remove it
///
/// "Product code must not call a helper with a page shift baked into its name"
/// is not a rule a reader has to hold, and it is not something to go looking
/// for: this gate and the one on [`define_task_pages_arm64e`] are the only two
/// arch-fixed functions in the crate, and behind them a product call is a
/// `cannot find function` from rustc rather than a finding. `reims_vgpu_paging::geometry`
/// exposes the arch-fixed *constants* ungated, which is fine — a shift is
/// picked from `state.page_shift` at the call site, and a constant cannot
/// silently walk a page table at the wrong stride the way a helper can.
///
/// Ungating either one to share it with an integration test would take the
/// enforcement away and leave nothing, so a caller outside the crate is a
/// reason to move the fixture, not to widen the gate.
#[cfg(test)]
#[track_caller]
pub fn write_task_gva_arm64e<M: HostMemory>(host: &mut M, task: &TaskEntry, gva: u64, buf: &[u8]) {
    assert!(
        write_task_gva(host, task, gva, buf, crate::model::PAGE_SHIFT_ARM64E).is_ok(),
        "fixture write of {} bytes at {gva:#x} failed",
        buf.len()
    );
}

/// Translate `gva` under `task` and write `buf` into guest RAM via `write_gpa`.
///
/// **Tests / fixtures only.** Product paths must use [`write_task_gva_product`]
/// (contig HostOps view). Do not call from product encode/blit/compute.
#[cfg(test)]
pub fn write_task_gva<M: HostMemory>(
    host: &mut M,
    task: &TaskEntry,
    gva: u64,
    buf: &[u8],
    page_shift: u32,
) -> Result<(), MemError> {
    write_task_gva_once(host, task, gva, buf, page_shift)
}

/// `file:line` of whoever called the `#[track_caller]` function above this one.
///
/// Rendered as the repo-relative tail so the field stays short enough to sit on
/// an always-on line: `runtime/blit_exec/mod.rs:1039`. The tail is whatever
/// `Location::file()` gives after `/src/`, so a module that becomes a
/// directory changes what this field reads — as `blit_exec` just did.
#[track_caller]
fn via_caller() -> String {
    let loc = std::panic::Location::caller();
    let file = loc.file();
    let tail = file.rfind("/src/").map_or(file, |i| &file[i + 5..]);
    format!("{tail}:{}", loc.line())
}

/// Dedup key for the guest-memory censuses: two task ids **and** the call site.
///
/// The call site belongs in the identity. Without it the second site to reach a
/// given `(arm, task, other)` is silent for the life of the process, and
/// `first_sight` is per-process rather than per-boot — the hazard that has
/// already caused one census here to be read as a behavioural difference.
///
/// Hashed rather than bit-packed because both ids can carry a raw wire word, so
/// neither has a bound worth relying on. This is a set key for suppressing
/// repeats, not a value anything reads back. Takes the `Location` rather than
/// its rendering so callers on a per-row path can key without allocating.
fn latch_key(task_id: u32, other: u32, loc: &std::panic::Location<'_>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    task_id.hash(&mut h);
    other.hash(&mut h);
    loc.file().hash(&mut h);
    loc.line().hash(&mut h);
    h.finish()
}

/// Whether any page of `[gva, gva+span)` resolves under `task_id`'s tables.
///
/// Separates "there is nowhere to put this" from "putting it there went wrong",
/// which callers that degrade gracefully need and a writer returning one status
/// for both cannot give them. Stops at the first hit, so the common answer costs
/// one translate rather than a walk of the whole span.
pub fn any_task_gva_page_resolves<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> bool {
    let mut found = false;
    visit_task_gva_page_gpas(
        host,
        tasks,
        task_id,
        gva,
        span.max(1),
        page_shift,
        &mut |_| {
            found = true;
            false
        },
    );
    found
}

/// Resolve pages of `[gva, gva + span)` under the task the guest named — the
/// same selection as [`read_task_gva_by_id`] and
/// [`crate::runtime::gva_view::write_span_within`]'s resolver — and call `visit` with
/// each page-aligned GPA. Stops early when `visit` returns `false`.
///
/// This is a lookup, not a validator: pages that fail to translate are
/// skipped silently — the content read that follows fails (and fail-logs) on
/// its own terms. One root read and one descent span the whole range.
///
/// **The named task, or no pages.** This was the last of four sites that fell
/// back to `task_id >> 1` when the named slot had no page table to walk. The
/// other three are gone — `resolve_task_word` decides raw-only
/// (`raw_live.then_some(raw)`), `read_task_gva_by_id` refuses, and
/// `gva_view::resolve_task_for_walk` returns `None` — all on the same contract
/// argument: a GVA has no meaning apart from the page table it is resolved
/// against, and slots run densely from 0, so `task_id >> 1` is almost always
/// some *other* live task whose table happens to have something mapped there.
///
/// Here the substitution was invisible rather than merely wrong, because
/// the page-drift guard that decides whether a resolved span may still be
/// written to guest RAM re-resolves
/// through *this* function with the *same* task id the window was armed under.
/// A window indexed under the neighbour's table was therefore re-indexed under
/// the neighbour's table, the two sets matched, and the guard reported "still
/// ours". It could not see a hazard it reproduced.
///
/// A short walk is what every caller already fails closed on: the guest-run
/// builder and the deferred-Store arm both compare the page count against the
/// span and decline, and the compute rail reports its count as `pages=` on an
/// always-on line.
pub fn visit_task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    visit: &mut dyn FnMut(u64) -> bool,
) {
    visit_task_gva_pages(
        host,
        tasks,
        task_id,
        gva,
        span,
        page_shift,
        &mut |gpa| match gpa {
            Some(gpa) => visit(gpa),
            None => true,
        },
    );
}

/// The resolved page GPAs of `[gva, gva+span)` under `task_id`'s page table, in
/// GVA order, with unresolved pages dropped.
///
/// The ordered form, for callers that walk the result as a window —
/// neighbouring entries differing by exactly one page is what lets a gather
/// coalesce them. Compare `len()` against
/// [`reims_vgpu_paging::span::pages_spanned`] to learn whether
/// anything was dropped.
pub fn task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> Vec<u64> {
    let mut out = Vec::new();
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift, &mut |gpa| {
        out.push(gpa);
        true
    });
    out
}

/// The distinct page GPAs of `[gva, gva+span)` under `task_id`'s page table.
///
/// The set form, for callers that only ask "is this page one of mine?" — the
/// deferred-window page indexes and the blit/Store destination bounds. Order
/// is not preserved and repeats collapse, so `len()` is a lower bound on the
/// pages walked; that is what every caller compares against
/// [`reims_vgpu_paging::span::pages_spanned`].
pub fn task_gva_page_gpa_set<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> std::collections::HashSet<u64> {
    let mut out = std::collections::HashSet::new();
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift, &mut |gpa| {
        out.insert(gpa);
        true
    });
    out
}

/// Shared page-table walk behind [`visit_task_gva_page_gpas`]: one root read
/// and one descent for the whole range, visiting every page in order. Reports
/// an unresolved page as `None` rather than dropping it, which is what a caller
/// recording *which* pages it read needs.
fn visit_task_gva_pages<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    visit: &mut dyn FnMut(Option<u64>) -> bool,
) {
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return;
    };
    let reader = HostPhys(host);
    let Some(task) = tasks.get(task_id) else {
        return;
    };
    if !task.active || task.directory_pfn == 0 {
        return;
    }
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    // Every page of the run, which is the shape the licence check and the
    // guest-run resolvers ask for. One descent is shared across the pages whose
    // upper indices match, instead of `depth` guest reads per page.
    //
    // A setup refusal visits nothing and is dropped rather than reported: this
    // function's contract is that a caller compares what it saw against what it
    // expected, so it is the only one of the span readers for which "no pages"
    // is an answer rather than an error.
    let _ = walk_span(&reader, geom, &gr_task, gva, span, &mut |_, r| {
        visit(r.ok())
    });
}

/// Every page of `[gva, gva+span)` in order, resolved through one root read and
/// one descent, with `None` for a page the table cannot translate.
///
/// [`visit_task_gva_page_gpas`] drops the unresolved pages; a caller checking a
/// cached page list against the live table needs them, because "page 40 does not
/// translate" and "page 40 translates elsewhere" are different findings and only
/// one of them is about the guest. Stride is fixed at one page for the same
/// reason: a check that samples cannot conclude anything about the pages it
/// skipped.
///
/// The visitor stops when `visit` answers `false`, and it visits nothing at all
/// for an inactive task, an absent directory or an unwalkable page geometry — so
/// a caller must compare what it saw against what it expected rather than
/// treating a quiet return as agreement.
pub fn visit_task_gva_pages_in_order<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    visit: &mut dyn FnMut(Option<u64>) -> bool,
) {
    visit_task_gva_pages(host, tasks, task_id, gva, span, page_shift, visit);
}

/// Translate one GVA to a GPA under the task directory (single page).
pub fn translate_task_gva<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    page_shift: u32,
) -> Option<u64> {
    if !task.active || task.directory_pfn == 0 {
        return None;
    }
    let geom = geometry_for_page_shift(page_shift)?;
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    // A one-byte span, so the single chunk's `gpa` is this GVA's own address —
    // page base plus its offset within the page. Going through the span cutter
    // rather than `read_task_root` + `translate_root` by hand is what keeps the
    // zero-root and zero-depth refusals here identical to the ones every other
    // rail gets: written out at this call site they were a fifth copy, and this
    // copy was the one that did not have them, reaching the same answer only
    // because the descent refuses a zero root a second time further down.
    let mut gpa = None;
    visit_span_chunks(&reader, geom, &gr_task, gva, 1, &mut |chunk| {
        gpa = Some(chunk.gpa);
        false
    })
    .ok()?;
    gpa
}

/// One-line walk diagnosis for a single task slot (measure-only; no product gates).
///
/// Example: `tid=2 act=1 dir=0xabc root=0xdef depth=2 st=zero-pfn pte=0 lvl=1 idx=4`
pub fn diagnose_task_slot<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    task_id: u32,
    gva: u64,
    page_shift: u32,
) -> String {
    if !task.active {
        return format!(
            "tid={task_id} act=0 dir={:#x} st=inactive",
            task.directory_pfn
        );
    }
    if task.directory_pfn == 0 {
        return format!("tid={task_id} act=1 dir=0 st=no-directory");
    }
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return format!(
            "tid={task_id} act=1 dir={:#x} st=bad-page-shift({page_shift})",
            task.directory_pfn
        );
    };
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = match read_task_root(&reader, &gr_task, geom) {
        Ok(r) => r,
        Err(st) => {
            return format!(
                "tid={task_id} act=1 dir={:#x} st=root({})",
                task.directory_pfn,
                resolve_status_name(st)
            );
        }
    };
    let t = translate_root(&reader, geom, root.root_pfn, root.depth, gva);
    if t.status == ResolveStatus::Ok {
        format!(
            "tid={task_id} act=1 dir={:#x} root={:#x} depth={} st=ok gpa={:#x} leaf_pfn={:#x}",
            task.directory_pfn, root.root_pfn, root.depth, t.gpa, t.leaf_pfn
        )
    } else {
        format!(
            "tid={task_id} act=1 dir={:#x} root={:#x} depth={} st={} pte={:#x} lvl={} idx={}",
            task.directory_pfn,
            root.root_pfn,
            root.depth,
            resolve_status_name(t.status),
            t.raw_pte,
            t.level,
            t.entry_index
        )
    }
}

/// Diagnose walk under wire `task_id`, `task_id>>1`, and a few active peers.
///
/// Compact multi-clause string for one fail-log line (MapMemory2 / stage Unmapped).
pub fn diagnose_gva_walk<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    page_shift: u32,
) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(6);
    let mut tried = std::collections::BTreeSet::new();
    let try_id = |id: u32, parts: &mut Vec<String>, tried: &mut std::collections::BTreeSet<u32>| {
        if !tried.insert(id) {
            return;
        }
        let Some(task) = tasks.get(id) else {
            // No task under this id at all. `st=undefined` rather than the
            // `st=oob` this printed against the old fixed array: there is no
            // range to be outside of now, and the two say different things —
            // one was "the id is too large", this is "the guest never defined
            // it", which is the only way to reach here.
            parts.push(format!("tid={id} st=undefined"));
            return;
        };
        parts.push(diagnose_task_slot(host, task, id, gva, page_shift));
    };
    try_id(task_id, &mut parts, &mut tried);
    try_id(task_id >> 1, &mut parts, &mut tried);
    // Peer scan: active tasks with a directory (cap 4 extras) — catches wrong-task walks.
    let peer_ids: Vec<u32> = tasks
        .live()
        .filter(|(id, t)| !tried.contains(id) && t.directory_pfn != 0)
        .map(|(id, _)| id)
        .take(4)
        .collect();
    for id in peer_ids {
        try_id(id, &mut parts, &mut tried);
    }
    format!(
        "gva={gva:#x} page_shift={page_shift} | {}",
        parts.join(" || ")
    )
}

/// Snapshot of active task directories (for periodic map census).
pub fn format_active_tasks(tasks: &TaskTable) -> String {
    let mut bits = Vec::new();
    for (i, t) in tasks.live() {
        bits.push(format!(
            "t{i}:dir={:#x},len={:#x},ol_pfn={:#x},ol_n={}",
            t.directory_pfn, t.length, t.object_list_pfn, t.object_list_count
        ));
    }
    if bits.is_empty() {
        "tasks=none".into()
    } else {
        format!("tasks[{}]={}", bits.len(), bits.join(";"))
    }
}
