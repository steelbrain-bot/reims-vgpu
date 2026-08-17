//! Content identity for sampled guest-resource windows.
//!
//! A retained sampled image may be reused only while neither writer of its
//! source resource has changed the bytes:
//!
//! - the guest writer is named by the decoded resource-validity statement;
//! - device writes are named by the page-exact HostWrites record.
//!
//! The guest statement is carried in its owning namespace: mapping-backed
//! windows use MappingEntry::content_generation, while task-local resources
//! use BufferWriteStamp. An unchanged generation and a quiet device-write
//! verdict preserve the sampled identity. A changed generation, a device write,
//! or an unaddressed statement spends the identity and forces a gather.
//! Hypervisor dirty-page observations are not an input: the decoded resource
//! table already states the answer per resource.
//!
//! The content fold is a diagnostic audit, never an input to the shipping
//! decision. AuditDensity::Strided samples it; EveryBind is the driven
//! soundness arm. A disagreement is fail-visible and spends the identity that
//! was just disproved.
//!
//! Entries are unbounded while their owners are live. Task entries retire with
//! the task and mapping entries with the mapping; device reset clears both.

use std::collections::HashMap;

use crate::contract::fnv;

/// Which zero-copy sampled producer built the window.
///
/// The 2x2 below says whether the witness is sound; this says whose gathers it
/// would be sound *for*. The aggregate reading that opened this — 360 gathers and
/// 842.4 MB a second — is the sum over all three rails and has never been split,
/// so which of them to fix is not yet known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherRail {
    /// Linear guest texture addressed through task GVA.
    Linear,
    /// Type-11 mapping-backed sampled bind.
    Type11,
    /// Type-5 serialized IOSurface plane view (the video path).
    Type5,
}

impl GatherRail {
    /// Census names for the rail's gather count and its gathered kilobytes.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Linear => ("gw_rail_linear", "gw_rail_linear_kb"),
            Self::Type11 => ("gw_rail_t11", "gw_rail_t11_kb"),
            Self::Type5 => ("gw_rail_t5", "gw_rail_t5_kb"),
        }
    }
}

/// Which sampled window a witness entry describes.
///
/// The two shapes are the two ways the producers name a window: a task-GVA span
/// (the linear texture rail, which has no mapping) and a mapping-relative offset
/// (the type-11 and type-5 rails). Those two rails can name the same
/// `(mid, base_off)` for a single-plane surface, and that is harmless — same
/// mapping, same offset and same span is the same bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GatherKey {
    /// A texture window addressed through a task's GVA space.
    TaskGva {
        task_id: u32,
        resource_ref: u32,
        gva: u64,
    },
    /// A window at a byte offset into a mapping's page list.
    Mapping { mid: u32, base_off: u64 },
}

impl GatherKey {
    /// A 64-bit name for this window in the device-wide sampled-identity
    /// keyspace.
    ///
    /// Collisions across the two shapes, or with any other producer's keys, are
    /// harmless and do not need to be designed out: the engine matches on
    /// `(key, generation)` and generations come from one device-global counter
    /// that issues each value once and never again. The key only has to be
    /// *stable* for one window, so that a window's own binds find each other.
    pub fn content_key(self) -> u64 {
        // FNV-1a over the discriminant and fields. A hash rather than a packing
        // because both shapes carry more than 64 bits. The discriminant is
        // folded first so the two shapes cannot alias each other.
        let mut h = fnv::FNV_OFFSET_BASIS;
        let mut eat = |v: u64| h = fnv::fold_u64(h, v);
        match self {
            Self::TaskGva {
                task_id,
                resource_ref,
                gva,
            } => {
                eat(1);
                eat(task_id as u64);
                eat(resource_ref as u64);
                eat(gva);
            }
            Self::Mapping { mid, base_off } => {
                eat(2);
                eat(mid as u64);
                eat(base_off);
            }
        }
        h
    }

    /// Whitespace-free rendering for the always-on log, which is parsed by
    /// splitting on spaces.
    fn log_token(self) -> String {
        match self {
            Self::TaskGva {
                task_id,
                resource_ref,
                gva,
            } => format!("gva:{task_id}:{resource_ref}:{gva:#x}"),
            Self::Mapping { mid, base_off } => format!("map:{mid}:{base_off:#x}"),
        }
    }
}

/// Guest-declared content generation in the identity space that owns it.
///
/// Mapping-backed resources carry the mapping generation. Task-local GVA
/// resources carry the generation of the resource object whose dirty bit the
/// submission consumed. The variants cannot compare across namespaces, which
/// makes replacing one resource shape with the other a write rather than a
/// coincidental equal integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatedGeneration {
    Mapping(u32),
    TaskResource(crate::runtime::buffer_write_gen::BufferWriteStamp),
}

/// What the last bind of one window observed.
#[derive(Clone, Debug)]
struct Entry {
    /// The exact page set the gather read, in window order. A change here means
    /// the window was re-pointed and there is nothing to compare against.
    gpas: Vec<u64>,
    /// Byte length of the window (a geometry change is also a re-point).
    span: u64,
    /// Content fold from the last audit of this window.
    fold: u128,
    /// Whether `fold` still describes the window's bytes.
    ///
    /// True from the audit that recorded it for as long as every bind since was
    /// [`GatherVerdict::Vouched`] — which is the claim the audit exists to check,
    /// so comparing across that run is exactly the right comparison and a longer
    /// run is a stronger one. A bind the witness refused may have changed the
    /// bytes with nothing reading them, and clears it.
    fold_valid: bool,
    /// Whether this window has ever been folded, latched on the first audit.
    ///
    /// Separate from [`Self::fold_valid`], which answers whether the stored
    /// fold is still a *baseline*. Together they separate the two ways an audit
    /// can find nothing to compare against — never folded, or folded and then
    /// invalidated — which read identically without this and are
    /// [`ContentAudit::Seeded`] and [`ContentAudit::Restarted`] with it.
    fold_seeded: bool,
    /// Binds of this window since its last audit, against [`AUDIT_STRIDE`].
    ///
    /// Per window rather than device-wide: a global stride would audit whichever
    /// window happened to land on the multiple and could starve a busy one
    /// indefinitely, where the alarm's whole job is bounded latency per window.
    binds_since_fold: u32,
    /// A baseline is held and the audit is waiting for a vouched bind to check
    /// it against.
    ///
    /// The arm is what makes the comparison reachable. Without it the audit both
    /// took its baseline and tried to compare on the same stride bind, so a
    /// comparison needed [`AUDIT_STRIDE`] consecutive vouched binds — a run this
    /// workload does not produce, which is why `gw_audit_ok` read 0 on three
    /// consecutive boots and `gw_audit_unsound`'s zero meant nothing.
    audit_armed: bool,
    /// Refused binds this arm has re-baselined through, against
    /// [`AUDIT_REBASELINE_LIMIT`].
    rebaselines: u8,
    /// `HostWrites::epoch` at the previous bind, against which the page-exact
    /// question "did this device write any of *these pages* since" is asked.
    pages_epoch: u64,
    /// `MappingEntry::content_generation` at the previous bind, against which
    /// the guest's own account of its CPU writes is asked — see
    /// [`StatedGuestWrite`].
    ///
    /// `None` when the channel could not address this window at that bind, which
    /// is not the same as a generation of 0: a mapping genuinely sitting at
    /// generation 0 has been addressed and has been written zero times, and
    /// comparing it against a later 0 is a real quiet answer.
    stated_gen: Option<StatedGeneration>,
    /// Sampled-content generation currently vouched for these bytes.
    ///
    /// Held across binds for as long as both halves of the witness say the bytes
    /// cannot have changed, and replaced the moment either says otherwise. The
    /// engine's sampled cache binds a retained image on `(key, generation)` with
    /// no compare at all, so a generation that outlives its content by one bind
    /// is a wrong picture that then persists.
    generation: u64,
}

/// Per-device witness state: one entry per sampled window seen.
#[derive(Debug)]
pub struct GatherWitness {
    entries: HashMap<GatherKey, Entry>,
    /// How often this device's content audit is allowed to compare.
    ///
    /// On the witness rather than in a process-wide `OnceLock` so a test can
    /// state the arm it is testing. [`Default`] reads the environment, which is
    /// what makes every construction site — the one in `DeviceState` and the
    /// ones in this module's tests — pick the switch up without naming it.
    audit: AuditDensity,
}

impl Default for GatherWitness {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            audit: AuditDensity::from_env(),
        }
    }
}

/// How often the content audit compares a window against its own past.
///
/// The audit is a standing alarm on the one rule this whole module exists to
/// uphold, and its density is the difference between believing that rule and
/// having measured it — so the density is a stated policy rather than a
/// constant read at the decision site.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AuditDensity {
    /// One comparison per [`AUDIT_STRIDE`] binds of a window. The shipping arm:
    /// the fold is a read of the window, which is the rail this cache removes.
    #[default]
    Strided,
    /// Every bind this device vouches for is compared against the one before
    /// it. A soundness sweep, never a timing — see [`crate::env::GATHER_AUDIT_ALL`].
    EveryBind,
}

impl AuditDensity {
    fn from_env() -> Self {
        match crate::env::switch(crate::env::GATHER_AUDIT_ALL) {
            crate::env::Switch::On => Self::EveryBind,
            _ => Self::default(),
        }
    }

    /// Binds of one window between baselines.
    fn stride(self) -> u32 {
        match self {
            Self::Strided => AUDIT_STRIDE,
            Self::EveryBind => 1,
        }
    }

    /// Whether a completed comparison leaves the window armed.
    ///
    /// The fold a comparison just took describes the window as of that bind, so
    /// it is already the baseline the next bind would be judged against. Staying
    /// armed is what makes [`Self::EveryBind`] mean every bind rather than every
    /// third — arm, compare, disarm is three binds per comparison, and a stride
    /// of 1 alone would still judge only a third of the population.
    fn stays_armed(self) -> bool {
        matches!(self, Self::EveryBind)
    }
}

/// Binds of one window between content audits.
///
/// The fold no longer decides a skip, so its only remaining job is to catch the
/// witness going unsound — and that is a systematic fault rather than a one-off.
/// Both holes found while building this witness repeated tens to hundreds of
/// times per boot, so an audit that sees one bind in `AUDIT_STRIDE` still sees
/// them within seconds.
///
/// The value is the two bounds meeting. A window re-presented at frame rate
/// binds about sixty times a second, so sixty-four bounds the alarm at roughly a
/// second of stale pixels; and one bind in sixty-four is 1.6% of the gathered
/// bytes, about 13 MB/s against the 842 MB/s rail this cache was built to
/// remove. Both the latency and the cost degrade smoothly, so neither edge is
/// fitted to an observation.
pub const AUDIT_STRIDE: u32 = 64;

/// How many consecutive refused binds an armed window re-baselines through
/// before the audit gives up and waits for the stride again.
///
/// An armed window folds on every bind until it meets the vouched bind it is
/// waiting for, so without a bound a window the witness always refuses would
/// fold on all of them — which is the whole 842 MB/s rail this cache exists to
/// remove, arriving through the audit. Eight bounds that at eight folds per
/// stride window in the worst case, against the one the common case costs.
///
/// Eight rather than a larger number because a window that has been refused
/// eight times running is not one a vouch is being claimed about, and the
/// comparison is only interesting where a vouch actually happens.
pub const AUDIT_REBASELINE_LIMIT: u8 = 8;

impl GatherWitness {
    /// Drop every sampled window at device reset.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Retire every sampled window owned by a task lifetime.
    pub fn retire_task(&mut self, task_id: u32) {
        self.retire_where(
            |key| matches!(key, GatherKey::TaskGva { task_id: t, .. } if *t == task_id),
        )
    }

    /// Retire every sampled window owned by a mapping lifetime.
    pub fn retire_mapping(&mut self, mapping_id: u32) {
        self.retire_where(|key| matches!(key, GatherKey::Mapping { mid, .. } if *mid == mapping_id))
    }

    fn retire_where(&mut self, mut doomed: impl FnMut(&GatherKey) -> bool) {
        self.entries.retain(|key, _| !doomed(key));
    }

    /// The host-write epoch recorded at the previous bind of `key`, if any.
    fn previous_pages_epoch(&self, key: &GatherKey) -> Option<u64> {
        self.entries.get(key).map(|entry| entry.pages_epoch)
    }
}

/// Fold `span` bytes of a gathered window into a 128-bit value.
///
/// Word-wise rather than byte-wise, and two accumulators mixed differently so the
/// result is position-sensitive: a fold that only summed words would call any
/// permutation of a window unchanged, and a scrolled tile atlas is exactly a
/// permutation of itself.
///
/// # Safety
/// Every run's `host_ptr` must be a live mapping of at least `len` bytes — the
/// same precondition the gather itself relies on, read at the same point in the
/// draw.
pub(crate) unsafe fn fold_runs(
    runs: &[crate::backend::vulkan::engine::GuestRun],
    span: u64,
) -> u128 {
    let mut a: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut b: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let mut remaining = span;
    for run in runs {
        if remaining == 0 {
            break;
        }
        let n = run.len.min(remaining) as usize;
        remaining -= n as u64;
        // SAFETY: caller's precondition — `host_ptr` is a stable RAMBlock alias
        // valid for at least `run.len` bytes, and `n <= run.len`.
        let bytes = unsafe { std::slice::from_raw_parts(run.host_ptr as *const u8, n) };
        let (words, tail) = bytes.split_at(n & !7);
        for chunk in words.chunks_exact(8) {
            let w = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            a = (a ^ w).rotate_left(29).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            b = b.rotate_left(7).wrapping_add(w ^ a);
        }
        for (i, &byte) in tail.iter().enumerate() {
            a ^= (byte as u64) << (8 * i);
        }
        // Fold the run boundary in so two windows with the same bytes split into
        // different runs are still distinguishable.
        b = b.wrapping_mul(0xff51_afd7_ed55_8ccd) ^ (n as u64);
    }
    ((a as u128) << 64) | b as u128
}

/// Every account of one bind's writers that is read out of device state, taken
/// together before the witness is touched.
///
/// Gathered up front because each needs something the witness cannot reach from
/// inside itself: the page-exact question needs the epoch recorded at the
/// previous bind, and both it and the guest's stated generation are read through
/// the same device state the witness lives in. Passing them in keeps [`observe`]
/// a function of its inputs, which is what lets a test state the writers it is
/// testing.
///
/// One field per writer, and they are not interchangeable —
/// [`Self::pages_wrote`] is this device and [`Self::stated_gen`] is the guest.
///
/// Two coarser counts used to be asked here beside it — the device-global host
/// write sequence and a per-mapping share of it — scoring the two candidate
/// invalidation rules that lost. The global rule invalidates a texture because
/// an unrelated scanout was composited; the per-mapping one read fifteen stale
/// binds a minute, because guest pages are reachable under more than one mapping
/// id. Neither is a rule this device could use, so neither is a count it still
/// takes.
#[derive(Clone, Copy, Debug)]
struct WitnessReadings {
    /// `HostWrites::epoch()` now, to be recorded for the next bind to ask against.
    pages_epoch: u64,
    /// Whether this device wrote any of this window's pages since the previous
    /// bind, and on what grounds. `None` when there is no previous bind to ask
    /// about.
    ///
    /// Carried as the verdict rather than a `bool` because three of its four
    /// non-quiet values are this device declining to rule the write out rather
    /// than a write that landed here, and the three want different repairs.
    pages_wrote: Option<crate::runtime::host_writes::HostWriteVerdict>,
    /// The guest's own account in the identity space named by the key, now.
    ///
    /// `None` when the guest's statements are not addressed to this window at
    /// all — see [`StatedGuestWrite::Unaddressed`]. Compared against the reading
    /// the previous bind left in the entry.
    stated_gen: Option<StatedGeneration>,
    /// Whether a guest-page write this device has **submitted but the GPU has
    /// not yet executed** could land in this window.
    ///
    /// This does **not** feed the vouch, and the reason is the whole of why it
    /// exists. The gather this cache elides is a GPU copy on the same queue as
    /// the writeback, so it is ordered behind it and a retained image cannot
    /// contain pre-copy bytes. [`fold_runs`] is not: it is a **CPU** read of the
    /// same guest pages, and `render_writeback`'s rule for those is that a
    /// host-side reader must settle first or it reads the pre-Store bytes. The
    /// audit was added to this call path after `draw::vulkan`'s zero-copy rail
    /// had already recorded "no settle here — this rail does not read anything",
    /// which stopped being true when the fold arrived.
    ///
    /// So a fold taken while a copy is in flight over the window reads pre-copy
    /// bytes, the next one reads post-copy bytes, both binds are legitimately
    /// vouched, and the audit reports `gw_audit_unsound` for an image that was
    /// never stale. This field is what lets the audit decline to compare across
    /// that, so its remaining alarms are about the cache rather than about
    /// itself.
    pending: PendingWrites,
}

/// Whether a guest-page write this device has submitted but the GPU has not yet
/// executed could land in the window being judged.
///
/// The three arms are the engine's `GuestWriteReach`, restated here so this
/// module's signature does not name a backend type — the Metal arm has no such
/// queue and answers [`Self::Disjoint`] by construction. Only `Disjoint` may
/// vouch, but the other two are kept apart because they want different repairs:
/// an `Overlap` is this device really writing the window it samples, and an
/// `Unnamed` is a footprint nobody could name.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PendingWrites {
    /// Nothing outstanding, or the outstanding footprint provably misses these
    /// pages.
    #[default]
    Disjoint,
    /// An outstanding write names one of these pages.
    Overlap,
    /// Something is outstanding and its pages could not be ruled out.
    Unnamed,
}

impl PendingWrites {
    /// What this device has submitted over `gpas` and the GPU has not run yet.
    ///
    /// One relaxed atomic load in the common case — the same gate
    /// `settle_guest_writes_unless_disjoint` opens with — so a bind with nothing
    /// outstanding pays what it paid before.
    fn over(gpas: &[u64]) -> Self {
        #[cfg(feature = "backend-vulkan")]
        {
            use crate::backend::vulkan::engine::GuestWriteReach as Reach;
            if !crate::backend::vulkan::engine::guest_writes_outstanding() {
                return Self::Disjoint;
            }
            match crate::backend::vulkan::engine::guest_writes_reaching(gpas) {
                Reach::Disjoint => Self::Disjoint,
                Reach::Overlap => Self::Overlap,
                Reach::Unnamed => Self::Unnamed,
            }
        }
        #[cfg(not(feature = "backend-vulkan"))]
        {
            let _ = gpas;
            Self::Disjoint
        }
    }

    /// Census route, so the new refusals are bandable rather than a silent drop
    /// in the vouch rate.
    fn route(self) -> &'static str {
        match self {
            Self::Disjoint => "gw_pending_disjoint",
            Self::Overlap => "gw_pending_overlap",
            Self::Unnamed => "gw_pending_unnamed",
        }
    }

    /// Whether a vouch is still available. Only a proof of disjointness buys
    /// one; both other answers are this device declining to rule the write out.
    fn settled(self) -> bool {
        matches!(self, Self::Disjoint)
    }
}

/// The resolved window one gather will read.
///
/// The pages and the host spans over them are both needed and neither implies
/// the other: guest-write tracking registers a page set, and the content fold
/// reads through the coalesced host pointers.
pub struct GatherWindow<'a> {
    /// Page-aligned guest addresses the window covers, in window order.
    pub gpas: &'a [u64],
    /// Coalesced host spans the gather reads, covering `span` bytes in order.
    pub runs: &'a [crate::backend::vulkan::engine::GuestRun],
    /// Byte length of the window.
    pub span: u64,
    /// Guest page size the `gpas` are expressed in.
    pub page_size: usize,
}

/// What the two halves of the witness said about one bind of a window.
///
/// Returned rather than only counted so a test can drive the witness against a
/// host whose writes it controls, and so the census emission is one place instead
/// of five.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherVerdict {
    /// First sight of the window, or its page set / span moved. Nothing to
    /// compare against; the entry now holds this bind's answers.
    Rearmed,
    /// No guest-declared generation addresses this window. Fail closed: nothing
    /// is vouched for.
    Unarmed,
    /// Both halves quiet — no guest store into the pages, and no write by this
    /// device either. The gather is skippable and the entry keeps its generation.
    Vouched,
    /// At least one half saw a write. The generation is spent and the bytes are
    /// read.
    ///
    /// Both flags can be set at once, and are counted apart because they name
    /// different work: a guest store is the guest repainting, while a write by
    /// this device is our own writeback landing in pages a sampler also reads.
    Refused {
        /// The guest declared a store into this resource.
        guest_wrote: bool,
        /// This device wrote at least one of these pages.
        host_wrote_pages: bool,
    },
}

/// What the guest's invalidation statement says about one window.
///
/// The guest states CPU writes itself: byte `+4` of
/// each `EXEC_INDIRECT2` resource-table record is a test-and-clear of the
/// resource's dirty bit, so the statement is addressed to an **object id**,
/// delivered in the **same submission as the bind** that would consume a stale
/// copy, and sent exactly once. [`crate::runtime::resource_validity::apply`]
/// already lands it on [`crate::model::MappingEntry::content_generation`].
///
/// [`GatherKey::Mapping`] names the very mapping that generation belongs to, so
/// the witness asks the guest what it did instead of reconstructing that answer
/// from hypervisor dirty-page observations.
///
/// Fail-closed by construction: a window the channel cannot address reads
/// [`Self::Unaddressed`] rather than quiet, so an absent statement can never be
/// mistaken for a statement of silence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StatedGuestWrite {
    /// The channel has no answer for this window. Either the key is a
    /// task-local resource with no statement, or the mapping the key names is
    /// not one this device holds. Not evidence in either direction.
    Unaddressed,
    /// The mapping's `content_generation` is where the previous bind left it, so
    /// the guest has stated no CPU write to this resource in between.
    Quiet,
    /// The generation moved: the guest stated at least one CPU write to this
    /// resource since the previous bind of this window.
    Wrote,
}

/// What the content fold said, on the binds where it ran.
///
/// The fold is the audit of [`GatherVerdict::Vouched`], not an input to it —
/// see [`AUDIT_STRIDE`] for why it runs on one bind in sixty-four rather than
/// all of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContentAudit {
    /// Not due this bind: no byte of the window was read.
    Skipped,
    /// Folded for the first time on this window. There was nothing to compare
    /// against; this bind only records one.
    Seeded,
    /// The stride came due and there *was* a fold, but a refused bind since it
    /// was taken invalidated it, so this bind could only record a new one.
    ///
    /// Split out from [`Self::Seeded`] because the two say opposite things
    /// about whether the alarm is running. A seed is a window being met for
    /// the first time and is expected. This is the audit **declining to
    /// compare**, and where it dominates, [`Self::Disagreed`] reading zero says
    /// nothing at all — which is exactly how it read while a writer that
    /// escaped both halves went unnoticed.
    ///
    /// It used to be common by construction rather than by accident: the
    /// baseline was dropped by any single refusal and a comparison was only
    /// attempted once [`AUDIT_STRIDE`] binds had passed, so reaching one needed
    /// sixty-four *consecutive* vouched binds of a window. At the refusal rates a
    /// driven boot measures — 4 669 refusals against 7 347 vouches — that is a
    /// run this workload never produces, and `gw_audit_ok` read 0 on three
    /// consecutive boots while a real escaping writer went unnoticed.
    ///
    /// It now means the armed window was refused [`AUDIT_REBASELINE_LIMIT`]
    /// times running without ever reaching a vouched bind, so there was nothing
    /// to check. That is a real "declined to compare" rather than a structural
    /// one, and it should be rare.
    Restarted,
    /// Armed, and this bind was refused — so the bytes were free to move and the
    /// baseline is taken again from the bytes the gather is about to read.
    ///
    /// The window stays armed. This is the arm *waiting* for the vouched bind it
    /// exists to check, and it costs one fold of a window whose bytes are being
    /// read regardless.
    Rebaselined,
    /// Folded under a vouch, and the bytes are where the vouch said they were.
    Agreed,
    /// Folded under a vouch, and the bytes had moved. Some writer reaches these
    /// guest pages without either half of the witness seeing it, so every gather
    /// skipped since the last audit bound a stale image.
    Disagreed,
    /// Not folded: a guest-page copy this device submitted is still in flight
    /// over this window, so a CPU read of it now is neither the bytes before nor
    /// reliably the bytes after.
    ///
    /// The audit's own limitation and not a finding about the cache. The gather
    /// this cache elides is a GPU copy ordered behind that writeback on the same
    /// queue; the fold is not ordered against it by anything, which is the rule
    /// `render_writeback` states for every host-side reader of guest bytes.
    /// Comparing across it reported the device's own queue as a stale image.
    Indebted,
}

/// One bind's answers: what the witness decided, what the audit found, and the
/// generation the window is left naming.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GatherObservation {
    /// The decision, from the two witness halves alone.
    pub verdict: GatherVerdict,
    /// The check on it, on the binds where the fold ran.
    pub audit: ContentAudit,
    /// The generation this window names *after* the bind — the one the entry
    /// carried in, where it survived, and `fresh_generation` where it did not.
    ///
    /// Returned rather than looked back up so the identity has no absent case to
    /// spell. Reading it back out of the map produced an `Option` that was
    /// `Some` on every path through [`observe`], which is how
    /// `sampled_gather_unvouched` came to be a counter that could not fire.
    pub generation: u64,
    /// Whether [`Self::generation`] is the one the entry carried in or one spent
    /// this bind. Decided beside the assignment that spends it, never
    /// re-derived from [`Self::verdict`] — a `Disagreed` audit vouches and still
    /// spends, so the two do not agree.
    pub vouch: GatherVouch,
    /// What the guest's resource statement said about this bind.
    pub stated: StatedGuestWrite,
}

/// What this witness reports when its soundness audit finds stale bytes.
#[derive(Clone, Copy, Debug)]
pub enum GatherWitnessFault {
    /// Both halves vouched for a window and the content audit found its bytes
    /// moved. Names the window so the writer can be hunted, and the bind count
    /// so the number of stale frames served is bounded rather than guessed.
    VouchedBytesMoved {
        key: GatherKey,
        span: u64,
        binds: u32,
    },
}

impl crate::observe::decline::Decline for GatherWitnessFault {
    fn slug(&self) -> &'static str {
        match self {
            Self::VouchedBytesMoved { .. } => "gather_witness_vouched_bytes_moved",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::VouchedBytesMoved { key, span, binds } => vec![
                ("window", key.log_token()),
                ("span", span.to_string()),
                ("binds", binds.to_string()),
            ],
        }
    }
}

/// One bind's answer to the engine: what to bind on, and whether it is worth
/// anything.
///
/// Both halves are always present. The type exists so they travel together —
/// carrying the identity alone is what let the engine ask "is there an identity"
/// and read the answer as "did the witness vouch".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherOutcome {
    /// What the engine looks the retained image up under, and retains under.
    pub identity: GatheredIdentity,
    /// Whether that identity can name an image the cache already holds.
    pub vouch: GatherVouch,
}

/// Record one zero-copy sampled gather against the guest-write witness, and
/// report it to the census.
///
/// Called from the producers with the window already resolved and with the
/// decoded resource generation captured in that resource's identity space.
///
/// # Why this does not return an `Option`
///
/// It used to, and the `Option` was `Some` on every path: the identity was read
/// back with `vouched_identity`, which answered "is this window tracked", and
/// [`observe`] leaves an entry for every key it is given — the re-point branch
/// inserts one and returns, and the surviving branch holds a `&mut` to one. The engine spent
/// a boot counting `identity.is_some()` as the witness's verdict and read the
/// resulting zero as "the witness never refused a gather". It cannot refuse
/// through this return value at all; [`GatherVouch`] is where the verdict lives.
#[must_use = "the identity is what lets the engine skip the gather; dropping it \
              silently keeps the copy"]
pub fn note_gather(
    state: &mut crate::model::DeviceState,
    rail: GatherRail,
    key: GatherKey,
    stated_gen: StatedGeneration,
    window: GatherWindow<'_>,
) -> GatherOutcome {
    use crate::runtime::drain::{note_store_route, note_store_route_n};

    let span = window.span;
    let (rail_count, rail_kb) = rail.names();
    note_store_route(rail_count);
    note_store_route_n(rail_kb, span / 1024);

    // Both writers' accounts, taken before the witness is touched: the
    // page-exact question needs the epoch recorded at the *previous* bind, which
    // is inside the witness, and the ring that answers it is read through the
    // same device state.
    let counts = WitnessReadings {
        pages_epoch: state.host_writes.epoch(),
        pages_wrote: state
            .gather_witness
            .previous_pages_epoch(&key)
            .map(|since| state.host_writes.wrote_any_since(since, window.gpas)),
        // The guest's own statement about this resource, captured by the caller
        // in the identity space that owns it.
        stated_gen: Some(stated_gen),
        pending: PendingWrites::over(window.gpas),
    };
    // Every bind, vouched or not, so the route is a denominator rather than a
    // tally of refusals — the reading wanted is what fraction of binds this
    // device has a copy in flight over, and a count with no denominator cannot
    // say whether a repair moved it.
    note_store_route(counts.pending.route());
    // Report the host-write half's grounds, not just its answer. Three of its
    // four non-quiet values are this device declining to rule a write out rather
    // than one that landed here, and they want different repairs — name the
    // writer's pages, widen the ring, or stop writing the window at all. Taken
    // for every bind that had a previous one to ask about, so the split covers
    // the vouched binds too and `gw_hw_quiet` is the denominator.
    if let Some(verdict) = counts.pages_wrote {
        note_store_route(verdict.route());
    }
    // A generation is issued from the device-global counter and never reused, so
    // it is taken before the witness runs and spent only if the witness refuses
    // to vouch for the previous one. An unspent generation is not a leak: the
    // counter's whole contract is that a value is issued once and never again.
    let fresh = state.next_sampled_content_generation();
    let seen = observe(&mut state.gather_witness, key, window, counts, fresh);

    match seen.verdict {
        GatherVerdict::Rearmed => note_store_route("gw_rearm"),
        GatherVerdict::Unarmed => note_store_route("gw_unarmed"),
        GatherVerdict::Vouched => {
            note_store_route("gw_vouched");
            note_store_route_n("gw_vouched_kb", span / 1024);
        }
        GatherVerdict::Refused {
            guest_wrote,
            host_wrote_pages,
        } => {
            if guest_wrote {
                note_store_route("gw_refused_guest_store");
            }
            if host_wrote_pages {
                note_store_route("gw_refused_host_write");
            }
        }
    }
    // `gw_audit_kb` is every byte the fold still reads, so the cost of keeping
    // the alarm is reported in the same units as the gathers it saves.
    if !matches!(seen.audit, ContentAudit::Skipped) {
        note_store_route_n("gw_audit_kb", span / 1024);
    }
    match seen.audit {
        ContentAudit::Skipped => {}
        ContentAudit::Seeded => note_store_route("gw_audit_seed"),
        // The denominator `gw_audit_unsound` never had. Read the two together:
        // while this dominates `gw_audit_ok`, the alarm is not running and a
        // zero from it is not a measurement.
        ContentAudit::Restarted => note_store_route("gw_audit_restart"),
        // The arm holding itself open across a refusal. Costs one fold of a
        // window the gather reads anyway, and the count is what the alarm pays
        // to stay reachable at all.
        ContentAudit::Rebaselined => note_store_route("gw_audit_rebaseline"),
        ContentAudit::Agreed => note_store_route("gw_audit_ok"),
        // Read beside `gw_audit_ok` the same way `gw_audit_restart` is: it is
        // the audit's blind spot, and a boot where it dominates has an alarm
        // that is looking away rather than one that is seeing nothing.
        ContentAudit::Indebted => note_store_route("gw_audit_indebted"),
        ContentAudit::Disagreed => {
            note_store_route("gw_audit_unsound");
            // Once per window: a writer escaping both halves escapes them on
            // every bind, and the second line says nothing the first did not.
            // The count above carries the magnitude.
            crate::observe::emit::Emit::decline(
                "gather_witness",
                &GatherWitnessFault::VouchedBytesMoved {
                    key,
                    span,
                    binds: AUDIT_STRIDE,
                },
            )
            .fail_once(key.content_key());
        }
    }
    GatherOutcome {
        identity: GatheredIdentity {
            key: key.content_key(),
            generation: seen.generation,
        },
        vouch: seen.vouch,
    }
}

/// Whether this bind's identity names bytes some earlier gather already moved,
/// or one minted for bytes nothing has ever gathered.
///
/// The distinction decides whether a lookup miss is a fault at all, and it is
/// not recoverable from the identity: a `Fresh` identity is by construction one
/// no cache entry can have been retained under, so it *must* miss and the gather
/// that follows is the witness working. Only a `Vouched` identity that misses
/// says an image was lost.
///
/// Carried beside [`GatheredIdentity`] rather than folded into it because the
/// identity is what the engine *binds on* and this is what it *reports* — a
/// `Fresh` bind still retains under its new identity, which is exactly what lets
/// the next quiet bind hit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatherVouch {
    /// Both halves said the bytes cannot have moved since the gather that filled
    /// the retained image, so the identity is one the cache may already hold.
    Vouched,
    /// Either half saw a write, the window was re-pointed, or no token could
    /// answer — the generation was spent this bind and names bytes no retained
    /// image was ever built from.
    Fresh,
}

impl GatherVouch {
    /// True only for [`GatherVouch::Vouched`], so a caller cannot spell the
    /// question as "is there an identity" — there always is.
    pub fn is_vouched(self) -> bool {
        matches!(self, Self::Vouched)
    }
}

/// What the engine may bind a retained image on without looking at a byte.
///
/// Produced on **every** bind, not only vouched ones — the generation is what
/// separates the two. Where both halves agree the window's bytes cannot have
/// moved (no guest store into the pages, and no write by this device either) the
/// generation is the one the previous gather retained under, and the engine's
/// lookup hits. Where either half saw a write the generation is spent, so the
/// lookup misses, the bytes are read, and the new identity is what the retain
/// lands under — which is what makes the *following* quiet bind hit.
///
/// [`GatherVouch`] says which of the two this is. Do not reconstruct it from the
/// identity's presence: an absent identity would mean the witness was never
/// asked, and that is not a case [`note_gather`] can return.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatheredIdentity {
    /// Stable name for the window, in the device-wide sampled-identity keyspace.
    pub key: u64,
    /// Generation vouched for these bytes, from
    /// `DeviceState::next_sampled_content_generation`.
    pub generation: u64,
}

/// The witness itself: ask both halves about the last bind of the same window,
/// audit the answer on the stride, and leave the entry describing this bind.
fn observe(
    witness: &mut GatherWitness,
    key: GatherKey,
    window: GatherWindow<'_>,
    counts: WitnessReadings,
    fresh_generation: u64,
) -> GatherObservation {
    let GatherWindow {
        gpas,
        runs,
        span,
        page_size: _page_size,
    } = window;
    let WitnessReadings {
        pages_epoch,
        pages_wrote,
        pending,
        stated_gen: stated_now,
    } = counts;

    let stale = match witness.entries.get(&key) {
        Some(entry) => entry.gpas != gpas || entry.span != span,
        None => true,
    };
    if stale {
        witness.entries.remove(&key);
        witness.entries.insert(
            key,
            Entry {
                gpas: gpas.to_vec(),
                span,
                // A re-point gathers unconditionally, so folding here would buy
                // nothing the first audit does not: the stride seeds one before
                // there is any vouch for it to check.
                fold: 0,
                fold_valid: false,
                fold_seeded: false,
                binds_since_fold: 0,
                audit_armed: false,
                rebaselines: 0,
                pages_epoch,
                stated_gen: stated_now,
                generation: fresh_generation,
            },
        );
        return GatherObservation {
            verdict: GatherVerdict::Rearmed,
            audit: ContentAudit::Skipped,
            generation: fresh_generation,
            // The other place a generation is assigned, and the only one that
            // assigns unconditionally: a re-pointed window has no previous bind
            // of these pages to have vouched for them.
            vouch: GatherVouch::Fresh,
            // A re-point has no previous bind to compare a generation against,
            // which is the same "no answer" the channel gives an unaddressable
            // window. Reporting it as quiet would credit the stated channel with
            // vouching for a window that gathers unconditionally.
            stated: StatedGuestWrite::Unaddressed,
        };
    }

    // Copied out before the entry is borrowed mutably: the policy belongs to the
    // witness and the decisions that read it belong to one of its entries.
    let density = witness.audit;
    let entry = witness
        .entries
        .get_mut(&key)
        .expect("the stale branch above returns for every absent key");
    // The guest's statement is compared in its owning identity space. Only a
    // resource addressed at both binds has an answer; either side absent is an
    // explicit refusal rather than silence.
    let stated = match (entry.stated_gen, stated_now) {
        (Some(before), Some(now)) if before == now => StatedGuestWrite::Quiet,
        (Some(_), Some(_)) => StatedGuestWrite::Wrote,
        _ => StatedGuestWrite::Unaddressed,
    };
    entry.stated_gen = stated_now;

    // `pages_wrote == None` cannot happen beside a live entry, and reading a
    // missing answer as quiet would vouch on the strength of not having asked.
    // Taken once: the vouch arm and the refusal arm below want exact complements
    // of this, and two spellings of it is one edit away from a witness that
    // vouches and reports a host write in the same breath.
    let host_quiet = pages_wrote.is_some_and(|seen| !seen.wrote());
    let verdict = match stated {
        StatedGuestWrite::Unaddressed => GatherVerdict::Unarmed,
        StatedGuestWrite::Quiet if host_quiet => GatherVerdict::Vouched,
        StatedGuestWrite::Quiet | StatedGuestWrite::Wrote => GatherVerdict::Refused {
            guest_wrote: matches!(stated, StatedGuestWrite::Wrote),
            host_wrote_pages: !host_quiet,
        },
    };
    let vouched = matches!(verdict, GatherVerdict::Vouched);

    // SAFETY (every `fold_runs` below): `runs` describe the window this draw is
    // about to gather from, so their pointers are live here for the same reason
    // they are live there. On a vouched bind the gather will be skipped, but the
    // runs were resolved by the same producer in the same call and name the same
    // pages, which the entry's page set is checked against above.
    let audit = if !pending.settled() {
        // A copy this device submitted is in flight over these pages and the
        // fold is a CPU read of them, so whatever it reads now is neither the
        // before nor reliably the after. Comparing across that reports the
        // device's own queue as a stale image.
        //
        // Declines rather than settles. The audit is a diagnostic and must not
        // introduce a stall the shipping path does not have — and it must not
        // take the engine lock from inside the witness, which the drain thread
        // reaches with its own locks held. Dropping the baseline is the honest
        // answer: this window is not comparable right now, and the stride will
        // bring it back when the queue is quiet.
        entry.audit_armed = false;
        entry.fold_valid = false;
        entry.rebaselines = 0;
        entry.binds_since_fold = 0;
        ContentAudit::Indebted
    } else if entry.audit_armed {
        // The claim under test is "a vouched bind means these bytes did not
        // move", so the bind that tests it is the *next vouched one* after a
        // baseline — not one sixty-four binds later. Waiting for the stride
        // again is what made the comparison unreachable: any refusal in between
        // drops the baseline, and a run of sixty-four vouched binds is not
        // something this workload produces.
        if vouched {
            let fold = unsafe { fold_runs(runs, span) };
            let audit = match fold == entry.fold {
                true => ContentAudit::Agreed,
                false => ContentAudit::Disagreed,
            };
            entry.fold = fold;
            entry.fold_valid = true;
            entry.audit_armed = density.stays_armed();
            entry.rebaselines = 0;
            entry.binds_since_fold = 0;
            audit
        } else if entry.rebaselines < AUDIT_REBASELINE_LIMIT {
            // Refused, so the bytes were free to move and the old baseline says
            // nothing. The gather is about to read this window anyway, so a
            // fresh baseline costs the fold and keeps the arm alive for the
            // vouched bind it is waiting for.
            entry.fold = unsafe { fold_runs(runs, span) };
            entry.fold_seeded = true;
            entry.fold_valid = true;
            entry.rebaselines += 1;
            ContentAudit::Rebaselined
        } else {
            // A window refused this many times running is not one a vouch is
            // being claimed about, and holding the arm open folds it on every
            // bind. Disarm and let the stride bring it back.
            entry.audit_armed = false;
            entry.rebaselines = 0;
            entry.fold_valid = false;
            entry.binds_since_fold = 0;
            ContentAudit::Restarted
        }
    } else if entry.binds_since_fold >= density.stride() {
        // Arm: take the baseline whatever this bind's verdict is. The fold reads
        // the guest pages directly, so it describes the window on a vouched bind
        // (where the gather is skipped) exactly as it does on a refused one.
        entry.fold = unsafe { fold_runs(runs, span) };
        entry.fold_seeded = true;
        entry.fold_valid = true;
        entry.audit_armed = true;
        entry.rebaselines = 0;
        entry.binds_since_fold = 0;
        ContentAudit::Seeded
    } else {
        entry.binds_since_fold += 1;
        // A bind the witness refused may have moved the bytes with nothing
        // reading them, which is precisely when the stored fold stops describing
        // the window.
        entry.fold_valid &= vouched;
        ContentAudit::Skipped
    };

    // Keep the generation only where both halves vouch for the bytes *and* the
    // audit did not just catch them out. A `Disagreed` audit is not only an
    // alarm: the vouch it refutes is live, so dropping the generation here is
    // what stops the next bind serving the stale image again.
    //
    // This one expression decides both what the entry names and what the caller
    // is told the name is worth. Re-deriving the second from `verdict` would get
    // a `Disagreed` audit wrong — that arm vouches and still spends the
    // generation — and a reader comparing the two spellings could not tell which
    // was the rule.
    let kept = vouched && !matches!(audit, ContentAudit::Disagreed);
    if !kept {
        entry.generation = fresh_generation;
    }
    entry.pages_epoch = pages_epoch;
    GatherObservation {
        verdict,
        audit,
        generation: entry.generation,
        vouch: if kept {
            GatherVouch::Vouched
        } else {
            GatherVouch::Fresh
        },
        stated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::GuestRun;

    const KEY: GatherKey = GatherKey::Mapping {
        mid: 11,
        base_off: 0,
    };
    const PAGE: usize = 4096;
    const GPAS: [u64; 1] = [8 * PAGE as u64];

    /// A one-page window over `runs`, at `gpas`, judged against a device that has
    /// written nothing.
    fn one_page<'a>(gpas: &'a [u64], runs: &'a [GuestRun]) -> GatherWindow<'a> {
        GatherWindow {
            gpas,
            runs,
            span: PAGE as u64,
            page_size: PAGE,
        }
    }

    /// Neither the guest nor this device wrote the window since the previous
    /// bind.
    const QUIET: WitnessReadings = WitnessReadings {
        pages_epoch: 1,
        pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Quiet),
        pending: PendingWrites::Disjoint,
        stated_gen: Some(StatedGeneration::Mapping(0)),
    };

    /// One bind, discarding the audit — for the tests that are about the verdict.
    fn verdict(
        w: &mut GatherWitness,
        window: GatherWindow<'_>,
        counts: WitnessReadings,
        gen: u64,
    ) -> GatherVerdict {
        observe(w, KEY, window, counts, gen).verdict
    }

    /// Bind `n` times with nothing writing anything, returning the last
    /// observation.
    fn bind_quietly(
        w: &mut GatherWitness,
        gpas: &[u64],
        runs: &[GuestRun],
        n: u32,
    ) -> GatherObservation {
        let mut last = None;
        for _ in 0..n {
            last = Some(observe(w, KEY, one_page(gpas, runs), QUIET, next_gen()));
        }
        last.expect("bind_quietly is never called with n == 0")
    }

    /// Bind quietly until the audit next runs, and return that bind.
    ///
    /// Spelled as "until it fires" rather than as a bind count so the tests say
    /// what they mean and do not encode the stride's off-by-ones — the exact
    /// bind an audit lands on is [`AUDIT_STRIDE`]'s business, not theirs.
    fn bind_to_next_audit(
        w: &mut GatherWitness,
        gpas: &[u64],
        runs: &[GuestRun],
    ) -> GatherObservation {
        bind_to_next_audit_with(w, gpas, runs, QUIET)
    }

    fn bind_to_next_audit_with(
        w: &mut GatherWitness,
        gpas: &[u64],
        runs: &[GuestRun],
        counts: WitnessReadings,
    ) -> GatherObservation {
        for _ in 0..=2 * AUDIT_STRIDE {
            let seen = observe(w, KEY, one_page(gpas, runs), counts, next_gen());
            if seen.audit != ContentAudit::Skipped {
                return seen;
            }
        }
        panic!("no audit within two strides, so the fold is never reached at all");
    }

    /// A generation that has never been issued before, as the device's own
    /// counter promises.
    fn next_gen() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(1);
        SEQ.fetch_add(1, Ordering::Relaxed)
    }

    fn run_over(buf: &[u8]) -> GuestRun {
        GuestRun {
            host_ptr: buf.as_ptr() as usize,
            len: buf.len() as u64,
        }
    }

    #[test]
    fn the_fold_sees_a_single_changed_byte_anywhere_in_the_window() {
        let mut buf = vec![7u8; 4096 + 3];
        let base = unsafe { fold_runs(&[run_over(&buf)], buf.len() as u64) };
        for at in [0usize, 1, 8, 1000, 4095, 4096, 4098] {
            let saved = buf[at];
            buf[at] ^= 0x40;
            let moved = unsafe { fold_runs(&[run_over(&buf)], buf.len() as u64) };
            assert_ne!(base, moved, "a flipped byte at {at} folded the same");
            buf[at] = saved;
        }
        assert_eq!(base, unsafe {
            fold_runs(&[run_over(&buf)], buf.len() as u64)
        });
    }

    #[test]
    fn the_fold_is_position_sensitive_so_a_permuted_window_is_not_unchanged() {
        // Distinct bytes at the two swapped indices, or the "permutation" is the
        // identity and the test proves nothing.
        let a: Vec<u8> = (0..512u32).map(|i| (i / 2) as u8).collect();
        let mut b = a.clone();
        assert_ne!(a[0], a[256]);
        b.swap(0, 256);
        assert_ne!(
            unsafe { fold_runs(&[run_over(&a)], a.len() as u64) },
            unsafe { fold_runs(&[run_over(&b)], b.len() as u64) },
            "swapping two words folded the same, so the fold sums rather than orders"
        );
    }

    /// The generation is the whole product of this witness, and its contract is
    /// that it survives exactly as long as the bytes it names.
    ///
    /// Held while both halves vouch, and replaced by every other verdict — the
    /// bytes being unchanged is not the question, because a bind where either
    /// half saw a write is a bind whose bytes nothing has vouched for.
    ///
    /// Asserted on the observation the bind returns rather than by reading the
    /// map back, because that read is what the engine used to do and it cannot
    /// come back absent: every arm here leaves an entry, so an `Option` from it
    /// is `Some` whatever the verdict was. The [`GatherVouch`] beside each
    /// generation is the part that varies, and it is checked at every step.
    #[test]
    fn the_vouched_generation_outlives_a_quiet_bind_and_no_other_kind() {
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];

        let first = observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, 10);
        assert_eq!((first.generation, first.vouch), (10, GatherVouch::Fresh));

        // Quiet at both halves: the same bytes, so the same generation, and the
        // only bind of the four that names an image an earlier gather filled.
        let quiet = observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, 11);
        assert_eq!((quiet.generation, quiet.vouch), (10, GatherVouch::Vouched));

        // A host write into the pages, with the bytes unchanged. Unchanged is
        // not enough: this device wrote them, so nothing vouches for them.
        let host_wrote = observe(
            &mut w,
            KEY,
            one_page(&GPAS, &runs),
            WitnessReadings {
                pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Overlap),
                ..QUIET
            },
            12,
        );
        assert_eq!(
            (host_wrote.generation, host_wrote.vouch),
            (12, GatherVouch::Fresh),
            "a generation survived a write to its own pages"
        );

        // A guest store, likewise. The resource-table statement advances the
        // resource generation consumed by this bind.
        buf[3] ^= 0xff;
        let guest_wrote = observe(
            &mut w,
            KEY,
            one_page(&GPAS, &runs),
            WitnessReadings {
                stated_gen: Some(StatedGeneration::Mapping(1)),
                ..QUIET
            },
            13,
        );
        assert_eq!(
            (guest_wrote.generation, guest_wrote.vouch),
            (13, GatherVouch::Fresh)
        );
    }

    /// The guest's resource statement answers only where it addresses the
    /// sampled resource.
    ///
    /// Four claims, and the first two are what make the reading usable: a
    /// generation that has not moved is [`StatedGuestWrite::Quiet`] *including at
    /// generation 0*, and one that has moved is [`StatedGuestWrite::Wrote`]. The
    /// third is the fail-closed rule — a window the channel cannot address at
    /// either bind is `Unaddressed` and never quiet, so an absent statement is
    /// not read as a statement of silence. The fourth is that a re-point reports
    /// `Unaddressed` too, because it has no previous bind to compare against and
    /// gathers unconditionally.
    #[test]
    fn the_stated_channel_answers_only_where_the_guest_addresses_it() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let stated = |gen: Option<u32>| WitnessReadings {
            stated_gen: gen.map(StatedGeneration::Mapping),
            ..QUIET
        };

        // First sight of the window: a re-point, so no comparison exists yet.
        let first = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(0)), 10);
        assert_eq!(first.stated, StatedGuestWrite::Unaddressed);

        // Generation 0 twice is a real quiet answer, not an absent one — the
        // mapping has been addressed and the guest has written it zero times.
        let quiet = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(0)), 11);
        assert_eq!(quiet.stated, StatedGuestWrite::Quiet);

        // The guest states a CPU write: `resource_validity::apply` bumps the
        // mapping's generation, and the channel reports it.
        let wrote = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(1)), 12);
        assert_eq!(wrote.stated, StatedGuestWrite::Wrote);

        // Settled at the new generation, quiet again.
        let settled = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(1)), 13);
        assert_eq!(settled.stated, StatedGuestWrite::Quiet);

        // The mapping goes away. Fail closed: not quiet, whatever the device
        // write record says, and the bind before it does not become quiet
        // retroactively.
        let gone = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(None), 14);
        assert_eq!(gone.stated, StatedGuestWrite::Unaddressed);
        let still_gone = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(None), 15);
        assert_eq!(still_gone.stated, StatedGuestWrite::Unaddressed);
    }

    /// A witness at a stated audit density, for the two tests that are about the
    /// density rather than about the witness.
    fn witness_auditing(density: AuditDensity) -> GatherWitness {
        GatherWitness {
            audit: density,
            ..GatherWitness::default()
        }
    }

    /// [`AuditDensity::EveryBind`] judges every bind it can, and the shipping
    /// stride judges none of the same population.
    ///
    /// Both arms are asserted because only the pair says the switch does
    /// anything: the dense arm alone would pass against a witness that always
    /// audited, and the strided arm alone against one that never did.
    ///
    /// Six binds rather than a computed count, and the dense arm is allowed its
    /// first three — a comparison needs a baseline bind and a bind to spend it
    /// on, and the first sight of a window is a rearm that has neither.
    #[test]
    fn every_bind_compares_where_the_shipping_stride_has_not_yet_looked() {
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let compares = |density| {
            let mut w = witness_auditing(density);
            (0..6)
                .map(|_| observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, next_gen()))
                .filter(|seen| seen.audit == ContentAudit::Agreed)
                .count()
        };
        assert_eq!(
            compares(AuditDensity::Strided),
            0,
            "the shipping stride is {AUDIT_STRIDE} binds, so six cannot reach a comparison"
        );
        assert_eq!(
            compares(AuditDensity::EveryBind),
            3,
            "every bind after the first three must compare against the bind before it"
        );
    }

    /// The reading the switch exists to produce: a writer that escapes **both**
    /// halves of the witness is caught on the very next bind.
    ///
    /// This is the failure the sampled cache's identity-only lookup cannot report
    /// any other way — an elision correctly taken and one wrongly taken are the
    /// same absence — so the audit catching it is the only instrument there is.
    /// The bytes here move with no guest store and no recorded host write, which
    /// is exactly the shape of an unrecorded writer.
    ///
    /// The vouch is asserted too, and it is the half that matters at a bind: a
    /// `Disagreed` audit that still handed back a live generation would leave the
    /// next bind serving the same stale image the audit had just convicted.
    #[test]
    fn an_unrecorded_write_is_convicted_on_the_next_bind_and_spends_the_generation() {
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let settled = bind_quietly(&mut w, &GPAS, &runs, 4);
        assert_eq!(
            (settled.audit, settled.vouch),
            (ContentAudit::Agreed, GatherVouch::Vouched),
            "the window has to be under a live vouch before the escape means anything"
        );

        // Neither half is told. This is the writer the module's whole soundness
        // argument assumes does not exist.
        buf[2048] ^= 0xff;

        let caught = observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, 77);
        assert_eq!(
            (caught.verdict, caught.audit),
            (GatherVerdict::Vouched, ContentAudit::Disagreed),
            "both halves vouched and the bytes had moved, which is the alarm's whole purpose"
        );
        assert_eq!(
            (caught.generation, caught.vouch),
            (77, GatherVouch::Fresh),
            "a convicted vouch must not be handed on, or the next bind serves the stale image"
        );
    }

    /// The audit declines to compare across a copy this device has submitted and
    /// the GPU has not run — and the *vouch* is untouched by it.
    ///
    /// Both halves matter and they pull opposite ways. The fold is a CPU read of
    /// guest pages and is ordered against that copy by nothing, so comparing
    /// across it reports the device's own queue as a stale image. The gather the
    /// cache elides is a GPU copy on the same queue as the writeback, so it *is*
    /// ordered and the vouch is still sound — making this refuse the vouch too
    /// would cost re-gathers to fix a defect in the instrument.
    ///
    /// `Unnamed` is asserted beside `Overlap` because a footprint nobody could
    /// name is not a proof of disjointness, and reading it as one is how the
    /// blind spot would come back as "we could not tell, so we compared".
    #[test]
    fn an_unlanded_copy_stops_the_audit_comparing_and_leaves_the_vouch_alone() {
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let bind = |w: &mut GatherWitness, pending| {
            observe(
                w,
                KEY,
                one_page(&GPAS, &runs),
                WitnessReadings { pending, ..QUIET },
                next_gen(),
            )
        };

        // Quiet queue: the audit reaches a comparison, which is the control —
        // without it the assertions below would pass against an audit that never
        // ran at all.
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let settled = bind_quietly(&mut w, &GPAS, &runs, 4);
        assert_eq!(
            (settled.verdict, settled.audit),
            (GatherVerdict::Vouched, ContentAudit::Agreed)
        );

        for pending in [PendingWrites::Overlap, PendingWrites::Unnamed] {
            let seen = bind(&mut w, pending);
            assert_eq!(
                seen.audit,
                ContentAudit::Indebted,
                "{pending:?} folded across a copy that has not landed"
            );
            assert_eq!(
                seen.verdict,
                GatherVerdict::Vouched,
                "{pending:?} moved the vouch, which is ordered behind that copy and did not need to"
            );
            assert_eq!(seen.vouch, GatherVouch::Vouched);
        }
    }

    /// A window whose bytes and pages both stand still, bound twice: the whole
    /// point of the exercise, and the verdict whose count says what the cache
    /// saves.
    #[test]
    fn a_window_nothing_writes_is_vouched_for_on_the_second_bind() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed,
            "first sight has nothing to compare against"
        );
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Vouched
        );
    }

    /// The guest's resource statement reports the store, so the vouch is
    /// refused and the bytes are read.
    #[test]
    fn a_guest_store_into_the_window_refuses_the_vouch() {
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        buf[100] ^= 0xff;
        assert_eq!(
            verdict(
                &mut w,
                one_page(&GPAS, &runs),
                WitnessReadings {
                    stated_gen: Some(StatedGeneration::Mapping(1)),
                    ..QUIET
                },
                next_gen()
            ),
            GatherVerdict::Refused {
                guest_wrote: true,
                host_wrote_pages: false
            }
        );
    }

    /// The whole point of moving the fold onto a stride: a vouched bind reads no
    /// byte of the window at all.
    ///
    /// [`ContentAudit::Skipped`] *is* that statement — it is returned only where
    /// `fold_runs` was not called — so this is the test that would fail if the
    /// fold went back on the per-bind path, and the reason the audit's outcome is
    /// reported rather than kept inside the function.
    #[test]
    fn a_vouched_bind_before_the_stride_reads_none_of_the_window() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        // The rearm, then every bind up to but not including the audit.
        let last = bind_quietly(&mut w, &GPAS, &runs, AUDIT_STRIDE + 1);
        assert_eq!(last.verdict, GatherVerdict::Vouched);
        assert_eq!(
            last.audit,
            ContentAudit::Skipped,
            "a bind inside the stride folded the window anyway"
        );
        // And the one that lands on the stride does fold, with nothing to compare
        // against yet.
        let due = bind_to_next_audit(&mut w, &GPAS, &runs);
        assert_eq!(due.audit, ContentAudit::Seeded);
        // Which then gives the next audit something to check.
        let checked = bind_to_next_audit(&mut w, &GPAS, &runs);
        assert_eq!(checked.audit, ContentAudit::Agreed);
    }

    /// A refusal inside a stride no longer disarms the alarm: the next arm takes
    /// a fresh baseline and the comparison still happens.
    ///
    /// This is the repair, and the reason it was needed. Refusals are roughly
    /// two binds in five, and the audit used to take its baseline and compare on
    /// the *same* stride bind — so reaching a comparison needed `AUDIT_STRIDE`
    /// consecutive vouched binds of one window, which this workload does not
    /// produce. Three consecutive driven boots read `gw_audit_ok` **0** against
    /// `gw_audit_seed` in the hundreds, so `gw_audit_unsound`'s zero was a check
    /// that never ran while reading exactly like one that ran and agreed. A real
    /// writer escaping both halves hid behind it.
    ///
    /// The test drives one refusal into an otherwise quiet run, which is the
    /// smallest thing that put the old audit into the state the whole workload
    /// was permanently in.
    #[test]
    fn a_refusal_inside_a_stride_still_leaves_the_alarm_able_to_compare() {
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];
        bind_quietly(&mut w, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded,
            "arming takes the baseline"
        );
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Agreed,
            "and the very next vouched bind is the one that checks it"
        );

        // One refused bind: this device wrote a page of the window. Nothing
        // about the bytes changed.
        let refused = observe(
            &mut w,
            KEY,
            one_page(&GPAS, &runs),
            WitnessReadings {
                pages_epoch: 2,
                pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Overlap),
                ..QUIET
            },
            next_gen(),
        );
        assert!(
            matches!(
                refused.verdict,
                GatherVerdict::Refused {
                    host_wrote_pages: true,
                    ..
                }
            ),
            "the fixture must actually refuse, or the rest proves nothing"
        );

        // The old design answered `Restarted` here and never compared again
        // until sixty-four consecutive vouches, which never came.
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded,
            "the arm after a refusal takes a fresh baseline rather than giving up"
        );
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Agreed,
            "and the alarm is running again one bind later"
        );
    }

    /// A refused bind between the baseline and the check must not produce a
    /// `Disagreed` for a witness that was right.
    ///
    /// An alarm that cries wolf is worse than no alarm, since the whole value of
    /// this one is that a nonzero count means something. The refusal is the
    /// witness working — it saw the store — so the baseline is retaken from the
    /// bytes the gather is about to read, and the check that follows compares
    /// across a vouched bind only.
    #[test]
    fn a_refused_bind_between_audits_does_not_leave_a_false_alarm_behind() {
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        bind_quietly(&mut w, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded
        );

        // A stated guest store repaints the window while the audit is armed.
        // The gather happens, so nothing is stale — but the baseline is now
        // from before the repaint.
        buf[11] ^= 0xff;
        let after_store = WitnessReadings {
            stated_gen: Some(StatedGeneration::Mapping(1)),
            ..QUIET
        };
        let refused = observe(&mut w, KEY, one_page(&GPAS, &runs), after_store, next_gen());
        assert!(matches!(refused.verdict, GatherVerdict::Refused { .. }));
        assert_eq!(
            refused.audit,
            ContentAudit::Rebaselined,
            "the armed window retakes its baseline from the repainted bytes"
        );

        assert_eq!(
            bind_to_next_audit_with(&mut w, &GPAS, &runs, after_store).audit,
            ContentAudit::Agreed,
            "comparing across the repaint would have been a false alarm"
        );
    }

    /// An armed window that is only ever refused gives up rather than folding on
    /// every bind.
    ///
    /// The arm costs one fold per refused bind, and the rail it audits moves
    /// 842 MB/s — so a window the witness never vouches for would pull the whole
    /// of it back through the audit. `AUDIT_REBASELINE_LIMIT` bounds that, and
    /// the stride is what brings the window back.
    #[test]
    fn an_armed_window_that_is_never_vouched_gives_up_instead_of_folding_forever() {
        let mut w = GatherWitness::default();
        let buf = vec![0x11u8; PAGE];
        let runs = [run_over(&buf)];
        bind_quietly(&mut w, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded
        );

        let refuse = |w: &mut GatherWitness| {
            observe(
                w,
                KEY,
                one_page(&GPAS, &runs),
                WitnessReadings {
                    pages_epoch: 2,
                    pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Overlap),
                    ..QUIET
                },
                next_gen(),
            )
            .audit
        };
        for i in 0..AUDIT_REBASELINE_LIMIT {
            assert_eq!(
                refuse(&mut w),
                ContentAudit::Rebaselined,
                "refusal {i} is still inside the arm's budget"
            );
        }
        assert_eq!(
            refuse(&mut w),
            ContentAudit::Restarted,
            "past the budget the arm gives up rather than folding on every bind"
        );
        // And having given up it stops folding, so the cost really is bounded.
        assert_eq!(refuse(&mut w), ContentAudit::Skipped);
    }

    /// The unsound case, produced deliberately: bytes changed under pages neither
    /// half of the witness saw written. This is the shape a host-side writer into
    /// guest RAM makes, and it is what the audit exists to catch — so if a driven
    /// boot ever reports `gw_audit_unsound`, this test says what that means.
    ///
    /// The audit is a repair as well as an alarm: the generation it refutes is
    /// live, so it must not survive the bind that caught it.
    #[test]
    fn bytes_moving_under_a_vouch_are_caught_by_the_audit_and_cost_the_generation() {
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        // Rearm, then seed a fold the next audit can compare against.
        bind_quietly(&mut w, &GPAS, &runs, 1);
        let seeded = bind_to_next_audit(&mut w, &GPAS, &runs);
        assert_eq!(seeded.audit, ContentAudit::Seeded);
        let vouched_gen = seeded.generation;

        // No `guest_wrote_page` and no host write recorded: the bytes move with
        // both halves of the witness none the wiser.
        buf[7] ^= 0xff;
        let caught = bind_to_next_audit(&mut w, &GPAS, &runs);
        assert_eq!(
            caught.verdict,
            GatherVerdict::Vouched,
            "the witness is what is being caught out, so it must still be vouching"
        );
        assert_eq!(caught.audit, ContentAudit::Disagreed);
        assert_ne!(
            caught.generation, vouched_gen,
            "the refuted generation survived the audit that refuted it, so the \
             next bind serves the stale image again"
        );
        // The one bind where the verdict and the vouch disagree, and the reason
        // the engine is told the vouch rather than the verdict: this bind
        // vouches and still spends its generation, so an engine deriving
        // "vouched" from the verdict would count a guaranteed miss as a
        // retention failure.
        assert_eq!(
            caught.vouch,
            GatherVouch::Fresh,
            "a generation the audit just spent was reported as one the cache \
             could still be holding an image under"
        );
    }

    /// A window with no guest resource statement must never vouch, however
    /// still the bytes are. Fail closed: half a witness is not a witness.
    #[test]
    fn an_unaddressed_resource_statement_never_vouches() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let unaddressed = WitnessReadings {
            stated_gen: None,
            ..QUIET
        };
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), unaddressed, next_gen()),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), unaddressed, next_gen()),
            GatherVerdict::Unarmed
        );
    }

    /// A window re-pointed at different pages has no predecessor, even though its
    /// key repeats. Comparing across the move would compare two different surfaces.
    #[test]
    fn a_window_whose_pages_move_rearms_rather_than_comparing_across_the_move() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let moved = [9 * PAGE as u64];
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            verdict(&mut w, one_page(&moved, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed,
            "same key, different pages: nothing to compare"
        );
        assert_eq!(
            verdict(&mut w, one_page(&moved, &runs), QUIET, next_gen()),
            GatherVerdict::Vouched
        );
    }

    #[test]
    fn the_fold_stops_at_span_even_when_the_runs_are_longer() {
        let buf = vec![3u8; 256];
        let short = unsafe { fold_runs(&[run_over(&buf)], 64) };
        let head = vec![3u8; 64];
        assert_eq!(short, unsafe { fold_runs(&[run_over(&head)], 64) });
    }

    /// Live sampled windows are retained without a capacity eviction.
    #[test]
    fn live_windows_are_not_evicted_by_capacity() {
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];

        let key_at = |i: u64| GatherKey::Mapping {
            mid: 11,
            base_off: i * PAGE as u64,
        };
        let gpas_at = |i: u64| [(64 + i) * PAGE as u64];

        const DISTINCT_WINDOWS: u64 = 512;
        for i in 0..DISTINCT_WINDOWS {
            let gpas = gpas_at(i);
            observe(&mut w, key_at(i), one_page(&gpas, &runs), QUIET, next_gen());
        }
        assert_eq!(w.entries.len(), DISTINCT_WINDOWS as usize);
        assert!((0..DISTINCT_WINDOWS).all(|i| w.entries.contains_key(&key_at(i))));
    }

    /// Witness entries end with the task or mapping that owns the sampled
    /// window; unrelated live resources survive either end.
    #[test]
    fn resource_lifetime_retirement_releases_only_the_owned_windows() {
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];
        let keys = [
            GatherKey::TaskGva {
                task_id: 7,
                resource_ref: 3,
                gva: 0x1000,
            },
            GatherKey::TaskGva {
                task_id: 8,
                resource_ref: 4,
                gva: 0x1000,
            },
            GatherKey::Mapping {
                mid: 11,
                base_off: 0,
            },
            GatherKey::Mapping {
                mid: 12,
                base_off: 0,
            },
        ];
        for key in keys {
            observe(&mut w, key, one_page(&GPAS, &runs), QUIET, next_gen());
        }

        w.retire_task(7);
        w.retire_mapping(11);
        assert_eq!(w.entries.len(), 2);
        assert!(!w.entries.contains_key(&keys[0]));
        assert!(!w.entries.contains_key(&keys[2]));
        assert!(w.entries.contains_key(&keys[1]));
        assert!(w.entries.contains_key(&keys[3]));
    }
}
