//! Resource-validity ownership for render targets.
//!
//! A render Store preserves pixels in the host attachment. It does not imply a
//! host-to-guest transfer. The guest makes that transfer observable by naming
//! the resource in `CmdSynchronizeResources`, or this device needs the guest
//! bytes itself for a fallback reader. Until then, [`PendingWritebacks`] records
//! that the engine image is authoritative and repeated Stores into the resource
//! replace one another without touching guest RAM.
//!
//! # A resource owns its transfer backing
//!
//! Debts carry the task-local texture reference, GVA declaration, geometry,
//! format, and resource generation. The live GVA resource separately retains the
//! ordered physical pages of its transfer backing. Ordinary task unmap changes
//! virtual-address bookkeeping but does not retarget that resource. Explicit
//! discard drops the transfer backing, and the next prepare or synchronize
//! resolves it again without replacing the host texture.
//!
//! This is the safety property the former deferred-window design lacked: it
//! parked raw host pointers across guest execution. This model retains page
//! identities, not pointers; every transfer still constructs bounded
//! `GuestSlice`s from the owning RAMBlock import.
//!
//! # Validity transitions decide direction
//!
//! A GPU Store makes the host image authoritative. A later guest write makes
//! the transfer backing newer; payment then abandons the host image rather than
//! overwriting the guest's work. Task-GVA resources use the validity generation
//! keyed by `(task, texture_ref)`.
//!
//! A named synchronize pays only its object list through
//! [`submit_for_resources`]. Readers that know a texture call
//! [`pay_for_texture`]. Only a genuinely unnameable reader uses [`pay_all`].
//! Completion stamps alone do not publish resources.
//!
//! The engine's `gpu_only_content` flag keeps an unpaid image alive. A
//! successful payment calls `note_resident_content_copied_out`; replacement,
//! invalidation, task retirement, and generation movement release the same
//! ownership without inventing a guest write.
//!
use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// The guest resource that owns one GVA render attachment.
///
/// Unlike the address, this is also what `CmdSynchronizeResources` names. A
/// task is part of the key because object references are task-local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaResourceKey {
    pub task_id: u32,
    pub texture_ref: u32,
}

/// One render plane of a GVA resource.
///
/// # Why the ledger's unit is not the resource
///
/// A render pass targets exactly one mip level, and a level is a sub-range of
/// the resource's single allocation — `runtime::draw::render_target`'s
/// `level != 0` arm resolves it to that level's own `(gva, row_stride, height)`.
/// So one reference legitimately owns several live planes at once, and a ledger
/// keyed by the reference holds one entry where the guest is using three.
///
/// That was measured rather than reasoned. A driven macos-26 boot cycles one
/// reference through three declarations whose addresses are contiguous and
/// whose spans fall in exact 4:1 ratios — 256×192, 128×96, 64×48 of one RGBA8
/// allocation, the compositor's blur/backdrop pyramid. Keyed by the reference,
/// arming level 1's Store drops level 0's unpaid frame and every level change
/// mints a new generation, so no level's resident is ever reused.
///
/// [`GvaResourceKey`] stays the **resource**, because that is what
/// `CmdSynchronizeResources` and `CmdDeleteResource` name and what
/// `resource_validity` and `blit_exec` ask with — neither holds an address. The
/// derived `Ord` puts `resource` first, so `BTreeMap::range` over
/// [`GvaResourceKey::planes`] is every plane of one resource and the
/// resource-wide operations stay one lookup shape rather than a second map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaPlaneKey {
    pub resource: GvaResourceKey,
    pub gva: u64,
}

impl GvaResourceKey {
    /// This resource's plane at one guest address.
    pub fn plane(self, gva: u64) -> GvaPlaneKey {
        GvaPlaneKey {
            resource: self,
            gva,
        }
    }

    /// Every plane of this resource, as a `BTreeMap` range.
    ///
    /// Total by construction: the bounds are this resource's own lowest and
    /// highest representable plane, so no plane of it can sort outside them and
    /// no plane of another resource can sort inside.
    fn planes(self) -> std::ops::RangeInclusive<GvaPlaneKey> {
        self.plane(0)..=self.plane(u64::MAX)
    }
}

/// A frame held only by a GVA target's engine-resident image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GvaWritebackDebt {
    pub linear: crate::runtime::draw::LinearColorTarget,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub generation: u64,
    /// Exact semantic resource and content version held by the resident.
    /// `None` is limited to resources not yet constructed in the canonical
    /// object graph and to synthetic tests.
    pub content: Option<(
        reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
        reims_vgpu_protocol::ContentVersion,
    )>,
    pub guest_write: crate::runtime::buffer_write_gen::ResourceWriteStamp,
    pub seq: u64,
}

/// The transfer backing retained by one live plane of a GVA texture resource.
///
/// The plane owns this physical-page identity after its virtual declaration has
/// been resolved. Task unmap changes the task's CPU mapping bookkeeping; it does
/// not retarget a live resource. An explicit resource discard drops only
/// `pages` — on every plane — allowing the next prepare/synchronize to establish
/// a new transfer backing without changing any host texture's identity.
///
/// The address is in the [`GvaPlaneKey`], so what remains here is what varies
/// per plane at one address: its length, its host-texture identity, and its
/// pages.
#[derive(Clone, Debug)]
struct GvaResourceState {
    generation: u64,
    span: u64,
    pages: Option<std::sync::Arc<[u64]>>,
}

/// Every GVA render resource whose current frame exists only in a host resident.
///
/// Resources key by the plane of the task-local reference a pass rendered into
/// — see [`GvaPlaneKey`] for why the plane and not the reference. A second Store
/// into the same plane replaces the first rather than queueing another frame.
#[derive(Debug, Default)]
pub struct PendingWritebacks {
    gva_debts: std::collections::BTreeMap<GvaPlaneKey, GvaWritebackDebt>,
    gva_resources: std::collections::BTreeMap<GvaPlaneKey, GvaResourceState>,
    next_seq: u64,
    next_gva_generation: u64,
}

impl PendingWritebacks {
    /// Mappings currently owed a frame.
    pub fn len(&self) -> usize {
        self.gva_debts.len()
    }

    /// Whether anything is owed at all — the check every reader makes, and the
    /// one that has to be free.
    pub fn is_empty(&self) -> bool {
        self.gva_debts.is_empty()
    }

    /// Record a host-authoritative frame for one plane of a GVA resource.
    ///
    /// A second Store into the same plane replaces the earlier debt. The
    /// returned previous debt names an older resident identity that the caller
    /// must release when the declaration changed.
    ///
    /// The debt's own `gva` picks the plane, so a pass into a *different* level
    /// of the same reference queues beside the first rather than dropping it.
    /// Keyed by the reference, arming a blur pyramid's level 1 discarded level
    /// 0's unpaid frame — see [`GvaPlaneKey`].
    #[must_use = "a replaced resource debt may own an older resident identity"]
    pub fn arm_gva(
        &mut self,
        key: GvaResourceKey,
        mut debt: GvaWritebackDebt,
    ) -> Option<GvaWritebackDebt> {
        debt.seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let previous = self
            .gva_debts
            .insert(key.plane(debt.linear.target_gva()), debt);
        if previous.is_some() {
            crate::runtime::drain::note_store_route("gvadebt_superseded");
        }
        crate::runtime::drain::note_store_route("gvadebt_armed");
        previous
    }

    /// Establish or retrieve the lifetime identity of one live plane of a GVA
    /// resource.
    ///
    /// `pages` is accepted only on the first resolution after construction or
    /// explicit discard. Repeated draws and ordinary task unmaps keep the
    /// retained physical backing and the same host-texture generation.
    ///
    /// A plane of this reference at an address it has not used before is simply
    /// a new plane — the mip level case [`GvaPlaneKey`] describes — and gets its
    /// own generation without disturbing the others.
    ///
    /// # A second span at one address is a second resource
    ///
    /// A plane's length is fixed for its life: it comes from the creation
    /// descriptor and nothing in the protocol retargets it. So a draw naming
    /// this reference's plane at *this* address with a different span is not
    /// that plane at all — the guest retired the object and its object-list slot
    /// now holds another one, without the `CmdDeleteResource` that would have
    /// said so.
    ///
    /// That makes the new span a **lifetime boundary**, and the entry is
    /// replaced with a fresh generation rather than kept. Keeping it is what a
    /// caller cannot do anything correct with: the old generation names an image
    /// holding the old object's pixels, and the new declaration's image must not
    /// be that one.
    ///
    /// [`Self::arm_gva`] likewise replaces a plane's prior debt. The caller owns
    /// releasing whatever the old generation held; see
    /// [`gva_resource_generation`].
    pub fn ensure_gva_resource(
        &mut self,
        key: GvaResourceKey,
        gva: u64,
        span: u64,
        pages: Option<Vec<u64>>,
    ) -> u64 {
        let plane = key.plane(gva);
        if let Some(resource) = self.gva_resources.get_mut(&plane) {
            if resource.span == span {
                if resource.pages.is_none() {
                    resource.pages = pages.map(std::sync::Arc::from);
                }
                return resource.generation;
            }
        }
        self.next_gva_generation = self.next_gva_generation.wrapping_add(1);
        if self.next_gva_generation == 0 {
            self.next_gva_generation = 1;
        }
        let generation = self.next_gva_generation;
        self.gva_resources.insert(
            plane,
            GvaResourceState {
                generation,
                span,
                pages: pages.map(std::sync::Arc::from),
            },
        );
        generation
    }

    /// Give a live plane back the transfer backing an explicit discard took,
    /// without touching its declaration or its generation.
    ///
    /// This is what the payment path needs and all it may have. Payment names a
    /// plane it did not declare — the declaration it holds is the debt's,
    /// recorded when the frame was armed — so letting it reach
    /// [`Self::ensure_gva_resource`] gives a stale debt the power to resurrect a
    /// retired plane or to re-declare a live one out from under the draw that
    /// owns it. Asking here instead makes that unrepresentable: absent the
    /// plane, there is nothing to reback and nothing is created.
    ///
    /// `pages` is adopted only into a plane that has none, exactly as on the
    /// establishing path.
    #[cfg(feature = "backend-vulkan")]
    fn reback_gva_resource(&mut self, plane: GvaPlaneKey, pages: Option<Vec<u64>>) -> bool {
        let Some(resource) = self.gva_resources.get_mut(&plane) else {
            return false;
        };
        if resource.pages.is_none() {
            resource.pages = pages.map(std::sync::Arc::from);
        }
        true
    }

    #[cfg(any(feature = "backend-vulkan", test))]
    fn gva_resource_backing(
        &self,
        plane: GvaPlaneKey,
    ) -> Option<(u64, u64, std::sync::Arc<[u64]>)> {
        let resource = self.gva_resources.get(&plane)?;
        Some((
            resource.generation,
            resource.span,
            std::sync::Arc::clone(resource.pages.as_ref()?),
        ))
    }

    /// Gated on the arm that calls it. Unlike [`Self::gva_resource_backing`],
    /// which the tests in this module exercise directly, the only reader of
    /// this one is [`gva_resource_generation`], which is Vulkan-only — so
    #[cfg(feature = "backend-vulkan")]
    fn gva_resource_status(&self, plane: GvaPlaneKey) -> Option<(u64, u64, bool)> {
        self.gva_resources
            .get(&plane)
            .map(|resource| (resource.generation, resource.span, resource.pages.is_some()))
    }

    /// Release the transfer buffer of each named resource while preserving its
    /// host texture and lifetime identity.
    ///
    /// Every plane of a named resource, because the guest's discard names the
    /// resource and a resource holds all of its levels' backings.
    pub fn discard_gva_resources(&mut self, task_id: u32, object_ids: &[u32]) -> usize {
        let mut discarded = 0;
        for &texture_ref in object_ids {
            let key = GvaResourceKey {
                task_id,
                texture_ref,
            };
            for resource in self.gva_resources.range_mut(key.planes()) {
                discarded += usize::from(resource.1.pages.take().is_some());
            }
        }
        discarded
    }

    /// Every plane of one resource goes at once: `CmdDeleteResource` names the
    /// resource, and a level that outlived its allocation names nothing.
    fn retire_gva_resource(&mut self, key: GvaResourceKey) -> (bool, Vec<GvaWritebackDebt>) {
        let planes: Vec<GvaPlaneKey> = self
            .gva_resources
            .range(key.planes())
            .map(|(plane, _)| *plane)
            .chain(self.gva_debts.range(key.planes()).map(|(plane, _)| *plane))
            .collect();
        let mut existed = false;
        let mut debts = Vec::new();
        for plane in planes {
            existed |= self.gva_resources.remove(&plane).is_some();
            debts.extend(self.gva_debts.remove(&plane));
        }
        (existed, debts)
    }

    /// The one plane debt this resource owes, or `None` when it owes zero or
    /// several.
    ///
    /// The caller — `blit_exec`'s whole-plane GPU copy — names a resource and
    /// holds no address, so with several planes owed it cannot say which one its
    /// source level is. Declining costs it the GPU shortcut and nothing else:
    /// that path's own doc records that a fall-through spends a frame and cannot
    /// lose one.
    pub fn get_gva(&self, key: GvaResourceKey) -> Option<GvaWritebackDebt> {
        let mut owed = self.gva_debts.range(key.planes());
        let (_, only) = owed.next()?;
        match owed.next() {
            None => Some(*only),
            Some(_) => {
                crate::runtime::drain::note_store_route("gvadebt_resource_owes_many_planes");
                None
            }
        }
    }

    pub fn has_gva(&self, key: GvaResourceKey) -> bool {
        self.gva_debts.range(key.planes()).next().is_some()
    }

    /// Every plane debt this resource owes, taken out of the ledger.
    ///
    /// The resource is the unit the guest synchronizes and the unit a sampled
    /// read names, so a payment for it owes every level's frame and not the one
    /// that happened to sort first.
    pub fn take_gva(&mut self, key: GvaResourceKey) -> Vec<(GvaPlaneKey, GvaWritebackDebt)> {
        let planes: Vec<GvaPlaneKey> = self
            .gva_debts
            .range(key.planes())
            .map(|(plane, _)| *plane)
            .collect();
        planes
            .into_iter()
            .filter_map(|plane| self.gva_debts.remove(&plane).map(|debt| (plane, debt)))
            .collect()
    }

    fn take_gva_plane(&mut self, plane: GvaPlaneKey) -> Option<GvaWritebackDebt> {
        self.gva_debts.remove(&plane)
    }

    /// Put back a debt whose guest backing was temporarily unavailable.
    /// Preserves its original age: inability to pay does not make an old frame
    /// the newest member of the ledger.
    #[cfg(feature = "backend-vulkan")]
    fn restore_gva(&mut self, plane: GvaPlaneKey, debt: GvaWritebackDebt) {
        let previous = self.gva_debts.insert(plane, debt);
        debug_assert!(
            previous.is_none(),
            "a taken debt restores into its own hole"
        );
    }

    fn gvas_by_age(&self) -> Vec<GvaPlaneKey> {
        let mut all: Vec<(u64, GvaPlaneKey)> = self
            .gva_debts
            .iter()
            .map(|(key, debt)| (debt.seq, *key))
            .collect();
        all.sort_unstable();
        all.into_iter().map(|(_, key)| key).collect()
    }

    /// Distinct resources of one task, deduped across their planes: task
    /// teardown retires resources, and [`Self::retire_gva_resource`] already
    /// takes every plane of the one it is given.
    fn gvas_for_task(&self, task_id: u32) -> Vec<GvaResourceKey> {
        let mut all: Vec<GvaResourceKey> = self
            .gva_resources
            .keys()
            .map(|plane| plane.resource)
            .filter(|key| key.task_id == task_id)
            .collect();
        all.dedup();
        all
    }

    #[cfg(feature = "backend-vulkan")]
    fn gva_for_identity(
        &self,
        identity: &crate::backend::vulkan::engine::TargetIdentity,
    ) -> Option<(GvaPlaneKey, GvaWritebackDebt)> {
        let crate::backend::vulkan::engine::TargetIdentity::Gva {
            gva,
            width,
            height,
            generation,
            ..
        } = *identity
        else {
            return None;
        };
        self.gva_debts
            .iter()
            .find(|(_, debt)| {
                debt.linear.target_gva() == gva
                    && debt.width == width
                    && debt.height == height
                    && debt.generation == generation
            })
            .map(|(key, debt)| (*key, *debt))
    }
}

/// Pay every owed GVA resource frame.
pub fn pay_all<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    for plane in state.pending_writebacks.gvas_by_age() {
        let Some(debt) = state.pending_writebacks.take_gva_plane(plane) else {
            continue;
        };
        let _ = pay_gva(state, host, plane, debt, GvaPaySite::All);
    }
}

/// Pay every plane owed by one task-local GVA resource.
pub fn pay_for_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let gva_key = GvaResourceKey {
        task_id,
        texture_ref,
    };
    // Every plane the reference owes, not the one that sorts first: a sampled
    // read names the resource, and a mip pyramid's levels are separate debts.
    for (plane, debt) in state.pending_writebacks.take_gva(gva_key) {
        let _ = pay_gva(state, host, plane, debt, GvaPaySite::Named);
    }
}

/// The stable host-texture identity for the GVA resource a draw is declaring.
///
/// The first successful resolution retains the ordered physical pages that the
/// resource's transfer buffer names. Later calls return the same generation and
/// backing even if the task removes its virtual mapping. After explicit
/// discard, the next call may establish a replacement transfer backing while
/// preserving the host texture's generation.
///
/// # A changed declaration ends one lifetime and begins the next
///
/// This used to answer `0` and emit `gva_resource_refused
/// reason=declaration_changed` when the draw's `(gva, span)` differed from the
/// one the entry was established with, on the reading that a live resource
/// cannot move. The reading is right and the response was not: the resource did
/// not move, the *reference* was reused, and the entry describing the retired
/// object is the thing that has to go.
///
/// Answering `0` never recovered. The entry stayed, so every later draw into
/// that reference compared against the same dead declaration and refused again —
/// one macos-26 report carried 5 197 of these lines over 280 references, one of
/// them refused 803 times in a single boot. What `0` costs depends on which
/// caller asked: `draw::vulkan`'s resident resolve turns it into
/// `GvaResidentRefusal::NoGeneration` and loses the frame, while the secondary
/// MRT builder puts it straight into [`TargetIdentity::Gva`], where generation
/// zero is the one value that cannot distinguish two allocations — the
/// wrong-content class that identity exists to close.
///
/// So a differing declaration is handled as what it is, a lifetime boundary,
/// through the same [`retire_gva_resource`] that `CmdDeleteResource` uses: the
/// old generation's unpaid frame is released rather than written into storage
/// the retired object no longer owns — the rule [`retire_gva_for_task`] already
/// states for task teardown — and [`PendingWritebacks::ensure_gva_resource`]
/// then establishes the new object's own generation.
///
/// It stays fail-visible, because a *frequent* redeclaration would say something
/// different: that some producer in this device describes one live resource two
/// ways, in which case each draw would mint a generation and no resident could
/// ever be reused. The line names both declarations so that reading can be made
/// from a log rather than from a rebuild.
///
/// [`TargetIdentity::Gva`]: crate::backend::vulkan::engine::TargetIdentity::Gva
#[cfg(feature = "backend-vulkan")]
pub fn gva_resource_generation<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    key: GvaResourceKey,
    gva: u64,
    span: u64,
) -> u64 {
    if let Some((generation, declared_span, has_pages)) =
        state.pending_writebacks.gva_resource_status(key.plane(gva))
    {
        if declared_span == span {
            if has_pages {
                return generation;
            }
        } else {
            crate::observe::Emit::decline(
                "gva_resource_redeclared",
                &GvaResourceRedeclared {
                    gva,
                    was_span: declared_span,
                    now_span: span,
                },
            )
            .field("task", key.task_id)
            .field("texture", key.texture_ref)
            .fail();
            retire_gva_resource(state, key.task_id, key.texture_ref);
        }
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        key.task_id,
        gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state
        .pending_writebacks
        .ensure_gva_resource(key, gva, span, pages)
}

/// One plane of a reference observed at two different lengths.
///
/// A *different address* under one reference is not this: that is another plane
/// of the same resource — a mip level — and [`GvaPlaneKey`] gives it its own
/// entry. What remains here is one address whose length moved, which the
/// contract has no room for, so the reference has been reused for a second
/// object.
///
/// Carries both lengths because neither alone says anything: the question a
/// reader has is whether they are *stable* — a reference reused, ordinary guest
/// lifetime — or whether they alternate, which would be this device describing
/// one plane two ways.
#[cfg(feature = "backend-vulkan")]
struct GvaResourceRedeclared {
    gva: u64,
    was_span: u64,
    now_span: u64,
}

#[cfg(feature = "backend-vulkan")]
impl crate::observe::Decline for GvaResourceRedeclared {
    fn slug(&self) -> &'static str {
        "gva_resource_declaration_changed"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("gva", format!("{:#x}", self.gva)),
            ("was_span", self.was_span.to_string()),
            ("now_span", self.now_span.to_string()),
        ]
    }
}

#[cfg(feature = "backend-vulkan")]
crate::observe::decline_display!(GvaResourceRedeclared);

/// Re-establish the transfer backing of the plane a debt names, without any
/// power to declare one.
///
/// The payment path's counterpart to [`gva_resource_generation`]. It asks only
/// the question payment has standing to ask — "does this plane still exist, and
/// does it still have its pages" — using the plane's *own* span, never the
/// debt's. A debt that outlived its plane therefore finds nothing here and is
/// released by the caller, where before it reached
/// [`PendingWritebacks::ensure_gva_resource`] and could re-create the retired
/// object at the dead declaration it was carrying.
#[cfg(feature = "backend-vulkan")]
fn reback_gva_resource<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    plane: GvaPlaneKey,
) -> bool {
    let Some((_, span, has_pages)) = state.pending_writebacks.gva_resource_status(plane) else {
        return false;
    };
    if has_pages {
        return true;
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        plane.resource.task_id,
        plane.gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(plane.gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state.pending_writebacks.reback_gva_resource(plane, pages)
}

/// Record a GVA render result as host-authoritative without touching guest
/// pages. Returns `false` when the attachment has no resource identity and must
/// use the eager transfer path.
#[cfg(feature = "backend-vulkan")]
pub fn arm_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    task_id: u32,
    c0: &crate::runtime::draw::ColorRtRequest,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let Some(generation) = (match *identity {
        crate::backend::vulkan::engine::TargetIdentity::Gva { generation, .. } => Some(generation),
        _ => None,
    }) else {
        return false;
    };
    if c0.texture_ref == 0 || generation == 0 {
        return false;
    }
    // Every older host-side spelling of this resource is stale as soon as the
    // render finishes. In particular, a compute storage resident and the
    // linear byte cache can otherwise sit above the guest-page reader and serve
    // the frame that preceded this Store indefinitely.
    state.invalidate_object_host_copies(task_id, c0.texture_ref);
    crate::runtime::surface_cache::evict_gva(state, c0.target_gva());
    let key = GvaResourceKey {
        task_id,
        texture_ref: c0.texture_ref,
    };
    let Some(linear) = c0.linear_target().copied() else {
        return false;
    };
    let debt = GvaWritebackDebt {
        linear,
        width: c0.width,
        height: c0.height,
        format: c0.format,
        generation,
        content: state.active_submission.as_ref().and_then(|submission| {
            state.task_resources.record_completed_gpu_store(
                task_id,
                c0.texture_ref,
                submission.identity.id,
            )
        }),
        guest_write: state.resource_write_stamp(task_id, c0.texture_ref),
        seq: 0,
    };
    let previous = state.pending_writebacks.arm_gva(key, debt);
    if let Some(previous) = previous.filter(|previous| !same_gva_identity(*previous, debt)) {
        release_gva(previous);
    }
    true
}

/// Whether this exact GVA resident is the host-authoritative copy named by an
/// unpaid resource debt.
#[cfg(feature = "backend-vulkan")]
pub fn gva_resident_authoritative(
    state: &DeviceState,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let Some((plane, debt)) = state.pending_writebacks.gva_for_identity(identity) else {
        return false;
    };
    state
        .resource_write_stamp(plane.resource.task_id, plane.resource.texture_ref)
        .quiet_since(debt.guest_write)
}

/// Retire host-authoritative resources whose task-local references are about to
/// be replaced. The pixels are deliberately not copied: after this lifecycle
/// transition the old object no longer names guest storage to synchronize.
pub fn retire_gva_for_task(state: &mut DeviceState, task_id: u32) -> usize {
    let keys = state.pending_writebacks.gvas_for_task(task_id);
    let mut retired = 0;
    for key in keys {
        let (_, debts) = state.pending_writebacks.retire_gva_resource(key);
        retired += 1;
        #[cfg(feature = "backend-vulkan")]
        for debt in debts {
            release_gva(debt);
        }
        #[cfg(not(feature = "backend-vulkan"))]
        let _ = debts;
    }
    if retired != 0 {
        crate::runtime::drain::note_store_route_n("gvadebt_retired_task", retired as u64);
    }
    retired
}

/// Retire one resource at its explicit lifetime boundary.
pub fn retire_gva_resource(state: &mut DeviceState, task_id: u32, texture_ref: u32) -> bool {
    let key = GvaResourceKey {
        task_id,
        texture_ref,
    };
    let (existed, debts) = state.pending_writebacks.retire_gva_resource(key);
    let owed = !debts.is_empty();
    #[cfg(feature = "backend-vulkan")]
    for debt in debts {
        release_gva(debt);
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = debts;
    existed || owed
}

/// Release named resources' retained transfer backings.
pub fn discard_gva_resources(state: &mut DeviceState, task_id: u32, object_ids: &[u32]) -> usize {
    state
        .pending_writebacks
        .discard_gva_resources(task_id, object_ids)
}

#[cfg(feature = "backend-vulkan")]
fn same_gva_identity(a: GvaWritebackDebt, b: GvaWritebackDebt) -> bool {
    a.linear.target_gva() == b.linear.target_gva()
        && a.width == b.width
        && a.height == b.height
        && a.generation == b.generation
        && a.format == b.format
}

/// The engine resident one armed GVA debt names.
///
/// `pub(crate)` because a debt is not only something to pay: a reader that wants
/// the *content* rather than the guest's copy of it — the blit rail's whole-plane
/// GPU arm — needs exactly this identity, and deriving a second one from the same
/// debt fields is how two spellings of one resident start disagreeing. There is
/// one derivation and it is here.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn gva_identity(
    debt: GvaWritebackDebt,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: debt.linear.target_gva(),
        width: debt.width,
        height: debt.height,
        generation: debt.generation,
        format: crate::runtime::draw::gva_resident_format(debt.format),
    }
}

#[cfg(feature = "backend-vulkan")]
fn release_gva(debt: GvaWritebackDebt) {
    crate::backend::vulkan::engine::note_resident_content_copied_out(&gva_identity(debt));
}

/// Wait only for submitted writes that can reach one mapping's pages.
///
/// The page set comes from [`DeviceState::mapping_reach_pages`], the same rule
/// the write path uses for its destination. A mapping that cannot name its pages
/// answers `None`, which conservatively waits.
pub fn settle_for_mapping(
    state: &mut DeviceState,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    let reach_started = std::time::Instant::now();
    let s = &*state;
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(site, || {
        crate::runtime::drain::note_store_route("wbdebt_reach_walk_n");
        s.mapping_reach_pages(mapping_id)
    });
    crate::runtime::drain::note_store_route_us(
        "wbdebt_reach_us",
        reach_started.elapsed().as_micros() as u64,
    );
}

/// Materialize one named GVA resource, then wait for submitted writes that can
/// reach the task-GVA span a CPU reader is about to access.
pub fn settle_for_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    span: u64,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_for_texture(state, host, task_id, texture_ref);
    let (tasks, page_shift, page_size) = (&state.tasks, state.page_shift, state.page_size());
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(site, || {
        let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
        let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
            host, tasks, task_id, gva, span, page_shift,
        );
        (gpas.len() as u64 == want).then_some(gpas)
    });
}

/// [`settle_for_mapping`] for a caller that cannot name the mapping it is about
/// to touch, so it owes every debt.
pub fn settle_unnamed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_all(state, host);
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// Submit exactly the resources named by an asynchronous synchronize command.
///
/// The object list is the scope of the API operation; an unrelated host-valid
/// texture remains resident-authoritative. Completion belongs to the FIFO: the
/// transfers recorded here precede that packet's queue point, and its pending
/// stamp publishes only after that point completes. Waiting here would turn the
/// asynchronous command into a device-wide drain and then make the stamp wait a
/// second time for work already known complete.
pub fn submit_for_resources<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    object_ids: &[u32],
) {
    for &object_id in object_ids {
        pay_for_texture(state, host, task_id, object_id);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GvaPaySite {
    Named,
    All,
}

#[cfg(feature = "backend-vulkan")]
impl GvaPaySite {
    fn route(self) -> &'static str {
        match self {
            Self::Named => "gvadebt_paid_named",
            Self::All => "gvadebt_paid_all",
        }
    }
}

/// Materialize one host-authoritative GVA resource into its retained transfer
/// backing. After explicit discard, synchronize lazily recreates that backing;
/// ordinary virtual-memory unmap does not participate in resource lifetime.
#[cfg(feature = "backend-vulkan")]
fn pay_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plane: GvaPlaneKey,
    debt: GvaWritebackDebt,
    site: GvaPaySite,
) -> bool {
    let key = plane.resource;
    let identity = gva_identity(debt);
    let now = state.resource_write_stamp(key.task_id, key.texture_ref);
    if !now.quiet_since(debt.guest_write) {
        crate::runtime::drain::note_store_route("gvadebt_abandoned_guest_wrote");
        release_gva(debt);
        return true;
    }
    let Some(span) = u64::from(debt.linear.row_stride).checked_mul(u64::from(debt.height)) else {
        crate::observe::fail(format!(
            "gvadebt_pay_lost task={} texture={} reason=span_overflow",
            key.task_id, key.texture_ref
        ));
        release_gva(debt);
        return true;
    };
    // The resource's own declaration decides whether its pages come back, not
    // this debt's — see [`reback_gva_resource`]. A debt whose resource is gone
    // names storage that object no longer owns, so it is released here rather
    // than restored: restoring one would park it in the ledger forever, since
    // nothing retired can grow pages back.
    if !reback_gva_resource(state, host, plane) {
        crate::runtime::drain::note_store_route("gvadebt_resource_retired");
        release_gva(debt);
        return true;
    }
    let Some((backing_generation, backing_span, ordered)) =
        state.pending_writebacks.gva_resource_backing(plane)
    else {
        state.pending_writebacks.restore_gva(plane, debt);
        crate::runtime::drain::note_store_route(match site {
            GvaPaySite::Named => "gvadebt_named_unmapped",
            GvaPaySite::All => "gvadebt_all_unmapped",
        });
        if site == GvaPaySite::Named {
            crate::observe::fail(format!(
                "gvadebt_pay_blocked task={} texture={} reason=span_unresolved",
                key.task_id, key.texture_ref
            ));
        }
        return false;
    };
    // The plane key already carries the address, so a mismatched one cannot
    // reach here: it would have found no plane at all above.
    if backing_generation != debt.generation || backing_span != span {
        crate::runtime::drain::note_store_route("gvadebt_generation_moved");
        release_gva(debt);
        return true;
    }
    let pages = crate::runtime::draw::StoreTargetPages::from_ordered(&ordered, span);
    let request = crate::runtime::draw::ColorRtRequest {
        texture_ref: key.texture_ref,
        storage: crate::runtime::draw::ColorTargetStorage::Linear(debt.linear),
        width: debt.width,
        height: debt.height,
        format: debt.format,
        store_action: crate::contract::pass_action::MTL_STORE_ACTION_STORE,
        ..Default::default()
    };
    crate::runtime::drain::note_store_route(site.route());
    if let Err(reason) = crate::runtime::render_writeback::store_gva_frame(
        state,
        host,
        key.task_id,
        &identity,
        &request,
        key.texture_ref,
        Some(&pages),
    ) {
        // Through the builder rather than by interpolating the decline, which
        // renders its own `reason=` and produced `reason=reason=<slug>` — a line
        // the standard ranking grep drops. The builder also carries the
        // decline's own fields, so the `via=` that says which check inside the
        // store refused now reaches the log instead of being formatted away.
        crate::observe::Emit::decline("gvadebt_pay_lost", &reason)
            .field("task", key.task_id)
            .field("texture", key.texture_ref)
            .fail();
        release_gva(debt);
    } else if let Some((resource, version)) = debt.content {
        if !state
            .task_resources
            .record_gpu_to_guest_copy(resource, version)
        {
            crate::observe::fail(format!(
                "gvadebt_content_transition task={} texture={} reason=stale_content_version",
                key.task_id, key.texture_ref
            ));
        }
    }
    true
}

#[cfg(not(feature = "backend-vulkan"))]
fn pay_gva<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _plane: GvaPlaneKey,
    _debt: GvaWritebackDebt,
    _site: GvaPaySite,
) -> bool {
    true
}

#[cfg(test)]
mod tests {

    use super::*;

    fn gva_debt(generation: u64) -> GvaWritebackDebt {
        GvaWritebackDebt {
            linear: crate::runtime::draw::LinearColorTarget {
                allocation_gva: 0x4000,
                allocation_size: 64 * 256,
                plane_offset: 0,
                row_stride: 256,
            },
            width: 64,
            height: 64,
            format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            generation,
            content: None,
            guest_write: Default::default(),
            seq: 0,
        }
    }

    /// The resource reference, not the GVA, owns coherence. Reusing the same
    /// resource for another Store replaces its debt exactly as repeated Stores
    /// into one IOSurface do.
    #[test]
    fn a_second_gva_store_on_one_resource_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        assert_eq!(pending.arm_gva(key, gva_debt(7)), None);
        let previous = pending.arm_gva(key, gva_debt(8));
        assert_eq!(previous.map(|debt| debt.generation), Some(7));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.get_gva(key).map(|debt| debt.generation), Some(8));
    }

    /// GVA resources have protocol lifetime, not an arbitrary ledger capacity.
    #[test]
    fn gva_debts_are_not_evicted_by_capacity() {
        let mut pending = PendingWritebacks::default();
        const DISTINCT_RESOURCES: u32 = 64;
        for texture_ref in 1..=DISTINCT_RESOURCES {
            let key = GvaResourceKey {
                task_id: 2,
                texture_ref,
            };
            pending.ensure_gva_resource(
                key,
                u64::from(texture_ref) << 16,
                4096,
                Some(vec![u64::from(texture_ref) << 12]),
            );
            assert_eq!(pending.arm_gva(key, gva_debt(texture_ref.into())), None);
        }
        assert_eq!(pending.len(), DISTINCT_RESOURCES as usize);
        assert_eq!(pending.gvas_by_age().len(), DISTINCT_RESOURCES as usize);
    }

    /// Ordinary virtual-memory bookkeeping does not retarget a live resource.
    /// A repeated prepare with a different walk keeps the original transfer
    /// backing until the protocol explicitly discards it.
    #[test]
    fn a_live_resource_retains_its_backing_until_discard() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let generation = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation
        );
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0x9000]
        );

        assert_eq!(pending.discard_gva_resources(3, &[19]), 1);
        assert!(pending.gva_resource_backing(key.plane(0x4000)).is_none());
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation,
            "discard replaces the transfer backing, not the host texture"
        );
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0xa000]
        );
    }

    /// Delete is the resource lifetime boundary. Reusing the same task-local
    /// reference after delete receives a new host-texture identity.
    #[test]
    fn deleting_and_recreating_a_resource_changes_its_generation() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert!(pending.retire_gva_resource(key).0);
        let second = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000]));
        assert_ne!(first, second);
    }

    /// Delete is the *announced* lifetime boundary; a plane's length moving at
    /// one address is the same boundary observed instead of announced. A
    /// plane's length is fixed for its life, so this is a different object in a
    /// reused slot and it must get a different host texture.
    ///
    /// Asserting the third call is what makes that visible: a fix that only
    /// stopped refusing, without replacing the entry, still fails here.
    #[test]
    fn one_plane_redeclared_at_a_new_length_is_a_new_resource() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        let second = pending.ensure_gva_resource(key, 0x4000, 8192, Some(vec![0xa000, 0xb000]));
        assert_ne!(first, second, "a new length is a new host texture");
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0xa000, 0xb000],
            "the new object's pages replace the retired one's"
        );
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 8192, None),
            second,
            "the new declaration is the live one, so it is stable"
        );
    }

    /// A mip pyramid is one resource with several live planes, and the ledger
    /// has to hold all of them at once.
    ///
    /// Measured on a driven macos-26 boot: one reference cycling three
    /// contiguous declarations in exact 4:1 ratios — 256x192, 128x96, 64x48 of
    /// one RGBA8 allocation, the compositor's blur/backdrop pyramid. Keyed by
    /// the reference, each level change replaced the entry, so no level's
    /// resident could ever be reused and arming one level's Store dropped the
    /// previous level's unpaid frame. Both halves are asserted here: the
    /// generations are distinct **and** stable, and three debts coexist.
    #[test]
    fn the_levels_of_one_pyramid_are_separate_planes_of_one_resource() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 1,
            texture_ref: 135,
        };
        let levels = [
            (0x11af000_u64, 196_608_u64),
            (0x11df000, 49_152),
            (0x11eb000, 12_288),
        ];
        let generations: Vec<u64> = levels
            .iter()
            .map(|&(gva, span)| pending.ensure_gva_resource(key, gva, span, Some(vec![gva])))
            .collect();
        let mut distinct = generations.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3, "each level is its own host texture");

        // The cycle the boot showed: re-declaring level 0 after levels 1 and 2
        // must return level 0's own generation, not mint a fourth.
        for (i, &(gva, span)) in levels.iter().enumerate() {
            assert_eq!(
                pending.ensure_gva_resource(key, gva, span, None),
                generations[i],
                "a live plane is stable across its siblings"
            );
        }

        for (i, &(gva, _)) in levels.iter().enumerate() {
            let mut debt = gva_debt(generations[i]);
            debt.linear.allocation_gva = gva;
            assert_eq!(
                pending.arm_gva(key, debt),
                None,
                "arming one level must not supersede another"
            );
        }
        assert_eq!(
            pending.take_gva(key).len(),
            3,
            "the resource owes all three"
        );
    }

    /// A guest validity transition after the Store makes guest memory newer
    /// than the held resident. The debt remains available for an orderly
    /// abandon, but it must immediately stop licensing host-resident reads.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_guest_write_revokes_gva_resident_authority() {
        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let key = GvaResourceKey {
            task_id: 4,
            texture_ref: 12,
        };
        let debt = gva_debt(99);
        let _ = state.pending_writebacks.arm_gva(key, debt);
        let identity = gva_identity(debt);
        assert!(gva_resident_authoritative(&state, &identity));
        state
            .buffer_write_gen
            .note_write(key.task_id, key.texture_ref);
        assert!(!gva_resident_authoritative(&state, &identity));
        assert!(state.pending_writebacks.get_gva(key).is_some());
    }
}
