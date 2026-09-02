//! From a packet this device drained to a packet the semantic model admits.
//!
//! # What this is for
//!
//! [`crate::runtime::drain`] takes bytes out of a ring and produces a
//! [`drain::Packet`]: an opcode, a record array, a completion word and a
//! payload. [`reims_vgpu_core::session::SessionModel`] admits a
//! [`reims_vgpu_core::session::Packet`]: an ordering domain, a semantic
//! lifetime, typed stamps and a resolved [`Payload`]. Those are the two ends of
//! the cutover and **nothing joined them** — the model could describe every
//! packet this device receives and could be handed none of them.
//!
//! This is that join. It touches no guest memory and mutates nothing;
//! everything it cannot answer it returns as a named [`Gap`] rather than
//! approximating, so the set of classes that can cross today is a value a test
//! can assert rather than a claim a reader has to believe.
//!
//! It is **not** a pure function of the drained packet, and stopped being one
//! the moment the first namespace gap closed. A ref, a mapping id and a page
//! frame are numbers whose meaning lives in a namespace, and a bridge that
//! could not consult one could only ever carry the classes that name nothing —
//! which is what [`Gap`] was a list of. So the namespaces arrive as
//! [`Resolvers`]: a bundle rather than a parameter each, because every
//! remaining gap closes by *adding a resolver*, and a bundle makes that a field
//! rather than another signature change rippling through every caller.
//!
//! # The gaps are the cutover's remaining work, stated
//!
//! What crosses is what resolves from the packet's own bytes.
//! [`reims_vgpu_core::control::resolve`] is such a function, so the whole
//! control class needs nothing this device has not already put in the
//! `drain::Packet` — and so is
//! [`reims_vgpu_core::lifecycle::task_lifetime`], which is why two members of
//! the resource-lifetime class cross with it.
//!
//! **The class is not the unit, and that is the point of naming the gaps at
//! all.** The resource-lifetime class has been three different partitions in
//! three commits, and each time the coarser name was hiding a member that did
//! not belong to it. "Lifetime commands need a namespace" was false of the two
//! that name a task and nothing inside it. Once the namespaces landed it was
//! false of all twelve — what the remaining ones are short of is the *access*
//! each named resource takes, which is a different missing thing with a
//! different owner, and two of them are short of neither. A gap that
//! over-claims is one nobody thinks to close; a gap that closes and leaves its
//! name behind is one everybody thinks is closed.
//!
//! What remains, each naming one input the model needs and this function is not
//! given:
//!
//! - **Exec** needs the object-list resolver and the access source that
//!   [`reims_vgpu_core::walk::exec`] walks a command stream with.
//! - **Resource lifetime** is down to two. All twelve resolve their refs, and
//!   ten of them now build a payload: five name no resource, and five name
//!   resources and state what they do to them —
//!   `reims_vgpu_core::lifecycle::LifecycleOp::declared_access` is the mode and
//!   [`Resolvers::storage`] is the key. The last two are short of something
//!   that is not in their packet at all: the guest's object-list table, and the
//!   pages behind a re-pointed object.
//! - **Query** needs its reply destination *resolved* — a backing and a window,
//!   not the address the request names.
//!
//! Present is no longer among them. Its target is a mapping, the device answers
//! `reims_vgpu_core::resolve::MappingResolver` over that namespace, and the
//! frame it shows is the whole of that mapping's surface — so the access is an
//! `AccessKey::Whole` over the resolved backing and nothing about it is
//! approximated. See [`present_payload`] for the one case that is still
//! imprecise and why the imprecision is the honest answer there.
//!
//! None of the remaining three is missing decode work: this device resolves
//! them all today. What it does not have is a *generation-stamped* namespace to
//! resolve them into, which is the model's, and giving this function a
//! half-resolved answer would be the adapter between two semantic models that
//! the replacement plan forbids. So they are gaps, they are named, and they
//! close when their owner lands — not here.
//!
//! # Every packet carries a completion word
//!
//! [`reims_vgpu_core::session::Packet::completion`] is an `Option`, because a
//! model may have packets that publish nothing. **This interface does not**:
//! the drain writes the header's completion word into the channel's stamp slot
//! for every packet it processes, and a packet that does not advance the fence
//! repeats the slot's current value rather than leaving it alone. Repeating is
//! idempotent and a wait decided against a repeat is decided the same way it
//! was before, so "does not signal" and "signals the value already there" are
//! the same event on the wire. This bridge therefore always produces `Some`,
//! and `None` stays available for a model that has the other case.
//!
//! # The slot is the channel's and the value is the packet's
//!
//! A `drain::Packet` carries a completion *value* and no slot. The slot belongs
//! to the FIFO — the root's is slot 0 and a child's is read once per drain from
//! its register block — so it arrives here as an argument. Wait records carry
//! their own raw index, masked to a slot by
//! [`stamp_slot_index`], which is the one place that mask is applied.

use crate::model::{is_child_channel, stamp_slot_index};
use crate::runtime::drain;
use reims_vgpu_core::control;
use reims_vgpu_core::identity::{
    ChannelId, CompletionStamp, SessionGeneration, StampSlot, StampValue, StampWait,
};
use reims_vgpu_core::resolve::{MappingResolver, ResourceStorage, TaskNamespaces};
use reims_vgpu_core::session::Packet;
use reims_vgpu_core::transaction::{classify, Payload, PayloadClass};
use reims_vgpu_protocol::packets::Channel;

/// Which FIFO a packet was drained from.
///
/// **One value, because the two questions have one answer.**
/// [`reims_vgpu_core::session::Packet`] carries both a [`Channel`] — which
/// dispatch table the opcode is read against — and a [`ChannelId`] — which
/// ordering domain the packet joins. They are not independent: this device
/// numbers the root FIFO 0 and its children 1..[`MAX_CHANNELS`], which is the
/// rule [`is_child_channel`] states and the rule
/// [`reims_vgpu_core::control::ControlOp::Channel`] hands back a domain under.
/// A bridge taking the pair separately could be given `Channel::Root` with a
/// child's domain, and the packet would then be classified against the root's
/// opcode table and ordered on a child's channel. Deriving the channel from the
/// id makes that unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fifo(ChannelId);

impl Fifo {
    /// The device's own channel, numbered 0.
    pub const ROOT: Fifo = Fifo(ChannelId(0));

    /// A guest-defined channel, or `None` when the id names no FIFO this
    /// device has. The bound is [`is_child_channel`]'s and not restated here.
    #[must_use]
    pub fn child(channel_id: u32) -> Option<Fifo> {
        is_child_channel(channel_id).then_some(Fifo(ChannelId(channel_id)))
    }

    /// The ordering domain packets on this FIFO join.
    #[must_use]
    pub const fn domain(self) -> ChannelId {
        self.0
    }

    /// Which opcode table this FIFO's packets are read against.
    #[must_use]
    pub const fn channel(self) -> Channel {
        if self.0 .0 == 0 {
            Channel::Root
        } else {
            Channel::Child
        }
    }
}

/// An input the model needs that this bridge is not given.
///
/// Not a refusal: the guest's packet is well formed and this device answers it
/// today. Each variant names the one thing missing, so a suite can assert the
/// partition — which classes cross and which do not — instead of a reader
/// having to infer it from what the code happens to handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    /// The ledger has not closed this row, so [`classify`] gives it no class
    /// and the model may not claim it. Production answers it alone.
    Unresolved,
    /// The object-list resolver and the access source an EXEC's command stream
    /// is walked with.
    ExecResolution,
    /// The guest's object-list table, walked, which is what a rebind's operation
    /// *is*.
    ///
    /// `SetObjectList` says where a task's list lives and how many entries it
    /// has; the operation is the per-entry result of reading it, and every byte
    /// of that lives in pages `crates/reims-vgpu-memory` owns. Named rather than
    /// approximated, because a rebind built from the packet alone would rebind
    /// nothing while reporting that it had — which is
    /// `reims_vgpu_core::lifecycle::ResolveRefusal::NeedsGuestTable`, restated
    /// here as this device's own incompleteness rather than as the guest's bytes
    /// being wrong.
    ///
    /// Walking it eagerly is not the closure. A driven boot's `SetObjectList`
    /// packets each declare **1 048 576** entries — the list's capacity, not its
    /// population. See `crate::runtime::objects::name_resource`.
    ObjectListTable,
    /// The pages behind a re-pointed object, which its packet does not carry.
    ///
    /// `ReplacePhysical` is a bare `{task, object}`: the guest re-points a
    /// resource at host frames it has already wired, at the same GPU-VA, so the
    /// new backing and extent are nowhere on the wire. The ref resolves — that
    /// half is done — and the operation still cannot be built, which is
    /// `reims_vgpu_core::lifecycle::ResolveRefusal::NeedsStorage`. An operation
    /// carrying the *old* backing would re-point nothing while reporting
    /// success.
    ReplacementStorage,
    /// The reply destination, resolved to a backing and a window of it.
    ///
    /// **The four questions do not share a destination space, and whoever
    /// closes this has to resolve two things and not one.** `CmdGetComputeInfo`
    /// and the heap-texture size query carry a `reply_gva` — an address in the
    /// task, in the same space every object-list object's window is in, so a
    /// reply buffer *can* fall inside an allocation this device already has an
    /// identity for and must then resolve to that identity rather than to one of
    /// its own. `CmdGetDeviceInfo` carries a `reply_pfn` instead: a guest page
    /// frame, in no task's address space at all.
    ///
    /// `reims_vgpu_core::query::ReplyDestination` wants one `BackingId` for
    /// both, which is right — a destination is storage like any other — but the
    /// device mints identities from `(task, allocation base, incarnation)` and a
    /// page frame has no task and no allocation base. That is the term this gap
    /// is actually short of.
    ///
    /// # Both halves are now instrumented, and neither is closed by a reading
    ///
    /// `crate::runtime::objects::note_query_reply_destination` asks the GVA half
    /// whether a reply buffer lies inside an allocation this device already
    /// identifies — the case where a destination given an identity of its own
    /// leaves the reply write unordered against every access to that object.
    ///
    /// `crate::runtime::objects::note_device_info_reply_destination` asks the
    /// page-frame half the only question that can be asked of it. A page frame
    /// cannot be compared against a table of task windows, so the instrument
    /// asks about the **identity space**: this device interns every key space
    /// on one monotone counter, so a freshly minted identity is sound exactly
    /// when that counter has handed nothing out, and no page arithmetic is
    /// needed to say so. The reply path documents the contract that would make
    /// it hold — the guest asks once, at accelerator start, and frees the
    /// buffer after the single parse — and whether "at accelerator start"
    /// precedes this device's first mint is a fact about a driven guest.
    ///
    /// A driven boot has already corrected one reading of that instrument. Its
    /// first form counted live tasks and resources and reported `tasks=1
    /// resources=0`, which reads as the open case and is not one: a task is a
    /// namespace and has interned nothing, and the number the dependency
    /// compiler compares is the mint counter. Neither denominator is large
    /// enough to conclude from yet, and that is recorded here so the next
    /// attempt spends its boot rather than its reasoning.
    ReplyDestination,
}

impl Gap {
    /// The name this reaches the failure channel under.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unresolved => "ingress_row_unresolved",
            Self::ExecResolution => "ingress_needs_exec_resolution",
            Self::ObjectListTable => "ingress_needs_object_list_table",
            Self::ReplacementStorage => "ingress_needs_replacement_storage",
            Self::ReplyDestination => "ingress_needs_reply_destination",
        }
    }
}

/// The guest's bytes not being the command its opcode names, as the resolver
/// that judged them said it.
///
/// Two vocabularies and not one string, because two resolvers judge two classes
/// here and they refuse different things — a control packet that is not control,
/// a lifetime payload too short for the record its opcode implies. Folded into
/// one name, a reading could not say which layer decided, and the two have
/// different fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    Control(control::ResolveRefusal),
    Lifecycle(reims_vgpu_core::lifecycle::ResolveRefusal),
    /// A present packet's bytes are not a present, or are too short for the
    /// trailer its form has.
    Present(reims_vgpu_core::present::ResolveRefusal),
    /// The accesses built for a lifetime command do not describe the resources
    /// its operation named.
    ///
    /// Unreachable by construction today — [`lifecycle_accesses`] walks
    /// `LifecycleOp::resources` itself and pairs each entry with exactly one
    /// access — and carried for [`Self::PresentAccesses`]'s reason: the
    /// constructor is what holds an operation and its accesses together, and a
    /// bridge that unwrapped it would be asserting a property it had stopped
    /// letting anything check.
    LifecycleAccesses(reims_vgpu_core::transaction::AccessMismatch),
    /// The accesses built for a present do not describe the frame its packet
    /// asked for.
    ///
    /// Unreachable by construction today — [`present_payload`] builds exactly
    /// one read of the packet's own target — and carried anyway, because the
    /// constructor is what holds the packet and its accesses together and a
    /// bridge that unwrapped it would be asserting a property it had stopped
    /// letting anything check.
    PresentAccesses(reims_vgpu_core::transaction::PresentMismatch),
}

impl Refused {
    /// The name this reaches the failure channel under, from the refusal's own
    /// owner rather than restated here.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::Control(inner) => inner.slug(),
            Self::Lifecycle(inner) => inner.slug(),
            Self::LifecycleAccesses(inner) => inner.slug(),
            Self::Present(inner) => inner.slug(),
            Self::PresentAccesses(inner) => inner.slug(),
        }
    }
}

/// The namespaces this bridge resolves the guest's numbers in.
///
/// **A bundle and not a parameter each.** Three of the model's inputs are still
/// gaps and every one of them closes by giving this bridge another namespace to
/// consult; carried as separate arguments, each closure would be a signature
/// change at every call site, and a call site would be free to pass the mapping
/// namespace where the object namespace was wanted. Carried as one value with
/// named fields, a closure adds a field and the compiler names every caller that
/// has to answer for it.
///
/// Borrowed rather than owned, and `dyn` rather than generic: the device is the
/// only implementor there will ever be, the call is once per packet against a
/// `BTreeMap` lookup, and a generic here would put a type parameter on
/// [`packet`] that every caller and every test would have to spell.
#[derive(Clone, Copy)]
pub struct Resolvers<'a> {
    /// Which backing a mapping's surface currently occupies.
    pub mappings: &'a dyn MappingResolver,
    /// Every task's object namespace, asked for by the task a packet names.
    ///
    /// Not one namespace: which task's slots a lifecycle packet's refs are
    /// indices into is stated inside that packet, so only the join that decoded
    /// it can bind the namespace. See `reims_vgpu_core::resolve::TaskNamespaces`.
    pub objects: &'a dyn TaskNamespaces,
    /// Where a named resource's bytes are, which is what an access is keyed on.
    ///
    /// Beside [`Self::objects`] rather than inside it: what a ref *names* and
    /// where the named thing *lives* are two questions with two holders, and
    /// `reims_vgpu_core::resolve::ResourceStorage`'s own doc says why folding
    /// them would make "no such slot" and "no storage term" one answer.
    pub storage: &'a dyn ResourceStorage,
}

/// Why a drained packet did not become a model packet.
///
/// The two arms are different obligations and are deliberately not one type. A
/// [`Self::Gap`] is this device's own incompleteness and closes when its owner
/// lands; a [`Self::Refused`] is the guest's bytes not being the command its
/// opcode names, and closes never.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocked {
    Gap {
        channel: Channel,
        opcode: u16,
        gap: Gap,
    },
    Refused(Refused),
}

impl Blocked {
    /// The name this reaches the failure channel under.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::Gap { gap, .. } => gap.slug(),
            Self::Refused(refusal) => refusal.slug(),
        }
    }

    /// The gap, for a caller deciding whether to fall back to the legacy path.
    #[must_use]
    pub const fn gap(self) -> Option<Gap> {
        match self {
            Self::Gap { gap, .. } => Some(gap),
            Self::Refused(_) => None,
        }
    }
}

/// Build the model packet one drained packet describes.
///
/// `session` is the semantic lifetime the packet was **read under**, which is
/// the reader's fact and not this function's — see
/// [`reims_vgpu_core::session::Packet::session`]. `completion_slot` is the
/// FIFO's stamp slot, already masked by whoever read it.
///
/// # Errors
///
/// [`Blocked::Gap`] for every class this bridge cannot yet build, and
/// [`Blocked::Refused`] when a control packet's bytes are not its command.
pub fn packet(
    fifo: Fifo,
    session: SessionGeneration,
    completion_slot: StampSlot,
    drained: &drain::Packet,
    resolvers: Resolvers<'_>,
) -> Result<Packet, Blocked> {
    let channel = fifo.channel();
    let opcode = drained.opcode;
    let blocked = |gap| Blocked::Gap {
        channel,
        opcode,
        gap,
    };

    // Exhaustive over `PayloadClass`, so a sixth class is a compile error here
    // rather than a packet that quietly reaches whichever arm came last. The
    // gap each class names is stated once, in this match, and nowhere else.
    let payload = match classify(channel, opcode) {
        None => return Err(blocked(Gap::Unresolved)),
        Some(PayloadClass::Exec) => return Err(blocked(Gap::ExecResolution)),
        Some(PayloadClass::ResourceLifecycle) => Payload::ResourceLifecycle(resource_lifetime(
            fifo,
            channel,
            opcode,
            &drained.payload,
            resolvers,
        )?),
        Some(PayloadClass::Query) => return Err(blocked(Gap::ReplyDestination)),
        Some(PayloadClass::Present) => Payload::Present(present_payload(
            fifo,
            channel,
            opcode,
            &drained.payload,
            resolvers,
        )?),
        Some(PayloadClass::Control) => Payload::Control(
            control::resolve(channel, opcode, &drained.payload)
                .map_err(|refusal| Blocked::Refused(Refused::Control(refusal)))?,
        ),
    };

    Ok(Packet {
        channel,
        domain: fifo.domain(),
        session,
        opcode,
        stamp_waits: drained
            .stamp_waits
            .iter()
            .map(|wait| StampWait {
                slot: StampSlot(stamp_slot_index(wait.index)),
                value: StampValue(wait.value),
            })
            .collect(),
        completion: Some(CompletionStamp {
            slot: completion_slot,
            value: StampValue(drained.completion_stamp),
        }),
        payload,
    })
}

/// What a present asks to show, and the one access that orders it.
///
/// # A present reads the whole of one surface
///
/// The packet names a mapping and a form; it names no byte range, no level and
/// no slice, because showing a frame is not a subresource operation — the
/// display reads the surface. So the access is `AccessKey::Whole` over the
/// backing the mapping resolves to, which is the precision ladder's rung 2 and
/// is *exact* here rather than approximate: there is no finer answer the packet
/// contains and none that would order differently, since anything writing any
/// part of that backing must be ordered before the frame is shown.
///
/// It is a read. `reims_vgpu_core::transaction::PresentPayload` enforces that,
/// and its doc says why: modelled as a writer, a present would reserve a
/// content version, beat the real writer's, and publish bytes nothing produced.
///
/// # The one imprecise case, and why it is not the `DomainOnly` this bridge
/// refused to use
///
/// A mapping the device has no live surface for resolves to no backing. There
/// is then nothing to name, and `AccessKey::DomainOnly` — "participation is
/// incomplete, ordering comes from the submission domain alone" — is the
/// vocabulary for precisely that. That is a different act from handing *every*
/// present a `DomainOnly`, which is what this gap was deliberately not closed
/// with: that would have under-ordered every frame on the device while reading,
/// from outside, as a gap that had closed. Here the imprecision is the target's
/// and it is counted, so a boot says how many frames were shown against a
/// mapping this device could not resolve rather than leaving it to inference.
///
/// # Errors
///
/// [`Blocked::Refused`] when the payload is not a present or is too short for
/// its own trailer.
fn present_payload(
    fifo: Fifo,
    channel: Channel,
    opcode: u16,
    payload: &[u8],
    resolvers: Resolvers<'_>,
) -> Result<reims_vgpu_core::transaction::PresentPayload, Blocked> {
    use reims_vgpu_core::access::{AccessIntent, AccessKey, AccessMode, ResourceKey};
    use reims_vgpu_core::transaction::PresentPayload;

    let packet = reims_vgpu_core::present::resolve(channel, opcode, payload)
        .map_err(|refusal| Blocked::Refused(Refused::Present(refusal)))?;
    let key = match resolvers.mappings.backing(packet.mapping) {
        Some(backing) => {
            crate::runtime::drain::note_store_route("ingress_present_target_resolved");
            AccessKey::Whole(ResourceKey {
                backing,
                heap: None,
            })
        }
        None => {
            // Counted, and counted here rather than left to the rung census, so
            // the reading distinguishes "this rail shows frames the device
            // cannot resolve" from "this rail shows no frames".
            crate::runtime::drain::note_store_route("ingress_present_target_unresolved");
            AccessKey::DomainOnly
        }
    };
    let access = AccessIntent {
        domain: fifo.domain(),
        key,
        mode: AccessMode::Read,
        // A present names no shader stage. Zero always means the record carried
        // none, never that one was dropped.
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    };
    PresentPayload::new(packet, vec![(packet.mapping, access)])
        .map_err(|mismatch| Blocked::Refused(Refused::PresentAccesses(mismatch)))
}

/// Every resource-lifetime command, resolved in the namespaces the packet names.
///
/// # The namespaces are here, and they were not the only thing missing
///
/// This function was two commands wide: a task definition and a task deletion
/// name a task and nothing inside it, so they resolved from their own bytes,
/// and the other ten waited on a namespace nobody had given this bridge. The
/// namespaces are now here — `reims_vgpu_core::lifecycle::operation` is the one
/// join that picks which of its five sub-joins a kind belongs to, and it takes
/// both — and **ten of the twelve now produce an operation**.
///
/// An operation is not a payload. `reims_vgpu_core::transaction::LifecyclePayload`
/// holds an operation together with the accesses its named resources take, and
/// refuses a list that disagrees with the operation's own resource set. That
/// refusal is what partitions the ten: four of them name no resource, so the
/// empty list is not a shortfall but the correct statement, and the rest name
/// resources and owe each one an access. Both halves of that access are now
/// answerable — the mode from the operation, the key from
/// [`Resolvers::storage`] — so all ten cross.
///
/// **The partition is the type's and not a list here.** [`lifecycle_accesses`]
/// asks the operation which resources it names and what it declares of them; it
/// does not enumerate which kinds have a list. A thirteenth command, or a kind
/// that gains a resource list, moves itself.
///
/// # Two commands are short of neither a namespace nor an access
///
/// `SetObjectList`'s operation is the result of *walking* the guest's table, and
/// `ReplacePhysical`'s names pages that are not on the wire at all. Both resolve
/// their refs and still cannot be built, which is why the core names them with
/// their own refusals rather than with `UnknownRef` — and why they arrive here
/// as [`Gap::ObjectListTable`] and [`Gap::ReplacementStorage`] rather than as one
/// gap called "namespaces" that would have read as closed the day the namespaces
/// landed.
///
/// # A ref that names nothing is the guest's, not this device's
///
/// [`reims_vgpu_core::lifecycle::ResolveRefusal::UnknownRef`] stays a
/// [`Blocked::Refused`], and that is a claim worth stating because it was not
/// true before `crate::runtime::objects::name_resource`. While this device named
/// a reference only when something constructed it, an unresolved ref meant "no
/// draw has touched this yet" — device incompleteness — and 49 well-formed
/// packets a boot would have been refused under it. Resolution now reads the
/// guest's own object list, so a ref that still resolves to nothing is an empty
/// slot or an unreadable descriptor: the packet's, and not a gap that closes
/// later.
///
/// # Errors
///
/// [`Blocked::Gap`] for the commands whose payload this bridge cannot yet
/// complete, and [`Blocked::Refused`] when a lifetime packet's bytes are not the
/// command its opcode names.
fn resource_lifetime(
    fifo: Fifo,
    channel: Channel,
    opcode: u16,
    payload: &[u8],
    resolvers: Resolvers<'_>,
) -> Result<reims_vgpu_core::transaction::LifecyclePayload, Blocked> {
    use reims_vgpu_core::lifecycle::{self, LifecycleKind, ResolveRefusal};
    use reims_vgpu_core::transaction::LifecyclePayload;
    let gap = |gap| Blocked::Gap {
        channel,
        opcode,
        gap,
    };
    // Asked of the same table `classify` read, rather than of the opcode again:
    // the class said this is a lifetime command and the kind says which one.
    let Some(kind) = LifecycleKind::of(channel, opcode) else {
        // Unreachable through [`packet`], which reached this arm because
        // `classify` said `ResourceLifecycle` and `LifecycleKind::of` asks that
        // same table. Answered rather than unwrapped, because the two are
        // separate functions and a disagreement between them must be a value a
        // reader can find in a log.
        return Err(gap(Gap::Unresolved));
    };
    let op = lifecycle::operation(kind, payload, resolvers.objects, resolvers.mappings).map_err(
        |refusal| match refusal {
            // This device's own incompleteness, each named for what it is short
            // of. The core states them as refusals because the core has no
            // vocabulary for "the caller could give me this later"; this bridge
            // does.
            ResolveRefusal::NeedsGuestTable { .. } => gap(Gap::ObjectListTable),
            ResolveRefusal::NeedsStorage { .. } => gap(Gap::ReplacementStorage),
            other => Blocked::Refused(Refused::Lifecycle(other)),
        },
    )?;
    // The empty access list, offered to the type that knows whether it is the
    // truth. It is, for an operation that names no resource; for one that does,
    // the constructor says which resource is unaccounted for and that is the
    // gap.
    let accesses = lifecycle_accesses(fifo, &op, resolvers);
    LifecyclePayload::new(op, accesses)
        .map_err(|mismatch| Blocked::Refused(Refused::LifecycleAccesses(mismatch)))
}

/// The access every resource a lifetime operation names is subject to.
///
/// # The mode is the operation's and the key is the device's
///
/// Two terms with two owners, joined here and derived in neither place. The
/// direction — what the command does to the memory it names — is a property of
/// the command, so `reims_vgpu_core::lifecycle::LifecycleOp::declared_access`
/// states it once for the whole model; a table here would be that statement's
/// second copy, and the copy a device kept would be the one that drifted. Where
/// those bytes are is not the model's to know at all, so it arrives through
/// `reims_vgpu_core::resolve::ResourceStorage`.
///
/// # An unresolvable target is `DomainOnly`, counted, and not dropped
///
/// A resource whose storage this device has no key for still takes part in the
/// operation, and an access list missing it is a list
/// `reims_vgpu_core::transaction::LifecyclePayload` refuses — which would lose
/// the whole packet over one resource. `AccessKey::DomainOnly` is the
/// vocabulary for exactly this: participation is real, the memory is not
/// established, ordering comes from the submission domain alone. It is the same
/// judgement [`present_payload`] makes for a mapping with no live surface, and
/// it is counted for the same reason — so a boot says how many lifetime edges
/// were bought at the coarse rung rather than leaving it to inference.
///
/// The whole backing, because the operation names no narrower extent: a
/// synchronise names a resource, and even an invalidate's validity quad is not
/// carried into the operation. Rung 2 is the finest answer the packet contains.
fn lifecycle_accesses(
    fifo: Fifo,
    op: &reims_vgpu_core::lifecycle::LifecycleOp,
    resolvers: Resolvers<'_>,
) -> Vec<(
    reims_vgpu_core::identity::ResourceId,
    reims_vgpu_core::access::AccessIntent,
)> {
    use reims_vgpu_core::access::{AccessIntent, AccessKey};
    // Asked of the operation rather than of the kind: an operation that names
    // no resource declares no access, and the two answers are one match in one
    // file so they cannot disagree about which commands those are.
    let Some(mode) = op.declared_access() else {
        return Vec::new();
    };
    let task = op.task();
    op.resources()
        .iter()
        .map(|&resource| {
            let key = match resolvers.storage.storage(task, resource) {
                Some(key) => {
                    crate::runtime::drain::note_store_route("ingress_lifecycle_target_resolved");
                    AccessKey::Whole(key)
                }
                None => {
                    crate::runtime::drain::note_store_route("ingress_lifecycle_target_unresolved");
                    AccessKey::DomainOnly
                }
            };
            (
                resource,
                AccessIntent {
                    domain: fifo.domain(),
                    key,
                    mode,
                    // A lifetime command names no shader stage. Zero always
                    // means the record carried none, never that one was dropped.
                    api_stages: 0,
                    // No content version is established for a lifecycle access.
                    // The content authority is `reims_vgpu_core::content`'s and
                    // it is not consulted here; `None` says the term is absent,
                    // which is what it is, rather than reserving a version this
                    // bridge has no authority to hand out.
                    input_content_version: None,
                    output_content_version: None,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // The channel bound, read only by the test that sweeps it: `Fifo::child`
    // asks `is_child_channel` and never the constant behind it.
    use crate::model::MAX_CHANNELS;
    use reims_vgpu_protocol::packets::LEDGER;

    /// A payload long enough for every control command's own layout and every
    /// present trailer. Only the two channel-lifetime commands and the three
    /// present forms read one at all.
    const ROOMY: usize = 64;

    /// A mapping namespace a test states outright, so what a present resolves
    /// to is the test's choice and not a device's state.
    struct Mappings(Vec<u32>);

    impl reims_vgpu_core::resolve::MappingResolver for Mappings {
        fn backing(
            &self,
            mapping: reims_vgpu_core::identity::MappingId,
        ) -> Option<reims_vgpu_core::access::BackingId> {
            self.0
                .contains(&mapping.0)
                .then(|| reims_vgpu_core::access::BackingId(u64::from(mapping.0) | 1 << 40))
        }
    }

    /// A namespace nothing is live in, for the rows whose class never asks.
    const NOTHING_MAPPED: Mappings = Mappings(Vec::new());

    /// A mapping namespace that answers for every id, so a ledger row that
    /// gapped because a fixture held no mapping would not be reporting on the
    /// fixture.
    struct EveryMapping;

    impl reims_vgpu_core::resolve::MappingResolver for EveryMapping {
        fn backing(
            &self,
            mapping: reims_vgpu_core::identity::MappingId,
        ) -> Option<reims_vgpu_core::access::BackingId> {
            Some(reims_vgpu_core::access::BackingId(
                u64::from(mapping.0) | 1 << 41,
            ))
        }
    }

    /// An object namespace that answers for every ref in every task.
    ///
    /// The ledger sweep is about which join a row reaches and what it is still
    /// short of, so a row that gapped because a fixture happened not to hold an
    /// object would be reporting on the fixture.
    struct EveryObject;

    impl reims_vgpu_core::resolve::TaskNamespaces for EveryObject {
        fn resource(
            &self,
            task: reims_vgpu_core::identity::TaskId,
            object_ref: u32,
        ) -> Option<reims_vgpu_core::identity::ResourceId> {
            Some(reims_vgpu_core::identity::ResourceId {
                slot: reims_vgpu_core::identity::ObjectListRef(object_ref),
                generation: reims_vgpu_core::identity::SlotGeneration(
                    u64::from(task.0).saturating_add(1),
                ),
            })
        }
    }

    /// A storage source that answers for every resource, keyed on the name so
    /// two different names never collide on one backing.
    ///
    /// Paired with [`EveryObject`] for the ledger sweep's reason: a row that
    /// came out at the coarse rung because a fixture held no storage would be
    /// reporting on the fixture.
    struct EveryStorage;

    impl reims_vgpu_core::resolve::ResourceStorage for EveryStorage {
        fn storage(
            &self,
            task: reims_vgpu_core::identity::TaskId,
            resource: reims_vgpu_core::identity::ResourceId,
        ) -> Option<reims_vgpu_core::access::ResourceKey> {
            Some(reims_vgpu_core::access::ResourceKey {
                backing: reims_vgpu_core::access::BackingId(
                    u64::from(task.0) << 32 | u64::from(resource.slot.0) | 1 << 42,
                ),
                heap: None,
            })
        }
    }

    /// A storage source with no term for anything, for the test that asserts
    /// what a lifetime packet does when its target cannot be keyed.
    struct NoStorage;

    impl reims_vgpu_core::resolve::ResourceStorage for NoStorage {
        fn storage(
            &self,
            _task: reims_vgpu_core::identity::TaskId,
            _resource: reims_vgpu_core::identity::ResourceId,
        ) -> Option<reims_vgpu_core::access::ResourceKey> {
            None
        }
    }

    fn resolvers(mappings: &dyn MappingResolver) -> Resolvers<'_> {
        Resolvers {
            mappings,
            objects: &EveryObject,
            storage: &EveryStorage,
        }
    }

    fn drained(opcode: u16) -> drain::Packet {
        drain::Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: 0,
            payload: vec![0u8; ROOMY],
            next_head: 0,
        }
    }

    /// The same packet, with a one-record resource list for the four commands
    /// that carry one.
    ///
    /// A zero-filled payload decodes those four as a list of **no** resources,
    /// and an operation naming no resource legitimately crosses — so a ledger
    /// sweep on the bare fixture would report the list commands as crossing and
    /// never exercise the path where a named resource owes an access. The list
    /// is one record long, and the invalidate's record carries the guest-write
    /// quad because a zero quad is a refusal about the guest's bytes rather than
    /// a statement about this bridge.
    fn drained_with_list(channel: Channel, opcode: u16) -> drain::Packet {
        use reims_vgpu_core::lifecycle::LifecycleKind;
        use reims_vgpu_protocol::fifo::{
            InvalidateValidityOps, CHILD_INVALIDATE_RECORD_LEN, CHILD_SYNCHRONIZE_RECORD_LEN,
        };
        let mut packet = drained(opcode);
        let Some(record_len) =
            LifecycleKind::of(channel, opcode).and_then(LifecycleKind::resource_list_record_len)
        else {
            return packet;
        };
        // `{u32 task, u32 count}` then `count` records. The count's offset is
        // the header's own, and the record's shape is the length's.
        packet.payload[4..8].copy_from_slice(&1u32.to_le_bytes());
        packet.payload[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert!(
            record_len == CHILD_INVALIDATE_RECORD_LEN || record_len == CHILD_SYNCHRONIZE_RECORD_LEN
        );
        if record_len == CHILD_INVALIDATE_RECORD_LEN {
            packet.payload[12..16]
                .copy_from_slice(&InvalidateValidityOps::PAGEON.to_le_dword().to_le_bytes());
        }
        packet
    }

    fn fifo_for(channel: Channel) -> Fifo {
        match channel {
            Channel::Root => Fifo::ROOT,
            Channel::Child => Fifo::child(1).expect("channel 1 is a child"),
        }
    }

    /// **The cutover ledger.** Every row the protocol crate judged, put through
    /// this bridge, and the answer asserted against the class the row has.
    ///
    /// A row that changes class, or a gap that closes without this being
    /// updated, fails here rather than becoming a silent change in what crosses
    /// to the model.
    #[test]
    fn every_ledger_row_either_crosses_or_names_what_it_needs() {
        let mut crossed = 0usize;
        let mut gapped = 0usize;
        for row in LEDGER {
            let fifo = fifo_for(row.channel);
            let outcome = packet(
                fifo,
                SessionGeneration::FIRST,
                StampSlot(0),
                &drained_with_list(row.channel, row.opcode),
                resolvers(&EveryMapping),
            );
            let expected = match classify(row.channel, row.opcode) {
                Some(PayloadClass::Control) => None,
                Some(PayloadClass::Exec) => Some(Gap::ExecResolution),
                // The class is not the unit here, and this is the one place
                // that says so. Two of its members — a task definition and a
                // task deletion — name a task and nothing inside it, so they
                // need no namespace and cross; the other ten name objects,
                // mappings or counted lists of them and do not.
                // The class is not the unit, and this is the one place that
                // says how it splits. Asked of the kind table rather than of a
                // list of opcodes written here, so the test cannot drift into
                // agreeing with itself.
                Some(PayloadClass::ResourceLifecycle) => lifetime_gap(row.channel, row.opcode),
                Some(PayloadClass::Query) => Some(Gap::ReplyDestination),
                // Crosses, and crosses on a namespace nothing is live in: the
                // target resolves to no backing and the access says so. That
                // the *unresolved* case still crosses is the assertion — a
                // bridge that gapped when it could not resolve a mapping would
                // stop presenting the moment a surface was reconfigured.
                Some(PayloadClass::Present) => None,
                None => Some(Gap::Unresolved),
            };
            match (expected, outcome) {
                (None, Ok(built)) => {
                    crossed += 1;
                    assert_eq!(built.opcode, row.opcode);
                    assert_eq!(built.channel, row.channel);
                    assert_eq!(built.domain, fifo.domain());
                    let right_class = match classify(row.channel, row.opcode) {
                        Some(PayloadClass::Control) => matches!(built.payload, Payload::Control(_)),
                        Some(PayloadClass::ResourceLifecycle) => {
                            matches!(built.payload, Payload::ResourceLifecycle(_))
                        }
                        Some(PayloadClass::Present) => matches!(built.payload, Payload::Present(_)),
                        _ => false,
                    };
                    assert!(
                        right_class,
                        "{} {:#04x} ({}) built a payload that is not its class",
                        row.channel.name(),
                        row.opcode,
                        row.name
                    );
                }
                (
                    Some(want),
                    Err(Blocked::Gap {
                        gap,
                        channel,
                        opcode,
                    }),
                ) => {
                    gapped += 1;
                    assert_eq!(
                        gap,
                        want,
                        "{} {:#04x} ({}) named the wrong missing input",
                        row.channel.name(),
                        row.opcode,
                        row.name
                    );
                    assert_eq!((channel, opcode), (row.channel, row.opcode));
                }
                (want, got) => panic!(
                    "{} {:#04x} ({}) expected {want:?} and got {got:?}",
                    row.channel.name(),
                    row.opcode,
                    row.name
                ),
            }
        }
        assert_eq!(
            crossed + gapped,
            LEDGER.len(),
            "every row is accounted for exactly once"
        );
        assert!(
            crossed > 0 && gapped > 0,
            "one side of the partition is empty, so the assertions above compared nothing: \
             {crossed} crossed, {gapped} gapped"
        );
    }

    /// What a resource-lifetime row is still short of, or `None` if it crosses.
    ///
    /// Asked of the same kind table the bridge asks, so the test cannot drift
    /// into its own list of opcodes and then agree with itself. **The split is
    /// down to two**: a rebind's operation is the result of walking the guest's
    /// object-list table and a re-point's names pages that are not on the wire,
    /// and neither is a thing this bridge can be handed. Everything else in the
    /// class builds a payload — the five that name resources now state both
    /// halves of the access each one takes, so naming a resource stopped being
    /// what holds a lifetime command back.
    fn lifetime_gap(channel: Channel, opcode: u16) -> Option<Gap> {
        use reims_vgpu_core::lifecycle::LifecycleKind as K;
        match K::of(channel, opcode)? {
            K::SetObjectList => Some(Gap::ObjectListTable),
            K::ReplacePhysical => Some(Gap::ReplacementStorage),
            K::DeleteResource
            | K::Invalidate
            | K::Synchronize
            | K::SynchronizeAndDiscard
            | K::Discard
            | K::DeleteBacking
            | K::DefineTask
            | K::DeleteTask
            | K::MapMemory
            | K::UnmapMemory => None,
        }
    }

    /// What crosses is exactly the whole control class, the whole present class,
    /// and every resource-lifetime command except the two whose operation needs
    /// something that is not in their packet — and nothing else.
    ///
    /// The counts are spelled out because this is the cutover's own scoreboard:
    /// a gap that closes moves a number here, and a row that quietly changed
    /// class moves one without anybody deciding to.
    #[test]
    fn what_crosses_is_control_present_and_every_lifetime_command_but_two() {
        let crossing: Vec<_> = LEDGER
            .iter()
            .filter(|row| {
                packet(
                    fifo_for(row.channel),
                    SessionGeneration::FIRST,
                    StampSlot(0),
                    &drained_with_list(row.channel, row.opcode),
                    resolvers(&EveryMapping),
                )
                .is_ok()
            })
            .map(|row| (row.channel, row.opcode))
            .collect();
        let expected: Vec<_> = LEDGER
            .iter()
            .filter(|row| {
                matches!(
                    classify(row.channel, row.opcode),
                    Some(PayloadClass::Control | PayloadClass::Present)
                ) || (classify(row.channel, row.opcode) == Some(PayloadClass::ResourceLifecycle)
                    && lifetime_gap(row.channel, row.opcode).is_none())
            })
            .map(|row| (row.channel, row.opcode))
            .collect();
        assert_eq!(crossing, expected);
        let control = LEDGER
            .iter()
            .filter(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Control))
            .count();
        let present = LEDGER
            .iter()
            .filter(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Present))
            .count();
        assert_eq!(
            (control, present, crossing.len()),
            (23, 3, 38),
            "the ledger's crossing rows changed; what reaches the model is not what the \
             module documentation says it is"
        );
    }

    /// A task definition crosses whole: the packet's own fields, decoded, with
    /// no access list because the operation names nothing for one to be about.
    ///
    /// The two crossing lifetime commands are asserted for their *content* and
    /// not only for the fact that they cross. A bridge that produced a
    /// well-formed envelope around a task id it had defaulted would pass the
    /// ledger test above and hand the model a definition for task zero.
    #[test]
    fn a_task_definition_crosses_carrying_its_own_decoded_fields() {
        use reims_vgpu_core::identity::{DirectoryFrame, TaskId};
        use reims_vgpu_core::lifecycle::LifecycleOp;

        let mut drained = drained(0x38);
        // `(task_id << 1) | is_kernel_task`, an address-space length this model
        // deliberately does not carry, then the directory page frame — at the
        // protocol's own offsets rather than at ones restated here.
        use reims_vgpu_protocol::fifo::{DEFINE_TASK_DIRECTORY_PFN, DEFINE_TASK_RAW_ID};
        drained.payload[DEFINE_TASK_RAW_ID..DEFINE_TASK_RAW_ID + 4]
            .copy_from_slice(&((7u32 << 1) | 1).to_le_bytes());
        drained.payload[DEFINE_TASK_DIRECTORY_PFN..DEFINE_TASK_DIRECTORY_PFN + 4]
            .copy_from_slice(&0x1234u32.to_le_bytes());

        let built = packet(
            Fifo::ROOT,
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained,
            resolvers(&NOTHING_MAPPED),
        )
        .expect("a task definition needs no namespace");
        let Payload::ResourceLifecycle(payload) = &built.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert_eq!(
            payload.op(),
            &LifecycleOp::DefineTask {
                task: TaskId(7),
                kernel: true,
                directory: DirectoryFrame(0x1234),
            },
            "the fields are the packet's, not defaults around a well-formed \
             envelope"
        );
        assert!(
            payload.accesses().is_empty(),
            "a task definition names no resource, so there is nothing for an \
             access to be about"
        );
    }

    /// A channel-lifetime command with no room for its domain is refused, not
    /// defaulted. Opening domain 0 would name the root FIFO.
    #[test]
    fn a_channel_command_too_short_to_name_a_domain_is_refused() {
        let mut short = drained(0x30);
        short.payload.clear();
        let refusal = packet(
            Fifo::ROOT,
            SessionGeneration::FIRST,
            StampSlot(0),
            &short,
            resolvers(&NOTHING_MAPPED),
        )
        .expect_err("no domain, no transition");
        assert!(matches!(refusal, Blocked::Refused(_)));
        assert_eq!(
            refusal.gap(),
            None,
            "a short payload is not a missing input"
        );
        assert_eq!(
            control::ControlKind::of(Channel::Root, 0x30)
                .and_then(control::ControlKind::channel_transition),
            Some(control::ChannelTransition::Open),
            "the opcode above stopped being the channel-open command, so this test refuses \
             something else"
        );
    }

    /// The envelope: the slot is the FIFO's, the value is the packet's, and a
    /// wait's raw index is masked exactly once.
    #[test]
    fn the_envelope_carries_the_channels_slot_and_the_packets_values() {
        let raw_index = u32::MAX;
        let mut nop = drained(0x1e);
        nop.completion_stamp = 0xDEAD_BEEF;
        nop.stamp_waits = vec![drain::StampWait {
            index: raw_index,
            value: 7,
        }];
        let built = packet(
            fifo_for(Channel::Child),
            SessionGeneration::FIRST,
            StampSlot(3),
            &nop,
            resolvers(&NOTHING_MAPPED),
        )
        .expect("CmdNOP is control");
        assert_eq!(
            built.completion,
            Some(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(0xDEAD_BEEF),
            })
        );
        assert_eq!(
            built.stamp_waits,
            vec![StampWait {
                slot: StampSlot(stamp_slot_index(raw_index)),
                value: StampValue(7),
            }]
        );
        assert_ne!(
            stamp_slot_index(raw_index),
            raw_index,
            "the raw index used above is not masked by anything, so the assertion that it was \
             masked would hold for a bridge that carried it through untouched"
        );
    }

    /// A lifetime command's refs resolve, and what the packet is still short of
    /// depends on whether it named a resource — not on which opcode it is.
    ///
    /// The same opcode, twice, with two payloads. An invalidate naming nothing
    /// is an operation over no resources and crosses; an invalidate naming one
    /// owes that resource an access and does not. A bridge whose partition was
    /// per-opcode would give both answers to whichever list it was written
    /// against, and the wrong one is silent: an invalidate that crossed with an
    /// empty access list would move content authority with nothing ordered
    /// against it.
    #[test]
    fn a_lifetime_command_owes_an_access_for_each_resource_it_names() {
        use reims_vgpu_core::lifecycle::{LifecycleKind, LifecycleOp};
        use reims_vgpu_protocol::packets::LEDGER;

        let row = LEDGER
            .iter()
            .find(|row| {
                LifecycleKind::of(row.channel, row.opcode) == Some(LifecycleKind::Invalidate)
            })
            .expect("the ledger has an invalidate row");
        let fifo = fifo_for(row.channel);
        let put = |packet: &drain::Packet| packet_of(fifo, packet);

        // Zero records. The operation names no resource, so the empty access
        // list is its own statement rather than a shortfall.
        let empty = put(&drained(row.opcode)).expect("an invalidate over nothing is an operation");
        let Payload::ResourceLifecycle(payload) = &empty.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert!(matches!(
            payload.op(),
            LifecycleOp::Invalidate { resources, .. } if resources.is_empty()
        ));
        assert!(payload.accesses().is_empty());

        // One record. The ref resolves and the resource it names is keyed, so
        // the packet crosses with exactly one access — over that resource, at
        // the whole-backing rung, in a direction that orders against a reader.
        let named = put(&drained_with_list(row.channel, row.opcode))
            .expect("a named resource's access is answerable");
        let Payload::ResourceLifecycle(payload) = &named.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        let LifecycleOp::Invalidate { task, resources } = payload.op() else {
            panic!("the invalidate opcode built another operation");
        };
        assert_eq!(resources.len(), 1, "the list carried one record");
        assert_eq!(
            payload.accesses().len(),
            1,
            "one resource, one access — the envelope is what holds those together"
        );
        let access = payload.accesses()[0];
        assert_eq!(
            access.key,
            reims_vgpu_core::access::AccessKey::Whole(
                EveryStorage
                    .storage(*task, resources[0])
                    .expect("the fixture keys every resource")
            ),
            "the access is keyed on the storage the resolver gave for the resource \
             the operation named, not on a default or on another resource's"
        );
        assert_eq!(access.mode, reims_vgpu_core::access::AccessMode::Write);
        assert_eq!(access.domain, fifo.domain());

        // The same packet with nothing to key it on. The resource is still the
        // operation's, so the access is still there — at the coarse rung rather
        // than absent, because an access list one short of the operation's own
        // resources is a packet the envelope refuses outright.
        let unkeyed = packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained_with_list(row.channel, row.opcode),
            Resolvers {
                mappings: &EveryMapping,
                objects: &EveryObject,
                storage: &NoStorage,
            },
        )
        .expect("an unkeyable target is not a lost packet");
        let Payload::ResourceLifecycle(payload) = &unkeyed.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert_eq!(
            payload.accesses().iter().map(|a| a.key).collect::<Vec<_>>(),
            vec![reims_vgpu_core::access::AccessKey::DomainOnly],
            "with no storage term the ordering is the submission domain's, and the \
             access is still declared"
        );
    }

    /// [`packet`] over the ledger sweep's namespaces, so a test naming one
    /// packet does not restate the four arguments the sweep already fixes.
    fn packet_of(fifo: Fifo, drained: &drain::Packet) -> Result<Packet, Blocked> {
        packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            drained,
            resolvers(&EveryMapping),
        )
    }

    /// A present crosses carrying one read of the backing its target resolves
    /// to, and the frame's target is the packet's own word rather than a
    /// default.
    ///
    /// Two things a well-formed envelope would pass without: the access is over
    /// the backing *this* mapping resolved to, and it is a read. A present
    /// modelled as a writer would reserve a content version, beat the real
    /// writer's, and publish bytes nothing produced.
    #[test]
    fn a_present_crosses_carrying_one_read_of_its_targets_backing() {
        use reims_vgpu_core::access::{AccessKey, AccessMode, ResourceKey};
        use reims_vgpu_core::identity::MappingId;
        use reims_vgpu_core::present::PresentForm;
        use reims_vgpu_protocol::packets::LEDGER;

        let row = LEDGER
            .iter()
            .find(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Present))
            .expect("the ledger has a present row");
        let form = PresentForm::of(row.channel, row.opcode).expect("a present row has a form");

        let mut shown = drained(row.opcode);
        let target = 9u32;
        let at = form.target_offset();
        shown.payload[at..at + 4].copy_from_slice(&target.to_le_bytes());

        let live = Mappings(vec![target]);
        let fifo = fifo_for(row.channel);
        let built = packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            &shown,
            resolvers(&live),
        )
        .expect("a present resolves in the mapping namespace");
        let Payload::Present(payload) = &built.payload else {
            panic!("a present must not build another class's payload");
        };
        assert_eq!(
            payload.packet().mapping,
            MappingId(target),
            "the target is the packet's own word, not a default"
        );
        let backing = {
            use reims_vgpu_core::resolve::MappingResolver as _;
            live.backing(MappingId(target)).expect("live")
        };
        assert_eq!(
            payload.accesses(),
            &[reims_vgpu_core::access::AccessIntent {
                domain: fifo.domain(),
                key: AccessKey::Whole(ResourceKey {
                    backing,
                    heap: None,
                }),
                mode: AccessMode::Read,
                api_stages: 0,
                input_content_version: None,
                output_content_version: None,
            }],
            "one read of the whole of the backing the target resolved to"
        );

        // The same packet against a namespace the target is not live in. It
        // still crosses — a frame the device cannot resolve is still a frame the
        // guest is waiting on — and it says outright that it does not know what
        // it touches.
        let unresolved = packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            &shown,
            resolvers(&NOTHING_MAPPED),
        )
        .expect("an unresolvable target is not a missing input");
        let Payload::Present(payload) = &unresolved.payload else {
            panic!("still a present");
        };
        assert_eq!(
            payload.accesses()[0].key,
            AccessKey::DomainOnly,
            "no backing to name, and the vocabulary for that is not a guess"
        );
        assert_eq!(
            payload.accesses()[0].key.rung(),
            3,
            "and it prices itself on the precision ladder as the coarsest rung"
        );
    }

    /// A present too short for its own trailer is refused, not shown.
    ///
    /// Clamping would present mapping zero and complete the packet in silence,
    /// which is a frame the guest believes it showed.
    #[test]
    fn a_present_too_short_for_its_trailer_is_refused_rather_than_shown() {
        use reims_vgpu_protocol::packets::LEDGER;

        let row = LEDGER
            .iter()
            .find(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Present))
            .expect("the ledger has a present row");
        let mut short = drained(row.opcode);
        short.payload.clear();
        let refusal = packet(
            fifo_for(row.channel),
            SessionGeneration::FIRST,
            StampSlot(0),
            &short,
            resolvers(&NOTHING_MAPPED),
        )
        .expect_err("no trailer, no frame");
        assert_eq!(
            refusal.gap(),
            None,
            "a short payload is this guest's bytes, not this device's incompleteness"
        );
        assert!(matches!(
            refusal,
            Blocked::Refused(Refused::Present(
                reims_vgpu_core::present::ResolveRefusal::Payload(_)
            ))
        ));
    }

    /// The device's channel numbering, which is what makes the channel and the
    /// domain one value rather than two that can disagree.
    #[test]
    fn the_root_fifo_is_channel_zero_and_children_are_the_rest() {
        assert_eq!(Fifo::ROOT.domain(), ChannelId(0));
        assert_eq!(Fifo::ROOT.channel(), Channel::Root);
        assert_eq!(
            Fifo::child(0),
            None,
            "channel 0 is the root FIFO, not a child"
        );
        assert_eq!(
            Fifo::child(MAX_CHANNELS as u32),
            None,
            "one past the last channel this device has"
        );
        for id in 1..MAX_CHANNELS as u32 {
            let fifo = Fifo::child(id).expect("a child this device has");
            assert_eq!(fifo.domain(), ChannelId(id));
            assert_eq!(fifo.channel(), Channel::Child);
        }
    }
}
