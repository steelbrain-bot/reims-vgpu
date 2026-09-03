// DISCONNECTED — not a module, not compiled, not linkable. Source to read.
//
// Moved here by the commit that wired the ordering and publication group (G3).
// See `crates/reims-vgpu/src/dead/README.md` for the register row, and
// `AGENTS.md` for why nothing is resurrected from this directory.
//
// WHAT THIS WAS
//
// The drain's own stamp machinery: the per-drain coalescing latch, the verdict
// it produced, and the three `break`s that verdict drove. A packet whose stamp
// wait was unmet stopped its whole timeline at the ring's consumer pointer, and
// the latch existed because the word that would clear it was sitting unwritten
// in the same drain's stack frame.
//
// WHAT REPLACED IT
//
// `reims_vgpu_core::session::SessionModel::admit` takes a packet's stamp waits
// as an *ordering position*, and `runtime::drain::settle_model_work` releases
// the position when the slot reaches it. There is no latch because there is no
// window in which a completed packet's word is unpublished: completing a
// position publishes it. There is no verdict because nothing chooses — a
// position either has been released or has not.
//
// The one arm that survived the move is `Unevaluable`, and it moved rather than
// disappeared: `admit_and_park` drops a wait naming a slot outside the stamp
// page before admission and counts `packet_stamp_wait_unresolvable`. The
// asymmetry the variant's doc argues is the same one, stated where the packet
// is admitted instead of where a ring head is moved.

/// The completion stamp one drain of one channel owes, coalesced.
///
/// Every packet in a `drain_child_fifo` call stamps **the same slot** — the
/// index is read once from the channel's register block before the loop — and a
/// stamp wait is satisfied by any value at or past the one awaited
/// ([`StampWait::satisfied_by`]). So a run of stamps to one slot is observable
/// only through its greatest value unless the guest samples between the writes,
/// and writing only that value discharges every wait the run would have.
///
/// That matters because a stamp is a FIFO completion, not merely a word write.
/// Coalescing avoids one completion record per packet while preserving the
/// greatest value the guest can observe from the drain.
///
/// # What it bought, six driven macos-13 boots, one binary, both arms
///
/// All six in one compositing regime (995-999 draws a frame), no panics, same
/// desktop. `gpu_stamps` fell to **1.9 %** of the per-packet arm and the span
/// with it:
///
/// ```text
///                   coalesced    per packet
/// stamp ms/s         5.9-6.8     77.1-97.2
/// unnamed in drain   224-231      314-317
/// duty              0.80-0.81    0.89-0.90
/// slot_us              29 049        37 874
/// ```
///
/// **The arithmetic closes**: the stamp span fell 90.3 ms/s and the whole drain
/// residue fell 88.6 ms/s, so the time was removed rather than relocated — and
/// `proc - draw - compute` is 197 against 201 ms/s, unchanged, which says the
/// same from the other side.
///
/// It buys **headroom, not frames**. `present_hz`/`offered_hz` are 15.05 against
/// 14.90; the guest paces this rail and four CPU wins in a row have moved it by
/// nothing.
///
/// # It holds on all six guest drivers
///
/// The measurement above is macos-13 alone, and *when a completion becomes
/// visible* is exactly the class that can be fine on one guest driver and stall
/// another — failing as a frozen desktop rather than as a decline, which no
/// counter reports. So all six x86 rails were swept:
///
/// ```text
/// macos-11  dev=1 ssh=1 dock=1 sd=184 panic=0
/// macos-12  dev=1 ssh=1 dock=1 sd=147 panic=0
/// macos-13  dev=1 ssh=1 dock=1 sd=131 panic=0
/// macos-14  dev=1 ssh=1 dock=1 sd=215 panic=0
/// macos-15  dev=1 ssh=1 dock=1 sd=281 panic=0
/// macos-26  dev=1 ssh=1 dock=1 sd=126 panic=0
/// ```
///
/// `sd` is the field that answers the question — host-window standard deviation,
/// ~38 for the boot screen and >100 for a composited desktop. Every rail cleared
/// 126, so every one of them put a picture up rather than sitting on a stamp it
/// was still waiting for. `stamp_write_forward` runs 345-1815 a boot across the
/// six, so the coalesced write is landing everywhere and not only where it was
/// tuned.
///
/// These are undriven boots: they prove the desktop composites on each driver,
/// not that the saving reproduces there. Only macos-13 has been driven.
///
/// Completion publication does not close an open draw batch. It registers the
/// stamp in the bounded pending queue and the batch's eventual successful
/// submission assigns its completion point. The pending-stamp capacity remains
/// the pressure bound: filling it submits the batch rather than sleeping while
/// holding the only command buffer that can make room.
///
/// [`Self::latch`] takes the **maximum in wrapping-signed order** rather than
/// the last value seen. For a well-formed guest those are the same, and taking
/// the maximum means this device cannot introduce a regressing stamp even if a
/// regressing one arrives — a slot going backwards would unsatisfy a wait the
/// guest had already been told was met.
#[derive(Clone, Copy, Default)]
pub struct PendingStamp {
    /// `None` until a packet in this drain has completed. A drain that stamps
    /// nothing owes nothing and must submit nothing.
    value: Option<u32>,
}

impl PendingStamp {
    /// Fold one packet's completion stamp in, keeping the later of the two in
    /// the same wrapping-signed order [`StampWait::satisfied_by`] compares in.
    pub fn latch(&mut self, stamp: u32) {
        self.value = Some(match self.value {
            Some(held) if stamp.wrapping_sub(held) as i32 <= 0 => held,
            _ => stamp,
        });
    }

    /// The value owed, or `None` when this drain stamped nothing.
    pub fn owed(self) -> Option<u32> {
        self.value
    }

    /// Whether `wait` is already discharged by what this drain has latched but
    /// not yet written.
    ///
    /// Without this a packet waiting on the slot an earlier packet in the same
    /// drain stamped would read the stale word out of guest RAM, return
    /// [`StampVerdict::Hold`], and park the channel against a stamp this device
    /// is itself holding. `slot` is the drain's own stamp index; a wait naming
    /// any other slot is not ours to answer.
    ///
    /// # It fires zero times, and the hazard it guards is real
    ///
    /// The A/B above is the same six boots. `packet_stamp_wait_met_pending` is
    /// **0** on the coalesced arm while `packet_stamp_wait_held` **more than
    /// doubled**, 5 073 against 2 237, taking `setup_calls` up 47 % with it in
    /// re-drains. Those two readings together say this comparison never matches:
    /// the packets that should have been answered here fall through to the stale
    /// word instead.
    ///
    /// It is not a correctness failure — every `break` flushes the pending
    /// stamp, so a held packet's retry finds the word — but it is ~2 800
    /// avoidable round trips a boot behind a guard that reads as working.
    ///
    /// # Why, measured rather than guessed
    ///
    /// It is **not** an encoding mismatch, which was the first suspicion. The
    /// boot's own `packet_stamp_wait_unmet` lines name both sides, and they
    /// disagree about the *channel*, not the spelling:
    ///
    /// ```text
    /// packet_stamp_wait_unmet opcode=0x37 ch1 index=2 awaited=0xe  current=0xa
    /// packet_stamp_wait_unmet opcode=0x6  ch5 index=1 awaited=0x5  current=0x3
    /// packet_stamp_wait_unmet opcode=0x22 ch2 index=4 awaited=0x1  current=0x0
    /// ```
    ///
    /// Channel 1 waits on slot 2, channel 5 on slot 1, channel 2 on slot 4.
    /// **Every one of them is waiting on a slot some other channel writes**, and
    /// this guard answers only the drain's own slot — by construction, as its
    /// last line above says. So it is not broken; it covers a case this workload
    /// does not produce, while the case the workload does produce is the one it
    /// declines.
    ///
    /// The mechanism is nesting: `process_child_packet` can reach `drain_other`,
    /// so channel B's drain runs inside channel A's, and A's latched stamps are
    /// sitting in A's stack frame where B cannot see them. Answering from them
    /// would be correct for the same reason it is correct here — the drain
    /// thread is single-threaded, so a latched stamp is work that finished
    /// before the waiting packet was decoded — but it needs the latch to live in
    /// `DeviceState` keyed by channel rather than in a local.
    ///
    /// **That is worth ~0.2 ms/s and no more**: 2 836 extra holds over a 45 s
    /// boot, each costing a re-drain, against a change that bought 88 ms/s. It
    /// is recorded because a guard that fires zero times should say why, not
    /// because it is the next thing to fix.
    fn discharges(self, slot: u32, wait: StampWait) -> bool {
        stamp_slot_index(wait.index) == slot && self.value.is_some_and(|v| wait.satisfied_by(v))
    }
}

/// What a packet's stamp waits say the drain should do with it.
///
/// The three answers are not degrees of the same thing. [`Self::Ready`] and
/// [`Self::Hold`] are the wait working; [`Self::Unevaluable`] is the device
/// unable to decide, and collapsing it into `Hold` is how a report becomes a
/// hang — see the variant's own note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StampVerdict {
    /// Every wait is satisfied, or there were none. Run the packet.
    Ready,
    /// At least one wait is genuinely behind. Hold the packet: a stamp this
    /// device will publish is what clears it.
    Hold,
    /// A wait this device cannot decide, and **no future event changes that**.
    ///
    /// Holding on one would park the timeline forever, which is strictly worse
    /// than the ordering slip it was meant to prevent: an ordering slip loses
    /// one packet's ordering, a parked root FIFO loses the guest. So this runs
    /// the packet, loudly. Every case here is a refusal with a named reason, and
    /// none of them fired on a driven boot.
    Unevaluable,
}

impl StampVerdict {
    /// Fold one wait's answer into the packet's, most restrictive winning.
    ///
    /// `Unevaluable` outranks `Hold`, which outranks `Ready`. That order is the
    /// whole point: a packet with one wait genuinely behind and one that can
    /// never be decided must **run**, because holding for the first would still
    /// park the timeline forever on the second.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unevaluable, _) | (_, Self::Unevaluable) => Self::Unevaluable,
            (Self::Hold, _) | (_, Self::Hold) => Self::Hold,
            _ => Self::Ready,
        }
    }
}

// --- the tests that moved with it ---

    // A packet carrying both an ordinary unmet wait and an undecidable one still
    // runs: holding for the first would park the timeline forever on the second.
    let mut both = vec![0u8; (PACKET_HEADER_LEN + 2 * PACKET_STAMP_LEN) as usize];
    st16(&mut both[PACKET_OPCODE..], 0);
    st16(&mut both[PACKET_STAMP_COUNT..], 2);
    st32(
        &mut both[PACKET_TOTAL_SIZE..],
        PACKET_HEADER_LEN + 2 * PACKET_STAMP_LEN,
    );
    st32(&mut both[PACKET_HEADER_LEN as usize..], 6);
    st32(&mut both[PACKET_HEADER_LEN as usize + 4..], 0x99);
    st32(
        &mut both[(PACKET_HEADER_LEN + PACKET_STAMP_LEN) as usize..],
        bad_slot,
    );
    st32(
        &mut both[(PACKET_HEADER_LEN + PACKET_STAMP_LEN) as usize + 4..],
        1,
    );
    let decoded = decode_packet(&both, 0, both.len() as u32, RING).expect("two records decode");
    assert_eq!(
        note_packet_stamp_waits(&state, &host, None, &decoded, None),
        StampVerdict::Unevaluable,
        "Unevaluable outranks Hold, or the packet parks forever on the wait that \
         cannot clear while waiting for the one that could"
    );


/// The coalesced stamp keeps the **greatest** value a drain latched, in the same
/// wrapping-signed order a wait is compared in.
///
/// Not "the last one seen". For a well-formed guest the two agree, and the whole
/// point of taking the maximum is that a regressing stamp arriving from the
/// guest cannot make this device publish a slot going backwards — which would
/// unsatisfy a wait the guest had already been told was met.
#[test]
fn a_coalesced_stamp_keeps_the_latest_value_across_the_u32_wrap() {
    let mut pending = PendingStamp::default();
    assert_eq!(
        pending.owed(),
        None,
        "a drain that stamped nothing owes nothing"
    );

    pending.latch(7);
    pending.latch(9);
    assert_eq!(pending.owed(), Some(9), "the later of two ascending stamps");

    pending.latch(8);
    assert_eq!(
        pending.owed(),
        Some(9),
        "a stamp behind the one held must not pull the slot backwards"
    );

    // Across the wrap: 0xffff_fff0 then 4. The signed difference is +20, so 4 is
    // *later*, and a plain `>=` would keep 0xffff_fff0 and stall every wait on
    // the far side of the wrap.
    let mut wrapped = PendingStamp::default();
    wrapped.latch(0xffff_fff0);
    wrapped.latch(4);
    assert_eq!(
        wrapped.owed(),
        Some(4),
        "the wrap is a signed difference, not a magnitude comparison"
    );
}

/// A wait on the slot the drain is holding is answered from the latch.
///
/// Without this the packet reads the stale word out of guest RAM, returns
/// `Hold`, and parks the channel against a stamp this device is itself sitting
/// on — a deadlock introduced by the coalescing rather than by the guest.
#[test]
fn a_pending_stamp_discharges_a_wait_on_its_own_slot_and_no_other() {
    const SLOT: u32 = 3;
    let mut pending = PendingStamp::default();
    pending.latch(20);

    let met = StampWait {
        index: SLOT,
        value: 20,
    };
    assert!(
        pending.discharges(SLOT, met),
        "a wait at exactly the latched value is discharged"
    );
    assert!(
        pending.discharges(
            SLOT,
            StampWait {
                index: SLOT,
                value: 12
            }
        ),
        "and so is one behind it"
    );
    assert!(
        !pending.discharges(
            SLOT,
            StampWait {
                index: SLOT,
                value: 21
            }
        ),
        "a wait past the latched value is not discharged by it"
    );
    assert!(
        !pending.discharges(
            SLOT,
            StampWait {
                index: SLOT + 1,
                value: 1
            }
        ),
        "a wait on another slot is not this drain's to answer"
    );
    assert!(
        !PendingStamp::default().discharges(SLOT, met),
        "a drain that has latched nothing discharges nothing"
    );
}
