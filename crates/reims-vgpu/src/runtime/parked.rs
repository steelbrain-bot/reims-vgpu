//! The bytes an admitted packet is executed from, held while the model decides
//! when it runs.
//!
//! # Why there is a store at all
//!
//! Today a drain loop that cannot run a packet leaves the ring head where it
//! is, so the bytes stay in the guest's ring and the whole timeline stops
//! behind them. The semantic model replaces that: `SessionModel::admit` gives
//! an unready packet an ordering position and lets later packets proceed, and
//! `take_ready` hands the position back when it may run. Between those two
//! moments the ring has been consumed and the guest may have overwritten it, so
//! whatever the execution reads has to have been taken out of the ring first
//! and kept.
//!
//! # Why it belongs to the device and not to the model
//!
//! `reims_vgpu_core` holds ordering, admission, hazards and publication; it can
//! name no guest-RAM pointer and no decoded device payload, and its dependency
//! list is that claim. An `IngressOrdinal` is the model's, the bytes are this
//! device's, and this map is the join — which is exactly the shape that keeps
//! the model free of them.
//!
//! # What the store owns, and what it deliberately does not
//!
//! It owns the *identity* of a parked position: at most one entry per ordinal,
//! taken out exactly once. It does **not** decide whether an ordinal may run —
//! that is `take_ready`'s — and it does not know whether an ordinal was
//! admitted, because only the model can answer that.
//!
//! It also owns the accounting the switch introduces. Leaving the head unmoved
//! used to be this device's backpressure; parking removes it, so a position the
//! model never releases becomes retained host memory instead of a stalled ring.
//! That is not a reason to refuse work — the contract provides no lawful loss
//! here — but it is a reason to make the retention *countable*, which
//! [`ParkedWork::retained_bytes`] and [`ParkedStore::retained_bytes`] are.

use std::collections::BTreeMap;

use reims_vgpu_core::identity::IngressOrdinal;

use crate::runtime::drain::Packet;
use crate::runtime::exec::ExecSubmission;

/// One admitted packet's retained inputs.
///
/// The decoded packet is always here; it owns its payload `Vec<u8>`, so
/// retaining it is retaining the bytes. The submission is here only for the
/// exec class, because it is the only class whose inputs are *not* all in the
/// packet: `read_submission` loads the command buffers out of task GVA, and a
/// device that re-read them at execution would walk whatever the guest put
/// there after the packet was admitted.
pub struct ParkedWork {
    /// The ordering domain the packet arrived in, which is the channel its
    /// execution and its completion word belong to. Carried because the ordinal
    /// alone does not spell it and the model hands back ordinals.
    domain: u32,
    packet: Packet,
    submission: Option<ExecSubmission>,
}

impl ParkedWork {
    /// Retain a packet that names no host-side inputs beyond its own payload.
    #[must_use]
    pub const fn new(domain: u32, packet: Packet) -> Self {
        Self {
            domain,
            packet,
            submission: None,
        }
    }

    /// Retain an exec packet together with the command buffers already read out
    /// of guest memory for it.
    #[must_use]
    pub const fn with_submission(domain: u32, packet: Packet, submission: ExecSubmission) -> Self {
        Self {
            domain,
            packet,
            submission: Some(submission),
        }
    }

    #[must_use]
    pub const fn domain(&self) -> u32 {
        self.domain
    }

    #[must_use]
    pub const fn packet(&self) -> &Packet {
        &self.packet
    }

    /// The command buffers this packet was admitted against, for the exec class.
    #[must_use]
    pub const fn submission(&self) -> Option<&ExecSubmission> {
        self.submission.as_ref()
    }

    /// Host bytes this position is holding.
    ///
    /// The payload and the command buffers, which are the two allocations
    /// parking keeps alive; the stamp-wait records are bounded by the header's
    /// `u16` count and are counted with them rather than being a third term
    /// nobody could act on separately.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.packet.payload.len()
            + self.packet.stamp_waits.len()
                * std::mem::size_of::<crate::runtime::drain::StampWait>()
            + self
                .submission
                .as_ref()
                .map_or(0, ExecSubmission::retained_bytes)
    }
}

/// Why a parked position was let go without running.
///
/// Named rather than collapsed into one `remove`, because the two reach this
/// store from different model calls and a store that could not tell them apart
/// would report a device loss as ordinary progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Release {
    /// `take_ready` said it may run; the bytes go to the executor.
    Ready,
    /// The model withdrew it — a device loss, a closed generation's pipeline
    /// lease, or a refusal the caller is naming on its failure channel. The
    /// bytes are dropped and nothing runs.
    Withdrawn,
}

impl Release {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Ready => "parked_taken_ready",
            Self::Withdrawn => "parked_withdrawn",
        }
    }
}

/// Every admitted position whose work has not run yet.
///
/// Keyed by `IngressOrdinal` and not by channel head: the ordinal is what the
/// model hands back, and the head has moved on by the time it does. That is
/// also why a held-packet cache keyed on a channel's consumer pointer is not
/// this and could not become it.
#[derive(Default)]
pub struct ParkedStore {
    work: BTreeMap<IngressOrdinal, ParkedWork>,
    retained_bytes: usize,
    /// The most bytes ever parked at once, so a boot can report the retention
    /// the switch introduced instead of only its instantaneous value.
    peak_bytes: usize,
    peak_len: usize,
}

impl ParkedStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold a position's inputs until the model releases it.
    ///
    /// # Panics
    ///
    /// If the ordinal is already parked. An `IngressOrdinal` is issued once per
    /// admission and never reused, so a collision is this device parking one
    /// arrival twice — which would drop the first packet's bytes and run the
    /// second's under the first's ordering position. There is no answer that
    /// keeps both, so it is not an error a caller could handle.
    pub fn park(&mut self, ingress: IngressOrdinal, work: ParkedWork) {
        let bytes = work.retained_bytes();
        assert!(
            self.work.insert(ingress, work).is_none(),
            "ingress ordinal parked twice"
        );
        self.retained_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.retained_bytes);
        self.peak_len = self.peak_len.max(self.work.len());
    }

    /// Take a position's inputs out, once.
    ///
    /// Consuming, and that is the whole guard against a transaction running
    /// twice on this side: the model's `take_ready` is the one door work leaves
    /// by, and this is the one door its bytes leave by. A borrowing lookup
    /// beside it would be a second door.
    ///
    /// `None` for an ordinal this store never held, which is not an error here:
    /// a class that is admitted and executed in the same breath never parks,
    /// and it still completes through the same call.
    #[must_use = "a parked position taken out and not run is a packet that never runs"]
    pub fn release(&mut self, ingress: IngressOrdinal, why: Release) -> Option<ParkedWork> {
        let work = self.work.remove(&ingress)?;
        self.retained_bytes = self.retained_bytes.saturating_sub(work.retained_bytes());
        crate::runtime::drain::note_store_route(why.slug());
        match why {
            Release::Ready => Some(work),
            // Dropped here rather than handed back, so a withdrawal cannot be
            // spelled as a run by a caller that ignores the reason it passed.
            Release::Withdrawn => {
                drop(work);
                None
            }
        }
    }

    /// Whether this ordinal is holding bytes. For a caller deciding whether a
    /// completion names parked work, never for one about to run it.
    #[must_use]
    pub fn is_parked(&self, ingress: IngressOrdinal) -> bool {
        self.work.contains_key(&ingress)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.work.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.work.is_empty()
    }

    /// Host bytes every parked position is holding together.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// The high-water marks, as `(positions, bytes)`.
    #[must_use]
    pub const fn peak(&self) -> (usize, usize) {
        (self.peak_len, self.peak_bytes)
    }

    /// Drop everything, for the events that end every position at once — a
    /// device loss, a session reset that strands work, a channel teardown that
    /// takes its domain with it. Returns what was dropped, so the caller can
    /// say so on its failure channel; the ordinals come back because the guest
    /// is owed a typed reason per position and this is the only place they are
    /// still enumerable.
    #[must_use = "the guest is owed a typed reason for every position this dropped"]
    pub fn drain_all(&mut self) -> Vec<IngressOrdinal> {
        self.retained_bytes = 0;
        std::mem::take(&mut self.work).into_keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(payload: usize) -> Packet {
        Packet {
            opcode: 0x37,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: 0,
            payload: vec![0u8; payload],
            next_head: 0,
        }
    }

    const fn ordinal(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }

    /// The bytes handed back are the bytes parked, not a re-read of anything.
    #[test]
    fn a_parked_packet_comes_back_whole() {
        let mut store = ParkedStore::new();
        let mut p = packet(8);
        p.payload[3] = 0xab;
        store.park(ordinal(1), ParkedWork::new(5, p));

        let taken = store
            .release(ordinal(1), Release::Ready)
            .expect("parked a moment ago");
        assert_eq!(taken.domain(), 5);
        assert_eq!(taken.packet().payload[3], 0xab);
    }

    /// One position's bytes leave by one door, once.
    #[test]
    fn a_position_is_taken_out_once() {
        let mut store = ParkedStore::new();
        store.park(ordinal(2), ParkedWork::new(1, packet(4)));

        assert!(store.release(ordinal(2), Release::Ready).is_some());
        assert!(store.release(ordinal(2), Release::Ready).is_none());
        assert!(store.is_empty());
    }

    /// A withdrawal is not a run: the store empties and hands nothing back, so
    /// a caller that ignored its own reason cannot execute the packet anyway.
    #[test]
    fn a_withdrawal_hands_nothing_back() {
        let mut store = ParkedStore::new();
        store.park(ordinal(3), ParkedWork::new(1, packet(4096)));

        assert!(store.release(ordinal(3), Release::Withdrawn).is_none());
        assert!(!store.is_parked(ordinal(3)));
        assert_eq!(store.retained_bytes(), 0);
    }

    /// The retention the switch introduces is countable while it is held and
    /// after it is let go.
    #[test]
    fn retention_is_counted_up_and_down() {
        let mut store = ParkedStore::new();
        store.park(ordinal(1), ParkedWork::new(1, packet(1000)));
        store.park(ordinal(2), ParkedWork::new(1, packet(24)));
        assert_eq!(store.retained_bytes(), 1024);
        assert_eq!(store.peak(), (2, 1024));

        let _ = store.release(ordinal(1), Release::Ready);
        assert_eq!(store.retained_bytes(), 24);
        // The peak is a high-water mark and does not follow the release down.
        assert_eq!(store.peak(), (2, 1024));
    }

    /// Positions come out in ingress order, because that is the order the guest
    /// is owed its typed reasons in.
    #[test]
    fn everything_dropped_at_once_comes_back_in_order() {
        let mut store = ParkedStore::new();
        for n in [7, 2, 5] {
            store.park(ordinal(n), ParkedWork::new(1, packet(16)));
        }

        assert_eq!(store.drain_all(), vec![ordinal(2), ordinal(5), ordinal(7)],);
        assert!(store.is_empty());
        assert_eq!(store.retained_bytes(), 0);
    }

    /// An ordinal is issued once per admission; parking one twice would run the
    /// second packet under the first's ordering position.
    #[test]
    #[should_panic(expected = "ingress ordinal parked twice")]
    fn one_ordinal_cannot_hold_two_packets() {
        let mut store = ParkedStore::new();
        store.park(ordinal(4), ParkedWork::new(1, packet(4)));
        store.park(ordinal(4), ParkedWork::new(1, packet(4)));
    }
}
