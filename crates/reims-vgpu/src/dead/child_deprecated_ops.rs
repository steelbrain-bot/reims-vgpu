// DISCONNECTED SOURCE — this file does not compile, is not feature-gated, and
// is not linkable. No `mod` declaration reaches `dead/`. It is here to be read.
//
// What this was: the drain's own transcription of the fifteen child opcodes the
// reference host routes to its shared deprecated handler, and the predicate the
// dispatcher matched them with. It lived in `crate::model::regs` beside this
// device's live command constants.
//
// Why it went: `reims_vgpu_protocol::packets::LEDGER` judges those same fifteen
// rows `Closure::ProvenNoOp`, and `reims_vgpu_core::control::ControlKind::of`
// turns that judgement into `ControlKind::RetiredSlot`. Two lists of one set is
// two places a slot that goes live has to be remembered, and the cost of
// forgetting one is a live command swallowed by a handler that does nothing.
//
// Do not resurrect any of this. When a boot regresses on a retired slot, read
// it to learn what the old arm did and fix `reims_vgpu_core::control` or the
// ledger row.

/// The opcodes the reference host routes to its one shared deprecated handler.
///
/// These are real slots with a real handler, not holes: the host accepts the
/// packet, does nothing with the payload and retires the stamps. They are
/// listed rather than folded into the unknown arm so a guest that still emits
/// one is reported as "sent a retired command" and not as "sent something this
/// device cannot decode" — the first is expected of an older guest and the
/// second is a gap in this device.
///
/// `0x2d` is in this set on the reference host, and is also
/// [`ROOT_OP_DEVICE_INFO_MONTEREY`]. Both are true: the Monterey-era device-info
/// command was retired in favour of [`ROOT_OP_DEVICE_INFO_TAHOE`] (`0x3a`), and
/// this device still answers it for a guest old enough to ask. The root arm runs
/// first, so the deprecated arm never sees it on the root channel.
pub const CHILD_DEPRECATED_OPS: [u16; 15] = [
    0x03, 0x1f, 0x21, 0x23, 0x24, 0x26, 0x27, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x32,
];

/// True for the opcodes in [`CHILD_DEPRECATED_OPS`].
///
/// A function rather than a match arm because a `match` cannot pattern on an
/// array's contents, and spelling the fifteen numbers a second time in a pattern
/// is exactly the duplicated-constant mistake `no_two_child_opcodes_share_a_number`
/// exists to catch.
#[must_use]
pub const fn is_deprecated_child_opcode(opcode: u16) -> bool {
    let mut i = 0;
    while i < CHILD_DEPRECATED_OPS.len() {
        if CHILD_DEPRECATED_OPS[i] == opcode {
            return true;
        }
        i += 1;
    }
    false
}

// ---- The legacy test that moved with it, from `runtime::decode::ledger` ----
//
// Circular the moment the table above went: it asked the ledger whether it
// agreed with a transcription of itself. Replaced by
// `the_ledger_judges_a_retired_set_at_all` in the same module — which asserts
// the set's size against the reference host rather than against a copy — and by
// `a_retired_slot_is_reported_as_retired_and_not_as_undecodable` in
// `runtime::drain::tests`, now driven off the ledger and so covering every
// judged slot rather than every transcribed one.
    /// The retired slots are commands too — the reference host has one handler
    /// for all fifteen — so each is a row, and a row that says the shared
    /// handler's behavior *is* the contract rather than a gap.
    #[test]
    fn every_retired_slot_has_a_ledger_row() {
        for op in CHILD_DEPRECATED_OPS {
            let row = find(Channel::Child, op)
                .unwrap_or_else(|| panic!("retired slot {op:#04x} is unjudged"));
            assert!(
                matches!(
                    row.closure,
                    reims_vgpu_protocol::closure::Closure::ProvenNoOp { .. }
                ),
                "retired slot {op:#04x} reads as {} — the reference host's shared handler is the \
                 whole contract, so anything else claims this device owes more or less than the \
                 host it is imitating",
                row.closure.name()
            );
        }
    }

