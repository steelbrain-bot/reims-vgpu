# `dead/` — disconnected legacy source

This directory is **not a module.** No `mod` declaration reaches it, nothing here
compiles, nothing here is feature-gated, and nothing here is linkable. It is
source to read.

## Why it exists

`AGENTS.md` and the replacement plan's Seam 6 say a group's legacy counterpart is
*deleted* in the commit that wires the group. The project owner amended that to
**disconnect instead of delete**, for one reason: when a live boot regresses on a
packet class that has just moved, the question is "what did the old arm do", and
`git log` over a 400k-line tree is a poor way to ask it. Unreachability is what
the prohibition on two semantic models actually needs, and removing the `mod`
declaration gives that by construction rather than by a flag.

## The rules, and they are not negotiable

- **Nothing is resurrected from here.** Not by a `mod`, not by a feature, not by
  a copy-paste. A regression is read here and *fixed in the new owner*.
- **No build-time or run-time switch may reach `dead/`.** A `#[cfg]` that turns a
  file here back on is the second semantic model the plan forbids, wearing a
  different hat.
- **Every move appends a row below**, naming what moved, the commit that
  replaced it, and the owner-level tests that replaced the legacy tests moving
  with it. Those tests stop running the moment they move; a group that ships
  without replacement coverage has silently lost it, and the row is where that
  is checked.
- **`dead/` is deleted wholesale, in one commit,** once every group has moved and
  the Seam 6 gates are green. It is not pruned incrementally — a half-emptied
  `dead/` reads as "these were the ones worth keeping", which is the opposite of
  what it means.

## Register

Rows are ordered as they landed. A row's **step** is either a *wiring step*
(`W`n) — a pure translation or plan module whose legacy caller keeps its
ownership and loses only its duplicate arithmetic, per the plan's 2026-09-01
amendment — or a *group* (`G`n), a set of packet classes moving to
`DeviceTransaction` together under the 2026-09-02 decomposition amendment. The
two are not the same claim and are not recorded as if they were: a wiring step
moves a decision, a group moves a lifetime.

| Step | Moved | From | Replaced by | Owner-level tests that replaced the moved ones |
|---|---|---|---|---|
| W1 | `child_deprecated_ops.rs` — `CHILD_DEPRECATED_OPS`, `is_deprecated_child_opcode`, and the ledger test that walked them | `model/regs.rs`, `runtime/decode/ledger.rs` | `reims_vgpu_core::control::ControlKind::of` ⇒ `RetiredSlot`, read through `runtime::drain::inert_control_kind`. **What moved is the classification, not the envelope**: the drain still owns the stamp waits, the ordering position and the completion word for these packets, so this is a wiring step and not the inert class's cutover. | `runtime::decode::ledger::packets::the_ledger_judges_a_retired_set_at_all` (new — asserts the set's size against the reference host rather than against a transcription of the ledger); `runtime::decode::ledger::packets::no_live_command_is_also_a_retired_slot` and `no_ledger_row_names_a_command_this_device_cannot_receive` (re-sourced off the ledger, replacing the `const` assertion in `model::regs` that went with the table); `runtime::drain::tests::a_retired_slot_is_reported_as_retired_and_not_as_undecodable` (re-sourced off the ledger, so it now covers every judged slot rather than every transcribed one) |
