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
| W1 | `child_deprecated_ops.rs` — `CHILD_DEPRECATED_OPS`, `is_deprecated_child_opcode`, and the ledger test that walked them | `model/regs.rs`, `runtime/decode/ledger.rs` | `reims_vgpu_core::control::ControlKind::of` ⇒ `RetiredSlot`, read through `runtime::drain::is_retired_control_slot`. **What moved is the classification, not the envelope**: the drain still owns the stamp waits, the ordering position and the completion word for these packets, so this is a wiring step and not the inert class's cutover. | `runtime::decode::ledger::packets::the_ledger_judges_a_retired_set_at_all` (new — asserts the set's size against the reference host rather than against a transcription of the ledger); `runtime::decode::ledger::packets::no_live_command_is_also_a_retired_slot` and `no_ledger_row_names_a_command_this_device_cannot_receive` (re-sourced off the ledger, replacing the `const` assertion in `model::regs` that went with the table); `runtime::drain::tests::a_retired_slot_is_reported_as_retired_and_not_as_undecodable` (re-sourced off the ledger, so it now covers every judged slot rather than every transcribed one) |
| W2 | *(nothing moved)* | `runtime/drain/mod.rs` — the inline `if opcode == TAHOE { WithKeyLimit } else { WithoutKeyLimit }`, and the two arms that each named their own request decoder | `reims_vgpu_core::query::QueryKind::of` + `query::request_words`, joined by `runtime::drain::query_request` | No row content moves: the legacy counterpart was an expression inside a live match arm, not a file, so there is nothing `dead/` can hold. Recorded anyway so the register is the list of wiring steps and not only the list of files. New coverage: `runtime::decode::ledger::packets::the_query_arms_and_the_models_questions_are_the_same_four` — the check the drain cannot make for itself, since it sees one opcode at a time. |
| W3 | *(nothing moved)* | `runtime/drain/mod.rs` — the inline `packet.opcode == ROOT_OP_DEFINE_FIFO` that decided which end of a channel lifetime a packet was, and the silent `Ok(_) => {}` beside the range gate | `reims_vgpu_core::control::resolve` ⇒ `ControlOp::Channel { transition, domain }` | Again an expression rather than a file. New coverage: `runtime::drain::tests::a_channel_lifetime_outside_the_channel_range_is_reported` — the drop the old arm made in silence. |
| W4 | `event_decoder.rs` — all of `runtime/decode/event.rs` (the decoder, its `Command`/`Kind`/`DecodeStatus`, the refused-opcode list and its five tests), plus `runtime::fence_exec::tests::event_wait_timeout_unsupported` and the ledger test that drove it | `runtime/decode/event.rs` (whole file), `runtime/decode/mod.rs` (`pub mod event;`), `runtime/fence_exec.rs`, `runtime/decode/ledger.rs`, `runtime/plan/event_sync.rs` | `reims_vgpu_protocol::decode::sync::decode` for the lift, called from `runtime::exec::handle_event_record`; `reims_vgpu_protocol::sync::EventKind` for the two kinds the wire has. `plan_event` lost its `has_timeout` parameter and `Decision`/`Reason` lost their `WaitTimeoutUnsupported` variants — the settled row is refused where the ledger settled it, not a second time in the planner. Still a wiring step and not a group: `execute_event` keeps the fence-generation store and `core::resolve::event` is **not** wired, because it resolves an event ref through a namespace this device does not hold events in. | `runtime::exec::tests::a_bounded_event_wait_is_refused_by_contract_and_leaves_the_generation_alone` (new — the bounded wait's refusal reaches the failure channel from the segment walk, under the row's own name, and the signal ahead of it still lands); `runtime::decode::ledger::the_event_rail_lifts_its_records_and_refuses_its_one_settled_row` (replaces the per-rail decoder cross-check, and adds the claim the old one could not make: exactly one event row is `RefusedByContract` rather than `Unjudged`); `golden_vectors::event_signal_golden` (same bytes, re-pointed at the decoder in production) |

