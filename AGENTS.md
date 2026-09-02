# AGENTS.md

This project emulates Apple's paravirtualized GPU. The unmodified macOS guest supplies the driver;
QEMU and the Rust backend decode its command stream and execute it through Metal or Vulkan.

Nested `AGENTS.md` files apply within their directories.

## Build the contract into the structure

- Implement behavior from decoded fields, layouts, calling conventions, or host capabilities. If a
  decision-affecting contract term is unknown, recover it or return a typed refusal; do not guess.
- Put each invariant in the type, resolver, transaction, or state machine that owns it. Change the
  owner when it cannot express the invariant instead of adding call-site flags, duplicate lookups,
  fixup passes, or special cases around it.
- Parse guest ordinals once into total Rust types and carry those types. Keep page geometry explicit
  through `page_shift` or `page_size`, and derive constants from the contract.
- Prefer making invalid states unrepresentable, deriving duplicated values from one source, and
  testing behavior at the owner's boundary.
- Gate optional paths on structural capabilities, not device names. Environment overrides may
  narrow capability but never widen it.

Conformance records compatibility; it does not define an unknown API contract. Static inspection
of locally available binaries is acceptable for recovering missing contracts, but third-party
bytes, disassembly, extracted assets, and binary provenance must remain local and uncommitted.
Persist only the resulting field, layout, lifetime, ordering, or calling-convention contract.

## Ownership boundaries

- `vendor/qemu`: thin QEMU device shim for QOM, MMIO/BAR, IRQ/MSI, display integration, and
  `HostOps` plumbing.
- `crates/reims-vgpu`: device model, decode, mapping, scheduling, command execution, presentation,
  and backend policy.
- `crates/reims-vgpu-wire`: derived wire views; its nested instructions also apply.
- `crates/reims-vgpu-protocol`: `no_std`. The first layer allowed to assign meaning to a wire tag —
  backend-neutral layouts, formats, geometry, page arithmetic, closed ordinal rules, and the
  contract refusals they name.
- `crates/reims-vgpu-core`: the backend-independent semantic model. Transactions, dependency and
  publication domains, session generations and device epochs, the fixed executor, and the serial
  reference interpreter every parallel schedule is checked against. It can name no Vulkan handle,
  Metal object, QEMU structure or guest-RAM pointer, and its dependency list is that claim.
- `crates/reims-vgpu-memory`: the guest-RAM bound — imports, checked slices, page footprints.
- `crates/reims-vgpu-config`: the single parse and declaration point for operator switches. An
  override may only narrow what the device does; it may never widen it.
- `crates/reims-vgpu-observe`: typed observations and refusals. It describes decisions but does not
  select behavior.
- `crates/reims-vgpu-vulkan`: the Vulkan rail — the only layer that turns host
  capabilities into placement and transfer policy. It sees `ash`, the semantic
  model, the refusal vocabulary and the operator switches, and nothing of QEMU,
  the device model, guest-RAM ownership or decode.
- `crates/reims-vgpu-testkit`: shared behavioral fixtures and test instruments —
  where the oracle's capture is, whether it is there, how the suites that read it
  read it, and the allocation counter the structural-zero suites measure with.
- `conformance`: native-oracle and guest-visible compatibility cases.
- `vm`: rail-selected, snapshot-reverting boot harnesses.

Product logic belongs in Rust. C and Objective-C connect QEMU to Rust and must not reconstruct
policy from multiple Rust queries. Shared C/Rust constants need a `qemu::abi::header_define` test.
No panic may cross an FFI boundary.

Guest RAM bounds and provenance belong to `crates/reims-vgpu-memory`; extend `GuestRamImport` or
`GuestSlice` instead of exposing raw pointers and offsets. Resource state that represents guest
work follows the contract-owned guest lifetime. Do not silently evict it with arbitrary cache
bounds; refuse excess work explicitly when the contract provides no lawful loss.

For asynchronous work, the owning transaction must retain inputs through host completion, make
results visible before the completion word or interrupt, and prevent callbacks or memory access
after the guest may release or reuse resources. Submission is not completion.

Unknown, dropped, rejected, degraded, or unsupported guest work must produce a typed reason on the
always-on failure channel. Expected not-ready control flow stays quiet. Read environment variables
only through the configuration owner.

## Supported pathways

| Pathway | Host | Guest | Attach | Page shift | Backend |
|---|---|---|---|---|---|
| x86 macOS | Linux x86_64 KVM | x86_64 macOS | PCI | 12 | Vulkan |
| arm64 macOS | Apple Silicon HVF | arm64 macOS | sysbus | 14 | Metal |
| arm64 macOS | Apple Silicon HVF | arm64 macOS | sysbus | 14 | Vulkan/MoltenVK |
| arm64 macOS | Apple Silicon HVF | arm64 macOS | sysbus | 14 | both, chosen at run time |

The last row is one binary carrying both rails, built with
`scripts/qemu-build/qemu-build.sh --backend both` and selected per run with `REIMS_VGPU_RAIL`.
It exists so the same guest stream can be executed through native Metal and through MoltenVK
without a rebuild, which is the only way to attribute a wrong frame to metal2vulkan rather than to
this device. It is a measurement configuration, not a shipping one; the three rows above it are
what ships.

**A `cfg` may answer only "what did this build compile". It may never answer "which rail is
running".** Those are different questions on the fourth row and the compiler cannot tell them
apart: a `not(feature = "backend-vulkan")` block meaning "the Metal arm" disappears silently when
both features are on, and a `feature = "backend-vulkan"` block meaning "the running rail" executes
against Metal. Every decision the running rail makes goes through `backend::Backend`, whose
implementation `backend::select` picks once per process. Tests obey the same rule: name the rail
(`MetalBackend`, `VulkanBackend`) when the test is about one, and ask `selected().rail()` when the
expected answer legitimately differs.

Do not generalize observations between architectures, backends, memory topologies, host GPU
classes, or guest rails. Vulkan 1.2 is the baseline; newer functionality requires a
capability-gated fallback. Host-pointer import is optional, and guest-visible semantics must be the
same on imported and copying paths.

## Wire a finished subsystem immediately

The replacement architecture is landing subsystem by subsystem, and each one **joins production in
the commit that finishes it**. A subsystem that exists but is reachable only from its own tests has
not been verified against a guest; a release that switches thirty of them at once has no bisect
that can attribute a regression to one. So a subsystem is done when its legacy counterpart is gone
or delegates to it — not when it compiles.

What this does *not* license is two semantic models running at once. No per-packet feature switch
choosing between executors, no shadow execution that mutates state twice, no adapter translating
between an old model and a new one. A call site that *replaces* its own logic with a call into the
owning crate creates no second model, and is the shape to reach for.

Wire in order of how little state moves. A pure translation or plan module — a function of its
inputs returning a value — wires first: the caller keeps its ownership and loses only its duplicate
arithmetic, so a regression points at one table. Modules owning handles, caches, or submission
lifetimes wire once the model that owns those lifetimes is in place, because they cannot be split.

The same rule decides how the final ingress switch is cut. **A packet class moves to the new model
alone exactly when nothing it owns is ordered against, or shares mutable state with, anything still
on the legacy path. Classes that do share such state move together, and that group — not
"everything" — is the atomic unit.** Two models are dangerous because they can disagree about one
piece of state; where there is no shared state there is nothing to disagree about, and holding a
disjoint class back buys no safety while costing the bisect it could have provided.

State the disjointness before moving a group and make it hold structurally — named owners, not an
argument that the current call sites happen not to overlap. If it cannot be made to hold, the group
is larger than it looked: enlarge it rather than move anyway. Within a group the switch is atomic
and the legacy counterpart is disconnected in the same commit. Order the groups by how little state
they move; the ordering and publication core is last, and carries the scheduler's deletion.

**Disconnect, do not delete — amending "the legacy counterpart is deleted in the same commit"
above, and the plan's Seam 6.** In the commit that wires a group, move its legacy files to
`crates/<crate>/src/dead/` and remove every `mod` declaration that reaches them. `dead/` does not
compile, is not feature-gated, and is not linkable; it is source to read. Unreachability by
construction is what the prohibition on two semantic models needs, and it is what a removed `mod`
gives — a flag or a `cfg` would give the prohibition's opposite.

Each move appends a row to `crates/<crate>/src/dead/README.md` naming what moved, which commit
replaced it, and which new owner-level tests replaced the legacy tests that moved with it. Those
tests stop running the moment they move, so a group that ships without that replacement coverage
has silently lost it and the row is where that is caught.

Nothing is resurrected from `dead/`, and no build-time or run-time switch may reach it. When a live
boot regresses, `dead/` is read to learn what the old code did and the fix lands in the new owner.
`dead/` is deleted wholesale, in one commit, once every group has moved and the plan's gates are
green — never pruned incrementally, because a half-emptied `dead/` reads as "these were the ones
worth keeping".

## Working and verification

Use a workflow proportionate to the change:

1. Reproduce or otherwise identify the behavior being changed.
2. Establish the relevant contract and its owner. Use focused instrumentation when necessary.
3. Add an owner-level regression test when the invariant is testable there.
4. Implement the invariant in its owner.
5. Format and run the focused and affected tests. Run GPU-touching Rust tests serially.
6. For guest-visible translation or rendering changes, run the relevant conformance case and live
   pathway. Use the broader conformance suite when the change can affect unrelated cases.

Mechanical changes do not require VM or conformance work unless they can change behavior. A defect
seen only in an unmodified guest requires a live guest check. Intermittent failures require repeated
runs before claiming a fix. State what was actually verified without treating one pathway as proof
of another.

For VM work, select and report the rail explicitly. Ensure an older VM cannot answer probes, clear
`/tmp/reims-vgpu-fail.log` before a new evidence run, and preserve the useful crash or serial output
before rebooting. Use host-driven input and host-owned frame capture for visual interactions.

Follow `conformance/README.md` when changing or running conformance. Native results establish the
expected Metal behavior; existing classified guest failures are compatibility debt, not permission
to add or hide regressions.

Run formatting, tests, clippy, feature checks, documentation checks, and live validation only for
the workspaces, features, APIs, and pathways the change can affect. Do not weaken warnings or tests
to make a gate pass. Treat every commit as a release candidate: clear regressions exposed by the
change before committing rather than recording knowingly broken intermediate states.

## Repository safety

Existing dirty changes belong to the user. Preserve them and avoid unrelated edits. Do not use
checkout, switch, stash, reset, or restore to manufacture a baseline in the shared checkout; use an
isolated copy when a control build is needed. Create a new commit by default; amend an existing
commit only when the user specifically requests or authorizes it.

Do not commit guest images, firmware, captured shaders, AIR, SPIR-V, disassembly, extracted assets,
or other third-party binary material. Keep investigation artifacts in ignored locations. Commit
only task-related source and state validation results narrowly.
