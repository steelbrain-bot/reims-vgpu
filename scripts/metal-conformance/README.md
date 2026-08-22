# metal-conformance

A self-verifying Metal battery that runs the same source on a native macOS host
and inside the guest, and names the seam when the two disagree.

Every result this project had about Maps' missing type layer was scored by
opening a screenshot, and a screenshot names no seam: "labels absent" is what a
wrong pitch, a wrong swizzle, a dropped dispatch and a lost mip level all look
like. Each case here computes a value the CPU can predict exactly, asks the GPU
for it, and compares — so a failure names the case, the bytes wanted and the
bytes returned.

```
CASE <name> PASS|FAIL|SKIP <detail>
SUMMARY cases=N failures=N skipped=N
```

## Reading a result

The comparison between the two hosts is the whole instrument:

| native | guest | meaning |
|---|---|---|
| PASS | PASS | nothing to see |
| PASS | FAIL | a named device defect |
| FAIL | — | a wrong expectation in `conformance.swift`, not a finding |
| — | SKIP | the device's own reported limits make the case inexpressible |

`baseline-native.txt` is the native run. Re-record it whenever a case is added,
on an Apple host, and never treat a guest failure as a finding until the same
case is green there.

A `SKIP` is not a soft failure. `minimumLinearTextureAlignment` is 16 on an
M-series device and 256 on Apple's paravirtual one, so a literal pitch from a
guest census is not expressible on every host — Metal itself rejects the
descriptor. The battery also derives padded pitches from whatever the running
device reports, so every host runs padded-pitch cases whatever its limit.

## Running it

Native, on a macOS host:

```sh
swiftc -O conformance.swift -o conf && ./conf
```

In the guest, from the repository root — this boots a rail, runs the battery and
collects the device's own fail log beside the results:

```sh
scripts/metal-conformance/conformance-run.sh /tmp/conf-out
```

Environment passes through, so an arm is
`REIMS_VGPU_GUEST_IMPORT=off scripts/metal-conformance/conformance-run.sh /tmp/conf-off`.
`RAIL=macos-11 …` picks a rail.

The runner builds in the guest where the rail has `swiftc` and otherwise ships
`conformance-x86_64`, a Mach-O built on an Apple host. Rails without developer
tools are the normal case, so keep that binary current when a case is added.

## A refusal is not a mismatch, and the kernel is what says which

This device refuses a dispatch on the *host* side, so the guest's command buffer
completes clean and the output buffer keeps whatever was in it. Left alone,
every case here then compares its sentinel fill against what it wanted and
reports a **content** failure — "the device returned the wrong bytes" for a
device that returned none.

The `offset_oracle` cases showed how far that misleads. Their fill is
`1 + (i % 251)` and zero means "a byte nothing in this buffer ever held", but
`readBack`'s `0xEE` sentinel is 238 — inside the fill's own range. A refused
dispatch inverted to a constant source offset of 237, every texel landed in a
different delta bucket, and four cases reported `absent=0 shifted=4080`: a
precise, plausible account of a defect that did not exist.

No sentinel value fixes this. A battery covering many formats has no byte that
is out of range for all of them. So each readback kernel writes `ran[0] = 1u`
into a buffer of its own, **before** its grid guard — so the witness says the
kernel was reached rather than that some thread was in range — and `readBack`
returns `nil` when that word stays zero. A case that gets `nil` calls `refused`,
which prints one wording for all of them.

Two consequences when adding a case:

- **A new readback kernel must take `device uint *ran [[buffer(4)]]` and write
  it first.** A kernel without the witness reports every refusal as a content
  failure, which is the state this section exists to describe.
- **A case that cannot run because another case did not must `skipDependent`,
  not vanish.** The case *count* is what a reader diffs two runs by, and a name
  that stops appearing reads as a deleted case. One refusal of
  `incremental_a8_first_read` silently took three other names out of the totals
  before this was added.

## Reading it beside the device log

`conformance-run.sh` copies `/tmp/reims-vgpu-fail.log` to `device.log` in the
output directory. A case that fails with a refusal in that log is an
unimplemented rail refusing by name; a case that fails with *nothing* in the log
is silent loss, which is the worse of the two and the one worth chasing first.

## A `.private` render target does not reach the target-import rail

Every render target in sections A-H is `rd.storageMode = .private`, which means
the device allocates it and the guest never names the pages behind it. That is
half the render targets a compositing app uses. The other half are `.shared`:
layers the CPU also rasterizes into, or that another process composites, whose
bytes are guest memory the device may bind a Vulkan image *directly over*
instead of copying.

Those are two rails, and for a long time this battery had exactly one case on
the second of them — `cpu_write_after_render`, section F5. A whole rail behind
one case is not coverage. It is a single sample that happens to pass, and it is
what let a live defect sit underneath 173 green cases: a driven Maps boot lost
its entire type layer on the arm where guest-backed targets are imported, and
nothing in this file could see it, because nothing in this file created a
guest-backed target except that one case.

Section I is that coverage. When adding a render-target case, ask which rail it
is on and say so in the case name — `srt_` is the guest-backed prefix. A new
case that reaches for `.private` because that is what the case above it used has
tested the rail that already had coverage.

The widths there (60, 256, 1000) are not decoration. 60 and 1000 texels are 240
and 4000 bytes, neither a multiple of the 256-byte linear alignment this device
reports, so a rail that confuses the guest's stride with a padded one has
somewhere to show it. A case list of only round widths cannot see a shear.

## A case that is never called reports nothing, and the totals do not notice

`cpu_write_after_render_256x64` was absent from every run of this battery, on
both arms, because the invocation list called three of its four arms. Nothing
flagged it: `ran` counts cases that reported, so a case that was never invoked
is indistinguishable from one that does not exist.

That is the same failure mode as the `refused(); return` bug that `skipDependent`
exists to fix, one level up — there the case vanished from the totals mid-run,
here it never entered them. When adding a parameterized case, count the
invocations against the parameter grid before trusting a green summary.
