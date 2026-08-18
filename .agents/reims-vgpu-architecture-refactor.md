# Reims vGPU Architecture Refactor Plan

Status: in progress; sanity-checked against the available static contract set, with the arm mapper
storage and kernel-lifecycle gap explicitly unresolved

Implementation checkpoint (2026-08-18):

- `reims-vgpu-protocol`, `reims-vgpu-core`, and `reims-vgpu-vulkan` now exist as real workspace
  boundaries. Semantic identities, texel layouts/image formats, the canonical resource graph,
  content authority, submission context, execution envelope, executor capabilities, resident
  service verdicts, memory topology policy, device selection, feature negotiation, host-pointer
  capability, and Vulkan format conversion have moved to their intended owners.
- Guest-memory runs, attachment backing, LOAD seeds, and guest-page destinations are owned by
  `reims-vgpu-memory`. Guest-page destinations retain the complete semantic storage format, so
  wide compute formats remain representable through direct writeback instead of being narrowed to
  the render-oriented texel-layout set. Primary, secondary, and sampled image requests retain
  semantic storage and transfer formats through execution; native formats are produced only inside Vulkan. Resolved
  render and compute requests, their semantic operands, samplers, bounded sources/destinations,
  residency generations, and typed results are now core-owned rather than declared by the Vulkan
  engine. Vulkan retains only native conversion and cache-key projections; the resolved submission
  envelope no longer contains a Vulkan-declared command payload. Compute storage and sampled-image
  format selection is also core-owned; Vulkan maps that selected format into a native view.
  Resolved linear and registered-surface texture backings, including their level geometry and
  typed mapping identity, are core-owned as well. Every implemented texture-copy family now enters
  execution as an immutable core command.
- Task resources receive generational core identities and explicit storage/view relations for task
  addresses, buffer ranges, registered surfaces, mapper storage, and heap placement. Submission
  context preserves the full ordered segment envelope and resource-participation snapshot.
- The product executor is device/session-owned. Command execution, capability discovery,
  resident-content service, guest-write synchronization, compute residency, readback, and
  presentation are distinct core contracts. Guest-RAM transfer, maintenance, session lifecycle,
  and telemetry are separate compatibility ports; the `Executor` body is now empty and composes
  those narrow services instead of owning their methods.
- Unified/discrete policy and the four import/topology cells are isolated in the Vulkan crate. The
  Vulkan engine and translation implementation now live there physically; the staticlib retains a
  compatibility re-export and telemetry adapter while callers are moved to the executor boundary.
- The PCI and MMIO adapters share console-ownership, cursor-position, cursor-glyph, input, guest-RAM
  span, and host-memory plumbing through the common QEMU shim. Raw console-feed, scanout-verdict,
  and cursor-glyph ABI calls each have one caller; bus files retain only attach, IRQ, surface-update,
  and pathway-specific console cadence. The frame shared with the host-window adapter carries a
  core-owned semantic presentation source rather than a Vulkan-declared request. The host-window
  adapter retains its device executor and performs native attach, resize, present, recovery
  classification, and detach through a session-owned presentation port; it no longer reaches the
  Vulkan engine facade directly.
- GPU Store, GPU-to-guest, and sampled draw/compute guest-to-GPU materialization facts now update
  canonical content authority using exact `(ResourceId, ContentVersion)` stamps returned by
  successful executor completion. Address, mapping generation, cache identity, or upload counters
  are not substitutes. Buffer blits, buffer↔texture copies, and resource-state operations now use
  the same immutable command envelope and completion-owned content stamps. Buffer↔texture executor
  requests carry generational buffer ranges and texture endpoints, semantic aspect, region, and
  strides rather than serializer ordinals. Ordinary texture-region copies now do the same while
  preserving the direct resident-to-guest optimization as executor policy. Multi-slice/multi-level
  copies are non-empty immutable batches: planning resolves storage without forcing guest bytes
  current, then execution either consumes the authoritative resident or explicitly settles the
  guest-byte fallback. Texture endpoint storage no longer belongs to the Vulkan runtime.
- Compute residency identity and service ownership are core contracts, and live mirrors are tied
  to guest lifetimes rather than an invented capacity. Backend resident leases are now owned by
  the device executor and keyed by weak semantic resource lifetimes, rather than stored on
  `TaskResource`; maintenance, reset, and executor teardown reap them. Guest-read debt,
  guest-write debt, and the outstanding writeback page footprint are session-owned signals, so
  one vGPU cannot make another wait or license a stale read; the hot no-debt query retains a
  thread-local session handle and does not acquire the process session-map lock. Completion-stamp
  pending/unsubmitted projections, per-FIFO pressure, and interrupt announcement hooks are also
  session-owned even though the physical timeline worker is shared. Host-window presenter
  attachment is published per session, including device-loss cleanup of inactive sessions. A
  leased readback carries its originating pool's return channel and outstanding count, so another
  session cannot consume and discard its return token or make teardown wait on the wrong device.
  Mapping/object mutations
  emit typed host-release effects instead of writing three unrelated retirement vectors, with
  guest imports retired before aliased host views. The semantic runtime no longer calls Vulkan
  engine operations directly; its remaining Vulkan dependency is the compatibility request/type
  vocabulary used to construct texture-blit and guest-transfer plans.
- The bounded memo audit distinguishes lifecycle state from derived performance data. The three
  byte-bounded CPU memos are byte-exact and revalidated on every lookup, so eviction costs only
  conversion/re-upload. The GVA cache can evict only entries whose bytes are already present in
  guest RAM; sole-copy entries remain admitted over the cap and fail-visible. These are retained
  under the project's explicit recomputation-cache exception, not treated as resource authority.
- Verification at this checkpoint: `reims-vgpu-core` (79 tests), `reims-vgpu-vulkan` with the
  supported host-window feature (722),
  focused executor ownership (11), drain census (7), and 1,317 of the full product library suite's
  1,318 tests pass serially. One
  shared guest-import fixture failed only in the full-suite order and passed immediately in
  isolation; it is not counted as evidence for or against the refactor. The supported Vulkan
  product configuration compiles. A no-backend build is intentionally rejected by the crate and
  is not a supported verification arm.

Scope: the Rust device model, protocol boundary, resource lifecycle, Vulkan executor, memory
topology policy, QEMU composition layer, presentation, observability, and verification architecture.

## Architectural verdict

The project's central problem is not unified versus discrete memory. Reims has no canonical
boundary between:

1. what a decoded guest command means;
2. which guest resource owns that meaning;
3. where the resource's current bytes live;
4. how a host GPU should transport those bytes; and
5. how execution completion changes guest-visible state.

Render, compute, blit, mapping, writeback, residency, and presentation reconstruct those concerns
independently. A performance optimization therefore edits lifecycle or coherency behavior as well
as transport behavior, allowing a unified-memory optimization to hurt a discrete host and the
reverse.

The target is:

> A backend-independent semantic core centered on a canonical resource graph, feeding one
> immutable execution IR through one real executor boundary, with unified and discrete memory
> implemented as placement and transfer policies inside the Vulkan executor.

Unified and discrete memory must not become two complete renderers. They are two narrowly scoped
policy modules sharing resource lifecycle, Vulkan execution, synchronization, completion, and
guest-visible semantics.

## Contract-grounding sanity check

### Evidence verdict

The available static set has an important coverage boundary. It contains a kernel/device lifecycle
for the PCI pathway and matching user-space serializer code for both instruction-set slices. It
therefore supports common serializer semantics and the PCI resource/page-table lifecycle, but it
does not supply the mapper-capable pathway's kernel object implementation. A method or feature name
present in both user-space slices is evidence for shared vocabulary, not evidence that the two
kernel pathways have identical storage or teardown behavior.

The sanity check used three evidence grades, and the plan must not silently promote one into
another:

1. Interface and symbol presence establishes that two concepts or operations exist separately. It
   does not establish ownership, ordering, success behavior, or that every advertised combination
   is reachable.
2. A followed call relationship establishes the ordering visible in that implementation. The
   mapping-release sequence below is in this grade; it is stronger than a method-name inventory
   and narrower than a cross-pathway rule.
3. Matching serializer selectors and record producers in both instruction-set slices establish a
   shared user-space vocabulary for those forms. They do not establish a shared kernel storage
   model. Mapper object ownership therefore remains unknown even where both slices name the same
   construction.

Two negative conclusions are architectural inputs too. First, the guest contract evidence does
not expose a unified-versus-discrete resource lifecycle; that classification is a host capability
and may only select executor placement and transfer plans. Second, no available mapper-side kernel
lifecycle establishes that a mapper reference is a page-table mapping, a registered surface, or an
owned backing. The core must represent the unknown relation without guessing any of those three.

The audit maps contract evidence to architecture as follows:

| Contract distinction | Architectural consequence | Confidence boundary |
|---|---|---|
| Task resource heap, resource-heap namespace, object handles, resources, memory maps, and page table are separate interfaces | Separate namespace, resource, storage, mapping, and address-space identities | Established on the PCI lifecycle; mapper kernel ownership remains open |
| Object-table entries expose a resource index while serializer object families expose their own create/get/delete-by-reference APIs | `ObjectTableRef` and `SerializerRef<T>` are different types and never key the same map | Established by static interface shape plus the existing driven-boot collision evidence; static names alone would not prove independence |
| Mapping commit allocates page-table coverage; mapping release synchronizes for unwire, retires child host resources, submits release work, then deallocates page-table coverage | Mapping release is an ordered core transition with backend release effects, not a cache deletion | Established for the PCI pathway only |
| Resource backing replacement is an operation on an existing resource | Preserve resource identity and advance a backing generation/storage edge | Established; exact failure ordering is not |
| Segment resource lists independently initialize, update in-channel, prepare, and complete; queue processing parses them and writes invalidations | Preserve segment boundaries and the complete resource-participation envelope through execution | Established |
| IOSurface construction carries descriptor and plane; mapper capability, mapper-reference texture/buffer, and display-mapper capability are distinct interface concepts | Name the object class semantically, but keep mapper reference, IOSurface backing, view, and display use as separate relations | Class established; mapper storage ownership and descriptor tails remain open |
| Display begin, validate, submit, completion query/signal, framebuffer resource, and resource-heap operations are distinct | Display is a core transaction completed from presenter/executor facts | Established for the PCI display lifecycle |
| Object tables, mapper IOSurfaces, display mapper surfaces, heaps, buffer-from-IOSurface, discard, and synchronize/discard are independently advertised features | Guest protocol capabilities are a typed profile independent of host Vulkan capabilities | Established as independently named capabilities; availability combinations still require tests |

This table is deliberately not a wire-layout ledger. Exact offsets, opcode values, and binary
provenance stay out of the architecture plan; they belong at the decoding boundary and in local
investigation notes.

### Conclusions that changed or constrained the design

- The two-topology seam is deliberately below the executor contract. Two complete topology
  implementations of resource lifetime, coherency, or completion would contradict the evidence:
  those are guest-semantic operations, while topology is not a guest contract term.
- The immutable submission is segmented rather than a flat list because segment resource lists
  are constructed, parsed, processed, reclaimed, and completed as objects in their own right.
  Flattening them would erase the participation and completion boundary the kernel interface
  preserves.
- Mapping teardown is modeled as an ordered transition because the PCI implementation first
  synchronizes affected resources for unwire, then retires child host resources, submits release
  work, and only then deallocates page-table coverage. The mapper pathway does not inherit that
  sequence until equivalent evidence exists.
- Numeric object names stop at the decode boundary. The serializer names the tag-11 construction
  as an IOSurface texture, so that semantic name is established even though the mapper-backed
  storage relation behind it is not. A known object class and an unknown ownership edge can and
  should coexist in the type system.
- Object-table and serializer-reference destruction remain separate transitions. Static APIs
  distinguish resource-indexed entries from per-family reference operations, and existing runtime
  evidence demonstrates that equal integers collide without denoting the same object. The core
  correlates the two only through an explicit decoded operation.
- Display remains a semantic transaction rather than a presentation callback because begin,
  validate, submit, completion query/signal, framebuffer-resource, and resource-heap operations
  are independently visible interfaces.

These conclusions are the minimum architecture justified by the evidence. Generational IDs,
replica versions, immutable submissions, and executor tickets remain our mechanisms for enforcing
them; the plan does not claim those mechanisms are guest wire structures.

The static interface supports the architecture's central split, but not every concrete type in the
first draft. Keep three confidence levels separate during implementation:

**Established contract:**

- The task resource heap, its handle namespace, resource objects, memory maps, and page table are
  separate interfaces with separate setup, resize, add/remove, commit/update/release, and
  allocate/deallocate operations.
- The object-table/resource-heap namespace and serializer references are distinct namespaces.
  Resources expose resource indices and typed serializer references separately, and serializer
  references are allocated and released by object family. The same integer appearing in both
  namespaces does not identify the same object.
- Resource references have distinct allocate, release, and delete operations across buffers,
  textures, samplers, depth state, pipelines, functions, fences, and heaps. The internal
  generational `ResourceId` is therefore a safety property of our resolver; it is not asserted to
  be a guest wire field.
- Resources expose prepare/complete, page-on/page-off, backing replacement, host invalidation,
  synchronize-for-unwire, child add/remove/delete, and distinct host-resource/backing deletion
  operations.
- Segment resource lists have their own construction, prepare, complete, and channel-update
  lifecycle. Command-queue processing parses those lists and emits write invalidations.
- On the PCI lifecycle, mapping release synchronizes affected resources for unwire and retires
  child host resources before page-table coverage is deallocated. This ordering is contractual for
  that pathway and must not be generalized to the mapper pathway without matching evidence.
- Command buffers explicitly record resource reads/writes, page-off participation, state
  references, chunks, segments, splits, continuations, and merges.
- Textures have explicit buffer-backed, parent-texture view, IOSurface-plane, and heap-placement
  construction forms. Buffers and textures retain fields identifying their parent/storage
  relation rather than presenting every view as a fresh allocation.
- Mapper-capable contract code advertises mapper-backed surfaces and has distinct mapper-reference
  texture, mapper-reference buffer, and backing-resource classes. This establishes a mapper
  reference relation; it does not establish the complete lifetime of that backing or equivalence
  with the registered-surface pathway.
- Render, compute, blit, and parallel-render encoders are distinct producers over shared command
  buffer and segment machinery. Resource/heap declarations, barriers, and fences are encoder
  operations in their own right.
- Display exposes separate begin, validate, submit, completion-query, completion-signal,
  framebuffer-resource, mode/power, cursor, and resource-heap operations.
- The common user-space serializer contains both supported instruction-set slices. That supports a
  shared semantic protocol layer for the serializer forms that match; it does not establish that
  the pathway-specific kernel contracts match.

**Architecture justified by those facts, but intentionally ours:** generational internal IDs, a
canonical resource graph, immutable resolved submissions, replica-version content authority, and
executor tickets are implementation structures chosen to make the established distinctions
unrepresentable as accidental aliases. Their exact state enumeration is not claimed to be a wire
layout.

**Not established by the present static set:** the complete arm mapper object lifecycle, ownership
of the mapper's backing object, the meaning of undecoded mapper descriptor tails, exact failure
ordering among every page/resource operation outside the PCI release sequence above, or any host
unified/discrete placement rule. Those remain gated migrations. Runtime topology equivalence is a
design invariant to test, not a conclusion obtained from guest code.

The decoded interface is object- and lifecycle-oriented. The target architecture must preserve the
following contract boundaries rather than replacing them with convenient backend abstractions:

- A task owns a resource namespace/heap. Object references are entries in that namespace, not
  process-global resource IDs.
- A task's object-table reference and a serializer's typed resource reference are not one generic
  `u32` namespace. Resolve each once at its own boundary. If the core must correlate them, it does
  so through an explicit relation established by a decoded contract operation, never by equal
  ordinals.
- Allocating, releasing, and deleting an object reference are distinct operations. A released slot
  can later be allocated again; our resolver must assign a new internal generation and must not
  treat slot release as synonymous with destroying every resource or backing formerly named by
  that slot.
- A resource object, its storage/backing, a memory mapping, and the task page table are distinct
  objects with distinct create, update, replace, release, and free operations.
- A resource may own child resources and views. Buffer-backed textures, texture views, shared
  textures, IOSurface plane views, and heap placements are relations between objects and storage;
  they are not independent allocations merely because the host backend materializes them that way.
- Replacing physical backing mutates the backing of a stable resource lifetime. It advances a
  backing generation; it does not silently manufacture a new guest object lifetime.
- Resource participation is submission-scoped. Command buffers and their segments carry resource
  references, write intent, page-off requests, heap references, metadata references, fences, and
  protection information. Prepare/complete and invalidation work is coupled to that participation.
- Render, compute, blit, resource-state, and parallel render encoders are distinct semantic
  producers, but they share command-buffer segmentation, splitting/continuation, resource lists,
  and completion.
- Stamps, barriers, and events are explicit completion machinery. They are not incidental Vulkan
  fence bookkeeping.
- Display has its own begin/validate/submit/complete transaction lifecycle and resource namespace.
  A host presenter consumes a validated/submitted transaction and returns the fact from which the
  core completes it; the presenter does not replace that protocol.
- Guest protocol and serializer features are structural capabilities: serializer version and
  advertised feature state select available forms. Model object tables, mapper-backed surfaces,
  shared textures, heaps, resource discard/synchronization, and display facilities as independent
  capabilities unless the decoded contract explicitly couples them. They must not be inferred
  from an OS version.

These facts validate the semantic-core/executor split, while correcting two parts of the original
draft:

1. `AllocationId` cannot stand for resource, backing, IOSurface, and address-space mapping at once.
   The canonical graph needs separate resource, storage/backing, and mapping identities.
2. `ExecutionBatch` cannot be only a flat command list plus accessed allocations. It must preserve
   command-buffer and segment boundaries plus the complete submission resource-participation set.

The common serializer implementation is available for both supported architectures, but the
complete arm-only mapper backing and kernel-side object lifecycle are not established by the
present static contract set. In particular, do not generalize the x86 resource/backing lifecycle
into the arm mapper, and do not assign meaning to undecoded tails of mapper-path descriptors. Those
remain pathway-specific questions requiring arm evidence before the corresponding migration lands.

## Current structural findings

The repository's stated layers do not match its actual dependencies.

- `runtime` says it plans work and makes no GPU API calls, while draw, compute, blit, mapping,
  writeback, scanout, and presentation call the Vulkan engine directly.
- `backend::Backend` is effectively a reset hook. `Device<B>` is generic over a backend that does
  not execute its work.
- `DeviceState` combines protocol and scheduler state with mappings, host caches, coherency,
  resident identities, backend imports, retirement ledgers, presentation, and observability.
- `TaskResource` owns raw descriptor bytes, decoded construction semantics, guest-object lifetime,
  mapped-surface registration, render-target history, and Vulkan resident leases.
- `MappingEntry` combines guest mapping lifecycle, page-table state, geometry, content generations,
  coherency, host views, imports, backend eligibility, and instrumentation.
- `ComputeStorageResidencyKey` represents mapped surfaces, task-GVA textures, and heap textures by
  repurposing fields and sentinel zero values instead of carrying a typed variant.
- The Vulkan engine's guest-derived state is process-global even though QEMU can bind more than one
  device instance.
- Topology classification is reasonably centralized, but its consequences are distributed among
  import, cache, resident, writeback, pinning, batching, and presentation paths.
- The required `backend-vulkan` feature still shapes hundreds of conditional compilation sites,
  preserving the source structure of a backend fork that no longer exists.
- Raw wire names such as `type11` and `type7` persist after decode. Tag 7 has decoded semantic
  classes; tag 11 is an IOSurface texture construction whose mapper/storage relation still
  contains arm-specific unknowns. Both need boundary-local names matching only what is
  established.
- Draw preparation and Vulkan execution exchange separate wide request structures, both mixing
  semantic state, resource resolution, transport choices, and mutable output fields.

The missing owner can be summarized as:

```text
one guest resource
    |- task resource
    |- texture-to-mapping association
    |- mapping entry
    |- host surface or linear texture
    |- render-target identity
    |- compute residency identity
    |- GVA resource, plane, and store identities
    |- gather witness
    |- writeback debt
    `- Vulkan resident
```

Deletion, replacement, unmapping, and reset must update several of these representations and queue
keys for other modules to retire later. No aggregate owns the complete lifetime.

## Target dependency graph

```text
reims-vgpu-wire ----> reims-vgpu-protocol ---+
                                              |
reims-vgpu-paging ----------------------------+
                                              v
                                      reims-vgpu-core
           device state + resource graph
           command normalization + ordering
           content authority + guest effects
                         |
                  immutable ResolvedSubmission
                         |
                         v
                  reims-vgpu-vulkan
         +---------------+----------------+
         v                                v
  residency/unified.rs          residency/discrete.rs
         +---------------+----------------+
                         v
               Vulkan submission/completion

Current reims-vgpu staticlib:
QEMU ABI + pathway adapters + composition only
```

The compile-time dependency direction is:

```text
wire ------> protocol ----+
                          v
paging ----------------> core <---- vulkan
                          ^
                     composition root
```

Protocol decoding names mapping commands but does not walk page tables. Paging consumes explicit
page geometry and already-decoded mapping inputs; core composes the two. This keeps byte decoding,
address translation, and resource lifetime as three separate responsibilities.

Core never imports Vulkan, QEMU, winit, MoltenVK, or host-OS types. Vulkan depends on the semantic
execution types and implements the executor port defined by the core. The staticlib owns both and
coordinates them.

## Proposed crates

### `reims-vgpu-wire`

Keep the existing no-allocation, borrowed, byte-accurate wire views. Raw numeric tags and fields may
exist here. Safe accessors may return semantic raw-tag enums, but wire structs continue to contain
only all-bytes-valid align-1 fields.

### `reims-vgpu-paging`

Keep pure page-table resolution, span walking, GPA run coalescing, and region planning. Page
geometry remains explicit. The crate does not learn about QEMU, Vulkan, topology, or resource
lifecycle.

### `reims-vgpu-protocol`

Create a semantic protocol boundary depending on `wire`. It owns:

- typed object kinds and construction descriptors;
- typed render, compute, blit, event, fence, and lifecycle commands;
- Metal-semantic enums and flags;
- task, mapping, reference, address, offset, and length newtypes;
- total parsing from guest values;
- typed decode refusals;
- pathway-specific normalization where the two guest contracts genuinely converge.

No raw object tag leaves this crate as the identity used by a consumer.

### `reims-vgpu-core`

Create a backend-independent stateful domain core owning:

- register, FIFO, task, channel, stamp, event, and fence state;
- task-owned resource namespaces and the canonical resource/storage/mapping graph;
- resource lifetime and reference namespaces;
- submission resource lists, segment boundaries, and prepare/complete state;
- content authority and replica versions;
- semantic encoder state and command ordering;
- resource resolution and immutable resolved submissions;
- display transactions, presentation requests, and presentation completion effects;
- guest-visible completion effects;
- executor and host-service ports.

### `reims-vgpu-vulkan`

Move all Vulkan implementation concerns here:

- Ash instance, device, queue, and capability discovery;
- Metal-to-Vulkan translation;
- shader and pipeline compilation and immutable content-keyed caches;
- GPU allocation and resource registry;
- submission, completion, device loss, and fence-safe retirement;
- import, upload, readback, and presentation implementations;
- unified and discrete placement policies;
- internal performance counters.

No Vulkan handle or `vk::*` type crosses into core state or semantic identity.

### `reims-vgpu-testkit`

Create reusable test infrastructure:

- fake guest physical memory and page views;
- scripted executor implementing the production executor port;
- unified/discrete and import/no-import capability fixtures;
- semantic command traces;
- delayed completion, allocation refusal, and device-loss injection;
- guest-visible effect comparison.

### Current `reims-vgpu` staticlib

Reduce the current product crate to the composition root:

- QEMU ABI and ABI/header agreement tests;
- bound-device registry;
- PCI and MMIO pathway adapters;
- construction of `DeviceCore`, `VulkanExecutor`, host services, and event sinks;
- panic containment and action delivery.

The C shims remain thin and receive final answers from Rust rather than inputs from which they can
reconstruct product policy.

### `contract/` disposition

Do not preserve `contract/` as a shared grab-bag. Move each item to its actual owner:

- byte layouts and raw tags to `wire`;
- semantic Metal values and operations to `protocol`;
- resource and coherency rules to `core`;
- Vulkan translations and capability interpretation to `vulkan`.

## Canonical resource graph

Every guest object resolves once to a generational internal resource identity.

```text
TaskId -> ObjectTableNamespace -> ObjectTableRef -> ResourceId<T>
                                                     |     |      |
                                     immutable kind -+     |      +-> parent/child/view edges
                                                           |
                                                           v
                                                 StorageRef / StorageSlice

TaskId -> SerializerNamespace<T> -> SerializerRef<T> --explicit decoded relation--> ResourceId<T>
                                                    |
                     +------------------------------+--------------------------+
                     v                              v                          v
                  StorageId                 SurfaceBackingId                HeapId
                     |                              |                          |
                     +------------------------------+--------------------------+
                                                    |
                                                    v
                                           subresource ContentState

TaskId -> AddressSpace -> MappingId -> mapped MemoryObject/page spans -> PageTable

ResolvedSubmission -> SegmentResourceList -> ResourceUse(ResourceId, access, range)
```

The graph must express these distinctions directly:

- `ResourceId<T>` identifies one guest object lifetime after namespace resolution.
- `ObjectTableRef` identifies a task-local object-table slot whose allocation/release lifetime is
  separate from the resolved resource and any host-object deletion command.
- `SerializerRef<T>` identifies a slot in an independent, object-family-specific serializer
  namespace. It is not interchangeable with `ObjectTableRef`, even when the raw integers match.
- `StorageId` identifies guest-semantic storage/backing. A backend allocation is an executor-local
  implementation detail and has its own identity.
- `MappingId` identifies an address-space mapping. A mapping may expose storage, but is not itself
  the resource or the storage lifetime.
- `SurfaceBackingId` identifies a shared/IOSurface backing independently of any texture view over a
  plane of it.
- A texture view references a parent resource or storage slice and an exact subresource range.
- Several resources may alias one storage object, and one storage object may be reachable through
  more than one mapping over its life.
- Deleting a view does not delete storage retained by another live view.
- Releasing a mapping does not imply resource deletion; deleting a resource does not implicitly
  release an independently live mapping.
- Parent/child resource edges and heap membership are explicit and impose destruction order.
- Reusing a guest reference produces a new internal generation.
- Geometry and format belong to an immutable descriptor, not a reconstructed identity key.
- Backend handles are indexed by internal IDs; core resources do not hold Vulkan leases.
- Replacing backing preserves `ResourceId` and changes `BackingGeneration` plus the storage edge.

A semantic texture has this shape:

```rust
TextureResource {
    id: ResourceId<Texture>,
    descriptor: TextureDescriptor,
    storage: TextureStorage,
    view: SubresourceView,
}

enum TextureStorage {
    Dedicated(StorageId),
    TaskAddress {
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    },
    BufferRange {
        buffer: ResourceId<Buffer>,
        offset: ByteOffset,
        bytes_per_row: ByteLength,
    },
    IOSurfacePlane {
        surface: SurfaceBackingId,
        plane: PlaneIndex,
    },
    HeapPlacement {
        heap: ResourceId<Heap>,
        offset: ByteOffset,
    },
}
```

The mapper pathway initially carries a typed IOSurface-texture construction record with its
explicit mapper reference, while the other pathway can resolve a registered surface backing. The
mapper reference must remain its own pathway-specific identity; do not automatically turn it into
a page-table `MappingId`, a registered `SurfaceBackingId`, or an owned IOSurface backing merely to
fit this graph. Normalize those origins only after their shared storage and lifetime semantics are
established. The actual types may differ, but all identity and lifetime distinctions above must
remain representable and exhaustive.

### Semantic naming

Raw object tag 11 denotes an IOSurface texture construction on the mapper-capable path. Calling the
object class `IOSurfaceTexture` is therefore justified; calling it `Type11`, a mapping, or a
registered surface is not. What remains unresolved is the storage relation and lifetime behind its
mapper reference. The protocol boundary preserves uninterpreted descriptor-tail variants and
refuses any operation whose answer depends on them. If arm evidence establishes an owned or shared
IOSurface-plane backing relation, downstream code should then see the semantic object and that
relation:

```text
IOSurfaceTexture {
    descriptor: { format, extent, usage, ... },
    storage: IOSurfacePlane(surface_backing, plane),
}
```

It does not see `type11`, and it does not call the texture itself a mapping. Until the relation is
fully established, it carries a typed IOSurface-texture descriptor and an explicit
`MapperSurfaceRef` (or equivalently narrow type) rather than inventing a page-table `MappingId`, a
registered `SurfaceBackingId`, or an owned IOSurface-plane relation. Raw tag 7 immediately becomes
`Sampler`, `DepthStencil`, `RenderPipeline`, `ComputePipeline`, or `IndirectCommandBuffer` when its
decoded discriminator establishes that class; there is no semantic `Type7`.

Raw values may remain on a boundary diagnostic as fields such as `wire_tag=11`. APIs, state fields,
tests, refusal slugs, and module names use semantic vocabulary.

### Typed identities and values

Introduce distinct types for unrelated namespaces:

- `TaskId`
- `ObjectTableNamespaceId`
- `SerializerNamespaceId<T>`
- `ObjectTableRef`
- `SerializerRef<T>`
- `MappingId`
- `MapperSurfaceRef`
- `SurfaceId`
- `SurfaceBackingId`
- `StorageId`
- `FenceRef<Domain>`
- `GuestVirtualAddress`
- `GuestPhysicalAddress`
- `GuestKernelAddress`
- `ByteOffset`
- `ByteLength`
- `ContentVersion`
- `BackingGeneration`
- `SubmissionId`

Parse guest ordinals once into total semantic types:

- `PrimitiveTopology`
- `LoadAction`
- `StoreAction`
- `CullMode`
- `Winding`
- `FillMode`
- `DepthClipMode`
- `PixelFormat`
- `TextureUsage`

An unknown guest value becomes a typed refusal at the protocol boundary. An internal producer
carries the type, never its ordinal.

## One resource-lifecycle and content-authority model

Separate namespace-slot, semantic-resource, storage/backing, and mapping lifetimes from the
question of which replica contains the current bytes. They are coordinated state machines, not one
bag of flags. In particular, reference release is not a terminal resource state.

Conceptually, the lifetimes record:

```rust
ObjectTableSlot {
    generation: ObjectGeneration,
    state: Free | Bound(ResourceId) | Released,
}

SerializerSlot<T> {
    generation: ReferenceGeneration,
    state: Free | Bound(ResourceId<T>) | Released,
}

ResourceLifecycle {
    state: Live | Retiring | Destroyed,
    backing: StorageRef,
    backing_generation: BackingGeneration,
    parents: Vec<ResourceId>,
    children: Vec<ResourceId>,
    prepared_by: Set<SubmissionId>,
    in_flight: Set<SubmissionId>,
}

StorageLifecycle {
    state: Live | Replaced | Retiring | Destroyed,
    owners: Set<ResourceId>,
    mappings: Set<MappingId>,
}
```

The exact representation may be more compact. It must still express prepare/complete pairing,
page-on/page-off, child/view retention, backing replacement, synchronization-for-unwire, host
invalidation requirements, explicit host-backing deletion, and deferred destruction while work is
in flight. Releasing either reference slot removes only that namespace edge. An explicit object or
host-resource deletion initiates the corresponding resource/storage retirement, and actual
destruction waits for children, owners, mappings, and in-flight submissions as the contract
requires. Mapping commit/update/release is a separate address-space transition linked to affected
storage; it is not encoded as a resource lifecycle state.

The current system distributes authority among validity bits, several generations and epochs,
host-write ledgers, gather and store witnesses, writeback debt, host copies, resident generations,
and engine-only-content flags. Replace those behavior-selecting representations with one
per-subresource state machine.

Conceptually:

```rust
ContentState {
    current: ContentVersion,
    guest: ReplicaVersion,
    gpu: Option<GpuReplica>,
    host: Option<HostReplica>,
    pending: Option<PendingSubmission>,
}
```

Domain transitions include:

- `guest_wrote(range)`
- `gpu_store_planned(submission)`
- `gpu_store_completed(version)`
- `copy_gpu_to_guest_completed(version)`
- `copy_guest_to_gpu_completed(version)`
- `host_invalidation_required(range)`
- `synchronize_for_unwire(range)`
- `page_on()` / `page_off()`
- `backing_replaced(new_backing_generation)`
- `resource_delete_requested()`
- `storage_retired()`

Reference release is intentionally absent from `ContentState`: removing a namespace edge does not
change which replica contains the current bytes of a still-live resource.

A replica is current when its version equals `ContentState.current`. GPU-only content is derived
from replica versions instead of duplicated as an independent flag. Writeback debt is the explicit
fact that the guest replica trails the current version plus the contract rule governing when that
replica must be updated.

Keep the four wire validity operations, explicit resource-list write intent, synchronization and
discard commands, and mapping/backing operations as decoded protocol observations. Their
behavioral effects call the lifecycle and content state machines. The observation record and the
authority decision are not two competing sources of truth.

Use distinct types for:

- object lifetime generation;
- storage or backing generation;
- content version;
- submission identity.

Do not reuse a generic `generation` field for more than one of those meanings.

## One executor boundary

The real backend boundary is submission and completion. It is not one method per Metal operation,
and it is not a collection of process-global Vulkan entry points.

The core constructs an immutable `ResolvedSubmission` containing:

- command-buffer identity plus ordered segment/chunk and split/continuation boundaries;
- ordered render, compute, blit, resource-state, event, and fence commands grouped by encoder;
- the complete segment resource-participation list;
- resolved `ResourceId`s, `StorageId`s, and subresource ranges;
- typed Metal semantics;
- declared read, write, and read/write accesses;
- page-off, heap, resource-metadata, purge/discard, and protection participation where present;
- expected content versions;
- synchronization dependencies;
- guest-visible completion, stamp, and barrier identities.

Do not reconstruct this envelope from Vulkan bind use alone. An explicitly declared resource or
heap can participate in lifetime, invalidation, synchronization, or protection even when no draw
binding mentions it.

The executor returns a ticket and later an immutable completion:

```rust
trait Executor {
    fn capabilities(&self) -> &ExecutorCapabilities;

    fn submit(
        &mut self,
        submission: ResolvedSubmission,
    ) -> Result<SubmissionTicket, ExecutionRefusal>;

    fn poll(&mut self) -> Vec<ExecutionCompletion>;

    fn release(&mut self, release: ReleaseBatch);
}
```

The exact signatures may evolve, but the boundary must preserve these properties:

- submission inputs are immutable;
- results are separate values, not output fields on requests;
- all resource references are resolved semantic identities;
- segment/resource-list ordering survives the boundary;
- topology and Vulkan placement details do not appear in the submission;
- a core transition to completed content occurs only after a successful completion fact;
- resource `complete` and release eligibility are driven from that same completion fact;
- stamps and interrupts are published by the core from completion, not speculatively by the
  executor.

`Arc`-backed immutable bind snapshots are compatible with this boundary and should remain.

## Unified and discrete topology policies

The topology seam belongs inside `reims-vgpu-vulkan`:

```text
vulkan/residency/
    mod.rs
    unified.rs
    discrete.rs
```

Both implement a sealed policy over the same request and result types:

```rust
trait PlacementPolicy {
    fn plan(
        &self,
        request: &PlacementRequest,
        replicas: &ReplicaSnapshot,
        caps: &MemoryCapabilities,
    ) -> Result<ResidencyPlan, ResidencyRefusal>;
}
```

A residency plan may select:

- an existing current resident;
- imported guest memory;
- host-visible memory;
- device-local memory;
- DMA upload or readback;
- CPU fallback where the contract permits it;
- deferred writeback where the contract permits it.

A topology policy may not:

- create or end a guest resource lifetime;
- change content authority before completion;
- invent a resource identity;
- evict or discard the only current copy of guest-visible content;
- decide whether the guest command semantically succeeds;
- special-case object IDs, sizes, names, observed content, or other side channels.

### Unified policy

Prefer, subject to measured capabilities:

- direct or imported guest memory;
- host-visible device-local allocations;
- zero-copy reads and writes;
- avoiding staging transfers;
- submission shapes measured safe for unified memory.

### Discrete policy

Prefer, subject to measured capabilities:

- device-local working residents;
- explicit DMA uploads and readbacks;
- cached system-memory readback buffers;
- transfer batching;
- residents retained for the guest-owned resource lifetime.

Host-pointer import is an orthogonal capability, not a third topology. The executor must support:

```text
                import available    no import
unified
discrete
```

Submission batching is a separate `SubmissionPolicy`, even when defaults are derived from the same
capability profile. The discrete policy must not become a container for every tuning choice made on
one discrete host.

The governing equivalence property is:

> The same semantic trace under unified, discrete, import-enabled, and import-disabled policies
> produces identical guest memory, stamps, interrupts, refusals, and presented content. Only
> allocation, transfer, cache, and timing events may differ.

## Keep pathway, topology, import, and presentation orthogonal

| Axis | Owner |
|---|---|
| PCI versus MMIO attach | QEMU adapter |
| 4 KiB versus 16 KiB guest pages | explicit page geometry in core and paging |
| arm mapper versus x86 surface backing | pathway decoder plus core mapping service |
| unified versus discrete memory | Vulkan placement policy |
| host-pointer import availability | Vulkan capability |
| native ICD versus MoltenVK | Vulkan loader adapter |
| QEMU console versus host window | presentation adapter |

Construct three explicit, non-interchangeable profiles:

- `PathwayProfile`: attach transport, page geometry, address-space form, and required host ports;
- `GuestProtocolCapabilities`: negotiated object-table, mapper-surface, shared-texture, heap,
  synchronization/discard, display, and command-stream features;
- `ExecutorCapabilities`: measured Vulkan memory, import, format, synchronization, and queue
  capabilities.

`GuestProtocolCapabilities` controls which guest contracts can be decoded or advertised.
`ExecutorCapabilities` controls how an already-decoded contract is implemented. Neither profile is
inferred from an OS release, device name, GPU vendor, or driver string, and a host capability may
not be substituted for guest intent.

No `cfg(target_os)` belongs in the semantic core. Platform gates are confined to the Vulkan loader,
window implementation, QEMU host adapter, and truly platform-specific test support.

Split the current broad host interface into narrower ports:

- guest physical memory;
- host page-view provider;
- guest CPU and KVA introspection;
- interrupt and host-effect sink;
- scheduler wake;
- presenter. The presenter consumes a validated and submitted display transaction and returns a
  presentation completion; it does not consume a transaction already marked complete.

Validate the required ports against the selected pathway at construction rather than discovering a
missing capability inside a deep operation.

## Lifecycle and engine ownership

Split process-wide physical GPU state from per-vGPU session state.

```text
PhysicalGpuContext
    Vulkan instance and device
    measured capabilities
    immutable content-keyed shader and pipeline caches
    queue infrastructure

VgpuSession
    per-device residents
    imports and aliases
    pending submissions
    presentation resources
    resource front indexes
    counters and device-loss state
```

`PhysicalGpuContext` may be shared through `Arc`. One `VgpuSession` belongs to one bound device.
Resetting one guest does not reset another guest's residents, imports, presenter, or submissions.

A resource, storage, surface backing, mapping, or session release emits a distinct typed release
effect. Parent/child resources and heap members retire in dependency order; an in-flight prepared
resource becomes releasable only after its matching completion. Replacing physical backing emits a
storage transition while preserving the resource identity. The Vulkan session performs fence-safe
destruction. Host page mappings are represented by ownership-bearing leases rather than raw
pointer/length pairs and deferred retirement side vectors. If a release must run on a particular
thread, dropping the lease queues that typed release to the owning executor rather than losing the
ownership relation.

The reset sequence is explicit:

1. stop accepting new submissions for the session;
2. resolve or report guest-visible outstanding work;
3. complete or explicitly refuse prepared resource participation;
4. release task namespaces, resources, children/views, storage, and mappings in contract order;
5. fence-safely retire GPU children and imports;
6. release host page-view leases and page-table bindings;
7. reset protocol, display-transaction, and register state;
8. reopen the session.

## Core module organization

Organize the core around domain ownership instead of a horizontal model/runtime split:

```text
core/
    device/
        registers
        fifo
        scheduler
        task
    resource/
        namespace
        registry
        object
        storage
        mapping
        view
        content
        lifecycle
    command/
        render_encoder
        compute_encoder
        blit_encoder
        resource_state_encoder
        resolve
        submission
        resource_list
        segment
    sync/
        events
        fences
        stamps
    display/
        transaction
        present_effect
    ports/
```

State and the transitions allowed to mutate it live together. Render, compute, blit, resource
state, and display resolve their endpoints through the same resource API rather than carrying
independent ladders for mapper-backed IOSurface textures, task-address textures, registered
surface views, buffer views, and heap resources.

## Observability

Preserve typed declines, the fail channel, and always-on explanations for guest-work loss.

Change the dependency direction:

- domain layers own their typed events;
- a device-scoped sink records structured events;
- the Vulkan executor returns completion metrics or emits Vulkan-local events;
- the engine does not import drain modules merely to increment string counters;
- first-sighting and deduplication state is scoped to the device or physical context whose event it
  describes;
- observation-only censuses do not share storage with behavior-selecting state.

Expected waits remain quiet. Typed refusals remain the outcome when the contract is not known or a
guest command cannot be honored.

## Verification architecture

### Protocol tests

- Wire fixtures decode into semantic commands and resource descriptors.
- No raw object tag is required after protocol decoding.
- Unknown values produce the exact typed refusal.
- x86 and arm fixture paths normalize only where their contracts agree.
- Negotiated guest feature combinations enable only their declared command and object forms.
- Existing serializer-fixture divergence and unwritten-bit tests remain authoritative.

### Resource state-machine tests

Exercise sequences including:

```text
create -> map -> make view -> render -> synchronize -> delete
create -> render -> guest write -> synchronize
create -> submit -> delete while in flight -> complete
map -> import -> repoint pages -> complete
create -> prepare -> submit -> complete -> pageoff -> unwire
create parent -> create child view -> delete parent -> use child -> delete child
create -> replace backing -> delayed old completion -> synchronize
map -> commit -> create resource -> release mapping -> delete resource
task redefine -> reference reuse
device reset with pending work
```

Assert:

- every live guest object has exactly one current internal resource;
- stale generational IDs cannot resolve;
- each backend allocation belongs to live semantic storage or a fence-safe retirement queue;
- replacing backing preserves the resource identity and invalidates stale backing generations;
- mapping commit/update/release does not accidentally create or delete resource lifetimes;
- parent/child and view/storage retention follows explicit graph edges;
- every successful prepare has exactly one completion or typed teardown refusal;
- the current content version exists in at least one replica;
- a guest write is never overwritten by an older delayed GPU result;
- deletion cannot leak a sole-copy resident;
- reset leaves no session-owned alias;
- a second record is accumulated or replaced according to the decoded API contract, never because
  a state slot had accidental capacity one.

### Submission and display tests

Exercise segmented command buffers with splits and continuations, mixed encoder kinds, resource and
heap declarations, metadata references, page-off requests, fences, protection options, and write
invalidations. Assert that normalization preserves their order and scope and that an explicitly
declared resource is not dropped merely because no Vulkan binding names it.

Exercise display begin, validate, submit, completion, mode/power changes, cursor updates, and
framebuffer resource replacement. Assert that presentation occurs only from a validated and
submitted semantic display transaction, and that the resulting completion fact drives display
transaction completion. Presenter failure cannot mutate the renderer's resource lifetime or be
reported as a successful completion.

### Topology equivalence tests

Run every semantic trace under:

- unified with import;
- unified without import;
- discrete with import;
- discrete without import.

Compare guest-visible effects exactly. Compare internal performance metrics separately:

- bytes uploaded and downloaded;
- DMA operations;
- CPU copies;
- allocation count;
- readbacks;
- waits;
- resident hits.

### Executor conformance

Replace the structural role of `NullBackend` with `ScriptedExecutor`, implementing the exact
production executor port. It can complete immediately, delay completion, refuse allocations,
simulate device loss, and reorder only where the contract allows.

### Integration matrix

Keep serial Rust tests, wire fixture gates, cross-target builds, and pathway-specific live boots.
Run pathway-specific facts on their owning pathway. Add explicit multi-device isolation coverage:
resetting device A must not modify device B's resources, submissions, presenter, or guest effects.
Run guest-feature profiles independently from the four host-memory cells so a protocol capability
does not accidentally become a topology classifier.

### Architectural enforcement

Enforce dependency direction with crate boundaries and visibility, not source scanners:

- core has no Ash, QEMU, windowing, or platform-loader dependency;
- only the Vulkan crate knows Vulkan handles and formats;
- only the composition crate exposes the QEMU ABI;
- protocol depends on wire, never on paging or runtime execution;
- paging depends on neither protocol lifetime nor backend execution;
- semantic values cross boundaries as types rather than duplicated ordinals.

## Contract questions that gate migration

The architecture does not require every field to be known before work begins, but a migration may
not erase an unknown by folding it into a generic abstraction. These questions gate the named
families:

- What semantics, if any, live in the undecoded tail variants of the mapper-path IOSurface texture
  descriptor? Until known, preserve the bytes at the boundary and issue a typed refusal for a case
  that depends on them.
- Which parts of mapper-backed IOSurface lifetime and registered-surface/ref-texture lifetime are
  truly equivalent across pathways? Share the semantic texture/view contract only after this is
  established; keep pathway construction and teardown distinct meanwhile.
- Beyond the established PCI mapping-release sequence, what are the exact ordering and failure
  rules for resource prepare/complete, page-on/page-off, write invalidation,
  synchronize/discard, and backing deletion around a submitted segment?
- Which child-resource classes retain their parent or shared storage, and which deletion operation
  retires the host handle versus the backing itself?
- For each object class, which operation only releases the task-local reference and which one
  deletes the semantic object or its host representation?
- Which command-buffer resource-list entries are declarative hazards, which request residency or
  paging, and which demand a guest-visible completion effect?
- Which display transaction fields establish scanout identity and completion, independently of the
  host presentation mechanism?

Resolve each question into a typed contract, focused fixture/state-machine test, and a comment next
to the owning code before migrating behavior that depends on it. Static interface evidence may
establish shapes and call relationships; pathway-specific behavior still needs verification on the
pathway that exercises it.

## Migration sequence

Each phase must have focused tests and keep the affected pathways buildable. Behavior-preserving
phases should not make broader correctness claims. A behavior change must have a failing regression
test or measured proxy before it lands.

### 1. Establish semantic vocabulary

Create `reims-vgpu-protocol`.

Move object and command decoding behind semantic types. Introduce separate typed object-table and
serializer namespaces, resource, storage, surface-backing and mapping IDs plus addresses, lengths,
formats, actions, and selectors. Replace `type11` with `IOSurfaceTexture` at the decoder boundary
and eliminate `type7` from decoded values. Carry the mapper reference as its own typed relation; do
not assign it a page-table mapping identity, registered-surface identity, or owned IOSurface-plane
storage edge until arm evidence establishes that relation. Preserve mapper and registered-surface
construction variants until their lifecycle equivalence is established.

Do not change execution behavior in this phase.

Exit criteria:

- protocol fixtures produce the new semantic values;
- object-table refs and typed serializer refs cannot be interchanged;
- unknown tags and ordinals retain typed refusals;
- new consumers do not branch on raw object tags.

### 2. Establish the real executor port

Introduce `ResolvedSubmission`, `ExecutionCompletion`, and `Executor`.

Initially, one compatibility adapter may translate existing draw, compute, and blit request shapes
plus current segment/resource-list metadata into the current engine. Route all product execution
through that adapter before changing the requests themselves.

Exit criteria:

- one product submission/completion boundary exists;
- segment boundaries and all resource-list participation cross it;
- no semantic runtime module calls individual Vulkan engine operations directly;
- a scripted executor drives device completion tests through the same boundary.

### 3. Make Vulkan state device-scoped

Turn the zero-sized backend shell into an owned executor and per-device session. Separate shareable
physical context and immutable content caches from guest-derived pools, residents, imports,
submissions, and presenter state.

Exit criteria:

- a two-device test proves reset and deletion isolation;
- reset operates on a session, not a process-global guest state;
- immutable cache sharing remains content-keyed and guest-identity-free.

### 4. Introduce the canonical resource graph

Create task-owned object-table and typed serializer namespaces, generational resource IDs, storage
and surface-backing IDs, mapping identities, backing variants, parent/child/view edges, and
lifecycle effects.

Migrate one end-to-end resource family at a time:

1. registered IOSurface backings and serialized plane/view resources;
2. task-address textures;
3. buffers and buffer-backed textures;
4. heaps and heap-placed resources;
5. pipelines, samplers, depth state, fences, and indirect command buffers;
6. mapper-backed surface textures, after the arm mapper contract questions above are answered.

The mapper family may receive the generic generational identity and minimum-established semantic
name earlier, but its storage/mapping edges and teardown transitions do not become authoritative
until their arm-side contract is established. This keeps an evidence gap from becoming a guessed
cross-pathway abstraction merely because it appeared first in the old implementation.

During migration, each family has one authoritative representation. A temporary bridge may project
new state into an old request shape. Do not dual-write old and new lifecycle maps indefinitely, and
do not use agreement between duplicated states as the permanent invariant.

Exit criteria for each family:

- every operation applicable to the family—create, prepare, bind, use, complete, replace backing,
  page, synchronize, delete, task teardown, and reset—flows through the new aggregate;
- aliases share one storage identity where the contract says they alias;
- object-table release, serializer-ref release, mapping release, resource deletion, and backing
  deletion remain independent where the contract says they are;
- the old family-specific lifecycle maps and retirement keys are removed.

### 5. Replace coherency and writeback ledgers

Move validity decisions, resource-list write invalidation, resident-only content, host copies,
gather/store witnesses, writeback debt, and behavior-selecting content generations into explicit
resource-lifecycle and `ContentState` transitions.

Keep independent observation-only instruments only when they measure a real question and never
select guest-visible behavior.

Exit criteria:

- delayed completion cannot overwrite a newer guest write;
- synchronization pays exactly the named resource or subresource obligation;
- sole-copy status is derived from replica versions;
- each content or backing generation has one meaning and one type.

### 6. Finish immutable command normalization

Replace mutable stream accumulators and output-bearing requests with:

```text
DecodedCommandBuffer
    -> ResolvedCommandBuffer
    -> ResolvedSubmission
    -> ExecutionCompletion
    -> core state transition
```

Render, compute, blit, and resource-state encoders all resolve endpoints through the resource
registry. Encoder state carries lists or replacement slots according to the API contract. Command
buffer splits and continuations preserve the same resource-list and completion envelope.

Exit criteria:

- request inputs are immutable;
- completion outputs are separate typed values;
- the executor receives no raw guest object tag or unresolved reference;
- no backend-specific value is stored in the core command representation.

### 7. Consolidate topology policy

Move placement, import, staging, upload, readback, deferred transfer, and topology-derived batching
decisions into the Vulkan capability profile and policy modules.

Exit criteria:

- the four-cell topology equivalence suite is green;
- topology affects internal plans and metrics, not guest-visible semantics;
- no vendor or driver name participates in a product decision;
- neither topology owns a separate resource lifecycle implementation.

### 8. Reduce the composition layer

Split the broad host interface into capability-specific ports. Make pathway construction explicit,
confine platform conditionals, and make presentation consume semantic resource identity and content
version.

Exit criteria:

- PCI and MMIO adapters contain transport plumbing but no product rule;
- page geometry is explicit at every portable boundary;
- arm-only mapper behavior remains isolated and verified on arm;
- host window and QEMU console are adapters over the same semantic presentation result.

### 9. Delete the old architecture

Remove:

- reset-only `Backend` and `NullBackend`;
- process-global guest-derived engine state;
- redundant `backend-vulkan` feature arms;
- Vulkan types and leases from core/model state;
- raw `type11` and `type7` semantic names;
- sentinel-based multi-kind residency keys;
- duplicate target, GVA, and gather identities that reconstruct one resource;
- retirement side vectors replaced by owned leases and typed release effects;
- mutable output fields on execution requests;
- host and resident caches from the guest-visible device state;
- direct runtime-to-engine calls;
- old maps immediately after their resource family has one new authoritative owner.

Deletion is supported by behavior and lifecycle tests, not by a source-text census. A decoded but
unexercised contract arm remains unless the guest action that would take it no longer exists in the
supported contract.

## Invariants of the finished architecture

1. Raw guest bytes are parsed once into total semantic types.
2. A task-owned guest name resolves to one generational resource lifetime.
3. Resource, storage/backing, view, mapping, backing generation, and content version identities are
   distinct.
4. Mapping update/commit/release and resource create/replace/delete remain separate transitions.
5. Render, compute, blit, resource state, synchronization, display, and presentation use the same
   resource resolver.
6. Resolved submissions preserve segment boundaries and the full resource-participation list.
7. Core owns guest-visible lifecycle, display transactions, and content authority.
8. Guest protocol capabilities and host executor capabilities are separate typed profiles.
9. The topology policy owns only placement and transfer planning.
10. The Vulkan executor owns GPU handles, submission, completion, and fence-safe destruction.
11. Completion facts, not mutable request output flags, advance core state.
12. Reset and deletion are device/session scoped.
13. A topology misclassification can cost performance but cannot change guest-visible behavior.
14. C and Objective-C shims remain transport-only.
15. An unsupported or unknown contract case produces a typed refusal rather than a guess.
16. State whose eviction loses guest work is tied to guest lifetimes and has no invented capacity.
17. Cross-layer invariants are enforced by types, ownership, and crate dependencies rather than
    source scanners.

## Existing work to preserve

Do not rewrite proven components merely because their callers move. Preserve and build around:

- byte-safe borrowed wire views;
- explicit page geometry and pure paging algorithms;
- structural memory-topology classification and purpose-based memory classes;
- bounded guest-RAM reference and import-range types;
- typed declines and the always-on fail channel;
- immutable, content-keyed shader and pipeline caches;
- shared immutable bind snapshots;
- serializer fixture and decoder-divergence instruments;
- pathway-specific live boot verification;
- measurement proxies that observe guest loss or performance without selecting behavior.

## End state

The seam running through the project is:

```text
wire bytes
  -> semantic command
  -> task namespace and canonical resource
  -> storage/view/mapping relation
  -> resolved submission + resource-participation list
  -> declared access and pending lifecycle/content transition
  -> topology-specific transfer plan
  -> Vulkan execution
  -> completion fact
  -> completed lifecycle/content/display transition
  -> guest-visible effects
```

The first implementation work is the semantic protocol boundary and the real executor port. The
canonical resource graph follows. Only after those exist should the unified and discrete policy
modules become the exclusive owners of placement and transfer choices. At that point topology
optimizations can be independent because they no longer define resource identity, coherency, or
lifetime.
