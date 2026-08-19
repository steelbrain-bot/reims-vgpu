# Reims vGPU Architecture Refactor Plan

This is the completed implementation record. The concise, maintained ownership contract and
regression gates now live in [`docs/architecture.md`](../docs/architecture.md).

Status: implementation complete for the established contract; grounded against paired mapper
producer/consumer analysis, controlled serializer oracles, and a driven arm64 boot. Mapper
descriptor composition, identity width, ordinary ownership, rollback, and pathway differences are
established. Four arm teardown races remain explicitly unresolved and are isolated behind the arm
policy rather than hidden behind a shared lifecycle implementation.

Implementation checkpoint (2026-08-19):

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
  context preserves the full ordered segment envelope and resource-participation snapshot. The
  registered-surface resolver now publishes successful construction into `TaskResources` and the
  canonical graph; resource-validity classification asks that registry instead of the legacy
  live-object membership set. Mapper-backed texture resolution likewise relies on the retained
  resource lifetime rather than dual-writing that set. The membership set no longer exists in the
  product build; it is compiled only for synthetic delete-packet tests which intentionally do not
  construct descriptors. Product deletion derives its effect from the canonical resource,
  ref-keyed host materializations, and explicit mapper association.
  The IOSurface texture-to-mapping relation is likewise no longer a product side map: it is stored
  once on the retained `TaskResource`, and validity, replacement, draw-target resolution, deletion,
  and task teardown derive the relation from that owner. The former map is test-only support for
  synthetic fixtures which intentionally do not construct descriptors, matching the treatment of
  the old live-object membership set. A resource's reference and registered-surface mapping
  relations are explicit fields rather than a fixed two-slot collection, so a third insertion or
  ordering-dependent overwrite is no longer representable.
  The retained `TaskResource` itself is now partitioned into immutable construction input (object
  entry, descriptor snapshot, and once-decoded semantic descriptor), retryable graph-relation
  publication state, canonical generational identity, and operational lifetime/use state. Runtime
  consumers receive construction values through read-only accessors; descriptor bytes, relation
  publication, mapping association, lifetime retention, and render-target history are no longer
  peer fields that a new optimization can mutate as though they shared one lifecycle.
  The generic retained-reference registry used by samplers, functions, render/compute pipelines,
  and depth/stencil objects has moved into core beside `ReferenceNamespace`. It owns publication,
  lookup, generational identity, explicit deletion, and task teardown with no capacity eviction;
  model code supplies only each API's retained semantic value and marker type.
  Samplers now retain the protocol-owned `SamplerDescriptor` directly in that registry; the
  one-field model wrapper and its second vocabulary were deleted.
  Render-pipeline buffer classification moved with the semantic pipeline vocabulary into core as
  `VertexBindPlan`. Constant-step staging and stage-input participation are derived once from the
  protocol descriptor; neither model state nor draw orchestration rebuilds those sets. The
  complete retained render-pipeline value—semantic descriptor, prepared shader families, object
  lifetime, and derived bind plan—is now core-owned too; the model keeps only its typed task
  namespace.
  Submission identity allocation, the complete post-validity resource-participation snapshot,
  ordered segment envelope, active segment cursor, and single completion boundary are now owned by
  one core `SubmissionTracker`. Device state no longer carries a context and identity counter as
  independently mutable peers, and the stream walker cannot replace the participation list while
  moving between segments. Executors receive immutable context snapshots; focused walkers and
  direct tools have an explicit standalone context with no invented resource participation.
  Nested child-FIFO drains likewise have one core-owned ordered stack instead of a mutable current
  channel beside a separately maintained mask. The current channel and complete active set are
  projections of that stack; same-channel re-entry and out-of-order exit are typed, fail-visible
  refusals. An inner rescue drain therefore cannot clear the outer drain's re-entry protection.
  Surface validity ordering is no longer a peer device counter. Its single monotonic timeline now
  belongs to `SurfaceMappingRegistry`, and registry transitions issue the order stamp while applying
  guest invalidation, device publication, or guest-page write currency to the named content state.
  A topology-specific materialization path cannot advance or bypass that happens-before relation
  independently of the mapping content it governs.
  Sampled-source identity and revalidation are also one device-owned `SampledContentState` rather
  than a raw generation counter beside three memos, two scratch buffers, and a gather witness.
  Surface, GVA, linear, IOSurface-plane, IOSurface-texture, and zero-copy gather producers all spend
  the same nonzero, non-reused sampled generation namespace. Compute-resident generations remain
  separate because they name residency of a storage result rather than sampled byte identity.
  `PendingWork` remains the single ingress/worker scheduling owner, but its latches and child set
  are now private. MMIO, drain, device, and test paths use named request, clear, merge, take,
  yield, and resume transitions. In particular, adding a child request cannot accidentally replace
  sibling work, and taking a tranche atomically clears exactly the set the worker received.
  Presentation has begun the same partition by behavior. `PresentBackingEvidence` owns decoded
  full-frame publication, per-surface last-presented comparison, and lifetime retirement; recycling
  a surface cannot preserve either half of its predecessor's witness. `PresentBackpressureState`
  owns accepted-but-unpainted count, held `(channel, head)` episode coalescing, and paint
  consumption. Entry-gate, starvation diagnostics, console paint, and window acknowledgement now
  use its semantic transitions rather than independently mutating five fields.
  Capture policy is separate from retained-frame identity as well.
  `PresentCapturePolicy` owns whether a resident carries the next display and the full-versus-light
  capture census; window publication selects that policy and scanout consumes it. Retained CPU
  pixels and mapping/geometry can no longer be mistaken for the policy that decided whether those
  pixels needed to be captured.
  `RetainedPresentFrame` now owns the retained CPU pixels, mapping and geometry, guest-page and
  semantic-content identities, validity, encode obligation, and the warm replacement buffer.
  Light publication, full publication, failed-capture scratch return, and successful encode are
  named transitions. A capture failure therefore cannot publish half a new identity, full capture
  recycles the prior retain as scratch, and resident-carried publication makes the absence of CPU
  pixels part of the new frame value rather than leaving stale bytes beside new metadata.
  Mapping roles are no longer peer integers either. `PresentRoutingState` keeps the mapping named
  by the display transaction, the mapping carried by the host action, the sticky composited early
  front, and the content-boundary transition distinct. Beginning a present changes the presented
  and host roles together; a pre-boundary writeback may change only the candidate or composited
  role; crossing the boundary is idempotent. Scanout and drain now ask whether a mapping is the
  current present instead of reconstructing that predicate from two public fields.
  `PresentConsoleState` closes the other half of presentation routing: console validity, latched
  dimensions, content generation, successful `(mapping, generation)` paint witness, live window
  ownership, and the monotonic publication epoch are one private owner. Establishing geometry,
  recording a real mapping paint, recording an EFI paint without falsely stamping a mapping, and
  advancing publication cadence are distinct transitions. The owner also makes the previously
  implicit distinction between latched dimensions and a valid console explicit; the former remains
  available to dimension fallback while the latter gates established-console behavior.
- Mapper-backed IOSurface texture views now cross the wire boundary as one protocol-owned semantic
  value. The decoder retains the complete 64-bit mapper identity, reuses the three checked nested
  serializer variants, keeps plane and rotation separate, excludes verified unwritten bytes, and
  refuses unknown or inconsistent variants. The former headerless geometry decoder and its
  truncated 32-bit shadow identity have been removed. Runtime resolution now installs and follows
  an explicit `MapperSurfaceRef -> MapperResolvedSurfaceId` relation; it neither narrows the
  reference nor treats the mapper-service reference, resolved surface, canonical backing, or
  page-table mapping namespaces as interchangeable. The legacy integer-keyed mapping table is now
  entered only by projecting the resolved-surface type at its adapter. A high-bit regression fixture
  proves that a mapper reference cannot alias the low 32-bit object with the same suffix.
- Raw pipeline state, vertex formats and step functions, sampler state, blend state, stencil state,
  visibility modes, index types, and pixel-format ordinals now have protocol-owned decoders and
  typed semantic refusals. Runtime preparation retains those errors instead of converting them to
  Vulkan `TranslateReason`s. The raw Metal pixel vocabulary and storage-width/aspect rules are
  authoritative in `reims-vgpu-protocol`; core conversion helpers derive from that authority.
  Direct and indirect draw primitive topology and the stream's sticky cull, winding, triangle-fill,
  depth-clip, and visibility-result state now cross the normalization boundary as protocol semantic
  values as well.
  Full-width setter ordinals are retained until validation; an invalid sticky field refuses every
  snapshot that would consume it until that same field is replaced by a valid setter. Independent
  invalid fields remain independently recoverable. The executor no longer receives raw raster or
  visibility ordinals, reparses guest state, or silently substitutes Metal defaults for values
  outside the contract. The obsolete Vulkan visibility-ordinal translator and backend refusal were
  deleted; Vulkan now maps only the semantic query mode to native flags and capabilities.
  Render-pass load and store actions now follow the same rule. Protocol owns strict `LoadAction`
  and `StoreAction` decoders; resolved color, depth, and stencil attachment state carries those
  enums, not adjacent raw words. Snapshot normalization validates the whole attachment set and
  later store-action overrides; an invalid field blocks draws until that exact attachment state is
  replaced. Unknown actions refuse request construction (including clear-only passes) with their
  exact ordinal instead of being logged and then executed as `DontCare`, a depth clear, or a dropped
  Store. Load census, alias initialization, resident reuse, multisample validation, and writeback
  policy consume the same semantic values, so unified and discrete paths cannot assign different
  fallback meanings.
  The complete draw-preparation refusal vocabulary is core-owned and generic over the executor's
  translation failure. Vulkan supplies only `M2vCacheDecline` through a concrete alias; pipeline,
  function, buffer, texture, reflection-interface, sampler, MRT, and index preparation failures no
  longer live physically under the Vulkan engine.
  Immutable render-pipeline, compute-pipeline, shader-function, sampler, and depth/stencil
  descriptor values are now protocol-owned. Their retained object states live in the model layer;
  `model` no longer names `runtime::pipeline_resolve`, `runtime::compute_exec`, or
  `runtime::mtlb` construction types. Runtime byte decoders produce these semantic declarations,
  and operational modules only construct or consume the retained objects.
- Resolved draw and compute commands no longer carry SPIR-V or another backend-native shader
  payload. Core carries a typed `PreparedShaderId` plus the stage's semantic descriptor-use
  interface. Vulkan assigns that identity to each translated or specialized module, resolves it
  at the executor boundary, and returns a typed refusal for a missing or retired program. The
  registry retains only weak references and removes entries with the owning shader variant, so it
  does not create a second shader lifetime or an invented cache bound. Draw and compute pipeline
  creation, validation breadcrumbs, and native module caches consume the resolved backend variant
  only after crossing the executor boundary. Compute's format-less storage-write decision and
  draw's sampled-layout filter support now enter semantic planning through the executor capability
  service rather than direct runtime calls into the Vulkan engine.
  Translation reflection now crosses into preparation as a core-owned `ShaderInterface`, not as
  the translator's reflection graph. The projection retains stage, descriptor location/count,
  resource kind and access, texture shape and storage format, conservative buffer extent and
  invocation footprint, local size, unsupported stage-level interfaces, and typed constexpr
  sampler state. Draw and compute preparation consume only that projection; the native reflection
  is private to `reims-vgpu-vulkan`. Render translation stages are requested through a Vulkan-owned
  `RenderTranslationStage`, and `reims-vgpu` no longer has a production dependency on the
  translator crate (legacy translator-oracle integration fixtures retain a dev dependency).
  Texture-array descriptor resolution, sampled shape, compute shape/access, storage format, and
  buffer-access classification are methods of the core semantic interface; Vulkan keeps only the
  native-reflection projection and native-module transforms. Compute preparation now consumes the
  canonical translated word representation instead of reparsing a public SPIR-V byte payload and
  carrying a second malformed-module path. Storage-use analysis, neutral sampled-binding
  discovery, format specialization, capability injection, sampler-interface extraction, and
  prepared-stage publication are operations of the Vulkan-owned translated module; the compute
  runtime no longer reads or mutates its raw SPIR-V. Retained render-pipeline state likewise no
  longer owns `CachedShader` or `ShaderVariant`: Vulkan projects every executable numbering into a
  core `PreparedShaderFamily` containing only prepared IDs, the semantic interface, sampler
  declarations, descriptor-use facts, declared bindings, and diagnostic size. Native words remain
  in the Vulkan content cache and resolve only after the executor boundary. This also removes the
  process-global, address-keyed, capacity-reset declared-binding memo; the binding set is derived
  once with the prepared variant and dies with that semantic projection.
  Render translation and publication now cross the device-owned `ShaderTranslationService` on the
  executor. Runtime supplies extracted AIR plus a core stage and retains only the resulting
  `PreparedShaderFamily`; asynchronous render and compute preflight use the same port. Prepared
  render variants also project texture descriptor use by semantic Metal argument index after
  backend relocation. Draw planning therefore no longer imports Vulkan binding-band constants to
  detect sampled-stage collisions or decide whether an unbound texture is statically used. Focused
  fixtures cover all fragment relocation variants and semantic cross-stage collision.
  Compute translation now crosses the same service as an opaque translated-module contract:
  runtime asks only for the semantic shader interface, buffer extents, storage access, neutral
  sampled bindings, sampler declarations, and publication of a prepared program. The executor
  adapter retains the native translated shader and performs SPIR-V specialization behind that
  boundary. Storage-image specialization itself is a core semantic decision over
  `StorageImageFormat`, with typed refusal reasons; backend-native image formats are introduced only
  by the adapter. Reflected samplers carry their Metal argument index separately from relocated
  executable bindings, so compute lookup no longer treats Vulkan descriptor numbering as guest
  identity.
  Render executable variants now also own the complete buffer, texture, and sampler binding
  projection for each relocation combination. Draw orchestration supplies Metal indices and
  reflected descriptor locations without importing Vulkan binding bases or offsets. Reflected
  buffer extent and neutral-substitution policy cross a dedicated executor service, sampled and
  colour-attachment classification are core-owned typed decisions, and the Vulkan-specific GPU
  hang trail is fed through the observation port using semantic formats. The draw planner therefore
  has no executable dependency on the Vulkan crate; remaining textual references document the
  adapter implementation rather than selecting guest-visible behavior.
- The product executor is device/session-owned. Command execution, capability discovery,
  resident-content service, guest-write synchronization, compute residency, readback, and
  presentation are distinct core contracts. Guest-RAM transfer, maintenance, session lifecycle,
  and telemetry are separate compatibility ports; the `Executor` body is now empty and composes
  those narrow services instead of owning their methods.
  The executor now strongly owns a `SessionHandle` containing its guest-derived Vulkan pools,
  resource-front indexes, counters, presenter, residency epoch, and completion/publication
  signals. The process-wide engine retains only the physical context, immutable content caches,
  and the state temporarily borrowed by the active call. Its live-session registry contains weak
  handles solely so a physical device loss can invalidate every live session; it cannot keep a
  deleted vGPU or any of its GPU objects alive. Session activation parks state back in the owning
  handle, reset and release operate through that handle, and resident leases carry the originating
  handle rather than recovering ownership from a numeric session ID. `ResourcePools::new` no
  longer mutates ambient session state; batch closure is published by the reset/device-loss
  transition that owns it.
- Unified/discrete policy and the four import/topology cells are isolated in the Vulkan crate. The
  Vulkan engine and translation implementation live there physically; the old staticlib re-export
  is gone, and the executor adapter owns the device telemetry connection.
  The four-cell equivalence fixture now drives the canonical `ContentState` through imported and
  staged guest-to-GPU materialization, a completed GPU store, and GPU-to-guest synchronization. It
  compares the resulting content versions and replica state, not a preconstructed identical result,
  while retaining topology-specific placement and batching as internal metrics.
  `MemoryTopology` is now classification data only; it no longer exposes allocation behavior.
  `MemoryPlacementPolicy` is the sole interpreter for allocation requests and topology-selected
  batching, and `HostGpuCaps::memory_policy` is the device-scoped access point. Host-pointer import
  remains an orthogonal measured capability, so neither topology policy assumes it is present.
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
  guest imports retired before aliased host views. Ordinary flush and full device reset both drain
  that same ordered effect stream; reset can no longer extract raw views first and revoke their
  imports afterward. A mapping incarnation's packed host pointer, length, exact page footprint,
  and optional GPU import are now one `MappingHostView` aggregate. Its constructor requires the
  footprint to cover exactly the view, and replacement/invalidation atomically removes the whole
  aggregate before emitting import and host-view release effects; the formerly independent
  `contig_ptr`, `contig_len`, `contig_footprint`, and `contig_import` states no longer exist.
  Task-GVA aliases now use the same ownership shape: a nonzero pointer and length can only be
  created as one `HostPageView`, and overlap retirement, stale-view replacement, task teardown,
  and reset consume that non-cloneable token exactly once into the typed release stream. Replacing
  a mapping's GPU import retires and emits the prior import identity instead of overwriting its
  last owner. Raw pointer/length
  pairs are no longer independently mutable fields of either mapping-view family. The semantic
  runtime no longer calls Vulkan
  engine operations directly; its remaining Vulkan dependency is the compatibility request/type
  vocabulary used to construct texture-blit and guest-transfer plans.
  `SurfaceMappingEntry` is no longer one flat record in which logical lifetime, physical page identity,
  content currency, and topology-selected host aliases are peer fields. Logical mapping activity,
  incarnation, and mapper-object association now form `SurfaceMappingLifecycle`; the physical plan,
  generation, condemned fingerprint, page-table source, and surface-walk proof form
  `SurfacePageState`; and the contiguous host view plus its generation-keyed refusal form
  `SurfaceMaterialization`. Retiring a materialization consumes its import and host-view ownership
  without changing the logical mapping or its page plan. Mapping content has one transition object
  for guest-page generation, semantic surface epoch, and decoded validity ordering. Guest writes,
  host publications, and host writes into guest pages update that object through distinct methods,
  eliminating the former split between saturating and nonzero-wrapping generation updates. The
  unused render-flush witness and its non-reproducible observation history were deleted rather than
  retained as apparent lifecycle authority. The content transition object and validity
  happens-before record now live in `reims-vgpu-core::mapping`; the device model embeds that core
  state instead of defining its own authority vocabulary.
  Mapping construction facts are no longer five independently mutable peer fields. A private
  `SurfaceDeclaration` owns an optional complete `SurfaceGeometry { width, height, format }` and
  the optional device descriptor. Absence therefore cannot retain stale dimensions or format, and
  map/remap/unmap/new-internal transitions clear geometry and descriptor together. The validated
  geometry transition is the only production publisher; render, compute, blit, mapper, scanout,
  writeback, object resolution, and reporting consume read-only projections. A behavioral fixture
  pins that remap preserves the mapping registration while atomically withdrawing the entire old
  declaration.
  The device table itself is now a `SurfaceMappingRegistry` keyed internally by protocol
  `SurfaceId`. It is no longer a raw `BTreeMap<u32, _>` named `mappings`, which falsely suggested
  that registered/IOSurface slots were interchangeable with the canonical graph's GPU page-table
  `MappingId`. The mapper-service edge remains `MapperSurfaceRef -> MapperResolvedSurfaceId` and
  projects explicitly into this surface namespace; it is not inserted into the page-table mapping
  graph. Runtime compatibility boundaries still carry the guest's raw `u32`, but the owning table
  cannot be merged with another integer-keyed identity by accident.
  Mapper page-plan adoption is now one registry transition: condemned-fingerprint comparison,
  reprieve classification, logical/page generation advances, old host-view/import retirement,
  page-table publication, mapper-internal association, and optional descriptor publication cannot
  be independently reordered by the runtime resolver. The pure decision helper formerly beside
  the mapper was deleted; tests drive the owned transition itself. Decoded resource-validity
  statements likewise update guest-write currency and the validity quad together.
  The registered-surface pathway now crosses a sibling owned transition: fingerprint comparison,
  generation change, page publication, surface-walk derivation, device descriptor, and retirement
  of the prior materialization are published as one state change. Thus arm mapper and registered
  surface construction remain distinct policies over the same typed surface registry instead of
  being interleaved mutations of a shared raw map.
  Retained task resources now store their registered IOSurface relation as a typed `SurfaceId`,
  converting back to the compatibility integer only at legacy runtime boundaries. The canonical
  resource relation therefore cannot silently become a page-table mapping identity.
  Mapper-service state is also one private aggregate now: the directed MMIO capture, retained
  mapper-device identity, and `MapperSurfaceRef -> MapperResolvedSurfaceId` edges are no longer
  public peer fields. MMIO publishes captures, ring drain consumes or restores them by producer,
  and MAP/UNMAP own edge publication and retirement through named transitions. A zero capture
  cannot erase an established mapper-device identity.
  Presentation's per-surface Clear/Composite classification now belongs to `PresentState` and is
  keyed by `SurfaceId`. Mapping retirement removes it with the other incarnation evidence, so a
  later surface reusing the same wire integer cannot inherit its predecessor's keep-prior routing.
  The semantic generation mirror for compute-resident subresources is now a core-owned
  `ComputeResidencyLedger`: publication, lookup, and byte-window invalidation are its only mutation
  operations. Executor resident availability remains an independent service fact, so losing a
  native resident cannot silently erase or recreate guest-visible content authority.
- Host copies are no longer eight peer fields in guest device state. Surface-, task-texture-,
  GVA-, and native-linear replicas plus the GVA replacement bookkeeping have one
  `HostReplicaState` owner; guest object and mapping identities only name entries in that
  aggregate. Product builds expose none of its maps for mutation. Surface and texture publication,
  row-preserving replacement, object/task retirement, and all native-linear store, resident,
  materialize, lookup, and release transitions now cross the owner. The complete native-linear
  window identity moved with those transitions, so a reused task/object key cannot update bytes,
  descriptor identity, and resident authority independently. GVA restore, touch, successful guest
  landing, eviction, byte accounting, and eviction-witness updates are likewise single owner
  transitions; focused owner tests pin replacement accounting, recency, eviction, generation
  supersession, and materialization. Immutable lookup and level snapshots are the only production
  projections. Existing deletion/repoint/reset behavior crosses the same seam. Observation-only map pairing,
  page-table node, released-page, and stale-view instruments likewise live in a separate
  `DeviceObservations` aggregate instead of beside behavior-selecting lifecycle state. Two stale
  ledgers that had neither producer nor consumer were removed rather than preserved as fictitious
  authority. The map/unmap pairing, page-table-node write guard, and released-page write guard
  state machines now live in `reims-vgpu-core`. They consume only typed state and return typed
  verdicts; runtime retains environment switches, clocks, paging walks, counters, census cadence,
  and failure-channel emission. The `DeviceObservations` aggregate itself is now core-owned and no
  longer stores concrete runtime instrument types. Namespace-reach maxima and map-family cadence
  have joined it as well; semantic `DeviceState` no longer carries observation-only counters as
  peer fields, and the aggregate exposes only monotonic observation transitions and snapshots.
- Linear-buffer and texture-view construction values are protocol-owned. Buffer address
  construction requires an explicit guest page shift, and the texture-view boundary consumes the
  three serializer opcodes into `TextureViewForm`; downstream code can ask for ranged or swizzled
  semantics but cannot branch on the raw opcode.
- The rest of the retained resource-construction vocabulary is protocol-owned as well: linear
  texture level geometry, complete texture declarations, buffer-texture views, indirect-command
  layouts and declarations, and the typed resource-decode refusal. `TaskResource` stores those
  semantic values directly and no model type imports the runtime decoder. The decoder retains only
  byte offsets, length checks, and conversion into the protocol values. Protocol tests now pin
  explicit page geometry, level bounds and row spans, and ICB flag classification.
- Pre-construction write generations, page-exact host-write currency, named GVA Store witnesses,
  GVA plane/resource lifetime, retained page identities, and unpaid-frame ordering now live in a
  core `content_tracking` domain. Core transitions have no logging or host transfer dependency;
  runtime performs page walks and payment and emits observations at the orchestration boundary.
  `DeviceState` no longer stores these behavior-selecting ledgers under runtime-owned types. Their
  device-scoped instances—sampled revalidation, GVA Store proof, pre-construction currency,
  page-exact host writes, compute residency, and pending writeback—now compose through one
  `ContentAuthorityState`. The distinct typed ledgers remain distinct, while task retirement and
  reset cross one content boundary instead of coordinating peer fields at call sites.
  The task-local object families now follow the same ownership rule without collapsing their
  identities: resources, samplers, compute and render pipelines, functions, depth-stencil states,
  indirect-command buffers, fences, and events remain nine independent serializer namespaces
  inside `TaskObjectNamespaces`, while redefine/delete crosses its single task-retirement
  transition. Their former parallel `DeviceState` fields are gone. Existing lifecycle tests pin
  exact per-family retirement counts, and the cut passes the complete drain suite plus strict
  native and Apple Silicon clippy.
  Task-directory entries and their unbounded full-`u32` table now live in core as well; task
  definition cannot invent an object-list page, and task iteration remains tied to guest-owned
  lifetimes. The gfx and IOSurface-command register banks moved into a core `DeviceRegisters`
  aggregate, including the lock-free interrupt/doorbell identities that reset preserves. FIFO
  ingress, child-ring page plans, translation ordering, and nested drain ownership now compose as
  `WorkSchedulingState`. DEFINE/FREE child FIFO are single transitions over admission, pending
  work, translation holds, and the decoded ring cache rather than three ordered mutations in
  runtime. Core owner tests pin full-width task identity, sparse-register bounds, and atomic
  channel retirement.
  The two shortcuts used by packet fixtures which deliberately omit canonical descriptor
  construction are isolated in a `#[cfg(test)]` `SyntheticFixtureState`; neither appears as a
  product `DeviceState` field or resource authority.
  Presentation now has the same device-level seam. Present routing/backing evidence, cursor
  publication, and the guest shared-page handshake compose under one `PresentationState` rather
  than three peer device fields. Cursor glyph publication and the display handshake live in core.
  The handshake fields are private: shared-page reinitialization atomically withdraws every prior
  online witness, acknowledgement is one transition, and a poll returns `Idle`, `Exhausted`, or an
  admitted `Inspect` decision carrying the page, poll count, retry cadence, and first-pulse fact.
  Runtime can no longer advance poll cadence, spend a retry, or combine an address with a different
  pipe index independently. Display transaction-shape dedupe remains observation-only in the core
  `DeviceObservations` owner instead of contaminating lifecycle state.
  The arm mapper service is now a core state machine as well. Its pending producer capture,
  mapper-device identity, and `MapperSurfaceRef -> MapperResolvedSurfaceId` edges move together;
  capture consumption is scoped to the publishing entry, zero cannot erase an established device
  identity, and surface retirement removes every related mapper edge. A core fixture uses nonzero
  high bits to pin the 64-bit mapper namespace against accidental narrowing. At device scope,
  `SurfaceState` composes this arm-only service beside—never into—the canonical surface-mapping
  registry. Shared declaration/page/content/materialization state is therefore common, while arm
  mapper lookup and x86 registered-surface construction remain distinct rails.
  Raw object ordinals are now confined to the protocol/wire boundary in this path. Draw-time
  sampled-resource diagnostics name `mapper_iosurface_texture_view`, `iosurface_plane_view`,
  `texture_view`, or `buffer_texture`; they no longer carry `type=11`, `type=5`, or `type=8` as the
  semantic identity of an already decoded object.
  Indirect-command-buffer declarations and their independently bound command-memory association
  likewise moved into a core generational registry, while wire decoding, host reads, fill replay,
  and refusal emission remain in runtime.
- Sampled-window identities now carry a typed page-table `MappingId` and `ByteOffset` in core
  rather than another raw `(u32, u64)` mapping namespace in runtime. Their stable content-key fold
  is core-owned and shared by derivation instead of restating the hash constants. Completion-stamp
  records normalize their raw index through a protocol `StampWait`; wrapping satisfaction,
  coalesced-versus-queued publication state, and the pending drain stamp now live in core. Runtime
  retains packet decoding, guest-page access, queue submission, IRQ publication, and census
  emission.
- Event and encoder-fence semantics now live in core as one pure synchronization state machine.
  Signal/wait decisions, typed reasons, monotonic generation storage, task-local serializer
  namespaces, deletion, task teardown, and reference-reuse generations moved together; runtime
  retains wire-command adaptation and refusal emission only. Equal event and fence integers cannot
  alias, while render, compute, and blit deliberately share one fence namespace. The historical
  `runtime::plan` façade is deleted rather than kept around the core owner.
- The sampled-window witness is now a core state machine rather than a runtime cache with hidden
  correctness authority. Core owns its unbounded lifecycle, guest/device writer verdict,
  diagnostic-audit schedule, and the rule that an audit disagreement spends the refuted content
  generation. Runtime supplies page-exact executor readings, selects the diagnostic density from
  process configuration, folds the already-resolved host spans, and publishes counters and typed
  failures. The adapter can no longer change reuse validity by rearranging logging code.
- Ref-keyed host materializations now use a core-owned, unbounded `MaterializationRegistry` keyed
  by typed task/object ownership and semantic byte windows. Core owns lookup and task, object,
  range, and reset retirement; runtime owns only the host import and packed-buffer payloads. A
  host optimization can no longer redefine when a guest resource or one of its byte windows dies.
- Registered-surface backings, IOSurface plane views, buffer textures, ordinary texture views, and
  narrow/wide heap textures now cross the byte boundary as distinct protocol-owned semantic
  descriptors. The type-5 descriptor preserves its outer surface relation even when the nested
  operation is absent, unknown, or geometrically invalid, and carries that nested state explicitly
  rather than converting the whole resource into a miss. The type-8 decoder is total over its
  supported families; compute no longer peeks at serializer opcodes and reparses an embedded
  texture body to decide whether it received a heap texture, buffer texture, or view. Blit,
  compute, mipmap, sampled-load, render-target, and view-chain behavior consume the one retained
  semantic descriptor for the resource lifetime. Raw construction bytes remain only for boundary
  diagnostics and the separate first-construction registries for serializer/function objects.
- Draw encoding now takes an immutable `DrawEncodeRequest`. Allocation generation is resolved once
  and passed explicitly; a failed resident LOAD returns its recovered CPU seed as a typed
  resolution instead of writing it back into the request. Visibility output, CPU chain pixels, and
  the exact resident identity now leave encoding in `DrawChainResult`, so abandonment consumes the
  identity that execution actually established rather than re-deriving it from mutable mapping
  state. The generic resolved-command envelope also boxes its large owned blit variant, keeping the
  command discriminant compact without changing the payload's ownership or lifetime.
  The formerly monolithic physical draw file is now named for semantic execution rather than its
  historical backend, and its first ownership block has been extracted: `draw::resident` owns
  render-target identity, allocation-generation derivation, resident-content currency, readback,
  and Store publication. Those policies consume the executor's capability and resident services;
  they do not select a native API or memory topology. Reflected sampled-image dimensionality is a
  core preparation projection, including the explicit refusal of cube shapes the current executor
  request cannot represent, rather than a locally reconstructed native-image shape.
  Sampled-resource resolution is now a separate `draw::sampled_source` module: task-resource
  lookup, IOSurface seed selection, guest-run gathering, zero-copy eligibility, buffer/index
  materialization, format-preserving CPU conversion, and sampled-content identities produce a
  semantic source request consumed by draw execution. The execution module has fallen from roughly
  9,900 to 5,400 lines and no longer owns or can reach the private intermediate state of that
  resolution ladder. This is an ownership split, not a backend fork: capability checks enter
  through the executor, while guest identities and content authority remain common.
- The deleted `backend-vulkan` compatibility feature has been removed from the repository's actual
  verification commands as well as Cargo and QEMU. The feature matrix now exercises the shipping
  unconditional Vulkan dependency with only the optional `host-window` adapter enabled. Argument
  groups at recursive texture resolution, whole-plane GPU copy, chain abandonment, native scratch
  conversion, and macOS window construction are typed aggregates rather than growing positional
  parameter seams.
- The one-line `runtime::{m2v_cache, spirv_bind, spirv_vertex_input, gpu_hang_trail}` compatibility
  modules have been deleted. Their APIs are now visibly owned by `reims-vgpu-vulkan`, and
  backend error, refusal, and observation types used by orchestration are concentrated at the
  executor compatibility boundary. Outside that adapter, production runtime code no longer calls
  the Vulkan engine directly. Integration fixtures exercise the sibling backend crate at its own
  public boundary. Translation and SPIR-V preparation still form the explicitly documented
  compatibility vocabulary to migrate behind a narrower executor service.
- The one-line `contract::{draw, endian, extent, fnv, pixel_format, vertex_step, visibility}`
  façades have also been deleted. Product code and integration fixtures now name the owning core
  or protocol crate directly, so the composition crate no longer supplies a second path for those
  semantic values. Render-pass load/store actions and compute/mesh dispatch geometry have moved
  with their behavioral tests into `reims-vgpu-protocol`, consuming two more substantive
  `contract` modules. Page-table geometry and explicit PFN conversion have moved into
  `reims-vgpu-paging`, eliminating the device-local GVA naming façade as well. The remaining
  checked geometry helpers moved into protocol geometry. The remaining IOSurface catch-all has now
  been split rather than renamed: mapper ring records and their typed selector live in protocol,
  IOSurface device-descriptor records/window derivation live in protocol, page-entry interpretation
  and the host-neutral mapper-internal walk live in paging, and runtime owns capture registers plus
  failure-channel projection. `MapperCapture` carries `MapperRequestKind` rather than an ordinal.
  The product `contract` module and directory are deleted; its twelve cross-owner fixtures remain
  as a test-only module and exercise the new owners directly.
- Task definition and deletion no longer mutate runtime census state from inside `DeviceState`.
  The model returns a typed definition kind plus exact per-namespace retirement counts, shares one
  namespace-retirement transition between redefine and delete, and preserves their distinct
  replica and address-space cleanup. FIFO orchestration alone maps those semantic effects onto the
  existing observation routes. A behavioral model test pins first definition, same-root and
  new-root redefinition, successful deletion, repeated deletion, and exact namespace counts.
- Child-channel admission is now a pure model predicate with its refusal/census adapter at the
  runtime boundary. Mapping page-plan invalidation likewise returns a typed effect distinguishing
  retired page/view state from a dropped host replica; runtime publishes the latter instead of the
  model mutating census state. Task/object-list and mapping mutators now follow the same rule:
  `DeviceState` returns a typed `StateMutationDecline`, while the composition-owned runtime device
  maps that result onto the fail channel. Observation formatting implementations for semantic
  model events live under `observe`, not beside the state machine. A behavioral test proves that
  invoking semantic state directly is quiet and that the runtime composition reports the same
  typed refusal. Production model state and registers contain no executable observation call.
- `runtime::Device` is now the composition root for one virtual GPU. It owns the semantic
  `DeviceState`, the injected executor/session, and address-bound host materializations. Runtime,
  device, scanout, and optional host-window paths consume that aggregate instead of recovering an
  executor or materialization registry from semantic state. `DeviceState` constructors no longer
  construct Vulkan objects or read executor policy. Reset returns a typed semantic effect;
  composition performs executor reset, host release, materialization retirement, and failure
  publication. Focused tests prove that resetting one device preserves its injected executor and
  cannot reset another, that diagnostic audit policy survives reset, and that unresolved
  translation holds are reported only at the composition edge.
- Diagnostic gather policy is now selected at the composition edge and injected into semantic
  state. `DeviceState` no longer reads process environment policy through a runtime constructor,
  and reset preserves the injected policy; a focused lifecycle test pins that behavior.
- Explicit IOSurface plane selection is fail-closed at the protocol boundary. A non-planar surface
  accepts explicit plane zero, while a nonzero plane can no longer be discarded and silently
  aliased to the whole surface. The protocol test covers both outcomes.
- The obsolete in-crate `backend::vulkan` re-export façade is deleted. Runtime translation and
  executor adaptation, integration tests, and documentation now name the extracted
  `reims-vgpu-vulkan` crate directly. The two composition hooks formerly hidden in the façade—
  telemetry installation and drain-thread attribution—are owned by the executor adapter.
- The bounded memo audit distinguishes lifecycle state from derived performance data. The three
  byte-bounded CPU memos are byte-exact and revalidated on every lookup, so eviction costs only
  conversion/re-upload. The GVA cache can evict only entries whose bytes are already present in
  guest RAM; sole-copy entries remain admitted over the cap and fail-visible. These are retained
  under the project's explicit recomputation-cache exception, not treated as resource authority.
- Verification at the preceding full-workspace checkpoint: `cargo test --workspace -- --test-threads=1` passed after the
  core content/ICB, protocol-descriptor, stamp, and gather-witness ownership tranches. It covers
  1,265 product library tests plus its integration suites, 705 Vulkan tests, 152 wire tests, 99
  core tests, 36 paging tests, 35 protocol
  tests, 23 memory tests, the remaining workspace crates, and doc tests. The hardware-independent
  Vulkan integration fixtures for draw, compute, batching, storage images, topology-equivalent
  content, reset, and device loss are included. Oracle-backed tests which explicitly require the
  external serializer harness remain ignored by their existing contract. Product, Vulkan, paging,
  protocol, and wire crates also pass fresh `aarch64-apple-darwin` checks;
  native Linux covers the x86 compile boundary. `git diff --check` is clean. Architectural
  deletion still gates plan completion.
  The subsequent materialization and total resource-descriptor tranche passes 1,266 product
  library tests, 37 protocol tests, and its focused core/object/blit/compute/mipmap/view suites;
  the immutable draw/output tranche also passes the 1,266 product library tests. The corrected
  feature matrix passes native Linux, cross-compiled Apple Silicon, the option ROM, and both
  formatting cells; native and `aarch64-apple-darwin` all-target clippy runs pass with `-D warnings`.
  After the façade deletion and observation-state migration, the full workspace matrix passes
  again: 1,226 product library tests, 705 Vulkan tests, 152 wire tests, 118 core tests, 43 protocol
  tests, and the remaining workspace and doc-test suites. The lower product count reflects tests
  moving with their owners into core and protocol, not deleted coverage. Native and
  `aarch64-apple-darwin` all-target clippy still pass with `-D warnings`; the feature matrix passes
  both formatting cells, Linux Vulkan/window, Apple Silicon Vulkan/window, and the option ROM.
  `git diff --check` is clean. The later task-lifecycle effect tranche passes a fresh full workspace
  run with 1,222 product library tests plus all integration and doc-test suites; the lower count
  again reflects moved tests. Its product all-target check and the focused child-channel and
  mapping-invalidation tests pass. After the IOSurface split, a second full workspace run passes
  the same 1,222 product tests and all integration/doc suites. Native and Apple Silicon all-target
  clippy pass with warnings denied; the complete feature matrix passes after this
  checkpoint. After diagnostic-policy injection, strict plane selection, and deletion of the old
  Vulkan façade, the full workspace passes again with 1,223 product library tests plus every
  integration and doc-test suite. Native and Apple Silicon all-target clippy pass with warnings
  denied, the complete feature matrix passes, and `git diff --check` is clean.
  After extracting executor/materialization ownership and semantic observation effects, the full
  workspace passes with 1,224 product library tests plus every integration and doc-test suite.
  Native and Apple Silicon host-window all-target clippy pass with warnings denied; the complete
  feature matrix passes Linux Vulkan/window, Apple Silicon Vulkan/window, the option ROM, and both
  formatting cells. `git diff --check` is clean.
  After the render/compute translation ports, draw-planning boundary, and observation-instrument
  relocation, the full workspace again passes 1,224 product tests, 704 Vulkan tests, 119 core tests,
  37 observation tests, and every integration and doc-test suite. The two tests removed from the
  Vulkan count moved with the shared sRGB instrument into the observation crate. Native and Apple
  Silicon host-window all-target clippy pass with warnings denied; the complete compile matrix
  passes Linux Vulkan/window, Apple Silicon Vulkan/window, the option ROM, and both formatting
  cells. `git diff --check` is clean.
  After splitting mapping lifetime, page plans, host materialization, and content currency, all
  1,227 product library tests plus the product integration and doc-test suites pass serially. The
  three added content-transition tests pin nonzero wrap, host-only publication, and ordered
  validity operations. `cargo check -p reims-vgpu --tests`, native all-target host-window clippy
  with warnings denied, and `git diff --check` are clean. The next full workspace and cross-target
  matrix follows after the surrounding mapping-registry tranche is complete.
  After moving synchronization, retained reference namespaces, render-pipeline semantics, and
  mapping content transitions to core, the product suite passes with 1,220 library tests plus all
  integration and doc-test suites, and core passes 128 tests. The seven fewer product tests are the
  mapping and synchronization state-machine tests now run by their core owner. Native and Apple
  Silicon all-target host-window clippy remain clean with warnings denied, as does
  `git diff --check`.
  After the resident-target and sampled-source draw split, the serial product suite passes with
  1,217 library tests plus every integration and doc-test suite; the product count falls by three
  because sampled-shape behavior moved to core, whose suite now passes 130 tests. Native and Apple
  Silicon all-target host-window clippy pass with warnings denied, `cargo fmt --all -- --check`
  passes after formatting, and `git diff --check` remains clean. A fresh full-workspace serial run
  also passes all product, Vulkan (704), wire (152), core, protocol, paging, memory, observation,
  integration, and doc-test suites; oracle-backed tests retain their documented ignored status.
  After making mapping declaration atomic, the serial product suite passes with 1,218 library
  tests plus all integration and doc-test suites. Native and Apple Silicon all-target host-window
  clippy pass with warnings denied, and formatting/diff checks are clean.
  After separating the `SurfaceId` registry from page-table `MappingId`, owning mapper page-plan
  adoption, and coupling decoded validity updates, the serial product suite passes with 1,219
  library tests plus every integration and doc-test suite. Native and Apple Silicon all-target
  host-window clippy pass with warnings denied; formatting and `git diff --check` are clean.
  After adding the registered-surface adoption transition and its behavioral fixture, the same
  gates pass with 1,220 product library tests.
  After centralizing topology allocation behavior and mapper-service state, a full serial workspace
  run passes every unit, integration, and doc-test suite. Native and Apple Silicon all-target
  host-window clippy pass with warnings denied; formatting and `git diff --check` are clean. Two
  mapper-service state tests raise the product library count to 1,222. A focused lifecycle fixture
  then pinned retirement of presentation write classification on surface-id reuse, raising it to
  1,223. After moving primitive topology and sticky raster/visibility state to the semantic
  normalization boundary, the full serial workspace passes with 1,225 product library tests plus
  every integration and doc-test suite. Native and Apple Silicon all-target host-window clippy
  pass with warnings denied. The two added behavioral fixtures pin full-width sticky-state refusal,
  recovery by field replacement, and refusal of unknown primitive topology without an executor
  fallback. The adjacent semantic pass-action tranche then passes a fresh full serial workspace
  with 1,226 product library tests plus all integration and doc-test suites. Native and Apple
  Silicon all-target host-window clippy remain clean with warnings denied; its regression proves
  unknown load/store actions cannot construct an executor request. Extending the same state to
  depth/stencil and override-time validation raises the product suite to 1,227; a fresh full serial
  workspace and both clippy targets pass again. The added snapshot fixture proves invalid color,
  depth, and stencil actions remain distinct and recover only when their exact attachment state is
  replaced. The fixed-function pipeline-state tranche then raises the product suite to 1,228:
  blend descriptors and nontrivial depth/stencil descriptors normalize once into semantic core
  state, and unresolved bound state or unknown blend, compare, and enabled-face stencil ordinals
  are typed draw-preparation refusals rather than log-and-continue degradation. A fresh full serial
  workspace run passes every unit, integration, and doc-test suite; native and Apple Silicon
  all-target host-window clippy pass with warnings denied, and formatting plus
  `git diff --check` are clean. The subsequent prepared-draw boundary raises the product suite to
  1,230: semantic request construction fixes one completion route before submission, route
  conflicts are typed preparation refusals rather than branch-order choices, executor completion
  and materialization accounting live behind the immutable handoff, and pixel diagnostics are an
  observation-only consumer downstream of execution. A fresh full serial workspace run passes all
  unit, integration, and doc-test suites; native and Apple Silicon all-target host-window clippy
  pass with warnings denied, and formatting plus `git diff --check` remain clean. Target planning
  now lives behind the same boundary: resident-chain, deferred GVA Store, surface Store, and
  resident LOAD jointly select executor identity, readback policy, and completion ownership in one
  operation. Focused fixtures pin surface and GVA-load routes plus fail-closed missing resident
  identity. Fixed-function normalization is also a dedicated module rather than a collection of
  helpers embedded in orchestration. The resulting full serial workspace passes with 1,231 product
  library tests and every integration/doc-test suite; native and Apple Silicon host-window
  all-target clippy, formatting, and `git diff --check` are clean. The adjacent attachment audit
  found and closed another fail-open seam: an MRT attachment whose geometry differed from color0
  was logged and omitted while the other slots executed. The builder now refuses the whole pass,
  and a behavioral fixture proves a two-target pass cannot collapse into a one-target draw. The
  serial product suite rises to 1,232; both host-target clippy passes and formatting/diff checks
  remain clean. Attachment construction now has its own module and a total result boundary:
  expected absence is `Ok(None)`, while invalid load/store actions, unresolved source or resolve
  identities, incompatible resolve targets, and inconsistent pass geometry are distinct typed
  refusals. Source and resolve roles are typed rather than stringly identified, and the refusal
  latch includes class, slot, and subject identity so observation cannot merge different failures
  merely because they occupied the same slot. Focused core and product fixtures pass; full
  workspace and dual-target verification follows with the surrounding bind-planning cut. The
  first such cut now returns one `BoundBufferPlan`: vertex/fragment buffer materialization,
  reflected extent and access, zero-copy eligibility, dynamic attribute stride, vertex format and
  step normalization, and stage-in presence are owned together rather than assembled inside
  execution. The full serial workspace passes with 1,233 product tests, 703 Vulkan tests, 152 wire
  tests, 131 core tests, and all integration/doc-test suites; native and Apple Silicon host-window
  all-target clippy pass with warnings denied, and formatting plus `git diff --check` are clean.
  Sampled texture and sampler planning now have their own complete modules as well. Both stream and
  reflected sampler collisions refuse before execution, reflected shape gaps no longer fall back
  to 2D, and a focused fixture proves a colliding reflected declaration cannot produce a partial
  sampler list. A fresh full serial workspace passes with 1,234 product tests, 703 Vulkan tests,
  152 wire tests, 131 core tests, and all integration/doc-test suites; both host-target clippy
  passes remain clean with warnings denied. Final `DrawRequest` assembly is now isolated from
  execution, Store, and diagnostics: all resolved ingredients cross one request-plan aggregate,
  which returns the immutable request and its single completion route. Resource/hang diagnostics
  inspect only the finished request, and MRT route counters no longer live in the semantic planner.
  The full serial workspace remains green at 1,234 product tests; native and Apple Silicon
  host-window all-target clippy, formatting, and `git diff --check` remain clean. The first two
  scheduler ownership cuts now replace loose `DeviceState` fields: completion publication owns
  debt, ordered-rail handoff, wait classification, visibility progress, and the exact FIFO
  timelines held for an unmet word; translation scheduling owns defer/ready, sibling holds, present
  barriers, episode accounting, channel retirement, and reset evidence. The full drain suite
  passes 116 behavioral tests, and the fresh full serial
  workspace passes 1,234 product tests, 703 Vulkan tests, 152 wire tests, 135 core tests, and all
  integration/doc-test suites. Native and Apple Silicon host-window all-target clippy remain clean
  with warnings denied; formatting and `git diff --check` are clean. The first shared transfer-plan
  boundary is now concrete: a bounded guest-page destination produces a topology-independent
  `GuestPageTransferPlan` distinguishing padding-preserving rectangles from dense detile/scatter,
  and the Vulkan executor materializes that plan. Guest reads now have the matching allocation-free
  `GuestReadTransferPlan`: checked pages either expose one exact direct stretch and a complete
  gather iterator, or explicitly remain CPU-only. Buffer, compute-buffer, and sampled-texture
  execution consume that one classification before applying host import capability and Vulkan
  alignment rules. The former guest-memory compatibility port is
  split into guest-page transfer, completion ordering, and guest-import lifetime capabilities, so
  transfer planning cannot retire imports or publish completion words. A fresh full serial
  workspace and both host-target clippy passes remain green at the same product/Vulkan/wire counts,
  135 core tests, and 24 memory tests. Every runtime resident read now consumes one atomic
  `ResidentReadPlan` for readiness, content epoch, and absent-after-reclaim evidence; semantic
  lifetime retention and content-state transitions remain separate operations. This prevents one
  preparation decision from combining mutable registry facts observed across three backend
  transactions. The draw and blit suites pass with 161 and 60 focused tests respectively, and
  focused resident-ownership and Vulkan classification tests are green, including the invariant
  that a live resident never simultaneously carries reclaim history. Direct host-window
  publication now has the same single-owner shape: one engine transaction verifies presenter
  ownership, maintains the resident working set, validates identity and geometry, and returns the
  exact `PresentationSource` that may be published. The device no longer combines that result with
  a separately published attached bit or reconstructs the checked source. Focused Vulkan
  presentation and device lifecycle tests are green. Deferred submission ownership now crosses a
  dedicated `SubmissionBatchService`: completion waits and tranche-tail flushing no longer control
  one open batch through unrelated completion and maintenance ports, and the waiter result is a
  typed submitted/already-in-flight transition rather than a boolean. The 116 drain fixtures remain
  green. The native-window port no longer exposes Vulkan `DrawError`; its boundary error preserves
  the backend's exact decline slug, fields, and display text while exposing only the semantic
  presenter-detached recovery fact. Focused error-preservation and all 14 host-window tests pass.
  The window-to-executor call now carries one immutable frame offer rather than independent
  resident and CPU options, keeping sequence, geometry, fallback bytes, and the preferred resident
  in one contract. Four multisample rules evaluated before submission now return core-owned
  `DrawPreparationDecline` variants instead of constructing Vulkan `DrawReason`; the Vulkan-only
  capability and execution refusals remain inside the executor. The focused 107-test draw suite
  and core decline tests are green. The complete buffer, sampled-texture, sampler, and final
  request planners now return only that core preparation vocabulary; `execution.rs` wraps each
  completed plan once at the executor boundary. No planner module imports or constructs the
  backend error type. Draw orchestration now returns a product-owned `DrawAttemptError`: its
  preparation arm contains the core decline directly and its execution arm contains only a
  lossless `ExecutorDiagnostic` projection (slug, fields, and display detail). The native
  `DrawError`, including its translator-fixed payload, terminates inside the executor adapter and
  no longer inhabits draw orchestration. Focused tests separately pin semantic preparation and
  native executor-diagnostic preservation. A fresh full serial workspace
  passes with 1,234 product tests and all integration/doc-test suites; native and Apple Silicon
  host-window all-target clippy pass with warnings denied, and formatting plus `git diff --check`
  are clean. After the diagnostic cut, the 107 public draw tests and 24 execution-split tests pass,
  both host-window and default-feature product checks compile, and `git diff --check` remains
  clean. Retained pipeline resolution, semantic blend construction, sample/interface admission,
  and executor-capability geometry validation now terminate in one `PipelinePlan`; downstream bind
  planning cannot receive peer pipeline values from a partially accepted preflight, and absence of
  a color target is an explicit no-plan result. The same focused draw suites and core decline tests
  remain green. The full serial workspace then passes with 1,235 product tests, 704 Vulkan tests,
  152 wire tests, 135 core tests, and every integration/doc-test suite. Native and Apple Silicon
  host-window all-target clippy pass with warnings denied; formatting and `git diff --check` are
  clean. The Apple target reports only the existing third-party `block` future-incompatibility
  notice.
  The completed draw-planning and submission-lifecycle tranche then passes a fresh full workspace
  at 1,236 product tests. Submission identity, post-validity participation, segment movement, and
  completion are one core tracker; successful draw completion owns target-use publication. The
  subsequent child-drain, validity-registry, and sampled-content ownership cuts pass the full
  workspace at 1,237 product tests plus every integration/doc-test suite. Native and Apple Silicon
  host-window all-target clippy pass with warnings denied, formatting and `git diff --check` are
  clean, and the only diagnostic remains the third-party `block` future-incompatibility notice.
  Named pending-work transitions raise the product suite to 1,239; presentation backing and
  backpressure owner fixtures raise it to 1,241. Fresh full-workspace runs after both tranches pass
  every unit, integration, and doc-test suite, and native plus Apple Silicon host-window all-target
  clippy remain clean with warnings denied. The capture-policy split preserves the 28-test scanout
  suite and passes both default-feature checking and strict host-window clippy. The retained-frame
  aggregate preserves all 28 scanout and 116 drain tests, adds an owner-level publish/rollback/
  recycle fixture, and passes the host-window device suite. Its fresh full-workspace checkpoint
  passes with 1,242 product tests and every integration/doc-test suite; native and Apple Silicon
  host-window all-target clippy pass with warnings denied, formatting and `git diff --check` are
  clean, and the only diagnostic is the existing third-party `block` future-incompatibility notice.
  The mapping-role routing cut then adds one owner-level transition fixture and passes a fresh
  full-workspace checkpoint at 1,243 product tests. Native and Apple Silicon host-window
  all-target clippy, formatting, and `git diff --check` remain clean under the same conditions.
  The console/paint owner adds the second routing fixture and passes the full workspace at 1,244
  product tests. Both strict host-window clippy targets, formatting, and `git diff --check` remain
  clean; Apple targeting still reports only the third-party `block` notice.
  The prepared-presentation boundary keeps the 1,244 product tests green and raises the core suite
  to 138 with an exact-source fixture. Every workspace integration/doc-test suite, native and Apple
  Silicon strict host-window clippy, formatting, and `git diff --check` pass; the same third-party
  notice is the only diagnostic.
  Total window payload, typed completion route, single-source resident geometry, and private
  presentation-source construction keep the full workspace green at 1,244 product tests and raise
  the core suite to 139. Native and Apple Silicon strict host-window clippy, formatting, and
  `git diff --check` pass with only the same third-party notice.
  Host-replica, host-materialization, child-work, and observation ownership cuts now pass the full
  `--workspace --all-targets --features host-window` checkpoint: 1,279 product tests, 140 core
  tests, 720 Vulkan tests, 152 wire tests, and every workspace integration target pass (with the
  explicitly ignored oracle/measurement cases unchanged). Native workspace clippy and Apple
  Silicon product clippy remain warning-clean; the only toolchain notice is still third-party
  `block`. The current host-window documentation build succeeds after using the feature surface
  left by deletion of the `backend-vulkan` façade.
  The subsequent task-directory, register-bank, channel-scheduling, test-fixture, and presentation
  cuts pass a fresh full-workspace checkpoint at 1,276 product tests, 147 core tests, 720 Vulkan
  tests, 152 wire tests, and every integration target before the final display-state-machine
  fixtures were added. The completed display seam raises core to 150 tests; all 116 drain and 28
  scanout behaviors pass, and native plus Apple Silicon host-window all-target clippy remain clean
  with warnings denied. Formatting and `git diff --check` pass; the sole notice remains the
  third-party `block` future-incompatibility warning.
  The core mapper-service extraction adds two owner fixtures and raises the full checkpoint to 152
  core tests while preserving all 1,276 product, 720 Vulkan, and 152 wire tests. Workspace clippy,
  Apple Silicon product clippy, the host-window documentation build, formatting, and
  `git diff --check` all complete successfully. The documentation build retains its pre-existing
  unresolved-link warnings; it introduces no build failure.
  Presentation viewport fitting and pointer projection now live in core as
  `PresentationViewport`, `aspect_fit_viewport`, and `pointer_to_guest`; both the native-window
  adapter and Vulkan presenter consume that one semantic rule. The former Vulkan-owned viewport
  module was deleted. Two core fixtures, all 15 host-window presentation tests, and all 12 Vulkan
  window-present tests pass.
  Composition-owned bound-buffer materializations and retained GVA resources now retire in typed
  task-definition, task-deletion, object-list-replacement, and object-deletion transitions beside
  the corresponding semantic mutation. `ReplacePhysical` likewise owns its GVA-resource,
  bound-buffer, host-copy, and mapping-page invalidation effects in one transition. Product
  lifecycle packets no longer depend on manually ordering those retirements before a
  `DeviceState` mutation. Preconstruction write currency retires inside semantic task and
  object-list lifetimes instead of hitching a ride on a host-buffer helper. Three owner-level
  transition fixtures, all 50 object tests, and all 116 drain tests pass.
  The final serial `--workspace --all-targets --features host-window` checkpoint passes with 1,278
  product tests, 154 core tests, 715 Vulkan tests, 152 wire tests, and every integration target;
  the explicitly ignored measurement/oracle cases remain unchanged. Native workspace clippy and
  Apple Silicon product clippy pass with warnings denied. The host-window documentation build,
  formatting, and `git diff --check` pass; rustdoc retains 116 pre-existing unresolved-link
  warnings, and Apple targeting retains only the third-party `block` future-incompatibility notice.
  A post-completion lifetime audit then found one remaining class of reusable wire names in durable
  execution state. GVA writeback debt, Store and gather witnesses, compute residency, resolved blit
  commands, and resource-state commands now carry the canonical generational `ResourceId` instead
  of `(task, object-table reference)`. Compatibility APIs recover a wire reference from the graph
  only at the final legacy host boundary; it is no longer an executor identity. Resource deletion
  also retires native linear residents before removing the graph edge needed to name their owner.
  Focused fixtures prove that deleting and recreating the same object-table slot cannot inherit
  writeback debt, a gather witness, or compute residency from the retired generation. The audit
  also removed the last test that parsed a QEMU source file to assert a call shape. The cursor
  allocator bound and shim call ownership remain documented review invariants; actual ABI constants
  continue to be checked at the Rust/C header boundary. The resulting serial full-workspace
  checkpoint passes 1,277 product tests, 156 core
  tests, 715 Vulkan tests, 152 wire tests, and all integration and documentation targets; only the
  existing explicitly ignored measurement/oracle cases remain. Native workspace and Apple Silicon
  product clippy pass with warnings denied. The complete feature matrix passes Linux
  Vulkan/window, Apple Silicon Vulkan/window, the option ROM, and both formatting cells; the
  host-window documentation build and `git diff --check` also pass with the same 116 pre-existing
  unresolved-link warnings.

Completion audit (2026-08-19):

| Phase | Exit-criteria evidence | Verdict |
|---|---|---|
| 1. Semantic vocabulary | Protocol owns typed task/object/serializer/resource/storage/mapping identities; mapper fixtures retain 64-bit refs and independent plane/rotation; unknown variants decline by type. | Complete |
| 2. Executor port | Core owns `ResolvedSubmission`, ordered command buffers, `ExecutionCompletion`, and `ExecutionPort`; product draw/compute execution reaches Vulkan only through `VulkanExecutor`; scripted executors validate identity, kind, and completion count. | Complete |
| 3. Device-scoped Vulkan | `VulkanExecutor` owns a `SessionHandle`, resident leases, imports, presenter, submission state, and reset; live-session registration is weak and immutable caches remain content-keyed. | Complete |
| 4. Canonical resource graph | `TaskResources` and the core graph own generational resources, storage/view/backing/mapping relations, participation, deletion, and in-flight retention. Durable GVA debt/witness, gather, and compute-residency keys carry `ResourceId`, so a reused object-table slot cannot inherit the retired generation. Product texture-to-mapping and live-object side maps are gone; mapper and registered-surface policies remain distinct. | Complete within established mapper contract |
| 5. Content authority | Resource versions, mapping currency, sampled identity, GVA Store witness, pending writeback, host replicas, and executor materialization use typed transitions; delayed-completion, sole-copy, and stale-slot fixtures pin ordering and generation isolation. | Complete |
| 6. Immutable normalization | Core command buffers carry resolved generational endpoints; `PreparedDraw` and compute/blit/resource-state commands are immutable, carry no unresolved task-local object reference, and return operation outputs only in typed completion facts. | Complete |
| 7. Topology policy | Structural classification has one owner; sealed unified/discrete policies choose only memory requests and batch defaults. Four-cell tests prove every policy/capability combination retains a valid correctness route. | Complete |
| 8. Composition reduction | Host capabilities are split into narrow ports; page geometry is explicit; presentation viewport/source/result are core semantics; PCI/MMIO use the shared shim and arm mapper behavior stays in its pathway policy. | Complete |
| 9. Old architecture deletion | Reset-only backend, backend feature fork, contract façade, direct draw-to-Vulkan module, raw type-7/type-11 semantics, sentinel and task/ref residency identities, duplicated product identity maps, mutable output requests, direct runtime engine calls outside the executor adapter, and the source-text call-shape test are removed. | Complete |

The four undriven arm experiments—task death with live mapper views, live reset, interrupted queued
teardown, and display-versus-ordinary retirement—remain validation work, not guessed shared
behavior. They can refine the arm policy without reopening the cross-crate architecture.

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

The original static set established the PCI resource/page-table lifecycle and matching serializer
vocabulary in both instruction-set slices, but left the mapper kernel lifecycle open. A follow-up
arm64 investigation paired the mapper producer and consumer, perturbed serializer properties one
at a time, followed rollback paths, and observed the representative mapper form on a driven boot.
That evidence establishes descriptor composition, identity width, view/backing separation, and the
ordinary mapper view ownership relation. It does not establish every teardown race.

Keep five evidence strengths distinct during implementation:

1. Interface and symbol presence establishes that concepts or operations exist separately, not
   their ownership or ordering.
2. Paired producer and consumer analysis establishes a field, transition, or interpretation in
   the followed implementation.
3. A controlled serializer oracle establishes which source property selects or populates a wire
   form; correlation without a matching producer or consumer does not earn a semantic name.
4. A driven pathway boot establishes that the observed form and transition are exercised on that
   pathway, not that every accepted form is reachable.
5. A statically followed failure path establishes its intended rollback ordering; timing,
   interruption, and race behavior still require a driven fault experiment.

Two negative conclusions are architectural inputs too. First, the guest contract evidence does
not expose a unified-versus-discrete resource lifecycle; that classification is a host capability
and may only select executor placement and transfer plans. Second, `MapperSurfaceRef` is a 64-bit
mapper-service lookup identity, not a page-table `MappingId`, registered backing ID, or storage
owner. A mapper view retains its mapper-resolved surface/host representation; it does not own the
mapper mapping or backing storage.

The most dangerous disproven mapper assumption is stronger than a field-width correction:
`MapperSurfaceRef` cannot be collapsed into the x86 registered-backing model or into a page-table
mapping. The arm object uses it to resolve a surface for view construction; the resulting view
holds a surface/host-representation retain. The mapper registry entry, mapper memory, task object,
host view, IOSurface backing, and any GPU address-space mapping therefore remain separate graph
nodes with explicit edges. Numeric coincidence between any of their external names has no
semantic force.

The audit maps contract evidence to architecture as follows:

| Contract distinction | Architectural consequence | Confidence boundary |
|---|---|---|
| Task resource heap, resource-heap namespace, object handles, resources, memory maps, and page table are separate interfaces | Separate namespace, resource, storage, mapping, and address-space identities | Established on PCI and for ordinary mapper construction; teardown races remain open |
| Object-table entries expose a resource index while serializer object families expose their own create/get/delete-by-reference APIs | `ObjectTableRef` and `SerializerRef<T>` are different types and never key the same map | Established by static interface shape plus the existing driven-boot collision evidence; static names alone would not prove independence |
| Mapping commit allocates page-table coverage; mapping release synchronizes for unwire, retires child host resources, submits release work, then deallocates page-table coverage | Mapping release is an ordered core transition with backend release effects, not a cache deletion | Established for the PCI pathway only |
| Mapper memory has no ordinary GPU virtual address or page-table commit; its release queues synchronize/discard rather than applying the PCI deallocation sequence | Keep mapper paging and teardown behind the arm pathway policy | Normal transition established; interruption remains open |
| Resource backing replacement is an operation on an existing resource | Preserve resource identity and advance a backing generation/storage edge | Established; exact failure ordering is not |
| Segment resource lists independently initialize, update in-channel, prepare, and complete; queue processing parses them and writes invalidations | Preserve segment boundaries and the complete resource-participation envelope through execution | Established |
| The mapper texture object is a 64-bit mapper reference followed by one complete nested IOSurface-texture serializer operation | Decode `MapperIOSurfaceTextureView` as a typed envelope; reuse the nested operation decoder and keep mapper reference, backing, view, and display use as separate relations | All accepted descriptor variants established; private producer fields remain typed opaque values |
| A mapper view retains the resolved surface/host representation but does not own the mapper mapping or backing storage | Share the semantic IOSurface view relation, not construction, coherency, paging, discard, or teardown policy | Ordinary ownership and rollback established; task-exit, live-reset, interrupted-teardown, and display-retirement races remain open |
| Mapper textures have no-op prepare/synchronize, mapper buffers materialize lazily and discard only that materialization, while x86 reference textures delegate coherency to a retained registered backing | Put prepare, synchronize, and discard behind capability-returning pathway policy; never infer equivalence from matching method names | Established for ordinary operations and statically followed rollback; race and interruption outcomes remain open |
| Display begin, validate, submit, completion query/signal, framebuffer resource, and resource-heap operations are distinct | Display is a core transaction completed from presenter/executor facts | Established for the PCI display lifecycle |
| Object tables, mapper IOSurfaces, display mapper surfaces, heaps, buffer-from-IOSurface, discard, and synchronize/discard are independently advertised features | Guest protocol capabilities are a typed profile independent of host Vulkan capabilities | Established as independently named capabilities; availability combinations still require tests |

This table is deliberately not a wire-layout ledger. Exact offsets, opcode values, and binary
provenance stay out of the architecture plan; they belong at the decoding boundary and in local
investigation notes.

### Mapper contract now available to the refactor

The mapper envelope has exactly three accepted semantic variants: legacy narrow,
rotation-capable narrow, and rotation-capable wide. Each is an eight-byte mapper reference followed
by one complete nested texture serializer operation. Plane is a two-byte field in every form;
rotation is a separate two-byte field in the versioned forms. Verified unwritten bytes are not
semantic fields. Private producer flags and one forwarded descriptor member remain intentionally
opaque. An unfamiliar tag/length pair is not a fourth guessed form; it is a typed refusal.

The common abstraction justified by this evidence is deliberately small. It may carry the typed
descriptor, plane, optional rotation, canonical surface relation, view kind, host representation,
and typed effects such as `ReleaseView`, `ReleaseSurfaceRetain`,
`DiscardHostMaterialization`, `DeleteHostRepresentation`, and `RetireBacking`. It must not prescribe
one construction, retention, coherence, paging, discard, or teardown algorithm for both pathways.
Those remain arm-mapper and x86-registered-surface policies behind the common semantic view
contract.

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
  sequence: it has no ordinary page-table commit and queues synchronize/discard on release.
- Numeric object names stop at the decode boundary. Tag 11 is a mapper-backed IOSurface texture
  view envelope: a 64-bit `MapperSurfaceRef` followed by one complete nested serializer operation.
  The nested operation supplies the texture declaration, plane, and optional rotation. It is not
  an opaque mapper-specific tail and it is not a page-table mapping.
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
  that pathway and must not be generalized to mapper memory, whose ordinary page-table commit is a
  no-op and whose release queues synchronize/discard.
- Command buffers explicitly record resource reads/writes, page-off participation, state
  references, chunks, segments, splits, continuations, and merges.
- Textures have explicit buffer-backed, parent-texture view, IOSurface-plane, and heap-placement
  construction forms. Buffers and textures retain fields identifying their parent/storage
  relation rather than presenting every view as a fresh allocation.
- Mapper-capable contract code advertises mapper-backed surfaces and has distinct mapper-reference
  texture, mapper-reference buffer, and backing-resource classes. A mapper view retains its
  resolved surface/host representation, while mapper mapping and backing storage retain separate
  owners. Construction, coherency, discard, paging, and teardown are not equivalent to the
  registered-surface pathway.
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

**Not established by the present evidence:** arm task exit while mapper views remain live, reset
with live mapper mappings, interrupted queued synchronize/discard or host-resource deletion, and
display retirement relative to ordinary view teardown; exact failure ordering among unrelated
page/resource operations outside the established pathway sequences; or any host unified/discrete
placement rule. These are narrow lifecycle gates, not a reason to keep mapper descriptors or
ordinary view ownership opaque. Runtime topology equivalence remains a design invariant to test,
not a conclusion obtained from guest code.

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

The common serializer implementation is available for both supported architectures, and the arm
mapper envelope and ordinary view ownership have now been established. The nested descriptor is a
complete serializer operation rather than an opaque mapper tail. Do not generalize the x86
resource/backing lifecycle into the arm mapper: task-exit, live-reset, interrupted-teardown, and
display-retirement behavior remain pathway-specific questions.

## Baseline structural findings and remaining seams

The baseline dependency mismatch is now resolved; this section records the resulting boundaries and
the deliberately retained pathway-specific seams.

- Vulkan execution is concentrated in the executor adapter. Render translation, render executable
  publication, and render/compute asynchronous translation preflight now cross a device-owned
  `ShaderTranslationService`; pipeline retention sees only core prepared shader families. Compute
  reflection, native-module specialization, and executable publication now remain behind that
  service, while the format-selection rule is core-owned. Draw binding projection, buffer-bound
  planning, sampled/attachment format classification, and hang diagnostics now cross semantic
  executor ports. Production runtime calls now reach Vulkan only inside `VulkanExecutor`; drain and
  census obtain backend diagnostics through the observation service, and the shared sRGB downgrade
  latch is backend-neutral. Draw execution is split by ownership: resident target
  identity/currency/Store policy, sampled-source/content resolution, pipeline preflight, shader
  resource planning, attachment construction, LOAD selection, and executor-request assembly each
  have a total semantic planner. The remaining orchestration composes those results, executes one
  immutable prepared draw, and publishes completed Store effects; it imports no Vulkan vocabulary.
  Fixed-function render state crosses a single
  semantic normalization seam: primitive, raster, visibility, pass actions, blend, depth compare,
  and enabled stencil faces either produce typed state or refuse preparation. Semantic request
  construction now terminates in an immutable `PreparedDraw` carrying exactly one completion
  route; the executor consumes it, while Store and observation consume only its validated
  completion. Target identity, resident LOAD, `skip_readback`, and completion ownership now come
  from one target planner, and fixed-function normalization has its own module. Attachment
  construction is now a dedicated planner with a total result: expected absence is separate from
  typed action, source/resolve identity, resolve-compatibility, and pass-geometry refusals, and no
  partial attachment list can reach execution. Buffer/stage-in, shader relocation, sampled-texture,
  sampler, and final request construction now have complete planners. Sampler occupancy is shared
  across stream and reflected declarations; either kind of
  collision is a typed refusal rather than first-writer-wins execution. Unsupported reflected
  sampled shapes are no longer coerced to 2D, and an unrepresentable required neutral texture no
  longer leaves a descriptor hole after merely logging. Final executor-request assembly now also
  has one owner: `RequestPlanInputs` becomes an immutable core request plus its sole completion
  route. Pre-submit resource diagnostics consume that finished request from the observation module,
  and Store routing begins only after execution completion. Preparation and execution failures now
  meet only in a product-owned attempt envelope: core preparation remains typed, while the executor
  contributes an opaque but lossless diagnostic. `DrawError` is confined to the executor adapter.
  Pipeline/interface preflight is now its own total planner, pairing retained pipeline state,
  semantic blend state, and validated extent before bind planning begins. Shader numbering and
  directly-bound resource occupancy now have the matching total `ShaderResourcePlan`: buffer
  loading, fragment relocation, storage binding projection, statically-used descriptor-gap
  classification, neutral-texture obligation, and framebuffer-fetch admission are decided
  together. `DrawResourcePlan` composes that internal result with complete sampled-image and sampler
  planners, so relocation choices and neutral-substitution obligations no longer escape into draw
  orchestration. LOAD selection now terminates in one `LoadPlan`: guest/host seeds, resident
  currency, clear state, deferred-content capability, surface target, and GVA load identity are one
  snapshot consumed by final request assembly. Finally, `PreparedDraw` retains the semantic target
  resource only when its exact native request has a resident target, and publishes render-target
  use only after successful executor completion; orchestration no longer combines a completion
  boolean with a second lookup into the original request. Focused draw, execution-split, and
  prepared-draw tests and the subsequent full-workspace/dual-target checkpoints pass.
- The reset-only `Backend` abstraction, generic `Device<B>`, and the later `backend::vulkan`
  re-export façade have been deleted. `VulkanExecutor` is the concrete product adapter.
- `runtime::Device` now owns the executor/session and host address materializations. `DeviceState`
  remains the composition's broad semantic root, but submission lifecycle, translation scheduling,
  nested child-drain state, mapping validity ordering, sampled-content identity/revalidation,
  completion publication, mappings, host replicas, retirement effects, presentation, and
  observations now have distinct owners. `PendingWork`, backing evidence, present backpressure,
  capture policy, retained-frame state, mapping-role routing, and console/paint state now have
  transition APIs rather than public peer-field mutation. Completion
  publication was the first
  synchronization tranche removed from that field bag: one core-owned state machine now owns
  coalesced debt, handoff to the ordered publication rail, wait classification, and the progress
  witness used to retry held FIFO timelines. Drain paths no longer mutate an independent stamp
  ledger and sequence counter. Translation scheduling likewise owns immutable-translation
  deferral, sibling FIFO holds, present barriers, episode coalescing, channel retirement, and reset
  evidence now share one core owner. Runtime scheduling consumes typed hold/barrier results instead
  of keeping five masks and counters consistent by call ordering.
  Host alias lifetime is now another owned aggregate: registered task-GVA views and the private
  release queue move together. Publishing, exact lookup, overlap/task retirement, stale-view
  replacement, reset drainage, guest-import revocation, view unmapping, and native-resident release
  cross `HostMaterializationState`; removing a registered view queues its host release in the same
  transition, and the owner drains imports before aliased views. Runtime no longer mutates the view
  vector or release-effect list independently.
  Child-drain publication has also lost its peer active mask. `PendingWork` owns active and pending
  child sets with atomic activate-and-request and retire transitions; DEFINE, FREE, locked MMIO,
  lock-free doorbell folding, stranded rescue, and nested sibling draining consume those
  projections. A channel can no longer be freed from one set while remaining scheduled in the
  other, or be rung on the lock-free rail without becoming visible to stranded-FIFO rescue.
- `TaskResource` owns raw descriptor bytes, decoded construction semantics, guest-object lifetime,
  mapped-surface registration, and render-target history. Vulkan resident leases have moved to the
  device executor and are keyed by weak semantic resource lifetimes. Mutable resident read facts
  now cross separately as one atomic executor snapshot, so querying readiness/currency cannot
  acquire or release a lifetime lease.
- `SurfaceMappingEntry` has been split into lifecycle, declaration, page-plan, content, and
  materialization subobjects, and its registry now owns the `SurfaceId` namespace. Product builds
  expose no mutable registry lookup outside `DeviceState`; runtime page adoption, revalidation,
  validity, materialization, import replacement, and owner hints all cross named transitions.
- `ComputeStorageResidencyKey` now carries a typed `ComputeStorageOrigin` variant for mapped
  surfaces, task-GVA textures, and heap textures; its former sentinel-zero origin encoding is gone.
- The Vulkan engine's guest-derived pools, residents, imports, presenter, counters, completion
  signals, and publication state are device-session owned. Only the physical context and immutable
  content caches remain process-shared, with weak live-session registration for device loss.
- Topology classification and placement/batching consequences are centralized. Import availability
  is separately capability-driven. An end-to-end transfer audit found no second product topology
  classifier: direct import, GPU gather, CPU staging, scatter, and writeback routes are selected by
  measured import capability, decoded row/run shape, alignment, aliasing, and destination contract.
  Those are legitimate plan inputs and should not be forced into unified/discrete modules. The
  shared transfer-plan vocabulary now sits with bounded guest reads and writes: it separates
  complete direct/gather/CPU-only read visibility and semantic destination row shape from Vulkan
  realization. Guest-page transfer, completion ordering, and import lifetime also cross separate
  executor capabilities. Resident readiness, currency, and reclaim evidence now cross one atomic
  executor-owned read plan, while semantic lifetime retention remains a distinct capability. The
  Direct presentation preparation now also returns the exact checked semantic source from one
  executor transaction; the separately published attached shadow has been deleted. Open-batch
  transitions share one executor port, and native-window composition receives a presentation
  boundary error rather than a Vulkan execution error. The presentation request/executor contract
  now distinguishes unchecked `PresentationSource` intent from `PreparedPresentation`: only the
  executor's registry/window-policy transition produces the latter, and the device-to-window slot,
  host-window loop, executor presentation frame, and Vulkan presenter accept only that prepared
  value. Composition can no longer pass an arbitrary semantic target directly to the native
  presenter or mistake a request for evidence that the resident rail admitted it. Remaining work
  in this area is typed presentation completion/lifetime, not another topology branch and not
  moved lifecycle authority.
  The device-to-window frame is total as well: `FramePayload` and
  `WindowPresentationPayload` carry exactly one of prepared resident or CPU BGRA. The former
  `Vec<u8>` plus `Option<PresentationSource>` shape admitted both sources, neither source, and a CPU
  fallback whose geometry could disagree with the resident request. Device publication, the
  host-window slot, the loop-to-executor projection, and executor-to-Vulkan adapter now preserve
  the one selected payload without reconstructing it from optional peer fields.
  Successful presentation completion names that same choice as exhaustive
  `PresentationRoute::{Resident, CpuBgra}` rather than a `direct` boolean. Vulkan may retain a
  private boolean for its cadence arithmetic, but the executor boundary and host-window consumer
  cannot invert or silently extend the route meaning.
  Resident geometry is no longer duplicated outside the prepared value. CPU BGRA carries its own
  geometry in the CPU payload variant; a resident payload derives geometry from its checked
  `PresentationSource`. The outer frame owns only publication sequence and payload, so a resident
  cannot be queued with a second, disagreeing width/height pair.
  `PresentationSource` construction is atomic and its identity/geometry fields are private after
  construction. Preparation and native presentation read them through accessors; no intermediate
  layer can rewrite one term while retaining the other two.
- The former required `backend-vulkan` feature shaped hundreds of conditional compilation sites,
  preserving the source structure of a backend fork that no longer existed. Those source forks,
  the empty compatibility feature, integration-test gates, and QEMU build invocation have now been
  removed. Vulkan is the product executor; `host-window` is the only optional execution feature.
- Raw wire names `type11` and `type7` no longer persist after decode. Tag 7 has decoded semantic
  classes; tag 11 is a `MapperIOSurfaceTextureView` envelope containing a mapper lookup identity
  and a nested IOSurface-texture operation.
- Draw preparation now produces an immutable core `PreparedDraw`; Vulkan returns a separate typed
  completion and cannot mutate the preparation request to publish semantic output. Resource,
  storage, surface, view, content-version, and native-resident identities remain deliberately
  distinct, but they are related through the canonical graph and owner-keyed materialization
  registries rather than reconstructed by duplicated identity maps. Task definition/deletion,
  object-list replacement, and resource deletion now retire composition-owned bound-buffer
  materializations in the same typed transition as semantic state. The remaining lifetime work is
  therefore a transition-by-transition audit of specialized mapper/display teardown—not the
  absence of a resource aggregate, and not a second topology lifecycle.

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
- `MapperSurfaceRef` is a 64-bit identity in the mapper-service namespace. Resolving it creates an
  explicit relation to the canonical surface and a retain owned by the materialized host view; it
  is not an address-space `MappingId` or a storage owner.
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

The mapper pathway carries a typed IOSurface-texture view envelope with a 64-bit mapper reference;
the other pathway resolves a registered surface backing. The mapper reference is a lookup identity
for a mapper-managed surface relation. It is not a page-table `MappingId`, a registered
`SurfaceBackingId`, or an owner of IOSurface storage. The view retains the resolved surface/host
representation, while the mapper mapping and backing storage remain independently owned. The
actual types may differ, but every identity and lifetime distinction above must remain
representable and exhaustive.

### Semantic naming

Raw object tag 11 denotes a mapper-backed IOSurface texture-view envelope. Call the decoded object
`MapperIOSurfaceTextureView`, not `Type11`, a page-table mapping, or a registered surface. Its first
field is a 64-bit `MapperSurfaceRef`; the remainder is one complete nested IOSurface-texture
serializer operation. Decode that nested operation through the same semantic variants used at its
ordinary boundary: legacy narrow, rotation-capable narrow, and rotation-capable wide. Private
producer-populated fields remain typed opaque values, verified unwritten bytes are not fields, and
an unknown tag/length pair is a typed refusal rather than a guessed fourth variant.

Downstream code sees the semantic view and its explicit relation:

```text
MapperIOSurfaceTextureView {
    mapper_surface: MapperSurfaceRef,
    descriptor: { format, extent, usage, ... },
    plane: PlaneIndex,
    rotation: Option<Rotation>,
}
```

It does not see `type11`, and it does not call the texture itself a mapping. Resolution creates an
explicit edge from the view to the mapper-resolved surface/host representation; it does not turn
the mapper reference into a page-table `MappingId`, registered `SurfaceBackingId`, or storage
owner. Raw tag 7 immediately becomes `Sampler`, `DepthStencil`, `RenderPipeline`,
`ComputePipeline`, or `IndirectCommandBuffer` when its decoded discriminator establishes that
class; there is no semantic `Type7`.

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
- `MapperSurfaceRef` (64-bit wire identity)
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

- For mapper-backed views, what exact retirement behavior applies when (1) a task exits with
  multiple live views, (2) the device resets with live mapper objects, (3) queued
  synchronize/discard or child/parent host deletion is interrupted after submission, or (4) a
  display-bound view retires before or after an ordinary view of the same backing? Keep these
  teardown transitions in the arm pathway policy until each is driven and expressed as a typed
  transition. The required experiment must vary prepared/materialized state and both deletion
  orders, and assert that every surface retain, host representation, task object, and mapper
  registry entry retires exactly once without a callback through a dead task edge.
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

The former mapper descriptor-tail gate is closed. All accepted outer forms contain a 64-bit mapper
reference followed by a complete nested serializer operation. The shared semantic view relation is
also established: views and backing storage are separate, and a mapper view retains the resolved
surface/host representation without owning the mapper mapping or backing. This permits the common
view interface and resource-graph edges to migrate now. It does not permit shared construction,
coherency, discard, paging, or teardown policy.

Static analysis has reached its limit on the remaining mapper gate. It establishes intended normal
ordering and rollback branches, but cannot establish race outcomes involving task death, reset,
submitted asynchronous work, or display retirement. Those four cases require a driven arm64
experiment with controlled lifetime and fault boundaries. Until that is available, the correct
architecture is explicit pathway policy plus typed unresolved transitions—not an inferred common
lifecycle and not a blocker on the already-established descriptor and ownership migration.

The mapper migration therefore carries this focused verification set:

- semantic fixtures for all three nested variants, including nonzero high mapper-reference bits,
  separate plane/rotation values, poisoned unwritten bytes, and typed refusal of inconsistent or
  unfamiliar tag/length pairs;
- rollback tests at surface lookup, descriptor validation, host texture/buffer creation, wrapper
  publication, and task-heap insertion, proving that no surface retain or partial host/object edge
  leaks;
- two-view deletion tests over one backing, in both orders, proving that each host representation
  releases once while mapping and backing survive their independent owners;
- mapper-buffer materialize/discard/rematerialize tests proving that discard removes only the
  derived host buffer and its surface retain;
- pathway-dispatch tests proving that mapper texture prepare/synchronize has no transfer effect,
  x86 reference texture delegates to registered backing, and mapper paging never acquires the PCI
  page-table transition;
- nested prepare/complete and child-before-parent deletion tests, including first-wire rollback
  and missing-child failure;
- driven arm64 tests for the four unresolved scenarios above. Failure to construct the public
  graphics device in a controlled guest is an experiment-environment limitation, not permission
  to infer the missing teardown behavior.

## Migration sequence

Each phase must have focused tests and keep the affected pathways buildable. Behavior-preserving
phases should not make broader correctness claims. A behavior change must have a failing regression
test or measured proxy before it lands.

### 1. Establish semantic vocabulary

Create `reims-vgpu-protocol`.

Move object and command decoding behind semantic types. Introduce separate typed object-table and
serializer namespaces, resource, storage, surface-backing and mapping IDs plus addresses, lengths,
formats, actions, and selectors. Replace `type11` with `MapperIOSurfaceTextureView` at the decoder
boundary and eliminate `type7` from decoded values. Model the outer object as
`MapperSurfaceRef(u64)` plus one complete nested IOSurface-texture operation. Reuse the nested
semantic decoder for its three accepted variants, preserve private producer fields as opaque typed
values, and refuse unknown tag/length combinations. Carry the mapper reference as its own typed
relation; do not assign it a page-table mapping identity, registered-surface identity, or owned
IOSurface-plane storage edge. Preserve mapper and registered-surface construction variants because
their coherency and teardown policies are demonstrably different.

Two correctness repairs are prerequisites to migrating mapper behavior:

1. widen `MapperSurfaceRef` and the tag-11 decoder from 32 to 64 bits, with a fixture whose high
   bits are nonzero;
2. model the rotation-capable narrow and wide tails as a two-byte plane plus two-byte rotation,
   rather than widening the wide form's complete slot into a four-byte plane. Verify legacy
   unwritten bytes without interpreting them.

Do not change execution behavior in this phase.

Exit criteria:

- protocol fixtures produce the new semantic values;
- object-table refs and typed serializer refs cannot be interchanged;
- mapper fixtures retain nonzero high reference bits and decode plane/rotation independently;
- every accepted nested mapper variant uses the shared semantic operation decoder;
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
6. mapper-backed surface textures, after the 64-bit reference and plane/rotation wire repairs land.

The mapper family may now receive its generic generational identity, semantic view relation, and
explicit edge to the mapper-resolved surface/host representation. Its construction, coherency,
discard, paging, and teardown behavior remain behind an arm-specific policy. Task-exit, live-reset,
interrupted-teardown, and display-retirement transitions do not become authoritative core behavior
until their remaining arm experiments establish them. This keeps a narrow evidence gap from
becoming a guessed cross-pathway lifecycle merely because the view interface is shared.

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
