# Reims vGPU architecture

Reims has one semantic device model and one Vulkan executor across both supported pathways. The
PCI/MMIO attach mechanism, guest page geometry, mapper availability, host-pointer import, and memory
topology differ; guest-visible resource identity, content authority, synchronization, and command
meaning do not.

## The seam

```text
guest bytes
  -> checked wire view
  -> semantic command and typed guest identities
  -> task namespace and generational resource graph
  -> immutable resolved submission
  -> Vulkan placement/transfer plan and execution
  -> typed completion fact
  -> semantic lifecycle/content/display transition
  -> guest-visible completion
```

No layer may skip across this chain to reconstruct an answer owned by another layer. In particular,
a memory optimization does not define lifetime or coherency, and a QEMU shim does not combine
semantic queries into a product rule.

## Ownership by crate

| Crate | Owns | Must not own |
|---|---|---|
| `reims-vgpu-wire` | Borrowed, checked wire views and record framing | Semantic defaults, allocation, lifecycle, or execution |
| `reims-vgpu-protocol` | Decoded enums/descriptors, typed guest identities, protocol refusals | Device state, Vulkan types, or host policy |
| `reims-vgpu-paging` | Page-table interpretation, span walks, GPA-run and window planning | Host mappings or resource lifetime |
| `reims-vgpu-memory` | Bounded guest-memory runs, slices, destinations, and transfer-plan vocabulary | Vulkan allocation policy or content authority |
| `reims-vgpu-core` | Task namespaces, generational resource graph, lifecycle/content authority, immutable command envelopes, executor ports, synchronization, and presentation semantics | QEMU, Vulkan handles/formats, environment policy, or native shader payloads |
| `reims-vgpu-vulkan` | Capability discovery, structural topology classification, placement/batching policy, translated/native shaders, GPU objects, submissions, residents, and per-device sessions | Guest object lifetime or an alternative content model |
| `reims-vgpu` | Decode orchestration, composition-owned adapters, device scheduling, QEMU ABI, and failure projection | A second semantic vocabulary or direct engine path beside the executor |
| `reims-vgpu-observe` | Shared typed emission and measurement support | Behavior-selecting state |
| `reims-vgpu-config` | Operator configuration names and parsing | Host capability or guest semantics |
| `vendor/qemu` | QOM attach, PCI/MMIO, IRQ, console/input, and host-memory plumbing | Protocol, resource, topology, or presentation policy |

The shipping artifact remains the `reims-vgpu` static library linked into QEMU. The crate is a
composition root, not the owner of every concern it links.

## Identity and lifetime

A numeric value can occur in several independent namespaces. These are never interchangeable:

- task-local `ObjectTableRef` and serializer references;
- generational `ResourceId`;
- storage/backing and view identities;
- `SurfaceId` and page-table `MappingId`;
- 64-bit `MapperSurfaceRef` and mapper-resolved surface identity;
- content, backing, mapping, and resident generations.

Task/object references are reusable wire names. A durable execution, residency, Store/gather
witness, or content-authority entry must resolve them to the canonical `ResourceId` first. Raw
names remain legitimate at the byte decoder, task namespace, pre-construction currency, and a
documented compatibility adapter which immediately projects them from or into the graph. They are
not executor identity.

Deleting a view does not imply deleting its storage. Replacing physical backing advances backing
state without manufacturing a new object lifetime. Task deletion, object deletion, mapping release,
backing retirement, host-materialization release, and display retirement are distinct typed
effects, even when one guest packet triggers several of them.

Mapper-backed arm resources share the semantic view/backing distinction with registered x86
resources, but not their construction, coherency, paging, discard, or teardown implementation.
Task death with live mapper views, live reset, interrupted queued teardown, and display-versus-
ordinary retirement remain arm-specific validation questions; shared code must not guess them.

## Commands, execution, and completion

Decoded operations are normalized into immutable core commands. A resolved command contains
generational resources, typed surfaces/mappings, semantic descriptors, byte windows, and declared
access—not raw object tags, unresolved task-local references, Vulkan handles, or SPIR-V/native
payloads.

`ResolvedSubmission` preserves command-buffer segmentation and resource participation. The
executor returns typed completion values; it does not mutate request fields to publish success.
Only a successful completion fact may advance content versions, Store authority, synchronization,
or presentation state.

The product `Executor` composes narrow capability, translation, residency, transfer, execution,
maintenance, presentation, and session services. `VulkanExecutor` is its implementation. Legacy
host APIs may require a task-local reference; recover it from the resource graph at that final
adapter and fail by type if the resource has retired.

## Content authority

The canonical content model records which version exists in guest pages, host replicas, and GPU
residents. Addresses, page-set hashes, cache hits, upload counters, and native allocation IDs are
evidence used to execute a plan; none substitutes for `(ResourceId, ContentVersion)`.

Guest writes, GPU Stores, synchronization, discard, readback, and delayed completion are explicit
transitions. A delayed completion cannot overwrite a newer guest write. Synchronization pays only
the named resource/subresource obligation. State whose eviction would lose the only copy is tied to
the guest lifetime and has no invented capacity; bounded caches may retain only recomputable data.

## Topology and sessions

`reims-vgpu-vulkan::memory` classifies unified versus discrete memory from structural heap/type
properties. `reims-vgpu-vulkan::policy` may select memory requests and batching defaults. Host-
pointer import is an orthogonal measured capability. All four combinations must preserve the same
semantic trace:

| | Import available | No import |
|---|---|---|
| Unified | Directly bind eligible guest memory | Copy through unified-host staging |
| Discrete | Import as backing and copy GPU-side into working memory | Stage every crossing |

Topology may change placement, transfer scheduling, and metrics. It must not change resource
lifetime, correctness, refusal meaning, or guest-visible output. Vendor and driver names are not
policy inputs.

Guest-derived GPU state is owned by a device/session handle: pools, residents, imports, submissions,
presenter, counters, and completion signals. Only the physical context and immutable content-keyed
caches may be shared. Reset or deletion of one vGPU must not invalidate another.

## Regression gates

Architecture changes need behavioral tests at the owner boundary. Preserve at least these classes:

- deleting and recreating one object-table slot cannot inherit debt, witnesses, replicas, or
  residency from the retired `ResourceId`;
- two devices isolate reset, deletion, leases, counters, presentation, and guest-write debt;
- unified/discrete × import/no-import produces equivalent content and guest effects;
- resolved commands contain semantic endpoints and separate completion outputs;
- mapper fixtures preserve high reference bits and independent plane/rotation fields;
- PCI and MMIO adapters consume the same semantic presentation result.

Run the serial workspace suite and the feature matrix for cross-target changes:

```sh
cargo test --workspace --all-targets --features host-window -- --test-threads=1
scripts/feature-matrix/feature-matrix.sh
cargo clippy --workspace --all-targets --features host-window -- -D warnings
cargo clippy -p reims-vgpu --target aarch64-apple-darwin \
  --all-targets --features host-window -- -D warnings
```

Use types, constructors, dependency direction, and behavioral fixtures to preserve the seam. Do not
add tests that parse repository source to police architecture by spelling.
