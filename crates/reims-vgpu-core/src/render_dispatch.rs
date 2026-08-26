//! Prepared backend-neutral render draws.
//!
//! The operation carries decoded pass, draw, and binding terms plus canonical
//! generational resources. Guest pointers, materialized bytes, native views,
//! descriptor handles, and Vulkan pipeline state cannot cross this seam.

use crate::{
    AccessIntent, AccessMode, AccessScope, AccessTarget, BackingRegion, GpuWriteBatchError,
    GpuWriteId, GpuWriteRequest, GpuWriteReservation, ImageSubresourceRange, ManagedBackingError,
    ManagedBackingProgress, ReadyPipelineLease, RepresentationUse, ResolvedResourceCompletion,
    ResourceLifecycleOwner, ResourceUseBatchError, StageScope,
};
use reims_vgpu_protocol::{
    BackingId, CullMode, DepthClipMode, DepthStencilObject, FillMode, HazardDomainId, IndexType,
    LoadAction, PrimitiveTopology, RenderPipelineObject, RenderStages, ResourceId, ResourceObject,
    StoreAction, SubmissionId, TransactionId, VisibilityResultMode,
};
use std::collections::{BTreeMap, BTreeSet};

/// Byte width of the four-word indirect primitive argument structure shared
/// by Metal and Vulkan.
pub const RENDER_INDIRECT_ARGUMENT_BYTES: u64 = std::mem::size_of::<[u32; 4]>() as u64;
/// Byte width of the five-word indexed indirect argument structure shared by
/// Metal and Vulkan.
pub const RENDER_INDEXED_INDIRECT_ARGUMENT_BYTES: u64 = std::mem::size_of::<[u32; 5]>() as u64;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RenderBindingClass {
    VertexBuffer,
    /// The buffer argument of an indexed draw. It has no shader descriptor
    /// binding and therefore cannot share the vertex descriptor namespace.
    IndexBuffer,
    /// The argument structure consumed by an indirect draw. Like an index
    /// buffer this is fixed-function input, not a shader descriptor.
    IndirectBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderBindingView {
    Buffer(crate::LinearRange),
    Image(crate::ResolvedTextureBindingView),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRenderResourceBinding {
    pub class: RenderBindingClass,
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub stages: RenderStages,
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub view: RenderBindingView,
    pub regions: Box<[BackingRegion]>,
    pub mode: AccessMode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRenderSamplerBinding {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub stages: RenderStages,
    /// Complete decoded sampler value. This covers guest sampler objects with
    /// per-bind LOD overrides, shader-declared constexpr samplers, and the
    /// explicit null descriptor without sending an object-table name or a
    /// backend handle across the semantic seam.
    pub sampler: crate::SamplerResource,
}

/// One explicitly null image descriptor required by the translated interface.
/// It has descriptor shape and stage visibility but deliberately no resource,
/// backing, region, or access intent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedRenderNullBinding {
    pub class: RenderBindingClass,
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub stages: RenderStages,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderViewport {
    pub origin_x_bits: u64,
    pub origin_y_bits: u64,
    pub width_bits: u64,
    pub height_bits: u64,
    pub near_bits: u64,
    pub far_bits: u64,
}

impl RenderViewport {
    pub fn from_values(values: [f64; 6]) -> Self {
        Self {
            origin_x_bits: values[0].to_bits(),
            origin_y_bits: values[1].to_bits(),
            width_bits: values[2].to_bits(),
            height_bits: values[3].to_bits(),
            near_bits: values[4].to_bits(),
            far_bits: values[5].to_bits(),
        }
    }

    pub fn values(self) -> [f64; 6] {
        [
            f64::from_bits(self.origin_x_bits),
            f64::from_bits(self.origin_y_bits),
            f64::from_bits(self.width_bits),
            f64::from_bits(self.height_bits),
            f64::from_bits(self.near_bits),
            f64::from_bits(self.far_bits),
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderScissor {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRenderRasterState {
    pub viewports: Box<[RenderViewport]>,
    pub scissors: Box<[RenderScissor]>,
    pub cull_mode: CullMode,
    pub front_face_ccw: bool,
    pub fill_mode: FillMode,
    pub line_width_bits: u32,
    pub depth_clip_mode: DepthClipMode,
    pub depth_bias_bits: Option<[u32; 3]>,
    pub blend_color_bits: Option<[u32; 4]>,
    pub stencil_reference: [u32; 2],
}

impl Default for ResolvedRenderRasterState {
    fn default() -> Self {
        Self {
            viewports: Box::new([]),
            scissors: Box::new([]),
            cull_mode: CullMode::None,
            front_face_ccw: false,
            fill_mode: FillMode::Fill,
            line_width_bits: 1.0f32.to_bits(),
            depth_clip_mode: DepthClipMode::Clip,
            depth_bias_bits: None,
            blend_color_bits: None,
            stencil_reference: [0, 0],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RenderAttachmentRole {
    Color(u32),
    Depth,
    Stencil,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderAttachmentClear {
    Color([u32; 4]),
    Depth(u32),
    Stencil(u32),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRenderAttachment {
    pub role: RenderAttachmentRole,
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub regions: Box<[BackingRegion]>,
    pub pixel_format: u16,
    pub extent: [u32; 3],
    pub sample_count: u32,
    pub load: LoadAction,
    pub store: StoreAction,
    pub clear: RenderAttachmentClear,
    pub resolve: Option<ResolvedRenderResolveAttachment>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRenderResolveAttachment {
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub regions: Box<[BackingRegion]>,
    pub pixel_format: u16,
    pub extent: [u32; 3],
    pub sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedRenderVisibility {
    pub mode: VisibilityResultMode,
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub range: crate::LinearRange,
}

/// Effective stride of one vertex-buffer binding for this draw.
///
/// The immutable pipeline descriptor supplies the default. An encoder bind may
/// override it per draw, so native variant identity must consume this resolved
/// value rather than re-reading either mutable encoder state or only the
/// pipeline default.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ResolvedVertexBufferLayout {
    pub binding: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedRenderDraw {
    Direct {
        topology: PrimitiveTopology,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Indexed {
        topology: PrimitiveTopology,
        index_type: IndexType,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    Indirect {
        topology: PrimitiveTopology,
    },
    IndexedIndirect {
        topology: PrimitiveTopology,
        index_type: IndexType,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRenderDispatch {
    pub pipeline: ResourceId<RenderPipelineObject>,
    /// Exact translated stage variants selected for this draw. Constexpr
    /// sampler specialization can change these within one pipeline lifetime.
    pub program: crate::PreparedRenderProgram,
    pub depth_stencil: Option<ResourceId<DepthStencilObject>>,
    pub render_extent: [u32; 2],
    pub raster: ResolvedRenderRasterState,
    pub visibility: Option<ResolvedRenderVisibility>,
    pub begins_encoder: bool,
    pub ends_encoder: bool,
    pub draw: ResolvedRenderDraw,
    pub vertex_buffers: Box<[ResolvedVertexBufferLayout]>,
    pub attachments: Box<[ResolvedRenderAttachment]>,
    pub resources: Box<[ResolvedRenderResourceBinding]>,
    pub null_bindings: Box<[ResolvedRenderNullBinding]>,
    pub samplers: Box<[ResolvedRenderSamplerBinding]>,
}

impl ResolvedRenderDispatch {
    /// Exact canonical content a render command may consume before producing
    /// any of its writes.  Representation preparation uses the same decoded
    /// access modes and attachment load actions as render reservation.
    pub fn content_synchronization_requests(&self) -> Box<[crate::ContentSynchronizationRequest]> {
        let mut grouped = BTreeMap::<BackingId, BTreeSet<BackingRegion>>::new();
        for resource in &self.resources {
            if matches!(
                resource.mode,
                AccessMode::Read | AccessMode::ReadWrite | AccessMode::Unknown
            ) {
                grouped
                    .entry(resource.backing)
                    .or_default()
                    .extend(resource.regions.iter().copied());
            }
        }
        if self.begins_encoder {
            for attachment in &self.attachments {
                if attachment.load == LoadAction::Load {
                    grouped
                        .entry(attachment.backing)
                        .or_default()
                        .extend(attachment.regions.iter().copied());
                }
            }
        }
        grouped
            .into_iter()
            .map(|(backing, regions)| crate::ContentSynchronizationRequest {
                backing,
                regions: regions.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn accesses(&self, hazard_domain: HazardDomainId) -> Box<[AccessIntent]> {
        let resources = self.resources.iter().flat_map(|resource| {
            resource
                .regions
                .iter()
                .copied()
                .map(move |region| AccessIntent {
                    hazard_domain,
                    target: Some(AccessTarget::Backing(resource.backing)),
                    resource: Some(resource.resource),
                    scope: access_scope(region),
                    mode: resource.mode,
                    stages: match resource.class {
                        RenderBindingClass::IndirectBuffer => StageScope::Indirect,
                        RenderBindingClass::VertexBuffer => StageScope::VertexInput,
                        RenderBindingClass::IndexBuffer => StageScope::IndexInput,
                        RenderBindingClass::StorageBuffer
                        | RenderBindingClass::SampledImage
                        | RenderBindingClass::StorageImage => StageScope::Render(resource.stages),
                    },
                })
        });
        let attachments = self.attachments.iter().flat_map(|attachment| {
            attachment
                .regions
                .iter()
                .copied()
                .map(move |region| AccessIntent {
                    hazard_domain,
                    target: Some(AccessTarget::Backing(attachment.backing)),
                    resource: Some(attachment.resource),
                    scope: access_scope(region),
                    mode: if attachment.load == LoadAction::Load {
                        AccessMode::ReadWrite
                    } else {
                        AccessMode::Write
                    },
                    stages: match attachment.role {
                        RenderAttachmentRole::Color(_) => StageScope::ColorAttachment,
                        RenderAttachmentRole::Depth | RenderAttachmentRole::Stencil => {
                            StageScope::DepthStencilAttachment
                        }
                    },
                })
        });
        let resolves = self
            .attachments
            .iter()
            .filter_map(|attachment| {
                attachment.resolve.as_ref().map(|resolve| {
                    resolve
                        .regions
                        .iter()
                        .copied()
                        .map(move |region| AccessIntent {
                            hazard_domain,
                            target: Some(AccessTarget::Backing(resolve.backing)),
                            resource: Some(resolve.resource),
                            scope: access_scope(region),
                            mode: AccessMode::Write,
                            stages: match attachment.role {
                                RenderAttachmentRole::Color(_) => StageScope::ColorAttachment,
                                RenderAttachmentRole::Depth | RenderAttachmentRole::Stencil => {
                                    StageScope::DepthStencilAttachment
                                }
                            },
                        })
                })
            })
            .flatten();
        let visibility = self.visibility.into_iter().map(|visibility| AccessIntent {
            hazard_domain,
            target: Some(AccessTarget::Backing(visibility.backing)),
            resource: Some(visibility.resource),
            scope: AccessScope::Linear(visibility.range),
            mode: AccessMode::Write,
            stages: StageScope::QueryResolve,
        });
        resources
            .chain(attachments)
            .chain(resolves)
            .chain(visibility)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

fn access_scope(region: BackingRegion) -> AccessScope {
    match region {
        BackingRegion::Whole => AccessScope::WholeBacking,
        BackingRegion::Linear(range) => AccessScope::Linear(range),
        BackingRegion::Image(region) => AccessScope::Image(
            ImageSubresourceRange::new(
                region.aspect,
                region.mip,
                1,
                region.layer,
                1,
                Some(region.texels),
            )
            .expect("one exact image mip and layer are nonempty"),
        ),
    }
}

#[derive(Debug)]
pub struct PreparedRenderDispatch<NativePipeline, Operation = ResolvedRenderDispatch> {
    transaction: TransactionId,
    operation_index: usize,
    operation: Operation,
    pipeline: ReadyPipelineLease<RenderPipelineObject, NativePipeline>,
    uses: Box<[RepresentationUse]>,
    writes: Box<[GpuWriteReservation]>,
    completions: Box<[ResolvedResourceCompletion]>,
}

impl<NativePipeline, Operation> PreparedRenderDispatch<NativePipeline, Operation> {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }
    pub const fn operation_index(&self) -> usize {
        self.operation_index
    }
    pub const fn operation(&self) -> &Operation {
        &self.operation
    }
    pub const fn pipeline(&self) -> &ReadyPipelineLease<RenderPipelineObject, NativePipeline> {
        &self.pipeline
    }
    pub const fn uses(&self) -> &[RepresentationUse] {
        &self.uses
    }
    pub const fn writes(&self) -> &[GpuWriteReservation] {
        &self.writes
    }
    pub const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }
    pub fn into_operation(self) -> Operation {
        self.operation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RenderDispatchPreparationError {
    PipelineMismatch,
    EmptyRenderExtent,
    EmptyAttachmentExtent(RenderAttachmentRole),
    EmptyAttachmentSamples(RenderAttachmentRole),
    DuplicateAttachment(RenderAttachmentRole),
    AttachmentClearMismatch(RenderAttachmentRole),
    ResolveAttachmentMissing(RenderAttachmentRole),
    UnexpectedResolveAttachment(RenderAttachmentRole),
    ResolveAttachmentMismatch(RenderAttachmentRole),
    InvalidVisibilityRange,
    EmptyRegions {
        binding: u32,
        array_element: u32,
    },
    DuplicateBinding {
        class: RenderBindingClass,
        binding: u32,
        array_element: u32,
    },
    DuplicateSamplerBinding {
        binding: u32,
        array_element: u32,
    },
    BindingViewMismatch {
        class: RenderBindingClass,
        binding: u32,
        array_element: u32,
    },
    EmptyDescriptorArray {
        binding: u32,
    },
    ArrayElementPastDescriptorCount {
        binding: u32,
        array_element: u32,
        descriptor_count: u32,
    },
    DescriptorCountMismatch {
        binding: u32,
    },
    DescriptorClassCollision {
        binding: u32,
    },
    EmptyVertexStride(u32),
    DuplicateVertexLayout(u32),
    MissingVertexBinding(u32),
    MissingVertexLayout(u32),
    InvalidVertexBinding(u32),
    MissingIndexBinding(u32),
    InvalidIndexBinding(u32),
    MissingIndirectBinding(u32),
    InvalidIndirectBinding(u32),
    Backing {
        backing: BackingId,
        resources: Box<[ResourceId<ResourceObject>]>,
        reads: bool,
        writes: bool,
        reason: ManagedBackingError,
    },
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
}

pub fn prepare_render_dispatch<T, NativePipeline>(
    owner: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    submission: SubmissionId,
    operation_index: usize,
    operation: ResolvedRenderDispatch,
    pipeline: ReadyPipelineLease<RenderPipelineObject, NativePipeline>,
) -> Result<PreparedRenderDispatch<NativePipeline>, RenderDispatchPreparationError> {
    if pipeline.pipeline != operation.pipeline {
        return Err(RenderDispatchPreparationError::PipelineMismatch);
    }
    if operation.render_extent.contains(&0) {
        return Err(RenderDispatchPreparationError::EmptyRenderExtent);
    }
    validate_shape(&operation)?;

    let mut grouped = BTreeMap::<
        BackingId,
        (
            BTreeSet<BackingRegion>,
            bool,
            bool,
            BTreeSet<ResourceId<ResourceObject>>,
        ),
    >::new();
    for resource in &operation.resources {
        let group = grouped.entry(resource.backing).or_default();
        group.0.extend(resource.regions.iter().copied());
        group.1 |= matches!(
            resource.mode,
            AccessMode::Read | AccessMode::ReadWrite | AccessMode::Unknown
        );
        group.2 |= matches!(
            resource.mode,
            AccessMode::Write | AccessMode::ReadWrite | AccessMode::Unknown
        );
        group.3.insert(resource.resource);
    }
    for attachment in &operation.attachments {
        let group = grouped.entry(attachment.backing).or_default();
        group.0.extend(attachment.regions.iter().copied());
        group.1 |= operation.begins_encoder && attachment.load == LoadAction::Load;
        group.2 |= operation.ends_encoder
            && matches!(
                attachment.store,
                StoreAction::Store | StoreAction::StoreAndMultisampleResolve
            );
        group.3.insert(attachment.resource);
        if let Some(resolve) = &attachment.resolve {
            let group = grouped.entry(resolve.backing).or_default();
            group.0.extend(resolve.regions.iter().copied());
            group.2 |= operation.ends_encoder;
            group.3.insert(resolve.resource);
        }
    }
    if let Some(visibility) = operation.visibility {
        let group = grouped.entry(visibility.backing).or_default();
        group.0.insert(BackingRegion::Linear(visibility.range));
        group.2 = true;
        group.3.insert(visibility.resource);
    }

    let mut uses = Vec::with_capacity(grouped.len());
    let mut write_requests = Vec::new();
    for (backing, (regions, reads, writes, resources)) in grouped {
        let regions = regions.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let resources = resources.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let backing_error = |reason| RenderDispatchPreparationError::Backing {
            backing,
            resources: resources.clone(),
            reads,
            writes,
            reason,
        };
        let representation = owner
            .execution_representation_id(backing)
            .map_err(&backing_error)?;
        if reads {
            let snapshot = owner
                .snapshot_content(backing, &regions)
                .map_err(&backing_error)?;
            owner
                .execution_representation_for_snapshot(backing, &snapshot)
                .map_err(backing_error)?;
        }
        uses.push(RepresentationUse {
            backing,
            representations: Box::new([representation]),
        });
        if writes {
            write_requests.push(GpuWriteRequest {
                backing,
                representation,
                regions,
            });
        }
    }

    let write = GpuWriteId::operation(submission, operation_index);
    owner
        .validate_plan_gpu_writes(write, &write_requests)
        .map_err(RenderDispatchPreparationError::Writes)?;
    owner
        .validate_accept_uses(transaction, &uses)
        .map_err(RenderDispatchPreparationError::Uses)?;
    let writes = owner
        .plan_gpu_writes(write, write_requests.into_boxed_slice())
        .expect("render writes were prevalidated");
    owner
        .accept_uses(transaction, &uses)
        .expect("render uses were prevalidated");
    let mut completions = writes
        .iter()
        .map(|write| ResolvedResourceCompletion::GpuWrite {
            backing: write.backing,
            write: write.write,
            representation: write.representation,
        })
        .collect::<Vec<_>>();
    completions.extend(
        operation
            .attachments
            .iter()
            .filter(|attachment| {
                operation.ends_encoder
                    && matches!(
                        attachment.store,
                        StoreAction::DontCare | StoreAction::MultisampleResolve
                    )
            })
            .flat_map(|attachment| {
                attachment.regions.iter().copied().map(|region| {
                    ResolvedResourceCompletion::Discard {
                        backing: attachment.backing,
                        region,
                    }
                })
            }),
    );
    completions.sort_unstable();
    completions.dedup();
    Ok(PreparedRenderDispatch {
        transaction,
        operation_index,
        operation,
        pipeline,
        uses: uses.into_boxed_slice(),
        writes,
        completions: completions.into_boxed_slice(),
    })
}

fn validate_shape(
    operation: &ResolvedRenderDispatch,
) -> Result<(), RenderDispatchPreparationError> {
    if operation.visibility.is_some_and(|visibility| {
        visibility.range.start() % 8 != 0 || visibility.range.end() - visibility.range.start() != 8
    }) {
        return Err(RenderDispatchPreparationError::InvalidVisibilityRange);
    }
    let mut attachments = BTreeSet::new();
    for attachment in &operation.attachments {
        if !attachments.insert(attachment.role) {
            return Err(RenderDispatchPreparationError::DuplicateAttachment(
                attachment.role,
            ));
        }
        if attachment.extent.contains(&0) {
            return Err(RenderDispatchPreparationError::EmptyAttachmentExtent(
                attachment.role,
            ));
        }
        if attachment.sample_count == 0 {
            return Err(RenderDispatchPreparationError::EmptyAttachmentSamples(
                attachment.role,
            ));
        }
        let needs_resolve = matches!(
            attachment.store,
            StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
        );
        match (needs_resolve, attachment.resolve.as_ref()) {
            (true, None) => {
                return Err(RenderDispatchPreparationError::ResolveAttachmentMissing(
                    attachment.role,
                ));
            }
            (false, Some(_)) => {
                return Err(RenderDispatchPreparationError::UnexpectedResolveAttachment(
                    attachment.role,
                ));
            }
            (true, Some(resolve))
                if attachment.sample_count <= 1
                    || resolve.sample_count != 1
                    || resolve.pixel_format != attachment.pixel_format
                    || resolve.extent != attachment.extent
                    || resolve.regions.is_empty() =>
            {
                return Err(RenderDispatchPreparationError::ResolveAttachmentMismatch(
                    attachment.role,
                ));
            }
            _ => {}
        }
        let clear_matches = matches!(
            (attachment.role, attachment.clear),
            (
                RenderAttachmentRole::Color(_),
                RenderAttachmentClear::Color(_)
            ) | (RenderAttachmentRole::Depth, RenderAttachmentClear::Depth(_))
                | (
                    RenderAttachmentRole::Stencil,
                    RenderAttachmentClear::Stencil(_)
                )
        );
        if !clear_matches {
            return Err(RenderDispatchPreparationError::AttachmentClearMismatch(
                attachment.role,
            ));
        }
    }

    let mut slots = BTreeSet::new();
    let mut declarations = BTreeMap::<u32, (Option<RenderBindingClass>, u32)>::new();
    for resource in &operation.resources {
        if resource.regions.is_empty() {
            return Err(RenderDispatchPreparationError::EmptyRegions {
                binding: resource.binding,
                array_element: resource.array_element,
            });
        }
        if resource.descriptor_count == 0 {
            return Err(RenderDispatchPreparationError::EmptyDescriptorArray {
                binding: resource.binding,
            });
        }
        if resource.array_element >= resource.descriptor_count {
            return Err(
                RenderDispatchPreparationError::ArrayElementPastDescriptorCount {
                    binding: resource.binding,
                    array_element: resource.array_element,
                    descriptor_count: resource.descriptor_count,
                },
            );
        }
        if !slots.insert((resource.class, resource.binding, resource.array_element)) {
            return Err(RenderDispatchPreparationError::DuplicateBinding {
                class: resource.class,
                binding: resource.binding,
                array_element: resource.array_element,
            });
        }
        if !matches!(
            resource.class,
            RenderBindingClass::VertexBuffer
                | RenderBindingClass::IndexBuffer
                | RenderBindingClass::IndirectBuffer
        ) {
            if let Some(found) = declarations.get(&resource.binding) {
                if found.0 != Some(resource.class) {
                    return Err(RenderDispatchPreparationError::DescriptorClassCollision {
                        binding: resource.binding,
                    });
                }
                if found.1 != resource.descriptor_count {
                    return Err(RenderDispatchPreparationError::DescriptorCountMismatch {
                        binding: resource.binding,
                    });
                }
            } else {
                declarations.insert(
                    resource.binding,
                    (Some(resource.class), resource.descriptor_count),
                );
            }
        }
        if !matches!(
            (resource.class, resource.view),
            (
                RenderBindingClass::VertexBuffer
                    | RenderBindingClass::IndexBuffer
                    | RenderBindingClass::IndirectBuffer
                    | RenderBindingClass::StorageBuffer,
                RenderBindingView::Buffer(_)
            ) | (
                RenderBindingClass::SampledImage | RenderBindingClass::StorageImage,
                RenderBindingView::Image(_)
            )
        ) {
            return Err(RenderDispatchPreparationError::BindingViewMismatch {
                class: resource.class,
                binding: resource.binding,
                array_element: resource.array_element,
            });
        }
    }
    let mut sampler_slots = BTreeSet::new();
    for sampler in &operation.samplers {
        if sampler.descriptor_count == 0 {
            return Err(RenderDispatchPreparationError::EmptyDescriptorArray {
                binding: sampler.binding,
            });
        }
        if sampler.array_element >= sampler.descriptor_count {
            return Err(
                RenderDispatchPreparationError::ArrayElementPastDescriptorCount {
                    binding: sampler.binding,
                    array_element: sampler.array_element,
                    descriptor_count: sampler.descriptor_count,
                },
            );
        }
        if !sampler_slots.insert((sampler.binding, sampler.array_element)) {
            return Err(RenderDispatchPreparationError::DuplicateSamplerBinding {
                binding: sampler.binding,
                array_element: sampler.array_element,
            });
        }
        if let Some(found) = declarations.get(&sampler.binding) {
            if found.0.is_some() {
                return Err(RenderDispatchPreparationError::DescriptorClassCollision {
                    binding: sampler.binding,
                });
            }
            if found.1 != sampler.descriptor_count {
                return Err(RenderDispatchPreparationError::DescriptorCountMismatch {
                    binding: sampler.binding,
                });
            }
        } else {
            declarations.insert(sampler.binding, (None, sampler.descriptor_count));
        }
    }
    for null in &operation.null_bindings {
        if !matches!(
            null.class,
            RenderBindingClass::SampledImage | RenderBindingClass::StorageImage
        ) {
            return Err(RenderDispatchPreparationError::BindingViewMismatch {
                class: null.class,
                binding: null.binding,
                array_element: null.array_element,
            });
        }
        if null.descriptor_count == 0 {
            return Err(RenderDispatchPreparationError::EmptyDescriptorArray {
                binding: null.binding,
            });
        }
        if null.array_element >= null.descriptor_count {
            return Err(
                RenderDispatchPreparationError::ArrayElementPastDescriptorCount {
                    binding: null.binding,
                    array_element: null.array_element,
                    descriptor_count: null.descriptor_count,
                },
            );
        }
        if !slots.insert((null.class, null.binding, null.array_element)) {
            return Err(RenderDispatchPreparationError::DuplicateBinding {
                class: null.class,
                binding: null.binding,
                array_element: null.array_element,
            });
        }
        if let Some(found) = declarations.get(&null.binding) {
            if found.0 != Some(null.class) {
                return Err(RenderDispatchPreparationError::DescriptorClassCollision {
                    binding: null.binding,
                });
            }
            if found.1 != null.descriptor_count {
                return Err(RenderDispatchPreparationError::DescriptorCountMismatch {
                    binding: null.binding,
                });
            }
        } else {
            declarations.insert(null.binding, (Some(null.class), null.descriptor_count));
        }
    }
    let mut vertex_layouts = BTreeSet::new();
    for layout in &operation.vertex_buffers {
        if layout.stride == 0 {
            return Err(RenderDispatchPreparationError::EmptyVertexStride(
                layout.binding,
            ));
        }
        if !vertex_layouts.insert(layout.binding) {
            return Err(RenderDispatchPreparationError::DuplicateVertexLayout(
                layout.binding,
            ));
        }
        let mut bindings = operation.resources.iter().filter(|resource| {
            resource.class == RenderBindingClass::VertexBuffer && resource.binding == layout.binding
        });
        let Some(binding) = bindings.next() else {
            return Err(RenderDispatchPreparationError::MissingVertexBinding(
                layout.binding,
            ));
        };
        if bindings.next().is_some() || binding.mode != AccessMode::Read {
            return Err(RenderDispatchPreparationError::InvalidVertexBinding(
                layout.binding,
            ));
        }
    }
    if let Some(binding) = operation.resources.iter().find(|resource| {
        resource.class == RenderBindingClass::VertexBuffer
            && !vertex_layouts.contains(&resource.binding)
    }) {
        return Err(RenderDispatchPreparationError::MissingVertexLayout(
            binding.binding,
        ));
    }
    if matches!(
        operation.draw,
        ResolvedRenderDraw::Indexed { .. } | ResolvedRenderDraw::IndexedIndirect { .. }
    ) {
        let mut indexes = operation
            .resources
            .iter()
            .filter(|resource| resource.class == RenderBindingClass::IndexBuffer);
        let Some(index) = indexes.next() else {
            return Err(RenderDispatchPreparationError::MissingIndexBinding(0));
        };
        if indexes.next().is_some() || index.mode != AccessMode::Read {
            return Err(RenderDispatchPreparationError::InvalidIndexBinding(
                index.binding,
            ));
        }
    }
    if matches!(
        operation.draw,
        ResolvedRenderDraw::Indirect { .. } | ResolvedRenderDraw::IndexedIndirect { .. }
    ) {
        let mut arguments = operation
            .resources
            .iter()
            .filter(|resource| resource.class == RenderBindingClass::IndirectBuffer);
        let Some(binding) = arguments.next() else {
            return Err(RenderDispatchPreparationError::MissingIndirectBinding(0));
        };
        if arguments.next().is_some() || binding.mode != AccessMode::Read {
            return Err(RenderDispatchPreparationError::InvalidIndirectBinding(
                binding.binding,
            ));
        }
    }
    Ok(())
}

pub fn cancel_prepared_render_dispatch<T, NativePipeline, Operation>(
    owner: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedRenderDispatch<NativePipeline, Operation>,
) -> Result<Operation, PreparedRenderDispatch<NativePipeline, Operation>> {
    if owner.validate_cancel_gpu_writes(prepared.writes()).is_err()
        || owner
            .validate_cancel_representation_uses(prepared.transaction, prepared.uses())
            .is_err()
    {
        return Err(prepared);
    }
    owner
        .cancel_gpu_writes(prepared.writes())
        .expect("render write cancellation was prevalidated");
    let _: Vec<(BackingId, ManagedBackingProgress<T>)> = owner
        .cancel_representation_uses(prepared.transaction, prepared.uses())
        .expect("render use cancellation was prevalidated");
    Ok(prepared.into_operation())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assemble_prepared_exec_resources, cancel_prepared_exec_resources, ExecTransaction,
        PipelineLifecycle, PipelineReadiness, PreparedExecResourceInputs, RepresentationRoute,
        ResolvedExecSegment, ResolvedExecStream, ResolvedInfoOperation, ResolvedOperation,
        ResolvedResourceLifecycle, ResolvedTextureBindingView, ResolvedTextureViewRange,
        ResourceLifecycleEffect, SessionGeneration, StorageBacking, VulkanDeviceEpoch,
        GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ContentVersion, SegmentBoundary, SegmentKind, SessionGenerationId, SubmissionIdentity,
        TaskId, VulkanDeviceEpochId,
    };

    const EPOCH: VulkanDeviceEpochId = VulkanDeviceEpochId::new(5);

    fn image_view(resource: ResourceId<ResourceObject>) -> ResolvedTextureBindingView {
        ResolvedTextureBindingView {
            resource,
            base: resource,
            range: ResolvedTextureViewRange {
                level_base: 0,
                level_count: 1,
                slice_base: 0,
                slice_count: 1,
            },
            texture_type: reims_vgpu_protocol::TextureType::D2,
            pixel_format: 80,
            swizzle: reims_vgpu_protocol::swizzle_identity(),
        }
    }

    fn pipeline() -> ReadyPipelineLease<RenderPipelineObject, u32> {
        let id = ResourceId::new(2, 1);
        let mut owner = PipelineLifecycle::<RenderPipelineObject, (), u32, ()>::default();
        owner.declare(id, ()).unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(
                compile,
                crate::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                17,
            )
            .unwrap();
        let PipelineReadiness::Ready(ready) = owner.readiness(id, TransactionId::new(7)).unwrap()
        else {
            unreachable!()
        };
        ready
    }

    fn backing(owner: &mut ResourceLifecycleOwner<()>, current: bool) -> BackingId {
        let ResourceLifecycleEffect::BackingCreated(backing) = owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = owner
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        if current {
            let snapshot = owner
                .snapshot_content(backing, &[BackingRegion::Whole])
                .unwrap();
            for transfer in owner
                .plan_transfers(backing, GUEST_REPRESENTATION, representation, &snapshot)
                .unwrap()
            {
                owner.complete_transfer(transfer).unwrap();
            }
        }
        backing
    }

    fn direct(
        sampled: BackingId,
        target: BackingId,
        target_load: LoadAction,
    ) -> ResolvedRenderDispatch {
        ResolvedRenderDispatch {
            pipeline: ResourceId::new(2, 1),
            program: Default::default(),
            depth_stencil: None,
            render_extent: [32, 16],
            raster: ResolvedRenderRasterState::default(),
            visibility: None,
            begins_encoder: true,
            ends_encoder: true,
            draw: ResolvedRenderDraw::Direct {
                topology: PrimitiveTopology::Triangle,
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            vertex_buffers: Box::new([]),
            attachments: Box::new([ResolvedRenderAttachment {
                role: RenderAttachmentRole::Color(0),
                resource: ResourceId::new(4, 1),
                backing: target,
                regions: Box::new([BackingRegion::Whole]),
                pixel_format: 80,
                extent: [32, 16, 1],
                sample_count: 1,
                load: target_load,
                store: StoreAction::Store,
                clear: RenderAttachmentClear::Color([0; 4]),
                resolve: None,
            }]),
            resources: Box::new([ResolvedRenderResourceBinding {
                class: RenderBindingClass::SampledImage,
                binding: 3,
                array_element: 0,
                descriptor_count: 1,
                stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
                resource: ResourceId::new(3, 1),
                backing: sampled,
                view: RenderBindingView::Image(image_view(ResourceId::new(3, 1))),
                regions: Box::new([BackingRegion::Whole]),
                mode: AccessMode::Read,
            }]),
            null_bindings: Box::new([]),
            samplers: Box::new([]),
        }
    }

    #[test]
    fn prepared_render_owns_pipeline_exact_reads_target_write_and_cancellation() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, false);
        let operation = direct(sampled, target, LoadAction::Clear);
        let accesses = operation.accesses(HazardDomainId::new(9));
        assert_eq!(accesses.len(), 2);
        assert_eq!(accesses[0].mode, AccessMode::Read);
        assert_eq!(
            accesses[0].stages,
            StageScope::Render(RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap())
        );
        assert_eq!(accesses[1].mode, AccessMode::Write);
        assert_eq!(accesses[1].stages, StageScope::ColorAttachment);
        let mut visible = operation.clone();
        visible.visibility = Some(ResolvedRenderVisibility {
            mode: reims_vgpu_protocol::VisibilityResultMode::Boolean,
            resource: ResourceId::new(6, 1),
            backing: BackingId::new(99),
            range: crate::LinearRange::new(24, 8).unwrap(),
        });
        assert!(visible
            .accesses(HazardDomainId::new(9))
            .iter()
            .any(|intent| {
                intent.resource == Some(ResourceId::new(6, 1))
                    && intent.scope == AccessScope::Linear(crate::LinearRange::new(24, 8).unwrap())
                    && intent.mode == AccessMode::Write
                    && intent.stages == StageScope::QueryResolve
            }));
        let prepared = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(7),
            SubmissionId::new(11),
            4,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        assert_eq!(prepared.pipeline().pipeline, operation.pipeline);
        assert_eq!(prepared.uses().len(), 2);
        assert_eq!(prepared.writes().len(), 1);
        assert_eq!(prepared.writes()[0].backing, target);
        assert_eq!(prepared.completions().len(), 1);
        let operation = cancel_prepared_render_dispatch(&mut owner, prepared).unwrap();
        let retry = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(7),
            SubmissionId::new(11),
            4,
            operation,
            pipeline(),
        )
        .unwrap();
        assert_eq!(retry.writes()[0].regions[0].version, ContentVersion::new(3));
        cancel_prepared_render_dispatch(&mut owner, retry).unwrap();
    }

    #[test]
    fn stale_load_refuses_before_another_backing_reserves_a_write() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let writable = backing(&mut owner, true);
        let stale_target = backing(&mut owner, false);
        let mut operation = direct(writable, stale_target, LoadAction::Load);
        operation.resources[0].mode = AccessMode::Write;
        assert!(matches!(
            prepare_render_dispatch(
                &mut owner,
                TransactionId::new(8),
                SubmissionId::new(12),
                2,
                operation.clone(),
                pipeline(),
            ),
            Err(RenderDispatchPreparationError::Backing {
                backing,
                reason: ManagedBackingError::StaleExecutionRepresentation,
                ..
            }) if backing == stale_target
        ));
        let mut retry_operation = direct(writable, stale_target, LoadAction::Clear);
        retry_operation.resources[0].mode = AccessMode::Write;
        let retry = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(8),
            SubmissionId::new(12),
            2,
            retry_operation,
            pipeline(),
        )
        .unwrap();
        assert_eq!(
            retry
                .writes()
                .iter()
                .find(|write| write.backing == writable)
                .unwrap()
                .regions[0]
                .version,
            ContentVersion::new(2)
        );
        cancel_prepared_render_dispatch(&mut owner, retry).unwrap();
    }

    #[test]
    fn whole_exec_envelope_retains_render_pipeline_resources_and_cancellation() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, false);
        let transaction = TransactionId::new(10);
        let submission = SubmissionId::new(14);
        let operation = direct(sampled, target, LoadAction::Clear);
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: submission,
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Render,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::<
                        ResolvedRenderDispatch,
                        (),
                        ResolvedInfoOperation,
                        (),
                        (),
                    >::Render(operation.clone())]),
                }]),
            }]),
            accesses: operation.accesses(HazardDomainId::new(2)),
        };
        let prepared = prepare_render_dispatch(
            &mut owner,
            transaction,
            submission,
            0,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        let resources = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([]),
                compute_dispatches: Box::<[crate::PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches: Box::new([prepared]),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        assert_eq!(resources.backings(), [sampled, target]);
        assert_eq!(resources.resource_completions().len(), 1);
        let cancelled = cancel_prepared_exec_resources(&mut owner, resources).unwrap();
        assert_eq!(cancelled.render_dispatches.as_ref(), [operation]);
    }

    #[test]
    fn descriptor_arrays_accept_distinct_elements_and_reject_sampler_collisions() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, false);
        let mut operation = direct(sampled, target, LoadAction::Clear);
        operation.resources[0].descriptor_count = 2;
        let mut second = operation.resources[0].clone();
        second.array_element = 1;
        operation.resources = Box::new([operation.resources[0].clone(), second]);
        let prepared = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(11),
            SubmissionId::new(15),
            0,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        cancel_prepared_render_dispatch(&mut owner, prepared).unwrap();

        operation.samplers = Box::new([ResolvedRenderSamplerBinding {
            binding: 3,
            array_element: 0,
            descriptor_count: 2,
            stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
            sampler: crate::SamplerResource::normalized_default(3),
        }]);
        assert!(matches!(
            prepare_render_dispatch(
                &mut owner,
                TransactionId::new(11),
                SubmissionId::new(15),
                0,
                operation,
                pipeline(),
            ),
            Err(RenderDispatchPreparationError::DescriptorClassCollision { binding: 3 })
        ));
    }

    #[test]
    fn vertex_layout_and_buffer_binding_are_one_validated_shape() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, false);
        let mut operation = direct(sampled, target, LoadAction::Clear);
        operation.vertex_buffers = Box::new([ResolvedVertexBufferLayout {
            binding: 7,
            stride: 20,
        }]);
        assert_eq!(
            validate_shape(&operation),
            Err(RenderDispatchPreparationError::MissingVertexBinding(7))
        );

        let range = crate::LinearRange::new(0, 64).unwrap();
        operation.resources = Box::new([ResolvedRenderResourceBinding {
            class: RenderBindingClass::VertexBuffer,
            binding: 7,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::VERTEX.into()).unwrap(),
            resource: ResourceId::new(8, 1),
            backing: sampled,
            view: RenderBindingView::Buffer(range),
            regions: Box::new([BackingRegion::Linear(range)]),
            mode: AccessMode::Read,
        }]);
        assert_eq!(validate_shape(&operation), Ok(()));
        assert!(operation
            .accesses(HazardDomainId::new(9))
            .iter()
            .any(|intent| {
                intent.resource == Some(ResourceId::new(8, 1))
                    && intent.scope == AccessScope::Linear(range)
                    && intent.mode == AccessMode::Read
                    && intent.stages == StageScope::VertexInput
            }));

        operation.vertex_buffers = Box::new([]);
        assert_eq!(
            validate_shape(&operation),
            Err(RenderDispatchPreparationError::MissingVertexLayout(7))
        );
    }

    #[test]
    fn indirect_draw_requires_one_read_only_argument_binding() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, false);
        let mut operation = direct(sampled, target, LoadAction::Clear);
        operation.draw = ResolvedRenderDraw::Indirect {
            topology: PrimitiveTopology::Triangle,
        };
        assert_eq!(
            validate_shape(&operation),
            Err(RenderDispatchPreparationError::MissingIndirectBinding(0))
        );

        let range = crate::LinearRange::new(32, RENDER_INDIRECT_ARGUMENT_BYTES).unwrap();
        let mut resources = operation.resources.into_vec();
        resources.push(ResolvedRenderResourceBinding {
            class: RenderBindingClass::IndirectBuffer,
            binding: 0,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::VERTEX.into()).unwrap(),
            resource: ResourceId::new(8, 1),
            backing: sampled,
            view: RenderBindingView::Buffer(range),
            regions: Box::new([BackingRegion::Linear(range)]),
            mode: AccessMode::Read,
        });
        operation.resources = resources.into_boxed_slice();
        assert_eq!(validate_shape(&operation), Ok(()));
        assert!(operation
            .accesses(HazardDomainId::new(9))
            .iter()
            .any(|intent| {
                intent.resource == Some(ResourceId::new(8, 1))
                    && intent.scope == AccessScope::Linear(range)
                    && intent.mode == AccessMode::Read
                    && intent.stages == StageScope::Indirect
            }));

        operation.resources.last_mut().unwrap().mode = AccessMode::Write;
        assert_eq!(
            validate_shape(&operation),
            Err(RenderDispatchPreparationError::InvalidIndirectBinding(0))
        );
    }

    #[test]
    fn explicit_null_image_has_descriptor_shape_without_a_resource_access() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, false);
        let mut operation = direct(sampled, target, LoadAction::Clear);
        let access_count = operation.accesses(HazardDomainId::new(1)).len();
        operation.null_bindings = Box::new([ResolvedRenderNullBinding {
            class: RenderBindingClass::SampledImage,
            binding: 8,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
        }]);
        assert_eq!(
            operation.accesses(HazardDomainId::new(1)).len(),
            access_count
        );
        let prepared = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(31),
            SubmissionId::new(32),
            0,
            operation,
            pipeline(),
        )
        .unwrap();
        cancel_prepared_render_dispatch(&mut owner, prepared).unwrap();
    }

    #[test]
    fn dont_care_store_discards_only_at_timeline_completion_and_resolve_refuses() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, true);
        let mut operation = direct(sampled, target, LoadAction::Clear);
        operation.attachments[0].store = StoreAction::DontCare;
        let prepared = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(12),
            SubmissionId::new(16),
            0,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        assert!(prepared.writes().is_empty());
        assert_eq!(
            prepared.completions(),
            [ResolvedResourceCompletion::Discard {
                backing: target,
                region: BackingRegion::Whole,
            }]
        );
        assert!(!owner
            .snapshot_content(target, &[BackingRegion::Whole])
            .unwrap()
            .is_empty());
        assert_eq!(
            owner.complete_resources(prepared.completions()).unwrap(),
            [crate::ResourceCompletionEffect::Discard {
                backing: target,
                region: BackingRegion::Whole,
            }]
        );
        assert!(owner
            .snapshot_content(target, &[BackingRegion::Whole])
            .unwrap()
            .is_empty());
        cancel_prepared_render_dispatch(&mut owner, prepared).unwrap();

        operation.attachments[0].store = StoreAction::MultisampleResolve;
        assert!(matches!(
            prepare_render_dispatch(
                &mut owner,
                TransactionId::new(12),
                SubmissionId::new(16),
                0,
                operation.clone(),
                pipeline(),
            ),
            Err(RenderDispatchPreparationError::ResolveAttachmentMissing(
                RenderAttachmentRole::Color(0)
            ))
        ));

        let resolve = backing(&mut owner, false);
        operation.attachments[0].sample_count = 4;
        operation.attachments[0].resolve = Some(ResolvedRenderResolveAttachment {
            resource: ResourceId::new(6, 1),
            backing: resolve,
            regions: Box::new([BackingRegion::Whole]),
            pixel_format: operation.attachments[0].pixel_format,
            extent: operation.attachments[0].extent,
            sample_count: 1,
        });
        let prepared = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(12),
            SubmissionId::new(16),
            0,
            operation,
            pipeline(),
        )
        .unwrap();
        assert_eq!(prepared.writes().len(), 1);
        assert_eq!(prepared.writes()[0].backing, resolve);
        assert!(prepared
            .completions()
            .contains(&ResolvedResourceCompletion::Discard {
                backing: target,
                region: BackingRegion::Whole,
            }));
        cancel_prepared_render_dispatch(&mut owner, prepared).unwrap();
    }

    #[test]
    fn continued_encoder_defers_attachment_load_and_store_to_its_boundaries() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut owner, true);
        let target = backing(&mut owner, true);
        let mut operation = direct(sampled, target, LoadAction::Load);
        operation.attachments[0].store = StoreAction::DontCare;
        operation.begins_encoder = false;
        operation.ends_encoder = false;

        let prepared = prepare_render_dispatch(
            &mut owner,
            TransactionId::new(13),
            SubmissionId::new(17),
            1,
            operation,
            pipeline(),
        )
        .unwrap();
        assert!(prepared.writes().is_empty());
        assert!(prepared.completions().is_empty());
        cancel_prepared_render_dispatch(&mut owner, prepared).unwrap();
    }
}
