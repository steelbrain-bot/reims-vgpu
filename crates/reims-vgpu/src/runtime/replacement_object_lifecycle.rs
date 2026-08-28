//! Typed replacement boundary for serializer object-lifecycle packets.
//!
//! The destroy record's opcode is its only object-family tag. Decoding keeps
//! every established family distinct; unsupported lifecycle relations remain
//! representable and cannot fall through to a raw-name deletion in another
//! namespace.

#![allow(dead_code)]

use reims_vgpu_core::endian::ld32;
use reims_vgpu_protocol::{
    ComputePipelineObject, DepthStencilDescriptor, DepthStencilObject, FenceObject, FunctionObject,
    HeapObject, IndirectCommandBufferObject, RasterizationRateMapDescriptor,
    RasterizationRateMapObject, RenderPipelineObject, ResourceDescriptor, ResourceId,
    SamplerDescriptor, SamplerObject, SerializerRef, TaskId, TransactionId,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementDeletedObjectKind {
    Buffer,
    Texture,
    DepthStencil,
    Sampler,
    Function,
    ComputePipeline,
    RenderPipeline,
    Fence,
    Heap,
    RasterizationRateMap,
    IndirectCommandBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementObjectDelete {
    pub task: TaskId,
    pub kind: ReplacementDeletedObjectKind,
    pub reference: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementObjectDeleteDecodeError {
    ShortEnvelope,
    MalformedRecord,
    NotDelete { opcode: u32 },
    ReferenceUnreadable,
}

#[derive(Debug)]
pub(crate) enum ReplacementObjectDeleteEffect {
    Sampler {
        identity: ResourceId<SamplerObject>,
        descriptor: Arc<SamplerDescriptor>,
    },
    DepthStencil {
        identity: ResourceId<DepthStencilObject>,
        descriptor: Arc<DepthStencilDescriptor>,
    },
    Function {
        identity: ResourceId<FunctionObject>,
        function: Arc<crate::runtime::replacement_pipeline_contract::LoadedFunction>,
    },
    RenderPipeline {
        identity: ResourceId<RenderPipelineObject>,
        waiters: Box<[TransactionId]>,
    },
    ComputePipeline {
        identity: ResourceId<ComputePipelineObject>,
        waiters: Box<[TransactionId]>,
    },
    Fence(ResourceId<FenceObject>),
    Heap(ResourceId<HeapObject>),
    RasterizationRateMap {
        identity: ResourceId<RasterizationRateMapObject>,
        descriptor: Arc<RasterizationRateMapDescriptor>,
    },
    IndirectCommandBuffer(
        crate::runtime::replacement_session::ReplacementRetiredIndirectCommandBuffer,
    ),
    /// A delete naming an object this device holds no record of.
    ///
    /// There is nothing to free: no native object was created, no descriptor
    /// retained, no transaction is waiting on it. The guest has already
    /// stopped tracking that name, so the delete's *effect* --- that the name
    /// is gone --- is already true, and this is a contract no-op in the same
    /// sense as `ReplacementControlEffect::AbsentResourceDelete`.
    ///
    /// Refusing instead cost the guest the whole CPU transaction the delete
    /// arrived in, which carried other work. It is reported by name on the
    /// failure channel, because the other reading --- an object this device
    /// *should* have registered and did not --- is a real defect and looks
    /// identical from here.
    AbsentObject(ReplacementDeletedObjectKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementObjectDeleteRefusal {
    ResourceLifecycleUnresolved(ReplacementDeletedObjectKind),
    /// The namespace resolved the object and then declined to release it.
    ///
    /// Not "this device never had it" --- that is
    /// [`ReplacementObjectDeleteEffect::AbsentObject`] and is a no-op. This is
    /// the two halves of one namespace disagreeing about a name inside one
    /// call, which is a device-state defect rather than a wait.
    UnknownObject(ReplacementDeletedObjectKind),
    Object(crate::runtime::replacement_session::ReplacementObjectRetirementError),
    Pipeline(reims_vgpu_core::PipelineLifecycleError),
}

impl ReplacementObjectDeleteRefusal {
    /// Whether no later guest packet can make this delete succeed.
    ///
    /// A namespace that resolved a name and then refused to release it is the
    /// one such arm: both answers came from the same state in the same call,
    /// so no later packet changes either of them. The other three are
    /// genuinely pending -- an unresolved resource lifecycle, an object still
    /// held by live work, a pipeline mid-flight -- and a later packet is
    /// exactly what clears them.
    pub(crate) const fn is_terminal_refusal(&self) -> bool {
        match self {
            Self::UnknownObject(_) => true,
            Self::ResourceLifecycleUnresolved(_) | Self::Object(_) | Self::Pipeline(_) => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementObjectDeleteFailure {
    pub reason: ReplacementObjectDeleteRefusal,
    pub command: DecodedReplacementObjectDelete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementSerializerResourceKind {
    Sampler,
    DepthStencil,
    RenderPipeline,
    ComputePipeline,
    IndirectCommandBuffer,
    RasterizationRateMap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementSerializerResourceEffect {
    Sampler {
        identity: ResourceId<SamplerObject>,
        newly_declared: bool,
    },
    DepthStencil {
        identity: ResourceId<DepthStencilObject>,
        newly_declared: bool,
    },
    IndirectCommandBuffer {
        identity: ResourceId<IndirectCommandBufferObject>,
        newly_declared: bool,
    },
    RasterizationRateMap {
        identity: ResourceId<RasterizationRateMapObject>,
        newly_declared: bool,
    },
    RenderPipeline {
        identity: ResourceId<RenderPipelineObject>,
        newly_declared: bool,
    },
    ComputePipeline {
        identity: ResourceId<ComputePipelineObject>,
        newly_declared: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ReplacementSerializerResourceRefusal {
    NotSerializerResource,
    ConflictingDeclaration(ReplacementSerializerResourceKind),
    RequiredFunctionUnbound(ReplacementSerializerResourceKind),
    FunctionUnavailable {
        pipeline: ReplacementSerializerResourceKind,
        reference: u32,
    },
    RenderMeshFunctionsUnresolved {
        object_function: u32,
        mesh_function: u32,
    },
    RenderTessellationUnresolved {
        max_factor: u32,
        factor_step_function: u32,
        output_winding_order: u32,
    },
    ComputeStageInputUnresolved(Box<reims_vgpu_protocol::ComputeStageInputDescriptor>),
    SamplerPropertiesUnresolved {
        support_argument_buffers: bool,
        unidentified_flags: u8,
        lod_average: bool,
    },
    DepthStencilPropertiesUnresolved {
        front_unidentified_ops: u32,
        back_unidentified_ops: u32,
    },
    IndirectCommandDescriptorUnresolved {
        flags: u16,
        unapplied: Box<[reims_vgpu_protocol::IcbUnappliedFlag]>,
        unidentified_delta: u16,
        unidentified_u8_a: u8,
        unidentified_u8_b: u8,
        unidentified_u32: u32,
        options: u16,
    },
    Declaration(crate::runtime::replacement_session::ReplacementObjectDeclarationError),
    PipelineDeclaration(crate::runtime::replacement_session::ReplacementPipelineDeclarationError),
    PipelineTranslationLifecycle(reims_vgpu_core::PipelineLifecycleError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementFunctionEffect {
    pub identity: ResourceId<FunctionObject>,
    pub newly_declared: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementFunctionRefusal {
    ConflictingDeclaration,
    Declaration(crate::runtime::replacement_session::ReplacementObjectDeclarationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementLinearResourceRefusal {
    NotLinearResource,
    BufferDescriptorUnresolved {
        handle64: u64,
        unidentified_flags: u64,
    },
    InvalidBackingGeometry,
    Declaration(crate::runtime::replacement_session::ReplacementLinearResourceDeclarationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRegisteredSurfaceRefusal {
    NotRegisteredSurface,
    Declaration(crate::runtime::replacement_session::ReplacementRegisteredSurfaceDeclarationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTextureViewRefusal {
    NotTextureView,
    DeclaredReferenceMismatch { declared: u32, object: u32 },
    ParentUnbound,
    ParentUnavailable(u32),
    ParentKindUnsupported(reims_vgpu_protocol::ObjectKind),
    Declaration(crate::runtime::replacement_session::ReplacementViewResourceDeclarationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementIOSurfacePlaneViewRefusal {
    NotIOSurfacePlaneView,
    Incomplete(reims_vgpu_protocol::IOSurfacePlaneViewDecodeState),
    DeclaredReferenceMismatch {
        declared: Option<u32>,
        object: u32,
    },
    /// The descriptor's surface field is the unbound reference. An unbound
    /// `ref == 0` is ordinary control flow elsewhere in this device, so this
    /// arm is separated from the self-reference below: the two say different
    /// things about the record and only one of them is a shape this decoder
    /// does not understand.
    SurfaceRefUnbound,
    /// The descriptor's surface field names the plane view's own object slot
    /// *in its own task's namespace*, which leaves no parent to resolve.
    ///
    /// The surface field and the view's own ref are two object ids in two
    /// object lists: the descriptor carries `owner_task` precisely because the
    /// surface is registered against the task that owns it while the view
    /// belongs to the task whose list is being admitted, and the two differ by
    /// construction. Equal ids across different tasks name different objects
    /// and are ordinary, so the task has to match before the equality means
    /// anything at all.
    SurfaceRefSelf(u32),
    SurfaceUnavailable(u32),
    ParentKindUnsupported(reims_vgpu_protocol::ObjectKind),
    ParentDescriptorUnavailable,
    PlaneOutOfBounds {
        plane: u32,
        count: u8,
    },
    /// The view's own geometry is not the geometry the parent declared for
    /// the plane it selects.
    ///
    /// Both geometries, the plane, and the record variant, because none of
    /// them is inferable from the others and the repair differs by which
    /// term is wrong: a factor of two on one axis is chroma subsampling
    /// counted in the wrong units, a small excess is an alignment the parent
    /// declaration rounds and the view does not, and a `ColorView` that
    /// mismatches where a `Plane` would not means the two record variants do
    /// not in fact share a geometry contract. A boot has already been spent
    /// on this refusal reporting only its own name.
    PlaneGeometryMismatch {
        record: Option<reims_vgpu_protocol::IOSurfacePlaneViewRecordKind>,
        plane: u32,
        view: (u32, u32, u32),
        declared: (u32, u32),
    },
    Declaration(crate::runtime::replacement_session::ReplacementIOSurfacePlaneViewDeclarationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementBufferTextureRefusal {
    NotBufferTexture,
    DeclaredReferenceMismatch { declared: u32, object: u32 },
    BufferUnbound,
    BufferUnavailable(u32),
    ParentKindUnsupported(reims_vgpu_protocol::ObjectKind),
    Declaration(crate::runtime::replacement_session::ReplacementBufferTextureDeclarationError),
}

pub(crate) fn apply_replacement_linear_resource<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    page_shift: u32,
    task: TaskId,
    reference: u32,
    descriptor: ResourceDescriptor,
) -> Result<
    crate::runtime::replacement_session::ReplacementLinearResourceDeclaration,
    ReplacementLinearResourceRefusal,
> {
    let (kind, address, length) = match &descriptor {
        ResourceDescriptor::Buffer(buffer) => {
            let unidentified_flags = buffer.unidentified_handle_flags();
            if unidentified_flags != 0 {
                return Err(
                    ReplacementLinearResourceRefusal::BufferDescriptorUnresolved {
                        handle64: buffer.handle64,
                        unidentified_flags,
                    },
                );
            }
            let (address, length) = buffer
                .backing_gva_size(page_shift)
                .ok_or(ReplacementLinearResourceRefusal::InvalidBackingGeometry)?;
            (reims_vgpu_protocol::ObjectKind::Buffer, address, length)
        }
        ResourceDescriptor::Texture(texture) => {
            let address = texture
                .allocation_base_gva(page_shift)
                .ok_or(ReplacementLinearResourceRefusal::InvalidBackingGeometry)?;
            if texture.allocation_size == 0 {
                return Err(ReplacementLinearResourceRefusal::InvalidBackingGeometry);
            }
            (
                reims_vgpu_protocol::ObjectKind::Texture,
                address,
                texture.allocation_size,
            )
        }
        _ => return Err(ReplacementLinearResourceRefusal::NotLinearResource),
    };
    runtime
        .declare_task_address_resource(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(reference),
            kind,
            Arc::new(descriptor),
            reims_vgpu_protocol::GuestVirtualAddress::new(address),
            reims_vgpu_protocol::ByteLength::new(length),
        )
        .map_err(ReplacementLinearResourceRefusal::Declaration)
}

pub(crate) fn apply_replacement_registered_surface<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: ResourceDescriptor,
) -> Result<
    crate::runtime::replacement_session::ReplacementRegisteredSurfaceDeclaration,
    ReplacementRegisteredSurfaceRefusal,
> {
    let ResourceDescriptor::SurfaceBacking(descriptor) = descriptor else {
        return Err(ReplacementRegisteredSurfaceRefusal::NotRegisteredSurface);
    };
    runtime
        .declare_registered_surface_resource(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(reference),
            Arc::new(descriptor),
        )
        .map_err(ReplacementRegisteredSurfaceRefusal::Declaration)
}

pub(crate) fn apply_replacement_texture_view<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: ResourceDescriptor,
) -> Result<
    crate::runtime::replacement_session::ReplacementViewResourceDeclaration,
    ReplacementTextureViewRefusal,
> {
    let ResourceDescriptor::TextureView(view) = &descriptor else {
        return Err(ReplacementTextureViewRefusal::NotTextureView);
    };
    if view.view_texture_ref != reference {
        return Err(ReplacementTextureViewRefusal::DeclaredReferenceMismatch {
            declared: view.view_texture_ref,
            object: reference,
        });
    }
    if view.base_texture_ref == 0 || view.base_texture_ref == reference {
        return Err(ReplacementTextureViewRefusal::ParentUnbound);
    }
    let parent = runtime
        .resolve_resource(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(view.base_texture_ref),
        )
        .ok_or(ReplacementTextureViewRefusal::ParentUnavailable(
            view.base_texture_ref,
        ))?;
    let parent_kind =
        runtime
            .resource_kind(parent)
            .ok_or(ReplacementTextureViewRefusal::ParentUnavailable(
                view.base_texture_ref,
            ))?;
    if !matches!(
        parent_kind,
        reims_vgpu_protocol::ObjectKind::Texture
            | reims_vgpu_protocol::ObjectKind::TextureView
            | reims_vgpu_protocol::ObjectKind::IOSurfaceTexture
            | reims_vgpu_protocol::ObjectKind::IOSurfacePlaneView
    ) {
        return Err(ReplacementTextureViewRefusal::ParentKindUnsupported(
            parent_kind,
        ));
    }
    let declaration = runtime
        .declare_resource_view(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(reference),
            reims_vgpu_protocol::ObjectKind::TextureView,
            Arc::new(descriptor),
            parent,
        )
        .map_err(ReplacementTextureViewRefusal::Declaration)?;
    // Installing the view is part of declaring it, whenever the image it
    // belongs on already exists.
    //
    // The other order needs nothing: a texture materialized after its views
    // were declared reads them and carries them, so the two rules between them
    // cover both and neither has to know which happened. Doing it anywhere
    // else means knowing which resources some later packet will name, and a
    // binding reached through the object table -- every sampled compute
    // texture -- is named nowhere a materialization pass can see.
    runtime.install_declared_texture_view(declaration.resource);
    Ok(declaration)
}

pub(crate) fn apply_replacement_iosurface_plane_view<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: ResourceDescriptor,
) -> Result<
    crate::runtime::replacement_session::ReplacementViewResourceDeclaration,
    ReplacementIOSurfacePlaneViewRefusal,
> {
    let ResourceDescriptor::IOSurfacePlaneView(view_resource) = descriptor else {
        return Err(ReplacementIOSurfacePlaneViewRefusal::NotIOSurfacePlaneView);
    };
    if view_resource.decode_state != reims_vgpu_protocol::IOSurfacePlaneViewDecodeState::Complete {
        return Err(ReplacementIOSurfacePlaneViewRefusal::Incomplete(
            view_resource.decode_state,
        ));
    }
    if view_resource.own_ref.map(|object| object.get()) != Some(reference) {
        return Err(
            ReplacementIOSurfacePlaneViewRefusal::DeclaredReferenceMismatch {
                declared: view_resource.own_ref.map(|object| object.get()),
                object: reference,
            },
        );
    }
    let surface_ref = view_resource.surface.get();
    if surface_ref == 0 {
        return Err(ReplacementIOSurfacePlaneViewRefusal::SurfaceRefUnbound);
    }
    if view_resource.owner_task == task && surface_ref == reference {
        return Err(ReplacementIOSurfacePlaneViewRefusal::SurfaceRefSelf(
            surface_ref,
        ));
    }
    // The descriptor's owner task names the namespace containing the parent
    // surface. The plane view itself belongs to the task whose object list is
    // being admitted, so cross-task parentage is part of the wire contract.
    let surface = runtime
        .resolve_resource(view_resource.owner_task, view_resource.surface)
        .ok_or(ReplacementIOSurfacePlaneViewRefusal::SurfaceUnavailable(
            surface_ref,
        ))?;
    let parent_kind = runtime.resource_kind(surface).ok_or(
        ReplacementIOSurfacePlaneViewRefusal::SurfaceUnavailable(surface_ref),
    )?;
    if parent_kind != reims_vgpu_protocol::ObjectKind::SurfaceBacking {
        return Err(ReplacementIOSurfacePlaneViewRefusal::ParentKindUnsupported(
            parent_kind,
        ));
    }
    let graph = runtime.execution().resources().graph();
    let parent = graph
        .resource(surface)
        .and_then(|node| node.descriptor.as_deref())
        .and_then(|descriptor| match descriptor {
            ResourceDescriptor::SurfaceBacking(descriptor) => Some(descriptor),
            _ => None,
        })
        .ok_or(ReplacementIOSurfacePlaneViewRefusal::ParentDescriptorUnavailable)?;
    let view = view_resource
        .view
        .ok_or(ReplacementIOSurfacePlaneViewRefusal::Incomplete(
            view_resource.decode_state,
        ))?;
    let plane = usize::try_from(view.plane_index).ok().filter(|plane| {
        *plane < usize::from(parent.plane_count)
            && *plane < reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_CAP
    });
    let Some(plane) = plane else {
        return Err(ReplacementIOSurfacePlaneViewRefusal::PlaneOutOfBounds {
            plane: view.plane_index,
            count: parent.plane_count,
        });
    };
    let declared_plane = parent.planes[plane];
    // `depth` is not re-checked here: the decoder admits a view only when it
    // decodes as 1, so a second test of it could never fail and would read as
    // cover this refusal does not have.
    if view.width != declared_plane.width || view.height != declared_plane.height {
        return Err(
            ReplacementIOSurfacePlaneViewRefusal::PlaneGeometryMismatch {
                record: view_resource.record_kind,
                plane: view.plane_index,
                view: (view.width, view.height, view.depth),
                declared: (declared_plane.width, declared_plane.height),
            },
        );
    }
    runtime
        .declare_io_surface_plane_view(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(reference),
            Arc::new(ResourceDescriptor::IOSurfacePlaneView(view_resource)),
            surface,
            reims_vgpu_protocol::PlaneIndex::new(view.plane_index),
        )
        .map_err(ReplacementIOSurfacePlaneViewRefusal::Declaration)
}

pub(crate) fn apply_replacement_buffer_texture<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: ResourceDescriptor,
) -> Result<
    crate::runtime::replacement_session::ReplacementViewResourceDeclaration,
    ReplacementBufferTextureRefusal,
> {
    let ResourceDescriptor::BufferTexture(buffer_texture) = descriptor else {
        return Err(ReplacementBufferTextureRefusal::NotBufferTexture);
    };
    if buffer_texture.new_texture_ref != reference {
        return Err(ReplacementBufferTextureRefusal::DeclaredReferenceMismatch {
            declared: buffer_texture.new_texture_ref,
            object: reference,
        });
    }
    if buffer_texture.buffer_ref == 0 || buffer_texture.buffer_ref == reference {
        return Err(ReplacementBufferTextureRefusal::BufferUnbound);
    }
    let buffer = runtime
        .resolve_resource(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(buffer_texture.buffer_ref),
        )
        .ok_or(ReplacementBufferTextureRefusal::BufferUnavailable(
            buffer_texture.buffer_ref,
        ))?;
    let kind =
        runtime
            .resource_kind(buffer)
            .ok_or(ReplacementBufferTextureRefusal::BufferUnavailable(
                buffer_texture.buffer_ref,
            ))?;
    if kind != reims_vgpu_protocol::ObjectKind::Buffer {
        return Err(ReplacementBufferTextureRefusal::ParentKindUnsupported(kind));
    }
    runtime
        .declare_buffer_texture(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(reference),
            Arc::new(buffer_texture),
            buffer,
        )
        .map_err(ReplacementBufferTextureRefusal::Declaration)
}

pub(crate) fn apply_replacement_function_bytes<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    mtlb: Arc<[u8]>,
) -> Result<ReplacementFunctionEffect, ReplacementFunctionRefusal> {
    let reference = SerializerRef::new(reference);
    if let Some((identity, current)) = runtime.resolve_function(task, reference) {
        return if current.mtlb.as_ref() == mtlb.as_ref() {
            Ok(ReplacementFunctionEffect {
                identity,
                newly_declared: false,
            })
        } else {
            Err(ReplacementFunctionRefusal::ConflictingDeclaration)
        };
    }
    runtime
        .declare_function(
            task,
            reference,
            crate::runtime::replacement_pipeline_contract::LoadedFunction { mtlb },
        )
        .map(|identity| ReplacementFunctionEffect {
            identity,
            newly_declared: true,
        })
        .map_err(ReplacementFunctionRefusal::Declaration)
}

/// Publish one already-decoded object-list serializer resource into the active
/// semantic generation. Re-reading the same immutable object-list entry is
/// idempotent; changing its descriptor without the corresponding destroy is a
/// typed lifecycle conflict.
pub(crate) fn apply_replacement_serializer_resource<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: ResourceDescriptor,
) -> Result<ReplacementSerializerResourceEffect, ReplacementSerializerResourceRefusal> {
    match descriptor {
        ResourceDescriptor::Sampler(descriptor) => {
            if descriptor.support_argument_buffers
                || descriptor.unidentified_flags != 0
                || descriptor.lod_average
            {
                return Err(
                    ReplacementSerializerResourceRefusal::SamplerPropertiesUnresolved {
                        support_argument_buffers: descriptor.support_argument_buffers,
                        unidentified_flags: descriptor.unidentified_flags,
                        lod_average: descriptor.lod_average,
                    },
                );
            }
            let reference = SerializerRef::new(reference);
            if let Some((identity, current)) = runtime.resolve_sampler(task, reference) {
                return if sampler_declarations_equal(&current, &descriptor) {
                    Ok(ReplacementSerializerResourceEffect::Sampler {
                        identity,
                        newly_declared: false,
                    })
                } else {
                    Err(
                        ReplacementSerializerResourceRefusal::ConflictingDeclaration(
                            ReplacementSerializerResourceKind::Sampler,
                        ),
                    )
                };
            }
            runtime
                .declare_sampler(task, reference, descriptor)
                .map(|identity| ReplacementSerializerResourceEffect::Sampler {
                    identity,
                    newly_declared: true,
                })
                .map_err(ReplacementSerializerResourceRefusal::Declaration)
        }
        ResourceDescriptor::DepthStencil(descriptor) => {
            if descriptor.front_face.unidentified_ops != 0
                || descriptor.back_face.unidentified_ops != 0
            {
                return Err(
                    ReplacementSerializerResourceRefusal::DepthStencilPropertiesUnresolved {
                        front_unidentified_ops: descriptor.front_face.unidentified_ops,
                        back_unidentified_ops: descriptor.back_face.unidentified_ops,
                    },
                );
            }
            let reference = SerializerRef::new(reference);
            if let Some((identity, current)) = runtime.resolve_depth_stencil(task, reference) {
                return if current.as_ref() == &descriptor {
                    Ok(ReplacementSerializerResourceEffect::DepthStencil {
                        identity,
                        newly_declared: false,
                    })
                } else {
                    Err(
                        ReplacementSerializerResourceRefusal::ConflictingDeclaration(
                            ReplacementSerializerResourceKind::DepthStencil,
                        ),
                    )
                };
            }
            runtime
                .declare_depth_stencil(task, reference, descriptor)
                .map(
                    |identity| ReplacementSerializerResourceEffect::DepthStencil {
                        identity,
                        newly_declared: true,
                    },
                )
                .map_err(ReplacementSerializerResourceRefusal::Declaration)
        }
        ResourceDescriptor::IndirectCommandBuffer(descriptor) => {
            let unapplied = descriptor.unapplied_flags();
            let unidentified_delta = descriptor.unidentified_flags()
                ^ reims_vgpu_protocol::IndirectCommandBufferDescriptor::UNIDENTIFIED_FLAGS_DEFAULT;
            if !unapplied.is_empty()
                || unidentified_delta != 0
                || descriptor.unidentified_u8_a
                    != reims_vgpu_protocol::IndirectCommandBufferDescriptor::UNIDENTIFIED_U8_A_DEFAULT
                || descriptor.unidentified_u8_b
                    != reims_vgpu_protocol::IndirectCommandBufferDescriptor::UNIDENTIFIED_U8_B_DEFAULT
                || descriptor.unidentified_u32
                    != reims_vgpu_protocol::IndirectCommandBufferDescriptor::UNIDENTIFIED_U32_DEFAULT
                || descriptor.options != 0
            {
                return Err(
                    ReplacementSerializerResourceRefusal::IndirectCommandDescriptorUnresolved {
                        flags: descriptor.flags,
                        unapplied: unapplied.into_boxed_slice(),
                        unidentified_delta,
                        unidentified_u8_a: descriptor.unidentified_u8_a,
                        unidentified_u8_b: descriptor.unidentified_u8_b,
                        unidentified_u32: descriptor.unidentified_u32,
                        options: descriptor.options,
                    },
                );
            }
            let reference = SerializerRef::new(reference);
            if let Some((identity, current)) =
                runtime.resolve_indirect_command_buffer(task, reference)
            {
                return if current.as_ref() == &descriptor {
                    Ok(ReplacementSerializerResourceEffect::IndirectCommandBuffer {
                        identity,
                        newly_declared: false,
                    })
                } else {
                    Err(
                        ReplacementSerializerResourceRefusal::ConflictingDeclaration(
                            ReplacementSerializerResourceKind::IndirectCommandBuffer,
                        ),
                    )
                };
            }
            runtime
                .declare_indirect_command_buffer(task, reference, descriptor)
                .map(
                    |identity| ReplacementSerializerResourceEffect::IndirectCommandBuffer {
                        identity,
                        newly_declared: true,
                    },
                )
                .map_err(ReplacementSerializerResourceRefusal::Declaration)
        }
        ResourceDescriptor::RenderPipeline(descriptor) => {
            apply_render_pipeline_resource(runtime, task, reference, descriptor)
        }
        ResourceDescriptor::ComputePipeline(descriptor) => {
            apply_compute_pipeline_resource(runtime, task, reference, descriptor)
        }
        ResourceDescriptor::RasterizationRateMap(descriptor) => {
            let reference = SerializerRef::new(reference);
            if let Some((identity, current)) =
                runtime.resolve_rasterization_rate_map(task, reference)
            {
                return if current.as_ref() == &descriptor {
                    Ok(ReplacementSerializerResourceEffect::RasterizationRateMap {
                        identity,
                        newly_declared: false,
                    })
                } else {
                    Err(
                        ReplacementSerializerResourceRefusal::ConflictingDeclaration(
                            ReplacementSerializerResourceKind::RasterizationRateMap,
                        ),
                    )
                };
            }
            runtime
                .declare_rasterization_rate_map(task, reference, descriptor)
                .map(
                    |identity| ReplacementSerializerResourceEffect::RasterizationRateMap {
                        identity,
                        newly_declared: true,
                    },
                )
                .map_err(ReplacementSerializerResourceRefusal::Declaration)
        }
        ResourceDescriptor::Buffer(_)
        | ResourceDescriptor::Texture(_)
        | ResourceDescriptor::SurfaceBacking(_)
        | ResourceDescriptor::Function(_)
        | ResourceDescriptor::TextureView(_)
        | ResourceDescriptor::BufferTexture(_)
        | ResourceDescriptor::HeapTexture(_)
        | ResourceDescriptor::IOSurfacePlaneView(_)
        | ResourceDescriptor::MapperIOSurfaceTextureView(_) => {
            Err(ReplacementSerializerResourceRefusal::NotSerializerResource)
        }
    }
}

fn apply_render_pipeline_resource<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: reims_vgpu_protocol::RenderPipelineDescriptor,
) -> Result<ReplacementSerializerResourceEffect, ReplacementSerializerResourceRefusal> {
    let kind = ReplacementSerializerResourceKind::RenderPipeline;
    if descriptor.object_func_ref != 0 || descriptor.mesh_func_ref != 0 {
        return Err(
            ReplacementSerializerResourceRefusal::RenderMeshFunctionsUnresolved {
                object_function: descriptor.object_func_ref,
                mesh_function: descriptor.mesh_func_ref,
            },
        );
    }
    if descriptor.max_tessellation_factor != 0
        || descriptor.tessellation_factor_step_function != 0
        || descriptor.tessellation_output_winding_order != 0
    {
        return Err(
            ReplacementSerializerResourceRefusal::RenderTessellationUnresolved {
                max_factor: descriptor.max_tessellation_factor,
                factor_step_function: descriptor.tessellation_factor_step_function,
                output_winding_order: descriptor.tessellation_output_winding_order,
            },
        );
    }
    if descriptor.vertex_func_ref == 0 || descriptor.fragment_func_ref == 0 {
        return Err(ReplacementSerializerResourceRefusal::RequiredFunctionUnbound(kind));
    }
    let vertex = runtime
        .resolve_function(task, SerializerRef::new(descriptor.vertex_func_ref))
        .map(|(_, function)| function)
        .ok_or(ReplacementSerializerResourceRefusal::FunctionUnavailable {
            pipeline: kind,
            reference: descriptor.vertex_func_ref,
        })?;
    let fragment = runtime
        .resolve_function(task, SerializerRef::new(descriptor.fragment_func_ref))
        .map(|(_, function)| function)
        .ok_or(ReplacementSerializerResourceRefusal::FunctionUnavailable {
            pipeline: kind,
            reference: descriptor.fragment_func_ref,
        })?;
    let contract = crate::runtime::replacement_session::RenderPipelineContract {
        descriptor: Arc::new(descriptor),
        vertex_library: Arc::clone(&vertex.mtlb),
        fragment_library: Arc::clone(&fragment.mtlb),
    };
    let reference = SerializerRef::new(reference);
    if let Some(identity) = runtime.resolve_render_pipeline(task, reference) {
        return if runtime.session().render_contract(identity).as_deref() == Some(&contract) {
            Ok(ReplacementSerializerResourceEffect::RenderPipeline {
                identity,
                newly_declared: false,
            })
        } else {
            Err(ReplacementSerializerResourceRefusal::ConflictingDeclaration(kind))
        };
    }
    let identity = runtime
        .declare_render_pipeline(task, reference, contract)
        .map_err(ReplacementSerializerResourceRefusal::PipelineDeclaration)?;
    runtime
        .session()
        .schedule_render_translation(identity, reference.get())
        .map_err(ReplacementSerializerResourceRefusal::PipelineTranslationLifecycle)?;
    Ok(ReplacementSerializerResourceEffect::RenderPipeline {
        identity,
        newly_declared: true,
    })
}

fn apply_compute_pipeline_resource<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: TaskId,
    reference: u32,
    descriptor: reims_vgpu_protocol::ComputePipelineDescriptor,
) -> Result<ReplacementSerializerResourceEffect, ReplacementSerializerResourceRefusal> {
    let kind = ReplacementSerializerResourceKind::ComputePipeline;
    if let Some(stage_input) = descriptor.stage_input.as_ref() {
        return Err(
            ReplacementSerializerResourceRefusal::ComputeStageInputUnresolved(Box::new(
                stage_input.clone(),
            )),
        );
    }
    if descriptor.kernel_func_ref == 0 {
        return Err(ReplacementSerializerResourceRefusal::RequiredFunctionUnbound(kind));
    }
    let kernel = runtime
        .resolve_function(task, SerializerRef::new(descriptor.kernel_func_ref))
        .map(|(_, function)| function)
        .ok_or(ReplacementSerializerResourceRefusal::FunctionUnavailable {
            pipeline: kind,
            reference: descriptor.kernel_func_ref,
        })?;
    let contract = crate::runtime::replacement_pipeline_contract::LoadedComputePipeline {
        kernel_func_ref: descriptor.kernel_func_ref,
        kernel_mtlb: Arc::clone(&kernel.mtlb),
        max_total_threads_per_threadgroup: descriptor.max_total_threads_per_threadgroup,
        supports_indirect_command_buffers: descriptor.supports_indirect_command_buffers,
        stage_input: descriptor.stage_input,
    };
    let reference = SerializerRef::new(reference);
    if let Some(identity) = runtime.resolve_compute_pipeline(task, reference) {
        return if runtime.session().compute_contract(identity).as_deref() == Some(&contract) {
            Ok(ReplacementSerializerResourceEffect::ComputePipeline {
                identity,
                newly_declared: false,
            })
        } else {
            Err(ReplacementSerializerResourceRefusal::ConflictingDeclaration(kind))
        };
    }
    runtime
        .declare_compute_pipeline(task, reference, contract)
        .map(
            |identity| ReplacementSerializerResourceEffect::ComputePipeline {
                identity,
                newly_declared: true,
            },
        )
        .map_err(ReplacementSerializerResourceRefusal::PipelineDeclaration)
}

fn sampler_declarations_equal(left: &SamplerDescriptor, right: &SamplerDescriptor) -> bool {
    left.min_filter == right.min_filter
        && left.mag_filter == right.mag_filter
        && left.mip_filter == right.mip_filter
        && left.s_address == right.s_address
        && left.t_address == right.t_address
        && left.r_address == right.r_address
        && left.max_anisotropy == right.max_anisotropy
        && left.lod_min_clamp.to_bits() == right.lod_min_clamp.to_bits()
        && left.lod_max_clamp.to_bits() == right.lod_max_clamp.to_bits()
        && left.compare_function == right.compare_function
        && left.border_color == right.border_color
        && left.normalized_coordinates == right.normalized_coordinates
        && left.support_argument_buffers == right.support_argument_buffers
        && left.unidentified_flags == right.unidentified_flags
        && left.lod_average == right.lod_average
}

pub(crate) fn decode_replacement_object_delete(
    payload: &[u8],
) -> Result<DecodedReplacementObjectDelete, ReplacementObjectDeleteDecodeError> {
    if payload.len() < 4 {
        return Err(ReplacementObjectDeleteDecodeError::ShortEnvelope);
    }
    let record = &payload[4..];
    let op = reims_vgpu_wire::op::op(record, 0)
        .map_err(|_| ReplacementObjectDeleteDecodeError::MalformedRecord)?;
    let kind = delete_kind(op.opcode()).ok_or(ReplacementObjectDeleteDecodeError::NotDelete {
        opcode: op.opcode(),
    })?;
    let delete = reims_vgpu_wire::ops::destroy::delete(&op)
        .map_err(|_| ReplacementObjectDeleteDecodeError::ReferenceUnreadable)?;
    Ok(DecodedReplacementObjectDelete {
        task: TaskId::new(ld32(payload)),
        kind,
        reference: delete.object_ref.get(),
    })
}

/// One stable key per (task, kind, reference), so a guest that recycles a small
/// pool of names reports each distinct absent delete once rather than on every
/// create/delete round.
fn absent_object_key(command: DecodedReplacementObjectDelete) -> u64 {
    (u64::from(command.task.get()) << 40)
        | (u64::from(command.kind as u8) << 32)
        | u64::from(command.reference)
}

fn report_absent_object(command: DecodedReplacementObjectDelete) -> ReplacementObjectDeleteEffect {
    if crate::observe::first_sight(
        "replacement_object_delete_absent",
        absent_object_key(command),
    ) {
        crate::observe::fail(format!(
            "replacement_object_delete_absent task={} kind={:?} reference={} \
             reason=no_registered_object",
            command.task.get(),
            command.kind,
            command.reference
        ));
    }
    ReplacementObjectDeleteEffect::AbsentObject(command.kind)
}

pub(crate) fn apply_replacement_object_delete<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    command: DecodedReplacementObjectDelete,
) -> Result<ReplacementObjectDeleteEffect, ReplacementObjectDeleteFailure> {
    let fail = |reason| ReplacementObjectDeleteFailure { reason, command };
    let reference = command.reference;
    Ok(match command.kind {
        ReplacementDeletedObjectKind::Buffer | ReplacementDeletedObjectKind::Texture => {
            return Err(fail(
                ReplacementObjectDeleteRefusal::ResourceLifecycleUnresolved(command.kind),
            ));
        }
        ReplacementDeletedObjectKind::RasterizationRateMap => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime
                .resolve_rasterization_rate_map(command.task, serializer)
                .map(|(identity, _)| identity)
            else {
                return Ok(report_absent_object(command));
            };
            let descriptor = runtime
                .retire_rasterization_rate_map(command.task, serializer)
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Object(reason)))?;
            ReplacementObjectDeleteEffect::RasterizationRateMap {
                identity,
                descriptor,
            }
        }
        ReplacementDeletedObjectKind::Sampler => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime
                .resolve_sampler(command.task, serializer)
                .map(|(identity, _)| identity)
            else {
                return Ok(report_absent_object(command));
            };
            let descriptor = runtime
                .retire_sampler(command.task, serializer)
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Object(reason)))?;
            ReplacementObjectDeleteEffect::Sampler {
                identity,
                descriptor,
            }
        }
        ReplacementDeletedObjectKind::DepthStencil => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime
                .resolve_depth_stencil(command.task, serializer)
                .map(|(identity, _)| identity)
            else {
                return Ok(report_absent_object(command));
            };
            let descriptor = runtime
                .retire_depth_stencil(command.task, serializer)
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Object(reason)))?;
            ReplacementObjectDeleteEffect::DepthStencil {
                identity,
                descriptor,
            }
        }
        ReplacementDeletedObjectKind::Function => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime
                .resolve_function(command.task, serializer)
                .map(|(identity, _)| identity)
            else {
                return Ok(report_absent_object(command));
            };
            let function = runtime
                .retire_function(command.task, serializer)
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Object(reason)))?;
            ReplacementObjectDeleteEffect::Function { identity, function }
        }
        ReplacementDeletedObjectKind::RenderPipeline => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime.resolve_render_pipeline(command.task, serializer) else {
                return Ok(report_absent_object(command));
            };
            let waiters = runtime
                .retire_render_pipeline(command.task, serializer)
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Pipeline(reason)))?;
            ReplacementObjectDeleteEffect::RenderPipeline { identity, waiters }
        }
        ReplacementDeletedObjectKind::ComputePipeline => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime.resolve_compute_pipeline(command.task, serializer) else {
                return Ok(report_absent_object(command));
            };
            let waiters = runtime
                .retire_compute_pipeline(command.task, serializer)
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Pipeline(reason)))?;
            ReplacementObjectDeleteEffect::ComputePipeline { identity, waiters }
        }
        ReplacementDeletedObjectKind::Fence => {
            let serializer = SerializerRef::new(reference);
            let Some(identity) = runtime.resolve_fence(command.task, serializer) else {
                return Ok(report_absent_object(command));
            };
            if !runtime.release_fence(command.task, serializer) {
                return Err(fail(ReplacementObjectDeleteRefusal::UnknownObject(
                    command.kind,
                )));
            }
            ReplacementObjectDeleteEffect::Fence(identity)
        }
        ReplacementDeletedObjectKind::Heap => {
            let identity = runtime
                .retire_heap(command.task, SerializerRef::new(reference))
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Object(reason)))?;
            ReplacementObjectDeleteEffect::Heap(identity)
        }
        ReplacementDeletedObjectKind::IndirectCommandBuffer => {
            let retired = runtime
                .retire_indirect_command_buffer(command.task, SerializerRef::new(reference))
                .map_err(|reason| fail(ReplacementObjectDeleteRefusal::Object(reason)))?;
            ReplacementObjectDeleteEffect::IndirectCommandBuffer(retired)
        }
    })
}

fn delete_kind(opcode: u32) -> Option<ReplacementDeletedObjectKind> {
    use reims_vgpu_wire::ops::destroy;
    Some(match opcode {
        destroy::OPCODE_DELETE_BUFFER => ReplacementDeletedObjectKind::Buffer,
        destroy::OPCODE_DELETE_TEXTURE => ReplacementDeletedObjectKind::Texture,
        destroy::OPCODE_DELETE_DEPTH_STENCIL_STATE => ReplacementDeletedObjectKind::DepthStencil,
        destroy::OPCODE_DELETE_SAMPLER_STATE => ReplacementDeletedObjectKind::Sampler,
        destroy::OPCODE_DELETE_FUNCTION => ReplacementDeletedObjectKind::Function,
        destroy::OPCODE_DELETE_COMPUTE_PIPELINE_STATE => {
            ReplacementDeletedObjectKind::ComputePipeline
        }
        destroy::OPCODE_DELETE_RENDER_PIPELINE_STATE => {
            ReplacementDeletedObjectKind::RenderPipeline
        }
        destroy::OPCODE_DELETE_FENCE => ReplacementDeletedObjectKind::Fence,
        destroy::OPCODE_DELETE_HEAP => ReplacementDeletedObjectKind::Heap,
        destroy::OPCODE_DELETE_RASTERIZATION_RATE_MAP => {
            ReplacementDeletedObjectKind::RasterizationRateMap
        }
        destroy::OPCODE_DELETE_INDIRECT_COMMAND_BUFFER => {
            ReplacementDeletedObjectKind::IndirectCommandBuffer
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::st32;

    fn payload(opcode: u32, reference: u32) -> Vec<u8> {
        let mut payload = vec![0; 4 + reims_vgpu_wire::ops::destroy::DELETE_TOTAL_LEN as usize];
        st32(&mut payload[0..], 7);
        st32(&mut payload[4..], opcode);
        st32(
            &mut payload[8..],
            reims_vgpu_wire::ops::destroy::DELETE_TOTAL_LEN,
        );
        st32(&mut payload[12..], reference);
        payload
    }

    #[test]
    fn every_destroy_opcode_retains_its_exact_family() {
        use reims_vgpu_wire::ops::destroy;
        let cases = [
            (
                destroy::OPCODE_DELETE_BUFFER,
                ReplacementDeletedObjectKind::Buffer,
            ),
            (
                destroy::OPCODE_DELETE_TEXTURE,
                ReplacementDeletedObjectKind::Texture,
            ),
            (
                destroy::OPCODE_DELETE_DEPTH_STENCIL_STATE,
                ReplacementDeletedObjectKind::DepthStencil,
            ),
            (
                destroy::OPCODE_DELETE_SAMPLER_STATE,
                ReplacementDeletedObjectKind::Sampler,
            ),
            (
                destroy::OPCODE_DELETE_FUNCTION,
                ReplacementDeletedObjectKind::Function,
            ),
            (
                destroy::OPCODE_DELETE_COMPUTE_PIPELINE_STATE,
                ReplacementDeletedObjectKind::ComputePipeline,
            ),
            (
                destroy::OPCODE_DELETE_RENDER_PIPELINE_STATE,
                ReplacementDeletedObjectKind::RenderPipeline,
            ),
            (
                destroy::OPCODE_DELETE_FENCE,
                ReplacementDeletedObjectKind::Fence,
            ),
            (
                destroy::OPCODE_DELETE_HEAP,
                ReplacementDeletedObjectKind::Heap,
            ),
            (
                destroy::OPCODE_DELETE_RASTERIZATION_RATE_MAP,
                ReplacementDeletedObjectKind::RasterizationRateMap,
            ),
            (
                destroy::OPCODE_DELETE_INDIRECT_COMMAND_BUFFER,
                ReplacementDeletedObjectKind::IndirectCommandBuffer,
            ),
        ];
        for (opcode, kind) in cases {
            assert_eq!(
                decode_replacement_object_delete(&payload(opcode, 19)),
                Ok(DecodedReplacementObjectDelete {
                    task: TaskId::new(7),
                    kind,
                    reference: 19,
                })
            );
        }
    }

    #[test]
    fn malformed_and_unclaimed_records_refuse_without_a_partial_delete() {
        assert_eq!(
            decode_replacement_object_delete(&[]),
            Err(ReplacementObjectDeleteDecodeError::ShortEnvelope)
        );
        assert_eq!(
            decode_replacement_object_delete(&payload(0x3ec, 19)),
            Err(ReplacementObjectDeleteDecodeError::NotDelete { opcode: 0x3ec })
        );
        let mut short = payload(reims_vgpu_wire::ops::destroy::OPCODE_DELETE_HEAP, 19);
        short.pop();
        assert_eq!(
            decode_replacement_object_delete(&short),
            Err(ReplacementObjectDeleteDecodeError::MalformedRecord)
        );
    }
}
