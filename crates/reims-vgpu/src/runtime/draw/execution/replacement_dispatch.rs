//! Composition projection into the backend-neutral replacement render command.
//!
//! This is the sole boundary where retained encoder objects, resolved shader
//! numbering, and the canonical resource graph become one immutable draw.

use super::{resource_plan::DrawResourcePlan, DrawEncodeRequest};
use crate::{model::TaskResource, runtime::Device};
use reims_vgpu_core::{
    AccessMode, BackingRegion, ImageAspect, ImageRegion, RenderAttachmentClear,
    RenderAttachmentRole, RenderBindingClass, RenderBindingView, RenderScissor, RenderViewport,
    ResolvedRenderAttachment, ResolvedRenderDispatch, ResolvedRenderDraw,
    ResolvedRenderNullBinding, ResolvedRenderRasterState, ResolvedRenderResolveAttachment,
    ResolvedRenderResourceBinding, ResolvedRenderSamplerBinding, ResolvedRenderVisibility,
    ResolvedVertexBufferLayout,
};
use reims_vgpu_protocol::{BackingId, RenderStages, ResourceId, ResourceObject, SerializerRef};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderConstructionError {
    PipelineMissing,
    DepthStencilStateMissing,
    ColorResourceMissing(u32),
    MultisampleSourceMissing(u32),
    ResourceIdentityMissing,
    ResourceBackingMissing(ResourceId<ResourceObject>),
    TextureView(reims_vgpu_core::TextureViewResolveError),
    BufferRangeMissing(ResourceId<ResourceObject>),
    VertexBindingMissing(u32),
    StorageBindingMissing(u32),
    SampledBindingMissing(u32),
    StorageImageAccessMissing(u32),
    StorageImageAccessAmbiguous(u32),
    StorageImageBindingCollision(u32),
    IndexTypeUnsupported,
    IndexVertexOffsetOutOfRange,
    IndexRangeOverflow,
    VisibilityBufferMissing,
    VisibilityRangeMissing,
    DepthStencilAttachmentMismatch,
    DepthStencilDescriptorMissing,
    DepthStencilResolveMissing,
    DepthStencilFormatMissing,
    DepthStencilLevelMissing,
    DepthStencilSubresourceOutOfRange,
    EmptyAttachmentExtent,
    Icb(crate::runtime::icb::IcbStatus),
}

struct DepthStencilAttachmentMetadata {
    resource: ResourceId<ResourceObject>,
    backing: BackingId,
    format: u16,
    extent: [u32; 3],
    samples: u32,
    region: BackingRegion,
}

struct BufferBindingRequest {
    class: RenderBindingClass,
    binding: u32,
    stages: RenderStages,
    offset: u64,
    length: u64,
    mode: AccessMode,
}

pub(crate) struct ResolvedRenderConstructionInput {
    pub pipeline: ResourceId<reims_vgpu_protocol::RenderPipelineObject>,
    pub program: reims_vgpu_core::PreparedRenderProgram,
    pub depth_stencil: Option<ResourceId<reims_vgpu_protocol::DepthStencilObject>>,
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

pub(crate) fn construct_from_resolved(
    input: ResolvedRenderConstructionInput,
) -> ResolvedRenderDispatch {
    ResolvedRenderDispatch {
        pipeline: input.pipeline,
        program: input.program,
        depth_stencil: input.depth_stencil,
        render_extent: input.render_extent,
        raster: input.raster,
        visibility: input.visibility,
        begins_encoder: input.begins_encoder,
        ends_encoder: input.ends_encoder,
        draw: input.draw,
        vertex_buffers: input.vertex_buffers,
        attachments: input.attachments,
        resources: input.resources,
        null_bindings: input.null_bindings,
        samplers: input.samplers,
    }
}

pub(crate) fn construct(
    state: &Device,
    request: &DrawEncodeRequest,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
    resources: &DrawResourcePlan,
) -> Result<ResolvedRenderDispatch, ReplacementRenderConstructionError> {
    let pipeline = state
        .task_objects
        .render_pipelines
        .identity(request.task_id, SerializerRef::new(request.pipeline_ref))
        .ok_or(ReplacementRenderConstructionError::PipelineMissing)?;
    let depth_stencil = (request.depth_stencil_ref != 0)
        .then(|| {
            state.task_objects.depth_stencil.identity(
                request.task_id,
                SerializerRef::new(request.depth_stencil_ref),
            )
        })
        .flatten();
    if request.depth_stencil_ref != 0 && depth_stencil.is_none() {
        return Err(ReplacementRenderConstructionError::DepthStencilStateMissing);
    }
    if request
        .depth_attach
        .as_ref()
        .map(|attachment| attachment.texture_ref)
        .zip(
            request
                .stencil_attach
                .as_ref()
                .map(|attachment| attachment.texture_ref),
        )
        .is_some_and(|(depth, stencil)| depth != stencil)
    {
        return Err(ReplacementRenderConstructionError::DepthStencilAttachmentMismatch);
    }

    let mut attachments = request
        .colors
        .iter()
        .map(|color| {
            let (source, resolve) = if color.multisample_source_ref != 0 {
                (
                    color.multisample_source_resource.as_ref().ok_or(
                        ReplacementRenderConstructionError::MultisampleSourceMissing(color.slot),
                    )?,
                    Some(color.resource.as_ref().ok_or(
                        ReplacementRenderConstructionError::ColorResourceMissing(color.slot),
                    )?),
                )
            } else {
                (
                    color.resource.as_ref().ok_or(
                        ReplacementRenderConstructionError::ColorResourceMissing(color.slot),
                    )?,
                    None,
                )
            };
            let (resource, backing) = resource_and_backing(state, source)?;
            let resolve = resolve
                .map(|target| {
                    let (resource, backing) = resource_and_backing(state, target)?;
                    Ok(ResolvedRenderResolveAttachment {
                        resource,
                        backing,
                        regions: Box::new([attachment_region(
                            ImageAspect::Color,
                            color.subresource,
                            [color.width, color.height, 1],
                        )?]),
                        pixel_format: color.format,
                        extent: [color.width, color.height, 1],
                        sample_count: 1,
                    })
                })
                .transpose()?;
            Ok(ResolvedRenderAttachment {
                role: RenderAttachmentRole::Color(color.slot),
                resource,
                backing,
                regions: Box::new([attachment_region(
                    ImageAspect::Color,
                    if color.multisample_source_ref != 0 {
                        color.multisample_source_subresource
                    } else {
                        color.subresource
                    },
                    [color.width, color.height, 1],
                )?]),
                pixel_format: color.format,
                extent: [color.width, color.height, 1],
                sample_count: color.sample_count.max(1),
                load: color.load_action,
                store: color.store_action,
                clear: RenderAttachmentClear::Color(
                    color.clear_color.map(|value| (value as f32).to_bits()),
                ),
                resolve,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(depth) = request.depth_attach.as_ref() {
        let retained = request
            .depth_attachment_resource
            .as_ref()
            .ok_or(ReplacementRenderConstructionError::DepthStencilDescriptorMissing)?;
        let metadata =
            depth_stencil_attachment(state, retained, depth.subresource, ImageAspect::Depth)?;
        let resolve = (depth.resolve_texture_ref != 0)
            .then(|| {
                let retained = request
                    .depth_resolve_resource
                    .as_ref()
                    .ok_or(ReplacementRenderConstructionError::DepthStencilResolveMissing)?;
                let resolve = depth_stencil_attachment(
                    state,
                    retained,
                    depth.resolve_subresource,
                    ImageAspect::Depth,
                )?;
                Ok(ResolvedRenderResolveAttachment {
                    resource: resolve.resource,
                    backing: resolve.backing,
                    regions: Box::new([resolve.region]),
                    pixel_format: resolve.format,
                    extent: resolve.extent,
                    sample_count: resolve.samples,
                })
            })
            .transpose()?;
        attachments.push(ResolvedRenderAttachment {
            role: RenderAttachmentRole::Depth,
            resource: metadata.resource,
            backing: metadata.backing,
            regions: Box::new([metadata.region]),
            pixel_format: metadata.format,
            extent: metadata.extent,
            sample_count: metadata.samples,
            load: depth.load_action,
            store: depth.store_action,
            clear: RenderAttachmentClear::Depth((depth.clear_depth as f32).to_bits()),
            resolve,
        });
    }
    if let Some(stencil) = request.stencil_attach.as_ref() {
        let retained = request
            .stencil_attachment_resource
            .as_ref()
            .ok_or(ReplacementRenderConstructionError::DepthStencilDescriptorMissing)?;
        let metadata =
            depth_stencil_attachment(state, retained, stencil.subresource, ImageAspect::Stencil)?;
        let resolve = (stencil.resolve_texture_ref != 0)
            .then(|| {
                let retained = request
                    .stencil_resolve_resource
                    .as_ref()
                    .ok_or(ReplacementRenderConstructionError::DepthStencilResolveMissing)?;
                let resolve = depth_stencil_attachment(
                    state,
                    retained,
                    stencil.resolve_subresource,
                    ImageAspect::Stencil,
                )?;
                Ok(ResolvedRenderResolveAttachment {
                    resource: resolve.resource,
                    backing: resolve.backing,
                    regions: Box::new([resolve.region]),
                    pixel_format: resolve.format,
                    extent: resolve.extent,
                    sample_count: resolve.samples,
                })
            })
            .transpose()?;
        attachments.push(ResolvedRenderAttachment {
            role: RenderAttachmentRole::Stencil,
            resource: metadata.resource,
            backing: metadata.backing,
            regions: Box::new([metadata.region]),
            pixel_format: metadata.format,
            extent: metadata.extent,
            sample_count: metadata.samples,
            load: stencil.load_action,
            store: stencil.store_action,
            clear: RenderAttachmentClear::Stencil(stencil.clear_stencil),
            resolve,
        });
    }

    let mut bindings = Vec::new();
    let mut null_bindings = Vec::new();
    for attribute in &resources.attributes {
        let declaration = resolved
            .desc
            .vertex_attributes
            .iter()
            .find(|candidate| candidate.location == attribute.location)
            .ok_or(ReplacementRenderConstructionError::VertexBindingMissing(
                attribute.binding,
            ))?;
        let bound = request
            .vertex_buffers
            .iter()
            .find(|bound| bound.index == declaration.buffer_index)
            .ok_or(ReplacementRenderConstructionError::VertexBindingMissing(
                attribute.binding,
            ))?;
        bindings.push(buffer_binding(
            state,
            bound.resource.as_ref(),
            BufferBindingRequest {
                class: RenderBindingClass::VertexBuffer,
                binding: attribute.binding,
                stages: vertex_stages(),
                offset: bound.offset,
                length: attribute.content.len() as u64,
                mode: AccessMode::Read,
            },
        )?);
    }
    for storage in &resources.storage_buffers {
        let (bound, stages, mode) = if let Some(bound) = request
            .vertex_buffers
            .iter()
            .find(|bound| bound.index == storage.binding)
        {
            (
                bound,
                vertex_stages(),
                buffer_access_mode(resolved.vertex.interface.buffer_access(bound.index), true),
            )
        } else {
            let bound = request
                .fragment_buffers
                .iter()
                .find(|bound| {
                    resources.fragment_variant.buffer_binding(bound.index) == storage.binding
                })
                .ok_or(ReplacementRenderConstructionError::StorageBindingMissing(
                    storage.binding,
                ))?;
            (
                bound,
                fragment_stages(),
                buffer_access_mode(
                    resolved.fragment.interface.buffer_access(bound.index),
                    false,
                ),
            )
        };
        bindings.push(buffer_binding(
            state,
            bound.resource.as_ref(),
            BufferBindingRequest {
                class: RenderBindingClass::StorageBuffer,
                binding: storage.binding,
                stages,
                offset: bound.offset,
                length: storage.content.len() as u64,
                mode,
            },
        )?);
    }
    for sampled in &resources.sampled_images {
        if matches!(sampled.source, reims_vgpu_core::SampledSource::Null) {
            null_bindings.push(ResolvedRenderNullBinding {
                class: RenderBindingClass::SampledImage,
                binding: sampled.binding,
                array_element: sampled.array_element,
                descriptor_count: sampled.descriptor_count,
                stages: stages(
                    resources
                        .vertex_variant
                        .declares_descriptor(sampled.binding),
                    resources
                        .fragment_variant
                        .declares_descriptor(sampled.binding),
                ),
            });
            continue;
        }
        let (bound, stages) = texture_binding(request, resolved, resources, sampled.binding)
            .ok_or(ReplacementRenderConstructionError::SampledBindingMissing(
                sampled.binding,
            ))?;
        let (resource, backing) = resource_and_backing(
            state,
            bound
                .resource
                .as_ref()
                .ok_or(ReplacementRenderConstructionError::ResourceIdentityMissing)?,
        )?;
        let view = state
            .task_objects
            .resources
            .texture_binding_view(resource)
            .map_err(ReplacementRenderConstructionError::TextureView)?;
        bindings.push(ResolvedRenderResourceBinding {
            class: RenderBindingClass::SampledImage,
            binding: sampled.binding,
            array_element: sampled.array_element,
            descriptor_count: sampled.descriptor_count,
            stages,
            resource,
            backing,
            view: RenderBindingView::Image(view),
            regions: Box::new([BackingRegion::Whole]),
            mode: AccessMode::Read,
        });
    }
    for (bound_table, interface, variant, stage) in [
        (
            request.vertex_textures.as_ref(),
            &resolved.vertex.interface,
            &resources.vertex_variant,
            vertex_stages(),
        ),
        (
            request.fragment_textures.as_ref(),
            &resolved.fragment.interface,
            &resources.fragment_variant,
            fragment_stages(),
        ),
    ] {
        for bound in bound_table {
            let Some(descriptor) = interface.texture_descriptor(bound.index) else {
                continue;
            };
            if descriptor.access != reims_vgpu_core::ReflectedTextureAccess::Storage {
                continue;
            }
            let binding = variant.texture_binding(bound.index, Some(descriptor.binding));
            let access = variant
                .storage_image_access(binding)
                .ok_or(ReplacementRenderConstructionError::StorageImageAccessMissing(binding))?;
            let mode = match access {
                reims_vgpu_core::StorageImageAccess::ReadOnly => AccessMode::Read,
                reims_vgpu_core::StorageImageAccess::WriteOnly => AccessMode::Write,
                reims_vgpu_core::StorageImageAccess::ReadWrite => AccessMode::ReadWrite,
                reims_vgpu_core::StorageImageAccess::Unknown => AccessMode::Unknown,
                reims_vgpu_core::StorageImageAccess::AmbiguousBinding => {
                    return Err(
                        ReplacementRenderConstructionError::StorageImageAccessAmbiguous(binding),
                    )
                }
            };
            if bound.texture_ref == 0 {
                push_null_binding(
                    &mut null_bindings,
                    ResolvedRenderNullBinding {
                        class: RenderBindingClass::StorageImage,
                        binding,
                        array_element: descriptor.array_element,
                        descriptor_count: descriptor.descriptor_count,
                        stages: stage,
                    },
                )?;
                continue;
            }
            let (resource, backing) = resource_and_backing(
                state,
                bound
                    .resource
                    .as_ref()
                    .ok_or(ReplacementRenderConstructionError::ResourceIdentityMissing)?,
            )?;
            let view = state
                .task_objects
                .resources
                .texture_binding_view(resource)
                .map_err(ReplacementRenderConstructionError::TextureView)?;
            push_storage_image_binding(
                &mut bindings,
                ResolvedRenderResourceBinding {
                    class: RenderBindingClass::StorageImage,
                    binding,
                    array_element: descriptor.array_element,
                    descriptor_count: descriptor.descriptor_count,
                    stages: stage,
                    resource,
                    backing,
                    view: RenderBindingView::Image(view),
                    regions: Box::new([BackingRegion::Whole]),
                    mode,
                },
            )?;
        }
    }

    let draw = if let Some(indexed) = &request.indexed {
        let index_type = indexed
            .index_type
            .map_err(|_| ReplacementRenderConstructionError::IndexTypeUnsupported)?;
        let width = u64::try_from(index_type.byte_size())
            .map_err(|_| ReplacementRenderConstructionError::IndexRangeOverflow)?;
        let offset = indexed
            .index_buffer_offset
            .checked_add(
                u64::from(indexed.index_start)
                    .checked_mul(width)
                    .ok_or(ReplacementRenderConstructionError::IndexRangeOverflow)?,
            )
            .ok_or(ReplacementRenderConstructionError::IndexRangeOverflow)?;
        let length = u64::from(indexed.index_count)
            .checked_mul(width)
            .ok_or(ReplacementRenderConstructionError::IndexRangeOverflow)?;
        bindings.push(buffer_binding(
            state,
            indexed.index_buffer_resource.as_ref(),
            BufferBindingRequest {
                class: RenderBindingClass::IndexBuffer,
                binding: 0,
                stages: vertex_stages(),
                offset,
                length,
                mode: AccessMode::Read,
            },
        )?);
        ResolvedRenderDraw::Indexed {
            topology: request.primitive_topology,
            index_type,
            index_count: indexed.index_count,
            instance_count: request.instance_count,
            first_index: 0,
            vertex_offset: i32::try_from(indexed.base_vertex)
                .map_err(|_| ReplacementRenderConstructionError::IndexVertexOffsetOutOfRange)?,
            first_instance: request.base_instance,
        }
    } else {
        ResolvedRenderDraw::Direct {
            topology: request.primitive_topology,
            vertex_count: request.vertex_count,
            instance_count: request.instance_count,
            first_vertex: request.first_vertex,
            first_instance: request.base_instance,
        }
    };

    let visibility = request
        .visibility
        .map(|arming| {
            let retained = request
                .visibility_buffer_resource
                .as_ref()
                .ok_or(ReplacementRenderConstructionError::VisibilityBufferMissing)?;
            let (resource, _) = resource_and_backing(state, retained)?;
            let (backing, range) = state
                .task_objects
                .resources
                .linear_backing_region(resource, arming.offset, u64::from(u64::BITS / 8))
                .ok_or(ReplacementRenderConstructionError::VisibilityRangeMissing)?;
            Ok(ResolvedRenderVisibility {
                mode: arming.mode,
                resource,
                backing,
                range,
            })
        })
        .transpose()?;

    let sampler_stages = |binding| {
        let vertex = resources.vertex_variant.declares_descriptor(binding);
        let fragment = resources.fragment_variant.declares_descriptor(binding);
        stages(vertex, fragment)
    };
    let samplers = resources
        .samplers
        .iter()
        .cloned()
        .map(|sampler| ResolvedRenderSamplerBinding {
            binding: sampler.binding,
            array_element: 0,
            descriptor_count: 1,
            stages: sampler_stages(sampler.binding),
            sampler,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let minimum_width = attachments
        .iter()
        .map(|attachment| attachment.extent[0])
        .min()
        .unwrap_or(0);
    let minimum_height = attachments
        .iter()
        .map(|attachment| attachment.extent[1])
        .min()
        .unwrap_or(0);
    let mut vertex_buffers = Vec::<ResolvedVertexBufferLayout>::new();
    for attribute in &resources.attributes {
        if let Some(existing) = vertex_buffers
            .iter()
            .find(|layout| layout.binding == attribute.binding)
        {
            if existing.stride != attribute.stride {
                return Err(ReplacementRenderConstructionError::VertexBindingMissing(
                    attribute.binding,
                ));
            }
        } else {
            vertex_buffers.push(ResolvedVertexBufferLayout {
                binding: attribute.binding,
                stride: attribute.stride,
            });
        }
    }
    vertex_buffers.sort_unstable();

    Ok(construct_from_resolved(ResolvedRenderConstructionInput {
        pipeline,
        program: reims_vgpu_core::PreparedRenderProgram {
            vertex: resources.vertex_variant.program.clone(),
            fragment: resources.fragment_variant.program.clone(),
        },
        depth_stencil,
        render_extent: [
            request.render_target_extent.raster_width(minimum_width),
            request.render_target_extent.raster_height(minimum_height),
        ],
        raster: ResolvedRenderRasterState {
            viewports: request
                .viewports
                .iter()
                .copied()
                .map(RenderViewport::from_values)
                .collect(),
            scissors: request
                .scissors
                .iter()
                .map(|scissor| RenderScissor {
                    x: scissor.x,
                    y: scissor.y,
                    width: scissor.width,
                    height: scissor.height,
                })
                .collect(),
            cull_mode: request.cull_mode,
            front_face_ccw: request.front_face_ccw,
            fill_mode: request.fill_mode,
            line_width_bits: request.line_width.bits(),
            depth_clip_mode: request.depth_clip_mode,
            depth_bias_bits: request.depth_bias.map(|values| values.map(f32::to_bits)),
            blend_color_bits: request.blend_color.map(|values| values.map(f32::to_bits)),
            stencil_reference: request.stencil_ref.unwrap_or((0, 0)).into(),
        },
        visibility,
        begins_encoder: !request.continues_render_pass,
        ends_encoder: !request.render_pass_continues,
        draw,
        vertex_buffers: vertex_buffers.into_boxed_slice(),
        attachments: attachments.into_boxed_slice(),
        resources: bindings.into_boxed_slice(),
        null_bindings: null_bindings.into_boxed_slice(),
        samplers,
    }))
}

/// Resolve one render ICB slot against the immutable render-encoder snapshot
/// captured at its execute record, then enter the direct draw projector.
pub(crate) fn construct_icb<M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps>(
    state: &Device,
    host: &M,
    descriptor: &reims_vgpu_protocol::IndirectCommandBufferDescriptor,
    inherited: &DrawEncodeRequest,
    fill: &crate::runtime::icb::IcbRenderFill,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
    resources: &DrawResourcePlan,
) -> Result<Option<ResolvedRenderDispatch>, ReplacementRenderConstructionError> {
    use crate::runtime::icb::{IcbRenderBindStage, IcbRenderDraw, IcbStatus};

    let icb = |reason| ReplacementRenderConstructionError::Icb(reason);
    if !fill.object_threadgroup_memory.is_empty() {
        return Err(icb(IcbStatus::Unsupported(
            "render_icb_object_threadgroup_memory_unimplemented",
        )));
    }
    let mut request = inherited.clone();
    if descriptor.inherit_pipeline_state() {
        if fill.pipeline_ref != 0 {
            return Err(icb(IcbStatus::Args(
                "render_icb_inherited_pipeline_ref_nonzero",
            )));
        }
    } else {
        request.pipeline_ref = fill.pipeline_ref;
    }
    if request.pipeline_ref == 0 {
        return Err(icb(IcbStatus::Missing("render_icb_pipeline_ref_zero")));
    }

    if descriptor.inherit_buffers() {
        if !fill.buffers.is_empty() {
            return Err(icb(IcbStatus::Args(
                "render_icb_inherited_buffers_nonempty",
            )));
        }
    } else {
        let mut vertex = Vec::new();
        let mut fragment = Vec::new();
        for binding in &fill.buffers {
            let target = match binding.effective_stage() {
                IcbRenderBindStage::Vertex => &mut vertex,
                IcbRenderBindStage::Fragment => &mut fragment,
                IcbRenderBindStage::Object | IcbRenderBindStage::Mesh => {
                    return Err(icb(IcbStatus::Unsupported(
                        "render_icb_object_mesh_buffer_unimplemented",
                    )));
                }
            };
            target.push(super::super::BufferBind {
                index: binding.index,
                buffer_ref: binding.buffer_ref,
                resource: crate::runtime::objects::resolve_resource(
                    state,
                    host,
                    inherited.task_id,
                    binding.buffer_ref,
                )
                .ok(),
                offset: binding.offset,
                attribute_stride: binding
                    .has_attribute_stride
                    .then_some(binding.attribute_stride),
            });
        }
        request.vertex_buffers = Arc::new(vertex);
        request.fragment_buffers = Arc::new(fragment);
    }

    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            if vertex_count == 0 || instance_count == 0 {
                return Ok(None);
            }
            request.vertex_count = u32::try_from(vertex_count)
                .map_err(|_| icb(IcbStatus::Args("render_icb_vertex_count_too_wide")))?;
            request.instance_count = u32::try_from(instance_count)
                .map_err(|_| icb(IcbStatus::Args("render_icb_instance_count_too_wide")))?;
            request.primitive_topology =
                reims_vgpu_protocol::primitive_topology(u32::from(primitive_type))
                    .map_err(|_| icb(IcbStatus::Args("render_icb_primitive_type_invalid")))?;
            request.first_vertex = u32::try_from(vertex_start)
                .map_err(|_| icb(IcbStatus::Args("render_icb_vertex_start_too_wide")))?;
            request.base_instance = u32::try_from(base_instance)
                .map_err(|_| icb(IcbStatus::Args("render_icb_base_instance_too_wide")))?;
            request.indexed = None;
        }
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_buffer_ref,
            index_count,
            index_buffer_offset,
            instance_count,
            base_vertex,
            base_instance,
            ..
        } => {
            if index_count == 0 || instance_count == 0 {
                return Ok(None);
            }
            let count = u32::try_from(index_count)
                .map_err(|_| icb(IcbStatus::Args("render_icb_index_count_too_wide")))?;
            request.vertex_count = count;
            request.instance_count = u32::try_from(instance_count)
                .map_err(|_| icb(IcbStatus::Args("render_icb_instance_count_too_wide")))?;
            request.primitive_topology =
                reims_vgpu_protocol::primitive_topology(u32::from(primitive_type))
                    .map_err(|_| icb(IcbStatus::Args("render_icb_primitive_type_invalid")))?;
            request.first_vertex = 0;
            request.base_instance = u32::try_from(base_instance)
                .map_err(|_| icb(IcbStatus::Args("render_icb_base_instance_too_wide")))?;
            request.indexed = Some(super::super::IndexedDrawInfo {
                index_type: reims_vgpu_protocol::decode_index_type(u32::from(index_type)),
                index_count: count,
                index_buffer_ref,
                index_buffer_resource: crate::runtime::objects::resolve_resource(
                    state,
                    host,
                    inherited.task_id,
                    index_buffer_ref,
                )
                .ok(),
                index_buffer_offset,
                index_start: 0,
                base_vertex,
            });
        }
        IcbRenderDraw::Patches { .. } | IcbRenderDraw::IndexedPatches { .. } => {
            return Err(icb(IcbStatus::Unsupported(
                "render_icb_tessellation_draw_unimplemented",
            )));
        }
        IcbRenderDraw::MeshThreads(_) | IcbRenderDraw::MeshThreadgroups(_) => {
            return Err(icb(IcbStatus::Unsupported(
                "render_icb_mesh_draw_unimplemented",
            )));
        }
    }
    construct(state, &request, resolved, resources).map(Some)
}

fn depth_stencil_attachment(
    state: &Device,
    retained: &Arc<TaskResource>,
    subresource: super::RenderAttachmentSubresource,
    aspect: ImageAspect,
) -> Result<DepthStencilAttachmentMetadata, ReplacementRenderConstructionError> {
    let (resource, backing) = resource_and_backing(state, retained)?;
    let reims_vgpu_protocol::ResourceDescriptor::Texture(descriptor) =
        crate::runtime::objects::decoded_resource(retained)
            .as_ref()
            .map_err(|_| ReplacementRenderConstructionError::DepthStencilDescriptorMissing)?
    else {
        return Err(ReplacementRenderConstructionError::DepthStencilDescriptorMissing);
    };
    let declaration = descriptor
        .declaration
        .as_ref()
        .ok_or(ReplacementRenderConstructionError::DepthStencilFormatMissing)?;
    let format = declaration
        .declared_pixel_format()
        .ok_or(ReplacementRenderConstructionError::DepthStencilFormatMissing)?;
    let level = descriptor
        .level(subresource.level)
        .ok_or(ReplacementRenderConstructionError::DepthStencilLevelMissing)?;
    let is_volume = level.planes() > 1;
    if (is_volume && (subresource.slice != 0 || subresource.depth_plane >= level.planes()))
        || (!is_volume
            && (subresource.depth_plane != 0
                || descriptor
                    .physical_slice_count()
                    .is_none_or(|layers| subresource.slice >= layers)))
    {
        return Err(ReplacementRenderConstructionError::DepthStencilSubresourceOutOfRange);
    }
    let extent = [level.width, level.height, 1];
    Ok(DepthStencilAttachmentMetadata {
        resource,
        backing,
        format,
        extent,
        samples: u32::from(declaration.sample_count).max(1),
        region: attachment_region(aspect, subresource, extent)?,
    })
}

fn attachment_region(
    aspect: ImageAspect,
    subresource: super::RenderAttachmentSubresource,
    extent: [u32; 3],
) -> Result<BackingRegion, ReplacementRenderConstructionError> {
    let texels = reims_vgpu_core::TexelBox::new([0, 0, subresource.depth_plane], extent)
        .ok_or(ReplacementRenderConstructionError::EmptyAttachmentExtent)?;
    Ok(BackingRegion::Image(ImageRegion {
        aspect,
        mip: subresource.level,
        layer: subresource.slice,
        texels,
    }))
}

fn resource_and_backing(
    state: &Device,
    resource: &Arc<TaskResource>,
) -> Result<(ResourceId<ResourceObject>, BackingId), ReplacementRenderConstructionError> {
    let resource = resource
        .semantic_id()
        .ok_or(ReplacementRenderConstructionError::ResourceIdentityMissing)?;
    let backing = state.task_objects.resources.backing(resource).ok_or(
        ReplacementRenderConstructionError::ResourceBackingMissing(resource),
    )?;
    Ok((resource, backing))
}

fn buffer_binding(
    state: &Device,
    retained: Option<&Arc<TaskResource>>,
    request: BufferBindingRequest,
) -> Result<ResolvedRenderResourceBinding, ReplacementRenderConstructionError> {
    let retained = retained.ok_or(ReplacementRenderConstructionError::ResourceIdentityMissing)?;
    let resource = retained
        .semantic_id()
        .ok_or(ReplacementRenderConstructionError::ResourceIdentityMissing)?;
    let (backing, range) = state
        .task_objects
        .resources
        .linear_backing_region(resource, request.offset, request.length)
        .ok_or(ReplacementRenderConstructionError::BufferRangeMissing(
            resource,
        ))?;
    Ok(ResolvedRenderResourceBinding {
        class: request.class,
        binding: request.binding,
        array_element: 0,
        descriptor_count: 1,
        stages: request.stages,
        resource,
        backing,
        view: RenderBindingView::Buffer(range),
        regions: Box::new([BackingRegion::Linear(range)]),
        mode: request.mode,
    })
}

fn texture_binding<'a>(
    request: &'a DrawEncodeRequest,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
    resources: &DrawResourcePlan,
    binding: u32,
) -> Option<(&'a super::TextureBind, RenderStages)> {
    for (bound, interface, variant, stages) in [
        (
            request.vertex_textures.as_ref(),
            &resolved.vertex.interface,
            &resources.vertex_variant,
            vertex_stages(),
        ),
        (
            request.fragment_textures.as_ref(),
            &resolved.fragment.interface,
            &resources.fragment_variant,
            fragment_stages(),
        ),
    ] {
        if let Some(bound) = bound.iter().find(|bound| {
            interface
                .texture_descriptor(bound.index)
                .is_some_and(|descriptor| {
                    variant.texture_binding(bound.index, Some(descriptor.binding)) == binding
                })
        }) {
            return Some((bound, stages));
        }
    }
    None
}

fn stages(vertex: bool, fragment: bool) -> RenderStages {
    let bits = (if vertex { RenderStages::VERTEX } else { 0 })
        | (if fragment { RenderStages::FRAGMENT } else { 0 });
    RenderStages::from_bits(bits.into()).expect("the two render-stage bits are valid")
}

fn merge_stages(left: RenderStages, right: RenderStages) -> RenderStages {
    RenderStages::from_bits((left.bits() | right.bits()).into())
        .expect("the union of valid render-stage bits is valid")
}

fn merge_access(left: AccessMode, right: AccessMode) -> AccessMode {
    match (left, right) {
        (AccessMode::Unknown, _) | (_, AccessMode::Unknown) => AccessMode::Unknown,
        (AccessMode::ReadWrite, _) | (_, AccessMode::ReadWrite) => AccessMode::ReadWrite,
        (AccessMode::Read, AccessMode::Write) | (AccessMode::Write, AccessMode::Read) => {
            AccessMode::ReadWrite
        }
        (left, _) => left,
    }
}

fn push_storage_image_binding(
    bindings: &mut Vec<ResolvedRenderResourceBinding>,
    binding: ResolvedRenderResourceBinding,
) -> Result<(), ReplacementRenderConstructionError> {
    let Some(existing) = bindings.iter_mut().find(|existing| {
        existing.class == binding.class
            && existing.binding == binding.binding
            && existing.array_element == binding.array_element
    }) else {
        bindings.push(binding);
        return Ok(());
    };
    if existing.descriptor_count != binding.descriptor_count
        || existing.resource != binding.resource
        || existing.backing != binding.backing
        || existing.view != binding.view
        || existing.regions != binding.regions
    {
        return Err(
            ReplacementRenderConstructionError::StorageImageBindingCollision(binding.binding),
        );
    }
    existing.stages = merge_stages(existing.stages, binding.stages);
    existing.mode = merge_access(existing.mode, binding.mode);
    Ok(())
}

fn push_null_binding(
    bindings: &mut Vec<ResolvedRenderNullBinding>,
    binding: ResolvedRenderNullBinding,
) -> Result<(), ReplacementRenderConstructionError> {
    let Some(existing) = bindings.iter_mut().find(|existing| {
        existing.class == binding.class
            && existing.binding == binding.binding
            && existing.array_element == binding.array_element
    }) else {
        bindings.push(binding);
        return Ok(());
    };
    if existing.descriptor_count != binding.descriptor_count {
        return Err(
            ReplacementRenderConstructionError::StorageImageBindingCollision(binding.binding),
        );
    }
    existing.stages = merge_stages(existing.stages, binding.stages);
    Ok(())
}

fn vertex_stages() -> RenderStages {
    stages(true, false)
}

fn fragment_stages() -> RenderStages {
    stages(false, true)
}

fn buffer_access_mode(
    access: reims_vgpu_core::ReflectedBufferAccess,
    stage_in_fallback: bool,
) -> AccessMode {
    match access {
        reims_vgpu_core::ReflectedBufferAccess::ReadOnly => AccessMode::Read,
        reims_vgpu_core::ReflectedBufferAccess::Writable => AccessMode::ReadWrite,
        reims_vgpu_core::ReflectedBufferAccess::Unknown => AccessMode::Unknown,
        reims_vgpu_core::ReflectedBufferAccess::Unused
        | reims_vgpu_core::ReflectedBufferAccess::Absent => {
            if stage_in_fallback {
                AccessMode::Read
            } else {
                AccessMode::Unknown
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::draw::{ColorRtRequest, TextureBind};
    use reims_vgpu_protocol::{ObjectKind, ObjectListEntry};

    fn resource(state: &Device, task: u32, reference: u32, address: u64) -> Arc<TaskResource> {
        let resource = Arc::new(TaskResource::new_decoded(
            ObjectListEntry::new(ObjectKind::Texture, 0, 0),
            reims_vgpu_protocol::ResourceDescriptor::Texture(
                reims_vgpu_protocol::LinearTextureDescriptor {
                    allocation_size: 0x4000,
                    declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                        texture_type: reims_vgpu_protocol::TextureType::D2,
                        framebuffer_only: false,
                        is_drawable: false,
                        write_swizzle_enabled: None,
                        allow_gpu_optimized_contents: false,
                        usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ
                            | reims_vgpu_protocol::TEXTURE_USAGE_SHADER_WRITE
                            | reims_vgpu_protocol::TEXTURE_USAGE_RENDER_TARGET,
                        pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                        width: 32,
                        height: 16,
                        depth: 1,
                        mipmap_level_count: 1,
                        sample_count: 1,
                        array_length: 1,
                        resource_options: 0,
                        protection_options: 0,
                        swizzle: None,
                    }),
                    ..Default::default()
                },
            ),
        ));
        let resource = state
            .task_objects
            .resources
            .register(task, reference, resource);
        assert!(state
            .task_objects
            .resources
            .attach_task_address(task, reference, address, 0x4000));
        resource
    }

    fn depth_resource(
        state: &Device,
        task: u32,
        reference: u32,
        address: u64,
        samples: u16,
    ) -> Arc<TaskResource> {
        let descriptor = reims_vgpu_protocol::LinearTextureDescriptor {
            allocation_size: 0x4000,
            handle: 1,
            mipmap_level_count: 2,
            slice_count: 4,
            width: 32,
            height: 16,
            depth: 1,
            declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                texture_type: reims_vgpu_protocol::TextureType::D2Array,
                framebuffer_only: false,
                is_drawable: false,
                write_swizzle_enabled: None,
                allow_gpu_optimized_contents: false,
                usage: 0,
                pixel_format: 252,
                width: 32,
                height: 16,
                depth: 1,
                mipmap_level_count: 2,
                sample_count: samples,
                array_length: 4,
                resource_options: 0,
                protection_options: 0,
                swizzle: None,
            }),
            levels: vec![
                reims_vgpu_protocol::TextureLevelLayout {
                    width: 32,
                    height: 16,
                    depth: 1,
                    ..Default::default()
                },
                reims_vgpu_protocol::TextureLevelLayout {
                    width: 16,
                    height: 8,
                    depth: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let resource = Arc::new(TaskResource::new_decoded(
            ObjectListEntry::new(ObjectKind::Texture, 0, 0),
            reims_vgpu_protocol::ResourceDescriptor::Texture(descriptor),
        ));
        let resource = state
            .task_objects
            .resources
            .register(task, reference, resource);
        assert!(state
            .task_objects
            .resources
            .attach_task_address(task, reference, address, 0x4000));
        resource
    }

    fn setup() -> (Device, Arc<reims_vgpu_core::ResolvedRenderPipeline>) {
        let state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        let pipeline = crate::runtime::pipeline_resolve::retained_pipeline_for_test();
        state.task_objects.render_pipelines.register(
            7,
            SerializerRef::new(9),
            Arc::clone(&pipeline),
        );
        (state, pipeline)
    }

    fn plan(pipeline: &reims_vgpu_core::ResolvedRenderPipeline) -> DrawResourcePlan {
        DrawResourcePlan {
            attributes: Vec::new(),
            storage_buffers: Vec::new(),
            sampled_images: Vec::new(),
            samplers: Vec::new(),
            sampler_provenance: Default::default(),
            vertex_variant: pipeline.vertex.variant().clone(),
            fragment_variant: pipeline.fragment.variant().clone(),
            fragment_color_input: false,
        }
    }

    #[test]
    fn constructed_attachment_identity_survives_later_slot_reuse() {
        let (mut state, pipeline) = setup();
        let first = resource(&state, 7, 11, 0x10_0000);
        let first_id = first.semantic_id().unwrap();
        let request = DrawEncodeRequest {
            task_id: 7,
            pipeline_ref: 9,
            vertex_count: 3,
            instance_count: 1,
            colors: vec![ColorRtRequest {
                slot: 0,
                texture_ref: 11,
                resource: Some(first),
                subresource: super::super::RenderAttachmentSubresource {
                    level: 2,
                    slice: 3,
                    depth_plane: 4,
                },
                width: 32,
                height: 16,
                format: 80,
                sample_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let operation = construct(&state, &request, &pipeline, &plan(&pipeline)).unwrap();

        assert!(state.delete_object_transition(7, 11).semantic_removed);
        let replacement = resource(&state, 7, 11, 0x20_0000);
        assert_ne!(replacement.semantic_id(), Some(first_id));
        assert_eq!(operation.attachments[0].resource, first_id);
        let BackingRegion::Image(region) = operation.attachments[0].regions[0] else {
            panic!("the decoded attachment subresource must remain exact")
        };
        assert_eq!(
            (region.mip, region.layer, region.texels.origin[2]),
            (2, 3, 4)
        );
    }

    #[test]
    fn multisample_source_and_resolve_destination_remain_distinct() {
        let (state, pipeline) = setup();
        let source = resource(&state, 7, 12, 0x30_0000);
        let destination = resource(&state, 7, 13, 0x40_0000);
        let source_id = source.semantic_id().unwrap();
        let destination_id = destination.semantic_id().unwrap();
        let request = DrawEncodeRequest {
            task_id: 7,
            pipeline_ref: 9,
            vertex_count: 3,
            instance_count: 1,
            colors: vec![ColorRtRequest {
                slot: 0,
                texture_ref: 13,
                resource: Some(destination),
                width: 32,
                height: 16,
                format: 80,
                sample_count: 4,
                subresource: super::super::RenderAttachmentSubresource {
                    level: 0,
                    slice: 3,
                    depth_plane: 0,
                },
                multisample_source_ref: 12,
                multisample_source_subresource: super::super::RenderAttachmentSubresource {
                    level: 1,
                    slice: 2,
                    depth_plane: 0,
                },
                multisample_source_resource: Some(source),
                store_action: reims_vgpu_protocol::StoreAction::MultisampleResolve,
                ..Default::default()
            }],
            ..Default::default()
        };

        let operation = construct(&state, &request, &pipeline, &plan(&pipeline)).unwrap();
        assert_eq!(operation.attachments[0].resource, source_id);
        let BackingRegion::Image(source_region) = operation.attachments[0].regions[0] else {
            panic!("multisample source must retain its image subresource")
        };
        assert_eq!(
            operation.attachments[0].resolve.as_ref().unwrap().resource,
            destination_id
        );
        let BackingRegion::Image(resolve_region) =
            operation.attachments[0].resolve.as_ref().unwrap().regions[0]
        else {
            panic!("resolve destination must retain its image subresource")
        };
        assert_eq!((source_region.mip, source_region.layer), (1, 2));
        assert_eq!((resolve_region.mip, resolve_region.layer), (0, 3));
    }

    #[test]
    fn render_icb_inherits_encoder_state_and_projects_the_slot_draw_exactly() {
        let (state, pipeline) = setup();
        let target = resource(&state, 7, 11, 0x10_0000);
        let host = crate::runtime::host::FakeHost::new();
        let descriptor = reims_vgpu_protocol::IndirectCommandBufferDescriptor {
            flags: 3,
            ..Default::default()
        };
        let inherited = DrawEncodeRequest {
            task_id: 7,
            pipeline_ref: 9,
            colors: vec![ColorRtRequest {
                slot: 0,
                texture_ref: 11,
                resource: Some(target),
                width: 32,
                height: 16,
                format: 80,
                sample_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let fill = crate::runtime::icb::IcbRenderFill {
            command_index: 4,
            pipeline_ref: 0,
            buffers: Vec::new(),
            object_threadgroup_memory: Vec::new(),
            draw: crate::runtime::icb::IcbRenderDraw::Primitives {
                primitive_type: 3,
                vertex_start: 5,
                vertex_count: 7,
                instance_count: 2,
                base_instance: 11,
            },
        };
        let resources = plan(&pipeline);
        let icb = ResourceId::new(14, 1);
        let mut owner = reims_vgpu_core::IndirectCommandSlotOwner::default();
        owner.register(icb, 8).unwrap();
        crate::runtime::icb::populate_resolved_replacement_icb(
            &mut owner,
            icb,
            &inherited,
            vec![crate::runtime::icb::DecodedIcbCommandSlot {
                command_index: 4,
                command: Some(crate::runtime::icb::IcbCommandFill::Render(fill)),
            }],
            |inherited, fill| {
                construct_icb(
                    &state,
                    &host,
                    &descriptor,
                    inherited,
                    fill,
                    &pipeline,
                    &resources,
                )
            },
            |_, _| unreachable!("render ICB cannot resolve as compute"),
        )
        .expect("publish canonical render ICB slot");
        let Some(reims_vgpu_core::ResolvedIndirectCommandSlot::Render(operation)) =
            owner.get(icb, 4).unwrap()
        else {
            panic!("canonical render slot")
        };
        assert_eq!(
            operation.pipeline,
            state
                .task_objects
                .render_pipelines
                .identity(7, SerializerRef::new(9),)
                .unwrap()
        );
        assert_eq!(
            operation.draw,
            ResolvedRenderDraw::Direct {
                topology: reims_vgpu_protocol::PrimitiveTopology::Triangle,
                vertex_count: 7,
                instance_count: 2,
                first_vertex: 5,
                first_instance: 11,
            }
        );
    }

    #[test]
    fn render_storage_image_keeps_exact_binding_access_and_resource_generation() {
        use reims_vgpu_core::{
            PreparedShaderFamily, ShaderDescriptorLocation, ShaderResourceAccess,
            ShaderResourceBinding, ShaderResourceKind, ShaderTextureComponent,
            ShaderTextureDimension, ShaderTextureShape, StorageImageAccess,
        };

        let state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        let mut pipeline = crate::runtime::pipeline_resolve::retained_pipeline_for_test();
        let pipeline_mut = Arc::get_mut(&mut pipeline).expect("new test pipeline is uniquely held");
        let mut variant = pipeline_mut.fragment.variant().clone();
        variant.declared_bindings = Arc::from([37]);
        variant.storage_image_accesses = Arc::from([(37, StorageImageAccess::WriteOnly)]);
        pipeline_mut.fragment = PreparedShaderFamily::new(
            Arc::new(reims_vgpu_core::ShaderInterface {
                stage: reims_vgpu_core::ReflectedShaderStage::Fragment,
                bindings: vec![ShaderResourceBinding {
                    kind: ShaderResourceKind::StorageImage,
                    metal_index: 4,
                    descriptor: Some(ShaderDescriptorLocation {
                        set: 0,
                        binding: 37,
                        count: 1,
                    }),
                    extent: None,
                    footprint: None,
                    texture_shape: Some(ShaderTextureShape {
                        dimension: ShaderTextureDimension::D2,
                        arrayed: false,
                        multisampled: false,
                        component: ShaderTextureComponent::Float,
                        writable: true,
                        array_ref: false,
                        array_length: None,
                        storage_format: Some(reims_vgpu_protocol::StorageImageFormat::Rgba8Unorm),
                    }),
                    access: Some(ShaderResourceAccess::Storage),
                }],
                local_size: None,
                unsupported: None,
            }),
            variant,
        );
        state.task_objects.render_pipelines.register(
            7,
            SerializerRef::new(9),
            Arc::clone(&pipeline),
        );
        let image = resource(&state, 7, 12, 0x50_0000);
        let image_id = image.semantic_id().unwrap();
        let request = DrawEncodeRequest {
            task_id: 7,
            pipeline_ref: 9,
            vertex_count: 3,
            instance_count: 1,
            fragment_textures: Arc::new(vec![TextureBind {
                index: 4,
                texture_ref: 12,
                resource: Some(image),
            }]),
            ..Default::default()
        };

        let operation = construct(&state, &request, &pipeline, &plan(&pipeline)).unwrap();
        let storage = operation
            .resources
            .iter()
            .find(|binding| binding.class == RenderBindingClass::StorageImage)
            .expect("storage image is projected");
        assert_eq!((storage.binding, storage.array_element), (37, 0));
        assert_eq!(storage.mode, AccessMode::Write);
        assert_eq!(storage.stages, fragment_stages());
        assert_eq!(storage.resource, image_id);
    }

    #[test]
    fn depth_subresource_and_resolve_destination_are_projected_independently() {
        let (state, pipeline) = setup();
        let source = depth_resource(&state, 7, 20, 0x60_0000, 4);
        let resolve = depth_resource(&state, 7, 21, 0x70_0000, 1);
        let source_id = source.semantic_id().unwrap();
        let resolve_id = resolve.semantic_id().unwrap();
        let request = DrawEncodeRequest {
            task_id: 7,
            pipeline_ref: 9,
            vertex_count: 3,
            instance_count: 1,
            depth_attach: Some(super::super::DepthAttachmentState {
                texture_ref: 20,
                resolve_texture_ref: 21,
                subresource: super::super::RenderAttachmentSubresource {
                    level: 1,
                    slice: 2,
                    depth_plane: 0,
                },
                store_action: reims_vgpu_protocol::StoreAction::MultisampleResolve,
                ..Default::default()
            }),
            depth_attachment_resource: Some(source),
            depth_resolve_resource: Some(resolve),
            ..Default::default()
        };

        let operation = construct(&state, &request, &pipeline, &plan(&pipeline)).unwrap();
        let depth = operation
            .attachments
            .iter()
            .find(|attachment| attachment.role == RenderAttachmentRole::Depth)
            .unwrap();
        assert_eq!(depth.resource, source_id);
        assert_eq!((depth.extent, depth.sample_count), ([16, 8, 1], 4));
        let BackingRegion::Image(source_region) = depth.regions[0] else {
            panic!("depth source must name its exact image subresource")
        };
        assert_eq!(
            (source_region.aspect, source_region.mip, source_region.layer),
            (ImageAspect::Depth, 1, 2)
        );
        let resolve = depth.resolve.as_ref().unwrap();
        assert_eq!(resolve.resource, resolve_id);
        assert_eq!((resolve.extent, resolve.sample_count), ([32, 16, 1], 1));
        let BackingRegion::Image(resolve_region) = resolve.regions[0] else {
            panic!("depth resolve must name its exact image subresource")
        };
        assert_eq!((resolve_region.mip, resolve_region.layer), (0, 0));
    }
}
