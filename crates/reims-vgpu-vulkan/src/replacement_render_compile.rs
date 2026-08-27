//! Actual Vulkan construction for one replacement render-pipeline variant.
//!
//! The semantic pipeline owns shader and vertex-interface state. A variant plan
//! adds only pass and static raster state. This module is the sole point that
//! combines those two immutable contracts with reported device capabilities and
//! creates the native handles owned by a ready replacement variant.

use crate::{
    engine::{context::SharedDeviceContext, VertexStepFunction},
    replacement_render::{
        ReplacementRenderPipeline, ReplacementRenderPipelineFamily, ReplacementRenderPipelinePlan,
        ReplacementRenderPipelineVariant, ReplacementRenderPipelineVariantKey,
    },
    translate::{self, reason::TranslateReason},
};
use ash::vk;
use reims_vgpu_core::{
    DescriptorUse, ReflectedShaderStage, ResolvedRenderPipeline, ShaderResourceKind,
};
use reims_vgpu_protocol::{
    resource::PipelineColorAttachment, BlendFactor, DepthStencilDescriptor, DepthStencilFace,
    LoadAction, SamplerCompareFunction, StoreAction,
};
use std::{
    collections::BTreeMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

#[derive(Clone)]
pub struct ReplacementRenderPipelineCompiler {
    context: Arc<SharedDeviceContext>,
    lifetime: reims_vgpu_core::VulkanDeviceEpoch,
}

impl ReplacementRenderPipelineCompiler {
    pub(crate) fn new(
        context: Arc<SharedDeviceContext>,
        lifetime: reims_vgpu_core::VulkanDeviceEpoch,
    ) -> Self {
        Self { context, lifetime }
    }

    pub fn compile_family_job(
        &self,
        family: &ReplacementRenderPipelineFamily<ReplacementRenderPipelineCompileError>,
        job: reims_vgpu_core::PipelineVariantCompileJob<ReplacementRenderPipelineVariantKey>,
        semantic: &ResolvedRenderPipeline,
        plan: ReplacementRenderPipelinePlan,
        depth_stencil: Option<&DepthStencilDescriptor>,
    ) -> Result<
        reims_vgpu_core::PipelineVariantPublication<ReplacementRenderPipelineVariant>,
        ReplacementRenderFamilyCompileError,
    > {
        match self.lifetime.with_active((job, plan), |(job, plan)| {
            let native = match catch_compile(|| unsafe {
                compile_variant(Arc::clone(&self.context), semantic, plan, depth_stencil)
            }) {
                Ok(native) => native,
                Err(reason) => {
                    let waiters = family
                        .refuse(job, reason)
                        .map_err(ReplacementRenderFamilyCompileError::Lifecycle)?;
                    return Err(ReplacementRenderFamilyCompileError::Refused { reason, waiters });
                }
            };
            family
                .compile_complete(job, native)
                .map_err(ReplacementRenderFamilyCompileError::Lifecycle)
        }) {
            Ok(result) => result,
            Err((job, _)) => {
                let reason = ReplacementRenderPipelineCompileError::DeviceLifetimeClosed;
                let waiters = family
                    .refuse(job, reason)
                    .map_err(ReplacementRenderFamilyCompileError::Lifecycle)?;
                Err(ReplacementRenderFamilyCompileError::Refused { reason, waiters })
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementRenderFamilyCompileError {
    Refused {
        reason: ReplacementRenderPipelineCompileError,
        waiters: Box<[reims_vgpu_protocol::TransactionId]>,
    },
    Lifecycle(reims_vgpu_core::PipelineVariantLifecycleError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementRenderPipelineCompileError {
    DeviceLifetimeClosed,
    WorkerPanicked,
    VertexProgramUnavailable,
    FragmentProgramUnavailable,
    ShaderInterfaceStageMismatch,
    UnsupportedShaderResource(&'static str),
    DescriptorSetUnsupported(u32),
    DescriptorCollision(u32),
    DescriptorCountZero(u32),
    DescriptorArrayIndexingUnsupported {
        binding: u32,
        descriptor_type: i32,
        count: u32,
    },
    DescriptorCountOverflow {
        descriptor_type: i32,
    },
    DescriptorLimitExceeded {
        descriptor_type: i32,
        requested: u32,
        supported: u32,
    },
    PipelineColorAttachmentMismatch(u32),
    PipelineColorAttachmentMissing(u32),
    PipelineSampleCountMismatch,
    ShaderProgramMismatch,
    ResolveActionMismatch(u32),
    DepthStencilStateMismatch,
    DualSourceBlendUnsupported(PipelineColorAttachment),
    FillModeNonSolidUnsupported(reims_vgpu_protocol::FillMode),
    WideLinesUnsupported {
        requested_bits: u32,
    },
    LineWidthOutOfRange {
        requested_bits: u32,
        minimum_bits: u32,
        maximum_bits: u32,
    },
    DepthClampUnsupported(reims_vgpu_protocol::DepthClipMode),
    ViewportCountUnsupported {
        requested: u32,
        supported: u32,
    },
    AttachmentFeedbackLoopUnsupported(vk::ImageAspectFlags),
    VertexBindingCollision(u32),
    VertexLayoutMissing(u32),
    VertexDivisorUnsupported(u32),
    Translate(TranslateReason),
    Vulkan {
        operation: &'static str,
        result: vk::Result,
    },
}

fn catch_compile<T>(
    compile: impl FnOnce() -> Result<T, ReplacementRenderPipelineCompileError>,
) -> Result<T, ReplacementRenderPipelineCompileError> {
    catch_unwind(AssertUnwindSafe(compile))
        .unwrap_or(Err(ReplacementRenderPipelineCompileError::WorkerPanicked))
}

#[derive(Clone, Copy)]
struct DescriptorDeclaration {
    ty: vk::DescriptorType,
    count: u32,
    stages: vk::ShaderStageFlags,
}

#[derive(Clone, Copy)]
struct VertexDeclaration {
    location: u32,
    binding: u32,
    format: vk::Format,
    offset: u32,
    stride: u32,
    step: VertexStepFunction,
    divisor: Option<u32>,
}

unsafe fn compile_variant(
    context: Arc<SharedDeviceContext>,
    semantic: &ResolvedRenderPipeline,
    plan: ReplacementRenderPipelinePlan,
    depth_stencil: Option<&DepthStencilDescriptor>,
) -> Result<ReplacementRenderPipelineVariant, ReplacementRenderPipelineCompileError> {
    validate_capabilities(&context, &plan)?;
    if semantic.desc.effective_raster_sample_count() != plan.sample_count.as_raw() {
        return Err(ReplacementRenderPipelineCompileError::PipelineSampleCountMismatch);
    }
    if semantic.vertex.variant().program != plan.program.vertex
        || semantic.fragment.variant().program != plan.program.fragment
    {
        return Err(ReplacementRenderPipelineCompileError::ShaderProgramMismatch);
    }
    let vertex = crate::m2v_cache::resolve_prepared_shader(semantic.vertex.variant().program.id)
        .ok_or(ReplacementRenderPipelineCompileError::VertexProgramUnavailable)?;
    let fragment =
        crate::m2v_cache::resolve_prepared_shader(semantic.fragment.variant().program.id)
            .ok_or(ReplacementRenderPipelineCompileError::FragmentProgramUnavailable)?;
    if semantic.vertex.interface.stage != ReflectedShaderStage::Vertex
        || semantic.fragment.interface.stage != ReflectedShaderStage::Fragment
    {
        return Err(ReplacementRenderPipelineCompileError::ShaderInterfaceStageMismatch);
    }
    let descriptors = descriptor_declarations(semantic)?;
    let vertices = vertex_declarations(&context, semantic, &plan, &vertex.vertex_inputs)?;
    let colors = color_states(semantic, &plan)?;
    let depth_state = depth_stencil_state(
        plan.depth_stencil_attachment.is_some(),
        plan.depth_stencil.is_some(),
        depth_stencil,
    )?;

    let vertex_module = context
        .device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vertex.words),
            None,
        )
        .map_err(|result| vk_error("create_vertex_shader_module", result))?;
    let fragment_module = match context.device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(&fragment.words),
        None,
    ) {
        Ok(module) => module,
        Err(result) => {
            context.device.destroy_shader_module(vertex_module, None);
            return Err(vk_error("create_fragment_shader_module", result));
        }
    };

    let result = create_variant_handles(
        &context,
        semantic,
        &plan,
        &descriptors,
        &vertices,
        &colors,
        depth_state.as_ref(),
        vertex_module,
        fragment_module,
    );
    context.device.destroy_shader_module(fragment_module, None);
    context.device.destroy_shader_module(vertex_module, None);
    let native = result?;
    context.persist_pipeline_cache();
    Ok(ReplacementRenderPipelineVariant::new(context, native))
}

fn validate_capabilities(
    context: &SharedDeviceContext,
    plan: &ReplacementRenderPipelinePlan,
) -> Result<(), ReplacementRenderPipelineCompileError> {
    if !plan.feedback_loop_aspects.is_empty() && !context.features.attachment_feedback_loop_layout {
        return Err(
            ReplacementRenderPipelineCompileError::AttachmentFeedbackLoopUnsupported(
                plan.feedback_loop_aspects,
            ),
        );
    }
    let max_viewports = if context.features.multi_viewport {
        context.features.max_viewports
    } else {
        1
    };
    validate_raster_capabilities(
        plan.static_state,
        plan.viewport_count,
        RasterCapabilities {
            fill_mode_non_solid: context.features.fill_mode_non_solid,
            wide_lines: context.features.wide_lines,
            line_width_range: context.features.line_width_range,
            depth_clamp: context.features.depth_clamp,
            max_viewports,
        },
    )
}

#[derive(Clone, Copy)]
struct RasterCapabilities {
    fill_mode_non_solid: bool,
    wide_lines: bool,
    line_width_range: [f32; 2],
    depth_clamp: bool,
    max_viewports: u32,
}

fn validate_raster_capabilities(
    state: crate::replacement_render::ReplacementRenderStaticState,
    viewport_count: u32,
    capabilities: RasterCapabilities,
) -> Result<(), ReplacementRenderPipelineCompileError> {
    let line_width = f32::from_bits(state.line_width_bits);
    if state.fill_mode != reims_vgpu_protocol::FillMode::Fill && !capabilities.fill_mode_non_solid {
        return Err(
            ReplacementRenderPipelineCompileError::FillModeNonSolidUnsupported(state.fill_mode),
        );
    }
    if line_width != 1.0 && !capabilities.wide_lines {
        return Err(
            ReplacementRenderPipelineCompileError::WideLinesUnsupported {
                requested_bits: state.line_width_bits,
            },
        );
    }
    let [minimum, maximum] = capabilities.line_width_range;
    if !line_width.is_finite() || line_width < minimum || line_width > maximum {
        return Err(ReplacementRenderPipelineCompileError::LineWidthOutOfRange {
            requested_bits: state.line_width_bits,
            minimum_bits: minimum.to_bits(),
            maximum_bits: maximum.to_bits(),
        });
    }
    if state.depth_clip_mode != reims_vgpu_protocol::DepthClipMode::Clip
        && !capabilities.depth_clamp
    {
        return Err(
            ReplacementRenderPipelineCompileError::DepthClampUnsupported(state.depth_clip_mode),
        );
    }
    if viewport_count == 0 || viewport_count > capabilities.max_viewports {
        return Err(
            ReplacementRenderPipelineCompileError::ViewportCountUnsupported {
                requested: viewport_count,
                supported: capabilities.max_viewports,
            },
        );
    }
    Ok(())
}

fn descriptor_declarations(
    semantic: &ResolvedRenderPipeline,
) -> Result<BTreeMap<u32, DescriptorDeclaration>, ReplacementRenderPipelineCompileError> {
    let mut declarations = BTreeMap::new();
    add_stage_descriptors(
        &mut declarations,
        semantic.vertex.interface.as_ref(),
        semantic.vertex.variant(),
        vk::ShaderStageFlags::VERTEX,
    )?;
    add_stage_descriptors(
        &mut declarations,
        semantic.fragment.interface.as_ref(),
        semantic.fragment.variant(),
        vk::ShaderStageFlags::FRAGMENT,
    )?;
    Ok(declarations)
}

fn add_stage_descriptors(
    declarations: &mut BTreeMap<u32, DescriptorDeclaration>,
    interface: &reims_vgpu_core::ShaderInterface,
    variant: &reims_vgpu_core::PreparedShaderVariant,
    stage: vk::ShaderStageFlags,
) -> Result<(), ReplacementRenderPipelineCompileError> {
    for resource in &interface.bindings {
        let Some(location) = resource.descriptor else {
            if let Some(feature) = resource.kind.unsupported_vulkan_name() {
                return Err(
                    ReplacementRenderPipelineCompileError::UnsupportedShaderResource(feature),
                );
            }
            continue;
        };
        if location.set != 0 {
            return Err(
                ReplacementRenderPipelineCompileError::DescriptorSetUnsupported(location.set),
            );
        }
        if location.count == 0 {
            return Err(ReplacementRenderPipelineCompileError::DescriptorCountZero(
                location.binding,
            ));
        }
        if variant.descriptor_use(location.binding) == DescriptorUse::NotDeclared {
            continue;
        }
        let ty = descriptor_type(resource.kind).ok_or_else(|| {
            ReplacementRenderPipelineCompileError::UnsupportedShaderResource(
                resource
                    .kind
                    .unsupported_vulkan_name()
                    .unwrap_or("render_resource"),
            )
        })?;
        match declarations.get_mut(&location.binding) {
            Some(found) if found.ty == ty && found.count == location.count => {
                found.stages |= stage;
            }
            Some(_) => {
                return Err(ReplacementRenderPipelineCompileError::DescriptorCollision(
                    location.binding,
                ));
            }
            None => {
                declarations.insert(
                    location.binding,
                    DescriptorDeclaration {
                        ty,
                        count: location.count,
                        stages: stage,
                    },
                );
            }
        }
    }
    Ok(())
}

fn descriptor_type(kind: ShaderResourceKind) -> Option<vk::DescriptorType> {
    match kind {
        ShaderResourceKind::Buffer => Some(vk::DescriptorType::STORAGE_BUFFER),
        ShaderResourceKind::Texture | ShaderResourceKind::TextureArray => {
            Some(vk::DescriptorType::SAMPLED_IMAGE)
        }
        ShaderResourceKind::StorageImage => Some(vk::DescriptorType::STORAGE_IMAGE),
        ShaderResourceKind::Sampler | ShaderResourceKind::StaticSampler => {
            Some(vk::DescriptorType::SAMPLER)
        }
        ShaderResourceKind::ColorInput => Some(vk::DescriptorType::INPUT_ATTACHMENT),
        ShaderResourceKind::ThreadgroupBuffer
        | ShaderResourceKind::KernelStageInput
        | ShaderResourceKind::AccelerationStructureShadow
        | ShaderResourceKind::PrimitiveAccelerationStructure
        | ShaderResourceKind::VisibleFunctionTable
        | ShaderResourceKind::IntersectionFunctionTable
        | ShaderResourceKind::EmbeddedArgBufferTexture
        | ShaderResourceKind::EmbeddedArgBufferBuffer
        | ShaderResourceKind::BufferAddressTable => None,
    }
}

fn vertex_declarations(
    context: &SharedDeviceContext,
    semantic: &ResolvedRenderPipeline,
    plan: &ReplacementRenderPipelinePlan,
    widths: &crate::spirv_vertex_input::VertexInputWidths,
) -> Result<Vec<VertexDeclaration>, ReplacementRenderPipelineCompileError> {
    let mut declarations = Vec::new();
    let mut bindings = BTreeMap::<u32, (u32, VertexStepFunction, Option<u32>)>::new();
    for attribute in &semantic.desc.vertex_attributes {
        if attribute.format == 0 {
            continue;
        }
        let stride = plan
            .vertex_buffers
            .iter()
            .find(|layout| layout.binding == attribute.buffer_index)
            .map(|layout| layout.stride)
            .ok_or(ReplacementRenderPipelineCompileError::VertexLayoutMissing(
                attribute.buffer_index,
            ))?;
        let source_format = translate::vertex::attribute_format(attribute.format)
            .map_err(ReplacementRenderPipelineCompileError::Translate)?;
        let resolved = context
            .vertex_formats
            .resolve(source_format, attribute.offset, stride, || {
                widths.at(attribute.location)
            })
            .map_err(ReplacementRenderPipelineCompileError::Translate)?;
        let step = translate::vertex::step_function(attribute.declared_step_function)
            .map_err(ReplacementRenderPipelineCompileError::Translate)?;
        let divisor = match step {
            VertexStepFunction::Constant => Some(0),
            VertexStepFunction::PerVertex => None,
            VertexStepFunction::PerInstance if attribute.step_rate() == 1 => None,
            VertexStepFunction::PerInstance => Some(attribute.step_rate()),
        };
        if let Some(divisor) = divisor {
            if (divisor == 0 && !context.vertex_divisor.zero_divisor)
                || (divisor > 1 && !context.vertex_divisor.instance_rate_divisor)
                || divisor > context.vertex_divisor.max_divisor
            {
                return Err(
                    ReplacementRenderPipelineCompileError::VertexDivisorUnsupported(divisor),
                );
            }
        }
        let signature = (stride, step, divisor);
        if bindings
            .insert(attribute.buffer_index, signature)
            .is_some_and(|found| found != signature)
        {
            return Err(
                ReplacementRenderPipelineCompileError::VertexBindingCollision(
                    attribute.buffer_index,
                ),
            );
        }
        declarations.push(VertexDeclaration {
            location: attribute.location,
            binding: attribute.buffer_index,
            format: resolved.format,
            offset: attribute.offset,
            stride,
            step,
            divisor,
        });
    }
    Ok(declarations)
}

fn color_states(
    semantic: &ResolvedRenderPipeline,
    plan: &ReplacementRenderPipelinePlan,
) -> Result<Vec<vk::PipelineColorBlendAttachmentState>, ReplacementRenderPipelineCompileError> {
    let mut states = Vec::with_capacity(plan.color_attachments.len());
    for (slot, native) in plan.color_attachments.iter().enumerate() {
        let slot = u32::try_from(slot).expect("color attachment count is protocol bounded");
        let declared = pipeline_color_attachment(semantic, slot)
            .ok_or(ReplacementRenderPipelineCompileError::PipelineColorAttachmentMissing(slot))?;
        if !declared.has_pixel_format || declared.pixel_format != u32::from(native.format) {
            return Err(
                ReplacementRenderPipelineCompileError::PipelineColorAttachmentMismatch(slot),
            );
        }
        let has_resolve = native.resolve_format.is_some();
        if has_resolve
            != matches!(
                native.store,
                StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
            )
        {
            return Err(ReplacementRenderPipelineCompileError::ResolveActionMismatch(slot));
        }
        let mut state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(translate::blend::vk_color_write_mask(declared.write_mask));
        if declared.blending_enabled {
            let blend = translate::blend::state(declared)
                .map_err(ReplacementRenderPipelineCompileError::Translate)?;
            state = state
                .blend_enable(true)
                .src_color_blend_factor(translate::blend::vk_factor(blend.src_color))
                .dst_color_blend_factor(translate::blend::vk_factor(blend.dst_color))
                .color_blend_op(translate::blend::vk_operation(blend.color_op))
                .src_alpha_blend_factor(translate::blend::vk_factor(blend.src_alpha))
                .dst_alpha_blend_factor(translate::blend::vk_factor(blend.dst_alpha))
                .alpha_blend_op(translate::blend::vk_operation(blend.alpha_op));
        }
        states.push(state);
    }
    Ok(states)
}

fn pipeline_color_attachment(
    semantic: &ResolvedRenderPipeline,
    slot: u32,
) -> Option<&PipelineColorAttachment> {
    std::iter::once(&semantic.desc.color0)
        .chain(&semantic.desc.color_attachments)
        .find(|attachment| attachment.slot == slot)
}

#[derive(Clone, Copy)]
struct DepthStencilState {
    depth_test: bool,
    depth_write: bool,
    depth_compare: SamplerCompareFunction,
    front: Option<vk::StencilOpState>,
    back: Option<vk::StencilOpState>,
}

fn depth_stencil_state(
    has_attachment: bool,
    state_bound: bool,
    descriptor: Option<&DepthStencilDescriptor>,
) -> Result<Option<DepthStencilState>, ReplacementRenderPipelineCompileError> {
    if state_bound != descriptor.is_some() {
        return Err(ReplacementRenderPipelineCompileError::DepthStencilStateMismatch);
    }
    if !has_attachment {
        return Ok(None);
    }
    let Some(descriptor) = descriptor else {
        return Ok(Some(DepthStencilState {
            depth_test: false,
            depth_write: false,
            depth_compare: SamplerCompareFunction::Always,
            front: None,
            back: None,
        }));
    };
    let depth_compare = translate::raster::compare_function(descriptor.depth_compare_function)
        .map_err(ReplacementRenderPipelineCompileError::Translate)?;
    let front = descriptor
        .front_stencil_present
        .then(|| stencil_face(&descriptor.front_face))
        .transpose()?;
    let back = descriptor
        .back_stencil_present
        .then(|| stencil_face(&descriptor.back_face))
        .transpose()?;
    Ok(Some(DepthStencilState {
        depth_test: descriptor.depth_write_enabled
            || depth_compare != SamplerCompareFunction::Always,
        depth_write: descriptor.depth_write_enabled,
        depth_compare,
        front,
        back,
    }))
}

fn stencil_face(
    face: &DepthStencilFace,
) -> Result<vk::StencilOpState, ReplacementRenderPipelineCompileError> {
    Ok(vk::StencilOpState::default()
        .compare_op(translate::raster::vk_compare_op(
            translate::raster::compare_function(face.compare_function)
                .map_err(ReplacementRenderPipelineCompileError::Translate)?,
        ))
        .fail_op(translate::raster::vk_stencil_op(
            translate::raster::stencil_operation(face.stencil_failure_operation)
                .map_err(ReplacementRenderPipelineCompileError::Translate)?,
        ))
        .depth_fail_op(translate::raster::vk_stencil_op(
            translate::raster::stencil_operation(face.depth_failure_operation)
                .map_err(ReplacementRenderPipelineCompileError::Translate)?,
        ))
        .pass_op(translate::raster::vk_stencil_op(
            translate::raster::stencil_operation(face.depth_stencil_pass_operation)
                .map_err(ReplacementRenderPipelineCompileError::Translate)?,
        ))
        .compare_mask(face.read_mask)
        .write_mask(face.write_mask))
}

fn dual_source_attachment(
    descriptor: &reims_vgpu_protocol::RenderPipelineDescriptor,
) -> Option<PipelineColorAttachment> {
    std::iter::once(&descriptor.color0)
        .chain(&descriptor.color_attachments)
        .filter(|attachment| attachment.blending_enabled)
        .find(|attachment| {
            let Ok(blend) = reims_vgpu_protocol::blend_state(attachment) else {
                return false;
            };
            [
                blend.src_color,
                blend.dst_color,
                blend.src_alpha,
                blend.dst_alpha,
            ]
            .into_iter()
            .any(BlendFactor::is_dual_source)
        })
        .copied()
}

fn load_ops(load: LoadAction, layout: vk::ImageLayout) -> (vk::AttachmentLoadOp, vk::ImageLayout) {
    match load {
        LoadAction::DontCare => (vk::AttachmentLoadOp::DONT_CARE, vk::ImageLayout::UNDEFINED),
        LoadAction::Load => (vk::AttachmentLoadOp::LOAD, layout),
        LoadAction::Clear => (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED),
    }
}

fn store_op(store: StoreAction) -> vk::AttachmentStoreOp {
    match store {
        StoreAction::Store | StoreAction::StoreAndMultisampleResolve => {
            vk::AttachmentStoreOp::STORE
        }
        StoreAction::DontCare | StoreAction::MultisampleResolve => vk::AttachmentStoreOp::DONT_CARE,
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the constructor consumes the independently validated Vulkan object inputs"
)]
unsafe fn create_variant_handles(
    context: &Arc<SharedDeviceContext>,
    semantic: &ResolvedRenderPipeline,
    plan: &ReplacementRenderPipelinePlan,
    descriptors: &BTreeMap<u32, DescriptorDeclaration>,
    vertices: &[VertexDeclaration],
    colors: &[vk::PipelineColorBlendAttachmentState],
    depth: Option<&DepthStencilState>,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
) -> Result<ReplacementRenderPipeline, ReplacementRenderPipelineCompileError> {
    if let Some(attachment) = dual_source_attachment(&semantic.desc) {
        if !context.features.dual_src_blend {
            return Err(
                ReplacementRenderPipelineCompileError::DualSourceBlendUnsupported(attachment),
            );
        }
    }
    validate_descriptor_capabilities(
        descriptors,
        DescriptorCapabilities {
            sampled_dynamic_indexing: context.features.sampled_image_array_dynamic_indexing,
            sampled_limit: context.features.sampled_image_descriptor_limit,
            storage_dynamic_indexing: context.features.storage_image_array_dynamic_indexing,
            storage_limit: context.features.storage_image_descriptor_limit,
        },
    )?;
    let descriptor_bindings = descriptors
        .iter()
        .map(|(binding, declaration)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(*binding)
                .descriptor_type(declaration.ty)
                .descriptor_count(declaration.count)
                .stage_flags(declaration.stages)
        })
        .collect::<Vec<_>>();
    let descriptor_flags = descriptors
        .values()
        .map(|declaration| {
            if declaration.count > 1 && context.features.descriptor_binding_partially_bound {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
            } else {
                vk::DescriptorBindingFlags::empty()
            }
        })
        .collect::<Vec<_>>();
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&descriptor_flags);
    let descriptor_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&descriptor_bindings)
        .push_next(&mut binding_flags);
    let descriptor_set_layout = context
        .device
        .create_descriptor_set_layout(&descriptor_info, None)
        .map_err(|result| vk_error("create_descriptor_set_layout", result))?;
    let pipeline_layout = match context.device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_set_layout]),
        None,
    ) {
        Ok(layout) => layout,
        Err(result) => {
            context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            return Err(vk_error("create_pipeline_layout", result));
        }
    };
    let render_pass = match create_render_pass(context, plan) {
        Ok(pass) => pass,
        Err(error) => {
            context
                .device
                .destroy_pipeline_layout(pipeline_layout, None);
            context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            return Err(error);
        }
    };
    let pipeline = match create_graphics_pipeline(
        context,
        semantic,
        plan,
        vertices,
        colors,
        depth,
        vertex_module,
        fragment_module,
        pipeline_layout,
        render_pass,
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            context.device.destroy_render_pass(render_pass, None);
            context
                .device
                .destroy_pipeline_layout(pipeline_layout, None);
            context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            return Err(error);
        }
    };
    Ok(ReplacementRenderPipeline {
        pipeline,
        layout: pipeline_layout,
        descriptor_set_layout,
        render_pass,
        program: plan.program.clone(),
        vertex_buffers: plan.vertex_buffers.clone(),
        color_attachments: plan.color_attachments.clone(),
        depth_stencil_attachment: plan.depth_stencil_attachment,
        feedback_loop_aspects: plan.feedback_loop_aspects,
        color_input: plan_uses_color_input(plan),
        sample_count: plan.sample_count,
        viewport_count: plan.viewport_count,
        static_state: plan.static_state,
        dynamic_states: plan_dynamic_states(plan, semantic, depth),
        depth_stencil: plan.depth_stencil,
    })
}

/// The one derivation of a pipeline's dynamic-state set.
///
/// The compiled pipeline and the recorder both read this, so a state cannot be
/// set without having been declared.
fn plan_dynamic_states(
    plan: &ReplacementRenderPipelinePlan,
    semantic: &ResolvedRenderPipeline,
    depth: Option<&DepthStencilState>,
) -> crate::replacement_render::ReplacementRenderDynamicStates {
    crate::replacement_render::ReplacementRenderDynamicStates {
        depth_bias: plan.static_state.depth_bias_enabled,
        stencil_reference: depth.is_some_and(|state| state.front.is_some() || state.back.is_some()),
        blend_constants: semantic_uses_blend_constants(semantic),
    }
}

#[derive(Clone, Copy)]
struct DescriptorCapabilities {
    sampled_dynamic_indexing: bool,
    sampled_limit: u32,
    storage_dynamic_indexing: bool,
    storage_limit: u32,
}

fn validate_descriptor_capabilities(
    descriptors: &BTreeMap<u32, DescriptorDeclaration>,
    capabilities: DescriptorCapabilities,
) -> Result<(), ReplacementRenderPipelineCompileError> {
    for (descriptor_type, dynamic_indexing, supported) in [
        (
            vk::DescriptorType::SAMPLED_IMAGE,
            capabilities.sampled_dynamic_indexing,
            capabilities.sampled_limit,
        ),
        (
            vk::DescriptorType::STORAGE_IMAGE,
            capabilities.storage_dynamic_indexing,
            capabilities.storage_limit,
        ),
    ] {
        if !dynamic_indexing {
            if let Some((binding, declaration)) = descriptors
                .iter()
                .find(|(_, declaration)| declaration.ty == descriptor_type && declaration.count > 1)
            {
                return Err(
                    ReplacementRenderPipelineCompileError::DescriptorArrayIndexingUnsupported {
                        binding: *binding,
                        descriptor_type: descriptor_type.as_raw(),
                        count: declaration.count,
                    },
                );
            }
        }
        let requested = descriptors
            .values()
            .filter(|declaration| declaration.ty == descriptor_type)
            .try_fold(0u32, |total, declaration| {
                total.checked_add(declaration.count)
            })
            .ok_or(
                ReplacementRenderPipelineCompileError::DescriptorCountOverflow {
                    descriptor_type: descriptor_type.as_raw(),
                },
            )?;
        if requested > supported {
            return Err(
                ReplacementRenderPipelineCompileError::DescriptorLimitExceeded {
                    descriptor_type: descriptor_type.as_raw(),
                    requested,
                    supported,
                },
            );
        }
    }
    Ok(())
}

unsafe fn create_render_pass(
    context: &SharedDeviceContext,
    plan: &ReplacementRenderPipelinePlan,
) -> Result<vk::RenderPass, ReplacementRenderPipelineCompileError> {
    let color_input = plan_uses_color_input(plan);
    let color_layout = |slot: usize| {
        let attachment = plan.color_attachments[slot];
        crate::replacement_render::render_attachment_layout(
            reims_vgpu_core::RenderAttachmentRole::Color(slot as u32),
            attachment.feedback_loop,
            attachment.input_attachment,
        )
    };
    // A resolve target is written by the pass and never read by it, so it
    // takes the plain attachment layout however the attachment it resolves is
    // read. This is the same statement the image-state derivation makes about
    // the same image; both call one function so they cannot drift apart.
    let resolve_layout = |slot: usize| {
        crate::replacement_render::render_attachment_layout(
            reims_vgpu_core::RenderAttachmentRole::Color(slot as u32),
            false,
            false,
        )
    };
    let mut attachments = Vec::new();
    let mut color_refs = Vec::new();
    for (slot, color) in plan.color_attachments.iter().enumerate() {
        let layout = color_layout(slot);
        let (load, initial) = load_ops(color.load, layout);
        let format = translate::pixel::translate(color.format)
            .map_err(ReplacementRenderPipelineCompileError::Translate)?
            .vk;
        color_refs.push(
            vk::AttachmentReference2::default()
                .attachment(attachments.len() as u32)
                .layout(layout),
        );
        attachments.push(
            vk::AttachmentDescription2::default()
                .format(format)
                .samples(plan.sample_count)
                .load_op(load)
                .store_op(store_op(color.store))
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(initial)
                .final_layout(layout),
        );
    }
    let has_color_resolve = plan
        .color_attachments
        .iter()
        .any(|attachment| attachment.resolve_format.is_some());
    let mut resolve_refs = Vec::new();
    if has_color_resolve {
        for (slot, color) in plan.color_attachments.iter().enumerate() {
            let layout = resolve_layout(slot);
            if let Some(format) = color.resolve_format {
                resolve_refs.push(
                    vk::AttachmentReference2::default()
                        .attachment(attachments.len() as u32)
                        .layout(layout),
                );
                attachments.push(
                    vk::AttachmentDescription2::default()
                        .format(
                            translate::pixel::translate(format)
                                .map_err(ReplacementRenderPipelineCompileError::Translate)?
                                .vk,
                        )
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(layout),
                );
            } else {
                resolve_refs.push(
                    vk::AttachmentReference2::default()
                        .attachment(vk::ATTACHMENT_UNUSED)
                        .layout(vk::ImageLayout::UNDEFINED),
                );
            }
        }
    }

    let mut depth_ref = None;
    let mut depth_resolve_ref = None;
    if let Some(depth) = plan.depth_stencil_attachment {
        validate_depth_resolve_actions(depth)?;
        let format = combined_depth_stencil_format(depth.depth_format, depth.stencil_format)?;
        let layout = crate::replacement_render::render_attachment_layout(
            reims_vgpu_core::RenderAttachmentRole::Depth,
            !depth.feedback_loop_aspects.is_empty(),
            false,
        );
        let (depth_load, depth_initial) = load_ops(depth.depth_load, layout);
        let (stencil_load, stencil_initial) = load_ops(depth.stencil_load, layout);
        let initial = if depth_initial == layout || stencil_initial == layout {
            layout
        } else {
            vk::ImageLayout::UNDEFINED
        };
        depth_ref = Some(
            vk::AttachmentReference2::default()
                .attachment(attachments.len() as u32)
                .layout(layout),
        );
        attachments.push(
            vk::AttachmentDescription2::default()
                .format(format)
                .samples(plan.sample_count)
                .load_op(depth_load)
                .store_op(store_op(depth.depth_store))
                .stencil_load_op(stencil_load)
                .stencil_store_op(store_op(depth.stencil_store))
                .initial_layout(initial)
                .final_layout(layout),
        );
        if depth.depth_resolve_format.is_some() || depth.stencil_resolve_format.is_some() {
            let resolve_layout = crate::replacement_render::render_attachment_layout(
                reims_vgpu_core::RenderAttachmentRole::Depth,
                false,
                false,
            );
            depth_resolve_ref = Some(
                vk::AttachmentReference2::default()
                    .attachment(attachments.len() as u32)
                    .layout(resolve_layout),
            );
            attachments.push(
                vk::AttachmentDescription2::default()
                    .format(combined_depth_stencil_format(
                        depth.depth_resolve_format,
                        depth.stencil_resolve_format,
                    )?)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(if depth.depth_resolve_format.is_some() {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(if depth.stencil_resolve_format.is_some() {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(layout),
            );
        }
    }

    let input_ref = [vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::GENERAL)];
    let mut subpass = vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    if !resolve_refs.is_empty() {
        subpass = subpass.resolve_attachments(&resolve_refs);
    }
    if color_input {
        if color_refs.is_empty() {
            return Err(ReplacementRenderPipelineCompileError::PipelineColorAttachmentMissing(0));
        }
        subpass = subpass.input_attachments(&input_ref);
    }
    if let Some(depth_ref) = depth_ref.as_ref() {
        subpass = subpass.depth_stencil_attachment(depth_ref);
    }
    let mut depth_resolve = vk::SubpassDescriptionDepthStencilResolve::default()
        .depth_resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO)
        .stencil_resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO);
    if let Some(reference) = depth_resolve_ref.as_ref() {
        depth_resolve = depth_resolve.depth_stencil_resolve_attachment(reference);
        subpass = subpass.push_next(&mut depth_resolve);
    }
    let subpasses = [subpass];
    let render_pass_info = vk::RenderPassCreateInfo2::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    context
        .device
        .create_render_pass2(&render_pass_info, None)
        .map_err(|result| vk_error("create_render_pass2", result))
}

/// Whether the pass declares its first colour attachment as a subpass input.
///
/// The plan carries the fact, stamped over the whole encoder from the fragment
/// interfaces of the draws in it, because the pass declares one layout per
/// attachment for its life and a read by any draw decides it for all of them.
fn plan_uses_color_input(plan: &ReplacementRenderPipelinePlan) -> bool {
    plan.color_attachments
        .first()
        .is_some_and(|attachment| attachment.input_attachment)
}

fn validate_depth_resolve_actions(
    depth: crate::replacement_render::ReplacementRenderDepthStencilAttachment,
) -> Result<(), ReplacementRenderPipelineCompileError> {
    for (slot, store, resolve) in [
        (u32::MAX - 1, depth.depth_store, depth.depth_resolve_format),
        (u32::MAX, depth.stencil_store, depth.stencil_resolve_format),
    ] {
        if resolve.is_some()
            != matches!(
                store,
                StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
            )
        {
            return Err(ReplacementRenderPipelineCompileError::ResolveActionMismatch(slot));
        }
    }
    Ok(())
}

fn combined_depth_stencil_format(
    depth: Option<u16>,
    stencil: Option<u16>,
) -> Result<vk::Format, ReplacementRenderPipelineCompileError> {
    let depth = depth
        .map(translate::pixel::translate)
        .transpose()
        .map_err(ReplacementRenderPipelineCompileError::Translate)?
        .map(|format| format.vk);
    let stencil = stencil
        .map(translate::pixel::translate)
        .transpose()
        .map_err(ReplacementRenderPipelineCompileError::Translate)?
        .map(|format| format.vk);
    match (depth, stencil) {
        (Some(depth), Some(stencil)) if depth != stencil => {
            Err(ReplacementRenderPipelineCompileError::DepthStencilStateMismatch)
        }
        (Some(format), _) | (_, Some(format)) => Ok(format),
        (None, None) => Err(ReplacementRenderPipelineCompileError::DepthStencilStateMismatch),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Vulkan graphics creation names every independently owned state block"
)]
unsafe fn create_graphics_pipeline(
    context: &SharedDeviceContext,
    semantic: &ResolvedRenderPipeline,
    plan: &ReplacementRenderPipelinePlan,
    vertices: &[VertexDeclaration],
    colors: &[vk::PipelineColorBlendAttachmentState],
    depth: Option<&DepthStencilState>,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline, ReplacementRenderPipelineCompileError> {
    let entry = crate::engine::context::main_entry();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&entry),
    ];
    let mut binding_map = BTreeMap::new();
    for vertex in vertices {
        binding_map
            .entry(vertex.binding)
            .or_insert((vertex.stride, vertex.step, vertex.divisor));
    }
    let binding_descriptions = binding_map
        .iter()
        .map(|(binding, (stride, step, _))| {
            vk::VertexInputBindingDescription::default()
                .binding(*binding)
                .stride(*stride)
                .input_rate(translate::vertex::vk_input_rate(*step))
        })
        .collect::<Vec<_>>();
    let attribute_descriptions = vertices
        .iter()
        .map(|vertex| {
            vk::VertexInputAttributeDescription::default()
                .location(vertex.location)
                .binding(vertex.binding)
                .format(vertex.format)
                .offset(vertex.offset)
        })
        .collect::<Vec<_>>();
    let divisor_descriptions = binding_map
        .iter()
        .filter_map(|(binding, (_, _, divisor))| {
            divisor.map(|divisor| {
                vk::VertexInputBindingDivisorDescriptionKHR::default()
                    .binding(*binding)
                    .divisor(divisor)
            })
        })
        .collect::<Vec<_>>();
    let mut divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
        .vertex_binding_divisors(&divisor_descriptions);
    let mut vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);
    if !divisor_descriptions.is_empty() {
        vertex_input = vertex_input.push_next(&mut divisor_state);
    }
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(translate::raster::vk_topology(plan.static_state.topology));
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(plan.viewport_count)
        .scissor_count(plan.viewport_count);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(translate::raster::vk_polygon_mode(
            plan.static_state.fill_mode,
        ))
        .cull_mode(translate::raster::vk_cull_mode(plan.static_state.cull_mode))
        .front_face(translate::raster::vk_front_face(
            plan.static_state.front_face_ccw,
        ))
        .depth_clamp_enable(translate::raster::vk_depth_clamp_enable(
            plan.static_state.depth_clip_mode,
        ))
        .depth_bias_enable(plan.static_state.depth_bias_enabled)
        .line_width(f32::from_bits(plan.static_state.line_width_bits));
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(plan.sample_count);
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(colors)
        .blend_constants([0.0; 4]);
    let dynamic_states = plan_dynamic_states(plan, semantic, depth).declarations();
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut depth_info = vk::PipelineDepthStencilStateCreateInfo::default();
    if let Some(depth) = depth {
        depth_info = depth_info
            .depth_test_enable(depth.depth_test)
            .depth_write_enable(depth.depth_write)
            .depth_compare_op(translate::raster::vk_compare_op(depth.depth_compare))
            .stencil_test_enable(depth.front.is_some() || depth.back.is_some())
            .front(depth.front.unwrap_or_default())
            .back(depth.back.unwrap_or_default());
    }
    let mut flags = vk::PipelineCreateFlags::empty();
    if plan
        .feedback_loop_aspects
        .contains(vk::ImageAspectFlags::COLOR)
    {
        flags |= vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    if plan
        .feedback_loop_aspects
        .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
    {
        flags |= vk::PipelineCreateFlags::DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .flags(flags)
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    if depth.is_some() {
        info = info.depth_stencil_state(&depth_info);
    }
    context
        .device
        .create_graphics_pipelines(context.pipeline_cache, &[info], None)
        .map(|pipelines| pipelines[0])
        .map_err(|(_, result)| vk_error("create_graphics_pipeline", result))
}

fn semantic_uses_blend_constants(semantic: &ResolvedRenderPipeline) -> bool {
    std::iter::once(&semantic.desc.color0)
        .chain(&semantic.desc.color_attachments)
        .filter(|attachment| attachment.blending_enabled)
        .filter_map(|attachment| reims_vgpu_protocol::blend_state(attachment).ok())
        .flat_map(|blend| {
            [
                blend.src_color,
                blend.dst_color,
                blend.src_alpha,
                blend.dst_alpha,
            ]
        })
        .any(|factor| {
            matches!(
                factor,
                BlendFactor::ConstantColor
                    | BlendFactor::OneMinusConstantColor
                    | BlendFactor::ConstantAlpha
                    | BlendFactor::OneMinusConstantAlpha
            )
        })
}

fn vk_error(operation: &'static str, result: vk::Result) -> ReplacementRenderPipelineCompileError {
    ReplacementRenderPipelineCompileError::Vulkan { operation, result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::context::DeviceContext;

    #[test]
    fn compiler_panic_becomes_a_typed_refusal() {
        assert_eq!(
            catch_compile::<()>(|| panic!("fixture compiler panic")),
            Err(ReplacementRenderPipelineCompileError::WorkerPanicked)
        );
    }

    #[test]
    fn raster_capability_refusals_retain_requests_and_host_limits() {
        let capabilities = RasterCapabilities {
            fill_mode_non_solid: false,
            wide_lines: false,
            line_width_range: [1.0, 8.0],
            depth_clamp: false,
            max_viewports: 2,
        };
        assert_eq!(
            validate_raster_capabilities(
                crate::replacement_render::ReplacementRenderStaticState {
                    fill_mode: reims_vgpu_protocol::FillMode::Lines,
                    ..Default::default()
                },
                1,
                capabilities,
            ),
            Err(
                ReplacementRenderPipelineCompileError::FillModeNonSolidUnsupported(
                    reims_vgpu_protocol::FillMode::Lines,
                )
            )
        );
        assert_eq!(
            validate_raster_capabilities(
                crate::replacement_render::ReplacementRenderStaticState {
                    line_width_bits: 2.5f32.to_bits(),
                    ..Default::default()
                },
                1,
                capabilities,
            ),
            Err(
                ReplacementRenderPipelineCompileError::WideLinesUnsupported {
                    requested_bits: 2.5f32.to_bits(),
                }
            )
        );
        assert_eq!(
            validate_raster_capabilities(
                crate::replacement_render::ReplacementRenderStaticState {
                    line_width_bits: 9.0f32.to_bits(),
                    ..Default::default()
                },
                1,
                RasterCapabilities {
                    wide_lines: true,
                    ..capabilities
                },
            ),
            Err(ReplacementRenderPipelineCompileError::LineWidthOutOfRange {
                requested_bits: 9.0f32.to_bits(),
                minimum_bits: 1.0f32.to_bits(),
                maximum_bits: 8.0f32.to_bits(),
            })
        );
        assert_eq!(
            validate_raster_capabilities(
                crate::replacement_render::ReplacementRenderStaticState {
                    depth_clip_mode: reims_vgpu_protocol::DepthClipMode::Clamp,
                    ..Default::default()
                },
                1,
                capabilities,
            ),
            Err(
                ReplacementRenderPipelineCompileError::DepthClampUnsupported(
                    reims_vgpu_protocol::DepthClipMode::Clamp,
                )
            )
        );
        assert_eq!(
            validate_raster_capabilities(
                crate::replacement_render::ReplacementRenderStaticState::default(),
                3,
                capabilities,
            ),
            Err(
                ReplacementRenderPipelineCompileError::ViewportCountUnsupported {
                    requested: 3,
                    supported: 2,
                }
            )
        );
    }

    #[test]
    fn dual_source_capability_refusal_retains_the_exact_attachment() {
        let dual_source = PipelineColorAttachment {
            slot: 3,
            has_pixel_format: true,
            pixel_format: 80,
            blending_enabled: true,
            src_rgb: 15,
            dst_rgb: 1,
            op_rgb: 0,
            src_alpha: 2,
            dst_alpha: 18,
            op_alpha: 1,
            write_mask: reims_vgpu_protocol::ColorWriteMask::default(),
        };
        let descriptor = reims_vgpu_protocol::RenderPipelineDescriptor {
            color_attachments: vec![dual_source],
            ..Default::default()
        };
        assert_eq!(dual_source_attachment(&descriptor), Some(dual_source));
        assert_eq!(
            ReplacementRenderPipelineCompileError::DualSourceBlendUnsupported(
                dual_source_attachment(&descriptor).unwrap(),
            ),
            ReplacementRenderPipelineCompileError::DualSourceBlendUnsupported(dual_source),
        );
    }

    #[test]
    fn descriptor_capability_refusals_distinguish_indexing_overflow_and_limit() {
        let declaration = |ty, count| DescriptorDeclaration {
            ty,
            count,
            stages: vk::ShaderStageFlags::FRAGMENT,
        };
        let capabilities = DescriptorCapabilities {
            sampled_dynamic_indexing: false,
            sampled_limit: 2,
            storage_dynamic_indexing: true,
            storage_limit: u32::MAX,
        };
        let arrays = BTreeMap::from([(7, declaration(vk::DescriptorType::SAMPLED_IMAGE, 3))]);
        assert_eq!(
            validate_descriptor_capabilities(&arrays, capabilities),
            Err(
                ReplacementRenderPipelineCompileError::DescriptorArrayIndexingUnsupported {
                    binding: 7,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE.as_raw(),
                    count: 3,
                }
            )
        );
        assert_eq!(
            validate_descriptor_capabilities(
                &arrays,
                DescriptorCapabilities {
                    sampled_dynamic_indexing: true,
                    ..capabilities
                },
            ),
            Err(
                ReplacementRenderPipelineCompileError::DescriptorLimitExceeded {
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE.as_raw(),
                    requested: 3,
                    supported: 2,
                }
            )
        );
        let overflow = BTreeMap::from([
            (1, declaration(vk::DescriptorType::STORAGE_IMAGE, u32::MAX)),
            (2, declaration(vk::DescriptorType::STORAGE_IMAGE, 1)),
        ]);
        assert_eq!(
            validate_descriptor_capabilities(&overflow, capabilities),
            Err(
                ReplacementRenderPipelineCompileError::DescriptorCountOverflow {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE.as_raw(),
                }
            )
        );
    }

    fn compile_glsl(path: &str, stage: &str) -> Option<Vec<u32>> {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join(path);
        let output = std::env::temp_dir().join(format!(
            "reims-vgpu-replacement-compile-{}-{stage}.spv",
            std::process::id()
        ));
        let status = std::process::Command::new("glslc")
            .args([format!("-fshader-stage={stage}"), "-O".to_owned()])
            .arg(source)
            .arg("-o")
            .arg(&output)
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        let bytes = std::fs::read(&output).ok()?;
        let _ = std::fs::remove_file(output);
        Some(
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
                .collect(),
        )
    }

    fn shader_family(
        words: Vec<u32>,
        stage: ReflectedShaderStage,
    ) -> (
        reims_vgpu_core::PreparedShaderFamily,
        Arc<crate::m2v_cache::ShaderVariant>,
    ) {
        let native = crate::m2v_cache::prepare_shader_words(words);
        let program = crate::m2v_cache::prepared_stage(&native);
        (
            reims_vgpu_core::PreparedShaderFamily::new(
                Arc::new(reims_vgpu_core::ShaderInterface {
                    stage,
                    bindings: Vec::new(),
                    local_size: None,
                    unsupported: None,
                }),
                reims_vgpu_core::PreparedShaderVariant {
                    program,
                    samplers: Arc::from([]),
                    declared_bindings: Arc::from([]),
                    descriptor_uses: Arc::from([]),
                    texture_uses: Arc::from([]),
                    storage_image_accesses: Arc::from([]),
                    buffer_binding_base: 0,
                    texture_binding_base: 0,
                    sampler_binding_base: 0,
                    word_count: u32::try_from(native.words.len()).expect("fixture word count"),
                },
            ),
            native,
        )
    }

    #[test]
    fn actual_family_compile_publishes_and_owns_every_handle_until_drop() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let Some(vertex_words) = compile_glsl("replacement_render_test.vert", "vertex") else {
            eprintln!("SKIP replacement render compile: glslc unavailable");
            return;
        };
        let Some(fragment_words) = compile_glsl("replacement_render_test.frag", "fragment") else {
            eprintln!("SKIP replacement render compile: glslc unavailable");
            return;
        };
        let context = match unsafe { DeviceContext::create() } {
            Ok(context) => Arc::new(SharedDeviceContext::new(context)),
            Err(error) => {
                eprintln!("SKIP replacement render compile: no device ({error})");
                return;
            }
        };
        let (vertex, _vertex_owner) = shader_family(vertex_words, ReflectedShaderStage::Vertex);
        let (fragment, _fragment_owner) =
            shader_family(fragment_words, ReflectedShaderStage::Fragment);
        let semantic = ResolvedRenderPipeline {
            pipeline_lifetime: None,
            desc: Arc::new(reims_vgpu_protocol::RenderPipelineDescriptor {
                raster_sample_count: 0,
                ..Default::default()
            }),
            vertex,
            fragment,
            bind_plan: Arc::new(reims_vgpu_core::VertexBindPlan::default()),
        };
        let compiler = ReplacementRenderPipelineCompiler::new(
            Arc::clone(&context),
            reims_vgpu_core::VulkanDeviceEpoch::new(reims_vgpu_protocol::VulkanDeviceEpochId::new(
                1,
            )),
        );
        let plan = ReplacementRenderPipelinePlan {
            program: reims_vgpu_core::PreparedRenderProgram {
                vertex: semantic.vertex.variant().program.clone(),
                fragment: semantic.fragment.variant().program.clone(),
            },
            vertex_buffers: Box::new([]),
            color_attachments: Box::new([]),
            depth_stencil_attachment: None,
            feedback_loop_aspects: vk::ImageAspectFlags::empty(),
            sample_count: vk::SampleCountFlags::TYPE_1,
            viewport_count: 1,
            static_state: Default::default(),
            depth_stencil: None,
        };
        let key = plan.variant_key();
        let family = ReplacementRenderPipelineFamily::default();
        let job = family.begin_compile(key.clone()).unwrap();
        let variant = compiler
            .compile_family_job(&family, job, &semantic, plan, None)
            .expect("actual attachmentless render variant must compile")
            .native;
        assert_ne!(variant.pipeline, vk::Pipeline::null());
        assert_ne!(variant.render_pass, vk::RenderPass::null());
        assert_ne!(variant.layout, vk::PipelineLayout::null());
        assert_ne!(
            variant.descriptor_set_layout,
            vk::DescriptorSetLayout::null()
        );
        assert!(matches!(
            family.readiness(&key),
            Ok(reims_vgpu_core::PipelineVariantReadiness::Ready(_))
        ));
        drop(variant);
        assert_eq!(Arc::strong_count(&context), 3);
        drop(family);
        drop(compiler);
        assert_eq!(Arc::strong_count(&context), 1);
    }
}
