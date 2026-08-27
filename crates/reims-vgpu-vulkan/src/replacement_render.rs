//! Vulkan projection for prepared canonical render draws.

use crate::{
    replacement_buffer_blit::{NativeBufferTarget, ReplacementBufferResolver},
    replacement_compute::NativeComputeDescriptor,
    replacement_image_state::{
        PreparedImageState, ReplacementImageKey, ReplacementImageStateError,
        ReplacementImageStateOwner, ReplacementImageUse,
    },
    replacement_image_transition::{
        resolve_image_transitions, ImageTransitionResolveError, NativeImageTarget,
        PreparedNativeImageState, ReplacementImageResolver,
    },
    replacement_sampler::ReplacementSamplerLease,
};
use ash::vk;
use reims_vgpu_core::{
    BackingView, PreparedRenderDispatch, RenderAttachmentClear, RenderAttachmentRole,
    RenderBindingClass, RenderBindingView, ResolvedRenderDispatch, ResolvedRenderDraw,
    ResolvedResourceCompletion, ViewRepresentation,
};
use reims_vgpu_protocol::{
    BackingId, CullMode, DepthClipMode, FillMode, PrimitiveTopology, RepresentationId, StoreAction,
};
use std::{collections::BTreeMap, sync::Arc};

const _: () = assert!(
    reims_vgpu_core::RENDER_INDIRECT_ARGUMENT_BYTES
        == std::mem::size_of::<vk::DrawIndirectCommand>() as u64
);
const _: () = assert!(
    reims_vgpu_core::RENDER_INDEXED_INDIRECT_ARGUMENT_BYTES
        == std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64
);

/// The dynamic states one compiled render pipeline declared.
///
/// Viewport and scissor are always dynamic here. The remaining three are
/// declared only when the resolved pipeline actually varies them, and a
/// recording may issue a `vkCmdSet*` for exactly the states named: setting one
/// the pipeline specified statically invalidates every draw that follows the
/// bind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplacementRenderDynamicStates {
    pub depth_bias: bool,
    pub stencil_reference: bool,
    pub blend_constants: bool,
}

impl ReplacementRenderDynamicStates {
    pub fn declarations(self) -> Vec<vk::DynamicState> {
        let mut states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        if self.depth_bias {
            states.push(vk::DynamicState::DEPTH_BIAS);
        }
        if self.stencil_reference {
            states.push(vk::DynamicState::STENCIL_REFERENCE);
        }
        if self.blend_constants {
            states.push(vk::DynamicState::BLEND_CONSTANTS);
        }
        states
    }
}

#[derive(Clone, Debug)]
pub struct ReplacementRenderPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub render_pass: vk::RenderPass,
    pub program: reims_vgpu_core::PreparedRenderProgram,
    pub vertex_buffers: Box<[reims_vgpu_core::ResolvedVertexBufferLayout]>,
    pub color_attachments: Box<[ReplacementRenderColorAttachment]>,
    pub depth_stencil_attachment: Option<ReplacementRenderDepthStencilAttachment>,
    pub feedback_loop_aspects: vk::ImageAspectFlags,
    pub color_input: bool,
    pub sample_count: vk::SampleCountFlags,
    pub viewport_count: u32,
    pub static_state: ReplacementRenderStaticState,
    /// The states this pipeline left dynamic. The recorder sets these and no
    /// others.
    pub dynamic_states: ReplacementRenderDynamicStates,
    pub depth_stencil:
        Option<reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::DepthStencilObject>>,
}

/// Handle-free native variant plan derived only from the resolved draw and the
/// immutable render-pipeline sample count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplacementRenderPipelinePlan {
    pub program: reims_vgpu_core::PreparedRenderProgram,
    pub vertex_buffers: Box<[reims_vgpu_core::ResolvedVertexBufferLayout]>,
    pub color_attachments: Box<[ReplacementRenderColorAttachment]>,
    pub depth_stencil_attachment: Option<ReplacementRenderDepthStencilAttachment>,
    pub feedback_loop_aspects: vk::ImageAspectFlags,
    pub sample_count: vk::SampleCountFlags,
    pub viewport_count: u32,
    pub static_state: ReplacementRenderStaticState,
    pub depth_stencil:
        Option<reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::DepthStencilObject>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementRenderPipelinePlanError {
    UnsupportedSampleCount(u32),
    AttachmentSampleCountMismatch(RenderAttachmentRole),
    ColorAttachmentGap(u32),
    ViewportCountOverflow,
}

/// Device-lifetime owner used to destroy one complete native render variant.
pub trait ReplacementRenderPipelineDevice: Send + Sync {
    fn destroy_render_pipeline_variant(&self, native: &ReplacementRenderPipeline);
}

impl ReplacementRenderPipelineDevice for crate::engine::context::SharedDeviceContext {
    fn destroy_render_pipeline_variant(&self, native: &ReplacementRenderPipeline) {
        unsafe {
            self.device.destroy_pipeline(native.pipeline, None);
            self.device.destroy_render_pass(native.render_pass, None);
            self.device.destroy_pipeline_layout(native.layout, None);
            self.device
                .destroy_descriptor_set_layout(native.descriptor_set_layout, None);
        }
    }
}

/// Owned native realization of one exact render-pipeline variant.
///
/// Retaining this value also retains the Vulkan device incarnation. Dropping
/// the semantic family therefore cannot invalidate a variant already carried
/// by accepted recording work, and the device is destroyed only after the
/// variant has destroyed its own handles.
pub struct ReplacementRenderPipelineVariant {
    native: ReplacementRenderPipeline,
    context: Option<Arc<dyn ReplacementRenderPipelineDevice>>,
}

impl std::fmt::Debug for ReplacementRenderPipelineVariant {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplacementRenderPipelineVariant")
            .field("native", &self.native)
            .field("owns_context", &self.context.is_some())
            .finish()
    }
}

impl ReplacementRenderPipelineVariant {
    pub const fn native(&self) -> &ReplacementRenderPipeline {
        &self.native
    }

    pub fn new(
        context: Arc<dyn ReplacementRenderPipelineDevice>,
        native: ReplacementRenderPipeline,
    ) -> Self {
        Self {
            native,
            context: Some(context),
        }
    }

    #[cfg(test)]
    pub(crate) fn synthetic(native: ReplacementRenderPipeline) -> Self {
        Self {
            native,
            context: None,
        }
    }
}

impl std::ops::Deref for ReplacementRenderPipelineVariant {
    type Target = ReplacementRenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.native
    }
}

impl Drop for ReplacementRenderPipelineVariant {
    fn drop(&mut self) {
        let Some(context) = self.context.as_ref() else {
            return;
        };
        context.destroy_render_pipeline_variant(&self.native);
    }
}

pub type RenderDepthStencilVariantKey = (
    Option<u16>,
    Option<u16>,
    Option<u16>,
    Option<u16>,
    u16,
    u16,
    u16,
    u16,
    u32,
);
pub type RenderColorVariantKey = (u16, Option<u16>, u16, u16, bool);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReplacementRenderPipelineVariantKey {
    pub vertex_program: reims_vgpu_protocol::PreparedShaderId,
    pub fragment_program: reims_vgpu_protocol::PreparedShaderId,
    pub topology: u32,
    pub cull_mode: u8,
    pub front_face_ccw: bool,
    pub fill_mode: u8,
    pub line_width_bits: u32,
    pub depth_clip_mode: u8,
    pub depth_bias_enabled: bool,
    pub vertex_buffers: Box<[reims_vgpu_core::ResolvedVertexBufferLayout]>,
    pub color_attachments: Box<[RenderColorVariantKey]>,
    pub depth_stencil_attachment: Option<RenderDepthStencilVariantKey>,
    pub feedback_loop_aspects: u32,
    pub sample_count: u32,
    pub viewport_count: u32,
    pub depth_stencil:
        Option<reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::DepthStencilObject>>,
}

pub type ReplacementRenderPipelineVariants<E> = reims_vgpu_core::PipelineVariantFamily<
    ReplacementRenderPipelineVariantKey,
    ReplacementRenderPipelineVariant,
    E,
>;

/// Thread-safe native variant family retained as the compiled value of one
/// semantic render-pipeline generation.
#[derive(Debug)]
pub struct ReplacementRenderPipelineFamily<E> {
    variants: parking_lot::Mutex<ReplacementRenderPipelineVariants<E>>,
}

impl<E> Default for ReplacementRenderPipelineFamily<E> {
    fn default() -> Self {
        Self {
            variants: parking_lot::Mutex::new(ReplacementRenderPipelineVariants::default()),
        }
    }
}

impl<E> ReplacementRenderPipelineFamily<E> {
    pub fn readiness_or_begin(
        &self,
        key: ReplacementRenderPipelineVariantKey,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> reims_vgpu_core::PipelineVariantAdmission<
        ReplacementRenderPipelineVariantKey,
        ReplacementRenderPipelineVariant,
        E,
    >
    where
        E: Clone,
    {
        self.variants.lock().readiness_or_begin(key, transaction)
    }

    pub fn begin_compile(
        &self,
        key: ReplacementRenderPipelineVariantKey,
    ) -> Result<
        reims_vgpu_core::PipelineVariantCompileJob<ReplacementRenderPipelineVariantKey>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    > {
        self.variants.lock().begin_compile(key)
    }

    pub fn compile_complete(
        &self,
        job: reims_vgpu_core::PipelineVariantCompileJob<ReplacementRenderPipelineVariantKey>,
        native: ReplacementRenderPipelineVariant,
    ) -> Result<
        reims_vgpu_core::PipelineVariantPublication<ReplacementRenderPipelineVariant>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    > {
        self.variants.lock().compile_complete(job, native)
    }

    pub fn refuse(
        &self,
        job: reims_vgpu_core::PipelineVariantCompileJob<ReplacementRenderPipelineVariantKey>,
        reason: E,
    ) -> Result<
        Box<[reims_vgpu_protocol::TransactionId]>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    > {
        self.variants.lock().refuse(job, reason)
    }

    pub fn readiness(
        &self,
        key: &ReplacementRenderPipelineVariantKey,
    ) -> Result<
        reims_vgpu_core::PipelineVariantReadiness<ReplacementRenderPipelineVariant, E>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    >
    where
        E: Clone,
    {
        self.variants.lock().readiness(key)
    }

    pub fn census(&self) -> reims_vgpu_core::PipelineVariantCensus {
        self.variants.lock().census()
    }

    pub fn retire_all_waiters(
        &self,
    ) -> reims_vgpu_core::RetiredPipelineVariantWaiters<ReplacementRenderPipelineVariantKey> {
        self.variants.lock().retire_all_waiters()
    }
}

/// Acquire one exact native variant while preserving the semantic pipeline and
/// device-epoch lease already accepted for the family.
#[derive(Debug)]
pub enum ReplacementRenderPipelineVariantReadiness<E> {
    Ready(
        reims_vgpu_core::ReadyPipelineLease<
            reims_vgpu_protocol::RenderPipelineObject,
            ReplacementRenderPipelineVariant,
        >,
    ),
    Pending,
    Refused(E),
}

pub fn acquire_render_pipeline_variant<E: Clone>(
    family: &reims_vgpu_core::ReadyPipelineLease<
        reims_vgpu_protocol::RenderPipelineObject,
        ReplacementRenderPipelineFamily<E>,
    >,
    key: &ReplacementRenderPipelineVariantKey,
) -> Result<
    ReplacementRenderPipelineVariantReadiness<E>,
    reims_vgpu_core::PipelineVariantLifecycleError,
> {
    Ok(match family.native.readiness(key)? {
        reims_vgpu_core::PipelineVariantReadiness::Pending => {
            ReplacementRenderPipelineVariantReadiness::Pending
        }
        reims_vgpu_core::PipelineVariantReadiness::Refused(reason) => {
            ReplacementRenderPipelineVariantReadiness::Refused(reason)
        }
        reims_vgpu_core::PipelineVariantReadiness::Ready(native) => {
            ReplacementRenderPipelineVariantReadiness::Ready(reims_vgpu_core::ReadyPipelineLease {
                pipeline: family.pipeline,
                native_object: family.native_object.clone(),
                native,
            })
        }
    })
}

impl ReplacementRenderPipeline {
    pub fn variant_key(&self) -> ReplacementRenderPipelineVariantKey {
        render_pipeline_variant_key(ReplacementRenderPipelineKeyInput {
            program: &self.program,
            vertex_buffers: &self.vertex_buffers,
            color_attachments: &self.color_attachments,
            depth_stencil_attachment: self.depth_stencil_attachment,
            feedback_loop_aspects: self.feedback_loop_aspects,
            sample_count: self.sample_count,
            viewport_count: self.viewport_count,
            static_state: self.static_state,
            depth_stencil: self.depth_stencil,
        })
    }
}

impl ReplacementRenderPipelinePlan {
    pub fn variant_key(&self) -> ReplacementRenderPipelineVariantKey {
        render_pipeline_variant_key(ReplacementRenderPipelineKeyInput {
            program: &self.program,
            vertex_buffers: &self.vertex_buffers,
            color_attachments: &self.color_attachments,
            depth_stencil_attachment: self.depth_stencil_attachment,
            feedback_loop_aspects: self.feedback_loop_aspects,
            sample_count: self.sample_count,
            viewport_count: self.viewport_count,
            static_state: self.static_state,
            depth_stencil: self.depth_stencil,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementRenderColorAttachment {
    pub format: u16,
    pub resolve_format: Option<u16>,
    pub load: reims_vgpu_protocol::LoadAction,
    pub store: StoreAction,
    pub feedback_loop: bool,
    pub input_attachment: bool,
}

/// The layout one render-pass attachment is declared in.
///
/// This is the only derivation of that layout. The compiled pass names it in
/// `initialLayout`, `finalLayout` and every attachment reference, and the
/// image-state transition that runs before the pass delivers it -- so the two
/// cannot be computed from different inputs and disagree. A disagreement does
/// not stop at one pass: `finalLayout` then leaves the image in a layout the
/// layout record never committed, and the next recording's barrier names a
/// stale `oldLayout` for as long as the image lives.
pub const fn render_attachment_layout(
    role: RenderAttachmentRole,
    feedback_loop: bool,
    input_attachment: bool,
) -> vk::ImageLayout {
    if feedback_loop {
        return vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
    }
    match role {
        // A subpass input read and a colour write of one attachment is the one
        // pair Vulkan expresses with a single general layout rather than an
        // attachment-optimal one.
        RenderAttachmentRole::Color(_) if input_attachment => vk::ImageLayout::GENERAL,
        RenderAttachmentRole::Color(_) => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        RenderAttachmentRole::Depth | RenderAttachmentRole::Stencil => {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
    }
}

/// The image usage one render-pass attachment must have been created with.
pub const fn render_attachment_usage(
    role: RenderAttachmentRole,
    feedback_loop: bool,
    input_attachment: bool,
) -> vk::ImageUsageFlags {
    let role_usage = match role {
        RenderAttachmentRole::Color(_) => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        RenderAttachmentRole::Depth | RenderAttachmentRole::Stencil => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        }
    };
    let mut usage = role_usage;
    if feedback_loop {
        usage = vk::ImageUsageFlags::from_raw(
            usage.as_raw() | vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT.as_raw(),
        );
    }
    if input_attachment {
        usage = vk::ImageUsageFlags::from_raw(
            usage.as_raw() | vk::ImageUsageFlags::INPUT_ATTACHMENT.as_raw(),
        );
    }
    usage
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementRenderDepthStencilAttachment {
    pub depth_format: Option<u16>,
    pub stencil_format: Option<u16>,
    pub depth_resolve_format: Option<u16>,
    pub stencil_resolve_format: Option<u16>,
    pub depth_load: reims_vgpu_protocol::LoadAction,
    pub depth_store: StoreAction,
    pub stencil_load: reims_vgpu_protocol::LoadAction,
    pub stencil_store: StoreAction,
    pub feedback_loop_aspects: vk::ImageAspectFlags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementRenderStaticState {
    pub topology: PrimitiveTopology,
    pub cull_mode: CullMode,
    pub front_face_ccw: bool,
    pub fill_mode: FillMode,
    pub line_width_bits: u32,
    pub depth_clip_mode: DepthClipMode,
    pub depth_bias_enabled: bool,
}

impl Default for ReplacementRenderStaticState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::Triangle,
            cull_mode: CullMode::None,
            front_face_ccw: false,
            fill_mode: FillMode::Fill,
            line_width_bits: 1.0f32.to_bits(),
            depth_clip_mode: DepthClipMode::Clip,
            depth_bias_enabled: false,
        }
    }
}

struct ReplacementRenderPipelineKeyInput<'a> {
    program: &'a reims_vgpu_core::PreparedRenderProgram,
    vertex_buffers: &'a [reims_vgpu_core::ResolvedVertexBufferLayout],
    color_attachments: &'a [ReplacementRenderColorAttachment],
    depth_stencil_attachment: Option<ReplacementRenderDepthStencilAttachment>,
    feedback_loop_aspects: vk::ImageAspectFlags,
    sample_count: vk::SampleCountFlags,
    viewport_count: u32,
    static_state: ReplacementRenderStaticState,
    depth_stencil: Option<reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::DepthStencilObject>>,
}

fn render_pipeline_variant_key(
    input: ReplacementRenderPipelineKeyInput<'_>,
) -> ReplacementRenderPipelineVariantKey {
    let ReplacementRenderPipelineKeyInput {
        program,
        vertex_buffers,
        color_attachments,
        depth_stencil_attachment,
        feedback_loop_aspects,
        sample_count,
        viewport_count,
        static_state,
        depth_stencil,
    } = input;
    ReplacementRenderPipelineVariantKey {
        vertex_program: program.vertex.id,
        fragment_program: program.fragment.id,
        vertex_buffers: vertex_buffers.into(),
        topology: static_state.topology.guest_ordinal(),
        cull_mode: match static_state.cull_mode {
            CullMode::None => 0,
            CullMode::Front => 1,
            CullMode::Back => 2,
        },
        front_face_ccw: static_state.front_face_ccw,
        fill_mode: match static_state.fill_mode {
            FillMode::Fill => 0,
            FillMode::Lines => 1,
        },
        line_width_bits: static_state.line_width_bits,
        depth_clip_mode: match static_state.depth_clip_mode {
            DepthClipMode::Clip => 0,
            DepthClipMode::Clamp => 1,
        },
        depth_bias_enabled: static_state.depth_bias_enabled,
        color_attachments: color_attachments
            .iter()
            .map(|attachment| {
                (
                    attachment.format,
                    attachment.resolve_format,
                    attachment.load.guest_ordinal(),
                    attachment.store.guest_ordinal(),
                    attachment.feedback_loop,
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        depth_stencil_attachment: depth_stencil_attachment.map(|attachment| {
            (
                attachment.depth_format,
                attachment.stencil_format,
                attachment.depth_resolve_format,
                attachment.stencil_resolve_format,
                attachment.depth_load.guest_ordinal(),
                attachment.depth_store.guest_ordinal(),
                attachment.stencil_load.guest_ordinal(),
                attachment.stencil_store.guest_ordinal(),
                attachment.feedback_loop_aspects.as_raw(),
            )
        }),
        feedback_loop_aspects: feedback_loop_aspects.as_raw(),
        sample_count: sample_count.as_raw(),
        viewport_count,
        depth_stencil,
    }
}

/// Derive the exact native render-variant inputs without observing or creating
/// Vulkan handles. The immutable pipeline descriptor supplies the sample count
/// even for attachmentless draws.
pub fn resolve_render_pipeline_plan(
    operation: &ResolvedRenderDispatch,
    pipeline_sample_count: u32,
) -> Result<ReplacementRenderPipelinePlan, ReplacementRenderPipelinePlanError> {
    let sample_count = match pipeline_sample_count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        value => {
            return Err(ReplacementRenderPipelinePlanError::UnsupportedSampleCount(
                value,
            ))
        }
    };
    let mut feedback_loop_aspects = vk::ImageAspectFlags::empty();
    let mut colors = operation
        .attachments
        .iter()
        .filter_map(|attachment| match attachment.role {
            RenderAttachmentRole::Color(index) => Some((index, attachment)),
            RenderAttachmentRole::Depth | RenderAttachmentRole::Stencil => None,
        })
        .collect::<Vec<_>>();
    colors.sort_unstable_by_key(|(index, _)| *index);
    let mut color_attachments = Vec::with_capacity(colors.len());
    for (expected, (index, attachment)) in colors.into_iter().enumerate() {
        let expected = u32::try_from(expected)
            .map_err(|_| ReplacementRenderPipelinePlanError::ColorAttachmentGap(u32::MAX))?;
        if index != expected {
            return Err(ReplacementRenderPipelinePlanError::ColorAttachmentGap(
                expected,
            ));
        }
        if attachment.sample_count != pipeline_sample_count {
            return Err(
                ReplacementRenderPipelinePlanError::AttachmentSampleCountMismatch(attachment.role),
            );
        }
        if attachment.feedback_loop {
            feedback_loop_aspects |= vk::ImageAspectFlags::COLOR;
        }
        color_attachments.push(ReplacementRenderColorAttachment {
            format: attachment.pixel_format,
            resolve_format: attachment
                .resolve
                .as_ref()
                .map(|resolve| resolve.pixel_format),
            load: attachment.load,
            store: attachment.store,
            feedback_loop: attachment.feedback_loop,
            input_attachment: attachment.input_attachment,
        });
    }
    let depth = operation
        .attachments
        .iter()
        .find(|attachment| attachment.role == RenderAttachmentRole::Depth);
    let stencil = operation
        .attachments
        .iter()
        .find(|attachment| attachment.role == RenderAttachmentRole::Stencil);
    for attachment in [depth, stencil].into_iter().flatten() {
        if attachment.sample_count != pipeline_sample_count {
            return Err(
                ReplacementRenderPipelinePlanError::AttachmentSampleCountMismatch(attachment.role),
            );
        }
        if attachment.feedback_loop {
            feedback_loop_aspects |= match attachment.role {
                RenderAttachmentRole::Depth => vk::ImageAspectFlags::DEPTH,
                RenderAttachmentRole::Stencil => vk::ImageAspectFlags::STENCIL,
                RenderAttachmentRole::Color(_) => unreachable!(),
            };
        }
    }
    let depth_stencil_attachment =
        (depth.is_some() || stencil.is_some()).then(|| ReplacementRenderDepthStencilAttachment {
            depth_format: depth.map(|attachment| attachment.pixel_format),
            stencil_format: stencil.map(|attachment| attachment.pixel_format),
            depth_resolve_format: depth.and_then(|attachment| {
                attachment
                    .resolve
                    .as_ref()
                    .map(|resolve| resolve.pixel_format)
            }),
            stencil_resolve_format: stencil.and_then(|attachment| {
                attachment
                    .resolve
                    .as_ref()
                    .map(|resolve| resolve.pixel_format)
            }),
            depth_load: depth.map_or(reims_vgpu_protocol::LoadAction::DontCare, |attachment| {
                attachment.load
            }),
            depth_store: depth.map_or(StoreAction::DontCare, |attachment| attachment.store),
            stencil_load: stencil.map_or(reims_vgpu_protocol::LoadAction::DontCare, |attachment| {
                attachment.load
            }),
            stencil_store: stencil.map_or(StoreAction::DontCare, |attachment| attachment.store),
            feedback_loop_aspects: feedback_loop_aspects
                & (vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL),
        });
    let topology = match operation.draw {
        ResolvedRenderDraw::Direct { topology, .. }
        | ResolvedRenderDraw::Indexed { topology, .. }
        | ResolvedRenderDraw::Indirect { topology }
        | ResolvedRenderDraw::IndexedIndirect { topology, .. } => topology,
    };
    let viewport_count = operation
        .raster
        .viewports
        .len()
        .max(operation.raster.scissors.len())
        .max(1);
    let viewport_count = u32::try_from(viewport_count)
        .map_err(|_| ReplacementRenderPipelinePlanError::ViewportCountOverflow)?;
    Ok(ReplacementRenderPipelinePlan {
        program: operation.program.clone(),
        vertex_buffers: operation.vertex_buffers.clone(),
        color_attachments: color_attachments.into_boxed_slice(),
        depth_stencil_attachment,
        feedback_loop_aspects,
        sample_count,
        viewport_count,
        static_state: ReplacementRenderStaticState {
            topology,
            cull_mode: operation.raster.cull_mode,
            front_face_ccw: operation.raster.front_face_ccw,
            fill_mode: operation.raster.fill_mode,
            line_width_bits: operation.raster.line_width_bits,
            depth_clip_mode: operation.raster.depth_clip_mode,
            depth_bias_enabled: operation.raster.depth_bias_bits.is_some(),
        },
        depth_stencil: operation.depth_stencil,
    })
}

pub trait ReplacementRenderResolver: ReplacementBufferResolver + ReplacementImageResolver {
    fn resolve_sampler(
        &self,
        pipeline: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::RenderPipelineObject>,
        sampler: &reims_vgpu_core::SamplerResource,
    ) -> Option<ReplacementSamplerLease>;
    fn max_storage_buffer_range(&self) -> u64;
    fn min_storage_buffer_offset_alignment(&self) -> u64;
    fn max_viewports(&self) -> u32;
    fn precise_occlusion_queries(&self) -> bool;
    fn null_descriptors(&self) -> bool;
    fn resolve_attachment_view(
        &self,
        image: ReplacementImageKey,
        region: reims_vgpu_core::BackingRegion,
    ) -> Option<NativeImageTarget> {
        match region {
            reims_vgpu_core::BackingRegion::Whole => self.resolve_image(image),
            region => self.resolve_image_view(
                image,
                crate::replacement_image_transition::exact_image_subresource_range(region)?,
            ),
        }
    }
}

mod image_bindings_sealed {
    pub trait Sealed {}
    impl Sealed for () {}
    impl Sealed for reims_vgpu_core::ResolvedRenderDispatch {}
}

type RenderImageBinding = (
    BackingId,
    vk::ImageUsageFlags,
    vk::ImageLayout,
    Option<RenderAttachmentRole>,
    bool,
);

pub trait ReplacementRenderImageBindings: image_bindings_sealed::Sealed {
    fn render_image_bindings(&self) -> Box<[RenderImageBinding]>;
}

impl ReplacementRenderImageBindings for () {
    fn render_image_bindings(&self) -> Box<[RenderImageBinding]> {
        Box::new([])
    }
}

impl ReplacementRenderImageBindings for ResolvedRenderDispatch {
    fn render_image_bindings(&self) -> Box<[RenderImageBinding]> {
        // The layouts here are the ones the compiled render pass declares for
        // the same attachments, from the same derivation, and the pass is the
        // encoder's rather than this draw's -- so a feedback loop or an input
        // read anywhere in the encoder decides the layout for every draw in it.
        fn attachment_binding(
            backing: BackingId,
            role: RenderAttachmentRole,
            feedback_loop: bool,
            input_attachment: bool,
            is_resolve: bool,
        ) -> RenderImageBinding {
            (
                backing,
                render_attachment_usage(role, feedback_loop, input_attachment),
                render_attachment_layout(role, feedback_loop, input_attachment),
                Some(role),
                is_resolve,
            )
        }

        self.attachments
            .iter()
            .flat_map(|attachment| {
                std::iter::once(attachment_binding(
                    attachment.backing,
                    attachment.role,
                    attachment.feedback_loop,
                    attachment.input_attachment,
                    false,
                ))
                .chain(attachment.resolve.iter().map(|resolve| {
                    // A resolve target is written by the pass and never read
                    // by it, so it is in no feedback loop and is no input
                    // attachment however the attachment it resolves is read.
                    attachment_binding(resolve.backing, attachment.role, false, false, true)
                }))
            })
            .chain(
                self.resources
                    .iter()
                    .filter_map(|resource| match resource.class {
                        RenderBindingClass::SampledImage => Some((
                            resource.backing,
                            vk::ImageUsageFlags::SAMPLED,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            None,
                            false,
                        )),
                        RenderBindingClass::StorageImage => Some((
                            resource.backing,
                            vk::ImageUsageFlags::STORAGE,
                            vk::ImageLayout::GENERAL,
                            None,
                            false,
                        )),
                        RenderBindingClass::VertexBuffer
                        | RenderBindingClass::IndexBuffer
                        | RenderBindingClass::IndirectBuffer
                        | RenderBindingClass::StorageBuffer => None,
                    }),
            )
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

pub fn derive_render_image_uses<NativePipeline, Operation: ReplacementRenderImageBindings>(
    prepared: &PreparedRenderDispatch<NativePipeline, Operation>,
) -> Result<Box<[ReplacementImageUse]>, RenderImageStateError> {
    let representations = prepared.representations();
    let mut images = BTreeMap::<
        ReplacementImageKey,
        (
            vk::ImageUsageFlags,
            vk::ImageLayout,
            Option<RenderAttachmentRole>,
            bool,
        ),
    >::new();
    let bindings = prepared.operation().render_image_bindings();
    for (backing, usage, layout, role, is_resolve) in bindings.into_vec() {
        // Every render image binding names the backing's image view; a buffer
        // view of the same bytes has no image state.
        let image = ReplacementImageKey {
            backing,
            representation: ViewRepresentation::lookup(
                representations,
                backing,
                BackingView::Image,
            )
            .ok_or(RenderImageStateError::RepresentationUseMismatch(backing))?,
        };
        match images.get_mut(&image) {
            Some((found_usage, found_layout, found_role, found_is_resolve)) => {
                if (*found_usage).intersects(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                ) || usage.intersects(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                ) {
                    let depth_stencil_pair = *found_is_resolve == is_resolve
                        && matches!(
                            (*found_role, role),
                            (
                                Some(RenderAttachmentRole::Depth),
                                Some(RenderAttachmentRole::Stencil)
                            ) | (
                                Some(RenderAttachmentRole::Stencil),
                                Some(RenderAttachmentRole::Depth)
                            )
                        );
                    let sampled_attachment_feedback = found_role.is_some() != role.is_some()
                        && if found_role.is_some() {
                            usage == vk::ImageUsageFlags::SAMPLED
                        } else {
                            *found_usage == vk::ImageUsageFlags::SAMPLED
                        };
                    if sampled_attachment_feedback {
                        // The attachment carries the encoder-wide feedback
                        // layout already; this pair only adds the usage the
                        // sampled read needs. Re-deriving the layout here
                        // would be the pass's rule written a second time, so
                        // an attachment that does not already carry it is a
                        // refusal rather than a local repair.
                        let attachment_layout = if found_role.is_some() {
                            *found_layout
                        } else {
                            layout
                        };
                        if attachment_layout
                            != vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                        {
                            return Err(RenderImageStateError::FeedbackRepresentationRequired(
                                image,
                            ));
                        }
                        *found_usage |= usage;
                        *found_layout = attachment_layout;
                        continue;
                    }
                    if !depth_stencil_pair {
                        return Err(RenderImageStateError::FeedbackRepresentationRequired(image));
                    }
                }
                *found_usage |= usage;
                // A combined depth/stencil attachment is one pass attachment
                // with one layout, and the pass unions the feedback-loop
                // aspects of both -- so a loop on either aspect is the layout
                // for both.
                if *found_layout == vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                    || layout == vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                {
                    *found_layout = vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
                } else if layout == vk::ImageLayout::GENERAL {
                    *found_layout = layout;
                }
            }
            None => {
                images.insert(image, (usage, layout, role, is_resolve));
            }
        }
    }
    Ok(images
        .into_iter()
        .map(
            |(image, (required_usage, layout, _, _))| ReplacementImageUse {
                image,
                required_usage,
                use_layout: layout,
                final_layout: layout,
            },
        )
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

pub fn prepare_render_image_state<NativePipeline>(
    owner: &mut ReplacementImageStateOwner,
    prepared: &PreparedRenderDispatch<NativePipeline>,
    queue_family: u32,
) -> Result<PreparedImageState, RenderImageStateError> {
    let uses = derive_render_image_uses(prepared)?;
    owner
        .prepare_operation(
            prepared.transaction(),
            prepared.operation_index(),
            queue_family,
            uses,
        )
        .map_err(RenderImageStateError::State)
}

pub fn validate_render_image_state<NativePipeline, Operation: ReplacementRenderImageBindings>(
    prepared: &PreparedRenderDispatch<NativePipeline, Operation>,
    state: &PreparedImageState,
) -> Result<(), RenderImageStateError> {
    let uses = derive_render_image_uses(prepared)?;
    if state.transaction() != prepared.transaction()
        || state.operation_index() != Some(prepared.operation_index())
        || state.transitions().len() != uses.len()
    {
        return Err(RenderImageStateError::StateOperationMismatch);
    }
    for use_ in uses {
        let transition = state
            .transitions()
            .iter()
            .find(|transition| transition.image == use_.image)
            .ok_or(RenderImageStateError::StateOperationMismatch)?;
        if transition.required_usage != use_.required_usage
            || transition.use_layout != use_.use_layout
            || transition.final_layout != use_.final_layout
        {
            return Err(RenderImageStateError::StateOperationMismatch);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderImageStateError {
    RepresentationUseMismatch(BackingId),
    FeedbackRepresentationRequired(ReplacementImageKey),
    StateOperationMismatch,
    State(ReplacementImageStateError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRenderBufferBinding {
    pub binding: u32,
    pub buffer: vk::Buffer,
    pub offset: u64,
}

/// One dynamic-state value a draw recording issues after binding its pipeline.
///
/// A `vkCmdSet*` for a state the bound pipeline specified statically
/// invalidates every draw that follows, so the settings a dispatch emits are
/// derived from the pipeline's own declarations rather than from whether the
/// resolved draw happened to carry a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReplacementRenderDynamicSetting {
    DepthBias([f32; 3]),
    BlendConstants([f32; 4]),
    StencilReference([u32; 2]),
}

impl ReplacementRenderDynamicSetting {
    pub const fn state(self) -> vk::DynamicState {
        match self {
            Self::DepthBias(_) => vk::DynamicState::DEPTH_BIAS,
            Self::BlendConstants(_) => vk::DynamicState::BLEND_CONSTANTS,
            Self::StencilReference(_) => vk::DynamicState::STENCIL_REFERENCE,
        }
    }
}

pub fn dynamic_settings(native: &NativeRenderDispatch) -> Vec<ReplacementRenderDynamicSetting> {
    let declared = native.pipeline.native().dynamic_states;
    let mut settings = Vec::new();
    if let Some(bias) = native.depth_bias.filter(|_| declared.depth_bias) {
        settings.push(ReplacementRenderDynamicSetting::DepthBias(bias));
    }
    if let Some(color) = native.blend_color.filter(|_| declared.blend_constants) {
        settings.push(ReplacementRenderDynamicSetting::BlendConstants(color));
    }
    if declared.stencil_reference {
        settings.push(ReplacementRenderDynamicSetting::StencilReference(
            native.stencil_reference,
        ));
    }
    settings
}

#[derive(Clone, Debug)]
pub struct NativeRenderDispatch {
    pub pipeline: Arc<ReplacementRenderPipelineVariant>,
    pub descriptors: Box<[NativeComputeDescriptor]>,
    pub sampler_leases: Box<[ReplacementSamplerLease]>,
    pub descriptor_counts: Box<[(vk::DescriptorType, u32)]>,
    pub vertex_buffers: Box<[NativeRenderBufferBinding]>,
    pub index_buffer: Option<(vk::Buffer, u64, vk::IndexType)>,
    pub indirect_buffer: Option<(vk::Buffer, u64)>,
    pub attachment_views: Box<[vk::ImageView]>,
    pub clear_values: Box<[NativeRenderClear]>,
    pub extent: vk::Extent2D,
    pub viewports: Box<[vk::Viewport]>,
    pub scissors: Box<[vk::Rect2D]>,
    pub depth_bias: Option<[f32; 3]>,
    pub blend_color: Option<[f32; 4]>,
    pub stencil_reference: [u32; 2],
    pub visibility: Option<NativeRenderVisibility>,
    pub begins_native_pass: bool,
    pub ends_native_pass: bool,
    pub draw: ResolvedRenderDraw,
    pub image_state: PreparedNativeImageState,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeRenderVisibility {
    pub mode: reims_vgpu_protocol::VisibilityResultMode,
    pub buffer: vk::Buffer,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeRenderClear {
    Color([u32; 4]),
    DepthStencil { depth_bits: u32, stencil: u32 },
}

#[derive(Clone, Debug)]
pub struct ReplacementRenderProgram<Operation = ResolvedRenderDispatch> {
    index: usize,
    transaction: reims_vgpu_protocol::TransactionId,
    operation: Operation,
    native: NativeRenderDispatch,
    backings: Box<[BackingId]>,
    completions: Box<[ResolvedResourceCompletion]>,
}

pub fn resolve_exec_render_programs<
    Compute: crate::replacement_compute::ReplacementComputeImageBindings,
    NativeCompute,
>(
    resources: &reims_vgpu_core::PreparedExecResources<
        Compute,
        NativeCompute,
        ResolvedRenderDispatch,
        ReplacementRenderPipelineVariant,
    >,
    states: Option<&crate::replacement_image_state::PreparedImageStateBatch>,
    resolver: &impl ReplacementRenderResolver,
) -> Result<Box<[ReplacementRenderProgram]>, RenderExecProgramError> {
    let render_uses = resources
        .inputs()
        .render_dispatches
        .iter()
        .map(derive_render_image_uses)
        .collect::<Result<Vec<_>, _>>()?;
    if render_uses.iter().any(|uses| !uses.is_empty()) && states.is_none() {
        return Err(RenderExecProgramError::ImageStateBatchMissing);
    }
    if let Some(states) = states {
        crate::replacement_exec_image::validate_exec_image_states(resources, states)
            .map_err(RenderExecProgramError::ExecImageState)?;
    }
    let mut programs = resources
        .inputs()
        .render_dispatches
        .iter()
        .map(|prepared| {
            let state = states
                .and_then(|states| {
                    states
                        .operations()
                        .iter()
                        .find(|state| state.operation_index() == Some(prepared.operation_index()))
                })
                .ok_or(RenderExecProgramError::ImageStateMissing(
                    prepared.operation_index(),
                ))?;
            ReplacementRenderProgram::resolve(prepared, state, resolver)
                .map_err(RenderExecProgramError::Record)
        })
        .collect::<Result<Vec<_>, _>>()?;
    join_native_render_passes(&mut programs)?;
    Ok(programs.into_boxed_slice())
}

fn join_native_render_passes(
    programs: &mut [ReplacementRenderProgram],
) -> Result<(), RenderExecProgramError> {
    for index in 0..programs.len().saturating_sub(1) {
        let continues_encoder = programs[index].index + 1 == programs[index + 1].index
            && !programs[index].operation.ends_encoder
            && !programs[index + 1].operation.begins_encoder;
        if !continues_encoder {
            continue;
        }
        let compatible = render_passes_compatible(
            programs[index].native.pipeline.native(),
            programs[index + 1].native.pipeline.native(),
        ) && programs[index].native.attachment_views
            == programs[index + 1].native.attachment_views
            && programs[index].native.extent == programs[index + 1].native.extent;
        if !compatible {
            return Err(RenderExecProgramError::NativePassReopenRequired(
                programs[index + 1].index,
            ));
        }
        programs[index].native.ends_native_pass = false;
        programs[index + 1].native.begins_native_pass = false;
    }
    Ok(())
}

pub(crate) fn render_passes_compatible(
    first: &ReplacementRenderPipeline,
    second: &ReplacementRenderPipeline,
) -> bool {
    first.sample_count == second.sample_count
        && first.color_input == second.color_input
        && first.color_attachments.len() == second.color_attachments.len()
        && first
            .color_attachments
            .iter()
            .zip(&second.color_attachments)
            .all(|(first, second)| {
                first.format == second.format && first.resolve_format == second.resolve_format
            })
        && match (
            first.depth_stencil_attachment,
            second.depth_stencil_attachment,
        ) {
            (None, None) => true,
            (Some(first), Some(second)) => {
                first.depth_format == second.depth_format
                    && first.stencil_format == second.stencil_format
                    && first.depth_resolve_format == second.depth_resolve_format
                    && first.stencil_resolve_format == second.stencil_resolve_format
            }
            _ => false,
        }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderExecProgramError {
    ImageState(RenderImageStateError),
    ImageStateBatchMissing,
    ImageStateMissing(usize),
    ExecImageState(crate::replacement_exec_image::ExecImageStateError),
    Record(RenderRecordError),
    NativePassReopenRequired(usize),
}

impl RenderExecProgramError {
    /// Whether a later packet could make this program record.
    ///
    /// See [`crate::replacement_image_transition::TextureBindingViewDecline::is_unimplemented`].
    pub const fn is_terminal_refusal(&self) -> bool {
        matches!(
            self,
            Self::Record(RenderRecordError::UnknownImageView { reason, .. }) if reason.is_unimplemented()
        )
    }
}

impl From<RenderImageStateError> for RenderExecProgramError {
    fn from(reason: RenderImageStateError) -> Self {
        Self::ImageState(reason)
    }
}

impl<Operation> ReplacementRenderProgram<Operation> {
    pub const fn index(&self) -> usize {
        self.index
    }
    pub const fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        self.transaction
    }
    pub const fn operation(&self) -> &Operation {
        &self.operation
    }
    pub const fn native(&self) -> &NativeRenderDispatch {
        &self.native
    }
    pub const fn backings(&self) -> &[BackingId] {
        &self.backings
    }
    pub const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }
}

#[cfg(test)]
impl<Operation> ReplacementRenderProgram<Operation> {
    pub(crate) fn synthetic(
        index: usize,
        transaction: reims_vgpu_protocol::TransactionId,
        operation: Operation,
        native: NativeRenderDispatch,
        backings: impl Into<Box<[BackingId]>>,
        completions: impl Into<Box<[ResolvedResourceCompletion]>>,
    ) -> Self {
        Self {
            index,
            transaction,
            operation,
            native,
            backings: backings.into(),
            completions: completions.into(),
        }
    }
}

impl ReplacementRenderProgram<ResolvedRenderDispatch> {
    pub fn resolve(
        prepared: &PreparedRenderDispatch<ReplacementRenderPipelineVariant>,
        state: &PreparedImageState,
        resolver: &impl ReplacementRenderResolver,
    ) -> Result<Self, RenderRecordError> {
        validate_render_image_state(prepared, state).map_err(RenderRecordError::ImageState)?;
        if !prepared
            .pipeline()
            .native_object
            .native_handles_are_usable()
        {
            return Err(RenderRecordError::DeviceEpochLost);
        }
        let pipeline = Arc::clone(&prepared.pipeline().native);
        if pipeline.pipeline == vk::Pipeline::null()
            || pipeline.layout == vk::PipelineLayout::null()
            || pipeline.render_pass == vk::RenderPass::null()
            || pipeline.descriptor_set_layout == vk::DescriptorSetLayout::null()
        {
            return Err(RenderRecordError::InvalidPipeline);
        }
        let topology = match prepared.operation().draw {
            ResolvedRenderDraw::Direct { topology, .. }
            | ResolvedRenderDraw::Indexed { topology, .. }
            | ResolvedRenderDraw::Indirect { topology }
            | ResolvedRenderDraw::IndexedIndirect { topology, .. } => topology,
        };
        let raster = &prepared.operation().raster;
        let static_state = ReplacementRenderStaticState {
            topology,
            cull_mode: raster.cull_mode,
            front_face_ccw: raster.front_face_ccw,
            fill_mode: raster.fill_mode,
            line_width_bits: raster.line_width_bits,
            depth_clip_mode: raster.depth_clip_mode,
            depth_bias_enabled: raster.depth_bias_bits.is_some(),
        };
        if pipeline.static_state != static_state {
            return Err(RenderRecordError::PipelineRasterStateMismatch);
        }
        if pipeline.program != prepared.operation().program {
            return Err(RenderRecordError::PipelineProgramMismatch);
        }
        if pipeline.depth_stencil != prepared.operation().depth_stencil {
            return Err(RenderRecordError::PipelineDepthStencilStateMismatch);
        }
        let viewport_count = raster.viewports.len().max(raster.scissors.len()).max(1);
        let viewport_count =
            u32::try_from(viewport_count).map_err(|_| RenderRecordError::ViewportCountPastLimit)?;
        if viewport_count != pipeline.viewport_count || viewport_count > resolver.max_viewports() {
            return Err(RenderRecordError::ViewportCountPastLimit);
        }
        let (viewports, scissors) = project_viewport_scissors(prepared.operation())?;
        let visibility = prepared
            .operation()
            .visibility
            .map(|visibility| {
                if visibility.mode == reims_vgpu_protocol::VisibilityResultMode::Counting
                    && !resolver.precise_occlusion_queries()
                {
                    return Err(RenderRecordError::PreciseVisibilityUnavailable);
                }
                let representation = representations_for_visibility(prepared, visibility.backing)?;
                let target = buffer_target(visibility.backing, representation, resolver)?;
                if !target.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
                    || visibility.range.end() > target.size
                {
                    return Err(RenderRecordError::VisibilityBufferMismatch(
                        visibility.backing,
                    ));
                }
                Ok(NativeRenderVisibility {
                    mode: visibility.mode,
                    buffer: target.buffer,
                    offset: target
                        .base_offset
                        .checked_add(visibility.range.start())
                        .ok_or(RenderRecordError::BufferAddressOverflow(visibility.backing))?,
                })
            })
            .transpose()?;
        let image_state = resolve_image_transitions(state, resolver)
            .map_err(RenderRecordError::ImageTransition)?;
        if !image_state.releases.is_empty() {
            return Err(RenderRecordError::ImageReleasePending);
        }
        let representations = prepared.representations();
        // Attachments, resolves and sampled bindings all name a backing's
        // image view; a bound buffer names its byte view. `image_view` and
        // `byte_view` say which, so a backing carrying both objects resolves
        // correctly at every site below.
        let image_view = |backing: BackingId| {
            ViewRepresentation::lookup(representations, backing, BackingView::Image)
                .ok_or(RenderRecordError::RepresentationUseMismatch(backing))
        };
        let byte_view = |backing: BackingId| {
            ViewRepresentation::lookup(representations, backing, BackingView::Bytes)
                .ok_or(RenderRecordError::RepresentationUseMismatch(backing))
        };
        let mut attachment_views = Vec::new();
        let mut clear_values = Vec::new();
        for (expected_index, expected) in pipeline.color_attachments.iter().enumerate() {
            let role = RenderAttachmentRole::Color(expected_index as u32);
            let attachment = prepared
                .operation()
                .attachments
                .iter()
                .find(|attachment| attachment.role == role)
                .ok_or(RenderRecordError::PipelineAttachmentMismatch(role))?;
            if (expected.format, expected.load, expected.store)
                != (attachment.pixel_format, attachment.load, attachment.store)
            {
                return Err(RenderRecordError::PipelineAttachmentMismatch(
                    attachment.role,
                ));
            }
            let target = attachment_target(attachment, representations, resolver)?;
            let key = ReplacementImageKey {
                backing: attachment.backing,
                representation: image_view(attachment.backing)?,
            };
            validate_attachment_target(key, attachment, target, pipeline.sample_count)?;
            attachment_views.push(target.view);
            let RenderAttachmentClear::Color(bits) = attachment.clear else {
                unreachable!()
            };
            clear_values.push(NativeRenderClear::Color(bits));
        }
        let color_count = prepared
            .operation()
            .attachments
            .iter()
            .filter(|attachment| matches!(attachment.role, RenderAttachmentRole::Color(_)))
            .count();
        if color_count != pipeline.color_attachments.len() {
            return Err(RenderRecordError::PipelineColorCountMismatch);
        }
        for (expected_index, expected) in pipeline.color_attachments.iter().enumerate() {
            let role = RenderAttachmentRole::Color(expected_index as u32);
            let attachment = prepared
                .operation()
                .attachments
                .iter()
                .find(|attachment| attachment.role == role)
                .expect("the color attachment set was validated above");
            if expected.resolve_format
                != attachment
                    .resolve
                    .as_ref()
                    .map(|resolve| resolve.pixel_format)
            {
                return Err(RenderRecordError::PipelineAttachmentMismatch(role));
            }
            if let Some(resolve) = &attachment.resolve {
                let target = resolve_target(resolve, representations, resolver)?;
                let key = ReplacementImageKey {
                    backing: resolve.backing,
                    representation: image_view(resolve.backing)?,
                };
                validate_resolve_target(key, attachment.role, resolve, target)?;
                attachment_views.push(target.view);
                clear_values.push(NativeRenderClear::Color([0; 4]));
            }
        }
        let depth = prepared
            .operation()
            .attachments
            .iter()
            .find(|attachment| attachment.role == RenderAttachmentRole::Depth);
        let stencil = prepared
            .operation()
            .attachments
            .iter()
            .find(|attachment| attachment.role == RenderAttachmentRole::Stencil);
        match (pipeline.depth_stencil_attachment, depth, stencil) {
            (None, None, None) => {}
            (Some(expected), depth, stencil) => {
                if expected.depth_format != depth.map(|attachment| attachment.pixel_format)
                    || expected.stencil_format != stencil.map(|attachment| attachment.pixel_format)
                    || expected.depth_resolve_format
                        != depth.and_then(|attachment| {
                            attachment
                                .resolve
                                .as_ref()
                                .map(|resolve| resolve.pixel_format)
                        })
                    || expected.stencil_resolve_format
                        != stencil.and_then(|attachment| {
                            attachment
                                .resolve
                                .as_ref()
                                .map(|resolve| resolve.pixel_format)
                        })
                    || depth.is_some_and(|attachment| {
                        (attachment.load, attachment.store)
                            != (expected.depth_load, expected.depth_store)
                    })
                    || stencil.is_some_and(|attachment| {
                        (attachment.load, attachment.store)
                            != (expected.stencil_load, expected.stencil_store)
                    })
                {
                    return Err(RenderRecordError::PipelineDepthStencilMismatch);
                }
                let first = depth
                    .or(stencil)
                    .expect("a depth/stencil plan names one aspect");
                let first_target = attachment_target(first, representations, resolver)?;
                let first_key = ReplacementImageKey {
                    backing: first.backing,
                    representation: image_view(first.backing)?,
                };
                validate_attachment_target(first_key, first, first_target, pipeline.sample_count)?;
                if let Some(second) = stencil.filter(|_| depth.is_some()) {
                    let second_target = attachment_target(second, representations, resolver)?;
                    let second_key = ReplacementImageKey {
                        backing: second.backing,
                        representation: image_view(second.backing)?,
                    };
                    validate_attachment_target(
                        second_key,
                        second,
                        second_target,
                        pipeline.sample_count,
                    )?;
                    if second_target.view != first_target.view {
                        return Err(RenderRecordError::SeparateDepthStencilViews);
                    }
                }
                attachment_views.push(first_target.view);
                let depth_bits = depth
                    .map(|attachment| match attachment.clear {
                        RenderAttachmentClear::Depth(bits) => bits,
                        _ => unreachable!(),
                    })
                    .unwrap_or(0);
                let stencil_clear = stencil
                    .map(|attachment| match attachment.clear {
                        RenderAttachmentClear::Stencil(value) => value,
                        _ => unreachable!(),
                    })
                    .unwrap_or(0);
                clear_values.push(NativeRenderClear::DepthStencil {
                    depth_bits,
                    stencil: stencil_clear,
                });
                let depth_resolve = depth.and_then(|attachment| attachment.resolve.as_ref());
                let stencil_resolve = stencil.and_then(|attachment| attachment.resolve.as_ref());
                if depth_resolve.is_some() || stencil_resolve.is_some() {
                    let first_resolve = depth_resolve
                        .or(stencil_resolve)
                        .expect("a depth/stencil resolve plan names one aspect");
                    let first_target = resolve_target(first_resolve, representations, resolver)?;
                    let first_key = ReplacementImageKey {
                        backing: first_resolve.backing,
                        representation: image_view(first_resolve.backing)?,
                    };
                    validate_resolve_target(
                        first_key,
                        if depth_resolve.is_some() {
                            RenderAttachmentRole::Depth
                        } else {
                            RenderAttachmentRole::Stencil
                        },
                        first_resolve,
                        first_target,
                    )?;
                    if let Some(second) = stencil_resolve.filter(|_| depth_resolve.is_some()) {
                        let second_target = resolve_target(second, representations, resolver)?;
                        let second_key = ReplacementImageKey {
                            backing: second.backing,
                            representation: image_view(second.backing)?,
                        };
                        validate_resolve_target(
                            second_key,
                            RenderAttachmentRole::Stencil,
                            second,
                            second_target,
                        )?;
                        if second_target.view != first_target.view {
                            return Err(RenderRecordError::SeparateDepthStencilResolveViews);
                        }
                    }
                    attachment_views.push(first_target.view);
                    clear_values.push(NativeRenderClear::DepthStencil {
                        depth_bits: 0,
                        stencil: 0,
                    });
                }
            }
            _ => return Err(RenderRecordError::PipelineDepthStencilMismatch),
        }
        let mut descriptors = Vec::new();
        let mut declarations = BTreeMap::<u32, (vk::DescriptorType, u32)>::new();
        let mut vertex_buffers = Vec::new();
        for resource in &prepared.operation().resources {
            // The descriptor class picks the view: a vertex or uniform buffer
            // binding names the bytes, a sampled or storage image binding
            // names the texels.
            let view = match resource.view {
                RenderBindingView::Buffer(_) => BackingView::Bytes,
                RenderBindingView::Image(_) => BackingView::Image,
            };
            let representation =
                ViewRepresentation::lookup(representations, resource.backing, view).ok_or(
                    RenderRecordError::RepresentationUseMismatch(resource.backing),
                )?;
            match (resource.class, resource.view) {
                (RenderBindingClass::VertexBuffer, RenderBindingView::Buffer(range)) => {
                    let target = buffer_target(resource.backing, representation, resolver)?;
                    if !target.usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
                        return Err(RenderRecordError::MissingVertexBufferUsage(
                            resource.backing,
                        ));
                    }
                    validate_buffer_range(resource.backing, range, target)?;
                    vertex_buffers.push(NativeRenderBufferBinding {
                        binding: resource.binding,
                        buffer: target.buffer,
                        offset: target
                            .base_offset
                            .checked_add(range.start())
                            .ok_or(RenderRecordError::BufferAddressOverflow(resource.backing))?,
                    });
                }
                (RenderBindingClass::IndexBuffer, RenderBindingView::Buffer(_)) => {}
                (RenderBindingClass::IndirectBuffer, RenderBindingView::Buffer(_)) => {}
                (RenderBindingClass::StorageBuffer, RenderBindingView::Buffer(range)) => {
                    declare(
                        &mut declarations,
                        resource.binding,
                        vk::DescriptorType::STORAGE_BUFFER,
                        resource.descriptor_count,
                    )?;
                    let target = buffer_target(resource.backing, representation, resolver)?;
                    if !target.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                        return Err(RenderRecordError::MissingStorageBufferUsage(
                            resource.backing,
                        ));
                    }
                    validate_buffer_range(resource.backing, range, target)?;
                    let offset = target
                        .base_offset
                        .checked_add(range.start())
                        .ok_or(RenderRecordError::BufferAddressOverflow(resource.backing))?;
                    if resolver.min_storage_buffer_offset_alignment() != 0
                        && !offset.is_multiple_of(resolver.min_storage_buffer_offset_alignment())
                    {
                        return Err(RenderRecordError::StorageBufferOffsetMisaligned(
                            resource.backing,
                        ));
                    }
                    if range.end() - range.start() > resolver.max_storage_buffer_range() {
                        return Err(RenderRecordError::StorageBufferRangePastLimit(
                            resource.backing,
                        ));
                    }
                    descriptors.push(NativeComputeDescriptor::StorageBuffer {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        buffer: target.buffer,
                        offset,
                        range: range.end() - range.start(),
                    });
                }
                (
                    class @ (RenderBindingClass::SampledImage | RenderBindingClass::StorageImage),
                    RenderBindingView::Image(view),
                ) => {
                    let (ty, usage) = if class == RenderBindingClass::SampledImage {
                        (
                            vk::DescriptorType::SAMPLED_IMAGE,
                            vk::ImageUsageFlags::SAMPLED,
                        )
                    } else {
                        (
                            vk::DescriptorType::STORAGE_IMAGE,
                            vk::ImageUsageFlags::STORAGE,
                        )
                    };
                    declare(
                        &mut declarations,
                        resource.binding,
                        ty,
                        resource.descriptor_count,
                    )?;
                    let key = ReplacementImageKey {
                        backing: resource.backing,
                        representation,
                    };
                    let target =
                        resolver
                            .resolve_texture_binding_view(key, view)
                            .map_err(|reason| RenderRecordError::UnknownImageView {
                                image: key,
                                resource: view.resource,
                                reason,
                            })?;
                    if target.view == vk::ImageView::null() {
                        return Err(RenderRecordError::MissingImageView(key));
                    }
                    if !target.usage.contains(usage) {
                        return Err(RenderRecordError::MissingImageUsage {
                            image: key,
                            required: usage,
                        });
                    }
                    let layout = state
                        .transitions()
                        .iter()
                        .find(|transition| transition.image == key)
                        .ok_or(RenderRecordError::ImageState(
                            RenderImageStateError::StateOperationMismatch,
                        ))?
                        .use_layout;
                    descriptors.push(NativeComputeDescriptor::Image {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        descriptor_type: ty,
                        view: target.view,
                        layout,
                    });
                }
                _ => return Err(RenderRecordError::BindingViewMismatch(resource.binding)),
            }
        }
        let mut sampler_leases = Vec::with_capacity(prepared.operation().samplers.len());
        for sampler in &prepared.operation().samplers {
            declare(
                &mut declarations,
                sampler.binding,
                vk::DescriptorType::SAMPLER,
                sampler.descriptor_count,
            )?;
            let native = if sampler.sampler.source == reims_vgpu_core::SamplerSource::Null {
                if !resolver.null_descriptors() {
                    return Err(RenderRecordError::NullDescriptorUnavailable(
                        sampler.binding,
                    ));
                }
                vk::Sampler::null()
            } else {
                let lease = resolver
                    .resolve_sampler(prepared.operation().pipeline, &sampler.sampler)
                    .filter(|sampler| sampler.handle() != vk::Sampler::null())
                    .ok_or(RenderRecordError::UnknownSampler(sampler.binding))?;
                let native = lease.handle();
                sampler_leases.push(lease);
                native
            };
            descriptors.push(NativeComputeDescriptor::Sampler {
                binding: sampler.binding,
                array_element: sampler.array_element,
                sampler: native,
            });
        }
        for null in &prepared.operation().null_bindings {
            if !resolver.null_descriptors() {
                return Err(RenderRecordError::NullDescriptorUnavailable(null.binding));
            }
            let descriptor_type = match null.class {
                RenderBindingClass::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
                RenderBindingClass::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                RenderBindingClass::VertexBuffer
                | RenderBindingClass::IndexBuffer
                | RenderBindingClass::IndirectBuffer
                | RenderBindingClass::StorageBuffer => {
                    return Err(RenderRecordError::BindingViewMismatch(null.binding));
                }
            };
            declare(
                &mut declarations,
                null.binding,
                descriptor_type,
                null.descriptor_count,
            )?;
            descriptors.push(NativeComputeDescriptor::Image {
                binding: null.binding,
                array_element: null.array_element,
                descriptor_type,
                view: vk::ImageView::null(),
                layout: vk::ImageLayout::UNDEFINED,
            });
        }
        vertex_buffers.sort_unstable_by_key(|binding| binding.binding);
        let index_buffer = match prepared.operation().draw {
            ResolvedRenderDraw::Indexed { index_type, .. }
            | ResolvedRenderDraw::IndexedIndirect { index_type, .. } => {
                let semantic = prepared
                    .operation()
                    .resources
                    .iter()
                    .find(|resource| resource.class == RenderBindingClass::IndexBuffer)
                    .expect("core preparation proved the index binding");
                let RenderBindingView::Buffer(_) = semantic.view else {
                    unreachable!()
                };
                let target =
                    buffer_target(semantic.backing, byte_view(semantic.backing)?, resolver)?;
                if !target.usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
                    return Err(RenderRecordError::MissingIndexBufferUsage(semantic.backing));
                }
                let RenderBindingView::Buffer(range) = semantic.view else {
                    unreachable!()
                };
                validate_buffer_range(semantic.backing, range, target)?;
                Some((
                    target.buffer,
                    target
                        .base_offset
                        .checked_add(range.start())
                        .ok_or(RenderRecordError::BufferAddressOverflow(semantic.backing))?,
                    match index_type {
                        reims_vgpu_protocol::IndexType::U16 => vk::IndexType::UINT16,
                        reims_vgpu_protocol::IndexType::U32 => vk::IndexType::UINT32,
                    },
                ))
            }
            ResolvedRenderDraw::Direct { .. } | ResolvedRenderDraw::Indirect { .. } => None,
        };
        let indirect_buffer = match prepared.operation().draw {
            ResolvedRenderDraw::Indirect { .. } | ResolvedRenderDraw::IndexedIndirect { .. } => {
                let semantic = prepared
                    .operation()
                    .resources
                    .iter()
                    .find(|resource| resource.class == RenderBindingClass::IndirectBuffer)
                    .expect("core preparation proved the indirect binding");
                let RenderBindingView::Buffer(range) = semantic.view else {
                    unreachable!()
                };
                let target =
                    buffer_target(semantic.backing, byte_view(semantic.backing)?, resolver)?;
                if !target.usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
                    return Err(RenderRecordError::MissingIndirectBufferUsage(
                        semantic.backing,
                    ));
                }
                validate_buffer_range(semantic.backing, range, target)?;
                Some((
                    target.buffer,
                    target
                        .base_offset
                        .checked_add(range.start())
                        .ok_or(RenderRecordError::BufferAddressOverflow(semantic.backing))?,
                ))
            }
            ResolvedRenderDraw::Direct { .. } | ResolvedRenderDraw::Indexed { .. } => None,
        };
        let mut backings = representations
            .iter()
            .map(|representation| representation.backing)
            .collect::<Vec<_>>();
        backings.sort_unstable();
        backings.dedup();
        Ok(Self {
            index: prepared.operation_index(),
            transaction: prepared.transaction(),
            operation: prepared.operation().clone(),
            native: NativeRenderDispatch {
                pipeline,
                descriptors: descriptors.into_boxed_slice(),
                sampler_leases: sampler_leases.into_boxed_slice(),
                descriptor_counts: declarations
                    .into_values()
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                vertex_buffers: vertex_buffers.into_boxed_slice(),
                index_buffer,
                indirect_buffer,
                attachment_views: attachment_views.into_boxed_slice(),
                clear_values: clear_values.into_boxed_slice(),
                extent: vk::Extent2D {
                    width: prepared.operation().render_extent[0],
                    height: prepared.operation().render_extent[1],
                },
                viewports,
                scissors,
                depth_bias: raster.depth_bias_bits.map(|bits| bits.map(f32::from_bits)),
                blend_color: raster.blend_color_bits.map(|bits| bits.map(f32::from_bits)),
                stencil_reference: raster.stencil_reference,
                visibility,
                begins_native_pass: true,
                ends_native_pass: true,
                draw: prepared.operation().draw,
                image_state,
            },
            backings: backings.into_boxed_slice(),
            completions: prepared.completions().into(),
        })
    }
}

fn attachment_target(
    attachment: &reims_vgpu_core::ResolvedRenderAttachment,
    representations: &[ViewRepresentation],
    resolver: &impl ReplacementRenderResolver,
) -> Result<NativeImageTarget, RenderRecordError> {
    let [region] = attachment.regions.as_ref() else {
        return Err(RenderRecordError::AttachmentRegionUnsupported(
            attachment.role,
        ));
    };
    let key = ReplacementImageKey {
        backing: attachment.backing,
        representation: ViewRepresentation::lookup(
            representations,
            attachment.backing,
            BackingView::Image,
        )
        .ok_or(RenderRecordError::RepresentationUseMismatch(
            attachment.backing,
        ))?,
    };
    resolver.resolve_attachment_view(key, *region).ok_or(
        RenderRecordError::AttachmentRegionUnsupported(attachment.role),
    )
}

fn resolve_target(
    resolve: &reims_vgpu_core::ResolvedRenderResolveAttachment,
    representations: &[ViewRepresentation],
    resolver: &impl ReplacementRenderResolver,
) -> Result<NativeImageTarget, RenderRecordError> {
    let [region] = resolve.regions.as_ref() else {
        return Err(RenderRecordError::ResolveRegionUnsupported);
    };
    let key = ReplacementImageKey {
        backing: resolve.backing,
        representation: ViewRepresentation::lookup(
            representations,
            resolve.backing,
            BackingView::Image,
        )
        .ok_or(RenderRecordError::RepresentationUseMismatch(
            resolve.backing,
        ))?,
    };
    resolver
        .resolve_attachment_view(key, *region)
        .ok_or(RenderRecordError::ResolveRegionUnsupported)
}

/// The visibility result is written as bytes, so it resolves the backing's
/// byte view whatever else the same allocation is bound as.
fn representations_for_visibility<NativePipeline>(
    prepared: &PreparedRenderDispatch<NativePipeline>,
    backing: BackingId,
) -> Result<RepresentationId, RenderRecordError> {
    ViewRepresentation::lookup(prepared.representations(), backing, BackingView::Bytes)
        .ok_or(RenderRecordError::RepresentationUseMismatch(backing))
}
fn buffer_target(
    backing: BackingId,
    representation: RepresentationId,
    resolver: &impl ReplacementRenderResolver,
) -> Result<NativeBufferTarget, RenderRecordError> {
    resolver
        .resolve_buffer(backing, representation)
        .ok_or(RenderRecordError::UnknownBuffer {
            backing,
            representation,
        })
}
fn validate_buffer_range(
    backing: BackingId,
    range: reims_vgpu_core::LinearRange,
    target: NativeBufferTarget,
) -> Result<(), RenderRecordError> {
    if range.end() > target.size {
        return Err(RenderRecordError::BufferRangeOutOfBounds(backing));
    }
    target
        .base_offset
        .checked_add(range.start())
        .ok_or(RenderRecordError::BufferAddressOverflow(backing))?;
    Ok(())
}
fn validate_attachment_target(
    key: ReplacementImageKey,
    attachment: &reims_vgpu_core::ResolvedRenderAttachment,
    target: NativeImageTarget,
    samples: vk::SampleCountFlags,
) -> Result<(), RenderRecordError> {
    let semantic_samples = match attachment.sample_count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => {
            return Err(RenderRecordError::UnsupportedAttachmentSampleCount(
                attachment.role,
            ));
        }
    };
    let (usage, aspect) = match attachment.role {
        RenderAttachmentRole::Color(_) => (
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        ),
        RenderAttachmentRole::Depth => (
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        ),
        RenderAttachmentRole::Stencil => (
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::STENCIL,
        ),
    };
    if target.view == vk::ImageView::null() {
        return Err(RenderRecordError::MissingImageView(key));
    }
    if !target.usage.contains(usage) {
        return Err(RenderRecordError::MissingImageUsage {
            image: key,
            required: usage,
        });
    }
    if !target.full_range.aspect_mask.contains(aspect) {
        return Err(RenderRecordError::MissingAttachmentAspect {
            image: key,
            required: aspect,
        });
    }
    if target.pixel_format != attachment.pixel_format
        || target.extent.width < attachment.extent[0]
        || target.extent.height < attachment.extent[1]
        || semantic_samples != samples
        || target.samples != samples
    {
        return Err(RenderRecordError::AttachmentTargetMismatch(attachment.role));
    }
    Ok(())
}

fn validate_resolve_target(
    key: ReplacementImageKey,
    role: RenderAttachmentRole,
    resolve: &reims_vgpu_core::ResolvedRenderResolveAttachment,
    target: NativeImageTarget,
) -> Result<(), RenderRecordError> {
    let (usage, aspect) = match role {
        RenderAttachmentRole::Color(_) => (
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        ),
        RenderAttachmentRole::Depth => (
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        ),
        RenderAttachmentRole::Stencil => (
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::STENCIL,
        ),
    };
    if target.view == vk::ImageView::null() {
        return Err(RenderRecordError::MissingImageView(key));
    }
    if !target.usage.contains(usage) {
        return Err(RenderRecordError::MissingImageUsage {
            image: key,
            required: usage,
        });
    }
    if !target.full_range.aspect_mask.contains(aspect)
        || target.pixel_format != resolve.pixel_format
        || target.extent.width < resolve.extent[0]
        || target.extent.height < resolve.extent[1]
        || target.samples != vk::SampleCountFlags::TYPE_1
    {
        return Err(RenderRecordError::ResolveTargetMismatch(role));
    }
    Ok(())
}

type NativeViewportScissors = (Box<[vk::Viewport]>, Box<[vk::Rect2D]>);

fn project_viewport_scissors(
    operation: &ResolvedRenderDispatch,
) -> Result<NativeViewportScissors, RenderRecordError> {
    let slots = operation
        .raster
        .viewports
        .len()
        .max(operation.raster.scissors.len())
        .max(1);
    let mut viewports = Vec::with_capacity(slots);
    let mut scissors = Vec::with_capacity(slots);
    for index in 0..slots {
        let values = operation
            .raster
            .viewports
            .get(index)
            .map(|viewport| viewport.values())
            .unwrap_or([
                0.0,
                0.0,
                f64::from(operation.render_extent[0]),
                f64::from(operation.render_extent[1]),
                0.0,
                1.0,
            ]);
        if values.iter().any(|value| !value.is_finite())
            || values[2] < 0.0
            || values[3] < 0.0
            || values.iter().any(|value| value.abs() > f64::from(f32::MAX))
        {
            return Err(RenderRecordError::InvalidViewport(index));
        }
        viewports.push(vk::Viewport {
            x: values[0] as f32,
            y: (values[1] + values[3]) as f32,
            width: values[2] as f32,
            height: -(values[3] as f32),
            min_depth: values[4] as f32,
            max_depth: values[5] as f32,
        });
        let scissor = operation.raster.scissors.get(index).copied().unwrap_or(
            reims_vgpu_core::RenderScissor {
                x: 0,
                y: 0,
                width: operation.render_extent[0],
                height: operation.render_extent[1],
            },
        );
        let x = scissor.x.min(operation.render_extent[0]);
        let y = scissor.y.min(operation.render_extent[1]);
        scissors.push(vk::Rect2D {
            offset: vk::Offset2D {
                x: i32::try_from(x).map_err(|_| RenderRecordError::InvalidScissor(index))?,
                y: i32::try_from(y).map_err(|_| RenderRecordError::InvalidScissor(index))?,
            },
            extent: vk::Extent2D {
                width: scissor.width.min(operation.render_extent[0] - x),
                height: scissor.height.min(operation.render_extent[1] - y),
            },
        });
    }
    Ok((viewports.into_boxed_slice(), scissors.into_boxed_slice()))
}
fn declare(
    declarations: &mut BTreeMap<u32, (vk::DescriptorType, u32)>,
    binding: u32,
    ty: vk::DescriptorType,
    count: u32,
) -> Result<(), RenderRecordError> {
    if declarations
        .insert(binding, (ty, count))
        .is_some_and(|found| found != (ty, count))
    {
        return Err(RenderRecordError::DescriptorDeclarationMismatch(binding));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderRecordError {
    DeviceEpochLost,
    InvalidPipeline,
    PipelineRasterStateMismatch,
    PipelineProgramMismatch,
    PipelineDepthStencilStateMismatch,
    ViewportCountPastLimit,
    InvalidViewport(usize),
    InvalidScissor(usize),
    RepresentationUseMismatch(BackingId),
    ImageState(RenderImageStateError),
    ImageTransition(ImageTransitionResolveError),
    ImageReleasePending,
    FeedbackRepresentationRequired(ReplacementImageKey),
    PipelineAttachmentMismatch(RenderAttachmentRole),
    PipelineColorCountMismatch,
    PipelineDepthStencilMismatch,
    SeparateDepthStencilViews,
    SeparateDepthStencilResolveViews,
    AttachmentTargetMismatch(RenderAttachmentRole),
    UnsupportedAttachmentSampleCount(RenderAttachmentRole),
    ResolveTargetMismatch(RenderAttachmentRole),
    UnknownImage(ReplacementImageKey),
    /// A decoded texture-binding view this backend has no `VkImageView` for.
    /// The reason names which of the view's terms it could not build, because
    /// they are unrelated pieces of work -- see
    /// [`crate::replacement_image_transition::TextureBindingViewDecline`].
    UnknownImageView {
        image: ReplacementImageKey,
        resource: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
        reason: crate::replacement_image_transition::TextureBindingViewDecline,
    },
    MissingImageView(ReplacementImageKey),
    MissingImageUsage {
        image: ReplacementImageKey,
        required: vk::ImageUsageFlags,
    },
    MissingAttachmentAspect {
        image: ReplacementImageKey,
        required: vk::ImageAspectFlags,
    },
    UnknownBuffer {
        backing: BackingId,
        representation: RepresentationId,
    },
    BufferRangeOutOfBounds(BackingId),
    BufferAddressOverflow(BackingId),
    MissingVertexBufferUsage(BackingId),
    MissingIndexBufferUsage(BackingId),
    MissingIndirectBufferUsage(BackingId),
    MissingStorageBufferUsage(BackingId),
    StorageBufferOffsetMisaligned(BackingId),
    StorageBufferRangePastLimit(BackingId),
    BindingViewMismatch(u32),
    DescriptorDeclarationMismatch(u32),
    UnknownSampler(u32),
    NullDescriptorUnavailable(u32),
    AttachmentRegionUnsupported(RenderAttachmentRole),
    ResolveRegionUnsupported,
    MissingIndexBinding(u32),
    VisibilityBufferMismatch(BackingId),
    PreciseVisibilityUnavailable,
}

/// Record one framebuffer-backed draw. The framebuffer must be compatible
/// with the retained pipeline render pass and remain live through retirement.
///
/// # Safety
///
/// Every native handle must belong to `device`, the command buffer must be in
/// the recording state, and the framebuffer, descriptor set, and retained
/// pipeline resources must remain live until the submission retires.
pub unsafe fn record_render_dispatch(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    descriptor_set: Option<vk::DescriptorSet>,
    framebuffer: vk::Framebuffer,
    visibility_query: Option<(vk::QueryPool, u32)>,
    native: &NativeRenderDispatch,
) {
    for descriptor in native.descriptors.iter().copied() {
        let set = descriptor_set.expect("a render descriptor declaration owns one set");
        match descriptor {
            NativeComputeDescriptor::StorageBuffer {
                binding,
                array_element,
                buffer,
                offset,
                range,
            } => {
                let info = [vk::DescriptorBufferInfo {
                    buffer,
                    offset,
                    range,
                }];
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .dst_array_element(array_element)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)],
                    &[],
                );
            }
            NativeComputeDescriptor::Sampler {
                binding,
                array_element,
                sampler,
            } => {
                let info = [vk::DescriptorImageInfo::default().sampler(sampler)];
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .dst_array_element(array_element)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&info)],
                    &[],
                );
            }
            NativeComputeDescriptor::Image {
                binding,
                array_element,
                descriptor_type,
                view,
                layout,
            } => {
                let info = [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(layout)];
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .dst_array_element(array_element)
                        .descriptor_type(descriptor_type)
                        .image_info(&info)],
                    &[],
                );
            }
        }
    }
    let clear_values = native
        .clear_values
        .iter()
        .map(|clear| match *clear {
            NativeRenderClear::Color(bits) => vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: bits.map(f32::from_bits),
                },
            },
            NativeRenderClear::DepthStencil {
                depth_bits,
                stencil,
            } => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: f32::from_bits(depth_bits),
                    stencil,
                },
            },
        })
        .collect::<Vec<_>>();
    if native.begins_native_pass {
        device.cmd_begin_render_pass(
            command_buffer,
            &vk::RenderPassBeginInfo::default()
                .render_pass(native.pipeline.render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: native.extent,
                })
                .clear_values(&clear_values),
            vk::SubpassContents::INLINE,
        );
    }
    if let Some(visibility) = native.visibility {
        let (pool, query) = visibility_query.expect("a visibility draw owns one worker query");
        let flags = match visibility.mode {
            reims_vgpu_protocol::VisibilityResultMode::Boolean => vk::QueryControlFlags::empty(),
            reims_vgpu_protocol::VisibilityResultMode::Counting => vk::QueryControlFlags::PRECISE,
        };
        device.cmd_begin_query(command_buffer, pool, query, flags);
    }
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        native.pipeline.pipeline,
    );
    device.cmd_set_viewport(command_buffer, 0, &native.viewports);
    device.cmd_set_scissor(command_buffer, 0, &native.scissors);
    for setting in dynamic_settings(native) {
        match setting {
            ReplacementRenderDynamicSetting::DepthBias([constant_factor, slope_factor, clamp]) => {
                device.cmd_set_depth_bias(command_buffer, constant_factor, clamp, slope_factor);
            }
            ReplacementRenderDynamicSetting::BlendConstants(color) => {
                device.cmd_set_blend_constants(command_buffer, &color);
            }
            ReplacementRenderDynamicSetting::StencilReference([front, back]) => {
                device.cmd_set_stencil_reference(
                    command_buffer,
                    vk::StencilFaceFlags::FRONT,
                    front,
                );
                device.cmd_set_stencil_reference(command_buffer, vk::StencilFaceFlags::BACK, back);
            }
        }
    }
    if let Some(set) = descriptor_set {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            native.pipeline.layout,
            0,
            &[set],
            &[],
        );
    }
    for binding in &native.vertex_buffers {
        device.cmd_bind_vertex_buffers(
            command_buffer,
            binding.binding,
            &[binding.buffer],
            &[binding.offset],
        );
    }
    match native.draw {
        ResolvedRenderDraw::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            ..
        } => device.cmd_draw(
            command_buffer,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        ),
        ResolvedRenderDraw::Indexed {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            ..
        } => {
            let (buffer, offset, ty) = native
                .index_buffer
                .expect("an indexed preparation owns its index binding");
            device.cmd_bind_index_buffer(command_buffer, buffer, offset, ty);
            device.cmd_draw_indexed(
                command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
        ResolvedRenderDraw::Indirect { .. } => {
            let (buffer, offset) = native
                .indirect_buffer
                .expect("an indirect preparation owns its argument binding");
            device.cmd_draw_indirect(
                command_buffer,
                buffer,
                offset,
                1,
                reims_vgpu_core::RENDER_INDIRECT_ARGUMENT_BYTES as u32,
            );
        }
        ResolvedRenderDraw::IndexedIndirect { .. } => {
            let (index_buffer, index_offset, ty) = native
                .index_buffer
                .expect("an indexed indirect preparation owns its index binding");
            let (arguments, argument_offset) = native
                .indirect_buffer
                .expect("an indexed indirect preparation owns its argument binding");
            device.cmd_bind_index_buffer(command_buffer, index_buffer, index_offset, ty);
            device.cmd_draw_indexed_indirect(
                command_buffer,
                arguments,
                argument_offset,
                1,
                reims_vgpu_core::RENDER_INDEXED_INDIRECT_ARGUMENT_BYTES as u32,
            );
        }
    }
    if native.visibility.is_some() {
        let (pool, query) = visibility_query.expect("a visibility draw owns one worker query");
        device.cmd_end_query(command_buffer, pool, query);
    }
    if native.ends_native_pass {
        device.cmd_end_render_pass(command_buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::BackingView;
    use reims_vgpu_core::{
        prepare_render_dispatch, AccessMode, BackingRegion, PipelineLifecycle, PipelineReadiness,
        RepresentationRoute, ResolvedRenderAttachment, ResolvedRenderResourceBinding,
        ResolvedResourceLifecycle, ResourceLifecycleEffect, ResourceLifecycleOwner,
        SessionGeneration, StorageBacking, VulkanDeviceEpoch, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        LoadAction, PrimitiveTopology, RenderPipelineObject, RenderStages, ResourceId,
        ResourceObject, SessionGenerationId, StoreAction, SubmissionId, TransactionId,
        VulkanDeviceEpochId,
    };

    const EPOCH: VulkanDeviceEpochId = VulkanDeviceEpochId::new(6);

    fn image_view(
        resource: ResourceId<ResourceObject>,
    ) -> reims_vgpu_core::ResolvedTextureBindingView {
        reims_vgpu_core::ResolvedTextureBindingView {
            resource,
            base: resource,
            range: reims_vgpu_core::ResolvedTextureViewRange {
                level_base: 0,
                level_count: 1,
                slice_base: 0,
                slice_count: 1,
            },
            texture_type: reims_vgpu_protocol::TextureType::D2,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            swizzle: reims_vgpu_protocol::swizzle_identity(),
        }
    }

    struct Resolver {
        attachment: ReplacementImageKey,
        sampled: ReplacementImageKey,
        depth_stencil: Option<ReplacementImageKey>,
        resolve: Option<ReplacementImageKey>,
        visibility: Option<(BackingId, RepresentationId)>,
        arguments: Option<(BackingId, RepresentationId)>,
        precise_visibility: bool,
        null_descriptors: bool,
    }

    impl ReplacementBufferResolver for Resolver {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            if self.arguments == Some((backing, representation)) {
                return Some(NativeBufferTarget {
                    buffer: vk::Buffer::from_raw(51),
                    base_offset: 8,
                    accessible_size: 64,
                    size: 64,
                    usage: vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER,
                });
            }
            (self.visibility == Some((backing, representation))).then_some(NativeBufferTarget {
                buffer: vk::Buffer::from_raw(50),
                base_offset: 16,
                accessible_size: 64,
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
            })
        }
    }

    impl ReplacementImageResolver for Resolver {
        fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
            let (view, usage, aspect, pixel_format) = if image == self.attachment {
                (
                    41,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::ImageAspectFlags::COLOR,
                    80,
                )
            } else if image == self.sampled {
                (
                    42,
                    vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                    80,
                )
            } else if Some(image) == self.depth_stencil {
                (
                    43,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    252,
                )
            } else if Some(image) == self.resolve {
                (
                    44,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::ImageAspectFlags::COLOR,
                    80,
                )
            } else {
                return None;
            };
            Some(NativeImageTarget {
                image: vk::Image::from_raw(view - 10),
                view: vk::ImageView::from_raw(view),
                image_type: vk::ImageType::TYPE_2D,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                usage,
                pixel_format,
                extent: vk::Extent3D {
                    width: 32,
                    height: 16,
                    depth: 1,
                },
                samples: if image == self.attachment && self.resolve.is_some() {
                    vk::SampleCountFlags::TYPE_4
                } else {
                    vk::SampleCountFlags::TYPE_1
                },
            })
        }

        fn resolve_texture_binding_view(
            &self,
            image: ReplacementImageKey,
            _: reims_vgpu_core::ResolvedTextureBindingView,
        ) -> Result<NativeImageTarget, crate::replacement_image_transition::TextureBindingViewDecline>
        {
            self.resolve_image(image).ok_or(
                crate::replacement_image_transition::TextureBindingViewDecline::UnknownRepresentation,
            )
        }
    }

    impl ReplacementRenderResolver for Resolver {
        fn resolve_sampler(
            &self,
            _: ResourceId<RenderPipelineObject>,
            _: &reims_vgpu_core::SamplerResource,
        ) -> Option<crate::replacement_sampler::ReplacementSamplerLease> {
            None
        }
        fn max_storage_buffer_range(&self) -> u64 {
            1024
        }
        fn min_storage_buffer_offset_alignment(&self) -> u64 {
            16
        }
        fn max_viewports(&self) -> u32 {
            16
        }
        fn precise_occlusion_queries(&self) -> bool {
            self.precise_visibility
        }
        fn null_descriptors(&self) -> bool {
            self.null_descriptors
        }
    }

    fn pipeline(
    ) -> reims_vgpu_core::ReadyPipelineLease<RenderPipelineObject, ReplacementRenderPipelineVariant>
    {
        let id = ResourceId::new(2, 1);
        let mut owner = PipelineLifecycle::<
            RenderPipelineObject,
            (),
            ReplacementRenderPipelineVariant,
            (),
        >::default();
        owner.declare(id, ()).unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(
                compile,
                reims_vgpu_core::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                ReplacementRenderPipelineVariant::synthetic(ReplacementRenderPipeline {
                    pipeline: vk::Pipeline::from_raw(1),
                    layout: vk::PipelineLayout::from_raw(2),
                    descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                    render_pass: vk::RenderPass::from_raw(4),
                    program: Default::default(),
                    vertex_buffers: Box::new([]),
                    color_attachments: Box::new([ReplacementRenderColorAttachment {
                        format: 80,
                        resolve_format: None,
                        load: reims_vgpu_protocol::LoadAction::Clear,
                        store: StoreAction::Store,
                        feedback_loop: false,
                        input_attachment: false,
                    }]),
                    depth_stencil_attachment: None,
                    feedback_loop_aspects: vk::ImageAspectFlags::empty(),
                    color_input: false,
                    sample_count: vk::SampleCountFlags::TYPE_1,
                    viewport_count: 1,
                    static_state: ReplacementRenderStaticState::default(),
                    dynamic_states: ReplacementRenderDynamicStates::default(),
                    depth_stencil: None,
                }),
            )
            .unwrap();
        let PipelineReadiness::Ready(ready) = owner.readiness(id, TransactionId::new(7)).unwrap()
        else {
            unreachable!()
        };
        ready
    }

    fn depth_pipeline(
    ) -> reims_vgpu_core::ReadyPipelineLease<RenderPipelineObject, ReplacementRenderPipelineVariant>
    {
        let id = ResourceId::new(2, 1);
        let mut owner = PipelineLifecycle::<
            RenderPipelineObject,
            (),
            ReplacementRenderPipelineVariant,
            (),
        >::default();
        owner.declare(id, ()).unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(
                compile,
                reims_vgpu_core::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                ReplacementRenderPipelineVariant::synthetic(ReplacementRenderPipeline {
                    pipeline: vk::Pipeline::from_raw(1),
                    layout: vk::PipelineLayout::from_raw(2),
                    descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                    render_pass: vk::RenderPass::from_raw(4),
                    program: Default::default(),
                    vertex_buffers: Box::new([]),
                    color_attachments: Box::new([]),
                    depth_stencil_attachment: Some(ReplacementRenderDepthStencilAttachment {
                        depth_format: Some(252),
                        stencil_format: Some(252),
                        depth_resolve_format: None,
                        stencil_resolve_format: None,
                        depth_load: LoadAction::Clear,
                        depth_store: StoreAction::Store,
                        stencil_load: LoadAction::Clear,
                        stencil_store: StoreAction::Store,
                        feedback_loop_aspects: vk::ImageAspectFlags::empty(),
                    }),
                    feedback_loop_aspects: vk::ImageAspectFlags::empty(),
                    color_input: false,
                    sample_count: vk::SampleCountFlags::TYPE_1,
                    viewport_count: 1,
                    static_state: ReplacementRenderStaticState::default(),
                    dynamic_states: ReplacementRenderDynamicStates::default(),
                    depth_stencil: None,
                }),
            )
            .unwrap();
        let PipelineReadiness::Ready(ready) = owner.readiness(id, TransactionId::new(7)).unwrap()
        else {
            unreachable!()
        };
        ready
    }

    fn color_resolve_pipeline(
    ) -> reims_vgpu_core::ReadyPipelineLease<RenderPipelineObject, ReplacementRenderPipelineVariant>
    {
        let id = ResourceId::new(2, 1);
        let mut owner = PipelineLifecycle::<
            RenderPipelineObject,
            (),
            ReplacementRenderPipelineVariant,
            (),
        >::default();
        owner.declare(id, ()).unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(
                compile,
                reims_vgpu_core::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                ReplacementRenderPipelineVariant::synthetic(ReplacementRenderPipeline {
                    pipeline: vk::Pipeline::from_raw(1),
                    layout: vk::PipelineLayout::from_raw(2),
                    descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                    render_pass: vk::RenderPass::from_raw(4),
                    program: Default::default(),
                    vertex_buffers: Box::new([]),
                    color_attachments: Box::new([ReplacementRenderColorAttachment {
                        format: 80,
                        resolve_format: Some(80),
                        load: LoadAction::Clear,
                        store: StoreAction::MultisampleResolve,
                        feedback_loop: false,
                        input_attachment: false,
                    }]),
                    depth_stencil_attachment: None,
                    feedback_loop_aspects: vk::ImageAspectFlags::empty(),
                    color_input: false,
                    sample_count: vk::SampleCountFlags::TYPE_4,
                    viewport_count: 1,
                    static_state: ReplacementRenderStaticState::default(),
                    dynamic_states: ReplacementRenderDynamicStates::default(),
                    depth_stencil: None,
                }),
            )
            .unwrap();
        let PipelineReadiness::Ready(ready) = owner.readiness(id, TransactionId::new(7)).unwrap()
        else {
            unreachable!()
        };
        ready
    }

    /// A backing whose execution representation is an image, which is what
    /// every attachment and sampled binding in these fixtures needs. A
    /// fixture binding a backing as a buffer — a visibility result, an
    /// indirect argument — asks for [`BackingView::Bytes`] instead.
    fn backing(owner: &mut ResourceLifecycleOwner<()>, current: bool) -> BackingId {
        view_backing(owner, current, BackingView::Image)
    }

    fn view_backing(
        owner: &mut ResourceLifecycleOwner<()>,
        current: bool,
        view: BackingView,
    ) -> BackingId {
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
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                view,
                (),
            )
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

    fn plan_attachment(role: RenderAttachmentRole, format: u16) -> ResolvedRenderAttachment {
        ResolvedRenderAttachment {
            role,
            resource: ResourceId::new(u32::from(format), 1),
            backing: BackingId::new(u64::from(format)),
            regions: Box::new([BackingRegion::Whole]),
            pixel_format: format,
            extent: [32, 16, 1],
            sample_count: 4,
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear: match role {
                RenderAttachmentRole::Color(_) => RenderAttachmentClear::Color([0; 4]),
                RenderAttachmentRole::Depth => RenderAttachmentClear::Depth(0),
                RenderAttachmentRole::Stencil => RenderAttachmentClear::Stencil(0),
            },
            resolve: None,
            feedback_loop: false,
            input_attachment: false,
        }
    }

    fn plan_operation(
        attachments: impl IntoIterator<Item = ResolvedRenderAttachment>,
    ) -> ResolvedRenderDispatch {
        ResolvedRenderDispatch {
            pipeline: ResourceId::new(2, 1),
            program: Default::default(),
            vertex_buffers: Box::new([reims_vgpu_core::ResolvedVertexBufferLayout {
                binding: 7,
                stride: 20,
            }]),
            depth_stencil: Some(ResourceId::new(3, 1)),
            render_extent: [32, 16],
            raster: reims_vgpu_core::ResolvedRenderRasterState {
                cull_mode: CullMode::Back,
                front_face_ccw: true,
                fill_mode: FillMode::Lines,
                line_width_bits: 2.0f32.to_bits(),
                depth_clip_mode: DepthClipMode::Clamp,
                depth_bias_bits: Some([1, 2, 3]),
                ..Default::default()
            },
            visibility: None,
            begins_encoder: true,
            ends_encoder: true,
            draw: ResolvedRenderDraw::Direct {
                topology: PrimitiveTopology::Line,
                vertex_count: 2,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            attachments: attachments
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            resources: Box::new([]),
            null_bindings: Box::new([]),
            samplers: Box::new([]),
        }
    }

    #[test]
    fn render_pipeline_plan_normalizes_roles_and_matches_native_variant_key() {
        let operation = plan_operation([
            plan_attachment(RenderAttachmentRole::Color(1), 81),
            plan_attachment(RenderAttachmentRole::Stencil, 253),
            plan_attachment(RenderAttachmentRole::Color(0), 80),
            plan_attachment(RenderAttachmentRole::Depth, 252),
        ]);
        let plan = resolve_render_pipeline_plan(&operation, 4).unwrap();
        assert_eq!(
            plan.color_attachments
                .iter()
                .map(|attachment| attachment.format)
                .collect::<Vec<_>>(),
            [80, 81]
        );
        assert_eq!(plan.sample_count, vk::SampleCountFlags::TYPE_4);
        assert_eq!(plan.viewport_count, 1);
        assert_eq!(plan.vertex_buffers, operation.vertex_buffers);
        let mut other_stride = plan.clone();
        other_stride.vertex_buffers[0].stride = 24;
        assert_ne!(plan.variant_key(), other_stride.variant_key());
        let native = ReplacementRenderPipeline {
            pipeline: vk::Pipeline::from_raw(10),
            layout: vk::PipelineLayout::from_raw(11),
            descriptor_set_layout: vk::DescriptorSetLayout::from_raw(12),
            render_pass: vk::RenderPass::from_raw(13),
            program: plan.program.clone(),
            vertex_buffers: plan.vertex_buffers.clone(),
            color_attachments: plan.color_attachments.clone(),
            depth_stencil_attachment: plan.depth_stencil_attachment,
            feedback_loop_aspects: plan.feedback_loop_aspects,
            color_input: false,
            sample_count: plan.sample_count,
            viewport_count: plan.viewport_count,
            static_state: plan.static_state,
            dynamic_states: ReplacementRenderDynamicStates::default(),
            depth_stencil: plan.depth_stencil,
        };
        assert_eq!(plan.variant_key(), native.variant_key());
    }

    #[test]
    fn render_pipeline_plan_carries_attachment_sampling_feedback() {
        let mut attachment = plan_attachment(RenderAttachmentRole::Color(0), 80);
        // The encoder decides this, not the draw: see
        // `ResolvedRenderAttachment::feedback_loop`.
        attachment.feedback_loop = true;
        let mut operation = plan_operation([attachment.clone()]);
        operation.resources = Box::new([ResolvedRenderResourceBinding {
            class: RenderBindingClass::SampledImage,
            binding: 4,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
            resource: attachment.resource,
            backing: attachment.backing,
            view: RenderBindingView::Image(image_view(attachment.resource)),
            regions: Box::new([BackingRegion::Whole]),
            mode: AccessMode::Read,
        }]);

        let plan = resolve_render_pipeline_plan(&operation, 4).unwrap();
        assert_eq!(plan.feedback_loop_aspects, vk::ImageAspectFlags::COLOR);
        assert!(plan.color_attachments[0].feedback_loop);
        let mut without_loop = attachment;
        without_loop.feedback_loop = false;
        assert_ne!(
            plan.variant_key(),
            resolve_render_pipeline_plan(&plan_operation([without_loop]), 4)
                .unwrap()
                .variant_key()
        );
    }

    #[test]
    fn attachment_sampling_uses_the_feedback_layout_on_one_representation() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let backing = backing(&mut lifecycle, true);
        let resource = ResourceId::new(80, 1);
        let mut attachment = plan_attachment(RenderAttachmentRole::Color(0), 80);
        attachment.resource = resource;
        attachment.backing = backing;
        attachment.sample_count = 1;
        // The encoder stamped the loop, which is what the pass reads.
        attachment.feedback_loop = true;
        let sampled_binding = ResolvedRenderResourceBinding {
            class: RenderBindingClass::SampledImage,
            binding: 4,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
            resource,
            backing,
            view: RenderBindingView::Image(image_view(resource)),
            regions: Box::new([BackingRegion::Whole]),
            mode: AccessMode::Read,
        };
        let mut operation = plan_operation([attachment.clone()]);
        operation.vertex_buffers = Box::new([]);
        operation.resources = Box::new([sampled_binding.clone()]);
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(9),
            SubmissionId::new(10),
            0,
            operation,
            pipeline(),
        )
        .unwrap();

        let uses = derive_render_image_uses(&prepared).unwrap();
        assert_eq!(uses.len(), 1);
        assert_eq!(
            uses[0].use_layout,
            vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
        );

        // Without the stamp the pass would declare a plain attachment layout
        // while this bind asks for the loop's. The barrier may not repair that
        // on its own -- the pass would still declare the other layout -- so it
        // is refused by name.
        let mut unstamped = attachment;
        unstamped.feedback_loop = false;
        let mut operation = plan_operation([unstamped]);
        operation.vertex_buffers = Box::new([]);
        operation.resources = Box::new([sampled_binding]);
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(11),
            SubmissionId::new(12),
            0,
            operation,
            pipeline(),
        )
        .unwrap();
        assert!(matches!(
            derive_render_image_uses(&prepared),
            Err(RenderImageStateError::FeedbackRepresentationRequired(_))
        ));
        assert!(uses[0]
            .required_usage
            .contains(vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT));
    }

    #[test]
    fn a_draw_that_samples_nothing_still_takes_its_encoder_feedback_layout() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let backing = backing(&mut lifecycle, true);
        let resource = ResourceId::new(80, 1);
        let mut attachment = plan_attachment(RenderAttachmentRole::Color(0), 80);
        attachment.resource = resource;
        attachment.backing = backing;
        attachment.sample_count = 1;
        // Another draw in the same encoder reads this attachment; this one
        // does not. One native render pass covers both, so both must name the
        // one layout the pass declares.
        attachment.feedback_loop = true;
        let mut operation = plan_operation([attachment]);
        operation.vertex_buffers = Box::new([]);
        operation.resources = Box::new([]);
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(9),
            SubmissionId::new(10),
            0,
            operation,
            pipeline(),
        )
        .unwrap();

        let uses = derive_render_image_uses(&prepared).unwrap();
        assert_eq!(uses.len(), 1);
        assert_eq!(
            uses[0].use_layout,
            vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
        );
        assert_eq!(
            uses[0].final_layout,
            vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
        );
        assert!(uses[0]
            .required_usage
            .contains(vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT));
    }

    #[test]
    fn a_resolve_target_of_a_feedback_attachment_is_no_feedback_attachment() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let target = backing(&mut lifecycle, true);
        let resolved = backing(&mut lifecycle, true);
        let mut attachment = plan_attachment(RenderAttachmentRole::Color(0), 80);
        attachment.backing = target;
        attachment.sample_count = 4;
        // Another draw in the encoder reads the multisampled attachment, so
        // the pass declares it in the feedback layout. Its resolve target is
        // written by the pass and never read by it.
        attachment.feedback_loop = true;
        attachment.store = StoreAction::StoreAndMultisampleResolve;
        attachment.resolve = Some(reims_vgpu_core::ResolvedRenderResolveAttachment {
            resource: ResourceId::new(81, 1),
            backing: resolved,
            regions: Box::new([BackingRegion::Whole]),
            pixel_format: 80,
            extent: [32, 16, 1],
            sample_count: 1,
        });
        let mut operation = plan_operation([attachment]);
        operation.vertex_buffers = Box::new([]);
        operation.resources = Box::new([]);
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(9),
            SubmissionId::new(10),
            0,
            operation,
            pipeline(),
        )
        .unwrap();

        let uses = derive_render_image_uses(&prepared).unwrap();
        let resolve_use = uses
            .iter()
            .find(|use_| use_.image.backing == resolved)
            .expect("the resolve target is one of the pass's images");
        assert_eq!(
            resolve_use.use_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert!(!resolve_use
            .required_usage
            .contains(vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT));
    }

    #[test]
    fn a_color_input_attachment_binds_in_the_layout_its_pass_declares() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let target = backing(&mut lifecycle, true);
        let mut attachment = plan_attachment(RenderAttachmentRole::Color(0), 80);
        attachment.backing = target;
        attachment.sample_count = 1;
        // The fragment stage reads this attachment through a subpass input,
        // which the pass can only express as one general layout.
        attachment.input_attachment = true;
        let mut operation = plan_operation([attachment.clone()]);
        operation.vertex_buffers = Box::new([]);
        operation.resources = Box::new([]);
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(9),
            SubmissionId::new(10),
            0,
            operation,
            pipeline(),
        )
        .unwrap();

        let uses = derive_render_image_uses(&prepared).unwrap();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].use_layout, vk::ImageLayout::GENERAL);
        assert!(uses[0]
            .required_usage
            .contains(vk::ImageUsageFlags::INPUT_ATTACHMENT));

        let mut sampled = attachment;
        sampled.sample_count = 4;
        let plan = resolve_render_pipeline_plan(&plan_operation([sampled]), 4).unwrap();
        assert!(plan.color_attachments[0].input_attachment);
    }

    #[test]
    fn one_feedback_aspect_puts_the_whole_depth_stencil_attachment_in_the_loop() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let combined = backing(&mut lifecycle, true);
        let resource = ResourceId::new(252, 1);
        let mut depth = plan_attachment(RenderAttachmentRole::Depth, 252);
        depth.resource = resource;
        depth.backing = combined;
        depth.sample_count = 1;
        let mut stencil = plan_attachment(RenderAttachmentRole::Stencil, 252);
        stencil.resource = resource;
        stencil.backing = combined;
        stencil.sample_count = 1;
        // The encoder reads the depth aspect; the pass declares one layout for
        // the combined attachment and unions the two aspects to reach it.
        depth.feedback_loop = true;
        let mut operation = plan_operation([depth, stencil]);
        operation.vertex_buffers = Box::new([]);
        operation.resources = Box::new([]);
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(9),
            SubmissionId::new(10),
            0,
            operation,
            pipeline(),
        )
        .unwrap();

        let uses = derive_render_image_uses(&prepared).unwrap();
        assert_eq!(uses.len(), 1);
        assert_eq!(
            uses[0].use_layout,
            vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
        );
    }

    #[test]
    fn render_pipeline_plan_refuses_color_gaps_and_sample_mismatch() {
        let gap = plan_operation([plan_attachment(RenderAttachmentRole::Color(1), 81)]);
        assert_eq!(
            resolve_render_pipeline_plan(&gap, 4),
            Err(ReplacementRenderPipelinePlanError::ColorAttachmentGap(0))
        );

        let mut wrong_samples =
            plan_operation([plan_attachment(RenderAttachmentRole::Color(0), 80)]);
        wrong_samples.attachments[0].sample_count = 1;
        assert_eq!(
            resolve_render_pipeline_plan(&wrong_samples, 4),
            Err(
                ReplacementRenderPipelinePlanError::AttachmentSampleCountMismatch(
                    RenderAttachmentRole::Color(0)
                )
            )
        );
        assert_eq!(
            resolve_render_pipeline_plan(&wrong_samples, 3),
            Err(ReplacementRenderPipelinePlanError::UnsupportedSampleCount(
                3
            ))
        );
    }

    #[test]
    fn a_full_view_cannot_stand_in_for_another_attachment_subresource() {
        let key = ReplacementImageKey {
            backing: BackingId::new(80),
            representation: RepresentationId::new(81),
        };
        let resolver = Resolver {
            attachment: key,
            sampled: ReplacementImageKey {
                backing: BackingId::new(82),
                representation: RepresentationId::new(83),
            },
            depth_stencil: None,
            resolve: None,
            visibility: None,
            arguments: None,
            precise_visibility: true,
            null_descriptors: true,
        };
        let region = |mip| {
            BackingRegion::Image(reims_vgpu_core::ImageRegion {
                aspect: reims_vgpu_core::ImageAspect::Color,
                mip,
                layer: 0,
                texels: reims_vgpu_core::TexelBox::new([0, 0, 0], [32, 16, 1]).unwrap(),
            })
        };
        assert!(resolver.resolve_attachment_view(key, region(0)).is_some());
        assert!(resolver.resolve_attachment_view(key, region(1)).is_none());
    }

    #[test]
    fn render_variant_key_separates_exact_shader_specializations() {
        let mut first = plan_operation([]);
        first.program.vertex.id = reims_vgpu_protocol::PreparedShaderId::new(11);
        first.program.fragment.id = reims_vgpu_protocol::PreparedShaderId::new(12);
        let mut second = first.clone();
        second.program.fragment.id = reims_vgpu_protocol::PreparedShaderId::new(13);

        let first = resolve_render_pipeline_plan(&first, 4)
            .expect("first specialization is representable")
            .variant_key();
        let second = resolve_render_pipeline_plan(&second, 4)
            .expect("second specialization is representable")
            .variant_key();

        assert_ne!(first, second);
        assert_eq!(first.vertex_program, second.vertex_program);
        assert_ne!(first.fragment_program, second.fragment_program);
    }

    #[test]
    fn render_variant_key_names_contract_choices_not_native_outputs() {
        let base = pipeline().native.native().clone();
        let key = base.variant_key();
        let mut family = ReplacementRenderPipelineVariants::<()>::default();
        let job = family.begin_compile(key.clone()).unwrap();
        let retained = family
            .compile_complete(
                job,
                ReplacementRenderPipelineVariant::synthetic(base.clone()),
            )
            .unwrap()
            .native;
        assert!(matches!(
            family.readiness(&key).unwrap(),
            reims_vgpu_core::PipelineVariantReadiness::Ready(found)
                if std::sync::Arc::ptr_eq(&found, &retained)
        ));
        let mut changed = base.clone();
        changed.pipeline = vk::Pipeline::from_raw(900);
        changed.render_pass = vk::RenderPass::from_raw(901);
        assert_eq!(changed.variant_key(), key);

        changed = base.clone();
        changed.static_state.cull_mode = CullMode::Back;
        assert_ne!(changed.variant_key(), key);
        changed = base.clone();
        changed.color_attachments[0].store = StoreAction::DontCare;
        assert_ne!(changed.variant_key(), key);
        changed = base.clone();
        changed.viewport_count += 1;
        assert_ne!(changed.variant_key(), key);
        changed = base;
        changed.depth_stencil = Some(ResourceId::new(12, 3));
        assert_ne!(changed.variant_key(), key);
    }

    #[test]
    fn acquired_render_variant_destroys_only_after_recording_retention_ends() {
        struct Device(std::sync::Arc<std::sync::atomic::AtomicUsize>);

        impl ReplacementRenderPipelineDevice for Device {
            fn destroy_render_pipeline_variant(&self, _: &ReplacementRenderPipeline) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let native = pipeline().native.native().clone();
        let key = native.variant_key();
        let destroyed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let device: std::sync::Arc<dyn ReplacementRenderPipelineDevice> =
            std::sync::Arc::new(Device(std::sync::Arc::clone(&destroyed)));
        let mut family = ReplacementRenderPipelineVariants::<()>::default();
        let job = family.begin_compile(key).unwrap();
        let retained = family
            .compile_complete(job, ReplacementRenderPipelineVariant::new(device, native))
            .unwrap();
        drop(family);
        assert_eq!(destroyed.load(std::sync::atomic::Ordering::SeqCst), 0);
        drop(retained);
        assert_eq!(destroyed.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn semantic_pipeline_family_acquires_the_exact_ready_native_variant_lease() {
        let id = ResourceId::new(22, 4);
        let native = pipeline().native.native().clone();
        let key = native.variant_key();
        let family = ReplacementRenderPipelineFamily::<&'static str>::default();
        let job = family.begin_compile(key.clone()).unwrap();
        let retained = family
            .compile_complete(job, ReplacementRenderPipelineVariant::synthetic(native))
            .unwrap()
            .native;

        let mut pipelines = PipelineLifecycle::<
            RenderPipelineObject,
            (),
            ReplacementRenderPipelineFamily<&'static str>,
            (),
        >::default();
        pipelines.declare(id, ()).unwrap();
        let translation = pipelines.begin_translation(id).unwrap();
        let compile = pipelines.translation_complete(translation, ()).unwrap();
        pipelines
            .compile_complete(
                compile,
                reims_vgpu_core::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                family,
            )
            .unwrap();
        let PipelineReadiness::Ready(family_lease) =
            pipelines.readiness(id, TransactionId::new(41)).unwrap()
        else {
            unreachable!()
        };
        let ReplacementRenderPipelineVariantReadiness::Ready(variant_lease) =
            acquire_render_pipeline_variant(&family_lease, &key).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(variant_lease.pipeline, id);
        assert_eq!(
            variant_lease.native_object.vulkan_epoch.id(),
            family_lease.native_object.vulkan_epoch.id()
        );
        assert!(std::sync::Arc::ptr_eq(&variant_lease.native, &retained));
    }

    #[test]
    fn color_draw_projects_exact_attachment_sampled_descriptor_and_draw() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let sampled = backing(&mut lifecycle, true);
        let attachment = backing(&mut lifecycle, false);
        let visibility = view_backing(&mut lifecycle, false, BackingView::Bytes);
        let arguments = view_backing(&mut lifecycle, true, BackingView::Bytes);
        let operation = ResolvedRenderDispatch {
            pipeline: ResourceId::new(2, 1),
            program: Default::default(),
            depth_stencil: None,
            render_extent: [32, 16],
            raster: reims_vgpu_core::ResolvedRenderRasterState {
                viewports: Box::new([reims_vgpu_core::RenderViewport::from_values([
                    1.0, 2.0, 10.0, 8.0, 0.25, 0.75,
                ])]),
                scissors: Box::new([reims_vgpu_core::RenderScissor {
                    x: 3,
                    y: 4,
                    width: 5,
                    height: 6,
                }]),
                blend_color_bits: Some([
                    0.1f32.to_bits(),
                    0.2f32.to_bits(),
                    0.3f32.to_bits(),
                    0.4f32.to_bits(),
                ]),
                stencil_reference: [17, 19],
                ..Default::default()
            },
            visibility: Some(reims_vgpu_core::ResolvedRenderVisibility {
                mode: reims_vgpu_protocol::VisibilityResultMode::Counting,
                resource: ResourceId::new(7, 1),
                backing: visibility,
                range: reims_vgpu_core::LinearRange::new(8, 8).unwrap(),
            }),
            begins_encoder: true,
            ends_encoder: true,
            draw: ResolvedRenderDraw::Direct {
                topology: PrimitiveTopology::Triangle,
                vertex_count: 3,
                instance_count: 2,
                first_vertex: 4,
                first_instance: 5,
            },
            vertex_buffers: Box::new([]),
            attachments: Box::new([ResolvedRenderAttachment {
                role: RenderAttachmentRole::Color(0),
                resource: ResourceId::<ResourceObject>::new(4, 1),
                backing: attachment,
                regions: Box::new([BackingRegion::Whole]),
                pixel_format: 80,
                extent: [32, 16, 1],
                sample_count: 1,
                load: LoadAction::Clear,
                store: StoreAction::Store,
                clear: RenderAttachmentClear::Color([
                    0.25f32.to_bits(),
                    0.5f32.to_bits(),
                    0.75f32.to_bits(),
                    1f32.to_bits(),
                ]),
                resolve: None,
                feedback_loop: false,
                input_attachment: false,
            }]),
            resources: Box::new([ResolvedRenderResourceBinding {
                class: RenderBindingClass::SampledImage,
                binding: 6,
                array_element: 0,
                descriptor_count: 1,
                stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
                resource: ResourceId::new(3, 1),
                backing: sampled,
                view: RenderBindingView::Image(image_view(ResourceId::new(3, 1))),
                regions: Box::new([BackingRegion::Whole]),
                mode: AccessMode::Read,
            }]),
            null_bindings: Box::new([reims_vgpu_core::ResolvedRenderNullBinding {
                class: RenderBindingClass::SampledImage,
                binding: 7,
                array_element: 0,
                descriptor_count: 1,
                stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
            }]),
            samplers: Box::new([reims_vgpu_core::ResolvedRenderSamplerBinding {
                binding: 8,
                array_element: 0,
                descriptor_count: 1,
                stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
                sampler: reims_vgpu_core::SamplerResource::null(8),
            }]),
        };
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(7),
            SubmissionId::new(12),
            3,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        let attachment_key = ReplacementImageKey {
            backing: attachment,
            representation: lifecycle.execution_representation_id(attachment).unwrap(),
        };
        let sampled_key = ReplacementImageKey {
            backing: sampled,
            representation: lifecycle.execution_representation_id(sampled).unwrap(),
        };
        let visibility_representation = lifecycle.execution_representation_id(visibility).unwrap();
        let mut images = ReplacementImageStateOwner::new(EPOCH);
        for (key, layout) in [
            (attachment_key, vk::ImageLayout::UNDEFINED),
            (sampled_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ] {
            images
                .register(
                    key,
                    crate::replacement_image_state::ReplacementImageState {
                        layout,
                        sharing:
                            crate::replacement_image_state::ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let state = prepare_render_image_state(&mut images, &prepared, 2).unwrap();
        let mut resolver = Resolver {
            attachment: attachment_key,
            sampled: sampled_key,
            depth_stencil: None,
            resolve: None,
            visibility: Some((visibility, visibility_representation)),
            arguments: None,
            precise_visibility: false,
            null_descriptors: false,
        };
        assert!(matches!(
            ReplacementRenderProgram::resolve(&prepared, &state, &resolver),
            Err(RenderRecordError::PreciseVisibilityUnavailable)
        ));
        resolver.precise_visibility = true;
        assert!(matches!(
            ReplacementRenderProgram::resolve(&prepared, &state, &resolver),
            Err(RenderRecordError::NullDescriptorUnavailable(8))
        ));
        resolver.null_descriptors = true;
        let program = ReplacementRenderProgram::resolve(&prepared, &state, &resolver).unwrap();
        assert_eq!(program.index(), 3);
        assert_eq!(program.backings(), [sampled, attachment, visibility]);
        assert_eq!(
            program.native().attachment_views.as_ref(),
            [vk::ImageView::from_raw(41)]
        );
        assert_eq!(
            program.native().clear_values[0],
            NativeRenderClear::Color([
                0.25f32.to_bits(),
                0.5f32.to_bits(),
                0.75f32.to_bits(),
                1f32.to_bits()
            ])
        );
        assert!(matches!(
            program.native().descriptors.as_ref(),
            [NativeComputeDescriptor::Image {
                binding: 6,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ..
            }, NativeComputeDescriptor::Sampler {
                binding: 8,
                sampler,
                ..
            }, NativeComputeDescriptor::Image {
                binding: 7,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                view,
                layout: vk::ImageLayout::UNDEFINED,
                ..
            }] if *sampler == vk::Sampler::null() && *view == vk::ImageView::null()
        ));
        assert!(matches!(
            program.native().draw,
            ResolvedRenderDraw::Direct {
                vertex_count: 3,
                instance_count: 2,
                first_vertex: 4,
                first_instance: 5,
                ..
            }
        ));
        assert!(program.native().image_state.releases.is_empty());
        let [viewport] = program.native().viewports.as_ref() else {
            panic!("one exact viewport must be projected")
        };
        assert_eq!(
            [
                viewport.x,
                viewport.y,
                viewport.width,
                viewport.height,
                viewport.min_depth,
                viewport.max_depth,
            ],
            [1.0, 10.0, 10.0, -8.0, 0.25, 0.75]
        );
        let [scissor] = program.native().scissors.as_ref() else {
            panic!("one exact scissor must be projected")
        };
        assert_eq!([scissor.offset.x, scissor.offset.y], [3, 4]);
        assert_eq!([scissor.extent.width, scissor.extent.height], [5, 6]);
        assert_eq!(program.native().blend_color, Some([0.1, 0.2, 0.3, 0.4]));
        assert_eq!(program.native().stencil_reference, [17, 19]);
        let visibility = program
            .native()
            .visibility
            .expect("the exact visibility write must be projected");
        assert_eq!(visibility.buffer, vk::Buffer::from_raw(50));
        assert_eq!(visibility.offset, 24);
        assert_eq!(
            visibility.mode,
            reims_vgpu_protocol::VisibilityResultMode::Counting
        );
        assert_eq!(
            program.native().image_state.transitions.before.images.len(),
            2
        );

        let argument_representation = lifecycle.execution_representation_id(arguments).unwrap();
        let mut indirect = operation;
        indirect.draw = ResolvedRenderDraw::IndexedIndirect {
            topology: PrimitiveTopology::Triangle,
            index_type: reims_vgpu_protocol::IndexType::U16,
        };
        let argument_range =
            reims_vgpu_core::LinearRange::new(16, reims_vgpu_core::RENDER_INDIRECT_ARGUMENT_BYTES)
                .unwrap();
        let mut resources = indirect.resources.into_vec();
        let index_range = reims_vgpu_core::LinearRange::new(0, 64).unwrap();
        resources.push(ResolvedRenderResourceBinding {
            class: RenderBindingClass::IndexBuffer,
            binding: 0,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::VERTEX.into()).unwrap(),
            resource: ResourceId::new(9, 1),
            backing: arguments,
            view: RenderBindingView::Buffer(index_range),
            regions: Box::new([BackingRegion::Linear(index_range)]),
            mode: AccessMode::Read,
        });
        resources.push(ResolvedRenderResourceBinding {
            class: RenderBindingClass::IndirectBuffer,
            binding: 0,
            array_element: 0,
            descriptor_count: 1,
            stages: RenderStages::from_bits(RenderStages::VERTEX.into()).unwrap(),
            resource: ResourceId::new(9, 1),
            backing: arguments,
            view: RenderBindingView::Buffer(argument_range),
            regions: Box::new([BackingRegion::Linear(argument_range)]),
            mode: AccessMode::Read,
        });
        indirect.resources = resources.into_boxed_slice();
        let indirect = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(8),
            SubmissionId::new(13),
            4,
            indirect,
            pipeline(),
        )
        .unwrap();
        let mut indirect_images = ReplacementImageStateOwner::new(EPOCH);
        for (key, layout) in [
            (attachment_key, vk::ImageLayout::UNDEFINED),
            (sampled_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ] {
            indirect_images
                .register(
                    key,
                    crate::replacement_image_state::ReplacementImageState {
                        layout,
                        sharing:
                            crate::replacement_image_state::ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let indirect_state =
            prepare_render_image_state(&mut indirect_images, &indirect, 2).unwrap();
        resolver.arguments = Some((arguments, argument_representation));
        let indirect =
            ReplacementRenderProgram::resolve(&indirect, &indirect_state, &resolver).unwrap();
        assert_eq!(
            indirect.native().indirect_buffer,
            Some((vk::Buffer::from_raw(51), 24))
        );
        assert_eq!(
            indirect.native().index_buffer,
            Some((vk::Buffer::from_raw(51), 8, vk::IndexType::UINT16,))
        );

        let mut continued_packet = program.clone();
        continued_packet.operation.begins_encoder = false;
        continued_packet.operation.ends_encoder = false;
        join_native_render_passes(std::slice::from_mut(&mut continued_packet)).unwrap();
        assert!(continued_packet.native.begins_native_pass);
        assert!(continued_packet.native.ends_native_pass);

        let mut first = program.clone();
        first.operation.ends_encoder = false;
        let mut last = program.clone();
        last.index += 1;
        last.operation.begins_encoder = false;
        let mut joined = [first, last];
        join_native_render_passes(&mut joined).unwrap();
        assert!(!joined[0].native.ends_native_pass);
        assert!(!joined[1].native.begins_native_pass);

        joined[0].native.ends_native_pass = true;
        joined[1].native.begins_native_pass = true;
        let mut incompatible = joined[1].native.pipeline.native().clone();
        incompatible.render_pass = vk::RenderPass::from_raw(99);
        joined[1].native.pipeline =
            Arc::new(ReplacementRenderPipelineVariant::synthetic(incompatible));
        join_native_render_passes(&mut joined).unwrap();
        assert!(!joined[0].native.ends_native_pass);
        assert!(!joined[1].native.begins_native_pass);

        joined[0].native.ends_native_pass = true;
        joined[1].native.begins_native_pass = true;
        let mut incompatible = joined[1].native.pipeline.native().clone();
        incompatible.color_attachments[0].format = 81;
        joined[1].native.pipeline =
            Arc::new(ReplacementRenderPipelineVariant::synthetic(incompatible));
        assert_eq!(
            join_native_render_passes(&mut joined),
            Err(RenderExecProgramError::NativePassReopenRequired(4))
        );
    }

    #[test]
    fn combined_depth_stencil_projects_one_view_and_independent_clear_values() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let depth_stencil = backing(&mut lifecycle, false);
        let operation = ResolvedRenderDispatch {
            pipeline: ResourceId::new(2, 1),
            program: Default::default(),
            depth_stencil: None,
            render_extent: [32, 16],
            raster: Default::default(),
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
            attachments: Box::new([
                ResolvedRenderAttachment {
                    role: RenderAttachmentRole::Depth,
                    resource: ResourceId::new(5, 1),
                    backing: depth_stencil,
                    regions: Box::new([BackingRegion::Whole]),
                    pixel_format: 252,
                    extent: [32, 16, 1],
                    sample_count: 1,
                    load: LoadAction::Clear,
                    store: StoreAction::Store,
                    clear: RenderAttachmentClear::Depth(0.75f32.to_bits()),
                    resolve: None,
                    feedback_loop: false,
                    input_attachment: false,
                },
                ResolvedRenderAttachment {
                    role: RenderAttachmentRole::Stencil,
                    resource: ResourceId::new(5, 1),
                    backing: depth_stencil,
                    regions: Box::new([BackingRegion::Whole]),
                    pixel_format: 252,
                    extent: [32, 16, 1],
                    sample_count: 1,
                    load: LoadAction::Clear,
                    store: StoreAction::Store,
                    clear: RenderAttachmentClear::Stencil(27),
                    resolve: None,
                    feedback_loop: false,
                    input_attachment: false,
                },
            ]),
            resources: Box::new([]),
            null_bindings: Box::new([]),
            samplers: Box::new([]),
        };
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(7),
            SubmissionId::new(13),
            1,
            operation,
            depth_pipeline(),
        )
        .unwrap();
        let key = ReplacementImageKey {
            backing: depth_stencil,
            representation: lifecycle
                .execution_representation_id(depth_stencil)
                .unwrap(),
        };
        let mut images = ReplacementImageStateOwner::new(EPOCH);
        images
            .register(
                key,
                crate::replacement_image_state::ReplacementImageState {
                    layout: vk::ImageLayout::UNDEFINED,
                    sharing: crate::replacement_image_state::ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let state = prepare_render_image_state(&mut images, &prepared, 2).unwrap();
        assert_eq!(state.transitions().len(), 1);
        let program = ReplacementRenderProgram::resolve(
            &prepared,
            &state,
            &Resolver {
                attachment: ReplacementImageKey {
                    backing: BackingId::new(u64::MAX),
                    representation: RepresentationId::new(u64::MAX),
                },
                sampled: ReplacementImageKey {
                    backing: BackingId::new(u64::MAX - 1),
                    representation: RepresentationId::new(u64::MAX - 1),
                },
                depth_stencil: Some(key),
                resolve: None,
                visibility: None,
                arguments: None,
                precise_visibility: true,
                null_descriptors: true,
            },
        )
        .unwrap();
        assert_eq!(
            program.native().attachment_views.as_ref(),
            [vk::ImageView::from_raw(43)]
        );
        assert_eq!(
            program.native().clear_values.as_ref(),
            [NativeRenderClear::DepthStencil {
                depth_bits: 0.75f32.to_bits(),
                stencil: 27,
            }]
        );
    }

    #[test]
    fn multisample_color_resolve_retains_both_views_in_render_pass_order() {
        let mut lifecycle = ResourceLifecycleOwner::new(EPOCH);
        let attachment = backing(&mut lifecycle, false);
        let resolve = backing(&mut lifecycle, false);
        let operation = ResolvedRenderDispatch {
            pipeline: ResourceId::new(2, 1),
            program: Default::default(),
            depth_stencil: None,
            render_extent: [32, 16],
            raster: Default::default(),
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
                resource: ResourceId::new(5, 1),
                backing: attachment,
                regions: Box::new([BackingRegion::Whole]),
                pixel_format: 80,
                extent: [32, 16, 1],
                sample_count: 4,
                load: LoadAction::Clear,
                store: StoreAction::MultisampleResolve,
                clear: RenderAttachmentClear::Color([0; 4]),
                resolve: Some(reims_vgpu_core::ResolvedRenderResolveAttachment {
                    resource: ResourceId::new(6, 1),
                    backing: resolve,
                    regions: Box::new([BackingRegion::Whole]),
                    pixel_format: 80,
                    extent: [32, 16, 1],
                    sample_count: 1,
                }),
                feedback_loop: false,
                input_attachment: false,
            }]),
            resources: Box::new([]),
            null_bindings: Box::new([]),
            samplers: Box::new([]),
        };
        let prepared = prepare_render_dispatch(
            &mut lifecycle,
            TransactionId::new(7),
            SubmissionId::new(14),
            2,
            operation,
            color_resolve_pipeline(),
        )
        .unwrap();
        let attachment_key = ReplacementImageKey {
            backing: attachment,
            representation: lifecycle.execution_representation_id(attachment).unwrap(),
        };
        let resolve_key = ReplacementImageKey {
            backing: resolve,
            representation: lifecycle.execution_representation_id(resolve).unwrap(),
        };
        let mut images = ReplacementImageStateOwner::new(EPOCH);
        for key in [attachment_key, resolve_key] {
            images
                .register(
                    key,
                    crate::replacement_image_state::ReplacementImageState {
                        layout: vk::ImageLayout::UNDEFINED,
                        sharing:
                            crate::replacement_image_state::ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let state = prepare_render_image_state(&mut images, &prepared, 2).unwrap();
        let program = ReplacementRenderProgram::resolve(
            &prepared,
            &state,
            &Resolver {
                attachment: attachment_key,
                sampled: ReplacementImageKey {
                    backing: BackingId::new(u64::MAX),
                    representation: RepresentationId::new(u64::MAX),
                },
                depth_stencil: None,
                resolve: Some(resolve_key),
                visibility: None,
                arguments: None,
                precise_visibility: true,
                null_descriptors: true,
            },
        )
        .unwrap();
        assert_eq!(
            program.native().attachment_views.as_ref(),
            [vk::ImageView::from_raw(41), vk::ImageView::from_raw(44)]
        );
        assert_eq!(
            program.native().clear_values.as_ref(),
            [
                NativeRenderClear::Color([0; 4]),
                NativeRenderClear::Color([0; 4]),
            ]
        );
        assert!(prepared
            .completions()
            .contains(&ResolvedResourceCompletion::Discard {
                backing: attachment,
                region: BackingRegion::Whole,
            }));
        assert!(prepared.completions().iter().any(|completion| {
            matches!(completion, ResolvedResourceCompletion::GpuWrite { backing, .. } if *backing == resolve)
        }));
    }
}
