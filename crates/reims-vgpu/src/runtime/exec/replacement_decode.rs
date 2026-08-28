//! Lossless replacement decoding for one complete EXEC child-stream set.
//!
//! This boundary owns segment framing, encoder continuation, and operation
//! family decoding. It deliberately stops before task-name resolution: a
//! caller either receives every decoded record in order or one typed refusal,
//! so production cutover cannot retain the legacy pattern of executing a
//! decoded prefix and silently dropping the malformed suffix.

#![allow(dead_code)]

use crate::runtime::decode::{blit, compute, event, render, stream};
use crate::runtime::replacement_exec_support::{
    classify_info_record, semantic_segment_boundary, InfoRecordDecline,
};
use crate::runtime::replacement_session::ReplacementRuntimeSession;
use reims_vgpu_core::{EncoderBoundary, ResolvedExecSegment, ResolvedExecStream};
use reims_vgpu_protocol::{InfoOperation, SegmentBoundary};

#[derive(Default)]
pub(crate) struct ReplacementRecordProjectionState {
    render: Option<ReplacementRenderEncoderState>,
    compute: Option<crate::runtime::replacement_compute_state::ComputeAccum>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementRenderBufferBinding {
    pub reference: u32,
    pub offset: u64,
    pub attribute_stride: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementRenderSamplerBinding {
    pub reference: u32,
    pub lod_clamp: Option<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementTessellationFactorBufferBinding {
    pub reference: u32,
    pub offset: u64,
    pub instance_stride: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReplacementRenderEncoderState {
    pub pipeline_ref: u32,
    pub vertex_buffers: Vec<Option<ReplacementRenderBufferBinding>>,
    pub fragment_buffers: Vec<Option<ReplacementRenderBufferBinding>>,
    pub vertex_textures: Vec<Option<u32>>,
    pub fragment_textures: Vec<Option<u32>>,
    pub vertex_samplers: Vec<Option<ReplacementRenderSamplerBinding>>,
    pub fragment_samplers: Vec<Option<ReplacementRenderSamplerBinding>>,
    pub viewports_bits: Vec<[u64; 6]>,
    pub scissors: Vec<render::ScissorRect>,
    pub blend_color_bits: Option<[u32; 4]>,
    pub cull_mode: reims_vgpu_protocol::CullMode,
    pub front_face_ccw: bool,
    pub fill_mode: reims_vgpu_protocol::FillMode,
    pub line_width: reims_vgpu_core::LineWidth,
    pub depth_clip_mode: reims_vgpu_protocol::DepthClipMode,
    pub depth_bias_bits: Option<[u32; 3]>,
    pub depth_stencil_ref: u32,
    pub stencil_reference: Option<(u32, u32)>,
    pub tessellation_factor_buffer: Option<ReplacementTessellationFactorBufferBinding>,
    pub color_attachments: Vec<render::ColorAttachment>,
    pub depth_attachment: render::DepthAttachment,
    pub stencil_attachment: render::StencilAttachment,
    pub visibility_result_buffer_ref: u32,
    pub visibility: Option<(reims_vgpu_protocol::VisibilityResultMode, u64)>,
    pub render_target_array_length: u64,
    pub render_target_width: u64,
    pub render_target_height: u64,
}

impl Default for ReplacementRenderEncoderState {
    fn default() -> Self {
        Self {
            pipeline_ref: 0,
            vertex_buffers: vec![None; reims_vgpu_core::MAX_BUFFER_BIND_SLOTS as usize],
            fragment_buffers: vec![None; reims_vgpu_core::MAX_BUFFER_BIND_SLOTS as usize],
            vertex_textures: vec![None; reims_vgpu_core::MAX_TEXTURE_BIND_SLOTS as usize],
            fragment_textures: vec![None; reims_vgpu_core::MAX_TEXTURE_BIND_SLOTS as usize],
            vertex_samplers: vec![None; reims_vgpu_core::MAX_SAMPLER_BIND_SLOTS as usize],
            fragment_samplers: vec![None; reims_vgpu_core::MAX_SAMPLER_BIND_SLOTS as usize],
            viewports_bits: Vec::new(),
            scissors: Vec::new(),
            blend_color_bits: None,
            cull_mode: Default::default(),
            front_face_ccw: false,
            fill_mode: Default::default(),
            line_width: Default::default(),
            depth_clip_mode: Default::default(),
            depth_bias_bits: None,
            depth_stencil_ref: 0,
            stencil_reference: None,
            tessellation_factor_buffer: None,
            color_attachments: Vec::new(),
            depth_attachment: Default::default(),
            stencil_attachment: Default::default(),
            visibility_result_buffer_ref: 0,
            visibility: None,
            render_target_array_length: 0,
            render_target_width: 0,
            render_target_height: 0,
        }
    }
}

impl ReplacementRecordProjectionState {
    pub(crate) fn encoder_boundary(&mut self, boundary: EncoderBoundary) {
        match boundary {
            EncoderBoundary::Begin(reims_vgpu_protocol::SegmentKind::Render) => {
                self.render = Some(ReplacementRenderEncoderState::default());
            }
            EncoderBoundary::End(reims_vgpu_protocol::SegmentKind::Render) => {
                self.render = None;
            }
            EncoderBoundary::Begin(reims_vgpu_protocol::SegmentKind::Compute) => {
                self.compute =
                    Some(crate::runtime::replacement_compute_state::ComputeAccum::default());
            }
            EncoderBoundary::End(reims_vgpu_protocol::SegmentKind::Compute) => {
                self.compute = None;
            }
            EncoderBoundary::Begin(_) | EncoderBoundary::End(_) => {}
        }
    }

    fn compute_mut(
        &mut self,
    ) -> Result<
        &mut crate::runtime::replacement_compute_state::ComputeAccum,
        ReplacementComputeStateResolutionError,
    > {
        self.compute
            .as_mut()
            .ok_or(ReplacementComputeStateResolutionError::OutsideEncoder)
    }

    pub(crate) fn render(&self) -> Option<&ReplacementRenderEncoderState> {
        self.render.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DecodedReplacementRecord {
    Render(Box<render::Command>),
    Compute(Box<compute::Command>),
    Blit(Box<blit::Command>),
    Event(event::Command),
    Info(InfoOperation),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecodedReplacementSegment {
    pub boundary: SegmentBoundary,
    pub records: Box<[DecodedReplacementRecord]>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecodedReplacementStream {
    pub stream_index: u32,
    pub segments: Box<[DecodedReplacementSegment]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementStreamObjectManifest {
    pub objects: Box<[reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>]>,
}

/// Collect every object-list slot explicitly named by already-decoded stream
/// records. Serializer families such as pipelines, samplers, depth state,
/// functions reached by pipelines, rate maps, and ICBs are constructed from
/// this same heterogeneous list; heap, event, fence, and mapper identities are
/// deliberately excluded because their namespaces have different owners.
pub(crate) fn replacement_stream_object_manifest(
    streams: &[DecodedReplacementStream],
) -> ReplacementStreamObjectManifest {
    use reims_vgpu_protocol::{InfoOperation, ObjectTableRef, ResourceObject};

    fn push(
        objects: &mut Vec<ObjectTableRef<ResourceObject>>,
        seen: &mut std::collections::BTreeSet<u32>,
        reference: u32,
    ) {
        if reference != 0 && seen.insert(reference) {
            objects.push(ObjectTableRef::new(reference));
        }
    }

    fn push_reply(
        objects: &mut Vec<ObjectTableRef<ResourceObject>>,
        seen: &mut std::collections::BTreeSet<u32>,
        reply: reims_vgpu_protocol::InfoReplyTarget,
    ) {
        push(objects, seen, reply.buffer.get());
    }

    let mut objects = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for record in streams
        .iter()
        .flat_map(|stream| stream.segments.iter())
        .flat_map(|segment| segment.records.iter())
    {
        match record {
            DecodedReplacementRecord::Render(command) => {
                push(&mut objects, &mut seen, command.pipeline_ref);
                push(&mut objects, &mut seen, command.buffer_ref);
                for binding in &command.buffer_binds {
                    push(&mut objects, &mut seen, binding.buffer_ref);
                }
                push(&mut objects, &mut seen, command.texture_ref);
                for &reference in &command.ref_binds {
                    push(&mut objects, &mut seen, reference);
                }
                push(&mut objects, &mut seen, command.sampler_ref);
                for resource in &command.participation_resources {
                    push(&mut objects, &mut seen, resource.get());
                }
                for resource in &command.barrier_resources {
                    push(&mut objects, &mut seen, resource.get());
                }
                push(&mut objects, &mut seen, command.index_buffer_ref);
                push(&mut objects, &mut seen, command.indirect_buffer_ref);
                push(
                    &mut objects,
                    &mut seen,
                    command.pass_visibility_result_buffer_ref,
                );
                push(&mut objects, &mut seen, command.depth_stencil_ref);
                push(&mut objects, &mut seen, command.indirect_command_buffer_ref);
                push(&mut objects, &mut seen, command.icb_args_buffer_ref);
                for attachment in command
                    .color_attachments
                    .iter()
                    .chain(std::iter::once(&command.color0))
                {
                    push(&mut objects, &mut seen, attachment.texture_ref);
                    push(&mut objects, &mut seen, attachment.resolve_texture_ref);
                }
                push(&mut objects, &mut seen, command.depth.texture_ref);
                push(&mut objects, &mut seen, command.depth.resolve_texture_ref);
                push(&mut objects, &mut seen, command.stencil.texture_ref);
                push(&mut objects, &mut seen, command.stencil.resolve_texture_ref);
                if let Some(draw) = command.patch_draw {
                    let (patch_indices, control_points, arguments) = match draw {
                        render::PatchDraw::Direct {
                            patch_indices,
                            control_point_indices,
                            ..
                        } => (patch_indices, control_point_indices, None),
                        render::PatchDraw::Indirect {
                            patch_indices,
                            control_point_indices,
                            arguments,
                            ..
                        } => (patch_indices, control_point_indices, Some(arguments)),
                    };
                    push(&mut objects, &mut seen, patch_indices.reference);
                    if let Some(control_points) = control_points {
                        push(&mut objects, &mut seen, control_points.reference);
                    }
                    if let Some(arguments) = arguments {
                        push(&mut objects, &mut seen, arguments.reference);
                    }
                }
                if let Some(bind) = &command.tile_bind {
                    match bind {
                        render::TileBind::Buffer { bindings, .. } => {
                            for binding in bindings {
                                push(&mut objects, &mut seen, binding.buffer_ref);
                            }
                        }
                        render::TileBind::Texture { references, .. } => {
                            for &reference in references {
                                push(&mut objects, &mut seen, reference);
                            }
                        }
                        render::TileBind::Sampler { bindings, .. } => {
                            for binding in bindings {
                                push(&mut objects, &mut seen, binding.reference);
                            }
                        }
                        render::TileBind::BufferOffset { .. }
                        | render::TileBind::ThreadgroupMemory(_) => {}
                    }
                }
            }
            DecodedReplacementRecord::Compute(command) => {
                push(&mut objects, &mut seen, command.pipeline_ref);
                for binding in &command.buffers {
                    push(&mut objects, &mut seen, binding.ref_);
                }
                for binding in &command.textures {
                    push(&mut objects, &mut seen, binding.ref_);
                }
                for binding in &command.samplers {
                    push(&mut objects, &mut seen, binding.ref_);
                }
                for resource in &command.resources {
                    push(&mut objects, &mut seen, resource.get());
                }
                push(&mut objects, &mut seen, command.indirect_buffer_ref);
                push(
                    &mut objects,
                    &mut seen,
                    command.stage_in_indirect_buffer_ref,
                );
                push(&mut objects, &mut seen, command.condition_buffer_ref);
                push(&mut objects, &mut seen, command.indirect_command_buffer_ref);
                push(
                    &mut objects,
                    &mut seen,
                    command.indirect_command_arguments_buffer_ref,
                );
            }
            DecodedReplacementRecord::Blit(command) => {
                push(&mut objects, &mut seen, command.source);
                push(&mut objects, &mut seen, command.destination);
                push(&mut objects, &mut seen, command.resource);
                push(&mut objects, &mut seen, command.buffer);
                push(&mut objects, &mut seen, command.texture);
                push(&mut objects, &mut seen, command.fill_bytes_ref);
            }
            DecodedReplacementRecord::Info(operation) => match operation {
                InfoOperation::PipelineState {
                    pipeline_ref,
                    reply,
                    ..
                } => {
                    push(&mut objects, &mut seen, *pipeline_ref);
                    push_reply(&mut objects, &mut seen, *reply);
                }
                InfoOperation::ResourceHost {
                    resource, reply, ..
                } => {
                    push(&mut objects, &mut seen, resource.get());
                    push_reply(&mut objects, &mut seen, *reply);
                }
                InfoOperation::SamplerHost { sampler, reply } => {
                    push(&mut objects, &mut seen, sampler.get());
                    push_reply(&mut objects, &mut seen, *reply);
                }
                InfoOperation::RenderPipelineImageblock {
                    pipeline, reply, ..
                } => {
                    push(&mut objects, &mut seen, pipeline.get());
                    push_reply(&mut objects, &mut seen, *reply);
                }
                InfoOperation::ComputePipelineImageblock {
                    pipeline, reply, ..
                } => {
                    push(&mut objects, &mut seen, pipeline.get());
                    push_reply(&mut objects, &mut seen, *reply);
                }
                InfoOperation::RateMapInfo {
                    rate_map, reply, ..
                }
                | InfoOperation::MapCoordinate {
                    rate_map, reply, ..
                } => {
                    push(&mut objects, &mut seen, rate_map.get());
                    push_reply(&mut objects, &mut seen, *reply);
                }
                InfoOperation::CopyRateParameterBuffer {
                    rate_map,
                    destination,
                    ..
                } => {
                    push(&mut objects, &mut seen, rate_map.get());
                    push(&mut objects, &mut seen, destination.get());
                }
                InfoOperation::HeapHost { reply, .. }
                | InfoOperation::HeapTextureSizeAndAlign { reply, .. } => {
                    push_reply(&mut objects, &mut seen, *reply);
                }
            },
            DecodedReplacementRecord::Event(_) => {}
        }
    }
    ReplacementStreamObjectManifest {
        objects: objects.into_boxed_slice(),
    }
}

#[derive(Debug)]
pub(crate) struct ReplacementExecProjectionFailure<Error> {
    pub reason: Error,
    pub streams: Box<[DecodedReplacementStream]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplacementRecordDecodeRefusal {
    Render(render::DecodeStatus),
    Compute(compute::DecodeStatus),
    Blit(blit::DecodeStatus),
    Event(event::DecodeStatus),
    Info(InfoRecordDecline),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplacementExecDecodeRefusal {
    Stream {
        stream_index: u32,
        reason: stream::DecodeStatus,
    },
    UnknownSegment {
        stream_index: u32,
        segment_index: u32,
        type_: u8,
    },
    ContinuationWithoutPrevious {
        boundary: SegmentBoundary,
    },
    ContinuationTypeMismatch {
        boundary: SegmentBoundary,
        previous_type: u8,
    },
    RestartBeforeClose {
        boundary: SegmentBoundary,
        previous_type: u8,
    },
    UnclosedEncoder {
        previous_type: u8,
    },
    Record {
        boundary: SegmentBoundary,
        opcode: u32,
        offset: u32,
        reason: ReplacementRecordDecodeRefusal,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementEventResolutionError {
    TimeoutUnsupported,
    UnknownKind,
    Namespace(reims_vgpu_core::ConditionNamespaceError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementFenceResolutionError {
    NotFenceRecord,
    UnknownFenceOpcode,
    UnsupportedRenderStages(u32),
    Namespace(reims_vgpu_core::ConditionNamespaceError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementParticipationResolutionError {
    NotParticipationRecord,
    MissingResourceUsage,
    ResourceUnavailable { index: usize, reference: u32 },
    HeapUnavailable { index: usize, reference: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementBarrierResolutionError {
    NotBarrierRecord,
    TextureBarrierContractUnresolved,
    UnsupportedProducerStages(u16),
    UnsupportedConsumerStages(u16),
    UnsupportedScope(u16),
    UnidentifiedRenderField(u8),
    ResourceUnavailable { index: usize, reference: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementBufferBlitEndpoint {
    Source,
    Destination,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementBufferBlitResolutionError {
    NotBufferBlit,
    UnalignedWordPattern(u64),
    Range {
        endpoint: ReplacementBufferBlitEndpoint,
        reference: u32,
        reason: crate::runtime::replacement_session::ReplacementBufferRangeResolutionError,
    },
    OverlappingCopy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementIndirectCommandResolutionError {
    NotIndirectCommand,
    UnknownRangeOpcode(u32),
    Resolve(reims_vgpu_core::IndirectCommandResolutionError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeStateResolutionError {
    NotStateRecord,
    OutsideEncoder,
    BufferIndexOverflow { first: u32, count: u64, limit: u32 },
    TextureIndexOverflow { first: u32, count: u64, limit: u32 },
    SamplerIndexOverflow { first: u32, count: u64, limit: u32 },
    BufferOffsetUnbound { index: u32 },
    DispatchTypeUnsupported(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderStateResolutionError {
    NotStateRecord,
    OutsideEncoder,
    UnknownStage,
    BufferIndexOverflow {
        index: u32,
        limit: u32,
    },
    TextureIndexOverflow {
        index: u32,
        limit: u32,
    },
    SamplerIndexOverflow {
        index: u32,
        limit: u32,
    },
    BufferOffsetUnbound {
        index: u32,
    },
    EmptyScissor {
        index: usize,
    },
    InvalidRasterState {
        opcode: u32,
        raw: u64,
    },
    InvalidStoreAction {
        raw: u64,
    },
    InvalidStoreActionOptions {
        raw: u64,
    },
    StoreActionAttachmentAbsent,
    StoreActionColorSlotOutOfRange {
        index: u32,
    },
    InvalidVisibilityMode {
        raw: u64,
    },
    VertexAmplificationUnsupported {
        count: u32,
        mode: u64,
        value: u32,
    },
    VertexAmplificationMappingUnsupported {
        index: usize,
        viewport_offset: u32,
        render_target_offset: u32,
    },
    TessellationFactorScaleUnsupported {
        bits: u32,
    },
    DefaultRasterSampleCountUnsupported {
        count: u64,
    },
    RenderPassPropertyUnsupported {
        opcode: u32,
        value: u64,
        reference: u32,
        count: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderPassActionKind {
    Load,
    Store,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderPassResolutionError {
    OutsideEncoder,
    PassAction {
        role: reims_vgpu_core::RenderAttachmentRole,
        kind: ReplacementRenderPassActionKind,
        raw: u16,
    },
    Attachment {
        role: reims_vgpu_core::RenderAttachmentRole,
        reason: crate::runtime::replacement_session::ReplacementRenderAttachmentResolutionError,
    },
    StoreActionOptionsUnsupported {
        role: reims_vgpu_core::RenderAttachmentRole,
        raw: u16,
    },
    ResolveFilterUnsupported {
        role: reims_vgpu_core::RenderAttachmentRole,
        raw: u16,
    },
    ResolveMismatch(reims_vgpu_core::RenderAttachmentRole),
    DepthStencilAttachmentMismatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderDrawResolutionError {
    NotDraw,
    OutsideEncoder,
    PipelineUnbound,
    Pipeline(crate::runtime::replacement_session::ReplacementRenderSemanticAvailability),
    PrimitiveTopology(reims_vgpu_protocol::PipelineStateDecodeError),
    RenderExtentOutOfRange,
    RenderExtentPastAttachment {
        role: reims_vgpu_core::RenderAttachmentRole,
        requested: [u32; 2],
        available: [u32; 2],
    },
    RenderTargetArrayLengthUnsupported {
        length: u64,
    },
    DepthStencilStateMissing,
    ReflectedResourceUnrepresented {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
        kind: reims_vgpu_core::ShaderResourceKind,
    },
    ReflectedInterfaceUnrepresented {
        stage: reims_vgpu_core::ShaderStage,
        feature: &'static str,
        count: usize,
    },
    SamplerMissing {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
    },
    SamplerState(crate::runtime::replacement_services::RenderTranslationDecline),
    SamplerBindingCollision(u32),
    Texture {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
        reason: crate::runtime::replacement_session::ReplacementTextureResolutionError,
    },
    TextureShape {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
        shape: reims_vgpu_core::ReflectedSampledKind,
    },
    TextureShapeMismatch {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
        expected: reims_vgpu_core::SampledImageKind,
        actual: reims_vgpu_protocol::TextureType,
    },
    TextureAccessUnknown {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
    },
    StorageImageAccessMissing(u32),
    StorageImageAccessAmbiguous(u32),
    TextureBindingCollision {
        class: reims_vgpu_core::RenderBindingClass,
        binding: u32,
        array_element: u32,
    },
    BufferMissing {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
    },
    Buffer {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
        reason: crate::runtime::replacement_session::ReplacementBufferRangeResolutionError,
    },
    BufferExtentPastBinding {
        stage: reims_vgpu_core::ShaderStage,
        index: u32,
        required: u64,
        available: u64,
    },
    BufferBindingCollision(u32),
    VertexAttributeFormat {
        location: u32,
        reason: reims_vgpu_protocol::VertexFormatDecodeError,
    },
    VertexStepFunction {
        location: u32,
        reason: reims_vgpu_protocol::VertexStepDecodeError,
    },
    VertexStepRate {
        location: u32,
        step_function: u32,
        step_rate: u32,
    },
    VertexBufferMissing(u32),
    DynamicVertexStrideOutOfRange {
        index: u32,
        bound: u64,
    },
    VertexBuffer {
        index: u32,
        reason: crate::runtime::replacement_session::ReplacementBufferRangeResolutionError,
    },
    VertexBindingCollision(u32),
    Attachment(ReplacementRenderPassResolutionError),
    IndexType(reims_vgpu_protocol::IndexTypeDecodeError),
    IndexVertexOffsetOutOfRange,
    IndexRangeOverflow,
    IndexBuffer(crate::runtime::replacement_session::ReplacementBufferRangeResolutionError),
    IndirectBuffer(crate::runtime::replacement_session::ReplacementBufferRangeResolutionError),
    IndexedBufferMissing,
    VisibilityBufferMissing,
    VisibilityBuffer(crate::runtime::replacement_session::ReplacementBufferRangeResolutionError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderTessellationResolutionError {
    OutsideEncoder {
        opcode: u32,
    },
    GeometryMissing {
        opcode: u32,
    },
    Unsupported {
        opcode: u32,
        draw: render::PatchDraw,
        factor_buffer: Option<ReplacementTessellationFactorBufferBinding>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderTileResolutionError {
    OutsideEncoder {
        kind: render::Kind,
        opcode: u32,
    },
    Malformed {
        kind: render::Kind,
        opcode: u32,
    },
    BindUnsupported {
        opcode: u32,
        binding: render::TileBind,
    },
    DispatchUnsupported {
        opcode: u32,
        threads_per_tile: [u64; 3],
        region: Option<render::TileRegion>,
        render_target_array_index: Option<u32>,
    },
    DimensionsQueryUnsupported {
        opcode: u32,
        buffer_ref: u32,
        offset: u64,
    },
}

fn render_stage_bits(stage: reims_vgpu_core::ShaderStage) -> reims_vgpu_protocol::RenderStages {
    let bits = match stage {
        reims_vgpu_core::ShaderStage::Vertex => reims_vgpu_protocol::RenderStages::VERTEX,
        reims_vgpu_core::ShaderStage::Fragment => reims_vgpu_protocol::RenderStages::FRAGMENT,
        reims_vgpu_core::ShaderStage::Unknown => unreachable!("a render shader stage is known"),
    };
    reims_vgpu_protocol::RenderStages::from_bits(bits.into())
        .expect("one declared render-stage bit is valid")
}

fn resolve_render_samplers<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRenderEncoderState,
    pipeline: &reims_vgpu_core::ResolvedRenderPipeline,
) -> Result<
    Box<[reims_vgpu_core::ResolvedRenderSamplerBinding]>,
    ReplacementRenderDrawResolutionError,
> {
    let mut resolved: Vec<reims_vgpu_core::ResolvedRenderSamplerBinding> = Vec::new();
    for (stage, family, table) in [
        (
            reims_vgpu_core::ShaderStage::Vertex,
            &pipeline.vertex,
            &state.vertex_samplers,
        ),
        (
            reims_vgpu_core::ShaderStage::Fragment,
            &pipeline.fragment,
            &state.fragment_samplers,
        ),
    ] {
        for reflected in family.variant().samplers.iter() {
            let mut sampler = if let Some(static_state) = reflected.static_state {
                crate::runtime::replacement_sampler_projection::reflected_sampler(
                    stage.name(),
                    reflected.binding,
                    static_state,
                )
                .map_err(ReplacementRenderDrawResolutionError::SamplerState)?
            } else if let Some(bound) = table.get(reflected.metal_index as usize).copied().flatten()
            {
                let (identity, descriptor) = runtime
                    .resolve_sampler(
                        task,
                        reims_vgpu_protocol::SerializerRef::new(bound.reference),
                    )
                    .ok_or(ReplacementRenderDrawResolutionError::SamplerMissing {
                        stage,
                        index: reflected.metal_index,
                    })?;
                let mut sampler = crate::runtime::replacement_sampler_projection::decoded_sampler(
                    bound.reference,
                    reflected.binding,
                    identity,
                    &descriptor,
                )
                .map_err(ReplacementRenderDrawResolutionError::SamplerState)?;
                if let Some((lod_min, lod_max)) = bound.lod_clamp {
                    sampler.lod_min = lod_min;
                    sampler.lod_max = lod_max;
                }
                sampler
            } else {
                reims_vgpu_core::SamplerResource::null(reflected.binding)
            };
            sampler.binding = reflected.binding;
            let stages = render_stage_bits(stage);
            if let Some(existing) = resolved
                .iter_mut()
                .find(|existing| existing.binding == reflected.binding)
            {
                if existing.sampler != sampler {
                    return Err(
                        ReplacementRenderDrawResolutionError::SamplerBindingCollision(
                            reflected.binding,
                        ),
                    );
                }
                existing.stages = reims_vgpu_protocol::RenderStages::from_bits(
                    (existing.stages.bits() | stages.bits()).into(),
                )
                .expect("the union of render-stage bits is valid");
            } else {
                resolved.push(reims_vgpu_core::ResolvedRenderSamplerBinding {
                    binding: reflected.binding,
                    array_element: 0,
                    descriptor_count: 1,
                    stages,
                    sampler,
                });
            }
        }
    }
    Ok(resolved.into_boxed_slice())
}

fn sampled_texture_type_matches(
    expected: reims_vgpu_core::SampledImageKind,
    actual: reims_vgpu_protocol::TextureType,
) -> bool {
    matches!(
        (expected, actual),
        (
            reims_vgpu_core::SampledImageKind::D1,
            reims_vgpu_protocol::TextureType::D1
        ) | (
            reims_vgpu_core::SampledImageKind::D1Array,
            reims_vgpu_protocol::TextureType::D1Array
        ) | (
            reims_vgpu_core::SampledImageKind::D2,
            reims_vgpu_protocol::TextureType::D2
        ) | (
            reims_vgpu_core::SampledImageKind::D2Multisample,
            reims_vgpu_protocol::TextureType::D2Multisample
        ) | (
            reims_vgpu_core::SampledImageKind::D2Array,
            reims_vgpu_protocol::TextureType::D2Array
        ) | (
            reims_vgpu_core::SampledImageKind::D3,
            reims_vgpu_protocol::TextureType::D3
        ) | (
            reims_vgpu_core::SampledImageKind::Cube,
            reims_vgpu_protocol::TextureType::Cube
        ) | (
            reims_vgpu_core::SampledImageKind::CubeArray,
            reims_vgpu_protocol::TextureType::CubeArray
        )
    )
}

struct ReplacementResolvedRenderTextures {
    resources: Box<[reims_vgpu_core::ResolvedRenderResourceBinding]>,
    null_bindings: Box<[reims_vgpu_core::ResolvedRenderNullBinding]>,
}

fn resolve_render_textures<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRenderEncoderState,
    pipeline: &reims_vgpu_core::ResolvedRenderPipeline,
) -> Result<ReplacementResolvedRenderTextures, ReplacementRenderDrawResolutionError> {
    use reims_vgpu_core::{
        AccessMode, BackingRegion, ReflectedSampledKind, ReflectedTextureAccess,
        RenderBindingClass, RenderBindingView, ResolvedRenderNullBinding,
        ResolvedRenderResourceBinding, StorageImageAccess,
    };

    let mut resources: Vec<ResolvedRenderResourceBinding> = Vec::new();
    let mut nulls: Vec<ResolvedRenderNullBinding> = Vec::new();
    for (stage, family, table) in [
        (
            reims_vgpu_core::ShaderStage::Vertex,
            &pipeline.vertex,
            &state.vertex_textures,
        ),
        (
            reims_vgpu_core::ShaderStage::Fragment,
            &pipeline.fragment,
            &state.fragment_textures,
        ),
    ] {
        for index in 0..reims_vgpu_core::MAX_TEXTURE_BIND_SLOTS {
            let Some(descriptor) = family.interface.texture_descriptor(index) else {
                continue;
            };
            let expected = match family.interface.sampled_kind(descriptor.binding) {
                ReflectedSampledKind::Kind(kind) => kind,
                shape => {
                    return Err(ReplacementRenderDrawResolutionError::TextureShape {
                        stage,
                        index,
                        shape,
                    });
                }
            };
            let binding = family
                .variant()
                .texture_binding(index, Some(descriptor.binding));
            let (class, mode) = match descriptor.access {
                ReflectedTextureAccess::Sampled => {
                    (RenderBindingClass::SampledImage, AccessMode::Read)
                }
                ReflectedTextureAccess::Storage => {
                    let mode = match family.variant().storage_image_access(binding).ok_or(
                        ReplacementRenderDrawResolutionError::StorageImageAccessMissing(binding),
                    )? {
                        StorageImageAccess::ReadOnly => AccessMode::Read,
                        StorageImageAccess::WriteOnly => AccessMode::Write,
                        StorageImageAccess::ReadWrite => AccessMode::ReadWrite,
                        StorageImageAccess::Unknown => AccessMode::Unknown,
                        StorageImageAccess::AmbiguousBinding => {
                            return Err(
                                ReplacementRenderDrawResolutionError::StorageImageAccessAmbiguous(
                                    binding,
                                ),
                            );
                        }
                    };
                    (RenderBindingClass::StorageImage, mode)
                }
                ReflectedTextureAccess::Unknown => {
                    return Err(ReplacementRenderDrawResolutionError::TextureAccessUnknown {
                        stage,
                        index,
                    });
                }
            };
            let stages = render_stage_bits(stage);
            let location_matches = |candidate_class, candidate_binding, candidate_element| {
                candidate_class == class
                    && candidate_binding == binding
                    && candidate_element == descriptor.array_element
            };
            let reference = table[index as usize].unwrap_or(0);
            if reference == 0 {
                if resources.iter().any(|existing| {
                    location_matches(existing.class, existing.binding, existing.array_element)
                }) {
                    return Err(
                        ReplacementRenderDrawResolutionError::TextureBindingCollision {
                            class,
                            binding,
                            array_element: descriptor.array_element,
                        },
                    );
                }
                if let Some(existing) = nulls.iter_mut().find(|existing| {
                    location_matches(existing.class, existing.binding, existing.array_element)
                }) {
                    if existing.descriptor_count != descriptor.descriptor_count {
                        return Err(
                            ReplacementRenderDrawResolutionError::TextureBindingCollision {
                                class,
                                binding,
                                array_element: descriptor.array_element,
                            },
                        );
                    }
                    existing.stages = reims_vgpu_protocol::RenderStages::from_bits(
                        (existing.stages.bits() | stages.bits()).into(),
                    )
                    .expect("the union of render-stage bits is valid");
                } else {
                    nulls.push(ResolvedRenderNullBinding {
                        class,
                        binding,
                        array_element: descriptor.array_element,
                        descriptor_count: descriptor.descriptor_count,
                        stages,
                    });
                }
                continue;
            }
            if nulls.iter().any(|existing| {
                location_matches(existing.class, existing.binding, existing.array_element)
            }) {
                return Err(
                    ReplacementRenderDrawResolutionError::TextureBindingCollision {
                        class,
                        binding,
                        array_element: descriptor.array_element,
                    },
                );
            }
            let resolved = runtime
                .resolve_texture_binding(task, reims_vgpu_protocol::ObjectTableRef::new(reference))
                .map_err(|reason| ReplacementRenderDrawResolutionError::Texture {
                    stage,
                    index,
                    reason,
                })?;
            if !sampled_texture_type_matches(expected, resolved.view.texture_type) {
                return Err(ReplacementRenderDrawResolutionError::TextureShapeMismatch {
                    stage,
                    index,
                    expected,
                    actual: resolved.view.texture_type,
                });
            }
            let candidate = ResolvedRenderResourceBinding {
                class,
                binding,
                array_element: descriptor.array_element,
                descriptor_count: descriptor.descriptor_count,
                stages,
                resource: resolved.resource,
                backing: resolved.backing,
                view: RenderBindingView::Image(resolved.view),
                regions: Box::new([BackingRegion::Whole]),
                mode,
            };
            if let Some(existing) = resources.iter_mut().find(|existing| {
                location_matches(existing.class, existing.binding, existing.array_element)
            }) {
                if existing.descriptor_count != candidate.descriptor_count
                    || existing.resource != candidate.resource
                    || existing.backing != candidate.backing
                    || existing.view != candidate.view
                    || existing.regions != candidate.regions
                {
                    return Err(
                        ReplacementRenderDrawResolutionError::TextureBindingCollision {
                            class,
                            binding,
                            array_element: descriptor.array_element,
                        },
                    );
                }
                existing.stages = reims_vgpu_protocol::RenderStages::from_bits(
                    (existing.stages.bits() | stages.bits()).into(),
                )
                .expect("the union of render-stage bits is valid");
                existing.mode = match (existing.mode, mode) {
                    (AccessMode::Unknown, _) | (_, AccessMode::Unknown) => AccessMode::Unknown,
                    (AccessMode::ReadWrite, _) | (_, AccessMode::ReadWrite) => {
                        AccessMode::ReadWrite
                    }
                    (AccessMode::Read, AccessMode::Write)
                    | (AccessMode::Write, AccessMode::Read) => AccessMode::ReadWrite,
                    (left, _) => left,
                };
            } else {
                resources.push(candidate);
            }
        }
    }
    Ok(ReplacementResolvedRenderTextures {
        resources: resources.into_boxed_slice(),
        null_bindings: nulls.into_boxed_slice(),
    })
}

fn resolve_render_buffers<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRenderEncoderState,
    pipeline: &reims_vgpu_core::ResolvedRenderPipeline,
    command: &render::Command,
) -> Result<
    Box<[reims_vgpu_core::ResolvedRenderResourceBinding]>,
    ReplacementRenderDrawResolutionError,
> {
    use reims_vgpu_core::{
        AccessMode, BackingRegion, DescriptorUse, ReflectedBufferAccess, RenderBindingClass,
        RenderBindingView, ResolvedRenderResourceBinding, ShaderResourceKind,
    };

    let bounds = reims_vgpu_vulkan::spirv_bind::RenderBufferIndexBounds::new(
        command.vertex_start,
        command.vertex_count,
        command.base_instance,
        command.instance_count,
        command.index_buffer_ref != 0 || command.index_count != 0,
    );
    let mut resources: Vec<ResolvedRenderResourceBinding> = Vec::new();
    for (stage, family, table) in [
        (
            reims_vgpu_core::ShaderStage::Vertex,
            &pipeline.vertex,
            &state.vertex_buffers,
        ),
        (
            reims_vgpu_core::ShaderStage::Fragment,
            &pipeline.fragment,
            &state.fragment_buffers,
        ),
    ] {
        for reflected in family
            .interface
            .bindings
            .iter()
            .filter(|binding| binding.kind == ShaderResourceKind::Buffer)
        {
            let index = reflected.metal_index;
            let binding = family.variant().buffer_binding(index);
            if matches!(
                family.variant().buffer_use(index),
                DescriptorUse::NotDeclared | DescriptorUse::DeclaredUnused
            ) {
                continue;
            }
            let bound = table
                .get(index as usize)
                .copied()
                .flatten()
                .ok_or(ReplacementRenderDrawResolutionError::BufferMissing { stage, index })?;
            let mode = match family.interface.buffer_access(index) {
                ReflectedBufferAccess::ReadOnly => AccessMode::Read,
                ReflectedBufferAccess::Writable => AccessMode::ReadWrite,
                ReflectedBufferAccess::Unknown | ReflectedBufferAccess::Absent => {
                    AccessMode::Unknown
                }
                ReflectedBufferAccess::Unused => continue,
            };
            let available = runtime
                .resolve_buffer_tail(
                    task,
                    reims_vgpu_protocol::ObjectTableRef::new(bound.reference),
                    bound.offset,
                )
                .map_err(|reason| ReplacementRenderDrawResolutionError::Buffer {
                    stage,
                    index,
                    reason,
                })?;
            let required = reims_vgpu_vulkan::spirv_bind::reflected_render_buffer_extent_interface(
                &family.interface,
                index,
                bounds,
            );
            if let Some(required) = required {
                if required > available.length.get() {
                    return Err(
                        ReplacementRenderDrawResolutionError::BufferExtentPastBinding {
                            stage,
                            index,
                            required,
                            available: available.length.get(),
                        },
                    );
                }
            }
            let length = required.unwrap_or(available.length.get());
            let range = if length == available.length.get() {
                available
            } else {
                runtime
                    .resolve_buffer_range(
                        task,
                        reims_vgpu_protocol::ObjectTableRef::new(bound.reference),
                        bound.offset,
                        length,
                    )
                    .map_err(|reason| ReplacementRenderDrawResolutionError::Buffer {
                        stage,
                        index,
                        reason,
                    })?
            };
            let stages = render_stage_bits(stage);
            let candidate = ResolvedRenderResourceBinding {
                class: RenderBindingClass::StorageBuffer,
                binding,
                array_element: 0,
                descriptor_count: 1,
                stages,
                resource: range.resource,
                backing: range.storage,
                view: RenderBindingView::Buffer(range.region),
                regions: Box::new([BackingRegion::Linear(range.region)]),
                mode,
            };
            if let Some(existing) = resources.iter_mut().find(|existing| {
                existing.class == candidate.class && existing.binding == candidate.binding
            }) {
                if existing.resource != candidate.resource
                    || existing.backing != candidate.backing
                    || existing.view != candidate.view
                    || existing.regions != candidate.regions
                {
                    return Err(
                        ReplacementRenderDrawResolutionError::BufferBindingCollision(binding),
                    );
                }
                existing.stages = reims_vgpu_protocol::RenderStages::from_bits(
                    (existing.stages.bits() | stages.bits()).into(),
                )
                .expect("the union of render-stage bits is valid");
                existing.mode = match (existing.mode, mode) {
                    (AccessMode::Unknown, _) | (_, AccessMode::Unknown) => AccessMode::Unknown,
                    (AccessMode::ReadWrite, _) | (_, AccessMode::ReadWrite) => {
                        AccessMode::ReadWrite
                    }
                    (left, _) => left,
                };
            } else {
                resources.push(candidate);
            }
        }
    }
    Ok(resources.into_boxed_slice())
}

struct ReplacementResolvedVertexBuffers {
    resources: Box<[reims_vgpu_core::ResolvedRenderResourceBinding]>,
    layouts: Box<[reims_vgpu_core::ResolvedVertexBufferLayout]>,
}

fn resolve_vertex_buffers<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRenderEncoderState,
    pipeline: &reims_vgpu_core::ResolvedRenderPipeline,
) -> Result<ReplacementResolvedVertexBuffers, ReplacementRenderDrawResolutionError> {
    use reims_vgpu_core::{
        AccessMode, BackingRegion, RenderBindingClass, RenderBindingView,
        ResolvedRenderResourceBinding,
    };

    let mut resources: Vec<ResolvedRenderResourceBinding> = Vec::new();
    let mut layouts: Vec<reims_vgpu_core::ResolvedVertexBufferLayout> = Vec::new();
    for attribute in pipeline
        .desc
        .vertex_attributes
        .iter()
        .filter(|attribute| attribute.format != 0 && attribute.stride != 0)
    {
        reims_vgpu_protocol::decode_vertex_attribute_format(attribute.format).map_err(
            |reason| ReplacementRenderDrawResolutionError::VertexAttributeFormat {
                location: attribute.location,
                reason,
            },
        )?;
        let step =
            reims_vgpu_protocol::decode_vertex_step_function(attribute.declared_step_function)
                .map_err(
                    |reason| ReplacementRenderDrawResolutionError::VertexStepFunction {
                        location: attribute.location,
                        reason,
                    },
                )?;
        let step_rate = attribute.step_rate();
        if !reims_vgpu_protocol::step_rate_in_contract(step.mtl_ordinal(), step_rate) {
            return Err(ReplacementRenderDrawResolutionError::VertexStepRate {
                location: attribute.location,
                step_function: step.mtl_ordinal(),
                step_rate,
            });
        }
        let bound = state
            .vertex_buffers
            .get(attribute.buffer_index as usize)
            .copied()
            .flatten()
            .ok_or(ReplacementRenderDrawResolutionError::VertexBufferMissing(
                attribute.buffer_index,
            ))?;
        let stride = bound
            .attribute_stride
            .map(u32::try_from)
            .transpose()
            .map_err(
                |_| ReplacementRenderDrawResolutionError::DynamicVertexStrideOutOfRange {
                    index: attribute.buffer_index,
                    bound: bound
                        .attribute_stride
                        .expect("the conversion source exists"),
                },
            )?
            .unwrap_or(attribute.stride);
        if let Some(existing) = layouts
            .iter()
            .find(|layout| layout.binding == attribute.buffer_index)
        {
            if existing.stride != stride {
                return Err(
                    ReplacementRenderDrawResolutionError::VertexBindingCollision(
                        attribute.buffer_index,
                    ),
                );
            }
        } else {
            layouts.push(reims_vgpu_core::ResolvedVertexBufferLayout {
                binding: attribute.buffer_index,
                stride,
            });
        }
        let range = runtime
            .resolve_buffer_tail(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(bound.reference),
                bound.offset,
            )
            .map_err(
                |reason| ReplacementRenderDrawResolutionError::VertexBuffer {
                    index: attribute.buffer_index,
                    reason,
                },
            )?;
        let candidate = ResolvedRenderResourceBinding {
            class: RenderBindingClass::VertexBuffer,
            binding: attribute.buffer_index,
            array_element: 0,
            descriptor_count: 1,
            stages: render_stage_bits(reims_vgpu_core::ShaderStage::Vertex),
            resource: range.resource,
            backing: range.storage,
            view: RenderBindingView::Buffer(range.region),
            regions: Box::new([BackingRegion::Linear(range.region)]),
            mode: AccessMode::Read,
        };
        if let Some(existing) = resources
            .iter()
            .find(|existing| existing.binding == candidate.binding)
        {
            if existing.resource != candidate.resource
                || existing.backing != candidate.backing
                || existing.view != candidate.view
                || existing.regions != candidate.regions
            {
                return Err(
                    ReplacementRenderDrawResolutionError::VertexBindingCollision(
                        attribute.buffer_index,
                    ),
                );
            }
        } else {
            resources.push(candidate);
        }
    }
    layouts.sort_unstable();
    Ok(ReplacementResolvedVertexBuffers {
        resources: resources.into_boxed_slice(),
        layouts: layouts.into_boxed_slice(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeDispatchResolutionError {
    NotDirectDispatch,
    IndirectBuffer(crate::runtime::replacement_session::ReplacementBufferRangeResolutionError),
    IndirectThreadsRequireGpuWorkgroupConversion {
        buffer_ref: u32,
        offset: u64,
    },
    GridDimensionOutOfRange,
    GridInvalid,
    Pipeline(crate::runtime::replacement_session::ReplacementComputeTranslationAvailability),
    Texture(crate::runtime::replacement_session::ReplacementTextureResolutionError),
    StageInputUnsupported(Box<reims_vgpu_protocol::ComputeStageInputDescriptor>),
    ImageblockUnsupported(crate::runtime::replacement_compute_state::ImageblockDimensions),
    Construction(
        crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError,
    ),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementResolvedComputeControlPredicate {
    pub buffer: reims_vgpu_core::ResolvedBufferRange,
    pub comparison: u32,
    pub reference_value: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeControlFlowResolutionError {
    OutsideEncoder {
        kind: compute::Kind,
        opcode: u32,
    },
    PredicateBuffer {
        kind: compute::Kind,
        opcode: u32,
        buffer_ref: u32,
        reason: crate::runtime::replacement_session::ReplacementBufferRangeResolutionError,
    },
    Unsupported {
        kind: compute::Kind,
        opcode: u32,
        predicate: Option<ReplacementResolvedComputeControlPredicate>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeIcbPopulationError {
    OutsideEncoder,
    DecodedRangeMismatch,
    RenderCommandInComputeEncoder {
        index: u64,
    },
    Dispatch {
        index: u64,
        reason: ReplacementComputeDispatchResolutionError,
    },
    Population(reims_vgpu_core::IndirectCommandMutationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementIcbCommandMemoryTransportError {
    Unavailable,
    HostLengthOverflow(u64),
    Memory(crate::runtime::host::MemError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeIcbExecutionError {
    ReadPlan(crate::runtime::replacement_session::ReplacementIcbCommandMemoryReadError),
    Transport(ReplacementIcbCommandMemoryTransportError),
    Decode(crate::runtime::replacement_session::ReplacementIcbCommandBytesDecodeError),
    Population(ReplacementComputeIcbPopulationError),
    IndirectRangeRequiresAsynchronousReadback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderIcbDrawError {
    InheritedPipelineOverride,
    PipelineMissing,
    InheritedBuffersOverride,
    BufferIndexOverflow(u32),
    ObjectOrMeshBuffer,
    ObjectThreadgroupMemory,
    GeometryOutOfRange,
    UnsupportedDraw,
    Draw(ReplacementRenderDrawResolutionError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderIcbPopulationError {
    OutsideEncoder,
    DecodedRangeMismatch,
    ComputeCommandInRenderEncoder {
        index: u64,
    },
    Draw {
        index: u64,
        reason: ReplacementRenderIcbDrawError,
    },
    Population(reims_vgpu_core::IndirectCommandMutationError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRenderIcbExecutionError {
    ReadPlan(crate::runtime::replacement_session::ReplacementIcbCommandMemoryReadError),
    Transport(ReplacementIcbCommandMemoryTransportError),
    Decode(crate::runtime::replacement_session::ReplacementIcbCommandBytesDecodeError),
    Population(ReplacementRenderIcbPopulationError),
    IndirectRangeRequiresAsynchronousReadback,
}

#[derive(Debug)]
pub(crate) struct ReplacementComputeIcbPopulationFailure {
    pub reason: ReplacementComputeIcbPopulationError,
    pub slots: Vec<crate::runtime::icb::DecodedIcbCommandSlot>,
}

#[derive(Debug)]
pub(crate) struct ReplacementPopulatedComputeIcb {
    pub prior: reims_vgpu_core::PriorIndirectCommandPopulation<
        reims_vgpu_core::ResolvedIndirectCommandSlot<
            reims_vgpu_core::ResolvedRenderDispatch,
            reims_vgpu_core::ResolvedComputeDispatch,
        >,
    >,
    pub execution: reims_vgpu_core::ResolvedIndirectCommand,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementImageBlitEndpoint {
    Source,
    Destination,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementImageBlitResolutionError {
    NotImageCopy,
    Buffer {
        endpoint: ReplacementImageBlitEndpoint,
        reference: u32,
        reason: crate::runtime::replacement_session::ReplacementBufferRangeResolutionError,
    },
    Texture {
        endpoint: ReplacementImageBlitEndpoint,
        reference: u32,
        reason: crate::runtime::replacement_session::ReplacementTextureEndpointResolutionError,
    },
    Options(blit::BlitOptionError),
    AspectUnsupported {
        endpoint: ReplacementImageBlitEndpoint,
        pixel_format: u16,
    },
    TexelWidthMismatch {
        source: u32,
        destination: u32,
    },
    OverlappingImageCopy,
    LevelOverflow(ReplacementImageBlitEndpoint),
    SliceOverflow(ReplacementImageBlitEndpoint),
    VolumeSliceConstraint,
    SubresourceGeometryMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTextureFillValue {
    Color {
        raw: [u64; 4],
        pixel_format: u16,
    },
    Bytes {
        reference: u32,
        offset: u64,
        length: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementTextureFill {
    pub texture_ref: u32,
    pub level: u16,
    pub slice: u16,
    pub origin: blit::Point,
    pub size: blit::Size,
    pub value: ReplacementTextureFillValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTextureBlitResolutionError {
    Malformed {
        kind: blit::Kind,
        opcode: u32,
        fill_source: blit::FillSource,
    },
    GenerateMipmapsUnsupported {
        opcode: u32,
        texture_ref: u32,
    },
    SynchronizeResourceUnsupported {
        opcode: u32,
        resource_ref: u32,
    },
    SynchronizeTextureUnsupported {
        opcode: u32,
        texture_ref: u32,
        level: u16,
        slice: u16,
    },
    FillUnsupported {
        opcode: u32,
        fill: ReplacementTextureFill,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementOperationProjectionError {
    Event(ReplacementEventResolutionError),
    Fence(ReplacementFenceResolutionError),
    Participation(ReplacementParticipationResolutionError),
    Barrier(ReplacementBarrierResolutionError),
    BufferBlit(ReplacementBufferBlitResolutionError),
    ImageBlit(ReplacementImageBlitResolutionError),
    TextureBlit(Box<ReplacementTextureBlitResolutionError>),
    IndirectCommand(ReplacementIndirectCommandResolutionError),
    ComputeState(ReplacementComputeStateResolutionError),
    RenderState(ReplacementRenderStateResolutionError),
    RenderDraw(ReplacementRenderDrawResolutionError),
    RenderTessellation(Box<ReplacementRenderTessellationResolutionError>),
    RenderTile(Box<ReplacementRenderTileResolutionError>),
    ComputeDispatch(ReplacementComputeDispatchResolutionError),
    ComputeControlFlow(Box<ReplacementComputeControlFlowResolutionError>),
    ComputeIcb(ReplacementComputeIcbExecutionError),
    RenderIcb(ReplacementRenderIcbExecutionError),
    RenderUnresolved { kind: render::Kind, opcode: u32 },
    ComputeUnresolved { kind: compute::Kind, opcode: u32 },
    BlitUnresolved { kind: blit::Kind, opcode: u32 },
    Info(reims_vgpu_core::InfoResolutionError),
}

pub(crate) type ProjectedReplacementOperation<Completion> = reims_vgpu_core::ResolvedOperation<
    reims_vgpu_core::ResolvedRenderDispatch,
    reims_vgpu_core::ResolvedComputeDispatch,
    reims_vgpu_core::ResolvedInfoOperation,
    reims_vgpu_core::ResolvedIndirectCommand,
    Completion,
>;

pub(crate) type ProjectedReplacementExec<Completion> =
    Box<[ResolvedExecStream<ProjectedReplacementOperation<Completion>>]>;

pub(crate) fn mark_render_dispatch_encoder_boundaries<Completion>(
    streams: &mut [ResolvedExecStream<ProjectedReplacementOperation<Completion>>],
) {
    let mut open = None::<Vec<(usize, usize, usize)>>;
    let mut groups = Vec::new();
    for (stream_index, stream) in streams.iter().enumerate() {
        for (segment_index, segment) in stream.segments.iter().enumerate() {
            for (operation_index, operation) in segment.operations.iter().enumerate() {
                match operation {
                    reims_vgpu_core::ResolvedOperation::EncoderBoundary(
                        EncoderBoundary::Begin(reims_vgpu_protocol::SegmentKind::Render),
                    ) => open = Some(Vec::new()),
                    reims_vgpu_core::ResolvedOperation::Render(_) => {
                        if let Some(group) = open.as_mut() {
                            group.push((stream_index, segment_index, operation_index));
                        }
                    }
                    reims_vgpu_core::ResolvedOperation::EncoderBoundary(EncoderBoundary::End(
                        reims_vgpu_protocol::SegmentKind::Render,
                    )) => {
                        if let Some(group) = open.take() {
                            groups.push(group);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for group in groups {
        // A render encoder holds its attachments for its whole life, so an
        // attachment any draw in it reads as a texture is in a feedback loop
        // for every draw in it. Deriving that per draw instead would describe
        // one pass with as many access modes as it has draws.
        let mut sampled = std::collections::BTreeSet::new();
        // The same holds for a colour input read: the draw that declares it
        // stamped its own attachment, and the pass the whole encoder shares
        // must declare that layout for every draw in it.
        let mut input_read = std::collections::BTreeSet::new();
        for &(stream_index, segment_index, operation_index) in &group {
            let reims_vgpu_core::ResolvedOperation::Render(operation) =
                &streams[stream_index].segments[segment_index].operations[operation_index]
            else {
                unreachable!("the group contains only resolved render operations")
            };
            sampled.extend(
                operation
                    .resources
                    .iter()
                    .filter(|resource| {
                        resource.class == reims_vgpu_core::RenderBindingClass::SampledImage
                    })
                    .map(|resource| resource.backing),
            );
            input_read.extend(
                operation
                    .attachments
                    .iter()
                    .filter(|attachment| attachment.input_attachment)
                    .map(|attachment| attachment.backing),
            );
        }
        let last = group.len().checked_sub(1);
        for (index, (stream_index, segment_index, operation_index)) in group.into_iter().enumerate()
        {
            let reims_vgpu_core::ResolvedOperation::Render(operation) =
                &mut streams[stream_index].segments[segment_index].operations[operation_index]
            else {
                unreachable!("the group contains only resolved render operations")
            };
            operation.begins_encoder = index == 0;
            operation.ends_encoder = Some(index) == last;
            for attachment in operation.attachments.iter_mut() {
                attachment.feedback_loop = sampled.contains(&attachment.backing);
                attachment.input_attachment = input_read.contains(&attachment.backing);
            }
        }
    }
}

pub(crate) type ReplacementOperationProjectionFailure =
    ReplacementExecProjectionFailure<ReplacementOperationProjectionError>;

pub(crate) fn resolve_event_record<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &event::Command,
) -> Result<reims_vgpu_core::EventOperation, ReplacementEventResolutionError> {
    if command.has_timeout {
        return Err(ReplacementEventResolutionError::TimeoutUnsupported);
    }
    let kind = match command.kind {
        event::Kind::SignalEvent => reims_vgpu_core::EventOperationKind::Signal,
        event::Kind::WaitEvent => reims_vgpu_core::EventOperationKind::Wait,
        event::Kind::Unknown => return Err(ReplacementEventResolutionError::UnknownKind),
    };
    runtime
        .resolve_event_operation(
            task,
            reims_vgpu_protocol::SerializerRef::new(command.event_ref),
            kind,
            command.value,
        )
        .map_err(ReplacementEventResolutionError::Namespace)
}

pub(crate) fn resolve_compute_fence_record<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &compute::Command,
) -> Result<Option<reims_vgpu_core::FenceOperation>, ReplacementFenceResolutionError> {
    let kind = match command.kind {
        compute::Kind::UpdateFence => reims_vgpu_core::FenceOperationKind::Update,
        compute::Kind::WaitFence => reims_vgpu_core::FenceOperationKind::Wait,
        _ => return Err(ReplacementFenceResolutionError::NotFenceRecord),
    };
    resolve_fence(
        runtime,
        task,
        command.fence_ref,
        kind,
        reims_vgpu_core::FenceScope::Compute,
    )
}

pub(crate) fn resolve_blit_fence_record<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &blit::Command,
) -> Result<Option<reims_vgpu_core::FenceOperation>, ReplacementFenceResolutionError> {
    if command.kind != blit::Kind::Fence {
        return Err(ReplacementFenceResolutionError::NotFenceRecord);
    }
    let kind = match command.opcode {
        reims_vgpu_wire::ops::blit::OPCODE_UPDATE_FENCE => {
            reims_vgpu_core::FenceOperationKind::Update
        }
        reims_vgpu_wire::ops::blit::OPCODE_WAIT_FOR_FENCE => {
            reims_vgpu_core::FenceOperationKind::Wait
        }
        _ => return Err(ReplacementFenceResolutionError::UnknownFenceOpcode),
    };
    resolve_fence(
        runtime,
        task,
        command.fence,
        kind,
        reims_vgpu_core::FenceScope::Blit,
    )
}

pub(crate) fn resolve_render_fence_record<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &render::Command,
) -> Result<Option<reims_vgpu_core::FenceOperation>, ReplacementFenceResolutionError> {
    if command.kind != render::Kind::Fence {
        return Err(ReplacementFenceResolutionError::NotFenceRecord);
    }
    let kind = match command.opcode {
        reims_vgpu_wire::ops::render::OPCODE_UPDATE_FENCE => {
            reims_vgpu_core::FenceOperationKind::Update
        }
        reims_vgpu_wire::ops::render::OPCODE_WAIT_FOR_FENCE => {
            reims_vgpu_core::FenceOperationKind::Wait
        }
        _ => return Err(ReplacementFenceResolutionError::UnknownFenceOpcode),
    };
    if command.fence_ref == 0 {
        return Ok(None);
    }
    let stages = u16::try_from(command.fence_stages)
        .ok()
        .and_then(reims_vgpu_core::RenderBarrierStages::from_bits)
        .ok_or(ReplacementFenceResolutionError::UnsupportedRenderStages(
            command.fence_stages,
        ))?;
    resolve_fence(
        runtime,
        task,
        command.fence_ref,
        kind,
        reims_vgpu_core::FenceScope::Render(stages),
    )
}

fn resolve_fence<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    reference: u32,
    kind: reims_vgpu_core::FenceOperationKind,
    scope: reims_vgpu_core::FenceScope,
) -> Result<Option<reims_vgpu_core::FenceOperation>, ReplacementFenceResolutionError> {
    if reference == 0 {
        return Ok(None);
    }
    runtime
        .resolve_fence_operation(
            task,
            reims_vgpu_protocol::SerializerRef::new(reference),
            kind,
            scope,
        )
        .map(Some)
        .map_err(ReplacementFenceResolutionError::Namespace)
}

pub(crate) fn resolve_render_participation_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &render::Command,
) -> Result<Box<[reims_vgpu_core::ParticipationOperation]>, ReplacementParticipationResolutionError>
{
    let scope = reims_vgpu_core::ParticipationScope::Render {
        stages: command.participation_stages,
    };
    match command.kind {
        render::Kind::UseResource => resolve_participating_resources(
            runtime,
            task,
            &command.participation_resources,
            command
                .participation_usage
                .ok_or(ReplacementParticipationResolutionError::MissingResourceUsage)?,
            scope,
        ),
        render::Kind::UseHeap => {
            resolve_participating_heaps(runtime, task, &command.participation_heaps, scope)
        }
        _ => Err(ReplacementParticipationResolutionError::NotParticipationRecord),
    }
}

pub(crate) fn resolve_compute_participation_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &compute::Command,
) -> Result<Box<[reims_vgpu_core::ParticipationOperation]>, ReplacementParticipationResolutionError>
{
    match command.kind {
        compute::Kind::UseResources => resolve_participating_resources(
            runtime,
            task,
            &command.resources,
            command
                .resource_usage
                .ok_or(ReplacementParticipationResolutionError::MissingResourceUsage)?,
            reims_vgpu_core::ParticipationScope::Compute,
        ),
        compute::Kind::UseHeaps => resolve_participating_heaps(
            runtime,
            task,
            &command.heaps,
            reims_vgpu_core::ParticipationScope::Compute,
        ),
        _ => Err(ReplacementParticipationResolutionError::NotParticipationRecord),
    }
}

fn resolve_participating_resources<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    references: &[reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>],
    usage: reims_vgpu_protocol::ResourceUsage,
    scope: reims_vgpu_core::ParticipationScope,
) -> Result<Box<[reims_vgpu_core::ParticipationOperation]>, ReplacementParticipationResolutionError>
{
    references
        .iter()
        .copied()
        .enumerate()
        .map(|(index, reference)| {
            runtime
                .resolve_resource(task, reference)
                .map(
                    |resource| reims_vgpu_core::ParticipationOperation::Resource {
                        resource,
                        usage,
                        scope,
                    },
                )
                .ok_or(
                    ReplacementParticipationResolutionError::ResourceUnavailable {
                        index,
                        reference: reference.get(),
                    },
                )
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn resolve_participating_heaps<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    references: &[reims_vgpu_protocol::SerializerRef<reims_vgpu_protocol::HeapObject>],
    scope: reims_vgpu_core::ParticipationScope,
) -> Result<Box<[reims_vgpu_core::ParticipationOperation]>, ReplacementParticipationResolutionError>
{
    references
        .iter()
        .copied()
        .enumerate()
        .map(|(index, reference)| {
            runtime
                .resolve_heap(task, reference)
                .map(|heap| reims_vgpu_core::ParticipationOperation::Heap { heap, scope })
                .ok_or(ReplacementParticipationResolutionError::HeapUnavailable {
                    index,
                    reference: reference.get(),
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

pub(crate) fn resolve_compute_barrier_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &compute::Command,
) -> Result<Option<reims_vgpu_core::BarrierOperation>, ReplacementBarrierResolutionError> {
    match command.kind {
        compute::Kind::BarrierResources => resolve_barrier_resources(
            runtime,
            task,
            &command.resources,
            reims_vgpu_core::StageScope::Compute,
            reims_vgpu_core::StageScope::Compute,
        ),
        compute::Kind::BarrierScope => {
            let scope = reims_vgpu_core::MemoryBarrierScope::from_bits(command.barrier_scope)
                .ok_or(ReplacementBarrierResolutionError::UnsupportedScope(
                    command.barrier_scope,
                ))?;
            Ok(
                (!scope.is_empty()).then_some(reims_vgpu_core::BarrierOperation::Scope {
                    scope,
                    before: reims_vgpu_core::StageScope::Compute,
                    after: reims_vgpu_core::StageScope::Compute,
                }),
            )
        }
        _ => Err(ReplacementBarrierResolutionError::NotBarrierRecord),
    }
}

pub(crate) fn resolve_render_barrier_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &render::Command,
) -> Result<Option<reims_vgpu_core::BarrierOperation>, ReplacementBarrierResolutionError> {
    if command.kind == render::Kind::TextureBarrier {
        return Err(ReplacementBarrierResolutionError::TextureBarrierContractUnresolved);
    }
    if !matches!(
        command.kind,
        render::Kind::BarrierResources | render::Kind::BarrierScope
    ) {
        return Err(ReplacementBarrierResolutionError::NotBarrierRecord);
    }
    let producer = reims_vgpu_protocol::RenderStages::from_bits(command.barrier_after_stages)
        .map_err(|_| {
            ReplacementBarrierResolutionError::UnsupportedProducerStages(
                command.barrier_after_stages,
            )
        })?;
    let consumer = reims_vgpu_protocol::RenderStages::from_bits(command.barrier_before_stages)
        .map_err(|_| {
            ReplacementBarrierResolutionError::UnsupportedConsumerStages(
                command.barrier_before_stages,
            )
        })?;
    if producer.bits() == 0 || consumer.bits() == 0 {
        return Ok(None);
    }
    let before = reims_vgpu_core::StageScope::Render(producer);
    let after = reims_vgpu_core::StageScope::Render(consumer);
    match command.kind {
        render::Kind::BarrierResources => {
            resolve_barrier_resources(runtime, task, &command.barrier_resources, before, after)
        }
        render::Kind::BarrierScope => {
            if command.barrier_unidentified_u8 != 0 {
                return Err(ReplacementBarrierResolutionError::UnidentifiedRenderField(
                    command.barrier_unidentified_u8,
                ));
            }
            let scope =
                reims_vgpu_core::MemoryBarrierScope::from_bits(u16::from(command.barrier_scope))
                    .ok_or(ReplacementBarrierResolutionError::UnsupportedScope(
                        u16::from(command.barrier_scope),
                    ))?;
            Ok(
                (!scope.is_empty()).then_some(reims_vgpu_core::BarrierOperation::Scope {
                    scope,
                    before,
                    after,
                }),
            )
        }
        _ => unreachable!("render barrier family was checked above"),
    }
}

fn resolve_barrier_resources<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    references: &[reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>],
    before: reims_vgpu_core::StageScope,
    after: reims_vgpu_core::StageScope,
) -> Result<Option<reims_vgpu_core::BarrierOperation>, ReplacementBarrierResolutionError> {
    let resources = references
        .iter()
        .copied()
        .enumerate()
        .map(|(index, reference)| {
            runtime.resolve_resource(task, reference).ok_or(
                ReplacementBarrierResolutionError::ResourceUnavailable {
                    index,
                    reference: reference.get(),
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(
        (!resources.is_empty()).then_some(reims_vgpu_core::BarrierOperation::Resources {
            resources: resources.into_boxed_slice(),
            before,
            after,
        }),
    )
}

pub(crate) fn resolve_buffer_blit_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &blit::Command,
) -> Result<Option<reims_vgpu_core::ResolvedBlit>, ReplacementBufferBlitResolutionError> {
    let range = |endpoint, reference, offset, length| {
        runtime
            .resolve_buffer_range(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(reference),
                offset,
                length,
            )
            .map_err(|reason| ReplacementBufferBlitResolutionError::Range {
                endpoint,
                reference,
                reason,
            })
    };
    match command.kind {
        blit::Kind::FillBuffer | blit::Kind::FillBufferPattern4 => {
            if command.range_length == 0 {
                return Ok(None);
            }
            let pattern = if command.kind == blit::Kind::FillBuffer {
                reims_vgpu_core::BufferFillPattern::Byte(command.fill_value)
            } else {
                if !command.range_location.is_multiple_of(4) {
                    return Err(ReplacementBufferBlitResolutionError::UnalignedWordPattern(
                        command.range_location,
                    ));
                }
                reims_vgpu_core::BufferFillPattern::Word(command.fill_pattern.to_le_bytes())
            };
            Ok(Some(reims_vgpu_core::ResolvedBlit::Fill {
                destination: range(
                    ReplacementBufferBlitEndpoint::Destination,
                    command.buffer,
                    command.range_location,
                    command.range_length,
                )?,
                pattern,
            }))
        }
        blit::Kind::Copy
            if command.copy_kind == blit::CopyKind::BufferToBuffer
                && command.source_kind == blit::RefKind::Buffer
                && command.destination_kind == blit::RefKind::Buffer =>
        {
            if command.size == 0 {
                return Ok(None);
            }
            let source = range(
                ReplacementBufferBlitEndpoint::Source,
                command.source,
                command.source_offset,
                command.size,
            )?;
            let destination = range(
                ReplacementBufferBlitEndpoint::Destination,
                command.destination,
                command.destination_offset,
                command.size,
            )?;
            if source.storage == destination.storage && source.region.overlaps(destination.region) {
                return Err(ReplacementBufferBlitResolutionError::OverlappingCopy);
            }
            Ok(Some(reims_vgpu_core::ResolvedBlit::Copy {
                source,
                destination,
            }))
        }
        _ => Err(ReplacementBufferBlitResolutionError::NotBufferBlit),
    }
}

pub(crate) fn resolve_indirect_command_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &blit::Command,
) -> Result<reims_vgpu_core::ResolvedIndirectCommand, ReplacementIndirectCommandResolutionError> {
    let range = reims_vgpu_protocol::IndirectCommandRange {
        location: command.range_location,
        length: command.range_length,
    };
    let operation = match command.kind {
        blit::Kind::IcbRange => match command.opcode {
            reims_vgpu_wire::ops::blit::OPCODE_OPTIMIZE_ICB => {
                reims_vgpu_protocol::IndirectCommandOperation::Optimize {
                    icb: reims_vgpu_protocol::SerializerRef::new(command.resource),
                    range,
                }
            }
            reims_vgpu_wire::ops::blit::OPCODE_RESET_ICB => {
                reims_vgpu_protocol::IndirectCommandOperation::Reset {
                    icb: reims_vgpu_protocol::SerializerRef::new(command.resource),
                    range,
                }
            }
            opcode => {
                return Err(ReplacementIndirectCommandResolutionError::UnknownRangeOpcode(opcode));
            }
        },
        blit::Kind::IcbCopy => reims_vgpu_protocol::IndirectCommandOperation::Copy {
            source: reims_vgpu_protocol::SerializerRef::new(command.source),
            source_range: range,
            destination: reims_vgpu_protocol::SerializerRef::new(command.destination),
            destination_index: command.destination_index,
        },
        _ => return Err(ReplacementIndirectCommandResolutionError::NotIndirectCommand),
    };
    reims_vgpu_core::resolve_indirect_command(task, operation, runtime)
        .map_err(ReplacementIndirectCommandResolutionError::Resolve)
}

pub(crate) fn resolve_image_blit_record<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &blit::Command,
) -> Result<Option<reims_vgpu_core::ResolvedBlit>, ReplacementImageBlitResolutionError> {
    use ReplacementImageBlitEndpoint::{Destination, Source};
    use ReplacementImageBlitResolutionError as Error;

    if command.kind != blit::Kind::Copy
        || matches!(
            command.copy_kind,
            blit::CopyKind::None | blit::CopyKind::BufferToBuffer
        )
    {
        return Err(Error::NotImageCopy);
    }
    if command.copy_kind == blit::CopyKind::TextureToTextureSliceLevel {
        return resolve_texture_copy_batch(runtime, task, command);
    }
    let extent = reims_vgpu_core::TextureExtent {
        width: command.source_size.width,
        height: command.source_size.height,
        depth: command.source_size.depth,
    };
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Ok(None);
    }
    let aspect =
        blit::parse_blit_options(command.has_options, command.options).map_err(Error::Options)?;
    let texture = |endpoint, reference, level, slice| {
        runtime
            .resolve_linear_texture_endpoint(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(reference),
                level,
                slice,
            )
            .map_err(|reason| Error::Texture {
                endpoint,
                reference,
                reason,
            })
    };
    let buffer = |endpoint, reference, offset| {
        runtime
            .resolve_buffer_tail(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(reference),
                offset,
            )
            .map_err(|reason| Error::Buffer {
                endpoint,
                reference,
                reason,
            })
    };
    let origin = |point: blit::Point| reims_vgpu_core::TextureOrigin {
        x: point.x,
        y: point.y,
        z: point.z,
    };
    let aspect_width = |endpoint, texture: &reims_vgpu_core::ResolvedTextureEndpoint| {
        reims_vgpu_core::pixel_format::blit_aspect_bytes_per_pixel(
            texture.backing.pixel_format(),
            aspect,
        )
        .ok_or(Error::AspectUnsupported {
            endpoint,
            pixel_format: texture.backing.pixel_format(),
        })
    };

    Ok(Some(match command.copy_kind {
        blit::CopyKind::BufferToTexture => {
            let destination = texture(
                Destination,
                command.destination,
                command.destination_level,
                command.destination_slice,
            )?;
            aspect_width(Destination, &destination)?;
            reims_vgpu_core::ResolvedBlit::BufferToTexture(
                reims_vgpu_core::ResolvedBufferToTextureBlit {
                    source: buffer(Source, command.source, command.source_offset)?,
                    source_bytes_per_row: command.source_bytes_per_row,
                    source_bytes_per_image: command.source_bytes_per_image,
                    destination,
                    destination_origin: origin(command.destination_origin),
                    extent,
                    aspect,
                },
            )
        }
        blit::CopyKind::TextureToBuffer => {
            let source = texture(
                Source,
                command.source,
                command.source_level,
                command.source_slice,
            )?;
            aspect_width(Source, &source)?;
            reims_vgpu_core::ResolvedBlit::TextureToBuffer(
                reims_vgpu_core::ResolvedTextureToBufferBlit {
                    source,
                    source_origin: origin(command.source_origin),
                    extent,
                    destination: buffer(
                        Destination,
                        command.destination,
                        command.destination_offset,
                    )?,
                    destination_bytes_per_row: command.destination_bytes_per_row,
                    destination_bytes_per_image: command.destination_bytes_per_image,
                    aspect,
                },
            )
        }
        blit::CopyKind::TextureToTexture => {
            let source = texture(
                Source,
                command.source,
                command.source_level,
                command.source_slice,
            )?;
            let destination = texture(
                Destination,
                command.destination,
                command.destination_level,
                command.destination_slice,
            )?;
            let source_width = aspect_width(Source, &source)?;
            let destination_width = aspect_width(Destination, &destination)?;
            if source_width != destination_width {
                return Err(Error::TexelWidthMismatch {
                    source: source_width,
                    destination: destination_width,
                });
            }
            if source.storage == destination.storage
                && source.level == destination.level
                && source.slice == destination.slice
                && texture_boxes_overlap(
                    origin(command.source_origin),
                    origin(command.destination_origin),
                    extent,
                )
            {
                return Err(Error::OverlappingImageCopy);
            }
            reims_vgpu_core::ResolvedBlit::TextureToTexture(
                reims_vgpu_core::ResolvedTextureToTextureBlit {
                    source,
                    source_origin: origin(command.source_origin),
                    destination,
                    destination_origin: origin(command.destination_origin),
                    extent,
                    aspect,
                },
            )
        }
        blit::CopyKind::None
        | blit::CopyKind::BufferToBuffer
        | blit::CopyKind::TextureToTextureSliceLevel => unreachable!(),
    }))
}

fn resolve_texture_blit(command: &blit::Command) -> ReplacementTextureBlitResolutionError {
    use ReplacementTextureBlitResolutionError as Error;

    match (command.kind, command.opcode) {
        (blit::Kind::Resource, reims_vgpu_wire::ops::blit::OPCODE_GENERATE_MIPMAPS) => {
            Error::GenerateMipmapsUnsupported {
                opcode: command.opcode,
                texture_ref: command.resource,
            }
        }
        (blit::Kind::Resource, reims_vgpu_wire::ops::blit::OPCODE_SYNCHRONIZE_RESOURCE) => {
            Error::SynchronizeResourceUnsupported {
                opcode: command.opcode,
                resource_ref: command.resource,
            }
        }
        (blit::Kind::Image, reims_vgpu_wire::ops::blit::OPCODE_SYNCHRONIZE_TEXTURE) => {
            Error::SynchronizeTextureUnsupported {
                opcode: command.opcode,
                texture_ref: command.texture,
                level: command.level,
                slice: command.slice,
            }
        }
        (blit::Kind::FillTexture, reims_vgpu_wire::ops::blit::OPCODE_FILL_TEXTURE_COLOR)
            if command.fill_source == blit::FillSource::Color =>
        {
            Error::FillUnsupported {
                opcode: command.opcode,
                fill: ReplacementTextureFill {
                    texture_ref: command.texture,
                    level: command.level,
                    slice: command.slice,
                    origin: command.fill_origin,
                    size: command.fill_size,
                    value: ReplacementTextureFillValue::Color {
                        raw: command.fill_color_raw,
                        pixel_format: command.fill_pixel_format,
                    },
                },
            }
        }
        (blit::Kind::FillTexture, reims_vgpu_wire::ops::blit::OPCODE_FILL_TEXTURE_BYTES)
            if command.fill_source == blit::FillSource::Bytes =>
        {
            Error::FillUnsupported {
                opcode: command.opcode,
                fill: ReplacementTextureFill {
                    texture_ref: command.texture,
                    level: command.level,
                    slice: command.slice,
                    origin: command.fill_origin,
                    size: command.fill_size,
                    value: ReplacementTextureFillValue::Bytes {
                        reference: command.fill_bytes_ref,
                        offset: command.fill_bytes_offset,
                        length: command.fill_bytes_length,
                    },
                },
            }
        }
        _ => Error::Malformed {
            kind: command.kind,
            opcode: command.opcode,
            fill_source: command.fill_source,
        },
    }
}

fn resolve_texture_copy_batch<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    command: &blit::Command,
) -> Result<Option<reims_vgpu_core::ResolvedBlit>, ReplacementImageBlitResolutionError> {
    use ReplacementImageBlitEndpoint::{Destination, Source};
    use ReplacementImageBlitResolutionError as Error;

    if command.slice_count == 0 || command.level_count == 0 {
        return Ok(None);
    }
    let texture = |endpoint, reference, level, slice| {
        runtime
            .resolve_linear_texture_endpoint(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(reference),
                level,
                slice,
            )
            .map_err(|reason| Error::Texture {
                endpoint,
                reference,
                reason,
            })
    };
    let mut levels = Vec::with_capacity(usize::from(command.level_count));
    for level_delta in 0..command.level_count {
        let source_level = command
            .source_level
            .checked_add(level_delta)
            .ok_or(Error::LevelOverflow(Source))?;
        let destination_level = command
            .destination_level
            .checked_add(level_delta)
            .ok_or(Error::LevelOverflow(Destination))?;
        let mut slices = Vec::with_capacity(usize::from(command.slice_count));
        for slice_delta in 0..command.slice_count {
            let source_slice = command
                .source_slice
                .checked_add(slice_delta)
                .ok_or(Error::SliceOverflow(Source))?;
            let destination_slice = command
                .destination_slice
                .checked_add(slice_delta)
                .ok_or(Error::SliceOverflow(Destination))?;
            let source = texture(Source, command.source, source_level, source_slice)?;
            let destination = texture(
                Destination,
                command.destination,
                destination_level,
                destination_slice,
            )?;
            if (source.backing.depth() > 1 || destination.backing.depth() > 1)
                && (command.slice_count != 1
                    || command.source_slice != 0
                    || command.destination_slice != 0)
            {
                return Err(Error::VolumeSliceConstraint);
            }
            if source.backing.width() != destination.backing.width()
                || source.backing.height() != destination.backing.height()
                || source.backing.depth() != destination.backing.depth()
                || source.backing.bpp() != destination.backing.bpp()
            {
                return Err(Error::SubresourceGeometryMismatch);
            }
            slices.push((source, destination));
        }
        let mut slices = slices.into_iter();
        let first_slice = slices
            .next()
            .expect("the nonzero slice count was validated");
        levels.push(reims_vgpu_core::ResolvedTextureLevelCopy {
            first_slice,
            remaining_slices: slices.collect::<Vec<_>>().into_boxed_slice(),
        });
    }
    let mut levels = levels.into_iter();
    let first_level = levels
        .next()
        .expect("the nonzero level count was validated");
    Ok(Some(reims_vgpu_core::ResolvedBlit::TextureCopyBatch(
        reims_vgpu_core::ResolvedTextureCopyBatch {
            source_base_slice: command.source_slice,
            destination_base_slice: command.destination_slice,
            first_level,
            remaining_levels: levels.collect::<Vec<_>>().into_boxed_slice(),
        },
    )))
}

fn texture_boxes_overlap(
    source: reims_vgpu_core::TextureOrigin,
    destination: reims_vgpu_core::TextureOrigin,
    extent: reims_vgpu_core::TextureExtent,
) -> bool {
    [
        (source.x, destination.x, extent.width),
        (source.y, destination.y, extent.height),
        (source.z, destination.z, extent.depth),
    ]
    .into_iter()
    .all(|(source, destination, length)| {
        source
            .checked_add(length)
            .zip(destination.checked_add(length))
            .is_some_and(|(source_end, destination_end)| {
                source < destination_end && destination < source_end
            })
    })
}

fn replacement_render_indices(
    first: u32,
    count: usize,
    limit: u32,
    overflow: impl Fn(u32, u32) -> ReplacementRenderStateResolutionError,
) -> Result<std::ops::Range<usize>, ReplacementRenderStateResolutionError> {
    let count = u32::try_from(count).map_err(|_| overflow(first, limit))?;
    let end = first
        .checked_add(count)
        .ok_or_else(|| overflow(first, limit))?;
    if end > limit {
        return Err(overflow(end - 1, limit));
    }
    Ok(first as usize..end as usize)
}

fn resolve_render_state_record(
    state: &mut ReplacementRecordProjectionState,
    command: &render::Command,
) -> Result<(), ReplacementRenderStateResolutionError> {
    let render = state
        .render
        .as_mut()
        .ok_or(ReplacementRenderStateResolutionError::OutsideEncoder)?;
    match command.kind {
        render::Kind::SetPipeline => render.pipeline_ref = command.pipeline_ref,
        render::Kind::SetBuffer => {
            let indices = replacement_render_indices(
                command.first,
                command.buffer_binds.len(),
                reims_vgpu_core::MAX_BUFFER_BIND_SLOTS,
                |index, limit| ReplacementRenderStateResolutionError::BufferIndexOverflow {
                    index,
                    limit,
                },
            )?;
            let table = match command.stage {
                render::Stage::Vertex => &mut render.vertex_buffers,
                render::Stage::Fragment => &mut render.fragment_buffers,
                render::Stage::Unknown => {
                    return Err(ReplacementRenderStateResolutionError::UnknownStage)
                }
            };
            for (index, binding) in indices.zip(&command.buffer_binds) {
                table[index] =
                    (binding.buffer_ref != 0).then_some(ReplacementRenderBufferBinding {
                        reference: binding.buffer_ref,
                        offset: binding.offset,
                        attribute_stride: binding.attribute_stride,
                    });
            }
        }
        render::Kind::SetBufferOffset => {
            if command.first >= reims_vgpu_core::MAX_BUFFER_BIND_SLOTS {
                return Err(ReplacementRenderStateResolutionError::BufferIndexOverflow {
                    index: command.first,
                    limit: reims_vgpu_core::MAX_BUFFER_BIND_SLOTS,
                });
            }
            let table = match command.stage {
                render::Stage::Vertex => &mut render.vertex_buffers,
                render::Stage::Fragment => &mut render.fragment_buffers,
                render::Stage::Unknown => {
                    return Err(ReplacementRenderStateResolutionError::UnknownStage)
                }
            };
            let binding = table[command.first as usize].as_mut().ok_or(
                ReplacementRenderStateResolutionError::BufferOffsetUnbound {
                    index: command.first,
                },
            )?;
            binding.offset = command.buffer_offset;
            if let Some(stride) = command.attribute_stride {
                binding.attribute_stride = Some(stride);
            }
        }
        render::Kind::SetTexture => {
            let indices = replacement_render_indices(
                command.first,
                command.ref_binds.len(),
                reims_vgpu_core::MAX_TEXTURE_BIND_SLOTS,
                |index, limit| ReplacementRenderStateResolutionError::TextureIndexOverflow {
                    index,
                    limit,
                },
            )?;
            let table = match command.stage {
                render::Stage::Vertex => &mut render.vertex_textures,
                render::Stage::Fragment => &mut render.fragment_textures,
                render::Stage::Unknown => {
                    return Err(ReplacementRenderStateResolutionError::UnknownStage)
                }
            };
            for (index, reference) in indices.zip(&command.ref_binds) {
                table[index] = (*reference != 0).then_some(*reference);
            }
        }
        render::Kind::SetSampler => {
            let indices = replacement_render_indices(
                command.first,
                command.ref_binds.len(),
                reims_vgpu_core::MAX_SAMPLER_BIND_SLOTS,
                |index, limit| ReplacementRenderStateResolutionError::SamplerIndexOverflow {
                    index,
                    limit,
                },
            )?;
            let table = match command.stage {
                render::Stage::Vertex => &mut render.vertex_samplers,
                render::Stage::Fragment => &mut render.fragment_samplers,
                render::Stage::Unknown => {
                    return Err(ReplacementRenderStateResolutionError::UnknownStage)
                }
            };
            for (offset, (index, reference)) in indices.zip(&command.ref_binds).enumerate() {
                table[index] = (*reference != 0).then_some(ReplacementRenderSamplerBinding {
                    reference: *reference,
                    lod_clamp: command.sampler_lod_binds.get(offset).copied(),
                });
            }
        }
        render::Kind::SetViewport => {
            render.viewports_bits = command
                .viewports
                .iter()
                .map(|viewport| viewport.map(f64::to_bits))
                .collect();
        }
        render::Kind::SetScissor => {
            if let Some((index, _)) = command
                .scissors
                .iter()
                .enumerate()
                .find(|(_, scissor)| scissor.is_empty())
            {
                return Err(ReplacementRenderStateResolutionError::EmptyScissor { index });
            }
            render.scissors.clone_from(&command.scissors);
        }
        render::Kind::SetBlendColor => {
            render.blend_color_bits = Some(command.blend_color.map(f32::to_bits));
        }
        render::Kind::SetCullMode => {
            let raw = u32::try_from(command.cull_mode).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidRasterState {
                    opcode: command.opcode,
                    raw: command.cull_mode,
                }
            })?;
            render.cull_mode = reims_vgpu_protocol::cull_mode(raw).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidRasterState {
                    opcode: command.opcode,
                    raw: command.cull_mode,
                }
            })?;
        }
        render::Kind::SetFrontFacing => {
            let raw = u32::try_from(command.front_facing).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidRasterState {
                    opcode: command.opcode,
                    raw: command.front_facing,
                }
            })?;
            render.front_face_ccw = reims_vgpu_protocol::front_face_ccw(raw).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidRasterState {
                    opcode: command.opcode,
                    raw: command.front_facing,
                }
            })?;
        }
        render::Kind::SetDepthBias => {
            render.depth_bias_bits = Some(command.depth_bias.map(f32::to_bits));
        }
        render::Kind::SetDepthStencil => render.depth_stencil_ref = command.depth_stencil_ref,
        render::Kind::SetStencilReference => {
            render.stencil_reference = Some((command.stencil_ref_front, command.stencil_ref_back));
        }
        render::Kind::SetTessellationFactorBuffer => {
            render.tessellation_factor_buffer =
                (command.buffer_ref != 0).then_some(ReplacementTessellationFactorBufferBinding {
                    reference: command.buffer_ref,
                    offset: command.buffer_offset,
                    instance_stride: command.tessellation_factor_instance_stride,
                });
        }
        render::Kind::SetRasterState => {
            let raw = u32::try_from(command.mode).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidRasterState {
                    opcode: command.opcode,
                    raw: command.mode,
                }
            })?;
            match command.opcode {
                reims_vgpu_wire::ops::render::OPCODE_SET_TRIANGLE_FILL_MODE => {
                    render.fill_mode = reims_vgpu_protocol::fill_mode(raw).map_err(|_| {
                        ReplacementRenderStateResolutionError::InvalidRasterState {
                            opcode: command.opcode,
                            raw: command.mode,
                        }
                    })?;
                }
                reims_vgpu_wire::ops::render::OPCODE_SET_DEPTH_CLIP_MODE => {
                    render.depth_clip_mode =
                        reims_vgpu_protocol::depth_clip_mode(raw).map_err(|_| {
                            ReplacementRenderStateResolutionError::InvalidRasterState {
                                opcode: command.opcode,
                                raw: command.mode,
                            }
                        })?;
                }
                _ => {
                    return Err(ReplacementRenderStateResolutionError::InvalidRasterState {
                        opcode: command.opcode,
                        raw: command.mode,
                    })
                }
            }
        }
        render::Kind::SetFloatState
            if command.opcode == reims_vgpu_wire::ops::render::OPCODE_SET_LINE_WIDTH =>
        {
            render.line_width = reims_vgpu_core::LineWidth::from_f32(command.float_value);
        }
        render::Kind::SetFloatState
            if command.opcode
                == reims_vgpu_wire::ops::render::OPCODE_SET_TESSELLATION_FACTOR_SCALE =>
        {
            if command.float_value.to_bits() != 1.0f32.to_bits() {
                return Err(
                    ReplacementRenderStateResolutionError::TessellationFactorScaleUnsupported {
                        bits: command.float_value.to_bits(),
                    },
                );
            }
        }
        render::Kind::SetVertexAmplification => match command.opcode {
            reims_vgpu_wire::ops::render::OPCODE_SET_VERTEX_AMPLIFICATION_MODE => {
                if command.mode != 0 || command.amplification_value != 0 {
                    return Err(
                        ReplacementRenderStateResolutionError::VertexAmplificationUnsupported {
                            count: 1,
                            mode: command.mode,
                            value: command.amplification_value,
                        },
                    );
                }
            }
            reims_vgpu_wire::ops::render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT => {
                if command.count != 1 || command.amplification_mappings.len() != 1 {
                    return Err(
                        ReplacementRenderStateResolutionError::VertexAmplificationUnsupported {
                            count: command.count,
                            mode: 0,
                            value: 0,
                        },
                    );
                }
                let [viewport_offset, render_target_offset] = command.amplification_mappings[0];
                if viewport_offset != 0 || render_target_offset != 0 {
                    return Err(ReplacementRenderStateResolutionError::
                        VertexAmplificationMappingUnsupported {
                            index: 0,
                            viewport_offset,
                            render_target_offset,
                        });
                }
            }
            _ => return Err(ReplacementRenderStateResolutionError::NotStateRecord),
        },
        render::Kind::RenderPassProperty
            if command.opcode
                == reims_vgpu_wire::ops::render_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT =>
        {
            if command.mode != 1 {
                return Err(
                    ReplacementRenderStateResolutionError::DefaultRasterSampleCountUnsupported {
                        count: command.mode,
                    },
                );
            }
        }
        render::Kind::RenderPassProperty => {
            return Err(
                ReplacementRenderStateResolutionError::RenderPassPropertyUnsupported {
                    opcode: command.opcode,
                    value: command.mode,
                    reference: command.texture_ref,
                    count: command.count,
                },
            );
        }
        render::Kind::SetStoreAction => {
            let raw = u16::try_from(command.mode).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidStoreAction { raw: command.mode }
            })?;
            reims_vgpu_protocol::pass_action::store_action(raw).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidStoreAction { raw: command.mode }
            })?;
            match command.opcode {
                reims_vgpu_wire::ops::render::OPCODE_SET_COLOR_STORE_ACTION => {
                    let attachment = render
                        .color_attachments
                        .get_mut(command.first as usize)
                        .ok_or(
                            ReplacementRenderStateResolutionError::StoreActionColorSlotOutOfRange {
                                index: command.first,
                            },
                        )?;
                    if attachment.texture_ref == 0 {
                        return Err(
                            ReplacementRenderStateResolutionError::StoreActionAttachmentAbsent,
                        );
                    }
                    attachment.store_action = raw;
                }
                reims_vgpu_wire::ops::render::OPCODE_SET_DEPTH_STORE_ACTION => {
                    if render.depth_attachment.texture_ref == 0 {
                        return Err(
                            ReplacementRenderStateResolutionError::StoreActionAttachmentAbsent,
                        );
                    }
                    render.depth_attachment.store_action = raw;
                }
                reims_vgpu_wire::ops::render::OPCODE_SET_STENCIL_STORE_ACTION => {
                    if render.stencil_attachment.texture_ref == 0 {
                        return Err(
                            ReplacementRenderStateResolutionError::StoreActionAttachmentAbsent,
                        );
                    }
                    render.stencil_attachment.store_action = raw;
                }
                _ => {
                    return Err(ReplacementRenderStateResolutionError::InvalidStoreAction {
                        raw: command.mode,
                    })
                }
            }
        }
        render::Kind::SetStoreActionOptions => {
            let raw = u16::try_from(command.mode).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidStoreActionOptions {
                    raw: command.mode,
                }
            })?;
            match command.opcode {
                reims_vgpu_wire::ops::render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS => {
                    let attachment = render
                        .color_attachments
                        .get_mut(command.first as usize)
                        .ok_or(
                            ReplacementRenderStateResolutionError::StoreActionColorSlotOutOfRange {
                                index: command.first,
                            },
                        )?;
                    if attachment.texture_ref == 0 {
                        return Err(
                            ReplacementRenderStateResolutionError::StoreActionAttachmentAbsent,
                        );
                    }
                    attachment.store_action_options = raw;
                }
                reims_vgpu_wire::ops::render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS => {
                    if render.depth_attachment.texture_ref == 0 {
                        return Err(
                            ReplacementRenderStateResolutionError::StoreActionAttachmentAbsent,
                        );
                    }
                    render.depth_attachment.store_action_options = raw;
                }
                reims_vgpu_wire::ops::render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => {
                    if render.stencil_attachment.texture_ref == 0 {
                        return Err(
                            ReplacementRenderStateResolutionError::StoreActionAttachmentAbsent,
                        );
                    }
                    render.stencil_attachment.store_action_options = raw;
                }
                _ => {
                    return Err(
                        ReplacementRenderStateResolutionError::InvalidStoreActionOptions {
                            raw: command.mode,
                        },
                    )
                }
            }
        }
        render::Kind::SetVisibilityResultMode => {
            let raw = u32::try_from(command.mode).map_err(|_| {
                ReplacementRenderStateResolutionError::InvalidVisibilityMode { raw: command.mode }
            })?;
            render.visibility = reims_vgpu_protocol::visibility_result_mode(raw)
                .map_err(
                    |_| ReplacementRenderStateResolutionError::InvalidVisibilityMode {
                        raw: command.mode,
                    },
                )?
                .map(|mode| (mode, command.visibility_result_offset));
        }
        render::Kind::RenderPass => {
            render
                .color_attachments
                .clone_from(&command.color_attachments);
            render.depth_attachment = command.depth;
            render.stencil_attachment = command.stencil;
            render.visibility_result_buffer_ref = command.pass_visibility_result_buffer_ref;
            render.render_target_array_length = command.pass_render_target_array_length;
            render.render_target_width = command.pass_render_target_width;
            render.render_target_height = command.pass_render_target_height;
        }
        _ => return Err(ReplacementRenderStateResolutionError::NotStateRecord),
    }
    Ok(())
}

fn resolve_render_pass_actions(
    role: reims_vgpu_core::RenderAttachmentRole,
    load: u16,
    store: u16,
) -> Result<
    (
        reims_vgpu_protocol::pass_action::LoadAction,
        reims_vgpu_protocol::pass_action::StoreAction,
    ),
    ReplacementRenderPassResolutionError,
> {
    let load = reims_vgpu_protocol::pass_action::load_action(load).map_err(|error| {
        ReplacementRenderPassResolutionError::PassAction {
            role,
            kind: ReplacementRenderPassActionKind::Load,
            raw: error.raw(),
        }
    })?;
    let store = reims_vgpu_protocol::pass_action::store_action(store).map_err(|error| {
        ReplacementRenderPassResolutionError::PassAction {
            role,
            kind: ReplacementRenderPassActionKind::Store,
            raw: error.raw(),
        }
    })?;
    Ok((load, store))
}

/// The census name for a colour attachment's load action.
///
/// Which of the three the guest's compositor asks for is the difference between
/// "this pass owns the whole target" and "this pass adds to what is already
/// there", and a target that comes out banded is a target where those two
/// answers disagree with what the pass actually wrote. Counting them costs one
/// increment per attachment and is the only way to read the guest's own
/// statement without a decoder trace.
const fn load_action_census_name(
    load: reims_vgpu_protocol::pass_action::LoadAction,
) -> &'static str {
    use reims_vgpu_protocol::pass_action::LoadAction;
    match load {
        LoadAction::DontCare => "render_color_load_dont_care",
        LoadAction::Load => "render_color_load_load",
        LoadAction::Clear => "render_color_load_clear",
    }
}

/// The census name for a colour attachment's store action.
const fn store_action_census_name(
    store: reims_vgpu_protocol::pass_action::StoreAction,
) -> &'static str {
    use reims_vgpu_protocol::pass_action::StoreAction;
    match store {
        StoreAction::DontCare => "render_color_store_dont_care",
        StoreAction::Store => "render_color_store_store",
        StoreAction::MultisampleResolve => "render_color_store_resolve",
        StoreAction::StoreAndMultisampleResolve => "render_color_store_store_and_resolve",
    }
}

pub(crate) fn resolve_render_pass_attachments<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
) -> Result<Box<[reims_vgpu_core::ResolvedRenderAttachment]>, ReplacementRenderPassResolutionError>
{
    use reims_vgpu_core::{RenderAttachmentClear, RenderAttachmentRole, ResolvedRenderAttachment};

    let render = state
        .render()
        .ok_or(ReplacementRenderPassResolutionError::OutsideEncoder)?;
    if render.depth_attachment.texture_ref != 0
        && render.stencil_attachment.texture_ref != 0
        && render.depth_attachment.texture_ref != render.stencil_attachment.texture_ref
    {
        return Err(ReplacementRenderPassResolutionError::DepthStencilAttachmentMismatch);
    }
    let mut resolved = Vec::new();
    for (slot, attachment) in render.color_attachments.iter().enumerate() {
        if attachment.texture_ref == 0 {
            continue;
        }
        let role = RenderAttachmentRole::Color(slot as u32);
        if attachment.store_action_options != 0 {
            return Err(
                ReplacementRenderPassResolutionError::StoreActionOptionsUnsupported {
                    role,
                    raw: attachment.store_action_options,
                },
            );
        }
        let (load, store) =
            resolve_render_pass_actions(role, attachment.load_action, attachment.store_action)?;
        crate::runtime::contract_census::note(load_action_census_name(load));
        crate::runtime::contract_census::note(store_action_census_name(store));
        let target = runtime
            .resolve_render_attachment(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(attachment.texture_ref),
                attachment.level,
                attachment.slice,
                attachment.depth_plane,
                reims_vgpu_core::ImageAspect::Color,
            )
            .map_err(|reason| ReplacementRenderPassResolutionError::Attachment { role, reason })?;
        let resolve = (attachment.resolve_texture_ref != 0)
            .then(|| {
                let resolve = runtime
                    .resolve_render_attachment(
                        task,
                        reims_vgpu_protocol::ObjectTableRef::new(attachment.resolve_texture_ref),
                        attachment.resolve_level,
                        attachment.resolve_slice,
                        attachment.resolve_depth_plane,
                        reims_vgpu_core::ImageAspect::Color,
                    )
                    .map_err(|reason| ReplacementRenderPassResolutionError::Attachment {
                        role,
                        reason,
                    })?;
                if resolve.extent != target.extent
                    || resolve.pixel_format != target.pixel_format
                    || resolve.sample_count != 1
                {
                    return Err(ReplacementRenderPassResolutionError::ResolveMismatch(role));
                }
                Ok(reims_vgpu_core::ResolvedRenderResolveAttachment {
                    resource: resolve.resource,
                    image_owner: resolve.image_owner,
                    backing: resolve.backing,
                    regions: Box::new([resolve.region]),
                    pixel_format: resolve.pixel_format,
                    extent: resolve.extent,
                    sample_count: resolve.sample_count,
                })
            })
            .transpose()?;
        // Which backing this pass writes and how it opens it, once per
        // backing and again whenever that backing's opening changes. A guest
        // that composites only its damage opens the scanout it took with
        // `Load` and relies on the buffer still holding the frame it last
        // presented; a scanout opened with `Clear` or `DontCare` instead is a
        // black screen carrying only the newest damage, which is what the
        // counters alone cannot separate from a present that never ran.
        if reims_vgpu_observe::state_changed(
            "replacement_render_attachment_open",
            target.backing.get(),
            u64::from(load as u32) << 8 | u64::from(store as u32),
        ) {
            crate::observe::off(format!(
                "replacement_render_attachment_open backing={} slot={slot} load={} store={} \
                 extent={}x{}",
                target.backing.get(),
                load_action_census_name(load),
                store_action_census_name(store),
                target.extent[0],
                target.extent[1],
            ));
        }
        resolved.push(ResolvedRenderAttachment {
            role,
            resource: target.resource,
            image_owner: target.image_owner,
            backing: target.backing,
            regions: Box::new([target.region]),
            pixel_format: target.pixel_format,
            extent: target.extent,
            sample_count: target.sample_count,
            load,
            store,
            clear: RenderAttachmentClear::Color(
                attachment.clear_color.map(|value| (value as f32).to_bits()),
            ),
            resolve,
            // The encoder's own reads decide both of these; they are stamped
            // once the whole encoder is in hand.
            feedback_loop: false,
            input_attachment: false,
        });
    }
    for (
        role,
        attachment_ref,
        resolve_ref,
        level,
        slice,
        depth_plane,
        resolve_level,
        resolve_slice,
        resolve_depth_plane,
        load_raw,
        store_raw,
        store_options,
        resolve_filter,
        clear,
    ) in [
        (
            RenderAttachmentRole::Depth,
            render.depth_attachment.texture_ref,
            render.depth_attachment.resolve_texture_ref,
            render.depth_attachment.level,
            render.depth_attachment.slice,
            render.depth_attachment.depth_plane,
            render.depth_attachment.resolve_level,
            render.depth_attachment.resolve_slice,
            render.depth_attachment.resolve_depth_plane,
            render.depth_attachment.load_action,
            render.depth_attachment.store_action,
            render.depth_attachment.store_action_options,
            render.depth_attachment.resolve_filter,
            RenderAttachmentClear::Depth((render.depth_attachment.clear_depth as f32).to_bits()),
        ),
        (
            RenderAttachmentRole::Stencil,
            render.stencil_attachment.texture_ref,
            render.stencil_attachment.resolve_texture_ref,
            render.stencil_attachment.level,
            render.stencil_attachment.slice,
            render.stencil_attachment.depth_plane,
            render.stencil_attachment.resolve_level,
            render.stencil_attachment.resolve_slice,
            render.stencil_attachment.resolve_depth_plane,
            render.stencil_attachment.load_action,
            render.stencil_attachment.store_action,
            render.stencil_attachment.store_action_options,
            render.stencil_attachment.resolve_filter,
            RenderAttachmentClear::Stencil(render.stencil_attachment.clear_stencil),
        ),
    ] {
        if attachment_ref == 0 {
            continue;
        }
        if store_options != 0 {
            return Err(
                ReplacementRenderPassResolutionError::StoreActionOptionsUnsupported {
                    role,
                    raw: store_options,
                },
            );
        }
        if resolve_ref != 0 && resolve_filter != 0 {
            return Err(
                ReplacementRenderPassResolutionError::ResolveFilterUnsupported {
                    role,
                    raw: resolve_filter,
                },
            );
        }
        let (load, store) = resolve_render_pass_actions(role, load_raw, store_raw)?;
        let aspect = match role {
            RenderAttachmentRole::Depth => reims_vgpu_core::ImageAspect::Depth,
            RenderAttachmentRole::Stencil => reims_vgpu_core::ImageAspect::Stencil,
            RenderAttachmentRole::Color(_) => unreachable!("color attachments resolve above"),
        };
        let target = runtime
            .resolve_render_attachment(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(attachment_ref),
                level,
                slice,
                depth_plane,
                aspect,
            )
            .map_err(|reason| ReplacementRenderPassResolutionError::Attachment { role, reason })?;
        let resolve = (resolve_ref != 0)
            .then(|| {
                let resolve = runtime
                    .resolve_render_attachment(
                        task,
                        reims_vgpu_protocol::ObjectTableRef::new(resolve_ref),
                        resolve_level,
                        resolve_slice,
                        resolve_depth_plane,
                        aspect,
                    )
                    .map_err(|reason| ReplacementRenderPassResolutionError::Attachment {
                        role,
                        reason,
                    })?;
                if resolve.extent != target.extent
                    || resolve.pixel_format != target.pixel_format
                    || resolve.sample_count != 1
                {
                    return Err(ReplacementRenderPassResolutionError::ResolveMismatch(role));
                }
                Ok(reims_vgpu_core::ResolvedRenderResolveAttachment {
                    resource: resolve.resource,
                    image_owner: resolve.image_owner,
                    backing: resolve.backing,
                    regions: Box::new([resolve.region]),
                    pixel_format: resolve.pixel_format,
                    extent: resolve.extent,
                    sample_count: resolve.sample_count,
                })
            })
            .transpose()?;
        resolved.push(ResolvedRenderAttachment {
            role,
            resource: target.resource,
            image_owner: target.image_owner,
            backing: target.backing,
            regions: Box::new([target.region]),
            pixel_format: target.pixel_format,
            extent: target.extent,
            sample_count: target.sample_count,
            load,
            store,
            clear,
            resolve,
            // See the colour arm above: stamped over the whole encoder.
            feedback_loop: false,
            input_attachment: false,
        });
    }
    Ok(resolved.into_boxed_slice())
}

pub(crate) fn resolve_render_draw<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
    command: &render::Command,
) -> Result<Option<reims_vgpu_core::ResolvedRenderDispatch>, ReplacementRenderDrawResolutionError> {
    use reims_vgpu_core::{
        AccessMode, BackingRegion, RenderBindingClass, RenderBindingView, RenderScissor,
        RenderViewport, ResolvedRenderDraw, ResolvedRenderRasterState,
        ResolvedRenderResourceBinding,
    };

    if !matches!(
        command.kind,
        render::Kind::Draw | render::Kind::DrawIndirect
    ) {
        return Err(ReplacementRenderDrawResolutionError::NotDraw);
    }
    let snapshot = state
        .render()
        .ok_or(ReplacementRenderDrawResolutionError::OutsideEncoder)?;
    if snapshot.render_target_array_length != 0 {
        return Err(
            ReplacementRenderDrawResolutionError::RenderTargetArrayLengthUnsupported {
                length: snapshot.render_target_array_length,
            },
        );
    }
    if snapshot.pipeline_ref == 0 {
        return Err(ReplacementRenderDrawResolutionError::PipelineUnbound);
    }
    let pipeline_reference = reims_vgpu_protocol::SerializerRef::new(snapshot.pipeline_ref);
    let pipeline_identity = runtime
        .resolve_render_pipeline(task, pipeline_reference)
        .ok_or(ReplacementRenderDrawResolutionError::Pipeline(
        crate::runtime::replacement_session::ReplacementRenderSemanticAvailability::UnknownPipeline,
    ))?;
    let pipeline = runtime
        .resolve_render_semantic(task, pipeline_reference)
        .map_err(ReplacementRenderDrawResolutionError::Pipeline)?;
    for (stage, expected, interface) in [
        (
            reims_vgpu_core::ShaderStage::Vertex,
            reims_vgpu_core::ReflectedShaderStage::Vertex,
            pipeline.vertex.interface.as_ref(),
        ),
        (
            reims_vgpu_core::ShaderStage::Fragment,
            reims_vgpu_core::ReflectedShaderStage::Fragment,
            pipeline.fragment.interface.as_ref(),
        ),
    ] {
        if let Some(unsupported) = interface.first_unsupported_interface(expected) {
            return Err(
                ReplacementRenderDrawResolutionError::ReflectedInterfaceUnrepresented {
                    stage,
                    feature: unsupported.feature,
                    count: unsupported.count,
                },
            );
        }
        if let Some(binding) = interface.bindings.iter().find(|binding| {
            !matches!(
                binding.kind,
                reims_vgpu_core::ShaderResourceKind::ColorInput
                    | reims_vgpu_core::ShaderResourceKind::Sampler
                    | reims_vgpu_core::ShaderResourceKind::StaticSampler
                    | reims_vgpu_core::ShaderResourceKind::Texture
                    | reims_vgpu_core::ShaderResourceKind::TextureArray
                    | reims_vgpu_core::ShaderResourceKind::StorageImage
                    | reims_vgpu_core::ShaderResourceKind::Buffer
            )
        }) {
            return Err(
                ReplacementRenderDrawResolutionError::ReflectedResourceUnrepresented {
                    stage,
                    index: binding.metal_index,
                    kind: binding.kind,
                },
            );
        }
    }
    let topology = reims_vgpu_protocol::primitive_topology(command.primitive_type)
        .map_err(ReplacementRenderDrawResolutionError::PrimitiveTopology)?;
    let indirect = command.kind == render::Kind::DrawIndirect;
    let indexed = command.index_buffer_ref != 0 || command.index_count != 0;
    let count = if indexed {
        command.index_count
    } else {
        command.vertex_count
    };
    if !indirect && (count == 0 || command.instance_count == 0) {
        return Ok(None);
    }

    let mut attachments = resolve_render_pass_attachments(runtime, task, state)
        .map_err(ReplacementRenderDrawResolutionError::Attachment)?;
    // A fragment stage that declares a colour input attachment reads the
    // attachment it also writes, and the pass must declare that attachment in
    // a layout permitting both. Metal names only the first destination this
    // way, which is the attachment Vulkan wires as the subpass input.
    if pipeline
        .fragment
        .interface
        .bindings
        .iter()
        .any(|binding| binding.kind == reims_vgpu_core::ShaderResourceKind::ColorInput)
    {
        for attachment in attachments.iter_mut() {
            if attachment.role == reims_vgpu_core::RenderAttachmentRole::Color(0) {
                attachment.input_attachment = true;
            }
        }
    }
    let samplers = resolve_render_samplers(runtime, task, snapshot, &pipeline)?;
    let textures = resolve_render_textures(runtime, task, snapshot, &pipeline)?;
    let buffer_resources = resolve_render_buffers(runtime, task, snapshot, &pipeline, command)?;
    let vertex_resources = resolve_vertex_buffers(runtime, task, snapshot, &pipeline)?;
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
    let explicit_extent = |value: u64, fallback| {
        if value == 0 {
            Ok(fallback)
        } else {
            u32::try_from(value)
                .map_err(|_| ReplacementRenderDrawResolutionError::RenderExtentOutOfRange)
        }
    };
    let render_extent = [
        explicit_extent(snapshot.render_target_width, minimum_width)?,
        explicit_extent(snapshot.render_target_height, minimum_height)?,
    ];
    if let Some(attachment) = attachments.iter().find(|attachment| {
        render_extent[0] > attachment.extent[0] || render_extent[1] > attachment.extent[1]
    }) {
        return Err(
            ReplacementRenderDrawResolutionError::RenderExtentPastAttachment {
                role: attachment.role,
                requested: render_extent,
                available: [attachment.extent[0], attachment.extent[1]],
            },
        );
    }
    let depth_stencil = if snapshot.depth_stencil_ref == 0 {
        None
    } else {
        Some(
            runtime
                .resolve_depth_stencil(
                    task,
                    reims_vgpu_protocol::SerializerRef::new(snapshot.depth_stencil_ref),
                )
                .ok_or(ReplacementRenderDrawResolutionError::DepthStencilStateMissing)?
                .0,
        )
    };
    let visibility = snapshot
        .visibility
        .map(|(mode, offset)| {
            if snapshot.visibility_result_buffer_ref == 0 {
                return Err(ReplacementRenderDrawResolutionError::VisibilityBufferMissing);
            }
            let range = runtime
                .resolve_buffer_range(
                    task,
                    reims_vgpu_protocol::ObjectTableRef::new(snapshot.visibility_result_buffer_ref),
                    offset,
                    u64::from(u64::BITS / 8),
                )
                .map_err(ReplacementRenderDrawResolutionError::VisibilityBuffer)?;
            Ok(reims_vgpu_core::ResolvedRenderVisibility {
                mode,
                resource: range.resource,
                backing: range.storage,
                range: range.region,
            })
        })
        .transpose()?;
    let mut resources = textures.resources.into_vec();
    resources.extend(buffer_resources);
    resources.extend(vertex_resources.resources);
    let draw = if indexed {
        if command.index_buffer_ref == 0 {
            return Err(ReplacementRenderDrawResolutionError::IndexedBufferMissing);
        }
        let index_type = reims_vgpu_protocol::decode_index_type(command.index_type)
            .map_err(ReplacementRenderDrawResolutionError::IndexType)?;
        let range = if indirect {
            runtime.resolve_buffer_tail(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(command.index_buffer_ref),
                command.index_buffer_offset,
            )
        } else {
            let width = index_type.byte_size() as u64;
            let length = u64::from(command.index_count)
                .checked_mul(width)
                .ok_or(ReplacementRenderDrawResolutionError::IndexRangeOverflow)?;
            runtime.resolve_buffer_range(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(command.index_buffer_ref),
                command.index_buffer_offset,
                length,
            )
        }
        .map_err(ReplacementRenderDrawResolutionError::IndexBuffer)?;
        resources.push(ResolvedRenderResourceBinding {
            class: RenderBindingClass::IndexBuffer,
            binding: 0,
            array_element: 0,
            descriptor_count: 1,
            stages: reims_vgpu_protocol::RenderStages::from_bits(
                reims_vgpu_protocol::RenderStages::VERTEX.into(),
            )
            .expect("the vertex render-stage bit is valid"),
            resource: range.resource,
            backing: range.storage,
            view: RenderBindingView::Buffer(range.region),
            regions: Box::new([BackingRegion::Linear(range.region)]),
            mode: AccessMode::Read,
        });
        if indirect {
            ResolvedRenderDraw::IndexedIndirect {
                topology,
                index_type,
            }
        } else {
            ResolvedRenderDraw::Indexed {
                topology,
                index_type,
                index_count: command.index_count,
                instance_count: command.instance_count,
                first_index: 0,
                vertex_offset: i32::try_from(command.base_vertex).map_err(|_| {
                    ReplacementRenderDrawResolutionError::IndexVertexOffsetOutOfRange
                })?,
                first_instance: command.base_instance,
            }
        }
    } else if indirect {
        ResolvedRenderDraw::Indirect { topology }
    } else {
        ResolvedRenderDraw::Direct {
            topology,
            vertex_count: command.vertex_count,
            instance_count: command.instance_count,
            first_vertex: command.vertex_start,
            first_instance: command.base_instance,
        }
    };
    if indirect {
        let argument_length = if indexed {
            reims_vgpu_core::RENDER_INDEXED_INDIRECT_ARGUMENT_BYTES
        } else {
            reims_vgpu_core::RENDER_INDIRECT_ARGUMENT_BYTES
        };
        let range = runtime
            .resolve_buffer_range(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(command.indirect_buffer_ref),
                command.indirect_buffer_offset,
                argument_length,
            )
            .map_err(ReplacementRenderDrawResolutionError::IndirectBuffer)?;
        resources.push(ResolvedRenderResourceBinding {
            class: RenderBindingClass::IndirectBuffer,
            binding: 0,
            array_element: 0,
            descriptor_count: 1,
            stages: render_stage_bits(reims_vgpu_core::ShaderStage::Vertex),
            resource: range.resource,
            backing: range.storage,
            view: RenderBindingView::Buffer(range.region),
            regions: Box::new([BackingRegion::Linear(range.region)]),
            mode: AccessMode::Read,
        });
    }
    let operation = crate::runtime::replacement_render_projection::construct(
        crate::runtime::replacement_render_projection::ResolvedRenderConstructionInput {
            pipeline: pipeline_identity,
            program: reims_vgpu_core::PreparedRenderProgram {
                vertex: pipeline.vertex.variant().program.clone(),
                fragment: pipeline.fragment.variant().program.clone(),
            },
            depth_stencil,
            render_extent,
            raster: ResolvedRenderRasterState {
                viewports: snapshot
                    .viewports_bits
                    .iter()
                    .map(|bits| RenderViewport {
                        origin_x_bits: bits[0],
                        origin_y_bits: bits[1],
                        width_bits: bits[2],
                        height_bits: bits[3],
                        near_bits: bits[4],
                        far_bits: bits[5],
                    })
                    .collect(),
                scissors: snapshot
                    .scissors
                    .iter()
                    .map(|scissor| RenderScissor {
                        x: scissor.x,
                        y: scissor.y,
                        width: scissor.width,
                        height: scissor.height,
                    })
                    .collect(),
                cull_mode: snapshot.cull_mode,
                front_face_ccw: snapshot.front_face_ccw,
                fill_mode: snapshot.fill_mode,
                line_width_bits: snapshot.line_width.bits(),
                depth_clip_mode: snapshot.depth_clip_mode,
                depth_bias_bits: snapshot.depth_bias_bits,
                blend_color_bits: snapshot.blend_color_bits,
                stencil_reference: snapshot.stencil_reference.unwrap_or((0, 0)).into(),
            },
            visibility,
            begins_encoder: true,
            ends_encoder: true,
            draw,
            vertex_buffers: vertex_resources.layouts,
            attachments,
            resources: resources.into_boxed_slice(),
            null_bindings: textures.null_bindings,
            samplers,
        },
    );
    Ok(Some(operation))
}

fn resolve_render_tessellation(
    state: &ReplacementRecordProjectionState,
    command: &render::Command,
) -> ReplacementRenderTessellationResolutionError {
    use ReplacementRenderTessellationResolutionError as Error;

    let Some(render) = state.render() else {
        return Error::OutsideEncoder {
            opcode: command.opcode,
        };
    };
    let Some(draw) = command.patch_draw else {
        return Error::GeometryMissing {
            opcode: command.opcode,
        };
    };
    Error::Unsupported {
        opcode: command.opcode,
        draw,
        factor_buffer: render.tessellation_factor_buffer,
    }
}

fn resolve_render_tile(
    state: &ReplacementRecordProjectionState,
    command: &render::Command,
) -> Result<(), ReplacementRenderTileResolutionError> {
    use ReplacementRenderTileResolutionError as Error;

    if state.render().is_none() {
        return Err(Error::OutsideEncoder {
            kind: command.kind,
            opcode: command.opcode,
        });
    }
    match command.kind {
        render::Kind::TileBind => Err(Error::BindUnsupported {
            opcode: command.opcode,
            binding: command.tile_bind.clone().ok_or(Error::Malformed {
                kind: command.kind,
                opcode: command.opcode,
            })?,
        }),
        render::Kind::TileDispatch if command.tile_threads.contains(&0) => Ok(()),
        render::Kind::TileDispatch => Err(Error::DispatchUnsupported {
            opcode: command.opcode,
            threads_per_tile: command.tile_threads,
            region: command.tile_region,
            render_target_array_index: command.tile_render_target_array_index,
        }),
        render::Kind::TileDimensionsQuery => Err(Error::DimensionsQueryUnsupported {
            opcode: command.opcode,
            buffer_ref: command.buffer_ref,
            offset: command.buffer_offset,
        }),
        _ => Err(Error::Malformed {
            kind: command.kind,
            opcode: command.opcode,
        }),
    }
}

fn resolve_compute_state_record(
    state: &mut ReplacementRecordProjectionState,
    command: &compute::Command,
) -> Result<(), ReplacementComputeStateResolutionError> {
    use crate::runtime::replacement_compute_state::{
        ImageblockDimensions, StageInRegion, MAX_COMPUTE_BUFFER_SLOTS, MAX_COMPUTE_SAMPLER_SLOTS,
        MAX_COMPUTE_TEXTURE_SLOTS,
    };

    let accum = state.compute_mut()?;
    match command.kind {
        compute::Kind::Pipeline => accum.pipeline_ref = command.pipeline_ref,
        compute::Kind::BufferBind | compute::Kind::BufferBindAttributeStride => {
            replacement_compute_binding_range(
                command.first,
                command.buffers.len(),
                MAX_COMPUTE_BUFFER_SLOTS,
                |first, count, limit| ReplacementComputeStateResolutionError::BufferIndexOverflow {
                    first,
                    count,
                    limit,
                },
            )?;
            accum.bind_buffers(command.first, &command.buffers);
        }
        compute::Kind::BufferOffset => {
            replacement_compute_buffer_offset(accum, command.first)?;
            accum.set_buffer_offset(command.first, command.buffer_offset, None);
        }
        compute::Kind::BufferOffsetAttributeStride => {
            replacement_compute_buffer_offset(accum, command.first)?;
            accum.set_buffer_offset(
                command.first,
                command.buffer_offset,
                Some(command.attribute_stride),
            );
        }
        compute::Kind::TextureBind => {
            replacement_compute_binding_range(
                command.first,
                command.textures.len(),
                MAX_COMPUTE_TEXTURE_SLOTS,
                |first, count, limit| {
                    ReplacementComputeStateResolutionError::TextureIndexOverflow {
                        first,
                        count,
                        limit,
                    }
                },
            )?;
            accum.bind_textures(command.first, &command.textures);
        }
        compute::Kind::SamplerBind | compute::Kind::SamplerLod => {
            replacement_compute_binding_range(
                command.first,
                command.samplers.len(),
                MAX_COMPUTE_SAMPLER_SLOTS,
                |first, count, limit| {
                    ReplacementComputeStateResolutionError::SamplerIndexOverflow {
                        first,
                        count,
                        limit,
                    }
                },
            )?;
            accum.bind_samplers(command.first, &command.samplers);
        }
        compute::Kind::DispatchType => {
            if !reims_vgpu_protocol::dispatch::is_declared_dispatch_type(command.dispatch_type) {
                return Err(
                    ReplacementComputeStateResolutionError::DispatchTypeUnsupported(
                        command.dispatch_type,
                    ),
                );
            }
            accum.dispatch_type = command.dispatch_type;
        }
        compute::Kind::StageInRegion => accum.set_stage_in_region(StageInRegion {
            origin_x: command.stage_in_region.origin.x,
            origin_y: command.stage_in_region.origin.y,
            origin_z: command.stage_in_region.origin.z,
            size_x: command.stage_in_region.size.x,
            size_y: command.stage_in_region.size.y,
            size_z: command.stage_in_region.size.z,
        }),
        compute::Kind::StageInRegionIndirect => accum.set_stage_in_region_indirect(
            command.stage_in_indirect_buffer_ref,
            command.stage_in_indirect_buffer_offset,
        ),
        compute::Kind::ThreadgroupMemory => accum.set_threadgroup_memory(
            command.threadgroup_memory_index,
            command.threadgroup_memory_length,
        ),
        compute::Kind::ImageblockDimensions => {
            let dimensions = ImageblockDimensions {
                width: command.imageblock_width,
                height: command.imageblock_height,
            };
            accum.set_imageblock(dimensions.width, dimensions.height);
        }
        compute::Kind::CompressedTextureFlush => {}
        _ => return Err(ReplacementComputeStateResolutionError::NotStateRecord),
    }
    Ok(())
}

fn replacement_compute_binding_range(
    first: u32,
    count: usize,
    limit: u32,
    overflow: impl Fn(u32, u64, u32) -> ReplacementComputeStateResolutionError,
) -> Result<(), ReplacementComputeStateResolutionError> {
    let count = u64::try_from(count).unwrap_or(u64::MAX);
    let end = u64::from(first).checked_add(count);
    if end.is_none_or(|end| end > u64::from(limit)) {
        return Err(overflow(first, count, limit));
    }
    Ok(())
}

fn replacement_compute_buffer_offset(
    accum: &crate::runtime::replacement_compute_state::ComputeAccum,
    index: u32,
) -> Result<(), ReplacementComputeStateResolutionError> {
    if index >= crate::runtime::replacement_compute_state::MAX_COMPUTE_BUFFER_SLOTS {
        return Err(
            ReplacementComputeStateResolutionError::BufferIndexOverflow {
                first: index,
                count: 1,
                limit: crate::runtime::replacement_compute_state::MAX_COMPUTE_BUFFER_SLOTS,
            },
        );
    }
    if !accum.buffers.iter().any(|binding| binding.index == index) {
        return Err(ReplacementComputeStateResolutionError::BufferOffsetUnbound { index });
    }
    Ok(())
}

struct RootComputeResolver<'a, Semantic> {
    runtime: &'a ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
}

impl<Semantic: Clone> crate::runtime::replacement_compute_projection::ReplacementComputeResolver
    for RootComputeResolver<'_, Semantic>
{
    fn pipeline(
        &mut self,
        reference: u32,
    ) -> Result<
        reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ComputePipelineObject>,
        crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError,
    > {
        self.runtime
            .resolve_compute_pipeline(self.task, reims_vgpu_protocol::SerializerRef::new(reference))
            .ok_or(
                crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError::PipelineMissing,
            )
    }

    fn buffer(
        &mut self,
        index: u32,
        reference: u32,
        offset: u64,
        length: Option<u64>,
    ) -> Result<
        crate::runtime::replacement_compute_projection::ReplacementComputeBufferBinding,
        crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError,
    > {
        use crate::runtime::replacement_compute_projection::{
            ReplacementComputeBufferBinding, ReplacementComputeConstructionError as Error,
        };
        let object = reims_vgpu_protocol::ObjectTableRef::new(reference);
        let resolved = match length {
            Some(length) => self
                .runtime
                .resolve_buffer_range(self.task, object, offset, length),
            None => self.runtime.resolve_buffer_tail(self.task, object, offset),
        }
        .map_err(|_| Error::BufferMissing(index))?;
        Ok(ReplacementComputeBufferBinding {
            resource: resolved.resource,
            backing: resolved.storage,
            range: resolved.region,
        })
    }

    fn texture(
        &mut self,
        index: u32,
        reference: u32,
    ) -> Result<
        crate::runtime::replacement_compute_projection::ReplacementComputeTextureBinding,
        crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError,
    > {
        use crate::runtime::replacement_compute_projection::{
            ReplacementComputeConstructionError as Error, ReplacementComputeTextureBinding,
        };
        let object = reims_vgpu_protocol::ObjectTableRef::new(reference);
        let resolved = self
            .runtime
            .resolve_texture_binding(self.task, object)
            .map_err(|reason| match reason {
                crate::runtime::replacement_session::ReplacementTextureResolutionError::ResourceUnavailable => {
                    Error::TextureMissing(index)
                }
                crate::runtime::replacement_session::ReplacementTextureResolutionError::View(reason) => {
                    Error::TextureView(reason)
                }
                crate::runtime::replacement_session::ReplacementTextureResolutionError::StorageUnavailable => {
                    self.runtime
                        .resolve_resource(self.task, object)
                        .map(Error::ResourceBackingMissing)
                        .unwrap_or(Error::TextureMissing(index))
                }
            })?;
        Ok(ReplacementComputeTextureBinding {
            resource: resolved.resource,
            backing: resolved.backing,
            view: resolved.view,
        })
    }

    fn sampler(
        &mut self,
        index: u32,
        reference: u32,
        binding: u32,
    ) -> Result<
        reims_vgpu_core::SamplerResource,
        crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError,
    > {
        use crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError as Error;
        let (identity, descriptor) = self
            .runtime
            .resolve_sampler(
                self.task,
                reims_vgpu_protocol::SerializerRef::new(reference),
            )
            .ok_or(Error::SamplerMissing(index))?;
        crate::runtime::replacement_sampler_projection::decoded_sampler(
            reference,
            binding,
            identity,
            &descriptor,
        )
        .map_err(|_| Error::SamplerState)
    }
}

fn resolve_direct_compute_dispatch<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    accum: &crate::runtime::replacement_compute_state::ComputeAccum,
    command: &compute::Command,
) -> Result<reims_vgpu_core::ResolvedComputeDispatch, ReplacementComputeDispatchResolutionError> {
    use crate::runtime::replacement_compute_projection::{
        construct_resolved, ReplacementComputeConstructionError as Construction,
    };

    let grid_is_threads = match command.kind {
        compute::Kind::DispatchThreadgroups => Some(false),
        compute::Kind::DispatchThreads => Some(true),
        compute::Kind::DispatchThreadgroupsIndirect => None,
        compute::Kind::DispatchThreadsIndirect => {
            return Err(ReplacementComputeDispatchResolutionError::
                IndirectThreadsRequireGpuWorkgroupConversion {
                    buffer_ref: command.indirect_buffer_ref,
                    offset: command.indirect_buffer_offset,
                });
        }
        _ => return Err(ReplacementComputeDispatchResolutionError::NotDirectDispatch),
    };
    let narrow = |value: u64| {
        u32::try_from(value)
            .ok()
            .filter(|value| *value != 0)
            .ok_or(ReplacementComputeDispatchResolutionError::GridDimensionOutOfRange)
    };
    let group = [
        narrow(command.threads_per_threadgroup.x)?,
        narrow(command.threads_per_threadgroup.y)?,
        narrow(command.threads_per_threadgroup.z)?,
    ];
    let launch = if let Some(grid_is_threads) = grid_is_threads {
        let grid = [
            narrow(command.grid.x)?,
            narrow(command.grid.y)?,
            narrow(command.grid.z)?,
        ];
        reims_vgpu_core::ResolvedComputeLaunch::Direct(
            reims_vgpu_protocol::dispatch::workgroup_counts(grid, group, grid_is_threads)
                .ok_or(ReplacementComputeDispatchResolutionError::GridInvalid)?,
        )
    } else {
        let arguments = runtime
            .resolve_buffer_range(
                task,
                reims_vgpu_protocol::ObjectTableRef::new(command.indirect_buffer_ref),
                command.indirect_buffer_offset,
                reims_vgpu_core::COMPUTE_INDIRECT_ARGUMENT_BYTES,
            )
            .map_err(ReplacementComputeDispatchResolutionError::IndirectBuffer)?;
        reims_vgpu_core::ResolvedComputeLaunch::IndirectThreadgroups {
            arguments,
            threads_per_threadgroup: group,
        }
    };
    let pipeline_reference = reims_vgpu_protocol::SerializerRef::new(accum.pipeline_ref);
    let pipeline = runtime
        .resolve_compute_pipeline(task, pipeline_reference)
        .ok_or(ReplacementComputeDispatchResolutionError::Construction(
            Construction::PipelineMissing,
        ))?;
    let contract = runtime.session().compute_contract(pipeline).ok_or(
        ReplacementComputeDispatchResolutionError::Construction(Construction::PipelineMissing),
    )?;
    if let Some(stage_input) = contract.stage_input.as_ref() {
        return Err(
            ReplacementComputeDispatchResolutionError::StageInputUnsupported(Box::new(
                stage_input.clone(),
            )),
        );
    }
    if let Some(imageblock) = accum.imageblock {
        return Err(ReplacementComputeDispatchResolutionError::ImageblockUnsupported(imageblock));
    }
    // Concurrent dispatch permits command overlap but does not require it.
    // Replacement scheduling remains conservatively ordered until native
    // independence is proven; executing the same dispatch serially preserves
    // the encoder's guest-visible synchronization contract.
    let translation = runtime
        .resolve_compute_translation(task, pipeline_reference, group)
        .map_err(ReplacementComputeDispatchResolutionError::Pipeline)?;
    construct_resolved(
        &mut RootComputeResolver { runtime, task },
        accum,
        launch,
        translation.as_ref(),
    )
    .map_err(ReplacementComputeDispatchResolutionError::Construction)
}

fn resolve_compute_control_flow<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
    command: &compute::Command,
) -> ReplacementComputeControlFlowResolutionError {
    use ReplacementComputeControlFlowResolutionError as Error;

    if state.compute.is_none() {
        return Error::OutsideEncoder {
            kind: command.kind,
            opcode: command.opcode,
        };
    }
    let predicate = if matches!(
        command.kind,
        compute::Kind::ControlStartWhile
            | compute::Kind::ControlStartIf
            | compute::Kind::ControlEndDoWhile
    ) {
        const CONDITION_WORD_BYTES: u64 = core::mem::size_of::<u32>() as u64;
        match runtime.resolve_buffer_range(
            task,
            reims_vgpu_protocol::ObjectTableRef::new(command.condition_buffer_ref),
            command.condition_buffer_offset,
            CONDITION_WORD_BYTES,
        ) {
            Ok(buffer) => Some(ReplacementResolvedComputeControlPredicate {
                buffer,
                comparison: command.condition_comparison,
                reference_value: command.condition_reference_value,
            }),
            Err(reason) => {
                return Error::PredicateBuffer {
                    kind: command.kind,
                    opcode: command.opcode,
                    buffer_ref: command.condition_buffer_ref,
                    reason,
                };
            }
        }
    } else {
        None
    };
    Error::Unsupported {
        kind: command.kind,
        opcode: command.opcode,
        predicate,
    }
}

fn resolve_compute_icb_dispatch<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    inherited: &crate::runtime::replacement_compute_state::ComputeAccum,
    descriptor: &reims_vgpu_protocol::IndirectCommandBufferDescriptor,
    fill: &crate::runtime::icb::IcbComputeFill,
) -> Result<reims_vgpu_core::ResolvedComputeDispatch, ReplacementComputeDispatchResolutionError> {
    let pipeline_ref = if descriptor.inherit_pipeline_state() {
        inherited.pipeline_ref
    } else {
        fill.pipeline_ref
    };
    if pipeline_ref == 0 {
        return Err(ReplacementComputeDispatchResolutionError::Construction(
            crate::runtime::replacement_compute_projection::ReplacementComputeConstructionError::IcbPipelineMissing,
        ));
    }
    let local_size = match fill.dispatch {
        crate::runtime::icb::IcbFillDispatch::ConcurrentThreadgroups {
            tg_x, tg_y, tg_z, ..
        }
        | crate::runtime::icb::IcbFillDispatch::ConcurrentThreads {
            tg_x, tg_y, tg_z, ..
        } => [tg_x, tg_y, tg_z],
    };
    let translation = runtime
        .resolve_compute_translation(
            task,
            reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
            local_size,
        )
        .map_err(ReplacementComputeDispatchResolutionError::Pipeline)?;
    crate::runtime::replacement_compute_projection::construct_icb_resolved(
        &mut RootComputeResolver { runtime, task },
        inherited,
        fill,
        descriptor,
        translation.as_ref(),
    )
    .map_err(ReplacementComputeDispatchResolutionError::Construction)
}

pub(crate) fn populate_decoded_compute_icb<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
    read: &crate::runtime::replacement_session::ReplacementIcbCommandMemoryRead,
    slots: Vec<crate::runtime::icb::DecodedIcbCommandSlot>,
) -> Result<ReplacementPopulatedComputeIcb, ReplacementComputeIcbPopulationFailure> {
    let Some(range_end) = read.range.location.checked_add(read.range.length) else {
        return Err(ReplacementComputeIcbPopulationFailure {
            reason: ReplacementComputeIcbPopulationError::DecodedRangeMismatch,
            slots,
        });
    };
    if u64::try_from(slots.len()) != Ok(read.range.length)
        || slots
            .iter()
            .map(|slot| slot.command_index)
            .ne(read.range.location..range_end)
    {
        return Err(ReplacementComputeIcbPopulationFailure {
            reason: ReplacementComputeIcbPopulationError::DecodedRangeMismatch,
            slots,
        });
    }
    let Some(inherited) = state.compute.as_ref() else {
        return Err(ReplacementComputeIcbPopulationFailure {
            reason: ReplacementComputeIcbPopulationError::OutsideEncoder,
            slots,
        });
    };
    let mut resolved = Vec::with_capacity(slots.len());
    for slot in &slots {
        let command = match &slot.command {
            None => None,
            Some(crate::runtime::icb::IcbCommandFill::Render(_)) => {
                return Err(ReplacementComputeIcbPopulationFailure {
                    reason: ReplacementComputeIcbPopulationError::RenderCommandInComputeEncoder {
                        index: slot.command_index,
                    },
                    slots: slots.clone(),
                });
            }
            Some(crate::runtime::icb::IcbCommandFill::Compute(fill)) => Some(
                resolve_compute_icb_dispatch(runtime, task, inherited, &read.descriptor, fill)
                    .map(reims_vgpu_core::ResolvedIndirectCommandSlot::Compute)
                    .map_err(|reason| ReplacementComputeIcbPopulationFailure {
                        reason: ReplacementComputeIcbPopulationError::Dispatch {
                            index: slot.command_index,
                            reason,
                        },
                        slots: slots.clone(),
                    })?,
            ),
        };
        resolved.push((slot.command_index, command));
    }
    let prior = runtime
        .execution_mut()
        .indirect_mut()
        .set_batch(read.identity, resolved.into_boxed_slice())
        .map_err(|failure| ReplacementComputeIcbPopulationFailure {
            reason: ReplacementComputeIcbPopulationError::Population(failure.reason),
            slots,
        })?;
    Ok(ReplacementPopulatedComputeIcb {
        prior,
        execution: reims_vgpu_core::ResolvedIndirectCommand::Execute {
            icb: read.identity,
            range: read.range,
            kind: reims_vgpu_core::IndirectCommandExecutionKind::Compute,
        },
    })
}

fn resolve_compute_icb_execution<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
    command: &compute::Command,
    read_memory: &mut impl FnMut(
        reims_vgpu_protocol::TaskId,
        reims_vgpu_core::IndirectCommandMemoryReadPlan,
    ) -> Result<Vec<u8>, ReplacementIcbCommandMemoryTransportError>,
) -> Result<reims_vgpu_core::ResolvedIndirectCommand, ReplacementComputeIcbExecutionError> {
    if command.kind == compute::Kind::ExecuteCommandsInBufferIndirect {
        return Err(ReplacementComputeIcbExecutionError::IndirectRangeRequiresAsynchronousReadback);
    }
    let reference = reims_vgpu_protocol::SerializerRef::new(command.indirect_command_buffer_ref);
    let range = reims_vgpu_core::ResolvedIndirectCommandRange {
        location: command.indirect_command_range_location,
        length: command.indirect_command_range_length,
    };
    let read = runtime
        .resolve_indirect_command_memory_read(task, reference, range)
        .map_err(ReplacementComputeIcbExecutionError::ReadPlan)?;
    let bytes = match read.plan {
        Some(plan) => {
            read_memory(task, plan).map_err(ReplacementComputeIcbExecutionError::Transport)?
        }
        None => Vec::new(),
    };
    let slots = runtime
        .decode_indirect_command_bytes(task, reference, &read, &bytes)
        .map_err(ReplacementComputeIcbExecutionError::Decode)?;
    populate_decoded_compute_icb(runtime, task, state, &read, slots)
        .map(|populated| populated.execution)
        .map_err(|failure| ReplacementComputeIcbExecutionError::Population(failure.reason))
}

fn resolve_render_icb_draw<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    inherited: &ReplacementRenderEncoderState,
    descriptor: &reims_vgpu_protocol::IndirectCommandBufferDescriptor,
    fill: &crate::runtime::icb::IcbRenderFill,
) -> Result<Option<reims_vgpu_core::ResolvedRenderDispatch>, ReplacementRenderIcbDrawError> {
    use crate::runtime::icb::{IcbRenderBindStage, IcbRenderDraw};

    if !fill.object_threadgroup_memory.is_empty() {
        return Err(ReplacementRenderIcbDrawError::ObjectThreadgroupMemory);
    }
    let mut render = inherited.clone();
    if descriptor.inherit_pipeline_state() {
        if fill.pipeline_ref != 0 {
            return Err(ReplacementRenderIcbDrawError::InheritedPipelineOverride);
        }
    } else {
        render.pipeline_ref = fill.pipeline_ref;
    }
    if render.pipeline_ref == 0 {
        return Err(ReplacementRenderIcbDrawError::PipelineMissing);
    }
    if descriptor.inherit_buffers() {
        if !fill.buffers.is_empty() {
            return Err(ReplacementRenderIcbDrawError::InheritedBuffersOverride);
        }
    } else {
        render.vertex_buffers.fill(None);
        render.fragment_buffers.fill(None);
        for binding in &fill.buffers {
            let table = match binding.effective_stage() {
                IcbRenderBindStage::Vertex => &mut render.vertex_buffers,
                IcbRenderBindStage::Fragment => &mut render.fragment_buffers,
                IcbRenderBindStage::Object | IcbRenderBindStage::Mesh => {
                    return Err(ReplacementRenderIcbDrawError::ObjectOrMeshBuffer);
                }
            };
            let slot = table.get_mut(binding.index as usize).ok_or(
                ReplacementRenderIcbDrawError::BufferIndexOverflow(binding.index),
            )?;
            *slot = (binding.buffer_ref != 0).then_some(ReplacementRenderBufferBinding {
                reference: binding.buffer_ref,
                offset: binding.offset,
                attribute_stride: binding
                    .has_attribute_stride
                    .then_some(binding.attribute_stride),
            });
        }
    }

    let mut command = render::Command {
        kind: render::Kind::Draw,
        instance_count: 1,
        ..Default::default()
    };
    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            command.primitive_type = u32::from(primitive_type);
            command.vertex_start = u32::try_from(vertex_start)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
            command.vertex_count = u32::try_from(vertex_count)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
            command.instance_count = u32::try_from(instance_count)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
            command.base_instance = u32::try_from(base_instance)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
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
            command.primitive_type = u32::from(primitive_type);
            command.index_type = u32::from(index_type);
            command.index_buffer_ref = index_buffer_ref;
            command.index_count = u32::try_from(index_count)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
            command.index_buffer_offset = index_buffer_offset;
            command.instance_count = u32::try_from(instance_count)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
            command.base_vertex = base_vertex;
            command.base_instance = u32::try_from(base_instance)
                .map_err(|_| ReplacementRenderIcbDrawError::GeometryOutOfRange)?;
        }
        IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => {
            return Err(ReplacementRenderIcbDrawError::UnsupportedDraw);
        }
    }
    let state = ReplacementRecordProjectionState {
        render: Some(render),
        compute: None,
    };
    let mut operation = resolve_render_draw(runtime, task, &state, &command)
        .map_err(ReplacementRenderIcbDrawError::Draw)?;
    if let Some(operation) = operation.as_mut() {
        operation.begins_encoder = false;
        operation.ends_encoder = false;
    }
    Ok(operation)
}

pub(crate) fn populate_decoded_render_icb<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
    read: &crate::runtime::replacement_session::ReplacementIcbCommandMemoryRead,
    slots: Vec<crate::runtime::icb::DecodedIcbCommandSlot>,
) -> Result<reims_vgpu_core::ResolvedIndirectCommand, ReplacementRenderIcbPopulationError> {
    let range_end = read
        .range
        .location
        .checked_add(read.range.length)
        .ok_or(ReplacementRenderIcbPopulationError::DecodedRangeMismatch)?;
    if u64::try_from(slots.len()) != Ok(read.range.length)
        || slots
            .iter()
            .map(|slot| slot.command_index)
            .ne(read.range.location..range_end)
    {
        return Err(ReplacementRenderIcbPopulationError::DecodedRangeMismatch);
    }
    let inherited = state
        .render
        .as_ref()
        .ok_or(ReplacementRenderIcbPopulationError::OutsideEncoder)?;
    let mut resolved = Vec::with_capacity(slots.len());
    for slot in &slots {
        let command = match &slot.command {
            None => None,
            Some(crate::runtime::icb::IcbCommandFill::Compute(_)) => {
                return Err(
                    ReplacementRenderIcbPopulationError::ComputeCommandInRenderEncoder {
                        index: slot.command_index,
                    },
                );
            }
            Some(crate::runtime::icb::IcbCommandFill::Render(fill)) => {
                resolve_render_icb_draw(runtime, task, inherited, &read.descriptor, fill)
                    .map(|operation| {
                        operation.map(reims_vgpu_core::ResolvedIndirectCommandSlot::Render)
                    })
                    .map_err(|reason| ReplacementRenderIcbPopulationError::Draw {
                        index: slot.command_index,
                        reason,
                    })?
            }
        };
        resolved.push((slot.command_index, command));
    }
    runtime
        .execution_mut()
        .indirect_mut()
        .set_batch(read.identity, resolved.into_boxed_slice())
        .map_err(|failure| ReplacementRenderIcbPopulationError::Population(failure.reason))?;
    Ok(reims_vgpu_core::ResolvedIndirectCommand::Execute {
        icb: read.identity,
        range: read.range,
        kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
    })
}

fn resolve_render_icb_execution<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &ReplacementRecordProjectionState,
    command: &render::Command,
    read_memory: &mut impl FnMut(
        reims_vgpu_protocol::TaskId,
        reims_vgpu_core::IndirectCommandMemoryReadPlan,
    ) -> Result<Vec<u8>, ReplacementIcbCommandMemoryTransportError>,
) -> Result<reims_vgpu_core::ResolvedIndirectCommand, ReplacementRenderIcbExecutionError> {
    if !command.icb_is_range {
        return Err(ReplacementRenderIcbExecutionError::IndirectRangeRequiresAsynchronousReadback);
    }
    let reference = reims_vgpu_protocol::SerializerRef::new(command.indirect_command_buffer_ref);
    let range = reims_vgpu_core::ResolvedIndirectCommandRange {
        location: command.icb_range_location,
        length: command.icb_range_length,
    };
    let read = runtime
        .resolve_indirect_command_memory_read(task, reference, range)
        .map_err(ReplacementRenderIcbExecutionError::ReadPlan)?;
    let bytes = match read.plan {
        Some(plan) => {
            read_memory(task, plan).map_err(ReplacementRenderIcbExecutionError::Transport)?
        }
        None => Vec::new(),
    };
    let slots = runtime
        .decode_indirect_command_bytes(task, reference, &read, &bytes)
        .map_err(ReplacementRenderIcbExecutionError::Decode)?;
    populate_decoded_render_icb(runtime, task, state, &read, slots)
        .map_err(ReplacementRenderIcbExecutionError::Population)
}

pub(crate) fn project_replacement_record<Semantic: Clone, Completion>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &mut ReplacementRecordProjectionState,
    record: &DecodedReplacementRecord,
) -> Result<Box<[ProjectedReplacementOperation<Completion>]>, ReplacementOperationProjectionError> {
    project_replacement_record_with_icb_reader(runtime, task, state, record, &mut |_, _| {
        Err(ReplacementIcbCommandMemoryTransportError::Unavailable)
    })
}

pub(crate) fn project_replacement_record_with_icb_reader<Semantic: Clone, Completion>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    task: reims_vgpu_protocol::TaskId,
    state: &mut ReplacementRecordProjectionState,
    record: &DecodedReplacementRecord,
    read_memory: &mut impl FnMut(
        reims_vgpu_protocol::TaskId,
        reims_vgpu_core::IndirectCommandMemoryReadPlan,
    ) -> Result<Vec<u8>, ReplacementIcbCommandMemoryTransportError>,
) -> Result<Box<[ProjectedReplacementOperation<Completion>]>, ReplacementOperationProjectionError> {
    use reims_vgpu_core::ResolvedOperation;

    let operations = match record {
        DecodedReplacementRecord::Event(command) => vec![ResolvedOperation::Event(
            resolve_event_record(runtime, task, command)
                .map_err(ReplacementOperationProjectionError::Event)?,
        )],
        DecodedReplacementRecord::Render(command) => match command.kind {
            render::Kind::SetPipeline
            | render::Kind::SetBuffer
            | render::Kind::SetBufferOffset
            | render::Kind::SetTexture
            | render::Kind::SetSampler
            | render::Kind::SetViewport
            | render::Kind::SetScissor
            | render::Kind::SetBlendColor
            | render::Kind::SetCullMode
            | render::Kind::SetFrontFacing
            | render::Kind::SetDepthBias
            | render::Kind::SetDepthStencil
            | render::Kind::SetStencilReference
            | render::Kind::SetTessellationFactorBuffer
            | render::Kind::SetRasterState
            | render::Kind::SetFloatState
            | render::Kind::SetVertexAmplification
            | render::Kind::SetStoreAction
            | render::Kind::SetStoreActionOptions
            | render::Kind::SetVisibilityResultMode
            | render::Kind::RenderPassProperty
            | render::Kind::RenderPass => {
                resolve_render_state_record(state, command)
                    .map_err(ReplacementOperationProjectionError::RenderState)?;
                Vec::new()
            }
            render::Kind::Fence => resolve_render_fence_record(runtime, task, command)
                .map_err(ReplacementOperationProjectionError::Fence)?
                .into_iter()
                .map(ResolvedOperation::Fence)
                .collect(),
            render::Kind::Draw | render::Kind::DrawIndirect => {
                resolve_render_draw(runtime, task, state, command)
                    .map_err(ReplacementOperationProjectionError::RenderDraw)?
                    .into_iter()
                    .map(ResolvedOperation::Render)
                    .collect()
            }
            render::Kind::DrawPatches => {
                return Err(ReplacementOperationProjectionError::RenderTessellation(
                    Box::new(resolve_render_tessellation(state, command)),
                ));
            }
            render::Kind::TileBind
            | render::Kind::TileDispatch
            | render::Kind::TileDimensionsQuery => {
                resolve_render_tile(state, command).map_err(|reason| {
                    ReplacementOperationProjectionError::RenderTile(Box::new(reason))
                })?;
                Vec::new()
            }
            render::Kind::ExecuteCommands => vec![ResolvedOperation::IndirectCommand(
                resolve_render_icb_execution(runtime, task, state, command, read_memory)
                    .map_err(ReplacementOperationProjectionError::RenderIcb)?,
            )],
            render::Kind::UseResource | render::Kind::UseHeap => {
                resolve_render_participation_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::Participation)?
                    .into_vec()
                    .into_iter()
                    .map(ResolvedOperation::Participation)
                    .collect()
            }
            render::Kind::BarrierResources
            | render::Kind::BarrierScope
            | render::Kind::TextureBarrier => resolve_render_barrier_record(runtime, task, command)
                .map_err(ReplacementOperationProjectionError::Barrier)?
                .into_iter()
                .map(ResolvedOperation::Barrier)
                .collect(),
            kind => {
                return Err(ReplacementOperationProjectionError::RenderUnresolved {
                    kind,
                    opcode: command.opcode,
                })
            }
        },
        DecodedReplacementRecord::Compute(command) => match command.kind {
            compute::Kind::Pipeline
            | compute::Kind::BufferBind
            | compute::Kind::BufferOffset
            | compute::Kind::TextureBind
            | compute::Kind::SamplerBind
            | compute::Kind::SamplerLod
            | compute::Kind::StageInRegion
            | compute::Kind::StageInRegionIndirect
            | compute::Kind::ThreadgroupMemory
            | compute::Kind::ImageblockDimensions
            | compute::Kind::BufferBindAttributeStride
            | compute::Kind::BufferOffsetAttributeStride
            | compute::Kind::DispatchType
            | compute::Kind::CompressedTextureFlush => {
                resolve_compute_state_record(state, command)
                    .map_err(ReplacementOperationProjectionError::ComputeState)?;
                Vec::new()
            }
            compute::Kind::DispatchThreadgroups
            | compute::Kind::DispatchThreads
            | compute::Kind::DispatchThreadgroupsIndirect
            | compute::Kind::DispatchThreadsIndirect => {
                let accum = state.compute.as_ref().ok_or(
                    ReplacementOperationProjectionError::ComputeState(
                        ReplacementComputeStateResolutionError::OutsideEncoder,
                    ),
                )?;
                vec![ResolvedOperation::Compute(
                    resolve_direct_compute_dispatch(runtime, task, accum, command)
                        .map_err(ReplacementOperationProjectionError::ComputeDispatch)?,
                )]
            }
            compute::Kind::ExecuteCommandsInBuffer
            | compute::Kind::ExecuteCommandsInBufferIndirect => {
                vec![ResolvedOperation::IndirectCommand(
                    resolve_compute_icb_execution(runtime, task, state, command, read_memory)
                        .map_err(ReplacementOperationProjectionError::ComputeIcb)?,
                )]
            }
            compute::Kind::UpdateFence | compute::Kind::WaitFence => {
                resolve_compute_fence_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::Fence)?
                    .into_iter()
                    .map(ResolvedOperation::Fence)
                    .collect()
            }
            compute::Kind::UseResources | compute::Kind::UseHeaps => {
                resolve_compute_participation_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::Participation)?
                    .into_vec()
                    .into_iter()
                    .map(ResolvedOperation::Participation)
                    .collect()
            }
            compute::Kind::BarrierResources | compute::Kind::BarrierScope => {
                resolve_compute_barrier_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::Barrier)?
                    .into_iter()
                    .map(ResolvedOperation::Barrier)
                    .collect()
            }
            compute::Kind::ControlStartDoWhile
            | compute::Kind::ControlEndDoWhile
            | compute::Kind::ControlStartWhile
            | compute::Kind::ControlEndWhile
            | compute::Kind::ControlStartIf
            | compute::Kind::ControlStartElse
            | compute::Kind::ControlEndIf => {
                return Err(ReplacementOperationProjectionError::ComputeControlFlow(
                    Box::new(resolve_compute_control_flow(runtime, task, state, command)),
                ));
            }
            kind => {
                return Err(ReplacementOperationProjectionError::ComputeUnresolved {
                    kind,
                    opcode: command.opcode,
                })
            }
        },
        DecodedReplacementRecord::Blit(command) => match command.kind {
            blit::Kind::Fence => resolve_blit_fence_record(runtime, task, command)
                .map_err(ReplacementOperationProjectionError::Fence)?
                .into_iter()
                .map(ResolvedOperation::Fence)
                .collect(),
            blit::Kind::FillBuffer | blit::Kind::FillBufferPattern4 => {
                resolve_buffer_blit_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::BufferBlit)?
                    .into_iter()
                    .map(|operation| ResolvedOperation::Blit(Box::new(operation)))
                    .collect()
            }
            blit::Kind::Copy if command.copy_kind == blit::CopyKind::BufferToBuffer => {
                resolve_buffer_blit_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::BufferBlit)?
                    .into_iter()
                    .map(|operation| ResolvedOperation::Blit(Box::new(operation)))
                    .collect()
            }
            blit::Kind::Copy => resolve_image_blit_record(runtime, task, command)
                .map_err(ReplacementOperationProjectionError::ImageBlit)?
                .into_iter()
                .map(|operation| ResolvedOperation::Blit(Box::new(operation)))
                .collect(),
            blit::Kind::IcbRange | blit::Kind::IcbCopy => vec![ResolvedOperation::IndirectCommand(
                resolve_indirect_command_record(runtime, task, command)
                    .map_err(ReplacementOperationProjectionError::IndirectCommand)?,
            )],
            blit::Kind::Resource
                if matches!(
                    command.opcode,
                    reims_vgpu_wire::ops::blit::OPCODE_OPTIMIZE_FOR_CPU
                        | reims_vgpu_wire::ops::blit::OPCODE_OPTIMIZE_FOR_GPU
                ) =>
            {
                Vec::new()
            }
            blit::Kind::Image
                if matches!(
                    command.opcode,
                    reims_vgpu_wire::ops::blit::OPCODE_OPTIMIZE_FOR_CPU_SLICE_LEVEL
                        | reims_vgpu_wire::ops::blit::OPCODE_OPTIMIZE_FOR_GPU_SLICE_LEVEL
                ) =>
            {
                Vec::new()
            }
            blit::Kind::InvalidateCompressedTexture => Vec::new(),
            blit::Kind::Resource | blit::Kind::Image | blit::Kind::FillTexture => {
                return Err(ReplacementOperationProjectionError::TextureBlit(Box::new(
                    resolve_texture_blit(command),
                )));
            }
            kind => {
                return Err(ReplacementOperationProjectionError::BlitUnresolved {
                    kind,
                    opcode: command.opcode,
                })
            }
        },
        DecodedReplacementRecord::Info(operation) => {
            vec![ResolvedOperation::InfoQuery(
                reims_vgpu_core::resolve_info_operation(task, operation, runtime)
                    .map_err(ReplacementOperationProjectionError::Info)?,
            )]
        }
    };
    Ok(operations.into_boxed_slice())
}

/// Decode the complete counted child-stream set as one structural operation.
///
/// Directional continuation is submission state, so `open_type` deliberately
/// spans child-stream boundaries. Protection-option envelopes are validated by
/// the stream framer but do not fabricate an encoder or semantic operation.
pub(crate) fn decode_replacement_exec_streams(
    streams: &[Vec<u8>],
) -> Result<Box<[DecodedReplacementStream]>, ReplacementExecDecodeRefusal> {
    let mut decoded_streams = Vec::with_capacity(streams.len());
    let mut open_type = None;

    for (stream_index, bytes) in (0u32..).zip(streams) {
        let segments = stream::iter_segments(bytes).map_err(|reason| {
            ReplacementExecDecodeRefusal::Stream {
                stream_index,
                reason,
            }
        })?;
        let mut decoded_segments = Vec::new();

        for segment in segments {
            match stream::segment_disposition(segment.type_) {
                stream::SegmentDisposition::Envelope => continue,
                stream::SegmentDisposition::Unknown => {
                    return Err(ReplacementExecDecodeRefusal::UnknownSegment {
                        stream_index,
                        segment_index: segment.index,
                        type_: segment.type_,
                    });
                }
                stream::SegmentDisposition::Walk => {}
            }

            let boundary = semantic_segment_boundary(stream_index, &segment)
                .expect("the walk disposition consists exactly of semantic encoder families");
            if segment.continues_previous {
                match open_type.take() {
                    None => {
                        return Err(ReplacementExecDecodeRefusal::ContinuationWithoutPrevious {
                            boundary,
                        });
                    }
                    Some(previous_type) if previous_type != segment.type_ => {
                        return Err(ReplacementExecDecodeRefusal::ContinuationTypeMismatch {
                            boundary,
                            previous_type,
                        });
                    }
                    Some(_) => {}
                }
            } else if let Some(previous_type) = open_type.take() {
                return Err(ReplacementExecDecodeRefusal::RestartBeforeClose {
                    boundary,
                    previous_type,
                });
            }

            let records = decode_segment_records(bytes, &segment, boundary)?;
            if segment.continues_next {
                open_type = Some(segment.type_);
            }
            decoded_segments.push(DecodedReplacementSegment { boundary, records });
        }

        decoded_streams.push(DecodedReplacementStream {
            stream_index,
            segments: decoded_segments.into_boxed_slice(),
        });
    }

    if let Some(previous_type) = open_type {
        return Err(ReplacementExecDecodeRefusal::UnclosedEncoder { previous_type });
    }
    Ok(decoded_streams.into_boxed_slice())
}

/// Project an already complete decoded stream set into immutable semantic
/// segments without losing the decoded input on a resolution refusal.
///
/// Encoder begin/end operations derive solely from the directional segment
/// flags. A continuation segment therefore does not fabricate a second begin,
/// and only the segment that closes the encoder receives its end operation.
pub(crate) fn project_replacement_exec_streams<Operation, Error>(
    streams: Box<[DecodedReplacementStream]>,
    mut boundary_operation: impl FnMut(EncoderBoundary) -> Operation,
    mut resolve_record: impl FnMut(
        SegmentBoundary,
        &DecodedReplacementRecord,
    ) -> Result<Box<[Operation]>, Error>,
) -> Result<Box<[ResolvedExecStream<Operation>]>, ReplacementExecProjectionFailure<Error>> {
    let projected = project_replacement_exec_streams_ref(
        &streams,
        &mut boundary_operation,
        &mut resolve_record,
    );
    match projected {
        Ok(projected) => Ok(projected),
        Err(reason) => Err(ReplacementExecProjectionFailure { reason, streams }),
    }
}

fn project_replacement_exec_streams_ref<Operation, Error>(
    streams: &[DecodedReplacementStream],
    boundary_operation: &mut impl FnMut(EncoderBoundary) -> Operation,
    resolve_record: &mut impl FnMut(
        SegmentBoundary,
        &DecodedReplacementRecord,
    ) -> Result<Box<[Operation]>, Error>,
) -> Result<Box<[ResolvedExecStream<Operation>]>, Error> {
    streams
        .iter()
        .map(|stream| {
            let segments = stream
                .segments
                .iter()
                .map(|segment| {
                    let mut operations = Vec::new();
                    if !segment.boundary.continues_previous {
                        operations.push(boundary_operation(EncoderBoundary::Begin(
                            segment.boundary.kind,
                        )));
                    }
                    for record in &segment.records {
                        operations.extend(resolve_record(segment.boundary, record)?.into_vec());
                    }
                    if !segment.boundary.continues_next {
                        operations.push(boundary_operation(EncoderBoundary::End(
                            segment.boundary.kind,
                        )));
                    }
                    Ok(ResolvedExecSegment {
                        boundary: segment.boundary,
                        operations: operations.into_boxed_slice(),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(ResolvedExecStream {
                stream_index: stream.stream_index,
                segments: segments.into_boxed_slice(),
            })
        })
        .collect::<Result<Vec<_>, Error>>()
        .map(Vec::into_boxed_slice)
}

fn decode_segment_records(
    bytes: &[u8],
    segment: &stream::Segment,
    boundary: SegmentBoundary,
) -> Result<Box<[DecodedReplacementRecord]>, ReplacementExecDecodeRefusal> {
    let mut records = Vec::new();
    let mut cursor = segment.command_offset as usize;
    loop {
        let record = match stream::decode_next_record(bytes, segment, &mut cursor) {
            Ok(record) => record,
            Err(stream::DecodeStatus::Done) => break,
            Err(reason) => {
                return Err(ReplacementExecDecodeRefusal::Stream {
                    stream_index: boundary.stream_index,
                    reason,
                });
            }
        };
        let start = record.bytes_offset as usize;
        let end = start + record.length as usize;
        let record_bytes = &bytes[start..end];
        let decoded = match segment.type_ {
            stream::SEGMENT_TYPE_RENDER => render::decode(record_bytes)
                .map(Box::new)
                .map(DecodedReplacementRecord::Render)
                .map_err(ReplacementRecordDecodeRefusal::Render),
            stream::SEGMENT_TYPE_COMPUTE => compute::decode(record_bytes)
                .map(Box::new)
                .map(DecodedReplacementRecord::Compute)
                .map_err(ReplacementRecordDecodeRefusal::Compute),
            stream::SEGMENT_TYPE_BLIT => blit::decode(record_bytes)
                .map(Box::new)
                .map(DecodedReplacementRecord::Blit)
                .map_err(ReplacementRecordDecodeRefusal::Blit),
            stream::SEGMENT_TYPE_EVENT => event::decode(record_bytes)
                .map(DecodedReplacementRecord::Event)
                .map_err(ReplacementRecordDecodeRefusal::Event),
            stream::SEGMENT_TYPE_INFO => classify_info_record(record.opcode, record_bytes)
                .map(DecodedReplacementRecord::Info)
                .map_err(ReplacementRecordDecodeRefusal::Info),
            _ => unreachable!("only walkable segment families reach record decoding"),
        }
        .map_err(|reason| ReplacementExecDecodeRefusal::Record {
            boundary,
            opcode: record.opcode,
            offset: record.offset,
            reason,
        })?;
        records.push(decoded);
    }
    Ok(records.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::{st16, st32};
    use reims_vgpu_wire::ops::render as wire_render;

    #[test]
    fn an_encoder_read_of_an_attachment_marks_every_draw_in_it() {
        use reims_vgpu_core::{
            AccessMode, BackingRegion, PrimitiveTopology, RenderAttachmentClear,
            RenderAttachmentRole, RenderBindingClass, RenderBindingView, ResolvedExecSegment,
            ResolvedExecStream, ResolvedRenderAttachment, ResolvedRenderDraw,
            ResolvedRenderRasterState, ResolvedRenderResourceBinding, ResolvedTextureBindingView,
            ResolvedTextureViewRange,
        };
        use reims_vgpu_protocol::{
            BackingId, LoadAction, RenderStages, ResourceId, SegmentKind, StoreAction, TextureType,
        };

        let attachment_backing = BackingId::new(7);
        let attachment_resource = ResourceId::new(4, 1);
        let draw = |samples_attachment: bool, reads_input: bool| {
            reims_vgpu_core::ResolvedOperation::Render(reims_vgpu_core::ResolvedRenderDispatch {
                pipeline: ResourceId::new(2, 1),
                program: Default::default(),
                depth_stencil: None,
                render_extent: [32, 16],
                raster: ResolvedRenderRasterState::default(),
                visibility: None,
                begins_encoder: false,
                ends_encoder: false,
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
                    resource: attachment_resource,
                    image_owner: attachment_resource,
                    backing: attachment_backing,
                    regions: Box::new([BackingRegion::Whole]),
                    pixel_format: 80,
                    extent: [32, 16, 1],
                    sample_count: 1,
                    load: LoadAction::Clear,
                    store: StoreAction::Store,
                    clear: RenderAttachmentClear::Color([0; 4]),
                    resolve: None,
                    feedback_loop: false,
                    input_attachment: reads_input,
                }]),
                resources: if samples_attachment {
                    Box::new([ResolvedRenderResourceBinding {
                        class: RenderBindingClass::SampledImage,
                        binding: 3,
                        array_element: 0,
                        descriptor_count: 1,
                        stages: RenderStages::from_bits(RenderStages::FRAGMENT.into()).unwrap(),
                        resource: attachment_resource,
                        backing: attachment_backing,
                        view: RenderBindingView::Image(ResolvedTextureBindingView {
                            resource: attachment_resource,
                            base: attachment_resource,
                            image_owner: attachment_resource,
                            range: ResolvedTextureViewRange {
                                level_base: 0,
                                level_count: 1,
                                slice_base: 0,
                                slice_count: 1,
                            },
                            texture_type: TextureType::D2,
                            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                            swizzle: reims_vgpu_protocol::swizzle_identity(),
                        }),
                        regions: Box::new([BackingRegion::Whole]),
                        mode: AccessMode::Read,
                    }])
                } else {
                    Box::new([])
                },
                null_bindings: Box::new([]),
                samplers: Box::new([]),
            })
        };
        let mut streams = [ResolvedExecStream::<ProjectedReplacementOperation<()>> {
            stream_index: 0,
            segments: Box::new([ResolvedExecSegment {
                boundary: reims_vgpu_protocol::SegmentBoundary {
                    stream_index: 0,
                    index: 0,
                    kind: SegmentKind::Render,
                    continues_previous: false,
                    continues_next: false,
                },
                operations: Box::new([
                    reims_vgpu_core::ResolvedOperation::EncoderBoundary(
                        reims_vgpu_core::EncoderBoundary::Begin(SegmentKind::Render),
                    ),
                    draw(false, true),
                    draw(true, false),
                    reims_vgpu_core::ResolvedOperation::EncoderBoundary(
                        reims_vgpu_core::EncoderBoundary::End(SegmentKind::Render),
                    ),
                ]),
            }]),
        }];

        mark_render_dispatch_encoder_boundaries(&mut streams);

        // Only the second draw samples the attachment and only the first
        // reads it as a colour input, but one native render pass covers both,
        // so both must describe the same layout.
        for index in [1, 2] {
            let reims_vgpu_core::ResolvedOperation::Render(operation) =
                &streams[0].segments[0].operations[index]
            else {
                panic!("operation {index} must remain a render dispatch")
            };
            assert!(
                operation.attachments[0].feedback_loop,
                "draw {index} lost its encoder's feedback loop"
            );
            // Only the first draw's fragment stage declares the colour input,
            // and the same pass covers both, so both name the layout it forces.
            assert!(
                operation.attachments[0].input_attachment,
                "draw {index} lost its encoder's colour input read"
            );
        }
    }

    fn segment(
        type_: u8,
        records: &[u8],
        continues_previous: bool,
        continues_next: bool,
    ) -> Vec<u8> {
        let mut bytes = vec![0; stream::SEGMENT_HEADER_LEN];
        st32(
            &mut bytes[0..4],
            (stream::SEGMENT_HEADER_LEN + records.len()) as u32,
        );
        bytes[4] = type_;
        bytes[5] = u8::from(continues_previous);
        bytes[6] = u8::from(continues_next);
        bytes.extend_from_slice(records);
        bytes
    }

    fn render_draw() -> Vec<u8> {
        let mut draw = vec![0; wire_render::DRAW_TOTAL_LEN as usize];
        st32(&mut draw[0..4], wire_render::OPCODE_DRAW);
        st32(&mut draw[4..8], wire_render::DRAW_TOTAL_LEN);
        st32(&mut draw[8..12], 3);
        st16(&mut draw[12..14], 0);
        st16(&mut draw[14..16], 3);
        draw
    }

    #[test]
    fn complete_decode_preserves_records_and_cross_stream_continuation() {
        let draw = render_draw();
        let streams = vec![
            segment(stream::SEGMENT_TYPE_RENDER, &draw, false, true),
            segment(stream::SEGMENT_TYPE_RENDER, &draw, true, false),
        ];
        let decoded = decode_replacement_exec_streams(&streams).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].stream_index, 0);
        assert_eq!(decoded[1].stream_index, 1);
        assert_eq!(decoded[0].segments[0].records.len(), 1);
        assert_eq!(decoded[1].segments[0].records.len(), 1);
        assert!(decoded[0].segments[0].boundary.continues_next);
        assert!(decoded[1].segments[0].boundary.continues_previous);
        assert!(matches!(
            decoded[1].segments[0].records[0],
            DecodedReplacementRecord::Render(ref command)
                if matches!(command.as_ref(), render::Command {
                kind: render::Kind::Draw,
                ..
            })
        ));
    }

    #[test]
    fn malformed_late_record_returns_no_partial_stream_set() {
        let mut records = render_draw();
        let mut malformed = vec![0; reims_vgpu_wire::OP_HEADER_LEN];
        st32(&mut malformed[0..4], wire_render::OPCODE_DRAW);
        st32(&mut malformed[4..8], reims_vgpu_wire::OP_HEADER_LEN as u32);
        records.extend_from_slice(&malformed);
        let streams = vec![segment(stream::SEGMENT_TYPE_RENDER, &records, false, false)];

        assert!(matches!(
            decode_replacement_exec_streams(&streams),
            Err(ReplacementExecDecodeRefusal::Record {
                reason: ReplacementRecordDecodeRefusal::Render(_),
                ..
            })
        ));
    }

    #[test]
    fn continuation_type_change_is_a_structural_refusal() {
        let streams = vec![
            segment(stream::SEGMENT_TYPE_RENDER, &[], false, true),
            segment(stream::SEGMENT_TYPE_COMPUTE, &[], true, false),
        ];
        assert!(matches!(
            decode_replacement_exec_streams(&streams),
            Err(ReplacementExecDecodeRefusal::ContinuationTypeMismatch {
                previous_type: stream::SEGMENT_TYPE_RENDER,
                ..
            })
        ));
    }

    #[test]
    fn compute_projection_state_follows_encoder_lifetime_and_retains_continuation_deltas() {
        let mut state = ReplacementRecordProjectionState::default();
        let compute_kind = reims_vgpu_protocol::SegmentKind::Compute;
        state.encoder_boundary(EncoderBoundary::Begin(compute_kind));
        resolve_compute_state_record(
            &mut state,
            &compute::Command {
                kind: compute::Kind::Pipeline,
                pipeline_ref: 41,
                ..Default::default()
            },
        )
        .unwrap();
        resolve_compute_state_record(
            &mut state,
            &compute::Command {
                kind: compute::Kind::BufferBindAttributeStride,
                first: 3,
                buffers: vec![compute::BufferBinding {
                    ref_: 17,
                    offset: 64,
                    attribute_stride: 12,
                    has_attribute_stride: true,
                }],
                ..Default::default()
            },
        )
        .unwrap();

        // A continuation has no second Begin boundary. Its delta therefore
        // updates the state established by the preceding segment.
        resolve_compute_state_record(
            &mut state,
            &compute::Command {
                kind: compute::Kind::BufferOffset,
                first: 3,
                buffer_offset: 96,
                ..Default::default()
            },
        )
        .unwrap();
        let accum = state.compute.as_ref().unwrap();
        assert_eq!(accum.pipeline_ref, 41);
        assert_eq!(accum.buffers.len(), 1);
        assert_eq!(accum.buffers[0].index, 3);
        assert_eq!(accum.buffers[0].buffer_ref, 17);
        assert_eq!(accum.buffers[0].offset, 96);
        assert_eq!(accum.buffers[0].attribute_stride, 12);

        let error = resolve_compute_state_record(
            &mut state,
            &compute::Command {
                kind: compute::Kind::TextureBind,
                first: crate::runtime::replacement_compute_state::MAX_COMPUTE_TEXTURE_SLOTS,
                textures: vec![compute::RefBinding { ref_: 99 }],
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            ReplacementComputeStateResolutionError::TextureIndexOverflow {
                first: crate::runtime::replacement_compute_state::MAX_COMPUTE_TEXTURE_SLOTS,
                count: 1,
                limit: crate::runtime::replacement_compute_state::MAX_COMPUTE_TEXTURE_SLOTS,
            }
        );
        assert!(state.compute.as_ref().unwrap().textures.is_empty());

        assert_eq!(
            resolve_compute_state_record(
                &mut state,
                &compute::Command {
                    kind: compute::Kind::TextureBind,
                    first: crate::runtime::replacement_compute_state::MAX_COMPUTE_TEXTURE_SLOTS,
                    textures: vec![compute::RefBinding { ref_: 0 }],
                    ..Default::default()
                },
            ),
            Err(
                ReplacementComputeStateResolutionError::TextureIndexOverflow {
                    first: crate::runtime::replacement_compute_state::MAX_COMPUTE_TEXTURE_SLOTS,
                    count: 1,
                    limit: crate::runtime::replacement_compute_state::MAX_COMPUTE_TEXTURE_SLOTS,
                }
            )
        );
        assert_eq!(
            resolve_compute_state_record(
                &mut state,
                &compute::Command {
                    kind: compute::Kind::SamplerBind,
                    first: crate::runtime::replacement_compute_state::MAX_COMPUTE_SAMPLER_SLOTS - 1,
                    samplers: vec![compute::SamplerBinding::default(); 2],
                    ..Default::default()
                },
            ),
            Err(
                ReplacementComputeStateResolutionError::SamplerIndexOverflow {
                    first: crate::runtime::replacement_compute_state::MAX_COMPUTE_SAMPLER_SLOTS - 1,
                    count: 2,
                    limit: crate::runtime::replacement_compute_state::MAX_COMPUTE_SAMPLER_SLOTS,
                }
            )
        );
        assert_eq!(
            resolve_compute_state_record(
                &mut state,
                &compute::Command {
                    kind: compute::Kind::BufferOffset,
                    first: 4,
                    buffer_offset: 128,
                    ..Default::default()
                },
            ),
            Err(ReplacementComputeStateResolutionError::BufferOffsetUnbound { index: 4 })
        );
        resolve_compute_state_record(
            &mut state,
            &compute::Command {
                kind: compute::Kind::Pipeline,
                pipeline_ref: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(state.compute.as_ref().unwrap().pipeline_ref, 0);

        state.encoder_boundary(EncoderBoundary::End(compute_kind));
        assert_eq!(
            resolve_compute_state_record(
                &mut state,
                &compute::Command {
                    kind: compute::Kind::Pipeline,
                    pipeline_ref: 42,
                    ..Default::default()
                },
            ),
            Err(ReplacementComputeStateResolutionError::OutsideEncoder)
        );
    }

    #[test]
    fn vertex_amplification_accepts_only_the_exact_single_view_default() {
        let mut state = ReplacementRecordProjectionState::default();
        state.encoder_boundary(EncoderBoundary::Begin(
            reims_vgpu_protocol::SegmentKind::Render,
        ));
        resolve_render_state_record(
            &mut state,
            &render::Command {
                kind: render::Kind::SetVertexAmplification,
                opcode: wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
                ..Default::default()
            },
        )
        .unwrap();
        resolve_render_state_record(
            &mut state,
            &render::Command {
                kind: render::Kind::SetVertexAmplification,
                opcode: wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
                count: 1,
                amplification_mappings: vec![[0, 0]],
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            resolve_render_state_record(
                &mut state,
                &render::Command {
                    kind: render::Kind::SetVertexAmplification,
                    opcode: wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
                    count: 1,
                    amplification_mappings: vec![[0, 2]],
                    ..Default::default()
                },
            ),
            Err(
                ReplacementRenderStateResolutionError::VertexAmplificationMappingUnsupported {
                    index: 0,
                    viewport_offset: 0,
                    render_target_offset: 2,
                }
            )
        );
        assert_eq!(
            resolve_render_state_record(
                &mut state,
                &render::Command {
                    kind: render::Kind::SetVertexAmplification,
                    opcode: wire_render::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
                    mode: 3,
                    amplification_value: 4,
                    ..Default::default()
                },
            ),
            Err(
                ReplacementRenderStateResolutionError::VertexAmplificationUnsupported {
                    count: 1,
                    mode: 3,
                    value: 4,
                }
            )
        );
    }

    #[test]
    fn tessellation_scale_accepts_only_the_contract_default() {
        let mut state = ReplacementRecordProjectionState::default();
        state.encoder_boundary(EncoderBoundary::Begin(
            reims_vgpu_protocol::SegmentKind::Render,
        ));
        resolve_render_state_record(
            &mut state,
            &render::Command {
                kind: render::Kind::SetFloatState,
                opcode: wire_render::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
                float_value: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            resolve_render_state_record(
                &mut state,
                &render::Command {
                    kind: render::Kind::SetFloatState,
                    opcode: wire_render::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
                    float_value: 2.5,
                    ..Default::default()
                },
            ),
            Err(
                ReplacementRenderStateResolutionError::TessellationFactorScaleUnsupported {
                    bits: 2.5f32.to_bits(),
                }
            )
        );
    }

    #[test]
    fn render_pass_property_accepts_only_the_established_sample_default() {
        let mut state = ReplacementRecordProjectionState::default();
        state.encoder_boundary(EncoderBoundary::Begin(
            reims_vgpu_protocol::SegmentKind::Render,
        ));
        resolve_render_state_record(
            &mut state,
            &render::Command {
                kind: render::Kind::RenderPassProperty,
                opcode: reims_vgpu_wire::ops::render_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
                mode: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            resolve_render_state_record(
                &mut state,
                &render::Command {
                    kind: render::Kind::RenderPassProperty,
                    opcode: reims_vgpu_wire::ops::render_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
                    mode: 4,
                    ..Default::default()
                },
            ),
            Err(
                ReplacementRenderStateResolutionError::DefaultRasterSampleCountUnsupported {
                    count: 4,
                }
            )
        );
        assert_eq!(
            resolve_render_state_record(
                &mut state,
                &render::Command {
                    kind: render::Kind::RenderPassProperty,
                    opcode: reims_vgpu_wire::ops::render_pass::OPCODE_RASTERIZATION_RATE_MAP,
                    texture_ref: 17,
                    ..Default::default()
                },
            ),
            Err(
                ReplacementRenderStateResolutionError::RenderPassPropertyUnsupported {
                    opcode: reims_vgpu_wire::ops::render_pass::OPCODE_RASTERIZATION_RATE_MAP,
                    value: 0,
                    reference: 17,
                    count: 0,
                }
            )
        );
    }

    #[derive(Clone, Debug, PartialEq)]
    enum Projected {
        Boundary(EncoderBoundary),
        Render(u32),
    }

    #[test]
    fn projection_places_one_begin_and_end_around_a_continued_encoder() {
        let draw = render_draw();
        let decoded = decode_replacement_exec_streams(&[
            segment(stream::SEGMENT_TYPE_RENDER, &draw, false, true),
            segment(stream::SEGMENT_TYPE_RENDER, &draw, true, false),
        ])
        .unwrap();
        let projected =
            project_replacement_exec_streams(decoded, Projected::Boundary, |_boundary, record| {
                match record {
                    DecodedReplacementRecord::Render(command) => {
                        Ok::<_, ()>(Box::new([Projected::Render(command.opcode)]))
                    }
                    _ => unreachable!(),
                }
            })
            .unwrap();

        assert_eq!(
            projected[0].segments[0].operations.as_ref(),
            [
                Projected::Boundary(EncoderBoundary::Begin(
                    reims_vgpu_protocol::SegmentKind::Render,
                )),
                Projected::Render(wire_render::OPCODE_DRAW),
            ]
        );
        assert_eq!(
            projected[1].segments[0].operations.as_ref(),
            [
                Projected::Render(wire_render::OPCODE_DRAW),
                Projected::Boundary(EncoderBoundary::End(
                    reims_vgpu_protocol::SegmentKind::Render,
                )),
            ]
        );
    }

    #[test]
    fn projection_refusal_returns_the_complete_decoded_input() {
        let draw = render_draw();
        let decoded = decode_replacement_exec_streams(&[segment(
            stream::SEGMENT_TYPE_RENDER,
            &draw,
            false,
            false,
        )])
        .unwrap();
        let failure =
            project_replacement_exec_streams(decoded, Projected::Boundary, |_boundary, _record| {
                Err::<Box<[Projected]>, _>("not-ready")
            })
            .unwrap_err();
        assert_eq!(failure.reason, "not-ready");
        assert_eq!(failure.streams.len(), 1);
        assert_eq!(failure.streams[0].segments[0].records.len(), 1);
    }

    #[test]
    fn stream_object_manifest_keeps_object_slots_and_excludes_other_namespaces() {
        let render = render::Command {
            pipeline_ref: 7,
            buffer_ref: 9,
            ref_binds: vec![11, 0, 9],
            fence_ref: 13,
            participation_heaps: vec![reims_vgpu_protocol::SerializerRef::new(15)],
            ..Default::default()
        };
        let compute = compute::Command {
            pipeline_ref: 7,
            textures: vec![compute::RefBinding { ref_: 12 }],
            fence_ref: 14,
            heaps: vec![reims_vgpu_protocol::SerializerRef::new(16)],
            ..Default::default()
        };
        let blit = blit::Command {
            source: 9,
            destination: 17,
            fence: 18,
            ..Default::default()
        };
        let info = reims_vgpu_protocol::InfoOperation::SamplerHost {
            sampler: reims_vgpu_protocol::SerializerRef::new(19),
            reply: reims_vgpu_protocol::InfoReplyTarget {
                buffer: reims_vgpu_protocol::ObjectTableRef::new(20),
                offset: 0,
                length: 8,
                alignment: 8,
            },
        };
        let streams = [DecodedReplacementStream {
            stream_index: 0,
            segments: vec![DecodedReplacementSegment {
                boundary: reims_vgpu_protocol::SegmentBoundary {
                    stream_index: 0,
                    index: 0,
                    kind: reims_vgpu_protocol::SegmentKind::Render,
                    continues_previous: false,
                    continues_next: false,
                },
                records: vec![
                    DecodedReplacementRecord::Render(Box::new(render)),
                    DecodedReplacementRecord::Compute(Box::new(compute)),
                    DecodedReplacementRecord::Blit(Box::new(blit)),
                    DecodedReplacementRecord::Info(info),
                ]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
        }];

        assert_eq!(
            replacement_stream_object_manifest(&streams)
                .objects
                .iter()
                .map(|object| object.get())
                .collect::<Vec<_>>(),
            vec![7, 9, 11, 12, 17, 19, 20]
        );
    }
}
