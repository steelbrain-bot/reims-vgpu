//! Final immutable render-dispatch assembly for replacement EXEC decoding.

use reims_vgpu_core::{
    PreparedRenderProgram, ResolvedRenderAttachment, ResolvedRenderDispatch, ResolvedRenderDraw,
    ResolvedRenderNullBinding, ResolvedRenderRasterState, ResolvedRenderResourceBinding,
    ResolvedRenderSamplerBinding, ResolvedRenderVisibility, ResolvedVertexBufferLayout,
};
use reims_vgpu_protocol::{DepthStencilObject, RenderPipelineObject, ResourceId};

pub(crate) struct ResolvedRenderConstructionInput {
    pub pipeline: ResourceId<RenderPipelineObject>,
    pub program: PreparedRenderProgram,
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

pub(crate) fn construct(input: ResolvedRenderConstructionInput) -> ResolvedRenderDispatch {
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
