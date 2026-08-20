//! Backend-independent resolved render commands and typed results.

use crate::{ContentStamp, GatherVouch, ResourceLifetime, ResourceLifetimeRef, SamplerResource};
use reims_vgpu_memory::{GuestRunSource, GuestTargetPlan};
pub use reims_vgpu_protocol::{
    BlendFactor, BlendOp, BlendStateResource, CullMode, DepthClipMode, FillMode, IndexType,
    PrimitiveTopology, StencilOp, VertexAttributeFormat, VertexStepFunction, VisibilityResultMode,
};
use reims_vgpu_protocol::{ColorWriteMask, ImageFormat, SwizzlePlan};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct DepthState {
    pub identity: Option<crate::TargetIdentity>,
    pub test_enable: bool,
    pub write_enable: bool,
    pub compare: crate::SamplerCompareFunction,
    pub clear_value: f32,
    pub load: bool,
    pub stencil: Option<StencilState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceOps {
    pub compare: crate::SamplerCompareFunction,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StencilState {
    pub front: StencilFaceOps,
    pub back: StencilFaceOps,
    pub reference_front: u32,
    pub reference_back: u32,
    pub clear_value: u32,
}

/// One executor-prepared shader stage and the semantic descriptor interface
/// the resolved module statically uses.
#[derive(Clone, Debug, Default)]
pub struct PreparedShaderStage {
    pub id: reims_vgpu_protocol::PreparedShaderId,
    pub used_descriptor_bindings: Arc<[u32]>,
}

/// The two prepared stages required by one resolved render pipeline.
#[derive(Clone, Debug, Default)]
pub struct PreparedRenderProgram {
    pub vertex: PreparedShaderStage,
    pub fragment: PreparedShaderStage,
}

/// Fully resolved inputs for one draw.
///
/// Guest names, wire tags, and host-native handles are absent. Resource
/// identities are generational core values; formats retain semantic layout and
/// transfer; guest-memory operands are bounded memory contracts.
#[derive(Debug, Default)]
pub struct DrawRequest {
    pub pipeline_lifetime: Option<ResourceLifetime>,
    pub program: PreparedRenderProgram,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub first_vertex: u32,
    pub instance_count: Option<u32>,
    pub base_instance: u32,
    pub primitive_topology: PrimitiveTopology,
    pub raster_sample_count: u32,
    pub color_sample_count: u32,
    pub multisample_resolve: bool,
    pub viewports: Vec<ViewportResource>,
    pub scissors: Vec<ScissorResource>,
    pub occlusion_query: Option<VisibilityResultMode>,
    pub indexed: Option<IndexedDrawResource>,
    pub vertex_attributes: Vec<VertexAttributeResource>,
    pub storage_buffers: Vec<StorageBufferResource>,
    pub sampled_images: Vec<SampledImageResource>,
    pub samplers: Vec<SamplerResource>,
    pub target_rgba8: Option<Arc<Vec<u8>>>,
    pub target_guest: Option<GuestTargetPlan>,
    pub target_seed_order: SeedOrder,
    pub blend: Option<BlendStateResource>,
    pub color_write_mask: ColorWriteMask,
    pub target_identity: Option<crate::TargetIdentity>,
    pub color_attachment_format: Option<ImageFormat>,
    /// Decoded color-attachment load operation. Content placement and seed
    /// availability do not change this contract term.
    pub color_load_action: ColorLoadAction,
    pub load_from_target: bool,
    pub target_clear: [f32; 4],
    pub skip_readback: bool,
    pub seed_from_target: Option<crate::TargetIdentity>,
    pub secondary_targets: Vec<SecondaryColorTarget>,
    pub cull_mode: CullMode,
    pub front_face_ccw: bool,
    pub fill_mode: FillMode,
    pub depth_clip: DepthClipMode,
    pub depth: Option<DepthState>,
    pub color_input: bool,
    pub continues_render_pass: bool,
    pub render_pass_continues: bool,
}

impl DrawRequest {
    pub fn writes_attachment(&self, identity: &crate::TargetIdentity) -> bool {
        self.attachment_slot(identity).is_some()
    }

    pub fn color_attachment_index(&self, identity: &crate::TargetIdentity) -> Option<usize> {
        if self.target_identity.as_ref() == Some(identity) {
            Some(0)
        } else {
            self.secondary_targets
                .iter()
                .position(|target| &target.identity == identity)
                .map(|index| index + 1)
        }
    }

    pub fn attachment_slot(&self, identity: &crate::TargetIdentity) -> Option<AttachmentSlot> {
        if let Some(index) = self.color_attachment_index(identity) {
            Some(if index == 0 {
                AttachmentSlot::Primary
            } else {
                AttachmentSlot::Secondary
            })
        } else if self.depth.as_ref().and_then(|d| d.identity.as_ref()) == Some(identity) {
            Some(AttachmentSlot::Depth)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentSlot {
    Primary,
    Secondary,
    Depth,
}

impl AttachmentSlot {
    pub fn sampled_self_route(self) -> &'static str {
        match self {
            Self::Primary => "sampled_self_primary",
            Self::Secondary => "sampled_self_secondary",
            Self::Depth => "sampled_self_depth",
        }
    }
}

pub fn viewport_slot_count(req: &DrawRequest) -> usize {
    req.viewports.len().max(req.scissors.len()).max(1)
}

#[derive(Debug, Clone)]
pub struct SecondaryColorTarget {
    pub identity: crate::TargetIdentity,
    /// Canonical guest allocation for this attachment when its declared
    /// layout is directly representable by the backend.
    pub target_guest: Option<reims_vgpu_memory::GuestTargetMemory>,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub clear: [f32; 4],
    pub load_action: ColorLoadAction,
    pub blend: Option<BlendStateResource>,
    pub color_write_mask: ColorWriteMask,
}

/// Backend-independent color-attachment load operation.
///
/// This stays distinct from whether a LOAD source has already been
/// materialized. In particular, `DontCare` is not a black clear.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ColorLoadAction {
    Load,
    #[default]
    Clear,
    DontCare,
}

#[derive(Debug, Default)]
pub struct DrawOutput {
    pub pixels: Vec<u8>,
    pub pixels_bgra: bool,
    pub occlusion_samples: Option<u64>,
    /// Exact guest pages whose directly bound attachment Store completed before
    /// this result was returned. `None` means this draw published no direct
    /// guest Store; the resident may still require its ordinary materialization.
    pub guest_store_pages: Option<reims_vgpu_memory::GuestWritePages>,
    /// Allocation-relative byte window occupied by that completed direct Store.
    /// Carried with the completion so surface publication never re-derives a
    /// possibly newer mapping layout after execution.
    pub guest_store_window: Option<std::ops::Range<u64>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewportResource {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScissorResource {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct IndexedDrawResource {
    pub index_type: IndexType,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub content: BufferContent,
}

#[derive(Debug)]
pub struct VertexAttributeResource {
    pub location: u32,
    pub binding: u32,
    pub format: VertexAttributeFormat,
    pub offset: u32,
    pub stride: u32,
    pub step_function: VertexStepFunction,
    pub step_rate: u32,
    pub content: BufferContent,
}

#[derive(Debug)]
pub struct StorageBufferResource {
    pub binding: u32,
    pub content: BufferContent,
}

#[derive(Clone, Debug)]
pub enum BufferContent {
    Bytes(Arc<Vec<u8>>),
    GuestRuns(GuestRunSource),
}

impl BufferContent {
    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::GuestRuns(source) => source.total_len as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Test/instrumentation view of a potentially scattered guest source.
    pub fn cpu_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Self::Bytes(bytes) => std::borrow::Cow::Borrowed(bytes.as_slice()),
            Self::GuestRuns(source) => {
                let mut out = Vec::with_capacity(source.total_len as usize);
                let mut skip = source.source_offset;
                for run in source.runs.iter() {
                    let take = (source.total_len as usize).saturating_sub(out.len());
                    if take == 0 {
                        break;
                    }
                    if skip >= run.len {
                        skip -= run.len;
                        continue;
                    }
                    let within = skip as usize;
                    skip = 0;
                    let len = (run.len as usize).saturating_sub(within).min(take);
                    // SAFETY: the memory contract retains each stable host alias
                    // for the request lifetime and bounds the declared span.
                    unsafe {
                        out.extend_from_slice(std::slice::from_raw_parts(
                            (run.host_ptr as *const u8).add(within),
                            len,
                        ));
                    }
                }
                out.resize(source.total_len as usize, 0);
                std::borrow::Cow::Owned(out)
            }
        }
    }
}

impl From<Vec<u8>> for BufferContent {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(Arc::new(bytes))
    }
}

#[derive(Debug)]
pub struct SampledImageResource {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub arrayed: bool,
    pub volume: bool,
    pub cube: bool,
    pub one_dim: bool,
    pub multisampled: bool,
    pub source: SampledSource,
    pub content: Option<ContentStamp>,
    pub byte_origin: SampledByteOrigin,
    pub format: ImageFormat,
    pub identity: Option<SampledContentIdentity>,
    pub resource_lifetime: Option<ResourceLifetimeRef>,
    pub swizzle: SwizzlePlan,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SampledByteOrigin {
    #[default]
    Synthetic,
    BufferBackedTexture,
    SerializedSurfaceView,
    SurfaceHostCache,
    SurfaceGuestFallback,
    LinearTexture,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SeedOrder {
    #[default]
    Rgba8,
    Bgra8,
}

#[derive(Debug)]
pub enum SampledSource {
    /// A serialized texture slot containing no object.
    Null,
    Bytes(Arc<Vec<u8>>),
    Target(crate::TargetIdentity),
    Attachment {
        identity: crate::TargetIdentity,
        initial: AttachmentInitial,
    },
    /// An image view over authoritative guest storage, with an exact transfer
    /// representation for backends whose image layout cannot alias it.
    GuestImage(reims_vgpu_memory::GuestImageSource, GatherVouch),
    GuestRuns(GuestRunSource, GatherVouch),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttachmentInitial {
    Clear([f32; 4]),
    Seed,
    DontCare,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampledContentIdentity {
    pub key: u64,
    pub generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::TexelLayout;

    #[test]
    fn viewport_and_attachment_relations_are_semantic() {
        let request = DrawRequest {
            target_identity: Some(crate::TargetIdentity::Gva {
                gva: 1,
                width: 4,
                height: 4,
                generation: 1,
                format: TexelLayout::Rgba8,
            }),
            ..Default::default()
        };
        assert_eq!(viewport_slot_count(&request), 1);
        assert_eq!(
            request.attachment_slot(request.target_identity.as_ref().unwrap()),
            Some(AttachmentSlot::Primary)
        );
    }
}
