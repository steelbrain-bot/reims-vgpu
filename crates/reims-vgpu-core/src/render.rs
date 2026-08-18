//! Backend-independent resolved render commands and typed results.

use crate::{ContentStamp, GatherVouch, ResourceLifetime, ResourceLifetimeRef, SamplerResource};
use reims_vgpu_memory::{GuestRunSource, GuestTargetSeed};
use reims_vgpu_protocol::{ColorWriteMask, ImageFormat, SwizzlePlan};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VisibilityResultMode {
    Boolean,
    Counting,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum FillMode {
    #[default]
    Fill,
    Lines,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DepthClipMode {
    #[default]
    Clip,
    Clamp,
}

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
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

/// Fully resolved inputs for one draw.
///
/// Guest names, wire tags, and host-native handles are absent. Resource
/// identities are generational core values; formats retain semantic layout and
/// transfer; guest-memory operands are bounded memory contracts.
#[derive(Debug, Default)]
pub struct DrawRequest {
    pub pipeline_lifetime: Option<ResourceLifetime>,
    pub vert_spirv: Arc<Vec<u32>>,
    pub frag_spirv: Arc<Vec<u32>>,
    pub vert_used_descriptor_bindings: Arc<[u32]>,
    pub frag_used_descriptor_bindings: Arc<[u32]>,
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
    pub target_guest_seed: Option<GuestTargetSeed>,
    pub target_seed_order: SeedOrder,
    pub blend: Option<BlendStateResource>,
    pub color_write_mask: ColorWriteMask,
    pub target_identity: Option<crate::TargetIdentity>,
    pub color_attachment_format: Option<ImageFormat>,
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
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub clear: [f32; 4],
    pub load: bool,
    pub blend: Option<BlendStateResource>,
    pub color_write_mask: ColorWriteMask,
}

#[derive(Debug, Default)]
pub struct DrawOutput {
    pub pixels: Vec<u8>,
    pub pixels_bgra: bool,
    pub occlusion_samples: Option<u64>,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PrimitiveTopology {
    Point,
    Line,
    LineStrip,
    #[default]
    Triangle,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexType {
    U16,
    U32,
}

impl IndexType {
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

#[derive(Debug)]
pub struct IndexedDrawResource {
    pub index_type: IndexType,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub content: BufferContent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VertexAttributeFormat {
    UChar2,
    UChar3,
    UChar4,
    Char2,
    Char3,
    Char4,
    UChar2Normalized,
    UChar3Normalized,
    UChar4Normalized,
    Char2Normalized,
    Char3Normalized,
    Char4Normalized,
    UShort2,
    UShort3,
    UShort4,
    Short2,
    Short3,
    Short4,
    UShort2Normalized,
    UShort3Normalized,
    UShort4Normalized,
    Short2Normalized,
    Short3Normalized,
    Short4Normalized,
    Half2,
    Half3,
    Half4,
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    Int1010102Normalized,
    UInt1010102Normalized,
    UChar4NormalizedBgra,
    UChar,
    Char,
    UCharNormalized,
    CharNormalized,
    UShort,
    Short,
    UShortNormalized,
    ShortNormalized,
    Half,
    FloatRg11B10,
    FloatRgb9E5,
}

impl VertexAttributeFormat {
    pub const fn byte_size(self) -> u32 {
        use VertexAttributeFormat as F;
        match self {
            F::UChar | F::Char | F::UCharNormalized | F::CharNormalized => 1,
            F::UChar2
            | F::Char2
            | F::UChar2Normalized
            | F::Char2Normalized
            | F::UShort
            | F::Short
            | F::UShortNormalized
            | F::ShortNormalized
            | F::Half => 2,
            F::UChar3 | F::Char3 | F::UChar3Normalized | F::Char3Normalized => 3,
            F::UChar4
            | F::Char4
            | F::UChar4Normalized
            | F::Char4Normalized
            | F::UChar4NormalizedBgra
            | F::UShort2
            | F::Short2
            | F::UShort2Normalized
            | F::Short2Normalized
            | F::Half2
            | F::Float
            | F::Int
            | F::UInt
            | F::Int1010102Normalized
            | F::UInt1010102Normalized
            | F::FloatRg11B10
            | F::FloatRgb9E5 => 4,
            F::UShort3 | F::Short3 | F::UShort3Normalized | F::Short3Normalized | F::Half3 => 6,
            F::UShort4
            | F::Short4
            | F::UShort4Normalized
            | F::Short4Normalized
            | F::Half4
            | F::Float2
            | F::Int2
            | F::UInt2 => 8,
            F::Float3 | F::Int3 | F::UInt3 => 12,
            F::Float4 | F::Int4 | F::UInt4 => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum VertexStepFunction {
    Constant,
    #[default]
    PerVertex,
    PerInstance,
}

impl VertexStepFunction {
    pub const fn mtl_ordinal(self) -> u32 {
        use reims_vgpu_protocol::vertex_step as step;
        match self {
            Self::Constant => step::MTL_VERTEX_STEP_FUNCTION_CONSTANT,
            Self::PerVertex => step::MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            Self::PerInstance => step::MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

impl BlendFactor {
    pub const fn is_dual_source(self) -> bool {
        matches!(
            self,
            Self::Src1Color | Self::OneMinusSrc1Color | Self::Src1Alpha | Self::OneMinusSrc1Alpha
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendStateResource {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
    pub constants: [f32; 4],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SeedOrder {
    #[default]
    Rgba8,
    Bgra8,
}

#[derive(Debug)]
pub enum SampledSource {
    Bytes(Arc<Vec<u8>>),
    Target(crate::TargetIdentity),
    Attachment {
        identity: crate::TargetIdentity,
        initial: AttachmentInitial,
    },
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
        let mut request = DrawRequest::default();
        request.target_identity = Some(crate::TargetIdentity::Gva {
            gva: 1,
            width: 4,
            height: 4,
            generation: 1,
            format: TexelLayout::Rgba8,
        });
        assert_eq!(viewport_slot_count(&request), 1);
        assert_eq!(
            request.attachment_slot(request.target_identity.as_ref().unwrap()),
            Some(AttachmentSlot::Primary)
        );
    }
}
