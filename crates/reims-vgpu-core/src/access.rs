//! Backend-neutral resource participation and proven access precision.

use reims_vgpu_protocol::{
    BackingId, HazardDomainId, HeapObject, RenderStages, ResourceId, ResourceObject,
};

/// Contract-proven read/write participation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    /// The operation participates, but its direction is not yet established.
    /// This conflicts conservatively and remains visible in the census.
    Unknown,
}

impl AccessMode {
    pub const fn conflicts_with(self, other: Self) -> bool {
        !matches!((self, other), (Self::Read, Self::Read))
    }
}

/// API stage scope retained without converting it to Vulkan stage masks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageScope {
    All,
    Vertex,
    Fragment,
    Tile,
    Object,
    Mesh,
    Compute,
    /// Fixed-function consumption of draw or dispatch argument structures.
    Indirect,
    /// Fixed-function vertex attribute fetch.
    VertexInput,
    /// Fixed-function index fetch.
    IndexInput,
    /// Fixed-function color attachment load, blend, store, and resolve.
    ColorAttachment,
    /// Fixed-function depth/stencil test, load, store, and resolve.
    DepthStencilAttachment,
    /// Copy of a completed query result into its guest-visible buffer.
    QueryResolve,
    /// Exact possibly-multi-stage scope carried by a qualified render
    /// participation declaration.
    Render(RenderStages),
    Blit,
    Host,
    /// A declared stage value accepted at the boundary but not yet mapped to a
    /// narrower semantic scope.
    Unknown,
}

/// Non-empty half-open byte interval in canonical backing coordinates.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LinearRange {
    start: u64,
    end: u64,
}

impl LinearRange {
    pub fn new(start: u64, length: u64) -> Option<Self> {
        let end = start.checked_add(length)?;
        (start < end).then_some(Self { start, end })
    }

    pub const fn start(self) -> u64 {
        self.start
    }

    pub const fn end(self) -> u64 {
        self.end
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ImageAspect {
    Color,
    Depth,
    Stencil,
    Plane(u8),
}

/// Non-empty three-dimensional half-open texel box.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TexelBox {
    pub origin: [u32; 3],
    pub end: [u32; 3],
}

impl TexelBox {
    pub fn new(origin: [u32; 3], extent: [u32; 3]) -> Option<Self> {
        if extent.contains(&0) {
            return None;
        }
        let end = [
            origin[0].checked_add(extent[0])?,
            origin[1].checked_add(extent[1])?,
            origin[2].checked_add(extent[2])?,
        ];
        Some(Self { origin, end })
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.origin[0] < other.end[0]
            && other.origin[0] < self.end[0]
            && self.origin[1] < other.end[1]
            && other.origin[1] < self.end[1]
            && self.origin[2] < other.end[2]
            && other.origin[2] < self.end[2]
    }
}

/// Image coordinates established by the decoded operation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ImageSubresourceRange {
    pub aspect: ImageAspect,
    mip_start: u32,
    mip_end: u32,
    layer_start: u32,
    layer_end: u32,
    /// `None` means the whole selected mip/layer subresources.
    pub texels: Option<TexelBox>,
}

impl ImageSubresourceRange {
    pub fn new(
        aspect: ImageAspect,
        mip_start: u32,
        mip_count: u32,
        layer_start: u32,
        layer_count: u32,
        texels: Option<TexelBox>,
    ) -> Option<Self> {
        let mip_end = mip_start.checked_add(mip_count)?;
        let layer_end = layer_start.checked_add(layer_count)?;
        (mip_start < mip_end && layer_start < layer_end).then_some(Self {
            aspect,
            mip_start,
            mip_end,
            layer_start,
            layer_end,
            texels,
        })
    }

    pub const fn overlaps(self, other: Self) -> bool {
        if !matches_aspect(self.aspect, other.aspect)
            || self.mip_start >= other.mip_end
            || other.mip_start >= self.mip_end
            || self.layer_start >= other.layer_end
            || other.layer_start >= self.layer_end
        {
            return false;
        }
        match (self.texels, other.texels) {
            (Some(left), Some(right)) => left.overlaps(right),
            _ => true,
        }
    }

    pub const fn mip_start(self) -> u32 {
        self.mip_start
    }

    pub const fn mip_end(self) -> u32 {
        self.mip_end
    }

    pub const fn layer_start(self) -> u32 {
        self.layer_start
    }

    pub const fn layer_end(self) -> u32 {
        self.layer_end
    }
}

const fn matches_aspect(left: ImageAspect, right: ImageAspect) -> bool {
    match (left, right) {
        (ImageAspect::Color, ImageAspect::Color)
        | (ImageAspect::Depth, ImageAspect::Depth)
        | (ImageAspect::Stencil, ImageAspect::Stencil) => true,
        (ImageAspect::Plane(left), ImageAspect::Plane(right)) => left == right,
        _ => false,
    }
}

/// Highest contract-proven dependency-key rung for one access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessScope {
    Linear(LinearRange),
    Image(ImageSubresourceRange),
    WholeBacking,
    WholeHeap,
    WholeDomain,
}

impl AccessScope {
    pub const fn precision(self) -> AccessPrecision {
        match self {
            Self::Linear(_) => AccessPrecision::ExactRange,
            Self::Image(_) => AccessPrecision::ExactSubresource,
            Self::WholeBacking => AccessPrecision::WholeBacking,
            Self::WholeHeap => AccessPrecision::WholeHeap,
            Self::WholeDomain => AccessPrecision::WholeDomain,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessPrecision {
    ExactRange,
    ExactSubresource,
    WholeBacking,
    WholeHeap,
    WholeDomain,
}

/// One immutable access summary consumed by dependency compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccessIntent {
    pub hazard_domain: HazardDomainId,
    /// Canonical identity joins aliases. It is absent only at the explicit
    /// whole-domain precision rung.
    pub target: Option<AccessTarget>,
    /// View/resource identity is diagnostic and lets the compiler distinguish
    /// an alias edge from repeated use of the same semantic resource.
    pub resource: Option<ResourceId<ResourceObject>>,
    pub scope: AccessScope,
    pub mode: AccessMode,
    pub stages: StageScope,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum AccessTarget {
    Backing(BackingId),
    Heap(ResourceId<HeapObject>),
}

impl AccessIntent {
    pub fn for_backing(
        hazard_domain: HazardDomainId,
        backing: BackingId,
        resource: Option<ResourceId<ResourceObject>>,
        scope: AccessScope,
        mode: AccessMode,
        stages: StageScope,
    ) -> Option<Self> {
        (!matches!(scope, AccessScope::WholeDomain)).then_some(Self {
            hazard_domain,
            target: Some(AccessTarget::Backing(backing)),
            resource,
            scope,
            mode,
            stages,
        })
    }

    pub const fn for_heap(
        hazard_domain: HazardDomainId,
        heap: ResourceId<HeapObject>,
        mode: AccessMode,
        stages: StageScope,
    ) -> Self {
        Self {
            hazard_domain,
            target: Some(AccessTarget::Heap(heap)),
            resource: None,
            scope: AccessScope::WholeHeap,
            mode,
            stages,
        }
    }

    pub const fn whole_domain(
        hazard_domain: HazardDomainId,
        mode: AccessMode,
        stages: StageScope,
    ) -> Self {
        Self {
            hazard_domain,
            target: None,
            resource: None,
            scope: AccessScope::WholeDomain,
            mode,
            stages,
        }
    }

    pub const fn precision(self) -> AccessPrecision {
        self.scope.precision()
    }
}
