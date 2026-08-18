//! Backend-independent identities for renderable guest resources.

use crate::contract::pixel_format::TexelLayout;

/// Protocol-derived render-target identity (resource state, not content hash).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TargetIdentity {
    Surface {
        id: u32,
        width: u32,
        height: u32,
        generation: u64,
        format: TexelLayout,
    },
    Texture {
        ref_: u32,
        width: u32,
        height: u32,
        generation: u64,
        stencil: bool,
    },
    Gva {
        gva: u64,
        width: u32,
        height: u32,
        generation: u64,
        format: TexelLayout,
    },
    Anonymous {
        slot: u64,
    },
}

impl Default for TargetIdentity {
    fn default() -> Self {
        Self::Anonymous { slot: 0 }
    }
}

/// First semantic field on which two target keys disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKeyDivergence {
    Absent,
    Namespace,
    Geometry,
    Generation,
    Other,
}

impl TargetKeyDivergence {
    pub fn label(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Namespace => "namespace",
            Self::Geometry => "geometry",
            Self::Generation => "generation",
            Self::Other => "other",
        }
    }
}

impl TargetIdentity {
    pub fn width(&self) -> u32 {
        match self {
            Self::Surface { width, .. } | Self::Texture { width, .. } | Self::Gva { width, .. } => {
                *width
            }
            Self::Anonymous { .. } => 0,
        }
    }
    pub fn height(&self) -> u32 {
        match self {
            Self::Surface { height, .. }
            | Self::Texture { height, .. }
            | Self::Gva { height, .. } => *height,
            Self::Anonymous { .. } => 0,
        }
    }
    pub fn generation(&self) -> u64 {
        match self {
            Self::Surface { generation, .. }
            | Self::Texture { generation, .. }
            | Self::Gva { generation, .. } => *generation,
            Self::Anonymous { .. } => 0,
        }
    }
    pub fn namespaced_id(&self) -> (u8, u64) {
        match self {
            Self::Surface { id, .. } => (0, u64::from(*id)),
            Self::Texture { ref_, .. } => (1, u64::from(*ref_)),
            Self::Gva { gva, .. } => (2, *gva),
            Self::Anonymous { slot } => (3, *slot),
        }
    }
    pub fn diverges_from(&self, held: &Self) -> TargetKeyDivergence {
        if self.namespaced_id() != held.namespaced_id() {
            return TargetKeyDivergence::Namespace;
        }
        if (self.width(), self.height()) != (held.width(), held.height()) {
            return TargetKeyDivergence::Geometry;
        }
        let mut regenerated = self.clone();
        match &mut regenerated {
            Self::Surface { generation, .. }
            | Self::Texture { generation, .. }
            | Self::Gva { generation, .. } => *generation = held.generation(),
            Self::Anonymous { .. } => {}
        }
        if regenerated == *held {
            TargetKeyDivergence::Generation
        } else {
            TargetKeyDivergence::Other
        }
    }
    pub fn with_generation(&self, generation: u64) -> Self {
        let mut next = self.clone();
        match &mut next {
            Self::Surface {
                generation: value, ..
            }
            | Self::Texture {
                generation: value, ..
            }
            | Self::Gva {
                generation: value, ..
            } => *value = generation,
            Self::Anonymous { .. } => {}
        }
        next
    }
    pub fn aliases(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Surface { id: a, .. }, Self::Surface { id: b, .. }) => a == b,
            (Self::Gva { gva: a, .. }, Self::Gva { gva: b, .. }) => a == b,
            (Self::Texture { ref_: a, .. }, Self::Texture { ref_: b, .. }) => a == b,
            (Self::Anonymous { slot: a }, Self::Anonymous { slot: b }) => a == b,
            _ => false,
        }
    }
    pub fn resident_layout(&self) -> TexelLayout {
        match self {
            Self::Surface { format, .. } | Self::Gva { format, .. } => *format,
            Self::Texture { .. } | Self::Anonymous { .. } => TexelLayout::Rgba8,
        }
    }
}
