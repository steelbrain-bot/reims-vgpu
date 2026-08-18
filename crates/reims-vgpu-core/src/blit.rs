//! Immutable, resource-resolved blit operations.

use crate::ContentStamp;
use reims_vgpu_protocol::{ByteLength, GuestVirtualAddress, ResourceId, ResourceObject};

/// One checked byte range over a resolved buffer resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedBufferRange {
    pub content: ContentStamp,
    pub address: GuestVirtualAddress,
    pub length: ByteLength,
}

impl ResolvedBufferRange {
    pub const fn resource(self) -> ResourceId<ResourceObject> {
        self.content.resource
    }
}

/// The contract-defined repeating unit of a buffer fill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferFillPattern {
    Byte(u8),
    Word([u8; 4]),
}

impl BufferFillPattern {
    pub const fn bytes(&self) -> &[u8] {
        match self {
            Self::Byte(value) => core::slice::from_ref(value),
            Self::Word(value) => value,
        }
    }
}

/// A blit whose serializer references, bounds, and backing addresses are resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedBufferBlit {
    Fill {
        destination: ResolvedBufferRange,
        pattern: BufferFillPattern,
    },
    Copy {
        source: ResolvedBufferRange,
        destination: ResolvedBufferRange,
    },
}

impl ResolvedBufferBlit {
    pub const fn destination(self) -> ResolvedBufferRange {
        match self {
            Self::Fill { destination, .. } | Self::Copy { destination, .. } => destination,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{ContentVersion, ResourceId};

    fn range(index: u32, generation: u32, address: u64) -> ResolvedBufferRange {
        ResolvedBufferRange {
            content: ContentStamp {
                resource: ResourceId::new(index, generation),
                version: ContentVersion::new(4),
            },
            address: GuestVirtualAddress::new(address),
            length: ByteLength::new(16),
        }
    }

    #[test]
    fn resolved_blits_carry_generational_resources_not_serializer_ordinals() {
        let operation = ResolvedBufferBlit::Copy {
            source: range(7, 2, 0x1000),
            destination: range(7, 3, 0x2000),
        };

        assert_eq!(operation.destination().resource(), ResourceId::new(7, 3));
    }
}
