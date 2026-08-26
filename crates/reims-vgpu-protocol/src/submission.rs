//! Semantic metadata carried by every backend submission.

use crate::{ContentVersion, ObjectTableRef, ResourceId, SubmissionId, TaskId};

/// Marker for a heterogeneous task resource-list reference.
pub enum ResourceObject {}

/// Marker for the serializer's heap-specific reference namespace.
///
/// Heap refs are resolved before heap-placed resources are constructed and do
/// not name slots in the task's heterogeneous object list. Keeping a separate
/// marker prevents an equal integer in those two namespaces from becoming an
/// accidental resource relation.
pub enum HeapObject {}

/// Declared access to a resource reached indirectly through an argument buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceUsage(u8);

impl ResourceUsage {
    pub const READ: u8 = 1 << 0;
    pub const WRITE: u8 = 1 << 1;
    pub const SAMPLE: u8 = 1 << 2;
    pub const KNOWN: u8 = Self::READ | Self::WRITE | Self::SAMPLE;

    pub const fn from_bits(bits: u32) -> Result<Self, ParticipationDecodeError> {
        if bits > u8::MAX as u32 || (bits as u8) & !Self::KNOWN != 0 {
            Err(ParticipationDecodeError::UnknownResourceUsage(bits))
        } else {
            Ok(Self(bits as u8))
        }
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn reads(self) -> bool {
        self.0 & Self::READ != 0
    }

    pub const fn writes(self) -> bool {
        self.0 & Self::WRITE != 0
    }

    pub const fn samples(self) -> bool {
        self.0 & Self::SAMPLE != 0
    }
}

/// Render stages in which an indirectly declared resource may participate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderStages(u8);

impl RenderStages {
    pub const VERTEX: u8 = 1 << 0;
    pub const FRAGMENT: u8 = 1 << 1;
    pub const TILE: u8 = 1 << 2;
    pub const OBJECT: u8 = 1 << 3;
    pub const MESH: u8 = 1 << 4;
    pub const KNOWN: u8 = Self::VERTEX | Self::FRAGMENT | Self::TILE | Self::OBJECT | Self::MESH;

    pub const fn from_bits(bits: u16) -> Result<Self, ParticipationDecodeError> {
        if bits > u8::MAX as u16 || (bits as u8) & !Self::KNOWN != 0 {
            Err(ParticipationDecodeError::UnknownRenderStages(bits))
        } else {
            Ok(Self(bits as u8))
        }
    }

    pub const fn bits(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipationDecodeError {
    UnknownResourceUsage(u32),
    UnknownRenderStages(u16),
}

/// Marker for the indirect-command-buffer allocator's reference namespace.
///
/// These references are created and destroyed independently of task resource
/// list entries. An equal integer in the two namespaces does not identify the
/// same object.
#[derive(Debug)]
pub enum IndirectCommandBufferObject {}

/// The four validity transitions carried beside one submitted resource.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceValidity {
    pub clear_host: bool,
    pub set_host: bool,
    pub clear_guest: bool,
    pub set_guest: bool,
}

/// One resource participating in a guest submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubmissionResourceUse {
    pub object: ObjectTableRef<ResourceObject>,
    /// Canonical identity when the object has already been constructed.
    /// Resource tables may also name declared residency entries which no
    /// command has resolved yet; those deliberately remain unresolved.
    pub resource: Option<ResourceId<ResourceObject>>,
    /// Content version observed after applying this record's pre-submission
    /// validity transition.
    pub expected_content: Option<ContentVersion>,
    pub validity: ResourceValidity,
}

/// Semantic encoder family selected by a segment header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Render,
    Compute,
    Blit,
    Event,
    Info,
}

/// The segment containing a backend operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBoundary {
    /// Child serialized-stream position within the packet's counted stream list.
    pub stream_index: u32,
    /// Segment position within that serialized stream.
    pub index: u32,
    pub kind: SegmentKind,
    pub continues_previous: bool,
    pub continues_next: bool,
}

/// Stable identity shared by all operations decoded from one guest submission.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SubmissionIdentity {
    pub id: SubmissionId,
    pub task: TaskId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChannelId, ChannelSequence, DomainSequence, HazardDomainId, PublicationDomainId,
        PublicationSequence, SubmissionDomainId,
    };

    #[test]
    fn resource_validity_is_not_a_wire_dword() {
        let use_ = SubmissionResourceUse {
            object: ObjectTableRef::new(7),
            resource: None,
            expected_content: None,
            validity: ResourceValidity {
                clear_host: true,
                set_guest: true,
                ..ResourceValidity::default()
            },
        };
        assert_eq!(use_.object.get(), 7);
        assert!(use_.validity.clear_host);
        assert!(use_.validity.set_guest);
    }

    #[test]
    fn participation_flags_are_total_only_over_the_sdk_vocabulary() {
        let usage = ResourceUsage::from_bits(7).unwrap();
        assert!(usage.reads() && usage.writes() && usage.samples());
        assert_eq!(
            ResourceUsage::from_bits(8),
            Err(ParticipationDecodeError::UnknownResourceUsage(8))
        );

        let stages = RenderStages::from_bits(0x1f).unwrap();
        assert_eq!(stages.bits(), 0x1f);
        assert_eq!(
            RenderStages::from_bits(0x20),
            Err(ParticipationDecodeError::UnknownRenderStages(0x20))
        );
    }

    #[test]
    fn queue_submission_order_is_derived_from_fifo_order() {
        assert_eq!(
            SubmissionDomainId::for_fifo_channel(ChannelId::new(7)),
            SubmissionDomainId::new(7)
        );
        assert_eq!(
            DomainSequence::for_channel_sequence(ChannelSequence::new(11)),
            DomainSequence::new(11)
        );
        assert_eq!(
            HazardDomainId::for_submission_domain(SubmissionDomainId::new(7)),
            HazardDomainId::new(7)
        );
        assert_eq!(
            PublicationDomainId::for_fifo_channel(ChannelId::new(7)),
            PublicationDomainId::new(7)
        );
        assert_eq!(
            PublicationSequence::for_channel_sequence(ChannelSequence::new(11)),
            PublicationSequence::new(11)
        );
    }
}
