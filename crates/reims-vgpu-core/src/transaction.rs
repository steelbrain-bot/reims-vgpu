//! Immutable device transactions and backend-neutral operation families.
//!
//! Every accepted packet carries the same ingress, prerequisite, completion,
//! and publication envelope. EXEC child streams deliberately remain
//! [`ResolvedExecStream`] rather than being called API command buffers: their
//! counted order and segment continuations are established, while independent
//! commit/recording boundaries are not. Each EXEC carries its own fresh,
//! wrapping FIFO completion point; that point identifies completion of the
//! packet, not membership in a source API command buffer. No backend may split
//! one merely to create executor work.

use crate::{
    AccessIntent, CompletionStamp, MemoryBarrierScope, ParticipationOperation, PublicationPosition,
    ResolvedBlit, ResolvedResourceState, SessionGenerationLease, StageScope,
};
use reims_vgpu_protocol::{
    ChannelId, ChannelSequence, DomainSequence, EventObject, FenceObject, IngressOrdinal,
    ResourceId, ResourceObject, SegmentBoundary, SegmentKind, StampWait, SubmissionDomainId,
    SubmissionIdentity, TransactionId,
};

/// Explicit dependency decoded before a packet may have a semantic effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionPrerequisite {
    Stamp {
        wait: StampWait,
    },
    Event {
        event: ResourceId<EventObject>,
        value: u64,
    },
    Fence {
        fence: ResourceId<FenceObject>,
        generation: u64,
    },
}

/// One accepted packet with fully owned semantic payload.
#[derive(Clone, Debug)]
pub struct DeviceTransaction<Operation, Lifecycle, Query, Present, Control> {
    pub id: TransactionId,
    pub session_generation: SessionGenerationLease,
    pub channel: ChannelId,
    pub channel_sequence: ChannelSequence,
    pub ingress_ordinal: IngressOrdinal,
    pub prerequisites: Box<[TransactionPrerequisite]>,
    pub completion_stamp: Option<CompletionStamp>,
    pub payload: DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>,
}

impl<Operation, Lifecycle, Query, Present, Control>
    DeviceTransaction<Operation, Lifecycle, Query, Present, Control>
{
    pub const fn class(&self) -> TransactionClass {
        self.payload.class()
    }

    /// The source command queue's ordering domain, derived from its live FIFO
    /// channel rather than carried as a second independently writable value.
    pub const fn submission_domain(&self) -> SubmissionDomainId {
        SubmissionDomainId::for_fifo_channel(self.channel)
    }

    /// Position within [`Self::submission_domain`].
    pub const fn domain_sequence(&self) -> DomainSequence {
        DomainSequence::for_channel_sequence(self.channel_sequence)
    }

    /// Guest-visible effects publish at the same FIFO position that admitted
    /// the packet; publication order is not independently assignable.
    pub const fn publication(&self) -> PublicationPosition {
        PublicationPosition {
            domain: reims_vgpu_protocol::PublicationDomainId::for_fifo_channel(self.channel),
            sequence: reims_vgpu_protocol::PublicationSequence::for_channel_sequence(
                self.channel_sequence,
            ),
        }
    }

    /// Check the packet-class invariants required before semantic admission.
    ///
    /// EXEC completion is packet-local. The boundary decoder establishes that
    /// its value is the fresh point assigned by the FIFO producer; downstream
    /// code must not use equal or adjacent values to reconstruct a source API
    /// command-buffer identity.
    pub const fn validate_envelope(&self) -> Result<(), TransactionEnvelopeError> {
        if let Some(completion) = self.completion_stamp {
            if completion.slot != self.channel.get() {
                return Err(
                    TransactionEnvelopeError::CompletionSlotDoesNotMatchChannel {
                        channel: self.channel,
                        slot: completion.slot,
                    },
                );
            }
        }
        if matches!(&self.payload, DeviceTransactionPayload::Exec(_))
            && self.completion_stamp.is_none()
        {
            return Err(TransactionEnvelopeError::ExecMissingCompletionPoint);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionEnvelopeError {
    ExecMissingCompletionPoint,
    CompletionSlotDoesNotMatchChannel { channel: ChannelId, slot: u32 },
}

/// The five semantic packet classes. Unsupported packets never enter this enum;
/// their decoder returns an exact typed refusal instead of a catch-all value.
#[derive(Clone, Debug)]
pub enum DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control> {
    Exec(ExecTransaction<Operation>),
    ResourceLifecycle(Lifecycle),
    Query(Query),
    Present(Present),
    Control(Control),
}

impl<Operation, Lifecycle, Query, Present, Control>
    DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>
{
    pub const fn class(&self) -> TransactionClass {
        match self {
            Self::Exec(_) => TransactionClass::Exec,
            Self::ResourceLifecycle(_) => TransactionClass::ResourceLifecycle,
            Self::Query(_) => TransactionClass::Query,
            Self::Present(_) => TransactionClass::Present,
            Self::Control(_) => TransactionClass::Control,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionClass {
    Exec,
    ResourceLifecycle,
    Query,
    Present,
    Control,
}

/// One counted EXEC and the ordered child streams proven to belong to it.
#[derive(Clone, Debug)]
pub struct ExecTransaction<Operation> {
    pub identity: SubmissionIdentity,
    pub prologue: ExecPrologue<Operation>,
    pub streams: Box<[ResolvedExecStream<Operation>]>,
    pub accesses: Box<[AccessIntent]>,
}

impl<Operation> ExecTransaction<Operation> {
    pub fn operations(&self) -> impl Iterator<Item = &Operation> {
        self.prologue.operations.iter().chain(
            self.streams
                .iter()
                .flat_map(|stream| stream.segments.iter())
                .flat_map(|segment| segment.operations.iter()),
        )
    }
}

/// Native execution views of one immutable semantic EXEC when its resource
/// state prologue must complete before the encoder-stream suffix can prepare.
/// Both views retain the same submission identity and never become separate
/// semantic transactions.
#[derive(Clone, Debug)]
pub struct ResourceStateExecutionChain<Operation> {
    transaction: TransactionId,
    prefix: ExecTransaction<Operation>,
    suffix: ExecTransaction<Operation>,
    suffix_operation_base: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceStateExecutionChainError {
    EmptyResourceStatePrefix,
}

impl<Operation> ResourceStateExecutionChain<Operation> {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn prefix(&self) -> &ExecTransaction<Operation> {
        &self.prefix
    }

    pub const fn suffix(&self) -> &ExecTransaction<Operation> {
        &self.suffix
    }

    pub const fn suffix_operation_base(&self) -> usize {
        self.suffix_operation_base
    }

    pub fn into_parts(
        self,
    ) -> (
        ExecTransaction<Operation>,
        ExecTransaction<Operation>,
        usize,
    ) {
        (self.prefix, self.suffix, self.suffix_operation_base)
    }
}

pub fn resource_state_execution_chain<Operation>(
    transaction: TransactionId,
    exec: ExecTransaction<Operation>,
) -> Result<ResourceStateExecutionChain<Operation>, ResourceStateExecutionChainError> {
    let ExecTransaction {
        identity,
        prologue,
        streams,
        accesses,
    } = exec;
    let suffix_operation_base = prologue.operations.len();
    if suffix_operation_base == 0 {
        return Err(ResourceStateExecutionChainError::EmptyResourceStatePrefix);
    }
    Ok(ResourceStateExecutionChain {
        transaction,
        prefix: ExecTransaction {
            identity,
            prologue,
            streams: Box::new([]),
            accesses: accesses.clone(),
        },
        suffix: ExecTransaction {
            identity,
            prologue: ExecPrologue::default(),
            streams,
            accesses,
        },
        suffix_operation_base,
    })
}

/// Submission-level commands ordered before the first encoder stream.
///
/// The EXEC resource table is not an encoder and therefore cannot borrow a
/// fabricated segment boundary. Construction is restricted to its one decoded
/// semantic family while the generic container keeps flattened operation
/// traversal uniform for admission, preparation, and recording.
#[derive(Clone, Debug)]
pub struct ExecPrologue<Operation> {
    operations: Box<[Operation]>,
}

impl<Operation> Default for ExecPrologue<Operation> {
    fn default() -> Self {
        Self {
            operations: Box::new([]),
        }
    }
}

impl<Operation> ExecPrologue<Operation> {
    pub const fn operations(&self) -> &[Operation] {
        &self.operations
    }
}

impl<Render, Compute, Info, Indirect, Completion>
    ExecPrologue<ResolvedOperation<Render, Compute, Info, Indirect, Completion>>
{
    pub fn resource_states(states: impl Into<Box<[ResolvedResourceState]>>) -> Self {
        Self {
            operations: states
                .into()
                .into_vec()
                .into_iter()
                .map(ResolvedOperation::ResourceState)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

/// One counted child serialized stream in its established EXEC order.
#[derive(Clone, Debug)]
pub struct ResolvedExecStream<Operation> {
    pub stream_index: u32,
    pub segments: Box<[ResolvedExecSegment<Operation>]>,
}

/// One segment and its immutable, ordered operations.
#[derive(Clone, Debug)]
pub struct ResolvedExecSegment<Operation> {
    pub boundary: SegmentBoundary,
    pub operations: Box<[Operation]>,
}

/// Exhaustive operation families admitted to backend-neutral execution.
///
/// Family payloads stay generic while their wire contracts are closed. A
/// production decoder supplies closed enums for those parameters; there is no
/// `Other`, raw opcode, or backend-native payload arm here.
#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedOperation<Render, Compute, Info, Indirect, Completion> {
    EncoderBoundary(EncoderBoundary),
    Render(Render),
    Compute(Compute),
    Blit(Box<ResolvedBlit>),
    Event(EventOperation),
    Fence(FenceOperation),
    Barrier(BarrierOperation),
    Participation(ParticipationOperation),
    ResourceState(ResolvedResourceState),
    InfoQuery(Info),
    IndirectCommand(Indirect),
    CompletionEffect(Completion),
}

impl<Render, Compute, Info, Indirect, Completion>
    ResolvedOperation<Render, Compute, Info, Indirect, Completion>
{
    pub const fn kind(&self) -> OperationKind {
        match self {
            Self::EncoderBoundary(_) => OperationKind::EncoderBoundary,
            Self::Render(_) => OperationKind::Render,
            Self::Compute(_) => OperationKind::Compute,
            Self::Blit(_) => OperationKind::Blit,
            Self::Event(_) => OperationKind::Event,
            Self::Fence(_) => OperationKind::Fence,
            Self::Barrier(_) => OperationKind::Barrier,
            Self::Participation(_) => OperationKind::Participation,
            Self::ResourceState(_) => OperationKind::ResourceState,
            Self::InfoQuery(_) => OperationKind::InfoQuery,
            Self::IndirectCommand(_) => OperationKind::IndirectCommand,
            Self::CompletionEffect(_) => OperationKind::CompletionEffect,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum OperationKind {
    EncoderBoundary,
    Render,
    Compute,
    Blit,
    Event,
    Fence,
    Barrier,
    Participation,
    ResourceState,
    InfoQuery,
    IndirectCommand,
    CompletionEffect,
}

impl OperationKind {
    /// Complete backend-neutral operation-family vocabulary. Backends use this
    /// typed surface for cutover coverage rather than deriving a roster from
    /// source text or observed traffic.
    pub const ALL: [Self; 12] = [
        Self::EncoderBoundary,
        Self::Render,
        Self::Compute,
        Self::Blit,
        Self::Event,
        Self::Fence,
        Self::Barrier,
        Self::Participation,
        Self::ResourceState,
        Self::InfoQuery,
        Self::IndirectCommand,
        Self::CompletionEffect,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncoderBoundary {
    Begin(SegmentKind),
    End(SegmentKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventOperation {
    pub event: ResourceId<EventObject>,
    pub kind: EventOperationKind,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventOperationKind {
    Signal,
    Wait,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FenceOperation {
    pub fence: ResourceId<FenceObject>,
    pub kind: FenceOperationKind,
    pub generation: u64,
    /// Encoder domain and exact consumer/producer stage scope carried by this
    /// operation. Render update and wait records use the same wire field for
    /// `afterStages` and `beforeStages` respectively.
    pub scope: FenceScope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FenceOperationKind {
    Update,
    Wait,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FenceScope {
    Blit,
    Compute,
    Render(crate::RenderBarrierStages),
}

impl FenceScope {
    pub fn stage_scope(self) -> StageScope {
        match self {
            Self::Blit => StageScope::Blit,
            Self::Compute => StageScope::Compute,
            Self::Render(stages) => StageScope::Render(
                reims_vgpu_protocol::RenderStages::from_bits(stages.bits())
                    .expect("fence render stages use the same complete API bit vocabulary"),
            ),
        }
    }
}

/// API barrier semantics before any Vulkan-stage projection.
///
/// Resource-list and resource-scope records are distinct wire forms. Keeping
/// them as distinct variants prevents a backend from choosing between two
/// simultaneously populated targets or assigning meaning to an empty list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BarrierOperation {
    Resources {
        resources: Box<[ResourceId<ResourceObject>]>,
        before: StageScope,
        after: StageScope,
    },
    Scope {
        scope: MemoryBarrierScope,
        before: StageScope,
        after: StageScope,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SessionGeneration, StageScope};
    use reims_vgpu_protocol::{
        PublicationDomainId, PublicationSequence, SessionGenerationId, SubmissionId, TaskId,
    };

    type Payload =
        DeviceTransactionPayload<(), &'static str, &'static str, &'static str, &'static str>;

    fn generation() -> SessionGenerationLease {
        SessionGeneration::new(SessionGenerationId::new(1))
            .try_lease()
            .unwrap()
    }

    fn transaction(
        id: u64,
        payload: Payload,
    ) -> DeviceTransaction<(), &'static str, &'static str, &'static str, &'static str> {
        DeviceTransaction {
            id: TransactionId::new(id),
            session_generation: generation(),
            channel: ChannelId::new(2),
            channel_sequence: ChannelSequence::new(id),
            ingress_ordinal: IngressOrdinal::new(id),
            prerequisites: Box::new([]),
            completion_stamp: None,
            payload,
        }
    }

    #[test]
    fn every_packet_class_uses_the_same_order_and_publication_envelope() {
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(3),
            },
            prologue: ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        };
        let transactions = [
            transaction(1, Payload::Exec(exec)),
            transaction(2, Payload::ResourceLifecycle("create")),
            transaction(3, Payload::Query("limits")),
            transaction(4, Payload::Present("surface")),
            transaction(5, Payload::Control("cursor")),
        ];
        assert_eq!(
            transactions
                .iter()
                .map(DeviceTransaction::class)
                .collect::<Vec<_>>(),
            vec![
                TransactionClass::Exec,
                TransactionClass::ResourceLifecycle,
                TransactionClass::Query,
                TransactionClass::Present,
                TransactionClass::Control,
            ]
        );
        assert!(transactions
            .windows(2)
            .all(|pair| pair[0].ingress_ordinal < pair[1].ingress_ordinal));
    }

    #[test]
    fn resource_state_prologue_precedes_every_encoder_operation() {
        type Operation = ResolvedOperation<(), (), (), (), ()>;
        let exec = ExecTransaction::<Operation> {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: ExecPrologue::resource_states([ResolvedResourceState {
                resource: None,
                mappings: Box::new([]),
                targets: Box::new([]),
                ops: reims_vgpu_protocol::ResourceValidityOps::default(),
            }]),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Compute,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::Compute(())]),
                }]),
            }]),
            accesses: Box::new([]),
        };

        assert_eq!(
            exec.operations()
                .map(ResolvedOperation::kind)
                .collect::<Vec<_>>(),
            vec![OperationKind::ResourceState, OperationKind::Compute]
        );
    }

    #[test]
    fn exec_requires_its_packet_local_completion_point() {
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(3),
            },
            prologue: ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        };
        let mut transaction = transaction(1, Payload::Exec(exec));
        assert_eq!(
            transaction.validate_envelope(),
            Err(TransactionEnvelopeError::ExecMissingCompletionPoint)
        );

        transaction.completion_stamp = Some(CompletionStamp::new(2, u32::MAX));
        assert_eq!(transaction.validate_envelope(), Ok(()));
        assert_eq!(
            transaction.completion_stamp,
            Some(CompletionStamp::new(2, u32::MAX))
        );
        assert_eq!(transaction.submission_domain(), SubmissionDomainId::new(2));
        assert_eq!(transaction.domain_sequence(), DomainSequence::new(1));
        assert_eq!(
            transaction.publication(),
            PublicationPosition {
                domain: PublicationDomainId::new(2),
                sequence: PublicationSequence::new(1),
            }
        );
    }

    #[test]
    fn exec_keeps_counted_stream_segment_and_operation_order() {
        let segments = vec![
            ResolvedExecSegment {
                boundary: SegmentBoundary {
                    stream_index: 0,
                    index: 0,
                    kind: SegmentKind::Render,
                    continues_previous: false,
                    continues_next: true,
                },
                operations: vec![1, 2].into_boxed_slice(),
            },
            ResolvedExecSegment {
                boundary: SegmentBoundary {
                    stream_index: 0,
                    index: 1,
                    kind: SegmentKind::Render,
                    continues_previous: true,
                    continues_next: false,
                },
                operations: vec![3].into_boxed_slice(),
            },
        ];
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: ExecPrologue::default(),
            streams: vec![ResolvedExecStream {
                stream_index: 0,
                segments: segments.into_boxed_slice(),
            }]
            .into_boxed_slice(),
            accesses: Box::new([]),
        };
        assert_eq!(
            exec.operations().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(exec.streams[0].segments[0].boundary.continues_next);
        assert!(exec.streams[0].segments[1].boundary.continues_previous);
    }

    #[test]
    fn resource_state_native_chain_preserves_one_identity_and_flattened_suffix_base() {
        let identity = SubmissionIdentity {
            id: SubmissionId::new(7),
            task: TaskId::new(3),
        };
        let exec = ExecTransaction {
            identity,
            prologue: ExecPrologue {
                operations: vec![10, 11].into_boxed_slice(),
            },
            streams: vec![ResolvedExecStream {
                stream_index: 0,
                segments: vec![ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: vec![12, 13].into_boxed_slice(),
                }]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
            accesses: Box::new([]),
        };
        let chain = resource_state_execution_chain(TransactionId::new(9), exec).unwrap();
        assert_eq!(chain.transaction(), TransactionId::new(9));
        assert_eq!(chain.prefix().identity, identity);
        assert_eq!(chain.suffix().identity, identity);
        assert_eq!(
            chain.prefix().operations().copied().collect::<Vec<_>>(),
            [10, 11]
        );
        assert_eq!(
            chain.suffix().operations().copied().collect::<Vec<_>>(),
            [12, 13]
        );
        assert_eq!(chain.suffix_operation_base(), 2);
    }

    #[test]
    fn resolved_operation_family_is_backend_neutral_and_closed() {
        type Operation = ResolvedOperation<(), (), (), (), ()>;
        let barrier = Operation::Barrier(BarrierOperation::Scope {
            scope: MemoryBarrierScope::ALL,
            before: StageScope::Vertex,
            after: StageScope::Fragment,
        });
        let event = Operation::Event(EventOperation {
            event: ResourceId::new(4, 2),
            kind: EventOperationKind::Signal,
            value: 9,
        });
        assert_eq!(barrier.kind(), OperationKind::Barrier);
        assert_eq!(event.kind(), OperationKind::Event);
        let resource_barrier = Operation::Barrier(BarrierOperation::Resources {
            resources: Box::new([ResourceId::new(8, 3)]),
            before: StageScope::Compute,
            after: StageScope::Compute,
        });
        assert!(matches!(
            resource_barrier,
            Operation::Barrier(BarrierOperation::Resources { resources, .. })
                if resources.as_ref() == [ResourceId::new(8, 3)]
        ));
        let participation = Operation::Participation(ParticipationOperation::Heap {
            heap: ResourceId::new(5, 2),
            scope: crate::ParticipationScope::Compute,
        });
        assert_eq!(participation.kind(), OperationKind::Participation);
    }

    #[test]
    fn operation_kind_surface_contains_every_family_once() {
        let kinds = OperationKind::ALL
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(kinds.len(), OperationKind::ALL.len());
        assert_eq!(kinds.first(), Some(&OperationKind::EncoderBoundary));
        assert_eq!(kinds.last(), Some(&OperationKind::CompletionEffect));
    }
}
