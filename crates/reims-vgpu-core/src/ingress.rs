//! Single owner for immutable transaction-envelope admission.
//!
//! Wire channels are live ordering domains. Definition opens a channel
//! lifetime, retirement closes it, and redefinition starts its sequence again
//! only after all owners of the former lifetime have retired their domain.
//! Packet, ingress, hazard, and publication positions are derived here once;
//! callers cannot assign mutually inconsistent copies.

use crate::{
    CompletionStamp, DeviceTransaction, DeviceTransactionPayload, SessionGeneration,
    TransactionEnvelopeError, TransactionPrerequisite,
};
use reims_vgpu_protocol::{ChannelId, ChannelSequence, IngressOrdinal, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionIngressError {
    DuplicateChannel,
    UnknownChannel,
    SessionGenerationClosed,
    IdentitySpaceExhausted,
    IngressSpaceExhausted,
    ChannelSequenceExhausted,
    Envelope(TransactionEnvelopeError),
}

#[derive(Clone, Debug)]
pub struct TransactionIngressOwner {
    generation: SessionGeneration,
    next_transaction: u64,
    next_ingress: u64,
    channels: BTreeMap<ChannelId, u64>,
}

impl TransactionIngressOwner {
    pub fn new(generation: SessionGeneration) -> Self {
        Self {
            generation,
            next_transaction: 1,
            next_ingress: 1,
            channels: BTreeMap::new(),
        }
    }

    pub fn session_generation(&self) -> reims_vgpu_protocol::SessionGenerationId {
        self.generation.id()
    }

    pub fn has_channel(&self, channel: ChannelId) -> bool {
        self.channels.contains_key(&channel)
    }

    /// Start a semantic generation with fresh channel domains while preserving
    /// the session-wide internal identity spaces. Native epoch owners retain
    /// submitted work across guest reset and therefore must never observe a
    /// reused transaction identity from the successor generation.
    pub(crate) fn successor_generation(&self, generation: SessionGeneration) -> Self {
        Self {
            generation,
            next_transaction: self.next_transaction,
            next_ingress: self.next_ingress,
            channels: BTreeMap::new(),
        }
    }

    pub fn define_channel(&mut self, channel: ChannelId) -> Result<(), TransactionIngressError> {
        if self.channels.contains_key(&channel) {
            return Err(TransactionIngressError::DuplicateChannel);
        }
        self.channels.insert(channel, 1);
        Ok(())
    }

    pub fn retire_channel(&mut self, channel: ChannelId) -> Result<(), TransactionIngressError> {
        self.channels
            .remove(&channel)
            .map(|_| ())
            .ok_or(TransactionIngressError::UnknownChannel)
    }

    pub fn admit<Operation, Lifecycle, Query, Present, Control>(
        &mut self,
        channel: ChannelId,
        prerequisites: impl Into<Box<[TransactionPrerequisite]>>,
        completion_stamp: Option<CompletionStamp>,
        payload: DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>,
    ) -> Result<
        DeviceTransaction<Operation, Lifecycle, Query, Present, Control>,
        TransactionIngressError,
    > {
        let session_generation = self
            .generation
            .try_lease()
            .ok_or(TransactionIngressError::SessionGenerationClosed)?;
        let sequence = *self
            .channels
            .get(&channel)
            .ok_or(TransactionIngressError::UnknownChannel)?;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(TransactionIngressError::ChannelSequenceExhausted)?;
        let next_transaction = self
            .next_transaction
            .checked_add(1)
            .ok_or(TransactionIngressError::IdentitySpaceExhausted)?;
        let next_ingress = self
            .next_ingress
            .checked_add(1)
            .ok_or(TransactionIngressError::IngressSpaceExhausted)?;

        let transaction = DeviceTransaction {
            id: TransactionId::new(self.next_transaction),
            session_generation,
            channel,
            channel_sequence: ChannelSequence::new(sequence),
            ingress_ordinal: IngressOrdinal::new(self.next_ingress),
            prerequisites: prerequisites.into(),
            completion_stamp,
            payload,
        };
        transaction
            .validate_envelope()
            .map_err(TransactionIngressError::Envelope)?;

        self.next_transaction = next_transaction;
        self.next_ingress = next_ingress;
        self.channels.insert(channel, next_sequence);
        Ok(transaction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecTransaction, PublicationPosition};
    use reims_vgpu_protocol::{
        DomainSequence, PublicationDomainId, PublicationSequence, SessionGenerationId,
        SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId,
    };

    type Payload = DeviceTransactionPayload<(), (), (), (), ()>;

    fn owner() -> TransactionIngressOwner {
        TransactionIngressOwner::new(SessionGeneration::new(SessionGenerationId::new(1)))
    }

    fn exec() -> Payload {
        Payload::Exec(ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([]),
            accesses: Box::new([]),
        })
    }

    #[test]
    fn one_admission_derives_every_ordering_position() {
        let mut owner = owner();
        owner.define_channel(ChannelId::new(7)).unwrap();
        let transaction = owner
            .admit(
                ChannelId::new(7),
                Box::<[TransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(7, 9)),
                exec(),
            )
            .unwrap();

        assert_eq!(transaction.id, TransactionId::new(1));
        assert_eq!(transaction.ingress_ordinal, IngressOrdinal::new(1));
        assert_eq!(transaction.channel_sequence, ChannelSequence::new(1));
        assert_eq!(transaction.submission_domain(), SubmissionDomainId::new(7));
        assert_eq!(transaction.domain_sequence(), DomainSequence::new(1));
        assert_eq!(
            transaction.publication(),
            PublicationPosition {
                domain: PublicationDomainId::new(7),
                sequence: PublicationSequence::new(1),
            }
        );
    }

    #[test]
    fn malformed_exec_does_not_consume_any_position() {
        let mut owner = owner();
        owner.define_channel(ChannelId::new(2)).unwrap();
        assert!(matches!(
            owner.admit(
                ChannelId::new(2),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                exec()
            ),
            Err(TransactionIngressError::Envelope(
                TransactionEnvelopeError::ExecMissingCompletionPoint
            ))
        ));
        let transaction = owner
            .admit(
                ChannelId::new(2),
                Box::<[TransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(2, 1)),
                exec(),
            )
            .unwrap();
        assert_eq!(transaction.id, TransactionId::new(1));
        assert_eq!(transaction.channel_sequence, ChannelSequence::new(1));
        assert_eq!(transaction.ingress_ordinal, IngressOrdinal::new(1));
    }

    #[test]
    fn completion_slot_is_the_admitting_fifo_and_refusal_consumes_no_position() {
        let mut owner = owner();
        let channel = ChannelId::new(2);
        owner.define_channel(channel).unwrap();
        assert!(matches!(
            owner.admit(
                channel,
                Box::<[TransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(3, 7)),
                Payload::Control(()),
            ),
            Err(TransactionIngressError::Envelope(
                TransactionEnvelopeError::CompletionSlotDoesNotMatchChannel {
                    channel: rejected,
                    slot: 3
                }
            )) if rejected == channel
        ));
        let admitted = owner
            .admit(
                channel,
                Box::<[TransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 7)),
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(admitted.id, TransactionId::new(1));
        assert_eq!(admitted.channel_sequence, ChannelSequence::new(1));
    }

    #[test]
    fn channels_sequence_independently_and_redefinition_opens_a_new_lifetime() {
        let mut owner = owner();
        owner.define_channel(ChannelId::new(1)).unwrap();
        owner.define_channel(ChannelId::new(2)).unwrap();
        let first = owner
            .admit(
                ChannelId::new(1),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        let independent = owner
            .admit(
                ChannelId::new(2),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(first.channel_sequence, ChannelSequence::new(1));
        assert_eq!(independent.channel_sequence, ChannelSequence::new(1));

        owner.retire_channel(ChannelId::new(1)).unwrap();
        owner.define_channel(ChannelId::new(1)).unwrap();
        let redefined = owner
            .admit(
                ChannelId::new(1),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(redefined.channel_sequence, ChannelSequence::new(1));
        assert_eq!(redefined.ingress_ordinal, IngressOrdinal::new(3));
    }

    #[test]
    fn duplicate_definition_does_not_rewind_a_live_channel() {
        let mut owner = owner();
        owner.define_channel(ChannelId::new(1)).unwrap();
        owner
            .admit(
                ChannelId::new(1),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(
            owner.define_channel(ChannelId::new(1)),
            Err(TransactionIngressError::DuplicateChannel)
        );
        let next = owner
            .admit(
                ChannelId::new(1),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(next.channel_sequence, ChannelSequence::new(2));
    }

    #[test]
    fn closing_the_generation_prevents_new_admission() {
        let generation = SessionGeneration::new(SessionGenerationId::new(1));
        let mut owner = TransactionIngressOwner::new(generation.clone());
        owner.define_channel(ChannelId::new(1)).unwrap();
        generation.close();
        assert!(matches!(
            owner.admit(
                ChannelId::new(1),
                Box::<[TransactionPrerequisite]>::default(),
                None,
                Payload::Control(())
            ),
            Err(TransactionIngressError::SessionGenerationClosed)
        ));
    }
}
