//! Ordered guest publication without device-global head-of-line blocking.
//!
//! Completion is an immutable fact and can arrive in any host order. The sole
//! owner for each proven publication domain releases only its next position;
//! a blocked domain does not prevent another domain from publishing. This
//! owner does not infer publication domains from FIFO channels or Vulkan
//! queues: callers supply identities established by the API contract.

use reims_vgpu_protocol::{PublicationDomainId, PublicationSequence, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionStamp {
    pub slot: u32,
    pub value: u32,
}

impl CompletionStamp {
    pub const fn new(slot: u32, value: u32) -> Self {
        Self { slot, value }
    }
}

/// Number of four-byte completion words owned by one guest stamp page.
pub fn completion_stamp_slot_count(page_bytes: u64) -> u32 {
    (page_bytes / size_of::<u32>() as u64).min(u64::from(u32::MAX)) as u32
}

/// Byte offset of one completion word inside the explicit guest stamp page.
pub fn completion_stamp_slot_offset(slot: u32, page_bytes: u64) -> Option<u64> {
    (slot < completion_stamp_slot_count(page_bytes))
        .then_some(u64::from(slot) * size_of::<u32>() as u64)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicationPosition {
    pub domain: PublicationDomainId,
    pub sequence: PublicationSequence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationFact<T> {
    pub transaction: TransactionId,
    pub semantic: T,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedFact<T> {
    pub transaction: TransactionId,
    pub position: PublicationPosition,
    pub completion_stamp: Option<CompletionStamp>,
    pub semantic: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublicationError {
    DuplicateTransaction,
    SequenceDidNotIncrease,
    SequenceExhausted,
    UnknownTransaction,
    AlreadyCompleted,
    UnknownDomain,
    DomainNotDrained,
}

#[derive(Clone, Debug)]
struct Position<T> {
    transaction: TransactionId,
    completion_stamp: Option<CompletionStamp>,
    completed: Option<T>,
}

#[derive(Clone, Debug)]
struct Domain<T> {
    positions: BTreeMap<PublicationSequence, Position<T>>,
    next_to_publish: Option<PublicationSequence>,
    last_registered: Option<PublicationSequence>,
}

impl<T> Default for Domain<T> {
    fn default() -> Self {
        Self {
            positions: BTreeMap::new(),
            next_to_publish: None,
            last_registered: None,
        }
    }
}

/// Single semantic-publication owner parameterized by immutable completion
/// payload. It invokes no callbacks; returned facts are applied by the
/// authorized guest/host-action owner.
#[derive(Clone, Debug)]
pub struct PublicationOwner<T> {
    domains: BTreeMap<PublicationDomainId, Domain<T>>,
    transactions: BTreeMap<TransactionId, PublicationPosition>,
}

impl<T> Default for PublicationOwner<T> {
    fn default() -> Self {
        Self {
            domains: BTreeMap::new(),
            transactions: BTreeMap::new(),
        }
    }
}

impl<T> PublicationOwner<T> {
    /// Ends a publication-domain lifetime after all of its registered facts
    /// have published. A later registration for the same numeric identity is
    /// then the first position of a new lifetime and need not continue the
    /// former sequence.
    pub fn retire_domain(&mut self, domain: PublicationDomainId) -> Result<(), PublicationError> {
        let state = self
            .domains
            .get(&domain)
            .ok_or(PublicationError::UnknownDomain)?;
        if !state.positions.is_empty() {
            return Err(PublicationError::DomainNotDrained);
        }
        self.domains.remove(&domain);
        Ok(())
    }

    pub fn register(
        &mut self,
        transaction: TransactionId,
        position: PublicationPosition,
        completion_stamp: Option<CompletionStamp>,
    ) -> Result<(), PublicationError> {
        if self.transactions.contains_key(&transaction) {
            return Err(PublicationError::DuplicateTransaction);
        }
        let domain = self.domains.entry(position.domain).or_default();
        if let Some(last) = domain.last_registered {
            let expected = last
                .get()
                .checked_add(1)
                .ok_or(PublicationError::SequenceExhausted)?;
            if position.sequence.get() != expected {
                return Err(PublicationError::SequenceDidNotIncrease);
            }
        }
        domain.next_to_publish.get_or_insert(position.sequence);
        domain.last_registered = Some(position.sequence);
        domain.positions.insert(
            position.sequence,
            Position {
                transaction,
                completion_stamp,
                completed: None,
            },
        );
        self.transactions.insert(transaction, position);
        Ok(())
    }

    pub fn complete(&mut self, fact: PublicationFact<T>) -> Result<(), PublicationError> {
        let position = *self
            .transactions
            .get(&fact.transaction)
            .ok_or(PublicationError::UnknownTransaction)?;
        let entry = self
            .domains
            .get_mut(&position.domain)
            .and_then(|domain| domain.positions.get_mut(&position.sequence))
            .expect("registered transaction has exactly one publication position");
        if entry.completed.is_some() {
            return Err(PublicationError::AlreadyCompleted);
        }
        entry.completed = Some(fact.semantic);
        Ok(())
    }

    pub fn publish_ready(&mut self) -> Vec<PublishedFact<T>> {
        let mut published = Vec::new();
        let domain_ids: Vec<_> = self.domains.keys().copied().collect();
        for domain_id in domain_ids {
            while let Some(sequence) = self
                .domains
                .get(&domain_id)
                .and_then(|domain| domain.next_to_publish)
            {
                let ready = self
                    .domains
                    .get(&domain_id)
                    .and_then(|domain| domain.positions.get(&sequence))
                    .is_some_and(|position| position.completed.is_some());
                if !ready {
                    break;
                }
                let domain = self.domains.get_mut(&domain_id).unwrap();
                let position = domain.positions.remove(&sequence).unwrap();
                let semantic = position.completed.unwrap();
                self.transactions.remove(&position.transaction);
                domain.next_to_publish = domain.positions.first_key_value().map(|(key, _)| *key);
                published.push(PublishedFact {
                    transaction: position.transaction,
                    position: PublicationPosition {
                        domain: domain_id,
                        sequence,
                    },
                    completion_stamp: position.completion_stamp,
                    semantic,
                });
            }
        }
        published
    }

    pub fn pending(&self) -> usize {
        self.transactions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_word_geometry_is_bounded_by_the_explicit_guest_page() {
        assert_eq!(completion_stamp_slot_count(4096), 1024);
        assert_eq!(completion_stamp_slot_offset(1023, 4096), Some(4092));
        assert_eq!(completion_stamp_slot_offset(1024, 4096), None);
        assert_eq!(completion_stamp_slot_count(16384), 4096);
        assert_eq!(completion_stamp_slot_offset(4095, 16384), Some(16380));
    }

    fn position(domain: u64, sequence: u64) -> PublicationPosition {
        PublicationPosition {
            domain: PublicationDomainId::new(domain),
            sequence: PublicationSequence::new(sequence),
        }
    }

    fn register(owner: &mut PublicationOwner<&'static str>, id: u64, domain: u64, sequence: u64) {
        owner
            .register(TransactionId::new(id), position(domain, sequence), None)
            .unwrap();
    }

    fn complete(owner: &mut PublicationOwner<&'static str>, id: u64, semantic: &'static str) {
        owner
            .complete(PublicationFact {
                transaction: TransactionId::new(id),
                semantic,
            })
            .unwrap();
    }

    #[test]
    fn same_domain_completion_cannot_overtake_publication() {
        let mut owner = PublicationOwner::default();
        register(&mut owner, 1, 7, 1);
        register(&mut owner, 2, 7, 2);
        complete(&mut owner, 2, "second");
        assert!(owner.publish_ready().is_empty());
        complete(&mut owner, 1, "first");
        let published = owner.publish_ready();
        assert_eq!(
            published
                .iter()
                .map(|fact| (fact.transaction.get(), fact.semantic))
                .collect::<Vec<_>>(),
            vec![(1, "first"), (2, "second")]
        );
    }

    #[test]
    fn blocked_publication_domain_does_not_block_an_independent_domain() {
        let mut owner = PublicationOwner::default();
        register(&mut owner, 1, 7, 1);
        register(&mut owner, 2, 7, 2);
        register(&mut owner, 3, 8, 1);
        complete(&mut owner, 2, "blocked");
        complete(&mut owner, 3, "independent");
        let published = owner.publish_ready();
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].transaction.get(), 3);
        assert_eq!(owner.pending(), 2);
    }

    #[test]
    fn completion_stamp_stays_attached_to_its_transaction_fact() {
        let mut owner = PublicationOwner::default();
        owner
            .register(
                TransactionId::new(4),
                position(1, 1),
                Some(CompletionStamp::new(0xffff_0002, u32::MAX)),
            )
            .unwrap();
        complete(&mut owner, 4, "done");
        let published = owner.publish_ready().pop().unwrap();
        assert_eq!(
            published.completion_stamp,
            Some(CompletionStamp {
                slot: 0xffff_0002,
                value: u32::MAX,
            })
        );
    }

    #[test]
    fn duplicate_and_gapped_positions_are_typed() {
        let mut owner = PublicationOwner::<()>::default();
        owner
            .register(TransactionId::new(1), position(1, 4), None)
            .unwrap();
        assert_eq!(
            owner.register(TransactionId::new(1), position(2, 1), None),
            Err(PublicationError::DuplicateTransaction)
        );
        assert_eq!(
            owner.register(TransactionId::new(2), position(1, 6), None),
            Err(PublicationError::SequenceDidNotIncrease)
        );
    }

    #[test]
    fn a_drained_domain_retains_its_contract_sequence() {
        let mut owner = PublicationOwner::default();
        register(&mut owner, 1, 3, 9);
        complete(&mut owner, 1, "done");
        assert_eq!(owner.publish_ready().len(), 1);
        assert_eq!(
            owner.register(TransactionId::new(2), position(3, 11), None),
            Err(PublicationError::SequenceDidNotIncrease)
        );
        owner
            .register(TransactionId::new(2), position(3, 10), None)
            .unwrap();
    }

    #[test]
    fn a_pending_domain_cannot_be_retired() {
        let mut owner = PublicationOwner::<()>::default();
        owner
            .register(TransactionId::new(1), position(3, 1), None)
            .unwrap();
        assert_eq!(
            owner.retire_domain(PublicationDomainId::new(3)),
            Err(PublicationError::DomainNotDrained)
        );
        assert_eq!(owner.pending(), 1);
    }

    #[test]
    fn retiring_a_drained_domain_ends_its_sequence_lifetime() {
        let mut owner = PublicationOwner::default();
        register(&mut owner, 1, 3, 9);
        complete(&mut owner, 1, "done");
        assert_eq!(owner.publish_ready().len(), 1);
        owner.retire_domain(PublicationDomainId::new(3)).unwrap();
        owner
            .register(TransactionId::new(2), position(3, 1), None)
            .unwrap();
    }

    #[test]
    fn cpu_only_fact_publishes_without_a_gpu_state_or_submission() {
        let mut owner = PublicationOwner::default();
        register(&mut owner, 5, 1, 1);
        complete(&mut owner, 5, "control effect complete");
        let fact = owner.publish_ready().pop().unwrap();
        assert_eq!(fact.semantic, "control effect complete");
        assert_eq!(owner.pending(), 0);
    }
}
