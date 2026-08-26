//! Logical queue submission order after independent recording.
//!
//! Recording may finish in any host order. Each source-queue/FIFO domain
//! nevertheless reaches a queue owner in its contract sequence, while a
//! blocked domain does not hold an independent one. Taking a ready submission
//! does not imply GPU or semantic completion.

use reims_vgpu_protocol::{DomainSequence, SubmissionDomainId, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionOrderError {
    DuplicateTransaction,
    SequenceDidNotIncrease,
    SequenceExhausted,
    UnknownTransaction,
    AlreadyRecorded,
    AlreadyIssued,
    NotIssued,
    UnknownDomain,
    DomainNotDrained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionReady {
    pub transaction: TransactionId,
    pub domain: SubmissionDomainId,
    pub sequence: DomainSequence,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    transaction: TransactionId,
    recorded: bool,
}

#[derive(Clone, Debug, Default)]
struct Domain {
    pending: BTreeMap<DomainSequence, Pending>,
    issued: Option<(DomainSequence, Pending)>,
    last_accepted: Option<DomainSequence>,
}

#[derive(Clone, Debug, Default)]
pub struct SubmissionOrderOwner {
    domains: BTreeMap<SubmissionDomainId, Domain>,
    transactions: BTreeMap<TransactionId, (SubmissionDomainId, DomainSequence)>,
}

impl SubmissionOrderOwner {
    pub fn accept(
        &mut self,
        transaction: TransactionId,
        domain: SubmissionDomainId,
        sequence: DomainSequence,
    ) -> Result<(), SubmissionOrderError> {
        if self.transactions.contains_key(&transaction) {
            return Err(SubmissionOrderError::DuplicateTransaction);
        }
        let state = self.domains.entry(domain).or_default();
        if let Some(last) = state.last_accepted {
            let expected = last
                .get()
                .checked_add(1)
                .ok_or(SubmissionOrderError::SequenceExhausted)?;
            if sequence.get() != expected {
                return Err(SubmissionOrderError::SequenceDidNotIncrease);
            }
        }
        state.last_accepted = Some(sequence);
        state.pending.insert(
            sequence,
            Pending {
                transaction,
                recorded: false,
            },
        );
        self.transactions.insert(transaction, (domain, sequence));
        Ok(())
    }

    pub fn recorded(&mut self, transaction: TransactionId) -> Result<(), SubmissionOrderError> {
        let &(domain, sequence) = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        let state = self
            .domains
            .get_mut(&domain)
            .expect("transaction index names one submission domain");
        let pending = if let Some(pending) = state.pending.get_mut(&sequence) {
            pending
        } else if let Some((issued_sequence, pending)) = state.issued.as_mut() {
            assert_eq!(*issued_sequence, sequence);
            assert_eq!(pending.transaction, transaction);
            pending
        } else {
            unreachable!("transaction index names pending or issued submission")
        };
        if pending.recorded {
            return Err(SubmissionOrderError::AlreadyRecorded);
        }
        pending.recorded = true;
        Ok(())
    }

    /// Reserve one exact source-domain head for a multi-submit native chain.
    ///
    /// Unlike [`Self::take_ready_transaction_if`], this transition does not
    /// require the complete semantic transaction to be recorded. The issued
    /// head continues to hold its source domain until the final native submit
    /// is accepted, and a later [`Self::recorded`] call marks that same issued
    /// transaction complete without releasing the domain.
    pub fn reserve_head_transaction_if(
        &mut self,
        transaction: TransactionId,
        predicate: impl FnOnce(TransactionId) -> bool,
    ) -> Result<Option<SubmissionReady>, SubmissionOrderError> {
        let &(domain, sequence) = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        let state = self
            .domains
            .get(&domain)
            .expect("transaction index names one submission domain");
        if state.issued.is_some() {
            return Err(SubmissionOrderError::AlreadyIssued);
        }
        if !state
            .pending
            .first_key_value()
            .is_some_and(|(&head_sequence, pending)| {
                head_sequence == sequence && pending.transaction == transaction
            })
            || !predicate(transaction)
        {
            return Ok(None);
        }
        let pending = self
            .domains
            .get_mut(&domain)
            .unwrap()
            .pending
            .remove(&sequence)
            .expect("the exact reservable head was validated");
        self.domains.get_mut(&domain).unwrap().issued = Some((sequence, pending));
        Ok(Some(SubmissionReady {
            transaction,
            domain,
            sequence,
        }))
    }

    /// Issue at most one recorded head from each domain. A domain advances
    /// only after its current head is accepted by a native queue owner.
    pub fn take_ready(&mut self) -> Vec<SubmissionReady> {
        self.take_ready_if(|_| true)
    }

    /// Issue recorded domain heads which have also cleared an external
    /// readiness predicate, such as resolution of a wait-before-signal
    /// condition. A blocked head remains pending while independent domains
    /// continue.
    pub fn take_ready_if(
        &mut self,
        predicate: impl Fn(TransactionId) -> bool,
    ) -> Vec<SubmissionReady> {
        let mut ready = Vec::new();
        let domains = self.domains.keys().copied().collect::<Vec<_>>();
        for domain in domains {
            if self.domains[&domain].issued.is_some() {
                continue;
            }
            if let Some((&sequence, pending)) = self.domains[&domain].pending.first_key_value() {
                if !pending.recorded {
                    continue;
                }
                if !predicate(pending.transaction) {
                    continue;
                }
                let pending = self
                    .domains
                    .get_mut(&domain)
                    .unwrap()
                    .pending
                    .remove(&sequence)
                    .unwrap();
                self.domains.get_mut(&domain).unwrap().issued = Some((sequence, pending));
                ready.push(SubmissionReady {
                    transaction: pending.transaction,
                    domain,
                    sequence,
                });
            }
        }
        ready
    }

    /// Issue one exact domain head without also consuming ready work from
    /// unrelated domains. `None` means the transaction is known but is not
    /// yet the recorded, externally-ready head of its own domain.
    pub fn take_ready_transaction_if(
        &mut self,
        transaction: TransactionId,
        predicate: impl FnOnce(TransactionId) -> bool,
    ) -> Result<Option<SubmissionReady>, SubmissionOrderError> {
        let &(domain, sequence) = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        let state = self
            .domains
            .get(&domain)
            .expect("transaction index names one submission domain");
        if state.issued.is_some()
            || !state
                .pending
                .first_key_value()
                .is_some_and(|(&head_sequence, pending)| {
                    head_sequence == sequence
                        && pending.transaction == transaction
                        && pending.recorded
                })
            || !predicate(transaction)
        {
            return Ok(None);
        }
        let pending = self
            .domains
            .get_mut(&domain)
            .unwrap()
            .pending
            .remove(&sequence)
            .expect("the exact ready head was validated");
        self.domains.get_mut(&domain).unwrap().issued = Some((sequence, pending));
        Ok(Some(SubmissionReady {
            transaction,
            domain,
            sequence,
        }))
    }

    pub fn submitted(&mut self, transaction: TransactionId) -> Result<(), SubmissionOrderError> {
        let &(domain, sequence) = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        let issued = self.domains.get(&domain).and_then(|state| state.issued);
        if !issued.is_some_and(|(issued_sequence, pending)| {
            issued_sequence == sequence && pending.transaction == transaction
        }) {
            return Err(SubmissionOrderError::NotIssued);
        }
        self.domains.get_mut(&domain).unwrap().issued = None;
        self.transactions.remove(&transaction);
        Ok(())
    }

    pub fn retire_domain(
        &mut self,
        domain: SubmissionDomainId,
    ) -> Result<(), SubmissionOrderError> {
        let state = self
            .domains
            .get(&domain)
            .ok_or(SubmissionOrderError::UnknownDomain)?;
        if !state.pending.is_empty() || state.issued.is_some() {
            return Err(SubmissionOrderError::DomainNotDrained);
        }
        self.domains.remove(&domain);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accept(owner: &mut SubmissionOrderOwner, id: u64, domain: u64, sequence: u64) {
        owner
            .accept(
                TransactionId::new(id),
                SubmissionDomainId::new(domain),
                DomainSequence::new(sequence),
            )
            .unwrap();
    }

    #[test]
    fn recording_finish_order_cannot_overtake_logical_queue_submission() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 7, 2);
        owner.recorded(TransactionId::new(2)).unwrap();
        assert!(owner.take_ready().is_empty());
        owner.recorded(TransactionId::new(1)).unwrap();
        assert_eq!(
            owner
                .take_ready()
                .into_iter()
                .map(|ready| ready.transaction)
                .collect::<Vec<_>>(),
            vec![TransactionId::new(1)]
        );
        owner.submitted(TransactionId::new(1)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(2));
    }

    #[test]
    fn an_unrecorded_domain_does_not_hold_an_independent_domain() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 8, 1);
        owner.recorded(TransactionId::new(2)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(2));
    }

    #[test]
    fn a_reserved_chain_head_stays_issued_until_final_recording_and_submission() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 7, 2);

        let reserved = owner
            .reserve_head_transaction_if(TransactionId::new(1), |_| true)
            .unwrap()
            .unwrap();
        assert_eq!(reserved.transaction, TransactionId::new(1));
        assert!(owner.take_ready().is_empty());
        assert_eq!(
            owner.reserve_head_transaction_if(TransactionId::new(2), |_| true),
            Err(SubmissionOrderError::AlreadyIssued)
        );

        owner.recorded(TransactionId::new(1)).unwrap();
        assert!(owner.take_ready().is_empty());
        owner.submitted(TransactionId::new(1)).unwrap();
        owner.recorded(TransactionId::new(2)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(2));
    }

    #[test]
    fn taking_one_exact_head_does_not_issue_another_ready_domain() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 8, 1);
        owner.recorded(TransactionId::new(1)).unwrap();
        owner.recorded(TransactionId::new(2)).unwrap();
        assert_eq!(
            owner
                .take_ready_transaction_if(TransactionId::new(2), |_| true)
                .unwrap()
                .unwrap()
                .transaction,
            TransactionId::new(2)
        );
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(1));
    }

    #[test]
    fn domain_reuse_requires_explicit_drained_retirement() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 9);
        owner.recorded(TransactionId::new(1)).unwrap();
        owner.take_ready();
        owner.submitted(TransactionId::new(1)).unwrap();
        assert_eq!(
            owner.accept(
                TransactionId::new(2),
                SubmissionDomainId::new(7),
                DomainSequence::new(1),
            ),
            Err(SubmissionOrderError::SequenceDidNotIncrease)
        );
        owner.retire_domain(SubmissionDomainId::new(7)).unwrap();
        accept(&mut owner, 2, 7, 1);
    }
}
