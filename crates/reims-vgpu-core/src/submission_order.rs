//! Logical queue submission order after independent recording.
//!
//! Recording may finish in any host order. Each source-queue/FIFO domain
//! nevertheless reaches a queue owner in its contract sequence, while a
//! blocked domain does not hold an independent one. Taking a ready submission
//! does not imply GPU or semantic completion.
//!
//! A transaction keeps its (domain, sequence) identity from acceptance until
//! [`SubmissionOrderOwner::retire`], not until it submits. Submission releases
//! the *domain head* so the next transaction can issue; it does not end the
//! question [`SubmissionOrderOwner::relation`] answers. A submitted producer is
//! precisely the case a consumer asks about — "did the work that writes this
//! content already reach the queue ahead of me?" — and dropping the entry at
//! submission made that question answer `UnknownTransaction` for every producer
//! still in flight, which is the only state in which the answer matters.

use reims_vgpu_protocol::{DomainSequence, SubmissionDomainId, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionOrderError {
    DuplicateTransaction,
    SequenceExhausted,
    UnknownTransaction,
    AlreadyRecorded,
    AlreadyIssued,
    AlreadySubmitted,
    NotIssued,
    UnknownDomain,
    DomainNotDrained,
    AlreadyAbandoned,
    StillHoldsDomain,
    /// A predecessor holding a domain position that this owner's index does
    /// not track. Nothing can name it, so nothing can release it.
    IssuedHeadUntracked(TransactionId),
}

/// A transaction ahead of another in one domain that has not submitted, and
/// the position of the one that asked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsubmittedPredecessor {
    pub transaction: TransactionId,
    pub domain: SubmissionDomainId,
    pub sequence: DomainSequence,
    /// Where the transaction that asked sits in the same domain.
    pub behind: DomainSequence,
}

/// One tracked transaction's position and progress, for the census.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionOrderEntry {
    pub transaction: TransactionId,
    pub domain: SubmissionDomainId,
    pub sequence: DomainSequence,
    pub recorded: bool,
    pub issued: bool,
    pub submitted: bool,
    pub abandoned: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionReady {
    pub transaction: TransactionId,
    pub domain: SubmissionDomainId,
    pub sequence: DomainSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionOrderRelation {
    Same,
    Before,
    After,
    Independent,
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

/// Where one accepted transaction sits in its domain, for as long as any
/// consumer may still ask about it.
#[derive(Clone, Copy, Debug)]
struct TransactionOrder {
    domain: SubmissionDomainId,
    sequence: DomainSequence,
    submitted: bool,
    abandoned: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SubmissionOrderOwner {
    domains: BTreeMap<SubmissionDomainId, Domain>,
    transactions: BTreeMap<TransactionId, TransactionOrder>,
}

impl SubmissionOrderOwner {
    pub fn relation(
        &self,
        first: TransactionId,
        second: TransactionId,
    ) -> Result<SubmissionOrderRelation, SubmissionOrderError> {
        let &first_order = self
            .transactions
            .get(&first)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        let &second_order = self
            .transactions
            .get(&second)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if first == second {
            return Ok(SubmissionOrderRelation::Same);
        }
        if first_order.domain != second_order.domain {
            return Ok(SubmissionOrderRelation::Independent);
        }
        Ok(if first_order.sequence < second_order.sequence {
            SubmissionOrderRelation::Before
        } else {
            SubmissionOrderRelation::After
        })
    }

    /// Whether this owner is the one tracking `transaction`'s order.
    ///
    /// A transaction's order lives in exactly one owner, so a device holding a
    /// runtime per session generation has exactly one runtime that can answer
    /// an order question about it. Asking the wrong one does not fail -- it
    /// answers about a domain the transaction is not in -- so a caller has to
    /// select before it asks.
    #[must_use]
    pub fn tracks(&self, transaction: TransactionId) -> bool {
        self.transactions.contains_key(&transaction)
    }

    /// The nearest transaction ahead of `transaction` in its own domain that
    /// has not submitted yet.
    ///
    /// A domain submits in sequence order, so any work a later transaction
    /// claims exclusively -- an image's one prepared transition, above all --
    /// is claimed ahead of a transaction that will submit first. That
    /// inversion is not a slow path, it is a cycle: the earlier transaction
    /// cannot prepare because the later one holds the claim, and the later one
    /// cannot submit or cancel because the earlier one holds the domain head.
    /// Neither side can move and nothing times out.
    ///
    /// **A merely pending predecessor counts, and this is the whole point.**
    /// An earlier rule asked only about the *issued* head, reasoning that a
    /// pending predecessor holds no claim and no domain and so is in no cycle.
    /// It holds no claim *yet*. A driven boot formed the cycle in the order the
    /// narrower rule allows: with nothing issued, a transaction at sequence 58
    /// took an image; the transaction at sequence 56 then became the head,
    /// needed that image, and neither could move again. The claim has to be
    /// ordered against the order the domain will submit in, not against
    /// whoever happens to hold the head at the moment of asking.
    ///
    /// The positions come back with the answer because a reading that
    /// disagrees with the census has to be able to say which domain it read.
    pub fn unsubmitted_predecessor(
        &self,
        transaction: TransactionId,
    ) -> Result<Option<UnsubmittedPredecessor>, SubmissionOrderError> {
        let &TransactionOrder {
            domain, sequence, ..
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        let state = self
            .domains
            .get(&domain)
            .expect("transaction index names one submission domain");
        // The issued head is removed from `pending`, so the two have to be
        // compared against each other: whichever sits at the higher sequence
        // below this one is the nearest predecessor.
        let pending = state
            .pending
            .range(..sequence)
            .next_back()
            .map(|(&sequence, pending)| (sequence, pending.transaction));
        let issued = state
            .issued
            .as_ref()
            .filter(|(issued_sequence, _)| *issued_sequence < sequence)
            .map(|&(sequence, pending)| (sequence, pending.transaction));
        let Some((predecessor_sequence, predecessor)) = [pending, issued]
            .into_iter()
            .flatten()
            .max_by_key(|&(sequence, _)| sequence)
        else {
            return Ok(None);
        };
        // A predecessor this owner does not track is not an order, it is
        // corruption: nothing can name it, so nothing can release it, and
        // every claim behind it would wait for a transaction that cannot move
        // again. Saying so is the caller's only chance to tell that apart from
        // an ordinary wait.
        if !self.transactions.contains_key(&predecessor) {
            return Err(SubmissionOrderError::IssuedHeadUntracked(predecessor));
        }
        Ok(Some(UnsubmittedPredecessor {
            transaction: predecessor,
            domain,
            sequence: predecessor_sequence,
            behind: sequence,
        }))
    }

    /// Take the next position in `domain` for `transaction`.
    ///
    /// The sequence is assigned here rather than supplied, because this owner
    /// is the only thing that knows which population it numbers. It accepts
    /// exactly the transactions that record and submit GPU work, and a
    /// transaction's channel position counts *every* packet the channel
    /// carried -- a validity invalidation and a resource synchronize among
    /// them. Handing this the channel position therefore made the domain's
    /// numbering skip on the first non-recording packet a channel carried, and
    /// a contiguity check against a counter over a different population then
    /// refused every later transaction on that channel forever. The channel's
    /// own position is still the right thing for publication, which registers
    /// every packet; it was never the right thing for this.
    ///
    /// Assigning removes the question rather than answering it: there is no
    /// supplied value left to disagree with, so the numbering is contiguous by
    /// construction and no caller can get it wrong.
    pub fn accept(
        &mut self,
        transaction: TransactionId,
        domain: SubmissionDomainId,
    ) -> Result<DomainSequence, SubmissionOrderError> {
        if self.transactions.contains_key(&transaction) {
            return Err(SubmissionOrderError::DuplicateTransaction);
        }
        let state = self.domains.entry(domain).or_default();
        let sequence = match state.last_accepted {
            Some(last) => DomainSequence::new(
                last.get()
                    .checked_add(1)
                    .ok_or(SubmissionOrderError::SequenceExhausted)?,
            ),
            None => DomainSequence::new(1),
        };
        state.last_accepted = Some(sequence);
        state.pending.insert(
            sequence,
            Pending {
                transaction,
                recorded: false,
            },
        );
        self.transactions.insert(
            transaction,
            TransactionOrder {
                domain,
                sequence,
                submitted: false,
                abandoned: false,
            },
        );
        Ok(sequence)
    }

    pub fn recorded(&mut self, transaction: TransactionId) -> Result<(), SubmissionOrderError> {
        let &TransactionOrder {
            domain,
            sequence,
            submitted,
            ..
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if submitted {
            return Err(SubmissionOrderError::AlreadySubmitted);
        }
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
        let &TransactionOrder {
            domain,
            sequence,
            submitted,
            ..
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if submitted {
            return Err(SubmissionOrderError::AlreadySubmitted);
        }
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
    /// Where every tracked transaction sits, for the census.
    ///
    /// A blocked head names the transaction it is waiting for, and the next
    /// question is always what *that* transaction is waiting for -- which no
    /// counter answers, because a count of live recordings cannot say whether
    /// one of them is the producer somebody is parked on. This reports the
    /// per-transaction state so the two readings can be joined.
    /// One transaction's order entry, without building the whole census.
    ///
    /// [`Self::census`] allocates a vector of every tracked transaction, which
    /// is the wrong shape for a caller that already knows which transaction it
    /// is asking about -- a refusal path asking once per retry is asking
    /// thousands of times a second.
    pub fn entry(&self, transaction: TransactionId) -> Option<SubmissionOrderEntry> {
        let order = self.transactions.get(&transaction)?;
        let state = self.domains.get(&order.domain);
        Some(SubmissionOrderEntry {
            transaction,
            domain: order.domain,
            sequence: order.sequence,
            submitted: order.submitted,
            abandoned: order.abandoned,
            recorded: state.is_some_and(|state| {
                state
                    .pending
                    .get(&order.sequence)
                    .or_else(|| {
                        state
                            .issued
                            .as_ref()
                            .filter(|(sequence, _)| *sequence == order.sequence)
                            .map(|(_, pending)| pending)
                    })
                    .is_some_and(|pending| pending.recorded)
            }),
            issued: state
                .and_then(|state| state.issued.as_ref())
                .is_some_and(|(sequence, _)| *sequence == order.sequence),
        })
    }

    /// Every tracked transaction's [`Self::entry`], in transaction order.
    pub fn census(&self) -> Vec<SubmissionOrderEntry> {
        self.transactions
            .keys()
            .filter_map(|&transaction| self.entry(transaction))
            .collect()
    }

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
        let &TransactionOrder {
            domain,
            sequence,
            submitted,
            ..
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if submitted {
            return Err(SubmissionOrderError::AlreadySubmitted);
        }
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

    /// Release the domain head this transaction was holding. The transaction
    /// keeps its order identity until [`Self::retire`]: it is exactly while a
    /// submission is in flight that a consumer needs to know it went first.
    pub fn submitted(&mut self, transaction: TransactionId) -> Result<(), SubmissionOrderError> {
        let &TransactionOrder {
            domain,
            sequence,
            submitted,
            ..
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if submitted {
            return Err(SubmissionOrderError::AlreadySubmitted);
        }
        let issued = self.domains.get(&domain).and_then(|state| state.issued);
        if !issued.is_some_and(|(issued_sequence, pending)| {
            issued_sequence == sequence && pending.transaction == transaction
        }) {
            return Err(SubmissionOrderError::NotIssued);
        }
        self.domains.get_mut(&domain).unwrap().issued = None;
        self.transactions
            .get_mut(&transaction)
            .expect("the order entry was just read")
            .submitted = true;
        Ok(())
    }

    /// Release every order claim of a transaction that will never submit.
    ///
    /// [`Self::retire`] ends an order identity that has already been through
    /// [`Self::submitted`], so it drops the index entry and leaves the domain
    /// alone -- by then the domain head has moved on. A transaction that
    /// reaches a terminal refusal before submission still holds one, either
    /// pending at its own sequence or issued as its domain's head, and nothing
    /// ever releases it: every later transaction on that domain is refused
    /// behind it for the life of the device.
    ///
    /// This is the terminal transition for that case: the domain claim goes and
    /// the next transaction becomes the head. The order *identity* survives to
    /// [`Self::retire`] exactly as a submitted one does, so a consumer that
    /// already named this transaction as a producer is still told where it sat
    /// rather than being handed `UnknownTransaction`.
    ///
    /// `last_accepted` deliberately survives: abandoning work does not make its
    /// sequence available again, and a domain whose numbering rewound would
    /// admit a successor as the head of an order it was not accepted into.
    pub fn abandon(&mut self, transaction: TransactionId) -> Result<(), SubmissionOrderError> {
        let &TransactionOrder {
            domain,
            sequence,
            submitted,
            abandoned,
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if submitted {
            return Err(SubmissionOrderError::AlreadySubmitted);
        }
        if abandoned {
            return Err(SubmissionOrderError::AlreadyAbandoned);
        }
        let state = self
            .domains
            .get_mut(&domain)
            .expect("transaction index names one submission domain");
        if state.pending.remove(&sequence).is_none() {
            let issued = state.issued.take();
            let Some((issued_sequence, pending)) = issued else {
                unreachable!("an unsubmitted transaction is pending or issued")
            };
            assert_eq!(issued_sequence, sequence);
            assert_eq!(pending.transaction, transaction);
        }
        self.transactions
            .get_mut(&transaction)
            .expect("the order entry was just read")
            .abandoned = true;
        Ok(())
    }

    /// Forget one transaction's order identity. Called from transaction
    /// retirement, which is the point at which no accepted successor can still
    /// name it.
    ///
    /// A transaction that still holds its domain claim is refused. Dropping the
    /// index entry does not drop the claim, so retiring one that never reached
    /// [`Self::submitted`] or [`Self::abandon`] leaves a `pending` entry no
    /// caller can name any more: the domain head is held forever, `take_ready`
    /// never issues behind it, [`Self::retire_domain`] answers
    /// `DomainNotDrained` for the life of the device, and the census cannot
    /// show any of it because the census iterates the index this just removed.
    ///
    /// Refusing here is what makes that unrepresentable. The two terminal
    /// transitions each release the claim, so a caller reaching this with a
    /// live one has skipped a step rather than found a special case.
    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), SubmissionOrderError> {
        let &TransactionOrder {
            submitted,
            abandoned,
            ..
        } = self
            .transactions
            .get(&transaction)
            .ok_or(SubmissionOrderError::UnknownTransaction)?;
        if !submitted && !abandoned {
            return Err(SubmissionOrderError::StillHoldsDomain);
        }
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
        self.transactions.retain(|_, order| order.domain != domain);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accept and assert the position the owner chose.
    ///
    /// `sequence` is no longer an input, so every existing case that named the
    /// position it expected now checks that the assignment is the contiguous
    /// one -- which is the property that used to be a runtime refusal.
    fn accept(owner: &mut SubmissionOrderOwner, id: u64, domain: u64, sequence: u64) {
        assert_eq!(
            owner
                .accept(TransactionId::new(id), SubmissionDomainId::new(domain))
                .unwrap(),
            DomainSequence::new(sequence),
        );
    }

    /// A transaction that fails before submission must not hold its domain.
    #[test]
    fn an_abandoned_pending_transaction_releases_the_domain_to_its_successor() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 7, 2);
        owner.recorded(TransactionId::new(2)).unwrap();
        assert!(
            owner.take_ready().is_empty(),
            "the unrecorded head blocks its successor"
        );
        owner.abandon(TransactionId::new(1)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(2));
        assert_eq!(
            owner.relation(TransactionId::new(1), TransactionId::new(2)),
            Ok(SubmissionOrderRelation::Before),
            "the order identity survives abandonment, as it survives submission"
        );
        assert_eq!(
            owner.abandon(TransactionId::new(1)),
            Err(SubmissionOrderError::AlreadyAbandoned)
        );
        owner.retire(TransactionId::new(1)).unwrap();
    }

    /// A refusal that lands after the head was issued releases the same claim.
    #[test]
    fn an_abandoned_issued_head_releases_the_domain_to_its_successor() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 7, 2);
        owner.recorded(TransactionId::new(1)).unwrap();
        owner.recorded(TransactionId::new(2)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(1));
        assert!(owner.take_ready().is_empty());
        owner.abandon(TransactionId::new(1)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(2));
    }

    /// Abandonment is a pre-submission transition. A submitted transaction owns
    /// a timeline point, and forgetting it would answer the order question with
    /// `UnknownTransaction` for work that is still in flight.
    #[test]
    fn a_submitted_transaction_cannot_be_abandoned() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        owner.recorded(TransactionId::new(1)).unwrap();
        owner.take_ready();
        owner.submitted(TransactionId::new(1)).unwrap();
        assert_eq!(
            owner.abandon(TransactionId::new(1)),
            Err(SubmissionOrderError::AlreadySubmitted)
        );
        owner.retire(TransactionId::new(1)).unwrap();
    }

    /// Abandoning the last claim leaves the domain drained rather than stuck
    /// reporting `DomainNotDrained` for the life of the device.
    #[test]
    fn an_abandoned_domain_drains() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        assert_eq!(
            owner.retire_domain(SubmissionDomainId::new(7)),
            Err(SubmissionOrderError::DomainNotDrained)
        );
        owner.abandon(TransactionId::new(1)).unwrap();
        owner.retire(TransactionId::new(1)).unwrap();
        owner.retire_domain(SubmissionDomainId::new(7)).unwrap();
    }

    /// An abandoned sequence is not reusable: numbering only ever moves
    /// forward, so the successor takes the next position and not the abandoned
    /// one, even though nothing occupies it any more.
    #[test]
    fn abandonment_does_not_rewind_domain_numbering() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        owner.abandon(TransactionId::new(1)).unwrap();
        owner.retire(TransactionId::new(1)).unwrap();
        accept(&mut owner, 3, 7, 2);
        owner.recorded(TransactionId::new(3)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(3));
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

    /// A consumer asking about a producer that is already on the queue gets an
    /// order, not `UnknownTransaction`.
    ///
    /// This is the only state in which the question is ever asked: content
    /// readiness looks up the producer of a *pending* GPU write, and a write is
    /// pending precisely because its producer has submitted and not yet
    /// completed. Releasing the order identity at submission answered every one
    /// of those with an error, which the readiness check reports as a refusal
    /// and retries forever.
    #[test]
    fn a_submitted_producer_still_answers_the_order_question() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 88, 7, 1);
        accept(&mut owner, 89, 7, 2);
        owner.recorded(TransactionId::new(88)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(88));
        owner.submitted(TransactionId::new(88)).unwrap();

        assert_eq!(
            owner.relation(TransactionId::new(88), TransactionId::new(89)),
            Ok(SubmissionOrderRelation::Before),
            "the producer submitted first and the consumer must be told so"
        );
        // The head it was holding is free, so the consumer can still issue.
        owner.recorded(TransactionId::new(89)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(89));

        // And retirement is what ends the question.
        owner.retire(TransactionId::new(88)).unwrap();
        assert_eq!(
            owner.relation(TransactionId::new(88), TransactionId::new(89)),
            Err(SubmissionOrderError::UnknownTransaction)
        );
    }

    /// A submitted transaction has released its domain head and may not be
    /// re-recorded, re-issued, or submitted twice through its surviving entry.
    #[test]
    fn a_submitted_transaction_refuses_every_pre_submission_transition() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 3, 1);
        owner.recorded(TransactionId::new(1)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(1));
        owner.submitted(TransactionId::new(1)).unwrap();
        assert_eq!(
            owner.recorded(TransactionId::new(1)),
            Err(SubmissionOrderError::AlreadySubmitted)
        );
        assert_eq!(
            owner.submitted(TransactionId::new(1)),
            Err(SubmissionOrderError::AlreadySubmitted)
        );
        assert_eq!(
            owner.take_ready_transaction_if(TransactionId::new(1), |_| true),
            Err(SubmissionOrderError::AlreadySubmitted)
        );
        assert_eq!(
            owner.reserve_head_transaction_if(TransactionId::new(1), |_| true),
            Err(SubmissionOrderError::AlreadySubmitted)
        );
    }

    #[test]
    fn relation_uses_contract_domain_sequence_not_transaction_identity() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 90, 7, 1);
        accept(&mut owner, 91, 7, 2);
        // A higher transaction id at a lower sequence is still earlier.
        accept(&mut owner, 20, 7, 3);
        assert_eq!(
            owner.relation(TransactionId::new(90), TransactionId::new(91)),
            Ok(SubmissionOrderRelation::Before)
        );
        assert_eq!(
            owner.relation(TransactionId::new(20), TransactionId::new(91)),
            Ok(SubmissionOrderRelation::After)
        );
        assert_eq!(
            owner.relation(TransactionId::new(90), TransactionId::new(90)),
            Ok(SubmissionOrderRelation::Same)
        );
        // A different domain orders nothing.
        accept(&mut owner, 92, 8, 1);
        assert_eq!(
            owner.relation(TransactionId::new(90), TransactionId::new(92)),
            Ok(SubmissionOrderRelation::Independent)
        );
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
        accept(&mut owner, 1, 7, 1);
        owner.recorded(TransactionId::new(1)).unwrap();
        owner.take_ready();
        owner.submitted(TransactionId::new(1)).unwrap();
        // Until the domain is retired its numbering carries on from where it
        // was. Retirement is what restarts it at the first position.
        accept(&mut owner, 2, 7, 2);
        owner.recorded(TransactionId::new(2)).unwrap();
        owner.take_ready();
        owner.submitted(TransactionId::new(2)).unwrap();
        owner.retire(TransactionId::new(1)).unwrap();
        owner.retire(TransactionId::new(2)).unwrap();
        owner.retire_domain(SubmissionDomainId::new(7)).unwrap();
        accept(&mut owner, 3, 7, 1);
    }

    /// The predecessor query sees the issued head, which `pending` no longer
    /// holds.
    ///
    /// This is the shape that deadlocked a boot: 539 reserved its domain head
    /// and then needed an image, 541 had already claimed that image, and 541
    /// could not submit or cancel because 539 held the head. The query has to
    /// answer for the *issued* transaction as well as the pending ones, or a
    /// gate built on it opens at exactly the moment the cycle forms.
    #[test]
    fn an_issued_head_is_still_an_unsubmitted_predecessor() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 539, 1, 1);
        accept(&mut owner, 540, 1, 2);
        accept(&mut owner, 541, 1, 3);
        accept(&mut owner, 546, 4, 1);

        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(539)),
            Ok(None),
            "the domain head has nothing ahead of it"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(541)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(540),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(2),
                behind: DomainSequence::new(3),
            })),
            "the nearest predecessor is named, not merely reported to exist"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(546)),
            Ok(None),
            "another domain orders nothing here"
        );

        owner
            .reserve_head_transaction_if(TransactionId::new(539), |_| true)
            .unwrap()
            .unwrap();
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(541)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(540),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(2),
                behind: DomainSequence::new(3),
            }))
        );
        owner.abandon(TransactionId::new(540)).unwrap();
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(541)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(539),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(1),
                behind: DomainSequence::new(3),
            })),
            "the issued head is invisible to a scan of the pending map"
        );

        owner.submitted(TransactionId::new(539)).unwrap();
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(541)),
            Ok(None),
            "a submitted predecessor no longer holds the claim order"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(9_999)),
            Err(SubmissionOrderError::UnknownTransaction)
        );
    }

    /// Retiring a transaction that never submitted must not strand its domain.
    ///
    /// `retire` drops the index entry, and the index is the only name anyone
    /// has for the claim. A transaction retired while still pending therefore
    /// leaves a `pending` entry that cannot be reached, cannot be abandoned,
    /// and holds the domain head for the life of the device -- and the census
    /// cannot report it, because the census iterates the index.
    #[test]
    fn retiring_a_transaction_that_still_holds_its_domain_is_refused() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 1, 7, 1);
        accept(&mut owner, 2, 7, 2);
        assert_eq!(
            owner.retire(TransactionId::new(1)),
            Err(SubmissionOrderError::StillHoldsDomain)
        );

        // Either terminal transition releases the claim and makes retirement
        // legal. Abandonment is the one a published-but-never-submitted
        // transaction takes.
        owner.abandon(TransactionId::new(1)).unwrap();
        owner.retire(TransactionId::new(1)).unwrap();
        assert!(
            owner
                .unsubmitted_predecessor(TransactionId::new(2))
                .unwrap()
                .is_none(),
            "the released claim leaves nothing ahead of the successor"
        );
        owner.recorded(TransactionId::new(2)).unwrap();
        assert_eq!(owner.take_ready()[0].transaction, TransactionId::new(2));
        owner.submitted(TransactionId::new(2)).unwrap();
        owner.retire(TransactionId::new(2)).unwrap();
        owner
            .retire_domain(SubmissionDomainId::new(7))
            .expect("both claims were released, so the domain drained");
    }

    /// Any unsubmitted predecessor blocks a claim, issued or merely pending.
    ///
    /// Two driven macos-13 boots formed the same cycle by two routes, and only
    /// the second says why the pending case counts:
    ///
    /// - 630 was issued and holding domain 1; 632 sat behind it having already
    ///   claimed an image 630 then needed. Neither could move.
    /// - nothing was issued at all; 503 at sequence 58 claimed an image, 501 at
    ///   sequence 56 then became the head and needed it. Neither could move.
    ///
    /// A rule that asks only about the issued head allows the second, so this
    /// asks about the nearest unsubmitted predecessor of either kind.
    #[test]
    fn any_unsubmitted_predecessor_blocks_a_claim_from_the_transactions_behind_it() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 630, 1, 1);
        accept(&mut owner, 631, 1, 2);
        accept(&mut owner, 632, 1, 3);
        accept(&mut owner, 700, 4, 1);

        // Nothing is issued, and the pending predecessors still order the
        // claims. This is the case the narrower rule let through.
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(630)),
            Ok(None),
            "the first in a domain has nothing ahead of it"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(632)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(631),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(2),
                behind: DomainSequence::new(3),
            })),
            "the nearest predecessor is reported, not the domain head"
        );

        owner.recorded(TransactionId::new(630)).unwrap();
        owner
            .reserve_head_transaction_if(TransactionId::new(630), |_| true)
            .unwrap()
            .unwrap();
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(632)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(631),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(2),
                behind: DomainSequence::new(3),
            })),
            "issuing 630 does not make the nearer pending 631 stop being ahead"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(631)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(630),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(1),
                behind: DomainSequence::new(2),
            })),
            "the issued head is the predecessor the pending range cannot see"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(630)),
            Ok(None),
            "the head is not behind itself, or it could never prepare"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(700)),
            Ok(None),
            "another domain's order says nothing here"
        );

        // Submitting clears the claim in order: 630 first, then 631.
        owner.submitted(TransactionId::new(630)).unwrap();
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(631)),
            Ok(None),
            "630 submitted, so 631 is now first"
        );
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(632)),
            Ok(Some(UnsubmittedPredecessor {
                transaction: TransactionId::new(631),
                domain: SubmissionDomainId::new(1),
                sequence: DomainSequence::new(2),
                behind: DomainSequence::new(3),
            })),
            "632 still waits on 631, which has not submitted"
        );
        owner.recorded(TransactionId::new(631)).unwrap();
        owner
            .reserve_head_transaction_if(TransactionId::new(631), |_| true)
            .unwrap()
            .unwrap();
        owner.submitted(TransactionId::new(631)).unwrap();
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(632)),
            Ok(None),
            "the whole prefix submitted, so 632 is free to claim"
        );

        // The order lives in exactly one owner, which is what a caller holding
        // several must select on before it asks.
        assert!(owner.tracks(TransactionId::new(632)));
        assert!(!owner.tracks(TransactionId::new(9_999)));
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(9_999)),
            Err(SubmissionOrderError::UnknownTransaction)
        );
    }

    /// A domain head no owner tracks is corruption, and must be reported
    /// rather than waited on.
    ///
    /// A driven macos-13 boot held `domains[1].issued` at transaction 141,
    /// sequence 22, while 141 appeared in no census the whole boot -- the
    /// index the census iterates did not have it. Nothing could name that head
    /// and so nothing could release it, and every claim behind it would have
    /// waited on a transaction that could never move again. Reporting it is
    /// what tells that apart from an ordinary wait.
    #[test]
    fn an_issued_head_the_index_lost_is_reported_and_not_waited_on() {
        let mut owner = SubmissionOrderOwner::default();
        accept(&mut owner, 141, 1, 1);
        accept(&mut owner, 144, 1, 2);
        owner.recorded(TransactionId::new(141)).unwrap();
        owner
            .reserve_head_transaction_if(TransactionId::new(141), |_| true)
            .unwrap()
            .unwrap();
        assert!(matches!(
            owner.unsubmitted_predecessor(TransactionId::new(144)),
            Ok(Some(_)),
        ));

        // Drop the head from the index without releasing its claim, which is
        // the state the boot was in.
        owner.transactions.remove(&TransactionId::new(141));
        assert_eq!(
            owner.unsubmitted_predecessor(TransactionId::new(144)),
            Err(SubmissionOrderError::IssuedHeadUntracked(
                TransactionId::new(141)
            )),
            "an unnameable head is named as corrupt, not returned as an order"
        );
    }
}
