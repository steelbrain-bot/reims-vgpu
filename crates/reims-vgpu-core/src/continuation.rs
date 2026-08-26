//! Ordered encoder-continuation ownership across EXEC packet boundaries.
//!
//! Source API command-buffer identity does not cross the FIFO boundary. What
//! does cross is a directional segment edge: a segment can promise that its
//! encoder state continues into the next segment, and the next segment must
//! consume that promise. This owner validates the chain per source-queue/FIFO
//! domain and returns the exact recording prerequisite. It does not infer a
//! chain from packet adjacency, completion values, or encoder kind alone.

use crate::ExecTransaction;
use reims_vgpu_protocol::{SegmentBoundary, SegmentKind, SubmissionDomainId, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContinuationDependency {
    pub successor: TransactionId,
    pub predecessor: TransactionId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuationError {
    DuplicateTransaction,
    UnknownTransaction,
    MissingPredecessor,
    PromisedContinuationNotConsumed,
    EncoderKindChanged,
    BrokenInternalEdge,
    OpenContinuation,
    DomainHasLiveTransactions,
}

#[derive(Clone, Copy, Debug)]
struct OpenTail {
    transaction: TransactionId,
    kind: SegmentKind,
}

#[derive(Clone, Debug, Default)]
pub struct EncoderContinuationOwner {
    open: BTreeMap<SubmissionDomainId, OpenTail>,
    admitted: BTreeMap<TransactionId, SubmissionDomainId>,
}

impl EncoderContinuationOwner {
    /// Admit one resolved EXEC atomically and return its cross-packet recording
    /// dependency, if its first segment consumes a predecessor's continuation.
    pub fn admit<Operation>(
        &mut self,
        transaction: TransactionId,
        domain: SubmissionDomainId,
        exec: &ExecTransaction<Operation>,
    ) -> Result<Option<ContinuationDependency>, ContinuationError> {
        if self.admitted.contains_key(&transaction) {
            return Err(ContinuationError::DuplicateTransaction);
        }

        let boundaries = exec
            .streams
            .iter()
            .flat_map(|stream| stream.segments.iter())
            .map(|segment| segment.boundary)
            .collect::<Vec<_>>();
        let Some(first) = boundaries.first().copied() else {
            self.admitted.insert(transaction, domain);
            return Ok(None);
        };

        let predecessor = match (self.open.get(&domain).copied(), first.continues_previous) {
            (Some(tail), true) => {
                if tail.kind != first.kind {
                    return Err(ContinuationError::EncoderKindChanged);
                }
                Some(tail.transaction)
            }
            (Some(_), false) => {
                return Err(ContinuationError::PromisedContinuationNotConsumed);
            }
            (None, true) => return Err(ContinuationError::MissingPredecessor),
            (None, false) => None,
        };

        for pair in boundaries.windows(2) {
            validate_internal_edge(pair[0], pair[1])?;
        }

        let last = *boundaries.last().unwrap();
        if last.continues_next {
            self.open.insert(
                domain,
                OpenTail {
                    transaction,
                    kind: last.kind,
                },
            );
        } else {
            self.open.remove(&domain);
        }
        self.admitted.insert(transaction, domain);

        Ok(predecessor.map(|predecessor| ContinuationDependency {
            successor: transaction,
            predecessor,
        }))
    }

    pub fn has_open_continuation(&self, domain: SubmissionDomainId) -> bool {
        self.open.contains_key(&domain)
    }

    /// Releases duplicate-detection state after the transaction no longer has
    /// semantic dependents. An open tail remains a required predecessor and
    /// therefore cannot retire.
    pub fn retire_transaction(
        &mut self,
        transaction: TransactionId,
    ) -> Result<(), ContinuationError> {
        if self
            .open
            .values()
            .any(|tail| tail.transaction == transaction)
        {
            return Err(ContinuationError::OpenContinuation);
        }
        self.admitted
            .remove(&transaction)
            .map(|_| ())
            .ok_or(ContinuationError::UnknownTransaction)
    }

    /// Verifies that no continuation or admitted transaction survives the
    /// end of a source-queue/FIFO lifetime.
    pub fn retire_domain(&mut self, domain: SubmissionDomainId) -> Result<(), ContinuationError> {
        if self.open.contains_key(&domain) {
            return Err(ContinuationError::OpenContinuation);
        }
        if self
            .admitted
            .values()
            .any(|admitted_domain| *admitted_domain == domain)
        {
            return Err(ContinuationError::DomainHasLiveTransactions);
        }
        Ok(())
    }
}

fn validate_internal_edge(
    previous: SegmentBoundary,
    next: SegmentBoundary,
) -> Result<(), ContinuationError> {
    if previous.continues_next != next.continues_previous {
        return Err(ContinuationError::BrokenInternalEdge);
    }
    if previous.continues_next && previous.kind != next.kind {
        return Err(ContinuationError::EncoderKindChanged);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResolvedExecSegment, ResolvedExecStream};
    use reims_vgpu_protocol::{SubmissionId, SubmissionIdentity, TaskId};

    fn exec(boundaries: &[(SegmentKind, bool, bool)]) -> ExecTransaction<()> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: vec![ResolvedExecStream {
                stream_index: 0,
                segments: boundaries
                    .iter()
                    .enumerate()
                    .map(|(index, &(kind, continues_previous, continues_next))| {
                        ResolvedExecSegment {
                            boundary: SegmentBoundary {
                                stream_index: 0,
                                index: index as u32,
                                kind,
                                continues_previous,
                                continues_next,
                            },
                            operations: Box::new([]),
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            }]
            .into_boxed_slice(),
            accesses: Box::new([]),
        }
    }

    #[test]
    fn a_directional_cross_packet_edge_names_the_exact_predecessor() {
        let mut owner = EncoderContinuationOwner::default();
        let domain = SubmissionDomainId::new(4);
        assert_eq!(
            owner.admit(
                TransactionId::new(1),
                domain,
                &exec(&[(SegmentKind::Render, false, true)])
            ),
            Ok(None)
        );
        assert_eq!(
            owner.admit(
                TransactionId::new(2),
                domain,
                &exec(&[(SegmentKind::Render, true, false)])
            ),
            Ok(Some(ContinuationDependency {
                successor: TransactionId::new(2),
                predecessor: TransactionId::new(1),
            }))
        );
        assert!(!owner.has_open_continuation(domain));
    }

    #[test]
    fn independent_domains_do_not_share_continuation_state() {
        let mut owner = EncoderContinuationOwner::default();
        owner
            .admit(
                TransactionId::new(1),
                SubmissionDomainId::new(1),
                &exec(&[(SegmentKind::Compute, false, true)]),
            )
            .unwrap();
        assert_eq!(
            owner.admit(
                TransactionId::new(2),
                SubmissionDomainId::new(2),
                &exec(&[(SegmentKind::Blit, false, false)])
            ),
            Ok(None)
        );
    }

    #[test]
    fn a_failed_chain_check_changes_no_owner_state() {
        let mut owner = EncoderContinuationOwner::default();
        let domain = SubmissionDomainId::new(1);
        owner
            .admit(
                TransactionId::new(1),
                domain,
                &exec(&[(SegmentKind::Render, false, true)]),
            )
            .unwrap();
        assert_eq!(
            owner.admit(
                TransactionId::new(2),
                domain,
                &exec(&[(SegmentKind::Compute, true, false)])
            ),
            Err(ContinuationError::EncoderKindChanged)
        );
        assert!(owner.has_open_continuation(domain));
        assert_eq!(
            owner.admit(
                TransactionId::new(2),
                domain,
                &exec(&[(SegmentKind::Render, true, false)])
            ),
            Ok(Some(ContinuationDependency {
                successor: TransactionId::new(2),
                predecessor: TransactionId::new(1),
            }))
        );
    }

    #[test]
    fn internal_edges_must_be_directionally_consistent() {
        let mut owner = EncoderContinuationOwner::default();
        assert_eq!(
            owner.admit(
                TransactionId::new(1),
                SubmissionDomainId::new(1),
                &exec(&[
                    (SegmentKind::Render, false, false),
                    (SegmentKind::Render, true, false),
                ])
            ),
            Err(ContinuationError::BrokenInternalEdge)
        );
    }

    #[test]
    fn transaction_and_domain_retirement_preserve_open_edges() {
        let mut owner = EncoderContinuationOwner::default();
        let domain = SubmissionDomainId::new(1);
        owner
            .admit(
                TransactionId::new(1),
                domain,
                &exec(&[(SegmentKind::Render, false, true)]),
            )
            .unwrap();
        assert_eq!(
            owner.retire_transaction(TransactionId::new(1)),
            Err(ContinuationError::OpenContinuation)
        );
        assert_eq!(
            owner.retire_domain(domain),
            Err(ContinuationError::OpenContinuation)
        );
        owner
            .admit(
                TransactionId::new(2),
                domain,
                &exec(&[(SegmentKind::Render, true, false)]),
            )
            .unwrap();
        owner.retire_transaction(TransactionId::new(1)).unwrap();
        assert_eq!(
            owner.retire_domain(domain),
            Err(ContinuationError::DomainHasLiveTransactions)
        );
        owner.retire_transaction(TransactionId::new(2)).unwrap();
        assert_eq!(owner.retire_domain(domain), Ok(()));
    }
}
