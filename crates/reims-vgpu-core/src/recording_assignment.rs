//! Stable recording-worker assignment for immutable EXEC transactions.
//!
//! Independent recordings spread across a fixed population. A directional
//! continuation inherits its exact predecessor's worker so mutable encoder
//! state never crosses worker-owned native pools.

use reims_vgpu_protocol::{SubmissionDomainId, TransactionId};
use std::collections::BTreeMap;

/// Stable identity of one member of a Vulkan epoch's fixed recording-worker
/// population.
///
/// This identity is assigned once and remains the key for every worker-owned
/// command-pool and descriptor-arena operation until native retirement.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RecordingWorkerId(usize);

impl RecordingWorkerId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordingAssignmentError {
    NoWorkers,
    DuplicateTransaction,
    UnknownPredecessor,
    ContinuationDomainChanged,
    UnknownTransaction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Assignment {
    domain: SubmissionDomainId,
    worker: RecordingWorkerId,
}

#[derive(Clone, Debug)]
pub struct RecordingAssignmentOwner {
    workers: usize,
    next: usize,
    assignments: BTreeMap<TransactionId, Assignment>,
}

impl RecordingAssignmentOwner {
    pub fn new(workers: usize) -> Result<Self, RecordingAssignmentError> {
        if workers == 0 {
            return Err(RecordingAssignmentError::NoWorkers);
        }
        Ok(Self {
            workers,
            next: 0,
            assignments: BTreeMap::new(),
        })
    }

    pub const fn worker_count(&self) -> usize {
        self.workers
    }

    pub fn assign(
        &mut self,
        transaction: TransactionId,
        domain: SubmissionDomainId,
        continuation_predecessor: Option<TransactionId>,
    ) -> Result<RecordingWorkerId, RecordingAssignmentError> {
        if self.assignments.contains_key(&transaction) {
            return Err(RecordingAssignmentError::DuplicateTransaction);
        }
        let worker = if let Some(predecessor) = continuation_predecessor {
            let predecessor = self
                .assignments
                .get(&predecessor)
                .ok_or(RecordingAssignmentError::UnknownPredecessor)?;
            if predecessor.domain != domain {
                return Err(RecordingAssignmentError::ContinuationDomainChanged);
            }
            predecessor.worker
        } else {
            let worker = RecordingWorkerId::new(self.next);
            self.next = (self.next + 1) % self.workers;
            worker
        };
        self.assignments
            .insert(transaction, Assignment { domain, worker });
        Ok(worker)
    }

    pub fn worker(&self, transaction: TransactionId) -> Option<RecordingWorkerId> {
        self.assignments
            .get(&transaction)
            .map(|assignment| assignment.worker)
    }

    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), RecordingAssignmentError> {
        self.assignments
            .remove(&transaction)
            .map(|_| ())
            .ok_or(RecordingAssignmentError::UnknownTransaction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_work_spreads_and_continuation_stays_pinned() {
        let mut owner = RecordingAssignmentOwner::new(2).unwrap();
        assert_eq!(
            owner
                .assign(TransactionId::new(1), SubmissionDomainId::new(7), None)
                .unwrap(),
            RecordingWorkerId::new(0)
        );
        assert_eq!(
            owner
                .assign(TransactionId::new(2), SubmissionDomainId::new(8), None)
                .unwrap(),
            RecordingWorkerId::new(1)
        );
        assert_eq!(
            owner
                .assign(
                    TransactionId::new(3),
                    SubmissionDomainId::new(7),
                    Some(TransactionId::new(1)),
                )
                .unwrap(),
            RecordingWorkerId::new(0)
        );
    }

    #[test]
    fn a_continuation_cannot_move_to_another_domain() {
        let mut owner = RecordingAssignmentOwner::new(1).unwrap();
        owner
            .assign(TransactionId::new(1), SubmissionDomainId::new(1), None)
            .unwrap();
        assert_eq!(
            owner.assign(
                TransactionId::new(2),
                SubmissionDomainId::new(2),
                Some(TransactionId::new(1)),
            ),
            Err(RecordingAssignmentError::ContinuationDomainChanged)
        );
    }
}
