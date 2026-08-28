//! Translate semantic producer identities into native timeline waits.
//!
//! Recording and logical queue ordering may make a consumer ready before a
//! future producer has a native timeline point. This owner parks only that
//! consumer until every producer has been submitted, then returns immutable
//! timeline waits. It never waits for those points to complete and never turns
//! a Vulkan-only fact into semantic completion.

use crate::{HazardRequirement, QueueTimelinePoint, WaitDependencyCause};
use reims_vgpu_protocol::TransactionId;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeWait {
    pub producer: TransactionId,
    pub point: QueueTimelinePoint,
    pub cause: WaitDependencyCause,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeSubmissionPlan {
    pub transaction: TransactionId,
    pub waits: Box<[NativeWait]>,
    pub hazards: Box<[HazardRequirement]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDependencyError {
    DuplicateCandidate,
    DuplicateSubmissionPoint,
    UnknownTransaction,
    NotSubmitted,
    PlanMismatch,
}

#[derive(Clone, Debug)]
struct Candidate {
    prerequisites: Box<[(TransactionId, WaitDependencyCause)]>,
    hazards: Box<[HazardRequirement]>,
}

#[derive(Clone, Debug, Default)]
pub struct NativeDependencyOwner {
    candidates: BTreeMap<TransactionId, Candidate>,
    issued: BTreeMap<TransactionId, NativeSubmissionPlan>,
    points: BTreeMap<TransactionId, QueueTimelinePoint>,
}

impl NativeDependencyOwner {
    /// Register one recorded, logically ordered candidate and return it
    /// immediately when every semantic producer already has a native point.
    pub fn queue_ready(
        &mut self,
        transaction: TransactionId,
        prerequisites: impl Into<Box<[(TransactionId, WaitDependencyCause)]>>,
    ) -> Result<Vec<NativeSubmissionPlan>, NativeDependencyError> {
        self.queue_ready_with_hazards(
            transaction,
            prerequisites,
            Box::<[HazardRequirement]>::default(),
        )
    }

    pub fn queue_ready_with_hazards(
        &mut self,
        transaction: TransactionId,
        prerequisites: impl Into<Box<[(TransactionId, WaitDependencyCause)]>>,
        hazards: impl Into<Box<[HazardRequirement]>>,
    ) -> Result<Vec<NativeSubmissionPlan>, NativeDependencyError> {
        if self.candidates.contains_key(&transaction)
            || self.issued.contains_key(&transaction)
            || self.points.contains_key(&transaction)
        {
            return Err(NativeDependencyError::DuplicateCandidate);
        }
        self.candidates.insert(
            transaction,
            Candidate {
                prerequisites: prerequisites.into(),
                hazards: hazards.into(),
            },
        );
        Ok(self.take_ready())
    }

    /// Record successful native queue acceptance. Newly resolvable consumers
    /// may now be submitted with waits on this point, without waiting on the
    /// GPU or a host condition variable.
    pub fn submitted(
        &mut self,
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<Vec<NativeSubmissionPlan>, NativeDependencyError> {
        if self.points.contains_key(&transaction) {
            return Err(NativeDependencyError::DuplicateSubmissionPoint);
        }
        if self.issued.remove(&transaction).is_none() {
            return Err(NativeDependencyError::UnknownTransaction);
        }
        self.points.insert(transaction, point);
        Ok(self.take_ready())
    }

    pub fn validate_plan(&self, plan: &NativeSubmissionPlan) -> Result<(), NativeDependencyError> {
        match self.issued.get(&plan.transaction) {
            Some(issued) if issued == plan => Ok(()),
            Some(_) => Err(NativeDependencyError::PlanMismatch),
            None => Err(NativeDependencyError::UnknownTransaction),
        }
    }

    pub fn cancel_issued(
        &mut self,
        plan: &NativeSubmissionPlan,
    ) -> Result<(), NativeDependencyError> {
        self.validate_plan(plan)?;
        self.issued.remove(&plan.transaction);
        Ok(())
    }

    /// Whether this transaction has a native submission point yet.
    ///
    /// A producer without one cannot be waited on natively, so a consumer that
    /// names it is parked until it appears. Asked *before* a consumer takes an
    /// exclusive claim, this is what keeps the consumer from holding what the
    /// producer needs in order to reach a point at all.
    #[must_use]
    pub fn has_submission_point(&self, transaction: TransactionId) -> bool {
        self.points.contains_key(&transaction)
    }

    /// Candidates still parked, each with the producers that have no native
    /// submission point yet.
    ///
    /// A candidate is released only by a producer being submitted, so a parked
    /// set whose unmet producers never shrink is a native deadlock: nothing
    /// remaining will submit them. That is invisible from any other reading —
    /// the consumer looks like ordinary in-flight work — so it is reported
    /// rather than inferred.
    pub fn parked_candidates(&self) -> Vec<(TransactionId, Vec<TransactionId>)> {
        self.candidates
            .iter()
            .map(|(&transaction, candidate)| {
                (
                    transaction,
                    candidate
                        .prerequisites
                        .iter()
                        .filter(|(producer, _)| !self.points.contains_key(producer))
                        .map(|&(producer, _)| producer)
                        .collect(),
                )
            })
            .collect()
    }

    fn take_ready(&mut self) -> Vec<NativeSubmissionPlan> {
        let ready = self
            .candidates
            .iter()
            .filter_map(|(&transaction, candidate)| {
                candidate
                    .prerequisites
                    .iter()
                    .all(|(producer, _)| self.points.contains_key(producer))
                    .then_some(transaction)
            })
            .collect::<Vec<_>>();
        let plans = ready
            .into_iter()
            .map(|transaction| {
                let candidate = self.candidates.remove(&transaction).unwrap();
                let waits = candidate
                    .prerequisites
                    .iter()
                    .map(|&(producer, cause)| NativeWait {
                        producer,
                        point: self.points[&producer],
                        cause,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                NativeSubmissionPlan {
                    transaction,
                    waits,
                    hazards: candidate.hazards,
                }
            })
            .collect::<Vec<_>>();
        for plan in &plans {
            self.issued.insert(plan.transaction, plan.clone());
        }
        plans
    }

    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), NativeDependencyError> {
        if self.candidates.contains_key(&transaction) || self.issued.contains_key(&transaction) {
            return Err(NativeDependencyError::NotSubmitted);
        }
        self.points
            .remove(&transaction)
            .map(|_| ())
            .ok_or(NativeDependencyError::UnknownTransaction)
    }

    /// Return candidates that never reached a native queue. Submitted points
    /// are owned by completion abandonment instead.
    pub fn abandon(self) -> Vec<TransactionId> {
        let mut transactions = self.candidates.into_keys().collect::<Vec<_>>();
        transactions.extend(self.issued.into_keys());
        transactions.sort();
        transactions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExplicitWaitCause, HazardCause};
    use reims_vgpu_protocol::{ChannelId, QueueOwnerId, QueueTimelineValue, VulkanDeviceEpochId};

    fn point(queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(1),
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    #[test]
    fn future_producer_blocks_only_native_submission_not_a_host_thread() {
        let mut owner = NativeDependencyOwner::default();
        let cause = WaitDependencyCause::Explicit(ExplicitWaitCause::Stamp {
            source_channel: ChannelId::new(2),
            value: 9,
        });
        assert!(owner
            .queue_ready(TransactionId::new(1), [(TransactionId::new(2), cause)],)
            .unwrap()
            .is_empty());
        let producer = owner
            .queue_ready(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap();
        assert_eq!(producer[0].transaction, TransactionId::new(2));
        let ready = owner.submitted(TransactionId::new(2), point(1, 4)).unwrap();
        assert_eq!(
            ready,
            vec![NativeSubmissionPlan {
                transaction: TransactionId::new(1),
                waits: Box::new([NativeWait {
                    producer: TransactionId::new(2),
                    point: point(1, 4),
                    cause,
                }]),
                hazards: Box::new([]),
            }]
        );
    }

    #[test]
    fn independent_candidate_is_returned_while_another_awaits_a_point() {
        let mut owner = NativeDependencyOwner::default();
        owner
            .queue_ready(
                TransactionId::new(1),
                [(
                    TransactionId::new(3),
                    WaitDependencyCause::ResourceHazard(HazardCause::Buffer),
                )],
            )
            .unwrap();
        assert_eq!(
            owner
                .queue_ready(
                    TransactionId::new(2),
                    Box::<[(TransactionId, WaitDependencyCause)]>::default(),
                )
                .unwrap()[0]
                .transaction,
            TransactionId::new(2)
        );
    }
}
