//! Timeline-submit projection for the replacement replay backend.
//!
//! Semantic dependencies arrive as immutable native timeline points. This
//! module groups waits by physical queue timeline, taking the greatest value
//! on each semaphore, while retaining every original cause for barrier and
//! diagnostic planning. It does not submit or publish semantic completion.

use ash::vk;
use reims_vgpu_core::{
    HazardRequirement, NativeSubmissionPlan, NativeWait, PreparedAuxiliaryNativeSubmission,
    PreparedNativeSubmission, PreparedPresentNativeSubmission, QueueTimelinePoint,
};
use reims_vgpu_protocol::{QueueOwnerId, VulkanDeviceEpochId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimelineSubmitPlanError {
    MissingQueueTimeline(QueueOwnerId),
    MixedEpochs,
    SignalValueZero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemaphoreTimelineWait {
    pub queue: QueueOwnerId,
    pub semaphore: vk::Semaphore,
    pub value: u64,
    pub stage_mask: vk::PipelineStageFlags,
}

#[derive(Clone, Debug)]
pub struct TimelineSubmitPlan {
    pub transaction: reims_vgpu_protocol::TransactionId,
    pub waits: Box<[SemaphoreTimelineWait]>,
    /// Uncoalesced semantic prerequisites consumed by synchronization/barrier
    /// planning. Semaphore coalescing never erases their causes.
    pub semantic_waits: Box<[NativeWait]>,
    /// Accepted native prerequisite points, such as source-family image
    /// releases, which order host API ownership but create no semantic edge.
    pub auxiliary_waits: Box<[QueueTimelinePoint]>,
    /// Exact producer/consumer access pairs retained for synchronization2
    /// barrier projection.
    pub hazards: Box<[HazardRequirement]>,
    pub signal_semaphore: vk::Semaphore,
    pub signal_value: u64,
    signal_queue: QueueOwnerId,
}

impl TimelineSubmitPlan {
    pub const fn signal_queue(&self) -> QueueOwnerId {
        self.signal_queue
    }
}

#[derive(Clone, Debug)]
pub struct QueueTimelineSemaphores {
    epoch: VulkanDeviceEpochId,
    semaphores: BTreeMap<QueueOwnerId, vk::Semaphore>,
}

impl QueueTimelineSemaphores {
    pub fn new(
        epoch: VulkanDeviceEpochId,
        timelines: impl IntoIterator<Item = (QueueOwnerId, vk::Semaphore)>,
    ) -> Self {
        Self {
            epoch,
            semaphores: timelines.into_iter().collect(),
        }
    }

    pub fn plan<Semantic>(
        &self,
        prepared: &PreparedNativeSubmission<Semantic>,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        self.plan_parts(prepared.plan(), prepared.point())
    }

    pub fn plan_auxiliary(
        &self,
        prepared: &PreparedAuxiliaryNativeSubmission,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        self.plan_auxiliary_with_waits(prepared, Box::<[QueueTimelinePoint]>::default())
    }

    pub fn plan_present(
        &self,
        prepared: &PreparedPresentNativeSubmission,
        waits: impl Into<Box<[QueueTimelinePoint]>>,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        let point = prepared.point();
        if point.epoch != self.epoch {
            return Err(TimelineSubmitPlanError::MixedEpochs);
        }
        if point.value.get() == 0 {
            return Err(TimelineSubmitPlanError::SignalValueZero);
        }
        let signal_semaphore = *self
            .semaphores
            .get(&point.queue)
            .ok_or(TimelineSubmitPlanError::MissingQueueTimeline(point.queue))?;
        self.add_auxiliary_waits(
            TimelineSubmitPlan {
                transaction: prepared.transaction(),
                waits: Box::new([]),
                semantic_waits: Box::new([]),
                auxiliary_waits: Box::new([]),
                hazards: Box::new([]),
                signal_semaphore,
                signal_value: point.value.get(),
                signal_queue: point.queue,
            },
            waits.into(),
        )
    }

    pub fn plan_auxiliary_with_waits(
        &self,
        prepared: &PreparedAuxiliaryNativeSubmission,
        auxiliary_waits: impl Into<Box<[QueueTimelinePoint]>>,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        let auxiliary_waits = auxiliary_waits.into();
        let point = prepared.point();
        let plan = if let Some(prerequisites) = prepared.prerequisite_plan() {
            self.plan_parts(prerequisites, point)?
        } else {
            if point.epoch != self.epoch {
                return Err(TimelineSubmitPlanError::MixedEpochs);
            }
            if point.value.get() == 0 {
                return Err(TimelineSubmitPlanError::SignalValueZero);
            }
            let signal_semaphore = *self
                .semaphores
                .get(&point.queue)
                .ok_or(TimelineSubmitPlanError::MissingQueueTimeline(point.queue))?;
            TimelineSubmitPlan {
                transaction: prepared.transaction(),
                waits: Box::new([]),
                semantic_waits: Box::new([]),
                auxiliary_waits: Box::new([]),
                hazards: Box::new([]),
                signal_semaphore,
                signal_value: point.value.get(),
                signal_queue: point.queue,
            }
        };
        self.add_auxiliary_waits(plan, auxiliary_waits)
    }

    fn plan_parts(
        &self,
        submission: &NativeSubmissionPlan,
        signal: QueueTimelinePoint,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        if signal.epoch != self.epoch
            || submission
                .waits
                .iter()
                .any(|wait| wait.point.epoch != self.epoch)
        {
            return Err(TimelineSubmitPlanError::MixedEpochs);
        }
        if signal.value.get() == 0 {
            return Err(TimelineSubmitPlanError::SignalValueZero);
        }
        let signal_semaphore = *self
            .semaphores
            .get(&signal.queue)
            .ok_or(TimelineSubmitPlanError::MissingQueueTimeline(signal.queue))?;
        let mut waits = BTreeMap::<QueueOwnerId, (u64, vk::PipelineStageFlags)>::new();
        for wait in &submission.waits {
            if !self.semaphores.contains_key(&wait.point.queue) {
                return Err(TimelineSubmitPlanError::MissingQueueTimeline(
                    wait.point.queue,
                ));
            }
            let stage_mask = submission
                .hazards
                .iter()
                .filter(|hazard| {
                    hazard.edge.older == wait.producer
                        && hazard.edge.newer == submission.transaction
                })
                .fold(vk::PipelineStageFlags2::empty(), |stages, hazard| {
                    stages | crate::replacement_barriers::stage_flags(hazard.later.stages)
                });
            let stage_mask = if stage_mask.is_empty() {
                vk::PipelineStageFlags::ALL_COMMANDS
            } else {
                debug_assert_eq!(stage_mask.as_raw() & !u32::MAX as u64, 0);
                vk::PipelineStageFlags::from_raw(stage_mask.as_raw() as u32)
            };
            waits
                .entry(wait.point.queue)
                .and_modify(|(value, stages)| {
                    *value = (*value).max(wait.point.value.get());
                    *stages |= stage_mask;
                })
                .or_insert((wait.point.value.get(), stage_mask));
        }
        let waits = waits
            .into_iter()
            .map(|(queue, (value, stage_mask))| SemaphoreTimelineWait {
                queue,
                semaphore: self.semaphores[&queue],
                value,
                stage_mask,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(TimelineSubmitPlan {
            transaction: submission.transaction,
            waits,
            semantic_waits: submission.waits.clone(),
            auxiliary_waits: Box::new([]),
            hazards: submission.hazards.clone(),
            signal_semaphore,
            signal_value: signal.value.get(),
            signal_queue: signal.queue,
        })
    }

    pub fn plan_with_auxiliary_waits<Semantic>(
        &self,
        prepared: &PreparedNativeSubmission<Semantic>,
        auxiliary_waits: impl Into<Box<[QueueTimelinePoint]>>,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        let plan = self.plan(prepared)?;
        self.add_auxiliary_waits(plan, auxiliary_waits.into())
    }

    fn add_auxiliary_waits(
        &self,
        mut plan: TimelineSubmitPlan,
        auxiliary_waits: Box<[QueueTimelinePoint]>,
    ) -> Result<TimelineSubmitPlan, TimelineSubmitPlanError> {
        let mut waits = plan
            .waits
            .iter()
            .map(|wait| (wait.queue, (wait.semaphore, wait.value, wait.stage_mask)))
            .collect::<BTreeMap<_, _>>();
        for point in auxiliary_waits.iter().copied() {
            if point.epoch != self.epoch {
                return Err(TimelineSubmitPlanError::MixedEpochs);
            }
            let semaphore = *self
                .semaphores
                .get(&point.queue)
                .ok_or(TimelineSubmitPlanError::MissingQueueTimeline(point.queue))?;
            waits
                .entry(point.queue)
                .and_modify(|(_, value, stages)| {
                    *value = (*value).max(point.value.get());
                    *stages |= vk::PipelineStageFlags::ALL_COMMANDS;
                })
                .or_insert((
                    semaphore,
                    point.value.get(),
                    vk::PipelineStageFlags::ALL_COMMANDS,
                ));
        }
        plan.waits = waits
            .into_iter()
            .map(
                |(queue, (semaphore, value, stage_mask))| SemaphoreTimelineWait {
                    queue,
                    semaphore,
                    value,
                    stage_mask,
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();
        plan.auxiliary_waits = auxiliary_waits;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::{
        AccessIntent, AccessMode, AccessScope, ExplicitWaitCause, HazardCause, HazardEdge,
        HazardRequirement, LinearRange, NativeWait, StageScope, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        BackingId, ChannelId, HazardDomainId, IngressOrdinal, QueueTimelineValue, ResourceId,
        ResourceObject, SessionGenerationId, TransactionId,
    };

    fn point(epoch: u64, queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(epoch),
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    fn hazard(producer: u64, consumer: u64, later_stage: StageScope) -> HazardRequirement {
        let intent = |mode, stages| {
            AccessIntent::for_backing(
                HazardDomainId::new(1),
                BackingId::new(2),
                Some(ResourceId::<ResourceObject>::new(3, 1)),
                AccessScope::Linear(LinearRange::new(0, 64).unwrap()),
                mode,
                stages,
            )
            .unwrap()
        };
        HazardRequirement {
            edge: HazardEdge {
                newer: TransactionId::new(consumer),
                older: TransactionId::new(producer),
                newer_ordinal: IngressOrdinal::new(consumer),
                older_ordinal: IngressOrdinal::new(producer),
                cause: HazardCause::Buffer,
            },
            earlier: intent(AccessMode::Write, StageScope::Compute),
            later: intent(AccessMode::Read, later_stage),
        }
    }

    #[test]
    fn present_plan_uses_its_allocator_owned_signal_and_exact_native_waits() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(1);
        let producer = QueueOwnerId::new(2);
        let timelines = QueueTimelineSemaphores::new(
            epoch,
            [
                (queue, vk::Semaphore::from_raw(10)),
                (producer, vk::Semaphore::from_raw(11)),
            ],
        );
        let mut owner = reims_vgpu_core::DirectReplayNativeOwner::<()>::new(epoch, 1).unwrap();
        let prepared = owner.prepare_present(TransactionId::new(4), queue).unwrap();
        let plan = timelines
            .plan_present(&prepared, vec![point(1, 2, 7)].into_boxed_slice())
            .unwrap();
        assert_eq!(plan.transaction, TransactionId::new(4));
        assert_eq!(plan.signal_queue(), queue);
        assert_eq!(plan.signal_value, prepared.point().value.get());
        assert_eq!(plan.waits.len(), 1);
        assert_eq!(plan.waits[0].queue, producer);
        assert_eq!(plan.waits[0].value, 7);
        assert!(plan.semantic_waits.is_empty());
        assert!(plan.hazards.is_empty());
    }

    #[test]
    fn waits_coalesce_by_timeline_without_erasing_semantic_causes() {
        let timelines = QueueTimelineSemaphores::new(
            VulkanDeviceEpochId::new(1),
            [
                (QueueOwnerId::new(0), vk::Semaphore::from_raw(10)),
                (QueueOwnerId::new(1), vk::Semaphore::from_raw(11)),
            ],
        );
        let submission = NativeSubmissionPlan {
            transaction: TransactionId::new(3),
            waits: vec![
                NativeWait {
                    producer: TransactionId::new(1),
                    point: point(1, 0, 2),
                    cause: WaitDependencyCause::ResourceHazard(HazardCause::Buffer),
                },
                NativeWait {
                    producer: TransactionId::new(2),
                    point: point(1, 0, 5),
                    cause: WaitDependencyCause::Explicit(ExplicitWaitCause::Stamp {
                        source_channel: ChannelId::new(7),
                        value: 9,
                    }),
                },
            ]
            .into_boxed_slice(),
            hazards: Box::new([hazard(2, 3, StageScope::Vertex)]),
        };
        let plan = timelines.plan_parts(&submission, point(1, 1, 1)).unwrap();
        assert_eq!(plan.waits.len(), 1);
        assert_eq!(plan.waits[0].value, 5);
        assert_eq!(
            plan.waits[0].stage_mask,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::VERTEX_SHADER
        );
        assert_eq!(plan.semantic_waits.len(), 2);
        assert_eq!(plan.signal_semaphore, vk::Semaphore::from_raw(11));
        assert_eq!(plan.signal_value, 1);
    }

    #[test]
    fn every_wait_and_signal_must_belong_to_the_epoch_registry() {
        let timelines = QueueTimelineSemaphores::new(
            VulkanDeviceEpochId::new(1),
            [(QueueOwnerId::new(0), vk::Semaphore::from_raw(10))],
        );
        let missing = NativeSubmissionPlan {
            transaction: TransactionId::new(1),
            waits: Box::new([NativeWait {
                producer: TransactionId::new(2),
                point: point(1, 3, 1),
                cause: WaitDependencyCause::ResourceHazard(HazardCause::Image),
            }]),
            hazards: Box::new([]),
        };
        assert_eq!(
            timelines.plan_parts(&missing, point(1, 0, 1)).unwrap_err(),
            TimelineSubmitPlanError::MissingQueueTimeline(QueueOwnerId::new(3))
        );
        let empty = NativeSubmissionPlan {
            transaction: TransactionId::new(1),
            waits: Box::new([]),
            hazards: Box::new([]),
        };
        assert_eq!(
            timelines.plan_parts(&empty, point(2, 0, 1)).unwrap_err(),
            TimelineSubmitPlanError::MixedEpochs
        );
    }

    #[test]
    fn auxiliary_release_wait_is_retained_without_fabricating_a_semantic_cause() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(1);
        let mut native = reims_vgpu_core::DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(reims_vgpu_core::TransactionRecordingPlan {
                transaction: TransactionId::new(1),
                domain: reims_vgpu_protocol::SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = native
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(plan, queue, SessionGenerationId::new(1), ())
            .unwrap();
        let release = point(1, 2, 4);
        let timelines = QueueTimelineSemaphores::new(
            epoch,
            [
                (queue, vk::Semaphore::from_raw(10)),
                (release.queue, vk::Semaphore::from_raw(11)),
            ],
        );
        let projected = timelines
            .plan_with_auxiliary_waits(&prepared, [release])
            .unwrap();
        assert_eq!(projected.auxiliary_waits.as_ref(), [release]);
        assert!(projected.semantic_waits.is_empty());
        assert_eq!(projected.waits.len(), 1);
        assert_eq!(projected.waits[0].queue, release.queue);
        assert_eq!(projected.waits[0].value, 4);
        assert_eq!(
            projected.waits[0].stage_mask,
            vk::PipelineStageFlags::ALL_COMMANDS
        );
    }

    #[test]
    fn first_auxiliary_phase_inherits_the_parent_semantic_waits() {
        let epoch = VulkanDeviceEpochId::new(1);
        let producer_queue = QueueOwnerId::new(0);
        let consumer_queue = QueueOwnerId::new(1);
        let producer = TransactionId::new(1);
        let consumer = TransactionId::new(2);
        let mut native = reims_vgpu_core::DirectReplayNativeOwner::new(epoch, 1).unwrap();
        for transaction in [producer, consumer] {
            native
                .assign_recording(reims_vgpu_core::TransactionRecordingPlan {
                    transaction,
                    domain: reims_vgpu_protocol::SubmissionDomainId::new(1),
                    continuation_predecessor: None,
                })
                .unwrap();
        }
        let cause = WaitDependencyCause::ResourceHazard(HazardCause::Buffer);
        assert!(native
            .queue_candidate(consumer, [(producer, cause)])
            .unwrap()
            .is_empty());
        let producer_plan = native
            .queue_candidate(
                producer,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let producer_prepared = native
            .prepare(
                producer_plan,
                producer_queue,
                SessionGenerationId::new(1),
                (),
            )
            .unwrap();
        let consumer_plan = native
            .accepted(producer_prepared)
            .unwrap()
            .newly_ready
            .pop()
            .unwrap();
        let parent = native
            .prepare(
                consumer_plan,
                consumer_queue,
                SessionGenerationId::new(1),
                (),
            )
            .unwrap();
        let auxiliary = native.prepare_auxiliary(&parent, consumer_queue).unwrap();
        let timelines = QueueTimelineSemaphores::new(
            epoch,
            [
                (producer_queue, vk::Semaphore::from_raw(10)),
                (consumer_queue, vk::Semaphore::from_raw(11)),
            ],
        );
        let projected = timelines.plan_auxiliary(&auxiliary).unwrap();
        assert_eq!(projected.semantic_waits.len(), 1);
        assert_eq!(projected.semantic_waits[0].producer, producer);
        assert_eq!(projected.waits.len(), 1);
        assert_eq!(projected.waits[0].queue, producer_queue);
        assert_eq!(projected.waits[0].value, 1);
    }
}
