//! Per-device-epoch ownership of actual queue submission and completion lanes.

use crate::{
    replacement_capabilities::ReplacementCapabilities,
    replacement_completion::{
        ReplacementTimelineObservation, ReplacementTimelineWatcher,
        ReplacementTimelineWatcherStartError,
    },
    replacement_queue::ReplacementQueueOwner,
    replacement_recording::{
        ReplacementRecordingOperation, ReplacementRecordingRequest, ReplacementRecordingWorker,
    },
    replacement_recording_queue::{
        dispatch_prepared_replacement_auxiliary_recording,
        dispatch_prepared_replacement_recording_with_auxiliary_waits,
        PendingPreparedReplacementAuxiliaryRecording, PendingPreparedReplacementRecording,
        PreparedReplacementAuxiliaryRecordingFailure, PreparedReplacementRecordingFailure,
    },
    replacement_submit::QueueTimelineSemaphores,
};
use ash::vk;
use reims_vgpu_core::{
    FixedExecutor, FixedExecutorError, PreparedAuxiliaryNativeSubmission, PreparedNativeSubmission,
    QueueTimelinePoint,
};
use reims_vgpu_protocol::{QueueOwnerId, VulkanDeviceEpochId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct ReplacementQueueBinding {
    pub id: QueueOwnerId,
    pub queue_family: u32,
    pub flags: vk::QueueFlags,
    pub queue: vk::Queue,
    pub timeline: vk::Semaphore,
}

#[derive(Debug)]
pub enum ReplacementQueueEpochStartError {
    NoQueues,
    DuplicateQueue(QueueOwnerId),
    SubmitThread {
        queue: QueueOwnerId,
        error: std::io::Error,
    },
    CompletionThread {
        queue: QueueOwnerId,
        error: ReplacementTimelineWatcherStartError,
    },
    RecordingWorkers(FixedExecutorError),
}

pub struct ReplacementQueueLane {
    pub queue_family: u32,
    pub flags: vk::QueueFlags,
    pub submit: ReplacementQueueOwner,
    pub completion: ReplacementTimelineWatcher,
}

/// Every mutable queue object belonging to one replacement Vulkan epoch.
pub struct ReplacementQueueEpoch {
    epoch: VulkanDeviceEpochId,
    timelines: QueueTimelineSemaphores,
    lanes: BTreeMap<QueueOwnerId, ReplacementQueueLane>,
    recording_workers: FixedExecutor<ReplacementRecordingWorker>,
}

impl ReplacementQueueEpoch {
    pub fn start(
        epoch: VulkanDeviceEpochId,
        capabilities: ReplacementCapabilities,
        device: &ash::Device,
        recording_worker_count: usize,
        bindings: impl IntoIterator<Item = ReplacementQueueBinding>,
    ) -> Result<Self, ReplacementQueueEpochStartError> {
        let bindings = bindings.into_iter().collect::<Vec<_>>();
        validate_bindings(&bindings)?;
        validate_recording_worker_count(recording_worker_count)?;
        let timelines = QueueTimelineSemaphores::new(
            epoch,
            bindings
                .iter()
                .map(|binding| (binding.id, binding.timeline)),
        );
        let mut lanes = BTreeMap::new();
        for binding in bindings {
            let submit = ReplacementQueueOwner::start(binding.id, device, binding.queue).map_err(
                |error| ReplacementQueueEpochStartError::SubmitThread {
                    queue: binding.id,
                    error,
                },
            )?;
            let completion =
                ReplacementTimelineWatcher::start(epoch, binding.id, device, binding.timeline)
                    .map_err(|error| ReplacementQueueEpochStartError::CompletionThread {
                        queue: binding.id,
                        error,
                    })?;
            lanes.insert(
                binding.id,
                ReplacementQueueLane {
                    queue_family: binding.queue_family,
                    flags: binding.flags,
                    submit,
                    completion,
                },
            );
        }
        let recording_workers = FixedExecutor::new(recording_worker_count, |worker| {
            ReplacementRecordingWorker::new(worker, device, capabilities.descriptor_tier())
        })
        .map_err(ReplacementQueueEpochStartError::RecordingWorkers)?;
        Ok(Self {
            epoch,
            timelines,
            lanes,
            recording_workers,
        })
    }

    pub const fn epoch(&self) -> VulkanDeviceEpochId {
        self.epoch
    }

    pub const fn timelines(&self) -> &QueueTimelineSemaphores {
        &self.timelines
    }

    pub fn lane(&self, queue: QueueOwnerId) -> Option<&ReplacementQueueLane> {
        self.lanes.get(&queue)
    }

    pub const fn recording_workers(&self) -> &FixedExecutor<ReplacementRecordingWorker> {
        &self.recording_workers
    }

    /// Replace the fixed recording population after a platform reset.
    ///
    /// The caller must first establish a queue-idle boundary. Dropping the old
    /// workers then destroys their command/descriptor pools, including every
    /// reset-cancelled recording allocated from those pools, while queue lanes
    /// and their timeline semaphores remain unchanged.
    pub fn reset_recording_workers(
        &mut self,
        device: &ash::Device,
        capabilities: ReplacementCapabilities,
    ) -> Result<(), FixedExecutorError> {
        let count = self.recording_workers.census().workers;
        let replacement = FixedExecutor::new(count, |worker| {
            ReplacementRecordingWorker::new(worker, device, capabilities.descriptor_tier())
        })?;
        let retired = std::mem::replace(&mut self.recording_workers, replacement);
        drop(retired);
        Ok(())
    }

    /// Join every native worker after the owning Vulkan epoch enters loss.
    /// No replacement native object is constructed in the closed epoch.
    pub fn terminate_workers_after_loss(&mut self) {
        self.recording_workers.stop();
        for lane in self.lanes.values_mut() {
            lane.submit.shutdown();
            lane.completion.shutdown();
        }
    }

    pub fn record_prepared<
        Semantic,
        Operation: Clone + Send + ReplacementRecordingOperation + 'static,
    >(
        &self,
        prepared: PreparedNativeSubmission<Semantic>,
        request: ReplacementRecordingRequest<Operation>,
    ) -> Result<
        PendingPreparedReplacementRecording<Semantic, Operation>,
        Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
    > {
        self.record_prepared_with_auxiliary_waits(
            prepared,
            request,
            Box::<[QueueTimelinePoint]>::default(),
        )
    }

    pub fn record_prepared_with_auxiliary_waits<
        Semantic,
        Operation: Clone + Send + ReplacementRecordingOperation + 'static,
    >(
        &self,
        prepared: PreparedNativeSubmission<Semantic>,
        request: ReplacementRecordingRequest<Operation>,
        auxiliary_waits: impl Into<Box<[QueueTimelinePoint]>>,
    ) -> Result<
        PendingPreparedReplacementRecording<Semantic, Operation>,
        Box<PreparedReplacementRecordingFailure<Semantic, Operation>>,
    > {
        let queue = prepared.point().queue;
        if let Err(reason) = validate_queue_family(
            self.lanes.get(&queue).map(|lane| lane.queue_family),
            queue,
            request.queue_family,
        ) {
            return Err(Box::new(PreparedReplacementRecordingFailure {
                reason,
                prepared,
                recovery: crate::replacement_recording_queue::PreparedReplacementRecordingRecovery::Request(request),
            }));
        }
        let required = request.required_queue_flags();
        if let Err(reason) = validate_queue_capabilities(self.lanes[&queue].flags, queue, required)
        {
            return Err(Box::new(PreparedReplacementRecordingFailure {
                reason,
                prepared,
                recovery: crate::replacement_recording_queue::PreparedReplacementRecordingRecovery::Request(request),
            }));
        }
        dispatch_prepared_replacement_recording_with_auxiliary_waits(
            &self.recording_workers,
            &self.timelines,
            prepared,
            request,
            auxiliary_waits,
        )
    }

    pub fn record_prepared_auxiliary<
        Operation: Clone + Send + ReplacementRecordingOperation + 'static,
    >(
        &self,
        prepared: PreparedAuxiliaryNativeSubmission,
        request: ReplacementRecordingRequest<Operation>,
        auxiliary_waits: impl Into<Box<[QueueTimelinePoint]>>,
    ) -> Result<
        PendingPreparedReplacementAuxiliaryRecording<Operation>,
        Box<PreparedReplacementAuxiliaryRecordingFailure<Operation>>,
    > {
        let queue = prepared.point().queue;
        if let Err(reason) = validate_queue_family(
            self.lanes.get(&queue).map(|lane| lane.queue_family),
            queue,
            request.queue_family,
        ) {
            return Err(Box::new(PreparedReplacementAuxiliaryRecordingFailure {
                reason,
                prepared,
                recovery: crate::replacement_recording_queue::PreparedReplacementRecordingRecovery::Request(request),
            }));
        }
        let required = request.required_queue_flags();
        if let Err(reason) = validate_queue_capabilities(self.lanes[&queue].flags, queue, required)
        {
            return Err(Box::new(PreparedReplacementAuxiliaryRecordingFailure {
                reason,
                prepared,
                recovery: crate::replacement_recording_queue::PreparedReplacementRecordingRecovery::Request(request),
            }));
        }
        dispatch_prepared_replacement_auxiliary_recording(
            &self.recording_workers,
            &self.timelines,
            prepared,
            request,
            auxiliary_waits,
        )
    }

    pub fn try_observe(&self) -> Vec<ReplacementTimelineObservation> {
        self.lanes
            .values()
            .filter_map(|lane| lane.completion.try_observe())
            .collect()
    }
}

fn validate_queue_capabilities(
    available: vk::QueueFlags,
    queue: QueueOwnerId,
    required: vk::QueueFlags,
) -> Result<(), crate::replacement_recording_queue::PreparedReplacementRecordingError> {
    if !available.contains(required) {
        return Err(crate::replacement_recording_queue::PreparedReplacementRecordingError::QueueCapabilitiesMissing {
            queue,
            required,
            available,
        });
    }
    Ok(())
}

fn validate_queue_family(
    bound_family: Option<u32>,
    queue: QueueOwnerId,
    requested_family: u32,
) -> Result<(), crate::replacement_recording_queue::PreparedReplacementRecordingError> {
    let Some(bound_family) = bound_family else {
        return Err(
            crate::replacement_recording_queue::PreparedReplacementRecordingError::QueueAbsent(
                queue,
            ),
        );
    };
    if bound_family != requested_family {
        return Err(crate::replacement_recording_queue::PreparedReplacementRecordingError::QueueFamilyMismatch {
            prepared: bound_family,
            request: requested_family,
        });
    }
    Ok(())
}

fn validate_bindings(
    bindings: &[ReplacementQueueBinding],
) -> Result<(), ReplacementQueueEpochStartError> {
    if bindings.is_empty() {
        return Err(ReplacementQueueEpochStartError::NoQueues);
    }
    let mut ids = BTreeSet::new();
    for binding in bindings {
        if !ids.insert(binding.id) {
            return Err(ReplacementQueueEpochStartError::DuplicateQueue(binding.id));
        }
    }
    Ok(())
}

fn validate_recording_worker_count(
    recording_worker_count: usize,
) -> Result<(), ReplacementQueueEpochStartError> {
    if recording_worker_count == 0 {
        Err(ReplacementQueueEpochStartError::RecordingWorkers(
            FixedExecutorError::NoWorkers,
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn binding(id: u32) -> ReplacementQueueBinding {
        ReplacementQueueBinding {
            id: QueueOwnerId::new(id),
            queue_family: id,
            flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            queue: vk::Queue::from_raw(u64::from(id) + 1),
            timeline: vk::Semaphore::from_raw(u64::from(id) + 10),
        }
    }

    #[test]
    fn queue_epoch_requires_a_nonempty_unique_actual_queue_set() {
        assert!(matches!(
            validate_bindings(&[]),
            Err(ReplacementQueueEpochStartError::NoQueues)
        ));
        assert!(matches!(
            validate_bindings(&[binding(2), binding(2)]),
            Err(ReplacementQueueEpochStartError::DuplicateQueue(id)) if id == QueueOwnerId::new(2)
        ));
        assert!(validate_bindings(&[binding(1), binding(2)]).is_ok());
    }

    #[test]
    fn queue_epoch_requires_a_nonempty_fixed_recording_population() {
        assert!(matches!(
            validate_recording_worker_count(0),
            Err(ReplacementQueueEpochStartError::RecordingWorkers(
                FixedExecutorError::NoWorkers
            ))
        ));
        assert!(validate_recording_worker_count(1).is_ok());
    }

    #[test]
    fn recording_family_must_be_the_actual_queue_bindings_family() {
        let queue = QueueOwnerId::new(3);
        assert_eq!(
            validate_queue_family(None, queue, 7),
            Err(
                crate::replacement_recording_queue::PreparedReplacementRecordingError::QueueAbsent(
                    queue
                )
            )
        );
        assert_eq!(
            validate_queue_family(Some(5), queue, 7),
            Err(crate::replacement_recording_queue::PreparedReplacementRecordingError::QueueFamilyMismatch {
                prepared: 5,
                request: 7,
            })
        );
        assert_eq!(validate_queue_family(Some(7), queue, 7), Ok(()));
        assert_eq!(
            validate_queue_capabilities(vk::QueueFlags::TRANSFER, queue, vk::QueueFlags::COMPUTE),
            Err(crate::replacement_recording_queue::PreparedReplacementRecordingError::QueueCapabilitiesMissing {
                queue,
                required: vk::QueueFlags::COMPUTE,
                available: vk::QueueFlags::TRANSFER,
            })
        );
        assert_eq!(
            validate_queue_capabilities(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                queue,
                vk::QueueFlags::COMPUTE,
            ),
            Ok(())
        );
    }
}
