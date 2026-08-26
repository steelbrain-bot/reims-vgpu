//! Ownership-preserving source-queue image release submission.
//!
//! A release is auxiliary native work for one semantic transaction. It shares
//! the source queue's timeline allocator but creates no semantic completion of
//! its own. Driver acceptance advances the image token into acquire-pending
//! state; refusal returns the auxiliary point, recording, and release plan.

use crate::{
    replacement_epoch::ReplacementQueueEpoch,
    replacement_image_state::{
        PreparedImageState, PreparedImageStateBatch, ReplacementImageReleaseKey,
        ReplacementImageStateError, ReplacementImageStateOwner,
    },
    replacement_image_transition::NativeImageRelease,
    replacement_queue::{
        PendingPreparedReplacementAuxiliaryQueueSubmit,
        PreparedReplacementAuxiliaryQueueEnqueueFailure,
        PreparedReplacementAuxiliaryQueueSubmission, ReplacementAuxiliaryQueuePreparationFailure,
        ReplacementQueueError,
    },
    replacement_recording::{
        dispatch_image_release_recording, PendingReplacementImageReleaseRecording,
        ReplacementImageReleaseRecordingFailure, ReplacementImageReleaseRecordingRequest,
        ReplacementNativeRecording, ReplacementRecordingDispatchError,
    },
    replacement_replay::ReplacementRecordingOwner,
    replacement_submit::QueueTimelineSemaphores,
};
use reims_vgpu_core::{
    DirectReplayError, DirectReplayNativeOwner, FixedExecutor, NativeRetirementDisposition,
    NativeRetirementError, PreparedAuxiliaryNativeSubmission, PreparedNativeSubmission,
    QueueTimelinePoint, RecordingWorkerId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub struct PreparedReplacementImageRelease {
    auxiliary: PreparedAuxiliaryNativeSubmission,
    release: NativeImageRelease,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementImageReleasePrepareError {
    SourceQueueMismatch,
    Native(DirectReplayError),
}

pub fn prepare_replacement_image_release<Semantic: Clone>(
    native: &mut DirectReplayNativeOwner<Semantic>,
    parent: &PreparedNativeSubmission<Semantic>,
    release: NativeImageRelease,
) -> Result<PreparedReplacementImageRelease, ReplacementImageReleasePrepareError> {
    if release.predecessor.queue != release.source_queue {
        return Err(ReplacementImageReleasePrepareError::SourceQueueMismatch);
    }
    let auxiliary = native
        .prepare_auxiliary_after(parent, release.predecessor)
        .map_err(ReplacementImageReleasePrepareError::Native)?;
    Ok(PreparedReplacementImageRelease { auxiliary, release })
}

#[derive(Debug)]
pub struct PreparedReplacementImageReleaseBatch {
    image_states: PreparedImageStateBatch,
    releases: Box<[PreparedReplacementImageRelease]>,
}

impl PreparedReplacementImageReleaseBatch {
    pub fn into_parts(
        self,
    ) -> (
        PreparedImageStateBatch,
        Box<[PreparedReplacementImageRelease]>,
    ) {
        (self.image_states, self.releases)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementImageReleaseBatchPrepareError {
    TransactionMismatch,
    DuplicateRelease(ReplacementImageReleaseKey),
    MissingRelease(ReplacementImageReleaseKey),
    UnexpectedRelease(ReplacementImageReleaseKey),
    SourcePointMismatch(ReplacementImageReleaseKey),
    Native(ReplacementImageReleasePrepareError),
}

#[derive(Debug)]
pub struct ReplacementImageReleaseBatchPrepareFailure {
    pub reason: ReplacementImageReleaseBatchPrepareError,
    pub image_states: PreparedImageStateBatch,
    /// Points in this prefix were allocated and are explicitly skipped if the
    /// caller does not submit them. They are never silently reused.
    pub prepared: Box<[PreparedReplacementImageRelease]>,
    pub remaining: Box<[NativeImageRelease]>,
}

/// Validate the complete release set before allocating any auxiliary point,
/// then reserve each exact source-queue successor. Allocation failure returns
/// the prepared prefix and untouched suffix with the image batch.
pub fn prepare_replacement_image_release_batch<Semantic: Clone>(
    native: &mut DirectReplayNativeOwner<Semantic>,
    parent: &PreparedNativeSubmission<Semantic>,
    image_states: PreparedImageStateBatch,
    releases: impl Into<Box<[NativeImageRelease]>>,
) -> Result<PreparedReplacementImageReleaseBatch, Box<ReplacementImageReleaseBatchPrepareFailure>> {
    let releases = releases.into();
    if image_states.transaction() != parent.plan().transaction {
        return Err(Box::new(ReplacementImageReleaseBatchPrepareFailure {
            reason: ReplacementImageReleaseBatchPrepareError::TransactionMismatch,
            image_states,
            prepared: Box::new([]),
            remaining: releases,
        }));
    }
    let mut expected = BTreeMap::<ReplacementImageReleaseKey, QueueTimelinePoint>::new();
    for operation in image_states.operations() {
        for transfer in operation
            .transitions()
            .iter()
            .filter_map(|transition| transition.queue_transfer)
        {
            let key = ReplacementImageReleaseKey {
                source_queue_family: transfer.source,
                source_queue: transfer.source_point.queue,
            };
            if operation.release_accepted(key) {
                continue;
            }
            expected
                .entry(key)
                .and_modify(|predecessor| {
                    if transfer.source_point.value > predecessor.value {
                        *predecessor = transfer.source_point;
                    }
                })
                .or_insert(transfer.source_point);
        }
    }
    let mut supplied = BTreeSet::new();
    for release in releases.iter() {
        let key = ReplacementImageReleaseKey {
            source_queue_family: release.source_queue_family,
            source_queue: release.source_queue,
        };
        let reason = if !supplied.insert(key) {
            Some(ReplacementImageReleaseBatchPrepareError::DuplicateRelease(
                key,
            ))
        } else {
            match expected.get(&key) {
                None => Some(ReplacementImageReleaseBatchPrepareError::UnexpectedRelease(
                    key,
                )),
                Some(predecessor) if *predecessor != release.predecessor => {
                    Some(ReplacementImageReleaseBatchPrepareError::SourcePointMismatch(key))
                }
                Some(_) => None,
            }
        };
        if let Some(reason) = reason {
            return Err(Box::new(ReplacementImageReleaseBatchPrepareFailure {
                reason,
                image_states,
                prepared: Box::new([]),
                remaining: releases,
            }));
        }
    }
    if let Some(missing) = expected.keys().find(|key| !supplied.contains(key)).copied() {
        return Err(Box::new(ReplacementImageReleaseBatchPrepareFailure {
            reason: ReplacementImageReleaseBatchPrepareError::MissingRelease(missing),
            image_states,
            prepared: Box::new([]),
            remaining: releases,
        }));
    }

    let mut remaining = releases.into_vec().into_iter();
    let mut prepared = Vec::new();
    while let Some(release) = remaining.next() {
        match prepare_replacement_image_release(native, parent, release.clone()) {
            Ok(release) => prepared.push(release),
            Err(reason) => {
                return Err(Box::new(ReplacementImageReleaseBatchPrepareFailure {
                    reason: ReplacementImageReleaseBatchPrepareError::Native(reason),
                    image_states,
                    prepared: prepared.into_boxed_slice(),
                    remaining: std::iter::once(release)
                        .chain(remaining)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }));
            }
        }
    }
    Ok(PreparedReplacementImageReleaseBatch {
        image_states,
        releases: prepared.into_boxed_slice(),
    })
}

#[derive(Debug)]
pub struct ReplacementImageReleaseDispatchFailure {
    pub reason: ReplacementRecordingDispatchError,
    pub prepared: PreparedReplacementImageRelease,
}

#[must_use = "a prepared image release recording must be observed"]
pub struct PendingPreparedReplacementImageRelease {
    auxiliary: PreparedAuxiliaryNativeSubmission,
    release: NativeImageRelease,
    pending: PendingReplacementImageReleaseRecording,
}

pub fn dispatch_prepared_image_release(
    executor: &FixedExecutor<crate::replacement_recording::ReplacementRecordingWorker>,
    prepared: PreparedReplacementImageRelease,
) -> Result<PendingPreparedReplacementImageRelease, Box<ReplacementImageReleaseDispatchFailure>> {
    let PreparedReplacementImageRelease { auxiliary, release } = prepared;
    let request = ReplacementImageReleaseRecordingRequest {
        transaction: auxiliary.transaction(),
        worker: auxiliary.recording_worker(),
        release: release.clone(),
    };
    match dispatch_image_release_recording(executor, request) {
        Ok(pending) => Ok(PendingPreparedReplacementImageRelease {
            auxiliary,
            release,
            pending,
        }),
        Err(failure) => {
            let ReplacementImageReleaseRecordingFailure { reason, .. } = *failure;
            Err(Box::new(ReplacementImageReleaseDispatchFailure {
                reason,
                prepared: PreparedReplacementImageRelease { auxiliary, release },
            }))
        }
    }
}

#[derive(Debug)]
pub struct RecordedReplacementImageRelease {
    auxiliary: PreparedAuxiliaryNativeSubmission,
    release: NativeImageRelease,
    recording: ReplacementNativeRecording,
}

impl PendingPreparedReplacementImageRelease {
    pub fn wait(
        self,
    ) -> Result<RecordedReplacementImageRelease, Box<ReplacementImageReleaseDispatchFailure>> {
        match self.pending.wait() {
            Ok(recording) => Ok(RecordedReplacementImageRelease {
                auxiliary: self.auxiliary,
                release: self.release,
                recording,
            }),
            Err(failure) => {
                let ReplacementImageReleaseRecordingFailure { reason, .. } = *failure;
                Err(Box::new(ReplacementImageReleaseDispatchFailure {
                    reason,
                    prepared: PreparedReplacementImageRelease {
                        auxiliary: self.auxiliary,
                        release: self.release,
                    },
                }))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementImageReleaseQueueError {
    WorkerMismatch {
        expected: RecordingWorkerId,
        actual: RecordingWorkerId,
    },
    QueueFamilyMismatch,
    Plan(crate::replacement_submit::TimelineSubmitPlanError),
    QueueAbsent,
    Enqueue(ReplacementQueueError),
    Driver(ReplacementQueueError),
}

#[derive(Debug)]
pub struct ReplacementImageReleaseQueueFailure {
    pub reason: ReplacementImageReleaseQueueError,
    pub recorded: RecordedReplacementImageRelease,
}

#[derive(Debug)]
pub struct PreparedReplacementImageReleaseQueue {
    release: NativeImageRelease,
    submission: PreparedReplacementAuxiliaryQueueSubmission,
}

pub fn prepare_recorded_image_release(
    timelines: &QueueTimelineSemaphores,
    recorded: RecordedReplacementImageRelease,
) -> Result<PreparedReplacementImageReleaseQueue, Box<ReplacementImageReleaseQueueFailure>> {
    if recorded.recording.worker != recorded.auxiliary.recording_worker() {
        return Err(Box::new(ReplacementImageReleaseQueueFailure {
            reason: ReplacementImageReleaseQueueError::WorkerMismatch {
                expected: recorded.auxiliary.recording_worker(),
                actual: recorded.recording.worker,
            },
            recorded,
        }));
    }
    if recorded.recording.queue_family != recorded.release.source_queue_family {
        return Err(Box::new(ReplacementImageReleaseQueueFailure {
            reason: ReplacementImageReleaseQueueError::QueueFamilyMismatch,
            recorded,
        }));
    }
    let RecordedReplacementImageRelease {
        auxiliary,
        release,
        recording,
    } = recorded;
    match PreparedReplacementAuxiliaryQueueSubmission::new(auxiliary, timelines, recording) {
        Ok(submission) => Ok(PreparedReplacementImageReleaseQueue {
            release,
            submission,
        }),
        Err(failure) => {
            let ReplacementAuxiliaryQueuePreparationFailure {
                reason,
                prepared,
                recording,
            } = *failure;
            Err(Box::new(ReplacementImageReleaseQueueFailure {
                reason: ReplacementImageReleaseQueueError::Plan(reason),
                recorded: RecordedReplacementImageRelease {
                    auxiliary: prepared,
                    release,
                    recording,
                },
            }))
        }
    }
}

#[must_use = "an enqueued image release must be observed to driver acceptance"]
pub struct PendingReplacementImageReleaseSubmit {
    release: NativeImageRelease,
    pending: PendingPreparedReplacementAuxiliaryQueueSubmit,
}

pub fn enqueue_prepared_image_release(
    queues: &ReplacementQueueEpoch,
    prepared: PreparedReplacementImageReleaseQueue,
) -> Result<PendingReplacementImageReleaseSubmit, Box<ReplacementImageReleaseQueueFailure>> {
    let PreparedReplacementImageReleaseQueue {
        release,
        submission,
    } = prepared;
    let Some(lane) = queues.lane(release.source_queue) else {
        let (auxiliary, recording) = submission.into_parts();
        return Err(Box::new(ReplacementImageReleaseQueueFailure {
            reason: ReplacementImageReleaseQueueError::QueueAbsent,
            recorded: RecordedReplacementImageRelease {
                auxiliary,
                release,
                recording,
            },
        }));
    };
    match lane.submit.submit_auxiliary(submission) {
        Ok(pending) => Ok(PendingReplacementImageReleaseSubmit { release, pending }),
        Err(failure) => {
            let PreparedReplacementAuxiliaryQueueEnqueueFailure { reason, submission } = *failure;
            let (auxiliary, recording) = submission.into_parts();
            Err(Box::new(ReplacementImageReleaseQueueFailure {
                reason: ReplacementImageReleaseQueueError::Enqueue(reason),
                recorded: RecordedReplacementImageRelease {
                    auxiliary,
                    release,
                    recording,
                },
            }))
        }
    }
}

#[derive(Debug)]
pub struct DriverAcceptedReplacementImageRelease {
    release: NativeImageRelease,
    submission: PreparedReplacementAuxiliaryQueueSubmission,
}

impl PendingReplacementImageReleaseSubmit {
    pub fn wait(
        self,
    ) -> Result<DriverAcceptedReplacementImageRelease, Box<ReplacementImageReleaseQueueFailure>>
    {
        match self.pending.wait() {
            Ok(submission) => Ok(DriverAcceptedReplacementImageRelease {
                release: self.release,
                submission,
            }),
            Err(failure) => {
                let (reason, submission) = *failure;
                let (auxiliary, recording) = submission.into_parts();
                Err(Box::new(ReplacementImageReleaseQueueFailure {
                    reason: ReplacementImageReleaseQueueError::Driver(reason),
                    recorded: RecordedReplacementImageRelease {
                        auxiliary,
                        release: self.release,
                        recording,
                    },
                }))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementImageReleaseAcceptanceError {
    TransactionMismatch,
    SourcePointMismatch,
    AuxiliaryPointOrder,
    ImageState(ReplacementImageStateError),
    Recording(NativeRetirementError),
    QueueAbsent,
    Watch(crate::replacement_completion::ReplacementTimelineWatchError),
}

#[derive(Debug)]
pub struct ReplacementImageReleaseAcceptanceFailure {
    pub reason: ReplacementImageReleaseAcceptanceError,
    pub accepted: DriverAcceptedReplacementImageRelease,
    pub image_state: PreparedImageState,
}

#[derive(Debug)]
pub struct AcceptedReplacementImageRelease {
    pub image_state: PreparedImageState,
    pub ready_recording: Option<ReplacementNativeRecording>,
}

#[derive(Debug)]
pub struct ReplacementImageBatchReleaseAcceptanceFailure {
    pub reason: ReplacementImageReleaseAcceptanceError,
    pub accepted: DriverAcceptedReplacementImageRelease,
    pub image_states: PreparedImageStateBatch,
}

#[derive(Debug)]
pub struct AcceptedReplacementImageBatchRelease {
    pub image_states: PreparedImageStateBatch,
    pub ready_recording: Option<ReplacementNativeRecording>,
}

pub fn commit_driver_accepted_image_batch_release(
    images: &mut ReplacementImageStateOwner,
    recordings: &mut ReplacementRecordingOwner,
    queues: &mut ReplacementQueueEpoch,
    accepted: DriverAcceptedReplacementImageRelease,
    image_states: PreparedImageStateBatch,
) -> Result<AcceptedReplacementImageBatchRelease, Box<ReplacementImageBatchReleaseAcceptanceFailure>>
{
    let point = accepted.submission.prepared.point();
    let Some(lane) = queues.lane(point.queue) else {
        return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::QueueAbsent,
            accepted,
            image_states,
        }));
    };
    commit_driver_accepted_image_batch_release_with_watch(
        images,
        recordings,
        accepted,
        image_states,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_image_batch_release_with_watch(
    images: &mut ReplacementImageStateOwner,
    recordings: &mut ReplacementRecordingOwner,
    accepted: DriverAcceptedReplacementImageRelease,
    image_states: PreparedImageStateBatch,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> Result<AcceptedReplacementImageBatchRelease, Box<ReplacementImageBatchReleaseAcceptanceFailure>>
{
    let transaction = accepted.submission.prepared.transaction();
    let point = accepted.submission.prepared.point();
    if transaction != image_states.transaction() {
        return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::TransactionMismatch,
            accepted,
            image_states,
        }));
    }
    let release_key = ReplacementImageReleaseKey {
        source_queue_family: accepted.release.source_queue_family,
        source_queue: accepted.release.source_queue,
    };
    let predecessor = match images.validate_batch_release(&image_states, release_key) {
        Ok(predecessor) => predecessor,
        Err(reason) => {
            return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
                reason: ReplacementImageReleaseAcceptanceError::ImageState(reason),
                accepted,
                image_states,
            }));
        }
    };
    if predecessor != accepted.release.predecessor {
        return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::SourcePointMismatch,
            accepted,
            image_states,
        }));
    }
    if point.queue != accepted.release.source_queue
        || point.epoch != predecessor.epoch
        || point.value <= predecessor.value
    {
        return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::AuxiliaryPointOrder,
            accepted,
            image_states,
        }));
    }
    if let Err(reason) = recordings.validate_auxiliary_accept(transaction, point) {
        return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::Recording(reason),
            accepted,
            image_states,
        }));
    }
    if let Err(reason) = watch(point) {
        return Err(Box::new(ReplacementImageBatchReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::Watch(reason),
            accepted,
            image_states,
        }));
    }
    let DriverAcceptedReplacementImageRelease {
        release: _,
        submission,
    } = accepted;
    let (_, recording) = submission.into_parts();
    let ready_recording = match recordings
        .accept_auxiliary(transaction, point, recording)
        .unwrap_or_else(|_| unreachable!("auxiliary recording acceptance was prevalidated"))
    {
        NativeRetirementDisposition::Deferred => None,
        NativeRetirementDisposition::Ready(recording) => Some(recording),
    };
    let image_states = images
        .batch_release_accepted(image_states, release_key, point)
        .unwrap_or_else(|_| unreachable!("image batch release acceptance was prevalidated"));
    Ok(AcceptedReplacementImageBatchRelease {
        image_states,
        ready_recording,
    })
}

pub fn commit_driver_accepted_image_release(
    images: &mut ReplacementImageStateOwner,
    recordings: &mut ReplacementRecordingOwner,
    queues: &mut ReplacementQueueEpoch,
    accepted: DriverAcceptedReplacementImageRelease,
    image_state: PreparedImageState,
) -> Result<AcceptedReplacementImageRelease, Box<ReplacementImageReleaseAcceptanceFailure>> {
    let point = accepted.submission.prepared.point();
    let Some(lane) = queues.lane(point.queue) else {
        return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::QueueAbsent,
            accepted,
            image_state,
        }));
    };
    commit_driver_accepted_image_release_with_watch(
        images,
        recordings,
        accepted,
        image_state,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_image_release_with_watch(
    images: &mut ReplacementImageStateOwner,
    recordings: &mut ReplacementRecordingOwner,
    accepted: DriverAcceptedReplacementImageRelease,
    image_state: PreparedImageState,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> Result<AcceptedReplacementImageRelease, Box<ReplacementImageReleaseAcceptanceFailure>> {
    let transaction = accepted.submission.prepared.transaction();
    let point = accepted.submission.prepared.point();
    if transaction != image_state.transaction() {
        return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::TransactionMismatch,
            accepted,
            image_state,
        }));
    }
    let release_key = ReplacementImageReleaseKey {
        source_queue_family: accepted.release.source_queue_family,
        source_queue: accepted.release.source_queue,
    };
    let predecessor = match images.validate_release(&image_state, release_key) {
        Ok(predecessor) => predecessor,
        Err(reason) => {
            return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
                reason: ReplacementImageReleaseAcceptanceError::ImageState(reason),
                accepted,
                image_state,
            }));
        }
    };
    if predecessor != accepted.release.predecessor {
        return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::SourcePointMismatch,
            accepted,
            image_state,
        }));
    }
    if point.queue != accepted.release.source_queue
        || point.epoch != predecessor.epoch
        || point.value <= predecessor.value
    {
        return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::AuxiliaryPointOrder,
            accepted,
            image_state,
        }));
    }
    if let Err(reason) = recordings.validate_auxiliary_accept(transaction, point) {
        return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::Recording(reason),
            accepted,
            image_state,
        }));
    }
    if let Err(reason) = watch(point) {
        return Err(Box::new(ReplacementImageReleaseAcceptanceFailure {
            reason: ReplacementImageReleaseAcceptanceError::Watch(reason),
            accepted,
            image_state,
        }));
    }
    let DriverAcceptedReplacementImageRelease {
        release: _,
        submission,
    } = accepted;
    let (_, recording) = submission.into_parts();
    let ready_recording = match recordings
        .accept_auxiliary(transaction, point, recording)
        .unwrap_or_else(|_| unreachable!("auxiliary recording acceptance was prevalidated"))
    {
        NativeRetirementDisposition::Deferred => None,
        NativeRetirementDisposition::Ready(recording) => Some(recording),
    };
    let image_state = images
        .release_accepted(image_state, release_key, point)
        .unwrap_or_else(|_| unreachable!("image release acceptance was prevalidated"));
    Ok(AcceptedReplacementImageRelease {
        image_state,
        ready_recording,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_barrier_record::NativeBarrierBatch;
    use crate::replacement_image_state::{
        ReplacementImageKey, ReplacementImageSharing, ReplacementImageState, ReplacementImageUse,
    };
    use crate::replacement_recording::ReplacementNativeRecording;
    use crate::replacement_submit::QueueTimelineSemaphores;
    use ash::vk;
    use reims_vgpu_core::{QueueTimelinePoint, TransactionRecordingPlan, WaitDependencyCause};
    use reims_vgpu_protocol::{
        BackingId, QueueOwnerId, QueueTimelineValue, RepresentationId, SessionGenerationId,
        SubmissionDomainId, TransactionId, VulkanDeviceEpochId,
    };

    fn parent_and_predecessor() -> (
        DirectReplayNativeOwner<&'static str>,
        PreparedNativeSubmission<&'static str>,
        QueueTimelinePoint,
    ) {
        let epoch = VulkanDeviceEpochId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: TransactionId::new(1),
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let producer = native
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let producer = native
            .prepare(
                producer,
                QueueOwnerId::new(1),
                SessionGenerationId::new(1),
                "producer",
            )
            .unwrap();
        let predecessor = native.accepted(producer).unwrap().point;
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: TransactionId::new(2),
                domain: SubmissionDomainId::new(2),
                continuation_predecessor: None,
            })
            .unwrap();
        let parent = native
            .queue_candidate(
                TransactionId::new(2),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let parent = native
            .prepare(
                parent,
                QueueOwnerId::new(2),
                SessionGenerationId::new(1),
                "consumer",
            )
            .unwrap();
        (native, parent, predecessor)
    }

    fn release(predecessor: QueueTimelinePoint) -> NativeImageRelease {
        let mut barriers = NativeBarrierBatch::default();
        barriers.memory.push(
            vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
        );
        NativeImageRelease {
            source_queue_family: 3,
            source_queue: predecessor.queue,
            predecessor,
            barriers,
        }
    }

    #[test]
    fn release_reserves_the_next_source_point_and_keeps_worker_identity() {
        let (mut native, parent, predecessor) = parent_and_predecessor();
        let prepared =
            prepare_replacement_image_release(&mut native, &parent, release(predecessor)).unwrap();
        assert_eq!(prepared.auxiliary.transaction(), TransactionId::new(2));
        assert_eq!(
            prepared.auxiliary.recording_worker(),
            parent.recording_worker()
        );
        assert_eq!(prepared.auxiliary.point().queue, QueueOwnerId::new(1));
        assert_eq!(prepared.auxiliary.point().value, QueueTimelineValue::new(2));
    }

    #[test]
    fn malformed_release_batch_allocates_no_auxiliary_point() {
        let epoch = VulkanDeviceEpochId::new(1);
        let (mut native, parent, predecessor) = parent_and_predecessor();
        let image = ReplacementImageKey {
            backing: BackingId::new(1),
            representation: RepresentationId::new(2),
        };
        let mut images = ReplacementImageStateOwner::new(epoch);
        images
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 3 },
                    last_use: Some(predecessor),
                },
            )
            .unwrap();
        let prepare_states = |images: &mut ReplacementImageStateOwner| {
            images
                .prepare_batch(
                    TransactionId::new(2),
                    4,
                    vec![(
                        0,
                        Box::new([ReplacementImageUse {
                            image,
                            required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                            use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            final_layout: vk::ImageLayout::GENERAL,
                        }]) as Box<[_]>,
                    )]
                    .into_boxed_slice(),
                )
                .unwrap()
        };
        let states = prepare_states(&mut images);
        let release = release(predecessor);
        let failure = prepare_replacement_image_release_batch(
            &mut native,
            &parent,
            states,
            vec![release.clone(), release.clone()].into_boxed_slice(),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementImageReleaseBatchPrepareError::DuplicateRelease(
                ReplacementImageReleaseKey {
                    source_queue_family: 3,
                    source_queue: QueueOwnerId::new(1),
                }
            )
        );
        assert!(failure.prepared.is_empty());
        assert_eq!(failure.remaining.len(), 2);

        images.cancel_batch(failure.image_states).unwrap();
        let prepared = prepare_replacement_image_release_batch(
            &mut native,
            &parent,
            prepare_states(&mut images),
            vec![release].into_boxed_slice(),
        )
        .unwrap();
        let (_, releases) = prepared.into_parts();
        assert_eq!(
            releases[0].auxiliary.point().value,
            QueueTimelineValue::new(2)
        );
    }

    #[test]
    fn unissued_source_predecessor_refuses_before_an_auxiliary_point_exists() {
        let (mut native, parent, _) = parent_and_predecessor();
        let fabricated = QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(1),
            queue: QueueOwnerId::new(4),
            value: QueueTimelineValue::new(1),
        };
        assert!(matches!(
            prepare_replacement_image_release(&mut native, &parent, release(fabricated)),
            Err(ReplacementImageReleasePrepareError::Native(
                DirectReplayError::Timeline(
                    reims_vgpu_core::QueueTimelineError::PredecessorNotAllocated
                )
            ))
        ));
    }

    #[test]
    fn driver_acceptance_moves_release_to_acquire_pending_and_retires_its_recording() {
        let epoch = VulkanDeviceEpochId::new(1);
        let (mut native, parent, predecessor) = parent_and_predecessor();
        let release = release(predecessor);
        let prepared_release =
            prepare_replacement_image_release(&mut native, &parent, release.clone()).unwrap();
        let PreparedReplacementImageRelease { auxiliary, .. } = prepared_release;
        let point = auxiliary.point();
        let timelines =
            QueueTimelineSemaphores::new(epoch, [(QueueOwnerId::new(1), vk::Semaphore::null())]);
        let submission = PreparedReplacementAuxiliaryQueueSubmission::new(
            auxiliary,
            &timelines,
            ReplacementNativeRecording::synthetic(
                parent.recording_worker(),
                Box::<[vk::CommandBuffer]>::default(),
                vk::Fence::null(),
            ),
        )
        .unwrap();
        let accepted = DriverAcceptedReplacementImageRelease {
            release,
            submission,
        };
        let image = ReplacementImageKey {
            backing: BackingId::new(1),
            representation: RepresentationId::new(2),
        };
        let mut images = ReplacementImageStateOwner::new(epoch);
        images
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 3 },
                    last_use: Some(predecessor),
                },
            )
            .unwrap();
        let image_state = images
            .prepare(
                TransactionId::new(2),
                4,
                [ReplacementImageUse {
                    image,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                }],
            )
            .unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let accepted = commit_driver_accepted_image_release_with_watch(
            &mut images,
            &mut recordings,
            accepted,
            image_state,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(
            accepted.image_state.accepted_releases(),
            [crate::replacement_image_state::AcceptedImageRelease {
                release: ReplacementImageReleaseKey {
                    source_queue_family: 3,
                    source_queue: QueueOwnerId::new(1),
                },
                point,
            }]
        );
        assert!(accepted.ready_recording.is_none());
        assert_eq!(
            images.state(image).unwrap().sharing,
            ReplacementImageSharing::Exclusive { owner: 3 }
        );
        assert_eq!(
            recordings.advance(point.queue, point.value).unwrap().len(),
            1
        );
    }

    #[test]
    fn one_driver_accepted_release_advances_every_matching_image_in_a_batch() {
        let epoch = VulkanDeviceEpochId::new(1);
        let (mut native, parent, predecessor) = parent_and_predecessor();
        let release = release(predecessor);
        let prepared_release =
            prepare_replacement_image_release(&mut native, &parent, release.clone()).unwrap();
        let PreparedReplacementImageRelease { auxiliary, .. } = prepared_release;
        let point = auxiliary.point();
        let timelines =
            QueueTimelineSemaphores::new(epoch, [(QueueOwnerId::new(1), vk::Semaphore::null())]);
        let submission = PreparedReplacementAuxiliaryQueueSubmission::new(
            auxiliary,
            &timelines,
            ReplacementNativeRecording::synthetic(
                parent.recording_worker(),
                Box::<[vk::CommandBuffer]>::default(),
                vk::Fence::null(),
            ),
        )
        .unwrap();
        let accepted = DriverAcceptedReplacementImageRelease {
            release,
            submission,
        };
        let first = ReplacementImageKey {
            backing: BackingId::new(1),
            representation: RepresentationId::new(2),
        };
        let second = ReplacementImageKey {
            backing: BackingId::new(3),
            representation: RepresentationId::new(4),
        };
        let mut images = ReplacementImageStateOwner::new(epoch);
        for image in [first, second] {
            images
                .register(
                    image,
                    ReplacementImageState {
                        layout: vk::ImageLayout::GENERAL,
                        sharing: ReplacementImageSharing::Exclusive { owner: 3 },
                        last_use: Some(predecessor),
                    },
                )
                .unwrap();
        }
        let use_ = |image| ReplacementImageUse {
            image,
            required_usage: vk::ImageUsageFlags::TRANSFER_DST,
            use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            final_layout: vk::ImageLayout::GENERAL,
        };
        let image_states = images
            .prepare_batch(
                TransactionId::new(2),
                4,
                vec![
                    (0, Box::new([use_(first)]) as Box<[_]>),
                    (1, Box::new([use_(second)]) as Box<[_]>),
                ]
                .into_boxed_slice(),
            )
            .unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let accepted = commit_driver_accepted_image_batch_release_with_watch(
            &mut images,
            &mut recordings,
            accepted,
            image_states,
            |_| Ok(()),
        )
        .unwrap();

        assert_eq!(accepted.image_states.release_points().as_ref(), [point]);
        for operation in accepted.image_states.operations() {
            assert_eq!(
                operation.accepted_releases(),
                [crate::replacement_image_state::AcceptedImageRelease {
                    release: ReplacementImageReleaseKey {
                        source_queue_family: 3,
                        source_queue: QueueOwnerId::new(1),
                    },
                    point,
                }]
            );
        }
        assert!(accepted.ready_recording.is_none());
        assert_eq!(
            recordings.advance(point.queue, point.value).unwrap().len(),
            1
        );
    }
}
