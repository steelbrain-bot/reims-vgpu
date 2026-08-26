//! Join actual queue-timeline observations to every replacement replay owner.

use crate::{
    replacement_completion::{
        ReplacementTimelineFailure, ReplacementTimelineObservation, ReplacementTimelineWatchError,
    },
    replacement_epoch::ReplacementQueueEpoch,
    replacement_indirect_range::RetiredReplacementIndirectRanges,
    replacement_queue::{
        PendingPreparedReplacementQueueSubmit, PreparedReplacementQueueEnqueueFailure,
        PreparedReplacementQueueSubmission, PreparedReplacementQueueSubmitPoll,
        ReplacementQueueError,
    },
    replacement_recording::ReplacementNativeRecording,
    replacement_recording::ReplacementRecordingRecycleFailure,
};
use reims_vgpu_core::{
    commit_replay_acceptance, validate_replay_acceptance, DirectReplayNativeOwner, FixedExecutor,
    FixedExecutorError, HostLandingBatchError, NativeRetirement, NativeRetirementDisposition,
    NativeRetirementError, NativeRetirementFailure, QueueTimelinePoint, RecordingWorkerId,
    ReplayAcceptance, ReplayAcceptanceError, ReplayTimelineProgress, ReplayTimelineProgressError,
    ResolvedReplayCompletion, ResolvedResourceCompletion, ResourceCompletionBatchError,
    ResourceCompletionEffect, ResourceLifecycleOwner, TransactionRuntime,
};
use reims_vgpu_protocol::{
    BackingId, QueueOwnerId, QueueTimelineValue, TransactionId, VulkanDeviceEpochId,
};
use std::{
    collections::BTreeSet,
    sync::{mpsc, Arc, Mutex},
};

#[derive(Debug)]
pub struct ReplacementRecordingFailure {
    pub reason: NativeRetirementError,
    pub transaction: TransactionId,
    pub recording: ReplacementNativeRecording,
}

#[derive(Debug)]
pub struct ReplacementRecordingCleanupFailure {
    pub reason: FixedExecutorError,
    pub recording: Box<ReplacementNativeRecording>,
}

#[derive(Debug)]
pub enum ReplacementRecordingCleanupCompletionError {
    Native(ReplacementRecordingRecycleFailure),
    WorkerStopped,
}

#[must_use = "native cleanup completion must be observed"]
pub struct PendingReplacementRecordingCleanup {
    receiver: mpsc::Receiver<Result<(), ReplacementRecordingRecycleFailure>>,
}

pub enum ReplacementRecordingCleanupPoll {
    Pending(PendingReplacementRecordingCleanup),
    Completed(Result<(), ReplacementRecordingCleanupCompletionError>),
}

impl PendingReplacementRecordingCleanup {
    pub fn try_complete(self) -> ReplacementRecordingCleanupPoll {
        match self.receiver.try_recv() {
            Ok(result) => ReplacementRecordingCleanupPoll::Completed(
                result.map_err(ReplacementRecordingCleanupCompletionError::Native),
            ),
            Err(mpsc::TryRecvError::Empty) => ReplacementRecordingCleanupPoll::Pending(self),
            Err(mpsc::TryRecvError::Disconnected) => ReplacementRecordingCleanupPoll::Completed(
                Err(ReplacementRecordingCleanupCompletionError::WorkerStopped),
            ),
        }
    }

    pub fn wait(self) -> Result<(), ReplacementRecordingCleanupCompletionError> {
        self.receiver
            .recv()
            .map_err(|_| ReplacementRecordingCleanupCompletionError::WorkerStopped)?
            .map_err(ReplacementRecordingCleanupCompletionError::Native)
    }
}

/// Return a retired recording to the fixed worker that owns its command pools
/// and descriptor arena. Failed dispatch returns the exact recording; no raw
/// handle remains hidden in a dropped closure.
pub fn recycle_replacement_recording<W: Send + 'static>(
    executor: &FixedExecutor<W>,
    recording: ReplacementNativeRecording,
    recycle: impl FnOnce(&mut W, ReplacementNativeRecording) + Send + 'static,
) -> Result<(), ReplacementRecordingCleanupFailure> {
    let worker = recording.worker;
    let recovery = Arc::new(Mutex::new(Some(recording)));
    let worker_recording = Arc::clone(&recovery);
    if let Err(reason) = executor.submit_to(worker, move |state| {
        let recording = worker_recording
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .expect("cleanup job owns one recording");
        recycle(state, recording);
    }) {
        let recording = recovery
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .expect("refused cleanup retains its recording");
        return Err(ReplacementRecordingCleanupFailure {
            reason,
            recording: Box::new(recording),
        });
    }
    Ok(())
}

/// Dispatch actual Vulkan destruction to the epoch worker that allocated the
/// recording. Native validation failures remain observable and return the
/// complete recording through the cleanup receipt.
pub fn recycle_epoch_recording(
    queues: &ReplacementQueueEpoch,
    recording: ReplacementNativeRecording,
) -> Result<PendingReplacementRecordingCleanup, ReplacementRecordingCleanupFailure> {
    let (sender, receiver) = mpsc::sync_channel(1);
    recycle_replacement_recording(
        queues.recording_workers(),
        recording,
        move |worker, recording| {
            let result = worker.recycle(recording);
            let _ = sender.send(result);
        },
    )?;
    Ok(PendingReplacementRecordingCleanup { receiver })
}

/// Timeline retirement for command buffers and fences accepted by the driver.
pub struct ReplacementRecordingOwner {
    epoch: VulkanDeviceEpochId,
    retirement: NativeRetirement<ReplacementRecordingTicket, ReplacementNativeRecording>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ReplacementRecordingTicket {
    Semantic(TransactionId),
    Auxiliary {
        transaction: TransactionId,
        point: QueueTimelinePoint,
    },
}

impl ReplacementRecordingOwner {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            epoch,
            retirement: NativeRetirement::new(epoch),
        }
    }

    pub fn accept(
        &mut self,
        transaction: TransactionId,
        point: QueueTimelinePoint,
        recording: ReplacementNativeRecording,
    ) -> Result<
        NativeRetirementDisposition<ReplacementNativeRecording>,
        Box<ReplacementRecordingFailure>,
    > {
        if point.epoch != self.epoch {
            return Err(Box::new(ReplacementRecordingFailure {
                reason: NativeRetirementError::MixedEpochs,
                transaction,
                recording,
            }));
        }
        match self.retirement.defer(
            ReplacementRecordingTicket::Semantic(transaction),
            recording,
            [point],
        ) {
            Ok(disposition) => Ok(disposition),
            Err(NativeRetirementFailure { reason, value }) => {
                Err(Box::new(ReplacementRecordingFailure {
                    reason,
                    transaction,
                    recording: value,
                }))
            }
        }
    }

    pub fn validate_accept(
        &self,
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<(), NativeRetirementError> {
        self.retirement.validate_defer(
            &ReplacementRecordingTicket::Semantic(transaction),
            &BTreeSet::from([point]),
        )
    }

    pub(crate) fn acceptance_is_ready(&self, point: QueueTimelinePoint) -> bool {
        self.retirement
            .obligations_are_complete(&BTreeSet::from([point]))
    }

    fn resource_completions_after(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Box<[ResolvedResourceCompletion]>, NativeRetirementError> {
        Ok(self
            .retirement
            .values_ready_after(queue, completed)?
            .into_iter()
            .flat_map(|recording| recording.resource_completions.iter().copied())
            .collect())
    }

    fn host_landing_programs_after(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<
        Box<[crate::replacement_resource_state::ReplacementHostLandingProgram]>,
        NativeRetirementError,
    > {
        Ok(self
            .retirement
            .values_ready_after(queue, completed)?
            .into_iter()
            .flat_map(|recording| recording.host_landing_programs.iter().cloned())
            .collect())
    }

    pub fn validate_auxiliary_accept(
        &self,
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<(), NativeRetirementError> {
        self.retirement.validate_defer(
            &ReplacementRecordingTicket::Auxiliary { transaction, point },
            &BTreeSet::from([point]),
        )
    }

    pub fn accept_auxiliary(
        &mut self,
        transaction: TransactionId,
        point: QueueTimelinePoint,
        recording: ReplacementNativeRecording,
    ) -> Result<
        NativeRetirementDisposition<ReplacementNativeRecording>,
        Box<ReplacementRecordingFailure>,
    > {
        if point.epoch != self.epoch {
            return Err(Box::new(ReplacementRecordingFailure {
                reason: NativeRetirementError::MixedEpochs,
                transaction,
                recording,
            }));
        }
        match self.retirement.defer(
            ReplacementRecordingTicket::Auxiliary { transaction, point },
            recording,
            [point],
        ) {
            Ok(disposition) => Ok(disposition),
            Err(NativeRetirementFailure { reason, value }) => {
                Err(Box::new(ReplacementRecordingFailure {
                    reason,
                    transaction,
                    recording: value,
                }))
            }
        }
    }

    pub fn validate_advance(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<(), NativeRetirementError> {
        self.retirement.validate_advance(queue, completed)
    }

    pub fn advance(
        &mut self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Vec<ReplacementNativeRecording>, NativeRetirementError> {
        self.retirement
            .advance(queue, completed)
            .map(|ready| ready.into_iter().map(|(_, recording)| recording).collect())
    }

    pub fn abandon(self) -> Vec<ReplacementNativeRecording> {
        self.retirement
            .abandon()
            .into_iter()
            .map(|(_, recording)| recording)
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementReplayAcceptanceError {
    QueueAbsent(QueueOwnerId),
    Core(ReplayAcceptanceError),
    Recording(NativeRetirementError),
    MissingRecordingWorker(TransactionId),
    RecordingWorkerMismatch {
        expected: RecordingWorkerId,
        actual: RecordingWorkerId,
    },
    ResourceCompletionBackingAbsent(BackingId),
    Watch(ReplacementTimelineWatchError),
    ResourceCompletions(ResourceCompletionBatchError),
}

#[derive(Debug)]
pub struct ReplacementReplayAcceptanceFailure<Semantic> {
    pub reason: ReplacementReplayAcceptanceError,
    pub submission: PreparedReplacementQueueSubmission<Semantic>,
}

#[derive(Debug)]
pub struct AcceptedReplacementReplay<T> {
    pub replay: ReplayAcceptance<T>,
    pub resource_completions: Vec<ResourceCompletionEffect>,
    /// A point may already have been observed before its driver receipt was
    /// consumed. Such a recording is immediately ready for owner cleanup.
    pub ready_recording: Option<ReplacementNativeRecording>,
}

impl<T> AcceptedReplacementReplay<T> {
    /// Take range staging only from a recording whose timeline point had
    /// already completed when the driver receipt was accepted.
    pub fn take_ready_indirect_ranges(
        &mut self,
        transaction: TransactionId,
    ) -> RetiredReplacementIndirectRanges {
        let programs = self
            .ready_recording
            .as_mut()
            .map(|recording| recording.take_indirect_range_programs_for(transaction))
            .unwrap_or_default();
        RetiredReplacementIndirectRanges::new(programs)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementReplayEnqueueError {
    QueueAbsent(QueueOwnerId),
    Queue(ReplacementQueueError),
}

#[derive(Debug)]
pub struct ReplacementReplayEnqueueFailure<Semantic> {
    pub reason: ReplacementReplayEnqueueError,
    pub submission: PreparedReplacementQueueSubmission<Semantic>,
}

/// Driver-receipt ownership for one enqueued replacement replay. The recording
/// retains its lifecycle-derived backing leases until the queue owner either
/// returns acceptance or gives all native ownership back.
#[must_use = "an enqueued replay must be polled to driver acceptance or explicit refusal"]
pub struct PendingReplacementReplaySubmit<Semantic> {
    pending: PendingPreparedReplacementQueueSubmit<Semantic>,
}

#[cfg(test)]
impl<Semantic> PendingReplacementReplaySubmit<Semantic> {
    pub(crate) fn disconnected_for_test(
        submission: PreparedReplacementQueueSubmission<Semantic>,
    ) -> Self {
        Self {
            pending: PendingPreparedReplacementQueueSubmit::disconnected_for_test(submission),
        }
    }
}

pub enum ReplacementReplaySubmitPoll<Semantic, T> {
    Pending(PendingReplacementReplaySubmit<Semantic>),
    Accepted(AcceptedReplacementReplay<T>),
    DriverRefused {
        reason: ReplacementQueueError,
        submission: PreparedReplacementQueueSubmission<Semantic>,
    },
    AcceptanceRefused(Box<ReplacementReplayAcceptanceFailure<Semantic>>),
}

#[derive(Debug)]
pub struct DriverAcceptedReplacementReplay<Semantic> {
    submission: PreparedReplacementQueueSubmission<Semantic>,
}

impl<Semantic> DriverAcceptedReplacementReplay<Semantic> {
    pub(crate) fn into_submission(self) -> PreparedReplacementQueueSubmission<Semantic> {
        self.submission
    }
}

pub enum ReplacementReplayDriverPoll<Semantic> {
    Pending(PendingReplacementReplaySubmit<Semantic>),
    DriverAccepted(DriverAcceptedReplacementReplay<Semantic>),
    DriverRefused {
        reason: ReplacementQueueError,
        submission: PreparedReplacementQueueSubmission<Semantic>,
    },
}

/// Transfer one prepared submission to the physical queue owner. Its native
/// recording already owns the exact lifecycle-derived backing set; queue
/// admission refusal returns that complete ownership unchanged.
pub fn enqueue_replacement_replay<Semantic>(
    queues: &ReplacementQueueEpoch,
    submission: PreparedReplacementQueueSubmission<Semantic>,
) -> Result<PendingReplacementReplaySubmit<Semantic>, Box<ReplacementReplayEnqueueFailure<Semantic>>>
{
    let queue = submission.prepared.point().queue;
    let Some(lane) = queues.lane(queue) else {
        return Err(Box::new(ReplacementReplayEnqueueFailure {
            reason: ReplacementReplayEnqueueError::QueueAbsent(queue),
            submission,
        }));
    };
    match lane.submit.submit_prepared(submission) {
        Ok(pending) => Ok(PendingReplacementReplaySubmit { pending }),
        Err(failure) => {
            let PreparedReplacementQueueEnqueueFailure { reason, submission } = *failure;
            Err(Box::new(ReplacementReplayEnqueueFailure {
                reason: ReplacementReplayEnqueueError::Queue(reason),
                submission,
            }))
        }
    }
}

impl<Semantic: Clone> PendingReplacementReplaySubmit<Semantic> {
    pub fn poll_driver(self) -> ReplacementReplayDriverPoll<Semantic> {
        match self.pending.try_complete() {
            PreparedReplacementQueueSubmitPoll::Pending(pending) => {
                ReplacementReplayDriverPoll::Pending(Self { pending })
            }
            PreparedReplacementQueueSubmitPoll::DriverRefused { reason, submission } => {
                ReplacementReplayDriverPoll::DriverRefused { reason, submission }
            }
            PreparedReplacementQueueSubmitPoll::DriverAccepted(submission) => {
                ReplacementReplayDriverPoll::DriverAccepted(DriverAcceptedReplacementReplay {
                    submission,
                })
            }
        }
    }

    /// Poll the driver receipt and, only after driver acceptance, atomically
    /// join semantic, backing, completion-watch, and native-recording owners.
    pub fn try_complete<T>(
        self,
        runtime: &mut TransactionRuntime<Semantic>,
        native: &mut DirectReplayNativeOwner<Semantic>,
        resources: &mut ResourceLifecycleOwner<T>,
        recordings: &mut ReplacementRecordingOwner,
        queues: &mut ReplacementQueueEpoch,
    ) -> ReplacementReplaySubmitPoll<Semantic, T> {
        match self.poll_driver() {
            ReplacementReplayDriverPoll::Pending(pending) => {
                ReplacementReplaySubmitPoll::Pending(pending)
            }
            ReplacementReplayDriverPoll::DriverRefused { reason, submission } => {
                ReplacementReplaySubmitPoll::DriverRefused { reason, submission }
            }
            ReplacementReplayDriverPoll::DriverAccepted(accepted) => {
                match commit_driver_accepted_replacement(
                    runtime,
                    native,
                    resources,
                    recordings,
                    queues,
                    accepted.submission,
                ) {
                    Ok(accepted) => ReplacementReplaySubmitPoll::Accepted(accepted),
                    Err(failure) => ReplacementReplaySubmitPoll::AcceptanceRefused(failure),
                }
            }
        }
    }
}

pub(crate) fn commit_driver_accepted_with_watch<
    RuntimeCompletion: Clone,
    NativeSemantic: Clone,
    T,
>(
    runtime: &mut TransactionRuntime<RuntimeCompletion>,
    native: &mut DirectReplayNativeOwner<NativeSemantic>,
    resources: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    submission: PreparedReplacementQueueSubmission<NativeSemantic>,
    watch: impl FnOnce(QueueTimelinePoint) -> Result<(), ReplacementTimelineWatchError>,
) -> Result<AcceptedReplacementReplay<T>, Box<ReplacementReplayAcceptanceFailure<NativeSemantic>>> {
    let backings = submission.recording().backings.clone();
    let transaction = submission.prepared.plan().transaction;
    let point = submission.prepared.point();
    if let Some(backing) = missing_completion_backing(submission.recording(), &backings) {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::ResourceCompletionBackingAbsent(backing),
            submission,
        }));
    }
    if let Err(reason) = validate_replay_acceptance(
        runtime,
        native,
        resources,
        &submission.prepared,
        transaction,
        &backings,
    ) {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::Core(reason),
            submission,
        }));
    }
    if let Err(reason) = recordings.validate_accept(transaction, point) {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::Recording(reason),
            submission,
        }));
    }
    let Some(worker) = native.recording_worker(transaction) else {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::MissingRecordingWorker(transaction),
            submission,
        }));
    };
    if submission.recording().worker != worker {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::RecordingWorkerMismatch {
                expected: worker,
                actual: submission.recording().worker,
            },
            submission,
        }));
    }
    if let Err(reason) =
        resources.validate_resource_completions(&submission.recording().resource_completions)
    {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::ResourceCompletions(reason),
            submission,
        }));
    }
    let recording_is_ready = recordings.acceptance_is_ready(point);
    let ready_completions = if recording_is_ready {
        submission.recording().resource_completions.clone()
    } else {
        Box::new([])
    };
    if let Err(reason) = watch(point) {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::Watch(reason),
            submission,
        }));
    }
    let (prepared, recording) = submission.into_parts();
    let ready_recording = match recordings
        .accept(transaction, point, recording)
        .unwrap_or_else(|_| unreachable!("recording acceptance was prevalidated"))
    {
        NativeRetirementDisposition::Deferred => None,
        NativeRetirementDisposition::Ready(recording) => Some(recording),
    };
    let replay =
        commit_replay_acceptance(runtime, native, resources, prepared, transaction, backings)
            .unwrap_or_else(|_| unreachable!("replay acceptance was prevalidated"));
    let resource_completions = resources
        .complete_resources(&ready_completions)
        .unwrap_or_else(|_| unreachable!("ready resource completions were prevalidated"));
    Ok(AcceptedReplacementReplay {
        replay,
        resource_completions,
        ready_recording,
    })
}

fn missing_completion_backing(
    recording: &ReplacementNativeRecording,
    backings: &[BackingId],
) -> Option<BackingId> {
    recording
        .resource_completions
        .iter()
        .map(|completion| completion.backing())
        .find(|backing| !backings.contains(backing))
}

/// Join a successful queue-driver receipt to its semantic owners and exact
/// completion watcher. Holding the queue epoch for the operation prevents a
/// caller from draining an observation between watch registration and owner
/// acceptance.
pub fn commit_driver_accepted_replacement<RuntimeCompletion: Clone, NativeSemantic: Clone, T>(
    runtime: &mut TransactionRuntime<RuntimeCompletion>,
    native: &mut DirectReplayNativeOwner<NativeSemantic>,
    resources: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    queues: &mut ReplacementQueueEpoch,
    submission: PreparedReplacementQueueSubmission<NativeSemantic>,
) -> Result<AcceptedReplacementReplay<T>, Box<ReplacementReplayAcceptanceFailure<NativeSemantic>>> {
    let queue = submission.prepared.point().queue;
    let Some(lane) = queues.lane(queue) else {
        return Err(Box::new(ReplacementReplayAcceptanceFailure {
            reason: ReplacementReplayAcceptanceError::QueueAbsent(queue),
            submission,
        }));
    };
    commit_driver_accepted_with_watch(
        runtime,
        native,
        resources,
        recordings,
        submission,
        |point| lane.completion.watch(point),
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementReplayObservationError {
    Timeline(ReplacementTimelineFailure),
    Replay(ReplayTimelineProgressError),
    Recordings(NativeRetirementError),
    ResourceCompletions(ResourceCompletionBatchError),
    HostLandingRead(crate::replacement_representation::ReplacementBufferAllocationError),
    HostLandingStore(reims_vgpu_memory::GuestStoreError),
    HostLandings(HostLandingBatchError),
}

#[derive(Debug)]
pub struct ReplacementObservedTimelineProgress {
    pub queue: reims_vgpu_protocol::QueueOwnerId,
    pub completed: reims_vgpu_protocol::QueueTimelineValue,
}

#[derive(Debug)]
pub struct ReplacementReplayProgress<Semantic, T> {
    /// Exact queue counter whose observation produced this retirement batch.
    pub observed: ReplacementObservedTimelineProgress,
    pub replay: ReplayTimelineProgress<Semantic, T>,
    pub resource_completions: Vec<ResourceCompletionEffect>,
    pub retired_recordings: Vec<ReplacementNativeRecording>,
}

impl<Semantic, T> ReplacementReplayProgress<Semantic, T> {
    /// Remove every indirect-range staging program from the recordings proven
    /// retired by this observation. The recordings remain available for their
    /// fixed-worker cleanup after the opaque range proof is consumed.
    pub fn take_retired_indirect_ranges(
        &mut self,
        transaction: TransactionId,
    ) -> RetiredReplacementIndirectRanges {
        let programs = self
            .retired_recordings
            .iter_mut()
            .flat_map(|recording| {
                recording
                    .take_indirect_range_programs_for(transaction)
                    .into_vec()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        RetiredReplacementIndirectRanges::new(programs)
    }
}

/// Consume only real timeline-counter progress. The replay completion owner,
/// managed backing retirement, and native recording retirement all validate
/// monotonicity before any of them consumes the observation.
pub fn apply_replacement_timeline_observation<Semantic: Clone, T>(
    native: &mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    observation: ReplacementTimelineObservation,
) -> Result<ReplacementReplayProgress<Semantic, T>, ReplacementReplayObservationError> {
    match observation {
        ReplacementTimelineObservation::Progress(progress) => {
            let observed = ReplacementObservedTimelineProgress {
                queue: progress.queue,
                completed: progress.completed,
            };
            native
                .validate_advance(progress.queue, progress.completed)
                .map_err(|error| {
                    ReplacementReplayObservationError::Replay(ReplayTimelineProgressError::Native(
                        error,
                    ))
                })?;
            resources
                .validate_native_retirement_advance(progress.queue, progress.completed)
                .map_err(|error| {
                    ReplacementReplayObservationError::Replay(
                        ReplayTimelineProgressError::Resources(error),
                    )
                })?;
            recordings
                .validate_advance(progress.queue, progress.completed)
                .map_err(ReplacementReplayObservationError::Recordings)?;
            let resource_completions = recordings
                .resource_completions_after(progress.queue, progress.completed)
                .map_err(ReplacementReplayObservationError::Recordings)?;
            let host_landing_programs = recordings
                .host_landing_programs_after(progress.queue, progress.completed)
                .map_err(ReplacementReplayObservationError::Recordings)?;
            let prepared_host_landings = host_landing_programs
                .iter()
                .map(|program| {
                    program
                        .prepare_after_timeline()
                        .map_err(ReplacementReplayObservationError::HostLandingRead)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for landing in &prepared_host_landings {
                landing
                    .validate_store()
                    .map_err(ReplacementReplayObservationError::HostLandingStore)?;
            }
            resources
                .validate_resource_completions(&resource_completions)
                .map_err(ReplacementReplayObservationError::ResourceCompletions)?;
            let host_landings = host_landing_programs
                .iter()
                .map(|program| program.landing())
                .collect::<Vec<_>>();
            resources
                .validate_host_landings_after_resource_completions(
                    &resource_completions,
                    &host_landings,
                )
                .map_err(ReplacementReplayObservationError::HostLandings)?;
            let completions = native
                .advance(progress.queue, progress.completed)
                .unwrap_or_else(|_| unreachable!("replay timeline progress was prevalidated"))
                .into_iter()
                .map(|fact| reims_vgpu_core::CompletionFact {
                    transaction: fact.transaction,
                    session_generation: fact.session_generation,
                    evidence: fact.evidence,
                    semantic: fact.semantic.semantic,
                })
                .collect();
            let retired_recordings = recordings
                .advance(progress.queue, progress.completed)
                .unwrap_or_else(|_| {
                    unreachable!("the complete timeline observation was prevalidated")
                });
            let resource_completions = resources
                .complete_resources(&resource_completions)
                .unwrap_or_else(|_| unreachable!("resource completions were prevalidated"));
            for landing in prepared_host_landings {
                landing
                    .store()
                    .unwrap_or_else(|_| unreachable!("guest landing store was prevalidated"));
            }
            resources
                .complete_host_landings(&host_landings)
                .unwrap_or_else(|_| unreachable!("host landing semantics were prevalidated"));
            // Resource completion must precede native retirement at the same
            // observed point. A physical-incarnation replacement can revoke an
            // execution representation while its last accepted transfer or
            // write is in flight; that completion still names the old
            // representation and consumes its pending authority record.
            let retired_native = resources
                .advance_native_retirement(progress.queue, progress.completed)
                .unwrap_or_else(|_| unreachable!("resource retirement was prevalidated"));
            let replay = ReplayTimelineProgress {
                completions,
                retired_native,
            };
            Ok(ReplacementReplayProgress {
                observed,
                replay,
                resource_completions,
                retired_recordings,
            })
        }
        ReplacementTimelineObservation::Failed(failure) => {
            Err(ReplacementReplayObservationError::Timeline(failure))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_completion::{ReplacementTimelineFailure, ReplacementTimelineProgress};
    use crate::replacement_submit::QueueTimelineSemaphores;
    use ash::vk;
    use reims_vgpu_core::{
        BackingRegion, CompletionStamp, DeviceTransactionPayload, ExecTransaction,
        RepresentationRoute, ResolvedExecSegment, ResolvedExecStream, ResolvedResourceLifecycle,
        ResolvedTransactionPrerequisite, ResourceLifecycleEffect, SessionGeneration,
        StorageBacking, TransactionRecordingPlan, WaitDependencyCause, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ChannelId, ObjectKind, ObjectTableRef, ResourceDescriptor, SegmentBoundary, SegmentKind,
        SessionGenerationId, SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId,
    };

    fn replay_completion(
        resources: impl Into<Box<[ResolvedResourceCompletion]>>,
    ) -> ResolvedReplayCompletion<&'static str> {
        ResolvedReplayCompletion {
            semantic: "done",
            resources: resources.into(),
        }
    }

    type PreparedReplayFixture = (
        TransactionRuntime<&'static str>,
        DirectReplayNativeOwner<ResolvedReplayCompletion<&'static str>>,
        PreparedReplacementQueueSubmission<ResolvedReplayCompletion<&'static str>>,
        TransactionId,
    );

    fn prepared_submission(
        epoch: VulkanDeviceEpochId,
        queue: QueueOwnerId,
    ) -> PreparedReplayFixture {
        let generation = SessionGenerationId::new(1);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(3);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<(), (), (), (), ()>::Exec(ExecTransaction {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(1),
                        task: TaskId::new(1),
                    },
                    prologue: reims_vgpu_core::ExecPrologue::default(),
                    streams: Box::new([ResolvedExecStream {
                        stream_index: 0,
                        segments: Box::new([ResolvedExecSegment {
                            boundary: SegmentBoundary {
                                stream_index: 0,
                                index: 0,
                                kind: SegmentKind::Render,
                                continues_previous: false,
                                continues_next: false,
                            },
                            operations: Box::new([]),
                        }]),
                    }]),
                    accesses: Box::new([]),
                }),
            )
            .unwrap();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(runtime.recording_plan(transaction.id).unwrap())
            .unwrap();
        runtime.recorded(transaction.id).unwrap();
        assert_eq!(
            runtime.take_submission_ready()[0].transaction,
            transaction.id
        );
        let plan = native
            .queue_candidate(
                transaction.id,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(
                plan,
                queue,
                generation,
                replay_completion(Box::<[ResolvedResourceCompletion]>::default()),
            )
            .unwrap();
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let submission = PreparedReplacementQueueSubmission::new(
            prepared,
            &timelines,
            ReplacementNativeRecording::synthetic(
                RecordingWorkerId::new(0),
                Box::<[vk::CommandBuffer]>::default(),
                vk::Fence::null(),
            ),
        )
        .unwrap();
        (runtime, native, submission, transaction.id)
    }

    #[test]
    fn watcher_failure_does_not_manufacture_timeline_progress() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: TransactionId::new(1),
                domain: SubmissionDomainId::new(1),
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
            .prepare(
                plan,
                queue,
                SessionGenerationId::new(1),
                replay_completion(Box::<[ResolvedResourceCompletion]>::default()),
            )
            .unwrap();
        native.accepted(prepared).unwrap();
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let failure = ReplacementTimelineFailure {
            queue,
            waiting_for: QueueTimelineValue::new(1),
            result: vk::Result::ERROR_DEVICE_LOST,
        };
        assert!(matches!(
            apply_replacement_timeline_observation(
                &mut native,
                &mut resources,
                &mut recordings,
                ReplacementTimelineObservation::Failed(failure),
            ),
            Err(ReplacementReplayObservationError::Timeline(found)) if found == failure
        ));
        let progress = apply_replacement_timeline_observation(
            &mut native,
            &mut resources,
            &mut recordings,
            ReplacementTimelineObservation::Progress(ReplacementTimelineProgress {
                queue,
                completed: QueueTimelineValue::new(1),
            }),
        )
        .unwrap();
        assert_eq!(progress.replay.completions.len(), 1);
        assert_eq!(
            progress.replay.completions[0].transaction,
            TransactionId::new(1)
        );
    }

    #[test]
    fn recording_retires_only_at_its_accepted_queue_point() {
        use ash::vk::Handle;

        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let command = vk::CommandBuffer::from_raw(7);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        recordings
            .accept(
                TransactionId::new(3),
                QueueTimelinePoint {
                    epoch,
                    queue,
                    value: QueueTimelineValue::new(4),
                },
                ReplacementNativeRecording::synthetic(
                    RecordingWorkerId::new(0),
                    [command],
                    vk::Fence::null(),
                ),
            )
            .unwrap();
        assert!(recordings
            .advance(queue, QueueTimelineValue::new(3))
            .unwrap()
            .is_empty());
        assert_eq!(
            recordings
                .advance(queue, QueueTimelineValue::new(4))
                .unwrap()[0]
                .command_buffers
                .as_ref(),
            [command]
        );
    }

    #[test]
    fn recording_retirement_completes_its_exact_content_transfer() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: TransactionId::new(1),
                domain: SubmissionDomainId::new(1),
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
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = resources
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        resources
            .plan_gpu_write(
                backing,
                SubmissionId::new(1),
                source,
                [BackingRegion::Whole],
            )
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let snapshot = resources
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        let transfer = resources
            .plan_transfers(backing, source, GUEST_REPRESENTATION, &snapshot)
            .unwrap()[0];
        let completion = ResolvedResourceCompletion::Transfer(transfer);
        let prepared = native
            .prepare(
                plan,
                queue,
                SessionGenerationId::new(1),
                replay_completion(vec![completion].into_boxed_slice()),
            )
            .unwrap();
        let point = prepared.point();
        native.accepted(prepared).unwrap();

        let mut recording = ReplacementNativeRecording::synthetic(
            RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.resource_completions = Box::new([completion]);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        recordings
            .accept(TransactionId::new(1), point, recording)
            .unwrap();

        let progress = apply_replacement_timeline_observation(
            &mut native,
            &mut resources,
            &mut recordings,
            ReplacementTimelineObservation::Progress(ReplacementTimelineProgress {
                queue,
                completed: point.value,
            }),
        )
        .unwrap();
        assert_eq!(
            progress.resource_completions,
            [ResourceCompletionEffect::Transfer(transfer)]
        );
        assert_eq!(progress.retired_recordings.len(), 1);
        assert_eq!(progress.replay.completions.len(), 1);
        assert_eq!(progress.replay.completions[0].semantic, "done");
        assert!(resources
            .graph()
            .storage(backing)
            .unwrap()
            .content
            .representation_matches(GUEST_REPRESENTATION, &snapshot));
    }

    #[test]
    fn content_completion_precedes_same_point_representation_retirement() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let transaction = TransactionId::new(1);
        let submission = SubmissionId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = native
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(
                plan,
                queue,
                SessionGenerationId::new(1),
                replay_completion(Box::<[ResolvedResourceCompletion]>::default()),
            )
            .unwrap();
        let point = prepared.point();
        native.accepted(prepared).unwrap();

        let mut resources = ResourceLifecycleOwner::<&'static str>::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let ResourceLifecycleEffect::ResourceCreated(resource) = resources
            .apply(ResolvedResourceLifecycle::CreateResource {
                task: TaskId::new(1),
                object: ObjectTableRef::new(1),
                kind: ObjectKind::Buffer,
                descriptor: Arc::new(ResourceDescriptor::Buffer(Default::default())),
                storage: Some(backing),
                parents: Box::new([]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                "old",
            )
            .unwrap();
        resources
            .plan_gpu_write(backing, submission, representation, [BackingRegion::Whole])
            .unwrap();
        resources
            .accept_use(backing, transaction, [representation])
            .unwrap();
        resources.submit_use(backing, transaction, point).unwrap();
        resources
            .apply(ResolvedResourceLifecycle::ReleaseResource { resource })
            .unwrap();
        resources
            .apply(ResolvedResourceLifecycle::RetireBacking { backing })
            .unwrap();

        let mut recording = ReplacementNativeRecording::synthetic(
            RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.resource_completions = Box::new([ResolvedResourceCompletion::GpuWrite {
            backing,
            write: submission.into(),
            representation,
        }]);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        recordings.accept(transaction, point, recording).unwrap();

        let progress = apply_replacement_timeline_observation(
            &mut native,
            &mut resources,
            &mut recordings,
            ReplacementTimelineObservation::Progress(ReplacementTimelineProgress {
                queue,
                completed: point.value,
            }),
        )
        .unwrap();
        assert_eq!(progress.replay.retired_native, ["old"]);
        assert!(matches!(
            progress.resource_completions.as_slice(),
            [ResourceCompletionEffect::GpuWrite {
                backing: found,
                submission: found_submission,
                ..
            }] if *found == backing && *found_submission == submission
        ));
    }

    #[test]
    fn recording_retirement_completes_an_info_reply_gpu_write() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let transaction = TransactionId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = native
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(
                plan,
                queue,
                SessionGenerationId::new(1),
                replay_completion(Box::<[ResolvedResourceCompletion]>::default()),
            )
            .unwrap();
        let point = prepared.point();
        native.accepted(prepared).unwrap();

        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let submission = SubmissionId::new(8);
        let regions = resources
            .plan_gpu_write(backing, submission, representation, [BackingRegion::Whole])
            .unwrap();
        let completion = ResolvedResourceCompletion::GpuWrite {
            backing,
            write: submission.into(),
            representation,
        };
        let mut recording = ReplacementNativeRecording::synthetic(
            RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.resource_completions = Box::new([completion]);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        recordings.accept(transaction, point, recording).unwrap();

        let progress = apply_replacement_timeline_observation(
            &mut native,
            &mut resources,
            &mut recordings,
            ReplacementTimelineObservation::Progress(ReplacementTimelineProgress {
                queue,
                completed: point.value,
            }),
        )
        .unwrap();
        assert!(matches!(
            progress.resource_completions.as_slice(),
            [ResourceCompletionEffect::GpuWrite {
                backing: completed_backing,
                submission: completed_submission,
                regions: completed_regions,
            }] if *completed_backing == backing
                && *completed_submission == submission
                && completed_regions.as_ref() == regions.as_ref()
        ));
        let snapshot = resources
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        assert_eq!(snapshot.as_ref(), regions.as_ref());
        assert!(resources
            .graph()
            .storage(backing)
            .unwrap()
            .content
            .representation_matches(representation, &snapshot));
    }

    #[test]
    fn every_recording_completion_requires_its_backing_at_acceptance() {
        let first = BackingId::new(3);
        let second = BackingId::new(4);
        let mut recording = ReplacementNativeRecording::synthetic(
            RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.resource_completions = Box::new([
            ResolvedResourceCompletion::GpuWrite {
                backing: first,
                write: SubmissionId::new(1).into(),
                representation: reims_vgpu_protocol::RepresentationId::new(2),
            },
            ResolvedResourceCompletion::ValidityHostWrite {
                backing: second,
                write: SubmissionId::new(1).into(),
                representation: reims_vgpu_protocol::RepresentationId::new(3),
            },
        ]);

        assert_eq!(missing_completion_backing(&recording, &[]), Some(first));
        assert_eq!(
            missing_completion_backing(&recording, &[first]),
            Some(second)
        );
        assert_eq!(
            missing_completion_backing(&recording, &[second, first]),
            None
        );
    }

    #[test]
    fn missing_completion_backing_returns_every_acceptance_owner() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let (mut runtime, mut native, submission, transaction) = prepared_submission(epoch, queue);
        let (prepared, mut recording) = submission.into_parts();
        let backing = BackingId::new(9);
        recording.resource_completions = Box::new([ResolvedResourceCompletion::GpuWrite {
            backing,
            write: SubmissionId::new(2).into(),
            representation: reims_vgpu_protocol::RepresentationId::new(3),
        }]);
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let mut recordings = ReplacementRecordingOwner::new(epoch);

        let failure = commit_driver_accepted_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            |_| panic!("missing completion backing must refuse before watch registration"),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementReplayAcceptanceError::ResourceCompletionBackingAbsent(backing)
        );
        assert!(failure.submission.recording().backings.is_empty());
        assert_eq!(
            failure.submission.recording().resource_completions.as_ref(),
            [ResolvedResourceCompletion::GpuWrite {
                backing,
                write: SubmissionId::new(2).into(),
                representation: reims_vgpu_protocol::RepresentationId::new(3),
            }]
        );
        assert!(runtime.validate_submitted(transaction).is_ok());
        assert!(native
            .validate_acceptance(&failure.submission.prepared)
            .is_ok());
    }

    #[test]
    fn invalid_completion_batch_refuses_before_watch_or_acceptance_mutation() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let (mut runtime, mut native, submission, transaction) = prepared_submission(epoch, queue);
        let (prepared, mut recording) = submission.into_parts();
        let point = prepared.point();
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        resources
            .accept_use(backing, transaction, [representation])
            .unwrap();
        recording.resource_completions = Box::new([ResolvedResourceCompletion::GpuWrite {
            backing,
            write: SubmissionId::new(99).into(),
            representation,
        }]);
        recording.backings = Box::new([backing]);
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);

        let failure = commit_driver_accepted_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            |_| panic!("invalid completions must refuse before watch registration"),
        )
        .unwrap_err();
        assert!(matches!(
            failure.reason,
            ReplacementReplayAcceptanceError::ResourceCompletions(_)
        ));
        assert!(runtime.validate_submitted(transaction).is_ok());
        assert!(native
            .validate_acceptance(&failure.submission.prepared)
            .is_ok());
        assert!(resources
            .validate_submit_uses(&[backing], transaction, point)
            .is_ok());
        assert!(recordings.validate_accept(transaction, point).is_ok());
    }

    #[test]
    fn invalid_recording_transfer_preserves_all_timeline_owners() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let transaction = TransactionId::new(1);
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = native
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared = native
            .prepare(
                plan,
                queue,
                SessionGenerationId::new(1),
                replay_completion(Box::<[ResolvedResourceCompletion]>::default()),
            )
            .unwrap();
        let point = prepared.point();
        native.accepted(prepared).unwrap();

        let invalid = reims_vgpu_core::TransferKey {
            backing: BackingId::new(99),
            region: BackingRegion::Whole,
            version: reims_vgpu_protocol::ContentVersion::new(1),
            source: reims_vgpu_protocol::RepresentationId::new(2),
            destination: GUEST_REPRESENTATION,
        };
        let mut recording = ReplacementNativeRecording::synthetic(
            RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.resource_completions = Box::new([ResolvedResourceCompletion::Transfer(invalid)]);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        recordings.accept(transaction, point, recording).unwrap();
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);

        assert!(matches!(
            apply_replacement_timeline_observation(
                &mut native,
                &mut resources,
                &mut recordings,
                ReplacementTimelineObservation::Progress(ReplacementTimelineProgress {
                    queue,
                    completed: point.value,
                }),
            ),
            Err(ReplacementReplayObservationError::ResourceCompletions(_))
        ));
        assert_eq!(native.pending_completions(), 1);
        assert_eq!(recordings.advance(queue, point.value).unwrap().len(), 1);
        assert_eq!(native.advance(queue, point.value).unwrap().len(), 1);
    }

    #[test]
    fn watch_refusal_preserves_every_acceptance_owner_for_exact_retry() {
        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let (mut runtime, mut native, submission, transaction) = prepared_submission(epoch, queue);
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let failure = commit_driver_accepted_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            |_| Err(ReplacementTimelineWatchError::OwnerStopped),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementReplayAcceptanceError::Watch(ReplacementTimelineWatchError::OwnerStopped)
        );
        assert!(runtime.validate_submitted(transaction).is_ok());
        assert!(native
            .validate_acceptance(&failure.submission.prepared)
            .is_ok());
        assert!(recordings
            .validate_accept(transaction, failure.submission.prepared.point())
            .is_ok());

        let accepted = commit_driver_accepted_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            failure.submission,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.replay.native.transaction, transaction);
        assert_eq!(accepted.replay.resources, []);
        assert!(accepted.ready_recording.is_none());
        assert_eq!(native.pending_completions(), 1);
        assert!(
            recordings
                .advance(queue, QueueTimelineValue::new(1))
                .unwrap()
                .len()
                == 1
        );
    }

    #[test]
    fn wrong_recording_worker_refuses_before_watch_and_preserves_native_ownership() {
        use ash::vk::Handle;

        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let (mut runtime, mut native, submission, transaction) = prepared_submission(epoch, queue);
        let (prepared, _) = submission.into_parts();
        let command = vk::CommandBuffer::from_raw(37);
        let fence = vk::Fence::from_raw(41);
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let submission = PreparedReplacementQueueSubmission::new(
            prepared,
            &timelines,
            ReplacementNativeRecording::synthetic(RecordingWorkerId::new(1), [command], fence),
        )
        .unwrap();
        let mut resources = ResourceLifecycleOwner::<()>::new(epoch);
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let failure = commit_driver_accepted_with_watch(
            &mut runtime,
            &mut native,
            &mut resources,
            &mut recordings,
            submission,
            |_| panic!("mismatched worker must refuse before watch registration"),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementReplayAcceptanceError::RecordingWorkerMismatch {
                expected: RecordingWorkerId::new(0),
                actual: RecordingWorkerId::new(1),
            }
        );
        assert!(runtime.validate_submitted(transaction).is_ok());
        assert!(native
            .validate_acceptance(&failure.submission.prepared)
            .is_ok());
        assert_eq!(
            failure.submission.recording().command_buffers.as_ref(),
            [command]
        );
        assert_eq!(failure.submission.recording().fence, fence);
    }

    #[test]
    fn disconnected_driver_receipt_returns_submission_and_backing_ownership() {
        use ash::vk::Handle;

        let epoch = VulkanDeviceEpochId::new(2);
        let queue = QueueOwnerId::new(1);
        let (_, _, submission, transaction) = prepared_submission(epoch, queue);
        let command = vk::CommandBuffer::from_raw(29);
        let fence = vk::Fence::from_raw(31);
        let (prepared, _) = submission.into_parts();
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let mut recording =
            ReplacementNativeRecording::synthetic(RecordingWorkerId::new(0), [command], fence);
        recording.backings = Box::new([BackingId::new(7)]);
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let pending = PendingReplacementReplaySubmit {
            pending: PendingPreparedReplacementQueueSubmit::disconnected_for_test(submission),
        };
        let ReplacementReplayDriverPoll::DriverRefused { reason, submission } =
            pending.poll_driver()
        else {
            panic!("a disconnected owner must refuse its retained receipt")
        };
        assert_eq!(reason, ReplacementQueueError::OwnerStopped);
        assert_eq!(submission.prepared.plan().transaction, transaction);
        let (_, recording) = submission.into_parts();
        assert_eq!(recording.command_buffers.as_ref(), [command]);
        assert_eq!(recording.fence, fence);
        assert_eq!(recording.backings.as_ref(), [BackingId::new(7)]);
    }

    #[test]
    fn cleanup_dispatch_refusal_returns_every_native_handle() {
        use ash::vk::Handle;

        let executor = FixedExecutor::new(1, |_| ()).unwrap();
        let command = vk::CommandBuffer::from_raw(17);
        let fence = vk::Fence::from_raw(19);
        let failure = recycle_replacement_recording(
            &executor,
            ReplacementNativeRecording::synthetic(RecordingWorkerId::new(4), [command], fence),
            |_, _| unreachable!(),
        )
        .unwrap_err();
        assert_eq!(failure.reason, FixedExecutorError::UnknownWorker);
        assert_eq!(failure.recording.worker, RecordingWorkerId::new(4));
        assert_eq!(failure.recording.command_buffers.as_ref(), [command]);
        assert_eq!(failure.recording.fence, fence);
    }

    #[test]
    fn retired_recording_returns_to_its_exact_pool_worker() {
        use ash::vk::Handle;
        use std::sync::mpsc;

        let (sender, receiver) = mpsc::channel();
        let executor = FixedExecutor::new(2, |worker| (worker, sender.clone())).unwrap();
        let command = vk::CommandBuffer::from_raw(23);
        recycle_replacement_recording(
            &executor,
            ReplacementNativeRecording::synthetic(
                RecordingWorkerId::new(1),
                [command],
                vk::Fence::null(),
            ),
            |(worker, sender), recording| {
                sender.send((*worker, recording.command_buffers)).unwrap();
            },
        )
        .unwrap();
        let (worker, commands) = receiver.recv().unwrap();
        assert_eq!(worker, RecordingWorkerId::new(1));
        assert_eq!(commands.as_ref(), [command]);
    }
}
