//! Atomic driver acceptance of a whole EXEC's prepared resource envelope.

use crate::{
    replacement_compute::ReplacementComputeImageBindings,
    replacement_epoch::ReplacementQueueEpoch,
    replacement_exec_image::{exec_has_image_uses, validate_exec_image_states},
    replacement_image_state::{
        PreparedImageStateBatch, ReplacementImageStateError, ReplacementImageStateOwner,
    },
    replacement_queue::PreparedReplacementAuxiliaryQueueSubmission,
    replacement_queue::PreparedReplacementQueueSubmission,
    replacement_render::ReplacementRenderImageBindings,
    replacement_replay::{
        commit_driver_accepted_with_watch, AcceptedReplacementReplay, ReplacementRecordingOwner,
        ReplacementReplayAcceptanceError,
    },
};
use reims_vgpu_core::{
    AcceptedExecResourceOutcomes, DirectReplayNativeOwner, PreparedExecResources,
    ResolvedReplayCompletion, ResourceLifecycleOwner, TransactionRuntime,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementAuxiliaryExecAcceptanceError {
    QueueAbsent,
    TransactionMismatch,
    RecordingWorkerMismatch,
    CompletionSetMismatch,
    HostLandingSetMismatch,
    BackingSetMismatch,
    ImageStateBatchMissing,
    UnexpectedImageStateBatch,
    ImageStateOperationMismatch,
    ImageQueueFamilyMismatch,
    ImageAuxiliaryWaitMismatch,
    ImageState(ReplacementImageStateError),
    Resources(reims_vgpu_core::ResourceUseBatchError),
    ResourceCompletions(reims_vgpu_core::ResourceCompletionBatchError),
    Recording(reims_vgpu_core::NativeRetirementError),
    Watch(crate::replacement_completion::ReplacementTimelineWatchError),
}

#[derive(Debug)]
pub struct ReplacementAuxiliaryExecAcceptanceFailure<
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ReplacementAuxiliaryExecAcceptanceError,
    pub submission: PreparedReplacementAuxiliaryQueueSubmission,
    pub resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    pub image_states: Option<PreparedImageStateBatch>,
}

#[derive(Debug)]
pub struct AcceptedReplacementAuxiliaryExec<T, Compute = (), Render = ()> {
    pub point: reims_vgpu_core::QueueTimelinePoint,
    pub resources: Vec<(
        reims_vgpu_protocol::BackingId,
        reims_vgpu_core::ManagedBackingProgress<T>,
    )>,
    pub resource_completions: Vec<reims_vgpu_core::ResourceCompletionEffect>,
    pub ready_recording: Option<crate::replacement_recording::ReplacementNativeRecording>,
    pub outcomes: AcceptedExecResourceOutcomes<Compute, Render>,
}

type AuxiliaryExecAcceptanceResult<T, Compute, NativeCompute, Render, NativeRender> = Result<
    AcceptedReplacementAuxiliaryExec<T, Compute, Render>,
    Box<ReplacementAuxiliaryExecAcceptanceFailure<Compute, NativeCompute, Render, NativeRender>>,
>;

/// Accept an auxiliary, buffer-only EXEC phase without advancing the parent
/// transaction's submitted/completed state. Its resource leases and recording
/// retire from the auxiliary point just like any other native use.
pub fn commit_driver_accepted_auxiliary_exec<T, Compute, NativeCompute, Render, NativeRender>(
    resources_owner: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    queues: &mut ReplacementQueueEpoch,
    images: &mut ReplacementImageStateOwner,
    submission: PreparedReplacementAuxiliaryQueueSubmission,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
) -> AuxiliaryExecAcceptanceResult<T, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let point = submission.prepared.point();
    let Some(lane) = queues.lane(point.queue) else {
        return Err(Box::new(ReplacementAuxiliaryExecAcceptanceFailure {
            reason: ReplacementAuxiliaryExecAcceptanceError::QueueAbsent,
            submission,
            resources,
            image_states,
        }));
    };
    commit_driver_accepted_auxiliary_exec_with_watch(
        resources_owner,
        recordings,
        images,
        submission,
        resources,
        image_states,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_auxiliary_exec_with_watch<
    T,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    resources_owner: &mut ResourceLifecycleOwner<T>,
    recordings: &mut ReplacementRecordingOwner,
    images: &mut ReplacementImageStateOwner,
    submission: PreparedReplacementAuxiliaryQueueSubmission,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> AuxiliaryExecAcceptanceResult<T, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let transaction = submission.prepared.transaction();
    let point = submission.prepared.point();
    let has_images = exec_has_image_uses(&resources);
    let reason = if transaction != resources.transaction() {
        Some(ReplacementAuxiliaryExecAcceptanceError::TransactionMismatch)
    } else if has_images.as_ref().is_ok_and(|has_images| *has_images) && image_states.is_none() {
        Some(ReplacementAuxiliaryExecAcceptanceError::ImageStateBatchMissing)
    } else if has_images.as_ref().is_ok_and(|has_images| !*has_images)
        && image_states
            .as_ref()
            .is_some_and(|states| validate_exec_image_states(&resources, states).is_err())
    {
        Some(ReplacementAuxiliaryExecAcceptanceError::UnexpectedImageStateBatch)
    } else if has_images.is_err()
        || image_states
            .as_ref()
            .is_some_and(|states| validate_exec_image_states(&resources, states).is_err())
    {
        Some(ReplacementAuxiliaryExecAcceptanceError::ImageStateOperationMismatch)
    } else if image_states
        .as_ref()
        .is_some_and(|states| states.queue_family() != submission.recording().queue_family)
    {
        Some(ReplacementAuxiliaryExecAcceptanceError::ImageQueueFamilyMismatch)
    } else if image_states
        .as_ref()
        .is_some_and(|states| !same_points(submission.auxiliary_waits(), &states.release_points()))
    {
        Some(ReplacementAuxiliaryExecAcceptanceError::ImageAuxiliaryWaitMismatch)
    } else if let Some(reason) = image_states
        .as_ref()
        .and_then(|states| images.validate_batch_acceptance(states, point).err())
    {
        Some(ReplacementAuxiliaryExecAcceptanceError::ImageState(reason))
    } else if submission.recording().worker != submission.prepared.recording_worker() {
        Some(ReplacementAuxiliaryExecAcceptanceError::RecordingWorkerMismatch)
    } else if submission.recording().resource_completions.as_ref()
        != resources.resource_completions()
    {
        Some(ReplacementAuxiliaryExecAcceptanceError::CompletionSetMismatch)
    } else if submission.recording().host_landings().as_ref() != resources.host_landings() {
        Some(ReplacementAuxiliaryExecAcceptanceError::HostLandingSetMismatch)
    } else if submission.recording().backings() != resources.backings() {
        Some(ReplacementAuxiliaryExecAcceptanceError::BackingSetMismatch)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementAuxiliaryExecAcceptanceFailure {
            reason,
            submission,
            resources,
            image_states,
        }));
    }
    if let Err(reason) =
        resources_owner.validate_submit_uses(resources.backings(), transaction, point)
    {
        return Err(Box::new(ReplacementAuxiliaryExecAcceptanceFailure {
            reason: ReplacementAuxiliaryExecAcceptanceError::Resources(reason),
            submission,
            resources,
            image_states,
        }));
    }
    if let Err(reason) =
        resources_owner.validate_resource_completions(&submission.recording().resource_completions)
    {
        return Err(Box::new(ReplacementAuxiliaryExecAcceptanceFailure {
            reason: ReplacementAuxiliaryExecAcceptanceError::ResourceCompletions(reason),
            submission,
            resources,
            image_states,
        }));
    }
    if let Err(reason) = recordings.validate_auxiliary_accept(transaction, point) {
        return Err(Box::new(ReplacementAuxiliaryExecAcceptanceFailure {
            reason: ReplacementAuxiliaryExecAcceptanceError::Recording(reason),
            submission,
            resources,
            image_states,
        }));
    }
    let recording_is_ready = recordings.acceptance_is_ready(point);
    let ready_completions = if recording_is_ready {
        submission.recording().resource_completions.clone()
    } else {
        Box::new([])
    };
    if let Err(reason) = watch(point) {
        return Err(Box::new(ReplacementAuxiliaryExecAcceptanceFailure {
            reason: ReplacementAuxiliaryExecAcceptanceError::Watch(reason),
            submission,
            resources,
            image_states,
        }));
    }
    let (_, recording) = submission.into_parts();
    let ready_recording = match recordings
        .accept_auxiliary(transaction, point, recording)
        .unwrap_or_else(|_| unreachable!("auxiliary recording acceptance was prevalidated"))
    {
        reims_vgpu_core::NativeRetirementDisposition::Deferred => None,
        reims_vgpu_core::NativeRetirementDisposition::Ready(recording) => Some(recording),
    };
    let resource_progress = resources_owner
        .submit_uses(resources.backings(), transaction, point)
        .unwrap_or_else(|_| unreachable!("auxiliary resource acceptance was prevalidated"));
    let resource_completions = resources_owner
        .complete_resources(&ready_completions)
        .unwrap_or_else(|_| unreachable!("ready resource completions were prevalidated"));
    if let Some(image_states) = image_states {
        images
            .accepted_batch(image_states, point)
            .expect("auxiliary image acceptance was prevalidated");
    }
    Ok(AcceptedReplacementAuxiliaryExec {
        point,
        resources: resource_progress,
        resource_completions,
        ready_recording,
        outcomes: resources.into_outcomes(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementExecAcceptanceError {
    TransactionMismatch,
    CompletionSetMismatch,
    RecordingCompletionSetMismatch,
    HostLandingSetMismatch,
    BackingSetMismatch,
    ImageStateBatchMissing,
    UnexpectedImageStateBatch,
    ImageStateOperationMismatch,
    ImageQueueFamilyMismatch,
    ImageAuxiliaryWaitMismatch,
    ImageState(ReplacementImageStateError),
    Replay(ReplacementReplayAcceptanceError),
}

#[derive(Debug)]
pub struct ReplacementExecAcceptanceFailure<
    Semantic,
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ReplacementExecAcceptanceError,
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    pub image_states: Option<PreparedImageStateBatch>,
}

#[derive(Debug)]
pub struct AcceptedReplacementExec<T, Compute = (), Render = ()> {
    pub replay: AcceptedReplacementReplay<T>,
    pub outcomes: AcceptedExecResourceOutcomes<Compute, Render>,
}

type ExecAcceptanceResult<Semantic, T, Compute, NativeCompute, Render, NativeRender> = Result<
    AcceptedReplacementExec<T, Compute, Render>,
    Box<ReplacementExecAcceptanceFailure<Semantic, Compute, NativeCompute, Render, NativeRender>>,
>;

pub struct ReplacementExecAcceptanceOwners<'a, Semantic, T> {
    pub runtime: &'a mut TransactionRuntime<Semantic>,
    pub native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    pub resources: &'a mut ResourceLifecycleOwner<T>,
    pub recordings: &'a mut ReplacementRecordingOwner,
    pub queues: &'a mut ReplacementQueueEpoch,
    pub images: &'a mut ReplacementImageStateOwner,
}

struct ReplacementExecMutationOwners<'a, Semantic, T> {
    runtime: &'a mut TransactionRuntime<Semantic>,
    native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &'a mut ResourceLifecycleOwner<T>,
    recordings: &'a mut ReplacementRecordingOwner,
    images: &'a mut ReplacementImageStateOwner,
}

pub fn commit_driver_accepted_exec<
    Semantic: Clone,
    T,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    owners: ReplacementExecAcceptanceOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
) -> ExecAcceptanceResult<Semantic, T, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    commit_driver_accepted_exec_with_additional_waits(
        owners,
        submission,
        resources,
        image_states,
        Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
    )
}

pub fn commit_driver_accepted_exec_with_additional_waits<
    Semantic: Clone,
    T,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    owners: ReplacementExecAcceptanceOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
    additional_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
) -> ExecAcceptanceResult<Semantic, T, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let queue = submission.prepared.point().queue;
    let Some(lane) = owners.queues.lane(queue) else {
        return Err(Box::new(ReplacementExecAcceptanceFailure {
            reason: ReplacementExecAcceptanceError::Replay(
                ReplacementReplayAcceptanceError::QueueAbsent(queue),
            ),
            submission,
            resources,
            image_states,
        }));
    };
    commit_driver_accepted_exec_with_watch(
        ReplacementExecMutationOwners {
            runtime: owners.runtime,
            native: owners.native,
            resources: owners.resources,
            recordings: owners.recordings,
            images: owners.images,
        },
        submission,
        resources,
        image_states,
        additional_waits,
        |point| lane.completion.watch(point),
    )
}

fn commit_driver_accepted_exec_with_watch<
    Semantic: Clone,
    T,
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    owners: ReplacementExecMutationOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
    additional_waits: Box<[reims_vgpu_core::QueueTimelinePoint]>,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> ExecAcceptanceResult<Semantic, T, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let transaction = submission.prepared.plan().transaction;
    let has_images = exec_has_image_uses(&resources);
    let reason = if transaction != resources.transaction() {
        Some(ReplacementExecAcceptanceError::TransactionMismatch)
    } else if has_images.as_ref().is_ok_and(|has_images| *has_images) && image_states.is_none() {
        Some(ReplacementExecAcceptanceError::ImageStateBatchMissing)
    } else if has_images.as_ref().is_ok_and(|has_images| !*has_images)
        && image_states
            .as_ref()
            .is_some_and(|states| validate_exec_image_states(&resources, states).is_err())
    {
        Some(ReplacementExecAcceptanceError::UnexpectedImageStateBatch)
    } else if has_images.is_err()
        || image_states
            .as_ref()
            .is_some_and(|states| validate_exec_image_states(&resources, states).is_err())
    {
        Some(ReplacementExecAcceptanceError::ImageStateOperationMismatch)
    } else if image_states
        .as_ref()
        .is_some_and(|states| states.queue_family() != submission.recording().queue_family)
    {
        Some(ReplacementExecAcceptanceError::ImageQueueFamilyMismatch)
    } else if !same_points(
        submission.auxiliary_waits(),
        &combined_auxiliary_waits(image_states.as_ref(), &additional_waits),
    ) {
        Some(ReplacementExecAcceptanceError::ImageAuxiliaryWaitMismatch)
    } else if let Some(reason) = image_states.as_ref().and_then(|states| {
        owners
            .images
            .validate_batch_acceptance(states, submission.prepared.point())
            .err()
    }) {
        Some(ReplacementExecAcceptanceError::ImageState(reason))
    } else if submission.prepared.semantic().resources.as_ref() != resources.resource_completions()
    {
        Some(ReplacementExecAcceptanceError::CompletionSetMismatch)
    } else if submission.recording().resource_completions.as_ref()
        != resources.resource_completions()
    {
        Some(ReplacementExecAcceptanceError::RecordingCompletionSetMismatch)
    } else if submission.recording().host_landings().as_ref() != resources.host_landings() {
        Some(ReplacementExecAcceptanceError::HostLandingSetMismatch)
    } else if submission.recording().backings() != resources.backings() {
        Some(ReplacementExecAcceptanceError::BackingSetMismatch)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementExecAcceptanceFailure {
            reason,
            submission,
            resources,
            image_states,
        }));
    }
    let replay = match commit_driver_accepted_with_watch(
        owners.runtime,
        owners.native,
        owners.resources,
        owners.recordings,
        submission,
        watch,
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(ReplacementExecAcceptanceFailure {
                reason: ReplacementExecAcceptanceError::Replay(failure.reason.clone()),
                submission: failure.submission,
                resources,
                image_states,
            }));
        }
    };
    if let Some(image_states) = image_states {
        owners
            .images
            .accepted_batch(image_states, replay.replay.native.point)
            .expect("image batch acceptance was prevalidated before replay mutation");
    }
    Ok(AcceptedReplacementExec {
        replay,
        outcomes: resources.into_outcomes(),
    })
}

fn same_points(
    left: &[reims_vgpu_core::QueueTimelinePoint],
    right: &[reims_vgpu_core::QueueTimelinePoint],
) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn combined_auxiliary_waits(
    image_states: Option<&PreparedImageStateBatch>,
    additional_waits: &[reims_vgpu_core::QueueTimelinePoint],
) -> Box<[reims_vgpu_core::QueueTimelinePoint]> {
    let mut waits = image_states
        .map(PreparedImageStateBatch::release_points)
        .unwrap_or_default()
        .into_vec();
    waits.extend_from_slice(additional_waits);
    waits.sort_unstable();
    waits.dedup();
    waits.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        replacement_image_blit::{prepare_exec_image_blit_states, ReplacementImageFinalLayout},
        replacement_image_state::{
            ReplacementImageKey, ReplacementImageSharing, ReplacementImageState,
        },
        replacement_recording::ReplacementNativeRecording,
        replacement_submit::QueueTimelineSemaphores,
    };
    use ash::vk;
    use reims_vgpu_core::BackingView;
    use reims_vgpu_core::{
        assemble_prepared_exec_resources, prepare_buffer_blit_with_write,
        prepare_image_blit_with_write, BackingRegion, BufferFillPattern, CompletionStamp,
        DeviceTransactionPayload, ExecTransaction, GpuWriteId, LinearRange,
        PreparedExecResourceInputs, RepresentationRoute, ResolvedBlit, ResolvedBufferRange,
        ResolvedBufferToTextureBlit, ResolvedExecSegment, ResolvedExecStream,
        ResolvedInfoOperation, ResolvedLinearTextureLevel, ResolvedOperation,
        ResolvedResourceLifecycle, ResolvedTextureBacking, ResolvedTextureEndpoint,
        ResourceLifecycleEffect, SessionGeneration, StorageBacking, TextureExtent, TextureOrigin,
        TransactionRecordingPlan, WaitDependencyCause, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ByteLength, ChannelId, GuestVirtualAddress, QueueOwnerId, QueueTimelineValue, ResourceId,
        SegmentBoundary, SegmentKind, SessionGenerationId, SubmissionDomainId, SubmissionId,
        SubmissionIdentity, TaskId, TransactionId, VulkanDeviceEpochId,
    };

    fn fill(backing: reims_vgpu_protocol::BackingId, index: u32) -> ResolvedBlit {
        ResolvedBlit::Fill {
            destination: ResolvedBufferRange {
                resource: ResourceId::new(index + 1, 1),
                storage: backing,
                region: LinearRange::new(u64::from(index) * 16, 16).unwrap(),
                address: GuestVirtualAddress::new(0x1000 + u64::from(index) * 16),
                length: ByteLength::new(16),
            },
            pattern: BufferFillPattern::Byte(index as u8),
        }
    }

    #[test]
    fn one_driver_receipt_accepts_two_ordered_buffer_writes_together() {
        let epoch = VulkanDeviceEpochId::new(1);
        let generation = SessionGenerationId::new(1);
        let queue = QueueOwnerId::new(2);
        let submission = SubmissionId::new(1);
        let mut runtime: TransactionRuntime<&'static str> =
            TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let mut lifecycle = ResourceLifecycleOwner::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = lifecycle
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 32).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        lifecycle
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let first = fill(backing, 0);
        let second = fill(backing, 1);
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: submission,
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([
                        ResolvedOperation::<(), (), ResolvedInfoOperation, (), ()>::Blit(Box::new(
                            first.clone(),
                        )),
                        ResolvedOperation::Blit(Box::new(second.clone())),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let admitted = runtime
            .admit_resolved(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<_, (), (), (), ()>::Exec(exec.clone()),
            )
            .unwrap();
        let resources = assemble_prepared_exec_resources(
            admitted.id,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([
                    prepare_buffer_blit_with_write(
                        &mut lifecycle,
                        admitted.id,
                        GpuWriteId::operation(admitted.id, submission, 0),
                        first,
                    )
                    .unwrap(),
                    prepare_buffer_blit_with_write(
                        &mut lifecycle,
                        admitted.id,
                        GpuWriteId::operation(admitted.id, submission, 1),
                        second,
                    )
                    .unwrap(),
                ]),
                image_blits: Box::new([]),
                compute_dispatches:
                    Box::<[reims_vgpu_core::PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches:
                    Box::<[reims_vgpu_core::PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        let completions: Box<_> = resources.resource_completions().into();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: admitted.id,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        runtime.recorded(admitted.id).unwrap();
        runtime.take_submission_ready();
        let plan = native
            .queue_candidate(
                admitted.id,
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
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: completions,
                },
            )
            .unwrap();
        let predecessor = reims_vgpu_core::QueueTimelinePoint {
            epoch,
            queue: QueueOwnerId::new(3),
            value: QueueTimelineValue::new(1),
        };
        let timelines = QueueTimelineSemaphores::new(
            epoch,
            [
                (queue, vk::Semaphore::null()),
                (predecessor.queue, vk::Semaphore::null()),
            ],
        );
        let mut recording = ReplacementNativeRecording::synthetic(
            reims_vgpu_core::RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.backings = resources.backings().into();
        let submission = PreparedReplacementQueueSubmission::new_with_auxiliary_waits(
            prepared,
            &timelines,
            recording,
            [predecessor],
        )
        .unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let mut images = ReplacementImageStateOwner::new(epoch);
        let failure = commit_driver_accepted_exec_with_watch(
            ReplacementExecMutationOwners {
                runtime: &mut runtime,
                native: &mut native,
                resources: &mut lifecycle,
                recordings: &mut recordings,
                images: &mut images,
            },
            submission,
            resources,
            None,
            Box::new([predecessor]),
            |_| panic!("an incomplete recording must refuse before watch registration"),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementExecAcceptanceError::RecordingCompletionSetMismatch
        );
        let (prepared, mut recording) = failure.submission.into_parts();
        recording.resource_completions = failure.resources.resource_completions().into();
        let submission = PreparedReplacementQueueSubmission::new_with_auxiliary_waits(
            prepared,
            &timelines,
            recording,
            [predecessor],
        )
        .unwrap();
        let accepted = commit_driver_accepted_exec_with_watch(
            ReplacementExecMutationOwners {
                runtime: &mut runtime,
                native: &mut native,
                resources: &mut lifecycle,
                recordings: &mut recordings,
                images: &mut images,
            },
            submission,
            failure.resources,
            None,
            Box::new([predecessor]),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.outcomes.buffer_blits.len(), 2);
        assert_eq!(accepted.replay.replay.resources.len(), 1);
    }

    #[test]
    fn auxiliary_range_phase_attaches_resources_without_submitting_the_parent_transaction() {
        let epoch = VulkanDeviceEpochId::new(21);
        let generation = SessionGenerationId::new(22);
        let queue = QueueOwnerId::new(1);
        let channel = ChannelId::new(3);
        let mut runtime: TransactionRuntime<&'static str> =
            TransactionRuntime::new(SessionGeneration::new(generation));
        runtime.define_channel(channel).unwrap();
        let operation = reims_vgpu_core::ResolvedIndirectCommand::ExecuteIndirectRange {
            icb: reims_vgpu_protocol::ResourceId::new(8, 1),
            arguments_resource: reims_vgpu_protocol::ResourceId::new(9, 1),
            arguments_backing: reims_vgpu_protocol::BackingId::new(1),
            arguments_range: LinearRange::new(0, 8).unwrap(),
            kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
        };
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(23),
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
                    operations: Box::new([ResolvedOperation::<
                        (),
                        (),
                        ResolvedInfoOperation,
                        reims_vgpu_core::ResolvedIndirectCommand,
                        (),
                    >::IndirectCommand(operation)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let admitted = runtime
            .admit_resolved(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<_, (), (), (), ()>::Exec(exec.clone()),
            )
            .unwrap();
        let transaction = admitted.id;
        let mut lifecycle = ResourceLifecycleOwner::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = lifecycle
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 8).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(backing, reims_vgpu_protocol::BackingId::new(1));
        lifecycle
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let readback = reims_vgpu_core::prepare_indirect_range_readback(
            &mut lifecycle,
            transaction,
            0,
            operation,
        )
        .unwrap();
        let resources = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([]),
                compute_dispatches:
                    Box::<[reims_vgpu_core::PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches:
                    Box::<[reims_vgpu_core::PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([readback]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        runtime.recorded(transaction).unwrap();
        runtime.take_submission_ready();
        let plan = native
            .queue_candidate(
                transaction,
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let parent = native
            .prepare(
                plan,
                queue,
                generation,
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: Box::new([]),
                },
            )
            .unwrap();
        let auxiliary = native.prepare_auxiliary(&parent, queue).unwrap();
        let point = auxiliary.point();
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let mut recording = ReplacementNativeRecording::synthetic(
            reims_vgpu_core::RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.backings = resources.backings().into();
        let submission =
            PreparedReplacementAuxiliaryQueueSubmission::new(auxiliary, &timelines, recording)
                .unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let mut images = ReplacementImageStateOwner::new(epoch);
        let accepted = commit_driver_accepted_auxiliary_exec_with_watch(
            &mut lifecycle,
            &mut recordings,
            &mut images,
            submission,
            resources,
            None,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.point, point);
        assert_eq!(accepted.outcomes.indirect_range_readbacks.len(), 1);
        assert_eq!(accepted.resources.len(), 1);
        assert!(runtime.validate_submitted(transaction).is_ok());
        assert!(native.validate_acceptance(&parent).is_ok());
    }

    struct ShaderReadFinal;

    impl ReplacementImageFinalLayout for ShaderReadFinal {
        fn final_layout(
            &self,
            _image: ReplacementImageKey,
            _required_usage: vk::ImageUsageFlags,
        ) -> Option<vk::ImageLayout> {
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        }
    }

    #[test]
    fn whole_exec_image_acceptance_commits_content_and_layout_together() {
        let epoch = VulkanDeviceEpochId::new(1);
        let generation = SessionGenerationId::new(1);
        let queue = QueueOwnerId::new(2);
        let submission_id = SubmissionId::new(4);
        let mut lifecycle = ResourceLifecycleOwner::new(epoch);
        let mut create_backing = |regions| match lifecycle
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions,
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let source = create_backing(Box::new([BackingRegion::Linear(
            LinearRange::new(0, 64).unwrap(),
        )]));
        let destination = create_backing(Box::new([BackingRegion::Whole]));
        let source_representation = lifecycle
            .create_execution_representation(
                source,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        // The image is keyed by the texture declared over the backing; this
        // fixture names only one.
        let destination_texture = ResourceId::<reims_vgpu_protocol::ResourceObject>::new(2, 1);
        let destination_representation = lifecycle
            .create_execution_representation(
                destination,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Image(reims_vgpu_core::ImageOwner::owning(destination_texture)),
                (),
            )
            .unwrap();
        let source_region = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let snapshot = lifecycle
            .snapshot_content(source, &[source_region])
            .unwrap();
        for transfer in lifecycle
            .plan_transfers(
                source,
                GUEST_REPRESENTATION,
                source_representation,
                &snapshot,
            )
            .unwrap()
            .iter()
            .copied()
        {
            lifecycle.complete_transfer(transfer).unwrap();
        }
        let operation = ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: ResolvedBufferRange {
                resource: ResourceId::new(1, 1),
                storage: source,
                region: LinearRange::new(0, 64).unwrap(),
                address: GuestVirtualAddress::new(0x1000),
                length: ByteLength::new(64),
            },
            source_bytes_per_row: 16,
            source_bytes_per_image: 64,
            destination: ResolvedTextureEndpoint {
                resource: ResourceId::new(2, 1),
                image_owner: ResourceId::new(2, 1),
                storage: destination,
                level: 0,
                slice: 0,
                backing: ResolvedTextureBacking::Linear(ResolvedLinearTextureLevel {
                    base_gva: 0x2000,
                    alloc_size: 64,
                    level_offset: 0,
                    row_stride: 16,
                    slice_stride: 0,
                    slice_index: 0,
                    width: 4,
                    height: 4,
                    depth: 1,
                    bpp: 4,
                    pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                }),
            },
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 4,
                depth: 1,
            },
            aspect: reims_vgpu_core::pixel_format::BlitAspect::Full,
        });
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: submission_id,
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::<
                        (),
                        (),
                        ResolvedInfoOperation,
                        (),
                        (),
                    >::Blit(Box::new(
                        operation.clone(),
                    ))]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_resolved(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<_, (), (), (), ()>::Exec(exec.clone()),
            )
            .unwrap();
        let resources = assemble_prepared_exec_resources(
            admitted.id,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([prepare_image_blit_with_write(
                    &mut lifecycle,
                    admitted.id,
                    GpuWriteId::operation(admitted.id, submission_id, 0),
                    operation,
                )
                .unwrap()]),
                compute_dispatches:
                    Box::<[reims_vgpu_core::PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches:
                    Box::<[reims_vgpu_core::PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        let image = ReplacementImageKey {
            backing: destination,
            representation: destination_representation,
        };
        let mut images = ReplacementImageStateOwner::new(epoch);
        images
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let image_states =
            prepare_exec_image_blit_states(&mut images, &resources, 0, &ShaderReadFinal).unwrap();
        let completions: Box<_> = resources.resource_completions().into();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: admitted.id,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        runtime.recorded(admitted.id).unwrap();
        runtime.take_submission_ready();
        let plan = native
            .queue_candidate(
                admitted.id,
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
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: completions,
                },
            )
            .unwrap();
        let point = prepared.point();
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::null())]);
        let mut recording = ReplacementNativeRecording::synthetic(
            reims_vgpu_core::RecordingWorkerId::new(0),
            Box::<[vk::CommandBuffer]>::default(),
            vk::Fence::null(),
        );
        recording.queue_family = 0;
        recording.backings = resources.backings().into();
        recording.resource_completions = resources.resource_completions().into();
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let accepted = commit_driver_accepted_exec_with_watch(
            ReplacementExecMutationOwners {
                runtime: &mut runtime,
                native: &mut native,
                resources: &mut lifecycle,
                recordings: &mut recordings,
                images: &mut images,
            },
            submission,
            resources,
            Some(image_states),
            Box::new([]),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.outcomes.image_blits.len(), 1);
        assert_eq!(
            images.state(image),
            Some(ReplacementImageState {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sharing: ReplacementImageSharing::Concurrent,
                last_use: Some(point),
            })
        );
    }
}
