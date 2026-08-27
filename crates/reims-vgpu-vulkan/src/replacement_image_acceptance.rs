//! Atomic destination acceptance for prepared image blits.

use crate::{
    replacement_epoch::ReplacementQueueEpoch,
    replacement_image_state::{
        PreparedImageState, ReplacementImageStateError, ReplacementImageStateOwner,
    },
    replacement_queue::PreparedReplacementQueueSubmission,
    replacement_recording::{
        ReplacementRecordingOperation, ReplacementRecordingRequest, ReplacementRecordingWorker,
    },
    replacement_recording_queue::{
        dispatch_prepared_replacement_recording_with_auxiliary_waits,
        PendingPreparedReplacementRecording, PreparedReplacementRecordingError,
        PreparedReplacementRecordingFailure, PreparedReplacementRecordingRecovery,
    },
    replacement_replay::{
        commit_driver_accepted_with_watch, AcceptedReplacementReplay, ReplacementRecordingOwner,
        ReplacementReplayAcceptanceError,
    },
};
use reims_vgpu_core::{
    DirectReplayNativeOwner, FixedExecutor, PreparedImageBlit, PreparedNativeSubmission,
    ResolvedBlit, ResolvedReplayCompletion, ResolvedResourceCompletion, ResourceLifecycleOwner,
    TransactionRuntime,
};

#[derive(Debug)]
pub struct PreparedReplacementImageBlitRecordingFailure<Semantic, Operation> {
    pub recording:
        PreparedReplacementRecordingFailure<ResolvedReplayCompletion<Semantic>, Operation>,
    pub blit: PreparedImageBlit,
    pub image_state: PreparedImageState,
}

#[must_use = "image semantic, state, and recording ownership must be observed together"]
pub struct PendingPreparedReplacementImageBlitRecording<Semantic, Operation> {
    pending: PendingPreparedReplacementRecording<ResolvedReplayCompletion<Semantic>, Operation>,
    blit: PreparedImageBlit,
    image_state: PreparedImageState,
}

#[derive(Debug)]
pub struct PreparedReplacementImageBlitQueue<Semantic> {
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub blit: PreparedImageBlit,
    pub image_state: PreparedImageState,
}

pub fn dispatch_prepared_image_blit_recording<Semantic, Operation>(
    executor: &FixedExecutor<ReplacementRecordingWorker>,
    timelines: &crate::replacement_submit::QueueTimelineSemaphores,
    prepared: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    request: ReplacementRecordingRequest<Operation>,
    blit: PreparedImageBlit,
    image_state: PreparedImageState,
) -> Result<
    PendingPreparedReplacementImageBlitRecording<Semantic, Operation>,
    Box<PreparedReplacementImageBlitRecordingFailure<Semantic, Operation>>,
>
where
    Operation: Clone + Send + ReplacementRecordingOperation + 'static,
{
    let transaction = prepared.plan().transaction;
    if transaction != blit.transaction() || transaction != image_state.transaction() {
        return Err(Box::new(PreparedReplacementImageBlitRecordingFailure {
            recording: PreparedReplacementRecordingFailure {
                reason: PreparedReplacementRecordingError::TransactionMismatch {
                    prepared: transaction,
                    request: blit.transaction(),
                },
                prepared,
                recovery: PreparedReplacementRecordingRecovery::Request(request),
            },
            blit,
            image_state,
        }));
    }
    if request.queue_family != image_state.queue_family() {
        return Err(Box::new(PreparedReplacementImageBlitRecordingFailure {
            recording: PreparedReplacementRecordingFailure {
                reason: PreparedReplacementRecordingError::QueueFamilyMismatch {
                    prepared: image_state.queue_family(),
                    request: request.queue_family,
                },
                prepared,
                recovery: PreparedReplacementRecordingRecovery::Request(request),
            },
            blit,
            image_state,
        }));
    }
    let waits = image_state.release_points();
    match dispatch_prepared_replacement_recording_with_auxiliary_waits(
        executor, timelines, prepared, request, waits,
    ) {
        Ok(pending) => Ok(PendingPreparedReplacementImageBlitRecording {
            pending,
            blit,
            image_state,
        }),
        Err(failure) => Err(Box::new(PreparedReplacementImageBlitRecordingFailure {
            recording: *failure,
            blit,
            image_state,
        })),
    }
}

impl<Semantic, Operation> PendingPreparedReplacementImageBlitRecording<Semantic, Operation> {
    pub fn wait(
        self,
    ) -> Result<
        PreparedReplacementImageBlitQueue<Semantic>,
        Box<PreparedReplacementImageBlitRecordingFailure<Semantic, Operation>>,
    > {
        match self.pending.wait() {
            Ok(submission) => Ok(PreparedReplacementImageBlitQueue {
                submission,
                blit: self.blit,
                image_state: self.image_state,
            }),
            Err(failure) => Err(Box::new(PreparedReplacementImageBlitRecordingFailure {
                recording: *failure,
                blit: self.blit,
                image_state: self.image_state,
            })),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementImageBlitAcceptanceError {
    TransactionMismatch,
    CompletionSetMismatch,
    BackingSetMismatch,
    QueueFamilyMismatch,
    AuxiliaryWaitSetMismatch,
    ImageState(ReplacementImageStateError),
    Replay(ReplacementReplayAcceptanceError),
}

#[derive(Debug)]
pub struct ReplacementImageBlitAcceptanceFailure<Semantic> {
    pub reason: ReplacementImageBlitAcceptanceError,
    pub submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    pub blit: PreparedImageBlit,
    pub image_state: PreparedImageState,
}

#[derive(Debug)]
pub struct AcceptedReplacementImageBlit<T> {
    pub replay: AcceptedReplacementReplay<T>,
    pub operation: ResolvedBlit,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

pub struct ReplacementImageBlitAcceptanceOwners<'a, Semantic, T> {
    pub runtime: &'a mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    pub native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    pub resources: &'a mut ResourceLifecycleOwner<T>,
    pub recordings: &'a mut ReplacementRecordingOwner,
    pub queues: &'a mut ReplacementQueueEpoch,
    pub images: &'a mut ReplacementImageStateOwner,
}

pub fn commit_driver_accepted_image_blit<Semantic: Clone, T>(
    owners: ReplacementImageBlitAcceptanceOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    blit: PreparedImageBlit,
    image_state: PreparedImageState,
) -> Result<AcceptedReplacementImageBlit<T>, Box<ReplacementImageBlitAcceptanceFailure<Semantic>>> {
    let ReplacementImageBlitAcceptanceOwners {
        runtime,
        native,
        resources,
        recordings,
        queues,
        images,
    } = owners;
    let queue = submission.prepared.point().queue;
    let Some(lane) = queues.lane(queue) else {
        return Err(Box::new(ReplacementImageBlitAcceptanceFailure {
            reason: ReplacementImageBlitAcceptanceError::Replay(
                ReplacementReplayAcceptanceError::QueueAbsent(queue),
            ),
            submission,
            blit,
            image_state,
        }));
    };
    commit_driver_accepted_image_blit_with_watch(
        ReplacementImageBlitAcceptanceCoreOwners {
            runtime,
            native,
            resources,
            recordings,
            images,
        },
        submission,
        blit,
        image_state,
        |point| lane.completion.watch(point),
    )
}

struct ReplacementImageBlitAcceptanceCoreOwners<'a, Semantic, T> {
    runtime: &'a mut TransactionRuntime<ResolvedReplayCompletion<Semantic>>,
    native: &'a mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &'a mut ResourceLifecycleOwner<T>,
    recordings: &'a mut ReplacementRecordingOwner,
    images: &'a mut ReplacementImageStateOwner,
}

fn commit_driver_accepted_image_blit_with_watch<Semantic: Clone, T>(
    owners: ReplacementImageBlitAcceptanceCoreOwners<'_, Semantic, T>,
    submission: PreparedReplacementQueueSubmission<ResolvedReplayCompletion<Semantic>>,
    blit: PreparedImageBlit,
    image_state: PreparedImageState,
    watch: impl FnOnce(
        reims_vgpu_core::QueueTimelinePoint,
    ) -> Result<(), crate::replacement_completion::ReplacementTimelineWatchError>,
) -> Result<AcceptedReplacementImageBlit<T>, Box<ReplacementImageBlitAcceptanceFailure<Semantic>>> {
    let ReplacementImageBlitAcceptanceCoreOwners {
        runtime,
        native,
        resources,
        recordings,
        images,
    } = owners;
    let transaction = submission.prepared.plan().transaction;
    let point = submission.prepared.point();
    let resources_completion = blit.resource_completions();
    let reason = if transaction != blit.transaction() || transaction != image_state.transaction() {
        Some(ReplacementImageBlitAcceptanceError::TransactionMismatch)
    } else if submission.prepared.semantic().resources != resources_completion {
        Some(ReplacementImageBlitAcceptanceError::CompletionSetMismatch)
    } else if !same_backings(submission.recording().backings.as_ref(), &blit.backings()) {
        Some(ReplacementImageBlitAcceptanceError::BackingSetMismatch)
    } else if submission.recording().queue_family != image_state.queue_family() {
        Some(ReplacementImageBlitAcceptanceError::QueueFamilyMismatch)
    } else if !same_points(submission.auxiliary_waits(), &image_state.release_points()) {
        Some(ReplacementImageBlitAcceptanceError::AuxiliaryWaitSetMismatch)
    } else {
        images
            .validate_acceptance(&image_state, point)
            .err()
            .map(ReplacementImageBlitAcceptanceError::ImageState)
    };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementImageBlitAcceptanceFailure {
            reason,
            submission,
            blit,
            image_state,
        }));
    }
    let replay = match commit_driver_accepted_with_watch(
        runtime, native, resources, recordings, submission, watch,
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(ReplacementImageBlitAcceptanceFailure {
                reason: ReplacementImageBlitAcceptanceError::Replay(failure.reason.clone()),
                submission: failure.submission,
                blit,
                image_state,
            }));
        }
    };
    images
        .accepted(image_state, replay.replay.native.point)
        .unwrap_or_else(|_| unreachable!("image acceptance was prevalidated"));
    Ok(AcceptedReplacementImageBlit {
        replay,
        operation: blit.operation().clone(),
        resources: resources_completion,
    })
}

fn same_backings(
    left: &[reims_vgpu_protocol::BackingId],
    right: &[reims_vgpu_protocol::BackingId],
) -> bool {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_unstable();
    left.dedup();
    right.sort_unstable();
    right.dedup();
    left == right
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_image_state::{
        ReplacementImageKey, ReplacementImageSharing, ReplacementImageState, ReplacementImageUse,
    };
    use crate::replacement_recording::ReplacementNativeRecording;
    use crate::replacement_submit::QueueTimelineSemaphores;
    use ash::vk;
    use reims_vgpu_core::BackingView;
    use reims_vgpu_core::{
        prepare_image_blit, BackingRegion, CompletionStamp, DeviceTransactionPayload,
        ExecTransaction, LinearRange, QueueTimelinePoint, RepresentationRoute, ResolvedBufferRange,
        ResolvedBufferToTextureBlit, ResolvedLinearTextureLevel, ResolvedResourceLifecycle,
        ResolvedTextureBacking, ResolvedTextureEndpoint, ResourceLifecycleEffect,
        SessionGeneration, StorageBacking, TextureExtent, TextureOrigin, TransactionRecordingPlan,
        WaitDependencyCause, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ByteLength, ChannelId, GuestVirtualAddress, QueueOwnerId, QueueTimelineValue, ResourceId,
        SessionGenerationId, SubmissionDomainId, SubmissionId, SubmissionIdentity, TaskId,
        TransactionId, VulkanDeviceEpochId,
    };

    fn point(queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(1),
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    #[test]
    fn auxiliary_wait_comparison_is_order_independent_but_exact() {
        assert!(same_points(
            &[point(1, 2), point(2, 3)],
            &[point(2, 3), point(1, 2)]
        ));
        assert!(!same_points(&[point(1, 2)], &[point(1, 3)]));
        assert!(!same_points(&[point(1, 2)], &[point(1, 2), point(1, 2)]));
    }

    #[test]
    fn destination_acceptance_commits_replay_content_and_image_state_together() {
        let epoch = VulkanDeviceEpochId::new(1);
        let generation = SessionGenerationId::new(1);
        let queue = QueueOwnerId::new(2);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<(), (), (), (), ()>::Exec(ExecTransaction {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(1),
                        task: TaskId::new(1),
                    },
                    prologue: reims_vgpu_core::ExecPrologue::default(),
                    streams: Box::new([]),
                    accesses: Box::new([]),
                }),
            )
            .unwrap();
        let mut resources = ResourceLifecycleOwner::new(epoch);
        let mut create_backing = |region| match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let source = create_backing(BackingRegion::Linear(LinearRange::new(0, 64).unwrap()));
        let destination = create_backing(BackingRegion::Whole);
        let source_representation = resources
            .create_execution_representation(
                source,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                "source",
            )
            .unwrap();
        // The image is keyed by the texture declared over the backing; this
        // fixture names only one.
        let destination_texture = ResourceId::<reims_vgpu_protocol::ResourceObject>::new(2, 1);
        let destination_representation = resources
            .create_execution_representation(
                destination,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Image(reims_vgpu_core::ImageOwner::base(destination_texture)),
                "destination",
            )
            .unwrap();
        let source_region = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let snapshot = resources
            .snapshot_content(source, &[source_region])
            .unwrap();
        for transfer in resources
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
            resources.complete_transfer(transfer).unwrap();
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
                base: ResourceId::new(2, 1),
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
        let blit = prepare_image_blit(
            &mut resources,
            transaction.id,
            SubmissionId::new(1),
            operation,
        )
        .unwrap();
        let completions = blit.resource_completions();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(TransactionRecordingPlan {
                transaction: transaction.id,
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        runtime.recorded(transaction.id).unwrap();
        runtime.take_submission_ready();
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
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: completions.clone(),
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
        recording.backings = Box::new([destination]);
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
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
        let image_state = images
            .prepare(
                transaction.id,
                0,
                [ReplacementImageUse {
                    image,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }],
            )
            .unwrap();
        let mut recordings = ReplacementRecordingOwner::new(epoch);
        let failure = commit_driver_accepted_image_blit_with_watch(
            ReplacementImageBlitAcceptanceCoreOwners {
                runtime: &mut runtime,
                native: &mut native,
                resources: &mut resources,
                recordings: &mut recordings,
                images: &mut images,
            },
            submission,
            blit,
            image_state,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementImageBlitAcceptanceError::BackingSetMismatch
        );
        let (prepared, mut recording) = failure.submission.into_parts();
        recording.backings = failure.blit.backings();
        let submission =
            PreparedReplacementQueueSubmission::new(prepared, &timelines, recording).unwrap();
        let accepted = commit_driver_accepted_image_blit_with_watch(
            ReplacementImageBlitAcceptanceCoreOwners {
                runtime: &mut runtime,
                native: &mut native,
                resources: &mut resources,
                recordings: &mut recordings,
                images: &mut images,
            },
            submission,
            failure.blit,
            failure.image_state,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(accepted.resources, completions);
        assert_eq!(accepted.replay.replay.native.point, point);
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
