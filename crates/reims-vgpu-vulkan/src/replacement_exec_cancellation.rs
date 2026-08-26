//! Atomic pre-submit cancellation of a whole EXEC's native reservations.

use crate::{
    replacement_compute::ReplacementComputeImageBindings,
    replacement_exec_image::{
        exec_has_image_uses, validate_exec_image_states, ExecImageStateError,
    },
    replacement_image_state::{
        PreparedImageStateBatch, ReplacementImageStateError, ReplacementImageStateOwner,
    },
    replacement_render::ReplacementRenderImageBindings,
};
use reims_vgpu_core::{
    cancel_prepared_exec_resources, validate_cancel_prepared_exec_resources,
    CancelledExecResources, ExecResourceCancellationError, PreparedExecResources,
    ResourceLifecycleOwner,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementExecCancellationError {
    ImageStateBatchMissing,
    UnexpectedImageStateBatch,
    ImageStateProgram(ExecImageStateError),
    Resources(ExecResourceCancellationError),
    Images(ReplacementImageStateError),
}

#[derive(Debug)]
pub struct ReplacementExecCancellationFailure<
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ReplacementExecCancellationError,
    pub resources: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    pub image_states: Option<PreparedImageStateBatch>,
}

type ExecCancellationResult<T, Compute, NativeCompute, Render, NativeRender> = Result<
    CancelledExecResources<T, Compute, Render>,
    Box<ReplacementExecCancellationFailure<Compute, NativeCompute, Render, NativeRender>>,
>;

pub fn cancel_prepared_exec<T, Compute, NativeCompute, Render, NativeRender>(
    resources: &mut ResourceLifecycleOwner<T>,
    images: &mut ReplacementImageStateOwner,
    prepared: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    image_states: Option<PreparedImageStateBatch>,
) -> ExecCancellationResult<T, Compute, NativeCompute, Render, NativeRender>
where
    Compute: ReplacementComputeImageBindings,
    Render: ReplacementRenderImageBindings,
{
    let has_images = exec_has_image_uses(&prepared);
    let reason =
        if has_images.as_ref().is_ok_and(|has_images| *has_images) && image_states.is_none() {
            Some(ReplacementExecCancellationError::ImageStateBatchMissing)
        } else if has_images.as_ref().is_ok_and(|has_images| !*has_images)
            && image_states
                .as_ref()
                .is_some_and(|states| validate_exec_image_states(&prepared, states).is_err())
        {
            Some(ReplacementExecCancellationError::UnexpectedImageStateBatch)
        } else if let Err(reason) = has_images {
            Some(ReplacementExecCancellationError::ImageStateProgram(reason))
        } else if let Some(reason) = image_states
            .as_ref()
            .and_then(|states| validate_exec_image_states(&prepared, states).err())
        {
            Some(ReplacementExecCancellationError::ImageStateProgram(reason))
        } else if let Err(reason) = validate_cancel_prepared_exec_resources(resources, &prepared) {
            Some(ReplacementExecCancellationError::Resources(reason))
        } else {
            image_states
                .as_ref()
                .and_then(|states| images.validate_batch(states).err())
                .map(ReplacementExecCancellationError::Images)
        };
    if let Some(reason) = reason {
        return Err(Box::new(ReplacementExecCancellationFailure {
            reason,
            resources: prepared,
            image_states,
        }));
    }

    let cancelled = cancel_prepared_exec_resources(resources, prepared)
        .unwrap_or_else(|_| unreachable!("whole-EXEC resources were prevalidated"));
    if let Some(states) = image_states {
        images
            .cancel_batch(states)
            .expect("whole-EXEC image state was prevalidated before cancellation");
    }
    Ok(cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_image_state::{
        ReplacementImageKey, ReplacementImageSharing, ReplacementImageState, ReplacementImageUse,
    };
    use ash::vk;
    use reims_vgpu_core::{
        assemble_prepared_exec_resources, prepare_buffer_blit_with_write, BackingRegion,
        BufferFillPattern, ExecTransaction, GpuWriteId, LinearRange, PreparedExecResourceInputs,
        RepresentationRoute, ResolvedBlit, ResolvedBufferRange, ResolvedExecSegment,
        ResolvedExecStream, ResolvedInfoOperation, ResolvedOperation, ResolvedResourceLifecycle,
        ResourceLifecycleEffect, StorageBacking,
    };
    use reims_vgpu_protocol::{
        BackingId, ByteLength, GuestVirtualAddress, RepresentationId, ResourceId, SegmentBoundary,
        SegmentKind, SubmissionId, SubmissionIdentity, TaskId, TransactionId, VulkanDeviceEpochId,
    };

    #[test]
    fn unexpected_image_batch_refuses_without_consuming_either_owner() {
        let epoch = VulkanDeviceEpochId::new(1);
        let transaction = TransactionId::new(3);
        let submission = SubmissionId::new(4);
        let mut resources = ResourceLifecycleOwner::new(epoch);
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 16).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let operation = ResolvedBlit::Fill {
            destination: ResolvedBufferRange {
                resource: ResourceId::new(1, 1),
                storage: backing,
                region: LinearRange::new(0, 16).unwrap(),
                address: GuestVirtualAddress::new(0x1000),
                length: ByteLength::new(16),
            },
            pattern: BufferFillPattern::Byte(7),
        };
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
        let prepared = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([prepare_buffer_blit_with_write(
                    &mut resources,
                    transaction,
                    GpuWriteId::operation(submission, 0),
                    operation,
                )
                .unwrap()]),
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

        let image = ReplacementImageKey {
            backing: BackingId::new(99),
            representation: RepresentationId::new(100),
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
        let image_states = images
            .prepare_batch(
                transaction,
                0,
                vec![(
                    1,
                    Box::new([ReplacementImageUse {
                        image,
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    }]) as Box<[_]>,
                )]
                .into_boxed_slice(),
            )
            .unwrap();
        let failure =
            cancel_prepared_exec(&mut resources, &mut images, prepared, Some(image_states))
                .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementExecCancellationError::UnexpectedImageStateBatch
        );
        images
            .validate_batch(failure.image_states.as_ref().unwrap())
            .unwrap();
        let cancelled =
            cancel_prepared_exec(&mut resources, &mut images, failure.resources, None).unwrap();
        assert_eq!(cancelled.buffer_blits.len(), 1);
        images.cancel_batch(failure.image_states.unwrap()).unwrap();
    }
}
