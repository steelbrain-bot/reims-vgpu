//! Failure-safe preparation of resolved info reply writes.
//!
//! Query evaluation supplies bytes derived from the queried live object. This
//! owner verifies their exact decoded destination shape, reserves the content
//! version, and retains the construction-designated execution representation
//! before a native recorder can observe the payload.

use crate::{
    BackingRegion, EvaluatedInfoQuery, GpuWriteBatchError, GpuWriteRequest, GpuWriteReservation,
    ManagedBackingError, ManagedBackingProgress, PreparedNativeBufferRange, RepresentationUse,
    ResolvedInfoOperation, ResolvedResourceCompletion, ResourceLifecycleOwner,
    ResourceUseBatchError,
};
use reims_vgpu_protocol::{BackingId, SubmissionId, TransactionId};

#[derive(Debug)]
pub struct PreparedInfoQuery {
    transaction: TransactionId,
    index: usize,
    write: crate::GpuWriteId,
    operation: ResolvedInfoOperation,
    destination: PreparedNativeBufferRange,
    bytes: Box<[u8]>,
    uses: Box<[RepresentationUse]>,
    writes: Box<[GpuWriteReservation]>,
}

impl PreparedInfoQuery {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn submission(&self) -> SubmissionId {
        self.write.submission()
    }

    pub const fn write(&self) -> crate::GpuWriteId {
        self.write
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn operation(&self) -> &ResolvedInfoOperation {
        &self.operation
    }

    pub const fn destination(&self) -> PreparedNativeBufferRange {
        self.destination
    }

    pub const fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_evaluated(self) -> EvaluatedInfoQuery {
        EvaluatedInfoQuery::from_parts(self.transaction, self.index, self.operation, self.bytes)
    }

    pub const fn uses(&self) -> &[RepresentationUse] {
        &self.uses
    }

    pub const fn writes(&self) -> &[GpuWriteReservation] {
        &self.writes
    }

    pub fn resource_completions(&self) -> Box<[ResolvedResourceCompletion]> {
        self.writes
            .iter()
            .map(|write| ResolvedResourceCompletion::GpuWrite {
                backing: write.backing,
                write: write.write,
                representation: write.representation,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InfoQueryPreparationError {
    ReplyLengthMismatch {
        expected: u64,
        actual: usize,
    },
    NativeUpdateAlignment,
    Destination(ManagedBackingError),
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
    WriteRollback {
        admission: ResourceUseBatchError,
        cancellation: GpuWriteBatchError,
    },
}

#[derive(Debug)]
pub struct InfoQueryPreparationFailure {
    pub reason: InfoQueryPreparationError,
    pub evaluated: EvaluatedInfoQuery,
    pub live_writes: Box<[GpuWriteReservation]>,
}

fn destination(operation: &ResolvedInfoOperation) -> (BackingId, crate::LinearRange) {
    match operation {
        ResolvedInfoOperation::RenderPipelineState { reply, .. }
        | ResolvedInfoOperation::ComputePipelineState { reply, .. }
        | ResolvedInfoOperation::ResourceHost { reply, .. }
        | ResolvedInfoOperation::HeapHost { reply, .. }
        | ResolvedInfoOperation::SamplerHost { reply, .. }
        | ResolvedInfoOperation::HeapTextureSizeAndAlign { reply, .. }
        | ResolvedInfoOperation::RenderPipelineImageblock { reply, .. }
        | ResolvedInfoOperation::ComputePipelineImageblock { reply, .. }
        | ResolvedInfoOperation::RateMapInfo { reply, .. }
        | ResolvedInfoOperation::MapCoordinate { reply, .. } => (reply.backing, reply.range),
        ResolvedInfoOperation::CopyRateParameterBuffer { destination, .. } => {
            (destination.backing, destination.range)
        }
    }
}

pub fn prepare_info_query<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    submission: SubmissionId,
    evaluated: EvaluatedInfoQuery,
) -> Result<PreparedInfoQuery, Box<InfoQueryPreparationFailure>> {
    let (transaction, index, operation, bytes) = evaluated.into_parts();
    let recover =
        |bytes: Box<[u8]>| EvaluatedInfoQuery::from_parts(transaction, index, operation, bytes);
    let (backing, region) = destination(&operation);
    let expected = region.end() - region.start();
    if u64::try_from(bytes.len()).ok() != Some(expected) {
        return Err(Box::new(InfoQueryPreparationFailure {
            reason: InfoQueryPreparationError::ReplyLengthMismatch {
                expected,
                actual: bytes.len(),
            },
            evaluated: recover(bytes),
            live_writes: Box::new([]),
        }));
    }
    if !region.start().is_multiple_of(4) || !expected.is_multiple_of(4) {
        return Err(Box::new(InfoQueryPreparationFailure {
            reason: InfoQueryPreparationError::NativeUpdateAlignment,
            evaluated: recover(bytes),
            live_writes: Box::new([]),
        }));
    }
    let representation = match resources.view_representation(backing, crate::BackingView::Bytes) {
        Ok(representation) => representation,
        Err(reason) => {
            return Err(Box::new(InfoQueryPreparationFailure {
                reason: InfoQueryPreparationError::Destination(reason),
                evaluated: recover(bytes),
                live_writes: Box::new([]),
            }));
        }
    };
    let write = crate::GpuWriteId::operation(transaction, submission, index);
    let writes = resources
        .plan_gpu_writes(
            write,
            [GpuWriteRequest {
                backing,
                representation,
                regions: Box::new([BackingRegion::Linear(region)]),
            }],
        )
        .map_err(|reason| {
            Box::new(InfoQueryPreparationFailure {
                reason: InfoQueryPreparationError::Writes(reason),
                evaluated: recover(bytes.clone()),
                live_writes: Box::new([]),
            })
        })?;
    let uses = vec![RepresentationUse {
        backing,
        representations: Box::new([representation]),
    }]
    .into_boxed_slice();
    if let Err(admission) = resources.accept_uses(transaction, &uses) {
        return match resources.cancel_gpu_writes(&writes) {
            Ok(()) => Err(Box::new(InfoQueryPreparationFailure {
                reason: InfoQueryPreparationError::Uses(admission),
                evaluated: recover(bytes),
                live_writes: Box::new([]),
            })),
            Err(cancellation) => Err(Box::new(InfoQueryPreparationFailure {
                reason: InfoQueryPreparationError::WriteRollback {
                    admission,
                    cancellation,
                },
                evaluated: recover(bytes),
                live_writes: writes,
            })),
        };
    }
    Ok(PreparedInfoQuery {
        transaction,
        index,
        write,
        operation,
        destination: PreparedNativeBufferRange {
            backing,
            representation,
            region,
        },
        bytes,
        uses,
        writes,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InfoQueryCancellationError {
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
}

#[derive(Debug)]
pub struct CancelledInfoQuery<T> {
    pub evaluated: EvaluatedInfoQuery,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

#[derive(Debug)]
pub struct InfoQueryCancellationFailure {
    pub reason: InfoQueryCancellationError,
    pub prepared: PreparedInfoQuery,
}

pub fn cancel_prepared_info_query<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedInfoQuery,
) -> Result<CancelledInfoQuery<T>, Box<InfoQueryCancellationFailure>> {
    if let Err(reason) = resources.validate_cancel_gpu_writes(&prepared.writes) {
        return Err(Box::new(InfoQueryCancellationFailure {
            reason: InfoQueryCancellationError::Writes(reason),
            prepared,
        }));
    }
    if let Err(reason) =
        resources.validate_cancel_representation_uses(prepared.transaction, &prepared.uses)
    {
        return Err(Box::new(InfoQueryCancellationFailure {
            reason: InfoQueryCancellationError::Uses(reason),
            prepared,
        }));
    }
    resources
        .cancel_gpu_writes(&prepared.writes)
        .expect("the complete info write cancellation was prevalidated");
    let progress = resources
        .cancel_representation_uses(prepared.transaction, &prepared.uses)
        .expect("the complete info representation cancellation was prevalidated");
    Ok(CancelledInfoQuery {
        evaluated: EvaluatedInfoQuery::from_parts(
            prepared.transaction,
            prepared.index,
            prepared.operation,
            prepared.bytes,
        ),
        resources: progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackingView;
    use crate::{
        LinearRange, RepresentationRoute, ResolvedInfoReplyTarget, ResolvedResourceLifecycle,
        ResourceLifecycleEffect, StorageBacking,
    };
    use reims_vgpu_protocol::{RenderPipelineObject, ResourceId, VulkanDeviceEpochId};

    fn evaluated(
        operation: ResolvedInfoOperation,
        bytes: impl Into<Box<[u8]>>,
    ) -> EvaluatedInfoQuery {
        EvaluatedInfoQuery::from_parts(TransactionId::new(5), 0, operation, bytes.into())
    }

    #[test]
    fn exact_reply_preparation_owns_destination_write_and_is_recoverable() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 64).unwrap())]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let representation = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let operation = ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::<RenderPipelineObject>::new(3, 1),
            reply: ResolvedInfoReplyTarget {
                resource: ResourceId::new(4, 1),
                backing,
                range: LinearRange::new(8, 16).unwrap(),
                requested_alignment: 4,
            },
        };
        let prepared = prepare_info_query(
            &mut resources,
            SubmissionId::new(6),
            evaluated(operation, [0x5au8; 16]),
        )
        .unwrap();
        assert_eq!(prepared.transaction(), TransactionId::new(5));
        assert_eq!(prepared.index(), 0);
        assert_eq!(prepared.destination().representation, representation);
        assert_eq!(prepared.bytes(), &[0x5a; 16]);
        assert_eq!(prepared.writes().len(), 1);
        assert_eq!(prepared.resource_completions().len(), 1);
        let cancelled = cancel_prepared_info_query(&mut resources, prepared).unwrap();
        assert_eq!(cancelled.evaluated.operation(), &operation);
        assert_eq!(cancelled.evaluated.bytes(), [0x5a; 16]);
        assert_eq!(cancelled.evaluated.transaction(), TransactionId::new(5));
        assert_eq!(cancelled.evaluated.index(), 0);
    }

    #[test]
    fn wrong_reply_length_changes_no_resource_owner() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let operation = ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::<RenderPipelineObject>::new(3, 1),
            reply: ResolvedInfoReplyTarget {
                resource: ResourceId::new(4, 1),
                backing: BackingId::new(9),
                range: LinearRange::new(8, 16).unwrap(),
                requested_alignment: 4,
            },
        };
        let failure = prepare_info_query(
            &mut resources,
            SubmissionId::new(6),
            evaluated(operation, [0u8; 12]),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            InfoQueryPreparationError::ReplyLengthMismatch {
                expected: 16,
                actual: 12,
            }
        );
        assert!(failure.live_writes.is_empty());
        assert_eq!(failure.evaluated.transaction(), TransactionId::new(5));
        assert_eq!(failure.evaluated.index(), 0);
        assert_eq!(failure.evaluated.operation(), &operation);
        assert_eq!(failure.evaluated.bytes(), [0; 12]);
    }
}
