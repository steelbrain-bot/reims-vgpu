//! Failure-safe preparation of native buffer blits.
//!
//! The dependency/resource owner captures source content, selects only the
//! construction-designated execution representations, reserves destination
//! versions, and retains every representation before recording can begin.
//! A prepared value is the ownership token for those changes; cancellation
//! validates the complete token before returning both reservations and native
//! lifetime holds.

use crate::{
    BackingRegion, BackingView, BlitKind, BufferFillPattern, DirectReplayNativeOwner,
    GpuWriteBatchError, GpuWriteRequest, GpuWriteReservation, LinearRange, ManagedBackingError,
    ManagedBackingProgress, PreparedNativeSubmission, ReplayAcceptance, ReplayAcceptanceError,
    RepresentationUse, ResolvedBlit, ResolvedReplayCompletion, ResolvedResourceCompletion,
    ResourceLifecycleOwner, ResourceUseBatchError, TransactionRuntime,
};
use reims_vgpu_protocol::{BackingId, RepresentationId, SubmissionId, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedNativeBufferRange {
    pub backing: BackingId,
    pub representation: RepresentationId,
    pub region: LinearRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedNativeBufferBlit {
    Fill {
        destination: PreparedNativeBufferRange,
        pattern: BufferFillPattern,
    },
    Copy {
        source: PreparedNativeBufferRange,
        destination: PreparedNativeBufferRange,
    },
}

#[derive(Debug)]
pub struct PreparedBufferBlit {
    transaction: TransactionId,
    write: crate::GpuWriteId,
    operation: ResolvedBlit,
    native: PreparedNativeBufferBlit,
    uses: Box<[RepresentationUse]>,
    writes: Box<[GpuWriteReservation]>,
}

impl PreparedBufferBlit {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn submission(&self) -> SubmissionId {
        self.write.submission()
    }

    pub const fn write(&self) -> crate::GpuWriteId {
        self.write
    }

    pub const fn operation(&self) -> &ResolvedBlit {
        &self.operation
    }

    pub fn into_operation(self) -> ResolvedBlit {
        self.operation
    }

    pub const fn native(&self) -> PreparedNativeBufferBlit {
        self.native
    }

    pub const fn uses(&self) -> &[RepresentationUse] {
        &self.uses
    }

    pub const fn writes(&self) -> &[GpuWriteReservation] {
        &self.writes
    }

    pub fn backings(&self) -> Box<[BackingId]> {
        self.uses
            .iter()
            .map(|use_| use_.backing)
            .collect::<Vec<_>>()
            .into_boxed_slice()
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
pub enum BufferBlitPreparationError {
    VariantRequiresImageState(BlitKind),
    UnestablishedWordPatternPhase,
    Source {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    Destination {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
    WriteRollback {
        admission: ResourceUseBatchError,
        cancellation: GpuWriteBatchError,
    },
}

#[derive(Debug)]
pub struct BufferBlitPreparationFailure {
    pub reason: BufferBlitPreparationError,
    pub operation: ResolvedBlit,
    pub live_writes: Box<[GpuWriteReservation]>,
}

pub fn prepare_buffer_blit<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    submission: SubmissionId,
    operation: ResolvedBlit,
) -> Result<PreparedBufferBlit, Box<BufferBlitPreparationFailure>> {
    prepare_buffer_blit_with_write(resources, transaction, submission.into(), operation)
}

pub fn prepare_buffer_blit_with_write<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    write: crate::GpuWriteId,
    operation: ResolvedBlit,
) -> Result<PreparedBufferBlit, Box<BufferBlitPreparationFailure>> {
    let native = match &operation {
        ResolvedBlit::Fill {
            destination,
            pattern,
        } => {
            if matches!(pattern, BufferFillPattern::Word(_))
                && !destination.region.start().is_multiple_of(4)
            {
                return Err(Box::new(BufferBlitPreparationFailure {
                    reason: BufferBlitPreparationError::UnestablishedWordPatternPhase,
                    operation: operation.clone(),
                    live_writes: Box::new([]),
                }));
            }
            let representation = resources
                .view_representation(destination.storage, BackingView::Bytes)
                .map_err(|reason| {
                    Box::new(BufferBlitPreparationFailure {
                        reason: BufferBlitPreparationError::Destination {
                            backing: destination.storage,
                            reason,
                        },
                        operation: operation.clone(),
                        live_writes: Box::new([]),
                    })
                })?;
            PreparedNativeBufferBlit::Fill {
                destination: PreparedNativeBufferRange {
                    backing: destination.storage,
                    representation,
                    region: destination.region,
                },
                pattern: *pattern,
            }
        }
        ResolvedBlit::Copy {
            source,
            destination,
        } => {
            let snapshot = resources
                .snapshot_content(source.storage, &[BackingRegion::Linear(source.region)])
                .map_err(|reason| {
                    Box::new(BufferBlitPreparationFailure {
                        reason: BufferBlitPreparationError::Source {
                            backing: source.storage,
                            reason,
                        },
                        operation: operation.clone(),
                        live_writes: Box::new([]),
                    })
                })?;
            let source_representation = resources
                .view_representation_for_snapshot(source.storage, BackingView::Bytes, &snapshot)
                .map_err(|reason| {
                    Box::new(BufferBlitPreparationFailure {
                        reason: BufferBlitPreparationError::Source {
                            backing: source.storage,
                            reason,
                        },
                        operation: operation.clone(),
                        live_writes: Box::new([]),
                    })
                })?;
            let destination_representation = resources
                .view_representation(destination.storage, BackingView::Bytes)
                .map_err(|reason| {
                    Box::new(BufferBlitPreparationFailure {
                        reason: BufferBlitPreparationError::Destination {
                            backing: destination.storage,
                            reason,
                        },
                        operation: operation.clone(),
                        live_writes: Box::new([]),
                    })
                })?;
            PreparedNativeBufferBlit::Copy {
                source: PreparedNativeBufferRange {
                    backing: source.storage,
                    representation: source_representation,
                    region: source.region,
                },
                destination: PreparedNativeBufferRange {
                    backing: destination.storage,
                    representation: destination_representation,
                    region: destination.region,
                },
            }
        }
        operation => {
            return Err(Box::new(BufferBlitPreparationFailure {
                reason: BufferBlitPreparationError::VariantRequiresImageState(operation.kind()),
                operation: operation.clone(),
                live_writes: Box::new([]),
            }));
        }
    };

    let (destination, source) = match native {
        PreparedNativeBufferBlit::Fill { destination, .. } => (destination, None),
        PreparedNativeBufferBlit::Copy {
            source,
            destination,
        } => (destination, Some(source)),
    };
    let writes = resources
        .plan_gpu_writes(
            write,
            [GpuWriteRequest {
                backing: destination.backing,
                representation: destination.representation,
                regions: Box::new([BackingRegion::Linear(destination.region)]),
            }],
        )
        .map_err(|reason| {
            Box::new(BufferBlitPreparationFailure {
                reason: BufferBlitPreparationError::Writes(reason),
                operation: operation.clone(),
                live_writes: Box::new([]),
            })
        })?;
    let mut use_map = BTreeMap::<BackingId, Vec<RepresentationId>>::new();
    use_map
        .entry(destination.backing)
        .or_default()
        .push(destination.representation);
    if let Some(source) = source {
        use_map
            .entry(source.backing)
            .or_default()
            .push(source.representation);
    }
    let uses = use_map
        .into_iter()
        .map(|(backing, mut representations)| {
            representations.sort();
            representations.dedup();
            RepresentationUse {
                backing,
                representations: representations.into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    if let Err(admission) = resources.accept_uses(transaction, &uses) {
        return match resources.cancel_gpu_writes(&writes) {
            Ok(()) => Err(Box::new(BufferBlitPreparationFailure {
                reason: BufferBlitPreparationError::Uses(admission),
                operation,
                live_writes: Box::new([]),
            })),
            Err(cancellation) => Err(Box::new(BufferBlitPreparationFailure {
                reason: BufferBlitPreparationError::WriteRollback {
                    admission,
                    cancellation,
                },
                operation,
                live_writes: writes,
            })),
        };
    }
    Ok(PreparedBufferBlit {
        transaction,
        write,
        operation,
        native,
        uses,
        writes,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BufferBlitCancellationError {
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
}

#[derive(Debug)]
pub struct CancelledBufferBlit<T> {
    pub operation: ResolvedBlit,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

#[derive(Debug)]
pub struct BufferBlitCancellationFailure {
    pub reason: BufferBlitCancellationError,
    pub prepared: PreparedBufferBlit,
}

pub fn cancel_prepared_buffer_blit<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedBufferBlit,
) -> Result<CancelledBufferBlit<T>, Box<BufferBlitCancellationFailure>> {
    if let Err(reason) = resources.validate_cancel_gpu_writes(&prepared.writes) {
        return Err(Box::new(BufferBlitCancellationFailure {
            reason: BufferBlitCancellationError::Writes(reason),
            prepared,
        }));
    }
    if let Err(reason) =
        resources.validate_cancel_representation_uses(prepared.transaction, &prepared.uses)
    {
        return Err(Box::new(BufferBlitCancellationFailure {
            reason: BufferBlitCancellationError::Uses(reason),
            prepared,
        }));
    }
    resources
        .cancel_gpu_writes(&prepared.writes)
        .expect("the complete write cancellation was prevalidated");
    let progress = resources
        .cancel_representation_uses(prepared.transaction, &prepared.uses)
        .expect("the complete representation-use cancellation was prevalidated");
    Ok(CancelledBufferBlit {
        operation: prepared.operation,
        resources: progress,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BufferBlitAcceptanceError {
    CompletionSetMismatch,
    Replay(ReplayAcceptanceError),
}

#[derive(Debug)]
pub struct BufferBlitAcceptanceFailure<Semantic> {
    pub reason: BufferBlitAcceptanceError,
    pub native: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    pub blit: PreparedBufferBlit,
}

#[derive(Debug)]
pub struct AcceptedBufferBlit<T> {
    pub replay: ReplayAcceptance<T>,
    pub operation: ResolvedBlit,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

fn completion_set_matches<Semantic>(
    semantic: &ResolvedReplayCompletion<Semantic>,
    prepared: &PreparedBufferBlit,
) -> bool {
    semantic.resources == prepared.resource_completions()
}

/// Join a driver-accepted queue token with the exact prepared blit. The native
/// semantic completion set must be the one derived from its write tokens;
/// otherwise neither replay nor resource acceptance changes.
pub fn commit_buffer_blit_acceptance<Semantic: Clone, T>(
    runtime: &mut TransactionRuntime<Semantic>,
    native: &mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &mut ResourceLifecycleOwner<T>,
    prepared_native: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    prepared_blit: PreparedBufferBlit,
) -> Result<AcceptedBufferBlit<T>, Box<BufferBlitAcceptanceFailure<Semantic>>> {
    let completions = prepared_blit.resource_completions();
    if !completion_set_matches(prepared_native.semantic(), &prepared_blit) {
        return Err(Box::new(BufferBlitAcceptanceFailure {
            reason: BufferBlitAcceptanceError::CompletionSetMismatch,
            native: prepared_native,
            blit: prepared_blit,
        }));
    }
    let replay = match crate::commit_replay_acceptance(
        runtime,
        native,
        resources,
        prepared_native,
        prepared_blit.transaction,
        prepared_blit.backings(),
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(BufferBlitAcceptanceFailure {
                reason: BufferBlitAcceptanceError::Replay(failure.reason),
                native: failure.prepared,
                blit: prepared_blit,
            }));
        }
    };
    Ok(AcceptedBufferBlit {
        replay,
        operation: prepared_blit.operation,
        resources: completions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompletionStamp, DeviceTransactionPayload, DirectReplayNativeOwner, RepresentationRoute,
        ResolvedBufferRange, ResolvedResourceLifecycle, ResourceLifecycleEffect, SessionGeneration,
        StorageBacking, TransactionRuntime, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ByteLength, ChannelId, ContentVersion, GuestVirtualAddress, QueueOwnerId, ResourceId,
        SessionGenerationId, VulkanDeviceEpochId,
    };

    fn backing(resources: &mut ResourceLifecycleOwner<&'static str>) -> BackingId {
        match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 256).unwrap())]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        }
    }

    fn range(backing: BackingId, resource: u32, start: u64, length: u64) -> ResolvedBufferRange {
        ResolvedBufferRange {
            resource: ResourceId::new(resource, 1),
            storage: backing,
            region: LinearRange::new(start, length).unwrap(),
            address: GuestVirtualAddress::new(0x1000 + start),
            length: ByteLength::new(length),
        }
    }

    fn execution(
        resources: &mut ResourceLifecycleOwner<&'static str>,
        backing: BackingId,
        name: &'static str,
    ) -> RepresentationId {
        resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                name,
            )
            .unwrap()
    }

    fn materialize(
        resources: &mut ResourceLifecycleOwner<&'static str>,
        backing: BackingId,
        representation: RepresentationId,
        region: BackingRegion,
    ) {
        let snapshot = resources.snapshot_content(backing, &[region]).unwrap();
        for transfer in resources
            .plan_transfers(backing, GUEST_REPRESENTATION, representation, &snapshot)
            .unwrap()
            .iter()
            .copied()
        {
            resources.complete_transfer(transfer).unwrap();
        }
    }

    #[test]
    fn fill_preparation_owns_write_and_use_until_failure_cancellation() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = backing(&mut resources);
        let representation = execution(&mut resources, backing, "destination");
        let operation = ResolvedBlit::Fill {
            destination: range(backing, 1, 32, 64),
            pattern: BufferFillPattern::Byte(0x5a),
        };
        let prepared = prepare_buffer_blit(
            &mut resources,
            TransactionId::new(7),
            SubmissionId::new(9),
            operation.clone(),
        )
        .unwrap();
        assert_eq!(
            prepared.native(),
            PreparedNativeBufferBlit::Fill {
                destination: PreparedNativeBufferRange {
                    backing,
                    representation,
                    region: LinearRange::new(32, 64).unwrap(),
                },
                pattern: BufferFillPattern::Byte(0x5a),
            }
        );
        assert_eq!(prepared.uses().len(), 1);
        assert_eq!(
            prepared.writes()[0].regions[0].version,
            ContentVersion::new(2)
        );
        assert_eq!(prepared.backings().as_ref(), [backing]);
        assert_eq!(
            prepared.resource_completions().as_ref(),
            [ResolvedResourceCompletion::GpuWrite {
                backing,
                write: SubmissionId::new(9).into(),
                representation,
            }]
        );
        assert!(!completion_set_matches(
            &ResolvedReplayCompletion {
                semantic: (),
                resources: Box::new([]),
            },
            &prepared,
        ));
        assert!(completion_set_matches(
            &ResolvedReplayCompletion {
                semantic: (),
                resources: prepared.resource_completions(),
            },
            &prepared,
        ));

        let cancelled = cancel_prepared_buffer_blit(&mut resources, prepared).unwrap();
        assert_eq!(cancelled.operation, operation);
        let retry = prepare_buffer_blit(
            &mut resources,
            TransactionId::new(7),
            SubmissionId::new(9),
            cancelled.operation,
        )
        .unwrap();
        assert_eq!(retry.writes()[0].regions[0].version, ContentVersion::new(3));
    }

    #[test]
    fn unaligned_word_fill_refuses_before_reserving_content_or_native_use() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = backing(&mut resources);
        execution(&mut resources, backing, "destination");
        let operation = ResolvedBlit::Fill {
            destination: range(backing, 1, 1, 7),
            pattern: BufferFillPattern::Word([1, 2, 3, 4]),
        };
        let failure = prepare_buffer_blit(
            &mut resources,
            TransactionId::new(7),
            SubmissionId::new(9),
            operation.clone(),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            BufferBlitPreparationError::UnestablishedWordPatternPhase
        );
        assert_eq!(failure.operation, operation);
        assert!(failure.live_writes.is_empty());
    }

    #[test]
    fn copy_refuses_a_stale_source_before_reserving_its_destination() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let source = backing(&mut resources);
        let destination = backing(&mut resources);
        let source_representation = execution(&mut resources, source, "source");
        execution(&mut resources, destination, "destination");
        let operation = ResolvedBlit::Copy {
            source: range(source, 1, 0, 64),
            destination: range(destination, 2, 64, 64),
        };
        let failure = prepare_buffer_blit(
            &mut resources,
            TransactionId::new(2),
            SubmissionId::new(3),
            operation.clone(),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            BufferBlitPreparationError::Source {
                backing: source,
                // The guest still holds this content, so the copy is early
                // rather than lost. See
                // [`ManagedBackingError::ExecutionRepresentationAwaitingContent`].
                reason: ManagedBackingError::ExecutionRepresentationAwaitingContent,
            }
        );
        assert!(failure.live_writes.is_empty());

        materialize(
            &mut resources,
            source,
            source_representation,
            BackingRegion::Linear(LinearRange::new(0, 64).unwrap()),
        );
        let prepared = prepare_buffer_blit(
            &mut resources,
            TransactionId::new(2),
            SubmissionId::new(3),
            failure.operation,
        )
        .unwrap();
        assert_eq!(prepared.operation(), &operation);
        assert_eq!(prepared.uses().len(), 2);
    }

    #[test]
    fn repeated_transaction_use_is_counted_and_exact_cancellation_keeps_the_prior_use() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = backing(&mut resources);
        let representation = execution(&mut resources, backing, "destination");
        let transaction = TransactionId::new(4);
        resources
            .accept_use(backing, transaction, [representation])
            .unwrap();
        let operation = ResolvedBlit::Fill {
            destination: range(backing, 1, 0, 32),
            pattern: BufferFillPattern::Word([1, 2, 3, 4]),
        };
        let prepared =
            prepare_buffer_blit(&mut resources, transaction, SubmissionId::new(5), operation)
                .unwrap();
        cancel_prepared_buffer_blit(&mut resources, prepared).unwrap();
        // The contribution installed before preparation is still live.
        resources.cancel_use(backing, transaction).unwrap();
        assert!(prepare_buffer_blit(
            &mut resources,
            TransactionId::new(6),
            SubmissionId::new(5),
            ResolvedBlit::Fill {
                destination: range(backing, 1, 0, 32),
                pattern: BufferFillPattern::Word([1, 2, 3, 4]),
            },
        )
        .is_ok());
    }

    #[test]
    fn two_preparations_share_one_source_use_and_cancel_independently() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let source = backing(&mut resources);
        let first_destination = backing(&mut resources);
        let second_destination = backing(&mut resources);
        let source_representation = execution(&mut resources, source, "source");
        execution(&mut resources, first_destination, "first");
        execution(&mut resources, second_destination, "second");
        materialize(
            &mut resources,
            source,
            source_representation,
            BackingRegion::Linear(LinearRange::new(0, 32).unwrap()),
        );
        let transaction = TransactionId::new(21);
        let submission = SubmissionId::new(22);
        let first = prepare_buffer_blit(
            &mut resources,
            transaction,
            submission,
            ResolvedBlit::Copy {
                source: range(source, 1, 0, 32),
                destination: range(first_destination, 2, 0, 32),
            },
        )
        .unwrap();
        let second = prepare_buffer_blit(
            &mut resources,
            transaction,
            submission,
            ResolvedBlit::Copy {
                source: range(source, 1, 0, 32),
                destination: range(second_destination, 3, 0, 32),
            },
        )
        .unwrap();

        cancel_prepared_buffer_blit(&mut resources, first).unwrap();
        cancel_prepared_buffer_blit(&mut resources, second).unwrap();
        assert_eq!(
            resources.cancel_use(source, transaction),
            Err(ManagedBackingError::UnknownAcceptedUse)
        );
    }

    #[test]
    fn cancellation_validates_use_state_before_returning_write_reservations() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = backing(&mut resources);
        let representation = execution(&mut resources, backing, "destination");
        let transaction = TransactionId::new(8);
        let prepared = prepare_buffer_blit(
            &mut resources,
            transaction,
            SubmissionId::new(10),
            ResolvedBlit::Fill {
                destination: range(backing, 1, 0, 16),
                pattern: BufferFillPattern::Byte(0),
            },
        )
        .unwrap();
        resources.cancel_use(backing, transaction).unwrap();
        let failure = cancel_prepared_buffer_blit(&mut resources, prepared).unwrap_err();
        assert!(matches!(
            failure.reason,
            BufferBlitCancellationError::Uses(ResourceUseBatchError::Backing {
                backing: found,
                reason: ManagedBackingError::UnknownAcceptedUse,
            }) if found == backing
        ));
        resources
            .accept_use(backing, transaction, [representation])
            .unwrap();
        assert!(cancel_prepared_buffer_blit(&mut resources, failure.prepared).is_ok());
    }

    #[test]
    fn queue_acceptance_joins_the_exact_blit_completion_set_and_native_uses() {
        let generation = SessionGenerationId::new(3);
        let epoch = VulkanDeviceEpochId::new(4);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[crate::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<(), (), (), (), ()>::Exec(crate::ExecTransaction {
                    identity: reims_vgpu_protocol::SubmissionIdentity {
                        id: SubmissionId::new(30),
                        task: reims_vgpu_protocol::TaskId::new(1),
                    },
                    prologue: crate::ExecPrologue::default(),
                    streams: Box::new([]),
                    accesses: Box::new([]),
                }),
            )
            .unwrap();
        let mut resources = ResourceLifecycleOwner::new(epoch);
        let backing = backing(&mut resources);
        let representation = execution(&mut resources, backing, "destination");
        let blit = prepare_buffer_blit(
            &mut resources,
            transaction.id,
            SubmissionId::new(30),
            ResolvedBlit::Fill {
                destination: range(backing, 1, 0, 64),
                pattern: BufferFillPattern::Byte(0x11),
            },
        )
        .unwrap();
        let completions = blit.resource_completions();

        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(runtime.recording_plan(transaction.id).unwrap())
            .unwrap();
        runtime.recorded(transaction.id).unwrap();
        runtime.take_submission_ready();
        let plan = native
            .queue_candidate(
                transaction.id,
                Box::<[(TransactionId, crate::WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(plan.transaction, transaction.id);
        let prepared_native = native
            .prepare(
                plan,
                QueueOwnerId::new(1),
                generation,
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: completions.clone(),
                },
            )
            .unwrap();
        let accepted = commit_buffer_blit_acceptance(
            &mut runtime,
            &mut native,
            &mut resources,
            prepared_native,
            blit,
        )
        .unwrap();
        assert_eq!(accepted.replay.native.transaction, transaction.id);
        assert_eq!(accepted.resources, completions);
        assert!(matches!(
            accepted.resources.as_ref(),
            [ResolvedResourceCompletion::GpuWrite {
                backing: found_backing,
                representation: found_representation,
                ..
            }] if *found_backing == backing && *found_representation == representation
        ));
    }
}
