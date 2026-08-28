//! Exact resource-state projection from an admitted EXEC.
//!
//! Validity operations whose clear-guest statement is paired with a
//! clear-host statement need no representation copy: the clear-host action
//! establishes a new guest version directly. A clear-guest statement without
//! clear-host may need a host-to-guest transfer, so it remains a distinct
//! native-emitter case until the current representation snapshot is prepared.

use crate::{BackingRegion, ExecTransaction, ResolvedOperation, ResolvedResourceState};
use reims_vgpu_protocol::{BackingId, TransactionId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmittedResourceStateOperation {
    Semantic(ResolvedResourceState),
    NativeTransferMayBeRequired(ResolvedResourceState),
}

impl AdmittedResourceStateOperation {
    pub const fn operation(&self) -> &ResolvedResourceState {
        match self {
            Self::Semantic(operation) | Self::NativeTransferMayBeRequired(operation) => operation,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedResourceStates {
    transaction: TransactionId,
    operations: Box<[(usize, AdmittedResourceStateOperation)]>,
}

impl AdmittedResourceStates {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub fn operations(&self) -> &[(usize, AdmittedResourceStateOperation)] {
        &self.operations
    }

    pub fn semantic_operations(&self) -> impl Iterator<Item = &ResolvedResourceState> {
        self.operations
            .iter()
            .filter_map(|(_, operation)| match operation {
                AdmittedResourceStateOperation::Semantic(operation) => Some(operation),
                AdmittedResourceStateOperation::NativeTransferMayBeRequired(_) => None,
            })
    }

    /// Parts of `region` not established by an earlier ordered GPU write in
    /// this EXEC's resource-state prologue.
    pub fn regions_not_written_before(
        &self,
        position: usize,
        backing: BackingId,
        region: BackingRegion,
        mut write_targets_current_representation: impl FnMut(usize) -> bool,
    ) -> Vec<BackingRegion> {
        let mut missing = vec![region];
        for (write_position, state) in self.operations.iter() {
            if *write_position >= position
                || state.operation().ops.set_host_valid == 0
                || !write_targets_current_representation(*write_position)
            {
                continue;
            }
            for target in state
                .operation()
                .targets
                .iter()
                .filter(|target| target.backing == backing)
            {
                for written in target.regions.iter().copied() {
                    missing = missing
                        .into_iter()
                        .flat_map(|required| {
                            crate::content_authority::not_covered_by(required, written)
                        })
                        .collect();
                    if missing.is_empty() {
                        return missing;
                    }
                }
            }
        }
        missing
    }

    pub fn for_operation_range(&self, start: usize, end: usize) -> Self {
        Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .filter(|(index, _)| *index >= start && *index < end)
                .cloned()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn shifted(&self, base: usize) -> Option<Self> {
        Some(Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .map(|(index, operation)| Some((index.checked_add(base)?, operation.clone())))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }

    pub fn remapped_positions(&self, positions: &[usize]) -> Option<Self> {
        Some(Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .map(|(index, state)| Some((*positions.get(*index)?, state.clone())))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }

    pub(crate) fn from_exec<Render, Compute, Info, Indirect, Completion>(
        transaction: TransactionId,
        exec: &ExecTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Completion>>,
    ) -> Self {
        let operations = exec
            .operations()
            .enumerate()
            .filter_map(|(index, operation)| match operation {
                ResolvedOperation::ResourceState(operation) => {
                    let operation = if operation.ops.clear_guest_valid != 0
                        && operation.ops.clear_host_valid == 0
                    {
                        AdmittedResourceStateOperation::NativeTransferMayBeRequired(
                            operation.clone(),
                        )
                    } else {
                        AdmittedResourceStateOperation::Semantic(operation.clone())
                    };
                    Some((index, operation))
                }
                ResolvedOperation::EncoderBoundary(_)
                | ResolvedOperation::Render(_)
                | ResolvedOperation::Compute(_)
                | ResolvedOperation::Blit(_)
                | ResolvedOperation::Event(_)
                | ResolvedOperation::Fence(_)
                | ResolvedOperation::Barrier(_)
                | ResolvedOperation::Participation(_)
                | ResolvedOperation::InfoQuery(_)
                | ResolvedOperation::IndirectCommand(_)
                | ResolvedOperation::CompletionEffect(_) => None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            transaction,
            operations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResolvedExecSegment, ResolvedExecStream};
    use reims_vgpu_protocol::{
        ResourceValidityOps, SegmentBoundary, SegmentKind, SubmissionId, SubmissionIdentity, TaskId,
    };

    type Operation = ResolvedOperation<(), (), (), (), ()>;

    fn state(ops: ResourceValidityOps) -> Operation {
        Operation::ResourceState(ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([]),
            ops,
        })
    }

    fn exec(operations: impl Into<Box<[Operation]>>) -> ExecTransaction<Operation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(2),
            },
            prologue: crate::ExecPrologue::default(),
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
                    operations: operations.into(),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    #[test]
    fn projection_distinguishes_semantic_only_from_possible_transfer() {
        let admitted = AdmittedResourceStates::from_exec(
            TransactionId::new(7),
            &exec([
                state(ResourceValidityOps::PAGE_ON),
                state(ResourceValidityOps {
                    clear_guest_valid: 1,
                    set_guest_valid: 1,
                    ..ResourceValidityOps::default()
                }),
                state(ResourceValidityOps {
                    clear_host_valid: 1,
                    clear_guest_valid: 1,
                    ..ResourceValidityOps::default()
                }),
            ]),
        );
        assert_eq!(admitted.transaction(), TransactionId::new(7));
        assert!(matches!(
            admitted.operations()[0].1,
            AdmittedResourceStateOperation::Semantic(_)
        ));
        assert!(matches!(
            admitted.operations()[1].1,
            AdmittedResourceStateOperation::NativeTransferMayBeRequired(_)
        ));
        assert!(matches!(
            admitted.operations()[2].1,
            AdmittedResourceStateOperation::Semantic(_)
        ));
        assert_eq!(admitted.semantic_operations().count(), 2);
    }

    #[test]
    fn earlier_resource_state_writes_remove_only_their_exact_read_regions() {
        let backing = BackingId::new(3);
        let linear = |offset, length| {
            BackingRegion::Linear(crate::LinearRange::new(offset, length).unwrap())
        };
        let write = |region| {
            AdmittedResourceStateOperation::Semantic(ResolvedResourceState {
                resource: None,
                mappings: Box::new([]),
                targets: Box::new([crate::ResolvedResourceStateTarget {
                    backing,
                    regions: Box::new([region]),
                }]),
                ops: ResourceValidityOps {
                    set_host_valid: 1,
                    ..ResourceValidityOps::default()
                },
            })
        };
        let admitted = AdmittedResourceStates {
            transaction: TransactionId::new(8),
            operations: Box::new([(2, write(linear(32, 32))), (8, write(linear(64, 32)))]),
        };

        assert_eq!(
            admitted.regions_not_written_before(5, backing, linear(0, 96), |_| true),
            [linear(0, 32), linear(64, 32)]
        );
        assert_eq!(
            admitted.regions_not_written_before(9, backing, linear(0, 96), |_| true),
            [linear(0, 32)]
        );
        assert_eq!(
            admitted
                .regions_not_written_before(9, backing, linear(0, 96), |position| position == 8),
            [linear(0, 64)]
        );
    }
}
