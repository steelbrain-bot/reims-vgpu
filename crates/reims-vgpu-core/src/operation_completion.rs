//! Exact completion-effect projection from an admitted EXEC.
//!
//! Completion effects remain ordered operations for auditability, but have no
//! standalone native command. Admission captures their positions and payloads
//! once; that opaque value is both recording proof and the semantic payload a
//! queue completion retains until timeline-proven commit.

use crate::{ExecTransaction, ResolvedOperation};
use reims_vgpu_protocol::TransactionId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedCompletionEffects<Effect> {
    transaction: TransactionId,
    operations: Box<[(usize, Effect)]>,
}

impl<Effect> AdmittedCompletionEffects<Effect> {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub fn operations(&self) -> &[(usize, Effect)] {
        &self.operations
    }

    pub fn effects(&self) -> impl Iterator<Item = &Effect> {
        self.operations.iter().map(|(_, effect)| effect)
    }
}

impl<Effect: Clone> AdmittedCompletionEffects<Effect> {
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
                .map(|(index, effect)| Some((index.checked_add(base)?, effect.clone())))
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
                .map(|(index, effect)| Some((*positions.get(*index)?, effect.clone())))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }

    pub(crate) fn from_exec<Render, Compute, Info, Indirect>(
        transaction: TransactionId,
        exec: &ExecTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Effect>>,
    ) -> Self {
        let operations = exec
            .operations()
            .enumerate()
            .filter_map(|(index, operation)| match operation {
                ResolvedOperation::CompletionEffect(effect) => Some((index, effect.clone())),
                ResolvedOperation::EncoderBoundary(_)
                | ResolvedOperation::Render(_)
                | ResolvedOperation::Compute(_)
                | ResolvedOperation::Blit(_)
                | ResolvedOperation::Event(_)
                | ResolvedOperation::Fence(_)
                | ResolvedOperation::Barrier(_)
                | ResolvedOperation::Participation(_)
                | ResolvedOperation::ResourceState(_)
                | ResolvedOperation::InfoQuery(_)
                | ResolvedOperation::IndirectCommand(_) => None,
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
    use crate::{EncoderBoundary, ResolvedExecSegment, ResolvedExecStream};
    use reims_vgpu_protocol::{
        SegmentBoundary, SegmentKind, SubmissionId, SubmissionIdentity, TaskId,
    };

    #[test]
    fn projection_keeps_exact_flattened_positions_and_order() {
        let exec = ExecTransaction {
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
                    operations: Box::new([
                        ResolvedOperation::<(), (), (), (), _>::EncoderBoundary(
                            EncoderBoundary::Begin(SegmentKind::Blit),
                        ),
                        ResolvedOperation::CompletionEffect("first"),
                        ResolvedOperation::CompletionEffect("second"),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let admitted = AdmittedCompletionEffects::from_exec(TransactionId::new(7), &exec);
        assert_eq!(admitted.transaction(), TransactionId::new(7));
        assert_eq!(admitted.operations(), &[(1, "first"), (2, "second")]);
        assert_eq!(
            admitted.effects().copied().collect::<Vec<_>>(),
            ["first", "second"]
        );
    }
}
