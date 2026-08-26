//! Exact Info-query projection from one admitted EXEC.
//!
//! The projection retains each flattened operation position beside the query
//! payload. Evaluation consumes this proof, so reply bytes cannot be paired
//! with a separately authored operation or transaction identity.

use crate::{ExecTransaction, ResolvedOperation};
use reims_vgpu_protocol::TransactionId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedInfoQueries<Info> {
    transaction: TransactionId,
    operations: Box<[(usize, Info)]>,
}

impl<Info> AdmittedInfoQueries<Info> {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn operations(&self) -> &[(usize, Info)] {
        &self.operations
    }

    pub(crate) fn from_exec<Render, Compute, Indirect, Completion>(
        transaction: TransactionId,
        exec: &ExecTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Completion>>,
    ) -> Self
    where
        Info: Clone,
    {
        Self {
            transaction,
            operations: exec
                .operations()
                .enumerate()
                .filter_map(|(index, operation)| match operation {
                    ResolvedOperation::InfoQuery(query) => Some((index, query.clone())),
                    ResolvedOperation::EncoderBoundary(_)
                    | ResolvedOperation::Render(_)
                    | ResolvedOperation::Compute(_)
                    | ResolvedOperation::Blit(_)
                    | ResolvedOperation::Event(_)
                    | ResolvedOperation::Fence(_)
                    | ResolvedOperation::Barrier(_)
                    | ResolvedOperation::Participation(_)
                    | ResolvedOperation::ResourceState(_)
                    | ResolvedOperation::IndirectCommand(_)
                    | ResolvedOperation::CompletionEffect(_) => None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl<Info: Clone> AdmittedInfoQueries<Info> {
    pub fn shifted(&self, base: usize) -> Option<Self> {
        Some(Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .map(|(index, info)| Some((index.checked_add(base)?, info.clone())))
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
                .map(|(index, info)| Some((*positions.get(*index)?, info.clone())))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResolvedExecSegment, ResolvedExecStream};
    use reims_vgpu_protocol::{
        SegmentBoundary, SegmentKind, SubmissionId, SubmissionIdentity, TaskId,
    };

    #[test]
    fn projection_retains_exact_flattened_positions() {
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
                        kind: SegmentKind::Info,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([
                        ResolvedOperation::<(), (), u32, (), ()>::Render(()),
                        ResolvedOperation::InfoQuery(7),
                        ResolvedOperation::Compute(()),
                        ResolvedOperation::InfoQuery(9),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let admitted = AdmittedInfoQueries::from_exec(TransactionId::new(3), &exec);
        assert_eq!(admitted.transaction(), TransactionId::new(3));
        assert_eq!(admitted.operations(), &[(1, 7), (3, 9)]);
    }
}
