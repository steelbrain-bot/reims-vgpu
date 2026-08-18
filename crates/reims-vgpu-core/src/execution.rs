//! Backend-independent execution-port envelope.
//!
//! The operation payloads are generic because command normalization is being
//! migrated independently from the port itself.  This still fixes the
//! important ownership rule: submission context, operation ordering, and
//! completion identity belong to the semantic core; a backend supplies only
//! its operation payload and result types.

use crate::SubmissionContext;
use reims_vgpu_protocol::SubmissionIdentity;

/// One fully owned operation accepted by an execution backend.
#[derive(Debug)]
pub enum ResolvedSubmission<Draw, Compute> {
    Draw {
        context: SubmissionContext,
        request: Draw,
    },
    Compute {
        context: SubmissionContext,
        request: Compute,
    },
}

impl<Draw, Compute> ResolvedSubmission<Draw, Compute> {
    pub const fn kind(&self) -> ExecutionKind {
        match self {
            Self::Draw { .. } => ExecutionKind::Draw,
            Self::Compute { .. } => ExecutionKind::Compute,
        }
    }
}

/// Semantic operation class used to validate backend completions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionKind {
    Draw,
    Compute,
}

impl ExecutionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Draw => "draw",
            Self::Compute => "compute",
        }
    }
}

/// Operation-specific result carried by a completion fact.
#[derive(Debug)]
pub enum ExecutionOutput<Draw, Compute> {
    Draw(Draw),
    Compute(Compute),
}

impl<Draw, Compute> ExecutionOutput<Draw, Compute> {
    pub const fn kind(&self) -> ExecutionKind {
        match self {
            Self::Draw(_) => ExecutionKind::Draw,
            Self::Compute(_) => ExecutionKind::Compute,
        }
    }
}

/// Immutable completion returned through the same port as its submission.
#[derive(Debug)]
pub struct ExecutionCompletion<Output> {
    pub submission: SubmissionIdentity,
    pub output: Output,
}

/// Validated completion identity paired with its operation-specific output.
#[derive(Debug)]
pub struct ExecutionReceipt<Output> {
    pub submission: SubmissionIdentity,
    pub output: Output,
}

/// The core-owned submission/completion boundary implemented by a backend.
pub trait ExecutionPort: std::fmt::Debug + Send + Sync {
    type Submission;
    type Completion;
    type Error;

    fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{SubmissionId, TaskId};

    #[test]
    fn the_owned_envelope_keeps_context_and_operation_together() {
        let context = SubmissionContext::standalone(7);
        let submission: ResolvedSubmission<u32, ()> = ResolvedSubmission::Draw {
            context: context.clone(),
            request: 41,
        };

        assert_eq!(submission.kind(), ExecutionKind::Draw);
        match submission {
            ResolvedSubmission::Draw {
                context: actual,
                request,
            } => {
                assert_eq!(actual, context);
                assert_eq!(request, 41);
            }
            ResolvedSubmission::Compute { .. } => panic!("draw changed operation kind"),
        }
    }

    #[test]
    fn completion_identity_is_a_separate_fact_from_payload() {
        let completion = ExecutionCompletion {
            submission: SubmissionIdentity {
                id: SubmissionId::new(9),
                task: TaskId::new(3),
            },
            output: ExecutionOutput::<u32, ()>::Draw(17),
        };

        assert_eq!(completion.output.kind(), ExecutionKind::Draw);
        assert_eq!(completion.submission.id, SubmissionId::new(9));
    }
}
