//! Backend-independent immutable execution IR and executor port.
//!
//! A submitted command buffer is an ordered, owned value. Decoders and
//! mutable encoder accumulators do not cross this boundary: draw and compute
//! payloads are already prepared, blits carry resolved resource identities and
//! checked ranges, and resource-state commands carry the exact resource
//! lifetime they may update. [`SubmissionContext`] retains the command stream's
//! complete segment and resource-participation envelope beside those commands.

use crate::{ContentStamp, ResolvedBufferBlit, ResolvedResourceState, SubmissionContext};
use reims_vgpu_protocol::SubmissionIdentity;

/// One fully owned command in a resolved command buffer.
#[derive(Debug)]
pub enum ResolvedCommand<Draw, Compute> {
    Draw(Draw),
    Compute(Compute),
    Blit(ResolvedBufferBlit),
    ResourceState(ResolvedResourceState),
}

impl<Draw, Compute> ResolvedCommand<Draw, Compute> {
    pub const fn kind(&self) -> ExecutionKind {
        match self {
            Self::Draw(_) => ExecutionKind::Draw,
            Self::Compute(_) => ExecutionKind::Compute,
            Self::Blit(_) => ExecutionKind::Blit,
            Self::ResourceState(_) => ExecutionKind::ResourceState,
        }
    }
}

/// Ordered commands from one semantic command-buffer boundary.
#[derive(Debug)]
pub struct ResolvedCommandBuffer<Draw, Compute> {
    commands: Box<[ResolvedCommand<Draw, Compute>]>,
}

impl<Draw, Compute> ResolvedCommandBuffer<Draw, Compute> {
    pub fn new(commands: impl Into<Box<[ResolvedCommand<Draw, Compute>]>>) -> Self {
        Self {
            commands: commands.into(),
        }
    }

    pub fn single(command: ResolvedCommand<Draw, Compute>) -> Self {
        Self::new(vec![command])
    }

    pub fn commands(&self) -> &[ResolvedCommand<Draw, Compute>] {
        &self.commands
    }

    pub fn into_commands(self) -> Box<[ResolvedCommand<Draw, Compute>]> {
        self.commands
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// One immutable submission: protocol context and its ordered command buffer.
#[derive(Debug)]
pub struct ResolvedSubmission<Draw, Compute> {
    pub context: SubmissionContext,
    pub command_buffer: ResolvedCommandBuffer<Draw, Compute>,
}

impl<Draw, Compute> ResolvedSubmission<Draw, Compute> {
    pub fn single(context: SubmissionContext, command: ResolvedCommand<Draw, Compute>) -> Self {
        Self {
            context,
            command_buffer: ResolvedCommandBuffer::single(command),
        }
    }
}

/// Semantic operation class used to validate executor completions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionKind {
    Draw,
    Compute,
    Blit,
    ResourceState,
}

impl ExecutionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Draw => "draw",
            Self::Compute => "compute",
            Self::Blit => "blit",
            Self::ResourceState => "resource_state",
        }
    }
}

/// A successful resolved blit's semantic effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlitCompletion {
    /// The destination version whose bytes were written, when the command has
    /// a destination. A no-op command completes with `None`.
    pub written: Option<ContentStamp>,
}

/// A resource-state command accepted at its ordered point in the submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceStateCompletion {
    pub update: ResolvedResourceState,
}

/// Operation-specific result carried by a completion fact.
#[derive(Debug)]
pub enum ExecutionOutput<Draw, Compute> {
    Draw(Draw),
    Compute(Compute),
    Blit(BlitCompletion),
    ResourceState(ResourceStateCompletion),
}

impl<Draw, Compute> ExecutionOutput<Draw, Compute> {
    pub const fn kind(&self) -> ExecutionKind {
        match self {
            Self::Draw(_) => ExecutionKind::Draw,
            Self::Compute(_) => ExecutionKind::Compute,
            Self::Blit(_) => ExecutionKind::Blit,
            Self::ResourceState(_) => ExecutionKind::ResourceState,
        }
    }
}

/// Immutable completion returned through the same port as its submission.
#[derive(Debug)]
pub struct ExecutionCompletion<Output> {
    pub submission: SubmissionIdentity,
    pub output: Output,
    /// Current semantic versions materialized as persistent GPU replicas.
    pub gpu_materialized: std::sync::Arc<[ContentStamp]>,
}

/// Validated completion identity paired with its operation-specific output.
#[derive(Debug)]
pub struct ExecutionReceipt<Output> {
    pub submission: SubmissionIdentity,
    pub output: Output,
    pub gpu_materialized: std::sync::Arc<[ContentStamp]>,
}

/// The core-owned submission/completion boundary implemented by an executor.
pub trait ExecutionPort: std::fmt::Debug + Send + Sync {
    type Submission;
    type Completion;
    type Error;

    fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{
        ByteLength, ContentVersion, GuestVirtualAddress, ResourceId, SubmissionId, TaskId,
    };

    fn range(index: u32) -> crate::ResolvedBufferRange {
        crate::ResolvedBufferRange {
            content: ContentStamp {
                resource: ResourceId::new(index, 1),
                version: ContentVersion::new(2),
            },
            address: GuestVirtualAddress::new(u64::from(index) << 12),
            length: ByteLength::new(16),
        }
    }

    #[test]
    fn the_owned_envelope_keeps_order_context_and_operation_kinds_together() {
        let context = SubmissionContext::standalone(7);
        let submission: ResolvedSubmission<u32, u32> = ResolvedSubmission {
            context: context.clone(),
            command_buffer: ResolvedCommandBuffer::new(vec![
                ResolvedCommand::Draw(41),
                ResolvedCommand::Blit(ResolvedBufferBlit::Copy {
                    source: range(1),
                    destination: range(2),
                }),
                ResolvedCommand::Compute(43),
            ]),
        };

        assert_eq!(submission.context, context);
        assert_eq!(
            submission
                .command_buffer
                .commands()
                .iter()
                .map(ResolvedCommand::kind)
                .collect::<Vec<_>>(),
            vec![
                ExecutionKind::Draw,
                ExecutionKind::Blit,
                ExecutionKind::Compute
            ]
        );
    }

    #[test]
    fn completion_identity_is_separate_from_the_ordered_outputs() {
        let completion = ExecutionCompletion {
            submission: SubmissionIdentity {
                id: SubmissionId::new(9),
                task: TaskId::new(3),
            },
            output: vec![ExecutionOutput::<u32, ()>::Draw(17)].into_boxed_slice(),
            gpu_materialized: std::sync::Arc::from([]),
        };

        assert_eq!(completion.output[0].kind(), ExecutionKind::Draw);
        assert_eq!(completion.submission.id, SubmissionId::new(9));
    }

    #[test]
    fn resource_state_is_a_command_not_mutable_submission_metadata() {
        let update = ResolvedResourceState {
            object: reims_vgpu_protocol::ObjectRef::new(5),
            resource: None,
            ops: reims_vgpu_protocol::ResourceValidityOps::PAGE_ON,
        };
        let buffer: ResolvedCommandBuffer<(), ()> =
            ResolvedCommandBuffer::single(ResolvedCommand::ResourceState(update));
        assert_eq!(buffer.commands()[0].kind(), ExecutionKind::ResourceState);
    }
}
