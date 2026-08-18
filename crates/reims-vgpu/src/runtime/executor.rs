//! Device-owned backend submission port.
//!
//! The request payloads are the existing engine request types during the
//! compatibility migration. The surrounding identity, full resource list and
//! segment boundary are backend-independent and already cross the final port.

use crate::backend::vulkan::engine::{
    ComputeOutput, ComputeRequest, DrawError, DrawOutput, DrawRequest, EngineFacadeDecline,
};
use reims_vgpu_protocol::{SegmentBoundary, SubmissionIdentity, SubmissionResourceUse};
use std::sync::Arc;

/// Protocol context shared by every operation in one submitted command stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionContext {
    pub identity: SubmissionIdentity,
    pub resources: Arc<[SubmissionResourceUse]>,
    pub segment: Option<SegmentBoundary>,
}

impl SubmissionContext {
    /// Context for direct test/tool calls that do not originate in an EXEC
    /// packet. Product submissions always replace this with their decoded
    /// resource list and segment boundary.
    pub fn standalone(task_id: u32) -> Self {
        Self {
            identity: SubmissionIdentity {
                id: reims_vgpu_protocol::SubmissionId::new(0),
                task: reims_vgpu_protocol::TaskId::new(task_id),
            },
            resources: Arc::from([]),
            segment: None,
        }
    }
}

/// Snapshot the active protocol context before entering a backend call.
pub fn context_for(state: &crate::model::DeviceState, task_id: u32) -> SubmissionContext {
    state
        .active_submission
        .clone()
        .unwrap_or_else(|| SubmissionContext::standalone(task_id))
}

/// One fully resolved operation accepted by the execution port.
pub enum ResolvedSubmission<'a> {
    Draw {
        context: &'a SubmissionContext,
        request: &'a DrawRequest,
    },
    Compute {
        context: &'a SubmissionContext,
        request: &'a ComputeRequest,
    },
}

impl ResolvedSubmission<'_> {
    fn kind(&self) -> &'static str {
        match self {
            Self::Draw { .. } => "draw",
            Self::Compute { .. } => "compute",
        }
    }
}

/// Completion returned through the same port as its submission.
#[derive(Debug)]
pub enum ExecutionCompletion {
    Draw(DrawOutput),
    Compute(ComputeOutput),
}

impl ExecutionCompletion {
    fn kind(&self) -> &'static str {
        match self {
            Self::Draw(_) => "draw",
            Self::Compute(_) => "compute",
        }
    }
}

/// Backend execution contract implemented per device.
pub trait Executor: std::fmt::Debug + Send + Sync {
    fn execute(&self, submission: ResolvedSubmission<'_>)
        -> Result<ExecutionCompletion, DrawError>;

    /// End one guest lifetime while preserving shareable physical-GPU state.
    fn reset(&self) {}
}

/// Compatibility adapter over the current Vulkan engine facade.
#[derive(Debug, Default)]
pub struct VulkanExecutor;

impl Executor for VulkanExecutor {
    fn execute(
        &self,
        submission: ResolvedSubmission<'_>,
    ) -> Result<ExecutionCompletion, DrawError> {
        match submission {
            ResolvedSubmission::Draw { request, .. } => {
                crate::backend::vulkan::engine::execute_draw_request(request)
                    .map(ExecutionCompletion::Draw)
            }
            ResolvedSubmission::Compute { request, .. } => {
                crate::backend::vulkan::engine::execute_compute_request(request)
                    .map(ExecutionCompletion::Compute)
            }
        }
    }

    fn reset(&self) {
        crate::backend::vulkan::engine::reset_guest_state();
    }
}

/// Execute a draw and enforce that the executor returns the matching completion.
pub fn execute_draw(
    executor: &dyn Executor,
    context: &SubmissionContext,
    request: &DrawRequest,
) -> Result<DrawOutput, DrawError> {
    let expected = ResolvedSubmission::Draw { context, request };
    let expected_kind = expected.kind();
    match executor.execute(expected)? {
        ExecutionCompletion::Draw(output) => Ok(output),
        other => Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: expected_kind,
                actual: other.kind(),
            },
        )),
    }
}

/// Execute a compute dispatch and enforce the matching completion kind.
pub fn execute_compute(
    executor: &dyn Executor,
    context: &SubmissionContext,
    request: &ComputeRequest,
) -> Result<ComputeOutput, DrawError> {
    let expected = ResolvedSubmission::Compute { context, request };
    let expected_kind = expected.kind();
    match executor.execute(expected)? {
        ExecutionCompletion::Compute(output) => Ok(output),
        other => Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: expected_kind,
                actual: other.kind(),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, DeviceState};
    use reims_vgpu_protocol::{
        ObjectRef, ResourceValidity, SegmentKind, SubmissionId, SubmissionResourceUse, TaskId,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };

    #[derive(Clone, Copy, Debug)]
    enum ScriptedCompletion {
        Draw,
        Compute,
    }

    #[derive(Debug)]
    struct ScriptedExecutor {
        completion: ScriptedCompletion,
        seen: Mutex<Vec<SubmissionContext>>,
        resets: AtomicUsize,
    }

    impl ScriptedExecutor {
        fn new(completion: ScriptedCompletion) -> Self {
            Self {
                completion,
                seen: Mutex::new(Vec::new()),
                resets: AtomicUsize::new(0),
            }
        }
    }

    impl Executor for ScriptedExecutor {
        fn execute(
            &self,
            submission: ResolvedSubmission<'_>,
        ) -> Result<ExecutionCompletion, DrawError> {
            let context = match submission {
                ResolvedSubmission::Draw { context, .. }
                | ResolvedSubmission::Compute { context, .. } => context,
            };
            self.seen.lock().unwrap().push(context.clone());
            Ok(match self.completion {
                ScriptedCompletion::Draw => ExecutionCompletion::Draw(DrawOutput::default()),
                ScriptedCompletion::Compute => {
                    ExecutionCompletion::Compute(ComputeOutput::default())
                }
            })
        }

        fn reset(&self) {
            self.resets.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn context() -> SubmissionContext {
        SubmissionContext {
            identity: SubmissionIdentity {
                id: SubmissionId::new(19),
                task: TaskId::new(7),
            },
            resources: Arc::from([SubmissionResourceUse {
                object: ObjectRef::new(31),
                validity: ResourceValidity {
                    clear_host: true,
                    set_host: false,
                    clear_guest: false,
                    set_guest: true,
                },
            }]),
            segment: Some(SegmentBoundary {
                index: 3,
                kind: SegmentKind::Render,
                continues_previous: true,
                continues_next: false,
            }),
        }
    }

    #[test]
    fn device_injected_executor_receives_the_complete_submission_context() {
        let scripted = Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw));
        let state = DeviceState::new_with_executor(DeviceId(1), 12, scripted.clone());
        let context = context();

        execute_draw(state.executor.as_ref(), &context, &DrawRequest::default()).unwrap();

        let seen = scripted.seen.lock().unwrap();
        assert_eq!(seen.as_slice(), &[context]);
    }

    #[test]
    fn executor_cannot_return_a_completion_for_another_operation_kind() {
        let scripted = ScriptedExecutor::new(ScriptedCompletion::Compute);
        let error = execute_draw(&scripted, &context(), &DrawRequest::default()).unwrap_err();

        assert!(matches!(
            error,
            DrawError::Facade(EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: "draw",
                actual: "compute",
            })
        ));
    }

    #[test]
    fn compute_uses_the_same_execution_port() {
        let scripted = ScriptedExecutor::new(ScriptedCompletion::Compute);
        execute_compute(&scripted, &context(), &ComputeRequest::default()).unwrap();

        assert_eq!(scripted.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn resetting_one_device_preserves_its_executor_and_does_not_reset_another() {
        let first_executor = Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw));
        let second_executor = Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw));
        let mut first =
            crate::model::Device::new_with_executor(DeviceId(1), 12, first_executor.clone());
        let mut second =
            crate::model::Device::new_with_executor(DeviceId(2), 12, second_executor.clone());

        first.reset();
        execute_draw(
            first.state.executor.as_ref(),
            &context(),
            &DrawRequest::default(),
        )
        .unwrap();
        execute_draw(
            second.state.executor.as_ref(),
            &context(),
            &DrawRequest::default(),
        )
        .unwrap();

        assert_eq!(first_executor.resets.load(Ordering::Relaxed), 1);
        assert_eq!(second_executor.resets.load(Ordering::Relaxed), 0);
        assert_eq!(first_executor.seen.lock().unwrap().len(), 1);
        assert_eq!(second_executor.seen.lock().unwrap().len(), 1);
        second.reset();
    }
}
