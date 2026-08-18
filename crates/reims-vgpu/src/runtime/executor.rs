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

/// Dynamic executor-session scope for one device operation.
pub struct ExecutionScope {
    _engine: Option<crate::backend::vulkan::engine::SessionScope>,
}

impl ExecutionScope {
    fn none() -> Self {
        Self { _engine: None }
    }
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

/// One fully resolved, owned operation accepted by the execution port.
///
/// Owning both context and request lets an executor retain or queue the work;
/// no decoded accumulator can mutate the inputs after submission.
pub enum ResolvedSubmission {
    Draw {
        context: SubmissionContext,
        request: Box<DrawRequest>,
    },
    Compute {
        context: SubmissionContext,
        request: Box<ComputeRequest>,
    },
}

impl ResolvedSubmission {
    fn kind(&self) -> &'static str {
        match self {
            Self::Draw { .. } => "draw",
            Self::Compute { .. } => "compute",
        }
    }
}

/// Operation-specific result carried by a completion fact.
#[derive(Debug)]
pub enum ExecutionOutput {
    Draw(DrawOutput),
    Compute(ComputeOutput),
}

impl ExecutionOutput {
    fn kind(&self) -> &'static str {
        match self {
            Self::Draw(_) => "draw",
            Self::Compute(_) => "compute",
        }
    }
}

/// Immutable completion returned through the same port as its submission.
#[derive(Debug)]
pub struct ExecutionCompletion {
    pub submission: SubmissionIdentity,
    pub output: ExecutionOutput,
}

/// Backend execution contract implemented per device.
pub trait Executor: std::fmt::Debug + Send + Sync {
    fn execute(&self, submission: ResolvedSubmission) -> Result<ExecutionCompletion, DrawError>;

    /// End one guest lifetime while preserving shareable physical-GPU state.
    fn reset(&self) {}

    /// Select this executor's device-local backend session for a product call.
    fn enter(&self) -> ExecutionScope {
        ExecutionScope::none()
    }
}

/// Compatibility adapter over the current Vulkan engine facade.
#[derive(Debug)]
pub struct VulkanExecutor {
    session: crate::backend::vulkan::engine::SessionId,
}

impl Default for VulkanExecutor {
    fn default() -> Self {
        Self {
            session: crate::backend::vulkan::engine::SessionId::allocate(),
        }
    }
}

impl Drop for VulkanExecutor {
    fn drop(&mut self) {
        crate::backend::vulkan::engine::release_session(self.session);
    }
}

impl Executor for VulkanExecutor {
    fn execute(&self, submission: ResolvedSubmission) -> Result<ExecutionCompletion, DrawError> {
        let _scope = self.enter();
        match submission {
            ResolvedSubmission::Draw { context, request } => {
                crate::backend::vulkan::engine::execute_draw_request(&request).map(|output| {
                    ExecutionCompletion {
                        submission: context.identity,
                        output: ExecutionOutput::Draw(output),
                    }
                })
            }
            ResolvedSubmission::Compute { context, request } => {
                crate::backend::vulkan::engine::execute_compute_request(&request).map(|output| {
                    ExecutionCompletion {
                        submission: context.identity,
                        output: ExecutionOutput::Compute(output),
                    }
                })
            }
        }
    }

    fn reset(&self) {
        let _scope = self.enter();
        crate::backend::vulkan::engine::reset_guest_state();
    }

    fn enter(&self) -> ExecutionScope {
        ExecutionScope {
            _engine: Some(crate::backend::vulkan::engine::enter_session(self.session)),
        }
    }
}

/// Execute a draw and enforce that the executor returns the matching completion.
pub fn execute_draw(
    executor: &dyn Executor,
    context: SubmissionContext,
    request: DrawRequest,
) -> Result<DrawOutput, DrawError> {
    let expected_identity = context.identity;
    let expected = ResolvedSubmission::Draw {
        context,
        request: Box::new(request),
    };
    let expected_kind = expected.kind();
    let completion = executor.execute(expected)?;
    if completion.submission != expected_identity {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected: expected_identity,
                actual: completion.submission,
            },
        ));
    }
    match completion.output {
        ExecutionOutput::Draw(output) => Ok(output),
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
    context: SubmissionContext,
    request: ComputeRequest,
) -> Result<ComputeOutput, DrawError> {
    let expected_identity = context.identity;
    let expected = ResolvedSubmission::Compute {
        context,
        request: Box::new(request),
    };
    let expected_kind = expected.kind();
    let completion = executor.execute(expected)?;
    if completion.submission != expected_identity {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected: expected_identity,
                actual: completion.submission,
            },
        ));
    }
    match completion.output {
        ExecutionOutput::Compute(output) => Ok(output),
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
            submission: ResolvedSubmission,
        ) -> Result<ExecutionCompletion, DrawError> {
            let context = match submission {
                ResolvedSubmission::Draw { context, .. }
                | ResolvedSubmission::Compute { context, .. } => context,
            };
            let identity = context.identity;
            self.seen.lock().unwrap().push(context.clone());
            Ok(ExecutionCompletion {
                submission: identity,
                output: match self.completion {
                    ScriptedCompletion::Draw => ExecutionOutput::Draw(DrawOutput::default()),
                    ScriptedCompletion::Compute => {
                        ExecutionOutput::Compute(ComputeOutput::default())
                    }
                },
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
                resource: None,
                expected_content: None,
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

        execute_draw(
            state.executor.as_ref(),
            context.clone(),
            DrawRequest::default(),
        )
        .unwrap();

        let seen = scripted.seen.lock().unwrap();
        assert_eq!(seen.as_slice(), &[context]);
    }

    #[test]
    fn executor_cannot_return_a_completion_for_another_operation_kind() {
        let scripted = ScriptedExecutor::new(ScriptedCompletion::Compute);
        let error = execute_draw(&scripted, context(), DrawRequest::default()).unwrap_err();

        assert!(matches!(
            error,
            DrawError::Facade(EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: "draw",
                actual: "compute",
            })
        ));
    }

    #[derive(Debug)]
    struct WrongIdentityExecutor;

    impl Executor for WrongIdentityExecutor {
        fn execute(
            &self,
            submission: ResolvedSubmission,
        ) -> Result<ExecutionCompletion, DrawError> {
            let task = match submission {
                ResolvedSubmission::Draw { context, .. }
                | ResolvedSubmission::Compute { context, .. } => context.identity.task,
            };
            Ok(ExecutionCompletion {
                submission: SubmissionIdentity {
                    id: SubmissionId::new(20),
                    task,
                },
                output: ExecutionOutput::Draw(DrawOutput::default()),
            })
        }
    }

    #[test]
    fn completion_identity_must_match_the_owned_submission() {
        let error =
            execute_draw(&WrongIdentityExecutor, context(), DrawRequest::default()).unwrap_err();
        assert!(matches!(
            error,
            DrawError::Facade(EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected,
                actual,
            }) if expected == context().identity && actual == SubmissionIdentity {
                id: SubmissionId::new(20),
                task: TaskId::new(7),
            }
        ));
    }

    #[test]
    fn compute_uses_the_same_execution_port() {
        let scripted = ScriptedExecutor::new(ScriptedCompletion::Compute);
        execute_compute(&scripted, context(), ComputeRequest::default()).unwrap();

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
            context(),
            DrawRequest::default(),
        )
        .unwrap();
        execute_draw(
            second.state.executor.as_ref(),
            context(),
            DrawRequest::default(),
        )
        .unwrap();

        assert_eq!(first_executor.resets.load(Ordering::Relaxed), 1);
        assert_eq!(second_executor.resets.load(Ordering::Relaxed), 0);
        assert_eq!(first_executor.seen.lock().unwrap().len(), 1);
        assert_eq!(second_executor.seen.lock().unwrap().len(), 1);
        second.reset();
    }
}
