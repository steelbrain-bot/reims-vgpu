//! Device-owned backend submission port.
//!
//! The request payloads are the existing engine request types during the
//! compatibility migration. The surrounding identity, full resource list and
//! segment boundary are backend-independent and already cross the final port.

use crate::backend::vulkan::engine::{
    ComputeOutput, ComputeRequest, DrawError, DrawOutput, DrawRequest, EngineFacadeDecline,
    ResidentContent, ResidentContentBacking, StorageImageFormat, TargetIdentity,
};
use reims_vgpu_protocol::{SegmentBoundary, SubmissionIdentity, SubmissionResourceUse};
use std::sync::Arc;

/// Host-GPU facts available to semantic planning.
///
/// These values describe what the executor can implement. They do not encode
/// guest protocol features or select a guest resource lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutorCapabilities {
    pub device_info: crate::model::DeviceInfoLimits,
    pub max_compute_workgroup_invocations: u32,
    pub thread_execution_width: u32,
    pub max_render_target_dimension: u32,
    pub deferred_gpu_only_content: bool,
}

impl Default for ExecutorCapabilities {
    fn default() -> Self {
        Self {
            device_info: crate::model::DeviceInfoLimits {
                max_sample_count: 1,
                d24_stencil8: false,
                max_threads_per_threadgroup: [128, 128, 64],
                max_threadgroup_memory_bytes: 16_384,
                native_fp16: false,
            },
            max_compute_workgroup_invocations: 128,
            thread_execution_width: 1,
            max_render_target_dimension: 4096,
            deferred_gpu_only_content: false,
        }
    }
}

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

/// Validated completion identity paired with its operation-specific output.
#[derive(Debug)]
pub struct ExecutionReceipt<T> {
    pub submission: SubmissionIdentity,
    pub output: T,
}

/// Backend execution contract implemented per device.
pub trait Executor: std::fmt::Debug + Send + Sync {
    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities::default()
    }

    fn render_target_layout_supported(
        &self,
        layout: crate::contract::pixel_format::TexelLayout,
    ) -> bool {
        matches!(
            layout,
            crate::contract::pixel_format::TexelLayout::Rgba8
                | crate::contract::pixel_format::TexelLayout::Bgra8
        )
    }

    /// Current engine-owned content state for one resolved render target.
    fn resident_content_backing(&self, _identity: &TargetIdentity) -> ResidentContentBacking {
        ResidentContentBacking::NotReady
    }

    fn resident_absent_after_reclaim(
        &self,
        _identity: &TargetIdentity,
    ) -> Option<(crate::backend::vulkan::engine::types::ResidentReclaim, u64)> {
        None
    }

    fn resident_content_epoch(&self, _identity: &TargetIdentity) -> Option<u32> {
        None
    }

    fn resident_content_state(&self, _identity: &TargetIdentity) -> ResidentContent {
        ResidentContent::Absent
    }

    fn stamp_resident_content_epoch(&self, _identity: &TargetIdentity, _epoch: u32) -> bool {
        false
    }

    fn note_resident_content_copied_out(&self, _identity: &TargetIdentity) -> bool {
        false
    }

    fn compute_resident_storage_generation(
        &self,
        _identity: &crate::model::ComputeStorageResidencyKey,
    ) -> Option<u32> {
        None
    }

    fn compute_resident_sample_source(
        &self,
        _identity: &crate::model::ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        None
    }

    fn unpin_resident_storage(&self, _identity: &crate::model::ComputeStorageResidencyKey) {}

    fn retire_resident_storage_content(
        &self,
        _identity: &crate::model::ComputeStorageResidencyKey,
    ) {
    }

    fn note_resident_storage_copied_out(
        &self,
        _identity: &crate::model::ComputeStorageResidencyKey,
    ) {
    }

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
    fn capabilities(&self) -> ExecutorCapabilities {
        let (max_compute_workgroup_invocations, thread_execution_width) =
            crate::backend::vulkan::engine::compute_threadgroup_limits();
        ExecutorCapabilities {
            device_info: crate::backend::vulkan::engine::device_info_limits(),
            max_compute_workgroup_invocations,
            thread_execution_width,
            max_render_target_dimension:
                crate::backend::vulkan::engine::max_render_target_dimension(),
            deferred_gpu_only_content:
                crate::backend::vulkan::engine::deferred_gpu_only_content_allowed(),
        }
    }

    fn render_target_layout_supported(
        &self,
        layout: crate::contract::pixel_format::TexelLayout,
    ) -> bool {
        crate::backend::vulkan::engine::render_target_layout_supported(layout)
    }

    fn resident_content_backing(&self, identity: &TargetIdentity) -> ResidentContentBacking {
        crate::backend::vulkan::engine::resident_content_backing(identity)
    }

    fn resident_absent_after_reclaim(
        &self,
        identity: &TargetIdentity,
    ) -> Option<(crate::backend::vulkan::engine::types::ResidentReclaim, u64)> {
        crate::backend::vulkan::engine::resident_absent_after_reclaim(identity)
    }

    fn resident_content_epoch(&self, identity: &TargetIdentity) -> Option<u32> {
        crate::backend::vulkan::engine::resident_content_epoch(identity)
    }

    fn resident_content_state(&self, identity: &TargetIdentity) -> ResidentContent {
        crate::backend::vulkan::engine::resident_content_state(identity)
    }

    fn stamp_resident_content_epoch(&self, identity: &TargetIdentity, epoch: u32) -> bool {
        crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch)
    }

    fn note_resident_content_copied_out(&self, identity: &TargetIdentity) -> bool {
        crate::backend::vulkan::engine::note_resident_content_copied_out(identity)
    }

    fn compute_resident_storage_generation(
        &self,
        identity: &crate::model::ComputeStorageResidencyKey,
    ) -> Option<u32> {
        crate::backend::vulkan::engine::compute_resident_storage_generation(identity)
    }

    fn compute_resident_sample_source(
        &self,
        identity: &crate::model::ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        crate::backend::vulkan::engine::compute_resident_sample_source(identity)
    }

    fn unpin_resident_storage(&self, identity: &crate::model::ComputeStorageResidencyKey) {
        crate::backend::vulkan::engine::unpin_resident_storage(identity);
    }

    fn retire_resident_storage_content(&self, identity: &crate::model::ComputeStorageResidencyKey) {
        crate::backend::vulkan::engine::retire_resident_storage_content(identity);
    }

    fn note_resident_storage_copied_out(
        &self,
        identity: &crate::model::ComputeStorageResidencyKey,
    ) {
        crate::backend::vulkan::engine::note_resident_storage_copied_out(identity);
    }

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
) -> Result<ExecutionReceipt<DrawOutput>, DrawError> {
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
        ExecutionOutput::Draw(output) => Ok(ExecutionReceipt {
            submission: completion.submission,
            output,
        }),
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
) -> Result<ExecutionReceipt<ComputeOutput>, DrawError> {
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
        ExecutionOutput::Compute(output) => Ok(ExecutionReceipt {
            submission: completion.submission,
            output,
        }),
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
        capabilities: ExecutorCapabilities,
        resident_generation: Option<u32>,
        seen: Mutex<Vec<SubmissionContext>>,
        resets: AtomicUsize,
    }

    impl ScriptedExecutor {
        fn new(completion: ScriptedCompletion) -> Self {
            Self {
                completion,
                capabilities: ExecutorCapabilities::default(),
                resident_generation: None,
                seen: Mutex::new(Vec::new()),
                resets: AtomicUsize::new(0),
            }
        }

        fn with_max_render_target_dimension(mut self, dimension: u32) -> Self {
            self.capabilities.max_render_target_dimension = dimension;
            self
        }

        fn with_resident_generation(mut self, generation: u32) -> Self {
            self.resident_generation = Some(generation);
            self
        }
    }

    impl Executor for ScriptedExecutor {
        fn capabilities(&self) -> ExecutorCapabilities {
            self.capabilities
        }

        fn compute_resident_storage_generation(
            &self,
            _identity: &crate::model::ComputeStorageResidencyKey,
        ) -> Option<u32> {
            self.resident_generation
        }

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
    fn device_injected_executor_owns_residency_queries() {
        let scripted = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Compute).with_resident_generation(41),
        );
        let state = DeviceState::new_with_executor(DeviceId(1), 12, scripted);
        let key = crate::model::ComputeStorageResidencyKey::linear(
            2,
            3,
            0x4000,
            256,
            4096,
            64,
            16,
            crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM,
        );

        assert_eq!(
            state.executor.compute_resident_storage_generation(&key),
            Some(41)
        );
    }

    #[test]
    fn executor_capabilities_are_device_owned() {
        let first = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Draw).with_max_render_target_dimension(4096),
        );
        let second = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Draw)
                .with_max_render_target_dimension(16_384),
        );
        let first_state = DeviceState::new_with_executor(DeviceId(1), 12, first);
        let second_state = DeviceState::new_with_executor(DeviceId(2), 12, second);

        assert_eq!(
            first_state
                .executor
                .capabilities()
                .max_render_target_dimension,
            4096
        );
        assert_eq!(
            second_state
                .executor
                .capabilities()
                .max_render_target_dimension,
            16_384
        );
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
