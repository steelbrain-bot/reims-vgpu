//! Device-owned backend submission port.
//!
//! The request payloads are the existing engine request types during the
//! compatibility migration. The surrounding identity, full resource list and
//! segment boundary are backend-independent and already cross the final port.

use crate::backend::vulkan::engine::{
    ComputeOutput, ComputeRequest, DrawError, DrawOutput, DrawRequest, EngineFacadeDecline,
};
use crate::model::TargetIdentity;
use reims_vgpu_protocol::StorageImageFormat;

pub use reims_vgpu_core::{
    ExecutionPort, ExecutorCapabilities, GuestWriteReach, GuestWriteService, ResidentContent,
    ResidentContentBacking, ResidentLease, ResidentService, SubmissionContext, TargetReadback,
};

impl ResidentLease for crate::backend::vulkan::engine::ResidentResourceLease {
    fn matches(&self, identity: &TargetIdentity) -> bool {
        self.matches(identity)
    }

    fn backing(&self) -> ResidentContentBacking {
        self.backing()
    }
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

/// Snapshot the active protocol context before entering a backend call.
pub fn context_for(state: &crate::model::DeviceState, task_id: u32) -> SubmissionContext {
    state
        .active_submission
        .clone()
        .unwrap_or_else(|| SubmissionContext::standalone(task_id))
}

pub type ResolvedSubmission =
    reims_vgpu_core::ResolvedSubmission<Box<DrawRequest>, Box<ComputeRequest>>;
pub type ExecutionOutput = reims_vgpu_core::ExecutionOutput<DrawOutput, ComputeOutput>;
pub type ExecutionCompletion = reims_vgpu_core::ExecutionCompletion<ExecutionOutput>;
pub type ExecutionReceipt<T> = reims_vgpu_core::ExecutionReceipt<T>;

/// Backend execution contract implemented per device.
pub trait Executor:
    ExecutionPort<
        Submission = ResolvedSubmission,
        Completion = ExecutionCompletion,
        Error = DrawError,
    > + ResidentService
    + GuestWriteService
{
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

    fn read_target(&self, _identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
        Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "target_readback",
            },
        ))
    }

    fn read_target_leased(
        &self,
        _identity: &TargetIdentity,
    ) -> Result<Option<crate::backend::vulkan::engine::LeasedFrame>, DrawError> {
        Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "leased_target_readback",
            },
        ))
    }

    fn copy_target_to_guest_pages(
        &self,
        _identity: &TargetIdentity,
        _target: &crate::backend::vulkan::engine::GuestPageTarget,
        _pages: &[u64],
    ) -> Result<(), DrawError> {
        Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "target_to_guest_pages",
            },
        ))
    }

    fn read_resident_bgra(&self, _identity: &TargetIdentity, _need: usize) -> Option<Vec<u8>> {
        None
    }

    fn guest_access_outstanding(&self) -> bool {
        false
    }

    fn completion_stamp_pending(&self, _index: u32) -> bool {
        false
    }

    fn submit_batch_for_waiting_stamp(&self, _index: u32) -> bool {
        false
    }

    fn write_completion_stamp(
        &self,
        _guest_ref: &crate::runtime::guest_ram::GuestRef,
        _index: u32,
        _value: u32,
    ) -> Result<(), DrawError> {
        Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "completion_stamp",
            },
        ))
    }

    fn quiesce_completion_stamps(&self, _index: u32) {}

    fn quiesce_guest_reads(&self) {}

    fn warm_guest_ram_imports(
        &self,
        _imports: &[std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>],
    ) -> (usize, u64) {
        (0, 0)
    }

    fn retire_guest_import(&self, _import: crate::runtime::guest_ram::ImportId) {}

    fn resident_presentable(&self, _identity: &TargetIdentity, _width: u32, _height: u32) -> bool {
        false
    }

    fn flush_batched_draws(&self) {}

    fn maintain_resources(&self, _now_ms: u64) {}

    /// Observation-only snapshots. Semantic planning must never read these.
    fn sampled_working_set_census(&self) -> Option<String> {
        None
    }

    fn buffer_gather_working_set_census(&self) -> Option<String> {
        None
    }

    fn guest_import_census(&self) -> (u64, usize, usize) {
        (0, 0, 0)
    }

    fn object_cache_levels(&self) -> [usize; 6] {
        [0; 6]
    }

    fn counter_snapshot(&self) -> crate::backend::vulkan::engine::CounterSnapshot {
        Default::default()
    }

    fn draw_phase_window(&self) -> Option<crate::backend::vulkan::engine::DrawPhaseWindow> {
        None
    }

    fn take_engine_lock_census(&self, _win_ms: u64) -> Option<String> {
        None
    }

    fn prepare_window_resident_present(
        &self,
        _identity: &TargetIdentity,
        _width: u32,
        _height: u32,
    ) -> Result<(), &'static str> {
        Err("winpub_window_not_attached")
    }

    fn window_present_attached(&self) -> bool {
        false
    }

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

    fn read_target(&self, identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
        crate::backend::vulkan::engine::read_target(identity)
    }

    fn read_target_leased(
        &self,
        identity: &TargetIdentity,
    ) -> Result<Option<crate::backend::vulkan::engine::LeasedFrame>, DrawError> {
        crate::backend::vulkan::engine::read_target_leased(identity)
    }

    fn copy_target_to_guest_pages(
        &self,
        identity: &TargetIdentity,
        target: &crate::backend::vulkan::engine::GuestPageTarget,
        pages: &[u64],
    ) -> Result<(), DrawError> {
        crate::backend::vulkan::engine::copy_target_to_guest_pages(identity, target, pages)
    }

    fn read_resident_bgra(&self, identity: &TargetIdentity, need: usize) -> Option<Vec<u8>> {
        crate::backend::vulkan::engine::read_resident_bgra(identity, need)
    }

    fn guest_access_outstanding(&self) -> bool {
        crate::backend::vulkan::engine::guest_access_outstanding()
    }

    fn completion_stamp_pending(&self, index: u32) -> bool {
        crate::backend::vulkan::engine::stamp_completion::fifo_has_pending_stamp(index)
    }

    fn submit_batch_for_waiting_stamp(&self, index: u32) -> bool {
        crate::backend::vulkan::engine::submit_batch_for_waiting_stamp(index)
    }

    fn write_completion_stamp(
        &self,
        guest_ref: &crate::runtime::guest_ram::GuestRef,
        index: u32,
        value: u32,
    ) -> Result<(), DrawError> {
        crate::backend::vulkan::engine::write_completion_stamp(guest_ref, index, value)
    }

    fn quiesce_completion_stamps(&self, index: u32) {
        crate::backend::vulkan::engine::quiesce_completion_stamps(index);
    }

    fn quiesce_guest_reads(&self) {
        crate::backend::vulkan::engine::quiesce_guest_reads();
    }

    fn warm_guest_ram_imports(
        &self,
        imports: &[std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>],
    ) -> (usize, u64) {
        crate::backend::vulkan::engine::warm_guest_ram_imports(imports)
    }

    fn retire_guest_import(&self, import: crate::runtime::guest_ram::ImportId) {
        crate::backend::vulkan::engine::retire_guest_import(import);
    }

    fn resident_presentable(&self, identity: &TargetIdentity, width: u32, height: u32) -> bool {
        crate::backend::vulkan::engine::resident_presentable(identity, width, height)
    }

    fn flush_batched_draws(&self) {
        crate::backend::vulkan::engine::flush_batched_draws();
    }

    fn maintain_resources(&self, now_ms: u64) {
        crate::backend::vulkan::engine::maintain_resources(now_ms);
    }

    fn sampled_working_set_census(&self) -> Option<String> {
        crate::backend::vulkan::engine::sampled_working_set_census()
    }

    fn buffer_gather_working_set_census(&self) -> Option<String> {
        crate::backend::vulkan::engine::buffer_gather_working_set_census()
    }

    fn guest_import_census(&self) -> (u64, usize, usize) {
        crate::backend::vulkan::engine::guest_import_census()
    }

    fn object_cache_levels(&self) -> [usize; 6] {
        crate::backend::vulkan::engine::object_cache_levels()
    }

    fn counter_snapshot(&self) -> crate::backend::vulkan::engine::CounterSnapshot {
        crate::backend::vulkan::engine::counter_snapshot()
    }

    fn draw_phase_window(&self) -> Option<crate::backend::vulkan::engine::DrawPhaseWindow> {
        crate::backend::vulkan::engine::draw_phase_window()
    }

    fn take_engine_lock_census(&self, win_ms: u64) -> Option<String> {
        crate::backend::vulkan::engine::take_engine_lock_census(win_ms)
    }

    fn prepare_window_resident_present(
        &self,
        identity: &TargetIdentity,
        width: u32,
        height: u32,
    ) -> Result<(), &'static str> {
        #[cfg(feature = "host-window")]
        return crate::backend::vulkan::engine::prepare_window_resident_present(
            identity, width, height,
        );
        #[cfg(not(feature = "host-window"))]
        {
            let _ = (identity, width, height);
            Err("winpub_window_not_attached")
        }
    }

    fn window_present_attached(&self) -> bool {
        #[cfg(feature = "host-window")]
        return crate::backend::vulkan::engine::window_present_attached();
        #[cfg(not(feature = "host-window"))]
        false
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

impl ResidentService for VulkanExecutor {
    fn resident_content_backing(&self, identity: &TargetIdentity) -> ResidentContentBacking {
        crate::backend::vulkan::engine::resident_content_backing(identity)
    }

    fn resident_absent_after_reclaim(
        &self,
        identity: &TargetIdentity,
    ) -> Option<(reims_vgpu_core::ResidentReclaim, u64)> {
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

    fn retain_resident_resource(
        &self,
        identity: &TargetIdentity,
    ) -> Option<Box<dyn ResidentLease>> {
        crate::backend::vulkan::engine::retain_resident_resource(identity)
            .map(|lease| Box::new(lease) as Box<dyn ResidentLease>)
    }
}

impl GuestWriteService for VulkanExecutor {
    fn guest_writes_outstanding(&self) -> bool {
        crate::backend::vulkan::engine::guest_writes_outstanding()
    }

    fn guest_writes_reaching(&self, pages: &[u64]) -> GuestWriteReach {
        crate::backend::vulkan::engine::guest_writes_reaching(pages)
    }

    fn quiesce_guest_writes(&self) {
        crate::backend::vulkan::engine::quiesce_guest_writes();
    }
}

impl ExecutionPort for VulkanExecutor {
    type Submission = ResolvedSubmission;
    type Completion = ExecutionCompletion;
    type Error = DrawError;

    fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
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
    let expected_kind = expected.kind().as_str();
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
                actual: other.kind().as_str(),
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
    let expected_kind = expected.kind().as_str();
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
                actual: other.kind().as_str(),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, DeviceState};
    use reims_vgpu_protocol::{
        ObjectRef, ResourceValidity, SegmentBoundary, SegmentKind, SubmissionId,
        SubmissionIdentity, SubmissionResourceUse, TaskId,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
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
        guest_writes: GuestWriteReach,
        seen: Mutex<Vec<SubmissionContext>>,
        resets: AtomicUsize,
        write_quiesces: AtomicUsize,
    }

    impl ScriptedExecutor {
        fn new(completion: ScriptedCompletion) -> Self {
            Self {
                completion,
                capabilities: ExecutorCapabilities::default(),
                resident_generation: None,
                guest_writes: GuestWriteReach::Disjoint,
                seen: Mutex::new(Vec::new()),
                resets: AtomicUsize::new(0),
                write_quiesces: AtomicUsize::new(0),
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

        fn with_guest_writes(mut self, reach: GuestWriteReach) -> Self {
            self.guest_writes = reach;
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

        fn reset(&self) {
            self.resets.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl ResidentService for ScriptedExecutor {}

    impl GuestWriteService for ScriptedExecutor {
        fn guest_writes_outstanding(&self) -> bool {
            self.guest_writes != GuestWriteReach::Disjoint
        }

        fn guest_writes_reaching(&self, _pages: &[u64]) -> GuestWriteReach {
            self.guest_writes
        }

        fn quiesce_guest_writes(&self) {
            self.write_quiesces.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl ExecutionPort for ScriptedExecutor {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
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
            segments: Arc::from([SegmentBoundary {
                stream_index: 2,
                index: 3,
                kind: SegmentKind::Render,
                continues_previous: false,
                continues_next: true,
            }]),
            segment: Some(SegmentBoundary {
                stream_index: 2,
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
    fn an_executor_without_readback_refuses_by_service_name() {
        let state = DeviceState::new_with_executor(
            DeviceId(1),
            12,
            Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw)),
        );
        let identity = TargetIdentity::Gva {
            gva: 0x8000,
            width: 16,
            height: 16,
            generation: 3,
            format: crate::contract::pixel_format::TexelLayout::Rgba8,
        };

        assert!(matches!(
            state.executor.read_target(&identity),
            Err(DrawError::Facade(
                EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "target_readback"
                }
            ))
        ));
    }

    #[test]
    fn guest_write_settlement_uses_the_injected_executor() {
        let scripted = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Draw)
                .with_guest_writes(GuestWriteReach::Overlap),
        );

        crate::runtime::render_writeback::settle_guest_writes(
            scripted.as_ref(),
            crate::runtime::render_writeback::SettleSite::CompletionStamp,
        );

        assert_eq!(scripted.write_quiesces.load(Ordering::Relaxed), 1);
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

    impl Executor for WrongIdentityExecutor {}
    impl ResidentService for WrongIdentityExecutor {}
    impl GuestWriteService for WrongIdentityExecutor {}

    impl ExecutionPort for WrongIdentityExecutor {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
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
