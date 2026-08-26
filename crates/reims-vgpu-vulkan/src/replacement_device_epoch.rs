//! Complete ownership of one replacement Vulkan device incarnation.
//!
//! This owner is constructed per vGPU session. It is the only route that
//! starts replacement queue and recording services, so no process-global
//! engine lock or legacy queue owner participates in replacement execution.

use crate::{
    engine::context::{DeviceContextStartError, SharedDeviceContext},
    replacement_capabilities::{ReplacementCapabilities, ReplacementCapabilityError},
    replacement_compute::{
        ReplacementComputePipelineFamily, ReplacementComputePipelinePlan,
        ReplacementComputePipelineVariant,
    },
    replacement_compute_compile::{
        ReplacementComputeFamilyCompileError, ReplacementComputePipelineCompileError,
        ReplacementComputePipelineCompiler,
    },
    replacement_epoch::{
        ReplacementQueueBinding, ReplacementQueueEpoch, ReplacementQueueEpochStartError,
    },
    replacement_render::{
        ReplacementRenderPipelineFamily, ReplacementRenderPipelinePlan,
        ReplacementRenderPipelineVariant,
    },
    replacement_render_compile::{
        ReplacementRenderFamilyCompileError, ReplacementRenderPipelineCompileError,
        ReplacementRenderPipelineCompiler,
    },
    replacement_sampler::ReplacementSamplerRegistry,
};
use ash::vk;
use reims_vgpu_core::{
    DescriptorCapabilities, NativeObjectLease, PipelineVariantCompileJob, ResolvedRenderPipeline,
    SessionGeneration, VulkanDeviceEpoch, VulkanDeviceEpochState,
};
use reims_vgpu_protocol::{DepthStencilDescriptor, QueueOwnerId, VulkanDeviceEpochId};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

pub use crate::engine::host_ram::HostRamDecline as ReplacementGuestImportDecline;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementGuestBufferImportError {
    DeviceLifetimeClosed,
    Bound(reims_vgpu_memory::GuestRamError),
    OffsetOverflow,
    Import(ReplacementGuestImportDecline),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementGuestImportRetirement {
    NotImported,
    Retired { representation_owners: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementGuestImportCensus {
    pub live: usize,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementStorageImageCapabilities {
    pub storage_image: bool,
    pub storage_image_atomic: bool,
    pub read_without_format: bool,
    pub write_without_format: bool,
}

struct ReplacementImportedGuestAllocation {
    context: Arc<SharedDeviceContext>,
    _import: Arc<reims_vgpu_memory::GuestRamImport>,
    native: crate::engine::host_ram::ImportedHostRam,
}

impl Drop for ReplacementImportedGuestAllocation {
    fn drop(&mut self) {
        unsafe {
            self.native.destroy(&self.context.device);
        }
    }
}

#[derive(Debug)]
pub enum ReplacementDeviceEpochStartError {
    Device(DeviceContextStartError),
    Capabilities(ReplacementCapabilityError),
    Timeline(vk::Result),
    Queues(ReplacementQueueEpochStartError),
}

/// Every native service and immutable capability belonging to one exact
/// `VkDevice` incarnation.
pub struct ReplacementDeviceEpoch {
    id: VulkanDeviceEpochId,
    lifetime: VulkanDeviceEpoch,
    work_queue: QueueOwnerId,
    work_queue_family: u32,
    capabilities: ReplacementCapabilities,
    device_info_limits: reims_vgpu_core::DeviceInfoLimits,
    compute_info_limits: reims_vgpu_core::ComputeInfoLimits,
    queues: Option<ReplacementQueueEpoch>,
    compiler: ReplacementRenderPipelineCompiler,
    compute_compiler: ReplacementComputePipelineCompiler,
    samplers: ReplacementSamplerRegistry,
    execution_limits: crate::replacement_representation::ReplacementExecutionLimits,
    guest_imports:
        Mutex<BTreeMap<reims_vgpu_memory::ImportId, Arc<ReplacementImportedGuestAllocation>>>,
    guest_import_hits: AtomicU64,
    guest_import_misses: AtomicU64,
    context: Arc<SharedDeviceContext>,
    work_timelines: Box<[vk::Semaphore]>,
    #[cfg(feature = "host-window")]
    window_presenter:
        std::sync::Mutex<Option<crate::replacement_window_present::ReplacementWindowPresenter>>,
    #[cfg(feature = "host-window")]
    window_swapchain_generations: Arc<AtomicU64>,
}

impl ReplacementDeviceEpoch {
    /// Exact enabled-device and per-format facts consumed by compute shader
    /// specialization in this Vulkan incarnation.
    pub fn runtime_storage_image_capabilities(
        &self,
        format: reims_vgpu_protocol::StorageImageFormat,
    ) -> ReplacementStorageImageCapabilities {
        let features = unsafe {
            self.context.instance.get_physical_device_format_properties(
                self.context.pd,
                crate::format::vk_storage_image(format),
            )
        }
        .optimal_tiling_features;
        ReplacementStorageImageCapabilities {
            storage_image: features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
            storage_image_atomic: features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC),
            read_without_format: self.context.spirv_storage_read_without_format,
            write_without_format: self.context.spirv_storage_write_without_format,
        }
    }

    pub fn create(
        lifetime: VulkanDeviceEpoch,
        queue: QueueOwnerId,
        recording_worker_count: usize,
    ) -> Result<Self, ReplacementDeviceEpochStartError> {
        let id = lifetime.id();
        let context = Arc::new(SharedDeviceContext::new(
            unsafe { crate::engine::context::DeviceContext::create_replacement() }
                .map_err(ReplacementDeviceEpochStartError::Device)?,
        ));
        let capabilities = ReplacementCapabilities::require(
            &context.features,
            DescriptorCapabilities {
                descriptor_buffer: false,
                push_descriptor: context.push_descriptor.is_some(),
            },
        )
        .map_err(ReplacementDeviceEpochStartError::Capabilities)?;
        let timeline =
            create_timeline(&context.device).map_err(ReplacementDeviceEpochStartError::Timeline)?;
        let flags = vk::QueueFlags::GRAPHICS
            | vk::QueueFlags::TRANSFER
            | if context.compute_capable {
                vk::QueueFlags::COMPUTE
            } else {
                vk::QueueFlags::empty()
            };
        let bindings = [ReplacementQueueBinding {
            id: queue,
            queue_family: context.gq,
            flags,
            queue: unsafe { context.device.get_device_queue(context.gq, 0) },
            timeline,
        }];
        let queues = match ReplacementQueueEpoch::start(
            id,
            capabilities,
            &context.device,
            recording_worker_count,
            bindings,
        ) {
            Ok(queues) => queues,
            Err(error) => {
                unsafe { context.device.destroy_semaphore(timeline, None) };
                return Err(ReplacementDeviceEpochStartError::Queues(error));
            }
        };
        let compiler =
            ReplacementRenderPipelineCompiler::new(Arc::clone(&context), lifetime.clone());
        let compute_compiler =
            ReplacementComputePipelineCompiler::new(Arc::clone(&context), lifetime.clone());
        let samplers = ReplacementSamplerRegistry::new(Arc::clone(&context));
        let physical_limits = unsafe {
            context
                .instance
                .get_physical_device_properties(context.pd)
                .limits
        };
        let execution_limits = crate::replacement_representation::ReplacementExecutionLimits {
            compute_fill: context.compute_capable.then_some(
                crate::replacement_buffer_blit::NativeComputeFillLimits {
                    min_storage_buffer_offset_alignment: context.storage_buffer_offset_align,
                    max_storage_buffer_range: context.max_storage_buffer_range,
                    max_compute_work_group_count_x: physical_limits.max_compute_work_group_count[0],
                },
            ),
            max_storage_buffer_range: context.max_storage_buffer_range,
            storage_buffer_offset_alignment: context.storage_buffer_offset_align,
            max_viewports: context.features.max_viewports,
            precise_occlusion_queries: context.features.occlusion_query_precise,
            null_descriptors: context.features.null_descriptor,
        };
        let device_info_limits = reims_vgpu_core::DeviceInfoLimits {
            max_sample_count: context.features.max_sample_count,
            d24_stencil8: context.features.d24_unorm_s8_attachment,
            max_threads_per_threadgroup: context.features.max_compute_workgroup_size,
            max_threadgroup_memory_bytes: context.features.max_compute_shared_memory_bytes,
            native_fp16: context.features.float16,
        };
        let compute_info_limits = reims_vgpu_core::ComputeInfoLimits {
            max_total_threads_per_threadgroup: context.features.max_compute_workgroup_invocations,
            thread_execution_width: context.features.subgroup_size,
        };
        Ok(Self {
            id,
            lifetime,
            work_queue: queue,
            work_queue_family: context.gq,
            capabilities,
            device_info_limits,
            compute_info_limits,
            queues: Some(queues),
            compiler,
            compute_compiler,
            samplers,
            execution_limits,
            guest_imports: Mutex::new(BTreeMap::new()),
            guest_import_hits: AtomicU64::new(0),
            guest_import_misses: AtomicU64::new(0),
            context,
            work_timelines: Box::new([timeline]),
            #[cfg(feature = "host-window")]
            window_presenter: std::sync::Mutex::new(None),
            #[cfg(feature = "host-window")]
            window_swapchain_generations: Arc::new(AtomicU64::new(1)),
        })
    }

    pub const fn id(&self) -> VulkanDeviceEpochId {
        self.id
    }

    /// Actual work queue installed for this epoch. Queue selection is an
    /// epoch construction fact, not a recording-adapter input.
    pub const fn work_queue(&self) -> QueueOwnerId {
        self.work_queue
    }

    pub const fn work_queue_family(&self) -> u32 {
        self.work_queue_family
    }

    pub const fn capabilities(&self) -> ReplacementCapabilities {
        self.capabilities
    }

    pub const fn device_info_limits(&self) -> reims_vgpu_core::DeviceInfoLimits {
        self.device_info_limits
    }

    pub const fn compute_info_limits(&self) -> reims_vgpu_core::ComputeInfoLimits {
        self.compute_info_limits
    }

    /// Host-reported memory facts available to semantic representation
    /// planning. Direct guest aliases and imported transfer endpoints remain
    /// false until their replacement-native constructors exist; topology and
    /// working-memory availability come directly from this physical device.
    pub fn representation_environment(
        &self,
    ) -> (
        reims_vgpu_core::HostMemoryTopology,
        reims_vgpu_core::RepresentationCapabilities,
    ) {
        let topology = match self.context.caps.memory.topology {
            crate::memory::MemoryTopology::Unified => reims_vgpu_core::HostMemoryTopology::Unified,
            crate::memory::MemoryTopology::Discrete => {
                reims_vgpu_core::HostMemoryTopology::Discrete
            }
        };
        let memory_types = &self.context.memory_properties.memory_types
            [..self.context.memory_properties.memory_type_count as usize];
        let host_visible_working = memory_types.iter().any(|memory| {
            memory
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        });
        let device_local_working = memory_types.iter().any(|memory| {
            memory
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        });
        (
            topology,
            reims_vgpu_core::RepresentationCapabilities {
                direct_guest_backing: false,
                imported_transfer: self.context.caps.host_pointer.rung.is_available(),
                host_visible_working,
                device_local_working,
            },
        )
    }

    pub fn guest_import_alignment(&self) -> Option<u64> {
        self.context
            .caps
            .host_pointer
            .rung
            .is_available()
            .then_some(self.context.caps.host_pointer.min_alignment)
    }

    /// End one guest allocation identity and remove the epoch registry's
    /// ownership. Any already-submitted representation keeps the imported
    /// parent alive through its own timeline retirement; no later
    /// representation can recreate this identity because `GuestRamImport` is
    /// marked retired before the registry entry is removed.
    pub fn retire_guest_import(
        &self,
        import: &Arc<reims_vgpu_memory::GuestRamImport>,
    ) -> ReplacementGuestImportRetirement {
        import.retire();
        let allocation = self
            .guest_imports
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&import.id());
        let Some(allocation) = allocation else {
            return ReplacementGuestImportRetirement::NotImported;
        };
        let representation_owners = Arc::strong_count(&allocation).saturating_sub(1);
        drop(allocation);
        ReplacementGuestImportRetirement::Retired {
            representation_owners,
        }
    }

    pub fn guest_import_census(&self) -> ReplacementGuestImportCensus {
        ReplacementGuestImportCensus {
            live: self
                .guest_imports
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len(),
            hits: self.guest_import_hits.load(Ordering::Relaxed),
            misses: self.guest_import_misses.load(Ordering::Relaxed),
        }
    }

    /// Acquire the dual semantic/native lifetime attached to newly accepted
    /// work. The returned lease remains owned by accepted work through normal
    /// guest reset, but becomes unusable as soon as this Vulkan epoch loses.
    pub fn acquire_native_object(
        &self,
        generation: &SessionGeneration,
    ) -> Option<NativeObjectLease> {
        NativeObjectLease::acquire(generation, &self.lifetime)
    }

    /// Atomically close native-handle admission after the first device-loss
    /// result. Queue teardown is owned by `Drop`; this transition prevents any
    /// later pipeline or operation lease from entering the lost incarnation.
    pub fn begin_loss(&self) -> bool {
        self.lifetime.begin_loss()
    }

    pub fn state(&self) -> VulkanDeviceEpochState {
        self.lifetime.state()
    }

    pub fn queues(&self) -> &ReplacementQueueEpoch {
        self.queues
            .as_ref()
            .expect("replacement queue epoch exists until device-epoch teardown")
    }

    /// Borrow the queue epoch only through the owning device incarnation.
    /// Composition uses this to join driver acceptance to the exact queues
    /// whose timelines and recording workers produced the submission.
    pub fn queues_mut(&mut self) -> &mut ReplacementQueueEpoch {
        self.queues
            .as_mut()
            .expect("replacement queue epoch exists until device-epoch teardown")
    }

    pub fn platform_reset_wait_idle(
        &self,
    ) -> Result<(), crate::replacement_queue::ReplacementQueueError> {
        self.queues()
            .lane(self.work_queue)
            .expect("the replacement epoch always owns its work queue lane")
            .submit
            .wait_idle()
    }

    pub fn platform_reset_recording_workers(
        &mut self,
    ) -> Result<(), reims_vgpu_core::FixedExecutorError> {
        let context = Arc::clone(&self.context);
        let capabilities = self.capabilities;
        self.queues_mut()
            .reset_recording_workers(&context.device, capabilities)
    }

    /// Stop and join every worker owned by this lost incarnation. Admission
    /// must already be closed through [`Self::begin_loss`].
    pub fn terminate_workers_after_loss(&mut self) {
        debug_assert_ne!(self.state(), VulkanDeviceEpochState::Active);
        self.queues_mut().terminate_workers_after_loss();
    }

    pub fn null_descriptors(&self) -> bool {
        self.context.features.null_descriptor
    }

    /// Allocate and record one transaction-owned BGRA frame for the QEMU
    /// console endpoint.
    ///
    /// # Safety
    ///
    /// The source and transition handles must belong to this exact live epoch
    /// and remain valid through queue acceptance.
    pub unsafe fn prepare_console_present(
        &self,
        source: crate::replacement_image_transition::NativeImageTarget,
        transitions: &crate::replacement_image_transition::NativeImageUseTransitions,
    ) -> Result<
        crate::replacement_console_present::ReplacementPreparedConsolePresent,
        crate::replacement_console_present::ReplacementConsolePresentError,
    > {
        unsafe {
            crate::replacement_console_present::prepare_console_present(
                Arc::clone(&self.context),
                source,
                transitions,
            )
        }
    }

    #[cfg(feature = "host-window")]
    /// Attach the native window to this exact device epoch.
    ///
    /// # Safety
    ///
    /// Both raw handles must identify the same live native window and remain
    /// valid until `detach_window` or epoch teardown.
    pub unsafe fn attach_window(
        &self,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<(), crate::replacement_window_present::ReplacementWindowAttachError> {
        use crate::replacement_window_present::ReplacementWindowAttachError;
        let mut presenter = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if presenter.is_some() {
            return Err(ReplacementWindowAttachError::AlreadyAttached);
        }
        *presenter = Some(
            unsafe {
                crate::replacement_window_present::ReplacementWindowPresenter::create(
                    &self.context,
                    display,
                    window,
                    width,
                    height,
                    Arc::clone(&self.window_swapchain_generations),
                )
            }
            .map_err(ReplacementWindowAttachError::Window)?,
        );
        Ok(())
    }

    #[cfg(feature = "host-window")]
    pub fn resize_window(&self, width: u32, height: u32) {
        if let Some(presenter) = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            presenter.resize(width, height);
        }
    }

    #[cfg(feature = "host-window")]
    pub fn window_snapshot(
        &self,
    ) -> Option<crate::replacement_window_present::ReplacementWindowSnapshot> {
        self.window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(crate::replacement_window_present::ReplacementWindowPresenter::snapshot)
    }

    #[cfg(feature = "host-window")]
    /// Acquire and record one replacement presentation blit.
    ///
    /// # Safety
    ///
    /// `source` and every barrier handle must belong to this exact live device
    /// epoch and remain alive through the returned submission's completion.
    pub unsafe fn prepare_window_present(
        &self,
        source: crate::replacement_image_transition::NativeImageTarget,
        transitions: &crate::replacement_image_transition::NativeImageUseTransitions,
    ) -> Result<
        crate::replacement_window_present::ReplacementWindowPresentDispatch,
        crate::replacement_window_present::ReplacementWindowPresentPrepareError,
    > {
        use crate::replacement_window_present::ReplacementWindowPresentPrepareError;
        let mut presenter = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(presenter) = presenter.as_mut() else {
            return Err(ReplacementWindowPresentPrepareError::NotAttached);
        };
        unsafe { presenter.prepare(&self.context, source, transitions) }
    }

    #[cfg(feature = "host-window")]
    pub fn accept_window_present(
        &self,
        slot: usize,
        acquire_suboptimal: bool,
        present_result: Result<bool, vk::Result>,
    ) -> Result<
        crate::replacement_window_present::ReplacementWindowPresentOutcome,
        crate::replacement_window_present::ReplacementWindowPresentStateError,
    > {
        let mut presenter = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(presenter) = presenter.as_mut() else {
            return Err(
                crate::replacement_window_present::ReplacementWindowPresentStateError::SlotAbsent,
            );
        };
        presenter.accept(slot, acquire_suboptimal, present_result)
    }

    #[cfg(feature = "host-window")]
    pub fn abandon_window_present(
        &self,
        slot: usize,
    ) -> Result<(), crate::replacement_window_present::ReplacementWindowPresentStateError> {
        let mut presenter = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(presenter) = presenter.as_mut() else {
            return Err(
                crate::replacement_window_present::ReplacementWindowPresentStateError::SlotAbsent,
            );
        };
        presenter.abandon(slot)
    }

    #[cfg(feature = "host-window")]
    /// Detach and destroy the native window presentation objects.
    ///
    /// # Safety
    ///
    /// The native window handles supplied at attach must still be valid.
    pub unsafe fn detach_window(
        &self,
    ) -> Result<(), crate::replacement_window_present::ReplacementWindowDetachError> {
        use crate::replacement_window_present::ReplacementWindowDetachError;
        let mut presenter = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(mut presenter) = presenter.take() else {
            return Ok(());
        };
        self.queues()
            .lane(self.work_queue)
            .expect("the replacement epoch always owns its work queue lane")
            .submit
            .wait_idle()
            .map_err(ReplacementWindowDetachError::Queue)?;
        unsafe { presenter.destroy_after_idle(&self.context) };
        Ok(())
    }

    pub const fn samplers(&self) -> &ReplacementSamplerRegistry {
        &self.samplers
    }

    pub fn execution_resolver<'a>(
        &'a self,
        resources: &'a reims_vgpu_core::ResourceLifecycleOwner<
            crate::replacement_representation::ReplacementNativeRepresentation,
        >,
        images: &'a crate::replacement_image_state::ReplacementImageStateOwner,
    ) -> crate::replacement_representation::ReplacementExecutionResolver<'a> {
        crate::replacement_representation::ReplacementExecutionResolver::new(
            resources,
            images,
            &self.samplers,
            self.execution_limits,
        )
    }

    /// Assemble all native operation programs while the Vulkan incarnation's
    /// publication gate is active. Indirect-range staging allocation therefore
    /// cannot race the loss transition or bypass the epoch that owns it.
    pub fn resolve_prepared_exec_recording<Completion: Clone + PartialEq>(
        &self,
        input: crate::replacement_recording::ReplacementRecordingInput<
            crate::replacement_exec_recording::CanonicalReplacementOperation<Completion>,
        >,
        resources: &reims_vgpu_core::PreparedExecResources<
            reims_vgpu_core::ResolvedComputeDispatch,
            crate::replacement_compute::ReplacementComputePipelineVariant,
            reims_vgpu_core::ResolvedRenderDispatch,
            crate::replacement_render::ReplacementRenderPipelineVariant,
        >,
        image_states: Option<&crate::replacement_image_state::PreparedImageStateBatch>,
        semantics: crate::replacement_exec_recording::ReplacementExecSemanticAdmissions<
            '_,
            Completion,
        >,
        resolver: &crate::replacement_representation::ReplacementExecutionResolver<'_>,
    ) -> Result<
        crate::replacement_recording::ReplacementRecordingRequest<
            crate::replacement_exec_recording::CanonicalReplacementOperation<Completion>,
        >,
        Box<
            crate::replacement_exec_recording::ReplacementExecProgramFailure<
                crate::replacement_exec_recording::CanonicalReplacementOperation<Completion>,
            >,
        >,
    > {
        match self.lifetime.with_active(input, |input| {
            let context: Arc<dyn crate::replacement_indirect_range::ReplacementIndirectRangeDevice> =
                self.context.clone();
            crate::replacement_exec_recording::resolve_prepared_exec_recording(
                input,
                resources,
                image_states,
                semantics,
                Some(context),
                resolver,
            )
        }) {
            Ok(result) => result,
            Err(input) => Err(Box::new(
                crate::replacement_exec_recording::ReplacementExecProgramFailure {
                    reason: crate::replacement_exec_recording::ReplacementExecProgramError::DeviceLifetimeClosed,
                    input,
                },
            )),
        }
    }

    /// Assemble a resumed indirect phase without renumbering the suffix that
    /// follows its retired range readback.
    #[allow(
        clippy::too_many_arguments,
        reason = "each origin carries distinct semantic and expanded continuation identities"
    )]
    pub fn resolve_prepared_exec_continuation_recording<Completion: Clone + PartialEq>(
        &self,
        input: crate::replacement_recording::ReplacementRecordingInput<
            crate::replacement_exec_recording::CanonicalReplacementOperation<Completion>,
        >,
        resources: &reims_vgpu_core::PreparedExecResources<
            reims_vgpu_core::ResolvedComputeDispatch,
            crate::replacement_compute::ReplacementComputePipelineVariant,
            reims_vgpu_core::ResolvedRenderDispatch,
            crate::replacement_render::ReplacementRenderPipelineVariant,
        >,
        image_states: Option<&crate::replacement_image_state::PreparedImageStateBatch>,
        semantics: crate::replacement_exec_recording::ReplacementExecSemanticAdmissions<
            '_,
            Completion,
        >,
        resolver: &crate::replacement_representation::ReplacementExecutionResolver<'_>,
        operation_origins: &[reims_vgpu_core::ExpandedIndirectOperationOrigin],
    ) -> Result<
        crate::replacement_recording::ReplacementRecordingRequest<
            crate::replacement_exec_recording::CanonicalReplacementOperation<Completion>,
        >,
        Box<
            crate::replacement_exec_recording::ReplacementExecProgramFailure<
                crate::replacement_exec_recording::CanonicalReplacementOperation<Completion>,
            >,
        >,
    > {
        match self.lifetime.with_active(input, |input| {
            let context: Arc<dyn crate::replacement_indirect_range::ReplacementIndirectRangeDevice> =
                self.context.clone();
            crate::replacement_exec_recording::resolve_prepared_exec_continuation_recording_at_positions(
                input,
                resources,
                image_states,
                semantics,
                Some(context),
                resolver,
                operation_origins,
            )
        }) {
            Ok(result) => result,
            Err(input) => Err(Box::new(
                crate::replacement_exec_recording::ReplacementExecProgramFailure {
                    reason: crate::replacement_exec_recording::ReplacementExecProgramError::DeviceLifetimeClosed,
                    input,
                },
            )),
        }
    }

    /// Clone immutable compiler services into the session's fixed CPU worker
    /// population. Their shared device context remains subordinate to this
    /// epoch because the session joins those workers before dropping the epoch.
    pub fn pipeline_compilers(
        &self,
    ) -> (
        ReplacementRenderPipelineCompiler,
        ReplacementComputePipelineCompiler,
    ) {
        (self.compiler.clone(), self.compute_compiler.clone())
    }

    /// Adopt a buffer and its allocation into the canonical backing owner.
    ///
    /// # Safety
    ///
    /// The handles must belong to this exact device incarnation, `memory`
    /// must be the allocation bound to `target.buffer`, and no other owner may
    /// destroy or free either handle after this call.
    pub unsafe fn adopt_owned_buffer(
        &self,
        target: crate::replacement_buffer_blit::NativeBufferTarget,
        memory: vk::DeviceMemory,
        queue_families: Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementAdoptionFailure<(
            crate::replacement_buffer_blit::NativeBufferTarget,
            vk::DeviceMemory,
            Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
        )>,
    > {
        self.lifetime
            .with_active((target, memory, queue_families), |(target, memory, queue_families)| {
                let device: Arc<
                    dyn crate::replacement_representation::ReplacementRepresentationDevice,
                > = self.context.clone();
                crate::replacement_representation::owned_buffer(
                    device,
                    target,
                    memory,
                    queue_families,
                )
            })
            .map_err(|input| crate::replacement_representation::ReplacementAdoptionFailure {
                reason: crate::replacement_representation::ReplacementAdoptionError::DeviceLifetimeClosed,
                input,
            })
    }

    pub fn create_owned_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_class: crate::memory::MemoryClass,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementBufferAllocationError,
    > {
        self.lifetime
            .with_active((size, usage, memory_class), |(size, usage, memory_class)| {
                crate::replacement_representation::allocate_owned_buffer(
                    self.context.clone(),
                    size,
                    usage,
                    memory_class,
                )
            })
            .unwrap_or(Err(
                crate::replacement_representation::ReplacementBufferAllocationError::DeviceLifetimeClosed,
            ))
    }

    /// Allocate the contract-unrestricted native representation of a Metal
    /// buffer. Unlike textures, buffer construction carries no usage mask, so
    /// every operation class supported by the replacement recorder is a legal
    /// use of the same object lifetime.
    pub fn create_owned_working_buffer(
        &self,
        size: u64,
        memory: reims_vgpu_core::WorkingMemoryClass,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementBufferAllocationError,
    > {
        let memory = match memory {
            reims_vgpu_core::WorkingMemoryClass::HostVisible => crate::memory::MemoryClass::Upload,
            reims_vgpu_core::WorkingMemoryClass::DeviceLocal => {
                crate::memory::MemoryClass::DeviceLocal
            }
        };
        self.create_owned_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER,
            memory,
        )
    }

    pub fn create_host_staging_buffer(
        &self,
        size: u64,
        guest: reims_vgpu_memory::GuestRef,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementBufferAllocationError,
    > {
        self.lifetime
            .with_active((size, guest), |(size, guest)| {
                crate::replacement_representation::allocate_host_staging_buffer(
                    self.context.clone(),
                    size,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    reims_vgpu_memory::GuestWindow::contiguous(guest),
                )
            })
            .unwrap_or(Err(
                crate::replacement_representation::ReplacementBufferAllocationError::DeviceLifetimeClosed,
            ))
    }

    pub fn create_host_staging_buffer_for_window(
        &self,
        size: u64,
        guest: reims_vgpu_memory::GuestWindow,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementBufferAllocationError,
    > {
        self.lifetime
            .with_active((size, guest), |(size, guest)| {
                crate::replacement_representation::allocate_host_staging_buffer(
                    self.context.clone(),
                    size,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    guest,
                )
            })
            .unwrap_or(Err(
                crate::replacement_representation::ReplacementBufferAllocationError::DeviceLifetimeClosed,
            ))
    }

    /// Import one contract-bounded guest allocation once for this Vulkan
    /// epoch and return an exact buffer window into it. The epoch registry is
    /// keyed by the allocation's non-reusable identity; representations share
    /// the parent import and cannot independently import overlapping slices.
    pub fn create_imported_guest_buffer(
        &self,
        guest: &reims_vgpu_memory::GuestRef,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        ReplacementGuestBufferImportError,
    > {
        let bound = guest
            .bound()
            .map_err(ReplacementGuestBufferImportError::Bound)?;
        let base_offset = bound
            .offset
            .checked_add(guest.head())
            .ok_or(ReplacementGuestBufferImportError::OffsetOverflow)?;
        let accessible_size = bound
            .len
            .checked_sub(guest.head())
            .ok_or(ReplacementGuestBufferImportError::OffsetOverflow)?;
        let import = guest.import();
        let input = (
            Arc::clone(import),
            base_offset,
            accessible_size,
            guest.requested(),
        );
        self.lifetime
            .with_active(
                input,
                |(import, base_offset, accessible_size, requested)| {
                    let mut imports = self
                        .guest_imports
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if import.is_retired() {
                        return Err(ReplacementGuestBufferImportError::Import(
                            ReplacementGuestImportDecline::Retired {
                                import_id: import.id().get(),
                            },
                        ));
                    }
                    let allocation = if let Some(allocation) = imports.get(&import.id()) {
                        self.guest_import_hits.fetch_add(1, Ordering::Relaxed);
                        Arc::clone(allocation)
                    } else {
                        self.guest_import_misses.fetch_add(1, Ordering::Relaxed);
                        let native = unsafe {
                            crate::engine::host_ram::import_host_allocation(&self.context, &import)
                        }
                        .map_err(ReplacementGuestBufferImportError::Import)?;
                        let allocation = Arc::new(ReplacementImportedGuestAllocation {
                            context: Arc::clone(&self.context),
                            _import: Arc::clone(&import),
                            native,
                        });
                        imports.insert(import.id(), Arc::clone(&allocation));
                        allocation
                    };
                    let owner: Arc<dyn std::any::Any + Send + Sync> = allocation.clone();
                    let device: Arc<
                        dyn crate::replacement_representation::ReplacementRepresentationDevice,
                    > = self.context.clone();
                    Ok(unsafe {
                        crate::replacement_representation::imported_guest_buffer(
                            device,
                            crate::replacement_buffer_blit::NativeBufferTarget {
                                buffer: allocation.native.buffer,
                                base_offset,
                                accessible_size,
                                size: requested,
                                usage: crate::host_pointer::GUEST_IMPORT_USAGE,
                            },
                            owner,
                        )
                    })
                },
            )
            .unwrap_or(Err(ReplacementGuestBufferImportError::DeviceLifetimeClosed))
    }

    pub fn create_owned_image(
        &self,
        plan: crate::replacement_representation::ReplacementImageCreatePlan,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementImageAllocationError,
    > {
        self.lifetime
            .with_active(plan, |plan| {
                crate::replacement_representation::allocate_owned_image(
                    self.context.clone(),
                    plan,
                )
            })
            .unwrap_or(Err(
                crate::replacement_representation::ReplacementImageAllocationError::DeviceLifetimeClosed,
            ))
    }

    pub fn create_owned_texture(
        &self,
        declaration: reims_vgpu_protocol::TextureDeclaration,
        attachment_views: Box<[crate::replacement_representation::ReplacementAttachmentViewPlan]>,
        shader_views: Box<[reims_vgpu_core::ResolvedTextureBindingView]>,
        memory: reims_vgpu_core::WorkingMemoryClass,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementTextureAllocationError,
    > {
        let translated =
            crate::translate::pixel::translate(declaration.pixel_format).map_err(|reason| {
                crate::replacement_representation::ReplacementTextureAllocationError::Plan(
                    crate::replacement_representation::ReplacementTexturePlanError::Format(reason),
                )
            })?;
        let available = unsafe {
            self.context
                .instance
                .get_physical_device_format_properties(self.context.pd, translated.vk)
                .optimal_tiling_features
        };
        let memory = match memory {
            reims_vgpu_core::WorkingMemoryClass::HostVisible => crate::memory::MemoryClass::Upload,
            reims_vgpu_core::WorkingMemoryClass::DeviceLocal => {
                crate::memory::MemoryClass::DeviceLocal
            }
        };
        let plan = crate::replacement_representation::plan_owned_texture(
            declaration,
            memory,
            available,
            attachment_views,
            shader_views,
        )
        .map_err(crate::replacement_representation::ReplacementTextureAllocationError::Plan)?;
        let limits = unsafe {
            self.context
                .instance
                .get_physical_device_image_format_properties(
                    self.context.pd,
                    plan.format,
                    plan.image_type,
                    plan.tiling,
                    plan.usage,
                    plan.flags,
                )
        }
        .map_err(
            crate::replacement_representation::ReplacementTextureAllocationError::FormatQuery,
        )?;
        crate::replacement_representation::validate_image_format_limits(&plan, limits).map_err(
            crate::replacement_representation::ReplacementTextureAllocationError::Limits,
        )?;
        self.create_owned_image(plan).map_err(
            crate::replacement_representation::ReplacementTextureAllocationError::Allocation,
        )
    }

    /// Query the exact Vulkan allocation geometry for a decoded texture
    /// declaration without publishing a resource or allocating memory.
    pub fn texture_size_and_alignment(
        &self,
        declaration: reims_vgpu_protocol::TextureDeclaration,
    ) -> Result<
        reims_vgpu_core::HeapTextureSizeAndAlignInfo,
        crate::replacement_representation::ReplacementTextureRequirementsError,
    > {
        self.lifetime
            .with_active(declaration, |declaration| {
                self.active_texture_size_and_alignment(declaration)
            })
            .unwrap_or(Err(
                crate::replacement_representation::ReplacementTextureRequirementsError::DeviceLifetimeClosed,
            ))
    }

    fn active_texture_size_and_alignment(
        &self,
        declaration: reims_vgpu_protocol::TextureDeclaration,
    ) -> Result<
        reims_vgpu_core::HeapTextureSizeAndAlignInfo,
        crate::replacement_representation::ReplacementTextureRequirementsError,
    > {
        use crate::replacement_representation::ReplacementTextureRequirementsError as Error;

        let translated =
            crate::translate::pixel::translate(declaration.pixel_format).map_err(|reason| {
                Error::Plan(
                    crate::replacement_representation::ReplacementTexturePlanError::Format(reason),
                )
            })?;
        let available = unsafe {
            self.context
                .instance
                .get_physical_device_format_properties(self.context.pd, translated.vk)
                .optimal_tiling_features
        };
        let plan = crate::replacement_representation::plan_owned_texture(
            declaration,
            crate::memory::MemoryClass::DeviceLocal,
            available,
            Box::new([]),
            Box::new([]),
        )
        .map_err(Error::Plan)?;
        let limits = unsafe {
            self.context
                .instance
                .get_physical_device_image_format_properties(
                    self.context.pd,
                    plan.format,
                    plan.image_type,
                    plan.tiling,
                    plan.usage,
                    plan.flags,
                )
        }
        .map_err(Error::Query)?;
        crate::replacement_representation::validate_image_format_limits(&plan, limits)
            .map_err(Error::FormatLimits)?;
        let image = unsafe {
            self.context.device.create_image(
                &vk::ImageCreateInfo::default()
                    .flags(plan.flags)
                    .image_type(plan.image_type)
                    .format(plan.format)
                    .extent(plan.extent)
                    .mip_levels(plan.mip_levels)
                    .array_layers(plan.array_layers)
                    .samples(plan.samples)
                    .tiling(plan.tiling)
                    .usage(plan.usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(Error::Create)?;
        let requirements = unsafe { self.context.device.get_image_memory_requirements(image) };
        unsafe { self.context.device.destroy_image(image, None) };
        if requirements.size == 0 || requirements.alignment == 0 {
            return Err(Error::ZeroRequirement);
        }
        Ok(reims_vgpu_core::HeapTextureSizeAndAlignInfo {
            size: requirements.size,
            align: requirements.alignment,
        })
    }

    pub fn install_texture_view(
        &self,
        native: &mut crate::replacement_representation::ReplacementNativeRepresentation,
        view: reims_vgpu_core::ResolvedTextureBindingView,
    ) -> Result<(), crate::replacement_representation::ReplacementShaderViewInstallError> {
        self.lifetime
            .with_active((native, view), |(native, view)| {
                native.install_shader_view(&self.context, view)
            })
            .unwrap_or(Err(
                crate::replacement_representation::ReplacementShaderViewInstallError::DeviceLifetimeClosed,
            ))
    }

    /// Adopt an externally allocated buffer while retaining its allocation
    /// owner through the canonical representation lifetime.
    ///
    /// # Safety
    ///
    /// `target.buffer` must belong to this exact device incarnation, and
    /// `owner` must keep its bound allocation valid until it is dropped.
    pub unsafe fn adopt_external_buffer<Owner: std::any::Any + Send + Sync>(
        &self,
        target: crate::replacement_buffer_blit::NativeBufferTarget,
        owner: Owner,
        queue_families: Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementAdoptionFailure<(
            crate::replacement_buffer_blit::NativeBufferTarget,
            Owner,
            Option<crate::replacement_barrier_record::QueueFamilyTransfer>,
        )>,
    > {
        self.lifetime
            .with_active((target, owner, queue_families), |(target, owner, queue_families)| {
                let device: Arc<
                    dyn crate::replacement_representation::ReplacementRepresentationDevice,
                > = self.context.clone();
                crate::replacement_representation::external_buffer(
                    device,
                    target,
                    Box::new(owner),
                    queue_families,
                )
            })
            .map_err(|input| crate::replacement_representation::ReplacementAdoptionFailure {
                reason: crate::replacement_representation::ReplacementAdoptionError::DeviceLifetimeClosed,
                input,
            })
    }

    /// Adopt an image, its default view, and its allocation into the canonical
    /// backing owner.
    ///
    /// # Safety
    ///
    /// Every handle must belong to this exact device incarnation, `memory`
    /// must be bound to `target.image`, and no other owner may destroy them.
    pub unsafe fn adopt_owned_image(
        &self,
        target: crate::replacement_image_transition::NativeImageTarget,
        memory: vk::DeviceMemory,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementAdoptionFailure<(
            crate::replacement_image_transition::NativeImageTarget,
            vk::DeviceMemory,
        )>,
    > {
        self.lifetime
            .with_active((target, memory), |(target, memory)| {
                let device: Arc<
                    dyn crate::replacement_representation::ReplacementRepresentationDevice,
                > = self.context.clone();
                crate::replacement_representation::owned_image(device, target, memory)
            })
            .map_err(|input| crate::replacement_representation::ReplacementAdoptionFailure {
                reason: crate::replacement_representation::ReplacementAdoptionError::DeviceLifetimeClosed,
                input,
            })
    }

    /// Adopt an externally allocated image and default view while retaining
    /// its allocation owner through the canonical representation lifetime.
    ///
    /// # Safety
    ///
    /// The image and view must belong to this exact device incarnation, and
    /// `owner` must keep their bound allocation valid until it is dropped.
    pub unsafe fn adopt_external_image<Owner: std::any::Any + Send + Sync>(
        &self,
        target: crate::replacement_image_transition::NativeImageTarget,
        owner: Owner,
    ) -> Result<
        crate::replacement_representation::ReplacementNativeRepresentation,
        crate::replacement_representation::ReplacementAdoptionFailure<(
            crate::replacement_image_transition::NativeImageTarget,
            Owner,
        )>,
    > {
        self.lifetime
            .with_active((target, owner), |(target, owner)| {
                let device: Arc<
                    dyn crate::replacement_representation::ReplacementRepresentationDevice,
                > = self.context.clone();
                crate::replacement_representation::external_image(
                    device,
                    target,
                    Box::new(owner),
                )
            })
            .map_err(|input| crate::replacement_representation::ReplacementAdoptionFailure {
                reason: crate::replacement_representation::ReplacementAdoptionError::DeviceLifetimeClosed,
                input,
            })
    }

    pub fn compile_render_family_job(
        &self,
        family: &ReplacementRenderPipelineFamily<ReplacementRenderPipelineCompileError>,
        job: PipelineVariantCompileJob<
            crate::replacement_render::ReplacementRenderPipelineVariantKey,
        >,
        semantic: &ResolvedRenderPipeline,
        plan: ReplacementRenderPipelinePlan,
        depth_stencil: Option<&DepthStencilDescriptor>,
    ) -> Result<
        reims_vgpu_core::PipelineVariantPublication<ReplacementRenderPipelineVariant>,
        ReplacementRenderFamilyCompileError,
    > {
        self.compiler
            .compile_family_job(family, job, semantic, plan, depth_stencil)
    }

    pub fn compile_compute_family_job(
        &self,
        family: &ReplacementComputePipelineFamily<ReplacementComputePipelineCompileError>,
        job: PipelineVariantCompileJob<
            crate::replacement_compute::ReplacementComputePipelineVariantKey,
        >,
        plan: ReplacementComputePipelinePlan,
    ) -> Result<
        reims_vgpu_core::PipelineVariantPublication<ReplacementComputePipelineVariant>,
        ReplacementComputeFamilyCompileError,
    > {
        self.compute_compiler.compile_family_job(family, job, plan)
    }
}

impl Drop for ReplacementDeviceEpoch {
    fn drop(&mut self) {
        self.lifetime.begin_retirement();
        #[cfg(feature = "host-window")]
        if let Some(mut presenter) = self
            .window_presenter
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let idle = self
                .queues()
                .lane(self.work_queue)
                .expect("the replacement epoch always owns its work queue lane")
                .submit
                .wait_idle();
            if let Err(error) = idle {
                reims_vgpu_observe::fail(format!(
                    "replacement_window_teardown reason=queue_idle_refused error={error:?}"
                ));
            }
            unsafe { presenter.destroy_after_idle(&self.context) };
        }
        drop(self.queues.take());
        unsafe {
            for timeline in &self.work_timelines {
                self.context.device.destroy_semaphore(*timeline, None);
            }
        }
        self.lifetime.finish_retirement();
    }
}

fn create_timeline(device: &ash::Device) -> Result<vk::Semaphore, vk::Result> {
    let mut timeline = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let create = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
    unsafe { device.create_semaphore(&create, None) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_epoch_owns_device_queue_workers_and_render_compiler_together() {
        let session = reims_vgpu_core::DeviceSession::new(
            reims_vgpu_protocol::SessionId::new(5),
            reims_vgpu_protocol::SessionGenerationId::new(3),
            VulkanDeviceEpochId::new(41),
        );
        let epoch =
            match ReplacementDeviceEpoch::create(session.vulkan_epoch(), QueueOwnerId::new(7), 2) {
                Ok(epoch) => epoch,
                Err(ReplacementDeviceEpochStartError::Device(error)) => {
                    eprintln!("skipping Vulkan device-epoch test: {error:?}");
                    return;
                }
                Err(error) => {
                    panic!("replacement device epoch failed after device creation: {error:?}")
                }
            };
        assert_eq!(epoch.id(), VulkanDeviceEpochId::new(41));
        assert_eq!(epoch.queues().epoch(), VulkanDeviceEpochId::new(41));
        assert!(epoch.queues().lane(QueueOwnerId::new(7)).is_some());
        assert_eq!(epoch.queues().recording_workers().worker_count(), 2);
        assert!(epoch.capabilities().timeline_semaphore());
        let requirements = epoch
            .texture_size_and_alignment(reims_vgpu_protocol::TextureDeclaration {
                texture_type: reims_vgpu_protocol::TextureType::D2,
                framebuffer_only: false,
                is_drawable: false,
                write_swizzle_enabled: None,
                allow_gpu_optimized_contents: false,
                usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                width: 4,
                height: 4,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                resource_options: 0,
                protection_options: 0,
                swizzle: None,
            })
            .expect("the live epoch answers exact texture allocation geometry");
        assert!(requirements.size >= 64);
        assert!(requirements.align.is_power_of_two());
        let buffer = epoch
            .create_owned_buffer(
                4096,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                crate::memory::MemoryClass::DeviceLocal,
            )
            .expect("replacement epoch allocates an owned working buffer");
        assert_ne!(buffer.buffer().unwrap().buffer, vk::Buffer::null());
        let image = epoch
            .create_owned_image(
                crate::replacement_representation::ReplacementImageCreatePlan {
                    flags: vk::ImageCreateFlags::empty(),
                    image_type: vk::ImageType::TYPE_2D,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    components: vk::ComponentMapping::default(),
                    extent: vk::Extent3D {
                        width: 4,
                        height: 4,
                        depth: 1,
                    },
                    mip_levels: 2,
                    array_layers: 2,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    full_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 2,
                        base_array_layer: 0,
                        layer_count: 2,
                    },
                    pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                    memory_class: crate::memory::MemoryClass::DeviceLocal,
                    attachment_views: Box::new([
                        crate::replacement_representation::ReplacementAttachmentViewPlan {
                            view_type: vk::ImageViewType::TYPE_2D,
                            format: vk::Format::R8G8B8A8_UNORM,
                            components: vk::ComponentMapping::default(),
                            range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 1,
                                level_count: 1,
                                base_array_layer: 1,
                                layer_count: 1,
                            },
                        },
                    ]),
                    shader_views: Box::new([]),
                },
            )
            .expect("replacement epoch allocates an owned working image");
        assert_ne!(image.image().unwrap().image, vk::Image::null());
        let attachment = image
            .image_view(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 1,
                level_count: 1,
                base_array_layer: 1,
                layer_count: 1,
            })
            .expect("declared exact attachment view");
        assert_ne!(attachment.view, vk::ImageView::null());
        assert_ne!(attachment.view, image.image().unwrap().view);
        let planned = epoch
            .create_owned_texture(
                reims_vgpu_protocol::TextureDeclaration {
                    texture_type: reims_vgpu_protocol::TextureType::D2Array,
                    framebuffer_only: false,
                    is_drawable: false,
                    write_swizzle_enabled: None,
                    allow_gpu_optimized_contents: false,
                    usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                    pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                    width: 4,
                    height: 4,
                    depth: 1,
                    mipmap_level_count: 2,
                    sample_count: 1,
                    array_length: 2,
                    resource_options: 0,
                    protection_options: 0,
                    swizzle: None,
                },
                Box::new([
                    crate::replacement_representation::ReplacementAttachmentViewPlan {
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: vk::Format::R8G8B8A8_UNORM,
                        components: crate::translate::pixel::vk_component_mapping(
                            &reims_vgpu_protocol::swizzle_identity(),
                        ),
                        range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 1,
                            level_count: 1,
                            base_array_layer: 1,
                            layer_count: 1,
                        },
                    },
                ]),
                Box::new([]),
                reims_vgpu_core::WorkingMemoryClass::DeviceLocal,
            )
            .expect("decoded texture declaration plans and allocates an exact image");
        assert_ne!(planned.image().unwrap().image, vk::Image::null());
        assert!(planned
            .image_view(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 1,
                level_count: 1,
                base_array_layer: 1,
                layer_count: 1,
            })
            .is_some());
        let identity = reims_vgpu_protocol::ResourceId::new(9, 1);
        let mut semantic = reims_vgpu_core::SamplerResource::normalized_default(3);
        semantic.identity = Some(identity);
        let first = epoch
            .samplers()
            .dynamic(&semantic)
            .expect("create dynamic sampler for one exact generation");
        let second = epoch
            .samplers()
            .dynamic(&semantic)
            .expect("reuse live dynamic sampler generation");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            epoch.samplers().census(),
            crate::replacement_sampler::ReplacementSamplerCensus {
                live: 1,
                hits: 1,
                misses: 1,
            }
        );
        epoch.samplers().retire_dynamic(identity);
        assert_eq!(epoch.samplers().census().live, 0);
        assert_eq!(first.handle(), second.handle());

        let generation = session.session_generation();
        let native = epoch
            .acquire_native_object(&generation)
            .expect("active semantic generation and Vulkan epoch");
        generation.close();
        assert!(native.native_handles_are_usable());
        assert!(epoch.begin_loss());
        assert!(!epoch.begin_loss());
        assert_eq!(epoch.state(), VulkanDeviceEpochState::Losing);
        assert!(!native.native_handles_are_usable());
        assert!(session.epoch_lease().is_none());
        assert!(epoch
            .acquire_native_object(&reims_vgpu_core::SessionGeneration::new(
                reims_vgpu_protocol::SessionGenerationId::new(4)
            ))
            .is_none());
        assert!(matches!(
            epoch.create_owned_buffer(
                4096,
                vk::BufferUsageFlags::TRANSFER_DST,
                crate::memory::MemoryClass::DeviceLocal,
            ),
            Err(crate::replacement_representation::ReplacementBufferAllocationError::DeviceLifetimeClosed)
        ));
        assert!(matches!(
            epoch.create_owned_image(
                crate::replacement_representation::ReplacementImageCreatePlan {
                    flags: vk::ImageCreateFlags::empty(),
                    image_type: vk::ImageType::TYPE_2D,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::R8_UNORM,
                    components: vk::ComponentMapping::default(),
                    extent: vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::SAMPLED,
                    full_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
                    memory_class: crate::memory::MemoryClass::DeviceLocal,
                    attachment_views: Box::new([]),
                    shader_views: Box::new([]),
                },
            ),
            Err(crate::replacement_representation::ReplacementImageAllocationError::DeviceLifetimeClosed)
        ));
    }
}
