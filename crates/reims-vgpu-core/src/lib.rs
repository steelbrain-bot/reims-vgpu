//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod access;
pub mod blit;
pub mod buffer_blit_preparation;
pub mod capabilities;
pub mod command_buffer_progress;
pub mod completion;
pub mod compute;
pub mod compute_dispatch;
pub mod condition_owner;
pub mod content_authority;
pub mod content_synchronization;
pub mod content_tracking;
pub mod continuation;
pub mod contract_coverage;
pub mod coordinator;
pub mod dependency;
pub mod device_session;
pub mod direct_replay;
pub mod display;
pub mod draw;
pub mod draw_preparation;
pub mod encoder_bindings;
pub mod endian;
pub mod exec_resource_preparation;
pub mod execution;
pub mod fixed_executor;
pub mod fnv;
pub mod gather;
pub mod icb;
pub mod image_blit_preparation;
pub mod indirect_command;
pub mod info_query;
pub mod info_query_preparation;
pub mod ingress;
pub mod lifecycle;
pub mod managed_backing;
pub mod managed_resource;
pub mod map_audit;
pub mod mapper;
pub mod mapping;
pub mod materialization;
pub mod namespace;
pub mod native_dependencies;
pub mod native_retirement;
pub mod node_guard;
pub mod object_state;
pub mod observation;
pub mod operation_completion;
pub mod operation_conditions;
pub mod operation_info;
pub mod operation_resource_state;
pub mod participation;
pub mod pipeline_lifecycle;
pub mod pixel_format;
pub mod preparation;
pub mod present_stream;
pub mod publication;
pub mod queue_timeline;
pub mod recording_assignment;
pub mod recording_order;
pub mod reference;
pub mod registers;
pub mod released_pages;
pub mod render;
pub mod render_dispatch;
pub mod replay_acceptance;
pub mod replay_completion;
pub mod replay_retirement;
pub mod residency;
pub mod resource;
pub mod resource_lifecycle;
pub mod resource_state;
pub mod resource_state_preparation;
pub mod scheduler;
pub mod service;
pub mod shader_interface;
pub mod stamp;
pub mod submission;
pub mod submission_order;
pub mod synchronization;
pub mod target;
pub mod task;
pub mod task_namespace;
pub mod texel;
pub mod transaction;
pub mod transaction_runtime;
pub mod transaction_state;
pub mod viewport;
pub mod visibility;
pub mod wait_graph;

pub use access::{
    AccessIntent, AccessMode, AccessPrecision, AccessScope, AccessTarget, ImageAspect,
    ImageSubresourceRange, LinearRange, StageScope, TexelBox,
};
pub use blit::{
    BlitKind, BufferFillPattern, ResolvedBlit, ResolvedBufferRange, ResolvedBufferToTextureBlit,
    ResolvedLinearTextureLevel, ResolvedSurfaceTextureBacking, ResolvedTextureBacking,
    ResolvedTextureCopyBatch, ResolvedTextureEndpoint, ResolvedTextureLevelCopy,
    ResolvedTextureToBufferBlit, ResolvedTextureToTextureBlit, TextureExtent, TextureOrigin,
};
pub use buffer_blit_preparation::{
    cancel_prepared_buffer_blit, commit_buffer_blit_acceptance, prepare_buffer_blit,
    prepare_buffer_blit_with_write, AcceptedBufferBlit, BufferBlitAcceptanceError,
    BufferBlitAcceptanceFailure, BufferBlitCancellationError, BufferBlitCancellationFailure,
    BufferBlitPreparationError, BufferBlitPreparationFailure, CancelledBufferBlit,
    PreparedBufferBlit, PreparedNativeBufferBlit, PreparedNativeBufferRange,
};
pub use capabilities::{
    CapabilityService, ComputeInfoLimits, DeviceInfoLimits, ExecutorCapabilities, MAX_CHANNELS,
};
pub use command_buffer_progress::{
    CommandBufferProgress, CommandBufferProgressError, CommandBufferProgressOwner,
    ProgressPublication, PublishedProgress,
};
pub use completion::{
    AbandonedCompletion, CompletionEvidence, CompletionFact, CompletionOwnerError,
    QueueTimelinePoint, TimelineCompletionOwner,
};
pub use compute::{
    ComputeBarrier, ComputeBufferBacking, ComputeBufferOutput, ComputeBufferResource,
    ComputeBufferResult, ComputeImageDestination, ComputeImageResult, ComputeOutput,
    ComputeRequest, ComputeResidentSampleBind, ComputeSampledImageResource,
    ComputeSampledImageSource, ComputeStorageImageResource, ComputeStorageImageSeed,
    ComputeStorageResidency, SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction,
    SamplerFilter, SamplerMipFilter, SamplerResource, SamplerSource,
};
pub use compute_dispatch::{
    cancel_prepared_compute_dispatch, prepare_compute_dispatch, ComputeBindingClass,
    ComputeBindingView, ComputeDispatchCancellationFailure, ComputeDispatchPreparationError,
    ComputeProgramDispatchContract, PreparedComputeDispatch, ResolvedComputeDispatch,
    ResolvedComputeLaunch, ResolvedComputeNullBinding, ResolvedComputeResourceBinding,
    ResolvedComputeSamplerBinding, COMPUTE_INDIRECT_ARGUMENT_BYTES,
};
pub use condition_owner::{
    ConditionBinding, ConditionNamespaceError, ConditionNamespaceOwner, ConditionOwnerError,
    ConditionPublicationBoundary, ConditionSignal, ConditionWaitResolution,
    ReleasedConditionIdentities, SynchronizationConditionOwner, TaskConditionIdentities,
};
pub use content_authority::{
    BackingRegion, ContentAuthority, ContentAuthorityError, GpuWriteId, HostIngressKey,
    HostIngressTransfer, HostLandingKey, ImageRegion, RegionContentState, RegionVersion,
    TransferKey, GPU_REPRESENTATION, GUEST_REPRESENTATION, HOST_REPRESENTATION,
};
pub use content_synchronization::*;
pub use content_tracking::{
    BufferWriteGens, BufferWriteStamp, GatherKey, GvaPlaneKey, GvaResourceKey, GvaStoreWitness,
    GvaTargetKey, GvaWriteReach, GvaWritebackDebt, HostWriteVerdict, HostWrites, LinearColorTarget,
    PendingWritebacks, ResourceWriteStamp, StatedGeneration,
};
pub use continuation::{ContinuationDependency, ContinuationError, EncoderContinuationOwner};
pub use contract_coverage::{
    ContractCoverageCounts, ContractCoverageError, ContractDisposition, RefusalClosureLedger,
};
pub use coordinator::{
    CoordinationError, DependencyCoordinator, ReadyTransaction, TransactionDependencies,
};
pub use dependency::{
    DependencyCompileError, DependencyCompiler, HazardCause, HazardCompilation, HazardEdge,
    HazardRequirement, HostSafetyCause, HostSafetyEdge,
};
pub use device_session::{DeviceSession, DeviceSessionError};
pub use direct_replay::{
    AcceptedNativeSubmission, CanceledNativeExecutionChain, CanceledNativeSubmission,
    DirectReplayAbandonment, DirectReplayError, DirectReplayNativeOwner, NativeAcceptanceFailure,
    NativeCancellationFailure, NativeChainCancellationFailure, NativeChainFinalPreparationFailure,
    NativePreparationFailure, PreparedAuxiliaryNativeSubmission, PreparedNativeExecutionChain,
    PreparedNativeSubmission, PreparedPresentNativeSubmission,
};
pub use display::{
    CursorGlyph, CursorPosition, CursorState, DisplayHandshake, DisplayOnlineNotification,
    DisplayOnlineNotificationError, DisplayOnlinePoll, DisplayPresentNotification,
    DisplayPresentNotificationError, DisplaySharedPage, DISPLAY_ONLINE_EVENT_MASK,
    DISPLAY_PRESENT_EVENT_MASK, DISPLAY_SHARED_ENABLE_MASK_OFFSET, DISPLAY_SHARED_PENDING_OFFSET,
};
pub use draw_preparation::{
    AttachmentPlanDecline, AttachmentTargetRole, DrawPreparationDecline, SamplerBindingSource,
};
pub use encoder_bindings::{
    select_descriptor_tier, BindingTable, BindingTableError, BindingUpdate, DescriptorCapabilities,
    DescriptorTier, ReflectedBindingUse,
};
pub use exec_resource_preparation::{
    assemble_prepared_exec_resources, assemble_prepared_exec_resources_at,
    assemble_prepared_exec_resources_at_positions, cancel_prepared_exec_resource_inputs,
    cancel_prepared_exec_resources, validate_cancel_prepared_exec_resources,
    AcceptedExecResourceOutcomes, CancelledExecResources, ComputeDispatchPreparationInput,
    ComputeDispatchPreparationStepResult, DispatchPreparationInput, ExecResourceCancellationError,
    ExecResourceCancellationFailure, ExecResourceCancellationResult,
    ExecResourceInputAdmissionFailure, ExecResourceInputAdmissionResult,
    ExecResourcePreparationError, ExecResourcePreparationFailure, ExecResourcePreparationOwner,
    ExecResourcePreparationResult, ExecResourcePreparationStepFailure,
    ExecResourcePreparationStepResult, PreparedExecResourceInputs, PreparedExecResources,
    RenderDispatchPreparationInput, RenderDispatchPreparationStepResult,
};
pub use execution::{
    execute_resolved_submission, execute_resolved_submission_progress, BlitCompletion,
    CommandExecution, ExecutionCompletion, ExecutionKind, ExecutionOutput, ExecutionPort,
    ExecutionReceipt, ResolvedCommand, ResolvedCommandBuffer, ResolvedExecutionCompletion,
    ResolvedSubmission, ResourceStateCompletion, SubmissionExecutionProgress,
};
pub use fixed_executor::{
    FixedExecutor, FixedExecutorCensus, FixedExecutorError, FixedExecutorEvent,
    FixedExecutorOutcome, FixedExecutorWakeInstallError,
};
pub use gather::{
    fold_runs, AuditDensity, ContentAudit, GatherObservation, GatherOutcome, GatherPolicies,
    GatherReadings, GatherVerdict, GatherWindow, GatherWitness, GatheredIdentity, StatedGuestWrite,
    VouchPolicy, AUDIT_REBASELINE_LIMIT,
};
pub use icb::{IcbRecord, IcbRegistry};
pub use image_blit_preparation::{
    blit_content_synchronization_requests, cancel_prepared_image_blit,
    commit_image_blit_acceptance, prepare_image_blit, prepare_image_blit_with_write,
    resolved_blit_accesses, AcceptedImageBlit, CancelledImageBlit, ImageBlitAcceptanceError,
    ImageBlitAcceptanceFailure, ImageBlitCancellationError, ImageBlitCancellationFailure,
    ImageBlitPreparationError, ImageBlitPreparationFailure, PreparedBlitRepresentation,
    PreparedImageBlit,
};
pub use indirect_command::{
    admit_indirect_commands, admit_indirect_commands_with_owner,
    cancel_prepared_indirect_range_readback, commit_indirect_commands,
    complete_indirect_range_readback, complete_indirect_range_readback_batch,
    expand_committed_indirect_executions, prepare_indirect_execution,
    prepare_indirect_range_readback, resolve_indirect_command,
    resolve_indirect_command_memory_read, resolve_indirect_command_population,
    resolve_indirect_execution, resolve_indirect_execution_arguments,
    resume_indirect_range_execution, AdmittedIndirectCommands, CancelledIndirectRangeReadback,
    CommittedIndirectCommands, CommittedIndirectExecution, DecodedIndirectCommandPopulation,
    ExpandedIndirectOperationOrigin, IndirectCommandAdmissionError,
    IndirectCommandBufferResolution, IndirectCommandCommitError, IndirectCommandExecutionKind,
    IndirectCommandKind, IndirectCommandMemoryReadError, IndirectCommandMemoryReadPlan,
    IndirectCommandMutationError, IndirectCommandPopulation, IndirectCommandPopulationFailure,
    IndirectCommandPopulationResolutionError, IndirectCommandPopulationResolutionFailure,
    IndirectCommandResolutionError, IndirectCommandResolver, IndirectCommandSlotOwner,
    IndirectExecutionArgumentsResolution, IndirectExecutionArgumentsResolver,
    IndirectExecutionExpansionError, IndirectExecutionOperation, IndirectExecutionPreparationError,
    IndirectExecutionPreparationFailure, IndirectExecutionTransaction,
    IndirectRangeExecutionContinuation, IndirectRangeExecutionPhase,
    IndirectRangeExecutionResumeError, IndirectRangeExecutionResumeFailure,
    IndirectRangeExecutionResumeResult, IndirectRangeExecutionStartError,
    IndirectRangeReadbackBatchError, IndirectRangeReadbackBatchFailure,
    IndirectRangeReadbackCancellationFailure, IndirectRangeReadbackError,
    IndirectRangeReadbackFailure, IndirectRangeResourceOperation, NextIndirectRangeExecution,
    PendingIndirectRangeExecution, PreparedIndirectExecution, PreparedIndirectRangeReadback,
    PriorIndirectCommandPopulation, ResolvedIndirectCommand, ResolvedIndirectCommandRange,
    ResolvedIndirectCommandSlot,
};
pub use info_query::{
    evaluate_info_query, render_pipeline_imageblock_memory_length, render_pipeline_state_info,
    resolve_info_operation, ComputePipelineStateInfo, EvaluatedInfoQuery, GpuAddressInfo,
    HeapTextureSizeAndAlignInfo, ImageblockMemoryLength, IndexedGpuResourceInfo,
    InfoOperationResolver, InfoQueryEvaluationError, InfoQueryEvaluator, InfoReplyResolutionError,
    InfoResolutionError, MappedRateCoordinate, RenderPipelineStateInfo, ResolvedInfoOperation,
    ResolvedInfoReplyTarget, ResolvedRateParameterDestination,
};
pub use info_query_preparation::{
    cancel_prepared_info_query, prepare_info_query, CancelledInfoQuery, InfoQueryCancellationError,
    InfoQueryCancellationFailure, InfoQueryPreparationError, InfoQueryPreparationFailure,
    PreparedInfoQuery,
};
pub use ingress::{TransactionIngressError, TransactionIngressOwner};
pub use lifecycle::{
    NativeObjectLease, SessionGeneration, SessionGenerationLease, VulkanDeviceEpoch,
    VulkanDeviceEpochLease, VulkanDeviceEpochState,
};
pub use managed_backing::{
    GpuWriteBatchError, GpuWriteRequest, GpuWriteReservation, ManagedBackingCensus,
    ManagedBackingError, ManagedBackingOwner, ManagedBackingProgress, ManagedRepresentationFailure,
    RepresentationUse, TransferBatchError,
};
pub use managed_resource::{
    plan_representation, HostMemoryTopology, RepresentationCapabilities, RepresentationRefusal,
    RepresentationRoute, WorkingMemoryClass,
};
pub use map_audit::{MapAudit, MapIntervals, PageSize};
pub use mapper::{
    resolve_mapper_request_read, resolve_mapper_surface_backing, MapperCapture,
    MapperCapturePublicationError, MapperRequestReadError, MapperRequestReadPlan, MapperService,
    MapperSurfaceBacking, MapperSurfaceBackingError, MapperTexturePlaneError,
    MapperTexturePlanePlan,
};
pub use mapping::{MappingContentState, ResourceValidity};
pub use materialization::{
    BoundWindowKey, GuestAddressSpan, MaterializationOwner, MaterializationRegistry,
    MaterializationRetirement, MaterializationShape,
};
pub use namespace::{NamespaceError, ReferenceNamespace, TaskReferenceStates};
pub use native_dependencies::{
    NativeDependencyError, NativeDependencyOwner, NativeSubmissionPlan, NativeWait,
};
pub use native_retirement::{
    NativeRetirement, NativeRetirementDisposition, NativeRetirementError, NativeRetirementFailure,
};
pub use node_guard::{NodeVerdict, NodeWatch};
pub use object_state::{ObjectStateError, ObjectStateOwner};
pub use observation::DeviceObservations;
pub use operation_completion::AdmittedCompletionEffects;
pub use operation_conditions::{
    AdmittedConditionOperation, AdmittedExecConditions, ExecConditionPlan,
};
pub use operation_info::AdmittedInfoQueries;
pub use operation_resource_state::{AdmittedResourceStateOperation, AdmittedResourceStates};
pub use participation::{
    ParticipationAccessError, ParticipationOperation, ParticipationResourceTarget,
    ParticipationScope,
};
pub use pipeline_lifecycle::{
    PipelineCompileJob, PipelineFailureStage, PipelineLifecycle, PipelineLifecycleCensus,
    PipelineLifecycleError, PipelineReadiness, PipelineRefusal, PipelineRetirement, PipelineState,
    PipelineTranslationJob, PipelineVariantAdmission, PipelineVariantCensus,
    PipelineVariantCompileJob, PipelineVariantFamily, PipelineVariantLifecycleError,
    PipelineVariantPublication, PipelineVariantReadiness, PipelineVariantState, ReadyPipelineLease,
    RetiredPipelineVariantWaiters, RetiredPipelines,
};
pub use preparation::{
    sampled_image_shape, BindTableClass, IndexLoadReason, MrtDrop, MtlbDecline, PastTableBind,
    ResolvedRenderPipeline, SampledImageShape, SecondaryMrtRefusal, ShaderStage, VertexBindPlan,
    MAX_ANY_BIND_SLOTS, MAX_BUFFER_BIND_SLOTS, MAX_SAMPLER_BIND_SLOTS, MAX_TEXTURE_BIND_SLOTS,
};
pub use present_stream::{
    PresentStream, PresentStreamError, PresentTicket, QueuedPresent, SwapchainState,
};
pub use publication::{
    completion_stamp_slot_count, completion_stamp_slot_offset, CompletionStamp, PublicationError,
    PublicationFact, PublicationOwner, PublicationPosition, PublishedFact,
};
pub use queue_timeline::{QueueTimelineError, QueueTimelineOwner};
pub use recording_assignment::{
    RecordingAssignmentError, RecordingAssignmentOwner, RecordingWorkerId,
};
pub use recording_order::{RecordingOrderError, RecordingOrderOwner};
pub use reference::{
    CompletionPath, ReferenceCompletion, ReferenceError, ReferenceSemantics,
    SerialReferenceInterpreter,
};
pub use registers::{DeviceRegisters, GfxRegisters, IosfcRegisters, GFX_MMIO_SIZE};
pub use released_pages::{ReleasedPages, ReleasedVerdict, RELEASED_PAGE_WATCH_CAP};
pub use render::{
    viewport_slot_count, AttachmentInitial, AttachmentSlot, BlendFactor, BlendOp,
    BlendStateResource, BufferContent, ColorLoadAction, CullMode, DepthAspectAttachment,
    DepthAttachment, DepthClipMode, DepthState, DrawOutput, DrawRequest, FillMode, IndexType,
    IndexedDrawResource, LineWidth, PreparedRenderProgram, PreparedShaderStage, PrimitiveTopology,
    RenderBarrier, RenderBarrierStages, RenderEncoderDelta, RenderTargetExtent, SampledByteOrigin,
    SampledContentIdentity, SampledImageResource, SampledSource, ScissorResource,
    SecondaryColorTarget, SeedOrder, StencilAttachment, StencilFaceOps, StencilOp, StencilState,
    StorageBufferResource, VertexAttributeFormat, VertexAttributeResource, VertexStepFunction,
    ViewportResource, VisibilityResultMode,
};
pub use render_dispatch::*;
pub use replay_acceptance::{
    commit_replay_acceptance, validate_replay_acceptance, ReplayAcceptance, ReplayAcceptanceError,
    ReplayAcceptanceFailure,
};
pub use replay_completion::{
    commit_replay_completion, commit_replay_semantic_completion, commit_replay_timeline_progress,
    CommittedReplayCompletion, ReplayCompletionError, ReplayCompletionFailure,
    ReplaySemanticCompletionError, ReplaySemanticCompletionFailure, ReplayTimelineProgress,
    ReplayTimelineProgressError, ResolvedReplayCompletion,
};
pub use replay_retirement::{commit_replay_retirement, ReplayRetirementError};
pub use residency::{
    ComputeResidencyLedger, ComputeResidencyService, ComputeStorageOrigin,
    ComputeStorageResidencyKey, GatherVouch, ResidentContentBacking,
};
pub use resource::{
    ContentError, ContentStamp, ContentState, GraphError, LifecycleState, MappingNode,
    PendingContentWrite, ReplicaVersions, ResolvedTextureBindingView, ResolvedTextureView,
    ResolvedTextureViewRange, ResourceGraph, ResourceLifetime, ResourceLifetimeRef, ResourceNode,
    StorageBacking, StorageNode, TextureViewResolveError, MAX_TEXTURE_VIEW_CHAIN,
};
pub use resource_lifecycle::{
    HostIngressBatchError, HostLandingBatchError, ResolvedResourceCompletion,
    ResolvedResourceLifecycle, ResolvedValidityTarget, ResolvedValidityTransition,
    ResourceCompletionBatchError, ResourceCompletionEffect, ResourceLifecycleEffect,
    ResourceLifecycleError, ResourceLifecycleOwner, ResourceUseBatchError, RetiredBacking,
    ValidityRepresentations, ValidityTransitionError,
};
pub use resource_state::{ResolvedResourceState, ResolvedResourceStateTarget};
pub use resource_state_preparation::{
    assemble_prepared_resource_states, cancel_prepared_resource_state,
    cancel_prepared_resource_state_batch, prepare_resource_state, CancelledResourceState,
    CancelledResourceStateBatch, CancelledResourceStatePreparation, PreparedResourceState,
    PreparedResourceStateBatch, ResourceStateBatchCancellationFailure, ResourceStateBatchError,
    ResourceStateBatchFailure, ResourceStateCancellationError, ResourceStateCancellationFailure,
    ResourceStateOutcome, ResourceStatePreparationCancellationFailure,
    ResourceStatePreparationError, ResourceStatePreparationFinishFailure,
    ResourceStatePreparationOwner, ResourceStatePreparationOwnerError,
};
pub use scheduler::{
    ChannelRing, ChildDrainNestingError, ChildDrainStack, PendingWork, PresentTranslationBarrier,
    TranslationOrderHold, TranslationScheduling, UnreleasedTranslationHold, WorkSchedulingState,
};
pub use service::{
    GuestWriteReach, GuestWriteService, HeapTextureImagePlan, HeapTextureRequirements,
    PreparedPresentation, PresentDecline, PresentationRoute, PresentationService,
    PresentationSource, ReadbackLease, ReadbackService, ResidentContent, ResidentReadPlan,
    ResidentReclaim, ResidentService, TargetReadback,
};
pub use shader_interface::*;
pub use stamp::{CompletionPublications, PendingStamp, StampLedger, StampWait, UnmetSource};
pub use submission::{
    SubmissionAcceptError, SubmissionAdmissionRefusal, SubmissionAdmissions, SubmissionCommitOrder,
    SubmissionCommitOrderError, SubmissionConflict, SubmissionContext, SubmissionDispatch,
    SubmissionFootprint, SubmissionRecordError, SubmissionScheduler, SubmissionTracker,
    SubmissionWork,
};
pub use submission_order::{SubmissionOrderError, SubmissionOrderOwner, SubmissionReady};
pub use synchronization::{
    plan_event, plan_fence, BarrierResource, Decision as SynchronizationDecision,
    Domain as SynchronizationDomain, EventKind, FenceAction, FenceSignal, MemoryBarrierScope,
    Plan as SynchronizationPlan, Reason as SynchronizationReason, TaskEventStates, TaskFenceStates,
    TaskGenerationStates, FENCE_INITIAL_GENERATION,
};
pub use target::{TargetIdentity, TargetKeyDivergence};
pub use task::{
    resolve_object_list_entry_read, ObjectListEntryReadError, ObjectListEntryReadPlan, TaskEntry,
    TaskTable,
};
pub use task_namespace::{ReleasedTaskNamespace, TaskNamespaceOwner, TaskNamespaceSnapshot};
pub use texel::{
    expand_rgba8_to_texel, f16_to_f32, f16_to_unorm8, narrow_texel_to_rgba8, unorm8_to_f16,
};
pub use transaction::{
    resource_state_execution_chain, BarrierOperation, DeviceTransaction, DeviceTransactionPayload,
    EncoderBoundary, EventOperation, EventOperationKind, ExecPrologue, ExecTransaction,
    FenceOperation, FenceOperationKind, FenceScope, OperationKind, ResolvedExecSegment,
    ResolvedExecStream, ResolvedOperation, ResourceStateExecutionChain,
    ResourceStateExecutionChainError, TransactionClass, TransactionEnvelopeError,
    TransactionPrerequisite,
};
pub use transaction_runtime::{
    AdmittedExecTransaction, AdmittedExpandedExecProofs, ExpandedExecProofError,
    PrerequisiteResolution, ResolvedExecDeviceTransaction, ResolvedTransactionPrerequisite,
    TransactionRecordingPlan, TransactionRuntime, TransactionRuntimeError,
};
pub use transaction_state::{
    ResetDisposition, TransactionState, TransactionStateMachine, TransactionTransitionError,
};
pub use viewport::{aspect_fit_viewport, pointer_to_guest, PresentationViewport};
pub use wait_graph::{
    ExplicitWaitCause, WaitCycle, WaitDependencyCause, WaitGraph, WaitGraphError,
    WaitGraphRetireError,
};
