//! Native staging for GPU-produced indirect-command execution ranges.
//!
//! The semantic token already retains the exact source representation. This
//! module adds only an eight-byte host-visible destination and the Vulkan copy
//! operands. The bytes are not readable until the recording's timeline point
//! retires; callers complete the semantic continuation only after that proof.

use crate::{
    engine::context::SharedDeviceContext,
    replacement_buffer_blit::{NativeBufferTarget, ReplacementBufferResolver},
};
use ash::vk;
use reims_vgpu_core::PreparedIndirectRangeReadback;
use reims_vgpu_protocol::{BackingId, RepresentationId, TransactionId};
use std::sync::{Arc, Mutex};

const RANGE_BYTES: u64 = 8;

/// Device-incarnation services required to allocate and retire one range
/// staging buffer. Implementations are retained strongly by every allocation.
pub trait ReplacementIndirectRangeDevice: Send + Sync {
    fn device(&self) -> &ash::Device;
    fn readback_memory_type(&self, type_bits: u32, bytes: u64) -> Option<u32>;
    fn mapped_memory_kind(&self, memory_type: u32) -> crate::memory::MappedMemoryKind;
}

impl ReplacementIndirectRangeDevice for SharedDeviceContext {
    fn device(&self) -> &ash::Device {
        &self.device
    }

    fn readback_memory_type(&self, type_bits: u32, bytes: u64) -> Option<u32> {
        self.memory_type_for(type_bits, bytes, crate::memory::MemoryClass::Readback)
    }

    fn mapped_memory_kind(&self, memory_type: u32) -> crate::memory::MappedMemoryKind {
        crate::memory::MappedMemoryKind::of(&self.memory_properties, memory_type)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementIndirectRangeError {
    UnknownRepresentation {
        backing: BackingId,
        representation: RepresentationId,
    },
    SourceRangeOutOfBounds,
    SourceAddressOverflow,
    MissingTransferSource,
    CreateBuffer(vk::Result),
    NoReadbackMemory,
    AllocateMemory(vk::Result),
    BindMemory(vk::Result),
    MapMemory(vk::Result),
    InvalidateMemory(vk::Result),
}

struct ReplacementIndirectRangeStaging {
    context: Arc<dyn ReplacementIndirectRangeDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: usize,
    coherent: bool,
    bytes: Mutex<Option<[u8; RANGE_BYTES as usize]>>,
}

impl std::fmt::Debug for ReplacementIndirectRangeStaging {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplacementIndirectRangeStaging")
            .field("buffer", &self.buffer)
            .field("memory", &self.memory)
            .field("coherent", &self.coherent)
            .finish_non_exhaustive()
    }
}

impl ReplacementIndirectRangeStaging {
    fn read_after_timeline(
        &self,
    ) -> Result<[u8; RANGE_BYTES as usize], ReplacementIndirectRangeError> {
        let mut cached = self.bytes.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(bytes) = *cached {
            return Ok(bytes);
        }
        if !self.coherent {
            unsafe {
                self.context.device().invalidate_mapped_memory_ranges(&[
                    vk::MappedMemoryRange::default()
                        .memory(self.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE),
                ])
            }
            .map_err(ReplacementIndirectRangeError::InvalidateMemory)?;
        }
        let mut bytes = [0; RANGE_BYTES as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mapped as *const u8,
                bytes.as_mut_ptr(),
                RANGE_BYTES as usize,
            );
        }
        *cached = Some(bytes);
        Ok(bytes)
    }
}

impl Drop for ReplacementIndirectRangeStaging {
    fn drop(&mut self) {
        unsafe {
            self.context.device().unmap_memory(self.memory);
            self.context.device().destroy_buffer(self.buffer, None);
            self.context.device().free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReplacementIndirectRangeProgram {
    transaction: TransactionId,
    index: usize,
    operation: reims_vgpu_core::ResolvedIndirectCommand,
    backing: reims_vgpu_protocol::BackingId,
    source: vk::Buffer,
    source_offset: u64,
    staging: Arc<ReplacementIndirectRangeStaging>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementIndirectRangeJoinError {
    CountMismatch {
        readbacks: usize,
        programs: usize,
    },
    IdentityMismatch(usize),
    PhaseMismatch,
    Read {
        index: usize,
        reason: ReplacementIndirectRangeError,
    },
}

#[derive(Debug)]
pub struct ReplacementIndirectRangePhaseFailure<Render, Compute, Info, Completion> {
    pub pending: reims_vgpu_core::PendingIndirectRangeExecution<Render, Compute, Info, Completion>,
    pub reason: ReplacementIndirectRangeJoinFailure,
}

pub type ReplacementIndirectRangePhaseResult<Render, Compute, Info, Completion> = Result<
    reims_vgpu_core::IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>,
    Box<ReplacementIndirectRangePhaseFailure<Render, Compute, Info, Completion>>,
>;

#[derive(Debug)]
pub enum ReplacementIndirectRangeJoinFailure {
    Native {
        reason: ReplacementIndirectRangeJoinError,
        readbacks: Box<[PreparedIndirectRangeReadback]>,
        programs: Box<[ReplacementIndirectRangeProgram]>,
    },
    Semantic {
        failure: Box<reims_vgpu_core::IndirectRangeReadbackBatchFailure>,
        programs: Box<[ReplacementIndirectRangeProgram]>,
    },
}

/// Native range staging whose recording has crossed its queue timeline point.
///
/// Only replay retirement constructs this value. Keeping the programs opaque
/// prevents semantic range materialization from racing their GPU copies.
#[derive(Debug)]
pub struct RetiredReplacementIndirectRanges {
    programs: Box<[ReplacementIndirectRangeProgram]>,
}

impl RetiredReplacementIndirectRanges {
    pub(crate) fn new(programs: Box<[ReplacementIndirectRangeProgram]>) -> Self {
        Self { programs }
    }

    #[cfg(test)]
    fn after_explicit_timeline_wait(programs: Box<[ReplacementIndirectRangeProgram]>) -> Self {
        Self { programs }
    }
}

impl ReplacementIndirectRangeProgram {
    pub fn resolve(
        context: Arc<dyn ReplacementIndirectRangeDevice>,
        prepared: &PreparedIndirectRangeReadback,
        resolver: &impl ReplacementBufferResolver,
    ) -> Result<Self, ReplacementIndirectRangeError> {
        let backing = prepared.arguments_backing();
        let representation = prepared.arguments_representation();
        let source = resolver.resolve_buffer(backing, representation).ok_or(
            ReplacementIndirectRangeError::UnknownRepresentation {
                backing,
                representation,
            },
        )?;
        validate_source(source, prepared.arguments_range())?;
        let staging = allocate_staging(context)?;
        Ok(Self {
            transaction: prepared.transaction(),
            index: prepared.operation_index(),
            operation: prepared.operation(),
            backing,
            source: source.buffer,
            source_offset: source
                .base_offset
                .checked_add(prepared.arguments_range().start())
                .ok_or(ReplacementIndirectRangeError::SourceAddressOverflow)?,
            staging,
        })
    }

    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn operation(&self) -> reims_vgpu_core::ResolvedIndirectCommand {
        self.operation
    }

    pub const fn backing(&self) -> reims_vgpu_protocol::BackingId {
        self.backing
    }

    fn read_after_timeline(
        &self,
    ) -> Result<[u8; RANGE_BYTES as usize], ReplacementIndirectRangeError> {
        self.staging.read_after_timeline()
    }
}

/// Join timeline-retired staging with the whole-EXEC semantic tokens and
/// materialize the literal executions for continuation resubmission.
pub fn complete_indirect_ranges_after_timeline<Command>(
    owner: &reims_vgpu_core::IndirectCommandSlotOwner<Command>,
    readbacks: Box<[PreparedIndirectRangeReadback]>,
    retired: RetiredReplacementIndirectRanges,
) -> Result<Box<[reims_vgpu_core::ResolvedIndirectCommand]>, ReplacementIndirectRangeJoinFailure> {
    let programs = retired.programs;
    if readbacks.len() != programs.len() {
        return Err(ReplacementIndirectRangeJoinFailure::Native {
            reason: ReplacementIndirectRangeJoinError::CountMismatch {
                readbacks: readbacks.len(),
                programs: programs.len(),
            },
            readbacks,
            programs,
        });
    }
    for (index, (readback, program)) in readbacks.iter().zip(programs.iter()).enumerate() {
        if readback.transaction() != program.transaction()
            || readback.operation_index() != program.index()
            || readback.operation() != program.operation()
        {
            return Err(ReplacementIndirectRangeJoinFailure::Native {
                reason: ReplacementIndirectRangeJoinError::IdentityMismatch(index),
                readbacks,
                programs,
            });
        }
    }
    let mut bytes = Vec::with_capacity(programs.len());
    for (index, program) in programs.iter().enumerate() {
        match program.read_after_timeline() {
            Ok(entry) => bytes.push(entry),
            Err(reason) => {
                return Err(ReplacementIndirectRangeJoinFailure::Native {
                    reason: ReplacementIndirectRangeJoinError::Read { index, reason },
                    readbacks,
                    programs,
                });
            }
        }
    }
    reims_vgpu_core::complete_indirect_range_readback_batch(
        owner,
        readbacks,
        bytes.into_boxed_slice(),
    )
    .map_err(|failure| ReplacementIndirectRangeJoinFailure::Semantic { failure, programs })
}

/// Join one retired range-copy phase to the exact core continuation it paused.
/// Successful return is the sole path to the next auxiliary or final phase.
pub fn resume_indirect_range_after_timeline<Command, Render, Compute, Info, Completion>(
    owner: &reims_vgpu_core::IndirectCommandSlotOwner<Command>,
    pending: reims_vgpu_core::PendingIndirectRangeExecution<Render, Compute, Info, Completion>,
    readbacks: Box<[PreparedIndirectRangeReadback]>,
    retired: RetiredReplacementIndirectRanges,
) -> ReplacementIndirectRangePhaseResult<Render, Compute, Info, Completion> {
    let matches_phase = readbacks.len() == 1
        && retired.programs.len() == 1
        && readbacks[0].transaction() == pending.phase().transaction()
        && readbacks[0].operation_index() == pending.operation_index()
        && readbacks[0].operation() == pending.operation();
    if !matches_phase {
        return Err(Box::new(ReplacementIndirectRangePhaseFailure {
            pending,
            reason: ReplacementIndirectRangeJoinFailure::Native {
                reason: ReplacementIndirectRangeJoinError::PhaseMismatch,
                readbacks,
                programs: retired.programs,
            },
        }));
    }
    let literal = match complete_indirect_ranges_after_timeline(owner, readbacks, retired) {
        Ok(literal) => literal
            .into_vec()
            .pop()
            .expect("one phase owns exactly one prevalidated range"),
        Err(reason) => {
            return Err(Box::new(ReplacementIndirectRangePhaseFailure {
                pending,
                reason,
            }))
        }
    };
    match reims_vgpu_core::resume_indirect_range_execution(pending, literal) {
        Ok(continuation) => Ok(continuation),
        Err(_) => unreachable!("the exact pending operation identity was prevalidated"),
    }
}

fn validate_source(
    source: NativeBufferTarget,
    range: reims_vgpu_core::LinearRange,
) -> Result<(), ReplacementIndirectRangeError> {
    if range.end() > source.size {
        return Err(ReplacementIndirectRangeError::SourceRangeOutOfBounds);
    }
    if !source.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC) {
        return Err(ReplacementIndirectRangeError::MissingTransferSource);
    }
    source
        .base_offset
        .checked_add(range.start())
        .ok_or(ReplacementIndirectRangeError::SourceAddressOverflow)?;
    Ok(())
}

fn allocate_staging(
    context: Arc<dyn ReplacementIndirectRangeDevice>,
) -> Result<Arc<ReplacementIndirectRangeStaging>, ReplacementIndirectRangeError> {
    let buffer = unsafe {
        context.device().create_buffer(
            &vk::BufferCreateInfo::default()
                .size(RANGE_BYTES)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        )
    }
    .map_err(ReplacementIndirectRangeError::CreateBuffer)?;
    let requirements = unsafe { context.device().get_buffer_memory_requirements(buffer) };
    let Some(memory_type) =
        context.readback_memory_type(requirements.memory_type_bits, requirements.size)
    else {
        unsafe { context.device().destroy_buffer(buffer, None) };
        return Err(ReplacementIndirectRangeError::NoReadbackMemory);
    };
    let memory = match unsafe {
        context.device().allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(reason) => {
            unsafe { context.device().destroy_buffer(buffer, None) };
            return Err(ReplacementIndirectRangeError::AllocateMemory(reason));
        }
    };
    if let Err(reason) = unsafe { context.device().bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            context.device().destroy_buffer(buffer, None);
            context.device().free_memory(memory, None);
        }
        return Err(ReplacementIndirectRangeError::BindMemory(reason));
    }
    let mapped = match unsafe {
        context
            .device()
            .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
    } {
        Ok(mapped) => mapped,
        Err(reason) => {
            unsafe {
                context.device().destroy_buffer(buffer, None);
                context.device().free_memory(memory, None);
            }
            return Err(ReplacementIndirectRangeError::MapMemory(reason));
        }
    };
    Ok(Arc::new(ReplacementIndirectRangeStaging {
        coherent: context.mapped_memory_kind(memory_type).coherent,
        context,
        buffer,
        memory,
        mapped: mapped as usize,
        bytes: Mutex::new(None),
    }))
}

/// Record the exact eight-byte copy into this program's retained staging.
///
/// # Safety
///
/// `command_buffer` must be recording on `device`, and every source and
/// staging handle retained by `program` must belong to that same live device.
pub unsafe fn record_indirect_range_readback(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    program: &ReplacementIndirectRangeProgram,
) {
    device.cmd_copy_buffer(
        command_buffer,
        program.source,
        program.staging.buffer,
        &[vk::BufferCopy {
            src_offset: program.source_offset,
            dst_offset: 0,
            size: RANGE_BYTES,
        }],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::BackingView;

    struct Resolver {
        backing: BackingId,
        representation: RepresentationId,
        target: NativeBufferTarget,
    }

    impl ReplacementBufferResolver for Resolver {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            (backing == self.backing && representation == self.representation)
                .then_some(self.target)
        }
    }

    struct EmptyBarrierResolver;

    impl crate::replacement_barrier_record::ReplacementBarrierResolver for EmptyBarrierResolver {
        fn resolve(
            &self,
            _: BackingId,
        ) -> Option<crate::replacement_barrier_record::NativeBarrierResolution> {
            None
        }
    }

    impl crate::replacement_barrier_record::ReplacementBarrierResourceResolver
        for EmptyBarrierResolver
    {
        fn alias_backings(
            &self,
            _: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
        ) -> Option<Box<[BackingId]>> {
            None
        }
    }

    #[test]
    fn source_projection_requires_the_exact_transfer_capable_window() {
        let source = NativeBufferTarget {
            buffer: vk::Buffer::null(),
            base_offset: 16,
            accessible_size: 32,
            size: 32,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
        };
        assert_eq!(
            validate_source(source, reims_vgpu_core::LinearRange::new(24, 8).unwrap()),
            Ok(())
        );
        assert_eq!(
            validate_source(source, reims_vgpu_core::LinearRange::new(28, 8).unwrap()),
            Err(ReplacementIndirectRangeError::SourceRangeOutOfBounds)
        );
        assert_eq!(
            validate_source(
                NativeBufferTarget {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    ..source
                },
                reims_vgpu_core::LinearRange::new(24, 8).unwrap()
            ),
            Err(ReplacementIndirectRangeError::MissingTransferSource)
        );
    }

    #[test]
    fn actual_copy_becomes_readable_only_from_the_retained_staging_program() {
        reims_vgpu_observe::redirect_logs_for_tests();
        let context = match unsafe { crate::engine::context::DeviceContext::create() } {
            Ok(context) => context,
            Err(error) => {
                eprintln!("SKIP replacement indirect range: no device ({error})");
                return;
            }
        };
        let context = Arc::new(SharedDeviceContext::new(context));
        let source = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(32)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { context.device.get_buffer_memory_requirements(source) };
        let Some(memory_type) = context.memory_type_for(
            requirements.memory_type_bits,
            requirements.size,
            crate::memory::MemoryClass::Upload,
        ) else {
            unsafe { context.device.destroy_buffer(source, None) };
            eprintln!("SKIP replacement indirect range: no host-visible source memory");
            return;
        };
        let memory = unsafe {
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .unwrap();
        unsafe { context.device.bind_buffer_memory(source, memory, 0) }.unwrap();
        let mapped = unsafe {
            context
                .device
                .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
        }
        .unwrap()
        .cast::<u8>();
        unsafe {
            mapped.write_bytes(0, requirements.size as usize);
            std::ptr::copy_nonoverlapping(
                [2u8, 0, 0, 0, 3, 0, 0, 0].as_ptr(),
                mapped.add(16),
                RANGE_BYTES as usize,
            );
        }
        let source_kind = context.mapped_memory_kind(memory_type);
        if !source_kind.coherent {
            unsafe {
                context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .unwrap();
        }
        let mut resources = reims_vgpu_core::ResourceLifecycleOwner::new(
            reims_vgpu_protocol::VulkanDeviceEpochId::new(1),
        );
        let reims_vgpu_core::ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(reims_vgpu_core::ResolvedResourceLifecycle::CreateBacking {
                backing: reims_vgpu_core::StorageBacking::Dedicated,
                regions: Box::new([reims_vgpu_core::BackingRegion::Linear(
                    reims_vgpu_core::LinearRange::new(0, 32).unwrap(),
                )]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = resources
            .create_execution_representation(
                backing,
                reims_vgpu_core::RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let operation = reims_vgpu_core::ResolvedIndirectCommand::ExecuteIndirectRange {
            icb: reims_vgpu_protocol::ResourceId::new(3, 1),
            arguments_resource: reims_vgpu_protocol::ResourceId::new(4, 1),
            arguments_backing: backing,
            arguments_range: reims_vgpu_core::LinearRange::new(16, RANGE_BYTES).unwrap(),
            kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
        };
        let readback = reims_vgpu_core::prepare_indirect_range_readback(
            &mut resources,
            TransactionId::new(5),
            0,
            operation,
        )
        .unwrap();
        let owner: Arc<dyn ReplacementIndirectRangeDevice> = context.clone();
        let program = ReplacementIndirectRangeProgram::resolve(
            owner,
            &readback,
            &Resolver {
                backing,
                representation,
                target: NativeBufferTarget {
                    buffer: source,
                    base_offset: 0,
                    accessible_size: 32,
                    size: 32,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                },
            },
        )
        .unwrap();
        let exec = reims_vgpu_core::ExecTransaction {
            identity: reims_vgpu_protocol::SubmissionIdentity {
                id: reims_vgpu_protocol::SubmissionId::new(6),
                task: reims_vgpu_protocol::TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
            streams: Box::new([reims_vgpu_core::ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([reims_vgpu_core::ResolvedExecSegment {
                    boundary: reims_vgpu_protocol::SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: reims_vgpu_protocol::SegmentKind::Render,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([reims_vgpu_core::ResolvedOperation::<
                        (),
                        (),
                        (),
                        reims_vgpu_core::ResolvedIndirectCommand,
                        (),
                    >::IndirectCommand(operation)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let mut icbs = reims_vgpu_core::IndirectCommandSlotOwner::<()>::default();
        icbs.register(reims_vgpu_protocol::ResourceId::new(3, 1), 8)
            .unwrap();
        let admitted = reims_vgpu_core::admit_indirect_commands_with_owner(
            TransactionId::new(5),
            &exec,
            &icbs,
        )
        .unwrap();
        let pending = match reims_vgpu_core::IndirectRangeExecutionContinuation::new(
            TransactionId::new(5),
            exec.clone(),
        )
        .next()
        {
            reims_vgpu_core::NextIndirectRangeExecution::Readback(pending) => pending,
            reims_vgpu_core::NextIndirectRangeExecution::Final(_) => {
                panic!("the unresolved range must create a readback phase")
            }
        };
        let mut mismatched_program = program.clone();
        mismatched_program.operation = reims_vgpu_core::ResolvedIndirectCommand::Optimize {
            icb: reims_vgpu_protocol::ResourceId::new(3, 1),
            range: reims_vgpu_core::ResolvedIndirectCommandRange {
                location: 0,
                length: 1,
            },
        };
        let mismatch =
            crate::replacement_recording::ReplacementRecordingRequest::resolve_with_all_semantics(
                crate::replacement_recording::ReplacementRecordingInput {
                    transaction: TransactionId::new(5),
                    worker: reims_vgpu_core::RecordingWorkerId::new(0),
                    queue_family: context.gq,
                    exec: exec.clone(),
                    barriers: crate::replacement_barrier_record::NativeBarrierBatch::default(),
                },
                &EmptyBarrierResolver,
                &EmptyBarrierResolver,
                &[],
                &[],
                crate::replacement_recording::ReplacementSemanticAdmissions {
                    conditions: None,
                    completion_effects: None,
                    indirect_commands: Some(&admitted),
                    resource_states: None,
                    info_queries: &[],
                    indirect_range_programs: std::slice::from_ref(&mismatched_program),
                },
            )
            .unwrap_err();
        assert_eq!(
            mismatch.reason,
            crate::replacement_recording::ReplacementRecordingError::IndirectRangeProgramMismatch(
                0
            )
        );
        let request =
            crate::replacement_recording::ReplacementRecordingRequest::resolve_with_all_semantics(
                crate::replacement_recording::ReplacementRecordingInput {
                    transaction: TransactionId::new(5),
                    worker: reims_vgpu_core::RecordingWorkerId::new(0),
                    queue_family: context.gq,
                    exec,
                    barriers: crate::replacement_barrier_record::NativeBarrierBatch::default(),
                },
                &EmptyBarrierResolver,
                &EmptyBarrierResolver,
                &[],
                &[],
                crate::replacement_recording::ReplacementSemanticAdmissions {
                    conditions: None,
                    completion_effects: None,
                    indirect_commands: Some(&admitted),
                    resource_states: None,
                    info_queries: &[],
                    indirect_range_programs: std::slice::from_ref(&program),
                },
            )
            .unwrap();
        let workers = reims_vgpu_core::FixedExecutor::new(1, |worker| {
            crate::replacement_recording::ReplacementRecordingWorker::new(
                worker,
                &context.device,
                reims_vgpu_core::DescriptorTier::WorkerDescriptorPool,
            )
        })
        .unwrap();
        let mut recording =
            crate::replacement_recording::dispatch_replacement_recording(&workers, request)
                .unwrap()
                .wait()
                .unwrap();
        assert_eq!(recording.indirect_range_programs.len(), 1);
        let queue = unsafe { context.device.get_device_queue(context.gq, 0) };
        let submit = vk::SubmitInfo::default().command_buffers(&recording.command_buffers);
        unsafe {
            context
                .device
                .queue_submit(queue, &[submit], recording.fence)
        }
        .unwrap();
        unsafe {
            context
                .device
                .wait_for_fences(&[recording.fence], true, u64::MAX)
        }
        .unwrap();
        assert_eq!(
            recording.indirect_range_programs[0]
                .read_after_timeline()
                .unwrap(),
            [2, 0, 0, 0, 3, 0, 0, 0]
        );
        let continuation = resume_indirect_range_after_timeline(
            &icbs,
            pending,
            Box::new([readback]),
            RetiredReplacementIndirectRanges::after_explicit_timeline_wait(
                recording.take_indirect_range_programs_for(TransactionId::new(5)),
            ),
        )
        .unwrap();
        let reims_vgpu_core::NextIndirectRangeExecution::Final(final_phase) = continuation.next()
        else {
            panic!("one retired range leaves one range-free final phase")
        };
        let literal = final_phase
            .exec()
            .operations()
            .filter_map(|operation| match operation {
                reims_vgpu_core::ResolvedOperation::IndirectCommand(operation) => Some(*operation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            literal.as_slice(),
            &[reims_vgpu_core::ResolvedIndirectCommand::Execute {
                icb: reims_vgpu_protocol::ResourceId::new(3, 1),
                range: reims_vgpu_core::ResolvedIndirectCommandRange {
                    location: 2,
                    length: 3,
                },
                kind: reims_vgpu_core::IndirectCommandExecutionKind::Render,
            }]
        );
        let (cleanup_sender, cleanup_receiver) = std::sync::mpsc::sync_channel(1);
        crate::replacement_replay::recycle_replacement_recording(
            &workers,
            recording,
            move |worker, recording| {
                let _ = cleanup_sender.send(worker.recycle(recording));
            },
        )
        .unwrap();
        cleanup_receiver.recv().unwrap().unwrap();
        drop(program);
        unsafe {
            context.device.unmap_memory(memory);
            context.device.destroy_buffer(source, None);
            context.device.free_memory(memory, None);
        }
        drop(workers);
        drop(context);
    }
}
