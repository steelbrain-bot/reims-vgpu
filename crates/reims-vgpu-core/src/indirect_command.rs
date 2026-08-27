//! Resolution of indirect-command-buffer mutations.
//!
//! Command indices are validated against the live ICB declaration while its
//! serializer reference is replaced by a generational identity. Command-byte
//! storage is deliberately not reconstructed here: reset/copy execution needs
//! a separately established backing association, and absence of that contract
//! is an execution refusal rather than permission to infer one from an address.

use crate::{ExecTransaction, LinearRange, ResolvedOperation};
use reims_vgpu_protocol::{
    BackingId, IcbCommandMemory, IndirectCommandBufferDescriptor, IndirectCommandBufferObject,
    IndirectCommandOperation, IndirectCommandRange, ObjectTableRef, ResourceId, ResourceObject,
    SerializerRef, TaskId, TransactionId,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedIndirectCommandRange {
    pub location: u64,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectCommandMemoryReadPlan {
    pub gva: u64,
    pub byte_len: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandMemoryReadError {
    ZeroCommandSize,
    RangeOverflow,
    RangePastCapacity,
    CommandByteStartOverflow,
    CommandByteEndOverflow,
    CommandRangePastMemory,
    GuestAddressOverflow,
    CommandMemoryUnbound,
}

/// Resolve the exact guest-memory window containing an ICB command range.
/// Empty ranges are validated against the declaration but require no bound
/// memory and return no read plan.
pub fn resolve_indirect_command_memory_read(
    descriptor: &IndirectCommandBufferDescriptor,
    memory: Option<IcbCommandMemory>,
    range: ResolvedIndirectCommandRange,
) -> Result<Option<IndirectCommandMemoryReadPlan>, IndirectCommandMemoryReadError> {
    if descriptor.layout.command_size == 0 {
        return Err(IndirectCommandMemoryReadError::ZeroCommandSize);
    }
    let end = range
        .location
        .checked_add(range.length)
        .ok_or(IndirectCommandMemoryReadError::RangeOverflow)?;
    if end > u64::from(descriptor.max_command_count) {
        return Err(IndirectCommandMemoryReadError::RangePastCapacity);
    }
    if range.length == 0 {
        return Ok(None);
    }
    let memory = memory.ok_or(IndirectCommandMemoryReadError::CommandMemoryUnbound)?;
    let command_size = u64::from(descriptor.layout.command_size);
    let byte_start = range
        .location
        .checked_mul(command_size)
        .ok_or(IndirectCommandMemoryReadError::CommandByteStartOverflow)?;
    let byte_end = end
        .checked_mul(command_size)
        .ok_or(IndirectCommandMemoryReadError::CommandByteEndOverflow)?;
    if byte_end > memory.byte_len {
        return Err(IndirectCommandMemoryReadError::CommandRangePastMemory);
    }
    let gva = memory
        .gva
        .checked_add(byte_start)
        .ok_or(IndirectCommandMemoryReadError::GuestAddressOverflow)?;
    Ok(Some(IndirectCommandMemoryReadPlan {
        gva,
        byte_len: byte_end - byte_start,
    }))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedIndirectCommand {
    Optimize {
        icb: ResourceId<IndirectCommandBufferObject>,
        range: ResolvedIndirectCommandRange,
    },
    Reset {
        icb: ResourceId<IndirectCommandBufferObject>,
        range: ResolvedIndirectCommandRange,
    },
    Copy {
        source: ResourceId<IndirectCommandBufferObject>,
        source_range: ResolvedIndirectCommandRange,
        destination: ResourceId<IndirectCommandBufferObject>,
        destination_index: u64,
    },
    Execute {
        icb: ResourceId<IndirectCommandBufferObject>,
        range: ResolvedIndirectCommandRange,
        kind: IndirectCommandExecutionKind,
    },
    ExecuteIndirectRange {
        icb: ResourceId<IndirectCommandBufferObject>,
        arguments_resource: ResourceId<ResourceObject>,
        arguments_backing: BackingId,
        arguments_range: LinearRange,
        kind: IndirectCommandExecutionKind,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandExecutionKind {
    Render,
    Compute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandKind {
    Optimize,
    Reset,
    Copy,
    Execute,
}

impl ResolvedIndirectCommand {
    pub const fn kind(self) -> IndirectCommandKind {
        match self {
            Self::Optimize { .. } => IndirectCommandKind::Optimize,
            Self::Reset { .. } => IndirectCommandKind::Reset,
            Self::Copy { .. } => IndirectCommandKind::Copy,
            Self::Execute { .. } => IndirectCommandKind::Execute,
            Self::ExecuteIndirectRange { .. } => IndirectCommandKind::Execute,
        }
    }
}

/// Exact transaction-bound proof that an ICB optimization hint was consumed.
/// Reset and copy cannot enter this value until command storage has a canonical
/// backing identity and an ordered mutation owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedIndirectCommands<Operation> {
    transaction: TransactionId,
    operations: Box<[(usize, Operation)]>,
}

impl<Operation> AdmittedIndirectCommands<Operation> {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub fn operations(&self) -> &[(usize, Operation)] {
        &self.operations
    }
}

impl<Operation: Clone> AdmittedIndirectCommands<Operation> {
    /// Retain only proofs belonging to one globally positioned continuation
    /// phase. Recorder matching still requires every operation in that phase.
    pub fn for_operation_range(&self, start: usize, end: usize) -> Self {
        Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .filter(|(index, _)| *index >= start && *index < end)
                .cloned()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    /// Select one continuation phase and express its positions relative to
    /// the phase EXEC. Recording later restores `start` as its operation base.
    pub fn for_operation_range_rebased(&self, start: usize, end: usize) -> Self {
        Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .filter(|(index, _)| *index >= start && *index < end)
                .map(|(index, operation)| (index - start, operation.clone()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn shifted(&self, base: usize) -> Option<Self> {
        Some(Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .map(|(index, operation)| Some((index.checked_add(base)?, operation.clone())))
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
                .map(|(index, operation)| Some((*positions.get(*index)?, operation.clone())))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }
}

impl AdmittedIndirectCommands<ResolvedIndirectCommand> {
    /// Retain the semantic mutations which remain in an EXEC after literal
    /// execution operations have been replaced by their snapshotted commands.
    /// Positions are unchanged when no replacement expands the stream; a
    /// phased caller must additionally select its original operation range.
    pub fn without_literal_executions(&self) -> Self {
        Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .filter(|(_, operation)| {
                    !matches!(operation, ResolvedIndirectCommand::Execute { .. })
                })
                .copied()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectCommandAdmissionError {
    pub index: usize,
    pub kind: IndirectCommandKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectCommandCommitError {
    pub index: usize,
    pub kind: IndirectCommandKind,
    pub reason: IndirectCommandMutationError,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedIndirectCommands<Command> {
    pub transaction: TransactionId,
    pub operations: Box<[(usize, ResolvedIndirectCommand)]>,
    pub executions: Box<[CommittedIndirectExecution<Command>]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedIndirectExecution<Command> {
    pub index: usize,
    pub kind: IndirectCommandExecutionKind,
    pub icb: ResourceId<IndirectCommandBufferObject>,
    pub range: ResolvedIndirectCommandRange,
    pub slots: PriorIndirectCommandPopulation<Command>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectExecutionExpansionError {
    TransactionMismatch,
    MissingExecution(usize),
    ExecutionMismatch(usize),
    UnexpectedExecution(usize),
    CommandKindMismatch { operation: usize, slot: u64 },
}

pub type IndirectExecutionOperation<Render, Compute, Info, Completion> =
    ResolvedOperation<Render, Compute, Info, ResolvedIndirectCommand, Completion>;
pub type IndirectExecutionTransaction<Render, Compute, Info, Completion> =
    ExecTransaction<IndirectExecutionOperation<Render, Compute, Info, Completion>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectExecutionPreparationError {
    Commit(IndirectCommandCommitError),
    Expansion(IndirectExecutionExpansionError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndirectExecutionPreparationFailure {
    pub reason: IndirectExecutionPreparationError,
    pub admitted: AdmittedIndirectCommands<ResolvedIndirectCommand>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpandedIndirectOperationOrigin {
    pub expanded_position: usize,
    pub original_position: usize,
    pub indirect_slot: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PreparedIndirectExecution<Render, Compute, Info, Completion> {
    committed: CommittedIndirectCommands<ResolvedIndirectCommandSlot<Render, Compute>>,
    exec: IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    origins: Box<[ExpandedIndirectOperationOrigin]>,
}

impl<Render, Compute, Info, Completion>
    PreparedIndirectExecution<Render, Compute, Info, Completion>
{
    pub const fn committed(
        &self,
    ) -> &CommittedIndirectCommands<ResolvedIndirectCommandSlot<Render, Compute>> {
        &self.committed
    }

    pub const fn exec(&self) -> &IndirectExecutionTransaction<Render, Compute, Info, Completion> {
        &self.exec
    }

    pub const fn origins(&self) -> &[ExpandedIndirectOperationOrigin] {
        &self.origins
    }

    pub fn into_exec(self) -> IndirectExecutionTransaction<Render, Compute, Info, Completion> {
        self.exec
    }

    /// Keep phase-local expanded positions while restoring each origin to its
    /// position in the complete admitted EXEC.
    pub fn shifted_original_positions(mut self, base: usize) -> Option<Self> {
        for origin in &mut self.origins {
            origin.original_position = origin.original_position.checked_add(base)?;
        }
        Some(self)
    }

    /// Restore both identities carried by a continuation phase.
    ///
    /// `original_position` addresses semantic admissions derived from the
    /// source EXEC. `expanded_position` is the unique operation identity used
    /// by resource preparation after one literal execution becomes zero or
    /// more native operations.
    pub fn shifted_positions(mut self, original_base: usize, expanded_base: usize) -> Option<Self> {
        for origin in &mut self.origins {
            origin.original_position = origin.original_position.checked_add(original_base)?;
            origin.expanded_position = origin.expanded_position.checked_add(expanded_base)?;
        }
        Some(self)
    }
}

/// Apply ordered ICB mutations and expand execution snapshots atomically.
///
/// A staged owner absorbs all mutations first. Slot-kind or expansion refusal
/// returns the exact admission and leaves the live owner unchanged.
pub fn prepare_indirect_execution<Render: Clone, Compute: Clone, Info: Clone, Completion: Clone>(
    owner: &mut IndirectCommandSlotOwner<ResolvedIndirectCommandSlot<Render, Compute>>,
    transaction: TransactionId,
    exec: &IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    admitted: AdmittedIndirectCommands<ResolvedIndirectCommand>,
) -> Result<
    PreparedIndirectExecution<Render, Compute, Info, Completion>,
    IndirectExecutionPreparationFailure,
> {
    let mut staged = owner.clone();
    let committed = commit_indirect_commands(&mut staged, admitted.clone()).map_err(|reason| {
        IndirectExecutionPreparationFailure {
            reason: IndirectExecutionPreparationError::Commit(reason),
            admitted: admitted.clone(),
        }
    })?;
    let (expanded, origins) =
        expand_committed_indirect_executions_with_origins(transaction, exec, &committed).map_err(
            |reason| IndirectExecutionPreparationFailure {
                reason: IndirectExecutionPreparationError::Expansion(reason),
                admitted,
            },
        )?;
    *owner = staged;
    Ok(PreparedIndirectExecution {
        committed,
        exec: expanded,
        origins,
    })
}

/// Replace committed ICB execution operations with their position-snapshotted
/// canonical render or compute commands.
///
/// Empty slots emit no work. Mutation and optimization operations remain at
/// their original positions as semantic operations; only `Execute` expands.
pub fn expand_committed_indirect_executions<
    Render: Clone,
    Compute: Clone,
    Info: Clone,
    Completion: Clone,
>(
    transaction: TransactionId,
    exec: &IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    committed: &CommittedIndirectCommands<ResolvedIndirectCommandSlot<Render, Compute>>,
) -> Result<
    IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    IndirectExecutionExpansionError,
> {
    expand_committed_indirect_executions_with_origins(transaction, exec, committed)
        .map(|(expanded, _)| expanded)
}

type ExpandedIndirectExecutionWithOrigins<Render, Compute, Info, Completion> = (
    IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    Box<[ExpandedIndirectOperationOrigin]>,
);

fn expand_committed_indirect_executions_with_origins<
    Render: Clone,
    Compute: Clone,
    Info: Clone,
    Completion: Clone,
>(
    transaction: TransactionId,
    exec: &IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    committed: &CommittedIndirectCommands<ResolvedIndirectCommandSlot<Render, Compute>>,
) -> Result<
    ExpandedIndirectExecutionWithOrigins<Render, Compute, Info, Completion>,
    IndirectExecutionExpansionError,
> {
    if committed.transaction != transaction {
        return Err(IndirectExecutionExpansionError::TransactionMismatch);
    }
    let executions = committed
        .executions
        .iter()
        .map(|execution| (execution.index, execution))
        .collect::<BTreeMap<_, _>>();
    let mut consumed = std::collections::BTreeSet::new();
    let mut origins = (0..exec.prologue.operations().len())
        .map(|position| ExpandedIndirectOperationOrigin {
            expanded_position: position,
            original_position: position,
            indirect_slot: None,
        })
        .collect::<Vec<_>>();
    let mut expanded_position = origins.len();
    let mut original_index = exec.prologue.operations().len();
    let streams = exec
        .streams
        .iter()
        .map(|stream| {
            let segments = stream
                .segments
                .iter()
                .map(|segment| {
                    let mut operations = Vec::new();
                    for operation in segment.operations.iter() {
                        match operation {
                            ResolvedOperation::IndirectCommand(
                                ResolvedIndirectCommand::Execute { icb, range, kind },
                            ) => {
                                let execution = executions.get(&original_index).ok_or(
                                    IndirectExecutionExpansionError::MissingExecution(
                                        original_index,
                                    ),
                                )?;
                                if execution.icb != *icb
                                    || execution.range != *range
                                    || execution.kind != *kind
                                {
                                    return Err(
                                        IndirectExecutionExpansionError::ExecutionMismatch(
                                            original_index,
                                        ),
                                    );
                                }
                                consumed.insert(original_index);
                                for (slot, command) in execution.slots.iter() {
                                    match (kind, command) {
                                        (_, None) => {}
                                        (
                                            IndirectCommandExecutionKind::Render,
                                            Some(ResolvedIndirectCommandSlot::Render(render)),
                                        ) => {
                                            operations.push(ResolvedOperation::Render(render.clone()));
                                            origins.push(ExpandedIndirectOperationOrigin {
                                                expanded_position,
                                                original_position: original_index,
                                                indirect_slot: Some(*slot),
                                            });
                                            expanded_position += 1;
                                        }
                                        (
                                            IndirectCommandExecutionKind::Compute,
                                            Some(ResolvedIndirectCommandSlot::Compute(compute)),
                                        ) => {
                                            operations.push(ResolvedOperation::Compute(compute.clone()));
                                            origins.push(ExpandedIndirectOperationOrigin {
                                                expanded_position,
                                                original_position: original_index,
                                                indirect_slot: Some(*slot),
                                            });
                                            expanded_position += 1;
                                        }
                                        _ => {
                                            return Err(
                                                IndirectExecutionExpansionError::CommandKindMismatch {
                                                    operation: original_index,
                                                    slot: *slot,
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                            operation => {
                                operations.push(operation.clone());
                                origins.push(ExpandedIndirectOperationOrigin {
                                    expanded_position,
                                    original_position: original_index,
                                    indirect_slot: None,
                                });
                                expanded_position += 1;
                            }
                        }
                        original_index += 1;
                    }
                    Ok(crate::ResolvedExecSegment {
                        boundary: segment.boundary,
                        operations: operations.into_boxed_slice(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(crate::ResolvedExecStream {
                stream_index: stream.stream_index,
                segments: segments.into_boxed_slice(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(index) = executions
        .keys()
        .copied()
        .find(|index| !consumed.contains(index))
    {
        return Err(IndirectExecutionExpansionError::UnexpectedExecution(index));
    }
    Ok((
        ExecTransaction {
            identity: exec.identity,
            prologue: exec.prologue.clone(),
            streams: streams.into_boxed_slice(),
            accesses: exec.accesses.clone(),
        },
        origins.into_boxed_slice(),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandMutationError {
    DuplicateBuffer,
    DuplicateSlot,
    UnknownBuffer(ResourceId<IndirectCommandBufferObject>),
    SlotPastCapacity,
    SourceRangePastCapacity,
    DestinationRangePastCapacity,
    IndirectRangeResolutionRequired,
}

/// One immutable command stored in a live indirect-command-buffer slot.
///
/// Slot bytes are decoded and all task-local names are resolved before this
/// value is constructed. Execution therefore consumes the same canonical
/// render and compute vocabulary as direct encoder work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedIndirectCommandSlot<Render, Compute> {
    Render(Render),
    Compute(Compute),
}

/// A complete decoded slot population. `None` is an explicit empty command,
/// not an omitted update: publishing it removes any command previously stored
/// at that slot.
pub type IndirectCommandPopulation<Command> = Box<[(u64, Option<Command>)]>;
pub type PriorIndirectCommandPopulation<Command> = Box<[(u64, Option<Command>)]>;
pub type DecodedIndirectCommandPopulation<Input> = Box<[(u64, Option<Input>)]>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndirectCommandPopulationFailure<Command> {
    pub reason: IndirectCommandMutationError,
    pub commands: IndirectCommandPopulation<Command>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndirectCommandPopulationResolutionError<E> {
    Resolve { index: u64, reason: E },
    Population(IndirectCommandMutationError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndirectCommandPopulationResolutionFailure<Input, E> {
    pub reason: IndirectCommandPopulationResolutionError<E>,
    pub commands: DecodedIndirectCommandPopulation<Input>,
}

#[derive(Clone, Debug)]
struct IndirectCommandSlots<Command> {
    capacity: u32,
    slots: BTreeMap<u64, Command>,
}

/// Command slots owned by exact ICB generations.
///
/// Sparse storage is unbounded by policy: it grows only for command indices
/// the guest's declared capacity admits, and the complete map dies with that
/// ICB generation. An absent entry is an empty indirect command.
#[derive(Clone, Debug)]
pub struct IndirectCommandSlotOwner<Command> {
    buffers: BTreeMap<ResourceId<IndirectCommandBufferObject>, IndirectCommandSlots<Command>>,
}

impl<Command> Default for IndirectCommandSlotOwner<Command> {
    fn default() -> Self {
        Self {
            buffers: BTreeMap::new(),
        }
    }
}

impl<Command> IndirectCommandSlotOwner<Command> {
    pub fn contains(&self, icb: ResourceId<IndirectCommandBufferObject>) -> bool {
        self.buffers.contains_key(&icb)
    }

    pub fn register(
        &mut self,
        icb: ResourceId<IndirectCommandBufferObject>,
        capacity: u32,
    ) -> Result<(), IndirectCommandMutationError> {
        if self.buffers.contains_key(&icb) {
            return Err(IndirectCommandMutationError::DuplicateBuffer);
        }
        self.buffers.insert(
            icb,
            IndirectCommandSlots {
                capacity,
                slots: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub fn retire(
        &mut self,
        icb: ResourceId<IndirectCommandBufferObject>,
    ) -> Result<Box<[(u64, Command)]>, IndirectCommandMutationError> {
        self.buffers
            .remove(&icb)
            .map(|slots| {
                slots
                    .slots
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .ok_or(IndirectCommandMutationError::UnknownBuffer(icb))
    }

    pub fn set(
        &mut self,
        icb: ResourceId<IndirectCommandBufferObject>,
        index: u64,
        command: Command,
    ) -> Result<Option<Command>, IndirectCommandMutationError> {
        let slots = self
            .buffers
            .get_mut(&icb)
            .ok_or(IndirectCommandMutationError::UnknownBuffer(icb))?;
        if index >= u64::from(slots.capacity) {
            return Err(IndirectCommandMutationError::SlotPastCapacity);
        }
        Ok(slots.slots.insert(index, command))
    }

    /// Replace a complete decoded population batch atomically.
    ///
    /// Every position and the exact ICB generation are checked before any
    /// live slot changes, so a malformed suffix returns the full input batch.
    pub fn set_batch(
        &mut self,
        icb: ResourceId<IndirectCommandBufferObject>,
        commands: IndirectCommandPopulation<Command>,
    ) -> Result<PriorIndirectCommandPopulation<Command>, IndirectCommandPopulationFailure<Command>>
    {
        let Some(slots) = self.buffers.get(&icb) else {
            return Err(IndirectCommandPopulationFailure {
                reason: IndirectCommandMutationError::UnknownBuffer(icb),
                commands,
            });
        };
        let mut indices = std::collections::BTreeSet::new();
        for (index, _) in commands.iter() {
            if *index >= u64::from(slots.capacity) {
                return Err(IndirectCommandPopulationFailure {
                    reason: IndirectCommandMutationError::SlotPastCapacity,
                    commands,
                });
            }
            if !indices.insert(*index) {
                return Err(IndirectCommandPopulationFailure {
                    reason: IndirectCommandMutationError::DuplicateSlot,
                    commands,
                });
            }
        }
        let slots = &mut self
            .buffers
            .get_mut(&icb)
            .expect("the exact ICB generation was prevalidated")
            .slots;
        Ok(commands
            .into_vec()
            .into_iter()
            .map(|(index, command)| {
                let prior = match command {
                    Some(command) => slots.insert(index, command),
                    None => slots.remove(&index),
                };
                (index, prior)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    pub fn get(
        &self,
        icb: ResourceId<IndirectCommandBufferObject>,
        index: u64,
    ) -> Result<Option<&Command>, IndirectCommandMutationError> {
        let slots = self
            .buffers
            .get(&icb)
            .ok_or(IndirectCommandMutationError::UnknownBuffer(icb))?;
        if index >= u64::from(slots.capacity) {
            return Err(IndirectCommandMutationError::SlotPastCapacity);
        }
        Ok(slots.slots.get(&index))
    }

    pub fn validate(
        &self,
        operation: ResolvedIndirectCommand,
    ) -> Result<(), IndirectCommandMutationError> {
        match operation {
            ResolvedIndirectCommand::Optimize { icb, range }
            | ResolvedIndirectCommand::Reset { icb, range } => {
                self.validate_range(icb, range, false)
            }
            ResolvedIndirectCommand::Copy {
                source,
                source_range,
                destination,
                destination_index,
            } => {
                self.validate_range(source, source_range, false)?;
                let destination_length = ResolvedIndirectCommandRange {
                    location: destination_index,
                    length: source_range.length,
                };
                self.validate_range(destination, destination_length, true)
            }
            ResolvedIndirectCommand::Execute { icb, range, .. } => {
                self.validate_range(icb, range, false)
            }
            ResolvedIndirectCommand::ExecuteIndirectRange { icb, .. } => {
                if !self.buffers.contains_key(&icb) {
                    Err(IndirectCommandMutationError::UnknownBuffer(icb))
                } else {
                    Err(IndirectCommandMutationError::IndirectRangeResolutionRequired)
                }
            }
        }
    }

    pub fn validate_indirect_range_buffer(
        &self,
        icb: ResourceId<IndirectCommandBufferObject>,
    ) -> Result<(), IndirectCommandMutationError> {
        self.buffers
            .contains_key(&icb)
            .then_some(())
            .ok_or(IndirectCommandMutationError::UnknownBuffer(icb))
    }

    fn validate_range(
        &self,
        icb: ResourceId<IndirectCommandBufferObject>,
        range: ResolvedIndirectCommandRange,
        destination: bool,
    ) -> Result<(), IndirectCommandMutationError> {
        let slots = self
            .buffers
            .get(&icb)
            .ok_or(IndirectCommandMutationError::UnknownBuffer(icb))?;
        let end = range
            .location
            .checked_add(range.length)
            .ok_or(if destination {
                IndirectCommandMutationError::DestinationRangePastCapacity
            } else {
                IndirectCommandMutationError::SourceRangePastCapacity
            })?;
        if end > u64::from(slots.capacity) {
            return Err(if destination {
                IndirectCommandMutationError::DestinationRangePastCapacity
            } else {
                IndirectCommandMutationError::SourceRangePastCapacity
            });
        }
        Ok(())
    }
}

/// Resolve one complete decoded slot batch against one immutable encoder-state
/// snapshot, then publish it atomically to the exact ICB generation.
///
/// Resolution borrows every decoded record. A bad suffix therefore preserves
/// the complete input, and the slot owner is not touched until all records have
/// become canonical commands.
pub fn resolve_indirect_command_population<Input, State, Command, E>(
    owner: &mut IndirectCommandSlotOwner<Command>,
    icb: ResourceId<IndirectCommandBufferObject>,
    state: &State,
    commands: DecodedIndirectCommandPopulation<Input>,
    mut resolve: impl FnMut(&State, &Input) -> Result<Option<Command>, E>,
) -> Result<
    PriorIndirectCommandPopulation<Command>,
    IndirectCommandPopulationResolutionFailure<Input, E>,
> {
    let mut resolved = Vec::with_capacity(commands.len());
    for (index, input) in commands.iter() {
        let Some(input) = input else {
            resolved.push((*index, None));
            continue;
        };
        let command = match resolve(state, input) {
            Ok(command) => command,
            Err(reason) => {
                return Err(IndirectCommandPopulationResolutionFailure {
                    reason: IndirectCommandPopulationResolutionError::Resolve {
                        index: *index,
                        reason,
                    },
                    commands,
                });
            }
        };
        // A decoded command whose contract is an empty draw/dispatch resolves
        // to `None` deliberately. It clears an older population just like a
        // reset slot, while remaining distinct from resolution failure.
        resolved.push((*index, command));
    }
    owner
        .set_batch(icb, resolved.into_boxed_slice())
        .map_err(|failure| IndirectCommandPopulationResolutionFailure {
            reason: IndirectCommandPopulationResolutionError::Population(failure.reason),
            commands,
        })
}

impl<Command: Clone> IndirectCommandSlotOwner<Command> {
    /// Snapshot every slot in a validated execution range, including empty
    /// positions, before any later reset, copy, or population can change it.
    pub fn snapshot_range(
        &self,
        icb: ResourceId<IndirectCommandBufferObject>,
        range: ResolvedIndirectCommandRange,
    ) -> Result<PriorIndirectCommandPopulation<Command>, IndirectCommandMutationError> {
        self.validate_range(icb, range, false)?;
        let slots = &self
            .buffers
            .get(&icb)
            .expect("the exact ICB generation was prevalidated")
            .slots;
        Ok((range.location..range.location + range.length)
            .map(|index| (index, slots.get(&index).cloned()))
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    pub fn apply(
        &mut self,
        operation: ResolvedIndirectCommand,
    ) -> Result<(), IndirectCommandMutationError> {
        self.validate(operation)?;
        match operation {
            ResolvedIndirectCommand::Optimize { .. } => {}
            ResolvedIndirectCommand::Reset { icb, range } => {
                let slots = self
                    .buffers
                    .get_mut(&icb)
                    .expect("the exact ICB generation was prevalidated");
                for index in range.location..range.location + range.length {
                    slots.slots.remove(&index);
                }
            }
            ResolvedIndirectCommand::Copy {
                source,
                source_range,
                destination,
                destination_index,
            } => {
                let snapshot = {
                    let source = self
                        .buffers
                        .get(&source)
                        .expect("the source ICB generation was prevalidated");
                    (0..source_range.length)
                        .map(|offset| source.slots.get(&(source_range.location + offset)).cloned())
                        .collect::<Vec<_>>()
                };
                let destination = self
                    .buffers
                    .get_mut(&destination)
                    .expect("the destination ICB generation was prevalidated");
                for (offset, command) in snapshot.into_iter().enumerate() {
                    let index = destination_index
                        + u64::try_from(offset).expect("range length already fits u64");
                    match command {
                        Some(command) => {
                            destination.slots.insert(index, command);
                        }
                        None => {
                            destination.slots.remove(&index);
                        }
                    }
                }
            }
            ResolvedIndirectCommand::Execute { .. } => {}
            ResolvedIndirectCommand::ExecuteIndirectRange { .. } => {
                unreachable!("indirect range execution is refused during validation")
            }
        }
        Ok(())
    }
}

pub fn commit_indirect_commands<Command: Clone>(
    owner: &mut IndirectCommandSlotOwner<Command>,
    admitted: AdmittedIndirectCommands<ResolvedIndirectCommand>,
) -> Result<CommittedIndirectCommands<Command>, IndirectCommandCommitError> {
    for (index, operation) in admitted.operations.iter().copied() {
        let validation = match operation {
            ResolvedIndirectCommand::ExecuteIndirectRange { icb, .. } => {
                owner.validate_indirect_range_buffer(icb)
            }
            operation => owner.validate(operation),
        };
        if let Err(reason) = validation {
            return Err(IndirectCommandCommitError {
                index,
                kind: operation.kind(),
                reason,
            });
        }
    }
    let mut executions = Vec::new();
    for (index, operation) in admitted.operations.iter().copied() {
        match operation {
            ResolvedIndirectCommand::Execute { icb, range, kind } => {
                executions.push(CommittedIndirectExecution {
                    index,
                    kind,
                    icb,
                    range,
                    slots: owner
                        .snapshot_range(icb, range)
                        .expect("the complete ICB batch was prevalidated"),
                });
            }
            ResolvedIndirectCommand::ExecuteIndirectRange { .. } => {}
            _ => owner
                .apply(operation)
                .expect("the complete ICB mutation batch was prevalidated"),
        }
    }
    Ok(CommittedIndirectCommands {
        transaction: admitted.transaction,
        operations: admitted.operations,
        executions: executions.into_boxed_slice(),
    })
}

pub fn admit_indirect_commands_with_owner<Render, Compute, Info, Command, Completion>(
    transaction: TransactionId,
    exec: &ExecTransaction<
        ResolvedOperation<Render, Compute, Info, ResolvedIndirectCommand, Completion>,
    >,
    owner: &IndirectCommandSlotOwner<Command>,
) -> Result<AdmittedIndirectCommands<ResolvedIndirectCommand>, IndirectCommandAdmissionError> {
    let mut operations = Vec::new();
    for (index, operation) in exec.operations().enumerate() {
        let ResolvedOperation::IndirectCommand(operation) = operation else {
            continue;
        };
        let validation = match *operation {
            ResolvedIndirectCommand::ExecuteIndirectRange { icb, .. } => {
                owner.validate_indirect_range_buffer(icb)
            }
            operation => owner.validate(operation),
        };
        if validation.is_err() {
            return Err(IndirectCommandAdmissionError {
                index,
                kind: operation.kind(),
            });
        }
        operations.push((index, *operation));
    }
    Ok(AdmittedIndirectCommands {
        transaction,
        operations: operations.into_boxed_slice(),
    })
}

pub fn admit_indirect_commands<Render, Compute, Info, Completion>(
    transaction: TransactionId,
    exec: &ExecTransaction<
        ResolvedOperation<Render, Compute, Info, ResolvedIndirectCommand, Completion>,
    >,
) -> Result<AdmittedIndirectCommands<ResolvedIndirectCommand>, IndirectCommandAdmissionError> {
    let mut operations = Vec::new();
    for (index, operation) in exec.operations().enumerate() {
        let ResolvedOperation::IndirectCommand(operation) = operation else {
            continue;
        };
        match operation {
            ResolvedIndirectCommand::Optimize { .. } => operations.push((index, *operation)),
            ResolvedIndirectCommand::Reset { .. }
            | ResolvedIndirectCommand::Copy { .. }
            | ResolvedIndirectCommand::Execute { .. }
            | ResolvedIndirectCommand::ExecuteIndirectRange { .. } => {
                return Err(IndirectCommandAdmissionError {
                    index,
                    kind: operation.kind(),
                });
            }
        }
    }
    Ok(AdmittedIndirectCommands {
        transaction,
        operations: operations.into_boxed_slice(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectCommandBufferResolution {
    pub identity: ResourceId<IndirectCommandBufferObject>,
    pub max_command_count: u32,
}

pub trait IndirectCommandResolver {
    fn resolve_indirect_command_buffer(
        &self,
        task: TaskId,
        reference: SerializerRef<IndirectCommandBufferObject>,
    ) -> Option<IndirectCommandBufferResolution>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectExecutionArgumentsResolution {
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub size: u64,
}

/// Exact continuation owned while a GPU-produced ICB execution range is copied
/// into timeline-retired host-visible storage.
#[derive(Debug)]
pub struct PreparedIndirectRangeReadback {
    transaction: TransactionId,
    operation_index: usize,
    icb: ResourceId<IndirectCommandBufferObject>,
    arguments_resource: ResourceId<ResourceObject>,
    arguments_backing: BackingId,
    arguments_representation: reims_vgpu_protocol::RepresentationId,
    arguments_range: LinearRange,
    kind: IndirectCommandExecutionKind,
    uses: Box<[crate::RepresentationUse]>,
}

impl PreparedIndirectRangeReadback {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn operation_index(&self) -> usize {
        self.operation_index
    }

    pub const fn arguments_resource(&self) -> ResourceId<ResourceObject> {
        self.arguments_resource
    }

    pub const fn arguments_backing(&self) -> BackingId {
        self.arguments_backing
    }

    pub const fn arguments_representation(&self) -> reims_vgpu_protocol::RepresentationId {
        self.arguments_representation
    }

    pub const fn arguments_range(&self) -> LinearRange {
        self.arguments_range
    }

    pub const fn uses(&self) -> &[crate::RepresentationUse] {
        &self.uses
    }

    pub const fn operation(&self) -> ResolvedIndirectCommand {
        ResolvedIndirectCommand::ExecuteIndirectRange {
            icb: self.icb,
            arguments_resource: self.arguments_resource,
            arguments_backing: self.arguments_backing,
            arguments_range: self.arguments_range,
            kind: self.kind,
        }
    }
}

/// Whether one indirect-operation vocabulary member requires and matches an
/// asynchronous range readback in the whole-EXEC resource envelope.
pub trait IndirectRangeResourceOperation {
    fn requires_range_readback(&self) -> bool;
    fn matches_range_readback(&self, readback: &PreparedIndirectRangeReadback) -> bool;
}

impl IndirectRangeResourceOperation for ResolvedIndirectCommand {
    fn requires_range_readback(&self) -> bool {
        matches!(self, Self::ExecuteIndirectRange { .. })
    }

    fn matches_range_readback(&self, readback: &PreparedIndirectRangeReadback) -> bool {
        *self == readback.operation()
    }
}

impl IndirectRangeResourceOperation for () {
    fn requires_range_readback(&self) -> bool {
        false
    }

    fn matches_range_readback(&self, _: &PreparedIndirectRangeReadback) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectRangeReadbackError {
    LiteralExecutionRequired,
    Arguments(crate::ManagedBackingError),
    Uses(crate::ResourceUseBatchError),
    Owner(IndirectCommandMutationError),
}

#[derive(Debug)]
pub struct IndirectRangeReadbackFailure {
    pub reason: IndirectRangeReadbackError,
    pub readback: PreparedIndirectRangeReadback,
    pub bytes: [u8; 8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectRangeReadbackBatchError {
    CountMismatch {
        readbacks: usize,
        bytes: usize,
    },
    Readback {
        index: usize,
        reason: IndirectRangeReadbackError,
    },
}

#[derive(Debug)]
pub struct IndirectRangeReadbackBatchFailure {
    pub reason: IndirectRangeReadbackBatchError,
    pub readbacks: Box<[PreparedIndirectRangeReadback]>,
    pub bytes: Box<[[u8; 8]]>,
}

/// One native submission phase of an EXEC containing GPU-produced ICB ranges.
///
/// A readback phase includes the exact unresolved range operation as its last
/// operation. Its successor starts at that same position after the operation
/// has become literal, so work before the range is never replayed.
#[derive(Clone, Debug)]
pub struct IndirectRangeExecutionPhase<Render, Compute, Info, Completion> {
    transaction: TransactionId,
    operation_base: usize,
    exec: IndirectExecutionTransaction<Render, Compute, Info, Completion>,
}

impl<Render, Compute, Info, Completion>
    IndirectRangeExecutionPhase<Render, Compute, Info, Completion>
{
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn operation_base(&self) -> usize {
        self.operation_base
    }

    pub const fn exec(&self) -> &IndirectExecutionTransaction<Render, Compute, Info, Completion> {
        &self.exec
    }

    pub fn into_exec(self) -> IndirectExecutionTransaction<Render, Compute, Info, Completion> {
        self.exec
    }
}

#[derive(Clone, Debug)]
pub struct IndirectRangeExecutionContinuation<Render, Compute, Info, Completion> {
    transaction: TransactionId,
    cursor: usize,
    exec: IndirectExecutionTransaction<Render, Compute, Info, Completion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectRangeExecutionStartError {
    OperationBasePastEnd { base: usize, operations: usize },
}

#[derive(Clone, Debug)]
pub struct PendingIndirectRangeExecution<Render, Compute, Info, Completion> {
    operation_index: usize,
    operation: ResolvedIndirectCommand,
    phase: IndirectRangeExecutionPhase<Render, Compute, Info, Completion>,
    continuation: IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>,
}

impl<Render, Compute, Info, Completion>
    PendingIndirectRangeExecution<Render, Compute, Info, Completion>
{
    pub const fn operation_index(&self) -> usize {
        self.operation_index
    }

    pub const fn operation(&self) -> ResolvedIndirectCommand {
        self.operation
    }

    pub const fn phase(&self) -> &IndirectRangeExecutionPhase<Render, Compute, Info, Completion> {
        &self.phase
    }
}

#[derive(Clone, Debug)]
pub enum NextIndirectRangeExecution<Render, Compute, Info, Completion> {
    Readback(PendingIndirectRangeExecution<Render, Compute, Info, Completion>),
    Final(IndirectRangeExecutionPhase<Render, Compute, Info, Completion>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectRangeExecutionResumeError {
    NotLiteralExecution,
    IdentityMismatch,
}

#[derive(Clone, Debug)]
pub struct IndirectRangeExecutionResumeFailure<Render, Compute, Info, Completion> {
    pub reason: IndirectRangeExecutionResumeError,
    pub pending: PendingIndirectRangeExecution<Render, Compute, Info, Completion>,
    pub literal: ResolvedIndirectCommand,
}

pub type IndirectRangeExecutionResumeResult<Render, Compute, Info, Completion> = Result<
    IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>,
    Box<IndirectRangeExecutionResumeFailure<Render, Compute, Info, Completion>>,
>;

impl<Render, Compute, Info, Completion>
    IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>
{
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }
}

impl<Render: Clone, Compute: Clone, Info: Clone, Completion: Clone>
    IndirectRangeExecutionContinuation<Render, Compute, Info, Completion>
{
    pub fn new(
        transaction: TransactionId,
        exec: IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    ) -> Self {
        Self {
            transaction,
            cursor: 0,
            exec,
        }
    }

    /// Start at an already executed prefix boundary of this exact EXEC.
    ///
    /// The caller must own the completion of every operation before `base`.
    /// Keeping the cursor in this owner makes replaying that prefix
    /// unrepresentable in subsequent range phases.
    pub fn after_prefix(
        transaction: TransactionId,
        exec: IndirectExecutionTransaction<Render, Compute, Info, Completion>,
        base: usize,
    ) -> Result<Self, IndirectRangeExecutionStartError> {
        let operations = exec.operations().count();
        if base > operations {
            return Err(IndirectRangeExecutionStartError::OperationBasePastEnd {
                base,
                operations,
            });
        }
        Ok(Self {
            transaction,
            cursor: base,
            exec,
        })
    }

    pub fn next(self) -> NextIndirectRangeExecution<Render, Compute, Info, Completion> {
        let unresolved = self
            .exec
            .operations()
            .enumerate()
            .skip(self.cursor)
            .find_map(|(index, operation)| match operation {
                ResolvedOperation::IndirectCommand(
                    operation @ ResolvedIndirectCommand::ExecuteIndirectRange { .. },
                ) => Some((index, *operation)),
                _ => None,
            });
        match unresolved {
            Some((operation_index, operation)) => {
                let phase = phase_exec(
                    self.transaction,
                    &self.exec,
                    self.cursor,
                    operation_index + 1,
                );
                NextIndirectRangeExecution::Readback(PendingIndirectRangeExecution {
                    operation_index,
                    operation,
                    phase,
                    continuation: self,
                })
            }
            None => NextIndirectRangeExecution::Final(phase_exec(
                self.transaction,
                &self.exec,
                self.cursor,
                self.exec.operations().count(),
            )),
        }
    }
}

pub fn resume_indirect_range_execution<Render, Compute, Info, Completion>(
    mut pending: PendingIndirectRangeExecution<Render, Compute, Info, Completion>,
    literal: ResolvedIndirectCommand,
) -> IndirectRangeExecutionResumeResult<Render, Compute, Info, Completion> {
    let ResolvedIndirectCommand::Execute {
        icb,
        range: _,
        kind,
    } = literal
    else {
        return Err(Box::new(IndirectRangeExecutionResumeFailure {
            reason: IndirectRangeExecutionResumeError::NotLiteralExecution,
            pending,
            literal,
        }));
    };
    let ResolvedIndirectCommand::ExecuteIndirectRange {
        icb: expected_icb,
        kind: expected_kind,
        ..
    } = pending.operation
    else {
        unreachable!("a pending range phase always owns the unresolved range form")
    };
    if icb != expected_icb || kind != expected_kind {
        return Err(Box::new(IndirectRangeExecutionResumeFailure {
            reason: IndirectRangeExecutionResumeError::IdentityMismatch,
            pending,
            literal,
        }));
    }
    replace_operation(
        &mut pending.continuation.exec,
        pending.operation_index,
        ResolvedOperation::IndirectCommand(literal),
    );
    pending.continuation.cursor = pending.operation_index;
    Ok(pending.continuation)
}

fn phase_exec<Render: Clone, Compute: Clone, Info: Clone, Completion: Clone>(
    transaction: TransactionId,
    exec: &IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    start: usize,
    end: usize,
) -> IndirectRangeExecutionPhase<Render, Compute, Info, Completion> {
    let prologue = crate::ExecPrologue::resource_states(
        exec.prologue
            .operations()
            .iter()
            .enumerate()
            .filter_map(|(position, operation)| {
                (position >= start && position < end)
                    .then_some(operation)
                    .map(|operation| match operation {
                        ResolvedOperation::ResourceState(state) => state.clone(),
                        _ => unreachable!("the typed EXEC prologue contains only resource states"),
                    })
            })
            .collect::<Vec<_>>(),
    );
    let mut position = exec.prologue.operations().len();
    let streams = exec
        .streams
        .iter()
        .filter_map(|stream| {
            let segments = stream
                .segments
                .iter()
                .filter_map(|segment| {
                    let operations = segment
                        .operations
                        .iter()
                        .filter_map(|operation| {
                            let included = position >= start && position < end;
                            position += 1;
                            included.then(|| operation.clone())
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    (!operations.is_empty()).then_some(crate::ResolvedExecSegment {
                        boundary: segment.boundary,
                        operations,
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            (!segments.is_empty()).then_some(crate::ResolvedExecStream {
                stream_index: stream.stream_index,
                segments,
            })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    IndirectRangeExecutionPhase {
        transaction,
        operation_base: start,
        exec: ExecTransaction {
            identity: exec.identity,
            prologue,
            streams,
            accesses: exec.accesses.clone(),
        },
    }
}

fn replace_operation<Render, Compute, Info, Completion>(
    exec: &mut IndirectExecutionTransaction<Render, Compute, Info, Completion>,
    target: usize,
    replacement: IndirectExecutionOperation<Render, Compute, Info, Completion>,
) {
    let mut position = exec.prologue.operations().len();
    let mut replacement = Some(replacement);
    for stream in exec.streams.iter_mut() {
        for segment in stream.segments.iter_mut() {
            for operation in segment.operations.iter_mut() {
                if position == target {
                    *operation = replacement
                        .take()
                        .expect("one flattened operation has one replacement");
                    return;
                }
                position += 1;
            }
        }
    }
    unreachable!("the pending phase position came from this exact EXEC")
}

/// Move an indirect-range execution into its asynchronous readback phase.
pub fn prepare_indirect_range_readback<T>(
    resources: &mut crate::ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    operation_index: usize,
    operation: ResolvedIndirectCommand,
) -> Result<PreparedIndirectRangeReadback, IndirectRangeReadbackError> {
    let ResolvedIndirectCommand::ExecuteIndirectRange {
        icb,
        arguments_resource,
        arguments_backing,
        arguments_range,
        kind,
    } = operation
    else {
        return Err(IndirectRangeReadbackError::LiteralExecutionRequired);
    };
    let arguments_representation = resources
        .view_representation(arguments_backing, crate::BackingView::Bytes)
        .map_err(IndirectRangeReadbackError::Arguments)?;
    let uses = vec![crate::RepresentationUse {
        backing: arguments_backing,
        representations: Box::new([arguments_representation]),
    }]
    .into_boxed_slice();
    resources
        .accept_uses(transaction, &uses)
        .map_err(IndirectRangeReadbackError::Uses)?;
    Ok(PreparedIndirectRangeReadback {
        transaction,
        operation_index,
        icb,
        arguments_resource,
        arguments_backing,
        arguments_representation,
        arguments_range,
        kind,
        uses,
    })
}

#[derive(Debug)]
pub struct CancelledIndirectRangeReadback<T> {
    pub operation: ResolvedIndirectCommand,
    pub resources: Vec<(BackingId, crate::ManagedBackingProgress<T>)>,
}

#[derive(Debug)]
pub struct IndirectRangeReadbackCancellationFailure {
    pub reason: crate::ResourceUseBatchError,
    pub readback: PreparedIndirectRangeReadback,
}

/// Return a readback that never entered queue ownership without manufacturing
/// a completion or losing its retained native representation.
pub fn cancel_prepared_indirect_range_readback<T>(
    resources: &mut crate::ResourceLifecycleOwner<T>,
    readback: PreparedIndirectRangeReadback,
) -> Result<CancelledIndirectRangeReadback<T>, Box<IndirectRangeReadbackCancellationFailure>> {
    if let Err(reason) =
        resources.validate_cancel_representation_uses(readback.transaction, &readback.uses)
    {
        return Err(Box::new(IndirectRangeReadbackCancellationFailure {
            reason,
            readback,
        }));
    }
    let progress = resources
        .cancel_representation_uses(readback.transaction, &readback.uses)
        .expect("the indirect-range representation cancellation was prevalidated");
    Ok(CancelledIndirectRangeReadback {
        operation: readback.operation(),
        resources: progress,
    })
}

/// Materialize a literal ICB execution only after the exact range readback has
/// retired. The wire value is two little-endian 32-bit quantities: command
/// location followed by command length.
pub fn complete_indirect_range_readback<Command>(
    owner: &IndirectCommandSlotOwner<Command>,
    readback: PreparedIndirectRangeReadback,
    bytes: [u8; 8],
) -> Result<ResolvedIndirectCommand, IndirectRangeReadbackFailure> {
    let range = ResolvedIndirectCommandRange {
        location: u64::from(u32::from_le_bytes(bytes[..4].try_into().unwrap())),
        length: u64::from(u32::from_le_bytes(bytes[4..].try_into().unwrap())),
    };
    if let Err(reason) = owner.validate_range(readback.icb, range, false) {
        return Err(IndirectRangeReadbackFailure {
            reason: IndirectRangeReadbackError::Owner(reason),
            readback,
            bytes,
        });
    }
    Ok(ResolvedIndirectCommand::Execute {
        icb: readback.icb,
        range,
        kind: readback.kind,
    })
}

/// Materialize a complete continuation batch only after every returned range
/// validates against its still-live ICB generation.
pub fn complete_indirect_range_readback_batch<Command>(
    owner: &IndirectCommandSlotOwner<Command>,
    readbacks: Box<[PreparedIndirectRangeReadback]>,
    returned_bytes: Box<[[u8; 8]]>,
) -> Result<Box<[ResolvedIndirectCommand]>, Box<IndirectRangeReadbackBatchFailure>> {
    if readbacks.len() != returned_bytes.len() {
        return Err(Box::new(IndirectRangeReadbackBatchFailure {
            reason: IndirectRangeReadbackBatchError::CountMismatch {
                readbacks: readbacks.len(),
                bytes: returned_bytes.len(),
            },
            readbacks,
            bytes: returned_bytes,
        }));
    }
    let mut ranges = Vec::with_capacity(returned_bytes.len());
    for (index, (readback, entry)) in readbacks.iter().zip(returned_bytes.iter()).enumerate() {
        let range = ResolvedIndirectCommandRange {
            location: u64::from(u32::from_le_bytes(entry[..4].try_into().unwrap())),
            length: u64::from(u32::from_le_bytes(entry[4..].try_into().unwrap())),
        };
        if let Err(reason) = owner.validate_range(readback.icb, range, false) {
            return Err(Box::new(IndirectRangeReadbackBatchFailure {
                reason: IndirectRangeReadbackBatchError::Readback {
                    index,
                    reason: IndirectRangeReadbackError::Owner(reason),
                },
                readbacks,
                bytes: returned_bytes,
            }));
        }
        ranges.push(range);
    }
    Ok(readbacks
        .into_vec()
        .into_iter()
        .zip(ranges)
        .map(|(readback, range)| ResolvedIndirectCommand::Execute {
            icb: readback.icb,
            range,
            kind: readback.kind,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

pub trait IndirectExecutionArgumentsResolver {
    fn resolve_indirect_execution_arguments(
        &self,
        task: TaskId,
        reference: ObjectTableRef<ResourceObject>,
    ) -> Option<IndirectExecutionArgumentsResolution>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandResolutionError {
    SourceAbsent,
    DestinationAbsent,
    SourceRangeOverflow,
    SourceRangePastCapacity,
    DestinationRangeOverflow,
    DestinationRangePastCapacity,
    ArgumentsAbsent,
    ArgumentsRangeOverflow,
    ArgumentsRangePastResource,
}

fn resolve_range(
    range: IndirectCommandRange,
    capacity: u32,
) -> Result<ResolvedIndirectCommandRange, IndirectCommandResolutionError> {
    let end = range
        .location
        .checked_add(range.length)
        .ok_or(IndirectCommandResolutionError::SourceRangeOverflow)?;
    if end > u64::from(capacity) {
        return Err(IndirectCommandResolutionError::SourceRangePastCapacity);
    }
    Ok(ResolvedIndirectCommandRange {
        location: range.location,
        length: range.length,
    })
}

pub fn resolve_indirect_command(
    task: TaskId,
    operation: IndirectCommandOperation,
    resolver: &impl IndirectCommandResolver,
) -> Result<ResolvedIndirectCommand, IndirectCommandResolutionError> {
    let source = |reference| {
        resolver
            .resolve_indirect_command_buffer(task, reference)
            .ok_or(IndirectCommandResolutionError::SourceAbsent)
    };
    match operation {
        IndirectCommandOperation::Optimize { icb, range } => {
            let icb = source(icb)?;
            Ok(ResolvedIndirectCommand::Optimize {
                icb: icb.identity,
                range: resolve_range(range, icb.max_command_count)?,
            })
        }
        IndirectCommandOperation::Reset { icb, range } => {
            let icb = source(icb)?;
            Ok(ResolvedIndirectCommand::Reset {
                icb: icb.identity,
                range: resolve_range(range, icb.max_command_count)?,
            })
        }
        IndirectCommandOperation::Copy {
            source: source_ref,
            source_range,
            destination: destination_ref,
            destination_index,
        } => {
            let source = source(source_ref)?;
            let source_range = resolve_range(source_range, source.max_command_count)?;
            let destination = resolver
                .resolve_indirect_command_buffer(task, destination_ref)
                .ok_or(IndirectCommandResolutionError::DestinationAbsent)?;
            let destination_end = destination_index
                .checked_add(source_range.length)
                .ok_or(IndirectCommandResolutionError::DestinationRangeOverflow)?;
            if destination_end > u64::from(destination.max_command_count) {
                return Err(IndirectCommandResolutionError::DestinationRangePastCapacity);
            }
            Ok(ResolvedIndirectCommand::Copy {
                source: source.identity,
                source_range,
                destination: destination.identity,
                destination_index,
            })
        }
    }
}

pub fn resolve_indirect_execution(
    task: TaskId,
    icb: SerializerRef<IndirectCommandBufferObject>,
    range: IndirectCommandRange,
    kind: IndirectCommandExecutionKind,
    resolver: &impl IndirectCommandResolver,
) -> Result<ResolvedIndirectCommand, IndirectCommandResolutionError> {
    let icb = resolver
        .resolve_indirect_command_buffer(task, icb)
        .ok_or(IndirectCommandResolutionError::SourceAbsent)?;
    Ok(ResolvedIndirectCommand::Execute {
        icb: icb.identity,
        range: resolve_range(range, icb.max_command_count)?,
        kind,
    })
}

pub fn resolve_indirect_execution_arguments(
    task: TaskId,
    icb: SerializerRef<IndirectCommandBufferObject>,
    arguments: ObjectTableRef<ResourceObject>,
    offset: u64,
    kind: IndirectCommandExecutionKind,
    icb_resolver: &impl IndirectCommandResolver,
    arguments_resolver: &impl IndirectExecutionArgumentsResolver,
) -> Result<ResolvedIndirectCommand, IndirectCommandResolutionError> {
    let icb = icb_resolver
        .resolve_indirect_command_buffer(task, icb)
        .ok_or(IndirectCommandResolutionError::SourceAbsent)?;
    let arguments = arguments_resolver
        .resolve_indirect_execution_arguments(task, arguments)
        .ok_or(IndirectCommandResolutionError::ArgumentsAbsent)?;
    let end = offset
        .checked_add(8)
        .ok_or(IndirectCommandResolutionError::ArgumentsRangeOverflow)?;
    if end > arguments.size {
        return Err(IndirectCommandResolutionError::ArgumentsRangePastResource);
    }
    Ok(ResolvedIndirectCommand::ExecuteIndirectRange {
        icb: icb.identity,
        arguments_resource: arguments.resource,
        arguments_backing: arguments.backing,
        arguments_range: LinearRange::new(offset, 8)
            .expect("the exact eight-byte range was checked for overflow"),
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackingView;
    use crate::{
        BackingRegion, EncoderBoundary, RepresentationRoute, ResolvedExecSegment,
        ResolvedExecStream, ResolvedResourceLifecycle, ResolvedResourceState,
        ResourceLifecycleEffect, ResourceLifecycleOwner, StorageBacking,
    };
    use reims_vgpu_protocol::{
        SegmentBoundary, SegmentKind, SubmissionId, SubmissionIdentity, VulkanDeviceEpochId,
    };
    use std::collections::BTreeMap;

    struct Resolver(BTreeMap<u32, IndirectCommandBufferResolution>);

    impl IndirectCommandResolver for Resolver {
        fn resolve_indirect_command_buffer(
            &self,
            _task: TaskId,
            reference: SerializerRef<IndirectCommandBufferObject>,
        ) -> Option<IndirectCommandBufferResolution> {
            self.0.get(&reference.get()).copied()
        }
    }

    impl IndirectExecutionArgumentsResolver for Resolver {
        fn resolve_indirect_execution_arguments(
            &self,
            _task: TaskId,
            reference: ObjectTableRef<ResourceObject>,
        ) -> Option<IndirectExecutionArgumentsResolution> {
            (reference.get() == 6).then_some(IndirectExecutionArgumentsResolution {
                resource: ResourceId::new(7, 2),
                backing: BackingId::new(9),
                size: 32,
            })
        }
    }

    fn resolver() -> Resolver {
        Resolver(BTreeMap::from([
            (
                3,
                IndirectCommandBufferResolution {
                    identity: ResourceId::new(1, 4),
                    max_command_count: 8,
                },
            ),
            (
                5,
                IndirectCommandBufferResolution {
                    identity: ResourceId::new(2, 7),
                    max_command_count: 12,
                },
            ),
        ]))
    }

    fn readback_resources() -> ResourceLifecycleOwner<()> {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let mut arguments = None;
        for _ in 0..9 {
            let ResourceLifecycleEffect::BackingCreated(backing) = resources
                .apply(ResolvedResourceLifecycle::CreateBacking {
                    backing: StorageBacking::Dedicated,
                    regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 32).unwrap())]),
                })
                .unwrap()
            else {
                unreachable!()
            };
            arguments = Some(backing);
        }
        let arguments = arguments.unwrap();
        assert_eq!(arguments, BackingId::new(9));
        resources
            .create_execution_representation(
                arguments,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        resources
    }

    fn exec(
        operations: impl Into<Box<[ResolvedOperation<(), (), (), ResolvedIndirectCommand, ()>]>>,
    ) -> ExecTransaction<ResolvedOperation<(), (), (), ResolvedIndirectCommand, ()>> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Blit,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: operations.into(),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    fn optimize() -> ResolvedIndirectCommand {
        ResolvedIndirectCommand::Optimize {
            icb: ResourceId::new(1, 4),
            range: ResolvedIndirectCommandRange {
                location: 2,
                length: 3,
            },
        }
    }

    #[test]
    fn copy_resolves_both_generations_and_exact_indices() {
        assert_eq!(
            resolve_indirect_command(
                TaskId::new(9),
                IndirectCommandOperation::Copy {
                    source: SerializerRef::new(3),
                    source_range: IndirectCommandRange {
                        location: 2,
                        length: 4,
                    },
                    destination: SerializerRef::new(5),
                    destination_index: 7,
                },
                &resolver(),
            ),
            Ok(ResolvedIndirectCommand::Copy {
                source: ResourceId::new(1, 4),
                source_range: ResolvedIndirectCommandRange {
                    location: 2,
                    length: 4,
                },
                destination: ResourceId::new(2, 7),
                destination_index: 7,
            })
        );
    }

    #[test]
    fn execution_resolves_the_exact_generation_range_and_encoder_kind() {
        assert_eq!(
            resolve_indirect_execution(
                TaskId::new(9),
                SerializerRef::new(3),
                IndirectCommandRange {
                    location: 1,
                    length: 6,
                },
                IndirectCommandExecutionKind::Render,
                &resolver(),
            ),
            Ok(ResolvedIndirectCommand::Execute {
                icb: ResourceId::new(1, 4),
                range: ResolvedIndirectCommandRange {
                    location: 1,
                    length: 6,
                },
                kind: IndirectCommandExecutionKind::Render,
            })
        );
    }

    #[test]
    fn indirect_execution_range_retains_the_exact_argument_resource_window() {
        assert_eq!(
            resolve_indirect_execution_arguments(
                TaskId::new(9),
                SerializerRef::new(3),
                ObjectTableRef::new(6),
                16,
                IndirectCommandExecutionKind::Compute,
                &resolver(),
                &resolver(),
            ),
            Ok(ResolvedIndirectCommand::ExecuteIndirectRange {
                icb: ResourceId::new(1, 4),
                arguments_resource: ResourceId::new(7, 2),
                arguments_backing: BackingId::new(9),
                arguments_range: LinearRange::new(16, 8).unwrap(),
                kind: IndirectCommandExecutionKind::Compute,
            })
        );
        assert_eq!(
            resolve_indirect_execution_arguments(
                TaskId::new(9),
                SerializerRef::new(3),
                ObjectTableRef::new(6),
                28,
                IndirectCommandExecutionKind::Compute,
                &resolver(),
                &resolver(),
            ),
            Err(IndirectCommandResolutionError::ArgumentsRangePastResource)
        );
    }

    #[test]
    fn gpu_produced_execution_range_materializes_only_through_its_exact_readback() {
        let operation = resolve_indirect_execution_arguments(
            TaskId::new(9),
            SerializerRef::new(3),
            ObjectTableRef::new(6),
            16,
            IndirectCommandExecutionKind::Compute,
            &resolver(),
            &resolver(),
        )
        .unwrap();
        let mut resources = readback_resources();
        let readback =
            prepare_indirect_range_readback(&mut resources, TransactionId::new(12), 4, operation)
                .unwrap();
        assert_eq!(readback.transaction(), TransactionId::new(12));
        assert_eq!(readback.operation_index(), 4);
        assert_eq!(readback.arguments_resource(), ResourceId::new(7, 2));
        assert_eq!(readback.arguments_backing(), BackingId::new(9));
        assert_eq!(readback.uses().len(), 1);
        assert_eq!(
            readback.arguments_representation(),
            resources
                .execution_representation_id(BackingId::new(9))
                .unwrap()
        );
        assert_eq!(readback.arguments_range(), LinearRange::new(16, 8).unwrap());

        let icb = ResourceId::new(1, 4);
        let mut owner = IndirectCommandSlotOwner::<()>::default();
        owner.register(icb, 8).unwrap();
        let literal =
            complete_indirect_range_readback(&owner, readback, [2, 0, 0, 0, 3, 0, 0, 0]).unwrap();
        assert_eq!(
            literal,
            ResolvedIndirectCommand::Execute {
                icb,
                range: ResolvedIndirectCommandRange {
                    location: 2,
                    length: 3,
                },
                kind: IndirectCommandExecutionKind::Compute,
            }
        );
    }

    #[test]
    fn invalid_gpu_produced_range_returns_readback_ownership_and_bytes() {
        let operation = resolve_indirect_execution_arguments(
            TaskId::new(9),
            SerializerRef::new(3),
            ObjectTableRef::new(6),
            16,
            IndirectCommandExecutionKind::Render,
            &resolver(),
            &resolver(),
        )
        .unwrap();
        let mut resources = readback_resources();
        let readback =
            prepare_indirect_range_readback(&mut resources, TransactionId::new(13), 5, operation)
                .unwrap();
        let icb = ResourceId::new(1, 4);
        let mut owner = IndirectCommandSlotOwner::<()>::default();
        owner.register(icb, 8).unwrap();
        let bytes = [7, 0, 0, 0, 2, 0, 0, 0];
        let failure = complete_indirect_range_readback(&owner, readback, bytes).unwrap_err();
        assert_eq!(
            failure.reason,
            IndirectRangeReadbackError::Owner(
                IndirectCommandMutationError::SourceRangePastCapacity
            )
        );
        assert_eq!(failure.readback.transaction(), TransactionId::new(13));
        assert_eq!(failure.bytes.as_ref(), bytes.as_ref());
    }

    #[test]
    fn cancelled_range_readback_returns_the_operation_and_representation() {
        let operation = resolve_indirect_execution_arguments(
            TaskId::new(9),
            SerializerRef::new(3),
            ObjectTableRef::new(6),
            16,
            IndirectCommandExecutionKind::Render,
            &resolver(),
            &resolver(),
        )
        .unwrap();
        let mut resources = readback_resources();
        let readback =
            prepare_indirect_range_readback(&mut resources, TransactionId::new(14), 6, operation)
                .unwrap();
        let cancelled = cancel_prepared_indirect_range_readback(&mut resources, readback).unwrap();
        assert_eq!(cancelled.operation, operation);
        assert_eq!(cancelled.resources.len(), 1);
        assert_eq!(cancelled.resources[0].0, BackingId::new(9));
    }

    #[test]
    fn invalid_readback_batch_returns_every_token_and_byte_row() {
        let operation = resolve_indirect_execution_arguments(
            TaskId::new(9),
            SerializerRef::new(3),
            ObjectTableRef::new(6),
            16,
            IndirectCommandExecutionKind::Render,
            &resolver(),
            &resolver(),
        )
        .unwrap();
        let mut first_resources = readback_resources();
        let mut second_resources = readback_resources();
        let first = prepare_indirect_range_readback(
            &mut first_resources,
            TransactionId::new(15),
            2,
            operation,
        )
        .unwrap();
        let second = prepare_indirect_range_readback(
            &mut second_resources,
            TransactionId::new(15),
            3,
            operation,
        )
        .unwrap();
        let mut owner = IndirectCommandSlotOwner::<()>::default();
        owner.register(ResourceId::new(1, 4), 8).unwrap();
        let bytes = Box::new([[1, 0, 0, 0, 1, 0, 0, 0], [7, 0, 0, 0, 2, 0, 0, 0]]);
        let failure = complete_indirect_range_readback_batch(
            &owner,
            Box::new([first, second]),
            bytes.clone(),
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            IndirectRangeReadbackBatchError::Readback {
                index: 1,
                reason: IndirectRangeReadbackError::Owner(
                    IndirectCommandMutationError::SourceRangePastCapacity
                ),
            }
        );
        assert_eq!(failure.readbacks.len(), 2);
        assert_eq!(failure.bytes.as_ref(), bytes.as_ref());
    }

    #[test]
    fn source_and_destination_bounds_refuse_by_name() {
        let source_overflow = IndirectCommandOperation::Reset {
            icb: SerializerRef::new(3),
            range: IndirectCommandRange {
                location: u64::MAX,
                length: 1,
            },
        };
        assert_eq!(
            resolve_indirect_command(TaskId::new(1), source_overflow, &resolver()),
            Err(IndirectCommandResolutionError::SourceRangeOverflow)
        );

        let destination_past = IndirectCommandOperation::Copy {
            source: SerializerRef::new(3),
            source_range: IndirectCommandRange {
                location: 0,
                length: 6,
            },
            destination: SerializerRef::new(5),
            destination_index: 7,
        };
        assert_eq!(
            resolve_indirect_command(TaskId::new(1), destination_past, &resolver()),
            Err(IndirectCommandResolutionError::DestinationRangePastCapacity)
        );
    }

    #[test]
    fn empty_range_may_name_the_capacity_endpoint() {
        let operation = IndirectCommandOperation::Optimize {
            icb: SerializerRef::new(3),
            range: IndirectCommandRange {
                location: 8,
                length: 0,
            },
        };
        assert!(resolve_indirect_command(TaskId::new(1), operation, &resolver()).is_ok());
    }

    #[test]
    fn admission_proves_only_exact_optimize_positions() {
        let command = optimize();
        let exec = exec(vec![
            ResolvedOperation::EncoderBoundary(EncoderBoundary::Begin(SegmentKind::Blit)),
            ResolvedOperation::IndirectCommand(command),
        ]);
        let admitted = admit_indirect_commands(TransactionId::new(7), &exec).unwrap();
        assert_eq!(admitted.transaction(), TransactionId::new(7));
        assert_eq!(admitted.operations(), &[(1, command)]);
    }

    #[test]
    fn reset_and_copy_remain_typed_storage_contract_refusals() {
        let reset = ResolvedIndirectCommand::Reset {
            icb: ResourceId::new(1, 4),
            range: ResolvedIndirectCommandRange {
                location: 0,
                length: 1,
            },
        };
        assert_eq!(
            admit_indirect_commands(
                TransactionId::new(7),
                &exec(vec![ResolvedOperation::IndirectCommand(reset)]),
            ),
            Err(IndirectCommandAdmissionError {
                index: 0,
                kind: IndirectCommandKind::Reset,
            })
        );
    }

    #[test]
    fn slot_owner_reset_and_overlap_copy_preserve_empty_slots() {
        let icb = ResourceId::new(1, 4);
        let mut owner = IndirectCommandSlotOwner::default();
        owner.register(icb, 6).unwrap();
        owner.set(icb, 0, 10).unwrap();
        owner.set(icb, 1, 20).unwrap();
        owner.set(icb, 2, 30).unwrap();
        owner
            .apply(ResolvedIndirectCommand::Reset {
                icb,
                range: ResolvedIndirectCommandRange {
                    location: 1,
                    length: 1,
                },
            })
            .unwrap();
        owner
            .apply(ResolvedIndirectCommand::Copy {
                source: icb,
                source_range: ResolvedIndirectCommandRange {
                    location: 0,
                    length: 3,
                },
                destination: icb,
                destination_index: 2,
            })
            .unwrap();
        assert_eq!(owner.get(icb, 0), Ok(Some(&10)));
        assert_eq!(owner.get(icb, 1), Ok(None));
        assert_eq!(owner.get(icb, 2), Ok(Some(&10)));
        assert_eq!(owner.get(icb, 3), Ok(None));
        assert_eq!(owner.get(icb, 4), Ok(Some(&30)));
    }

    #[test]
    fn owner_admission_proves_reset_and_copy_for_exact_live_generations() {
        let source = ResourceId::new(1, 4);
        let destination = ResourceId::new(2, 7);
        let reset = ResolvedIndirectCommand::Reset {
            icb: source,
            range: ResolvedIndirectCommandRange {
                location: 0,
                length: 2,
            },
        };
        let copy = ResolvedIndirectCommand::Copy {
            source,
            source_range: ResolvedIndirectCommandRange {
                location: 0,
                length: 2,
            },
            destination,
            destination_index: 3,
        };
        let exec = exec([
            ResolvedOperation::IndirectCommand(reset),
            ResolvedOperation::IndirectCommand(copy),
        ]);
        let mut owner = IndirectCommandSlotOwner::<u32>::default();
        owner.register(source, 8).unwrap();
        owner.register(destination, 8).unwrap();
        let admitted =
            admit_indirect_commands_with_owner(TransactionId::new(8), &exec, &owner).unwrap();
        assert_eq!(admitted.operations(), &[(0, reset), (1, copy)]);

        owner.retire(destination).unwrap();
        assert_eq!(
            admit_indirect_commands_with_owner(TransactionId::new(8), &exec, &owner),
            Err(IndirectCommandAdmissionError {
                index: 1,
                kind: IndirectCommandKind::Copy,
            })
        );
    }

    #[test]
    fn commit_prevalidates_the_complete_mutation_batch() {
        let source = ResourceId::new(1, 4);
        let destination = ResourceId::new(2, 7);
        let mut owner = IndirectCommandSlotOwner::default();
        owner.register(source, 4).unwrap();
        owner.register(destination, 4).unwrap();
        owner.set(source, 0, 9u32).unwrap();
        owner.set(destination, 0, 3u32).unwrap();
        let exec = exec([
            ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Reset {
                icb: destination,
                range: ResolvedIndirectCommandRange {
                    location: 0,
                    length: 1,
                },
            }),
            ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Copy {
                source,
                source_range: ResolvedIndirectCommandRange {
                    location: 0,
                    length: 1,
                },
                destination,
                destination_index: 0,
            }),
        ]);
        let admitted =
            admit_indirect_commands_with_owner(TransactionId::new(9), &exec, &owner).unwrap();
        owner.retire(source).unwrap();
        let failure = commit_indirect_commands(&mut owner, admitted).unwrap_err();
        assert_eq!(failure.index, 1);
        assert_eq!(failure.kind, IndirectCommandKind::Copy);
        assert_eq!(owner.get(destination, 0), Ok(Some(&3)));
    }

    #[test]
    fn execution_snapshots_its_exact_position_between_ordered_mutations() {
        let icb = ResourceId::new(3, 8);
        let mut owner = IndirectCommandSlotOwner::default();
        owner.register(icb, 2).unwrap();
        owner
            .set(icb, 0, ResolvedIndirectCommandSlot::<(), ()>::Render(()))
            .unwrap();
        owner
            .set(icb, 1, ResolvedIndirectCommandSlot::<(), ()>::Render(()))
            .unwrap();
        let reset = |location| ResolvedIndirectCommand::Reset {
            icb,
            range: ResolvedIndirectCommandRange {
                location,
                length: 1,
            },
        };
        let execute = ResolvedIndirectCommand::Execute {
            icb,
            range: ResolvedIndirectCommandRange {
                location: 0,
                length: 2,
            },
            kind: IndirectCommandExecutionKind::Render,
        };
        let exec = exec([
            ResolvedOperation::IndirectCommand(reset(0)),
            ResolvedOperation::IndirectCommand(execute),
            ResolvedOperation::IndirectCommand(reset(1)),
        ]);
        let admitted =
            admit_indirect_commands_with_owner(TransactionId::new(10), &exec, &owner).unwrap();
        let committed = commit_indirect_commands(&mut owner, admitted).unwrap();
        assert_eq!(committed.executions.len(), 1);
        assert_eq!(committed.executions[0].index, 1);
        assert_eq!(
            committed.executions[0].slots.as_ref(),
            [
                (0, None),
                (1, Some(ResolvedIndirectCommandSlot::Render(())))
            ]
        );
        let (expanded, origins) = expand_committed_indirect_executions_with_origins(
            TransactionId::new(10),
            &exec,
            &committed,
        )
        .unwrap();
        let expanded = expanded.operations().collect::<Vec<_>>();
        assert!(matches!(
            expanded.as_slice(),
            [
                ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Reset { .. }),
                ResolvedOperation::Render(()),
                ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Reset { .. }),
            ]
        ));
        assert_eq!(
            origins.as_ref(),
            [
                ExpandedIndirectOperationOrigin {
                    expanded_position: 0,
                    original_position: 0,
                    indirect_slot: None,
                },
                ExpandedIndirectOperationOrigin {
                    expanded_position: 1,
                    original_position: 1,
                    indirect_slot: Some(1),
                },
                ExpandedIndirectOperationOrigin {
                    expanded_position: 2,
                    original_position: 2,
                    indirect_slot: None,
                },
            ]
        );
        assert_eq!(owner.get(icb, 0), Ok(None));
        assert_eq!(owner.get(icb, 1), Ok(None));
    }

    #[test]
    fn execution_expansion_refusal_rolls_back_preceding_slot_mutations() {
        let icb = ResourceId::new(4, 9);
        let mut owner = IndirectCommandSlotOwner::default();
        owner.register(icb, 2).unwrap();
        owner
            .set(icb, 0, ResolvedIndirectCommandSlot::<(), ()>::Compute(()))
            .unwrap();
        owner
            .set(icb, 1, ResolvedIndirectCommandSlot::<(), ()>::Render(()))
            .unwrap();
        let operations = exec([
            ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Reset {
                icb,
                range: ResolvedIndirectCommandRange {
                    location: 1,
                    length: 1,
                },
            }),
            ResolvedOperation::IndirectCommand(ResolvedIndirectCommand::Execute {
                icb,
                range: ResolvedIndirectCommandRange {
                    location: 0,
                    length: 1,
                },
                kind: IndirectCommandExecutionKind::Render,
            }),
        ]);
        let admitted =
            admit_indirect_commands_with_owner(TransactionId::new(11), &operations, &owner)
                .unwrap();
        let failure =
            prepare_indirect_execution(&mut owner, TransactionId::new(11), &operations, admitted)
                .unwrap_err();
        assert_eq!(
            failure.reason,
            IndirectExecutionPreparationError::Expansion(
                IndirectExecutionExpansionError::CommandKindMismatch {
                    operation: 1,
                    slot: 0,
                }
            )
        );
        assert_eq!(
            owner.get(icb, 1),
            Ok(Some(&ResolvedIndirectCommandSlot::Render(())))
        );
    }

    #[test]
    fn population_is_atomic_and_execution_snapshots_empty_slots_in_order() {
        let icb = ResourceId::new(4, 2);
        let mut owner = IndirectCommandSlotOwner::default();
        owner.register(icb, 4).unwrap();
        owner
            .set_batch(
                icb,
                Box::new([
                    (0, Some(ResolvedIndirectCommandSlot::<u32, u32>::Render(11))),
                    (2, Some(ResolvedIndirectCommandSlot::Compute(22))),
                ]),
            )
            .unwrap();
        let invalid = Box::new([
            (1, Some(ResolvedIndirectCommandSlot::Render(33))),
            (4, Some(ResolvedIndirectCommandSlot::Compute(44))),
        ]);
        let failure = owner.set_batch(icb, invalid).unwrap_err();
        assert_eq!(
            failure.reason,
            IndirectCommandMutationError::SlotPastCapacity
        );
        assert_eq!(failure.commands.len(), 2);
        assert_eq!(owner.get(icb, 1), Ok(None));
        assert_eq!(
            owner
                .snapshot_range(
                    icb,
                    ResolvedIndirectCommandRange {
                        location: 0,
                        length: 4,
                    },
                )
                .unwrap()
                .as_ref(),
            [
                (0, Some(ResolvedIndirectCommandSlot::Render(11))),
                (1, None),
                (2, Some(ResolvedIndirectCommandSlot::Compute(22))),
                (3, None),
            ]
        );
    }

    #[test]
    fn decoded_population_resolves_against_one_state_snapshot_before_publication() {
        let icb = ResourceId::new(5, 3);
        let mut owner = IndirectCommandSlotOwner::default();
        owner.register(icb, 3).unwrap();
        owner.set(icb, 0, 7u32).unwrap();
        let decoded = Box::new([(0, Some(2u32)), (1, Some(9u32))]);
        let failure = resolve_indirect_command_population(
            &mut owner,
            icb,
            &10u32,
            decoded,
            |inherited, input| {
                if *input == 9 {
                    Err("bad slot")
                } else {
                    Ok(Some(inherited + input))
                }
            },
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            IndirectCommandPopulationResolutionError::Resolve {
                index: 1,
                reason: "bad slot",
            }
        );
        assert_eq!(failure.commands.as_ref(), [(0, Some(2)), (1, Some(9))]);
        assert_eq!(owner.get(icb, 0), Ok(Some(&7)));
        assert_eq!(owner.get(icb, 1), Ok(None));

        let prior = resolve_indirect_command_population(
            &mut owner,
            icb,
            &10u32,
            Box::new([(0, Some(2u32)), (1, Some(3u32))]),
            |inherited, input| Ok::<_, ()>(Some(inherited + input)),
        )
        .unwrap();
        assert_eq!(prior.as_ref(), [(0, Some(7)), (1, None)]);
        assert_eq!(owner.get(icb, 0), Ok(Some(&12)));
        assert_eq!(owner.get(icb, 1), Ok(Some(&13)));

        let prior = resolve_indirect_command_population(
            &mut owner,
            icb,
            &10u32,
            Box::new([(0, None), (1, Some(4u32))]),
            |inherited, input| Ok::<_, ()>(Some(inherited + input)),
        )
        .unwrap();
        assert_eq!(prior.as_ref(), [(0, Some(12)), (1, Some(13))]);
        assert_eq!(owner.get(icb, 0), Ok(None));
        assert_eq!(owner.get(icb, 1), Ok(Some(&14)));

        let prior = resolve_indirect_command_population(
            &mut owner,
            icb,
            &10u32,
            Box::new([(1, Some(0u32))]),
            |_, _| Ok::<_, ()>(None),
        )
        .unwrap();
        assert_eq!(prior.as_ref(), [(1, Some(14))]);
        assert_eq!(owner.get(icb, 1), Ok(None));
    }

    #[test]
    fn range_continuation_never_replays_a_prefix_and_completes_only_the_suffix() {
        let icb = ResourceId::new(1, 4);
        let indirect = |offset| ResolvedIndirectCommand::ExecuteIndirectRange {
            icb,
            arguments_resource: ResourceId::new(7, 2),
            arguments_backing: BackingId::new(9),
            arguments_range: LinearRange::new(offset, 8).unwrap(),
            kind: IndirectCommandExecutionKind::Render,
        };
        let operations = vec![
            ResolvedOperation::IndirectCommand(optimize()),
            ResolvedOperation::IndirectCommand(indirect(0)),
            ResolvedOperation::IndirectCommand(optimize()),
            ResolvedOperation::IndirectCommand(indirect(8)),
            ResolvedOperation::IndirectCommand(optimize()),
        ]
        .into_boxed_slice();
        let continuation =
            IndirectRangeExecutionContinuation::new(TransactionId::new(11), exec(operations));

        let NextIndirectRangeExecution::Readback(first) = continuation.next() else {
            panic!("the first GPU range must end an auxiliary phase")
        };
        assert_eq!(first.operation_index(), 1);
        assert_eq!(first.phase().operation_base(), 0);
        assert_eq!(first.phase().exec().operations().count(), 2);
        let failure = resume_indirect_range_execution(
            first,
            ResolvedIndirectCommand::Execute {
                icb: ResourceId::new(2, 1),
                range: ResolvedIndirectCommandRange {
                    location: 1,
                    length: 2,
                },
                kind: IndirectCommandExecutionKind::Render,
            },
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            IndirectRangeExecutionResumeError::IdentityMismatch
        );
        let continuation = resume_indirect_range_execution(
            failure.pending,
            ResolvedIndirectCommand::Execute {
                icb,
                range: ResolvedIndirectCommandRange {
                    location: 1,
                    length: 2,
                },
                kind: IndirectCommandExecutionKind::Render,
            },
        )
        .unwrap();

        let NextIndirectRangeExecution::Readback(second) = continuation.next() else {
            panic!("the second GPU range must end another auxiliary phase")
        };
        assert_eq!(second.operation_index(), 3);
        assert_eq!(second.phase().operation_base(), 1);
        assert_eq!(second.phase().exec().operations().count(), 3);
        assert!(matches!(
            second.phase().exec().operations().next(),
            Some(ResolvedOperation::IndirectCommand(
                ResolvedIndirectCommand::Execute { .. }
            ))
        ));
        let continuation = resume_indirect_range_execution(
            second,
            ResolvedIndirectCommand::Execute {
                icb,
                range: ResolvedIndirectCommandRange {
                    location: 4,
                    length: 1,
                },
                kind: IndirectCommandExecutionKind::Render,
            },
        )
        .unwrap();
        let NextIndirectRangeExecution::Final(final_phase) = continuation.next() else {
            panic!("a range-free suffix is the one semantic completion phase")
        };
        assert_eq!(final_phase.operation_base(), 3);
        assert_eq!(final_phase.exec().operations().count(), 2);
        assert!(matches!(
            final_phase.exec().operations().next(),
            Some(ResolvedOperation::IndirectCommand(
                ResolvedIndirectCommand::Execute { .. }
            ))
        ));
    }

    #[test]
    fn range_positions_include_the_resource_state_prologue_without_fabricating_a_segment() {
        let icb = ResourceId::new(1, 4);
        let range = ResolvedIndirectCommand::ExecuteIndirectRange {
            icb,
            arguments_resource: ResourceId::new(7, 2),
            arguments_backing: BackingId::new(9),
            arguments_range: LinearRange::new(0, 8).unwrap(),
            kind: IndirectCommandExecutionKind::Render,
        };
        let mut transaction = exec([
            ResolvedOperation::IndirectCommand(optimize()),
            ResolvedOperation::IndirectCommand(range),
        ]);
        transaction.prologue = crate::ExecPrologue::resource_states([ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([]),
            ops: reims_vgpu_protocol::ResourceValidityOps::default(),
        }]);
        let continuation =
            IndirectRangeExecutionContinuation::new(TransactionId::new(11), transaction);

        let NextIndirectRangeExecution::Readback(readback) = continuation.next() else {
            panic!("the range operation requires an auxiliary phase")
        };
        assert_eq!(readback.operation_index(), 2);
        assert_eq!(readback.phase().operation_base(), 0);
        assert_eq!(readback.phase().exec().prologue.operations().len(), 1);
        assert_eq!(readback.phase().exec().operations().count(), 3);
    }

    #[test]
    fn range_continuation_after_completed_prefix_starts_at_the_exact_boundary() {
        let icb = ResourceId::new(1, 4);
        let range = ResolvedIndirectCommand::ExecuteIndirectRange {
            icb,
            arguments_resource: ResourceId::new(7, 2),
            arguments_backing: BackingId::new(9),
            arguments_range: LinearRange::new(0, 8).unwrap(),
            kind: IndirectCommandExecutionKind::Compute,
        };
        let transaction = exec([
            ResolvedOperation::IndirectCommand(optimize()),
            ResolvedOperation::IndirectCommand(optimize()),
            ResolvedOperation::IndirectCommand(range),
            ResolvedOperation::IndirectCommand(optimize()),
        ]);
        let continuation = IndirectRangeExecutionContinuation::after_prefix(
            TransactionId::new(11),
            transaction.clone(),
            2,
        )
        .unwrap();
        let NextIndirectRangeExecution::Readback(readback) = continuation.next() else {
            panic!("the suffix range must end the first resumed phase")
        };
        assert_eq!(readback.operation_index(), 2);
        assert_eq!(readback.phase().operation_base(), 2);
        assert_eq!(readback.phase().exec().operations().count(), 1);

        assert_eq!(
            IndirectRangeExecutionContinuation::after_prefix(
                TransactionId::new(11),
                transaction,
                5,
            )
            .unwrap_err(),
            IndirectRangeExecutionStartError::OperationBasePastEnd {
                base: 5,
                operations: 4,
            }
        );
    }

    #[test]
    fn command_memory_read_plan_is_exact_and_empty_ranges_do_not_require_memory() {
        let mut descriptor = IndirectCommandBufferDescriptor {
            max_command_count: 8,
            ..Default::default()
        };
        descriptor.layout.command_size = 32;
        let memory = IcbCommandMemory {
            gva: 0x4000,
            byte_len: 256,
        };
        assert_eq!(
            resolve_indirect_command_memory_read(
                &descriptor,
                Some(memory),
                ResolvedIndirectCommandRange {
                    location: 2,
                    length: 3,
                },
            ),
            Ok(Some(IndirectCommandMemoryReadPlan {
                gva: 0x4040,
                byte_len: 96,
            }))
        );
        assert_eq!(
            resolve_indirect_command_memory_read(
                &descriptor,
                None,
                ResolvedIndirectCommandRange {
                    location: 8,
                    length: 0,
                },
            ),
            Ok(None)
        );
        assert_eq!(
            resolve_indirect_command_memory_read(
                &descriptor,
                Some(IcbCommandMemory {
                    gva: 0x4000,
                    byte_len: 255,
                }),
                ResolvedIndirectCommandRange {
                    location: 7,
                    length: 1,
                },
            ),
            Err(IndirectCommandMemoryReadError::CommandRangePastMemory)
        );
        assert_eq!(
            resolve_indirect_command_memory_read(
                &descriptor,
                None,
                ResolvedIndirectCommandRange {
                    location: 0,
                    length: 1,
                },
            ),
            Err(IndirectCommandMemoryReadError::CommandMemoryUnbound)
        );
    }
}
