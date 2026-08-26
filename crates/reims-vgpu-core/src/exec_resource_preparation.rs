//! Whole-EXEC ownership of operation-local native resource preparations.
//!
//! Buffer blits, image blits, compute dispatches, Info replies, indirect-range reads, and
//! resource-state rows reserve content and native representation lifetimes independently while being
//! prepared. This envelope proves that the complete immutable EXEC has exactly
//! one matching token at every such operation position, then derives the one
//! backing-use and timeline-completion sets consumed by queue acceptance.

use crate::{
    BufferBlitPreparationFailure, ComputeDispatchPreparationError, EvaluatedInfoQuery,
    ExecTransaction, GpuWriteBatchError, ImageBlitPreparationFailure, InfoQueryPreparationFailure,
    ManagedBackingProgress, PreparedBufferBlit, PreparedComputeDispatch, PreparedImageBlit,
    PreparedIndirectRangeReadback, PreparedInfoQuery, PreparedRenderDispatch,
    PreparedResourceState, PreparedResourceStateBatch, ReadyPipelineLease, ResolvedBlit,
    ResolvedComputeDispatch, ResolvedIndirectCommand, ResolvedInfoOperation, ResolvedOperation,
    ResolvedRenderDispatch, ResolvedResourceCompletion, ResourceLifecycleOwner,
    ResourceStateOutcome, ResourceUseBatchError, TransferBatchError,
};
use reims_vgpu_protocol::{
    BackingId, ComputePipelineObject, RenderPipelineObject, SubmissionId, TransactionId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub struct PreparedExecResourceInputs<
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub buffer_blits: Box<[PreparedBufferBlit]>,
    pub image_blits: Box<[PreparedImageBlit]>,
    pub compute_dispatches: Box<[PreparedComputeDispatch<NativeCompute, Compute>]>,
    pub render_dispatches: Box<[PreparedRenderDispatch<NativeRender, Render>]>,
    pub info_queries: Box<[PreparedInfoQuery]>,
    pub indirect_range_readbacks: Box<[PreparedIndirectRangeReadback]>,
    pub resource_states: Option<PreparedResourceStateBatch>,
    pub content_synchronization: Option<crate::PreparedContentSynchronizationBatch>,
}

#[derive(Debug)]
pub struct PreparedExecResources<Compute = (), NativeCompute = (), Render = (), NativeRender = ()> {
    transaction: TransactionId,
    inputs: PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender>,
    backings: Box<[BackingId]>,
    completions: Box<[ResolvedResourceCompletion]>,
    host_landings: Box<[crate::HostLandingKey]>,
}

#[derive(Debug)]
#[must_use = "partial whole-EXEC preparation must be assembled or cancelled"]
pub struct ExecResourcePreparationOwner<
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    transaction: TransactionId,
    positions: BTreeSet<usize>,
    buffer_blits: Vec<PreparedBufferBlit>,
    image_blits: Vec<PreparedImageBlit>,
    compute_dispatches: Vec<PreparedComputeDispatch<NativeCompute, Compute>>,
    render_dispatches: Vec<PreparedRenderDispatch<NativeRender, Render>>,
    info_queries: Vec<PreparedInfoQuery>,
    indirect_range_readbacks: Vec<PreparedIndirectRangeReadback>,
    resource_states: Option<PreparedResourceStateBatch>,
    content_synchronization: Option<crate::PreparedContentSynchronizationBatch>,
}

#[derive(Debug)]
pub struct ExecResourceInputAdmissionFailure<Input> {
    pub reason: ExecResourcePreparationError,
    pub input: Input,
}

pub type ExecResourceInputAdmissionResult<Input> =
    Result<(), Box<ExecResourceInputAdmissionFailure<Input>>>;

#[derive(Debug)]
pub struct DispatchPreparationInput<Operation, Kind, NativePipeline> {
    pub operation: Operation,
    pub pipeline: ReadyPipelineLease<Kind, NativePipeline>,
}

#[derive(Debug)]
pub enum ExecResourcePreparationStepFailure<PreparationFailure, Input> {
    TransactionMismatch {
        expected: TransactionId,
        actual: TransactionId,
        input: Input,
    },
    PositionOccupied {
        position: usize,
        input: Input,
    },
    Preparation(PreparationFailure),
}

pub type ExecResourcePreparationStepResult<PreparationFailure, Input> =
    Result<(), Box<ExecResourcePreparationStepFailure<PreparationFailure, Input>>>;

pub type ComputeDispatchPreparationInput<NativePipeline> =
    DispatchPreparationInput<ResolvedComputeDispatch, ComputePipelineObject, NativePipeline>;
pub type ComputeDispatchPreparationStepResult<NativePipeline> = ExecResourcePreparationStepResult<
    (
        ComputeDispatchPreparationError,
        ComputeDispatchPreparationInput<NativePipeline>,
    ),
    ComputeDispatchPreparationInput<NativePipeline>,
>;
pub type RenderDispatchPreparationInput<NativePipeline> =
    DispatchPreparationInput<ResolvedRenderDispatch, RenderPipelineObject, NativePipeline>;
pub type RenderDispatchPreparationStepResult<NativePipeline> = ExecResourcePreparationStepResult<
    (
        crate::RenderDispatchPreparationError,
        RenderDispatchPreparationInput<NativePipeline>,
    ),
    RenderDispatchPreparationInput<NativePipeline>,
>;

impl<Compute, NativeCompute, Render, NativeRender>
    ExecResourcePreparationOwner<Compute, NativeCompute, Render, NativeRender>
{
    pub fn new(transaction: TransactionId) -> Self {
        Self {
            transaction,
            positions: BTreeSet::new(),
            buffer_blits: Vec::new(),
            image_blits: Vec::new(),
            compute_dispatches: Vec::new(),
            render_dispatches: Vec::new(),
            info_queries: Vec::new(),
            indirect_range_readbacks: Vec::new(),
            resource_states: None,
            content_synchronization: None,
        }
    }

    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    fn position_is_occupied(&self, position: usize) -> bool {
        self.positions.contains(&position)
    }

    pub fn first_occupied_position(
        &self,
        positions: impl IntoIterator<Item = usize>,
    ) -> Option<usize> {
        positions
            .into_iter()
            .find(|position| self.position_is_occupied(*position))
    }

    fn admit_position<Input>(
        &mut self,
        transaction: TransactionId,
        position: Option<usize>,
        input: Input,
        unpositioned: ExecResourcePreparationError,
    ) -> Result<Input, Box<ExecResourceInputAdmissionFailure<Input>>> {
        if transaction != self.transaction {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::TransactionMismatch { index: position },
                input,
            }));
        }
        let Some(position) = position else {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: unpositioned,
                input,
            }));
        };
        if !self.positions.insert(position) {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::DuplicatePosition(position),
                input,
            }));
        }
        Ok(input)
    }

    pub fn push_buffer_blit(
        &mut self,
        prepared: PreparedBufferBlit,
    ) -> ExecResourceInputAdmissionResult<PreparedBufferBlit> {
        let transaction = prepared.transaction();
        let position = prepared.write().operation_index();
        let prepared = self.admit_position(
            transaction,
            position,
            prepared,
            ExecResourcePreparationError::UnpositionedBufferBlit,
        )?;
        self.buffer_blits.push(prepared);
        Ok(())
    }

    pub fn push_image_blit(
        &mut self,
        prepared: PreparedImageBlit,
    ) -> ExecResourceInputAdmissionResult<PreparedImageBlit> {
        let transaction = prepared.transaction();
        let position = prepared.write().operation_index();
        let prepared = self.admit_position(
            transaction,
            position,
            prepared,
            ExecResourcePreparationError::UnpositionedImageBlit,
        )?;
        self.image_blits.push(prepared);
        Ok(())
    }

    pub fn push_compute(
        &mut self,
        prepared: PreparedComputeDispatch<NativeCompute, Compute>,
    ) -> ExecResourceInputAdmissionResult<PreparedComputeDispatch<NativeCompute, Compute>> {
        let transaction = prepared.transaction();
        let position = prepared.operation_index();
        let prepared = self.admit_position(
            transaction,
            Some(position),
            prepared,
            ExecResourcePreparationError::OperationAbsent(position),
        )?;
        self.compute_dispatches.push(prepared);
        Ok(())
    }

    pub fn push_render(
        &mut self,
        prepared: PreparedRenderDispatch<NativeRender, Render>,
    ) -> ExecResourceInputAdmissionResult<PreparedRenderDispatch<NativeRender, Render>> {
        let transaction = prepared.transaction();
        let position = prepared.operation_index();
        let prepared = self.admit_position(
            transaction,
            Some(position),
            prepared,
            ExecResourcePreparationError::OperationAbsent(position),
        )?;
        self.render_dispatches.push(prepared);
        Ok(())
    }

    pub fn push_info_query(
        &mut self,
        prepared: PreparedInfoQuery,
    ) -> ExecResourceInputAdmissionResult<PreparedInfoQuery> {
        let transaction = prepared.transaction();
        let position = prepared.index();
        let prepared = self.admit_position(
            transaction,
            Some(position),
            prepared,
            ExecResourcePreparationError::OperationAbsent(position),
        )?;
        self.info_queries.push(prepared);
        Ok(())
    }

    pub fn push_indirect_range(
        &mut self,
        prepared: PreparedIndirectRangeReadback,
    ) -> ExecResourceInputAdmissionResult<PreparedIndirectRangeReadback> {
        let transaction = prepared.transaction();
        let position = prepared.operation_index();
        let prepared = self.admit_position(
            transaction,
            Some(position),
            prepared,
            ExecResourcePreparationError::OperationAbsent(position),
        )?;
        self.indirect_range_readbacks.push(prepared);
        Ok(())
    }

    pub fn push_resource_states(
        &mut self,
        prepared: PreparedResourceStateBatch,
    ) -> ExecResourceInputAdmissionResult<PreparedResourceStateBatch> {
        if self.resource_states.is_some() {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::UnexpectedResourceStateBatch,
                input: prepared,
            }));
        }
        if prepared.transaction() != self.transaction {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::TransactionMismatch { index: None },
                input: prepared,
            }));
        }
        if let Some(position) = prepared
            .states()
            .iter()
            .map(PreparedResourceState::index)
            .find(|position| self.positions.contains(position))
        {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::DuplicatePosition(position),
                input: prepared,
            }));
        }
        self.positions
            .extend(prepared.states().iter().map(PreparedResourceState::index));
        self.resource_states = Some(prepared);
        Ok(())
    }

    pub fn push_content_synchronization(
        &mut self,
        prepared: crate::PreparedContentSynchronizationBatch,
    ) -> ExecResourceInputAdmissionResult<crate::PreparedContentSynchronizationBatch> {
        if self.content_synchronization.is_some() {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::UnexpectedContentSynchronizationBatch,
                input: prepared,
            }));
        }
        if prepared.transaction() != self.transaction {
            return Err(Box::new(ExecResourceInputAdmissionFailure {
                reason: ExecResourcePreparationError::TransactionMismatch { index: None },
                input: prepared,
            }));
        }
        self.content_synchronization = Some(prepared);
        Ok(())
    }

    pub fn host_ingresses(&self) -> Box<[crate::HostIngressKey]> {
        self.resource_states
            .as_ref()
            .map(PreparedResourceStateBatch::host_ingresses)
            .into_iter()
            .flatten()
            .chain(
                self.content_synchronization
                    .as_ref()
                    .into_iter()
                    .flat_map(|batch| batch.host_ingresses().iter().copied()),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Settle the resource-state prefix's CPU ingress obligations and move
    /// their lifecycle-authored GPU uploads into this same partial EXEC owner.
    pub fn publish_host_ingresses_after_copy<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
    ) -> Result<Box<[crate::TransferKey]>, crate::HostIngressBatchError> {
        let mut planned = match self.resource_states.as_mut() {
            Some(states) => states
                .publish_host_ingresses_after_copy(resources)?
                .into_vec(),
            None => Vec::new(),
        };
        if let Some(synchronization) = self.content_synchronization.as_mut() {
            planned.extend(synchronization.publish_host_ingresses_after_copy(resources)?);
        }
        Ok(planned.into_boxed_slice())
    }

    pub fn assemble<Indirect, Completion>(
        self,
        exec: &ExecTransaction<
            ResolvedOperation<Render, Compute, ResolvedInfoOperation, Indirect, Completion>,
        >,
        operation_base: usize,
    ) -> ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender>
    where
        Compute: PartialEq,
        Render: PartialEq,
        Indirect: crate::IndirectRangeResourceOperation,
    {
        let transaction = self.transaction;
        assemble_prepared_exec_resources_at(transaction, exec, self.into_inputs(), operation_base)
    }

    pub fn assemble_at_positions<Indirect, Completion>(
        self,
        exec: &ExecTransaction<
            ResolvedOperation<Render, Compute, ResolvedInfoOperation, Indirect, Completion>,
        >,
        operation_positions: &[usize],
    ) -> ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender>
    where
        Compute: PartialEq,
        Render: PartialEq,
        Indirect: crate::IndirectRangeResourceOperation,
    {
        let transaction = self.transaction;
        assemble_prepared_exec_resources_at_positions(
            transaction,
            exec,
            self.into_inputs(),
            operation_positions,
        )
    }

    pub fn assemble_at_origins<Indirect, Completion>(
        self,
        exec: &ExecTransaction<
            ResolvedOperation<Render, Compute, ResolvedInfoOperation, Indirect, Completion>,
        >,
        origins: &[crate::ExpandedIndirectOperationOrigin],
    ) -> ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender>
    where
        Compute: PartialEq,
        Render: PartialEq,
        Indirect: crate::IndirectRangeResourceOperation,
    {
        let positions = origins
            .iter()
            .map(|origin| origin.expanded_position)
            .collect::<Vec<_>>();
        self.assemble_at_positions(exec, &positions)
    }

    pub fn cancel<T>(
        self,
        owner: &mut ResourceLifecycleOwner<T>,
    ) -> ExecResourceCancellationResult<T, Compute, NativeCompute, Render, NativeRender> {
        let transaction = self.transaction;
        cancel_prepared_exec_resource_inputs(owner, transaction, self.into_inputs())
    }

    fn into_inputs(
        self,
    ) -> PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender> {
        PreparedExecResourceInputs {
            buffer_blits: self.buffer_blits.into_boxed_slice(),
            image_blits: self.image_blits.into_boxed_slice(),
            compute_dispatches: self.compute_dispatches.into_boxed_slice(),
            render_dispatches: self.render_dispatches.into_boxed_slice(),
            info_queries: self.info_queries.into_boxed_slice(),
            indirect_range_readbacks: self.indirect_range_readbacks.into_boxed_slice(),
            resource_states: self.resource_states,
            content_synchronization: self.content_synchronization,
        }
    }
}

impl<NativeCompute, NativeRender>
    ExecResourcePreparationOwner<
        ResolvedComputeDispatch,
        NativeCompute,
        ResolvedRenderDispatch,
        NativeRender,
    >
{
    pub fn prepare_buffer_blit<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        submission: SubmissionId,
        position: usize,
        operation: ResolvedBlit,
    ) -> ExecResourcePreparationStepResult<Box<BufferBlitPreparationFailure>, ResolvedBlit> {
        if self.position_is_occupied(position) {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::PositionOccupied {
                    position,
                    input: operation,
                },
            ));
        }
        let prepared = crate::prepare_buffer_blit_with_write(
            resources,
            self.transaction,
            crate::GpuWriteId::operation(submission, position),
            operation,
        )
        .map_err(|failure| Box::new(ExecResourcePreparationStepFailure::Preparation(failure)))?;
        self.push_buffer_blit(prepared)
            .expect("the owner supplied the exact vacant transaction position");
        Ok(())
    }

    pub fn prepare_image_blit<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        submission: SubmissionId,
        position: usize,
        operation: ResolvedBlit,
    ) -> ExecResourcePreparationStepResult<Box<ImageBlitPreparationFailure>, ResolvedBlit> {
        if self.position_is_occupied(position) {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::PositionOccupied {
                    position,
                    input: operation,
                },
            ));
        }
        let prepared = crate::prepare_image_blit_with_write(
            resources,
            self.transaction,
            crate::GpuWriteId::operation(submission, position),
            operation,
        )
        .map_err(|failure| Box::new(ExecResourcePreparationStepFailure::Preparation(failure)))?;
        self.push_image_blit(prepared)
            .expect("the owner supplied the exact vacant transaction position");
        Ok(())
    }

    pub fn prepare_compute<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        submission: SubmissionId,
        position: usize,
        operation: ResolvedComputeDispatch,
        pipeline: ReadyPipelineLease<ComputePipelineObject, NativeCompute>,
    ) -> ComputeDispatchPreparationStepResult<NativeCompute> {
        let input = DispatchPreparationInput {
            operation,
            pipeline,
        };
        if self.position_is_occupied(position) {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::PositionOccupied { position, input },
            ));
        }
        let recovery = DispatchPreparationInput {
            operation: input.operation.clone(),
            pipeline: input.pipeline.clone(),
        };
        let prepared = crate::prepare_compute_dispatch(
            resources,
            self.transaction,
            submission,
            position,
            input.operation,
            input.pipeline,
        )
        .map_err(|reason| {
            Box::new(ExecResourcePreparationStepFailure::Preparation((
                reason, recovery,
            )))
        })?;
        if self.push_compute(prepared).is_err() {
            unreachable!("the owner supplied the exact vacant transaction position");
        }
        Ok(())
    }

    pub fn prepare_render<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        submission: SubmissionId,
        position: usize,
        operation: ResolvedRenderDispatch,
        pipeline: ReadyPipelineLease<RenderPipelineObject, NativeRender>,
    ) -> RenderDispatchPreparationStepResult<NativeRender> {
        let input = DispatchPreparationInput {
            operation,
            pipeline,
        };
        if self.position_is_occupied(position) {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::PositionOccupied { position, input },
            ));
        }
        let recovery = DispatchPreparationInput {
            operation: input.operation.clone(),
            pipeline: input.pipeline.clone(),
        };
        let prepared = crate::prepare_render_dispatch(
            resources,
            self.transaction,
            submission,
            position,
            input.operation,
            input.pipeline,
        )
        .map_err(|reason| {
            Box::new(ExecResourcePreparationStepFailure::Preparation((
                reason, recovery,
            )))
        })?;
        if self.push_render(prepared).is_err() {
            unreachable!("the owner supplied the exact vacant transaction position");
        }
        Ok(())
    }

    pub fn prepare_info<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        submission: SubmissionId,
        evaluated: EvaluatedInfoQuery,
    ) -> ExecResourcePreparationStepResult<Box<InfoQueryPreparationFailure>, EvaluatedInfoQuery>
    {
        let position = evaluated.index();
        if evaluated.transaction() != self.transaction {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::TransactionMismatch {
                    expected: self.transaction,
                    actual: evaluated.transaction(),
                    input: evaluated,
                },
            ));
        }
        if self.position_is_occupied(position) {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::PositionOccupied {
                    position,
                    input: evaluated,
                },
            ));
        }
        let prepared =
            crate::prepare_info_query(resources, submission, evaluated).map_err(|failure| {
                Box::new(ExecResourcePreparationStepFailure::Preparation(failure))
            })?;
        self.push_info_query(prepared)
            .expect("the owner supplied the exact vacant transaction position");
        Ok(())
    }

    pub fn prepare_indirect_range<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        position: usize,
        operation: ResolvedIndirectCommand,
    ) -> ExecResourcePreparationStepResult<
        (crate::IndirectRangeReadbackError, ResolvedIndirectCommand),
        ResolvedIndirectCommand,
    > {
        if self.position_is_occupied(position) {
            return Err(Box::new(
                ExecResourcePreparationStepFailure::PositionOccupied {
                    position,
                    input: operation,
                },
            ));
        }
        let recovery = operation;
        let prepared = crate::prepare_indirect_range_readback(
            resources,
            self.transaction,
            position,
            operation,
        )
        .map_err(|reason| {
            Box::new(ExecResourcePreparationStepFailure::Preparation((
                reason, recovery,
            )))
        })?;
        self.push_indirect_range(prepared)
            .expect("the owner supplied the exact vacant transaction position");
        Ok(())
    }
}

#[derive(Debug)]
pub struct AcceptedExecResourceOutcomes<Compute = (), Render = ()> {
    pub buffer_blits: Box<[crate::ResolvedBlit]>,
    pub image_blits: Box<[crate::ResolvedBlit]>,
    pub compute_dispatches: Box<[Compute]>,
    pub render_dispatches: Box<[Render]>,
    pub info_queries: Box<[ResolvedInfoOperation]>,
    pub indirect_range_readbacks: Box<[PreparedIndirectRangeReadback]>,
    pub resource_states: Box<[crate::ResourceStateOutcome]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecResourceCancellationError {
    Writes(GpuWriteBatchError),
    Transfers(TransferBatchError),
    HostIngresses(crate::HostIngressBatchError),
    Uses(ResourceUseBatchError),
}

#[derive(Debug)]
pub struct ExecResourceCancellationFailure<
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ExecResourceCancellationError,
    pub prepared: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
}

pub type ExecResourceCancellationResult<T, Compute, NativeCompute, Render, NativeRender> = Result<
    CancelledExecResources<T, Compute, Render>,
    Box<ExecResourceCancellationFailure<Compute, NativeCompute, Render, NativeRender>>,
>;

#[derive(Debug)]
pub struct CancelledExecResources<T, Compute = (), Render = ()> {
    pub buffer_blits: Box<[crate::ResolvedBlit]>,
    pub image_blits: Box<[crate::ResolvedBlit]>,
    pub compute_dispatches: Box<[Compute]>,
    pub render_dispatches: Box<[Render]>,
    pub info_queries: Box<[EvaluatedInfoQuery]>,
    pub indirect_range_readbacks: Box<[crate::ResolvedIndirectCommand]>,
    pub resource_states: Box<[ResourceStateOutcome]>,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

impl<Compute, NativeCompute, Render, NativeRender>
    PreparedExecResources<Compute, NativeCompute, Render, NativeRender>
{
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn inputs(
        &self,
    ) -> &PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender> {
        &self.inputs
    }

    pub fn into_inputs(
        self,
    ) -> PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender> {
        self.inputs
    }

    pub const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    pub const fn resource_completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }

    pub const fn host_landings(&self) -> &[crate::HostLandingKey] {
        &self.host_landings
    }

    pub fn host_ingresses(&self) -> Box<[crate::HostIngressKey]> {
        self.inputs
            .resource_states
            .as_ref()
            .map(PreparedResourceStateBatch::host_ingresses)
            .into_iter()
            .flatten()
            .chain(
                self.inputs
                    .content_synchronization
                    .as_ref()
                    .into_iter()
                    .flat_map(|batch| batch.host_ingresses().iter().copied()),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn publish_host_ingresses_after_copy<T>(
        &mut self,
        owner: &mut ResourceLifecycleOwner<T>,
    ) -> Result<Box<[crate::TransferKey]>, crate::HostIngressBatchError> {
        let mut planned = match self.inputs.resource_states.as_mut() {
            Some(states) => states.publish_host_ingresses_after_copy(owner)?.into_vec(),
            None => Vec::new(),
        };
        if let Some(synchronization) = self.inputs.content_synchronization.as_mut() {
            planned.extend(synchronization.publish_host_ingresses_after_copy(owner)?);
        }
        Ok(planned.into_boxed_slice())
    }

    pub fn into_outcomes(self) -> AcceptedExecResourceOutcomes<Compute, Render> {
        AcceptedExecResourceOutcomes {
            buffer_blits: self
                .inputs
                .buffer_blits
                .into_vec()
                .into_iter()
                .map(|blit| blit.operation().clone())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            image_blits: self
                .inputs
                .image_blits
                .into_vec()
                .into_iter()
                .map(|blit| blit.operation().clone())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            compute_dispatches: self
                .inputs
                .compute_dispatches
                .into_vec()
                .into_iter()
                .map(PreparedComputeDispatch::into_operation)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            render_dispatches: self
                .inputs
                .render_dispatches
                .into_vec()
                .into_iter()
                .map(PreparedRenderDispatch::into_operation)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            info_queries: self
                .inputs
                .info_queries
                .into_vec()
                .into_iter()
                .map(|info| *info.operation())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            indirect_range_readbacks: self.inputs.indirect_range_readbacks,
            resource_states: self
                .inputs
                .resource_states
                .map(PreparedResourceStateBatch::into_outcomes)
                .unwrap_or_default(),
        }
    }
}

pub fn cancel_prepared_exec_resources<T, Compute, NativeCompute, Render, NativeRender>(
    owner: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
) -> ExecResourceCancellationResult<T, Compute, NativeCompute, Render, NativeRender> {
    if let Err(reason) = validate_cancel_prepared_exec_resources(owner, &prepared) {
        return Err(Box::new(ExecResourceCancellationFailure {
            reason,
            prepared,
        }));
    }
    let mut writes = Vec::new();
    let mut transfers = Vec::new();
    let mut uses = Vec::new();
    for blit in &prepared.inputs.buffer_blits {
        writes.extend_from_slice(blit.writes());
        uses.extend_from_slice(blit.uses());
    }
    for blit in &prepared.inputs.image_blits {
        writes.extend_from_slice(blit.writes());
        uses.extend_from_slice(blit.uses());
    }
    for compute in &prepared.inputs.compute_dispatches {
        writes.extend_from_slice(compute.writes());
        uses.extend_from_slice(compute.uses());
    }
    for render in &prepared.inputs.render_dispatches {
        writes.extend_from_slice(render.writes());
        uses.extend_from_slice(render.uses());
    }
    for info in &prepared.inputs.info_queries {
        writes.extend_from_slice(info.writes());
        uses.extend_from_slice(info.uses());
    }
    for readback in &prepared.inputs.indirect_range_readbacks {
        uses.extend_from_slice(readback.uses());
    }
    if let Some(states) = &prepared.inputs.resource_states {
        for state in states.states() {
            writes.extend_from_slice(state.gpu_reservations());
            transfers.extend_from_slice(state.transfers());
            uses.extend_from_slice(state.uses());
        }
    }
    if let Some(synchronization) = &prepared.inputs.content_synchronization {
        transfers.extend_from_slice(synchronization.transfers());
        uses.extend_from_slice(synchronization.uses());
    }
    owner
        .cancel_gpu_writes(&writes)
        .expect("whole-EXEC write cancellation was prevalidated");
    owner
        .cancel_transfers(&transfers)
        .expect("whole-EXEC transfer cancellation was prevalidated");
    if let Some(synchronization) = &prepared.inputs.content_synchronization {
        owner
            .cancel_host_ingresses(synchronization.host_ingresses())
            .expect("whole-EXEC host ingress cancellation was prevalidated");
    }
    let progress = owner
        .cancel_representation_uses(prepared.transaction, &uses)
        .expect("whole-EXEC representation cancellation was prevalidated");
    let PreparedExecResourceInputs {
        buffer_blits,
        image_blits,
        compute_dispatches,
        render_dispatches,
        info_queries,
        indirect_range_readbacks,
        resource_states,
        content_synchronization: _,
    } = prepared.inputs;
    Ok(CancelledExecResources {
        buffer_blits: buffer_blits
            .into_vec()
            .into_iter()
            .map(PreparedBufferBlit::into_operation)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        image_blits: image_blits
            .into_vec()
            .into_iter()
            .map(PreparedImageBlit::into_operation)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        compute_dispatches: compute_dispatches
            .into_vec()
            .into_iter()
            .map(PreparedComputeDispatch::into_operation)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        render_dispatches: render_dispatches
            .into_vec()
            .into_iter()
            .map(PreparedRenderDispatch::into_operation)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        info_queries: info_queries
            .into_vec()
            .into_iter()
            .map(PreparedInfoQuery::into_evaluated)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        indirect_range_readbacks: indirect_range_readbacks
            .into_vec()
            .into_iter()
            .map(|readback| readback.operation())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        resource_states: resource_states
            .map(PreparedResourceStateBatch::into_outcomes)
            .unwrap_or_default(),
        resources: progress,
    })
}

/// Cancel a partially constructed whole-EXEC family set before structural
/// assembly is possible. This is the rollback path for a late operation-local
/// preparation refusal: every token already acquired is validated and returned
/// together, even though the missing suffix means it cannot form a complete
/// [`PreparedExecResources`] yet.
pub fn cancel_prepared_exec_resource_inputs<T, Compute, NativeCompute, Render, NativeRender>(
    owner: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    inputs: PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender>,
) -> ExecResourceCancellationResult<T, Compute, NativeCompute, Render, NativeRender> {
    cancel_prepared_exec_resources(
        owner,
        PreparedExecResources {
            transaction,
            inputs,
            backings: Box::new([]),
            completions: Box::new([]),
            host_landings: Box::new([]),
        },
    )
}

pub fn validate_cancel_prepared_exec_resources<T, Compute, NativeCompute, Render, NativeRender>(
    owner: &ResourceLifecycleOwner<T>,
    prepared: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
) -> Result<(), ExecResourceCancellationError> {
    let mut writes = Vec::new();
    let mut transfers = Vec::new();
    let mut uses = Vec::new();
    for blit in &prepared.inputs.buffer_blits {
        writes.extend_from_slice(blit.writes());
        uses.extend_from_slice(blit.uses());
    }
    for blit in &prepared.inputs.image_blits {
        writes.extend_from_slice(blit.writes());
        uses.extend_from_slice(blit.uses());
    }
    for compute in &prepared.inputs.compute_dispatches {
        writes.extend_from_slice(compute.writes());
        uses.extend_from_slice(compute.uses());
    }
    for render in &prepared.inputs.render_dispatches {
        writes.extend_from_slice(render.writes());
        uses.extend_from_slice(render.uses());
    }
    for info in &prepared.inputs.info_queries {
        writes.extend_from_slice(info.writes());
        uses.extend_from_slice(info.uses());
    }
    for readback in &prepared.inputs.indirect_range_readbacks {
        uses.extend_from_slice(readback.uses());
    }
    if let Some(states) = &prepared.inputs.resource_states {
        for state in states.states() {
            writes.extend_from_slice(state.gpu_reservations());
            transfers.extend_from_slice(state.transfers());
            uses.extend_from_slice(state.uses());
        }
    }
    if let Some(synchronization) = &prepared.inputs.content_synchronization {
        transfers.extend_from_slice(synchronization.transfers());
        uses.extend_from_slice(synchronization.uses());
        owner
            .validate_cancel_host_ingresses(synchronization.host_ingresses())
            .map_err(ExecResourceCancellationError::HostIngresses)?;
    }
    owner
        .validate_cancel_gpu_writes(&writes)
        .map_err(ExecResourceCancellationError::Writes)?;
    owner
        .validate_cancel_transfers(&transfers)
        .map_err(ExecResourceCancellationError::Transfers)?;
    owner
        .validate_cancel_representation_uses(prepared.transaction, &uses)
        .map_err(ExecResourceCancellationError::Uses)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecResourcePreparationError {
    UnpositionedBufferBlit,
    UnpositionedImageBlit,
    TransactionMismatch { index: Option<usize> },
    DuplicatePosition(usize),
    OperationAbsent(usize),
    OperationMismatch(usize),
    PreparationMissing(usize),
    OperationPositionCountMismatch { operations: usize, positions: usize },
    DuplicateOperationPosition(usize),
    UnexpectedResourceStateBatch,
    UnexpectedContentSynchronizationBatch,
    ResourceStateBatchMissing,
}

#[derive(Debug)]
pub struct ExecResourcePreparationFailure<
    Compute = (),
    NativeCompute = (),
    Render = (),
    NativeRender = (),
> {
    pub reason: ExecResourcePreparationError,
    pub inputs: PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender>,
}

pub type ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender> = Result<
    PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    Box<ExecResourcePreparationFailure<Compute, NativeCompute, Render, NativeRender>>,
>;

pub fn assemble_prepared_exec_resources<
    Render,
    Compute: PartialEq,
    NativeCompute,
    NativeRender,
    Indirect,
    Completion,
>(
    transaction: TransactionId,
    exec: &ExecTransaction<
        ResolvedOperation<Render, Compute, ResolvedInfoOperation, Indirect, Completion>,
    >,
    inputs: PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender>,
) -> ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender>
where
    Render: PartialEq,
    Indirect: crate::IndirectRangeResourceOperation,
{
    assemble_prepared_exec_resources_at(transaction, exec, inputs, 0)
}

/// Assemble one continuation phase while matching every preparation against
/// its flattened position in the original EXEC.
pub fn assemble_prepared_exec_resources_at<
    Render,
    Compute: PartialEq,
    NativeCompute,
    NativeRender,
    Indirect,
    Completion,
>(
    transaction: TransactionId,
    exec: &ExecTransaction<
        ResolvedOperation<Render, Compute, ResolvedInfoOperation, Indirect, Completion>,
    >,
    inputs: PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender>,
    operation_base: usize,
) -> ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender>
where
    Render: PartialEq,
    Indirect: crate::IndirectRangeResourceOperation,
{
    let operation_count = exec.operations().count();
    let Some(operation_end) = operation_base.checked_add(operation_count) else {
        return Err(Box::new(ExecResourcePreparationFailure {
            reason: ExecResourcePreparationError::OperationAbsent(usize::MAX),
            inputs,
        }));
    };
    assemble_prepared_exec_resources_at_positions(
        transaction,
        exec,
        inputs,
        &(operation_base..operation_end).collect::<Vec<_>>(),
    )
}

/// Assemble an expanded continuation whose exact source positions need not
/// form a contiguous suffix after literal ICB commands are removed.
pub fn assemble_prepared_exec_resources_at_positions<
    Render,
    Compute: PartialEq,
    NativeCompute,
    NativeRender,
    Indirect,
    Completion,
>(
    transaction: TransactionId,
    exec: &ExecTransaction<
        ResolvedOperation<Render, Compute, ResolvedInfoOperation, Indirect, Completion>,
    >,
    inputs: PreparedExecResourceInputs<Compute, NativeCompute, Render, NativeRender>,
    operation_positions: &[usize],
) -> ExecResourcePreparationResult<Compute, NativeCompute, Render, NativeRender>
where
    Render: PartialEq,
    Indirect: crate::IndirectRangeResourceOperation,
{
    let fail = |reason, inputs| Box::new(ExecResourcePreparationFailure { reason, inputs });
    let mut positions = BTreeSet::new();
    let mut buffer_blits = BTreeMap::new();
    for blit in &inputs.buffer_blits {
        if blit.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch {
                    index: blit.write().operation_index(),
                },
                inputs,
            ));
        }
        let Some(index) = blit.write().operation_index() else {
            return Err(fail(
                ExecResourcePreparationError::UnpositionedBufferBlit,
                inputs,
            ));
        };
        if !positions.insert(index) {
            return Err(fail(
                ExecResourcePreparationError::DuplicatePosition(index),
                inputs,
            ));
        }
        buffer_blits.insert(index, blit);
    }
    let mut image_blits = BTreeMap::new();
    for blit in &inputs.image_blits {
        if blit.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch {
                    index: blit.write().operation_index(),
                },
                inputs,
            ));
        }
        let Some(index) = blit.write().operation_index() else {
            return Err(fail(
                ExecResourcePreparationError::UnpositionedImageBlit,
                inputs,
            ));
        };
        if !positions.insert(index) {
            return Err(fail(
                ExecResourcePreparationError::DuplicatePosition(index),
                inputs,
            ));
        }
        image_blits.insert(index, blit);
    }
    let mut compute_dispatches = BTreeMap::new();
    for compute in &inputs.compute_dispatches {
        if compute.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch {
                    index: Some(compute.operation_index()),
                },
                inputs,
            ));
        }
        if !positions.insert(compute.operation_index()) {
            return Err(fail(
                ExecResourcePreparationError::DuplicatePosition(compute.operation_index()),
                inputs,
            ));
        }
        compute_dispatches.insert(compute.operation_index(), compute);
    }
    let mut render_dispatches = BTreeMap::new();
    for render in &inputs.render_dispatches {
        if render.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch {
                    index: Some(render.operation_index()),
                },
                inputs,
            ));
        }
        if !positions.insert(render.operation_index()) {
            return Err(fail(
                ExecResourcePreparationError::DuplicatePosition(render.operation_index()),
                inputs,
            ));
        }
        render_dispatches.insert(render.operation_index(), render);
    }
    let mut info_queries = BTreeMap::new();
    for info in &inputs.info_queries {
        if info.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch {
                    index: Some(info.index()),
                },
                inputs,
            ));
        }
        if !positions.insert(info.index()) {
            return Err(fail(
                ExecResourcePreparationError::DuplicatePosition(info.index()),
                inputs,
            ));
        }
        info_queries.insert(info.index(), info);
    }
    let mut indirect_range_readbacks = BTreeMap::new();
    for readback in &inputs.indirect_range_readbacks {
        if readback.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch {
                    index: Some(readback.operation_index()),
                },
                inputs,
            ));
        }
        if !positions.insert(readback.operation_index()) {
            return Err(fail(
                ExecResourcePreparationError::DuplicatePosition(readback.operation_index()),
                inputs,
            ));
        }
        indirect_range_readbacks.insert(readback.operation_index(), readback);
    }
    let mut resource_states = BTreeMap::new();
    if let Some(states) = &inputs.resource_states {
        if states.transaction() != transaction {
            return Err(fail(
                ExecResourcePreparationError::TransactionMismatch { index: None },
                inputs,
            ));
        }
        for state in states.states() {
            if !positions.insert(state.index()) {
                return Err(fail(
                    ExecResourcePreparationError::DuplicatePosition(state.index()),
                    inputs,
                ));
            }
            resource_states.insert(state.index(), state);
        }
    }
    if inputs
        .content_synchronization
        .as_ref()
        .is_some_and(|batch| batch.transaction() != transaction)
    {
        return Err(fail(
            ExecResourcePreparationError::TransactionMismatch { index: None },
            inputs,
        ));
    }

    let operations = exec.operations().collect::<Vec<_>>();
    if operations.len() != operation_positions.len() {
        return Err(fail(
            ExecResourcePreparationError::OperationPositionCountMismatch {
                operations: operations.len(),
                positions: operation_positions.len(),
            },
            inputs,
        ));
    }
    let mut unique_operation_positions = BTreeSet::new();
    if let Some(position) = operation_positions
        .iter()
        .copied()
        .find(|position| !unique_operation_positions.insert(*position))
    {
        return Err(fail(
            ExecResourcePreparationError::DuplicateOperationPosition(position),
            inputs,
        ));
    }
    for (local_index, operation) in operations.iter().copied().enumerate() {
        let index = operation_positions[local_index];
        let matches = match operation {
            ResolvedOperation::Blit(operation) => {
                buffer_blits
                    .get(&index)
                    .is_some_and(|prepared| prepared.operation() == operation.as_ref())
                    || image_blits
                        .get(&index)
                        .is_some_and(|prepared| prepared.operation() == operation.as_ref())
            }
            ResolvedOperation::InfoQuery(operation) => info_queries
                .get(&index)
                .is_some_and(|prepared| prepared.operation() == operation),
            ResolvedOperation::Compute(operation) => compute_dispatches
                .get(&index)
                .is_some_and(|prepared| prepared.operation() == operation),
            ResolvedOperation::Render(operation) => render_dispatches
                .get(&index)
                .is_some_and(|prepared| operation == prepared.operation()),
            ResolvedOperation::ResourceState(operation) => resource_states
                .get(&index)
                .is_some_and(|prepared| prepared.operation() == operation),
            ResolvedOperation::IndirectCommand(operation) => {
                if !operation.requires_range_readback() {
                    continue;
                }
                indirect_range_readbacks
                    .get(&index)
                    .is_some_and(|prepared| operation.matches_range_readback(prepared))
            }
            _ => continue,
        };
        if !matches {
            let reason = if positions.contains(&index) {
                ExecResourcePreparationError::OperationMismatch(index)
            } else {
                ExecResourcePreparationError::PreparationMissing(index)
            };
            return Err(fail(reason, inputs));
        }
    }
    if let Some(index) = positions
        .iter()
        .copied()
        .find(|index| !unique_operation_positions.contains(index))
    {
        return Err(fail(
            ExecResourcePreparationError::OperationAbsent(index),
            inputs,
        ));
    }
    let operations_by_position = operation_positions
        .iter()
        .copied()
        .zip(operations.iter().copied())
        .collect::<BTreeMap<_, _>>();
    if let Some(index) = positions.iter().copied().find(|index| {
        !matches!(
            operations_by_position[index],
            ResolvedOperation::Blit(_)
                | ResolvedOperation::Compute(_)
                | ResolvedOperation::Render(_)
                | ResolvedOperation::InfoQuery(_)
                | ResolvedOperation::IndirectCommand(_)
                | ResolvedOperation::ResourceState(_)
        )
    }) {
        return Err(fail(
            ExecResourcePreparationError::OperationMismatch(index),
            inputs,
        ));
    }
    if resource_states.is_empty() && inputs.resource_states.is_some() {
        return Err(fail(
            ExecResourcePreparationError::UnexpectedResourceStateBatch,
            inputs,
        ));
    }
    if operations
        .iter()
        .any(|operation| matches!(operation, ResolvedOperation::ResourceState(_)))
        && inputs.resource_states.is_none()
    {
        return Err(fail(
            ExecResourcePreparationError::ResourceStateBatchMissing,
            inputs,
        ));
    }

    let mut backings = BTreeSet::new();
    let mut completions = Vec::new();
    let mut transfers = BTreeSet::new();
    let mut host_landings = BTreeSet::new();
    if let Some(synchronization) = &inputs.content_synchronization {
        backings.extend(synchronization.backings());
        completions.extend(synchronization.resource_completions());
        transfers.extend(synchronization.transfers().iter().copied());
    }
    for (local_index, operation) in operations.into_iter().enumerate() {
        let index = operation_positions[local_index];
        match operation {
            ResolvedOperation::Blit(_) => {
                if let Some(prepared) = buffer_blits.get(&index) {
                    backings.extend(prepared.backings());
                    completions.extend(prepared.resource_completions());
                } else if let Some(prepared) = image_blits.get(&index) {
                    backings.extend(prepared.backings());
                    completions.extend(prepared.resource_completions());
                }
            }
            ResolvedOperation::InfoQuery(_) => {
                let prepared = info_queries[&index];
                backings.insert(prepared.destination().backing);
                completions.extend(prepared.resource_completions());
            }
            ResolvedOperation::Compute(_) => {
                let prepared = compute_dispatches[&index];
                backings.extend(prepared.uses().iter().map(|use_| use_.backing));
                completions.extend(prepared.completions().iter().copied());
            }
            ResolvedOperation::Render(_) => {
                let prepared = render_dispatches[&index];
                backings.extend(prepared.uses().iter().map(|use_| use_.backing));
                completions.extend(prepared.completions().iter().copied());
            }
            ResolvedOperation::ResourceState(_) => {
                let prepared = resource_states[&index];
                backings.extend(prepared.backings());
                completions.extend(prepared.resource_completions().iter().copied());
                completions.extend(
                    prepared
                        .transfers()
                        .iter()
                        .copied()
                        .filter(|transfer| transfers.insert(*transfer))
                        .map(ResolvedResourceCompletion::Transfer),
                );
                host_landings.extend(prepared.host_landings().iter().copied());
            }
            ResolvedOperation::IndirectCommand(operation)
                if operation.requires_range_readback() =>
            {
                backings.insert(indirect_range_readbacks[&index].arguments_backing());
            }
            _ => {}
        }
    }
    Ok(PreparedExecResources {
        transaction,
        inputs,
        backings: backings.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        completions: completions.into_boxed_slice(),
        host_landings: host_landings
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        prepare_buffer_blit_with_write, BackingRegion, BufferFillPattern, GpuWriteId, LinearRange,
        RepresentationRoute, ResolvedBufferRange, ResolvedExecSegment, ResolvedExecStream,
        ResolvedResourceLifecycle, ResourceLifecycleEffect, ResourceLifecycleOwner, StorageBacking,
    };
    use reims_vgpu_protocol::{
        ByteLength, GuestVirtualAddress, SegmentBoundary, SegmentKind, SubmissionId,
        SubmissionIdentity, TaskId, VulkanDeviceEpochId,
    };

    fn fill(backing: BackingId, index: u32) -> crate::ResolvedBlit {
        crate::ResolvedBlit::Fill {
            destination: ResolvedBufferRange {
                resource: reims_vgpu_protocol::ResourceId::new(index + 1, 1),
                storage: backing,
                region: LinearRange::new(u64::from(index) * 16, 16).unwrap(),
                address: GuestVirtualAddress::new(0x1000 + u64::from(index) * 16),
                length: ByteLength::new(16),
            },
            pattern: BufferFillPattern::Byte(index as u8),
        }
    }

    #[test]
    fn partial_exec_resource_inputs_cancel_every_acquired_obligation() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 32).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let transaction = TransactionId::new(6);
        let submission = SubmissionId::new(8);
        let operation = fill(backing, 0);
        let prepared = prepare_buffer_blit_with_write(
            &mut resources,
            transaction,
            GpuWriteId::operation(submission, 0),
            operation.clone(),
        )
        .unwrap();
        let mut partial = ExecResourcePreparationOwner::<(), (), (), ()>::new(transaction);
        partial.push_buffer_blit(prepared).unwrap();
        let cancelled = partial.cancel(&mut resources).unwrap();
        assert_eq!(
            cancelled.buffer_blits.as_ref(),
            std::slice::from_ref(&operation)
        );
        assert_eq!(cancelled.resources.len(), 1);

        assert!(prepare_buffer_blit_with_write(
            &mut resources,
            transaction,
            GpuWriteId::operation(submission, 0),
            operation,
        )
        .is_ok());
    }

    #[test]
    fn whole_exec_resources_derive_one_backing_and_ordered_operation_completions() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 32).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let transaction = TransactionId::new(7);
        let submission = SubmissionId::new(9);
        let first_operation = fill(backing, 0);
        let second_operation = fill(backing, 1);
        let first = prepare_buffer_blit_with_write(
            &mut resources,
            transaction,
            GpuWriteId::operation(submission, 0),
            first_operation.clone(),
        )
        .unwrap();
        let second = prepare_buffer_blit_with_write(
            &mut resources,
            transaction,
            GpuWriteId::operation(submission, 1),
            second_operation.clone(),
        )
        .unwrap();
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: submission,
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
                    operations: Box::new([
                        ResolvedOperation::<(), (), ResolvedInfoOperation, (), ()>::Blit(Box::new(
                            first_operation,
                        )),
                        ResolvedOperation::Blit(Box::new(second_operation)),
                    ]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let failure = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([first]),
                image_blits: Box::new([]),
                compute_dispatches: Box::<[PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches: Box::<[PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ExecResourcePreparationError::PreparationMissing(1)
        );
        let first = failure.inputs.buffer_blits.into_vec().pop().unwrap();
        let prepared = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([first, second]),
                image_blits: Box::new([]),
                compute_dispatches: Box::<[PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches: Box::<[PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        assert_eq!(prepared.backings(), [backing]);
        assert_eq!(prepared.resource_completions().len(), 2);
        assert!(matches!(
            prepared.resource_completions(),
            [ResolvedResourceCompletion::GpuWrite { write: first, .. },
             ResolvedResourceCompletion::GpuWrite { write: second, .. }]
                if *first == GpuWriteId::operation(submission, 0)
                    && *second == GpuWriteId::operation(submission, 1)
        ));
        let cancelled = cancel_prepared_exec_resources(&mut resources, prepared).unwrap();
        assert_eq!(cancelled.buffer_blits.len(), 2);
        assert_eq!(cancelled.resources.len(), 1);

        let retry = assemble_prepared_exec_resources(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([
                    prepare_buffer_blit_with_write(
                        &mut resources,
                        transaction,
                        GpuWriteId::operation(submission, 0),
                        fill(backing, 0),
                    )
                    .unwrap(),
                    prepare_buffer_blit_with_write(
                        &mut resources,
                        transaction,
                        GpuWriteId::operation(submission, 1),
                        fill(backing, 1),
                    )
                    .unwrap(),
                ]),
                image_blits: Box::new([]),
                compute_dispatches: Box::<[PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches: Box::<[PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([]),
                resource_states: None,
                content_synchronization: None,
            },
        )
        .unwrap();
        cancel_prepared_exec_resources(&mut resources, retry).unwrap();
    }

    #[test]
    fn continuation_phase_owns_the_exact_global_range_position_until_cancellation() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 32).unwrap())]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let transaction = TransactionId::new(18);
        let operation = crate::ResolvedIndirectCommand::ExecuteIndirectRange {
            icb: reims_vgpu_protocol::ResourceId::new(3, 2),
            arguments_resource: reims_vgpu_protocol::ResourceId::new(4, 1),
            arguments_backing: backing,
            arguments_range: LinearRange::new(8, 8).unwrap(),
            kind: crate::IndirectCommandExecutionKind::Render,
        };
        let readback =
            crate::prepare_indirect_range_readback(&mut resources, transaction, 7, operation)
                .unwrap();
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(19),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Render,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::<
                        (),
                        (),
                        ResolvedInfoOperation,
                        crate::ResolvedIndirectCommand,
                        (),
                    >::IndirectCommand(operation)]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        let prepared = assemble_prepared_exec_resources_at(
            transaction,
            &exec,
            PreparedExecResourceInputs {
                buffer_blits: Box::new([]),
                image_blits: Box::new([]),
                compute_dispatches: Box::<[PreparedComputeDispatch<(), ()>]>::default(),
                render_dispatches: Box::<[PreparedRenderDispatch<(), ()>]>::default(),
                info_queries: Box::new([]),
                indirect_range_readbacks: Box::new([readback]),
                resource_states: None,
                content_synchronization: None,
            },
            7,
        )
        .unwrap();
        assert_eq!(prepared.backings(), [backing]);
        assert_eq!(prepared.inputs().indirect_range_readbacks.len(), 1);
        let cancelled = cancel_prepared_exec_resources(&mut resources, prepared).unwrap();
        assert_eq!(cancelled.indirect_range_readbacks.as_ref(), &[operation]);
        assert_eq!(cancelled.resources.len(), 1);
    }
}
