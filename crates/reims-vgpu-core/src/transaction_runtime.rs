//! Atomic ownership boundary for resolved transaction replay.
//!
//! This owner composes ingress position allocation, directional encoder
//! continuation, semantic dependency compilation, and guest publication.
//! Every mutating operation is prepared on cloned owners and committed only
//! when all participating invariants accept it. It performs no native work and
//! is therefore suitable for direct replay and backend conformance adapters.

use crate::{
    ConditionOwnerError, ConditionPublicationBoundary, ConditionSignal, ConditionWaitResolution,
    ContinuationError, CoordinationError, DependencyCoordinator, DeviceTransaction,
    DeviceTransactionPayload, EncoderContinuationOwner, ExplicitWaitCause, PublicationError,
    PublicationFact, PublicationOwner, PublishedFact, ReadyTransaction, RecordingOrderError,
    RecordingOrderOwner, ResolvedOperation, SessionGeneration, SubmissionOrderError,
    SubmissionOrderOwner, SubmissionReady, SynchronizationConditionOwner, TransactionIngressError,
    TransactionIngressOwner, TransactionPrerequisite, WaitDependencyCause,
};
use reims_vgpu_protocol::{ChannelId, PublicationDomainId, SubmissionDomainId, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedTransactionPrerequisite {
    pub prerequisite: TransactionPrerequisite,
    pub resolution: PrerequisiteResolution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrerequisiteResolution {
    Satisfied,
    Producer(TransactionId),
    /// No satisfying signal transaction has been admitted yet. The packet is
    /// retained but cannot reach a native queue until this condition binds or
    /// is observed satisfied.
    Pending,
}

impl ResolvedTransactionPrerequisite {
    fn explicit_wait(self) -> Option<(TransactionId, ExplicitWaitCause)> {
        let PrerequisiteResolution::Producer(producer) = self.resolution else {
            return None;
        };
        Some((producer, self.cause()))
    }

    fn unresolved(self) -> Option<ExplicitWaitCause> {
        (self.resolution == PrerequisiteResolution::Pending).then(|| self.cause())
    }

    fn cause(self) -> ExplicitWaitCause {
        match self.prerequisite {
            TransactionPrerequisite::Stamp { wait } => ExplicitWaitCause::Stamp {
                source_channel: ChannelId::new(wait.index),
                value: wait.value,
            },
            TransactionPrerequisite::Event { event, value } => {
                ExplicitWaitCause::Event { event, value }
            }
            TransactionPrerequisite::Fence { fence, generation } => {
                ExplicitWaitCause::Fence { fence, generation }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransactionRuntimeError {
    Ingress(TransactionIngressError),
    Continuation(ContinuationError),
    Coordination(CoordinationError),
    Recording(RecordingOrderError),
    Submission(SubmissionOrderError),
    Publication(PublicationError),
    Condition(ConditionOwnerError),
    NonStampEnvelopePrerequisite(Box<TransactionPrerequisite>),
    DuplicateChannel,
    UnknownChannel,
    UnknownPrerequisiteChannel(ChannelId),
    ChannelHasLiveTransactions,
    UnknownTransaction,
    TransactionNotSubmitted,
    TransactionNotPublished,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransactionRuntimeDiagnostic {
    pub transaction: TransactionId,
    pub channel: ChannelId,
    pub unresolved: Box<[ExplicitWaitCause]>,
    pub semantic_ready: bool,
    pub continuation_owned: bool,
    pub submitted: bool,
    pub published: bool,
}

pub type ResolvedExecDeviceTransaction<Render, Compute, Info, Indirect, Effect> =
    DeviceTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Effect>, (), (), (), ()>;

pub type AdmittedExecParts<Render, Compute, Info, Indirect, Effect> = (
    ResolvedExecDeviceTransaction<Render, Compute, Info, Indirect, Effect>,
    crate::AdmittedExecConditions,
    crate::AdmittedCompletionEffects<Effect>,
    crate::AdmittedResourceStates,
    crate::AdmittedInfoQueries<Info>,
);

#[derive(Clone, Debug)]
pub struct AdmittedExecTransaction<Render, Compute, Info, Indirect, Effect> {
    pub transaction: ResolvedExecDeviceTransaction<Render, Compute, Info, Indirect, Effect>,
    conditions: crate::AdmittedExecConditions,
    completion_effects: crate::AdmittedCompletionEffects<Effect>,
    resource_states: crate::AdmittedResourceStates,
    info_queries: crate::AdmittedInfoQueries<Info>,
}

#[derive(Clone, Debug)]
pub struct AdmittedExpandedExecProofs<Info, Effect> {
    conditions: crate::AdmittedExecConditions,
    completion_effects: crate::AdmittedCompletionEffects<Effect>,
    resource_states: crate::AdmittedResourceStates,
    info_queries: crate::AdmittedInfoQueries<Info>,
    indirect_commands: crate::AdmittedIndirectCommands<crate::ResolvedIndirectCommand>,
}

impl<Info, Effect> AdmittedExpandedExecProofs<Info, Effect> {
    pub const fn conditions(&self) -> &crate::AdmittedExecConditions {
        &self.conditions
    }

    pub const fn completion_effects(&self) -> &crate::AdmittedCompletionEffects<Effect> {
        &self.completion_effects
    }

    pub const fn resource_states(&self) -> &crate::AdmittedResourceStates {
        &self.resource_states
    }

    pub const fn info_queries(&self) -> &crate::AdmittedInfoQueries<Info> {
        &self.info_queries
    }

    pub const fn indirect_commands(
        &self,
    ) -> &crate::AdmittedIndirectCommands<crate::ResolvedIndirectCommand> {
        &self.indirect_commands
    }
}

impl<Info: Clone, Effect: Clone> AdmittedExpandedExecProofs<Info, Effect> {
    pub fn shifted(&self, base: usize) -> Option<Self> {
        Some(Self {
            conditions: self.conditions.shifted(base)?,
            completion_effects: self.completion_effects.shifted(base)?,
            resource_states: self.resource_states.shifted(base)?,
            info_queries: self.info_queries.shifted(base)?,
            indirect_commands: self.indirect_commands.shifted(base)?,
        })
    }

    pub fn remapped_positions(&self, positions: &[usize]) -> Option<Self> {
        Some(Self {
            conditions: self.conditions.remapped_positions(positions)?,
            completion_effects: self.completion_effects.remapped_positions(positions)?,
            resource_states: self.resource_states.remapped_positions(positions)?,
            info_queries: self.info_queries.remapped_positions(positions)?,
            indirect_commands: self.indirect_commands.remapped_positions(positions)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpandedExecProofError {
    TransactionMismatch,
    Indirect(crate::IndirectCommandAdmissionError),
}

impl<Render, Compute, Info, Indirect, Effect>
    AdmittedExecTransaction<Render, Compute, Info, Indirect, Effect>
{
    pub const fn conditions(&self) -> &crate::AdmittedExecConditions {
        &self.conditions
    }

    pub const fn completion_effects(&self) -> &crate::AdmittedCompletionEffects<Effect> {
        &self.completion_effects
    }

    pub const fn resource_states(&self) -> &crate::AdmittedResourceStates {
        &self.resource_states
    }

    pub const fn info_queries(&self) -> &crate::AdmittedInfoQueries<Info> {
        &self.info_queries
    }

    pub fn into_parts(self) -> AdmittedExecParts<Render, Compute, Info, Indirect, Effect> {
        (
            self.transaction,
            self.conditions,
            self.completion_effects,
            self.resource_states,
            self.info_queries,
        )
    }
}

impl<Render: Clone, Compute: Clone, Info: Clone, Effect: Clone>
    AdmittedExecTransaction<Render, Compute, Info, crate::ResolvedIndirectCommand, Effect>
{
    /// Re-derive position-bearing recording proofs from the exact literal ICB
    /// expansion. Transaction conditions were already installed from the
    /// original EXEC; these values authorize only recording of the expanded
    /// immutable stream and cannot change scheduling semantics.
    pub fn proofs_for_prepared_indirect_execution(
        &self,
        prepared: &crate::PreparedIndirectExecution<Render, Compute, Info, Effect>,
        owner: &crate::IndirectCommandSlotOwner<
            crate::ResolvedIndirectCommandSlot<Render, Compute>,
        >,
    ) -> Result<AdmittedExpandedExecProofs<Info, Effect>, ExpandedExecProofError> {
        let transaction = self.transaction.id;
        if prepared.committed().transaction != transaction {
            return Err(ExpandedExecProofError::TransactionMismatch);
        }
        let exec = prepared.exec();
        let indirect_commands = crate::admit_indirect_commands_with_owner(transaction, exec, owner)
            .map_err(ExpandedExecProofError::Indirect)?;
        Ok(AdmittedExpandedExecProofs {
            conditions: crate::AdmittedExecConditions::from_exec(transaction, exec),
            completion_effects: crate::AdmittedCompletionEffects::from_exec(transaction, exec),
            resource_states: crate::AdmittedResourceStates::from_exec(transaction, exec),
            info_queries: crate::AdmittedInfoQueries::from_exec(transaction, exec),
            indirect_commands,
        })
    }
}

impl From<TransactionIngressError> for TransactionRuntimeError {
    fn from(error: TransactionIngressError) -> Self {
        Self::Ingress(error)
    }
}

impl From<ContinuationError> for TransactionRuntimeError {
    fn from(error: ContinuationError) -> Self {
        Self::Continuation(error)
    }
}

impl From<CoordinationError> for TransactionRuntimeError {
    fn from(error: CoordinationError) -> Self {
        Self::Coordination(error)
    }
}

impl From<RecordingOrderError> for TransactionRuntimeError {
    fn from(error: RecordingOrderError) -> Self {
        Self::Recording(error)
    }
}

impl From<SubmissionOrderError> for TransactionRuntimeError {
    fn from(error: SubmissionOrderError) -> Self {
        Self::Submission(error)
    }
}

impl From<PublicationError> for TransactionRuntimeError {
    fn from(error: PublicationError) -> Self {
        Self::Publication(error)
    }
}

impl From<ConditionOwnerError> for TransactionRuntimeError {
    fn from(error: ConditionOwnerError) -> Self {
        Self::Condition(error)
    }
}

#[derive(Clone, Copy, Debug)]
struct ChannelState {
    admitted_once: bool,
    exec_admitted_once: bool,
}

#[derive(Clone, Copy, Debug)]
struct TransactionState {
    channel: ChannelId,
    ingress_ordinal: reims_vgpu_protocol::IngressOrdinal,
    continuation_owned: bool,
    recording_predecessor: Option<TransactionId>,
    submitted: bool,
    published: bool,
}

#[derive(Clone, Debug)]
pub struct TransactionRuntime<Completion> {
    ingress: TransactionIngressOwner,
    continuations: EncoderContinuationOwner,
    recording: RecordingOrderOwner,
    submission_order: SubmissionOrderOwner,
    dependencies: DependencyCoordinator,
    publication: PublicationOwner<Completion>,
    conditions: SynchronizationConditionOwner,
    channels: BTreeMap<ChannelId, ChannelState>,
    transactions: BTreeMap<TransactionId, TransactionState>,
    submission_dependencies: BTreeMap<TransactionId, Box<[(TransactionId, WaitDependencyCause)]>>,
    submission_hazards: BTreeMap<TransactionId, Box<[crate::HazardRequirement]>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransactionRecordingPlan {
    pub transaction: TransactionId,
    pub domain: SubmissionDomainId,
    pub continuation_predecessor: Option<TransactionId>,
}

impl<Completion: Clone> TransactionRuntime<Completion> {
    pub fn new(generation: SessionGeneration) -> Self {
        Self {
            ingress: TransactionIngressOwner::new(generation),
            continuations: EncoderContinuationOwner::default(),
            recording: RecordingOrderOwner::default(),
            submission_order: SubmissionOrderOwner::default(),
            dependencies: DependencyCoordinator::default(),
            publication: PublicationOwner::default(),
            conditions: SynchronizationConditionOwner::default(),
            channels: BTreeMap::new(),
            transactions: BTreeMap::new(),
            submission_dependencies: BTreeMap::new(),
            submission_hazards: BTreeMap::new(),
        }
    }

    pub fn session_generation(&self) -> reims_vgpu_protocol::SessionGenerationId {
        self.ingress.session_generation()
    }

    /// Construct the next semantic generation without reusing identities that
    /// may still be named by epoch-owned submitted work.
    pub fn successor_generation(&self, generation: SessionGeneration) -> Self {
        Self {
            ingress: self.ingress.successor_generation(generation),
            continuations: EncoderContinuationOwner::default(),
            recording: RecordingOrderOwner::default(),
            submission_order: SubmissionOrderOwner::default(),
            dependencies: DependencyCoordinator::default(),
            publication: PublicationOwner::default(),
            conditions: SynchronizationConditionOwner::default(),
            channels: BTreeMap::new(),
            transactions: BTreeMap::new(),
            submission_dependencies: BTreeMap::new(),
            submission_hazards: BTreeMap::new(),
        }
    }

    pub fn define_channel(&mut self, channel: ChannelId) -> Result<(), TransactionRuntimeError> {
        if self.channels.contains_key(&channel) {
            return Err(TransactionRuntimeError::DuplicateChannel);
        }
        let mut ingress = self.ingress.clone();
        ingress.define_channel(channel)?;
        self.ingress = ingress;
        self.channels.insert(
            channel,
            ChannelState {
                admitted_once: false,
                exec_admitted_once: false,
            },
        );
        Ok(())
    }

    /// Atomically admit one transaction and every condition signal encoded in
    /// its immutable operation stream. A signal-binding cycle or duplicate
    /// signal leaves ingress, publication, dependency, and condition ownership
    /// unchanged.
    fn admit_resolved_with_signals<Operation, Lifecycle, Query, Present, Control>(
        &mut self,
        channel: ChannelId,
        prerequisites: impl Into<Box<[ResolvedTransactionPrerequisite]>>,
        completion_stamp: Option<crate::CompletionStamp>,
        payload: DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>,
        signals: impl IntoIterator<Item = ConditionSignal>,
    ) -> Result<
        DeviceTransaction<Operation, Lifecycle, Query, Present, Control>,
        TransactionRuntimeError,
    > {
        let mut next = self.clone();
        let transaction =
            next.admit_resolved_inner(channel, prerequisites, completion_stamp, payload)?;
        next.register_signals(
            transaction.id,
            signals
                .into_iter()
                .chain(completion_stamp.map(|stamp| ConditionSignal::Stamp {
                    channel,
                    value: stamp.value,
                })),
        )?;
        *self = next;
        Ok(transaction)
    }

    /// Admit one fully resolved packet without exposing any partially updated
    /// owner when an envelope, continuation, dependency, or publication rule
    /// refuses it.
    pub fn admit_resolved<Operation, Lifecycle, Query, Present, Control>(
        &mut self,
        channel: ChannelId,
        prerequisites: impl Into<Box<[ResolvedTransactionPrerequisite]>>,
        completion_stamp: Option<crate::CompletionStamp>,
        payload: DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>,
    ) -> Result<
        DeviceTransaction<Operation, Lifecycle, Query, Present, Control>,
        TransactionRuntimeError,
    > {
        self.admit_resolved_with_signals(
            channel,
            prerequisites,
            completion_stamp,
            payload,
            std::iter::empty(),
        )
    }

    /// Atomically derive event/fence waits and signals from an immutable EXEC
    /// instead of accepting a second caller-authored spelling of them.
    ///
    /// Only packet-envelope stamp waits are accepted separately. Event and
    /// fence conditions belong to the operation stream and must pass through
    /// [`crate::ExecConditionPlan`].
    pub fn admit_exec_operations<Render, Compute, Info: Clone, Indirect, Effect: Clone>(
        &mut self,
        channel: ChannelId,
        stamp_prerequisites: impl Into<Box<[ResolvedTransactionPrerequisite]>>,
        completion_stamp: Option<crate::CompletionStamp>,
        exec: crate::ExecTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Effect>>,
    ) -> Result<
        AdmittedExecTransaction<Render, Compute, Info, Indirect, Effect>,
        TransactionRuntimeError,
    > {
        let mut prerequisites = stamp_prerequisites.into().into_vec();
        if let Some(unexpected) = prerequisites.iter().find(|resolved| {
            !matches!(resolved.prerequisite, TransactionPrerequisite::Stamp { .. })
        }) {
            return Err(TransactionRuntimeError::NonStampEnvelopePrerequisite(
                Box::new(unexpected.prerequisite),
            ));
        }
        let (operation_prerequisites, signals) = exec.condition_plan().into_parts();
        prerequisites.extend(operation_prerequisites.iter().copied().map(|prerequisite| {
            ResolvedTransactionPrerequisite {
                prerequisite,
                resolution: PrerequisiteResolution::Pending,
            }
        }));
        let transaction = self.admit_resolved_with_signals(
            channel,
            prerequisites.into_boxed_slice(),
            completion_stamp,
            DeviceTransactionPayload::Exec(exec),
            signals,
        )?;
        let DeviceTransactionPayload::Exec(exec) = &transaction.payload else {
            unreachable!("the admission payload was constructed as EXEC")
        };
        let conditions = crate::AdmittedExecConditions::from_exec(transaction.id, exec);
        let completion_effects = crate::AdmittedCompletionEffects::from_exec(transaction.id, exec);
        let resource_states = crate::AdmittedResourceStates::from_exec(transaction.id, exec);
        let info_queries = crate::AdmittedInfoQueries::from_exec(transaction.id, exec);
        Ok(AdmittedExecTransaction {
            transaction,
            conditions,
            completion_effects,
            resource_states,
            info_queries,
        })
    }

    fn admit_resolved_inner<Operation, Lifecycle, Query, Present, Control>(
        &mut self,
        channel: ChannelId,
        prerequisites: impl Into<Box<[ResolvedTransactionPrerequisite]>>,
        completion_stamp: Option<crate::CompletionStamp>,
        payload: DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>,
    ) -> Result<
        DeviceTransaction<Operation, Lifecycle, Query, Present, Control>,
        TransactionRuntimeError,
    > {
        if !self.channels.contains_key(&channel) {
            return Err(TransactionRuntimeError::UnknownChannel);
        }
        let prerequisites = prerequisites.into();
        if let Some(source_channel) = prerequisites.iter().find_map(|resolved| {
            let TransactionPrerequisite::Stamp { wait } = resolved.prerequisite else {
                return None;
            };
            let source_channel = ChannelId::new(wait.index);
            (!self.ingress.has_channel(source_channel)).then_some(source_channel)
        }) {
            return Err(TransactionRuntimeError::UnknownPrerequisiteChannel(
                source_channel,
            ));
        }
        let envelope_prerequisites = prerequisites
            .iter()
            .map(|resolved| resolved.prerequisite)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut ingress = self.ingress.clone();
        let mut continuations = self.continuations.clone();
        let mut recording = self.recording.clone();
        let mut submission_order = self.submission_order.clone();
        let mut dependencies = self.dependencies.clone();
        let mut publication = self.publication.clone();
        let mut conditions = self.conditions.clone();
        let transaction =
            ingress.admit(channel, envelope_prerequisites, completion_stamp, payload)?;

        let effective_prerequisites = prerequisites
            .iter()
            .copied()
            .map(|mut resolved| {
                if resolved.resolution == PrerequisiteResolution::Pending {
                    resolved.resolution =
                        match conditions.register_wait(transaction.id, resolved.cause())? {
                            ConditionWaitResolution::Satisfied => PrerequisiteResolution::Satisfied,
                            ConditionWaitResolution::Producer(producer) => {
                                PrerequisiteResolution::Producer(producer)
                            }
                            ConditionWaitResolution::Pending => PrerequisiteResolution::Pending,
                        };
                }
                Ok(resolved)
            })
            .collect::<Result<Vec<_>, ConditionOwnerError>>()?;
        let explicit_waits = effective_prerequisites
            .iter()
            .copied()
            .filter_map(ResolvedTransactionPrerequisite::explicit_wait)
            .collect::<Vec<_>>();
        let unresolved_waits = effective_prerequisites
            .iter()
            .copied()
            .filter_map(ResolvedTransactionPrerequisite::unresolved)
            .collect::<Vec<_>>();

        let (accesses, continuation_owned, recording_predecessor) = match &transaction.payload {
            DeviceTransactionPayload::Exec(exec) => {
                let continuation =
                    continuations.admit(transaction.id, transaction.submission_domain(), exec)?;
                let predecessor = continuation.map(|dependency| {
                    debug_assert_eq!(dependency.successor, transaction.id);
                    dependency.predecessor
                });
                (exec.accesses.clone(), true, predecessor)
            }
            _ => (Box::<[crate::AccessIntent]>::default(), false, None),
        };
        if continuation_owned {
            recording.accept(
                transaction.id,
                transaction.ingress_ordinal,
                recording_predecessor,
            )?;
            submission_order.accept(
                transaction.id,
                transaction.submission_domain(),
                transaction.domain_sequence(),
            )?;
        }
        let compiled_dependencies = dependencies.accept_with_unresolved(
            transaction.id,
            transaction.ingress_ordinal,
            accesses,
            &explicit_waits,
            &unresolved_waits,
        )?;
        publication.register(
            transaction.id,
            transaction.publication(),
            transaction.completion_stamp,
        )?;

        self.ingress = ingress;
        self.continuations = continuations;
        self.recording = recording;
        self.submission_order = submission_order;
        self.dependencies = dependencies;
        self.publication = publication;
        self.conditions = conditions;
        self.channels.get_mut(&channel).unwrap().admitted_once = true;
        self.channels.get_mut(&channel).unwrap().exec_admitted_once |= continuation_owned;
        self.transactions.insert(
            transaction.id,
            TransactionState {
                channel,
                ingress_ordinal: transaction.ingress_ordinal,
                continuation_owned,
                recording_predecessor,
                submitted: false,
                published: false,
            },
        );
        self.submission_hazards.insert(
            transaction.id,
            compiled_dependencies
                .hazards
                .requirements
                .into_boxed_slice(),
        );
        self.submission_dependencies
            .insert(transaction.id, compiled_dependencies.prerequisites);
        Ok(transaction)
    }

    pub fn recording_ready(&self) -> Vec<TransactionId> {
        self.recording.ready()
    }

    pub fn recording_plan(&self, transaction: TransactionId) -> Option<TransactionRecordingPlan> {
        let state = self.transactions.get(&transaction)?;
        state
            .continuation_owned
            .then_some(TransactionRecordingPlan {
                transaction,
                domain: SubmissionDomainId::for_fifo_channel(state.channel),
                continuation_predecessor: state.recording_predecessor,
            })
    }

    pub fn submission_dependencies(
        &self,
        transaction: TransactionId,
    ) -> Option<&[(TransactionId, WaitDependencyCause)]> {
        self.submission_dependencies
            .get(&transaction)
            .map(Box::as_ref)
    }

    pub fn submission_hazards(
        &self,
        transaction: TransactionId,
    ) -> Option<&[crate::HazardRequirement]> {
        self.submission_hazards.get(&transaction).map(Box::as_ref)
    }

    pub fn recorded(&mut self, transaction: TransactionId) -> Result<(), TransactionRuntimeError> {
        let mut recording = self.recording.clone();
        let mut submission_order = self.submission_order.clone();
        recording.recorded(transaction)?;
        submission_order.recorded(transaction)?;
        self.recording = recording;
        self.submission_order = submission_order;
        Ok(())
    }

    pub fn bind_prerequisite(
        &mut self,
        transaction: TransactionId,
        prerequisite: TransactionPrerequisite,
        producer: TransactionId,
    ) -> Result<(), TransactionRuntimeError> {
        let resolved = ResolvedTransactionPrerequisite {
            prerequisite,
            resolution: PrerequisiteResolution::Pending,
        };
        let cause = resolved.cause();
        let mut dependencies = self.dependencies.clone();
        let mut conditions = self.conditions.clone();
        conditions.bind_wait(transaction, cause, producer)?;
        dependencies.bind_unresolved_wait(transaction, producer, cause)?;
        let mut submission_dependencies = self.submission_dependencies.clone();
        let dependencies_for_transaction = submission_dependencies
            .get_mut(&transaction)
            .ok_or(TransactionRuntimeError::UnknownTransaction)?;
        let mut updated = dependencies_for_transaction.to_vec();
        updated.push((producer, WaitDependencyCause::Explicit(cause)));
        *dependencies_for_transaction = updated.into_boxed_slice();
        self.dependencies = dependencies;
        self.conditions = conditions;
        self.submission_dependencies = submission_dependencies;
        Ok(())
    }

    /// Register signal operations after their transaction identity and ingress
    /// ordinal are allocated. Any earlier wait-before-signal holds bind
    /// atomically to this producer and become native submission dependencies.
    fn register_signals(
        &mut self,
        transaction: TransactionId,
        signals: impl IntoIterator<Item = ConditionSignal>,
    ) -> Result<(), TransactionRuntimeError> {
        let ordinal = self
            .transactions
            .get(&transaction)
            .ok_or(TransactionRuntimeError::UnknownTransaction)?
            .ingress_ordinal;
        let mut conditions = self.conditions.clone();
        let mut dependencies = self.dependencies.clone();
        let mut submission_dependencies = self.submission_dependencies.clone();
        for signal in signals {
            for binding in conditions.register_signal(transaction, ordinal, signal)? {
                dependencies.bind_unresolved_wait(
                    binding.consumer,
                    binding.producer,
                    binding.cause,
                )?;
                let existing = submission_dependencies
                    .get_mut(&binding.consumer)
                    .ok_or(TransactionRuntimeError::UnknownTransaction)?;
                let mut updated = existing.to_vec();
                updated.push((
                    binding.producer,
                    WaitDependencyCause::Explicit(binding.cause),
                ));
                *existing = updated.into_boxed_slice();
            }
        }
        self.conditions = conditions;
        self.dependencies = dependencies;
        self.submission_dependencies = submission_dependencies;
        Ok(())
    }

    pub fn publish_gpu_signals(
        &mut self,
        transaction: TransactionId,
    ) -> Result<(), TransactionRuntimeError> {
        let mut conditions = self.conditions.clone();
        conditions
            .publish_transaction_at(transaction, ConditionPublicationBoundary::GpuCompletion)?;
        self.conditions = conditions;
        Ok(())
    }

    /// Validate and apply one timeline-proven GPU completion as a single
    /// semantic transition. Event/fence signals publish at this boundary;
    /// completion stamps remain ordered by guest publication.
    pub fn gpu_complete(
        &mut self,
        transaction: TransactionId,
        semantic: Completion,
    ) -> Result<Vec<PublishedFact<Completion>>, TransactionRuntimeError> {
        let mut next = self.clone();
        if next
            .conditions
            .has_signals_at(transaction, ConditionPublicationBoundary::GpuCompletion)
        {
            next.publish_gpu_signals(transaction)?;
        }
        let published = next.semantic_complete(transaction, semantic)?;
        *self = next;
        Ok(published)
    }

    pub fn validate_gpu_complete(
        &self,
        transaction: TransactionId,
        semantic: Completion,
    ) -> Result<(), TransactionRuntimeError> {
        let mut next = self.clone();
        next.gpu_complete(transaction, semantic)?;
        Ok(())
    }

    pub fn satisfy_prerequisite(
        &mut self,
        transaction: TransactionId,
        prerequisite: TransactionPrerequisite,
    ) -> Result<(), TransactionRuntimeError> {
        let cause = ResolvedTransactionPrerequisite {
            prerequisite,
            resolution: PrerequisiteResolution::Pending,
        }
        .cause();
        let mut dependencies = self.dependencies.clone();
        let mut conditions = self.conditions.clone();
        conditions.satisfy_wait(transaction, cause)?;
        dependencies.satisfy_unresolved_wait(transaction, cause)?;
        self.dependencies = dependencies;
        self.conditions = conditions;
        Ok(())
    }

    pub fn take_submission_ready(&mut self) -> Vec<SubmissionReady> {
        let dependencies = &self.dependencies;
        self.submission_order
            .take_ready_if(|transaction| !dependencies.has_unresolved(transaction))
    }

    /// Issue one exact recorded source-domain head without consuming ready
    /// heads from independent domains that their native recordings still own.
    pub fn take_submission_ready_transaction(
        &mut self,
        transaction: TransactionId,
    ) -> Result<Option<SubmissionReady>, TransactionRuntimeError> {
        let dependencies = &self.dependencies;
        self.submission_order
            .take_ready_transaction_if(transaction, |transaction| {
                !dependencies.has_unresolved(transaction)
            })
            .map_err(TransactionRuntimeError::Submission)
    }

    /// Reserve one exact source-domain head for a native execution chain
    /// whose final semantic recording is not complete yet.
    pub fn reserve_submission_head_transaction(
        &mut self,
        transaction: TransactionId,
    ) -> Result<Option<SubmissionReady>, TransactionRuntimeError> {
        let dependencies = &self.dependencies;
        self.submission_order
            .reserve_head_transaction_if(transaction, |transaction| {
                !dependencies.has_unresolved(transaction)
            })
            .map_err(TransactionRuntimeError::Submission)
    }

    /// Record successful native queue acceptance. Driver return is not GPU or
    /// semantic completion; it only releases the next logical FIFO head.
    pub fn validate_submitted(
        &self,
        transaction: TransactionId,
    ) -> Result<(), TransactionRuntimeError> {
        let mut submission_order = self.submission_order.clone();
        submission_order.submitted(transaction)?;
        if !self.transactions.contains_key(&transaction) {
            return Err(TransactionRuntimeError::UnknownTransaction);
        }
        Ok(())
    }

    /// Record successful native queue acceptance. Driver return is not GPU or
    /// semantic completion; it only releases the next logical FIFO head.
    pub fn submitted(&mut self, transaction: TransactionId) -> Result<(), TransactionRuntimeError> {
        self.validate_submitted(transaction)?;
        let mut submission_order = self.submission_order.clone();
        let mut transactions = self.transactions.clone();
        submission_order
            .submitted(transaction)
            .expect("submission acceptance was prevalidated");
        transactions
            .get_mut(&transaction)
            .expect("submission transaction was prevalidated")
            .submitted = true;
        self.submission_order = submission_order;
        self.transactions = transactions;
        Ok(())
    }

    pub fn semantic_ready(&self) -> Vec<ReadyTransaction> {
        self.dependencies.ready()
    }

    /// Commit one immutable semantic completion fact and publish every domain
    /// position newly released by that fact.
    pub fn semantic_complete(
        &mut self,
        transaction: TransactionId,
        semantic: Completion,
    ) -> Result<Vec<PublishedFact<Completion>>, TransactionRuntimeError> {
        if !self.transactions.contains_key(&transaction) {
            return Err(TransactionRuntimeError::UnknownTransaction);
        }
        if self.transactions[&transaction].continuation_owned {
            if !self.recording.is_recorded(transaction) {
                return Err(TransactionRuntimeError::Recording(
                    RecordingOrderError::NotRecorded,
                ));
            }
            if !self.transactions[&transaction].submitted {
                return Err(TransactionRuntimeError::TransactionNotSubmitted);
            }
        }
        let mut dependencies = self.dependencies.clone();
        let mut publication = self.publication.clone();
        let mut conditions = self.conditions.clone();
        let mut transactions = self.transactions.clone();
        dependencies.semantic_complete(transaction)?;
        publication.complete(PublicationFact {
            transaction,
            semantic,
        })?;
        let published = publication.publish_ready();
        for fact in &published {
            if fact.completion_stamp.is_some() {
                conditions.publish_transaction_at(
                    fact.transaction,
                    ConditionPublicationBoundary::GuestPublication,
                )?;
            }
            transactions
                .get_mut(&fact.transaction)
                .expect("publication contains only admitted transactions")
                .published = true;
        }
        self.dependencies = dependencies;
        self.publication = publication;
        self.conditions = conditions;
        self.transactions = transactions;
        Ok(published)
    }

    /// Retire a published transaction after the dependency graph proves that
    /// no accepted successor still names it.
    pub fn retire_transaction(
        &mut self,
        transaction: TransactionId,
    ) -> Result<(), TransactionRuntimeError> {
        self.validate_retire_transaction(transaction)?;
        let state = *self
            .transactions
            .get(&transaction)
            .expect("transaction retirement was prevalidated");
        let mut dependencies = self.dependencies.clone();
        let mut continuations = self.continuations.clone();
        let mut recording = self.recording.clone();
        let mut conditions = self.conditions.clone();
        dependencies.retire(transaction)?;
        conditions.retire_transaction(transaction)?;
        if state.continuation_owned {
            recording.retire(transaction)?;
            continuations.retire_transaction(transaction)?;
        }
        self.dependencies = dependencies;
        self.continuations = continuations;
        self.recording = recording;
        self.conditions = conditions;
        self.transactions.remove(&transaction);
        self.submission_dependencies.remove(&transaction);
        self.submission_hazards.remove(&transaction);
        Ok(())
    }

    pub fn validate_retire_transaction(
        &self,
        transaction: TransactionId,
    ) -> Result<(), TransactionRuntimeError> {
        let next = self.clone();
        let state = *next
            .transactions
            .get(&transaction)
            .ok_or(TransactionRuntimeError::UnknownTransaction)?;
        if !state.published {
            return Err(TransactionRuntimeError::TransactionNotPublished);
        }
        let mut dependencies = next.dependencies.clone();
        let mut continuations = next.continuations.clone();
        let mut recording = next.recording.clone();
        let mut conditions = next.conditions.clone();
        dependencies.retire(transaction)?;
        conditions.retire_transaction(transaction)?;
        if state.continuation_owned {
            recording.retire(transaction)?;
            continuations.retire_transaction(transaction)?;
        }
        Ok(())
    }

    /// End a channel lifetime only after every transaction, continuation, and
    /// publication obligation from that lifetime has retired.
    pub fn retire_channel(&mut self, channel: ChannelId) -> Result<(), TransactionRuntimeError> {
        let state = *self
            .channels
            .get(&channel)
            .ok_or(TransactionRuntimeError::UnknownChannel)?;
        if self
            .transactions
            .values()
            .any(|transaction| transaction.channel == channel)
        {
            return Err(TransactionRuntimeError::ChannelHasLiveTransactions);
        }

        let mut ingress = self.ingress.clone();
        let mut continuations = self.continuations.clone();
        let mut submission_order = self.submission_order.clone();
        let mut publication = self.publication.clone();
        let mut conditions = self.conditions.clone();
        continuations.retire_domain(SubmissionDomainId::for_fifo_channel(channel))?;
        if state.exec_admitted_once {
            submission_order.retire_domain(SubmissionDomainId::for_fifo_channel(channel))?;
        }
        if state.admitted_once {
            publication.retire_domain(PublicationDomainId::for_fifo_channel(channel))?;
        }
        ingress.retire_channel(channel)?;
        conditions.clear_stamp_channel(channel)?;
        self.ingress = ingress;
        self.continuations = continuations;
        self.submission_order = submission_order;
        self.publication = publication;
        self.conditions = conditions;
        self.channels.remove(&channel);
        Ok(())
    }

    pub fn live_transactions(&self) -> usize {
        self.transactions.len()
    }

    pub fn diagnostics(&self) -> Box<[TransactionRuntimeDiagnostic]> {
        let ready = self
            .semantic_ready()
            .into_iter()
            .map(ReadyTransaction::id)
            .collect::<std::collections::BTreeSet<_>>();
        self.transactions
            .iter()
            .map(|(transaction, state)| TransactionRuntimeDiagnostic {
                transaction: *transaction,
                channel: state.channel,
                unresolved: self.dependencies.unresolved(*transaction),
                semantic_ready: ready.contains(transaction),
                continuation_owned: state.continuation_owned,
                submitted: state.submitted,
                published: state.published,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompletionStamp, DirectReplayNativeOwner, ExecTransaction, RecordingAssignmentOwner,
        ResolvedExecSegment, ResolvedExecStream,
    };
    use reims_vgpu_protocol::{
        ChannelSequence, EventObject, QueueOwnerId, QueueTimelineValue, ResourceId,
        SegmentBoundary, SegmentKind, SessionGenerationId, StampWait, SubmissionId,
        SubmissionIdentity, TaskId, VulkanDeviceEpochId,
    };

    type Payload = DeviceTransactionPayload<(), (), (), (), ()>;

    fn runtime() -> TransactionRuntime<&'static str> {
        TransactionRuntime::new(SessionGeneration::new(SessionGenerationId::new(1)))
    }

    fn exec(kind: SegmentKind, previous: bool, next: bool) -> Payload {
        Payload::Exec(ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: vec![ResolvedExecStream {
                stream_index: 0,
                segments: vec![ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind,
                        continues_previous: previous,
                        continues_next: next,
                    },
                    operations: Box::new([]),
                }]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
            accesses: Box::new([]),
        })
    }

    fn admit_exec(
        runtime: &mut TransactionRuntime<&'static str>,
        channel: ChannelId,
        payload: Payload,
    ) -> DeviceTransaction<(), (), (), (), ()> {
        runtime
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                payload,
            )
            .unwrap()
    }

    #[test]
    fn successor_generation_preserves_session_wide_transaction_identity_space() {
        let mut first = runtime();
        let channel = ChannelId::new(4);
        first.define_channel(channel).unwrap();
        let old = first
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                None,
                DeviceTransactionPayload::<(), (), (), (), ()>::Control(()),
            )
            .unwrap();

        let mut next =
            first.successor_generation(SessionGeneration::new(SessionGenerationId::new(2)));
        next.define_channel(channel).unwrap();
        let new = next
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                None,
                DeviceTransactionPayload::<(), (), (), (), ()>::Control(()),
            )
            .unwrap();

        assert_eq!(old.id, TransactionId::new(1));
        assert_eq!(new.id, TransactionId::new(2));
        assert_eq!(
            new.ingress_ordinal,
            reims_vgpu_protocol::IngressOrdinal::new(2)
        );
        assert_eq!(new.channel_sequence, ChannelSequence::new(1));
    }

    #[test]
    fn failed_continuation_admission_consumes_no_position() {
        let mut runtime = runtime();
        let channel = ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let first = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Render, false, true),
        );
        assert_eq!(first.channel_sequence, ChannelSequence::new(1));
        assert!(matches!(
            runtime.admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(4, 2)),
                exec(SegmentKind::Compute, true, false),
            ),
            Err(TransactionRuntimeError::Continuation(
                ContinuationError::EncoderKindChanged
            ))
        ));
        let second = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Render, true, false),
        );
        let mut assignments = RecordingAssignmentOwner::new(2).unwrap();
        let first_plan = runtime.recording_plan(first.id).unwrap();
        let first_worker = assignments
            .assign(
                first_plan.transaction,
                first_plan.domain,
                first_plan.continuation_predecessor,
            )
            .unwrap();
        let second_plan = runtime.recording_plan(second.id).unwrap();
        assert_eq!(
            assignments
                .assign(
                    second_plan.transaction,
                    second_plan.domain,
                    second_plan.continuation_predecessor,
                )
                .unwrap(),
            first_worker
        );
        assert_eq!(second.id, TransactionId::new(2));
        assert_eq!(second.channel_sequence, ChannelSequence::new(2));
    }

    #[test]
    fn continuation_releases_on_recording_while_publication_remains_fifo_ordered() {
        let mut runtime = runtime();
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let first = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Render, false, true),
        );
        let second = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Render, true, false),
        );
        assert_eq!(runtime.recording_ready(), vec![first.id]);
        assert_eq!(
            runtime.semantic_ready().len(),
            2,
            "recording structure is not a semantic wait"
        );
        assert_eq!(
            runtime.semantic_complete(first.id, "too early"),
            Err(TransactionRuntimeError::Recording(
                RecordingOrderError::NotRecorded
            ))
        );
        runtime.recorded(first.id).unwrap();
        assert_eq!(runtime.recording_ready(), vec![second.id]);
        runtime.recorded(second.id).unwrap();
        assert_eq!(
            runtime
                .take_submission_ready()
                .into_iter()
                .map(|ready| ready.transaction)
                .collect::<Vec<_>>(),
            vec![first.id]
        );
        runtime.submitted(first.id).unwrap();
        assert_eq!(runtime.take_submission_ready()[0].transaction, second.id);
        runtime.submitted(second.id).unwrap();
        assert_eq!(
            runtime.semantic_complete(first.id, "first").unwrap()[0].transaction,
            first.id
        );
        assert_eq!(
            runtime.semantic_complete(second.id, "second").unwrap()[0].transaction,
            second.id
        );
        runtime.retire_transaction(first.id).unwrap();
        runtime.retire_transaction(second.id).unwrap();
        runtime.retire_channel(channel).unwrap();
    }

    #[test]
    fn independent_recording_finish_cannot_overtake_fifo_submission() {
        let mut runtime = runtime();
        let channel = ChannelId::new(5);
        runtime.define_channel(channel).unwrap();
        let first = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Render, false, false),
        );
        let second = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Compute, false, false),
        );
        runtime.recorded(second.id).unwrap();
        assert!(runtime.take_submission_ready().is_empty());
        runtime.recorded(first.id).unwrap();
        assert_eq!(
            runtime
                .take_submission_ready()
                .into_iter()
                .map(|ready| ready.transaction)
                .collect::<Vec<_>>(),
            vec![first.id]
        );
        runtime.submitted(first.id).unwrap();
        assert_eq!(runtime.take_submission_ready()[0].transaction, second.id);
    }

    #[test]
    fn unresolved_future_wait_holds_only_its_logical_domain_until_binding() {
        let mut runtime = runtime();
        let waiter_channel = ChannelId::new(7);
        let producer_channel = ChannelId::new(8);
        runtime.define_channel(waiter_channel).unwrap();
        runtime.define_channel(producer_channel).unwrap();
        let prerequisite = TransactionPrerequisite::Stamp {
            wait: StampWait {
                index: producer_channel.get(),
                value: 5,
            },
        };
        let waiter = runtime
            .admit_resolved(
                waiter_channel,
                [ResolvedTransactionPrerequisite {
                    prerequisite,
                    resolution: PrerequisiteResolution::Pending,
                }],
                Some(CompletionStamp::new(waiter_channel.get(), 1)),
                exec(SegmentKind::Compute, false, false),
            )
            .unwrap();
        runtime.recorded(waiter.id).unwrap();
        assert!(runtime.take_submission_ready().is_empty());
        let producer = runtime
            .admit_resolved(
                producer_channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(producer_channel.get(), 5)),
                exec(SegmentKind::Compute, false, false),
            )
            .unwrap();
        runtime.recorded(producer.id).unwrap();
        assert_eq!(
            runtime
                .take_submission_ready()
                .into_iter()
                .map(|ready| ready.transaction)
                .collect::<Vec<_>>(),
            vec![waiter.id, producer.id]
        );
        runtime.submitted(producer.id).unwrap();
        assert_eq!(
            runtime.submission_dependencies(waiter.id).unwrap(),
            [(
                producer.id,
                WaitDependencyCause::Explicit(ExplicitWaitCause::Stamp {
                    source_channel: producer_channel,
                    value: 5,
                })
            )]
        );
    }

    #[test]
    fn published_condition_outlives_its_producer_transaction() {
        let mut runtime = runtime();
        let producer_channel = ChannelId::new(9);
        let waiter_channel = ChannelId::new(10);
        runtime.define_channel(producer_channel).unwrap();
        runtime.define_channel(waiter_channel).unwrap();
        let event = ResourceId::<EventObject>::new(4, 2);
        let producer = runtime
            .admit_resolved_with_signals(
                producer_channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(producer_channel.get(), 1)),
                exec(SegmentKind::Event, false, false),
                [ConditionSignal::Event { event, value: 7 }],
            )
            .unwrap();
        runtime.recorded(producer.id).unwrap();
        runtime.take_submission_ready();
        runtime.submitted(producer.id).unwrap();
        runtime.semantic_complete(producer.id, "signal").unwrap();
        assert_eq!(
            runtime.retire_transaction(producer.id),
            Err(TransactionRuntimeError::Condition(
                ConditionOwnerError::SignalNotPublished
            ))
        );
        runtime.publish_gpu_signals(producer.id).unwrap();
        runtime.retire_transaction(producer.id).unwrap();

        let waiter = runtime
            .admit_resolved(
                waiter_channel,
                [
                    ResolvedTransactionPrerequisite {
                        prerequisite: TransactionPrerequisite::Event { event, value: 5 },
                        resolution: PrerequisiteResolution::Pending,
                    },
                    ResolvedTransactionPrerequisite {
                        prerequisite: TransactionPrerequisite::Stamp {
                            wait: StampWait {
                                index: producer_channel.get(),
                                value: 1,
                            },
                        },
                        resolution: PrerequisiteResolution::Pending,
                    },
                ],
                Some(CompletionStamp::new(waiter_channel.get(), 1)),
                exec(SegmentKind::Event, false, false),
            )
            .unwrap();
        runtime.recorded(waiter.id).unwrap();
        assert_eq!(runtime.take_submission_ready()[0].transaction, waiter.id);
        assert!(runtime
            .submission_dependencies(waiter.id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn gpu_completion_publishes_gpu_signals_and_semantics_atomically() {
        let mut runtime = runtime();
        let producer_channel = ChannelId::new(11);
        let waiter_channel = ChannelId::new(12);
        runtime.define_channel(producer_channel).unwrap();
        runtime.define_channel(waiter_channel).unwrap();
        let event = ResourceId::<EventObject>::new(8, 2);
        let producer = runtime
            .admit_resolved_with_signals(
                producer_channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(producer_channel.get(), 1)),
                exec(SegmentKind::Event, false, false),
                [ConditionSignal::Event { event, value: 7 }],
            )
            .unwrap();
        runtime.recorded(producer.id).unwrap();
        runtime.take_submission_ready();
        runtime.submitted(producer.id).unwrap();
        assert_eq!(
            runtime.gpu_complete(producer.id, "signal").unwrap()[0].transaction,
            producer.id
        );
        runtime.retire_transaction(producer.id).unwrap();

        let waiter = runtime
            .admit_resolved(
                waiter_channel,
                [ResolvedTransactionPrerequisite {
                    prerequisite: TransactionPrerequisite::Event { event, value: 7 },
                    resolution: PrerequisiteResolution::Pending,
                }],
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert!(runtime
            .submission_dependencies(waiter.id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn channel_reuse_starts_a_new_ordering_and_publication_lifetime() {
        let mut runtime = runtime();
        let channel = ChannelId::new(2);
        runtime.define_channel(channel).unwrap();
        let first = runtime
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(
            runtime.retire_channel(channel),
            Err(TransactionRuntimeError::ChannelHasLiveTransactions)
        );
        runtime.semantic_complete(first.id, "done").unwrap();
        runtime.retire_transaction(first.id).unwrap();
        runtime.retire_channel(channel).unwrap();

        runtime.define_channel(channel).unwrap();
        let reused = runtime
            .admit_resolved(
                channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(reused.channel_sequence, ChannelSequence::new(1));
    }

    #[test]
    fn channel_retirement_waits_for_cross_channel_stamp_ownership() {
        let mut runtime = runtime();
        let producer_channel = ChannelId::new(2);
        let waiter_channel = ChannelId::new(3);
        runtime.define_channel(producer_channel).unwrap();
        runtime.define_channel(waiter_channel).unwrap();

        let waiter = runtime
            .admit_resolved(
                waiter_channel,
                [ResolvedTransactionPrerequisite {
                    prerequisite: TransactionPrerequisite::Stamp {
                        wait: StampWait {
                            index: producer_channel.get(),
                            value: 7,
                        },
                    },
                    resolution: PrerequisiteResolution::Pending,
                }],
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(
            runtime.retire_channel(producer_channel),
            Err(TransactionRuntimeError::Condition(
                ConditionOwnerError::ConditionInUse
            ))
        );

        let producer = runtime
            .admit_resolved(
                producer_channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(producer_channel.get(), 7)),
                Payload::Control(()),
            )
            .unwrap();
        runtime.semantic_complete(producer.id, "producer").unwrap();
        assert_eq!(runtime.semantic_ready()[0].id(), waiter.id);
        runtime.semantic_complete(waiter.id, "waiter").unwrap();
        runtime.retire_transaction(waiter.id).unwrap();
        runtime.retire_transaction(producer.id).unwrap();
        runtime.retire_channel(producer_channel).unwrap();

        runtime.define_channel(producer_channel).unwrap();
        let reused = runtime
            .admit_resolved(
                producer_channel,
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(reused.channel_sequence, ChannelSequence::new(1));
    }

    #[test]
    fn only_pending_resolved_prerequisites_enter_the_wait_graph() {
        let mut runtime = runtime();
        runtime.define_channel(ChannelId::new(1)).unwrap();
        runtime.define_channel(ChannelId::new(2)).unwrap();
        let producer = runtime
            .admit_resolved(
                ChannelId::new(1),
                Box::<[ResolvedTransactionPrerequisite]>::default(),
                None,
                Payload::Control(()),
            )
            .unwrap();
        let waiter = runtime
            .admit_resolved(
                ChannelId::new(2),
                [ResolvedTransactionPrerequisite {
                    prerequisite: TransactionPrerequisite::Stamp {
                        wait: StampWait { index: 1, value: 7 },
                    },
                    resolution: PrerequisiteResolution::Producer(producer.id),
                }],
                None,
                Payload::Control(()),
            )
            .unwrap();
        assert_eq!(
            runtime.submission_dependencies(waiter.id).unwrap(),
            [(
                producer.id,
                WaitDependencyCause::Explicit(ExplicitWaitCause::Stamp {
                    source_channel: ChannelId::new(1),
                    value: 7,
                })
            )]
        );
        assert_eq!(runtime.semantic_ready()[0].id(), producer.id);
        runtime.semantic_complete(producer.id, "producer").unwrap();
        assert_eq!(runtime.semantic_ready()[0].id(), waiter.id);
    }

    #[test]
    fn direct_native_replay_publishes_only_timeline_completion_facts() {
        let mut runtime = runtime();
        let mut native = DirectReplayNativeOwner::new(VulkanDeviceEpochId::new(1), 2).unwrap();
        let channel = ChannelId::new(6);
        runtime.define_channel(channel).unwrap();
        let first = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Render, false, false),
        );
        let second = admit_exec(
            &mut runtime,
            channel,
            exec(SegmentKind::Compute, false, false),
        );
        native
            .assign_recording(runtime.recording_plan(first.id).unwrap())
            .unwrap();
        native
            .assign_recording(runtime.recording_plan(second.id).unwrap())
            .unwrap();
        runtime.recorded(second.id).unwrap();
        runtime.recorded(first.id).unwrap();

        for (transaction, semantic) in [(first.id, "first"), (second.id, "second")] {
            let ready = runtime.take_submission_ready();
            assert_eq!(ready.len(), 1);
            assert_eq!(ready[0].transaction, transaction);
            let plan = native
                .queue_candidate_with_hazards(
                    transaction,
                    runtime
                        .submission_dependencies(transaction)
                        .unwrap()
                        .to_vec(),
                    runtime.submission_hazards(transaction).unwrap().to_vec(),
                )
                .unwrap()
                .pop()
                .unwrap();
            let prepared = native
                .prepare(
                    plan,
                    QueueOwnerId::new(0),
                    SessionGenerationId::new(1),
                    semantic,
                )
                .unwrap();
            native.accepted(prepared).unwrap();
            runtime.submitted(transaction).unwrap();
        }
        assert_eq!(native.pending_completions(), 2);
        let facts = native
            .advance(QueueOwnerId::new(0), QueueTimelineValue::new(2))
            .unwrap();
        assert_eq!(facts.len(), 2);
        let mut published = Vec::new();
        for fact in facts {
            published.extend(
                runtime
                    .semantic_complete(fact.transaction, fact.semantic)
                    .unwrap(),
            );
        }
        assert_eq!(
            published
                .iter()
                .map(|fact| (fact.transaction, fact.semantic))
                .collect::<Vec<_>>(),
            vec![(first.id, "first"), (second.id, "second")]
        );
    }
}
