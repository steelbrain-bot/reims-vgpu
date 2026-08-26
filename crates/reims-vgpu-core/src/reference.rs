//! Serial backend-independent interpreter for resolved device transactions.
//!
//! This interpreter is the semantic oracle for model schedules. It consumes
//! only transactions proven ready by [`crate::DependencyCoordinator`], applies
//! one whole transaction atomically through [`ReferenceSemantics`], advances
//! the correct GPU/CPU/present lifecycle branch, and gives immutable completion
//! facts to [`crate::PublicationOwner`]. It performs no Vulkan work and cannot
//! manufacture a GPU phase for a CPU-only EXEC.

use crate::{
    AccessIntent, CoordinationError, DependencyCoordinator, DeviceTransaction,
    DeviceTransactionPayload, ExplicitWaitCause, PublicationError, PublicationFact,
    PublicationOwner, PublishedFact, TransactionState, TransactionStateMachine,
    TransactionTransitionError,
};
use reims_vgpu_protocol::TransactionId;
use std::collections::BTreeMap;

type Transaction<Operation, Lifecycle, Query, Present, Control> =
    DeviceTransaction<Operation, Lifecycle, Query, Present, Control>;
type Payload<Operation, Lifecycle, Query, Present, Control> =
    DeviceTransactionPayload<Operation, Lifecycle, Query, Present, Control>;

/// Planned completion branch. This is derived from transaction semantics, not
/// from packet class: an EXEC can legitimately be CPU-only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionPath {
    Gpu,
    Effect,
    Present,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceCompletion<T> {
    pub path: CompletionPath,
    pub semantic: T,
}

/// Atomic semantic implementation used by the serial oracle.
///
/// Returning `Err` must leave `self` unchanged. This is the same all-or-nothing
/// boundary required before a transaction's first externally visible effect;
/// backends that can only report a successful prefix do not implement this
/// trait until that prefix behavior is established by the API contract.
pub trait ReferenceSemantics<P> {
    type Completion;
    type Error;

    fn apply(
        &mut self,
        transaction: TransactionId,
        payload: &P,
    ) -> Result<Self::Completion, Self::Error>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceError<SemanticError> {
    DuplicateTransaction,
    Coordination(CoordinationError),
    Publication(PublicationError),
    Transition(TransactionTransitionError),
    Semantic(SemanticError),
    UnknownTransaction,
    NotPublished,
}

impl<SemanticError> From<CoordinationError> for ReferenceError<SemanticError> {
    fn from(error: CoordinationError) -> Self {
        Self::Coordination(error)
    }
}

impl<SemanticError> From<PublicationError> for ReferenceError<SemanticError> {
    fn from(error: PublicationError) -> Self {
        Self::Publication(error)
    }
}

impl<SemanticError> From<TransactionTransitionError> for ReferenceError<SemanticError> {
    fn from(error: TransactionTransitionError) -> Self {
        Self::Transition(error)
    }
}

#[derive(Clone, Debug)]
struct Entry<Operation, Lifecycle, Query, Present, Control> {
    transaction: Transaction<Operation, Lifecycle, Query, Present, Control>,
    path: CompletionPath,
    state: TransactionStateMachine,
}

#[derive(Clone, Debug)]
pub struct SerialReferenceInterpreter<Operation, Lifecycle, Query, Present, Control, Completion> {
    coordinator: DependencyCoordinator,
    publication: PublicationOwner<ReferenceCompletion<Completion>>,
    transactions: BTreeMap<TransactionId, Entry<Operation, Lifecycle, Query, Present, Control>>,
}

impl<Operation, Lifecycle, Query, Present, Control, Completion> Default
    for SerialReferenceInterpreter<Operation, Lifecycle, Query, Present, Control, Completion>
{
    fn default() -> Self {
        Self {
            coordinator: DependencyCoordinator::default(),
            publication: PublicationOwner::default(),
            transactions: BTreeMap::new(),
        }
    }
}

impl<Operation, Lifecycle, Query, Present, Control, Completion>
    SerialReferenceInterpreter<Operation, Lifecycle, Query, Present, Control, Completion>
where
    Completion: Clone,
{
    pub fn admit<SemanticError>(
        &mut self,
        transaction: Transaction<Operation, Lifecycle, Query, Present, Control>,
        accesses: impl Into<Box<[AccessIntent]>>,
        explicit_waits: &[(TransactionId, ExplicitWaitCause)],
        path: CompletionPath,
    ) -> Result<(), ReferenceError<SemanticError>> {
        if self.transactions.contains_key(&transaction.id) {
            return Err(ReferenceError::DuplicateTransaction);
        }

        let mut coordinator = self.coordinator.clone();
        coordinator.accept(
            transaction.id,
            transaction.ingress_ordinal,
            accesses,
            explicit_waits,
        )?;
        let mut publication = self.publication.clone();
        publication.register(
            transaction.id,
            transaction.publication(),
            transaction.completion_stamp,
        )?;

        let mut state = TransactionStateMachine::new();
        state.advance(TransactionState::Decoding)?;
        state.advance(TransactionState::Resolved)?;
        state.advance(TransactionState::Planned)?;

        self.coordinator = coordinator;
        self.publication = publication;
        self.transactions.insert(
            transaction.id,
            Entry {
                transaction,
                path,
                state,
            },
        );
        Ok(())
    }

    /// Run ready transactions until no semantic prerequisite can advance, then
    /// publish every independently unblocked completion domain.
    pub fn run_ready<S>(
        &mut self,
        semantics: &mut S,
    ) -> Result<Vec<PublishedFact<ReferenceCompletion<Completion>>>, ReferenceError<S::Error>>
    where
        S: ReferenceSemantics<
            Payload<Operation, Lifecycle, Query, Present, Control>,
            Completion = Completion,
        >,
    {
        loop {
            let ready = self.coordinator.ready();
            let Some(ready) = ready.first().copied() else {
                break;
            };
            let id = ready.id();
            let entry = self
                .transactions
                .get_mut(&id)
                .ok_or(ReferenceError::UnknownTransaction)?;

            // The semantic adapter is all-or-nothing. Only after it returns a
            // complete fact do structural execution phases advance.
            let semantic = semantics
                .apply(id, &entry.transaction.payload)
                .map_err(ReferenceError::Semantic)?;
            advance_completion_path(&mut entry.state, entry.path)?;
            self.coordinator.semantic_complete(id)?;
            self.publication.complete(PublicationFact {
                transaction: id,
                semantic: ReferenceCompletion {
                    path: entry.path,
                    semantic,
                },
            })?;
        }

        let published = self.publication.publish_ready();
        for fact in &published {
            self.transactions
                .get_mut(&fact.transaction)
                .ok_or(ReferenceError::UnknownTransaction)?
                .state
                .advance(TransactionState::GuestPublished)?;
        }
        Ok(published)
    }

    pub fn state(&self, transaction: TransactionId) -> Option<TransactionState> {
        self.transactions
            .get(&transaction)
            .map(|entry| entry.state.state())
    }

    pub fn retire_published<SemanticError>(
        &mut self,
        transaction: TransactionId,
    ) -> Result<(), ReferenceError<SemanticError>> {
        let entry = self
            .transactions
            .get(&transaction)
            .ok_or(ReferenceError::UnknownTransaction)?;
        if entry.state.state() != TransactionState::GuestPublished {
            return Err(ReferenceError::NotPublished);
        }
        self.coordinator.retire(transaction)?;
        let mut entry = self.transactions.remove(&transaction).unwrap();
        entry.state.advance(TransactionState::Retired)?;
        Ok(())
    }

    pub fn pending(&self) -> usize {
        self.transactions.len()
    }
}

fn advance_completion_path(
    state: &mut TransactionStateMachine,
    path: CompletionPath,
) -> Result<(), TransactionTransitionError> {
    let phases: &[TransactionState] = match path {
        CompletionPath::Gpu => &[
            TransactionState::Recording,
            TransactionState::Recorded,
            TransactionState::Queued,
            TransactionState::Submitted,
            TransactionState::GpuComplete,
            TransactionState::SemanticCommitted,
        ],
        CompletionPath::Effect => &[
            TransactionState::Executing,
            TransactionState::EffectComplete,
            TransactionState::SemanticCommitted,
        ],
        CompletionPath::Present => &[
            TransactionState::AcquirePending,
            TransactionState::PresentReady,
            TransactionState::PresentQueued,
            TransactionState::PresentComplete,
            TransactionState::SemanticCommitted,
        ],
    };
    for phase in phases {
        state.advance(*phase)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionStamp, DeviceTransactionPayload, SessionGeneration};
    use reims_vgpu_protocol::{
        ChannelId, ChannelSequence, EventObject, IngressOrdinal, PublicationDomainId, ResourceId,
        SessionGenerationId,
    };

    type Interpreter = SerialReferenceInterpreter<
        (),
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    >;
    type TestPayload =
        DeviceTransactionPayload<(), &'static str, &'static str, &'static str, &'static str>;

    #[derive(Default)]
    struct Semantics {
        applied: Vec<TransactionId>,
        fail: Option<TransactionId>,
    }

    impl ReferenceSemantics<TestPayload> for Semantics {
        type Completion = &'static str;
        type Error = &'static str;

        fn apply(
            &mut self,
            transaction: TransactionId,
            payload: &TestPayload,
        ) -> Result<Self::Completion, Self::Error> {
            if self.fail == Some(transaction) {
                return Err("refused atomically");
            }
            self.applied.push(transaction);
            Ok(match payload {
                TestPayload::Exec(_) => "exec",
                TestPayload::ResourceLifecycle(_) => "lifecycle",
                TestPayload::Query(_) => "query",
                TestPayload::Present(_) => "present",
                TestPayload::Control(_) => "control",
            })
        }
    }

    fn transaction(
        id: u64,
        domain: u64,
        sequence: u64,
        payload: TestPayload,
    ) -> Transaction<(), &'static str, &'static str, &'static str, &'static str> {
        DeviceTransaction {
            id: TransactionId::new(id),
            session_generation: SessionGeneration::new(SessionGenerationId::new(1))
                .try_lease()
                .unwrap(),
            channel: ChannelId::new(domain as u32),
            channel_sequence: ChannelSequence::new(sequence),
            ingress_ordinal: IngressOrdinal::new(id),
            prerequisites: Box::new([]),
            completion_stamp: Some(CompletionStamp::new(2, id as u32)),
            payload,
        }
    }

    fn event(value: u64) -> ExplicitWaitCause {
        ExplicitWaitCause::Event {
            event: ResourceId::<EventObject>::new(1, 1),
            value,
        }
    }

    #[test]
    fn cpu_only_exec_uses_the_effect_path_and_publishes_without_gpu_semantics() {
        let mut interpreter = Interpreter::default();
        interpreter
            .admit::<&'static str>(
                transaction(1, 1, 1, TestPayload::Control("noop")),
                Box::<[AccessIntent]>::default(),
                &[],
                CompletionPath::Effect,
            )
            .unwrap();
        let published = interpreter.run_ready(&mut Semantics::default()).unwrap();
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].semantic.path, CompletionPath::Effect);
        assert_eq!(published[0].semantic.semantic, "control");
        assert_eq!(
            interpreter.state(TransactionId::new(1)),
            Some(TransactionState::GuestPublished)
        );
    }

    #[test]
    fn blocked_domain_does_not_prevent_independent_effect_publication() {
        let mut interpreter = Interpreter::default();
        interpreter
            .admit::<&'static str>(
                transaction(1, 1, 1, TestPayload::Control("blocked")),
                Box::<[AccessIntent]>::default(),
                &[(TransactionId::new(3), event(1))],
                CompletionPath::Effect,
            )
            .unwrap();
        interpreter
            .admit::<&'static str>(
                transaction(2, 2, 1, TestPayload::Query("ready")),
                Box::<[AccessIntent]>::default(),
                &[],
                CompletionPath::Effect,
            )
            .unwrap();
        let mut semantics = Semantics::default();
        let published = interpreter.run_ready(&mut semantics).unwrap();
        assert_eq!(semantics.applied, vec![TransactionId::new(2)]);
        assert_eq!(published[0].transaction, TransactionId::new(2));
        assert_eq!(
            interpreter.state(TransactionId::new(1)),
            Some(TransactionState::Planned)
        );
    }

    #[test]
    fn same_publication_domain_releases_completed_facts_in_sequence() {
        let mut interpreter = Interpreter::default();
        interpreter
            .admit::<&'static str>(
                transaction(1, 1, 1, TestPayload::Control("blocked")),
                Box::<[AccessIntent]>::default(),
                &[(TransactionId::new(3), event(1))],
                CompletionPath::Effect,
            )
            .unwrap();
        interpreter
            .admit::<&'static str>(
                transaction(2, 1, 2, TestPayload::Control("second")),
                Box::<[AccessIntent]>::default(),
                &[],
                CompletionPath::Effect,
            )
            .unwrap();
        interpreter
            .admit::<&'static str>(
                transaction(3, 2, 1, TestPayload::Control("signal")),
                Box::<[AccessIntent]>::default(),
                &[],
                CompletionPath::Effect,
            )
            .unwrap();
        let published = interpreter.run_ready(&mut Semantics::default()).unwrap();
        let domain_one: Vec<_> = published
            .iter()
            .filter(|fact| fact.position.domain == PublicationDomainId::new(1))
            .map(|fact| fact.transaction)
            .collect();
        assert_eq!(
            domain_one,
            vec![TransactionId::new(1), TransactionId::new(2)]
        );
    }

    #[test]
    fn semantic_failure_does_not_complete_or_publish_a_prefix() {
        let mut interpreter = Interpreter::default();
        interpreter
            .admit::<&'static str>(
                transaction(1, 1, 1, TestPayload::ResourceLifecycle("create")),
                Box::<[AccessIntent]>::default(),
                &[],
                CompletionPath::Effect,
            )
            .unwrap();
        let mut semantics = Semantics {
            fail: Some(TransactionId::new(1)),
            ..Semantics::default()
        };
        assert_eq!(
            interpreter.run_ready(&mut semantics),
            Err(ReferenceError::Semantic("refused atomically"))
        );
        assert_eq!(
            interpreter.state(TransactionId::new(1)),
            Some(TransactionState::Planned)
        );
    }

    #[test]
    fn retirement_waits_for_dependents_even_after_guest_publication() {
        let mut interpreter = Interpreter::default();
        interpreter
            .admit::<&'static str>(
                transaction(1, 1, 1, TestPayload::Control("producer")),
                Box::<[AccessIntent]>::default(),
                &[],
                CompletionPath::Effect,
            )
            .unwrap();
        interpreter
            .admit::<&'static str>(
                transaction(2, 2, 1, TestPayload::Control("consumer")),
                Box::<[AccessIntent]>::default(),
                &[(TransactionId::new(1), event(1))],
                CompletionPath::Effect,
            )
            .unwrap();
        interpreter.run_ready(&mut Semantics::default()).unwrap();
        assert!(matches!(
            interpreter.retire_published::<&'static str>(TransactionId::new(1)),
            Err(ReferenceError::Coordination(CoordinationError::Retire(_)))
        ));
        interpreter
            .retire_published::<&'static str>(TransactionId::new(2))
            .unwrap();
        interpreter
            .retire_published::<&'static str>(TransactionId::new(1))
            .unwrap();
        assert_eq!(interpreter.pending(), 0);
    }
}
