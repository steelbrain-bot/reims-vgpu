//! Atomic dependency compilation and nonblocking readiness coordination.
//!
//! Resource hazards and explicit future waits retain distinct causes but meet
//! in one wait-for graph, because a real cycle can contain one edge of each.
//! Vulkan-only host-safety constraints do not enter this coordinator and
//! therefore cannot advance or order semantic completion.

use crate::{
    AccessIntent, DependencyCompileError, DependencyCompiler, ExplicitWaitCause, HazardCompilation,
    WaitDependencyCause, WaitGraph, WaitGraphError, WaitGraphRetireError,
};
use reims_vgpu_protocol::{IngressOrdinal, TransactionId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoordinationError {
    Dependency(DependencyCompileError),
    Wait(WaitGraphError),
    UnknownTransaction,
    AlreadyCompleted,
    NotReady,
    Retire(WaitGraphRetireError),
}

impl From<DependencyCompileError> for CoordinationError {
    fn from(error: DependencyCompileError) -> Self {
        Self::Dependency(error)
    }
}

impl From<WaitGraphError> for CoordinationError {
    fn from(error: WaitGraphError) -> Self {
        Self::Wait(error)
    }
}

/// Proof that every currently known semantic prerequisite is complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadyTransaction {
    id: TransactionId,
}

impl ReadyTransaction {
    pub const fn id(self) -> TransactionId {
        self.id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransactionDependencies {
    pub hazards: HazardCompilation,
    pub explicit_waits: usize,
    pub prerequisites: Box<[(TransactionId, WaitDependencyCause)]>,
}

#[derive(Clone, Debug, Default)]
pub struct DependencyCoordinator {
    hazards: DependencyCompiler,
    waits: WaitGraph,
}

impl DependencyCoordinator {
    /// Atomically admit one transaction and all currently established semantic
    /// dependency edges. A cycle or malformed access leaves both owners
    /// unchanged.
    pub fn accept(
        &mut self,
        transaction: TransactionId,
        ordinal: IngressOrdinal,
        intents: impl Into<Box<[AccessIntent]>>,
        explicit_waits: &[(TransactionId, ExplicitWaitCause)],
    ) -> Result<TransactionDependencies, CoordinationError> {
        self.accept_with_unresolved(transaction, ordinal, intents, explicit_waits, &[])
    }

    pub fn accept_with_unresolved(
        &mut self,
        transaction: TransactionId,
        ordinal: IngressOrdinal,
        intents: impl Into<Box<[AccessIntent]>>,
        explicit_waits: &[(TransactionId, ExplicitWaitCause)],
        unresolved_waits: &[ExplicitWaitCause],
    ) -> Result<TransactionDependencies, CoordinationError> {
        let mut hazards = self.hazards.clone();
        let mut waits = self.waits.clone();
        let compilation = hazards.compile(transaction, ordinal, intents)?;
        waits.accept(transaction)?;
        for edge in &compilation.edges {
            waits.add_hazard_dependency(edge.newer, edge.older, edge.cause)?;
        }
        for &(producer, cause) in explicit_waits {
            waits.add_wait(transaction, producer, cause)?;
        }
        for &cause in unresolved_waits {
            waits.add_unresolved_wait(transaction, cause)?;
        }
        self.hazards = hazards;
        self.waits = waits;
        let prerequisites = compilation
            .edges
            .iter()
            .map(|edge| (edge.older, WaitDependencyCause::ResourceHazard(edge.cause)))
            .chain(
                explicit_waits
                    .iter()
                    .map(|&(producer, cause)| (producer, WaitDependencyCause::Explicit(cause))),
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(TransactionDependencies {
            hazards: compilation,
            explicit_waits: explicit_waits.len(),
            prerequisites,
        })
    }

    pub fn bind_unresolved_wait(
        &mut self,
        transaction: TransactionId,
        producer: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), CoordinationError> {
        let mut waits = self.waits.clone();
        waits.bind_unresolved_wait(transaction, producer, cause)?;
        self.waits = waits;
        Ok(())
    }

    pub fn satisfy_unresolved_wait(
        &mut self,
        transaction: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), CoordinationError> {
        let mut waits = self.waits.clone();
        waits.satisfy_unresolved_wait(transaction, cause)?;
        self.waits = waits;
        Ok(())
    }

    pub fn has_unresolved(&self, transaction: TransactionId) -> bool {
        self.waits.has_unresolved(transaction)
    }

    pub fn unresolved(&self, transaction: TransactionId) -> Box<[ExplicitWaitCause]> {
        self.waits.unresolved(transaction)
    }

    pub fn ready(&self) -> Vec<ReadyTransaction> {
        self.waits
            .ready()
            .into_iter()
            .map(|id| ReadyTransaction { id })
            .collect()
    }

    /// Semantic completion satisfies dependents and retires this transaction's
    /// live hazard records. Native lifetime retirement remains separate.
    pub fn semantic_complete(
        &mut self,
        transaction: TransactionId,
    ) -> Result<(), CoordinationError> {
        if !self.waits.is_accepted(transaction) {
            return Err(CoordinationError::UnknownTransaction);
        }
        if self.waits.is_completed(transaction) {
            return Err(CoordinationError::AlreadyCompleted);
        }
        if !self.waits.is_ready(transaction) {
            return Err(CoordinationError::NotReady);
        }
        debug_assert!(self.waits.complete(transaction));
        debug_assert!(self.hazards.retire(transaction));
        Ok(())
    }

    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), CoordinationError> {
        self.waits
            .retire(transaction)
            .map_err(CoordinationError::Retire)
    }

    pub fn live_transactions(&self) -> usize {
        self.waits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccessMode, AccessScope, LinearRange, StageScope};
    use reims_vgpu_protocol::{BackingId, EventObject, HazardDomainId, ResourceId, ResourceObject};

    fn access(mode: AccessMode) -> AccessIntent {
        AccessIntent::for_backing(
            HazardDomainId::new(1),
            BackingId::new(1),
            Some(ResourceId::<ResourceObject>::new(1, 1)),
            AccessScope::Linear(LinearRange::new(0, 64).unwrap()),
            mode,
            StageScope::Compute,
        )
        .unwrap()
    }

    fn event(value: u64) -> ExplicitWaitCause {
        ExplicitWaitCause::Event {
            event: ResourceId::<EventObject>::new(1, 1),
            value,
        }
    }

    #[test]
    fn disjoint_ready_work_is_not_held_behind_a_waiter() {
        let mut coordinator = DependencyCoordinator::default();
        coordinator
            .accept(
                TransactionId::new(1),
                IngressOrdinal::new(1),
                Box::<[AccessIntent]>::default(),
                &[(TransactionId::new(3), event(1))],
            )
            .unwrap();
        coordinator
            .accept(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                Box::<[AccessIntent]>::default(),
                &[],
            )
            .unwrap();
        assert_eq!(
            coordinator
                .ready()
                .into_iter()
                .map(ReadyTransaction::id)
                .collect::<Vec<_>>(),
            vec![TransactionId::new(2)]
        );
    }

    #[test]
    fn an_explicit_future_wait_and_resource_hazard_form_one_diagnosable_cycle() {
        let mut coordinator = DependencyCoordinator::default();
        coordinator
            .accept(
                TransactionId::new(1),
                IngressOrdinal::new(1),
                [access(AccessMode::Write)],
                &[(TransactionId::new(2), event(1))],
            )
            .unwrap();
        let before = coordinator.clone();
        let error = coordinator
            .accept(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                [access(AccessMode::Read)],
                &[],
            )
            .unwrap_err();
        assert!(matches!(
            error,
            CoordinationError::Wait(WaitGraphError::Cycle(_))
        ));
        assert_eq!(coordinator.live_transactions(), before.live_transactions());
        assert!(coordinator.ready().is_empty());
    }

    #[test]
    fn semantic_completion_releases_only_exact_dependents() {
        let mut coordinator = DependencyCoordinator::default();
        coordinator
            .accept(
                TransactionId::new(1),
                IngressOrdinal::new(1),
                [access(AccessMode::Write)],
                &[],
            )
            .unwrap();
        coordinator
            .accept(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                [access(AccessMode::Read)],
                &[],
            )
            .unwrap();
        assert_eq!(coordinator.ready()[0].id(), TransactionId::new(1));
        assert_eq!(
            coordinator.semantic_complete(TransactionId::new(2)),
            Err(CoordinationError::NotReady)
        );
        coordinator
            .semantic_complete(TransactionId::new(1))
            .unwrap();
        assert_eq!(coordinator.ready()[0].id(), TransactionId::new(2));
    }
}
