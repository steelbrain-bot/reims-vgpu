//! Recording-only prerequisites that never advance semantic completion.
//!
//! A directional encoder continuation requires the predecessor's recorded
//! encoder state, not its GPU completion. This owner therefore has its own
//! readiness and retirement lifecycle instead of placing structural edges in
//! the semantic wait graph.

use reims_vgpu_protocol::{IngressOrdinal, TransactionId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordingOrderError {
    DuplicateTransaction,
    UnknownTransaction,
    UnknownPredecessor,
    PredecessorDidNotPrecede,
    NotReady,
    AlreadyRecorded,
    NotRecorded,
}

#[derive(Clone, Debug)]
struct RecordingNode {
    ordinal: IngressOrdinal,
    blocked: bool,
    recorded: bool,
    dependents: BTreeSet<TransactionId>,
}

#[derive(Clone, Debug, Default)]
pub struct RecordingOrderOwner {
    nodes: BTreeMap<TransactionId, RecordingNode>,
}

impl RecordingOrderOwner {
    pub fn accept(
        &mut self,
        transaction: TransactionId,
        ordinal: IngressOrdinal,
        predecessor: Option<TransactionId>,
    ) -> Result<(), RecordingOrderError> {
        if self.nodes.contains_key(&transaction) {
            return Err(RecordingOrderError::DuplicateTransaction);
        }
        let blocked = if let Some(predecessor) = predecessor {
            let predecessor_node = self
                .nodes
                .get(&predecessor)
                .ok_or(RecordingOrderError::UnknownPredecessor)?;
            if predecessor_node.ordinal >= ordinal {
                return Err(RecordingOrderError::PredecessorDidNotPrecede);
            }
            !predecessor_node.recorded
        } else {
            false
        };
        self.nodes.insert(
            transaction,
            RecordingNode {
                ordinal,
                blocked,
                recorded: false,
                dependents: BTreeSet::new(),
            },
        );
        if blocked {
            self.nodes
                .get_mut(&predecessor.unwrap())
                .expect("predecessor was validated")
                .dependents
                .insert(transaction);
        }
        Ok(())
    }

    pub fn ready(&self) -> Vec<TransactionId> {
        self.nodes
            .iter()
            .filter_map(|(&transaction, node)| {
                (!node.blocked && !node.recorded).then_some(transaction)
            })
            .collect()
    }

    pub fn is_recorded(&self, transaction: TransactionId) -> bool {
        self.nodes
            .get(&transaction)
            .is_some_and(|node| node.recorded)
    }

    pub fn recorded(&mut self, transaction: TransactionId) -> Result<(), RecordingOrderError> {
        let node = self
            .nodes
            .get(&transaction)
            .ok_or(RecordingOrderError::UnknownTransaction)?;
        if node.recorded {
            return Err(RecordingOrderError::AlreadyRecorded);
        }
        if node.blocked {
            return Err(RecordingOrderError::NotReady);
        }
        let dependents = self
            .nodes
            .get_mut(&transaction)
            .unwrap()
            .dependents
            .iter()
            .copied()
            .collect::<Vec<_>>();
        self.nodes.get_mut(&transaction).unwrap().recorded = true;
        self.nodes.get_mut(&transaction).unwrap().dependents.clear();
        for dependent in dependents {
            self.nodes
                .get_mut(&dependent)
                .expect("recording dependent remains admitted")
                .blocked = false;
        }
        Ok(())
    }

    /// Release a transaction's recording claim without it having recorded.
    ///
    /// [`Self::recorded`] is the successful end of a recording obligation and
    /// requires the encoder continuation ahead of it to have recorded first. A
    /// transaction refused before it records never satisfies that, and every
    /// continuation successor stays blocked behind it for the life of the
    /// device. This is the terminal transition for that case: the successors
    /// are released, exactly as a real recording releases them, because the
    /// encoder state they were waiting to continue is one that will never
    /// exist.
    ///
    /// Abandoning an already-recorded transaction is not an error. A refusal
    /// can land after recording finished -- at queue admission or at acceptance
    /// -- and the claim it must release is the same one.
    pub fn abandon(&mut self, transaction: TransactionId) -> Result<(), RecordingOrderError> {
        let node = self
            .nodes
            .get(&transaction)
            .ok_or(RecordingOrderError::UnknownTransaction)?;
        if node.recorded {
            return Ok(());
        }
        let dependents = self
            .nodes
            .get_mut(&transaction)
            .unwrap()
            .dependents
            .iter()
            .copied()
            .collect::<Vec<_>>();
        self.nodes.get_mut(&transaction).unwrap().recorded = true;
        self.nodes.get_mut(&transaction).unwrap().dependents.clear();
        for dependent in dependents {
            self.nodes
                .get_mut(&dependent)
                .expect("recording dependent remains admitted")
                .blocked = false;
        }
        Ok(())
    }

    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), RecordingOrderError> {
        let node = self
            .nodes
            .get(&transaction)
            .ok_or(RecordingOrderError::UnknownTransaction)?;
        if !node.recorded {
            return Err(RecordingOrderError::NotRecorded);
        }
        debug_assert!(node.dependents.is_empty());
        self.nodes.remove(&transaction);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_releases_on_recording_not_semantic_completion() {
        let mut owner = RecordingOrderOwner::default();
        owner
            .accept(TransactionId::new(1), IngressOrdinal::new(1), None)
            .unwrap();
        owner
            .accept(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                Some(TransactionId::new(1)),
            )
            .unwrap();
        assert_eq!(owner.ready(), vec![TransactionId::new(1)]);
        owner.recorded(TransactionId::new(1)).unwrap();
        assert_eq!(owner.ready(), vec![TransactionId::new(2)]);
    }

    #[test]
    fn independent_recording_is_never_held_by_a_structural_edge_elsewhere() {
        let mut owner = RecordingOrderOwner::default();
        owner
            .accept(TransactionId::new(1), IngressOrdinal::new(1), None)
            .unwrap();
        owner
            .accept(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                Some(TransactionId::new(1)),
            )
            .unwrap();
        owner
            .accept(TransactionId::new(3), IngressOrdinal::new(3), None)
            .unwrap();
        assert_eq!(
            owner.ready(),
            vec![TransactionId::new(1), TransactionId::new(3)]
        );
    }

    #[test]
    fn retirement_requires_recording_but_not_dependent_completion() {
        let mut owner = RecordingOrderOwner::default();
        owner
            .accept(TransactionId::new(1), IngressOrdinal::new(1), None)
            .unwrap();
        owner
            .accept(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                Some(TransactionId::new(1)),
            )
            .unwrap();
        assert_eq!(
            owner.retire(TransactionId::new(1)),
            Err(RecordingOrderError::NotRecorded)
        );
        owner.recorded(TransactionId::new(1)).unwrap();
        owner.retire(TransactionId::new(1)).unwrap();
        assert_eq!(owner.ready(), vec![TransactionId::new(2)]);
    }
}
