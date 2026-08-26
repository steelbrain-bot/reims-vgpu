//! Explicit future-prerequisite graph and cycle diagnostics.
//!
//! Resource hazards always point from newer ingress to older ingress and are
//! compiled by [`crate::DependencyCompiler`]. Event, fence, and stamp waits can
//! name a producer accepted later, so they live here instead and may form a
//! real wait cycle. This graph never blocks a host thread; readiness is a query
//! over immutable transaction identities and completion facts.

use reims_vgpu_protocol::{ChannelId, EventObject, FenceObject, ResourceId, TransactionId};
use std::collections::{BTreeMap, BTreeSet};

use crate::HazardCause;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ExplicitWaitCause {
    Stamp {
        source_channel: ChannelId,
        value: u32,
    },
    Event {
        event: ResourceId<EventObject>,
        value: u64,
    },
    Fence {
        fence: ResourceId<FenceObject>,
        generation: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WaitDependencyCause {
    ResourceHazard(HazardCause),
    Explicit(ExplicitWaitCause),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaitCycle {
    /// Closed path: the first and last identities are equal.
    pub path: Box<[TransactionId]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WaitGraphError {
    DuplicateTransaction,
    UnknownWaiter,
    DuplicateEdge,
    DuplicateUnresolvedWait,
    UnknownUnresolvedWait,
    Cycle(WaitCycle),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitGraphRetireError {
    UnknownTransaction,
    NotCompleted,
    StillRequired,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WaitEdge {
    producer: TransactionId,
    cause: WaitDependencyCause,
}

#[derive(Clone, Debug, Default)]
struct WaitNode {
    accepted: bool,
    completed: bool,
    prerequisites: BTreeSet<WaitEdge>,
    unresolved: BTreeSet<ExplicitWaitCause>,
    dependents: BTreeSet<TransactionId>,
}

#[derive(Clone, Debug, Default)]
pub struct WaitGraph {
    nodes: BTreeMap<TransactionId, WaitNode>,
}

impl WaitGraph {
    pub fn accept(&mut self, transaction: TransactionId) -> Result<(), WaitGraphError> {
        let node = self.nodes.entry(transaction).or_default();
        if node.accepted {
            return Err(WaitGraphError::DuplicateTransaction);
        }
        node.accepted = true;
        Ok(())
    }

    pub fn add_wait(
        &mut self,
        waiter: TransactionId,
        producer: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), WaitGraphError> {
        self.add_dependency(waiter, producer, WaitDependencyCause::Explicit(cause))
    }

    pub fn add_unresolved_wait(
        &mut self,
        waiter: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), WaitGraphError> {
        let node = self
            .nodes
            .get_mut(&waiter)
            .filter(|node| node.accepted)
            .ok_or(WaitGraphError::UnknownWaiter)?;
        if !node.unresolved.insert(cause) {
            return Err(WaitGraphError::DuplicateUnresolvedWait);
        }
        Ok(())
    }

    pub fn bind_unresolved_wait(
        &mut self,
        waiter: TransactionId,
        producer: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), WaitGraphError> {
        if !self
            .nodes
            .get(&waiter)
            .is_some_and(|node| node.unresolved.contains(&cause))
        {
            return Err(WaitGraphError::UnknownUnresolvedWait);
        }
        let mut next = self.clone();
        next.nodes
            .get_mut(&waiter)
            .unwrap()
            .unresolved
            .remove(&cause);
        next.add_wait(waiter, producer, cause)?;
        *self = next;
        Ok(())
    }

    pub fn satisfy_unresolved_wait(
        &mut self,
        waiter: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), WaitGraphError> {
        let node = self
            .nodes
            .get_mut(&waiter)
            .ok_or(WaitGraphError::UnknownWaiter)?;
        if !node.unresolved.remove(&cause) {
            return Err(WaitGraphError::UnknownUnresolvedWait);
        }
        Ok(())
    }

    pub fn add_hazard_dependency(
        &mut self,
        waiter: TransactionId,
        producer: TransactionId,
        cause: HazardCause,
    ) -> Result<(), WaitGraphError> {
        self.add_dependency(waiter, producer, WaitDependencyCause::ResourceHazard(cause))
    }

    fn add_dependency(
        &mut self,
        waiter: TransactionId,
        producer: TransactionId,
        cause: WaitDependencyCause,
    ) -> Result<(), WaitGraphError> {
        if !self.nodes.get(&waiter).is_some_and(|node| node.accepted) {
            return Err(WaitGraphError::UnknownWaiter);
        }
        let edge = WaitEdge { producer, cause };
        if self.nodes[&waiter].prerequisites.contains(&edge) {
            return Err(WaitGraphError::DuplicateEdge);
        }
        if let Some(mut path) = self.path(producer, waiter) {
            path.insert(0, waiter);
            return Err(WaitGraphError::Cycle(WaitCycle {
                path: path.into_boxed_slice(),
            }));
        }
        self.nodes
            .entry(producer)
            .or_default()
            .dependents
            .insert(waiter);
        self.nodes
            .get_mut(&waiter)
            .unwrap()
            .prerequisites
            .insert(edge);
        Ok(())
    }

    pub fn complete(&mut self, transaction: TransactionId) -> bool {
        if !self.is_ready(transaction) {
            return false;
        }
        let Some(node) = self.nodes.get_mut(&transaction) else {
            return false;
        };
        node.completed = true;
        true
    }

    pub fn is_ready(&self, transaction: TransactionId) -> bool {
        let Some(node) = self.nodes.get(&transaction) else {
            return false;
        };
        node.accepted
            && !node.completed
            && node.unresolved.is_empty()
            && node.prerequisites.iter().all(|edge| {
                self.nodes
                    .get(&edge.producer)
                    .is_some_and(|producer| producer.completed)
            })
    }

    pub fn ready(&self) -> Vec<TransactionId> {
        self.nodes
            .keys()
            .copied()
            .filter(|transaction| self.is_ready(*transaction))
            .collect()
    }

    pub fn contains(&self, transaction: TransactionId) -> bool {
        self.nodes.contains_key(&transaction)
    }

    pub fn is_accepted(&self, transaction: TransactionId) -> bool {
        self.nodes
            .get(&transaction)
            .is_some_and(|node| node.accepted)
    }

    pub fn is_completed(&self, transaction: TransactionId) -> bool {
        self.nodes
            .get(&transaction)
            .is_some_and(|node| node.completed)
    }

    pub fn has_unresolved(&self, transaction: TransactionId) -> bool {
        self.nodes
            .get(&transaction)
            .is_some_and(|node| !node.unresolved.is_empty())
    }

    pub fn unresolved(&self, transaction: TransactionId) -> Box<[ExplicitWaitCause]> {
        self.nodes
            .get(&transaction)
            .map(|node| node.unresolved.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.nodes.values().filter(|node| node.accepted).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn retire(&mut self, transaction: TransactionId) -> Result<(), WaitGraphRetireError> {
        let node = self
            .nodes
            .get(&transaction)
            .ok_or(WaitGraphRetireError::UnknownTransaction)?;
        if !node.completed {
            return Err(WaitGraphRetireError::NotCompleted);
        }
        if !node.dependents.is_empty() {
            return Err(WaitGraphRetireError::StillRequired);
        }
        let node = self.nodes.remove(&transaction).unwrap();
        for prerequisite in node.prerequisites {
            if let Some(producer) = self.nodes.get_mut(&prerequisite.producer) {
                producer.dependents.remove(&transaction);
            }
        }
        Ok(())
    }

    fn path(&self, start: TransactionId, target: TransactionId) -> Option<Vec<TransactionId>> {
        if start == target {
            return Some(vec![start]);
        }
        let mut stack = vec![(start, vec![start])];
        let mut visited = BTreeSet::new();
        while let Some((current, path)) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(node) = self.nodes.get(&current) else {
                continue;
            };
            for edge in &node.prerequisites {
                let mut next_path = path.clone();
                next_path.push(edge.producer);
                if edge.producer == target {
                    return Some(next_path);
                }
                stack.push((edge.producer, next_path));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: u64) -> ExplicitWaitCause {
        ExplicitWaitCause::Event {
            event: ResourceId::new(1, 1),
            value,
        }
    }

    #[test]
    fn a_wait_can_name_a_future_producer_without_blocking_the_owner() {
        let mut graph = WaitGraph::default();
        graph.accept(TransactionId::new(1)).unwrap();
        graph
            .add_wait(TransactionId::new(1), TransactionId::new(2), event(7))
            .unwrap();
        assert!(!graph.is_ready(TransactionId::new(1)));
        graph.accept(TransactionId::new(2)).unwrap();
        assert_eq!(graph.ready(), vec![TransactionId::new(2)]);
        assert!(graph.complete(TransactionId::new(2)));
        assert_eq!(graph.ready(), vec![TransactionId::new(1)]);
    }

    #[test]
    fn unresolved_wait_is_not_ready_until_it_binds_or_is_satisfied() {
        let mut graph = WaitGraph::default();
        let waiter = TransactionId::new(1);
        let producer = TransactionId::new(2);
        graph.accept(waiter).unwrap();
        graph.add_unresolved_wait(waiter, event(7)).unwrap();
        assert!(graph.has_unresolved(waiter));
        assert!(!graph.is_ready(waiter));
        graph
            .bind_unresolved_wait(waiter, producer, event(7))
            .unwrap();
        assert!(!graph.has_unresolved(waiter));
        graph.accept(producer).unwrap();
        assert_eq!(graph.ready(), vec![producer]);
        assert!(graph.complete(producer));
        assert!(graph.is_ready(waiter));

        let mut satisfied = WaitGraph::default();
        satisfied.accept(waiter).unwrap();
        satisfied.add_unresolved_wait(waiter, event(9)).unwrap();
        satisfied.satisfy_unresolved_wait(waiter, event(9)).unwrap();
        assert!(satisfied.is_ready(waiter));
    }

    #[test]
    fn failed_unresolved_binding_preserves_the_hold() {
        let mut graph = WaitGraph::default();
        let first = TransactionId::new(1);
        let second = TransactionId::new(2);
        graph.accept(first).unwrap();
        graph.add_unresolved_wait(first, event(1)).unwrap();
        graph.accept(second).unwrap();
        graph.add_wait(second, first, event(2)).unwrap();
        assert!(matches!(
            graph.bind_unresolved_wait(first, second, event(1)),
            Err(WaitGraphError::Cycle(_))
        ));
        assert!(graph.has_unresolved(first));
        assert!(!graph.is_ready(first));
    }

    #[test]
    fn a_future_wait_cycle_is_typed_and_the_rejected_edge_is_not_installed() {
        let mut graph = WaitGraph::default();
        for id in 1..=3 {
            graph.accept(TransactionId::new(id)).unwrap();
        }
        graph
            .add_wait(TransactionId::new(1), TransactionId::new(2), event(1))
            .unwrap();
        graph
            .add_wait(TransactionId::new(2), TransactionId::new(3), event(2))
            .unwrap();
        assert_eq!(
            graph.add_wait(TransactionId::new(3), TransactionId::new(1), event(3)),
            Err(WaitGraphError::Cycle(WaitCycle {
                path: vec![
                    TransactionId::new(3),
                    TransactionId::new(1),
                    TransactionId::new(2),
                    TransactionId::new(3),
                ]
                .into_boxed_slice(),
            }))
        );
        assert_eq!(graph.ready(), vec![TransactionId::new(3)]);
    }

    #[test]
    fn independent_wait_domains_do_not_head_of_line_block() {
        let mut graph = WaitGraph::default();
        for id in 1..=3 {
            graph.accept(TransactionId::new(id)).unwrap();
        }
        graph
            .add_wait(TransactionId::new(1), TransactionId::new(2), event(1))
            .unwrap();
        assert_eq!(
            graph.ready(),
            vec![TransactionId::new(2), TransactionId::new(3)]
        );
    }

    #[test]
    fn completed_producers_retire_only_after_their_dependents() {
        let mut graph = WaitGraph::default();
        graph.accept(TransactionId::new(1)).unwrap();
        graph.accept(TransactionId::new(2)).unwrap();
        graph
            .add_wait(TransactionId::new(2), TransactionId::new(1), event(1))
            .unwrap();
        graph.complete(TransactionId::new(1));
        assert_eq!(
            graph.retire(TransactionId::new(1)),
            Err(WaitGraphRetireError::StillRequired)
        );
        graph.complete(TransactionId::new(2));
        graph.retire(TransactionId::new(2)).unwrap();
        graph.retire(TransactionId::new(1)).unwrap();
    }

    #[test]
    fn equal_integer_event_and_fence_identities_do_not_alias() {
        let event = ExplicitWaitCause::Event {
            event: ResourceId::new(4, 1),
            value: 9,
        };
        let fence = ExplicitWaitCause::Fence {
            fence: ResourceId::new(4, 1),
            generation: 9,
        };
        assert_ne!(event, fence);
    }
}
