//! Cutover ledger for every decodable packet and stream operation.
//!
//! The universe is supplied by the decoder boundary as typed keys. This module
//! does not inspect source text and does not infer support from runtime traffic.
//! A cutover audit succeeds only when every declared key has exactly one closed
//! disposition and no unresolved row remains.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractDisposition<Refusal> {
    Implemented,
    ProvenNoOp,
    Unsupported(Refusal),
    Unresolved,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContractCoverageCounts {
    pub implemented: usize,
    pub proven_no_op: usize,
    pub unsupported: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractCoverageError<Key> {
    DuplicateSurfaceKey(Key),
    DuplicateDisposition(Key),
    UndeclaredKey(Key),
    MissingDisposition(Box<[Key]>),
    Unresolved(Box<[Key]>),
}

#[derive(Clone, Debug)]
pub struct RefusalClosureLedger<Key, Refusal> {
    surface: BTreeSet<Key>,
    dispositions: BTreeMap<Key, ContractDisposition<Refusal>>,
}

impl<Key: Clone + Ord, Refusal> RefusalClosureLedger<Key, Refusal> {
    pub fn new(surface: impl IntoIterator<Item = Key>) -> Result<Self, ContractCoverageError<Key>> {
        let mut declared = BTreeSet::new();
        for key in surface {
            if !declared.insert(key.clone()) {
                return Err(ContractCoverageError::DuplicateSurfaceKey(key));
            }
        }
        Ok(Self {
            surface: declared,
            dispositions: BTreeMap::new(),
        })
    }

    pub fn record(
        &mut self,
        key: Key,
        disposition: ContractDisposition<Refusal>,
    ) -> Result<(), ContractCoverageError<Key>> {
        if !self.surface.contains(&key) {
            return Err(ContractCoverageError::UndeclaredKey(key));
        }
        if self.dispositions.contains_key(&key) {
            return Err(ContractCoverageError::DuplicateDisposition(key));
        }
        self.dispositions.insert(key, disposition);
        Ok(())
    }

    pub fn disposition(&self, key: &Key) -> Option<&ContractDisposition<Refusal>> {
        self.dispositions.get(key)
    }

    pub fn audit(&self) -> Result<ContractCoverageCounts, ContractCoverageError<Key>> {
        let present = self.dispositions.keys().cloned().collect::<BTreeSet<_>>();
        let missing = self
            .surface
            .difference(&present)
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(ContractCoverageError::MissingDisposition(
                missing.into_boxed_slice(),
            ));
        }

        let unresolved = self
            .dispositions
            .iter()
            .filter(|(_, disposition)| matches!(disposition, ContractDisposition::Unresolved))
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if !unresolved.is_empty() {
            return Err(ContractCoverageError::Unresolved(
                unresolved.into_boxed_slice(),
            ));
        }

        let mut counts = ContractCoverageCounts::default();
        for disposition in self.dispositions.values() {
            match disposition {
                ContractDisposition::Implemented => counts.implemented += 1,
                ContractDisposition::ProvenNoOp => counts.proven_no_op += 1,
                ContractDisposition::Unsupported(_) => counts.unsupported += 1,
                ContractDisposition::Unresolved => unreachable!("handled above"),
            }
        }
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum Operation {
        Exec,
        Query,
        Delay,
        Retired,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Refusal {
        DelayUnavailable,
    }

    #[test]
    fn cutover_requires_one_closed_row_for_every_declared_operation() {
        let mut ledger = RefusalClosureLedger::new([
            Operation::Exec,
            Operation::Query,
            Operation::Delay,
            Operation::Retired,
        ])
        .unwrap();
        ledger
            .record(Operation::Exec, ContractDisposition::Implemented)
            .unwrap();
        ledger
            .record(Operation::Query, ContractDisposition::Implemented)
            .unwrap();
        ledger
            .record(
                Operation::Delay,
                ContractDisposition::Unsupported(Refusal::DelayUnavailable),
            )
            .unwrap();
        assert_eq!(
            ledger.audit(),
            Err(ContractCoverageError::MissingDisposition(Box::new([
                Operation::Retired
            ])))
        );
        ledger
            .record(Operation::Retired, ContractDisposition::ProvenNoOp)
            .unwrap();
        assert_eq!(
            ledger.audit().unwrap(),
            ContractCoverageCounts {
                implemented: 2,
                proven_no_op: 1,
                unsupported: 1,
            }
        );
    }

    #[test]
    fn unresolved_is_a_cutover_failure_not_an_unsupported_alias() {
        let mut ledger = RefusalClosureLedger::<_, Refusal>::new([Operation::Delay]).unwrap();
        ledger
            .record(Operation::Delay, ContractDisposition::Unresolved)
            .unwrap();
        assert_eq!(
            ledger.audit(),
            Err(ContractCoverageError::Unresolved(Box::new([
                Operation::Delay
            ])))
        );
    }

    #[test]
    fn duplicate_and_undeclared_rows_are_rejected() {
        assert_eq!(
            RefusalClosureLedger::<_, Refusal>::new([Operation::Exec, Operation::Exec])
                .unwrap_err(),
            ContractCoverageError::DuplicateSurfaceKey(Operation::Exec)
        );
        let mut ledger = RefusalClosureLedger::new([Operation::Exec]).unwrap();
        ledger
            .record(Operation::Exec, ContractDisposition::<Refusal>::Implemented)
            .unwrap();
        assert_eq!(
            ledger.record(Operation::Exec, ContractDisposition::Implemented),
            Err(ContractCoverageError::DuplicateDisposition(Operation::Exec))
        );
        assert_eq!(
            ledger.record(Operation::Query, ContractDisposition::Implemented),
            Err(ContractCoverageError::UndeclaredKey(Operation::Query))
        );
    }
}
