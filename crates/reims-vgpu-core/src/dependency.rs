//! Ordered resource-hazard compilation over immutable access summaries.
//!
//! Hazard domains are contract inputs. Two accesses in different domains do
//! not acquire a guest-semantic edge merely because they touch one backing;
//! explicit synchronization and Vulkan-only host-safety constraints are
//! compiled by their own owners.

use crate::{AccessIntent, AccessMode, AccessScope, AccessTarget};
use reims_vgpu_protocol::{HazardDomainId, IngressOrdinal, TransactionId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HazardCause {
    Buffer,
    Image,
    WholeBacking,
    WholeHeap,
    WholeDomain,
    Alias,
    UnknownMode,
}

/// Vulkan-only ordering that cannot advance guest semantic state.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HostSafetyCause {
    QueueExternalSynchronization,
    QueueFamilyOwnership,
    NativeObjectLifetime,
    SamePhysicalQueueProducerFirst,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HostSafetyEdge {
    pub constrained: TransactionId,
    pub prerequisite: TransactionId,
    pub cause: HostSafetyCause,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HazardEdge {
    pub newer: TransactionId,
    pub older: TransactionId,
    pub newer_ordinal: IngressOrdinal,
    pub older_ordinal: IngressOrdinal,
    pub cause: HazardCause,
}

/// Exact conflicting access pair behind one semantic dependency.
///
/// The edge is the scheduling fact; these access operands are retained for the
/// backend to derive Vulkan stage, access, range, and subresource barriers
/// without reconstructing them from a coarse cause label.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HazardRequirement {
    pub edge: HazardEdge,
    pub earlier: AccessIntent,
    pub later: AccessIntent,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HazardCompilation {
    pub edges: Vec<HazardEdge>,
    pub requirements: Vec<HazardRequirement>,
    pub intents: u64,
    pub records_examined: u64,
    pub unknown_modes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyCompileError {
    DuplicateTransaction,
    IngressDidNotIncrease,
    MalformedIntent,
}

#[derive(Clone, Debug)]
struct TransactionAccesses {
    transaction: TransactionId,
    ordinal: IngressOrdinal,
    intents: Box<[AccessIntent]>,
}

#[derive(Clone, Debug, Default)]
struct DomainHazards {
    by_target: BTreeMap<AccessTarget, BTreeMap<IngressOrdinal, TransactionAccesses>>,
    whole_domain: BTreeMap<IngressOrdinal, TransactionAccesses>,
}

#[derive(Clone, Debug)]
struct RetirementKeys {
    domain: HazardDomainId,
    ordinal: IngressOrdinal,
    targets: BTreeSet<AccessTarget>,
    whole_domain: bool,
}

/// Serial owner of live hazard metadata.
#[derive(Clone, Debug, Default)]
pub struct DependencyCompiler {
    domains: BTreeMap<HazardDomainId, DomainHazards>,
    retirement: BTreeMap<TransactionId, BTreeMap<HazardDomainId, RetirementKeys>>,
    last_ordinal: Option<IngressOrdinal>,
}

impl DependencyCompiler {
    pub fn compile(
        &mut self,
        transaction: TransactionId,
        ordinal: IngressOrdinal,
        intents: impl Into<Box<[AccessIntent]>>,
    ) -> Result<HazardCompilation, DependencyCompileError> {
        if self.retirement.contains_key(&transaction) {
            return Err(DependencyCompileError::DuplicateTransaction);
        }
        if self.last_ordinal.is_some_and(|last| ordinal <= last) {
            return Err(DependencyCompileError::IngressDidNotIncrease);
        }
        let intents = intents.into();
        if intents.iter().any(|intent| {
            !matches!(
                (intent.scope, intent.target),
                (AccessScope::WholeDomain, None)
                    | (AccessScope::WholeHeap, Some(AccessTarget::Heap(_)))
                    | (
                        AccessScope::Linear(_) | AccessScope::Image(_) | AccessScope::WholeBacking,
                        Some(AccessTarget::Backing(_))
                    )
            )
        }) {
            return Err(DependencyCompileError::MalformedIntent);
        }

        let mut compilation = HazardCompilation {
            intents: intents.len() as u64,
            unknown_modes: intents
                .iter()
                .filter(|intent| intent.mode == AccessMode::Unknown)
                .count() as u64,
            ..HazardCompilation::default()
        };
        let mut edges = BTreeSet::new();

        for intent in &intents {
            if let Some(domain) = self.domains.get(&intent.hazard_domain) {
                for older in domain.whole_domain.values() {
                    examine(
                        transaction,
                        ordinal,
                        intent,
                        older,
                        &mut compilation.records_examined,
                        &mut edges,
                        &mut compilation.requirements,
                    );
                }
                match intent.target {
                    Some(target) => {
                        if let Some(records) = domain.by_target.get(&target) {
                            for older in records.values() {
                                examine(
                                    transaction,
                                    ordinal,
                                    intent,
                                    older,
                                    &mut compilation.records_examined,
                                    &mut edges,
                                    &mut compilation.requirements,
                                );
                            }
                        }
                    }
                    None => {
                        for records in domain.by_target.values() {
                            for older in records.values() {
                                examine(
                                    transaction,
                                    ordinal,
                                    intent,
                                    older,
                                    &mut compilation.records_examined,
                                    &mut edges,
                                    &mut compilation.requirements,
                                );
                            }
                        }
                    }
                }
            }
        }

        compilation.edges = edges.into_iter().collect();
        self.insert(transaction, ordinal, intents.into_vec());
        self.last_ordinal = Some(ordinal);
        Ok(compilation)
    }

    fn insert(
        &mut self,
        transaction: TransactionId,
        ordinal: IngressOrdinal,
        intents: Vec<AccessIntent>,
    ) {
        let mut grouped: BTreeMap<(HazardDomainId, Option<AccessTarget>), Vec<AccessIntent>> =
            BTreeMap::new();
        for intent in intents {
            grouped
                .entry((intent.hazard_domain, intent.target))
                .or_default()
                .push(intent);
        }

        let mut retirement_by_domain: BTreeMap<HazardDomainId, RetirementKeys> = BTreeMap::new();
        for ((domain_id, target), grouped_intents) in grouped {
            let domain = self.domains.entry(domain_id).or_default();
            let access = TransactionAccesses {
                transaction,
                ordinal,
                intents: grouped_intents.into_boxed_slice(),
            };
            let keys = retirement_by_domain
                .entry(domain_id)
                .or_insert_with(|| RetirementKeys {
                    domain: domain_id,
                    ordinal,
                    targets: BTreeSet::new(),
                    whole_domain: false,
                });
            match target {
                Some(target) => {
                    domain
                        .by_target
                        .entry(target)
                        .or_default()
                        .insert(ordinal, access);
                    keys.targets.insert(target);
                }
                None => {
                    domain.whole_domain.insert(ordinal, access);
                    keys.whole_domain = true;
                }
            }
        }

        self.retirement.insert(transaction, retirement_by_domain);
    }

    pub fn retire(&mut self, transaction: TransactionId) -> bool {
        let Some(domains) = self.retirement.remove(&transaction) else {
            return false;
        };
        for keys in domains.values() {
            self.remove_keys(keys);
        }
        true
    }

    fn remove_keys(&mut self, keys: &RetirementKeys) {
        let mut remove_domain = false;
        if let Some(domain) = self.domains.get_mut(&keys.domain) {
            if keys.whole_domain {
                domain.whole_domain.remove(&keys.ordinal);
            }
            for target in &keys.targets {
                if let Some(records) = domain.by_target.get_mut(target) {
                    records.remove(&keys.ordinal);
                    if records.is_empty() {
                        domain.by_target.remove(target);
                    }
                }
            }
            remove_domain = domain.whole_domain.is_empty() && domain.by_target.is_empty();
        }
        if remove_domain {
            self.domains.remove(&keys.domain);
        }
    }

    pub fn live_transactions(&self) -> usize {
        self.retirement.len()
    }
}

fn examine(
    newer_transaction: TransactionId,
    newer_ordinal: IngressOrdinal,
    newer: &AccessIntent,
    older: &TransactionAccesses,
    records_examined: &mut u64,
    edges: &mut BTreeSet<HazardEdge>,
    requirements: &mut Vec<HazardRequirement>,
) {
    for old_intent in older.intents.iter() {
        *records_examined += 1;
        if !newer.mode.conflicts_with(old_intent.mode) || !scopes_overlap(newer, old_intent) {
            continue;
        }
        let edge = HazardEdge {
            newer: newer_transaction,
            older: older.transaction,
            newer_ordinal,
            older_ordinal: older.ordinal,
            cause: cause(newer, old_intent),
        };
        edges.insert(edge);
        requirements.push(HazardRequirement {
            edge,
            earlier: *old_intent,
            later: *newer,
        });
    }
}

fn scopes_overlap(left: &AccessIntent, right: &AccessIntent) -> bool {
    match (left.scope, right.scope) {
        (AccessScope::WholeDomain, _) | (_, AccessScope::WholeDomain) => true,
        _ if left.target != right.target => false,
        (AccessScope::WholeBacking, _) | (_, AccessScope::WholeBacking) => true,
        (AccessScope::WholeHeap, _) | (_, AccessScope::WholeHeap) => true,
        (AccessScope::Linear(left), AccessScope::Linear(right)) => left.overlaps(right),
        (AccessScope::Image(left), AccessScope::Image(right)) => left.overlaps(right),
        // A backing used through incompatible coordinate descriptions cannot
        // be proved disjoint, so it meets at whole-backing precision.
        _ => true,
    }
}

fn cause(left: &AccessIntent, right: &AccessIntent) -> HazardCause {
    if left.mode == AccessMode::Unknown || right.mode == AccessMode::Unknown {
        return HazardCause::UnknownMode;
    }
    if left.resource.is_some() && right.resource.is_some() && left.resource != right.resource {
        return HazardCause::Alias;
    }
    match (left.scope, right.scope) {
        (AccessScope::WholeDomain, _) | (_, AccessScope::WholeDomain) => HazardCause::WholeDomain,
        (AccessScope::WholeBacking, _) | (_, AccessScope::WholeBacking) => {
            HazardCause::WholeBacking
        }
        (AccessScope::WholeHeap, _) | (_, AccessScope::WholeHeap) => HazardCause::WholeHeap,
        (AccessScope::Image(_), AccessScope::Image(_)) => HazardCause::Image,
        (AccessScope::Linear(_), AccessScope::Linear(_)) => HazardCause::Buffer,
        _ => HazardCause::WholeBacking,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageAspect, ImageSubresourceRange, LinearRange, StageScope, TexelBox};
    use reims_vgpu_protocol::{BackingId, HeapObject, ResourceId, ResourceObject};

    fn intent(
        domain: u64,
        backing: u64,
        resource: u32,
        scope: AccessScope,
        mode: AccessMode,
    ) -> AccessIntent {
        AccessIntent::for_backing(
            HazardDomainId::new(domain),
            BackingId::new(backing),
            Some(ResourceId::<ResourceObject>::new(resource, 1)),
            scope,
            mode,
            StageScope::Compute,
        )
        .unwrap()
    }

    fn compile(
        compiler: &mut DependencyCompiler,
        id: u64,
        intents: Vec<AccessIntent>,
    ) -> HazardCompilation {
        compiler
            .compile(TransactionId::new(id), IngressOrdinal::new(id), intents)
            .unwrap()
    }

    #[test]
    fn read_read_has_no_resource_hazard_and_write_orders_after_all_readers() {
        let range = AccessScope::Linear(LinearRange::new(0, 64).unwrap());
        let mut compiler = DependencyCompiler::default();
        assert!(compile(
            &mut compiler,
            1,
            vec![intent(1, 9, 1, range, AccessMode::Read)]
        )
        .edges
        .is_empty());
        assert!(compile(
            &mut compiler,
            2,
            vec![intent(1, 9, 1, range, AccessMode::Read)]
        )
        .edges
        .is_empty());
        let write = compile(
            &mut compiler,
            3,
            vec![intent(1, 9, 1, range, AccessMode::Write)],
        );
        assert_eq!(
            write
                .edges
                .iter()
                .map(|edge| edge.older.get())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(write
            .edges
            .iter()
            .all(|edge| edge.newer_ordinal > edge.older_ordinal));
    }

    #[test]
    fn heap_declaration_meets_only_accesses_from_the_same_heap_generation() {
        let mut compiler = DependencyCompiler::default();
        let domain = HazardDomainId::new(1);
        let heap = ResourceId::<HeapObject>::new(4, 2);
        compile(
            &mut compiler,
            1,
            vec![AccessIntent::for_heap(
                domain,
                heap,
                AccessMode::Unknown,
                StageScope::Compute,
            )],
        );
        let same = compile(
            &mut compiler,
            2,
            vec![AccessIntent::for_heap(
                domain,
                heap,
                AccessMode::Write,
                StageScope::Compute,
            )],
        );
        assert_eq!(same.edges.len(), 1);
        assert_eq!(same.edges[0].cause, HazardCause::UnknownMode);

        let reused = compile(
            &mut compiler,
            3,
            vec![AccessIntent::for_heap(
                domain,
                ResourceId::new(4, 3),
                AccessMode::Write,
                StageScope::Compute,
            )],
        );
        assert!(reused.edges.is_empty());
    }

    #[test]
    fn disjoint_ranges_do_not_conflict_and_touching_half_open_ranges_do_not_overlap() {
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![intent(
                1,
                9,
                1,
                AccessScope::Linear(LinearRange::new(0, 64).unwrap()),
                AccessMode::Write,
            )],
        );
        let disjoint = compile(
            &mut compiler,
            2,
            vec![intent(
                1,
                9,
                1,
                AccessScope::Linear(LinearRange::new(64, 64).unwrap()),
                AccessMode::Write,
            )],
        );
        assert!(disjoint.edges.is_empty());
    }

    #[test]
    fn aliases_meet_through_backing_identity() {
        let range = AccessScope::Linear(LinearRange::new(4, 8).unwrap());
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![intent(1, 9, 1, range, AccessMode::Write)],
        );
        let alias = compile(
            &mut compiler,
            2,
            vec![intent(1, 9, 2, range, AccessMode::Read)],
        );
        assert_eq!(alias.edges.len(), 1);
        assert_eq!(alias.edges[0].cause, HazardCause::Alias);
        assert_eq!(alias.requirements.len(), 1);
        assert_eq!(alias.requirements[0].earlier.mode, AccessMode::Write);
        assert_eq!(alias.requirements[0].later.mode, AccessMode::Read);
        assert_eq!(alias.requirements[0].earlier.scope, range);
        assert_eq!(alias.requirements[0].later.scope, range);
    }

    #[test]
    fn image_aspects_mips_layers_and_texel_boxes_are_independent_when_disjoint() {
        let image = |aspect, mip, layer, origin| {
            AccessScope::Image(
                ImageSubresourceRange::new(
                    aspect,
                    mip,
                    1,
                    layer,
                    1,
                    Some(TexelBox::new(origin, [8, 8, 1]).unwrap()),
                )
                .unwrap(),
            )
        };
        for (first, second) in [
            (
                image(ImageAspect::Color, 0, 0, [0, 0, 0]),
                image(ImageAspect::Depth, 0, 0, [0, 0, 0]),
            ),
            (
                image(ImageAspect::Color, 0, 0, [0, 0, 0]),
                image(ImageAspect::Color, 1, 0, [0, 0, 0]),
            ),
            (
                image(ImageAspect::Color, 0, 0, [0, 0, 0]),
                image(ImageAspect::Color, 0, 1, [0, 0, 0]),
            ),
            (
                image(ImageAspect::Color, 0, 0, [0, 0, 0]),
                image(ImageAspect::Color, 0, 0, [8, 0, 0]),
            ),
        ] {
            let mut compiler = DependencyCompiler::default();
            compile(
                &mut compiler,
                1,
                vec![intent(1, 4, 1, first, AccessMode::Write)],
            );
            assert!(compile(
                &mut compiler,
                2,
                vec![intent(1, 4, 1, second, AccessMode::Write)]
            )
            .edges
            .is_empty());
        }
    }

    #[test]
    fn different_hazard_domains_do_not_invent_cross_queue_semantics() {
        let range = AccessScope::WholeBacking;
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![intent(1, 9, 1, range, AccessMode::Write)],
        );
        let cross_domain = compile(
            &mut compiler,
            2,
            vec![intent(2, 9, 1, range, AccessMode::Read)],
        );
        assert!(cross_domain.edges.is_empty());
    }

    #[test]
    fn whole_domain_precision_conflicts_with_every_backing_in_only_that_domain() {
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![
                intent(1, 9, 1, AccessScope::WholeBacking, AccessMode::Read),
                intent(1, 10, 2, AccessScope::WholeBacking, AccessMode::Read),
                intent(2, 11, 3, AccessScope::WholeBacking, AccessMode::Read),
            ],
        );
        let domain = compile(
            &mut compiler,
            2,
            vec![AccessIntent::whole_domain(
                HazardDomainId::new(1),
                AccessMode::Write,
                StageScope::All,
            )],
        );
        assert_eq!(domain.records_examined, 2);
        assert_eq!(
            domain.edges.len(),
            1,
            "edge causes de-duplicate one dependency"
        );
        assert_eq!(domain.edges[0].cause, HazardCause::WholeDomain);
        assert_eq!(domain.edges[0].older.get(), 1);
    }

    #[test]
    fn backing_precision_does_not_scan_unrelated_backing_records() {
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            (0..64)
                .map(|backing| {
                    intent(
                        1,
                        backing,
                        backing as u32,
                        AccessScope::WholeBacking,
                        AccessMode::Write,
                    )
                })
                .collect(),
        );
        let one = compile(
            &mut compiler,
            2,
            vec![intent(
                1,
                31,
                31,
                AccessScope::WholeBacking,
                AccessMode::Read,
            )],
        );
        assert_eq!(one.records_examined, 1);
        assert_eq!(one.edges.len(), 1);
    }

    #[test]
    fn unknown_mode_conflicts_visibly() {
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![intent(1, 9, 1, AccessScope::WholeBacking, AccessMode::Read)],
        );
        let unknown = compile(
            &mut compiler,
            2,
            vec![intent(
                1,
                9,
                1,
                AccessScope::WholeBacking,
                AccessMode::Unknown,
            )],
        );
        assert_eq!(unknown.unknown_modes, 1);
        assert_eq!(unknown.edges[0].cause, HazardCause::UnknownMode);
    }

    #[test]
    fn retirement_removes_only_the_completed_transactions_hazard_records() {
        let range = AccessScope::WholeBacking;
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![intent(1, 9, 1, range, AccessMode::Write)],
        );
        compile(
            &mut compiler,
            2,
            vec![intent(1, 10, 2, range, AccessMode::Write)],
        );
        assert!(compiler.retire(TransactionId::new(1)));
        assert_eq!(compiler.live_transactions(), 1);
        assert!(compile(
            &mut compiler,
            3,
            vec![intent(1, 9, 1, range, AccessMode::Read)]
        )
        .edges
        .is_empty());
        assert_eq!(
            compile(
                &mut compiler,
                4,
                vec![intent(1, 10, 2, range, AccessMode::Read)]
            )
            .edges[0]
                .older
                .get(),
            2
        );
    }

    #[test]
    fn retirement_removes_every_domain_owned_by_one_transaction() {
        let mut compiler = DependencyCompiler::default();
        compile(
            &mut compiler,
            1,
            vec![
                intent(1, 9, 1, AccessScope::WholeBacking, AccessMode::Write),
                intent(2, 10, 2, AccessScope::WholeBacking, AccessMode::Write),
            ],
        );
        assert!(compiler.retire(TransactionId::new(1)));
        assert_eq!(compiler.live_transactions(), 0);
        assert!(compile(
            &mut compiler,
            2,
            vec![
                intent(1, 9, 1, AccessScope::WholeBacking, AccessMode::Read),
                intent(2, 10, 2, AccessScope::WholeBacking, AccessMode::Read),
            ]
        )
        .edges
        .is_empty());
    }

    #[test]
    fn ingress_order_and_identity_errors_are_typed() {
        let mut compiler = DependencyCompiler::default();
        compile(&mut compiler, 1, Vec::new());
        assert_eq!(
            compiler.compile(TransactionId::new(1), IngressOrdinal::new(2), Vec::new()),
            Err(DependencyCompileError::DuplicateTransaction)
        );
        assert_eq!(
            compiler.compile(TransactionId::new(2), IngressOrdinal::new(1), Vec::new()),
            Err(DependencyCompileError::IngressDidNotIncrease)
        );
    }

    #[derive(Clone, Copy)]
    struct ModelAccess {
        id: u64,
        start: usize,
        end: usize,
        mode: AccessMode,
    }

    fn run_model(order: &[usize], accesses: &[ModelAccess]) -> ([u8; 16], BTreeMap<u64, Vec<u8>>) {
        let mut bytes = [0u8; 16];
        let mut reads = BTreeMap::new();
        for &index in order {
            let access = accesses[index];
            if matches!(access.mode, AccessMode::Read | AccessMode::ReadWrite) {
                reads.insert(access.id, bytes[access.start..access.end].to_vec());
            }
            if matches!(access.mode, AccessMode::Write | AccessMode::ReadWrite) {
                bytes[access.start..access.end].fill(access.id as u8);
            }
        }
        (bytes, reads)
    }

    #[test]
    fn generated_parallel_schedules_match_the_serial_interpreter() {
        for seed in 1..=64u64 {
            let mut random = seed;
            let mut next = || {
                random = random
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                random
            };
            let accesses: Vec<_> = (0..32)
                .map(|index| {
                    let start = (next() as usize) % 16;
                    let length = 1 + (next() as usize) % (16 - start);
                    let mode = match next() % 3 {
                        0 => AccessMode::Read,
                        1 => AccessMode::Write,
                        _ => AccessMode::ReadWrite,
                    };
                    ModelAccess {
                        id: index + 1,
                        start,
                        end: start + length,
                        mode,
                    }
                })
                .collect();

            let mut compiler = DependencyCompiler::default();
            let mut prerequisites: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
            for (index, access) in accesses.iter().enumerate() {
                let compilation = compile(
                    &mut compiler,
                    access.id,
                    vec![intent(
                        1,
                        1,
                        1,
                        AccessScope::Linear(
                            LinearRange::new(
                                access.start as u64,
                                (access.end - access.start) as u64,
                            )
                            .unwrap(),
                        ),
                        access.mode,
                    )],
                );
                prerequisites.entry(index).or_default().extend(
                    compilation
                        .edges
                        .iter()
                        .map(|edge| (edge.older.get() - 1) as usize),
                );
            }

            let serial_order: Vec<_> = (0..accesses.len()).collect();
            let expected = run_model(&serial_order, &accesses);
            let mut completed = BTreeSet::new();
            let mut parallel_order = Vec::new();
            while parallel_order.len() < accesses.len() {
                let ready: Vec<_> = (0..accesses.len())
                    .filter(|index| {
                        !completed.contains(index)
                            && prerequisites
                                .get(index)
                                .is_none_or(|required| required.is_subset(&completed))
                    })
                    .collect();
                assert!(!ready.is_empty(), "seed {seed}: compiler made a cycle");
                let chosen = ready[(next() as usize) % ready.len()];
                completed.insert(chosen);
                parallel_order.push(chosen);
            }
            assert_eq!(
                run_model(&parallel_order, &accesses),
                expected,
                "seed {seed}, order {parallel_order:?}"
            );
        }
    }
}
