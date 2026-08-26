//! Resolved argument-buffer participation before dependency-domain compilation.
//!
//! These operations preserve exactly what an encoder declared. They are not
//! residency hints and they do not retain the named API object. Dependency
//! compilation receives the source-queue/FIFO hazard domain plus canonical
//! backing and optional heap membership from the resource graph. Deprecated
//! unqualified render declarations remain semantic participation but produce
//! no hazard intent, as required by their API contract.

use crate::{AccessIntent, AccessMode, AccessScope, StageScope};
use reims_vgpu_protocol::{
    BackingId, HazardDomainId, HeapObject, RenderStages, ResourceId, ResourceObject, ResourceUsage,
};

/// Encoder context carried by one participation declaration.
///
/// An unqualified render declaration is distinct from a compute declaration:
/// their hazard behavior differs even though neither record carries stage
/// bits. Qualified render declarations retain the complete stage bitset rather
/// than narrowing a multi-stage value to one enum member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticipationScope {
    Compute,
    Render { stages: Option<RenderStages> },
}

/// One resolved participation operation at its exact encoder position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticipationOperation {
    Resource {
        resource: ResourceId<ResourceObject>,
        usage: ResourceUsage,
        scope: ParticipationScope,
    },
    Heap {
        heap: ResourceId<HeapObject>,
        scope: ParticipationScope,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParticipationResourceTarget {
    /// Complete canonical alias closure in stable identity order.
    pub backings: Box<[BackingId]>,
    pub heap: Option<ResourceId<HeapObject>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticipationAccessError {
    ResourceBackingAbsent,
}

impl ParticipationOperation {
    pub const fn scope(self) -> ParticipationScope {
        match self {
            Self::Resource { scope, .. } | Self::Heap { scope, .. } => scope,
        }
    }

    /// Compile the declaration into exact dependency intents.
    ///
    /// `resource_target` is required only for a hazard-tracked resource use.
    /// Heap membership emits a second identity so a direct resource use meets
    /// a `useHeap` declaration without enumerating heap contents.
    pub fn compile_accesses(
        self,
        hazard_domain: HazardDomainId,
        resource_target: Option<ParticipationResourceTarget>,
    ) -> Result<Box<[AccessIntent]>, ParticipationAccessError> {
        let Some(stages) = hazard_stages(self.scope()) else {
            return Ok(Box::new([]));
        };
        match self {
            Self::Resource {
                resource, usage, ..
            } => {
                let Some(mode) = access_mode(usage) else {
                    return Ok(Box::new([]));
                };
                let target =
                    resource_target.ok_or(ParticipationAccessError::ResourceBackingAbsent)?;
                if target.backings.is_empty() {
                    return Err(ParticipationAccessError::ResourceBackingAbsent);
                }
                let mut intents = target
                    .backings
                    .iter()
                    .copied()
                    .map(|backing| {
                        AccessIntent::for_backing(
                            hazard_domain,
                            backing,
                            Some(resource),
                            AccessScope::WholeBacking,
                            mode,
                            stages,
                        )
                        .expect("whole backing is a backing-scoped access")
                    })
                    .collect::<Vec<_>>();
                if let Some(heap) = target.heap {
                    intents.push(AccessIntent::for_heap(hazard_domain, heap, mode, stages));
                }
                Ok(intents.into_boxed_slice())
            }
            Self::Heap { heap, .. } => Ok(Box::new([AccessIntent::for_heap(
                hazard_domain,
                heap,
                AccessMode::Unknown,
                stages,
            )])),
        }
    }
}

fn hazard_stages(scope: ParticipationScope) -> Option<StageScope> {
    match scope {
        ParticipationScope::Compute => Some(StageScope::Compute),
        ParticipationScope::Render {
            stages: Some(stages),
        } => Some(StageScope::Render(stages)),
        ParticipationScope::Render { stages: None } => None,
    }
}

fn access_mode(usage: ResourceUsage) -> Option<AccessMode> {
    match (usage.reads() || usage.samples(), usage.writes()) {
        (true, false) => Some(AccessMode::Read),
        (false, true) => Some(AccessMode::Write),
        (true, true) => Some(AccessMode::ReadWrite),
        (false, false) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_participation_keeps_combined_usage_and_stage_bits() {
        let usage = ResourceUsage::from_bits(u32::from(
            ResourceUsage::READ | ResourceUsage::WRITE | ResourceUsage::SAMPLE,
        ))
        .unwrap();
        let stages = RenderStages::from_bits(u16::from(
            RenderStages::VERTEX | RenderStages::FRAGMENT | RenderStages::MESH,
        ))
        .unwrap();
        let operation = ParticipationOperation::Resource {
            resource: ResourceId::new(7, 3),
            usage,
            scope: ParticipationScope::Render {
                stages: Some(stages),
            },
        };

        assert_eq!(
            operation.scope(),
            ParticipationScope::Render {
                stages: Some(stages)
            }
        );
        let ParticipationOperation::Resource { usage, .. } = operation else {
            unreachable!()
        };
        assert_eq!(usage.bits(), ResourceUsage::KNOWN);
    }

    #[test]
    fn heap_participation_has_identity_and_scope_but_no_fabricated_usage() {
        let first = ParticipationOperation::Heap {
            heap: ResourceId::new(5, 1),
            scope: ParticipationScope::Compute,
        };
        let reused = ParticipationOperation::Heap {
            heap: ResourceId::new(5, 2),
            scope: ParticipationScope::Compute,
        };

        assert_ne!(first, reused);
        assert_eq!(first.scope(), ParticipationScope::Compute);
    }

    #[test]
    fn qualified_resource_participation_compiles_backing_and_heap_hazards() {
        let stages =
            RenderStages::from_bits(u16::from(RenderStages::VERTEX | RenderStages::FRAGMENT))
                .unwrap();
        let operation = ParticipationOperation::Resource {
            resource: ResourceId::new(7, 3),
            usage: ResourceUsage::from_bits(3).unwrap(),
            scope: ParticipationScope::Render {
                stages: Some(stages),
            },
        };
        let heap = ResourceId::new(9, 2);
        let intents = operation
            .compile_accesses(
                HazardDomainId::new(4),
                Some(ParticipationResourceTarget {
                    backings: Box::new([BackingId::new(11)]),
                    heap: Some(heap),
                }),
            )
            .unwrap();

        assert_eq!(intents.len(), 2);
        assert_eq!(intents[0].mode, AccessMode::ReadWrite);
        assert_eq!(intents[0].stages, StageScope::Render(stages));
        assert_eq!(
            intents[1],
            AccessIntent::for_heap(
                HazardDomainId::new(4),
                heap,
                AccessMode::ReadWrite,
                StageScope::Render(stages)
            )
        );
    }

    #[test]
    fn deprecated_unqualified_render_participation_has_no_hazard_intent() {
        let operation = ParticipationOperation::Resource {
            resource: ResourceId::new(7, 3),
            usage: ResourceUsage::from_bits(u32::from(ResourceUsage::WRITE)).unwrap(),
            scope: ParticipationScope::Render { stages: None },
        };
        assert!(operation
            .compile_accesses(HazardDomainId::new(1), None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn heap_participation_meets_direct_heap_member_access_conservatively() {
        let heap = ResourceId::new(5, 1);
        let operation = ParticipationOperation::Heap {
            heap,
            scope: ParticipationScope::Compute,
        };
        assert_eq!(
            operation
                .compile_accesses(HazardDomainId::new(2), None)
                .unwrap()
                .as_ref(),
            [AccessIntent::for_heap(
                HazardDomainId::new(2),
                heap,
                AccessMode::Unknown,
                StageScope::Compute
            )]
        );
    }
}
