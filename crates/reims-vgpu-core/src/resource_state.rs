//! Resource-resolved state transitions carried outside the wire decoder.

use crate::BackingRegion;
use reims_vgpu_protocol::{BackingId, ResourceId, ResourceObject, ResourceValidityOps, SurfaceId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedResourceStateTarget {
    pub backing: BackingId,
    pub regions: Box<[BackingRegion]>,
}

/// One decoded validity statement paired with the resource lifetime it names.
///
/// `resource` is absent when the statement names only resolved surface
/// mappings and no constructed task resource. Deferred pre-construction
/// currency is recorded before this resolved command is formed; no task-local
/// object reference crosses the execution boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedResourceState {
    pub resource: Option<ResourceId<ResourceObject>>,
    pub mappings: Box<[SurfaceId]>,
    pub targets: Box<[ResolvedResourceStateTarget]>,
    pub ops: ResourceValidityOps,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_state_carries_only_generational_and_surface_identities() {
        let update = ResolvedResourceState {
            resource: None,
            mappings: vec![SurfaceId::new(7)].into_boxed_slice(),
            targets: Box::new([]),
            ops: ResourceValidityOps::PAGE_ON,
        };
        assert_eq!(update.resource, None);
        assert_eq!(update.mappings.as_ref(), [SurfaceId::new(7)]);
        assert!(update.targets.is_empty());
    }

    #[test]
    fn backend_binding_preserves_semantic_backing_coverage() {
        let backing = BackingId::new(4);
        let region = BackingRegion::Linear(crate::LinearRange::new(8, 16).unwrap());
        let update = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([region]),
            }]),
            ops: ResourceValidityOps {
                clear_host_valid: 0,
                set_host_valid: 1,
                clear_guest_valid: 0,
                set_guest_valid: 0,
            },
        };
        let transition = crate::ResolvedValidityTransition::bind(
            &update,
            Some(reims_vgpu_protocol::SubmissionId::new(9).into()),
            None,
            |found| {
                assert_eq!(found, backing);
                crate::ValidityRepresentations {
                    host_write: Some(reims_vgpu_protocol::RepresentationId::new(6)),
                    host_ingress_destination: None,
                    guest_upload_destination: None,
                    guest_visibility_source: None,
                    guest_visibility_destination: crate::GUEST_REPRESENTATION,
                }
            },
        );
        assert_eq!(transition.targets[0].backing, backing);
        assert_eq!(transition.targets[0].regions.as_ref(), [region]);
    }
}
