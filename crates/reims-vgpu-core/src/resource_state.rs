//! Resource-resolved state transitions carried outside the wire decoder.

use reims_vgpu_protocol::{ObjectRef, ResourceId, ResourceObject, ResourceValidityOps};

/// One decoded validity statement paired with the resource lifetime it names.
///
/// `resource` is absent when the submission declares an object which has not
/// been constructed in this device yet. The typed serializer reference remains
/// available for mapping participation and deferred construction; numeric
/// equality never substitutes for a resource identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedResourceState {
    pub object: ObjectRef<ResourceObject>,
    pub resource: Option<ResourceId<ResourceObject>>,
    pub ops: ResourceValidityOps,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unconstructed_reference_is_not_invented_as_a_resource_identity() {
        let update = ResolvedResourceState {
            object: ObjectRef::new(7),
            resource: None,
            ops: ResourceValidityOps::PAGE_ON,
        };
        assert_eq!(update.object.get(), 7);
        assert_eq!(update.resource, None);
    }
}
