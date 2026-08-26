//! Immutable semantic values owned by exact generational object lifetimes.

use reims_vgpu_protocol::ResourceId;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectStateError {
    DuplicateIdentity,
    UnknownIdentity,
}

/// One unbounded semantic value table keyed only by resolved object identity.
///
/// Raw task-local names remain in the namespace owner. This owner cannot
/// resolve or reuse them, and values leave it only at their exact lifecycle
/// retirement event. Already accepted work may retain the returned `Arc`.
pub struct ObjectStateOwner<M, V> {
    values: BTreeMap<ResourceId<M>, Arc<V>>,
}

impl<M, V> Default for ObjectStateOwner<M, V> {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }
}

impl<M, V> ObjectStateOwner<M, V> {
    pub fn declare(
        &mut self,
        identity: ResourceId<M>,
        value: V,
    ) -> Result<Arc<V>, ObjectStateError> {
        if self.values.contains_key(&identity) {
            return Err(ObjectStateError::DuplicateIdentity);
        }
        let value = Arc::new(value);
        self.values.insert(identity, Arc::clone(&value));
        Ok(value)
    }

    pub fn resolve(&self, identity: ResourceId<M>) -> Option<Arc<V>> {
        self.values.get(&identity).cloned()
    }

    pub fn retire(&mut self, identity: ResourceId<M>) -> Result<Arc<V>, ObjectStateError> {
        self.values
            .remove(&identity)
            .ok_or(ObjectStateError::UnknownIdentity)
    }

    pub fn live(&self) -> usize {
        self.values.len()
    }

    pub fn identities(&self) -> Box<[ResourceId<M>]> {
        self.values
            .keys()
            .copied()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Sampler {}

    #[test]
    fn retirement_removes_only_the_exact_generation_and_leases_remain_valid() {
        let mut owner = ObjectStateOwner::<Sampler, String>::default();
        let first = ResourceId::new(3, 1);
        let replacement = ResourceId::new(3, 2);
        let lease = owner.declare(first, "first".to_string()).unwrap();
        owner
            .declare(replacement, "replacement".to_string())
            .unwrap();
        assert_eq!(
            owner.declare(first, "duplicate".to_string()),
            Err(ObjectStateError::DuplicateIdentity)
        );

        assert_eq!(&*owner.retire(first).unwrap(), "first");
        assert_eq!(&*lease, "first");
        assert_eq!(&*owner.resolve(replacement).unwrap(), "replacement");
        assert_eq!(owner.live(), 1);
        assert_eq!(owner.retire(first), Err(ObjectStateError::UnknownIdentity));
    }
}
