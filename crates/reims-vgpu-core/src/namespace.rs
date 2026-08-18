//! Generational task/reference namespaces for semantic objects without storage.

use reims_vgpu_protocol::{ObjectRef, ResourceId, TaskId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NamespaceError {
    IdentitySpaceExhausted,
}

#[derive(Debug)]
struct Slot<M> {
    index: u32,
    next_generation: u32,
    current: Option<ResourceId<M>>,
}

/// One API-specific reference namespace, partitioned by task.
///
/// Samplers, pipelines, heaps, and fences allocate references independently.
/// The marker `M` keeps equal integers in those namespaces non-interchangeable,
/// while the generation makes deletion followed by reference reuse a new
/// internal lifetime.
#[derive(Debug)]
pub struct ReferenceNamespace<M> {
    slots: BTreeMap<(TaskId, ObjectRef<M>), Slot<M>>,
    next_index: u32,
}

impl<M> Default for ReferenceNamespace<M> {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
            next_index: 1,
        }
    }
}

impl<M> ReferenceNamespace<M> {
    pub fn publish(
        &mut self,
        task: TaskId,
        object: ObjectRef<M>,
    ) -> Result<ResourceId<M>, NamespaceError> {
        if let Some(current) = self
            .slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
        {
            return Ok(current);
        }
        let slot = if let Some(slot) = self.slots.get_mut(&(task, object)) {
            slot
        } else {
            let index = self.next_index;
            self.next_index = self
                .next_index
                .checked_add(1)
                .ok_or(NamespaceError::IdentitySpaceExhausted)?;
            self.slots.entry((task, object)).or_insert(Slot {
                index,
                next_generation: 1,
                current: None,
            })
        };
        let id = ResourceId::new(slot.index, slot.next_generation);
        slot.next_generation = slot
            .next_generation
            .checked_add(1)
            .ok_or(NamespaceError::IdentitySpaceExhausted)?;
        slot.current = Some(id);
        Ok(id)
    }

    pub fn resolve(&self, task: TaskId, object: ObjectRef<M>) -> Option<ResourceId<M>> {
        self.slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
    }

    pub fn release(&mut self, task: TaskId, object: ObjectRef<M>) -> bool {
        self.slots
            .get_mut(&(task, object))
            .and_then(|slot| slot.current.take())
            .is_some()
    }

    pub fn release_task(&mut self, task: TaskId) -> usize {
        let mut released = 0;
        for ((slot_task, _), slot) in &mut self.slots {
            if *slot_task == task && slot.current.take().is_some() {
                released += 1;
            }
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Sampler {}

    #[test]
    fn reference_reuse_advances_generation_and_tasks_are_independent() {
        let mut namespace = ReferenceNamespace::<Sampler>::default();
        let object = ObjectRef::new(7);
        let first = namespace.publish(TaskId::new(1), object).unwrap();
        assert_eq!(namespace.publish(TaskId::new(1), object).unwrap(), first);
        let other_task = namespace.publish(TaskId::new(2), object).unwrap();
        assert_ne!(first, other_task);

        assert!(namespace.release(TaskId::new(1), object));
        let replacement = namespace.publish(TaskId::new(1), object).unwrap();
        assert_eq!(first.index(), replacement.index());
        assert_ne!(first.generation(), replacement.generation());
        assert_eq!(namespace.release_task(TaskId::new(2)), 1);
        assert_eq!(namespace.resolve(TaskId::new(2), object), None);
    }
}
