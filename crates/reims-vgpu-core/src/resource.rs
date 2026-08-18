//! Canonical task-owned resource, storage, and mapping graph.

use reims_vgpu_protocol::{
    BackingGeneration, ByteLength, ByteOffset, GuestVirtualAddress, MappingId, ObjectKind,
    ObjectRef, PlaneIndex, ResourceId, ResourceObject, StorageId, SubmissionId, SurfaceBackingId,
    TaskId,
};
use std::collections::{BTreeMap, BTreeSet};

type AnyResourceId = ResourceId<ResourceObject>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleState {
    Created,
    Prepared,
    InFlight,
    Released,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StorageBacking {
    Dedicated,
    TaskAddress {
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    },
    BufferRange {
        buffer: AnyResourceId,
        offset: ByteOffset,
        length: ByteLength,
    },
    IOSurfacePlane {
        surface: SurfaceBackingId,
        plane: PlaneIndex,
    },
    /// Mapper-path surface storage whose shared-backing identity is not yet
    /// established independently of the mapping object.
    MapperSurface {
        mapping: MappingId,
        plane: PlaneIndex,
    },
    HeapPlacement {
        heap: AnyResourceId,
        offset: ByteOffset,
        length: ByteLength,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageNode {
    pub id: StorageId,
    pub backing: StorageBacking,
    pub owners: BTreeSet<AnyResourceId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingNode {
    pub id: MappingId,
    pub task: TaskId,
    pub address: GuestVirtualAddress,
    pub length: ByteLength,
    pub storage: Option<StorageId>,
    pub committed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceNode {
    pub id: AnyResourceId,
    pub task: TaskId,
    pub object: ObjectRef<ResourceObject>,
    pub kind: ObjectKind,
    pub lifecycle: LifecycleState,
    pub storage: Option<StorageId>,
    pub backing_generation: BackingGeneration,
    pub parents: BTreeSet<AnyResourceId>,
    pub children: BTreeSet<AnyResourceId>,
    pub in_flight: BTreeSet<SubmissionId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphError {
    ReferenceAlreadyBound,
    ReferenceUnbound,
    ResourceAbsent,
    ParentAbsent,
    StorageAbsent,
    MappingAlreadyExists,
    MappingAbsent,
    SubmissionNotPrepared,
    IdentitySpaceExhausted,
}

#[derive(Debug)]
struct NamespaceSlot {
    index: u32,
    next_generation: u32,
    current: Option<AnyResourceId>,
}

/// One authority for object-name reuse and resource/storage/mapping lifetime.
#[derive(Debug)]
pub struct ResourceGraph {
    slots: BTreeMap<(TaskId, ObjectRef<ResourceObject>), NamespaceSlot>,
    resources: BTreeMap<AnyResourceId, ResourceNode>,
    storage: BTreeMap<StorageId, StorageNode>,
    mappings: BTreeMap<MappingId, MappingNode>,
    next_resource_index: u32,
    next_storage_id: u64,
}

impl Default for ResourceGraph {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
            resources: BTreeMap::new(),
            storage: BTreeMap::new(),
            mappings: BTreeMap::new(),
            next_resource_index: 1,
            next_storage_id: 1,
        }
    }
}

impl ResourceGraph {
    pub fn create_resource(
        &mut self,
        task: TaskId,
        object: ObjectRef<ResourceObject>,
        kind: ObjectKind,
        storage: Option<StorageId>,
        parents: impl IntoIterator<Item = AnyResourceId>,
    ) -> Result<AnyResourceId, GraphError> {
        if storage.is_some_and(|id| !self.storage.contains_key(&id)) {
            return Err(GraphError::StorageAbsent);
        }
        let parents: BTreeSet<_> = parents.into_iter().collect();
        if parents.iter().any(|id| !self.resources.contains_key(id)) {
            return Err(GraphError::ParentAbsent);
        }
        let key = (task, object);
        let slot = if let Some(slot) = self.slots.get_mut(&key) {
            if slot.current.is_some() {
                return Err(GraphError::ReferenceAlreadyBound);
            }
            slot
        } else {
            let index = self.next_resource_index;
            self.next_resource_index = self
                .next_resource_index
                .checked_add(1)
                .ok_or(GraphError::IdentitySpaceExhausted)?;
            self.slots.entry(key).or_insert(NamespaceSlot {
                index,
                next_generation: 1,
                current: None,
            })
        };
        let id = ResourceId::new(slot.index, slot.next_generation);
        slot.next_generation = slot
            .next_generation
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        slot.current = Some(id);
        for parent in &parents {
            self.resources
                .get_mut(parent)
                .expect("parents validated")
                .children
                .insert(id);
        }
        if let Some(storage) = storage {
            self.storage
                .get_mut(&storage)
                .expect("storage validated")
                .owners
                .insert(id);
        }
        self.resources.insert(
            id,
            ResourceNode {
                id,
                task,
                object,
                kind,
                lifecycle: LifecycleState::Created,
                storage,
                backing_generation: BackingGeneration::new(1),
                parents,
                children: BTreeSet::new(),
                in_flight: BTreeSet::new(),
            },
        );
        Ok(id)
    }

    pub fn resolve(
        &self,
        task: TaskId,
        object: ObjectRef<ResourceObject>,
    ) -> Option<AnyResourceId> {
        self.slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
    }

    pub fn resource(&self, id: AnyResourceId) -> Option<&ResourceNode> {
        self.resources.get(&id)
    }

    pub fn create_storage(&mut self, backing: StorageBacking) -> Result<StorageId, GraphError> {
        let id = StorageId::new(self.next_storage_id);
        self.next_storage_id = self
            .next_storage_id
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        self.storage.insert(
            id,
            StorageNode {
                id,
                backing,
                owners: BTreeSet::new(),
            },
        );
        Ok(id)
    }

    pub fn storage(&self, id: StorageId) -> Option<&StorageNode> {
        self.storage.get(&id)
    }

    pub fn mapper_storage(
        &mut self,
        mapping: MappingId,
        plane: PlaneIndex,
    ) -> Result<StorageId, GraphError> {
        if let Some(id) = self.storage.values().find_map(|storage| {
            (storage.backing == StorageBacking::MapperSurface { mapping, plane })
                .then_some(storage.id)
        }) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::MapperSurface { mapping, plane })
    }

    pub fn attach_initial_storage(
        &mut self,
        id: AnyResourceId,
        storage: StorageId,
    ) -> Result<(), GraphError> {
        if !self.storage.contains_key(&storage) {
            return Err(GraphError::StorageAbsent);
        }
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if node.storage.is_some() {
            return Err(GraphError::ReferenceAlreadyBound);
        }
        node.storage = Some(storage);
        self.storage.get_mut(&storage).unwrap().owners.insert(id);
        Ok(())
    }

    pub fn replace_backing(
        &mut self,
        id: AnyResourceId,
        storage: StorageId,
    ) -> Result<BackingGeneration, GraphError> {
        if !self.storage.contains_key(&storage) {
            return Err(GraphError::StorageAbsent);
        }
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if let Some(old) = node.storage.replace(storage) {
            if let Some(old) = self.storage.get_mut(&old) {
                old.owners.remove(&id);
            }
        }
        self.storage.get_mut(&storage).unwrap().owners.insert(id);
        let next = node
            .backing_generation
            .get()
            .checked_add(1)
            .ok_or(GraphError::IdentitySpaceExhausted)?;
        node.backing_generation = BackingGeneration::new(next);
        Ok(node.backing_generation)
    }

    pub fn prepare(
        &mut self,
        id: AnyResourceId,
        submission: SubmissionId,
    ) -> Result<(), GraphError> {
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        node.in_flight.insert(submission);
        if node.lifecycle != LifecycleState::Released {
            node.lifecycle = LifecycleState::Prepared;
        }
        Ok(())
    }

    pub fn submit(
        &mut self,
        id: AnyResourceId,
        submission: SubmissionId,
    ) -> Result<(), GraphError> {
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if !node.in_flight.contains(&submission) {
            return Err(GraphError::SubmissionNotPrepared);
        }
        if node.lifecycle != LifecycleState::Released {
            node.lifecycle = LifecycleState::InFlight;
        }
        Ok(())
    }

    pub fn complete(
        &mut self,
        id: AnyResourceId,
        submission: SubmissionId,
    ) -> Result<(), GraphError> {
        let node = self
            .resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?;
        if !node.in_flight.remove(&submission) {
            return Err(GraphError::SubmissionNotPrepared);
        }
        if node.lifecycle != LifecycleState::Released {
            node.lifecycle = if node.in_flight.is_empty() {
                LifecycleState::Created
            } else {
                LifecycleState::InFlight
            };
        }
        self.collect_if_unowned(id);
        Ok(())
    }

    pub fn release_reference(
        &mut self,
        task: TaskId,
        object: ObjectRef<ResourceObject>,
    ) -> Result<AnyResourceId, GraphError> {
        let id = self
            .slots
            .get_mut(&(task, object))
            .and_then(|slot| slot.current.take())
            .ok_or(GraphError::ReferenceUnbound)?;
        self.resources
            .get_mut(&id)
            .ok_or(GraphError::ResourceAbsent)?
            .lifecycle = LifecycleState::Released;
        self.collect_if_unowned(id);
        Ok(id)
    }

    pub fn release_task(&mut self, task: TaskId) -> usize {
        let refs: Vec<_> = self
            .slots
            .iter()
            .filter_map(|(&(owner, object), slot)| {
                (owner == task && slot.current.is_some()).then_some(object)
            })
            .collect();
        for object in &refs {
            let _ = self.release_reference(task, *object);
        }
        refs.len()
    }

    pub fn create_mapping(&mut self, mapping: MappingNode) -> Result<(), GraphError> {
        if mapping
            .storage
            .is_some_and(|id| !self.storage.contains_key(&id))
        {
            return Err(GraphError::StorageAbsent);
        }
        if self.mappings.insert(mapping.id, mapping).is_some() {
            return Err(GraphError::MappingAlreadyExists);
        }
        Ok(())
    }

    pub fn release_mapping(&mut self, id: MappingId) -> Result<MappingNode, GraphError> {
        self.mappings.remove(&id).ok_or(GraphError::MappingAbsent)
    }

    pub fn mapping(&self, id: MappingId) -> Option<&MappingNode> {
        self.mappings.get(&id)
    }

    fn collect_if_unowned(&mut self, id: AnyResourceId) {
        let collect = self.resources.get(&id).is_some_and(|node| {
            node.lifecycle == LifecycleState::Released
                && node.in_flight.is_empty()
                && node.children.is_empty()
        });
        if !collect {
            return;
        }
        let node = self
            .resources
            .remove(&id)
            .expect("collection candidate exists");
        if let Some(storage) = node.storage {
            if let Some(storage) = self.storage.get_mut(&storage) {
                storage.owners.remove(&id);
            }
        }
        for parent in node.parents {
            if let Some(parent_node) = self.resources.get_mut(&parent) {
                parent_node.children.remove(&id);
            }
            self.collect_if_unowned(parent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> TaskId {
        TaskId::new(3)
    }

    fn object(value: u32) -> ObjectRef<ResourceObject> {
        ObjectRef::new(value)
    }

    #[test]
    fn released_reference_reuse_gets_a_new_generation_and_rejects_stale_ids() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(7), ObjectKind::Buffer, None, [])
            .unwrap();
        graph.release_reference(task(), object(7)).unwrap();
        let second = graph
            .create_resource(task(), object(7), ObjectKind::Buffer, None, [])
            .unwrap();

        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());
        assert!(graph.resource(first).is_none());
        assert_eq!(graph.resolve(task(), object(7)), Some(second));
    }

    #[test]
    fn deleting_a_parent_keeps_it_until_its_child_view_is_released() {
        let mut graph = ResourceGraph::default();
        let parent = graph
            .create_resource(task(), object(1), ObjectKind::Texture, None, [])
            .unwrap();
        let child = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [parent])
            .unwrap();

        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(parent).is_some());
        assert_eq!(graph.resolve(task(), object(2)), Some(child));

        graph.release_reference(task(), object(2)).unwrap();
        assert!(graph.resource(child).is_none());
        assert!(graph.resource(parent).is_none());
    }

    #[test]
    fn delete_while_in_flight_retires_only_after_matching_completion() {
        let mut graph = ResourceGraph::default();
        let id = graph
            .create_resource(task(), object(1), ObjectKind::Texture, None, [])
            .unwrap();
        let submission = SubmissionId::new(9);
        graph.prepare(id, submission).unwrap();
        graph.submit(id, submission).unwrap();
        graph.release_reference(task(), object(1)).unwrap();

        assert_eq!(
            graph.resource(id).unwrap().lifecycle,
            LifecycleState::Released
        );
        graph.complete(id, submission).unwrap();
        assert!(graph.resource(id).is_none());
    }

    #[test]
    fn backing_replacement_preserves_resource_identity_and_mapping_release_is_independent() {
        let mut graph = ResourceGraph::default();
        let first_storage = graph.create_storage(StorageBacking::Dedicated).unwrap();
        let second_storage = graph.create_storage(StorageBacking::Dedicated).unwrap();
        let id = graph
            .create_resource(
                task(),
                object(1),
                ObjectKind::IOSurfaceTexture,
                Some(first_storage),
                [],
            )
            .unwrap();
        let mapping = MappingId::new(4);
        graph
            .create_mapping(MappingNode {
                id: mapping,
                task: task(),
                address: GuestVirtualAddress::new(0x4000),
                length: ByteLength::new(0x1000),
                storage: Some(first_storage),
                committed: true,
            })
            .unwrap();

        assert_eq!(graph.replace_backing(id, second_storage).unwrap().get(), 2);
        assert_eq!(graph.resource(id).unwrap().id, id);
        graph.release_mapping(mapping).unwrap();
        assert!(graph.resource(id).is_some());
    }
}
