//! Canonical task-owned resource, storage, and mapping graph.

use reims_vgpu_protocol::{
    BackingGeneration, ByteLength, ByteOffset, ContentVersion, GuestVirtualAddress,
    MapperSurfaceRef, MappingId, ObjectKind, ObjectTableRef, PlaneIndex, ResourceId,
    ResourceObject, StorageId, SubmissionId, SurfaceBackingId, TaskId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{atomic::AtomicU64, Weak};
use std::sync::{Arc, Mutex};

type AnyResourceId = ResourceId<ResourceObject>;

static NEXT_RESOURCE_LIFETIME: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct ResourceLifetimeToken {
    id: u64,
}

/// Strong ownership token for one constructed semantic resource lifetime.
///
/// Backend caches receive only [`ResourceLifetimeRef`]. Entries therefore die
/// with the guest-owned resource rather than an invented capacity or timer.
#[derive(Debug)]
pub struct ResourceLifetime(Arc<ResourceLifetimeToken>);

impl Default for ResourceLifetime {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceLifetime {
    pub fn new() -> Self {
        let id = NEXT_RESOURCE_LIFETIME
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |id| id.checked_add(1),
            )
            .expect("resource lifetime identity exhausted");
        Self(Arc::new(ResourceLifetimeToken { id }))
    }

    pub fn reference(&self) -> ResourceLifetimeRef {
        ResourceLifetimeRef {
            id: self.0.id,
            live: Arc::downgrade(&self.0),
        }
    }
}

/// Weak executor-facing proof that one semantic resource still exists.
#[derive(Clone, Debug)]
pub struct ResourceLifetimeRef {
    id: u64,
    live: Weak<ResourceLifetimeToken>,
}

impl ResourceLifetimeRef {
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn is_live(&self) -> bool {
        self.live.strong_count() != 0
    }
}

/// Exact semantic content represented by one executor operand or effect.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ContentStamp {
    pub resource: ResourceId<ResourceObject>,
    pub version: ContentVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleState {
    Created,
    Prepared,
    InFlight,
    Released,
}

/// Versions held by the three places resource bytes can reside.
///
/// `None` means that replica does not contain bytes from this resource's
/// current lifetime. Currency is always derived by comparing a replica with
/// [`ContentState::current`]; there is no independent "GPU only" latch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicaVersions {
    pub guest: Option<ContentVersion>,
    pub gpu: Option<ContentVersion>,
    pub host: Option<ContentVersion>,
}

/// A version reserved for a submitted GPU write which has not completed yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingContentWrite {
    pub submission: SubmissionId,
    pub version: ContentVersion,
}

/// The sole authority for which replica contains a resource's newest bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentState {
    pub current: ContentVersion,
    pub replicas: ReplicaVersions,
    pending_gpu_writes: BTreeMap<SubmissionId, ContentVersion>,
    next_version: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentError {
    SubmissionAlreadyWrites,
    SubmissionDidNotPlanWrite,
    StaleSource,
    WouldLoseCurrentContent,
    VersionSpaceExhausted,
}

impl Default for ContentState {
    fn default() -> Self {
        let initial = ContentVersion::new(1);
        Self {
            current: initial,
            replicas: ReplicaVersions {
                guest: Some(initial),
                gpu: None,
                host: None,
            },
            pending_gpu_writes: BTreeMap::new(),
            next_version: 2,
        }
    }
}

impl ContentState {
    fn reserve_version(&mut self) -> Result<ContentVersion, ContentError> {
        let version = ContentVersion::new(self.next_version);
        self.next_version = self
            .next_version
            .checked_add(1)
            .ok_or(ContentError::VersionSpaceExhausted)?;
        Ok(version)
    }

    pub fn guest_wrote(&mut self) -> Result<ContentVersion, ContentError> {
        let version = self.reserve_version()?;
        self.current = version;
        self.replicas.guest = Some(version);
        Ok(version)
    }

    pub fn gpu_store_planned(
        &mut self,
        submission: SubmissionId,
    ) -> Result<PendingContentWrite, ContentError> {
        if self.pending_gpu_writes.contains_key(&submission) {
            return Err(ContentError::SubmissionAlreadyWrites);
        }
        let version = self.reserve_version()?;
        self.pending_gpu_writes.insert(submission, version);
        Ok(PendingContentWrite {
            submission,
            version,
        })
    }

    pub fn gpu_store_completed(
        &mut self,
        submission: SubmissionId,
    ) -> Result<ContentVersion, ContentError> {
        let version = self
            .pending_gpu_writes
            .remove(&submission)
            .ok_or(ContentError::SubmissionDidNotPlanWrite)?;
        self.replicas.gpu = Some(version);
        if version > self.current {
            self.current = version;
        }
        Ok(version)
    }

    pub fn copy_gpu_to_guest_completed(
        &mut self,
        version: ContentVersion,
    ) -> Result<(), ContentError> {
        if self.replicas.gpu != Some(version) || self.current != version {
            return Err(ContentError::StaleSource);
        }
        self.replicas.guest = Some(version);
        Ok(())
    }

    pub fn copy_guest_to_gpu_completed(
        &mut self,
        version: ContentVersion,
    ) -> Result<(), ContentError> {
        if self.replicas.guest != Some(version) || self.current != version {
            return Err(ContentError::StaleSource);
        }
        self.replicas.gpu = Some(version);
        Ok(())
    }

    /// Record that an executor materialized this exact current version.
    ///
    /// Unlike [`Self::copy_guest_to_gpu_completed`], the source replica is
    /// already established by the stamped operand.  The version check is the
    /// authority: a completion for content superseded since submission cannot
    /// make the stale GPU replica current.
    pub fn gpu_materialized(&mut self, version: ContentVersion) -> Result<(), ContentError> {
        if self.current != version {
            return Err(ContentError::StaleSource);
        }
        self.replicas.gpu = Some(version);
        Ok(())
    }

    pub fn current_in_guest(&self) -> bool {
        self.replicas.guest == Some(self.current)
    }

    pub fn current_in_gpu(&self) -> bool {
        self.replicas.gpu == Some(self.current)
    }

    pub fn replace_guest_backing(&mut self) -> Result<(), ContentError> {
        if self.current_in_guest()
            && self.replicas.gpu != Some(self.current)
            && self.replicas.host != Some(self.current)
        {
            return Err(ContentError::WouldLoseCurrentContent);
        }
        self.replicas.guest = None;
        Ok(())
    }
}

/// Shared authority for every resource view over one storage object.
///
/// A resource constructed before its backing is known starts with a private
/// authority. Attaching storage replaces that handle with the storage-owned
/// one; every later alias therefore observes and mutates the same version
/// state instead of maintaining counters which merely happen to agree.
#[derive(Clone, Debug)]
pub struct ContentAuthority(Arc<Mutex<ContentState>>);

impl Default for ContentAuthority {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(ContentState::default())))
    }
}

impl PartialEq for ContentAuthority {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}

impl Eq for ContentAuthority {}

impl ContentAuthority {
    pub fn snapshot(&self) -> ContentState {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn same_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn current(&self) -> ContentVersion {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .current
    }

    pub fn guest_wrote(&self) -> Result<ContentVersion, ContentError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .guest_wrote()
    }

    pub fn gpu_store_planned(
        &self,
        submission: SubmissionId,
    ) -> Result<PendingContentWrite, ContentError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .gpu_store_planned(submission)
    }

    pub fn gpu_store_completed(
        &self,
        submission: SubmissionId,
    ) -> Result<ContentVersion, ContentError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .gpu_store_completed(submission)
    }

    pub fn copy_gpu_to_guest_completed(&self, version: ContentVersion) -> Result<(), ContentError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .copy_gpu_to_guest_completed(version)
    }

    pub fn gpu_materialized(&self, version: ContentVersion) -> Result<(), ContentError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .gpu_materialized(version)
    }
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
        bytes_per_row: ByteLength,
    },
    IOSurfacePlane {
        surface: SurfaceBackingId,
        plane: PlaneIndex,
    },
    RegisteredSurface {
        surface: SurfaceBackingId,
    },
    /// Mapper-path surface storage whose shared-backing identity is not yet
    /// established independently of the mapping object.
    MapperSurface {
        mapper: MapperSurfaceRef,
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
    pub content: ContentAuthority,
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
    pub object: ObjectTableRef<ResourceObject>,
    pub kind: ObjectKind,
    pub lifecycle: LifecycleState,
    pub storage: Option<StorageId>,
    pub backing_generation: BackingGeneration,
    pub content: ContentAuthority,
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
    ParentCycle,
    StorageAbsent,
    StorageConflict,
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
    slots: BTreeMap<(TaskId, ObjectTableRef<ResourceObject>), NamespaceSlot>,
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
    fn parent_edge_would_cycle(&self, parent: AnyResourceId, child: AnyResourceId) -> bool {
        if parent == child {
            return true;
        }
        let mut pending = vec![child];
        let mut visited = BTreeSet::new();
        while let Some(candidate) = pending.pop() {
            if !visited.insert(candidate) {
                continue;
            }
            if candidate == parent {
                return true;
            }
            if let Some(resource) = self.resources.get(&candidate) {
                pending.extend(resource.children.iter().copied());
            }
        }
        false
    }

    pub fn create_resource(
        &mut self,
        task: TaskId,
        object: ObjectTableRef<ResourceObject>,
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
                content: storage
                    .and_then(|storage| self.storage.get(&storage))
                    .map(|storage| storage.content.clone())
                    .unwrap_or_default(),
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
        object: ObjectTableRef<ResourceObject>,
    ) -> Option<AnyResourceId> {
        self.slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
    }

    pub fn resource(&self, id: AnyResourceId) -> Option<&ResourceNode> {
        self.resources.get(&id)
    }

    pub fn resource_mut(&mut self, id: AnyResourceId) -> Option<&mut ResourceNode> {
        self.resources.get_mut(&id)
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
                content: ContentAuthority::default(),
            },
        );
        Ok(id)
    }

    pub fn storage(&self, id: StorageId) -> Option<&StorageNode> {
        self.storage.get(&id)
    }

    pub fn mapper_storage(
        &mut self,
        mapper: MapperSurfaceRef,
        plane: PlaneIndex,
    ) -> Result<StorageId, GraphError> {
        if let Some(id) = self.storage.values().find_map(|storage| {
            (storage.backing == StorageBacking::MapperSurface { mapper, plane })
                .then_some(storage.id)
        }) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::MapperSurface { mapper, plane })
    }

    pub fn registered_surface_storage(
        &mut self,
        surface: SurfaceBackingId,
    ) -> Result<StorageId, GraphError> {
        if let Some(id) = self.storage.values().find_map(|storage| {
            (storage.backing == StorageBacking::RegisteredSurface { surface }).then_some(storage.id)
        }) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::RegisteredSurface { surface })
    }

    pub fn task_address_storage(
        &mut self,
        task: TaskId,
        address: GuestVirtualAddress,
        length: ByteLength,
    ) -> Result<StorageId, GraphError> {
        if let Some(id) = self.storage.values().find_map(|storage| {
            (storage.backing
                == StorageBacking::TaskAddress {
                    task,
                    address,
                    length,
                })
            .then_some(storage.id)
        }) {
            return Ok(id);
        }
        self.create_storage(StorageBacking::TaskAddress {
            task,
            address,
            length,
        })
    }

    pub fn link_parent(
        &mut self,
        child: AnyResourceId,
        parent: AnyResourceId,
    ) -> Result<(), GraphError> {
        let parent_storage = self
            .resources
            .get(&parent)
            .ok_or(GraphError::ParentAbsent)?
            .storage;
        let child_storage = self
            .resources
            .get(&child)
            .ok_or(GraphError::ResourceAbsent)?
            .storage;
        if self.parent_edge_would_cycle(parent, child) {
            return Err(GraphError::ParentCycle);
        }
        if matches!((child_storage, parent_storage), (Some(child), Some(parent)) if child != parent)
        {
            return Err(GraphError::StorageConflict);
        }
        self.resources
            .get_mut(&parent)
            .unwrap()
            .children
            .insert(child);
        let child_node = self.resources.get_mut(&child).unwrap();
        child_node.parents.insert(parent);
        if child_node.storage.is_none() {
            if let Some(storage) = parent_storage {
                child_node.storage = Some(storage);
                let storage_node = self
                    .storage
                    .get_mut(&storage)
                    .ok_or(GraphError::StorageAbsent)?;
                child_node.content = storage_node.content.clone();
                storage_node.owners.insert(child);
            }
        }
        Ok(())
    }

    pub fn link_buffer_range(
        &mut self,
        child: AnyResourceId,
        buffer: AnyResourceId,
        offset: ByteOffset,
        bytes_per_row: ByteLength,
    ) -> Result<(), GraphError> {
        if !self.resources.contains_key(&buffer) {
            return Err(GraphError::ParentAbsent);
        }
        let child_storage = self
            .resources
            .get(&child)
            .ok_or(GraphError::ResourceAbsent)?
            .storage;
        if self.parent_edge_would_cycle(buffer, child) {
            return Err(GraphError::ParentCycle);
        }
        let backing = StorageBacking::BufferRange {
            buffer,
            offset,
            bytes_per_row,
        };
        let existing = self
            .storage
            .values()
            .find_map(|storage| (storage.backing == backing).then_some(storage.id));
        if child_storage.is_some_and(|storage| Some(storage) != existing) {
            return Err(GraphError::StorageConflict);
        }
        let storage = match existing {
            Some(storage) => storage,
            None => self.create_storage(backing)?,
        };
        self.attach_initial_storage(child, storage)?;
        self.resources
            .get_mut(&buffer)
            .expect("parent validated")
            .children
            .insert(child);
        self.resources
            .get_mut(&child)
            .expect("child validated")
            .parents
            .insert(buffer);
        Ok(())
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
        if let Some(existing) = node.storage {
            return if existing == storage {
                Ok(())
            } else {
                Err(GraphError::StorageConflict)
            };
        }
        node.storage = Some(storage);
        let storage_node = self.storage.get_mut(&storage).unwrap();
        node.content = storage_node.content.clone();
        storage_node.owners.insert(id);
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
        node.content = self.storage.get(&storage).unwrap().content.clone();
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
        object: ObjectTableRef<ResourceObject>,
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

    fn object(value: u32) -> ObjectTableRef<ResourceObject> {
        ObjectTableRef::new(value)
    }

    #[test]
    fn mapper_references_do_not_become_page_table_mapping_identities() {
        let mut graph = ResourceGraph::default();
        let mapper = MapperSurfaceRef::new(12);
        let storage = graph.mapper_storage(mapper, PlaneIndex::new(0)).unwrap();

        assert_eq!(
            graph.storage(storage).unwrap().backing,
            StorageBacking::MapperSurface {
                mapper,
                plane: PlaneIndex::new(0),
            }
        );
        assert!(graph.mapping(MappingId::new(12)).is_none());
    }

    #[test]
    fn delayed_gpu_completion_cannot_overwrite_a_newer_guest_write() {
        let mut content = ContentState::default();
        let planned = content.gpu_store_planned(SubmissionId::new(4)).unwrap();
        let guest = content.guest_wrote().unwrap();

        assert!(guest > planned.version);
        assert_eq!(
            content.gpu_store_completed(SubmissionId::new(4)),
            Ok(planned.version)
        );
        assert_eq!(content.current, guest);
        assert!(content.current_in_guest());
        assert!(!content.current_in_gpu());
        assert_eq!(
            content.copy_gpu_to_guest_completed(planned.version),
            Err(ContentError::StaleSource)
        );
    }

    #[test]
    fn gpu_only_content_is_derived_from_replica_versions() {
        let mut content = ContentState::default();
        let planned = content.gpu_store_planned(SubmissionId::new(9)).unwrap();
        content.gpu_store_completed(SubmissionId::new(9)).unwrap();

        assert_eq!(content.current, planned.version);
        assert!(content.current_in_gpu());
        assert!(!content.current_in_guest());
        content
            .copy_gpu_to_guest_completed(planned.version)
            .unwrap();
        assert!(content.current_in_guest());
    }

    #[test]
    fn a_stamped_materialization_cannot_revive_superseded_content() {
        let mut content = ContentState::default();
        let submitted = content.current;
        let newer = content.guest_wrote().unwrap();

        assert_eq!(
            content.gpu_materialized(submitted),
            Err(ContentError::StaleSource)
        );
        assert_eq!(content.current, newer);
        assert!(!content.current_in_gpu());
        assert_eq!(content.gpu_materialized(newer), Ok(()));
        assert!(content.current_in_gpu());
    }

    #[test]
    fn replacing_the_only_current_replica_is_refused() {
        let mut content = ContentState::default();
        assert_eq!(
            content.replace_guest_backing(),
            Err(ContentError::WouldLoseCurrentContent)
        );

        let current = content.current;
        content.copy_guest_to_gpu_completed(current).unwrap();
        content.replace_guest_backing().unwrap();
        assert!(!content.current_in_guest());
        assert!(content.current_in_gpu());
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

    #[test]
    fn a_registered_surface_and_its_view_share_one_storage_identity() {
        let mut graph = ResourceGraph::default();
        let storage = graph
            .registered_surface_storage(SurfaceBackingId::new(44))
            .unwrap();
        let surface = graph
            .create_resource(
                task(),
                object(1),
                ObjectKind::SurfaceBacking,
                Some(storage),
                [],
            )
            .unwrap();
        let view = graph
            .create_resource(task(), object(2), ObjectKind::IOSurfacePlaneView, None, [])
            .unwrap();

        graph.link_parent(view, surface).unwrap();

        assert_eq!(graph.resource(view).unwrap().storage, Some(storage));
        assert_eq!(graph.storage(storage).unwrap().owners.len(), 2);
        let surface_content = graph.resource(surface).unwrap().content.clone();
        let view_content = graph.resource(view).unwrap().content.clone();
        assert!(surface_content.same_authority(&view_content));
        let before = view_content.current();
        surface_content.guest_wrote().unwrap();
        assert_ne!(view_content.current(), before);
        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(surface).is_some());
    }

    #[test]
    fn resources_over_the_same_task_allocation_share_storage_identity() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x2000),
            )
            .unwrap();
        let second = graph
            .task_address_storage(
                task(),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x2000),
            )
            .unwrap();
        let other_task = graph
            .task_address_storage(
                TaskId::new(8),
                GuestVirtualAddress::new(0x4000),
                ByteLength::new(0x2000),
            )
            .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, other_task);
    }

    #[test]
    fn a_buffer_texture_owns_a_typed_range_and_retains_its_buffer() {
        let mut graph = ResourceGraph::default();
        let buffer = graph
            .create_resource(task(), object(1), ObjectKind::Buffer, None, [])
            .unwrap();
        let texture = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();

        graph
            .link_buffer_range(texture, buffer, ByteOffset::new(96), ByteLength::new(512))
            .unwrap();

        let texture_node = graph.resource(texture).unwrap();
        assert!(texture_node.parents.contains(&buffer));
        assert_eq!(
            graph
                .storage(texture_node.storage.unwrap())
                .unwrap()
                .backing,
            StorageBacking::BufferRange {
                buffer,
                offset: ByteOffset::new(96),
                bytes_per_row: ByteLength::new(512),
            }
        );
        graph.release_reference(task(), object(1)).unwrap();
        assert!(graph.resource(buffer).is_some());
        graph.release_reference(task(), object(2)).unwrap();
        assert!(graph.resource(buffer).is_none());
    }

    #[test]
    fn parent_relations_cannot_form_a_retention_cycle() {
        let mut graph = ResourceGraph::default();
        let first = graph
            .create_resource(task(), object(1), ObjectKind::TextureView, None, [])
            .unwrap();
        let second = graph
            .create_resource(task(), object(2), ObjectKind::TextureView, None, [])
            .unwrap();

        graph.link_parent(second, first).unwrap();
        assert_eq!(
            graph.link_parent(first, second),
            Err(GraphError::ParentCycle)
        );
        assert!(!graph.resource(first).unwrap().parents.contains(&second));
        assert!(!graph.resource(second).unwrap().children.contains(&first));
    }

    #[test]
    fn cache_lifetime_refs_expire_only_with_the_owning_resource() {
        let lifetime = ResourceLifetime::new();
        let reference = lifetime.reference();
        let sibling = ResourceLifetime::new();

        assert!(reference.is_live());
        assert_ne!(reference.id(), sibling.reference().id());
        drop(sibling);
        assert!(
            reference.is_live(),
            "an unrelated resource cannot retire it"
        );
        drop(lifetime);
        assert!(!reference.is_live());
    }
}
