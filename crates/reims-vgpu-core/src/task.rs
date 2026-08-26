//! Guest task-directory lifetime and address-space roots.

use std::collections::BTreeMap;

use reims_vgpu_protocol::{ObjectTableRef, ResourceObject, TaskId};

/// Exact task-GVA read required to fetch one published object-list entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectListEntryReadPlan {
    pub task: TaskId,
    pub object: ObjectTableRef<ResourceObject>,
    pub page_shift: u32,
    pub list_pfn: u32,
    pub list_count: u32,
    pub gva: u64,
    pub byte_len: u64,
}

/// Typed refusal produced before object-list guest memory is accessed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectListEntryReadError {
    TaskUnavailable(TaskId),
    ObjectListUnpublished(TaskId),
    ReferencePastList {
        object: ObjectTableRef<ResourceObject>,
        count: u32,
    },
    ListBaseOverflow {
        pfn: u32,
        page_shift: u32,
    },
    EntryAddressOverflow,
    EntryEndOverflow,
}

/// Derive the exact object-list entry window from the task-owned publication.
///
/// The list PFN is a task virtual page number. The object reference is its
/// zero-based slot and the wire entry width is owned by the protocol crate.
/// Descriptor contents and descriptor memory are deliberately a later read:
/// neither can be trusted until this fixed entry has been fetched and decoded.
pub fn resolve_object_list_entry_read(
    tasks: &TaskTable,
    task: TaskId,
    object: ObjectTableRef<ResourceObject>,
    page_shift: u32,
) -> Result<ObjectListEntryReadPlan, ObjectListEntryReadError> {
    let entry = tasks
        .get(task.get())
        .filter(|entry| entry.active)
        .ok_or(ObjectListEntryReadError::TaskUnavailable(task))?;
    if entry.object_list_count == 0 {
        return Err(ObjectListEntryReadError::ObjectListUnpublished(task));
    }
    if object.get() >= entry.object_list_count {
        return Err(ObjectListEntryReadError::ReferencePastList {
            object,
            count: entry.object_list_count,
        });
    }
    let page_size =
        1u64.checked_shl(page_shift)
            .ok_or(ObjectListEntryReadError::ListBaseOverflow {
                pfn: entry.object_list_pfn,
                page_shift,
            })?;
    let base = u64::from(entry.object_list_pfn)
        .checked_mul(page_size)
        .ok_or(ObjectListEntryReadError::ListBaseOverflow {
            pfn: entry.object_list_pfn,
            page_shift,
        })?;
    let offset = u64::from(object.get())
        .checked_mul(reims_vgpu_protocol::OBJECT_LIST_ENTRY_LEN as u64)
        .ok_or(ObjectListEntryReadError::EntryAddressOverflow)?;
    let gva = base
        .checked_add(offset)
        .ok_or(ObjectListEntryReadError::EntryAddressOverflow)?;
    gva.checked_add(reims_vgpu_protocol::OBJECT_LIST_ENTRY_LEN as u64 - 1)
        .ok_or(ObjectListEntryReadError::EntryEndOverflow)?;
    Ok(ObjectListEntryReadPlan {
        task,
        object,
        page_shift,
        list_pfn: entry.object_list_pfn,
        list_count: entry.object_list_count,
        gva,
        byte_len: reims_vgpu_protocol::OBJECT_LIST_ENTRY_LEN as u64,
    })
}

/// One guest task directory and its optional object-list publication.
#[derive(Clone, Debug, Default)]
pub struct TaskEntry {
    pub active: bool,
    pub length: u64,
    pub directory_pfn: u32,
    pub object_list_pfn: u32,
    pub object_list_count: u32,
}

impl TaskEntry {
    /// A task the guest has defined but not yet given an object list.
    ///
    /// Object-list fields remain zero until the distinct object-list command
    /// publishes them; task definition does not invent a guest page or count.
    pub fn define(length: u64, directory_pfn: u32) -> Self {
        Self {
            active: true,
            length,
            directory_pfn,
            object_list_pfn: 0,
            object_list_count: 0,
        }
    }
}

/// Live tasks keyed by the guest's complete `u32` task namespace.
///
/// There is no host-selected capacity. Entries live and die with guest task
/// definition/deletion, and iteration preserves ascending task-id order.
#[derive(Clone, Debug, Default)]
pub struct TaskTable(BTreeMap<u32, TaskEntry>);

impl TaskTable {
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn get(&self, id: u32) -> Option<&TaskEntry> {
        self.0.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut TaskEntry> {
        self.0.get_mut(&id)
    }

    pub fn is_active(&self, id: u32) -> bool {
        self.get(id).is_some_and(|task| task.active)
    }

    pub fn define(&mut self, id: u32, entry: TaskEntry) {
        self.0.insert(id, entry);
    }

    pub fn remove(&mut self, id: u32) {
        self.0.remove(&id);
    }

    pub fn live(&self) -> impl Iterator<Item = (u32, &TaskEntry)> {
        self.0
            .iter()
            .filter(|(_, task)| task.active)
            .map(|(&id, task)| (id, task))
    }

    pub fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.live().map(|(id, _)| id)
    }

    pub fn live_count(&self) -> usize {
        self.live_ids().count()
    }
}

/// Fixture convenience for tests which mutate an already-defined task.
#[cfg(feature = "test-fixtures")]
impl std::ops::Index<u32> for TaskTable {
    type Output = TaskEntry;

    fn index(&self, id: u32) -> &TaskEntry {
        self.get(id)
            .unwrap_or_else(|| panic!("test indexed task {id}, which nothing defined"))
    }
}

#[cfg(feature = "test-fixtures")]
impl std::ops::IndexMut<u32> for TaskTable {
    fn index_mut(&mut self, id: u32) -> &mut TaskEntry {
        self.get_mut(id)
            .unwrap_or_else(|| panic!("test indexed task {id}, which nothing defined"))
    }
}

#[cfg(test)]
mod object_list_tests {
    use super::*;

    #[test]
    fn object_list_entry_read_uses_the_published_root_and_exact_slot_width() {
        let task = TaskId::new(7);
        let object = ObjectTableRef::new(36);
        let mut tasks = TaskTable::new();
        let mut entry = TaskEntry::define(0x1_0000_0000, 19);
        entry.object_list_pfn = 3;
        entry.object_list_count = 37;
        tasks.define(task.get(), entry);

        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, object, 14),
            Ok(ObjectListEntryReadPlan {
                task,
                object,
                page_shift: 14,
                list_pfn: 3,
                list_count: 37,
                gva: (3u64 << 14) + 36 * reims_vgpu_protocol::OBJECT_LIST_ENTRY_LEN as u64,
                byte_len: reims_vgpu_protocol::OBJECT_LIST_ENTRY_LEN as u64,
            })
        );
    }

    #[test]
    fn object_list_entry_read_refuses_before_transport_on_every_invalid_root() {
        let task = TaskId::new(9);
        let object = ObjectTableRef::new(1);
        let mut tasks = TaskTable::new();
        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, object, 12),
            Err(ObjectListEntryReadError::TaskUnavailable(task))
        );

        tasks.define(task.get(), TaskEntry::define(0x1_0000, 4));
        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, object, 12),
            Err(ObjectListEntryReadError::ObjectListUnpublished(task))
        );

        let entry = tasks.get_mut(task.get()).unwrap();
        entry.object_list_pfn = 4;
        entry.object_list_count = 1;
        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, object, 12),
            Err(ObjectListEntryReadError::ReferencePastList { object, count: 1 })
        );

        let entry = tasks.get_mut(task.get()).unwrap();
        entry.object_list_pfn = u32::MAX;
        entry.object_list_count = u32::MAX;
        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, object, 40),
            Err(ObjectListEntryReadError::ListBaseOverflow {
                pfn: u32::MAX,
                page_shift: 40,
            })
        );

        let address_overflow = ObjectTableRef::new(u32::MAX - 1);
        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, address_overflow, 32),
            Err(ObjectListEntryReadError::EntryAddressOverflow)
        );

        let end_overflow = ObjectTableRef::new(357_913_941);
        assert_eq!(
            resolve_object_list_entry_read(&tasks, task, end_overflow, 32),
            Err(ObjectListEntryReadError::EntryEndOverflow)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_width_task_ids_have_guest_owned_lifetimes() {
        let mut tasks = TaskTable::new();
        tasks.define(u32::MAX, TaskEntry::define(0x4000, 7));
        assert!(tasks.is_active(u32::MAX));
        assert_eq!(tasks.live_ids().collect::<Vec<_>>(), vec![u32::MAX]);

        tasks.remove(u32::MAX);
        assert!(!tasks.is_active(u32::MAX));
    }

    #[test]
    fn task_definition_does_not_invent_an_object_list() {
        let task = TaskEntry::define(0x8000, 9);
        assert_eq!((task.object_list_pfn, task.object_list_count), (0, 0));
    }
}
