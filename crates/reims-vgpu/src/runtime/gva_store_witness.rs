//! Contract-owned currency for copied-out GVA render targets.
//!
//! A named GVA target is a task-local resource. Its `clear_host_valid` validity
//! operation is the guest's notification that CPU writes superseded a host
//! copy, and [`crate::runtime::buffer_write_gen`] retains that notification in
//! the resource's native `(task, object)` namespace. Device writes are tracked
//! separately by [`crate::runtime::host_writes`] because different resource
//! names may alias the same guest pages.
//!
//! A Store records both generations beside the target's exact page footprint.
//! A resident may stand in for those pages only while both remain unchanged.
//! Anonymous GVA targets have no serialized resource lifetime or validity
//! record, so they acquire no entry and conservatively miss this shortcut.

use crate::model::DeviceState;
use crate::runtime::buffer_write_gen::ResourceWriteStamp;
use crate::runtime::host_writes::HostWriteVerdict;
use std::collections::BTreeMap;

/// The serialized resource and engine identity a Store published.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GvaTargetKey {
    pub task_id: u32,
    pub texture_ref: u32,
    pub gva: u64,
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: bool,
}

impl GvaTargetKey {
    /// Build a key only when the protocol supplies a resource identity.
    #[cfg(feature = "backend-vulkan")]
    pub fn of(
        task_id: u32,
        texture_ref: u32,
        identity: &crate::backend::vulkan::engine::TargetIdentity,
    ) -> Option<Self> {
        match *identity {
            crate::backend::vulkan::engine::TargetIdentity::Gva {
                gva,
                width,
                height,
                generation,
                format: _,
            } if texture_ref != 0 && generation != 0 && gva != 0 => Some(Self {
                task_id,
                texture_ref,
                gva,
                generation,
                width,
                height,
                bgra: identity.is_bgra(),
            }),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct Entry {
    gpas: Vec<u64>,
    guest_write: ResourceWriteStamp,
    host_epoch_at_store: u64,
}

/// One entry per live named target. Task teardown is the lifetime boundary.
#[derive(Default, Debug)]
pub struct GvaStoreWitness {
    entries: BTreeMap<GvaTargetKey, Entry>,
}

impl GvaStoreWitness {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn retire_task(&mut self, task_id: u32) {
        self.entries.retain(|key, _| key.task_id != task_id);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Stamp a Store after its own page writes have been recorded.
pub fn note_store(state: &mut DeviceState, key: GvaTargetKey, gpas: &[u64]) {
    if key.texture_ref == 0 || key.generation == 0 || gpas.is_empty() {
        crate::runtime::drain::note_store_route("gvaw_unnamed_resource");
        return;
    }
    let entry = Entry {
        gpas: gpas.to_vec(),
        guest_write: state.resource_write_stamp(key.task_id, key.texture_ref),
        host_epoch_at_store: state.host_writes.epoch(),
    };
    state.gva_store_witness.entries.insert(key, entry);
    crate::runtime::drain::note_store_route("gvaw_stamped");
}

/// Retire targets whose physical footprint is no longer owned by the task.
pub fn retire_pages(state: &mut DeviceState, gone: &[u64]) {
    if gone.is_empty() {
        return;
    }
    state
        .gva_store_witness
        .entries
        .retain(|_, entry| !entry.gpas.iter().any(|page| gone.contains(page)));
}

/// Why a copied resident cannot stand in for its guest pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaWriteReach {
    Quiet,
    NoEntry,
    GuestWrote,
    Host(HostWriteVerdict),
}

impl GvaWriteReach {
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gvaw_quiet",
            Self::NoEntry => "gvaw_no_entry",
            Self::GuestWrote => "gvaw_guest_wrote",
            Self::Host(HostWriteVerdict::Quiet) => "gvaw_host_quiet",
            Self::Host(HostWriteVerdict::Overlap) => "gvaw_host_overlap",
            Self::Host(HostWriteVerdict::Unnamed) => "gvaw_host_unnamed",
        }
    }

    pub fn is_quiet(self) -> bool {
        matches!(self, Self::Quiet)
    }
}

/// Compare the Store stamp with the guest's decoded validity statements and
/// this device's exact page writes.
pub fn reach(state: &DeviceState, key: GvaTargetKey) -> GvaWriteReach {
    let Some(entry) = state.gva_store_witness.entries.get(&key) else {
        return GvaWriteReach::NoEntry;
    };
    let now = state.resource_write_stamp(key.task_id, key.texture_ref);
    if !now.quiet_since(entry.guest_write) {
        return GvaWriteReach::GuestWrote;
    }
    match state
        .host_writes
        .wrote_any_since(entry.host_epoch_at_store, &entry.gpas)
    {
        HostWriteVerdict::Quiet => GvaWriteReach::Quiet,
        other => GvaWriteReach::Host(other),
    }
}

pub fn note_host_reach(state: &DeviceState, key: GvaTargetKey) {
    let Some(entry) = state.gva_store_witness.entries.get(&key) else {
        return;
    };
    let distance = state
        .host_writes
        .epoch()
        .saturating_sub(entry.host_epoch_at_store);
    crate::runtime::drain::note_store_route(if distance < 64 {
        "gvaw_reach_lt64"
    } else if distance < 512 {
        "gvaw_reach_lt512"
    } else if distance < 4096 {
        "gvaw_reach_lt4k"
    } else {
        "gvaw_reach_ge4k"
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    const PAGE: u64 = 1 << PAGE_SHIFT_X86;

    fn state() -> DeviceState {
        DeviceState::new(DeviceId::default(), PAGE_SHIFT_X86)
    }

    fn key(task_id: u32, texture_ref: u32) -> GvaTargetKey {
        GvaTargetKey {
            task_id,
            texture_ref,
            gva: PAGE,
            generation: 9,
            width: 16,
            height: 16,
            bgra: true,
        }
    }

    #[test]
    fn decoded_guest_write_invalidates_only_its_resource() {
        let mut state = state();
        let a = key(1, 7);
        let b = key(1, 8);
        note_store(&mut state, a, &[PAGE]);
        note_store(&mut state, b, &[2 * PAGE]);
        state.buffer_write_gen.note_write(1, 7);
        assert_eq!(reach(&state, a), GvaWriteReach::GuestWrote);
        assert_eq!(reach(&state, b), GvaWriteReach::Quiet);
    }

    #[test]
    fn device_write_invalidates_every_alias_of_its_page() {
        let mut state = state();
        let a = key(1, 7);
        let b = key(2, 8);
        note_store(&mut state, a, &[PAGE]);
        note_store(&mut state, b, &[PAGE]);
        state.note_host_wrote_pages(vec![PAGE]);
        assert_eq!(
            reach(&state, a),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
        assert_eq!(
            reach(&state, b),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
    }

    #[test]
    fn anonymous_or_unstamped_target_never_reads_quiet() {
        let mut state = state();
        let anonymous = key(1, 0);
        note_store(&mut state, anonymous, &[PAGE]);
        assert_eq!(state.gva_store_witness.len(), 0);
        assert_eq!(reach(&state, anonymous), GvaWriteReach::NoEntry);
    }

    #[test]
    fn task_retirement_ends_witness_lifetime() {
        let mut state = state();
        let target = key(4, 12);
        note_store(&mut state, target, &[PAGE]);
        state.gva_store_witness.retire_task(4);
        assert_eq!(reach(&state, target), GvaWriteReach::NoEntry);
    }
}
