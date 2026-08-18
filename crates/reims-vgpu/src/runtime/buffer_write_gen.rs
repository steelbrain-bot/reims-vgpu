//! Pre-construction fallback for guest-declared resource writes.
//!
//! # Why this state still exists
//!
//! [`crate::runtime::resource_validity::apply`] takes the guest's validity quad
//! for one object id. GVA render resources are owned by a task-local texture
//! reference, including resources that also resolve to a mapping. Their
//! guest-write declarations therefore need a generation keyed by
//! `(task, object)` rather than inferred from the physical backing chosen later.
//!
//! Constructed objects use the canonical resource graph's `(ResourceId,
//! ContentVersion)` through [`ResourceWriteStamp::Resolved`]. This ledger is
//! retained only for the ordering case where a validity record names a task
//! reference before its construction record has been decoded. Once the object
//! exists, this fallback can no longer decide its currency.
//!
//! # Lifetime
//!
//! One entry per `(task, object)` the guest has declared a write to. A task's
//! entries go when the task does, which is the same lifetime `bound_buffers`
//! retires on and the only announcement this device gets. There is no capacity
//! eviction: a generation is authoritative state about a live guest object, and
//! forgetting it while that object lives would make a later comparison read as
//! clean after the bytes moved.

use std::collections::HashMap;

use reims_vgpu_protocol::{ContentVersion, ResourceId, ResourceObject};

/// The resource-owned content observation recorded beside a derived copy.
///
/// A resolved stamp is generational: deleting and recreating the same guest
/// reference changes its `ResourceId`, even if its first content version has
/// the same numeric value. The unresolved variant is deliberately explicit and
/// cannot compare equal to a later resolved resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceWriteStamp {
    Resolved {
        resource: ResourceId<ResourceObject>,
        version: ContentVersion,
    },
    Unresolved(BufferWriteStamp),
}

impl Default for ResourceWriteStamp {
    fn default() -> Self {
        Self::Unresolved(BufferWriteStamp::default())
    }
}

impl ResourceWriteStamp {
    pub fn quiet_since(self, earlier: Self) -> bool {
        match (self, earlier) {
            (
                Self::Resolved {
                    resource: now_resource,
                    version: now_version,
                },
                Self::Resolved {
                    resource: old_resource,
                    version: old_version,
                },
            ) => now_resource == old_resource && now_version == old_version,
            (Self::Unresolved(now), Self::Unresolved(old)) => now.quiet_since(old),
            _ => false,
        }
    }
}

/// Per-object write generations in the guest's task-local resource namespace.
#[derive(Default, Debug)]
pub struct BufferWriteGens {
    gens: HashMap<(u32, u32), u64>,
    /// Bumped on task retirement, so a comparison spanning task-id reuse is not
    /// mistaken for a comparison that found the same generation twice. A debt
    /// stores this beside the generation and treats a change in it as unknown.
    epoch: u64,
}

impl BufferWriteGens {
    /// Record that the guest declared a write to `object_id` under `task_id`.
    ///
    /// Called for every decoded resource write statement. Mapping generations
    /// remain the corresponding authority in the mapping-id namespace; this
    /// ledger answers task-local resource references.
    pub fn note_write(&mut self, task_id: u32, object_id: u32) {
        let slot = self.gens.entry((task_id, object_id)).or_insert(0);
        *slot = slot.wrapping_add(1);
        // Keep the decoded write rate visible beside GVA debt abandonment. A
        // zero here with live GVA Stores means the task/object namespace is not
        // reaching the authority check at all.
        crate::runtime::drain::note_store_route("buffer_write_gen_bump");
    }

    /// What a reader records beside a copy it has just taken, and compares
    /// against later.
    ///
    /// The epoch travels with the generation so task-id reuse cannot be read as
    /// unchanged: an object with no entry reads `(epoch, 0)`, and after a task
    /// retirement the epoch differs from every stamp taken before it.
    pub fn stamp(&self, task_id: u32, object_id: u32) -> BufferWriteStamp {
        BufferWriteStamp {
            epoch: self.epoch,
            gen: self.gens.get(&(task_id, object_id)).copied().unwrap_or(0),
        }
    }

    /// Forget one task's objects, because the task's ids no longer name them.
    ///
    /// Retiring by task rather than by object for the reason
    /// [`crate::runtime::bound_buffers`] states about its own registry: mapping
    /// an object id back to what resolved through it is machinery bought with
    /// nothing, and task teardown is rare.
    ///
    /// This bumps the epoch because a guest reuses task ids. Drop `(5, 7)` at generation 3,
    /// let a *different* task 5 create a *different* object 7 and declare three
    /// writes to it, and a stamp taken before the retire compares equal to one
    /// taken after — same epoch, same generation, unrelated bytes. That is the
    /// one direction this whole type exists to refuse. Bumped only when an entry
    /// actually went, so retiring a task that declared no writes costs no
    /// reader's stamp.
    pub fn retire_task(&mut self, task_id: u32) {
        let before = self.gens.len();
        self.gens.retain(|&(task, _), _| task != task_id);
        if self.gens.len() != before {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    /// Entries held.
    ///
    /// Named for what it counts rather than `len`, because this is not a
    /// collection anything iterates and a `len`/`is_empty` pair would suggest it
    /// is.
    pub fn tracked(&self) -> usize {
        self.gens.len()
    }
}

/// One object's write generation as a reader saw it.
///
/// Two of these are comparable only when their epochs agree; see
/// [`BufferWriteGens::stamp`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferWriteStamp {
    epoch: u64,
    gen: u64,
}

impl BufferWriteStamp {
    /// Whether the guest has declared no write to this object between the two
    /// stamps.
    ///
    /// `false` for a pair that straddles task retirement, which is the unknown
    /// case answered in the safe direction.
    pub fn quiet_since(self, earlier: Self) -> bool {
        self.epoch == earlier.epoch && self.gen == earlier.gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generation moves only when the guest says it wrote, so a debt stays
    /// authoritative across unrelated activity.
    #[test]
    fn a_declared_write_moves_the_generation_and_nothing_else_does() {
        let mut g = BufferWriteGens::default();
        let before = g.stamp(1, 7);
        assert!(g.stamp(1, 7).quiet_since(before), "no write, no move");
        g.note_write(1, 7);
        assert!(!g.stamp(1, 7).quiet_since(before));
        let after = g.stamp(1, 7);
        assert!(g.stamp(1, 7).quiet_since(after), "still no further write");
    }

    /// A write to one object must not invalidate another's stamp, or an
    /// unrelated resource would abandon host-authoritative pixels.
    #[test]
    fn a_write_to_another_object_leaves_this_one_quiet() {
        let mut g = BufferWriteGens::default();
        let before = g.stamp(1, 7);
        g.note_write(1, 8);
        g.note_write(2, 7);
        assert!(g.stamp(1, 7).quiet_since(before));
    }

    /// Live object generations are retained without a capacity eviction.
    #[test]
    fn live_object_generations_are_not_evicted_by_capacity() {
        let mut g = BufferWriteGens::default();
        const DISTINCT_OBJECTS: u32 = 8192;
        for object in 0..DISTINCT_OBJECTS {
            g.note_write(9, object);
        }
        assert_eq!(g.tracked(), DISTINCT_OBJECTS as usize);
        assert!((0..DISTINCT_OBJECTS).all(|object| {
            let current = g.stamp(9, object);
            current.gen == 1
        }));
    }

    /// A task that goes takes its objects with it, so a later task reusing an
    /// id cannot inherit a stamp that was about something else.
    #[test]
    fn retiring_a_task_forgets_its_objects() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        g.note_write(2, 7);
        let before = g.stamp(1, 7);
        g.retire_task(1);
        assert_eq!(g.tracked(), 1, "only task 2's entry remains");
        assert!(
            !g.stamp(1, 7).quiet_since(before),
            "the entry is gone, so its generation reads as 0 and cannot match"
        );
    }

    /// The retire is a forgetting, so it has to move the epoch exactly as the
    /// clear does. A guest reuses task ids, and a generation that climbs back to
    /// the value a reader recorded is the one shape where the entry going is not
    /// enough on its own.
    #[test]
    fn a_stamp_that_straddles_a_retire_is_not_quiet() {
        let mut g = BufferWriteGens::default();
        for _ in 0..3 {
            g.note_write(5, 7);
        }
        let before = g.stamp(5, 7);
        g.retire_task(5);
        // A different task 5, a different object 7, back to generation 3.
        for _ in 0..3 {
            g.note_write(5, 7);
        }
        assert!(
            !g.stamp(5, 7).quiet_since(before),
            "the tracked object was retired under this reader, so nothing it \
             stamped is comparable across the gap however the count reads"
        );
    }

    /// Retiring a task nothing was tracked for must not move anyone's stamp, or
    /// task teardown alone would report every window as dirty.
    #[test]
    fn retiring_an_untracked_task_leaves_every_stamp_alone() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        let before = g.stamp(1, 7);
        g.retire_task(2);
        assert!(g.stamp(1, 7).quiet_since(before));
    }
}
