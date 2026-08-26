//! Fixed API binding tables and dirty-only descriptor emission.
//!
//! Binding deltas mutate the encoder-owned table. Emission intersects dirty
//! slots with complete reflection when available; incomplete reflection keeps
//! the conservative dirty set. Clearing happens only for updates actually
//! emitted, so an unused dirty binding remains available to a later pipeline.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingTableError {
    SlotOutOfRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingUpdate<T> {
    pub slot: usize,
    pub value: Option<T>,
}

#[derive(Clone, Copy, Debug)]
pub enum ReflectedBindingUse<'a> {
    Complete(&'a [usize]),
    Conservative,
}

#[derive(Clone, Debug)]
pub struct BindingTable<T, const N: usize> {
    slots: [Option<T>; N],
    dirty: BTreeSet<usize>,
}

impl<T, const N: usize> Default for BindingTable<T, N> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            dirty: BTreeSet::new(),
        }
    }
}

impl<T, const N: usize> BindingTable<T, N> {
    pub fn bind(&mut self, slot: usize, value: T) -> Result<(), BindingTableError> {
        let destination = self
            .slots
            .get_mut(slot)
            .ok_or(BindingTableError::SlotOutOfRange)?;
        *destination = Some(value);
        self.dirty.insert(slot);
        Ok(())
    }

    pub fn unbind(&mut self, slot: usize) -> Result<(), BindingTableError> {
        let destination = self
            .slots
            .get_mut(slot)
            .ok_or(BindingTableError::SlotOutOfRange)?;
        *destination = None;
        self.dirty.insert(slot);
        Ok(())
    }

    pub fn emit_dirty(&mut self, reflected: ReflectedBindingUse<'_>) -> Vec<BindingUpdate<T>>
    where
        T: Clone,
    {
        let emitted = match reflected {
            ReflectedBindingUse::Complete(used) => used
                .iter()
                .copied()
                .filter(|slot| self.dirty.contains(slot) && *slot < N)
                .collect::<BTreeSet<_>>(),
            ReflectedBindingUse::Conservative => self.dirty.clone(),
        };
        let updates = emitted
            .iter()
            .map(|slot| BindingUpdate {
                slot: *slot,
                value: self.slots[*slot].clone(),
            })
            .collect();
        self.dirty.retain(|slot| !emitted.contains(slot));
        updates
    }

    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DescriptorCapabilities {
    pub descriptor_buffer: bool,
    pub push_descriptor: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DescriptorTier {
    DescriptorBuffer,
    PushDescriptor,
    WorkerDescriptorPool,
}

pub const fn select_descriptor_tier(capabilities: DescriptorCapabilities) -> DescriptorTier {
    if capabilities.descriptor_buffer {
        DescriptorTier::DescriptorBuffer
    } else if capabilities.push_descriptor {
        DescriptorTier::PushDescriptor
    } else {
        DescriptorTier::WorkerDescriptorPool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_reflection_emits_only_used_dirty_slots() {
        let mut table = BindingTable::<&str, 8>::default();
        table.bind(1, "one").unwrap();
        table.bind(3, "three").unwrap();
        assert_eq!(
            table.emit_dirty(ReflectedBindingUse::Complete(&[3, 3, 7])),
            vec![BindingUpdate {
                slot: 3,
                value: Some("three")
            }]
        );
        assert_eq!(table.dirty_count(), 1);
        assert_eq!(
            table.emit_dirty(ReflectedBindingUse::Complete(&[1])),
            vec![BindingUpdate {
                slot: 1,
                value: Some("one")
            }]
        );
    }

    #[test]
    fn incomplete_reflection_conservatively_emits_every_dirty_delta() {
        let mut table = BindingTable::<u32, 4>::default();
        table.bind(0, 10).unwrap();
        table.bind(2, 20).unwrap();
        table.unbind(0).unwrap();
        assert_eq!(
            table.emit_dirty(ReflectedBindingUse::Conservative),
            vec![
                BindingUpdate {
                    slot: 0,
                    value: None
                },
                BindingUpdate {
                    slot: 2,
                    value: Some(20)
                }
            ]
        );
        assert_eq!(table.dirty_count(), 0);
    }

    #[test]
    fn unchanged_tables_emit_nothing_and_out_of_range_is_typed() {
        let mut table = BindingTable::<u32, 2>::default();
        assert!(table
            .emit_dirty(ReflectedBindingUse::Conservative)
            .is_empty());
        assert_eq!(table.bind(2, 1), Err(BindingTableError::SlotOutOfRange));
    }

    #[test]
    fn descriptor_tier_is_selected_only_from_reported_capabilities() {
        assert_eq!(
            select_descriptor_tier(DescriptorCapabilities {
                descriptor_buffer: true,
                push_descriptor: true
            }),
            DescriptorTier::DescriptorBuffer
        );
        assert_eq!(
            select_descriptor_tier(DescriptorCapabilities {
                descriptor_buffer: false,
                push_descriptor: true
            }),
            DescriptorTier::PushDescriptor
        );
        assert_eq!(
            select_descriptor_tier(DescriptorCapabilities {
                descriptor_buffer: false,
                push_descriptor: false
            }),
            DescriptorTier::WorkerDescriptorPool
        );
    }
}
