//! Opaque ownership contracts for executor-local residents.

use reims_vgpu_protocol::StorageImageFormat;
use std::collections::BTreeMap;

/// Whether a retained guest-memory gather is licensed for identity reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatherVouch {
    /// The bytes cannot have changed since the retained gather was produced.
    Vouched,
    /// The bind must gather current bytes and publish a new identity.
    Fresh,
}

impl GatherVouch {
    pub const fn is_vouched(self) -> bool {
        matches!(self, Self::Vouched)
    }
}

/// Guest-semantic origin of a compute-resident texture.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeStorageOrigin {
    Surface {
        mapping_id: u32,
        map_generation: u32,
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
    },
    Linear {
        task_id: u32,
        texture_ref: u32,
        gva: u64,
        row_stride: u32,
        span_end: u64,
    },
    Heap {
        task_id: u32,
        texture_ref: u32,
    },
}

/// Exact protocol-backed compute storage-image view eligible for residency.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeStorageResidencyKey {
    pub origin: ComputeStorageOrigin,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
}

impl ComputeStorageResidencyKey {
    #[allow(
        clippy::too_many_arguments,
        reason = "the key constructor names every contract identity component"
    )]
    pub fn surface(
        mapping_id: u32,
        map_generation: u32,
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::Surface {
                mapping_id,
                map_generation,
                surface_offset,
                surface_bpr,
                span_end,
            },
            width,
            height,
            pixel_format,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the key constructor names every contract identity component"
    )]
    pub fn linear(
        task_id: u32,
        texture_ref: u32,
        gva: u64,
        row_stride: u32,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::Linear {
                task_id,
                texture_ref,
                gva,
                row_stride,
                span_end,
            },
            width,
            height,
            pixel_format,
        }
    }

    pub fn heap(
        task_id: u32,
        texture_ref: u32,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::Heap {
                task_id,
                texture_ref,
            },
            width,
            height,
            pixel_format,
        }
    }

    pub fn is_linear(&self) -> bool {
        matches!(self.origin, ComputeStorageOrigin::Linear { .. })
    }

    pub fn is_heap(&self) -> bool {
        matches!(self.origin, ComputeStorageOrigin::Heap { .. })
    }

    pub fn surface_window(&self) -> Option<(u32, u64, u64)> {
        match self.origin {
            ComputeStorageOrigin::Surface {
                mapping_id,
                surface_offset,
                span_end,
                ..
            } => Some((mapping_id, surface_offset, span_end)),
            ComputeStorageOrigin::Linear { .. } | ComputeStorageOrigin::Heap { .. } => None,
        }
    }

    pub fn linear_window(&self) -> Option<(u32, u32, u64, u32, u64)> {
        match self.origin {
            ComputeStorageOrigin::Linear {
                task_id,
                texture_ref,
                gva,
                row_stride,
                span_end,
            } => Some((task_id, texture_ref, gva, row_stride, span_end)),
            ComputeStorageOrigin::Surface { .. } | ComputeStorageOrigin::Heap { .. } => None,
        }
    }

    pub fn resource_ref(&self) -> Option<u32> {
        match self.origin {
            ComputeStorageOrigin::Linear { texture_ref, .. }
            | ComputeStorageOrigin::Heap { texture_ref, .. } => Some(texture_ref),
            ComputeStorageOrigin::Surface { .. } => None,
        }
    }
}

/// Semantic generations of compute-resident subresources.
///
/// This ledger states which executor generation still represents the current
/// guest-visible content. It does not own the native resident: the executor
/// answers that independently through [`ComputeResidencyService`]. Keeping the
/// two facts separate makes a lost native resident a typed execution outcome
/// without turning backend availability into content authority.
#[derive(Debug, Default)]
pub struct ComputeResidencyLedger {
    generations: BTreeMap<ComputeStorageResidencyKey, u32>,
}

impl ComputeResidencyLedger {
    pub fn generation(&self, key: &ComputeStorageResidencyKey) -> Option<u32> {
        self.generations.get(key).copied()
    }

    pub fn publish(&mut self, key: ComputeStorageResidencyKey, generation: u32) {
        self.generations.insert(key, generation);
    }

    pub fn invalidate_surface_window(&mut self, mapping_id: u32, lo: u64, hi: u64) {
        self.generations.retain(|key, _| {
            key.surface_window().is_none_or(|(candidate, start, end)| {
                candidate != mapping_id || end <= lo || start >= hi
            })
        });
    }

    pub fn len(&self) -> usize {
        self.generations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.generations.is_empty()
    }

    pub fn contains(&self, key: &ComputeStorageResidencyKey) -> bool {
        self.generations.contains_key(key)
    }
}

/// Persistent compute-image residency service.
pub trait ComputeResidencyService: std::fmt::Debug + Send + Sync {
    fn compute_resident_storage_generation(
        &self,
        _identity: &ComputeStorageResidencyKey,
    ) -> Option<u32> {
        None
    }

    fn compute_resident_sample_source(
        &self,
        _identity: &ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        None
    }

    fn unpin_resident_storage(&self, _identity: &ComputeStorageResidencyKey) {}

    fn retire_resident_storage_content(&self, _identity: &ComputeStorageResidencyKey) {}

    fn note_resident_storage_copied_out(&self, _identity: &ComputeStorageResidencyKey) {}
}

/// Backend-independent classification of a retained target's current content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentContentBacking {
    NotReady,
    DeviceAllocation,
}

#[cfg(test)]
mod tests {
    use super::{ComputeResidencyLedger, ComputeStorageOrigin, ComputeStorageResidencyKey};

    #[test]
    fn compute_residency_origins_are_disjoint_typed_identities() {
        let surface = ComputeStorageResidencyKey::surface(7, 2, 0, 64, 4096, 16, 16, 0x50);
        let linear = ComputeStorageResidencyKey::linear(7, 2, 0, 64, 4096, 16, 16, 0x50);
        let heap = ComputeStorageResidencyKey::heap(7, 2, 16, 16, 0x50);

        assert_ne!(surface, linear);
        assert_ne!(linear, heap);
        assert!(matches!(
            surface.origin,
            ComputeStorageOrigin::Surface { .. }
        ));
        assert_eq!(surface.surface_window(), Some((7, 0, 4096)));
        assert_eq!(linear.resource_ref(), Some(2));
        assert!(heap.is_heap());
    }

    #[test]
    fn residency_invalidation_retires_only_intersecting_surface_windows() {
        let hit = ComputeStorageResidencyKey::surface(7, 1, 0, 16, 64, 4, 4, 0x50);
        let sibling = ComputeStorageResidencyKey::surface(7, 1, 128, 16, 192, 4, 4, 0x50);
        let heap = ComputeStorageResidencyKey::heap(1, 2, 4, 4, 0x50);
        let mut ledger = ComputeResidencyLedger::default();
        ledger.publish(hit, 3);
        ledger.publish(sibling, 4);
        ledger.publish(heap, 5);

        ledger.invalidate_surface_window(7, 32, 96);

        assert_eq!(ledger.generation(&hit), None);
        assert_eq!(ledger.generation(&sibling), Some(4));
        assert_eq!(ledger.generation(&heap), Some(5));
    }
}
