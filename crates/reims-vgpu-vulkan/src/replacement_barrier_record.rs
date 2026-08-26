//! Resolve canonical hazard barriers to native Vulkan structures.
//!
//! Backing kind, native handles, image layout, and queue-family ownership come
//! from the epoch-owned representation registry. The projector never infers
//! them from an address, resource kind, or hazard cause.

use crate::replacement_barriers::{
    stage_flags, BackingBarrierScope, BarrierAccess, BarrierTarget, HazardBarrier,
};
use ash::vk;
use reims_vgpu_core::{BarrierOperation, ImageAspect, ResourceGraph};
use reims_vgpu_protocol::{BackingId, ResourceId, ResourceObject};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueFamilyTransfer {
    pub source: u32,
    pub destination: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum NativeBarrierTarget {
    Buffer {
        buffer: vk::Buffer,
        base_offset: u64,
        size: u64,
        queue_families: Option<QueueFamilyTransfer>,
    },
    Image {
        image: vk::Image,
        layout: vk::ImageLayout,
        full_range: vk::ImageSubresourceRange,
        queue_families: Option<QueueFamilyTransfer>,
    },
}

pub trait ReplacementBarrierResolver {
    fn resolve(&self, backing: BackingId) -> Option<NativeBarrierTarget>;
}

pub trait ReplacementBarrierResourceResolver {
    fn alias_backings(&self, resource: ResourceId<ResourceObject>) -> Option<Box<[BackingId]>>;
}

impl ReplacementBarrierResourceResolver for ResourceGraph {
    fn alias_backings(&self, resource: ResourceId<ResourceObject>) -> Option<Box<[BackingId]>> {
        ResourceGraph::alias_backings(self, resource)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierRecordError {
    UnknownBacking(BackingId),
    RegionOutOfBounds(BackingId),
    UnsupportedPlane(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExplicitBarrierResolveError {
    EmptyResourceList,
    EmptyScope,
    UnknownResource {
        index: usize,
        resource: ResourceId<ResourceObject>,
    },
    Native(BarrierRecordError),
}

#[derive(Clone, Debug, Default)]
pub struct NativeBarrierBatch {
    pub memory: Vec<vk::MemoryBarrier2<'static>>,
    pub buffers: Vec<vk::BufferMemoryBarrier2<'static>>,
    pub images: Vec<vk::ImageMemoryBarrier2<'static>>,
}

impl NativeBarrierBatch {
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty() && self.buffers.is_empty() && self.images.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct LegacyBarrierBatch {
    pub source_stages: vk::PipelineStageFlags,
    pub destination_stages: vk::PipelineStageFlags,
    pub memory: Vec<vk::MemoryBarrier<'static>>,
    pub buffers: Vec<vk::BufferMemoryBarrier<'static>>,
    pub images: Vec<vk::ImageMemoryBarrier<'static>>,
}

impl NativeBarrierBatch {
    /// Project the same resolved plan to Vulkan 1.2 core barriers. Stage masks
    /// are unioned at the command boundary because the legacy command carries
    /// them once for the whole batch; access, range, layout, and ownership stay
    /// per barrier. Any non-legacy bit widens to the conservative legacy scope.
    pub fn legacy(&self) -> LegacyBarrierBatch {
        let mut source_stages = vk::PipelineStageFlags::empty();
        let mut destination_stages = vk::PipelineStageFlags::empty();
        for barrier in &self.memory {
            source_stages |= legacy_stage_flags(barrier.src_stage_mask);
            destination_stages |= legacy_stage_flags(barrier.dst_stage_mask);
        }
        for barrier in &self.buffers {
            source_stages |= legacy_stage_flags(barrier.src_stage_mask);
            destination_stages |= legacy_stage_flags(barrier.dst_stage_mask);
        }
        for barrier in &self.images {
            source_stages |= legacy_stage_flags(barrier.src_stage_mask);
            destination_stages |= legacy_stage_flags(barrier.dst_stage_mask);
        }
        LegacyBarrierBatch {
            source_stages,
            destination_stages,
            memory: self
                .memory
                .iter()
                .map(|barrier| {
                    vk::MemoryBarrier::default()
                        .src_access_mask(legacy_access_flags(barrier.src_access_mask))
                        .dst_access_mask(legacy_access_flags(barrier.dst_access_mask))
                })
                .collect(),
            buffers: self
                .buffers
                .iter()
                .map(|barrier| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(legacy_access_flags(barrier.src_access_mask))
                        .dst_access_mask(legacy_access_flags(barrier.dst_access_mask))
                        .src_queue_family_index(barrier.src_queue_family_index)
                        .dst_queue_family_index(barrier.dst_queue_family_index)
                        .buffer(barrier.buffer)
                        .offset(barrier.offset)
                        .size(barrier.size)
                })
                .collect(),
            images: self
                .images
                .iter()
                .map(|barrier| {
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(legacy_access_flags(barrier.src_access_mask))
                        .dst_access_mask(legacy_access_flags(barrier.dst_access_mask))
                        .old_layout(barrier.old_layout)
                        .new_layout(barrier.new_layout)
                        .src_queue_family_index(barrier.src_queue_family_index)
                        .dst_queue_family_index(barrier.dst_queue_family_index)
                        .image(barrier.image)
                        .subresource_range(barrier.subresource_range)
                })
                .collect(),
        }
    }
}

pub fn resolve_hazard_barriers(
    barriers: &[HazardBarrier],
    resolver: &impl ReplacementBarrierResolver,
) -> Result<NativeBarrierBatch, BarrierRecordError> {
    let mut batch = NativeBarrierBatch::default();
    for barrier in barriers {
        match barrier.target {
            BarrierTarget::Global => batch.memory.push(
                vk::MemoryBarrier2::default()
                    .src_stage_mask(barrier.source.stages)
                    .src_access_mask(barrier.source.access)
                    .dst_stage_mask(barrier.destination.stages)
                    .dst_access_mask(barrier.destination.access),
            ),
            BarrierTarget::Backing { backing, scope } => {
                let native = resolver
                    .resolve(backing)
                    .ok_or(BarrierRecordError::UnknownBacking(backing))?;
                resolve_backing(
                    barrier.source,
                    barrier.destination,
                    backing,
                    scope,
                    native,
                    &mut batch,
                )?;
            }
        }
    }
    Ok(batch)
}

/// Resolve one explicit API barrier without inventing a storage identity.
/// Resource-list barriers follow the canonical graph's complete alias closure;
/// scope barriers conservatively use one global Vulkan memory dependency.
pub fn resolve_explicit_barrier(
    operation: &BarrierOperation,
    resources: &impl ReplacementBarrierResourceResolver,
    native: &impl ReplacementBarrierResolver,
) -> Result<NativeBarrierBatch, ExplicitBarrierResolveError> {
    let (before, after) = match operation {
        BarrierOperation::Resources { before, after, .. }
        | BarrierOperation::Scope { before, after, .. } => (*before, *after),
    };
    let source = BarrierAccess {
        stages: stage_flags(before),
        access: vk::AccessFlags2::MEMORY_WRITE,
    };
    let destination = BarrierAccess {
        stages: stage_flags(after),
        access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    };
    let mut batch = NativeBarrierBatch::default();
    match operation {
        BarrierOperation::Scope { scope, .. } => {
            if scope.is_empty() {
                return Err(ExplicitBarrierResolveError::EmptyScope);
            }
            batch.memory.push(
                vk::MemoryBarrier2::default()
                    .src_stage_mask(source.stages)
                    .src_access_mask(source.access)
                    .dst_stage_mask(destination.stages)
                    .dst_access_mask(destination.access),
            );
        }
        BarrierOperation::Resources {
            resources: declared,
            ..
        } => {
            if declared.is_empty() {
                return Err(ExplicitBarrierResolveError::EmptyResourceList);
            }
            let mut backings = BTreeSet::new();
            for (index, resource) in declared.iter().copied().enumerate() {
                let aliases = resources
                    .alias_backings(resource)
                    .ok_or(ExplicitBarrierResolveError::UnknownResource { index, resource })?;
                backings.extend(aliases);
            }
            for backing in backings {
                let target = native
                    .resolve(backing)
                    .ok_or(ExplicitBarrierResolveError::Native(
                        BarrierRecordError::UnknownBacking(backing),
                    ))?;
                resolve_backing(
                    source,
                    destination,
                    backing,
                    BackingBarrierScope::WholeBacking,
                    target,
                    &mut batch,
                )
                .map_err(ExplicitBarrierResolveError::Native)?;
            }
        }
    }
    Ok(batch)
}

/// Record one already-resolved barrier batch.
///
/// # Safety
///
/// `command_buffer` must be valid, recording, and externally synchronized for
/// `device`. Every native handle retained in `batch` must remain valid for the
/// command buffer's submitted lifetime.
pub unsafe fn record_hazard_barriers(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    batch: &NativeBarrierBatch,
) {
    if !batch.is_empty() {
        let legacy = batch.legacy();
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                legacy.source_stages,
                legacy.destination_stages,
                vk::DependencyFlags::empty(),
                &legacy.memory,
                &legacy.buffers,
                &legacy.images,
            )
        };
    }
}

fn legacy_stage_flags(flags: vk::PipelineStageFlags2) -> vk::PipelineStageFlags {
    if flags.as_raw() & !u64::from(u32::MAX) != 0 {
        vk::PipelineStageFlags::ALL_COMMANDS
    } else {
        vk::PipelineStageFlags::from_raw(flags.as_raw() as u32)
    }
}

fn legacy_access_flags(flags: vk::AccessFlags2) -> vk::AccessFlags {
    if flags.as_raw() & !u64::from(u32::MAX) != 0 {
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
    } else {
        vk::AccessFlags::from_raw(flags.as_raw() as u32)
    }
}

fn resolve_backing(
    source: BarrierAccess,
    destination: BarrierAccess,
    backing: BackingId,
    scope: BackingBarrierScope,
    native: NativeBarrierTarget,
    batch: &mut NativeBarrierBatch,
) -> Result<(), BarrierRecordError> {
    // A backing's semantic coordinates describe the API access, while its
    // selected representation determines the native object kind. A barrier
    // moves no content, so when those coordinate systems differ the complete
    // native object is the exact conservative projection of the same backing.
    match (scope, native) {
        (
            BackingBarrierScope::Linear(range),
            NativeBarrierTarget::Buffer {
                buffer,
                base_offset,
                size,
                queue_families,
            },
        ) => {
            if range.end() > size {
                return Err(BarrierRecordError::RegionOutOfBounds(backing));
            }
            let offset = base_offset
                .checked_add(range.start())
                .ok_or(BarrierRecordError::RegionOutOfBounds(backing))?;
            batch.buffers.push(buffer_barrier(
                source,
                destination,
                buffer,
                offset,
                range.end() - range.start(),
                queue_families,
            ));
        }
        (
            BackingBarrierScope::WholeBacking,
            NativeBarrierTarget::Buffer {
                buffer,
                base_offset,
                size,
                queue_families,
            },
        ) => batch.buffers.push(buffer_barrier(
            source,
            destination,
            buffer,
            base_offset,
            size,
            queue_families,
        )),
        (
            BackingBarrierScope::Image {
                aspect,
                mip_start,
                mip_count,
                layer_start,
                layer_count,
            },
            NativeBarrierTarget::Image {
                image,
                layout,
                full_range,
                queue_families,
            },
        ) => {
            let aspect = image_aspect(aspect)?;
            let mip_end = mip_start
                .checked_add(mip_count)
                .ok_or(BarrierRecordError::RegionOutOfBounds(backing))?;
            let layer_end = layer_start
                .checked_add(layer_count)
                .ok_or(BarrierRecordError::RegionOutOfBounds(backing))?;
            let full_mip_end = full_range
                .base_mip_level
                .saturating_add(full_range.level_count);
            let full_layer_end = full_range
                .base_array_layer
                .saturating_add(full_range.layer_count);
            if !full_range.aspect_mask.contains(aspect)
                || mip_start < full_range.base_mip_level
                || mip_end > full_mip_end
                || layer_start < full_range.base_array_layer
                || layer_end > full_layer_end
            {
                return Err(BarrierRecordError::RegionOutOfBounds(backing));
            }
            batch.images.push(image_barrier(
                source,
                destination,
                image,
                layout,
                vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: mip_start,
                    level_count: mip_count,
                    base_array_layer: layer_start,
                    layer_count,
                },
                queue_families,
            ));
        }
        (
            BackingBarrierScope::WholeBacking,
            NativeBarrierTarget::Image {
                image,
                layout,
                full_range,
                queue_families,
            },
        ) => batch.images.push(image_barrier(
            source,
            destination,
            image,
            layout,
            full_range,
            queue_families,
        )),
        (
            BackingBarrierScope::Linear(_),
            NativeBarrierTarget::Image {
                image,
                layout,
                full_range,
                queue_families,
            },
        ) => batch.images.push(image_barrier(
            source,
            destination,
            image,
            layout,
            full_range,
            queue_families,
        )),
        (
            BackingBarrierScope::Image { .. },
            NativeBarrierTarget::Buffer {
                buffer,
                base_offset,
                size,
                queue_families,
            },
        ) => batch.buffers.push(buffer_barrier(
            source,
            destination,
            buffer,
            base_offset,
            size,
            queue_families,
        )),
    }
    Ok(())
}

fn buffer_barrier(
    source_access: BarrierAccess,
    destination_access: BarrierAccess,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    queue_families: Option<QueueFamilyTransfer>,
) -> vk::BufferMemoryBarrier2<'static> {
    let (source, destination) = queue_family_indices(queue_families);
    vk::BufferMemoryBarrier2::default()
        .src_stage_mask(source_access.stages)
        .src_access_mask(source_access.access)
        .dst_stage_mask(destination_access.stages)
        .dst_access_mask(destination_access.access)
        .src_queue_family_index(source)
        .dst_queue_family_index(destination)
        .buffer(buffer)
        .offset(offset)
        .size(size)
}

fn image_barrier(
    source_access: BarrierAccess,
    destination_access: BarrierAccess,
    image: vk::Image,
    layout: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
    queue_families: Option<QueueFamilyTransfer>,
) -> vk::ImageMemoryBarrier2<'static> {
    let (source, destination) = queue_family_indices(queue_families);
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(source_access.stages)
        .src_access_mask(source_access.access)
        .dst_stage_mask(destination_access.stages)
        .dst_access_mask(destination_access.access)
        .old_layout(layout)
        .new_layout(layout)
        .src_queue_family_index(source)
        .dst_queue_family_index(destination)
        .image(image)
        .subresource_range(range)
}

fn queue_family_indices(transfer: Option<QueueFamilyTransfer>) -> (u32, u32) {
    transfer
        .map(|transfer| (transfer.source, transfer.destination))
        .unwrap_or((vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED))
}

fn image_aspect(aspect: ImageAspect) -> Result<vk::ImageAspectFlags, BarrierRecordError> {
    match aspect {
        ImageAspect::Color => Ok(vk::ImageAspectFlags::COLOR),
        ImageAspect::Depth => Ok(vk::ImageAspectFlags::DEPTH),
        ImageAspect::Stencil => Ok(vk::ImageAspectFlags::STENCIL),
        ImageAspect::Plane(0) => Ok(vk::ImageAspectFlags::PLANE_0),
        ImageAspect::Plane(1) => Ok(vk::ImageAspectFlags::PLANE_1),
        ImageAspect::Plane(2) => Ok(vk::ImageAspectFlags::PLANE_2),
        ImageAspect::Plane(plane) => Err(BarrierRecordError::UnsupportedPlane(plane)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_barriers::{BackingBarrierScope, BarrierAccess, BarrierTarget};
    use ash::vk::Handle;
    use reims_vgpu_core::{MemoryBarrierScope, StageScope};
    use reims_vgpu_protocol::TransactionId;
    use std::collections::BTreeMap;

    struct Resolver(BTreeMap<BackingId, NativeBarrierTarget>);

    impl ReplacementBarrierResolver for Resolver {
        fn resolve(&self, backing: BackingId) -> Option<NativeBarrierTarget> {
            self.0.get(&backing).copied()
        }
    }

    struct ResourceAliases(BTreeMap<ResourceId<ResourceObject>, Box<[BackingId]>>);

    impl ReplacementBarrierResourceResolver for ResourceAliases {
        fn alias_backings(&self, resource: ResourceId<ResourceObject>) -> Option<Box<[BackingId]>> {
            self.0.get(&resource).cloned()
        }
    }

    fn barrier(scope: BackingBarrierScope) -> HazardBarrier {
        HazardBarrier {
            producer: TransactionId::new(1),
            consumer: TransactionId::new(2),
            source: BarrierAccess {
                stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::MEMORY_WRITE,
            },
            destination: BarrierAccess {
                stages: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::MEMORY_READ,
            },
            target: BarrierTarget::Backing {
                backing: BackingId::new(4),
                scope,
            },
        }
    }

    #[test]
    fn explicit_resource_barrier_uses_complete_deduplicated_alias_backings() {
        let first = ResourceId::new(1, 2);
        let second = ResourceId::new(3, 4);
        let resources = ResourceAliases(BTreeMap::from([
            (
                first,
                vec![BackingId::new(4), BackingId::new(5)].into_boxed_slice(),
            ),
            (second, vec![BackingId::new(5)].into_boxed_slice()),
        ]));
        let native = Resolver(BTreeMap::from([
            (
                BackingId::new(4),
                NativeBarrierTarget::Buffer {
                    buffer: vk::Buffer::from_raw(40),
                    base_offset: 8,
                    size: 64,
                    queue_families: None,
                },
            ),
            (
                BackingId::new(5),
                NativeBarrierTarget::Image {
                    image: vk::Image::from_raw(50),
                    layout: vk::ImageLayout::GENERAL,
                    full_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 3,
                        base_array_layer: 0,
                        layer_count: 2,
                    },
                    queue_families: None,
                },
            ),
        ]));
        let batch = resolve_explicit_barrier(
            &BarrierOperation::Resources {
                resources: Box::new([first, second]),
                before: StageScope::Compute,
                after: StageScope::Fragment,
            },
            &resources,
            &native,
        )
        .unwrap();

        assert_eq!(batch.buffers.len(), 1);
        assert_eq!(batch.images.len(), 1);
        assert_eq!(batch.buffers[0].buffer, vk::Buffer::from_raw(40));
        assert_eq!(batch.buffers[0].offset, 8);
        assert_eq!(batch.buffers[0].size, 64);
        assert_eq!(batch.images[0].image, vk::Image::from_raw(50));
        assert_eq!(batch.images[0].subresource_range.level_count, 3);
        assert_eq!(
            batch.buffers[0].src_stage_mask,
            vk::PipelineStageFlags2::COMPUTE_SHADER
        );
        assert_eq!(
            batch.buffers[0].dst_stage_mask,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        );
        assert_eq!(
            batch.buffers[0].src_access_mask,
            vk::AccessFlags2::MEMORY_WRITE
        );
        assert_eq!(
            batch.buffers[0].dst_access_mask,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
        );
    }

    #[test]
    fn explicit_scope_barrier_is_global_and_invalid_empty_forms_refuse() {
        let resources = ResourceAliases(BTreeMap::new());
        let native = Resolver(BTreeMap::new());
        let batch = resolve_explicit_barrier(
            &BarrierOperation::Scope {
                scope: MemoryBarrierScope::TEXTURES,
                before: StageScope::Vertex,
                after: StageScope::Fragment,
            },
            &resources,
            &native,
        )
        .unwrap();
        assert_eq!(batch.memory.len(), 1);
        assert!(batch.buffers.is_empty());
        assert!(batch.images.is_empty());

        assert_eq!(
            resolve_explicit_barrier(
                &BarrierOperation::Scope {
                    scope: MemoryBarrierScope::default(),
                    before: StageScope::Vertex,
                    after: StageScope::Fragment,
                },
                &resources,
                &native,
            )
            .unwrap_err(),
            ExplicitBarrierResolveError::EmptyScope
        );
        assert_eq!(
            resolve_explicit_barrier(
                &BarrierOperation::Resources {
                    resources: Box::new([]),
                    before: StageScope::Vertex,
                    after: StageScope::Fragment,
                },
                &resources,
                &native,
            )
            .unwrap_err(),
            ExplicitBarrierResolveError::EmptyResourceList
        );
    }

    #[test]
    fn explicit_resource_barrier_refuses_the_exact_unknown_generation() {
        let resource = ResourceId::new(9, 7);
        let error = resolve_explicit_barrier(
            &BarrierOperation::Resources {
                resources: Box::new([ResourceId::new(1, 1), resource]),
                before: StageScope::Compute,
                after: StageScope::Compute,
            },
            &ResourceAliases(BTreeMap::from([(
                ResourceId::new(1, 1),
                vec![BackingId::new(2)].into_boxed_slice(),
            )])),
            &Resolver(BTreeMap::new()),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ExplicitBarrierResolveError::UnknownResource { index: 1, resource }
        );
    }

    #[test]
    fn linear_scope_resolves_against_declared_buffer_window() {
        let resolver = Resolver(BTreeMap::from([(
            BackingId::new(4),
            NativeBarrierTarget::Buffer {
                buffer: vk::Buffer::from_raw(7),
                base_offset: 128,
                size: 256,
                queue_families: None,
            },
        )]));
        let batch = resolve_hazard_barriers(
            &[barrier(BackingBarrierScope::Linear(
                reims_vgpu_core::LinearRange::new(32, 64).unwrap(),
            ))],
            &resolver,
        )
        .unwrap();
        assert_eq!(batch.buffers.len(), 1);
        assert_eq!(batch.buffers[0].buffer, vk::Buffer::from_raw(7));
        assert_eq!(batch.buffers[0].offset, 160);
        assert_eq!(batch.buffers[0].size, 64);
        assert_eq!(
            batch.buffers[0].src_queue_family_index,
            vk::QUEUE_FAMILY_IGNORED
        );
    }

    #[test]
    fn image_scope_uses_registry_layout_and_explicit_family_transfer() {
        let resolver = Resolver(BTreeMap::from([(
            BackingId::new(4),
            NativeBarrierTarget::Image {
                image: vk::Image::from_raw(8),
                layout: vk::ImageLayout::GENERAL,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 4,
                    base_array_layer: 0,
                    layer_count: 3,
                },
                queue_families: Some(QueueFamilyTransfer {
                    source: 1,
                    destination: 2,
                }),
            },
        )]));
        let batch = resolve_hazard_barriers(
            &[barrier(BackingBarrierScope::Image {
                aspect: ImageAspect::Color,
                mip_start: 2,
                mip_count: 1,
                layer_start: 1,
                layer_count: 2,
            })],
            &resolver,
        )
        .unwrap();
        assert_eq!(batch.images.len(), 1);
        assert_eq!(batch.images[0].old_layout, vk::ImageLayout::GENERAL);
        assert_eq!(batch.images[0].new_layout, vk::ImageLayout::GENERAL);
        assert_eq!(batch.images[0].src_queue_family_index, 1);
        assert_eq!(batch.images[0].dst_queue_family_index, 2);
        assert_eq!(batch.images[0].subresource_range.base_mip_level, 2);
    }

    #[test]
    fn coordinate_mismatch_widens_while_linear_bounds_remain_typed() {
        let resolver = Resolver(BTreeMap::from([(
            BackingId::new(4),
            NativeBarrierTarget::Buffer {
                buffer: vk::Buffer::from_raw(7),
                base_offset: 0,
                size: 16,
                queue_families: None,
            },
        )]));
        assert_eq!(
            resolve_hazard_barriers(
                &[barrier(BackingBarrierScope::Linear(
                    reims_vgpu_core::LinearRange::new(8, 16).unwrap(),
                ))],
                &resolver,
            )
            .unwrap_err(),
            BarrierRecordError::RegionOutOfBounds(BackingId::new(4))
        );
        let batch = resolve_hazard_barriers(
            &[barrier(BackingBarrierScope::Image {
                aspect: ImageAspect::Color,
                mip_start: 0,
                mip_count: 1,
                layer_start: 0,
                layer_count: 1,
            })],
            &resolver,
        )
        .unwrap();
        assert_eq!(batch.buffers.len(), 1);
        assert_eq!(batch.buffers[0].offset, 0);
        assert_eq!(batch.buffers[0].size, 16);
    }

    #[test]
    fn vulkan_12_projection_preserves_legacy_bits_and_widens_high_bits() {
        let batch = NativeBarrierBatch {
            memory: vec![vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::from_raw(1u64 << 40))
                .src_access_mask(vk::AccessFlags2::from_raw(1u64 << 40))
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)],
            buffers: Vec::new(),
            images: Vec::new(),
        };
        let legacy = batch.legacy();
        assert_eq!(legacy.source_stages, vk::PipelineStageFlags::ALL_COMMANDS);
        assert_eq!(
            legacy.destination_stages,
            vk::PipelineStageFlags::FRAGMENT_SHADER
        );
        assert_eq!(
            legacy.memory[0].src_access_mask,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        );
        assert_eq!(
            legacy.memory[0].dst_access_mask,
            vk::AccessFlags::MEMORY_READ
        );
    }
}
