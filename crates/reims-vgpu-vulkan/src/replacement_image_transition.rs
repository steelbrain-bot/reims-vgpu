//! Project prepared image state into valid release/acquire and layout barriers.

use crate::{
    replacement_barrier_record::NativeBarrierBatch,
    replacement_image_state::{
        PlannedImageQueueTransfer, PreparedImageState, PreparedImageStateBatch,
        ReplacementImageKey, ReplacementImageReleaseKey, ReplacementImageTransition,
    },
};
use ash::vk;
use reims_vgpu_core::QueueTimelinePoint;
use reims_vgpu_protocol::QueueOwnerId;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct NativeImageTarget {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub image_type: vk::ImageType,
    pub full_range: vk::ImageSubresourceRange,
    pub usage: vk::ImageUsageFlags,
    pub pixel_format: u16,
    pub extent: vk::Extent3D,
    pub samples: vk::SampleCountFlags,
}

pub trait ReplacementImageResolver {
    fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget>;

    fn resolve_image_view(
        &self,
        image: ReplacementImageKey,
        range: vk::ImageSubresourceRange,
    ) -> Option<NativeImageTarget> {
        let target = self.resolve_image(image)?;
        same_subresource_range(target.full_range, range).then_some(target)
    }

    fn resolve_texture_binding_view(
        &self,
        image: ReplacementImageKey,
        view: reims_vgpu_core::ResolvedTextureBindingView,
    ) -> Result<NativeImageTarget, TextureBindingViewDecline> {
        // Ownership, not the chain: this resolver serves only a binding that
        // is the whole image, and a plane view is exactly that while naming
        // its parent surface as its base.
        if view.resource != view.image_owner {
            return Err(TextureBindingViewDecline::AliasedBaseResource);
        }
        if view.pixel_format == 0 {
            return Err(TextureBindingViewDecline::UndeclaredFormat);
        }
        if !view.swizzle.is_identity() {
            return Err(TextureBindingViewDecline::SwizzledView);
        }
        let target = self
            .resolve_image(image)
            .ok_or(TextureBindingViewDecline::UnknownRepresentation)?;
        if target.pixel_format != view.pixel_format {
            return Err(TextureBindingViewDecline::ReinterpretedFormat {
                declared: view.pixel_format,
                native: target.pixel_format,
            });
        }
        let subrange = |value: u64| u32::try_from(value).ok();
        let range = vk::ImageSubresourceRange {
            aspect_mask: target.full_range.aspect_mask,
            base_mip_level: subrange(view.range.level_base)
                .ok_or(TextureBindingViewDecline::SubresourceRange)?,
            level_count: subrange(view.range.level_count)
                .ok_or(TextureBindingViewDecline::SubresourceRange)?,
            base_array_layer: subrange(view.range.slice_base)
                .ok_or(TextureBindingViewDecline::SubresourceRange)?,
            layer_count: subrange(view.range.slice_count)
                .ok_or(TextureBindingViewDecline::SubresourceRange)?,
        };
        if !same_subresource_range(target.full_range, range) {
            return Err(TextureBindingViewDecline::PartialSubresourceRange);
        }
        Ok(target)
    }
}

/// Why one decoded texture-binding view has no native image behind it.
///
/// Every arm but [`Self::UnknownRepresentation`] is a decoded view this backend
/// does not yet build a `VkImageView` for. They were one `None` and reached the
/// log as `UnknownImageView`, which named the representation and said nothing
/// about which of five unrelated cases the guest had asked for -- a texture
/// view over a mip range, a format reinterpretation and a swizzle are three
/// different pieces of work, and only one of them is a lifetime problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureBindingViewDecline {
    /// The representation this key names is not resolvable. Unlike the rest,
    /// this is a lifetime fault rather than an unimplemented view.
    UnknownRepresentation,
    /// The representation resolved, but it is a buffer rather than an image.
    NotImage,
    /// The view names another resource as its base and no shader view has been
    /// installed for it.
    ShaderViewAbsent,
    /// An installed shader view exists for this resource but was built for a
    /// different decoded view.
    ShaderViewMismatch,
    /// The view names a base resource other than itself: a texture view onto
    /// another texture's storage.
    AliasedBaseResource,
    /// The decoded view carries no pixel format.
    UndeclaredFormat,
    /// The view applies a component swizzle.
    SwizzledView,
    /// The view reinterprets its base texture's pixel format.
    ReinterpretedFormat { declared: u16, native: u16 },
    /// A level or slice bound does not fit the Vulkan subresource width.
    SubresourceRange,
    /// The view covers part of the image rather than all of it.
    PartialSubresourceRange,
}

impl TextureBindingViewDecline {
    /// Whether asking again can ever produce a different answer.
    ///
    /// [`Self::UnknownRepresentation`] is a lifetime fault: the representation
    /// this key names has not been materialized yet, and materializing it is
    /// exactly what a later packet does. Every other arm is a decoded view
    /// this backend does not build a `VkImageView` for, decided by the view
    /// the guest declared and the image that already exists -- neither of
    /// which a retry changes. Re-offering one of those parks its channel for
    /// the life of the device, so a caller that can refuse should.
    pub const fn is_unimplemented(&self) -> bool {
        !matches!(self, Self::UnknownRepresentation)
    }
}

impl std::fmt::Display for TextureBindingViewDecline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRepresentation => formatter.write_str("unknown_representation"),
            Self::NotImage => formatter.write_str("not_image"),
            Self::ShaderViewAbsent => formatter.write_str("shader_view_absent"),
            Self::ShaderViewMismatch => formatter.write_str("shader_view_mismatch"),
            Self::AliasedBaseResource => formatter.write_str("aliased_base_resource"),
            Self::UndeclaredFormat => formatter.write_str("undeclared_format"),
            Self::SwizzledView => formatter.write_str("swizzled_view"),
            Self::ReinterpretedFormat { declared, native } => {
                write!(
                    formatter,
                    "reinterpreted_format declared={declared} native={native}"
                )
            }
            Self::SubresourceRange => formatter.write_str("subresource_range"),
            Self::PartialSubresourceRange => formatter.write_str("partial_subresource_range"),
        }
    }
}

pub(crate) fn same_subresource_range(
    left: vk::ImageSubresourceRange,
    right: vk::ImageSubresourceRange,
) -> bool {
    left.aspect_mask == right.aspect_mask
        && left.base_mip_level == right.base_mip_level
        && left.level_count == right.level_count
        && left.base_array_layer == right.base_array_layer
        && left.layer_count == right.layer_count
}

/// Translate one exact semantic image coordinate into the Vulkan view range
/// used by recording. Linear regions and whole-backing identities are not
/// single image subresources.
pub fn exact_image_subresource_range(
    region: reims_vgpu_core::BackingRegion,
) -> Option<vk::ImageSubresourceRange> {
    let reims_vgpu_core::BackingRegion::Image(region) = region else {
        return None;
    };
    let aspect_mask = match region.aspect {
        reims_vgpu_core::ImageAspect::Color => vk::ImageAspectFlags::COLOR,
        reims_vgpu_core::ImageAspect::Depth => vk::ImageAspectFlags::DEPTH,
        reims_vgpu_core::ImageAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        reims_vgpu_core::ImageAspect::Plane(index) => match index {
            0 => vk::ImageAspectFlags::PLANE_0,
            1 => vk::ImageAspectFlags::PLANE_1,
            2 => vk::ImageAspectFlags::PLANE_2,
            _ => return None,
        },
    };
    Some(vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: region.mip,
        level_count: 1,
        base_array_layer: region.layer,
        layer_count: 1,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageTransitionResolveError {
    UnknownRepresentation(ReplacementImageKey),
    MissingUsage {
        image: ReplacementImageKey,
        required: vk::ImageUsageFlags,
    },
    EmptyAspect(ReplacementImageKey),
    EmptySubresourceRange(ReplacementImageKey),
}

#[derive(Clone, Debug)]
pub struct NativeImageRelease {
    pub source_queue_family: u32,
    pub source_queue: QueueOwnerId,
    pub predecessor: QueueTimelinePoint,
    pub barriers: NativeBarrierBatch,
}

#[derive(Clone, Debug)]
pub struct NativeImageUseTransitions {
    pub before: NativeBarrierBatch,
    pub after: NativeBarrierBatch,
}

#[derive(Clone, Debug)]
pub struct PreparedNativeImageState {
    pub transaction: reims_vgpu_protocol::TransactionId,
    pub destination_queue_family: u32,
    pub releases: Box<[NativeImageRelease]>,
    pub transitions: NativeImageUseTransitions,
}

pub fn resolve_image_transitions(
    prepared: &PreparedImageState,
    resolver: &impl ReplacementImageResolver,
) -> Result<PreparedNativeImageState, ImageTransitionResolveError> {
    let mut releases =
        BTreeMap::<(u32, QueueOwnerId), (QueueTimelinePoint, NativeBarrierBatch)>::new();
    let mut before = NativeBarrierBatch::default();
    let mut after = NativeBarrierBatch::default();
    for transition in prepared.transitions().iter().copied() {
        let target = resolver.resolve_image(transition.image).ok_or(
            ImageTransitionResolveError::UnknownRepresentation(transition.image),
        )?;
        validate_target(transition, target)?;
        match transition.queue_transfer {
            Some(transfer) => {
                if !prepared.release_accepted(ReplacementImageReleaseKey {
                    source_queue_family: transfer.source,
                    source_queue: transfer.source_point.queue,
                }) {
                    let release = releases
                        .entry((transfer.source, transfer.source_point.queue))
                        .or_insert_with(|| (transfer.source_point, NativeBarrierBatch::default()));
                    if transfer.source_point.value > release.0.value {
                        release.0 = transfer.source_point;
                    }
                    release
                        .1
                        .images
                        .push(release_barrier(transition, target, transfer));
                }
                before
                    .images
                    .push(acquire_barrier(transition, target, transfer));
            }
            None => before.images.push(local_before_barrier(transition, target)),
        }
        after.images.push(local_after_barrier(transition, target));
    }
    Ok(PreparedNativeImageState {
        transaction: prepared.transaction(),
        destination_queue_family: prepared.queue_family(),
        releases: releases
            .into_iter()
            .map(
                |((source_queue_family, source_queue), (predecessor, barriers))| {
                    NativeImageRelease {
                        source_queue_family,
                        source_queue,
                        predecessor,
                        barriers,
                    }
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        transitions: NativeImageUseTransitions { before, after },
    })
}

pub fn resolve_image_batch_releases(
    prepared: &PreparedImageStateBatch,
    resolver: &impl ReplacementImageResolver,
) -> Result<Box<[NativeImageRelease]>, ImageTransitionResolveError> {
    let mut releases =
        BTreeMap::<(u32, QueueOwnerId), (QueueTimelinePoint, NativeBarrierBatch)>::new();
    for operation in prepared.operations() {
        for release in resolve_image_transitions(operation, resolver)?.releases {
            let aggregate = releases
                .entry((release.source_queue_family, release.source_queue))
                .or_insert_with(|| (release.predecessor, NativeBarrierBatch::default()));
            if release.predecessor.value > aggregate.0.value {
                aggregate.0 = release.predecessor;
            }
            aggregate.1.memory.extend(release.barriers.memory);
            aggregate.1.buffers.extend(release.barriers.buffers);
            aggregate.1.images.extend(release.barriers.images);
        }
    }
    Ok(releases
        .into_iter()
        .map(
            |((source_queue_family, source_queue), (predecessor, barriers))| NativeImageRelease {
                source_queue_family,
                source_queue,
                predecessor,
                barriers,
            },
        )
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn validate_target(
    transition: ReplacementImageTransition,
    target: NativeImageTarget,
) -> Result<(), ImageTransitionResolveError> {
    if !target.usage.contains(transition.required_usage) {
        return Err(ImageTransitionResolveError::MissingUsage {
            image: transition.image,
            required: transition.required_usage,
        });
    }
    if target.full_range.aspect_mask.is_empty() {
        return Err(ImageTransitionResolveError::EmptyAspect(transition.image));
    }
    if target.full_range.level_count == 0 || target.full_range.layer_count == 0 {
        return Err(ImageTransitionResolveError::EmptySubresourceRange(
            transition.image,
        ));
    }
    Ok(())
}

fn local_before_barrier(
    transition: ReplacementImageTransition,
    target: NativeImageTarget,
) -> vk::ImageMemoryBarrier2<'static> {
    barrier(
        transition,
        target,
        transition.initial_layout,
        transition.use_layout,
        vk::QUEUE_FAMILY_IGNORED,
        vk::QUEUE_FAMILY_IGNORED,
        initial_stage(transition.initial_layout),
        initial_access(transition.initial_layout),
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    )
}

fn release_barrier(
    transition: ReplacementImageTransition,
    target: NativeImageTarget,
    transfer: PlannedImageQueueTransfer,
) -> vk::ImageMemoryBarrier2<'static> {
    barrier(
        transition,
        target,
        transition.initial_layout,
        transition.use_layout,
        transfer.source,
        transfer.destination,
        initial_stage(transition.initial_layout),
        initial_access(transition.initial_layout),
        vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        vk::AccessFlags2::empty(),
    )
}

fn acquire_barrier(
    transition: ReplacementImageTransition,
    target: NativeImageTarget,
    transfer: PlannedImageQueueTransfer,
) -> vk::ImageMemoryBarrier2<'static> {
    barrier(
        transition,
        target,
        transition.initial_layout,
        transition.use_layout,
        transfer.source,
        transfer.destination,
        vk::PipelineStageFlags2::TOP_OF_PIPE,
        vk::AccessFlags2::empty(),
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    )
}

fn local_after_barrier(
    transition: ReplacementImageTransition,
    target: NativeImageTarget,
) -> vk::ImageMemoryBarrier2<'static> {
    barrier(
        transition,
        target,
        transition.use_layout,
        transition.final_layout,
        vk::QUEUE_FAMILY_IGNORED,
        vk::QUEUE_FAMILY_IGNORED,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    )
}

#[allow(clippy::too_many_arguments)]
fn barrier(
    _transition: ReplacementImageTransition,
    target: NativeImageTarget,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    source_queue_family: u32,
    destination_queue_family: u32,
    source_stage: vk::PipelineStageFlags2,
    source_access: vk::AccessFlags2,
    destination_stage: vk::PipelineStageFlags2,
    destination_access: vk::AccessFlags2,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(source_stage)
        .src_access_mask(source_access)
        .dst_stage_mask(destination_stage)
        .dst_access_mask(destination_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(source_queue_family)
        .dst_queue_family_index(destination_queue_family)
        .image(target.image)
        .subresource_range(target.full_range)
}

fn initial_stage(layout: vk::ImageLayout) -> vk::PipelineStageFlags2 {
    if layout == vk::ImageLayout::UNDEFINED {
        vk::PipelineStageFlags2::TOP_OF_PIPE
    } else {
        vk::PipelineStageFlags2::ALL_COMMANDS
    }
}

fn initial_access(layout: vk::ImageLayout) -> vk::AccessFlags2 {
    if layout == vk::ImageLayout::UNDEFINED {
        vk::AccessFlags2::empty()
    } else {
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_image_state::{
        ReplacementImageSharing, ReplacementImageState, ReplacementImageStateOwner,
        ReplacementImageUse,
    };
    use ash::vk::Handle;
    use reims_vgpu_core::QueueTimelinePoint;
    use reims_vgpu_protocol::{
        BackingId, QueueOwnerId, QueueTimelineValue, RepresentationId, TransactionId,
        VulkanDeviceEpochId,
    };

    const EPOCH: VulkanDeviceEpochId = VulkanDeviceEpochId::new(1);

    fn point(queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: EPOCH,
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    fn key(id: u64) -> ReplacementImageKey {
        ReplacementImageKey {
            backing: BackingId::new(id),
            representation: RepresentationId::new(id + 10),
        }
    }

    struct Resolver {
        key: ReplacementImageKey,
        target: NativeImageTarget,
    }

    impl ReplacementImageResolver for Resolver {
        fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
            (image == self.key).then_some(self.target)
        }
    }

    struct Resolvers(BTreeMap<ReplacementImageKey, NativeImageTarget>);

    impl ReplacementImageResolver for Resolvers {
        fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
            self.0.get(&image).copied()
        }
    }

    fn target(usage: vk::ImageUsageFlags) -> NativeImageTarget {
        NativeImageTarget {
            image: vk::Image::from_raw(7),
            view: vk::ImageView::null(),
            image_type: vk::ImageType::TYPE_2D,
            full_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            usage,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            extent: vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    #[test]
    fn exclusive_cross_queue_use_produces_paired_release_and_acquire() {
        let image = key(1);
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 2 },
                    last_use: Some(point(2, 3)),
                },
            )
            .unwrap();
        let prepared = owner
            .prepare(
                TransactionId::new(1),
                4,
                [ReplacementImageUse {
                    image,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }],
            )
            .unwrap();
        let native = resolve_image_transitions(
            &prepared,
            &Resolver {
                key: image,
                target: target(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
            },
        )
        .unwrap();
        assert_eq!(native.releases.len(), 1);
        assert_eq!(native.releases[0].source_queue_family, 2);
        assert_eq!(native.releases[0].source_queue, QueueOwnerId::new(2));
        assert_eq!(native.releases[0].predecessor, point(2, 3));
        let release = native.releases[0].barriers.images[0];
        let acquire = native.transitions.before.images[0];
        assert_eq!(release.src_queue_family_index, 2);
        assert_eq!(release.dst_queue_family_index, 4);
        assert_eq!(acquire.src_queue_family_index, 2);
        assert_eq!(acquire.dst_queue_family_index, 4);
        assert_eq!(release.old_layout, vk::ImageLayout::GENERAL);
        assert_eq!(release.new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        assert_eq!(acquire.old_layout, release.old_layout);
        assert_eq!(acquire.new_layout, release.new_layout);
        assert_eq!(
            native.transitions.after.images[0].new_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        let released = owner
            .release_accepted(
                prepared,
                ReplacementImageReleaseKey {
                    source_queue_family: 2,
                    source_queue: QueueOwnerId::new(2),
                },
                point(2, 4),
            )
            .unwrap();
        let retry = resolve_image_transitions(
            &released,
            &Resolver {
                key: image,
                target: target(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
            },
        )
        .unwrap();
        assert!(retry.releases.is_empty());
        let retry_acquire = retry.transitions.before.images[0];
        assert_eq!(retry_acquire.image, acquire.image);
        assert_eq!(retry_acquire.old_layout, acquire.old_layout);
        assert_eq!(retry_acquire.new_layout, acquire.new_layout);
        assert_eq!(retry_acquire.src_queue_family_index, 2);
        assert_eq!(retry_acquire.dst_queue_family_index, 4);
    }

    #[test]
    fn batch_coalesces_one_source_queue_release_with_the_latest_predecessor() {
        let first = key(3);
        let second = key(4);
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        for (image, predecessor) in [(first, 3), (second, 5)] {
            owner
                .register(
                    image,
                    ReplacementImageState {
                        layout: vk::ImageLayout::GENERAL,
                        sharing: ReplacementImageSharing::Exclusive { owner: 2 },
                        last_use: Some(point(2, predecessor)),
                    },
                )
                .unwrap();
        }
        let use_ = |image| ReplacementImageUse {
            image,
            required_usage: vk::ImageUsageFlags::TRANSFER_DST,
            use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            final_layout: vk::ImageLayout::GENERAL,
        };
        let prepared = owner
            .prepare_batch(
                TransactionId::new(2),
                4,
                vec![
                    (0, Box::new([use_(first)]) as Box<[_]>),
                    (1, Box::new([use_(second)]) as Box<[_]>),
                ]
                .into_boxed_slice(),
            )
            .unwrap();
        let releases = resolve_image_batch_releases(
            &prepared,
            &Resolvers(BTreeMap::from([
                (first, target(vk::ImageUsageFlags::TRANSFER_DST)),
                (second, target(vk::ImageUsageFlags::TRANSFER_DST)),
            ])),
        )
        .unwrap();
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].source_queue_family, 2);
        assert_eq!(releases[0].source_queue, QueueOwnerId::new(2));
        assert_eq!(releases[0].predecessor, point(2, 5));
        assert_eq!(releases[0].barriers.images.len(), 2);
    }

    #[test]
    fn concurrent_use_needs_no_release_and_missing_usage_is_typed() {
        let image = key(1);
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::UNDEFINED,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let prepared = owner
            .prepare(
                TransactionId::new(1),
                3,
                [ReplacementImageUse {
                    image,
                    required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                    use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                }],
            )
            .unwrap();
        assert!(matches!(
            resolve_image_transitions(
                &prepared,
                &Resolver {
                    key: image,
                    target: target(vk::ImageUsageFlags::TRANSFER_DST),
                },
            ),
            Err(ImageTransitionResolveError::MissingUsage { image: found, .. }) if found == image
        ));
        let native = resolve_image_transitions(
            &prepared,
            &Resolver {
                key: image,
                target: target(vk::ImageUsageFlags::TRANSFER_SRC),
            },
        )
        .unwrap();
        assert!(native.releases.is_empty());
        assert_eq!(
            native.transitions.before.images[0].src_queue_family_index,
            vk::QUEUE_FAMILY_IGNORED
        );
        assert!(native.transitions.before.images[0]
            .src_access_mask
            .is_empty());
    }

    /// An unimplemented view ends the packet; an unmaterialized one waits.
    ///
    /// A binding that owns the image it names resolves to it whole, however
    /// far its view chain runs.
    ///
    /// This resolver serves only the whole-image binding, and it told that
    /// apart by asking whether the resource is the end of its own view chain.
    /// An IOSurface plane view is not: it names the parent surface as its
    /// base, and the surface owns the allocation and no image at all. So the
    /// one binding class this rail exists to serve on a surface plane was
    /// refused `AliasedBaseResource` --- an unimplemented refusal, which ends
    /// the packet.
    #[test]
    fn a_binding_that_owns_its_image_resolves_whole_whatever_its_base_is() {
        let image = key(1);
        let mut whole = target(vk::ImageUsageFlags::SAMPLED);
        whole.pixel_format = 70;
        let resolver = Resolver {
            key: image,
            target: whole,
        };
        let plane = reims_vgpu_protocol::ResourceId::new(4, 1);
        let surface = reims_vgpu_protocol::ResourceId::new(3, 1);
        let view = reims_vgpu_core::ResolvedTextureBindingView {
            resource: plane,
            // The chain terminates at the surface; the image is the plane's.
            base: surface,
            image_owner: plane,
            range: reims_vgpu_core::ResolvedTextureViewRange {
                level_base: 0,
                level_count: 1,
                slice_base: 0,
                slice_count: 1,
            },
            texture_type: reims_vgpu_protocol::TextureType::D2,
            pixel_format: 70,
            swizzle: reims_vgpu_protocol::swizzle_identity(),
        };
        assert_eq!(
            resolver
                .resolve_texture_binding_view(image, view)
                .expect("the plane owns this image")
                .image,
            whole.image
        );
        // A resource that does not own the image is not this rail's business.
        let mut aliased = view;
        aliased.image_owner = surface;
        assert_eq!(
            resolver.resolve_texture_binding_view(image, aliased).err(),
            Some(TextureBindingViewDecline::AliasedBaseResource)
        );
    }

    /// The two arrive at the classifier as one `UnknownImageView` and mean
    /// opposite things. Reading a `ReinterpretedFormat` as waitable parks the
    /// channel for the life of the device -- a boot spent five minutes
    /// re-offering one aliased binding fifty-three thousand times, with every
    /// later transaction on that channel refusing behind it. Reading an
    /// `UnknownRepresentation` as terminal throws away an EXEC whose image was
    /// one packet away from existing.
    #[test]
    fn an_unimplemented_image_view_ends_a_packet_and_an_unmaterialized_one_does_not() {
        assert!(
            TextureBindingViewDecline::ReinterpretedFormat {
                declared: 0x50,
                native: 0x73,
            }
            .is_unimplemented(),
            "a view whose format disagrees with the image that already exists asks a question \
             neither side can answer differently later"
        );
        assert!(TextureBindingViewDecline::SwizzledView.is_unimplemented());
        assert!(TextureBindingViewDecline::PartialSubresourceRange.is_unimplemented());
        assert!(TextureBindingViewDecline::AliasedBaseResource.is_unimplemented());
        assert!(
            !TextureBindingViewDecline::UnknownRepresentation.is_unimplemented(),
            "materializing the representation is exactly what a later packet does"
        );
    }
}
