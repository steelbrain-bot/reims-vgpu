//! Native image-state ownership for replacement presentation.
//!
//! Presentation is a transfer read of the exact execution representation. It
//! uses the same pending/accepted image-state owner as EXEC, so a display blit
//! cannot race a prior image transition or publish a layout invented by WSI.

use crate::{
    replacement_image_state::{
        PreparedImageState, ReplacementImageKey, ReplacementImageStateError,
        ReplacementImageStateOwner, ReplacementImageUse,
    },
    replacement_image_transition::{
        resolve_image_transitions, ImageTransitionResolveError, PreparedNativeImageState,
        ReplacementImageResolver,
    },
};
use ash::vk;
use reims_vgpu_protocol::TransactionId;

const PRESENT_CLEAR: [f32; 4] = [0.05, 0.06, 0.08, 1.0];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementPresentRecordPhase {
    ResetFence,
    ResetCommandBuffer,
    BeginCommandBuffer,
    EndCommandBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementPresentRecordError {
    EmptyDestination,
    CoordinateOverflow,
    Driver {
        phase: ReplacementPresentRecordPhase,
        result: vk::Result,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReplacementPresentBlitGeometry {
    source: [vk::Offset3D; 2],
    destination: [vk::Offset3D; 2],
    clear_destination: bool,
}

fn plan_present_blit_geometry(
    source: vk::Extent3D,
    destination: vk::Extent2D,
) -> Result<ReplacementPresentBlitGeometry, ReplacementPresentRecordError> {
    if destination.width == 0 || destination.height == 0 {
        return Err(ReplacementPresentRecordError::EmptyDestination);
    }
    let source_right = i32::try_from(source.width)
        .map_err(|_| ReplacementPresentRecordError::CoordinateOverflow)?;
    let source_bottom = i32::try_from(source.height)
        .map_err(|_| ReplacementPresentRecordError::CoordinateOverflow)?;
    let viewport = reims_vgpu_core::aspect_fit_viewport(
        (source.width, source.height),
        (destination.width, destination.height),
    );
    let destination_left =
        i32::try_from(viewport.x).map_err(|_| ReplacementPresentRecordError::CoordinateOverflow)?;
    let destination_top =
        i32::try_from(viewport.y).map_err(|_| ReplacementPresentRecordError::CoordinateOverflow)?;
    let destination_right = i32::try_from(
        viewport
            .x
            .checked_add(viewport.width)
            .ok_or(ReplacementPresentRecordError::CoordinateOverflow)?,
    )
    .map_err(|_| ReplacementPresentRecordError::CoordinateOverflow)?;
    let destination_bottom = i32::try_from(
        viewport
            .y
            .checked_add(viewport.height)
            .ok_or(ReplacementPresentRecordError::CoordinateOverflow)?,
    )
    .map_err(|_| ReplacementPresentRecordError::CoordinateOverflow)?;
    Ok(ReplacementPresentBlitGeometry {
        source: [
            vk::Offset3D::default(),
            vk::Offset3D {
                x: source_right,
                y: source_bottom,
                z: 1,
            },
        ],
        destination: [
            vk::Offset3D {
                x: destination_left,
                y: destination_top,
                z: 0,
            },
            vk::Offset3D {
                x: destination_right,
                y: destination_bottom,
                z: 1,
            },
        ],
        clear_destination: !viewport.covers((destination.width, destination.height)),
    })
}

/// Record one resident-to-swapchain blit into a PresentStream-owned slot.
/// Source layout transitions come from the shared image-state owner; the
/// acquired destination's prior contents are deliberately discarded.
///
/// # Safety
///
/// Every handle must belong to `device` and the exact live epoch. The source,
/// destination, command slot, and resolved barriers must remain alive until
/// the returned recording's queue timeline point completes.
pub unsafe fn record_present_blit(
    device: &ash::Device,
    recording: crate::replacement_queue::ReplacementPresentRecording,
    source: crate::replacement_image_transition::NativeImageTarget,
    transitions: &crate::replacement_image_transition::NativeImageUseTransitions,
    destination: vk::Image,
    destination_extent: vk::Extent2D,
) -> Result<crate::replacement_queue::ReplacementPresentRecording, ReplacementPresentRecordError> {
    if destination == vk::Image::null() {
        return Err(ReplacementPresentRecordError::EmptyDestination);
    }
    let geometry = plan_present_blit_geometry(source.extent, destination_extent)?;
    unsafe {
        device.reset_fences(&[recording.fence]).map_err(|result| {
            ReplacementPresentRecordError::Driver {
                phase: ReplacementPresentRecordPhase::ResetFence,
                result,
            }
        })?;
        device
            .reset_command_buffer(
                recording.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
            .map_err(|result| ReplacementPresentRecordError::Driver {
                phase: ReplacementPresentRecordPhase::ResetCommandBuffer,
                result,
            })?;
        device
            .begin_command_buffer(
                recording.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|result| ReplacementPresentRecordError::Driver {
                phase: ReplacementPresentRecordPhase::BeginCommandBuffer,
                result,
            })?;
        crate::replacement_barrier_record::record_hazard_barriers(
            device,
            recording.command_buffer,
            &transitions.before,
        );
        let color_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        device.cmd_pipeline_barrier(
            recording.command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(destination)
                .subresource_range(color_range)],
        );
        if geometry.clear_destination {
            device.cmd_clear_color_image(
                recording.command_buffer,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: PRESENT_CLEAR,
                },
                &[color_range],
            );
            device.cmd_pipeline_barrier(
                recording.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
                &[],
                &[],
            );
        }
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        device.cmd_blit_image(
            recording.command_buffer,
            source.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(layers)
                .src_offsets(geometry.source)
                .dst_subresource(layers)
                .dst_offsets(geometry.destination)],
            crate::translate::sampler::PRESENT_BLIT_FILTER,
        );
        crate::replacement_barrier_record::record_hazard_barriers(
            device,
            recording.command_buffer,
            &transitions.after,
        );
        device.cmd_pipeline_barrier(
            recording.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .image(destination)
                .subresource_range(color_range)],
        );
        device
            .end_command_buffer(recording.command_buffer)
            .map_err(|result| ReplacementPresentRecordError::Driver {
                phase: ReplacementPresentRecordPhase::EndCommandBuffer,
                result,
            })?;
    }
    Ok(recording)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementPresentImageStateError {
    State(ReplacementImageStateError),
    Native(ImageTransitionResolveError),
    TransitionShape,
}

pub fn prepare_present_image_state(
    owner: &mut ReplacementImageStateOwner,
    transaction: TransactionId,
    queue_family: u32,
    image: ReplacementImageKey,
) -> Result<PreparedImageState, ReplacementPresentImageStateError> {
    owner
        .prepare(
            transaction,
            queue_family,
            vec![ReplacementImageUse {
                image,
                required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                final_layout: vk::ImageLayout::GENERAL,
            }]
            .into_boxed_slice(),
        )
        .map_err(ReplacementPresentImageStateError::State)
}

pub fn resolve_present_image_state(
    prepared: &PreparedImageState,
    resolver: &impl ReplacementImageResolver,
) -> Result<PreparedNativeImageState, ReplacementPresentImageStateError> {
    let [transition] = prepared.transitions() else {
        return Err(ReplacementPresentImageStateError::TransitionShape);
    };
    if transition.required_usage != vk::ImageUsageFlags::TRANSFER_SRC
        || transition.use_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        || transition.final_layout != vk::ImageLayout::GENERAL
    {
        return Err(ReplacementPresentImageStateError::TransitionShape);
    }
    resolve_image_transitions(prepared, resolver).map_err(ReplacementPresentImageStateError::Native)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_image_state::{ReplacementImageSharing, ReplacementImageState};
    use crate::replacement_image_transition::NativeImageTarget;
    use ash::vk::Handle;
    use reims_vgpu_protocol::{BackingId, RepresentationId, VulkanDeviceEpochId};

    struct Resolver {
        key: ReplacementImageKey,
        target: NativeImageTarget,
    }

    impl ReplacementImageResolver for Resolver {
        fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
            (image == self.key).then_some(self.target)
        }
    }

    #[test]
    fn blit_geometry_clears_only_the_letterboxed_destination() {
        let letterboxed = plan_present_blit_geometry(
            vk::Extent3D {
                width: 100,
                height: 100,
                depth: 1,
            },
            vk::Extent2D {
                width: 200,
                height: 100,
            },
        )
        .unwrap();
        assert_eq!(
            letterboxed.destination,
            [
                vk::Offset3D { x: 50, y: 0, z: 0 },
                vk::Offset3D {
                    x: 150,
                    y: 100,
                    z: 1,
                },
            ]
        );
        assert!(letterboxed.clear_destination);

        let covered = plan_present_blit_geometry(
            vk::Extent3D {
                width: 200,
                height: 100,
                depth: 1,
            },
            vk::Extent2D {
                width: 200,
                height: 100,
            },
        )
        .unwrap();
        assert!(!covered.clear_destination);
    }

    #[test]
    fn blit_geometry_refuses_unrepresentable_coordinates() {
        assert_eq!(
            plan_present_blit_geometry(
                vk::Extent3D {
                    width: i32::MAX as u32 + 1,
                    height: 1,
                    depth: 1,
                },
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            ),
            Err(ReplacementPresentRecordError::CoordinateOverflow)
        );
        assert_eq!(
            plan_present_blit_geometry(
                vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                vk::Extent2D {
                    width: 0,
                    height: 1,
                },
            ),
            Err(ReplacementPresentRecordError::EmptyDestination)
        );
    }

    #[test]
    fn presentation_uses_the_shared_image_state_contract() {
        let epoch = VulkanDeviceEpochId::new(1);
        let key = ReplacementImageKey {
            backing: BackingId::new(2),
            representation: RepresentationId::new(3),
        };
        let mut owner = ReplacementImageStateOwner::new(epoch);
        owner
            .register(
                key,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 4 },
                    last_use: None,
                },
            )
            .unwrap();
        let prepared =
            prepare_present_image_state(&mut owner, TransactionId::new(5), 4, key).unwrap();
        let target = NativeImageTarget {
            image: vk::Image::from_raw(6),
            view: vk::ImageView::from_raw(7),
            image_type: vk::ImageType::TYPE_2D,
            full_range: vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
            usage: vk::ImageUsageFlags::TRANSFER_SRC,
            pixel_format: 80,
            extent: vk::Extent3D {
                width: 64,
                height: 32,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        };
        let native = resolve_present_image_state(&prepared, &Resolver { key, target }).unwrap();
        assert!(native.releases.is_empty());
        assert_eq!(native.transitions.before.images.len(), 1);
        assert_eq!(
            native.transitions.before.images[0].old_layout,
            vk::ImageLayout::GENERAL
        );
        assert_eq!(
            native.transitions.before.images[0].new_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
        assert_eq!(native.transitions.after.images.len(), 1);
        assert_eq!(
            native.transitions.after.images[0].new_layout,
            vk::ImageLayout::GENERAL
        );
    }
}
