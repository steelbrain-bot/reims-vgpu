//! One operation-ordered image-state batch for every image use in an EXEC.

use crate::replacement_resource_state::TransferImageEndpoints;
use crate::{
    replacement_compute::{
        derive_compute_image_uses, validate_compute_image_state, ComputeImageStateError,
        ReplacementComputeImageBindings,
    },
    replacement_image_blit::{
        derive_image_uses, validate_exec_image_blit_state_subset, ImageBlitRecordError,
        ImageBlitStateError, ReplacementImageFinalLayout,
    },
    replacement_image_state::{
        PreparedImageStateBatch, ReplacementImageStateError, ReplacementImageStateOwner,
    },
    replacement_render::{
        derive_render_image_uses, validate_render_image_state, RenderImageStateError,
        ReplacementRenderImageBindings,
    },
};
use reims_vgpu_core::PreparedExecResources;
use reims_vgpu_core::{BackingRegion, TransferKey};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecImageStateError {
    BlitState(ImageBlitStateError),
    BlitProgram(ImageBlitRecordError),
    Compute(ComputeImageStateError),
    Render(RenderImageStateError),
    StateOperationMismatch,
    State(ReplacementImageStateError),
}

pub fn exec_has_image_uses<
    Compute: ReplacementComputeImageBindings,
    NativeCompute,
    Render: ReplacementRenderImageBindings,
    NativeRender,
>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
) -> Result<bool, ExecImageStateError> {
    exec_has_image_uses_with_transfer_classifier(resources, |_| TransferImageEndpoints::NEITHER)
}

pub fn exec_has_image_uses_with_transfer_classifier<
    Compute: ReplacementComputeImageBindings,
    NativeCompute,
    Render: ReplacementRenderImageBindings,
    NativeRender,
>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    mut transfer_image_endpoints: impl FnMut(TransferKey) -> TransferImageEndpoints,
) -> Result<bool, ExecImageStateError> {
    if !resources.inputs().image_blits.is_empty() {
        return Ok(true);
    }
    if resources
        .inputs()
        .resource_states
        .as_ref()
        .is_some_and(|states| {
            states.states().iter().any(|state| {
                state.transfers().iter().any(|transfer| {
                    matches!(transfer.region, BackingRegion::Image(_))
                        || transfer_image_endpoints(*transfer).any()
                })
            })
        })
    {
        return Ok(true);
    }
    if resources
        .inputs()
        .content_synchronization
        .as_ref()
        .is_some_and(|batch| {
            batch.transfers().iter().any(|transfer| {
                matches!(transfer.region, BackingRegion::Image(_))
                    || transfer_image_endpoints(*transfer).any()
            })
        })
    {
        return Ok(true);
    }
    for compute in &resources.inputs().compute_dispatches {
        if !derive_compute_image_uses(compute)
            .map_err(ExecImageStateError::Compute)?
            .is_empty()
        {
            return Ok(true);
        }
    }
    for render in &resources.inputs().render_dispatches {
        if !derive_render_image_uses(render)
            .map_err(ExecImageStateError::Render)?
            .is_empty()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn prepare_exec_image_states<
    Compute: ReplacementComputeImageBindings,
    NativeCompute,
    Render: ReplacementRenderImageBindings,
    NativeRender,
>(
    owner: &mut ReplacementImageStateOwner,
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    queue_family: u32,
    final_layouts: &impl ReplacementImageFinalLayout,
    mut transfer_image_endpoints: impl FnMut(TransferKey) -> TransferImageEndpoints,
) -> Result<PreparedImageStateBatch, ExecImageStateError> {
    let mut operations = Vec::new();
    for prepared in &resources.inputs().image_blits {
        let index = prepared
            .write()
            .operation_index()
            .ok_or(ExecImageStateError::StateOperationMismatch)?;
        let uses = derive_image_uses(
            prepared.operation(),
            prepared.representations(),
            final_layouts,
        )
        .map_err(ExecImageStateError::BlitState)?;
        operations.push((index, uses));
    }
    if let Some(states) = resources.inputs().resource_states.as_ref() {
        for prepared in states.states() {
            let uses = derive_resource_state_image_uses(prepared, |transfer| {
                transfer_image_endpoints(transfer)
            })?;
            if !uses.is_empty() {
                operations.push((prepared.index(), uses));
            }
        }
    }
    for prepared in &resources.inputs().compute_dispatches {
        let uses = derive_compute_image_uses(prepared).map_err(ExecImageStateError::Compute)?;
        if !uses.is_empty() {
            operations.push((prepared.operation_index(), uses));
        }
    }
    for prepared in &resources.inputs().render_dispatches {
        let uses = derive_render_image_uses(prepared).map_err(ExecImageStateError::Render)?;
        if !uses.is_empty() {
            operations.push((prepared.operation_index(), uses));
        }
    }
    let content_synchronization = resources
        .inputs()
        .content_synchronization
        .as_ref()
        .map(|batch| {
            derive_resource_state_transfer_image_uses(batch.transfers(), |transfer| {
                transfer_image_endpoints(transfer)
            })
        })
        .transpose()?
        .unwrap_or_default();
    operations.sort_unstable_by_key(|(index, _)| *index);
    owner
        .prepare_batch_with_auxiliary_tail(
            resources.transaction(),
            queue_family,
            operations.into_boxed_slice(),
            content_synchronization,
        )
        .map_err(ExecImageStateError::State)
}

pub fn validate_exec_image_states<
    Compute: ReplacementComputeImageBindings,
    NativeCompute,
    Render: ReplacementRenderImageBindings,
    NativeRender,
>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    states: &PreparedImageStateBatch,
) -> Result<(), ExecImageStateError> {
    validate_exec_image_blit_state_subset(resources, states)
        .map_err(ExecImageStateError::BlitProgram)?;
    let mut expected = resources.inputs().image_blits.len();
    if let Some(resource_states) = resources.inputs().resource_states.as_ref() {
        for prepared in resource_states.states() {
            let state = states
                .operations()
                .iter()
                .find(|state| state.operation_index() == Some(prepared.index()));
            let uses = derive_resource_state_image_uses(prepared, |transfer| {
                transitioned_endpoints(state, transfer)
            })?;
            if uses.is_empty() {
                if state.is_some() {
                    return Err(ExecImageStateError::StateOperationMismatch);
                }
            } else {
                expected += 1;
                validate_resource_state_image_uses(
                    &uses,
                    state.ok_or(ExecImageStateError::StateOperationMismatch)?,
                )?;
            }
        }
    }
    if let Some(content_synchronization) = resources.inputs().content_synchronization.as_ref() {
        let state = states
            .operations()
            .iter()
            .find(|state| state.operation_index().is_none());
        let uses = derive_resource_state_transfer_image_uses(
            content_synchronization.transfers(),
            |transfer| transitioned_endpoints(state, transfer),
        )?;
        if uses.is_empty() {
            if state.is_some() {
                return Err(ExecImageStateError::StateOperationMismatch);
            }
        } else {
            expected += 1;
            validate_resource_state_image_uses(
                &uses,
                state.ok_or(ExecImageStateError::StateOperationMismatch)?,
            )?;
        }
    }
    for prepared in &resources.inputs().compute_dispatches {
        let uses = derive_compute_image_uses(prepared).map_err(ExecImageStateError::Compute)?;
        let state = states
            .operations()
            .iter()
            .find(|state| state.operation_index() == Some(prepared.operation_index()));
        if uses.is_empty() {
            if state.is_some() {
                return Err(ExecImageStateError::StateOperationMismatch);
            }
        } else {
            expected += 1;
            validate_compute_image_state(prepared, state).map_err(ExecImageStateError::Compute)?;
        }
    }
    for prepared in &resources.inputs().render_dispatches {
        let uses = derive_render_image_uses(prepared).map_err(ExecImageStateError::Render)?;
        let state = states
            .operations()
            .iter()
            .find(|state| state.operation_index() == Some(prepared.operation_index()));
        if uses.is_empty() {
            if state.is_some() {
                return Err(ExecImageStateError::StateOperationMismatch);
            }
        } else {
            expected += 1;
            let state = state.ok_or(ExecImageStateError::StateOperationMismatch)?;
            validate_render_image_state(prepared, state).map_err(ExecImageStateError::Render)?;
        }
    }
    if resources.transaction() != states.transaction() || states.operations().len() != expected {
        return Err(ExecImageStateError::StateOperationMismatch);
    }
    Ok(())
}

/// Which endpoints of a transfer a prepared batch already transitioned.
///
/// Validation reads the batch rather than the registry, because what it has to
/// agree with is the batch that was built --- and an image the batch
/// transitioned is an image whatever the registry says about it now.
fn transitioned_endpoints(
    state: Option<&crate::replacement_image_state::PreparedImageState>,
    transfer: TransferKey,
) -> TransferImageEndpoints {
    let transitioned = |representation| {
        state.is_some_and(|state| {
            state.transitions().iter().any(|transition| {
                transition.image.backing == transfer.backing
                    && transition.image.representation == representation
            })
        })
    };
    TransferImageEndpoints {
        source: transitioned(transfer.source),
        destination: transitioned(transfer.destination),
    }
}

fn derive_resource_state_image_uses(
    prepared: &reims_vgpu_core::PreparedResourceState,
    transfer_image_endpoints: impl FnMut(TransferKey) -> TransferImageEndpoints,
) -> Result<Box<[crate::replacement_image_state::ReplacementImageUse]>, ExecImageStateError> {
    derive_resource_state_transfer_image_uses(prepared.transfers(), transfer_image_endpoints)
}

fn derive_resource_state_transfer_image_uses(
    transfers: &[TransferKey],
    mut transfer_image_endpoints: impl FnMut(TransferKey) -> TransferImageEndpoints,
) -> Result<Box<[crate::replacement_image_state::ReplacementImageUse]>, ExecImageStateError> {
    let mut roles =
        BTreeMap::<crate::replacement_image_state::ReplacementImageKey, (bool, bool)>::new();
    for &transfer in transfers {
        // Which side is an image is what the endpoints resolve to, and only
        // the classifier can answer it. A byte view of a backing is a
        // designated representation like any other --- reading "not the shared
        // endpoint identity, therefore an image" registers a buffer as an
        // image, and the batch then refuses to prepare a state for something
        // that has none.
        let endpoints = transfer_image_endpoints(transfer);
        if endpoints.source {
            roles
                .entry(crate::replacement_image_state::ReplacementImageKey {
                    backing: transfer.backing,
                    representation: transfer.source,
                })
                .or_default()
                .0 = true;
        }
        if endpoints.destination {
            roles
                .entry(crate::replacement_image_state::ReplacementImageKey {
                    backing: transfer.backing,
                    representation: transfer.destination,
                })
                .or_default()
                .1 = true;
        }
    }
    Ok(roles
        .into_iter()
        .map(|(image, (source, destination))| {
            let (required_usage, use_layout) = match (source, destination) {
                (true, false) => (
                    ash::vk::ImageUsageFlags::TRANSFER_SRC,
                    ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ),
                (false, true) => (
                    ash::vk::ImageUsageFlags::TRANSFER_DST,
                    ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
                (true, true) => (
                    ash::vk::ImageUsageFlags::TRANSFER_SRC | ash::vk::ImageUsageFlags::TRANSFER_DST,
                    ash::vk::ImageLayout::GENERAL,
                ),
                (false, false) => unreachable!(),
            };
            crate::replacement_image_state::ReplacementImageUse {
                image,
                required_usage,
                use_layout,
                final_layout: ash::vk::ImageLayout::GENERAL,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn validate_resource_state_image_uses(
    uses: &[crate::replacement_image_state::ReplacementImageUse],
    state: &crate::replacement_image_state::PreparedImageState,
) -> Result<(), ExecImageStateError> {
    if uses.len() != state.transitions().len()
        || uses.iter().any(|use_| {
            !state.transitions().iter().any(|transition| {
                transition.image == use_.image
                    && transition.required_usage == use_.required_usage
                    && transition.use_layout == use_.use_layout
                    && transition.final_layout == use_.final_layout
            })
        })
    {
        return Err(ExecImageStateError::StateOperationMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_transfer(source: u64, destination: u64) -> TransferKey {
        TransferKey {
            backing: reims_vgpu_protocol::BackingId::new(7),
            region: BackingRegion::Image(reims_vgpu_core::ImageRegion {
                aspect: reims_vgpu_core::ImageAspect::Color,
                mip: 1,
                layer: 2,
                texels: reims_vgpu_core::TexelBox::new([0, 0, 0], [4, 4, 1]).unwrap(),
            }),
            version: reims_vgpu_protocol::ContentVersion::new(3),
            source: reims_vgpu_protocol::RepresentationId::new(source),
            destination: reims_vgpu_protocol::RepresentationId::new(destination),
        }
    }

    /// The image roles a transfer contributes come from what its endpoints
    /// resolve to, not from which identity they carry.
    ///
    /// The rule used to be "an endpoint that is not the shared byte identity
    /// is the image", and a backing may designate a byte view of its own ---
    /// an ordinary representation with an ordinary identity and a buffer
    /// behind it. A transfer from an image into that view then registered it
    /// as an image, and the batch refused to prepare a state for something
    /// that has none, parking the channel it sat on.
    #[test]
    fn resource_state_image_roles_come_from_what_the_endpoints_resolve_to() {
        let image = reims_vgpu_protocol::RepresentationId::new(9);
        let bytes = reims_vgpu_protocol::RepresentationId::new(10);
        let endpoints = |transfer: TransferKey| TransferImageEndpoints {
            source: transfer.source == image,
            destination: transfer.destination == image,
        };

        let read =
            derive_resource_state_transfer_image_uses(&[image_transfer(9, 10)], endpoints).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].image.representation, image);
        assert_eq!(
            read[0].required_usage,
            ash::vk::ImageUsageFlags::TRANSFER_SRC
        );
        assert_eq!(
            read[0].use_layout,
            ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );

        let write =
            derive_resource_state_transfer_image_uses(&[image_transfer(10, 9)], endpoints).unwrap();
        assert_eq!(write.len(), 1);
        assert_eq!(write[0].image.representation, image);
        assert_eq!(
            write[0].required_usage,
            ash::vk::ImageUsageFlags::TRANSFER_DST
        );
        assert_eq!(
            write[0].use_layout,
            ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );

        // The byte view is never one of them, whichever end of the copy it is
        // on, and a copy between two byte views contributes nothing at all.
        assert!(read
            .iter()
            .chain(write.iter())
            .all(|use_| use_.image.representation != bytes));
        assert!(
            derive_resource_state_transfer_image_uses(&[image_transfer(10, 11)], endpoints)
                .unwrap()
                .is_empty()
        );
    }
}
