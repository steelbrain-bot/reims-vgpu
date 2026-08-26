//! One operation-ordered image-state batch for every image use in an EXEC.

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
use reims_vgpu_core::{BackingRegion, TransferKey, GUEST_REPRESENTATION, HOST_REPRESENTATION};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecImageStateError {
    BlitState(ImageBlitStateError),
    BlitProgram(ImageBlitRecordError),
    Compute(ComputeImageStateError),
    Render(RenderImageStateError),
    ResourceStateEndpointAmbiguous(TransferKey),
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
    exec_has_image_uses_with_transfer_classifier(resources, |_| false)
}

pub fn exec_has_image_uses_with_transfer_classifier<
    Compute: ReplacementComputeImageBindings,
    NativeCompute,
    Render: ReplacementRenderImageBindings,
    NativeRender,
>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    mut linear_transfer_uses_image: impl FnMut(TransferKey) -> bool,
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
                        || linear_transfer_uses_image(*transfer)
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
                    || linear_transfer_uses_image(*transfer)
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
    mut linear_transfer_uses_image: impl FnMut(TransferKey) -> bool,
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
                linear_transfer_uses_image(transfer)
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
                linear_transfer_uses_image(transfer)
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
                state.is_some_and(|state| {
                    state.transitions().iter().any(|transition| {
                        transition.image.backing == transfer.backing
                            && (transition.image.representation == transfer.source
                                || transition.image.representation == transfer.destination)
                    })
                })
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
            |transfer| {
                state.is_some_and(|state| {
                    state.transitions().iter().any(|transition| {
                        transition.image.backing == transfer.backing
                            && (transition.image.representation == transfer.source
                                || transition.image.representation == transfer.destination)
                    })
                })
            },
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

fn derive_resource_state_image_uses(
    prepared: &reims_vgpu_core::PreparedResourceState,
    linear_transfer_uses_image: impl FnMut(TransferKey) -> bool,
) -> Result<Box<[crate::replacement_image_state::ReplacementImageUse]>, ExecImageStateError> {
    derive_resource_state_transfer_image_uses(prepared.transfers(), linear_transfer_uses_image)
}

fn derive_resource_state_transfer_image_uses(
    transfers: &[TransferKey],
    mut linear_transfer_uses_image: impl FnMut(TransferKey) -> bool,
) -> Result<Box<[crate::replacement_image_state::ReplacementImageUse]>, ExecImageStateError> {
    let mut roles =
        BTreeMap::<crate::replacement_image_state::ReplacementImageKey, (bool, bool)>::new();
    for &transfer in transfers {
        if !matches!(transfer.region, BackingRegion::Image(_))
            && !linear_transfer_uses_image(transfer)
        {
            continue;
        }
        let source_endpoint = matches!(transfer.source, GUEST_REPRESENTATION | HOST_REPRESENTATION);
        let destination_endpoint = matches!(
            transfer.destination,
            GUEST_REPRESENTATION | HOST_REPRESENTATION
        );
        match (source_endpoint, destination_endpoint) {
            (true, false) => {
                roles
                    .entry(crate::replacement_image_state::ReplacementImageKey {
                        backing: transfer.backing,
                        representation: transfer.destination,
                    })
                    .or_default()
                    .1 = true
            }
            (false, true) => {
                roles
                    .entry(crate::replacement_image_state::ReplacementImageKey {
                        backing: transfer.backing,
                        representation: transfer.source,
                    })
                    .or_default()
                    .0 = true
            }
            (false, false) => {
                roles
                    .entry(crate::replacement_image_state::ReplacementImageKey {
                        backing: transfer.backing,
                        representation: transfer.source,
                    })
                    .or_default()
                    .0 = true;
                roles
                    .entry(crate::replacement_image_state::ReplacementImageKey {
                        backing: transfer.backing,
                        representation: transfer.destination,
                    })
                    .or_default()
                    .1 = true;
            }
            (true, true) => {
                return Err(ExecImageStateError::ResourceStateEndpointAmbiguous(
                    transfer,
                ));
            }
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

    #[test]
    fn resource_state_image_roles_come_only_from_exact_transfer_endpoints() {
        let working = reims_vgpu_protocol::RepresentationId::new(9);
        let read = derive_resource_state_transfer_image_uses(
            &[image_transfer(working.get(), HOST_REPRESENTATION.get())],
            |_| false,
        )
        .unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].image.representation, working);
        assert_eq!(
            read[0].required_usage,
            ash::vk::ImageUsageFlags::TRANSFER_SRC
        );
        assert_eq!(
            read[0].use_layout,
            ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );

        let write = derive_resource_state_transfer_image_uses(
            &[image_transfer(GUEST_REPRESENTATION.get(), working.get())],
            |_| false,
        )
        .unwrap();
        assert_eq!(write[0].image.representation, working);
        assert_eq!(
            write[0].required_usage,
            ash::vk::ImageUsageFlags::TRANSFER_DST
        );
        assert_eq!(
            write[0].use_layout,
            ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );

        assert!(matches!(
            derive_resource_state_transfer_image_uses(
                &[image_transfer(
                    GUEST_REPRESENTATION.get(),
                    HOST_REPRESENTATION.get(),
                )],
                |_| false,
            ),
            Err(ExecImageStateError::ResourceStateEndpointAmbiguous(_))
        ));
    }
}
