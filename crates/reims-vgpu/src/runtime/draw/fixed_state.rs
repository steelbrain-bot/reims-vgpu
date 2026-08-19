//! Semantic fixed-function draw state.
//!
//! Raw guest ordinals terminate here. Execution receives complete blend,
//! depth, and stencil state or a typed preparation refusal.

use super::{depth_chain_identity, DrawEncodeRequest, DrawPreparationDecline};

pub(super) fn depth_stencil_descriptor_is_trivial(
    descriptor: &crate::runtime::decode::resource::DepthStencilDescriptor,
) -> bool {
    const MTL_COMPARE_ALWAYS: u32 = 7;
    descriptor.depth_compare_function == MTL_COMPARE_ALWAYS
        && !descriptor.depth_write_enabled
        && !descriptor.front_stencil_enabled
        && !descriptor.back_stencil_enabled
}

fn stencil_face(
    face: &crate::runtime::decode::resource::DepthStencilFace,
) -> Result<reims_vgpu_core::StencilFaceOps, reims_vgpu_protocol::PipelineStateDecodeError> {
    Ok(reims_vgpu_core::StencilFaceOps {
        compare: reims_vgpu_protocol::compare_function(face.compare_function)?,
        fail_op: reims_vgpu_protocol::stencil_operation(face.stencil_failure_operation)?,
        depth_fail_op: reims_vgpu_protocol::stencil_operation(face.depth_failure_operation)?,
        pass_op: reims_vgpu_protocol::stencil_operation(face.depth_stencil_pass_operation)?,
        read_mask: face.read_mask,
        write_mask: face.write_mask,
    })
}

pub(super) fn semantic_blend_state(
    attachment: &reims_vgpu_protocol::resource::PipelineColorAttachment,
    constants: [f32; 4],
) -> Result<reims_vgpu_core::BlendStateResource, reims_vgpu_protocol::PipelineStateDecodeError> {
    reims_vgpu_protocol::blend_state(attachment, constants)
}

pub(super) fn semantic_blend_states(
    pipeline: &reims_vgpu_protocol::resource::RenderPipelineDescriptor,
    constants: [f32; 4],
) -> Result<Vec<(u32, reims_vgpu_core::BlendStateResource)>, DrawPreparationDecline> {
    pipeline
        .color_attachments
        .iter()
        .filter(|attachment| attachment.blending_enabled)
        .map(|attachment| {
            semantic_blend_state(attachment, constants)
                .map(|state| (attachment.slot, state))
                .map_err(|reason| DrawPreparationDecline::BlendState { reason })
        })
        .collect()
}

pub(super) fn semantic_depth_state(
    descriptor: &crate::runtime::decode::resource::DepthStencilDescriptor,
    request: &DrawEncodeRequest,
) -> Result<Option<reims_vgpu_core::DepthState>, DrawPreparationDecline> {
    use reims_vgpu_core::{SamplerCompareFunction, StencilFaceOps, StencilOp, StencilState};

    if depth_stencil_descriptor_is_trivial(descriptor) {
        return Ok(None);
    }
    let compare = reims_vgpu_protocol::compare_function(descriptor.depth_compare_function)
        .map_err(|reason| DrawPreparationDecline::DepthCompare { reason })?;
    let (clear_value, load_action) = request
        .depth_attach
        .as_ref()
        .map(|depth| (depth.clear_depth as f32, depth.load_action))
        .unwrap_or((1.0, reims_vgpu_protocol::pass_action::LoadAction::Clear));
    let stencil = if descriptor.front_stencil_enabled || descriptor.back_stencil_enabled {
        const PASS_THROUGH: StencilFaceOps = StencilFaceOps {
            compare: SamplerCompareFunction::Always,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        };
        let front = if descriptor.front_stencil_enabled {
            stencil_face(&descriptor.front_face).map_err(|reason| {
                DrawPreparationDecline::StencilState {
                    face: "front",
                    reason,
                }
            })?
        } else {
            PASS_THROUGH
        };
        let back = if descriptor.back_stencil_enabled {
            stencil_face(&descriptor.back_face).map_err(|reason| {
                DrawPreparationDecline::StencilState {
                    face: "back",
                    reason,
                }
            })?
        } else {
            PASS_THROUGH
        };
        let (reference_front, reference_back) = request.stencil_ref.unwrap_or((0, 0));
        let clear_value = request
            .stencil_attach
            .as_ref()
            .map(|stencil| stencil.clear_stencil)
            .unwrap_or(0);
        Some(StencilState {
            front,
            back,
            reference_front,
            reference_back,
            clear_value,
        })
    } else {
        None
    };

    Ok(Some(reims_vgpu_core::DepthState {
        identity: depth_chain_identity(request, stencil.is_some()),
        test_enable: true,
        write_enable: descriptor.depth_write_enabled,
        compare,
        clear_value,
        load: load_action == reims_vgpu_protocol::pass_action::LoadAction::Load,
        stencil,
    }))
}
