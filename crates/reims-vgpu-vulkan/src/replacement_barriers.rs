//! Synchronization2 barrier projection from exact semantic hazard operands.
//!
//! This stage retains canonical backing identities; native buffer/image handles
//! are resolved only by the queue-recording worker that owns them. Vulkan
//! barriers cannot restrict an image dependency to a texel box, so exact texel
//! hazards conservatively synchronize the intersecting mip/layer subresources.

use ash::vk;
use reims_vgpu_core::{
    AccessIntent, AccessMode, AccessScope, AccessTarget, HazardRequirement, ImageAspect,
    ImageSubresourceRange, LinearRange, StageScope,
};
use reims_vgpu_protocol::{BackingId, RenderStages};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BarrierAccess {
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackingBarrierScope {
    Linear(LinearRange),
    Image {
        aspect: ImageAspect,
        mip_start: u32,
        mip_count: u32,
        layer_start: u32,
        layer_count: u32,
    },
    WholeBacking,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierTarget {
    Global,
    Backing {
        backing: BackingId,
        scope: BackingBarrierScope,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HazardBarrier {
    pub producer: reims_vgpu_protocol::TransactionId,
    pub consumer: reims_vgpu_protocol::TransactionId,
    pub source: BarrierAccess,
    pub destination: BarrierAccess,
    pub target: BarrierTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierPlanError {
    NonConflictingRequirement,
    MismatchedBacking,
}

pub fn plan_hazard_barriers(
    requirements: &[HazardRequirement],
) -> Result<Box<[HazardBarrier]>, BarrierPlanError> {
    requirements
        .iter()
        .map(plan_hazard_barrier)
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

pub fn plan_hazard_barrier(
    requirement: &HazardRequirement,
) -> Result<HazardBarrier, BarrierPlanError> {
    if !requirement
        .earlier
        .mode
        .conflicts_with(requirement.later.mode)
    {
        return Err(BarrierPlanError::NonConflictingRequirement);
    }
    Ok(HazardBarrier {
        producer: requirement.edge.older,
        consumer: requirement.edge.newer,
        source: barrier_access(requirement.earlier),
        destination: barrier_access(requirement.later),
        target: barrier_target(requirement.earlier, requirement.later)?,
    })
}

fn barrier_access(intent: AccessIntent) -> BarrierAccess {
    BarrierAccess {
        stages: stage_flags(intent.stages),
        access: access_flags(intent.mode, intent.stages),
    }
}

pub fn stage_flags(stages: StageScope) -> vk::PipelineStageFlags2 {
    match stages {
        StageScope::All | StageScope::Unknown => vk::PipelineStageFlags2::ALL_COMMANDS,
        StageScope::Vertex => vk::PipelineStageFlags2::VERTEX_SHADER,
        StageScope::Fragment => vk::PipelineStageFlags2::FRAGMENT_SHADER,
        // Tile, object, and mesh stages require a capability-specific native
        // projection. ALL_COMMANDS is the Vulkan-defined conservative scope and
        // does not claim an unsupported stage feature.
        StageScope::Tile | StageScope::Object | StageScope::Mesh => {
            vk::PipelineStageFlags2::ALL_COMMANDS
        }
        StageScope::Compute => vk::PipelineStageFlags2::COMPUTE_SHADER,
        StageScope::Indirect => vk::PipelineStageFlags2::DRAW_INDIRECT,
        StageScope::VertexInput => vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
        StageScope::IndexInput => vk::PipelineStageFlags2::INDEX_INPUT,
        StageScope::ColorAttachment => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        StageScope::DepthStencilAttachment => {
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
        }
        StageScope::QueryResolve => vk::PipelineStageFlags2::COPY,
        StageScope::Render(stages) => render_stage_flags(stages),
        StageScope::Blit => vk::PipelineStageFlags2::TRANSFER,
        StageScope::Host => vk::PipelineStageFlags2::HOST,
    }
}

fn render_stage_flags(stages: RenderStages) -> vk::PipelineStageFlags2 {
    let bits = stages.bits();
    if bits & (RenderStages::TILE | RenderStages::OBJECT | RenderStages::MESH) != 0 {
        return vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    let mut flags = vk::PipelineStageFlags2::empty();
    if bits & RenderStages::VERTEX != 0 {
        flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if bits & RenderStages::FRAGMENT != 0 {
        flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if flags.is_empty() {
        vk::PipelineStageFlags2::ALL_COMMANDS
    } else {
        flags
    }
}

fn access_flags(mode: AccessMode, stages: StageScope) -> vk::AccessFlags2 {
    if mode == AccessMode::Read && stages == StageScope::Indirect {
        return vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }
    if mode == AccessMode::Read && stages == StageScope::VertexInput {
        return vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }
    if mode == AccessMode::Read && stages == StageScope::IndexInput {
        return vk::AccessFlags2::INDEX_READ;
    }
    if stages == StageScope::ColorAttachment {
        return match mode {
            AccessMode::Read => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            AccessMode::Write => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            AccessMode::ReadWrite | AccessMode::Unknown => {
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            }
        };
    }
    if stages == StageScope::DepthStencilAttachment {
        return match mode {
            AccessMode::Read => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            AccessMode::Write => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            AccessMode::ReadWrite | AccessMode::Unknown => {
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
        };
    }
    if mode == AccessMode::Write && stages == StageScope::QueryResolve {
        return vk::AccessFlags2::TRANSFER_WRITE;
    }
    match mode {
        AccessMode::Read => vk::AccessFlags2::MEMORY_READ,
        AccessMode::Write => vk::AccessFlags2::MEMORY_WRITE,
        AccessMode::ReadWrite | AccessMode::Unknown => {
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
        }
    }
}

fn barrier_target(
    earlier: AccessIntent,
    later: AccessIntent,
) -> Result<BarrierTarget, BarrierPlanError> {
    if matches!(
        earlier.scope,
        AccessScope::WholeDomain | AccessScope::WholeHeap
    ) || matches!(
        later.scope,
        AccessScope::WholeDomain | AccessScope::WholeHeap
    ) {
        return Ok(BarrierTarget::Global);
    }
    let (Some(AccessTarget::Backing(earlier_backing)), Some(AccessTarget::Backing(later_backing))) =
        (earlier.target, later.target)
    else {
        return Ok(BarrierTarget::Global);
    };
    if earlier_backing != later_backing {
        return Err(BarrierPlanError::MismatchedBacking);
    }
    let scope = match (earlier.scope, later.scope) {
        (AccessScope::WholeBacking, _) | (_, AccessScope::WholeBacking) => {
            BackingBarrierScope::WholeBacking
        }
        (AccessScope::Linear(earlier), AccessScope::Linear(later)) => {
            let start = earlier.start().max(later.start());
            let end = earlier.end().min(later.end());
            let range = LinearRange::new(start, end.saturating_sub(start))
                .ok_or(BarrierPlanError::NonConflictingRequirement)?;
            BackingBarrierScope::Linear(range)
        }
        (AccessScope::Image(earlier), AccessScope::Image(later)) => image_scope(earlier, later)?,
        _ => BackingBarrierScope::WholeBacking,
    };
    Ok(BarrierTarget::Backing {
        backing: earlier_backing,
        scope,
    })
}

fn image_scope(
    earlier: ImageSubresourceRange,
    later: ImageSubresourceRange,
) -> Result<BackingBarrierScope, BarrierPlanError> {
    if !earlier.overlaps(later) {
        return Err(BarrierPlanError::NonConflictingRequirement);
    }
    let mip_start = earlier.mip_start().max(later.mip_start());
    let mip_end = earlier.mip_end().min(later.mip_end());
    let layer_start = earlier.layer_start().max(later.layer_start());
    let layer_end = earlier.layer_end().min(later.layer_end());
    Ok(BackingBarrierScope::Image {
        aspect: earlier.aspect,
        mip_start,
        mip_count: mip_end - mip_start,
        layer_start,
        layer_count: layer_end - layer_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::{HazardCause, HazardEdge};
    use reims_vgpu_protocol::{HazardDomainId, IngressOrdinal, ResourceId, ResourceObject};

    fn intent(scope: AccessScope, mode: AccessMode, stages: StageScope) -> AccessIntent {
        AccessIntent::for_backing(
            HazardDomainId::new(1),
            BackingId::new(4),
            Some(ResourceId::<ResourceObject>::new(2, 1)),
            scope,
            mode,
            stages,
        )
        .unwrap()
    }

    fn requirement(earlier: AccessIntent, later: AccessIntent) -> HazardRequirement {
        HazardRequirement {
            edge: HazardEdge {
                newer: reims_vgpu_protocol::TransactionId::new(2),
                older: reims_vgpu_protocol::TransactionId::new(1),
                newer_ordinal: IngressOrdinal::new(2),
                older_ordinal: IngressOrdinal::new(1),
                cause: HazardCause::Buffer,
            },
            earlier,
            later,
        }
    }

    #[test]
    fn linear_barrier_uses_only_the_intersection_and_exact_stage_direction() {
        let earlier = intent(
            AccessScope::Linear(LinearRange::new(0, 96).unwrap()),
            AccessMode::Write,
            StageScope::Compute,
        );
        let later = intent(
            AccessScope::Linear(LinearRange::new(64, 64).unwrap()),
            AccessMode::Read,
            StageScope::Vertex,
        );
        let barrier = plan_hazard_barrier(&requirement(earlier, later)).unwrap();
        assert_eq!(
            barrier.source.stages,
            vk::PipelineStageFlags2::COMPUTE_SHADER
        );
        assert_eq!(barrier.source.access, vk::AccessFlags2::MEMORY_WRITE);
        assert_eq!(
            barrier.destination.stages,
            vk::PipelineStageFlags2::VERTEX_SHADER
        );
        assert_eq!(barrier.destination.access, vk::AccessFlags2::MEMORY_READ);
        assert_eq!(
            barrier.target,
            BarrierTarget::Backing {
                backing: BackingId::new(4),
                scope: BackingBarrierScope::Linear(LinearRange::new(64, 32).unwrap()),
            }
        );
    }

    #[test]
    fn indirect_argument_reads_use_the_fixed_function_stage_and_access() {
        let earlier = intent(
            AccessScope::Linear(LinearRange::new(0, 12).unwrap()),
            AccessMode::Write,
            StageScope::Compute,
        );
        let later = intent(
            AccessScope::Linear(LinearRange::new(0, 12).unwrap()),
            AccessMode::Read,
            StageScope::Indirect,
        );
        let barrier = plan_hazard_barrier(&requirement(earlier, later)).unwrap();
        assert_eq!(
            barrier.destination.stages,
            vk::PipelineStageFlags2::DRAW_INDIRECT
        );
        assert_eq!(
            barrier.destination.access,
            vk::AccessFlags2::INDIRECT_COMMAND_READ
        );
    }

    #[test]
    fn vertex_and_index_reads_use_their_fixed_function_stages_and_accesses() {
        for (scope, stage, access) in [
            (
                StageScope::VertexInput,
                vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            ),
            (
                StageScope::IndexInput,
                vk::PipelineStageFlags2::INDEX_INPUT,
                vk::AccessFlags2::INDEX_READ,
            ),
        ] {
            let earlier = intent(
                AccessScope::Linear(LinearRange::new(0, 16).unwrap()),
                AccessMode::Write,
                StageScope::Compute,
            );
            let later = intent(
                AccessScope::Linear(LinearRange::new(0, 16).unwrap()),
                AccessMode::Read,
                scope,
            );
            let barrier = plan_hazard_barrier(&requirement(earlier, later)).unwrap();
            assert_eq!(barrier.destination.stages, stage);
            assert_eq!(barrier.destination.access, access);
        }
    }

    #[test]
    fn attachment_accesses_use_color_output_and_depth_stencil_tests() {
        for (scope, stages, access) in [
            (
                StageScope::ColorAttachment,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            (
                StageScope::DepthStencilAttachment,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
        ] {
            let earlier = intent(
                AccessScope::Image(
                    ImageSubresourceRange::new(ImageAspect::Color, 0, 1, 0, 1, None).unwrap(),
                ),
                AccessMode::Write,
                StageScope::Compute,
            );
            let later = AccessIntent {
                mode: AccessMode::ReadWrite,
                stages: scope,
                ..earlier
            };
            let barrier = plan_hazard_barrier(&requirement(earlier, later)).unwrap();
            assert_eq!(barrier.destination.stages, stages);
            assert_eq!(barrier.destination.access, access);
        }
    }

    #[test]
    fn query_result_writes_use_the_native_copy_stage_and_access() {
        let earlier = intent(
            AccessScope::Linear(LinearRange::new(0, 8).unwrap()),
            AccessMode::Write,
            StageScope::QueryResolve,
        );
        let later = intent(
            AccessScope::Linear(LinearRange::new(0, 8).unwrap()),
            AccessMode::Read,
            StageScope::Compute,
        );
        let barrier = plan_hazard_barrier(&requirement(earlier, later)).unwrap();
        assert_eq!(barrier.source.stages, vk::PipelineStageFlags2::COPY);
        assert_eq!(barrier.source.access, vk::AccessFlags2::TRANSFER_WRITE);
    }

    #[test]
    fn image_texel_hazard_projects_to_the_intersecting_subresources() {
        let image = |mip_start, mip_count, layer_start, layer_count| {
            AccessScope::Image(
                ImageSubresourceRange::new(
                    ImageAspect::Color,
                    mip_start,
                    mip_count,
                    layer_start,
                    layer_count,
                    None,
                )
                .unwrap(),
            )
        };
        let barrier = plan_hazard_barrier(&requirement(
            intent(image(0, 3, 0, 4), AccessMode::Write, StageScope::Fragment),
            intent(image(2, 2, 3, 2), AccessMode::Read, StageScope::Fragment),
        ))
        .unwrap();
        assert_eq!(
            barrier.target,
            BarrierTarget::Backing {
                backing: BackingId::new(4),
                scope: BackingBarrierScope::Image {
                    aspect: ImageAspect::Color,
                    mip_start: 2,
                    mip_count: 1,
                    layer_start: 3,
                    layer_count: 1,
                },
            }
        );
    }

    #[test]
    fn unmapped_render_stages_and_unknown_access_remain_conservative() {
        let render = RenderStages::from_bits(RenderStages::TILE as u16).unwrap();
        let barrier = plan_hazard_barrier(&requirement(
            intent(
                AccessScope::WholeBacking,
                AccessMode::Unknown,
                StageScope::Render(render),
            ),
            intent(
                AccessScope::WholeBacking,
                AccessMode::Write,
                StageScope::Mesh,
            ),
        ))
        .unwrap();
        assert_eq!(barrier.source.stages, vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(
            barrier.source.access,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
        );
        assert_eq!(
            barrier.destination.stages,
            vk::PipelineStageFlags2::ALL_COMMANDS
        );
    }
}
