//! Complete sampler provisioning for one resolved draw.
//!
//! Stream binds and reflected sampler declarations share one binding namespace.
//! This planner owns that occupancy relation, applies per-bind LOD overrides,
//! records diagnostic provenance, and refuses collisions before execution.

use super::*;

pub(super) struct SamplerPlan {
    pub(super) resources: Vec<reims_vgpu_core::SamplerResource>,
    pub(super) provenance: std::collections::BTreeMap<u32, u8>,
}

pub(super) fn plan_samplers<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    vertex: &reims_vgpu_core::PreparedShaderVariant,
    fragment: &reims_vgpu_core::PreparedShaderVariant,
) -> Result<SamplerPlan, DrawPreparationDecline> {
    let mut resources = Vec::new();
    let mut occupied = std::collections::BTreeSet::new();
    // A translated guest sampler and a synthesized sampler can have identical
    // semantic state, so provenance remains separate observation metadata.
    let mut provenance = std::collections::BTreeMap::new();

    {
        let mut push_stream = |index: u32,
                               sampler_ref: u32,
                               lod_clamp: Option<(u32, u32)>,
                               stage: reims_vgpu_core::ShaderStage|
         -> Result<(), DrawPreparationDecline> {
            let variant = match stage {
                reims_vgpu_core::ShaderStage::Vertex => vertex,
                reims_vgpu_core::ShaderStage::Fragment => fragment,
                reims_vgpu_core::ShaderStage::Unknown => unreachable!("draw stage is known"),
            };
            let binding = variant.sampler_binding(index);
            if !occupied.insert(binding) {
                crate::runtime::drain::note_store_route("sampler_bind_collided");
                return Err(DrawPreparationDecline::SamplerBindingCollision {
                    stage,
                    index,
                    binding,
                    source: reims_vgpu_core::SamplerBindingSource::Stream,
                });
            }
            let mut sampler = if sampler_ref != 0 {
                provenance.insert(binding, b'g');
                load_vulkan_sampler(state, host, req.task_id, sampler_ref, binding)?
            } else {
                provenance.insert(binding, b'd');
                reims_vgpu_core::SamplerResource::normalized_default(binding)
            };
            if let Some((min_bits, max_bits)) = lod_clamp {
                sampler.lod_min = min_bits;
                sampler.lod_max = max_bits;
            }
            resources.push(sampler);
            Ok(())
        };

        let _span = crate::runtime::sampled_phase::Span::open(
            crate::runtime::sampled_phase::Part::Samplers,
        );
        for sampler in req
            .vertex_samplers
            .iter()
            .filter(|sampler| sampler.sampler_ref != 0)
        {
            push_stream(
                sampler.index,
                sampler.sampler_ref,
                sampler.lod_clamp,
                reims_vgpu_core::ShaderStage::Vertex,
            )?;
        }
        for sampler in req
            .fragment_samplers
            .iter()
            .filter(|sampler| sampler.sampler_ref != 0)
        {
            push_stream(
                sampler.index,
                sampler.sampler_ref,
                sampler.lod_clamp,
                reims_vgpu_core::ShaderStage::Fragment,
            )?;
        }
    }

    let _span =
        crate::runtime::sampled_phase::Span::open(crate::runtime::sampled_phase::Part::Reflect);
    for (variant, stage) in [
        (vertex, reims_vgpu_core::ShaderStage::Vertex),
        (fragment, reims_vgpu_core::ShaderStage::Fragment),
    ] {
        for reflected in variant.samplers.iter() {
            if !occupied.insert(reflected.binding) {
                crate::runtime::drain::note_store_route("sampler_bind_collided");
                return Err(DrawPreparationDecline::SamplerBindingCollision {
                    stage,
                    index: reflected.metal_index,
                    binding: reflected.binding,
                    source: reims_vgpu_core::SamplerBindingSource::Reflected,
                });
            }
            let binding = reflected.binding;
            if let Some(static_state) = reflected.static_state {
                provenance.insert(binding, b'c');
                resources.push(reflected_static_sampler_resource(
                    stage.name(),
                    binding,
                    static_state,
                )?);
            } else {
                provenance.insert(binding, b'd');
                resources.push(reims_vgpu_core::SamplerResource::normalized_default(
                    binding,
                ));
            }
        }
    }

    Ok(SamplerPlan {
        resources,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn variant(index: u32, binding: u32) -> reims_vgpu_core::PreparedShaderVariant {
        reims_vgpu_core::PreparedShaderVariant {
            program: Default::default(),
            samplers: Arc::from([reims_vgpu_core::ReflectedSamplerDescriptor {
                metal_index: index,
                binding,
                static_state: None,
            }]),
            declared_bindings: Arc::from([]),
            descriptor_uses: Arc::from([]),
            texture_uses: Arc::from([]),
            buffer_binding_offset: 0,
            sampled_binding_offset: 0,
            texture_binding_base: 32,
            sampler_binding_base: 64,
            word_count: 0,
        }
    }

    #[test]
    fn reflected_sampler_collision_refuses_the_complete_plan() {
        let mut state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        let mut host = crate::runtime::host::FakeHost::new();
        let request = DrawEncodeRequest::default();

        let result = plan_samplers(
            &mut state,
            &mut host,
            &request,
            &variant(1, 64),
            &variant(2, 64),
        );
        assert!(matches!(
            result,
            Err(DrawPreparationDecline::SamplerBindingCollision {
                stage: reims_vgpu_core::ShaderStage::Fragment,
                index: 2,
                binding: 64,
                source: reims_vgpu_core::SamplerBindingSource::Reflected,
            })
        ));
    }
}
