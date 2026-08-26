//! Device-epoch-owned sampler handles retained by native operation sidecars.

use ash::vk;
use reims_vgpu_core::{SamplerResource, SamplerSource};
use reims_vgpu_protocol::{ComputePipelineObject, RenderPipelineObject, ResourceId, SamplerObject};
use std::sync::{Arc, Mutex};

/// Device lifetime needed to retire one sampler after its last recording use.
pub trait ReplacementSamplerDevice: Send + Sync {
    fn destroy_replacement_sampler(&self, sampler: vk::Sampler);
}

impl ReplacementSamplerDevice for crate::engine::context::SharedDeviceContext {
    fn destroy_replacement_sampler(&self, sampler: vk::Sampler) {
        unsafe { self.device.destroy_sampler(sampler, None) }
    }
}

/// One native sampler and the exact Vulkan device incarnation that created it.
///
/// Resolvers return an `Arc` lease rather than a bare handle. Native render and
/// compute programs retain that lease until their recording ownership retires,
/// so deletion of the guest sampler object cannot invalidate queued work.
pub struct ReplacementSampler {
    handle: vk::Sampler,
    context: Arc<dyn ReplacementSamplerDevice>,
}

impl std::fmt::Debug for ReplacementSampler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplacementSampler")
            .field("handle", &self.handle)
            .finish()
    }
}

impl ReplacementSampler {
    pub fn new(context: Arc<dyn ReplacementSamplerDevice>, handle: vk::Sampler) -> Self {
        Self { handle, context }
    }

    pub const fn handle(&self) -> vk::Sampler {
        self.handle
    }
}

impl Drop for ReplacementSampler {
    fn drop(&mut self) {
        self.context.destroy_replacement_sampler(self.handle);
    }
}

pub type ReplacementSamplerLease = Arc<ReplacementSampler>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementSamplerCreateError {
    DynamicSourceRequired,
    StaticSourceRequired,
    DynamicIdentityMissing,
    SamplerState(crate::native_types::SamplerStateDecline),
    AnisotropyUnsupported { requested: u32, supported_bits: u32 },
    MirrorClampToEdgeUnsupported,
    Vulkan(vk::Result),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplacementSamplerCensus {
    pub live: usize,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Default)]
struct ReplacementSamplerRegistryState {
    dynamic: Vec<(
        ResourceId<SamplerObject>,
        crate::native_types::SamplerStateKey,
        ReplacementSamplerLease,
    )>,
    static_: Vec<(
        ResourceId<RenderPipelineObject>,
        u32,
        crate::native_types::SamplerStateKey,
        ReplacementSamplerLease,
    )>,
    static_compute: Vec<(
        ResourceId<ComputePipelineObject>,
        u32,
        crate::native_types::SamplerStateKey,
        ReplacementSamplerLease,
    )>,
    hits: u64,
    misses: u64,
}

/// Unbounded sampler realizations whose entries die with their contract owner.
///
/// A dynamic entry is owned by its exact generational sampler object. Multiple
/// LOD-override states may coexist for that object because the bind command is
/// part of the sampler contract. A constexpr entry is owned by the exact render
/// pipeline generation and descriptor binding. Neither class evicts a live
/// entry; deletion of its owner removes all lookup entries while outstanding
/// recording leases continue to keep the native handles alive.
pub struct ReplacementSamplerRegistry {
    context: Arc<crate::engine::context::SharedDeviceContext>,
    state: Mutex<ReplacementSamplerRegistryState>,
}

impl ReplacementSamplerRegistry {
    pub(crate) fn new(context: Arc<crate::engine::context::SharedDeviceContext>) -> Self {
        Self {
            context,
            state: Mutex::new(ReplacementSamplerRegistryState::default()),
        }
    }

    pub fn dynamic(
        &self,
        sampler: &SamplerResource,
    ) -> Result<ReplacementSamplerLease, ReplacementSamplerCreateError> {
        if sampler.source != SamplerSource::State {
            return Err(ReplacementSamplerCreateError::DynamicSourceRequired);
        }
        let identity = sampler
            .identity
            .ok_or(ReplacementSamplerCreateError::DynamicIdentityMissing)?;
        let key = crate::native_types::effective_sampler_state(sampler)
            .map_err(ReplacementSamplerCreateError::SamplerState)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some((_, _, lease)) = state
            .dynamic
            .iter()
            .find(|(candidate, candidate_key, _)| *candidate == identity && *candidate_key == key)
        {
            let lease = lease.clone();
            state.hits = state.hits.saturating_add(1);
            return Ok(lease);
        }
        let lease = self.create(key)?;
        state.misses = state.misses.saturating_add(1);
        state.dynamic.push((identity, key, lease.clone()));
        Ok(lease)
    }

    pub fn render(
        &self,
        pipeline: ResourceId<RenderPipelineObject>,
        sampler: &SamplerResource,
    ) -> Result<ReplacementSamplerLease, ReplacementSamplerCreateError> {
        match sampler.source {
            SamplerSource::State => self.dynamic(sampler),
            SamplerSource::Static => self.static_sampler(pipeline, sampler),
            SamplerSource::Null => Err(ReplacementSamplerCreateError::DynamicSourceRequired),
        }
    }

    pub fn static_sampler(
        &self,
        pipeline: ResourceId<RenderPipelineObject>,
        sampler: &SamplerResource,
    ) -> Result<ReplacementSamplerLease, ReplacementSamplerCreateError> {
        if sampler.source != SamplerSource::Static || sampler.identity.is_some() {
            return Err(ReplacementSamplerCreateError::StaticSourceRequired);
        }
        let key = crate::native_types::effective_sampler_state(sampler)
            .map_err(ReplacementSamplerCreateError::SamplerState)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some((_, _, _, lease)) =
            state
                .static_
                .iter()
                .find(|(candidate, binding, candidate_key, _)| {
                    *candidate == pipeline && *binding == sampler.binding && *candidate_key == key
                })
        {
            let lease = lease.clone();
            state.hits = state.hits.saturating_add(1);
            return Ok(lease);
        }
        let lease = self.create(key)?;
        state.misses = state.misses.saturating_add(1);
        state
            .static_
            .push((pipeline, sampler.binding, key, lease.clone()));
        Ok(lease)
    }

    pub fn compute(
        &self,
        pipeline: ResourceId<ComputePipelineObject>,
        sampler: &SamplerResource,
    ) -> Result<ReplacementSamplerLease, ReplacementSamplerCreateError> {
        match sampler.source {
            SamplerSource::State => self.dynamic(sampler),
            SamplerSource::Null => Err(ReplacementSamplerCreateError::DynamicSourceRequired),
            SamplerSource::Static => {
                if sampler.identity.is_some() {
                    return Err(ReplacementSamplerCreateError::StaticSourceRequired);
                }
                let key = crate::native_types::effective_sampler_state(sampler)
                    .map_err(ReplacementSamplerCreateError::SamplerState)?;
                let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
                if let Some((_, _, _, lease)) =
                    state
                        .static_compute
                        .iter()
                        .find(|(candidate, binding, candidate_key, _)| {
                            *candidate == pipeline
                                && *binding == sampler.binding
                                && *candidate_key == key
                        })
                {
                    let lease = lease.clone();
                    state.hits = state.hits.saturating_add(1);
                    return Ok(lease);
                }
                let lease = self.create(key)?;
                state.misses = state.misses.saturating_add(1);
                state
                    .static_compute
                    .push((pipeline, sampler.binding, key, lease.clone()));
                Ok(lease)
            }
        }
    }

    pub fn retire_dynamic(&self, sampler: ResourceId<SamplerObject>) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .dynamic
            .retain(|(candidate, _, _)| *candidate != sampler);
    }

    pub fn retire_render_pipeline(&self, pipeline: ResourceId<RenderPipelineObject>) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .static_
            .retain(|(candidate, _, _, _)| *candidate != pipeline);
    }

    pub fn retire_compute_pipeline(&self, pipeline: ResourceId<ComputePipelineObject>) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .static_compute
            .retain(|(candidate, _, _, _)| *candidate != pipeline);
    }

    pub fn census(&self) -> ReplacementSamplerCensus {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        ReplacementSamplerCensus {
            live: state.dynamic.len() + state.static_.len() + state.static_compute.len(),
            hits: state.hits,
            misses: state.misses,
        }
    }

    fn create(
        &self,
        key: crate::native_types::SamplerStateKey,
    ) -> Result<ReplacementSamplerLease, ReplacementSamplerCreateError> {
        let max_anisotropy = admitted_max_anisotropy(
            key.max_anisotropy,
            self.context.sampler_anisotropy,
            self.context.max_sampler_anisotropy,
        )?;
        let uses_mirror_clamp = [key.address_mode_u, key.address_mode_v, key.address_mode_w]
            .contains(&reims_vgpu_core::SamplerAddressMode::MirrorClampToEdge);
        if uses_mirror_clamp && !self.context.features.mirror_clamp_to_edge.is_available() {
            return Err(ReplacementSamplerCreateError::MirrorClampToEdgeUnsupported);
        }
        let not_mipmapped = key.mip_filter == reims_vgpu_core::SamplerMipFilter::NotMipmapped;
        let (min_lod, max_lod) = if key.unnormalized_coordinates || not_mipmapped {
            (0.0, 0.0)
        } else {
            (f32::from_bits(key.lod_min), f32::from_bits(key.lod_max))
        };
        let address_uses_zero = [key.address_mode_u, key.address_mode_v, key.address_mode_w]
            .contains(&reims_vgpu_core::SamplerAddressMode::ClampToZero);
        let create = vk::SamplerCreateInfo::default()
            .mag_filter(crate::translate::sampler::vk_filter(key.mag_filter))
            .min_filter(crate::translate::sampler::vk_filter(key.min_filter))
            .mipmap_mode(crate::translate::sampler::vk_mipmap_mode(key.mip_filter))
            .address_mode_u(crate::translate::sampler::vk_address_mode(
                key.address_mode_u,
            ))
            .address_mode_v(crate::translate::sampler::vk_address_mode(
                key.address_mode_v,
            ))
            .address_mode_w(crate::translate::sampler::vk_address_mode(
                key.address_mode_w,
            ))
            .anisotropy_enable(key.max_anisotropy > 1)
            .max_anisotropy(max_anisotropy)
            .compare_enable(key.compare_function != reims_vgpu_core::SamplerCompareFunction::Never)
            .compare_op(crate::translate::raster::vk_compare_op(
                key.compare_function,
            ))
            .min_lod(min_lod)
            .max_lod(max_lod)
            .border_color(
                crate::translate::sampler::vk_border_color_with_clamp_to_zero(
                    key.border_color,
                    address_uses_zero,
                ),
            )
            .unnormalized_coordinates(key.unnormalized_coordinates);
        let handle = unsafe { self.context.device.create_sampler(&create, None) }
            .map_err(ReplacementSamplerCreateError::Vulkan)?;
        let context: Arc<dyn ReplacementSamplerDevice> = self.context.clone();
        Ok(Arc::new(ReplacementSampler::new(context, handle)))
    }
}

fn admitted_max_anisotropy(
    requested: u32,
    sampler_anisotropy: bool,
    device_limit: f32,
) -> Result<f32, ReplacementSamplerCreateError> {
    let supported = if sampler_anisotropy {
        device_limit
    } else {
        1.0
    };
    if requested as f32 > supported {
        return Err(ReplacementSamplerCreateError::AnisotropyUnsupported {
            requested,
            supported_bits: supported.to_bits(),
        });
    }
    Ok(requested as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug)]
    struct Device(AtomicU32);

    impl ReplacementSamplerDevice for Device {
        fn destroy_replacement_sampler(&self, sampler: vk::Sampler) {
            assert_eq!(sampler.as_raw(), 17);
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn sampler_handle_dies_only_after_the_last_recording_lease() {
        let device = Arc::new(Device(AtomicU32::new(0)));
        let sampler = Arc::new(ReplacementSampler::new(
            device.clone(),
            vk::Sampler::from_raw(17),
        ));
        let recording = sampler.clone();
        drop(sampler);
        assert_eq!(device.0.load(Ordering::Relaxed), 0);
        drop(recording);
        assert_eq!(device.0.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn anisotropy_admission_never_clamps_the_guest_request() {
        assert_eq!(admitted_max_anisotropy(4, true, 8.0), Ok(4.0));
        assert_eq!(
            admitted_max_anisotropy(2, false, 16.0),
            Err(ReplacementSamplerCreateError::AnisotropyUnsupported {
                requested: 2,
                supported_bits: 1.0f32.to_bits(),
            })
        );
        assert_eq!(
            admitted_max_anisotropy(8, true, 4.0),
            Err(ReplacementSamplerCreateError::AnisotropyUnsupported {
                requested: 8,
                supported_bits: 4.0f32.to_bits(),
            })
        );
    }
}
