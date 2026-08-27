//! Host-window and shader-translation services owned by replacement execution.
//!
//! These contracts contain no compatibility executor or process-global Vulkan
//! state. Native lifetimes remain behind opaque values and every service is
//! supplied by the exact replacement session that consumes it.

use reims_vgpu_protocol::StorageImageFormat;
use std::sync::Arc;

pub(crate) type RenderTranslationDecline =
    reims_vgpu_core::DrawPreparationDecline<reims_vgpu_vulkan::m2v_cache::M2vCacheDecline>;

#[cfg(feature = "host-window")]
#[derive(Clone, Copy, Debug)]
pub struct WindowPresentationFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub seq: u64,
    pub payload: WindowPresentationPayload<'a>,
}

#[cfg(feature = "host-window")]
#[derive(Clone, Copy, Debug)]
pub enum WindowPresentationPayload<'a> {
    CpuBgra(&'a [u8]),
    Resident(&'a reims_vgpu_core::PreparedPresentation),
}

#[cfg(feature = "host-window")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowPresentOutcome {
    Busy,
    Presented {
        route: reims_vgpu_core::PresentationRoute,
        width: u32,
        height: u32,
        swapchain_images: usize,
        suboptimal: bool,
    },
}

#[cfg(feature = "host-window")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowPresentationError {
    service: &'static str,
    detail: String,
}

#[cfg(feature = "host-window")]
impl WindowPresentationError {
    pub(crate) fn replacement(service: &'static str, detail: String) -> Self {
        Self { service, detail }
    }
}

#[cfg(feature = "host-window")]
impl std::fmt::Display for WindowPresentationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "replacement window presentation refused: {}: {}",
            self.service, self.detail
        )
    }
}

#[cfg(feature = "host-window")]
impl std::error::Error for WindowPresentationError {}

#[cfg(feature = "host-window")]
impl crate::observe::Decline for WindowPresentationError {
    fn slug(&self) -> &'static str {
        "replacement_window_presentation_refused"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("service", self.service.to_string()),
            ("detail", self.detail.clone()),
        ]
    }
}

#[cfg(feature = "host-window")]
#[derive(Debug)]
pub struct WindowPresentationScope;

#[cfg(feature = "host-window")]
pub trait WindowPresentationService: std::fmt::Debug + Send + Sync {
    fn enter_window_presentation(&self) -> WindowPresentationScope {
        WindowPresentationScope
    }

    fn attach_window_presenter(
        &self,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<(), WindowPresentationError>;

    fn resize_window_presenter(&self, width: u32, height: u32);

    fn present_window_frame(
        &self,
        frame: Option<WindowPresentationFrame<'_>>,
    ) -> Result<WindowPresentOutcome, WindowPresentationError>;

    fn detach_window_presenter(&self);
}

/// Backend-owned translated compute module exposed only through semantic facts.
pub trait ComputeTranslation: std::fmt::Debug + Send + Sync {
    fn interface(&self) -> &reims_vgpu_core::ShaderInterface;
    fn static_threadgroup_memory_length(&self) -> Option<u64>;
    fn used_descriptor_bindings(&self) -> Arc<[u32]>;
    fn buffer_extent(
        &self,
        metal_index: u32,
        workgroups: [u32; 3],
        local_size: [u32; 3],
    ) -> Option<u64>;
    fn storage_image_access(&self, binding: u32) -> Option<reims_vgpu_core::StorageImageAccess>;
    fn samplers(&self) -> Arc<[reims_vgpu_core::ReflectedSamplerDescriptor]>;
    /// Select the exact translated kernel this dispatch needs.
    ///
    /// `samplers` are the dispatch's bound sampler states. A pixel-coordinate
    /// sampler changes what an image operation in the kernel means, so the
    /// module is specialized against them the same way it is against runtime
    /// storage-image formats.
    fn prepare_program(
        &self,
        requests: &[(u32, StorageImageFormat)],
        samplers: &[reims_vgpu_core::SamplerResource],
        dispatch: reims_vgpu_core::ComputeProgramDispatchContract,
    ) -> Result<PreparedComputeProgram, ComputeProgramDecline>;
}

#[derive(Debug)]
pub struct PreparedComputeProgram {
    pub stage: reims_vgpu_core::PreparedShaderStage,
    pub dispatch: reims_vgpu_core::ComputeProgramDispatchContract,
    pub(crate) _native_lifetime: Box<dyn std::any::Any + Send + Sync>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeProgramDecline {
    Specialization(reims_vgpu_vulkan::m2v_cache::M2vCacheDecline),
}

impl crate::observe::Decline for ComputeProgramDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Specialization(decline) => crate::observe::Decline::slug(decline),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Specialization(decline) => crate::observe::Decline::fields(decline),
        }
    }
}

crate::observe::decline_display!(ComputeProgramDecline);

impl std::error::Error for ComputeProgramDecline {}
