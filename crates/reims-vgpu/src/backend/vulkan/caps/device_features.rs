//! Compatibility surface for extracted Vulkan feature negotiation.

pub use reims_vgpu_vulkan::device_features::*;

#[cfg(test)]
mod tests {
    use super::DeviceFeatures;

    #[test]
    fn enabled_multi_viewport_matches_the_request_shape_the_engine_binds() {
        let enabled = DeviceFeatures {
            multi_viewport: true,
            ..DeviceFeatures::default()
        }
        .enabled_features();
        assert_eq!(enabled.multi_viewport, ash::vk::TRUE);

        let viewport = |x: f32| crate::backend::vulkan::engine::ViewportResource {
            x,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let request = crate::backend::vulkan::engine::DrawRequest {
            viewports: vec![viewport(0.0), viewport(1.0)],
            ..Default::default()
        };
        assert_eq!(
            crate::backend::vulkan::engine::viewport_slot_count(&request),
            2
        );
    }
}
