//! Vulkan implementation policy for Reims vGPU.
//!
//! This crate is the only layer allowed to interpret host Vulkan capabilities
//! as placement and transfer choices. Guest-visible resource lifetime and
//! content authority remain in `reims-vgpu-core`.

pub mod api_floor;
pub mod capabilities;
pub mod device_features;
pub mod device_select;
pub mod engine;
pub mod format;
pub mod gpu_hang_trail;
pub mod host_pointer;
pub mod m2v_cache;
pub mod memory;
pub mod native_types;
pub mod policy;
pub mod preparation;
pub mod push_descriptor;
pub mod replacement_barrier_record;
pub mod replacement_barriers;
pub mod replacement_buffer_acceptance;
pub mod replacement_buffer_blit;
pub mod replacement_capabilities;
pub mod replacement_completion;
pub mod replacement_compute;
pub mod replacement_compute_compile;
pub mod replacement_console_present;
pub mod replacement_device_epoch;
pub mod replacement_epoch;
pub mod replacement_exec_acceptance;
pub mod replacement_exec_cancellation;
pub mod replacement_exec_image;
pub mod replacement_exec_queue;
pub mod replacement_exec_recording;
mod replacement_fill;
mod replacement_fill_shader;
pub mod replacement_image_acceptance;
pub mod replacement_image_blit;
pub mod replacement_image_release;
pub mod replacement_image_state;
pub mod replacement_image_transition;
pub mod replacement_indirect_exec_chain;
pub mod replacement_indirect_range;
pub mod replacement_info_acceptance;
pub mod replacement_info_query;
pub mod replacement_present;
pub mod replacement_queue;
pub mod replacement_recording;
pub mod replacement_recording_queue;
pub mod replacement_render;
pub mod replacement_render_compile;
pub mod replacement_replay;
pub mod replacement_representation;
pub mod replacement_resource_state;
pub mod replacement_resource_state_acceptance;
pub mod replacement_sampler;
pub mod replacement_submit;
#[cfg(feature = "host-window")]
pub mod replacement_window_present;
#[cfg(feature = "host-window")]
mod replacement_wsi;
pub mod spirv_bind;
pub mod spirv_vertex_input;
pub mod srgb_census;
pub mod telemetry;
pub mod translate;
