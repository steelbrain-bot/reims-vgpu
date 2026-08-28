//! Replacement composition: decode guest contracts once, resolve semantic
//! identities, and hand immutable work to the per-device Vulkan epoch.

pub(crate) mod contract_census;
pub mod decode;
pub(crate) mod fifo_packet;
pub mod gpa_map;
pub mod guest_ram;
pub mod guest_ram_map;
pub mod gva_mem;
pub mod gva_refusal;
pub mod heap_query;
pub mod host;
pub(crate) mod host_action_census;
pub mod icb;
pub mod input;

pub(crate) mod replacement_air;
pub(crate) mod replacement_child_packet;
pub(crate) mod replacement_compute_projection;
pub(crate) mod replacement_compute_state;
pub(crate) mod replacement_coordinator;
pub(crate) mod replacement_display_descriptor;
#[path = "exec/replacement_decode.rs"]
pub(crate) mod replacement_exec_decode;
pub(crate) mod replacement_exec_support;
pub(crate) mod replacement_fifo_control;
pub(crate) mod replacement_mapper;
pub(crate) mod replacement_object_lifecycle;
pub(crate) mod replacement_packet;
pub(crate) mod replacement_pipeline_contract;
pub(crate) mod replacement_render_projection;
pub(crate) mod replacement_sampler_projection;
pub(crate) mod replacement_scanout;
pub(crate) mod replacement_services;
pub(crate) mod replacement_session;
pub(crate) mod replacement_task;
pub(crate) mod replacement_transport;

pub(crate) use host::HostAction;
