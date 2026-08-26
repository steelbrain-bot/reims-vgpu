//! Immutable function and compute-pipeline construction state.

use reims_vgpu_protocol::ComputeStageInputDescriptor;
use std::sync::Arc;

/// Construction state retained for one compute-pipeline generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoadedComputePipeline {
    pub kernel_func_ref: u32,
    pub kernel_mtlb: Arc<[u8]>,
    /// `None` inherits the native device limit; it is not a stated zero.
    pub max_total_threads_per_threadgroup: Option<u32>,
    pub supports_indirect_command_buffers: bool,
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

/// Immutable shader payload retained for the guest function lifetime.
#[derive(Debug)]
pub(crate) struct LoadedFunction {
    pub mtlb: Arc<[u8]>,
}
