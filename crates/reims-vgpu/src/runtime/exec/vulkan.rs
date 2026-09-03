//! Work this rail does before a submission may be consumed.
//!
//! Cold AIR translation is immutable CPU work with no protocol ownership, and
//! this rail wants it done before the packet is taken: a stream whose render or
//! compute pipelines are still translating is deferred rather than executed, so
//! a replay cannot duplicate clears, fences, dispatches or guest writeback.
//!
//! A rail that translates nothing has nothing to preflight, which is why the
//! whole of this is reached through
//! [`crate::backend::Backend::preflight_translations`] and `runtime::exec`
//! never names it.

use super::*;

/// `pipelines` is the **model's own lease list** for the packet, not a second
/// scan of its bytes.
///
/// The two used to be different answers to "which pipelines does this packet
/// bind": the walk collected `ResolvedOperation::pipeline_lease` and this
/// re-framed the stream looking for `SetPipeline` records. That difference is
/// load-bearing rather than cosmetic — admission readies *the walk's* leases on
/// *this* function's whole-submission verdict, so a pipeline the scan did not
/// reach was a lease declared ready on an answer that never examined it, which
/// is the shape of the `m2v_translation_pending_at_sync_boundary` loss measured
/// on a driven macos-15 desktop. Taking the list makes the two sets the same
/// set by construction, and there is nothing left to keep in step.
pub(crate) fn preflight_render_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipelines: &[u32],
    pending: &mut Vec<u32>,
) {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    for &pipeline_ref in pipelines {
        note_preflight_pipe();
        // The draw path's own memo already knows whether these two shaders are
        // translated, and answers for ~0.6 us against the 4.3 us of guest
        // resolves below. `translations_ready` states why that is not a weaker
        // answer — chiefly that the translate cache never evicts, so a shader
        // this memo saw translated is still translated.
        if crate::backend::vulkan::pipeline_resolve::translations_ready(
            state,
            host,
            task_id,
            pipeline_ref,
        ) {
            continue;
        }
        let air_started = std::time::Instant::now();
        // The MTLB containers, not owned copies of the AIR inside them: the two
        // `ensure_cached_async` calls below borrow, digest and drop, so copying
        // first would allocate twice per pipeline ref for bytes nothing keeps.
        let pair = draw::vulkan::load_render_mtlb_pair(state, host, task_id, pipeline_ref);
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Ok((v_mtlb, f_mtlb)) = pair else {
            // Normal execution emits the precise pipeline/MTLB failure. A
            // missing plan input is deterministic, not asynchronous work.
            //
            // Counted, because "not pending" is what this arm says and it is
            // the answer that readies the packet's pipeline leases at
            // admission. A draw has been measured reaching a shader that was
            // still translating while its lease read `ready`, and a pipeline
            // this pre-scan silently stopped examining is one of the two ways
            // that can happen — the other being a pipeline the scan never
            // reached at all. They were one silence.
            crate::runtime::drain::note_store_route("preflight_mtlb_unloadable");
            continue;
        };
        // A container whose AIR will not extract is the same "deterministic
        // missing plan input" as one that would not load: normal execution
        // reports it precisely, and there is no asynchronous work to await.
        let (Ok(v_air), Ok(f_air)) = (
            crate::runtime::mtlb::extract_air(&v_mtlb),
            crate::runtime::mtlb::extract_air(&f_mtlb),
        ) else {
            crate::runtime::drain::note_store_route("preflight_air_unextractable");
            continue;
        };
        let cache_started = std::time::Instant::now();
        if !crate::runtime::m2v_cache::ensure_cached_async(
            v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ) | !crate::runtime::m2v_cache::ensure_cached_async(
            f_air,
            metal2vulkan::passes::Stage::Fragment,
            pipeline_ref,
        ) {
            // `|` and not `||`: both stages must be *started*, so they
            // translate in parallel and the packet is retried once rather than
            // once per stage.
            if !pending.contains(&pipeline_ref) {
                pending.push(pipeline_ref);
            }
        }
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
    }
}

/// `dispatches` is the **model's own record of what this packet dispatches**,
/// not a second scan of its bytes.
///
/// The same correction the render arm took in the commit before this one, and
/// for the same reason: admission readies the walk's compute leases on this
/// function's whole-submission verdict, so a kernel the scan did not reach was
/// a lease declared ready on an answer that never examined it. The scan had
/// three ways not to reach one --- a selector carrying no threadgroup size, a
/// dispatch after an encoder continuation, and a framing refusal it swallowed
/// --- and [`reims_vgpu_core::exec::ExecWork::compute_dispatch_translations`]
/// has none of them, because it reads records the walk already resolved.
pub(crate) fn preflight_compute_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    dispatches: &[(u32, [u32; 3])],
    pending: &mut Vec<u32>,
) {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    for &(pipeline_ref, local_size) in dispatches {
        note_preflight_pipe();
        let air_started = std::time::Instant::now();
        let loaded = compute_exec::load_compute_pipeline(state, host, task_id, pipeline_ref)
            .and_then(|pipeline| {
                crate::runtime::mtlb::load_mtlb(
                    state,
                    host,
                    task_id,
                    pipeline.kernel_func_ref,
                    crate::runtime::mtlb::AirLoadRail::Compute,
                )
            });
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Some(mtlb) = loaded else {
            continue;
        };
        let Ok(air) = crate::runtime::mtlb::extract_air(&mtlb) else {
            continue;
        };
        let cache_started = std::time::Instant::now();
        let cached =
            crate::runtime::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref);
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
        if !cached && !pending.contains(&pipeline_ref) {
            pending.push(pipeline_ref);
        }
    }
}

/// [`crate::backend::Backend::preflight_translations`] for this rail: the
/// pipeline refs this packet binds that the rail does not hold a translation
/// for yet, deduplicated.
///
/// **The refs and not a bool**, because the caller has two different things to
/// do with the two halves of the answer. A lease this rail cannot serve must
/// stop being ready, and a lease it can serve must become ready; a single
/// "something is pending" cannot tell the caller which lease is which, so it
/// had to treat the whole packet as one — and the pipelines that *were* ready
/// stayed ready either way.
///
/// Every ref is examined, not just up to the first pending one: the point is
/// to *start* every cold translation this packet needs, so they proceed in
/// parallel and the packet is retried once rather than once per shader.
pub fn preflight_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    render_pipelines: &[u32],
    compute_dispatches: &[(u32, [u32; 3])],
) -> Vec<u32> {
    // Both halves are asked once for the whole packet, and both are asked
    // about what the walk resolved. Neither reads a stream: the packet's bytes
    // were read once, and a second reading is a second answer to "what does
    // this packet run".
    let mut pending = Vec::new();
    preflight_render_translations(state, host, task_id, render_pipelines, &mut pending);
    preflight_compute_translations(state, host, task_id, compute_dispatches, &mut pending);
    pending
}
