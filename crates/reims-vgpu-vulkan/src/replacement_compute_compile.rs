//! Device-incarnation compiler for replacement compute pipeline variants.

use crate::{
    engine::context::SharedDeviceContext,
    replacement_compute::{
        ReplacementComputePipeline, ReplacementComputePipelineFamily,
        ReplacementComputePipelinePlan, ReplacementComputePipelineVariant,
        ReplacementComputePipelineVariantKey,
    },
};
use ash::vk;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

#[derive(Clone)]
pub struct ReplacementComputePipelineCompiler {
    context: Arc<SharedDeviceContext>,
    lifetime: reims_vgpu_core::VulkanDeviceEpoch,
}

impl ReplacementComputePipelineCompiler {
    pub(crate) fn new(
        context: Arc<SharedDeviceContext>,
        lifetime: reims_vgpu_core::VulkanDeviceEpoch,
    ) -> Self {
        Self { context, lifetime }
    }

    pub fn compile_family_job(
        &self,
        family: &ReplacementComputePipelineFamily<ReplacementComputePipelineCompileError>,
        job: reims_vgpu_core::PipelineVariantCompileJob<ReplacementComputePipelineVariantKey>,
        plan: ReplacementComputePipelinePlan,
    ) -> Result<
        reims_vgpu_core::PipelineVariantPublication<ReplacementComputePipelineVariant>,
        ReplacementComputeFamilyCompileError,
    > {
        match self.lifetime.with_active((job, plan), |(job, plan)| {
            let native =
                match catch_compile(|| unsafe { compile_variant(Arc::clone(&self.context), plan) })
                {
                    Ok(native) => native,
                    Err(reason) => {
                        let waiters = family
                            .refuse(job, reason)
                            .map_err(ReplacementComputeFamilyCompileError::Lifecycle)?;
                        return Err(ReplacementComputeFamilyCompileError::Refused {
                            reason,
                            waiters,
                        });
                    }
                };
            family
                .compile_complete(job, native)
                .map_err(ReplacementComputeFamilyCompileError::Lifecycle)
        }) {
            Ok(result) => result,
            Err((job, _)) => {
                let reason = ReplacementComputePipelineCompileError::DeviceLifetimeClosed;
                let waiters = family
                    .refuse(job, reason)
                    .map_err(ReplacementComputeFamilyCompileError::Lifecycle)?;
                Err(ReplacementComputeFamilyCompileError::Refused { reason, waiters })
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplacementComputeFamilyCompileError {
    Refused {
        reason: ReplacementComputePipelineCompileError,
        waiters: Box<[reims_vgpu_protocol::TransactionId]>,
    },
    Lifecycle(reims_vgpu_core::PipelineVariantLifecycleError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementComputePipelineCompileError {
    DeviceLifetimeClosed,
    WorkerPanicked,
    ProgramUnavailable,
    ProgramMismatch,
    ProgramIsNotCompute,
    DescriptorTypeUnsupported(i32),
    DescriptorCountZero(u32),
    DescriptorArrayUnsupported(u32),
    Vulkan {
        operation: &'static str,
        result: vk::Result,
    },
}

fn catch_compile<T>(
    compile: impl FnOnce() -> Result<T, ReplacementComputePipelineCompileError>,
) -> Result<T, ReplacementComputePipelineCompileError> {
    catch_unwind(AssertUnwindSafe(compile))
        .unwrap_or(Err(ReplacementComputePipelineCompileError::WorkerPanicked))
}

unsafe fn compile_variant(
    context: Arc<SharedDeviceContext>,
    plan: ReplacementComputePipelinePlan,
) -> Result<ReplacementComputePipelineVariant, ReplacementComputePipelineCompileError> {
    let shader = crate::m2v_cache::resolve_prepared_shader(plan.program.id)
        .ok_or(ReplacementComputePipelineCompileError::ProgramUnavailable)?;
    if crate::m2v_cache::prepared_stage(&shader) != plan.program {
        return Err(ReplacementComputePipelineCompileError::ProgramMismatch);
    }
    if !is_compute_module(&shader.words) {
        return Err(ReplacementComputePipelineCompileError::ProgramIsNotCompute);
    }
    validate_descriptors(&context, &plan)?;

    let module = context
        .device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&shader.words),
            None,
        )
        .map_err(|result| vk_error("create_compute_shader_module", result))?;
    let result = create_handles(&context, &plan, &shader, module);
    context.device.destroy_shader_module(module, None);
    result.map(|native| {
        let owner: Arc<dyn crate::replacement_compute::ReplacementComputePipelineDevice> = context;
        ReplacementComputePipelineVariant::new(owner, native)
    })
}

fn is_compute_module(words: &[u32]) -> bool {
    // SPIR-V OpEntryPoint has opcode 15 and its first operand is ExecutionModel;
    // GLCompute is the contract-defined ordinal 5.
    let mut cursor = 5usize;
    while cursor < words.len() {
        let instruction = words[cursor];
        let word_count = (instruction >> 16) as usize;
        let opcode = instruction & 0xffff;
        if word_count == 0 || cursor.saturating_add(word_count) > words.len() {
            return false;
        }
        if opcode == 15 {
            return word_count > 1 && words[cursor + 1] == 5;
        }
        cursor += word_count;
    }
    false
}

fn descriptor_type(raw: i32) -> Option<vk::DescriptorType> {
    [
        vk::DescriptorType::STORAGE_BUFFER,
        vk::DescriptorType::SAMPLED_IMAGE,
        vk::DescriptorType::STORAGE_IMAGE,
        vk::DescriptorType::SAMPLER,
    ]
    .into_iter()
    .find(|ty| ty.as_raw() == raw)
}

fn validate_descriptors(
    context: &SharedDeviceContext,
    plan: &ReplacementComputePipelinePlan,
) -> Result<(), ReplacementComputePipelineCompileError> {
    let mut sampled = 0u32;
    let mut storage = 0u32;
    for &(binding, raw, count) in plan.descriptors.iter() {
        let ty = descriptor_type(raw)
            .ok_or(ReplacementComputePipelineCompileError::DescriptorTypeUnsupported(raw))?;
        if count == 0 {
            return Err(ReplacementComputePipelineCompileError::DescriptorCountZero(
                binding,
            ));
        }
        if count > 1 {
            let supported = match ty {
                vk::DescriptorType::SAMPLED_IMAGE => {
                    context.features.sampled_image_array_dynamic_indexing
                }
                vk::DescriptorType::STORAGE_IMAGE => {
                    context.features.storage_image_array_dynamic_indexing
                }
                _ => true,
            };
            if !supported {
                return Err(
                    ReplacementComputePipelineCompileError::DescriptorArrayUnsupported(binding),
                );
            }
        }
        match ty {
            vk::DescriptorType::SAMPLED_IMAGE => {
                sampled = sampled.checked_add(count).ok_or(
                    ReplacementComputePipelineCompileError::DescriptorArrayUnsupported(binding),
                )?
            }
            vk::DescriptorType::STORAGE_IMAGE => {
                storage = storage.checked_add(count).ok_or(
                    ReplacementComputePipelineCompileError::DescriptorArrayUnsupported(binding),
                )?
            }
            _ => {}
        }
    }
    if sampled > context.features.sampled_image_descriptor_limit {
        return Err(ReplacementComputePipelineCompileError::DescriptorArrayUnsupported(0));
    }
    if storage > context.features.storage_image_descriptor_limit {
        return Err(ReplacementComputePipelineCompileError::DescriptorArrayUnsupported(0));
    }
    Ok(())
}

unsafe fn create_handles(
    context: &SharedDeviceContext,
    plan: &ReplacementComputePipelinePlan,
    shader: &crate::m2v_cache::ShaderVariant,
    module: vk::ShaderModule,
) -> Result<ReplacementComputePipeline, ReplacementComputePipelineCompileError> {
    let descriptor_bindings = plan
        .descriptors
        .iter()
        .map(|&(binding, raw, count)| {
            Ok(vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type(raw).ok_or(
                    ReplacementComputePipelineCompileError::DescriptorTypeUnsupported(raw),
                )?)
                .descriptor_count(count)
                .stage_flags(vk::ShaderStageFlags::COMPUTE))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flags = plan
        .descriptors
        .iter()
        .map(|&(_, _, count)| {
            if count > 1 && context.features.descriptor_binding_partially_bound {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
            } else {
                vk::DescriptorBindingFlags::empty()
            }
        })
        .collect::<Vec<_>>();
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let descriptor_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&descriptor_bindings)
        .push_next(&mut binding_flags);
    let descriptor_set_layout = context
        .device
        .create_descriptor_set_layout(&descriptor_info, None)
        .map_err(|result| vk_error("create_compute_descriptor_set_layout", result))?;

    let push = shader.kernel_grid.map(|range| {
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(range.offset)
            .size(range.size)
    });
    let push_ranges = push.as_slice();
    let pipeline_layout = match context.device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[descriptor_set_layout])
            .push_constant_ranges(push_ranges),
        None,
    ) {
        Ok(layout) => layout,
        Err(result) => {
            context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            return Err(vk_error("create_compute_pipeline_layout", result));
        }
    };
    let entry = crate::engine::context::main_entry();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let create = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout)];
    let pipeline =
        match context
            .device
            .create_compute_pipelines(context.pipeline_cache, &create, None)
        {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, result)) => {
                for pipeline in pipelines {
                    context.device.destroy_pipeline(pipeline, None);
                }
                context
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
                context
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                return Err(vk_error("create_compute_pipeline", result));
            }
        };
    Ok(ReplacementComputePipeline {
        pipeline,
        layout: pipeline_layout,
        descriptor_set_layout,
        thread_grid_push_offset: shader.kernel_grid.map(|range| range.offset),
    })
}

fn vk_error(operation: &'static str, result: vk::Result) -> ReplacementComputePipelineCompileError {
    ReplacementComputePipelineCompileError::Vulkan { operation, result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::context::DeviceContext;
    use ash::vk::Handle as _;

    #[test]
    fn compiler_panic_becomes_a_typed_refusal() {
        assert_eq!(
            catch_compile::<()>(|| panic!("fixture compiler panic")),
            Err(ReplacementComputePipelineCompileError::WorkerPanicked)
        );
    }

    fn compile_fixture() -> Option<Vec<u32>> {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/replacement_compute_test.comp");
        let output = std::env::temp_dir().join(format!(
            "reims-vgpu-replacement-compute-{}.spv",
            std::process::id()
        ));
        let status = std::process::Command::new("glslc")
            .args(["-fshader-stage=compute", "-O"])
            .arg(source)
            .arg("-o")
            .arg(&output)
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        let bytes = std::fs::read(&output).ok()?;
        let _ = std::fs::remove_file(output);
        Some(
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte word")))
                .collect(),
        )
    }

    #[test]
    fn actual_compute_family_compile_publishes_and_owns_every_handle_until_drop() {
        let Some(words) = compile_fixture() else {
            eprintln!("SKIP replacement compute compile: glslc unavailable");
            return;
        };
        let context = match unsafe { DeviceContext::create() } {
            Ok(context) => Arc::new(SharedDeviceContext::new(context)),
            Err(error) => {
                eprintln!("SKIP replacement compute compile: no device ({error})");
                return;
            }
        };
        let owner = crate::m2v_cache::prepare_shader_words(words);
        let program = crate::m2v_cache::prepared_stage(&owner);
        let plan = ReplacementComputePipelinePlan {
            program,
            descriptors: Box::new([]),
        };
        let family = ReplacementComputePipelineFamily::default();
        let job = family.begin_compile(plan.variant_key()).unwrap();
        let compiler = ReplacementComputePipelineCompiler::new(
            context,
            reims_vgpu_core::VulkanDeviceEpoch::new(reims_vgpu_protocol::VulkanDeviceEpochId::new(
                1,
            )),
        );
        let native = compiler
            .compile_family_job(&family, job, plan)
            .expect("compile compute family")
            .native;
        assert_ne!(native.pipeline.as_raw(), 0);
        assert_ne!(native.layout.as_raw(), 0);
        assert_ne!(native.descriptor_set_layout.as_raw(), 0);
        assert_eq!(native.thread_grid_push_offset, None);
        assert_eq!(family.census().live, 1);
    }
}
