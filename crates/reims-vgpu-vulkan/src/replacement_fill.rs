//! Worker-owned compute emitter for byte-exact buffer fills.

use crate::{
    replacement_buffer_blit::NativeBufferBlit, replacement_fill_shader::REPLACEMENT_FILL_SPIRV,
};
use ash::vk;

pub(crate) const FILL_DESCRIPTOR_BINDING: u32 = 0;
const FILL_PUSH_BYTES: u32 = 5 * size_of::<u32>() as u32;

#[derive(Clone, Copy)]
pub(crate) struct ReplacementFillPipeline {
    module: vk::ShaderModule,
    pub(crate) descriptor_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ReplacementFillPipeline {
    /// # Safety
    ///
    /// `device` must remain live until the returned fixed pipeline is destroyed.
    pub(crate) unsafe fn create(device: &ash::Device) -> Result<Self, vk::Result> {
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&REPLACEMENT_FILL_SPIRV),
                None,
            )?
        };
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(FILL_DESCRIPTOR_BINDING)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout = match unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_shader_module(module, None) };
                return Err(error);
            }
        };
        let set_layouts = [descriptor_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(FILL_PUSH_BYTES)];
        let layout = match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                    device.destroy_shader_module(module, None);
                }
                return Err(error);
            }
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline = match unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
        } {
            Ok(pipelines) => pipelines[0],
            Err((_, error)) => {
                unsafe {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                    device.destroy_shader_module(module, None);
                }
                return Err(error);
            }
        };
        Ok(Self {
            module,
            descriptor_layout,
            layout,
            pipeline,
        })
    }

    /// # Safety
    ///
    /// No live command buffer may reference this pipeline.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_shader_module(self.module, None);
        }
    }

    /// # Safety
    ///
    /// `command_buffer` must be recording and `descriptor_set` must have been
    /// allocated from `self.descriptor_layout` on `device`.
    pub(crate) unsafe fn record(
        self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        operation: NativeBufferBlit,
    ) {
        let NativeBufferBlit::ComputeFill {
            buffer,
            binding_offset,
            binding_range,
            start,
            byte_count,
            pattern,
            pattern_width,
            word_count,
            dispatch_x,
        } = operation
        else {
            unreachable!("the fixed fill pipeline records only compute fills")
        };
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(binding_offset)
            .range(binding_range)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(FILL_DESCRIPTOR_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
        }
        let before = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .offset(binding_offset)
            .size(binding_range)];
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &before,
                &[],
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[descriptor_set],
                &[],
            );
        }
        let push = [start, byte_count, pattern, pattern_width, word_count];
        let push_bytes = unsafe {
            core::slice::from_raw_parts(push.as_ptr().cast::<u8>(), FILL_PUSH_BYTES as usize)
        };
        unsafe {
            device.cmd_push_constants(
                command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(command_buffer, dispatch_x, 1, 1);
        }
        let after = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .offset(binding_offset)
            .size(binding_range)];
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &after,
                &[],
            );
        }
    }
}
