//! Compute encoder state retained by replacement EXEC decoding.

use crate::runtime::decode::compute::{BufferBinding, RefBinding, SamplerBinding};

pub(crate) const MAX_COMPUTE_BUFFER_SLOTS: u32 = 31;
pub(crate) const MAX_COMPUTE_TEXTURE_SLOTS: u32 = 128;
pub(crate) const MAX_COMPUTE_SAMPLER_SLOTS: u32 = 16;

const _: () = assert!(reims_vgpu_wire::ops::bind_limit::BUFFER <= MAX_COMPUTE_BUFFER_SLOTS);
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::TEXTURE <= MAX_COMPUTE_TEXTURE_SLOTS);
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::SAMPLER <= MAX_COMPUTE_SAMPLER_SLOTS);

#[derive(Clone, Debug, Default)]
pub(crate) struct ComputeBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ComputeTextureBind {
    pub index: u32,
    pub texture_ref: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ComputeSamplerBind {
    pub index: u32,
    pub sampler_ref: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub has_lod_clamp: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ThreadgroupMemoryBind {
    pub index: u32,
    pub length: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct StageInRegion {
    pub origin_x: u64,
    pub origin_y: u64,
    pub origin_z: u64,
    pub size_x: u64,
    pub size_y: u64,
    pub size_z: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct StageInRegionIndirect {
    pub buffer_ref: u32,
    pub buffer_offset: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ImageblockDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ComputeAccum {
    pub pipeline_ref: u32,
    pub buffers: Vec<ComputeBufferBind>,
    pub textures: Vec<ComputeTextureBind>,
    pub samplers: Vec<ComputeSamplerBind>,
    pub threadgroup_memory: Vec<ThreadgroupMemoryBind>,
    pub stage_in_region: Option<StageInRegion>,
    pub stage_in_region_indirect: Option<StageInRegionIndirect>,
    pub imageblock: Option<ImageblockDimensions>,
    pub dispatch_type: u32,
}

impl ComputeAccum {
    pub(crate) fn bind_buffers(&mut self, first: u32, entries: &[BufferBinding]) {
        for (offset, entry) in entries.iter().enumerate() {
            let index = first + offset as u32;
            if entry.ref_ == 0 {
                self.buffers.retain(|binding| binding.index != index);
                continue;
            }
            let binding = ComputeBufferBind {
                index,
                buffer_ref: entry.ref_,
                offset: entry.offset,
                attribute_stride: entry.attribute_stride,
                has_attribute_stride: entry.has_attribute_stride,
            };
            replace_or_push(&mut self.buffers, index, binding, |value| value.index);
        }
    }

    pub(crate) fn set_buffer_offset(
        &mut self,
        index: u32,
        offset: u64,
        attribute_stride: Option<u64>,
    ) {
        if let Some(binding) = self
            .buffers
            .iter_mut()
            .find(|binding| binding.index == index)
        {
            binding.offset = offset;
            if let Some(stride) = attribute_stride {
                binding.attribute_stride = stride;
                binding.has_attribute_stride = true;
            }
        }
    }

    pub(crate) fn bind_textures(&mut self, first: u32, entries: &[RefBinding]) {
        for (offset, entry) in entries.iter().enumerate() {
            let index = first + offset as u32;
            if entry.ref_ == 0 {
                self.textures.retain(|binding| binding.index != index);
                continue;
            }
            replace_or_push(
                &mut self.textures,
                index,
                ComputeTextureBind {
                    index,
                    texture_ref: entry.ref_,
                },
                |value| value.index,
            );
        }
    }

    pub(crate) fn bind_samplers(&mut self, first: u32, entries: &[SamplerBinding]) {
        for (offset, entry) in entries.iter().enumerate() {
            let index = first + offset as u32;
            if entry.ref_ == 0 {
                self.samplers.retain(|binding| binding.index != index);
                continue;
            }
            replace_or_push(
                &mut self.samplers,
                index,
                ComputeSamplerBind {
                    index,
                    sampler_ref: entry.ref_,
                    lod_min_bits: entry.lod_min_bits,
                    lod_max_bits: entry.lod_max_bits,
                    has_lod_clamp: entry.has_lod_clamp,
                },
                |value| value.index,
            );
        }
    }

    pub(crate) fn set_threadgroup_memory(&mut self, index: u32, length: u64) {
        replace_or_push(
            &mut self.threadgroup_memory,
            index,
            ThreadgroupMemoryBind { index, length },
            |value| value.index,
        );
    }

    pub(crate) fn set_stage_in_region(&mut self, region: StageInRegion) {
        self.stage_in_region_indirect = None;
        self.stage_in_region = Some(region);
    }

    pub(crate) fn set_stage_in_region_indirect(&mut self, buffer_ref: u32, buffer_offset: u64) {
        if buffer_ref == 0 {
            return;
        }
        self.stage_in_region = None;
        self.stage_in_region_indirect = Some(StageInRegionIndirect {
            buffer_ref,
            buffer_offset,
        });
    }

    pub(crate) fn set_imageblock(&mut self, width: u32, height: u32) {
        if width != 0 && height != 0 {
            self.imageblock = Some(ImageblockDimensions { width, height });
        }
    }
}

fn replace_or_push<T>(values: &mut Vec<T>, index: u32, value: T, key: impl Fn(&T) -> u32) {
    if let Some(slot) = values.iter_mut().find(|candidate| key(candidate) == index) {
        *slot = value;
    } else {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nil_bind_retires_only_the_named_compute_slot() {
        let mut state = ComputeAccum::default();
        state.bind_textures(2, &[RefBinding { ref_: 7 }, RefBinding { ref_: 8 }]);
        state.bind_textures(2, &[RefBinding { ref_: 0 }]);
        assert_eq!(state.textures.len(), 1);
        assert_eq!(state.textures[0].index, 3);
        assert_eq!(state.textures[0].texture_ref, 8);
    }
}
