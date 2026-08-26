//! Composition projection into the backend-neutral replacement compute command.

use crate::runtime::replacement_compute_state::ComputeAccum;
use crate::runtime::replacement_services::ComputeTranslation;
use reims_vgpu_core::{
    AccessMode, BackingRegion, ComputeBindingClass, ComputeBindingView, ReflectedBufferAccess,
    ReflectedComputeTexture, ResolvedComputeDispatch, ResolvedComputeNullBinding,
    ResolvedComputeResourceBinding, ResolvedComputeSamplerBinding, SamplerResource,
    ShaderResourceKind, StorageImageAccess,
};
use reims_vgpu_protocol::{ResourceId, ResourceObject};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeConstructionError {
    PipelineMissing,
    ShaderLocalSizeMissing,
    UnsupportedShaderInterface {
        feature: &'static str,
        count: usize,
    },
    UnsupportedShaderResource {
        index: u32,
        kind: ShaderResourceKind,
    },
    ThreadgroupMemoryUnsupported {
        index: u32,
        length: Option<u64>,
    },
    IcbInheritedPipelineSpecified,
    IcbInheritedBuffersSpecified,
    IcbPipelineMissing,
    IcbDispatchGeometry,
    DescriptorMissing(u32),
    DescriptorSetUnsupported(u32),
    BufferMissing(u32),
    BufferExtentPastResource {
        resource: ResourceId<ResourceObject>,
        requested: u64,
        available: u64,
    },
    TextureMissing(u32),
    ResourceBackingMissing(ResourceId<ResourceObject>),
    TextureView(reims_vgpu_core::TextureViewResolveError),
    TextureShapeUnsupported(u32),
    TextureAccessMissing(u32),
    TextureAccessAmbiguous(u32),
    SamplerMissing(u32),
    SamplerState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementComputeBufferBinding {
    pub resource: ResourceId<ResourceObject>,
    pub backing: reims_vgpu_protocol::BackingId,
    pub range: reims_vgpu_core::LinearRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementComputeTextureBinding {
    pub resource: ResourceId<ResourceObject>,
    pub backing: reims_vgpu_protocol::BackingId,
    pub view: reims_vgpu_core::ResolvedTextureBindingView,
}

pub(crate) trait ReplacementComputeResolver {
    fn pipeline(
        &mut self,
        reference: u32,
    ) -> Result<
        reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ComputePipelineObject>,
        ReplacementComputeConstructionError,
    >;

    fn buffer(
        &mut self,
        index: u32,
        reference: u32,
        offset: u64,
        length: Option<u64>,
    ) -> Result<ReplacementComputeBufferBinding, ReplacementComputeConstructionError>;

    fn texture(
        &mut self,
        index: u32,
        reference: u32,
    ) -> Result<ReplacementComputeTextureBinding, ReplacementComputeConstructionError>;

    fn sampler(
        &mut self,
        index: u32,
        reference: u32,
        binding: u32,
    ) -> Result<SamplerResource, ReplacementComputeConstructionError>;
}

pub(crate) fn construct_resolved(
    resolver: &mut impl ReplacementComputeResolver,
    accum: &ComputeAccum,
    launch: reims_vgpu_core::ResolvedComputeLaunch,
    translation: &dyn ComputeTranslation,
) -> Result<ResolvedComputeDispatch, ReplacementComputeConstructionError> {
    let pipeline = resolver.pipeline(accum.pipeline_ref)?;
    let interface = translation.interface();
    if let Some(unsupported) =
        interface.first_unsupported_interface(reims_vgpu_core::ReflectedShaderStage::Kernel)
    {
        return Err(
            ReplacementComputeConstructionError::UnsupportedShaderInterface {
                feature: unsupported.feature,
                count: unsupported.count,
            },
        );
    }
    if let Some(resource) = interface.first_unsupported_resource() {
        return Err(
            ReplacementComputeConstructionError::UnsupportedShaderResource {
                index: resource.metal_index,
                kind: resource.kind,
            },
        );
    }
    if let Some(resource) = interface
        .bindings
        .iter()
        .find(|resource| resource.kind == ShaderResourceKind::ThreadgroupBuffer)
    {
        return Err(
            ReplacementComputeConstructionError::ThreadgroupMemoryUnsupported {
                index: resource.metal_index,
                length: accum
                    .threadgroup_memory
                    .iter()
                    .find(|binding| binding.index == resource.metal_index)
                    .map(|binding| binding.length),
            },
        );
    }
    let local_size = interface
        .local_size
        .ok_or(ReplacementComputeConstructionError::ShaderLocalSizeMissing)?;
    let mut resources = Vec::new();
    let mut samplers = Vec::new();
    let mut null_bindings = Vec::new();

    for bound in &accum.buffers {
        let access = interface.buffer_access(bound.index);
        if matches!(
            access,
            ReflectedBufferAccess::Unused | ReflectedBufferAccess::Absent
        ) {
            continue;
        }
        let reflected = interface
            .bindings
            .iter()
            .find(|resource| {
                resource.kind == ShaderResourceKind::Buffer && resource.metal_index == bound.index
            })
            .ok_or(ReplacementComputeConstructionError::DescriptorMissing(
                bound.index,
            ))?;
        let descriptor =
            reflected
                .descriptor
                .ok_or(ReplacementComputeConstructionError::DescriptorMissing(
                    bound.index,
                ))?;
        if descriptor.set != 0 {
            return Err(
                ReplacementComputeConstructionError::DescriptorSetUnsupported(descriptor.set),
            );
        }
        let available = resolver.buffer(bound.index, bound.buffer_ref, bound.offset, None)?;
        let available_length = available.range.end() - available.range.start();
        let length = match launch {
            reims_vgpu_core::ResolvedComputeLaunch::Direct(workgroups) => {
                translation.buffer_extent(bound.index, workgroups.counts, local_size)
            }
            reims_vgpu_core::ResolvedComputeLaunch::IndirectThreadgroups { .. } => None,
        }
        .unwrap_or(available_length);
        if length > available_length {
            return Err(
                ReplacementComputeConstructionError::BufferExtentPastResource {
                    resource: available.resource,
                    requested: length,
                    available: available_length,
                },
            );
        }
        let resolved = if length == available_length {
            available
        } else {
            resolver.buffer(bound.index, bound.buffer_ref, bound.offset, Some(length))?
        };
        resources.push(ResolvedComputeResourceBinding {
            class: ComputeBindingClass::Buffer,
            binding: descriptor.binding,
            array_element: 0,
            descriptor_count: descriptor.count,
            resource: resolved.resource,
            backing: resolved.backing,
            view: ComputeBindingView::Buffer(resolved.range),
            regions: Box::new([BackingRegion::Linear(resolved.range)]),
            mode: buffer_access(access),
        });
    }

    for bound in &accum.textures {
        let Some(descriptor) = interface.texture_descriptor(bound.index) else {
            continue;
        };
        let class = match interface.compute_texture(descriptor.binding) {
            ReflectedComputeTexture::Plain2d(reims_vgpu_core::ImageAccess::Sampled)
            | ReflectedComputeTexture::Multisampled2d => ComputeBindingClass::SampledImage,
            ReflectedComputeTexture::Plain2d(reims_vgpu_core::ImageAccess::Storage) => {
                ComputeBindingClass::StorageImage
            }
            ReflectedComputeTexture::Absent => continue,
            ReflectedComputeTexture::UnstageableShape { .. } => {
                return Err(
                    ReplacementComputeConstructionError::TextureShapeUnsupported(bound.index),
                );
            }
        };
        let resolved = resolver.texture(bound.index, bound.texture_ref)?;
        let mode = match class {
            ComputeBindingClass::SampledImage => AccessMode::Read,
            ComputeBindingClass::StorageImage => storage_access(
                translation.storage_image_access(descriptor.binding),
                descriptor.binding,
            )?,
            ComputeBindingClass::Buffer => unreachable!(),
        };
        resources.push(ResolvedComputeResourceBinding {
            class,
            binding: descriptor.binding,
            array_element: descriptor.array_element,
            descriptor_count: descriptor.descriptor_count,
            resource: resolved.resource,
            backing: resolved.backing,
            view: ComputeBindingView::Image(resolved.view),
            regions: Box::new([BackingRegion::Whole]),
            mode,
        });
    }

    for reflected in translation.samplers().iter() {
        let sampler = if let Some(static_state) = reflected.static_state {
            crate::runtime::replacement_sampler_projection::reflected_sampler(
                "kernel",
                reflected.binding,
                static_state,
            )
            .map_err(|_| ReplacementComputeConstructionError::SamplerState)?
        } else if let Some(bound) = accum
            .samplers
            .iter()
            .find(|bound| bound.index == reflected.metal_index)
        {
            let mut sampler =
                resolver.sampler(reflected.metal_index, bound.sampler_ref, reflected.binding)?;
            if bound.has_lod_clamp {
                sampler.lod_min = bound.lod_min_bits;
                sampler.lod_max = bound.lod_max_bits;
            }
            sampler
        } else {
            SamplerResource::null(reflected.binding)
        };
        samplers.push(ResolvedComputeSamplerBinding {
            binding: reflected.binding,
            array_element: 0,
            descriptor_count: 1,
            sampler,
        });
    }

    for binding in translation.used_descriptor_bindings().iter().copied() {
        // Sampler reflection is already a complete scalar declaration: the
        // translator's validated interface requires sampler descriptor count
        // one, and the loop above emits either its state or an explicit null.
        if samplers.iter().any(|sampler| sampler.binding == binding) {
            continue;
        }
        let reflected = interface
            .bindings
            .iter()
            .find(|resource| {
                resource.descriptor.map(|descriptor| descriptor.binding) == Some(binding)
            })
            .ok_or(ReplacementComputeConstructionError::DescriptorMissing(
                binding,
            ))?;
        let descriptor =
            reflected
                .descriptor
                .ok_or(ReplacementComputeConstructionError::DescriptorMissing(
                    binding,
                ))?;
        let class = match reflected.kind {
            ShaderResourceKind::Buffer => ComputeBindingClass::Buffer,
            kind if kind.is_texture() => match interface.compute_texture(binding) {
                ReflectedComputeTexture::Plain2d(reims_vgpu_core::ImageAccess::Storage) => {
                    ComputeBindingClass::StorageImage
                }
                ReflectedComputeTexture::Plain2d(reims_vgpu_core::ImageAccess::Sampled)
                | ReflectedComputeTexture::Multisampled2d => ComputeBindingClass::SampledImage,
                _ => {
                    return Err(
                        ReplacementComputeConstructionError::TextureShapeUnsupported(binding),
                    );
                }
            },
            ShaderResourceKind::Sampler | ShaderResourceKind::StaticSampler => continue,
            _ => {
                return Err(
                    ReplacementComputeConstructionError::UnsupportedShaderResource {
                        index: reflected.metal_index,
                        kind: reflected.kind,
                    },
                );
            }
        };
        for array_element in 0..descriptor.count {
            if resources.iter().any(|resource| {
                resource.binding == binding && resource.array_element == array_element
            }) {
                continue;
            }
            null_bindings.push(ResolvedComputeNullBinding {
                class,
                binding,
                array_element,
                descriptor_count: descriptor.count,
            });
        }
    }

    Ok(ResolvedComputeDispatch {
        pipeline,
        launch,
        wait_for_prior_commands: false,
        resources: resources.into_boxed_slice(),
        samplers: samplers.into_boxed_slice(),
        null_bindings: null_bindings.into_boxed_slice(),
    })
}

pub(crate) fn construct_icb_resolved(
    resolver: &mut impl ReplacementComputeResolver,
    inherited: &ComputeAccum,
    fill: &crate::runtime::icb::IcbComputeFill,
    descriptor: &reims_vgpu_protocol::IndirectCommandBufferDescriptor,
    translation: &dyn ComputeTranslation,
) -> Result<ResolvedComputeDispatch, ReplacementComputeConstructionError> {
    let mut accum = inherited.clone();
    if descriptor.inherit_pipeline_state() {
        if fill.pipeline_ref != 0 {
            return Err(ReplacementComputeConstructionError::IcbInheritedPipelineSpecified);
        }
    } else {
        accum.pipeline_ref = fill.pipeline_ref;
    }
    if accum.pipeline_ref == 0 {
        return Err(ReplacementComputeConstructionError::IcbPipelineMissing);
    }
    if !descriptor.inherit_buffers() {
        accum.buffers = fill
            .buffers
            .iter()
            .map(
                |binding| crate::runtime::replacement_compute_state::ComputeBufferBind {
                    index: binding.index,
                    buffer_ref: binding.buffer_ref,
                    offset: binding.offset,
                    attribute_stride: binding.attribute_stride,
                    has_attribute_stride: binding.has_attribute_stride,
                },
            )
            .collect();
    } else if !fill.buffers.is_empty() {
        return Err(ReplacementComputeConstructionError::IcbInheritedBuffersSpecified);
    }
    accum.threadgroup_memory = fill
        .threadgroup_memory
        .iter()
        .map(
            |binding| crate::runtime::replacement_compute_state::ThreadgroupMemoryBind {
                index: binding.index,
                length: binding.length,
            },
        )
        .collect();
    let (grid, group, grid_is_threads) = match fill.dispatch {
        crate::runtime::icb::IcbFillDispatch::ConcurrentThreadgroups {
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        } => ([grid_x, grid_y, grid_z], [tg_x, tg_y, tg_z], false),
        crate::runtime::icb::IcbFillDispatch::ConcurrentThreads {
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        } => ([threads_x, threads_y, threads_z], [tg_x, tg_y, tg_z], true),
    };
    let workgroups = reims_vgpu_protocol::dispatch::workgroup_counts(grid, group, grid_is_threads)
        .ok_or(ReplacementComputeConstructionError::IcbDispatchGeometry)?;
    let mut dispatch = construct_resolved(
        resolver,
        &accum,
        reims_vgpu_core::ResolvedComputeLaunch::Direct(workgroups),
        translation,
    )?;
    dispatch.wait_for_prior_commands = fill.barrier;
    Ok(dispatch)
}

pub(crate) fn buffer_access(access: ReflectedBufferAccess) -> AccessMode {
    match access {
        ReflectedBufferAccess::ReadOnly => AccessMode::Read,
        ReflectedBufferAccess::Writable => AccessMode::ReadWrite,
        ReflectedBufferAccess::Unknown => AccessMode::Unknown,
        ReflectedBufferAccess::Unused | ReflectedBufferAccess::Absent => unreachable!(),
    }
}

pub(crate) fn storage_access(
    access: Option<StorageImageAccess>,
    binding: u32,
) -> Result<AccessMode, ReplacementComputeConstructionError> {
    match access {
        Some(StorageImageAccess::ReadOnly) => Ok(AccessMode::Read),
        Some(StorageImageAccess::WriteOnly) => Ok(AccessMode::Write),
        Some(StorageImageAccess::ReadWrite) => Ok(AccessMode::ReadWrite),
        Some(StorageImageAccess::Unknown) => Ok(AccessMode::Unknown),
        Some(StorageImageAccess::AmbiguousBinding) => Err(
            ReplacementComputeConstructionError::TextureAccessAmbiguous(binding),
        ),
        None => Err(ReplacementComputeConstructionError::TextureAccessMissing(
            binding,
        )),
    }
}
