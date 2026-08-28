//! Resolution of info operations into generational identities and exact reply ranges.
//!
//! Reply allocation shape is decoded once in `reims-vgpu-protocol`. This
//! layer proves that the named buffer generation owns the complete range. The
//! rasterization-rate parameter copy is separate: its output length belongs
//! to the live rate-map object and is therefore supplied by that object's
//! resolver, never inferred from an observed buffer size.

use crate::LinearRange;
use reims_vgpu_protocol::{
    BackingId, ComputePipelineObject, CoordinateMapDirection, HeapObject, ImageblockDimensions,
    InfoOperation, InfoReplyTarget, ObjectTableRef, PipelineStateInfoKind,
    RasterizationRateMapObject, RateMapCoordinate, RenderPipelineObject, ResourceId,
    ResourceInfoKind, ResourceObject, SamplerObject, SerializerRef, TaskId, TextureDeclaration,
    TransactionId,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedInfoReplyTarget {
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub range: LinearRange,
    pub requested_alignment: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedRateParameterDestination {
    pub resource: ResourceId<ResourceObject>,
    pub backing: BackingId,
    pub range: LinearRange,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResolvedInfoOperation {
    RenderPipelineState {
        pipeline: ResourceId<RenderPipelineObject>,
        reply: ResolvedInfoReplyTarget,
    },
    ComputePipelineState {
        pipeline: ResourceId<ComputePipelineObject>,
        reply: ResolvedInfoReplyTarget,
    },
    ResourceHost {
        kind: ResourceInfoKind,
        resource: ResourceId<ResourceObject>,
        reply: ResolvedInfoReplyTarget,
    },
    HeapHost {
        heap: ResourceId<HeapObject>,
        reply: ResolvedInfoReplyTarget,
    },
    SamplerHost {
        sampler: ResourceId<SamplerObject>,
        reply: ResolvedInfoReplyTarget,
    },
    HeapTextureSizeAndAlign {
        descriptor: TextureDeclaration,
        reply: ResolvedInfoReplyTarget,
    },
    RenderPipelineImageblock {
        pipeline: ResourceId<RenderPipelineObject>,
        dimensions: ImageblockDimensions,
        reply: ResolvedInfoReplyTarget,
    },
    ComputePipelineImageblock {
        pipeline: ResourceId<ComputePipelineObject>,
        dimensions: ImageblockDimensions,
        reply: ResolvedInfoReplyTarget,
    },
    RateMapInfo {
        rate_map: ResourceId<RasterizationRateMapObject>,
        layer_count: u32,
        reply: ResolvedInfoReplyTarget,
    },
    CopyRateParameterBuffer {
        rate_map: ResourceId<RasterizationRateMapObject>,
        destination: ResolvedRateParameterDestination,
    },
    MapCoordinate {
        direction: CoordinateMapDirection,
        rate_map: ResourceId<RasterizationRateMapObject>,
        layer: u32,
        coordinate: RateMapCoordinate,
        reply: ResolvedInfoReplyTarget,
    },
}

/// Fixed reply returned for a compiled render pipeline.
///
/// The reply ABI groups its two 32-bit quantities before its two one-byte
/// flags and includes two bytes of tail padding.  Keeping the fields semantic
/// here prevents a live pipeline owner from manufacturing an untyped byte
/// array or depending on the host Rust ABI for the reply layout.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderPipelineStateInfo {
    pub max_total_threads_per_threadgroup: u32,
    pub imageblock_sample_length: u32,
    pub threadgroup_size_matches_tile_size: bool,
    pub supports_indirect_command_buffers: bool,
}

impl RenderPipelineStateInfo {
    pub const BYTE_LEN: usize = 12;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut bytes = [0; Self::BYTE_LEN];
        bytes[0..4].copy_from_slice(&self.max_total_threads_per_threadgroup.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.imageblock_sample_length.to_le_bytes());
        bytes[8] = u8::from(self.threadgroup_size_matches_tile_size);
        bytes[9] = u8::from(self.supports_indirect_command_buffers);
        bytes
    }
}

/// Derive the fixed state-query reply owned by a classic or mesh render
/// pipeline declaration. These declarations have no tile function, so their
/// tile-threadgroup quantities are the Metal non-tile values: zero and false.
/// The imageblock sample length is the sum of the declared color-attachment
/// texel widths; it is independent of framebuffer dimensions and sample count.
pub fn render_pipeline_state_info(
    descriptor: &reims_vgpu_protocol::RenderPipelineDescriptor,
) -> Option<RenderPipelineStateInfo> {
    let mut imageblock_sample_length = 0u32;
    let attachments: &[reims_vgpu_protocol::resource::PipelineColorAttachment] =
        if descriptor.color_attachments.is_empty() {
            std::slice::from_ref(&descriptor.color0)
        } else {
            &descriptor.color_attachments
        };
    for attachment in attachments
        .iter()
        .filter(|attachment| attachment.has_pixel_format && attachment.pixel_format != 0)
    {
        let format = u16::try_from(attachment.pixel_format).ok()?;
        imageblock_sample_length = imageblock_sample_length
            .checked_add(reims_vgpu_protocol::metal_pixel::bytes_per_pixel(format)?)?;
    }
    Some(RenderPipelineStateInfo {
        max_total_threads_per_threadgroup: 0,
        imageblock_sample_length,
        threadgroup_size_matches_tile_size: false,
        supports_indirect_command_buffers: descriptor.supports_indirect_command_buffers,
    })
}

/// Fixed reply returned for a compiled compute pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ComputePipelineStateInfo {
    pub max_total_threads_per_threadgroup: u64,
    pub thread_execution_width: u64,
    pub static_threadgroup_memory_length: u64,
    pub supports_indirect_command_buffers: bool,
}

impl ComputePipelineStateInfo {
    pub const BYTE_LEN: usize = 28;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut bytes = [0; Self::BYTE_LEN];
        bytes[0..8].copy_from_slice(&self.max_total_threads_per_threadgroup.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.thread_execution_width.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.static_threadgroup_memory_length.to_le_bytes());
        bytes[24] = u8::from(self.supports_indirect_command_buffers);
        bytes
    }
}

/// Size and alignment returned for one texture descriptor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeapTextureSizeAndAlignInfo {
    pub size: u64,
    pub align: u64,
}

impl HeapTextureSizeAndAlignInfo {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut bytes = [0; Self::BYTE_LEN];
        bytes[0..8].copy_from_slice(&self.size.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.align.to_le_bytes());
        bytes
    }
}

/// Imageblock storage required for the queried dimensions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImageblockMemoryLength(pub u32);

impl ImageblockMemoryLength {
    pub const BYTE_LEN: usize = 4;

    pub const fn encode(self) -> [u8; Self::BYTE_LEN] {
        self.0.to_le_bytes()
    }
}

/// Return the imageblock allocation for a non-tile render pipeline.
///
/// The allocation is one pipeline-declared imageblock sample for every
/// threadgroup position in the two-dimensional tile. The API's depth member
/// does not multiply this storage class. Arithmetic that cannot fit the
/// four-byte reply is not representable by the query ABI.
pub fn render_pipeline_imageblock_memory_length(
    info: RenderPipelineStateInfo,
    dimensions: ImageblockDimensions,
) -> Option<ImageblockMemoryLength> {
    let bytes = u64::from(info.imageblock_sample_length)
        .checked_mul(dimensions.width)?
        .checked_mul(dimensions.height)?;
    Some(ImageblockMemoryLength(u32::try_from(bytes).ok()?))
}

/// Coordinate returned by a rasterization-rate-map query.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MappedRateCoordinate {
    pub x: f32,
    pub y: f32,
}

/// GPU address returned for a buffer or heap host-resource query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuAddressInfo(pub u64);

impl GpuAddressInfo {
    pub const BYTE_LEN: usize = 8;

    pub const fn encode(self) -> [u8; Self::BYTE_LEN] {
        self.0.to_le_bytes()
    }
}

/// Native resource identity returned for a texture or sampler query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IndexedGpuResourceInfo {
    pub gpu_resource_id: u64,
    pub resource_index: u64,
}

impl IndexedGpuResourceInfo {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut bytes = [0; Self::BYTE_LEN];
        bytes[0..8].copy_from_slice(&self.gpu_resource_id.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.resource_index.to_le_bytes());
        bytes
    }
}

impl MappedRateCoordinate {
    pub const BYTE_LEN: usize = 8;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut bytes = [0; Self::BYTE_LEN];
        bytes[0..4].copy_from_slice(&self.x.to_bits().to_le_bytes());
        bytes[4..8].copy_from_slice(&self.y.to_bits().to_le_bytes());
        bytes
    }
}

/// Live-object answer owner for the closed info selector surface.
///
/// Fixed-size return types retain each selector's decoded reply allocation in
/// the type system. Rasterization-rate payloads are variable because their
/// shape belongs to the live map; preparation still checks the returned byte
/// count against the resolved destination range before reserving a write.
pub trait InfoQueryEvaluator {
    type Error;

    fn render_pipeline_state(
        &self,
        pipeline: ResourceId<RenderPipelineObject>,
    ) -> Result<RenderPipelineStateInfo, Self::Error>;

    fn compute_pipeline_state(
        &self,
        pipeline: ResourceId<ComputePipelineObject>,
    ) -> Result<ComputePipelineStateInfo, Self::Error>;

    fn buffer_host_resource(
        &self,
        resource: ResourceId<ResourceObject>,
    ) -> Result<GpuAddressInfo, Self::Error>;

    fn texture_host_resource(
        &self,
        resource: ResourceId<ResourceObject>,
    ) -> Result<IndexedGpuResourceInfo, Self::Error>;

    fn heap_host_resource(
        &self,
        heap: ResourceId<HeapObject>,
    ) -> Result<GpuAddressInfo, Self::Error>;

    fn sampler_host_resource(
        &self,
        sampler: ResourceId<SamplerObject>,
    ) -> Result<IndexedGpuResourceInfo, Self::Error>;

    fn heap_texture_size_and_align(
        &self,
        descriptor: TextureDeclaration,
    ) -> Result<HeapTextureSizeAndAlignInfo, Self::Error>;

    fn render_pipeline_imageblock(
        &self,
        pipeline: ResourceId<RenderPipelineObject>,
        dimensions: ImageblockDimensions,
    ) -> Result<ImageblockMemoryLength, Self::Error>;

    fn compute_pipeline_imageblock(
        &self,
        pipeline: ResourceId<ComputePipelineObject>,
        dimensions: ImageblockDimensions,
    ) -> Result<ImageblockMemoryLength, Self::Error>;

    fn rate_map_info(
        &self,
        rate_map: ResourceId<RasterizationRateMapObject>,
        layer_count: u32,
    ) -> Result<Box<[u8]>, Self::Error>;

    fn rate_parameter_buffer(
        &self,
        rate_map: ResourceId<RasterizationRateMapObject>,
    ) -> Result<Box<[u8]>, Self::Error>;

    fn map_coordinate(
        &self,
        direction: CoordinateMapDirection,
        rate_map: ResourceId<RasterizationRateMapObject>,
        layer: u32,
        coordinate: RateMapCoordinate,
    ) -> Result<MappedRateCoordinate, Self::Error>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedInfoQuery {
    transaction: TransactionId,
    index: usize,
    operation: ResolvedInfoOperation,
    bytes: Box<[u8]>,
}

impl EvaluatedInfoQuery {
    pub(crate) fn from_parts(
        transaction: TransactionId,
        index: usize,
        operation: ResolvedInfoOperation,
        bytes: Box<[u8]>,
    ) -> Self {
        Self {
            transaction,
            index,
            operation,
            bytes,
        }
    }

    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn operation(&self) -> &ResolvedInfoOperation {
        &self.operation
    }

    pub const fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn into_parts(self) -> (TransactionId, usize, ResolvedInfoOperation, Box<[u8]>) {
        (self.transaction, self.index, self.operation, self.bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InfoQueryEvaluationError<Error> {
    OperationAbsent(usize),
    Evaluator(Error),
}

pub fn evaluate_info_query<Evaluator: InfoQueryEvaluator>(
    admitted: &crate::AdmittedInfoQueries<ResolvedInfoOperation>,
    index: usize,
    evaluator: &Evaluator,
) -> Result<EvaluatedInfoQuery, InfoQueryEvaluationError<Evaluator::Error>> {
    let operation = admitted
        .operations()
        .iter()
        .find(|(position, _)| *position == index)
        .map(|(_, operation)| operation)
        .ok_or(InfoQueryEvaluationError::OperationAbsent(index))?;
    evaluate_info_operation(operation, evaluator)
        .map(|bytes| EvaluatedInfoQuery {
            transaction: admitted.transaction(),
            index,
            operation: *operation,
            bytes,
        })
        .map_err(InfoQueryEvaluationError::Evaluator)
}

fn evaluate_info_operation<Evaluator: InfoQueryEvaluator>(
    operation: &ResolvedInfoOperation,
    evaluator: &Evaluator,
) -> Result<Box<[u8]>, Evaluator::Error> {
    match *operation {
        ResolvedInfoOperation::RenderPipelineState { pipeline, .. } => evaluator
            .render_pipeline_state(pipeline)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::ComputePipelineState { pipeline, .. } => evaluator
            .compute_pipeline_state(pipeline)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::ResourceHost {
            kind: ResourceInfoKind::BufferHostResource,
            resource,
            ..
        } => evaluator
            .buffer_host_resource(resource)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::ResourceHost {
            kind: ResourceInfoKind::TextureHostResource,
            resource,
            ..
        } => evaluator
            .texture_host_resource(resource)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::HeapHost { heap, .. } => evaluator
            .heap_host_resource(heap)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::SamplerHost { sampler, .. } => evaluator
            .sampler_host_resource(sampler)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::HeapTextureSizeAndAlign { descriptor, .. } => evaluator
            .heap_texture_size_and_align(descriptor)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::RenderPipelineImageblock {
            pipeline,
            dimensions,
            ..
        } => evaluator
            .render_pipeline_imageblock(pipeline, dimensions)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::ComputePipelineImageblock {
            pipeline,
            dimensions,
            ..
        } => evaluator
            .compute_pipeline_imageblock(pipeline, dimensions)
            .map(|info| info.encode().to_vec().into_boxed_slice()),
        ResolvedInfoOperation::RateMapInfo {
            rate_map,
            layer_count,
            ..
        } => evaluator.rate_map_info(rate_map, layer_count),
        ResolvedInfoOperation::CopyRateParameterBuffer { rate_map, .. } => {
            evaluator.rate_parameter_buffer(rate_map)
        }
        ResolvedInfoOperation::MapCoordinate {
            direction,
            rate_map,
            layer,
            coordinate,
            ..
        } => evaluator
            .map_coordinate(direction, rate_map, layer, coordinate)
            .map(|coordinate| coordinate.encode().to_vec().into_boxed_slice()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfoReplyResolutionError {
    BufferAbsent,
    NotBuffer,
    BackingAbsent,
    RangeOverflow,
    RangeOutOfBounds,
    RateParameterLengthUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfoResolutionError {
    Reply(InfoReplyResolutionError),
    ResourceAbsent(ResourceInfoKind),
    HeapAbsent,
    SamplerAbsent,
    RenderPipelineAbsent,
    ComputePipelineAbsent,
    RateMapAbsent,
    RateParameterDestination(InfoReplyResolutionError),
}

impl InfoResolutionError {
    /// See [`crate::TextureViewResolveError::awaits_declaration`].
    ///
    /// Every `*Absent` arm names an object whose declaration may still arrive.
    /// A reply-target refusal is about a range the guest named in this packet
    /// and does not change.
    #[must_use]
    pub const fn awaits_declaration(&self) -> bool {
        match self {
            Self::ResourceAbsent(_)
            | Self::HeapAbsent
            | Self::SamplerAbsent
            | Self::RenderPipelineAbsent
            | Self::ComputePipelineAbsent
            | Self::RateMapAbsent => true,
            Self::Reply(_) | Self::RateParameterDestination(_) => false,
        }
    }
}

/// Namespace and storage owner used by info resolution.
///
/// Implementations resolve only live generations. `resolve_reply` additionally
/// proves `offset..offset+length`; `resolve_rate_parameter_destination` derives
/// its range length from the live rate-map contract.
pub trait InfoOperationResolver {
    fn resolve_reply(
        &self,
        task: TaskId,
        target: InfoReplyTarget,
    ) -> Result<ResolvedInfoReplyTarget, InfoReplyResolutionError>;

    fn resolve_resource(
        &self,
        task: TaskId,
        resource: ObjectTableRef<ResourceObject>,
        kind: ResourceInfoKind,
    ) -> Option<ResourceId<ResourceObject>>;

    fn resolve_heap(
        &self,
        task: TaskId,
        heap: SerializerRef<HeapObject>,
    ) -> Option<ResourceId<HeapObject>>;

    fn resolve_sampler(
        &self,
        task: TaskId,
        sampler: SerializerRef<SamplerObject>,
    ) -> Option<ResourceId<SamplerObject>>;

    fn resolve_render_pipeline(
        &self,
        task: TaskId,
        pipeline: SerializerRef<RenderPipelineObject>,
    ) -> Option<ResourceId<RenderPipelineObject>>;

    fn resolve_compute_pipeline(
        &self,
        task: TaskId,
        pipeline: SerializerRef<ComputePipelineObject>,
    ) -> Option<ResourceId<ComputePipelineObject>>;

    fn resolve_rate_map(
        &self,
        task: TaskId,
        rate_map: SerializerRef<RasterizationRateMapObject>,
    ) -> Option<ResourceId<RasterizationRateMapObject>>;

    fn resolve_rate_parameter_destination(
        &self,
        task: TaskId,
        rate_map: ResourceId<RasterizationRateMapObject>,
        buffer: ObjectTableRef<ResourceObject>,
        offset: u64,
    ) -> Result<ResolvedRateParameterDestination, InfoReplyResolutionError>;
}

pub fn resolve_info_operation(
    task: TaskId,
    operation: &InfoOperation,
    resolver: &impl InfoOperationResolver,
) -> Result<ResolvedInfoOperation, InfoResolutionError> {
    let reply = |target| {
        resolver
            .resolve_reply(task, target)
            .map_err(InfoResolutionError::Reply)
    };
    Ok(match *operation {
        InfoOperation::PipelineState {
            kind: PipelineStateInfoKind::Render,
            pipeline_ref,
            reply: target,
        } => ResolvedInfoOperation::RenderPipelineState {
            pipeline: resolver
                .resolve_render_pipeline(task, SerializerRef::new(pipeline_ref))
                .ok_or(InfoResolutionError::RenderPipelineAbsent)?,
            reply: reply(target)?,
        },
        InfoOperation::PipelineState {
            kind: PipelineStateInfoKind::Compute,
            pipeline_ref,
            reply: target,
        } => ResolvedInfoOperation::ComputePipelineState {
            pipeline: resolver
                .resolve_compute_pipeline(task, SerializerRef::new(pipeline_ref))
                .ok_or(InfoResolutionError::ComputePipelineAbsent)?,
            reply: reply(target)?,
        },
        InfoOperation::ResourceHost {
            kind,
            resource,
            reply: target,
        } => ResolvedInfoOperation::ResourceHost {
            kind,
            resource: resolver
                .resolve_resource(task, resource, kind)
                .ok_or(InfoResolutionError::ResourceAbsent(kind))?,
            reply: reply(target)?,
        },
        InfoOperation::HeapHost {
            heap,
            reply: target,
        } => ResolvedInfoOperation::HeapHost {
            heap: resolver
                .resolve_heap(task, heap)
                .ok_or(InfoResolutionError::HeapAbsent)?,
            reply: reply(target)?,
        },
        InfoOperation::SamplerHost {
            sampler,
            reply: target,
        } => ResolvedInfoOperation::SamplerHost {
            sampler: resolver
                .resolve_sampler(task, sampler)
                .ok_or(InfoResolutionError::SamplerAbsent)?,
            reply: reply(target)?,
        },
        InfoOperation::HeapTextureSizeAndAlign {
            descriptor,
            reply: target,
        } => ResolvedInfoOperation::HeapTextureSizeAndAlign {
            descriptor,
            reply: reply(target)?,
        },
        InfoOperation::RenderPipelineImageblock {
            pipeline,
            dimensions,
            reply: target,
        } => ResolvedInfoOperation::RenderPipelineImageblock {
            pipeline: resolver
                .resolve_render_pipeline(task, pipeline)
                .ok_or(InfoResolutionError::RenderPipelineAbsent)?,
            dimensions,
            reply: reply(target)?,
        },
        InfoOperation::ComputePipelineImageblock {
            pipeline,
            dimensions,
            reply: target,
        } => ResolvedInfoOperation::ComputePipelineImageblock {
            pipeline: resolver
                .resolve_compute_pipeline(task, pipeline)
                .ok_or(InfoResolutionError::ComputePipelineAbsent)?,
            dimensions,
            reply: reply(target)?,
        },
        InfoOperation::RateMapInfo {
            rate_map,
            layer_count,
            reply: target,
        } => ResolvedInfoOperation::RateMapInfo {
            rate_map: resolver
                .resolve_rate_map(task, rate_map)
                .ok_or(InfoResolutionError::RateMapAbsent)?,
            layer_count,
            reply: reply(target)?,
        },
        InfoOperation::CopyRateParameterBuffer {
            rate_map,
            destination,
            destination_offset,
        } => {
            let rate_map = resolver
                .resolve_rate_map(task, rate_map)
                .ok_or(InfoResolutionError::RateMapAbsent)?;
            ResolvedInfoOperation::CopyRateParameterBuffer {
                rate_map,
                destination: resolver
                    .resolve_rate_parameter_destination(
                        task,
                        rate_map,
                        destination,
                        destination_offset,
                    )
                    .map_err(InfoResolutionError::RateParameterDestination)?,
            }
        }
        InfoOperation::MapCoordinate {
            direction,
            rate_map,
            layer,
            coordinate,
            reply: target,
        } => ResolvedInfoOperation::MapCoordinate {
            direction,
            rate_map: resolver
                .resolve_rate_map(task, rate_map)
                .ok_or(InfoResolutionError::RateMapAbsent)?,
            layer,
            coordinate,
            reply: reply(target)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_pipeline_info_sums_declared_attachment_sample_widths() {
        use reims_vgpu_protocol::{
            metal_pixel::{MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM},
            resource::{PipelineColorAttachment, RenderPipelineDescriptor},
        };

        let descriptor = RenderPipelineDescriptor {
            supports_indirect_command_buffers: true,
            color_attachments: vec![
                PipelineColorAttachment {
                    slot: 0,
                    has_pixel_format: true,
                    pixel_format: u32::from(MTL_FORMAT_RGBA8_UNORM),
                    ..Default::default()
                },
                PipelineColorAttachment {
                    slot: 1,
                    has_pixel_format: true,
                    pixel_format: u32::from(MTL_FORMAT_RGBA16_FLOAT),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert_eq!(
            render_pipeline_state_info(&descriptor),
            Some(RenderPipelineStateInfo {
                max_total_threads_per_threadgroup: 0,
                imageblock_sample_length: 12,
                threadgroup_size_matches_tile_size: false,
                supports_indirect_command_buffers: true,
            })
        );
    }
    use crate::{ExecTransaction, ResolvedExecSegment, ResolvedExecStream, ResolvedOperation};
    use reims_vgpu_protocol::{SegmentBoundary, SegmentKind, SubmissionId, SubmissionIdentity};

    struct Resolver;

    struct Evaluator;

    impl InfoQueryEvaluator for Evaluator {
        type Error = ();

        fn render_pipeline_state(
            &self,
            _pipeline: ResourceId<RenderPipelineObject>,
        ) -> Result<RenderPipelineStateInfo, Self::Error> {
            Ok(RenderPipelineStateInfo {
                max_total_threads_per_threadgroup: 1,
                imageblock_sample_length: 2,
                threadgroup_size_matches_tile_size: true,
                supports_indirect_command_buffers: false,
            })
        }

        fn compute_pipeline_state(
            &self,
            _pipeline: ResourceId<ComputePipelineObject>,
        ) -> Result<ComputePipelineStateInfo, Self::Error> {
            Ok(ComputePipelineStateInfo {
                max_total_threads_per_threadgroup: 3,
                thread_execution_width: 4,
                static_threadgroup_memory_length: 5,
                supports_indirect_command_buffers: true,
            })
        }

        fn buffer_host_resource(
            &self,
            _resource: ResourceId<ResourceObject>,
        ) -> Result<GpuAddressInfo, Self::Error> {
            Ok(GpuAddressInfo(0x0303_0303_0303_0303))
        }

        fn texture_host_resource(
            &self,
            _resource: ResourceId<ResourceObject>,
        ) -> Result<IndexedGpuResourceInfo, Self::Error> {
            Ok(IndexedGpuResourceInfo {
                gpu_resource_id: 0x0404_0404_0404_0404,
                resource_index: 0x0505_0505_0505_0505,
            })
        }

        fn heap_host_resource(
            &self,
            _heap: ResourceId<HeapObject>,
        ) -> Result<GpuAddressInfo, Self::Error> {
            Ok(GpuAddressInfo(0x0606_0606_0606_0606))
        }

        fn sampler_host_resource(
            &self,
            _sampler: ResourceId<SamplerObject>,
        ) -> Result<IndexedGpuResourceInfo, Self::Error> {
            Ok(IndexedGpuResourceInfo {
                gpu_resource_id: 0x0707_0707_0707_0707,
                resource_index: 0x0808_0808_0808_0808,
            })
        }

        fn heap_texture_size_and_align(
            &self,
            _descriptor: TextureDeclaration,
        ) -> Result<HeapTextureSizeAndAlignInfo, Self::Error> {
            Ok(HeapTextureSizeAndAlignInfo { size: 7, align: 8 })
        }

        fn render_pipeline_imageblock(
            &self,
            _pipeline: ResourceId<RenderPipelineObject>,
            _dimensions: ImageblockDimensions,
        ) -> Result<ImageblockMemoryLength, Self::Error> {
            Ok(ImageblockMemoryLength(9))
        }

        fn compute_pipeline_imageblock(
            &self,
            _pipeline: ResourceId<ComputePipelineObject>,
            _dimensions: ImageblockDimensions,
        ) -> Result<ImageblockMemoryLength, Self::Error> {
            Ok(ImageblockMemoryLength(10))
        }

        fn rate_map_info(
            &self,
            _rate_map: ResourceId<RasterizationRateMapObject>,
            layer_count: u32,
        ) -> Result<Box<[u8]>, Self::Error> {
            Ok(vec![10; layer_count as usize].into_boxed_slice())
        }

        fn rate_parameter_buffer(
            &self,
            _rate_map: ResourceId<RasterizationRateMapObject>,
        ) -> Result<Box<[u8]>, Self::Error> {
            Ok(vec![11; 20].into_boxed_slice())
        }

        fn map_coordinate(
            &self,
            _direction: CoordinateMapDirection,
            _rate_map: ResourceId<RasterizationRateMapObject>,
            _layer: u32,
            _coordinate: RateMapCoordinate,
        ) -> Result<MappedRateCoordinate, Self::Error> {
            Ok(MappedRateCoordinate { x: 12.5, y: -3.0 })
        }
    }

    impl InfoOperationResolver for Resolver {
        fn resolve_reply(
            &self,
            _task: TaskId,
            target: InfoReplyTarget,
        ) -> Result<ResolvedInfoReplyTarget, InfoReplyResolutionError> {
            let range = LinearRange::new(target.offset, u64::from(target.length))
                .ok_or(InfoReplyResolutionError::RangeOverflow)?;
            if range.end() > 128 {
                return Err(InfoReplyResolutionError::RangeOutOfBounds);
            }
            Ok(ResolvedInfoReplyTarget {
                resource: ResourceId::new(target.buffer.get(), 3),
                backing: BackingId::new(8),
                range,
                requested_alignment: target.alignment,
            })
        }

        fn resolve_resource(
            &self,
            _task: TaskId,
            resource: ObjectTableRef<ResourceObject>,
            _kind: ResourceInfoKind,
        ) -> Option<ResourceId<ResourceObject>> {
            Some(ResourceId::new(resource.get(), 4))
        }

        fn resolve_heap(
            &self,
            _task: TaskId,
            heap: SerializerRef<HeapObject>,
        ) -> Option<ResourceId<HeapObject>> {
            Some(ResourceId::new(heap.get(), 5))
        }

        fn resolve_sampler(
            &self,
            _task: TaskId,
            sampler: SerializerRef<SamplerObject>,
        ) -> Option<ResourceId<SamplerObject>> {
            Some(ResourceId::new(sampler.get(), 6))
        }

        fn resolve_render_pipeline(
            &self,
            _task: TaskId,
            pipeline: SerializerRef<RenderPipelineObject>,
        ) -> Option<ResourceId<RenderPipelineObject>> {
            Some(ResourceId::new(pipeline.get(), 7))
        }

        fn resolve_compute_pipeline(
            &self,
            _task: TaskId,
            pipeline: SerializerRef<ComputePipelineObject>,
        ) -> Option<ResourceId<ComputePipelineObject>> {
            Some(ResourceId::new(pipeline.get(), 8))
        }

        fn resolve_rate_map(
            &self,
            _task: TaskId,
            rate_map: SerializerRef<RasterizationRateMapObject>,
        ) -> Option<ResourceId<RasterizationRateMapObject>> {
            Some(ResourceId::new(rate_map.get(), 9))
        }

        fn resolve_rate_parameter_destination(
            &self,
            _task: TaskId,
            _rate_map: ResourceId<RasterizationRateMapObject>,
            buffer: ObjectTableRef<ResourceObject>,
            offset: u64,
        ) -> Result<ResolvedRateParameterDestination, InfoReplyResolutionError> {
            Ok(ResolvedRateParameterDestination {
                resource: ResourceId::new(buffer.get(), 3),
                backing: BackingId::new(8),
                range: LinearRange::new(offset, 44).unwrap(),
            })
        }
    }

    fn target(offset: u64, length: u32) -> InfoReplyTarget {
        InfoReplyTarget {
            buffer: ObjectTableRef::new(17),
            offset,
            length,
            alignment: 4,
        }
    }

    fn admitted_query(
        operation: ResolvedInfoOperation,
    ) -> crate::AdmittedInfoQueries<ResolvedInfoOperation> {
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(2),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        index: 0,
                        kind: SegmentKind::Info,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: Box::new([ResolvedOperation::<(), (), _, (), ()>::InfoQuery(
                        operation,
                    )]),
                }]),
            }]),
            accesses: Box::new([]),
        };
        crate::AdmittedInfoQueries::from_exec(TransactionId::new(7), &exec)
    }

    fn descriptor() -> TextureDeclaration {
        TextureDeclaration {
            texture_type: reims_vgpu_protocol::TextureType::D2,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: 70,
            width: 1,
            height: 1,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            protection_options: 0,
            swizzle: None,
        }
    }

    #[test]
    fn reply_resolution_captures_generation_backing_and_exact_range() {
        let operation = InfoOperation::ResourceHost {
            kind: ResourceInfoKind::TextureHostResource,
            resource: ObjectTableRef::new(4),
            reply: target(32, 16),
        };
        assert_eq!(
            resolve_info_operation(TaskId::new(2), &operation, &Resolver),
            Ok(ResolvedInfoOperation::ResourceHost {
                kind: ResourceInfoKind::TextureHostResource,
                resource: ResourceId::new(4, 4),
                reply: ResolvedInfoReplyTarget {
                    resource: ResourceId::new(17, 3),
                    backing: BackingId::new(8),
                    range: LinearRange::new(32, 16).unwrap(),
                    requested_alignment: 4,
                },
            })
        );
    }

    #[test]
    fn out_of_bounds_reply_is_a_typed_refusal() {
        let operation = InfoOperation::HeapTextureSizeAndAlign {
            descriptor: descriptor(),
            reply: target(120, 16),
        };
        assert_eq!(
            resolve_info_operation(TaskId::new(2), &operation, &Resolver),
            Err(InfoResolutionError::Reply(
                InfoReplyResolutionError::RangeOutOfBounds
            ))
        );
    }

    #[test]
    fn rate_parameter_copy_length_comes_from_the_live_rate_map_resolver() {
        let operation = InfoOperation::CopyRateParameterBuffer {
            rate_map: SerializerRef::new(3),
            destination: ObjectTableRef::new(17),
            destination_offset: 20,
        };
        assert_eq!(
            resolve_info_operation(TaskId::new(2), &operation, &Resolver),
            Ok(ResolvedInfoOperation::CopyRateParameterBuffer {
                rate_map: ResourceId::new(3, 9),
                destination: ResolvedRateParameterDestination {
                    resource: ResourceId::new(17, 3),
                    backing: BackingId::new(8),
                    range: LinearRange::new(20, 44).unwrap(),
                },
            })
        );
    }

    #[test]
    fn evaluator_dispatch_preserves_fixed_and_variable_reply_shapes() {
        let reply = ResolvedInfoReplyTarget {
            resource: ResourceId::new(1, 1),
            backing: BackingId::new(1),
            range: LinearRange::new(0, 12).unwrap(),
            requested_alignment: 4,
        };
        assert_eq!(
            evaluate_info_query(
                &admitted_query(ResolvedInfoOperation::RenderPipelineState {
                    pipeline: ResourceId::new(2, 1),
                    reply,
                }),
                0,
                &Evaluator,
            )
            .unwrap()
            .bytes(),
            [1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(
            evaluate_info_query(
                &admitted_query(ResolvedInfoOperation::ResourceHost {
                    kind: ResourceInfoKind::BufferHostResource,
                    resource: ResourceId::new(4, 1),
                    reply,
                }),
                0,
                &Evaluator,
            )
            .unwrap()
            .bytes(),
            [3; GpuAddressInfo::BYTE_LEN]
        );
        assert_eq!(
            evaluate_info_query(
                &admitted_query(ResolvedInfoOperation::ResourceHost {
                    kind: ResourceInfoKind::TextureHostResource,
                    resource: ResourceId::new(4, 1),
                    reply,
                }),
                0,
                &Evaluator,
            )
            .unwrap()
            .bytes(),
            [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,]
        );
        assert_eq!(
            evaluate_info_query(
                &admitted_query(ResolvedInfoOperation::HeapHost {
                    heap: ResourceId::new(5, 1),
                    reply,
                }),
                0,
                &Evaluator,
            )
            .unwrap()
            .bytes(),
            [6; GpuAddressInfo::BYTE_LEN]
        );
        assert_eq!(
            evaluate_info_query(
                &admitted_query(ResolvedInfoOperation::SamplerHost {
                    sampler: ResourceId::new(6, 1),
                    reply,
                }),
                0,
                &Evaluator,
            )
            .unwrap()
            .bytes(),
            [7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,]
        );
        assert_eq!(
            evaluate_info_query(
                &admitted_query(ResolvedInfoOperation::RateMapInfo {
                    rate_map: ResourceId::new(3, 1),
                    layer_count: 5,
                    reply,
                }),
                0,
                &Evaluator,
            )
            .unwrap()
            .bytes(),
            [10; 5]
        );
    }

    #[test]
    fn absent_admitted_position_refuses_before_evaluation() {
        let reply = ResolvedInfoReplyTarget {
            resource: ResourceId::new(1, 1),
            backing: BackingId::new(1),
            range: LinearRange::new(0, RenderPipelineStateInfo::BYTE_LEN as u64).unwrap(),
            requested_alignment: 4,
        };
        let admitted = admitted_query(ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::new(2, 1),
            reply,
        });
        assert_eq!(
            evaluate_info_query(&admitted, 1, &Evaluator),
            Err(InfoQueryEvaluationError::OperationAbsent(1))
        );
    }

    #[test]
    fn semantic_fixed_replies_encode_the_exact_little_endian_abi() {
        assert_eq!(
            RenderPipelineStateInfo {
                max_total_threads_per_threadgroup: 0x1122_3344,
                imageblock_sample_length: 0x5566_7788,
                threadgroup_size_matches_tile_size: true,
                supports_indirect_command_buffers: true,
            }
            .encode(),
            [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 1, 1, 0, 0,]
        );
        assert_eq!(
            ComputePipelineStateInfo {
                max_total_threads_per_threadgroup: 1,
                thread_execution_width: 2,
                static_threadgroup_memory_length: 3,
                supports_indirect_command_buffers: true,
            }
            .encode(),
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,]
        );
        assert_eq!(
            HeapTextureSizeAndAlignInfo {
                size: 0x1234,
                align: 0x100,
            }
            .encode(),
            [0x34, 0x12, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,]
        );
        assert_eq!(ImageblockMemoryLength(0x1234).encode(), [0x34, 0x12, 0, 0]);
        assert_eq!(
            GpuAddressInfo(0x0102_0304_0506_0708).encode(),
            [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
        assert_eq!(
            IndexedGpuResourceInfo {
                gpu_resource_id: 0x0102_0304_0506_0708,
                resource_index: 0x1112_1314_1516_1718,
            }
            .encode(),
            [
                0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13,
                0x12, 0x11,
            ]
        );
        assert_eq!(
            MappedRateCoordinate { x: 1.0, y: -2.0 }.encode(),
            [0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0]
        );
    }

    #[test]
    fn non_tile_render_imageblock_uses_two_dimensional_sample_storage() {
        let info = RenderPipelineStateInfo {
            imageblock_sample_length: 12,
            ..Default::default()
        };
        assert_eq!(
            render_pipeline_imageblock_memory_length(
                info,
                ImageblockDimensions {
                    width: 7,
                    height: 5,
                    depth: 3,
                },
            ),
            Some(ImageblockMemoryLength(420))
        );
        assert_eq!(
            render_pipeline_imageblock_memory_length(
                info,
                ImageblockDimensions {
                    width: u64::MAX,
                    height: 2,
                    depth: 1,
                },
            ),
            None
        );
    }
}
