//! Semantic vocabulary of the info encoder.
//!
//! The encoder contains both queries and one resource-writing command. Opcode
//! selects the object namespace and reply shape; after this boundary no
//! consumer needs to compare a raw opcode.

use crate::{
    ComputePipelineObject, HeapObject, ObjectTableRef, RenderPipelineObject, ResourceObject,
    SamplerObject, SerializerRef, TextureDeclaration,
};

/// Marker for the rasterization-rate-map reference namespace.
pub enum RasterizationRateMapObject {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InfoReplyTarget {
    /// Heterogeneous object-table reference naming the destination buffer.
    pub buffer: ObjectTableRef<ResourceObject>,
    pub offset: u64,
    pub length: u32,
    pub alignment: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageblockDimensions {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateMapCoordinate {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceInfoKind {
    BufferHostResource,
    TextureHostResource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineStateInfoKind {
    Render,
    Compute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateMapDirection {
    ScreenToPhysical,
    PhysicalToScreen,
    Internal { command: u32 },
}

/// Closed semantic info operation surface currently emitted by the serializer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InfoOperation {
    PipelineState {
        kind: PipelineStateInfoKind,
        pipeline_ref: u32,
        reply: InfoReplyTarget,
    },
    ResourceHost {
        kind: ResourceInfoKind,
        /// Heterogeneous object-table reference, not a serializer-object name.
        resource: ObjectTableRef<ResourceObject>,
        reply: InfoReplyTarget,
    },
    HeapHost {
        heap: SerializerRef<HeapObject>,
        reply: InfoReplyTarget,
    },
    SamplerHost {
        sampler: SerializerRef<SamplerObject>,
        reply: InfoReplyTarget,
    },
    HeapTextureSizeAndAlign {
        descriptor: TextureDeclaration,
        reply: InfoReplyTarget,
    },
    RenderPipelineImageblock {
        pipeline: SerializerRef<RenderPipelineObject>,
        dimensions: ImageblockDimensions,
        reply: InfoReplyTarget,
    },
    ComputePipelineImageblock {
        pipeline: SerializerRef<ComputePipelineObject>,
        dimensions: ImageblockDimensions,
        reply: InfoReplyTarget,
    },
    RateMapInfo {
        rate_map: SerializerRef<RasterizationRateMapObject>,
        layer_count: u32,
        reply: InfoReplyTarget,
    },
    CopyRateParameterBuffer {
        rate_map: SerializerRef<RasterizationRateMapObject>,
        /// Heterogeneous object-table reference naming the destination buffer.
        destination: ObjectTableRef<ResourceObject>,
        destination_offset: u64,
    },
    MapCoordinate {
        direction: CoordinateMapDirection,
        rate_map: SerializerRef<RasterizationRateMapObject>,
        layer: u32,
        coordinate: RateMapCoordinate,
        reply: InfoReplyTarget,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfoOperationKind {
    RenderPipelineState,
    ComputePipelineState,
    BufferHostResource,
    TextureHostResource,
    HeapHostResource,
    SamplerHostResource,
    HeapTextureSizeAndAlign,
    RenderPipelineImageblock,
    ComputePipelineImageblock,
    RateMapInfo,
    CopyRateParameterBuffer,
    MapScreenToPhysical,
    MapPhysicalToScreen,
    MapCoordinateInternal,
}

impl InfoOperation {
    pub const fn kind(&self) -> InfoOperationKind {
        match self {
            Self::PipelineState {
                kind: PipelineStateInfoKind::Render,
                ..
            } => InfoOperationKind::RenderPipelineState,
            Self::PipelineState {
                kind: PipelineStateInfoKind::Compute,
                ..
            } => InfoOperationKind::ComputePipelineState,
            Self::ResourceHost {
                kind: ResourceInfoKind::BufferHostResource,
                ..
            } => InfoOperationKind::BufferHostResource,
            Self::ResourceHost {
                kind: ResourceInfoKind::TextureHostResource,
                ..
            } => InfoOperationKind::TextureHostResource,
            Self::HeapHost { .. } => InfoOperationKind::HeapHostResource,
            Self::SamplerHost { .. } => InfoOperationKind::SamplerHostResource,
            Self::HeapTextureSizeAndAlign { .. } => InfoOperationKind::HeapTextureSizeAndAlign,
            Self::RenderPipelineImageblock { .. } => InfoOperationKind::RenderPipelineImageblock,
            Self::ComputePipelineImageblock { .. } => InfoOperationKind::ComputePipelineImageblock,
            Self::RateMapInfo { .. } => InfoOperationKind::RateMapInfo,
            Self::CopyRateParameterBuffer { .. } => InfoOperationKind::CopyRateParameterBuffer,
            Self::MapCoordinate {
                direction: CoordinateMapDirection::ScreenToPhysical,
                ..
            } => InfoOperationKind::MapScreenToPhysical,
            Self::MapCoordinate {
                direction: CoordinateMapDirection::PhysicalToScreen,
                ..
            } => InfoOperationKind::MapPhysicalToScreen,
            Self::MapCoordinate {
                direction: CoordinateMapDirection::Internal { .. },
                ..
            } => InfoOperationKind::MapCoordinateInternal,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedInfoOperation {
    IcbHostResource,
    RenderPipelineHostResource,
    ComputePipelineHostResource,
    DepthStencilHostResource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfoDecodeError {
    BadLength,
    InvalidRateMapReplyLength(u32),
    Unsupported(UnsupportedInfoOperation),
    UnknownOpcode(u32),
}

/// Decode one already-framed info record into semantic meaning.
pub fn decode_info_operation(
    op: &reims_vgpu_wire::Op<'_>,
) -> Result<InfoOperation, InfoDecodeError> {
    use reims_vgpu_wire::ops::info as wire;

    let reply_target = |buffer, offset, layout: wire::ReplyLayout| InfoReplyTarget {
        buffer: ObjectTableRef::new(buffer),
        offset,
        length: layout.length,
        alignment: layout.alignment,
    };
    match op.opcode() {
        opcode if wire::is_query(opcode) => {
            let q = wire::query(op).map_err(|_| InfoDecodeError::BadLength)?;
            let layout = wire::fixed_reply_layout(opcode);
            let reply = layout
                .map(|layout| reply_target(q.reply_buffer_ref.get(), q.reply_offset.get(), layout));
            match opcode {
                wire::OPCODE_COMPUTE_PIPELINE_STATE_INFO => Ok(InfoOperation::PipelineState {
                    kind: PipelineStateInfoKind::Compute,
                    pipeline_ref: q.object_ref.get(),
                    reply: reply.expect("fixed layout exists for supported query"),
                }),
                wire::OPCODE_RENDER_PIPELINE_STATE_INFO => Ok(InfoOperation::PipelineState {
                    kind: PipelineStateInfoKind::Render,
                    pipeline_ref: q.object_ref.get(),
                    reply: reply.expect("fixed layout exists for supported query"),
                }),
                wire::OPCODE_BUFFER_HOST_RESOURCE_INFO => Ok(InfoOperation::ResourceHost {
                    kind: ResourceInfoKind::BufferHostResource,
                    resource: ObjectTableRef::new(q.object_ref.get()),
                    reply: reply.expect("fixed layout exists for supported query"),
                }),
                wire::OPCODE_TEXTURE_HOST_RESOURCE_INFO => Ok(InfoOperation::ResourceHost {
                    kind: ResourceInfoKind::TextureHostResource,
                    resource: ObjectTableRef::new(q.object_ref.get()),
                    reply: reply.expect("fixed layout exists for supported query"),
                }),
                wire::OPCODE_HEAP_HOST_RESOURCE_INFO => Ok(InfoOperation::HeapHost {
                    heap: SerializerRef::new(q.object_ref.get()),
                    reply: reply.expect("fixed layout exists for supported query"),
                }),
                wire::OPCODE_SAMPLER_HOST_RESOURCE_INFO => Ok(InfoOperation::SamplerHost {
                    sampler: SerializerRef::new(q.object_ref.get()),
                    reply: reply.expect("fixed layout exists for supported query"),
                }),
                wire::OPCODE_ICB_HOST_RESOURCE_INFO => Err(InfoDecodeError::Unsupported(
                    UnsupportedInfoOperation::IcbHostResource,
                )),
                wire::OPCODE_RENDER_PIPELINE_HOST_RESOURCE_INFO => {
                    Err(InfoDecodeError::Unsupported(
                        UnsupportedInfoOperation::RenderPipelineHostResource,
                    ))
                }
                wire::OPCODE_COMPUTE_PIPELINE_HOST_RESOURCE_INFO => {
                    Err(InfoDecodeError::Unsupported(
                        UnsupportedInfoOperation::ComputePipelineHostResource,
                    ))
                }
                wire::OPCODE_DEPTH_STENCIL_HOST_RESOURCE_INFO => Err(InfoDecodeError::Unsupported(
                    UnsupportedInfoOperation::DepthStencilHostResource,
                )),
                _ => unreachable!("wire::is_query is exhaustive"),
            }
        }
        wire::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN => {
            let q = wire::heap_texture_descriptor_size_and_align(op)
                .map_err(|_| InfoDecodeError::BadLength)?;
            let layout = wire::fixed_reply_layout(op.opcode()).unwrap();
            Ok(InfoOperation::HeapTextureSizeAndAlign {
                descriptor: crate::texture_declaration_from_narrow(&q.descriptor),
                reply: reply_target(q.reply_buffer_ref.get(), q.reply_offset.get(), layout),
            })
        }
        wire::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE => {
            let q = wire::heap_texture_descriptor_size_and_align_wide(op)
                .map_err(|_| InfoDecodeError::BadLength)?;
            let layout = wire::fixed_reply_layout(op.opcode()).unwrap();
            Ok(InfoOperation::HeapTextureSizeAndAlign {
                descriptor: crate::texture_declaration_from_wide(&q.descriptor),
                reply: reply_target(q.reply_buffer_ref.get(), q.reply_offset.get(), layout),
            })
        }
        opcode if wire::is_imageblock_query(opcode) => {
            let q = wire::imageblock_query(op).map_err(|_| InfoDecodeError::BadLength)?;
            let dimensions = ImageblockDimensions {
                width: q.width.get(),
                height: q.height.get(),
                depth: q.depth.get(),
            };
            let layout = wire::fixed_reply_layout(opcode).unwrap();
            let reply = reply_target(q.reply_buffer_ref.get(), q.reply_offset.get(), layout);
            if opcode == wire::OPCODE_RENDER_PIPELINE_IMAGEBLOCK {
                Ok(InfoOperation::RenderPipelineImageblock {
                    pipeline: SerializerRef::new(q.pipeline_ref.get()),
                    dimensions,
                    reply,
                })
            } else {
                Ok(InfoOperation::ComputePipelineImageblock {
                    pipeline: SerializerRef::new(q.pipeline_ref.get()),
                    dimensions,
                    reply,
                })
            }
        }
        wire::OPCODE_RATE_MAP_INFO => {
            let q = wire::rate_map_info(op).map_err(|_| InfoDecodeError::BadLength)?;
            let layout = wire::rate_map_reply_layout(q.reply_len.get()).ok_or(
                InfoDecodeError::InvalidRateMapReplyLength(q.reply_len.get()),
            )?;
            Ok(InfoOperation::RateMapInfo {
                rate_map: SerializerRef::new(q.rate_map_ref.get()),
                layer_count: wire::rate_map_layer_count(q.reply_len.get()).unwrap(),
                reply: reply_target(q.reply_buffer_ref.get(), q.reply_offset.get(), layout),
            })
        }
        wire::OPCODE_COPY_RATE_PARAMETER_BUFFER => {
            let c = wire::copy_rate_parameter_buffer(op).map_err(|_| InfoDecodeError::BadLength)?;
            Ok(InfoOperation::CopyRateParameterBuffer {
                rate_map: SerializerRef::new(c.rate_map_ref.get()),
                destination: ObjectTableRef::new(c.buffer_ref.get()),
                destination_offset: c.buffer_offset.get(),
            })
        }
        opcode
            if wire::is_map_coordinate(opcode) || op.length() == wire::MAP_COORDINATE_TOTAL_LEN =>
        {
            let m = wire::map_coordinate(op).map_err(|_| InfoDecodeError::BadLength)?;
            let direction = match opcode {
                wire::OPCODE_MAP_SCREEN_TO_PHYSICAL => CoordinateMapDirection::ScreenToPhysical,
                wire::OPCODE_MAP_PHYSICAL_TO_SCREEN => CoordinateMapDirection::PhysicalToScreen,
                command => CoordinateMapDirection::Internal { command },
            };
            Ok(InfoOperation::MapCoordinate {
                direction,
                rate_map: SerializerRef::new(m.rate_map_ref.get()),
                layer: m.layer.get(),
                coordinate: RateMapCoordinate {
                    x: m.x.get(),
                    y: m.y.get(),
                },
                reply: reply_target(
                    m.reply_buffer_ref.get(),
                    m.reply_offset.get(),
                    wire::ReplyLayout {
                        length: 8,
                        alignment: 4,
                    },
                ),
            })
        }
        opcode => Err(InfoDecodeError::UnknownOpcode(opcode)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use reims_vgpu_wire::ops::info as wire;

    fn record(opcode: u32, length: usize) -> alloc::vec::Vec<u8> {
        let mut bytes = vec![0; length];
        bytes[0..4].copy_from_slice(&opcode.to_le_bytes());
        bytes[4..8].copy_from_slice(&(length as u32).to_le_bytes());
        bytes
    }

    #[test]
    fn opcode_selects_object_namespace_and_exact_reply_layout() {
        let mut bytes = record(wire::OPCODE_TEXTURE_HOST_RESOURCE_INFO, 24);
        bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&9u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&32u64.to_le_bytes());
        let op = reims_vgpu_wire::op(&bytes, 0).unwrap();
        assert_eq!(
            decode_info_operation(&op),
            Ok(InfoOperation::ResourceHost {
                kind: ResourceInfoKind::TextureHostResource,
                resource: ObjectTableRef::new(7),
                reply: InfoReplyTarget {
                    buffer: ObjectTableRef::new(9),
                    offset: 32,
                    length: 16,
                    alignment: 4,
                },
            })
        );
    }

    #[test]
    fn variable_rate_map_reply_is_validated_before_semantics() {
        let mut bytes = record(wire::OPCODE_RATE_MAP_INFO, 32);
        bytes[24..28].copy_from_slice(&21u32.to_le_bytes());
        let op = reims_vgpu_wire::op(&bytes, 0).unwrap();
        assert_eq!(
            decode_info_operation(&op),
            Err(InfoDecodeError::InvalidRateMapReplyLength(21))
        );
    }

    #[test]
    fn unsupported_host_queries_are_named_not_collapsed_into_unknown() {
        let bytes = record(wire::OPCODE_ICB_HOST_RESOURCE_INFO, 24);
        let op = reims_vgpu_wire::op(&bytes, 0).unwrap();
        assert_eq!(
            decode_info_operation(&op),
            Err(InfoDecodeError::Unsupported(
                UnsupportedInfoOperation::IcbHostResource
            ))
        );
    }

    #[test]
    fn coordinate_command_value_does_not_become_a_new_operation_kind() {
        let mut bytes = record(0x77, wire::MAP_COORDINATE_TOTAL_LEN as usize);
        bytes[4..8].copy_from_slice(&wire::MAP_COORDINATE_TOTAL_LEN.to_le_bytes());
        let op = reims_vgpu_wire::op(&bytes, 0).unwrap();
        assert_eq!(
            decode_info_operation(&op).unwrap().kind(),
            InfoOperationKind::MapCoordinateInternal
        );
    }
}
