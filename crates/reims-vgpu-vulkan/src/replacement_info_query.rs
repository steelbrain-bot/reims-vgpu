//! Native buffer updates for fully evaluated info replies.

use ash::vk;
use reims_vgpu_core::{PreparedInfoQuery, ResolvedInfoOperation, ResolvedResourceCompletion};
use reims_vgpu_protocol::BackingId;

use crate::replacement_buffer_blit::ReplacementBufferResolver;

const UPDATE_LIMIT: usize = 65_536;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeInfoQuery {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub bytes: Box<[u8]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplacementInfoQueryProgram<Info = ResolvedInfoOperation> {
    index: usize,
    transaction: reims_vgpu_protocol::TransactionId,
    operation: Info,
    backing: BackingId,
    native: NativeInfoQuery,
    completions: Box<[ResolvedResourceCompletion]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfoQueryRecordError {
    UnknownRepresentation,
    RangeOutOfBounds,
    AddressOverflow,
    MissingTransferDestination,
    UpdateAlignment,
}

impl ReplacementInfoQueryProgram<ResolvedInfoOperation> {
    pub fn resolve(
        prepared: &PreparedInfoQuery,
        resolver: &impl ReplacementBufferResolver,
    ) -> Result<Self, InfoQueryRecordError> {
        let destination = prepared.destination();
        let target = resolver
            .resolve_buffer(destination.backing, destination.representation)
            .ok_or(InfoQueryRecordError::UnknownRepresentation)?;
        if destination.region.end() > target.size {
            return Err(InfoQueryRecordError::RangeOutOfBounds);
        }
        if !target.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
            return Err(InfoQueryRecordError::MissingTransferDestination);
        }
        let offset = target
            .base_offset
            .checked_add(destination.region.start())
            .ok_or(InfoQueryRecordError::AddressOverflow)?;
        if !offset.is_multiple_of(4) || !prepared.bytes().len().is_multiple_of(4) {
            return Err(InfoQueryRecordError::UpdateAlignment);
        }
        Ok(Self {
            index: prepared.index(),
            transaction: prepared.transaction(),
            operation: *prepared.operation(),
            backing: destination.backing,
            native: NativeInfoQuery {
                buffer: target.buffer,
                offset,
                bytes: prepared.bytes().into(),
            },
            completions: prepared.resource_completions(),
        })
    }
}

impl<Info> ReplacementInfoQueryProgram<Info> {
    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    pub(crate) const fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        self.transaction
    }

    pub(crate) const fn operation(&self) -> &Info {
        &self.operation
    }

    pub(crate) const fn native(&self) -> &NativeInfoQuery {
        &self.native
    }

    pub(crate) const fn backing(&self) -> BackingId {
        self.backing
    }

    pub(crate) const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }
}

/// Record one or more contract-sized Vulkan update commands.
///
/// # Safety
///
/// `command_buffer` must be recording and `native.buffer` must remain live
/// through submission completion.
pub unsafe fn record_info_query(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    native: &NativeInfoQuery,
) {
    for (index, bytes) in native.bytes.chunks(UPDATE_LIMIT).enumerate() {
        let offset = native.offset + (index * UPDATE_LIMIT) as u64;
        unsafe { device.cmd_update_buffer(command_buffer, native.buffer, offset, bytes) };
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::replacement_buffer_blit::{NativeBufferTarget, ReplacementBufferResolver};
    use ash::vk::Handle;
    use reims_vgpu_core::{
        evaluate_info_query, prepare_info_query, BackingRegion, ComputePipelineStateInfo,
        EvaluatedInfoQuery, ExecTransaction, GpuAddressInfo, HeapTextureSizeAndAlignInfo,
        ImageblockMemoryLength, IndexedGpuResourceInfo, InfoQueryEvaluator, LinearRange,
        MappedRateCoordinate, RepresentationRoute, ResolvedExecSegment, ResolvedExecStream,
        ResolvedInfoOperation, ResolvedInfoReplyTarget, ResolvedOperation,
        ResolvedResourceLifecycle, ResourceLifecycleEffect, ResourceLifecycleOwner,
        SessionGeneration, StorageBacking,
    };
    use reims_vgpu_protocol::{
        BackingId, ComputePipelineObject, CoordinateMapDirection, HeapObject, ImageblockDimensions,
        RasterizationRateMapObject, RateMapCoordinate, RenderPipelineObject, ResourceId,
        ResourceObject, SamplerObject, SegmentBoundary, SegmentKind, SessionGenerationId,
        SubmissionId, SubmissionIdentity, TaskId, TextureDeclaration, TransactionId,
        VulkanDeviceEpochId,
    };

    struct RenderEvaluator(reims_vgpu_core::RenderPipelineStateInfo);

    impl InfoQueryEvaluator for RenderEvaluator {
        type Error = ();

        fn render_pipeline_state(
            &self,
            _pipeline: ResourceId<RenderPipelineObject>,
        ) -> Result<reims_vgpu_core::RenderPipelineStateInfo, Self::Error> {
            Ok(self.0)
        }

        fn compute_pipeline_state(
            &self,
            _: ResourceId<ComputePipelineObject>,
        ) -> Result<ComputePipelineStateInfo, Self::Error> {
            unreachable!()
        }

        fn buffer_host_resource(
            &self,
            _: ResourceId<ResourceObject>,
        ) -> Result<GpuAddressInfo, Self::Error> {
            unreachable!()
        }

        fn texture_host_resource(
            &self,
            _: ResourceId<ResourceObject>,
        ) -> Result<IndexedGpuResourceInfo, Self::Error> {
            unreachable!()
        }

        fn heap_host_resource(
            &self,
            _: ResourceId<HeapObject>,
        ) -> Result<GpuAddressInfo, Self::Error> {
            unreachable!()
        }

        fn sampler_host_resource(
            &self,
            _: ResourceId<SamplerObject>,
        ) -> Result<IndexedGpuResourceInfo, Self::Error> {
            unreachable!()
        }

        fn heap_texture_size_and_align(
            &self,
            _: TextureDeclaration,
        ) -> Result<HeapTextureSizeAndAlignInfo, Self::Error> {
            unreachable!()
        }

        fn render_pipeline_imageblock(
            &self,
            _: ResourceId<RenderPipelineObject>,
            _: ImageblockDimensions,
        ) -> Result<ImageblockMemoryLength, Self::Error> {
            unreachable!()
        }

        fn compute_pipeline_imageblock(
            &self,
            _: ResourceId<ComputePipelineObject>,
            _: ImageblockDimensions,
        ) -> Result<ImageblockMemoryLength, Self::Error> {
            unreachable!()
        }

        fn rate_map_info(
            &self,
            _: ResourceId<RasterizationRateMapObject>,
            _: u32,
        ) -> Result<Box<[u8]>, Self::Error> {
            unreachable!()
        }

        fn rate_parameter_buffer(
            &self,
            _: ResourceId<RasterizationRateMapObject>,
        ) -> Result<Box<[u8]>, Self::Error> {
            unreachable!()
        }

        fn map_coordinate(
            &self,
            _: CoordinateMapDirection,
            _: ResourceId<RasterizationRateMapObject>,
            _: u32,
            _: RateMapCoordinate,
        ) -> Result<MappedRateCoordinate, Self::Error> {
            unreachable!()
        }
    }

    pub(crate) fn evaluated_render_query(
        operation: ResolvedInfoOperation,
        value: reims_vgpu_core::RenderPipelineStateInfo,
    ) -> EvaluatedInfoQuery {
        let exec = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: reims_vgpu_core::ExecPrologue::default(),
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
        let mut runtime = reims_vgpu_core::TransactionRuntime::<()>::new(SessionGeneration::new(
            SessionGenerationId::new(1),
        ));
        let channel = reims_vgpu_protocol::ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 1)),
                exec,
            )
            .unwrap();
        evaluate_info_query(admitted.info_queries(), 0, &RenderEvaluator(value)).unwrap()
    }

    struct Resolver {
        backing: BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
    }

    impl ReplacementBufferResolver for Resolver {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: reims_vgpu_protocol::RepresentationId,
        ) -> Option<NativeBufferTarget> {
            (backing == self.backing && representation == self.representation).then_some(
                NativeBufferTarget {
                    buffer: vk::Buffer::from_raw(11),
                    base_offset: 32,
                    accessible_size: 128,
                    size: 128,
                    usage: vk::BufferUsageFlags::TRANSFER_DST,
                },
            )
        }
    }

    #[test]
    fn prepared_bytes_resolve_to_the_exact_native_window() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let backing = match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Linear(LinearRange::new(0, 128).unwrap())]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        };
        let representation = resources
            .create_execution_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let operation = reims_vgpu_core::ResolvedInfoOperation::RenderPipelineState {
            pipeline: ResourceId::<RenderPipelineObject>::new(2, 1),
            reply: ResolvedInfoReplyTarget {
                resource: ResourceId::new(3, 1),
                backing,
                range: LinearRange::new(8, 12).unwrap(),
                requested_alignment: 4,
            },
        };
        let prepared = prepare_info_query(
            &mut resources,
            SubmissionId::new(5),
            evaluated_render_query(
                operation,
                reims_vgpu_core::RenderPipelineStateInfo {
                    max_total_threads_per_threadgroup: 0xa5a5_a5a5,
                    imageblock_sample_length: 0xa5a5_a5a5,
                    threadgroup_size_matches_tile_size: true,
                    supports_indirect_command_buffers: true,
                },
            ),
        )
        .unwrap();
        let program = ReplacementInfoQueryProgram::resolve(
            &prepared,
            &Resolver {
                backing,
                representation,
            },
        )
        .unwrap();
        assert_eq!(program.index(), 0);
        assert_eq!(program.transaction(), TransactionId::new(1));
        assert_eq!(program.operation(), &operation);
        assert_eq!(program.native().offset, 40);
        assert_eq!(
            program.native().bytes.as_ref(),
            [0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 1, 1, 0, 0]
        );
        assert_eq!(
            program.completions(),
            prepared.resource_completions().as_ref()
        );
    }
}
