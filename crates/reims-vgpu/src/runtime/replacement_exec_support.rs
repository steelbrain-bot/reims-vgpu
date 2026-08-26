//! Shared framing helpers for lossless replacement EXEC decoding.

use crate::runtime::decode::stream::{
    Segment, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE, SEGMENT_TYPE_EVENT, SEGMENT_TYPE_INFO,
    SEGMENT_TYPE_RENDER,
};
use reims_vgpu_protocol::{SegmentBoundary, SegmentKind};

pub(crate) fn semantic_segment_boundary(
    stream_index: u32,
    segment: &Segment,
) -> Option<SegmentBoundary> {
    let kind = match segment.type_ {
        SEGMENT_TYPE_RENDER => SegmentKind::Render,
        SEGMENT_TYPE_COMPUTE => SegmentKind::Compute,
        SEGMENT_TYPE_BLIT => SegmentKind::Blit,
        SEGMENT_TYPE_EVENT => SegmentKind::Event,
        SEGMENT_TYPE_INFO => SegmentKind::Info,
        _ => return None,
    };
    Some(SegmentBoundary {
        stream_index,
        index: segment.index,
        kind,
        continues_previous: segment.continues_previous,
        continues_next: segment.continues_next,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InfoRecordDecline {
    Framing,
    OpcodeMismatch,
    Decode(reims_vgpu_protocol::InfoDecodeError),
}

impl crate::observe::Decline for InfoRecordDecline {
    fn slug(&self) -> &'static str {
        use reims_vgpu_protocol::InfoDecodeError as Error;
        match self {
            Self::Framing => "info_record_framing",
            Self::OpcodeMismatch => "info_record_opcode_mismatch",
            Self::Decode(Error::BadLength) => "info_record_bad_length",
            Self::Decode(Error::InvalidRateMapReplyLength(_)) => {
                "info_record_invalid_rate_map_reply_length"
            }
            Self::Decode(Error::Unsupported(kind)) => match kind {
                reims_vgpu_protocol::UnsupportedInfoOperation::IcbHostResource => {
                    "info_record_icb_host_resource_unsupported"
                }
                reims_vgpu_protocol::UnsupportedInfoOperation::RenderPipelineHostResource => {
                    "info_record_render_pipeline_host_resource_unsupported"
                }
                reims_vgpu_protocol::UnsupportedInfoOperation::ComputePipelineHostResource => {
                    "info_record_compute_pipeline_host_resource_unsupported"
                }
                reims_vgpu_protocol::UnsupportedInfoOperation::DepthStencilHostResource => {
                    "info_record_depth_stencil_host_resource_unsupported"
                }
            },
            Self::Decode(Error::UnknownOpcode(_)) => "info_record_unknown_opcode",
        }
    }
}

crate::observe::decline_display!(InfoRecordDecline);

pub(crate) fn classify_info_record(
    opcode: u32,
    bytes: &[u8],
) -> Result<reims_vgpu_protocol::InfoOperation, InfoRecordDecline> {
    let operation = reims_vgpu_wire::op(bytes, 0).map_err(|_| InfoRecordDecline::Framing)?;
    if operation.length() as usize != bytes.len() {
        return Err(InfoRecordDecline::Framing);
    }
    if operation.opcode() != opcode {
        return Err(InfoRecordDecline::OpcodeMismatch);
    }
    reims_vgpu_protocol::decode_info_operation(&operation).map_err(InfoRecordDecline::Decode)
}
