//! Typed replacement boundary for child-FIFO lifetime packets.

#![allow(dead_code)]

use reims_vgpu_core::endian::ld32;
use reims_vgpu_protocol::ChannelId;

const FIFO_CHANNEL_LEN: usize = size_of::<u32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementFifoChannel {
    pub channel: ChannelId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementFifoChannelDecodeError {
    Short { plen: usize },
    NotChildChannel(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementDeviceInfo {
    pub key_table_len: Option<u32>,
    pub count: u32,
    pub reply_pfn: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementSharedState {
    pub index: u32,
    pub pfn: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementDeviceInfoReply {
    pub gpa: u64,
    pub bytes: Box<[u8]>,
    pub answered_keys: Box<[u32]>,
    pub dropped_keys: Box<[u32]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementDeviceInfoReplyError {
    InvalidPageGeometry { page_shift: u32 },
    AddressOverflow { pfn: u32, page_shift: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodedReplacementRootPacket {
    SetupSharedState(DecodedReplacementSharedState),
    DeleteTask(crate::runtime::replacement_task::DecodedReplacementTaskDelete),
    DeviceInfo(DecodedReplacementDeviceInfo),
    DefineFifo(DecodedReplacementFifoChannel),
    FreeFifo(DecodedReplacementFifoChannel),
    SetObjectList(crate::runtime::replacement_task::DecodedReplacementObjectList),
    DefineTask(crate::runtime::replacement_task::DecodedReplacementTaskDefinition),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRootPacketRoute {
    Control(crate::runtime::replacement_session::ReplacementControlCommand),
    DeviceInfo(DecodedReplacementDeviceInfo),
}

impl DecodedReplacementRootPacket {
    pub(crate) fn route(self) -> ReplacementRootPacketRoute {
        use crate::runtime::replacement_session::ReplacementControlCommand as Control;
        match self {
            Self::SetupSharedState(command) => {
                ReplacementRootPacketRoute::Control(Control::SetupSharedState(command))
            }
            Self::DeleteTask(command) => {
                ReplacementRootPacketRoute::Control(Control::DeleteTask(command))
            }
            Self::DeviceInfo(command) => ReplacementRootPacketRoute::DeviceInfo(command),
            Self::DefineFifo(command) => {
                ReplacementRootPacketRoute::Control(Control::DefineFifo(command))
            }
            Self::FreeFifo(command) => {
                ReplacementRootPacketRoute::Control(Control::FreeFifo(command))
            }
            Self::SetObjectList(command) => {
                ReplacementRootPacketRoute::Control(Control::SetObjectList(command))
            }
            Self::DefineTask(command) => {
                ReplacementRootPacketRoute::Control(Control::DefineTask(command))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRootPacketDecodeError {
    UnknownOpcode(u16),
    Short {
        opcode: u16,
        plen: usize,
        need: usize,
    },
    Fifo(ReplacementFifoChannelDecodeError),
    DefineTask(crate::runtime::replacement_task::ReplacementTaskDefinitionDecodeError),
    SetObjectList(crate::runtime::replacement_task::ReplacementObjectListDecodeError),
    DeleteTask(crate::runtime::replacement_task::ReplacementTaskDeleteDecodeError),
}

#[derive(Debug)]
pub(crate) enum ReplacementAdmittedRootPacket<Semantic> {
    Control(crate::runtime::replacement_session::ReplacementAdmittedControl<Semantic>),
    Query(crate::runtime::replacement_session::ReplacementAdmittedQuery<Semantic>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRootPacketIngressError {
    Decode(ReplacementRootPacketDecodeError),
    Admission {
        reason: reims_vgpu_core::TransactionRuntimeError,
        route: ReplacementRootPacketRoute,
    },
}

pub(crate) fn admit_replacement_root_packet<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    packet: crate::runtime::fifo_packet::Packet,
) -> Result<ReplacementAdmittedRootPacket<Semantic>, ReplacementRootPacketIngressError> {
    let envelope = crate::runtime::replacement_packet::ReplacementPacketEnvelope::from_packet(
        reims_vgpu_protocol::ChannelId::new(0),
        packet,
    );
    let route = decode_replacement_root_packet(envelope.opcode, &envelope.payload)
        .map_err(ReplacementRootPacketIngressError::Decode)?
        .route();
    match &route {
        ReplacementRootPacketRoute::Control(command) => runtime
            .admit_control(
                envelope.channel,
                envelope.prerequisites,
                Some(envelope.completion_stamp),
                command.clone(),
            )
            .map(ReplacementAdmittedRootPacket::Control)
            .map_err(|reason| ReplacementRootPacketIngressError::Admission {
                reason,
                route: route.clone(),
            }),
        ReplacementRootPacketRoute::DeviceInfo(request) => runtime
            .admit_query(
                envelope.channel,
                envelope.prerequisites,
                Some(envelope.completion_stamp),
                crate::runtime::replacement_session::ReplacementQueryCommand::DeviceInfo(*request),
            )
            .map(ReplacementAdmittedRootPacket::Query)
            .map_err(|reason| ReplacementRootPacketIngressError::Admission {
                reason,
                route: route.clone(),
            }),
    }
}

pub(crate) fn decode_replacement_root_packet(
    opcode: u16,
    payload: &[u8],
) -> Result<DecodedReplacementRootPacket, ReplacementRootPacketDecodeError> {
    use crate::model::*;
    // The same never-fired signal the child ring has: a decoder is silent on
    // success, so nothing else says which root commands a guest actually sends.
    crate::runtime::contract_census::note(match opcode {
        ROOT_OP_SETUP_SHARED_STATE => "root_setup_shared_state",
        ROOT_OP_DELETE_TASK => "root_delete_task",
        ROOT_OP_DEVICE_INFO_MONTEREY => "root_device_info_monterey",
        ROOT_OP_DEVICE_INFO_TAHOE => "root_device_info_tahoe",
        ROOT_OP_DEFINE_FIFO => "root_define_fifo",
        ROOT_OP_FREE_FIFO => "root_free_fifo",
        ROOT_OP_SET_OBJECT_LIST => "root_set_object_list",
        ROOT_OP_DEFINE_TASK2 => "root_define_task2",
        _ => "root_unknown_opcode",
    });
    match opcode {
        ROOT_OP_SETUP_SHARED_STATE => {
            require_len(opcode, payload, CHILD_SHARED_STATE_LEN)?;
            Ok(DecodedReplacementRootPacket::SetupSharedState(
                DecodedReplacementSharedState {
                    index: ld32(&payload[CHILD_SHARED_STATE_INDEX..]),
                    pfn: ld32(&payload[CHILD_SHARED_STATE_PFN..]),
                },
            ))
        }
        ROOT_OP_DELETE_TASK => {
            crate::runtime::replacement_task::decode_replacement_task_delete(payload)
                .map(DecodedReplacementRootPacket::DeleteTask)
                .map_err(ReplacementRootPacketDecodeError::DeleteTask)
        }
        ROOT_OP_DEVICE_INFO_MONTEREY => {
            require_len(
                opcode,
                payload,
                DEVICE_INFO_MONTEREY_REPLY_PFN + size_of::<u32>(),
            )?;
            Ok(DecodedReplacementRootPacket::DeviceInfo(
                DecodedReplacementDeviceInfo {
                    key_table_len: None,
                    count: ld32(&payload[DEVICE_INFO_MONTEREY_COUNT..]),
                    reply_pfn: ld32(&payload[DEVICE_INFO_MONTEREY_REPLY_PFN..]),
                },
            ))
        }
        ROOT_OP_DEFINE_FIFO | ROOT_OP_FREE_FIFO => {
            let fifo = decode_replacement_fifo_channel(payload)
                .map_err(ReplacementRootPacketDecodeError::Fifo)?;
            if opcode == ROOT_OP_DEFINE_FIFO {
                Ok(DecodedReplacementRootPacket::DefineFifo(fifo))
            } else {
                Ok(DecodedReplacementRootPacket::FreeFifo(fifo))
            }
        }
        ROOT_OP_SET_OBJECT_LIST => {
            crate::runtime::replacement_task::decode_replacement_object_list(payload)
                .map(DecodedReplacementRootPacket::SetObjectList)
                .map_err(ReplacementRootPacketDecodeError::SetObjectList)
        }
        ROOT_OP_DEFINE_TASK2 => {
            crate::runtime::replacement_task::decode_replacement_task_definition(payload)
                .map(DecodedReplacementRootPacket::DefineTask)
                .map_err(ReplacementRootPacketDecodeError::DefineTask)
        }
        ROOT_OP_DEVICE_INFO_TAHOE => {
            require_len(
                opcode,
                payload,
                DEVICE_INFO_TAHOE_REPLY_PFN + size_of::<u32>(),
            )?;
            Ok(DecodedReplacementRootPacket::DeviceInfo(
                DecodedReplacementDeviceInfo {
                    key_table_len: Some(ld32(&payload[DEVICE_INFO_TAHOE_KEY_TABLE_LEN..])),
                    count: ld32(&payload[DEVICE_INFO_TAHOE_COUNT..]),
                    reply_pfn: ld32(&payload[DEVICE_INFO_TAHOE_REPLY_PFN..]),
                },
            ))
        }
        _ => Err(ReplacementRootPacketDecodeError::UnknownOpcode(opcode)),
    }
}

fn require_len(
    opcode: u16,
    payload: &[u8],
    need: usize,
) -> Result<(), ReplacementRootPacketDecodeError> {
    if payload.len() < need {
        Err(ReplacementRootPacketDecodeError::Short {
            opcode,
            plen: payload.len(),
            need,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn prepare_replacement_device_info_reply(
    limits: reims_vgpu_core::DeviceInfoLimits,
    page_shift: u32,
    version: u32,
    request: DecodedReplacementDeviceInfo,
) -> Result<Option<ReplacementDeviceInfoReply>, ReplacementDeviceInfoReplyError> {
    if request.reply_pfn == 0 {
        return Ok(None);
    }
    let page_size = 1usize
        .checked_shl(page_shift)
        .ok_or(ReplacementDeviceInfoReplyError::InvalidPageGeometry { page_shift })?;
    let max_pairs = page_size / crate::model::DEVICE_INFO_REPLY_PAIR_LEN;
    if max_pairs == 0 {
        return Err(ReplacementDeviceInfoReplyError::InvalidPageGeometry { page_shift });
    }
    let page_size_u64 = u64::try_from(page_size)
        .map_err(|_| ReplacementDeviceInfoReplyError::InvalidPageGeometry { page_shift })?;
    let gpa = u64::from(request.reply_pfn)
        .checked_mul(page_size_u64)
        .ok_or(ReplacementDeviceInfoReplyError::AddressOverflow {
            pfn: request.reply_pfn,
            page_shift,
        })?;
    let key_table_len = request.key_table_len.unwrap_or(u32::MAX);
    let caps = crate::model::device_info_caps(&limits, version)
        .into_iter()
        .filter(|(key, _)| *key < key_table_len)
        .collect::<Vec<_>>();
    let requested_pairs = usize::try_from(request.count).unwrap_or(usize::MAX);
    let answered = caps.len().min(requested_pairs).min(max_pairs);
    let write_sentinel = answered < requested_pairs && answered < max_pairs;
    let byte_len = (answered + usize::from(write_sentinel))
        .checked_mul(crate::model::DEVICE_INFO_REPLY_PAIR_LEN)
        .ok_or(ReplacementDeviceInfoReplyError::InvalidPageGeometry { page_shift })?;
    let mut bytes = vec![0; byte_len];
    for (index, (key, value)) in caps.iter().copied().take(answered).enumerate() {
        let offset = index * crate::model::DEVICE_INFO_REPLY_PAIR_LEN;
        reims_vgpu_core::endian::st32(&mut bytes[offset..], key);
        reims_vgpu_core::endian::st32(&mut bytes[offset + size_of::<u32>()..], value);
    }
    Ok(Some(ReplacementDeviceInfoReply {
        gpa,
        bytes: bytes.into_boxed_slice(),
        answered_keys: caps[..answered]
            .iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        dropped_keys: caps[answered..]
            .iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }))
}

pub(crate) fn decode_replacement_fifo_channel(
    payload: &[u8],
) -> Result<DecodedReplacementFifoChannel, ReplacementFifoChannelDecodeError> {
    if payload.len() < FIFO_CHANNEL_LEN {
        return Err(ReplacementFifoChannelDecodeError::Short {
            plen: payload.len(),
        });
    }
    let raw = ld32(payload);
    if !crate::model::is_child_channel(raw) {
        return Err(ReplacementFifoChannelDecodeError::NotChildChannel(raw));
    }
    Ok(DecodedReplacementFifoChannel {
        channel: ChannelId::new(raw),
    })
}

pub(crate) fn apply_replacement_fifo_definition<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    definition: DecodedReplacementFifoChannel,
) -> Result<(), reims_vgpu_core::TransactionRuntimeError> {
    runtime.define_channel(definition.channel)
}

pub(crate) fn apply_replacement_fifo_retirement<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    retirement: DecodedReplacementFifoChannel,
) -> Result<(), reims_vgpu_core::TransactionRuntimeError> {
    runtime.retire_channel(retirement.channel)
}

pub(crate) fn apply_replacement_shared_state<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    page_shift: u32,
    shared: DecodedReplacementSharedState,
) -> Result<
    crate::runtime::replacement_session::ReplacementDisplaySharedStatePublication,
    crate::runtime::replacement_session::ReplacementDisplaySharedStateError,
> {
    runtime.publish_display_shared_state(page_shift, shared.index, shared.pfn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_channel_decode_accepts_only_the_child_namespace() {
        assert_eq!(
            decode_replacement_fifo_channel(&1u32.to_le_bytes()),
            Ok(DecodedReplacementFifoChannel {
                channel: ChannelId::new(1),
            })
        );
        assert_eq!(
            decode_replacement_fifo_channel(&0u32.to_le_bytes()),
            Err(ReplacementFifoChannelDecodeError::NotChildChannel(0))
        );
        assert_eq!(
            decode_replacement_fifo_channel(&(crate::model::MAX_CHANNELS as u32).to_le_bytes()),
            Err(ReplacementFifoChannelDecodeError::NotChildChannel(
                crate::model::MAX_CHANNELS as u32,
            ))
        );
        assert_eq!(
            decode_replacement_fifo_channel(&[0; FIFO_CHANNEL_LEN - 1]),
            Err(ReplacementFifoChannelDecodeError::Short {
                plen: FIFO_CHANNEL_LEN - 1,
            })
        );
    }

    #[test]
    fn every_declared_root_opcode_decodes_to_one_typed_family() {
        use crate::model::*;
        let cases: &[(u16, usize)] = &[
            (ROOT_OP_SETUP_SHARED_STATE, CHILD_SHARED_STATE_LEN),
            (ROOT_OP_DELETE_TASK, crate::model::DELETE_TASK_LEN),
            (
                ROOT_OP_DEVICE_INFO_MONTEREY,
                DEVICE_INFO_MONTEREY_REPLY_PFN + size_of::<u32>(),
            ),
            (ROOT_OP_DEFINE_FIFO, FIFO_CHANNEL_LEN),
            (ROOT_OP_FREE_FIFO, FIFO_CHANNEL_LEN),
            (ROOT_OP_SET_OBJECT_LIST, SET_OBJECT_LIST_LEN),
            (ROOT_OP_DEFINE_TASK2, DEFINE_TASK_LEN),
            (
                ROOT_OP_DEVICE_INFO_TAHOE,
                DEVICE_INFO_TAHOE_REPLY_PFN + size_of::<u32>(),
            ),
        ];
        for &(opcode, len) in cases {
            let mut payload = vec![0; len];
            if matches!(opcode, ROOT_OP_DEFINE_FIFO | ROOT_OP_FREE_FIFO) {
                payload[..FIFO_CHANNEL_LEN].copy_from_slice(&1u32.to_le_bytes());
            }
            assert!(
                decode_replacement_root_packet(opcode, &payload).is_ok(),
                "root opcode {opcode:#x}"
            );
        }
        assert_eq!(
            decode_replacement_root_packet(u16::MAX, &[]),
            Err(ReplacementRootPacketDecodeError::UnknownOpcode(u16::MAX))
        );
    }

    #[test]
    fn root_device_info_generations_retain_their_distinct_ceiling_contract() {
        use crate::model::*;
        let mut old = [0u8; DEVICE_INFO_MONTEREY_REPLY_PFN + size_of::<u32>()];
        old[0..4].copy_from_slice(&7u32.to_le_bytes());
        old[4..8].copy_from_slice(&9u32.to_le_bytes());
        assert_eq!(
            decode_replacement_root_packet(ROOT_OP_DEVICE_INFO_MONTEREY, &old),
            Ok(DecodedReplacementRootPacket::DeviceInfo(
                DecodedReplacementDeviceInfo {
                    key_table_len: None,
                    count: 7,
                    reply_pfn: 9,
                }
            ))
        );
        let mut current = [0u8; DEVICE_INFO_TAHOE_REPLY_PFN + size_of::<u32>()];
        current[0..4].copy_from_slice(&5u32.to_le_bytes());
        current[4..8].copy_from_slice(&7u32.to_le_bytes());
        current[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert_eq!(
            decode_replacement_root_packet(ROOT_OP_DEVICE_INFO_TAHOE, &current),
            Ok(DecodedReplacementRootPacket::DeviceInfo(
                DecodedReplacementDeviceInfo {
                    key_table_len: Some(5),
                    count: 7,
                    reply_pfn: 9,
                }
            ))
        );
    }

    #[test]
    fn device_info_reply_is_one_exact_page_bounded_host_action() {
        let limits = reims_vgpu_core::DeviceInfoLimits {
            max_sample_count: 4,
            d24_stencil8: true,
            max_threads_per_threadgroup: [256, 128, 64],
            max_threadgroup_memory_bytes: 32_768,
            native_fp16: true,
        };
        let request = DecodedReplacementDeviceInfo {
            key_table_len: Some(4),
            count: 8,
            reply_pfn: 3,
        };
        let reply = prepare_replacement_device_info_reply(limits, 12, u32::MAX, request)
            .unwrap()
            .unwrap();
        assert_eq!(reply.gpa, 3 << 12);
        assert!(reply.answered_keys.iter().all(|key| *key < 4));
        assert!(reply.dropped_keys.is_empty());
        assert_eq!(
            reply.bytes.len(),
            (reply.answered_keys.len() + 1) * crate::model::DEVICE_INFO_REPLY_PAIR_LEN
        );
        assert_eq!(
            &reply.bytes[reply.bytes.len() - crate::model::DEVICE_INFO_REPLY_PAIR_LEN..],
            &[0; crate::model::DEVICE_INFO_REPLY_PAIR_LEN]
        );

        let one = prepare_replacement_device_info_reply(
            limits,
            12,
            u32::MAX,
            DecodedReplacementDeviceInfo {
                key_table_len: None,
                count: 1,
                reply_pfn: 3,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(one.answered_keys.len(), 1);
        assert!(!one.dropped_keys.is_empty());
        assert_eq!(one.bytes.len(), crate::model::DEVICE_INFO_REPLY_PAIR_LEN);
        assert_eq!(
            prepare_replacement_device_info_reply(
                limits,
                12,
                u32::MAX,
                DecodedReplacementDeviceInfo {
                    key_table_len: None,
                    count: 1,
                    reply_pfn: 0,
                },
            ),
            Ok(None)
        );
    }
}
