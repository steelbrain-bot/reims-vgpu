//! Immutable replacement transaction envelope derived from one decoded FIFO packet.

#![allow(dead_code)]

use reims_vgpu_core::{
    CompletionStamp, PrerequisiteResolution, ResolvedTransactionPrerequisite,
    TransactionPrerequisite,
};
use reims_vgpu_protocol::ChannelId;

/// The transport-owned facts shared by every root and child packet family.
///
/// A stamp record names the exact FIFO whose completion word it waits for. The
/// packet's own completion word belongs to the FIFO carrying the packet. Those
/// identities are derived here once, before opcode routing, so an adapter cannot
/// assign different prerequisite or publication slots to the same packet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementPacketEnvelope {
    pub channel: ChannelId,
    pub opcode: u16,
    pub payload: Box<[u8]>,
    pub prerequisites: Box<[ResolvedTransactionPrerequisite]>,
    pub completion_stamp: CompletionStamp,
}

impl ReplacementPacketEnvelope {
    pub(crate) fn from_packet(
        channel: ChannelId,
        packet: crate::runtime::fifo_packet::Packet,
    ) -> Self {
        let prerequisites = packet
            .stamp_waits
            .into_iter()
            .map(|wait| ResolvedTransactionPrerequisite {
                prerequisite: TransactionPrerequisite::Stamp { wait },
                resolution: PrerequisiteResolution::Pending,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            channel,
            opcode: packet.opcode,
            payload: packet.payload.into_boxed_slice(),
            prerequisites,
            completion_stamp: CompletionStamp::new(channel.get(), packet.completion_stamp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::StampWait;

    #[test]
    fn fifo_identity_derives_wait_sources_and_completion_slot() {
        let envelope = ReplacementPacketEnvelope::from_packet(
            ChannelId::new(6),
            crate::runtime::fifo_packet::Packet {
                opcode: 0x1e,
                stamp_waits: vec![
                    StampWait {
                        index: 2,
                        value: u32::MAX,
                    },
                    StampWait { index: 4, value: 0 },
                ],
                total_size: crate::model::PACKET_HEADER_LEN,
                completion_stamp: 9,
                payload: Vec::new(),
                next_head: crate::model::PACKET_HEADER_LEN,
            },
        );

        assert_eq!(envelope.channel, ChannelId::new(6));
        assert_eq!(envelope.completion_stamp, CompletionStamp::new(6, 9));
        assert_eq!(
            envelope.prerequisites.as_ref(),
            [
                ResolvedTransactionPrerequisite {
                    prerequisite: TransactionPrerequisite::Stamp {
                        wait: StampWait {
                            index: 2,
                            value: u32::MAX,
                        },
                    },
                    resolution: PrerequisiteResolution::Pending,
                },
                ResolvedTransactionPrerequisite {
                    prerequisite: TransactionPrerequisite::Stamp {
                        wait: StampWait { index: 4, value: 0 },
                    },
                    resolution: PrerequisiteResolution::Pending,
                },
            ]
        );
    }
}
