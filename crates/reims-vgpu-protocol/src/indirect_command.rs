//! Semantic indirect-command-buffer mutation vocabulary.
//!
//! These operations are emitted by the blit encoder, but they do not copy
//! ordinary resource bytes. They mutate command slots later consumed by an
//! indirect render or compute execution, so they retain their own family after
//! the wire boundary.

use crate::{IndirectCommandBufferObject, SerializerRef};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectCommandRange {
    pub location: u64,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandOperation {
    Optimize {
        icb: SerializerRef<IndirectCommandBufferObject>,
        range: IndirectCommandRange,
    },
    Reset {
        icb: SerializerRef<IndirectCommandBufferObject>,
        range: IndirectCommandRange,
    },
    Copy {
        source: SerializerRef<IndirectCommandBufferObject>,
        source_range: IndirectCommandRange,
        destination: SerializerRef<IndirectCommandBufferObject>,
        destination_index: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndirectCommandDecodeError {
    BadLength,
    UnknownOpcode(u32),
}

/// Decode one already-framed indirect-command mutation record.
pub fn decode_indirect_command_operation(
    op: &reims_vgpu_wire::Op<'_>,
) -> Result<IndirectCommandOperation, IndirectCommandDecodeError> {
    use reims_vgpu_wire::ops::blit as wire;

    match op.opcode() {
        wire::OPCODE_OPTIMIZE_ICB | wire::OPCODE_RESET_ICB => {
            if op.length() != wire::ICB_RANGE_TOTAL_LEN {
                return Err(IndirectCommandDecodeError::BadLength);
            }
            let record = wire::icb_range(op).map_err(|_| IndirectCommandDecodeError::BadLength)?;
            let icb = SerializerRef::new(record.icb_ref.get());
            let range = IndirectCommandRange {
                location: record.range_location.get(),
                length: record.range_length.get(),
            };
            if op.opcode() == wire::OPCODE_OPTIMIZE_ICB {
                Ok(IndirectCommandOperation::Optimize { icb, range })
            } else {
                Ok(IndirectCommandOperation::Reset { icb, range })
            }
        }
        wire::OPCODE_COPY_ICB => {
            if op.length() != wire::COPY_ICB_TOTAL_LEN {
                return Err(IndirectCommandDecodeError::BadLength);
            }
            let record = wire::copy_icb(op).map_err(|_| IndirectCommandDecodeError::BadLength)?;
            Ok(IndirectCommandOperation::Copy {
                source: SerializerRef::new(record.source_ref.get()),
                source_range: IndirectCommandRange {
                    location: record.range_location.get(),
                    length: record.range_length.get(),
                },
                destination: SerializerRef::new(record.dest_ref.get()),
                destination_index: record.dest_index.get(),
            })
        }
        opcode => Err(IndirectCommandDecodeError::UnknownOpcode(opcode)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use reims_vgpu_wire::{op, ops::blit as wire};

    fn framed(opcode: u32, total_len: u32) -> Vec<u8> {
        let mut bytes = vec![0; total_len as usize];
        bytes[..4].copy_from_slice(&opcode.to_le_bytes());
        bytes[4..8].copy_from_slice(&total_len.to_le_bytes());
        bytes
    }

    #[test]
    fn all_three_wire_forms_decode_to_distinct_semantics() {
        for (opcode, expected) in [
            (
                wire::OPCODE_OPTIMIZE_ICB,
                IndirectCommandOperation::Optimize {
                    icb: SerializerRef::new(7),
                    range: IndirectCommandRange {
                        location: 11,
                        length: 13,
                    },
                },
            ),
            (
                wire::OPCODE_RESET_ICB,
                IndirectCommandOperation::Reset {
                    icb: SerializerRef::new(7),
                    range: IndirectCommandRange {
                        location: 11,
                        length: 13,
                    },
                },
            ),
        ] {
            let mut bytes = framed(opcode, wire::ICB_RANGE_TOTAL_LEN);
            bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
            bytes[12..20].copy_from_slice(&11u64.to_le_bytes());
            bytes[20..28].copy_from_slice(&13u64.to_le_bytes());
            assert_eq!(
                decode_indirect_command_operation(&op(&bytes, 0).unwrap()),
                Ok(expected)
            );
        }

        let mut bytes = framed(wire::OPCODE_COPY_ICB, wire::COPY_ICB_TOTAL_LEN);
        bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&5u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&17u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&19u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&23u64.to_le_bytes());
        assert_eq!(
            decode_indirect_command_operation(&op(&bytes, 0).unwrap()),
            Ok(IndirectCommandOperation::Copy {
                source: SerializerRef::new(3),
                source_range: IndirectCommandRange {
                    location: 17,
                    length: 19,
                },
                destination: SerializerRef::new(5),
                destination_index: 23,
            })
        );
    }

    #[test]
    fn framing_and_non_icb_opcodes_fail_closed() {
        let bytes = framed(wire::OPCODE_RESET_ICB, wire::ICB_RANGE_TOTAL_LEN - 4);
        assert_eq!(
            decode_indirect_command_operation(&op(&bytes, 0).unwrap()),
            Err(IndirectCommandDecodeError::BadLength)
        );
        let bytes = framed(wire::OPCODE_FILL_BUFFER, wire::FILL_BUFFER_TOTAL_LEN);
        assert_eq!(
            decode_indirect_command_operation(&op(&bytes, 0).unwrap()),
            Err(IndirectCommandDecodeError::UnknownOpcode(
                wire::OPCODE_FILL_BUFFER
            ))
        );
    }
}
