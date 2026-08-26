//! Borrowed root/child FIFO packet framing for replacement transport.
//!
//! This module owns only the shared ring framing contract. It does not dispatch
//! commands, mutate semantic state, or sample the compatibility drain census.

use crate::runtime::host::{HostMemory, MemError};
use reims_vgpu_core::endian::{ld16, ld32};
use reims_vgpu_core::StampWait;

use crate::model::{
    PACKET_COMPLETION_STAMP, PACKET_HEADER_LEN, PACKET_OPCODE, PACKET_STAMP_COUNT,
    PACKET_STAMP_LEN, PACKET_TOTAL_SIZE,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Packet {
    pub opcode: u16,
    pub stamp_waits: Vec<StampWait>,
    pub total_size: u32,
    pub completion_stamp: u32,
    pub payload: Vec<u8>,
    pub next_head: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PacketError {
    ShortHeader,
    BadSize,
    Incomplete,
    ShortSnapshot,
}

pub(crate) fn packet_snapshot_len(header: &[u8], available: u32, ring_capacity: u32) -> u32 {
    let total_size = ld32(&header[PACKET_TOTAL_SIZE..]);
    if total_size >= PACKET_HEADER_LEN && total_size <= ring_capacity && available >= total_size {
        total_size
    } else {
        PACKET_HEADER_LEN
    }
}

pub(crate) fn decode_packet(
    bytes: &[u8],
    head: u32,
    available: u32,
    ring_capacity: u32,
) -> Result<Packet, PacketError> {
    if available < PACKET_HEADER_LEN || bytes.len() < PACKET_HEADER_LEN as usize {
        return Err(PacketError::ShortHeader);
    }
    let opcode = ld16(&bytes[PACKET_OPCODE..]);
    let stamp_count = ld16(&bytes[PACKET_STAMP_COUNT..]);
    let total_size = ld32(&bytes[PACKET_TOTAL_SIZE..]);
    let completion_stamp = ld32(&bytes[PACKET_COMPLETION_STAMP..]);
    if total_size < PACKET_HEADER_LEN || total_size > ring_capacity {
        return Err(PacketError::BadSize);
    }
    if available < total_size {
        return Err(PacketError::Incomplete);
    }
    if (bytes.len() as u32) < total_size {
        return Err(PacketError::ShortSnapshot);
    }
    let stamps_bytes = u32::from(stamp_count) * PACKET_STAMP_LEN;
    let min_payload_off = PACKET_HEADER_LEN + stamps_bytes;
    if total_size < min_payload_off {
        return Err(PacketError::BadSize);
    }
    let stamp_waits = bytes[PACKET_HEADER_LEN as usize..min_payload_off as usize]
        .chunks_exact(PACKET_STAMP_LEN as usize)
        .map(|record| StampWait {
            index: ld32(record),
            value: ld32(&record[4..]),
        })
        .collect();
    Ok(Packet {
        opcode,
        stamp_waits,
        total_size,
        completion_stamp,
        payload: bytes[min_payload_off as usize..total_size as usize].to_vec(),
        next_head: head.wrapping_add(total_size),
    })
}

pub(crate) fn read_root_ring<M: HostMemory>(
    memory: &M,
    base_gpa: u64,
    ring_size: u32,
    absolute: u32,
    len: u32,
) -> Result<Vec<u8>, MemError> {
    let mut output = vec![0; len as usize];
    if ring_size == 0 || len == 0 {
        return Ok(output);
    }
    let mut copied = 0;
    while copied < len {
        let offset = absolute.wrapping_add(copied) % ring_size;
        let chunk = (ring_size - offset).min(len - copied);
        memory.read_gpa(
            base_gpa + u64::from(offset),
            &mut output[copied as usize..(copied + chunk) as usize],
        )?;
        copied += chunk;
    }
    Ok(output)
}

pub(crate) fn read_child_ring<M: HostMemory>(
    memory: &M,
    page_gpas: &[u64],
    ring_length: u32,
    absolute: u32,
    len: u32,
    page_shift: u32,
) -> Result<Vec<u8>, MemError> {
    let page_size = 1_u64 << page_shift;
    let mut output = vec![0; len as usize];
    if ring_length == 0 || page_gpas.is_empty() {
        return Ok(output);
    }
    for index in 0..len {
        let offset = absolute.wrapping_add(index) % ring_length;
        let page = u64::from(offset) >> page_shift;
        let page_offset = u64::from(offset) & (page_size - 1);
        if page as usize >= page_gpas.len() {
            continue;
        }
        memory.read_gpa(
            page_gpas[page as usize] + page_offset,
            &mut output[index as usize..=index as usize],
        )?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::{st16, st32};

    #[test]
    fn framing_retains_every_wait_and_payload_byte() {
        let mut bytes = vec![0; 23];
        st16(&mut bytes[PACKET_OPCODE..], 0x37);
        st16(&mut bytes[PACKET_STAMP_COUNT..], 1);
        st32(&mut bytes[PACKET_TOTAL_SIZE..], 23);
        st32(&mut bytes[PACKET_COMPLETION_STAMP..], 9);
        st32(&mut bytes[12..], 4);
        st32(&mut bytes[16..], 7);
        bytes[20..].copy_from_slice(&[1, 2, 3]);
        let packet = decode_packet(&bytes, 10, 23, 64).unwrap();
        assert_eq!(packet.opcode, 0x37);
        assert_eq!(packet.stamp_waits, vec![StampWait { index: 4, value: 7 }]);
        assert_eq!(packet.completion_stamp, 9);
        assert_eq!(packet.payload, [1, 2, 3]);
        assert_eq!(packet.next_head, 33);
    }
}
