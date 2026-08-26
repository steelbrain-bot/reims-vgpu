//! Structural extraction of one wrapped AIR blob from a retained function.

pub(crate) const AIR_WRAP_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
const WRAPPER_HEADER_LEN: usize = 0x14;

pub(crate) fn extract_air(data: &[u8]) -> Result<&[u8], reims_vgpu_core::MtlbDecline> {
    let start = find_wrap_magic(data).ok_or(reims_vgpu_core::MtlbDecline::WrappedAirMissing {
        data_len: data.len(),
    })?;
    blob_at(data, start)
}

fn find_wrap_magic(data: &[u8]) -> Option<usize> {
    (data.len() >= WRAPPER_HEADER_LEN)
        .then(|| {
            data.windows(AIR_WRAP_MAGIC.len())
                .position(|window| window == AIR_WRAP_MAGIC)
        })
        .flatten()
}

fn blob_at(data: &[u8], offset: usize) -> Result<&[u8], reims_vgpu_core::MtlbDecline> {
    let header_end = offset.saturating_add(WRAPPER_HEADER_LEN);
    if header_end > data.len() {
        return Err(reims_vgpu_core::MtlbDecline::WrapperHeaderTruncated {
            offset,
            data_len: data.len(),
        });
    }
    let bitcode_offset = u32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap());
    let bitcode_size = u32::from_le_bytes(data[offset + 12..offset + 16].try_into().unwrap());
    let blob_len = u64::from(bitcode_offset) + u64::from(bitcode_size);
    let blob_end = usize::try_from(blob_len)
        .ok()
        .and_then(|length| offset.checked_add(length));
    if blob_len < WRAPPER_HEADER_LEN as u64 || blob_end.is_none_or(|end| end > data.len()) {
        return Err(reims_vgpu_core::MtlbDecline::BlobOutOfBounds {
            offset,
            blob_len,
            data_len: data.len(),
        });
    }
    Ok(&data[offset..blob_end.expect("the declared wrapper bounds were validated")])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_returns_exact_declared_wrapper() {
        let mut bytes = vec![0xaa; 32];
        bytes[3..7].copy_from_slice(&AIR_WRAP_MAGIC);
        bytes[11..15].copy_from_slice(&20_u32.to_le_bytes());
        bytes[15..19].copy_from_slice(&4_u32.to_le_bytes());
        assert_eq!(extract_air(&bytes).unwrap(), &bytes[3..27]);
    }
}
