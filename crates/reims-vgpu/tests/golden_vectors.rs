//! Golden-vector and cross-module smoke tests for protocol packages.
//!
//! The vectors were originally extracted from the C decoder unit-test matrices
//! under `host/utils`. That tree is deleted, so there is nothing left to run a
//! differential C↔Rust comparison against: these expectations are now the
//! source of truth for the values they cover, not a copy of one.

use reims_vgpu::protocol::endian::{st32, st64};
use reims_vgpu::protocol::pixel_format;
use reims_vgpu::runtime::decode::{blit, stream};

/// Never share the live product logs with a concurrent boot.
fn isolate_logs() {
    reims_vgpu::observe::redirect_logs_for_tests();
}

#[test]
fn pixel_format_c_matrix_rows() {
    isolate_logs();
    // IOSurface row expectations, from the deleted C pixel-format matrix, read
    // through the mapper rail's own row-bytes rule rather than a second copy of
    // it that only this vector reached.
    use reims_vgpu::protocol::iosurface_pages::packed_span_estimate;
    assert_eq!(
        packed_span_estimate(pixel_format::MTL_FORMAT_BGRA8_UNORM, 200, 1),
        Some(896)
    );
    assert_eq!(
        packed_span_estimate(pixel_format::MTL_FORMAT_RGBA16_FLOAT, 200, 1),
        Some(1664)
    );
}

#[test]
fn stream_segment_record_roundtrip_shape() {
    isolate_logs();
    let mut payload = Vec::new();
    // record: opcode 0x12d, length 0x28
    let mut rec = vec![0u8; 0x28];
    st32(&mut rec[0..], 0x12d);
    st32(&mut rec[4..], 0x28);
    payload.extend_from_slice(&rec);

    let mut stream_bytes = Vec::new();
    let seg_len = (8 + payload.len()) as u32;
    let mut hdr = [0u8; 8];
    st32(&mut hdr[0..], seg_len);
    hdr[4] = stream::SEGMENT_TYPE_BLIT;
    stream_bytes.extend_from_slice(&hdr);
    stream_bytes.extend_from_slice(&payload);

    let segs = stream::iter_segments(&stream_bytes).unwrap();
    assert_eq!(segs.len(), 1);
    let mut c = 0;
    let rec = stream::decode_first_record(&stream_bytes, &segs[0], &mut c).unwrap();
    assert_eq!(rec.opcode, 0x12d);
    let blit_cmd = blit::decode(&stream_bytes[rec.offset as usize..]).unwrap();
    assert_eq!(blit_cmd.opcode, 0x12d);
}

/// The same golden vector, through the lift that now reads it.
///
/// This device's own event decoder is gone; `reims_vgpu_protocol::decode::sync`
/// lifts the event rail for every encoder at once. The bytes are unchanged, so
/// the vector still says what it always said — a signal at ref 3 with value
/// `0x42` — and it now says it about the decoder in production.
#[test]
fn event_signal_golden() {
    use reims_vgpu::protocol::closure::Rail;
    use reims_vgpu::protocol::decode::sync::{decode, SyncRecord};
    isolate_logs();
    let mut v = vec![0u8; 0x14];
    st32(&mut v[0..], 0x191);
    st32(&mut v[4..], 0x14);
    st32(&mut v[8..], 3);
    st64(&mut v[12..], 0x42);
    let op = reims_vgpu::protocol::decode::op(&v, 0).expect("the vector frames");
    let SyncRecord::Event(record) = decode(Rail::Event, &op).expect("the vector lifts") else {
        panic!("0x191 is the event signal and lifts as one");
    };
    assert_eq!(record.event_ref, 3);
    assert_eq!(record.value, 0x42);
}

#[test]
fn corpus_property_random_decode_no_panic() {
    isolate_logs();
    // Smoke fuzz: random-ish buffers through all byte parsers.
    let seeds: &[&[u8]] = &[
        &[],
        &[0],
        &[0xff; 7],
        &[0x00; 64],
        &[0x12, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00],
    ];
    for s in seeds {
        let _ = stream::iter_segments(s);
        let _ = blit::decode(s);
        if let Ok(op) = reims_vgpu::protocol::decode::op(s, 0) {
            let _ = reims_vgpu::protocol::decode::sync::decode(
                reims_vgpu::protocol::closure::Rail::Event,
                &op,
            );
        }
        let _ = reims_vgpu::runtime::decode::compute::decode(s);
        let _ = reims_vgpu::runtime::decode::render::decode(s);
        let _ = reims_vgpu::runtime::decode::resource::decode_list_object_entry(s);
    }
}
