// DISCONNECTED SOURCE — this file does not compile, is not feature-gated, and
// is not linkable. No `mod` declaration reaches `dead/`. It is here to be read.
//
// What this was: this device's own command-stream framer,
// `runtime::decode::stream`. It located segments, located the records inside
// one, and re-validated an already-parsed `Segment` against the buffer on every
// record — sixteen `stream_reval_*` refusals guarding a state a caller should
// not have been able to construct.
//
// Why it went, three reasons:
//
//   * `reims_vgpu_protocol::segment::SegmentStream` frames segments and
//     `reims_vgpu_wire::op::OpStream` frames the records inside one. Both live
//     in the layers that own those layouts.
//   * `FramedSegment` carries its command window as a *slice*, so a segment
//     cannot claim bytes the buffer does not hold. The whole `validate_segment`
//     family is gone with the state it guarded, not merely moved.
//   * `decode_next_segment` found each segment's index with
//     `segment_index_for_offset`, which re-walks the stream from zero. That is
//     quadratic in the segment count, and a driven boot puts ninety-five
//     thousand segments in a stream. `SegmentStream` counts the index it
//     already knows.
//
// THE ONE BEHAVIOUR CHANGE, recorded here because this is where a reader will
// come looking for it: an unknown segment type. This framer reported it and
// walked on to the next segment. `SegmentStream` stops. The reason is on
// `FramingRefusal::UnknownType` — a family whose record framing is unknown can
// only be skipped on its declared length, and skipping on that hands the
// following segment an encoder state derived from bytes nothing here
// understands. `segment_disposition`'s own doc, below, records that the
// reference host "rejects new non-continuation types >= 4" rather than stepping
// over them, so stopping is the closer reading of the host being imitated.
//
// Do not resurrect any of this. If a boot regresses on a stream this framer
// walked and `SegmentStream` refuses, read it here and fix
// `reims_vgpu_protocol::segment`.

//! Command-stream framing decoder (port of `host/utils/reims-vgpu-stream-decode`).

use crate::protocol::checked::size_fits_u32;
use crate::protocol::endian::ld32;

// Segment types and header length from `reims-vgpu-wire` (observed serializer
// surface). Re-exported so stream walkers share one path with the wire crate.
//
// `SEGMENT_TYPE_INFO` is 4, not the next integer in sequence — a guess would
// write 3, which is `SEGMENT_TYPE_EVENT` and comes from `reims-vgpu-protocol`
// below. Protection options joined once the capture drove that envelope.
use reims_vgpu_wire::ops::segment as wire_segment;
pub use reims_vgpu_wire::ops::segment::{
    SEGMENT_HEADER_LEN, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE, SEGMENT_TYPE_INFO,
    SEGMENT_TYPE_PROTECTION_OPTIONS, SEGMENT_TYPE_RENDER,
};

// The one type the wire crate deliberately does not name, because its capture
// has never driven the encoder that writes it. It is not this module's either:
// naming a value no fixture wrote is assigning a meaning, and
// `reims-vgpu-protocol` is the layer that does that. It re-derives it there
// from the deserializer's contiguous `0..=3` decoder set, and this is the
// re-export so the two readings cannot drift.
pub use reims_vgpu_protocol::segment::SEGMENT_TYPE_EVENT;

/// Segment-header field offsets, from the view that derived them.
///
/// `SEGMENT_UNWRITTEN_OFFSET` is `size_of` rather than `offset_of` because the
/// wire struct deliberately stops at seven bytes: the eighth is the one the
/// serializer never writes, so it is the byte *after* the header rather than a
/// field in it, and saying so here is the whole difference between reading it
/// as ring contents and reading it as padding.
pub const SEGMENT_LENGTH_OFFSET: usize = core::mem::offset_of!(wire_segment::SegmentHeader, length);
pub const SEGMENT_TYPE_OFFSET: usize =
    core::mem::offset_of!(wire_segment::SegmentHeader, segment_type);
pub const SEGMENT_BEGIN_FLAG_OFFSET: usize =
    core::mem::offset_of!(wire_segment::SegmentHeader, begin_flag);
pub const SEGMENT_UNIDENTIFIED_OFFSET: usize =
    core::mem::offset_of!(wire_segment::SegmentHeader, unidentified_u8);
pub const SEGMENT_UNWRITTEN_OFFSET: usize = core::mem::size_of::<wire_segment::SegmentHeader>();

/// Record-header field offsets. This is the serializer's op header, a different
/// protocol level from the segment header above — see [`SEGMENT_HEADER_LEN`].
pub const RECORD_OPCODE_OFFSET: usize = core::mem::offset_of!(reims_vgpu_wire::OpHeader, opcode);
pub const RECORD_LENGTH_OFFSET: usize = core::mem::offset_of!(reims_vgpu_wire::OpHeader, length);
/// Serializer op-header length ([`reims_vgpu_wire::OP_HEADER_LEN`]). Distinct
/// from [`SEGMENT_HEADER_LEN`]: both are 8, but they frame different protocol
/// levels — do not treat them as interchangeable.
use reims_vgpu_wire::OP_HEADER_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    Ok,
    /// End of stream, or end of a segment's records. Control flow — the walkers
    /// terminate on it, so it is never a refusal and never reaches the log.
    Done,
    /// Refused; the payload is the registered slug naming which check refused.
    ///
    /// The payload is not decoration. This decoder frames *every* guest command,
    /// and a single coarse `ErrBadLength` covers seventeen checks here — a
    /// segment header disagreeing with the buffer, a record header disagreeing
    /// with its segment, and the re-validation of an already-parsed segment are
    /// three very different bugs that would otherwise arrive at the sink
    /// wearing one name.
    ErrArgs(&'static str),
    ErrShort(&'static str),
    ErrBadLength(&'static str),
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `stream_` prefix: seven modules under `runtime/decode/`
    /// define a type called `DecodeStatus`, and five of them have an `ErrShort`
    /// meaning a different read. Without the prefix the crate-wide uniqueness
    /// gate could not tell this decoder's refusals from any other's.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::Done => None,
            Self::ErrArgs(slug) | Self::ErrShort(slug) | Self::ErrBadLength(slug) => Some(slug),
        }
    }
}

/// One segment of the command stream, as decoded from its header.
///
/// The three bytes after `type_` are carried but never acted on: the device
/// reads the length and the type and nothing else, and their only reader is
/// `validate_segment`, which re-reads the header and refuses a stream whose
/// bytes moved under it. They are named for what the oracle measured, because
/// the names they had were three claims it has since settled — `cont` for the
/// `BOOL` argument of `-beginSegment:protectionOptions:`, `chain` for a byte
/// that has never been made to move, and `pad` for one the serializer does not
/// write at all. See [`reims_vgpu_wire::ops::segment::SegmentHeader`], which
/// records what each was perturbed with.
///
/// # The two the reader's contract does act on
///
/// A conforming consumer of this stream treats `begin_flag` and
/// `unidentified_u8` as **encoder-lifetime control**, and it is the only thing
/// it treats them as. `begin_flag` set means this segment's records continue
/// the encoder the previous segment left open, and are a protocol error if that
/// encoder is absent or of a different type — the reader is required to refuse
/// rather than quietly open a fresh one. `unidentified_u8` set means the
/// encoder survives this segment and the next one may continue it; clear means
/// the encoder ends here. A render segment that does *not* continue a previous
/// one begins by decoding a render-pass descriptor out of its own records.
///
/// So one Metal render command encoder — one Vulkan render pass — may span an
/// unbounded number of records across an unbounded number of segments, and this
/// device instead opens and ends a render pass per draw.
///
/// **Whether that costs anything here is a question about this guest**, and
/// [`SEGMENT_CHAIN_ROUTES`] answered it. Driven x86 Safari-drag boot:
///
/// ```text
/// seg_chain_none    94 860
/// seg_chain_next         0
/// seg_chain_prev         0
/// seg_chain_both         0
/// ```
///
/// **Every segment of the boot.** This guest never asks for an encoder to
/// outlive a segment, so there is nothing here to honour and no pass to hold
/// open across one. Keep the routes: this is a property of a workload, not of
/// the protocol, and a guest that did chain would arrive as a non-zero here
/// rather than as a rendering defect.
///
/// The narrower question — whether a *single* segment carries more draws than
/// the one render pass this device opens per draw — has the same answer, from
/// the same boot: **94 860 segments against 96 351 draws**, so a render segment
/// carries about one. A render pass per draw is what this guest asks for, not a
/// translation artifact, and the contract permitting an unbounded number of
/// draws per encoder does not mean this workload presents one.
///
/// So the submission and render-pass granularity are both already the guest's.
/// Neither is where this device's remaining cost is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Segment {
    pub offset: u32,
    pub length: u32,
    pub type_: u8,
    /// `-beginSegment:`'s unnamed `BOOL` first argument, verbatim. The reader's
    /// contract makes this "continue the open encoder"; see the type doc.
    pub begin_flag: u8,
    /// Written, always `0` in every fixture. The reader's contract makes this
    /// "the encoder outlives this segment"; see the type doc.
    pub unidentified_u8: u8,
    /// The eighth header byte, which neither `-beginSegment:` call writes. On a
    /// real wire it is whatever the ring last contained, so it is not padding
    /// and nothing may read it as a value.
    pub unwritten_u8: u8,
    pub command_offset: u32,
    pub command_length: u32,
    pub index: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub segment_index: u32,
    pub segment_type: u32,
    pub offset: u32,
    pub length: u32,
    pub opcode: u32,
    /// Absolute offset of the record header in the stream bytes.
    pub bytes_offset: u32,
}

pub fn segment_type_name(type_: u32) -> &'static str {
    match type_ {
        0 => "render",
        1 => "compute",
        2 => "blit",
        3 => "event",
        4 => "info",
        5 => "protection-options",
        _ => "unknown",
    }
}

/// What the stream walker should do with a segment family.
///
/// This exists so the walker's "everything else" arm is a decision rather than a
/// fallthrough. It used to be `_ => {}`, which gave the same silence to a ref-texture
/// envelope the contract says to skip and to a segment family the host has never
/// seen — and the second of those is unknown wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentDisposition {
    /// A family with a record walker: render, compute, blit, event, info.
    Walk,
    /// Type 5. `-beginSegment:protectionOptions:` emits a segment-level envelope
    /// *before* the real segment. Skipping it is contract-correct, so it is
    /// control flow and stays silent — logging it would put a line in the sink
    /// on every healthy frame that carries one.
    ///
    /// # Its command window is the protection value, and this doc used to deny
    /// that
    ///
    /// The wording here was "raw envelope bytes carrying no decodable protection
    /// value", which was a guess written where a measurement now sits. Driven
    /// under `-setSupportsProtectionOptionsEnvelope:` the burst is exactly three
    /// records: this ref-texture header, then **eight fully-written bytes that are the
    /// `protectionOptions:` argument verbatim**, then the ordinary segment
    /// header. `reims_vgpu_wire::ops::segment::ProtectionOptionsEnvelope` is the
    /// view; `blit_begin_segment_protected` sends `0x44` and `..._alt` sends
    /// `0x33`, so it is the guest's value and not a constant.
    ///
    /// Skipping stays right — this device implements no protection domains, so
    /// there is nothing to do with the value — but "we cannot read it" and "we
    /// choose not to act on it" are different claims and only the second is
    /// true.
    ///
    /// The envelope needs **both** conditions: the `BOOL` argument clear *and*
    /// non-zero options. Either alone emits the ordinary single header, which is
    /// measured by `blit_begin_segment_protected_flag_set` and
    /// `blit_begin_segment_protection_zero` respectively.
    Envelope,
    /// A family this host has no contract for. MetalSerializer's deserializer
    /// constructs decoders for `0..3` and rejects new non-continuation types
    /// `>= 4`, so a type past the known set is not something to guess at:
    /// refuse it visibly instead of walking its bytes as records.
    Unknown,
}

impl crate::observe::Refusal for SegmentDisposition {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Walk | Self::Envelope => None,
            Self::Unknown => Some("stream_segment_type_unknown"),
        }
    }
}

/// The disposition of a segment type, derived from the one parse that owns it.
///
/// `reims_vgpu_protocol::segment::segment_role` decides which byte is which
/// family; this only says what *this* walker does with each answer. Listing the
/// five record-bearing types again here would be a second copy of the segment
/// map, and the failure mode of a second copy is a family admitted by one and
/// refused by the other.
pub fn segment_disposition(type_: u8) -> SegmentDisposition {
    use reims_vgpu_protocol::segment::SegmentRole;
    match reims_vgpu_protocol::segment::segment_role(type_) {
        Some(SegmentRole::Encoder(_)) => SegmentDisposition::Walk,
        Some(SegmentRole::ProtectionEnvelope) => SegmentDisposition::Envelope,
        None => SegmentDisposition::Unknown,
    }
}

fn validate_bytes(bytes: &[u8]) -> DecodeStatus {
    if !size_fits_u32(bytes.len()) {
        return DecodeStatus::ErrBadLength("stream_bytes_len_overflow");
    }
    DecodeStatus::Ok
}

fn segment_index_for_offset(bytes: &[u8], target_offset: u32) -> Result<u32, DecodeStatus> {
    let mut cursor = 0usize;
    let mut index = 0u32;
    while cursor < bytes.len() {
        if bytes.len() - cursor < SEGMENT_HEADER_LEN {
            return Err(DecodeStatus::ErrShort("stream_index_walk_short_header"));
        }
        if !size_fits_u32(cursor) {
            return Err(DecodeStatus::ErrBadLength(
                "stream_index_walk_cursor_overflow",
            ));
        }
        if cursor as u32 == target_offset {
            return Ok(index);
        }
        let segment_len = ld32(&bytes[cursor + SEGMENT_LENGTH_OFFSET..]) as usize;
        if segment_len < SEGMENT_HEADER_LEN || segment_len > bytes.len() - cursor {
            return Err(DecodeStatus::ErrBadLength("stream_index_walk_seg_len"));
        }
        cursor += segment_len;
        index += 1;
    }
    Err(DecodeStatus::ErrBadLength(
        "stream_index_target_offset_not_found",
    ))
}

/// Decode the next segment at `cursor`. On Ok advances cursor. Transactional: no partial out.
pub fn decode_next_segment(bytes: &[u8], cursor: &mut usize) -> Result<Segment, DecodeStatus> {
    let status = validate_bytes(bytes);
    if status != DecodeStatus::Ok {
        return Err(status);
    }
    if *cursor > bytes.len() {
        return Err(DecodeStatus::ErrArgs("stream_seg_cursor_past_end"));
    }
    if *cursor == bytes.len() {
        return Err(DecodeStatus::Done);
    }
    if bytes.len() - *cursor < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrShort("stream_seg_short_header"));
    }
    let header = &bytes[*cursor..];
    let segment_len = ld32(&header[SEGMENT_LENGTH_OFFSET..]) as usize;
    if segment_len < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_seg_len_below_header"));
    }
    if segment_len > bytes.len() - *cursor {
        return Err(DecodeStatus::ErrBadLength("stream_seg_len_past_buffer_end"));
    }
    if !size_fits_u32(*cursor) {
        return Err(DecodeStatus::ErrBadLength("stream_seg_cursor_overflow"));
    }
    let segment_index = segment_index_for_offset(bytes, *cursor as u32)?;
    crate::runtime::drain::note_store_route(segment_chain_route(
        header[SEGMENT_BEGIN_FLAG_OFFSET],
        header[SEGMENT_UNIDENTIFIED_OFFSET],
    ));
    let out = Segment {
        offset: *cursor as u32,
        length: segment_len as u32,
        type_: header[SEGMENT_TYPE_OFFSET],
        begin_flag: header[SEGMENT_BEGIN_FLAG_OFFSET],
        unidentified_u8: header[SEGMENT_UNIDENTIFIED_OFFSET],
        unwritten_u8: header[SEGMENT_UNWRITTEN_OFFSET],
        command_offset: (*cursor + SEGMENT_HEADER_LEN) as u32,
        command_length: (segment_len - SEGMENT_HEADER_LEN) as u32,
        index: segment_index,
    };
    *cursor += segment_len;
    Ok(out)
}

/// Every census route [`segment_chain_route`] can answer, in the order
/// `(continues_previous, continues_into_next)` counts up.
///
/// Exported so a reading is over a named set rather than over whichever names a
/// grep of the log happened to find, and so the four cannot be spelled twice.
pub const SEGMENT_CHAIN_ROUTES: [&str; 4] = [
    "seg_chain_none",
    "seg_chain_next",
    "seg_chain_prev",
    "seg_chain_both",
];

/// Which of [`SEGMENT_CHAIN_ROUTES`] a segment header's two encoder-lifetime
/// bytes select.
///
/// Any non-zero is a set flag: the bytes carry a `BOOL` and the stream is
/// guest-controlled, so `!= 0` is the test rather than `== 1`. A guest that
/// wrote `2` would otherwise be counted as "not chaining" and the whole reading
/// would be wrong in the direction that reads as a settled question.
pub fn segment_chain_route(continues_previous: u8, continues_into_next: u8) -> &'static str {
    let index = usize::from(continues_previous != 0) << 1 | usize::from(continues_into_next != 0);
    SEGMENT_CHAIN_ROUTES[index]
}

fn validate_segment(bytes: &[u8], segment: &Segment) -> Result<usize, DecodeStatus> {
    let status = validate_bytes(bytes);
    if status != DecodeStatus::Ok {
        return Err(status);
    }
    if (segment.length as usize) < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_reval_len_below_header"));
    }
    if (segment.offset as usize) > bytes.len()
        || (segment.length as usize) > bytes.len() - segment.offset as usize
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_span_oob"));
    }
    let header = &bytes[segment.offset as usize..];
    if ld32(&header[SEGMENT_LENGTH_OFFSET..]) != segment.length
        || header[SEGMENT_TYPE_OFFSET] != segment.type_
        || header[SEGMENT_BEGIN_FLAG_OFFSET] != segment.begin_flag
        || header[SEGMENT_UNIDENTIFIED_OFFSET] != segment.unidentified_u8
        || header[SEGMENT_UNWRITTEN_OFFSET] != segment.unwritten_u8
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_header_mismatch"));
    }
    if segment.command_offset != segment.offset + SEGMENT_HEADER_LEN as u32
        || segment.command_length != segment.length - SEGMENT_HEADER_LEN as u32
    {
        return Err(DecodeStatus::ErrBadLength(
            "stream_reval_command_span_mismatch",
        ));
    }
    if segment.command_offset < segment.offset
        || segment.command_length > u32::MAX - segment.command_offset
    {
        return Err(DecodeStatus::ErrBadLength(
            "stream_reval_command_offset_overflow",
        ));
    }
    let command_end = segment.command_offset as usize + segment.command_length as usize;
    if (segment.command_offset as usize) > command_end
        || command_end > segment.offset as usize + segment.length as usize
        || command_end > bytes.len()
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_command_end_oob"));
    }
    Ok(command_end)
}

pub fn decode_next_record(
    bytes: &[u8],
    segment: &Segment,
    cursor: &mut usize,
) -> Result<Record, DecodeStatus> {
    let command_end = validate_segment(bytes, segment)?;
    if *cursor < segment.command_offset as usize || *cursor > command_end {
        return Err(DecodeStatus::ErrArgs("stream_rec_cursor_out_of_segment"));
    }
    if segment.type_ == SEGMENT_TYPE_PROTECTION_OPTIONS {
        if *cursor != segment.command_offset as usize && *cursor != command_end {
            return Err(DecodeStatus::ErrArgs(
                "stream_rec_protection_cursor_misaligned",
            ));
        }
        *cursor = command_end;
        return Err(DecodeStatus::Done);
    }
    if *cursor == command_end {
        return Err(DecodeStatus::Done);
    }
    if command_end - *cursor < OP_HEADER_LEN {
        return Err(DecodeStatus::ErrShort("stream_rec_short_header"));
    }
    let header = &bytes[*cursor..];
    let opcode = ld32(&header[RECORD_OPCODE_OFFSET..]);
    let record_len = ld32(&header[RECORD_LENGTH_OFFSET..]) as usize;
    if record_len < OP_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_rec_len_below_header"));
    }
    if record_len > command_end - *cursor {
        return Err(DecodeStatus::ErrBadLength(
            "stream_rec_len_past_segment_end",
        ));
    }
    if !size_fits_u32(*cursor) {
        return Err(DecodeStatus::ErrBadLength("stream_rec_cursor_overflow"));
    }
    let out = Record {
        segment_index: segment.index,
        segment_type: segment.type_ as u32,
        offset: *cursor as u32,
        length: record_len as u32,
        opcode,
        bytes_offset: *cursor as u32,
    };
    *cursor += record_len;
    Ok(out)
}

pub fn decode_first_record(
    bytes: &[u8],
    segment: &Segment,
    cursor: &mut usize,
) -> Result<Record, DecodeStatus> {
    *cursor = segment.command_offset as usize;
    decode_next_record(bytes, segment, cursor)
}

/// Iterate all segments.
pub fn iter_segments(bytes: &[u8]) -> Result<Vec<Segment>, DecodeStatus> {
    let mut cursor = 0usize;
    let mut out = Vec::new();
    loop {
        match decode_next_segment(bytes, &mut cursor) {
            Ok(s) => out.push(s),
            Err(DecodeStatus::Done) => return Ok(out),
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::endian::st32;

    /// The four chain routes are distinct, and each pair of flags selects its
    /// own. A collision would fold two populations into one bucket, which is
    /// the failure that reads as an answer: `seg_chain_none` carrying the whole
    /// census is the reading that says this guest never chains encoders, and it
    /// must not also be what a mis-indexed table says.
    #[test]
    fn each_pair_of_chain_flags_selects_its_own_route() {
        let mut seen = std::collections::HashSet::new();
        for route in SEGMENT_CHAIN_ROUTES {
            assert!(
                seen.insert(route),
                "two chain routes share the name {route}"
            );
        }
        assert_eq!(segment_chain_route(0, 0), "seg_chain_none");
        assert_eq!(segment_chain_route(0, 1), "seg_chain_next");
        assert_eq!(segment_chain_route(1, 0), "seg_chain_prev");
        assert_eq!(segment_chain_route(1, 1), "seg_chain_both");
    }

    /// The bytes are guest-controlled, so the flag test is `!= 0` and not
    /// `== 1`. A guest writing any other truthy value must not be counted as
    /// "did not chain" — that would put a segment the reader's contract says
    /// continues an open encoder into the bucket whose emptiness is the whole
    /// question.
    #[test]
    fn any_non_zero_chain_byte_is_a_set_flag() {
        for v in [1u8, 2, 0x7f, 0x80, 0xff] {
            assert_eq!(segment_chain_route(v, 0), "seg_chain_prev", "prev={v:#x}");
            assert_eq!(segment_chain_route(0, v), "seg_chain_next", "next={v:#x}");
            assert_eq!(segment_chain_route(v, v), "seg_chain_both", "both={v:#x}");
        }
    }

    fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
        let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], len);
        hdr[4] = type_;
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    fn push_record(buf: &mut Vec<u8>, opcode: u32, payload: &[u8]) {
        let len = (OP_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], len);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    #[test]
    fn empty_stream_done() {
        let mut c = 0;
        assert_eq!(
            decode_next_segment(&[], &mut c).unwrap_err(),
            DecodeStatus::Done
        );
        assert_eq!(c, 0);
    }

    #[test]
    fn single_blit_segment_with_record() {
        let mut payload = Vec::new();
        push_record(&mut payload, 0x12d, &[0u8; 0x18]); // buffer-to-buffer shape
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_BLIT, &payload);

        let segs = iter_segments(&stream).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].type_, SEGMENT_TYPE_BLIT);
        assert_eq!(segs[0].index, 0);

        let mut rc = 0;
        let rec = decode_first_record(&stream, &segs[0], &mut rc).unwrap();
        assert_eq!(rec.opcode, 0x12d);
        assert_eq!(
            decode_next_record(&stream, &segs[0], &mut rc).unwrap_err(),
            DecodeStatus::Done
        );
    }

    #[test]
    fn short_and_bad_length_name_the_check_that_refused() {
        use crate::observe::Refusal;
        // Asserting the slug rather than the variant is the point: both of these
        // used to be one `ErrBadLength`/`ErrShort` shared with sixteen other
        // checks, so a passing test said nothing about *which* read disagreed.
        assert_eq!(
            decode_next_segment(&[1, 2, 3], &mut 0)
                .unwrap_err()
                .refusal(),
            Some("stream_seg_short_header")
        );
        let mut bad = [0u8; 8];
        st32(&mut bad[0..4], 4); // length < header
        assert_eq!(
            decode_next_segment(&bad, &mut 0).unwrap_err().refusal(),
            Some("stream_seg_len_below_header")
        );
        // A segment header that outruns the buffer is a different bug from one
        // that undershoots its own header, and now says so.
        let mut past = [0u8; 8];
        st32(&mut past[0..4], 64);
        assert_eq!(
            decode_next_segment(&past, &mut 0).unwrap_err().refusal(),
            Some("stream_seg_len_past_buffer_end")
        );
    }

    #[test]
    fn end_of_stream_and_end_of_segment_are_never_refusals() {
        use crate::observe::Refusal;
        // `Done` is how both walkers terminate. If it ever reported a reason the
        // sink would carry one line per segment per frame — the flood that the
        // speculative-return carve-out exists to prevent.
        assert_eq!(DecodeStatus::Done.refusal(), None);
        assert_eq!(DecodeStatus::Ok.refusal(), None);

        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_RENDER, &[]);
        let segs = iter_segments(&stream).unwrap();
        let mut c = 0;
        assert_eq!(
            decode_first_record(&stream, &segs[0], &mut c)
                .unwrap_err()
                .refusal(),
            None
        );
        let mut sc = stream.len();
        assert_eq!(
            decode_next_segment(&stream, &mut sc).unwrap_err().refusal(),
            None
        );
    }

    #[test]
    fn every_refusal_in_this_decoder_carries_a_registered_slug() {
        use crate::observe::Refusal;
        // What this pins is that no site returns a refusal whose payload is
        // empty or absent, which would render `reason=` bare.
        for status in [
            DecodeStatus::ErrArgs("stream_seg_cursor_past_end"),
            DecodeStatus::ErrShort("stream_seg_short_header"),
            DecodeStatus::ErrBadLength("stream_bytes_len_overflow"),
        ] {
            let slug = status.refusal().expect("a refusal names its check");
            assert!(
                slug.starts_with("stream_"),
                "{slug} lacks the module prefix"
            );
        }
    }

    #[test]
    fn multi_segment_indices() {
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_RENDER, &[]);
        push_segment(&mut stream, SEGMENT_TYPE_COMPUTE, &[]);
        let segs = iter_segments(&stream).unwrap();
        assert_eq!(segs[0].index, 0);
        assert_eq!(segs[1].index, 1);
        assert_eq!(segment_type_name(0), "render");
    }

    #[test]
    fn property_fuzz_random_headers() {
        // Smoke: random-ish short buffers must not panic.
        for n in 0..32usize {
            let bytes = vec![0xAAu8; n];
            let mut c = 0;
            let _ = decode_next_segment(&bytes, &mut c);
        }
    }
}

// ---- The legacy tests that moved with it, from `runtime::exec::tests` ----
//
// Both of these had subjects that no longer exist. The first constructed a
// `Segment` naming bytes the buffer did not hold — unrepresentable now — and
// the second asked `segment_disposition` for a verdict the framer makes
// itself. See README.md for what replaced each.
#[test]
fn a_truncated_segment_names_the_check_rather_than_looking_like_end_of_records() {
    use reims_vgpu_protocol::segment::{
        segment_type_name, Segment, SEGMENT_HEADER_LEN, SEGMENT_TYPE_INFO,
    };
    // `Err(_) => break` treated a self-inconsistent segment exactly like
    // `Done`: the remaining records went unexecuted with nothing logged.
    let stream = vec![0u8; SEGMENT_HEADER_LEN + 4];
    // A segment claiming a longer body than the buffer holds, handed straight
    // to the record walker — the shape `iter_segments` would have rejected but
    // that an already-parsed `Segment` can still carry.
    let seg = Segment {
        offset: 0,
        length: (SEGMENT_HEADER_LEN + 64) as u32,
        type_: SEGMENT_TYPE_INFO,
        command_offset: SEGMENT_HEADER_LEN as u32,
        command_length: 64,
        ..Segment::default()
    };
    let before = sink_body().len();
    let mut handled = 0usize;
    walk_segment_records(&stream, &seg, |_, _| handled += 1);
    let added = sink_body()[before..].to_string();
    assert_eq!(handled, 0, "the malformed segment yields no records");
    assert!(
        added.contains("stream_record_fail"),
        "dropping a segment's records must reach the sink, got:\n{added}"
    );
    assert!(
        added.contains("reason=stream_reval_span_oob"),
        "the line must name the failing re-validation check, got:\n{added}"
    );
    assert!(
        added.contains(&format!(
            "seg={}",
            segment_type_name(u32::from(SEGMENT_TYPE_INFO))
        )),
        "the line must say which segment family lost its records, got:\n{added}"
    );
}

#[test]
fn walking_a_well_formed_segment_to_its_end_logs_nothing() {
    use reims_vgpu_protocol::segment::{iter_segments, SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT};
    // The other half of the obligation: `Done` is how every segment ends, so
    // if it produced a line the sink would carry one per segment per frame.
    let mut records = [0u8; 8];
    st32(&mut records[0..4], 0x190);
    st32(&mut records[4..8], 8);
    let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
    st32(
        &mut stream[0..4],
        (SEGMENT_HEADER_LEN + records.len()) as u32,
    );
    stream[4] = SEGMENT_TYPE_EVENT;
    stream.extend_from_slice(&records);

    let segs = iter_segments(&stream).expect("a well-formed stream frames");
    let before = sink_body().len();
    let mut handled = 0usize;
    walk_segment_records(&stream, &segs[0], |_, _| handled += 1);
    let added = sink_body()[before..].to_string();
    assert_eq!(handled, 1, "the one record is handed over");
    assert!(
        !added.contains("stream_record_fail"),
        "end-of-segment is control flow and must stay out of the log, got:\n{added}"
    );
}

#[test]
fn an_unknown_segment_family_is_refused_and_the_type_5_envelope_is_not() {
    use crate::observe::Refusal;
    use reims_vgpu_protocol::segment::{
        segment_disposition, SegmentDisposition, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_PROTECTION_OPTIONS,
    };
    // `walk_stream` ended in `_ => {}`, which gave one silence to two very
    // different things. Ref-texture is a contract-correct skip; function is wire
    // format the host has never seen.
    assert_eq!(
        segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS),
        SegmentDisposition::Envelope
    );
    assert_eq!(
        segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS).refusal(),
        None,
        "the envelope arrives on healthy frames; a line here is a flood"
    );
    assert_eq!(
        segment_disposition(SEGMENT_TYPE_BLIT),
        SegmentDisposition::Walk
    );
    assert_eq!(
        segment_disposition(6).refusal(),
        Some("stream_segment_type_unknown")
    );
    assert_eq!(
        segment_disposition(0xff).refusal(),
        Some("stream_segment_type_unknown")
    );
}

