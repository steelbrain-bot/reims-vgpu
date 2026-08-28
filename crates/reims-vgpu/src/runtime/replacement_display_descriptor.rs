//! The display descriptor page this device publishes for one shared-state
//! registration.
//!
//! The guest's display pipe builds its attributes out of this page while it is
//! bringing the pipe up: the timing elements it picks a mode from, the serial
//! and product name it puts in the display's identity, the panel size its
//! synthesised EDID carries, and the cursor capability that decides whether it
//! doorbells glyph/show/move at all. It reads the page once per registration,
//! before it acknowledges the online offer.
//!
//! A registration whose descriptor is still zeroes is therefore not a display at
//! the wrong size. It is a display with **one** mode, of one pixel by one pixel:
//! the guest's framebuffer comes up, `WindowServer` opens it, and then has
//! nothing to composite, so it never issues a display transaction and never
//! swaps. Every counter downstream reads healthy while the screen stays on
//! whatever the firmware last drew.
//!
//! # The page is written sparsely, and that is a contract requirement
//!
//! The guest seeds the chromaticity block at `+0x2c … +0x48` with the sRGB
//! primaries and a D65 white *before* asking the host to fill the page, and
//! newer guests pass those straight into the EDID. Writing a zeroed image of the
//! page and writing nothing at all are therefore opposite outcomes for that
//! block — one is an sRGB display, the other is a display whose primaries are
//! black. So this builds a list of the fields this device owns and leaves every
//! byte it does not name exactly as the guest left it.

use crate::model::{
    display_dimension_mm, CURSOR_MAX_DIM, DISPLAY_CURSOR_FEATURE_HW, DISPLAY_DESC_FEATURES,
    DISPLAY_DESC_HEIGHT_MM, DISPLAY_DESC_HEIGHT_MM_F32, DISPLAY_DESC_INDEX,
    DISPLAY_DESC_PRODUCT_NAME, DISPLAY_DESC_SERIAL, DISPLAY_DESC_TIMING_COUNT,
    DISPLAY_DESC_WIDTH_MM, DISPLAY_DESC_WIDTH_MM_F32, DISPLAY_HEIGHT_MM, DISPLAY_MODE1_H,
    DISPLAY_MODE1_W, DISPLAY_MODE2_H, DISPLAY_MODE2_W, DISPLAY_MODE3_H, DISPLAY_MODE3_W,
    DISPLAY_MODE_EFI_H, DISPLAY_MODE_EFI_W, DISPLAY_PRODUCT_NAME, DISPLAY_REFRESH_HZ,
    DISPLAY_SERIAL_NUMBER, DISPLAY_SHARED_CURSOR_FEATURES, DISPLAY_SHARED_CURSOR_MAX_WH,
    DISPLAY_WIDTH_MM,
};
use crate::runtime::decode::fifo::{
    display_refresh_hz_1616, display_timing_entry_offset, encode_display_timing_entry,
    DisplayTimingEntry, DISPLAY_DESC_TIMING_STRIDE,
};

/// The modes this device advertises, native first.
///
/// Element 0 doubles as the preferred/native format, so it is the resolution the
/// guest comes up at; the rest are additional selectable modes. Every one is
/// advertised at [`DISPLAY_REFRESH_HZ`], because the guest paces CoreAnimation to
/// the advertised rate of the mode it latches and a mixed table lets it latch a
/// slower one.
const MODES: &[(u16, u16)] = &[
    (DISPLAY_MODE_EFI_W, DISPLAY_MODE_EFI_H),
    (DISPLAY_MODE1_W, DISPLAY_MODE1_H),
    (DISPLAY_MODE2_W, DISPLAY_MODE2_H),
    (DISPLAY_MODE3_W, DISPLAY_MODE3_H),
];

/// The longest field this device writes into the page.
const MAX_FIELD_LEN: usize = DISPLAY_DESC_TIMING_STRIDE as usize;

const _: () = assert!(
    DISPLAY_PRODUCT_NAME.len() <= MAX_FIELD_LEN,
    "the product name must fit one descriptor field"
);

/// One field of the descriptor page: an offset from the page base and the bytes
/// this device owns there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplacementDisplayDescriptorField {
    offset: u64,
    bytes: [u8; MAX_FIELD_LEN],
    length: usize,
}

impl ReplacementDisplayDescriptorField {
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }

    fn new(offset: u64, source: &[u8]) -> Self {
        let mut bytes = [0; MAX_FIELD_LEN];
        bytes[..source.len()].copy_from_slice(source);
        Self {
            offset,
            bytes,
            length: source.len(),
        }
    }

    fn u16(offset: u64, value: u16) -> Self {
        Self::new(offset, &value.to_le_bytes())
    }

    fn u32(offset: u64, value: u32) -> Self {
        Self::new(offset, &value.to_le_bytes())
    }
}

/// Why a descriptor could not be built for this registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplacementDisplayDescriptorError {
    /// The advertised refresh does not fit the wire's 16.16 fixed-point field.
    RefreshUnrepresentable { advertised_hz: u32 },
    /// A timing element would land past the end of the shared page.
    TimingElementPastPage { index: u32, page_bytes: u64 },
    /// A timing element refused to encode into its own stride.
    TimingElementUnencodable { index: u32 },
}

/// Build every field this device owns in the descriptor page, in the order the
/// reference host writes them: identity, panel, cursor capability, then the
/// timing count and the timing elements it counts.
///
/// `page_bytes` bounds the timing array, so a page too small to carry an element
/// refuses by name rather than writing past it.
pub fn replacement_display_descriptor(
    display_index: u32,
    page_bytes: u64,
) -> Result<Vec<ReplacementDisplayDescriptorField>, ReplacementDisplayDescriptorError> {
    let refresh = display_refresh_hz_1616(DISPLAY_REFRESH_HZ).ok_or(
        ReplacementDisplayDescriptorError::RefreshUnrepresentable {
            advertised_hz: DISPLAY_REFRESH_HZ,
        },
    )?;
    // Both encodings of each physical dimension, from one value, as the
    // reference host does: the integer pair is what a stock guest's EDID takes
    // its centimetre fields from, the float pair is what a guest at a higher
    // protocol rung reads, and leaving either unwritten is a zero-size panel to
    // whichever guest reads that one.
    let (width_f32, width_mm) = display_dimension_mm(DISPLAY_WIDTH_MM);
    let (height_f32, height_mm) = display_dimension_mm(DISPLAY_HEIGHT_MM);
    let cursor_max_wh = (CURSOR_MAX_DIM & 0xffff) | ((CURSOR_MAX_DIM & 0xffff) << 16);

    let mut fields = vec![
        ReplacementDisplayDescriptorField::u32(DISPLAY_DESC_SERIAL, DISPLAY_SERIAL_NUMBER),
        ReplacementDisplayDescriptorField::new(DISPLAY_DESC_PRODUCT_NAME, DISPLAY_PRODUCT_NAME),
        ReplacementDisplayDescriptorField::u16(DISPLAY_DESC_INDEX, display_index as u16),
        ReplacementDisplayDescriptorField::u16(DISPLAY_DESC_WIDTH_MM, width_mm),
        ReplacementDisplayDescriptorField::u16(DISPLAY_DESC_HEIGHT_MM, height_mm),
        ReplacementDisplayDescriptorField::u32(DISPLAY_DESC_WIDTH_MM_F32, width_f32.to_bits()),
        ReplacementDisplayDescriptorField::u32(DISPLAY_DESC_HEIGHT_MM_F32, height_f32.to_bits()),
        ReplacementDisplayDescriptorField::u32(DISPLAY_DESC_FEATURES, 0),
        ReplacementDisplayDescriptorField::u32(DISPLAY_SHARED_CURSOR_MAX_WH, cursor_max_wh),
        ReplacementDisplayDescriptorField::u32(
            DISPLAY_SHARED_CURSOR_FEATURES,
            DISPLAY_CURSOR_FEATURE_HW,
        ),
        ReplacementDisplayDescriptorField::u16(DISPLAY_DESC_TIMING_COUNT, MODES.len() as u16),
    ];

    for (index, &(width, height)) in MODES.iter().enumerate() {
        let index = index as u32;
        let offset = display_timing_entry_offset(index, page_bytes).ok_or(
            ReplacementDisplayDescriptorError::TimingElementPastPage { index, page_bytes },
        )?;
        let mut encoded = [0; DISPLAY_DESC_TIMING_STRIDE as usize];
        let entry = DisplayTimingEntry {
            width,
            height,
            refresh_1616: refresh,
            tail0: 0,
            tail1: 0,
        };
        if !encode_display_timing_entry(&entry, &mut encoded) {
            return Err(ReplacementDisplayDescriptorError::TimingElementUnencodable { index });
        }
        fields.push(ReplacementDisplayDescriptorField::new(offset, &encoded));
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_BYTES: u64 = 4096;

    #[test]
    fn the_descriptor_advertises_every_mode_it_counts() {
        let fields = replacement_display_descriptor(0, PAGE_BYTES).expect("descriptor");
        let count = fields
            .iter()
            .find(|field| field.offset() == DISPLAY_DESC_TIMING_COUNT)
            .expect("timing count");
        assert_eq!(count.bytes(), &(MODES.len() as u16).to_le_bytes());
        for index in 0..MODES.len() as u32 {
            let offset = display_timing_entry_offset(index, PAGE_BYTES).expect("offset");
            assert!(
                fields.iter().any(|field| field.offset() == offset),
                "timing element {index} is counted but not written"
            );
        }
    }

    #[test]
    fn the_first_timing_element_is_the_native_mode_the_guest_comes_up_at() {
        let fields = replacement_display_descriptor(0, PAGE_BYTES).expect("descriptor");
        let offset = display_timing_entry_offset(0, PAGE_BYTES).expect("offset");
        let first = fields
            .iter()
            .find(|field| field.offset() == offset)
            .expect("element 0");
        assert_eq!(
            &first.bytes()[..4],
            &[
                DISPLAY_MODE_EFI_W.to_le_bytes(),
                DISPLAY_MODE_EFI_H.to_le_bytes()
            ]
            .concat()[..]
        );
    }

    #[test]
    fn every_mode_is_advertised_at_the_rate_the_refresh_cadence_derives_from() {
        let fields = replacement_display_descriptor(0, PAGE_BYTES).expect("descriptor");
        let expected = display_refresh_hz_1616(DISPLAY_REFRESH_HZ).expect("refresh");
        for index in 0..MODES.len() as u32 {
            let offset = display_timing_entry_offset(index, PAGE_BYTES).expect("offset");
            let element = fields
                .iter()
                .find(|field| field.offset() == offset)
                .expect("element");
            let refresh = u32::from_le_bytes(element.bytes()[4..8].try_into().expect("refresh"));
            assert_eq!(refresh, expected, "element {index}");
        }
    }

    #[test]
    fn the_descriptor_leaves_the_chromaticity_block_the_guest_seeded_alone() {
        let fields = replacement_display_descriptor(0, PAGE_BYTES).expect("descriptor");
        for field in &fields {
            let start = field.offset();
            let end = start + field.bytes().len() as u64;
            assert!(
                end <= 0x2c || start >= 0x48,
                "field at {start:#x} overlaps the guest's own chromaticities"
            );
        }
    }

    #[test]
    fn a_page_too_small_for_a_timing_element_refuses_by_name() {
        assert_eq!(
            replacement_display_descriptor(0, 0x220),
            Err(ReplacementDisplayDescriptorError::TimingElementPastPage {
                index: 1,
                page_bytes: 0x220,
            })
        );
    }
}
