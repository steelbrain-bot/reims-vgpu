//! The walk's typed statuses on this device's failure channel.
//!
//! **This module is the impl below and nothing else.** It used to re-export the
//! whole of `reims_vgpu_paging::resolve` under device-local names, so a reader
//! at a call site could not tell whether `translate_root` was ours or the
//! crate's, and two of those names were renamed on the way through
//! (`ARM64E` arrived as `ARM64E_GEOMETRY`), which is a second vocabulary for one
//! set of items. Callers now name `reims_vgpu_paging` directly and the boundary
//! is visible in every `use` line that crosses it.
//!
//! What genuinely cannot move is here: the mapping from a paging status to the
//! device failure vocabulary.
//!
//! The guest-memory seam is the wire crate's
//! [`GuestMemory`](reims_vgpu_wire::mem::GuestMemory); the device implements
//! it over [`crate::runtime::host::HostMemory`] at each caller.

use reims_vgpu_paging::resolve::ResolveStatus;

/// Give every distinct guest page-table walk check its own failure slug.
///
/// The paging crate owns the status vocabulary. This device-local adapter owns
/// how those statuses appear in its failure stream, without attaching an
/// observation trait to a foreign type.
pub fn refusal(status: ResolveStatus) -> Option<&'static str> {
    Some(match status {
        ResolveStatus::Ok => return None,
        ResolveStatus::ErrArgs => "gva_args",
        ResolveStatus::ErrInactiveTask => "gva_inactive_task",
        ResolveStatus::ErrNoDirectory => "gva_no_directory",
        ResolveStatus::ErrDirectoryRead => "gva_directory_read",
        ResolveStatus::ErrZeroRootPfn => "gva_zero_root_pfn",
        ResolveStatus::ErrZeroDepth => "gva_zero_depth",
        ResolveStatus::ErrDepthTooDeep => "gva_depth_too_deep",
        ResolveStatus::ErrPageTableRead => "gva_page_table_read",
        ResolveStatus::ErrZeroPfn => "gva_zero_pfn",
        ResolveStatus::ErrMalformedPte => "gva_malformed_pte",
        ResolveStatus::ErrUnsupportedGeometry => "gva_unsupported_geometry",
    })
}
