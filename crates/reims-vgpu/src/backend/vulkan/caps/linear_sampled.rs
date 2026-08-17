//! Whether this device can sample guest pages **as an image**, without copying
//! their texels into a second allocation.
//!
//! # Why the question exists
//!
//! [`super::host_pointer`] brings guest RAM to the GPU as a `VkBuffer`, and
//! buffers-only is deliberate there: an optimally-tiled image backed by linear
//! guest bytes is not something this device may assume works on an unknown
//! driver. So guest texels reach a sampled image through
//! `vkCmdCopyBufferToImage`, and the sampled cache in `pools` exists to memoize
//! the result of that copy.
//!
//! The guest contract describes a persistent texture view over a buffer with an
//! explicit byte offset and row pitch. Metal represents that directly as a
//! buffer-backed texture. An ordinary Vulkan `LINEAR` image instead chooses its
//! own row pitch, so the direct representation exists only when that pitch
//! equals the decoded one exactly.
//!
//! # Admission is exact
//!
//! The image requires `SAMPLED_IMAGE`, `SAMPLED_IMAGE_FILTER_LINEAR`, and an
//! importable `HOST_ALLOCATION_EXT` external image without a dedicated-only
//! requirement. The query carries the same `ALIAS` image flag as creation.
//! Image pitch, memory type, bind alignment, subresource offset, and the
//! complete requirements range are checked after creation. A failed condition
//! is a named decline to the copied path.
//!
//! # Report once, query exact formats at admission
//!
//! [`report`] records the representative device answer once at creation.
//! [`format_verdict`] is the executable half: the direct-image registry asks it
//! for the exact decoded format. The created image's pitch, subresource offset,
//! memory-type bits, alignment, and bounds still have to agree with the retained
//! guest allocation; those per-resource checks live together at the bind site.
//!
//! A discrete GPU may fetch these texels across its host-memory link instead of
//! from device-local memory.  That affects measured performance, not API
//! correctness, and is therefore not a reason to replace persistent resource
//! semantics with per-draw copied content.  Hosts that cannot express the
//! alias, and individual layouts whose pitches disagree, retain the explicit
//! buffer-to-image copy fallback.

use ash::vk;

/// Formats the sampled rail builds images in, and therefore the ones worth
/// asking about.
///
/// Taken from `translate::pixel`'s mapping of the Metal pixel formats this
/// device decodes — the 8-bit RGBA/BGRA pairs are what a macOS guest's
/// compositing actually uses. Representative rather than exhaustive: a format
/// absent here is unmeasured, not unsupported, and the line names what it asked
/// about so a reader cannot mistake one for the other.
const PROBED: &[(&str, vk::Format)] = &[
    ("bgra8_unorm", vk::Format::B8G8R8A8_UNORM),
    ("bgra8_srgb", vk::Format::B8G8R8A8_SRGB),
    ("rgba8_unorm", vk::Format::R8G8B8A8_UNORM),
    ("rgba8_srgb", vk::Format::R8G8B8A8_SRGB),
];

/// What one format can do under `VK_IMAGE_TILING_LINEAR`.
///
/// Three bits rather than one because they fail differently. A format that is
/// `SAMPLED_IMAGE` but not `SAMPLED_IMAGE_FILTER_LINEAR` can be sampled only
/// with nearest filtering, which is a *different picture* rather than a slower
/// one — so a rail built on the first without the second would land wrong pixels
/// wherever the guest asked for a linear filter. And a device that samples the
/// format linearly but will not import a host pointer as an image cannot reach
/// the guest's bytes at all, however good the first two look.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearFormatVerdict {
    /// `linearTilingFeatures` contains `SAMPLED_IMAGE`.
    pub sampled: bool,
    /// `linearTilingFeatures` contains `SAMPLED_IMAGE_FILTER_LINEAR`.
    pub filter_linear: bool,
    /// The device reports `HOST_ALLOCATION_EXT` importable for a `LINEAR`,
    /// `SAMPLED` image of this format.
    pub importable: bool,
    /// The external image must own a dedicated allocation.  The direct rail
    /// aliases the allocation's existing whole-span buffer, so this flag makes
    /// that representation unavailable even though a separate import would be
    /// possible.
    pub dedicated_only: bool,
}

impl LinearFormatVerdict {
    /// Read the tiling features. `importable` is answered separately, by a query
    /// that can fail as a whole, so it is set by the caller.
    fn from_tiling(features: vk::FormatFeatureFlags) -> Self {
        Self {
            sampled: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
            filter_linear: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
            importable: false,
            dedicated_only: false,
        }
    }

    /// Whether the *device-level* conditions all hold. Never sufficient on its
    /// own: the row-pitch agreement in this module's doc is a per-window
    /// question and is not represented here.
    pub fn device_conditions_hold(self) -> bool {
        self.sampled && self.filter_linear && self.importable && !self.dedicated_only
    }

    /// Stable slug for the report line, naming which condition refused.
    fn slug(self) -> &'static str {
        match (
            self.sampled,
            self.filter_linear,
            self.importable,
            self.dedicated_only,
        ) {
            (true, true, true, false) => "alias_possible",
            (true, true, true, true) => "dedicated_only",
            (true, true, false, _) => "not_importable",
            (true, false, _, _) => "sampled_nearest_only",
            (false, _, _, _) => "not_sampled",
        }
    }
}

/// Query the device-level half of direct linear-image admission for one exact
/// format.  Extent/pitch/memory-requirement agreement remains a created-image
/// question and is deliberately absent from this answer.
///
/// # Safety
///
/// `pd` must belong to `instance`.
pub(crate) unsafe fn format_verdict(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    format: vk::Format,
) -> LinearFormatVerdict {
    let props = unsafe { instance.get_physical_device_format_properties(pd, format) };
    let mut verdict = LinearFormatVerdict::from_tiling(props.linear_tiling_features);
    let mut ext_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .flags(vk::ImageCreateFlags::ALIAS)
        .push_next(&mut ext_info);
    let mut ext_props = vk::ExternalImageFormatProperties::default();
    let mut out = vk::ImageFormatProperties2::default().push_next(&mut ext_props);
    if unsafe { instance.get_physical_device_image_format_properties2(pd, &info, &mut out) }.is_ok()
    {
        let features = ext_props
            .external_memory_properties
            .external_memory_features;
        verdict.importable = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
        verdict.dedicated_only = features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
    }
    verdict
}

/// Ask the device about every format in [`PROBED`] and emit the answer.
///
/// Emitted on the `OFF` channel at device create, beside `vk_caps`, because it
/// is a fact about the host and not a loss. One line per boot.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn report(instance: &ash::Instance, pd: vk::PhysicalDevice) {
    let mut fields = String::new();
    let mut possible = 0usize;
    for (name, format) in PROBED {
        let verdict = unsafe { format_verdict(instance, pd, *format) };

        if verdict.device_conditions_hold() {
            possible += 1;
        }
        fields.push_str(&format!(" {name}={}", verdict.slug()));
    }
    crate::observe::off(format!(
        "vk_linear_sampled alias_possible={possible}/{}{fields} (device conditions only — a window \
         also needs exact created-image pitch and memory requirements)",
        PROBED.len()
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every condition is load-bearing, and each one missing is its own slug. A
    /// format that samples but cannot filter linearly would land nearest-filtered
    /// pixels wherever the guest asked for a linear filter — a wrong picture, not
    /// a slow one — so no single bit may satisfy the verdict alone.
    #[test]
    fn every_device_condition_is_required() {
        let all = LinearFormatVerdict {
            sampled: true,
            filter_linear: true,
            importable: true,
            dedicated_only: false,
        };
        assert!(all.device_conditions_hold());
        assert_eq!(all.slug(), "alias_possible");

        for (drop_field, want_slug) in [
            ("filter", "sampled_nearest_only"),
            ("import", "not_importable"),
            ("sampled", "not_sampled"),
        ] {
            let mut v = all;
            match drop_field {
                "filter" => v.filter_linear = false,
                "import" => v.importable = false,
                _ => v.sampled = false,
            }
            assert!(!v.device_conditions_hold(), "{drop_field} must be required");
            assert_eq!(v.slug(), want_slug, "{drop_field}");
        }
    }

    /// The tiling flags are read from `linearTilingFeatures`, not invented, and a
    /// neighbouring bit must not be mistaken for either of ours.
    #[test]
    fn the_verdict_reads_the_two_tiling_flags_it_names() {
        let none = LinearFormatVerdict::from_tiling(vk::FormatFeatureFlags::empty());
        assert!(!none.sampled && !none.filter_linear);

        let sampled = LinearFormatVerdict::from_tiling(vk::FormatFeatureFlags::SAMPLED_IMAGE);
        assert!(sampled.sampled && !sampled.filter_linear);

        let both = LinearFormatVerdict::from_tiling(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        );
        assert!(both.sampled && both.filter_linear);

        let storage = LinearFormatVerdict::from_tiling(vk::FormatFeatureFlags::STORAGE_IMAGE);
        assert!(!storage.sampled && !storage.filter_linear);
    }

    /// `from_tiling` never claims importability. That answer comes from a
    /// separate query which can fail as a whole, and a verdict that defaulted it
    /// to `true` would report an alias as possible on a device that was never
    /// asked.
    #[test]
    fn tiling_alone_never_claims_importable() {
        let both = LinearFormatVerdict::from_tiling(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        );
        assert!(!both.importable);
        assert!(!both.device_conditions_hold());
    }

    /// Every probed format is named and distinct. A duplicate would report one
    /// device answer twice and inflate the `alias_possible=` numerator.
    #[test]
    fn every_probed_format_is_named_once() {
        let mut names: Vec<_> = PROBED.iter().map(|(n, _)| *n).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two probed formats share a name");

        let mut formats: Vec<_> = PROBED.iter().map(|(_, f)| *f).collect();
        formats.sort_unstable_by_key(|f| f.as_raw());
        formats.dedup();
        assert_eq!(formats.len(), count, "one format is probed twice");
    }
}
