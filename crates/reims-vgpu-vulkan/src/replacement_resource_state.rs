//! Native projection of one prepared resource-validity operation.
//!
//! The lifecycle token supplies both exact completion facts and any
//! [`TransferKey`] values. This module does not decide whether a transfer is
//! needed and cannot widen its canonical region. Buffer-backed regions project
//! directly to Vulkan copies. A linear texture endpoint is identified only by
//! its retained layout and exact buffer/image representation pair; its byte
//! region is inverted through that layout before image commands are recorded.

use crate::replacement_buffer_blit::{
    NativeBufferBlit, NativeBufferTarget, ReplacementBufferResolver,
};
use crate::replacement_representation::ReplacementBufferAllocationError;
use crate::{
    replacement_image_blit::{NativeBufferImageCopy, NativeImageBlitCommand, NativeImageCopy},
    replacement_image_state::{PreparedImageState, PreparedImageStateBatch, ReplacementImageKey},
    replacement_image_transition::{resolve_image_transitions, PreparedNativeImageState},
};
use ash::vk;
use reims_vgpu_core::{
    BackingRegion, HostLandingKey, PreparedContentSynchronizationBatch, PreparedResourceState,
    PreparedResourceStateBatch, ResolvedResourceCompletion, ResolvedResourceState, TransferKey,
    HOST_REPRESENTATION,
};
use reims_vgpu_protocol::{BackingId, TransactionId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceStateTransferRecordError {
    TransferBackingNotDeclared(BackingId),
    UnknownSource(TransferKey),
    UnknownDestination(TransferKey),
    MissingTransferSource(TransferKey),
    MissingTransferDestination(TransferKey),
    RangeOutOfBounds(TransferKey),
    LinearRangeOutOfBounds {
        transfer: TransferKey,
        source_size: u64,
        destination_size: u64,
    },
    RangeAddressOverflow(TransferKey),
    WholeBufferSizeMismatch(TransferKey),
    SameBufferOverlap(TransferKey),
    ImageTransferRequiresState(TransferKey),
    UnexpectedImageState,
    ImageStateMismatch(TransferKey),
    ImageTransition(crate::replacement_image_transition::ImageTransitionResolveError),
    ImageEndpointMissing(TransferKey),
    ImageToImageUnsupported {
        transfer: TransferKey,
        refusal: ImageToImageRefusal,
    },
    ImageLayoutMissing(TransferKey),
    ImageLayoutUnsupported {
        transfer: TransferKey,
        refusal: ImageLayoutRefusal,
    },
    ImageUsageMissing(TransferKey),
    MissingHostStaging(HostLandingKey),
    HostLandingImageLayoutMissing(HostLandingKey),
    HostLandingRangeOutOfBounds(HostLandingKey),
}

impl ResourceStateTransferRecordError {
    /// Whether re-offering this transfer could ever produce a different answer.
    ///
    /// Only the shapes this device does not implement are terminal. Both name
    /// a property of the transfer itself -- two image endpoints, or a linear
    /// layout the buffer/image copy cannot express -- so no later packet
    /// supplies anything that changes them, and retrying holds a submission
    /// head forever. Everything else here is either a device fault or a state
    /// a later packet supplies, and a wrong `true` throws away guest work that
    /// would have recorded.
    pub const fn is_terminal_refusal(&self) -> bool {
        matches!(
            self,
            Self::ImageToImageUnsupported { .. } | Self::ImageLayoutUnsupported { .. }
        )
    }
}

/// Which term of a linear texture layout a buffer/image copy could not express.
///
/// One name for thirty-one refusal sites was one name too few. Every one of
/// them says "this layout is unsupported", they are spread over three
/// functions that walk the same descriptor, and a boot that hits one reports a
/// transfer key and nothing about which term of the layout stopped it -- so the
/// next question, what to implement, has nowhere to start. This is the same
/// move [`ImageEndpointDisagreement`] makes for the image-to-image pair, for
/// the same reason.
///
/// Deciding and reporting stay one walk: these are constructed exactly where
/// the refusal is decided, never reconstructed afterwards from the layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageLayoutRefusal {
    /// The layout declares no level at this mip.
    LevelAbsent { mip: u32 },
    /// A compressed layout. This device does not translate block coordinates.
    CompressedLayout,
    /// The layout declares a zero element size, so no byte offset divides into
    /// texels.
    ZeroBytesPerElement,
    /// A row stride that is not a whole number of elements.
    /// `vkCmdCopyBufferToImage` takes `bufferRowLength` in texels and there is
    /// no remainder to give it.
    RowStrideNotElementAligned {
        row_stride: u64,
        bytes_per_element: u64,
    },
    /// A row length in elements that does not fit the Vulkan field.
    RowLengthNotRepresentable { row_length: u64 },
    /// The layout declares no offset for this subresource.
    SubresourceOffsetAbsent { layer: u32, mip: u32 },
    /// The region does not reduce to one exact Vulkan subresource range.
    SubresourceRangeInexact,
    /// A texel coordinate or extent that does not fit its Vulkan field.
    CoordinateNotRepresentable {
        term: LayoutCoordinateTerm,
        value: u64,
    },
    /// The layout declares no pixel format, so nothing says how wide a texel
    /// is.
    PixelFormatUndeclared,
    /// A depth or stencil format reached through a byte range. A linear range
    /// names bytes and a copy needs an aspect, and this device does not pick
    /// one for the guest.
    DepthStencilFormat { pixel_format: u16 },
    /// The layout declares no physical slice count.
    SliceCountUndeclared,
    /// A zero row or image stride, which no byte offset divides by.
    ZeroStride { row_stride: u64, image_stride: u64 },
    /// The byte range does not begin on an element boundary.
    OffsetNotElementAligned { offset: u64, bytes_per_element: u64 },
    /// The byte range is not a whole number of elements.
    LengthNotElementAligned { length: u64, bytes_per_element: u64 },
    /// The derived origin and extent are not a texel box.
    ExtentNotATexelBox { origin: [u32; 3], extent: [u32; 3] },
    /// A transfer region that is neither an image region nor a byte range.
    RegionNotLinear,
}

/// Which coordinate a layout translation could not fit into its Vulkan field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutCoordinateTerm {
    OriginX,
    OriginY,
    OriginZ,
    Width,
    Height,
    Depth,
}

/// One of the five terms `same_subresource_range` compares.
///
/// `vk::ImageSubresourceRange` implements neither `PartialEq` nor `Eq`, which
/// is why that helper exists at all; these name the comparisons it makes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubresourceRangeTerm {
    AspectMask,
    BaseMipLevel,
    LevelCount,
    BaseArrayLayer,
    LayerCount,
}

/// Which term of the agreement two image endpoints failed, and both values.
///
/// Two textures over one allocation are two native images this device believes
/// carry identical content, so a copy between them is expressible only while
/// all four of these agree. A format pair and an extent pair fail at the same
/// call site and are different defects -- one is a planner that named siblings
/// the guest never aliased, the other is a geometry this device derived
/// wrongly -- so the term is what has to be recorded, not the fact of a
/// disagreement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageEndpointDisagreement {
    PixelFormat {
        source: u16,
        destination: u16,
    },
    Extent {
        source: [u32; 3],
        destination: [u32; 3],
    },
    Samples {
        source: vk::SampleCountFlags,
        destination: vk::SampleCountFlags,
    },
    /// The ranges disagree, on this term. Naming the term rather than the two
    /// ranges is the same move this whole refusal makes one level up: a level
    /// count and an aspect mask fail the same comparison and mean different
    /// things.
    SubresourceRange {
        term: SubresourceRangeTerm,
        source: u32,
        destination: u32,
    },
}

/// What a round trip through an allocation's bytes was missing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AliasRoundTripTerm {
    /// The region names texels. Those are one endpoint's coordinates and the
    /// disagreement is precisely that they do not carry over to the other's;
    /// only a byte-denominated region names the same thing to both.
    RegionNamesTexels,
    /// The source declares no layout of its own, so where its texels sit in
    /// the allocation's bytes is not established.
    SourceLayout,
    /// The destination declares no layout of its own.
    DestinationLayout,
    /// The destination image has no byte buffer to round trip through, so the
    /// two were not recorded as aliases of one allocation when they were
    /// declared.
    TransferBytes,
}

/// The one place that decides whether two image endpoints agree, and says
/// which term did not.
///
/// Deciding and reporting are the same walk: a second spelling of the terms
/// somewhere else is the next divergence.
fn image_endpoints_disagree(
    source: &crate::replacement_image_transition::NativeImageTarget,
    destination: &crate::replacement_image_transition::NativeImageTarget,
) -> Option<ImageEndpointDisagreement> {
    if source.pixel_format != destination.pixel_format {
        return Some(ImageEndpointDisagreement::PixelFormat {
            source: source.pixel_format,
            destination: destination.pixel_format,
        });
    }
    let extent = |extent: vk::Extent3D| [extent.width, extent.height, extent.depth];
    if source.extent != destination.extent {
        return Some(ImageEndpointDisagreement::Extent {
            source: extent(source.extent),
            destination: extent(destination.extent),
        });
    }
    if source.samples != destination.samples {
        return Some(ImageEndpointDisagreement::Samples {
            source: source.samples,
            destination: destination.samples,
        });
    }
    // The same five comparisons `same_subresource_range` makes, in its order,
    // reported rather than reduced to a bool.
    for (term, left, right) in [
        (
            SubresourceRangeTerm::AspectMask,
            source.full_range.aspect_mask.as_raw(),
            destination.full_range.aspect_mask.as_raw(),
        ),
        (
            SubresourceRangeTerm::BaseMipLevel,
            source.full_range.base_mip_level,
            destination.full_range.base_mip_level,
        ),
        (
            SubresourceRangeTerm::LevelCount,
            source.full_range.level_count,
            destination.full_range.level_count,
        ),
        (
            SubresourceRangeTerm::BaseArrayLayer,
            source.full_range.base_array_layer,
            destination.full_range.base_array_layer,
        ),
        (
            SubresourceRangeTerm::LayerCount,
            source.full_range.layer_count,
            destination.full_range.layer_count,
        ),
    ] {
        if left != right {
            return Some(ImageEndpointDisagreement::SubresourceRange {
                term,
                source: left,
                destination: right,
            });
        }
    }
    None
}

/// Why an image-to-image transfer is not a copy this device can record.
///
/// Four conditions reach this refusal and they call for different repairs, so
/// the family name alone cannot be acted on: a boot reporting it says only
/// that a content repair between two siblings stopped, and the submission head
/// it holds stays held whichever one fired.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageToImageRefusal {
    /// The endpoints disagree on one of the terms a copy between them needs.
    /// `vkCmdCopyImage` between mismatched formats or extents is not a
    /// narrower copy, it is undefined.
    EndpointsDisagree(ImageEndpointDisagreement),
    /// Both representations resolved to one native image, so the copy would
    /// read and write the same memory. The authority planned a transfer whose
    /// endpoints are not two places.
    SameImage,
    /// A byte range names no image geometry of its own. That is the
    /// buffer/image pair's shape, and it is recorded there.
    LinearRegion,
    /// The endpoints disagree on a term the allocation's bytes could have
    /// bridged, and a part of that round trip is not established.
    /// `disagreement` names why a direct copy was unavailable and `missing`
    /// names what the round trip lacked, because the two are separate repairs.
    AliasBytesUnavailable {
        disagreement: ImageEndpointDisagreement,
        missing: AliasRoundTripTerm,
    },
    /// The source declares no aspect `vkCmdCopyImage` can name, so the region
    /// walk produced no command. An empty batch is not a completed transfer:
    /// the destination would stay stale with nothing saying so.
    NoCopyableAspect { aspect_mask: vk::ImageAspectFlags },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeResourceStateTransfer {
    Buffer(NativeBufferBlit),
    /// One semantic backing transfer may cross several mip or layer regions.
    /// The batch remains one completion owner even though Vulkan records one
    /// buffer/image copy command per representable subresource intersection.
    Image(Box<[NativeImageBlitCommand]>),
}

#[derive(Clone, Debug)]
pub struct ReplacementResourceStateProgram {
    transaction: TransactionId,
    index: usize,
    operation: ResolvedResourceState,
    backings: Box<[BackingId]>,
    transfers: Box<[TransferKey]>,
    completions: Box<[ResolvedResourceCompletion]>,
    native_transfers: Box<[NativeResourceStateTransfer]>,
    image_state: Option<PreparedNativeImageState>,
    host_landings: Box<[ReplacementHostLandingProgram]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplacementHostLandingProgram {
    landing: HostLandingKey,
    staging: crate::replacement_representation::ReplacementHostStagingBuffer,
    linear_texture: Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>>,
}

#[derive(Debug)]
pub struct PreparedReplacementHostLanding {
    pub landing: HostLandingKey,
    guest: reims_vgpu_memory::GuestWindow,
    stores: Vec<PreparedGuestStore>,
}

#[derive(Debug)]
struct PreparedGuestStore {
    offset: u64,
    bytes: Vec<u8>,
}

impl PreparedReplacementHostLanding {
    pub fn validate_store(&self) -> Result<(), reims_vgpu_memory::GuestStoreError> {
        for store in &self.stores {
            self.guest
                .validate_store_range(store.offset, store.bytes.len())?;
        }
        Ok(())
    }

    pub fn store(self) -> Result<HostLandingKey, reims_vgpu_memory::GuestStoreError> {
        for store in self.stores {
            self.guest.store_range(store.offset, &store.bytes)?;
        }
        Ok(self.landing)
    }
}

impl ReplacementHostLandingProgram {
    pub const fn landing(&self) -> HostLandingKey {
        self.landing
    }

    pub fn prepare_after_timeline(
        &self,
    ) -> Result<PreparedReplacementHostLanding, ReplacementBufferAllocationError> {
        let staged = self.staging.read_after_timeline()?;
        let stores = match self.landing.region {
            BackingRegion::Whole => contiguous_guest_store(
                &staged,
                0,
                u64::try_from(staged.len())
                    .map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?,
            ),
            BackingRegion::Linear(range) => {
                contiguous_guest_store(&staged, range.start(), range.end())
            }
            BackingRegion::Image(region) => image_guest_stores(
                &staged,
                self.linear_texture
                    .as_deref()
                    .expect("image host landing layout was resolved"),
                region,
            ),
        };
        let stores = stores?;
        Ok(PreparedReplacementHostLanding {
            landing: self.landing,
            guest: self.staging.guest().clone(),
            stores,
        })
    }
}

fn contiguous_guest_store(
    staged: &[u8],
    offset: u64,
    end: u64,
) -> Result<Vec<PreparedGuestStore>, ReplacementBufferAllocationError> {
    let start =
        usize::try_from(offset).map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
    let end = usize::try_from(end).map_err(|_| ReplacementBufferAllocationError::SizeOverflow)?;
    let bytes = staged
        .get(start..end)
        .ok_or(ReplacementBufferAllocationError::SizeOverflow)?
        .to_vec();
    Ok(vec![PreparedGuestStore { offset, bytes }])
}

fn image_guest_stores(
    staged: &[u8],
    descriptor: &reims_vgpu_protocol::LinearTextureDescriptor,
    region: reims_vgpu_core::ImageRegion,
) -> Result<Vec<PreparedGuestStore>, ReplacementBufferAllocationError> {
    let level = descriptor
        .level(region.mip)
        .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
    let bytes_per_element = u64::from(descriptor.bytes_per_element);
    if descriptor.compressed_layout || bytes_per_element == 0 {
        return Err(ReplacementBufferAllocationError::SizeOverflow);
    }
    if region.texels.end[0] > level.width
        || region.texels.end[1] > level.height
        || region.texels.end[2] > level.planes()
    {
        return Err(ReplacementBufferAllocationError::SizeOverflow);
    }
    let level_offset = descriptor
        .subresource_offset(region.layer, region.mip)
        .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
    let row_bytes = u64::from(region.texels.end[0] - region.texels.origin[0])
        .checked_mul(bytes_per_element)
        .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
    let image_stride = level
        .row_stride
        .checked_mul(u64::from(level.height))
        .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
    let mut stores = Vec::new();
    for z in region.texels.origin[2]..region.texels.end[2] {
        for y in region.texels.origin[1]..region.texels.end[1] {
            let offset = level_offset
                .checked_add(
                    u64::from(z)
                        .checked_mul(image_stride)
                        .ok_or(ReplacementBufferAllocationError::SizeOverflow)?,
                )
                .and_then(|offset| offset.checked_add(u64::from(y).checked_mul(level.row_stride)?))
                .and_then(|offset| {
                    offset.checked_add(
                        u64::from(region.texels.origin[0]).checked_mul(bytes_per_element)?,
                    )
                })
                .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
            let end = offset
                .checked_add(row_bytes)
                .ok_or(ReplacementBufferAllocationError::SizeOverflow)?;
            stores.extend(contiguous_guest_store(staged, offset, end)?);
        }
    }
    Ok(stores)
}

fn image_region_fits_layout(
    descriptor: &reims_vgpu_protocol::LinearTextureDescriptor,
    region: reims_vgpu_core::ImageRegion,
    buffer_size: u64,
) -> bool {
    let Some(level) = descriptor.level(region.mip) else {
        return false;
    };
    let bytes_per_element = u64::from(descriptor.bytes_per_element);
    if descriptor.compressed_layout
        || bytes_per_element == 0
        || region.texels.end[0] > level.width
        || region.texels.end[1] > level.height
        || region.texels.end[2] > level.planes()
    {
        return false;
    }
    let Some(level_offset) = descriptor.subresource_offset(region.layer, region.mip) else {
        return false;
    };
    let Some(image_stride) = level.row_stride.checked_mul(u64::from(level.height)) else {
        return false;
    };
    let Some(last_z) = region.texels.end[2].checked_sub(1) else {
        return false;
    };
    let Some(last_y) = region.texels.end[1].checked_sub(1) else {
        return false;
    };
    u64::from(last_z)
        .checked_mul(image_stride)
        .and_then(|z| level_offset.checked_add(z))
        .zip(u64::from(last_y).checked_mul(level.row_stride))
        .and_then(|(offset, y)| offset.checked_add(y))
        .zip(u64::from(region.texels.end[0]).checked_mul(bytes_per_element))
        .and_then(|(offset, x_end)| offset.checked_add(x_end))
        .is_some_and(|end| end <= descriptor.allocation_size && end <= buffer_size)
}

#[derive(Clone, Debug)]
pub struct ReplacementResourceStateBatchProgram {
    transaction: TransactionId,
    programs: Box<[ReplacementResourceStateProgram]>,
}

#[derive(Clone, Debug)]
pub struct ReplacementContentSynchronizationProgram {
    transaction: TransactionId,
    backings: Box<[BackingId]>,
    transfers: Box<[TransferKey]>,
    native_transfers: Box<[NativeResourceStateTransfer]>,
}

impl ReplacementContentSynchronizationProgram {
    pub fn resolve(
        prepared: &PreparedContentSynchronizationBatch,
        image_states: Option<&PreparedImageStateBatch>,
        resolver: &(impl ReplacementBufferResolver
              + crate::replacement_image_transition::ReplacementImageResolver),
    ) -> Result<Self, ResourceStateTransferRecordError> {
        let backings = prepared.backings();
        let transfers: Box<[TransferKey]> = prepared.transfers().into();
        let image_state = image_states.and_then(|states| {
            states
                .operations()
                .iter()
                .find(|state| state.operation_index().is_none())
        });
        let native_transfers = transfers
            .iter()
            .copied()
            .map(|transfer| resolve_transfer(&backings, transfer, image_state, resolver))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        Ok(Self {
            transaction: prepared.transaction(),
            backings,
            transfers,
            native_transfers,
        })
    }

    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    pub const fn transfers(&self) -> &[TransferKey] {
        &self.transfers
    }

    pub const fn native_transfers(&self) -> &[NativeResourceStateTransfer] {
        &self.native_transfers
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceStateBatchResolveError {
    pub index: usize,
    pub reason: ResourceStateTransferRecordError,
}

impl ReplacementResourceStateBatchProgram {
    /// Project the complete lifecycle-owned resource-state sidecar set before
    /// native recording begins. A failing suffix cannot leave a partial native
    /// program reachable by the recorder.
    pub fn resolve(
        prepared: &PreparedResourceStateBatch,
        resolver: &impl ReplacementBufferResolver,
    ) -> Result<Self, ResourceStateBatchResolveError> {
        let programs = prepared
            .states()
            .iter()
            .map(|state| {
                ReplacementResourceStateProgram::resolve(state, resolver).map_err(|reason| {
                    ResourceStateBatchResolveError {
                        index: state.index(),
                        reason,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        Ok(Self {
            transaction: prepared.transaction(),
            programs,
        })
    }

    pub fn resolve_with_image_states(
        prepared: &PreparedResourceStateBatch,
        image_states: Option<&PreparedImageStateBatch>,
        resolver: &(impl ReplacementBufferResolver
              + crate::replacement_image_transition::ReplacementImageResolver),
    ) -> Result<Self, ResourceStateBatchResolveError> {
        let programs = prepared
            .states()
            .iter()
            .map(|state| {
                let image_state = image_states.and_then(|states| {
                    states
                        .operations()
                        .iter()
                        .find(|image| image.operation_index() == Some(state.index()))
                });
                ReplacementResourceStateProgram::resolve_with_image_state(
                    state,
                    image_state,
                    resolver,
                )
                .map_err(|reason| ResourceStateBatchResolveError {
                    index: state.index(),
                    reason,
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        Ok(Self {
            transaction: prepared.transaction(),
            programs,
        })
    }

    pub(crate) const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub(crate) const fn programs(&self) -> &[ReplacementResourceStateProgram] {
        &self.programs
    }
}

impl ReplacementResourceStateProgram {
    pub fn resolve(
        prepared: &PreparedResourceState,
        resolver: &impl ReplacementBufferResolver,
    ) -> Result<Self, ResourceStateTransferRecordError> {
        struct NoImages;
        impl crate::replacement_image_transition::ReplacementImageResolver for NoImages {
            fn resolve_image(
                &self,
                _image: ReplacementImageKey,
            ) -> Option<crate::replacement_image_transition::NativeImageTarget> {
                None
            }
        }
        struct Resolver<'a, T>(&'a T);
        impl<T: ReplacementBufferResolver> ReplacementBufferResolver for Resolver<'_, T> {
            fn resolve_buffer(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<NativeBufferTarget> {
                self.0.resolve_buffer(backing, representation)
            }
            fn resolve_host_staging(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<crate::replacement_representation::ReplacementHostStagingBuffer>
            {
                self.0.resolve_host_staging(backing, representation)
            }
            fn resolve_linear_texture_layout(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                self.0
                    .resolve_linear_texture_layout(backing, representation)
            }
            fn compute_fill_limits(
                &self,
            ) -> Option<crate::replacement_buffer_blit::NativeComputeFillLimits> {
                self.0.compute_fill_limits()
            }
        }
        struct Combined<'a, T> {
            buffers: Resolver<'a, T>,
            images: NoImages,
        }
        impl<T: ReplacementBufferResolver> ReplacementBufferResolver for Combined<'_, T> {
            fn resolve_buffer(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<NativeBufferTarget> {
                self.buffers.resolve_buffer(backing, representation)
            }
            fn resolve_host_staging(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<crate::replacement_representation::ReplacementHostStagingBuffer>
            {
                self.buffers.resolve_host_staging(backing, representation)
            }
            fn resolve_linear_texture_layout(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                self.buffers
                    .resolve_linear_texture_layout(backing, representation)
            }
        }
        impl<T> crate::replacement_image_transition::ReplacementImageResolver for Combined<'_, T> {
            fn resolve_image(
                &self,
                image: ReplacementImageKey,
            ) -> Option<crate::replacement_image_transition::NativeImageTarget> {
                self.images.resolve_image(image)
            }
        }
        Self::resolve_with_image_state(
            prepared,
            None,
            &Combined {
                buffers: Resolver(resolver),
                images: NoImages,
            },
        )
    }

    pub fn resolve_with_image_state(
        prepared: &PreparedResourceState,
        image_state: Option<&PreparedImageState>,
        resolver: &(impl ReplacementBufferResolver
              + crate::replacement_image_transition::ReplacementImageResolver),
    ) -> Result<Self, ResourceStateTransferRecordError> {
        let transfers: Box<[TransferKey]> = prepared.transfers().into();
        let native_image_state = if let Some(state) = image_state {
            if state.transaction() != prepared.transaction()
                || state.operation_index() != Some(prepared.index())
            {
                return Err(ResourceStateTransferRecordError::ImageStateMismatch(
                    transfers
                        .first()
                        .copied()
                        .ok_or(ResourceStateTransferRecordError::UnexpectedImageState)?,
                ));
            }
            Some(
                resolve_image_transitions(state, resolver)
                    .map_err(ResourceStateTransferRecordError::ImageTransition)?,
            )
        } else {
            None
        };
        let resolved_commands = transfers
            .iter()
            .copied()
            .map(|transfer| {
                resolve_transfer(
                    &prepared
                        .operation()
                        .targets
                        .iter()
                        .map(|target| target.backing)
                        .collect::<Vec<_>>(),
                    transfer,
                    image_state,
                    resolver,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let host_landings = prepared
            .host_landings()
            .iter()
            .copied()
            .map(|landing| {
                let staging = resolver
                    .resolve_host_staging(landing.backing, HOST_REPRESENTATION)
                    .ok_or(ResourceStateTransferRecordError::MissingHostStaging(
                        landing,
                    ))?;
                let target = resolver
                    .resolve_buffer(landing.backing, HOST_REPRESENTATION)
                    .ok_or(ResourceStateTransferRecordError::MissingHostStaging(
                        landing,
                    ))?;
                let linear_texture =
                    resolver.resolve_linear_texture_layout(landing.backing, HOST_REPRESENTATION);
                match landing.region {
                    BackingRegion::Whole if target.size != staging.guest().requested() => {
                        return Err(
                            ResourceStateTransferRecordError::HostLandingRangeOutOfBounds(landing),
                        );
                    }
                    BackingRegion::Linear(range)
                        if range.end() > target.size
                            || range.end() > staging.guest().requested() =>
                    {
                        return Err(
                            ResourceStateTransferRecordError::HostLandingRangeOutOfBounds(landing),
                        );
                    }
                    BackingRegion::Image(region) => {
                        let Some(layout) = linear_texture.as_deref() else {
                            return Err(
                                ResourceStateTransferRecordError::HostLandingImageLayoutMissing(
                                    landing,
                                ),
                            );
                        };
                        if !image_region_fits_layout(layout, region, target.size) {
                            return Err(
                                ResourceStateTransferRecordError::HostLandingRangeOutOfBounds(
                                    landing,
                                ),
                            );
                        }
                    }
                    BackingRegion::Whole | BackingRegion::Linear(_) => {}
                }
                Ok(ReplacementHostLandingProgram {
                    landing,
                    staging,
                    linear_texture,
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        Ok(Self {
            transaction: prepared.transaction(),
            index: prepared.index(),
            operation: prepared.operation().clone(),
            backings: prepared.backings(),
            transfers,
            completions: prepared
                .transfers()
                .iter()
                .copied()
                .map(ResolvedResourceCompletion::Transfer)
                .chain(prepared.resource_completions().iter().copied())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            native_transfers: resolved_commands,
            image_state: native_image_state,
            host_landings,
        })
    }

    pub(crate) const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    pub(crate) const fn operation(&self) -> &ResolvedResourceState {
        &self.operation
    }

    pub(crate) const fn transfers(&self) -> &[TransferKey] {
        &self.transfers
    }

    pub(crate) const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }

    pub(crate) const fn native_transfers(&self) -> &[NativeResourceStateTransfer] {
        &self.native_transfers
    }

    pub(crate) const fn image_state(&self) -> Option<&PreparedNativeImageState> {
        self.image_state.as_ref()
    }

    pub(crate) const fn host_landings(&self) -> &[ReplacementHostLandingProgram] {
        &self.host_landings
    }

    pub(crate) const fn backings(&self) -> &[BackingId] {
        &self.backings
    }
}

fn resolve_buffer_transfer(
    backings: &[BackingId],
    transfer: TransferKey,
    resolver: &impl ReplacementBufferResolver,
) -> Result<NativeBufferBlit, ResourceStateTransferRecordError> {
    if !backings.contains(&transfer.backing) {
        return Err(ResourceStateTransferRecordError::TransferBackingNotDeclared(transfer.backing));
    }
    let source = resolver
        .resolve_buffer(transfer.backing, transfer.source)
        .ok_or(ResourceStateTransferRecordError::UnknownSource(transfer))?;
    let destination = resolver
        .resolve_buffer(transfer.backing, transfer.destination)
        .ok_or(ResourceStateTransferRecordError::UnknownDestination(
            transfer,
        ))?;
    if !source.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC) {
        return Err(ResourceStateTransferRecordError::MissingTransferSource(
            transfer,
        ));
    }
    if !destination
        .usage
        .contains(vk::BufferUsageFlags::TRANSFER_DST)
    {
        return Err(ResourceStateTransferRecordError::MissingTransferDestination(transfer));
    }
    let (source_offset, destination_offset, size) = match transfer.region {
        BackingRegion::Whole => {
            if source.size != destination.size {
                return Err(ResourceStateTransferRecordError::WholeBufferSizeMismatch(
                    transfer,
                ));
            }
            (source.base_offset, destination.base_offset, source.size)
        }
        BackingRegion::Linear(range) => {
            let end = range.end();
            if end > source.size || end > destination.size {
                return Err(ResourceStateTransferRecordError::LinearRangeOutOfBounds {
                    transfer,
                    source_size: source.size,
                    destination_size: destination.size,
                });
            }
            let source_offset = source.base_offset.checked_add(range.start()).ok_or(
                ResourceStateTransferRecordError::RangeAddressOverflow(transfer),
            )?;
            let destination_offset = destination.base_offset.checked_add(range.start()).ok_or(
                ResourceStateTransferRecordError::RangeAddressOverflow(transfer),
            )?;
            (source_offset, destination_offset, end - range.start())
        }
        BackingRegion::Image(_) => {
            return Err(ResourceStateTransferRecordError::ImageTransferRequiresState(transfer));
        }
    };
    validate_native_end(source, source_offset, size, transfer)?;
    validate_native_end(destination, destination_offset, size, transfer)?;
    if source.buffer == destination.buffer
        && ranges_overlap(source_offset, size, destination_offset, size)
    {
        return Err(ResourceStateTransferRecordError::SameBufferOverlap(
            transfer,
        ));
    }
    Ok(NativeBufferBlit::Copy {
        source: source.buffer,
        destination: destination.buffer,
        source_offset,
        destination_offset,
        size,
    })
}

/// The endpoint shapes a transfer between two representations over one backing
/// can have, decided once from the native registry.
///
/// Recording a transfer and deciding whether the EXEC needs a prepared
/// image-state batch are the same question asked at two moments, so they ask
/// it through this one classification. A second spelling of it drifts: a
/// classifier that qualified a buffer/image pair only when the buffer already
/// carried a linear texture layout withheld the batch from a pair that
/// `resolve_transfer` then declined for wanting it, and that decline is
/// waitable, so the channel head retried it forever.
pub enum TransferEndpoints {
    /// Neither endpoint resolves to an image: a plain buffer copy.
    Buffers,
    /// One endpoint is an image and the other its bytes.
    BufferImage {
        buffer: crate::replacement_buffer_blit::NativeBufferTarget,
        image: crate::replacement_image_transition::NativeImageTarget,
        image_key: ReplacementImageKey,
        /// The image is the destination.
        buffer_to_image: bool,
    },
    /// Two images over one allocation.
    ImageImage {
        source: crate::replacement_image_transition::NativeImageTarget,
        source_key: ReplacementImageKey,
        destination: crate::replacement_image_transition::NativeImageTarget,
        destination_key: ReplacementImageKey,
    },
    /// An endpoint reaches an image and the pair is none of the above --- a
    /// representation carrying both shapes, or one carrying neither opposite a
    /// one that carries an image.
    Unrecordable,
}

/// Classify a transfer's endpoints through the native registry.
pub fn classify_transfer_endpoints(
    transfer: TransferKey,
    resolver: &(impl ReplacementBufferResolver
          + crate::replacement_image_transition::ReplacementImageResolver),
) -> TransferEndpoints {
    let source_buffer = resolver.resolve_buffer(transfer.backing, transfer.source);
    let destination_buffer = resolver.resolve_buffer(transfer.backing, transfer.destination);
    let source_key = ReplacementImageKey {
        backing: transfer.backing,
        representation: transfer.source,
    };
    let destination_key = ReplacementImageKey {
        backing: transfer.backing,
        representation: transfer.destination,
    };
    let source_image = resolver.resolve_image(source_key);
    let destination_image = resolver.resolve_image(destination_key);
    match (
        source_buffer,
        source_image,
        destination_buffer,
        destination_image,
    ) {
        (_, None, _, None) => TransferEndpoints::Buffers,
        (Some(buffer), None, None, Some(image)) => TransferEndpoints::BufferImage {
            buffer,
            image,
            image_key: destination_key,
            buffer_to_image: true,
        },
        (None, Some(image), Some(buffer), None) => TransferEndpoints::BufferImage {
            buffer,
            image,
            image_key: source_key,
            buffer_to_image: false,
        },
        (None, Some(source), None, Some(destination)) => TransferEndpoints::ImageImage {
            source,
            source_key,
            destination,
            destination_key,
        },
        _ => TransferEndpoints::Unrecordable,
    }
}

/// Which endpoints of a transfer resolve to native images.
///
/// The image-state batch is built from this: an image read by a transfer needs
/// a transfer-source transition and one written needs a transfer-destination
/// one, and the pair says which. Deriving it from the representation
/// identities instead --- "an endpoint identity is the shared byte endpoint,
/// so the other side must be the image" --- is a rule about identities rather
/// than about what they resolve to, and it registers a designated *byte* view
/// as an image, which then fails to prepare as one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransferImageEndpoints {
    pub source: bool,
    pub destination: bool,
}

impl TransferImageEndpoints {
    /// Neither endpoint is an image, which is what a caller with no resolver
    /// to ask must assume.
    pub const NEITHER: Self = Self {
        source: false,
        destination: false,
    };

    /// Whether this transfer reaches an image at all.
    pub const fn any(self) -> bool {
        self.source || self.destination
    }
}

/// Which endpoints of a transfer the native registry resolves to images.
pub fn transfer_image_endpoints(
    transfer: TransferKey,
    resolver: &(impl ReplacementBufferResolver
          + crate::replacement_image_transition::ReplacementImageResolver),
) -> TransferImageEndpoints {
    match classify_transfer_endpoints(transfer, resolver) {
        TransferEndpoints::Buffers | TransferEndpoints::Unrecordable => {
            TransferImageEndpoints::NEITHER
        }
        TransferEndpoints::BufferImage {
            buffer_to_image, ..
        } => TransferImageEndpoints {
            source: !buffer_to_image,
            destination: buffer_to_image,
        },
        TransferEndpoints::ImageImage { .. } => TransferImageEndpoints {
            source: true,
            destination: true,
        },
    }
}

/// Whether recording this transfer reaches a native image, and so needs the
/// prepared image-state batch that [`resolve_transfer`] reads.
pub fn transfer_requires_image_state(
    transfer: TransferKey,
    resolver: &(impl ReplacementBufferResolver
          + crate::replacement_image_transition::ReplacementImageResolver),
) -> bool {
    !matches!(
        classify_transfer_endpoints(transfer, resolver),
        TransferEndpoints::Buffers
    )
}

fn resolve_transfer(
    backings: &[BackingId],
    transfer: TransferKey,
    image_state: Option<&PreparedImageState>,
    resolver: &(impl ReplacementBufferResolver
          + crate::replacement_image_transition::ReplacementImageResolver),
) -> Result<NativeResourceStateTransfer, ResourceStateTransferRecordError> {
    if !backings.contains(&transfer.backing) {
        return Err(ResourceStateTransferRecordError::TransferBackingNotDeclared(transfer.backing));
    }
    // The prepared image state is a prerequisite of every endpoint shape that
    // reaches an image, and of none that does not. Demanding it before the
    // shape is known reports a plain buffer copy as a missing preparation,
    // which reads as "not ready yet" and is retried forever.
    let state = || {
        image_state.ok_or(ResourceStateTransferRecordError::ImageTransferRequiresState(transfer))
    };
    match classify_transfer_endpoints(transfer, resolver) {
        TransferEndpoints::Buffers => resolve_buffer_transfer(backings, transfer, resolver)
            .map(NativeResourceStateTransfer::Buffer),
        TransferEndpoints::BufferImage {
            buffer,
            image,
            image_key,
            buffer_to_image,
        } => resolve_buffer_image_transfer(
            transfer,
            buffer,
            image,
            image_key,
            buffer_to_image,
            state()?,
            resolver,
        ),
        TransferEndpoints::ImageImage {
            source,
            source_key,
            destination,
            destination_key,
        } => resolve_image_image_transfer(
            transfer,
            source,
            source_key,
            destination,
            destination_key,
            state()?,
            resolver,
        ),
        TransferEndpoints::Unrecordable => Err(
            ResourceStateTransferRecordError::ImageEndpointMissing(transfer),
        ),
    }
}

/// Make one image over a backing current from another image over the same
/// backing.
///
/// Two textures declared over one allocation are two native images holding one
/// set of bytes, and the content authority designates both. When the guest's
/// own content lands in one of them, the other is brought current from it --
/// there is no third place those bytes exist, which is why the planner names a
/// sibling representation as the source rather than a staging endpoint.
///
/// The copy is therefore between images this device believes carry identical
/// content, so their subresource shape and pixel format must agree exactly. A
/// pair that disagrees is not a copy this device can express and is refused by
/// name: `vkCmdCopyImage` between mismatched formats or extents is not a
/// narrower copy, it is undefined.
#[allow(
    clippy::too_many_arguments,
    reason = "each endpoint contributes an image, a state key and a resolver query"
)]
fn resolve_image_image_transfer(
    transfer: TransferKey,
    source: crate::replacement_image_transition::NativeImageTarget,
    source_key: ReplacementImageKey,
    destination: crate::replacement_image_transition::NativeImageTarget,
    destination_key: ReplacementImageKey,
    state: &PreparedImageState,
    resolver: &impl ReplacementBufferResolver,
) -> Result<NativeResourceStateTransfer, ResourceStateTransferRecordError> {
    if let Some(disagreement) = image_endpoints_disagree(&source, &destination) {
        // Only a disagreement about how bytes are read can be bridged by the
        // bytes. A sample count is not a property the allocation carries at
        // all -- a multisampled image has no linear layout to walk -- and a
        // subresource-range disagreement is a mismatch in what each endpoint
        // claims to hold rather than in how it reads what it holds.
        return match disagreement {
            ImageEndpointDisagreement::PixelFormat { .. }
            | ImageEndpointDisagreement::Extent { .. } => resolve_alias_bytes_round_trip(
                transfer,
                source,
                source_key,
                destination,
                destination_key,
                state,
                resolver,
                disagreement,
            ),
            ImageEndpointDisagreement::Samples { .. }
            | ImageEndpointDisagreement::SubresourceRange { .. } => {
                Err(ResourceStateTransferRecordError::ImageToImageUnsupported {
                    transfer,
                    refusal: ImageToImageRefusal::EndpointsDisagree(disagreement),
                })
            }
        };
    }
    if !source.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
        || !destination
            .usage
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        return Err(ResourceStateTransferRecordError::ImageUsageMissing(
            transfer,
        ));
    }
    let source_layout = transfer_transition_layout(
        state,
        source_key,
        vk::ImageUsageFlags::TRANSFER_SRC,
        transfer,
    )?;
    let destination_layout = transfer_transition_layout(
        state,
        destination_key,
        vk::ImageUsageFlags::TRANSFER_DST,
        transfer,
    )?;
    if source.image == destination.image {
        return Err(ResourceStateTransferRecordError::ImageToImageUnsupported {
            transfer,
            refusal: ImageToImageRefusal::SameImage,
        });
    }
    let subresources: Vec<(u32, u32, [u32; 3], [i32; 3])> = match transfer.region {
        // The whole image is every subresource it declares, at that level's
        // own extent. A content version names the bytes, not one mip.
        BackingRegion::Whole => {
            let mut levels = Vec::new();
            for mip_offset in 0..source.full_range.level_count {
                let mip = source
                    .full_range
                    .base_mip_level
                    .checked_add(mip_offset)
                    .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
                        transfer,
                    ))?;
                let extent = [
                    (source.extent.width >> mip_offset).max(1),
                    (source.extent.height >> mip_offset).max(1),
                    (source.extent.depth >> mip_offset).max(1),
                ];
                for layer_offset in 0..source.full_range.layer_count {
                    let layer = source
                        .full_range
                        .base_array_layer
                        .checked_add(layer_offset)
                        .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
                            transfer,
                        ))?;
                    levels.push((mip, layer, extent, [0, 0, 0]));
                }
            }
            levels
        }
        BackingRegion::Image(region) => {
            let extent = [
                region.texels.end[0].saturating_sub(region.texels.origin[0]),
                region.texels.end[1].saturating_sub(region.texels.origin[1]),
                region.texels.end[2].saturating_sub(region.texels.origin[2]),
            ];
            let origin = [
                i32::try_from(region.texels.origin[0]),
                i32::try_from(region.texels.origin[1]),
                i32::try_from(region.texels.origin[2]),
            ];
            let [x, y, z] = origin;
            let origin = [
                x.map_err(|_| ResourceStateTransferRecordError::RangeOutOfBounds(transfer))?,
                y.map_err(|_| ResourceStateTransferRecordError::RangeOutOfBounds(transfer))?,
                z.map_err(|_| ResourceStateTransferRecordError::RangeOutOfBounds(transfer))?,
            ];
            vec![(region.mip, region.layer, extent, origin)]
        }
        // A byte range names no texels on its own, but these two endpoints
        // agree on how bytes are read -- that is what reaching this point
        // means -- so one layout translates the range for both, and the copy
        // stays a direct image-to-image one. Only a backing with no declared
        // layout at all leaves the range untranslatable.
        BackingRegion::Linear(_) => {
            let layout = resolver
                .resolve_linear_texture_layout(transfer.backing, transfer.source)
                .or_else(|| {
                    resolver.resolve_linear_texture_layout(transfer.backing, transfer.destination)
                })
                .ok_or(ResourceStateTransferRecordError::ImageToImageUnsupported {
                    transfer,
                    refusal: ImageToImageRefusal::LinearRegion,
                })?;
            transfer_image_regions(&layout, transfer)?
                .into_iter()
                .map(|region| {
                    (
                        region.mip,
                        region.layer,
                        [
                            region.texels.end[0] - region.texels.origin[0],
                            region.texels.end[1] - region.texels.origin[1],
                            region.texels.end[2] - region.texels.origin[2],
                        ],
                        [
                            i32::try_from(region.texels.origin[0]),
                            i32::try_from(region.texels.origin[1]),
                            i32::try_from(region.texels.origin[2]),
                        ],
                    )
                })
                .map(|(mip, layer, extent, origin)| {
                    let [x, y, z] = origin;
                    Ok((
                        mip,
                        layer,
                        extent,
                        [
                            x.map_err(|_| {
                                ResourceStateTransferRecordError::RangeOutOfBounds(transfer)
                            })?,
                            y.map_err(|_| {
                                ResourceStateTransferRecordError::RangeOutOfBounds(transfer)
                            })?,
                            z.map_err(|_| {
                                ResourceStateTransferRecordError::RangeOutOfBounds(transfer)
                            })?,
                        ],
                    ))
                })
                .collect::<Result<Vec<_>, ResourceStateTransferRecordError>>()?
        }
    };
    let mut commands = Vec::with_capacity(subresources.len());
    for (mip, layer, extent, offset) in subresources {
        if extent.contains(&0) {
            return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
        }
        // One command per aspect: `vkCmdCopyImage` takes a single aspect per
        // region, so a depth/stencil pair is two copies of the same box.
        for aspect in [
            vk::ImageAspectFlags::COLOR,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageAspectFlags::STENCIL,
        ] {
            if !source.full_range.aspect_mask.contains(aspect) {
                continue;
            }
            commands.push(NativeImageBlitCommand::ImageToImage(NativeImageCopy {
                source: source.image,
                source_layout,
                destination: destination.image,
                destination_layout,
                aspect,
                source_mip: mip,
                source_layer: layer,
                destination_mip: mip,
                destination_layer: layer,
                source_offset: offset,
                destination_offset: offset,
                extent,
            }));
        }
    }
    if commands.is_empty() {
        return Err(ResourceStateTransferRecordError::ImageToImageUnsupported {
            transfer,
            refusal: ImageToImageRefusal::NoCopyableAspect {
                aspect_mask: source.full_range.aspect_mask,
            },
        });
    }
    Ok(NativeResourceStateTransfer::Image(
        commands.into_boxed_slice(),
    ))
}

/// Make one image current from another over the same allocation when the two
/// interpret its bytes differently.
///
/// Two textures declared over one allocation at different pixel formats share
/// bytes and not texels: their texel blocks are different widths, so the same
/// byte range is a different subresource box in each. `vkCmdCopyImage` between
/// them is undefined and no image view can reinterpret one as the other, which
/// leaves exactly one thing they hold in common. The copy is therefore a round
/// trip: the source's texels out to the allocation's bytes in the source's own
/// layout, then those bytes back in to the destination's texels in the
/// destination's. The two halves are ordered by an explicit command, because a
/// command list is recorded in order and nothing else in it separates a write
/// from the read that follows.
///
/// The bytes go to a buffer the destination image owns for this purpose and
/// nothing else reads. They may not go to the guest's own memory or to the
/// shared transfer endpoint: both are representations the content authority
/// accounts for, and writing either would publish content at a version the
/// authority does not believe those bytes carry.
///
/// `disagreement` is the term the endpoints differed on. It is carried through
/// so that a refusal here still names why a direct copy was not available,
/// rather than reporting only the missing part of the round trip.
#[allow(
    clippy::too_many_arguments,
    reason = "each endpoint contributes an image, a state key, a layout and a resolver query"
)]
fn resolve_alias_bytes_round_trip(
    transfer: TransferKey,
    source: crate::replacement_image_transition::NativeImageTarget,
    source_key: ReplacementImageKey,
    destination: crate::replacement_image_transition::NativeImageTarget,
    destination_key: ReplacementImageKey,
    state: &PreparedImageState,
    resolver: &impl ReplacementBufferResolver,
    disagreement: ImageEndpointDisagreement,
) -> Result<NativeResourceStateTransfer, ResourceStateTransferRecordError> {
    let refuse = |missing| ResourceStateTransferRecordError::ImageToImageUnsupported {
        transfer,
        refusal: ImageToImageRefusal::AliasBytesUnavailable {
            disagreement,
            missing,
        },
    };
    let source_layout = resolver
        .resolve_linear_texture_layout(transfer.backing, transfer.source)
        .ok_or_else(|| refuse(AliasRoundTripTerm::SourceLayout))?;
    let destination_layout = resolver
        .resolve_linear_texture_layout(transfer.backing, transfer.destination)
        .ok_or_else(|| refuse(AliasRoundTripTerm::DestinationLayout))?;
    // Either endpoint's buffer will do -- the bytes are written and read
    // within one recording and nothing else reads them, so which image owns
    // the buffer decides nothing. It has to be either, because only the
    // texture that disagreed with the allocation's first-declared layout was
    // given one, and that is as often the source as the destination.
    let bytes = resolver
        .resolve_alias_transfer_bytes(transfer.backing, transfer.destination)
        .or_else(|| resolver.resolve_alias_transfer_bytes(transfer.backing, transfer.source))
        .ok_or_else(|| refuse(AliasRoundTripTerm::TransferBytes))?;
    let out = transfer_image_regions(&source_layout, transfer)?;
    // A byte-denominated region means the same thing to both endpoints and
    // each reads it in its own layout. A texel box does not: it is named in
    // the coordinates of whichever endpoint recorded the write, and the
    // disagreement is precisely that those do not carry over. The source is
    // the endpoint that has to be able to address it -- it is what the first
    // half of the round trip reads -- so the box is checked against the
    // source's layout, turned into the bytes it occupies there, and the
    // destination's texels are derived from those bytes rather than assumed
    // to be the same box.
    let back = match transfer.region {
        BackingRegion::Image(region) => {
            if !image_region_fits_layout(&source_layout, region, bytes.size) {
                return Err(refuse(AliasRoundTripTerm::RegionNamesTexels));
            }
            let mut derived = Vec::new();
            for range in image_region_byte_ranges(&source_layout, region)
                .ok_or_else(|| refuse(AliasRoundTripTerm::RegionNamesTexels))?
            {
                derived.extend(transfer_image_regions(
                    &destination_layout,
                    TransferKey {
                        region: BackingRegion::Linear(range),
                        ..transfer
                    },
                )?);
            }
            derived
        }
        BackingRegion::Whole | BackingRegion::Linear(_) => {
            transfer_image_regions(&destination_layout, transfer)?
        }
    };
    let mut commands = Vec::with_capacity(out.len() + back.len() + 1);
    for region in out {
        commands.push(resolve_buffer_image_transfer_region(
            transfer,
            bytes,
            source,
            source_key,
            false,
            state,
            &source_layout,
            region,
        )?);
    }
    commands.push(NativeImageBlitCommand::TransferBytesReady {
        buffer: bytes.buffer,
        offset: bytes.base_offset,
        size: bytes.accessible_size,
    });
    for region in back {
        commands.push(resolve_buffer_image_transfer_region(
            transfer,
            bytes,
            destination,
            destination_key,
            true,
            state,
            &destination_layout,
            region,
        )?);
    }
    Ok(NativeResourceStateTransfer::Image(
        commands.into_boxed_slice(),
    ))
}

/// The layout a prepared transition puts one endpoint in, once it has proved
/// the transition grants the usage this transfer needs.
fn transfer_transition_layout(
    state: &PreparedImageState,
    image: ReplacementImageKey,
    required_usage: vk::ImageUsageFlags,
    transfer: TransferKey,
) -> Result<vk::ImageLayout, ResourceStateTransferRecordError> {
    let transition = state
        .transitions()
        .iter()
        .find(|transition| transition.image == image)
        .ok_or(ResourceStateTransferRecordError::ImageStateMismatch(
            transfer,
        ))?;
    if !transition.required_usage.contains(required_usage) {
        return Err(ResourceStateTransferRecordError::ImageStateMismatch(
            transfer,
        ));
    }
    Ok(transition.use_layout)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the exact buffer/image endpoint pair and its direction are independent proofs"
)]
fn resolve_buffer_image_transfer(
    transfer: TransferKey,
    buffer: NativeBufferTarget,
    image: crate::replacement_image_transition::NativeImageTarget,
    image_key: ReplacementImageKey,
    buffer_to_image: bool,
    state: &PreparedImageState,
    resolver: &impl ReplacementBufferResolver,
) -> Result<NativeResourceStateTransfer, ResourceStateTransferRecordError> {
    // The layout describes how *this image's* texels sit in the backing's
    // bytes, so it must come from the image endpoint. Two textures declared
    // over one allocation differ in pixel format and therefore in width, row
    // stride and texel count, and the transfer endpoint carries whichever of
    // them was declared first -- reading the layout from there addresses the
    // second texture in the first one's coordinates. The endpoint's own
    // descriptor remains the answer when the image carries none, which is
    // every backing with exactly one texture over it.
    let (image_representation, buffer_representation) = if buffer_to_image {
        (transfer.destination, transfer.source)
    } else {
        (transfer.source, transfer.destination)
    };
    let layout = resolver
        .resolve_linear_texture_layout(transfer.backing, image_representation)
        .or_else(|| resolver.resolve_linear_texture_layout(transfer.backing, buffer_representation))
        .ok_or(ResourceStateTransferRecordError::ImageLayoutMissing(
            transfer,
        ))?;
    let regions = transfer_image_regions(&layout, transfer)?;
    let commands = regions
        .into_iter()
        .map(|region| {
            resolve_buffer_image_transfer_region(
                transfer,
                buffer,
                image,
                image_key,
                buffer_to_image,
                state,
                &layout,
                region,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    Ok(NativeResourceStateTransfer::Image(commands))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the exact buffer/image endpoint, subresource and direction are independent proofs"
)]
fn resolve_buffer_image_transfer_region(
    transfer: TransferKey,
    buffer: NativeBufferTarget,
    image: crate::replacement_image_transition::NativeImageTarget,
    image_key: ReplacementImageKey,
    buffer_to_image: bool,
    state: &PreparedImageState,
    layout: &reims_vgpu_protocol::LinearTextureDescriptor,
    region: reims_vgpu_core::ImageRegion,
) -> Result<NativeImageBlitCommand, ResourceStateTransferRecordError> {
    if !image_region_fits_layout(layout, region, buffer.size) {
        return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
    }
    let unsupported =
        |refusal| ResourceStateTransferRecordError::ImageLayoutUnsupported { transfer, refusal };
    let level = layout
        .level(region.mip)
        .ok_or_else(|| unsupported(ImageLayoutRefusal::LevelAbsent { mip: region.mip }))?;
    let bytes_per_element = u64::from(layout.bytes_per_element);
    if layout.compressed_layout {
        return Err(unsupported(ImageLayoutRefusal::CompressedLayout));
    }
    if bytes_per_element == 0 {
        return Err(unsupported(ImageLayoutRefusal::ZeroBytesPerElement));
    }
    if level.row_stride % bytes_per_element != 0 {
        return Err(unsupported(
            ImageLayoutRefusal::RowStrideNotElementAligned {
                row_stride: level.row_stride,
                bytes_per_element,
            },
        ));
    }
    let row_length_texels = level.row_stride / bytes_per_element;
    let row_length = u32::try_from(row_length_texels).map_err(|_| {
        unsupported(ImageLayoutRefusal::RowLengthNotRepresentable {
            row_length: row_length_texels,
        })
    })?;
    let level_offset = layout
        .subresource_offset(region.layer, region.mip)
        .ok_or_else(|| {
            unsupported(ImageLayoutRefusal::SubresourceOffsetAbsent {
                layer: region.layer,
                mip: region.mip,
            })
        })?;
    let image_stride = level
        .row_stride
        .checked_mul(u64::from(level.height))
        .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
            transfer,
        ))?;
    let relative_offset = level_offset
        .checked_add(
            u64::from(region.texels.origin[2])
                .checked_mul(image_stride)
                .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
                    transfer,
                ))?,
        )
        .and_then(|offset| {
            offset.checked_add(u64::from(region.texels.origin[1]).checked_mul(level.row_stride)?)
        })
        .and_then(|offset| {
            offset.checked_add(u64::from(region.texels.origin[0]).checked_mul(bytes_per_element)?)
        })
        .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
            transfer,
        ))?;
    let buffer_offset = buffer.base_offset.checked_add(relative_offset).ok_or(
        ResourceStateTransferRecordError::RangeAddressOverflow(transfer),
    )?;
    let required_usage = if buffer_to_image {
        vk::ImageUsageFlags::TRANSFER_DST
    } else {
        vk::ImageUsageFlags::TRANSFER_SRC
    };
    let required_buffer_usage = if buffer_to_image {
        vk::BufferUsageFlags::TRANSFER_SRC
    } else {
        vk::BufferUsageFlags::TRANSFER_DST
    };
    if !image.usage.contains(required_usage) || !buffer.usage.contains(required_buffer_usage) {
        return Err(ResourceStateTransferRecordError::ImageUsageMissing(
            transfer,
        ));
    }
    let transition = state
        .transitions()
        .iter()
        .find(|transition| transition.image == image_key)
        .ok_or(ResourceStateTransferRecordError::ImageStateMismatch(
            transfer,
        ))?;
    if !transition.required_usage.contains(required_usage) {
        return Err(ResourceStateTransferRecordError::ImageStateMismatch(
            transfer,
        ));
    }
    let range = crate::replacement_image_transition::exact_image_subresource_range(
        BackingRegion::Image(region),
    )
    .ok_or_else(|| unsupported(ImageLayoutRefusal::SubresourceRangeInexact))?;
    let mip_end = image
        .full_range
        .base_mip_level
        .checked_add(image.full_range.level_count)
        .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
            transfer,
        ))?;
    let layer_end = image
        .full_range
        .base_array_layer
        .checked_add(image.full_range.layer_count)
        .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
            transfer,
        ))?;
    if !image.full_range.aspect_mask.contains(range.aspect_mask)
        || region.mip < image.full_range.base_mip_level
        || region.mip >= mip_end
        || region.layer < image.full_range.base_array_layer
        || region.layer >= layer_end
    {
        return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
    }
    let copy = NativeBufferImageCopy {
        buffer: buffer.buffer,
        image: image.image,
        image_layout: transition.use_layout,
        buffer_offset,
        buffer_row_length: row_length,
        buffer_image_height: level.height,
        aspect: range.aspect_mask,
        mip: region.mip,
        layer: region.layer,
        image_offset: [
            i32::try_from(region.texels.origin[0]).map_err(|_| {
                unsupported(ImageLayoutRefusal::CoordinateNotRepresentable {
                    term: LayoutCoordinateTerm::OriginX,
                    value: u64::from(region.texels.origin[0]),
                })
            })?,
            i32::try_from(region.texels.origin[1]).map_err(|_| {
                unsupported(ImageLayoutRefusal::CoordinateNotRepresentable {
                    term: LayoutCoordinateTerm::OriginY,
                    value: u64::from(region.texels.origin[1]),
                })
            })?,
            i32::try_from(region.texels.origin[2]).map_err(|_| {
                unsupported(ImageLayoutRefusal::CoordinateNotRepresentable {
                    term: LayoutCoordinateTerm::OriginZ,
                    value: u64::from(region.texels.origin[2]),
                })
            })?,
        ],
        extent: [
            region.texels.end[0] - region.texels.origin[0],
            region.texels.end[1] - region.texels.origin[1],
            region.texels.end[2] - region.texels.origin[2],
        ],
    };
    Ok(if buffer_to_image {
        NativeImageBlitCommand::BufferToImage(copy)
    } else {
        NativeImageBlitCommand::ImageToBuffer(copy)
    })
}

/// The bytes one texel box occupies in a layout, as the fewest ranges that
/// cover them.
///
/// A row-strided image relates texels to bytes one row at a time, so a box
/// narrower than its level is a range per row; a full-width box is contiguous
/// across the rows it spans, and a full-height one across the planes, so those
/// coalesce back into single ranges. `None` where the arithmetic leaves the
/// layout's declared extents, which is the caller's cue that the box does not
/// belong to this layout at all.
fn image_region_byte_ranges(
    layout: &reims_vgpu_protocol::LinearTextureDescriptor,
    region: reims_vgpu_core::ImageRegion,
) -> Option<Vec<reims_vgpu_core::LinearRange>> {
    let level = layout.level(region.mip)?;
    let bytes_per_element = u64::from(layout.bytes_per_element);
    if bytes_per_element == 0 || level.row_stride == 0 {
        return None;
    }
    let level_offset = layout.subresource_offset(region.layer, region.mip)?;
    let image_stride = level.row_stride.checked_mul(u64::from(level.height))?;
    let width = u64::from(region.texels.end[0].checked_sub(region.texels.origin[0])?);
    let length = width.checked_mul(bytes_per_element)?;
    if length == 0 {
        return None;
    }
    let mut ranges: Vec<reims_vgpu_core::LinearRange> = Vec::new();
    for plane in region.texels.origin[2]..region.texels.end[2] {
        for row in region.texels.origin[1]..region.texels.end[1] {
            let start = level_offset
                .checked_add(u64::from(plane).checked_mul(image_stride)?)?
                .checked_add(u64::from(row).checked_mul(level.row_stride)?)?
                .checked_add(u64::from(region.texels.origin[0]).checked_mul(bytes_per_element)?)?;
            match ranges.last_mut() {
                Some(previous) if previous.end() == start => {
                    *previous = reims_vgpu_core::LinearRange::new(
                        previous.start(),
                        previous.end() - previous.start() + length,
                    )?;
                }
                _ => ranges.push(reims_vgpu_core::LinearRange::new(start, length)?),
            }
        }
    }
    (!ranges.is_empty()).then_some(ranges)
}

fn transfer_image_regions(
    layout: &reims_vgpu_protocol::LinearTextureDescriptor,
    transfer: TransferKey,
) -> Result<Vec<reims_vgpu_core::ImageRegion>, ResourceStateTransferRecordError> {
    if let BackingRegion::Image(region) = transfer.region {
        return Ok(vec![region]);
    }
    let unsupported =
        |refusal| ResourceStateTransferRecordError::ImageLayoutUnsupported { transfer, refusal };
    if matches!(transfer.region, BackingRegion::Whole) {
        let format = layout
            .declared_pixel_format()
            .ok_or_else(|| unsupported(ImageLayoutRefusal::PixelFormatUndeclared))?;
        if reims_vgpu_core::pixel_format::format_has_depth_aspect(format)
            || reims_vgpu_core::pixel_format::format_has_stencil_aspect(format)
        {
            return Err(unsupported(ImageLayoutRefusal::DepthStencilFormat {
                pixel_format: format,
            }));
        }
        let slices = layout
            .physical_slice_count()
            .ok_or_else(|| unsupported(ImageLayoutRefusal::SliceCountUndeclared))?;
        let mut regions = Vec::new();
        for layer in 0..slices {
            for mip in 0..layout.mipmap_level_count {
                let level = layout
                    .level(mip)
                    .ok_or_else(|| unsupported(ImageLayoutRefusal::LevelAbsent { mip }))?;
                let extent = [level.width, level.height, level.planes()];
                let texels =
                    reims_vgpu_core::TexelBox::new([0, 0, 0], extent).ok_or_else(|| {
                        unsupported(ImageLayoutRefusal::ExtentNotATexelBox {
                            origin: [0, 0, 0],
                            extent,
                        })
                    })?;
                regions.push(reims_vgpu_core::ImageRegion {
                    aspect: reims_vgpu_core::ImageAspect::Color,
                    mip,
                    layer,
                    texels,
                });
            }
        }
        return Ok(regions);
    }
    let BackingRegion::Linear(range) = transfer.region else {
        unreachable!("image and whole backing regions returned above")
    };
    if range.end() > layout.allocation_size {
        return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
    }
    let slices = layout
        .physical_slice_count()
        .ok_or_else(|| unsupported(ImageLayoutRefusal::SliceCountUndeclared))?;
    let mut regions = Vec::new();
    for layer in 0..slices {
        for mip in 0..layout.mipmap_level_count {
            let Some(level) = layout.level(mip) else {
                continue;
            };
            let Some(base) = layout.subresource_offset(layer, mip) else {
                continue;
            };
            let Some(end) = base.checked_add(level.size) else {
                continue;
            };
            let start = range.start().max(base);
            let end = range.end().min(end);
            if start >= end {
                continue;
            }
            let region_transfer = TransferKey {
                region: BackingRegion::Linear(
                    reims_vgpu_core::LinearRange::new(start, end - start)
                        .expect("a non-empty checked intersection is a linear range"),
                ),
                ..transfer
            };
            regions.extend(transfer_image_region(layout, region_transfer)?);
        }
    }
    Ok(regions)
}

/// The texel boxes one byte range covers inside one subresource.
///
/// A row-strided image relates bytes to texels one row at a time, so a byte
/// range is a union of row segments and not, in general, a rectangle: it may
/// start part way along a row, end part way along another, and pass over the
/// padding between a row's declared width and its stride, which belongs to no
/// texel and carries no content. Reading it as a single box refused every
/// range that was not already rectangular -- a whole class of ordinary guest
/// writes -- and reading it as one box per row would emit a copy command for
/// every row of every upload. So the walk is per row and the result is
/// coalesced back up: contiguous full-width rows become one box, and
/// contiguous full-height planes become one box after that.
fn transfer_image_region(
    layout: &reims_vgpu_protocol::LinearTextureDescriptor,
    transfer: TransferKey,
) -> Result<Vec<reims_vgpu_core::ImageRegion>, ResourceStateTransferRecordError> {
    if let BackingRegion::Image(region) = transfer.region {
        return Ok(vec![region]);
    }
    let unsupported =
        |refusal| ResourceStateTransferRecordError::ImageLayoutUnsupported { transfer, refusal };
    let BackingRegion::Linear(range) = transfer.region else {
        return Err(unsupported(ImageLayoutRefusal::RegionNotLinear));
    };
    let pixel_format = layout
        .declared_pixel_format()
        .ok_or_else(|| unsupported(ImageLayoutRefusal::PixelFormatUndeclared))?;
    if reims_vgpu_core::pixel_format::format_has_depth_aspect(pixel_format)
        || reims_vgpu_core::pixel_format::format_has_stencil_aspect(pixel_format)
    {
        return Err(unsupported(ImageLayoutRefusal::DepthStencilFormat {
            pixel_format,
        }));
    }
    if layout.compressed_layout {
        return Err(unsupported(ImageLayoutRefusal::CompressedLayout));
    }
    if layout.bytes_per_element == 0 {
        return Err(unsupported(ImageLayoutRefusal::ZeroBytesPerElement));
    }
    let bytes_per_element = u64::from(layout.bytes_per_element);
    let slices = layout
        .physical_slice_count()
        .ok_or_else(|| unsupported(ImageLayoutRefusal::SliceCountUndeclared))?;
    for layer in 0..slices {
        for mip in 0..layout.mipmap_level_count {
            let Some(level) = layout.level(mip) else {
                continue;
            };
            let Some(base) = layout.subresource_offset(layer, mip) else {
                continue;
            };
            let Some(end) = base.checked_add(level.size) else {
                continue;
            };
            if range.start() < base || range.end() > end {
                continue;
            }
            let image_stride = level
                .row_stride
                .checked_mul(u64::from(level.height))
                .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
                    transfer,
                ))?;
            if image_stride == 0 || level.row_stride == 0 {
                return Err(unsupported(ImageLayoutRefusal::ZeroStride {
                    row_stride: level.row_stride,
                    image_stride,
                }));
            }
            let tight_row = u64::from(level.width)
                .checked_mul(bytes_per_element)
                .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
                    transfer,
                ))?;
            let mut rows: Vec<[u64; 4]> = Vec::new();
            let mut cursor = range.start() - base;
            let relative_end = range.end() - base;
            while cursor < relative_end {
                let plane = cursor / image_stride;
                let within_image = cursor % image_stride;
                let row = within_image / level.row_stride;
                let offset_in_row = within_image % level.row_stride;
                let row_base = cursor - offset_in_row;
                let next_row = row_base + level.row_stride;
                // The padding past the declared width is not addressable as
                // texels and holds nothing, so a range that only reaches into
                // it contributes no copy and simply moves on.
                let texel_end = relative_end.min(next_row).min(row_base + tight_row);
                if texel_end > cursor {
                    if offset_in_row % bytes_per_element != 0 {
                        return Err(unsupported(ImageLayoutRefusal::OffsetNotElementAligned {
                            offset: offset_in_row,
                            bytes_per_element,
                        }));
                    }
                    let length = texel_end - cursor;
                    if length % bytes_per_element != 0 {
                        return Err(unsupported(ImageLayoutRefusal::LengthNotElementAligned {
                            length,
                            bytes_per_element,
                        }));
                    }
                    rows.push([
                        plane,
                        row,
                        offset_in_row / bytes_per_element,
                        (offset_in_row + length) / bytes_per_element,
                    ]);
                }
                cursor = next_row;
            }
            let coordinate = |term, value| {
                move |_| unsupported(ImageLayoutRefusal::CoordinateNotRepresentable { term, value })
            };
            // Contiguous full rows of one plane, then contiguous full planes:
            // an aligned whole-level range therefore records as one command
            // rather than one per row.
            let mut boxes: Vec<[u64; 6]> = Vec::new();
            for [plane, row, first, last] in rows {
                match boxes.last_mut() {
                    Some(previous)
                        if previous[2] == plane
                            && previous[0] == first
                            && previous[1] == last
                            && first == 0
                            && last == u64::from(level.width)
                            && previous[3] + previous[4] == row =>
                    {
                        previous[4] += 1;
                    }
                    _ => boxes.push([first, last, plane, row, 1, 1]),
                }
            }
            let mut merged: Vec<[u64; 6]> = Vec::new();
            for current in boxes {
                match merged.last_mut() {
                    Some(previous)
                        if previous[0] == current[0]
                            && previous[1] == current[1]
                            && previous[3] == current[3]
                            && previous[4] == current[4]
                            && current[3] == 0
                            && current[4] == u64::from(level.height)
                            && previous[2] + previous[5] == current[2] =>
                    {
                        previous[5] += 1;
                    }
                    _ => merged.push(current),
                }
            }
            let mut regions = Vec::with_capacity(merged.len());
            for [first, last, plane, row, height, depth] in merged {
                let width = last - first;
                let origin = [
                    u32::try_from(first)
                        .map_err(coordinate(LayoutCoordinateTerm::OriginX, first))?,
                    u32::try_from(row).map_err(coordinate(LayoutCoordinateTerm::OriginY, row))?,
                    u32::try_from(plane)
                        .map_err(coordinate(LayoutCoordinateTerm::OriginZ, plane))?,
                ];
                let extent = [
                    u32::try_from(width).map_err(coordinate(LayoutCoordinateTerm::Width, width))?,
                    u32::try_from(height)
                        .map_err(coordinate(LayoutCoordinateTerm::Height, height))?,
                    u32::try_from(depth).map_err(coordinate(LayoutCoordinateTerm::Depth, depth))?,
                ];
                let texels = reims_vgpu_core::TexelBox::new(origin, extent).ok_or_else(|| {
                    unsupported(ImageLayoutRefusal::ExtentNotATexelBox { origin, extent })
                })?;
                if texels.end[0] > level.width
                    || texels.end[1] > level.height
                    || texels.end[2] > level.planes()
                {
                    return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
                }
                regions.push(reims_vgpu_core::ImageRegion {
                    aspect: reims_vgpu_core::ImageAspect::Color,
                    mip,
                    layer,
                    texels,
                });
            }
            // A range wholly inside one subresource's row padding names no
            // texels at all. It is not a transfer this device can record and
            // it is not a loss either, but an empty command list would read as
            // a completed copy, so it refuses under the range's own name.
            if regions.is_empty() {
                return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
            }
            return Ok(regions);
        }
    }
    Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer))
}

fn validate_native_end(
    target: NativeBufferTarget,
    offset: u64,
    size: u64,
    transfer: TransferKey,
) -> Result<(), ResourceStateTransferRecordError> {
    let relative = offset
        .checked_sub(target.base_offset)
        .and_then(|start| start.checked_add(size))
        .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
            transfer,
        ))?;
    if relative > target.size {
        return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
    }
    Ok(())
}

fn ranges_overlap(left: u64, left_len: u64, right: u64, right_len: u64) -> bool {
    let Some(left_end) = left.checked_add(left_len) else {
        return true;
    };
    let Some(right_end) = right.checked_add(right_len) else {
        return true;
    };
    left < right_end && right < left_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::{
        prepare_resource_state, BackingRegion, CompletionStamp, ExecTransaction, LinearRange,
        RepresentationRoute, ResolvedExecSegment, ResolvedExecStream, ResolvedOperation,
        ResolvedResourceLifecycle, ResolvedResourceStateTarget, ResourceLifecycleEffect,
        ResourceLifecycleOwner, SessionGeneration, StorageBacking, TransactionRuntime,
        ValidityRepresentations, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ChannelId, RepresentationId, ResourceValidityOps, SegmentBoundary, SegmentKind,
        SessionGenerationId, SubmissionId, SubmissionIdentity, TaskId, VulkanDeviceEpochId,
    };
    use std::collections::BTreeMap;

    struct Resolver(BTreeMap<(BackingId, RepresentationId), NativeBufferTarget>);

    impl ReplacementBufferResolver for Resolver {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            self.0.get(&(backing, representation)).copied()
        }
    }

    fn prepared(region: BackingRegion) -> (PreparedResourceState, TransferKey, Resolver) {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = resources
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        assert_eq!(
            resources
                .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
                .unwrap(),
            GUEST_REPRESENTATION
        );
        resources
            .plan_gpu_write(backing, SubmissionId::new(1), source, [region])
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([region]),
            }]),
            ops: ResourceValidityOps {
                clear_guest_valid: 1,
                set_host_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let channel = ChannelId::new(1);
        let mut runtime =
            TransactionRuntime::<()>::new(SessionGeneration::new(SessionGenerationId::new(1)));
        runtime.define_channel(channel).unwrap();
        let admitted = runtime
            .admit_exec_operations(
                channel,
                Box::<[reims_vgpu_core::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                ExecTransaction::<ResolvedOperation<(), (), (), (), ()>> {
                    identity: SubmissionIdentity {
                        id: SubmissionId::new(2),
                        task: TaskId::new(1),
                    },
                    prologue: reims_vgpu_core::ExecPrologue::default(),
                    streams: Box::new([ResolvedExecStream {
                        stream_index: 0,
                        segments: Box::new([ResolvedExecSegment {
                            boundary: SegmentBoundary {
                                stream_index: 0,
                                index: 0,
                                kind: SegmentKind::Blit,
                                continues_previous: false,
                                continues_next: false,
                            },
                            operations: Box::new([ResolvedOperation::ResourceState(operation)]),
                        }]),
                    }]),
                    accesses: Box::new([]),
                },
            )
            .unwrap();
        let (envelope, _, _, states, _) = admitted.into_parts();
        let prepared =
            prepare_resource_state(&mut resources, &states, 0, SubmissionId::new(2), |_, _| {
                ValidityRepresentations {
                    host_write: Some(source),
                    host_ingress_destination: None,
                    guest_upload_destination: None,
                    guest_visibility_source: Some(source),
                    guest_visibility_destination: reims_vgpu_core::GUEST_REPRESENTATION,
                }
            })
            .unwrap();
        assert_eq!(prepared.transaction(), envelope.id);
        let transfer = prepared.transfers()[0];
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let resolver = Resolver(BTreeMap::from([
            (
                (backing, source),
                NativeBufferTarget {
                    buffer: vk::Buffer::from_raw(11),
                    base_offset: 100,
                    accessible_size: 64,
                    size: 64,
                    usage,
                },
            ),
            (
                (backing, GUEST_REPRESENTATION),
                NativeBufferTarget {
                    buffer: vk::Buffer::from_raw(12),
                    base_offset: 200,
                    accessible_size: 64,
                    size: 64,
                    usage,
                },
            ),
        ]));
        (prepared, transfer, resolver)
    }

    #[test]
    fn linear_transfer_preserves_canonical_coordinates() {
        let region = BackingRegion::Linear(LinearRange::new(8, 16).unwrap());
        let (prepared, key, resolver) = prepared(region);
        let program = ReplacementResourceStateProgram::resolve(&prepared, &resolver).unwrap();
        assert_eq!(program.transfers(), &[key]);
        assert_eq!(
            program.completions(),
            &[
                ResolvedResourceCompletion::Transfer(key),
                ResolvedResourceCompletion::ValidityHostWrite {
                    backing: key.backing,
                    write: reims_vgpu_core::GpuWriteId::operation(
                        prepared.transaction(),
                        SubmissionId::new(2),
                        0,
                    ),
                    representation: key.source,
                },
            ]
        );
        assert_eq!(
            program.native_transfers(),
            &[NativeResourceStateTransfer::Buffer(
                NativeBufferBlit::Copy {
                    source: vk::Buffer::from_raw(11),
                    destination: vk::Buffer::from_raw(12),
                    source_offset: 108,
                    destination_offset: 208,
                    size: 16,
                }
            )]
        );
    }

    #[test]
    fn linear_transfer_reports_both_semantic_endpoint_bounds() {
        let region = BackingRegion::Linear(LinearRange::new(8, 16).unwrap());
        let (prepared, key, mut resolver) = prepared(region);
        resolver.0.get_mut(&(key.backing, key.source)).unwrap().size = 24;
        resolver
            .0
            .get_mut(&(key.backing, key.destination))
            .unwrap()
            .size = 20;
        assert!(matches!(
            ReplacementResourceStateProgram::resolve(&prepared, &resolver),
            Err(ResourceStateTransferRecordError::LinearRangeOutOfBounds {
                transfer: found,
                source_size: 24,
                destination_size: 20,
            }) if found == key
        ));
    }

    #[test]
    fn image_transfer_refuses_without_fabricating_buffer_copy_state() {
        let image = BackingRegion::Image(reims_vgpu_core::ImageRegion {
            aspect: reims_vgpu_core::ImageAspect::Color,
            mip: 0,
            layer: 0,
            texels: reims_vgpu_core::TexelBox::new([0, 0, 0], [1, 1, 1]).unwrap(),
        });
        let (prepared, key, resolver) = prepared(image);
        assert!(matches!(
            ReplacementResourceStateProgram::resolve(&prepared, &resolver),
            Err(ResourceStateTransferRecordError::ImageTransferRequiresState(found))
                if found == key
        ));
    }

    #[test]
    fn an_unresolvable_endpoint_names_its_shape_and_not_a_missing_state() {
        use crate::replacement_image_transition::{NativeImageTarget, ReplacementImageResolver};
        // A transfer with one image endpoint and one endpoint that resolves to
        // nothing at all can never be recorded, whatever preparation runs.
        // Naming it as a missing image state makes a permanent refusal look
        // like a wait, and the channel retries it forever behind its own
        // submission head. The prepared state belongs to the arms that read
        // it, which this is not.
        struct OneImage {
            backing: BackingId,
            image: RepresentationId,
        }
        impl ReplacementBufferResolver for OneImage {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                None
            }
        }
        impl ReplacementImageResolver for OneImage {
            fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
                (image.backing == self.backing && image.representation == self.image).then_some(
                    NativeImageTarget {
                        image: vk::Image::from_raw(33),
                        view: vk::ImageView::from_raw(34),
                        image_type: vk::ImageType::TYPE_2D,
                        full_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
                        extent: vk::Extent3D {
                            width: 4,
                            height: 4,
                            depth: 1,
                        },
                        samples: vk::SampleCountFlags::TYPE_1,
                    },
                )
            }
        }
        let (prepared, key, _) = prepared(BackingRegion::Whole);
        let resolver = OneImage {
            backing: key.backing,
            image: key.source,
        };
        assert!(matches!(
            ReplacementResourceStateProgram::resolve_with_image_state(&prepared, None, &resolver),
            Err(ResourceStateTransferRecordError::ImageEndpointMissing(found))
                if found == key
        ));
    }

    #[test]
    fn a_whole_transfer_between_two_images_over_one_backing_copies_every_subresource() {
        use crate::replacement_image_transition::{NativeImageTarget, ReplacementImageResolver};
        // Two textures declared over one allocation are two native images
        // holding one set of bytes. When the guest's content lands in one, the
        // other is brought current from it, and there is no third place those
        // bytes exist -- so this is the copy, not a fallback for one.
        struct TwoImages {
            backing: BackingId,
        }
        fn target() -> NativeImageTarget {
            NativeImageTarget {
                image: vk::Image::from_raw(0),
                view: vk::ImageView::from_raw(0),
                image_type: vk::ImageType::TYPE_2D,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 2,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                samples: vk::SampleCountFlags::TYPE_1,
            }
        }
        impl ReplacementBufferResolver for TwoImages {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                None
            }
        }
        impl ReplacementImageResolver for TwoImages {
            fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
                (image.backing == self.backing).then(|| {
                    let mut target = target();
                    target.image = vk::Image::from_raw(image.representation.get() + 1);
                    target
                })
            }
        }
        let (prepared, key, _) = prepared(BackingRegion::Whole);
        let resolver = TwoImages {
            backing: key.backing,
        };
        let key_for = |representation| ReplacementImageKey {
            backing: key.backing,
            representation,
        };
        let mut images = crate::replacement_image_state::ReplacementImageStateOwner::new(
            reims_vgpu_protocol::VulkanDeviceEpochId::new(1),
        );
        for representation in [key.source, key.destination] {
            images
                .register(
                    key_for(representation),
                    crate::replacement_image_state::ReplacementImageState {
                        layout: vk::ImageLayout::GENERAL,
                        sharing:
                            crate::replacement_image_state::ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let state = images
            .prepare_operation(
                prepared.transaction(),
                prepared.index(),
                3,
                [
                    crate::replacement_image_state::ReplacementImageUse {
                        image: key_for(key.source),
                        required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    },
                    crate::replacement_image_state::ReplacementImageUse {
                        image: key_for(key.destination),
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    },
                ],
            )
            .unwrap();
        let program = ReplacementResourceStateProgram::resolve_with_image_state(
            &prepared,
            Some(&state),
            &resolver,
        )
        .expect("two images over one backing is a copy this device records");
        let [NativeResourceStateTransfer::Image(commands)] = program.native_transfers() else {
            panic!("a whole image-to-image transfer records image copies");
        };
        // Both declared mip levels, at their own extents, in one direction.
        assert_eq!(commands.len(), 2);
        let extents = commands
            .iter()
            .map(|command| {
                let NativeImageBlitCommand::ImageToImage(copy) = command else {
                    panic!("an image-to-image transfer records image copies")
                };
                assert_eq!(copy.source_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                assert_eq!(
                    copy.destination_layout,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                );
                assert_ne!(copy.source, copy.destination);
                (copy.source_mip, copy.extent)
            })
            .collect::<Vec<_>>();
        assert_eq!(extents, vec![(0, [4, 4, 1]), (1, [2, 2, 1])]);
    }

    #[test]
    fn one_linear_texture_transfer_expands_across_mips_and_skips_padding() {
        let layout = reims_vgpu_protocol::LinearTextureDescriptor {
            allocation_size: 64,
            mipmap_level_count: 2,
            bytes_per_slice: 64,
            slice_count: 1,
            bytes_per_element: 1,
            row_stride: 8,
            width: 4,
            height: 4,
            depth: 1,
            declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                texture_type: reims_vgpu_protocol::TextureType::D2,
                framebuffer_only: false,
                is_drawable: false,
                write_swizzle_enabled: None,
                allow_gpu_optimized_contents: false,
                usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
                width: 4,
                height: 4,
                depth: 1,
                mipmap_level_count: 2,
                sample_count: 1,
                array_length: 1,
                resource_options: 0,
                protection_options: 0,
                swizzle: None,
            }),
            levels: vec![
                reims_vgpu_protocol::TextureLevelLayout {
                    offset: 0,
                    size: 32,
                    row_stride: 8,
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                reims_vgpu_protocol::TextureLevelLayout {
                    offset: 40,
                    size: 16,
                    row_stride: 8,
                    width: 2,
                    height: 2,
                    depth: 1,
                },
            ],
            ..Default::default()
        };
        let transfer = TransferKey {
            backing: BackingId::new(1),
            region: BackingRegion::Linear(LinearRange::new(0, 64).unwrap()),
            version: reims_vgpu_protocol::ContentVersion::new(1),
            source: RepresentationId::new(3),
            destination: RepresentationId::new(4),
        };
        let regions = transfer_image_regions(&layout, transfer).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].mip, 0);
        assert_eq!(regions[0].texels.origin, [0, 0, 0]);
        assert_eq!(regions[0].texels.end, [4, 4, 1]);
        assert_eq!(regions[1].mip, 1);
        assert_eq!(regions[1].texels.origin, [0, 0, 0]);
        assert_eq!(regions[1].texels.end, [2, 2, 1]);

        let whole = TransferKey {
            region: BackingRegion::Whole,
            ..transfer
        };
        assert_eq!(transfer_image_regions(&layout, whole).unwrap(), regions);
    }

    /// A layout refusal names the term that stopped it, not just the transfer.
    ///
    /// Thirty-one sites over three functions answered "this layout is
    /// unsupported" and nothing else, so a boot that hit one reported a
    /// transfer key and left the next question -- what to implement -- with
    /// nowhere to start.
    #[test]
    fn an_unsupported_layout_refusal_names_which_term_it_refused() {
        let layout = |bytes_per_element: u8, compressed: bool| {
            reims_vgpu_protocol::LinearTextureDescriptor {
                allocation_size: 64,
                mipmap_level_count: 1,
                bytes_per_slice: 64,
                slice_count: 1,
                bytes_per_element,
                row_stride: 8,
                width: 4,
                height: 4,
                depth: 1,
                compressed_layout: compressed,
                declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                    texture_type: reims_vgpu_protocol::TextureType::D2,
                    framebuffer_only: false,
                    is_drawable: false,
                    write_swizzle_enabled: None,
                    allow_gpu_optimized_contents: false,
                    usage: reims_vgpu_protocol::TEXTURE_USAGE_SHADER_READ,
                    pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
                    width: 4,
                    height: 4,
                    depth: 1,
                    mipmap_level_count: 1,
                    sample_count: 1,
                    array_length: 1,
                    resource_options: 0,
                    protection_options: 0,
                    swizzle: None,
                }),
                levels: vec![reims_vgpu_protocol::TextureLevelLayout {
                    offset: 0,
                    size: 32,
                    row_stride: 8,
                    width: 4,
                    height: 4,
                    depth: 1,
                }],
                ..Default::default()
            }
        };
        let transfer = TransferKey {
            backing: BackingId::new(1),
            region: BackingRegion::Linear(LinearRange::new(0, 32).unwrap()),
            version: reims_vgpu_protocol::ContentVersion::new(1),
            source: RepresentationId::new(3),
            destination: RepresentationId::new(4),
        };

        // A compressed layout is a block coordinate space this device does not
        // translate, and it says so by that name.
        assert_eq!(
            transfer_image_region(&layout(1, true), transfer),
            Err(ResourceStateTransferRecordError::ImageLayoutUnsupported {
                transfer,
                refusal: ImageLayoutRefusal::CompressedLayout,
            })
        );

        // A zero element size is a different refusal and must not be reported
        // as the same one.
        assert_eq!(
            transfer_image_region(&layout(0, false), transfer),
            Err(ResourceStateTransferRecordError::ImageLayoutUnsupported {
                transfer,
                refusal: ImageLayoutRefusal::ZeroBytesPerElement,
            })
        );

        // A byte range that starts mid-row and ends mid-row is three row
        // segments: the tail of the row it starts in, the whole rows between,
        // and the head of the row it ends in. The four bytes of padding at
        // the end of each eight-byte row name no texels and are stepped over.
        let ragged = TransferKey {
            region: BackingRegion::Linear(LinearRange::new(1, 16).unwrap()),
            ..transfer
        };
        let regions = transfer_image_region(&layout(1, false), ragged).unwrap();
        assert_eq!(
            regions
                .iter()
                .map(|region| (region.texels.origin, region.texels.end))
                .collect::<Vec<_>>(),
            vec![
                ([1, 0, 0], [4, 1, 1]),
                ([0, 1, 0], [4, 2, 1]),
                ([0, 2, 0], [1, 3, 1]),
            ]
        );

        // A whole level is still one box and not one per row.
        let regions = transfer_image_region(&layout(1, false), transfer).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].texels.origin, [0, 0, 0]);
        assert_eq!(regions[0].texels.end, [4, 4, 1]);
    }

    #[test]
    fn image_transfer_uses_its_prepared_layout_and_linear_endpoint_packing() {
        use crate::replacement_image_state::{
            ReplacementImageSharing, ReplacementImageState, ReplacementImageStateOwner,
            ReplacementImageUse,
        };
        use crate::replacement_image_transition::{NativeImageTarget, ReplacementImageResolver};

        let image_region = reims_vgpu_core::ImageRegion {
            aspect: reims_vgpu_core::ImageAspect::Color,
            mip: 0,
            layer: 0,
            texels: reims_vgpu_core::TexelBox::new([1, 1, 0], [2, 2, 1]).unwrap(),
        };
        let (prepared, key, _) = prepared(BackingRegion::Image(image_region));
        let image_key = ReplacementImageKey {
            backing: key.backing,
            representation: key.source,
        };
        let mut images = ReplacementImageStateOwner::new(VulkanDeviceEpochId::new(1));
        images
            .register(
                image_key,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let state = images
            .prepare_operation(
                prepared.transaction(),
                prepared.index(),
                3,
                [ReplacementImageUse {
                    image: image_key,
                    required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                    use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                }],
            )
            .unwrap();
        struct ImageResolver {
            backing: BackingId,
            image: reims_vgpu_protocol::RepresentationId,
            endpoint: reims_vgpu_protocol::RepresentationId,
            layout: std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>,
        }
        impl ReplacementBufferResolver for ImageResolver {
            fn resolve_buffer(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<NativeBufferTarget> {
                (backing == self.backing && representation == self.endpoint).then_some(
                    NativeBufferTarget {
                        buffer: vk::Buffer::from_raw(22),
                        base_offset: 100,
                        accessible_size: 64,
                        size: 64,
                        usage: vk::BufferUsageFlags::TRANSFER_DST,
                    },
                )
            }

            fn resolve_linear_texture_layout(
                &self,
                backing: BackingId,
                representation: reims_vgpu_protocol::RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                (backing == self.backing && representation == self.endpoint)
                    .then(|| std::sync::Arc::clone(&self.layout))
            }
        }
        impl ReplacementImageResolver for ImageResolver {
            fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
                (image.backing == self.backing && image.representation == self.image).then_some(
                    NativeImageTarget {
                        image: vk::Image::from_raw(33),
                        view: vk::ImageView::from_raw(34),
                        image_type: vk::ImageType::TYPE_2D,
                        full_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
                        extent: vk::Extent3D {
                            width: 4,
                            height: 4,
                            depth: 1,
                        },
                        samples: vk::SampleCountFlags::TYPE_1,
                    },
                )
            }
        }
        let resolver = ImageResolver {
            backing: key.backing,
            image: key.source,
            endpoint: key.destination,
            layout: std::sync::Arc::new(reims_vgpu_protocol::LinearTextureDescriptor {
                allocation_size: 64,
                mipmap_level_count: 1,
                bytes_per_slice: 64,
                slice_count: 1,
                bytes_per_element: 1,
                row_stride: 8,
                width: 4,
                height: 4,
                depth: 1,
                levels: vec![reims_vgpu_protocol::TextureLevelLayout {
                    size: 32,
                    row_stride: 8,
                    width: 4,
                    height: 4,
                    depth: 1,
                    ..Default::default()
                }],
                ..Default::default()
            }),
        };
        let program = ReplacementResourceStateProgram::resolve_with_image_state(
            &prepared,
            Some(&state),
            &resolver,
        )
        .unwrap();
        assert!(program.image_state().is_some());
        assert!(matches!(
            program.native_transfers(),
            [NativeResourceStateTransfer::Image(commands)]
                if matches!(commands.as_ref(), [NativeImageBlitCommand::ImageToBuffer(copy)]
                if copy.buffer == vk::Buffer::from_raw(22)
                    && copy.image == vk::Image::from_raw(33)
                    && copy.buffer_offset == 109
                    && copy.buffer_row_length == 8
                    && copy.buffer_image_height == 4
                    && copy.image_offset == [1, 1, 0]
                    && copy.extent == [2, 2, 1])
        ));
    }

    #[test]
    fn image_landing_copies_only_declared_texel_rows_and_preserves_padding() {
        let descriptor = reims_vgpu_protocol::LinearTextureDescriptor {
            allocation_size: 64,
            mipmap_level_count: 1,
            bytes_per_slice: 64,
            slice_count: 1,
            bytes_per_element: 1,
            row_stride: 8,
            width: 4,
            height: 4,
            depth: 1,
            levels: vec![reims_vgpu_protocol::TextureLevelLayout {
                size: 32,
                row_stride: 8,
                width: 4,
                height: 4,
                depth: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let region = reims_vgpu_core::ImageRegion {
            aspect: reims_vgpu_core::ImageAspect::Color,
            mip: 0,
            layer: 0,
            texels: reims_vgpu_core::TexelBox::new([1, 1, 0], [2, 2, 1]).unwrap(),
        };
        assert!(image_region_fits_layout(&descriptor, region, 64));
        let staged = (0u8..64).collect::<Vec<_>>();
        let stores = image_guest_stores(&staged, &descriptor, region).unwrap();
        assert_eq!(stores.len(), 2);
        assert_eq!(stores[0].offset, 9);
        assert_eq!(stores[0].bytes, [9, 10]);
        assert_eq!(stores[1].offset, 17);
        assert_eq!(stores[1].bytes, [17, 18]);
    }

    /// Two textures over one allocation that read its bytes differently are
    /// copied through those bytes, in each one's own layout.
    ///
    /// This is the shape a guest produces by declaring two textures over one
    /// buffer at different pixel formats: 32 KiB is 64x64 at eight bytes a
    /// texel and 128x64 at four, one set of bytes and two texel counts.
    /// `vkCmdCopyImage` between them is undefined and no view reinterprets
    /// one as the other, so the recorded program has to leave the texel
    /// domain -- out to the bytes in the source's layout, an ordering
    /// command, then back in to the destination's texels in its own.
    #[test]
    fn two_textures_reading_one_allocation_differently_copy_through_its_bytes() {
        use crate::replacement_image_state::{
            ReplacementImageSharing, ReplacementImageState, ReplacementImageStateOwner,
            ReplacementImageUse,
        };

        let backing = BackingId::new(1);
        let source_key = ReplacementImageKey {
            backing,
            representation: RepresentationId::new(2),
        };
        let destination_key = ReplacementImageKey {
            backing,
            representation: RepresentationId::new(3),
        };
        let mut owner = ReplacementImageStateOwner::new(VulkanDeviceEpochId::new(1));
        for key in [source_key, destination_key] {
            owner
                .register(
                    key,
                    ReplacementImageState {
                        layout: vk::ImageLayout::UNDEFINED,
                        sharing: ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let state = owner
            .prepare(
                TransactionId::new(1),
                0,
                [
                    ReplacementImageUse {
                        image: source_key,
                        required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    },
                    ReplacementImageUse {
                        image: destination_key,
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    },
                ],
            )
            .unwrap();

        let layout = |bytes_per_element: u8, width: u32, pixel_format: u16| {
            std::sync::Arc::new(reims_vgpu_protocol::LinearTextureDescriptor {
                declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                    texture_type: reims_vgpu_protocol::TextureType::D2,
                    framebuffer_only: false,
                    is_drawable: false,
                    write_swizzle_enabled: None,
                    allow_gpu_optimized_contents: false,
                    usage: 0,
                    pixel_format,
                    width,
                    height: 64,
                    depth: 1,
                    mipmap_level_count: 1,
                    sample_count: 1,
                    array_length: 1,
                    resource_options: 0,
                    protection_options: 0,
                    swizzle: None,
                }),
                allocation_size: 32768,
                mipmap_level_count: 1,
                bytes_per_slice: 32768,
                slice_count: 1,
                bytes_per_element,
                row_stride: width * u32::from(bytes_per_element),
                width,
                height: 64,
                depth: 1,
                levels: vec![reims_vgpu_protocol::TextureLevelLayout {
                    size: 32768,
                    row_stride: u64::from(width) * u64::from(bytes_per_element),
                    width,
                    height: 64,
                    depth: 1,
                    ..Default::default()
                }],
                ..Default::default()
            })
        };
        let image = |handle: u64, pixel_format: u16, width: u32| {
            crate::replacement_image_transition::NativeImageTarget {
                image: vk::Image::from_raw(handle),
                view: vk::ImageView::null(),
                image_type: vk::ImageType::TYPE_2D,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                pixel_format,
                extent: vk::Extent3D {
                    width,
                    height: 64,
                    depth: 1,
                },
                samples: vk::SampleCountFlags::TYPE_1,
            }
        };

        struct Alias {
            source: std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>,
            destination: std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>,
            bytes: Option<NativeBufferTarget>,
        }
        impl ReplacementBufferResolver for Alias {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                None
            }

            fn resolve_linear_texture_layout(
                &self,
                _backing: BackingId,
                representation: RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                Some(if representation == RepresentationId::new(2) {
                    std::sync::Arc::clone(&self.source)
                } else {
                    std::sync::Arc::clone(&self.destination)
                })
            }

            fn resolve_alias_transfer_bytes(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                self.bytes
            }
        }

        let transfer = TransferKey {
            backing,
            region: BackingRegion::Linear(LinearRange::new(0, 32768).unwrap()),
            version: reims_vgpu_protocol::ContentVersion::new(19),
            source: RepresentationId::new(2),
            destination: RepresentationId::new(3),
        };
        let bytes = NativeBufferTarget {
            buffer: vk::Buffer::from_raw(77),
            base_offset: 0,
            accessible_size: 32768,
            size: 32768,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        };
        let resolver = Alias {
            source: layout(8, 64, 115),
            destination: layout(4, 128, 80),
            bytes: Some(bytes),
        };
        let recorded = resolve_image_image_transfer(
            transfer,
            image(1, 115, 64),
            source_key,
            image(2, 80, 128),
            destination_key,
            &state,
            &resolver,
        )
        .expect("two readings of one allocation are copied through its bytes");
        let NativeResourceStateTransfer::Image(commands) = recorded else {
            panic!("a round trip through the bytes records image commands");
        };
        assert_eq!(commands.len(), 3);
        assert!(matches!(
            commands[0],
            NativeImageBlitCommand::ImageToBuffer(copy)
                if copy.image == vk::Image::from_raw(1)
                    && copy.buffer == vk::Buffer::from_raw(77)
                    && copy.image_layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    // 64 texels a row in the source's own eight-byte texels.
                    && copy.buffer_row_length == 64
                    && copy.extent == [64, 64, 1]
        ));
        assert!(matches!(
            commands[1],
            NativeImageBlitCommand::TransferBytesReady {
                buffer,
                offset: 0,
                size: 32768,
            } if buffer == vk::Buffer::from_raw(77)
        ));
        assert!(matches!(
            commands[2],
            NativeImageBlitCommand::BufferToImage(copy)
                if copy.image == vk::Image::from_raw(2)
                    && copy.buffer == vk::Buffer::from_raw(77)
                    && copy.image_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    // The same bytes, read as 128 four-byte texels a row.
                    && copy.buffer_row_length == 128
                    && copy.extent == [128, 64, 1]
        ));

        // The same pair, with the region named in the source's texels rather
        // than in bytes. The box is the source's whole 128x64 image; the
        // destination's own layout reads those same 32 KiB as 64x64, and the
        // recorded copy has to say so rather than repeat the box.
        let resolver = Alias {
            source: layout(4, 128, 80),
            destination: layout(8, 64, 115),
            bytes: Some(bytes),
        };
        let recorded = resolve_image_image_transfer(
            TransferKey {
                region: BackingRegion::Image(reims_vgpu_core::ImageRegion {
                    aspect: reims_vgpu_core::ImageAspect::Color,
                    mip: 0,
                    layer: 0,
                    texels: reims_vgpu_core::TexelBox::new([0, 0, 0], [128, 64, 1]).unwrap(),
                }),
                ..transfer
            },
            image(1, 80, 128),
            source_key,
            image(2, 115, 64),
            destination_key,
            &state,
            &resolver,
        )
        .expect("a box in the source's texels reaches the destination through the bytes");
        let NativeResourceStateTransfer::Image(commands) = recorded else {
            panic!("a round trip through the bytes records image commands");
        };
        assert_eq!(commands.len(), 3);
        assert!(matches!(
            commands[0],
            NativeImageBlitCommand::ImageToBuffer(copy) if copy.extent == [128, 64, 1]
        ));
        assert!(matches!(
            commands[2],
            NativeImageBlitCommand::BufferToImage(copy) if copy.extent == [64, 64, 1]
        ));

        // The buffer belongs to whichever of the two disagreed with the
        // allocation's first-declared layout, which is as often the source as
        // the destination -- so a round trip finds it on either.
        struct SourceOwnsBytes(Alias, NativeBufferTarget);
        impl ReplacementBufferResolver for SourceOwnsBytes {
            fn resolve_buffer(
                &self,
                backing: BackingId,
                representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                self.0.resolve_buffer(backing, representation)
            }

            fn resolve_linear_texture_layout(
                &self,
                backing: BackingId,
                representation: RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                self.0
                    .resolve_linear_texture_layout(backing, representation)
            }

            fn resolve_alias_transfer_bytes(
                &self,
                _backing: BackingId,
                representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                (representation == RepresentationId::new(2)).then_some(self.1)
            }
        }
        let recorded = resolve_image_image_transfer(
            transfer,
            image(1, 115, 64),
            source_key,
            image(2, 80, 128),
            destination_key,
            &state,
            &SourceOwnsBytes(
                Alias {
                    source: layout(8, 64, 115),
                    destination: layout(4, 128, 80),
                    bytes: None,
                },
                bytes,
            ),
        )
        .expect("the round trip finds the bytes on whichever endpoint owns them");
        let NativeResourceStateTransfer::Image(commands) = recorded else {
            panic!("a round trip through the bytes records image commands");
        };
        assert_eq!(commands.len(), 3);

        // Without the bytes to round trip through, the refusal names both the
        // term that ruled out a direct copy and the part that was missing.
        let resolver = Alias {
            source: layout(8, 64, 115),
            destination: layout(4, 128, 80),
            bytes: None,
        };
        assert!(matches!(
            resolve_image_image_transfer(
                transfer,
                image(1, 115, 64),
                source_key,
                image(2, 80, 128),
                destination_key,
                &state,
                &resolver,
            ),
            Err(ResourceStateTransferRecordError::ImageToImageUnsupported {
                refusal: ImageToImageRefusal::AliasBytesUnavailable {
                    disagreement: ImageEndpointDisagreement::PixelFormat {
                        source: 115,
                        destination: 80,
                    },
                    missing: AliasRoundTripTerm::TransferBytes,
                },
                ..
            })
        ));
    }

    /// A buffer/image copy is addressed in the image's own layout.
    ///
    /// The shared transfer endpoint carries a layout too, and it is whichever
    /// texture was declared over the allocation first. Reading it instead
    /// addresses every later texture over that allocation in the first one's
    /// coordinates -- right row stride, wrong texel width -- which lands the
    /// bytes in the wrong place and reports nothing.
    #[test]
    fn a_buffer_image_copy_reads_the_image_endpoints_own_layout() {
        use crate::replacement_image_state::{
            ReplacementImageSharing, ReplacementImageState, ReplacementImageStateOwner,
            ReplacementImageUse,
        };

        let backing = BackingId::new(1);
        let image_key = ReplacementImageKey {
            backing,
            representation: RepresentationId::new(3),
        };
        let mut owner = ReplacementImageStateOwner::new(VulkanDeviceEpochId::new(1));
        owner
            .register(
                image_key,
                ReplacementImageState {
                    layout: vk::ImageLayout::UNDEFINED,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let state = owner
            .prepare(
                TransactionId::new(1),
                0,
                [ReplacementImageUse {
                    image: image_key,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                }],
            )
            .unwrap();

        // The endpoint's layout reads the allocation as eight-byte texels and
        // the image's own reads it as four, over the same 32 KiB and the same
        // 512-byte rows.
        struct Layouts;
        impl ReplacementBufferResolver for Layouts {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                None
            }

            fn resolve_linear_texture_layout(
                &self,
                _backing: BackingId,
                representation: RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                let (bytes_per_element, width, pixel_format) =
                    if representation == RepresentationId::new(3) {
                        (4u8, 128u32, 80u16)
                    } else {
                        (8, 64, 115)
                    };
                Some(std::sync::Arc::new(
                    reims_vgpu_protocol::LinearTextureDescriptor {
                        declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                            texture_type: reims_vgpu_protocol::TextureType::D2,
                            framebuffer_only: false,
                            is_drawable: false,
                            write_swizzle_enabled: None,
                            allow_gpu_optimized_contents: false,
                            usage: 0,
                            pixel_format,
                            width,
                            height: 64,
                            depth: 1,
                            mipmap_level_count: 1,
                            sample_count: 1,
                            array_length: 1,
                            resource_options: 0,
                            protection_options: 0,
                            swizzle: None,
                        }),
                        allocation_size: 32768,
                        mipmap_level_count: 1,
                        bytes_per_slice: 32768,
                        slice_count: 1,
                        bytes_per_element,
                        row_stride: 512,
                        width,
                        height: 64,
                        depth: 1,
                        levels: vec![reims_vgpu_protocol::TextureLevelLayout {
                            size: 32768,
                            row_stride: 512,
                            width,
                            height: 64,
                            depth: 1,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ))
            }
        }

        let recorded = resolve_buffer_image_transfer(
            TransferKey {
                backing,
                region: BackingRegion::Linear(LinearRange::new(0, 32768).unwrap()),
                version: reims_vgpu_protocol::ContentVersion::new(19),
                source: RepresentationId::new(2),
                destination: RepresentationId::new(3),
            },
            NativeBufferTarget {
                buffer: vk::Buffer::from_raw(77),
                base_offset: 0,
                accessible_size: 32768,
                size: 32768,
                usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            },
            crate::replacement_image_transition::NativeImageTarget {
                image: vk::Image::from_raw(2),
                view: vk::ImageView::null(),
                image_type: vk::ImageType::TYPE_2D,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                pixel_format: 80,
                extent: vk::Extent3D {
                    width: 128,
                    height: 64,
                    depth: 1,
                },
                samples: vk::SampleCountFlags::TYPE_1,
            },
            image_key,
            true,
            &state,
            &Layouts,
        )
        .expect("a whole-allocation upload into a declared texture is recordable");
        let NativeResourceStateTransfer::Image(commands) = recorded else {
            panic!("a buffer/image copy records image commands");
        };
        assert!(matches!(
            commands.as_ref(),
            [NativeImageBlitCommand::BufferToImage(copy)]
                // 512 bytes a row is 128 texels in the image's own four-byte
                // texels and 64 in the endpoint's eight-byte ones.
                if copy.buffer_row_length == 128 && copy.extent == [128, 64, 1]
        ));
    }

    /// Whether an EXEC needs a prepared image-state batch is decided by the
    /// same classification that records the transfer, so a pair the recorder
    /// resolves as buffer/image is a pair the classifier qualifies --- however
    /// little the bytes endpoint declares about itself.
    ///
    /// A classifier that additionally required the bytes endpoint to carry a
    /// linear texture layout withheld the batch from a pair the recorder then
    /// declined for wanting it. That decline is waitable, so the channel head
    /// retried it for the life of the boot.
    #[test]
    fn a_transfer_the_recorder_reaches_an_image_through_is_a_transfer_that_needs_state() {
        struct Endpoints;
        impl ReplacementBufferResolver for Endpoints {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                (representation == RepresentationId::new(2)).then_some(NativeBufferTarget {
                    buffer: vk::Buffer::from_raw(77),
                    base_offset: 0,
                    accessible_size: 32768,
                    size: 32768,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                })
            }
        }
        impl crate::replacement_image_transition::ReplacementImageResolver for Endpoints {
            fn resolve_image(
                &self,
                image: ReplacementImageKey,
            ) -> Option<crate::replacement_image_transition::NativeImageTarget> {
                (image.representation == RepresentationId::new(3)).then_some(
                    crate::replacement_image_transition::NativeImageTarget {
                        image: vk::Image::from_raw(2),
                        view: vk::ImageView::null(),
                        image_type: vk::ImageType::TYPE_2D,
                        full_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        usage: vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
                        pixel_format: 80,
                        extent: vk::Extent3D {
                            width: 128,
                            height: 64,
                            depth: 1,
                        },
                        samples: vk::SampleCountFlags::TYPE_1,
                    },
                )
            }
        }

        let backing = BackingId::new(1);
        let transfer = TransferKey {
            backing,
            region: BackingRegion::Whole,
            version: reims_vgpu_protocol::ContentVersion::new(3),
            source: RepresentationId::new(2),
            destination: RepresentationId::new(3),
        };
        assert!(transfer_requires_image_state(transfer, &Endpoints));
        assert!(matches!(
            resolve_transfer(&[backing], transfer, None, &Endpoints),
            Err(ResourceStateTransferRecordError::ImageTransferRequiresState(found))
                if found == transfer
        ));
        // And the pair with no image on either end is a plain buffer copy,
        // which must not be made to wait for a batch it never reads.
        assert!(!transfer_requires_image_state(
            TransferKey {
                destination: RepresentationId::new(2),
                ..transfer
            },
            &Endpoints
        ));
    }

    /// Two images that agree on how bytes are read copy a byte range
    /// directly, in the texels that range covers.
    ///
    /// A byte range names no texels on its own, and the resolver used to stop
    /// there -- so a guest that wrote part of one texture over an allocation
    /// and read it through another declared the same way lost the copy, on a
    /// pair where nothing was ambiguous. One layout answers for both, because
    /// agreeing is exactly what reaching this point means.
    #[test]
    fn a_byte_range_between_agreeing_images_copies_the_texels_it_covers() {
        use crate::replacement_image_state::{
            ReplacementImageSharing, ReplacementImageState, ReplacementImageStateOwner,
            ReplacementImageUse,
        };

        let backing = BackingId::new(1);
        let source_key = ReplacementImageKey {
            backing,
            representation: RepresentationId::new(2),
        };
        let destination_key = ReplacementImageKey {
            backing,
            representation: RepresentationId::new(3),
        };
        let mut owner = ReplacementImageStateOwner::new(VulkanDeviceEpochId::new(1));
        for key in [source_key, destination_key] {
            owner
                .register(
                    key,
                    ReplacementImageState {
                        layout: vk::ImageLayout::UNDEFINED,
                        sharing: ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let state = owner
            .prepare(
                TransactionId::new(1),
                0,
                [
                    ReplacementImageUse {
                        image: source_key,
                        required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    },
                    ReplacementImageUse {
                        image: destination_key,
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    },
                ],
            )
            .unwrap();

        struct OneLayout;
        impl ReplacementBufferResolver for OneLayout {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                None
            }

            fn resolve_linear_texture_layout(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<std::sync::Arc<reims_vgpu_protocol::LinearTextureDescriptor>> {
                Some(std::sync::Arc::new(
                    reims_vgpu_protocol::LinearTextureDescriptor {
                        declaration: Some(reims_vgpu_protocol::TextureDeclaration {
                            texture_type: reims_vgpu_protocol::TextureType::D2,
                            framebuffer_only: false,
                            is_drawable: false,
                            write_swizzle_enabled: None,
                            allow_gpu_optimized_contents: false,
                            usage: 0,
                            pixel_format: 80,
                            width: 64,
                            height: 64,
                            depth: 1,
                            mipmap_level_count: 1,
                            sample_count: 1,
                            array_length: 1,
                            resource_options: 0,
                            protection_options: 0,
                            swizzle: None,
                        }),
                        allocation_size: 16384,
                        mipmap_level_count: 1,
                        bytes_per_slice: 16384,
                        slice_count: 1,
                        bytes_per_element: 4,
                        row_stride: 256,
                        width: 64,
                        height: 64,
                        depth: 1,
                        levels: vec![reims_vgpu_protocol::TextureLevelLayout {
                            size: 16384,
                            row_stride: 256,
                            width: 64,
                            height: 64,
                            depth: 1,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ))
            }
        }

        let image = |handle: u64| crate::replacement_image_transition::NativeImageTarget {
            image: vk::Image::from_raw(handle),
            view: vk::ImageView::null(),
            image_type: vk::ImageType::TYPE_2D,
            full_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            pixel_format: 80,
            extent: vk::Extent3D {
                width: 64,
                height: 64,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        };
        // Rows 4 through 7 of a 64x64 four-byte-texel level.
        let recorded = resolve_image_image_transfer(
            TransferKey {
                backing,
                region: BackingRegion::Linear(LinearRange::new(1024, 1024).unwrap()),
                version: reims_vgpu_protocol::ContentVersion::new(33),
                source: RepresentationId::new(2),
                destination: RepresentationId::new(3),
            },
            image(1),
            source_key,
            image(2),
            destination_key,
            &state,
            &OneLayout,
        )
        .expect("a byte range between two identically declared images is a direct copy");
        let NativeResourceStateTransfer::Image(commands) = recorded else {
            panic!("an image-to-image copy records image commands");
        };
        assert!(matches!(
            commands.as_ref(),
            [NativeImageBlitCommand::ImageToImage(copy)]
                if copy.source == vk::Image::from_raw(1)
                    && copy.destination == vk::Image::from_raw(2)
                    && copy.source_offset == [0, 4, 0]
                    && copy.destination_offset == [0, 4, 0]
                    && copy.extent == [64, 4, 1]
        ));
    }

    /// Each condition that refuses an image-to-image copy names itself.
    ///
    /// The four reach one refusal family and call for different repairs -- a
    /// planner that paired siblings the guest never aliased, an authority that
    /// planned a copy between one image and itself, a byte range that belongs
    /// on the buffer/image path, and a source declaring no aspect to copy.
    /// A boot that reports only the family says which of those it hit by luck,
    /// and each holds its submission head until someone knows.
    #[test]
    fn every_unrecordable_image_pair_says_which_term_it_could_not_record() {
        use crate::replacement_image_state::{
            ReplacementImageSharing, ReplacementImageState, ReplacementImageStateOwner,
            ReplacementImageUse,
        };

        let source_key = ReplacementImageKey {
            backing: BackingId::new(1),
            representation: RepresentationId::new(2),
        };
        let destination_key = ReplacementImageKey {
            backing: BackingId::new(1),
            representation: RepresentationId::new(3),
        };
        let mut owner = ReplacementImageStateOwner::new(VulkanDeviceEpochId::new(1));
        for key in [source_key, destination_key] {
            owner
                .register(
                    key,
                    ReplacementImageState {
                        layout: vk::ImageLayout::GENERAL,
                        sharing: ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        let state = owner
            .prepare(
                TransactionId::new(1),
                0,
                [
                    ReplacementImageUse {
                        image: source_key,
                        required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    },
                    ReplacementImageUse {
                        image: destination_key,
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    },
                ],
            )
            .unwrap();

        let target = |image: u64, pixel_format: u16, aspect: vk::ImageAspectFlags| {
            crate::replacement_image_transition::NativeImageTarget {
                image: vk::Image::from_raw(image),
                view: vk::ImageView::null(),
                image_type: vk::ImageType::TYPE_2D,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                pixel_format,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                samples: vk::SampleCountFlags::TYPE_1,
            }
        };
        let key = |region| TransferKey {
            backing: BackingId::new(1),
            region,
            version: reims_vgpu_protocol::ContentVersion::new(1),
            source: RepresentationId::new(2),
            destination: RepresentationId::new(3),
        };
        struct NoAliasBytes;
        impl ReplacementBufferResolver for NoAliasBytes {
            fn resolve_buffer(
                &self,
                _backing: BackingId,
                _representation: RepresentationId,
            ) -> Option<NativeBufferTarget> {
                None
            }
        }
        let refusal =
            |transfer: TransferKey, source, destination| match resolve_image_image_transfer(
                transfer,
                source,
                source_key,
                destination,
                destination_key,
                &state,
                &NoAliasBytes,
            ) {
                Err(ResourceStateTransferRecordError::ImageToImageUnsupported {
                    refusal, ..
                }) => refusal,
                other => panic!("expected an unrecordable pair, got {other:?}"),
            };

        let color = vk::ImageAspectFlags::COLOR;
        // A format pair and an extent pair fail at the same line; only the
        // recorded endpoints say which term the planner got wrong.
        assert_eq!(
            refusal(
                key(BackingRegion::Whole),
                target(1, 7, color),
                target(2, 9, color),
            ),
            ImageToImageRefusal::AliasBytesUnavailable {
                disagreement: ImageEndpointDisagreement::PixelFormat {
                    source: 7,
                    destination: 9,
                },
                missing: AliasRoundTripTerm::SourceLayout,
            }
        );

        assert_eq!(
            refusal(
                key(BackingRegion::Whole),
                target(1, 7, color),
                target(1, 7, color),
            ),
            ImageToImageRefusal::SameImage
        );
        // A byte range is translatable through either endpoint's layout, and
        // these two agree -- so the range is only untranslatable when the
        // backing declares no layout at all, which is what this resolver is.
        assert_eq!(
            refusal(
                key(BackingRegion::Linear(LinearRange::new(0, 16).unwrap())),
                target(1, 7, color),
                target(2, 7, color),
            ),
            ImageToImageRefusal::LinearRegion
        );
        let empty = vk::ImageAspectFlags::empty();
        assert_eq!(
            refusal(
                key(BackingRegion::Whole),
                target(1, 7, empty),
                target(2, 7, empty),
            ),
            ImageToImageRefusal::NoCopyableAspect { aspect_mask: empty }
        );
    }

    /// The endpoint walk reports every term it compares, and agrees otherwise.
    ///
    /// Only the first term has a caller that reaches it in the boot this was
    /// written for; a term that is compared but can never be reported is a
    /// silent loss the moment a guest declares the pair that trips it.
    #[test]
    fn each_term_two_image_endpoints_must_agree_on_reports_itself() {
        use crate::replacement_image_transition::NativeImageTarget;

        let base = NativeImageTarget {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            image_type: vk::ImageType::TYPE_2D,
            full_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 1,
                level_count: 2,
                base_array_layer: 3,
                layer_count: 4,
            },
            usage: vk::ImageUsageFlags::TRANSFER_SRC,
            pixel_format: 7,
            extent: vk::Extent3D {
                width: 4,
                height: 5,
                depth: 1,
            },
            samples: vk::SampleCountFlags::TYPE_1,
        };
        assert_eq!(image_endpoints_disagree(&base, &base), None);

        let disagree = |edit: fn(&mut NativeImageTarget)| {
            let mut destination = base;
            edit(&mut destination);
            image_endpoints_disagree(&base, &destination)
                .expect("an edited endpoint disagrees with the one it was edited from")
        };
        assert_eq!(
            disagree(|target| target.pixel_format = 9),
            ImageEndpointDisagreement::PixelFormat {
                source: 7,
                destination: 9,
            }
        );
        assert_eq!(
            disagree(|target| target.extent.height = 6),
            ImageEndpointDisagreement::Extent {
                source: [4, 5, 1],
                destination: [4, 6, 1],
            }
        );
        assert_eq!(
            disagree(|target| target.samples = vk::SampleCountFlags::TYPE_4),
            ImageEndpointDisagreement::Samples {
                source: vk::SampleCountFlags::TYPE_1,
                destination: vk::SampleCountFlags::TYPE_4,
            }
        );
        for (term, edit, source, destination) in [
            (
                SubresourceRangeTerm::AspectMask,
                (|target: &mut NativeImageTarget| {
                    target.full_range.aspect_mask = vk::ImageAspectFlags::DEPTH
                }) as fn(&mut NativeImageTarget),
                vk::ImageAspectFlags::COLOR.as_raw(),
                vk::ImageAspectFlags::DEPTH.as_raw(),
            ),
            (
                SubresourceRangeTerm::BaseMipLevel,
                |target| target.full_range.base_mip_level = 9,
                1,
                9,
            ),
            (
                SubresourceRangeTerm::LevelCount,
                |target| target.full_range.level_count = 9,
                2,
                9,
            ),
            (
                SubresourceRangeTerm::BaseArrayLayer,
                |target| target.full_range.base_array_layer = 9,
                3,
                9,
            ),
            (
                SubresourceRangeTerm::LayerCount,
                |target| target.full_range.layer_count = 9,
                4,
                9,
            ),
        ] {
            assert_eq!(
                disagree(edit),
                ImageEndpointDisagreement::SubresourceRange {
                    term,
                    source,
                    destination,
                }
            );
        }
    }
}
