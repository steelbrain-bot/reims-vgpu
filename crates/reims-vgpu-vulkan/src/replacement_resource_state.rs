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
    replacement_image_blit::{NativeBufferImageCopy, NativeImageBlitCommand},
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
    ImageToImageUnsupported(TransferKey),
    ImageLayoutMissing(TransferKey),
    ImageLayoutUnsupported(TransferKey),
    ImageUsageMissing(TransferKey),
    MissingHostStaging(HostLandingKey),
    HostLandingImageLayoutMissing(HostLandingKey),
    HostLandingRangeOutOfBounds(HostLandingKey),
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
            completions: prepared.resource_completions().into(),
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
    if source_image.is_none() && destination_image.is_none() {
        return resolve_buffer_transfer(backings, transfer, resolver)
            .map(NativeResourceStateTransfer::Buffer);
    }
    let state = image_state
        .ok_or(ResourceStateTransferRecordError::ImageTransferRequiresState(transfer))?;
    match (
        source_buffer,
        source_image,
        destination_buffer,
        destination_image,
    ) {
        (Some(buffer), None, None, Some(image)) => resolve_buffer_image_transfer(
            transfer,
            buffer,
            image,
            destination_key,
            true,
            state,
            resolver,
        ),
        (None, Some(image), Some(buffer), None) => resolve_buffer_image_transfer(
            transfer, buffer, image, source_key, false, state, resolver,
        ),
        (None, Some(_), None, Some(_)) => Err(
            ResourceStateTransferRecordError::ImageToImageUnsupported(transfer),
        ),
        _ => Err(ResourceStateTransferRecordError::ImageEndpointMissing(
            transfer,
        )),
    }
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
    let layout_representation = if buffer_to_image {
        transfer.source
    } else {
        transfer.destination
    };
    let layout = resolver
        .resolve_linear_texture_layout(transfer.backing, layout_representation)
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
    let level = layout.level(region.mip).ok_or(
        ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
    )?;
    let bytes_per_element = u64::from(layout.bytes_per_element);
    if layout.compressed_layout
        || bytes_per_element == 0
        || level.row_stride % bytes_per_element != 0
    {
        return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
            transfer,
        ));
    }
    let row_length = u32::try_from(level.row_stride / bytes_per_element)
        .map_err(|_| ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer))?;
    let level_offset = layout.subresource_offset(region.layer, region.mip).ok_or(
        ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
    )?;
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
    .ok_or(ResourceStateTransferRecordError::ImageLayoutUnsupported(
        transfer,
    ))?;
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
            i32::try_from(region.texels.origin[0])
                .map_err(|_| ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer))?,
            i32::try_from(region.texels.origin[1])
                .map_err(|_| ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer))?,
            i32::try_from(region.texels.origin[2])
                .map_err(|_| ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer))?,
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

fn transfer_image_regions(
    layout: &reims_vgpu_protocol::LinearTextureDescriptor,
    transfer: TransferKey,
) -> Result<Vec<reims_vgpu_core::ImageRegion>, ResourceStateTransferRecordError> {
    if let BackingRegion::Image(region) = transfer.region {
        return Ok(vec![region]);
    }
    if matches!(transfer.region, BackingRegion::Whole) {
        let format = layout.declared_pixel_format().ok_or(
            ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
        )?;
        if reims_vgpu_core::pixel_format::format_has_depth_aspect(format)
            || reims_vgpu_core::pixel_format::format_has_stencil_aspect(format)
        {
            return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
                transfer,
            ));
        }
        let slices = layout.physical_slice_count().ok_or(
            ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
        )?;
        let mut regions = Vec::new();
        for layer in 0..slices {
            for mip in 0..layout.mipmap_level_count {
                let level = layout.level(mip).ok_or(
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
                )?;
                let texels = reims_vgpu_core::TexelBox::new(
                    [0, 0, 0],
                    [level.width, level.height, level.planes()],
                )
                .ok_or(
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
                )?;
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
    let slices = layout.physical_slice_count().ok_or(
        ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
    )?;
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
            regions.push(transfer_image_region(layout, region_transfer)?);
        }
    }
    Ok(regions)
}

fn transfer_image_region(
    layout: &reims_vgpu_protocol::LinearTextureDescriptor,
    transfer: TransferKey,
) -> Result<reims_vgpu_core::ImageRegion, ResourceStateTransferRecordError> {
    if let BackingRegion::Image(region) = transfer.region {
        return Ok(region);
    }
    let BackingRegion::Linear(range) = transfer.region else {
        return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
            transfer,
        ));
    };
    let pixel_format = layout.declared_pixel_format().ok_or(
        ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
    )?;
    if reims_vgpu_core::pixel_format::format_has_depth_aspect(pixel_format)
        || reims_vgpu_core::pixel_format::format_has_stencil_aspect(pixel_format)
        || layout.compressed_layout
        || layout.bytes_per_element == 0
    {
        return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
            transfer,
        ));
    }
    let bytes_per_element = u64::from(layout.bytes_per_element);
    let slices = layout.physical_slice_count().ok_or(
        ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
    )?;
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
            let relative = range.start() - base;
            let image_stride = level
                .row_stride
                .checked_mul(u64::from(level.height))
                .ok_or(ResourceStateTransferRecordError::RangeAddressOverflow(
                    transfer,
                ))?;
            if image_stride == 0 || level.row_stride == 0 {
                return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
                    transfer,
                ));
            }
            let z = relative / image_stride;
            let within_image = relative % image_stride;
            let y = within_image / level.row_stride;
            let x_bytes = within_image % level.row_stride;
            let length = range.end() - range.start();
            if x_bytes % bytes_per_element != 0 {
                return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
                    transfer,
                ));
            }
            let (width, height, depth) = if length <= level.row_stride - x_bytes {
                if length % bytes_per_element != 0 {
                    return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
                        transfer,
                    ));
                }
                (length / bytes_per_element, 1, 1)
            } else if x_bytes == 0 && length % level.row_stride == 0 {
                let rows = length / level.row_stride;
                let rows_left = u64::from(level.height).saturating_sub(y);
                if rows <= rows_left {
                    (u64::from(level.width), rows, 1)
                } else if y == 0
                    && level.row_stride == u64::from(level.width) * bytes_per_element
                    && rows % u64::from(level.height) == 0
                {
                    (
                        u64::from(level.width),
                        u64::from(level.height),
                        rows / u64::from(level.height),
                    )
                } else {
                    return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
                        transfer,
                    ));
                }
            } else {
                return Err(ResourceStateTransferRecordError::ImageLayoutUnsupported(
                    transfer,
                ));
            };
            let origin = [
                u32::try_from(x_bytes / bytes_per_element).map_err(|_| {
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer)
                })?,
                u32::try_from(y).map_err(|_| {
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer)
                })?,
                u32::try_from(z).map_err(|_| {
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer)
                })?,
            ];
            let extent = [
                u32::try_from(width).map_err(|_| {
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer)
                })?,
                u32::try_from(height).map_err(|_| {
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer)
                })?,
                u32::try_from(depth).map_err(|_| {
                    ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer)
                })?,
            ];
            let texels = reims_vgpu_core::TexelBox::new(origin, extent).ok_or(
                ResourceStateTransferRecordError::ImageLayoutUnsupported(transfer),
            )?;
            if texels.end[0] > level.width
                || texels.end[1] > level.height
                || texels.end[2] > level.planes()
            {
                return Err(ResourceStateTransferRecordError::RangeOutOfBounds(transfer));
            }
            return Ok(reims_vgpu_core::ImageRegion {
                aspect: reims_vgpu_core::ImageAspect::Color,
                mip,
                layer,
                texels,
            });
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
            &[ResolvedResourceCompletion::ValidityHostWrite {
                backing: key.backing,
                write: reims_vgpu_core::GpuWriteId::operation(SubmissionId::new(2), 0),
                representation: key.source,
            }]
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
                    ..Default::default()
                },
                reims_vgpu_protocol::TextureLevelLayout {
                    offset: 40,
                    size: 16,
                    row_stride: 8,
                    width: 2,
                    height: 2,
                    depth: 1,
                    ..Default::default()
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
}
