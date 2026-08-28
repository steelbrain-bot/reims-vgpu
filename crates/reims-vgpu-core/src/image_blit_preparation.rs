//! Content and native-lifetime preparation for image-bearing blits.

use crate::{
    pixel_format::{buffer_image_blit_aspect, BlitAspect},
    AccessIntent, AccessMode, AccessScope, AccessTarget, BackingRegion, BackingView,
    DirectReplayNativeOwner, GpuWriteBatchError, GpuWriteRequest, GpuWriteReservation, ImageOwner,
    ManagedBackingError, ManagedBackingProgress, PreparedNativeSubmission, ReplayAcceptance,
    ReplayAcceptanceError, RepresentationUse, ResolvedBlit, ResolvedReplayCompletion,
    ResolvedResourceCompletion, ResolvedTextureEndpoint, ResourceLifecycleOwner,
    ResourceUseBatchError, StageScope, TransactionRuntime, ViewRepresentation,
};
use reims_vgpu_protocol::{
    BackingId, HazardDomainId, RepresentationId, SubmissionId, TransactionId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub struct PreparedImageBlit {
    transaction: TransactionId,
    write: crate::GpuWriteId,
    operation: ResolvedBlit,
    representations: Box<[ViewRepresentation]>,
    uses: Box<[RepresentationUse]>,
    writes: Box<[GpuWriteReservation]>,
}

impl PreparedImageBlit {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn submission(&self) -> SubmissionId {
        self.write.submission()
    }

    pub const fn write(&self) -> crate::GpuWriteId {
        self.write
    }

    pub const fn operation(&self) -> &ResolvedBlit {
        &self.operation
    }

    pub fn into_operation(self) -> ResolvedBlit {
        self.operation
    }

    pub const fn representations(&self) -> &[ViewRepresentation] {
        &self.representations
    }

    pub const fn writes(&self) -> &[GpuWriteReservation] {
        &self.writes
    }

    pub const fn uses(&self) -> &[RepresentationUse] {
        &self.uses
    }

    pub fn backings(&self) -> Box<[BackingId]> {
        self.uses
            .iter()
            .map(|use_| use_.backing)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn resource_completions(&self) -> Box<[ResolvedResourceCompletion]> {
        self.writes
            .iter()
            .map(|write| ResolvedResourceCompletion::GpuWrite {
                backing: write.backing,
                write: write.write,
                representation: write.representation,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageBlitPreparationError {
    BufferOnlyVariant,
    CoordinateOverflow,
    EmptyImageRegion,
    Source {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    Destination {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
    WriteRollback {
        admission: ResourceUseBatchError,
        cancellation: GpuWriteBatchError,
    },
}

#[derive(Debug)]
pub struct ImageBlitPreparationFailure {
    pub reason: ImageBlitPreparationError,
    pub operation: ResolvedBlit,
    pub live_writes: Box<[GpuWriteReservation]>,
}

pub fn prepare_image_blit<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    submission: SubmissionId,
    operation: ResolvedBlit,
) -> Result<PreparedImageBlit, Box<ImageBlitPreparationFailure>> {
    prepare_image_blit_with_write(resources, transaction, submission.into(), operation)
}

pub fn prepare_image_blit_with_write<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    write: crate::GpuWriteId,
    operation: ResolvedBlit,
) -> Result<PreparedImageBlit, Box<ImageBlitPreparationFailure>> {
    let mut reads = BTreeMap::<(BackingId, BackingView), Vec<BackingRegion>>::new();
    let mut writes = BTreeMap::<(BackingId, BackingView), Vec<BackingRegion>>::new();
    collect_regions(&operation, &mut reads, &mut writes).map_err(|reason| {
        Box::new(ImageBlitPreparationFailure {
            reason,
            operation: operation.clone(),
            live_writes: Box::new([]),
        })
    })?;
    for regions in reads.values_mut().chain(writes.values_mut()) {
        regions.sort();
        regions.dedup();
    }

    let mut representations = BTreeMap::<(BackingId, BackingView), RepresentationId>::new();
    for (&(backing, view), regions) in &reads {
        let snapshot = resources
            .snapshot_content(backing, regions)
            .map_err(|reason| {
                Box::new(ImageBlitPreparationFailure {
                    reason: ImageBlitPreparationError::Source { backing, reason },
                    operation: operation.clone(),
                    live_writes: Box::new([]),
                })
            })?;
        let representation = resources
            .view_representation_for_snapshot(backing, view, &snapshot)
            .map_err(|reason| {
                Box::new(ImageBlitPreparationFailure {
                    reason: ImageBlitPreparationError::Source { backing, reason },
                    operation: operation.clone(),
                    live_writes: Box::new([]),
                })
            })?;
        representations.insert((backing, view), representation);
    }
    for &(backing, view) in writes.keys() {
        if representations.contains_key(&(backing, view)) {
            continue;
        }
        let representation = resources
            .view_representation(backing, view)
            .map_err(|reason| {
                Box::new(ImageBlitPreparationFailure {
                    reason: ImageBlitPreparationError::Destination { backing, reason },
                    operation: operation.clone(),
                    live_writes: Box::new([]),
                })
            })?;
        representations.insert((backing, view), representation);
    }

    let write_reservations = resources
        .plan_gpu_writes(
            write,
            writes
                .iter()
                .map(|(&key, regions)| GpuWriteRequest {
                    backing: key.0,
                    representation: representations[&key],
                    regions: regions.clone().into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .map_err(|reason| {
            Box::new(ImageBlitPreparationFailure {
                reason: ImageBlitPreparationError::Writes(reason),
                operation: operation.clone(),
                live_writes: Box::new([]),
            })
        })?;
    // One use per backing, naming every view of it this blit touches: a
    // backing read through its buffer view and written through its image view
    // is one backing holding two native objects, and both have to survive the
    // transaction.
    let mut uses = BTreeMap::<BackingId, Vec<RepresentationId>>::new();
    for (&(backing, _), &representation) in &representations {
        uses.entry(backing).or_default().push(representation);
    }
    let uses = uses
        .into_iter()
        .map(|(backing, mut representations)| {
            representations.sort();
            representations.dedup();
            RepresentationUse {
                backing,
                representations: representations.into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    if let Err(admission) = resources.accept_uses(transaction, &uses) {
        return match resources.cancel_gpu_writes(&write_reservations) {
            Ok(()) => Err(Box::new(ImageBlitPreparationFailure {
                reason: ImageBlitPreparationError::Uses(admission),
                operation,
                live_writes: Box::new([]),
            })),
            Err(cancellation) => Err(Box::new(ImageBlitPreparationFailure {
                reason: ImageBlitPreparationError::WriteRollback {
                    admission,
                    cancellation,
                },
                operation,
                live_writes: write_reservations,
            })),
        };
    }
    Ok(PreparedImageBlit {
        transaction,
        write,
        operation,
        representations: representations
            .into_iter()
            .map(|((backing, view), representation)| ViewRepresentation {
                backing,
                view,
                representation,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        uses,
        writes: write_reservations,
    })
}

/// Split a blit into the exact regions each endpoint reads and writes, keyed
/// by the backing *and the view of it* the endpoint is. A buffer endpoint
/// addresses the bytes and a texture endpoint addresses the texels; which one
/// an endpoint is is a property of the operation, and nothing downstream can
/// recover it once the key has dropped it.
fn collect_regions(
    operation: &ResolvedBlit,
    reads: &mut BTreeMap<(BackingId, BackingView), Vec<BackingRegion>>,
    writes: &mut BTreeMap<(BackingId, BackingView), Vec<BackingRegion>>,
) -> Result<(), ImageBlitPreparationError> {
    match operation {
        ResolvedBlit::Fill { .. } | ResolvedBlit::Copy { .. } => {
            return Err(ImageBlitPreparationError::BufferOnlyVariant);
        }
        ResolvedBlit::BufferToTexture(blit) => {
            let aspect =
                buffer_image_blit_aspect(blit.destination.backing.pixel_format(), blit.aspect);
            reads
                .entry((blit.source.storage, BackingView::Bytes))
                .or_default()
                .push(BackingRegion::Linear(blit.source.region));
            writes
                .entry((
                    blit.destination.storage,
                    BackingView::Image(ImageOwner::owning(blit.destination.image_owner)),
                ))
                .or_default()
                .extend(image_regions(
                    &blit.destination,
                    blit.destination_origin,
                    blit.extent,
                    aspect,
                )?);
        }
        ResolvedBlit::TextureToBuffer(blit) => {
            let aspect = buffer_image_blit_aspect(blit.source.backing.pixel_format(), blit.aspect);
            reads
                .entry((
                    blit.source.storage,
                    BackingView::Image(ImageOwner::owning(blit.source.image_owner)),
                ))
                .or_default()
                .extend(image_regions(
                    &blit.source,
                    blit.source_origin,
                    blit.extent,
                    aspect,
                )?);
            writes
                .entry((blit.destination.storage, BackingView::Bytes))
                .or_default()
                .push(BackingRegion::Linear(blit.destination.region));
        }
        ResolvedBlit::TextureToTexture(blit) => {
            reads
                .entry((
                    blit.source.storage,
                    BackingView::Image(ImageOwner::owning(blit.source.image_owner)),
                ))
                .or_default()
                .extend(image_regions(
                    &blit.source,
                    blit.source_origin,
                    blit.extent,
                    blit.aspect,
                )?);
            writes
                .entry((
                    blit.destination.storage,
                    BackingView::Image(ImageOwner::owning(blit.destination.image_owner)),
                ))
                .or_default()
                .extend(image_regions(
                    &blit.destination,
                    blit.destination_origin,
                    blit.extent,
                    blit.aspect,
                )?);
        }
        ResolvedBlit::TextureCopyBatch(batch) => {
            for level in std::iter::once(&batch.first_level).chain(batch.remaining_levels.iter()) {
                for (source, destination) in
                    std::iter::once(&level.first_slice).chain(level.remaining_slices.iter())
                {
                    let extent = crate::TextureExtent {
                        width: u64::from(source.backing.width()),
                        height: u64::from(source.backing.height()),
                        depth: u64::from(source.backing.depth()),
                    };
                    let origin = crate::TextureOrigin { x: 0, y: 0, z: 0 };
                    reads
                        .entry((
                            source.storage,
                            BackingView::Image(ImageOwner::owning(source.image_owner)),
                        ))
                        .or_default()
                        .extend(image_regions(source, origin, extent, BlitAspect::Full)?);
                    writes
                        .entry((
                            destination.storage,
                            BackingView::Image(ImageOwner::owning(destination.image_owner)),
                        ))
                        .or_default()
                        .extend(image_regions(
                            destination,
                            origin,
                            extent,
                            BlitAspect::Full,
                        )?);
                }
            }
        }
    }
    Ok(())
}

/// Collapse view-keyed regions onto the backings they name.
///
/// Content synchronization and scheduler hazards are statements about bytes,
/// and both views of one backing are the same bytes. Only native-object
/// selection needs the view.
fn fold_by_backing(
    keyed: BTreeMap<(BackingId, BackingView), Vec<BackingRegion>>,
) -> BTreeMap<BackingId, Vec<BackingRegion>> {
    let mut folded = BTreeMap::<BackingId, Vec<BackingRegion>>::new();
    for ((backing, _), regions) in keyed {
        folded.entry(backing).or_default().extend(regions);
    }
    folded
}

/// Exact canonical source regions a blit consumes. Destination-only fills
/// require no initial-content synchronization.
pub fn blit_content_synchronization_requests(
    operation: &ResolvedBlit,
) -> Result<Box<[crate::ContentSynchronizationRequest]>, ImageBlitPreparationError> {
    let mut reads = BTreeMap::<BackingId, (Vec<BackingRegion>, BTreeSet<BackingView>)>::new();
    match operation {
        ResolvedBlit::Fill { .. } => return Ok(Box::new([])),
        ResolvedBlit::Copy { source, .. } => {
            let entry = reads.entry(source.storage).or_default();
            entry.0.push(BackingRegion::Linear(source.region));
            entry.1.insert(BackingView::Bytes);
        }
        _ => {
            let mut keyed = BTreeMap::new();
            collect_regions(operation, &mut keyed, &mut BTreeMap::new())?;
            for ((backing, view), regions) in keyed {
                let entry = reads.entry(backing).or_default();
                entry.0.extend(regions);
                entry.1.insert(view);
            }
        }
    }
    Ok(reads
        .into_iter()
        .map(|(backing, (mut regions, views))| {
            regions.sort_unstable();
            regions.dedup();
            crate::ContentSynchronizationRequest {
                backing,
                regions: regions.into_boxed_slice(),
                permitted_pending_writes: Box::new([]),
                views: views.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

/// Compile one resolved blit's exact backing regions into scheduler hazards.
///
/// This shares the same region projection as resource preparation, so native
/// write reservation and pre-admission ordering cannot disagree about the
/// bytes or subresources the operation touches.
pub fn resolved_blit_accesses(
    operation: &ResolvedBlit,
    hazard_domain: HazardDomainId,
) -> Result<Box<[AccessIntent]>, ImageBlitPreparationError> {
    let mut reads = BTreeMap::<BackingId, Vec<BackingRegion>>::new();
    let mut writes = BTreeMap::<BackingId, Vec<BackingRegion>>::new();
    match operation {
        ResolvedBlit::Fill { destination, .. } => writes
            .entry(destination.storage)
            .or_default()
            .push(BackingRegion::Linear(destination.region)),
        ResolvedBlit::Copy {
            source,
            destination,
        } => {
            reads
                .entry(source.storage)
                .or_default()
                .push(BackingRegion::Linear(source.region));
            writes
                .entry(destination.storage)
                .or_default()
                .push(BackingRegion::Linear(destination.region));
        }
        _ => {
            let mut keyed_reads = BTreeMap::new();
            let mut keyed_writes = BTreeMap::new();
            collect_regions(operation, &mut keyed_reads, &mut keyed_writes)?;
            reads = fold_by_backing(keyed_reads);
            writes = fold_by_backing(keyed_writes);
        }
    }
    for regions in reads.values_mut().chain(writes.values_mut()) {
        regions.sort();
        regions.dedup();
    }
    Ok(reads
        .into_iter()
        .flat_map(|(backing, regions)| {
            regions
                .into_iter()
                .map(move |region| blit_access(hazard_domain, backing, region, AccessMode::Read))
        })
        .chain(writes.into_iter().flat_map(|(backing, regions)| {
            regions
                .into_iter()
                .map(move |region| blit_access(hazard_domain, backing, region, AccessMode::Write))
        }))
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn blit_access(
    hazard_domain: HazardDomainId,
    backing: BackingId,
    region: BackingRegion,
    mode: AccessMode,
) -> AccessIntent {
    let scope = match region {
        BackingRegion::Whole => AccessScope::WholeBacking,
        BackingRegion::Linear(range) => AccessScope::Linear(range),
        BackingRegion::Image(region) => AccessScope::Image(
            crate::ImageSubresourceRange::new(
                region.aspect,
                region.mip,
                1,
                region.layer,
                1,
                Some(region.texels),
            )
            .expect("one exact image region has a nonempty mip and layer"),
        ),
    };
    AccessIntent {
        hazard_domain,
        target: Some(AccessTarget::Backing(backing)),
        resource: None,
        scope,
        mode,
        stages: StageScope::Blit,
    }
}

fn image_regions(
    endpoint: &ResolvedTextureEndpoint,
    origin: crate::TextureOrigin,
    extent: crate::TextureExtent,
    _aspect: BlitAspect,
) -> Result<Box<[BackingRegion]>, ImageBlitPreparationError> {
    let [x, y, z] = [origin.x, origin.y, origin.z]
        .map(u32::try_from)
        .map(Result::ok);
    let [width, height, depth] = [extent.width, extent.height, extent.depth]
        .map(u32::try_from)
        .map(Result::ok);
    let (Some(x), Some(y), Some(z), Some(width), Some(height), Some(depth)) =
        (x, y, z, width, height, depth)
    else {
        return Err(ImageBlitPreparationError::CoordinateOverflow);
    };
    if width == 0 || height == 0 || depth == 0 {
        return Err(ImageBlitPreparationError::EmptyImageRegion);
    }
    match &endpoint.backing {
        crate::ResolvedTextureBacking::Linear(linear) => {
            if x.checked_add(width).is_none_or(|end| end > linear.width)
                || y.checked_add(height).is_none_or(|end| end > linear.height)
                || z.checked_add(depth).is_none_or(|end| end > linear.depth)
            {
                return Err(ImageBlitPreparationError::CoordinateOverflow);
            }
            let row_length = u64::from(width)
                .checked_mul(u64::from(linear.bpp))
                .ok_or(ImageBlitPreparationError::CoordinateOverflow)?;
            let mut regions = Vec::new();
            for plane in z..z + depth {
                for row in y..y + height {
                    let offset = linear
                        .texel_offset(u64::from(x), u64::from(row), u64::from(plane))
                        .ok_or(ImageBlitPreparationError::CoordinateOverflow)?;
                    let end = offset
                        .checked_add(row_length)
                        .ok_or(ImageBlitPreparationError::CoordinateOverflow)?;
                    if end > linear.alloc_size {
                        return Err(ImageBlitPreparationError::CoordinateOverflow);
                    }
                    regions.push(BackingRegion::Linear(
                        crate::LinearRange::new(offset, row_length)
                            .ok_or(ImageBlitPreparationError::EmptyImageRegion)?,
                    ));
                }
            }
            Ok(regions.into_boxed_slice())
        }
        crate::ResolvedTextureBacking::Surface(surface) => {
            if z != 0
                || depth != 1
                || x.checked_add(width).is_none_or(|end| end > surface.width)
                || y.checked_add(height).is_none_or(|end| end > surface.height)
            {
                return Err(ImageBlitPreparationError::CoordinateOverflow);
            }
            let row_length = u64::from(width)
                .checked_mul(u64::from(surface.bpp))
                .ok_or(ImageBlitPreparationError::CoordinateOverflow)?;
            let mut regions = Vec::new();
            for row in y..y + height {
                let offset = surface
                    .surface_offset
                    .checked_add(u64::from(row) * u64::from(surface.row_stride))
                    .and_then(|offset| offset.checked_add(u64::from(x) * u64::from(surface.bpp)))
                    .ok_or(ImageBlitPreparationError::CoordinateOverflow)?;
                let end = offset
                    .checked_add(row_length)
                    .ok_or(ImageBlitPreparationError::CoordinateOverflow)?;
                if end > surface.span_end {
                    return Err(ImageBlitPreparationError::CoordinateOverflow);
                }
                regions.push(BackingRegion::Linear(
                    crate::LinearRange::new(offset, row_length)
                        .ok_or(ImageBlitPreparationError::EmptyImageRegion)?,
                ));
            }
            Ok(regions.into_boxed_slice())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageBlitCancellationError {
    Writes(GpuWriteBatchError),
    Uses(ResourceUseBatchError),
}

#[derive(Debug)]
pub struct CancelledImageBlit<T> {
    pub operation: ResolvedBlit,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

#[derive(Debug)]
pub struct ImageBlitCancellationFailure {
    pub reason: ImageBlitCancellationError,
    pub prepared: PreparedImageBlit,
}

pub fn cancel_prepared_image_blit<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedImageBlit,
) -> Result<CancelledImageBlit<T>, Box<ImageBlitCancellationFailure>> {
    if let Err(reason) = resources.validate_cancel_gpu_writes(&prepared.writes) {
        return Err(Box::new(ImageBlitCancellationFailure {
            reason: ImageBlitCancellationError::Writes(reason),
            prepared,
        }));
    }
    if let Err(reason) =
        resources.validate_cancel_representation_uses(prepared.transaction, &prepared.uses)
    {
        return Err(Box::new(ImageBlitCancellationFailure {
            reason: ImageBlitCancellationError::Uses(reason),
            prepared,
        }));
    }
    resources
        .cancel_gpu_writes(&prepared.writes)
        .expect("image write cancellation was prevalidated");
    let progress = resources
        .cancel_representation_uses(prepared.transaction, &prepared.uses)
        .expect("image representation cancellation was prevalidated");
    Ok(CancelledImageBlit {
        operation: prepared.operation,
        resources: progress,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageBlitAcceptanceError {
    CompletionSetMismatch,
    Replay(ReplayAcceptanceError),
}

#[derive(Debug)]
pub struct ImageBlitAcceptanceFailure<Semantic> {
    pub reason: ImageBlitAcceptanceError,
    pub native: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    pub blit: PreparedImageBlit,
}

#[derive(Debug)]
pub struct AcceptedImageBlit<T> {
    pub replay: ReplayAcceptance<T>,
    pub operation: ResolvedBlit,
    pub resources: Box<[ResolvedResourceCompletion]>,
}

/// Join driver acceptance to the exact image blit's semantic completion set.
/// A mismatch changes neither replay nor resource ownership.
pub fn commit_image_blit_acceptance<Semantic: Clone, T>(
    runtime: &mut TransactionRuntime<Semantic>,
    native: &mut DirectReplayNativeOwner<ResolvedReplayCompletion<Semantic>>,
    resources: &mut ResourceLifecycleOwner<T>,
    prepared_native: PreparedNativeSubmission<ResolvedReplayCompletion<Semantic>>,
    prepared_blit: PreparedImageBlit,
) -> Result<AcceptedImageBlit<T>, Box<ImageBlitAcceptanceFailure<Semantic>>> {
    let completions = prepared_blit.resource_completions();
    if prepared_native.semantic().resources != completions {
        return Err(Box::new(ImageBlitAcceptanceFailure {
            reason: ImageBlitAcceptanceError::CompletionSetMismatch,
            native: prepared_native,
            blit: prepared_blit,
        }));
    }
    let replay = match crate::commit_replay_acceptance(
        runtime,
        native,
        resources,
        prepared_native,
        prepared_blit.transaction,
        prepared_blit.backings(),
    ) {
        Ok(replay) => replay,
        Err(failure) => {
            return Err(Box::new(ImageBlitAcceptanceFailure {
                reason: ImageBlitAcceptanceError::Replay(failure.reason),
                native: failure.prepared,
                blit: prepared_blit,
            }));
        }
    };
    Ok(AcceptedImageBlit {
        replay,
        operation: prepared_blit.operation,
        resources: completions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pixel_format::{MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, MTL_FORMAT_RGBA8_UNORM},
        BufferFillPattern, CompletionStamp, DeviceTransactionPayload, LinearRange,
        RepresentationRoute, ResolvedBufferRange, ResolvedBufferToTextureBlit,
        ResolvedLinearTextureLevel, ResolvedResourceLifecycle, ResolvedTextureBacking,
        ResourceLifecycleEffect, SessionGeneration, StorageBacking, TextureExtent, TextureOrigin,
        GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ByteLength, ChannelId, GuestVirtualAddress, QueueOwnerId, ResourceId, ResourceObject,
        SessionGenerationId, VulkanDeviceEpochId,
    };

    fn backing(resources: &mut ResourceLifecycleOwner<&'static str>) -> BackingId {
        match resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        {
            ResourceLifecycleEffect::BackingCreated(backing) => backing,
            _ => unreachable!(),
        }
    }

    /// An execution representation for one view of a backing. The view is the
    /// fixture's own statement about what it built: a buffer endpoint of a
    /// blit needs a buffer and a texture endpoint needs an image.
    fn execution(
        resources: &mut ResourceLifecycleOwner<&'static str>,
        backing: BackingId,
        view: BackingView,
        name: &'static str,
    ) -> RepresentationId {
        resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                view,
                name,
            )
            .unwrap()
    }

    fn materialize(
        resources: &mut ResourceLifecycleOwner<&'static str>,
        backing: BackingId,
        representation: RepresentationId,
        region: BackingRegion,
    ) {
        let snapshot = resources.snapshot_content(backing, &[region]).unwrap();
        for transfer in resources
            .plan_transfers(backing, GUEST_REPRESENTATION, representation, &snapshot)
            .unwrap()
            .iter()
            .copied()
        {
            resources.complete_transfer(transfer).unwrap();
        }
    }

    fn buffer(backing: BackingId) -> ResolvedBufferRange {
        ResolvedBufferRange {
            resource: ResourceId::new(1, 1),
            storage: backing,
            region: LinearRange::new(8, 80).unwrap(),
            address: GuestVirtualAddress::new(0x1008),
            length: ByteLength::new(80),
        }
    }

    /// The texture every endpoint in these fixtures names, and therefore the
    /// image view its backing carries.
    const TEXTURE: ResourceId<ResourceObject> = ResourceId::new(2, 1);

    fn texture(backing: BackingId, format: u16) -> ResolvedTextureEndpoint {
        ResolvedTextureEndpoint {
            resource: TEXTURE,
            image_owner: TEXTURE,
            storage: backing,
            level: 2,
            slice: 3,
            backing: ResolvedTextureBacking::Linear(ResolvedLinearTextureLevel {
                base_gva: 0x2000,
                alloc_size: 0x1000,
                level_offset: 0,
                row_stride: 64,
                slice_stride: 256,
                slice_index: 3,
                width: 16,
                height: 16,
                depth: 1,
                bpp: 4,
                pixel_format: format,
            }),
        }
    }

    #[test]
    fn buffer_copy_compiles_exact_read_and_write_hazards() {
        let source = BackingId::new(4);
        let destination = BackingId::new(5);
        let hazard_domain = HazardDomainId::new(6);
        let operation = ResolvedBlit::Copy {
            source: buffer(source),
            destination: buffer(destination),
        };

        let accesses = resolved_blit_accesses(&operation, hazard_domain).unwrap();

        assert_eq!(accesses.len(), 2);
        assert_eq!(accesses[0].hazard_domain, hazard_domain);
        assert_eq!(accesses[0].target, Some(AccessTarget::Backing(source)));
        assert_eq!(
            accesses[0].scope,
            AccessScope::Linear(LinearRange::new(8, 80).unwrap())
        );
        assert_eq!(accesses[0].mode, AccessMode::Read);
        assert_eq!(accesses[0].stages, StageScope::Blit);
        assert_eq!(accesses[1].target, Some(AccessTarget::Backing(destination)));
        assert_eq!(accesses[1].mode, AccessMode::Write);
    }

    /// One guest allocation declared as both a buffer and a linear texture is
    /// one backing carrying two native objects: the image the texture owner
    /// materialized, and the host staging endpoint that image transfers
    /// through. A blit reading it as a buffer must resolve the endpoint.
    ///
    /// Without the view on the key this resolves the backing's single
    /// execution representation for both endpoints, hands the recorder an
    /// image where it asked for a buffer, and the copy refuses as
    /// `UnknownBuffer` on every retry — a permanent submission-head stall
    /// rather than one lost command.
    #[test]
    fn a_backing_read_as_bytes_resolves_its_endpoint_and_not_its_image() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let source = backing(&mut resources);
        let destination = backing(&mut resources);
        // The texture owner's materialization: a staging endpoint for the
        // bytes and an image for the texels, both on the one backing.
        let endpoint = resources
            .create_representation(source, RepresentationRoute::HostStagingEndpoint, "endpoint")
            .unwrap();
        assert_eq!(endpoint, crate::HOST_REPRESENTATION);
        let image = resources
            .create_execution_representation(
                source,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Image(ImageOwner::owning(TEXTURE)),
                "image",
            )
            .unwrap();
        let destination_representation = execution(
            &mut resources,
            destination,
            BackingView::Image(ImageOwner::owning(TEXTURE)),
            "destination",
        );
        materialize(
            &mut resources,
            source,
            endpoint,
            BackingRegion::Linear(LinearRange::new(8, 80).unwrap()),
        );

        let operation = ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: buffer(source),
            source_bytes_per_row: 16,
            source_bytes_per_image: 80,
            destination: texture(destination, MTL_FORMAT_RGBA8_UNORM),
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 5,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });
        let prepared = prepare_image_blit(
            &mut resources,
            TransactionId::new(1),
            SubmissionId::new(2),
            operation,
        )
        .unwrap();

        assert_eq!(
            ViewRepresentation::lookup(prepared.representations(), source, BackingView::Bytes),
            Some(endpoint)
        );
        assert_ne!(endpoint, image);
        assert_eq!(
            ViewRepresentation::lookup(
                prepared.representations(),
                destination,
                BackingView::Image(ImageOwner::owning(TEXTURE))
            ),
            Some(destination_representation)
        );
        // The use keeps the endpoint alive for the transaction; the image is
        // this blit's business only if it names it.
        let [use_] = prepared
            .uses()
            .iter()
            .filter(|use_| use_.backing == source)
            .collect::<Vec<_>>()[..]
        else {
            unreachable!("the source backing is used exactly once")
        };
        assert_eq!(use_.representations.as_ref(), [endpoint]);
    }

    #[test]
    fn buffer_to_texture_preparation_proves_source_and_reserves_exact_storage_rows() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let source = backing(&mut resources);
        let destination = backing(&mut resources);
        let source_representation = execution(&mut resources, source, BackingView::Bytes, "source");
        let destination_representation = execution(
            &mut resources,
            destination,
            BackingView::Image(ImageOwner::owning(TEXTURE)),
            "destination",
        );
        materialize(
            &mut resources,
            source,
            source_representation,
            BackingRegion::Linear(LinearRange::new(8, 80).unwrap()),
        );
        let operation = ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: buffer(source),
            source_bytes_per_row: 16,
            source_bytes_per_image: 80,
            destination: texture(destination, MTL_FORMAT_RGBA8_UNORM),
            destination_origin: TextureOrigin { x: 1, y: 2, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 5,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });
        let prepared = prepare_image_blit(
            &mut resources,
            TransactionId::new(1),
            SubmissionId::new(2),
            operation.clone(),
        )
        .unwrap();
        assert_eq!(
            prepared.representations(),
            [
                ViewRepresentation {
                    backing: source,
                    view: BackingView::Bytes,
                    representation: source_representation,
                },
                ViewRepresentation {
                    backing: destination,
                    view: BackingView::Image(ImageOwner::owning(TEXTURE)),
                    representation: destination_representation,
                },
            ]
        );
        assert_eq!(
            prepared.writes()[0]
                .regions
                .iter()
                .map(|region| region.region)
                .collect::<Vec<_>>(),
            [900, 964, 1028, 1092, 1156]
                .map(|offset| BackingRegion::Linear(LinearRange::new(offset, 16).unwrap()))
        );
        assert_eq!(
            prepared.resource_completions().as_ref(),
            [ResolvedResourceCompletion::GpuWrite {
                backing: destination,
                write: SubmissionId::new(2).into(),
                representation: destination_representation,
            }]
        );
        assert_eq!(
            cancel_prepared_image_blit(&mut resources, prepared)
                .unwrap()
                .operation,
            operation
        );
    }

    #[test]
    fn packed_depth_stencil_region_reserves_each_exact_storage_row_once() {
        let endpoint = texture(BackingId::new(9), MTL_FORMAT_DEPTH32_FLOAT_STENCIL8);
        let regions = image_regions(
            &endpoint,
            TextureOrigin { x: 0, y: 0, z: 0 },
            TextureExtent {
                width: 4,
                height: 4,
                depth: 1,
            },
            BlitAspect::Full,
        )
        .unwrap();
        assert_eq!(
            regions.as_ref(),
            [768, 832, 896, 960]
                .map(|offset| BackingRegion::Linear(LinearRange::new(offset, 16).unwrap()))
        );
    }

    #[test]
    fn option_free_combined_buffer_image_copy_reserves_packed_storage_once() {
        let destination = BackingId::new(9);
        let operation = ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: buffer(BackingId::new(8)),
            source_bytes_per_row: 16,
            source_bytes_per_image: 64,
            destination: texture(destination, MTL_FORMAT_DEPTH32_FLOAT_STENCIL8),
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 4,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });
        let mut reads = BTreeMap::new();
        let mut writes = BTreeMap::new();
        collect_regions(&operation, &mut reads, &mut writes).unwrap();
        assert_eq!(
            writes[&(destination, BackingView::Image(ImageOwner::owning(TEXTURE)))].as_slice(),
            [768, 832, 896, 960]
                .map(|offset| BackingRegion::Linear(LinearRange::new(offset, 16).unwrap()))
        );
    }

    #[test]
    fn buffer_only_operations_refuse_without_mutating_resource_state() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let operation = ResolvedBlit::Fill {
            destination: ResolvedBufferRange {
                resource: ResourceId::new(1, 1),
                storage: BackingId::new(99),
                region: LinearRange::new(0, 4).unwrap(),
                address: GuestVirtualAddress::new(0),
                length: ByteLength::new(4),
            },
            pattern: BufferFillPattern::Byte(0),
        };
        let failure = prepare_image_blit(
            &mut resources,
            TransactionId::new(1),
            SubmissionId::new(1),
            operation.clone(),
        )
        .unwrap_err();
        assert_eq!(failure.reason, ImageBlitPreparationError::BufferOnlyVariant);
        assert_eq!(failure.operation, operation);
        assert!(failure.live_writes.is_empty());
    }

    #[test]
    fn queue_acceptance_joins_the_exact_image_blit_completion_set() {
        let generation = SessionGenerationId::new(3);
        let epoch = VulkanDeviceEpochId::new(4);
        let mut runtime = TransactionRuntime::new(SessionGeneration::new(generation));
        let channel = ChannelId::new(1);
        runtime.define_channel(channel).unwrap();
        let transaction = runtime
            .admit_resolved(
                channel,
                Box::<[crate::ResolvedTransactionPrerequisite]>::default(),
                Some(CompletionStamp::new(channel.get(), 1)),
                DeviceTransactionPayload::<(), (), (), (), ()>::Exec(crate::ExecTransaction {
                    identity: reims_vgpu_protocol::SubmissionIdentity {
                        id: SubmissionId::new(30),
                        task: reims_vgpu_protocol::TaskId::new(1),
                    },
                    prologue: crate::ExecPrologue::default(),
                    streams: Box::new([]),
                    accesses: Box::new([]),
                }),
            )
            .unwrap();
        let mut resources = ResourceLifecycleOwner::new(epoch);
        let source = backing(&mut resources);
        let destination = backing(&mut resources);
        let source_representation = execution(&mut resources, source, BackingView::Bytes, "source");
        let destination_representation = execution(
            &mut resources,
            destination,
            BackingView::Image(ImageOwner::owning(TEXTURE)),
            "destination",
        );
        materialize(
            &mut resources,
            source,
            source_representation,
            BackingRegion::Linear(LinearRange::new(8, 80).unwrap()),
        );
        let operation = ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: buffer(source),
            source_bytes_per_row: 16,
            source_bytes_per_image: 80,
            destination: texture(destination, MTL_FORMAT_RGBA8_UNORM),
            destination_origin: TextureOrigin { x: 1, y: 2, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 5,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });
        let blit = prepare_image_blit(
            &mut resources,
            transaction.id,
            SubmissionId::new(30),
            operation,
        )
        .unwrap();
        let completions = blit.resource_completions();
        let mut native = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        native
            .assign_recording(runtime.recording_plan(transaction.id).unwrap())
            .unwrap();
        runtime.recorded(transaction.id).unwrap();
        runtime.take_submission_ready();
        let plan = native
            .queue_candidate(
                transaction.id,
                Box::<[(TransactionId, crate::WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        let prepared_native = native
            .prepare(
                plan,
                QueueOwnerId::new(1),
                generation,
                ResolvedReplayCompletion {
                    semantic: "done",
                    resources: completions.clone(),
                },
            )
            .unwrap();
        let accepted = commit_image_blit_acceptance(
            &mut runtime,
            &mut native,
            &mut resources,
            prepared_native,
            blit,
        )
        .unwrap();
        assert_eq!(accepted.replay.native.transaction, transaction.id);
        assert_eq!(accepted.resources, completions);
        assert!(matches!(
            accepted.resources.as_ref(),
            [ResolvedResourceCompletion::GpuWrite {
                backing: found_backing,
                representation: found_representation,
                ..
            }] if *found_backing == destination
                && *found_representation == destination_representation
        ));
    }
}
