//! Ordered synchronization of current backing content before native reads.
//!
//! A newly constructed execution representation starts without content.  An
//! EXEC read snapshots the canonical backing versions and, when its selected
//! route requires a transfer, owns that transfer as an auxiliary prefix.  The
//! prefix does not create a guest write or otherwise advance content versions.

use crate::{
    BackingRegion, HostIngressBatchError, HostIngressKey, HostIngressTransfer, ManagedBackingError,
    ManagedBackingProgress, RepresentationRoute, RepresentationUse, ResourceLifecycleOwner,
    ResourceUseBatchError, TransferBatchError, TransferKey, GUEST_REPRESENTATION,
    HOST_REPRESENTATION,
};
use reims_vgpu_protocol::{BackingId, TransactionId};
use std::collections::{BTreeMap, BTreeSet};

fn take_available_regions(
    remaining: &mut Vec<BackingRegion>,
    available: &[BackingRegion],
) -> Vec<BackingRegion> {
    let mut claimed = Vec::new();
    for available in available.iter().copied() {
        for region in remaining.iter().copied() {
            if let Some(overlap) = crate::content_authority::intersection(region, available) {
                claimed.push(overlap);
            }
        }
        *remaining = remaining
            .drain(..)
            .flat_map(|region| crate::content_authority::subtract(region, available))
            .collect();
    }
    claimed
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentSynchronizationRequest {
    pub backing: BackingId,
    pub regions: Box<[BackingRegion]>,
    /// Pending writes owned by the ordered EXEC suffix for which this
    /// synchronization is the queue prefix.
    pub permitted_pending_writes: Box<[crate::GpuWriteId]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentSynchronizationError {
    EmptyRegions(BackingId),
    /// A GPU write over this region is outstanding and no ordering permits
    /// reading past it.
    ///
    /// `consumer` and the write's own producer are the pair to read first, and
    /// they must not be equal: a transaction can only wait for a write of its
    /// own until it submits, and it cannot submit while this refusal holds its
    /// preparation. `permits` says which of the two ways the permit could be
    /// missing --- zero means the caller computed no permitted set for this
    /// backing at all, and non-zero means it computed one that this write is
    /// not in, which is a set that went stale rather than one that was never
    /// asked for.
    PendingGpuWrite {
        backing: BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
        write: crate::GpuWriteId,
        consumer: TransactionId,
        permits: usize,
    },
    Backing {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    RouteCannotSynchronize {
        backing: BackingId,
        route: RepresentationRoute,
    },
    CurrentSourceAbsent {
        backing: BackingId,
        required: crate::RegionVersion,
    },
    Uses(ResourceUseBatchError),
    Cancellation(ContentSynchronizationCancellationError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentSynchronizationCancellationError {
    Transfers(TransferBatchError),
    HostIngresses(HostIngressBatchError),
    Uses(ResourceUseBatchError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedContentSynchronizationBatch {
    transaction: TransactionId,
    transfers: Box<[TransferKey]>,
    host_ingresses: Box<[HostIngressKey]>,
    deferred_host_ingress_transfers: Box<[HostIngressTransfer]>,
    uses: Box<[RepresentationUse]>,
}

impl PreparedContentSynchronizationBatch {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn transfers(&self) -> &[TransferKey] {
        &self.transfers
    }

    pub const fn host_ingresses(&self) -> &[HostIngressKey] {
        &self.host_ingresses
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

    pub fn resource_completions(&self) -> Box<[crate::ResolvedResourceCompletion]> {
        self.transfers
            .iter()
            .copied()
            .map(crate::ResolvedResourceCompletion::Transfer)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Publish staging coverage only after the exact guest bytes have been
    /// copied.  The resulting GPU transfers remain owned by this batch.
    pub fn publish_host_ingresses_after_copy<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
    ) -> Result<Box<[TransferKey]>, HostIngressBatchError> {
        let planned =
            resources.complete_host_ingress_transfers(&self.deferred_host_ingress_transfers)?;
        let mut transfers = std::mem::take(&mut self.transfers).into_vec();
        transfers.extend_from_slice(&planned);
        transfers.sort_unstable();
        transfers.dedup();
        self.transfers = transfers.into_boxed_slice();
        self.host_ingresses = Box::new([]);
        self.deferred_host_ingress_transfers = Box::new([]);
        Ok(planned)
    }
}

fn cancel_partial<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    transfers: &[TransferKey],
    host_ingresses: &[HostIngressKey],
    uses: &[RepresentationUse],
) -> Result<Vec<(BackingId, ManagedBackingProgress<T>)>, ContentSynchronizationCancellationError> {
    resources
        .validate_cancel_transfers(transfers)
        .map_err(ContentSynchronizationCancellationError::Transfers)?;
    resources
        .validate_cancel_host_ingresses(host_ingresses)
        .map_err(ContentSynchronizationCancellationError::HostIngresses)?;
    resources
        .validate_cancel_representation_uses(transaction, uses)
        .map_err(ContentSynchronizationCancellationError::Uses)?;
    resources
        .cancel_transfers(transfers)
        .expect("content synchronization transfer cancellation was prevalidated");
    resources
        .cancel_host_ingresses(host_ingresses)
        .expect("content synchronization ingress cancellation was prevalidated");
    resources
        .cancel_representation_uses(transaction, uses)
        .map_err(ContentSynchronizationCancellationError::Uses)
}

pub fn cancel_prepared_content_synchronization<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedContentSynchronizationBatch,
) -> Result<Vec<(BackingId, ManagedBackingProgress<T>)>, ContentSynchronizationCancellationError> {
    cancel_partial(
        resources,
        prepared.transaction,
        &prepared.transfers,
        &prepared.host_ingresses,
        &prepared.uses,
    )
}

pub fn prepare_content_synchronization<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    requests: impl IntoIterator<Item = ContentSynchronizationRequest>,
) -> Result<PreparedContentSynchronizationBatch, ContentSynchronizationError> {
    let mut grouped = BTreeMap::<BackingId, BTreeSet<BackingRegion>>::new();
    let mut permitted_pending = BTreeMap::<BackingId, BTreeSet<crate::GpuWriteId>>::new();
    for request in requests {
        if request.regions.is_empty() {
            return Err(ContentSynchronizationError::EmptyRegions(request.backing));
        }
        grouped
            .entry(request.backing)
            .or_default()
            .extend(request.regions);
        permitted_pending
            .entry(request.backing)
            .or_default()
            .extend(request.permitted_pending_writes);
    }

    let mut transfers = Vec::new();
    let mut host_ingresses = Vec::new();
    let mut deferred = Vec::new();
    let mut uses = Vec::new();
    for (backing, regions) in grouped {
        let regions = regions.into_iter().collect::<Vec<_>>();
        // A backing carries one image per texture declared over its range, and
        // each of them holds its own copy of these bytes. Synchronizing one
        // and leaving the rest stale is what a single designation did.
        let designated = resources
            .designated_views(backing)
            .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?;
        // One entry per backing, accumulated across its designated views.
        // Every view over these bytes holds its own copy and each may need a
        // different set of representations moved, but a use batch is keyed by
        // backing and refuses a backing named twice --- so a backing with two
        // designated views produced `DuplicateBacking` and parked its channel.
        // `RepresentationUse` carries a set for exactly this reason.
        let mut backing_uses = BTreeSet::new();
        for (_, destination) in designated {
            let result = (|| {
                let regions = regions.clone();
                let snapshot = resources
                    .snapshot_content(backing, &regions)
                    .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?;
                if resources
                    .representation_matches(backing, destination, &snapshot)
                    .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?
                {
                    return Ok(BTreeSet::new());
                }
                let pending = resources
                    .pending_gpu_writes_overlapping(backing, destination, &regions)
                    .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?;
                if let Some(write) = pending
                    .iter()
                    .find(|write| {
                        permitted_pending
                            .get(&backing)
                            .is_none_or(|permitted| !permitted.contains(write))
                    })
                    .copied()
                {
                    return Err(ContentSynchronizationError::PendingGpuWrite {
                        backing,
                        representation: destination,
                        write,
                        consumer: transaction,
                        permits: permitted_pending
                            .get(&backing)
                            .map_or(0, std::collections::BTreeSet::len),
                    });
                }
                let route = resources
                    .representation_route(backing, destination)
                    .expect("the execution representation has an owned route");
                if !matches!(
                    route,
                    RepresentationRoute::ImportedGuestTransfer { .. }
                        | RepresentationRoute::HostStagingTransfer { .. }
                ) {
                    return Err(ContentSynchronizationError::RouteCannotSynchronize {
                        backing,
                        route,
                    });
                }
                let endpoint = match route {
                    RepresentationRoute::ImportedGuestTransfer { .. } => GUEST_REPRESENTATION,
                    RepresentationRoute::HostStagingTransfer { .. } => HOST_REPRESENTATION,
                    _ => unreachable!("the synchronizable routes were matched above"),
                };
                let mut used_representations = BTreeSet::from([destination]);
                for required in snapshot.iter().copied() {
                    let destination_regions = resources
                        .current_regions_in_representation(backing, destination, required)
                        .map_err(|reason| ContentSynchronizationError::Backing {
                            backing,
                            reason,
                        })?;
                    let mut remaining = vec![required.region];
                    take_available_regions(&mut remaining, &destination_regions);
                    if remaining.is_empty() {
                        continue;
                    }

                    let endpoint_regions = resources
                        .current_regions_in_representation(backing, endpoint, required)
                        .map_err(|reason| ContentSynchronizationError::Backing {
                            backing,
                            reason,
                        })?;
                    let endpoint_claimed =
                        take_available_regions(&mut remaining, &endpoint_regions);
                    if !endpoint_claimed.is_empty() {
                        let snapshot = endpoint_claimed
                            .into_iter()
                            .map(|region| crate::RegionVersion {
                                region,
                                version: required.version,
                            })
                            .collect::<Vec<_>>();
                        transfers.extend(
                            resources
                                .plan_transfers(backing, endpoint, destination, &snapshot)
                                .map_err(|reason| ContentSynchronizationError::Backing {
                                    backing,
                                    reason,
                                })?,
                        );
                        used_representations.insert(endpoint);
                    }

                    for (source, source_regions) in resources
                        .current_native_regions_for_version(
                            backing,
                            &[GUEST_REPRESENTATION, HOST_REPRESENTATION, destination],
                            required,
                        )
                        .map_err(|reason| ContentSynchronizationError::Backing {
                            backing,
                            reason,
                        })?
                    {
                        let claimed = take_available_regions(&mut remaining, &source_regions);
                        if claimed.is_empty() {
                            continue;
                        }
                        let snapshot = claimed
                            .into_iter()
                            .map(|region| crate::RegionVersion {
                                region,
                                version: required.version,
                            })
                            .collect::<Vec<_>>();
                        transfers.extend(
                            resources
                                .plan_transfers(backing, source, destination, &snapshot)
                                .map_err(|reason| ContentSynchronizationError::Backing {
                                    backing,
                                    reason,
                                })?,
                        );
                        used_representations.insert(source);
                    }

                    if endpoint == HOST_REPRESENTATION && !remaining.is_empty() {
                        let guest_regions = resources
                            .current_regions_in_representation(
                                backing,
                                GUEST_REPRESENTATION,
                                required,
                            )
                            .map_err(|reason| ContentSynchronizationError::Backing {
                                backing,
                                reason,
                            })?;
                        for region in take_available_regions(&mut remaining, &guest_regions) {
                            let ingress = resources
                                .plan_host_ingress(
                                    backing,
                                    crate::RegionVersion {
                                        region,
                                        version: required.version,
                                    },
                                )
                                .map_err(|reason| ContentSynchronizationError::Backing {
                                    backing,
                                    reason,
                                })?;
                            host_ingresses.push(ingress);
                            deferred.push(HostIngressTransfer {
                                ingress,
                                destination,
                            });
                            used_representations.insert(HOST_REPRESENTATION);
                        }
                    }
                    if let Some(region) = remaining.first().copied() {
                        return Err(ContentSynchronizationError::CurrentSourceAbsent {
                            backing,
                            required: crate::RegionVersion {
                                region,
                                version: required.version,
                            },
                        });
                    }
                }
                Ok(used_representations)
            })();
            match result {
                Ok(used) => backing_uses.extend(used),
                Err(error) => {
                    return match cancel_partial(
                        resources,
                        transaction,
                        &transfers,
                        &host_ingresses,
                        &[],
                    ) {
                        Ok(_) => Err(error),
                        Err(reason) => Err(ContentSynchronizationError::Cancellation(reason)),
                    };
                }
            }
        }
        if !backing_uses.is_empty() {
            uses.push(RepresentationUse {
                backing,
                representations: backing_uses
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            });
        }
    }
    if let Err(reason) = resources.accept_uses(transaction, &uses) {
        return match cancel_partial(resources, transaction, &transfers, &host_ingresses, &[]) {
            Ok(_) => Err(ContentSynchronizationError::Uses(reason)),
            Err(reason) => Err(ContentSynchronizationError::Cancellation(reason)),
        };
    }
    Ok(PreparedContentSynchronizationBatch {
        transaction,
        transfers: transfers.into_boxed_slice(),
        host_ingresses: host_ingresses.into_boxed_slice(),
        deferred_host_ingress_transfers: deferred.into_boxed_slice(),
        uses: uses.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackingView;
    use crate::{LinearRange, ResolvedResourceLifecycle, ResourceLifecycleEffect, StorageBacking};
    use reims_vgpu_protocol::{SubmissionId, VulkanDeviceEpochId};

    #[test]
    fn staging_synchronization_preserves_the_guest_version_and_owns_one_ordered_transfer() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let region = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let destination = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let before = resources.snapshot_content(backing, &[region]).unwrap();
        let mut prepared = prepare_content_synchronization(
            &mut resources,
            TransactionId::new(7),
            [ContentSynchronizationRequest {
                backing,
                regions: Box::new([region]),
                permitted_pending_writes: Box::new([]),
            }],
        )
        .unwrap();
        assert_eq!(prepared.host_ingresses().len(), 1);
        assert!(prepared.transfers().is_empty());
        let transfers = prepared
            .publish_host_ingresses_after_copy(&mut resources)
            .unwrap();
        assert_eq!(transfers.len(), 1);
        assert_eq!(transfers[0].source, HOST_REPRESENTATION);
        assert_eq!(transfers[0].destination, destination);
        assert_eq!(
            resources.snapshot_content(backing, &[region]).unwrap(),
            before
        );
        assert_eq!(
            prepared.resource_completions().as_ref(),
            [crate::ResolvedResourceCompletion::Transfer(transfers[0])]
        );
        cancel_prepared_content_synchronization(&mut resources, prepared).unwrap();

        let prepared = prepare_content_synchronization(
            &mut resources,
            TransactionId::new(8),
            [ContentSynchronizationRequest {
                backing,
                regions: Box::new([region]),
                permitted_pending_writes: Box::new([]),
            }],
        )
        .unwrap();
        assert!(prepared.host_ingresses().is_empty());
        assert!(matches!(
            prepared.transfers(),
            [transfer]
                if transfer.source == HOST_REPRESENTATION
                    && transfer.destination == destination
                    && transfer.region == region
        ));
        cancel_prepared_content_synchronization(&mut resources, prepared).unwrap();
    }

    /// A backing with two designated views synchronizes both and names the
    /// backing once.
    ///
    /// Every view over one range holds its own copy of the bytes, so all of
    /// them have to be brought current --- which is why the synchronizer loops
    /// over the designations. It also emitted one `RepresentationUse` per
    /// designation, and a use batch is keyed by backing and refuses a backing
    /// named twice: a second view therefore turned a correct synchronization
    /// into `Uses(DuplicateBacking(..))`, which is not one lost command but a
    /// parked channel, because the refusal sits on a submission head. The
    /// entry carries a set of representations for exactly this reason.
    #[test]
    fn two_designated_views_over_one_backing_are_one_use_naming_both() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let region = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let route = RepresentationRoute::HostStagingTransfer {
            working: crate::WorkingMemoryClass::DeviceLocal,
        };
        let bytes = resources
            .create_execution_representation(backing, route, BackingView::Bytes, ())
            .unwrap();
        // A texture declared over the same range: its own view, its own image,
        // its own copy of these bytes.
        let image = resources
            .create_execution_representation(
                backing,
                route,
                BackingView::Image(crate::ImageOwner::owning(
                    reims_vgpu_protocol::ResourceId::new(9, 1),
                )),
                (),
            )
            .unwrap();
        assert_ne!(bytes, image);

        let prepared = prepare_content_synchronization(
            &mut resources,
            TransactionId::new(11),
            [ContentSynchronizationRequest {
                backing,
                regions: Box::new([region]),
                permitted_pending_writes: Box::new([]),
            }],
        )
        .expect("two designated views over one backing is one use, not a duplicate");
        // One entry for the backing, naming both destinations and the staging
        // endpoint they are filled from.
        assert_eq!(prepared.uses().len(), 1);
        assert_eq!(prepared.uses()[0].backing, backing);
        let named = prepared.uses()[0]
            .representations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert!(named.contains(&bytes));
        assert!(named.contains(&image));
        assert!(named.contains(&HOST_REPRESENTATION));
        // Both views are brought current, not just the first designation.
        assert_eq!(prepared.host_ingresses().len(), 2);
        cancel_prepared_content_synchronization(&mut resources, prepared).unwrap();
    }

    #[test]
    fn one_current_view_does_not_answer_for_a_backing_whose_other_view_is_empty() {
        // The question "is this backing already current?" has one answer per
        // designated view. A caller that asks one representation and
        // generalises reads "current" off the view that happens to hold the
        // content and drops the synchronization the empty one owes --- which
        // leaves a render bind addressing an image with nothing in it.
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let region = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let route = RepresentationRoute::HostStagingTransfer {
            working: crate::WorkingMemoryClass::DeviceLocal,
        };
        let bytes = resources
            .create_execution_representation(backing, route, BackingView::Bytes, ())
            .unwrap();
        let image_view = BackingView::Image(crate::ImageOwner::owning(
            reims_vgpu_protocol::ResourceId::new(9, 1),
        ));
        let image = resources
            .create_execution_representation(backing, route, image_view, ())
            .unwrap();

        // Only the bytes view is brought current.
        resources
            .plan_gpu_write(backing, SubmissionId::new(1), bytes, [region])
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), bytes)
            .unwrap();

        let snapshot = resources.snapshot_content(backing, &[region]).unwrap();
        // Asking the current one alone says the backing needs nothing.
        assert!(resources
            .representation_matches(backing, bytes, &snapshot)
            .unwrap());
        // Asking the owner of the designations names the one that does.
        assert_eq!(
            resources
                .stale_designated_representations(backing, &snapshot)
                .unwrap(),
            vec![(image_view, image)]
        );
    }

    #[test]
    fn synchronization_waits_for_an_overlapping_execution_write_instead_of_overwriting_it() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let region = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([region]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let destination = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let submission = SubmissionId::new(4);
        resources
            .plan_gpu_write(backing, submission, destination, [region])
            .unwrap();

        assert_eq!(
            prepare_content_synchronization(
                &mut resources,
                TransactionId::new(7),
                [ContentSynchronizationRequest {
                    backing,
                    regions: Box::new([region]),
                    permitted_pending_writes: Box::new([]),
                }],
            ),
            Err(ContentSynchronizationError::PendingGpuWrite {
                backing,
                representation: destination,
                write: submission.into(),
                consumer: TransactionId::new(7),
                permits: 0,
            })
        );

        let permitted = prepare_content_synchronization(
            &mut resources,
            TransactionId::new(8),
            [ContentSynchronizationRequest {
                backing,
                regions: Box::new([region]),
                permitted_pending_writes: Box::new([submission.into()]),
            }],
        )
        .unwrap();
        assert_eq!(permitted.host_ingresses().len(), 1);
        cancel_prepared_content_synchronization(&mut resources, permitted).unwrap();

        resources
            .complete_gpu_write(backing, submission, destination)
            .unwrap();
        let prepared = prepare_content_synchronization(
            &mut resources,
            TransactionId::new(7),
            [ContentSynchronizationRequest {
                backing,
                regions: Box::new([region]),
                permitted_pending_writes: Box::new([]),
            }],
        )
        .unwrap();
        assert!(prepared.transfers().is_empty());
        assert!(prepared.host_ingresses().is_empty());
        cancel_prepared_content_synchronization(&mut resources, prepared).unwrap();
    }

    #[test]
    fn staging_synchronization_combines_current_regions_from_distinct_representations() {
        let mut resources = ResourceLifecycleOwner::new(VulkanDeviceEpochId::new(1));
        let first = BackingRegion::Linear(LinearRange::new(0, 64).unwrap());
        let second = BackingRegion::Linear(LinearRange::new(64, 64).unwrap());
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([first, second]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let host = resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let retained = resources
            .create_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                (),
            )
            .unwrap();
        let destination = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();

        let first_snapshot = resources.snapshot_content(backing, &[first]).unwrap();
        let first_transfer = resources
            .plan_transfers(backing, GUEST_REPRESENTATION, host, &first_snapshot)
            .unwrap();
        resources.complete_transfer(first_transfer[0]).unwrap();
        let second_snapshot = resources.snapshot_content(backing, &[second]).unwrap();
        let second_transfer = resources
            .plan_transfers(backing, GUEST_REPRESENTATION, retained, &second_snapshot)
            .unwrap();
        resources.complete_transfer(second_transfer[0]).unwrap();

        let prepared = prepare_content_synchronization(
            &mut resources,
            TransactionId::new(7),
            [ContentSynchronizationRequest {
                backing,
                regions: Box::new([first, second]),
                permitted_pending_writes: Box::new([]),
            }],
        )
        .unwrap();

        assert!(prepared.host_ingresses().is_empty());
        assert_eq!(prepared.transfers().len(), 2);
        assert!(prepared.transfers().iter().any(|transfer| {
            transfer.source == host
                && transfer.destination == destination
                && transfer.region == first
        }));
        assert!(prepared.transfers().iter().any(|transfer| {
            transfer.source == retained
                && transfer.destination == destination
                && transfer.region == second
        }));
        cancel_prepared_content_synchronization(&mut resources, prepared).unwrap();
    }
}
