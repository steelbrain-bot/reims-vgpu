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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentSynchronizationRequest {
    pub backing: BackingId,
    pub regions: Box<[BackingRegion]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentSynchronizationError {
    EmptyRegions(BackingId),
    Backing {
        backing: BackingId,
        reason: ManagedBackingError,
    },
    RouteCannotSynchronize {
        backing: BackingId,
        route: RepresentationRoute,
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
    for request in requests {
        if request.regions.is_empty() {
            return Err(ContentSynchronizationError::EmptyRegions(request.backing));
        }
        grouped
            .entry(request.backing)
            .or_default()
            .extend(request.regions);
    }

    let mut transfers = Vec::new();
    let mut host_ingresses = Vec::new();
    let mut deferred = Vec::new();
    let mut uses = Vec::new();
    for (backing, regions) in grouped {
        let result = (|| {
            let regions = regions.into_iter().collect::<Vec<_>>();
            let destination = resources
                .execution_representation_id(backing)
                .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?;
            let snapshot = resources
                .snapshot_content(backing, &regions)
                .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?;
            if resources
                .representation_matches(backing, destination, &snapshot)
                .map_err(|reason| ContentSynchronizationError::Backing { backing, reason })?
            {
                return Ok(());
            }
            let route = resources
                .representation_route(backing, destination)
                .expect("the execution representation has an owned route");
            let source = match route {
                RepresentationRoute::ImportedGuestTransfer { .. } => GUEST_REPRESENTATION,
                RepresentationRoute::HostStagingTransfer { .. } => {
                    for write in snapshot.iter().copied() {
                        let ingress =
                            resources
                                .plan_host_ingress(backing, write)
                                .map_err(|reason| ContentSynchronizationError::Backing {
                                    backing,
                                    reason,
                                })?;
                        host_ingresses.push(ingress);
                        deferred.push(HostIngressTransfer {
                            ingress,
                            destination,
                        });
                    }
                    HOST_REPRESENTATION
                }
                route => {
                    return Err(ContentSynchronizationError::RouteCannotSynchronize {
                        backing,
                        route,
                    });
                }
            };
            if source == GUEST_REPRESENTATION {
                transfers.extend(
                    resources
                        .plan_transfers(backing, source, destination, &snapshot)
                        .map_err(|reason| ContentSynchronizationError::Backing {
                            backing,
                            reason,
                        })?,
                );
            }
            uses.push(RepresentationUse {
                backing,
                representations: Box::new([source, destination]),
            });
            Ok(())
        })();
        if let Err(error) = result {
            return match cancel_partial(resources, transaction, &transfers, &host_ingresses, &[]) {
                Ok(_) => Err(error),
                Err(reason) => Err(ContentSynchronizationError::Cancellation(reason)),
            };
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
    use crate::{LinearRange, ResolvedResourceLifecycle, ResourceLifecycleEffect, StorageBacking};
    use reims_vgpu_protocol::VulkanDeviceEpochId;

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
    }
}
