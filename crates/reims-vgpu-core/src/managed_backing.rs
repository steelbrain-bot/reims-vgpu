//! Native representation lifecycles owned by canonical backing lifetimes.
//!
//! Accepted transaction uses are retained before they have a queue timeline
//! point. Once the queue accepts a submission, that obligation moves to the
//! exact queue point. Backing retirement therefore cannot race either parked
//! accepted work or submitted work, and device loss abandons native ownership
//! without manufacturing successful completion.

use crate::{
    BackingRegion, ContentAuthority, ContentAuthorityError, GpuWriteId, HostIngressKey,
    HostIngressTransfer, HostLandingKey, NativeRetirement, NativeRetirementDisposition,
    NativeRetirementError, QueueTimelinePoint, RegionVersion, RepresentationRoute,
    ResourceValidity, TransferKey, GUEST_REPRESENTATION, HOST_REPRESENTATION,
};
#[cfg(test)]
use reims_vgpu_protocol::SubmissionId;
use reims_vgpu_protocol::{
    BackingId, QueueOwnerId, QueueTimelineValue, RepresentationId, ResourceValidityOps,
    TransactionId, VulkanDeviceEpochId,
};
use std::collections::{BTreeMap, BTreeSet};

const FIRST_NATIVE_REPRESENTATION: u64 = HOST_REPRESENTATION.get() + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackingLifecycle {
    Live,
    Retiring,
}

#[derive(Debug)]
struct NativeRepresentation<T> {
    route: RepresentationRoute,
    native: Option<T>,
    last_uses: BTreeMap<QueueOwnerId, QueueTimelinePoint>,
}

#[derive(Debug)]
struct ManagedBacking<T> {
    authority: ContentAuthority,
    lifecycle: BackingLifecycle,
    validity: ResourceValidity,
    representations: BTreeMap<RepresentationId, NativeRepresentation<T>>,
    execution_representation: Option<RepresentationId>,
    retiring_representations: BTreeSet<RepresentationId>,
    accepted_uses: BTreeMap<TransactionId, BTreeMap<RepresentationId, usize>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedBackingError {
    DuplicateBacking,
    UnknownBacking,
    AuthorityMismatch,
    BackingRetiring,
    AcceptedUseCountExhausted,
    UnknownAcceptedUse,
    EmptyRepresentationSet,
    UnknownRepresentation,
    DuplicateRepresentation,
    DuplicateExecutionRepresentation,
    MissingExecutionRepresentation,
    ExecutionRepresentationAlreadyRetiring,
    StaleExecutionRepresentation,
    RepresentationIdentityExhausted,
    MixedEpochs,
    TimelineRegressed,
    Content(ContentAuthorityError),
    Retirement(NativeRetirementError),
}

#[derive(Debug)]
pub struct ManagedRepresentationFailure<T> {
    pub reason: ManagedBackingError,
    pub native: T,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuWriteRequest {
    pub backing: BackingId,
    pub representation: RepresentationId,
    pub regions: Box<[BackingRegion]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuWriteReservation {
    pub backing: BackingId,
    pub write: GpuWriteId,
    pub representation: RepresentationId,
    pub regions: Box<[RegionVersion]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationUse {
    pub backing: BackingId,
    pub representations: Box<[RepresentationId]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuWriteBatchError {
    EmptyRegions(BackingId),
    DuplicateBacking(BackingId),
    DuplicateReservation {
        backing: BackingId,
        write: GpuWriteId,
    },
    Backing {
        backing: BackingId,
        reason: ManagedBackingError,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferBatchError {
    Duplicate(TransferKey),
    Transfer {
        key: TransferKey,
        reason: ManagedBackingError,
    },
}

#[derive(Debug, Eq, PartialEq)]
pub enum ManagedBackingProgress<T> {
    Live,
    WaitingForAcceptedUses,
    RepresentationsRetired { ready: Vec<T>, deferred: usize },
    RetirementStarted { ready: Vec<T>, deferred: usize },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ManagedBackingCensus {
    pub live_backings: usize,
    pub retiring_backings: usize,
    pub live_representations: usize,
    pub accepted_uses: usize,
    pub deferred_representations: usize,
}

/// One Vulkan-epoch owner for all native representations of managed backings.
#[derive(Debug)]
pub struct ManagedBackingOwner<T> {
    epoch: VulkanDeviceEpochId,
    next_representation: u64,
    backings: BTreeMap<BackingId, ManagedBacking<T>>,
    retirement: NativeRetirement<(BackingId, RepresentationId), T>,
}

impl<T> ManagedBackingOwner<T> {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            epoch,
            next_representation: FIRST_NATIVE_REPRESENTATION,
            backings: BTreeMap::new(),
            retirement: NativeRetirement::new(epoch),
        }
    }

    pub fn register_backing(
        &mut self,
        backing: BackingId,
        authority: ContentAuthority,
    ) -> Result<(), ManagedBackingError> {
        if authority.backing() != Some(backing) {
            return Err(ManagedBackingError::AuthorityMismatch);
        }
        if self.backings.contains_key(&backing) {
            return Err(ManagedBackingError::DuplicateBacking);
        }
        self.backings.insert(
            backing,
            ManagedBacking {
                authority,
                lifecycle: BackingLifecycle::Live,
                validity: ResourceValidity::default(),
                representations: BTreeMap::new(),
                execution_representation: None,
                retiring_representations: BTreeSet::new(),
                accepted_uses: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub fn contains_backing(&self, backing: BackingId) -> bool {
        self.backings.contains_key(&backing)
    }

    pub fn validity(&self, backing: BackingId) -> Option<ResourceValidity> {
        self.backings.get(&backing).map(|record| record.validity)
    }

    pub fn apply_validity(
        &mut self,
        backing: BackingId,
        ops: ResourceValidityOps,
    ) -> Result<ResourceValidity, ManagedBackingError> {
        let record = self.live_backing_mut(backing)?;
        record.validity.apply(ops);
        Ok(record.validity)
    }

    pub fn complete_validity(
        &mut self,
        backing: BackingId,
        ops: ResourceValidityOps,
    ) -> Result<ResourceValidity, ManagedBackingError> {
        let record = self
            .backings
            .get_mut(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record.validity.apply(ops);
        Ok(record.validity)
    }

    pub fn validate_reservations(
        &self,
        backing: BackingId,
        count: usize,
    ) -> Result<(), ManagedBackingError> {
        self.live_backing(backing)?
            .authority
            .validate_reservations(count)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_guest_write(&self, backing: BackingId) -> Result<(), ManagedBackingError> {
        self.live_backing(backing)?
            .authority
            .validate_guest_write_region(GUEST_REPRESENTATION)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_plan_gpu_write(
        &self,
        backing: BackingId,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        region_count: usize,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if !known_representation(record, representation) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .validate_plan_gpu_write_regions(write, representation, region_count)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_plan_transfers(
        &self,
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if !known_representation(record, source) || !known_representation(record, destination) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .validate_plan_transfers(source, destination, snapshot)
            .map_err(ManagedBackingError::Content)
    }

    fn live_backing(&self, backing: BackingId) -> Result<&ManagedBacking<T>, ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if record.lifecycle != BackingLifecycle::Live {
            return Err(ManagedBackingError::BackingRetiring);
        }
        Ok(record)
    }

    fn live_backing_mut(
        &mut self,
        backing: BackingId,
    ) -> Result<&mut ManagedBacking<T>, ManagedBackingError> {
        let record = self
            .backings
            .get_mut(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if record.lifecycle != BackingLifecycle::Live {
            return Err(ManagedBackingError::BackingRetiring);
        }
        Ok(record)
    }

    pub fn validate_begin_retirement(&self, backing: BackingId) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if record.lifecycle == BackingLifecycle::Retiring {
            return Err(ManagedBackingError::BackingRetiring);
        }
        Ok(())
    }

    pub fn create_representation(
        &mut self,
        backing: BackingId,
        route: RepresentationRoute,
        native: T,
    ) -> Result<RepresentationId, ManagedRepresentationFailure<T>> {
        let Some(record) = self.backings.get_mut(&backing) else {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::UnknownBacking,
                native,
            });
        };
        if record.lifecycle != BackingLifecycle::Live {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::BackingRetiring,
                native,
            });
        }
        let representation = if route == RepresentationRoute::DirectGuestAlias {
            GUEST_REPRESENTATION
        } else if route == RepresentationRoute::HostStagingEndpoint {
            HOST_REPRESENTATION
        } else {
            let representation = RepresentationId::new(self.next_representation);
            let Some(next_representation) = self.next_representation.checked_add(1) else {
                return Err(ManagedRepresentationFailure {
                    reason: ManagedBackingError::RepresentationIdentityExhausted,
                    native,
                });
            };
            self.next_representation = next_representation;
            representation
        };
        if record.representations.contains_key(&representation) {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::DuplicateRepresentation,
                native,
            });
        }
        record.authority.ensure_representation(representation);
        record.representations.insert(
            representation,
            NativeRepresentation {
                route,
                native: Some(native),
                last_uses: BTreeMap::new(),
            },
        );
        Ok(representation)
    }

    /// Create and designate the one native representation used for command
    /// execution on this backing. The designation is owned by the backing
    /// lifetime rather than reconstructed from route, handle kind, or insertion
    /// order. Validation precedes native ownership transfer, so a duplicate
    /// designation returns the caller's object unchanged.
    pub fn create_execution_representation(
        &mut self,
        backing: BackingId,
        route: RepresentationRoute,
        native: T,
    ) -> Result<RepresentationId, ManagedRepresentationFailure<T>> {
        let Some(record) = self.backings.get(&backing) else {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::UnknownBacking,
                native,
            });
        };
        if record.lifecycle != BackingLifecycle::Live {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::BackingRetiring,
                native,
            });
        }
        if record.execution_representation.is_some() {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::DuplicateExecutionRepresentation,
                native,
            });
        }
        let representation = self.create_representation(backing, route, native)?;
        self.backings
            .get_mut(&backing)
            .expect("representation creation retained its live backing")
            .execution_representation = Some(representation);
        Ok(representation)
    }

    /// Exact execution identity and native object selected at construction.
    pub fn execution_representation(&self, backing: BackingId) -> Option<(RepresentationId, &T)> {
        let record = self.backings.get(&backing)?;
        let representation = record.execution_representation?;
        Some((
            representation,
            record
                .representations
                .get(&representation)?
                .native
                .as_ref()?,
        ))
    }

    pub fn representation(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<&T> {
        self.backings
            .get(&backing)?
            .representations
            .get(&representation)
            .and_then(|record| record.native.as_ref())
    }

    /// Mutable access to one exact backing-owned representation. Construction
    /// uses this only to install derived native objects whose Vulkan parent is
    /// the already-owned representation.
    pub fn representation_mut(
        &mut self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<&mut T> {
        self.backings
            .get_mut(&backing)?
            .representations
            .get_mut(&representation)
            .and_then(|record| record.native.as_mut())
    }

    pub fn execution_representation_id(
        &self,
        backing: BackingId,
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.live_backing(backing)?
            .execution_representation
            .ok_or(ManagedBackingError::MissingExecutionRepresentation)
    }

    /// Revoke the construction-designated execution object for one physical
    /// incarnation. Existing accepted and submitted uses keep the old object
    /// alive, but no later preparation can select it and a subsequent
    /// materialization installs a fresh representation identity.
    pub fn replace_execution_representation(
        &mut self,
        backing: BackingId,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.validate_replace_execution_representation(backing)?;
        let record = self.live_backing_mut(backing)?;
        let Some(representation) = record.execution_representation.take() else {
            return Ok(ManagedBackingProgress::Live);
        };
        if !record.retiring_representations.insert(representation) {
            return Err(ManagedBackingError::ExecutionRepresentationAlreadyRetiring);
        }
        self.finish_retirement_if_ready(backing)
    }

    pub fn validate_replace_execution_representation(
        &self,
        backing: BackingId,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if record
            .execution_representation
            .is_some_and(|representation| record.retiring_representations.contains(&representation))
        {
            return Err(ManagedBackingError::ExecutionRepresentationAlreadyRetiring);
        }
        Ok(())
    }

    /// Select the construction-designated execution object only when it holds
    /// the exact regional content snapshot required by an immutable operation.
    pub fn execution_representation_for_snapshot(
        &self,
        backing: BackingId,
        snapshot: &[RegionVersion],
    ) -> Result<(RepresentationId, &T), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        let representation = record
            .execution_representation
            .ok_or(ManagedBackingError::MissingExecutionRepresentation)?;
        if !record
            .authority
            .representation_matches(representation, snapshot)
        {
            return Err(ManagedBackingError::StaleExecutionRepresentation);
        }
        Ok((
            representation,
            record
                .representations
                .get(&representation)
                .expect("execution identity is created with its backing")
                .native
                .as_ref()
                .expect("execution identity retains its native object"),
        ))
    }

    pub fn representation_route(
        &self,
        backing: BackingId,
        representation: RepresentationId,
    ) -> Option<RepresentationRoute> {
        self.backings
            .get(&backing)?
            .representations
            .get(&representation)
            .and_then(|record| record.native.as_ref().map(|_| record.route))
    }

    pub fn snapshot_content(
        &self,
        backing: BackingId,
        regions: &[BackingRegion],
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        Ok(record.authority.snapshot_regions(regions))
    }

    pub fn representation_matches(
        &self,
        backing: BackingId,
        representation: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<bool, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if !known_representation(record, representation) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        Ok(record
            .authority
            .representation_matches(representation, snapshot))
    }

    pub fn guest_write(
        &mut self,
        backing: BackingId,
        region: BackingRegion,
    ) -> Result<RegionVersion, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        record
            .authority
            .guest_write_region(GUEST_REPRESENTATION, region)
            .map_err(ManagedBackingError::Content)
    }

    pub fn plan_gpu_write(
        &mut self,
        backing: BackingId,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        record
            .authority
            .plan_gpu_write_regions(write, representation, regions)
            .map_err(ManagedBackingError::Content)
    }

    pub fn complete_gpu_write(
        &mut self,
        backing: BackingId,
        write: impl Into<GpuWriteId> + Copy,
        representation: RepresentationId,
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        self.validate_complete_gpu_write(backing, write, representation)?;
        let record = self
            .backings
            .get(&backing)
            .expect("GPU-write backing was prevalidated");
        record
            .authority
            .complete_gpu_write_regions(write, representation)
            .map_err(ManagedBackingError::Content)
    }

    /// Cancel a GPU-write reservation before it produces a queue completion
    /// fact. This changes no canonical or representation coverage.
    pub fn cancel_gpu_write(
        &mut self,
        backing: BackingId,
        write: impl Into<GpuWriteId>,
    ) -> Result<Box<[RegionVersion]>, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        record
            .authority
            .cancel_gpu_write_regions(write)
            .map_err(ManagedBackingError::Content)
    }

    /// Reserve every destination write as one semantic transition. Validation
    /// covers the complete unique backing set before any version is consumed.
    pub fn plan_gpu_writes(
        &mut self,
        write: impl Into<GpuWriteId> + Copy,
        requests: impl Into<Box<[GpuWriteRequest]>>,
    ) -> Result<Box<[GpuWriteReservation]>, GpuWriteBatchError> {
        let requests = requests.into();
        self.validate_plan_gpu_writes(write, &requests)?;
        Ok(requests
            .iter()
            .map(|request| GpuWriteReservation {
                backing: request.backing,
                write: write.into(),
                representation: request.representation,
                regions: self
                    .plan_gpu_write(
                        request.backing,
                        write,
                        request.representation,
                        request.regions.clone(),
                    )
                    .expect("the complete write batch was prevalidated"),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    pub fn validate_plan_gpu_writes(
        &self,
        write: impl Into<GpuWriteId> + Copy,
        requests: &[GpuWriteRequest],
    ) -> Result<(), GpuWriteBatchError> {
        let mut unique = BTreeSet::new();
        for request in requests.iter() {
            if request.regions.is_empty() {
                return Err(GpuWriteBatchError::EmptyRegions(request.backing));
            }
            if !unique.insert(request.backing) {
                return Err(GpuWriteBatchError::DuplicateBacking(request.backing));
            }
            self.validate_plan_gpu_write(
                request.backing,
                write,
                request.representation,
                request.regions.len(),
            )
            .map_err(|reason| GpuWriteBatchError::Backing {
                backing: request.backing,
                reason,
            })?;
        }
        Ok(())
    }

    /// Cancel the exact reservation tokens returned by [`Self::plan_gpu_writes`].
    /// Every token is checked before any pending write is removed.
    pub fn cancel_gpu_writes(
        &mut self,
        reservations: &[GpuWriteReservation],
    ) -> Result<(), GpuWriteBatchError> {
        self.validate_cancel_gpu_writes(reservations)?;
        for reservation in reservations {
            self.cancel_gpu_write(reservation.backing, reservation.write)
                .expect("the complete cancellation batch was prevalidated");
        }
        Ok(())
    }

    pub fn validate_cancel_gpu_writes(
        &self,
        reservations: &[GpuWriteReservation],
    ) -> Result<(), GpuWriteBatchError> {
        let mut unique = BTreeSet::new();
        for reservation in reservations {
            if !unique.insert((reservation.backing, reservation.write)) {
                return Err(GpuWriteBatchError::DuplicateReservation {
                    backing: reservation.backing,
                    write: reservation.write,
                });
            }
            let record = self.live_backing(reservation.backing).map_err(|reason| {
                GpuWriteBatchError::Backing {
                    backing: reservation.backing,
                    reason,
                }
            })?;
            record
                .authority
                .validate_gpu_write_reservation(
                    reservation.write,
                    reservation.representation,
                    &reservation.regions,
                )
                .map_err(|reason| GpuWriteBatchError::Backing {
                    backing: reservation.backing,
                    reason: ManagedBackingError::Content(reason),
                })?;
        }
        Ok(())
    }

    pub fn validate_complete_gpu_write(
        &self,
        backing: BackingId,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if !record.representations.contains_key(&representation) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .validate_complete_gpu_write_regions(write, representation)
            .map_err(ManagedBackingError::Content)
    }

    pub fn plan_transfers(
        &mut self,
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if !known_representation(record, source) || !known_representation(record, destination) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .plan_transfers(source, destination, snapshot)
            .map_err(ManagedBackingError::Content)
    }

    pub fn plan_transfer_demands(
        &mut self,
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if !known_representation(record, source) || !known_representation(record, destination) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .plan_transfer_demands(source, destination, snapshot)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_plan_transfer_demands(
        &self,
        backing: BackingId,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if !known_representation(record, source) || !known_representation(record, destination) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .validate_plan_transfer_demands(source, destination, snapshot)
            .map_err(ManagedBackingError::Content)
    }

    pub fn complete_transfer(&mut self, key: TransferKey) -> Result<(), ManagedBackingError> {
        self.validate_complete_transfer(key)?;
        let record = self
            .backings
            .get(&key.backing)
            .expect("transfer backing was prevalidated");
        record
            .authority
            .complete_transfer(key)
            .map_err(ManagedBackingError::Content)
    }

    pub fn plan_host_landing(
        &mut self,
        staged_transfer: TransferKey,
    ) -> Result<HostLandingKey, ManagedBackingError> {
        let record = self
            .backings
            .get(&staged_transfer.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record
            .authority
            .plan_host_landing(staged_transfer)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_complete_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&landing.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record
            .authority
            .validate_complete_host_landing(landing)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_host_landing_pending(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&landing.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record
            .authority
            .validate_host_landing_pending(landing)
            .map_err(ManagedBackingError::Content)
    }

    pub fn complete_host_landing(
        &mut self,
        landing: HostLandingKey,
    ) -> Result<(), ManagedBackingError> {
        self.validate_complete_host_landing(landing)?;
        self.backings
            .get(&landing.backing)
            .expect("host landing backing was prevalidated")
            .authority
            .complete_host_landing(landing)
            .map_err(ManagedBackingError::Content)
    }

    pub fn cancel_host_landing(
        &mut self,
        landing: HostLandingKey,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&landing.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record
            .authority
            .cancel_host_landing(landing)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_cancel_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&landing.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record
            .authority
            .validate_cancel_host_landing(landing)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_cancel_host_landing_demands(
        &self,
        landing: HostLandingKey,
        count: usize,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&landing.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        record
            .authority
            .validate_cancel_host_landing_demands(landing, count)
            .map_err(ManagedBackingError::Content)
    }

    pub fn plan_host_ingress(
        &mut self,
        backing: BackingId,
        write: RegionVersion,
    ) -> Result<HostIngressKey, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        record
            .authority
            .plan_host_ingress(write)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_complete_host_ingress(
        &self,
        ingress: HostIngressKey,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(ingress.backing)?;
        record
            .authority
            .validate_complete_host_ingress(ingress)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_host_ingress_transfer(
        &self,
        transfer: HostIngressTransfer,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(transfer.ingress.backing)?;
        if !known_representation(record, transfer.destination) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .validate_host_ingress_transfer(transfer)
            .map_err(ManagedBackingError::Content)
    }

    pub fn complete_host_ingress(
        &mut self,
        ingress: HostIngressKey,
    ) -> Result<(), ManagedBackingError> {
        self.validate_complete_host_ingress(ingress)?;
        self.backings
            .get(&ingress.backing)
            .expect("host ingress backing was prevalidated")
            .authority
            .complete_host_ingress(ingress)
            .map_err(ManagedBackingError::Content)
    }

    pub fn cancel_host_ingress(
        &mut self,
        ingress: HostIngressKey,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(ingress.backing)?;
        record
            .authority
            .cancel_host_ingress(ingress)
            .map_err(ManagedBackingError::Content)
    }

    pub fn validate_cancel_host_ingress_demands(
        &self,
        ingress: HostIngressKey,
        count: usize,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(ingress.backing)?;
        record
            .authority
            .validate_cancel_host_ingress_demands(ingress, count)
            .map_err(ManagedBackingError::Content)
    }

    pub fn cancel_transfers(
        &mut self,
        transfers: &[TransferKey],
    ) -> Result<(), TransferBatchError> {
        self.validate_cancel_transfers(transfers)?;
        for &key in transfers {
            self.backings
                .get(&key.backing)
                .expect("transfer backing was prevalidated")
                .authority
                .cancel_transfer(key)
                .expect("the complete transfer cancellation batch was prevalidated");
        }
        Ok(())
    }

    pub fn validate_cancel_transfers(
        &self,
        transfers: &[TransferKey],
    ) -> Result<(), TransferBatchError> {
        let mut demands = BTreeMap::<TransferKey, usize>::new();
        for &key in transfers {
            let count = demands.entry(key).or_default();
            *count = count.checked_add(1).ok_or(TransferBatchError::Transfer {
                key,
                reason: ManagedBackingError::Content(
                    crate::ContentAuthorityError::TransferDemandCountExhausted,
                ),
            })?;
        }
        for (key, count) in demands {
            let record = self
                .backings
                .get(&key.backing)
                .ok_or(TransferBatchError::Transfer {
                    key,
                    reason: ManagedBackingError::UnknownBacking,
                })?;
            record
                .authority
                .validate_cancel_transfer_demands(key, count)
                .map_err(|reason| TransferBatchError::Transfer {
                    key,
                    reason: ManagedBackingError::Content(reason),
                })?;
        }
        Ok(())
    }

    pub fn validate_complete_transfer(&self, key: TransferKey) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&key.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if !known_representation(record, key.destination) {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        record
            .authority
            .validate_complete_transfer(key)
            .map_err(ManagedBackingError::Content)
    }

    pub fn discard(
        &mut self,
        backing: BackingId,
        region: BackingRegion,
    ) -> Result<(), ManagedBackingError> {
        self.validate_discard(backing)?;
        let record = self
            .backings
            .get(&backing)
            .expect("discard backing was prevalidated");
        record.authority.discard(region);
        Ok(())
    }

    pub fn validate_discard(&self, backing: BackingId) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if record.lifecycle != BackingLifecycle::Live {
            return Err(ManagedBackingError::BackingRetiring);
        }
        Ok(())
    }

    /// Retain every native representation resolved for one accepted
    /// transaction before native submission exists.
    pub fn accept_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
        representations: impl IntoIterator<Item = RepresentationId>,
    ) -> Result<(), ManagedBackingError> {
        let representations = representations.into_iter().collect::<BTreeSet<_>>();
        self.validate_accept_use(backing, transaction, &representations)?;
        let accepted = self
            .backings
            .get_mut(&backing)
            .expect("accepted-use backing was prevalidated")
            .accepted_uses
            .entry(transaction)
            .or_default();
        for representation in representations {
            *accepted.entry(representation).or_default() += 1;
        }
        Ok(())
    }

    pub fn validate_accept_use(
        &self,
        backing: BackingId,
        transaction: TransactionId,
        representations: &BTreeSet<RepresentationId>,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if representations.is_empty() {
            return Err(ManagedBackingError::EmptyRepresentationSet);
        }
        if representations
            .iter()
            .any(|id| !record.representations.contains_key(id))
        {
            return Err(ManagedBackingError::UnknownRepresentation);
        }
        if record
            .accepted_uses
            .get(&transaction)
            .is_some_and(|accepted| {
                representations
                    .iter()
                    .any(|representation| accepted.get(representation) == Some(&usize::MAX))
            })
        {
            return Err(ManagedBackingError::AcceptedUseCountExhausted);
        }
        Ok(())
    }

    pub fn accept_uses(
        &mut self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<(), crate::ResourceUseBatchError> {
        let normalized = self.normalized_accept_uses(transaction, uses)?;
        for (backing, representations) in normalized {
            let accepted = self
                .backings
                .get_mut(&backing)
                .expect("the complete accepted-use batch was prevalidated")
                .accepted_uses
                .entry(transaction)
                .or_default();
            for representation in representations {
                *accepted.entry(representation).or_default() += 1;
            }
        }
        Ok(())
    }

    pub fn validate_accept_uses(
        &self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<(), crate::ResourceUseBatchError> {
        self.normalized_accept_uses(transaction, uses).map(drop)
    }

    fn normalized_accept_uses(
        &self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<Vec<(BackingId, BTreeSet<RepresentationId>)>, crate::ResourceUseBatchError> {
        let mut unique = BTreeSet::new();
        uses.iter()
            .map(|use_| {
                if !unique.insert(use_.backing) {
                    return Err(crate::ResourceUseBatchError::DuplicateBacking(use_.backing));
                }
                let representations = use_
                    .representations
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                self.validate_accept_use(use_.backing, transaction, &representations)
                    .map_err(|reason| crate::ResourceUseBatchError::Backing {
                        backing: use_.backing,
                        reason,
                    })?;
                Ok((use_.backing, representations))
            })
            .collect()
    }

    /// Move one accepted use to its exact native queue completion obligation.
    pub fn validate_submit_use(
        &self,
        backing: BackingId,
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<(), ManagedBackingError> {
        if point.epoch != self.epoch {
            return Err(ManagedBackingError::MixedEpochs);
        }
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        let representations = record
            .accepted_uses
            .get(&transaction)
            .ok_or(ManagedBackingError::UnknownAcceptedUse)?;
        for representation in representations.keys() {
            let native = record.representations.get(representation).unwrap();
            if native
                .last_uses
                .get(&point.queue)
                .is_some_and(|previous| previous.value > point.value)
            {
                return Err(ManagedBackingError::TimelineRegressed);
            }
        }
        if record.lifecycle == BackingLifecycle::Retiring && record.accepted_uses.len() == 1 {
            for (&representation, native) in &record.representations {
                let obligations = native
                    .last_uses
                    .values()
                    .copied()
                    .chain(
                        representations
                            .contains_key(&representation)
                            .then_some(point),
                    )
                    .collect::<BTreeSet<_>>();
                self.retirement
                    .validate_defer(&(backing, representation), &obligations)
                    .map_err(ManagedBackingError::Retirement)?;
            }
        }
        Ok(())
    }

    /// Move one accepted use to its exact native queue completion obligation.
    pub fn submit_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
        point: QueueTimelinePoint,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.validate_submit_use(backing, transaction, point)?;
        let record = self
            .backings
            .get_mut(&backing)
            .expect("validated backing remains owned during the transition");
        let representations = record
            .accepted_uses
            .get(&transaction)
            .cloned()
            .expect("validated accepted use remains owned during the transition");
        record.accepted_uses.remove(&transaction);
        for representation in representations.into_keys() {
            record
                .representations
                .get_mut(&representation)
                .unwrap()
                .last_uses
                .insert(point.queue, point);
        }
        self.finish_retirement_if_ready(backing)
    }

    /// Cancel accepted work which never reached a native queue. Cancellation
    /// is not completion and creates no timeline obligation.
    pub fn validate_cancel_use(
        &self,
        backing: BackingId,
        transaction: TransactionId,
    ) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if !record.accepted_uses.contains_key(&transaction) {
            return Err(ManagedBackingError::UnknownAcceptedUse);
        }
        if record.lifecycle == BackingLifecycle::Retiring && record.accepted_uses.len() == 1 {
            for (&representation, native) in &record.representations {
                let obligations = native.last_uses.values().copied().collect::<BTreeSet<_>>();
                self.retirement
                    .validate_defer(&(backing, representation), &obligations)
                    .map_err(ManagedBackingError::Retirement)?;
            }
        }
        Ok(())
    }

    /// Cancel accepted work which never reached a native queue. Cancellation
    /// is not completion and creates no timeline obligation.
    pub fn cancel_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.validate_cancel_use(backing, transaction)?;
        let record = self
            .backings
            .get_mut(&backing)
            .expect("validated backing remains owned during the transition");
        record.accepted_uses.remove(&transaction);
        self.finish_retirement_if_ready(backing)
    }

    /// Cancel one preparation's contribution to a transaction-wide accepted
    /// use. Repeated operations may retain the same representation; only the
    /// exact contribution being cancelled is removed.
    pub fn validate_cancel_representation_use(
        &self,
        backing: BackingId,
        transaction: TransactionId,
        representations: &BTreeSet<RepresentationId>,
    ) -> Result<(), ManagedBackingError> {
        if representations.is_empty() {
            return Err(ManagedBackingError::EmptyRepresentationSet);
        }
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        let accepted = record
            .accepted_uses
            .get(&transaction)
            .ok_or(ManagedBackingError::UnknownAcceptedUse)?;
        if representations
            .iter()
            .any(|representation| !accepted.contains_key(representation))
        {
            return Err(ManagedBackingError::UnknownAcceptedUse);
        }
        let removes_transaction = accepted
            .iter()
            .all(|(representation, count)| *count == 1 && representations.contains(representation));
        if record.lifecycle == BackingLifecycle::Retiring
            && record.accepted_uses.len() == 1
            && removes_transaction
        {
            for (&representation, native) in &record.representations {
                let obligations = native.last_uses.values().copied().collect::<BTreeSet<_>>();
                self.retirement
                    .validate_defer(&(backing, representation), &obligations)
                    .map_err(ManagedBackingError::Retirement)?;
            }
        }
        Ok(())
    }

    pub fn cancel_representation_use(
        &mut self,
        backing: BackingId,
        transaction: TransactionId,
        representations: &BTreeSet<RepresentationId>,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.validate_cancel_representation_use(backing, transaction, representations)?;
        let record = self.backings.get_mut(&backing).unwrap();
        let accepted = record.accepted_uses.get_mut(&transaction).unwrap();
        for representation in representations {
            let count = accepted.get_mut(representation).unwrap();
            *count -= 1;
            if *count == 0 {
                accepted.remove(representation);
            }
        }
        if accepted.is_empty() {
            record.accepted_uses.remove(&transaction);
        }
        self.finish_retirement_if_ready(backing)
    }

    pub fn validate_cancel_representation_uses(
        &self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<(), crate::ResourceUseBatchError> {
        let mut demands = BTreeMap::<BackingId, BTreeMap<RepresentationId, usize>>::new();
        for use_ in uses {
            let representations = use_
                .representations
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            if representations.is_empty() {
                return Err(crate::ResourceUseBatchError::Backing {
                    backing: use_.backing,
                    reason: ManagedBackingError::EmptyRepresentationSet,
                });
            }
            for representation in representations {
                let count = demands
                    .entry(use_.backing)
                    .or_default()
                    .entry(representation)
                    .or_default();
                *count = count
                    .checked_add(1)
                    .ok_or(crate::ResourceUseBatchError::Backing {
                        backing: use_.backing,
                        reason: ManagedBackingError::AcceptedUseCountExhausted,
                    })?;
            }
        }
        for (backing, requested) in demands {
            let record =
                self.backings
                    .get(&backing)
                    .ok_or(crate::ResourceUseBatchError::Backing {
                        backing,
                        reason: ManagedBackingError::UnknownBacking,
                    })?;
            let accepted = record.accepted_uses.get(&transaction).ok_or(
                crate::ResourceUseBatchError::Backing {
                    backing,
                    reason: ManagedBackingError::UnknownAcceptedUse,
                },
            )?;
            if requested.iter().any(|(representation, count)| {
                accepted.get(representation).is_none_or(|held| held < count)
            }) {
                return Err(crate::ResourceUseBatchError::Backing {
                    backing,
                    reason: ManagedBackingError::UnknownAcceptedUse,
                });
            }
            let removes_transaction = accepted.len() == requested.len()
                && accepted
                    .iter()
                    .all(|(representation, count)| requested.get(representation) == Some(count));
            if record.lifecycle == BackingLifecycle::Retiring
                && record.accepted_uses.len() == 1
                && removes_transaction
            {
                for (&representation, native) in &record.representations {
                    let obligations = native.last_uses.values().copied().collect::<BTreeSet<_>>();
                    self.retirement
                        .validate_defer(&(backing, representation), &obligations)
                        .map_err(|reason| crate::ResourceUseBatchError::Backing {
                            backing,
                            reason: ManagedBackingError::Retirement(reason),
                        })?;
                }
            }
        }
        Ok(())
    }

    pub fn cancel_representation_uses(
        &mut self,
        transaction: TransactionId,
        uses: &[RepresentationUse],
    ) -> Result<Vec<(BackingId, ManagedBackingProgress<T>)>, crate::ResourceUseBatchError> {
        self.validate_cancel_representation_uses(transaction, uses)?;
        let mut demands = BTreeMap::<BackingId, BTreeMap<RepresentationId, usize>>::new();
        for use_ in uses {
            for representation in use_
                .representations
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
            {
                *demands
                    .entry(use_.backing)
                    .or_default()
                    .entry(representation)
                    .or_default() += 1;
            }
        }
        Ok(demands
            .into_iter()
            .map(|(backing, requested)| {
                let record = self.backings.get_mut(&backing).unwrap();
                let accepted = record.accepted_uses.get_mut(&transaction).unwrap();
                for (representation, requested) in requested {
                    let count = accepted.get_mut(&representation).unwrap();
                    *count -= requested;
                    if *count == 0 {
                        accepted.remove(&representation);
                    }
                }
                if accepted.is_empty() {
                    record.accepted_uses.remove(&transaction);
                }
                let progress = self
                    .finish_retirement_if_ready(backing)
                    .expect("retirement was prevalidated for every complete use removal");
                (backing, progress)
            })
            .collect())
    }

    pub fn begin_retirement(
        &mut self,
        backing: BackingId,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        self.validate_begin_retirement(backing)?;
        let record = self.backings.get_mut(&backing).unwrap();
        record.lifecycle = BackingLifecycle::Retiring;
        self.finish_retirement_if_ready(backing)
    }

    fn finish_retirement_if_ready(
        &mut self,
        backing: BackingId,
    ) -> Result<ManagedBackingProgress<T>, ManagedBackingError> {
        let Some(record) = self.backings.get(&backing) else {
            return Err(ManagedBackingError::UnknownBacking);
        };
        let retiring_representations = record
            .retiring_representations
            .iter()
            .copied()
            .filter(|representation| {
                record
                    .accepted_uses
                    .values()
                    .all(|uses| !uses.contains_key(representation))
            })
            .collect::<Vec<_>>();

        for representation in &retiring_representations {
            let native = record
                .representations
                .get(representation)
                .expect("a retiring identity remains owned until retirement starts");
            if native.native.is_none() {
                continue;
            }
            let obligations = native.last_uses.values().copied().collect::<BTreeSet<_>>();
            self.retirement
                .validate_defer(&(backing, *representation), &obligations)
                .map_err(ManagedBackingError::Retirement)?;
        }

        let mut ready = Vec::new();
        let mut deferred = 0usize;
        if !retiring_representations.is_empty() {
            let record = self.backings.get_mut(&backing).unwrap();
            for representation in retiring_representations {
                record.retiring_representations.remove(&representation);
                let native = record
                    .representations
                    .get_mut(&representation)
                    .expect("validated retiring representation remains owned");
                let Some(native_object) = native.native.take() else {
                    continue;
                };
                let last_uses = std::mem::take(&mut native.last_uses);
                match self
                    .retirement
                    .defer(
                        (backing, representation),
                        native_object,
                        last_uses.into_values(),
                    )
                    .unwrap_or_else(|_| unreachable!("representation retirement was prevalidated"))
                {
                    NativeRetirementDisposition::Ready(native) => {
                        record.representations.remove(&representation);
                        record.authority.remove_representation(representation);
                        ready.push(native);
                    }
                    NativeRetirementDisposition::Deferred => deferred += 1,
                }
            }
        }

        let record = self.backings.get(&backing).unwrap();
        if record.lifecycle == BackingLifecycle::Live {
            return if ready.is_empty() && deferred == 0 {
                Ok(ManagedBackingProgress::Live)
            } else {
                Ok(ManagedBackingProgress::RepresentationsRetired { ready, deferred })
            };
        }
        if !record.accepted_uses.is_empty() {
            return if ready.is_empty() && deferred == 0 {
                Ok(ManagedBackingProgress::WaitingForAcceptedUses)
            } else {
                Ok(ManagedBackingProgress::RepresentationsRetired { ready, deferred })
            };
        }

        for (&representation, native) in &record.representations {
            if native.native.is_none() {
                continue;
            }
            let obligations = native.last_uses.values().copied().collect::<BTreeSet<_>>();
            self.retirement
                .validate_defer(&(backing, representation), &obligations)
                .map_err(ManagedBackingError::Retirement)?;
        }

        let record = self.backings.get_mut(&backing).unwrap();
        let representations = record.representations.keys().copied().collect::<Vec<_>>();
        for representation in representations {
            let native = record
                .representations
                .get_mut(&representation)
                .expect("validated representation remains owned");
            let Some(native_object) = native.native.take() else {
                continue;
            };
            let last_uses = std::mem::take(&mut native.last_uses);
            let disposition = self
                .retirement
                .defer(
                    (backing, representation),
                    native_object,
                    last_uses.into_values(),
                )
                .unwrap_or_else(|_| {
                    unreachable!(
                        "all retirement tickets and epochs were validated before ownership moved"
                    )
                });
            match disposition {
                NativeRetirementDisposition::Ready(native) => {
                    record.representations.remove(&representation);
                    record.authority.remove_representation(representation);
                    ready.push(native);
                }
                NativeRetirementDisposition::Deferred => deferred += 1,
            }
        }
        if record.representations.is_empty() {
            self.backings.remove(&backing);
        }
        Ok(ManagedBackingProgress::RetirementStarted { ready, deferred })
    }

    pub fn advance(
        &mut self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<Vec<T>, ManagedBackingError> {
        let ready = self
            .retirement
            .advance(queue, completed)
            .map_err(ManagedBackingError::Retirement)?;
        let mut native = Vec::with_capacity(ready.len());
        for ((backing, representation), object) in ready {
            let record = self
                .backings
                .get_mut(&backing)
                .expect("deferred native retirement retains its backing authority");
            let removed = record
                .representations
                .remove(&representation)
                .expect("deferred native retirement retains its representation identity");
            debug_assert!(removed.native.is_none());
            record.authority.remove_representation(representation);
            let remove_backing =
                record.lifecycle == BackingLifecycle::Retiring && record.representations.is_empty();
            if remove_backing {
                self.backings.remove(&backing);
            }
            native.push(object);
        }
        Ok(native)
    }

    pub fn validate_advance(
        &self,
        queue: QueueOwnerId,
        completed: QueueTimelineValue,
    ) -> Result<(), ManagedBackingError> {
        self.retirement
            .validate_advance(queue, completed)
            .map_err(ManagedBackingError::Retirement)
    }

    /// Take every native object whose ordinary destruction can no longer rely
    /// on successful timeline completion after device loss.
    pub fn abandon(self) -> Vec<T> {
        let mut native = self
            .backings
            .into_values()
            .flat_map(|record| {
                record
                    .representations
                    .into_values()
                    .filter_map(|representation| representation.native)
            })
            .collect::<Vec<_>>();
        native.extend(
            self.retirement
                .abandon()
                .into_iter()
                .map(|(_, native)| native),
        );
        native
    }

    pub fn census(&self) -> ManagedBackingCensus {
        ManagedBackingCensus {
            live_backings: self
                .backings
                .values()
                .filter(|record| record.lifecycle == BackingLifecycle::Live)
                .count(),
            retiring_backings: self
                .backings
                .values()
                .filter(|record| record.lifecycle == BackingLifecycle::Retiring)
                .count(),
            live_representations: self
                .backings
                .values()
                .map(|record| {
                    record
                        .representations
                        .values()
                        .filter(|representation| representation.native.is_some())
                        .count()
                })
                .sum(),
            accepted_uses: self
                .backings
                .values()
                .map(|record| record.accepted_uses.len())
                .sum(),
            deferred_representations: self.retirement.pending(),
        }
    }
}

fn known_representation<T>(backing: &ManagedBacking<T>, representation: RepresentationId) -> bool {
    representation == GUEST_REPRESENTATION
        || backing
            .representations
            .get(&representation)
            .is_some_and(|record| record.native.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackingRegion, ContentAuthority};

    fn point(queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(7),
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    fn owner() -> (ManagedBackingOwner<&'static str>, BackingId) {
        let backing = BackingId::new(3);
        let authority =
            ContentAuthority::for_backing_regions(backing, [BackingRegion::Whole]).unwrap();
        let mut owner = ManagedBackingOwner::new(VulkanDeviceEpochId::new(7));
        owner.register_backing(backing, authority).unwrap();
        (owner, backing)
    }

    #[test]
    fn execution_representation_is_an_explicit_backing_owned_relation() {
        let (mut owner, backing) = owner();
        let transfer = owner
            .create_representation(
                backing,
                RepresentationRoute::ImportedGuestTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                "transfer",
            )
            .unwrap();
        let execution = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::NativeWorking {
                    memory: crate::WorkingMemoryClass::DeviceLocal,
                },
                "execution",
            )
            .unwrap();
        assert_ne!(transfer, execution);
        assert_eq!(
            owner.execution_representation(backing),
            Some((execution, &"execution"))
        );
        let failure = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                "duplicate",
            )
            .unwrap_err();
        assert_eq!(
            failure.reason,
            ManagedBackingError::DuplicateExecutionRepresentation
        );
        assert_eq!(failure.native, "duplicate");
        assert_eq!(owner.census().live_representations, 2);
        assert_eq!(
            owner.execution_representation(backing),
            Some((execution, &"execution"))
        );
    }

    #[test]
    fn physical_replacement_revokes_selection_but_waits_for_the_exact_accepted_use() {
        let (mut owner, backing) = owner();
        let transaction = TransactionId::new(9);
        let old = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::NativeWorking {
                    memory: crate::WorkingMemoryClass::DeviceLocal,
                },
                "old",
            )
            .unwrap();
        owner.accept_use(backing, transaction, [old]).unwrap();

        assert_eq!(
            owner.replace_execution_representation(backing).unwrap(),
            ManagedBackingProgress::Live
        );
        assert_eq!(
            owner.execution_representation_id(backing),
            Err(ManagedBackingError::MissingExecutionRepresentation)
        );
        let new = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::NativeWorking {
                    memory: crate::WorkingMemoryClass::DeviceLocal,
                },
                "new",
            )
            .unwrap();
        assert_ne!(old, new);

        assert_eq!(
            owner.submit_use(backing, transaction, point(1, 4)).unwrap(),
            ManagedBackingProgress::RepresentationsRetired {
                ready: Vec::new(),
                deferred: 1,
            }
        );
        assert_eq!(owner.representation(backing, old), None);
        assert_eq!(owner.representation(backing, new), Some(&"new"));
        assert_eq!(
            owner
                .advance(QueueOwnerId::new(1), QueueTimelineValue::new(3))
                .unwrap(),
            Vec::<&str>::new()
        );
        assert_eq!(
            owner
                .advance(QueueOwnerId::new(1), QueueTimelineValue::new(4))
                .unwrap(),
            vec!["old"]
        );
    }

    #[test]
    fn execution_source_selection_requires_the_exact_content_snapshot() {
        let (mut owner, backing) = owner();
        let execution = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                "execution",
            )
            .unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        assert_eq!(
            owner.execution_representation_for_snapshot(backing, &snapshot),
            Err(ManagedBackingError::StaleExecutionRepresentation)
        );

        let transfer = owner
            .plan_transfers(backing, GUEST_REPRESENTATION, execution, &snapshot)
            .unwrap();
        owner.complete_transfer(transfer[0]).unwrap();
        assert_eq!(
            owner.execution_representation_for_snapshot(backing, &snapshot),
            Ok((execution, &"execution"))
        );

        owner.guest_write(backing, BackingRegion::Whole).unwrap();
        let newer = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        assert_eq!(
            owner.execution_representation_for_snapshot(backing, &newer),
            Err(ManagedBackingError::StaleExecutionRepresentation)
        );
    }

    #[test]
    fn cancelling_a_gpu_write_returns_the_exact_reservation_without_publishing_it() {
        let (mut owner, backing) = owner();
        let representation = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                "execution",
            )
            .unwrap();
        let submission = SubmissionId::new(11);
        let planned = owner
            .plan_gpu_write(backing, submission, representation, [BackingRegion::Whole])
            .unwrap();

        assert_eq!(
            owner.cancel_gpu_write(backing, submission).unwrap(),
            planned
        );
        assert_eq!(
            owner.complete_gpu_write(backing, submission, representation),
            Err(ManagedBackingError::Content(
                ContentAuthorityError::SubmissionDidNotPlanWrite
            ))
        );
        assert_eq!(
            owner
                .snapshot_content(backing, &[BackingRegion::Whole])
                .unwrap()[0]
                .version,
            reims_vgpu_protocol::ContentVersion::new(1)
        );
    }

    #[test]
    fn gpu_write_completion_is_bound_to_its_planned_representation() {
        let (mut owner, backing) = owner();
        let execution = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                "execution",
            )
            .unwrap();
        let other = owner
            .create_representation(
                backing,
                RepresentationRoute::ImportedGuestTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                "other",
            )
            .unwrap();
        let submission = SubmissionId::new(21);
        owner
            .plan_gpu_write(backing, submission, execution, [BackingRegion::Whole])
            .unwrap();
        assert_eq!(
            owner.complete_gpu_write(backing, submission, other),
            Err(ManagedBackingError::Content(
                ContentAuthorityError::GpuWriteRepresentationMismatch
            ))
        );
        assert!(owner
            .complete_gpu_write(backing, submission, execution)
            .is_ok());
    }

    #[test]
    fn multi_backing_write_cancellation_validates_every_exact_token_first() {
        let (mut owner, first) = owner();
        let second = BackingId::new(4);
        owner
            .register_backing(second, ContentAuthority::for_backing(second))
            .unwrap();
        let submission = SubmissionId::new(12);
        let reservations = owner
            .plan_gpu_writes(
                submission,
                [
                    GpuWriteRequest {
                        backing: first,
                        representation: GUEST_REPRESENTATION,
                        regions: Box::new([BackingRegion::Whole]),
                    },
                    GpuWriteRequest {
                        backing: second,
                        representation: GUEST_REPRESENTATION,
                        regions: Box::new([BackingRegion::Whole]),
                    },
                ],
            )
            .unwrap();
        let mut wrong = reservations.clone();
        wrong[1].regions[0].version = reims_vgpu_protocol::ContentVersion::new(99);

        assert_eq!(
            owner.cancel_gpu_writes(&wrong),
            Err(GpuWriteBatchError::Backing {
                backing: second,
                reason: ManagedBackingError::Content(
                    ContentAuthorityError::GpuWriteReservationMismatch
                ),
            })
        );
        owner.cancel_gpu_writes(&reservations).unwrap();
        assert_eq!(
            owner.cancel_gpu_writes(&reservations),
            Err(GpuWriteBatchError::Backing {
                backing: first,
                reason: ManagedBackingError::Content(
                    ContentAuthorityError::SubmissionDidNotPlanWrite
                ),
            })
        );
    }

    #[test]
    fn transfer_cancellation_validates_the_whole_batch_before_removing_any_key() {
        let (mut owner, backing) = owner();
        let destination = owner
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, "working")
            .unwrap();
        owner.guest_write(backing, BackingRegion::Whole).unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        let key = owner
            .plan_transfers(backing, GUEST_REPRESENTATION, destination, &snapshot)
            .unwrap()[0];
        let mut absent = key;
        absent.version = reims_vgpu_protocol::ContentVersion::new(key.version.get() + 1);

        assert_eq!(
            owner.cancel_transfers(&[key, absent]),
            Err(TransferBatchError::Transfer {
                key: absent,
                reason: ManagedBackingError::Content(ContentAuthorityError::TransferNotPlanned),
            })
        );
        owner.complete_transfer(key).unwrap();
        assert!(owner
            .representation_matches(backing, destination, &snapshot)
            .unwrap());
    }

    #[test]
    fn retirement_waits_first_for_accepted_use_then_for_exact_timeline() {
        let (mut owner, backing) = owner();
        let representation = owner
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, "image")
            .unwrap();
        assert_eq!(representation, GUEST_REPRESENTATION);
        owner
            .accept_use(backing, TransactionId::new(9), [representation])
            .unwrap();
        assert_eq!(
            owner.begin_retirement(backing),
            Ok(ManagedBackingProgress::WaitingForAcceptedUses)
        );
        assert_eq!(
            owner.submit_use(backing, TransactionId::new(9), point(1, 4)),
            Ok(ManagedBackingProgress::RetirementStarted {
                ready: Vec::new(),
                deferred: 1,
            })
        );
        assert!(owner
            .advance(QueueOwnerId::new(1), QueueTimelineValue::new(3))
            .unwrap()
            .is_empty());
        assert_eq!(
            owner
                .advance(QueueOwnerId::new(1), QueueTimelineValue::new(4))
                .unwrap(),
            vec!["image"]
        );
    }

    #[test]
    fn canceled_unsubmitted_use_is_not_reported_as_completion() {
        let (mut owner, backing) = owner();
        let representation = owner
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, "buffer")
            .unwrap();
        owner
            .accept_use(backing, TransactionId::new(2), [representation])
            .unwrap();
        owner.begin_retirement(backing).unwrap();
        assert_eq!(
            owner.cancel_use(backing, TransactionId::new(2)),
            Ok(ManagedBackingProgress::RetirementStarted {
                ready: vec!["buffer"],
                deferred: 0,
            })
        );
    }

    #[test]
    fn device_loss_returns_submitted_and_unsubmitted_native_ownership() {
        let (mut owner, backing) = owner();
        let first = owner
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, "first")
            .unwrap();
        let second = owner
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, "second")
            .unwrap();
        owner
            .accept_use(backing, TransactionId::new(1), [first])
            .unwrap();
        owner
            .submit_use(backing, TransactionId::new(1), point(0, 5))
            .unwrap();
        owner
            .accept_use(backing, TransactionId::new(2), [second])
            .unwrap();
        let mut abandoned = owner.abandon();
        abandoned.sort();
        assert_eq!(abandoned, vec!["first", "second"]);
    }

    #[test]
    fn failed_use_admission_changes_no_lifecycle_state() {
        let (mut owner, backing) = owner();
        let representation = owner
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, "image")
            .unwrap();
        let before = owner.census();
        assert_eq!(
            owner.accept_use(backing, TransactionId::new(1), [RepresentationId::new(999)]),
            Err(ManagedBackingError::UnknownRepresentation)
        );
        assert_eq!(owner.census(), before);
        owner
            .accept_use(backing, TransactionId::new(1), [representation])
            .unwrap();
    }

    #[test]
    fn refused_representation_creation_returns_native_ownership() {
        let mut owner = ManagedBackingOwner::new(VulkanDeviceEpochId::new(7));
        let failure = owner
            .create_representation(
                BackingId::new(99),
                RepresentationRoute::DirectGuestAlias,
                String::from("native"),
            )
            .unwrap_err();
        assert_eq!(failure.reason, ManagedBackingError::UnknownBacking);
        assert_eq!(failure.native, "native");
    }

    #[test]
    fn direct_guest_alias_uses_the_authoritative_guest_representation_once() {
        let (mut owner, backing) = owner();
        assert_eq!(
            owner
                .create_representation(backing, RepresentationRoute::DirectGuestAlias, "first",)
                .unwrap(),
            GUEST_REPRESENTATION
        );
        let failure = owner
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, "second")
            .unwrap_err();
        assert_eq!(failure.reason, ManagedBackingError::DuplicateRepresentation);
        assert_eq!(failure.native, "second");
    }

    #[test]
    fn transfer_lifecycle_uses_canonical_region_version_keys_once() {
        let backing = BackingId::new(8);
        let whole = BackingRegion::Linear(crate::LinearRange::new(0, 128).unwrap());
        let right = BackingRegion::Linear(crate::LinearRange::new(64, 64).unwrap());
        let authority = ContentAuthority::for_backing_regions(backing, [whole]).unwrap();
        let mut owner = ManagedBackingOwner::new(VulkanDeviceEpochId::new(7));
        owner.register_backing(backing, authority).unwrap();
        let working = owner
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, "working")
            .unwrap();

        let first_snapshot = owner.snapshot_content(backing, &[whole]).unwrap();
        let first = owner
            .plan_transfers(backing, GUEST_REPRESENTATION, working, &first_snapshot)
            .unwrap();
        assert_eq!(first.len(), 1);
        assert!(owner
            .plan_transfers(backing, GUEST_REPRESENTATION, working, &first_snapshot,)
            .unwrap()
            .is_empty());
        owner.complete_transfer(first[0]).unwrap();

        owner.guest_write(backing, right).unwrap();
        let newer = owner.snapshot_content(backing, &[right]).unwrap();
        let second = owner
            .plan_transfers(backing, GUEST_REPRESENTATION, working, &newer)
            .unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].region, right);
        owner.complete_transfer(second[0]).unwrap();
        assert!(owner
            .plan_transfers(backing, GUEST_REPRESENTATION, working, &newer)
            .unwrap()
            .is_empty());
    }
}
