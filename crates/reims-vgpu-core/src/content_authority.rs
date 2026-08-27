//! Region-version content authority shared by all memory topologies.
//!
//! Canonical versions belong to backing coordinates, not views or native
//! representations. Disjoint writes can therefore complete in either order,
//! while an older overlapping GPU completion cannot replace a newer canonical
//! guest write. Transfers are keyed by backing region, version, and exact
//! source/destination representations and cannot be planned twice while live.

use crate::{ImageAspect, LinearRange, TexelBox};
use reims_vgpu_protocol::{
    BackingId, ContentVersion, RepresentationId, SubmissionId, TransactionId,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Reserved representation identities used by the canonical compatibility
/// projection. Additional native representations receive their own identities.
pub const GUEST_REPRESENTATION: RepresentationId = RepresentationId::new(1);
pub const GPU_REPRESENTATION: RepresentationId = RepresentationId::new(2);
pub const HOST_REPRESENTATION: RepresentationId = RepresentationId::new(3);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ImageRegion {
    pub aspect: ImageAspect,
    pub mip: u32,
    pub layer: u32,
    pub texels: TexelBox,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BackingRegion {
    /// The complete backing when the contract does not establish a sound
    /// translation into finer backing coordinates.
    Whole,
    Linear(LinearRange),
    Image(ImageRegion),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RegionVersion {
    pub region: BackingRegion,
    pub version: ContentVersion,
}

/// Contract-owned identity of one GPU write reservation.
///
/// Standalone lifecycle operations use their submission identity. Ordered
/// EXEC operations additionally carry their flattened operation position so
/// two writes to the same backing in one submission remain distinct.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GpuWriteId {
    Submission(SubmissionId),
    Operation {
        transaction: TransactionId,
        submission: SubmissionId,
        index: usize,
    },
}

impl GpuWriteId {
    pub const fn operation(
        transaction: TransactionId,
        submission: SubmissionId,
        index: usize,
    ) -> Self {
        Self::Operation {
            transaction,
            submission,
            index,
        }
    }

    pub const fn transaction(self) -> Option<TransactionId> {
        match self {
            Self::Submission(_) => None,
            Self::Operation { transaction, .. } => Some(transaction),
        }
    }

    pub const fn submission(self) -> SubmissionId {
        match self {
            Self::Submission(submission) | Self::Operation { submission, .. } => submission,
        }
    }

    pub const fn operation_index(self) -> Option<usize> {
        match self {
            Self::Submission(_) => None,
            Self::Operation { index, .. } => Some(index),
        }
    }
}

impl From<SubmissionId> for GpuWriteId {
    fn from(submission: SubmissionId) -> Self {
        Self::Submission(submission)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TransferKey {
    pub backing: BackingId,
    pub region: BackingRegion,
    pub version: ContentVersion,
    pub source: RepresentationId,
    pub destination: RepresentationId,
}

/// A CPU landing reserved together with a GPU transfer into the canonical
/// host-staging representation. It becomes completable only after that exact
/// GPU transfer has made the staged version current.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HostLandingKey {
    pub backing: BackingId,
    pub region: BackingRegion,
    pub version: ContentVersion,
}

/// A CPU upload reserved from the authoritative guest representation into the
/// fixed host-staging representation. Completion is permitted while the exact
/// guest-authored regional version remains canonical, even if a later ordered
/// validity transition has removed guest visibility without replacing bytes.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HostIngressKey {
    pub backing: BackingId,
    pub region: BackingRegion,
    pub version: ContentVersion,
}

/// A deferred GPU upload whose HOST source becomes current only when its exact
/// CPU ingress completes.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HostIngressTransfer {
    pub ingress: HostIngressKey,
    pub destination: RepresentationId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentAuthorityError {
    EmptyBacking,
    UnboundBacking,
    VersionSpaceExhausted,
    GpuWriteAlreadyPlanned,
    SubmissionDidNotPlanWrite,
    GpuWriteReservationMismatch,
    GpuWriteRepresentationMismatch,
    UnknownRepresentation,
    StaleSource,
    TransferNotPlanned,
    InsufficientTransferDemand,
    TransferDemandCountExhausted,
    HostLandingSourceMismatch,
    HostLandingNotPlanned,
    HostLandingSourceNotCurrent,
    HostIngressNotPlanned,
    HostIngressSourceNotCurrent,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CoverageMap {
    entries: Vec<RegionVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingGpuWrite {
    representation: RepresentationId,
    regions: Box<[RegionVersion]>,
}

impl CoverageMap {
    fn assign(&mut self, region: BackingRegion, version: ContentVersion) {
        let mut next = Vec::with_capacity(self.entries.len() + 1);
        for existing in self.entries.drain(..) {
            for remainder in subtract(existing.region, region) {
                next.push(RegionVersion {
                    region: remainder,
                    version: existing.version,
                });
            }
        }
        next.push(RegionVersion { region, version });
        self.entries = next;
        self.coalesce();
    }

    fn assign_if_newer(&mut self, region: BackingRegion, version: ContentVersion) {
        let mut writable = vec![region];
        for protected in self.entries.iter().filter(|entry| entry.version >= version) {
            writable = writable
                .into_iter()
                .flat_map(|candidate| subtract(candidate, protected.region))
                .collect();
            if writable.is_empty() {
                return;
            }
        }
        for region in writable {
            self.assign(region, version);
        }
    }

    fn remove(&mut self, region: BackingRegion) {
        self.entries = self
            .entries
            .drain(..)
            .flat_map(|existing| {
                subtract(existing.region, region)
                    .into_iter()
                    .map(move |remainder| RegionVersion {
                        region: remainder,
                        version: existing.version,
                    })
            })
            .collect();
        self.coalesce();
    }

    fn intersecting(&self, region: BackingRegion) -> Vec<RegionVersion> {
        self.entries
            .iter()
            .filter_map(|entry| {
                intersection(entry.region, region).map(|region| RegionVersion {
                    region,
                    version: entry.version,
                })
            })
            .collect()
    }

    fn missing(&self, region: BackingRegion, version: ContentVersion) -> Vec<BackingRegion> {
        let mut missing = vec![region];
        for current in self.entries.iter().filter(|entry| entry.version == version) {
            missing = missing
                .into_iter()
                .flat_map(|candidate| subtract(candidate, current.region))
                .collect();
            if missing.is_empty() {
                break;
            }
        }
        missing
    }

    fn covers(&self, region: BackingRegion, version: ContentVersion) -> bool {
        self.missing(region, version).is_empty()
    }

    fn coalesce(&mut self) {
        self.entries.sort();
        loop {
            let mut merged = false;
            'outer: for left in 0..self.entries.len() {
                for right in (left + 1)..self.entries.len() {
                    if self.entries[left].version != self.entries[right].version {
                        continue;
                    }
                    if let Some(region) =
                        merge_regions(self.entries[left].region, self.entries[right].region)
                    {
                        self.entries[left].region = region;
                        self.entries.remove(right);
                        merged = true;
                        break 'outer;
                    }
                }
            }
            if !merged {
                break;
            }
        }
        self.entries.sort();
    }
}

/// Content authority owned by one canonical backing lifetime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegionContentState {
    backing: Option<BackingId>,
    next_version: u64,
    canonical: CoverageMap,
    representations: BTreeMap<RepresentationId, CoverageMap>,
    pending_gpu_writes: BTreeMap<GpuWriteId, PendingGpuWrite>,
    pending_transfers: BTreeMap<TransferKey, usize>,
    pending_host_landings: BTreeMap<HostLandingKey, usize>,
    pending_host_ingresses: BTreeMap<HostIngressKey, usize>,
    discarded: Vec<BackingRegion>,
}

impl RegionContentState {
    pub fn snapshot_all(&self) -> Box<[RegionVersion]> {
        self.canonical.entries.clone().into_boxed_slice()
    }

    pub fn validate_reservations(&self, count: usize) -> Result<(), ContentAuthorityError> {
        let count =
            u64::try_from(count).map_err(|_| ContentAuthorityError::VersionSpaceExhausted)?;
        self.next_version
            .checked_add(count)
            .ok_or(ContentAuthorityError::VersionSpaceExhausted)
            .map(|_| ())
    }

    pub fn validate_guest_write(
        &self,
        representation: RepresentationId,
    ) -> Result<(), ContentAuthorityError> {
        if !self.representations.contains_key(&representation) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        self.validate_reservations(1)
    }

    pub fn validate_plan_gpu_write(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        region_count: usize,
    ) -> Result<(), ContentAuthorityError> {
        if self.pending_gpu_writes.contains_key(&write.into()) {
            return Err(ContentAuthorityError::GpuWriteAlreadyPlanned);
        }
        if !self.representations.contains_key(&representation) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        self.validate_reservations(region_count)
    }

    pub fn validate_plan_transfers(
        &self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<(), ContentAuthorityError> {
        let source_coverage = self
            .representations
            .get(&source)
            .ok_or(ContentAuthorityError::UnknownRepresentation)?;
        if snapshot
            .iter()
            .any(|required| !source_coverage.covers(required.region, required.version))
        {
            return Err(ContentAuthorityError::StaleSource);
        }
        if !self.representations.contains_key(&destination) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        if self.backing.is_none() {
            return Err(ContentAuthorityError::UnboundBacking);
        }
        Ok(())
    }

    pub fn new(
        backing: BackingId,
        initial_representation: RepresentationId,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<Self, ContentAuthorityError> {
        let regions = regions.into();
        if regions.is_empty() {
            return Err(ContentAuthorityError::EmptyBacking);
        }
        let initial = ContentVersion::new(1);
        let mut canonical = CoverageMap::default();
        let mut representation = CoverageMap::default();
        for region in regions.iter().copied() {
            canonical.assign(region, initial);
            representation.assign(region, initial);
        }
        let mut representations = BTreeMap::new();
        representations.insert(initial_representation, representation);
        Ok(Self {
            backing: Some(backing),
            next_version: 2,
            canonical,
            representations,
            pending_gpu_writes: BTreeMap::new(),
            pending_transfers: BTreeMap::new(),
            pending_host_landings: BTreeMap::new(),
            pending_host_ingresses: BTreeMap::new(),
            discarded: Vec::new(),
        })
    }

    fn detached(initial_representation: RepresentationId) -> Self {
        let initial = ContentVersion::new(1);
        let mut canonical = CoverageMap::default();
        canonical.assign(BackingRegion::Whole, initial);
        let mut representation = CoverageMap::default();
        representation.assign(BackingRegion::Whole, initial);
        let mut representations = BTreeMap::new();
        representations.insert(initial_representation, representation);
        Self {
            backing: None,
            next_version: 2,
            canonical,
            representations,
            pending_gpu_writes: BTreeMap::new(),
            pending_transfers: BTreeMap::new(),
            pending_host_landings: BTreeMap::new(),
            pending_host_ingresses: BTreeMap::new(),
            discarded: Vec::new(),
        }
    }

    fn reserve_version(&mut self) -> Result<ContentVersion, ContentAuthorityError> {
        let version = ContentVersion::new(self.next_version);
        self.next_version = self
            .next_version
            .checked_add(1)
            .ok_or(ContentAuthorityError::VersionSpaceExhausted)?;
        Ok(version)
    }

    pub fn ensure_representation(&mut self, representation: RepresentationId) {
        self.representations.entry(representation).or_default();
    }

    pub(crate) fn remove_representation(&mut self, representation: RepresentationId) {
        self.representations.remove(&representation);
    }

    pub fn guest_write(
        &mut self,
        representation: RepresentationId,
        region: BackingRegion,
    ) -> Result<RegionVersion, ContentAuthorityError> {
        self.validate_guest_write(representation)?;
        let version = self.reserve_version()?;
        self.canonical.assign(region, version);
        self.representations
            .get_mut(&representation)
            .unwrap()
            .assign(region, version);
        self.discarded = self
            .discarded
            .drain(..)
            .flat_map(|discarded| subtract(discarded, region))
            .collect();
        Ok(RegionVersion { region, version })
    }

    pub fn plan_gpu_write(
        &mut self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<Box<[RegionVersion]>, ContentAuthorityError> {
        let regions = regions.into();
        let write = write.into();
        self.validate_plan_gpu_write(write, representation, regions.len())?;
        let mut planned = Vec::new();
        for region in regions.iter().copied() {
            planned.push(RegionVersion {
                region,
                version: self.reserve_version()?,
            });
        }
        let planned = planned.into_boxed_slice();
        self.pending_gpu_writes.insert(
            write,
            PendingGpuWrite {
                representation,
                regions: planned.clone(),
            },
        );
        Ok(planned)
    }

    pub fn complete_gpu_write(
        &mut self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
    ) -> Result<Box<[RegionVersion]>, ContentAuthorityError> {
        let write = write.into();
        self.validate_complete_gpu_write(write, representation)?;
        let planned = self
            .pending_gpu_writes
            .remove(&write)
            .expect("GPU-write completion was prevalidated");
        let native = self
            .representations
            .get_mut(&representation)
            .expect("GPU-write representation was prevalidated");
        for write in planned.regions.iter().copied() {
            native.assign_if_newer(write.region, write.version);
            self.canonical.assign_if_newer(write.region, write.version);
        }
        Ok(planned.regions)
    }

    pub fn validate_complete_gpu_write(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
    ) -> Result<(), ContentAuthorityError> {
        let planned = self
            .pending_gpu_writes
            .get(&write.into())
            .ok_or(ContentAuthorityError::SubmissionDidNotPlanWrite)?;
        if planned.representation != representation {
            return Err(ContentAuthorityError::GpuWriteRepresentationMismatch);
        }
        if !self.representations.contains_key(&representation) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        Ok(())
    }

    pub fn cancel_gpu_write(
        &mut self,
        write: impl Into<GpuWriteId>,
    ) -> Result<Box<[RegionVersion]>, ContentAuthorityError> {
        self.pending_gpu_writes
            .remove(&write.into())
            .map(|planned| planned.regions)
            .ok_or(ContentAuthorityError::SubmissionDidNotPlanWrite)
    }

    pub fn validate_gpu_write_reservation(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        expected: &[RegionVersion],
    ) -> Result<(), ContentAuthorityError> {
        let planned = self
            .pending_gpu_writes
            .get(&write.into())
            .ok_or(ContentAuthorityError::SubmissionDidNotPlanWrite)?;
        if planned.representation != representation {
            return Err(ContentAuthorityError::GpuWriteRepresentationMismatch);
        }
        if planned.regions.as_ref() != expected {
            return Err(ContentAuthorityError::GpuWriteReservationMismatch);
        }
        Ok(())
    }

    pub fn snapshot(&self, regions: &[BackingRegion]) -> Box<[RegionVersion]> {
        let mut snapshot = Vec::new();
        for region in regions {
            snapshot.extend(self.canonical.intersecting(*region));
        }
        snapshot.sort();
        snapshot.dedup();
        snapshot.into_boxed_slice()
    }

    pub fn representation_matches(
        &self,
        representation: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> bool {
        self.representations
            .get(&representation)
            .is_some_and(|coverage| {
                snapshot
                    .iter()
                    .all(|required| coverage.covers(required.region, required.version))
            })
    }

    /// Every region one representation currently holds, with the version it
    /// holds it at.
    ///
    /// `representation_matches` answers yes or no, which is the right shape for
    /// a decision and the wrong one for a diagnostic: a refusal that says only
    /// "stale" leaves the next question -- what does it hold instead -- with
    /// nowhere to start, and a boot spent twenty-eight thousand retries on one
    /// of these saying nothing more than the name.
    pub fn representation_coverage(&self, representation: RepresentationId) -> Vec<RegionVersion> {
        self.representations
            .get(&representation)
            .map(|coverage| coverage.entries.clone())
            .unwrap_or_default()
    }

    pub(crate) fn current_regions_in_representation(
        &self,
        representation: RepresentationId,
        required: RegionVersion,
    ) -> Box<[BackingRegion]> {
        self.representations
            .get(&representation)
            .into_iter()
            .flat_map(|coverage| coverage.entries.iter())
            .filter(|current| current.version == required.version)
            .filter_map(|current| intersection(current.region, required.region))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Whether removing one representation preserves every current canonical
    /// byte in the remaining available representations and abandons no
    /// content-producing obligation owned by the representation.
    pub(crate) fn representation_can_retire(
        &self,
        representation: RepresentationId,
        unavailable: &[RepresentationId],
    ) -> bool {
        if self
            .pending_gpu_writes
            .values()
            .any(|write| write.representation == representation)
            || self
                .pending_transfers
                .keys()
                .any(|transfer| transfer.source == representation)
        {
            return false;
        }

        self.canonical.entries.iter().all(|required| {
            let mut missing = vec![required.region];
            for (&candidate, coverage) in &self.representations {
                if candidate == representation || unavailable.contains(&candidate) {
                    continue;
                }
                for current in coverage
                    .entries
                    .iter()
                    .filter(|current| current.version == required.version)
                {
                    missing = missing
                        .into_iter()
                        .flat_map(|region| subtract(region, current.region))
                        .collect();
                    if missing.is_empty() {
                        return true;
                    }
                }
            }
            false
        })
    }

    pub(crate) fn pending_gpu_writes_overlapping(
        &self,
        representation: RepresentationId,
        regions: &[BackingRegion],
    ) -> Box<[GpuWriteId]> {
        self.pending_gpu_writes
            .iter()
            .filter_map(|(&identity, write)| {
                (write.representation == representation
                    && write.regions.iter().any(|write| {
                        regions
                            .iter()
                            .any(|required| intersection(write.region, *required).is_some())
                    }))
                .then_some(identity)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn plan_transfers(
        &mut self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ContentAuthorityError> {
        self.validate_plan_transfers(source, destination, snapshot)?;
        let destination_coverage = self
            .representations
            .get(&destination)
            .expect("transfer destination was prevalidated");
        let mut planned = Vec::new();
        for required in snapshot {
            for region in destination_coverage.missing(required.region, required.version) {
                let key = TransferKey {
                    backing: self.backing.ok_or(ContentAuthorityError::UnboundBacking)?,
                    region,
                    version: required.version,
                    source,
                    destination,
                };
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    self.pending_transfers.entry(key)
                {
                    entry.insert(1);
                    planned.push(key);
                }
            }
        }
        Ok(planned.into_boxed_slice())
    }

    /// Retain one demand for every missing destination region, including a
    /// region whose physical copy is already planned by another operation.
    pub fn plan_transfer_demands(
        &mut self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ContentAuthorityError> {
        self.validate_plan_transfer_demands(source, destination, snapshot)?;
        let destination_coverage = self.representations.get(&destination).unwrap();
        let mut demanded = Vec::new();
        for required in snapshot {
            for region in destination_coverage.missing(required.region, required.version) {
                let key = TransferKey {
                    backing: self.backing.ok_or(ContentAuthorityError::UnboundBacking)?,
                    region,
                    version: required.version,
                    source,
                    destination,
                };
                let count = self.pending_transfers.entry(key).or_default();
                *count = count
                    .checked_add(1)
                    .ok_or(ContentAuthorityError::VersionSpaceExhausted)?;
                demanded.push(key);
            }
        }
        Ok(demanded.into_boxed_slice())
    }

    pub fn plan_host_landing(
        &mut self,
        staged_transfer: TransferKey,
    ) -> Result<HostLandingKey, ContentAuthorityError> {
        if staged_transfer.destination != HOST_REPRESENTATION {
            return Err(ContentAuthorityError::HostLandingSourceMismatch);
        }
        if !self.pending_transfers.contains_key(&staged_transfer) {
            return Err(ContentAuthorityError::TransferNotPlanned);
        }
        if !self.representations.contains_key(&GUEST_REPRESENTATION) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        let landing = HostLandingKey {
            backing: staged_transfer.backing,
            region: staged_transfer.region,
            version: staged_transfer.version,
        };
        let count = self.pending_host_landings.entry(landing).or_default();
        *count = count
            .checked_add(1)
            .ok_or(ContentAuthorityError::TransferDemandCountExhausted)?;
        Ok(landing)
    }

    pub fn validate_complete_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        if !self.pending_host_landings.contains_key(&landing) {
            return Err(ContentAuthorityError::HostLandingNotPlanned);
        }
        let host = self
            .representations
            .get(&HOST_REPRESENTATION)
            .ok_or(ContentAuthorityError::UnknownRepresentation)?;
        if !host.covers(landing.region, landing.version) {
            return Err(ContentAuthorityError::HostLandingSourceNotCurrent);
        }
        if !self.representations.contains_key(&GUEST_REPRESENTATION) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        Ok(())
    }

    pub fn validate_host_landing_pending(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.pending_host_landings
            .contains_key(&landing)
            .then_some(())
            .ok_or(ContentAuthorityError::HostLandingNotPlanned)
    }

    pub fn complete_host_landing(
        &mut self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.validate_complete_host_landing(landing)?;
        let count = self.pending_host_landings.get_mut(&landing).unwrap();
        *count -= 1;
        if *count == 0 {
            self.pending_host_landings.remove(&landing);
        }
        self.representations
            .get_mut(&GUEST_REPRESENTATION)
            .expect("host landing destination was prevalidated")
            .assign(landing.region, landing.version);
        Ok(())
    }

    pub fn cancel_host_landing(
        &mut self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.validate_cancel_host_landing(landing)?;
        let count = self.pending_host_landings.get_mut(&landing).unwrap();
        *count -= 1;
        if *count == 0 {
            self.pending_host_landings.remove(&landing);
        }
        Ok(())
    }

    pub fn validate_cancel_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.validate_cancel_host_landing_demands(landing, 1)
    }

    pub fn validate_cancel_host_landing_demands(
        &self,
        landing: HostLandingKey,
        count: usize,
    ) -> Result<(), ContentAuthorityError> {
        match self.pending_host_landings.get(&landing) {
            None => Err(ContentAuthorityError::HostLandingNotPlanned),
            Some(available) if *available < count => {
                Err(ContentAuthorityError::InsufficientTransferDemand)
            }
            Some(_) => Ok(()),
        }
    }

    pub fn plan_host_ingress(
        &mut self,
        write: RegionVersion,
    ) -> Result<HostIngressKey, ContentAuthorityError> {
        let guest = self
            .representations
            .get(&GUEST_REPRESENTATION)
            .ok_or(ContentAuthorityError::UnknownRepresentation)?;
        if !guest.covers(write.region, write.version) {
            return Err(ContentAuthorityError::HostIngressSourceNotCurrent);
        }
        if !self.representations.contains_key(&HOST_REPRESENTATION) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        let ingress = HostIngressKey {
            backing: self.backing.ok_or(ContentAuthorityError::UnboundBacking)?,
            region: write.region,
            version: write.version,
        };
        let count = self.pending_host_ingresses.entry(ingress).or_default();
        *count = count
            .checked_add(1)
            .ok_or(ContentAuthorityError::TransferDemandCountExhausted)?;
        Ok(ingress)
    }

    pub fn validate_complete_host_ingress(
        &self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        if !self.pending_host_ingresses.contains_key(&ingress) {
            return Err(ContentAuthorityError::HostIngressNotPlanned);
        }
        if !self.canonical.covers(ingress.region, ingress.version) {
            return Err(ContentAuthorityError::HostIngressSourceNotCurrent);
        }
        if !self.representations.contains_key(&HOST_REPRESENTATION) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        Ok(())
    }

    pub fn validate_host_ingress_transfer(
        &self,
        transfer: HostIngressTransfer,
    ) -> Result<(), ContentAuthorityError> {
        let mut projected = self.clone();
        projected.complete_host_ingress(transfer.ingress)?;
        projected.validate_plan_transfer_demands(
            HOST_REPRESENTATION,
            transfer.destination,
            &[RegionVersion {
                region: transfer.ingress.region,
                version: transfer.ingress.version,
            }],
        )
    }

    pub fn complete_host_ingress(
        &mut self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        self.validate_complete_host_ingress(ingress)?;
        let count = self.pending_host_ingresses.get_mut(&ingress).unwrap();
        *count -= 1;
        if *count == 0 {
            self.pending_host_ingresses.remove(&ingress);
        }
        self.representations
            .get_mut(&HOST_REPRESENTATION)
            .expect("host ingress destination was prevalidated")
            .assign(ingress.region, ingress.version);
        Ok(())
    }

    pub fn cancel_host_ingress(
        &mut self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        self.validate_cancel_host_ingress(ingress)?;
        let count = self
            .pending_host_ingresses
            .get_mut(&ingress)
            .expect("host ingress was prevalidated");
        *count -= 1;
        if *count == 0 {
            self.pending_host_ingresses.remove(&ingress);
        }
        Ok(())
    }

    pub fn validate_cancel_host_ingress(
        &self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        self.validate_cancel_host_ingress_demands(ingress, 1)
    }

    pub fn validate_cancel_host_ingress_demands(
        &self,
        ingress: HostIngressKey,
        count: usize,
    ) -> Result<(), ContentAuthorityError> {
        match self.pending_host_ingresses.get(&ingress) {
            None => Err(ContentAuthorityError::HostIngressNotPlanned),
            Some(available) if *available < count => {
                Err(ContentAuthorityError::InsufficientTransferDemand)
            }
            Some(_) => Ok(()),
        }
    }

    pub fn validate_plan_transfer_demands(
        &self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<(), ContentAuthorityError> {
        self.validate_plan_transfers(source, destination, snapshot)?;
        let destination_coverage = self.representations.get(&destination).unwrap();
        for required in snapshot {
            for region in destination_coverage.missing(required.region, required.version) {
                let key = TransferKey {
                    backing: self.backing.ok_or(ContentAuthorityError::UnboundBacking)?,
                    region,
                    version: required.version,
                    source,
                    destination,
                };
                if self.pending_transfers.get(&key) == Some(&usize::MAX) {
                    return Err(ContentAuthorityError::TransferDemandCountExhausted);
                }
            }
        }
        Ok(())
    }

    pub fn complete_transfer(&mut self, key: TransferKey) -> Result<(), ContentAuthorityError> {
        self.validate_complete_transfer(key)?;
        self.pending_transfers.remove(&key);
        self.representations
            .get_mut(&key.destination)
            .expect("transfer destination was prevalidated")
            .assign(key.region, key.version);
        Ok(())
    }

    /// Return one exact planned transfer before native submission. Cancellation
    /// changes no representation coverage; a later request for the same
    /// version therefore plans the transfer again.
    pub fn cancel_transfer(&mut self, key: TransferKey) -> Result<(), ContentAuthorityError> {
        self.validate_cancel_transfer(key)?;
        let count = self.pending_transfers.get_mut(&key).unwrap();
        *count -= 1;
        if *count == 0 {
            self.pending_transfers.remove(&key);
        }
        Ok(())
    }

    pub fn validate_cancel_transfer(&self, key: TransferKey) -> Result<(), ContentAuthorityError> {
        self.validate_cancel_transfer_demands(key, 1)
    }

    pub fn validate_cancel_transfer_demands(
        &self,
        key: TransferKey,
        count: usize,
    ) -> Result<(), ContentAuthorityError> {
        match self.pending_transfers.get(&key) {
            None => Err(ContentAuthorityError::TransferNotPlanned),
            Some(available) if *available < count => {
                Err(ContentAuthorityError::InsufficientTransferDemand)
            }
            Some(_) => Ok(()),
        }
    }

    pub fn validate_complete_transfer(
        &self,
        key: TransferKey,
    ) -> Result<(), ContentAuthorityError> {
        if !self.pending_transfers.contains_key(&key) {
            return Err(ContentAuthorityError::TransferNotPlanned);
        }
        if !self.representations.contains_key(&key.destination) {
            return Err(ContentAuthorityError::UnknownRepresentation);
        }
        Ok(())
    }

    pub fn discard(&mut self, region: BackingRegion) {
        self.canonical.remove(region);
        for coverage in self.representations.values_mut() {
            coverage.remove(region);
        }
        self.discarded.push(region);
        self.discarded = coalesce_regions(std::mem::take(&mut self.discarded));
    }

    pub fn pending_transfer_count(&self) -> usize {
        self.pending_transfers.len()
    }

    fn materialize(
        &mut self,
        representation: RepresentationId,
        required: RegionVersion,
    ) -> Result<(), ContentAuthorityError> {
        if !self.canonical.covers(required.region, required.version) {
            return Err(ContentAuthorityError::StaleSource);
        }
        self.representations
            .entry(representation)
            .or_default()
            .assign(required.region, required.version);
        Ok(())
    }

    fn remove_representation_region(
        &mut self,
        representation: RepresentationId,
        region: BackingRegion,
    ) {
        if let Some(coverage) = self.representations.get_mut(&representation) {
            coverage.remove(region);
        }
    }
}

/// Shared regional authority for every view over one canonical backing.
///
/// A resource constructed before its backing is attached owns a detached
/// whole-backing authority. Attaching storage replaces it with the authority
/// owned by that `BackingId`; transfer planning on a detached authority is a
/// typed refusal because no canonical transfer key can be formed.
#[derive(Clone, Debug)]
pub struct ContentAuthority(Arc<Mutex<RegionContentState>>);

impl Default for ContentAuthority {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(RegionContentState::detached(
            GUEST_REPRESENTATION,
        ))))
    }
}

impl ContentAuthority {
    pub fn snapshot_all(&self) -> Box<[RegionVersion]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .snapshot_all()
    }

    pub fn validate_reservations(&self, count: usize) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_reservations(count)
    }

    pub fn validate_guest_write_region(
        &self,
        representation: RepresentationId,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_guest_write(representation)
    }

    pub fn validate_plan_gpu_write_regions(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        region_count: usize,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_plan_gpu_write(write, representation, region_count)
    }

    pub fn validate_plan_transfers(
        &self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_plan_transfers(source, destination, snapshot)
    }

    pub fn for_backing(backing: BackingId) -> Self {
        Self::for_backing_regions(backing, [BackingRegion::Whole])
            .expect("one whole-backing region is non-empty")
    }

    pub fn for_backing_regions(
        backing: BackingId,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<Self, ContentAuthorityError> {
        Ok(Self(Arc::new(Mutex::new(RegionContentState::new(
            backing,
            GUEST_REPRESENTATION,
            regions,
        )?))))
    }

    pub fn same_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn backing(&self) -> Option<BackingId> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .backing
    }

    pub fn snapshot_regions(&self, regions: &[BackingRegion]) -> Box<[RegionVersion]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .snapshot(regions)
    }

    pub fn representation_matches(
        &self,
        representation: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .representation_matches(representation, snapshot)
    }

    /// See [`RegionContentState::representation_coverage`].
    pub fn representation_coverage(&self, representation: RepresentationId) -> Vec<RegionVersion> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .representation_coverage(representation)
    }

    pub(crate) fn current_regions_in_representation(
        &self,
        representation: RepresentationId,
        required: RegionVersion,
    ) -> Box<[BackingRegion]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .current_regions_in_representation(representation, required)
    }

    pub(crate) fn pending_gpu_writes_overlapping(
        &self,
        representation: RepresentationId,
        regions: &[BackingRegion],
    ) -> Box<[GpuWriteId]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pending_gpu_writes_overlapping(representation, regions)
    }

    pub(crate) fn representation_can_retire(
        &self,
        representation: RepresentationId,
        unavailable: &[RepresentationId],
    ) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .representation_can_retire(representation, unavailable)
    }

    pub fn ensure_representation(&self, representation: RepresentationId) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .ensure_representation(representation);
    }

    pub fn guest_write_region(
        &self,
        representation: RepresentationId,
        region: BackingRegion,
    ) -> Result<RegionVersion, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .guest_write(representation, region)
    }

    pub fn plan_gpu_write_regions(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        regions: impl Into<Box<[BackingRegion]>>,
    ) -> Result<Box<[RegionVersion]>, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .plan_gpu_write(write, representation, regions)
    }

    pub fn complete_gpu_write_regions(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
    ) -> Result<Box<[RegionVersion]>, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .complete_gpu_write(write, representation)
    }

    pub fn validate_complete_gpu_write_regions(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_complete_gpu_write(write, representation)
    }

    /// Return a pre-completion GPU-write reservation to its lifecycle owner.
    /// Reserved content versions remain consumed, so a retry cannot reuse an
    /// identity that may already appear in immutable transaction state.
    pub fn cancel_gpu_write_regions(
        &self,
        write: impl Into<GpuWriteId>,
    ) -> Result<Box<[RegionVersion]>, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cancel_gpu_write(write)
    }

    pub fn validate_gpu_write_reservation(
        &self,
        write: impl Into<GpuWriteId>,
        representation: RepresentationId,
        expected: &[RegionVersion],
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_gpu_write_reservation(write, representation, expected)
    }

    pub fn plan_transfers(
        &self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .plan_transfers(source, destination, snapshot)
    }

    pub fn plan_transfer_demands(
        &self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<Box<[TransferKey]>, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .plan_transfer_demands(source, destination, snapshot)
    }

    pub fn validate_plan_transfer_demands(
        &self,
        source: RepresentationId,
        destination: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_plan_transfer_demands(source, destination, snapshot)
    }

    pub fn plan_host_landing(
        &self,
        staged_transfer: TransferKey,
    ) -> Result<HostLandingKey, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .plan_host_landing(staged_transfer)
    }

    pub fn validate_complete_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_complete_host_landing(landing)
    }

    pub fn validate_host_landing_pending(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_host_landing_pending(landing)
    }

    pub fn complete_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .complete_host_landing(landing)
    }

    pub fn cancel_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cancel_host_landing(landing)
    }

    pub fn validate_cancel_host_landing(
        &self,
        landing: HostLandingKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_cancel_host_landing(landing)
    }

    pub fn validate_cancel_host_landing_demands(
        &self,
        landing: HostLandingKey,
        count: usize,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_cancel_host_landing_demands(landing, count)
    }

    pub fn plan_host_ingress(
        &self,
        write: RegionVersion,
    ) -> Result<HostIngressKey, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .plan_host_ingress(write)
    }

    pub fn validate_complete_host_ingress(
        &self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_complete_host_ingress(ingress)
    }

    pub fn validate_host_ingress_transfer(
        &self,
        transfer: HostIngressTransfer,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_host_ingress_transfer(transfer)
    }

    pub fn complete_host_ingress(
        &self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .complete_host_ingress(ingress)
    }

    pub fn cancel_host_ingress(
        &self,
        ingress: HostIngressKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cancel_host_ingress(ingress)
    }

    pub fn validate_cancel_host_ingress_demands(
        &self,
        ingress: HostIngressKey,
        count: usize,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_cancel_host_ingress_demands(ingress, count)
    }

    pub fn complete_transfer(&self, key: TransferKey) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .complete_transfer(key)
    }

    pub fn cancel_transfer(&self, key: TransferKey) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cancel_transfer(key)
    }

    pub fn validate_cancel_transfer(&self, key: TransferKey) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_cancel_transfer(key)
    }

    pub fn validate_cancel_transfer_demands(
        &self,
        key: TransferKey,
        count: usize,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_cancel_transfer_demands(key, count)
    }

    pub fn validate_complete_transfer(
        &self,
        key: TransferKey,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_complete_transfer(key)
    }

    pub fn discard(&self, region: BackingRegion) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .discard(region);
    }

    pub fn pending_transfer_count(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pending_transfer_count()
    }

    pub(crate) fn whole_current(&self) -> ContentVersion {
        self.snapshot_regions(&[BackingRegion::Whole])
            .first()
            .expect("whole-backing authority always has canonical coverage")
            .version
    }

    pub(crate) fn whole_matches(&self, representation: RepresentationId) -> bool {
        let snapshot = self.snapshot_regions(&[BackingRegion::Whole]);
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .representation_matches(representation, &snapshot)
    }

    pub(crate) fn materialize_whole(
        &self,
        representation: RepresentationId,
        version: ContentVersion,
    ) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .materialize(
                representation,
                RegionVersion {
                    region: BackingRegion::Whole,
                    version,
                },
            )
    }

    pub(crate) fn remove_whole_representation(&self, representation: RepresentationId) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove_representation_region(representation, BackingRegion::Whole);
    }

    pub(crate) fn remove_representation(&self, representation: RepresentationId) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove_representation(representation);
    }
}

impl PartialEq for ContentAuthority {
    fn eq(&self, other: &Self) -> bool {
        let left = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let right = other
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        left == right
    }
}

impl Eq for ContentAuthority {}

pub(crate) fn intersection(left: BackingRegion, right: BackingRegion) -> Option<BackingRegion> {
    match (left, right) {
        (BackingRegion::Whole, BackingRegion::Whole) => Some(BackingRegion::Whole),
        (BackingRegion::Linear(left), BackingRegion::Linear(right)) => {
            let start = left.start().max(right.start());
            let end = left.end().min(right.end());
            LinearRange::new(start, end.checked_sub(start)?).map(BackingRegion::Linear)
        }
        (BackingRegion::Image(left), BackingRegion::Image(right))
            if left.aspect == right.aspect
                && left.mip == right.mip
                && left.layer == right.layer =>
        {
            let origin = [
                left.texels.origin[0].max(right.texels.origin[0]),
                left.texels.origin[1].max(right.texels.origin[1]),
                left.texels.origin[2].max(right.texels.origin[2]),
            ];
            let end = [
                left.texels.end[0].min(right.texels.end[0]),
                left.texels.end[1].min(right.texels.end[1]),
                left.texels.end[2].min(right.texels.end[2]),
            ];
            if (0..3).any(|axis| origin[axis] >= end[axis]) {
                return None;
            }
            Some(BackingRegion::Image(ImageRegion {
                aspect: left.aspect,
                mip: left.mip,
                layer: left.layer,
                texels: TexelBox { origin, end },
            }))
        }
        _ => None,
    }
}

pub(crate) fn subtract(region: BackingRegion, cut: BackingRegion) -> Vec<BackingRegion> {
    let Some(overlap) = intersection(region, cut) else {
        return vec![region];
    };
    match (region, overlap) {
        (BackingRegion::Whole, BackingRegion::Whole) => Vec::new(),
        (BackingRegion::Linear(region), BackingRegion::Linear(overlap)) => {
            let mut out = Vec::with_capacity(2);
            if region.start() < overlap.start() {
                out.push(BackingRegion::Linear(
                    LinearRange::new(region.start(), overlap.start() - region.start()).unwrap(),
                ));
            }
            if overlap.end() < region.end() {
                out.push(BackingRegion::Linear(
                    LinearRange::new(overlap.end(), region.end() - overlap.end()).unwrap(),
                ));
            }
            out
        }
        (BackingRegion::Image(region), BackingRegion::Image(overlap)) => {
            subtract_image(region, overlap)
                .into_iter()
                .map(BackingRegion::Image)
                .collect()
        }
        _ => unreachable!("intersection preserves region kind"),
    }
}

fn subtract_image(region: ImageRegion, overlap: ImageRegion) -> Vec<ImageRegion> {
    let old = region.texels;
    let mid = overlap.texels;
    let mut out = Vec::with_capacity(6);
    let mut push = |origin: [u32; 3], end: [u32; 3]| {
        if (0..3).all(|axis| origin[axis] < end[axis]) {
            out.push(ImageRegion {
                aspect: region.aspect,
                mip: region.mip,
                layer: region.layer,
                texels: TexelBox { origin, end },
            });
        }
    };

    push(old.origin, [mid.origin[0], old.end[1], old.end[2]]);
    push([mid.end[0], old.origin[1], old.origin[2]], old.end);
    push(
        [mid.origin[0], old.origin[1], old.origin[2]],
        [mid.end[0], mid.origin[1], old.end[2]],
    );
    push(
        [mid.origin[0], mid.end[1], old.origin[2]],
        [mid.end[0], old.end[1], old.end[2]],
    );
    push(
        [mid.origin[0], mid.origin[1], old.origin[2]],
        [mid.end[0], mid.end[1], mid.origin[2]],
    );
    push(
        [mid.origin[0], mid.origin[1], mid.end[2]],
        [mid.end[0], mid.end[1], old.end[2]],
    );
    out
}

fn merge_regions(left: BackingRegion, right: BackingRegion) -> Option<BackingRegion> {
    match (left, right) {
        (BackingRegion::Whole, BackingRegion::Whole) => Some(BackingRegion::Whole),
        (BackingRegion::Linear(left), BackingRegion::Linear(right))
            if left.end() == right.start() || right.end() == left.start() =>
        {
            let start = left.start().min(right.start());
            let end = left.end().max(right.end());
            Some(BackingRegion::Linear(
                LinearRange::new(start, end - start).unwrap(),
            ))
        }
        (BackingRegion::Image(left), BackingRegion::Image(right))
            if left.aspect == right.aspect
                && left.mip == right.mip
                && left.layer == right.layer =>
        {
            merge_boxes(left.texels, right.texels)
                .map(|texels| BackingRegion::Image(ImageRegion { texels, ..left }))
        }
        _ => None,
    }
}

fn merge_boxes(left: TexelBox, right: TexelBox) -> Option<TexelBox> {
    for axis in 0..3 {
        let other_axes_match = (0..3)
            .filter(|candidate| *candidate != axis)
            .all(|candidate| {
                left.origin[candidate] == right.origin[candidate]
                    && left.end[candidate] == right.end[candidate]
            });
        if other_axes_match
            && (left.end[axis] == right.origin[axis] || right.end[axis] == left.origin[axis])
        {
            let mut origin = left.origin;
            let mut end = left.end;
            origin[axis] = left.origin[axis].min(right.origin[axis]);
            end[axis] = left.end[axis].max(right.end[axis]);
            return Some(TexelBox { origin, end });
        }
    }
    None
}

fn coalesce_regions(mut regions: Vec<BackingRegion>) -> Vec<BackingRegion> {
    loop {
        let mut merged = false;
        'outer: for left in 0..regions.len() {
            for right in (left + 1)..regions.len() {
                if let Some(region) = merge_regions(regions[left], regions[right]) {
                    regions[left] = region;
                    regions.remove(right);
                    merged = true;
                    break 'outer;
                }
            }
        }
        if !merged {
            break;
        }
    }
    regions.sort();
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    const GUEST: RepresentationId = RepresentationId::new(1);
    const GPU: RepresentationId = RepresentationId::new(2);

    fn linear(start: u64, length: u64) -> BackingRegion {
        BackingRegion::Linear(LinearRange::new(start, length).unwrap())
    }

    fn image(origin: [u32; 3], extent: [u32; 3]) -> BackingRegion {
        BackingRegion::Image(ImageRegion {
            aspect: ImageAspect::Color,
            mip: 0,
            layer: 0,
            texels: TexelBox::new(origin, extent).unwrap(),
        })
    }

    fn state(region: BackingRegion) -> RegionContentState {
        let mut state = RegionContentState::new(BackingId::new(7), GUEST, [region]).unwrap();
        state.ensure_representation(GPU);
        state
    }

    #[test]
    fn disjoint_gpu_writes_complete_in_either_order_and_both_remain_canonical() {
        let mut state = state(linear(0, 128));
        let left = state
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(0, 64)])
            .unwrap()[0];
        let right = state
            .plan_gpu_write(SubmissionId::new(2), GPU, [linear(64, 64)])
            .unwrap()[0];
        state.complete_gpu_write(SubmissionId::new(2), GPU).unwrap();
        state.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();
        assert_eq!(state.snapshot(&[linear(0, 64)]).as_ref(), [left]);
        assert_eq!(state.snapshot(&[linear(64, 64)]).as_ref(), [right]);
    }

    #[test]
    fn operation_identity_separates_two_writes_in_one_submission() {
        let mut state = state(linear(0, 128));
        let transaction = TransactionId::new(3);
        let submission = SubmissionId::new(4);
        let first = GpuWriteId::operation(transaction, submission, 2);
        let second = GpuWriteId::operation(transaction, submission, 5);
        let older = state.plan_gpu_write(first, GPU, [linear(0, 128)]).unwrap()[0];
        let newer = state.plan_gpu_write(second, GPU, [linear(32, 32)]).unwrap()[0];

        state.complete_gpu_write(second, GPU).unwrap();
        state.complete_gpu_write(first, GPU).unwrap();
        assert_eq!(state.snapshot(&[linear(32, 32)]).as_ref(), [newer]);
        assert_eq!(
            state
                .snapshot(&[linear(0, 32), linear(64, 64)])
                .iter()
                .map(|region| region.version)
                .collect::<Vec<_>>(),
            [older.version, older.version]
        );
    }

    #[test]
    fn stale_overlapping_gpu_completion_cannot_replace_a_newer_guest_write() {
        let mut state = state(linear(0, 128));
        state
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(0, 128)])
            .unwrap();
        let guest = state.guest_write(GUEST, linear(32, 32)).unwrap();
        state.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();
        assert_eq!(state.snapshot(&[linear(32, 32)]).as_ref(), [guest]);
        let outside = state.snapshot(&[linear(0, 32), linear(64, 64)]);
        assert!(outside.iter().all(|version| version.version.get() == 2));
    }

    #[test]
    fn stale_gpu_completion_observation_cannot_replace_newer_representation_coverage() {
        let mut state = state(linear(0, 128));
        let older = SubmissionId::new(1);
        let newer = SubmissionId::new(2);
        state.plan_gpu_write(older, GPU, [linear(0, 128)]).unwrap();
        state.plan_gpu_write(newer, GPU, [linear(0, 128)]).unwrap();

        state.complete_gpu_write(newer, GPU).unwrap();
        let newest = state.snapshot(&[linear(0, 128)]);
        assert!(state.representation_matches(GPU, &newest));

        state.complete_gpu_write(older, GPU).unwrap();
        assert_eq!(state.snapshot(&[linear(0, 128)]), newest);
        assert!(state.representation_matches(GPU, &newest));
    }

    #[test]
    fn cancelled_gpu_write_changes_no_coverage_and_never_reuses_its_version() {
        let mut state = state(linear(0, 128));
        let before = state.snapshot(&[linear(0, 128)]);
        let cancelled = state
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(32, 32)])
            .unwrap();
        assert_eq!(cancelled[0].version, ContentVersion::new(2));

        assert_eq!(
            state.cancel_gpu_write(SubmissionId::new(1)).unwrap(),
            cancelled
        );
        assert_eq!(state.snapshot(&[linear(0, 128)]), before);
        assert_eq!(
            state.complete_gpu_write(SubmissionId::new(1), GPU),
            Err(ContentAuthorityError::SubmissionDidNotPlanWrite)
        );

        let retry = state
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(32, 32)])
            .unwrap();
        assert_eq!(retry[0].version, ContentVersion::new(3));
    }

    #[test]
    fn transfer_plans_only_missing_coverage_and_never_duplicates_a_live_key() {
        let mut state = state(linear(0, 128));
        let first = state.guest_write(GUEST, linear(0, 128)).unwrap();
        let snapshot = state.snapshot(&[linear(0, 128)]);
        let planned = state.plan_transfers(GUEST, GPU, &snapshot).unwrap();
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].region, first.region);
        assert!(state
            .plan_transfers(GUEST, GPU, &snapshot)
            .unwrap()
            .is_empty());
        state.complete_transfer(planned[0]).unwrap();
        assert!(state.representation_matches(GPU, &snapshot));
        assert_eq!(state.pending_transfer_count(), 0);
    }

    #[test]
    fn host_landing_cannot_publish_guest_coverage_before_its_staged_transfer() {
        let backing = BackingId::new(7);
        let working = RepresentationId::new(4);
        let mut state = RegionContentState::new(
            backing,
            GUEST_REPRESENTATION,
            vec![BackingRegion::Whole].into_boxed_slice(),
        )
        .unwrap();
        state.ensure_representation(working);
        state.ensure_representation(HOST_REPRESENTATION);
        state
            .plan_gpu_write(SubmissionId::new(1), working, [BackingRegion::Whole])
            .unwrap();
        state
            .complete_gpu_write(SubmissionId::new(1), working)
            .unwrap();
        let snapshot = state.snapshot(&[BackingRegion::Whole]);
        let staged = state
            .plan_transfer_demands(working, HOST_REPRESENTATION, &snapshot)
            .unwrap()[0];
        let landing = state.plan_host_landing(staged).unwrap();

        assert_eq!(
            state.validate_complete_host_landing(landing),
            Err(ContentAuthorityError::HostLandingSourceNotCurrent)
        );
        state.complete_transfer(staged).unwrap();
        assert!(!state.representation_matches(GUEST_REPRESENTATION, &snapshot));
        state.complete_host_landing(landing).unwrap();
        assert!(state.representation_matches(GUEST_REPRESENTATION, &snapshot));
    }

    #[test]
    fn host_ingress_publishes_only_the_exact_still_current_guest_write() {
        let backing = BackingId::new(8);
        let region = linear(16, 32);
        let mut state = RegionContentState::new(
            backing,
            GUEST_REPRESENTATION,
            vec![linear(0, 128)].into_boxed_slice(),
        )
        .unwrap();
        state.ensure_representation(HOST_REPRESENTATION);

        let write = state.guest_write(GUEST_REPRESENTATION, region).unwrap();
        let ingress = state.plan_host_ingress(write).unwrap();
        assert!(!state.representation_matches(HOST_REPRESENTATION, &[write]));
        state.remove_representation_region(GUEST_REPRESENTATION, region);
        assert!(!state.representation_matches(GUEST_REPRESENTATION, &[write]));
        state.complete_host_ingress(ingress).unwrap();
        assert!(state.representation_matches(HOST_REPRESENTATION, &[write]));

        let stale = state.guest_write(GUEST_REPRESENTATION, region).unwrap();
        let stale_ingress = state.plan_host_ingress(stale).unwrap();
        state.guest_write(GUEST_REPRESENTATION, region).unwrap();
        assert_eq!(
            state.validate_complete_host_ingress(stale_ingress),
            Err(ContentAuthorityError::HostIngressSourceNotCurrent)
        );
        assert_eq!(
            state.complete_host_ingress(stale_ingress),
            Err(ContentAuthorityError::HostIngressSourceNotCurrent)
        );
        state.cancel_host_ingress(stale_ingress).unwrap();
        assert_eq!(
            state.cancel_host_ingress(stale_ingress),
            Err(ContentAuthorityError::HostIngressNotPlanned)
        );
    }

    #[test]
    fn cancelled_transfer_publishes_no_coverage_and_can_be_planned_again() {
        let mut state = state(linear(0, 128));
        state.guest_write(GUEST, linear(16, 32)).unwrap();
        let snapshot = state.snapshot(&[linear(16, 32)]);
        let key = state.plan_transfers(GUEST, GPU, &snapshot).unwrap()[0];

        state.cancel_transfer(key).unwrap();
        assert!(!state.representation_matches(GPU, &snapshot));
        assert_eq!(state.pending_transfer_count(), 0);
        assert_eq!(
            state.complete_transfer(key),
            Err(ContentAuthorityError::TransferNotPlanned)
        );
        assert_eq!(state.plan_transfers(GUEST, GPU, &snapshot).unwrap()[0], key);
    }

    #[test]
    fn shared_transfer_demands_cancel_independently_and_complete_once() {
        let mut state = state(linear(0, 128));
        state.guest_write(GUEST, linear(0, 64)).unwrap();
        let snapshot = state.snapshot(&[linear(0, 64)]);
        let first = state.plan_transfer_demands(GUEST, GPU, &snapshot).unwrap()[0];
        let second = state.plan_transfer_demands(GUEST, GPU, &snapshot).unwrap()[0];
        assert_eq!(first, second);

        state.cancel_transfer(first).unwrap();
        state.complete_transfer(second).unwrap();
        assert!(state.representation_matches(GPU, &snapshot));
        assert_eq!(state.pending_transfer_count(), 0);
    }

    #[test]
    fn a_partially_current_destination_transfers_only_the_missing_interval() {
        let mut state = state(linear(0, 128));
        let initial = state.snapshot(&[linear(0, 64)]);
        let initial_transfer = state.plan_transfers(GUEST, GPU, &initial).unwrap();
        state.complete_transfer(initial_transfer[0]).unwrap();

        let snapshot = state.snapshot(&[linear(0, 128)]);
        let planned = state.plan_transfers(GUEST, GPU, &snapshot).unwrap();
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].region, linear(64, 64));
    }

    #[test]
    fn image_box_subtraction_preserves_every_texel_outside_the_write() {
        let whole = image([0, 0, 0], [8, 8, 1]);
        let center = image([2, 2, 0], [4, 4, 1]);
        let remainder = subtract(whole, center);
        assert_eq!(remainder.len(), 4);
        let volume: u32 = remainder
            .iter()
            .map(|region| match region {
                BackingRegion::Image(region) => {
                    (region.texels.end[0] - region.texels.origin[0])
                        * (region.texels.end[1] - region.texels.origin[1])
                        * (region.texels.end[2] - region.texels.origin[2])
                }
                BackingRegion::Whole | BackingRegion::Linear(_) => unreachable!(),
            })
            .sum();
        assert_eq!(volume, 64 - 16);
        assert!(remainder
            .iter()
            .all(|region| intersection(*region, center).is_none()));
    }

    #[test]
    fn discard_removes_only_the_named_region_until_new_content_arrives() {
        let mut state = state(linear(0, 128));
        state.discard(linear(32, 32));
        assert!(state.snapshot(&[linear(32, 32)]).is_empty());
        assert_eq!(state.snapshot(&[linear(0, 32)]).len(), 1);
        let restored = state.guest_write(GUEST, linear(32, 32)).unwrap();
        assert_eq!(state.snapshot(&[linear(32, 32)]).as_ref(), [restored]);
    }

    #[test]
    fn shared_authority_owns_regional_versions_for_one_backing() {
        let authority = ContentAuthority::for_backing_regions(
            BackingId::new(9),
            [linear(0, 64), linear(64, 64)],
        )
        .unwrap();
        let alias = authority.clone();
        authority.ensure_representation(GPU);
        let planned = authority
            .plan_gpu_write_regions(SubmissionId::new(3), GPU, [linear(0, 64), linear(64, 64)])
            .unwrap();
        let completed = alias
            .complete_gpu_write_regions(SubmissionId::new(3), GPU)
            .unwrap();
        assert_eq!(completed, planned);
        assert!(authority.same_authority(&alias));
        assert_eq!(
            alias.snapshot_regions(&[linear(0, 128)]).as_ref(),
            planned.as_ref()
        );
    }

    #[test]
    fn detached_authority_refuses_to_invent_a_transfer_backing() {
        let authority = ContentAuthority::default();
        authority.ensure_representation(GPU);
        let snapshot = authority.snapshot_regions(&[BackingRegion::Whole]);
        assert_eq!(
            authority.plan_transfers(GUEST, GPU, &snapshot),
            Err(ContentAuthorityError::UnboundBacking)
        );
    }
}
