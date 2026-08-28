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
use std::collections::BTreeSet;
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

/// Which admitted operation a guest write belongs to.
///
/// A guest write is one operation's statement that the guest wrote these bytes,
/// and preparing that operation twice is the same statement made twice rather
/// than two writes. An EXEC that refuses after its resource states are prepared
/// gives its claims up and re-prepares them on the retry, so without an
/// identity every retry minted a fresh content version -- which invalidates
/// every representation of the backing, including the one whose upload the
/// retry is waiting on. That is a live-lock, and it is not a slow one: a driven
/// macos-13 conformance boot reached 126 667 retries against one backing, with
/// the required version 126 036 ahead of anything any representation held, and
/// the guest never got its first case out.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GuestWriteId {
    pub transaction: TransactionId,
    pub submission: SubmissionId,
    pub index: usize,
}

impl GuestWriteId {
    pub const fn operation(
        transaction: TransactionId,
        submission: SubmissionId,
        index: usize,
    ) -> Self {
        Self {
            transaction,
            submission,
            index,
        }
    }
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
            for remainder in remaining_after(existing.region, region) {
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
                .flat_map(|candidate| remaining_after(candidate, protected.region))
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
                remaining_after(existing.region, region)
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
                .flat_map(|candidate| not_covered_by(candidate, current.region))
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
    /// The coordinate vocabulary this backing was created with.
    ///
    /// A backing is declared once, in one coordinate space -- byte ranges for
    /// a linear allocation, texel boxes where the contract declares the image
    /// geometry, or the complete backing where it declares neither -- and that
    /// space is fixed for its lifetime. Coverage *entries* drift as writes land
    /// and subdivide, so the entries currently present are an accident of who
    /// wrote last; the declaration is not, and it is what every operation over
    /// the backing must be expressed in to be issuable at all.
    declared: Box<[BackingRegion]>,
    next_version: u64,
    canonical: CoverageMap,
    representations: BTreeMap<RepresentationId, CoverageMap>,
    /// Representations whose bytes *are* the guest's bytes.
    ///
    /// A direct guest alias is a native object bound over the guest's own
    /// pages: there is no second copy, so a write through it is a write the
    /// guest can already see, and a guest store is a store the object can
    /// already read. Coverage for such an object is therefore not a separate
    /// statement from the guest's -- it is the same statement addressed twice,
    /// and every assignment to one member of this class is made to all of them
    /// and to [`GUEST_REPRESENTATION`].
    ///
    /// The alias needs its own identity even though it shares the guest's
    /// content, because identity is what object lifetime is keyed on: the
    /// guest representation is permanent and a native object is not, so
    /// pinning the alias to [`GUEST_REPRESENTATION`] makes a physical
    /// replacement either unable to retire the object or able to erase the
    /// guest's canonical coverage with it. The set is what keeps the two facts
    /// apart -- one identity per object lifetime, one content statement across
    /// all of them.
    guest_aliases: BTreeSet<RepresentationId>,
    pending_gpu_writes: BTreeMap<GpuWriteId, PendingGpuWrite>,
    pending_transfers: BTreeMap<TransferKey, usize>,
    pending_host_landings: BTreeMap<HostLandingKey, usize>,
    pending_host_ingresses: BTreeMap<HostIngressKey, usize>,
    /// Which operation last wrote each region as the guest, and at what
    /// version, so re-preparing that operation repeats its statement instead of
    /// making a new one. See [`GuestWriteId`].
    ///
    /// One entry per region the guest has written, superseded by the next
    /// operation to write it -- so this is bounded by the backing's own
    /// declared regions and needs no sweep to stay that way.
    guest_writes: BTreeMap<BackingRegion, (GuestWriteId, ContentVersion)>,
    discarded: Vec<BackingRegion>,
}

impl RegionContentState {
    /// See the field's own documentation: this is the backing's coordinate
    /// vocabulary, not what it currently holds.
    pub fn declared_regions(&self) -> &[BackingRegion] {
        &self.declared
    }

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
        let destination_coverage = self
            .representations
            .get(&destination)
            .ok_or(ContentAuthorityError::UnknownRepresentation)?;
        // The source is asked to cover what will actually be copied, which is
        // what the destination is missing and not the whole snapshot the
        // consumer named. The two differ whenever canonical content is split
        // across representations -- a GPU write of four bytes into a page the
        // guest otherwise still holds current leaves no single object covering
        // the page, and demanding one refuses a readback as `StaleSource` over
        // content nothing is stale about. Both planners below narrow by the
        // same `missing`, so a plan this admits is a plan they can build.
        if snapshot.iter().any(|required| {
            destination_coverage
                .missing(required.region, required.version)
                .into_iter()
                .any(|region| !source_coverage.covers(region, required.version))
        }) {
            return Err(ContentAuthorityError::StaleSource);
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
            declared: regions,
            next_version: 2,
            canonical,
            representations,
            guest_aliases: BTreeSet::new(),
            pending_gpu_writes: BTreeMap::new(),
            pending_transfers: BTreeMap::new(),
            pending_host_landings: BTreeMap::new(),
            pending_host_ingresses: BTreeMap::new(),
            guest_writes: BTreeMap::new(),
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
            declared: Box::new([BackingRegion::Whole]),
            next_version: 2,
            canonical,
            representations,
            guest_aliases: BTreeSet::new(),
            pending_gpu_writes: BTreeMap::new(),
            pending_transfers: BTreeMap::new(),
            pending_host_landings: BTreeMap::new(),
            pending_host_ingresses: BTreeMap::new(),
            guest_writes: BTreeMap::new(),
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

    /// Declare that `representation` addresses the guest's own pages, so it
    /// and [`GUEST_REPRESENTATION`] hold one content statement between them.
    ///
    /// It starts out holding exactly what the guest holds, because binding an
    /// object over pages copies nothing: the bytes it can read are the bytes
    /// already there. See [`RegionContentState::guest_aliases`].
    pub fn alias_guest_representation(&mut self, representation: RepresentationId) {
        self.representations.entry(representation).or_default();
        if representation == GUEST_REPRESENTATION {
            return;
        }
        let guest = self
            .representations
            .get(&GUEST_REPRESENTATION)
            .cloned()
            .unwrap_or_default();
        self.representations.insert(representation, guest);
        self.guest_aliases.insert(representation);
    }

    /// Assign coverage to one representation and to everything that shares its
    /// content statement. See [`RegionContentState::guest_aliases`].
    fn assign_coverage(
        &mut self,
        representation: RepresentationId,
        region: BackingRegion,
        version: ContentVersion,
        if_newer: bool,
    ) {
        let mut targets = vec![representation];
        if representation == GUEST_REPRESENTATION || self.guest_aliases.contains(&representation) {
            targets.extend(
                self.guest_aliases
                    .iter()
                    .copied()
                    .chain(std::iter::once(GUEST_REPRESENTATION))
                    .filter(|target| *target != representation),
            );
        }
        for target in targets {
            let Some(coverage) = self.representations.get_mut(&target) else {
                continue;
            };
            if if_newer {
                coverage.assign_if_newer(region, version);
            } else {
                coverage.assign(region, version);
            }
        }
    }

    /// Restate the guest as canonical over every declared region, because the
    /// physical pages under this backing have been replaced.
    ///
    /// A physical replacement is the guest re-pointing a resource's pages. The
    /// bytes behind the backing afterwards are the guest's, and nothing a
    /// native object held before describes them --- which is why replacement
    /// retires every designated representation. Content authority has to hear
    /// the same fact, or the canonical version keeps naming content that only
    /// the retired objects held.
    ///
    /// Leaving it unsaid is not a stale pixel, it is a permanent stall: the
    /// readiness check for a guest-visibility request asks for a designated
    /// representation holding the canonical version, finds none, and refuses
    /// `StaleExecutionRepresentation` --- forever, because no route plans a
    /// transfer out of an object that no longer exists. A driven macos-13
    /// conformance boot spent the rest of its life re-offering one synchronize
    /// at channel 4's head for exactly this reason, 25 s after the replacement
    /// that caused it.
    ///
    /// It is a guest write with no operation identity, the same statement a
    /// standalone invalidation packet makes, because that is precisely what
    /// happened: the guest replaced these bytes.
    pub fn guest_replaced_physical(&mut self) -> Result<(), ContentAuthorityError> {
        for region in self.declared.clone().iter().copied() {
            self.guest_write(None, GUEST_REPRESENTATION, region)?;
        }
        Ok(())
    }

    pub(crate) fn remove_representation(&mut self, representation: RepresentationId) {
        self.representations.remove(&representation);
        self.guest_aliases.remove(&representation);
    }

    /// Record that the guest wrote one region, under the operation that says so.
    ///
    /// `write` is what makes this idempotent: an operation that already wrote
    /// this region gets the version it was given rather than a new one, because
    /// re-preparing an operation is the same statement made twice. `None` is
    /// for the routes with no operation identity to offer -- a standalone
    /// invalidation packet, applied once -- and always mints.
    pub fn guest_write(
        &mut self,
        write: Option<GuestWriteId>,
        representation: RepresentationId,
        region: BackingRegion,
    ) -> Result<RegionVersion, ContentAuthorityError> {
        self.validate_guest_write(representation)?;
        // Repeated only while the version it left is still exactly what the
        // region holds. Anything that moved the region since -- a discard, a
        // completed GPU write, a transfer landing -- makes this preparation a
        // new statement about bytes that are no longer the ones it wrote, and
        // reusing the version there would hand the caller a version nothing
        // covers.
        if let Some((owner, version)) = self.guest_writes.get(&region) {
            if Some(*owner) == write
                && self.canonical.covers(region, *version)
                && self
                    .representations
                    .get(&representation)
                    .is_some_and(|coverage| coverage.covers(region, *version))
            {
                return Ok(RegionVersion {
                    region,
                    version: *version,
                });
            }
        }
        let version = self.reserve_version()?;
        if let Some(write) = write {
            self.guest_writes.insert(region, (write, version));
        } else {
            self.guest_writes.remove(&region);
        }
        self.canonical.assign(region, version);
        self.assign_coverage(representation, region, version, false);
        self.discarded = self
            .discarded
            .drain(..)
            .flat_map(|discarded| remaining_after(discarded, region))
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
        for write in planned.regions.iter().copied() {
            self.assign_coverage(representation, write.region, write.version, true);
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

    /// The parts of `snapshot` one representation does not already hold.
    ///
    /// [`RegionContentState::representation_matches`] collapses this to a
    /// yes/no, and the yes/no is the wrong shape for a *source* search. A
    /// backing whose GPU wrote four bytes of a page has its canonical content
    /// split across two representations -- the object that wrote those bytes,
    /// and the guest that still holds the rest -- so no single object covers
    /// the whole snapshot and a search for one refuses stale over content
    /// nothing is stale about. What a consumer is owed is only what it does
    /// not already hold, and that remainder is held whole by the object that
    /// produced it.
    pub fn outstanding_snapshot(
        &self,
        representation: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Box<[RegionVersion]> {
        let Some(coverage) = self.representations.get(&representation) else {
            return snapshot.to_vec().into_boxed_slice();
        };
        snapshot
            .iter()
            .flat_map(|required| {
                coverage
                    .missing(required.region, required.version)
                    .into_iter()
                    .map(|region| RegionVersion {
                        region,
                        version: required.version,
                    })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
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

    /// The parts of `required` one representation already holds, named in the
    /// asker's coordinates.
    ///
    /// This is the *read*: what a consumer wants to know is which of the bytes
    /// it named are current, in the terms it named them.
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

    /// The parts of `required` one representation can be asked to *transfer*,
    /// named in coordinates that representation can issue against.
    ///
    /// The sibling of the read above, and the distinction is the same one
    /// [`transferable_from`] draws: a source whose coverage is the complete
    /// backing has no finer coordinates to copy from, so narrowing to the
    /// consumer's region here produces a copy nothing downstream can express.
    /// Every caller that goes on to plan a transfer or an ingress asks this
    /// one, so the two questions cannot be confused at a call site.
    pub(crate) fn transferable_regions_in_representation(
        &self,
        representation: RepresentationId,
        required: RegionVersion,
    ) -> Box<[BackingRegion]> {
        self.representations
            .get(&representation)
            .into_iter()
            .flat_map(|coverage| coverage.entries.iter())
            .filter(|current| current.version == required.version)
            .filter_map(|current| transferable_from(required.region, current.region))
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
                        .flat_map(|region| not_covered_by(region, current.region))
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
        self.assign_coverage(GUEST_REPRESENTATION, landing.region, landing.version, false);
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
        self.assign_coverage(key.destination, key.region, key.version, false);
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
        self.representations.entry(representation).or_default();
        self.assign_coverage(representation, required.region, required.version, false);
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

    /// See [`RegionContentState::declared_regions`].
    pub fn declared_regions(&self) -> Box<[BackingRegion]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .declared_regions()
            .into()
    }

    pub fn validate_reservations(&self, count: usize) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .validate_reservations(count)
    }

    /// See [`RegionContentState::guest_replaced_physical`].
    pub fn guest_replaced_physical(&self) -> Result<(), ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .guest_replaced_physical()
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

    /// See [`RegionContentState::outstanding_snapshot`].
    pub fn outstanding_snapshot(
        &self,
        representation: RepresentationId,
        snapshot: &[RegionVersion],
    ) -> Box<[RegionVersion]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .outstanding_snapshot(representation, snapshot)
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

    /// See [`RegionContentState::transferable_regions_in_representation`].
    pub(crate) fn transferable_regions_in_representation(
        &self,
        representation: RepresentationId,
        required: RegionVersion,
    ) -> Box<[BackingRegion]> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .transferable_regions_in_representation(representation, required)
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

    /// See [`RegionContentState::alias_guest_representation`].
    pub fn alias_guest_representation(&self, representation: RepresentationId) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .alias_guest_representation(representation);
    }

    pub fn guest_write_region(
        &self,
        write: Option<GuestWriteId>,
        representation: RepresentationId,
        region: BackingRegion,
    ) -> Result<RegionVersion, ContentAuthorityError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .guest_write(write, representation, region)
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

/// The part of the backing both regions name, or `None` if they share none.
///
/// [`BackingRegion::Whole`] is the complete backing, so it contains every
/// other region and intersecting with it yields the other region unchanged.
/// That is not a translation into finer coordinates and does not need one: it
/// is the statement that all the bytes include these bytes, which holds
/// whatever the finer coordinates mean. The direction that *does* need a
/// translation is subtraction, and [`subtract`] refuses it.
pub(crate) fn intersection(left: BackingRegion, right: BackingRegion) -> Option<BackingRegion> {
    match (left, right) {
        (BackingRegion::Whole, BackingRegion::Whole) => Some(BackingRegion::Whole),
        (BackingRegion::Whole, region) | (region, BackingRegion::Whole) => Some(region),
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

/// The part of `region` outside `cut`, or `None` where the algebra cannot
/// express it.
///
/// The only inexpressible case is the complete backing minus part of it.
/// [`BackingRegion::Whole`] exists precisely for a backing whose contract
/// establishes no sound translation into finer coordinates, so "everything
/// except this box" has no name, and there is no honest region set to return.
/// Callers must decide which way to err; [`remaining_after`] and
/// [`not_covered_by`] are the two answers and every caller uses one of them.
fn subtract(region: BackingRegion, cut: BackingRegion) -> Option<Vec<BackingRegion>> {
    let Some(overlap) = intersection(region, cut) else {
        return Some(vec![region]);
    };
    if region == BackingRegion::Whole && overlap != BackingRegion::Whole {
        return None;
    }
    Some(match (region, overlap) {
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
        _ => unreachable!("an inexpressible remainder returned above"),
    })
}

/// The part of `required` a source covering `available` can be asked to
/// transfer, named in coordinates that source can address.
///
/// This is deliberately not [`intersection`]. A *read* asks whether bytes are
/// current and is answered in the reader's own coordinates, so whole-backing
/// coverage answers an image query about that image. A *transfer* has to be
/// issued against the source, and a source whose coverage is the complete
/// backing has no finer coordinates to issue against -- the only copy it can
/// perform is the whole backing. Naming the requirement's own region there
/// produces a copy request the source cannot express, and the refusal then
/// lands at the executor, on a transfer the planner should never have chosen.
///
/// Widening to the whole backing copies more than was asked for and copies the
/// right bytes. The alternative is a transfer that never runs.
pub(crate) fn transferable_from(
    required: BackingRegion,
    available: BackingRegion,
) -> Option<BackingRegion> {
    let overlap = intersection(required, available)?;
    Some(if available == BackingRegion::Whole {
        available
    } else {
        overlap
    })
}

/// The part of `region` that is provably still `region` and not `cut`.
///
/// Where the remainder has no expression this is empty, so a caller recording
/// what content it *holds* gives up a claim rather than keeping a stale one.
/// The complete backing minus a freshly written box is exactly that case:
/// keeping `Whole` at the old version would go on asserting the old content
/// over the bytes just overwritten, and a later read of those bytes would be
/// answered from it. Giving the claim up costs a synchronization that was not
/// needed; keeping it costs the wrong pixels, silently.
pub(crate) fn remaining_after(region: BackingRegion, cut: BackingRegion) -> Vec<BackingRegion> {
    subtract(region, cut).unwrap_or_default()
}

/// The part of `region` that `cut` has not been shown to cover.
///
/// The dual of [`remaining_after`], for a caller asking what still needs
/// filling. Where the remainder has no expression this is the whole region, so
/// the answer over-reports what is missing and plans a transfer that was not
/// needed rather than skipping one that was.
pub(crate) fn not_covered_by(region: BackingRegion, cut: BackingRegion) -> Vec<BackingRegion> {
    subtract(region, cut).unwrap_or_else(|| vec![region])
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

    /// The complete backing contains every finer region, for reading.
    ///
    /// `Whole` covers content whose contract establishes no translation into
    /// finer coordinates. That blocks *subdividing* it, and it does not block
    /// the containment: all the bytes include these bytes, whatever the finer
    /// coordinates mean. Before this held, an image query against `Whole`
    /// coverage returned nothing, `representation_matches` was an `all` over
    /// nothing and therefore vacuously true, and the synchronization that
    /// would have filled the view was dropped as already satisfied.
    #[test]
    fn whole_coverage_answers_a_finer_read_rather_than_nothing() {
        let mut state = state(BackingRegion::Whole);
        let write = state
            .plan_gpu_write(SubmissionId::new(1), GPU, [BackingRegion::Whole])
            .unwrap()[0];
        state.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();

        let region = image([0, 0, 0], [16, 16, 1]);
        assert_eq!(
            state.snapshot(&[region]).as_ref(),
            [RegionVersion {
                region,
                version: write.version,
            }]
        );
        assert!(state.representation_matches(GPU, &state.snapshot(&[region])));
        assert!(state.representation_matches(GPU, &state.snapshot(&[linear(0, 64)])));
    }

    /// An alias and the guest hold one content statement in both directions.
    ///
    /// The alias is a native object bound over the guest's own pages, so a GPU
    /// write through it is a write the guest can already read and a guest
    /// store is a store the object can already read. Without that, a `Shared`
    /// allocation on a unified host would owe a transfer back to the guest
    /// that Metal never asks for, and every readback of GPU-written bytes
    /// would return what the guest wrote last.
    #[test]
    fn a_guest_alias_and_the_guest_hold_one_content_statement() {
        let page = linear(0, 4096);
        let mut state = state(page);
        state.alias_guest_representation(GPU);
        assert!(state.representation_matches(GPU, &state.snapshot(&[page])));

        let head = linear(0, 4);
        state
            .plan_gpu_write(SubmissionId::new(1), GPU, [head])
            .unwrap();
        state.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();
        let snapshot = state.snapshot(&[page]);
        assert!(state.representation_matches(GUEST, &snapshot));
        assert!(state.outstanding_snapshot(GUEST, &snapshot).is_empty());

        let tail = linear(4, 4092);
        state.guest_write(None, GUEST, tail).unwrap();
        let snapshot = state.snapshot(&[page]);
        assert!(state.representation_matches(GPU, &snapshot));
        assert!(state.outstanding_snapshot(GPU, &snapshot).is_empty());
    }

    /// A source covers what will be copied, not what the consumer named.
    ///
    /// A compute readback of one `u32` writes four bytes of a page and leaves
    /// the rest with the guest, so the page's canonical content is split and
    /// no single representation covers it. Requiring one to made the guest's
    /// own readback route unsatisfiable: the transfer that would have served
    /// it plans nothing but the four bytes, and refusing it as a stale source
    /// refused content nothing was stale about.
    #[test]
    fn a_split_page_transfers_from_the_object_holding_the_part_that_moved() {
        let page = linear(0, 4096);
        let mut state = state(page);
        let head = linear(0, 4);
        state
            .plan_gpu_write(SubmissionId::new(1), GPU, [head])
            .unwrap();
        state.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();

        // The guest holds the tail and the GPU object holds the head, and
        // neither holds the page.
        let snapshot = state.snapshot(&[page]);
        assert!(!state.representation_matches(GPU, &snapshot));
        assert!(!state.representation_matches(GUEST, &snapshot));
        assert_eq!(
            state.outstanding_snapshot(GUEST, &snapshot).as_ref(),
            [RegionVersion {
                region: head,
                version: state.snapshot(&[head])[0].version,
            }]
        );

        state
            .validate_plan_transfers(GPU, GUEST, &snapshot)
            .unwrap();
        let planned = state.plan_transfers(GPU, GUEST, &snapshot).unwrap();
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].region, head);
    }

    /// Writing part of a whole-covered backing gives up the whole claim.
    ///
    /// "Everything except this box" has no expression, so the choice is
    /// between keeping `Whole` at the old version -- which goes on asserting
    /// the old content over the bytes just overwritten, and answers a later
    /// read of them from it -- and giving the claim up. Giving it up costs a
    /// synchronization that was not needed. Keeping it costs the wrong pixels,
    /// silently.
    #[test]
    fn a_partial_write_over_whole_coverage_gives_up_the_claim_it_cannot_narrow() {
        let mut state = state(BackingRegion::Whole);
        state
            .plan_gpu_write(SubmissionId::new(1), GPU, [BackingRegion::Whole])
            .unwrap();
        let old = state.complete_gpu_write(SubmissionId::new(1), GPU).unwrap()[0].version;

        let part = image([0, 0, 0], [8, 8, 1]);
        state
            .plan_gpu_write(SubmissionId::new(2), GPU, [part])
            .unwrap();
        let new = state.complete_gpu_write(SubmissionId::new(2), GPU).unwrap()[0].version;
        assert!(new > old);

        // The written box reads at the new version, and nothing anywhere still
        // reads at the old one.
        assert_eq!(
            state.snapshot(&[part]).as_ref(),
            [RegionVersion {
                region: part,
                version: new,
            }]
        );
        assert!(state
            .snapshot(&[BackingRegion::Whole, image([0, 0, 0], [16, 16, 1])])
            .iter()
            .all(|entry| entry.version == new));
    }

    /// The read and the transfer disagree about a whole-covered source, and
    /// each accessor answers only its own question.
    ///
    /// A consumer asking which of its bytes are current wants them back in the
    /// terms it named. A planner asking what a source can copy wants them in
    /// terms that source can issue, and a whole-backing source can only issue
    /// the whole backing. One accessor answering both narrowed every transfer
    /// to the consumer's coordinates, and a host ingress -- a byte copy --
    /// then arrived asking for a texel box.
    #[test]
    fn a_transfer_source_reports_regions_it_can_actually_issue() {
        let mut whole = state(BackingRegion::Whole);
        let write = whole
            .plan_gpu_write(SubmissionId::new(1), GPU, [BackingRegion::Whole])
            .unwrap()[0];
        let required = RegionVersion {
            region: image([0, 0, 0], [1280, 1024, 1]),
            version: write.version,
        };
        whole.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();

        assert_eq!(
            whole
                .current_regions_in_representation(GPU, required)
                .as_ref(),
            [required.region]
        );
        assert_eq!(
            whole
                .transferable_regions_in_representation(GPU, required)
                .as_ref(),
            [BackingRegion::Whole]
        );

        // A source that can address the requirement answers the same either
        // way, so the two only ever differ where it matters.
        let mut bytes = state(linear(0, 128));
        let byte_write = bytes
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(0, 128)])
            .unwrap()[0];
        bytes.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();
        let byte_required = RegionVersion {
            region: linear(0, 64),
            version: byte_write.version,
        };
        assert_eq!(
            bytes.current_regions_in_representation(GPU, byte_required),
            bytes.transferable_regions_in_representation(GPU, byte_required)
        );
    }

    /// A backing's coordinate vocabulary is what it was declared with, and
    /// writes never change it.
    ///
    /// Coverage entries subdivide as writes land, so the entries present at
    /// any moment say who wrote last and not what coordinates this backing can
    /// be addressed in. A validity statement built from the entries inherits
    /// whatever the previous writer used -- a render pass leaves texel boxes
    /// behind -- and the host ingress that follows it is a byte copy asked for
    /// in texels, which nothing downstream can issue.
    #[test]
    fn the_declared_coordinate_vocabulary_outlives_every_write_over_it() {
        let mut bytes = state(linear(0, 128));
        assert_eq!(bytes.declared_regions(), [linear(0, 128)]);

        bytes
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(32, 32)])
            .unwrap();
        bytes.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();

        // The entries have fragmented and the vocabulary has not.
        assert!(bytes.snapshot_all().len() > 1);
        assert_eq!(bytes.declared_regions(), [linear(0, 128)]);

        // A whole-backing declaration written through in texels keeps saying
        // the complete backing, which is the only form a byte copy can issue.
        let mut whole = state(BackingRegion::Whole);
        whole
            .plan_gpu_write(
                SubmissionId::new(1),
                GPU,
                [image([0, 0, 0], [1280, 1024, 1])],
            )
            .unwrap();
        whole.complete_gpu_write(SubmissionId::new(1), GPU).unwrap();
        assert_eq!(
            whole.snapshot_all()[0].region,
            image([0, 0, 0], [1280, 1024, 1])
        );
        assert_eq!(whole.declared_regions(), [BackingRegion::Whole]);
    }

    /// A transfer is named in coordinates its source can issue against.
    ///
    /// A read of whole-backing coverage is answered about the image the reader
    /// asked for, which is right for a read and wrong for a copy: the source
    /// has no finer coordinates to copy from, so the only transfer it can
    /// perform is the whole backing. Naming the reader's region instead
    /// produces a request the executor refuses, on a transfer the planner
    /// chose.
    #[test]
    fn a_transfer_from_whole_coverage_is_named_as_the_whole_backing() {
        let region = image([0, 0, 0], [1280, 1024, 1]);

        // The read and the transfer disagree, and each is right for its own
        // question.
        assert_eq!(intersection(region, BackingRegion::Whole), Some(region));
        assert_eq!(
            transferable_from(region, BackingRegion::Whole),
            Some(BackingRegion::Whole)
        );

        // Coverage that can address the requirement names the requirement, and
        // coverage that does not overlap it at all names nothing.
        assert_eq!(
            transferable_from(linear(0, 64), linear(0, 128)),
            Some(linear(0, 64))
        );
        assert_eq!(transferable_from(linear(0, 64), linear(64, 64)), None);

        // A whole-backing requirement against whole-backing coverage is the
        // whole backing either way.
        assert_eq!(
            transferable_from(BackingRegion::Whole, BackingRegion::Whole),
            Some(BackingRegion::Whole)
        );
    }

    /// Subtraction errs in the direction its caller must err in.
    ///
    /// The complete backing minus a box has no expression. A caller recording
    /// what it *holds* must give the claim up; a caller asking what is still
    /// *missing* must report all of it. One answer would be wrong for one of
    /// them, and both wrongs are silent.
    #[test]
    fn the_two_subtractions_disagree_only_where_the_remainder_has_no_name() {
        let part = image([2, 2, 0], [4, 4, 1]);
        assert!(remaining_after(BackingRegion::Whole, part).is_empty());
        assert_eq!(
            not_covered_by(BackingRegion::Whole, part),
            [BackingRegion::Whole]
        );

        // Everywhere the remainder does have a name they agree, including the
        // other direction: removing all the bytes removes any subset of them.
        assert!(remaining_after(part, BackingRegion::Whole).is_empty());
        assert!(not_covered_by(part, BackingRegion::Whole).is_empty());
        assert_eq!(
            remaining_after(linear(0, 128), linear(0, 64)),
            not_covered_by(linear(0, 128), linear(0, 64))
        );
        assert_eq!(
            remaining_after(BackingRegion::Whole, BackingRegion::Whole),
            not_covered_by(BackingRegion::Whole, BackingRegion::Whole)
        );
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

    /// One operation's guest write is one version however many times that
    /// operation is prepared.
    ///
    /// An EXEC that refuses after its resource states are prepared gives its
    /// claims up and prepares them again on the retry. Minting a version per
    /// preparation makes every representation of the backing stale on every
    /// retry -- including the one an upload is running to bring current, which
    /// then can never finish, which is what the retry is waiting for. The
    /// device live-locks with nothing refusing and every counter reading
    /// healthy; the only line that says so is the required version running away
    /// from what anything holds.
    #[test]
    fn one_operations_guest_write_is_one_version_however_often_it_is_prepared() {
        let mut state = state(linear(0, 128));
        let region = linear(0, 128);
        let operation = GuestWriteId::operation(TransactionId::new(7), SubmissionId::new(3), 1);

        let first = state.guest_write(Some(operation), GUEST, region).unwrap();
        for _ in 0..4 {
            assert_eq!(
                state.guest_write(Some(operation), GUEST, region).unwrap(),
                first,
                "re-preparing one operation repeats its statement"
            );
        }
        assert_eq!(state.snapshot(&[region]).as_ref(), [first]);

        // A different operation is a different write and takes its own version.
        let later = GuestWriteId::operation(TransactionId::new(7), SubmissionId::new(3), 2);
        let second = state.guest_write(Some(later), GUEST, region).unwrap();
        assert!(second.version > first.version);

        // And the first operation prepared again after that is a new statement
        // too: the region it wrote is no longer the one it left behind.
        let again = state.guest_write(Some(operation), GUEST, region).unwrap();
        assert!(again.version > second.version);

        // Discarding the region takes its coverage away, so the repeat is no
        // longer a repeat: handing back a version nothing covers would make
        // every later read of it refuse as a stale source.
        state.discard(region);
        let after_discard = state.guest_write(Some(operation), GUEST, region).unwrap();
        assert!(after_discard.version > again.version);
        assert_eq!(state.snapshot(&[region]).as_ref(), [after_discard]);
    }

    #[test]
    fn stale_overlapping_gpu_completion_cannot_replace_a_newer_guest_write() {
        let mut state = state(linear(0, 128));
        state
            .plan_gpu_write(SubmissionId::new(1), GPU, [linear(0, 128)])
            .unwrap();
        let guest = state.guest_write(None, GUEST, linear(32, 32)).unwrap();
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
        let first = state.guest_write(None, GUEST, linear(0, 128)).unwrap();
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

        let write = state
            .guest_write(None, GUEST_REPRESENTATION, region)
            .unwrap();
        let ingress = state.plan_host_ingress(write).unwrap();
        assert!(!state.representation_matches(HOST_REPRESENTATION, &[write]));
        state.remove_representation_region(GUEST_REPRESENTATION, region);
        assert!(!state.representation_matches(GUEST_REPRESENTATION, &[write]));
        state.complete_host_ingress(ingress).unwrap();
        assert!(state.representation_matches(HOST_REPRESENTATION, &[write]));

        let stale = state
            .guest_write(None, GUEST_REPRESENTATION, region)
            .unwrap();
        let stale_ingress = state.plan_host_ingress(stale).unwrap();
        state
            .guest_write(None, GUEST_REPRESENTATION, region)
            .unwrap();
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
        state.guest_write(None, GUEST, linear(16, 32)).unwrap();
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
        state.guest_write(None, GUEST, linear(0, 64)).unwrap();
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
        let remainder = remaining_after(whole, center);
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
        let restored = state.guest_write(None, GUEST, linear(32, 32)).unwrap();
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
