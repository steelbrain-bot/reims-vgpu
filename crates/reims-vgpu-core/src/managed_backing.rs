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
use reims_vgpu_protocol::{
    BackingId, QueueOwnerId, QueueTimelineValue, RepresentationId, ResourceId, ResourceObject,
    ResourceValidityOps, TransactionId, VulkanDeviceEpochId,
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
    /// The representation serving each view of these bytes.
    ///
    /// A backing owes at most one buffer, and one image for every texture
    /// declared over its range. Metal aliases textures over one allocation
    /// deliberately and each keeps its own format and geometry, so this is a
    /// map rather than the single designation it used to be: with one slot the
    /// first alias to materialize served every other one, and no image view
    /// bridges two formats of different bit widths.
    execution_representations: BTreeMap<BackingView, RepresentationId>,
    retiring_representations: BTreeSet<RepresentationId>,
    accepted_uses: BTreeMap<TransactionId, BTreeMap<RepresentationId, usize>>,
}

/// The subregions a set of native representations currently hold at one
/// canonical version. A version may be replicated regionally, so a request is
/// answered by however many representations it takes to cover it.
pub type RepresentationRegionCoverage = Vec<(RepresentationId, Box<[BackingRegion]>)>;

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
    /// The representation is still owned by its backing, but its native object
    /// has already been taken for retirement. This is a different fact from
    /// [`Self::UnknownRepresentation`] and reaches the same call sites: one
    /// says the identity was never registered or is fully gone, the other says
    /// a use outlived the object it names.
    RepresentationNativeReleased,
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

/// The texture that owns a native image.
///
/// A native image belongs to the texture that declared the storage it covers.
/// A texture *view* — a format reinterpretation, a mip or slice window, a
/// swizzle — is a view onto that image and never owns one, so a view is never
/// the key. Its own native view is installed onto the base's image and found
/// there by the resource that names it.
///
/// This is a type rather than a convention because the correct key and the
/// wrong one are one field apart on the same struct and agree on every plain
/// texture, where the base is the resource. Only a bound view separates them,
/// and it separates them silently: keying a view by itself resolves an image
/// that was built for it and had no view installed onto it, so the record
/// refuses with the shader view absent rather than with anything that names
/// the substitution.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ImageOwner(ResourceId<ResourceObject>);

impl ImageOwner {
    /// A resource that owns the storage its image covers, and so owns the
    /// image.
    ///
    /// For the resolvers this is the owner an endpoint already carries, and
    /// for a materialized declaration it is the declared resource: what
    /// reaches a materializer owns storage by construction.
    pub const fn owning(texture: ResourceId<ResourceObject>) -> Self {
        Self(texture)
    }

    /// The owner a resolved binding reads through, whatever the guest named.
    pub const fn of_view(view: crate::ResolvedTextureBindingView) -> Self {
        Self(view.image_owner)
    }

    /// The owning texture, for diagnostics and for the graph lookups keyed by
    /// resource identity.
    pub const fn texture(self) -> ResourceId<ResourceObject> {
        self.0
    }
}

/// Which view of a backing's bytes a native object is.
///
/// A backing is one run of guest bytes and its representations are views of
/// them. A guest allocation may be declared as both a buffer and a linear
/// texture, and those two views need different native objects, so the view is
/// stated by the materializer, which knows, rather than inferred later by a
/// consumer that does not.
///
/// The image view is named by the texture that declared it, because a backing
/// may carry more than one. Two textures declared over one guest range are two
/// textures under the contract — Metal aliases them deliberately and each
/// keeps its own format, geometry and subresource ranges — so each needs its
/// own native image over the same bytes. Keying the view by the backing alone
/// gave a backing a single image, which meant the first alias to materialize
/// served every other one; at 32 bits per texel against 64 no image view
/// reinterprets between them, and both the binding and the attachment paths
/// refused on that pair.
///
/// The texture, not its declaration, is the key. Two textures whose
/// declarations happen to match are still two textures, and a resource
/// re-created in a later generation is a different one for free.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BackingView {
    /// The backing's bytes, addressed linearly.
    Bytes,
    /// One texture's texels over the backing, addressed by subresource and
    /// coordinate.
    Image(ImageOwner),
}

/// One endpoint's native object, named by the backing it addresses and by
/// which view of that backing the endpoint is.
///
/// A guest allocation declared as both a buffer and a linear texture is one
/// backing with two views. An operation names each endpoint by role — a blit's
/// buffer side, a binding's descriptor class — and this is that role carried
/// through preparation, so a recorder asks for the buffer of a backing whose
/// execution object is an image rather than resolving whichever single
/// representation the backing designated.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ViewRepresentation {
    pub backing: BackingId,
    pub view: BackingView,
    pub representation: RepresentationId,
}

impl ViewRepresentation {
    /// The native object one endpoint resolves to.
    ///
    /// Every set is built from a map keyed by backing and view, so it is
    /// sorted on that pair and a lookup is a search rather than a scan.
    pub fn lookup(
        representations: &[Self],
        backing: BackingId,
        view: BackingView,
    ) -> Option<RepresentationId> {
        representations
            .binary_search_by_key(&(backing, view), |entry| (entry.backing, entry.view))
            .ok()
            .map(|index| representations[index].representation)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationUse {
    pub backing: BackingId,
    pub representations: Box<[RepresentationId]>,
}

/// One exact native identity retained by an accepted transaction.
///
/// Native backends use this list to acquire object-lifetime leases before an
/// immutable recording request leaves the resource owner. The identity list
/// comes from the accepted-use ledger itself, so recording does not have to
/// reconstruct it from raw handles or backing-wide guesses.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AcceptedRepresentation {
    pub backing: BackingId,
    pub representation: RepresentationId,
}

/// See [`ManagedBackingOwner::representation_census`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationCensus {
    pub representation: RepresentationId,
    pub has_native: bool,
    pub retiring: bool,
    pub accepted_uses: usize,
    pub last_uses: usize,
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
                execution_representations: BTreeMap::new(),
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
        known_representation(record, representation)?;
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
        known_representation(record, source)?;
        known_representation(record, destination)?;
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
        view: BackingView,
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
        if record.execution_representations.contains_key(&view) {
            return Err(ManagedRepresentationFailure {
                reason: ManagedBackingError::DuplicateExecutionRepresentation,
                native,
            });
        }
        let representation = self.create_representation(backing, route, native)?;
        let record = self
            .backings
            .get_mut(&backing)
            .expect("representation creation retained its live backing");
        record
            .execution_representations
            .insert(view, representation);
        Ok(representation)
    }

    /// The representation serving one view of a backing's bytes.
    ///
    /// A view a materialization built is answered directly. An image view that
    /// has not been built is named rather than substituted: it is a texture
    /// declared over these bytes whose native image does not exist yet, and
    /// serving it with another texture's image is exactly the substitution
    /// that made two formats collide.
    ///
    /// The byte view is the one that can still be answered without having been
    /// designated, because a backing that owes an image also owes the endpoint
    /// that image transfers through, and that endpoint is a property of the
    /// backing's own storage rather than of any one texture over it. Every
    /// image on a backing shares its guest bytes and therefore its route
    /// family, so which image the route is read from does not change the
    /// answer.
    pub fn view_representation(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Result<RepresentationId, ManagedBackingError> {
        let record = self
            .backings
            .get(&backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
        if let Some(representation) = record.execution_representations.get(&view) {
            return Ok(*representation);
        }
        if record.execution_representations.is_empty() {
            return Err(ManagedBackingError::MissingExecutionRepresentation);
        }
        if matches!(view, BackingView::Image(_)) {
            return Err(ManagedBackingError::MissingExecutionRepresentation);
        }
        // The byte endpoint the images transfer through. The route names it:
        // an imported or directly aliased backing keeps the guest's own pages,
        // and a staged one keeps the host copy.
        let route = record
            .execution_representations
            .values()
            .find_map(|representation| {
                record
                    .representations
                    .get(representation)
                    .map(|held| held.route)
            });
        let endpoint = match route {
            Some(
                RepresentationRoute::ImportedGuestTransfer { .. }
                | RepresentationRoute::DirectGuestAlias,
            ) => crate::GUEST_REPRESENTATION,
            Some(RepresentationRoute::HostStagingTransfer { .. }) => crate::HOST_REPRESENTATION,
            _ => return Err(ManagedBackingError::MissingExecutionRepresentation),
        };
        if record.representations.contains_key(&endpoint) {
            Ok(endpoint)
        } else {
            Err(ManagedBackingError::UnknownRepresentation)
        }
    }

    /// Exact execution identity and native object one view was built with.
    pub fn execution_representation(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Option<(RepresentationId, &T)> {
        let record = self.backings.get(&backing)?;
        let representation = *record.execution_representations.get(&view)?;
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

    /// Select an owned native representation that contains the exact current
    /// snapshot. This is used when physical replacement leaves the prior
    /// execution object as the only current transfer source until its accepted
    /// uses retire.
    pub fn current_native_representation_for_snapshot(
        &self,
        backing: BackingId,
        excluded: &[RepresentationId],
        snapshot: &[RegionVersion],
    ) -> Result<Option<RepresentationId>, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        Ok(record
            .representations
            .iter()
            .find(|(representation, native)| {
                !excluded.contains(representation)
                    && native.native.is_some()
                    && record
                        .authority
                        .representation_matches(**representation, snapshot)
            })
            .map(|(&representation, _)| representation))
    }

    /// Return the exact current subregions available from every owned native
    /// representation. A canonical version may be replicated regionally, so
    /// no single representation is required to cover the whole request.
    pub fn current_native_regions_for_version(
        &self,
        backing: BackingId,
        excluded: &[RepresentationId],
        required: RegionVersion,
    ) -> Result<RepresentationRegionCoverage, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        Ok(record
            .representations
            .iter()
            .filter(|(representation, native)| {
                !excluded.contains(representation) && native.native.is_some()
            })
            .filter_map(|(&representation, _)| {
                let regions = record
                    .authority
                    .current_regions_in_representation(representation, required);
                (!regions.is_empty()).then_some((representation, regions))
            })
            .collect())
    }

    pub fn current_regions_in_representation(
        &self,
        backing: BackingId,
        representation: RepresentationId,
        required: RegionVersion,
    ) -> Result<Box<[BackingRegion]>, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        known_representation(record, representation)?;
        Ok(record
            .authority
            .current_regions_in_representation(representation, required))
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

    /// What one backing's execution representation currently holds, for a
    /// diagnostic that has only the backing to go on.
    ///
    /// `StaleExecutionRepresentation` names the backing and nothing else, and
    /// the operation's own required snapshot is not in reach from where that
    /// refusal is reported. This is the other half: pair it with the regions
    /// the blocked operation names and the disagreement is readable.
    pub fn execution_representation_coverage(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Option<(RepresentationId, Vec<RegionVersion>)> {
        let record = self.backings.get(&backing)?;
        let representation = *record.execution_representations.get(&view)?;
        Some((
            representation,
            record.authority.representation_coverage(representation),
        ))
    }

    /// Any one of a backing's designated representations.
    ///
    /// For questions about the backing's *storage* rather than about a
    /// texture: which transfer route it takes, which endpoint it stages
    /// through. Every representation over one backing addresses the same guest
    /// bytes and is therefore built on the same route family, so which one
    /// answers does not change the answer. A question that a different view
    /// would answer differently is a question that must name its view.
    pub fn any_designated_representation(
        &self,
        backing: BackingId,
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.live_backing(backing)?
            .execution_representations
            .values()
            .next()
            .copied()
            .ok_or(ManagedBackingError::MissingExecutionRepresentation)
    }

    /// Any designated representation holding the exact regional content an
    /// immutable operation requires, with its native object.
    ///
    /// The view-free counterpart of
    /// [`Self::execution_representation_for_snapshot`], for a caller reading a
    /// backing's content rather than one texture's. Content is per
    /// representation, so this can find one current object while another over
    /// the same bytes is stale; the refusal is
    /// [`ManagedBackingError::StaleExecutionRepresentation`] only when none
    /// of them holds it.
    pub fn designated_representation_for_snapshot(
        &self,
        backing: BackingId,
        snapshot: &[RegionVersion],
    ) -> Result<(RepresentationId, &T), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if record.execution_representations.is_empty() {
            return Err(ManagedBackingError::MissingExecutionRepresentation);
        }
        record
            .execution_representations
            .values()
            .find(|representation| {
                record
                    .authority
                    .representation_matches(**representation, snapshot)
            })
            .and_then(|representation| {
                Some((
                    *representation,
                    record
                        .representations
                        .get(representation)?
                        .native
                        .as_ref()?,
                ))
            })
            .ok_or(ManagedBackingError::StaleExecutionRepresentation)
    }

    /// Whether one representation is still a view this backing designates.
    ///
    /// A recorded plan names exact representation identities, and the question
    /// it asks later is whether they are still current — not which view they
    /// serve. Replacement retires every designation at once, so membership is
    /// the whole answer.
    pub fn is_designated(&self, backing: BackingId, representation: RepresentationId) -> bool {
        self.backings.get(&backing).is_some_and(|record| {
            record
                .execution_representations
                .values()
                .any(|designated| *designated == representation)
        })
    }

    /// Every view of a backing a materialization has designated, in view
    /// order.
    ///
    /// A backing may carry an image for each texture declared over its range,
    /// so a caller that owes work to *the* execution representation — content
    /// synchronization is the one that does — owes it to all of them. Asking
    /// for one and serving the rest with it is what a single designation used
    /// to do silently.
    pub fn designated_views(
        &self,
        backing: BackingId,
    ) -> Result<Vec<(BackingView, RepresentationId)>, ManagedBackingError> {
        Ok(self
            .live_backing(backing)?
            .execution_representations
            .iter()
            .map(|(view, representation)| (*view, *representation))
            .collect())
    }

    /// Every designated view whose representation does not already hold the
    /// required content, in view order.
    ///
    /// The answer to "is this backing current?" is per view and there is no
    /// single answer for the backing. Each designated view carries its own
    /// native object over the same bytes and its own content record, so one
    /// may be current while another holds nothing at all --- and a caller that
    /// asks one representation and generalises drops the work the others owe.
    /// An empty result is the only reading that means the backing needs
    /// nothing.
    ///
    /// This exists so that question cannot be asked one view at a time by
    /// accident. Callers that must synchronize, or must decide whether a
    /// synchronization request is necessary, ask here.
    pub fn stale_designated_representations(
        &self,
        backing: BackingId,
        snapshot: &[RegionVersion],
    ) -> Result<Vec<(BackingView, RepresentationId)>, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        Ok(record
            .execution_representations
            .iter()
            .filter(|(_, representation)| {
                !record
                    .authority
                    .representation_matches(**representation, snapshot)
            })
            .map(|(view, representation)| (*view, *representation))
            .collect())
    }

    pub fn execution_representation_id(
        &self,
        backing: BackingId,
        view: BackingView,
    ) -> Result<RepresentationId, ManagedBackingError> {
        self.live_backing(backing)?
            .execution_representations
            .get(&view)
            .copied()
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
        let designated = std::mem::take(&mut record.execution_representations);
        if designated.is_empty() {
            return Ok(ManagedBackingProgress::Live);
        }
        for representation in designated.into_values() {
            if !record.retiring_representations.insert(representation) {
                return Err(ManagedBackingError::ExecutionRepresentationAlreadyRetiring);
            }
        }
        self.finish_retirement_if_ready(backing)
    }

    pub fn validate_replace_execution_representation(
        &self,
        backing: BackingId,
    ) -> Result<(), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        if record
            .execution_representations
            .values()
            .any(|representation| record.retiring_representations.contains(representation))
        {
            return Err(ManagedBackingError::ExecutionRepresentationAlreadyRetiring);
        }
        Ok(())
    }

    /// Select the representation serving one view of a backing only when it
    /// holds the exact regional content snapshot required by an immutable
    /// operation.
    ///
    /// The currency check is the point on the non-execution view: a backing
    /// whose execution representation is an image serves its byte view through
    /// the transfer endpoint, and reading that endpoint is only sound once the
    /// image's writes have landed back in it. That is the same statement
    /// [`Self::execution_representation_for_snapshot`] makes about the
    /// execution object, asked of whichever object the view names.
    pub fn view_representation_for_snapshot(
        &self,
        backing: BackingId,
        view: BackingView,
        snapshot: &[RegionVersion],
    ) -> Result<RepresentationId, ManagedBackingError> {
        let representation = self.view_representation(backing, view)?;
        let record = self.live_backing(backing)?;
        if !record
            .authority
            .representation_matches(representation, snapshot)
        {
            return Err(ManagedBackingError::StaleExecutionRepresentation);
        }
        Ok(representation)
    }

    /// Select the construction-designated execution object only when it holds
    /// the exact regional content snapshot required by an immutable operation.
    pub fn execution_representation_for_snapshot(
        &self,
        backing: BackingId,
        view: BackingView,
        snapshot: &[RegionVersion],
    ) -> Result<(RepresentationId, &T), ManagedBackingError> {
        let record = self.live_backing(backing)?;
        let representation = record
            .execution_representations
            .get(&view)
            .copied()
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
        known_representation(record, representation)?;
        Ok(record
            .authority
            .representation_matches(representation, snapshot))
    }

    pub fn pending_gpu_writes_overlapping(
        &self,
        backing: BackingId,
        representation: RepresentationId,
        regions: &[BackingRegion],
    ) -> Result<Box<[GpuWriteId]>, ManagedBackingError> {
        let record = self.live_backing(backing)?;
        known_representation(record, representation)?;
        Ok(record
            .authority
            .pending_gpu_writes_overlapping(representation, regions))
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
        known_representation(record, source)?;
        known_representation(record, destination)?;
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
        known_representation(record, source)?;
        known_representation(record, destination)?;
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
        known_representation(record, source)?;
        known_representation(record, destination)?;
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
        known_representation(record, transfer.destination)?;
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

    /// Every representation this backing owns, with the four facts that decide
    /// whether a completion naming one can still be applied.
    ///
    /// A resource completion refused for a representation is a lifetime
    /// question — was the identity never registered, has its native object
    /// already been taken for retirement, does an accepted use still hold it,
    /// which queue points does it still owe — and none of those is observable
    /// from any other reading this device takes. Without them the refusal names
    /// a representation and nothing about why it is in that state.
    pub fn representation_census(&self, backing: BackingId) -> Vec<RepresentationCensus> {
        let Some(record) = self.backings.get(&backing) else {
            return Vec::new();
        };
        record
            .representations
            .iter()
            .map(|(&representation, native)| RepresentationCensus {
                representation,
                has_native: native.native.is_some(),
                retiring: record.retiring_representations.contains(&representation),
                accepted_uses: record
                    .accepted_uses
                    .values()
                    .filter(|uses| uses.contains_key(&representation))
                    .count(),
                last_uses: native.last_uses.len(),
            })
            .collect()
    }

    /// Completing a transfer retires an authority promise; it does not touch a
    /// native object, so it does not require one.
    ///
    /// A physical-incarnation replacement revokes the execution representation
    /// while a transfer into it is still in flight. Its native object is then
    /// held by the retirement owner against that very obligation, and its
    /// authority record deliberately survives — `finish_retirement_if_ready`
    /// removes the authority record only on `Ready`, never on `Deferred`. The
    /// completion that arrives afterwards still names the old representation
    /// and consumes exactly that pending record, which is the case
    /// `apply_replacement_timeline_observation` orders resource completion
    /// ahead of native retirement for.
    ///
    /// So the gate here is the authority, which already requires the transfer
    /// to be planned and its destination to be known to it. Requiring a live
    /// native object as well refused the completion of every transfer whose
    /// destination had been revoked, which stalls the whole observed batch —
    /// and [`Self::validate_complete_gpu_write`], the other completion arm over
    /// the same records, has never required one.
    pub fn validate_complete_transfer(&self, key: TransferKey) -> Result<(), ManagedBackingError> {
        let record = self
            .backings
            .get(&key.backing)
            .ok_or(ManagedBackingError::UnknownBacking)?;
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

    pub fn accepted_representations(
        &self,
        transaction: TransactionId,
        backings: &[BackingId],
    ) -> Result<Box<[AcceptedRepresentation]>, crate::ResourceUseBatchError> {
        let mut unique = BTreeSet::new();
        let mut accepted = Vec::new();
        for &backing in backings {
            if !unique.insert(backing) {
                return Err(crate::ResourceUseBatchError::DuplicateBacking(backing));
            }
            let record =
                self.backings
                    .get(&backing)
                    .ok_or(crate::ResourceUseBatchError::Backing {
                        backing,
                        reason: ManagedBackingError::UnknownBacking,
                    })?;
            let representations = record.accepted_uses.get(&transaction).ok_or(
                crate::ResourceUseBatchError::Backing {
                    backing,
                    reason: ManagedBackingError::UnknownAcceptedUse,
                },
            )?;
            for &representation in representations.keys() {
                accepted.push(AcceptedRepresentation {
                    backing,
                    representation,
                });
            }
        }
        Ok(accepted.into_boxed_slice())
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
        let retirement_candidates = record
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
        let mut unavailable = record
            .representations
            .iter()
            .filter_map(|(&representation, native)| {
                native.native.is_none().then_some(representation)
            })
            .collect::<Vec<_>>();
        let mut retiring_representations = Vec::new();
        for representation in retirement_candidates {
            if record
                .authority
                .representation_can_retire(representation, &unavailable)
            {
                unavailable.push(representation);
                retiring_representations.push(representation);
            }
        }

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

/// Whether one representation can serve as an end of a transfer or write, and
/// which way it cannot.
///
/// The guest representation is always available: it names guest memory rather
/// than a native object. Every other identity must be owned by this backing and
/// still hold its native object, and the two ways that can fail are separate
/// diagnoses — an identity that was never registered or has fully retired, and
/// one whose object was taken while a use still named it.
fn known_representation<T>(
    backing: &ManagedBacking<T>,
    representation: RepresentationId,
) -> Result<(), ManagedBackingError> {
    if representation == GUEST_REPRESENTATION {
        return Ok(());
    }
    match backing.representations.get(&representation) {
        Some(record) if record.native.is_some() => Ok(()),
        Some(_) => Err(ManagedBackingError::RepresentationNativeReleased),
        None => Err(ManagedBackingError::UnknownRepresentation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackingRegion, ContentAuthority, ContentAuthorityError};
    use reims_vgpu_protocol::SubmissionId;

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
                BackingView::Bytes,
                "execution",
            )
            .unwrap();
        assert_ne!(transfer, execution);
        assert_eq!(
            owner.execution_representation(backing, BackingView::Bytes),
            Some((execution, &"execution"))
        );
        let failure = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
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
            owner.execution_representation(backing, BackingView::Bytes),
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
                BackingView::Bytes,
                "old",
            )
            .unwrap();
        owner.accept_use(backing, transaction, [old]).unwrap();
        assert_eq!(
            owner
                .accepted_representations(transaction, &[backing])
                .unwrap()
                .as_ref(),
            [AcceptedRepresentation {
                backing,
                representation: old,
            }]
        );

        assert_eq!(
            owner.replace_execution_representation(backing).unwrap(),
            ManagedBackingProgress::Live
        );
        assert_eq!(
            owner.execution_representation_id(backing, BackingView::Bytes),
            Err(ManagedBackingError::MissingExecutionRepresentation)
        );
        let new = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::NativeWorking {
                    memory: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
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
                BackingView::Bytes,
                "execution",
            )
            .unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        assert_eq!(
            owner.execution_representation_for_snapshot(backing, BackingView::Bytes, &snapshot),
            Err(ManagedBackingError::StaleExecutionRepresentation)
        );

        let transfer = owner
            .plan_transfers(backing, GUEST_REPRESENTATION, execution, &snapshot)
            .unwrap();
        owner.complete_transfer(transfer[0]).unwrap();
        let write = SubmissionId::new(11);
        owner
            .plan_gpu_write(backing, write, execution, [BackingRegion::Whole])
            .unwrap();
        owner.complete_gpu_write(backing, write, execution).unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        assert_eq!(
            owner.execution_representation_for_snapshot(backing, BackingView::Bytes, &snapshot),
            Ok((execution, &"execution"))
        );

        let transaction = TransactionId::new(10);
        owner.accept_use(backing, transaction, [execution]).unwrap();
        owner.replace_execution_representation(backing).unwrap();
        let replacement = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                "replacement",
            )
            .unwrap();
        assert_eq!(
            owner
                .current_native_representation_for_snapshot(
                    backing,
                    &[GUEST_REPRESENTATION, HOST_REPRESENTATION, replacement],
                    &snapshot,
                )
                .unwrap(),
            Some(execution)
        );
        assert_eq!(
            owner.cancel_use(backing, transaction).unwrap(),
            ManagedBackingProgress::Live
        );
        assert_eq!(owner.representation(backing, execution), Some(&"execution"));

        let transfer = owner
            .plan_transfers(backing, execution, replacement, &snapshot)
            .unwrap();
        owner.complete_transfer(transfer[0]).unwrap();
        let replacement_use = TransactionId::new(12);
        owner
            .accept_use(backing, replacement_use, [replacement])
            .unwrap();
        assert_eq!(
            owner.cancel_use(backing, replacement_use).unwrap(),
            ManagedBackingProgress::RepresentationsRetired {
                ready: vec!["execution"],
                deferred: 0,
            }
        );
        assert_eq!(owner.representation(backing, execution), None);

        owner.guest_write(backing, BackingRegion::Whole).unwrap();
        let newer = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        assert_eq!(
            owner.execution_representation_for_snapshot(backing, BackingView::Bytes, &newer),
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
                BackingView::Bytes,
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
                BackingView::Bytes,
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

    /// A physical-incarnation replacement revokes the execution representation
    /// while a transfer into it is still in flight. The retirement owner then
    /// holds its native object against that obligation and the authority record
    /// survives, so the completion arriving afterwards must consume that record
    /// rather than be refused for the object it no longer needs. Refusing it
    /// fails the entire observed batch, which stops every later retirement on
    /// that queue for the rest of the boot.
    #[test]
    fn a_transfer_completes_into_a_representation_whose_native_object_was_revoked() {
        let (mut owner, backing) = owner();
        let destination = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                "working",
            )
            .unwrap();
        owner.guest_write(backing, BackingRegion::Whole).unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        let key = owner
            .plan_transfers(backing, GUEST_REPRESENTATION, destination, &snapshot)
            .unwrap()[0];

        let transaction = TransactionId::new(1);
        let submitted = point(1, 4);
        owner
            .accept_use(backing, transaction, [destination])
            .unwrap();
        owner.submit_use(backing, transaction, submitted).unwrap();

        // The guest reconstructs the resource: the execution identity is
        // revoked while this transfer is still in flight.
        owner.replace_execution_representation(backing).unwrap();
        let census = owner
            .representation_census(backing)
            .into_iter()
            .find(|entry| entry.representation == destination)
            .expect("the revoked representation is retained until its obligation clears");
        assert!(!census.has_native);

        // The completion consumes the pending authority record, which is the
        // whole of what completing a transfer owns.
        owner.complete_transfer(key).unwrap();
        assert_eq!(
            owner.complete_transfer(key),
            Err(ManagedBackingError::Content(
                ContentAuthorityError::TransferNotPlanned
            ))
        );
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

    /// The refusal names the backing; this is what the backing holds instead.
    ///
    /// A `StaleExecutionRepresentation` is reported from a place that has the
    /// failure and not the operation's required snapshot, so without this the
    /// only reading available is "some content is not current" -- which is what
    /// the refusal already said. A boot spent twenty-eight thousand retries on
    /// one of these.
    /// Two textures declared over one guest range are two textures, and a
    /// backing that gave them one image served whichever materialized first to
    /// both. At 32 bits per texel against 64 no image view reinterprets
    /// between them, so the second alias lost every binding and every
    /// attachment it named.
    #[test]
    fn two_textures_over_one_backing_each_get_their_own_image() {
        let (mut owner, backing) = owner();
        let wide = BackingView::Image(ImageOwner::owning(ResourceId::new(7, 1)));
        let narrow = BackingView::Image(ImageOwner::owning(ResourceId::new(8, 1)));

        let wide_representation = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                wide,
                "wide",
            )
            .unwrap();
        // The same backing, a second texture: not a duplicate designation.
        let narrow_representation = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                narrow,
                "narrow",
            )
            .unwrap();
        assert_ne!(wide_representation, narrow_representation);

        // Each view resolves to its own image, and neither is substituted for
        // the other -- which is the whole defect this key exists to prevent.
        assert_eq!(
            owner.view_representation(backing, wide),
            Ok(wide_representation)
        );
        assert_eq!(
            owner.view_representation(backing, narrow),
            Ok(narrow_representation)
        );
        assert_eq!(
            owner.execution_representation(backing, wide).map(|_| ()),
            Some(())
        );

        // A third texture over the same bytes has no image yet, and that is
        // named rather than answered with one of the two that do.
        assert_eq!(
            owner.view_representation(
                backing,
                BackingView::Image(ImageOwner::owning(ResourceId::new(9, 1)))
            ),
            Err(ManagedBackingError::MissingExecutionRepresentation)
        );

        // Re-declaring the same texture is still a duplicate.
        let duplicate = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                wide,
                "again",
            )
            .expect_err("one texture owns one image over one backing");
        assert_eq!(
            duplicate.reason,
            ManagedBackingError::DuplicateExecutionRepresentation
        );
        assert_eq!(duplicate.native, "again");

        assert_eq!(
            owner.designated_views(backing).unwrap(),
            vec![(wide, wide_representation), (narrow, narrow_representation)]
        );
        assert!(owner.is_designated(backing, narrow_representation));
    }

    #[test]
    fn a_backing_can_say_what_its_execution_representation_holds() {
        let (mut owner, backing) = owner();
        assert_eq!(
            owner.execution_representation_coverage(backing, BackingView::Bytes),
            None,
            "a backing with no execution representation holds nothing to report"
        );

        let execution = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                "execution",
            )
            .unwrap();
        let (reported, coverage) = owner
            .execution_representation_coverage(backing, BackingView::Bytes)
            .unwrap();
        assert_eq!(reported, execution);
        assert_eq!(
            coverage,
            owner
                .live_backing(backing)
                .unwrap()
                .authority
                .representation_coverage(execution),
            "the reported coverage is the authority's own account and not a second one"
        );

        assert_eq!(
            owner.execution_representation_coverage(BackingId::new(99), BackingView::Bytes),
            None,
            "a backing this owner does not hold has no coverage to report"
        );
    }
}
