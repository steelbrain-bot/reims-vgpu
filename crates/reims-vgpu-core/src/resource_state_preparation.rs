//! Lifecycle-owned preparation of resource-state operations.
//!
//! The admitted operation supplies the semantic statement and the resource
//! owner supplies every content reservation and transfer key. Native code can
//! project this opaque result, but cannot author a second transfer set.

use crate::{
    AdmittedResourceStateOperation, AdmittedResourceStates, GpuWriteReservation, HostIngressKey,
    HostIngressTransfer, HostLandingKey, ManagedBackingProgress, RegionVersion, RepresentationUse,
    ResolvedResourceCompletion, ResolvedResourceLifecycle, ResolvedResourceState,
    ResolvedValidityTransition, ResourceLifecycleEffect, ResourceLifecycleError,
    ResourceLifecycleOwner, ResourceUseBatchError, ResourceValidity, TransferBatchError,
    TransferKey, ValidityRepresentations, GUEST_REPRESENTATION,
};
use reims_vgpu_protocol::{BackingId, RepresentationId, SubmissionId, TransactionId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedResourceState {
    transaction: TransactionId,
    index: usize,
    operation: ResolvedResourceState,
    guest_writes: Box<[(BackingId, RegionVersion)]>,
    host_ingresses: Box<[HostIngressKey]>,
    deferred_host_ingress_transfers: Box<[HostIngressTransfer]>,
    gpu_reservations: Box<[GpuWriteReservation]>,
    gpu_completions: Box<[ResolvedResourceCompletion]>,
    transfers: Box<[TransferKey]>,
    host_landings: Box<[HostLandingKey]>,
    states: Box<[(BackingId, ResourceValidity)]>,
    uses: Box<[RepresentationUse]>,
}

impl PreparedResourceState {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn operation(&self) -> &ResolvedResourceState {
        &self.operation
    }

    pub const fn guest_writes(&self) -> &[(BackingId, RegionVersion)] {
        &self.guest_writes
    }

    pub const fn host_ingresses(&self) -> &[HostIngressKey] {
        &self.host_ingresses
    }

    pub const fn deferred_host_ingress_transfers(&self) -> &[HostIngressTransfer] {
        &self.deferred_host_ingress_transfers
    }

    pub const fn gpu_reservations(&self) -> &[GpuWriteReservation] {
        &self.gpu_reservations
    }

    pub const fn resource_completions(&self) -> &[ResolvedResourceCompletion] {
        &self.gpu_completions
    }

    pub const fn transfers(&self) -> &[TransferKey] {
        &self.transfers
    }

    pub const fn host_landings(&self) -> &[HostLandingKey] {
        &self.host_landings
    }

    pub const fn states(&self) -> &[(BackingId, ResourceValidity)] {
        &self.states
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
}

/// Complete lifecycle-authored native preparation for one admitted EXEC.
///
/// Construction proves that every admitted resource-state position appears
/// exactly once and that no sidecar from another transaction or operation can
/// enter native projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedResourceStateBatch {
    transaction: TransactionId,
    states: Box<[PreparedResourceState]>,
}

#[derive(Debug)]
#[must_use = "partial resource-state preparation must be finished or cancelled"]
pub struct ResourceStatePreparationOwner {
    admitted: AdmittedResourceStates,
    positions: BTreeSet<usize>,
    states: Vec<PreparedResourceState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceStatePreparationOwnerError {
    DuplicatePosition(usize),
    Preparation(ResourceStatePreparationError),
}

#[derive(Debug)]
pub struct ResourceStatePreparationFinishFailure {
    pub reason: ResourceStateBatchError,
    pub owner: ResourceStatePreparationOwner,
}

#[derive(Debug)]
pub struct ResourceStatePreparationCancellationFailure {
    pub reason: ResourceStateCancellationError,
    pub owner: ResourceStatePreparationOwner,
}

#[derive(Debug)]
pub struct CancelledResourceStatePreparation<T> {
    pub admitted: AdmittedResourceStates,
    pub prepared: CancelledResourceStateBatch<T>,
}

impl ResourceStatePreparationOwner {
    pub fn new(admitted: AdmittedResourceStates) -> Self {
        Self {
            admitted,
            positions: BTreeSet::new(),
            states: Vec::new(),
        }
    }

    pub const fn transaction(&self) -> TransactionId {
        self.admitted.transaction()
    }

    pub fn host_ingresses(&self) -> Box<[HostIngressKey]> {
        self.states
            .iter()
            .flat_map(|state| state.host_ingresses.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Settle every ingress prepared so far before the next ordered validity
    /// operation can replace its canonical guest-authored version.
    pub fn publish_host_ingresses_after_copy<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
    ) -> Result<Box<[TransferKey]>, crate::HostIngressBatchError> {
        let deferred = self
            .states
            .iter()
            .flat_map(|state| state.deferred_host_ingress_transfers.iter().copied())
            .collect::<Vec<_>>();
        let planned = resources.complete_host_ingress_transfers(&deferred)?;
        for state in &mut self.states {
            let mut transfers = std::mem::take(&mut state.transfers).into_vec();
            for deferred in &state.deferred_host_ingress_transfers {
                transfers.extend(planned.iter().copied().filter(|transfer| {
                    transfer.backing == deferred.ingress.backing
                        && transfer.region == deferred.ingress.region
                        && transfer.version == deferred.ingress.version
                        && transfer.source == crate::HOST_REPRESENTATION
                        && transfer.destination == deferred.destination
                }));
            }
            state.transfers = transfers.into_boxed_slice();
            state.host_ingresses = Box::new([]);
            state.deferred_host_ingress_transfers = Box::new([]);
        }
        Ok(planned)
    }

    pub fn prepare<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
        index: usize,
        submission: SubmissionId,
        representations: impl FnMut(usize, BackingId) -> ValidityRepresentations,
    ) -> Result<(), ResourceStatePreparationOwnerError> {
        if self.positions.contains(&index) {
            return Err(ResourceStatePreparationOwnerError::DuplicatePosition(index));
        }
        let prepared = prepare_resource_state(
            resources,
            &self.admitted,
            index,
            submission,
            representations,
        )
        .map_err(ResourceStatePreparationOwnerError::Preparation)?;
        self.positions.insert(index);
        self.states.push(prepared);
        Ok(())
    }

    pub fn finish(
        self,
    ) -> Result<PreparedResourceStateBatch, Box<ResourceStatePreparationFinishFailure>> {
        if let Some(index) = self
            .admitted
            .operations()
            .iter()
            .map(|(index, _)| *index)
            .find(|index| !self.positions.contains(index))
        {
            return Err(Box::new(ResourceStatePreparationFinishFailure {
                reason: ResourceStateBatchError::PreparationMissing(index),
                owner: self,
            }));
        }
        match assemble_prepared_resource_states(&self.admitted, self.states) {
            Ok(prepared) => Ok(prepared),
            Err(_) => unreachable!("the owner prepared every exact admitted position once"),
        }
    }

    pub fn cancel<T>(
        self,
        resources: &mut ResourceLifecycleOwner<T>,
    ) -> Result<
        CancelledResourceStatePreparation<T>,
        Box<ResourceStatePreparationCancellationFailure>,
    > {
        if let Err(reason) = validate_cancel_resource_state_rows(
            resources,
            self.admitted.transaction(),
            &self.states,
        ) {
            return Err(Box::new(ResourceStatePreparationCancellationFailure {
                reason,
                owner: self,
            }));
        }
        let prepared =
            cancel_resource_state_rows(resources, self.admitted.transaction(), self.states);
        Ok(CancelledResourceStatePreparation {
            admitted: self.admitted,
            prepared,
        })
    }
}

impl PreparedResourceStateBatch {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn states(&self) -> &[PreparedResourceState] {
        &self.states
    }

    pub fn into_states(self) -> Box<[PreparedResourceState]> {
        self.states
    }

    pub fn backings(&self) -> Box<[BackingId]> {
        self.states
            .iter()
            .flat_map(|state| state.uses.iter().map(|use_| use_.backing))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Exact ordered facts the recording's one timeline point completes.
    /// Shared transfer demands produce one physical command and therefore one
    /// completion at their first operation position.
    pub fn resource_completions(&self) -> Box<[ResolvedResourceCompletion]> {
        let mut transfers = BTreeSet::new();
        let mut completions = Vec::new();
        for state in &self.states {
            completions.extend(
                state
                    .transfers
                    .iter()
                    .copied()
                    .filter(|transfer| transfers.insert(*transfer))
                    .map(ResolvedResourceCompletion::Transfer),
            );
            completions.extend(state.gpu_completions.iter().copied());
        }
        completions.into_boxed_slice()
    }

    pub fn host_landings(&self) -> Box<[HostLandingKey]> {
        self.states
            .iter()
            .flat_map(|state| state.host_landings.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn host_ingresses(&self) -> Box<[HostIngressKey]> {
        self.states
            .iter()
            .flat_map(|state| state.host_ingresses.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn deferred_host_ingress_transfers(&self) -> Box<[HostIngressTransfer]> {
        self.states
            .iter()
            .flat_map(|state| state.deferred_host_ingress_transfers.iter().copied())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Publish host-staging coverage only after the caller has copied every
    /// returned key's exact guest-authored bytes into its staging endpoint.
    /// The settled obligations are removed so later preparation cancellation
    /// cannot withdraw an ingress that has already physically completed.
    pub fn publish_host_ingresses_after_copy<T>(
        &mut self,
        resources: &mut ResourceLifecycleOwner<T>,
    ) -> Result<Box<[TransferKey]>, crate::HostIngressBatchError> {
        let deferred = self.deferred_host_ingress_transfers();
        let planned = resources.complete_host_ingress_transfers(&deferred)?;
        for state in &mut self.states {
            let mut transfers = std::mem::take(&mut state.transfers).into_vec();
            for deferred in &state.deferred_host_ingress_transfers {
                transfers.extend(planned.iter().copied().filter(|transfer| {
                    transfer.backing == deferred.ingress.backing
                        && transfer.region == deferred.ingress.region
                        && transfer.version == deferred.ingress.version
                        && transfer.source == crate::HOST_REPRESENTATION
                        && transfer.destination == deferred.destination
                }));
            }
            state.transfers = transfers.into_boxed_slice();
            state.host_ingresses = Box::new([]);
            state.deferred_host_ingress_transfers = Box::new([]);
        }
        Ok(planned)
    }

    pub fn into_outcomes(self) -> Box<[ResourceStateOutcome]> {
        self.states
            .into_vec()
            .into_iter()
            .map(|state| ResourceStateOutcome {
                operation: state.operation,
                guest_writes: state.guest_writes,
                states: state.states,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceStateBatchError {
    TransactionMismatch { index: usize },
    DuplicatePosition(usize),
    OperationAbsent(usize),
    OperationMismatch(usize),
    PreparationMissing(usize),
}

#[derive(Debug)]
pub struct ResourceStateBatchFailure {
    pub reason: ResourceStateBatchError,
    pub states: Box<[PreparedResourceState]>,
}

/// Join operation-local preparations into the exact admitted resource-state
/// projection for one EXEC. Validation is read-only and returns every token on
/// failure, so callers retain complete cancellation ownership.
pub fn assemble_prepared_resource_states(
    admitted: &AdmittedResourceStates,
    states: impl Into<Box<[PreparedResourceState]>>,
) -> Result<PreparedResourceStateBatch, Box<ResourceStateBatchFailure>> {
    let states = states.into();
    let expected = admitted
        .operations()
        .iter()
        .map(|(index, operation)| {
            let operation = match operation {
                AdmittedResourceStateOperation::Semantic(operation)
                | AdmittedResourceStateOperation::NativeTransferMayBeRequired(operation) => {
                    operation
                }
            };
            (*index, operation)
        })
        .collect::<BTreeMap<_, _>>();
    let mut present = BTreeSet::new();
    for state in &states {
        let reason = if state.transaction != admitted.transaction() {
            Some(ResourceStateBatchError::TransactionMismatch { index: state.index })
        } else if !present.insert(state.index) {
            Some(ResourceStateBatchError::DuplicatePosition(state.index))
        } else {
            match expected.get(&state.index) {
                None => Some(ResourceStateBatchError::OperationAbsent(state.index)),
                Some(operation) if **operation != state.operation => {
                    Some(ResourceStateBatchError::OperationMismatch(state.index))
                }
                Some(_) => None,
            }
        };
        if let Some(reason) = reason {
            return Err(Box::new(ResourceStateBatchFailure { reason, states }));
        }
    }
    if let Some(index) = expected
        .keys()
        .copied()
        .find(|index| !present.contains(index))
    {
        return Err(Box::new(ResourceStateBatchFailure {
            reason: ResourceStateBatchError::PreparationMissing(index),
            states,
        }));
    }
    Ok(PreparedResourceStateBatch {
        transaction: admitted.transaction(),
        states,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceStatePreparationError {
    OperationAbsent(usize),
    Uses(ResourceUseBatchError),
    Lifecycle(ResourceLifecycleError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceStateCancellationError {
    Writes(crate::GpuWriteBatchError),
    Transfers(TransferBatchError),
    HostIngresses(crate::HostIngressBatchError),
    HostLandings(crate::HostLandingBatchError),
    Uses(ResourceUseBatchError),
}

#[derive(Debug)]
pub struct ResourceStateCancellationFailure {
    pub reason: ResourceStateCancellationError,
    pub prepared: PreparedResourceState,
}

#[derive(Debug)]
pub struct CancelledResourceState<T> {
    pub operation: ResolvedResourceState,
    pub guest_writes: Box<[(BackingId, RegionVersion)]>,
    pub states: Box<[(BackingId, ResourceValidity)]>,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

#[derive(Debug)]
pub struct ResourceStateOutcome {
    pub operation: ResolvedResourceState,
    pub guest_writes: Box<[(BackingId, RegionVersion)]>,
    pub states: Box<[(BackingId, ResourceValidity)]>,
}

#[derive(Debug)]
pub struct CancelledResourceStateBatch<T> {
    pub outcomes: Box<[ResourceStateOutcome]>,
    pub resources: Vec<(BackingId, ManagedBackingProgress<T>)>,
}

#[derive(Debug)]
pub struct ResourceStateBatchCancellationFailure {
    pub reason: ResourceStateCancellationError,
    pub prepared: PreparedResourceStateBatch,
}

/// Apply one exactly admitted validity operation and retain every lifecycle
/// result required by native recording, cancellation, and GPU completion.
pub fn prepare_resource_state<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    admitted: &AdmittedResourceStates,
    index: usize,
    submission: SubmissionId,
    mut representations: impl FnMut(usize, BackingId) -> ValidityRepresentations,
) -> Result<PreparedResourceState, ResourceStatePreparationError> {
    let operation = admitted
        .operations()
        .iter()
        .find(|(position, _)| *position == index)
        .map(|(_, operation)| match operation {
            AdmittedResourceStateOperation::Semantic(operation)
            | AdmittedResourceStateOperation::NativeTransferMayBeRequired(operation) => operation,
        })
        .ok_or(ResourceStatePreparationError::OperationAbsent(index))?;
    let resolved = operation
        .targets
        .iter()
        .map(|target| (target, representations(index, target.backing)))
        .collect::<Vec<_>>();
    let transition = ResolvedValidityTransition {
        ops: operation.ops,
        write: (operation.ops.set_host_valid != 0).then_some(crate::GpuWriteId::operation(
            admitted.transaction(),
            submission,
            index,
        )),
        targets: resolved
            .iter()
            .map(|(target, representations)| crate::ResolvedValidityTarget {
                backing: target.backing,
                regions: target.regions.clone(),
                host_representation: representations.host_write,
                host_ingress_destination: representations.host_ingress_destination,
                guest_upload_destination: representations.guest_upload_destination,
                guest_visibility_source: representations.guest_visibility_source,
                guest_visibility_destination: representations.guest_visibility_destination,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    };
    resources
        .validate_validity_transition(&transition)
        .map_err(|reason| {
            ResourceStatePreparationError::Lifecycle(ResourceLifecycleError::Validity(reason))
        })?;
    let mut uses = Vec::new();
    for (target, representations) in resolved {
        let mut native = Vec::<RepresentationId>::new();
        if let Some(representation) = representations.host_write {
            native.push(representation);
        }
        if let Some(representation) = representations.host_ingress_destination {
            native.push(representation);
        }
        // A guest upload plans a content transfer into this representation, and
        // the transfer's completion is validated against it when the submission
        // is accepted. Without the use the representation carries no timeline
        // obligation, so retirement is free to take it while the submission is
        // still in flight -- and the acceptance then refuses a completion whose
        // destination no longer exists, which retires nothing and stops the
        // channel behind it.
        if let Some(representation) = representations.guest_upload_destination {
            native.push(representation);
        }
        if operation.ops.clear_guest_valid != 0 && operation.ops.clear_host_valid == 0 {
            let snapshot = resources
                .snapshot_content(target.backing, &target.regions)
                .map_err(|reason| {
                    ResourceStatePreparationError::Lifecycle(ResourceLifecycleError::Native(reason))
                })?;
            if !resources
                .representation_matches(target.backing, GUEST_REPRESENTATION, &snapshot)
                .map_err(|reason| {
                    ResourceStatePreparationError::Lifecycle(ResourceLifecycleError::Native(reason))
                })?
            {
                native.push(
                    representations
                        .guest_visibility_source
                        .expect("validity source was prevalidated"),
                );
                native.push(representations.guest_visibility_destination);
            }
        }
        native.sort();
        native.dedup();
        if !native.is_empty() {
            uses.push(RepresentationUse {
                backing: target.backing,
                representations: native.into_boxed_slice(),
            });
        }
    }
    let uses = uses.into_boxed_slice();
    resources
        .accept_uses(admitted.transaction(), &uses)
        .map_err(ResourceStatePreparationError::Uses)?;
    let effect = resources
        .apply(ResolvedResourceLifecycle::ApplyValidity(transition))
        .unwrap_or_else(|_| unreachable!("validity transition was prevalidated"));
    let ResourceLifecycleEffect::ValidityApplied {
        guest_writes,
        host_ingresses,
        deferred_host_ingress_transfers,
        gpu_reservations,
        gpu_completions,
        transfers,
        host_landings,
        states,
    } = effect
    else {
        unreachable!("validity lifecycle command has one effect variant")
    };
    Ok(PreparedResourceState {
        transaction: admitted.transaction(),
        index,
        operation: operation.clone(),
        guest_writes,
        host_ingresses,
        deferred_host_ingress_transfers,
        gpu_reservations,
        gpu_completions,
        transfers,
        host_landings,
        states,
        uses,
    })
}

/// Return every native reservation owned by a prepared validity operation.
/// The validity statement and any guest-authored versions have already taken
/// effect at its ordered semantic point, so they are returned as the durable
/// outcome rather than rolled back. GPU-write and transfer reservations have
/// not completed and are cancelled without publishing representation content.
pub fn cancel_prepared_resource_state<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedResourceState,
) -> Result<CancelledResourceState<T>, Box<ResourceStateCancellationFailure>> {
    if let Err(reason) = resources.validate_cancel_gpu_writes(&prepared.gpu_reservations) {
        return Err(Box::new(ResourceStateCancellationFailure {
            reason: ResourceStateCancellationError::Writes(reason),
            prepared,
        }));
    }
    if let Err(reason) = resources.validate_cancel_transfers(&prepared.transfers) {
        return Err(Box::new(ResourceStateCancellationFailure {
            reason: ResourceStateCancellationError::Transfers(reason),
            prepared,
        }));
    }
    if let Err(reason) = resources.validate_cancel_host_ingresses(&prepared.host_ingresses) {
        return Err(Box::new(ResourceStateCancellationFailure {
            reason: ResourceStateCancellationError::HostIngresses(reason),
            prepared,
        }));
    }
    if let Err(reason) = resources.validate_cancel_host_landings(&prepared.host_landings) {
        return Err(Box::new(ResourceStateCancellationFailure {
            reason: ResourceStateCancellationError::HostLandings(reason),
            prepared,
        }));
    }
    if let Err(reason) =
        resources.validate_cancel_representation_uses(prepared.transaction, &prepared.uses)
    {
        return Err(Box::new(ResourceStateCancellationFailure {
            reason: ResourceStateCancellationError::Uses(reason),
            prepared,
        }));
    }
    resources
        .cancel_gpu_writes(&prepared.gpu_reservations)
        .expect("resource-state write cancellation was prevalidated");
    resources
        .cancel_transfers(&prepared.transfers)
        .expect("resource-state transfer cancellation was prevalidated");
    resources
        .cancel_host_ingresses(&prepared.host_ingresses)
        .expect("resource-state host ingress cancellation was prevalidated");
    resources
        .cancel_host_landings(&prepared.host_landings)
        .expect("resource-state host landing cancellation was prevalidated");
    let progress = resources
        .cancel_representation_uses(prepared.transaction, &prepared.uses)
        .expect("resource-state use cancellation was prevalidated");
    Ok(CancelledResourceState {
        operation: prepared.operation,
        guest_writes: prepared.guest_writes,
        states: prepared.states,
        resources: progress,
    })
}

/// Atomically withdraw every incomplete native obligation owned by a complete
/// prepared EXEC batch. Repeated backing writes, representation uses, and
/// shared transfer demands are validated with their full multiplicity before
/// the first owner changes.
pub fn cancel_prepared_resource_state_batch<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    prepared: PreparedResourceStateBatch,
) -> Result<CancelledResourceStateBatch<T>, Box<ResourceStateBatchCancellationFailure>> {
    if let Err(reason) =
        validate_cancel_resource_state_rows(resources, prepared.transaction, &prepared.states)
    {
        return Err(Box::new(ResourceStateBatchCancellationFailure {
            reason,
            prepared,
        }));
    }
    Ok(cancel_resource_state_rows(
        resources,
        prepared.transaction,
        prepared.states.into_vec(),
    ))
}

fn validate_cancel_resource_state_rows<T>(
    resources: &ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    states: &[PreparedResourceState],
) -> Result<(), ResourceStateCancellationError> {
    let gpu_reservations = states
        .iter()
        .flat_map(|state| state.gpu_reservations.iter().cloned())
        .collect::<Vec<_>>();
    let transfers = states
        .iter()
        .flat_map(|state| state.transfers.iter().copied())
        .collect::<Vec<_>>();
    let host_landings = states
        .iter()
        .flat_map(|state| state.host_landings.iter().copied())
        .collect::<Vec<_>>();
    let host_ingresses = states
        .iter()
        .flat_map(|state| state.host_ingresses.iter().copied())
        .collect::<Vec<_>>();
    let uses = states
        .iter()
        .flat_map(|state| state.uses.iter().cloned())
        .collect::<Vec<_>>();
    resources
        .validate_cancel_gpu_writes(&gpu_reservations)
        .map_err(ResourceStateCancellationError::Writes)?;
    resources
        .validate_cancel_transfers(&transfers)
        .map_err(ResourceStateCancellationError::Transfers)?;
    resources
        .validate_cancel_host_ingresses(&host_ingresses)
        .map_err(ResourceStateCancellationError::HostIngresses)?;
    resources
        .validate_cancel_host_landings(&host_landings)
        .map_err(ResourceStateCancellationError::HostLandings)?;
    resources
        .validate_cancel_representation_uses(transaction, &uses)
        .map_err(ResourceStateCancellationError::Uses)
}

fn cancel_resource_state_rows<T>(
    resources: &mut ResourceLifecycleOwner<T>,
    transaction: TransactionId,
    states: Vec<PreparedResourceState>,
) -> CancelledResourceStateBatch<T> {
    let gpu_reservations = states
        .iter()
        .flat_map(|state| state.gpu_reservations.iter().cloned())
        .collect::<Vec<_>>();
    let transfers = states
        .iter()
        .flat_map(|state| state.transfers.iter().copied())
        .collect::<Vec<_>>();
    let host_landings = states
        .iter()
        .flat_map(|state| state.host_landings.iter().copied())
        .collect::<Vec<_>>();
    let host_ingresses = states
        .iter()
        .flat_map(|state| state.host_ingresses.iter().copied())
        .collect::<Vec<_>>();
    let uses = states
        .iter()
        .flat_map(|state| state.uses.iter().cloned())
        .collect::<Vec<_>>();
    resources
        .cancel_gpu_writes(&gpu_reservations)
        .expect("the complete batch write cancellation was prevalidated");
    resources
        .cancel_transfers(&transfers)
        .expect("the complete batch transfer cancellation was prevalidated");
    resources
        .cancel_host_ingresses(&host_ingresses)
        .expect("the complete batch host ingress cancellation was prevalidated");
    resources
        .cancel_host_landings(&host_landings)
        .expect("the complete batch host landing cancellation was prevalidated");
    let progress = resources
        .cancel_representation_uses(transaction, &uses)
        .expect("the complete batch use cancellation was prevalidated");
    let outcomes = states
        .into_iter()
        .map(|state| ResourceStateOutcome {
            operation: state.operation,
            guest_writes: state.guest_writes,
            states: state.states,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    CancelledResourceStateBatch {
        outcomes,
        resources: progress,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackingView;
    use crate::{
        BackingRegion, ExecTransaction, RepresentationRoute, ResolvedExecSegment,
        ResolvedExecStream, ResolvedOperation, ResolvedResourceStateTarget, StorageBacking,
        GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ContentVersion, ResourceValidityOps, SegmentBoundary, SegmentKind, SubmissionIdentity,
        TaskId, VulkanDeviceEpochId,
    };

    fn admitted(operation: ResolvedResourceState) -> AdmittedResourceStates {
        admitted_many(vec![operation])
    }

    fn admitted_many(operations: Vec<ResolvedResourceState>) -> AdmittedResourceStates {
        let exec: ExecTransaction<ResolvedOperation<(), (), (), (), ()>> = ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(4),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
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
                    operations: operations
                        .into_iter()
                        .map(ResolvedOperation::ResourceState)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                }]),
            }]),
            accesses: Box::new([]),
        };
        AdmittedResourceStates::from_exec(TransactionId::new(7), &exec)
    }

    fn synthetic_prepared(
        transaction: TransactionId,
        index: usize,
        operation: ResolvedResourceState,
    ) -> PreparedResourceState {
        PreparedResourceState {
            transaction,
            index,
            operation,
            guest_writes: Box::new([]),
            host_ingresses: Box::new([]),
            deferred_host_ingress_transfers: Box::new([]),
            gpu_reservations: Box::new([]),
            gpu_completions: Box::new([]),
            transfers: Box::new([]),
            host_landings: Box::new([]),
            states: Box::new([]),
            uses: Box::new([]),
        }
    }

    #[test]
    fn batch_owns_exactly_every_admitted_resource_state_position() {
        let first = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([]),
            ops: ResourceValidityOps::default(),
        };
        let second = ResolvedResourceState {
            ops: ResourceValidityOps {
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
            ..first.clone()
        };
        let admitted = admitted_many(vec![first.clone(), second.clone()]);
        let batch = assemble_prepared_resource_states(
            &admitted,
            vec![
                synthetic_prepared(admitted.transaction(), 0, first),
                synthetic_prepared(admitted.transaction(), 1, second),
            ],
        )
        .unwrap();
        assert_eq!(batch.transaction(), admitted.transaction());
        assert_eq!(
            batch
                .states()
                .iter()
                .map(PreparedResourceState::index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn partial_resource_state_owner_returns_admission_and_ordered_outcomes() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let admitted = admitted_many(vec![operation.clone(), operation.clone()]);
        let mut preparation = ResourceStatePreparationOwner::new(admitted.clone());
        preparation
            .prepare(&mut resources, 0, SubmissionId::new(4), |_, _| {
                ValidityRepresentations::default()
            })
            .unwrap();

        let cancelled = preparation.cancel(&mut resources).unwrap();
        assert_eq!(cancelled.admitted, admitted);
        assert_eq!(cancelled.prepared.outcomes.len(), 1);
        assert_eq!(cancelled.prepared.outcomes[0].operation, operation);
        assert!(cancelled.prepared.resources.is_empty());
    }

    #[test]
    fn malformed_batch_returns_every_preparation_without_partial_ownership_loss() {
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([]),
            ops: ResourceValidityOps::default(),
        };
        let admitted = admitted_many(vec![operation.clone(), operation.clone()]);
        let failure = assemble_prepared_resource_states(
            &admitted,
            vec![
                synthetic_prepared(admitted.transaction(), 0, operation.clone()),
                synthetic_prepared(TransactionId::new(99), 1, operation),
            ],
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ResourceStateBatchError::TransactionMismatch { index: 1 }
        );
        assert_eq!(failure.states.len(), 2);

        let missing = assemble_prepared_resource_states(
            &admitted,
            vec![synthetic_prepared(
                admitted.transaction(),
                0,
                match &admitted.operations()[0].1 {
                    AdmittedResourceStateOperation::Semantic(operation)
                    | AdmittedResourceStateOperation::NativeTransferMayBeRequired(operation) => {
                        operation.clone()
                    }
                },
            )],
        )
        .unwrap_err();
        assert_eq!(
            missing.reason,
            ResourceStateBatchError::PreparationMissing(1)
        );
        assert_eq!(missing.states.len(), 1);
    }

    #[test]
    fn transfer_keys_come_from_the_live_content_authority() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
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
            .plan_gpu_write(
                backing,
                SubmissionId::new(1),
                source,
                [BackingRegion::Whole],
            )
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_guest_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let prepared = prepare_resource_state(
            &mut resources,
            &admitted(operation.clone()),
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: Some(source),
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        assert_eq!(prepared.operation(), &operation);
        assert_eq!(prepared.transaction(), TransactionId::new(7));
        assert_eq!(prepared.index(), 0);
        assert_eq!(prepared.transfers().len(), 1);
        assert_eq!(prepared.transfers()[0].backing, backing);
        assert_eq!(prepared.transfers()[0].source, source);
        assert_eq!(prepared.transfers()[0].destination, GUEST_REPRESENTATION);
        assert_eq!(prepared.transfers()[0].version, ContentVersion::new(2));
        assert!(prepared.resource_completions().is_empty());
    }

    #[test]
    fn staged_guest_visibility_owns_distinct_gpu_and_cpu_completion_keys() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        assert_eq!(
            resources
                .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
                .unwrap(),
            crate::HOST_REPRESENTATION
        );
        resources
            .plan_gpu_write(
                backing,
                SubmissionId::new(1),
                source,
                [BackingRegion::Whole],
            )
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let snapshot = resources
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_guest_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let prepared = prepare_resource_state(
            &mut resources,
            &admitted(operation),
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: Some(source),
                guest_visibility_destination: crate::HOST_REPRESENTATION,
            },
        )
        .unwrap();
        let transfer = prepared.transfers()[0];
        let landing = prepared.host_landings()[0];
        assert_eq!(transfer.destination, crate::HOST_REPRESENTATION);
        assert_eq!(landing.backing, backing);

        resources
            .complete_resources(&[ResolvedResourceCompletion::Transfer(transfer)])
            .unwrap();
        assert!(!resources
            .representation_matches(backing, GUEST_REPRESENTATION, &snapshot)
            .unwrap());
        resources.complete_host_landings(&[landing]).unwrap();
        assert!(resources
            .representation_matches(backing, GUEST_REPRESENTATION, &snapshot)
            .unwrap());
    }

    #[test]
    fn staged_guest_write_owns_exact_host_ingress_until_cpu_copy_succeeds() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let working = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        assert_eq!(
            resources
                .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
                .unwrap(),
            crate::HOST_REPRESENTATION
        );
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_host_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let prepared = prepare_resource_state(
            &mut resources,
            &admitted(operation),
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: Some(crate::HOST_REPRESENTATION),
                guest_upload_destination: Some(working),
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let write = prepared.guest_writes()[0].1;
        let ingress = prepared.host_ingresses()[0];
        assert_eq!(ingress.backing, backing);
        assert_eq!(ingress.region, write.region);
        assert_eq!(ingress.version, write.version);
        assert!(!resources
            .representation_matches(backing, crate::HOST_REPRESENTATION, &[write])
            .unwrap());

        let states = admitted(prepared.operation().clone());
        let mut prepared = assemble_prepared_resource_states(&states, vec![prepared]).unwrap();
        let planned = prepared
            .publish_host_ingresses_after_copy(&mut resources)
            .unwrap();
        assert!(matches!(
            planned.as_ref(),
            [transfer]
                if transfer.backing == backing
                    && transfer.region == ingress.region
                    && transfer.version == ingress.version
                    && transfer.source == crate::HOST_REPRESENTATION
                    && transfer.destination == working
        ));
        assert!(resources
            .representation_matches(backing, crate::HOST_REPRESENTATION, &[write])
            .unwrap());
        cancel_prepared_resource_state_batch(&mut resources, prepared).unwrap();
    }

    #[test]
    fn a_guest_upload_destination_is_a_use_of_the_transaction_that_plans_it() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let working = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_host_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let prepared = prepare_resource_state(
            &mut resources,
            &admitted(operation),
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: Some(crate::HOST_REPRESENTATION),
                guest_upload_destination: Some(working),
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();

        // The upload plans a transfer into `working`, and the acceptance of the
        // submission carrying it validates that completion against the
        // representation. Declaring the use is what keeps the representation
        // from retiring underneath the submission.
        let declares_destination = prepared
            .uses()
            .iter()
            .filter(|use_| use_.backing == backing)
            .any(|use_| use_.representations.contains(&working));
        assert!(
            declares_destination,
            "the guest-upload destination is not a declared use"
        );

        let transaction = prepared.transaction();
        cancel_resource_state_rows(&mut resources, transaction, vec![prepared]);
    }

    #[test]
    fn repeated_guest_writes_publish_each_ordered_ingress_before_the_next_version() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let working = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::HostStagingTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        resources
            .create_representation(backing, RepresentationRoute::HostStagingEndpoint, ())
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_host_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let admitted = admitted_many(vec![operation.clone(), operation]);
        let mut owner = ResourceStatePreparationOwner::new(admitted);
        let representations = |_, _| ValidityRepresentations {
            host_write: None,
            host_ingress_destination: Some(crate::HOST_REPRESENTATION),
            guest_upload_destination: Some(working),
            guest_visibility_source: None,
            guest_visibility_destination: GUEST_REPRESENTATION,
        };

        owner
            .prepare(&mut resources, 0, SubmissionId::new(4), representations)
            .unwrap();
        let first = owner.host_ingresses()[0];
        owner
            .publish_host_ingresses_after_copy(&mut resources)
            .unwrap();
        owner
            .prepare(&mut resources, 1, SubmissionId::new(4), representations)
            .unwrap();
        let second = owner.host_ingresses()[0];
        assert!(second.version > first.version);
        owner
            .publish_host_ingresses_after_copy(&mut resources)
            .unwrap();

        let prepared = owner.finish().unwrap();
        assert_eq!(prepared.resource_completions().len(), 2);
        cancel_prepared_resource_state_batch(&mut resources, prepared).unwrap();
    }

    #[test]
    fn imported_guest_write_plans_its_exact_guest_to_working_upload_immediately() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        resources
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        let working = resources
            .create_execution_representation(
                backing,
                RepresentationRoute::ImportedGuestTransfer {
                    working: crate::WorkingMemoryClass::DeviceLocal,
                },
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_host_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let prepared = prepare_resource_state(
            &mut resources,
            &admitted(operation),
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: None,
                guest_upload_destination: Some(working),
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        assert!(prepared.host_ingresses().is_empty());
        assert!(prepared.deferred_host_ingress_transfers().is_empty());
        assert!(matches!(
            prepared.transfers(),
            [transfer]
                if transfer.backing == backing
                    && transfer.source == GUEST_REPRESENTATION
                    && transfer.destination == working
                    && transfer.version == prepared.guest_writes()[0].1.version
        ));
    }

    #[test]
    fn cancellation_returns_the_exact_transfer_and_accepted_use_without_publishing_content() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = resources
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        resources
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        resources
            .plan_gpu_write(
                backing,
                SubmissionId::new(1),
                source,
                [BackingRegion::Whole],
            )
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_guest_valid: 1,
                set_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let admissions = admitted(operation.clone());
        let prepared = prepare_resource_state(
            &mut resources,
            &admissions,
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: Some(source),
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let key = prepared.transfers()[0];

        let cancelled = cancel_prepared_resource_state(&mut resources, prepared).unwrap();
        assert_eq!(cancelled.operation, operation);
        assert_eq!(
            cancelled.resources,
            [(backing, ManagedBackingProgress::Live)]
        );
        assert_eq!(
            resources.complete_transfer(key),
            Err(crate::ManagedBackingError::Content(
                crate::ContentAuthorityError::TransferNotPlanned
            ))
        );

        let retry = prepare_resource_state(
            &mut resources,
            &admissions,
            0,
            SubmissionId::new(4),
            |_, _| ValidityRepresentations {
                host_write: None,
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: Some(source),
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        assert_eq!(retry.transfers(), [key]);
        cancel_prepared_resource_state(&mut resources, retry).unwrap();
    }

    #[test]
    fn absent_position_changes_no_resource_state() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([]),
            ops: ResourceValidityOps::default(),
        };
        assert_eq!(
            prepare_resource_state(
                &mut resources,
                &admitted(operation),
                2,
                SubmissionId::new(4),
                |_, _| ValidityRepresentations::default(),
            ),
            Err(ResourceStatePreparationError::OperationAbsent(2))
        );
    }

    #[test]
    fn two_host_writes_to_one_backing_use_distinct_operation_identities() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let host = resources
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                set_host_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let admissions = admitted_many(vec![operation.clone(), operation.clone()]);
        let first = prepare_resource_state(
            &mut resources,
            &admissions,
            0,
            SubmissionId::new(31),
            |_, _| ValidityRepresentations {
                host_write: Some(host),
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let second = prepare_resource_state(
            &mut resources,
            &admissions,
            1,
            SubmissionId::new(31),
            |_, _| ValidityRepresentations {
                host_write: Some(host),
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();

        assert_eq!(
            first.gpu_reservations()[0].write,
            crate::GpuWriteId::operation(admissions.transaction(), SubmissionId::new(31), 0,)
        );
        assert_eq!(
            second.gpu_reservations()[0].write,
            crate::GpuWriteId::operation(admissions.transaction(), SubmissionId::new(31), 1,)
        );
        let mut batch =
            assemble_prepared_resource_states(&admissions, vec![first, second]).unwrap();
        assert_eq!(batch.backings().as_ref(), [backing]);
        assert_eq!(batch.resource_completions().len(), 2);
        let exact_write = batch.states[1].gpu_reservations[0].write;
        batch.states[1].gpu_reservations[0].write =
            crate::GpuWriteId::operation(admissions.transaction(), SubmissionId::new(31), 99);
        let failure = cancel_prepared_resource_state_batch(&mut resources, batch).unwrap_err();
        assert!(matches!(
            failure.reason,
            ResourceStateCancellationError::Writes(crate::GpuWriteBatchError::Backing { .. })
        ));
        let mut batch = failure.prepared;
        batch.states[1].gpu_reservations[0].write = exact_write;
        let cancelled = cancel_prepared_resource_state_batch(&mut resources, batch).unwrap();
        assert_eq!(cancelled.outcomes.len(), 2);
        assert_eq!(
            cancelled.resources,
            [(backing, ManagedBackingProgress::Live)]
        );

        let retry_first = prepare_resource_state(
            &mut resources,
            &admissions,
            0,
            SubmissionId::new(31),
            |_, _| ValidityRepresentations {
                host_write: Some(host),
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let retry_second = prepare_resource_state(
            &mut resources,
            &admissions,
            1,
            SubmissionId::new(31),
            |_, _| ValidityRepresentations {
                host_write: Some(host),
                host_ingress_destination: None,
                guest_upload_destination: None,
                guest_visibility_source: None,
                guest_visibility_destination: GUEST_REPRESENTATION,
            },
        )
        .unwrap();
        let retry = assemble_prepared_resource_states(&admissions, vec![retry_first, retry_second])
            .unwrap();
        cancel_prepared_resource_state_batch(&mut resources, retry).unwrap();
    }

    #[test]
    fn two_resource_states_share_one_transfer_and_one_cancellation_keeps_it_live() {
        let mut resources = ResourceLifecycleOwner::<()>::new(VulkanDeviceEpochId::new(1));
        let ResourceLifecycleEffect::BackingCreated(backing) = resources
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let source = resources
            .create_representation(backing, RepresentationRoute::HostVisibleWorking, ())
            .unwrap();
        resources
            .create_representation(backing, RepresentationRoute::DirectGuestAlias, ())
            .unwrap();
        resources
            .plan_gpu_write(
                backing,
                SubmissionId::new(1),
                source,
                [BackingRegion::Whole],
            )
            .unwrap();
        resources
            .complete_gpu_write(backing, SubmissionId::new(1), source)
            .unwrap();
        let operation = ResolvedResourceState {
            resource: None,
            mappings: Box::new([]),
            targets: Box::new([ResolvedResourceStateTarget {
                backing,
                regions: Box::new([BackingRegion::Whole]),
            }]),
            ops: ResourceValidityOps {
                clear_guest_valid: 1,
                ..ResourceValidityOps::default()
            },
        };
        let admissions = admitted_many(vec![operation.clone(), operation]);
        let prepare = |resources: &mut ResourceLifecycleOwner<()>, index| {
            prepare_resource_state(
                resources,
                &admissions,
                index,
                SubmissionId::new(8),
                |_, _| ValidityRepresentations {
                    host_write: None,
                    host_ingress_destination: None,
                    guest_upload_destination: None,
                    guest_visibility_source: Some(source),
                    guest_visibility_destination: GUEST_REPRESENTATION,
                },
            )
            .unwrap()
        };
        let first = prepare(&mut resources, 0);
        let second = prepare(&mut resources, 1);
        assert_eq!(first.transfers(), second.transfers());
        let key = first.transfers()[0];

        let mut batch =
            assemble_prepared_resource_states(&admissions, vec![first, second]).unwrap();
        assert_eq!(batch.backings().as_ref(), [backing]);
        assert_eq!(
            batch.resource_completions().as_ref(),
            [ResolvedResourceCompletion::Transfer(key)]
        );
        batch.states[1].transfers[0].version = ContentVersion::new(key.version.get() + 1);
        let failure = cancel_prepared_resource_state_batch(&mut resources, batch).unwrap_err();
        assert!(matches!(
            failure.reason,
            ResourceStateCancellationError::Transfers(TransferBatchError::Transfer { .. })
        ));
        let mut batch = failure.prepared;
        batch.states[1].transfers[0] = key;
        let cancelled = cancel_prepared_resource_state_batch(&mut resources, batch).unwrap();
        assert_eq!(cancelled.outcomes.len(), 2);
        assert_eq!(
            cancelled.resources,
            [(backing, ManagedBackingProgress::Live)]
        );

        let first = prepare(&mut resources, 0);
        let second = prepare(&mut resources, 1);
        cancel_prepared_resource_state(&mut resources, first).unwrap();
        resources.complete_transfer(key).unwrap();
        resources
            .cancel_representation_uses(second.transaction(), second.uses())
            .unwrap();
    }
}
