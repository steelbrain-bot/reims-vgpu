//! Queue-acceptance-owned image layout and queue-family state.
//!
//! One image representation has at most one prepared transition. A later use
//! of that representation is not planned from speculative state: it waits
//! until the earlier native submit is either accepted (committing its final
//! state) or canceled (leaving the prior state unchanged). Disjoint images can
//! still prepare independently.

use ash::vk;
use reims_vgpu_core::QueueTimelinePoint;
use reims_vgpu_protocol::{BackingId, RepresentationId, TransactionId, VulkanDeviceEpochId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReplacementImageKey {
    pub backing: BackingId,
    pub representation: RepresentationId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementImageSharing {
    Concurrent,
    Exclusive { owner: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementImageState {
    pub layout: vk::ImageLayout,
    pub sharing: ReplacementImageSharing,
    /// Last accepted native use. Defined exclusive ownership in another queue
    /// family cannot be released without this ordering point.
    pub last_use: Option<QueueTimelinePoint>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementImageUse {
    pub image: ReplacementImageKey,
    pub required_usage: vk::ImageUsageFlags,
    pub use_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedImageQueueTransfer {
    pub source: u32,
    pub destination: u32,
    pub source_point: QueueTimelinePoint,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReplacementImageReleaseKey {
    pub source_queue_family: u32,
    pub source_queue: reims_vgpu_protocol::QueueOwnerId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedImageRelease {
    pub release: ReplacementImageReleaseKey,
    pub point: QueueTimelinePoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementImageTransition {
    pub image: ReplacementImageKey,
    pub required_usage: vk::ImageUsageFlags,
    pub initial_layout: vk::ImageLayout,
    pub use_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
    pub queue_transfer: Option<PlannedImageQueueTransfer>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedImageState {
    transaction: TransactionId,
    operation_index: Option<usize>,
    queue_family: u32,
    transitions: Box<[ReplacementImageTransition]>,
    accepted_releases: Box<[AcceptedImageRelease]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedImageStateBatch {
    transaction: TransactionId,
    queue_family: u32,
    operations: Box<[PreparedImageState]>,
}

impl PreparedImageStateBatch {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub const fn operations(&self) -> &[PreparedImageState] {
        &self.operations
    }

    pub fn release_points(&self) -> Box<[QueueTimelinePoint]> {
        let mut points = self
            .operations
            .iter()
            .flat_map(|operation| operation.release_points())
            .collect::<Vec<_>>();
        points.sort_unstable();
        points.dedup();
        points.into_boxed_slice()
    }
}

impl PreparedImageState {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub const fn operation_index(&self) -> Option<usize> {
        self.operation_index
    }

    pub const fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub const fn transitions(&self) -> &[ReplacementImageTransition] {
        &self.transitions
    }

    pub fn release_accepted(&self, release: ReplacementImageReleaseKey) -> bool {
        self.accepted_releases
            .iter()
            .any(|accepted| accepted.release == release)
    }

    pub const fn accepted_releases(&self) -> &[AcceptedImageRelease] {
        &self.accepted_releases
    }

    pub fn release_points(&self) -> Box<[QueueTimelinePoint]> {
        self.accepted_releases
            .iter()
            .map(|accepted| accepted.point)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementImageStateError {
    EmptyUseSet,
    DuplicateImage(ReplacementImageKey),
    UnknownImage(ReplacementImageKey),
    ImageBusy {
        image: ReplacementImageKey,
        transaction: TransactionId,
    },
    ImageHasDependentOperation(ReplacementImageKey),
    OperationOrder,
    InvalidUseLayout(ReplacementImageKey),
    InvalidFinalLayout(ReplacementImageKey),
    EmptyRequiredUsage(ReplacementImageKey),
    PreparedMismatch(ReplacementImageKey),
    UnknownRelease(ReplacementImageReleaseKey),
    ReleaseAlreadyAccepted(ReplacementImageReleaseKey),
    ReleaseCannotCancel(ReplacementImageReleaseKey),
    ReleasePending(ReplacementImageReleaseKey),
    MissingSourcePoint(ReplacementImageKey),
    MixedEpoch,
    InvalidReleasePoint(ReplacementImageReleaseKey),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingImageState {
    transaction: TransactionId,
    operation_index: Option<usize>,
    queue_family: u32,
    transition: ReplacementImageTransition,
    release_accepted: bool,
}

#[derive(Clone, Debug)]
pub struct ReplacementImageStateOwner {
    epoch: VulkanDeviceEpochId,
    images: BTreeMap<ReplacementImageKey, ReplacementImageState>,
    pending: BTreeMap<ReplacementImageKey, Vec<PendingImageState>>,
}

impl ReplacementImageStateOwner {
    pub fn new(epoch: VulkanDeviceEpochId) -> Self {
        Self {
            epoch,
            images: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    pub fn register(
        &mut self,
        image: ReplacementImageKey,
        state: ReplacementImageState,
    ) -> Result<(), ReplacementImageStateError> {
        if self.images.contains_key(&image) {
            return Err(ReplacementImageStateError::DuplicateImage(image));
        }
        if state
            .last_use
            .is_some_and(|point| point.epoch != self.epoch)
        {
            return Err(ReplacementImageStateError::MixedEpoch);
        }
        self.images.insert(image, state);
        Ok(())
    }

    /// Register a newly created exclusive image before its first native use.
    /// The constructor has not submitted work, so `UNDEFINED` and no prior
    /// timeline point are the only valid initial state.
    pub fn register_new_image(
        &mut self,
        image: ReplacementImageKey,
        queue_family: u32,
    ) -> Result<(), ReplacementImageStateError> {
        self.register(
            image,
            ReplacementImageState {
                layout: vk::ImageLayout::UNDEFINED,
                sharing: ReplacementImageSharing::Exclusive {
                    owner: queue_family,
                },
                last_use: None,
            },
        )
    }

    pub fn state(&self, image: ReplacementImageKey) -> Option<ReplacementImageState> {
        self.images.get(&image).copied()
    }

    pub fn prepare(
        &mut self,
        transaction: TransactionId,
        queue_family: u32,
        uses: impl Into<Box<[ReplacementImageUse]>>,
    ) -> Result<PreparedImageState, ReplacementImageStateError> {
        let uses = uses.into();
        self.prepare_inner(transaction, None, queue_family, &uses)
    }

    pub fn prepare_operation(
        &mut self,
        transaction: TransactionId,
        operation_index: usize,
        queue_family: u32,
        uses: impl Into<Box<[ReplacementImageUse]>>,
    ) -> Result<PreparedImageState, ReplacementImageStateError> {
        let uses = uses.into();
        self.prepare_inner(transaction, Some(operation_index), queue_family, &uses)
    }

    pub fn prepare_batch(
        &mut self,
        transaction: TransactionId,
        queue_family: u32,
        operations: impl Into<Box<[(usize, Box<[ReplacementImageUse]>)]>>,
    ) -> Result<PreparedImageStateBatch, ReplacementImageStateError> {
        let operations = operations.into();
        if operations.is_empty() {
            return Err(ReplacementImageStateError::EmptyUseSet);
        }
        if operations.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(ReplacementImageStateError::OperationOrder);
        }
        let mut prepared = Vec::with_capacity(operations.len());
        for (index, uses) in operations.iter() {
            match self.prepare_inner(transaction, Some(*index), queue_family, uses) {
                Ok(operation) => prepared.push(operation),
                Err(reason) => {
                    for operation in prepared.into_iter().rev() {
                        self.cancel(operation)
                            .expect("batch preparation rollback owns every pending suffix");
                    }
                    return Err(reason);
                }
            }
        }
        Ok(PreparedImageStateBatch {
            transaction,
            queue_family,
            operations: prepared.into_boxed_slice(),
        })
    }

    /// Prepare operation-indexed image uses followed by one native auxiliary
    /// tail owned by the same transaction. The tail has no semantic operation
    /// index and is used for lifecycle-authored transfers that run after the
    /// resource-state prefix and before the encoder suffix.
    pub fn prepare_batch_with_auxiliary_tail(
        &mut self,
        transaction: TransactionId,
        queue_family: u32,
        operations: impl Into<Box<[(usize, Box<[ReplacementImageUse]>)]>>,
        auxiliary_tail: impl Into<Box<[ReplacementImageUse]>>,
    ) -> Result<PreparedImageStateBatch, ReplacementImageStateError> {
        let operations = operations.into();
        let auxiliary_tail = auxiliary_tail.into();
        if auxiliary_tail.is_empty() {
            return self.prepare_batch(transaction, queue_family, operations);
        }
        if operations.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(ReplacementImageStateError::OperationOrder);
        }
        let mut prepared = Vec::with_capacity(operations.len() + 1);
        for (index, uses) in operations.iter() {
            match self.prepare_inner(transaction, Some(*index), queue_family, uses) {
                Ok(operation) => prepared.push(operation),
                Err(reason) => {
                    for operation in prepared.into_iter().rev() {
                        self.cancel(operation)
                            .expect("batch rollback owns every prepared image state");
                    }
                    return Err(reason);
                }
            }
        }
        match self.prepare_inner(transaction, None, queue_family, &auxiliary_tail) {
            Ok(tail) => prepared.push(tail),
            Err(reason) => {
                for operation in prepared.into_iter().rev() {
                    self.cancel(operation)
                        .expect("batch rollback owns every prepared image state");
                }
                return Err(reason);
            }
        }
        Ok(PreparedImageStateBatch {
            transaction,
            queue_family,
            operations: prepared.into_boxed_slice(),
        })
    }

    fn prepare_inner(
        &mut self,
        transaction: TransactionId,
        operation_index: Option<usize>,
        queue_family: u32,
        uses: &[ReplacementImageUse],
    ) -> Result<PreparedImageState, ReplacementImageStateError> {
        if uses.is_empty() {
            return Err(ReplacementImageStateError::EmptyUseSet);
        }
        let mut unique = BTreeSet::new();
        let mut transitions = Vec::with_capacity(uses.len());
        for use_ in uses.iter().copied() {
            if !unique.insert(use_.image) {
                return Err(ReplacementImageStateError::DuplicateImage(use_.image));
            }
            if matches!(
                use_.use_layout,
                vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED
            ) {
                return Err(ReplacementImageStateError::InvalidUseLayout(use_.image));
            }
            if matches!(
                use_.final_layout,
                vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED
            ) {
                return Err(ReplacementImageStateError::InvalidFinalLayout(use_.image));
            }
            if use_.required_usage.is_empty() {
                return Err(ReplacementImageStateError::EmptyRequiredUsage(use_.image));
            }
            let committed = self
                .images
                .get(&use_.image)
                .copied()
                .ok_or(ReplacementImageStateError::UnknownImage(use_.image))?;
            let predecessor = self
                .pending
                .get(&use_.image)
                .and_then(|pending| pending.last())
                .copied();
            if let Some(pending) = predecessor {
                let follows_in_order = match (pending.operation_index, operation_index) {
                    (Some(previous), Some(current)) => previous < current,
                    (Some(_), None) => true,
                    _ => false,
                };
                let follows_same_exec = follows_in_order
                    && pending.transaction == transaction
                    && pending.queue_family == queue_family;
                if !follows_same_exec {
                    return Err(ReplacementImageStateError::ImageBusy {
                        image: use_.image,
                        transaction: pending.transaction,
                    });
                }
            }
            let state = predecessor.map_or(committed, |pending| ReplacementImageState {
                layout: pending.transition.final_layout,
                sharing: match committed.sharing {
                    ReplacementImageSharing::Concurrent => ReplacementImageSharing::Concurrent,
                    ReplacementImageSharing::Exclusive { .. } => {
                        ReplacementImageSharing::Exclusive {
                            owner: pending.queue_family,
                        }
                    }
                },
                last_use: committed.last_use,
            });
            let queue_transfer = match state.sharing {
                ReplacementImageSharing::Concurrent => None,
                ReplacementImageSharing::Exclusive { owner } if owner == queue_family => None,
                ReplacementImageSharing::Exclusive { owner } => {
                    let source_point = state
                        .last_use
                        .ok_or(ReplacementImageStateError::MissingSourcePoint(use_.image))?;
                    Some(PlannedImageQueueTransfer {
                        source: owner,
                        destination: queue_family,
                        source_point,
                    })
                }
            };
            transitions.push(ReplacementImageTransition {
                image: use_.image,
                required_usage: use_.required_usage,
                initial_layout: state.layout,
                use_layout: use_.use_layout,
                final_layout: use_.final_layout,
                queue_transfer,
            });
        }
        let prepared = PreparedImageState {
            transaction,
            operation_index,
            queue_family,
            transitions: transitions.into_boxed_slice(),
            accepted_releases: Box::new([]),
        };
        for transition in prepared.transitions.iter().copied() {
            self.pending
                .entry(transition.image)
                .or_default()
                .push(PendingImageState {
                    transaction,
                    operation_index,
                    queue_family,
                    transition,
                    release_accepted: false,
                });
        }
        Ok(prepared)
    }

    pub fn validate_prepared(
        &self,
        prepared: &PreparedImageState,
    ) -> Result<(), ReplacementImageStateError> {
        for transition in prepared.transitions.iter().copied() {
            let expected = PendingImageState {
                transaction: prepared.transaction,
                operation_index: prepared.operation_index,
                queue_family: prepared.queue_family,
                transition,
                release_accepted: transition.queue_transfer.is_some_and(|transfer| {
                    prepared.release_accepted(ReplacementImageReleaseKey {
                        source_queue_family: transfer.source,
                        source_queue: transfer.source_point.queue,
                    })
                }),
            };
            match self.pending.get(&transition.image) {
                Some(found) if found.contains(&expected) => {}
                _ => {
                    return Err(ReplacementImageStateError::PreparedMismatch(
                        transition.image,
                    ))
                }
            }
        }
        Ok(())
    }

    pub fn validate_batch(
        &self,
        batch: &PreparedImageStateBatch,
    ) -> Result<(), ReplacementImageStateError> {
        let mut expected = BTreeMap::<ReplacementImageKey, Vec<PendingImageState>>::new();
        let mut previous_index = None;
        for (position, operation) in batch.operations.iter().enumerate() {
            let index_is_ordered = match operation.operation_index {
                Some(index) => previous_index.is_none_or(|previous| index > previous),
                None => position + 1 == batch.operations.len(),
            };
            if operation.transaction != batch.transaction
                || operation.queue_family != batch.queue_family
                || !index_is_ordered
            {
                return Err(ReplacementImageStateError::OperationOrder);
            }
            if let Some(index) = operation.operation_index {
                previous_index = Some(index);
            }
            self.validate_prepared(operation)?;
            for transition in operation.transitions.iter().copied() {
                expected
                    .entry(transition.image)
                    .or_default()
                    .push(PendingImageState {
                        transaction: operation.transaction,
                        operation_index: operation.operation_index,
                        queue_family: operation.queue_family,
                        transition,
                        release_accepted: transition.queue_transfer.is_some_and(|transfer| {
                            operation.release_accepted(ReplacementImageReleaseKey {
                                source_queue_family: transfer.source,
                                source_queue: transfer.source_point.queue,
                            })
                        }),
                    });
            }
        }
        for (image, expected) in expected {
            if self.pending.get(&image) != Some(&expected) {
                return Err(ReplacementImageStateError::PreparedMismatch(image));
            }
        }
        Ok(())
    }

    pub fn cancel_batch(
        &mut self,
        batch: PreparedImageStateBatch,
    ) -> Result<(), ReplacementImageStateError> {
        self.validate_batch(&batch)?;
        if let Some(accepted) = batch
            .operations
            .iter()
            .flat_map(|operation| operation.accepted_releases.iter())
            .next()
            .copied()
        {
            return Err(ReplacementImageStateError::ReleaseCannotCancel(
                accepted.release,
            ));
        }
        let images = batch
            .operations
            .iter()
            .flat_map(|operation| operation.transitions.iter())
            .map(|transition| transition.image)
            .collect::<BTreeSet<_>>();
        for image in images {
            self.pending.remove(&image);
        }
        Ok(())
    }

    pub fn accepted_batch(
        &mut self,
        batch: PreparedImageStateBatch,
        point: QueueTimelinePoint,
    ) -> Result<(), ReplacementImageStateError> {
        self.validate_batch_acceptance(&batch, point)?;
        let mut final_states = BTreeMap::new();
        for operation in batch.operations.iter() {
            for transition in operation.transitions.iter().copied() {
                final_states.insert(
                    transition.image,
                    (transition.final_layout, operation.queue_family),
                );
            }
        }
        for (image, (layout, queue_family)) in final_states {
            let state = self
                .images
                .get_mut(&image)
                .expect("prepared image remains registered");
            state.layout = layout;
            state.last_use = Some(point);
            if matches!(state.sharing, ReplacementImageSharing::Exclusive { .. }) {
                state.sharing = ReplacementImageSharing::Exclusive {
                    owner: queue_family,
                };
            }
            self.pending.remove(&image);
        }
        Ok(())
    }

    pub fn validate_batch_acceptance(
        &self,
        batch: &PreparedImageStateBatch,
        point: QueueTimelinePoint,
    ) -> Result<(), ReplacementImageStateError> {
        if point.epoch != self.epoch {
            return Err(ReplacementImageStateError::MixedEpoch);
        }
        self.validate_batch(batch)?;
        if let Some(release) = batch.operations.iter().find_map(|operation| {
            operation.transitions.iter().find_map(|transition| {
                transition.queue_transfer.and_then(|transfer| {
                    let release = ReplacementImageReleaseKey {
                        source_queue_family: transfer.source,
                        source_queue: transfer.source_point.queue,
                    };
                    (!operation.release_accepted(release)).then_some(release)
                })
            })
        }) {
            return Err(ReplacementImageStateError::ReleasePending(release));
        }
        Ok(())
    }

    pub fn validate_batch_release(
        &self,
        batch: &PreparedImageStateBatch,
        release: ReplacementImageReleaseKey,
    ) -> Result<QueueTimelinePoint, ReplacementImageStateError> {
        self.validate_batch(batch)?;
        let mut predecessor = None;
        for operation in batch.operations.iter() {
            for transfer in operation
                .transitions
                .iter()
                .filter_map(|transition| transition.queue_transfer)
                .filter(|transfer| {
                    transfer.source == release.source_queue_family
                        && transfer.source_point.queue == release.source_queue
                })
            {
                if operation.release_accepted(release) {
                    return Err(ReplacementImageStateError::ReleaseAlreadyAccepted(release));
                }
                predecessor = Some(predecessor.map_or(
                    transfer.source_point,
                    |found: QueueTimelinePoint| {
                        if transfer.source_point.value > found.value {
                            transfer.source_point
                        } else {
                            found
                        }
                    },
                ));
            }
        }
        predecessor.ok_or(ReplacementImageStateError::UnknownRelease(release))
    }

    pub fn batch_release_accepted(
        &mut self,
        batch: PreparedImageStateBatch,
        release: ReplacementImageReleaseKey,
        point: QueueTimelinePoint,
    ) -> Result<PreparedImageStateBatch, ReplacementImageStateError> {
        let predecessor = self.validate_batch_release(&batch, release)?;
        if point.epoch != self.epoch
            || point.queue != release.source_queue
            || point.value <= predecessor.value
        {
            return Err(ReplacementImageStateError::InvalidReleasePoint(release));
        }
        let mut operations = batch.operations.into_vec();
        for operation in operations.iter_mut() {
            let matching_images = operation
                .transitions
                .iter()
                .filter(|transition| {
                    transition.queue_transfer.is_some_and(|transfer| {
                        transfer.source == release.source_queue_family
                            && transfer.source_point.queue == release.source_queue
                    })
                })
                .map(|transition| transition.image)
                .collect::<Vec<_>>();
            if matching_images.is_empty() {
                continue;
            }
            for image in matching_images {
                self.pending
                    .get_mut(&image)
                    .expect("batch release was prevalidated")
                    .iter_mut()
                    .find(|pending| {
                        pending.transaction == operation.transaction
                            && pending.operation_index == operation.operation_index
                    })
                    .expect("batch release operation was prevalidated")
                    .release_accepted = true;
            }
            let mut accepted = operation.accepted_releases.clone().into_vec();
            accepted.push(AcceptedImageRelease { release, point });
            accepted.sort_unstable_by_key(|accepted| accepted.release);
            operation.accepted_releases = accepted.into_boxed_slice();
        }
        Ok(PreparedImageStateBatch {
            transaction: batch.transaction,
            queue_family: batch.queue_family,
            operations: operations.into_boxed_slice(),
        })
    }

    /// Cancel before driver acceptance. Committed layout/ownership is unchanged.
    pub fn cancel(
        &mut self,
        prepared: PreparedImageState,
    ) -> Result<(), ReplacementImageStateError> {
        self.validate_prepared(&prepared)?;
        if let Some(accepted) = prepared.accepted_releases.first().copied() {
            return Err(ReplacementImageStateError::ReleaseCannotCancel(
                accepted.release,
            ));
        }
        for transition in prepared.transitions.iter().copied() {
            if self.pending.get(&transition.image).is_some_and(|pending| {
                pending.last().is_some_and(|found| {
                    found.transaction == prepared.transaction
                        && found.operation_index == prepared.operation_index
                })
            }) {
                continue;
            }
            return Err(ReplacementImageStateError::ImageHasDependentOperation(
                transition.image,
            ));
        }
        for transition in prepared.transitions {
            let pending = self
                .pending
                .get_mut(&transition.image)
                .expect("prepared image remains pending");
            pending.pop();
            if pending.is_empty() {
                self.pending.remove(&transition.image);
            }
        }
        Ok(())
    }

    /// Commit final layout and ownership after the native queue accepted the
    /// submission containing the matching transition program.
    pub fn accepted(
        &mut self,
        prepared: PreparedImageState,
        point: QueueTimelinePoint,
    ) -> Result<(), ReplacementImageStateError> {
        self.validate_acceptance(&prepared, point)?;
        for transition in prepared.transitions {
            let state = self
                .images
                .get_mut(&transition.image)
                .expect("prepared image remains registered");
            state.layout = transition.final_layout;
            state.last_use = Some(point);
            if matches!(state.sharing, ReplacementImageSharing::Exclusive { .. }) {
                state.sharing = ReplacementImageSharing::Exclusive {
                    owner: prepared.queue_family,
                };
            }
            let pending = self
                .pending
                .get(&transition.image)
                .expect("prepared image remains pending");
            if pending.len() != 1 {
                return Err(ReplacementImageStateError::ImageHasDependentOperation(
                    transition.image,
                ));
            }
            self.pending.remove(&transition.image);
        }
        Ok(())
    }

    pub fn validate_acceptance(
        &self,
        prepared: &PreparedImageState,
        point: QueueTimelinePoint,
    ) -> Result<(), ReplacementImageStateError> {
        if point.epoch != self.epoch {
            return Err(ReplacementImageStateError::MixedEpoch);
        }
        self.validate_prepared(prepared)?;
        if let Some(image) = prepared.transitions.iter().find_map(|transition| {
            (self
                .pending
                .get(&transition.image)
                .is_some_and(|pending| pending.len() != 1))
            .then_some(transition.image)
        }) {
            return Err(ReplacementImageStateError::ImageHasDependentOperation(
                image,
            ));
        }
        if let Some(release) = prepared.transitions.iter().find_map(|transition| {
            transition.queue_transfer.and_then(|transfer| {
                let release = ReplacementImageReleaseKey {
                    source_queue_family: transfer.source,
                    source_queue: transfer.source_point.queue,
                };
                (!prepared.release_accepted(release)).then_some(release)
            })
        }) {
            return Err(ReplacementImageStateError::ReleasePending(release));
        }
        Ok(())
    }

    /// Commit driver acceptance of one source-family release batch. The image
    /// remains unavailable to later operations until the matching destination
    /// acquire submission is accepted. This phase cannot be canceled back to
    /// the pre-release ownership state.
    pub fn release_accepted(
        &mut self,
        prepared: PreparedImageState,
        release: ReplacementImageReleaseKey,
        point: QueueTimelinePoint,
    ) -> Result<PreparedImageState, ReplacementImageStateError> {
        let predecessor = self.validate_release(&prepared, release)?;
        if point.epoch != self.epoch
            || point.queue != release.source_queue
            || point.value <= predecessor.value
        {
            return Err(ReplacementImageStateError::InvalidReleasePoint(release));
        }
        for transition in prepared.transitions.iter().copied() {
            if transition.queue_transfer.is_some_and(|transfer| {
                transfer.source == release.source_queue_family
                    && transfer.source_point.queue == release.source_queue
            }) {
                self.pending
                    .get_mut(&transition.image)
                    .expect("prepared image release was prevalidated")
                    .iter_mut()
                    .find(|pending| {
                        pending.transaction == prepared.transaction
                            && pending.operation_index == prepared.operation_index
                    })
                    .expect("prepared image release identity was prevalidated")
                    .release_accepted = true;
            }
        }
        let mut accepted = prepared.accepted_releases.into_vec();
        accepted.push(AcceptedImageRelease { release, point });
        accepted.sort_unstable_by_key(|accepted| accepted.release);
        Ok(PreparedImageState {
            transaction: prepared.transaction,
            operation_index: prepared.operation_index,
            queue_family: prepared.queue_family,
            transitions: prepared.transitions,
            accepted_releases: accepted.into_boxed_slice(),
        })
    }

    pub fn validate_release(
        &self,
        prepared: &PreparedImageState,
        release: ReplacementImageReleaseKey,
    ) -> Result<QueueTimelinePoint, ReplacementImageStateError> {
        self.validate_prepared(prepared)?;
        let predecessor = prepared
            .transitions
            .iter()
            .filter_map(|transition| transition.queue_transfer)
            .filter(|transfer| {
                transfer.source == release.source_queue_family
                    && transfer.source_point.queue == release.source_queue
            })
            .map(|transfer| transfer.source_point)
            .max_by_key(|point| point.value)
            .ok_or(ReplacementImageStateError::UnknownRelease(release))?;
        if prepared.release_accepted(release) {
            return Err(ReplacementImageStateError::ReleaseAlreadyAccepted(release));
        }
        Ok(predecessor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{QueueOwnerId, QueueTimelineValue};

    const EPOCH: VulkanDeviceEpochId = VulkanDeviceEpochId::new(1);

    fn point(queue: u32, value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: EPOCH,
            queue: QueueOwnerId::new(queue),
            value: QueueTimelineValue::new(value),
        }
    }

    fn release(family: u32, queue: u32) -> ReplacementImageReleaseKey {
        ReplacementImageReleaseKey {
            source_queue_family: family,
            source_queue: QueueOwnerId::new(queue),
        }
    }

    fn key(backing: u64) -> ReplacementImageKey {
        ReplacementImageKey {
            backing: BackingId::new(backing),
            representation: RepresentationId::new(backing + 10),
        }
    }

    fn use_(image: ReplacementImageKey, final_layout: vk::ImageLayout) -> ReplacementImageUse {
        ReplacementImageUse {
            image,
            required_usage: vk::ImageUsageFlags::TRANSFER_DST,
            use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            final_layout,
        }
    }

    #[test]
    fn cancellation_preserves_committed_state_and_retry_gets_the_same_transition() {
        let image = key(1);
        let initial = ReplacementImageState {
            layout: vk::ImageLayout::GENERAL,
            sharing: ReplacementImageSharing::Exclusive { owner: 2 },
            last_use: Some(point(2, 3)),
        };
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner.register(image, initial).unwrap();
        let first = owner
            .prepare(
                TransactionId::new(1),
                4,
                [use_(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
            )
            .unwrap();
        assert_eq!(
            first.transitions()[0].queue_transfer,
            Some(PlannedImageQueueTransfer {
                source: 2,
                destination: 4,
                source_point: point(2, 3),
            })
        );
        assert_eq!(
            owner.prepare(
                TransactionId::new(2),
                4,
                [use_(image, vk::ImageLayout::GENERAL)],
            ),
            Err(ReplacementImageStateError::ImageBusy {
                image,
                transaction: TransactionId::new(1),
            })
        );
        owner.cancel(first.clone()).unwrap();
        assert_eq!(owner.state(image), Some(initial));
        let retry = owner
            .prepare(
                TransactionId::new(1),
                4,
                [use_(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
            )
            .unwrap();
        assert_eq!(retry, first);
    }

    #[test]
    fn operation_preparation_retains_the_exact_flattened_position() {
        let image = key(4);
        let state = ReplacementImageState {
            layout: vk::ImageLayout::GENERAL,
            sharing: ReplacementImageSharing::Concurrent,
            last_use: None,
        };
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner.register(image, state).unwrap();
        let prepared = owner
            .prepare_operation(TransactionId::new(3), 7, 0, [use_(image, state.layout)])
            .unwrap();

        assert_eq!(prepared.operation_index(), Some(7));
        owner.cancel(prepared).unwrap();
        assert_eq!(owner.state(image), Some(state));
    }

    #[test]
    fn same_exec_image_chain_plans_from_prior_final_state_and_commits_atomically() {
        let image = key(5);
        let initial = ReplacementImageState {
            layout: vk::ImageLayout::GENERAL,
            sharing: ReplacementImageSharing::Exclusive { owner: 3 },
            last_use: Some(point(3, 4)),
        };
        let operations = || -> Box<[(usize, Box<[ReplacementImageUse]>)]> {
            vec![
                (
                    2,
                    Box::new([use_(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]) as Box<[_]>,
                ),
                (
                    6,
                    Box::new([use_(image, vk::ImageLayout::GENERAL)]) as Box<[_]>,
                ),
            ]
            .into_boxed_slice()
        };
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner.register(image, initial).unwrap();
        let first = owner
            .prepare_batch(TransactionId::new(8), 3, operations())
            .unwrap();
        assert_eq!(first.operations()[0].operation_index(), Some(2));
        assert_eq!(first.operations()[1].operation_index(), Some(6));
        assert_eq!(
            first.operations()[1].transitions()[0].initial_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(owner.state(image), Some(initial));

        owner.cancel_batch(first.clone()).unwrap();
        assert_eq!(owner.state(image), Some(initial));
        let retry = owner
            .prepare_batch(TransactionId::new(8), 3, operations())
            .unwrap();
        assert_eq!(retry, first);
        owner.accepted_batch(retry, point(3, 5)).unwrap();
        assert_eq!(
            owner.state(image),
            Some(ReplacementImageState {
                layout: vk::ImageLayout::GENERAL,
                sharing: ReplacementImageSharing::Exclusive { owner: 3 },
                last_use: Some(point(3, 5)),
            })
        );
    }

    #[test]
    fn auxiliary_tail_is_owned_after_the_last_semantic_image_operation() {
        let image = key(15);
        let initial = ReplacementImageState {
            layout: vk::ImageLayout::GENERAL,
            sharing: ReplacementImageSharing::Concurrent,
            last_use: None,
        };
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner.register(image, initial).unwrap();
        let batch = owner
            .prepare_batch_with_auxiliary_tail(
                TransactionId::new(18),
                0,
                vec![(
                    3,
                    vec![use_(image, vk::ImageLayout::GENERAL)].into_boxed_slice(),
                )]
                .into_boxed_slice(),
                vec![ReplacementImageUse {
                    image,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                }]
                .into_boxed_slice(),
            )
            .unwrap();

        assert_eq!(batch.operations().len(), 2);
        assert_eq!(batch.operations()[0].operation_index(), Some(3));
        assert_eq!(batch.operations()[1].operation_index(), None);
        assert_eq!(
            batch.operations()[1].transitions()[0].initial_layout,
            vk::ImageLayout::GENERAL
        );
        owner.cancel_batch(batch).unwrap();
        assert_eq!(owner.state(image), Some(initial));
    }

    #[test]
    fn malformed_batch_suffix_rolls_back_every_prepared_image() {
        let image = key(6);
        let missing = key(9);
        let initial = ReplacementImageState {
            layout: vk::ImageLayout::GENERAL,
            sharing: ReplacementImageSharing::Concurrent,
            last_use: None,
        };
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner.register(image, initial).unwrap();
        assert_eq!(
            owner.prepare_batch(
                TransactionId::new(9),
                0,
                vec![
                    (1, Box::new([use_(image, initial.layout)]) as Box<[_]>),
                    (2, Box::new([use_(missing, initial.layout)]) as Box<[_]>),
                ]
                .into_boxed_slice(),
            ),
            Err(ReplacementImageStateError::UnknownImage(missing))
        );
        let retry = owner
            .prepare_operation(TransactionId::new(9), 1, 0, [use_(image, initial.layout)])
            .unwrap();
        owner.cancel(retry).unwrap();
    }

    #[test]
    fn one_batch_release_orders_the_first_use_and_unlocks_the_whole_chain() {
        let image = key(10);
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 2 },
                    last_use: Some(point(2, 5)),
                },
            )
            .unwrap();
        let batch = owner
            .prepare_batch(
                TransactionId::new(10),
                4,
                vec![
                    (
                        1,
                        Box::new([use_(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)])
                            as Box<[_]>,
                    ),
                    (
                        2,
                        Box::new([use_(image, vk::ImageLayout::GENERAL)]) as Box<[_]>,
                    ),
                ]
                .into_boxed_slice(),
            )
            .unwrap();
        assert!(batch.operations()[0].transitions()[0]
            .queue_transfer
            .is_some());
        assert!(batch.operations()[1].transitions()[0]
            .queue_transfer
            .is_none());
        assert_eq!(
            owner.accepted_batch(batch.clone(), point(4, 6)),
            Err(ReplacementImageStateError::ReleasePending(release(2, 2)))
        );
        let batch = owner
            .batch_release_accepted(batch, release(2, 2), point(2, 6))
            .unwrap();
        assert_eq!(batch.release_points().as_ref(), [point(2, 6)]);
        owner.accepted_batch(batch, point(4, 6)).unwrap();
        assert_eq!(
            owner.state(image).unwrap().sharing,
            ReplacementImageSharing::Exclusive { owner: 4 }
        );
    }

    #[test]
    fn acceptance_commits_final_layout_and_exclusive_queue_owner() {
        let image = key(1);
        let other = key(2);
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 1 },
                    last_use: Some(point(1, 7)),
                },
            )
            .unwrap();
        owner
            .register(
                other,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let prepared = owner
            .prepare(
                TransactionId::new(3),
                5,
                [
                    use_(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    use_(other, vk::ImageLayout::GENERAL),
                ],
            )
            .unwrap();
        let prepared = owner
            .release_accepted(prepared, release(1, 1), point(1, 8))
            .unwrap();
        owner.accepted(prepared, point(5, 8)).unwrap();
        assert_eq!(
            owner.state(image),
            Some(ReplacementImageState {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sharing: ReplacementImageSharing::Exclusive { owner: 5 },
                last_use: Some(point(5, 8)),
            })
        );
        assert_eq!(
            owner.state(other),
            Some(ReplacementImageState {
                layout: vk::ImageLayout::GENERAL,
                sharing: ReplacementImageSharing::Concurrent,
                last_use: Some(point(5, 8)),
            })
        );
    }

    #[test]
    fn one_busy_or_duplicate_image_refuses_the_whole_batch() {
        let first = key(1);
        let second = key(2);
        let state = ReplacementImageState {
            layout: vk::ImageLayout::GENERAL,
            sharing: ReplacementImageSharing::Concurrent,
            last_use: None,
        };
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner.register(first, state).unwrap();
        owner.register(second, state).unwrap();
        owner
            .prepare(TransactionId::new(1), 0, [use_(second, state.layout)])
            .unwrap();
        assert!(matches!(
            owner.prepare(
                TransactionId::new(2),
                0,
                [use_(first, state.layout), use_(second, state.layout)],
            ),
            Err(ReplacementImageStateError::ImageBusy { image, .. }) if image == second
        ));
        let first_only = owner
            .prepare(TransactionId::new(2), 0, [use_(first, state.layout)])
            .unwrap();
        owner.cancel(first_only).unwrap();
        assert_eq!(
            owner.prepare(
                TransactionId::new(2),
                0,
                [use_(first, state.layout), use_(first, state.layout)],
            ),
            Err(ReplacementImageStateError::DuplicateImage(first))
        );
    }

    #[test]
    fn accepted_release_cannot_rollback_and_destination_acceptance_requires_it() {
        let image = key(7);
        let mut owner = ReplacementImageStateOwner::new(EPOCH);
        owner
            .register(
                image,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Exclusive { owner: 2 },
                    last_use: Some(point(2, 5)),
                },
            )
            .unwrap();
        let prepared = owner
            .prepare(
                TransactionId::new(9),
                4,
                [use_(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
            )
            .unwrap();
        assert_eq!(
            owner.accepted(prepared.clone(), point(4, 6)),
            Err(ReplacementImageStateError::ReleasePending(release(2, 2)))
        );
        let released = owner
            .release_accepted(prepared, release(2, 2), point(2, 6))
            .unwrap();
        assert_eq!(
            released.accepted_releases(),
            [AcceptedImageRelease {
                release: release(2, 2),
                point: point(2, 6),
            }]
        );
        assert_eq!(
            owner.cancel(released.clone()),
            Err(ReplacementImageStateError::ReleaseCannotCancel(release(
                2, 2
            )))
        );
        assert_eq!(
            owner.state(image).unwrap().sharing,
            ReplacementImageSharing::Exclusive { owner: 2 }
        );
        owner.accepted(released, point(4, 6)).unwrap();
        assert_eq!(
            owner.state(image).unwrap().sharing,
            ReplacementImageSharing::Exclusive { owner: 4 }
        );
    }
}
