//! Resolve event, fence, and completion-stamp conditions to exact producers.
//!
//! A wait may arrive before any satisfying signal transaction. It remains an
//! unresolved hold rather than inventing a future transaction identity. Signal
//! admission binds only matching holds; signal publication records the
//! condition value for waits admitted later. No method waits on a host thread.

use crate::{
    EventOperation, EventOperationKind, ExplicitWaitCause, NamespaceError, ReferenceNamespace,
};
use reims_vgpu_protocol::{
    ChannelId, EventObject, FenceObject, IngressOrdinal, ResourceId, SerializerRef, StampWait,
    TaskId, TransactionId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionNamespaceError {
    UnboundReference,
    FenceGenerationExhausted,
    Namespace(NamespaceError),
}

impl From<NamespaceError> for ConditionNamespaceError {
    fn from(error: NamespaceError) -> Self {
        Self::Namespace(error)
    }
}

/// Generational event and fence identities for one semantic session.
///
/// Identity publication is separate from signal-value publication. A future
/// wait can therefore resolve the object created by the API before any signal
/// value exists, while equal event and fence reference integers remain in
/// distinct typed namespaces.
#[derive(Clone, Debug, Default)]
pub struct ConditionNamespaceOwner {
    events: ReferenceNamespace<EventObject>,
    fences: ReferenceNamespace<FenceObject>,
    fence_generations: BTreeMap<(TaskId, SerializerRef<FenceObject>), u64>,
}

impl ConditionNamespaceOwner {
    pub fn task_identities(&self, task: TaskId) -> TaskConditionIdentities {
        TaskConditionIdentities {
            events: self.events.live_for_task(task),
            fences: self.fences.live_for_task(task),
        }
    }

    pub fn publish_event(
        &mut self,
        task: TaskId,
        reference: SerializerRef<EventObject>,
    ) -> Result<ResourceId<EventObject>, ConditionNamespaceError> {
        if reference.get() == 0 {
            return Err(ConditionNamespaceError::UnboundReference);
        }
        self.events.publish(task, reference).map_err(Into::into)
    }

    pub fn publish_fence(
        &mut self,
        task: TaskId,
        reference: SerializerRef<FenceObject>,
    ) -> Result<ResourceId<FenceObject>, ConditionNamespaceError> {
        if reference.get() == 0 {
            return Err(ConditionNamespaceError::UnboundReference);
        }
        self.fences.publish(task, reference).map_err(Into::into)
    }

    pub fn resolve_event(
        &self,
        task: TaskId,
        reference: SerializerRef<EventObject>,
    ) -> Option<ResourceId<EventObject>> {
        self.events.resolve(task, reference)
    }

    pub fn resolve_event_operation(
        &mut self,
        task: TaskId,
        reference: SerializerRef<EventObject>,
        kind: EventOperationKind,
        value: u64,
    ) -> Result<EventOperation, ConditionNamespaceError> {
        if reference.get() == 0 {
            return Err(ConditionNamespaceError::UnboundReference);
        }
        let event = self.publish_event(task, reference)?;
        Ok(EventOperation { event, kind, value })
    }

    pub fn resolve_fence(
        &self,
        task: TaskId,
        reference: SerializerRef<FenceObject>,
    ) -> Option<ResourceId<FenceObject>> {
        self.fences.resolve(task, reference)
    }

    pub fn resolve_fence_operation(
        &mut self,
        task: TaskId,
        reference: SerializerRef<FenceObject>,
        kind: crate::FenceOperationKind,
        scope: crate::FenceScope,
    ) -> Result<crate::FenceOperation, ConditionNamespaceError> {
        if reference.get() == 0 {
            return Err(ConditionNamespaceError::UnboundReference);
        }
        let fence = self.publish_fence(task, reference)?;
        let generation = match kind {
            crate::FenceOperationKind::Update => {
                let next = self
                    .fence_generations
                    .get(&(task, reference))
                    .copied()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or(ConditionNamespaceError::FenceGenerationExhausted)?;
                self.fence_generations.insert((task, reference), next);
                next
            }
            crate::FenceOperationKind::Wait => self
                .fence_generations
                .get(&(task, reference))
                .copied()
                .unwrap_or(1),
        };
        Ok(crate::FenceOperation {
            fence,
            kind,
            generation,
            scope,
        })
    }

    pub fn release_event(&mut self, task: TaskId, reference: SerializerRef<EventObject>) -> bool {
        self.events.release(task, reference)
    }

    pub fn release_fence(&mut self, task: TaskId, reference: SerializerRef<FenceObject>) -> bool {
        self.fence_generations.remove(&(task, reference));
        self.fences.release(task, reference)
    }

    pub fn release_task(&mut self, task: TaskId) -> ReleasedConditionIdentities {
        self.fence_generations
            .retain(|(owner, _), _| *owner != task);
        ReleasedConditionIdentities {
            events: self.events.release_task(task),
            fences: self.fences.release_task(task),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleasedConditionIdentities {
    pub events: usize,
    pub fences: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskConditionIdentities {
    pub events: Box<[(SerializerRef<EventObject>, ResourceId<EventObject>)]>,
    pub fences: Box<[(SerializerRef<FenceObject>, ResourceId<FenceObject>)]>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ConditionKey {
    Stamp(ChannelId),
    Event(ResourceId<EventObject>),
    Fence(ResourceId<FenceObject>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionSignal {
    Stamp {
        channel: ChannelId,
        value: u32,
    },
    Event {
        event: ResourceId<EventObject>,
        value: u64,
    },
    Fence {
        fence: ResourceId<FenceObject>,
        generation: u64,
    },
}

impl ConditionSignal {
    fn key(self) -> ConditionKey {
        match self {
            Self::Stamp { channel, .. } => ConditionKey::Stamp(channel),
            Self::Event { event, .. } => ConditionKey::Event(event),
            Self::Fence { fence, .. } => ConditionKey::Fence(fence),
        }
    }

    fn publication_boundary(self) -> ConditionPublicationBoundary {
        match self {
            Self::Stamp { .. } => ConditionPublicationBoundary::GuestPublication,
            Self::Event { .. } | Self::Fence { .. } => ConditionPublicationBoundary::GpuCompletion,
        }
    }
}

/// The contract point at which a signal becomes observable to a later wait.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionPublicationBoundary {
    GpuCompletion,
    GuestPublication,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionWaitResolution {
    Satisfied,
    Producer(TransactionId),
    Pending,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionBinding {
    pub consumer: TransactionId,
    pub producer: TransactionId,
    pub cause: ExplicitWaitCause,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionOwnerError {
    DuplicateWait,
    DuplicateSignal,
    UnknownWait,
    WaitAlreadyResolved,
    UnknownSignalTransaction,
    NoSignalsAtBoundary,
    SignalAlreadyPublished,
    SignalNotPublished,
    TransactionHasPendingWait,
    ConditionInUse,
}

#[derive(Clone, Copy, Debug)]
struct SignalRecord {
    transaction: TransactionId,
    ordinal: IngressOrdinal,
    signal: ConditionSignal,
    published: bool,
}

#[derive(Clone, Copy, Debug)]
struct WaitRecord {
    resolution: ConditionWaitResolution,
}

#[derive(Clone, Debug, Default)]
pub struct SynchronizationConditionOwner {
    signals: BTreeMap<ConditionKey, Vec<SignalRecord>>,
    signal_transactions: BTreeMap<TransactionId, BTreeSet<ConditionKey>>,
    published: BTreeMap<ConditionKey, ConditionSignal>,
    waits: BTreeMap<(TransactionId, ExplicitWaitCause), WaitRecord>,
}

impl SynchronizationConditionOwner {
    pub fn has_signals_at(
        &self,
        transaction: TransactionId,
        boundary: ConditionPublicationBoundary,
    ) -> bool {
        self.signal_transactions
            .get(&transaction)
            .into_iter()
            .flat_map(|keys| keys.iter())
            .any(|key| {
                self.signals[key].iter().any(|record| {
                    record.transaction == transaction
                        && record.signal.publication_boundary() == boundary
                })
            })
    }

    /// The signal this condition has actually published, if any.
    ///
    /// [`Self::classify_wait`] answers satisfied, bound or pending, which is
    /// the right shape for a decision and the wrong one for a diagnostic: a
    /// wait reported as `Pending` says that nothing has satisfied it and not
    /// how far behind the condition is. A stamp wait for value 109 against a
    /// channel that has published 108 and one against a channel that has
    /// published nothing at all read identically, and they are different bugs.
    ///
    /// Keyed through the same `condition_key` the classifier uses, so the two
    /// cannot come to disagree about which condition a cause names.
    pub fn published_signal(&self, cause: ExplicitWaitCause) -> Option<ConditionSignal> {
        self.published.get(&condition_key(cause)).copied()
    }

    pub fn classify_wait(&self, cause: ExplicitWaitCause) -> ConditionWaitResolution {
        let key = condition_key(cause);
        if self
            .published
            .get(&key)
            .is_some_and(|signal| satisfies(*signal, cause))
        {
            ConditionWaitResolution::Satisfied
        } else if let Some(producer) = self.first_satisfying_producer(key, cause) {
            ConditionWaitResolution::Producer(producer)
        } else {
            ConditionWaitResolution::Pending
        }
    }

    pub fn register_wait(
        &mut self,
        consumer: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<ConditionWaitResolution, ConditionOwnerError> {
        if self.waits.contains_key(&(consumer, cause)) {
            return Err(ConditionOwnerError::DuplicateWait);
        }
        let resolution = self.classify_wait(cause);
        self.waits
            .insert((consumer, cause), WaitRecord { resolution });
        Ok(resolution)
    }

    pub fn register_signal(
        &mut self,
        transaction: TransactionId,
        ordinal: IngressOrdinal,
        signal: ConditionSignal,
    ) -> Result<Vec<ConditionBinding>, ConditionOwnerError> {
        let key = signal.key();
        if self.signals.get(&key).is_some_and(|signals| {
            signals
                .iter()
                .any(|record| record.transaction == transaction)
        }) {
            return Err(ConditionOwnerError::DuplicateSignal);
        }
        self.signals.entry(key).or_default().push(SignalRecord {
            transaction,
            ordinal,
            signal,
            published: false,
        });
        self.signals
            .get_mut(&key)
            .unwrap()
            .sort_by_key(|record| (record.ordinal, record.transaction));
        self.signal_transactions
            .entry(transaction)
            .or_default()
            .insert(key);

        let pending = self
            .waits
            .iter()
            .filter_map(|(&(consumer, cause), wait)| {
                (wait.resolution == ConditionWaitResolution::Pending
                    && condition_key(cause) == key
                    && satisfies(signal, cause))
                .then_some((consumer, cause))
            })
            .collect::<Vec<_>>();
        let mut bindings = Vec::with_capacity(pending.len());
        for (consumer, cause) in pending {
            self.waits.get_mut(&(consumer, cause)).unwrap().resolution =
                ConditionWaitResolution::Producer(transaction);
            bindings.push(ConditionBinding {
                consumer,
                producer: transaction,
                cause,
            });
        }
        Ok(bindings)
    }

    pub fn bind_wait(
        &mut self,
        consumer: TransactionId,
        cause: ExplicitWaitCause,
        producer: TransactionId,
    ) -> Result<(), ConditionOwnerError> {
        let wait = self
            .waits
            .get_mut(&(consumer, cause))
            .ok_or(ConditionOwnerError::UnknownWait)?;
        if wait.resolution != ConditionWaitResolution::Pending {
            return Err(ConditionOwnerError::WaitAlreadyResolved);
        }
        wait.resolution = ConditionWaitResolution::Producer(producer);
        Ok(())
    }

    pub fn satisfy_wait(
        &mut self,
        consumer: TransactionId,
        cause: ExplicitWaitCause,
    ) -> Result<(), ConditionOwnerError> {
        let wait = self
            .waits
            .get_mut(&(consumer, cause))
            .ok_or(ConditionOwnerError::UnknownWait)?;
        if wait.resolution != ConditionWaitResolution::Pending {
            return Err(ConditionOwnerError::WaitAlreadyResolved);
        }
        wait.resolution = ConditionWaitResolution::Satisfied;
        Ok(())
    }

    /// Publish the transaction's signals that become observable at `boundary`.
    /// The complete matching set is validated before any condition changes.
    pub fn publish_transaction_at(
        &mut self,
        transaction: TransactionId,
        boundary: ConditionPublicationBoundary,
    ) -> Result<(), ConditionOwnerError> {
        let keys = self
            .signal_transactions
            .get(&transaction)
            .cloned()
            .ok_or(ConditionOwnerError::UnknownSignalTransaction)?;
        let records = keys
            .iter()
            .filter_map(|key| {
                self.signals
                    .get(key)
                    .and_then(|signals| {
                        signals
                            .iter()
                            .find(|record| record.transaction == transaction)
                    })
                    .filter(|record| record.signal.publication_boundary() == boundary)
                    .map(|record| (*key, *record))
            })
            .collect::<Vec<_>>();
        if records.is_empty() {
            return Err(ConditionOwnerError::NoSignalsAtBoundary);
        }
        if records.iter().any(|(_, record)| record.published) {
            return Err(ConditionOwnerError::SignalAlreadyPublished);
        }
        for (key, published_record) in records {
            let record = self
                .signals
                .get_mut(&key)
                .and_then(|signals| {
                    signals
                        .iter_mut()
                        .find(|record| record.transaction == transaction)
                })
                .expect("transaction index names a signal record");
            record.published = true;
            self.published.insert(key, published_record.signal);
        }
        Ok(())
    }

    pub fn retire_transaction(
        &mut self,
        transaction: TransactionId,
    ) -> Result<(), ConditionOwnerError> {
        if self.waits.iter().any(|(&(consumer, _), wait)| {
            consumer == transaction && wait.resolution == ConditionWaitResolution::Pending
        }) {
            return Err(ConditionOwnerError::TransactionHasPendingWait);
        }
        if self
            .signal_transactions
            .get(&transaction)
            .into_iter()
            .flat_map(|keys| keys.iter())
            .any(|key| {
                self.signals[key]
                    .iter()
                    .any(|record| record.transaction == transaction && !record.published)
            })
        {
            return Err(ConditionOwnerError::SignalNotPublished);
        }
        self.waits
            .retain(|&(consumer, _), _| consumer != transaction);
        if let Some(keys) = self.signal_transactions.remove(&transaction) {
            for key in keys {
                if let Some(signals) = self.signals.get_mut(&key) {
                    signals.retain(|record| record.transaction != transaction);
                    if signals.is_empty() {
                        self.signals.remove(&key);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn clear_event(
        &mut self,
        event: ResourceId<EventObject>,
    ) -> Result<(), ConditionOwnerError> {
        self.clear_keys([ConditionKey::Event(event)])
    }

    pub fn clear_fence(
        &mut self,
        fence: ResourceId<FenceObject>,
    ) -> Result<(), ConditionOwnerError> {
        self.clear_keys([ConditionKey::Fence(fence)])
    }

    pub fn clear_stamp_channel(&mut self, channel: ChannelId) -> Result<(), ConditionOwnerError> {
        let mut keys = self
            .published
            .keys()
            .chain(self.signals.keys())
            .copied()
            .filter(|key| matches!(key, ConditionKey::Stamp(owner) if *owner == channel))
            .collect::<BTreeSet<_>>();
        keys.extend(
            self.waits
                .keys()
                .map(|(_, cause)| condition_key(*cause))
                .filter(|key| matches!(key, ConditionKey::Stamp(owner) if *owner == channel)),
        );
        self.clear_keys(keys)
    }

    fn clear_keys(
        &mut self,
        keys: impl IntoIterator<Item = ConditionKey>,
    ) -> Result<(), ConditionOwnerError> {
        let keys = keys.into_iter().collect::<BTreeSet<_>>();
        if keys.iter().any(|key| {
            self.signals
                .get(key)
                .is_some_and(|signals| !signals.is_empty())
                || self
                    .waits
                    .keys()
                    .any(|(_, cause)| condition_key(*cause) == *key)
        }) {
            return Err(ConditionOwnerError::ConditionInUse);
        }
        for key in keys {
            self.clear_key(key);
        }
        Ok(())
    }

    fn clear_key(&mut self, key: ConditionKey) {
        self.published.remove(&key);
        self.signals.remove(&key);
        self.signal_transactions.values_mut().for_each(|keys| {
            keys.remove(&key);
        });
        self.signal_transactions.retain(|_, keys| !keys.is_empty());
    }

    fn first_satisfying_producer(
        &self,
        key: ConditionKey,
        cause: ExplicitWaitCause,
    ) -> Option<TransactionId> {
        self.signals.get(&key).and_then(|signals| {
            signals
                .iter()
                .find(|record| satisfies(record.signal, cause))
                .map(|record| record.transaction)
        })
    }
}

fn condition_key(cause: ExplicitWaitCause) -> ConditionKey {
    match cause {
        ExplicitWaitCause::Stamp { source_channel, .. } => ConditionKey::Stamp(source_channel),
        ExplicitWaitCause::Event { event, .. } => ConditionKey::Event(event),
        ExplicitWaitCause::Fence { fence, .. } => ConditionKey::Fence(fence),
    }
}

fn satisfies(signal: ConditionSignal, cause: ExplicitWaitCause) -> bool {
    match (signal, cause) {
        (
            ConditionSignal::Stamp { channel, value },
            ExplicitWaitCause::Stamp {
                source_channel,
                value: wait_value,
            },
        ) => {
            channel == source_channel
                && StampWait {
                    index: source_channel.get(),
                    value: wait_value,
                }
                .satisfied_by(value)
        }
        (
            ConditionSignal::Event { event, value },
            ExplicitWaitCause::Event {
                event: wait_event,
                value: wait_value,
            },
        ) => event == wait_event && value >= wait_value,
        (
            ConditionSignal::Fence { fence, generation },
            ExplicitWaitCause::Fence {
                fence: wait_fence,
                generation: wait_generation,
            },
        ) => fence == wait_fence && generation >= wait_generation,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_identity_precedes_value_and_reuse_advances_generation() {
        let mut namespaces = ConditionNamespaceOwner::default();
        let task = TaskId::new(3);
        let event_ref = SerializerRef::new(7);
        let fence_ref = SerializerRef::new(7);
        let event = namespaces.publish_event(task, event_ref).unwrap();
        let fence = namespaces.publish_fence(task, fence_ref).unwrap();
        assert_eq!(namespaces.resolve_event(task, event_ref), Some(event));
        assert_eq!(namespaces.resolve_fence(task, fence_ref), Some(fence));

        let mut conditions = SynchronizationConditionOwner::default();
        assert_eq!(
            namespaces.resolve_event_operation(task, event_ref, EventOperationKind::Wait, 9),
            Ok(EventOperation {
                event,
                kind: EventOperationKind::Wait,
                value: 9,
            })
        );
        assert_eq!(
            conditions
                .register_wait(
                    TransactionId::new(1),
                    ExplicitWaitCause::Event { event, value: 9 },
                )
                .unwrap(),
            ConditionWaitResolution::Pending
        );

        assert!(namespaces.release_event(task, event_ref));
        let reused = namespaces.publish_event(task, event_ref).unwrap();
        assert_eq!(reused.index(), event.index());
        assert_ne!(reused.generation(), event.generation());
    }

    #[test]
    fn fence_operations_allocate_in_record_order_without_consulting_publication() {
        let mut namespaces = ConditionNamespaceOwner::default();
        let task = TaskId::new(3);
        let reference = SerializerRef::new(7);
        let first_wait = namespaces
            .resolve_fence_operation(
                task,
                reference,
                crate::FenceOperationKind::Wait,
                crate::FenceScope::Render(crate::RenderBarrierStages::VERTEX),
            )
            .unwrap();
        assert_eq!(first_wait.generation, 1);
        assert_eq!(
            first_wait.scope,
            crate::FenceScope::Render(crate::RenderBarrierStages::VERTEX)
        );
        let first_update = namespaces
            .resolve_fence_operation(
                task,
                reference,
                crate::FenceOperationKind::Update,
                crate::FenceScope::Render(crate::RenderBarrierStages::FRAGMENT),
            )
            .unwrap();
        assert_eq!(first_update.fence, first_wait.fence);
        assert_eq!(first_update.generation, 1);
        assert_eq!(
            first_update.scope,
            crate::FenceScope::Render(crate::RenderBarrierStages::FRAGMENT)
        );
        let second_update = namespaces
            .resolve_fence_operation(
                task,
                reference,
                crate::FenceOperationKind::Update,
                crate::FenceScope::Compute,
            )
            .unwrap();
        assert_eq!(second_update.generation, 2);
        let later_wait = namespaces
            .resolve_fence_operation(
                task,
                reference,
                crate::FenceOperationKind::Wait,
                crate::FenceScope::Compute,
            )
            .unwrap();
        assert_eq!(later_wait.generation, 2);

        assert!(namespaces.release_fence(task, reference));
        let reused = namespaces
            .resolve_fence_operation(
                task,
                reference,
                crate::FenceOperationKind::Wait,
                crate::FenceScope::Compute,
            )
            .unwrap();
        assert_eq!(reused.generation, 1);
        assert_eq!(reused.fence.index(), first_wait.fence.index());
        assert_ne!(reused.fence.generation(), first_wait.fence.generation());
    }

    #[test]
    fn zero_is_not_a_condition_identity() {
        let mut namespaces = ConditionNamespaceOwner::default();
        assert_eq!(
            namespaces.publish_event(TaskId::new(1), SerializerRef::new(0)),
            Err(ConditionNamespaceError::UnboundReference)
        );
        assert_eq!(
            namespaces.publish_fence(TaskId::new(1), SerializerRef::new(0)),
            Err(ConditionNamespaceError::UnboundReference)
        );
        let first_wait = namespaces
            .resolve_event_operation(
                TaskId::new(1),
                SerializerRef::new(9),
                EventOperationKind::Wait,
                1,
            )
            .unwrap();
        assert_eq!(
            namespaces.resolve_event(TaskId::new(1), SerializerRef::new(9)),
            Some(first_wait.event)
        );
    }

    fn event(value: u64) -> ExplicitWaitCause {
        ExplicitWaitCause::Event {
            event: ResourceId::new(4, 2),
            value,
        }
    }

    #[test]
    fn future_signal_binds_only_matching_pending_waits() {
        let mut owner = SynchronizationConditionOwner::default();
        assert_eq!(
            owner.register_wait(TransactionId::new(1), event(7)),
            Ok(ConditionWaitResolution::Pending)
        );
        assert_eq!(
            owner
                .register_signal(
                    TransactionId::new(2),
                    IngressOrdinal::new(2),
                    ConditionSignal::Event {
                        event: ResourceId::new(4, 2),
                        value: 7,
                    },
                )
                .unwrap(),
            vec![ConditionBinding {
                consumer: TransactionId::new(1),
                producer: TransactionId::new(2),
                cause: event(7),
            }]
        );
    }

    #[test]
    fn published_signal_satisfies_later_wait_without_a_retired_producer_edge() {
        let mut owner = SynchronizationConditionOwner::default();
        owner
            .register_signal(
                TransactionId::new(2),
                IngressOrdinal::new(2),
                ConditionSignal::Event {
                    event: ResourceId::new(4, 2),
                    value: 9,
                },
            )
            .unwrap();
        owner
            .publish_transaction_at(
                TransactionId::new(2),
                ConditionPublicationBoundary::GpuCompletion,
            )
            .unwrap();
        owner.retire_transaction(TransactionId::new(2)).unwrap();
        assert_eq!(
            owner.register_wait(TransactionId::new(3), event(7)),
            Ok(ConditionWaitResolution::Satisfied)
        );
    }

    #[test]
    fn stamp_satisfaction_uses_signed_wrapping_comparison() {
        let mut owner = SynchronizationConditionOwner::default();
        let channel = ChannelId::new(3);
        owner
            .register_signal(
                TransactionId::new(1),
                IngressOrdinal::new(1),
                ConditionSignal::Stamp { channel, value: 0 },
            )
            .unwrap();
        owner
            .publish_transaction_at(
                TransactionId::new(1),
                ConditionPublicationBoundary::GuestPublication,
            )
            .unwrap();
        assert_eq!(
            owner.register_wait(
                TransactionId::new(2),
                ExplicitWaitCause::Stamp {
                    source_channel: channel,
                    value: u32::MAX,
                },
            ),
            Ok(ConditionWaitResolution::Satisfied)
        );
    }

    #[test]
    fn generational_event_clear_withdraws_only_that_condition_state() {
        let mut owner = SynchronizationConditionOwner::default();
        let old = ResourceId::new(4, 2);
        owner
            .register_signal(
                TransactionId::new(1),
                IngressOrdinal::new(1),
                ConditionSignal::Event {
                    event: old,
                    value: 5,
                },
            )
            .unwrap();
        owner
            .publish_transaction_at(
                TransactionId::new(1),
                ConditionPublicationBoundary::GpuCompletion,
            )
            .unwrap();
        owner.retire_transaction(TransactionId::new(1)).unwrap();
        owner.clear_event(old).unwrap();
        assert_eq!(
            owner.register_wait(
                TransactionId::new(2),
                ExplicitWaitCause::Event {
                    event: ResourceId::new(4, 3),
                    value: 5,
                },
            ),
            Ok(ConditionWaitResolution::Pending)
        );
    }

    #[test]
    fn condition_clear_refuses_live_signal_and_wait_ownership() {
        let mut owner = SynchronizationConditionOwner::default();
        let event_id = ResourceId::new(4, 2);
        owner
            .register_signal(
                TransactionId::new(1),
                IngressOrdinal::new(1),
                ConditionSignal::Event {
                    event: event_id,
                    value: 5,
                },
            )
            .unwrap();
        assert_eq!(
            owner.clear_event(event_id),
            Err(ConditionOwnerError::ConditionInUse)
        );
        owner
            .publish_transaction_at(
                TransactionId::new(1),
                ConditionPublicationBoundary::GpuCompletion,
            )
            .unwrap();
        owner.retire_transaction(TransactionId::new(1)).unwrap();
        owner
            .register_wait(TransactionId::new(2), event(5))
            .unwrap();
        assert_eq!(
            owner.clear_event(event_id),
            Err(ConditionOwnerError::ConditionInUse)
        );
    }

    #[test]
    fn mixed_signals_publish_only_at_their_contract_boundary() {
        let mut owner = SynchronizationConditionOwner::default();
        let transaction = TransactionId::new(1);
        let channel = ChannelId::new(3);
        let event_id = ResourceId::new(4, 2);
        owner
            .register_signal(
                transaction,
                IngressOrdinal::new(1),
                ConditionSignal::Stamp { channel, value: 9 },
            )
            .unwrap();
        owner
            .register_signal(
                transaction,
                IngressOrdinal::new(1),
                ConditionSignal::Event {
                    event: event_id,
                    value: 7,
                },
            )
            .unwrap();

        owner
            .publish_transaction_at(transaction, ConditionPublicationBoundary::GpuCompletion)
            .unwrap();
        assert_eq!(
            owner.classify_wait(event(7)),
            ConditionWaitResolution::Satisfied
        );
        assert_eq!(
            owner.classify_wait(ExplicitWaitCause::Stamp {
                source_channel: channel,
                value: 9,
            }),
            ConditionWaitResolution::Producer(transaction)
        );
        assert_eq!(
            owner.publish_transaction_at(transaction, ConditionPublicationBoundary::GpuCompletion,),
            Err(ConditionOwnerError::SignalAlreadyPublished)
        );

        owner
            .publish_transaction_at(transaction, ConditionPublicationBoundary::GuestPublication)
            .unwrap();
        assert_eq!(
            owner.classify_wait(ExplicitWaitCause::Stamp {
                source_channel: channel,
                value: 9,
            }),
            ConditionWaitResolution::Satisfied
        );
    }

    /// A pending wait is reported beside how far its condition actually got.
    ///
    /// `classify_wait` collapses "nothing has signalled this channel at all"
    /// and "this channel has signalled, one short" into the same `Pending`, and
    /// a device that has stopped needs those told apart.
    #[test]
    fn a_pending_stamp_wait_reports_what_its_channel_has_published() {
        let mut owner = SynchronizationConditionOwner::default();
        let channel = ChannelId::new(3);
        let wait = |value| ExplicitWaitCause::Stamp {
            source_channel: channel,
            value,
        };

        // Nothing signalled: pending, with nothing published to compare to.
        assert_eq!(
            owner.classify_wait(wait(9)),
            ConditionWaitResolution::Pending
        );
        assert_eq!(owner.published_signal(wait(9)), None);

        let transaction = TransactionId::new(1);
        owner
            .register_signal(
                transaction,
                IngressOrdinal::new(1),
                ConditionSignal::Stamp { channel, value: 8 },
            )
            .unwrap();
        owner
            .publish_transaction_at(transaction, ConditionPublicationBoundary::GuestPublication)
            .unwrap();

        // One short: still pending, and now the reading says by how much.
        assert_eq!(
            owner.classify_wait(wait(9)),
            ConditionWaitResolution::Pending
        );
        assert_eq!(
            owner.published_signal(wait(9)),
            Some(ConditionSignal::Stamp { channel, value: 8 })
        );

        // A different channel is a different condition and answers for itself.
        assert_eq!(
            owner.published_signal(ExplicitWaitCause::Stamp {
                source_channel: ChannelId::new(4),
                value: 8,
            }),
            None
        );
    }
}
