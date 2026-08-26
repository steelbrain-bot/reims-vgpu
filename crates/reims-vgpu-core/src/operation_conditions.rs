//! Derive transaction conditions from ordered event and fence operations.
//!
//! Event and fence records remain in the immutable operation stream so their
//! encoder position is auditable. Transaction admission consumes this one
//! projection: waits not already satisfied by an earlier operation in the same
//! EXEC become packet prerequisites, while only the final value produced for
//! each condition is published at GPU completion.

use crate::{
    ConditionSignal, EventOperation, EventOperationKind, ExecTransaction, FenceOperation,
    FenceOperationKind, ResolvedOperation, TransactionPrerequisite,
};
use reims_vgpu_protocol::{EventObject, FenceObject, ResourceId, TransactionId};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecConditionPlan {
    prerequisites: Box<[TransactionPrerequisite]>,
    signals: Box<[ConditionSignal]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmittedConditionOperation {
    Event(EventOperation),
    Fence(FenceOperation),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedExecConditions {
    transaction: TransactionId,
    operations: Box<[(usize, AdmittedConditionOperation)]>,
}

impl AdmittedExecConditions {
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    pub fn operations(&self) -> &[(usize, AdmittedConditionOperation)] {
        &self.operations
    }

    pub fn for_operation_range(&self, start: usize, end: usize) -> Self {
        Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .filter(|(index, _)| *index >= start && *index < end)
                .copied()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub fn shifted(&self, base: usize) -> Option<Self> {
        Some(Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .map(|(index, operation)| Some((index.checked_add(base)?, *operation)))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }

    pub fn remapped_positions(&self, positions: &[usize]) -> Option<Self> {
        Some(Self {
            transaction: self.transaction,
            operations: self
                .operations
                .iter()
                .map(|(index, operation)| Some((*positions.get(*index)?, *operation)))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        })
    }

    pub(crate) fn from_exec<Render, Compute, Info, Indirect, Completion>(
        transaction: TransactionId,
        exec: &ExecTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Completion>>,
    ) -> Self {
        let operations = exec
            .operations()
            .enumerate()
            .filter_map(|(index, operation)| match operation {
                ResolvedOperation::Event(event) => {
                    Some((index, AdmittedConditionOperation::Event(*event)))
                }
                ResolvedOperation::Fence(fence) => {
                    Some((index, AdmittedConditionOperation::Fence(*fence)))
                }
                ResolvedOperation::EncoderBoundary(_)
                | ResolvedOperation::Render(_)
                | ResolvedOperation::Compute(_)
                | ResolvedOperation::Blit(_)
                | ResolvedOperation::Barrier(_)
                | ResolvedOperation::Participation(_)
                | ResolvedOperation::ResourceState(_)
                | ResolvedOperation::InfoQuery(_)
                | ResolvedOperation::IndirectCommand(_)
                | ResolvedOperation::CompletionEffect(_) => None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            transaction,
            operations,
        }
    }
}

impl ExecConditionPlan {
    pub fn prerequisites(&self) -> &[TransactionPrerequisite] {
        &self.prerequisites
    }

    pub fn signals(&self) -> &[ConditionSignal] {
        &self.signals
    }

    pub fn into_parts(self) -> (Box<[TransactionPrerequisite]>, Box<[ConditionSignal]>) {
        (self.prerequisites, self.signals)
    }
}

impl<Render, Compute, Info, Indirect, Completion>
    ExecTransaction<ResolvedOperation<Render, Compute, Info, Indirect, Completion>>
{
    /// Project the exact ordered event/fence condition effects of this EXEC.
    ///
    /// A signal followed by a satisfied wait in the same stream needs no
    /// inter-transaction edge. A wait that precedes its producer remains an
    /// explicit prerequisite; admission may bind it to an earlier or future
    /// transaction and diagnose a cycle. Signals publish only at the EXEC's
    /// GPU-completion boundary, so multiple updates of one condition collapse
    /// to the greatest value without changing guest visibility.
    pub fn condition_plan(&self) -> ExecConditionPlan {
        let mut event_values = BTreeMap::<ResourceId<EventObject>, u64>::new();
        let mut fence_generations = BTreeMap::<ResourceId<FenceObject>, u64>::new();
        let mut prerequisites = Vec::new();

        for operation in self.operations() {
            match operation {
                ResolvedOperation::Event(event) => match event.kind {
                    EventOperationKind::Signal => {
                        event_values
                            .entry(event.event)
                            .and_modify(|value| *value = (*value).max(event.value))
                            .or_insert(event.value);
                    }
                    EventOperationKind::Wait => {
                        if event_values
                            .get(&event.event)
                            .is_none_or(|value| *value < event.value)
                        {
                            push_unique(
                                &mut prerequisites,
                                TransactionPrerequisite::Event {
                                    event: event.event,
                                    value: event.value,
                                },
                            );
                        }
                    }
                },
                ResolvedOperation::Fence(fence) => match fence.kind {
                    FenceOperationKind::Update => {
                        fence_generations
                            .entry(fence.fence)
                            .and_modify(|generation| {
                                *generation = (*generation).max(fence.generation)
                            })
                            .or_insert(fence.generation);
                    }
                    FenceOperationKind::Wait => {
                        if fence_generations
                            .get(&fence.fence)
                            .is_none_or(|generation| *generation < fence.generation)
                        {
                            push_unique(
                                &mut prerequisites,
                                TransactionPrerequisite::Fence {
                                    fence: fence.fence,
                                    generation: fence.generation,
                                },
                            );
                        }
                    }
                },
                ResolvedOperation::EncoderBoundary(_)
                | ResolvedOperation::Render(_)
                | ResolvedOperation::Compute(_)
                | ResolvedOperation::Blit(_)
                | ResolvedOperation::Barrier(_)
                | ResolvedOperation::Participation(_)
                | ResolvedOperation::ResourceState(_)
                | ResolvedOperation::InfoQuery(_)
                | ResolvedOperation::IndirectCommand(_)
                | ResolvedOperation::CompletionEffect(_) => {}
            }
        }

        let signals = event_values
            .into_iter()
            .map(|(event, value)| ConditionSignal::Event { event, value })
            .chain(
                fence_generations
                    .into_iter()
                    .map(|(fence, generation)| ConditionSignal::Fence { fence, generation }),
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ExecConditionPlan {
            prerequisites: prerequisites.into_boxed_slice(),
            signals,
        }
    }
}

fn push_unique(values: &mut Vec<TransactionPrerequisite>, value: TransactionPrerequisite) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventOperation, FenceOperation, ResolvedExecSegment, ResolvedExecStream};
    use reims_vgpu_protocol::{
        ChannelId, SegmentBoundary, SegmentKind, SessionGenerationId, SubmissionId,
        SubmissionIdentity, TaskId,
    };

    type Operation = ResolvedOperation<(), (), (), (), ()>;

    fn exec(operations: impl Into<Box<[Operation]>>) -> ExecTransaction<Operation> {
        ExecTransaction {
            identity: SubmissionIdentity {
                id: SubmissionId::new(1),
                task: TaskId::new(1),
            },
            prologue: crate::ExecPrologue::default(),
            streams: Box::new([ResolvedExecStream {
                stream_index: 0,
                segments: Box::new([ResolvedExecSegment {
                    boundary: SegmentBoundary {
                        stream_index: 0,
                        kind: SegmentKind::Event,
                        index: 0,
                        continues_previous: false,
                        continues_next: false,
                    },
                    operations: operations.into(),
                }]),
            }]),
            accesses: Box::new([]),
        }
    }

    #[test]
    fn prior_local_signals_satisfy_waits_and_only_final_values_publish() {
        let event = ResourceId::new(4, 1);
        let fence = ResourceId::new(4, 1);
        let plan = exec(vec![
            Operation::Event(EventOperation {
                event,
                kind: EventOperationKind::Signal,
                value: 7,
            }),
            Operation::Event(EventOperation {
                event,
                kind: EventOperationKind::Wait,
                value: 7,
            }),
            Operation::Fence(FenceOperation {
                fence,
                kind: FenceOperationKind::Update,
                generation: 1,
                scope: crate::FenceScope::Compute,
            }),
            Operation::Fence(FenceOperation {
                fence,
                kind: FenceOperationKind::Wait,
                generation: 1,
                scope: crate::FenceScope::Compute,
            }),
            Operation::Event(EventOperation {
                event,
                kind: EventOperationKind::Signal,
                value: 9,
            }),
            Operation::Fence(FenceOperation {
                fence,
                kind: FenceOperationKind::Update,
                generation: 2,
                scope: crate::FenceScope::Compute,
            }),
        ])
        .condition_plan();

        assert!(plan.prerequisites().is_empty());
        assert_eq!(
            plan.signals(),
            [
                ConditionSignal::Event { event, value: 9 },
                ConditionSignal::Fence {
                    fence,
                    generation: 2,
                },
            ]
        );
    }

    #[test]
    fn unsatisfied_waits_remain_exact_deduplicated_packet_prerequisites() {
        let event = ResourceId::new(8, 2);
        let fence = ResourceId::new(8, 2);
        let plan = exec(vec![
            Operation::Event(EventOperation {
                event,
                kind: EventOperationKind::Wait,
                value: 11,
            }),
            Operation::Event(EventOperation {
                event,
                kind: EventOperationKind::Wait,
                value: 11,
            }),
            Operation::Fence(FenceOperation {
                fence,
                kind: FenceOperationKind::Wait,
                generation: 3,
                scope: crate::FenceScope::Compute,
            }),
            Operation::Event(EventOperation {
                event,
                kind: EventOperationKind::Signal,
                value: 12,
            }),
            Operation::Fence(FenceOperation {
                fence,
                kind: FenceOperationKind::Update,
                generation: 3,
                scope: crate::FenceScope::Compute,
            }),
        ])
        .condition_plan();

        assert_eq!(
            plan.prerequisites(),
            [
                TransactionPrerequisite::Event { event, value: 11 },
                TransactionPrerequisite::Fence {
                    fence,
                    generation: 3,
                },
            ]
        );
        assert_eq!(
            plan.signals(),
            [
                ConditionSignal::Event { event, value: 12 },
                ConditionSignal::Fence {
                    fence,
                    generation: 3,
                },
            ]
        );
    }

    #[test]
    fn runtime_admission_derives_future_event_edges_from_the_exec_once() {
        let mut runtime = crate::TransactionRuntime::<&'static str>::new(
            crate::SessionGeneration::new(SessionGenerationId::new(1)),
        );
        let waiter_channel = ChannelId::new(3);
        let producer_channel = ChannelId::new(4);
        runtime.define_channel(waiter_channel).unwrap();
        runtime.define_channel(producer_channel).unwrap();
        let event = ResourceId::new(5, 2);
        let waiter = runtime
            .admit_exec_operations(
                waiter_channel,
                Box::<[crate::ResolvedTransactionPrerequisite]>::default(),
                Some(crate::CompletionStamp::new(waiter_channel.get(), 1)),
                exec(vec![Operation::Event(EventOperation {
                    event,
                    kind: EventOperationKind::Wait,
                    value: 7,
                })]),
            )
            .unwrap();
        let waiter_id = waiter.transaction.id;
        assert_eq!(waiter.conditions().transaction(), waiter_id);
        assert_eq!(
            waiter.conditions().operations(),
            [(
                0,
                AdmittedConditionOperation::Event(EventOperation {
                    event,
                    kind: EventOperationKind::Wait,
                    value: 7,
                })
            )]
        );
        runtime.recorded(waiter_id).unwrap();
        assert!(runtime.take_submission_ready().is_empty());

        let producer = runtime
            .admit_exec_operations(
                producer_channel,
                Box::<[crate::ResolvedTransactionPrerequisite]>::default(),
                Some(crate::CompletionStamp::new(producer_channel.get(), 1)),
                exec(vec![Operation::Event(EventOperation {
                    event,
                    kind: EventOperationKind::Signal,
                    value: 7,
                })]),
            )
            .unwrap();
        let producer_id = producer.transaction.id;
        runtime.recorded(producer_id).unwrap();
        assert_eq!(
            runtime.submission_dependencies(waiter_id).unwrap(),
            [(
                producer_id,
                crate::WaitDependencyCause::Explicit(crate::ExplicitWaitCause::Event {
                    event,
                    value: 7,
                }),
            )]
        );
        let ready = runtime
            .take_submission_ready()
            .into_iter()
            .map(|item| item.transaction)
            .collect::<Vec<_>>();
        assert_eq!(ready, [waiter_id, producer_id]);
    }

    #[test]
    fn runtime_admission_refuses_a_second_event_prerequisite_spelling() {
        let mut runtime = crate::TransactionRuntime::<&'static str>::new(
            crate::SessionGeneration::new(SessionGenerationId::new(1)),
        );
        let channel = ChannelId::new(6);
        runtime.define_channel(channel).unwrap();
        let event = ResourceId::new(9, 3);
        let result = runtime.admit_exec_operations(
            channel,
            [crate::ResolvedTransactionPrerequisite {
                prerequisite: TransactionPrerequisite::Event { event, value: 4 },
                resolution: crate::PrerequisiteResolution::Pending,
            }],
            Some(crate::CompletionStamp::new(channel.get(), 1)),
            exec(Vec::<Operation>::new()),
        );
        assert_eq!(
            result.unwrap_err(),
            crate::TransactionRuntimeError::NonStampEnvelopePrerequisite(Box::new(
                TransactionPrerequisite::Event { event, value: 4 }
            ))
        );

        let accepted = runtime
            .admit_exec_operations(
                channel,
                Box::<[crate::ResolvedTransactionPrerequisite]>::default(),
                Some(crate::CompletionStamp::new(channel.get(), 1)),
                exec(Vec::<Operation>::new()),
            )
            .unwrap();
        assert_eq!(
            accepted.transaction.id,
            reims_vgpu_protocol::TransactionId::new(1)
        );
    }
}
