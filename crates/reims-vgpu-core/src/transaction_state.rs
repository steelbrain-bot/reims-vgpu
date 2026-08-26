//! Structural transaction phases and reset/device-loss separation.
//!
//! Reset disposition is reported without guessing the still-open guest result.
//! In particular, submitted work remains submitted and retains its native
//! leases; a reset request cannot turn it into successful completion or revoke
//! Vulkan lifetime safety.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionState {
    Accepted,
    Decoding,
    Resolved,
    Planned,
    Recording,
    Recorded,
    Queued,
    Submitted,
    GpuComplete,
    Executing,
    EffectComplete,
    AcquirePending,
    PresentReady,
    PresentQueued,
    PresentComplete,
    SemanticCommitted,
    GuestPublished,
    Retired,
    Refused,
    CancelledByReset,
    FailedByDeviceLoss,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransactionTransitionError {
    pub from: TransactionState,
    pub to: TransactionState,
}

/// Internal obligation when guest reset closes the current semantic generation.
/// The value deliberately does not choose a guest completion/error result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResetDisposition {
    /// Accepted work has no native effect and awaits the established reset
    /// publication rule.
    QuiesceUnsubmitted,
    /// A command buffer was recorded but never submitted and can be discarded.
    DiscardRecording,
    /// Queue admission must remove this transaction before it reaches Vulkan.
    RemoveFromQueue,
    /// A CPU/control effect is already executing and must reach an immutable
    /// effect fact before reset publication is decided.
    FinishInProgressEffect,
    /// Issued native work and both lifetime leases remain retained.
    RetainSubmittedWork,
    /// Preserve an immutable native/CPU/present completion fact.
    PreserveCompletionFact,
    /// Semantic state has advanced; only its publication obligation remains.
    PreservePublication,
    /// Guest publication happened; retire after remaining native obligations.
    RetireWhenSafe,
    /// No reset action remains for an already terminal or retired transaction.
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransactionStateMachine {
    state: TransactionState,
}

impl Default for TransactionStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionStateMachine {
    pub const fn new() -> Self {
        Self {
            state: TransactionState::Accepted,
        }
    }

    pub const fn state(self) -> TransactionState {
        self.state
    }

    pub fn advance(&mut self, next: TransactionState) -> Result<(), TransactionTransitionError> {
        if !normal_transition(self.state, next) {
            return Err(TransactionTransitionError {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }

    /// Apply a contract-established refusal before the first native effect.
    pub fn refuse(&mut self) -> Result<(), TransactionTransitionError> {
        let next = TransactionState::Refused;
        if !matches!(
            self.state,
            TransactionState::Accepted
                | TransactionState::Decoding
                | TransactionState::Resolved
                | TransactionState::Planned
        ) {
            return Err(TransactionTransitionError {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }

    /// Apply cancellation only after the reset contract selected it for this
    /// unsubmitted state. Merely observing reset must use [`Self::on_reset`].
    pub fn apply_established_reset_cancellation(
        &mut self,
    ) -> Result<(), TransactionTransitionError> {
        let next = TransactionState::CancelledByReset;
        if !matches!(
            self.on_reset(),
            ResetDisposition::QuiesceUnsubmitted
                | ResetDisposition::DiscardRecording
                | ResetDisposition::RemoveFromQueue
        ) {
            return Err(TransactionTransitionError {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }

    /// Record loss of the Vulkan epoch. The caller must already have
    /// established that this transaction depends on that epoch.
    pub fn fail_by_device_loss(&mut self) -> Result<(), TransactionTransitionError> {
        let next = TransactionState::FailedByDeviceLoss;
        if matches!(
            self.state,
            TransactionState::GuestPublished
                | TransactionState::Retired
                | TransactionState::Refused
                | TransactionState::CancelledByReset
                | TransactionState::FailedByDeviceLoss
        ) {
            return Err(TransactionTransitionError {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }

    pub const fn on_reset(self) -> ResetDisposition {
        match self.state {
            TransactionState::Accepted
            | TransactionState::Decoding
            | TransactionState::Resolved
            | TransactionState::Planned => ResetDisposition::QuiesceUnsubmitted,
            TransactionState::Recording | TransactionState::Recorded => {
                ResetDisposition::DiscardRecording
            }
            TransactionState::Queued => ResetDisposition::RemoveFromQueue,
            TransactionState::Submitted | TransactionState::PresentQueued => {
                ResetDisposition::RetainSubmittedWork
            }
            TransactionState::GpuComplete
            | TransactionState::EffectComplete
            | TransactionState::PresentComplete => ResetDisposition::PreserveCompletionFact,
            TransactionState::SemanticCommitted => ResetDisposition::PreservePublication,
            TransactionState::GuestPublished => ResetDisposition::RetireWhenSafe,
            TransactionState::Executing => ResetDisposition::FinishInProgressEffect,
            TransactionState::AcquirePending | TransactionState::PresentReady => {
                ResetDisposition::QuiesceUnsubmitted
            }
            TransactionState::Retired
            | TransactionState::Refused
            | TransactionState::CancelledByReset
            | TransactionState::FailedByDeviceLoss => ResetDisposition::Terminal,
        }
    }
}

const fn normal_transition(from: TransactionState, to: TransactionState) -> bool {
    matches!(
        (from, to),
        (TransactionState::Accepted, TransactionState::Decoding)
            | (TransactionState::Decoding, TransactionState::Resolved)
            | (TransactionState::Resolved, TransactionState::Planned)
            | (TransactionState::Planned, TransactionState::Recording)
            | (TransactionState::Recording, TransactionState::Recorded)
            | (TransactionState::Recorded, TransactionState::Queued)
            | (TransactionState::Queued, TransactionState::Submitted)
            | (TransactionState::Submitted, TransactionState::GpuComplete)
            | (
                TransactionState::GpuComplete,
                TransactionState::SemanticCommitted
            )
            | (TransactionState::Planned, TransactionState::Executing)
            | (
                TransactionState::Executing,
                TransactionState::EffectComplete
            )
            | (
                TransactionState::EffectComplete,
                TransactionState::SemanticCommitted
            )
            | (TransactionState::Planned, TransactionState::AcquirePending)
            | (
                TransactionState::AcquirePending,
                TransactionState::PresentReady
            )
            | (
                TransactionState::PresentReady,
                TransactionState::PresentQueued
            )
            | (
                TransactionState::PresentQueued,
                TransactionState::PresentComplete
            )
            | (
                TransactionState::PresentComplete,
                TransactionState::SemanticCommitted
            )
            | (
                TransactionState::SemanticCommitted,
                TransactionState::GuestPublished
            )
            | (TransactionState::GuestPublished, TransactionState::Retired)
            | (TransactionState::Refused, TransactionState::GuestPublished)
            | (
                TransactionState::CancelledByReset,
                TransactionState::GuestPublished
            )
            | (
                TransactionState::FailedByDeviceLoss,
                TransactionState::GuestPublished
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reach(states: &[TransactionState]) -> TransactionStateMachine {
        let mut machine = TransactionStateMachine::new();
        for state in states {
            machine.advance(*state).unwrap();
        }
        machine
    }

    #[test]
    fn gpu_cpu_and_present_branches_commit_only_after_their_own_completion_fact() {
        for path in [
            vec![
                TransactionState::Decoding,
                TransactionState::Resolved,
                TransactionState::Planned,
                TransactionState::Recording,
                TransactionState::Recorded,
                TransactionState::Queued,
                TransactionState::Submitted,
                TransactionState::GpuComplete,
                TransactionState::SemanticCommitted,
            ],
            vec![
                TransactionState::Decoding,
                TransactionState::Resolved,
                TransactionState::Planned,
                TransactionState::Executing,
                TransactionState::EffectComplete,
                TransactionState::SemanticCommitted,
            ],
            vec![
                TransactionState::Decoding,
                TransactionState::Resolved,
                TransactionState::Planned,
                TransactionState::AcquirePending,
                TransactionState::PresentReady,
                TransactionState::PresentQueued,
                TransactionState::PresentComplete,
                TransactionState::SemanticCommitted,
            ],
        ] {
            assert_eq!(reach(&path).state(), TransactionState::SemanticCommitted);
        }
    }

    #[test]
    fn recording_is_not_a_semantic_completion_fact() {
        let mut machine = reach(&[
            TransactionState::Decoding,
            TransactionState::Resolved,
            TransactionState::Planned,
            TransactionState::Recording,
            TransactionState::Recorded,
        ]);
        assert_eq!(
            machine.advance(TransactionState::SemanticCommitted),
            Err(TransactionTransitionError {
                from: TransactionState::Recorded,
                to: TransactionState::SemanticCommitted,
            })
        );
    }

    #[test]
    fn reset_disposition_is_defined_for_every_state_without_guessing_a_result() {
        use ResetDisposition as D;
        use TransactionState as S;
        const CASES: &[(S, D)] = &[
            (S::Accepted, D::QuiesceUnsubmitted),
            (S::Decoding, D::QuiesceUnsubmitted),
            (S::Resolved, D::QuiesceUnsubmitted),
            (S::Planned, D::QuiesceUnsubmitted),
            (S::Recording, D::DiscardRecording),
            (S::Recorded, D::DiscardRecording),
            (S::Queued, D::RemoveFromQueue),
            (S::Submitted, D::RetainSubmittedWork),
            (S::GpuComplete, D::PreserveCompletionFact),
            (S::Executing, D::FinishInProgressEffect),
            (S::EffectComplete, D::PreserveCompletionFact),
            (S::AcquirePending, D::QuiesceUnsubmitted),
            (S::PresentReady, D::QuiesceUnsubmitted),
            (S::PresentQueued, D::RetainSubmittedWork),
            (S::PresentComplete, D::PreserveCompletionFact),
            (S::SemanticCommitted, D::PreservePublication),
            (S::GuestPublished, D::RetireWhenSafe),
            (S::Retired, D::Terminal),
            (S::Refused, D::Terminal),
            (S::CancelledByReset, D::Terminal),
            (S::FailedByDeviceLoss, D::Terminal),
        ];
        for &(state, expected) in CASES {
            assert_eq!(
                TransactionStateMachine { state }.on_reset(),
                expected,
                "{state:?}"
            );
        }
    }

    #[test]
    fn reset_cannot_revoke_submitted_work() {
        let mut machine = reach(&[
            TransactionState::Decoding,
            TransactionState::Resolved,
            TransactionState::Planned,
            TransactionState::Recording,
            TransactionState::Recorded,
            TransactionState::Queued,
            TransactionState::Submitted,
        ]);
        assert_eq!(machine.on_reset(), ResetDisposition::RetainSubmittedWork);
        assert!(machine.apply_established_reset_cancellation().is_err());
        assert_eq!(machine.state(), TransactionState::Submitted);
    }

    #[test]
    fn device_loss_and_reset_are_distinct_terminal_paths() {
        let mut reset = reach(&[
            TransactionState::Decoding,
            TransactionState::Resolved,
            TransactionState::Planned,
            TransactionState::Recording,
        ]);
        reset.apply_established_reset_cancellation().unwrap();
        let mut loss = reach(&[
            TransactionState::Decoding,
            TransactionState::Resolved,
            TransactionState::Planned,
            TransactionState::Recording,
        ]);
        loss.fail_by_device_loss().unwrap();
        assert_eq!(reset.state(), TransactionState::CancelledByReset);
        assert_eq!(loss.state(), TransactionState::FailedByDeviceLoss);
    }
}
