//! One externally synchronized owner thread for a replacement Vulkan queue.
//!
//! Enqueue transfers all submission arrays to the owner. The returned receipt
//! reports only whether `vkQueueSubmit` accepted the work; GPU completion is
//! observed separately through the queue timeline and is never inferred from
//! this driver return.

use crate::replacement_recording::ReplacementNativeRecording;
use crate::replacement_submit::{
    QueueTimelineSemaphores, TimelineSubmitPlan, TimelineSubmitPlanError,
};
use ash::vk;
use reims_vgpu_core::{
    PreparedAuxiliaryNativeSubmission, PreparedNativeSubmission, PreparedPresentNativeSubmission,
};
use reims_vgpu_protocol::QueueOwnerId;
use std::sync::{mpsc, Arc, Mutex};

#[derive(Clone, Debug)]
struct ReplacementQueueSubmission {
    pub plan: TimelineSubmitPlan,
    pub recording: ReplacementQueueRecording,
    #[cfg(feature = "host-window")]
    pub present: Option<ReplacementQueuePresent>,
    #[cfg(feature = "host-window")]
    pub present_result: Option<Result<bool, vk::Result>>,
}

#[derive(Clone, Debug)]
enum ReplacementQueueRecording {
    Execution(ReplacementNativeRecording),
    Present(ReplacementPresentRecording),
}

impl ReplacementQueueRecording {
    fn execution(&self) -> &ReplacementNativeRecording {
        match self {
            Self::Execution(recording) => recording,
            Self::Present(_) => {
                unreachable!("an execution queue owner cannot carry a Present recording")
            }
        }
    }

    fn command_buffers(&self) -> &[vk::CommandBuffer] {
        match self {
            Self::Execution(recording) => &recording.command_buffers,
            Self::Present(recording) => std::slice::from_ref(&recording.command_buffer),
        }
    }

    const fn fence(&self) -> vk::Fence {
        match self {
            Self::Execution(recording) => recording.fence,
            Self::Present(recording) => recording.fence,
        }
    }

    fn into_execution(self) -> ReplacementNativeRecording {
        match self {
            Self::Execution(recording) => recording,
            Self::Present(_) => {
                unreachable!("an execution queue owner cannot return Present recording ownership")
            }
        }
    }

    fn into_present(self) -> ReplacementPresentRecording {
        match self {
            Self::Present(recording) => recording,
            Self::Execution(_) => {
                unreachable!("a Present queue owner cannot return EXEC recording ownership")
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReplacementPresentRecording {
    pub(crate) slot: usize,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) fence: vk::Fence,
}

impl ReplacementPresentRecording {
    pub const fn slot(&self) -> usize {
        self.slot
    }

    pub const fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

#[cfg(feature = "host-window")]
#[derive(Clone)]
pub struct ReplacementQueuePresent {
    pub(crate) loader: ash::khr::swapchain::Device,
    pub(crate) acquire_wait: vk::Semaphore,
    pub(crate) render_finished: vk::Semaphore,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) image_index: u32,
}

#[cfg(feature = "host-window")]
impl std::fmt::Debug for ReplacementQueuePresent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplacementQueuePresent")
            .field("acquire_wait", &self.acquire_wait)
            .field("render_finished", &self.render_finished)
            .field("swapchain", &self.swapchain)
            .field("image_index", &self.image_index)
            .finish_non_exhaustive()
    }
}

impl ReplacementQueueSubmission {
    fn new(plan: TimelineSubmitPlan, recording: ReplacementNativeRecording) -> Self {
        Self {
            plan,
            recording: ReplacementQueueRecording::Execution(recording),
            #[cfg(feature = "host-window")]
            present: None,
            #[cfg(feature = "host-window")]
            present_result: None,
        }
    }

    fn new_present(plan: TimelineSubmitPlan, recording: ReplacementPresentRecording) -> Self {
        Self {
            plan,
            recording: ReplacementQueueRecording::Present(recording),
            #[cfg(feature = "host-window")]
            present: None,
            #[cfg(feature = "host-window")]
            present_result: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementQueueError {
    WrongQueue,
    SignalValueOutOfOrder,
    OwnerStopped,
    Driver(vk::Result),
}

impl ReplacementQueueError {
    /// Whether this refusal is the Vulkan incarnation's terminal loss result.
    pub const fn is_device_lost(self) -> bool {
        matches!(self, Self::Driver(vk::Result::ERROR_DEVICE_LOST))
    }
}

#[derive(Clone, Debug)]
struct ReplacementQueueEnqueueFailure {
    pub reason: ReplacementQueueError,
    pub submission: ReplacementQueueSubmission,
}

#[derive(Debug)]
#[must_use = "a prepared queue submission must be enqueued or recovered from a typed refusal"]
pub struct PreparedReplacementQueueSubmission<Semantic> {
    pub prepared: PreparedNativeSubmission<Semantic>,
    native: ReplacementQueueSubmission,
}

#[derive(Debug)]
pub struct ReplacementQueuePreparationFailure<Semantic> {
    pub reason: TimelineSubmitPlanError,
    pub prepared: PreparedNativeSubmission<Semantic>,
    pub recording: ReplacementNativeRecording,
}

#[derive(Debug)]
#[must_use = "an auxiliary queue submission must be enqueued or recovered"]
pub struct PreparedReplacementAuxiliaryQueueSubmission {
    pub prepared: PreparedAuxiliaryNativeSubmission,
    native: ReplacementQueueSubmission,
}

#[derive(Debug)]
pub struct ReplacementAuxiliaryQueuePreparationFailure {
    pub reason: TimelineSubmitPlanError,
    pub prepared: PreparedAuxiliaryNativeSubmission,
    pub recording: ReplacementNativeRecording,
}

#[derive(Debug)]
#[must_use = "a prepared presentation blit must be enqueued or recovered"]
pub struct PreparedReplacementPresentQueueSubmission {
    pub prepared: PreparedPresentNativeSubmission,
    native: ReplacementQueueSubmission,
}

#[derive(Debug)]
pub struct ReplacementPresentQueuePreparationFailure {
    pub reason: TimelineSubmitPlanError,
    pub prepared: PreparedPresentNativeSubmission,
    pub recording: ReplacementPresentRecording,
}

impl PreparedReplacementPresentQueueSubmission {
    pub fn new(
        prepared: PreparedPresentNativeSubmission,
        timelines: &QueueTimelineSemaphores,
        recording: ReplacementPresentRecording,
        waits: impl Into<Box<[reims_vgpu_core::QueueTimelinePoint]>>,
    ) -> Result<Self, Box<ReplacementPresentQueuePreparationFailure>> {
        let plan = match timelines.plan_present(&prepared, waits) {
            Ok(plan) => plan,
            Err(reason) => {
                return Err(Box::new(ReplacementPresentQueuePreparationFailure {
                    reason,
                    prepared,
                    recording,
                }));
            }
        };
        Ok(Self {
            prepared,
            native: ReplacementQueueSubmission::new_present(plan, recording),
        })
    }

    pub const fn point(&self) -> reims_vgpu_core::QueueTimelinePoint {
        self.prepared.point()
    }

    pub fn into_parts(self) -> (PreparedPresentNativeSubmission, ReplacementPresentRecording) {
        (self.prepared, self.native.recording.into_present())
    }

    #[cfg(feature = "host-window")]
    pub fn with_present(mut self, present: ReplacementQueuePresent) -> Self {
        self.native.present = Some(present);
        self
    }
}

#[derive(Debug)]
pub struct PreparedReplacementPresentQueueEnqueueFailure {
    pub reason: ReplacementQueueError,
    pub submission: PreparedReplacementPresentQueueSubmission,
}

#[derive(Debug)]
pub struct DriverAcceptedReplacementPresentQueueSubmission {
    prepared: PreparedPresentNativeSubmission,
    recording: ReplacementPresentRecording,
    #[cfg(feature = "host-window")]
    present_result: Option<Result<bool, vk::Result>>,
}

impl DriverAcceptedReplacementPresentQueueSubmission {
    pub const fn prepared(&self) -> &PreparedPresentNativeSubmission {
        &self.prepared
    }

    pub fn into_parts(self) -> (PreparedPresentNativeSubmission, ReplacementPresentRecording) {
        (self.prepared, self.recording)
    }

    pub const fn recording(&self) -> &ReplacementPresentRecording {
        &self.recording
    }

    #[cfg(feature = "host-window")]
    pub const fn present_result(&self) -> Option<Result<bool, vk::Result>> {
        self.present_result
    }
}

#[must_use = "presentation driver acceptance must be observed"]
pub struct PendingPreparedReplacementPresentQueueSubmit {
    pending: PendingReplacementQueueSubmit,
    prepared: PreparedPresentNativeSubmission,
}

pub enum PreparedReplacementPresentQueueSubmitPoll {
    Pending(PendingPreparedReplacementPresentQueueSubmit),
    DriverAccepted(DriverAcceptedReplacementPresentQueueSubmission),
    DriverRefused {
        reason: ReplacementQueueError,
        submission: PreparedReplacementPresentQueueSubmission,
    },
}

impl PendingPreparedReplacementPresentQueueSubmit {
    pub fn try_complete(mut self) -> PreparedReplacementPresentQueueSubmitPoll {
        match self.pending.try_complete() {
            None => PreparedReplacementPresentQueueSubmitPoll::Pending(self),
            Some(Ok(native)) => PreparedReplacementPresentQueueSubmitPoll::DriverAccepted(
                DriverAcceptedReplacementPresentQueueSubmission {
                    prepared: self.prepared,
                    recording: native.recording.into_present(),
                    #[cfg(feature = "host-window")]
                    present_result: native.present_result,
                },
            ),
            Some(Err(failure)) => {
                let (reason, native) = *failure;
                PreparedReplacementPresentQueueSubmitPoll::DriverRefused {
                    reason,
                    submission: PreparedReplacementPresentQueueSubmission {
                        prepared: self.prepared,
                        native,
                    },
                }
            }
        }
    }

    pub fn wait(
        self,
    ) -> Result<
        DriverAcceptedReplacementPresentQueueSubmission,
        Box<(
            ReplacementQueueError,
            PreparedReplacementPresentQueueSubmission,
        )>,
    > {
        match self.pending.wait() {
            Ok(native) => Ok(DriverAcceptedReplacementPresentQueueSubmission {
                prepared: self.prepared,
                recording: native.recording.into_present(),
                #[cfg(feature = "host-window")]
                present_result: native.present_result,
            }),
            Err(failure) => {
                let (reason, native) = *failure;
                Err(Box::new((
                    reason,
                    PreparedReplacementPresentQueueSubmission {
                        prepared: self.prepared,
                        native,
                    },
                )))
            }
        }
    }
}

impl PreparedReplacementAuxiliaryQueueSubmission {
    pub fn new(
        prepared: PreparedAuxiliaryNativeSubmission,
        timelines: &QueueTimelineSemaphores,
        recording: ReplacementNativeRecording,
    ) -> Result<Self, Box<ReplacementAuxiliaryQueuePreparationFailure>> {
        Self::new_with_auxiliary_waits(
            prepared,
            timelines,
            recording,
            Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
        )
    }

    pub fn new_with_auxiliary_waits(
        prepared: PreparedAuxiliaryNativeSubmission,
        timelines: &QueueTimelineSemaphores,
        recording: ReplacementNativeRecording,
        auxiliary_waits: impl Into<Box<[reims_vgpu_core::QueueTimelinePoint]>>,
    ) -> Result<Self, Box<ReplacementAuxiliaryQueuePreparationFailure>> {
        let plan = match timelines.plan_auxiliary_with_waits(&prepared, auxiliary_waits) {
            Ok(plan) => plan,
            Err(reason) => {
                return Err(Box::new(ReplacementAuxiliaryQueuePreparationFailure {
                    reason,
                    prepared,
                    recording,
                }));
            }
        };
        Ok(Self {
            prepared,
            native: ReplacementQueueSubmission::new(plan, recording),
        })
    }

    pub fn recording(&self) -> &ReplacementNativeRecording {
        self.native.recording.execution()
    }

    pub const fn auxiliary_waits(&self) -> &[reims_vgpu_core::QueueTimelinePoint] {
        &self.native.plan.auxiliary_waits
    }

    pub fn into_parts(
        self,
    ) -> (
        PreparedAuxiliaryNativeSubmission,
        ReplacementNativeRecording,
    ) {
        (self.prepared, self.native.recording.into_execution())
    }
}

impl<Semantic> PreparedReplacementQueueSubmission<Semantic> {
    pub fn recording(&self) -> &ReplacementNativeRecording {
        self.native.recording.execution()
    }

    pub const fn auxiliary_waits(&self) -> &[reims_vgpu_core::QueueTimelinePoint] {
        &self.native.plan.auxiliary_waits
    }

    pub fn new(
        prepared: PreparedNativeSubmission<Semantic>,
        timelines: &QueueTimelineSemaphores,
        recording: ReplacementNativeRecording,
    ) -> Result<Self, Box<ReplacementQueuePreparationFailure<Semantic>>> {
        Self::new_with_auxiliary_waits(
            prepared,
            timelines,
            recording,
            Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
        )
    }

    pub fn new_with_auxiliary_waits(
        prepared: PreparedNativeSubmission<Semantic>,
        timelines: &QueueTimelineSemaphores,
        recording: ReplacementNativeRecording,
        auxiliary_waits: impl Into<Box<[reims_vgpu_core::QueueTimelinePoint]>>,
    ) -> Result<Self, Box<ReplacementQueuePreparationFailure<Semantic>>> {
        let plan = match timelines.plan_with_auxiliary_waits(&prepared, auxiliary_waits) {
            Ok(plan) => plan,
            Err(reason) => {
                return Err(Box::new(ReplacementQueuePreparationFailure {
                    reason,
                    prepared,
                    recording,
                }));
            }
        };
        Ok(Self {
            prepared,
            native: ReplacementQueueSubmission::new(plan, recording),
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        PreparedNativeSubmission<Semantic>,
        ReplacementNativeRecording,
    ) {
        (self.prepared, self.native.recording.into_execution())
    }
}

#[derive(Debug)]
pub struct PreparedReplacementQueueEnqueueFailure<Semantic> {
    pub reason: ReplacementQueueError,
    pub submission: PreparedReplacementQueueSubmission<Semantic>,
}

#[derive(Debug)]
pub struct PreparedSignalSkipFailure<Semantic> {
    pub reason: ReplacementQueueError,
    pub prepared: PreparedNativeSubmission<Semantic>,
}

struct PendingReplacementQueueSubmit {
    receiver: mpsc::Receiver<
        Result<
            ReplacementQueueSubmission,
            Box<(ReplacementQueueError, ReplacementQueueSubmission)>,
        >,
    >,
    recovery: Option<ReplacementQueueSubmission>,
}

#[derive(Debug)]
pub struct PreparedReplacementAuxiliaryQueueEnqueueFailure {
    pub reason: ReplacementQueueError,
    pub submission: PreparedReplacementAuxiliaryQueueSubmission,
}

#[must_use = "auxiliary driver acceptance must be observed"]
pub struct PendingPreparedReplacementAuxiliaryQueueSubmit {
    pending: PendingReplacementQueueSubmit,
    prepared: PreparedAuxiliaryNativeSubmission,
}

pub enum PreparedReplacementAuxiliaryQueueSubmitPoll {
    Pending(PendingPreparedReplacementAuxiliaryQueueSubmit),
    DriverAccepted(PreparedReplacementAuxiliaryQueueSubmission),
    DriverRefused {
        reason: ReplacementQueueError,
        submission: PreparedReplacementAuxiliaryQueueSubmission,
    },
}

impl PendingPreparedReplacementAuxiliaryQueueSubmit {
    pub fn try_complete(mut self) -> PreparedReplacementAuxiliaryQueueSubmitPoll {
        match self.pending.try_complete() {
            None => PreparedReplacementAuxiliaryQueueSubmitPoll::Pending(self),
            Some(Ok(native)) => PreparedReplacementAuxiliaryQueueSubmitPoll::DriverAccepted(
                PreparedReplacementAuxiliaryQueueSubmission {
                    prepared: self.prepared,
                    native,
                },
            ),
            Some(Err(failure)) => {
                let (reason, native) = *failure;
                PreparedReplacementAuxiliaryQueueSubmitPoll::DriverRefused {
                    reason,
                    submission: PreparedReplacementAuxiliaryQueueSubmission {
                        prepared: self.prepared,
                        native,
                    },
                }
            }
        }
    }

    pub fn wait(
        self,
    ) -> Result<
        PreparedReplacementAuxiliaryQueueSubmission,
        Box<(
            ReplacementQueueError,
            PreparedReplacementAuxiliaryQueueSubmission,
        )>,
    > {
        match self.pending.wait() {
            Ok(native) => Ok(PreparedReplacementAuxiliaryQueueSubmission {
                prepared: self.prepared,
                native,
            }),
            Err(failure) => {
                let (reason, native) = *failure;
                Err(Box::new((
                    reason,
                    PreparedReplacementAuxiliaryQueueSubmission {
                        prepared: self.prepared,
                        native,
                    },
                )))
            }
        }
    }
}

#[must_use = "driver acceptance must be observed before committing the prepared replay token"]
pub struct PendingPreparedReplacementQueueSubmit<Semantic> {
    pending: PendingReplacementQueueSubmit,
    prepared: PreparedNativeSubmission<Semantic>,
}

pub enum PreparedReplacementQueueSubmitPoll<Semantic> {
    Pending(PendingPreparedReplacementQueueSubmit<Semantic>),
    DriverAccepted(PreparedReplacementQueueSubmission<Semantic>),
    DriverRefused {
        reason: ReplacementQueueError,
        submission: PreparedReplacementQueueSubmission<Semantic>,
    },
}

impl<Semantic> PendingPreparedReplacementQueueSubmit<Semantic> {
    pub fn try_complete(mut self) -> PreparedReplacementQueueSubmitPoll<Semantic> {
        match self.pending.try_complete() {
            None => PreparedReplacementQueueSubmitPoll::Pending(self),
            Some(Ok(native)) => PreparedReplacementQueueSubmitPoll::DriverAccepted(
                PreparedReplacementQueueSubmission {
                    prepared: self.prepared,
                    native,
                },
            ),
            Some(Err(failure)) => {
                let (reason, native) = *failure;
                PreparedReplacementQueueSubmitPoll::DriverRefused {
                    reason,
                    submission: PreparedReplacementQueueSubmission {
                        prepared: self.prepared,
                        native,
                    },
                }
            }
        }
    }

    pub fn wait(
        self,
    ) -> Result<
        PreparedReplacementQueueSubmission<Semantic>,
        Box<(
            ReplacementQueueError,
            PreparedReplacementQueueSubmission<Semantic>,
        )>,
    > {
        match self.pending.wait() {
            Ok(native) => Ok(PreparedReplacementQueueSubmission {
                prepared: self.prepared,
                native,
            }),
            Err(failure) => {
                let (reason, native) = *failure;
                Err(Box::new((
                    reason,
                    PreparedReplacementQueueSubmission {
                        prepared: self.prepared,
                        native,
                    },
                )))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn disconnected_for_test(
        submission: PreparedReplacementQueueSubmission<Semantic>,
    ) -> Self {
        let PreparedReplacementQueueSubmission { prepared, native } = submission;
        let (sender, receiver) = mpsc::channel();
        drop(sender);
        Self {
            pending: PendingReplacementQueueSubmit {
                receiver,
                recovery: Some(native),
            },
            prepared,
        }
    }
}

impl PendingReplacementQueueSubmit {
    fn try_complete(
        &mut self,
    ) -> Option<
        Result<
            ReplacementQueueSubmission,
            Box<(ReplacementQueueError, ReplacementQueueSubmission)>,
        >,
    > {
        match self.receiver.try_recv() {
            Ok(result) => {
                self.recovery.take();
                Some(result)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(Box::new((
                ReplacementQueueError::OwnerStopped,
                self.recovery
                    .take()
                    .expect("pending submission retains disconnect recovery"),
            )))),
        }
    }

    fn wait(
        self,
    ) -> Result<ReplacementQueueSubmission, Box<(ReplacementQueueError, ReplacementQueueSubmission)>>
    {
        match self.receiver.recv() {
            Ok(result) => result,
            Err(_) => Err(Box::new((
                ReplacementQueueError::OwnerStopped,
                self.recovery
                    .expect("pending submission retains disconnect recovery"),
            ))),
        }
    }
}

enum Request {
    Submit {
        submission: Box<ReplacementQueueSubmission>,
        reply: mpsc::SyncSender<
            Result<
                ReplacementQueueSubmission,
                Box<(ReplacementQueueError, ReplacementQueueSubmission)>,
            >,
        >,
    },
    WaitIdle {
        reply: mpsc::SyncSender<Result<(), ReplacementQueueError>>,
    },
    Stop,
}

/// The first failed driver submission is terminal for this queue owner.
#[derive(Default)]
struct FailureLatch(Mutex<Option<ReplacementQueueError>>);

impl FailureLatch {
    fn get(&self) -> Option<ReplacementQueueError> {
        *self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set(&self, error: ReplacementQueueError) {
        let mut failure = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if failure.is_none() {
            *failure = Some(error);
        }
    }
}

#[derive(Default)]
struct SignalAdmission(Mutex<u64>);

impl SignalAdmission {
    fn admit(&self, value: u64) -> Result<std::sync::MutexGuard<'_, u64>, ReplacementQueueError> {
        let mut last = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if last.checked_add(1) != Some(value) {
            return Err(ReplacementQueueError::SignalValueOutOfOrder);
        }
        *last = value;
        Ok(last)
    }
}

pub struct ReplacementQueueOwner {
    id: QueueOwnerId,
    sender: mpsc::Sender<Request>,
    failure: Arc<FailureLatch>,
    signal_admission: SignalAdmission,
    join: Option<std::thread::JoinHandle<()>>,
}

impl ReplacementQueueOwner {
    pub fn start(
        id: QueueOwnerId,
        device: &ash::Device,
        queue: vk::Queue,
    ) -> std::io::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let failure = Arc::new(FailureLatch::default());
        let thread_failure = Arc::clone(&failure);
        let thread_device = device.clone();
        let join = std::thread::Builder::new()
            .name(format!("reims-vgpu-replay-submit-{}", id.get()))
            .spawn(move || run(&thread_device, queue, receiver, &thread_failure))?;
        Ok(Self {
            id,
            sender,
            failure,
            signal_admission: SignalAdmission::default(),
            join: Some(join),
        })
    }

    fn submit(
        &self,
        submission: ReplacementQueueSubmission,
    ) -> Result<PendingReplacementQueueSubmit, Box<ReplacementQueueEnqueueFailure>> {
        if submission.plan.signal_queue() != self.id {
            return Err(Box::new(ReplacementQueueEnqueueFailure {
                reason: ReplacementQueueError::WrongQueue,
                submission,
            }));
        }
        if let Some(error) = self.failure.get() {
            return Err(Box::new(ReplacementQueueEnqueueFailure {
                reason: error,
                submission,
            }));
        }
        let admission = match self.signal_admission.admit(submission.plan.signal_value) {
            Ok(admission) => admission,
            Err(reason) => {
                return Err(Box::new(ReplacementQueueEnqueueFailure {
                    reason,
                    submission,
                }));
            }
        };
        let (reply, receiver) = mpsc::sync_channel(1);
        let recovery = submission.clone();
        if let Err(error) = self.sender.send(Request::Submit {
            submission: Box::new(submission),
            reply,
        }) {
            drop(admission);
            let Request::Submit { submission, .. } = error.0 else {
                unreachable!("submit sends only a submit request")
            };
            return Err(Box::new(ReplacementQueueEnqueueFailure {
                reason: ReplacementQueueError::OwnerStopped,
                submission: *submission,
            }));
        }
        drop(admission);
        Ok(PendingReplacementQueueSubmit {
            receiver,
            recovery: Some(recovery),
        })
    }

    /// Enqueue a submission whose timeline signal was reserved by the replay
    /// owner. Successful driver return gives the caller back that same token;
    /// only then may it enter semantic dependency/completion ownership.
    pub fn submit_prepared<Semantic>(
        &self,
        submission: PreparedReplacementQueueSubmission<Semantic>,
    ) -> Result<
        PendingPreparedReplacementQueueSubmit<Semantic>,
        Box<PreparedReplacementQueueEnqueueFailure<Semantic>>,
    > {
        let PreparedReplacementQueueSubmission { prepared, native } = submission;
        match self.submit(native) {
            Ok(pending) => Ok(PendingPreparedReplacementQueueSubmit { pending, prepared }),
            Err(failure) => Err(Box::new(PreparedReplacementQueueEnqueueFailure {
                reason: failure.reason,
                submission: PreparedReplacementQueueSubmission {
                    prepared,
                    native: failure.submission,
                },
            })),
        }
    }

    pub fn submit_auxiliary(
        &self,
        submission: PreparedReplacementAuxiliaryQueueSubmission,
    ) -> Result<
        PendingPreparedReplacementAuxiliaryQueueSubmit,
        Box<PreparedReplacementAuxiliaryQueueEnqueueFailure>,
    > {
        let PreparedReplacementAuxiliaryQueueSubmission { prepared, native } = submission;
        match self.submit(native) {
            Ok(pending) => Ok(PendingPreparedReplacementAuxiliaryQueueSubmit { pending, prepared }),
            Err(failure) => Err(Box::new(PreparedReplacementAuxiliaryQueueEnqueueFailure {
                reason: failure.reason,
                submission: PreparedReplacementAuxiliaryQueueSubmission {
                    prepared,
                    native: failure.submission,
                },
            })),
        }
    }

    pub fn submit_present(
        &self,
        submission: PreparedReplacementPresentQueueSubmission,
    ) -> Result<
        PendingPreparedReplacementPresentQueueSubmit,
        Box<PreparedReplacementPresentQueueEnqueueFailure>,
    > {
        let PreparedReplacementPresentQueueSubmission { prepared, native } = submission;
        match self.submit(native) {
            Ok(pending) => Ok(PendingPreparedReplacementPresentQueueSubmit { pending, prepared }),
            Err(failure) => Err(Box::new(PreparedReplacementPresentQueueEnqueueFailure {
                reason: failure.reason,
                submission: PreparedReplacementPresentQueueSubmission {
                    prepared,
                    native: failure.submission,
                },
            })),
        }
    }

    /// Explicitly consume a reserved signal point for work that will never be
    /// enqueued. This is the only way a later prepared point may skip it.
    pub fn skip_prepared<Semantic>(
        &self,
        prepared: PreparedNativeSubmission<Semantic>,
    ) -> Result<PreparedNativeSubmission<Semantic>, PreparedSignalSkipFailure<Semantic>> {
        if prepared.point().queue != self.id {
            return Err(PreparedSignalSkipFailure {
                reason: ReplacementQueueError::WrongQueue,
                prepared,
            });
        }
        let admission = match self.signal_admission.admit(prepared.point().value.get()) {
            Ok(admission) => admission,
            Err(reason) => return Err(PreparedSignalSkipFailure { reason, prepared }),
        };
        drop(admission);
        Ok(prepared)
    }

    pub fn skip_auxiliary(
        &self,
        prepared: PreparedAuxiliaryNativeSubmission,
    ) -> Result<
        PreparedAuxiliaryNativeSubmission,
        (ReplacementQueueError, PreparedAuxiliaryNativeSubmission),
    > {
        if prepared.point().queue != self.id {
            return Err((ReplacementQueueError::WrongQueue, prepared));
        }
        let admission = match self.signal_admission.admit(prepared.point().value.get()) {
            Ok(admission) => admission,
            Err(reason) => return Err((reason, prepared)),
        };
        drop(admission);
        Ok(prepared)
    }

    pub fn skip_present(
        &self,
        prepared: PreparedPresentNativeSubmission,
    ) -> Result<
        PreparedPresentNativeSubmission,
        (ReplacementQueueError, PreparedPresentNativeSubmission),
    > {
        if prepared.point().queue != self.id {
            return Err((ReplacementQueueError::WrongQueue, prepared));
        }
        let admission = match self.signal_admission.admit(prepared.point().value.get()) {
            Ok(admission) => admission,
            Err(reason) => return Err((reason, prepared)),
        };
        drop(admission);
        Ok(prepared)
    }

    /// Serialize a queue-idle boundary with all submissions owned by this
    /// lane. Swapchain replacement uses this before destroying images that
    /// earlier Present submissions may still reference.
    pub fn wait_idle(&self) -> Result<(), ReplacementQueueError> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Request::WaitIdle { reply })
            .map_err(|_| ReplacementQueueError::OwnerStopped)?;
        receiver
            .recv()
            .map_err(|_| ReplacementQueueError::OwnerStopped)?
    }

    pub fn stop(mut self) {
        self.stop_inner();
    }

    /// Stop the queue owner in place after device loss. Its retained sender
    /// then makes every later enqueue or idle request return `OwnerStopped`.
    pub fn shutdown(&mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = self.sender.send(Request::Stop);
            let _ = join.join();
        }
    }
}

impl Drop for ReplacementQueueOwner {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

fn run(
    device: &ash::Device,
    queue: vk::Queue,
    receiver: mpsc::Receiver<Request>,
    failure: &FailureLatch,
) {
    while let Ok(request) = receiver.recv() {
        match request {
            Request::Submit { submission, reply } => {
                let mut submission = submission;
                let result = if let Some(error) = failure.get() {
                    Err(error)
                } else {
                    submit(device, queue, &mut submission).map_err(ReplacementQueueError::Driver)
                };
                if let Err(error) = result {
                    failure.set(error);
                }
                let result = match result {
                    Ok(()) => Ok(*submission),
                    Err(error) => Err(Box::new((error, *submission))),
                };
                let _ = reply.send(result);
            }
            Request::WaitIdle { reply } => {
                let result = if let Some(error) = failure.get() {
                    Err(error)
                } else {
                    unsafe { device.queue_wait_idle(queue) }.map_err(ReplacementQueueError::Driver)
                };
                if let Err(error) = result {
                    failure.set(error);
                }
                let _ = reply.send(result);
            }
            Request::Stop => break,
        }
    }
}

fn submit(
    device: &ash::Device,
    queue: vk::Queue,
    submission: &mut ReplacementQueueSubmission,
) -> Result<(), vk::Result> {
    #[cfg(feature = "host-window")]
    let present_semaphores = submission
        .present
        .as_ref()
        .map(|present| (present.acquire_wait, present.render_finished));
    #[cfg(not(feature = "host-window"))]
    let present_semaphores = None;
    let operands = queue_submit_operands(&submission.plan, present_semaphores);
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&operands.wait_values)
        .signal_semaphore_values(&operands.signal_values);
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(&operands.wait_semaphores)
        .wait_dst_stage_mask(&operands.wait_stages)
        .command_buffers(submission.recording.command_buffers())
        .signal_semaphores(&operands.signal_semaphores)
        .push_next(&mut timeline);
    unsafe { device.queue_submit(queue, &[submit], submission.recording.fence()) }?;
    #[cfg(feature = "host-window")]
    if let Some(present) = submission.present.clone() {
        let waits = [present.render_finished];
        let swapchains = [present.swapchain];
        let indices = [present.image_index];
        submission.present_result = Some(unsafe {
            present.loader.queue_present(
                queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&waits)
                    .swapchains(&swapchains)
                    .image_indices(&indices),
            )
        });
    }
    Ok(())
}

#[derive(Debug)]
struct QueueSubmitOperands {
    wait_semaphores: Vec<vk::Semaphore>,
    wait_values: Vec<u64>,
    wait_stages: Vec<vk::PipelineStageFlags>,
    signal_semaphores: Vec<vk::Semaphore>,
    signal_values: Vec<u64>,
}

fn queue_submit_operands(
    plan: &TimelineSubmitPlan,
    present_semaphores: Option<(vk::Semaphore, vk::Semaphore)>,
) -> QueueSubmitOperands {
    let mut operands = QueueSubmitOperands {
        wait_semaphores: plan.waits.iter().map(|wait| wait.semaphore).collect(),
        wait_values: plan.waits.iter().map(|wait| wait.value).collect(),
        wait_stages: plan.waits.iter().map(|wait| wait.stage_mask).collect(),
        signal_semaphores: vec![plan.signal_semaphore],
        signal_values: vec![plan.signal_value],
    };
    if let Some((acquire_wait, render_finished)) = present_semaphores {
        operands.wait_semaphores.push(acquire_wait);
        operands.wait_values.push(0);
        operands.wait_stages.push(vk::PipelineStageFlags::TRANSFER);
        operands.signal_semaphores.push(render_finished);
        operands.signal_values.push(0);
    }
    operands
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::{
        DirectReplayNativeOwner, PreparedNativeSubmission, RecordingWorkerId,
        TransactionRecordingPlan, WaitDependencyCause,
    };
    use reims_vgpu_protocol::{
        QueueTimelineValue, SessionGenerationId, SubmissionDomainId, TransactionId,
        VulkanDeviceEpochId,
    };

    fn prepared<Semantic: Clone + std::fmt::Debug>(
        epoch: VulkanDeviceEpochId,
        queue: QueueOwnerId,
        semantic: Semantic,
    ) -> PreparedNativeSubmission<Semantic> {
        let mut owner = DirectReplayNativeOwner::new(epoch, 1).unwrap();
        owner
            .assign_recording(TransactionRecordingPlan {
                transaction: TransactionId::new(1),
                domain: SubmissionDomainId::new(1),
                continuation_predecessor: None,
            })
            .unwrap();
        let plan = owner
            .queue_candidate(
                TransactionId::new(1),
                Box::<[(TransactionId, WaitDependencyCause)]>::default(),
            )
            .unwrap()
            .pop()
            .unwrap();
        owner
            .prepare(plan, queue, SessionGenerationId::new(1), semantic)
            .unwrap()
    }

    fn prepared_present(
        epoch: VulkanDeviceEpochId,
        queue: QueueOwnerId,
    ) -> PreparedPresentNativeSubmission {
        DirectReplayNativeOwner::<()>::new(epoch, 1)
            .unwrap()
            .prepare_present(TransactionId::new(4), queue)
            .unwrap()
    }

    #[test]
    fn only_the_queue_reply_constructs_a_driver_accepted_present() {
        let epoch = VulkanDeviceEpochId::new(1);
        let queue = QueueOwnerId::new(3);
        let timelines = QueueTimelineSemaphores::new(epoch, [(queue, vk::Semaphore::from_raw(9))]);
        let command = vk::CommandBuffer::from_raw(17);
        let submission = PreparedReplacementPresentQueueSubmission::new(
            prepared_present(epoch, queue),
            &timelines,
            ReplacementPresentRecording {
                slot: 0,
                command_pool: vk::CommandPool::null(),
                command_buffer: command,
                fence: vk::Fence::null(),
            },
            Box::<[reims_vgpu_core::QueueTimelinePoint]>::default(),
        )
        .unwrap();
        let PreparedReplacementPresentQueueSubmission { prepared, native } = submission;
        let (reply, receiver) = mpsc::sync_channel(1);
        let recovery = native.clone();
        reply.send(Ok(native)).unwrap();
        let accepted = PendingPreparedReplacementPresentQueueSubmit {
            pending: PendingReplacementQueueSubmit {
                receiver,
                recovery: Some(recovery),
            },
            prepared,
        }
        .wait()
        .unwrap();
        assert_eq!(accepted.prepared().transaction(), TransactionId::new(4));
        assert_eq!(accepted.prepared().point().queue, queue);
        assert_eq!(accepted.into_parts().1.command_buffer, command);
    }

    #[test]
    fn present_submit_pairs_binary_semaphores_with_zero_timeline_values() {
        let epoch = VulkanDeviceEpochId::new(1);
        let work_queue = QueueOwnerId::new(3);
        let release_queue = QueueOwnerId::new(4);
        let timelines = QueueTimelineSemaphores::new(
            epoch,
            [
                (work_queue, vk::Semaphore::from_raw(9)),
                (release_queue, vk::Semaphore::from_raw(10)),
            ],
        );
        let plan = timelines
            .plan_present(
                &prepared_present(epoch, work_queue),
                [reims_vgpu_core::QueueTimelinePoint {
                    epoch,
                    queue: release_queue,
                    value: QueueTimelineValue::new(7),
                }],
            )
            .unwrap();
        let operands = queue_submit_operands(
            &plan,
            Some((vk::Semaphore::from_raw(11), vk::Semaphore::from_raw(12))),
        );

        assert_eq!(
            operands.wait_semaphores,
            [vk::Semaphore::from_raw(10), vk::Semaphore::from_raw(11)]
        );
        assert_eq!(operands.wait_values, [7, 0]);
        assert_eq!(
            operands.wait_stages,
            [
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
            ]
        );
        assert_eq!(
            operands.signal_semaphores,
            [vk::Semaphore::from_raw(9), vk::Semaphore::from_raw(12)]
        );
        assert_eq!(operands.signal_values, [1, 0]);
    }

    #[test]
    fn submit_plan_remembers_its_signal_queue_without_reconstructing_it_from_a_handle() {
        let timelines = crate::replacement_submit::QueueTimelineSemaphores::new(
            VulkanDeviceEpochId::new(1),
            [(QueueOwnerId::new(3), vk::Semaphore::from_raw(9))],
        );
        let prepared = prepared(VulkanDeviceEpochId::new(1), QueueOwnerId::new(3), ());
        let submission = PreparedReplacementQueueSubmission::new(
            prepared,
            &timelines,
            ReplacementNativeRecording::synthetic(
                RecordingWorkerId::new(0),
                Box::<[vk::CommandBuffer]>::default(),
                vk::Fence::null(),
            ),
        )
        .unwrap();
        assert_eq!(submission.native.plan.signal_queue(), QueueOwnerId::new(3));
    }

    #[test]
    fn preparation_refusal_returns_semantics_and_native_recording_ownership() {
        let timelines = crate::replacement_submit::QueueTimelineSemaphores::new(
            VulkanDeviceEpochId::new(1),
            [(QueueOwnerId::new(3), vk::Semaphore::from_raw(9))],
        );
        let prepared = prepared(
            VulkanDeviceEpochId::new(2),
            QueueOwnerId::new(3),
            String::from("owned"),
        );
        let command = vk::CommandBuffer::from_raw(17);
        let failure = PreparedReplacementQueueSubmission::new(
            prepared,
            &timelines,
            ReplacementNativeRecording::synthetic(
                RecordingWorkerId::new(0),
                [command],
                vk::Fence::null(),
            ),
        )
        .unwrap_err();
        assert_eq!(failure.reason, TimelineSubmitPlanError::MixedEpochs);
        assert_eq!(failure.prepared.semantic(), "owned");
        assert_eq!(failure.recording.command_buffers.as_ref(), [command]);
    }

    #[test]
    fn queue_signal_admission_never_allows_prepared_points_to_reorder() {
        let admission = SignalAdmission::default();
        assert!(matches!(
            admission.admit(2),
            Err(ReplacementQueueError::SignalValueOutOfOrder)
        ));
        drop(admission.admit(1).unwrap());
        assert!(matches!(
            admission.admit(1),
            Err(ReplacementQueueError::SignalValueOutOfOrder)
        ));
        drop(admission.admit(2).unwrap());
        assert!(matches!(
            admission.admit(4),
            Err(ReplacementQueueError::SignalValueOutOfOrder)
        ));
        drop(admission.admit(3).unwrap());
    }

    #[test]
    fn only_the_vulkan_device_loss_result_closes_the_device_incarnation() {
        assert!(ReplacementQueueError::Driver(vk::Result::ERROR_DEVICE_LOST).is_device_lost());
        assert!(
            !ReplacementQueueError::Driver(vk::Result::ERROR_OUT_OF_HOST_MEMORY).is_device_lost()
        );
        assert!(!ReplacementQueueError::OwnerStopped.is_device_lost());
    }

    #[test]
    fn driver_receipt_returns_the_exact_native_recording_on_acceptance() {
        let timelines = crate::replacement_submit::QueueTimelineSemaphores::new(
            VulkanDeviceEpochId::new(1),
            [(QueueOwnerId::new(3), vk::Semaphore::from_raw(9))],
        );
        let command = vk::CommandBuffer::from_raw(17);
        let fence = vk::Fence::from_raw(18);
        let submission = PreparedReplacementQueueSubmission::new(
            prepared(
                VulkanDeviceEpochId::new(1),
                QueueOwnerId::new(3),
                "semantic",
            ),
            &timelines,
            ReplacementNativeRecording::synthetic(RecordingWorkerId::new(0), [command], fence),
        )
        .unwrap();
        let PreparedReplacementQueueSubmission { prepared, native } = submission;
        let (reply, receiver) = mpsc::sync_channel(1);
        let recovery = native.clone();
        reply.send(Ok(native)).unwrap();
        let returned = PendingPreparedReplacementQueueSubmit {
            pending: PendingReplacementQueueSubmit {
                receiver,
                recovery: Some(recovery),
            },
            prepared,
        }
        .wait()
        .unwrap();
        let (prepared, recording) = returned.into_parts();
        assert_eq!(prepared.semantic(), &"semantic");
        assert_eq!(recording.command_buffers.as_ref(), [command]);
        assert_eq!(recording.fence, fence);
    }

    #[test]
    fn disconnected_queue_owner_returns_recovery_recording() {
        let timelines = crate::replacement_submit::QueueTimelineSemaphores::new(
            VulkanDeviceEpochId::new(1),
            [(QueueOwnerId::new(3), vk::Semaphore::from_raw(9))],
        );
        let command = vk::CommandBuffer::from_raw(21);
        let submission = PreparedReplacementQueueSubmission::new(
            prepared(
                VulkanDeviceEpochId::new(1),
                QueueOwnerId::new(3),
                "semantic",
            ),
            &timelines,
            ReplacementNativeRecording::synthetic(
                RecordingWorkerId::new(0),
                [command],
                vk::Fence::null(),
            ),
        )
        .unwrap();
        let PreparedReplacementQueueSubmission { prepared, native } = submission;
        let (reply, receiver) = mpsc::sync_channel(1);
        drop(reply);
        let error = PendingPreparedReplacementQueueSubmit {
            pending: PendingReplacementQueueSubmit {
                receiver,
                recovery: Some(native),
            },
            prepared,
        }
        .wait()
        .unwrap_err();
        assert_eq!(error.0, ReplacementQueueError::OwnerStopped);
        assert_eq!(error.1.into_parts().1.command_buffers.as_ref(), [command]);
    }
}
