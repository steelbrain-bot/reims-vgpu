//! Interruptible timeline completion observation for one actual Vulkan queue.
//!
//! The watcher blocks away from recording and guest-publication owners. A
//! dedicated host-signaled timeline semaphore interrupts teardown without
//! advancing the queue timeline and therefore cannot manufacture completion.

use ash::vk;
use reims_vgpu_core::QueueTimelinePoint;
use reims_vgpu_protocol::{QueueOwnerId, QueueTimelineValue, VulkanDeviceEpochId};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};

#[derive(Debug)]
pub enum ReplacementTimelineWatcherStartError {
    Vulkan(vk::Result),
    Thread(std::io::Error),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementTimelineWatchError {
    WrongEpoch,
    WrongQueue,
    PointDidNotIncrease,
    OwnerStopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementTimelineProgress {
    pub queue: QueueOwnerId,
    pub completed: QueueTimelineValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementTimelineFailure {
    pub queue: QueueOwnerId,
    pub waiting_for: QueueTimelineValue,
    pub result: vk::Result,
}

impl ReplacementTimelineFailure {
    pub fn is_device_lost(self) -> bool {
        self.result == vk::Result::ERROR_DEVICE_LOST
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementTimelineObservation {
    Progress(ReplacementTimelineProgress),
    Failed(ReplacementTimelineFailure),
}

enum Request {
    Watch(QueueTimelinePoint),
    Stop,
}

#[derive(Default)]
struct WatchAdmission(Mutex<Option<QueueTimelineValue>>);

impl WatchAdmission {
    fn accept(&self, value: QueueTimelineValue) -> Result<(), ReplacementTimelineWatchError> {
        let mut last = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if last.is_some_and(|last| value <= last) {
            return Err(ReplacementTimelineWatchError::PointDidNotIncrease);
        }
        *last = Some(value);
        Ok(())
    }
}

/// One blocking completion thread for one real queue timeline.
pub struct ReplacementTimelineWatcher {
    epoch: VulkanDeviceEpochId,
    queue: QueueOwnerId,
    device: ash::Device,
    interrupt: vk::Semaphore,
    requests: mpsc::Sender<Request>,
    observations: mpsc::Receiver<ReplacementTimelineObservation>,
    stopping: Arc<AtomicBool>,
    admission: WatchAdmission,
    join: Option<std::thread::JoinHandle<()>>,
}

impl ReplacementTimelineWatcher {
    pub fn start(
        epoch: VulkanDeviceEpochId,
        queue: QueueOwnerId,
        device: &ash::Device,
        timeline: vk::Semaphore,
    ) -> Result<Self, ReplacementTimelineWatcherStartError> {
        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type);
        let interrupt = unsafe { device.create_semaphore(&create, None) }
            .map_err(ReplacementTimelineWatcherStartError::Vulkan)?;
        let (requests, request_receiver) = mpsc::channel();
        let (observation_sender, observations) = mpsc::channel();
        let stopping = Arc::new(AtomicBool::new(false));
        let thread_stopping = Arc::clone(&stopping);
        let thread_device = device.clone();
        let join = match std::thread::Builder::new()
            .name(format!("reims-vgpu-replay-complete-{}", queue.get()))
            .spawn(move || {
                run(
                    &thread_device,
                    queue,
                    timeline,
                    interrupt,
                    request_receiver,
                    observation_sender,
                    &thread_stopping,
                )
            }) {
            Ok(join) => join,
            Err(error) => {
                unsafe { device.destroy_semaphore(interrupt, None) };
                return Err(ReplacementTimelineWatcherStartError::Thread(error));
            }
        };
        Ok(Self {
            epoch,
            queue,
            device: device.clone(),
            interrupt,
            requests,
            observations,
            stopping,
            admission: WatchAdmission::default(),
            join: Some(join),
        })
    }

    pub fn watch(&self, point: QueueTimelinePoint) -> Result<(), ReplacementTimelineWatchError> {
        if point.epoch != self.epoch {
            return Err(ReplacementTimelineWatchError::WrongEpoch);
        }
        if point.queue != self.queue {
            return Err(ReplacementTimelineWatchError::WrongQueue);
        }
        if self.stopping.load(Ordering::Acquire) {
            return Err(ReplacementTimelineWatchError::OwnerStopped);
        }
        self.admission.accept(point.value)?;
        self.requests
            .send(Request::Watch(point))
            .map_err(|_| ReplacementTimelineWatchError::OwnerStopped)
    }

    pub fn try_observe(&self) -> Option<ReplacementTimelineObservation> {
        self.observations.try_recv().ok()
    }

    pub fn stop(mut self) {
        self.stop_inner();
    }

    /// Interrupt and join the watcher in place after device loss. The
    /// stopping latch makes every later watch request a typed refusal.
    pub fn shutdown(&mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        let Some(join) = self.join.take() else {
            return;
        };
        self.stopping.store(true, Ordering::Release);
        let _ = self.requests.send(Request::Stop);
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(self.interrupt)
            .value(1);
        let _ = unsafe { self.device.signal_semaphore(&signal) };
        let _ = join.join();
        unsafe { self.device.destroy_semaphore(self.interrupt, None) };
    }
}

impl Drop for ReplacementTimelineWatcher {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

fn run(
    device: &ash::Device,
    queue: QueueOwnerId,
    timeline: vk::Semaphore,
    interrupt: vk::Semaphore,
    requests: mpsc::Receiver<Request>,
    observations: mpsc::Sender<ReplacementTimelineObservation>,
    stopping: &AtomicBool,
) {
    while let Ok(request) = requests.recv() {
        let Request::Watch(point) = request else {
            break;
        };
        let semaphores = [timeline, interrupt];
        let values = [point.value.get(), 1];
        let wait = vk::SemaphoreWaitInfo::default()
            .flags(vk::SemaphoreWaitFlags::ANY)
            .semaphores(&semaphores)
            .values(&values);
        let result = unsafe { device.wait_semaphores(&wait, u64::MAX) };
        if stopping.load(Ordering::Acquire) {
            break;
        }
        match result {
            Ok(()) => {
                let completed = match unsafe { device.get_semaphore_counter_value(timeline) } {
                    Ok(value) => QueueTimelineValue::new(value),
                    Err(result) => {
                        let _ = observations.send(ReplacementTimelineObservation::Failed(
                            ReplacementTimelineFailure {
                                queue,
                                waiting_for: point.value,
                                result,
                            },
                        ));
                        break;
                    }
                };
                if observations
                    .send(ReplacementTimelineObservation::Progress(
                        ReplacementTimelineProgress { queue, completed },
                    ))
                    .is_err()
                {
                    break;
                }
            }
            Err(result) => {
                let _ = observations.send(ReplacementTimelineObservation::Failed(
                    ReplacementTimelineFailure {
                        queue,
                        waiting_for: point.value,
                        result,
                    },
                ));
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watch_admission_is_monotonic_and_allows_skipped_signal_values() {
        let admission = WatchAdmission::default();
        assert_eq!(admission.accept(QueueTimelineValue::new(2)), Ok(()));
        assert_eq!(admission.accept(QueueTimelineValue::new(4)), Ok(()));
        assert_eq!(
            admission.accept(QueueTimelineValue::new(3)),
            Err(ReplacementTimelineWatchError::PointDidNotIncrease)
        );
        assert_eq!(
            admission.accept(QueueTimelineValue::new(4)),
            Err(ReplacementTimelineWatchError::PointDidNotIncrease)
        );
    }
}
