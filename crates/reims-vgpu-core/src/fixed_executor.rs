//! Fixed worker executor with exclusively worker-owned mutable state.
//!
//! The worker population is created once. Jobs transfer ownership through a
//! channel and execute against one worker's private state; no job creates a
//! thread and no mutable native pool is shared between workers.

use std::{
    fmt,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
};

use crate::RecordingWorkerId;
use reims_vgpu_protocol::TransactionId;

type Job<W> = Box<dyn FnOnce(&mut W) + Send + 'static>;
type EventWake = Arc<dyn Fn() + Send + Sync + 'static>;

struct JobEnvelope<W> {
    transaction: Option<TransactionId>,
    job: Job<W>,
}

enum Message<W> {
    Execute(JobEnvelope<W>),
    Stop,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedExecutorCensus {
    pub workers: usize,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedExecutorOutcome {
    Completed,
    Panicked,
    WorkerUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedExecutorEvent {
    pub worker: RecordingWorkerId,
    pub transaction: TransactionId,
    pub outcome: FixedExecutorOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedExecutorError {
    NoWorkers,
    UnknownWorker,
    Stopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedExecutorWakeInstallError {
    AlreadyInstalled,
}

pub struct FixedExecutor<W> {
    senders: Vec<mpsc::Sender<Message<W>>>,
    workers: Vec<JoinHandle<()>>,
    next: AtomicUsize,
    submitted: AtomicU64,
    completed: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
    events: Mutex<mpsc::Receiver<FixedExecutorEvent>>,
    event_wake: Arc<Mutex<Option<EventWake>>>,
}

impl<W> fmt::Debug for FixedExecutor<W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let census = FixedExecutorCensus {
            workers: self.senders.len(),
            submitted: self.submitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        };
        formatter
            .debug_struct("FixedExecutor")
            .field("census", &census)
            .finish_non_exhaustive()
    }
}

impl<W: Send + 'static> FixedExecutor<W> {
    pub fn new(
        worker_count: usize,
        mut create_worker: impl FnMut(RecordingWorkerId) -> W,
    ) -> Result<Self, FixedExecutorError> {
        if worker_count == 0 {
            return Err(FixedExecutorError::NoWorkers);
        }
        Self::from_workers(
            (0..worker_count)
                .map(|index| create_worker(RecordingWorkerId::new(index)))
                .collect(),
        )
    }

    pub fn new_with_event_wake(
        worker_count: usize,
        mut create_worker: impl FnMut(RecordingWorkerId) -> W,
        event_wake: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self, FixedExecutorError> {
        if worker_count == 0 {
            return Err(FixedExecutorError::NoWorkers);
        }
        Self::start(
            (0..worker_count)
                .map(|index| create_worker(RecordingWorkerId::new(index)))
                .collect(),
            Some(Arc::new(event_wake)),
        )
    }

    /// Start the fixed executor from an already constructed all-or-nothing
    /// worker population. Fallible native owners can be built first, before
    /// any worker thread becomes reachable.
    pub fn from_workers(worker_states: Vec<W>) -> Result<Self, FixedExecutorError> {
        Self::start(worker_states, None)
    }

    fn start(
        worker_states: Vec<W>,
        initial_event_wake: Option<EventWake>,
    ) -> Result<Self, FixedExecutorError> {
        if worker_states.is_empty() {
            return Err(FixedExecutorError::NoWorkers);
        }
        let worker_count = worker_states.len();
        let completed = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicU64::new(0));
        let (event_sender, events) = mpsc::channel();
        let event_wake = Arc::new(Mutex::new(initial_event_wake));
        let mut senders = Vec::with_capacity(worker_count);
        let mut workers = Vec::with_capacity(worker_count);
        for (index, mut worker) in worker_states.into_iter().enumerate() {
            let worker_id = RecordingWorkerId::new(index);
            let (sender, receiver) = mpsc::channel();
            let worker_completed = Arc::clone(&completed);
            let worker_failed = Arc::clone(&failed);
            let worker_events = event_sender.clone();
            let worker_event_wake = Arc::clone(&event_wake);
            workers.push(thread::spawn(move || {
                let mut available = true;
                while let Ok(message) = receiver.recv() {
                    match message {
                        Message::Execute(envelope) => {
                            let outcome = if !available {
                                worker_failed.fetch_add(1, Ordering::Relaxed);
                                FixedExecutorOutcome::WorkerUnavailable
                            } else if catch_unwind(AssertUnwindSafe(|| (envelope.job)(&mut worker)))
                                .is_ok()
                            {
                                worker_completed.fetch_add(1, Ordering::Relaxed);
                                FixedExecutorOutcome::Completed
                            } else {
                                available = false;
                                worker_failed.fetch_add(1, Ordering::Relaxed);
                                FixedExecutorOutcome::Panicked
                            };
                            let completion_is_published =
                                envelope.transaction.is_none_or(|transaction| {
                                    worker_events
                                        .send(FixedExecutorEvent {
                                            worker: worker_id,
                                            transaction,
                                            outcome,
                                        })
                                        .is_ok()
                                });
                            if completion_is_published {
                                let wake = worker_event_wake
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                                    .as_ref()
                                    .cloned();
                                if let Some(wake) = wake {
                                    wake();
                                }
                            }
                        }
                        Message::Stop => break,
                    }
                }
            }));
            senders.push(sender);
        }
        Ok(Self {
            senders,
            workers,
            next: AtomicUsize::new(0),
            submitted: AtomicU64::new(0),
            completed,
            failed,
            events: Mutex::new(events),
            event_wake,
        })
    }

    pub fn submit(
        &self,
        job: impl FnOnce(&mut W) + Send + 'static,
    ) -> Result<(), FixedExecutorError> {
        if self.senders.is_empty() {
            return Err(FixedExecutorError::Stopped);
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        self.submit_envelope(RecordingWorkerId::new(index), None, job)
    }

    /// Submit transaction work to the next member of the fixed population and
    /// publish one event on the executor's shared completion stream.
    pub fn submit_transaction(
        &self,
        transaction: TransactionId,
        job: impl FnOnce(&mut W) + Send + 'static,
    ) -> Result<RecordingWorkerId, FixedExecutorError> {
        if self.senders.is_empty() {
            return Err(FixedExecutorError::Stopped);
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        let worker = RecordingWorkerId::new(index);
        self.submit_envelope(worker, Some(transaction), job)?;
        Ok(worker)
    }

    /// Submit to one established worker identity. Encoder continuations use
    /// this route when their mutable recording state remains worker-owned.
    pub fn submit_to(
        &self,
        worker: RecordingWorkerId,
        job: impl FnOnce(&mut W) + Send + 'static,
    ) -> Result<(), FixedExecutorError> {
        self.submit_envelope(worker, None, job)
    }

    pub fn submit_transaction_to(
        &self,
        worker: RecordingWorkerId,
        transaction: TransactionId,
        job: impl FnOnce(&mut W) + Send + 'static,
    ) -> Result<(), FixedExecutorError> {
        self.submit_envelope(worker, Some(transaction), job)
    }

    fn submit_envelope(
        &self,
        worker: RecordingWorkerId,
        transaction: Option<TransactionId>,
        job: impl FnOnce(&mut W) + Send + 'static,
    ) -> Result<(), FixedExecutorError> {
        let sender = self
            .senders
            .get(worker.index())
            .ok_or(FixedExecutorError::UnknownWorker)?;
        sender
            .send(Message::Execute(JobEnvelope {
                transaction,
                job: Box::new(job),
            }))
            .map_err(|_| FixedExecutorError::Stopped)?;
        self.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }

    pub fn take_events(&self) -> Vec<FixedExecutorEvent> {
        let events = self
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        events.try_iter().collect()
    }

    pub fn census(&self) -> FixedExecutorCensus {
        FixedExecutorCensus {
            workers: self.senders.len(),
            submitted: self.submitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }

    /// Install the scheduler edge that observes every accepted job after its
    /// completion event has been published. Executors are often constructed
    /// before their host scheduler exists, so installation is intentionally a
    /// separate one-time lifecycle transition.
    pub fn install_event_wake(
        &self,
        event_wake: impl Fn() + Send + Sync + 'static,
    ) -> Result<(), FixedExecutorWakeInstallError> {
        let mut slot = self
            .event_wake
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_some() {
            return Err(FixedExecutorWakeInstallError::AlreadyInstalled);
        }
        *slot = Some(Arc::new(event_wake));
        Ok(())
    }

    /// Stop and join the complete worker population without replacing it.
    ///
    /// Device-loss teardown uses this after closing native admission. The
    /// retained executor then refuses every later dispatch as `Stopped`.
    pub fn stop(&mut self) {
        self.stop_inner();
    }
}

impl<W> FixedExecutor<W> {
    fn stop_inner(&mut self) {
        for sender in self.senders.drain(..) {
            let _ = sender.send(Message::Stop);
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<W> Drop for FixedExecutor<W> {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, sync::mpsc, thread::ThreadId};

    #[derive(Debug)]
    struct WorkerState {
        index: usize,
        jobs: u64,
    }

    #[test]
    fn jobs_use_a_fixed_population_and_worker_local_state() {
        let executor = FixedExecutor::new(3, |id| WorkerState {
            index: id.index(),
            jobs: 0,
        })
        .unwrap();
        let (send, receive) = mpsc::channel();
        for _ in 0..6 {
            let send = send.clone();
            executor
                .submit(move |worker| {
                    worker.jobs += 1;
                    send.send((worker.index, worker.jobs, std::thread::current().id()))
                        .unwrap();
                })
                .unwrap();
        }
        drop(send);
        let observations = receive.iter().collect::<Vec<_>>();
        assert_eq!(observations.len(), 6);
        let threads = observations
            .iter()
            .map(|(_, _, thread)| *thread)
            .collect::<HashSet<ThreadId>>();
        assert_eq!(threads.len(), 3);
        for index in 0..3 {
            assert_eq!(
                observations
                    .iter()
                    .filter(|(worker, _, _)| *worker == index)
                    .map(|(_, jobs, _)| *jobs)
                    .collect::<Vec<_>>(),
                vec![1, 2]
            );
        }
    }

    #[test]
    fn stopped_executor_has_no_worker_population_and_refuses_dispatch() {
        let mut executor = FixedExecutor::new(2, |_| ()).unwrap();

        executor.stop();

        assert_eq!(executor.census().workers, 0);
        assert_eq!(executor.submit(|_| {}), Err(FixedExecutorError::Stopped));
        assert_eq!(
            executor.submit_transaction(TransactionId::new(1), |_| {}),
            Err(FixedExecutorError::Stopped)
        );
    }

    #[test]
    fn dropping_the_executor_drains_accepted_jobs_before_stopping_workers() {
        let (send, receive) = mpsc::channel();
        {
            let executor = FixedExecutor::new(2, |_| ()).unwrap();
            for value in 0..8 {
                let send = send.clone();
                executor.submit(move |_| send.send(value).unwrap()).unwrap();
            }
        }
        drop(send);
        let mut values = receive.iter().collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn zero_workers_is_a_typed_refusal() {
        assert!(matches!(
            FixedExecutor::<()>::new(0, |_| ()),
            Err(FixedExecutorError::NoWorkers)
        ));
        assert!(matches!(
            FixedExecutor::<()>::from_workers(Vec::new()),
            Err(FixedExecutorError::NoWorkers)
        ));
    }

    #[test]
    fn preconstructed_workers_keep_their_population_order() {
        let executor = FixedExecutor::from_workers(vec![10usize, 20usize]).unwrap();
        let (send, receive) = mpsc::channel();
        executor
            .submit_to(RecordingWorkerId::new(1), move |state| {
                send.send(*state).unwrap()
            })
            .unwrap();
        assert_eq!(receive.recv().unwrap(), 20);
    }

    #[test]
    fn pinned_jobs_reuse_exactly_one_workers_private_state() {
        let executor = FixedExecutor::new(2, |id| WorkerState {
            index: id.index(),
            jobs: 0,
        })
        .unwrap();
        let (send, receive) = mpsc::channel();
        for _ in 0..3 {
            let send = send.clone();
            executor
                .submit_to(RecordingWorkerId::new(1), move |worker| {
                    worker.jobs += 1;
                    send.send((worker.index, worker.jobs)).unwrap();
                })
                .unwrap();
        }
        drop(send);
        assert_eq!(
            receive.iter().collect::<Vec<_>>(),
            vec![(1, 1), (1, 2), (1, 3)]
        );
        assert_eq!(executor.worker_count(), 2);
        assert_eq!(
            executor.submit_to(RecordingWorkerId::new(2), |_| {}),
            Err(FixedExecutorError::UnknownWorker)
        );
    }

    #[test]
    fn a_panicked_worker_returns_one_terminal_outcome_for_every_accepted_transaction() {
        let (wake_sender, wake_receiver) = mpsc::channel();
        let executor = FixedExecutor::new_with_event_wake(
            1,
            |_| (),
            move || {
                let _ = wake_sender.send(());
            },
        )
        .unwrap();
        executor
            .submit_transaction_to(RecordingWorkerId::new(0), TransactionId::new(1), |_| {
                panic!("recording failed")
            })
            .unwrap();
        executor
            .submit_transaction_to(RecordingWorkerId::new(0), TransactionId::new(2), |_| {})
            .unwrap();

        wake_receiver.recv().unwrap();
        wake_receiver.recv().unwrap();
        assert_eq!(
            executor.take_events(),
            vec![
                FixedExecutorEvent {
                    worker: RecordingWorkerId::new(0),
                    transaction: TransactionId::new(1),
                    outcome: FixedExecutorOutcome::Panicked,
                },
                FixedExecutorEvent {
                    worker: RecordingWorkerId::new(0),
                    transaction: TransactionId::new(2),
                    outcome: FixedExecutorOutcome::WorkerUnavailable,
                },
            ]
        );
        assert_eq!(executor.census().failed, 2);
    }

    #[test]
    fn an_untracked_job_wakes_the_owner_after_its_completion_is_published() {
        let (completion_sender, completion_receiver) = mpsc::channel();
        let (wake_sender, wake_receiver) = mpsc::channel();
        let executor = FixedExecutor::new_with_event_wake(
            1,
            |_| (),
            move || {
                let _ = wake_sender.send(());
            },
        )
        .unwrap();
        executor
            .submit(move |_| {
                completion_sender.send(7).unwrap();
            })
            .unwrap();

        assert_eq!(completion_receiver.recv().unwrap(), 7);
        wake_receiver.recv().unwrap();
        assert!(executor.take_events().is_empty());
    }

    #[test]
    fn a_late_installed_wake_observes_transaction_completion_exactly_once() {
        let executor = FixedExecutor::new(1, |_| ()).unwrap();
        let (wake_sender, wake_receiver) = mpsc::channel();
        executor
            .install_event_wake(move || {
                let _ = wake_sender.send(());
            })
            .unwrap();
        assert_eq!(
            executor.install_event_wake(|| {}),
            Err(FixedExecutorWakeInstallError::AlreadyInstalled)
        );

        executor
            .submit_transaction(TransactionId::new(9), |_| {})
            .unwrap();
        wake_receiver.recv().unwrap();
        assert_eq!(executor.take_events().len(), 1);
        assert!(wake_receiver.try_recv().is_err());
    }
}
