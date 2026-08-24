//! Off-lock recording owners for scheduler-admitted EXEC transactions.
//!
//! Only work admitted by [`reims_vgpu_core::SubmissionScheduler`] reaches this
//! owner. Conflicting work remains in the scheduler's unbounded queue; every
//! active entry here therefore corresponds exactly to a live recording
//! admission and owns one thread until its terminal result is returned.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use reims_vgpu_protocol::SubmissionIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkerDispatchError {
    AlreadyActive(SubmissionIdentity),
}

#[derive(Debug)]
pub(crate) enum WorkerOutcome<T> {
    Completed(T),
    Panicked,
}

#[derive(Debug)]
pub(crate) struct WorkerResult<T> {
    pub(crate) identity: SubmissionIdentity,
    pub(crate) outcome: WorkerOutcome<T>,
}

/// Exact set of independently recording EXECs and their unbounded result inbox.
pub(crate) struct SubmissionWorkers<T> {
    result_tx: Sender<WorkerResult<T>>,
    result_rx: Receiver<WorkerResult<T>>,
    active: HashMap<SubmissionIdentity, JoinHandle<()>>,
}

impl<T> std::fmt::Debug for SubmissionWorkers<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmissionWorkers")
            .field("active", &self.active.len())
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Default for SubmissionWorkers<T> {
    fn default() -> Self {
        let (result_tx, result_rx) = mpsc::channel();
        Self {
            result_tx,
            result_rx,
            active: HashMap::new(),
        }
    }
}

impl<T: Send + 'static> SubmissionWorkers<T> {
    pub(crate) fn dispatch<W, F>(
        &mut self,
        identity: SubmissionIdentity,
        work: W,
        wake: crate::runtime::host::WorkerWake,
        record: F,
    ) -> Result<(), WorkerDispatchError>
    where
        W: Send + 'static,
        F: FnOnce(W) -> T + Send + 'static,
    {
        if self.active.contains_key(&identity) {
            return Err(WorkerDispatchError::AlreadyActive(identity));
        }
        let result_tx = self.result_tx.clone();
        let handle = std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| record(work)))
                .map(WorkerOutcome::Completed)
                .unwrap_or(WorkerOutcome::Panicked);
            // A dropped receiver means the device lifetime already ended. The
            // worker still owns and drops its recording result correctly.
            let _ = result_tx.send(WorkerResult { identity, outcome });
            wake.wake();
        });
        let displaced = self.active.insert(identity, handle);
        debug_assert!(displaced.is_none());
        Ok(())
    }

    /// Take every result already published without waiting for a recorder.
    pub(crate) fn take_finished(&mut self) -> Vec<WorkerResult<T>> {
        let mut finished = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            let handle = self
                .active
                .remove(&result.identity)
                .expect("every worker result owns one active thread");
            handle
                .join()
                .expect("the worker catches its recording panic before return");
            finished.push(result);
        }
        finished
    }

    /// Join every active recording and return all terminal results.
    ///
    /// Teardown has no timeout: destroying a command pool still used by a CPU
    /// recorder is invalid, so an unfinished recorder is work to wait for, not
    /// work a policy deadline may discard.
    pub(crate) fn quiesce(&mut self) -> Vec<WorkerResult<T>> {
        let owed = self.active.len();
        for (_, handle) in self.active.drain() {
            handle
                .join()
                .expect("the worker catches its recording panic before return");
        }
        (0..owed)
            .map(|_| {
                self.result_rx
                    .recv()
                    .expect("every joined recording worker publishes one terminal result")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{SubmissionId, TaskId};
    use std::sync::{Arc, Barrier};

    fn identity(id: u64) -> SubmissionIdentity {
        SubmissionIdentity {
            id: SubmissionId::new(id),
            task: TaskId::new(1),
        }
    }

    #[test]
    fn two_admitted_execs_record_at_the_same_time() {
        let mut workers = SubmissionWorkers::default();
        let wake = crate::runtime::host::WorkerWake::new(|| {});
        let rendezvous = Arc::new(Barrier::new(3));
        for id in [1, 2] {
            let worker_barrier = Arc::clone(&rendezvous);
            workers
                .dispatch(identity(id), id, wake.clone(), move |value| {
                    worker_barrier.wait();
                    value * 10
                })
                .unwrap();
        }

        rendezvous.wait();
        assert_eq!(workers.active.len(), 2);
        let mut results = workers.quiesce();
        results.sort_by_key(|result| result.identity.id.get());
        assert!(matches!(results[0].outcome, WorkerOutcome::Completed(10)));
        assert!(matches!(results[1].outcome, WorkerOutcome::Completed(20)));
    }

    #[test]
    fn duplicate_active_identity_is_refused_and_worker_panic_is_terminal() {
        let mut workers = SubmissionWorkers::default();
        let active = identity(3);
        let wake = crate::runtime::host::WorkerWake::new(|| {});
        workers
            .dispatch(active, (), wake.clone(), |_| panic!("recording failed"))
            .unwrap();
        assert_eq!(
            workers.dispatch(active, (), wake, |_| ()),
            Err(WorkerDispatchError::AlreadyActive(active))
        );

        let results = workers.quiesce();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].identity, active);
        assert!(matches!(results[0].outcome, WorkerOutcome::Panicked));
    }
}
