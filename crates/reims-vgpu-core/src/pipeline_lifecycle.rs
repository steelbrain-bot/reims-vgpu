//! Contract-owned pipeline translation, compilation, and native lifetime.
//!
//! Translation starts from the pipeline lifecycle, never from a draw. Exactly
//! one flight advances a declared identity. Dependent transactions receive a
//! ready lease or remain named waiters; unrelated work is not stalled.

use crate::NativeObjectLease;
use reims_vgpu_protocol::{ResourceId, TransactionId};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PipelineState {
    Declared,
    Translating,
    Compiling,
    Ready,
    Refused,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PipelineFailureStage {
    Translation,
    Compilation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineRefusal<E> {
    pub stage: PipelineFailureStage,
    pub reason: E,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PipelineLifecycleCensus {
    pub live: usize,
    pub readiness_hits: u64,
    pub readiness_misses: u64,
    pub duplicate_flight_attempts: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineVariantState {
    Compiling,
    Ready,
    Refused,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PipelineVariantCensus {
    pub live: usize,
    pub readiness_hits: u64,
    pub readiness_misses: u64,
    pub duplicate_flight_attempts: u64,
}

#[derive(Clone, Debug)]
pub enum PipelineVariantReadiness<N, E> {
    Ready(Arc<N>),
    Pending,
    Refused(E),
}

#[derive(Debug)]
pub enum PipelineVariantAdmission<K, N, E> {
    Ready(Arc<N>),
    Compile(PipelineVariantCompileJob<K>),
    Pending,
    Refused(E),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineVariantPublication<N> {
    pub native: Arc<N>,
    pub waiters: Box<[TransactionId]>,
}

pub type RetiredPipelineVariantWaiters<K> = Box<[(K, Box<[TransactionId]>)]>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineVariantLifecycleError {
    DuplicateVariant,
    UnknownVariant,
    WrongFamily,
    WrongState {
        expected: PipelineVariantState,
        actual: PipelineVariantState,
    },
}

#[derive(Debug)]
pub struct PipelineVariantCompileJob<K> {
    key: K,
    family: Arc<()>,
}

impl<K> PipelineVariantCompileJob<K> {
    pub const fn key(&self) -> &K {
        &self.key
    }
}

#[derive(Clone, Debug)]
struct PipelineVariant<N, E> {
    state: PipelineVariantState,
    native: Option<Arc<N>>,
    refusal: Option<E>,
    waiters: BTreeSet<TransactionId>,
}

/// Unbounded native variants owned by one contract pipeline lifetime.
///
/// A key may have exactly one compile flight. Ready values are retained until
/// the entire pipeline family retires, while acquired `Arc`s keep their native
/// variant alive through already-accepted work. No valid entry is evicted.
#[derive(Debug)]
pub struct PipelineVariantFamily<K, N, E> {
    family: Arc<()>,
    variants: BTreeMap<K, PipelineVariant<N, E>>,
    census: PipelineVariantCensus,
}

impl<K, N, E> Default for PipelineVariantFamily<K, N, E> {
    fn default() -> Self {
        Self {
            family: Arc::new(()),
            variants: BTreeMap::new(),
            census: PipelineVariantCensus::default(),
        }
    }
}

impl<K: Ord + Clone, N, E> PipelineVariantFamily<K, N, E> {
    /// Atomically acquire a ready variant, join its one compile flight, or
    /// become the owner of that flight when the key is absent.
    pub fn readiness_or_begin(
        &mut self,
        key: K,
        transaction: TransactionId,
    ) -> PipelineVariantAdmission<K, N, E>
    where
        E: Clone,
    {
        match self.variants.get_mut(&key) {
            Some(entry) => match entry.state {
                PipelineVariantState::Compiling => {
                    self.census.readiness_misses += 1;
                    entry.waiters.insert(transaction);
                    PipelineVariantAdmission::Pending
                }
                PipelineVariantState::Ready => {
                    self.census.readiness_hits += 1;
                    PipelineVariantAdmission::Ready(entry.native.as_ref().map(Arc::clone).unwrap())
                }
                PipelineVariantState::Refused => {
                    self.census.readiness_hits += 1;
                    PipelineVariantAdmission::Refused(entry.refusal.as_ref().unwrap().clone())
                }
            },
            None => {
                self.census.readiness_misses += 1;
                self.variants.insert(
                    key.clone(),
                    PipelineVariant {
                        state: PipelineVariantState::Compiling,
                        native: None,
                        refusal: None,
                        waiters: BTreeSet::from([transaction]),
                    },
                );
                self.census.live += 1;
                PipelineVariantAdmission::Compile(PipelineVariantCompileJob {
                    key,
                    family: Arc::clone(&self.family),
                })
            }
        }
    }

    pub fn begin_compile(
        &mut self,
        key: K,
    ) -> Result<PipelineVariantCompileJob<K>, PipelineVariantLifecycleError> {
        if self.variants.contains_key(&key) {
            self.census.duplicate_flight_attempts += 1;
            return Err(PipelineVariantLifecycleError::DuplicateVariant);
        }
        self.variants.insert(
            key.clone(),
            PipelineVariant {
                state: PipelineVariantState::Compiling,
                native: None,
                refusal: None,
                waiters: BTreeSet::new(),
            },
        );
        self.census.live += 1;
        Ok(PipelineVariantCompileJob {
            key,
            family: Arc::clone(&self.family),
        })
    }

    pub fn compile_complete(
        &mut self,
        job: PipelineVariantCompileJob<K>,
        native: N,
    ) -> Result<PipelineVariantPublication<N>, PipelineVariantLifecycleError> {
        let entry = self.job_entry(&job)?;
        let native = Arc::new(native);
        entry.native = Some(Arc::clone(&native));
        entry.state = PipelineVariantState::Ready;
        Ok(PipelineVariantPublication {
            native,
            waiters: take_variant_waiters(entry),
        })
    }

    pub fn refuse(
        &mut self,
        job: PipelineVariantCompileJob<K>,
        reason: E,
    ) -> Result<Box<[TransactionId]>, PipelineVariantLifecycleError> {
        let entry = self.job_entry(&job)?;
        entry.refusal = Some(reason);
        entry.state = PipelineVariantState::Refused;
        Ok(take_variant_waiters(entry))
    }

    pub fn readiness(
        &mut self,
        key: &K,
    ) -> Result<PipelineVariantReadiness<N, E>, PipelineVariantLifecycleError>
    where
        E: Clone,
    {
        let Some(entry) = self.variants.get(key) else {
            self.census.readiness_misses += 1;
            return Err(PipelineVariantLifecycleError::UnknownVariant);
        };
        match entry.state {
            PipelineVariantState::Compiling => {
                self.census.readiness_misses += 1;
                Ok(PipelineVariantReadiness::Pending)
            }
            PipelineVariantState::Ready => {
                self.census.readiness_hits += 1;
                Ok(PipelineVariantReadiness::Ready(
                    entry.native.as_ref().map(Arc::clone).unwrap(),
                ))
            }
            PipelineVariantState::Refused => {
                self.census.readiness_hits += 1;
                Ok(PipelineVariantReadiness::Refused(
                    entry.refusal.as_ref().unwrap().clone(),
                ))
            }
        }
    }

    pub const fn census(&self) -> PipelineVariantCensus {
        self.census
    }

    /// Drain every transaction waiting on a compile flight when the owning
    /// semantic pipeline lifetime retires. Ready/refused variants have no
    /// waiters and need no retirement record.
    pub fn retire_all_waiters(&mut self) -> RetiredPipelineVariantWaiters<K> {
        self.variants
            .iter_mut()
            .filter_map(|(key, entry)| {
                if entry.waiters.is_empty() {
                    None
                } else {
                    Some((key.clone(), take_variant_waiters(entry)))
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn job_entry(
        &mut self,
        job: &PipelineVariantCompileJob<K>,
    ) -> Result<&mut PipelineVariant<N, E>, PipelineVariantLifecycleError> {
        if !Arc::ptr_eq(&self.family, &job.family) {
            return Err(PipelineVariantLifecycleError::WrongFamily);
        }
        let entry = self
            .variants
            .get_mut(&job.key)
            .ok_or(PipelineVariantLifecycleError::UnknownVariant)?;
        if entry.state != PipelineVariantState::Compiling {
            return Err(PipelineVariantLifecycleError::WrongState {
                expected: PipelineVariantState::Compiling,
                actual: entry.state,
            });
        }
        Ok(entry)
    }
}

fn take_variant_waiters<N, E>(entry: &mut PipelineVariant<N, E>) -> Box<[TransactionId]> {
    std::mem::take(&mut entry.waiters)
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[derive(Debug)]
pub struct PipelineTranslationJob<Kind, C> {
    pipeline: ResourceId<Kind>,
    contract: Arc<C>,
    family: Arc<()>,
}

impl<Kind, C> PipelineTranslationJob<Kind, C> {
    pub const fn pipeline(&self) -> ResourceId<Kind> {
        self.pipeline
    }

    pub const fn contract(&self) -> &Arc<C> {
        &self.contract
    }
}

#[derive(Debug)]
pub struct PipelineCompileJob<Kind, T> {
    pipeline: ResourceId<Kind>,
    translated: T,
    family: Arc<()>,
}

impl<Kind, T> PipelineCompileJob<Kind, T> {
    pub const fn pipeline(&self) -> ResourceId<Kind> {
        self.pipeline
    }

    pub const fn translated(&self) -> &T {
        &self.translated
    }

    pub fn into_translated(self) -> T {
        self.translated
    }
}

pub type RetiredPipelines<Kind> = Box<[(ResourceId<Kind>, Box<[TransactionId]>)]>;

pub struct PipelineRetirement<Kind, N> {
    pub pipeline: ResourceId<Kind>,
    pub waiters: Box<[TransactionId]>,
    pub ready: Option<ReadyPipelineLease<Kind, N>>,
}

pub struct ReadyPipelineLease<Kind, N> {
    pub pipeline: ResourceId<Kind>,
    pub native_object: NativeObjectLease,
    pub native: Arc<N>,
}

impl<Kind, N: std::fmt::Debug> std::fmt::Debug for ReadyPipelineLease<Kind, N> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadyPipelineLease")
            .field("pipeline", &self.pipeline)
            .field("native_object", &self.native_object)
            .field("native", &self.native)
            .finish()
    }
}

impl<Kind, N> Clone for ReadyPipelineLease<Kind, N> {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline,
            native_object: self.native_object.clone(),
            native: Arc::clone(&self.native),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PipelineReadiness<N, E> {
    Ready(N),
    Pending,
    Refused(PipelineRefusal<E>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineLifecycleError {
    DuplicatePipeline,
    UnknownPipeline,
    WrongFamily,
    WrongState {
        expected: PipelineState,
        actual: PipelineState,
    },
}

#[derive(Clone, Debug)]
struct Pipeline<Kind, C, N, E> {
    contract: Arc<C>,
    state: PipelineState,
    ready: Option<ReadyPipelineLease<Kind, N>>,
    refusal: Option<PipelineRefusal<E>>,
    waiters: BTreeSet<TransactionId>,
}

#[derive(Clone, Debug)]
pub struct PipelineLifecycle<Kind, C, N, E> {
    family: Arc<()>,
    pipelines: BTreeMap<ResourceId<Kind>, Pipeline<Kind, C, N, E>>,
    census: PipelineLifecycleCensus,
}

impl<Kind, C, N, E> Default for PipelineLifecycle<Kind, C, N, E> {
    fn default() -> Self {
        Self {
            family: Arc::new(()),
            pipelines: BTreeMap::new(),
            census: PipelineLifecycleCensus::default(),
        }
    }
}

impl<Kind, C, N, E> PipelineLifecycle<Kind, C, N, E> {
    pub fn declare(
        &mut self,
        pipeline: ResourceId<Kind>,
        contract: C,
    ) -> Result<(), PipelineLifecycleError> {
        if self.pipelines.contains_key(&pipeline) {
            return Err(PipelineLifecycleError::DuplicatePipeline);
        }
        self.pipelines.insert(
            pipeline,
            Pipeline {
                contract: Arc::new(contract),
                state: PipelineState::Declared,
                ready: None,
                refusal: None,
                waiters: BTreeSet::new(),
            },
        );
        self.census.live += 1;
        Ok(())
    }

    pub fn begin_translation(
        &mut self,
        pipeline: ResourceId<Kind>,
    ) -> Result<PipelineTranslationJob<Kind, C>, PipelineLifecycleError> {
        let state = self.state_or_error(pipeline)?;
        if state != PipelineState::Declared {
            self.census.duplicate_flight_attempts += 1;
            return Err(PipelineLifecycleError::WrongState {
                expected: PipelineState::Declared,
                actual: state,
            });
        }
        let entry = self.pipelines.get_mut(&pipeline).unwrap();
        entry.state = PipelineState::Translating;
        Ok(PipelineTranslationJob {
            pipeline,
            contract: Arc::clone(&entry.contract),
            family: Arc::clone(&self.family),
        })
    }

    pub fn translation_complete<T>(
        &mut self,
        job: PipelineTranslationJob<Kind, C>,
        translated: T,
    ) -> Result<PipelineCompileJob<Kind, T>, PipelineLifecycleError> {
        self.validate_family(&job.family)?;
        let pipeline = job.pipeline;
        self.expect_state(pipeline, PipelineState::Translating)?
            .state = PipelineState::Compiling;
        Ok(PipelineCompileJob {
            pipeline,
            translated,
            family: Arc::clone(&self.family),
        })
    }

    pub fn compile_complete<T>(
        &mut self,
        job: PipelineCompileJob<Kind, T>,
        native_object: NativeObjectLease,
        native: N,
    ) -> Result<Box<[TransactionId]>, PipelineLifecycleError> {
        self.validate_family(&job.family)?;
        let pipeline = job.pipeline;
        let entry = self.expect_state(pipeline, PipelineState::Compiling)?;
        entry.ready = Some(ReadyPipelineLease {
            pipeline,
            native_object,
            native: Arc::new(native),
        });
        entry.state = PipelineState::Ready;
        Ok(take_waiters(entry))
    }

    pub fn refuse_translation(
        &mut self,
        job: PipelineTranslationJob<Kind, C>,
        reason: E,
    ) -> Result<Box<[TransactionId]>, PipelineLifecycleError> {
        self.validate_family(&job.family)?;
        let entry = self.expect_state(job.pipeline, PipelineState::Translating)?;
        entry.refusal = Some(PipelineRefusal {
            stage: PipelineFailureStage::Translation,
            reason,
        });
        entry.state = PipelineState::Refused;
        Ok(take_waiters(entry))
    }

    pub fn refuse_compilation<T>(
        &mut self,
        job: PipelineCompileJob<Kind, T>,
        reason: E,
    ) -> Result<Box<[TransactionId]>, PipelineLifecycleError> {
        self.validate_family(&job.family)?;
        let entry = self.expect_state(job.pipeline, PipelineState::Compiling)?;
        entry.refusal = Some(PipelineRefusal {
            stage: PipelineFailureStage::Compilation,
            reason,
        });
        entry.state = PipelineState::Refused;
        Ok(take_waiters(entry))
    }

    pub fn readiness(
        &mut self,
        pipeline: ResourceId<Kind>,
        transaction: TransactionId,
    ) -> Result<PipelineReadiness<ReadyPipelineLease<Kind, N>, E>, PipelineLifecycleError>
    where
        E: Clone,
    {
        match self.state_or_error(pipeline)? {
            PipelineState::Ready => {
                self.census.readiness_hits += 1;
                Ok(PipelineReadiness::Ready(
                    self.pipelines[&pipeline].ready.clone().unwrap(),
                ))
            }
            PipelineState::Refused => {
                self.census.readiness_hits += 1;
                Ok(PipelineReadiness::Refused(
                    self.pipelines[&pipeline].refusal.clone().unwrap(),
                ))
            }
            _ => {
                self.census.readiness_misses += 1;
                self.pipelines
                    .get_mut(&pipeline)
                    .unwrap()
                    .waiters
                    .insert(transaction);
                Ok(PipelineReadiness::Pending)
            }
        }
    }

    /// Observe published pipeline state without registering a transaction as a
    /// waiter or changing readiness census values.
    ///
    /// Semantic command projection uses this before transaction admission: it
    /// may clone immutable translated facts from a ready pipeline, but a
    /// pending observation must remain mutation-free so a later complete-EXEC
    /// retry is the only state transition.
    pub fn readiness_snapshot(
        &self,
        pipeline: ResourceId<Kind>,
    ) -> Result<PipelineReadiness<ReadyPipelineLease<Kind, N>, E>, PipelineLifecycleError>
    where
        E: Clone,
    {
        match self.state_or_error(pipeline)? {
            PipelineState::Ready => Ok(PipelineReadiness::Ready(
                self.pipelines[&pipeline].ready.clone().unwrap(),
            )),
            PipelineState::Refused => Ok(PipelineReadiness::Refused(
                self.pipelines[&pipeline].refusal.clone().unwrap(),
            )),
            _ => Ok(PipelineReadiness::Pending),
        }
    }

    /// Retires the exact contract lifetime and returns transactions waiting on it.
    ///
    /// Guest deletion is valid during every construction state. Removing the
    /// entry makes any late translation or compilation completion fail with
    /// `UnknownPipeline`, while ready leases already acquired by accepted work
    /// retain their native object independently.
    pub fn retire(
        &mut self,
        pipeline: ResourceId<Kind>,
    ) -> Result<Box<[TransactionId]>, PipelineLifecycleError> {
        let mut entry = self
            .pipelines
            .remove(&pipeline)
            .ok_or(PipelineLifecycleError::UnknownPipeline)?;
        self.census.live -= 1;
        Ok(take_waiters(&mut entry))
    }

    pub fn retire_owned(
        &mut self,
        pipeline: ResourceId<Kind>,
    ) -> Result<PipelineRetirement<Kind, N>, PipelineLifecycleError> {
        let mut entry = self
            .pipelines
            .remove(&pipeline)
            .ok_or(PipelineLifecycleError::UnknownPipeline)?;
        self.census.live -= 1;
        Ok(PipelineRetirement {
            pipeline,
            waiters: take_waiters(&mut entry),
            ready: entry.ready.take(),
        })
    }

    /// Retire every identity owned by a closing semantic generation.
    ///
    /// The returned identities let backend-owned children, such as constexpr
    /// samplers, retire through the same contract event. Ready leases already
    /// copied into accepted transactions remain independent of this registry.
    pub fn retire_all(&mut self) -> RetiredPipelines<Kind> {
        let retired = std::mem::take(&mut self.pipelines)
            .into_iter()
            .map(|(pipeline, mut entry)| (pipeline, take_waiters(&mut entry)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.census.live = 0;
        retired
    }

    pub fn retire_all_owned(&mut self) -> Box<[PipelineRetirement<Kind, N>]> {
        let retired = std::mem::take(&mut self.pipelines)
            .into_iter()
            .map(|(pipeline, mut entry)| PipelineRetirement {
                pipeline,
                waiters: take_waiters(&mut entry),
                ready: entry.ready.take(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.census.live = 0;
        retired
    }

    pub fn state(&self, pipeline: ResourceId<Kind>) -> Option<PipelineState> {
        self.pipelines.get(&pipeline).map(|entry| entry.state)
    }

    /// Snapshot every live identity's construction state for observability.
    /// The snapshot neither registers a waiter nor changes readiness census.
    pub fn states(&self) -> Box<[(ResourceId<Kind>, PipelineState)]> {
        self.pipelines
            .iter()
            .map(|(pipeline, entry)| (*pipeline, entry.state))
            .collect()
    }

    /// Snapshot exact typed refusals without consuming their lifecycle owner.
    pub fn refusals(&self) -> Box<[(ResourceId<Kind>, PipelineRefusal<E>)]>
    where
        E: Clone,
    {
        self.pipelines
            .iter()
            .filter_map(|(pipeline, entry)| {
                entry
                    .refusal
                    .as_ref()
                    .cloned()
                    .map(|refusal| (*pipeline, refusal))
            })
            .collect()
    }

    /// Immutable declaration retained by this exact pipeline generation.
    /// Translation, compilation, and readiness may advance without changing
    /// the contract that created the object.
    pub fn contract(&self, pipeline: ResourceId<Kind>) -> Option<Arc<C>> {
        self.pipelines
            .get(&pipeline)
            .map(|entry| Arc::clone(&entry.contract))
    }

    pub const fn census(&self) -> PipelineLifecycleCensus {
        self.census
    }

    fn state_or_error(
        &self,
        pipeline: ResourceId<Kind>,
    ) -> Result<PipelineState, PipelineLifecycleError> {
        self.state(pipeline)
            .ok_or(PipelineLifecycleError::UnknownPipeline)
    }

    fn expect_state(
        &mut self,
        pipeline: ResourceId<Kind>,
        expected: PipelineState,
    ) -> Result<&mut Pipeline<Kind, C, N, E>, PipelineLifecycleError> {
        let actual = self.state_or_error(pipeline)?;
        if actual != expected {
            return Err(PipelineLifecycleError::WrongState { expected, actual });
        }
        Ok(self.pipelines.get_mut(&pipeline).unwrap())
    }

    fn validate_family(&self, family: &Arc<()>) -> Result<(), PipelineLifecycleError> {
        if Arc::ptr_eq(&self.family, family) {
            Ok(())
        } else {
            Err(PipelineLifecycleError::WrongFamily)
        }
    }
}

fn take_waiters<Kind, C, N, E>(entry: &mut Pipeline<Kind, C, N, E>) -> Box<[TransactionId]> {
    std::mem::take(&mut entry.waiters)
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SessionGeneration, VulkanDeviceEpoch};
    use reims_vgpu_protocol::{RenderPipelineObject, SessionGenerationId, VulkanDeviceEpochId};

    fn pipeline(index: u32, generation: u32) -> ResourceId<RenderPipelineObject> {
        ResourceId::new(index, generation)
    }

    fn native_lease() -> NativeObjectLease {
        NativeObjectLease::acquire(
            &SessionGeneration::new(SessionGenerationId::new(1)),
            &VulkanDeviceEpoch::new(VulkanDeviceEpochId::new(2)),
        )
        .unwrap()
    }

    #[test]
    fn one_identity_has_exactly_one_flight() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let id = pipeline(1, 1);
        owner.declare(id, "contract").unwrap();
        assert_eq!(owner.contract(id).as_deref(), Some(&"contract"));
        let translation = owner.begin_translation(id).unwrap();
        assert_eq!(**translation.contract(), "contract");
        assert!(owner.begin_translation(id).is_err());
        let compile = owner
            .translation_complete(translation, "translated")
            .unwrap();
        assert_eq!(*compile.translated(), "translated");
        owner
            .compile_complete(compile, native_lease(), "native")
            .unwrap();
        assert_eq!(owner.contract(id).as_deref(), Some(&"contract"));
        assert_eq!(owner.census().duplicate_flight_attempts, 1);
    }

    #[test]
    fn slow_compile_releases_only_its_exact_waiters() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let slow = pipeline(1, 1);
        let ready = pipeline(2, 1);
        owner.declare(slow, "contract").unwrap();
        owner.declare(ready, "contract").unwrap();
        let slow_translation = owner.begin_translation(slow).unwrap();
        let ready_translation = owner.begin_translation(ready).unwrap();
        let ready_compile = owner.translation_complete(ready_translation, ()).unwrap();
        owner
            .compile_complete(ready_compile, native_lease(), "native")
            .unwrap();
        assert!(matches!(
            owner.readiness(slow, TransactionId::new(10)).unwrap(),
            PipelineReadiness::Pending
        ));
        assert!(matches!(
            owner.readiness(ready, TransactionId::new(11)).unwrap(),
            PipelineReadiness::Ready(_)
        ));
        let slow_compile = owner.translation_complete(slow_translation, ()).unwrap();
        assert_eq!(
            &*owner
                .compile_complete(slow_compile, native_lease(), "native")
                .unwrap(),
            &[TransactionId::new(10)]
        );
    }

    #[test]
    fn refusal_is_typed_and_wakes_all_exact_waiters() {
        let mut owner = PipelineLifecycle::<RenderPipelineObject, _, (), &'static str>::default();
        let id = pipeline(8, 1);
        owner.declare(id, "contract").unwrap();
        let translation = owner.begin_translation(id).unwrap();
        owner.readiness(id, TransactionId::new(2)).unwrap();
        owner.readiness(id, TransactionId::new(1)).unwrap();
        assert_eq!(
            &*owner
                .refuse_translation(translation, "invalid shader")
                .unwrap(),
            &[TransactionId::new(1), TransactionId::new(2)]
        );
        assert!(matches!(
            owner.readiness(id, TransactionId::new(3)).unwrap(),
            PipelineReadiness::Refused(PipelineRefusal {
                stage: PipelineFailureStage::Translation,
                reason: "invalid shader"
            })
        ));
        assert_eq!(&*owner.states(), &[(id, PipelineState::Refused)]);
        assert_eq!(
            &*owner.refusals(),
            &[(
                id,
                PipelineRefusal {
                    stage: PipelineFailureStage::Translation,
                    reason: "invalid shader",
                },
            )]
        );
    }

    #[test]
    fn acquired_native_lease_survives_object_retirement() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let id = pipeline(3, 1);
        owner.declare(id, "contract").unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(compile, native_lease(), "native")
            .unwrap();
        let lease = match owner.readiness(id, TransactionId::new(1)).unwrap() {
            PipelineReadiness::Ready(lease) => lease,
            _ => panic!("pipeline must be ready"),
        };
        assert!(owner.retire(id).unwrap().is_empty());
        assert_eq!(*lease.native, "native");
        assert_eq!(
            owner.readiness(id, TransactionId::new(2)).unwrap_err(),
            PipelineLifecycleError::UnknownPipeline
        );
        assert_eq!(owner.census().live, 0);
    }

    #[test]
    fn readiness_snapshot_neither_registers_a_waiter_nor_changes_the_census() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let id = pipeline(12, 1);
        owner.declare(id, "contract").unwrap();
        let before = owner.census();
        assert!(matches!(
            owner.readiness_snapshot(id).unwrap(),
            PipelineReadiness::Pending
        ));
        assert_eq!(owner.census(), before);

        let translation = owner.begin_translation(id).unwrap();
        assert!(owner
            .refuse_translation(translation, "invalid")
            .unwrap()
            .is_empty());
        assert!(matches!(
            owner.readiness_snapshot(id).unwrap(),
            PipelineReadiness::Refused(PipelineRefusal {
                stage: PipelineFailureStage::Translation,
                reason: "invalid"
            })
        ));
        assert_eq!(owner.census(), before);
    }

    #[test]
    fn slot_reuse_cannot_observe_or_retire_an_older_generation() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let old = pipeline(3, 1);
        let reused = pipeline(3, 2);
        owner.declare(old, "old").unwrap();
        let translation = owner.begin_translation(old).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(compile, native_lease(), "old-native")
            .unwrap();
        owner.declare(reused, "new").unwrap();

        assert_eq!(owner.state(old), Some(PipelineState::Ready));
        assert_eq!(owner.state(reused), Some(PipelineState::Declared));
        assert!(owner.retire(reused).unwrap().is_empty());
        assert_eq!(owner.state(old), Some(PipelineState::Ready));
    }

    #[test]
    fn retirement_during_translation_wakes_waiters_and_rejects_late_completion() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let id = pipeline(4, 1);
        owner.declare(id, "contract").unwrap();
        let translation = owner.begin_translation(id).unwrap();
        owner.readiness(id, TransactionId::new(7)).unwrap();
        owner.readiness(id, TransactionId::new(3)).unwrap();

        assert_eq!(
            &*owner.retire(id).unwrap(),
            &[TransactionId::new(3), TransactionId::new(7)]
        );
        assert!(matches!(
            owner.translation_complete(translation, "translated"),
            Err(PipelineLifecycleError::UnknownPipeline)
        ));
        assert_eq!(owner.census().live, 0);
    }

    #[test]
    fn retirement_during_compilation_rejects_late_native_publication() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let id = pipeline(5, 1);
        owner.declare(id, "contract").unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner
            .translation_complete(translation, "translated")
            .unwrap();
        owner.readiness(id, TransactionId::new(9)).unwrap();

        assert_eq!(&*owner.retire(id).unwrap(), &[TransactionId::new(9)]);
        assert_eq!(
            owner
                .compile_complete(compile, native_lease(), "native")
                .unwrap_err(),
            PipelineLifecycleError::UnknownPipeline
        );
        assert_eq!(owner.census().live, 0);
    }

    #[test]
    fn translation_jobs_cannot_publish_into_another_pipeline_owner() {
        let mut first = PipelineLifecycle::<RenderPipelineObject, _, (), &'static str>::default();
        let mut second = PipelineLifecycle::<RenderPipelineObject, _, (), &'static str>::default();
        let id = pipeline(6, 1);
        first.declare(id, "first").unwrap();
        second.declare(id, "second").unwrap();
        let job = first.begin_translation(id).unwrap();
        second.begin_translation(id).unwrap();
        assert!(matches!(
            second.translation_complete(job, ()),
            Err(PipelineLifecycleError::WrongFamily)
        ));
        assert_eq!(second.state(id), Some(PipelineState::Translating));
    }

    #[test]
    fn generation_retirement_returns_every_identity_and_exact_waiter_set() {
        let mut owner =
            PipelineLifecycle::<RenderPipelineObject, _, &'static str, &'static str>::default();
        let first = pipeline(2, 4);
        let second = pipeline(8, 3);
        owner.declare(second, "second").unwrap();
        owner.declare(first, "first").unwrap();
        owner.readiness(first, TransactionId::new(7)).unwrap();
        owner.readiness(second, TransactionId::new(9)).unwrap();
        owner.readiness(second, TransactionId::new(8)).unwrap();

        let retired = owner.retire_all();
        assert_eq!(retired.len(), 2);
        assert_eq!(retired[0].0, first);
        assert_eq!(&*retired[0].1, &[TransactionId::new(7)]);
        assert_eq!(retired[1].0, second);
        assert_eq!(
            &*retired[1].1,
            &[TransactionId::new(8), TransactionId::new(9)]
        );
        assert_eq!(owner.census().live, 0);
    }

    #[test]
    fn variant_family_has_one_exact_flight_and_never_evicts_live_variants() {
        let mut family = PipelineVariantFamily::<u32, String, &'static str>::default();
        let first = family.begin_compile(7).unwrap();
        assert_eq!(first.key(), &7);
        assert_eq!(
            family.begin_compile(7).unwrap_err(),
            PipelineVariantLifecycleError::DuplicateVariant
        );
        assert!(matches!(
            family.readiness(&7).unwrap(),
            PipelineVariantReadiness::Pending
        ));
        let retained = family
            .compile_complete(first, "seven".to_string())
            .unwrap()
            .native;
        for key in 0..1024 {
            if key == 7 {
                continue;
            }
            let job = family.begin_compile(key).unwrap();
            family.compile_complete(job, key.to_string()).unwrap();
        }
        assert_eq!(family.census().live, 1024);
        assert!(matches!(
            family.readiness(&7).unwrap(),
            PipelineVariantReadiness::Ready(native) if Arc::ptr_eq(&native, &retained)
        ));
        drop(family);
        assert_eq!(retained.as_str(), "seven");
    }

    #[test]
    fn variant_compile_jobs_cannot_cross_pipeline_families() {
        let mut first = PipelineVariantFamily::<u32, (), ()>::default();
        let mut second = PipelineVariantFamily::<u32, (), ()>::default();
        let job = first.begin_compile(1).unwrap();
        second.begin_compile(1).unwrap();
        assert_eq!(
            second.compile_complete(job, ()),
            Err(PipelineVariantLifecycleError::WrongFamily)
        );
        assert!(matches!(
            second.readiness(&1).unwrap(),
            PipelineVariantReadiness::Pending
        ));
    }

    #[test]
    fn atomic_variant_admission_creates_one_flight_and_joins_every_later_ask() {
        let mut family = PipelineVariantFamily::<u32, &'static str, &'static str>::default();
        let job = match family.readiness_or_begin(3, TransactionId::new(7)) {
            PipelineVariantAdmission::Compile(job) => job,
            _ => panic!("the first ask must own the compile flight"),
        };
        assert!(matches!(
            family.readiness_or_begin(3, TransactionId::new(8)),
            PipelineVariantAdmission::Pending
        ));
        let publication = family.compile_complete(job, "native").unwrap();
        assert_eq!(
            &*publication.waiters,
            &[TransactionId::new(7), TransactionId::new(8)]
        );
        assert!(matches!(
            family.readiness_or_begin(3, TransactionId::new(9)),
            PipelineVariantAdmission::Ready(native) if *native == "native"
        ));
        assert_eq!(family.census().live, 1);
        assert_eq!(family.census().duplicate_flight_attempts, 0);
    }

    #[test]
    fn variant_family_retirement_releases_every_exact_compile_waiter() {
        let mut family = PipelineVariantFamily::<u32, (), ()>::default();
        assert!(matches!(
            family.readiness_or_begin(4, TransactionId::new(9)),
            PipelineVariantAdmission::Compile(_)
        ));
        assert!(matches!(
            family.readiness_or_begin(4, TransactionId::new(7)),
            PipelineVariantAdmission::Pending
        ));
        let ready_job = family.begin_compile(8).unwrap();
        family.compile_complete(ready_job, ()).unwrap();

        let retired = family.retire_all_waiters();
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].0, 4);
        assert_eq!(
            &*retired[0].1,
            &[TransactionId::new(7), TransactionId::new(9)]
        );
        assert!(family.retire_all_waiters().is_empty());
    }
}
