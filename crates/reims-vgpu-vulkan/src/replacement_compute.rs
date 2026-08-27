//! Vulkan projection and recording for prepared compute dispatches.

use crate::{
    replacement_barrier_record::record_hazard_barriers,
    replacement_buffer_blit::{NativeBufferTarget, ReplacementBufferResolver},
    replacement_image_state::{
        PreparedImageState, ReplacementImageKey, ReplacementImageStateError,
        ReplacementImageStateOwner, ReplacementImageUse,
    },
    replacement_image_transition::{
        resolve_image_transitions, ImageTransitionResolveError, PreparedNativeImageState,
        ReplacementImageResolver,
    },
    replacement_sampler::ReplacementSamplerLease,
};
use ash::vk;
use reims_vgpu_core::{
    BackingView, ComputeBindingClass, ComputeBindingView, PreparedComputeDispatch,
    ResolvedComputeDispatch, ResolvedComputeLaunch, ResolvedResourceCompletion, ViewRepresentation,
};
use reims_vgpu_protocol::{BackingId, RepresentationId};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    /// Offset of the translated entry point's exact three-word thread grid.
    /// `None` means reflection proved the module needs no culling grid.
    pub thread_grid_push_offset: Option<u32>,
}

pub trait ReplacementComputePipelineDevice: Send + Sync {
    fn destroy_compute_pipeline_variant(&self, native: &ReplacementComputePipeline);
}

impl ReplacementComputePipelineDevice for crate::engine::context::SharedDeviceContext {
    fn destroy_compute_pipeline_variant(&self, native: &ReplacementComputePipeline) {
        unsafe {
            self.device.destroy_pipeline(native.pipeline, None);
            self.device.destroy_pipeline_layout(native.layout, None);
            self.device
                .destroy_descriptor_set_layout(native.descriptor_set_layout, None);
        }
    }
}

/// Owned realization of one translated compute program on one device epoch.
pub struct ReplacementComputePipelineVariant {
    native: ReplacementComputePipeline,
    context: Option<Arc<dyn ReplacementComputePipelineDevice>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplacementComputePipelinePlan {
    pub program: reims_vgpu_core::PreparedShaderStage,
    /// Binding, Vulkan descriptor-type ordinal, and descriptor count.
    pub descriptors: Box<[(u32, i32, u32)]>,
}

impl ReplacementComputePipelinePlan {
    pub fn variant_key(&self) -> ReplacementComputePipelineVariantKey {
        ReplacementComputePipelineVariantKey {
            program: self.program.id,
            descriptors: self.descriptors.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ReplacementComputePipelineVariantKey {
    pub program: reims_vgpu_protocol::PreparedShaderId,
    pub descriptors: Box<[(u32, i32, u32)]>,
}

pub type ReplacementComputePipelineVariants<E> = reims_vgpu_core::PipelineVariantFamily<
    ReplacementComputePipelineVariantKey,
    ReplacementComputePipelineVariant,
    E,
>;

#[derive(Debug)]
pub struct ReplacementComputePipelineFamily<E> {
    variants: parking_lot::Mutex<ReplacementComputePipelineVariants<E>>,
}

impl<E> Default for ReplacementComputePipelineFamily<E> {
    fn default() -> Self {
        Self {
            variants: parking_lot::Mutex::new(ReplacementComputePipelineVariants::default()),
        }
    }
}

impl<E> ReplacementComputePipelineFamily<E> {
    pub fn readiness_or_begin(
        &self,
        key: ReplacementComputePipelineVariantKey,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> reims_vgpu_core::PipelineVariantAdmission<
        ReplacementComputePipelineVariantKey,
        ReplacementComputePipelineVariant,
        E,
    >
    where
        E: Clone,
    {
        self.variants.lock().readiness_or_begin(key, transaction)
    }

    pub fn begin_compile(
        &self,
        key: ReplacementComputePipelineVariantKey,
    ) -> Result<
        reims_vgpu_core::PipelineVariantCompileJob<ReplacementComputePipelineVariantKey>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    > {
        self.variants.lock().begin_compile(key)
    }

    pub fn compile_complete(
        &self,
        job: reims_vgpu_core::PipelineVariantCompileJob<ReplacementComputePipelineVariantKey>,
        native: ReplacementComputePipelineVariant,
    ) -> Result<
        reims_vgpu_core::PipelineVariantPublication<ReplacementComputePipelineVariant>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    > {
        self.variants.lock().compile_complete(job, native)
    }

    pub fn refuse(
        &self,
        job: reims_vgpu_core::PipelineVariantCompileJob<ReplacementComputePipelineVariantKey>,
        reason: E,
    ) -> Result<
        Box<[reims_vgpu_protocol::TransactionId]>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    > {
        self.variants.lock().refuse(job, reason)
    }

    pub fn readiness(
        &self,
        key: &ReplacementComputePipelineVariantKey,
    ) -> Result<
        reims_vgpu_core::PipelineVariantReadiness<ReplacementComputePipelineVariant, E>,
        reims_vgpu_core::PipelineVariantLifecycleError,
    >
    where
        E: Clone,
    {
        self.variants.lock().readiness(key)
    }

    pub fn census(&self) -> reims_vgpu_core::PipelineVariantCensus {
        self.variants.lock().census()
    }

    pub fn retire_all_waiters(
        &self,
    ) -> reims_vgpu_core::RetiredPipelineVariantWaiters<ReplacementComputePipelineVariantKey> {
        self.variants.lock().retire_all_waiters()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementComputePipelinePlanError {
    DescriptorCollision(u32),
    DescriptorCountZero(u32),
    UsedDescriptorMissing(u32),
}

pub fn resolve_compute_pipeline_plan(
    operation: &ResolvedComputeDispatch,
    program: reims_vgpu_core::PreparedShaderStage,
) -> Result<ReplacementComputePipelinePlan, ReplacementComputePipelinePlanError> {
    let mut descriptors = BTreeMap::<u32, (vk::DescriptorType, u32)>::new();
    let mut declare = |binding, ty, count| {
        if count == 0 {
            return Err(ReplacementComputePipelinePlanError::DescriptorCountZero(
                binding,
            ));
        }
        match descriptors.insert(binding, (ty, count)) {
            Some(previous) if previous != (ty, count) => Err(
                ReplacementComputePipelinePlanError::DescriptorCollision(binding),
            ),
            _ => Ok(()),
        }
    };
    for resource in operation.resources.iter() {
        declare(
            resource.binding,
            descriptor_type(resource.class),
            resource.descriptor_count,
        )?;
    }
    for sampler in operation.samplers.iter() {
        declare(
            sampler.binding,
            vk::DescriptorType::SAMPLER,
            sampler.descriptor_count,
        )?;
    }
    for null in operation.null_bindings.iter() {
        declare(
            null.binding,
            descriptor_type(null.class),
            null.descriptor_count,
        )?;
    }
    for binding in program.used_descriptor_bindings.iter().copied() {
        if !descriptors.contains_key(&binding) {
            return Err(ReplacementComputePipelinePlanError::UsedDescriptorMissing(
                binding,
            ));
        }
    }
    Ok(ReplacementComputePipelinePlan {
        program,
        descriptors: descriptors
            .into_iter()
            .map(|(binding, (ty, count))| (binding, ty.as_raw(), count))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

fn descriptor_type(class: ComputeBindingClass) -> vk::DescriptorType {
    match class {
        ComputeBindingClass::Buffer => vk::DescriptorType::STORAGE_BUFFER,
        ComputeBindingClass::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        ComputeBindingClass::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
    }
}

impl std::fmt::Debug for ReplacementComputePipelineVariant {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplacementComputePipelineVariant")
            .field("native", &self.native)
            .field("owns_context", &self.context.is_some())
            .finish()
    }
}

impl ReplacementComputePipelineVariant {
    pub fn new(
        context: Arc<dyn ReplacementComputePipelineDevice>,
        native: ReplacementComputePipeline,
    ) -> Self {
        Self {
            native,
            context: Some(context),
        }
    }

    pub const fn native(&self) -> &ReplacementComputePipeline {
        &self.native
    }

    #[cfg(test)]
    pub(crate) const fn synthetic(native: ReplacementComputePipeline) -> Self {
        Self {
            native,
            context: None,
        }
    }
}

impl std::ops::Deref for ReplacementComputePipelineVariant {
    type Target = ReplacementComputePipeline;

    fn deref(&self) -> &Self::Target {
        &self.native
    }
}

impl Drop for ReplacementComputePipelineVariant {
    fn drop(&mut self) {
        let Some(context) = self.context.as_ref() else {
            return;
        };
        context.destroy_compute_pipeline_variant(&self.native);
    }
}

pub trait ReplacementComputeResolver: ReplacementBufferResolver + ReplacementImageResolver {
    fn resolve_sampler(
        &self,
        pipeline: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ComputePipelineObject>,
        sampler: &reims_vgpu_core::SamplerResource,
    ) -> Option<ReplacementSamplerLease>;

    fn max_storage_buffer_range(&self) -> u64;

    fn min_storage_buffer_offset_alignment(&self) -> u64;

    fn null_descriptors(&self) -> bool;
}

mod compute_image_bindings_sealed {
    pub trait Sealed {}

    impl Sealed for () {}
    impl Sealed for reims_vgpu_core::ResolvedComputeDispatch {}
}

/// One image a compute dispatch binds, named by the texture as well as the
/// backing.
///
/// A backing carries an image for each texture declared over its range, so the
/// backing alone does not say which image a binding means.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementComputeImageBinding {
    pub backing: BackingId,
    /// The texture whose image this binding names. A view binds its base's
    /// image, so this is never the view the guest named.
    pub owner: reims_vgpu_core::ImageOwner,
    pub usage: vk::ImageUsageFlags,
    pub storage: bool,
}

pub trait ReplacementComputeImageBindings: compute_image_bindings_sealed::Sealed {
    fn image_bindings(&self) -> Box<[ReplacementComputeImageBinding]>;
}

impl ReplacementComputeImageBindings for () {
    fn image_bindings(&self) -> Box<[ReplacementComputeImageBinding]> {
        Box::new([])
    }
}

impl ReplacementComputeImageBindings for ResolvedComputeDispatch {
    fn image_bindings(&self) -> Box<[ReplacementComputeImageBinding]> {
        self.resources
            .iter()
            .filter_map(|resource| {
                let (usage, storage) = match resource.class {
                    ComputeBindingClass::Buffer => return None,
                    ComputeBindingClass::SampledImage => (vk::ImageUsageFlags::SAMPLED, false),
                    ComputeBindingClass::StorageImage => (vk::ImageUsageFlags::STORAGE, true),
                };
                let ComputeBindingView::Image(view) = resource.view else {
                    return None;
                };
                Some(ReplacementComputeImageBinding {
                    backing: resource.backing,
                    owner: reims_vgpu_core::ImageOwner::of_view(view),
                    usage,
                    storage,
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeComputeDescriptor {
    StorageBuffer {
        binding: u32,
        array_element: u32,
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    },
    Sampler {
        binding: u32,
        array_element: u32,
        sampler: vk::Sampler,
    },
    Image {
        binding: u32,
        array_element: u32,
        descriptor_type: vk::DescriptorType,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    },
}

#[derive(Clone, Debug)]
pub struct NativeComputeDispatch {
    pub pipeline: Arc<ReplacementComputePipelineVariant>,
    pub wait_for_prior_commands: bool,
    pub descriptors: Box<[NativeComputeDescriptor]>,
    pub sampler_leases: Box<[ReplacementSamplerLease]>,
    pub descriptor_counts: Box<[(vk::DescriptorType, u32)]>,
    pub launch: NativeComputeLaunch,
    pub image_state: Option<PreparedNativeImageState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeComputeLaunch {
    Direct {
        thread_grid: [u32; 3],
        workgroups: [u32; 3],
    },
    IndirectThreadgroups {
        buffer: vk::Buffer,
        offset: u64,
    },
}

pub fn derive_compute_image_uses<NativePipeline, Operation: ReplacementComputeImageBindings>(
    prepared: &PreparedComputeDispatch<NativePipeline, Operation>,
) -> Result<Box<[ReplacementImageUse]>, ComputeImageStateError> {
    let representations = prepared.representations();
    let mut images = BTreeMap::<ReplacementImageKey, (vk::ImageUsageFlags, bool)>::new();
    for binding in prepared.operation().image_bindings() {
        let ReplacementComputeImageBinding {
            backing,
            owner,
            usage,
            storage,
        } = binding;
        // An image binding names one texture's image over the backing. A
        // backing bound once as a buffer and once as a texture holds both
        // objects, and two textures declared over one range hold one image
        // each -- only the one this binding names has its image state.
        let representation =
            ViewRepresentation::lookup(representations, backing, BackingView::Image(owner))
                .ok_or(ComputeImageStateError::RepresentationUseMismatch(backing))?;
        let image = ReplacementImageKey {
            backing,
            representation,
        };
        let entry = images.entry(image).or_default();
        entry.0 |= usage;
        entry.1 |= storage;
    }
    Ok(images
        .into_iter()
        .map(|(image, (required_usage, storage))| {
            let layout = if storage {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            ReplacementImageUse {
                image,
                required_usage,
                use_layout: layout,
                final_layout: layout,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

pub fn prepare_compute_image_state<NativePipeline, Operation: ReplacementComputeImageBindings>(
    owner: &mut ReplacementImageStateOwner,
    prepared: &PreparedComputeDispatch<NativePipeline, Operation>,
    queue_family: u32,
) -> Result<Option<PreparedImageState>, ComputeImageStateError> {
    let uses = derive_compute_image_uses(prepared)?;
    if uses.is_empty() {
        return Ok(None);
    }
    owner
        .prepare_operation(
            prepared.transaction(),
            prepared.operation_index(),
            queue_family,
            uses,
        )
        .map(Some)
        .map_err(ComputeImageStateError::State)
}

pub fn validate_compute_image_state<NativePipeline, Operation: ReplacementComputeImageBindings>(
    prepared: &PreparedComputeDispatch<NativePipeline, Operation>,
    state: Option<&PreparedImageState>,
) -> Result<(), ComputeImageStateError> {
    let uses = derive_compute_image_uses(prepared)?;
    let Some(state) = state else {
        return uses
            .is_empty()
            .then_some(())
            .ok_or(ComputeImageStateError::ImageStateRequired);
    };
    if uses.is_empty() {
        return Err(ComputeImageStateError::UnexpectedImageState);
    }
    if state.transaction() != prepared.transaction()
        || state.operation_index() != Some(prepared.operation_index())
        || state.transitions().len() != uses.len()
    {
        return Err(ComputeImageStateError::StateOperationMismatch);
    }
    for use_ in uses {
        let transition = state
            .transitions()
            .iter()
            .find(|transition| transition.image == use_.image)
            .ok_or(ComputeImageStateError::StateOperationMismatch)?;
        if transition.required_usage != use_.required_usage
            || transition.use_layout != use_.use_layout
            || transition.final_layout != use_.final_layout
        {
            return Err(ComputeImageStateError::StateOperationMismatch);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeImageStateError {
    RepresentationUseMismatch(BackingId),
    ImageStateRequired,
    UnexpectedImageState,
    StateOperationMismatch,
    State(ReplacementImageStateError),
}

#[derive(Clone, Debug)]
pub struct ReplacementComputeProgram<Operation = ResolvedComputeDispatch> {
    index: usize,
    transaction: reims_vgpu_protocol::TransactionId,
    operation: Operation,
    native: NativeComputeDispatch,
    backings: Box<[BackingId]>,
    completions: Box<[ResolvedResourceCompletion]>,
}

pub fn resolve_exec_compute_programs<
    Render: crate::replacement_render::ReplacementRenderImageBindings,
    NativeRender,
>(
    resources: &reims_vgpu_core::PreparedExecResources<
        ResolvedComputeDispatch,
        ReplacementComputePipelineVariant,
        Render,
        NativeRender,
    >,
    states: Option<&crate::replacement_image_state::PreparedImageStateBatch>,
    resolver: &impl ReplacementComputeResolver,
) -> Result<Box<[ReplacementComputeProgram]>, ComputeExecProgramError> {
    let has_images = resources
        .inputs()
        .compute_dispatches
        .iter()
        .map(derive_compute_image_uses)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|uses| !uses.is_empty());
    let exec_has_images = crate::replacement_exec_image::exec_has_image_uses(resources)
        .map_err(ComputeExecProgramError::ExecImageState)?;
    if let Some(states) = states {
        crate::replacement_exec_image::validate_exec_image_states(resources, states)
            .map_err(ComputeExecProgramError::ExecImageState)?;
    }
    validate_compute_image_state_batch_presence(
        has_images,
        exec_has_images || states.is_some(),
        states.is_some(),
    )?;
    resources
        .inputs()
        .compute_dispatches
        .iter()
        .map(|prepared| {
            let state = states.and_then(|states| {
                states
                    .operations()
                    .iter()
                    .find(|state| state.operation_index() == Some(prepared.operation_index()))
            });
            ReplacementComputeProgram::resolve_with_image_state(prepared, state, resolver)
                .map_err(ComputeExecProgramError::Record)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn validate_compute_image_state_batch_presence(
    compute_has_images: bool,
    exec_has_images: bool,
    states_present: bool,
) -> Result<(), ComputeExecProgramError> {
    match (compute_has_images, exec_has_images, states_present) {
        (true, _, false) => Err(ComputeExecProgramError::ImageStateBatchMissing),
        (_, false, true) => Err(ComputeExecProgramError::UnexpectedImageStateBatch),
        _ => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeExecProgramError {
    ImageState(ComputeImageStateError),
    ImageStateBatchMissing,
    UnexpectedImageStateBatch,
    ExecImageState(crate::replacement_exec_image::ExecImageStateError),
    Record(ComputeRecordError),
}

impl ComputeExecProgramError {
    /// Whether a later packet could make this program record.
    ///
    /// See [`crate::replacement_image_transition::TextureBindingViewDecline::is_unimplemented`].
    pub const fn is_terminal_refusal(&self) -> bool {
        matches!(
            self,
            Self::Record(ComputeRecordError::UnknownImageView { reason, .. }) if reason.is_unimplemented()
        )
    }
}

impl From<ComputeImageStateError> for ComputeExecProgramError {
    fn from(reason: ComputeImageStateError) -> Self {
        Self::ImageState(reason)
    }
}

impl<Operation> ReplacementComputeProgram<Operation> {
    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        self.transaction
    }

    pub const fn operation(&self) -> &Operation {
        &self.operation
    }

    pub const fn native(&self) -> &NativeComputeDispatch {
        &self.native
    }

    pub const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    pub const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }
}

#[cfg(test)]
impl<Operation> ReplacementComputeProgram<Operation> {
    pub(crate) fn synthetic(
        index: usize,
        transaction: reims_vgpu_protocol::TransactionId,
        operation: Operation,
        native: NativeComputeDispatch,
        backings: impl Into<Box<[BackingId]>>,
        completions: impl Into<Box<[ResolvedResourceCompletion]>>,
    ) -> Self {
        Self {
            index,
            transaction,
            operation,
            native,
            backings: backings.into(),
            completions: completions.into(),
        }
    }
}

impl ReplacementComputeProgram<ResolvedComputeDispatch> {
    pub fn resolve(
        prepared: &PreparedComputeDispatch<ReplacementComputePipelineVariant>,
        resolver: &impl ReplacementComputeResolver,
    ) -> Result<Self, ComputeRecordError> {
        Self::resolve_with_image_state(prepared, None, resolver)
    }

    pub fn resolve_with_image_state(
        prepared: &PreparedComputeDispatch<ReplacementComputePipelineVariant>,
        image_state: Option<&PreparedImageState>,
        resolver: &impl ReplacementComputeResolver,
    ) -> Result<Self, ComputeRecordError> {
        validate_compute_image_state(prepared, image_state)
            .map_err(ComputeRecordError::ImageState)?;
        let native_image_state = image_state
            .map(|state| resolve_image_transitions(state, resolver))
            .transpose()
            .map_err(ComputeRecordError::ImageTransition)?;
        if native_image_state
            .as_ref()
            .is_some_and(|state| !state.releases.is_empty())
        {
            return Err(ComputeRecordError::ImageReleasePending);
        }
        if !prepared
            .pipeline()
            .native_object
            .native_handles_are_usable()
        {
            return Err(ComputeRecordError::DeviceEpochLost);
        }
        let pipeline = prepared.pipeline().native.clone();
        if pipeline.pipeline == vk::Pipeline::null()
            || pipeline.layout == vk::PipelineLayout::null()
            || pipeline.descriptor_set_layout == vk::DescriptorSetLayout::null()
        {
            return Err(ComputeRecordError::InvalidPipeline);
        }
        let representations = prepared.representations();
        let mut descriptors = Vec::new();
        let mut declarations = BTreeMap::<u32, (vk::DescriptorType, u32)>::new();
        for resource in prepared.operation().resources.iter() {
            let descriptor_type = match resource.class {
                ComputeBindingClass::Buffer => vk::DescriptorType::STORAGE_BUFFER,
                ComputeBindingClass::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
                ComputeBindingClass::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            };
            declare_descriptor(
                &mut declarations,
                resource.binding,
                descriptor_type,
                resource.descriptor_count,
            )?;
            // The descriptor class picks the view, so a backing bound as both
            // a storage buffer and a texture resolves to the right native
            // object at each binding rather than to whichever one execution
            // designated.
            let view = match resource.view {
                ComputeBindingView::Buffer(_) => BackingView::Bytes,
                ComputeBindingView::Image(view) => {
                    BackingView::Image(reims_vgpu_core::ImageOwner::of_view(view))
                }
            };
            let representation =
                ViewRepresentation::lookup(representations, resource.backing, view).ok_or(
                    ComputeRecordError::RepresentationUseMismatch(resource.backing),
                )?;
            match (resource.class, resource.view) {
                (ComputeBindingClass::Buffer, ComputeBindingView::Buffer(range)) => {
                    let target = resolver
                        .resolve_buffer(resource.backing, representation)
                        .ok_or(ComputeRecordError::UnknownBuffer {
                            backing: resource.backing,
                            representation,
                        })?;
                    validate_buffer_target(resource.backing, range, target, resolver)?;
                    descriptors.push(NativeComputeDescriptor::StorageBuffer {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        buffer: target.buffer,
                        offset: target
                            .base_offset
                            .checked_add(range.start())
                            .ok_or(ComputeRecordError::BufferAddressOverflow(resource.backing))?,
                        range: range.end() - range.start(),
                    });
                }
                (
                    class @ (ComputeBindingClass::SampledImage | ComputeBindingClass::StorageImage),
                    ComputeBindingView::Image(view),
                ) => {
                    let image = ReplacementImageKey {
                        backing: resource.backing,
                        representation,
                    };
                    let target =
                        resolver
                            .resolve_texture_binding_view(image, view)
                            .map_err(|reason| ComputeRecordError::UnknownImageView {
                                image,
                                resource: view.resource,
                                reason,
                            })?;
                    if target.view == vk::ImageView::null() {
                        return Err(ComputeRecordError::MissingImageView(image));
                    }
                    let required_usage = match class {
                        ComputeBindingClass::SampledImage => vk::ImageUsageFlags::SAMPLED,
                        ComputeBindingClass::StorageImage => vk::ImageUsageFlags::STORAGE,
                        ComputeBindingClass::Buffer => unreachable!(),
                    };
                    if !target.usage.contains(required_usage) {
                        return Err(ComputeRecordError::MissingImageUsage {
                            image,
                            required: required_usage,
                        });
                    }
                    let prepared_layout = image_state
                        .and_then(|state| {
                            state
                                .transitions()
                                .iter()
                                .find(|transition| transition.image == image)
                        })
                        .map(|transition| transition.use_layout)
                        .ok_or(ComputeRecordError::ImageState(
                            ComputeImageStateError::StateOperationMismatch,
                        ))?;
                    descriptors.push(NativeComputeDescriptor::Image {
                        binding: resource.binding,
                        array_element: resource.array_element,
                        descriptor_type,
                        view: target.view,
                        layout: prepared_layout,
                    });
                }
                _ => {
                    return Err(ComputeRecordError::BindingViewMismatch {
                        binding: resource.binding,
                    });
                }
            }
        }
        let mut sampler_leases = Vec::with_capacity(prepared.operation().samplers.len());
        for sampler in prepared.operation().samplers.iter() {
            declare_descriptor(
                &mut declarations,
                sampler.binding,
                vk::DescriptorType::SAMPLER,
                sampler.descriptor_count,
            )?;
            let native = if sampler.sampler.source == reims_vgpu_core::SamplerSource::Null {
                if !resolver.null_descriptors() {
                    return Err(ComputeRecordError::NullDescriptorUnavailable(
                        sampler.binding,
                    ));
                }
                vk::Sampler::null()
            } else {
                let lease = resolver
                    .resolve_sampler(prepared.operation().pipeline, &sampler.sampler)
                    .filter(|sampler| sampler.handle() != vk::Sampler::null())
                    .ok_or(ComputeRecordError::UnknownSampler {
                        binding: sampler.binding,
                    })?;
                let native = lease.handle();
                sampler_leases.push(lease);
                native
            };
            descriptors.push(NativeComputeDescriptor::Sampler {
                binding: sampler.binding,
                array_element: sampler.array_element,
                sampler: native,
            });
        }
        for null in prepared.operation().null_bindings.iter() {
            if !resolver.null_descriptors() {
                return Err(ComputeRecordError::NullDescriptorUnavailable(null.binding));
            }
            let descriptor_type = match null.class {
                ComputeBindingClass::Buffer => vk::DescriptorType::STORAGE_BUFFER,
                ComputeBindingClass::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
                ComputeBindingClass::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            };
            declare_descriptor(
                &mut declarations,
                null.binding,
                descriptor_type,
                null.descriptor_count,
            )?;
            let descriptor = match null.class {
                ComputeBindingClass::Buffer => NativeComputeDescriptor::StorageBuffer {
                    binding: null.binding,
                    array_element: null.array_element,
                    buffer: vk::Buffer::null(),
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                ComputeBindingClass::SampledImage | ComputeBindingClass::StorageImage => {
                    NativeComputeDescriptor::Image {
                        binding: null.binding,
                        array_element: null.array_element,
                        descriptor_type,
                        view: vk::ImageView::null(),
                        layout: vk::ImageLayout::UNDEFINED,
                    }
                }
            };
            descriptors.push(descriptor);
        }
        let descriptor_counts = declarations
            .into_values()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let launch = match prepared.operation().launch {
            ResolvedComputeLaunch::Direct(workgroups) => NativeComputeLaunch::Direct {
                thread_grid: workgroups.threads_per_grid,
                workgroups: workgroups.counts,
            },
            ResolvedComputeLaunch::IndirectThreadgroups { arguments, .. } => {
                if pipeline.thread_grid_push_offset.is_some() {
                    return Err(ComputeRecordError::IndirectPipelineRequiresThreadGridPushConstant);
                }
                let representation = ViewRepresentation::lookup(
                    representations,
                    arguments.storage,
                    BackingView::Bytes,
                )
                .ok_or(ComputeRecordError::RepresentationUseMismatch(
                    arguments.storage,
                ))?;
                let target = resolver
                    .resolve_buffer(arguments.storage, representation)
                    .ok_or(ComputeRecordError::UnknownBuffer {
                        backing: arguments.storage,
                        representation,
                    })?;
                if !target.usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
                    return Err(ComputeRecordError::MissingIndirectBufferUsage(
                        arguments.storage,
                    ));
                }
                if arguments.region.end() > target.size {
                    return Err(ComputeRecordError::BufferRangeOutOfBounds(
                        arguments.storage,
                    ));
                }
                NativeComputeLaunch::IndirectThreadgroups {
                    buffer: target.buffer,
                    offset: target
                        .base_offset
                        .checked_add(arguments.region.start())
                        .ok_or(ComputeRecordError::BufferAddressOverflow(arguments.storage))?,
                }
            }
        };
        let mut backings = representations
            .iter()
            .map(|representation| representation.backing)
            .collect::<Vec<_>>();
        backings.sort_unstable();
        backings.dedup();
        Ok(Self {
            index: prepared.operation_index(),
            transaction: prepared.transaction(),
            operation: prepared.operation().clone(),
            native: NativeComputeDispatch {
                pipeline,
                wait_for_prior_commands: prepared.operation().wait_for_prior_commands,
                descriptors: descriptors.into_boxed_slice(),
                sampler_leases: sampler_leases.into_boxed_slice(),
                descriptor_counts,
                launch,
                image_state: native_image_state,
            },
            backings: backings.into_boxed_slice(),
            completions: prepared.completions().into(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeRecordError {
    DeviceEpochLost,
    InvalidPipeline,
    RepresentationUseMismatch(BackingId),
    BindingViewMismatch {
        binding: u32,
    },
    DescriptorDeclarationMismatch {
        binding: u32,
    },
    UnknownBuffer {
        backing: BackingId,
        representation: RepresentationId,
    },
    MissingStorageBufferUsage(BackingId),
    MissingIndirectBufferUsage(BackingId),
    IndirectPipelineRequiresThreadGridPushConstant,
    BufferRangeOutOfBounds(BackingId),
    BufferAddressOverflow(BackingId),
    StorageBufferOffsetMisaligned(BackingId),
    StorageBufferRangePastLimit(BackingId),
    ImageState(ComputeImageStateError),
    ImageTransition(ImageTransitionResolveError),
    ImageReleasePending,
    UnknownImage(ReplacementImageKey),
    /// A decoded texture-binding view this backend has no `VkImageView` for.
    /// The reason names which of the view's terms it could not build, because
    /// they are unrelated pieces of work -- see
    /// [`crate::replacement_image_transition::TextureBindingViewDecline`].
    UnknownImageView {
        image: ReplacementImageKey,
        resource: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
        reason: crate::replacement_image_transition::TextureBindingViewDecline,
    },
    MissingImageView(ReplacementImageKey),
    MissingImageUsage {
        image: ReplacementImageKey,
        required: vk::ImageUsageFlags,
    },
    UnknownSampler {
        binding: u32,
    },
    NullDescriptorUnavailable(u32),
}

fn declare_descriptor(
    declarations: &mut BTreeMap<u32, (vk::DescriptorType, u32)>,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    count: u32,
) -> Result<(), ComputeRecordError> {
    if declarations
        .insert(binding, (descriptor_type, count))
        .is_some_and(|found| found != (descriptor_type, count))
    {
        return Err(ComputeRecordError::DescriptorDeclarationMismatch { binding });
    }
    Ok(())
}

fn validate_buffer_target(
    backing: BackingId,
    range: reims_vgpu_core::LinearRange,
    target: NativeBufferTarget,
    resolver: &impl ReplacementComputeResolver,
) -> Result<(), ComputeRecordError> {
    if !target.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        return Err(ComputeRecordError::MissingStorageBufferUsage(backing));
    }
    if range.end() > target.size {
        return Err(ComputeRecordError::BufferRangeOutOfBounds(backing));
    }
    let offset = target
        .base_offset
        .checked_add(range.start())
        .ok_or(ComputeRecordError::BufferAddressOverflow(backing))?;
    let alignment = resolver.min_storage_buffer_offset_alignment();
    if alignment != 0 && !offset.is_multiple_of(alignment) {
        return Err(ComputeRecordError::StorageBufferOffsetMisaligned(backing));
    }
    let range = range.end() - range.start();
    if range > resolver.max_storage_buffer_range() {
        return Err(ComputeRecordError::StorageBufferRangePastLimit(backing));
    }
    Ok(())
}

/// Update one worker-owned descriptor set and emit the exact dispatch.
///
/// # Safety
///
/// The command buffer must be recording. Every pipeline, layout, descriptor
/// resource, and sampler retained by `native` must remain live through the
/// accepting queue point.
pub unsafe fn record_compute_dispatch(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    descriptor_set: Option<vk::DescriptorSet>,
    native: &NativeComputeDispatch,
) {
    if native.wait_for_prior_commands {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            )
        };
    }
    if let Some(image_state) = &native.image_state {
        unsafe { record_hazard_barriers(device, command_buffer, &image_state.transitions.before) };
    }
    debug_assert_eq!(
        descriptor_set.is_some(),
        !native.descriptor_counts.is_empty()
    );
    for descriptor in native.descriptors.iter().copied() {
        let descriptor_set = descriptor_set.expect("a descriptor declaration owns one set");
        match descriptor {
            NativeComputeDescriptor::StorageBuffer {
                binding,
                array_element,
                buffer,
                offset,
                range,
            } => {
                let info = [vk::DescriptorBufferInfo {
                    buffer,
                    offset,
                    range,
                }];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(array_element)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)];
                unsafe { device.update_descriptor_sets(&write, &[]) };
            }
            NativeComputeDescriptor::Sampler {
                binding,
                array_element,
                sampler,
            } => {
                let info = [vk::DescriptorImageInfo::default().sampler(sampler)];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(array_element)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&info)];
                unsafe { device.update_descriptor_sets(&write, &[]) };
            }
            NativeComputeDescriptor::Image {
                binding,
                array_element,
                descriptor_type,
                view,
                layout,
            } => {
                let info = [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(layout)];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(array_element)
                    .descriptor_type(descriptor_type)
                    .image_info(&info)];
                unsafe { device.update_descriptor_sets(&write, &[]) };
            }
        }
    }
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            native.pipeline.pipeline,
        );
        if let Some(descriptor_set) = descriptor_set {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                native.pipeline.layout,
                0,
                &[descriptor_set],
                &[],
            );
        }
    }
    unsafe {
        match native.launch {
            NativeComputeLaunch::Direct {
                thread_grid,
                workgroups,
            } => {
                let grid = thread_grid
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>();
                if let Some(offset) = native.pipeline.thread_grid_push_offset {
                    device.cmd_push_constants(
                        command_buffer,
                        native.pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        offset,
                        &grid,
                    );
                }
                device.cmd_dispatch(command_buffer, workgroups[0], workgroups[1], workgroups[2]);
            }
            NativeComputeLaunch::IndirectThreadgroups { buffer, offset } => {
                device.cmd_dispatch_indirect(command_buffer, buffer, offset);
            }
        }
        if let Some(image_state) = &native.image_state {
            record_hazard_barriers(device, command_buffer, &image_state.transitions.after);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::{
        prepare_compute_dispatch, AccessMode, BackingRegion, ComputeBindingView, PipelineLifecycle,
        PipelineReadiness, RepresentationRoute, ResolvedComputeResourceBinding,
        ResolvedResourceLifecycle, ResourceLifecycleEffect, ResourceLifecycleOwner,
        SessionGeneration, StorageBacking, VulkanDeviceEpoch, GUEST_REPRESENTATION,
    };
    use reims_vgpu_protocol::{
        ComputePipelineObject, ResourceId, ResourceObject, SessionGenerationId, SubmissionId,
        TransactionId, VulkanDeviceEpochId,
    };

    const EPOCH: VulkanDeviceEpochId = VulkanDeviceEpochId::new(3);

    #[test]
    fn render_owned_image_state_is_not_rejected_by_an_image_free_compute_subset() {
        assert_eq!(
            validate_compute_image_state_batch_presence(false, true, true),
            Ok(())
        );
        assert_eq!(
            validate_compute_image_state_batch_presence(false, false, true),
            Err(ComputeExecProgramError::UnexpectedImageStateBatch)
        );
        assert_eq!(
            validate_compute_image_state_batch_presence(true, true, false),
            Err(ComputeExecProgramError::ImageStateBatchMissing)
        );
    }

    fn image_view(
        resource: ResourceId<ResourceObject>,
    ) -> reims_vgpu_core::ResolvedTextureBindingView {
        reims_vgpu_core::ResolvedTextureBindingView {
            resource,
            base: resource,
            range: reims_vgpu_core::ResolvedTextureViewRange {
                level_base: 0,
                level_count: 1,
                slice_base: 0,
                slice_count: 1,
            },
            texture_type: reims_vgpu_protocol::TextureType::D2,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            swizzle: reims_vgpu_protocol::swizzle_identity(),
        }
    }

    fn pipeline_plan_operation() -> ResolvedComputeDispatch {
        ResolvedComputeDispatch {
            pipeline: ResourceId::new(2, 1),
            launch: reims_vgpu_core::ResolvedComputeLaunch::Direct(
                reims_vgpu_protocol::dispatch::WorkgroupPlan {
                    counts: [1, 1, 1],
                    threads_per_grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            ),
            wait_for_prior_commands: false,
            resources: Box::new([
                ResolvedComputeResourceBinding {
                    class: ComputeBindingClass::Buffer,
                    binding: 4,
                    array_element: 0,
                    descriptor_count: 1,
                    resource: ResourceId::new(7, 1),
                    backing: BackingId::new(8),
                    view: ComputeBindingView::Buffer(
                        reims_vgpu_core::LinearRange::new(0, 16).unwrap(),
                    ),
                    regions: Box::new([BackingRegion::Whole]),
                    mode: AccessMode::Read,
                },
                ResolvedComputeResourceBinding {
                    class: ComputeBindingClass::StorageImage,
                    binding: 9,
                    array_element: 1,
                    descriptor_count: 3,
                    resource: ResourceId::new(10, 1),
                    backing: BackingId::new(11),
                    view: ComputeBindingView::Image(image_view(ResourceId::new(10, 1))),
                    regions: Box::new([BackingRegion::Whole]),
                    mode: AccessMode::Write,
                },
            ]),
            samplers: Box::new([reims_vgpu_core::ResolvedComputeSamplerBinding {
                binding: 13,
                array_element: 0,
                descriptor_count: 1,
                sampler: reims_vgpu_core::SamplerResource::null(13),
            }]),
            null_bindings: Box::new([
                reims_vgpu_core::ResolvedComputeNullBinding {
                    class: ComputeBindingClass::StorageImage,
                    binding: 9,
                    array_element: 0,
                    descriptor_count: 3,
                },
                reims_vgpu_core::ResolvedComputeNullBinding {
                    class: ComputeBindingClass::SampledImage,
                    binding: 12,
                    array_element: 0,
                    descriptor_count: 2,
                },
            ]),
        }
    }

    #[test]
    fn compute_pipeline_plan_preserves_complete_descriptor_declarations() {
        let operation = pipeline_plan_operation();
        let stage = reims_vgpu_core::PreparedShaderStage {
            id: reims_vgpu_protocol::PreparedShaderId::new(17),
            used_descriptor_bindings: Arc::from([4, 9, 12, 13]),
        };
        let plan = resolve_compute_pipeline_plan(&operation, stage.clone())
            .expect("all used declarations are represented");
        assert_eq!(plan.program, stage);
        assert_eq!(
            plan.descriptors.as_ref(),
            [
                (4, vk::DescriptorType::STORAGE_BUFFER.as_raw(), 1),
                (9, vk::DescriptorType::STORAGE_IMAGE.as_raw(), 3),
                (12, vk::DescriptorType::SAMPLED_IMAGE.as_raw(), 2),
                (13, vk::DescriptorType::SAMPLER.as_raw(), 1),
            ]
        );
    }

    #[test]
    fn compute_pipeline_plan_refuses_missing_and_conflicting_declarations() {
        let operation = pipeline_plan_operation();
        let missing = resolve_compute_pipeline_plan(
            &operation,
            reims_vgpu_core::PreparedShaderStage {
                id: reims_vgpu_protocol::PreparedShaderId::new(17),
                used_descriptor_bindings: Arc::from([4, 18]),
            },
        );
        assert_eq!(
            missing,
            Err(ReplacementComputePipelinePlanError::UsedDescriptorMissing(
                18
            ))
        );

        let mut collision = pipeline_plan_operation();
        collision.null_bindings = Box::new([reims_vgpu_core::ResolvedComputeNullBinding {
            class: ComputeBindingClass::SampledImage,
            binding: 4,
            array_element: 0,
            descriptor_count: 1,
        }]);
        assert_eq!(
            resolve_compute_pipeline_plan(
                &collision,
                reims_vgpu_core::PreparedShaderStage {
                    id: reims_vgpu_protocol::PreparedShaderId::new(17),
                    used_descriptor_bindings: Arc::from([4]),
                },
            ),
            Err(ReplacementComputePipelinePlanError::DescriptorCollision(4))
        );
    }

    struct Resolver {
        backing: BackingId,
        representation: RepresentationId,
        usage: vk::BufferUsageFlags,
    }

    impl ReplacementBufferResolver for Resolver {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            (backing == self.backing && representation == self.representation).then_some(
                NativeBufferTarget {
                    buffer: vk::Buffer::from_raw(20),
                    base_offset: 64,
                    accessible_size: 128,
                    size: 64,
                    usage: self.usage,
                },
            )
        }
    }

    impl ReplacementImageResolver for Resolver {
        fn resolve_image(
            &self,
            _: ReplacementImageKey,
        ) -> Option<crate::replacement_image_transition::NativeImageTarget> {
            Some(crate::replacement_image_transition::NativeImageTarget {
                image: vk::Image::from_raw(30),
                view: vk::ImageView::from_raw(31),
                image_type: vk::ImageType::TYPE_2D,
                full_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
                pixel_format: 80,
                extent: vk::Extent3D {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
                samples: vk::SampleCountFlags::TYPE_1,
            })
        }

        fn resolve_texture_binding_view(
            &self,
            image: ReplacementImageKey,
            _: reims_vgpu_core::ResolvedTextureBindingView,
        ) -> Result<
            crate::replacement_image_transition::NativeImageTarget,
            crate::replacement_image_transition::TextureBindingViewDecline,
        > {
            self.resolve_image(image).ok_or(
                crate::replacement_image_transition::TextureBindingViewDecline::UnknownRepresentation,
            )
        }
    }

    impl ReplacementComputeResolver for Resolver {
        fn resolve_sampler(
            &self,
            _: ResourceId<ComputePipelineObject>,
            _: &reims_vgpu_core::SamplerResource,
        ) -> Option<crate::replacement_sampler::ReplacementSamplerLease> {
            None
        }

        fn max_storage_buffer_range(&self) -> u64 {
            64
        }

        fn min_storage_buffer_offset_alignment(&self) -> u64 {
            16
        }

        fn null_descriptors(&self) -> bool {
            true
        }
    }

    fn pipeline_with_thread_grid(
        thread_grid_push_offset: Option<u32>,
    ) -> reims_vgpu_core::ReadyPipelineLease<ComputePipelineObject, ReplacementComputePipelineVariant>
    {
        let id = ResourceId::new(2, 1);
        let mut owner = PipelineLifecycle::<
            ComputePipelineObject,
            (),
            ReplacementComputePipelineVariant,
            (),
        >::default();
        owner.declare(id, ()).unwrap();
        let translation = owner.begin_translation(id).unwrap();
        let compile = owner.translation_complete(translation, ()).unwrap();
        owner
            .compile_complete(
                compile,
                reims_vgpu_core::NativeObjectLease::acquire(
                    &SessionGeneration::new(SessionGenerationId::new(1)),
                    &VulkanDeviceEpoch::new(EPOCH),
                )
                .unwrap(),
                ReplacementComputePipelineVariant::synthetic(ReplacementComputePipeline {
                    pipeline: vk::Pipeline::from_raw(1),
                    layout: vk::PipelineLayout::from_raw(2),
                    descriptor_set_layout: vk::DescriptorSetLayout::from_raw(3),
                    thread_grid_push_offset,
                }),
            )
            .unwrap();
        let PipelineReadiness::Ready(ready) = owner.readiness(id, TransactionId::new(4)).unwrap()
        else {
            unreachable!()
        };
        ready
    }

    fn pipeline(
    ) -> reims_vgpu_core::ReadyPipelineLease<ComputePipelineObject, ReplacementComputePipelineVariant>
    {
        pipeline_with_thread_grid(Some(4))
    }

    #[test]
    fn prepared_buffer_dispatch_projects_exact_descriptor_grid_and_completion() {
        let mut owner = ResourceLifecycleOwner::<()>::new(EPOCH);
        let ResourceLifecycleEffect::BackingCreated(backing) = owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        for transfer in owner
            .plan_transfers(backing, GUEST_REPRESENTATION, representation, &snapshot)
            .unwrap()
        {
            owner.complete_transfer(transfer).unwrap();
        }
        let operation = ResolvedComputeDispatch {
            pipeline: ResourceId::new(2, 1),
            launch: reims_vgpu_core::ResolvedComputeLaunch::Direct(
                reims_vgpu_protocol::dispatch::WorkgroupPlan {
                    counts: [3, 2, 1],
                    threads_per_grid: [21, 9, 1],
                    threads_per_threadgroup: [8, 8, 1],
                },
            ),
            wait_for_prior_commands: false,
            resources: Box::new([ResolvedComputeResourceBinding {
                class: ComputeBindingClass::Buffer,
                binding: 5,
                array_element: 0,
                descriptor_count: 1,
                resource: ResourceId::<ResourceObject>::new(7, 1),
                backing,
                view: ComputeBindingView::Buffer(
                    reims_vgpu_core::LinearRange::new(16, 16).unwrap(),
                ),
                regions: Box::new([BackingRegion::Whole]),
                mode: AccessMode::ReadWrite,
            }]),
            samplers: Box::new([]),
            null_bindings: Box::new([]),
        };
        let prepared = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(4),
            SubmissionId::new(8),
            6,
            operation.clone(),
            pipeline(),
        )
        .unwrap();
        let program = ReplacementComputeProgram::resolve(
            &prepared,
            &Resolver {
                backing,
                representation,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        )
        .unwrap();

        assert_eq!(program.index(), 6);
        assert_eq!(program.transaction(), TransactionId::new(4));
        assert_eq!(program.operation(), &operation);
        assert_eq!(program.backings(), [backing]);
        assert_eq!(program.completions(), prepared.completions());
        assert_eq!(
            program.native().descriptors.as_ref(),
            [NativeComputeDescriptor::StorageBuffer {
                binding: 5,
                array_element: 0,
                buffer: vk::Buffer::from_raw(20),
                offset: 80,
                range: 16,
            }]
        );
        assert_eq!(
            program.native().descriptor_counts.as_ref(),
            [(vk::DescriptorType::STORAGE_BUFFER, 1)]
        );
        assert_eq!(
            program.native().launch,
            NativeComputeLaunch::Direct {
                thread_grid: [21, 9, 1],
                workgroups: [3, 2, 1],
            }
        );
    }

    fn prepared_indirect_dispatch(
        thread_grid_push_offset: Option<u32>,
    ) -> (
        PreparedComputeDispatch<ReplacementComputePipelineVariant>,
        BackingId,
        RepresentationId,
    ) {
        let mut owner = ResourceLifecycleOwner::<()>::new(EPOCH);
        let ResourceLifecycleEffect::BackingCreated(backing) = owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let representation = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Bytes,
                (),
            )
            .unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        for transfer in owner
            .plan_transfers(backing, GUEST_REPRESENTATION, representation, &snapshot)
            .unwrap()
        {
            owner.complete_transfer(transfer).unwrap();
        }
        let operation = ResolvedComputeDispatch {
            pipeline: ResourceId::new(2, 1),
            launch: ResolvedComputeLaunch::IndirectThreadgroups {
                arguments: reims_vgpu_core::ResolvedBufferRange {
                    resource: ResourceId::new(7, 1),
                    storage: backing,
                    region: reims_vgpu_core::LinearRange::new(24, 12).unwrap(),
                    address: reims_vgpu_protocol::GuestVirtualAddress::new(0x1018),
                    length: reims_vgpu_protocol::ByteLength::new(12),
                },
                threads_per_threadgroup: [4, 2, 1],
            },
            wait_for_prior_commands: false,
            resources: Box::new([]),
            samplers: Box::new([]),
            null_bindings: Box::new([]),
        };
        let prepared = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(4),
            SubmissionId::new(8),
            6,
            operation,
            pipeline_with_thread_grid(thread_grid_push_offset),
        )
        .unwrap();
        (prepared, backing, representation)
    }

    #[test]
    fn indirect_threadgroups_project_the_exact_vulkan_argument_and_require_its_contract() {
        let (prepared, backing, representation) = prepared_indirect_dispatch(None);
        let program = ReplacementComputeProgram::resolve(
            &prepared,
            &Resolver {
                backing,
                representation,
                usage: vk::BufferUsageFlags::INDIRECT_BUFFER,
            },
        )
        .unwrap();
        assert_eq!(program.backings(), [backing]);
        assert_eq!(
            program.native().launch,
            NativeComputeLaunch::IndirectThreadgroups {
                buffer: vk::Buffer::from_raw(20),
                offset: 88,
            }
        );

        assert_eq!(
            ReplacementComputeProgram::resolve(
                &prepared,
                &Resolver {
                    backing,
                    representation,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                },
            )
            .unwrap_err(),
            ComputeRecordError::MissingIndirectBufferUsage(backing)
        );

        let (dynamic, dynamic_backing, dynamic_representation) =
            prepared_indirect_dispatch(Some(4));
        assert_eq!(
            ReplacementComputeProgram::resolve(
                &dynamic,
                &Resolver {
                    backing: dynamic_backing,
                    representation: dynamic_representation,
                    usage: vk::BufferUsageFlags::INDIRECT_BUFFER,
                },
            )
            .unwrap_err(),
            ComputeRecordError::IndirectPipelineRequiresThreadGridPushConstant
        );
    }

    #[test]
    fn explicit_null_compute_sampler_needs_no_native_lease() {
        let mut owner = ResourceLifecycleOwner::<()>::new(EPOCH);
        let operation = ResolvedComputeDispatch {
            pipeline: ResourceId::new(2, 1),
            launch: reims_vgpu_core::ResolvedComputeLaunch::Direct(
                reims_vgpu_protocol::dispatch::WorkgroupPlan {
                    counts: [1, 1, 1],
                    threads_per_grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            ),
            wait_for_prior_commands: false,
            resources: Box::new([]),
            samplers: Box::new([reims_vgpu_core::ResolvedComputeSamplerBinding {
                binding: 7,
                array_element: 0,
                descriptor_count: 1,
                sampler: reims_vgpu_core::SamplerResource::null(7),
            }]),
            null_bindings: Box::new([]),
        };
        let prepared = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(4),
            SubmissionId::new(8),
            0,
            operation,
            pipeline(),
        )
        .unwrap();
        let program = ReplacementComputeProgram::resolve(
            &prepared,
            &Resolver {
                backing: BackingId::new(1),
                representation: RepresentationId::new(1),
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        )
        .unwrap();
        assert!(program.native().sampler_leases.is_empty());
        assert!(matches!(
            program.native().descriptors.as_ref(),
            [NativeComputeDescriptor::Sampler { binding: 7, sampler, .. }]
                if *sampler == vk::Sampler::null()
        ));
    }

    #[test]
    fn sampled_and_storage_bindings_share_one_general_image_transition() {
        let mut owner = ResourceLifecycleOwner::new(EPOCH);
        let ResourceLifecycleEffect::BackingCreated(backing) = owner
            .apply(ResolvedResourceLifecycle::CreateBacking {
                backing: StorageBacking::Dedicated,
                regions: Box::new([BackingRegion::Whole]),
            })
            .unwrap()
        else {
            unreachable!()
        };
        let resource = ResourceId::<ResourceObject>::new(9, 1);
        let representation = owner
            .create_execution_representation(
                backing,
                RepresentationRoute::HostVisibleWorking,
                BackingView::Image(reims_vgpu_core::ImageOwner::base(resource)),
                (),
            )
            .unwrap();
        let snapshot = owner
            .snapshot_content(backing, &[BackingRegion::Whole])
            .unwrap();
        for transfer in owner
            .plan_transfers(backing, GUEST_REPRESENTATION, representation, &snapshot)
            .unwrap()
        {
            owner.complete_transfer(transfer).unwrap();
        }
        let binding = |class, binding, mode| ResolvedComputeResourceBinding {
            class,
            binding,
            array_element: 0,
            descriptor_count: 1,
            resource,
            backing,
            view: ComputeBindingView::Image(image_view(resource)),
            regions: Box::new([BackingRegion::Whole]),
            mode,
        };
        let operation = ResolvedComputeDispatch {
            pipeline: ResourceId::new(2, 1),
            launch: reims_vgpu_core::ResolvedComputeLaunch::Direct(
                reims_vgpu_protocol::dispatch::WorkgroupPlan {
                    counts: [1, 1, 1],
                    threads_per_grid: [8, 8, 1],
                    threads_per_threadgroup: [8, 8, 1],
                },
            ),
            wait_for_prior_commands: false,
            resources: Box::new([
                binding(ComputeBindingClass::SampledImage, 2, AccessMode::Read),
                binding(ComputeBindingClass::StorageImage, 3, AccessMode::ReadWrite),
            ]),
            samplers: Box::new([]),
            null_bindings: Box::new([]),
        };
        let prepared = prepare_compute_dispatch(
            &mut owner,
            TransactionId::new(4),
            SubmissionId::new(8),
            6,
            operation,
            pipeline(),
        )
        .unwrap();
        let key = ReplacementImageKey {
            backing,
            representation,
        };
        assert_eq!(
            derive_compute_image_uses(&prepared).unwrap().as_ref(),
            [ReplacementImageUse {
                image: key,
                required_usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
                use_layout: vk::ImageLayout::GENERAL,
                final_layout: vk::ImageLayout::GENERAL,
            }]
        );
        let mut images = ReplacementImageStateOwner::new(EPOCH);
        images
            .register(
                key,
                crate::replacement_image_state::ReplacementImageState {
                    layout: vk::ImageLayout::UNDEFINED,
                    sharing: crate::replacement_image_state::ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let state = prepare_compute_image_state(&mut images, &prepared, 4)
            .unwrap()
            .unwrap();
        let program = ReplacementComputeProgram::resolve_with_image_state(
            &prepared,
            Some(&state),
            &Resolver {
                backing,
                representation,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        )
        .unwrap();
        assert!(matches!(
            program.native().descriptors.as_ref(),
            [
                NativeComputeDescriptor::Image {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    layout: vk::ImageLayout::GENERAL,
                    ..
                },
                NativeComputeDescriptor::Image {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    layout: vk::ImageLayout::GENERAL,
                    ..
                }
            ]
        ));
        let native_state = program.native().image_state.as_ref().unwrap();
        assert!(native_state.releases.is_empty());
        assert_eq!(native_state.transitions.before.images.len(), 1);
        assert_eq!(native_state.transitions.after.images.len(), 1);
    }
}
