//! Join prepared image blits to queue-ordered image layout state.

use crate::replacement_buffer_blit::ReplacementBufferResolver;
use crate::replacement_image_state::{
    PreparedImageState, ReplacementImageKey, ReplacementImageStateError,
    ReplacementImageStateOwner, ReplacementImageUse,
};
use crate::replacement_image_transition::{
    resolve_image_transitions, NativeImageTarget, PreparedNativeImageState,
    ReplacementImageResolver,
};
use ash::vk;
use reims_vgpu_core::{
    pixel_format::{
        blit_aspect_bytes_per_pixel, buffer_image_blit_aspect, format_has_depth_aspect,
        format_has_stencil_aspect, BlitAspect,
    },
    BackingView, PreparedExecResources, PreparedImageBlit, ResolvedBlit,
    ResolvedResourceCompletion, ResolvedTextureEndpoint, TextureExtent, TextureOrigin,
    ViewRepresentation,
};
use reims_vgpu_protocol::BackingId;
use std::collections::BTreeMap;

pub trait ReplacementImageFinalLayout {
    fn final_layout(
        &self,
        image: ReplacementImageKey,
        required_usage: vk::ImageUsageFlags,
    ) -> Option<vk::ImageLayout>;
}

/// Conservative post-blit layout valid for every image usage admitted by the
/// replacement representation planner.
///
/// Render and compute operations retain their narrower attachment or sampled
/// layouts independently. This policy is used only where a transfer operation
/// does not itself declare the image's next consumer.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplacementGeneralFinalLayout;

impl ReplacementImageFinalLayout for ReplacementGeneralFinalLayout {
    fn final_layout(
        &self,
        _image: ReplacementImageKey,
        _required_usage: vk::ImageUsageFlags,
    ) -> Option<vk::ImageLayout> {
        Some(vk::ImageLayout::GENERAL)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageBlitStateError {
    MissingRepresentation(BackingId),
    MissingFinalLayout(ReplacementImageKey),
    State(ReplacementImageStateError),
}

pub fn prepare_image_blit_state(
    owner: &mut ReplacementImageStateOwner,
    prepared: &PreparedImageBlit,
    queue_family: u32,
    final_layouts: &impl ReplacementImageFinalLayout,
) -> Result<PreparedImageState, ImageBlitStateError> {
    let uses = derive_image_uses(
        prepared.operation(),
        prepared.representations(),
        final_layouts,
    )?;
    match prepared.write().operation_index() {
        Some(index) => owner.prepare_operation(prepared.transaction(), index, queue_family, uses),
        None => owner.prepare(prepared.transaction(), queue_family, uses),
    }
    .map_err(ImageBlitStateError::State)
}

pub fn prepare_exec_image_blit_states<Compute, NativeCompute, Render, NativeRender>(
    owner: &mut ReplacementImageStateOwner,
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    queue_family: u32,
    final_layouts: &impl ReplacementImageFinalLayout,
) -> Result<crate::replacement_image_state::PreparedImageStateBatch, ImageBlitStateError> {
    let mut operations = resources
        .inputs()
        .image_blits
        .iter()
        .map(|prepared| {
            let index = prepared
                .write()
                .operation_index()
                .expect("whole-EXEC image preparations are operation-positioned");
            derive_image_uses(
                prepared.operation(),
                prepared.representations(),
                final_layouts,
            )
            .map(|uses| (index, uses))
        })
        .collect::<Result<Vec<_>, _>>()?;
    operations.sort_unstable_by_key(|(index, _)| *index);
    owner
        .prepare_batch(
            resources.transaction(),
            queue_family,
            operations.into_boxed_slice(),
        )
        .map_err(ImageBlitStateError::State)
}

pub fn validate_exec_image_blit_states<Compute, NativeCompute, Render, NativeRender>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    states: &crate::replacement_image_state::PreparedImageStateBatch,
) -> Result<(), ImageBlitRecordError> {
    if resources.transaction() != states.transaction()
        || resources.inputs().image_blits.len() != states.operations().len()
    {
        return Err(ImageBlitRecordError::StateOperationMismatch);
    }
    validate_exec_image_blit_state_subset(resources, states)
}

pub(crate) fn validate_exec_image_blit_state_subset<
    Compute,
    NativeCompute,
    Render,
    NativeRender,
>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    states: &crate::replacement_image_state::PreparedImageStateBatch,
) -> Result<(), ImageBlitRecordError> {
    if resources.transaction() != states.transaction() {
        return Err(ImageBlitRecordError::StateOperationMismatch);
    }
    for prepared in &resources.inputs().image_blits {
        let index = prepared
            .write()
            .operation_index()
            .ok_or(ImageBlitRecordError::StateOperationMismatch)?;
        let state = states
            .operations()
            .iter()
            .find(|state| state.operation_index() == Some(index))
            .ok_or(ImageBlitRecordError::StateOperationMismatch)?;
        validate_state_uses(prepared, state)?;
    }
    Ok(())
}

pub fn resolve_exec_image_blit_programs<Compute, NativeCompute, Render, NativeRender>(
    resources: &PreparedExecResources<Compute, NativeCompute, Render, NativeRender>,
    states: &crate::replacement_image_state::PreparedImageStateBatch,
    resolver: &(impl ReplacementImageResolver + ReplacementBufferResolver),
) -> Result<Box<[ReplacementImageBlitProgram]>, ImageBlitRecordError> {
    validate_exec_image_blit_state_subset(resources, states)?;
    let mut programs = Vec::with_capacity(resources.inputs().image_blits.len());
    for prepared in &resources.inputs().image_blits {
        let index = prepared
            .write()
            .operation_index()
            .ok_or(ImageBlitRecordError::StateOperationMismatch)?;
        let state = states
            .operations()
            .iter()
            .find(|state| state.operation_index() == Some(index))
            .ok_or(ImageBlitRecordError::StateOperationMismatch)?;
        programs.push(ReplacementImageBlitProgram::resolve(
            index, prepared, state, resolver,
        )?);
    }
    Ok(programs.into_boxed_slice())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ImageRoles {
    source: bool,
    destination: bool,
}

pub(crate) fn derive_image_uses(
    operation: &ResolvedBlit,
    representations: &[ViewRepresentation],
    final_layouts: &impl ReplacementImageFinalLayout,
) -> Result<Box<[ReplacementImageUse]>, ImageBlitStateError> {
    // Keyed by the texture as well as its backing: two textures declared over
    // one guest range are two images, and a blit between them would otherwise
    // collapse into one entry naming a single object for both endpoints.
    let mut roles = BTreeMap::<(BackingId, reims_vgpu_core::ImageOwner), ImageRoles>::new();
    match operation {
        ResolvedBlit::Fill { .. } | ResolvedBlit::Copy { .. } => {
            unreachable!("PreparedImageBlit cannot contain a buffer-only variant")
        }
        ResolvedBlit::BufferToTexture(blit) => {
            roles
                .entry((
                    blit.destination.storage,
                    reims_vgpu_core::ImageOwner::owning(blit.destination.image_owner),
                ))
                .or_default()
                .destination = true;
        }
        ResolvedBlit::TextureToBuffer(blit) => {
            roles
                .entry((
                    blit.source.storage,
                    reims_vgpu_core::ImageOwner::owning(blit.source.image_owner),
                ))
                .or_default()
                .source = true;
        }
        ResolvedBlit::TextureToTexture(blit) => {
            roles
                .entry((
                    blit.source.storage,
                    reims_vgpu_core::ImageOwner::owning(blit.source.image_owner),
                ))
                .or_default()
                .source = true;
            roles
                .entry((
                    blit.destination.storage,
                    reims_vgpu_core::ImageOwner::owning(blit.destination.image_owner),
                ))
                .or_default()
                .destination = true;
        }
        ResolvedBlit::TextureCopyBatch(batch) => {
            for level in std::iter::once(&batch.first_level).chain(batch.remaining_levels.iter()) {
                for (source, destination) in
                    std::iter::once(&level.first_slice).chain(level.remaining_slices.iter())
                {
                    roles
                        .entry((
                            source.storage,
                            reims_vgpu_core::ImageOwner::owning(source.image_owner),
                        ))
                        .or_default()
                        .source = true;
                    roles
                        .entry((
                            destination.storage,
                            reims_vgpu_core::ImageOwner::owning(destination.image_owner),
                        ))
                        .or_default()
                        .destination = true;
                }
            }
        }
    }
    roles
        .into_iter()
        .map(|((backing, owner), roles)| {
            // Image state is about the image, so it is this texture's image
            // over the backing that the use names — never a buffer view of the
            // same bytes that some other endpoint of the blit reads, and never
            // another texture's image over the same range.
            let representation =
                ViewRepresentation::lookup(representations, backing, BackingView::Image(owner))
                    .ok_or(ImageBlitStateError::MissingRepresentation(backing))?;
            let image = ReplacementImageKey {
                backing,
                representation,
            };
            let required_usage = match (roles.source, roles.destination) {
                (true, false) => vk::ImageUsageFlags::TRANSFER_SRC,
                (false, true) => vk::ImageUsageFlags::TRANSFER_DST,
                (true, true) => {
                    vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
                }
                (false, false) => unreachable!(),
            };
            let use_layout = match (roles.source, roles.destination) {
                (true, false) => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                (false, true) => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                (true, true) => vk::ImageLayout::GENERAL,
                (false, false) => unreachable!(),
            };
            let final_layout = final_layouts
                .final_layout(image, required_usage)
                .ok_or(ImageBlitStateError::MissingFinalLayout(image))?;
            Ok(ReplacementImageUse {
                image,
                required_usage,
                use_layout,
                final_layout,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeImageCopy {
    pub source: vk::Image,
    pub source_layout: vk::ImageLayout,
    pub destination: vk::Image,
    pub destination_layout: vk::ImageLayout,
    pub aspect: vk::ImageAspectFlags,
    pub source_mip: u32,
    pub source_layer: u32,
    pub destination_mip: u32,
    pub destination_layer: u32,
    pub source_offset: [i32; 3],
    pub destination_offset: [i32; 3],
    pub extent: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeBufferImageCopy {
    pub buffer: vk::Buffer,
    pub image: vk::Image,
    pub image_layout: vk::ImageLayout,
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub aspect: vk::ImageAspectFlags,
    pub mip: u32,
    pub layer: u32,
    pub image_offset: [i32; 3],
    pub extent: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeImageBlitCommand {
    BufferToImage(NativeBufferImageCopy),
    ImageToBuffer(NativeBufferImageCopy),
    ImageToImage(NativeImageCopy),
}

#[derive(Clone, Debug)]
pub struct PreparedNativeImageBlit {
    pub state: PreparedNativeImageState,
    pub commands: Box<[NativeImageBlitCommand]>,
}

impl PreparedNativeImageBlit {
    /// Queue capabilities required by the exact commands in this program.
    ///
    /// Depth/stencil buffer-image copies require a graphics-capable command
    /// pool in the Vulkan feature set used by this backend. Color copies and
    /// same-aspect image copies remain valid on a transfer-capable family.
    pub fn required_queue_flags(&self) -> vk::QueueFlags {
        if self.commands.iter().any(|command| {
            matches!(
                command,
                NativeImageBlitCommand::BufferToImage(copy)
                    | NativeImageBlitCommand::ImageToBuffer(copy)
                    if copy
                        .aspect
                        .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
            )
        }) {
            vk::QueueFlags::GRAPHICS
        } else {
            vk::QueueFlags::empty()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReplacementImageBlitProgram {
    index: usize,
    operation: ResolvedBlit,
    backings: Box<[BackingId]>,
    completions: Box<[ResolvedResourceCompletion]>,
    native: PreparedNativeImageBlit,
}

impl ReplacementImageBlitProgram {
    pub fn resolve(
        index: usize,
        prepared: &PreparedImageBlit,
        state: &PreparedImageState,
        resolver: &(impl ReplacementImageResolver + ReplacementBufferResolver),
    ) -> Result<Self, ImageBlitRecordError> {
        if prepared.write()
            != reims_vgpu_core::GpuWriteId::operation(
                prepared.transaction(),
                prepared.submission(),
                index,
            )
        {
            return Err(ImageBlitRecordError::WriteIdentityMismatch);
        }
        if state.operation_index() != Some(index) {
            return Err(ImageBlitRecordError::StateOperationMismatch);
        }
        Ok(Self {
            index,
            operation: prepared.operation().clone(),
            backings: prepared.backings(),
            completions: prepared.resource_completions(),
            native: resolve_native_image_blit(prepared, state, resolver)?,
        })
    }

    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    pub(crate) const fn operation(&self) -> &ResolvedBlit {
        &self.operation
    }

    pub(crate) const fn native(&self) -> &PreparedNativeImageBlit {
        &self.native
    }

    pub(crate) const fn backings(&self) -> &[BackingId] {
        &self.backings
    }

    pub(crate) const fn completions(&self) -> &[ResolvedResourceCompletion] {
        &self.completions
    }
}

#[cfg(test)]
impl ReplacementImageBlitProgram {
    pub(crate) fn synthetic(
        index: usize,
        operation: ResolvedBlit,
        native: PreparedNativeImageBlit,
    ) -> Self {
        Self {
            index,
            operation,
            backings: Box::new([]),
            completions: Box::new([]),
            native,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageBlitRecordError {
    WriteIdentityMismatch,
    StateOperationMismatch,
    TransactionMismatch,
    MissingRepresentation(BackingId),
    StateUseMismatch(ReplacementImageKey),
    UnknownImage(ReplacementImageKey),
    UnknownBuffer {
        backing: BackingId,
        representation: reims_vgpu_protocol::RepresentationId,
    },
    PixelFormatMismatch(ReplacementImageKey),
    CombinedDepthStencilUnsupported,
    AspectUnavailable(ReplacementImageKey),
    SubresourceOutOfBounds(ReplacementImageKey),
    IncompatibleImageGeometry(ReplacementImageKey),
    BufferRangeOutOfBounds(BackingId),
    BufferAddressOverflow(BackingId),
    MissingBufferUsage(BackingId),
    UnrepresentableBufferLayout,
    CoordinateOverflow,
    SameImageOverlap,
    Transition(crate::replacement_image_transition::ImageTransitionResolveError),
}

pub fn resolve_native_image_blit(
    prepared: &PreparedImageBlit,
    state: &PreparedImageState,
    resolver: &(impl ReplacementImageResolver + ReplacementBufferResolver),
) -> Result<PreparedNativeImageBlit, ImageBlitRecordError> {
    if prepared.transaction() != state.transaction() {
        return Err(ImageBlitRecordError::TransactionMismatch);
    }
    validate_state_uses(prepared, state)?;
    let native_state =
        resolve_image_transitions(state, resolver).map_err(ImageBlitRecordError::Transition)?;
    let mut commands = Vec::new();
    match prepared.operation() {
        ResolvedBlit::Fill { .. } | ResolvedBlit::Copy { .. } => unreachable!(),
        ResolvedBlit::BufferToTexture(blit) => commands.push(
            NativeImageBlitCommand::BufferToImage(resolve_buffer_image_copy(
                blit.source,
                blit.source_bytes_per_row,
                blit.source_bytes_per_image,
                &blit.destination,
                blit.destination_origin,
                blit.extent,
                blit.aspect,
                vk::BufferUsageFlags::TRANSFER_SRC,
                prepared.representations(),
                state,
                resolver,
            )?),
        ),
        ResolvedBlit::TextureToBuffer(blit) => commands.push(
            NativeImageBlitCommand::ImageToBuffer(resolve_buffer_image_copy(
                blit.destination,
                blit.destination_bytes_per_row,
                blit.destination_bytes_per_image,
                &blit.source,
                blit.source_origin,
                blit.extent,
                blit.aspect,
                vk::BufferUsageFlags::TRANSFER_DST,
                prepared.representations(),
                state,
                resolver,
            )?),
        ),
        ResolvedBlit::TextureToTexture(blit) => commands.extend(
            resolve_image_copies(
                &blit.source,
                blit.source_origin,
                &blit.destination,
                blit.destination_origin,
                blit.extent,
                blit.aspect,
                prepared.representations(),
                state,
                resolver,
            )?
            .into_vec()
            .into_iter()
            .map(NativeImageBlitCommand::ImageToImage),
        ),
        ResolvedBlit::TextureCopyBatch(batch) => {
            for level in std::iter::once(&batch.first_level).chain(batch.remaining_levels.iter()) {
                for (source, destination) in
                    std::iter::once(&level.first_slice).chain(level.remaining_slices.iter())
                {
                    commands.extend(
                        resolve_image_copies(
                            source,
                            TextureOrigin { x: 0, y: 0, z: 0 },
                            destination,
                            TextureOrigin { x: 0, y: 0, z: 0 },
                            TextureExtent {
                                width: u64::from(source.backing.width()),
                                height: u64::from(source.backing.height()),
                                depth: u64::from(source.backing.depth()),
                            },
                            BlitAspect::Full,
                            prepared.representations(),
                            state,
                            resolver,
                        )?
                        .into_vec()
                        .into_iter()
                        .map(NativeImageBlitCommand::ImageToImage),
                    );
                }
            }
        }
    }
    Ok(PreparedNativeImageBlit {
        state: native_state,
        commands: commands.into_boxed_slice(),
    })
}

fn validate_state_uses(
    prepared: &PreparedImageBlit,
    state: &PreparedImageState,
) -> Result<(), ImageBlitRecordError> {
    struct PreserveFinal<'a>(&'a PreparedImageState);
    impl ReplacementImageFinalLayout for PreserveFinal<'_> {
        fn final_layout(
            &self,
            image: ReplacementImageKey,
            _required_usage: vk::ImageUsageFlags,
        ) -> Option<vk::ImageLayout> {
            self.0
                .transitions()
                .iter()
                .find(|transition| transition.image == image)
                .map(|transition| transition.final_layout)
        }
    }
    let expected = derive_image_uses(
        prepared.operation(),
        prepared.representations(),
        &PreserveFinal(state),
    )
    .map_err(|error| match error {
        ImageBlitStateError::MissingRepresentation(backing) => {
            ImageBlitRecordError::MissingRepresentation(backing)
        }
        ImageBlitStateError::MissingFinalLayout(image) => {
            ImageBlitRecordError::StateUseMismatch(image)
        }
        ImageBlitStateError::State(_) => unreachable!(),
    })?;
    if expected.len() != state.transitions().len() {
        let image = state
            .transitions()
            .first()
            .map(|transition| transition.image)
            .or_else(|| expected.first().map(|use_| use_.image))
            .expect("an image blit has at least one image operand");
        return Err(ImageBlitRecordError::StateUseMismatch(image));
    }
    for expected in expected {
        let Some(found) = state
            .transitions()
            .iter()
            .find(|transition| transition.image == expected.image)
        else {
            return Err(ImageBlitRecordError::StateUseMismatch(expected.image));
        };
        if found.required_usage != expected.required_usage
            || found.use_layout != expected.use_layout
            || found.final_layout != expected.final_layout
        {
            return Err(ImageBlitRecordError::StateUseMismatch(expected.image));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_image_copies(
    source: &ResolvedTextureEndpoint,
    source_origin: TextureOrigin,
    destination: &ResolvedTextureEndpoint,
    destination_origin: TextureOrigin,
    extent: TextureExtent,
    aspect: BlitAspect,
    representations: &[ViewRepresentation],
    state: &PreparedImageState,
    resolver: &impl ReplacementImageResolver,
) -> Result<Box<[NativeImageCopy]>, ImageBlitRecordError> {
    let (source_key, source_target, source_layout) =
        resolve_endpoint(source, representations, state, resolver)?;
    let (destination_key, destination_target, destination_layout) =
        resolve_endpoint(destination, representations, state, resolver)?;
    if source_target.pixel_format != destination_target.pixel_format {
        return Err(ImageBlitRecordError::PixelFormatMismatch(destination_key));
    }
    let aspects = native_image_copy_aspects(source.backing.pixel_format(), aspect)?;
    for &aspect in aspects.iter() {
        if !source_target.full_range.aspect_mask.contains(aspect) {
            return Err(ImageBlitRecordError::AspectUnavailable(source_key));
        }
        if !destination_target.full_range.aspect_mask.contains(aspect) {
            return Err(ImageBlitRecordError::AspectUnavailable(destination_key));
        }
    }
    let source_offset = offset(source_origin)?;
    let destination_offset = offset(destination_origin)?;
    let extent = native_extent(extent)?;
    let (source_layer, source_offset) =
        validate_copy_geometry(source_key, source, source_target, source_offset, extent)?;
    let (destination_layer, destination_offset) = validate_copy_geometry(
        destination_key,
        destination,
        destination_target,
        destination_offset,
        extent,
    )?;
    if source_target.image == destination_target.image
        && source.level == destination.level
        && source_layer == destination_layer
        && boxes_overlap(source_offset, extent, destination_offset, extent)
    {
        return Err(ImageBlitRecordError::SameImageOverlap);
    }
    Ok(aspects
        .into_vec()
        .into_iter()
        .map(|aspect| NativeImageCopy {
            source: source_target.image,
            source_layout,
            destination: destination_target.image,
            destination_layout,
            aspect,
            source_mip: source.level,
            source_layer,
            destination_mip: destination.level,
            destination_layer,
            source_offset,
            destination_offset,
            extent,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn native_image_copy_aspects(
    pixel_format: u16,
    aspect: BlitAspect,
) -> Result<Box<[vk::ImageAspectFlags]>, ImageBlitRecordError> {
    if aspect == BlitAspect::Full
        && format_has_depth_aspect(pixel_format)
        && format_has_stencil_aspect(pixel_format)
    {
        Ok(Box::new([
            vk::ImageAspectFlags::DEPTH,
            vk::ImageAspectFlags::STENCIL,
        ]))
    } else {
        Ok(Box::new([native_aspect(pixel_format, aspect)?]))
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_buffer_image_copy(
    buffer: reims_vgpu_core::ResolvedBufferRange,
    bytes_per_row: u64,
    bytes_per_image: u64,
    image: &ResolvedTextureEndpoint,
    image_origin: TextureOrigin,
    copy_extent: TextureExtent,
    aspect: BlitAspect,
    required_buffer_usage: vk::BufferUsageFlags,
    representations: &[ViewRepresentation],
    state: &PreparedImageState,
    resolver: &(impl ReplacementImageResolver + ReplacementBufferResolver),
) -> Result<NativeBufferImageCopy, ImageBlitRecordError> {
    let aspect = buffer_image_blit_aspect(image.backing.pixel_format(), aspect);
    let (image_key, image_target, image_layout) =
        resolve_endpoint(image, representations, state, resolver)?;
    let image_offset = offset(image_origin)?;
    let extent = native_extent(copy_extent)?;
    let (layer, image_offset) =
        validate_copy_geometry(image_key, image, image_target, image_offset, extent)?;
    let aspect = native_aspect(image.backing.pixel_format(), aspect)?;
    if !image_target.full_range.aspect_mask.contains(aspect) {
        return Err(ImageBlitRecordError::AspectUnavailable(image_key));
    }
    // The buffer endpoint reads or writes the bytes, so it resolves the
    // backing's byte view. A guest allocation declared as both a buffer and a
    // linear texture is one backing serving both, and taking whichever
    // representation it designated for execution would hand a copy an image
    // where it asked for a buffer.
    let representation =
        ViewRepresentation::lookup(representations, buffer.storage, BackingView::Bytes)
            .ok_or(ImageBlitRecordError::MissingRepresentation(buffer.storage))?;
    let buffer_target = resolver
        .resolve_buffer(buffer.storage, representation)
        .ok_or(ImageBlitRecordError::UnknownBuffer {
            backing: buffer.storage,
            representation,
        })?;
    if !buffer_target.usage.contains(required_buffer_usage) {
        return Err(ImageBlitRecordError::MissingBufferUsage(buffer.storage));
    }
    let bpp = u64::from(
        blit_aspect_bytes_per_pixel(image.backing.pixel_format(), aspect_from_native(aspect)?)
            .ok_or(ImageBlitRecordError::UnrepresentableBufferLayout)?,
    );
    let row_bytes = u64::from(extent[0])
        .checked_mul(bpp)
        .ok_or(ImageBlitRecordError::UnrepresentableBufferLayout)?;
    let row_stride = if bytes_per_row == 0 {
        row_bytes
    } else {
        bytes_per_row
    };
    if row_stride < row_bytes || !row_stride.is_multiple_of(bpp) {
        return Err(ImageBlitRecordError::UnrepresentableBufferLayout);
    }
    let tightly_packed_image = row_stride
        .checked_mul(u64::from(extent[1]))
        .ok_or(ImageBlitRecordError::UnrepresentableBufferLayout)?;
    let image_stride = if bytes_per_image == 0 {
        tightly_packed_image
    } else {
        bytes_per_image
    };
    if image_stride < tightly_packed_image || !image_stride.is_multiple_of(row_stride) {
        return Err(ImageBlitRecordError::UnrepresentableBufferLayout);
    }
    let required_span = u64::from(extent[2] - 1)
        .checked_mul(image_stride)
        .and_then(|span| span.checked_add(u64::from(extent[1] - 1).checked_mul(row_stride)?))
        .and_then(|span| span.checked_add(row_bytes))
        .ok_or(ImageBlitRecordError::UnrepresentableBufferLayout)?;
    if required_span > buffer.length.get()
        || required_span > buffer.region.end() - buffer.region.start()
    {
        return Err(ImageBlitRecordError::BufferRangeOutOfBounds(buffer.storage));
    }
    if buffer.region.end() > buffer_target.size {
        return Err(ImageBlitRecordError::BufferRangeOutOfBounds(buffer.storage));
    }
    let buffer_offset = buffer_target
        .base_offset
        .checked_add(buffer.region.start())
        .ok_or(ImageBlitRecordError::BufferAddressOverflow(buffer.storage))?;
    if !buffer_offset.is_multiple_of(bpp) {
        return Err(ImageBlitRecordError::UnrepresentableBufferLayout);
    }
    Ok(NativeBufferImageCopy {
        buffer: buffer_target.buffer,
        image: image_target.image,
        image_layout,
        buffer_offset,
        buffer_row_length: u32::try_from(row_stride / bpp)
            .map_err(|_| ImageBlitRecordError::UnrepresentableBufferLayout)?,
        buffer_image_height: u32::try_from(image_stride / row_stride)
            .map_err(|_| ImageBlitRecordError::UnrepresentableBufferLayout)?,
        aspect,
        mip: image.level,
        layer,
        image_offset,
        extent,
    })
}

fn aspect_from_native(aspect: vk::ImageAspectFlags) -> Result<BlitAspect, ImageBlitRecordError> {
    if aspect == vk::ImageAspectFlags::DEPTH {
        Ok(BlitAspect::Depth)
    } else if aspect == vk::ImageAspectFlags::STENCIL {
        Ok(BlitAspect::Stencil)
    } else if aspect == vk::ImageAspectFlags::COLOR {
        Ok(BlitAspect::Full)
    } else {
        Err(ImageBlitRecordError::CombinedDepthStencilUnsupported)
    }
}

fn resolve_endpoint(
    endpoint: &ResolvedTextureEndpoint,
    representations: &[ViewRepresentation],
    state: &PreparedImageState,
    resolver: &impl ReplacementImageResolver,
) -> Result<(ReplacementImageKey, NativeImageTarget, vk::ImageLayout), ImageBlitRecordError> {
    let representation = ViewRepresentation::lookup(
        representations,
        endpoint.storage,
        BackingView::Image(reims_vgpu_core::ImageOwner::owning(endpoint.image_owner)),
    )
    .ok_or(ImageBlitRecordError::MissingRepresentation(
        endpoint.storage,
    ))?;
    let key = ReplacementImageKey {
        backing: endpoint.storage,
        representation,
    };
    let target = resolver
        .resolve_image(key)
        .ok_or(ImageBlitRecordError::UnknownImage(key))?;
    if target.pixel_format != endpoint.backing.pixel_format() {
        return Err(ImageBlitRecordError::PixelFormatMismatch(key));
    }
    let mip_end = target
        .full_range
        .base_mip_level
        .checked_add(target.full_range.level_count)
        .ok_or(ImageBlitRecordError::SubresourceOutOfBounds(key))?;
    if endpoint.level < target.full_range.base_mip_level || endpoint.level >= mip_end {
        return Err(ImageBlitRecordError::SubresourceOutOfBounds(key));
    }
    let layout = state
        .transitions()
        .iter()
        .find(|transition| transition.image == key)
        .map(|transition| transition.use_layout)
        .ok_or(ImageBlitRecordError::StateUseMismatch(key))?;
    Ok((key, target, layout))
}

fn validate_copy_geometry(
    key: ReplacementImageKey,
    endpoint: &ResolvedTextureEndpoint,
    target: NativeImageTarget,
    offset: [i32; 3],
    extent: [u32; 3],
) -> Result<(u32, [i32; 3]), ImageBlitRecordError> {
    let relative_mip = endpoint
        .level
        .checked_sub(target.full_range.base_mip_level)
        .ok_or(ImageBlitRecordError::SubresourceOutOfBounds(key))?;
    let mip_extent = [
        mip_dimension(target.extent.width, relative_mip),
        mip_dimension(target.extent.height, relative_mip),
        mip_dimension(target.extent.depth, relative_mip),
    ];
    let contract_extent = [
        endpoint.backing.width(),
        endpoint.backing.height(),
        endpoint.backing.depth(),
    ];
    let layer = match target.image_type {
        vk::ImageType::TYPE_1D => {
            if offset[1] != 0 || offset[2] != 0 || extent[1] != 1 || extent[2] != 1 {
                return Err(ImageBlitRecordError::IncompatibleImageGeometry(key));
            }
            validate_array_layer(key, endpoint.slice, target.full_range)?
        }
        vk::ImageType::TYPE_2D => {
            if offset[2] != 0 || extent[2] != 1 {
                return Err(ImageBlitRecordError::IncompatibleImageGeometry(key));
            }
            validate_array_layer(key, endpoint.slice, target.full_range)?
        }
        vk::ImageType::TYPE_3D => {
            if endpoint.slice != 0
                || target.full_range.base_array_layer != 0
                || target.full_range.layer_count != 1
            {
                return Err(ImageBlitRecordError::IncompatibleImageGeometry(key));
            }
            0
        }
        _ => return Err(ImageBlitRecordError::IncompatibleImageGeometry(key)),
    };
    for axis in 0..3 {
        let start = u32::try_from(offset[axis])
            .map_err(|_| ImageBlitRecordError::SubresourceOutOfBounds(key))?;
        let end = start
            .checked_add(extent[axis])
            .ok_or(ImageBlitRecordError::SubresourceOutOfBounds(key))?;
        if extent[axis] == 0 || end > mip_extent[axis] || end > contract_extent[axis] {
            return Err(ImageBlitRecordError::SubresourceOutOfBounds(key));
        }
    }
    Ok((layer, offset))
}

fn validate_array_layer(
    key: ReplacementImageKey,
    layer: u32,
    range: vk::ImageSubresourceRange,
) -> Result<u32, ImageBlitRecordError> {
    let end = range
        .base_array_layer
        .checked_add(range.layer_count)
        .ok_or(ImageBlitRecordError::SubresourceOutOfBounds(key))?;
    if layer < range.base_array_layer || layer >= end {
        return Err(ImageBlitRecordError::SubresourceOutOfBounds(key));
    }
    Ok(layer)
}

fn mip_dimension(base: u32, level: u32) -> u32 {
    base.checked_shr(level).unwrap_or(0).max(1)
}

fn native_aspect(
    pixel_format: u16,
    aspect: BlitAspect,
) -> Result<vk::ImageAspectFlags, ImageBlitRecordError> {
    Ok(match aspect {
        BlitAspect::Depth => vk::ImageAspectFlags::DEPTH,
        BlitAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        BlitAspect::Full => match (
            format_has_depth_aspect(pixel_format),
            format_has_stencil_aspect(pixel_format),
        ) {
            (false, false) => vk::ImageAspectFlags::COLOR,
            (true, false) => vk::ImageAspectFlags::DEPTH,
            (false, true) => vk::ImageAspectFlags::STENCIL,
            (true, true) => return Err(ImageBlitRecordError::CombinedDepthStencilUnsupported),
        },
    })
}

fn offset(origin: TextureOrigin) -> Result<[i32; 3], ImageBlitRecordError> {
    Ok([
        i32::try_from(origin.x).map_err(|_| ImageBlitRecordError::CoordinateOverflow)?,
        i32::try_from(origin.y).map_err(|_| ImageBlitRecordError::CoordinateOverflow)?,
        i32::try_from(origin.z).map_err(|_| ImageBlitRecordError::CoordinateOverflow)?,
    ])
}

fn native_extent(extent: TextureExtent) -> Result<[u32; 3], ImageBlitRecordError> {
    Ok([
        u32::try_from(extent.width).map_err(|_| ImageBlitRecordError::CoordinateOverflow)?,
        u32::try_from(extent.height).map_err(|_| ImageBlitRecordError::CoordinateOverflow)?,
        u32::try_from(extent.depth).map_err(|_| ImageBlitRecordError::CoordinateOverflow)?,
    ])
}

fn boxes_overlap(
    left: [i32; 3],
    left_extent: [u32; 3],
    right: [i32; 3],
    right_extent: [u32; 3],
) -> bool {
    (0..3).all(|axis| {
        let left_end = i64::from(left[axis]) + i64::from(left_extent[axis]);
        let right_end = i64::from(right[axis]) + i64::from(right_extent[axis]);
        i64::from(left[axis]) < right_end && i64::from(right[axis]) < left_end
    })
}

/// Record all image-to-image copies between the matching pre/post barriers.
///
/// # Safety
///
/// The command buffer must be recording, and the prepared resource uses must
/// remain retained through queue retirement.
pub unsafe fn record_native_image_copies(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    prepared: &PreparedNativeImageBlit,
) {
    crate::replacement_barrier_record::record_hazard_barriers(
        device,
        command_buffer,
        &prepared.state.transitions.before,
    );
    unsafe { record_native_image_commands(device, command_buffer, &prepared.commands) };
    crate::replacement_barrier_record::record_hazard_barriers(
        device,
        command_buffer,
        &prepared.state.transitions.after,
    );
}

/// Record an already-transitioned image command set.
///
/// # Safety
///
/// The command buffer must be recording and the caller must record the exact
/// prepared image-state barriers around these commands.
pub unsafe fn record_native_image_commands(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    commands: &[NativeImageBlitCommand],
) {
    for command in commands.iter().copied() {
        match command {
            NativeImageBlitCommand::ImageToImage(copy) => {
                let region = vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: copy.aspect,
                        mip_level: copy.source_mip,
                        base_array_layer: copy.source_layer,
                        layer_count: 1,
                    },
                    src_offset: vk::Offset3D {
                        x: copy.source_offset[0],
                        y: copy.source_offset[1],
                        z: copy.source_offset[2],
                    },
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: copy.aspect,
                        mip_level: copy.destination_mip,
                        base_array_layer: copy.destination_layer,
                        layer_count: 1,
                    },
                    dst_offset: vk::Offset3D {
                        x: copy.destination_offset[0],
                        y: copy.destination_offset[1],
                        z: copy.destination_offset[2],
                    },
                    extent: vk::Extent3D {
                        width: copy.extent[0],
                        height: copy.extent[1],
                        depth: copy.extent[2],
                    },
                };
                unsafe {
                    device.cmd_copy_image(
                        command_buffer,
                        copy.source,
                        copy.source_layout,
                        copy.destination,
                        copy.destination_layout,
                        &[region],
                    );
                }
            }
            NativeImageBlitCommand::BufferToImage(copy) => {
                let region = buffer_image_region(copy);
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        copy.buffer,
                        copy.image,
                        copy.image_layout,
                        &[region],
                    );
                }
            }
            NativeImageBlitCommand::ImageToBuffer(copy) => {
                let region = buffer_image_region(copy);
                unsafe {
                    device.cmd_copy_image_to_buffer(
                        command_buffer,
                        copy.image,
                        copy.image_layout,
                        copy.buffer,
                        &[region],
                    );
                }
            }
        }
    }
}

fn buffer_image_region(copy: NativeBufferImageCopy) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: copy.buffer_offset,
        buffer_row_length: copy.buffer_row_length,
        buffer_image_height: copy.buffer_image_height,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: copy.aspect,
            mip_level: copy.mip,
            base_array_layer: copy.layer,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: copy.image_offset[0],
            y: copy.image_offset[1],
            z: copy.image_offset[2],
        },
        image_extent: vk::Extent3D {
            width: copy.extent[0],
            height: copy.extent[1],
            depth: copy.extent[2],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replacement_buffer_blit::{NativeBufferTarget, ReplacementBufferResolver};
    use crate::replacement_image_state::{ReplacementImageSharing, ReplacementImageState};
    use ash::vk::Handle;
    use reims_vgpu_core::{
        pixel_format::BlitAspect, LinearRange, ResolvedBufferRange, ResolvedLinearTextureLevel,
        ResolvedTextureBacking, ResolvedTextureEndpoint, ResolvedTextureToTextureBlit,
        TextureExtent, TextureOrigin,
    };
    use reims_vgpu_protocol::{
        ByteLength, GuestVirtualAddress, RepresentationId, ResourceId, TransactionId,
    };

    struct General;

    impl ReplacementImageFinalLayout for General {
        fn final_layout(
            &self,
            _image: ReplacementImageKey,
            _required_usage: vk::ImageUsageFlags,
        ) -> Option<vk::ImageLayout> {
            Some(vk::ImageLayout::GENERAL)
        }
    }

    fn endpoint(backing: u64, resource: u32) -> ResolvedTextureEndpoint {
        ResolvedTextureEndpoint {
            resource: ResourceId::new(resource, 1),
            image_owner: ResourceId::new(resource, 1),
            storage: reims_vgpu_protocol::BackingId::new(backing),
            level: 0,
            slice: 0,
            backing: ResolvedTextureBacking::Linear(ResolvedLinearTextureLevel {
                base_gva: backing << 12,
                alloc_size: 4096,
                level_offset: 0,
                row_stride: 64,
                slice_stride: 0,
                slice_index: 0,
                width: 8,
                height: 8,
                depth: 1,
                bpp: 4,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            }),
        }
    }

    struct Images(BTreeMap<ReplacementImageKey, NativeImageTarget>);

    impl ReplacementImageResolver for Images {
        fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
            self.0.get(&image).copied()
        }
    }

    impl ReplacementBufferResolver for Images {
        fn resolve_buffer(
            &self,
            _backing: BackingId,
            _representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            None
        }
    }

    struct BufferAndImage {
        images: Images,
        backing: BackingId,
        representation: RepresentationId,
        buffer: NativeBufferTarget,
    }

    impl ReplacementImageResolver for BufferAndImage {
        fn resolve_image(&self, image: ReplacementImageKey) -> Option<NativeImageTarget> {
            self.images.resolve_image(image)
        }
    }

    impl ReplacementBufferResolver for BufferAndImage {
        fn resolve_buffer(
            &self,
            backing: BackingId,
            representation: RepresentationId,
        ) -> Option<NativeBufferTarget> {
            (backing == self.backing && representation == self.representation)
                .then_some(self.buffer)
        }
    }

    fn target(image: u64, image_type: vk::ImageType, layers: u32) -> NativeImageTarget {
        NativeImageTarget {
            image: vk::Image::from_raw(image),
            view: vk::ImageView::null(),
            image_type,
            full_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layers,
            },
            usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            extent: vk::Extent3D {
                width: 8,
                height: 8,
                depth: if image_type == vk::ImageType::TYPE_3D {
                    8
                } else {
                    1
                },
            },
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    fn prepared_state(
        source: ReplacementImageKey,
        destination: ReplacementImageKey,
    ) -> PreparedImageState {
        let mut owner =
            ReplacementImageStateOwner::new(reims_vgpu_protocol::VulkanDeviceEpochId::new(1));
        for image in [source, destination] {
            owner
                .register(
                    image,
                    ReplacementImageState {
                        layout: vk::ImageLayout::GENERAL,
                        sharing: ReplacementImageSharing::Concurrent,
                        last_use: None,
                    },
                )
                .unwrap();
        }
        owner
            .prepare(
                TransactionId::new(1),
                2,
                [
                    ReplacementImageUse {
                        image: source,
                        required_usage: vk::ImageUsageFlags::TRANSFER_SRC,
                        use_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    },
                    ReplacementImageUse {
                        image: destination,
                        required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                        use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout: vk::ImageLayout::GENERAL,
                    },
                ],
            )
            .unwrap()
    }

    #[test]
    fn distinct_images_use_transfer_optimal_layouts() {
        let operation = ResolvedBlit::TextureToTexture(ResolvedTextureToTextureBlit {
            source: endpoint(1, 1),
            source_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            destination: endpoint(2, 2),
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 8,
                height: 8,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });
        let uses = derive_image_uses(
            &operation,
            &[
                ViewRepresentation {
                    backing: BackingId::new(1),
                    view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(ResourceId::new(
                        1, 1,
                    ))),
                    representation: RepresentationId::new(11),
                },
                ViewRepresentation {
                    backing: BackingId::new(2),
                    view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(ResourceId::new(
                        2, 1,
                    ))),
                    representation: RepresentationId::new(12),
                },
            ],
            &General,
        )
        .unwrap();
        assert_eq!(uses[0].use_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(uses[0].required_usage, vk::ImageUsageFlags::TRANSFER_SRC);
        assert_eq!(uses[1].use_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        assert_eq!(uses[1].required_usage, vk::ImageUsageFlags::TRANSFER_DST);
    }

    #[test]
    fn same_image_copy_has_one_general_layout_with_combined_usage() {
        let operation = ResolvedBlit::TextureToTexture(ResolvedTextureToTextureBlit {
            source: endpoint(1, 1),
            source_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            // One texture, so one image: two resources over one backing are
            // two textures and would be two images.
            destination: endpoint(1, 1),
            destination_origin: TextureOrigin { x: 4, y: 4, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 4,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });
        let uses = derive_image_uses(
            &operation,
            &[ViewRepresentation {
                backing: BackingId::new(1),
                view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(ResourceId::new(
                    1, 1,
                ))),
                representation: RepresentationId::new(11),
            }],
            &General,
        )
        .unwrap();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].use_layout, vk::ImageLayout::GENERAL);
        assert_eq!(
            uses[0].required_usage,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
        );
    }

    #[test]
    fn native_copy_projects_exact_layers_offsets_extent_and_layouts() {
        let mut source = endpoint(1, 1);
        source.slice = 2;
        let mut destination = endpoint(2, 2);
        destination.slice = 4;
        let source_key = ReplacementImageKey {
            backing: source.storage,
            representation: RepresentationId::new(11),
        };
        let destination_key = ReplacementImageKey {
            backing: destination.storage,
            representation: RepresentationId::new(12),
        };
        let state = prepared_state(source_key, destination_key);
        let resolver = Images(BTreeMap::from([
            (source_key, target(21, vk::ImageType::TYPE_2D, 6)),
            (destination_key, target(22, vk::ImageType::TYPE_2D, 6)),
        ]));
        let copies = resolve_image_copies(
            &source,
            TextureOrigin { x: 1, y: 2, z: 0 },
            &destination,
            TextureOrigin { x: 3, y: 4, z: 0 },
            TextureExtent {
                width: 2,
                height: 3,
                depth: 1,
            },
            BlitAspect::Full,
            &[
                ViewRepresentation {
                    backing: source.storage,
                    view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(source.resource)),
                    representation: source_key.representation,
                },
                ViewRepresentation {
                    backing: destination.storage,
                    view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(
                        destination.resource,
                    )),
                    representation: destination_key.representation,
                },
            ],
            &state,
            &resolver,
        )
        .unwrap();
        assert_eq!(copies.len(), 1);
        let copy = copies[0];
        assert_eq!(copy.source, vk::Image::from_raw(21));
        assert_eq!(copy.destination, vk::Image::from_raw(22));
        assert_eq!(copy.source_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(
            copy.destination_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );
        assert_eq!(copy.source_layer, 2);
        assert_eq!(copy.destination_layer, 4);
        assert_eq!(copy.source_offset, [1, 2, 0]);
        assert_eq!(copy.destination_offset, [3, 4, 0]);
        assert_eq!(copy.extent, [2, 3, 1]);
    }

    #[test]
    fn three_dimensional_image_refuses_array_slice_addressing() {
        let mut source = endpoint(1, 1);
        source.slice = 1;
        let destination = endpoint(2, 2);
        let source_key = ReplacementImageKey {
            backing: source.storage,
            representation: RepresentationId::new(11),
        };
        let destination_key = ReplacementImageKey {
            backing: destination.storage,
            representation: RepresentationId::new(12),
        };
        let state = prepared_state(source_key, destination_key);
        let resolver = Images(BTreeMap::from([
            (source_key, target(21, vk::ImageType::TYPE_3D, 1)),
            (destination_key, target(22, vk::ImageType::TYPE_2D, 1)),
        ]));
        assert_eq!(
            resolve_image_copies(
                &source,
                TextureOrigin { x: 0, y: 0, z: 0 },
                &destination,
                TextureOrigin { x: 0, y: 0, z: 0 },
                TextureExtent {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                BlitAspect::Full,
                &[
                    ViewRepresentation {
                        backing: source.storage,
                        view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(
                            source.resource
                        )),
                        representation: source_key.representation,
                    },
                    ViewRepresentation {
                        backing: destination.storage,
                        view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(
                            destination.resource
                        )),
                        representation: destination_key.representation,
                    },
                ],
                &state,
                &resolver,
            ),
            Err(ImageBlitRecordError::IncompatibleImageGeometry(source_key))
        );
    }

    #[test]
    fn full_combined_depth_stencil_image_copy_projects_both_native_planes() {
        assert_eq!(
            native_image_copy_aspects(
                reims_vgpu_core::pixel_format::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
                BlitAspect::Full,
            )
            .unwrap()
            .as_ref(),
            [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL]
        );
        assert_eq!(
            native_image_copy_aspects(
                reims_vgpu_core::pixel_format::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
                BlitAspect::Depth,
            )
            .unwrap()
            .as_ref(),
            [vk::ImageAspectFlags::DEPTH]
        );
    }

    #[test]
    fn buffer_image_copy_projects_contract_strides_and_native_base_offset() {
        let mut image = endpoint(2, 2);
        let ResolvedTextureBacking::Linear(backing) = &mut image.backing else {
            unreachable!()
        };
        backing.depth = 8;
        let image_key = ReplacementImageKey {
            backing: image.storage,
            representation: RepresentationId::new(12),
        };
        let mut owner =
            ReplacementImageStateOwner::new(reims_vgpu_protocol::VulkanDeviceEpochId::new(1));
        owner
            .register(
                image_key,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let state = owner
            .prepare(
                TransactionId::new(1),
                2,
                [ReplacementImageUse {
                    image: image_key,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                }],
            )
            .unwrap();
        let buffer_backing = BackingId::new(1);
        let buffer_representation = RepresentationId::new(11);
        let resolver = BufferAndImage {
            images: Images(BTreeMap::from([(
                image_key,
                target(22, vk::ImageType::TYPE_3D, 1),
            )])),
            backing: buffer_backing,
            representation: buffer_representation,
            buffer: NativeBufferTarget {
                buffer: vk::Buffer::from_raw(31),
                base_offset: 100,
                accessible_size: 256,
                size: 256,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
            },
        };
        let buffer = ResolvedBufferRange {
            resource: ResourceId::new(1, 1),
            storage: buffer_backing,
            region: LinearRange::new(16, 128).unwrap(),
            address: GuestVirtualAddress::new(0x1010),
            length: ByteLength::new(112),
        };
        let representations = [
            ViewRepresentation {
                backing: buffer_backing,
                view: BackingView::Bytes,
                representation: buffer_representation,
            },
            ViewRepresentation {
                backing: image.storage,
                view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(image.resource)),
                representation: image_key.representation,
            },
        ];
        let copy = resolve_buffer_image_copy(
            buffer,
            16,
            64,
            &image,
            TextureOrigin { x: 1, y: 2, z: 3 },
            TextureExtent {
                width: 2,
                height: 3,
                depth: 2,
            },
            BlitAspect::Full,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &representations,
            &state,
            &resolver,
        )
        .unwrap();
        assert_eq!(copy.buffer_offset, 116);
        assert_eq!(copy.buffer_row_length, 4);
        assert_eq!(copy.buffer_image_height, 4);
        assert_eq!(copy.image_offset, [1, 2, 3]);
        assert_eq!(copy.extent, [2, 3, 2]);

        assert_eq!(
            resolve_buffer_image_copy(
                buffer,
                7,
                64,
                &image,
                TextureOrigin { x: 0, y: 0, z: 0 },
                TextureExtent {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                BlitAspect::Full,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &representations,
                &state,
                &resolver,
            ),
            Err(ImageBlitRecordError::UnrepresentableBufferLayout)
        );
    }

    #[test]
    fn combined_depth_stencil_plane_copy_uses_its_compatible_buffer_format() {
        let mut image = endpoint(2, 2);
        let ResolvedTextureBacking::Linear(backing) = &mut image.backing else {
            unreachable!()
        };
        backing.pixel_format = reims_vgpu_core::pixel_format::MTL_FORMAT_DEPTH24_UNORM_STENCIL8;
        backing.bpp = 4;
        let image_key = ReplacementImageKey {
            backing: image.storage,
            representation: RepresentationId::new(12),
        };
        let mut owner =
            ReplacementImageStateOwner::new(reims_vgpu_protocol::VulkanDeviceEpochId::new(1));
        owner
            .register(
                image_key,
                ReplacementImageState {
                    layout: vk::ImageLayout::GENERAL,
                    sharing: ReplacementImageSharing::Concurrent,
                    last_use: None,
                },
            )
            .unwrap();
        let state = owner
            .prepare(
                TransactionId::new(1),
                2,
                [ReplacementImageUse {
                    image: image_key,
                    required_usage: vk::ImageUsageFlags::TRANSFER_DST,
                    use_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout: vk::ImageLayout::GENERAL,
                }],
            )
            .unwrap();
        let buffer_backing = BackingId::new(1);
        let buffer_representation = RepresentationId::new(11);
        let mut image_target = target(22, vk::ImageType::TYPE_2D, 1);
        image_target.pixel_format = backing.pixel_format;
        image_target.full_range.aspect_mask =
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        let resolver = BufferAndImage {
            images: Images(BTreeMap::from([(image_key, image_target)])),
            backing: buffer_backing,
            representation: buffer_representation,
            buffer: NativeBufferTarget {
                buffer: vk::Buffer::from_raw(31),
                base_offset: 0,
                accessible_size: 64,
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
            },
        };
        let copy = resolve_buffer_image_copy(
            ResolvedBufferRange {
                resource: ResourceId::new(1, 1),
                storage: buffer_backing,
                region: LinearRange::new(0, 32).unwrap(),
                address: GuestVirtualAddress::new(0x1000),
                length: ByteLength::new(32),
            },
            16,
            0,
            &image,
            TextureOrigin { x: 0, y: 0, z: 0 },
            TextureExtent {
                width: 4,
                height: 2,
                depth: 1,
            },
            BlitAspect::Full,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &[
                ViewRepresentation {
                    backing: buffer_backing,
                    view: BackingView::Bytes,
                    representation: buffer_representation,
                },
                ViewRepresentation {
                    backing: image.storage,
                    view: BackingView::Image(reims_vgpu_core::ImageOwner::owning(image.resource)),
                    representation: image_key.representation,
                },
            ],
            &state,
            &resolver,
        )
        .unwrap();
        assert_eq!(copy.aspect, vk::ImageAspectFlags::DEPTH);
        assert_eq!(copy.buffer_row_length, 4);
        assert_eq!(copy.buffer_image_height, 2);

        let native_state = resolve_image_transitions(&state, &resolver).unwrap();
        let depth_copy = PreparedNativeImageBlit {
            state: native_state.clone(),
            commands: Box::new([NativeImageBlitCommand::BufferToImage(copy)]),
        };
        assert_eq!(depth_copy.required_queue_flags(), vk::QueueFlags::GRAPHICS);

        let color_copy = PreparedNativeImageBlit {
            state: native_state,
            commands: Box::new([NativeImageBlitCommand::BufferToImage(
                NativeBufferImageCopy {
                    aspect: vk::ImageAspectFlags::COLOR,
                    ..copy
                },
            )]),
        };
        assert_eq!(color_copy.required_queue_flags(), vk::QueueFlags::empty());
    }
}
