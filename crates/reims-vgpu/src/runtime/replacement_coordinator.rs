//! Production-shaped ownership transitions above replacement packet services.
//!
//! Packet decoding/admission, semantic mutation, host publication, and ordered
//! completion are separate fallible phases. Each transition consumes its input
//! and every refusal returns the furthest owner, so a coordinator can retry
//! without decoding a packet twice or repeating an already completed host
//! effect.

#![allow(dead_code)]

use crate::runtime::host::{HostAction, HostControl, HostMemory, MemError};
use crate::runtime::replacement_session::{
    ReplacementAdmittedControl, ReplacementAdmittedQuery, ReplacementAdmittedResourceLifecycle,
    ReplacementAppliedControl, ReplacementAppliedQuery, ReplacementAppliedResourceLifecycle,
    ReplacementControlApplyError, ReplacementControlCompletionError, ReplacementControlEffect,
    ReplacementQueryApplyError, ReplacementQueryCompletionError, ReplacementQueryEffect,
    ReplacementResourceLifecycleApplyError, ReplacementResourceLifecycleCompletionError,
    ReplacementRuntimeSession,
};

pub(crate) fn read_replacement_task_bytes<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    host: &impl HostMemory,
    page_shift: u32,
    task: reims_vgpu_protocol::TaskId,
    gva: u64,
    length: usize,
) -> Result<
    Vec<u8>,
    crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError,
> {
    read_replacement_task_bytes_from_table(runtime.tasks(), host, page_shift, task, gva, length)
}

fn read_replacement_task_bytes_from_table(
    tasks: &reims_vgpu_core::TaskTable,
    host: &impl HostMemory,
    page_shift: u32,
    task: reims_vgpu_protocol::TaskId,
    gva: u64,
    length: usize,
) -> Result<
    Vec<u8>,
    crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError,
> {
    let task = tasks.get(task.get()).ok_or(
        crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError::Memory(
            MemError::NoSuchTask,
        ),
    )?;
    let mut bytes = vec![0; length];
    crate::runtime::gva_mem::read_task_gva(host, task, gva, &mut bytes, page_shift).map_err(
        crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError::Memory,
    )?;
    Ok(bytes)
}

pub(crate) enum ReplacementHostExecDispatchFailure<Semantic> {
    Load {
        reason: crate::runtime::replacement_session::ReplacementExecPacketLoadError,
        packet: crate::runtime::replacement_child_packet::ReplacementDeferredExecPacket,
    },
    ObjectPreparation(
        Box<crate::runtime::replacement_session::ReplacementLoadedExecObjectPreparationFailure>,
    ),
    Representation {
        reason: ReplacementObjectRepresentationPreparationError,
        ready: Box<crate::runtime::replacement_session::ReplacementObjectReadyExecPacket>,
    },
    Dispatch {
        reason:
            crate::runtime::replacement_session::ReplacementExecIngressDispatchFailure<Semantic>,
        ready: Box<crate::runtime::replacement_session::ReplacementObjectReadyExecPacket>,
    },
    BackingRepresentation {
        backing: reims_vgpu_protocol::BackingId,
        reason: ReplacementObjectRepresentationPreparationError,
        dispatch:
            crate::runtime::replacement_session::ReplacementExecIngressDispatchFailure<Semantic>,
        ready: Box<crate::runtime::replacement_session::ReplacementObjectReadyExecPacket>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementObjectRepresentationPreparationError {
    SurfaceDescriptorUnavailable(
        reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
    ),
    TaskUnavailable(reims_vgpu_protocol::TaskId),
    UnsupportedPageShift(u32),
    PageCountOverflow(u64),
    PageAddressOverflow {
        page: u64,
    },
    PageUnmapped {
        task: reims_vgpu_protocol::TaskId,
        page: u64,
    },
    BackingUnavailable {
        backing: reims_vgpu_protocol::BackingId,
        reason: crate::runtime::replacement_session::ReplacementTaskAddressMaterializationRefusal,
    },
    GuestMap(crate::runtime::guest_ram_map::MapRefusal),
    GuestWindow(reims_vgpu_memory::GuestWindowError),
    Construction(crate::runtime::replacement_session::ReplacementRepresentationConstructionError),
}

impl ReplacementObjectRepresentationPreparationError {
    /// See [`crate::runtime::replacement_session::ReplacementRepresentationConstructionError::is_unimplemented_case`].
    ///
    /// Only construction carries a declared refusal. Everything else here is a
    /// task, page, mapping or backing this device is waiting to learn about,
    /// and a later packet on another channel is exactly what delivers it.
    const fn is_unimplemented_case(&self) -> bool {
        match self {
            Self::Construction(reason) => reason.is_unimplemented_case(),
            Self::SurfaceDescriptorUnavailable(_)
            | Self::TaskUnavailable(_)
            | Self::UnsupportedPageShift(_)
            | Self::PageCountOverflow(_)
            | Self::PageAddressOverflow { .. }
            | Self::PageUnmapped { .. }
            | Self::BackingUnavailable { .. }
            | Self::GuestMap(_)
            | Self::GuestWindow(_) => false,
        }
    }
}

impl<Semantic> ReplacementHostExecDispatchFailure<Semantic> {
    /// See [`crate::runtime::replacement_session::ReplacementExecAutomaticPreparationError::stale_backing`].
    fn stale_backing(&self) -> Option<reims_vgpu_protocol::BackingId> {
        match self {
            Self::Dispatch { reason, .. } => reason.stale_backing(),
            _ => None,
        }
    }

    /// See [`crate::runtime::replacement_session::ReplacementRepresentationConstructionError::is_unimplemented_case`].
    const fn is_unimplemented_case(&self) -> bool {
        match self {
            Self::Representation { reason, .. } | Self::BackingRepresentation { reason, .. } => {
                reason.is_unimplemented_case()
            }
            // A dispatch failure naming an object-table slot the guest has
            // released can never be repaired by a later packet, so re-offering
            // it parks the channel for the life of the device.
            Self::Dispatch { reason, .. } => reason.is_terminal_refusal(),
            Self::Load { .. } | Self::ObjectPreparation(_) => false,
        }
    }
}

/// A deferred child packet this backend has declared it does not implement,
/// carrying the ring lease that refusing it must consume.
///
/// A refusal cannot be expressed without a lease, which is the invariant: an
/// arm that classified itself as unimplemented but held no lease would consume
/// nothing, and the channel would go back to being blocked by a packet nothing
/// could spend.
pub(crate) struct ReplacementRefusedChildPacket {
    lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    detail: String,
}

impl ReplacementDeferredChildDispatchFailure<()> {
    /// Classify a dispatch failure as a declared refusal, or hand it back to be
    /// re-offered.
    ///
    /// See
    /// [`crate::runtime::replacement_session::ReplacementRepresentationConstructionError::is_unimplemented_case`]
    /// for what makes a reason declared rather than pending.
    fn into_refusal(self: Box<Self>) -> Result<ReplacementRefusedChildPacket, Box<Self>> {
        match *self {
            Self::Exec { failure, lease } if failure.is_unimplemented_case() => {
                let detail = replacement_host_exec_failure_diagnostic(&failure);
                Ok(ReplacementRefusedChildPacket { lease, detail })
            }
            failure => Err(Box::new(failure)),
        }
    }
}

fn replacement_prepared_recording_error_diagnostic<Completion>(
    reason: &crate::runtime::replacement_session::ReplacementPreparedAdmittedRecordingError<
        Completion,
    >,
) -> String {
    use crate::runtime::replacement_session::ReplacementPreparedAdmittedRecordingError as Error;
    match reason {
        Error::RecordingNotReady(transaction) => format!("RecordingNotReady({transaction:?})"),
        Error::MissingRecordingPlan(transaction) => {
            format!("MissingRecordingPlan({transaction:?})")
        }
        Error::Assignment(reason) => format!("Assignment({reason:?})"),
        Error::MissingHazardPlan(transaction) => {
            format!("MissingHazardPlan({transaction:?})")
        }
        Error::BarrierPlan(reason) => format!("BarrierPlan({reason:?})"),
        Error::BarrierResolution(reason) => format!("BarrierResolution({reason:?})"),
        Error::IndirectRangeContinuationRequired(transaction) => {
            format!("IndirectRangeContinuationRequired({transaction:?})")
        }
        Error::Program(failure) => format!("Program({:?})", failure.reason),
    }
}

fn replacement_host_exec_failure_diagnostic(
    failure: &ReplacementHostExecDispatchFailure<()>,
) -> String {
    use crate::runtime::replacement_session::ReplacementExecIngressDispatchFailure as Dispatch;
    match failure {
        ReplacementHostExecDispatchFailure::Load { reason, .. } => {
            format!("stage=load reason={reason:?}")
        }
        ReplacementHostExecDispatchFailure::ObjectPreparation(failure) => {
            replacement_object_preparation_failure_diagnostic(failure)
        }
        ReplacementHostExecDispatchFailure::Representation { reason, .. } => {
            format!("stage=object_representation reason={reason:?}")
        }
        ReplacementHostExecDispatchFailure::Dispatch { reason, .. } => match reason {
            Dispatch::Ingress(reason) => replacement_ingress_preparation_diagnostic(reason),
            Dispatch::DirectResources(reason) => match reason {
                crate::runtime::replacement_session::ReplacementResourceReadyExecFailure::Readiness { reason, .. } => {
                    format!("stage=direct_resources.readiness reason={reason:?}")
                }
                crate::runtime::replacement_session::ReplacementResourceReadyExecFailure::Resources { reason, .. } => {
                    format!(
                        "stage=direct_resources.resources {}",
                        replacement_automatic_preparation_diagnostic(reason)
                    )
                }
                crate::runtime::replacement_session::ReplacementResourceReadyExecFailure::Images { reason, .. } => {
                    format!("stage=direct_resources.images reason={reason:?}")
                }
            },
            Dispatch::DirectDispatch(reason) => match reason {
                crate::runtime::replacement_session::ReplacementResourceReadyDispatchFailure::Resolution(failure) => {
                    format!(
                        "stage=direct_dispatch.resolution reason={}",
                        replacement_prepared_recording_error_diagnostic(&failure.reason)
                    )
                }
                crate::runtime::replacement_session::ReplacementResourceReadyDispatchFailure::Dispatch(failure) => {
                    format!("stage=direct_dispatch.worker reason={:?}", failure.reason)
                }
            },
            Dispatch::GuestUploadPhase(reason) => match reason.as_ref() {
                crate::runtime::replacement_session::ReplacementGuestUploadPhasePreparationFailure::NoUpload { .. } => {
                    "stage=guest_upload.phase reason=no_upload".to_string()
                }
                crate::runtime::replacement_session::ReplacementGuestUploadPhasePreparationFailure::Chain { reason, .. } => {
                    format!("stage=guest_upload.chain reason={reason:?}")
                }
                crate::runtime::replacement_session::ReplacementGuestUploadPhasePreparationFailure::Resources { reason, .. } => {
                    format!(
                        "stage=guest_upload.resources {}",
                        replacement_automatic_preparation_diagnostic(reason)
                    )
                }
                crate::runtime::replacement_session::ReplacementGuestUploadPhasePreparationFailure::Images { reason, .. } => {
                    format!("stage=guest_upload.images reason={reason:?}")
                }
            },
            Dispatch::GuestUploadResolution(failure) => {
                format!(
                    "stage=guest_upload.resolution reason={}",
                    replacement_prepared_recording_error_diagnostic(&failure.reason)
                )
            }
            Dispatch::GuestUploadDispatch(failure) => {
                format!("stage=guest_upload.worker reason={:?}", failure.reason)
            }
            Dispatch::IndirectRangeReadiness { reason, .. } => {
                format!("stage=indirect_range.readiness reason={reason:?}")
            }
            Dispatch::IndirectRange(_) => {
                "stage=indirect_range reason=phase_dispatch_refused".to_string()
            }
        },
        ReplacementHostExecDispatchFailure::BackingRepresentation {
            backing, reason, ..
        } => format!("stage=backing_representation backing={backing:?} reason={reason:?}"),
    }
}

/// The submission positions in `order` that `owned` does not account for.
///
/// Two filters, and the second is the one that took a boot to get right.
///
/// A settled transaction is not one of these: `submitted` and `abandoned` are
/// the two transitions that release a domain claim, so an entry carrying either
/// owes nothing to anybody.
///
/// Neither is a transaction that has reached *neither* `recorded` nor `issued`.
/// That is the ordinary state of an EXEC whose prerequisites are still unmet:
/// it was accepted into the order, it holds its position, and no coordinator
/// owns it because none has started it -- the runtime does, and the runtime is
/// not one of the sets this diffs against. The first boot to carry this census
/// named exactly such a transaction, waiting on a resource hazard its producer
/// had not cleared, and reported a healthy device as leaking. An owner is owed
/// only once one has been taken.
fn unowned_submission_positions(
    order: &[reims_vgpu_core::SubmissionOrderEntry],
    owned: &std::collections::BTreeSet<reims_vgpu_protocol::TransactionId>,
) -> Vec<(reims_vgpu_protocol::TransactionId, String)> {
    order
        .iter()
        .filter(|entry| {
            (entry.recorded || entry.issued)
                && !entry.submitted
                && !entry.abandoned
                && !owned.contains(&entry.transaction)
        })
        .map(|entry| {
            (
                entry.transaction,
                format!(
                    "{}@{}.{}",
                    entry.transaction.get(),
                    entry.domain.get(),
                    entry.sequence.get()
                ),
            )
        })
        .collect()
}

fn missing_execution_representation<Semantic>(
    failure: &crate::runtime::replacement_session::ReplacementExecIngressDispatchFailure<Semantic>,
) -> Option<reims_vgpu_protocol::BackingId> {
    use crate::runtime::replacement_session::{
        ReplacementExecIngressDispatchFailure as Dispatch,
        ReplacementExecResourceReadinessError as Readiness,
        ReplacementResourceReadyExecFailure as Resources,
    };
    match failure {
        Dispatch::DirectResources(Resources::Readiness {
            reason:
                Readiness::ValidityRepresentation {
                    backing,
                    reason: reims_vgpu_core::ManagedBackingError::MissingExecutionRepresentation,
                },
            ..
        }) => Some(*backing),
        _ => None,
    }
}

/// Build the execution representation a backing is missing, whatever class of
/// storage it is.
///
/// This is the late repair: a dispatch that met a backing with no execution
/// representation names it, and this builds it and the dispatch is retried. The
/// object-ready route ahead of it materializes from the *resources* a packet
/// declares, and it is the only route that knows a plane view from a linear
/// allocation. A backing that reaches here did not come through that route --
/// it was declared by an earlier packet, or replaced -- so the class has to be
/// recovered from the storage node.
///
/// Dispatching on the class is what keeps the two routes from disagreeing. A
/// repair that served only task-address storage refused every plane of a
/// registered surface by name, and because a missing execution representation
/// holds the whole recorded chain, that refusal parked its channel's submission
/// head rather than costing one command. The classes with no materializer here
/// still refuse -- but they refuse saying which class, which is the difference
/// between a gap someone can go and close and a backing that is simply
/// "unavailable".
fn prepare_backing_representation<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    backing: reims_vgpu_protocol::BackingId,
) -> Result<(), ReplacementObjectRepresentationPreparationError>
where
    Semantic: Clone,
{
    if let Some(plane_view) = runtime.io_surface_plane_view_owner(backing) {
        return materialize_io_surface_plane_view(runtime, host, page_shift, plane_view);
    }
    prepare_task_address_backing_representation(runtime, host, page_shift, backing)
}

/// Build the image one declared plane of a registered surface is.
///
/// A registered surface is an allocation, not an image: it may declare several
/// planes at their own offsets with their own extent, row pitch and pixel
/// format, and a backing carries one representation and one layout. So the
/// plane is what becomes a native image, over the plane's own backing.
fn materialize_io_surface_plane_view<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    plane_view: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
) -> Result<(), ReplacementObjectRepresentationPreparationError>
where
    Semantic: Clone,
{
    let (task, descriptor, backing) = runtime
        .io_surface_plane_view_materialization_facts(plane_view)
        .ok_or(
            ReplacementObjectRepresentationPreparationError::SurfaceDescriptorUnavailable(
                plane_view,
            ),
        )?;
    if runtime.backing_has_execution_representation(backing) {
        return Ok(());
    }
    let gva = u64::from(descriptor.backing_pfn)
        .checked_shl(page_shift)
        .ok_or(ReplacementObjectRepresentationPreparationError::PageAddressOverflow { page: 0 })?;
    let guest = resolve_task_guest_window(runtime, host, page_shift, task, gva, descriptor.length)?;
    runtime
        .materialize_io_surface_plane_view_with_guest_window(plane_view, guest)
        .map_err(ReplacementObjectRepresentationPreparationError::Construction)?;
    Ok(())
}

fn prepare_task_address_backing_representation<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    backing: reims_vgpu_protocol::BackingId,
) -> Result<(), ReplacementObjectRepresentationPreparationError>
where
    Semantic: Clone,
{
    let (resource, task, address, length) = runtime
        .task_address_backing_materialization_facts(backing)
        .map_err(
            |reason| ReplacementObjectRepresentationPreparationError::BackingUnavailable {
                backing,
                reason,
            },
        )?;
    let guest =
        resolve_task_guest_window(runtime, host, page_shift, task, address.get(), length.get())?;
    runtime
        .materialize_resource_with_guest_window(resource, guest)
        .map_err(ReplacementObjectRepresentationPreparationError::Construction)?;
    Ok(())
}

/// Name the backings whose execution representation a lifecycle effect revoked.
fn replaced_physical_backings<Native>(
    effect: &reims_vgpu_core::ResourceLifecycleEffect<Native>,
) -> Vec<reims_vgpu_protocol::BackingId> {
    match effect {
        reims_vgpu_core::ResourceLifecycleEffect::PhysicalReplaced { backing, .. } => {
            backing.iter().copied().collect()
        }
        reims_vgpu_core::ResourceLifecycleEffect::PhysicalBatchReplaced { native, .. } => {
            native.iter().map(|(backing, _)| *backing).collect()
        }
        _ => Vec::new(),
    }
}

/// Reinstall the execution representations a physical replacement revoked.
///
/// `ManagedBackingOwner::replace_execution_representation` retires the
/// construction-designated object and states that a subsequent materialization
/// installs a fresh representation identity. This is that materialization, and
/// it runs where the replacement lands rather than where some later preparation
/// trips over the absence. Every EXEC ingress route resolves a backing's
/// execution representation — direct, guest-upload suffix and indirect range
/// alike — so a repair attached to one route's refusal shape leaves the other
/// two meeting a revoked backing as a terminal refusal that still holds the
/// whole recorded chain.
///
/// A backing this cannot serve is reported by name rather than guessed at: only
/// task-address storage carries the guest address and length a fresh
/// materialization needs, and the other backing classes have their own
/// materializers.
fn prepare_replaced_physical_representations<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    backings: &[reims_vgpu_protocol::BackingId],
) where
    Semantic: Clone,
{
    for &backing in backings {
        if runtime.backing_has_execution_representation(backing) {
            crate::observe::off(format!(
                "replacement_replaced_physical_representation backing={backing:?} status=retained"
            ));
            continue;
        }
        match prepare_backing_representation(runtime, host, page_shift, backing) {
            Ok(()) => crate::observe::off(format!(
                "replacement_replaced_physical_representation backing={backing:?} \
                 status=materialized"
            )),
            Err(reason) => report_replaced_physical_representation_refusal(backing, &reason),
        }
    }
}

fn report_replaced_physical_representation_refusal(
    backing: reims_vgpu_protocol::BackingId,
    reason: &ReplacementObjectRepresentationPreparationError,
) {
    let reason = format!("{reason:?}");
    let diagnostic = ReplacementCoordinatorDiagnostic {
        slug: "replacement_replaced_physical_representation_refused",
        fields: vec![
            ("backing", format!("{backing:?}")),
            ("reason", reason.clone()),
        ],
        discriminant: fnv_discriminant(&reason),
    };
    crate::observe::Emit::decline("replacement_replaced_physical_representation", &diagnostic)
        .fail_once(diagnostic.discriminant);
}

fn prepare_object_ready_representations<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    ready: &crate::runtime::replacement_session::ReplacementObjectReadyExecPacket,
) -> Result<(), ReplacementObjectRepresentationPreparationError>
where
    Semantic: Clone,
{
    use crate::runtime::replacement_session::{
        ReplacementAppliedLoadedObject, ReplacementLoadedObjectApplyEffect,
    };
    let mut plane_views = std::collections::BTreeSet::new();
    let mut views = std::collections::BTreeSet::new();
    let mut linears = std::collections::BTreeSet::new();
    for applied in ready.constructions.objects.iter() {
        let ReplacementAppliedLoadedObject::Descriptor(effect) = applied else {
            continue;
        };
        match effect {
            ReplacementLoadedObjectApplyEffect::Linear(declaration) => {
                linears.insert(declaration.resource);
            }
            ReplacementLoadedObjectApplyEffect::IOSurfacePlaneView(declaration) => {
                plane_views.insert(declaration.resource);
                views.insert(declaration.resource);
            }
            ReplacementLoadedObjectApplyEffect::TextureView(declaration) => {
                views.insert(declaration.resource);
            }
            _ => {}
        }
    }
    for descriptor in ready.resources() {
        let Some(resource) = runtime.resolve_resource(
            ready.task(),
            reims_vgpu_protocol::ObjectTableRef::new(descriptor.object_id),
        ) else {
            continue;
        };
        if runtime
            .linear_resource_materialization_facts(resource)
            .is_some()
        {
            linears.insert(resource);
        } else if runtime
            .io_surface_plane_view_materialization_facts(resource)
            .is_some()
        {
            plane_views.insert(resource);
        }
    }
    // A registered surface is an allocation, not an image. Each declared plane
    // view is materialized over its own backing, so a multi-plane surface
    // reaches the executor as several textures rather than as one allocation
    // with no single layout.
    for plane_view in plane_views {
        materialize_io_surface_plane_view(runtime, host, page_shift, plane_view)?;
    }
    for resource in linears {
        let (task, address, length) = runtime
            .linear_resource_materialization_facts(resource)
            .ok_or(
                ReplacementObjectRepresentationPreparationError::SurfaceDescriptorUnavailable(
                    resource,
                ),
            )?;
        let backing = runtime
            .execution()
            .resources()
            .graph()
            .resource(resource)
            .and_then(|node| node.storage)
            .ok_or(
                ReplacementObjectRepresentationPreparationError::SurfaceDescriptorUnavailable(
                    resource,
                ),
            )?;
        if runtime.backing_has_execution_representation(backing) {
            continue;
        }
        let guest = resolve_task_guest_window(
            runtime,
            host,
            page_shift,
            task,
            address.get(),
            length.get(),
        )?;
        runtime
            .materialize_resource_with_guest_window(resource, guest)
            .map_err(ReplacementObjectRepresentationPreparationError::Construction)?;
    }
    // A view declared after its base image was built is not in the set that
    // materialization handed to the image, so it has to be installed on the
    // image that already exists. A view declared before it needs nothing here:
    // the materialization ahead reads every view over the backing and carries
    // it. Which of the two happened is the backing's own state, so this asks
    // the backing rather than tracking the order.
    for view in views {
        let Some(backing) = runtime.resolved_backing(view) else {
            continue;
        };
        if !runtime.backing_has_execution_representation(backing) {
            continue;
        }
        if let Err(reason) = runtime.materialize_texture_view(view) {
            // Reported and not fatal. The image is built and every other view
            // over it is usable; only a bind naming *this* view is lost, and
            // that bind refuses by name at record time on the same image.
            let reason = format!("{reason:?}");
            let diagnostic = ReplacementCoordinatorDiagnostic {
                slug: "replacement_texture_view_install_refused",
                fields: vec![
                    ("resource", format!("{view:?}")),
                    ("backing", format!("{backing:?}")),
                    ("reason", reason.clone()),
                ],
                discriminant: fnv_discriminant(&reason),
            };
            crate::observe::Emit::decline("replacement_texture_view_install", &diagnostic)
                .fail_once(diagnostic.discriminant);
        }
    }
    Ok(())
}

fn resolve_task_guest_window<Semantic>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    task: reims_vgpu_protocol::TaskId,
    gva: u64,
    length: u64,
) -> Result<reims_vgpu_memory::GuestWindow, ReplacementObjectRepresentationPreparationError>
where
    Semantic: Clone,
{
    let page_size = 1_u64
        .checked_shl(page_shift)
        .ok_or(ReplacementObjectRepresentationPreparationError::UnsupportedPageShift(page_shift))?;
    let in_page = gva % page_size;
    let covered = in_page
        .checked_add(length)
        .ok_or(ReplacementObjectRepresentationPreparationError::PageCountOverflow(length))?;
    let page_count = covered
        .checked_add(page_size - 1)
        .ok_or(ReplacementObjectRepresentationPreparationError::PageCountOverflow(length))?
        / page_size;
    let page_capacity = usize::try_from(page_count)
        .map_err(|_| ReplacementObjectRepresentationPreparationError::PageCountOverflow(length))?;
    let task_entry = runtime
        .tasks()
        .get(task.get())
        .ok_or(ReplacementObjectRepresentationPreparationError::TaskUnavailable(task))?;
    let page_base = gva - in_page;
    let mut gpas = Vec::with_capacity(page_capacity);
    for page in 0..page_count {
        let gva_page = page
            .checked_mul(page_size)
            .and_then(|offset| page_base.checked_add(offset))
            .ok_or(ReplacementObjectRepresentationPreparationError::PageAddressOverflow { page })?;
        let gpa =
            crate::runtime::gva_mem::translate_task_gva(host, task_entry, gva_page, page_shift)
                .ok_or(
                    ReplacementObjectRepresentationPreparationError::PageUnmapped { task, page },
                )?;
        gpas.push(gpa - (gpa % page_size));
    }
    let runs =
        crate::runtime::guest_ram_map::references_for_runs(host, &gpas, page_size, in_page, length)
            .map_err(ReplacementObjectRepresentationPreparationError::GuestMap)?;
    reims_vgpu_memory::GuestWindow::new(runs)
        .map_err(ReplacementObjectRepresentationPreparationError::GuestWindow)
}

fn replacement_object_preparation_failure_diagnostic(
    failure: &crate::runtime::replacement_session::ReplacementLoadedExecObjectPreparationFailure,
) -> String {
    use crate::runtime::replacement_session::{
        ReplacementLoadedExecObjectPreparationRefusal as Preparation,
        ReplacementObjectConstructionApplyRefusal as Apply,
        ReplacementObjectConstructionStagingRefusal as Staging,
    };

    match &failure.reason {
        Preparation::Decode(reason) => {
            format!("stage=object_preparation.decode reason={reason:?}")
        }
        Preparation::Load(reason) => {
            format!("stage=object_preparation.load reason={reason:?}")
        }
        Preparation::Staging(failure) => {
            let reason = match &failure.failed {
                Staging::Function(failure) => format!(
                    "function object={} reason={:?}",
                    failure.descriptor.object.get(),
                    failure.reason
                ),
                Staging::HeapTexture { descriptor, reason } => {
                    format!(
                        "heap_texture object={} reason={reason:?}",
                        descriptor.object.get()
                    )
                }
            };
            format!(
                "stage=object_preparation.staging task={} ready={} remaining={} {reason}",
                failure.task.get(),
                failure.ready.len(),
                failure.remaining.len()
            )
        }
        Preparation::Apply(failure) => {
            let reason = match &failure.failed {
                Apply::Descriptor(failed) => format!(
                    "descriptor {}",
                    replacement_loaded_object_apply_refusal_diagnostic(
                        failed.loaded.object,
                        &failed.reason,
                    )
                ),
                Apply::Function(failed) => format!(
                    "function object={} reason={:?}",
                    failed.loaded.descriptor.object.get(),
                    failed.reason
                ),
                Apply::HeapTexture(failed) => format!(
                    "heap_texture object={} reason={:?}",
                    failed.loaded.descriptor.object.get(),
                    failed.reason
                ),
            };
            format!(
                "stage=object_preparation.apply task={} applied={} remaining={} {reason}",
                failure.task.get(),
                failure.applied.len(),
                failure.remaining.len()
            )
        }
    }
}

fn replacement_loaded_object_apply_refusal_diagnostic(
    object: reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>,
    reason: &crate::runtime::replacement_session::ReplacementLoadedObjectApplyRefusal,
) -> String {
    format!("object={} reason={reason:?}", object.get())
}

fn replacement_ingress_preparation_diagnostic(
    failure: &crate::runtime::replacement_session::ReplacementExecIngressPreparationError<()>,
) -> String {
    use crate::runtime::replacement_session::{
        ReplacementCanonicalExecDecodeError as Canonical,
        ReplacementDecodedExecAdmissionError as Admission,
        ReplacementExecIngressPreparationError as Ingress,
    };
    match failure {
        Ingress::Admission(Admission::Canonical(Canonical::Projection(failure))) => {
            format!(
                "stage=ingress.canonical_projection reason={:?}",
                failure.reason
            )
        }
        Ingress::Admission(Admission::Canonical(Canonical::Decode(reason))) => {
            format!("stage=ingress.canonical_decode reason={reason:?}")
        }
        Ingress::Admission(Admission::ResourceTable(reason)) => {
            format!("stage=ingress.resource_table reason={reason:?}")
        }
        Ingress::Admission(Admission::Accesses(failure)) => format!(
            "stage=ingress.accesses position={} reason={:?}",
            failure.position, failure.reason
        ),
        Ingress::Admission(Admission::Admission(reason)) => {
            format!("stage=ingress.admission reason={reason:?}")
        }
        Ingress::Direct(failure) => {
            format!("stage=ingress.direct reason={:?}", failure.reason)
        }
        Ingress::IndirectRange(failure) => {
            format!("stage=ingress.indirect_range reason={:?}", failure.reason)
        }
    }
}

fn replacement_automatic_preparation_diagnostic(
    reason: &crate::runtime::replacement_session::ReplacementExecAutomaticPreparationError,
) -> String {
    use crate::runtime::replacement_session::ReplacementExecAutomaticPreparationError as Error;
    match reason {
        Error::OriginCountMismatch {
            operations,
            origins,
        } => {
            format!("reason=origin_count_mismatch operations={operations} origins={origins}")
        }
        Error::ManifestTransactionMismatch { expected, actual } => {
            format!("reason=manifest_transaction_mismatch expected={expected:?} actual={actual:?}")
        }
        Error::ManifestSubmissionMismatch { expected, actual } => {
            format!("reason=manifest_submission_mismatch expected={expected:?} actual={actual:?}")
        }
        Error::ComputeLeaseAbsent(position) => {
            format!("reason=compute_lease_absent position={position}")
        }
        Error::RenderLeaseAbsent(position) => {
            format!("reason=render_lease_absent position={position}")
        }
        Error::InfoEvaluationAbsent(position) => {
            format!("reason=info_evaluation_absent position={position}")
        }
        Error::BufferBlit {
            position, failure, ..
        } => {
            format!("reason=buffer_blit position={position} detail={failure:?}")
        }
        Error::ImageBlit {
            position, failure, ..
        } => {
            format!("reason=image_blit position={position} detail={failure:?}")
        }
        Error::Compute {
            position, failure, ..
        } => match failure.as_ref() {
            reims_vgpu_core::ExecResourcePreparationStepFailure::TransactionMismatch {
                expected,
                actual,
                ..
            } => format!(
                "reason=compute_transaction_mismatch position={position} expected={expected:?} actual={actual:?}"
            ),
            reims_vgpu_core::ExecResourcePreparationStepFailure::PositionOccupied {
                position: occupied,
                ..
            } => format!(
                "reason=compute_position_occupied position={position} occupied={occupied}"
            ),
            reims_vgpu_core::ExecResourcePreparationStepFailure::Preparation((reason, _)) => {
                format!("reason=compute position={position} detail={reason:?}")
            }
        },
        Error::Render {
            position, failure, ..
        } => match failure.as_ref() {
            reims_vgpu_core::ExecResourcePreparationStepFailure::TransactionMismatch {
                expected,
                actual,
                ..
            } => format!(
                "reason=render_transaction_mismatch position={position} expected={expected:?} actual={actual:?}"
            ),
            reims_vgpu_core::ExecResourcePreparationStepFailure::PositionOccupied {
                position: occupied,
                ..
            } => format!(
                "reason=render_position_occupied position={position} occupied={occupied}"
            ),
            reims_vgpu_core::ExecResourcePreparationStepFailure::Preparation((reason, _)) => {
                format!("reason=render position={position} detail={reason:?}")
            }
        },
        Error::Info {
            position, failure, ..
        } => {
            format!("reason=info position={position} detail={failure:?}")
        }
        Error::IndirectRange {
            position, failure, ..
        } => {
            format!("reason=indirect_range position={position} detail={failure:?}")
        }
        Error::ResourceStates { failure, .. } => {
            format!("reason=resource_states detail={failure:?}")
        }
        Error::ContentSynchronization { reason, .. } => {
            format!("reason=content_synchronization detail={reason:?}")
        }
        Error::HostIngress { reason, .. } => {
            format!("reason=host_ingress detail={reason:?}")
        }
        Error::Assembly { reason, .. } => format!("reason=assembly detail={reason:?}"),
    }
}

/// Join a deferred EXEC to its declared task address space and dispatch it.
///
/// Each failure owns the latest durable phase: an unread packet, a loaded
/// packet inside object preparation, or an object-ready packet beside the
/// admitted/recording refusal. No caller has to decode the FIFO envelope or
/// repeat already-applied object construction to retry.
pub(crate) fn dispatch_host_exec_packet<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    packet: crate::runtime::replacement_child_packet::ReplacementDeferredExecPacket,
) -> Result<
    crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>,
    Box<ReplacementHostExecDispatchFailure<Semantic>>,
>
where
    Semantic: Clone + PartialEq + Send + 'static,
{
    let tasks = runtime.tasks().clone();
    let loaded = match runtime.load_exec_packet(&packet, |task, gva, length| {
        read_replacement_task_bytes_from_table(&tasks, host, page_shift, task, gva, length)
    }) {
        Ok(loaded) => loaded,
        Err(reason) => {
            return Err(Box::new(ReplacementHostExecDispatchFailure::Load {
                reason,
                packet,
            }));
        }
    };
    let ready = runtime
        .prepare_loaded_exec_packet_objects(page_shift, loaded, |task, gva, length| {
            read_replacement_task_bytes_from_table(&tasks, host, page_shift, task, gva, length)
        })
        .map_err(ReplacementHostExecDispatchFailure::ObjectPreparation)
        .map_err(Box::new)?;
    if let Err(reason) = prepare_object_ready_representations(runtime, host, page_shift, &ready) {
        return Err(Box::new(
            ReplacementHostExecDispatchFailure::Representation {
                reason,
                ready: Box::new(ready),
            },
        ));
    }
    let dispatched = dispatch_object_ready_host_exec(runtime, host, page_shift, &tasks, &ready);
    match dispatched {
        Ok(pending) => Ok(pending),
        Err(reason) => Err(Box::new(ReplacementHostExecDispatchFailure::Dispatch {
            reason,
            ready: Box::new(ready),
        })),
    }
}

fn dispatch_object_ready_host_exec<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &impl HostMemory,
    page_shift: u32,
    tasks: &reims_vgpu_core::TaskTable,
    ready: &crate::runtime::replacement_session::ReplacementObjectReadyExecPacket,
) -> Result<
    crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>,
    crate::runtime::replacement_session::ReplacementExecIngressDispatchFailure<Semantic>,
>
where
    Semantic: Clone + PartialEq + Send + 'static,
{
    runtime.dispatch_loaded_exec_packet_with_icb_reader(ready, &mut |task, plan| {
        let length = usize::try_from(plan.byte_len).map_err(|_| {
            crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError::HostLengthOverflow(
                plan.byte_len,
            )
        })?;
        read_replacement_task_bytes_from_table(
            tasks,
            host,
            page_shift,
            task,
            plan.gva,
            length,
        )
    })
}

/// Resume only host-side EXEC phases whose retained owner proves replay safe.
pub(crate) fn retry_host_exec_dispatch_failure<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    failure: ReplacementHostExecDispatchFailure<Semantic>,
) -> Result<
    crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>,
    Box<ReplacementHostExecDispatchFailure<Semantic>>,
>
where
    Semantic: Clone + PartialEq + Send + 'static,
{
    match failure {
        ReplacementHostExecDispatchFailure::Load { packet, .. } => {
            dispatch_host_exec_packet(runtime, host, page_shift, packet)
        }
        ReplacementHostExecDispatchFailure::ObjectPreparation(failure) => {
            let tasks = runtime.tasks().clone();
            let ready = runtime
                .retry_loaded_exec_packet_object_preparation(
                    page_shift,
                    *failure,
                    |task, gva, length| {
                        read_replacement_task_bytes_from_table(
                            &tasks, host, page_shift, task, gva, length,
                        )
                    },
                )
                .map_err(ReplacementHostExecDispatchFailure::ObjectPreparation)
                .map_err(Box::new)?;
            dispatch_object_ready_host_exec(runtime, host, page_shift, &tasks, &ready).map_err(
                |reason| {
                    Box::new(ReplacementHostExecDispatchFailure::Dispatch {
                        reason,
                        ready: Box::new(ready),
                    })
                },
            )
        }
        ReplacementHostExecDispatchFailure::Representation { ready, .. } => {
            if let Err(reason) =
                prepare_object_ready_representations(runtime, host, page_shift, &ready)
            {
                return Err(Box::new(
                    ReplacementHostExecDispatchFailure::Representation { reason, ready },
                ));
            }
            let tasks = runtime.tasks().clone();
            dispatch_object_ready_host_exec(runtime, host, page_shift, &tasks, &ready).map_err(
                |reason| Box::new(ReplacementHostExecDispatchFailure::Dispatch { reason, ready }),
            )
        }
        ReplacementHostExecDispatchFailure::Dispatch { reason, ready } => {
            let projection_retry = matches!(
                &reason,
                crate::runtime::replacement_session::ReplacementExecIngressDispatchFailure::Ingress(
                    crate::runtime::replacement_session::ReplacementExecIngressPreparationError::Admission(
                        crate::runtime::replacement_session::ReplacementDecodedExecAdmissionError::Canonical(
                            crate::runtime::replacement_session::ReplacementCanonicalExecDecodeError::Projection(_)
                        )
                    )
                )
            );
            if let Some(backing) = missing_execution_representation(&reason) {
                if let Err(preparation) =
                    prepare_backing_representation(runtime, host, page_shift, backing)
                {
                    return Err(Box::new(
                        ReplacementHostExecDispatchFailure::BackingRepresentation {
                            backing,
                            reason: preparation,
                            dispatch: reason,
                            ready,
                        },
                    ));
                }
                runtime
                    .retry_exec_ingress_dispatch(reason)
                    .map_err(|reason| {
                        Box::new(ReplacementHostExecDispatchFailure::Dispatch { reason, ready })
                    })
            } else if projection_retry {
                let tasks = runtime.tasks().clone();
                dispatch_object_ready_host_exec(runtime, host, page_shift, &tasks, &ready).map_err(
                    |reason| {
                        Box::new(ReplacementHostExecDispatchFailure::Dispatch { reason, ready })
                    },
                )
            } else {
                runtime
                    .retry_exec_ingress_dispatch(reason)
                    .map_err(|reason| {
                        Box::new(ReplacementHostExecDispatchFailure::Dispatch { reason, ready })
                    })
            }
        }
        ReplacementHostExecDispatchFailure::BackingRepresentation {
            backing,
            dispatch,
            ready,
            ..
        } => {
            if let Err(reason) = prepare_backing_representation(runtime, host, page_shift, backing)
            {
                return Err(Box::new(
                    ReplacementHostExecDispatchFailure::BackingRepresentation {
                        backing,
                        reason,
                        dispatch,
                        ready,
                    },
                ));
            }
            runtime
                .retry_exec_ingress_dispatch(dispatch)
                .map_err(|reason| {
                    Box::new(ReplacementHostExecDispatchFailure::Dispatch { reason, ready })
                })
        }
    }
}

#[derive(Debug)]
pub(crate) enum ReplacementHostCursorGlyphFailure {
    Load {
        reason: crate::runtime::replacement_child_packet::ReplacementCursorGlyphLoadError<
            crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError,
        >,
        packet: crate::runtime::replacement_child_packet::ReplacementDeferredCursorGlyph,
    },
    Admission(
        Box<crate::runtime::replacement_child_packet::ReplacementCursorGlyphAdmissionFailure>,
    ),
}

/// Resolve one deferred cursor glyph through the replacement task namespace.
/// A transport refusal returns the original deferred packet; an admission
/// refusal returns the already-loaded glyph and never repeats the guest read.
pub(crate) fn load_and_admit_host_cursor_glyph<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &impl HostMemory,
    page_shift: u32,
    packet: crate::runtime::replacement_child_packet::ReplacementDeferredCursorGlyph,
) -> Result<ReplacementAdmittedCpuPacket<Semantic>, Box<ReplacementHostCursorGlyphFailure>> {
    let loaded = match crate::runtime::replacement_child_packet::load_replacement_cursor_glyph(
        runtime,
        &packet,
        |task, gva, length| {
            read_replacement_task_bytes(runtime, host, page_shift, task, gva.get(), length)
        },
    ) {
        Ok(loaded) => loaded,
        Err(reason) => {
            return Err(Box::new(ReplacementHostCursorGlyphFailure::Load {
                reason,
                packet,
            }));
        }
    };
    crate::runtime::replacement_child_packet::admit_loaded_replacement_cursor_glyph(runtime, loaded)
        .map(ReplacementAdmittedCpuPacket::Control)
        .map_err(ReplacementHostCursorGlyphFailure::Admission)
        .map_err(Box::new)
}

pub(crate) fn retry_host_cursor_glyph_failure<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &impl HostMemory,
    page_shift: u32,
    failure: ReplacementHostCursorGlyphFailure,
) -> Result<ReplacementAdmittedCpuPacket<Semantic>, Box<ReplacementHostCursorGlyphFailure>> {
    match failure {
        ReplacementHostCursorGlyphFailure::Load { packet, .. } => {
            load_and_admit_host_cursor_glyph(runtime, host, page_shift, packet)
        }
        ReplacementHostCursorGlyphFailure::Admission(failure) => {
            crate::runtime::replacement_child_packet::admit_loaded_replacement_cursor_glyph(
                runtime,
                failure.loaded,
            )
            .map(ReplacementAdmittedCpuPacket::Control)
            .map_err(ReplacementHostCursorGlyphFailure::Admission)
            .map_err(Box::new)
        }
    }
}

#[derive(Debug)]
pub(crate) enum ReplacementRootPacketLeaseFailure<Semantic> {
    Validation {
        reason: crate::runtime::replacement_transport::ReplacementRootPacketCommitError,
        lease: crate::runtime::replacement_transport::ReplacementRootPacketLease,
    },
    Admission {
        reason: crate::runtime::replacement_fifo_control::ReplacementRootPacketIngressError,
        lease: crate::runtime::replacement_transport::ReplacementRootPacketLease,
    },
    Commit {
        failure: Box<crate::runtime::replacement_transport::ReplacementRootPacketCommitFailure>,
        admitted: crate::runtime::replacement_fifo_control::ReplacementAdmittedRootPacket<Semantic>,
    },
}

/// Admit exactly the packet named by a root-ring lease and advance the guest
/// consumer pointer only after semantic admission succeeds.
pub(crate) fn admit_root_packet_lease<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    lease: crate::runtime::replacement_transport::ReplacementRootPacketLease,
) -> Result<ReplacementAdmittedCpuPacket<Semantic>, Box<ReplacementRootPacketLeaseFailure<Semantic>>>
{
    if let Err(reason) = transport.validate_root_packet_commit(&lease) {
        return Err(Box::new(ReplacementRootPacketLeaseFailure::Validation {
            reason,
            lease,
        }));
    }
    let admitted = match crate::runtime::replacement_fifo_control::admit_replacement_root_packet(
        runtime,
        lease.packet.clone(),
    ) {
        Ok(admitted) => admitted,
        Err(reason) => {
            return Err(Box::new(ReplacementRootPacketLeaseFailure::Admission {
                reason,
                lease,
            }));
        }
    };
    if let Err(failure) = transport.commit_root_packet(lease) {
        return Err(Box::new(ReplacementRootPacketLeaseFailure::Commit {
            failure,
            admitted,
        }));
    }
    Ok(ReplacementAdmittedCpuPacket::from(admitted))
}

pub(crate) fn retry_root_packet_lease_failure<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    failure: ReplacementRootPacketLeaseFailure<Semantic>,
) -> Result<ReplacementAdmittedCpuPacket<Semantic>, Box<ReplacementRootPacketLeaseFailure<Semantic>>>
{
    match failure {
        ReplacementRootPacketLeaseFailure::Validation { lease, .. }
        | ReplacementRootPacketLeaseFailure::Admission { lease, .. } => {
            admit_root_packet_lease(runtime, transport, lease)
        }
        ReplacementRootPacketLeaseFailure::Commit { failure, admitted } => {
            match transport.commit_root_packet(failure.lease) {
                Ok(()) => Ok(ReplacementAdmittedCpuPacket::from(admitted)),
                Err(failure) => Err(Box::new(ReplacementRootPacketLeaseFailure::Commit {
                    failure,
                    admitted,
                })),
            }
        }
    }
}

pub(crate) enum ReplacementChildPacketLeaseIngress<Semantic> {
    Admitted(crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket<Semantic>),
    Deferred {
        transport: crate::runtime::replacement_child_packet::ReplacementChildPacketTransport,
        lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    },
}

#[derive(Debug)]
pub(crate) enum ReplacementChildPacketLeaseFailure<Semantic> {
    Ingress {
        reason: crate::runtime::replacement_child_packet::ReplacementChildCpuPacketIngressError,
        lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    },
    Commit {
        failure: Box<crate::runtime::replacement_transport::ReplacementChildPacketCommitFailure>,
        admitted:
            crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket<Semantic>,
    },
}

impl ReplacementChildPacketLeaseFailure<()> {
    /// Classify an ingress failure as a declared refusal, or hand it back to be
    /// re-offered.
    ///
    /// See
    /// [`crate::runtime::replacement_child_packet::ReplacementChildCpuPacketIngressError::is_terminal_refusal`].
    fn into_refusal(self: Box<Self>) -> Result<ReplacementRefusedChildPacket, Box<Self>> {
        match *self {
            Self::Ingress { reason, lease } if reason.is_terminal_refusal() => {
                Ok(ReplacementRefusedChildPacket {
                    lease,
                    detail: format!("stage=child_ingress reason={reason:?}"),
                })
            }
            failure => Err(Box::new(failure)),
        }
    }
}

pub(crate) fn commit_child_packet_after_admission<T>(
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut impl HostMemory,
    lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    admitted: T,
) -> Result<
    T,
    (
        Box<crate::runtime::replacement_transport::ReplacementChildPacketCommitFailure>,
        T,
    ),
> {
    match transport.commit_child_packet(host, lease) {
        Ok(()) => Ok(admitted),
        Err(failure) => Err((failure, admitted)),
    }
}

/// Route one child-ring lease. Immediate CPU/present admission consumes the
/// lease and advances the head; deferred data-plane routes retain the lease
/// until their later admission succeeds.
pub(crate) fn admit_child_packet_lease<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut impl HostMemory,
    lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
) -> Result<
    ReplacementChildPacketLeaseIngress<Semantic>,
    Box<ReplacementChildPacketLeaseFailure<Semantic>>,
> {
    let admitted = crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
        runtime,
        lease.channel,
        lease.packet.clone(),
    );
    match admitted {
        Ok(admitted) => commit_child_packet_after_admission(transport, host, lease, admitted)
            .map(ReplacementChildPacketLeaseIngress::Admitted)
            .map_err(|(failure, admitted)| {
                Box::new(ReplacementChildPacketLeaseFailure::Commit {
                    failure,
                    admitted,
                })
            }),
        Err(
            crate::runtime::replacement_child_packet::ReplacementChildCpuPacketIngressError::RequiresTransport(
                deferred,
            ),
        ) => Ok(ReplacementChildPacketLeaseIngress::Deferred {
            transport: deferred,
            lease,
        }),
        Err(reason) => Err(Box::new(ReplacementChildPacketLeaseFailure::Ingress {
            reason,
            lease,
        })),
    }
}

pub(crate) fn retry_child_packet_lease_failure<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut impl HostMemory,
    failure: ReplacementChildPacketLeaseFailure<Semantic>,
) -> Result<
    (
        ReplacementChildPacketLeaseIngress<Semantic>,
        reims_vgpu_protocol::ChannelId,
    ),
    Box<ReplacementChildPacketLeaseFailure<Semantic>>,
> {
    match failure {
        ReplacementChildPacketLeaseFailure::Ingress { lease, .. } => {
            let channel = lease.channel;
            admit_child_packet_lease(runtime, transport, host, lease)
                .map(|ingress| (ingress, channel))
        }
        ReplacementChildPacketLeaseFailure::Commit { failure, admitted } => {
            let channel = failure.lease.channel;
            commit_child_packet_after_admission(transport, host, failure.lease, admitted)
                .map(|admitted| {
                    (
                        ReplacementChildPacketLeaseIngress::Admitted(admitted),
                        channel,
                    )
                })
                .map_err(|(failure, admitted)| {
                    Box::new(ReplacementChildPacketLeaseFailure::Commit { failure, admitted })
                })
        }
    }
}

pub(crate) enum ReplacementDeferredChildAdmission<Semantic> {
    Cpu(ReplacementAdmittedCpuPacket<Semantic>),
    Exec(crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>),
}

pub(crate) enum ReplacementDeferredChildDispatchFailure<Semantic> {
    Exec {
        failure: Box<ReplacementHostExecDispatchFailure<Semantic>>,
        lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    },
    Cursor {
        failure: Box<ReplacementHostCursorGlyphFailure>,
        lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    },
    Synchronize {
        failure: Box<
            crate::runtime::replacement_child_packet::ReplacementDeferredSynchronizeDispatchFailure<
                Semantic,
            >,
        >,
        lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    },
    Blocked {
        transport: crate::runtime::replacement_child_packet::ReplacementChildPacketTransport,
        lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
    },
    Commit {
        failure: Box<crate::runtime::replacement_transport::ReplacementChildPacketCommitFailure>,
        admitted: ReplacementDeferredChildAdmission<Semantic>,
    },
}

/// Finish a deferred child packet at its furthest transport phase and consume
/// its ring lease only after a replacement transaction owns the packet.
pub(crate) fn dispatch_deferred_child_packet<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport_owner: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    transport: crate::runtime::replacement_child_packet::ReplacementChildPacketTransport,
    lease: crate::runtime::replacement_transport::ReplacementChildPacketLease,
) -> Result<
    ReplacementDeferredChildAdmission<Semantic>,
    Box<ReplacementDeferredChildDispatchFailure<Semantic>>,
>
where
    Semantic: Clone + PartialEq + Send + 'static,
{
    let admitted = match transport {
        crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::Exec(packet) => {
            match dispatch_host_exec_packet(runtime, host, transport_owner.page_shift(), packet) {
                Ok(pending) => ReplacementDeferredChildAdmission::Exec(pending),
                Err(failure) => {
                    return Err(Box::new(ReplacementDeferredChildDispatchFailure::Exec {
                        failure,
                        lease,
                    }));
                }
            }
        }
        crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::CursorGlyph(
            packet,
        ) => match load_and_admit_host_cursor_glyph(
            runtime,
            host,
            transport_owner.page_shift(),
            packet,
        ) {
            Ok(admitted) => ReplacementDeferredChildAdmission::Cpu(admitted),
            Err(failure) => {
                return Err(Box::new(ReplacementDeferredChildDispatchFailure::Cursor {
                    failure,
                    lease,
                }));
            }
        },
        crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::Synchronize(
            deferred,
        ) => match crate::runtime::replacement_child_packet::dispatch_deferred_replacement_synchronize(
            runtime,
            deferred,
        ) {
            Ok(pending) => ReplacementDeferredChildAdmission::Exec(pending),
            Err(failure) => {
                return Err(Box::new(
                    ReplacementDeferredChildDispatchFailure::Synchronize { failure, lease },
                ));
            }
        },
        blocked @ crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::Blocked { .. } => {
            return Err(Box::new(ReplacementDeferredChildDispatchFailure::Blocked {
                transport: blocked,
                lease,
            }));
        }
    };
    commit_child_packet_after_admission(transport_owner, host, lease, admitted).map_err(
        |(failure, admitted)| {
            Box::new(ReplacementDeferredChildDispatchFailure::Commit { failure, admitted })
        },
    )
}

pub(crate) fn retry_deferred_child_dispatch_failure<Semantic>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport_owner: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    failure: ReplacementDeferredChildDispatchFailure<Semantic>,
) -> Result<
    (
        ReplacementDeferredChildAdmission<Semantic>,
        reims_vgpu_protocol::ChannelId,
    ),
    Box<ReplacementDeferredChildDispatchFailure<Semantic>>,
>
where
    Semantic: Clone + PartialEq + Send + 'static,
{
    let (admitted, lease) = match failure {
        ReplacementDeferredChildDispatchFailure::Exec { failure, lease } => {
            let admitted = match retry_host_exec_dispatch_failure(
                runtime,
                host,
                transport_owner.page_shift(),
                *failure,
            ) {
                Ok(admitted) => ReplacementDeferredChildAdmission::Exec(admitted),
                Err(failure) => {
                    return Err(Box::new(ReplacementDeferredChildDispatchFailure::Exec {
                        failure,
                        lease,
                    }));
                }
            };
            (admitted, lease)
        }
        ReplacementDeferredChildDispatchFailure::Cursor { failure, lease } => {
            let admitted = match retry_host_cursor_glyph_failure(
                runtime,
                host,
                transport_owner.page_shift(),
                *failure,
            ) {
                Ok(admitted) => ReplacementDeferredChildAdmission::Cpu(admitted),
                Err(failure) => {
                    return Err(Box::new(ReplacementDeferredChildDispatchFailure::Cursor {
                        failure,
                        lease,
                    }));
                }
            };
            (admitted, lease)
        }
        ReplacementDeferredChildDispatchFailure::Synchronize { failure, lease } => {
            let admitted = match *failure {
                crate::runtime::replacement_child_packet::ReplacementDeferredSynchronizeDispatchFailure::PreAdmission { deferred, .. } => {
                    crate::runtime::replacement_child_packet::dispatch_deferred_replacement_synchronize(
                        runtime,
                        deferred,
                    )
                }
                crate::runtime::replacement_child_packet::ReplacementDeferredSynchronizeDispatchFailure::Admitted(failure) => runtime
                    .retry_synchronize_dispatch(failure)
                    .map_err(crate::runtime::replacement_child_packet::ReplacementDeferredSynchronizeDispatchFailure::Admitted)
                    .map_err(Box::new),
            };
            match admitted {
                Ok(admitted) => (ReplacementDeferredChildAdmission::Exec(admitted), lease),
                Err(failure) => {
                    return Err(Box::new(
                        ReplacementDeferredChildDispatchFailure::Synchronize { failure, lease },
                    ));
                }
            }
        }
        unchanged @ ReplacementDeferredChildDispatchFailure::Blocked { .. } => {
            return Err(Box::new(unchanged));
        }
        ReplacementDeferredChildDispatchFailure::Commit { failure, admitted } => {
            let channel = failure.lease.channel;
            return commit_child_packet_after_admission(
                transport_owner,
                host,
                failure.lease,
                admitted,
            )
            .map(|admitted| (admitted, channel))
            .map_err(|(failure, admitted)| {
                Box::new(ReplacementDeferredChildDispatchFailure::Commit { failure, admitted })
            });
        }
    };
    let channel = lease.channel;
    commit_child_packet_after_admission(transport_owner, host, lease, admitted)
        .map(|admitted| (admitted, channel))
        .map_err(|(failure, admitted)| {
            Box::new(ReplacementDeferredChildDispatchFailure::Commit { failure, admitted })
        })
}

#[derive(Debug)]
pub(crate) enum ReplacementMapperEntryDispatchFailure {
    Reserve(crate::runtime::replacement_transport::ReplacementMapperEntryError),
    Load {
        reason: crate::runtime::replacement_session::ReplacementMapperRequestLoadError,
        lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
    },
    Validation {
        reason: crate::runtime::replacement_transport::ReplacementMapperEntryError,
        loaded: crate::runtime::replacement_session::ReplacementLoadedMapperRequest,
        lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
    },
    Apply {
        reason: crate::runtime::replacement_session::ReplacementLoadedMapperRequestApplyError,
        loaded: crate::runtime::replacement_session::ReplacementLoadedMapperRequest,
        lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
    },
    PostApplyValidation {
        reason: crate::runtime::replacement_transport::ReplacementMapperEntryError,
        effect: crate::runtime::replacement_session::ReplacementMapperRequestEffect,
        lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
    },
    Backing {
        reason: reims_vgpu_core::MapperSurfaceBackingError,
        effect: crate::runtime::replacement_session::ReplacementMapperRequestEffect,
        lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
    },
    Commit {
        reason: crate::runtime::replacement_transport::ReplacementMapperEntryError,
        effect: crate::runtime::replacement_session::ReplacementMapperRequestEffect,
        lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
    },
}

#[derive(Debug)]
pub(crate) enum ReplacementIosfcWriteDispatchFailure {
    CapturePublication {
        reason: reims_vgpu_core::MapperCapturePublicationError,
        effect: crate::runtime::replacement_transport::ReplacementIosfcWriteEffect,
    },
    Drain {
        reason: Box<ReplacementMapperEntryDispatchFailure>,
        completed: Box<[crate::runtime::replacement_session::ReplacementMapperRequestEffect]>,
    },
}

/// Capture the directed mapper handoff while the publishing vCPU is still
/// current, then consume every published record through the transport lease.
/// The write effect retains the exact ring identity observed by MMIO, so a
/// later ring-base reprogram cannot redirect the capture read.
pub(crate) fn dispatch_iosfc_write_effect<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    effect: crate::runtime::replacement_transport::ReplacementIosfcWriteEffect,
) -> Result<
    Box<[crate::runtime::replacement_session::ReplacementMapperRequestEffect]>,
    Box<ReplacementIosfcWriteDispatchFailure>,
> {
    let crate::runtime::replacement_transport::ReplacementIosfcWriteEffect::CaptureAndDrain {
        ring_base,
        producer,
    } = effect;
    if let Some(capture) =
        crate::runtime::replacement_mapper::capture_at_producer(host, ring_base, producer)
    {
        if let Err(reason) = runtime.publish_mapper_capture(capture) {
            return Err(Box::new(
                ReplacementIosfcWriteDispatchFailure::CapturePublication { reason, effect },
            ));
        }
    }

    let mut completed = Vec::new();
    loop {
        match dispatch_next_mapper_entry(runtime, transport, host) {
            Ok(Some(effect)) => completed.push(effect),
            Ok(None) => return Ok(completed.into_boxed_slice()),
            Err(reason) => {
                return Err(Box::new(ReplacementIosfcWriteDispatchFailure::Drain {
                    reason,
                    completed: completed.into_boxed_slice(),
                }));
            }
        }
    }
}

pub(crate) fn retry_iosfc_write_dispatch_failure<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    failure: ReplacementIosfcWriteDispatchFailure,
) -> Result<
    Box<[crate::runtime::replacement_session::ReplacementMapperRequestEffect]>,
    Box<ReplacementIosfcWriteDispatchFailure>,
> {
    match failure {
        ReplacementIosfcWriteDispatchFailure::CapturePublication { effect, .. } => {
            dispatch_iosfc_write_effect(runtime, transport, host, effect)
        }
        ReplacementIosfcWriteDispatchFailure::Drain { reason, completed } => {
            let mut completed = completed.into_vec();
            match retry_mapper_entry_dispatch_failure(runtime, transport, host, *reason) {
                Ok(Some(effect)) => completed.push(effect),
                Ok(None) => {}
                Err(reason) => {
                    return Err(Box::new(ReplacementIosfcWriteDispatchFailure::Drain {
                        reason,
                        completed: completed.into_boxed_slice(),
                    }));
                }
            }
            loop {
                match dispatch_next_mapper_entry(runtime, transport, host) {
                    Ok(Some(effect)) => completed.push(effect),
                    Ok(None) => return Ok(completed.into_boxed_slice()),
                    Err(reason) => {
                        return Err(Box::new(ReplacementIosfcWriteDispatchFailure::Drain {
                            reason,
                            completed: completed.into_boxed_slice(),
                        }));
                    }
                }
            }
        }
    }
}

/// Consume the next IOSurface mapper entry in consumer order. The guest
/// consumer advances only after the decoded request mutates the replacement
/// mapper service; catching the producer emits the mapper IRQ once.
pub(crate) fn dispatch_next_mapper_entry<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
) -> Result<
    Option<crate::runtime::replacement_session::ReplacementMapperRequestEffect>,
    Box<ReplacementMapperEntryDispatchFailure>,
> {
    let lease = match transport.reserve_mapper_entry() {
        Ok(Some(lease)) => lease,
        Ok(None) => return Ok(None),
        Err(reason) => {
            return Err(Box::new(ReplacementMapperEntryDispatchFailure::Reserve(
                reason,
            )));
        }
    };
    dispatch_reserved_mapper_entry(runtime, transport, host, lease)
}

fn dispatch_reserved_mapper_entry<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
) -> Result<
    Option<crate::runtime::replacement_session::ReplacementMapperRequestEffect>,
    Box<ReplacementMapperEntryDispatchFailure>,
> {
    let loaded =
        match runtime.load_mapper_request(lease.ring_base, lease.producer, |gpa, length| {
            let mut bytes = vec![0; length];
            host.read_gpa(gpa, &mut bytes).map_err(
                crate::runtime::replacement_session::ReplacementMapperRequestTransportError::Memory,
            )?;
            Ok(bytes)
        }) {
            Ok(loaded) => loaded,
            Err(reason) => {
                return Err(Box::new(ReplacementMapperEntryDispatchFailure::Load {
                    reason,
                    lease,
                }));
            }
        };
    if let Err(reason) = transport.validate_mapper_entry(lease) {
        return Err(Box::new(
            ReplacementMapperEntryDispatchFailure::Validation {
                reason,
                loaded,
                lease,
            },
        ));
    }
    let effect = match runtime.apply_loaded_mapper_request(loaded, lease.ring_base, lease.producer)
    {
        Ok(effect) => effect,
        Err(reason) => {
            return Err(Box::new(ReplacementMapperEntryDispatchFailure::Apply {
                reason,
                loaded,
                lease,
            }));
        }
    };
    finish_applied_mapper_entry(runtime, transport, host, effect, lease)
}

pub(crate) fn retry_mapper_entry_dispatch_failure<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    failure: ReplacementMapperEntryDispatchFailure,
) -> Result<
    Option<crate::runtime::replacement_session::ReplacementMapperRequestEffect>,
    Box<ReplacementMapperEntryDispatchFailure>,
> {
    match failure {
        ReplacementMapperEntryDispatchFailure::Reserve(_) => {
            dispatch_next_mapper_entry(runtime, transport, host)
        }
        ReplacementMapperEntryDispatchFailure::Load { lease, .. } => {
            dispatch_reserved_mapper_entry(runtime, transport, host, lease)
        }
        ReplacementMapperEntryDispatchFailure::Validation { loaded, lease, .. }
        | ReplacementMapperEntryDispatchFailure::Apply { loaded, lease, .. } => {
            if let Err(reason) = transport.validate_mapper_entry(lease) {
                return Err(Box::new(
                    ReplacementMapperEntryDispatchFailure::Validation {
                        reason,
                        loaded,
                        lease,
                    },
                ));
            }
            let effect = match runtime.apply_loaded_mapper_request(
                loaded,
                lease.ring_base,
                lease.producer,
            ) {
                Ok(effect) => effect,
                Err(reason) => {
                    return Err(Box::new(ReplacementMapperEntryDispatchFailure::Apply {
                        reason,
                        loaded,
                        lease,
                    }));
                }
            };
            finish_applied_mapper_entry(runtime, transport, host, effect, lease)
        }
        ReplacementMapperEntryDispatchFailure::PostApplyValidation { effect, lease, .. }
        | ReplacementMapperEntryDispatchFailure::Backing { effect, lease, .. }
        | ReplacementMapperEntryDispatchFailure::Commit { effect, lease, .. } => {
            finish_applied_mapper_entry(runtime, transport, host, effect, lease)
        }
    }
}

/// Resume an already-applied mapper entry without decoding or applying its
/// ring record again. This is the retry edge for backing transport failures.
pub(crate) fn finish_applied_mapper_entry<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    transport: &mut crate::runtime::replacement_transport::ReplacementTransportOwner,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    mut effect: crate::runtime::replacement_session::ReplacementMapperRequestEffect,
    lease: crate::runtime::replacement_transport::ReplacementMapperEntryLease,
) -> Result<
    Option<crate::runtime::replacement_session::ReplacementMapperRequestEffect>,
    Box<ReplacementMapperEntryDispatchFailure>,
> {
    if let Err(reason) = transport.validate_mapper_entry(lease) {
        return Err(Box::new(
            ReplacementMapperEntryDispatchFailure::PostApplyValidation {
                reason,
                effect,
                lease,
            },
        ));
    }
    if let crate::runtime::replacement_session::ReplacementMapperRequestEffect::Map {
        resolved_surface,
        capture: Some(capture),
        backing_resolved,
        replaced_backing,
        ..
    } = &mut effect
    {
        if !*backing_resolved {
            let memory = crate::runtime::replacement_mapper::MapperMemory::new(host);
            match runtime.resolve_and_publish_mapper_backing(
                &memory,
                transport.page_shift(),
                *resolved_surface,
                *capture,
            ) {
                Ok(prior) => {
                    *replaced_backing = prior;
                    *backing_resolved = true;
                }
                Err(reason) => {
                    return Err(Box::new(ReplacementMapperEntryDispatchFailure::Backing {
                        reason,
                        effect,
                        lease,
                    }));
                }
            }
        }
    }
    let caught_up = match transport.commit_mapper_entry(lease) {
        Ok(caught_up) => caught_up,
        Err(reason) => {
            return Err(Box::new(ReplacementMapperEntryDispatchFailure::Commit {
                reason,
                effect,
                lease,
            }));
        }
    };
    if caught_up {
        host.enqueue(HostAction::irq_iosfc());
        host.schedule_bh();
    }
    Ok(Some(effect))
}

#[derive(Debug)]
pub(crate) enum ReplacementAdmittedCpuPacket<Semantic> {
    Control(ReplacementAdmittedControl<Semantic>),
    Query(ReplacementAdmittedQuery<Semantic>),
    ResourceLifecycle(ReplacementAdmittedResourceLifecycle<Semantic>),
}

impl<Semantic> ReplacementAdmittedCpuPacket<Semantic> {
    pub const fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        match self {
            Self::Control(admitted) => admitted.transaction(),
            Self::Query(admitted) => admitted.transaction(),
            Self::ResourceLifecycle(admitted) => admitted.transaction(),
        }
    }
}

impl<Semantic> ReplacementAppliedCpuPacket<Semantic> {
    pub const fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        match self {
            Self::Control(applied) => applied.transaction(),
            Self::Query(applied) => applied.transaction(),
            Self::ResourceLifecycle(applied) => applied.transaction(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum ReplacementNonCpuChildPacket<Semantic> {
    Present(crate::runtime::replacement_session::ReplacementAdmittedPresent<Semantic>),
}

impl<Semantic>
    From<crate::runtime::replacement_fifo_control::ReplacementAdmittedRootPacket<Semantic>>
    for ReplacementAdmittedCpuPacket<Semantic>
{
    fn from(
        admitted: crate::runtime::replacement_fifo_control::ReplacementAdmittedRootPacket<Semantic>,
    ) -> Self {
        match admitted {
            crate::runtime::replacement_fifo_control::ReplacementAdmittedRootPacket::Control(
                control,
            ) => Self::Control(control),
            crate::runtime::replacement_fifo_control::ReplacementAdmittedRootPacket::Query(
                query,
            ) => Self::Query(query),
        }
    }
}

impl<Semantic>
    TryFrom<crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket<Semantic>>
    for ReplacementAdmittedCpuPacket<Semantic>
{
    type Error = ReplacementNonCpuChildPacket<Semantic>;

    fn try_from(
        admitted: crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket<
            Semantic,
        >,
    ) -> Result<Self, Self::Error> {
        match admitted {
            crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket::Control(
                control,
            ) => Ok(Self::Control(control)),
            crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket::Query(
                query,
            ) => Ok(Self::Query(query)),
            crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket::ResourceLifecycle(
                lifecycle,
            ) => Ok(Self::ResourceLifecycle(lifecycle)),
            crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket::Present(
                present,
            ) => Err(ReplacementNonCpuChildPacket::Present(present)),
        }
    }
}

#[derive(Debug)]
pub(crate) enum ReplacementAppliedCpuPacket<Semantic> {
    Control(ReplacementAppliedControl<Semantic>),
    Query(ReplacementAppliedQuery<Semantic>),
    ResourceLifecycle(ReplacementAppliedResourceLifecycle<Semantic>),
}

#[derive(Debug)]
pub(crate) enum ReplacementCpuApplyError<Semantic> {
    Control(Box<ReplacementControlApplyError<Semantic>>),
    Query(Box<ReplacementQueryApplyError<Semantic>>),
    ResourceLifecycle(Box<ReplacementResourceLifecycleApplyError<Semantic>>),
}

impl<Semantic> ReplacementCpuApplyError<Semantic> {
    /// The refusal reason when no later guest packet can make this apply
    /// succeed, so retrying it only holds the channel.
    ///
    /// See
    /// [`crate::runtime::replacement_session::ReplacementControlApplyReason::is_terminal_refusal`].
    fn terminal_refusal(
        &self,
    ) -> Option<&crate::runtime::replacement_session::ReplacementControlApplyReason> {
        match self {
            Self::Control(failure) => match failure.as_ref() {
                ReplacementControlApplyError::Apply { reason, .. }
                    if reason.is_terminal_refusal() =>
                {
                    Some(reason)
                }
                ReplacementControlApplyError::Apply { .. }
                | ReplacementControlApplyError::NotReady(_) => None,
            },
            Self::Query(_) | Self::ResourceLifecycle(_) => None,
        }
    }
}

pub(crate) fn apply_ready_cpu_packet<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    page_shift: u32,
    version: u32,
    admitted: ReplacementAdmittedCpuPacket<Semantic>,
) -> Result<ReplacementAppliedCpuPacket<Semantic>, ReplacementCpuApplyError<Semantic>> {
    match admitted {
        ReplacementAdmittedCpuPacket::Control(admitted) => runtime
            .apply_admitted_control(page_shift, admitted)
            .map(ReplacementAppliedCpuPacket::Control)
            .map_err(ReplacementCpuApplyError::Control),
        ReplacementAdmittedCpuPacket::Query(admitted) => runtime
            .apply_admitted_query(page_shift, version, admitted)
            .map(ReplacementAppliedCpuPacket::Query)
            .map_err(ReplacementCpuApplyError::Query),
        ReplacementAdmittedCpuPacket::ResourceLifecycle(admitted) => runtime
            .apply_admitted_resource_lifecycle(admitted)
            .map(ReplacementAppliedCpuPacket::ResourceLifecycle)
            .map_err(ReplacementCpuApplyError::ResourceLifecycle),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementCpuHostEffectError {
    UnknownTask(reims_vgpu_protocol::TaskId),
    Memory(MemError),
}

#[derive(Debug)]
pub(crate) struct ReplacementCpuHostEffectFailure<Semantic> {
    pub reason: ReplacementCpuHostEffectError,
    pub applied: ReplacementAppliedCpuPacket<Semantic>,
}

#[derive(Debug)]
pub(crate) struct ReplacementHostAppliedCpuPacket<Semantic>(ReplacementAppliedCpuPacket<Semantic>);

impl<Semantic> ReplacementHostAppliedCpuPacket<Semantic> {
    pub const fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        self.0.transaction()
    }
}

fn apply_query_host_effect<Semantic: Clone>(
    runtime: &ReplacementRuntimeSession<Semantic>,
    host: &mut impl HostMemory,
    page_shift: u32,
    query: &ReplacementAppliedQuery<Semantic>,
) -> Result<(), ReplacementCpuHostEffectError> {
    match &query.effect {
        ReplacementQueryEffect::DeviceInfo(None) => Ok(()),
        ReplacementQueryEffect::DeviceInfo(Some(reply)) => host
            .write_gpa(reply.gpa, &reply.bytes)
            .map_err(ReplacementCpuHostEffectError::Memory),
        ReplacementQueryEffect::ComputeInfo(reply) => {
            let task = runtime
                .tasks()
                .get(reply.task.get())
                .ok_or(ReplacementCpuHostEffectError::UnknownTask(reply.task))?;
            crate::runtime::gva_mem::write_task_gva_once(
                host,
                task,
                reply.gva.get(),
                &reply.bytes,
                page_shift,
            )
            .map_err(ReplacementCpuHostEffectError::Memory)
        }
        ReplacementQueryEffect::HeapTexture(reply) => {
            let task = runtime
                .tasks()
                .get(reply.task.get())
                .ok_or(ReplacementCpuHostEffectError::UnknownTask(reply.task))?;
            crate::runtime::gva_mem::write_task_gva_once(
                host,
                task,
                reply.gva.get(),
                &reply.bytes,
                page_shift,
            )
            .map_err(ReplacementCpuHostEffectError::Memory)
        }
    }
}

fn apply_control_host_effect<Semantic>(
    host: &mut impl HostControl,
    control: &ReplacementAppliedControl<Semantic>,
) {
    let action = match &control.effect {
        ReplacementControlEffect::CursorShown(position) => {
            Some(HostAction::cursor(position.x, position.y, position.visible))
        }
        ReplacementControlEffect::CursorGlyphPublished(_) => Some(HostAction::cursor_glyph()),
        ReplacementControlEffect::ContractNoOp(_)
        | ReplacementControlEffect::AbsentResourceDelete { .. }
        | ReplacementControlEffect::DebugTrace(_)
        | ReplacementControlEffect::ResourcesDiscardHint(_)
        | ReplacementControlEffect::ObjectDeleted(_)
        | ReplacementControlEffect::TaskDefined(_)
        | ReplacementControlEffect::ObjectListPublished(_)
        | ReplacementControlEffect::TaskDeleted(_)
        | ReplacementControlEffect::FifoDefined(_)
        | ReplacementControlEffect::FifoRetired(_)
        | ReplacementControlEffect::SharedStatePublished(_)
        | ReplacementControlEffect::OnlineAcknowledged(_) => None,
    };
    if let Some(action) = action {
        host.enqueue(action);
        host.schedule_bh();
    }
}

pub(crate) fn apply_cpu_host_effect<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    page_shift: u32,
    applied: ReplacementAppliedCpuPacket<Semantic>,
) -> Result<ReplacementHostAppliedCpuPacket<Semantic>, Box<ReplacementCpuHostEffectFailure<Semantic>>>
{
    if let ReplacementAppliedCpuPacket::ResourceLifecycle(lifecycle) = &applied {
        let backings = replaced_physical_backings(&lifecycle.effect);
        prepare_replaced_physical_representations(runtime, host, page_shift, &backings);
    }
    let result = match &applied {
        ReplacementAppliedCpuPacket::Control(control) => {
            apply_control_host_effect(host, control);
            Ok(())
        }
        ReplacementAppliedCpuPacket::Query(query) => {
            apply_query_host_effect(runtime, host, page_shift, query)
        }
        ReplacementAppliedCpuPacket::ResourceLifecycle(_) => Ok(()),
    };
    match result {
        Ok(()) => Ok(ReplacementHostAppliedCpuPacket(applied)),
        Err(reason) => Err(Box::new(ReplacementCpuHostEffectFailure {
            reason,
            applied,
        })),
    }
}

#[derive(Debug)]
pub(crate) enum ReplacementCpuCompletionError<Semantic> {
    Control(Box<ReplacementControlCompletionError<Semantic>>),
    Query(Box<ReplacementQueryCompletionError<Semantic>>),
    ResourceLifecycle(Box<ReplacementResourceLifecycleCompletionError<Semantic>>),
}

pub(crate) fn complete_host_applied_cpu_packet<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    applied: ReplacementHostAppliedCpuPacket<Semantic>,
    semantic: Semantic,
) -> Result<Vec<reims_vgpu_core::PublishedFact<Semantic>>, ReplacementCpuCompletionError<Semantic>>
{
    match applied.0 {
        ReplacementAppliedCpuPacket::Control(applied) => runtime
            .complete_control(applied, semantic)
            .map_err(ReplacementCpuCompletionError::Control),
        ReplacementAppliedCpuPacket::Query(applied) => runtime
            .complete_query(applied, semantic)
            .map_err(ReplacementCpuCompletionError::Query),
        ReplacementAppliedCpuPacket::ResourceLifecycle(applied) => runtime
            .complete_resource_lifecycle(applied, semantic)
            .map_err(ReplacementCpuCompletionError::ResourceLifecycle),
    }
}

pub(crate) enum ReplacementCoordinatedCpuState<Semantic> {
    Admitted {
        packet: ReplacementAdmittedCpuPacket<Semantic>,
        semantic: Semantic,
    },
    Applied {
        packet: ReplacementAppliedCpuPacket<Semantic>,
        semantic: Semantic,
    },
    HostApplied {
        packet: ReplacementHostAppliedCpuPacket<Semantic>,
        semantic: Semantic,
    },
    ApplyFailed {
        failure: ReplacementCpuApplyError<Semantic>,
        semantic: Semantic,
    },
    HostEffectFailed {
        failure: Box<ReplacementCpuHostEffectFailure<Semantic>>,
        semantic: Semantic,
    },
    CompletionFailed(ReplacementCpuCompletionError<Semantic>),
}

impl<Semantic> ReplacementCoordinatedCpuState<Semantic> {
    pub fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        match self {
            Self::Admitted { packet, .. } => packet.transaction(),
            Self::Applied { packet, .. } => packet.transaction(),
            Self::HostApplied { packet, .. } => packet.transaction(),
            Self::ApplyFailed { failure, .. } => match failure {
                ReplacementCpuApplyError::Control(failure) => match failure.as_ref() {
                    ReplacementControlApplyError::NotReady(admitted)
                    | ReplacementControlApplyError::Apply { admitted, .. } => {
                        admitted.transaction()
                    }
                },
                ReplacementCpuApplyError::Query(failure) => match failure.as_ref() {
                    ReplacementQueryApplyError::NotReady(admitted)
                    | ReplacementQueryApplyError::DeviceInfo { admitted, .. }
                    | ReplacementQueryApplyError::ComputeInfo { admitted, .. }
                    | ReplacementQueryApplyError::HeapTexture { admitted, .. } => {
                        admitted.transaction()
                    }
                },
                ReplacementCpuApplyError::ResourceLifecycle(failure) => match failure.as_ref() {
                    ReplacementResourceLifecycleApplyError::NotReady(admitted)
                    | ReplacementResourceLifecycleApplyError::Lifecycle { admitted, .. } => {
                        admitted.transaction()
                    }
                },
            },
            Self::HostEffectFailed { failure, .. } => failure.applied.transaction(),
            Self::CompletionFailed(failure) => match failure {
                ReplacementCpuCompletionError::Control(failure) => failure.applied.transaction(),
                ReplacementCpuCompletionError::Query(failure) => failure.applied.transaction(),
                ReplacementCpuCompletionError::ResourceLifecycle(failure) => {
                    failure.applied.transaction()
                }
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementCpuFailureStage {
    Apply,
    HostEffect,
    Completion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementCpuProgress {
    Pending,
    Failed(ReplacementCpuFailureStage),
    Published { facts: usize },
}

pub(crate) struct ReplacementCpuCoordinator<Semantic> {
    packets: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedCpuState<Semantic>,
    >,
    published: std::collections::VecDeque<reims_vgpu_core::PublishedFact<Semantic>>,
}

impl<Semantic> Default for ReplacementCpuCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            packets: std::collections::BTreeMap::new(),
            published: std::collections::VecDeque::new(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReplacementCpuCoordinatorAdmissionFailure<Semantic> {
    pub packet: ReplacementAdmittedCpuPacket<Semantic>,
    pub semantic: Semantic,
}

impl<Semantic: Clone> ReplacementCpuCoordinator<Semantic> {
    pub fn admit(
        &mut self,
        packet: ReplacementAdmittedCpuPacket<Semantic>,
        semantic: Semantic,
    ) -> Result<(), Box<ReplacementCpuCoordinatorAdmissionFailure<Semantic>>> {
        let transaction = packet.transaction();
        if self.packets.contains_key(&transaction) {
            return Err(Box::new(ReplacementCpuCoordinatorAdmissionFailure {
                packet,
                semantic,
            }));
        }
        self.packets.insert(
            transaction,
            ReplacementCoordinatedCpuState::Admitted { packet, semantic },
        );
        Ok(())
    }

    pub fn progress(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
        page_shift: u32,
        version: u32,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementCpuProgress> {
        let state = self.packets.remove(&transaction)?;
        debug_assert_eq!(state.transaction(), transaction);
        let (applied, semantic) = match state {
            ReplacementCoordinatedCpuState::Admitted { packet, semantic } => {
                if !runtime
                    .execution()
                    .runtime()
                    .semantic_ready()
                    .iter()
                    .any(|ready| ready.id() == transaction)
                {
                    self.packets.insert(
                        transaction,
                        ReplacementCoordinatedCpuState::Admitted { packet, semantic },
                    );
                    return Some(ReplacementCpuProgress::Pending);
                }
                match apply_ready_cpu_packet(runtime, page_shift, version, packet) {
                    Ok(applied) => (applied, semantic),
                    Err(failure) => {
                        if let Some(reason) = failure.terminal_refusal() {
                            let reason = format!("{reason:?}");
                            return Some(self.refuse_cpu_packet(
                                runtime,
                                transaction,
                                &reason,
                                semantic,
                            ));
                        }
                        self.packets.insert(
                            transaction,
                            ReplacementCoordinatedCpuState::ApplyFailed { failure, semantic },
                        );
                        return Some(ReplacementCpuProgress::Failed(
                            ReplacementCpuFailureStage::Apply,
                        ));
                    }
                }
            }
            ReplacementCoordinatedCpuState::Applied { packet, semantic } => (packet, semantic),
            failed @ (ReplacementCoordinatedCpuState::ApplyFailed { .. }
            | ReplacementCoordinatedCpuState::HostEffectFailed { .. }
            | ReplacementCoordinatedCpuState::CompletionFailed(_)) => {
                let stage = match failed {
                    ReplacementCoordinatedCpuState::ApplyFailed { .. } => {
                        ReplacementCpuFailureStage::Apply
                    }
                    ReplacementCoordinatedCpuState::HostEffectFailed { .. } => {
                        ReplacementCpuFailureStage::HostEffect
                    }
                    ReplacementCoordinatedCpuState::CompletionFailed(_) => {
                        ReplacementCpuFailureStage::Completion
                    }
                    _ => unreachable!(),
                };
                self.packets.insert(transaction, failed);
                return Some(ReplacementCpuProgress::Failed(stage));
            }
            ReplacementCoordinatedCpuState::HostApplied { packet, semantic } => {
                return Some(self.finish_host_applied(runtime, packet, semantic));
            }
        };
        let host_applied = match apply_cpu_host_effect(runtime, host, page_shift, applied) {
            Ok(applied) => applied,
            Err(failure) => {
                self.packets.insert(
                    transaction,
                    ReplacementCoordinatedCpuState::HostEffectFailed { failure, semantic },
                );
                return Some(ReplacementCpuProgress::Failed(
                    ReplacementCpuFailureStage::HostEffect,
                ));
            }
        };
        Some(self.finish_host_applied(runtime, host_applied, semantic))
    }

    /// Give up a control transaction this device has declared it cannot apply,
    /// publishing its completion so the channel's stamp still advances.
    ///
    /// The guest loses the command and is told so by name. Retaining it instead
    /// cost the whole device: a permanently-refused apply was re-offered every
    /// tick forever, its completion stamp never posted, and the guest driver
    /// blocked on that stamp for the rest of the boot with nothing in any
    /// census saying which packet it was waiting for.
    fn refuse_cpu_packet(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
        reason: &str,
        semantic: Semantic,
    ) -> ReplacementCpuProgress {
        match runtime.abandon_transaction(transaction, None, None, semantic) {
            Ok(facts) => {
                let count = facts.len();
                self.published.extend(facts);
                crate::observe::fail(format!(
                    "replacement_cpu_transaction_abandoned transaction={} reason={reason}",
                    transaction.get()
                ));
                ReplacementCpuProgress::Published { facts: count }
            }
            Err(failure) => {
                crate::observe::fail(format!(
                    "replacement_cpu_transaction_abandon_refused transaction={} reason={reason} detail={failure:?}",
                    transaction.get()
                ));
                ReplacementCpuProgress::Failed(ReplacementCpuFailureStage::Apply)
            }
        }
    }

    fn finish_host_applied(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        packet: ReplacementHostAppliedCpuPacket<Semantic>,
        semantic: Semantic,
    ) -> ReplacementCpuProgress {
        let transaction = packet.transaction();
        match complete_host_applied_cpu_packet(runtime, packet, semantic) {
            Ok(facts) => {
                let count = facts.len();
                self.published.extend(facts);
                ReplacementCpuProgress::Published { facts: count }
            }
            Err(failure) => {
                self.packets.insert(
                    transaction,
                    ReplacementCoordinatedCpuState::CompletionFailed(failure),
                );
                ReplacementCpuProgress::Failed(ReplacementCpuFailureStage::Completion)
            }
        }
    }

    pub fn take_published(&mut self) -> Option<reims_vgpu_core::PublishedFact<Semantic>> {
        self.published.pop_front()
    }

    pub fn retry_failure(&mut self, transaction: reims_vgpu_protocol::TransactionId) -> bool {
        let Some(state) = self.packets.remove(&transaction) else {
            return false;
        };
        let retry = match state {
            ReplacementCoordinatedCpuState::ApplyFailed { failure, semantic } => {
                let packet = match failure {
                    ReplacementCpuApplyError::Control(failure) => {
                        ReplacementAdmittedCpuPacket::Control(match *failure {
                            ReplacementControlApplyError::NotReady(admitted)
                            | ReplacementControlApplyError::Apply { admitted, .. } => admitted,
                        })
                    }
                    ReplacementCpuApplyError::Query(failure) => {
                        ReplacementAdmittedCpuPacket::Query(match *failure {
                            ReplacementQueryApplyError::NotReady(admitted)
                            | ReplacementQueryApplyError::DeviceInfo { admitted, .. }
                            | ReplacementQueryApplyError::ComputeInfo { admitted, .. }
                            | ReplacementQueryApplyError::HeapTexture { admitted, .. } => admitted,
                        })
                    }
                    ReplacementCpuApplyError::ResourceLifecycle(failure) => {
                        ReplacementAdmittedCpuPacket::ResourceLifecycle(match *failure {
                            ReplacementResourceLifecycleApplyError::NotReady(admitted)
                            | ReplacementResourceLifecycleApplyError::Lifecycle {
                                admitted, ..
                            } => admitted,
                        })
                    }
                };
                ReplacementCoordinatedCpuState::Admitted { packet, semantic }
            }
            ReplacementCoordinatedCpuState::HostEffectFailed { failure, semantic } => {
                ReplacementCoordinatedCpuState::Applied {
                    packet: failure.applied,
                    semantic,
                }
            }
            ReplacementCoordinatedCpuState::CompletionFailed(failure) => match failure {
                ReplacementCpuCompletionError::Control(failure) => {
                    ReplacementCoordinatedCpuState::HostApplied {
                        packet: ReplacementHostAppliedCpuPacket(
                            ReplacementAppliedCpuPacket::Control(failure.applied),
                        ),
                        semantic: failure.semantic,
                    }
                }
                ReplacementCpuCompletionError::Query(failure) => {
                    ReplacementCoordinatedCpuState::HostApplied {
                        packet: ReplacementHostAppliedCpuPacket(
                            ReplacementAppliedCpuPacket::Query(failure.applied),
                        ),
                        semantic: failure.semantic,
                    }
                }
                ReplacementCpuCompletionError::ResourceLifecycle(failure) => {
                    ReplacementCoordinatedCpuState::HostApplied {
                        packet: ReplacementHostAppliedCpuPacket(
                            ReplacementAppliedCpuPacket::ResourceLifecycle(failure.applied),
                        ),
                        semantic: failure.semantic,
                    }
                }
            },
            unchanged => {
                self.packets.insert(transaction, unchanged);
                return false;
            }
        };
        debug_assert_eq!(retry.transaction(), transaction);
        self.packets.insert(transaction, retry);
        true
    }

    pub fn publish_next(
        &mut self,
        host: &mut (impl HostMemory + HostControl),
        transport: &crate::runtime::replacement_transport::ReplacementTransportOwner,
    ) -> Result<Option<ReplacementHostPublishedFact<Semantic>>, ReplacementPublishedFactHostError>
    {
        let Some(fact) = self.published.pop_front() else {
            return Ok(None);
        };
        match publish_ordered_fact_to_host(
            host,
            transport.gpu_interrupt_status(),
            transport.stamp_page(),
            fact,
        ) {
            Ok(published) => Ok(Some(published)),
            Err(failure) => {
                let ReplacementPublishedFactHostFailure { reason, fact } = *failure;
                self.published.push_front(fact);
                Err(reason)
            }
        }
    }

    pub fn live_packets(&self) -> usize {
        self.packets.len()
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.packets.keys().copied().collect()
    }

    pub fn pending_publications(&self) -> usize {
        self.published.len()
    }
}

impl<Semantic: std::fmt::Debug> ReplacementCpuCoordinator<Semantic> {
    fn failure_diagnostics(
        &self,
    ) -> Vec<(reims_vgpu_protocol::TransactionId, &'static str, String)> {
        self.packets
            .iter()
            .filter_map(|(transaction, state)| match state {
                ReplacementCoordinatedCpuState::ApplyFailed { failure, .. } => {
                    Some((*transaction, "apply", format!("{failure:?}")))
                }
                ReplacementCoordinatedCpuState::HostEffectFailed { failure, .. } => {
                    Some((*transaction, "host_effect", format!("{failure:?}")))
                }
                ReplacementCoordinatedCpuState::CompletionFailed(failure) => {
                    Some((*transaction, "completion", format!("{failure:?}")))
                }
                ReplacementCoordinatedCpuState::Admitted { .. }
                | ReplacementCoordinatedCpuState::Applied { .. }
                | ReplacementCoordinatedCpuState::HostApplied { .. } => None,
            })
            .collect()
    }
}

pub(crate) enum ReplacementCoordinatedPresentPreparation<Semantic> {
    Admitted(Box<crate::runtime::replacement_session::ReplacementAdmittedPresent<Semantic>>),
    Ready(Box<crate::runtime::replacement_session::ReplacementReadyPresent<Semantic>>),
    Prepared(Box<crate::runtime::replacement_session::ReplacementPreparedNativePresent<Semantic>>),
    Allocated(
        Box<crate::runtime::replacement_session::ReplacementAllocatedNativePresent<Semantic>>,
    ),
    NativePreparationFailed(
        Box<
            crate::runtime::replacement_session::ReplacementNativePresentPreparationError<Semantic>,
        >,
    ),
    AllocationFailed(
        Box<crate::runtime::replacement_session::ReplacementNativePresentAllocationError<Semantic>>,
    ),
    PreparedQueue(
        Box<crate::runtime::replacement_session::ReplacementPreparedNativePresentQueue<Semantic>>,
    ),
    PendingQueue(
        Box<crate::runtime::replacement_session::ReplacementPendingNativePresentQueue<Semantic>>,
    ),
    QueuePreparationFailed(
        Box<crate::runtime::replacement_session::ReplacementPresentQueuePreparationError<Semantic>>,
    ),
    ConsoleQueuePreparationFailed(
        Box<
            crate::runtime::replacement_session::ReplacementConsolePresentQueuePreparationError<
                Semantic,
            >,
        >,
    ),
    #[cfg(feature = "host-window")]
    WindowQueuePreparationFailed(
        Box<
            crate::runtime::replacement_session::ReplacementWindowPresentQueuePreparationError<
                Semantic,
            >,
        >,
    ),
    QueueEnqueueFailed(
        Box<crate::runtime::replacement_session::ReplacementPresentQueueEnqueueError<Semantic>>,
    ),
    DriverRefused {
        reason: reims_vgpu_vulkan::replacement_queue::ReplacementQueueError,
        prepared: Box<
            crate::runtime::replacement_session::ReplacementPreparedNativePresentQueue<Semantic>,
        >,
    },
    AcceptanceFailed(
        Box<
            crate::runtime::replacement_session::ReplacementNativePresentCompletionError<
                Semantic,
                (
                    crate::runtime::replacement_session::ReplacementPresentSubmitContext<Semantic>,
                    reims_vgpu_vulkan::replacement_queue::DriverAcceptedReplacementPresentQueueSubmission,
                ),
            >,
        >,
    ),
    Queued(Box<crate::runtime::replacement_session::ReplacementQueuedNativePresent<Semantic>>),
    TimelineFailed(
        Box<
            crate::runtime::replacement_session::ReplacementNativePresentCompletionError<
                Semantic,
                crate::runtime::replacement_session::ReplacementQueuedNativePresent<Semantic>,
            >,
        >,
    ),
    Completed(Box<crate::runtime::replacement_session::ReplacementCompletedNativePresent<Semantic>>),
    NotificationPreparationFailed(
        Box<
            crate::runtime::replacement_session::ReplacementPresentNotificationError<
                Semantic,
                crate::runtime::replacement_session::ReplacementCompletedNativePresent<Semantic>,
            >,
        >,
    ),
    PreparedNotification(
        Box<crate::runtime::replacement_session::ReplacementPreparedPresentNotification<Semantic>>,
    ),
    NotificationApplyFailed(
        Box<
            crate::runtime::replacement_session::ReplacementPresentNotificationError<
                Semantic,
                crate::runtime::replacement_session::ReplacementPreparedPresentNotification<Semantic>,
            >,
        >,
    ),
    Notified(Box<crate::runtime::replacement_session::ReplacementNotifiedNativePresent<Semantic>>),
    CompletionFailed(
        Box<crate::runtime::replacement_session::ReplacementPresentCompletionError<Semantic>>,
    ),
}

impl<Semantic> ReplacementCoordinatedPresentPreparation<Semantic> {
    pub fn transaction(&self) -> reims_vgpu_protocol::TransactionId {
        match self {
            Self::Admitted(state) => state.transaction(),
            Self::Ready(state) => state.transaction(),
            Self::Prepared(state) => state.transaction(),
            Self::Allocated(state) => state.transaction(),
            Self::NativePreparationFailed(failure) => failure.ready.transaction(),
            Self::AllocationFailed(failure) => failure.prepared.transaction(),
            Self::PreparedQueue(state) => state.transaction(),
            Self::PendingQueue(state) => state.transaction(),
            Self::QueuePreparationFailed(failure) => failure.allocated.transaction(),
            Self::ConsoleQueuePreparationFailed(failure) => failure.allocated.transaction(),
            #[cfg(feature = "host-window")]
            Self::WindowQueuePreparationFailed(failure) => failure.allocated.transaction(),
            Self::QueueEnqueueFailed(failure) => failure.prepared.transaction(),
            Self::DriverRefused { prepared, .. } => prepared.transaction(),
            Self::AcceptanceFailed(failure) => failure.state.0.transaction(),
            Self::Queued(state) => state.transaction(),
            Self::TimelineFailed(failure) => failure.state.transaction(),
            Self::Completed(state) => state.transaction(),
            Self::NotificationPreparationFailed(failure) => failure.state().transaction(),
            Self::PreparedNotification(state) => state.transaction(),
            Self::NotificationApplyFailed(failure) => failure.state().transaction(),
            Self::Notified(state) => state.transaction(),
            Self::CompletionFailed(failure) => match failure.as_ref() {
                crate::runtime::replacement_session::ReplacementPresentCompletionError::NotReady(state) => state.transaction(),
                crate::runtime::replacement_session::ReplacementPresentCompletionError::Publication { completion, .. } => completion.transaction(),
            },
        }
    }
}

pub(crate) struct ReplacementCoordinatedPresentPreparationEntry<Semantic> {
    state: ReplacementCoordinatedPresentPreparation<Semantic>,
    semantic: Semantic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementPresentPreparationProgress {
    Pending,
    FailedNativePreparation,
    FailedAllocation,
    Allocated,
    BeyondPreparation,
}

#[derive(Debug)]
pub(crate) enum ReplacementPresentQueueCoordinatorProgress {
    Prepared,
    Busy,
    Pending,
    DriverAccepted {
        backing: reims_vgpu_core::ManagedBackingProgress<
            reims_vgpu_vulkan::replacement_representation::ReplacementNativeRepresentation,
        >,
    },
    FailedPreparation,
    FailedEnqueue,
    DriverRefused,
    FailedAcceptance,
    WrongStage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementPresentCompletionProgress {
    TimelineComplete,
    NotificationPrepared,
    Notified,
    Published { facts: usize },
    FailedTimeline,
    FailedNotificationPreparation,
    FailedNotificationApply,
    FailedPublication,
    WrongStage,
}

pub(crate) struct ReplacementPresentCoordinator<Semantic> {
    preparations: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedPresentPreparationEntry<Semantic>,
    >,
    published: std::collections::VecDeque<reims_vgpu_core::PublishedFact<Semantic>>,
}

impl<Semantic> Default for ReplacementPresentCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            preparations: std::collections::BTreeMap::new(),
            published: std::collections::VecDeque::new(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReplacementPresentCoordinatorAdmissionFailure<Semantic> {
    pub present: crate::runtime::replacement_session::ReplacementAdmittedPresent<Semantic>,
    pub semantic: Semantic,
}

impl<Semantic: Clone> ReplacementPresentCoordinator<Semantic> {
    pub fn admit(
        &mut self,
        present: crate::runtime::replacement_session::ReplacementAdmittedPresent<Semantic>,
        semantic: Semantic,
    ) -> Result<(), Box<ReplacementPresentCoordinatorAdmissionFailure<Semantic>>> {
        let transaction = present.transaction();
        if self.preparations.contains_key(&transaction) {
            return Err(Box::new(ReplacementPresentCoordinatorAdmissionFailure {
                present,
                semantic,
            }));
        }
        self.preparations.insert(
            transaction,
            ReplacementCoordinatedPresentPreparationEntry {
                state: ReplacementCoordinatedPresentPreparation::Admitted(Box::new(present)),
                semantic,
            },
        );
        Ok(())
    }

    pub fn progress_preparation(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentPreparationProgress> {
        let ReplacementCoordinatedPresentPreparationEntry { state, semantic } =
            self.preparations.remove(&transaction)?;
        debug_assert_eq!(state.transaction(), transaction);
        let prepared = match state {
            ReplacementCoordinatedPresentPreparation::Prepared(prepared) => *prepared,
            ReplacementCoordinatedPresentPreparation::Allocated(allocated) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::Allocated(allocated),
                        semantic,
                    },
                );
                return Some(ReplacementPresentPreparationProgress::Allocated);
            }
            failed @ (ReplacementCoordinatedPresentPreparation::NativePreparationFailed(_)
            | ReplacementCoordinatedPresentPreparation::AllocationFailed(_)) => {
                let progress = match failed {
                    ReplacementCoordinatedPresentPreparation::NativePreparationFailed(_) => {
                        ReplacementPresentPreparationProgress::FailedNativePreparation
                    }
                    ReplacementCoordinatedPresentPreparation::AllocationFailed(_) => {
                        ReplacementPresentPreparationProgress::FailedAllocation
                    }
                    _ => unreachable!(),
                };
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: failed,
                        semantic,
                    },
                );
                return Some(progress);
            }
            #[cfg(feature = "host-window")]
            later @ ReplacementCoordinatedPresentPreparation::WindowQueuePreparationFailed(_) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: later,
                        semantic,
                    },
                );
                return Some(ReplacementPresentPreparationProgress::BeyondPreparation);
            }
            later @ (ReplacementCoordinatedPresentPreparation::PreparedQueue(_)
            | ReplacementCoordinatedPresentPreparation::PendingQueue(_)
            | ReplacementCoordinatedPresentPreparation::QueuePreparationFailed(_)
            | ReplacementCoordinatedPresentPreparation::ConsoleQueuePreparationFailed(_)
            | ReplacementCoordinatedPresentPreparation::QueueEnqueueFailed(_)
            | ReplacementCoordinatedPresentPreparation::DriverRefused { .. }
            | ReplacementCoordinatedPresentPreparation::AcceptanceFailed(_)
            | ReplacementCoordinatedPresentPreparation::Queued(_)
            | ReplacementCoordinatedPresentPreparation::TimelineFailed(_)
            | ReplacementCoordinatedPresentPreparation::Completed(_)
            | ReplacementCoordinatedPresentPreparation::NotificationPreparationFailed(_)
            | ReplacementCoordinatedPresentPreparation::PreparedNotification(_)
            | ReplacementCoordinatedPresentPreparation::NotificationApplyFailed(_)
            | ReplacementCoordinatedPresentPreparation::Notified(_)
            | ReplacementCoordinatedPresentPreparation::CompletionFailed(_)) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: later,
                        semantic,
                    },
                );
                return Some(ReplacementPresentPreparationProgress::BeyondPreparation);
            }
            early @ (ReplacementCoordinatedPresentPreparation::Admitted(_)
            | ReplacementCoordinatedPresentPreparation::Ready(_)) => {
                let ready = match early {
                    ReplacementCoordinatedPresentPreparation::Ready(ready) => *ready,
                    ReplacementCoordinatedPresentPreparation::Admitted(admitted) => {
                        match runtime.prepare_admitted_present(*admitted) {
                            Ok(ready) => ready,
                            Err(failure) => {
                                let admitted = match *failure {
                                    crate::runtime::replacement_session::ReplacementPresentCompletionError::NotReady(admitted) => admitted,
                                    crate::runtime::replacement_session::ReplacementPresentCompletionError::Publication { .. } => {
                                        unreachable!("readiness cannot publish a Present transaction")
                                    }
                                };
                                self.preparations.insert(
                                    transaction,
                                    ReplacementCoordinatedPresentPreparationEntry {
                                        state: ReplacementCoordinatedPresentPreparation::Admitted(
                                            Box::new(admitted),
                                        ),
                                        semantic,
                                    },
                                );
                                return Some(ReplacementPresentPreparationProgress::Pending);
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                match runtime.prepare_native_present(ready) {
                    Ok(prepared) => prepared,
                    Err(failure) => {
                        self.preparations.insert(
                            transaction,
                            ReplacementCoordinatedPresentPreparationEntry {
                                state: ReplacementCoordinatedPresentPreparation::NativePreparationFailed(
                                    failure,
                                ),
                                semantic,
                            },
                        );
                        return Some(
                            ReplacementPresentPreparationProgress::FailedNativePreparation,
                        );
                    }
                }
            }
        };
        match runtime.allocate_native_present(prepared) {
            Ok(allocated) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::Allocated(Box::new(
                            allocated,
                        )),
                        semantic,
                    },
                );
                Some(ReplacementPresentPreparationProgress::Allocated)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::AllocationFailed(failure),
                        semantic,
                    },
                );
                Some(ReplacementPresentPreparationProgress::FailedAllocation)
            }
        }
    }

    /// Record a transaction-owned console readback while this runtime owns
    /// the exact Vulkan epoch referenced by the allocated source.
    ///
    /// # Safety
    ///
    /// The coordinator and runtime must still own the live Vulkan epoch in
    /// which the allocated image and transitions were prepared.
    pub unsafe fn prepare_console_queue(
        &mut self,
        runtime: &ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentQueueCoordinatorProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let allocated = match entry.state {
            ReplacementCoordinatedPresentPreparation::Allocated(allocated) => *allocated,
            ReplacementCoordinatedPresentPreparation::ConsoleQueuePreparationFailed(failure) => {
                failure.allocated
            }
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentQueueCoordinatorProgress::WrongStage);
            }
        };
        match unsafe { runtime.prepare_console_native_present_queue(allocated) } {
            Ok(prepared) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::PreparedQueue(Box::new(
                            prepared,
                        )),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::Prepared)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state:
                            ReplacementCoordinatedPresentPreparation::ConsoleQueuePreparationFailed(
                                failure,
                            ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::FailedPreparation)
            }
        }
    }

    pub fn retry_preparation_failure(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> bool {
        let Some(entry) = self.preparations.remove(&transaction) else {
            return false;
        };
        let state = match entry.state {
            ReplacementCoordinatedPresentPreparation::NativePreparationFailed(failure) => {
                ReplacementCoordinatedPresentPreparation::Ready(Box::new(failure.ready))
            }
            ReplacementCoordinatedPresentPreparation::AllocationFailed(failure) => {
                ReplacementCoordinatedPresentPreparation::Prepared(Box::new(failure.prepared))
            }
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return false;
            }
        };
        self.preparations.insert(
            transaction,
            ReplacementCoordinatedPresentPreparationEntry {
                state,
                semantic: entry.semantic,
            },
        );
        true
    }

    pub fn prepare_queue(
        &mut self,
        runtime: &ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
        recording: reims_vgpu_vulkan::replacement_queue::ReplacementPresentRecording,
    ) -> Option<ReplacementPresentQueueCoordinatorProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let (allocated, recording) = match entry.state {
            ReplacementCoordinatedPresentPreparation::Allocated(allocated) => {
                (*allocated, recording)
            }
            ReplacementCoordinatedPresentPreparation::QueuePreparationFailed(failure) => {
                (failure.allocated, failure.recording)
            }
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentQueueCoordinatorProgress::WrongStage);
            }
        };
        match runtime.prepare_native_present_queue(allocated, recording) {
            Ok(prepared) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::PreparedQueue(Box::new(
                            prepared,
                        )),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::Prepared)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::QueuePreparationFailed(
                            failure,
                        ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::FailedPreparation)
            }
        }
    }

    #[cfg(feature = "host-window")]
    /// Acquire and record the host-window presentation while this runtime owns
    /// the exact Vulkan epoch referenced by the allocated source.
    ///
    /// # Safety
    ///
    /// The coordinator and runtime must still own the live Vulkan epoch in
    /// which the allocated image and transitions were prepared.
    pub unsafe fn prepare_window_queue(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentQueueCoordinatorProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let allocated = match entry.state {
            ReplacementCoordinatedPresentPreparation::Allocated(allocated) => *allocated,
            ReplacementCoordinatedPresentPreparation::WindowQueuePreparationFailed(failure) => {
                failure.allocated
            }
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentQueueCoordinatorProgress::WrongStage);
            }
        };
        match unsafe { runtime.prepare_window_native_present_queue(allocated) } {
            Ok(
                crate::runtime::replacement_session::ReplacementWindowPresentQueueDispatch::Busy(
                    allocated,
                ),
            ) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::Allocated(allocated),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::Busy)
            }
            Ok(
                crate::runtime::replacement_session::ReplacementWindowPresentQueueDispatch::Prepared(
                    prepared,
                ),
            ) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::PreparedQueue(prepared),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::Prepared)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::WindowQueuePreparationFailed(
                            failure,
                        ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::FailedPreparation)
            }
        }
    }

    pub fn enqueue_queue(
        &mut self,
        runtime: &ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentQueueCoordinatorProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let prepared = match entry.state {
            ReplacementCoordinatedPresentPreparation::PreparedQueue(prepared) => *prepared,
            ReplacementCoordinatedPresentPreparation::QueueEnqueueFailed(failure) => {
                failure.prepared
            }
            ReplacementCoordinatedPresentPreparation::DriverRefused { prepared, .. } => *prepared,
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentQueueCoordinatorProgress::WrongStage);
            }
        };
        match runtime.enqueue_native_present(prepared) {
            Ok(pending) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::PendingQueue(Box::new(
                            pending,
                        )),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::Pending)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::QueueEnqueueFailed(
                            failure,
                        ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::FailedEnqueue)
            }
        }
    }

    pub fn poll_queue(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentQueueCoordinatorProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let pending = match entry.state {
            ReplacementCoordinatedPresentPreparation::PendingQueue(pending) => *pending,
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentQueueCoordinatorProgress::WrongStage);
            }
        };
        match runtime.progress_native_present_queue(pending) {
            crate::runtime::replacement_session::ReplacementPresentQueueProgress::Pending(
                pending,
            ) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::PendingQueue(Box::new(
                            pending,
                        )),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::Pending)
            }
            crate::runtime::replacement_session::ReplacementPresentQueueProgress::DriverRefused {
                reason,
                prepared,
            } => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::DriverRefused {
                            reason,
                            prepared: Box::new(prepared),
                        },
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::DriverRefused)
            }
            crate::runtime::replacement_session::ReplacementPresentQueueProgress::DriverAccepted {
                context,
                accepted,
            } => match runtime.accept_native_present_submission(context, accepted) {
                Ok((queued, backing)) => {
                    self.preparations.insert(
                        transaction,
                        ReplacementCoordinatedPresentPreparationEntry {
                            state: ReplacementCoordinatedPresentPreparation::Queued(Box::new(
                                queued,
                            )),
                            semantic: entry.semantic,
                        },
                    );
                    Some(ReplacementPresentQueueCoordinatorProgress::DriverAccepted { backing })
                }
                Err(failure) => {
                    self.preparations.insert(
                        transaction,
                        ReplacementCoordinatedPresentPreparationEntry {
                            state: ReplacementCoordinatedPresentPreparation::AcceptanceFailed(
                                failure,
                            ),
                            semantic: entry.semantic,
                        },
                    );
                    Some(ReplacementPresentQueueCoordinatorProgress::FailedAcceptance)
                }
            },
        }
    }

    pub fn retry_acceptance(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentQueueCoordinatorProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let (context, accepted) = match entry.state {
            ReplacementCoordinatedPresentPreparation::AcceptanceFailed(failure) => failure.state,
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentQueueCoordinatorProgress::WrongStage);
            }
        };
        match runtime.accept_native_present_submission(context, accepted) {
            Ok((queued, backing)) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::Queued(Box::new(queued)),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::DriverAccepted { backing })
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::AcceptanceFailed(failure),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentQueueCoordinatorProgress::FailedAcceptance)
            }
        }
    }

    pub fn observe_timeline(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
        observed: reims_vgpu_core::QueueTimelinePoint,
    ) -> Option<ReplacementPresentCompletionProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let queued = match entry.state {
            ReplacementCoordinatedPresentPreparation::Queued(queued) => *queued,
            ReplacementCoordinatedPresentPreparation::TimelineFailed(failure) => failure.state,
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentCompletionProgress::WrongStage);
            }
        };
        match runtime.observe_native_present_completion(queued, observed) {
            Ok(completed) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::Completed(Box::new(
                            completed,
                        )),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::TimelineComplete)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::TimelineFailed(failure),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::FailedTimeline)
            }
        }
    }

    pub fn prepare_notification(
        &mut self,
        runtime: &ReplacementRuntimeSession<Semantic>,
        host: &impl HostMemory,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentCompletionProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let completed = match entry.state {
            ReplacementCoordinatedPresentPreparation::Completed(completed) => *completed,
            ReplacementCoordinatedPresentPreparation::NotificationPreparationFailed(failure) => {
                (*failure).into_state()
            }
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentCompletionProgress::WrongStage);
            }
        };
        match runtime.prepare_present_notification(host, completed) {
            Ok(prepared) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::PreparedNotification(
                            Box::new(prepared),
                        ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::NotificationPrepared)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state:
                            ReplacementCoordinatedPresentPreparation::NotificationPreparationFailed(
                                failure,
                            ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::FailedNotificationPreparation)
            }
        }
    }

    pub fn apply_notification(
        &mut self,
        runtime: &ReplacementRuntimeSession<Semantic>,
        host: &mut (impl HostMemory + HostControl),
        transport: &crate::runtime::replacement_transport::ReplacementTransportOwner,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentCompletionProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let prepared = match entry.state {
            ReplacementCoordinatedPresentPreparation::PreparedNotification(prepared) => *prepared,
            ReplacementCoordinatedPresentPreparation::NotificationApplyFailed(failure) => {
                (*failure).into_state()
            }
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentCompletionProgress::WrongStage);
            }
        };
        match runtime.apply_present_notification(host, transport.gpu_interrupt_status(), prepared) {
            Ok(notified) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::Notified(Box::new(
                            notified,
                        )),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::Notified)
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::NotificationApplyFailed(
                            failure,
                        ),
                        semantic: entry.semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::FailedNotificationApply)
            }
        }
    }

    pub fn complete(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementPresentCompletionProgress> {
        let entry = self.preparations.remove(&transaction)?;
        let retained_semantic = entry.semantic.clone();
        let result = match entry.state {
            ReplacementCoordinatedPresentPreparation::Notified(notified) => {
                runtime.complete_present(*notified, entry.semantic)
            }
            ReplacementCoordinatedPresentPreparation::CompletionFailed(failure) => match *failure {
                crate::runtime::replacement_session::ReplacementPresentCompletionError::Publication {
                    completion,
                    semantic,
                    ..
                } => runtime.retry_present_completion(*completion, semantic),
                crate::runtime::replacement_session::ReplacementPresentCompletionError::NotReady(_) => {
                    unreachable!("native completion cannot return to admission readiness")
                }
            },
            unchanged => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: unchanged,
                        semantic: entry.semantic,
                    },
                );
                return Some(ReplacementPresentCompletionProgress::WrongStage);
            }
        };
        match result {
            Ok(facts) => {
                let count = facts.len();
                self.published.extend(facts);
                Some(ReplacementPresentCompletionProgress::Published { facts: count })
            }
            Err(failure) => {
                self.preparations.insert(
                    transaction,
                    ReplacementCoordinatedPresentPreparationEntry {
                        state: ReplacementCoordinatedPresentPreparation::CompletionFailed(failure),
                        semantic: retained_semantic,
                    },
                );
                Some(ReplacementPresentCompletionProgress::FailedPublication)
            }
        }
    }

    pub fn take_published(&mut self) -> Option<reims_vgpu_core::PublishedFact<Semantic>> {
        self.published.pop_front()
    }

    pub fn publish_next(
        &mut self,
        host: &mut (impl HostMemory + HostControl),
        transport: &crate::runtime::replacement_transport::ReplacementTransportOwner,
    ) -> Result<Option<ReplacementHostPublishedFact<Semantic>>, ReplacementPublishedFactHostError>
    {
        let Some(fact) = self.published.pop_front() else {
            return Ok(None);
        };
        match publish_ordered_fact_to_host(
            host,
            transport.gpu_interrupt_status(),
            transport.stamp_page(),
            fact,
        ) {
            Ok(published) => Ok(Some(published)),
            Err(failure) => {
                let ReplacementPublishedFactHostFailure { reason, fact } = *failure;
                self.published.push_front(fact);
                Err(reason)
            }
        }
    }

    pub fn pending_publications(&self) -> usize {
        self.published.len()
    }

    pub fn live_presentations(&self) -> usize {
        self.preparations.len()
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.preparations.keys().copied().collect()
    }

    pub fn queued_point(
        &self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<reims_vgpu_core::QueueTimelinePoint> {
        match &self.preparations.get(&transaction)?.state {
            ReplacementCoordinatedPresentPreparation::Queued(queued) => Some(queued.point()),
            ReplacementCoordinatedPresentPreparation::TimelineFailed(failure) => {
                Some(failure.state.point())
            }
            _ => None,
        }
    }

    pub fn take_console_frame(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<reims_vgpu_vulkan::replacement_console_present::ReplacementConsoleFrame> {
        let entry = self.preparations.get_mut(&transaction)?;
        match &mut entry.state {
            ReplacementCoordinatedPresentPreparation::Completed(completed) => {
                completed.take_console_frame()
            }
            _ => None,
        }
    }

    pub fn has_console_frame(&self, transaction: reims_vgpu_protocol::TransactionId) -> bool {
        self.preparations
            .get(&transaction)
            .is_some_and(|entry| match &entry.state {
                ReplacementCoordinatedPresentPreparation::Completed(completed) => {
                    completed.has_console_frame()
                }
                _ => false,
            })
    }
}

pub(crate) enum ReplacementCoordinatedExecRecording<Semantic> {
    Pending(crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>),
    Failed(
        Box<
            crate::runtime::replacement_session::ReplacementExecIngressRecordingProgressFailure<
                Semantic,
            >,
        >,
    ),
}

/// What one poll of an admitted EXEC recording settled. Every arm states where
/// the exact owner lives afterwards, so a caller cannot observe readiness
/// without also receiving the owner and the route it takes. A terminal owner is
/// never retained here: readiness hands the owner out and parking hands it to
/// the runtime epoch, which is why a stale route-erased marker cannot exist.
pub(crate) enum ReplacementExecRecordingDisposition<Semantic> {
    /// Recording is still in flight. The coordinator retains the exact owner.
    Pending,
    /// The recording became queue-ready. The owner has left this coordinator
    /// and the caller routes it by its own variant.
    Ready(crate::runtime::replacement_session::ReplacementQueueReadyRecording<Semantic>),
    /// Ownership moved into the runtime epoch parked map behind a predecessor.
    /// Nothing remains here for the transaction; the epoch releases it through
    /// `take_newly_ready_recorded_execs` once that predecessor is accepted.
    Parked(reims_vgpu_protocol::TransactionId),
    /// Progress refused on this attempt, named for the always-on failure path.
    /// The coordinator retains the exact failure so the next poll resumes from
    /// it rather than decoding or preparing the EXEC again. A refusal is
    /// therefore not proof of lost work — an ordering refusal is expected to
    /// clear once the transaction ahead of it is accepted — but it is reported
    /// so a refusal that never clears is visible instead of silent.
    Failed(ReplacementExecRecordingRefusal),
}

/// One refused EXEC recording, named for the always-on failure path: the stage
/// that refused, and the typed reason that stage carried.
pub(crate) struct ReplacementExecRecordingRefusal {
    pub stage: &'static str,
    pub detail: Option<String>,
}

pub(crate) struct ReplacementExecRecordingCoordinator<Semantic> {
    recordings: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedExecRecording<Semantic>,
    >,
}

impl<Semantic> Default for ReplacementExecRecordingCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            recordings: std::collections::BTreeMap::new(),
        }
    }
}

pub(crate) struct ReplacementExecRecordingCoordinatorAdmissionFailure<Semantic> {
    pub pending: crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>,
}

impl<Semantic: Clone + PartialEq + Send + 'static> ReplacementExecRecordingCoordinator<Semantic> {
    pub fn admit(
        &mut self,
        pending: crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        Box<ReplacementExecRecordingCoordinatorAdmissionFailure<Semantic>>,
    > {
        let transaction = pending.transaction();
        if self.recordings.contains_key(&transaction) {
            return Err(Box::new(
                ReplacementExecRecordingCoordinatorAdmissionFailure { pending },
            ));
        }
        self.recordings.insert(
            transaction,
            ReplacementCoordinatedExecRecording::Pending(pending),
        );
        Ok(transaction)
    }

    pub fn poll(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementExecRecordingDisposition<Semantic>> {
        use crate::runtime::replacement_session as session;
        let state = self.recordings.remove(&transaction)?;
        let result = match state {
            ReplacementCoordinatedExecRecording::Pending(pending) => {
                runtime.progress_exec_ingress_recording(pending)
            }
            ReplacementCoordinatedExecRecording::Failed(failure) => {
                runtime.retry_exec_ingress_recording_progress(*failure)
            }
        };
        let progress = match result {
            Ok(progress) => progress,
            Err(failure) => {
                let refusal = ReplacementExecRecordingRefusal {
                    stage: failure.reason(),
                    detail: failure.detail(),
                };
                self.recordings.insert(
                    transaction,
                    ReplacementCoordinatedExecRecording::Failed(Box::new(failure)),
                );
                return Some(ReplacementExecRecordingDisposition::Failed(refusal));
            }
        };
        let pending = match progress {
            session::ReplacementExecIngressRecordingProgress::Direct(
                session::ReplacementDirectRecordingProgress::Pending(pending),
            ) => session::PendingReplacementIngressExec::Direct(pending),
            session::ReplacementExecIngressRecordingProgress::Direct(
                session::ReplacementDirectRecordingProgress::Ready(ready),
            ) => {
                return Some(ReplacementExecRecordingDisposition::Ready(
                    session::ReplacementQueueReadyRecording::Exec(ready),
                ));
            }
            session::ReplacementExecIngressRecordingProgress::Direct(
                session::ReplacementDirectRecordingProgress::Parked(parked),
            ) => return Some(ReplacementExecRecordingDisposition::Parked(parked)),
            session::ReplacementExecIngressRecordingProgress::GuestUpload(
                session::ReplacementGuestUploadIngressRecordingProgress::Pending(pending),
            ) => session::PendingReplacementIngressExec::GuestUpload(pending),
            session::ReplacementExecIngressRecordingProgress::GuestUpload(
                session::ReplacementGuestUploadIngressRecordingProgress::Ready(ready),
            ) => {
                return Some(ReplacementExecRecordingDisposition::Ready(
                    session::ReplacementQueueReadyRecording::GuestUpload(ready),
                ));
            }
            session::ReplacementExecIngressRecordingProgress::GuestUpload(
                session::ReplacementGuestUploadIngressRecordingProgress::Parked(parked),
            ) => return Some(ReplacementExecRecordingDisposition::Parked(parked)),
            session::ReplacementExecIngressRecordingProgress::IndirectRange(progress) => {
                match *progress {
                    session::ReplacementIndirectRangeIngressRecordingProgress::Pending(pending) => {
                        session::PendingReplacementIngressExec::IndirectRange(pending)
                    }
                    session::ReplacementIndirectRangeIngressRecordingProgress::Initial(
                        session::ReplacementRecordedIndirectRangeQueueDisposition::Ready(ready),
                    ) => {
                        return Some(ReplacementExecRecordingDisposition::Ready(
                            session::ReplacementQueueReadyRecording::IndirectRange(ready),
                        ));
                    }
                    session::ReplacementIndirectRangeIngressRecordingProgress::Initial(
                        session::ReplacementRecordedIndirectRangeQueueDisposition::Parked(parked),
                    ) => return Some(ReplacementExecRecordingDisposition::Parked(parked)),
                }
            }
        };
        self.recordings.insert(
            transaction,
            ReplacementCoordinatedExecRecording::Pending(pending),
        );
        Some(ReplacementExecRecordingDisposition::Pending)
    }

    pub fn live_recordings(&self) -> usize {
        self.recordings.len()
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.recordings.keys().copied().collect()
    }
}

pub(crate) enum ReplacementCoordinatedExecSubmit<Semantic> {
    Ready {
        ready: Box<crate::runtime::replacement_session::ReplacementQueueReadyExec>,
        semantic: Semantic,
    },
    Pending(Box<crate::runtime::replacement_session::PendingReplacementExecSubmit<Semantic>>),
    SubmitFailed(
        Box<crate::runtime::replacement_session::ReplacementQueueReadySubmitFailure<Semantic>>,
    ),
    Terminal(Box<crate::runtime::replacement_session::ReplacementExecSubmitPoll<Semantic>>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementExecSubmitCoordinatorProgress {
    Pending,
    Accepted,
    DriverRefused,
    AcceptanceRefused,
    FailedSubmit,
    WrongStage,
}

impl ReplacementExecSubmitCoordinatorProgress {
    /// Whether this step is guest work that stopped moving. `WrongStage` is the
    /// poll another arm already owns and is expected control flow, not a loss.
    const fn is_refusal(self) -> bool {
        match self {
            Self::Pending | Self::Accepted | Self::WrongStage => false,
            Self::DriverRefused | Self::AcceptanceRefused | Self::FailedSubmit => true,
        }
    }
}

pub(crate) struct ReplacementExecSubmitCoordinator<Semantic> {
    submissions: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedExecSubmit<Semantic>,
    >,
}

impl<Semantic> Default for ReplacementExecSubmitCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            submissions: std::collections::BTreeMap::new(),
        }
    }
}

pub(crate) struct ReplacementExecSubmitCoordinatorAdmissionFailure<Semantic> {
    pub ready: Box<crate::runtime::replacement_session::ReplacementQueueReadyExec>,
    pub semantic: Semantic,
}

impl<Semantic: Clone> ReplacementExecSubmitCoordinator<Semantic> {
    pub fn admit(
        &mut self,
        ready: Box<crate::runtime::replacement_session::ReplacementQueueReadyExec>,
        semantic: Semantic,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        Box<ReplacementExecSubmitCoordinatorAdmissionFailure<Semantic>>,
    > {
        let transaction = ready.plan.transaction;
        if self.submissions.contains_key(&transaction) {
            return Err(Box::new(ReplacementExecSubmitCoordinatorAdmissionFailure {
                ready,
                semantic,
            }));
        }
        self.submissions.insert(
            transaction,
            ReplacementCoordinatedExecSubmit::Ready { ready, semantic },
        );
        Ok(transaction)
    }

    pub fn submit(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementExecSubmitCoordinatorProgress> {
        let state = self.submissions.remove(&transaction)?;
        let result = match state {
            ReplacementCoordinatedExecSubmit::Ready { ready, semantic } => {
                runtime.submit_queue_ready_exec(*ready, semantic)
            }
            ReplacementCoordinatedExecSubmit::SubmitFailed(failure) => {
                runtime.retry_queue_ready_exec_submit(*failure)
            }
            unchanged => {
                self.submissions.insert(transaction, unchanged);
                return Some(ReplacementExecSubmitCoordinatorProgress::WrongStage);
            }
        };
        match result {
            Ok(pending) => {
                self.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::Pending(Box::new(pending)),
                );
                Some(ReplacementExecSubmitCoordinatorProgress::Pending)
            }
            Err(failure) => {
                self.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::SubmitFailed(Box::new(failure)),
                );
                Some(ReplacementExecSubmitCoordinatorProgress::FailedSubmit)
            }
        }
    }

    pub fn poll(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementExecSubmitCoordinatorProgress> {
        let state = self.submissions.remove(&transaction)?;
        let pending = match state {
            ReplacementCoordinatedExecSubmit::Pending(pending) => *pending,
            unchanged => {
                self.submissions.insert(transaction, unchanged);
                return Some(ReplacementExecSubmitCoordinatorProgress::WrongStage);
            }
        };
        let poll = runtime.poll_queue_exec_driver(pending);
        match poll {
            reims_vgpu_vulkan::replacement_exec_queue::ReplacementExecSubmitPoll::Pending(
                pending,
            ) => {
                self.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::Pending(Box::new(pending)),
                );
                Some(ReplacementExecSubmitCoordinatorProgress::Pending)
            }
            terminal @ reims_vgpu_vulkan::replacement_exec_queue::ReplacementExecSubmitPoll::Accepted(_) => {
                self.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::Terminal(Box::new(terminal)),
                );
                Some(ReplacementExecSubmitCoordinatorProgress::Accepted)
            }
            terminal @ reims_vgpu_vulkan::replacement_exec_queue::ReplacementExecSubmitPoll::DriverRefused { .. } => {
                self.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::Terminal(Box::new(terminal)),
                );
                Some(ReplacementExecSubmitCoordinatorProgress::DriverRefused)
            }
            terminal @ reims_vgpu_vulkan::replacement_exec_queue::ReplacementExecSubmitPoll::AcceptanceRefused(_) => {
                if let reims_vgpu_vulkan::replacement_exec_queue::ReplacementExecSubmitPoll::AcceptanceRefused(
                    failure,
                ) = &terminal
                {
                    report_acceptance_refusal(
                        "replacement_exec_submit",
                        transaction,
                        &failure.reason,
                    );
                }
                self.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::Terminal(Box::new(terminal)),
                );
                Some(ReplacementExecSubmitCoordinatorProgress::AcceptanceRefused)
            }
        }
    }

    pub fn take_terminal(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<crate::runtime::replacement_session::ReplacementExecSubmitPoll<Semantic>> {
        match self.submissions.remove(&transaction)? {
            ReplacementCoordinatedExecSubmit::Terminal(terminal) => Some(*terminal),
            unchanged => {
                self.submissions.insert(transaction, unchanged);
                None
            }
        }
    }

    pub fn live_submissions(&self) -> usize {
        self.submissions.len()
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.submissions.keys().copied().collect()
    }
}

pub(crate) enum ReplacementCoordinatedGuestUpload<Semantic> {
    Ready {
        ready: Box<crate::runtime::replacement_session::ReplacementQueueReadyGuestUpload<Semantic>>,
        semantic: Semantic,
    },
    ChainFailed(
        Box<
            crate::runtime::replacement_session::ReplacementGuestUploadChainPreparationFailure<
                Semantic,
            >,
        >,
    ),
    PreparationFailed(
        Box<
            crate::runtime::replacement_session::ReplacementRecordedGuestUploadPreparationFailure<
                Semantic,
            >,
        >,
    ),
    EnqueueFailed(
        Box<
            crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryEnqueueFailure<
                Semantic,
            >,
        >,
    ),
    Pending(
        Box<crate::runtime::replacement_session::PendingReplacementGuestUploadAuxiliary<Semantic>>,
    ),
    DriverRefused {
        reason: reims_vgpu_vulkan::replacement_queue::ReplacementQueueError,
        prepared: Box<
            crate::runtime::replacement_session::ReplacementPreparedRecordedGuestUpload<Semantic>,
        >,
    },
    AcceptanceRefused(
        Box<
            crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryAcceptanceFailure<
                Semantic,
            >,
        >,
    ),
    Accepted(
        Box<crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>>,
    ),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementGuestUploadCoordinatorProgress {
    Pending,
    Accepted,
    FailedPreparation,
    FailedEnqueue,
    DriverRefused,
    AcceptanceRefused,
    WrongStage,
}

impl ReplacementGuestUploadCoordinatorProgress {
    /// See [`ReplacementExecSubmitCoordinatorProgress::is_refusal`].
    const fn is_refusal(self) -> bool {
        match self {
            Self::Pending | Self::Accepted | Self::WrongStage => false,
            Self::FailedPreparation
            | Self::FailedEnqueue
            | Self::DriverRefused
            | Self::AcceptanceRefused => true,
        }
    }
}

impl<Semantic> ReplacementCoordinatedGuestUpload<Semantic> {
    /// The name of the state this owner is retained in.
    ///
    /// Only `Ready` and `Pending` are advanced by
    /// [`ReplacementGuestUploadCoordinator::progress`]; every other state is
    /// retained until a different caller claims it. An owner stuck in one of
    /// those holds its source domain's submission head, which every later
    /// transaction on that channel then refuses behind, so the census names
    /// the exact state rather than reporting only a live count.
    const fn stage(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "ready",
            Self::ChainFailed(_) => "chain_failed",
            Self::PreparationFailed(_) => "preparation_failed",
            Self::EnqueueFailed(_) => "enqueue_failed",
            Self::Pending(_) => "pending",
            Self::DriverRefused { .. } => "driver_refused",
            Self::AcceptanceRefused(_) => "acceptance_refused",
            Self::Accepted(_) => "accepted",
        }
    }
}

pub(crate) struct ReplacementGuestUploadCoordinator<Semantic> {
    uploads: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedGuestUpload<Semantic>,
    >,
}

impl<Semantic> Default for ReplacementGuestUploadCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            uploads: std::collections::BTreeMap::new(),
        }
    }
}

impl<Semantic: Clone> ReplacementGuestUploadCoordinator<Semantic> {
    pub fn admit_accepted(
        &mut self,
        accepted: Box<
            crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>,
        >,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        Box<crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>>,
    > {
        let transaction = accepted.transaction();
        if self.uploads.contains_key(&transaction) {
            return Err(accepted);
        }
        self.uploads.insert(
            transaction,
            ReplacementCoordinatedGuestUpload::Accepted(accepted),
        );
        Ok(transaction)
    }

    pub fn admit(
        &mut self,
        ready: Box<crate::runtime::replacement_session::ReplacementQueueReadyGuestUpload<Semantic>>,
        semantic: Semantic,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        (
            Box<crate::runtime::replacement_session::ReplacementQueueReadyGuestUpload<Semantic>>,
            Semantic,
        ),
    > {
        let transaction = ready.transaction();
        if self.uploads.contains_key(&transaction) {
            return Err((ready, semantic));
        }
        self.uploads.insert(
            transaction,
            ReplacementCoordinatedGuestUpload::Ready { ready, semantic },
        );
        Ok(transaction)
    }

    pub fn progress(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementGuestUploadCoordinatorProgress> {
        let state = self.uploads.remove(&transaction)?;
        match state {
            ReplacementCoordinatedGuestUpload::Ready { ready, semantic } => {
                let chain = match runtime.prepare_guest_upload_chain(*ready, semantic) {
                    Ok(chain) => chain,
                    Err(failure) => {
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::ChainFailed(failure),
                        );
                        return Some(ReplacementGuestUploadCoordinatorProgress::FailedPreparation);
                    }
                };
                let prepared = match runtime.prepare_recorded_guest_upload(chain) {
                    Ok(prepared) => prepared,
                    Err(failure) => {
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::PreparationFailed(failure),
                        );
                        return Some(ReplacementGuestUploadCoordinatorProgress::FailedPreparation);
                    }
                };
                match runtime.enqueue_recorded_guest_upload(prepared) {
                    Ok(pending) => {
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::Pending(Box::new(pending)),
                        );
                        Some(ReplacementGuestUploadCoordinatorProgress::Pending)
                    }
                    Err(failure) => {
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::EnqueueFailed(failure),
                        );
                        Some(ReplacementGuestUploadCoordinatorProgress::FailedEnqueue)
                    }
                }
            }
            ReplacementCoordinatedGuestUpload::Pending(pending) => {
                match runtime.progress_guest_upload_auxiliary(*pending) {
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::Pending(
                        pending,
                    ) => {
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::Pending(pending),
                        );
                        Some(ReplacementGuestUploadCoordinatorProgress::Pending)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::Accepted(
                        accepted,
                    ) => {
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::Accepted(accepted),
                        );
                        Some(ReplacementGuestUploadCoordinatorProgress::Accepted)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::DriverRefused {
                        reason,
                        prepared,
                    } => {
                        report_retained_failure_detail(
                            "replacement_guest_upload_driver",
                            transaction,
                            &format!("{reason:?}"),
                        );
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::DriverRefused { reason, prepared },
                        );
                        Some(ReplacementGuestUploadCoordinatorProgress::DriverRefused)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::AcceptanceRefused(
                        failure,
                    ) => {
                        report_acceptance_refusal(
                            "replacement_guest_upload",
                            transaction,
                            &failure.failure.reason,
                        );
                        self.uploads.insert(
                            transaction,
                            ReplacementCoordinatedGuestUpload::AcceptanceRefused(failure),
                        );
                        Some(ReplacementGuestUploadCoordinatorProgress::AcceptanceRefused)
                    }
                }
            }
            unchanged => {
                self.uploads.insert(transaction, unchanged);
                Some(ReplacementGuestUploadCoordinatorProgress::WrongStage)
            }
        }
    }

    pub fn take_accepted(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<
        Box<crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>>,
    > {
        match self.uploads.remove(&transaction)? {
            ReplacementCoordinatedGuestUpload::Accepted(accepted) => Some(accepted),
            unchanged => {
                self.uploads.insert(transaction, unchanged);
                None
            }
        }
    }

    pub fn accepted_point(
        &self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<reims_vgpu_core::QueueTimelinePoint> {
        match self.uploads.get(&transaction)? {
            ReplacementCoordinatedGuestUpload::Accepted(accepted) => Some(accepted.point()),
            _ => None,
        }
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.uploads.keys().copied().collect()
    }

    pub fn live_uploads(&self) -> usize {
        self.uploads.len()
    }

    /// See [`ReplacementCoordinatedGuestUpload::stage`].
    pub fn stages(&self) -> Vec<(reims_vgpu_protocol::TransactionId, &'static str)> {
        self.uploads
            .iter()
            .map(|(&transaction, upload)| (transaction, upload.stage()))
            .collect()
    }

    pub fn has_accepted_at_or_before(
        &self,
        queue: reims_vgpu_protocol::QueueOwnerId,
        completed: reims_vgpu_protocol::QueueTimelineValue,
    ) -> bool {
        self.uploads.values().any(|state| {
            matches!(
                state,
                ReplacementCoordinatedGuestUpload::Accepted(accepted)
                    if accepted.point().queue == queue && accepted.point().value <= completed
            )
        })
    }
}

pub(crate) enum ReplacementCoordinatedGuestUploadSuffix<Semantic> {
    Continuing(
        Box<crate::runtime::replacement_session::ReplacementContinuingGuestUpload<Semantic>>,
    ),
    PreparationFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadSuffixPreparationFailure<Semantic>>,
    ),
    ResolutionFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadSuffixResolutionFailure<Semantic>>,
    ),
    DispatchFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadSuffixDispatchFailure<Semantic>>,
    ),
    PendingRecording(
        Box<crate::runtime::replacement_session::PendingReplacementGuestUploadSuffixRecording<Semantic>>,
    ),
    PendingRefreshRecording(
        Box<crate::runtime::replacement_session::PendingReplacementGuestUploadRecording<Semantic>>,
    ),
    RefreshResolutionFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadRecordingResolutionFailure<Semantic>>,
    ),
    RefreshDispatchFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadRecordingDispatchFailure<Semantic>>,
    ),
    RefreshPreparationFailed(
        Box<crate::runtime::replacement_session::ReplacementRecordedGuestUploadPreparationFailure<Semantic>>,
    ),
    RefreshEnqueueFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryEnqueueFailure<Semantic>>,
    ),
    PendingRefresh(
        Box<crate::runtime::replacement_session::PendingReplacementGuestUploadAuxiliary<Semantic>>,
    ),
    RefreshDriverRefused {
        reason: reims_vgpu_vulkan::replacement_queue::ReplacementQueueError,
        prepared: Box<crate::runtime::replacement_session::ReplacementPreparedRecordedGuestUpload<Semantic>>,
    },
    RefreshAcceptanceRefused(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryAcceptanceFailure<Semantic>>,
    ),
    RefreshAccepted(
        Box<crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>>,
    ),
    FinalPreparationFailed(
        Box<crate::runtime::replacement_session::ReplacementGuestUploadFinalPreparationFailure<Semantic>>,
    ),
    FinalEnqueueFailed(
        Box<crate::runtime::replacement_session::ReplacementIndirectFinalEnqueueFailure<Semantic>>,
    ),
    PendingFinal(
        Box<crate::runtime::replacement_session::PendingReplacementIndirectFinalSubmit<Semantic>>,
    ),
    DriverRefused {
        reason: reims_vgpu_vulkan::replacement_queue::ReplacementQueueError,
        recorded: Box<crate::runtime::replacement_session::PreparedReplacementRecordedIndirectFinal<Semantic>>,
    },
    AcceptanceRefused(
        Box<
            reims_vgpu_vulkan::replacement_indirect_exec_chain::ReplacementIndirectFinalAcceptanceFailure<
                Semantic,
                reims_vgpu_core::ResolvedComputeDispatch,
                reims_vgpu_vulkan::replacement_compute::ReplacementComputePipelineVariant,
                reims_vgpu_core::ResolvedRenderDispatch,
                reims_vgpu_vulkan::replacement_render::ReplacementRenderPipelineVariant,
            >,
        >,
    ),
    Accepted(Box<crate::runtime::replacement_session::AcceptedReplacementIndirectFinal>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementGuestUploadSuffixProgress {
    WaitingForProducer,
    PendingRecording,
    PendingFinal,
    Accepted,
    FailedPreparation,
    FailedRecording,
    FailedEnqueue,
    DriverRefused,
    AcceptanceRefused,
    WrongStage,
}

impl ReplacementGuestUploadSuffixProgress {
    /// See [`ReplacementExecSubmitCoordinatorProgress::is_refusal`].
    const fn is_refusal(self) -> bool {
        match self {
            Self::WaitingForProducer
            | Self::PendingRecording
            | Self::PendingFinal
            | Self::Accepted
            | Self::WrongStage => false,
            Self::FailedPreparation
            | Self::FailedRecording
            | Self::FailedEnqueue
            | Self::DriverRefused
            | Self::AcceptanceRefused => true,
        }
    }
}

pub(crate) struct ReplacementGuestUploadSuffixCoordinator<Semantic> {
    suffixes: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedGuestUploadSuffix<Semantic>,
    >,
}

impl<Semantic> Default for ReplacementGuestUploadSuffixCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            suffixes: std::collections::BTreeMap::new(),
        }
    }
}

impl<Semantic: Clone + PartialEq + Send + 'static>
    ReplacementGuestUploadSuffixCoordinator<Semantic>
{
    pub fn admit(
        &mut self,
        continuing: crate::runtime::replacement_session::ReplacementContinuingGuestUpload<Semantic>,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        Box<crate::runtime::replacement_session::ReplacementContinuingGuestUpload<Semantic>>,
    > {
        let transaction = continuing.transaction();
        if self.suffixes.contains_key(&transaction) {
            return Err(Box::new(continuing));
        }
        self.suffixes.insert(
            transaction,
            ReplacementCoordinatedGuestUploadSuffix::Continuing(Box::new(continuing)),
        );
        Ok(transaction)
    }

    pub fn progress(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementGuestUploadSuffixProgress> {
        let state = self.suffixes.remove(&transaction)?;
        match state {
            ReplacementCoordinatedGuestUploadSuffix::Continuing(continuing) => {
                let continuation = match runtime.prepare_guest_upload_suffix(*continuing) {
                    Ok(continuation) => continuation,
                    Err(failure) => {
                        match failure.into_content_producer_retry() {
                            Ok(continuing) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::Continuing(Box::new(
                                        continuing,
                                    )),
                                );
                                return Some(
                                    ReplacementGuestUploadSuffixProgress::WaitingForProducer,
                                );
                            }
                        Err(failure) => {
                        report_retained_failure_detail(
                            "replacement_guest_upload_suffix_preparation",
                            transaction,
                            &failure.detail(),
                        );
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::PreparationFailed(failure),
                        );
                            return Some(ReplacementGuestUploadSuffixProgress::FailedPreparation);
                        }
                    }
                    }
                };
                let crate::runtime::replacement_session::ReplacementPreparedGuestUploadContinuation::Suffix(suffix) = continuation else {
                    let crate::runtime::replacement_session::ReplacementPreparedGuestUploadContinuation::Refresh(refresh) = continuation else {
                        unreachable!()
                    };
                    let resolved = match runtime.resolve_guest_upload_phase_recording(*refresh) {
                        Ok(resolved) => resolved,
                        Err(failure) => {
                            report_retained_failure_detail(
                                "replacement_guest_upload_refresh_resolution",
                                transaction,
                                &failure.detail(),
                            );
                            self.suffixes.insert(
                                transaction,
                                ReplacementCoordinatedGuestUploadSuffix::RefreshResolutionFailed(
                                    failure,
                                ),
                            );
                            return Some(ReplacementGuestUploadSuffixProgress::FailedPreparation);
                        }
                    };
                    return match runtime.dispatch_guest_upload_phase_recording(resolved) {
                        Ok(pending) => {
                            self.suffixes.insert(
                                transaction,
                                ReplacementCoordinatedGuestUploadSuffix::PendingRefreshRecording(
                                    Box::new(pending),
                                ),
                            );
                            Some(ReplacementGuestUploadSuffixProgress::PendingRecording)
                        }
                        Err(failure) => {
                            self.suffixes.insert(
                                transaction,
                                ReplacementCoordinatedGuestUploadSuffix::RefreshDispatchFailed(
                                    failure,
                                ),
                            );
                            Some(ReplacementGuestUploadSuffixProgress::FailedRecording)
                        }
                    };
                };
                let resolved = match runtime.resolve_guest_upload_suffix_recording(*suffix) {
                    Ok(resolved) => resolved,
                    Err(failure) => {
                        report_retained_failure_detail(
                            "replacement_guest_upload_suffix_resolution",
                            transaction,
                            &failure.detail(),
                        );
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::ResolutionFailed(failure),
                        );
                        return Some(ReplacementGuestUploadSuffixProgress::FailedPreparation);
                    }
                };
                match runtime.dispatch_guest_upload_suffix_recording(resolved) {
                    Ok(pending) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::PendingRecording(Box::new(
                                pending,
                            )),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::PendingRecording)
                    }
                    Err(failure) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::DispatchFailed(failure),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::FailedRecording)
                    }
                }
            }
            ReplacementCoordinatedGuestUploadSuffix::PendingRefreshRecording(pending) => {
                match (*pending).try_complete() {
                    crate::runtime::replacement_session::ReplacementGuestUploadRecordingPoll::Pending(
                        pending,
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::PendingRefreshRecording(
                                Box::new(pending),
                            ),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::PendingRecording)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadRecordingPoll::Completed(
                        Err(failure),
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::RefreshDispatchFailed(failure),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::FailedRecording)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadRecordingPoll::Completed(
                        Ok(recorded),
                    ) => {
                        let prepared = match runtime.prepare_recorded_guest_upload_refresh(recorded) {
                            Ok(prepared) => prepared,
                            Err(failure) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::RefreshPreparationFailed(
                                        failure,
                                    ),
                                );
                                return Some(
                                    ReplacementGuestUploadSuffixProgress::FailedPreparation,
                                );
                            }
                        };
                        match runtime.enqueue_recorded_guest_upload(prepared) {
                            Ok(pending) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::PendingRefresh(
                                        Box::new(pending),
                                    ),
                                );
                                Some(ReplacementGuestUploadSuffixProgress::PendingFinal)
                            }
                            Err(failure) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::RefreshEnqueueFailed(
                                        failure,
                                    ),
                                );
                                Some(ReplacementGuestUploadSuffixProgress::FailedEnqueue)
                            }
                        }
                    }
                }
            }
            ReplacementCoordinatedGuestUploadSuffix::PendingRefresh(pending) => {
                match runtime.progress_guest_upload_auxiliary(*pending) {
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::Pending(
                        pending,
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::PendingRefresh(pending),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::PendingFinal)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::Accepted(
                        accepted,
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::RefreshAccepted(accepted),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::Accepted)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::DriverRefused {
                        reason,
                        prepared,
                    } => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::RefreshDriverRefused {
                                reason,
                                prepared,
                            },
                        );
                        Some(ReplacementGuestUploadSuffixProgress::DriverRefused)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadAuxiliaryProgress::AcceptanceRefused(
                        failure,
                    ) => {
                        report_acceptance_refusal(
                            "replacement_guest_upload_refresh",
                            transaction,
                            &failure.failure.reason,
                        );
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::RefreshAcceptanceRefused(
                                failure,
                            ),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::AcceptanceRefused)
                    }
                }
            }
            ReplacementCoordinatedGuestUploadSuffix::PendingRecording(pending) => {
                match (*pending).try_complete() {
                    crate::runtime::replacement_session::ReplacementGuestUploadSuffixRecordingPoll::Pending(
                        pending,
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::PendingRecording(Box::new(
                                pending,
                            )),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::PendingRecording)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadSuffixRecordingPoll::Completed(
                        Err(failure),
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::DispatchFailed(failure),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::FailedRecording)
                    }
                    crate::runtime::replacement_session::ReplacementGuestUploadSuffixRecordingPoll::Completed(
                        Ok(recorded),
                    ) => {
                        let prepared = match runtime.prepare_recorded_guest_upload_final(recorded) {
                            Ok(prepared) => prepared,
                            Err(failure) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::FinalPreparationFailed(
                                        failure,
                                    ),
                                );
                                return Some(
                                    ReplacementGuestUploadSuffixProgress::FailedPreparation,
                                );
                            }
                        };
                        match runtime.submit_chained_final(prepared) {
                            Ok(pending) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::PendingFinal(
                                        Box::new(pending),
                                    ),
                                );
                                Some(ReplacementGuestUploadSuffixProgress::PendingFinal)
                            }
                            Err(failure) => {
                                self.suffixes.insert(
                                    transaction,
                                    ReplacementCoordinatedGuestUploadSuffix::FinalEnqueueFailed(
                                        failure,
                                    ),
                                );
                                Some(ReplacementGuestUploadSuffixProgress::FailedEnqueue)
                            }
                        }
                    }
                }
            }
            ReplacementCoordinatedGuestUploadSuffix::PendingFinal(pending) => {
                match runtime.progress_chained_final(*pending) {
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::Pending(
                        pending,
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::PendingFinal(pending),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::PendingFinal)
                    }
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::Accepted(
                        accepted,
                    ) => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::Accepted(accepted),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::Accepted)
                    }
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::DriverRefused {
                        reason,
                        recorded,
                    } => {
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::DriverRefused {
                                reason,
                                recorded,
                            },
                        );
                        Some(ReplacementGuestUploadSuffixProgress::DriverRefused)
                    }
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::AcceptanceRefused(
                        failure,
                    ) => {
                        // The point this submission was allocated, beside the
                        // high-water mark its queue has reached. The gap says
                        // how far ahead the queue had run when this acceptance
                        // came back, which is what a refusal naming a point
                        // cannot say on its own.
                        let point = failure.failure.submission.prepared.point();
                        report_retained_failure_detail(
                            "replacement_guest_upload_final",
                            transaction,
                            &format!(
                                "{:?} queue={} point={} queue_submitted={:?}",
                                failure.failure.reason,
                                point.queue.get(),
                                point.value.get(),
                                runtime.last_submitted_point(point.queue).map(|value| value.get()),
                            ),
                        );
                        self.suffixes.insert(
                            transaction,
                            ReplacementCoordinatedGuestUploadSuffix::AcceptanceRefused(failure),
                        );
                        Some(ReplacementGuestUploadSuffixProgress::AcceptanceRefused)
                    }
                }
            }
            unchanged => {
                self.suffixes.insert(transaction, unchanged);
                Some(ReplacementGuestUploadSuffixProgress::WrongStage)
            }
        }
    }

    /// Take one suffix retained in a terminal refusal, so its caller can give
    /// it up.
    ///
    /// Only `ResolutionFailed` is offered here, because it is the one terminal
    /// state whose owner holds exactly what abandonment releases -- a prepared
    /// native chain and a prepared resource envelope, both still cancellable.
    /// The other terminal states of this coordinator hold a native submission
    /// the driver has already seen, and giving one of those up is a different
    /// release that this does not pretend to perform.
    pub fn take_refused_resolution(
        &mut self,
    ) -> Option<(
        reims_vgpu_protocol::TransactionId,
        Box<
            crate::runtime::replacement_session::ReplacementGuestUploadSuffixResolutionFailure<
                Semantic,
            >,
        >,
    )> {
        let transaction = self.suffixes.iter().find_map(|(&transaction, suffix)| {
            matches!(
                suffix,
                ReplacementCoordinatedGuestUploadSuffix::ResolutionFailed(_)
            )
            .then_some(transaction)
        })?;
        let Some(ReplacementCoordinatedGuestUploadSuffix::ResolutionFailed(failure)) =
            self.suffixes.remove(&transaction)
        else {
            unreachable!("the refused resolution was just located by its state")
        };
        Some((transaction, failure))
    }

    pub fn take_accepted(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<Box<crate::runtime::replacement_session::AcceptedReplacementIndirectFinal>> {
        match self.suffixes.remove(&transaction)? {
            ReplacementCoordinatedGuestUploadSuffix::Accepted(accepted) => Some(accepted),
            unchanged => {
                self.suffixes.insert(transaction, unchanged);
                None
            }
        }
    }

    pub fn take_refresh_accepted(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<
        Box<crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>>,
    > {
        match self.suffixes.remove(&transaction)? {
            ReplacementCoordinatedGuestUploadSuffix::RefreshAccepted(accepted) => Some(accepted),
            unchanged => {
                self.suffixes.insert(transaction, unchanged);
                None
            }
        }
    }

    pub fn restore_refresh_accepted(
        &mut self,
        accepted: Box<
            crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>,
        >,
    ) -> Result<
        (),
        Box<crate::runtime::replacement_session::ReplacementAcceptedGuestUploadAuxiliary<Semantic>>,
    > {
        let transaction = accepted.transaction();
        if self.suffixes.contains_key(&transaction) {
            return Err(accepted);
        }
        self.suffixes.insert(
            transaction,
            ReplacementCoordinatedGuestUploadSuffix::RefreshAccepted(accepted),
        );
        Ok(())
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.suffixes.keys().copied().collect()
    }

    pub fn live_suffixes(&self) -> usize {
        self.suffixes.len()
    }
}

pub(crate) enum ReplacementIndirectCoordinatorFailure<Semantic> {
    Initial(crate::runtime::replacement_session::ReplacementInitialIndirectRangeSubmitFailure<Semantic>),
    Continuing(crate::runtime::replacement_session::ReplacementContinuingIndirectRangeDispatchFailure<Semantic>),
    ContinuingRecording(crate::runtime::replacement_session::ReplacementContinuingIndirectRangeProgressFailure<Semantic>),
    FinalEnqueue(Box<crate::runtime::replacement_session::ReplacementIndirectFinalEnqueueFailure<Semantic>>),
    AuxiliaryDriverRefused {
        reason: reims_vgpu_vulkan::replacement_queue::ReplacementQueueError,
        prepared: Box<crate::runtime::replacement_session::ReplacementPreparedRecordedIndirectAuxiliary<Semantic>>,
    },
    AuxiliaryAcceptance(
        Box<crate::runtime::replacement_session::ReplacementIndirectAuxiliaryAcceptanceFailure<Semantic>>,
    ),
    FinalDriverRefused {
        reason: reims_vgpu_vulkan::replacement_queue::ReplacementQueueError,
        recorded: Box<crate::runtime::replacement_session::PreparedReplacementRecordedIndirectFinal<Semantic>>,
    },
    FinalAcceptance(
        Box<
            reims_vgpu_vulkan::replacement_indirect_exec_chain::ReplacementIndirectFinalAcceptanceFailure<
                Semantic,
                reims_vgpu_core::ResolvedComputeDispatch,
                reims_vgpu_vulkan::replacement_compute::ReplacementComputePipelineVariant,
                reims_vgpu_core::ResolvedRenderDispatch,
                reims_vgpu_vulkan::replacement_render::ReplacementRenderPipelineVariant,
            >,
        >,
    ),
}

pub(crate) enum ReplacementCoordinatedIndirect<Semantic> {
    InitialReady {
        ready:
            Box<crate::runtime::replacement_session::ReplacementQueueReadyIndirectRange<Semantic>>,
        semantic: Semantic,
    },
    Continuing(
        Box<crate::runtime::replacement_session::ReplacementContinuingIndirectRangeChain<Semantic>>,
    ),
    PendingContinuing(
        Box<
            crate::runtime::replacement_session::PendingReplacementContinuingIndirectRangeRecording<
                Semantic,
            >,
        >,
    ),
    PendingAuxiliary(
        Box<
            crate::runtime::replacement_session::PendingReplacementRecordedIndirectAuxiliary<
                Semantic,
            >,
        >,
    ),
    AcceptedAuxiliary(
        Box<crate::runtime::replacement_session::ReplacementAcceptedIndirectAuxiliary<Semantic>>,
    ),
    PendingFinal(
        Box<crate::runtime::replacement_session::PendingReplacementIndirectFinalSubmit<Semantic>>,
    ),
    AcceptedFinal(Box<crate::runtime::replacement_session::AcceptedReplacementIndirectFinal>),
    Failed(Box<ReplacementIndirectCoordinatorFailure<Semantic>>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementIndirectCoordinatorProgress {
    PendingRecording,
    PendingAuxiliary,
    AuxiliaryAccepted,
    PendingFinal,
    FinalAccepted,
    Failed,
    WrongStage,
}

impl ReplacementIndirectCoordinatorProgress {
    /// See [`ReplacementExecSubmitCoordinatorProgress::is_refusal`].
    const fn is_refusal(self) -> bool {
        match self {
            Self::PendingRecording
            | Self::PendingAuxiliary
            | Self::AuxiliaryAccepted
            | Self::PendingFinal
            | Self::FinalAccepted
            | Self::WrongStage => false,
            Self::Failed => true,
        }
    }
}

pub(crate) struct ReplacementIndirectCoordinator<Semantic> {
    ranges: std::collections::BTreeMap<
        reims_vgpu_protocol::TransactionId,
        ReplacementCoordinatedIndirect<Semantic>,
    >,
}

impl<Semantic> Default for ReplacementIndirectCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            ranges: std::collections::BTreeMap::new(),
        }
    }
}

impl<Semantic: Clone + PartialEq + Send + 'static> ReplacementIndirectCoordinator<Semantic> {
    pub fn admit_initial(
        &mut self,
        ready: Box<
            crate::runtime::replacement_session::ReplacementQueueReadyIndirectRange<Semantic>,
        >,
        semantic: Semantic,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        (
            Box<crate::runtime::replacement_session::ReplacementQueueReadyIndirectRange<Semantic>>,
            Semantic,
        ),
    > {
        let transaction = ready.transaction();
        if self.ranges.contains_key(&transaction) {
            return Err((ready, semantic));
        }
        self.ranges.insert(
            transaction,
            ReplacementCoordinatedIndirect::InitialReady { ready, semantic },
        );
        Ok(transaction)
    }

    pub fn admit_continuing(
        &mut self,
        continuing: crate::runtime::replacement_session::ReplacementContinuingIndirectRangeChain<
            Semantic,
        >,
    ) -> Result<
        reims_vgpu_protocol::TransactionId,
        Box<crate::runtime::replacement_session::ReplacementContinuingIndirectRangeChain<Semantic>>,
    > {
        let transaction = continuing.transaction();
        if self.ranges.contains_key(&transaction) {
            return Err(Box::new(continuing));
        }
        self.ranges.insert(
            transaction,
            ReplacementCoordinatedIndirect::Continuing(Box::new(continuing)),
        );
        Ok(transaction)
    }

    pub fn progress(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<ReplacementIndirectCoordinatorProgress> {
        let state = self.ranges.remove(&transaction)?;
        match state {
            ReplacementCoordinatedIndirect::InitialReady { ready, semantic } => {
                match runtime.submit_initial_indirect_range(*ready, semantic) {
                    Ok(pending) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::PendingAuxiliary(Box::new(pending)),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::PendingAuxiliary)
                    }
                    Err(failure) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::Initial(failure),
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                }
            }
            ReplacementCoordinatedIndirect::Continuing(continuing) => {
                match runtime.dispatch_continuing_indirect_range_phase(*continuing) {
                    Ok(pending) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::PendingContinuing(Box::new(pending)),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::PendingRecording)
                    }
                    Err(failure) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::Continuing(failure),
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                }
            }
            ReplacementCoordinatedIndirect::PendingContinuing(pending) => {
                match runtime.progress_continuing_indirect_range_recording(*pending) {
                    Ok(
                        crate::runtime::replacement_session::ReplacementContinuingIndirectRangeProgress::Pending(
                            pending,
                        ),
                    ) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::PendingContinuing(pending),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::PendingRecording)
                    }
                    Ok(
                        crate::runtime::replacement_session::ReplacementContinuingIndirectRangeProgress::Auxiliary(
                            pending,
                        ),
                    ) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::PendingAuxiliary(pending),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::PendingAuxiliary)
                    }
                    Ok(
                        crate::runtime::replacement_session::ReplacementContinuingIndirectRangeProgress::Final(
                            prepared,
                        ),
                    ) => match runtime.submit_indirect_final(*prepared) {
                        Ok(pending) => {
                            self.ranges.insert(
                                transaction,
                                ReplacementCoordinatedIndirect::PendingFinal(Box::new(pending)),
                            );
                            Some(ReplacementIndirectCoordinatorProgress::PendingFinal)
                        }
                        Err(failure) => {
                            self.ranges.insert(
                                transaction,
                                ReplacementCoordinatedIndirect::Failed(Box::new(
                                    ReplacementIndirectCoordinatorFailure::FinalEnqueue(failure),
                                )),
                            );
                            Some(ReplacementIndirectCoordinatorProgress::Failed)
                        }
                    },
                    Err(failure) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::ContinuingRecording(
                                    failure,
                                ),
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                }
            }
            ReplacementCoordinatedIndirect::PendingAuxiliary(pending) => {
                match runtime.progress_indirect_auxiliary(*pending) {
                    crate::runtime::replacement_session::ReplacementIndirectAuxiliaryProgress::Pending(
                        pending,
                    ) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::PendingAuxiliary(pending),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::PendingAuxiliary)
                    }
                    crate::runtime::replacement_session::ReplacementIndirectAuxiliaryProgress::Accepted(
                        accepted,
                    ) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::AcceptedAuxiliary(accepted),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::AuxiliaryAccepted)
                    }
                    crate::runtime::replacement_session::ReplacementIndirectAuxiliaryProgress::DriverRefused {
                        reason,
                        prepared,
                    } => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::AuxiliaryDriverRefused {
                                    reason,
                                    prepared,
                                },
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                    crate::runtime::replacement_session::ReplacementIndirectAuxiliaryProgress::AcceptanceRefused(
                        failure,
                    ) => {
                        report_acceptance_refusal(
                            "replacement_indirect_auxiliary",
                            transaction,
                            &failure.failure.reason,
                        );
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::AuxiliaryAcceptance(
                                    failure,
                                ),
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                }
            }
            ReplacementCoordinatedIndirect::PendingFinal(pending) => {
                match runtime.progress_indirect_final(*pending) {
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::Pending(
                        pending,
                    ) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::PendingFinal(pending),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::PendingFinal)
                    }
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::Accepted(
                        accepted,
                    ) => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::AcceptedFinal(accepted),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::FinalAccepted)
                    }
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::DriverRefused {
                        reason,
                        recorded,
                    } => {
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::FinalDriverRefused {
                                    reason,
                                    recorded,
                                },
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                    crate::runtime::replacement_session::ReplacementChainedFinalProgress::AcceptanceRefused(
                        failure,
                    ) => {
                        report_acceptance_refusal(
                            "replacement_indirect_final",
                            transaction,
                            &failure.failure.reason,
                        );
                        self.ranges.insert(
                            transaction,
                            ReplacementCoordinatedIndirect::Failed(Box::new(
                                ReplacementIndirectCoordinatorFailure::FinalAcceptance(failure),
                            )),
                        );
                        Some(ReplacementIndirectCoordinatorProgress::Failed)
                    }
                }
            }
            unchanged => {
                self.ranges.insert(transaction, unchanged);
                Some(ReplacementIndirectCoordinatorProgress::WrongStage)
            }
        }
    }

    pub fn accepted_auxiliary_point(
        &self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<reims_vgpu_core::QueueTimelinePoint> {
        match self.ranges.get(&transaction)? {
            ReplacementCoordinatedIndirect::AcceptedAuxiliary(accepted) => Some(accepted.point()),
            _ => None,
        }
    }

    pub fn take_accepted_auxiliary(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<
        Box<crate::runtime::replacement_session::ReplacementAcceptedIndirectAuxiliary<Semantic>>,
    > {
        match self.ranges.remove(&transaction)? {
            ReplacementCoordinatedIndirect::AcceptedAuxiliary(accepted) => Some(accepted),
            unchanged => {
                self.ranges.insert(transaction, unchanged);
                None
            }
        }
    }

    pub fn take_accepted_final(
        &mut self,
        transaction: reims_vgpu_protocol::TransactionId,
    ) -> Option<Box<crate::runtime::replacement_session::AcceptedReplacementIndirectFinal>> {
        match self.ranges.remove(&transaction)? {
            ReplacementCoordinatedIndirect::AcceptedFinal(accepted) => Some(accepted),
            unchanged => {
                self.ranges.insert(transaction, unchanged);
                None
            }
        }
    }

    pub fn transaction_ids(&self) -> Vec<reims_vgpu_protocol::TransactionId> {
        self.ranges.keys().copied().collect()
    }

    pub fn live_ranges(&self) -> usize {
        self.ranges.len()
    }

    pub fn has_accepted_auxiliary_at_or_before(
        &self,
        queue: reims_vgpu_protocol::QueueOwnerId,
        completed: reims_vgpu_protocol::QueueTimelineValue,
    ) -> bool {
        self.ranges.values().any(|state| {
            matches!(
                state,
                ReplacementCoordinatedIndirect::AcceptedAuxiliary(accepted)
                    if accepted.point().queue == queue && accepted.point().value <= completed
            )
        })
    }
}

type ReplacementTimelineProgressOwner<Semantic> =
    reims_vgpu_vulkan::replacement_replay::ReplacementReplayProgress<
        Semantic,
        reims_vgpu_vulkan::replacement_representation::ReplacementNativeRepresentation,
    >;

pub(crate) struct ReplacementPendingTimelineSemanticCompletion<Semantic> {
    progress: ReplacementTimelineProgressOwner<Semantic>,
    completions: std::collections::VecDeque<reims_vgpu_core::CompletionFact<Semantic>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplacementTimelineCoordinatorProgress {
    pub observed: usize,
    pub observation_failures: usize,
    pub semantic_failures: usize,
    pub published: usize,
    pub retired_batches: usize,
}

/// Fold one formatted refusal reason into the `u64` the always-on failure path
/// dedupes by. Two distinct reasons must not share a key or the second is never
/// printed, so this is a hash of the whole reason and not of its variant alone.
fn fnv_discriminant(reason: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in reason.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Name one queue observation this device could not apply.
///
/// The observation is retained for retry and is invisible to every live count,
/// so a refusal that repeats forever reads as an idle device rather than as
/// lost guest work.
fn report_timeline_observation_refusal<Semantic: Clone>(
    failure: &crate::runtime::replacement_session::ReplacementTimelineRetirementFailure,
    runtime: &ReplacementRuntimeSession<Semantic>,
) {
    let reason = format!("{:?}", failure.reason);
    let mut fields = vec![("reason", reason.clone())];
    // A completion refused for a representation is a lifetime question, so the
    // refusal is only actionable beside the state of every representation on
    // the backing it names.
    if let reims_vgpu_vulkan::replacement_replay::ReplacementReplayObservationError::ResourceCompletions(
        reims_vgpu_core::ResourceCompletionBatchError::Duplicate(completion)
        | reims_vgpu_core::ResourceCompletionBatchError::Completion { completion, .. },
    ) = failure.reason
    {
        let backing = completion.backing();
        fields.push((
            "census",
            runtime
                .representation_census(backing)
                .into_iter()
                .map(|entry| {
                    format!(
                        "{}:native={},retiring={},accepted={},last={}",
                        entry.representation.get(),
                        u8::from(entry.has_native),
                        u8::from(entry.retiring),
                        entry.accepted_uses,
                        entry.last_uses,
                    )
                })
                .collect::<Vec<_>>()
                .join(" "),
        ));
    }
    let diagnostic = ReplacementCoordinatorDiagnostic {
        slug: "replacement_timeline_observation_refused",
        fields,
        discriminant: fnv_discriminant(&reason),
    };
    crate::observe::Emit::decline("replacement_timeline_observation", &diagnostic)
        .fail_once(diagnostic.discriminant);
}

/// Name one GPU completion the semantic runtime refused. See
/// [`report_timeline_observation_refusal`] for why it is reported rather than
/// only retained.
fn report_timeline_completion_refusal<Semantic: Clone>(
    failure: &crate::runtime::replacement_session::ReplacementExecutionCompletionError<Semantic>,
    runtime: &ReplacementRuntimeSession<Semantic>,
) {
    let (stage, transaction, reason) = match failure {
        crate::runtime::replacement_session::ReplacementExecutionCompletionError::UnknownGeneration {
            generation,
            fact,
        } => (
            "unknown_generation",
            fact.transaction,
            format!("{generation:?}"),
        ),
        crate::runtime::replacement_session::ReplacementExecutionCompletionError::Commit(
            failure,
        ) => ("commit", failure.fact.transaction, format!("{:?}", failure.reason)),
    };
    let state = runtime
        .transaction_state_diagnostics()
        .into_iter()
        .find_map(|(candidate, state)| (candidate == transaction.get()).then_some(state))
        .unwrap_or_else(|| "absent".to_string());
    let discriminant = format!("{transaction:?}:{reason}");
    let diagnostic = ReplacementCoordinatorDiagnostic {
        slug: "replacement_timeline_completion_refused",
        fields: vec![
            ("stage", stage.to_string()),
            ("transaction", transaction.get().to_string()),
            ("reason", reason),
            ("state", state),
        ],
        discriminant: fnv_discriminant(&discriminant),
    };
    crate::observe::Emit::decline("replacement_timeline_completion", &diagnostic)
        .fail_once(diagnostic.discriminant);
}

pub(crate) struct ReplacementTimelineCoordinator<Semantic> {
    observation_failures: std::collections::VecDeque<
        crate::runtime::replacement_session::ReplacementTimelineRetirementFailure,
    >,
    semantic_failures:
        std::collections::VecDeque<ReplacementPendingTimelineSemanticCompletion<Semantic>>,
    published: std::collections::VecDeque<reims_vgpu_core::PublishedFact<Semantic>>,
    retired: std::collections::VecDeque<ReplacementTimelineProgressOwner<Semantic>>,
}

fn attempt_each_once<Fact, Publication, Failure>(
    mut facts: std::collections::VecDeque<Fact>,
    mut attempt: impl FnMut(Fact) -> Result<Vec<Publication>, (Failure, Fact)>,
) -> (
    std::collections::VecDeque<Fact>,
    Vec<Publication>,
    Vec<Failure>,
) {
    let attempts = facts.len();
    let mut unresolved = std::collections::VecDeque::new();
    let mut publications = Vec::new();
    let mut failures = Vec::new();
    for _ in 0..attempts {
        let fact = facts
            .pop_front()
            .expect("the attempt count came from this fact queue");
        match attempt(fact) {
            Ok(published) => publications.extend(published),
            Err((failure, fact)) => {
                failures.push(failure);
                unresolved.push_back(fact);
            }
        }
    }
    (unresolved, publications, failures)
}

impl<Semantic> Default for ReplacementTimelineCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            observation_failures: std::collections::VecDeque::new(),
            semantic_failures: std::collections::VecDeque::new(),
            published: std::collections::VecDeque::new(),
            retired: std::collections::VecDeque::new(),
        }
    }
}

impl<Semantic: Clone> ReplacementTimelineCoordinator<Semantic> {
    pub fn poll(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
    ) -> ReplacementTimelineCoordinatorProgress {
        let mut result = ReplacementTimelineCoordinatorProgress::default();
        for observation in runtime.try_retire_replacement_timelines() {
            result.observed += 1;
            match observation {
                Ok(progress) => self.finish_progress(runtime, progress, &mut result),
                Err(failure) => {
                    report_timeline_observation_refusal(&failure, runtime);
                    self.observation_failures.push_back(failure);
                    result.observation_failures += 1;
                }
            }
        }
        result
    }

    pub fn retry_observation_failure(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
    ) -> Option<ReplacementTimelineCoordinatorProgress> {
        let failure = self.observation_failures.pop_front()?;
        let mut result = ReplacementTimelineCoordinatorProgress::default();
        match runtime.apply_replacement_timeline_observation(failure.observation) {
            Ok(progress) => self.finish_progress(runtime, progress, &mut result),
            Err(failure) => {
                self.observation_failures.push_back(failure);
                result.observation_failures = 1;
            }
        }
        Some(result)
    }

    pub fn retry_semantic_failure(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
    ) -> Option<ReplacementTimelineCoordinatorProgress> {
        let pending = self.semantic_failures.pop_front()?;
        let mut result = ReplacementTimelineCoordinatorProgress::default();
        self.finish_progress_completions(
            runtime,
            pending.progress,
            pending.completions,
            &mut result,
        );
        Some(result)
    }

    fn finish_progress(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        mut progress: ReplacementTimelineProgressOwner<Semantic>,
        result: &mut ReplacementTimelineCoordinatorProgress,
    ) {
        let completions = std::mem::take(&mut progress.replay.completions).into();
        self.finish_progress_completions(runtime, progress, completions, result);
    }

    fn finish_progress_completions(
        &mut self,
        runtime: &mut ReplacementRuntimeSession<Semantic>,
        progress: ReplacementTimelineProgressOwner<Semantic>,
        completions: std::collections::VecDeque<reims_vgpu_core::CompletionFact<Semantic>>,
        result: &mut ReplacementTimelineCoordinatorProgress,
    ) {
        let (unresolved, published, _failures) = attempt_each_once(completions, |fact| {
            runtime
                .execution_mut()
                .commit_completion(fact)
                .map_err(|failure| {
                    report_timeline_completion_refusal(&failure, runtime);
                    let fact = replacement_execution_completion_failure_fact(failure);
                    ((), fact)
                })
        });
        result.published += published.len();
        self.published.extend(published);
        if unresolved.is_empty() {
            self.retired.push_back(progress);
            result.retired_batches += 1;
        } else {
            result.semantic_failures += unresolved.len();
            self.semantic_failures
                .push_back(ReplacementPendingTimelineSemanticCompletion {
                    progress,
                    completions: unresolved,
                });
        }
    }

    pub fn take_published(&mut self) -> Option<reims_vgpu_core::PublishedFact<Semantic>> {
        self.published.pop_front()
    }

    pub fn take_retired(&mut self) -> Option<ReplacementTimelineProgressOwner<Semantic>> {
        self.retired.pop_front()
    }

    /// Observations this coordinator could not apply, and completions the
    /// semantic runtime refused. Both are retained for retry and neither is
    /// visible from any live count, so a queue that stops draining reads as an
    /// idle device until the census names it.
    pub fn retained_failures(&self) -> (usize, usize) {
        (
            self.observation_failures.len(),
            self.semantic_failures
                .iter()
                .map(|pending| pending.completions.len())
                .sum(),
        )
    }

    pub fn pending_publications(&self) -> usize {
        self.published.len()
    }
}

fn replacement_execution_completion_failure_fact<Semantic>(
    failure: crate::runtime::replacement_session::ReplacementExecutionCompletionError<Semantic>,
) -> reims_vgpu_core::CompletionFact<Semantic> {
    match failure {
        crate::runtime::replacement_session::ReplacementExecutionCompletionError::UnknownGeneration {
            fact,
            ..
        } => fact,
        crate::runtime::replacement_session::ReplacementExecutionCompletionError::Commit(
            failure,
        ) => failure.fact,
    }
}

use crate::runtime::replacement_transport::ReplacementStampPage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementPublishedFactHostError {
    StampPageUnavailable,
    InvalidPageShift(u32),
    SlotPastPage { slot: u32, page_bytes: u64 },
    AddressOverflow,
    Memory(MemError),
}

#[derive(Debug)]
pub(crate) struct ReplacementPublishedFactHostFailure<Semantic> {
    pub reason: ReplacementPublishedFactHostError,
    pub fact: reims_vgpu_core::PublishedFact<Semantic>,
}

#[derive(Debug)]
pub(crate) struct ReplacementHostPublishedFact<Semantic>(
    pub reims_vgpu_core::PublishedFact<Semantic>,
);

pub(crate) struct ReplacementPublicationCoordinator<Semantic> {
    facts: std::collections::VecDeque<reims_vgpu_core::PublishedFact<Semantic>>,
}

impl<Semantic> Default for ReplacementPublicationCoordinator<Semantic> {
    fn default() -> Self {
        Self {
            facts: std::collections::VecDeque::new(),
        }
    }
}

impl<Semantic> ReplacementPublicationCoordinator<Semantic> {
    pub fn enqueue(
        &mut self,
        facts: impl IntoIterator<Item = reims_vgpu_core::PublishedFact<Semantic>>,
    ) {
        self.facts.extend(facts);
    }

    pub fn publish_next(
        &mut self,
        host: &mut (impl HostMemory + HostControl),
        transport: &crate::runtime::replacement_transport::ReplacementTransportOwner,
    ) -> Result<Option<ReplacementHostPublishedFact<Semantic>>, ReplacementPublishedFactHostError>
    {
        let Some(fact) = self.facts.pop_front() else {
            return Ok(None);
        };
        match publish_ordered_fact_to_host(
            host,
            transport.gpu_interrupt_status(),
            transport.stamp_page(),
            fact,
        ) {
            Ok(published) => Ok(Some(published)),
            Err(failure) => {
                let ReplacementPublishedFactHostFailure { reason, fact } = *failure;
                self.facts.push_front(fact);
                Err(reason)
            }
        }
    }

    pub fn pending(&self) -> usize {
        self.facts.len()
    }
}

/// Aggregate replacement owner. Publication-producing coordinators are private
/// children so their facts can reach the host only after joining this single
/// FIFO sink.
pub(crate) struct ReplacementDeviceCoordinator<Semantic> {
    runtime: ReplacementRuntimeSession<Semantic>,
    pipeline_wake_installed: bool,
    /// Cumulative count of blocked-drain retries that failed again.
    ///
    /// Every blocked drain is re-offered unconditionally, so a packet whose
    /// refusal can never change is re-offered for the life of the device. The
    /// queue length alone cannot say which happened -- one packet stuck forever
    /// and one packet that failed once and then landed both read as a moment of
    /// `blocked_drains=1` -- and the diagnostic list dedupes by reason, so a
    /// reason reported once looks the same either way. This grows once per tick
    /// per stuck packet, which is the difference.
    blocked_drain_retries: u64,
    /// Cumulative count of child packets refused because this backend declared
    /// it does not implement them. Each is one lost guest command, named once
    /// on the failure channel.
    refused_child_packets: usize,
    /// Cumulative count of transactions given up after a terminal refusal.
    ///
    /// A high-water, not a per-window sample: it never resets, so the last
    /// census line carries the boot total. A non-zero reading is guest work
    /// lost -- the reason for each is on the failure channel, named by the
    /// stage that refused it.
    abandoned_transactions: usize,
    transport: crate::runtime::replacement_transport::ReplacementTransportOwner,
    cpu: ReplacementCpuCoordinator<Semantic>,
    present: ReplacementPresentCoordinator<Semantic>,
    exec_recordings: ReplacementExecRecordingCoordinator<Semantic>,
    exec_submissions: ReplacementExecSubmitCoordinator<Semantic>,
    guest_uploads: ReplacementGuestUploadCoordinator<Semantic>,
    guest_upload_suffixes: ReplacementGuestUploadSuffixCoordinator<Semantic>,
    indirects: ReplacementIndirectCoordinator<Semantic>,
    timelines: ReplacementTimelineCoordinator<Semantic>,
    publications: ReplacementPublicationCoordinator<Semantic>,
    ready_guest_uploads: std::collections::VecDeque<
        Box<crate::runtime::replacement_session::ReplacementQueueReadyGuestUpload<Semantic>>,
    >,
    /// Transactions no live owner held at the previous pipeline census. See
    /// the `replacement_unowned_transaction` emitter.
    previously_unowned: std::collections::BTreeSet<reims_vgpu_protocol::TransactionId>,
    ready_indirect_ranges: std::collections::VecDeque<
        Box<crate::runtime::replacement_session::ReplacementQueueReadyIndirectRange<Semantic>>,
    >,
    recordings_to_cleanup: std::collections::VecDeque<
        reims_vgpu_vulkan::replacement_recording::ReplacementNativeRecording,
    >,
    pending_recording_cleanups: std::collections::VecDeque<
        reims_vgpu_vulkan::replacement_replay::PendingReplacementRecordingCleanup,
    >,
    recording_cleanup_dispatch_failures: std::collections::VecDeque<
        reims_vgpu_vulkan::replacement_replay::ReplacementRecordingCleanupFailure,
    >,
    recording_cleanup_completion_failures: std::collections::VecDeque<
        reims_vgpu_vulkan::replacement_replay::ReplacementRecordingCleanupCompletionError,
    >,
    retired_batches: std::collections::VecDeque<ReplacementTimelineProgressOwner<Semantic>>,
    continuing_guest_uploads: std::collections::VecDeque<
        crate::runtime::replacement_session::ReplacementContinuingGuestUpload<Semantic>,
    >,
    continuing_indirect_ranges: std::collections::VecDeque<
        crate::runtime::replacement_session::ReplacementContinuingIndirectRangeChain<Semantic>,
    >,
    guest_upload_resume_failures: std::collections::VecDeque<
        Box<crate::runtime::replacement_session::ReplacementGuestUploadResumeFailure<Semantic>>,
    >,
    guest_upload_continuation_failures: std::collections::VecDeque<
        Box<
            crate::runtime::replacement_session::ReplacementGuestUploadContinuationFailure<
                Semantic,
            >,
        >,
    >,
    indirect_resume_failures: std::collections::VecDeque<
        Box<
            crate::runtime::replacement_session::ReplacementIndirectAuxiliaryResumeFailure<
                Semantic,
            >,
        >,
    >,
    accepted_routing_failures:
        std::collections::VecDeque<Box<ReplacementAcceptedExecRoutingFailure<Semantic>>>,
    /// Failure for the publication currently retained at the sink's front.
    /// Repeated ticks retry that same fact, so retaining one reason mirrors
    /// the one blocked obligation instead of manufacturing one diagnostic
    /// owner per attempt.
    publication_failure: Option<ReplacementPublishedFactHostError>,
    publication_retirement_failures:
        std::collections::VecDeque<Box<ReplacementPublishedFactRetirementFailure<Semantic>>>,
    mmio_failures:
        std::collections::VecDeque<crate::runtime::replacement_transport::ReplacementGfxMmioError>,
    root_read_failure:
        Option<crate::runtime::replacement_transport::ReplacementRootPacketReadError>,
    child_read_failures: std::collections::BTreeMap<
        reims_vgpu_protocol::ChannelId,
        crate::runtime::replacement_transport::ReplacementChildPacketReadError,
    >,
    blocked_drains: std::collections::VecDeque<Box<ReplacementBlockedDrain<Semantic>>>,
    drain_failures: std::collections::VecDeque<Box<ReplacementDeviceDrainFailure<Semantic>>>,
    /// Cadence latch for [`Self::report_pipeline_census`].
    last_pipeline_census_ms: u64,
    console_frames: std::collections::BTreeMap<
        (u32, u32),
        reims_vgpu_vulkan::replacement_console_present::ReplacementConsoleFrame,
    >,
    next_console_frame: u64,
    console_frame_failure: Option<ReplacementConsoleFrameError>,
    display_online_failure:
        Option<crate::runtime::replacement_session::ReplacementDisplayOnlineError>,
    product_presented: bool,
    device_loss_effect: Option<crate::runtime::replacement_session::ReplacementDeviceLossEffect>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementConsoleFrameError {
    DeviceAbsent,
    DeviceLost,
    IdentityExhausted,
    Missing { mapping_id: u32, generation: u32 },
    Copy(reims_vgpu_vulkan::replacement_console_present::ReplacementConsoleFrameCopyError),
}

pub(crate) struct ReplacementAcceptedExecRoutingFailure<Semantic> {
    pub reason: crate::runtime::replacement_session::ReplacementNewlyReadyExecError,
    pub accepted: reims_vgpu_vulkan::replacement_exec_acceptance::AcceptedReplacementExec<
        reims_vgpu_vulkan::replacement_representation::ReplacementNativeRepresentation,
        reims_vgpu_core::ResolvedComputeDispatch,
        reims_vgpu_core::ResolvedRenderDispatch,
    >,
    _semantic: std::marker::PhantomData<Semantic>,
}

pub(crate) enum ReplacementDeviceDrainFailure<Semantic> {
    RootRead(crate::runtime::replacement_transport::ReplacementRootPacketReadError),
    RootIngress(Box<ReplacementRootPacketLeaseFailure<Semantic>>),
    ChildRead {
        channel: reims_vgpu_protocol::ChannelId,
        reason: crate::runtime::replacement_transport::ReplacementChildPacketReadError,
    },
    ChildIngress(Box<ReplacementChildPacketLeaseFailure<Semantic>>),
    DeferredChild(Box<ReplacementDeferredChildDispatchFailure<Semantic>>),
    CpuCoordinator(Box<ReplacementCpuCoordinatorAdmissionFailure<Semantic>>),
    PresentCoordinator(Box<ReplacementPresentCoordinatorAdmissionFailure<Semantic>>),
    ExecRecordingCoordinator(Box<ReplacementExecRecordingCoordinatorAdmissionFailure<Semantic>>),
    ExecSubmitCoordinator(Box<ReplacementExecSubmitCoordinatorAdmissionFailure<Semantic>>),
    Mapper(Box<ReplacementMapperEntryDispatchFailure>),
    IosfcWrite(Box<ReplacementIosfcWriteDispatchFailure>),
}

pub(crate) enum ReplacementBlockedDrain<Semantic> {
    RootIngress(Box<ReplacementRootPacketLeaseFailure<Semantic>>),
    ChildIngress(Box<ReplacementChildPacketLeaseFailure<Semantic>>),
    DeferredChild(Box<ReplacementDeferredChildDispatchFailure<Semantic>>),
    Mapper(Box<ReplacementMapperEntryDispatchFailure>),
    IosfcWrite(Box<ReplacementIosfcWriteDispatchFailure>),
}

impl<Semantic> ReplacementBlockedDrain<Semantic> {
    fn child_channel(&self) -> Option<reims_vgpu_protocol::ChannelId> {
        match self {
            Self::ChildIngress(failure) => Some(match failure.as_ref() {
                ReplacementChildPacketLeaseFailure::Ingress { lease, .. } => lease.channel,
                ReplacementChildPacketLeaseFailure::Commit { failure, .. } => failure.lease.channel,
            }),
            Self::DeferredChild(failure) => Some(match failure.as_ref() {
                ReplacementDeferredChildDispatchFailure::Exec { lease, .. }
                | ReplacementDeferredChildDispatchFailure::Cursor { lease, .. }
                | ReplacementDeferredChildDispatchFailure::Synchronize { lease, .. }
                | ReplacementDeferredChildDispatchFailure::Blocked { lease, .. } => lease.channel,
                ReplacementDeferredChildDispatchFailure::Commit { failure, .. } => {
                    failure.lease.channel
                }
            }),
            Self::RootIngress(_) | Self::Mapper(_) | Self::IosfcWrite(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReplacementCoordinatorDiagnostic {
    slug: &'static str,
    fields: Vec<(&'static str, String)>,
    discriminant: u64,
}

impl crate::observe::Decline for ReplacementCoordinatorDiagnostic {
    fn slug(&self) -> &'static str {
        self.slug
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        self.fields.clone()
    }
}

/// Report one coordinator step that refused, on the always-on failure path.
///
/// A transaction that leaves the recording coordinator and stops inside a
/// downstream one is invisible otherwise: every `progress_*` result was
/// discarded, so a chain that could not prepare, enqueue, or be accepted read
/// exactly like one still in flight. Ordinary progress stays quiet.
/// Name why one retained coordinator failure refused.
///
/// [`report_coordinator_refusal`] reports the step an owner stopped at, which
/// names the phase and not the refusal. Where a phase retains a typed reason,
/// this reports that reason beside it, because an owner stuck in a terminal
/// phase holds every transaction behind it on its channel.
/// Report why an acceptance refused a transaction whose owner is then retained.
///
/// A retained refusal holds its source domain's submission head, so every later
/// transaction on that channel refuses behind it. The stage name and the live
/// count say that something is stuck; only the acceptance reason says what.
fn report_acceptance_refusal(
    stage: &'static str,
    transaction: reims_vgpu_protocol::TransactionId,
    reason: &impl std::fmt::Debug,
) {
    report_retained_failure_detail(stage, transaction, &format!("{reason:?}"));
}

fn report_retained_failure_detail(
    stage: &'static str,
    transaction: reims_vgpu_protocol::TransactionId,
    detail: &str,
) {
    let diagnostic = ReplacementCoordinatorDiagnostic {
        slug: "replacement_retained_failure",
        fields: vec![
            ("stage", stage.to_string()),
            ("transaction", transaction.get().to_string()),
            ("detail", detail.to_string()),
        ],
        discriminant: fnv_discriminant(detail),
    };
    crate::observe::Emit::decline(stage, &diagnostic).fail_once(diagnostic.discriminant);
}

fn report_coordinator_refusal(
    coordinator: &'static str,
    transaction: reims_vgpu_protocol::TransactionId,
    step: impl std::fmt::Debug,
) {
    let diagnostic = ReplacementCoordinatorDiagnostic {
        slug: "replacement_coordinator_refused",
        fields: vec![
            ("coordinator", coordinator.to_string()),
            ("transaction", transaction.get().to_string()),
            ("step", format!("{step:?}")),
        ],
        discriminant: transaction.get(),
    };
    crate::observe::Emit::decline(coordinator, &diagnostic).fail_once(diagnostic.discriminant);
}

fn replacement_root_read_diagnostic(
    reason: crate::runtime::replacement_transport::ReplacementRootPacketReadError,
) -> Option<ReplacementCoordinatorDiagnostic> {
    use crate::runtime::replacement_transport::ReplacementRootPacketReadError as Error;
    let diagnostic = match reason {
        Error::PacketAlreadyOwned => ("replacement_root_packet_already_owned", Vec::new(), 0),
        Error::RingUnavailable => ("replacement_root_ring_unavailable", Vec::new(), 0),
        Error::RingAddressOverflow => ("replacement_root_ring_address_overflow", Vec::new(), 0),
        Error::InvalidPublishedPointers {
            head,
            tail,
            capacity,
        } => (
            "replacement_root_ring_pointers_invalid",
            vec![
                ("head", head.to_string()),
                ("tail", tail.to_string()),
                ("capacity", capacity.to_string()),
            ],
            (u64::from(head) << u32::BITS) | u64::from(tail),
        ),
        Error::Memory(reason) => (
            "replacement_root_ring_memory_unavailable",
            vec![("memory", format!("{reason:?}"))],
            0,
        ),
        Error::Packet(crate::runtime::fifo_packet::PacketError::BadSize) => {
            ("replacement_root_packet_size_invalid", Vec::new(), 0)
        }
        Error::Packet(crate::runtime::fifo_packet::PacketError::ShortSnapshot) => {
            ("replacement_root_packet_snapshot_short", Vec::new(), 0)
        }
        Error::Packet(
            crate::runtime::fifo_packet::PacketError::ShortHeader
            | crate::runtime::fifo_packet::PacketError::Incomplete,
        ) => return None,
    };
    Some(ReplacementCoordinatorDiagnostic {
        slug: diagnostic.0,
        fields: diagnostic.1,
        discriminant: diagnostic.2,
    })
}

fn replacement_child_read_diagnostic(
    channel: reims_vgpu_protocol::ChannelId,
    reason: crate::runtime::replacement_transport::ReplacementChildPacketReadError,
) -> Option<ReplacementCoordinatorDiagnostic> {
    use crate::runtime::replacement_transport::ReplacementChildPacketReadError as Error;
    let channel_value = u64::from(channel.get());
    let mut fields = vec![("channel", channel.get().to_string())];
    let (slug, discriminant) = match reason {
        Error::InvalidChannel(found) => {
            fields.push(("found", found.get().to_string()));
            ("replacement_child_channel_invalid", u64::from(found.get()))
        }
        Error::PacketAlreadyOwned(found) => {
            fields.push(("found", found.get().to_string()));
            ("replacement_child_packet_already_owned", channel_value)
        }
        Error::RootPageUnavailable => ("replacement_child_root_page_unavailable", channel_value),
        Error::RingAddressOverflow => ("replacement_child_ring_address_overflow", channel_value),
        Error::RingUnavailable => ("replacement_child_ring_unavailable", channel_value),
        Error::RingLengthOverflow => ("replacement_child_ring_length_overflow", channel_value),
        Error::InvalidPublishedPointers {
            head,
            tail,
            capacity,
        } => {
            fields.extend([
                ("head", head.to_string()),
                ("tail", tail.to_string()),
                ("capacity", capacity.to_string()),
            ]);
            (
                "replacement_child_ring_pointers_invalid",
                channel_value ^ (u64::from(head) << u32::BITS) ^ u64::from(tail),
            )
        }
        Error::Memory { phase, reason } => {
            let (phase, entry) = match phase {
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::Head => {
                    ("head", None)
                }
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::StampIndex => {
                    ("stamp_index", None)
                }
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::BasePfn => {
                    ("base_pfn", None)
                }
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::PageList { entry } => {
                    ("page_list", Some(entry))
                }
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::Tail => {
                    ("tail", None)
                }
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::Header => {
                    ("header", None)
                }
                crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::Snapshot => {
                    ("snapshot", None)
                }
            };
            fields.push(("phase", phase.to_owned()));
            if let Some(entry) = entry {
                fields.push(("entry", entry.to_string()));
            }
            fields.push(("memory", format!("{reason:?}")));
            ("replacement_child_ring_memory_unavailable", channel_value)
        }
        Error::Packet(crate::runtime::fifo_packet::PacketError::BadSize) => {
            ("replacement_child_packet_size_invalid", channel_value)
        }
        Error::Packet(crate::runtime::fifo_packet::PacketError::ShortSnapshot) => {
            ("replacement_child_packet_snapshot_short", channel_value)
        }
        Error::Packet(
            crate::runtime::fifo_packet::PacketError::ShortHeader
            | crate::runtime::fifo_packet::PacketError::Incomplete,
        ) => return None,
    };
    Some(ReplacementCoordinatorDiagnostic {
        slug,
        fields,
        discriminant,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplacementDeviceDrainProgress {
    pub root_packets: usize,
    pub child_packets: usize,
    pub mapper_entries: usize,
    pub failures: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplacementDeviceTickProgress {
    pub drained: ReplacementDeviceDrainProgress,
    pub cpu_completed: usize,
    pub recordings_handed_off: usize,
    pub exec_acceptances: usize,
    pub timeline_observations: usize,
    pub guest_uploads_resumed: usize,
    pub retired_batches: usize,
    pub recording_cleanups: usize,
    pub publications: usize,
}

impl<Semantic: Clone + PartialEq + Send + 'static> ReplacementDeviceCoordinator<Semantic> {
    pub fn new(
        runtime: ReplacementRuntimeSession<Semantic>,
        page_shift: u32,
    ) -> Result<Self, crate::runtime::replacement_transport::ReplacementTransportStartError> {
        Ok(Self {
            abandoned_transactions: 0,
            blocked_drain_retries: 0,
            refused_child_packets: 0,
            runtime,
            pipeline_wake_installed: false,
            transport: crate::runtime::replacement_transport::ReplacementTransportOwner::new(
                page_shift,
            )?,
            cpu: ReplacementCpuCoordinator::default(),
            present: ReplacementPresentCoordinator::default(),
            exec_recordings: ReplacementExecRecordingCoordinator::default(),
            exec_submissions: ReplacementExecSubmitCoordinator::default(),
            guest_uploads: ReplacementGuestUploadCoordinator::default(),
            guest_upload_suffixes: ReplacementGuestUploadSuffixCoordinator::default(),
            indirects: ReplacementIndirectCoordinator::default(),
            timelines: ReplacementTimelineCoordinator::default(),
            publications: ReplacementPublicationCoordinator::default(),
            ready_guest_uploads: std::collections::VecDeque::new(),
            previously_unowned: std::collections::BTreeSet::new(),
            ready_indirect_ranges: std::collections::VecDeque::new(),
            recordings_to_cleanup: std::collections::VecDeque::new(),
            pending_recording_cleanups: std::collections::VecDeque::new(),
            recording_cleanup_dispatch_failures: std::collections::VecDeque::new(),
            recording_cleanup_completion_failures: std::collections::VecDeque::new(),
            retired_batches: std::collections::VecDeque::new(),
            continuing_guest_uploads: std::collections::VecDeque::new(),
            continuing_indirect_ranges: std::collections::VecDeque::new(),
            guest_upload_resume_failures: std::collections::VecDeque::new(),
            guest_upload_continuation_failures: std::collections::VecDeque::new(),
            indirect_resume_failures: std::collections::VecDeque::new(),
            accepted_routing_failures: std::collections::VecDeque::new(),
            publication_failure: None,
            publication_retirement_failures: std::collections::VecDeque::new(),
            mmio_failures: std::collections::VecDeque::new(),
            root_read_failure: None,
            child_read_failures: std::collections::BTreeMap::new(),
            blocked_drains: std::collections::VecDeque::new(),
            drain_failures: std::collections::VecDeque::new(),
            last_pipeline_census_ms: 0,
            console_frames: std::collections::BTreeMap::new(),
            next_console_frame: 0,
            console_frame_failure: None,
            display_online_failure: None,
            product_presented: false,
            device_loss_effect: None,
        })
    }

    fn clear_aggregate_owners(&mut self) {
        self.cpu = ReplacementCpuCoordinator::default();
        self.present = ReplacementPresentCoordinator::default();
        self.exec_recordings = ReplacementExecRecordingCoordinator::default();
        self.exec_submissions = ReplacementExecSubmitCoordinator::default();
        self.guest_uploads = ReplacementGuestUploadCoordinator::default();
        self.guest_upload_suffixes = ReplacementGuestUploadSuffixCoordinator::default();
        self.indirects = ReplacementIndirectCoordinator::default();
        self.timelines = ReplacementTimelineCoordinator::default();
        self.publications = ReplacementPublicationCoordinator::default();
        self.ready_guest_uploads.clear();
        self.ready_indirect_ranges.clear();
        self.recordings_to_cleanup.clear();
        self.pending_recording_cleanups.clear();
        self.recording_cleanup_dispatch_failures.clear();
        self.recording_cleanup_completion_failures.clear();
        self.retired_batches.clear();
        self.continuing_guest_uploads.clear();
        self.continuing_indirect_ranges.clear();
        self.guest_upload_resume_failures.clear();
        self.guest_upload_continuation_failures.clear();
        self.indirect_resume_failures.clear();
        self.accepted_routing_failures.clear();
        self.publication_failure = None;
        self.publication_retirement_failures.clear();
        self.mmio_failures.clear();
        self.root_read_failure = None;
        self.child_read_failures.clear();
        self.blocked_drains.clear();
        self.drain_failures.clear();
        self.console_frames.clear();
        self.next_console_frame = 0;
        self.console_frame_failure = None;
        self.display_online_failure = None;
        self.product_presented = false;
        self.transport.reset();
    }

    /// Close all aggregate ownership once the shared Vulkan epoch reports
    /// loss. The effect is retained exactly once for the always-on failure
    /// report; repeated polls perform no teardown and publish no success.
    pub fn terminalize_device_loss(&mut self) -> bool {
        if self.device_loss_effect.is_some() {
            return true;
        }
        if self.vulkan_state() == reims_vgpu_core::VulkanDeviceEpochState::Active {
            return false;
        }
        let effect = self.runtime.terminate_device_loss();
        self.clear_aggregate_owners();
        self.device_loss_effect = Some(effect);
        true
    }

    pub const fn device_loss_effect(
        &self,
    ) -> Option<crate::runtime::replacement_session::ReplacementDeviceLossEffect> {
        self.device_loss_effect
    }

    fn absorb_publications(&mut self) {
        while let Some(fact) = self.cpu.take_published() {
            self.publications.enqueue([fact]);
        }
        while let Some(fact) = self.present.take_published() {
            self.publications.enqueue([fact]);
        }
        while let Some(fact) = self.timelines.take_published() {
            self.publications.enqueue([fact]);
        }
    }

    pub fn gfx_read(&mut self, offset: u64, size: u32) -> u64 {
        if self.terminalize_device_loss() {
            return 0;
        }
        match self.transport.gfx_read(offset, size) {
            Ok(value) => value,
            Err(reason) => {
                self.mmio_failures.push_back(reason);
                0
            }
        }
    }

    pub fn iosfc_read(&self, offset: u64, size: u32) -> u64 {
        if self.device_loss_effect.is_some()
            || self.vulkan_state() != reims_vgpu_core::VulkanDeviceEpochState::Active
        {
            return 0;
        }
        self.transport.iosfc_read(offset, size)
    }

    #[cfg(feature = "host-window")]
    /// Attach the native window to this aggregate's exact Vulkan epoch.
    ///
    /// # Safety
    ///
    /// Both handles must name the same live window and remain valid until
    /// [`Self::detach_window`] completes or this coordinator is destroyed.
    pub unsafe fn attach_window(
        &self,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<(), reims_vgpu_vulkan::replacement_window_present::ReplacementWindowAttachError>
    {
        unsafe {
            self.runtime
                .session()
                .vulkan()
                .attach_window(display, window, width, height)
        }
    }

    #[cfg(feature = "host-window")]
    pub fn resize_window(&self, width: u32, height: u32) {
        self.runtime.session().vulkan().resize_window(width, height);
    }

    #[cfg(feature = "host-window")]
    /// Retire the swapchain and surface before their native window disappears.
    ///
    /// # Safety
    ///
    /// The handles supplied to [`Self::attach_window`] must still be live.
    pub unsafe fn detach_window(
        &self,
    ) -> Result<(), reims_vgpu_vulkan::replacement_window_present::ReplacementWindowDetachError>
    {
        unsafe { self.runtime.session().vulkan().detach_window() }
    }

    pub const fn page_shift(&self) -> u32 {
        self.transport.page_shift()
    }

    pub fn transport_registers(&self) -> &reims_vgpu_core::DeviceRegisters {
        self.transport.registers()
    }

    pub fn cursor(&self) -> &reims_vgpu_core::CursorState {
        self.runtime.cursor()
    }

    pub const fn product_presented(&self) -> bool {
        self.product_presented
    }

    pub fn vulkan_state(&self) -> reims_vgpu_core::VulkanDeviceEpochState {
        self.runtime.session().vulkan().state()
    }

    /// Re-offer every retained timeline fact to a fixed point. One observation
    /// may release a semantic fact in another retained batch, so stopping at
    /// the first unchanged head can leave a ready producer unvisited.
    pub fn retry_timeline_failures(&mut self) -> ReplacementTimelineCoordinatorProgress {
        let mut progress = ReplacementTimelineCoordinatorProgress::default();
        loop {
            let before = self.timelines.retained_failures();
            let observation_attempts = self.timelines.observation_failures.len();
            let semantic_attempts = self.timelines.semantic_failures.len();
            for _ in 0..observation_attempts {
                let step = self
                    .timelines
                    .retry_observation_failure(&mut self.runtime)
                    .expect("the attempt count came from the observation queue");
                progress.observed += step.observed;
                progress.observation_failures += step.observation_failures;
                progress.semantic_failures += step.semantic_failures;
                progress.published += step.published;
                progress.retired_batches += step.retired_batches;
            }
            for _ in 0..semantic_attempts {
                let step = self
                    .timelines
                    .retry_semantic_failure(&mut self.runtime)
                    .expect("the attempt count came from the semantic queue");
                progress.observed += step.observed;
                progress.observation_failures += step.observation_failures;
                progress.semantic_failures += step.semantic_failures;
                progress.published += step.published;
                progress.retired_batches += step.retired_batches;
            }
            let after = self.timelines.retained_failures();
            if after == before || after == (0, 0) {
                break;
            }
        }
        self.absorb_publications();
        while let Some(retired) = self.timelines.take_retired() {
            self.retired_batches.push_back(retired);
        }
        progress
    }

    pub fn poll_timelines(&mut self) -> ReplacementTimelineCoordinatorProgress {
        let progress = self.timelines.poll(&mut self.runtime);
        self.absorb_publications();
        while let Some(retired) = self.timelines.take_retired() {
            self.retired_batches.push_back(retired);
        }
        progress
    }

    pub fn publish_next(
        &mut self,
        host: &mut (impl HostMemory + HostControl),
    ) -> Result<Option<ReplacementHostPublishedFact<Semantic>>, ReplacementPublishedFactHostError>
    {
        self.absorb_publications();
        self.publications.publish_next(host, &self.transport)
    }

    pub fn retire_published(
        &mut self,
        published: ReplacementHostPublishedFact<Semantic>,
    ) -> Result<
        reims_vgpu_core::PublishedFact<Semantic>,
        Box<ReplacementPublishedFactRetirementFailure<Semantic>>,
    > {
        retire_host_published_fact(&mut self.runtime, published)
    }

    pub fn pending_publications(&self) -> usize {
        self.publications.pending()
            + self.cpu.pending_publications()
            + self.present.pending_publications()
            + self.timelines.pending_publications()
    }

    pub fn live_transactions(&self) -> usize {
        self.runtime.execution().runtime().live_transactions()
    }

    pub fn owned_phase_count(&self) -> usize {
        #[cfg(feature = "host-window")]
        let retired_swapchains = self.runtime.pending_retired_swapchain_generations();
        #[cfg(not(feature = "host-window"))]
        let retired_swapchains = 0;
        self.cpu.live_packets()
            + self.present.live_presentations()
            + self.exec_recordings.live_recordings()
            + self.exec_submissions.live_submissions()
            + self.guest_uploads.live_uploads()
            + self.guest_upload_suffixes.live_suffixes()
            + self.indirects.live_ranges()
            + self.ready_guest_uploads.len()
            + self.ready_indirect_ranges.len()
            + self.recordings_to_cleanup.len()
            + self.pending_recording_cleanups.len()
            + self.recording_cleanup_dispatch_failures.len()
            + self.recording_cleanup_completion_failures.len()
            + self.retired_batches.len()
            + self.continuing_guest_uploads.len()
            + self.continuing_indirect_ranges.len()
            + self.guest_upload_resume_failures.len()
            + self.guest_upload_continuation_failures.len()
            + self.indirect_resume_failures.len()
            + self.accepted_routing_failures.len()
            + usize::from(self.publication_failure.is_some())
            + self.publication_retirement_failures.len()
            + self.mmio_failures.len()
            + usize::from(self.root_read_failure.is_some())
            + self.child_read_failures.len()
            + self.blocked_drains.len()
            + self.drain_failures.len()
            + self.console_frames.len()
            + usize::from(self.console_frame_failure.is_some())
            + usize::from(self.display_online_failure.is_some())
            + retired_swapchains
    }

    pub fn poll_display_online(
        &mut self,
        host: &mut (impl HostMemory + HostControl),
    ) -> Result<
        reims_vgpu_core::DisplayOnlineNotification,
        crate::runtime::replacement_session::ReplacementDisplayOnlineError,
    > {
        if self.terminalize_device_loss() {
            return Err(
                crate::runtime::replacement_session::ReplacementDisplayOnlineError::NativeLifetimeClosed,
            );
        }
        match self.runtime.progress_display_online(
            host,
            self.transport
                .registers()
                .gfx
                .interrupt_status_disp
                .as_ref(),
        ) {
            Ok(notification) => {
                self.display_online_failure = None;
                Ok(notification)
            }
            Err(reason) => {
                self.display_online_failure = Some(reason);
                Err(reason)
            }
        }
    }

    fn next_console_frame_identity(&mut self) -> Result<(u32, u32), ReplacementConsoleFrameError> {
        let mapping_space = u64::from(u32::MAX);
        let generation = u32::try_from(self.next_console_frame / mapping_space)
            .map_err(|_| ReplacementConsoleFrameError::IdentityExhausted)?;
        let mapping_id = u32::try_from(self.next_console_frame % mapping_space)
            .expect("the remainder is narrower than u32::MAX")
            + 1;
        self.next_console_frame = self
            .next_console_frame
            .checked_add(1)
            .ok_or(ReplacementConsoleFrameError::IdentityExhausted)?;
        Ok((mapping_id, generation))
    }

    pub fn console_frame_may_paint(&self, mapping_id: u32) -> bool {
        if self.device_loss_effect.is_some()
            || self.vulkan_state() != reims_vgpu_core::VulkanDeviceEpochState::Active
        {
            return false;
        }
        self.console_frames
            .keys()
            .any(|(candidate, _)| *candidate == mapping_id)
    }

    pub fn copy_console_frame(
        &mut self,
        mapping_id: u32,
        generation: u32,
        destination: &mut [u8],
        destination_stride: u32,
        width: u32,
        height: u32,
    ) -> Result<(), ReplacementConsoleFrameError> {
        if self.terminalize_device_loss() {
            return Err(ReplacementConsoleFrameError::DeviceLost);
        }
        let key = (mapping_id, generation);
        let frame = self
            .console_frames
            .get(&key)
            .ok_or(ReplacementConsoleFrameError::Missing {
                mapping_id,
                generation,
            })?;
        frame
            .copy_to_bgra8(destination, destination_stride, width, height)
            .map_err(ReplacementConsoleFrameError::Copy)?;
        self.console_frames.remove(&key);
        Ok(())
    }

    pub fn take_drain_failure(&mut self) -> Option<Box<ReplacementDeviceDrainFailure<Semantic>>> {
        self.drain_failures.pop_front()
    }
}

impl ReplacementDeviceCoordinator<()> {
    pub fn cpu_failure_diagnostics(
        &self,
    ) -> Vec<(reims_vgpu_protocol::TransactionId, &'static str, String)> {
        self.cpu.failure_diagnostics()
    }

    pub fn pipeline_failure_diagnostics(
        &self,
    ) -> Vec<(
        &'static str,
        String,
        reims_vgpu_core::PipelineFailureStage,
        String,
    )> {
        self.runtime.session().pipeline_failure_diagnostics()
    }

    pub fn pipeline_state_diagnostics(
        &self,
    ) -> Vec<(&'static str, String, reims_vgpu_core::PipelineState)> {
        self.runtime.session().pipeline_state_diagnostics()
    }

    pub fn transaction_state_diagnostics(&self) -> Vec<(u64, String)> {
        self.runtime.transaction_state_diagnostics()
    }

    pub fn blocked_drain_diagnostics(&self) -> Vec<String> {
        self.blocked_drains
            .iter()
            .map(|blocked| match blocked.as_ref() {
                ReplacementBlockedDrain::RootIngress(failure) => match failure.as_ref() {
                    ReplacementRootPacketLeaseFailure::Validation { reason, lease } => format!(
                        "route=root_validation opcode={:#x} reason={reason:?}",
                        lease.packet.opcode
                    ),
                    ReplacementRootPacketLeaseFailure::Admission { reason, lease } => format!(
                        "route=root_ingress opcode={:#x} reason={reason:?}",
                        lease.packet.opcode
                    ),
                    ReplacementRootPacketLeaseFailure::Commit { failure, .. } => format!(
                        "route=root_commit opcode={:#x} reason={:?}",
                        failure.lease.packet.opcode, failure.reason
                    ),
                },
                ReplacementBlockedDrain::ChildIngress(failure) => match failure.as_ref() {
                    ReplacementChildPacketLeaseFailure::Ingress { reason, lease } => format!(
                        "route=child_ingress channel={} opcode={:#x} reason={reason:?}",
                        lease.channel.get(),
                        lease.packet.opcode
                    ),
                    ReplacementChildPacketLeaseFailure::Commit { failure, .. } => format!(
                        "route=child_commit channel={} opcode={:#x} reason={:?}",
                        failure.lease.channel.get(),
                        failure.lease.packet.opcode,
                        failure.reason
                    ),
                },
                ReplacementBlockedDrain::DeferredChild(failure) => match failure.as_ref() {
                    ReplacementDeferredChildDispatchFailure::Exec { failure, lease } => format!(
                        "route=deferred_exec channel={} {}",
                        lease.channel.get(),
                        replacement_host_exec_failure_diagnostic(failure)
                    ),
                    // A blocked head names *what* refused it. A route tag
                    // alone says a channel is stuck and nothing about why,
                    // which is the one question this line exists to answer.
                    ReplacementDeferredChildDispatchFailure::Cursor { failure, lease } => format!(
                        "route=deferred_cursor channel={} reason=cursor_dispatch_refused refusal={:?}",
                        lease.channel.get(),
                        failure.as_ref()
                    ),
                    ReplacementDeferredChildDispatchFailure::Synchronize { failure, lease } => {
                        format!(
                            "route=deferred_synchronize channel={} reason=synchronize_dispatch_refused refusal={}",
                            lease.channel.get(),
                            failure.diagnostic()
                        )
                    }
                    ReplacementDeferredChildDispatchFailure::Blocked { lease, .. } => format!(
                        "route=deferred_unknown channel={} reason=transport_classification_blocked",
                        lease.channel.get()
                    ),
                    ReplacementDeferredChildDispatchFailure::Commit { failure, .. } => format!(
                        "route=deferred_commit channel={} reason=ring_commit_refused",
                        failure.lease.channel.get()
                    ),
                },
                ReplacementBlockedDrain::Mapper(reason) => {
                    format!("route=mapper reason={reason:?}")
                }
                ReplacementBlockedDrain::IosfcWrite(reason) => {
                    format!("route=iosfc reason={reason:?}")
                }
            })
            .collect()
    }

    fn route_accepted_exec(
        &mut self,
        mut accepted: crate::runtime::replacement_session::AcceptedReplacementIndirectFinal,
    ) -> Result<(), Box<ReplacementAcceptedExecRoutingFailure<()>>> {
        let newly_ready = match self
            .runtime
            .take_newly_ready_recorded_execs(&accepted.replay.replay.native)
        {
            Ok(ready) => ready,
            Err(reason) => {
                return Err(Box::new(ReplacementAcceptedExecRoutingFailure {
                    reason,
                    accepted,
                    _semantic: std::marker::PhantomData,
                }));
            }
        };
        if let Some(recording) = accepted.replay.ready_recording.take() {
            self.recordings_to_cleanup.push_back(recording);
        }
        for ready in newly_ready {
            self.route_ready_recording(ready);
        }
        Ok(())
    }

    /// Hand one queue-ready recorded owner to the coordinator its own variant
    /// names. This is the single route out of readiness: recordings settled by
    /// `progress_exec_recordings` and owners released from the epoch parked map
    /// after a predecessor is accepted both pass through here, so neither can
    /// grow a route the other lacks.
    fn route_ready_recording(
        &mut self,
        ready: crate::runtime::replacement_session::ReplacementQueueReadyRecording<()>,
    ) {
        match ready {
            crate::runtime::replacement_session::ReplacementQueueReadyRecording::Exec(ready) => {
                if let Err(failure) = self.exec_submissions.admit(ready, ()) {
                    self.drain_failures.push_back(Box::new(
                        ReplacementDeviceDrainFailure::ExecSubmitCoordinator(failure),
                    ));
                }
            }
            crate::runtime::replacement_session::ReplacementQueueReadyRecording::GuestUpload(
                ready,
            ) => {
                if let Err((ready, _semantic)) = self.guest_uploads.admit(ready, ()) {
                    self.ready_guest_uploads.push_back(ready);
                }
            }
            crate::runtime::replacement_session::ReplacementQueueReadyRecording::IndirectRange(
                ready,
            ) => self.ready_indirect_ranges.push_back(ready),
        }
    }

    pub fn gfx_write(&mut self, host: &mut impl HostControl, offset: u64, data: u64, size: u32) {
        if self.terminalize_device_loss() {
            return;
        }
        let effects = match self.transport.gfx_write(offset, data, size) {
            Ok(effects) => effects,
            Err(reason) => {
                self.mmio_failures.push_back(reason);
                return;
            }
        };
        for effect in effects {
            match effect {
                crate::runtime::replacement_transport::ReplacementGfxWriteEffect::ScheduleDrain => {
                    host.schedule_bh();
                }
                crate::runtime::replacement_transport::ReplacementGfxWriteEffect::ProtocolNegotiated(
                    _,
                ) => {}
                crate::runtime::replacement_transport::ReplacementGfxWriteEffect::DisplayInterrupt(
                    _,
                ) => {
                    host.enqueue(HostAction::irq_gfx());
                    host.schedule_bh();
                }
            }
        }
    }

    fn retry_publication_retirement(&mut self) {
        let Some(failure) = self.publication_retirement_failures.pop_front() else {
            return;
        };
        match self.retire_published(failure.published) {
            Ok(_) => {}
            Err(failure) => self.publication_retirement_failures.push_front(failure),
        }
    }

    /// One-per-second census of where guest work is sitting.
    ///
    /// A transaction that leaves one coordinator and stops inside the next is
    /// otherwise indistinguishable from one that completed: the refusal
    /// reporters name work that was *refused*, and this names work that is
    /// merely *held*. Read the two together — a live count that stops falling
    /// with no refusal beside it is a stage that is waiting for something that
    /// will not arrive. Parked counts are the epoch's, so they say whether an
    /// owner is waiting on a predecessor rather than on its own chain.
    ///
    /// It reports only while the pipeline holds work, so an idle device is
    /// silent, and it is on the `OFF` channel because holding work is not a
    /// loss.
    fn report_pipeline_census(&mut self) {
        let parked = self.runtime.parked_recordings();
        // The gate is [`Self::owned_phase_count`] and nothing else, because a
        // second hand-written sum of "is this device holding work" is a second
        // rule that silently disagrees with the first. Both spellings of it
        // have now been wrong in the same direction: one left out the blocked
        // drains, so a packet the device could never admit went unreported once
        // the pipeline emptied, and the next left out the CPU coordinator, so
        // three control transactions stuck in a permanent apply failure -- with
        // the guest waiting on the completion stamps behind them -- read as a
        // quiet, healthy, idle device for the remaining nine minutes of a boot.
        if self.owned_phase_count() + parked.execs + parked.guest_uploads + parked.indirect_ranges
            == 0
        {
            return;
        }
        let now_ms = crate::observe::elapsed_us() / 1_000;
        if now_ms.saturating_sub(self.last_pipeline_census_ms) < 1_000 {
            return;
        }
        self.last_pipeline_census_ms = now_ms;
        let waiting = self
            .runtime
            .parked_native_candidates()
            .into_iter()
            .filter(|(_, unmet)| !unmet.is_empty())
            .map(|(transaction, unmet)| {
                format!(
                    "{}<-[{}]",
                    transaction.get(),
                    unmet
                        .iter()
                        .map(|producer| producer.get().to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let upload_stages = self
            .guest_uploads
            .stages()
            .into_iter()
            .map(|(transaction, stage)| format!("{}:{stage}", transaction.get()))
            .collect::<Vec<_>>()
            .join(",");
        crate::observe::off(format!(
            "replacement_pipeline_census recordings={} submissions={} uploads={} upload_stages=[{upload_stages}] retired_batches={} upload_suffixes={} indirects={} ready_uploads={} ready_indirects={} parked_execs={} parked_uploads={} parked_indirects={} blocked_drains={} drain_failures={}",
            self.exec_recordings.live_recordings(),
            self.exec_submissions.live_submissions(),
            self.guest_uploads.live_uploads(),
            self.retired_batches.len(),
            self.guest_upload_suffixes.live_suffixes(),
            self.indirects.live_ranges(),
            self.ready_guest_uploads.len(),
            self.ready_indirect_ranges.len(),
            parked.execs,
            parked.guest_uploads,
            parked.indirect_ranges,
            self.blocked_drains.len(),
            self.drain_failures.len(),
        ));
        let (timeline_observations, timeline_semantics) = self.timelines.retained_failures();
        let cpu_failed = self
            .cpu_failure_diagnostics()
            .into_iter()
            .map(|(transaction, stage, _)| format!("{}:{stage}", transaction.get()))
            .collect::<Vec<_>>()
            .join(",");
        let publish_fail = self
            .publication_failure
            .map(|reason| format!("{reason:?}"))
            .unwrap_or_default();
        let publish_retire_head = self
            .publication_retirement_failures
            .front()
            .map(|failure| {
                format!(
                    "{}:{:?}",
                    failure.published.0.transaction.get(),
                    failure.reason
                )
            })
            .unwrap_or_default();
        let blocked_head = self
            .blocked_drain_diagnostics()
            .first()
            .cloned()
            .unwrap_or_default();
        crate::observe::off(format!(
            "replacement_pipeline_stalls cpu_live={} cpu_failed=[{cpu_failed}] cpu_publications={} timeline_observations={timeline_observations} timeline_semantics={timeline_semantics} abandoned={} refused_packets={} blocked_retries={} blocked_head=[{blocked_head}] upload_resume={} upload_continuation={} indirect_resume={} accepted_routing={} publication_retire={} publish_fail=[{publish_fail}] publish_retire_head=[{publish_retire_head}] cleanup_dispatch={} cleanup_completion={} mmio={} continuing_uploads={} continuing_indirects={}",
            self.cpu.live_packets(),
            self.cpu.pending_publications(),
            self.abandoned_transactions,
            self.refused_child_packets,
            self.blocked_drain_retries,
            self.guest_upload_resume_failures.len(),
            self.guest_upload_continuation_failures.len(),
            self.indirect_resume_failures.len(),
            self.accepted_routing_failures.len(),
            self.publication_retirement_failures.len(),
            self.recording_cleanup_dispatch_failures.len(),
            self.recording_cleanup_completion_failures.len(),
            self.mmio_failures.len(),
            self.continuing_guest_uploads.len(),
            self.continuing_indirect_ranges.len(),
        ));
        // Where every tracked transaction sits. A blocked head names the
        // producer it waits for, and the only useful next question is what that
        // producer is itself waiting for -- which the live-recording gauge
        // cannot answer, because a count cannot say which of the six is the one
        // somebody is parked on. Flags are terse on purpose: this is one line a
        // second on a device that may hold many transactions.
        let order = self
            .runtime
            .submission_order_census()
            .into_iter()
            .map(|entry| {
                let mut flags = String::new();
                for (set, mark) in [
                    (entry.recorded, 'r'),
                    (entry.issued, 'i'),
                    (entry.submitted, 's'),
                    (entry.abandoned, 'a'),
                ] {
                    if set {
                        flags.push(mark);
                    }
                }
                format!(
                    "{}@{}.{}{}",
                    entry.transaction.get(),
                    entry.domain.get(),
                    entry.sequence.get(),
                    if flags.is_empty() {
                        String::from(":-")
                    } else {
                        format!(":{flags}")
                    }
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        // A transaction the submission order has not settled and no live owner
        // holds is lost guest work that nothing else reports. Its domain claim
        // is never released, so every later transaction on that domain refuses
        // behind it with `NotSubmissionHead` for the life of the device -- and
        // the counters all read as if the device were merely busy: live owner
        // gauges at zero, no retained failure, no refusal, and a blocked head
        // that names the *successor* rather than the transaction that was
        // dropped. Two boots were spent reading that as an ordering bug.
        //
        // This is a lead and not a verdict, which is why it is on the `OFF`
        // channel and why it reports only what was unowned at two consecutive
        // censuses. The owner sets below are the ones the pipeline holds
        // transactions in; a retained failure elsewhere is held for the moment
        // it takes to retry, which one census can catch and two cannot. A name
        // that appears here for the life of a boot is the real thing.
        let mut owned = std::collections::BTreeSet::new();
        owned.extend(self.cpu.transaction_ids());
        owned.extend(self.present.transaction_ids());
        owned.extend(self.exec_recordings.transaction_ids());
        owned.extend(self.exec_submissions.transaction_ids());
        owned.extend(self.guest_uploads.transaction_ids());
        owned.extend(self.guest_upload_suffixes.transaction_ids());
        owned.extend(self.indirects.transaction_ids());
        owned.extend(self.runtime.parked_recording_transactions());
        owned.extend(
            self.ready_guest_uploads
                .iter()
                .map(|ready| ready.transaction()),
        );
        owned.extend(
            self.ready_indirect_ranges
                .iter()
                .map(|ready| ready.transaction()),
        );
        owned.extend(
            self.continuing_guest_uploads
                .iter()
                .map(|continuing| continuing.transaction()),
        );
        owned.extend(
            self.continuing_indirect_ranges
                .iter()
                .map(|continuing| continuing.transaction()),
        );
        let orphaned =
            unowned_submission_positions(&self.runtime.submission_order_census(), &owned);
        // A stale execution representation names the backing and stops. What
        // that backing holds instead is the other half of the disagreement,
        // and without it the reading is "some content is not current" -- which
        // is what the refusal already said. One line per distinct backing,
        // deduped, because a head in this state repeats once a second.
        for backing in self
            .blocked_drains
            .iter()
            .filter_map(|blocked| match blocked.as_ref() {
                ReplacementBlockedDrain::DeferredChild(failure) => match failure.as_ref() {
                    ReplacementDeferredChildDispatchFailure::Exec { failure, .. } => {
                        failure.stale_backing()
                    }
                    _ => None,
                },
                _ => None,
            })
        {
            let Some((representation, holds)) =
                self.runtime.execution_representation_coverage(backing)
            else {
                continue;
            };
            crate::observe::off(format!(
                "replacement_stale_representation backing={} representation={} holds=[{holds}]",
                backing.get(),
                representation.get()
            ));
        }
        let unowned = orphaned
            .iter()
            .map(|(transaction, _)| *transaction)
            .collect::<std::collections::BTreeSet<_>>();
        let sustained = orphaned
            .iter()
            .filter(|(transaction, _)| self.previously_unowned.contains(transaction))
            .map(|(_, position)| position.clone())
            .collect::<Vec<_>>();
        self.previously_unowned = unowned;
        if !sustained.is_empty() {
            crate::observe::off(format!(
                "replacement_unowned_transaction reason=submission_claim_held_by_no_live_owner transactions=[{}]",
                sustained.join(" ")
            ));
        }
        if !order.is_empty() {
            crate::observe::off(format!(
                "replacement_submission_order transaction@domain.sequence:rias {order}"
            ));
        }
        if !waiting.is_empty() {
            crate::observe::off(format!(
                "replacement_pipeline_waits consumer<-unmet_producers {waiting}"
            ));
        }
    }

    pub fn tick(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    ) -> ReplacementDeviceTickProgress {
        if self.terminalize_device_loss() {
            return ReplacementDeviceTickProgress::default();
        }
        if !self.pipeline_wake_installed {
            match self
                .runtime
                .session()
                .install_pipeline_wake(host.worker_wake())
            {
                Ok(())
                | Err(
                    crate::runtime::replacement_session::PipelineWakeInstallError::AlreadyInstalled,
                ) => self.pipeline_wake_installed = true,
            }
        }
        let _ = self.runtime.session().progress_pipeline_completions();
        let drained = self.drain(host);
        let cpu_completed = self.progress_cpu_packets(host);
        let _ = self.progress_present_preparations();
        // SAFETY: the aggregate owns the one live runtime/epoch and never lets
        // a presentation allocation escape to another epoch.
        let _ = unsafe { self.progress_present_queues() };
        let recordings_handed_off = self.progress_exec_recordings();
        let _ = self.progress_exec_submissions();
        let exec_acceptances = self.harvest_exec_acceptances();
        let _ = self.progress_guest_uploads();
        let timeline = self.poll_timelines();
        // Both queues are retained for retry and nothing else drains them, so
        // without one attempt per tick a single refused observation or
        // completion stops every later retirement for the boot's life.
        let _ = self.retry_timeline_failures();
        let guest_uploads_resumed = self.resume_completed_guest_uploads();
        let _ = self.progress_guest_upload_suffixes();
        // A refused suffix is terminal and nothing else claims it, so without
        // one release per tick a single unimplemented case holds the channel's
        // submission head and publication position for the boot's life.
        let _ = self.release_refused_guest_upload_suffixes();
        let _ = self.progress_indirect_ranges();
        let _ = self.progress_present_completions(host);
        let retired_batches = self.harvest_retired_batches();
        let recording_cleanups = self.progress_recording_cleanup();
        self.report_pipeline_census();
        if self.terminalize_device_loss() {
            return ReplacementDeviceTickProgress {
                drained,
                cpu_completed,
                recordings_handed_off,
                exec_acceptances,
                timeline_observations: timeline.observed,
                guest_uploads_resumed,
                retired_batches,
                recording_cleanups,
                publications: 0,
            };
        }
        self.retry_publication_retirement();
        let mut publications = 0;
        loop {
            match self.publish_next(host) {
                Ok(Some(published)) => match self.retire_published(published) {
                    Ok(_) => {
                        self.publication_failure = None;
                        publications += 1;
                    }
                    Err(failure) => {
                        self.publication_retirement_failures.push_back(failure);
                        break;
                    }
                },
                Ok(None) => {
                    self.publication_failure = None;
                    break;
                }
                Err(reason) => {
                    self.publication_failure = Some(reason);
                    break;
                }
            }
        }
        ReplacementDeviceTickProgress {
            drained,
            cpu_completed,
            recordings_handed_off,
            exec_acceptances,
            timeline_observations: timeline.observed,
            guest_uploads_resumed,
            retired_batches,
            recording_cleanups,
            publications,
        }
    }

    pub fn progress_cpu_packets(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    ) -> usize {
        let ids = self.cpu.transaction_ids();
        let mut progressed = 0;
        for transaction in ids {
            // Every CPU failure retains the exact phase input. A later tick is
            // the retry boundary for unavailable guest memory or host service;
            // the transition itself rejects states that are not retryable.
            let _ = self.cpu.retry_failure(transaction);
            if matches!(
                self.cpu.progress(
                    &mut self.runtime,
                    host,
                    self.transport.page_shift(),
                    self.transport.registers().gfx.version,
                    transaction,
                ),
                Some(ReplacementCpuProgress::Published { .. })
            ) {
                progressed += 1;
            }
        }
        self.absorb_publications();
        progressed
    }

    pub fn progress_present_preparations(&mut self) -> Vec<ReplacementPresentPreparationProgress> {
        let ids = self.present.transaction_ids();
        ids.into_iter()
            .filter_map(|transaction| {
                // Native preparation may have failed because an earlier
                // transaction had not made the backing current yet. Restore
                // only the failure's own predecessor before trying again.
                let _ = self.present.retry_preparation_failure(transaction);
                self.present
                    .progress_preparation(&mut self.runtime, transaction)
            })
            .collect()
    }

    pub unsafe fn progress_present_queues(
        &mut self,
    ) -> Vec<ReplacementPresentQueueCoordinatorProgress> {
        let ids = self.present.transaction_ids();
        let mut progress = Vec::with_capacity(ids.len());
        for transaction in ids {
            #[cfg(feature = "host-window")]
            let has_window = self.runtime.session().vulkan().window_snapshot().is_some();
            #[cfg(not(feature = "host-window"))]
            let has_window = false;
            let prepared = if has_window {
                #[cfg(feature = "host-window")]
                {
                    unsafe {
                        self.present
                            .prepare_window_queue(&mut self.runtime, transaction)
                    }
                }
                #[cfg(not(feature = "host-window"))]
                {
                    unreachable!("a host window cannot exist without host-window support")
                }
            } else {
                unsafe {
                    self.present
                        .prepare_console_queue(&self.runtime, transaction)
                }
            };
            let step = match prepared {
                Some(ReplacementPresentQueueCoordinatorProgress::Prepared) => {
                    self.present.enqueue_queue(&self.runtime, transaction)
                }
                Some(ReplacementPresentQueueCoordinatorProgress::WrongStage) => {
                    match self.present.poll_queue(&mut self.runtime, transaction) {
                        Some(ReplacementPresentQueueCoordinatorProgress::WrongStage) => self
                            .present
                            .retry_acceptance(&mut self.runtime, transaction),
                        step => step,
                    }
                }
                step => step,
            };
            if let Some(step) = step {
                progress.push(step);
            }
        }
        progress
    }

    pub fn progress_present_completions(
        &mut self,
        host: &mut (impl HostMemory + HostControl),
    ) -> Vec<ReplacementPresentCompletionProgress> {
        let ids = self.present.transaction_ids();
        let mut progress = Vec::new();
        for transaction in ids {
            if let Some(point) = self.present.queued_point(transaction) {
                if let Some(retired) = self.retired_batches.iter().find(|retired| {
                    retired.observed.queue == point.queue
                        && retired.observed.completed >= point.value
                }) {
                    if let Some(step) = self.present.observe_timeline(
                        &mut self.runtime,
                        transaction,
                        reims_vgpu_core::QueueTimelinePoint {
                            epoch: point.epoch,
                            queue: retired.observed.queue,
                            value: retired.observed.completed,
                        },
                    ) {
                        if step == ReplacementPresentCompletionProgress::TimelineComplete {
                            self.product_presented = true;
                        }
                        progress.push(step);
                    }
                }
            }
            if self.present.has_console_frame(transaction) {
                match self.next_console_frame_identity() {
                    Ok((mapping_id, generation)) => {
                        let frame = self
                            .present
                            .take_console_frame(transaction)
                            .expect("the completed Present retained this console frame");
                        let width = frame.width;
                        let height = frame.height;
                        let previous = self.console_frames.insert((mapping_id, generation), frame);
                        debug_assert!(previous.is_none());
                        self.console_frame_failure = None;
                        host.enqueue(HostAction::scanout_gen(
                            mapping_id, width, height, generation,
                        ));
                    }
                    Err(reason) => self.console_frame_failure = Some(reason),
                }
            }
            if let Some(step) = self
                .present
                .prepare_notification(&self.runtime, host, transaction)
            {
                if step != ReplacementPresentCompletionProgress::WrongStage {
                    progress.push(step);
                }
            }
            if let Some(step) =
                self.present
                    .apply_notification(&self.runtime, host, &self.transport, transaction)
            {
                if step != ReplacementPresentCompletionProgress::WrongStage {
                    progress.push(step);
                }
            }
            if let Some(step) = self.present.complete(&mut self.runtime, transaction) {
                if step != ReplacementPresentCompletionProgress::WrongStage {
                    progress.push(step);
                }
            }
        }
        self.absorb_publications();
        progress
    }

    /// Settle every admitted EXEC recording and route each owner that became
    /// queue-ready. The count returned is recordings that left this coordinator
    /// owning ready work; a parked recording is progress too, but its owner now
    /// belongs to the runtime epoch and is released by predecessor acceptance.
    pub fn progress_exec_recordings(&mut self) -> usize {
        let ids = self.exec_recordings.transaction_ids();
        let mut handed_off = 0;
        for transaction in ids {
            let Some(disposition) = self.exec_recordings.poll(&mut self.runtime, transaction)
            else {
                continue;
            };
            match disposition {
                ReplacementExecRecordingDisposition::Pending
                | ReplacementExecRecordingDisposition::Parked(_) => {}
                ReplacementExecRecordingDisposition::Ready(ready) => {
                    self.route_ready_recording(ready);
                    handed_off += 1;
                }
                ReplacementExecRecordingDisposition::Failed(refusal) => {
                    let mut fields = vec![
                        ("transaction", transaction.get().to_string()),
                        ("stage", refusal.stage.to_string()),
                    ];
                    if let Some(detail) = refusal.detail {
                        fields.push(("detail", detail));
                    }
                    let diagnostic = ReplacementCoordinatorDiagnostic {
                        slug: "replacement_exec_recording_refused",
                        fields,
                        discriminant: transaction.get(),
                    };
                    crate::observe::Emit::decline(
                        "replacement_exec_recording_progress",
                        &diagnostic,
                    )
                    .fail_once(diagnostic.discriminant);
                }
            }
        }
        handed_off
    }

    pub fn progress_exec_submissions(&mut self) -> Vec<ReplacementExecSubmitCoordinatorProgress> {
        let ids = self.exec_submissions.transaction_ids();
        let mut progress = Vec::with_capacity(ids.len());
        for transaction in ids {
            let step = match self.exec_submissions.submit(&mut self.runtime, transaction) {
                Some(ReplacementExecSubmitCoordinatorProgress::WrongStage) => {
                    self.exec_submissions.poll(&mut self.runtime, transaction)
                }
                step => step,
            };
            if let Some(step) = step {
                if step.is_refusal() {
                    report_coordinator_refusal("replacement_exec_submit", transaction, step);
                }
                progress.push(step);
            }
        }
        progress
    }

    pub fn harvest_exec_acceptances(&mut self) -> usize {
        let mut accepted_count = 0;
        while let Some(failure) = self.accepted_routing_failures.pop_front() {
            match self.route_accepted_exec(failure.accepted) {
                Ok(()) => accepted_count += 1,
                Err(failure) => {
                    self.accepted_routing_failures.push_front(failure);
                    break;
                }
            }
        }
        let ids = self.exec_submissions.transaction_ids();
        for transaction in ids {
            let Some(poll) = self.exec_submissions.take_terminal(transaction) else {
                continue;
            };
            let reims_vgpu_vulkan::replacement_exec_queue::ReplacementExecSubmitPoll::Accepted(
                accepted,
            ) = poll
            else {
                self.exec_submissions.submissions.insert(
                    transaction,
                    ReplacementCoordinatedExecSubmit::Terminal(Box::new(poll)),
                );
                continue;
            };
            if let Err(failure) = self.route_accepted_exec(accepted) {
                self.accepted_routing_failures.push_back(failure);
                continue;
            }
            accepted_count += 1;
        }
        accepted_count
    }

    pub fn progress_guest_uploads(&mut self) -> Vec<ReplacementGuestUploadCoordinatorProgress> {
        while let Some(ready) = self.ready_guest_uploads.pop_front() {
            if let Err((ready, _semantic)) = self.guest_uploads.admit(ready, ()) {
                self.ready_guest_uploads.push_front(ready);
                break;
            }
        }
        let ids = self.guest_uploads.transaction_ids();
        let mut progress = Vec::with_capacity(ids.len());
        for transaction in ids {
            if let Some(step) = self.guest_uploads.progress(&mut self.runtime, transaction) {
                if step.is_refusal() {
                    report_coordinator_refusal("replacement_guest_upload", transaction, step);
                }
                progress.push(step);
            }
        }
        progress
    }

    pub fn resume_completed_guest_uploads(&mut self) -> usize {
        let ids = self.guest_uploads.transaction_ids();
        let mut resumed = 0;
        for transaction in ids {
            let Some(point) = self.guest_uploads.accepted_point(transaction) else {
                continue;
            };
            let Some(retired) = self.retired_batches.iter().find(|retired| {
                retired.observed.queue == point.queue && retired.observed.completed >= point.value
            }) else {
                continue;
            };
            let marker = ReplacementTimelineProgressOwner {
                observed:
                    reims_vgpu_vulkan::replacement_replay::ReplacementObservedTimelineProgress {
                        queue: retired.observed.queue,
                        completed: retired.observed.completed,
                    },
                replay: reims_vgpu_core::ReplayTimelineProgress {
                    completions: Vec::new(),
                    retired_native: Vec::new(),
                },
                resource_completions: Vec::new(),
                retired_recordings: Vec::new(),
            };
            let accepted = self
                .guest_uploads
                .take_accepted(transaction)
                .expect("the accepted point came from this exact upload owner");
            match self
                .runtime
                .resume_guest_upload_after_retirement(*accepted, marker)
            {
                Ok((continuing, mut outputs)) => {
                    if let Some(recording) = outputs.ready_recording.take() {
                        self.recordings_to_cleanup.push_back(recording);
                    }
                    match self.runtime.route_guest_upload_continuation(continuing) {
                        Ok(
                            crate::runtime::replacement_session::ReplacementGuestUploadContinuation::Direct(
                                continuing,
                            ),
                        ) => self.continuing_guest_uploads.push_back(continuing),
                        Ok(
                            crate::runtime::replacement_session::ReplacementGuestUploadContinuation::IndirectRange(
                                continuing,
                            ),
                        ) => self.continuing_indirect_ranges.push_back(continuing),
                        Err(failure) => self.guest_upload_continuation_failures.push_back(failure),
                    }
                    resumed += 1;
                }
                Err(failure) => self.guest_upload_resume_failures.push_back(failure),
            }
        }
        resumed
    }

    /// Give up every guest-upload suffix retained in a terminal resolution
    /// refusal, so one unimplemented case costs the guest that transaction and
    /// not the channel it arrived on.
    ///
    /// A retained refusal holds its domain's submission head, its encoder
    /// continuation and its publication position, so without this every later
    /// transaction on the channel refuses behind it and no fact after it ever
    /// reaches the guest. The refusal itself was already named on the failure
    /// channel by the stage that produced it; this is the release, not the
    /// report.
    pub fn release_refused_guest_upload_suffixes(&mut self) -> usize {
        let mut released = 0;

        while let Some((transaction, failure)) =
            self.guest_upload_suffixes.take_refused_resolution()
        {
            match self.runtime.abandon_guest_upload_suffix(failure.suffix, ()) {
                Ok(published) => {
                    self.publications.enqueue(published);
                    self.abandoned_transactions += 1;
                    released += 1;
                }
                Err(reason) => report_retained_failure_detail(
                    "replacement_guest_upload_suffix_abandonment",
                    transaction,
                    &format!("{reason:?}"),
                ),
            }
        }
        released
    }

    pub fn progress_guest_upload_suffixes(&mut self) -> Vec<ReplacementGuestUploadSuffixProgress> {
        while let Some(continuing) = self.continuing_guest_uploads.pop_front() {
            if let Err(continuing) = self.guest_upload_suffixes.admit(continuing) {
                self.continuing_guest_uploads.push_front(*continuing);
                break;
            }
        }
        let ids = self.guest_upload_suffixes.transaction_ids();
        let mut progress = Vec::with_capacity(ids.len());
        for transaction in ids.iter().copied() {
            if let Some(step) = self
                .guest_upload_suffixes
                .progress(&mut self.runtime, transaction)
            {
                if step.is_refusal() {
                    report_coordinator_refusal(
                        "replacement_guest_upload_suffix",
                        transaction,
                        step,
                    );
                }
                progress.push(step);
            }
        }
        for transaction in ids.iter().copied() {
            let Some(accepted) = self
                .guest_upload_suffixes
                .take_refresh_accepted(transaction)
            else {
                continue;
            };
            if let Err(accepted) = self.guest_uploads.admit_accepted(accepted) {
                self.guest_upload_suffixes
                    .restore_refresh_accepted(accepted)
                    .expect("the refresh owner was just removed from this coordinator");
            }
        }
        for transaction in ids {
            let Some(accepted) = self.guest_upload_suffixes.take_accepted(transaction) else {
                continue;
            };
            if let Err(failure) = self.route_accepted_exec(*accepted) {
                self.accepted_routing_failures.push_back(failure);
            }
        }
        progress
    }

    pub fn progress_indirect_ranges(&mut self) -> Vec<ReplacementIndirectCoordinatorProgress> {
        while let Some(ready) = self.ready_indirect_ranges.pop_front() {
            if let Err((ready, _semantic)) = self.indirects.admit_initial(ready, ()) {
                self.ready_indirect_ranges.push_front(ready);
                break;
            }
        }
        while let Some(continuing) = self.continuing_indirect_ranges.pop_front() {
            if let Err(continuing) = self.indirects.admit_continuing(continuing) {
                self.continuing_indirect_ranges.push_front(*continuing);
                break;
            }
        }
        let ids = self.indirects.transaction_ids();
        let mut progress = Vec::with_capacity(ids.len());
        for transaction in ids.iter().copied() {
            if let Some(step) = self.indirects.progress(&mut self.runtime, transaction) {
                if step.is_refusal() {
                    report_coordinator_refusal("replacement_indirect_range", transaction, step);
                }
                progress.push(step);
            }
        }
        for transaction in ids.iter().copied() {
            let Some(point) = self.indirects.accepted_auxiliary_point(transaction) else {
                continue;
            };
            let Some(batch_index) = self.retired_batches.iter().position(|retired| {
                retired.observed.queue == point.queue && retired.observed.completed >= point.value
            }) else {
                continue;
            };
            let retired =
                self.retired_batches[batch_index].take_retired_indirect_ranges(transaction);
            let accepted = self
                .indirects
                .take_accepted_auxiliary(transaction)
                .expect("the accepted point came from this exact indirect owner");
            match self
                .runtime
                .resume_indirect_range_after_retirement(*accepted, retired)
            {
                Ok((continuing, mut outputs)) => {
                    if let Some(recording) = outputs.ready_recording.take() {
                        self.recordings_to_cleanup.push_back(recording);
                    }
                    if let Err(continuing) = self.indirects.admit_continuing(continuing) {
                        self.continuing_indirect_ranges.push_back(*continuing);
                    }
                }
                Err(failure) => self.indirect_resume_failures.push_back(failure),
            }
        }
        for transaction in ids {
            let Some(accepted) = self.indirects.take_accepted_final(transaction) else {
                continue;
            };
            if let Err(failure) = self.route_accepted_exec(*accepted) {
                self.accepted_routing_failures.push_back(failure);
            }
        }
        progress
    }

    pub fn harvest_retired_batches(&mut self) -> usize {
        let mut retained = std::collections::VecDeque::new();
        let mut harvested = 0;
        while let Some(mut batch) = self.retired_batches.pop_front() {
            let queue = batch.observed.queue;
            let completed = batch.observed.completed;
            if self
                .guest_uploads
                .has_accepted_at_or_before(queue, completed)
                || self
                    .indirects
                    .has_accepted_auxiliary_at_or_before(queue, completed)
            {
                retained.push_back(batch);
                continue;
            }
            harvested += 1;
            self.recordings_to_cleanup
                .extend(std::mem::take(&mut batch.retired_recordings));
        }
        self.retired_batches = retained;
        harvested
    }

    pub fn progress_recording_cleanup(&mut self) -> usize {
        while let Some(failure) = self.recording_cleanup_dispatch_failures.pop_front() {
            self.recordings_to_cleanup.push_front(*failure.recording);
        }
        while let Some(recording) = self.recordings_to_cleanup.pop_front() {
            match self.runtime.recycle_retired_recording(recording) {
                Ok(pending) => self.pending_recording_cleanups.push_back(pending),
                Err(failure) => {
                    self.recording_cleanup_dispatch_failures.push_front(failure);
                    break;
                }
            }
        }
        let pending = std::mem::take(&mut self.pending_recording_cleanups);
        let mut completed = 0;
        for cleanup in pending {
            match cleanup.try_complete() {
                reims_vgpu_vulkan::replacement_replay::ReplacementRecordingCleanupPoll::Pending(
                    pending,
                ) => self.pending_recording_cleanups.push_back(pending),
                reims_vgpu_vulkan::replacement_replay::ReplacementRecordingCleanupPoll::Completed(
                    Ok(()),
                ) => completed += 1,
                reims_vgpu_vulkan::replacement_replay::ReplacementRecordingCleanupPoll::Completed(
                    Err(reason),
                ) => self
                    .recording_cleanup_completion_failures
                    .push_back(reason),
            }
        }
        completed
    }

    fn admit_cpu_and_progress(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
        admitted: ReplacementAdmittedCpuPacket<()>,
    ) -> Result<(), Box<ReplacementCpuCoordinatorAdmissionFailure<()>>> {
        let transaction = admitted.transaction();
        self.cpu.admit(admitted, ())?;
        let _ = self.cpu.progress(
            &mut self.runtime,
            host,
            self.transport.page_shift(),
            self.transport.registers().gfx.version,
            transaction,
        );
        self.absorb_publications();
        Ok(())
    }

    fn admit_child_owner(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
        admitted: crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket<()>,
    ) {
        match ReplacementAdmittedCpuPacket::try_from(admitted) {
            Ok(cpu) => {
                if let Err(failure) = self.admit_cpu_and_progress(host, cpu) {
                    self.drain_failures.push_back(Box::new(
                        ReplacementDeviceDrainFailure::CpuCoordinator(failure),
                    ));
                }
            }
            Err(ReplacementNonCpuChildPacket::Present(present)) => {
                if let Err(failure) = self.present.admit(present, ()) {
                    self.drain_failures.push_back(Box::new(
                        ReplacementDeviceDrainFailure::PresentCoordinator(failure),
                    ));
                }
            }
        }
    }

    /// Refuse one child packet whose reason the guest has already settled,
    /// consuming its ring lease so the channel advances past it.
    ///
    /// Both phases reach here: an ingress admission that named an object the
    /// guest's own table does not hold, and a deferred dispatch that named a
    /// case this backend has declared it does not build.
    ///
    /// This is the ingress counterpart of giving up a refused transaction. The
    /// packet is lost and the guest is told what was lost, once, by name -- the
    /// alternative was re-offering it on every tick for the life of the device,
    /// which cost the guest the whole channel rather than one packet and left
    /// nothing in any census to say so.
    fn refuse_child_packet(
        &mut self,
        host: &mut impl HostMemory,
        refused: ReplacementRefusedChildPacket,
    ) -> usize {
        let ReplacementRefusedChildPacket { lease, detail } = refused;
        let channel = lease.channel;
        let opcode = lease.packet.opcode;
        match self.transport.commit_child_packet(host, lease) {
            Ok(()) => {
                self.refused_child_packets += 1;
                crate::observe::fail(format!(
                    "replacement_child_packet_refused channel={} opcode={opcode:#x} {detail}",
                    channel.get()
                ));
                1
            }
            Err(failure) => {
                crate::observe::fail(format!(
                    "replacement_child_packet_refusal_uncommitted channel={} opcode={opcode:#x} reason={:?} {detail}",
                    channel.get(),
                    failure.reason
                ));
                0
            }
        }
    }

    fn retry_blocked_drain(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    ) -> Result<(usize, usize, usize), ()> {
        let Some(failure) = self.blocked_drains.pop_front() else {
            return Ok((0, 0, 0));
        };
        let (admitted, channel) = match *failure {
            ReplacementBlockedDrain::RootIngress(failure) => {
                match retry_root_packet_lease_failure(
                    &mut self.runtime,
                    &mut self.transport,
                    *failure,
                ) {
                    Ok(admitted) => {
                        if let Err(failure) = self.admit_cpu_and_progress(host, admitted) {
                            self.drain_failures.push_back(Box::new(
                                ReplacementDeviceDrainFailure::CpuCoordinator(failure),
                            ));
                        }
                        self.transport.request_root();
                        return Ok((1, 0, 0));
                    }
                    Err(failure) => {
                        self.blocked_drains
                            .push_back(Box::new(ReplacementBlockedDrain::RootIngress(failure)));
                        return Err(());
                    }
                }
            }
            ReplacementBlockedDrain::ChildIngress(failure) => {
                match retry_child_packet_lease_failure(
                    &mut self.runtime,
                    &mut self.transport,
                    host,
                    *failure,
                ) {
                    Ok((ReplacementChildPacketLeaseIngress::Admitted(admitted), channel)) => {
                        self.admit_child_owner(host, admitted);
                        (None, channel)
                    }
                    Ok((
                        ReplacementChildPacketLeaseIngress::Deferred { transport, lease },
                        channel,
                    )) => {
                        match dispatch_deferred_child_packet(
                            &mut self.runtime,
                            &mut self.transport,
                            host,
                            transport,
                            lease,
                        ) {
                            Ok(admitted) => (Some(admitted), channel),
                            Err(failure) => {
                                self.blocked_drains.push_back(Box::new(
                                    ReplacementBlockedDrain::DeferredChild(failure),
                                ));
                                return Err(());
                            }
                        }
                    }
                    Err(failure) => match failure.into_refusal() {
                        Ok(refused) => {
                            return Ok((0, 0, self.refuse_child_packet(host, refused)));
                        }
                        Err(failure) => {
                            self.blocked_drains.push_back(Box::new(
                                ReplacementBlockedDrain::ChildIngress(failure),
                            ));
                            return Err(());
                        }
                    },
                }
            }
            ReplacementBlockedDrain::DeferredChild(failure) => {
                match retry_deferred_child_dispatch_failure(
                    &mut self.runtime,
                    &mut self.transport,
                    host,
                    *failure,
                ) {
                    Ok((admitted, channel)) => (Some(admitted), channel),
                    Err(failure) => match failure.into_refusal() {
                        Ok(refused) => {
                            return Ok((0, 0, self.refuse_child_packet(host, refused)));
                        }
                        Err(failure) => {
                            self.blocked_drains.push_back(Box::new(
                                ReplacementBlockedDrain::DeferredChild(failure),
                            ));
                            return Err(());
                        }
                    },
                }
            }
            ReplacementBlockedDrain::Mapper(failure) => {
                match retry_mapper_entry_dispatch_failure(
                    &mut self.runtime,
                    &mut self.transport,
                    host,
                    *failure,
                ) {
                    Ok(effect) => {
                        self.transport.request_iosfc();
                        return Ok((0, 0, usize::from(effect.is_some())));
                    }
                    Err(failure) => {
                        self.blocked_drains
                            .push_back(Box::new(ReplacementBlockedDrain::Mapper(failure)));
                        return Err(());
                    }
                }
            }
            ReplacementBlockedDrain::IosfcWrite(failure) => {
                match retry_iosfc_write_dispatch_failure(
                    &mut self.runtime,
                    &mut self.transport,
                    host,
                    *failure,
                ) {
                    Ok(completed) => {
                        return Ok((0, 0, completed.len()));
                    }
                    Err(failure) => {
                        self.blocked_drains
                            .push_back(Box::new(ReplacementBlockedDrain::IosfcWrite(failure)));
                        return Err(());
                    }
                }
            }
        };
        if let Some(admitted) = admitted {
            match admitted {
                ReplacementDeferredChildAdmission::Cpu(admitted) => {
                    if let Err(failure) = self.admit_cpu_and_progress(host, admitted) {
                        self.drain_failures.push_back(Box::new(
                            ReplacementDeviceDrainFailure::CpuCoordinator(failure),
                        ));
                    }
                }
                ReplacementDeferredChildAdmission::Exec(pending) => {
                    if let Err(failure) = self.exec_recordings.admit(pending) {
                        self.drain_failures.push_back(Box::new(
                            ReplacementDeviceDrainFailure::ExecRecordingCoordinator(failure),
                        ));
                    }
                }
            }
        }
        let requested = self.transport.request_child(channel);
        debug_assert!(requested, "a retained child lease has a child channel");
        Ok((0, 1, 0))
    }

    pub fn iosfc_write(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
        offset: u64,
        data: u64,
    ) {
        if self.terminalize_device_loss() {
            return;
        }
        let Some(effect) = self.transport.iosfc_write(offset, data) else {
            return;
        };
        if let Err(failure) =
            dispatch_iosfc_write_effect(&mut self.runtime, &mut self.transport, host, effect)
        {
            self.blocked_drains
                .push_back(Box::new(ReplacementBlockedDrain::IosfcWrite(failure)));
        }
    }

    pub fn drain(
        &mut self,
        host: &mut (impl HostMemory + crate::runtime::host::HostOps),
    ) -> ReplacementDeviceDrainProgress {
        if self.terminalize_device_loss() {
            return ReplacementDeviceDrainProgress::default();
        }
        let blocked_before = self.blocked_drains.len();
        let mut progress = ReplacementDeviceDrainProgress::default();
        for _ in 0..blocked_before {
            match self.retry_blocked_drain(host) {
                Ok((root, child, mapper)) => {
                    progress.root_packets += root;
                    progress.child_packets += child;
                    progress.mapper_entries += mapper;
                }
                Err(()) => {
                    self.blocked_drain_retries = self.blocked_drain_retries.saturating_add(1);
                    progress.failures += 1;
                }
            }
        }
        let work = self.transport.take_work();
        let failures_before = self.drain_failures.len();
        let root_blocked = self
            .blocked_drains
            .iter()
            .any(|blocked| matches!(blocked.as_ref(), ReplacementBlockedDrain::RootIngress(_)));
        if work.root && !root_blocked {
            loop {
                let lease = match self.transport.read_root_packet(host) {
                    Ok(Some(lease)) => {
                        self.root_read_failure = None;
                        lease
                    }
                    Ok(None) => {
                        self.root_read_failure = None;
                        break;
                    }
                    Err(reason) => {
                        if let Some(diagnostic) = replacement_root_read_diagnostic(reason) {
                            let discriminant = diagnostic.discriminant;
                            crate::observe::Emit::decline(
                                "replacement_device_root_read",
                                &diagnostic,
                            )
                            .fail_once(discriminant);
                        }
                        self.root_read_failure = Some(reason);
                        self.transport.request_root();
                        progress.failures += 1;
                        break;
                    }
                };
                match admit_root_packet_lease(&mut self.runtime, &mut self.transport, lease) {
                    Ok(admitted) => {
                        progress.root_packets += 1;
                        if let Err(failure) = self.admit_cpu_and_progress(host, admitted) {
                            self.drain_failures.push_back(Box::new(
                                ReplacementDeviceDrainFailure::CpuCoordinator(failure),
                            ));
                            progress.failures += 1;
                            break;
                        }
                    }
                    Err(failure) => {
                        self.blocked_drains
                            .push_back(Box::new(ReplacementBlockedDrain::RootIngress(failure)));
                        progress.failures += 1;
                        break;
                    }
                }
            }
        }
        for raw_channel in 1..reims_vgpu_core::MAX_CHANNELS as u32 {
            if work.children & (1u32 << raw_channel) == 0 {
                continue;
            }
            let channel = reims_vgpu_protocol::ChannelId::new(raw_channel);
            if self
                .blocked_drains
                .iter()
                .any(|blocked| blocked.child_channel() == Some(channel))
            {
                continue;
            }
            loop {
                let lease = match self.transport.read_child_packet(host, channel) {
                    Ok(Some(lease)) => {
                        self.child_read_failures.remove(&channel);
                        lease
                    }
                    Ok(None) => {
                        self.child_read_failures.remove(&channel);
                        break;
                    }
                    Err(reason) => {
                        if let Some(diagnostic) = replacement_child_read_diagnostic(channel, reason)
                        {
                            let discriminant = diagnostic.discriminant;
                            crate::observe::Emit::decline(
                                "replacement_device_child_read",
                                &diagnostic,
                            )
                            .fail_once(discriminant);
                        }
                        self.child_read_failures.insert(channel, reason);
                        let requested = self.transport.request_child(channel);
                        debug_assert!(requested, "the drain iterates only child channels");
                        progress.failures += 1;
                        break;
                    }
                };
                match admit_child_packet_lease(&mut self.runtime, &mut self.transport, host, lease)
                {
                    Ok(ReplacementChildPacketLeaseIngress::Admitted(admitted)) => {
                        progress.child_packets += 1;
                        self.admit_child_owner(host, admitted);
                    }
                    Ok(ReplacementChildPacketLeaseIngress::Deferred { transport, lease }) => {
                        match dispatch_deferred_child_packet(
                            &mut self.runtime,
                            &mut self.transport,
                            host,
                            transport,
                            lease,
                        ) {
                            Ok(ReplacementDeferredChildAdmission::Cpu(admitted)) => {
                                progress.child_packets += 1;
                                if let Err(failure) = self.admit_cpu_and_progress(host, admitted) {
                                    self.drain_failures.push_back(Box::new(
                                        ReplacementDeviceDrainFailure::CpuCoordinator(failure),
                                    ));
                                    progress.failures += 1;
                                    break;
                                }
                            }
                            Ok(ReplacementDeferredChildAdmission::Exec(pending)) => {
                                progress.child_packets += 1;
                                if let Err(failure) = self.exec_recordings.admit(pending) {
                                    self.drain_failures.push_back(Box::new(
                                        ReplacementDeviceDrainFailure::ExecRecordingCoordinator(
                                            failure,
                                        ),
                                    ));
                                    progress.failures += 1;
                                    break;
                                }
                            }
                            Err(failure) => {
                                self.blocked_drains.push_back(Box::new(
                                    ReplacementBlockedDrain::DeferredChild(failure),
                                ));
                                progress.failures += 1;
                                break;
                            }
                        }
                    }
                    Err(failure) => {
                        self.blocked_drains
                            .push_back(Box::new(ReplacementBlockedDrain::ChildIngress(failure)));
                        progress.failures += 1;
                        break;
                    }
                }
            }
        }
        let iosfc_blocked = self.blocked_drains.iter().any(|blocked| {
            matches!(
                blocked.as_ref(),
                ReplacementBlockedDrain::Mapper(_) | ReplacementBlockedDrain::IosfcWrite(_)
            )
        });
        if work.iosfc && !iosfc_blocked {
            loop {
                match dispatch_next_mapper_entry(&mut self.runtime, &mut self.transport, host) {
                    Ok(Some(_)) => progress.mapper_entries += 1,
                    Ok(None) => break,
                    Err(failure) => {
                        self.blocked_drains
                            .push_back(Box::new(ReplacementBlockedDrain::Mapper(failure)));
                        progress.failures += 1;
                        break;
                    }
                }
            }
        }
        progress.failures = progress.failures.max(
            self.drain_failures.len() - failures_before
                + self.blocked_drains.len().saturating_sub(blocked_before),
        );
        progress
    }

    pub fn guest_reset(
        &mut self,
        next: reims_vgpu_protocol::SessionGenerationId,
    ) -> Result<
        crate::runtime::replacement_session::ReplacementRuntimeResetEffect,
        ReplacementDeviceResetError,
    > {
        let phases = self.owned_phase_count();
        let transactions = self.live_transactions();
        let publications = self.pending_publications();
        if phases != 0 || transactions != 0 || publications != 0 {
            return Err(ReplacementDeviceResetError::OutstandingOwnership {
                phases,
                transactions,
                publications,
            });
        }
        let effect = self
            .runtime
            .guest_reset(next)
            .map_err(ReplacementDeviceResetError::Generation)?;
        self.transport.reset();
        self.product_presented = false;
        self.display_online_failure = None;
        self.console_frame_failure = None;
        Ok(effect)
    }

    pub fn platform_reset(
        &mut self,
        next: reims_vgpu_protocol::SessionGenerationId,
    ) -> Result<
        crate::runtime::replacement_session::ReplacementPlatformResetEffect,
        ReplacementDeviceResetError,
    > {
        if self.device_loss_effect.is_some()
            || self.vulkan_state() != reims_vgpu_core::VulkanDeviceEpochState::Active
        {
            return Err(ReplacementDeviceResetError::DeviceLost);
        }
        let effect = self
            .runtime
            .platform_reset(next)
            .map_err(ReplacementDeviceResetError::Platform)?;
        self.clear_aggregate_owners();
        Ok(effect)
    }
}

#[derive(Debug)]
pub(crate) enum ReplacementDeviceResetError {
    DeviceAbsent,
    DeviceLost,
    OutstandingOwnership {
        phases: usize,
        transactions: usize,
        publications: usize,
    },
    Generation(crate::runtime::replacement_session::ReplacementRuntimeResetError),
    Platform(crate::runtime::replacement_session::ReplacementPlatformResetError),
}

#[derive(Debug)]
pub(crate) struct ReplacementPublishedFactRetirementFailure<Semantic> {
    pub reason: reims_vgpu_core::TransactionRuntimeError,
    pub published: ReplacementHostPublishedFact<Semantic>,
}

pub(crate) fn retire_host_published_fact<Semantic: Clone>(
    runtime: &mut ReplacementRuntimeSession<Semantic>,
    published: ReplacementHostPublishedFact<Semantic>,
) -> Result<
    reims_vgpu_core::PublishedFact<Semantic>,
    Box<ReplacementPublishedFactRetirementFailure<Semantic>>,
> {
    match runtime
        .execution_mut()
        .runtime_mut()
        .retire_transaction(published.0.transaction)
    {
        Ok(()) => Ok(published.0),
        Err(reason) => Err(Box::new(ReplacementPublishedFactRetirementFailure {
            reason,
            published,
        })),
    }
}

/// Publish one already ordered semantic fact to the guest transport.
///
/// Native/resource completion and query writes have already settled before
/// this boundary. The completion word is stored before the interrupt-status
/// bit and host IRQ action, preserving the guest's release fence.
pub(crate) fn publish_ordered_fact_to_host<Semantic>(
    host: &mut (impl HostMemory + HostControl),
    interrupt_status: &std::sync::atomic::AtomicU32,
    stamp_page: ReplacementStampPage,
    fact: reims_vgpu_core::PublishedFact<Semantic>,
) -> Result<
    ReplacementHostPublishedFact<Semantic>,
    Box<ReplacementPublishedFactHostFailure<Semantic>>,
> {
    let Some(stamp) = fact.completion_stamp else {
        return Ok(ReplacementHostPublishedFact(fact));
    };
    if stamp_page.base_pfn == 0 {
        return Err(Box::new(ReplacementPublishedFactHostFailure {
            reason: ReplacementPublishedFactHostError::StampPageUnavailable,
            fact,
        }));
    }
    let Some(page_bytes) = 1u64.checked_shl(stamp_page.page_shift) else {
        return Err(Box::new(ReplacementPublishedFactHostFailure {
            reason: ReplacementPublishedFactHostError::InvalidPageShift(stamp_page.page_shift),
            fact,
        }));
    };
    let Some(offset) = reims_vgpu_core::completion_stamp_slot_offset(stamp.slot, page_bytes) else {
        return Err(Box::new(ReplacementPublishedFactHostFailure {
            reason: ReplacementPublishedFactHostError::SlotPastPage {
                slot: stamp.slot,
                page_bytes,
            },
            fact,
        }));
    };
    let Some(gpa) = u64::from(stamp_page.base_pfn)
        .checked_mul(page_bytes)
        .and_then(|base| base.checked_add(offset))
    else {
        return Err(Box::new(ReplacementPublishedFactHostFailure {
            reason: ReplacementPublishedFactHostError::AddressOverflow,
            fact,
        }));
    };
    if let Err(reason) = host.write_gpa(gpa, &stamp.value.to_le_bytes()) {
        return Err(Box::new(ReplacementPublishedFactHostFailure {
            reason: ReplacementPublishedFactHostError::Memory(reason),
            fact,
        }));
    }
    let bit = 1u32 << (stamp.slot % u32::BITS);
    interrupt_status.fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
    host.schedule_bh();
    Ok(ReplacementHostPublishedFact(fact))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_upload_suffix_waiting_for_a_content_producer_is_retryable() {
        assert!(!ReplacementGuestUploadSuffixProgress::WaitingForProducer.is_refusal());
        assert!(ReplacementGuestUploadSuffixProgress::FailedPreparation.is_refusal());
    }

    #[test]
    fn one_completion_pass_does_not_hide_a_later_producer_behind_its_consumer() {
        let mut producer_completed = false;
        let facts = std::collections::VecDeque::from(["consumer", "producer"]);
        let (pending, published, failures) = attempt_each_once(facts, |fact| match fact {
            "consumer" if !producer_completed => Err(("not ready", fact)),
            "producer" => {
                producer_completed = true;
                Ok(vec![fact])
            }
            _ => Ok(vec![fact]),
        });

        assert_eq!(published, ["producer"]);
        assert_eq!(failures, ["not ready"]);
        assert_eq!(pending, ["consumer"]);
        let (pending, published, failures) =
            attempt_each_once(pending, |fact| Ok::<_, (&str, &str)>(vec![fact]));
        assert!(pending.is_empty());
        assert_eq!(published, ["consumer"]);
        assert!(failures.is_empty());
    }

    #[test]
    fn object_apply_diagnostic_reports_identity_and_typed_reason_only() {
        let diagnostic = replacement_loaded_object_apply_refusal_diagnostic(
            reims_vgpu_protocol::ObjectTableRef::new(11),
            &crate::runtime::replacement_session::ReplacementLoadedObjectApplyRefusal::IOSurfacePlaneView(
                crate::runtime::replacement_object_lifecycle::ReplacementIOSurfacePlaneViewRefusal::SurfaceUnavailable(1),
            ),
        );
        assert_eq!(
            diagnostic,
            "object=11 reason=IOSurfacePlaneView(SurfaceUnavailable(1))"
        );
    }
    use crate::runtime::host::HostActionKind;
    use reims_vgpu_protocol::{QueueOwnerId, SessionGenerationId, SessionId, VulkanDeviceEpochId};

    fn packet(
        opcode: u16,
        payload: impl Into<Vec<u8>>,
        completion_stamp: u32,
    ) -> crate::runtime::fifo_packet::Packet {
        crate::runtime::fifo_packet::Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: crate::model::PACKET_HEADER_LEN,
            completion_stamp,
            payload: payload.into(),
            next_head: crate::model::PACKET_HEADER_LEN,
        }
    }

    fn runtime() -> Option<ReplacementRuntimeSession<()>> {
        ReplacementRuntimeSession::create(
            SessionId::new(1),
            SessionGenerationId::new(1),
            VulkanDeviceEpochId::new(1),
            QueueOwnerId::new(1),
            1,
        )
        .ok()
    }

    #[test]
    fn first_device_tick_connects_pipeline_worker_completion_to_the_host_wake() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        let _ = device.tick(&mut host);

        let pipeline = reims_vgpu_protocol::ResourceId::new(1, 1);
        device
            .runtime
            .session()
            .declare_render(
                pipeline,
                crate::runtime::replacement_session::RenderPipelineContract {
                    descriptor: std::sync::Arc::new(
                        reims_vgpu_protocol::RenderPipelineDescriptor::default(),
                    ),
                    vertex_library: std::sync::Arc::from([1, 2, 3, 4]),
                    fragment_library: std::sync::Arc::from([5, 6, 7, 8]),
                },
            )
            .unwrap();
        device
            .runtime
            .session()
            .schedule_render_translation(pipeline, 1)
            .unwrap();
        for _ in 0..100_000 {
            if host.worker_wake_count() != 0 {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(host.worker_wake_count(), 1);
    }

    fn write_mapper_backing_fixture(
        host: &mut crate::runtime::host::FakeHost,
        internal: u64,
        mapper: u64,
        mapping_id: u32,
    ) {
        use crate::runtime::host::HostMemory;
        use reims_vgpu_paging::mapper as paging;

        host.map_range(internal, 0x4000, 0);
        let descriptor = internal + 0x1000;
        let page_owner = internal + 0x2000;
        let page_table = internal + 0x3000;
        for (address, bytes) in [
            (
                internal + paging::MAPPING_INTERNAL_BACKPTR,
                mapper.to_le_bytes().to_vec(),
            ),
            (
                internal + paging::MAPPING_INTERNAL_ID,
                mapping_id.to_le_bytes().to_vec(),
            ),
            (
                internal + paging::MAPPING_INTERNAL_DESC_PTR,
                descriptor.to_le_bytes().to_vec(),
            ),
            (
                internal + paging::MAPPING_INTERNAL_SIZE,
                paging::MAPPING_INTERNAL_EXPECTED_SIZE
                    .to_le_bytes()
                    .to_vec(),
            ),
            (
                internal + paging::MAPPING_INTERNAL_PAGE_FIELD_48,
                page_owner.to_le_bytes().to_vec(),
            ),
            (
                internal + paging::MAPPING_INTERNAL_PAGE_COUNT,
                2u64.to_le_bytes().to_vec(),
            ),
            (
                page_owner + paging::MAPPING_PAGE_TABLE_FROM_F48,
                page_table.to_le_bytes().to_vec(),
            ),
        ] {
            host.write_gpa(address, &bytes).unwrap();
        }
        host.write_gpa(page_table, &5u32.to_le_bytes()).unwrap();
        host.write_gpa(page_table + 4, &9u32.to_le_bytes()).unwrap();
        let mut descriptor_bytes = [0u8; reims_vgpu_protocol::DEVICE_DESC_LEN];
        reims_vgpu_core::endian::st32(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_PIXEL_FORMAT..],
            u32::from(reims_vgpu_protocol::metal_pixel::MTL_FORMAT_BGRA8_UNORM),
        );
        reims_vgpu_core::endian::st32(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_ALLOC_SIZE..],
            0x8000,
        );
        reims_vgpu_core::endian::st64(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_DIMS..],
            (64u64 << 8) | (32u64 << 40),
        );
        reims_vgpu_core::endian::st32(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_BPR..],
            256,
        );
        reims_vgpu_core::endian::st16(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_BPE..],
            4,
        );
        host.write_gpa(descriptor, &descriptor_bytes).unwrap();
    }

    fn child_lease(
        channel: reims_vgpu_protocol::ChannelId,
        opcode: u16,
        payload: &[u8],
        completion: u32,
    ) -> (
        crate::runtime::replacement_transport::ReplacementTransportOwner,
        crate::runtime::host::FakeHost,
        crate::runtime::replacement_transport::ReplacementChildPacketLease,
        u64,
    ) {
        use crate::runtime::host::HostMemory;

        let shift = crate::model::PAGE_SHIFT_X86;
        let page_size = 1u64 << shift;
        let mut transport =
            crate::runtime::replacement_transport::ReplacementTransportOwner::new(shift).unwrap();
        transport.registers_mut().gfx.root_page = 2;
        let registers_gpa =
            2 * page_size + crate::model::child_reg_block_offset(channel.get()).unwrap();
        let list_gpa = 3 * page_size;
        let ring_gpa = 5 * page_size;
        let total = crate::model::PACKET_HEADER_LEN + u32::try_from(payload.len()).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        host.map_range(2 * page_size, page_size as usize, 0);
        host.map_range(list_gpa, page_size as usize, 0);
        host.map_range(ring_gpa, page_size as usize, 0);
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_TAIL,
            &total.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_STAMP_INDEX,
            &channel.get().to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_BASE_PFN,
            &3u32.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(list_gpa, &5u32.to_le_bytes()).unwrap();
        let mut bytes = vec![0; total as usize];
        reims_vgpu_core::endian::st16(&mut bytes, opcode);
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_TOTAL_SIZE..], total);
        reims_vgpu_core::endian::st32(
            &mut bytes[crate::model::PACKET_COMPLETION_STAMP..],
            completion,
        );
        bytes[crate::model::PACKET_HEADER_LEN as usize..].copy_from_slice(payload);
        host.write_gpa(ring_gpa, &bytes).unwrap();
        let lease = transport
            .read_child_packet(&host, channel)
            .unwrap()
            .unwrap();
        (transport, host, lease, registers_gpa)
    }

    #[test]
    fn query_bytes_land_before_ordered_completion_becomes_visible() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut host = crate::runtime::host::FakeHost::new();
        let reply_pfn = 19u32;
        let reply_gpa = u64::from(reply_pfn) << crate::model::PAGE_SHIFT_X86;
        host.map_range(reply_gpa, 4096, 0);
        let mut payload = [0u8; 12];
        reims_vgpu_core::endian::st32(&mut payload, u32::MAX);
        reims_vgpu_core::endian::st32(&mut payload[4..], 1);
        reims_vgpu_core::endian::st32(&mut payload[8..], reply_pfn);
        let admitted = crate::runtime::replacement_fifo_control::admit_replacement_root_packet(
            &mut runtime,
            packet(crate::model::ROOT_OP_DEVICE_INFO_TAHOE, payload, 7),
        )
        .unwrap();
        let admitted = ReplacementAdmittedCpuPacket::from(admitted);
        let applied = apply_ready_cpu_packet(
            &mut runtime,
            crate::model::PAGE_SHIFT_X86,
            u32::MAX,
            admitted,
        )
        .unwrap();
        assert_eq!(host.get_u32(reply_gpa), 0);
        let applied = apply_cpu_host_effect(
            &mut runtime,
            &mut host,
            crate::model::PAGE_SHIFT_X86,
            applied,
        )
        .unwrap();
        assert_ne!(host.get_u32(reply_gpa), 0);
        let published = complete_host_applied_cpu_packet(&mut runtime, applied, ()).unwrap();
        assert_eq!(published.len(), 1);
        assert_eq!(
            published[0].completion_stamp,
            Some(reims_vgpu_core::CompletionStamp::new(0, 7))
        );
    }

    #[test]
    fn cursor_host_action_is_emitted_once_between_apply_and_completion() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let mut payload = [0u8; 8];
        reims_vgpu_core::endian::st32(&mut payload, 0);
        reims_vgpu_core::endian::st32(&mut payload[4..], 0);
        let admitted =
            crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
                &mut runtime,
                channel,
                packet(crate::model::CHILD_OP_CURSOR_SHOW, payload, 8),
            )
            .unwrap();
        let admitted = ReplacementAdmittedCpuPacket::try_from(admitted).unwrap();
        let applied = apply_ready_cpu_packet(
            &mut runtime,
            crate::model::PAGE_SHIFT_X86,
            u32::MAX,
            admitted,
        )
        .unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        let applied = apply_cpu_host_effect(
            &mut runtime,
            &mut host,
            crate::model::PAGE_SHIFT_X86,
            applied,
        )
        .unwrap();
        assert_eq!(host.action_count(HostActionKind::CursorUpdate), 1);
        let published = complete_host_applied_cpu_packet(&mut runtime, applied, ()).unwrap();
        assert_eq!(published.len(), 1);
        assert_eq!(host.action_count(HostActionKind::CursorUpdate), 1);
    }

    #[test]
    fn ordered_fact_stores_its_stamp_before_exposing_the_interrupt() {
        let mut host = crate::runtime::host::FakeHost::new();
        let page = ReplacementStampPage {
            base_pfn: 23,
            page_shift: crate::model::PAGE_SHIFT_X86,
        };
        let base = u64::from(page.base_pfn) << page.page_shift;
        host.map_range(base, 4096, 0);
        let interrupt = std::sync::atomic::AtomicU32::new(0);
        let fact = reims_vgpu_core::PublishedFact {
            transaction: reims_vgpu_protocol::TransactionId::new(9),
            position: reims_vgpu_core::PublicationPosition {
                domain: reims_vgpu_protocol::PublicationDomainId::new(3),
                sequence: reims_vgpu_protocol::PublicationSequence::new(4),
            },
            completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(17, 0xfeed_beef)),
            semantic: (),
        };
        publish_ordered_fact_to_host(&mut host, &interrupt, page, fact).unwrap();
        assert_eq!(host.get_u32(base + 17 * 4), 0xfeed_beef);
        assert_eq!(
            interrupt.load(std::sync::atomic::Ordering::Acquire),
            1 << 17
        );
        assert_eq!(host.action_count(HostActionKind::IrqGfxPulse), 1);

        let failure = publish_ordered_fact_to_host(
            &mut host,
            &interrupt,
            page,
            reims_vgpu_core::PublishedFact {
                transaction: reims_vgpu_protocol::TransactionId::new(10),
                position: reims_vgpu_core::PublicationPosition {
                    domain: reims_vgpu_protocol::PublicationDomainId::new(3),
                    sequence: reims_vgpu_protocol::PublicationSequence::new(5),
                },
                completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(1024, 1)),
                semantic: (),
            },
        )
        .unwrap_err();
        assert_eq!(
            failure.reason,
            ReplacementPublishedFactHostError::SlotPastPage {
                slot: 1024,
                page_bytes: 4096,
            }
        );
        assert_eq!(host.action_count(HostActionKind::IrqGfxPulse), 1);
    }

    /// A physical replacement revokes the backing's construction-designated
    /// execution object, and the contract says a subsequent materialization
    /// installs a fresh representation identity. Nothing performed that
    /// materialization where the replacement landed: one EXEC ingress route
    /// carried a repair keyed to its own refusal shape, so the guest-upload
    /// suffix and indirect-range routes met a revoked backing as a terminal
    /// refusal holding their whole recorded chain.
    /// A plane of a registered surface reaches the late repair as a backing,
    /// and the repair has to know what class of storage that is.
    ///
    /// The object-ready route materializes from the *resources* a packet
    /// declares, so it is the only route that can tell a plane view from a
    /// linear allocation. A backing that arrives at the late repair came from
    /// somewhere else -- an earlier packet's declaration, or a replacement --
    /// and the storage node is the only thing left that says which class it is.
    /// A repair that assumed task-address storage refused every plane by name,
    /// and a missing execution representation holds the whole recorded chain,
    /// so that refusal parked the channel's submission head instead of costing
    /// one command.
    #[test]
    fn the_late_repair_builds_a_registered_surface_plane_and_not_only_a_task_address() {
        use crate::runtime::host::HostMemory;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

        let Some(mut runtime) = runtime() else {
            return;
        };
        let shift = crate::model::PAGE_SHIFT_X86;
        let task = reims_vgpu_protocol::TaskId::new(11);
        runtime.define_task(task, 0x40_0000, 2).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        let directory = 2u64 << shift;
        let root = 3u64 << shift;
        host.map_range(directory, 1usize << shift, 0);
        host.map_range(root, 1usize << shift, 0);
        let mut directory_bytes = [0u8; 8];
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_ROOT_PFN as usize..], 3);
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(directory, &directory_bytes).unwrap();
        // The surface names guest page 0x200 and spans one page, so the task's
        // page table has to resolve exactly that page.
        let backing_pfn = 0x200u64;
        host.map_range(9u64 << shift, 1usize << shift, 0);
        host.write_gpa(root + backing_pfn * 4, &9u32.to_le_bytes())
            .unwrap();

        let surface_object = 41;
        let mut planes = [reims_vgpu_protocol::SurfaceBackingPlane::default();
            reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_CAP];
        planes[0] = reims_vgpu_protocol::SurfaceBackingPlane {
            offset: 0,
            width: 4,
            height: 4,
            bytes_per_row: 16,
            bytes_per_element: 4,
        };
        crate::runtime::replacement_object_lifecycle::apply_replacement_registered_surface(
            &mut runtime,
            task,
            surface_object,
            reims_vgpu_protocol::ResourceDescriptor::SurfaceBacking(
                reims_vgpu_protocol::SurfaceBackingDescriptor {
                    length: 1 << shift,
                    backing_pfn: backing_pfn as u32,
                    pixel_format: u32::from_be_bytes(*b"BGRA"),
                    plane_count: 1,
                    planes,
                    width: 4,
                    height: 4,
                    bytes_per_row: 16,
                },
            ),
        )
        .unwrap();
        let view =
            crate::runtime::replacement_object_lifecycle::apply_replacement_iosurface_plane_view(
                &mut runtime,
                task,
                42,
                reims_vgpu_protocol::ResourceDescriptor::IOSurfacePlaneView(
                    reims_vgpu_protocol::IOSurfacePlaneViewResourceDescriptor {
                        surface: reims_vgpu_protocol::ObjectTableRef::new(surface_object),
                        owner_task: task,
                        operation_kind: Some(5),
                        operation_length: Some(32),
                        own_ref: Some(reims_vgpu_protocol::ObjectTableRef::new(42)),
                        record_kind: Some(reims_vgpu_protocol::IOSurfacePlaneViewRecordKind::Plane),
                        unidentified_record_flags: 0,
                        view: Some(reims_vgpu_protocol::IOSurfacePlaneViewDescriptor {
                            pixel_format: 80,
                            width: 4,
                            height: 4,
                            depth: 1,
                            plane_index: 0,
                        }),
                        decode_state: reims_vgpu_protocol::IOSurfacePlaneViewDecodeState::Complete,
                    },
                ),
            )
            .unwrap();
        assert!(!runtime.backing_has_execution_representation(view.backing));

        prepare_backing_representation(&mut runtime, &mut host, shift, view.backing).unwrap();
        assert!(runtime.backing_has_execution_representation(view.backing));
    }

    #[test]
    fn a_physical_replacement_reinstalls_the_backing_execution_representation() {
        use crate::runtime::host::HostMemory;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

        let Some(mut runtime) = runtime() else {
            return;
        };
        let shift = crate::model::PAGE_SHIFT_X86;
        let task = reims_vgpu_protocol::TaskId::new(7);
        runtime.define_task(task, 0x10_0000, 2).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        let directory = 2u64 << shift;
        let root = 3u64 << shift;
        let data = 9u64 << shift;
        host.map_range(directory, 1usize << shift, 0);
        host.map_range(root, 1usize << shift, 0);
        host.map_range(data, 1usize << shift, 0);
        let mut directory_bytes = [0u8; 8];
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_ROOT_PFN as usize..], 3);
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(directory, &directory_bytes).unwrap();
        host.write_gpa(root + 0x10 * 4, &9u32.to_le_bytes())
            .unwrap();

        let channel = reims_vgpu_protocol::ChannelId::new(5);
        runtime.define_channel(channel).unwrap();
        let object = reims_vgpu_protocol::ObjectTableRef::new(17);
        let declaration =
            crate::runtime::replacement_object_lifecycle::apply_replacement_linear_resource(
                &mut runtime,
                shift,
                task,
                object.get(),
                reims_vgpu_protocol::ResourceDescriptor::Buffer(
                    reims_vgpu_protocol::BufferDescriptor {
                        allocation_size: 64,
                        handle: 0x10,
                        ..Default::default()
                    },
                ),
            )
            .unwrap();
        prepare_backing_representation(&mut runtime, &mut host, shift, declaration.backing)
            .unwrap();
        let first = runtime
            .execution()
            .resources()
            .execution_representation_id(declaration.backing)
            .unwrap();

        let replacement =
            crate::runtime::replacement_child_packet::admit_replacement_physical_replacement(
                &mut runtime,
                channel,
                Box::default(),
                None,
                crate::runtime::replacement_child_packet::DecodedReplacementPhysicalReplacement {
                    task,
                    object,
                },
            )
            .unwrap();
        let applied = runtime
            .apply_admitted_resource_lifecycle(replacement)
            .unwrap();
        let applied = apply_cpu_host_effect(
            &mut runtime,
            &mut host,
            shift,
            ReplacementAppliedCpuPacket::ResourceLifecycle(applied),
        )
        .unwrap();
        let second = runtime
            .execution()
            .resources()
            .execution_representation_id(declaration.backing)
            .expect("a physical replacement reinstalls the execution representation");
        assert_ne!(first, second);
        complete_host_applied_cpu_packet(&mut runtime, applied, ()).unwrap();
    }

    #[test]
    fn replacement_task_reader_walks_the_exact_declared_address_space() {
        use crate::runtime::host::HostMemory;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

        let Some(mut runtime) = runtime() else {
            return;
        };
        let task = reims_vgpu_protocol::TaskId::new(7);
        runtime.define_task(task, 0x1_0000, 2).unwrap();
        let shift = crate::model::PAGE_SHIFT_ARM64E;
        let mut host = crate::runtime::host::FakeHost::new();
        let directory = 2u64 << shift;
        let root = 3u64 << shift;
        let data = 9u64 << shift;
        host.map_range(directory, 1usize << shift, 0);
        host.map_range(root, 1usize << shift, 0);
        host.map_range(data, 1usize << shift, 0);
        let mut directory_bytes = [0u8; 8];
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_ROOT_PFN as usize..], 3);
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(directory, &directory_bytes).unwrap();
        host.write_gpa(root, &9u32.to_le_bytes()).unwrap();
        host.write_gpa(data + 12, &[1, 2, 3, 4]).unwrap();

        assert_eq!(
            read_replacement_task_bytes(&runtime, &host, shift, task, 12, 4).unwrap(),
            [1, 2, 3, 4]
        );
        assert_eq!(
            read_replacement_task_bytes(
                &runtime,
                &host,
                shift,
                reims_vgpu_protocol::TaskId::new(8),
                12,
                4,
            ),
            Err(
                crate::runtime::replacement_exec_decode::ReplacementIcbCommandMemoryTransportError::Memory(
                    MemError::NoSuchTask,
                )
            )
        );
    }

    /// A recording that becomes queue-ready as a staged guest upload must leave
    /// the exec-recording coordinator owning its upload route. Erasing the
    /// route at this seam strands the owner: the recording keeps reporting
    /// ready while no coordinator holds the work, and every later transaction
    /// in the same source domain queues behind it.
    #[test]
    fn a_queue_ready_guest_upload_recording_routes_into_its_own_coordinator() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(26);
        runtime.define_channel(channel).unwrap();
        let resources = crate::runtime::replacement_session::stage_guest_upload_resources_for_test(
            &mut runtime,
        );
        let Some(staged) = crate::runtime::replacement_session::stage_guest_upload_ingress_for_test(
            &mut runtime,
            resources,
            channel,
            69,
            None,
        ) else {
            return;
        };
        assert!(matches!(
            &staged,
            crate::runtime::replacement_session::PendingReplacementIngressExec::GuestUpload(_)
        ));
        let transaction = staged.transaction();
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device
            .exec_recordings
            .admit(staged)
            .unwrap_or_else(|_| panic!("the recording transaction must be unique"));
        let mut handed_off = false;
        for _ in 0..100_000 {
            if device.progress_exec_recordings() == 1 {
                handed_off = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            handed_off,
            "a queue-ready staged upload must leave the recording coordinator"
        );
        assert_eq!(
            device.exec_recordings.live_recordings(),
            0,
            "no route-erased marker may remain once the owner has left"
        );
        assert_eq!(
            device.exec_submissions.live_submissions(),
            0,
            "a staged upload is not a direct submission"
        );
        assert!(device.ready_guest_uploads.is_empty());
        assert_eq!(device.guest_uploads.transaction_ids(), vec![transaction]);
    }

    /// When a recorded owner parks behind a predecessor, ownership moves into
    /// the runtime epoch and nothing may remain in the recording coordinator.
    /// Retaining a marker there keeps the transaction live forever and holds
    /// every later transaction in its source domain behind it. Accepting the
    /// predecessor must then release the parked owner exactly once, into the
    /// coordinator its own route names.
    #[test]
    fn a_parked_recording_leaves_the_coordinator_and_returns_through_its_predecessor() {
        use crate::runtime::replacement_session as session;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let upload_channel = reims_vgpu_protocol::ChannelId::new(26);
        let producer_channel = reims_vgpu_protocol::ChannelId::new(28);
        runtime.define_channel(upload_channel).unwrap();
        runtime.define_channel(producer_channel).unwrap();
        let resources = session::stage_guest_upload_resources_for_test(&mut runtime);
        let event = reims_vgpu_protocol::ResourceId::new(41, 1);
        let Some(upload) = session::stage_guest_upload_ingress_for_test(
            &mut runtime,
            resources,
            upload_channel,
            69,
            Some(reims_vgpu_core::EventOperation {
                event,
                kind: reims_vgpu_core::EventOperationKind::Wait,
                value: 1,
            }),
        ) else {
            return;
        };
        let Some(producer) = session::stage_event_signal_ingress_for_test(
            &mut runtime,
            producer_channel,
            71,
            reims_vgpu_core::EventOperation {
                event,
                kind: reims_vgpu_core::EventOperationKind::Signal,
                value: 1,
            },
        ) else {
            return;
        };
        assert!(matches!(
            &upload,
            session::PendingReplacementIngressExec::GuestUpload(_)
        ));
        assert!(matches!(
            &producer,
            session::PendingReplacementIngressExec::Direct(_)
        ));
        let upload_transaction = upload.transaction();
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device
            .exec_recordings
            .admit(upload)
            .unwrap_or_else(|_| panic!("the upload transaction must be unique"));
        device
            .exec_recordings
            .admit(producer)
            .unwrap_or_else(|_| panic!("the producer transaction must be unique"));

        let mut settled = false;
        for _ in 0..100_000 {
            device.progress_exec_recordings();
            if device.exec_recordings.live_recordings() == 0 {
                settled = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            settled,
            "a parked owner belongs to the runtime epoch, not to the recording coordinator"
        );
        assert_eq!(
            device.guest_uploads.live_uploads(),
            0,
            "the parked upload may not reach its queue before its predecessor"
        );
        assert_eq!(device.exec_submissions.live_submissions(), 1);

        let mut accepted = false;
        for _ in 0..100_000 {
            let _ = device.progress_exec_submissions();
            if device.harvest_exec_acceptances() == 1 {
                accepted = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(accepted, "the producer must return driver acceptance");
        assert_eq!(
            device.guest_uploads.transaction_ids(),
            vec![upload_transaction],
            "predecessor acceptance must release the parked owner into its own coordinator"
        );
        assert!(device.ready_guest_uploads.is_empty());
        assert_eq!(device.exec_recordings.live_recordings(), 0);
        assert!(device.drain_failures.is_empty());
    }

    /// An indirect range reserves its own source head at the initial readback
    /// phase, so its queue-ready owner belongs to the indirect coordinator.
    /// Routing it as a direct submission, or leaving it behind in the recording
    /// coordinator, loses the continuation chain the guest asked for.
    #[test]
    fn an_initial_indirect_range_recording_routes_into_the_indirect_coordinator() {
        use crate::runtime::replacement_session as session;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(16);
        runtime.define_channel(channel).unwrap();
        let Some(pending) =
            session::stage_initial_indirect_range_ingress_for_test(&mut runtime, channel)
        else {
            return;
        };
        let transaction = pending.transaction();
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device
            .exec_recordings
            .admit(pending)
            .unwrap_or_else(|_| panic!("the recording transaction must be unique"));
        let mut handed_off = false;
        for _ in 0..100_000 {
            if device.progress_exec_recordings() == 1 {
                handed_off = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            handed_off,
            "a queue-ready initial indirect range must leave the recording coordinator"
        );
        assert_eq!(device.exec_recordings.live_recordings(), 0);
        assert_eq!(device.exec_submissions.live_submissions(), 0);
        assert_eq!(device.guest_uploads.live_uploads(), 0);
        assert_eq!(
            device
                .ready_indirect_ranges
                .iter()
                .map(|ready| ready.transaction())
                .collect::<Vec<_>>(),
            vec![transaction]
        );
    }

    /// A staged upload behind an earlier transaction on its own channel is not
    /// yet its source domain's submission head, so the queue refuses it. That
    /// refusal is an ordering fact, not a verdict: the recording must be asked
    /// again once the transaction ahead of it is accepted, exactly as a direct
    /// recording is. A guest-upload arm that refuses once and never retries
    /// loses the EXEC and every later transaction on the channel behind it.
    #[test]
    fn a_staged_upload_behind_its_channel_head_is_retried_until_the_head_is_accepted() {
        use crate::runtime::replacement_session as session;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(26);
        runtime.define_channel(channel).unwrap();
        let resources = session::stage_guest_upload_resources_for_test(&mut runtime);
        let Some(head) = session::stage_event_signal_ingress_for_test(
            &mut runtime,
            channel,
            68,
            reims_vgpu_core::EventOperation {
                event: reims_vgpu_protocol::ResourceId::new(41, 1),
                kind: reims_vgpu_core::EventOperationKind::Signal,
                value: 1,
            },
        ) else {
            return;
        };
        let Some(upload) = session::stage_guest_upload_ingress_for_test(
            &mut runtime,
            resources,
            channel,
            69,
            None,
        ) else {
            return;
        };
        assert!(matches!(
            &head,
            session::PendingReplacementIngressExec::Direct(_)
        ));
        assert!(matches!(
            &upload,
            session::PendingReplacementIngressExec::GuestUpload(_)
        ));
        let head_transaction = head.transaction();
        let upload_transaction = upload.transaction();
        assert!(
            head_transaction < upload_transaction,
            "the head must be the earlier transaction on this channel"
        );
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device
            .exec_recordings
            .admit(head)
            .unwrap_or_else(|_| panic!("the head transaction must be unique"));
        device
            .exec_recordings
            .admit(upload)
            .unwrap_or_else(|_| panic!("the upload transaction must be unique"));

        let mut routed = false;
        for _ in 0..100_000 {
            device.progress_exec_recordings();
            let _ = device.progress_exec_submissions();
            let _ = device.harvest_exec_acceptances();
            if device.guest_uploads.transaction_ids() == vec![upload_transaction] {
                routed = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            routed,
            "the staged upload must be retried until its channel head is accepted"
        );
        assert_eq!(
            device.exec_recordings.live_recordings(),
            0,
            "no refused recording may remain once it has been routed"
        );
        assert!(device.ready_guest_uploads.is_empty());
        assert!(device.drain_failures.is_empty());
    }

    #[test]
    fn deferred_exec_uses_replacement_task_memory_through_recording_dispatch() {
        use crate::runtime::host::HostMemory;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(6);
        let task = reims_vgpu_protocol::TaskId::new(7);
        runtime.define_channel(channel).unwrap();
        runtime.define_task(task, 0x1_0000, 2).unwrap();

        let mut stream = vec![0; crate::runtime::decode::stream::SEGMENT_HEADER_LEN];
        let stream_len = u32::try_from(stream.len()).unwrap();
        reims_vgpu_core::endian::st32(&mut stream, stream_len);
        stream[4] = crate::runtime::decode::stream::SEGMENT_TYPE_EVENT;
        let mut payload = vec![
            0;
            crate::runtime::decode::fifo::CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + crate::runtime::decode::fifo::CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN
                    as usize
        ];
        reims_vgpu_core::endian::st32(&mut payload, task.get());
        reims_vgpu_core::endian::st32(&mut payload[8..], 1);
        let descriptor = crate::runtime::decode::fifo::CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
        reims_vgpu_core::endian::st64(&mut payload[descriptor..], 12);
        reims_vgpu_core::endian::st64(&mut payload[descriptor + 8..], stream.len() as u64);
        let deferred = match
            crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
                &mut runtime,
                channel,
                packet(crate::model::CHILD_OP_EXEC_INDIRECT2, payload, 21),
            )
        {
            Err(crate::runtime::replacement_child_packet::ReplacementChildCpuPacketIngressError::RequiresTransport(
                crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::Exec(packet),
            )) => packet,
            _ => panic!("EXEC must defer its declared task-memory read"),
        };

        let mut host = crate::runtime::host::FakeHost::new();
        let shift = crate::model::PAGE_SHIFT_ARM64E;
        let failure = match dispatch_host_exec_packet(&mut runtime, &mut host, shift, deferred) {
            Err(failure) => failure,
            Ok(_) => panic!("unavailable task memory must retain the deferred EXEC"),
        };
        assert!(matches!(
            failure.as_ref(),
            ReplacementHostExecDispatchFailure::Load { .. }
        ));
        let directory = 2u64 << shift;
        let root = 3u64 << shift;
        let data = 9u64 << shift;
        host.map_range(directory, 1usize << shift, 0);
        host.map_range(root, 1usize << shift, 0);
        host.map_range(data, 1usize << shift, 0);
        let mut directory_bytes = [0u8; 8];
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_ROOT_PFN as usize..], 3);
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(directory, &directory_bytes).unwrap();
        host.write_gpa(root, &9u32.to_le_bytes()).unwrap();
        host.write_gpa(data + 12, &stream).unwrap();

        let mut device = ReplacementDeviceCoordinator::new(runtime, shift).unwrap();
        let _ = device.tick(&mut host);
        let wakes_before_recording = host.worker_wake_count();
        let pending =
            retry_host_exec_dispatch_failure(&mut device.runtime, &mut host, shift, *failure)
                .unwrap_or_else(|_| panic!("the retained task-memory EXEC must dispatch on retry"));
        assert!(matches!(
            &pending,
            crate::runtime::replacement_session::PendingReplacementIngressExec::Direct(_)
        ));
        let transaction = pending.transaction();
        device
            .exec_recordings
            .admit(pending)
            .unwrap_or_else(|_| panic!("the recording transaction must be unique"));
        for _ in 0..100_000 {
            if host.worker_wake_count() != wakes_before_recording {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(
            host.worker_wake_count(),
            wakes_before_recording + 1,
            "native recording completion must wake the scheduler that owns its receipt"
        );
        let mut handed_off = false;
        for _ in 0..100_000 {
            if device.progress_exec_recordings() == 1 {
                handed_off = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            handed_off,
            "the aggregate must hand the recording to its queue owner"
        );
        assert_eq!(device.exec_recordings.live_recordings(), 0);
        assert_eq!(device.exec_submissions.live_submissions(), 1);
        let mut accepted = false;
        for _ in 0..100_000 {
            for progress in device.progress_exec_submissions() {
                if progress == ReplacementExecSubmitCoordinatorProgress::Accepted {
                    accepted = true;
                    break;
                }
            }
            if accepted {
                break;
            }
            std::thread::yield_now();
        }
        assert!(accepted, "the queue owner must return driver acceptance");
        assert_eq!(device.harvest_exec_acceptances(), 1);
        assert_eq!(device.exec_submissions.live_submissions(), 0);

        for _ in 0..100_000 {
            let progress = device.poll_timelines();
            assert_eq!(progress.observation_failures, 0);
            assert_eq!(progress.semantic_failures, 0);
            if device.pending_publications() != 0 {
                break;
            }
            std::thread::yield_now();
        }
        let fact = device
            .publications
            .facts
            .pop_front()
            .expect("the accepted EXEC must retire to one semantic fact");
        assert_eq!(fact.transaction, transaction);
        assert_eq!(
            fact.completion_stamp,
            Some(reims_vgpu_core::CompletionStamp::new(channel.get(), 21))
        );
        assert_eq!(device.retired_batches.len(), 1);
        assert_eq!(device.harvest_retired_batches(), 1);
        for _ in 0..100_000 {
            device.progress_recording_cleanup();
            if device.pending_recording_cleanups.is_empty() {
                break;
            }
            std::thread::yield_now();
        }
        assert!(device.recording_cleanup_dispatch_failures.is_empty());
        assert!(device.recording_cleanup_completion_failures.is_empty());
    }

    #[test]
    fn deferred_cursor_glyph_reads_once_before_cpu_admission() {
        use crate::runtime::host::HostMemory;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        let task = reims_vgpu_protocol::TaskId::new(9);
        runtime.define_channel(channel).unwrap();
        runtime.define_task(task, 0x20_000, 2).unwrap();
        let mut payload = vec![0; crate::model::CURSOR_GLYPH_PAYLOAD_LEN];
        reims_vgpu_core::endian::st32(&mut payload, 3);
        reims_vgpu_core::endian::st32(&mut payload[0x04..], task.get());
        reims_vgpu_core::endian::st64(&mut payload[0x08..], 12);
        reims_vgpu_core::endian::st64(&mut payload[0x10..], 16);
        reims_vgpu_core::endian::st64(&mut payload[0x18..], 8);
        reims_vgpu_core::endian::st16(&mut payload[0x20..], 2);
        reims_vgpu_core::endian::st16(&mut payload[0x22..], 2);
        reims_vgpu_core::endian::st16(&mut payload[0x24..], 1);
        let deferred = match
            crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
                &mut runtime,
                channel,
                packet(crate::model::CHILD_OP_CURSOR_GLYPH, payload, 22),
            )
        {
            Err(crate::runtime::replacement_child_packet::ReplacementChildCpuPacketIngressError::RequiresTransport(
                crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::CursorGlyph(packet),
            )) => packet,
            _ => panic!("cursor glyph must defer its declared task-memory read"),
        };

        let mut host = crate::runtime::host::FakeHost::new();
        let shift = crate::model::PAGE_SHIFT_ARM64E;
        let directory = 2u64 << shift;
        let root = 3u64 << shift;
        let data = 9u64 << shift;
        host.map_range(directory, 1usize << shift, 0);
        host.map_range(root, 1usize << shift, 0);
        host.map_range(data, 1usize << shift, 0);
        let mut directory_bytes = [0; 8];
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_ROOT_PFN as usize..], 3);
        reims_vgpu_core::endian::st32(&mut directory_bytes[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(directory, &directory_bytes).unwrap();
        host.write_gpa(root, &9u32.to_le_bytes()).unwrap();
        host.write_gpa(
            data + 12,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();

        let admitted =
            load_and_admit_host_cursor_glyph(&mut runtime, &host, shift, deferred).unwrap();
        let applied = apply_ready_cpu_packet(&mut runtime, shift, u32::MAX, admitted).unwrap();
        assert!(matches!(
            applied,
            ReplacementAppliedCpuPacket::Control(ReplacementAppliedControl {
                effect: ReplacementControlEffect::CursorGlyphPublished(glyph),
                ..
            }) if glyph.pixels.as_ref()
                == [0x0403_0201, 0x0807_0605, 0x0c0b_0a09, 0x100f_0e0d]
        ));
    }

    #[test]
    fn synchronize_pre_admission_refusal_returns_the_exact_deferred_packet() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(12);
        let task = reims_vgpu_protocol::TaskId::new(91);
        runtime.define_channel(channel).unwrap();
        let mut payload = [0; 12];
        reims_vgpu_core::endian::st32(&mut payload, task.get());
        reims_vgpu_core::endian::st32(&mut payload[4..], 1);
        reims_vgpu_core::endian::st32(&mut payload[8..], 5);
        let deferred = match
            crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
                &mut runtime,
                channel,
                packet(crate::model::CHILD_OP_SYNCHRONIZE_RESOURCES, payload, 27),
            )
        {
            Err(crate::runtime::replacement_child_packet::ReplacementChildCpuPacketIngressError::RequiresTransport(
                crate::runtime::replacement_child_packet::ReplacementChildPacketTransport::Synchronize(packet),
            )) => packet,
            _ => panic!("nonempty synchronize must defer its execution transaction"),
        };
        let result =
            crate::runtime::replacement_child_packet::dispatch_deferred_replacement_synchronize(
                &mut runtime,
                deferred,
            );
        match result {
            Err(failure) => match *failure {
                crate::runtime::replacement_child_packet::ReplacementDeferredSynchronizeDispatchFailure::PreAdmission {
                reason: crate::runtime::replacement_child_packet::ReplacementSynchronizePreAdmissionError::Resolution(
                    crate::runtime::replacement_session::ReplacementSynchronizeResolutionError::UnknownTask(found),
                ),
                deferred,
            } => {
                assert_eq!(found, task);
                assert_eq!(deferred.envelope.channel, channel);
                assert_eq!(deferred.envelope.completion_stamp.value, 27);
                assert_eq!(deferred.command.objects[0].get(), 5);
            }
                _ => panic!("pre-admission refusal must retain the deferred synchronize packet"),
            },
            Ok(_) => panic!("an unknown task cannot dispatch synchronize work"),
        }
    }

    #[test]
    fn root_ring_head_advances_after_transaction_admission_not_before() {
        use crate::runtime::host::HostMemory;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let shift = crate::model::PAGE_SHIFT_X86;
        let mut transport =
            crate::runtime::replacement_transport::ReplacementTransportOwner::new(shift).unwrap();
        let pfn = 29u32;
        transport.registers_mut().gfx.control_fifo = 1;
        transport.registers_mut().gfx.fifo_base_page = pfn;
        transport.registers_mut().gfx.fifo_start = 0x100;
        transport.registers_mut().gfx.fifo_length = 0x200;
        transport.registers_mut().gfx.fifo_written = crate::model::PACKET_HEADER_LEN + 4;
        let mut bytes = vec![0; crate::model::PACKET_HEADER_LEN as usize + 4];
        reims_vgpu_core::endian::st16(&mut bytes, crate::model::ROOT_OP_DEFINE_FIFO);
        reims_vgpu_core::endian::st32(
            &mut bytes[crate::model::PACKET_TOTAL_SIZE..],
            crate::model::PACKET_HEADER_LEN + 4,
        );
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_COMPLETION_STAMP..], 31);
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_HEADER_LEN as usize..], 4);
        let mut host = crate::runtime::host::FakeHost::new();
        let page_size = 1u64 << shift;
        host.map_range(u64::from(pfn) * page_size, page_size as usize, 0);
        host.write_gpa(u64::from(pfn) * page_size + 0x100, &bytes)
            .unwrap();

        let lease = transport.read_root_packet(&host).unwrap().unwrap();
        assert_eq!(
            transport
                .registers()
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
        let admitted = admit_root_packet_lease(&mut runtime, &mut transport, lease).unwrap();
        assert_eq!(
            transport
                .registers()
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            crate::model::PACKET_HEADER_LEN + 4
        );
        let applied = apply_ready_cpu_packet(&mut runtime, shift, u32::MAX, admitted).unwrap();
        assert!(matches!(
            applied,
            ReplacementAppliedCpuPacket::Control(ReplacementAppliedControl {
                effect: ReplacementControlEffect::FifoDefined(channel),
                ..
            }) if channel == reims_vgpu_protocol::ChannelId::new(4)
        ));
    }

    #[test]
    fn device_coordinator_drains_root_cpu_work_into_its_only_publication_sink() {
        use crate::runtime::host::HostMemory;

        let Some(runtime) = runtime() else {
            return;
        };
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        let pfn = 31u32;
        let page_size = 1u64 << crate::model::PAGE_SHIFT_X86;
        device.transport.registers_mut().gfx.control_fifo = 1;
        device.transport.registers_mut().gfx.fifo_base_page = pfn;
        device.transport.registers_mut().gfx.fifo_start = 0x100;
        device.transport.registers_mut().gfx.fifo_length = 0x200;
        device.transport.registers_mut().gfx.fifo_written = crate::model::PACKET_HEADER_LEN + 4;
        let mut bytes = vec![0; crate::model::PACKET_HEADER_LEN as usize + 4];
        reims_vgpu_core::endian::st16(&mut bytes, crate::model::ROOT_OP_DEFINE_FIFO);
        reims_vgpu_core::endian::st32(
            &mut bytes[crate::model::PACKET_TOTAL_SIZE..],
            crate::model::PACKET_HEADER_LEN + 4,
        );
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_COMPLETION_STAMP..], 33);
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_HEADER_LEN as usize..], 4);
        let mut host = crate::runtime::host::FakeHost::new();
        let base = u64::from(pfn) * page_size;
        host.map_range(base, page_size as usize, 0);
        host.write_gpa(base + 0x100, &bytes).unwrap();
        device.transport.request_root();

        let tick = device.tick(&mut host);
        assert_eq!(
            tick.drained,
            ReplacementDeviceDrainProgress {
                root_packets: 1,
                child_packets: 0,
                mapper_entries: 0,
                failures: 0,
            }
        );
        assert_eq!(
            tick.cpu_completed, 0,
            "drain completed the ready CPU packet inline"
        );
        assert_eq!(tick.publications, 1);
        assert_eq!(device.pending_publications(), 0);
        assert_eq!(device.owned_phase_count(), 0);
        assert_eq!(host.get_u32(base), 33);
        assert_eq!(device.live_transactions(), 0);
    }

    #[test]
    fn device_tick_retains_one_failure_for_one_blocked_publication() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device
            .publications
            .enqueue([reims_vgpu_core::PublishedFact {
                transaction: reims_vgpu_protocol::TransactionId::new(91),
                position: reims_vgpu_core::PublicationPosition {
                    domain: reims_vgpu_protocol::PublicationDomainId::new(1),
                    sequence: reims_vgpu_protocol::PublicationSequence::new(1),
                },
                completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(0, 7)),
                semantic: (),
            }]);
        let mut host = crate::runtime::host::FakeHost::new();

        assert_eq!(device.tick(&mut host).publications, 0);
        assert!(matches!(
            device.publication_failure,
            Some(ReplacementPublishedFactHostError::StampPageUnavailable)
        ));
        let first_owned = device.owned_phase_count();
        assert_eq!(device.pending_publications(), 1);

        assert_eq!(device.tick(&mut host).publications, 0);
        assert_eq!(device.pending_publications(), 1);
        assert_eq!(device.owned_phase_count(), first_owned);
    }

    #[test]
    fn platform_reset_cancels_aggregate_ownership_without_replacing_or_rewinding_vulkan() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        let epoch = device.runtime.session().vulkan().id();
        let first_point = device
            .runtime
            .execution_mut()
            .native_mut()
            .prepare_present(
                reims_vgpu_protocol::TransactionId::new(1),
                reims_vgpu_protocol::QueueOwnerId::new(1),
            )
            .unwrap()
            .point();
        device
            .publications
            .enqueue([reims_vgpu_core::PublishedFact {
                transaction: reims_vgpu_protocol::TransactionId::new(2),
                position: reims_vgpu_core::PublicationPosition {
                    domain: reims_vgpu_protocol::PublicationDomainId::new(1),
                    sequence: reims_vgpu_protocol::PublicationSequence::new(1),
                },
                completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(0, 7)),
                semantic: (),
            }]);
        device.transport.registers_mut().gfx.control_fifo = 1;
        device.transport.registers_mut().iosfc.ring_base = 0x4000;
        device.transport.request_root();
        device.product_presented = true;
        let mut host = crate::runtime::host::FakeHost::new();
        assert_eq!(device.drain(&mut host).failures, 1);
        assert_eq!(device.pending_publications(), 1);
        assert_ne!(device.owned_phase_count(), 0);

        let effect = device
            .platform_reset(reims_vgpu_protocol::SessionGenerationId::new(2))
            .unwrap();

        assert_eq!(
            effect.runtime.execution_generation,
            SessionGenerationId::new(1)
        );
        assert_eq!(effect.abandonment, Default::default());
        assert_eq!(device.runtime.session().vulkan().id(), epoch);
        assert_eq!(
            device.vulkan_state(),
            reims_vgpu_core::VulkanDeviceEpochState::Active
        );
        assert_eq!(device.pending_publications(), 0);
        assert_eq!(device.owned_phase_count(), 0);
        assert_eq!(device.live_transactions(), 0);
        assert_eq!(device.transport_registers().gfx.control_fifo, 0);
        assert_eq!(device.transport_registers().iosfc.ring_base, 0);
        assert!(!device.product_presented());

        let second_point = device
            .runtime
            .execution_mut()
            .native_mut()
            .prepare_present(
                reims_vgpu_protocol::TransactionId::new(3),
                reims_vgpu_protocol::QueueOwnerId::new(1),
            )
            .unwrap()
            .point();
        assert_eq!(second_point.epoch, first_point.epoch);
        assert_eq!(second_point.queue, first_point.queue);
        assert!(second_point.value > first_point.value);
    }

    #[test]
    fn device_loss_joins_workers_and_abandons_all_owners_without_publication() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        let _admitted = device
            .runtime
            .admit_control(
                reims_vgpu_protocol::ChannelId::new(0),
                Box::default(),
                Some(reims_vgpu_core::CompletionStamp::new(0, 9)),
                crate::runtime::replacement_session::ReplacementControlCommand::ContractNoOp(
                    crate::model::CHILD_OP_NOP,
                ),
            )
            .unwrap();
        device
            .publications
            .enqueue([reims_vgpu_core::PublishedFact {
                transaction: reims_vgpu_protocol::TransactionId::new(2),
                position: reims_vgpu_core::PublicationPosition {
                    domain: reims_vgpu_protocol::PublicationDomainId::new(1),
                    sequence: reims_vgpu_protocol::PublicationSequence::new(1),
                },
                completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(0, 10)),
                semantic: (),
            }]);
        device.transport.registers_mut().gfx.control_fifo = 1;
        assert_eq!(device.live_transactions(), 1);
        assert_eq!(device.pending_publications(), 1);
        assert!(device.runtime.session().begin_device_loss());

        assert!(device.terminalize_device_loss());

        let effect = device.device_loss_effect().unwrap();
        assert_eq!(effect.abandonment.semantic_transactions, 1);
        assert_eq!(
            device.vulkan_state(),
            reims_vgpu_core::VulkanDeviceEpochState::Losing
        );
        assert_eq!(device.live_transactions(), 0);
        assert_eq!(device.pending_publications(), 0);
        assert_eq!(device.owned_phase_count(), 0);
        assert_eq!(device.transport_registers().gfx.control_fifo, 0);
        assert_eq!(device.runtime.session().pipeline_worker_census().workers, 0);
        assert_eq!(
            device
                .runtime
                .session()
                .vulkan()
                .queues()
                .recording_workers()
                .worker_count(),
            0
        );
        assert_eq!(
            device
                .runtime
                .session()
                .vulkan()
                .queues()
                .lane(reims_vgpu_protocol::QueueOwnerId::new(1))
                .unwrap()
                .submit
                .wait_idle(),
            Err(reims_vgpu_vulkan::replacement_queue::ReplacementQueueError::OwnerStopped)
        );
        let mut host = crate::runtime::host::FakeHost::new();
        assert_eq!(
            device.tick(&mut host),
            ReplacementDeviceTickProgress::default()
        );
        assert_eq!(host.action_count(HostActionKind::IrqGfxPulse), 0);
        assert!(matches!(
            device.platform_reset(SessionGenerationId::new(2)),
            Err(ReplacementDeviceResetError::DeviceLost)
        ));
    }

    #[test]
    fn device_tick_retains_one_root_read_reason_while_the_ring_is_unavailable() {
        let Some(runtime) = runtime() else {
            return;
        };
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        device.transport.request_root();

        assert_eq!(device.tick(&mut host).drained.failures, 1);
        assert!(matches!(
            device.root_read_failure,
            Some(crate::runtime::replacement_transport::ReplacementRootPacketReadError::RingUnavailable)
        ));
        let first_owned = device.owned_phase_count();
        assert_eq!(first_owned, 1);

        assert_eq!(device.tick(&mut host).drained.failures, 1);
        assert_eq!(device.owned_phase_count(), first_owned);
        assert!(device.drain_failures.is_empty());
    }

    #[test]
    fn replacement_read_diagnostics_distinguish_refusal_from_partial_publication() {
        assert!(replacement_root_read_diagnostic(
            crate::runtime::replacement_transport::ReplacementRootPacketReadError::Packet(
                crate::runtime::fifo_packet::PacketError::Incomplete,
            ),
        )
        .is_none());
        let root = replacement_root_read_diagnostic(
            crate::runtime::replacement_transport::ReplacementRootPacketReadError::InvalidPublishedPointers {
                head: 8,
                tail: 7,
                capacity: 64,
            },
        )
        .unwrap();
        assert_eq!(
            crate::observe::Emit::decline("replacement_device_root_read", &root).render(),
            "replacement_device_root_read reason=replacement_root_ring_pointers_invalid head=8 tail=7 capacity=64"
        );

        let channel = reims_vgpu_protocol::ChannelId::new(4);
        let child = replacement_child_read_diagnostic(
            channel,
            crate::runtime::replacement_transport::ReplacementChildPacketReadError::Memory {
                phase: crate::runtime::replacement_transport::ReplacementChildPacketReadPhase::PageList {
                    entry: 3,
                },
                reason: MemError::Unmapped,
            },
        )
        .unwrap();
        assert_eq!(
            crate::observe::Emit::decline("replacement_device_child_read", &child).render(),
            "replacement_device_child_read reason=replacement_child_ring_memory_unavailable channel=4 phase=page_list entry=3 memory=Unmapped"
        );
    }

    #[cfg(feature = "host-window")]
    #[test]
    fn unattached_window_lifecycle_is_owned_by_the_replacement_epoch() {
        let Some(runtime) = runtime() else {
            return;
        };
        let device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();

        // Resize before attach is expected control flow while a platform is
        // still creating its native window. Detach is idempotent so teardown
        // can always order the Vulkan surface before the window lifetime.
        device.resize_window(1280, 720);
        // SAFETY: no handles were attached, so there is no native lifetime to
        // uphold and detach performs no Vulkan destruction.
        assert!(unsafe { device.detach_window() }.is_ok());
    }

    #[test]
    fn child_ring_head_advances_only_for_an_admitted_packet() {
        use crate::runtime::host::HostMemory;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let shift = crate::model::PAGE_SHIFT_X86;
        let page_size = 1u64 << shift;
        let mut transport =
            crate::runtime::replacement_transport::ReplacementTransportOwner::new(shift).unwrap();
        transport.registers_mut().gfx.root_page = 2;
        let registers_gpa =
            2 * page_size + crate::model::child_reg_block_offset(channel.get()).unwrap();
        let list_gpa = 3 * page_size;
        let ring_gpa = 5 * page_size;
        let mut host = crate::runtime::host::FakeHost::new();
        host.map_range(2 * page_size, page_size as usize, 0);
        host.map_range(list_gpa, page_size as usize, 0);
        host.map_range(ring_gpa, page_size as usize, 0);
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_TAIL,
            &crate::model::PACKET_HEADER_LEN.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_STAMP_INDEX,
            &channel.get().to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_BASE_PFN,
            &3u32.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(list_gpa, &5u32.to_le_bytes()).unwrap();
        let mut bytes = vec![0; crate::model::PACKET_HEADER_LEN as usize];
        reims_vgpu_core::endian::st16(&mut bytes, crate::model::CHILD_OP_NOP);
        reims_vgpu_core::endian::st32(
            &mut bytes[crate::model::PACKET_TOTAL_SIZE..],
            crate::model::PACKET_HEADER_LEN,
        );
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_COMPLETION_STAMP..], 43);
        host.write_gpa(ring_gpa, &bytes).unwrap();

        let lease = transport
            .read_child_packet(&host, channel)
            .unwrap()
            .unwrap();
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            0
        );
        let ingress = admit_child_packet_lease(&mut runtime, &mut transport, &mut host, lease);
        assert!(matches!(
            ingress,
            Ok(ReplacementChildPacketLeaseIngress::Admitted(
                crate::runtime::replacement_child_packet::ReplacementAdmittedChildCpuPacket::Control(_)
            ))
        ));
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            crate::model::PACKET_HEADER_LEN
        );
    }

    #[test]
    fn deferred_child_failure_keeps_its_ring_head_and_lease() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let task = reims_vgpu_protocol::TaskId::new(91);
        let mut payload = [0; 12];
        reims_vgpu_core::endian::st32(&mut payload, task.get());
        reims_vgpu_core::endian::st32(&mut payload[4..], 1);
        reims_vgpu_core::endian::st32(&mut payload[8..], 5);
        let (mut transport, mut host, lease, registers_gpa) = child_lease(
            channel,
            crate::model::CHILD_OP_SYNCHRONIZE_RESOURCES,
            &payload,
            47,
        );
        let ingress = admit_child_packet_lease(&mut runtime, &mut transport, &mut host, lease);
        let ReplacementChildPacketLeaseIngress::Deferred {
            transport: deferred,
            lease,
        } = ingress.unwrap_or_else(|_| panic!("synchronize must defer before EXEC admission"))
        else {
            panic!("synchronize must retain its ring lease through deferred dispatch")
        };
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            0
        );
        let failure = match dispatch_deferred_child_packet(
            &mut runtime,
            &mut transport,
            &mut host,
            deferred,
            lease,
        ) {
            Err(failure) => {
                match failure.as_ref() {
                    ReplacementDeferredChildDispatchFailure::Synchronize { failure, lease } => {
                        assert_eq!(lease.channel, channel);
                        assert!(matches!(
                            failure.as_ref(),
                            crate::runtime::replacement_child_packet::ReplacementDeferredSynchronizeDispatchFailure::PreAdmission {
                                reason: crate::runtime::replacement_child_packet::ReplacementSynchronizePreAdmissionError::Resolution(
                                    crate::runtime::replacement_session::ReplacementSynchronizeResolutionError::UnknownTask(found),
                                ),
                                ..
                            } if *found == task
                        ));
                    }
                    _ => {
                        panic!("synchronize refusal must retain its deferred owner and ring lease")
                    }
                }
                failure
            }
            Ok(_) => panic!("an unknown task cannot dispatch synchronize work"),
        };
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            0
        );
        // The task arriving is what the retry was waiting for. The object it
        // names is still absent, and that is not a second wait: a synchronize
        // is a teardown statement about writes this device might still owe,
        // and it owes none against a resource it does not have. So the packet
        // dispatches, consumes its ring lease, and advances the head — rather
        // than holding the channel for a resource that is going away.
        runtime.define_task(task, 0x1_0000, 2).unwrap();
        retry_deferred_child_dispatch_failure(&mut runtime, &mut transport, &mut host, *failure)
            .unwrap_or_else(|_| {
                panic!("a synchronize naming no resource this device holds is already satisfied")
            });
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            crate::model::PACKET_HEADER_LEN + payload.len() as u32
        );
    }

    #[test]
    fn device_tick_keeps_one_owner_for_one_blocked_deferred_child() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let task = reims_vgpu_protocol::TaskId::new(91);
        let mut payload = [0; 12];
        reims_vgpu_core::endian::st32(&mut payload, task.get());
        reims_vgpu_core::endian::st32(&mut payload[4..], 1);
        reims_vgpu_core::endian::st32(&mut payload[8..], 5);
        let (mut transport, mut host, lease, registers_gpa) = child_lease(
            channel,
            crate::model::CHILD_OP_SYNCHRONIZE_RESOURCES,
            &payload,
            47,
        );
        let ReplacementChildPacketLeaseIngress::Deferred {
            transport: deferred,
            lease,
        } = admit_child_packet_lease(&mut runtime, &mut transport, &mut host, lease)
            .unwrap_or_else(|_| panic!("synchronize must retain its transport owner"))
        else {
            panic!("synchronize must defer")
        };
        let failure = dispatch_deferred_child_packet(
            &mut runtime,
            &mut transport,
            &mut host,
            deferred,
            lease,
        )
        .err()
        .expect("the unknown task must retain the exact deferred owner");
        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device.transport = transport;
        device
            .blocked_drains
            .push_back(Box::new(ReplacementBlockedDrain::DeferredChild(failure)));

        let first_owned = device.owned_phase_count();
        assert_eq!(first_owned, 1);
        assert_eq!(device.tick(&mut host).drained.failures, 1);
        assert_eq!(device.owned_phase_count(), first_owned);
        assert!(device.drain_failures.is_empty());
        assert_eq!(device.tick(&mut host).drained.failures, 1);
        assert_eq!(device.owned_phase_count(), first_owned);
        assert!(device.drain_failures.is_empty());
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            0
        );
    }

    #[test]
    fn blocked_child_does_not_stop_an_independent_child_fifo() {
        use crate::runtime::host::HostMemory;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let blocked_channel = reims_vgpu_protocol::ChannelId::new(4);
        let producer_channel = reims_vgpu_protocol::ChannelId::new(5);
        runtime.define_channel(blocked_channel).unwrap();
        runtime.define_channel(producer_channel).unwrap();
        let task = reims_vgpu_protocol::TaskId::new(91);
        let mut payload = [0; 12];
        reims_vgpu_core::endian::st32(&mut payload, task.get());
        reims_vgpu_core::endian::st32(&mut payload[4..], 1);
        reims_vgpu_core::endian::st32(&mut payload[8..], 5);
        let (mut transport, mut host, lease, blocked_registers) = child_lease(
            blocked_channel,
            crate::model::CHILD_OP_SYNCHRONIZE_RESOURCES,
            &payload,
            47,
        );
        let ReplacementChildPacketLeaseIngress::Deferred {
            transport: deferred,
            lease,
        } = admit_child_packet_lease(&mut runtime, &mut transport, &mut host, lease).unwrap()
        else {
            panic!("synchronize must retain its transport owner")
        };
        let failure = dispatch_deferred_child_packet(
            &mut runtime,
            &mut transport,
            &mut host,
            deferred,
            lease,
        )
        .err()
        .expect("the unresolved synchronize must remain blocked");

        let page_size = 1u64 << crate::model::PAGE_SHIFT_X86;
        let producer_registers =
            2 * page_size + crate::model::child_reg_block_offset(producer_channel.get()).unwrap();
        let producer_list = 7 * page_size;
        let producer_ring = 9 * page_size;
        host.map_range(0, page_size as usize, 0);
        host.map_range(producer_list, page_size as usize, 0);
        host.map_range(producer_ring, page_size as usize, 0);
        host.write_gpa(
            producer_registers + crate::model::CHILD_REG_TAIL,
            &crate::model::PACKET_HEADER_LEN.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            producer_registers + crate::model::CHILD_REG_STAMP_INDEX,
            &producer_channel.get().to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            producer_registers + crate::model::CHILD_REG_BASE_PFN,
            &7u32.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(producer_list, &9u32.to_le_bytes()).unwrap();
        let mut producer = vec![0; crate::model::PACKET_HEADER_LEN as usize];
        reims_vgpu_core::endian::st16(&mut producer, crate::model::CHILD_OP_NOP);
        reims_vgpu_core::endian::st32(
            &mut producer[crate::model::PACKET_TOTAL_SIZE..],
            crate::model::PACKET_HEADER_LEN,
        );
        reims_vgpu_core::endian::st32(&mut producer[crate::model::PACKET_COMPLETION_STAMP..], 48);
        host.write_gpa(producer_ring, &producer).unwrap();

        let mut device =
            ReplacementDeviceCoordinator::new(runtime, crate::model::PAGE_SHIFT_X86).unwrap();
        device.transport = transport;
        device
            .blocked_drains
            .push_back(Box::new(ReplacementBlockedDrain::DeferredChild(failure)));
        assert!(device.transport.request_child(producer_channel));

        let progress = device.tick(&mut host).drained;
        assert_eq!(progress.child_packets, 1);
        assert_eq!(progress.failures, 1);
        assert_eq!(device.blocked_drains.len(), 1);
        assert_eq!(
            host.get_u32(blocked_registers + crate::model::CHILD_REG_HEAD),
            0
        );
        assert_eq!(
            host.get_u32(producer_registers + crate::model::CHILD_REG_HEAD),
            crate::model::PACKET_HEADER_LEN
        );
    }

    #[test]
    fn cpu_coordinator_releases_fifo_publication_after_out_of_order_completion() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let first = crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
            &mut runtime,
            channel,
            packet(crate::model::CHILD_OP_NOP, [], 1),
        )
        .unwrap();
        let second = crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
            &mut runtime,
            channel,
            packet(crate::model::CHILD_OP_NOP, [], 2),
        )
        .unwrap();
        let first = ReplacementAdmittedCpuPacket::try_from(first).unwrap();
        let second = ReplacementAdmittedCpuPacket::try_from(second).unwrap();
        let first_id = first.transaction();
        let second_id = second.transaction();
        let mut coordinator = ReplacementCpuCoordinator::default();
        coordinator.admit(first, ()).unwrap();
        coordinator.admit(second, ()).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();

        assert_eq!(
            coordinator.progress(
                &mut runtime,
                &mut host,
                crate::model::PAGE_SHIFT_X86,
                u32::MAX,
                second_id,
            ),
            Some(ReplacementCpuProgress::Published { facts: 0 })
        );
        assert_eq!(coordinator.pending_publications(), 0);
        assert_eq!(
            coordinator.progress(
                &mut runtime,
                &mut host,
                crate::model::PAGE_SHIFT_X86,
                u32::MAX,
                first_id,
            ),
            Some(ReplacementCpuProgress::Published { facts: 2 })
        );
        assert_eq!(coordinator.live_packets(), 0);
        assert_eq!(
            coordinator
                .take_published()
                .unwrap()
                .completion_stamp
                .unwrap()
                .value,
            1
        );
        assert_eq!(
            coordinator
                .take_published()
                .unwrap()
                .completion_stamp
                .unwrap()
                .value,
            2
        );
        assert!(coordinator.take_published().is_none());
    }

    /// The record is dropped and the packet still lands.
    ///
    /// The concern is unchanged: a record naming a slot nothing ever populated
    /// must not be *deferred*, or it would overwrite whatever a later resource
    /// brings into that slot. Dropping the record meets that and refusing the
    /// packet also met it -- but refusing threw away the packet's completion
    /// stamp and, in an invalidation naming several slots, every other record
    /// beside the unknown one. This is the same shape
    /// `a_delete_of_an_object_this_device_never_had_does_not_hold_the_channel`
    /// already required of the delete route.
    #[test]
    fn an_invalidation_of_a_slot_this_device_never_filled_lands_without_deferring() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        runtime
            .define_task(reims_vgpu_protocol::TaskId::new(0), 0x1_0000, 2)
            .unwrap();
        // One record naming an object-table slot nothing ever populated. There
        // is no validity state to move, and deferring it would let it overwrite
        // whatever a later resource brings into that slot.
        let mut invalidate =
            vec![
                0;
                crate::runtime::decode::fifo::CHILD_RESOURCE_LIST_HEADER_LEN as usize
                    + crate::runtime::decode::fifo::CHILD_INVALIDATE_RECORD_LEN as usize
            ];
        reims_vgpu_core::endian::st32(
            &mut invalidate[crate::runtime::decode::fifo::CHILD_RESOURCE_LIST_COUNT as usize..],
            1,
        );
        reims_vgpu_core::endian::st32(
            &mut invalidate
                [crate::runtime::decode::fifo::CHILD_RESOURCE_LIST_HEADER_LEN as usize..],
            10,
        );
        let admitted =
            crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
                &mut runtime,
                channel,
                packet(crate::model::CHILD_OP_INVALIDATE_RESOURCES, invalidate, 1),
            )
            .unwrap();
        let admitted = ReplacementAdmittedCpuPacket::try_from(admitted).unwrap();
        let admitted_id = admitted.transaction();
        let mut coordinator = ReplacementCpuCoordinator::default();
        coordinator.admit(admitted, ()).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        assert_eq!(
            coordinator.progress(
                &mut runtime,
                &mut host,
                crate::model::PAGE_SHIFT_X86,
                u32::MAX,
                admitted_id,
            ),
            Some(ReplacementCpuProgress::Published { facts: 1 })
        );
        // The stamp is consumed rather than thrown away with the packet.
        assert_eq!(
            coordinator
                .take_published()
                .unwrap()
                .completion_stamp
                .unwrap()
                .value,
            1
        );
        assert_eq!(coordinator.live_packets(), 0);
    }

    #[test]
    fn a_delete_of_an_object_this_device_never_had_does_not_hold_the_channel() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        // A delete naming a sampler reference nothing ever created. No later
        // packet can make it succeed, so retaining it would hold the channel's
        // completion stamps for the life of the device.
        let mut delete = vec![0; 4 + reims_vgpu_wire::ops::destroy::DELETE_TOTAL_LEN as usize];
        reims_vgpu_core::endian::st32(&mut delete[0..], 7);
        reims_vgpu_core::endian::st32(
            &mut delete[4..],
            reims_vgpu_wire::ops::destroy::OPCODE_DELETE_SAMPLER_STATE,
        );
        reims_vgpu_core::endian::st32(
            &mut delete[8..],
            reims_vgpu_wire::ops::destroy::DELETE_TOTAL_LEN,
        );
        reims_vgpu_core::endian::st32(&mut delete[12..], 3);
        let refused = crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
            &mut runtime,
            channel,
            packet(crate::model::CHILD_OP_DELETE_OBJECT, delete, 1),
        )
        .unwrap();
        let behind = crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
            &mut runtime,
            channel,
            packet(crate::model::CHILD_OP_NOP, [], 2),
        )
        .unwrap();
        let refused = ReplacementAdmittedCpuPacket::try_from(refused).unwrap();
        let behind = ReplacementAdmittedCpuPacket::try_from(behind).unwrap();
        let refused_id = refused.transaction();
        let behind_id = behind.transaction();
        let mut coordinator = ReplacementCpuCoordinator::default();
        coordinator.admit(refused, ()).unwrap();
        coordinator.admit(behind, ()).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();

        assert_eq!(
            coordinator.progress(
                &mut runtime,
                &mut host,
                crate::model::PAGE_SHIFT_X86,
                u32::MAX,
                refused_id,
            ),
            Some(ReplacementCpuProgress::Published { facts: 1 })
        );
        assert_eq!(
            coordinator.progress(
                &mut runtime,
                &mut host,
                crate::model::PAGE_SHIFT_X86,
                u32::MAX,
                behind_id,
            ),
            Some(ReplacementCpuProgress::Published { facts: 1 })
        );
        assert_eq!(coordinator.live_packets(), 0);
        // Both stamps post, in order: the guest is told the refused command is
        // finished and never waits on the one behind it.
        let stamps = std::iter::from_fn(|| coordinator.take_published())
            .map(|fact| fact.completion_stamp.unwrap().value)
            .collect::<Vec<_>>();
        assert_eq!(stamps, vec![1, 2]);
    }

    #[test]
    fn cpu_coordinator_retries_host_publication_without_repeating_completion() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        runtime.define_channel(channel).unwrap();
        let admitted =
            crate::runtime::replacement_child_packet::admit_replacement_child_cpu_packet(
                &mut runtime,
                channel,
                packet(crate::model::CHILD_OP_NOP, [], 9),
            )
            .unwrap();
        let admitted = ReplacementAdmittedCpuPacket::try_from(admitted).unwrap();
        let transaction = admitted.transaction();
        let mut coordinator = ReplacementCpuCoordinator::default();
        coordinator.admit(admitted, ()).unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        assert_eq!(
            coordinator.progress(
                &mut runtime,
                &mut host,
                crate::model::PAGE_SHIFT_X86,
                u32::MAX,
                transaction,
            ),
            Some(ReplacementCpuProgress::Published { facts: 1 })
        );
        let mut transport = crate::runtime::replacement_transport::ReplacementTransportOwner::new(
            crate::model::PAGE_SHIFT_X86,
        )
        .unwrap();
        assert!(matches!(
            coordinator.publish_next(&mut host, &transport),
            Err(ReplacementPublishedFactHostError::StampPageUnavailable)
        ));
        assert_eq!(coordinator.pending_publications(), 1);

        transport.registers_mut().gfx.fifo_base_page = 23;
        let base = 23u64 << crate::model::PAGE_SHIFT_X86;
        host.map_range(base, 1usize << crate::model::PAGE_SHIFT_X86, 0);
        let published = coordinator
            .publish_next(&mut host, &transport)
            .unwrap()
            .unwrap();
        assert_eq!(host.get_u32(base + u64::from(channel.get()) * 4), 9);
        assert_eq!(coordinator.pending_publications(), 0);
        let retired = retire_host_published_fact(&mut runtime, published).unwrap();
        assert_eq!(retired.transaction, transaction);
    }

    #[test]
    fn unified_publication_sink_retries_the_same_front_fact_before_the_next() {
        let mut publications = ReplacementPublicationCoordinator::default();
        let position = |sequence| reims_vgpu_core::PublicationPosition {
            domain: reims_vgpu_protocol::PublicationDomainId::new(1),
            sequence: reims_vgpu_protocol::PublicationSequence::new(sequence),
        };
        publications.enqueue([
            reims_vgpu_core::PublishedFact {
                transaction: reims_vgpu_protocol::TransactionId::new(1),
                position: position(1),
                completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(0, 11)),
                semantic: 1u32,
            },
            reims_vgpu_core::PublishedFact {
                transaction: reims_vgpu_protocol::TransactionId::new(2),
                position: position(2),
                completion_stamp: Some(reims_vgpu_core::CompletionStamp::new(1, 22)),
                semantic: 2u32,
            },
        ]);
        let mut transport = crate::runtime::replacement_transport::ReplacementTransportOwner::new(
            crate::model::PAGE_SHIFT_X86,
        )
        .unwrap();
        let mut host = crate::runtime::host::FakeHost::new();
        assert!(matches!(
            publications.publish_next(&mut host, &transport),
            Err(ReplacementPublishedFactHostError::StampPageUnavailable)
        ));
        assert_eq!(publications.pending(), 2);

        transport.registers_mut().gfx.fifo_base_page = 17;
        let base = 17u64 << crate::model::PAGE_SHIFT_X86;
        host.map_range(base, 1usize << crate::model::PAGE_SHIFT_X86, 0);
        let first = publications
            .publish_next(&mut host, &transport)
            .unwrap()
            .unwrap();
        let second = publications
            .publish_next(&mut host, &transport)
            .unwrap()
            .unwrap();
        assert_eq!((first.0.semantic, second.0.semantic), (1, 2));
        assert_eq!(host.get_u32(base), 11);
        assert_eq!(host.get_u32(base + 4), 22);
        assert_eq!(host.action_count(HostActionKind::IrqGfxPulse), 2);
        assert_eq!(publications.pending(), 0);
    }

    #[test]
    fn present_coordinator_retains_native_preparation_failure_for_exact_retry() {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let present = crate::runtime::replacement_session::ReplacementResolvedDisplayPresent {
            command:
                crate::runtime::replacement_child_packet::DecodedReplacementDisplayPresent::Swap {
                    display: 0,
                    unidentified_word: 0,
                    mapping: reims_vgpu_protocol::MapperResolvedSurfaceId::new(1),
                    trailing: Box::new([]),
                },
            source: crate::runtime::replacement_session::ReplacementResolvedPresentationSource {
                display_index: 0,
                task: None,
                resource: None,
                backing: reims_vgpu_protocol::BackingId::new(99),
                width: 64,
                height: 32,
                pixel_format: reims_vgpu_protocol::metal_pixel::MTL_FORMAT_BGRA8_UNORM,
            },
        };
        let admitted = runtime
            .admit_present(
                reims_vgpu_protocol::ChannelId::new(0),
                Box::new([]),
                Some(reims_vgpu_core::CompletionStamp::new(0, 71)),
                present,
            )
            .unwrap();
        let transaction = admitted.transaction();
        let mut coordinator = ReplacementPresentCoordinator::default();
        coordinator.admit(admitted, ()).unwrap();

        assert_eq!(
            coordinator.progress_preparation(&mut runtime, transaction),
            Some(ReplacementPresentPreparationProgress::FailedNativePreparation)
        );
        assert_eq!(coordinator.live_presentations(), 1);
        assert!(coordinator.retry_preparation_failure(transaction));
        assert_eq!(
            coordinator.progress_preparation(&mut runtime, transaction),
            Some(ReplacementPresentPreparationProgress::FailedNativePreparation)
        );
        assert_eq!(runtime.execution().runtime().live_transactions(), 1);
    }

    #[test]
    fn mapper_transport_consumes_every_entry_and_irqs_only_when_caught_up() {
        use crate::runtime::host::HostMemory;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut transport = crate::runtime::replacement_transport::ReplacementTransportOwner::new(
            crate::model::PAGE_SHIFT_ARM64E,
        )
        .unwrap();
        let ring_base = 0x4000;
        transport.registers_mut().iosfc.ring_base = ring_base;
        transport.registers_mut().iosfc.producer = 2;
        let mut host = crate::runtime::host::FakeHost::new();
        host.map_range(
            ring_base,
            2 * reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN,
            0,
        );
        for (index, mapping) in [5u32, 6].into_iter().enumerate() {
            let mut bytes = [0; reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN];
            reims_vgpu_core::endian::st32(
                &mut bytes,
                reims_vgpu_protocol::MapperRequestKind::Map.raw(),
            );
            reims_vgpu_core::endian::st32(&mut bytes[4..], mapping);
            reims_vgpu_core::endian::st64(&mut bytes[8..], u64::from(mapping) << 32);
            host.write_gpa(
                ring_base
                    + u64::try_from(index).unwrap()
                        * reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN as u64,
                &bytes,
            )
            .unwrap();
        }

        assert!(matches!(
            dispatch_next_mapper_entry(&mut runtime, &mut transport, &mut host),
            Ok(Some(
                crate::runtime::replacement_session::ReplacementMapperRequestEffect::Map {
                    request: reims_vgpu_protocol::MapperRequestEntry { mapping_id: 5, .. },
                    ..
                }
            ))
        ));
        assert_eq!(transport.registers().iosfc.consumer, 1);
        assert_eq!(host.action_count(HostActionKind::IrqIosfcPulse), 0);
        assert!(matches!(
            dispatch_next_mapper_entry(&mut runtime, &mut transport, &mut host),
            Ok(Some(
                crate::runtime::replacement_session::ReplacementMapperRequestEffect::Map {
                    request: reims_vgpu_protocol::MapperRequestEntry { mapping_id: 6, .. },
                    ..
                }
            ))
        ));
        assert_eq!(transport.registers().iosfc.consumer, 2);
        assert_eq!(host.action_count(HostActionKind::IrqIosfcPulse), 1);
        assert!(matches!(
            dispatch_next_mapper_entry(&mut runtime, &mut transport, &mut host),
            Ok(None)
        ));
    }

    #[test]
    fn iosfc_write_captures_on_the_publisher_and_drains_that_exact_entry() {
        use crate::runtime::host::HostMemory;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut transport = crate::runtime::replacement_transport::ReplacementTransportOwner::new(
            crate::model::PAGE_SHIFT_ARM64E,
        )
        .unwrap();
        let ring_base = 0x4000;
        assert_eq!(
            transport.iosfc_write(crate::model::IOSFC_REG_RING_BASE, ring_base),
            None
        );
        let mut host = crate::runtime::host::FakeHost::new();
        host.map_range(ring_base, reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN, 0);
        let mapping_id = 7;
        let mut bytes = [0; reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN];
        reims_vgpu_core::endian::st32(
            &mut bytes,
            reims_vgpu_protocol::MapperRequestKind::Map.raw(),
        );
        reims_vgpu_core::endian::st32(&mut bytes[4..], mapping_id);
        host.write_gpa(ring_base, &bytes).unwrap();

        let internal = 0xffff_fe00_1000_0000;
        let mapper = internal + 0x1000;
        host.map_range(internal, 0x4000, 0);
        host.write_gpa(
            internal + reims_vgpu_paging::mapper::MAPPING_INTERNAL_BACKPTR,
            &mapper.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            internal + reims_vgpu_paging::mapper::MAPPING_INTERNAL_ID,
            &mapping_id.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            internal + reims_vgpu_paging::mapper::MAPPING_INTERNAL_SIZE,
            &reims_vgpu_paging::mapper::MAPPING_INTERNAL_EXPECTED_SIZE.to_le_bytes(),
        )
        .unwrap();
        let descriptor = internal + 0x1000;
        let page_owner = internal + 0x2000;
        let page_table = internal + 0x3000;
        host.write_gpa(
            internal + reims_vgpu_paging::mapper::MAPPING_INTERNAL_DESC_PTR,
            &descriptor.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            internal + reims_vgpu_paging::mapper::MAPPING_INTERNAL_PAGE_FIELD_48,
            &page_owner.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            internal + reims_vgpu_paging::mapper::MAPPING_INTERNAL_PAGE_COUNT,
            &2u64.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            page_owner + reims_vgpu_paging::mapper::MAPPING_PAGE_TABLE_FROM_F48,
            &page_table.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(page_table, &5u32.to_le_bytes()).unwrap();
        host.write_gpa(page_table + 4, &9u32.to_le_bytes()).unwrap();
        let mut descriptor_bytes = [0u8; reims_vgpu_protocol::DEVICE_DESC_LEN];
        reims_vgpu_core::endian::st32(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_PIXEL_FORMAT..],
            u32::from(reims_vgpu_protocol::metal_pixel::MTL_FORMAT_BGRA8_UNORM),
        );
        reims_vgpu_core::endian::st32(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_ALLOC_SIZE..],
            0x8000,
        );
        reims_vgpu_core::endian::st64(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_DIMS..],
            (64u64 << 8) | (32u64 << 40),
        );
        reims_vgpu_core::endian::st32(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_BPR..],
            256,
        );
        reims_vgpu_core::endian::st16(
            &mut descriptor_bytes[reims_vgpu_protocol::DEVICE_DESC_BPE..],
            4,
        );
        host.write_gpa(descriptor, &descriptor_bytes).unwrap();
        host.set_xreg(
            crate::runtime::replacement_mapper::MAPPER_CAPTURE_REG_MAPPER_DEVICE,
            mapper,
        );
        host.set_xreg(
            crate::runtime::replacement_mapper::MAPPER_CAPTURE_REG_REQUEST_TYPE,
            u64::from(reims_vgpu_protocol::MapperRequestKind::Map.raw()),
        );
        host.set_xreg(
            crate::runtime::replacement_mapper::MAPPER_CAPTURE_REG_MAPPING_INTERNAL,
            internal,
        );

        let effect = transport
            .iosfc_write(crate::model::IOSFC_REG_PRODUCER, 1)
            .unwrap();
        let completed =
            dispatch_iosfc_write_effect(&mut runtime, &mut transport, &mut host, effect)
                .unwrap_or_else(|_| panic!("the complete mapper publication must dispatch"));
        assert!(matches!(
            completed.as_ref(),
            [crate::runtime::replacement_session::ReplacementMapperRequestEffect::Map {
                request: reims_vgpu_protocol::MapperRequestEntry { mapping_id: 7, .. },
                capture: Some(reims_vgpu_core::MapperCapture {
                    producer: 1,
                    mapper_device_kva,
                    mapping_internal,
                    ..
                }),
                ..
            }] if *mapper_device_kva == mapper && *mapping_internal == internal
        ));
        assert_eq!(transport.registers().iosfc.consumer, 1);
        assert_eq!(host.action_count(HostActionKind::IrqIosfcPulse), 1);
        let backing = runtime
            .mapper_backing(reims_vgpu_protocol::MapperResolvedSurfaceId::new(
                mapping_id,
            ))
            .expect("MAP commit requires a published backing");
        assert_eq!(backing.footprint.pages(), &[0x4000, 0x8000]);
    }

    #[test]
    fn mapper_backing_failure_keeps_the_applied_effect_and_lease_for_retry() {
        use crate::runtime::host::HostMemory;

        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut transport = crate::runtime::replacement_transport::ReplacementTransportOwner::new(
            crate::model::PAGE_SHIFT_ARM64E,
        )
        .unwrap();
        let ring_base = 0x4000;
        let mapping_id = 9;
        transport.registers_mut().iosfc.ring_base = ring_base;
        transport.registers_mut().iosfc.producer = 1;
        let mut host = crate::runtime::host::FakeHost::new();
        host.map_range(ring_base, reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN, 0);
        let mut bytes = [0; reims_vgpu_protocol::MAPPER_REQUEST_ENTRY_LEN];
        reims_vgpu_core::endian::st32(
            &mut bytes,
            reims_vgpu_protocol::MapperRequestKind::Map.raw(),
        );
        reims_vgpu_core::endian::st32(&mut bytes[4..], mapping_id);
        host.write_gpa(ring_base, &bytes).unwrap();
        let capture = reims_vgpu_core::MapperCapture {
            producer: 1,
            mapper_device_kva: 0xffff_fe00_2000_1000,
            request_kind: reims_vgpu_protocol::MapperRequestKind::Map,
            mapping_internal: 0xffff_fe00_2000_0000,
        };
        runtime.publish_mapper_capture(capture).unwrap();

        let failure = dispatch_next_mapper_entry(&mut runtime, &mut transport, &mut host)
            .expect_err("an absent mapper internal must retain the entry");
        assert!(matches!(
            failure.as_ref(),
            ReplacementMapperEntryDispatchFailure::Backing { .. }
        ));
        assert_eq!(transport.registers().iosfc.consumer, 0);
        assert_eq!(
            runtime.resolve_mapper_surface(reims_vgpu_protocol::MapperSurfaceRef::new(u64::from(
                mapping_id
            ))),
            Some(reims_vgpu_protocol::MapperResolvedSurfaceId::new(
                mapping_id
            ))
        );
        assert!(runtime
            .mapper_backing(reims_vgpu_protocol::MapperResolvedSurfaceId::new(
                mapping_id
            ))
            .is_none());

        write_mapper_backing_fixture(
            &mut host,
            capture.mapping_internal,
            capture.mapper_device_kva,
            mapping_id,
        );
        let completed =
            retry_mapper_entry_dispatch_failure(&mut runtime, &mut transport, &mut host, *failure)
                .unwrap_or_else(|_| panic!("the retained applied phase must resume"))
                .expect("the retained entry must commit");
        assert!(matches!(
            completed,
            crate::runtime::replacement_session::ReplacementMapperRequestEffect::Map {
                backing_resolved: true,
                ..
            }
        ));
        assert_eq!(transport.registers().iosfc.consumer, 1);
        assert!(runtime
            .mapper_backing(reims_vgpu_protocol::MapperResolvedSurfaceId::new(
                mapping_id
            ))
            .is_some());
    }

    /// A settled position owes nothing; an unsettled one somebody has to hold.
    ///
    /// Reading this the other way is what makes a leak invisible: a transaction
    /// whose owner was dropped looks exactly like one in flight, because both
    /// are "not submitted yet" and no counter distinguishes them.
    #[test]
    fn only_an_unsettled_position_with_no_owner_is_reported_as_unowned() {
        let entry = |transaction: u64, sequence: u64, started, submitted, abandoned| {
            reims_vgpu_core::SubmissionOrderEntry {
                transaction: reims_vgpu_protocol::TransactionId::new(transaction),
                domain: reims_vgpu_protocol::SubmissionDomainId::new(1),
                sequence: reims_vgpu_protocol::DomainSequence::new(sequence),
                recorded: started,
                issued: started,
                submitted,
                abandoned,
            }
        };
        let order = [
            entry(10, 1, true, true, false),
            entry(11, 2, true, false, true),
            entry(12, 3, true, false, false),
            entry(13, 4, true, false, false),
            entry(14, 5, false, false, false),
        ];
        let owned = std::collections::BTreeSet::from([reims_vgpu_protocol::TransactionId::new(12)]);

        let unowned = unowned_submission_positions(&order, &owned);
        assert_eq!(
            unowned
                .iter()
                .map(|(_, position)| position.as_str())
                .collect::<Vec<_>>(),
            ["13@1.4"],
            "submitted and abandoned both release the domain claim, 12 is held, and 14 has not \
             been started by anybody yet so nobody owes it an owner"
        );
    }
}
