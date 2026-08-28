//! Exhaustive replacement cutover classification for the child FIFO opcode space.
//!
//! This is an opcode contract, not a production feature switch. The live FIFO
//! remains on its one legacy route until every `Blocked` family has a complete
//! typed transaction and the device changes owners atomically.

#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketFamily {
    Debug,
    SetupSharedState,
    OnlineAck,
    CursorGlyph,
    CursorShow,
    DisplayTransaction2,
    DisplayTransaction3,
    DisplaySwap,
    DisplaySleepState,
    DisplaySetProperties,
    Nop,
    DeleteTask,
    UnmapMemory,
    DeleteResource,
    DeleteObject,
    SetObjectList,
    InvalidateResources,
    SynchronizeResources,
    DeleteSurfaceBacking,
    ExecIndirect2,
    DefineTask2,
    MapMemory2,
    GetComputeInfo,
    ReplacePhysical,
    Delay,
    SynchronizeAndDiscardResources,
    DiscardResources,
    HeapTextureSizeAndAlign,
    Deprecated(u16),
}

impl ReplacementChildPacketFamily {
    /// The census name for this family.
    ///
    /// Every decoder in `decode/` is silent on success, so "opcode X never
    /// appears in the log" says nothing about whether the guest sent it. This
    /// counter is the only never-fired signal for the child stream, and it is
    /// what distinguishes "the guest asked and we refused" from "the guest
    /// never asked" -- two readings that send an investigation to opposite
    /// halves of the device.
    pub(crate) const fn census_route(self) -> &'static str {
        match self {
            Self::Debug => "child_debug",
            Self::SetupSharedState => "child_setup_shared_state",
            Self::OnlineAck => "child_online_ack",
            Self::CursorGlyph => "child_cursor_glyph",
            Self::CursorShow => "child_cursor_show",
            Self::DisplayTransaction2 => "child_display_transaction2",
            Self::DisplayTransaction3 => "child_display_transaction3",
            Self::DisplaySwap => "child_display_swap",
            Self::DisplaySleepState => "child_display_sleep_state",
            Self::DisplaySetProperties => "child_display_set_properties",
            Self::Nop => "child_nop",
            Self::DeleteTask => "child_delete_task",
            Self::UnmapMemory => "child_unmap_memory",
            Self::DeleteResource => "child_delete_resource",
            Self::DeleteObject => "child_delete_object",
            Self::SetObjectList => "child_set_object_list",
            Self::InvalidateResources => "child_invalidate_resources",
            Self::SynchronizeResources => "child_synchronize_resources",
            Self::DeleteSurfaceBacking => "child_delete_surface_backing",
            Self::ExecIndirect2 => "child_exec_indirect2",
            Self::DefineTask2 => "child_define_task2",
            Self::MapMemory2 => "child_map_memory2",
            Self::GetComputeInfo => "child_get_compute_info",
            Self::ReplacePhysical => "child_replace_physical",
            Self::Delay => "child_delay",
            Self::SynchronizeAndDiscardResources => "child_synchronize_and_discard_resources",
            Self::DiscardResources => "child_discard_resources",
            Self::HeapTextureSizeAndAlign => "child_heap_texture_size_and_align",
            Self::Deprecated(_) => "child_deprecated",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildCutoverStatus {
    Implemented,
    ContractNoOp,
    TypedRefusal,
    Blocked,
}

impl ReplacementChildCutoverStatus {
    /// The census name for a status that costs the guest its packet, or `None`
    /// for one that does not. An implemented family needs no second counter:
    /// its own family count already says the guest asked and was served.
    pub(crate) const fn census_route(self) -> Option<&'static str> {
        match self {
            Self::Implemented => None,
            Self::ContractNoOp => Some("child_contract_noop"),
            Self::TypedRefusal => Some("child_typed_refusal"),
            Self::Blocked => Some("child_blocked"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementChildOpcode {
    pub family: ReplacementChildPacketFamily,
    pub status: ReplacementChildCutoverStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildOpcodeError {
    Unassigned(u16),
    OutOfRange { opcode: u16, max: u16 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementOnlineAck {
    /// The acknowledgement payload is retained for diagnostics; no field in it
    /// is currently part of the display-handshake decision.
    pub payload: Box<[u8]>,
}

pub(crate) fn decode_replacement_online_ack(payload: &[u8]) -> DecodedReplacementOnlineAck {
    DecodedReplacementOnlineAck {
        payload: payload.into(),
    }
}

pub(crate) fn apply_replacement_online_ack<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    _ack: DecodedReplacementOnlineAck,
) -> Option<reims_vgpu_core::DisplaySharedPage> {
    runtime.acknowledge_display_online()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementCursorShow {
    pub display_index: u32,
    pub visible: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementCursorShowDecodeError {
    Short { plen: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DecodedReplacementDisplayPresent {
    Transaction2 {
        pipe: u32,
        surface: u32,
        task: reims_vgpu_protocol::TaskId,
        trailing: Box<[u8]>,
    },
    Transaction3 {
        pipe: u32,
        task: reims_vgpu_protocol::TaskId,
        surface: u32,
        gamma: [u8; 24],
        trailing: Box<[u8]>,
    },
    Swap {
        display: u32,
        unidentified_word: u32,
        mapping: reims_vgpu_protocol::MapperResolvedSurfaceId,
        trailing: Box<[u8]>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementDisplayPresentDecodeError {
    Short {
        family: ReplacementChildPacketFamily,
        expected: usize,
        actual: usize,
    },
}

pub(crate) fn decode_replacement_display_present(
    family: ReplacementChildPacketFamily,
    payload: &[u8],
) -> Result<DecodedReplacementDisplayPresent, ReplacementDisplayPresentDecodeError> {
    let expected = match family {
        ReplacementChildPacketFamily::DisplayTransaction2
        | ReplacementChildPacketFamily::DisplaySwap => 12,
        ReplacementChildPacketFamily::DisplayTransaction3 => 36,
        _ => unreachable!("only display-present families have a present payload"),
    };
    if payload.len() < expected {
        return Err(ReplacementDisplayPresentDecodeError::Short {
            family,
            expected,
            actual: payload.len(),
        });
    }
    Ok(match family {
        ReplacementChildPacketFamily::DisplayTransaction2 => {
            DecodedReplacementDisplayPresent::Transaction2 {
                pipe: reims_vgpu_core::endian::ld32(payload),
                surface: reims_vgpu_core::endian::ld32(&payload[4..]),
                task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(
                    &payload[8..],
                )),
                trailing: payload[expected..].into(),
            }
        }
        ReplacementChildPacketFamily::DisplayTransaction3 => {
            DecodedReplacementDisplayPresent::Transaction3 {
                pipe: reims_vgpu_core::endian::ld32(payload),
                task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(
                    &payload[4..],
                )),
                surface: reims_vgpu_core::endian::ld32(&payload[8..]),
                gamma: payload[12..36]
                    .try_into()
                    .expect("the fixed display transaction was length-validated"),
                trailing: payload[expected..].into(),
            }
        }
        ReplacementChildPacketFamily::DisplaySwap => DecodedReplacementDisplayPresent::Swap {
            display: reims_vgpu_core::endian::ld32(payload),
            unidentified_word: reims_vgpu_core::endian::ld32(&payload[4..]),
            mapping: reims_vgpu_protocol::MapperResolvedSurfaceId::new(
                reims_vgpu_core::endian::ld32(&payload[8..]),
            ),
            trailing: payload[expected..].into(),
        },
        _ => unreachable!("only display-present families have a present payload"),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementCursorGlyph {
    pub display_index: u32,
    pub task: reims_vgpu_protocol::TaskId,
    pub gva: reims_vgpu_protocol::GuestVirtualAddress,
    pub mapped_length: u64,
    pub stride: u64,
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub trailing_word: u32,
    pub read_length: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementCursorGlyphDecodeError {
    Short { plen: usize },
    InvalidGeometry,
    HostLengthOverflow { length: u64 },
    MappedLengthShort { mapped: u64, required: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementCursorGlyph {
    pub display_index: u32,
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub pixels: Box<[u32]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementResourceDelete {
    pub task: reims_vgpu_protocol::TaskId,
    pub object: reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementResourceDeleteDecodeError {
    Short { plen: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementComputeInfo {
    pub task: reims_vgpu_protocol::TaskId,
    pub pipeline_ref: u32,
    pub key_table_len: u32,
    pub count: u32,
    pub reply_gva: reims_vgpu_protocol::GuestVirtualAddress,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeInfoDecodeError {
    Short { plen: usize },
    EmptyReply,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementComputeInfoReply {
    pub task: reims_vgpu_protocol::TaskId,
    pub pipeline_ref: u32,
    pub gva: reims_vgpu_protocol::GuestVirtualAddress,
    pub bytes: Box<[u8]>,
    pub answered_keys: Box<[u32]>,
    pub dropped_keys: Box<[u32]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementComputeInfoReplyError {
    AddressOverflow {
        gva: reims_vgpu_protocol::GuestVirtualAddress,
        byte_len: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementDiscardResources {
    pub task: reims_vgpu_protocol::TaskId,
    pub objects: Box<[reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementSynchronizeResources {
    pub task: reims_vgpu_protocol::TaskId,
    pub objects: Box<[reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>]>,
    pub discard_transfer_backing: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementHeapTextureReply {
    pub task: reims_vgpu_protocol::TaskId,
    pub gva: reims_vgpu_protocol::GuestVirtualAddress,
    pub bytes: [u8; crate::runtime::heap_query::REPLY_LEN],
}

#[derive(Clone, Debug)]
pub(crate) enum ReplacementHeapTextureReplyError {
    AddressOverflow(reims_vgpu_protocol::GuestVirtualAddress),
    Requirements(
        reims_vgpu_vulkan::replacement_representation::ReplacementTextureRequirementsError,
    ),
}

pub(crate) type DecodedReplacementPhysicalReplacement = DecodedReplacementResourceDelete;
pub(crate) type ReplacementPhysicalReplacementDecodeError = ReplacementResourceDeleteDecodeError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementMappingChange {
    pub family: ReplacementChildPacketFamily,
    pub task: reims_vgpu_protocol::TaskId,
    pub address: reims_vgpu_protocol::GuestVirtualAddress,
    pub length: reims_vgpu_protocol::ByteLength,
    pub trailing: Box<[u8]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementMappingChangeDecodeError {
    Short { plen: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementSurfaceBackingDelete {
    pub object: reims_vgpu_protocol::ObjectTableRef<reims_vgpu_protocol::ResourceObject>,
    pub task: reims_vgpu_protocol::TaskId,
}

pub(crate) type ReplacementSurfaceBackingDeleteDecodeError = ReplacementResourceDeleteDecodeError;

pub(crate) fn decode_replacement_surface_backing_delete(
    payload: &[u8],
) -> Result<DecodedReplacementSurfaceBackingDelete, ReplacementSurfaceBackingDeleteDecodeError> {
    const LEN: usize = 2 * size_of::<u32>();
    if payload.len() < LEN {
        return Err(ReplacementResourceDeleteDecodeError::Short {
            plen: payload.len(),
        });
    }
    Ok(DecodedReplacementSurfaceBackingDelete {
        object: reims_vgpu_protocol::ObjectTableRef::new(reims_vgpu_core::endian::ld32(payload)),
        task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(&payload[4..])),
    })
}

pub(crate) fn decode_replacement_mapping_change(
    family: ReplacementChildPacketFamily,
    payload: &[u8],
) -> Result<DecodedReplacementMappingChange, ReplacementMappingChangeDecodeError> {
    const LEN: usize = size_of::<u32>() + 2 * size_of::<u64>();
    if payload.len() < LEN {
        return Err(ReplacementMappingChangeDecodeError::Short {
            plen: payload.len(),
        });
    }
    Ok(DecodedReplacementMappingChange {
        family,
        task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(payload)),
        address: reims_vgpu_protocol::GuestVirtualAddress::new(reims_vgpu_core::endian::ld64(
            &payload[4..],
        )),
        length: reims_vgpu_protocol::ByteLength::new(reims_vgpu_core::endian::ld64(&payload[12..])),
        trailing: payload[LEN..].into(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketRoute {
    Control(crate::runtime::replacement_session::ReplacementControlCommand),
    Query(crate::runtime::replacement_session::ReplacementQueryCommand),
    DeleteResource(DecodedReplacementResourceDelete),
    ReplacePhysical(DecodedReplacementPhysicalReplacement),
    MappingChange(DecodedReplacementMappingChange),
    DeleteSurfaceBacking(DecodedReplacementSurfaceBackingDelete),
    SynchronizeResources(DecodedReplacementSynchronizeResources),
    InvalidateResources(crate::runtime::decode::fifo::InvalidateResourcesCommand),
    Exec(crate::runtime::decode::fifo::ExecIndirect2Command),
    CursorGlyph(DecodedReplacementCursorGlyph),
    Present(DecodedReplacementDisplayPresent),
    Refused(ReplacementChildPacketFamily),
    Blocked(ReplacementChildPacketFamily),
}

/// A decoded EXEC that still owns the FIFO envelope from which it came.
///
/// Loading command-buffer bytes may be retried, but neither that transport
/// step nor eventual admission can replace the packet's ordering facts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementDeferredExecPacket {
    pub envelope: crate::runtime::replacement_packet::ReplacementPacketEnvelope,
    pub command: crate::runtime::decode::fifo::ExecIndirect2Command,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementDeferredCursorGlyph {
    pub envelope: crate::runtime::replacement_packet::ReplacementPacketEnvelope,
    pub command: DecodedReplacementCursorGlyph,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementDeferredSynchronizeResources {
    pub envelope: crate::runtime::replacement_packet::ReplacementPacketEnvelope,
    pub command: DecodedReplacementSynchronizeResources,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementLoadedCursorGlyph {
    envelope: crate::runtime::replacement_packet::ReplacementPacketEnvelope,
    glyph: ReplacementCursorGlyph,
}

#[derive(Debug)]
pub(crate) struct ReplacementCursorGlyphAdmissionFailure {
    pub reason: reims_vgpu_core::TransactionRuntimeError,
    pub loaded: ReplacementLoadedCursorGlyph,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementCursorGlyphLoadError<Transport> {
    UnknownTask(reims_vgpu_protocol::TaskId),
    Read(Transport),
    ReadLengthMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketTransport {
    Exec(ReplacementDeferredExecPacket),
    CursorGlyph(ReplacementDeferredCursorGlyph),
    Synchronize(ReplacementDeferredSynchronizeResources),
    Blocked {
        envelope: crate::runtime::replacement_packet::ReplacementPacketEnvelope,
        family: ReplacementChildPacketFamily,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketDecodeError {
    Opcode(ReplacementChildOpcodeError),
    DisplayPresent(ReplacementDisplayPresentDecodeError),
    CursorShow(ReplacementCursorShowDecodeError),
    CursorGlyph(ReplacementCursorGlyphDecodeError),
    DeleteTask(crate::runtime::replacement_task::ReplacementTaskDeleteDecodeError),
    DeleteResource(ReplacementResourceDeleteDecodeError),
    SetObjectList(crate::runtime::replacement_task::ReplacementObjectListDecodeError),
    Exec(crate::runtime::decode::fifo::ExecIndirect2DecodeError),
    DefineTask(crate::runtime::replacement_task::ReplacementTaskDefinitionDecodeError),
    ReplacePhysical(ReplacementPhysicalReplacementDecodeError),
    MappingChange(ReplacementMappingChangeDecodeError),
    DeleteSurfaceBacking(ReplacementSurfaceBackingDeleteDecodeError),
    ComputeInfo(ReplacementComputeInfoDecodeError),
    HeapTexture(crate::runtime::heap_query::QueryError),
    DiscardResources(crate::runtime::decode::fifo::ResourceListDecodeError),
    InvalidateResources(crate::runtime::decode::fifo::ResourceListDecodeError),
    SynchronizeResources(crate::runtime::decode::fifo::ResourceListDecodeError),
    DeleteObject(crate::runtime::replacement_object_lifecycle::ReplacementObjectDeleteDecodeError),
    ShortSharedState { plen: usize },
}

#[derive(Debug)]
pub(crate) enum ReplacementAdmittedChildCpuPacket<Semantic> {
    Control(crate::runtime::replacement_session::ReplacementAdmittedControl<Semantic>),
    Query(crate::runtime::replacement_session::ReplacementAdmittedQuery<Semantic>),
    ResourceLifecycle(
        crate::runtime::replacement_session::ReplacementAdmittedResourceLifecycle<Semantic>,
    ),
    Present(crate::runtime::replacement_session::ReplacementAdmittedPresent<Semantic>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildCpuPacketIngressError {
    Decode(ReplacementChildPacketDecodeError),
    RequiresTransport(ReplacementChildPacketTransport),
    TypedRefusal {
        envelope: crate::runtime::replacement_packet::ReplacementPacketEnvelope,
        family: ReplacementChildPacketFamily,
    },
    PresentResolution {
        reason: crate::runtime::replacement_session::ReplacementDisplayPresentResolutionError,
        route: ReplacementChildPacketRoute,
    },
    PresentAdmission {
        reason: reims_vgpu_core::TransactionRuntimeError,
        route: ReplacementChildPacketRoute,
    },
    ControlAdmission {
        reason: reims_vgpu_core::TransactionRuntimeError,
        route: ReplacementChildPacketRoute,
    },
    QueryAdmission {
        reason: ReplacementChildQueryAdmissionError,
        route: ReplacementChildPacketRoute,
    },
    ResourceAdmission {
        reason:
            crate::runtime::replacement_session::ReplacementNamedResourceLifecycleAdmissionError,
        route: ReplacementChildPacketRoute,
    },
    InvalidationAdmission {
        reason: ReplacementChildInvalidationAdmissionError,
        route: ReplacementChildPacketRoute,
    },
    MappingChangeAdmission {
        reason: ReplacementMappingChangeAdmissionError,
        route: ReplacementChildPacketRoute,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildInvalidationAdmissionError {
    Resolution(crate::runtime::replacement_session::ReplacementStandaloneInvalidationError),
    Admission(reims_vgpu_core::TransactionRuntimeError),
}

impl ReplacementChildCpuPacketIngressError {
    /// Whether re-offering this packet asks a question the guest has already
    /// answered, so the only thing retrying it buys is the channel behind it.
    ///
    /// See
    /// [`crate::runtime::replacement_session::ReplacementStandaloneInvalidationError::is_terminal_refusal`]
    /// and
    /// [`crate::runtime::replacement_session::ReplacementNamedResourceLifecycleAdmissionError::is_terminal_refusal`],
    /// which carry the argument for their own arms. Everything else here waits
    /// on state a later packet supplies -- a task, a mapping, a display
    /// present's resolution -- and the retry is exactly what delivers it.
    pub(crate) const fn is_terminal_refusal(&self) -> bool {
        match self {
            Self::InvalidationAdmission { reason, .. } => match reason {
                ReplacementChildInvalidationAdmissionError::Resolution(reason) => {
                    reason.is_terminal_refusal()
                }
                ReplacementChildInvalidationAdmissionError::Admission(_) => false,
            },
            Self::ResourceAdmission { reason, .. } => reason.is_terminal_refusal(),
            Self::Decode(_)
            | Self::RequiresTransport(_)
            | Self::TypedRefusal { .. }
            | Self::PresentResolution { .. }
            | Self::PresentAdmission { .. }
            | Self::ControlAdmission { .. }
            | Self::QueryAdmission { .. }
            | Self::MappingChangeAdmission { .. } => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementMappingChangeAdmissionError {
    Resolution(crate::runtime::replacement_session::ReplacementTaskAddressReplacementError),
    Admission(reims_vgpu_core::TransactionRuntimeError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildQueryAdmissionError {
    UnknownTask(reims_vgpu_protocol::TaskId),
    Admission(reims_vgpu_core::TransactionRuntimeError),
}

pub(crate) fn decode_replacement_child_packet(
    opcode: u16,
    payload: &[u8],
) -> Result<ReplacementChildPacketRoute, ReplacementChildPacketDecodeError> {
    use crate::model::*;
    use crate::runtime::replacement_session::ReplacementControlCommand as Control;

    let disposition = match classify_replacement_child_opcode(opcode) {
        Ok(disposition) => disposition,
        Err(reason) => {
            crate::runtime::contract_census::note(match reason {
                ReplacementChildOpcodeError::Unassigned(_) => "child_opcode_unassigned",
                ReplacementChildOpcodeError::OutOfRange { .. } => "child_opcode_out_of_range",
            });
            return Err(ReplacementChildPacketDecodeError::Opcode(reason));
        }
    };
    crate::runtime::contract_census::note(disposition.family.census_route());
    if let Some(route) = disposition.status.census_route() {
        crate::runtime::contract_census::note(route);
    }
    match disposition.status {
        ReplacementChildCutoverStatus::Blocked => {
            Ok(ReplacementChildPacketRoute::Blocked(disposition.family))
        }
        ReplacementChildCutoverStatus::ContractNoOp => Ok(ReplacementChildPacketRoute::Control(
            Control::ContractNoOp(opcode),
        )),
        ReplacementChildCutoverStatus::TypedRefusal => {
            Ok(ReplacementChildPacketRoute::Refused(disposition.family))
        }
        ReplacementChildCutoverStatus::Implemented => match opcode {
            CHILD_OP_DEBUG => Ok(ReplacementChildPacketRoute::Control(Control::Debug(
                payload.into(),
            ))),
            CHILD_OP_SETUP_SHARED_STATE => {
                if payload.len() < CHILD_SHARED_STATE_LEN {
                    return Err(ReplacementChildPacketDecodeError::ShortSharedState {
                        plen: payload.len(),
                    });
                }
                Ok(ReplacementChildPacketRoute::Control(
                    Control::SetupSharedState(
                        crate::runtime::replacement_fifo_control::DecodedReplacementSharedState {
                            index: reims_vgpu_core::endian::ld32(
                                &payload[CHILD_SHARED_STATE_INDEX..],
                            ),
                            pfn: reims_vgpu_core::endian::ld32(&payload[CHILD_SHARED_STATE_PFN..]),
                        },
                    ),
                ))
            }
            CHILD_OP_ONLINE_ACK => Ok(ReplacementChildPacketRoute::Control(Control::OnlineAck(
                decode_replacement_online_ack(payload),
            ))),
            CHILD_OP_CURSOR_SHOW => decode_replacement_cursor_show(payload)
                .map(Control::CursorShow)
                .map(ReplacementChildPacketRoute::Control)
                .map_err(ReplacementChildPacketDecodeError::CursorShow),
            CHILD_OP_CURSOR_GLYPH => decode_replacement_cursor_glyph(payload)
                .map(ReplacementChildPacketRoute::CursorGlyph)
                .map_err(ReplacementChildPacketDecodeError::CursorGlyph),
            CHILD_OP_DISPLAY_TRANSACTION2
            | CHILD_OP_DISPLAY_TRANSACTION3
            | CHILD_OP_DISPLAY_SWAP => {
                decode_replacement_display_present(disposition.family, payload)
                    .map(ReplacementChildPacketRoute::Present)
                    .map_err(ReplacementChildPacketDecodeError::DisplayPresent)
            }
            CHILD_OP_DELETE_TASK => {
                crate::runtime::replacement_task::decode_replacement_task_delete(payload)
                    .map(Control::DeleteTask)
                    .map(ReplacementChildPacketRoute::Control)
                    .map_err(ReplacementChildPacketDecodeError::DeleteTask)
            }
            CHILD_OP_MAP_MEMORY2 | CHILD_OP_UNMAP_MEMORY => {
                decode_replacement_mapping_change(disposition.family, payload)
                    .map(ReplacementChildPacketRoute::MappingChange)
                    .map_err(ReplacementChildPacketDecodeError::MappingChange)
            }
            CHILD_OP_DELETE_RESOURCE => decode_replacement_resource_delete(payload)
                .map(ReplacementChildPacketRoute::DeleteResource)
                .map_err(ReplacementChildPacketDecodeError::DeleteResource),
            CHILD_OP_DELETE_IOSURFACE_BACKING2 => {
                decode_replacement_surface_backing_delete(payload)
                    .map(ReplacementChildPacketRoute::DeleteSurfaceBacking)
                    .map_err(ReplacementChildPacketDecodeError::DeleteSurfaceBacking)
            }
            CHILD_OP_DELETE_OBJECT => {
                crate::runtime::replacement_object_lifecycle::decode_replacement_object_delete(
                    payload,
                )
                .map(Control::DeleteObject)
                .map(ReplacementChildPacketRoute::Control)
                .map_err(ReplacementChildPacketDecodeError::DeleteObject)
            }
            CHILD_OP_SET_OBJECT_LIST => {
                crate::runtime::replacement_task::decode_replacement_object_list(payload)
                    .map(Control::SetObjectList)
                    .map(ReplacementChildPacketRoute::Control)
                    .map_err(ReplacementChildPacketDecodeError::SetObjectList)
            }
            CHILD_OP_EXEC_INDIRECT2 => crate::runtime::decode::fifo::decode_exec_indirect2(payload)
                .map(ReplacementChildPacketRoute::Exec)
                .map_err(ReplacementChildPacketDecodeError::Exec),
            CHILD_OP_GET_COMPUTE_INFO => decode_replacement_compute_info(payload)
                .map(crate::runtime::replacement_session::ReplacementQueryCommand::ComputeInfo)
                .map(ReplacementChildPacketRoute::Query)
                .map_err(ReplacementChildPacketDecodeError::ComputeInfo),
            CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN => {
                crate::runtime::heap_query::decode_request(payload)
                    .map(crate::runtime::replacement_session::ReplacementQueryCommand::HeapTexture)
                    .map(ReplacementChildPacketRoute::Query)
                    .map_err(ReplacementChildPacketDecodeError::HeapTexture)
            }
            CHILD_OP_DISCARD_RESOURCES => decode_replacement_discard_resources(payload)
                .map(Control::DiscardResources)
                .map(ReplacementChildPacketRoute::Control)
                .map_err(ReplacementChildPacketDecodeError::DiscardResources),
            CHILD_OP_INVALIDATE_RESOURCES => {
                crate::runtime::decode::fifo::decode_invalidate_resources(payload)
                    .map(ReplacementChildPacketRoute::InvalidateResources)
                    .map_err(ReplacementChildPacketDecodeError::InvalidateResources)
            }
            CHILD_OP_SYNCHRONIZE_RESOURCES | CHILD_OP_SYNCHRONIZE_AND_DISCARD_RESOURCES => {
                decode_replacement_synchronize_resources(disposition.family, payload)
                    .map(ReplacementChildPacketRoute::SynchronizeResources)
                    .map_err(ReplacementChildPacketDecodeError::SynchronizeResources)
            }
            CHILD_OP_DEFINE_TASK2 => {
                crate::runtime::replacement_task::decode_replacement_task_definition(payload)
                    .map(Control::DefineTask)
                    .map(ReplacementChildPacketRoute::Control)
                    .map_err(ReplacementChildPacketDecodeError::DefineTask)
            }
            CHILD_OP_REPLACE_PHYSICAL => decode_replacement_physical_replacement(payload)
                .map(ReplacementChildPacketRoute::ReplacePhysical)
                .map_err(ReplacementChildPacketDecodeError::ReplacePhysical),
            _ => {
                unreachable!("the exhaustive disposition table marked no other opcode implemented")
            }
        },
    }
}

pub(crate) fn admit_replacement_child_cpu_packet<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    channel: reims_vgpu_protocol::ChannelId,
    packet: crate::runtime::fifo_packet::Packet,
) -> Result<ReplacementAdmittedChildCpuPacket<Semantic>, ReplacementChildCpuPacketIngressError> {
    let envelope =
        crate::runtime::replacement_packet::ReplacementPacketEnvelope::from_packet(channel, packet);
    let route = decode_replacement_child_packet(envelope.opcode, &envelope.payload)
        .map_err(ReplacementChildCpuPacketIngressError::Decode)?;
    match &route {
        ReplacementChildPacketRoute::Control(command) => runtime
            .admit_control(
                envelope.channel,
                envelope.prerequisites,
                Some(envelope.completion_stamp),
                command.clone(),
            )
            .map(ReplacementAdmittedChildCpuPacket::Control)
            .map_err(
                |reason| ReplacementChildCpuPacketIngressError::ControlAdmission {
                    reason,
                    route: route.clone(),
                },
            ),
        ReplacementChildPacketRoute::Query(command) => {
            let task = match command {
                crate::runtime::replacement_session::ReplacementQueryCommand::ComputeInfo(
                    request,
                ) => request.task,
                crate::runtime::replacement_session::ReplacementQueryCommand::HeapTexture(
                    request,
                ) => reims_vgpu_protocol::TaskId::new(request.task_id),
                crate::runtime::replacement_session::ReplacementQueryCommand::DeviceInfo(_) => {
                    unreachable!("device-info queries are root-only")
                }
            };
            if !runtime.tasks().is_active(task.get()) {
                return Err(ReplacementChildCpuPacketIngressError::QueryAdmission {
                    reason: ReplacementChildQueryAdmissionError::UnknownTask(task),
                    route,
                });
            }
            runtime
                .admit_query(
                    envelope.channel,
                    envelope.prerequisites,
                    Some(envelope.completion_stamp),
                    command.clone(),
                )
                .map(ReplacementAdmittedChildCpuPacket::Query)
                .map_err(
                    |reason| ReplacementChildCpuPacketIngressError::QueryAdmission {
                        reason: ReplacementChildQueryAdmissionError::Admission(reason),
                        route,
                    },
                )
        }
        ReplacementChildPacketRoute::DeleteResource(deletion) => {
            if runtime
                .resolve_resource(deletion.task, deletion.object)
                .is_none()
            {
                runtime
                    .admit_control(
                        envelope.channel,
                        envelope.prerequisites,
                        Some(envelope.completion_stamp),
                        crate::runtime::replacement_session::ReplacementControlCommand::AbsentResourceDelete {
                            task: deletion.task,
                            object: deletion.object,
                        },
                    )
                    .map(ReplacementAdmittedChildCpuPacket::Control)
                    .map_err(|reason| ReplacementChildCpuPacketIngressError::ControlAdmission {
                        reason,
                        route: route.clone(),
                    })
            } else {
                admit_replacement_resource_delete(
                    runtime,
                    envelope.channel,
                    envelope.prerequisites,
                    Some(envelope.completion_stamp),
                    *deletion,
                )
                .map(ReplacementAdmittedChildCpuPacket::ResourceLifecycle)
                .map_err(|reason| {
                    ReplacementChildCpuPacketIngressError::ResourceAdmission {
                        reason,
                        route: route.clone(),
                    }
                })
            }
        }
        ReplacementChildPacketRoute::ReplacePhysical(replacement) => {
            admit_replacement_physical_replacement(
                runtime,
                envelope.channel,
                envelope.prerequisites,
                Some(envelope.completion_stamp),
                *replacement,
            )
            .map(ReplacementAdmittedChildCpuPacket::ResourceLifecycle)
            .map_err(|reason| {
                ReplacementChildCpuPacketIngressError::ResourceAdmission {
                    reason,
                    route: route.clone(),
                }
            })
        }
        ReplacementChildPacketRoute::MappingChange(change) => {
            let lifecycle = runtime
                .resolve_task_address_physical_replacement(
                    change.task,
                    change.address,
                    change.length,
                )
                .map_err(|reason| {
                    ReplacementChildCpuPacketIngressError::MappingChangeAdmission {
                        reason: ReplacementMappingChangeAdmissionError::Resolution(reason),
                        route: route.clone(),
                    }
                })?;
            runtime
                .admit_resource_lifecycle(
                    envelope.channel,
                    envelope.prerequisites,
                    Some(envelope.completion_stamp),
                    lifecycle,
                )
                .map(ReplacementAdmittedChildCpuPacket::ResourceLifecycle)
                .map_err(
                    |reason| ReplacementChildCpuPacketIngressError::MappingChangeAdmission {
                        reason: ReplacementMappingChangeAdmissionError::Admission(reason),
                        route: route.clone(),
                    },
                )
        }
        ReplacementChildPacketRoute::DeleteSurfaceBacking(deletion) => runtime
            .admit_task_surface_backing_delete(
                envelope.channel,
                envelope.prerequisites,
                Some(envelope.completion_stamp),
                deletion.task,
                deletion.object,
            )
            .map(ReplacementAdmittedChildCpuPacket::ResourceLifecycle)
            .map_err(
                |reason| ReplacementChildCpuPacketIngressError::ResourceAdmission {
                    reason,
                    route: route.clone(),
                },
            ),
        ReplacementChildPacketRoute::InvalidateResources(command) => {
            let resolved = runtime
                .resolve_standalone_invalidation(command)
                .map_err(
                    |reason| ReplacementChildCpuPacketIngressError::InvalidationAdmission {
                        reason: ReplacementChildInvalidationAdmissionError::Resolution(reason),
                        route: route.clone(),
                    },
                )?;
            // Nothing left to move: the packet is admitted as a contract no-op
            // so it consumes its completion stamp and advances its channel,
            // which is what separates "already true" from "refused".
            let lifecycle = match resolved {
                crate::runtime::replacement_session::ReplacementResolvedInvalidation::Lifecycle(
                    lifecycle,
                ) => *lifecycle,
                crate::runtime::replacement_session::ReplacementResolvedInvalidation::Satisfied => {
                    return runtime
                        .admit_control(
                            envelope.channel,
                            envelope.prerequisites,
                            Some(envelope.completion_stamp),
                            crate::runtime::replacement_session::ReplacementControlCommand::ContractNoOp(
                                envelope.opcode,
                            ),
                        )
                        .map(ReplacementAdmittedChildCpuPacket::Control)
                        .map_err(|reason| {
                            ReplacementChildCpuPacketIngressError::ControlAdmission {
                                reason,
                                route: route.clone(),
                            }
                        });
                }
            };
            runtime
                .admit_resource_lifecycle(
                    envelope.channel,
                    envelope.prerequisites,
                    Some(envelope.completion_stamp),
                    lifecycle,
                )
                .map(ReplacementAdmittedChildCpuPacket::ResourceLifecycle)
                .map_err(
                    |reason| ReplacementChildCpuPacketIngressError::InvalidationAdmission {
                        reason: ReplacementChildInvalidationAdmissionError::Admission(reason),
                        route: route.clone(),
                    },
                )
        }
        ReplacementChildPacketRoute::SynchronizeResources(command) => {
            if command.objects.is_empty() {
                return runtime
                    .admit_control(
                        envelope.channel,
                        envelope.prerequisites,
                        Some(envelope.completion_stamp),
                        crate::runtime::replacement_session::ReplacementControlCommand::ContractNoOp(
                            envelope.opcode,
                        ),
                    )
                    .map(ReplacementAdmittedChildCpuPacket::Control)
                    .map_err(|reason| {
                        ReplacementChildCpuPacketIngressError::ControlAdmission {
                            reason,
                            route: route.clone(),
                        }
                    });
            }
            Err(ReplacementChildCpuPacketIngressError::RequiresTransport(
                ReplacementChildPacketTransport::Synchronize(
                    ReplacementDeferredSynchronizeResources {
                        envelope,
                        command: command.clone(),
                    },
                ),
            ))
        }
        ReplacementChildPacketRoute::Exec(command) => {
            Err(ReplacementChildCpuPacketIngressError::RequiresTransport(
                ReplacementChildPacketTransport::Exec(ReplacementDeferredExecPacket {
                    envelope,
                    command: command.clone(),
                }),
            ))
        }
        ReplacementChildPacketRoute::CursorGlyph(command) => {
            Err(ReplacementChildCpuPacketIngressError::RequiresTransport(
                ReplacementChildPacketTransport::CursorGlyph(ReplacementDeferredCursorGlyph {
                    envelope,
                    command: *command,
                }),
            ))
        }
        ReplacementChildPacketRoute::Present(command) => {
            let present = runtime
                .resolve_display_present(command.clone())
                .map_err(
                    |reason| ReplacementChildCpuPacketIngressError::PresentResolution {
                        reason,
                        route: route.clone(),
                    },
                )?;
            runtime
                .admit_present(
                    envelope.channel,
                    envelope.prerequisites,
                    Some(envelope.completion_stamp),
                    present,
                )
                .map(ReplacementAdmittedChildCpuPacket::Present)
                .map_err(
                    |reason| ReplacementChildCpuPacketIngressError::PresentAdmission {
                        reason,
                        route,
                    },
                )
        }
        ReplacementChildPacketRoute::Refused(family) => {
            Err(ReplacementChildCpuPacketIngressError::TypedRefusal {
                envelope,
                family: *family,
            })
        }
        ReplacementChildPacketRoute::Blocked(family) => {
            Err(ReplacementChildCpuPacketIngressError::RequiresTransport(
                ReplacementChildPacketTransport::Blocked {
                    envelope,
                    family: *family,
                },
            ))
        }
    }
}

pub(crate) fn decode_replacement_cursor_show(
    payload: &[u8],
) -> Result<DecodedReplacementCursorShow, ReplacementCursorShowDecodeError> {
    const LEN: usize = 2 * size_of::<u32>();
    if payload.len() < LEN {
        return Err(ReplacementCursorShowDecodeError::Short {
            plen: payload.len(),
        });
    }
    Ok(DecodedReplacementCursorShow {
        display_index: reims_vgpu_core::endian::ld32(payload),
        visible: reims_vgpu_core::endian::ld32(&payload[size_of::<u32>()..]) != 0,
    })
}

pub(crate) fn decode_replacement_cursor_glyph(
    payload: &[u8],
) -> Result<DecodedReplacementCursorGlyph, ReplacementCursorGlyphDecodeError> {
    if payload.len() < crate::model::CURSOR_GLYPH_PAYLOAD_LEN {
        return Err(ReplacementCursorGlyphDecodeError::Short {
            plen: payload.len(),
        });
    }
    let stride = reims_vgpu_core::endian::ld64(&payload[0x18..]);
    let width = reims_vgpu_core::endian::ld16(&payload[0x20..]);
    let height = reims_vgpu_core::endian::ld16(&payload[0x22..]);
    let hot_x = reims_vgpu_core::endian::ld16(&payload[0x24..]);
    let hot_y = reims_vgpu_core::endian::ld16(&payload[0x26..]);
    if width == 0
        || height == 0
        || u32::from(width) > crate::model::CURSOR_MAX_DIM
        || u32::from(height) > crate::model::CURSOR_MAX_DIM
        || stride < u64::from(width) * u64::from(crate::model::CURSOR_GLYPH_BPP)
        || hot_x >= width
        || hot_y >= height
    {
        return Err(ReplacementCursorGlyphDecodeError::InvalidGeometry);
    }
    let required = u64::from(height - 1)
        .checked_mul(stride)
        .and_then(|prefix| {
            prefix.checked_add(u64::from(width) * u64::from(crate::model::CURSOR_GLYPH_BPP))
        })
        .ok_or(ReplacementCursorGlyphDecodeError::HostLengthOverflow { length: u64::MAX })?;
    let mapped_length = reims_vgpu_core::endian::ld64(&payload[0x10..]);
    if mapped_length < required {
        return Err(ReplacementCursorGlyphDecodeError::MappedLengthShort {
            mapped: mapped_length,
            required,
        });
    }
    let read_length = usize::try_from(required)
        .map_err(|_| ReplacementCursorGlyphDecodeError::HostLengthOverflow { length: required })?;
    Ok(DecodedReplacementCursorGlyph {
        display_index: reims_vgpu_core::endian::ld32(payload),
        task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(&payload[0x04..])),
        gva: reims_vgpu_protocol::GuestVirtualAddress::new(reims_vgpu_core::endian::ld64(
            &payload[0x08..],
        )),
        mapped_length,
        stride,
        width,
        height,
        hot_x,
        hot_y,
        trailing_word: reims_vgpu_core::endian::ld32(&payload[0x28..]),
        read_length,
    })
}

pub(crate) fn load_replacement_cursor_glyph<Semantic: Clone, Transport>(
    runtime: &crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    packet: &ReplacementDeferredCursorGlyph,
    mut read: impl FnMut(
        reims_vgpu_protocol::TaskId,
        reims_vgpu_protocol::GuestVirtualAddress,
        usize,
    ) -> Result<Vec<u8>, Transport>,
) -> Result<ReplacementLoadedCursorGlyph, ReplacementCursorGlyphLoadError<Transport>> {
    let command = packet.command;
    if !runtime.tasks().is_active(command.task.get()) {
        return Err(ReplacementCursorGlyphLoadError::UnknownTask(command.task));
    }
    let bytes = read(command.task, command.gva, command.read_length)
        .map_err(ReplacementCursorGlyphLoadError::Read)?;
    if bytes.len() != command.read_length {
        return Err(ReplacementCursorGlyphLoadError::ReadLengthMismatch {
            expected: command.read_length,
            actual: bytes.len(),
        });
    }
    let stride = usize::try_from(command.stride)
        .expect("a host-sized complete glyph read has a host-sized stride");
    let mut pixels = Vec::with_capacity(usize::from(command.width) * usize::from(command.height));
    for y in 0..usize::from(command.height) {
        for x in 0..usize::from(command.width) {
            let offset = y * stride + x * crate::model::CURSOR_GLYPH_BPP as usize;
            let b = u32::from(bytes[offset]);
            let g = u32::from(bytes[offset + 1]);
            let r = u32::from(bytes[offset + 2]);
            let a = u32::from(bytes[offset + 3]);
            pixels.push((a << 24) | (r << 16) | (g << 8) | b);
        }
    }
    Ok(ReplacementLoadedCursorGlyph {
        envelope: packet.envelope.clone(),
        glyph: ReplacementCursorGlyph {
            display_index: command.display_index,
            width: command.width,
            height: command.height,
            hot_x: command.hot_x,
            hot_y: command.hot_y,
            pixels: pixels.into_boxed_slice(),
        },
    })
}

pub(crate) fn admit_loaded_replacement_cursor_glyph<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    loaded: ReplacementLoadedCursorGlyph,
) -> Result<
    crate::runtime::replacement_session::ReplacementAdmittedControl<Semantic>,
    Box<ReplacementCursorGlyphAdmissionFailure>,
> {
    match runtime.admit_control(
        loaded.envelope.channel,
        loaded.envelope.prerequisites.clone(),
        Some(loaded.envelope.completion_stamp),
        crate::runtime::replacement_session::ReplacementControlCommand::CursorGlyph(
            loaded.glyph.clone(),
        ),
    ) {
        Ok(admitted) => Ok(admitted),
        Err(reason) => Err(Box::new(ReplacementCursorGlyphAdmissionFailure {
            reason,
            loaded,
        })),
    }
}

pub(crate) fn decode_replacement_compute_info(
    payload: &[u8],
) -> Result<DecodedReplacementComputeInfo, ReplacementComputeInfoDecodeError> {
    const LEN: usize = 4 * size_of::<u32>() + size_of::<u64>();
    if payload.len() < LEN {
        return Err(ReplacementComputeInfoDecodeError::Short {
            plen: payload.len(),
        });
    }
    let count = reims_vgpu_core::endian::ld32(&payload[3 * size_of::<u32>()..]);
    let reply_gva = reims_vgpu_core::endian::ld64(&payload[4 * size_of::<u32>()..]);
    if count == 0 || reply_gva == 0 {
        return Err(ReplacementComputeInfoDecodeError::EmptyReply);
    }
    Ok(DecodedReplacementComputeInfo {
        task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(payload)),
        pipeline_ref: reims_vgpu_core::endian::ld32(&payload[size_of::<u32>()..]),
        key_table_len: reims_vgpu_core::endian::ld32(&payload[2 * size_of::<u32>()..]),
        count,
        reply_gva: reims_vgpu_protocol::GuestVirtualAddress::new(reply_gva),
    })
}

pub(crate) fn decode_replacement_discard_resources(
    payload: &[u8],
) -> Result<DecodedReplacementDiscardResources, crate::runtime::decode::fifo::ResourceListDecodeError>
{
    let decoded = crate::runtime::decode::fifo::decode_synchronize_resources(payload)?;
    Ok(DecodedReplacementDiscardResources {
        task: reims_vgpu_protocol::TaskId::new(decoded.task_id),
        objects: decoded
            .object_ids
            .into_iter()
            .map(reims_vgpu_protocol::ObjectTableRef::new)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

pub(crate) fn decode_replacement_synchronize_resources(
    family: ReplacementChildPacketFamily,
    payload: &[u8],
) -> Result<
    DecodedReplacementSynchronizeResources,
    crate::runtime::decode::fifo::ResourceListDecodeError,
> {
    let decoded = crate::runtime::decode::fifo::decode_synchronize_resources(payload)?;
    Ok(DecodedReplacementSynchronizeResources {
        task: reims_vgpu_protocol::TaskId::new(decoded.task_id),
        objects: decoded
            .object_ids
            .into_iter()
            .map(reims_vgpu_protocol::ObjectTableRef::new)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        discard_transfer_backing: family
            == ReplacementChildPacketFamily::SynchronizeAndDiscardResources,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementSynchronizePreAdmissionError {
    Resolution(crate::runtime::replacement_session::ReplacementSynchronizeResolutionError),
    Accesses(crate::runtime::replacement_session::ReplacementExecAccessCompilationFailure),
    Admission(reims_vgpu_core::TransactionRuntimeError),
}

pub(crate) enum ReplacementDeferredSynchronizeDispatchFailure<Semantic> {
    PreAdmission {
        reason: ReplacementSynchronizePreAdmissionError,
        deferred: ReplacementDeferredSynchronizeResources,
    },
    Admitted(crate::runtime::replacement_session::ReplacementSynchronizeDispatchFailure<Semantic>),
}

impl<Semantic> ReplacementDeferredSynchronizeDispatchFailure<Semantic> {
    /// What refused this deferred synchronize, as a diagnostic string.
    pub(crate) fn diagnostic(&self) -> String {
        match self {
            Self::PreAdmission { reason, .. } => format!("pre_admission={reason:?}"),
            Self::Admitted(failure) => failure.diagnostic(),
        }
    }

    /// The reservations a terminal refusal must give up, or this failure back
    /// if it is not one.
    ///
    /// A pre-admission refusal names a task or resource the guest has not
    /// declared yet and holds no runtime place, so it is always re-offered.
    /// Everything after admission answers through
    /// [`crate::runtime::replacement_session::ReplacementSynchronizeDispatchFailure::into_terminal_reservations`].
    pub(crate) fn into_terminal_reservations(
        self,
    ) -> Result<
        crate::runtime::replacement_session::ReplacementRefusedExecReservations<Semantic>,
        Box<Self>,
    > {
        match self {
            Self::Admitted(failure) => failure
                .into_terminal_reservations()
                .map_err(|failure| Box::new(Self::Admitted(failure))),
            failure => Err(Box::new(failure)),
        }
    }
}

pub(crate) fn dispatch_deferred_replacement_synchronize<Semantic>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    deferred: ReplacementDeferredSynchronizeResources,
) -> Result<
    crate::runtime::replacement_session::PendingReplacementIngressExec<Semantic>,
    Box<ReplacementDeferredSynchronizeDispatchFailure<Semantic>>,
>
where
    Semantic: Clone + PartialEq + Send + 'static,
{
    let dispatched = runtime.dispatch_synchronize_resources(
        deferred.envelope.channel,
        deferred.envelope.prerequisites.clone(),
        deferred.envelope.completion_stamp,
        deferred.command.task,
        &deferred.command.objects,
    );
    match dispatched {
        Ok(pending) => Ok(pending),
        Err(
            crate::runtime::replacement_session::ReplacementSynchronizeDispatchFailure::Resolution(
                reason,
            ),
        ) => Err(Box::new(
            ReplacementDeferredSynchronizeDispatchFailure::PreAdmission {
                reason: ReplacementSynchronizePreAdmissionError::Resolution(reason),
                deferred,
            },
        )),
        Err(
            crate::runtime::replacement_session::ReplacementSynchronizeDispatchFailure::Accesses(
                reason,
            ),
        ) => Err(Box::new(
            ReplacementDeferredSynchronizeDispatchFailure::PreAdmission {
                reason: ReplacementSynchronizePreAdmissionError::Accesses(reason),
                deferred,
            },
        )),
        Err(
            crate::runtime::replacement_session::ReplacementSynchronizeDispatchFailure::Admission(
                reason,
            ),
        ) => Err(Box::new(
            ReplacementDeferredSynchronizeDispatchFailure::PreAdmission {
                reason: ReplacementSynchronizePreAdmissionError::Admission(reason),
                deferred,
            },
        )),
        Err(failure) => Err(Box::new(
            ReplacementDeferredSynchronizeDispatchFailure::Admitted(failure),
        )),
    }
}

pub(crate) fn prepare_replacement_compute_info_reply(
    limits: reims_vgpu_core::ComputeInfoLimits,
    request: DecodedReplacementComputeInfo,
) -> Result<ReplacementComputeInfoReply, ReplacementComputeInfoReplyError> {
    const MAX_TOTAL_THREADS: u32 = 1;
    const THREAD_EXECUTION_WIDTH: u32 = 3;
    const STATIC_THREADGROUP_MEMORY: u32 = 4;
    const PAIR_LEN: usize = 2 * size_of::<u32>();
    let caps = [
        (MAX_TOTAL_THREADS, limits.max_total_threads_per_threadgroup),
        (THREAD_EXECUTION_WIDTH, limits.thread_execution_width),
        (STATIC_THREADGROUP_MEMORY, 0),
    ]
    .into_iter()
    .filter(|(key, _)| *key < request.key_table_len)
    .collect::<Vec<_>>();
    let requested = usize::try_from(request.count).unwrap_or(usize::MAX);
    let answered = caps.len().min(requested);
    let write_sentinel = answered < requested;
    let byte_len = (answered + usize::from(write_sentinel)) * PAIR_LEN;
    let last = u64::try_from(byte_len.saturating_sub(1)).unwrap_or(u64::MAX);
    request.reply_gva.get().checked_add(last).ok_or(
        ReplacementComputeInfoReplyError::AddressOverflow {
            gva: request.reply_gva,
            byte_len,
        },
    )?;
    let mut bytes = vec![0; byte_len];
    for (index, (key, value)) in caps.iter().copied().take(answered).enumerate() {
        let offset = index * PAIR_LEN;
        reims_vgpu_core::endian::st32(&mut bytes[offset..], key);
        reims_vgpu_core::endian::st32(&mut bytes[offset + size_of::<u32>()..], value);
    }
    Ok(ReplacementComputeInfoReply {
        task: request.task,
        pipeline_ref: request.pipeline_ref,
        gva: request.reply_gva,
        bytes: bytes.into_boxed_slice(),
        answered_keys: caps[..answered]
            .iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        dropped_keys: caps[answered..]
            .iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

pub(crate) fn apply_replacement_cursor_show<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    show: DecodedReplacementCursorShow,
) -> reims_vgpu_core::CursorPosition {
    runtime.set_cursor_visible(show.visible)
}

pub(crate) fn decode_replacement_resource_delete(
    payload: &[u8],
) -> Result<DecodedReplacementResourceDelete, ReplacementResourceDeleteDecodeError> {
    const LEN: usize = 2 * size_of::<u32>();
    if payload.len() < LEN {
        return Err(ReplacementResourceDeleteDecodeError::Short {
            plen: payload.len(),
        });
    }
    Ok(DecodedReplacementResourceDelete {
        task: reims_vgpu_protocol::TaskId::new(reims_vgpu_core::endian::ld32(payload)),
        object: reims_vgpu_protocol::ObjectTableRef::new(reims_vgpu_core::endian::ld32(
            &payload[size_of::<u32>()..],
        )),
    })
}

pub(crate) fn admit_replacement_resource_delete<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    channel: reims_vgpu_protocol::ChannelId,
    prerequisites: Box<[reims_vgpu_core::ResolvedTransactionPrerequisite]>,
    completion_stamp: Option<reims_vgpu_core::CompletionStamp>,
    deletion: DecodedReplacementResourceDelete,
) -> Result<
    crate::runtime::replacement_session::ReplacementAdmittedResourceLifecycle<Semantic>,
    crate::runtime::replacement_session::ReplacementNamedResourceLifecycleAdmissionError,
> {
    runtime.admit_task_resource_delete(
        channel,
        prerequisites,
        completion_stamp,
        deletion.task,
        deletion.object,
    )
}

pub(crate) fn decode_replacement_physical_replacement(
    payload: &[u8],
) -> Result<DecodedReplacementPhysicalReplacement, ReplacementPhysicalReplacementDecodeError> {
    decode_replacement_resource_delete(payload)
}

pub(crate) fn admit_replacement_physical_replacement<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    channel: reims_vgpu_protocol::ChannelId,
    prerequisites: Box<[reims_vgpu_core::ResolvedTransactionPrerequisite]>,
    completion_stamp: Option<reims_vgpu_core::CompletionStamp>,
    replacement: DecodedReplacementPhysicalReplacement,
) -> Result<
    crate::runtime::replacement_session::ReplacementAdmittedResourceLifecycle<Semantic>,
    crate::runtime::replacement_session::ReplacementNamedResourceLifecycleAdmissionError,
> {
    runtime.admit_task_physical_replacement(
        channel,
        prerequisites,
        completion_stamp,
        replacement.task,
        replacement.object,
    )
}

pub(crate) fn classify_replacement_child_opcode(
    opcode: u16,
) -> Result<ReplacementChildOpcode, ReplacementChildOpcodeError> {
    use crate::model::*;
    let (family, status) = match opcode {
        CHILD_OP_DEBUG => (
            ReplacementChildPacketFamily::Debug,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_SETUP_SHARED_STATE => (
            ReplacementChildPacketFamily::SetupSharedState,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_ONLINE_ACK => (
            ReplacementChildPacketFamily::OnlineAck,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_CURSOR_GLYPH => (
            ReplacementChildPacketFamily::CursorGlyph,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_CURSOR_SHOW => (
            ReplacementChildPacketFamily::CursorShow,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DISPLAY_TRANSACTION2 => (
            ReplacementChildPacketFamily::DisplayTransaction2,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DISPLAY_TRANSACTION3 => (
            ReplacementChildPacketFamily::DisplayTransaction3,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DISPLAY_SWAP => (
            ReplacementChildPacketFamily::DisplaySwap,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DISPLAY_SLEEP_STATE => (
            ReplacementChildPacketFamily::DisplaySleepState,
            ReplacementChildCutoverStatus::TypedRefusal,
        ),
        CHILD_OP_DISPLAY_SET_PROPERTIES => (
            ReplacementChildPacketFamily::DisplaySetProperties,
            ReplacementChildCutoverStatus::TypedRefusal,
        ),
        CHILD_OP_NOP => (
            ReplacementChildPacketFamily::Nop,
            ReplacementChildCutoverStatus::ContractNoOp,
        ),
        CHILD_OP_DELETE_TASK => (
            ReplacementChildPacketFamily::DeleteTask,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_UNMAP_MEMORY => (
            ReplacementChildPacketFamily::UnmapMemory,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DELETE_RESOURCE => (
            ReplacementChildPacketFamily::DeleteResource,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DELETE_OBJECT => (
            ReplacementChildPacketFamily::DeleteObject,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_SET_OBJECT_LIST => (
            ReplacementChildPacketFamily::SetObjectList,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_INVALIDATE_RESOURCES => (
            ReplacementChildPacketFamily::InvalidateResources,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_SYNCHRONIZE_RESOURCES => (
            ReplacementChildPacketFamily::SynchronizeResources,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DELETE_IOSURFACE_BACKING2 => (
            ReplacementChildPacketFamily::DeleteSurfaceBacking,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_EXEC_INDIRECT2 => (
            ReplacementChildPacketFamily::ExecIndirect2,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DEFINE_TASK2 => (
            ReplacementChildPacketFamily::DefineTask2,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_MAP_MEMORY2 => (
            ReplacementChildPacketFamily::MapMemory2,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_GET_COMPUTE_INFO => (
            ReplacementChildPacketFamily::GetComputeInfo,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_REPLACE_PHYSICAL => (
            ReplacementChildPacketFamily::ReplacePhysical,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DELAY => (
            ReplacementChildPacketFamily::Delay,
            ReplacementChildCutoverStatus::TypedRefusal,
        ),
        CHILD_OP_SYNCHRONIZE_AND_DISCARD_RESOURCES => (
            ReplacementChildPacketFamily::SynchronizeAndDiscardResources,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_DISCARD_RESOURCES => (
            ReplacementChildPacketFamily::DiscardResources,
            ReplacementChildCutoverStatus::Implemented,
        ),
        CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN => (
            ReplacementChildPacketFamily::HeapTextureSizeAndAlign,
            ReplacementChildCutoverStatus::Implemented,
        ),
        deprecated if is_deprecated_child_opcode(deprecated) => (
            ReplacementChildPacketFamily::Deprecated(deprecated),
            ReplacementChildCutoverStatus::ContractNoOp,
        ),
        unknown if unknown > CHILD_OP_MAX => {
            return Err(ReplacementChildOpcodeError::OutOfRange {
                opcode: unknown,
                max: CHILD_OP_MAX,
            });
        }
        unknown => return Err(ReplacementChildOpcodeError::Unassigned(unknown)),
    };
    Ok(ReplacementChildOpcode { family, status })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_child_command_has_one_cutover_disposition() {
        use crate::model::*;
        let commands = [
            CHILD_OP_DEBUG,
            CHILD_OP_SETUP_SHARED_STATE,
            CHILD_OP_ONLINE_ACK,
            CHILD_OP_CURSOR_GLYPH,
            CHILD_OP_CURSOR_SHOW,
            CHILD_OP_DISPLAY_TRANSACTION2,
            CHILD_OP_DISPLAY_TRANSACTION3,
            CHILD_OP_DISPLAY_SWAP,
            CHILD_OP_DISPLAY_SLEEP_STATE,
            CHILD_OP_DISPLAY_SET_PROPERTIES,
            CHILD_OP_NOP,
            CHILD_OP_DELETE_TASK,
            CHILD_OP_UNMAP_MEMORY,
            CHILD_OP_DELETE_RESOURCE,
            CHILD_OP_DELETE_OBJECT,
            CHILD_OP_SET_OBJECT_LIST,
            CHILD_OP_INVALIDATE_RESOURCES,
            CHILD_OP_SYNCHRONIZE_RESOURCES,
            CHILD_OP_DELETE_IOSURFACE_BACKING2,
            CHILD_OP_EXEC_INDIRECT2,
            CHILD_OP_DEFINE_TASK2,
            CHILD_OP_MAP_MEMORY2,
            CHILD_OP_GET_COMPUTE_INFO,
            CHILD_OP_REPLACE_PHYSICAL,
            CHILD_OP_DELAY,
            CHILD_OP_SYNCHRONIZE_AND_DISCARD_RESOURCES,
            CHILD_OP_DISCARD_RESOURCES,
            CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN,
        ];
        for opcode in commands {
            assert!(
                classify_replacement_child_opcode(opcode).is_ok(),
                "{opcode:#x}"
            );
        }
        for opcode in 0..=CHILD_OP_MAX {
            if is_deprecated_child_opcode(opcode) {
                assert_eq!(
                    classify_replacement_child_opcode(opcode).unwrap().status,
                    ReplacementChildCutoverStatus::ContractNoOp,
                );
            }
        }
        assert_eq!(
            classify_replacement_child_opcode(CHILD_OP_MAX + 1),
            Err(ReplacementChildOpcodeError::OutOfRange {
                opcode: CHILD_OP_MAX + 1,
                max: CHILD_OP_MAX,
            })
        );
    }

    /// Two families sharing a census name make one guest command invisible
    /// behind another's count, which is the reading this counter exists to
    /// prevent. The population comes from the classifier rather than from a
    /// hand-written list, so a family reached by a new opcode is covered
    /// without anyone remembering to add it.
    #[test]
    fn every_child_family_the_classifier_reaches_has_its_own_census_name() {
        use crate::model::CHILD_OP_MAX;
        let mut named = std::collections::BTreeMap::new();
        for opcode in 0..=CHILD_OP_MAX {
            let Ok(disposition) = classify_replacement_child_opcode(opcode) else {
                continue;
            };
            // Every deprecated opcode is one family by design.
            if matches!(
                disposition.family,
                ReplacementChildPacketFamily::Deprecated(_)
            ) {
                assert_eq!(disposition.family.census_route(), "child_deprecated");
                continue;
            }
            if let Some(other) = named.insert(disposition.family.census_route(), disposition.family)
            {
                assert_eq!(
                    other,
                    disposition.family,
                    "{:?} and {other:?} both census as {}",
                    disposition.family,
                    disposition.family.census_route()
                );
            }
        }
        assert_eq!(
            named.get("child_display_swap"),
            Some(&ReplacementChildPacketFamily::DisplaySwap),
            "the display families are the ones a boot is read for"
        );
        assert_eq!(
            named.get("child_display_transaction3"),
            Some(&ReplacementChildPacketFamily::DisplayTransaction3)
        );
    }

    /// A status that costs the guest its packet must be countable on its own;
    /// an implemented one must not add a second count, or the census stops
    /// adding up.
    #[test]
    fn only_a_child_status_that_loses_the_packet_carries_its_own_census_name() {
        assert_eq!(
            ReplacementChildCutoverStatus::Implemented.census_route(),
            None
        );
        for status in [
            ReplacementChildCutoverStatus::ContractNoOp,
            ReplacementChildCutoverStatus::TypedRefusal,
            ReplacementChildCutoverStatus::Blocked,
        ] {
            assert!(status.census_route().is_some(), "{status:?}");
        }
    }

    #[test]
    fn cursor_show_retains_display_word_and_boolean_visibility() {
        assert_eq!(
            decode_replacement_cursor_show(&[7, 0, 0, 0, 2, 0, 0, 0]),
            Ok(DecodedReplacementCursorShow {
                display_index: 7,
                visible: true,
            })
        );
        assert_eq!(
            decode_replacement_cursor_show(&[0; 7]),
            Err(ReplacementCursorShowDecodeError::Short { plen: 7 })
        );
    }

    #[test]
    fn display_present_shapes_keep_distinct_slots_and_every_trailing_byte() {
        let mut transaction2 = [0u8; 14];
        reims_vgpu_core::endian::st32(&mut transaction2, 2);
        reims_vgpu_core::endian::st32(&mut transaction2[4..], 17);
        reims_vgpu_core::endian::st32(&mut transaction2[8..], 9);
        transaction2[12..].copy_from_slice(&[0xaa, 0xbb]);
        assert_eq!(
            decode_replacement_display_present(
                ReplacementChildPacketFamily::DisplayTransaction2,
                &transaction2,
            ),
            Ok(DecodedReplacementDisplayPresent::Transaction2 {
                pipe: 2,
                surface: 17,
                task: reims_vgpu_protocol::TaskId::new(9),
                trailing: Box::new([0xaa, 0xbb]),
            })
        );

        let mut transaction3 = [0u8; 37];
        reims_vgpu_core::endian::st32(&mut transaction3, 3);
        reims_vgpu_core::endian::st32(&mut transaction3[4..], 10);
        reims_vgpu_core::endian::st32(&mut transaction3[8..], 18);
        transaction3[12..36].copy_from_slice(&[0x5a; 24]);
        transaction3[36] = 0xcc;
        assert_eq!(
            decode_replacement_display_present(
                ReplacementChildPacketFamily::DisplayTransaction3,
                &transaction3,
            ),
            Ok(DecodedReplacementDisplayPresent::Transaction3 {
                pipe: 3,
                task: reims_vgpu_protocol::TaskId::new(10),
                surface: 18,
                gamma: [0x5a; 24],
                trailing: Box::new([0xcc]),
            })
        );

        let mut swap = [0u8; 40];
        reims_vgpu_core::endian::st32(&mut swap, 4);
        reims_vgpu_core::endian::st32(&mut swap[4..], 0x1234);
        reims_vgpu_core::endian::st32(&mut swap[8..], 19);
        swap[12..].copy_from_slice(&[0xdd; 28]);
        assert_eq!(
            decode_replacement_display_present(ReplacementChildPacketFamily::DisplaySwap, &swap,),
            Ok(DecodedReplacementDisplayPresent::Swap {
                display: 4,
                unidentified_word: 0x1234,
                mapping: reims_vgpu_protocol::MapperResolvedSurfaceId::new(19),
                trailing: vec![0xdd; 28].into_boxed_slice(),
            })
        );
        assert_eq!(
            decode_replacement_display_present(
                ReplacementChildPacketFamily::DisplayTransaction3,
                &transaction3[..35],
            ),
            Err(ReplacementDisplayPresentDecodeError::Short {
                family: ReplacementChildPacketFamily::DisplayTransaction3,
                expected: 36,
                actual: 35,
            })
        );
    }

    #[test]
    fn cursor_glyph_refuses_invalid_geometry_before_memory_transport() {
        fn glyph_payload() -> Vec<u8> {
            let mut payload = vec![0u8; crate::model::CURSOR_GLYPH_PAYLOAD_LEN];
            reims_vgpu_core::endian::st64(&mut payload[0x10..], 16);
            reims_vgpu_core::endian::st64(&mut payload[0x18..], 8);
            reims_vgpu_core::endian::st16(&mut payload[0x20..], 2);
            reims_vgpu_core::endian::st16(&mut payload[0x22..], 2);
            reims_vgpu_core::endian::st16(&mut payload[0x24..], 1);
            payload
        }

        assert_eq!(
            decode_replacement_cursor_glyph(&glyph_payload()[..0x2b]),
            Err(ReplacementCursorGlyphDecodeError::Short { plen: 0x2b })
        );

        for mutate in [
            |payload: &mut [u8]| reims_vgpu_core::endian::st16(&mut payload[0x20..], 0),
            |payload: &mut [u8]| reims_vgpu_core::endian::st16(&mut payload[0x20..], 513),
            |payload: &mut [u8]| reims_vgpu_core::endian::st16(&mut payload[0x24..], 2),
            |payload: &mut [u8]| reims_vgpu_core::endian::st64(&mut payload[0x18..], 7),
        ] {
            let mut payload = glyph_payload();
            mutate(&mut payload);
            assert_eq!(
                decode_replacement_cursor_glyph(&payload),
                Err(ReplacementCursorGlyphDecodeError::InvalidGeometry)
            );
        }

        let mut short_mapping = glyph_payload();
        reims_vgpu_core::endian::st64(&mut short_mapping[0x10..], 15);
        assert_eq!(
            decode_replacement_cursor_glyph(&short_mapping),
            Err(ReplacementCursorGlyphDecodeError::MappedLengthShort {
                mapped: 15,
                required: 16,
            })
        );

        let mut overflow = glyph_payload();
        reims_vgpu_core::endian::st64(&mut overflow[0x10..], u64::MAX);
        reims_vgpu_core::endian::st64(&mut overflow[0x18..], u64::MAX);
        reims_vgpu_core::endian::st16(&mut overflow[0x20..], 1);
        reims_vgpu_core::endian::st16(&mut overflow[0x24..], 0);
        assert_eq!(
            decode_replacement_cursor_glyph(&overflow),
            Err(ReplacementCursorGlyphDecodeError::HostLengthOverflow { length: u64::MAX })
        );
    }

    #[test]
    fn compute_info_decode_and_reply_are_exact_and_bounded() {
        let mut payload = [0u8; 24];
        reims_vgpu_core::endian::st32(&mut payload[0..], 7);
        reims_vgpu_core::endian::st32(&mut payload[4..], 19);
        reims_vgpu_core::endian::st32(&mut payload[8..], 5);
        reims_vgpu_core::endian::st32(&mut payload[12..], 4);
        reims_vgpu_core::endian::st64(&mut payload[16..], 0x4000);
        let request = decode_replacement_compute_info(&payload).unwrap();
        assert_eq!(
            request,
            DecodedReplacementComputeInfo {
                task: reims_vgpu_protocol::TaskId::new(7),
                pipeline_ref: 19,
                key_table_len: 5,
                count: 4,
                reply_gva: reims_vgpu_protocol::GuestVirtualAddress::new(0x4000),
            }
        );
        let reply = prepare_replacement_compute_info_reply(
            reims_vgpu_core::ComputeInfoLimits {
                max_total_threads_per_threadgroup: 768,
                thread_execution_width: 64,
            },
            request,
        )
        .unwrap();
        assert_eq!(reply.answered_keys.as_ref(), [1, 3, 4]);
        assert!(reply.dropped_keys.is_empty());
        assert_eq!(reply.bytes.len(), 32);
        assert_eq!(reims_vgpu_core::endian::ld32(&reply.bytes[0..]), 1);
        assert_eq!(reims_vgpu_core::endian::ld32(&reply.bytes[4..]), 768);
        assert_eq!(reims_vgpu_core::endian::ld32(&reply.bytes[8..]), 3);
        assert_eq!(reims_vgpu_core::endian::ld32(&reply.bytes[12..]), 64);
        assert_eq!(reims_vgpu_core::endian::ld32(&reply.bytes[16..]), 4);
        assert_eq!(&reply.bytes[24..], &[0; 8]);

        assert_eq!(
            decode_replacement_compute_info(&payload[..23]),
            Err(ReplacementComputeInfoDecodeError::Short { plen: 23 })
        );
        reims_vgpu_core::endian::st32(&mut payload[12..], 0);
        assert_eq!(
            decode_replacement_compute_info(&payload),
            Err(ReplacementComputeInfoDecodeError::EmptyReply)
        );
    }

    #[test]
    fn resource_delete_retains_the_two_exact_namespaces() {
        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        payload[4..8].copy_from_slice(&17u32.to_le_bytes());
        assert_eq!(
            decode_replacement_resource_delete(&payload),
            Ok(DecodedReplacementResourceDelete {
                task: reims_vgpu_protocol::TaskId::new(u32::MAX),
                object: reims_vgpu_protocol::ObjectTableRef::new(17),
            })
        );
        assert_eq!(
            decode_replacement_resource_delete(&payload[..7]),
            Err(ReplacementResourceDeleteDecodeError::Short { plen: 7 })
        );
    }

    #[test]
    fn discard_resources_uses_the_shared_counted_object_list_contract() {
        let mut payload = [0u8; 16];
        reims_vgpu_core::endian::st32(&mut payload[0..], 9);
        reims_vgpu_core::endian::st32(&mut payload[4..], 2);
        reims_vgpu_core::endian::st32(&mut payload[8..], 17);
        reims_vgpu_core::endian::st32(&mut payload[12..], u32::MAX);
        assert_eq!(
            decode_replacement_discard_resources(&payload),
            Ok(DecodedReplacementDiscardResources {
                task: reims_vgpu_protocol::TaskId::new(9),
                objects: vec![
                    reims_vgpu_protocol::ObjectTableRef::new(17),
                    reims_vgpu_protocol::ObjectTableRef::new(u32::MAX),
                ]
                .into_boxed_slice(),
            })
        );
        assert!(matches!(
            decode_replacement_discard_resources(&payload[..15]),
            Err(crate::runtime::decode::fifo::ResourceListDecodeError::Truncated { .. })
        ));
    }

    #[test]
    fn synchronize_families_share_one_counted_list_and_keep_discard_intent() {
        let mut payload = [0u8; 16];
        reims_vgpu_core::endian::st32(&mut payload, 9);
        reims_vgpu_core::endian::st32(&mut payload[4..], 2);
        reims_vgpu_core::endian::st32(&mut payload[8..], 17);
        reims_vgpu_core::endian::st32(&mut payload[12..], 23);
        for (family, discard_transfer_backing) in [
            (ReplacementChildPacketFamily::SynchronizeResources, false),
            (
                ReplacementChildPacketFamily::SynchronizeAndDiscardResources,
                true,
            ),
        ] {
            assert_eq!(
                decode_replacement_synchronize_resources(family, &payload),
                Ok(DecodedReplacementSynchronizeResources {
                    task: reims_vgpu_protocol::TaskId::new(9),
                    objects: Box::new([
                        reims_vgpu_protocol::ObjectTableRef::new(17),
                        reims_vgpu_protocol::ObjectTableRef::new(23),
                    ]),
                    discard_transfer_backing,
                })
            );
        }
    }

    #[test]
    fn map_and_unmap_share_the_exact_task_address_change_contract() {
        let mut payload = [0u8; 22];
        reims_vgpu_core::endian::st32(&mut payload, 7);
        reims_vgpu_core::endian::st64(&mut payload[4..], 0x1_0000_4000);
        reims_vgpu_core::endian::st64(&mut payload[12..], 0x3000);
        payload[20..].copy_from_slice(&[0xaa, 0xbb]);

        for family in [
            ReplacementChildPacketFamily::MapMemory2,
            ReplacementChildPacketFamily::UnmapMemory,
        ] {
            assert_eq!(
                decode_replacement_mapping_change(family, &payload),
                Ok(DecodedReplacementMappingChange {
                    family,
                    task: reims_vgpu_protocol::TaskId::new(7),
                    address: reims_vgpu_protocol::GuestVirtualAddress::new(0x1_0000_4000),
                    length: reims_vgpu_protocol::ByteLength::new(0x3000),
                    trailing: Box::new([0xaa, 0xbb]),
                })
            );
        }
        assert_eq!(
            decode_replacement_mapping_change(
                ReplacementChildPacketFamily::MapMemory2,
                &payload[..19],
            ),
            Err(ReplacementMappingChangeDecodeError::Short { plen: 19 })
        );
    }

    #[test]
    fn surface_backing_delete_keeps_its_object_then_task_field_order() {
        let mut payload = [0u8; 8];
        reims_vgpu_core::endian::st32(&mut payload, 17);
        reims_vgpu_core::endian::st32(&mut payload[4..], u32::MAX);
        assert_eq!(
            decode_replacement_surface_backing_delete(&payload),
            Ok(DecodedReplacementSurfaceBackingDelete {
                object: reims_vgpu_protocol::ObjectTableRef::new(17),
                task: reims_vgpu_protocol::TaskId::new(u32::MAX),
            })
        );
        assert_eq!(
            decode_replacement_surface_backing_delete(&payload[..7]),
            Err(ReplacementResourceDeleteDecodeError::Short { plen: 7 })
        );
    }

    #[test]
    fn exhaustive_child_decode_routes_supported_work_and_names_every_blocker() {
        use crate::model::*;

        assert_eq!(
            decode_replacement_child_packet(CHILD_OP_DEBUG, &[0, 1, 0xff]),
            Ok(ReplacementChildPacketRoute::Control(
                crate::runtime::replacement_session::ReplacementControlCommand::Debug(
                    vec![0, 1, 0xff].into_boxed_slice(),
                )
            ))
        );

        let mut setup = [0u8; CHILD_SHARED_STATE_LEN];
        setup[0..4].copy_from_slice(&3u32.to_le_bytes());
        setup[4..8].copy_from_slice(&17u32.to_le_bytes());
        assert!(matches!(
            decode_replacement_child_packet(CHILD_OP_SETUP_SHARED_STATE, &setup),
            Ok(ReplacementChildPacketRoute::Control(
                crate::runtime::replacement_session::ReplacementControlCommand::SetupSharedState(
                    crate::runtime::replacement_fifo_control::DecodedReplacementSharedState {
                        index: 3,
                        pfn: 17,
                    }
                )
            ))
        ));
        assert!(matches!(
            decode_replacement_child_packet(CHILD_OP_NOP, &[]),
            Ok(ReplacementChildPacketRoute::Control(
                crate::runtime::replacement_session::ReplacementControlCommand::ContractNoOp(
                    CHILD_OP_NOP
                )
            ))
        ));
        assert_eq!(
            decode_replacement_child_packet(CHILD_OP_DELAY, &[0xff; 3]),
            Ok(ReplacementChildPacketRoute::Refused(
                ReplacementChildPacketFamily::Delay,
            ))
        );

        let mut exec = [0u8; 28];
        exec[0..4].copy_from_slice(&7u32.to_le_bytes());
        exec[8..12].copy_from_slice(&1u32.to_le_bytes());
        exec[12..20].copy_from_slice(&0x1000u64.to_le_bytes());
        exec[20..28].copy_from_slice(&4u64.to_le_bytes());
        assert!(matches!(
            decode_replacement_child_packet(CHILD_OP_EXEC_INDIRECT2, &exec),
            Ok(ReplacementChildPacketRoute::Exec(
                crate::runtime::decode::fifo::ExecIndirect2Command { task_id: 7, .. }
            ))
        ));

        for opcode in 0..=CHILD_OP_MAX {
            let disposition = classify_replacement_child_opcode(opcode);
            if let Ok(disposition) = disposition {
                if disposition.status == ReplacementChildCutoverStatus::Blocked {
                    assert_eq!(
                        decode_replacement_child_packet(opcode, &[]),
                        Ok(ReplacementChildPacketRoute::Blocked(disposition.family)),
                    );
                } else if disposition.status == ReplacementChildCutoverStatus::TypedRefusal {
                    assert_eq!(
                        decode_replacement_child_packet(opcode, &[]),
                        Ok(ReplacementChildPacketRoute::Refused(disposition.family)),
                    );
                }
            }
        }
        assert!((0..=CHILD_OP_MAX).all(|opcode| {
            classify_replacement_child_opcode(opcode).is_err()
                || classify_replacement_child_opcode(opcode).unwrap().status
                    != ReplacementChildCutoverStatus::Blocked
        }));
    }
}
