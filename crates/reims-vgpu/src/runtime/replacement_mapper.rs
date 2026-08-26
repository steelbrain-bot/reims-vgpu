//! Publishing-vCPU mapper capture and guest-memory adapter.

use crate::{
    model::MapperCapture,
    runtime::host::{GuestCpuAccess, HostMemory, HostPageViews, MemError},
};
use reims_vgpu_paging::mapper::{
    guest_kernel_va, read_mapper_identity, validate_mapper_internal, PagesMemory,
};
use reims_vgpu_protocol::{
    decode_mapper_request_entry, mapper_request_published_entry_offset, MapperRequestKind,
    MAPPER_REQUEST_ENTRY_LEN,
};

pub(crate) const MAPPER_CAPTURE_REG_MAPPER_DEVICE: u32 = 19;
pub(crate) const MAPPER_CAPTURE_REG_REQUEST_TYPE: u32 = 21;
pub(crate) const MAPPER_CAPTURE_REG_MAPPING_INTERNAL: u32 = 22;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureDecline {
    MapperRegister(MemError),
    RequestTypeRegister(MemError),
    InternalRegister(MemError),
    RequestTypeMismatch,
    InternalZero,
    InternalAddress,
    MapperAddress,
    InternalContract(&'static str),
}

impl crate::observe::Decline for CaptureDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MapperRegister(_) => "replacement_mapper_capture_mapper_register",
            Self::RequestTypeRegister(_) => "replacement_mapper_capture_request_type_register",
            Self::InternalRegister(_) => "replacement_mapper_capture_internal_register",
            Self::RequestTypeMismatch => "replacement_mapper_capture_request_type_mismatch",
            Self::InternalZero => "replacement_mapper_capture_internal_zero",
            Self::InternalAddress => "replacement_mapper_capture_internal_address",
            Self::MapperAddress => "replacement_mapper_capture_mapper_address",
            Self::InternalContract(_) => "replacement_mapper_capture_internal_contract",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::MapperRegister(reason)
            | Self::RequestTypeRegister(reason)
            | Self::InternalRegister(reason) => {
                vec![(
                    "host_reason",
                    crate::observe::Decline::slug(reason).to_string(),
                )]
            }
            Self::InternalContract(reason) => vec![("contract_reason", (*reason).to_string())],
            _ => Vec::new(),
        }
    }
}

pub(crate) struct MapperMemory<'a, H: HostMemory + GuestCpuAccess + HostPageViews> {
    host: &'a H,
    last_error: std::cell::Cell<Option<MemError>>,
}

impl<'a, H: HostMemory + GuestCpuAccess + HostPageViews> MapperMemory<'a, H> {
    pub(crate) fn new(host: &'a H) -> Self {
        Self {
            host,
            last_error: std::cell::Cell::new(None),
        }
    }
}

impl<H: HostMemory + GuestCpuAccess + HostPageViews> PagesMemory for MapperMemory<'_, H> {
    fn read(&self, address: u64, destination: &mut [u8]) -> bool {
        let result = if guest_kernel_va(address) {
            self.host.read_kva(address, destination)
        } else {
            self.host.read_gpa(address, destination)
        };
        match result {
            Ok(()) => true,
            Err(reason) => {
                self.last_error.set(Some(reason));
                false
            }
        }
    }

    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }

    fn is_ram_gpa(&self, address: u64) -> bool {
        self.host.is_ram_gpa(address)
    }
}

pub(crate) fn capture_at_producer<H: HostMemory + GuestCpuAccess + HostPageViews>(
    host: &H,
    ring_base: u64,
    producer: u32,
) -> Option<MapperCapture> {
    if producer == 0 || ring_base == 0 {
        return None;
    }
    let entry_offset = mapper_request_published_entry_offset(producer)?;
    let mut bytes = [0; MAPPER_REQUEST_ENTRY_LEN];
    host.read_gpa(ring_base + entry_offset, &mut bytes).ok()?;
    let request = decode_mapper_request_entry(&bytes).ok()?;
    if !request.kind.is_known() || !crate::model::is_surface_mapping_id(request.mapping_id) {
        return None;
    }
    let mapper = host
        .read_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE)
        .map_err(CaptureDecline::MapperRegister);
    let request_type = host
        .read_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE)
        .map(|value| value as u32)
        .map_err(CaptureDecline::RequestTypeRegister);
    let internal = host
        .read_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL)
        .map_err(CaptureDecline::InternalRegister);
    let (mapper, request_type, internal) = match (mapper, request_type, internal) {
        (Ok(mapper), Ok(request_type), Ok(internal)) => (mapper, request_type, internal),
        (Err(reason), _, _) | (_, Err(reason), _) | (_, _, Err(reason)) => {
            emit_capture_decline(request.mapping_id, producer, reason);
            return None;
        }
    };
    let decline = if MapperRequestKind::from_raw(request_type) != request.kind {
        Some(CaptureDecline::RequestTypeMismatch)
    } else if internal == 0 {
        Some(CaptureDecline::InternalZero)
    } else if !guest_kernel_va(internal) {
        Some(CaptureDecline::InternalAddress)
    } else if mapper != 0 && !guest_kernel_va(mapper) {
        Some(CaptureDecline::MapperAddress)
    } else {
        None
    };
    if let Some(reason) = decline {
        emit_capture_decline(request.mapping_id, producer, reason);
        return None;
    }
    let memory = MapperMemory::new(host);
    let fields = match read_mapper_identity(&memory, internal, mapper != 0, mapper) {
        Ok(fields) => fields,
        Err(status) => {
            emit_capture_decline(
                request.mapping_id,
                producer,
                CaptureDecline::InternalContract(mapper_status_reason(&status)),
            );
            return None;
        }
    };
    let status = validate_mapper_internal(&memory, request.mapping_id, &fields);
    if status != reims_vgpu_paging::mapper::Status::Ok {
        emit_capture_decline(
            request.mapping_id,
            producer,
            CaptureDecline::InternalContract(mapper_status_reason(&status)),
        );
        return None;
    }
    Some(MapperCapture {
        producer,
        mapper_device_kva: mapper,
        request_kind: request.kind,
        mapping_internal: internal,
    })
}

fn emit_capture_decline(mapping: u32, producer: u32, reason: CaptureDecline) {
    crate::observe::Emit::decline("replacement_mapper_capture", &reason)
        .field("mapping", mapping)
        .field("producer", producer)
        .fail_once((u64::from(mapping) << 32) | u64::from(producer));
}

fn mapper_status_reason(status: &reims_vgpu_paging::mapper::Status) -> &'static str {
    use reims_vgpu_paging::mapper::Status;
    match status {
        Status::Ok => "ok",
        Status::ErrShortDescriptor(reason)
        | Status::ErrNotKernelVa(reason)
        | Status::ErrInternalRead(reason)
        | Status::ErrInternalOwner(reason)
        | Status::ErrInternalMappingId(reason)
        | Status::ErrInternalSize(reason)
        | Status::ErrInternalFields(reason)
        | Status::ErrPageCount(reason)
        | Status::ErrPageTableRead(reason)
        | Status::ErrPageEntry(reason)
        | Status::ErrNoPageTable(reason) => reason,
    }
}
