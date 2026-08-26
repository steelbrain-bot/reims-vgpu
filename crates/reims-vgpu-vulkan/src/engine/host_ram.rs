//! Import a bounded host mapping as `VkDeviceMemory` and bind a `VkBuffer` over
//! all of it.
//!
//! This is the one place guest memory becomes something the engine can bind.
//! Which *bytes* a draw reaches is decided before it gets here and is carried by
//! a [`GuestRef`], whose bound cannot be skipped — see
//! `reims-vgpu::runtime::guest_ram` for why that is a type and not a review rule,
//! and [`crate::host_pointer`] for the capability that gates the
//! whole rail.
//!
//! # One import per allocation identity
//!
//! RAMBlocks are imported once for the device's life. A scattered task mapping
//! may also arrive as a stable packed host alias, created once for that mapping
//! and shared by its resources; their draw offsets are still only bounds
//! checks. A resource-owned alias is the fallback when no mapping import exists.
//!
//! [`HostRamImports`] keys both forms by
//! [`reims_vgpu_memory::ImportId`], so one allocation identity is never
//! imported twice. The driver is allowed to refuse a packed alias; that answer
//! is remembered and its caller gathers instead. The census separates RAMBlock
//! entries from aliases so resource-shaped growth is visible.
//!
//! # What the import does not promise
//!
//! Freeing the memory ends the GPU's access, but nothing in the extension's
//! specification says the pages were pinned while it lived. amdgpu and the
//! NVIDIA driver call `get_user_pages` at import time in practice; that is an
//! observation about two drivers rather than a contract. The honest statement is
//! in `reims-vgpu::runtime::guest_ram`'s module doc and is not repeated as a
//! guarantee here.

use ash::vk;

use crate::host_pointer::ImportTypeRefusal;
use reims_vgpu_memory::GuestRamImport;
use reims_vgpu_observe::Decline;

/// One host allocation living on the GPU as a bindable buffer, with no copy
/// between it and the guest's own view of those bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImportedHostRam {
    pub import_id: reims_vgpu_memory::ImportId,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl ImportedHostRam {
    /// Release both halves. Freeing the memory is what ends the GPU's access to
    /// guest RAM, so it must run even on a teardown path that is otherwise
    /// giving up.
    ///
    /// # Safety
    ///
    /// No submission may still reference `buffer`, and `device` must be the one
    /// the import was made against.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) -> reims_vgpu_memory::ImportId {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
        self.import_id
    }
}

/// A check that stopped guest RAM from becoming a bindable buffer.
///
/// Every variant is a distinct check with its own slug. An import that fails at
/// `vkAllocateMemory` and one the device declined a memory type for are two
/// different findings — the first is usually the driver refusing the pointer's
/// backing, the second is a memory-type intersection that came out empty — and a
/// shared reason would leave a reader unable to tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRamDecline {
    /// This device cannot import a host pointer at all. Carries the rung so the
    /// log says which check refused; expected on every host without the
    /// extension and on any host where an operator turned the rail off.
    Unsupported {
        rung: crate::host_pointer::HostPointerImport,
    },
    /// The guest ended this parent allocation's lifetime. Old child objects
    /// may finish retiring, but no new view may resurrect its import identity.
    Retired { import_id: u64 },
    /// No memory type could be named for the pointer. Carries which of the two
    /// checks refused — see [`ImportTypeRefusal`], whose doc says why they are
    /// not one finding.
    NoImportableMemoryType {
        host_base: usize,
        refusal: ImportTypeRefusal,
    },
    /// A memory type was named for the pointer and the *buffer* over the same
    /// span excludes it.
    ///
    /// A separate variant rather than a second use of the one above, which is
    /// what it was. The two are asked of different objects — the first of the
    /// host allocation, this one of `vkGetBufferMemoryRequirements` — and they
    /// have different repairs: the first is a request this device chose, and
    /// this one is two driver answers that do not intersect, which no policy
    /// here can widen. Sharing a slug meant a log line could not say which, and
    /// `bugs/bug-06` is a hundred of exactly that line.
    BufferExcludesMemoryType {
        host_base: usize,
        picked: u32,
        buffer_types: u32,
    },
    /// `vkCreateBuffer` over the whole span failed.
    CreateBuffer { result: vk::Result },
    /// The buffer the driver made needs more bytes than the span has. The import
    /// is sized to the RAMBlock exactly and may not be rounded up: the bytes past
    /// the end are this process's own memory.
    TooSmall { required: u64, available: u64 },
    /// `vkAllocateMemory` with the chained import failed. On most drivers this
    /// is the pointer being refused — not fd-backed, not aligned, or not a
    /// mapping the driver can take a reference on.
    AllocateMemory { result: vk::Result },
    /// `vkBindBufferMemory` failed after a successful import.
    BindBuffer { result: vk::Result },
}

impl Decline for HostRamDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "host_ram_import_unsupported",
            Self::Retired { .. } => "host_ram_import_retired",
            Self::NoImportableMemoryType { .. } => "host_ram_import_no_importable_memory_type",
            Self::BufferExcludesMemoryType { .. } => "host_ram_import_buffer_excludes_memory_type",
            Self::CreateBuffer { .. } => "host_ram_import_create_buffer",
            Self::TooSmall { .. } => "host_ram_import_too_small",
            Self::AllocateMemory { .. } => "host_ram_import_allocate_memory",
            Self::BindBuffer { .. } => "host_ram_import_bind_buffer",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::Retired { import_id } => vec![("import_id", import_id.to_string())],
            Self::NoImportableMemoryType { host_base, refusal } => {
                let mut fields = vec![("host_base", format!("{host_base:#x}"))];
                match refusal {
                    ImportTypeRefusal::PointerDeclined { result } => {
                        fields.push(("check", "pointer_declined".to_string()));
                        fields.push(("result", format!("{result:?}")));
                    }
                    ImportTypeRefusal::NoTypeMeetsRequest {
                        pointer_types,
                        refusal,
                    } => {
                        // The selector's own check, not just "no type": a guest
                        // this host has nowhere to put and a host that offers no
                        // importable memory at all are different reports.
                        fields.push(("check", refusal.slug().to_string()));
                        fields.push(("detail", refusal.to_string()));
                        fields.push(("pointer_types", format!("{pointer_types:#x}")));
                    }
                }
                fields
            }
            Self::BufferExcludesMemoryType {
                host_base,
                picked,
                buffer_types,
            } => vec![
                ("host_base", format!("{host_base:#x}")),
                ("picked", picked.to_string()),
                ("buffer_types", format!("{buffer_types:#x}")),
            ],
            Self::CreateBuffer { result }
            | Self::AllocateMemory { result }
            | Self::BindBuffer { result } => vec![("result", format!("{result:?}"))],
            Self::TooSmall {
                required,
                available,
            } => vec![
                ("required", required.to_string()),
                ("available", available.to_string()),
            ],
        }
    }
}

reims_vgpu_observe::decline_display!(HostRamDecline);

/// Import one bounded host allocation.
///
/// # Cost, and why it is timed
///
/// This is the only expensive step on the whole guest-memory rail, and it is
/// paid once per allocation identity. Which of its two halves costs what is not
/// a thing a reader can assume: `vkGetMemoryHostPointerPropertiesEXT` asks the
/// driver about a pointer and `vkAllocateMemory` is where a driver that pins
/// takes its `get_user_pages` over every page of a multi-gigabyte mapping. The
/// two are timed separately and emitted once per import, because the first draw
/// of a boot pays whichever of them is slow, and a display transaction the
/// guest abandons after 1000 ms is what that draw sits inside.
///
/// # Safety
///
/// As [`HostRamImports::bind`].
pub(crate) unsafe fn import_host_allocation(
    ctx: &super::context::DeviceContext,
    import: &GuestRamImport,
) -> Result<ImportedHostRam, HostRamDecline> {
    use crate::host_pointer::GUEST_IMPORT_USAGE;
    use std::time::Instant;

    let Some(loader) = ctx.external_memory_host.as_ref() else {
        return Err(HostRamDecline::Unsupported {
            rung: ctx.caps.host_pointer.rung,
        });
    };
    let handle_type = ctx.caps.host_pointer.handle_type;

    let host_base = import.host_base();
    let size = import.len();

    // Which memory types will accept *this* pointer. Asked before anything is
    // created, because the answer is a property of the mapping rather than of
    // the device — and it goes through the one memory-type selector this
    // backend has, so the ranking is not restated here.
    //
    // `Upload` is the class: guest RAM is host memory the GPU reaches, which is
    // exactly what that preference describes. On a discrete host the selector
    // will land on a host-visible type, and the copy into VRAM is a separate
    // decision made by the caller, not by this import.
    //
    // The RAMBlock's whole length goes with the request, and for this call site
    // that is the load-bearing argument rather than a detail. An imported host
    // pointer's pages do not move — the memory type cannot relocate a mapping
    // this process already holds — so the only thing the pick decides is which
    // heap the driver charges a multi-gigabyte allocation to. `Upload` on a
    // `Unified` classification prefers `DEVICE_LOCAL`, and on a part whose
    // device-local heap is a carve-out smaller than the guest (an APU with 2 GiB
    // against a 16 GiB guest) that preference asks the driver to keep the entire
    // guest resident in a pool with no room for it.
    let req = ctx.caps.memory_request(crate::memory::MemoryClass::Upload);
    let probe_started = Instant::now();
    let picked = unsafe {
        crate::host_pointer::import_memory_type(
            loader,
            &ctx.memory_properties,
            host_base as *const std::ffi::c_void,
            handle_type,
            &req,
            size,
            ctx.caps.max_allocation_size,
        )
    };
    let probe_us = probe_started.elapsed().as_micros() as u64;
    // A refusal here is the whole of the heap and allocation-size admission for
    // this rail. It reaches the caller as a decline, the copying rails take the
    // guest's bytes instead, and no `vkAllocateMemory` the specification forbids
    // is ever issued — which is the difference between the two drivers this was
    // reported on, one of which returns success and then loses the device.
    let pick =
        picked.map_err(|refusal| HostRamDecline::NoImportableMemoryType { host_base, refusal })?;
    let memory_type_index = pick.index;
    let alloc_started = Instant::now();

    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
    let create = vk::BufferCreateInfo::default()
        .size(size)
        .usage(GUEST_IMPORT_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let buffer = unsafe { ctx.device.create_buffer(&create, None) }
        .map_err(|result| HostRamDecline::CreateBuffer { result })?;

    // From here every failure must destroy the buffer before returning, so the
    // work is done in a closure and the cleanup happens once at the end.
    let bound = (|| {
        let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if reqs.size > size {
            // Not rounded up. The bytes past the end of a RAMBlock are this
            // process's own memory, and handing the GPU write access to them is
            // the one stray the bound exists to prevent.
            return Err(HostRamDecline::TooSmall {
                required: reqs.size,
                available: size,
            });
        }
        if reqs.memory_type_bits & (1u32 << memory_type_index) == 0 {
            return Err(HostRamDecline::BufferExcludesMemoryType {
                host_base,
                picked: memory_type_index,
                buffer_types: reqs.memory_type_bits,
            });
        }

        let mut host_import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle_type)
            .host_pointer(host_base as *mut std::ffi::c_void);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut host_import);
        let memory = unsafe { ctx.device.allocate_memory(&allocate, None) }
            .map_err(|result| HostRamDecline::AllocateMemory { result })?;

        match unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            Ok(()) => Ok(ImportedHostRam {
                import_id: import.id(),
                buffer,
                memory,
            }),
            Err(result) => {
                // Freeing the memory is what ends the GPU's access to the
                // pointer, so it happens even on this failure path.
                unsafe { ctx.device.free_memory(memory, None) };
                Err(HostRamDecline::BindBuffer { result })
            }
        }
    })();

    if bound.is_err() {
        unsafe { ctx.device.destroy_buffer(buffer, None) };
    }
    // The heap is on this line and not just the type index, because "which type"
    // is not answerable from the index alone on an unfamiliar device and the
    // heap is what a report of a slow host turns on. There is no `fits=` field
    // any more and there cannot be one: a pick whose heap could not hold the
    // import is a refusal now, so every line here is an import the device was
    // allowed to make.
    reims_vgpu_observe::off(format!(
        "host_ram_import id={} bytes={size} mtype={memory_type_index} heap={} heap_mb={} \
         probe_us={probe_us} alloc_us={} ok={}",
        import.id().get(),
        pick.heap_index,
        pick.heap_bytes >> 20,
        alloc_started.elapsed().as_micros() as u64,
        bound.is_ok(),
    ));
    bound
}
