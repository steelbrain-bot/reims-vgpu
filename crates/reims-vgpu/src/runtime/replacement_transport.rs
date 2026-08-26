//! Transport-only ownership for replacement MMIO, FIFO wakeups, and IRQ state.
//!
//! This owner deliberately contains no task, resource, execution, display, or
//! publication semantics. MMIO producers publish root/child work here; the
//! replacement coordinator consumes that work and changes semantic state in
//! [`crate::runtime::replacement_session::ReplacementRuntimeSession`].

#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTransportStartError {
    UnsupportedPageShift(u32),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplacementTransportWork {
    pub root: bool,
    pub children: u32,
    pub iosfc: bool,
}

impl ReplacementTransportWork {
    pub const fn is_empty(self) -> bool {
        !self.root && self.children == 0 && !self.iosfc
    }
}

pub(crate) struct ReplacementTransportOwner {
    page_shift: u32,
    registers: reims_vgpu_core::DeviceRegisters,
    pending: ReplacementTransportWork,
    root_packet: Option<ReplacementRootPacketOwnership>,
    child_rings: [reims_vgpu_core::ChannelRing; reims_vgpu_core::MAX_CHANNELS],
    child_packets: [Option<ReplacementChildPacketOwnership>; reims_vgpu_core::MAX_CHANNELS],
    mapper_entry: Option<ReplacementMapperEntryLease>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementMapperEntryLease {
    pub ring_base: u64,
    pub producer: u32,
    consumer: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementMapperEntryError {
    EntryAlreadyOwned,
    RingUnavailable,
    ProducerBehindConsumer { producer: u32, consumer: u32 },
    EntryNotOwned,
    OwnershipMismatch,
    PublicationChanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReplacementChildPacketOwnership {
    channel: reims_vgpu_protocol::ChannelId,
    registers_gpa: u64,
    base_pfn: u32,
    head: u32,
    next_head: u32,
}

#[derive(Debug)]
pub(crate) struct ReplacementChildPacketLease {
    pub packet: crate::runtime::fifo_packet::Packet,
    pub channel: reims_vgpu_protocol::ChannelId,
    pub stamp_index: u32,
    ownership: ReplacementChildPacketOwnership,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketReadPhase {
    Head,
    StampIndex,
    BasePfn,
    PageList { entry: u32 },
    Tail,
    Header,
    Snapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketReadError {
    InvalidChannel(reims_vgpu_protocol::ChannelId),
    PacketAlreadyOwned(reims_vgpu_protocol::ChannelId),
    RootPageUnavailable,
    RingAddressOverflow,
    RingUnavailable,
    RingLengthOverflow,
    InvalidPublishedPointers {
        head: u32,
        tail: u32,
        capacity: u32,
    },
    Memory {
        phase: ReplacementChildPacketReadPhase,
        reason: crate::runtime::host::MemError,
    },
    Packet(crate::runtime::fifo_packet::PacketError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementChildPacketCommitError {
    PacketNotOwned,
    OwnershipMismatch,
    HeadChanged { expected: u32, actual: u32 },
    Memory(crate::runtime::host::MemError),
}

#[derive(Debug)]
pub(crate) struct ReplacementChildPacketCommitFailure {
    pub reason: ReplacementChildPacketCommitError,
    pub lease: ReplacementChildPacketLease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReplacementRootPacketOwnership {
    head: u32,
    next_head: u32,
}

#[derive(Debug)]
pub(crate) struct ReplacementRootPacketLease {
    pub packet: crate::runtime::fifo_packet::Packet,
    ownership: ReplacementRootPacketOwnership,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRootPacketReadError {
    PacketAlreadyOwned,
    RingUnavailable,
    RingAddressOverflow,
    InvalidPublishedPointers { head: u32, tail: u32, capacity: u32 },
    Memory(crate::runtime::host::MemError),
    Packet(crate::runtime::fifo_packet::PacketError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementRootPacketCommitError {
    PacketNotOwned,
    OwnershipMismatch,
    HeadChanged { expected: u32, actual: u32 },
}

#[derive(Debug)]
pub(crate) struct ReplacementRootPacketCommitFailure {
    pub reason: ReplacementRootPacketCommitError,
    pub lease: ReplacementRootPacketLease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementGfxMmioError {
    UnsupportedSize { offset: u64, size: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementGfxWriteEffect {
    ScheduleDrain,
    ProtocolNegotiated(u32),
    DisplayInterrupt(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementIosfcWriteEffect {
    CaptureAndDrain { ring_base: u64, producer: u32 },
}

impl ReplacementTransportOwner {
    pub fn new(page_shift: u32) -> Result<Self, ReplacementTransportStartError> {
        if !matches!(
            page_shift,
            reims_vgpu_paging::geometry::PAGE_SHIFT_X86
                | reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E
        ) {
            return Err(ReplacementTransportStartError::UnsupportedPageShift(
                page_shift,
            ));
        }
        Ok(Self {
            page_shift,
            registers: reims_vgpu_core::DeviceRegisters::default(),
            pending: ReplacementTransportWork::default(),
            root_packet: None,
            child_rings: std::array::from_fn(|_| reims_vgpu_core::ChannelRing::default()),
            child_packets: std::array::from_fn(|_| None),
            mapper_entry: None,
        })
    }

    pub const fn page_shift(&self) -> u32 {
        self.page_shift
    }

    pub const fn registers(&self) -> &reims_vgpu_core::DeviceRegisters {
        &self.registers
    }

    pub fn registers_mut(&mut self) -> &mut reims_vgpu_core::DeviceRegisters {
        &mut self.registers
    }

    pub fn request_root(&mut self) {
        self.pending.root = true;
    }

    pub fn request_child(&mut self, channel: reims_vgpu_protocol::ChannelId) -> bool {
        let channel = channel.get();
        if !crate::model::is_child_channel(channel) {
            return false;
        }
        self.pending.children |= 1u32 << channel;
        true
    }

    pub fn request_iosfc(&mut self) {
        self.pending.iosfc = true;
    }

    pub fn take_work(&mut self) -> ReplacementTransportWork {
        std::mem::take(&mut self.pending)
    }

    pub fn stamp_page(&self) -> ReplacementStampPage {
        ReplacementStampPage {
            base_pfn: self.registers.gfx.fifo_base_page,
            page_shift: self.page_shift,
        }
    }

    pub fn gpu_interrupt_status(&self) -> &std::sync::atomic::AtomicU32 {
        &self.registers.gfx.interrupt_status_gpu
    }

    pub fn gfx_read(&mut self, offset: u64, size: u32) -> Result<u64, ReplacementGfxMmioError> {
        use crate::model::*;

        if offset < REG_BASE {
            return Ok(0);
        }
        if size == MMIO_U64 && offset == GFX_REG_EFI_FB_START {
            return Ok(self.registers.gfx.efi_fb_start);
        }
        if size == MMIO_U64 && offset == GFX_REG_FIFO_BASE_PAGE {
            return Ok(u64::from(self.registers.gfx.fifo_base_page));
        }
        if size == MMIO_U64 {
            let low = self.gfx_read(offset, MMIO_U32)?;
            let high = self.gfx_read(offset + u64::from(MMIO_U32), MMIO_U32)?;
            return Ok(low | (high << 32));
        }
        if size != MMIO_U32 {
            return Err(ReplacementGfxMmioError::UnsupportedSize { offset, size });
        }
        let gfx = &mut self.registers.gfx;
        Ok(match offset {
            GFX_REG_CONTROL_FIFO => u64::from(gfx.control_fifo),
            GFX_REG_FIFO_LENGTH => u64::from(gfx.fifo_length),
            GFX_REG_FIFO_WRITTEN => u64::from(gfx.fifo_written),
            GFX_REG_FIFO_READ => {
                u64::from(gfx.fifo_read.load(std::sync::atomic::Ordering::Acquire))
            }
            GFX_REG_FIFO_START => u64::from(gfx.fifo_start),
            GFX_REG_INTR_STATUS_DISP => u64::from(
                gfx.interrupt_status_disp
                    .swap(0, std::sync::atomic::Ordering::AcqRel),
            ),
            GFX_REG_INTR_STATUS_GPU => u64::from(
                gfx.interrupt_status_gpu
                    .swap(0, std::sync::atomic::Ordering::AcqRel),
            ),
            GFX_REG_ROOT_PAGE => u64::from(gfx.root_page),
            GFX_REG_INTR_FAULT => u64::from(
                gfx.interrupt_fault
                    .load(std::sync::atomic::Ordering::Acquire),
            ),
            GFX_REG_FIFO_BASE_PAGE => u64::from(gfx.fifo_base_page),
            GFX_REG_VERSION => u64::from(gfx.version),
            GFX_REG_EFI_DISPLAY => u64::from(gfx.efi_display),
            GFX_REG_EFI_MODE_COUNT => u64::from(EFI_MODE_COUNT),
            GFX_REG_EFI_MODE_SELECT => u64::from(gfx.efi_mode_select),
            GFX_REG_EFI_MODE_SIZE => {
                (u64::from(EFI_BOOT_WIDTH) << EFI_MODE_WIDTH_SHIFT) | u64::from(EFI_BOOT_HEIGHT)
            }
            GFX_REG_EFI_FB_START => gfx.efi_fb_start & u64::from(u32::MAX),
            GFX_REG_EFI_FB_LENGTH => u64::from(gfx.efi_fb_length),
            GFX_REG_EFI_FB_DEPTH => u64::from(gfx.efi_fb_depth),
            GFX_REG_EFI_FB_MODE => u64::from(gfx.efi_fb_mode),
            GFX_REG_EFI_STRIDE_ALIGN => u64::from(EFI_STRIDE_ALIGNMENT),
            GFX_REG_EFI_FB_STRIDE => u64::from(gfx.efi_fb_stride),
            GFX_REG_EFI_DISPLAY_PORTS => u64::from(EFI_DISPLAY_PORT_COUNT),
            GFX_REG_EFI_BUILTIN_CONNECTED => u64::from(EFI_BUILTIN_CONNECTED),
            _ => u64::from(gfx.sparse_get(offset)),
        })
    }

    pub fn gfx_write(
        &mut self,
        offset: u64,
        data: u64,
        size: u32,
    ) -> Result<Box<[ReplacementGfxWriteEffect]>, ReplacementGfxMmioError> {
        use crate::model::*;

        if offset < REG_BASE {
            return Ok(Box::new([]));
        }
        if size == MMIO_U64 && offset == GFX_REG_EFI_FB_START {
            self.registers.gfx.efi_fb_start = data;
            return Ok(Box::new([]));
        }
        if size == MMIO_U64 && offset == GFX_REG_FIFO_BASE_PAGE {
            self.registers.gfx.fifo_base_page = data as u32;
            return Ok(Box::new([]));
        }
        if size == MMIO_U64 {
            let mut effects = self
                .gfx_write(offset, data & u64::from(u32::MAX), MMIO_U32)?
                .into_vec();
            effects.extend(self.gfx_write(offset + u64::from(MMIO_U32), data >> 32, MMIO_U32)?);
            return Ok(effects.into_boxed_slice());
        }
        if size != MMIO_U32 {
            return Err(ReplacementGfxMmioError::UnsupportedSize { offset, size });
        }
        let value = data as u32;
        let mut effects = Vec::new();
        match offset {
            GFX_REG_CONTROL_FIFO => {
                self.registers.gfx.control_fifo = value;
                if value != 0 {
                    self.request_root();
                    effects.push(ReplacementGfxWriteEffect::ScheduleDrain);
                }
            }
            GFX_REG_FIFO_LENGTH => self.registers.gfx.fifo_length = value,
            GFX_REG_FIFO_WRITTEN => {
                self.registers.gfx.fifo_written = value;
                if self.registers.gfx.control_fifo != 0 {
                    self.request_root();
                    effects.push(ReplacementGfxWriteEffect::ScheduleDrain);
                }
            }
            GFX_REG_FIFO_START => self.registers.gfx.fifo_start = value,
            GFX_REG_INTR_STATUS_DISP => {
                self.registers
                    .gfx
                    .interrupt_status_disp
                    .fetch_and(!value, std::sync::atomic::Ordering::AcqRel);
            }
            GFX_REG_INTR_STATUS_GPU => {
                self.registers
                    .gfx
                    .interrupt_status_gpu
                    .fetch_and(!value, std::sync::atomic::Ordering::AcqRel);
            }
            GFX_REG_ROOT_PAGE => self.registers.gfx.root_page = value,
            GFX_REG_CHILD_DOORBELL | GFX_REG_CHILD_REPLAY_DOORBELL => {
                if self.request_child(reims_vgpu_protocol::ChannelId::new(value)) {
                    effects.push(ReplacementGfxWriteEffect::ScheduleDrain);
                }
            }
            GFX_REG_MAIN_KICK => {
                self.request_root();
                effects.push(ReplacementGfxWriteEffect::ScheduleDrain);
            }
            GFX_REG_INTR_FAULT => self
                .registers
                .gfx
                .interrupt_fault
                .store(value, std::sync::atomic::Ordering::Release),
            GFX_REG_FIFO_BASE_PAGE => self.registers.gfx.fifo_base_page = value,
            GFX_REG_VERSION => {
                let negotiated = negotiate_protocol_version(value);
                self.registers.gfx.version = negotiated;
                effects.push(ReplacementGfxWriteEffect::ProtocolNegotiated(negotiated));
            }
            GFX_REG_EFI_DISPLAY => self.registers.gfx.efi_display = value,
            GFX_REG_EFI_MODE_SELECT => self.registers.gfx.efi_mode_select = value,
            GFX_REG_EFI_FB_START => {
                self.registers.gfx.efi_fb_start =
                    (self.registers.gfx.efi_fb_start & !u64::from(u32::MAX)) | u64::from(value);
            }
            GFX_REG_EFI_FB_LENGTH => self.registers.gfx.efi_fb_length = value,
            GFX_REG_EFI_FB_DEPTH => self.registers.gfx.efi_fb_depth = value,
            GFX_REG_EFI_FB_MODE => self.registers.gfx.efi_fb_mode = value,
            GFX_REG_EFI_DISPLAY_IRQ if value < u32::BITS => {
                self.registers
                    .gfx
                    .interrupt_status_disp
                    .fetch_or(1u32 << value, std::sync::atomic::Ordering::AcqRel);
                effects.push(ReplacementGfxWriteEffect::DisplayInterrupt(value));
            }
            GFX_REG_EFI_DISPLAY_IRQ => {}
            GFX_REG_EFI_FB_STRIDE => self.registers.gfx.efi_fb_stride = value,
            _ => self.registers.gfx.sparse_set(offset, value),
        }
        Ok(effects.into_boxed_slice())
    }

    pub fn iosfc_read(&self, offset: u64, size: u32) -> u64 {
        use crate::model::*;

        let mut value = match offset {
            IOSFC_REG_RING_BASE => self.registers.iosfc.ring_base,
            IOSFC_REG_CAPACITY => u64::from(self.registers.iosfc.capacity),
            IOSFC_REG_DESC_TABLE => self.registers.iosfc.desc_table,
            IOSFC_REG_PRODUCER => u64::from(self.registers.iosfc.producer),
            IOSFC_REG_CONSUMER => u64::from(self.registers.iosfc.consumer),
            _ => 0,
        };
        if size < MMIO_U64 && size > 0 {
            let bits = u64::from(size).saturating_mul(8).min(64);
            if bits < 64 {
                value &= (1u64 << bits) - 1;
            }
        }
        value
    }

    pub fn iosfc_write(&mut self, offset: u64, data: u64) -> Option<ReplacementIosfcWriteEffect> {
        use crate::model::*;

        match offset {
            IOSFC_REG_RING_BASE => self.registers.iosfc.ring_base = data,
            IOSFC_REG_CAPACITY => self.registers.iosfc.capacity = data as u32,
            IOSFC_REG_DESC_TABLE => self.registers.iosfc.desc_table = data,
            IOSFC_REG_PRODUCER => {
                let producer = data as u32;
                self.registers.iosfc.producer = producer;
                if self.registers.iosfc.consumer != producer {
                    self.request_iosfc();
                    return Some(ReplacementIosfcWriteEffect::CaptureAndDrain {
                        ring_base: self.registers.iosfc.ring_base,
                        producer,
                    });
                }
            }
            IOSFC_REG_CONSUMER => self.registers.iosfc.consumer = data as u32,
            _ => {}
        }
        None
    }

    /// Snapshot and decode the current root FIFO head without advancing it.
    /// The returned lease is the only value that can commit this packet, so a
    /// decode/admission retry cannot accidentally consume the next packet.
    pub fn read_root_packet(
        &mut self,
        host: &impl crate::runtime::host::HostMemory,
    ) -> Result<Option<ReplacementRootPacketLease>, ReplacementRootPacketReadError> {
        if self.root_packet.is_some() {
            return Err(ReplacementRootPacketReadError::PacketAlreadyOwned);
        }
        let gfx = &self.registers.gfx;
        let capacity = crate::model::main_ring_data_size(gfx.fifo_length, gfx.fifo_start);
        if gfx.control_fifo == 0 || capacity == 0 || gfx.fifo_base_page == 0 {
            return Err(ReplacementRootPacketReadError::RingUnavailable);
        }
        let page_bytes = 1u64
            .checked_shl(self.page_shift)
            .ok_or(ReplacementRootPacketReadError::RingAddressOverflow)?;
        let base = u64::from(gfx.fifo_base_page)
            .checked_mul(page_bytes)
            .and_then(|base| base.checked_add(u64::from(gfx.fifo_start)))
            .ok_or(ReplacementRootPacketReadError::RingAddressOverflow)?;
        let head = gfx.fifo_read.load(std::sync::atomic::Ordering::Acquire);
        let tail = gfx.fifo_written;
        let available = tail.wrapping_sub(head);
        if available > capacity {
            return Err(ReplacementRootPacketReadError::InvalidPublishedPointers {
                head,
                tail,
                capacity,
            });
        }
        if available == 0 {
            return Ok(None);
        }
        if available < crate::model::PACKET_HEADER_LEN {
            return Err(ReplacementRootPacketReadError::Packet(
                crate::runtime::fifo_packet::PacketError::ShortHeader,
            ));
        }
        let header = crate::runtime::fifo_packet::read_root_ring(
            host,
            base,
            capacity,
            head,
            crate::model::PACKET_HEADER_LEN,
        )
        .map_err(ReplacementRootPacketReadError::Memory)?;
        let snapshot_len =
            crate::runtime::fifo_packet::packet_snapshot_len(&header, available, capacity);
        let snapshot = if snapshot_len == crate::model::PACKET_HEADER_LEN {
            header
        } else {
            crate::runtime::fifo_packet::read_root_ring(host, base, capacity, head, snapshot_len)
                .map_err(ReplacementRootPacketReadError::Memory)?
        };
        let packet =
            crate::runtime::fifo_packet::decode_packet(&snapshot, head, available, capacity)
                .map_err(ReplacementRootPacketReadError::Packet)?;
        let ownership = ReplacementRootPacketOwnership {
            head,
            next_head: packet.next_head,
        };
        self.root_packet = Some(ownership);
        Ok(Some(ReplacementRootPacketLease { packet, ownership }))
    }

    pub fn commit_root_packet(
        &mut self,
        lease: ReplacementRootPacketLease,
    ) -> Result<(), Box<ReplacementRootPacketCommitFailure>> {
        if let Err(reason) = self.validate_root_packet_commit(&lease) {
            return Err(Box::new(ReplacementRootPacketCommitFailure {
                reason,
                lease,
            }));
        }
        self.registers.gfx.fifo_read.store(
            lease.ownership.next_head,
            std::sync::atomic::Ordering::Release,
        );
        self.root_packet = None;
        Ok(())
    }

    pub fn validate_root_packet_commit(
        &self,
        lease: &ReplacementRootPacketLease,
    ) -> Result<(), ReplacementRootPacketCommitError> {
        let Some(owned) = self.root_packet else {
            return Err(ReplacementRootPacketCommitError::PacketNotOwned);
        };
        if owned != lease.ownership {
            return Err(ReplacementRootPacketCommitError::OwnershipMismatch);
        }
        let actual = self
            .registers
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire);
        if actual != owned.head {
            return Err(ReplacementRootPacketCommitError::HeadChanged {
                expected: owned.head,
                actual,
            });
        }
        Ok(())
    }

    pub fn read_child_packet(
        &mut self,
        host: &impl crate::runtime::host::HostMemory,
        channel: reims_vgpu_protocol::ChannelId,
    ) -> Result<Option<ReplacementChildPacketLease>, ReplacementChildPacketReadError> {
        let channel_index = channel.get() as usize;
        let Some(register_offset) = crate::model::child_reg_block_offset(channel.get()) else {
            return Err(ReplacementChildPacketReadError::InvalidChannel(channel));
        };
        if self.child_packets[channel_index].is_some() {
            return Err(ReplacementChildPacketReadError::PacketAlreadyOwned(channel));
        }
        if self.registers.gfx.root_page == 0 {
            return Err(ReplacementChildPacketReadError::RootPageUnavailable);
        }
        let page_bytes = 1u64
            .checked_shl(self.page_shift)
            .ok_or(ReplacementChildPacketReadError::RingAddressOverflow)?;
        let registers_gpa = u64::from(self.registers.gfx.root_page)
            .checked_mul(page_bytes)
            .and_then(|base| base.checked_add(register_offset))
            .ok_or(ReplacementChildPacketReadError::RingAddressOverflow)?;
        let read_register = |offset, phase| {
            crate::runtime::host::read_u32(host, registers_gpa + offset)
                .map_err(|reason| ReplacementChildPacketReadError::Memory { phase, reason })
        };
        let head = read_register(
            crate::model::CHILD_REG_HEAD,
            ReplacementChildPacketReadPhase::Head,
        )?;
        let stamp_index = read_register(
            crate::model::CHILD_REG_STAMP_INDEX,
            ReplacementChildPacketReadPhase::StampIndex,
        )?;
        let base_pfn = read_register(
            crate::model::CHILD_REG_BASE_PFN,
            ReplacementChildPacketReadPhase::BasePfn,
        )?;
        if base_pfn == 0 {
            return Err(ReplacementChildPacketReadError::RingUnavailable);
        }
        if !self.child_rings[channel_index].valid
            || self.child_rings[channel_index].base_pfn != base_pfn
        {
            let list_gpa = u64::from(base_pfn)
                .checked_mul(page_bytes)
                .ok_or(ReplacementChildPacketReadError::RingAddressOverflow)?;
            let max_entries = u32::try_from(page_bytes / crate::model::CHILD_RING_PFN_ENTRY_LEN)
                .expect("one supported guest page's PFN count fits u32");
            let mut page_gpas = Vec::new();
            for entry in 0..max_entries {
                let pfn = crate::runtime::host::read_u32(
                    host,
                    list_gpa + u64::from(entry) * crate::model::CHILD_RING_PFN_ENTRY_LEN,
                )
                .map_err(|reason| ReplacementChildPacketReadError::Memory {
                    phase: ReplacementChildPacketReadPhase::PageList { entry },
                    reason,
                })?;
                if pfn == 0 {
                    break;
                }
                page_gpas.push(
                    u64::from(pfn)
                        .checked_mul(page_bytes)
                        .ok_or(ReplacementChildPacketReadError::RingAddressOverflow)?,
                );
            }
            let length = u32::try_from(
                u64::try_from(page_gpas.len())
                    .expect("a guest page-list count fits u64")
                    .checked_mul(page_bytes)
                    .ok_or(ReplacementChildPacketReadError::RingLengthOverflow)?,
            )
            .map_err(|_| ReplacementChildPacketReadError::RingLengthOverflow)?;
            self.child_rings[channel_index] = reims_vgpu_core::ChannelRing {
                valid: length != 0,
                base_pfn,
                length,
                page_gpas,
            };
        }
        let ring = &self.child_rings[channel_index];
        if !ring.valid {
            return Err(ReplacementChildPacketReadError::RingUnavailable);
        }
        let tail = read_register(
            crate::model::CHILD_REG_TAIL,
            ReplacementChildPacketReadPhase::Tail,
        )?;
        let available = tail.wrapping_sub(head);
        if available > ring.length {
            return Err(ReplacementChildPacketReadError::InvalidPublishedPointers {
                head,
                tail,
                capacity: ring.length,
            });
        }
        if available == 0 {
            return Ok(None);
        }
        if available < crate::model::PACKET_HEADER_LEN {
            return Err(ReplacementChildPacketReadError::Packet(
                crate::runtime::fifo_packet::PacketError::ShortHeader,
            ));
        }
        let header = crate::runtime::fifo_packet::read_child_ring(
            host,
            &ring.page_gpas,
            ring.length,
            head,
            crate::model::PACKET_HEADER_LEN,
            self.page_shift,
        )
        .map_err(|reason| ReplacementChildPacketReadError::Memory {
            phase: ReplacementChildPacketReadPhase::Header,
            reason,
        })?;
        let snapshot_len =
            crate::runtime::fifo_packet::packet_snapshot_len(&header, available, ring.length);
        let snapshot = if snapshot_len == crate::model::PACKET_HEADER_LEN {
            header
        } else {
            crate::runtime::fifo_packet::read_child_ring(
                host,
                &ring.page_gpas,
                ring.length,
                head,
                snapshot_len,
                self.page_shift,
            )
            .map_err(|reason| ReplacementChildPacketReadError::Memory {
                phase: ReplacementChildPacketReadPhase::Snapshot,
                reason,
            })?
        };
        let packet =
            crate::runtime::fifo_packet::decode_packet(&snapshot, head, available, ring.length)
                .map_err(ReplacementChildPacketReadError::Packet)?;
        let ownership = ReplacementChildPacketOwnership {
            channel,
            registers_gpa,
            base_pfn,
            head,
            next_head: packet.next_head,
        };
        self.child_packets[channel_index] = Some(ownership);
        Ok(Some(ReplacementChildPacketLease {
            packet,
            channel,
            stamp_index,
            ownership,
        }))
    }

    pub fn commit_child_packet(
        &mut self,
        host: &mut impl crate::runtime::host::HostMemory,
        lease: ReplacementChildPacketLease,
    ) -> Result<(), Box<ReplacementChildPacketCommitFailure>> {
        let index = lease.channel.get() as usize;
        let fail = |reason, lease| Box::new(ReplacementChildPacketCommitFailure { reason, lease });
        let Some(owned) = self.child_packets[index] else {
            return Err(fail(
                ReplacementChildPacketCommitError::PacketNotOwned,
                lease,
            ));
        };
        if owned != lease.ownership {
            return Err(fail(
                ReplacementChildPacketCommitError::OwnershipMismatch,
                lease,
            ));
        }
        let actual = match crate::runtime::host::read_u32(
            host,
            owned.registers_gpa + crate::model::CHILD_REG_HEAD,
        ) {
            Ok(actual) => actual,
            Err(reason) => {
                return Err(fail(
                    ReplacementChildPacketCommitError::Memory(reason),
                    lease,
                ));
            }
        };
        if actual != owned.head {
            return Err(fail(
                ReplacementChildPacketCommitError::HeadChanged {
                    expected: owned.head,
                    actual,
                },
                lease,
            ));
        }
        if let Err(reason) = host.write_gpa(
            owned.registers_gpa + crate::model::CHILD_REG_HEAD,
            &owned.next_head.to_le_bytes(),
        ) {
            return Err(fail(
                ReplacementChildPacketCommitError::Memory(reason),
                lease,
            ));
        }
        self.child_packets[index] = None;
        Ok(())
    }

    pub fn reserve_mapper_entry(
        &mut self,
    ) -> Result<Option<ReplacementMapperEntryLease>, ReplacementMapperEntryError> {
        if self.mapper_entry.is_some() {
            return Err(ReplacementMapperEntryError::EntryAlreadyOwned);
        }
        let registers = &self.registers.iosfc;
        if registers.producer == registers.consumer {
            return Ok(None);
        }
        if registers.ring_base == 0 {
            return Err(ReplacementMapperEntryError::RingUnavailable);
        }
        if registers.producer < registers.consumer {
            return Err(ReplacementMapperEntryError::ProducerBehindConsumer {
                producer: registers.producer,
                consumer: registers.consumer,
            });
        }
        let lease = ReplacementMapperEntryLease {
            ring_base: registers.ring_base,
            producer: registers.consumer + 1,
            consumer: registers.consumer,
        };
        self.mapper_entry = Some(lease);
        Ok(Some(lease))
    }

    pub fn validate_mapper_entry(
        &self,
        lease: ReplacementMapperEntryLease,
    ) -> Result<(), ReplacementMapperEntryError> {
        let Some(owned) = self.mapper_entry else {
            return Err(ReplacementMapperEntryError::EntryNotOwned);
        };
        if owned != lease {
            return Err(ReplacementMapperEntryError::OwnershipMismatch);
        }
        let registers = &self.registers.iosfc;
        if registers.ring_base != lease.ring_base
            || registers.consumer != lease.consumer
            || registers.producer < lease.producer
        {
            return Err(ReplacementMapperEntryError::PublicationChanged);
        }
        Ok(())
    }

    pub fn commit_mapper_entry(
        &mut self,
        lease: ReplacementMapperEntryLease,
    ) -> Result<bool, ReplacementMapperEntryError> {
        self.validate_mapper_entry(lease)?;
        self.registers.iosfc.consumer = lease.producer;
        self.mapper_entry = None;
        Ok(self.registers.iosfc.consumer == self.registers.iosfc.producer)
    }

    /// Clear one guest session's transport values while retaining the Arc
    /// identities cloned by the QEMU-facing lock-free read paths.
    pub fn reset(&mut self) {
        let display_interrupt = self.registers.gfx.interrupt_status_disp.clone();
        let gpu_interrupt = self.registers.gfx.interrupt_status_gpu.clone();
        let fault_interrupt = self.registers.gfx.interrupt_fault.clone();
        let child_doorbell = self.registers.gfx.child_doorbell_rung.clone();
        let fifo_read = self.registers.gfx.fifo_read.clone();
        self.registers = reims_vgpu_core::DeviceRegisters::default();
        self.registers.gfx.interrupt_status_disp = display_interrupt;
        self.registers.gfx.interrupt_status_gpu = gpu_interrupt;
        self.registers.gfx.interrupt_fault = fault_interrupt;
        self.registers.gfx.child_doorbell_rung = child_doorbell;
        self.registers.gfx.fifo_read = fifo_read;
        self.registers
            .gfx
            .interrupt_status_disp
            .store(0, std::sync::atomic::Ordering::Release);
        self.registers
            .gfx
            .interrupt_status_gpu
            .store(0, std::sync::atomic::Ordering::Release);
        self.registers
            .gfx
            .interrupt_fault
            .store(0, std::sync::atomic::Ordering::Release);
        self.registers
            .gfx
            .child_doorbell_rung
            .store(0, std::sync::atomic::Ordering::Release);
        self.registers
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        self.pending = ReplacementTransportWork::default();
        self.root_packet = None;
        self.child_rings = std::array::from_fn(|_| reims_vgpu_core::ChannelRing::default());
        self.child_packets = std::array::from_fn(|_| None);
        self.mapper_entry = None;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementStampPage {
    pub base_pfn: u32,
    pub page_shift: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doorbells_coalesce_without_losing_independent_children() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_X86).unwrap();
        owner.request_root();
        assert!(owner.request_child(reims_vgpu_protocol::ChannelId::new(2)));
        assert!(owner.request_child(reims_vgpu_protocol::ChannelId::new(5)));
        assert!(owner.request_child(reims_vgpu_protocol::ChannelId::new(2)));
        owner.request_iosfc();
        assert_eq!(
            owner.take_work(),
            ReplacementTransportWork {
                root: true,
                children: (1 << 2) | (1 << 5),
                iosfc: true,
            }
        );
        assert!(owner.take_work().is_empty());
        assert!(!owner.request_child(reims_vgpu_protocol::ChannelId::new(0)));
    }

    #[test]
    fn reset_preserves_qemu_visible_atomic_identities_and_clears_values() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E).unwrap();
        let gpu_interrupt = owner.registers().gfx.interrupt_status_gpu.clone();
        let fifo_read = owner.registers().gfx.fifo_read.clone();
        owner.registers_mut().gfx.fifo_base_page = 17;
        gpu_interrupt.store(9, std::sync::atomic::Ordering::Release);
        fifo_read.store(31, std::sync::atomic::Ordering::Release);
        owner.request_root();
        owner.reset();

        assert!(std::sync::Arc::ptr_eq(
            &gpu_interrupt,
            &owner.registers().gfx.interrupt_status_gpu,
        ));
        assert!(std::sync::Arc::ptr_eq(
            &fifo_read,
            &owner.registers().gfx.fifo_read,
        ));
        assert_eq!(gpu_interrupt.load(std::sync::atomic::Ordering::Acquire), 0);
        assert_eq!(fifo_read.load(std::sync::atomic::Ordering::Acquire), 0);
        assert_eq!(owner.stamp_page().base_pfn, 0);
        assert!(owner.take_work().is_empty());
    }

    #[test]
    fn mmio_doorbells_publish_transport_work_and_status_reads_clear() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_X86).unwrap();
        assert_eq!(
            owner
                .gfx_write(
                    crate::model::GFX_REG_CHILD_DOORBELL,
                    4,
                    crate::model::MMIO_U32,
                )
                .unwrap()
                .as_ref(),
            [ReplacementGfxWriteEffect::ScheduleDrain]
        );
        assert_eq!(owner.take_work().children, 1 << 4);
        owner
            .gpu_interrupt_status()
            .store(0x51, std::sync::atomic::Ordering::Release);
        assert_eq!(
            owner
                .gfx_read(
                    crate::model::GFX_REG_INTR_STATUS_GPU,
                    crate::model::MMIO_U32,
                )
                .unwrap(),
            0x51
        );
        assert_eq!(
            owner
                .gfx_read(
                    crate::model::GFX_REG_INTR_STATUS_GPU,
                    crate::model::MMIO_U32,
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn protocol_handshake_is_an_explicit_host_effect() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E).unwrap();
        let effects = owner
            .gfx_write(
                crate::model::GFX_REG_VERSION,
                u64::from(u32::MAX),
                crate::model::MMIO_U32,
            )
            .unwrap();
        assert_eq!(
            effects.as_ref(),
            [ReplacementGfxWriteEffect::ProtocolNegotiated(
                crate::model::PROTOCOL_VERSION_MAX,
            )]
        );
        assert_eq!(
            owner
                .gfx_read(crate::model::GFX_REG_VERSION, crate::model::MMIO_U32)
                .unwrap(),
            u64::from(crate::model::PROTOCOL_VERSION_MAX)
        );
    }

    #[test]
    fn iosfc_producer_write_returns_the_exact_capture_and_drain_identity() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E).unwrap();
        assert_eq!(
            owner.iosfc_write(crate::model::IOSFC_REG_RING_BASE, 0x4000),
            None
        );
        assert_eq!(
            owner.iosfc_write(crate::model::IOSFC_REG_PRODUCER, 3),
            Some(ReplacementIosfcWriteEffect::CaptureAndDrain {
                ring_base: 0x4000,
                producer: 3,
            })
        );
        assert!(owner.take_work().iosfc);
    }

    #[test]
    fn root_packet_head_advances_only_with_its_exact_lease() {
        use crate::runtime::host::HostMemory;

        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_X86).unwrap();
        let pfn = 7u32;
        let page_size = 1u64 << owner.page_shift();
        owner.registers_mut().gfx.control_fifo = 1;
        owner.registers_mut().gfx.fifo_base_page = pfn;
        owner.registers_mut().gfx.fifo_start = 0x100;
        owner.registers_mut().gfx.fifo_length = 0x200;
        owner.registers_mut().gfx.fifo_written = crate::model::PACKET_HEADER_LEN + 4;
        let mut bytes = vec![0; crate::model::PACKET_HEADER_LEN as usize + 4];
        reims_vgpu_core::endian::st16(&mut bytes, crate::model::ROOT_OP_DEFINE_FIFO);
        reims_vgpu_core::endian::st32(
            &mut bytes[crate::model::PACKET_TOTAL_SIZE..],
            crate::model::PACKET_HEADER_LEN + 4,
        );
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_COMPLETION_STAMP..], 19);
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_HEADER_LEN as usize..], 4);
        let mut host = crate::runtime::host::FakeHost::new();
        host.map_range(u64::from(pfn) * page_size, page_size as usize, 0);
        host.write_gpa(u64::from(pfn) * page_size + 0x100, &bytes)
            .unwrap();

        let lease = owner.read_root_packet(&host).unwrap().unwrap();
        assert_eq!(lease.packet.completion_stamp, 19);
        assert!(matches!(
            owner.read_root_packet(&host),
            Err(ReplacementRootPacketReadError::PacketAlreadyOwned)
        ));
        assert_eq!(
            owner
                .registers()
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
        owner.commit_root_packet(lease).unwrap();
        assert_eq!(
            owner
                .registers()
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            crate::model::PACKET_HEADER_LEN + 4
        );
    }

    #[test]
    fn unpublished_root_header_does_not_read_guest_memory() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_X86).unwrap();
        owner.registers_mut().gfx.control_fifo = 1;
        owner.registers_mut().gfx.fifo_base_page = 77;
        owner.registers_mut().gfx.fifo_start = 0x100;
        owner.registers_mut().gfx.fifo_length = 0x200;
        owner.registers_mut().gfx.fifo_written = crate::model::PACKET_HEADER_LEN - 1;
        let host = crate::runtime::host::FakeHost::new();
        assert!(matches!(
            owner.read_root_packet(&host),
            Err(ReplacementRootPacketReadError::Packet(
                crate::runtime::fifo_packet::PacketError::ShortHeader
            ))
        ));
    }

    #[test]
    fn child_packet_lease_walks_page_list_and_commits_guest_head_once() {
        use crate::runtime::host::HostMemory;

        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_X86).unwrap();
        let channel = reims_vgpu_protocol::ChannelId::new(4);
        let page_size = 1u64 << owner.page_shift();
        owner.registers_mut().gfx.root_page = 2;
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
            &(crate::model::PACKET_HEADER_LEN + 4).to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(
            registers_gpa + crate::model::CHILD_REG_HEAD,
            &0u32.to_le_bytes(),
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
        host.write_gpa(list_gpa + 4, &0u32.to_le_bytes()).unwrap();
        let mut bytes = vec![0; crate::model::PACKET_HEADER_LEN as usize + 4];
        reims_vgpu_core::endian::st16(&mut bytes, crate::model::CHILD_OP_CURSOR_SHOW);
        reims_vgpu_core::endian::st32(
            &mut bytes[crate::model::PACKET_TOTAL_SIZE..],
            crate::model::PACKET_HEADER_LEN + 4,
        );
        reims_vgpu_core::endian::st32(&mut bytes[crate::model::PACKET_COMPLETION_STAMP..], 41);
        host.write_gpa(ring_gpa, &bytes).unwrap();

        let lease = owner.read_child_packet(&host, channel).unwrap().unwrap();
        assert_eq!(lease.channel, channel);
        assert_eq!(lease.stamp_index, channel.get());
        assert_eq!(lease.packet.completion_stamp, 41);
        assert!(matches!(
            owner.read_child_packet(&host, channel),
            Err(ReplacementChildPacketReadError::PacketAlreadyOwned(found)) if found == channel
        ));
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            0
        );
        owner.commit_child_packet(&mut host, lease).unwrap();
        assert_eq!(
            host.get_u32(registers_gpa + crate::model::CHILD_REG_HEAD),
            crate::model::PACKET_HEADER_LEN + 4
        );
    }

    #[test]
    fn mapper_entries_are_reserved_and_consumed_in_declared_index_order() {
        let mut owner =
            ReplacementTransportOwner::new(reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E).unwrap();
        owner.registers_mut().iosfc.ring_base = 0x4000;
        owner.registers_mut().iosfc.producer = 3;
        let first = owner.reserve_mapper_entry().unwrap().unwrap();
        assert_eq!(first.producer, 1);
        assert_eq!(
            owner.reserve_mapper_entry(),
            Err(ReplacementMapperEntryError::EntryAlreadyOwned)
        );
        assert!(!owner.commit_mapper_entry(first).unwrap());
        let second = owner.reserve_mapper_entry().unwrap().unwrap();
        assert_eq!(second.producer, 2);
        assert!(!owner.commit_mapper_entry(second).unwrap());
        let third = owner.reserve_mapper_entry().unwrap().unwrap();
        assert!(owner.commit_mapper_entry(third).unwrap());
        assert!(owner.reserve_mapper_entry().unwrap().is_none());
    }
}
