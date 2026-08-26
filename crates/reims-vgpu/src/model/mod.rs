//! Register-window and FIFO contract constants used by the replacement device.

mod regs;

pub(crate) use regs::*;
pub use reims_vgpu_core::{
    ChannelRing, CursorState, DisplayHandshake, DisplayOnlinePoll, DisplaySharedPage,
    GfxRegisters as GfxRegs, IosfcRegisters as IosfcRegs, MapperCapture, PendingWork,
    TargetIdentity, TargetKeyDivergence, GFX_MMIO_SIZE,
};
