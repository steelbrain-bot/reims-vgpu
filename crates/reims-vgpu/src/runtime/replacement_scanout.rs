//! QEMU console-copy result for replacement presentation ownership.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScanoutCopyResult {
    Painted,
    Unchanged,
    Failed,
}
