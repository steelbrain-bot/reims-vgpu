//! Backend-independent submission envelopes retained by device state.

use reims_vgpu_protocol::{
    SegmentBoundary, SubmissionId, SubmissionIdentity, SubmissionResourceUse, TaskId,
};
use std::sync::Arc;

/// Protocol context shared by every operation in one submitted command stream.
///
/// The envelope is immutable once published. Executors may retain it without
/// observing later mutation of the decoder or its resource-list accumulator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionContext {
    pub identity: SubmissionIdentity,
    pub resources: Arc<[SubmissionResourceUse]>,
    /// Every admitted segment in command-buffer order.
    pub segments: Arc<[SegmentBoundary]>,
    /// Segment containing the operation currently submitted to the executor.
    pub segment: Option<SegmentBoundary>,
}

impl SubmissionContext {
    /// Context for direct test and tool operations outside a decoded EXEC packet.
    pub fn standalone(task_id: u32) -> Self {
        Self {
            identity: SubmissionIdentity {
                id: SubmissionId::new(0),
                task: TaskId::new(task_id),
            },
            resources: Arc::from([]),
            segments: Arc::from([]),
            segment: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SubmissionContext;

    #[test]
    fn standalone_context_has_no_invented_participation() {
        let context = SubmissionContext::standalone(7);
        assert_eq!(context.identity.task.get(), 7);
        assert_eq!(context.identity.id.get(), 0);
        assert!(context.resources.is_empty());
        assert!(context.segments.is_empty());
        assert_eq!(context.segment, None);
    }
}
