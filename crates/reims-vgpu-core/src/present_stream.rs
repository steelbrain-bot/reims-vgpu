//! Per-surface FIFO presentation and exact swapchain retirement.

use crate::QueueTimelinePoint;
use reims_vgpu_protocol::{PresentTicketId, SurfaceId, SwapchainGenerationId};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwapchainState {
    pub generation: SwapchainGenerationId,
    /// Actual image count returned by the host, not the requested depth.
    pub image_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentTicket {
    pub id: PresentTicketId,
    pub swapchain: SwapchainGenerationId,
    pub image_index: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueuedPresent {
    pub ticket: PresentTicket,
    pub completion: QueueTimelinePoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RetiredSwapchain {
    generation: SwapchainGenerationId,
    last_use: Option<QueueTimelinePoint>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentStreamError {
    DuplicateTicket,
    WrongGeneration,
    ImageOutOfRange,
    TicketNotAtFifoHead,
    UnknownTicket,
    TimelineMismatch,
}

#[derive(Clone, Debug)]
pub struct PresentStream {
    surface: SurfaceId,
    current: SwapchainState,
    tickets: BTreeMap<PresentTicketId, QueuedPresent>,
    fifo: VecDeque<PresentTicketId>,
    retired: Vec<RetiredSwapchain>,
    timeline_identity: Option<(
        reims_vgpu_protocol::VulkanDeviceEpochId,
        reims_vgpu_protocol::QueueOwnerId,
    )>,
}

impl PresentStream {
    pub fn new(surface: SurfaceId, current: SwapchainState) -> Self {
        Self {
            surface,
            current,
            tickets: BTreeMap::new(),
            fifo: VecDeque::new(),
            retired: Vec::new(),
            timeline_identity: None,
        }
    }

    pub fn queue(
        &mut self,
        ticket: PresentTicket,
        completion: QueueTimelinePoint,
    ) -> Result<(), PresentStreamError> {
        if self.tickets.contains_key(&ticket.id) {
            return Err(PresentStreamError::DuplicateTicket);
        }
        if ticket.swapchain != self.current.generation {
            return Err(PresentStreamError::WrongGeneration);
        }
        if ticket.image_index >= self.current.image_count {
            return Err(PresentStreamError::ImageOutOfRange);
        }
        let identity = (completion.epoch, completion.queue);
        if self
            .timeline_identity
            .is_some_and(|established| established != identity)
        {
            return Err(PresentStreamError::TimelineMismatch);
        }
        self.timeline_identity.get_or_insert(identity);
        self.tickets
            .insert(ticket.id, QueuedPresent { ticket, completion });
        self.fifo.push_back(ticket.id);
        Ok(())
    }

    pub fn complete(
        &mut self,
        ticket: PresentTicketId,
    ) -> Result<QueuedPresent, PresentStreamError> {
        if self.fifo.front().copied() != Some(ticket) {
            return if self.tickets.contains_key(&ticket) {
                Err(PresentStreamError::TicketNotAtFifoHead)
            } else {
                Err(PresentStreamError::UnknownTicket)
            };
        }
        self.fifo.pop_front();
        Ok(self.tickets.remove(&ticket).unwrap())
    }

    pub fn replace(&mut self, replacement: SwapchainState) {
        let last_use = self
            .tickets
            .values()
            .filter(|present| present.ticket.swapchain == self.current.generation)
            .map(|present| present.completion)
            .max_by_key(|point| point.value);
        self.retired.push(RetiredSwapchain {
            generation: self.current.generation,
            last_use,
        });
        self.current = replacement;
    }

    pub fn retire_completed(
        &mut self,
        completed: QueueTimelinePoint,
    ) -> Box<[SwapchainGenerationId]> {
        let mut released = Vec::new();
        self.retired.retain(|retired| {
            let ready = retired.last_use.is_none_or(|point| {
                point.epoch == completed.epoch
                    && point.queue == completed.queue
                    && point.value <= completed.value
            });
            if ready {
                released.push(retired.generation);
            }
            !ready
        });
        released.into_boxed_slice()
    }

    pub const fn surface(&self) -> SurfaceId {
        self.surface
    }

    pub const fn current(&self) -> SwapchainState {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{QueueOwnerId, QueueTimelineValue, VulkanDeviceEpochId};

    fn point(value: u64) -> QueueTimelinePoint {
        QueueTimelinePoint {
            epoch: VulkanDeviceEpochId::new(1),
            queue: QueueOwnerId::new(0),
            value: QueueTimelineValue::new(value),
        }
    }

    fn swapchain(generation: u64, image_count: u32) -> SwapchainState {
        SwapchainState {
            generation: SwapchainGenerationId::new(generation),
            image_count,
        }
    }

    fn ticket(id: u64, generation: u64, image_index: u32) -> PresentTicket {
        PresentTicket {
            id: PresentTicketId::new(id),
            swapchain: SwapchainGenerationId::new(generation),
            image_index,
        }
    }

    #[test]
    fn actual_returned_image_count_is_authoritative() {
        let mut stream = PresentStream::new(SurfaceId::new(1), swapchain(1, 2));
        assert_eq!(
            stream.queue(ticket(1, 1, 2), point(1)),
            Err(PresentStreamError::ImageOutOfRange)
        );
        stream.queue(ticket(1, 1, 1), point(1)).unwrap();
    }

    #[test]
    fn fifo_completion_cannot_overtake() {
        let mut stream = PresentStream::new(SurfaceId::new(1), swapchain(1, 3));
        stream.queue(ticket(1, 1, 0), point(1)).unwrap();
        stream.queue(ticket(2, 1, 1), point(2)).unwrap();
        assert_eq!(
            stream.complete(PresentTicketId::new(2)),
            Err(PresentStreamError::TicketNotAtFifoHead)
        );
        stream.complete(PresentTicketId::new(1)).unwrap();
        stream.complete(PresentTicketId::new(2)).unwrap();
    }

    #[test]
    fn replaced_swapchain_retires_only_after_its_last_use() {
        let mut stream = PresentStream::new(SurfaceId::new(1), swapchain(1, 2));
        stream.queue(ticket(1, 1, 0), point(7)).unwrap();
        stream.replace(swapchain(2, 3));
        assert!(stream.retire_completed(point(6)).is_empty());
        assert_eq!(
            &*stream.retire_completed(point(7)),
            &[SwapchainGenerationId::new(1)]
        );
        assert_eq!(stream.current(), swapchain(2, 3));
    }

    #[test]
    fn one_stream_never_compares_values_from_different_queue_timelines() {
        let mut stream = PresentStream::new(SurfaceId::new(1), swapchain(1, 2));
        stream.queue(ticket(1, 1, 0), point(1)).unwrap();
        let mut other = point(2);
        other.queue = QueueOwnerId::new(1);
        assert_eq!(
            stream.queue(ticket(2, 1, 1), other),
            Err(PresentStreamError::TimelineMismatch)
        );
    }
}
