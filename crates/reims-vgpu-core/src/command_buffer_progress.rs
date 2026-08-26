//! Ordered scheduled/completed callback obligations for an API command buffer.
//!
//! The key stays generic until the wire-to-API command-buffer mapping is
//! established. In particular, this module does not promote counted EXEC child
//! streams into command-buffer identities. Callback actions are owned values
//! transferred exactly once to the authorized publication thread.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandBufferProgress {
    Committed,
    Scheduled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandBufferProgressError {
    DuplicateCommandBuffer,
    UnknownCommandBuffer,
    DuplicateScheduledPublication,
    CompletedBeforeScheduled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressPublication<K, A> {
    pub command_buffer: K,
    pub progress: PublishedProgress,
    pub actions: Box<[A]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublishedProgress {
    Scheduled,
    Completed,
}

#[derive(Clone, Debug)]
struct Entry<A> {
    progress: CommandBufferProgress,
    scheduled_actions: Option<Box<[A]>>,
    completed_actions: Box<[A]>,
}

#[derive(Clone, Debug)]
pub struct CommandBufferProgressOwner<K, A> {
    entries: BTreeMap<K, Entry<A>>,
}

impl<K, A> Default for CommandBufferProgressOwner<K, A> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<K: Clone + Ord, A> CommandBufferProgressOwner<K, A> {
    pub fn register(
        &mut self,
        command_buffer: K,
        scheduled_actions: impl Into<Box<[A]>>,
        completed_actions: impl Into<Box<[A]>>,
    ) -> Result<(), CommandBufferProgressError> {
        if self.entries.contains_key(&command_buffer) {
            return Err(CommandBufferProgressError::DuplicateCommandBuffer);
        }
        self.entries.insert(
            command_buffer,
            Entry {
                progress: CommandBufferProgress::Committed,
                scheduled_actions: Some(scheduled_actions.into()),
                completed_actions: completed_actions.into(),
            },
        );
        Ok(())
    }

    pub fn scheduled(
        &mut self,
        command_buffer: &K,
    ) -> Result<ProgressPublication<K, A>, CommandBufferProgressError> {
        let entry = self
            .entries
            .get_mut(command_buffer)
            .ok_or(CommandBufferProgressError::UnknownCommandBuffer)?;
        if entry.progress != CommandBufferProgress::Committed {
            return Err(CommandBufferProgressError::DuplicateScheduledPublication);
        }
        entry.progress = CommandBufferProgress::Scheduled;
        Ok(ProgressPublication {
            command_buffer: command_buffer.clone(),
            progress: PublishedProgress::Scheduled,
            actions: entry.scheduled_actions.take().unwrap(),
        })
    }

    pub fn completed(
        &mut self,
        command_buffer: &K,
    ) -> Result<ProgressPublication<K, A>, CommandBufferProgressError> {
        let progress = self
            .entries
            .get(command_buffer)
            .ok_or(CommandBufferProgressError::UnknownCommandBuffer)?
            .progress;
        if progress != CommandBufferProgress::Scheduled {
            return Err(CommandBufferProgressError::CompletedBeforeScheduled);
        }
        let entry = self.entries.remove(command_buffer).unwrap();
        Ok(ProgressPublication {
            command_buffer: command_buffer.clone(),
            progress: PublishedProgress::Completed,
            actions: entry.completed_actions,
        })
    }

    pub fn pending(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduled_actions_publish_before_completed_actions_exactly_once() {
        let mut owner = CommandBufferProgressOwner::default();
        owner.register(7, ["scheduled"], ["completed"]).unwrap();
        assert_eq!(
            owner.scheduled(&7).unwrap(),
            ProgressPublication {
                command_buffer: 7,
                progress: PublishedProgress::Scheduled,
                actions: Box::new(["scheduled"])
            }
        );
        assert_eq!(
            owner.scheduled(&7),
            Err(CommandBufferProgressError::DuplicateScheduledPublication)
        );
        assert_eq!(
            owner.completed(&7).unwrap(),
            ProgressPublication {
                command_buffer: 7,
                progress: PublishedProgress::Completed,
                actions: Box::new(["completed"])
            }
        );
        assert_eq!(owner.pending(), 0);
    }

    #[test]
    fn completion_cannot_be_inferred_from_commit_or_driver_return() {
        let mut owner = CommandBufferProgressOwner::default();
        owner
            .register(1, Box::<[&str]>::default(), ["done"])
            .unwrap();
        assert_eq!(
            owner.completed(&1),
            Err(CommandBufferProgressError::CompletedBeforeScheduled)
        );
    }

    #[test]
    fn one_blocked_command_buffer_does_not_order_an_independent_identity() {
        let mut owner = CommandBufferProgressOwner::default();
        owner.register(1, ["one scheduled"], ["one done"]).unwrap();
        owner.register(2, ["two scheduled"], ["two done"]).unwrap();
        owner.scheduled(&2).unwrap();
        assert_eq!(owner.completed(&2).unwrap().actions[0], "two done");
        assert_eq!(owner.pending(), 1);
    }
}
