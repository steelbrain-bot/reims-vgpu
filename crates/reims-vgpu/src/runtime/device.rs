//! Composition-owned device session.
//!
//! Guest-visible state lives in [`crate::model::DeviceState`]. This aggregate
//! owns the execution port and host materializations that implement that state;
//! neither is stored in the semantic model. Runtime operations receive this
//! type and reach semantic fields through `Deref`.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::model::{DeviceId, HostReleaseEffect, TaskDefinitionEffect, TaskNamespaceRetirement};
use crate::runtime::executor::{Executor, VulkanExecutor};
use crate::runtime::host::{HostMemory, HostOps};

pub struct Device {
    pub state: crate::model::DeviceState,
    pub(crate) executor: Arc<dyn Executor>,
    pub(crate) bound_buffers: crate::runtime::bound_buffers::BoundBuffers,
}

/// Composition result of replacing a task's semantic and host-side lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskDefinitionTransition {
    pub semantic: TaskDefinitionEffect,
    pub bound_buffers_retired: usize,
    pub gva_resources_retired: usize,
}

/// Composition result of retiring a task's semantic and host-side lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskDeletionTransition {
    pub semantic: Option<TaskNamespaceRetirement>,
    pub bound_buffers_retired: usize,
    pub gva_resources_retired: usize,
}

/// Composition result of deleting one resource and its host materializations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectDeletionTransition {
    pub semantic_removed: bool,
    pub bound_buffers_retired: usize,
    pub gva_resource_retired: bool,
}

/// Composition result of replacing the object-list naming input for a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectListTransition {
    pub applied: bool,
    pub bound_buffers_retired: usize,
    pub gva_resources_retired: usize,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("state", &self.state)
            .field("executor", &self.executor)
            .field("bound_buffers", &self.bound_buffers)
            .finish()
    }
}

impl Deref for Device {
    type Target = crate::model::DeviceState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for Device {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl Device {
    fn emit_state_mutation(decline: crate::model::StateMutationDecline) {
        let discriminant = decline.discriminant();
        crate::observe::Emit::decline("model_state_mutation", &decline).fail_once(discriminant);
    }

    /// Construct the product session for an explicitly selected guest page size.
    pub fn new(id: DeviceId, page_shift: u32) -> Self {
        Self::new_with_executor(id, page_shift, Arc::new(VulkanExecutor::default()))
    }

    pub fn new_with_executor(id: DeviceId, page_shift: u32, executor: Arc<dyn Executor>) -> Self {
        Self {
            state: crate::model::DeviceState::new_with_audit_density(
                id,
                page_shift,
                crate::runtime::gather_witness::audit_density(),
            ),
            executor,
            bound_buffers: Default::default(),
        }
    }

    fn reset_model(&mut self) {
        self.bound_buffers.clear();
        let effect = self.state.reset();
        if let Some(hold) = effect.translation_hold {
            crate::observe::fail(format!(
                "translation_hold_unreleased held_mask={:#x} producer_mask={:#x} episodes={} \
                 (device reset with guest packets still parked behind an AIR load)",
                hold.held_mask, hold.producer_mask, hold.episodes
            ));
        }
    }

    pub fn reset(&mut self) {
        {
            let executor = Arc::clone(&self.executor);
            let _scope = executor.enter();
            executor.reset();
        }
        self.reset_model();
    }

    /// Reset after releasing every host page view owned by this guest lifetime.
    pub fn reset_with_host<H: HostOps>(&mut self, host: &mut H) -> usize {
        let executor = Arc::clone(&self.executor);
        let _scope = executor.enter();
        executor.reset();
        let effects = self.state.take_all_host_release_effects();
        let mut count = 0;
        for effect in effects {
            match effect {
                HostReleaseEffect::RetireGuestImport(import) => {
                    self.executor.retire_guest_import(import);
                }
                HostReleaseEffect::ReleaseView { ptr, len } => {
                    host.unmap_pages(ptr, len);
                    count += 1;
                }
                HostReleaseEffect::RetireLinearResident(_) => unreachable!(),
            }
        }
        self.reset_model();
        count
    }

    pub fn set_object_list(&mut self, task_id: u32, pfn: u32, count: u32) -> bool {
        match self.state.try_set_object_list(task_id, pfn, count) {
            Ok(()) => true,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    /// Replace a task's object-list naming state and every resolution derived
    /// from the previous list as one composition-owned transition.
    pub fn replace_object_list(
        &mut self,
        task_id: u32,
        pfn: u32,
        count: u32,
    ) -> ObjectListTransition {
        let gva_resources_retired =
            crate::runtime::writeback_debt::retire_gva_for_task(self, task_id);
        let bound_buffers_retired = self.bound_buffers.retire_task(task_id);
        let applied = self.set_object_list(task_id, pfn, count);
        ObjectListTransition {
            applied,
            bound_buffers_retired,
            gva_resources_retired,
        }
    }

    /// Define or redefine a task and retire every host resolution derived
    /// from the previous task lifetime before publishing the new one.
    pub fn define_task_transition(
        &mut self,
        task_id: u32,
        length: u64,
        directory_pfn: u32,
    ) -> TaskDefinitionTransition {
        let gva_resources_retired =
            crate::runtime::writeback_debt::retire_gva_for_task(self, task_id);
        let bound_buffers_retired = self.bound_buffers.retire_task(task_id);
        let semantic = self.state.define_task(task_id, length, directory_pfn);
        TaskDefinitionTransition {
            semantic,
            bound_buffers_retired,
            gva_resources_retired,
        }
    }

    /// Delete a task and every host resolution owned by that task lifetime.
    pub fn delete_task_transition(&mut self, task_id: u32) -> TaskDeletionTransition {
        let gva_resources_retired =
            crate::runtime::writeback_debt::retire_gva_for_task(self, task_id);
        let bound_buffers_retired = self.bound_buffers.retire_task(task_id);
        let semantic = self.state.delete_task(task_id);
        TaskDeletionTransition {
            semantic,
            bound_buffers_retired,
            gva_resources_retired,
        }
    }

    /// Delete one resource and every host resolution owned by that reference.
    pub fn delete_object_transition(
        &mut self,
        task_id: u32,
        ref_: u32,
    ) -> ObjectDeletionTransition {
        let gva_resource_retired =
            crate::runtime::writeback_debt::retire_gva_resource(self, task_id, ref_);
        let bound_buffers_retired = self.bound_buffers.retire_ref(task_id, ref_);
        let semantic_removed = self.state.delete_object(task_id, ref_);
        ObjectDeletionTransition {
            semantic_removed,
            bound_buffers_retired,
            gva_resource_retired,
        }
    }

    #[cfg(test)]
    pub fn insert_object(&mut self, task_id: u32, object_ref: u32) -> bool {
        match self.state.try_insert_object(task_id, object_ref) {
            Ok(()) => true,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    pub fn map_surface(&mut self, mapping_id: u32) -> bool {
        match self.state.try_map_surface(mapping_id) {
            Ok(()) => true,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    pub fn map_mapper_surface(
        &mut self,
        mapper_surface: reims_vgpu_protocol::MapperSurfaceRef,
        surface: reims_vgpu_protocol::MapperResolvedSurfaceId,
    ) -> bool {
        match self.state.try_map_mapper_surface(mapper_surface, surface) {
            Ok(mapped) => mapped,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    pub fn unmap_surface(&mut self, mapping_id: u32) -> bool {
        match self.state.try_unmap_surface(mapping_id) {
            Ok(unmapped) => unmapped,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    pub fn attach_mapping_internal(&mut self, mapping_id: u32, mapping_internal: u64) -> bool {
        match self
            .state
            .try_attach_mapping_internal(mapping_id, mapping_internal)
        {
            Ok(()) => true,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    pub fn set_mapping_device_desc(&mut self, mapping_id: u32, desc: &[u8]) -> bool {
        match self.state.try_set_mapping_device_desc(mapping_id, desc) {
            Ok(()) => true,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    pub fn set_mapping_geom(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        format: u16,
    ) -> bool {
        match self
            .state
            .try_set_mapping_geom(mapping_id, width, height, format)
        {
            Ok(()) => true,
            Err(decline) => {
                Self::emit_state_mutation(decline);
                false
            }
        }
    }

    /// Retire every held address materialization for one object reference.
    pub(crate) fn retire_bound_buffers_for_ref(&mut self, task_id: u32, ref_: u32) -> usize {
        self.bound_buffers.retire_ref(task_id, ref_)
    }

    /// Retire held address materializations overlapping a changed task range.
    pub(crate) fn retire_bound_buffers_in_range(
        &mut self,
        task_id: u32,
        gva: u64,
        len: u64,
    ) -> usize {
        self.bound_buffers.retire_range(task_id, gva, len)
    }

    /// Publish one typed semantic failure through the device observation sink.
    pub fn record_fail(&mut self, event: crate::model::FailEvent) {
        crate::observe::Emit::decline("fail_event", &event).fail();
        #[cfg(test)]
        self.state.fails.push(event);
    }

    /// Publish one typed semantic failure at most once for its discriminant.
    pub fn record_fail_once(&mut self, event: crate::model::FailEvent, discriminant: u64) {
        if crate::observe::first_sight(crate::observe::Decline::slug(&event), discriminant) {
            crate::observe::Emit::decline("fail_event", &event).fail();
        }
        #[cfg(test)]
        self.state.fails.push(event);
        #[cfg(not(test))]
        let _ = event;
    }

    pub fn gfx_read(&mut self, offset: u64, size: u32) -> u64 {
        let executor = Arc::clone(&self.executor);
        let _scope = executor.enter();
        crate::runtime::mmio::gfx_read(self, offset, size)
    }

    pub fn gfx_write<H: HostMemory + HostOps>(
        &mut self,
        host: &mut H,
        offset: u64,
        data: u64,
        size: u32,
    ) {
        let executor = Arc::clone(&self.executor);
        let _scope = executor.enter();
        crate::runtime::mmio::gfx_write(self, host, offset, data, size);
    }

    pub fn iosfc_read(&self, offset: u64, size: u32) -> u64 {
        let _scope = self.executor.enter();
        crate::runtime::mmio::iosfc_read(self, offset, size)
    }

    pub fn iosfc_write<H: HostMemory + HostOps>(
        &mut self,
        host: &mut H,
        offset: u64,
        data: u64,
        size: u32,
    ) {
        let executor = Arc::clone(&self.executor);
        let _scope = executor.enter();
        crate::runtime::mmio::iosfc_write(self, host, offset, data, size);
    }

    pub fn drain<H: HostMemory + HostOps>(&mut self, host: &mut H) {
        let executor = Arc::clone(&self.executor);
        let _scope = executor.enter();
        crate::runtime::drain::drain_pending(self, host);
    }

    #[cfg(test)]
    pub fn fails(&self) -> &[crate::model::FailEvent] {
        &self.state.fails
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(gva: u64) -> crate::runtime::bound_buffers::BoundBuffer {
        crate::runtime::bound_buffers::BoundBuffer {
            gva,
            span: 0x1000,
            source_offset: 0,
            runs: Arc::new(Vec::new()),
            pages: None,
        }
    }

    #[test]
    fn semantic_mutation_is_quiet_and_composition_reports_the_typed_decline() {
        const TASK: u32 = 0xfeed_baad;

        let mut semantic =
            crate::model::DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        {
            let cap = crate::observe::FailCapture::start();
            assert!(!semantic.set_object_list(TASK, 1, 1));
            assert!(cap.lines().is_empty(), "semantic state must not own a sink");
        }

        let mut device = Device::new(DeviceId(2), crate::model::PAGE_SHIFT_X86);
        let cap = crate::observe::FailCapture::start();
        assert!(!device.set_object_list(TASK, 1, 1));
        let lines = cap.lines();
        assert!(
            lines.iter().any(|line| {
                line == "model_state_mutation reason=model_set_object_list_task_inactive \
                     task=4276992685"
            }),
            "composition must report the semantic decline: {lines:?}"
        );
    }

    #[test]
    fn task_transitions_retire_semantic_and_host_lifetimes_together() {
        let mut device = Device::new(DeviceId(3), crate::model::PAGE_SHIFT_X86);
        device.state.define_task(7, 0x4000, 1);
        assert!(device.insert_object(7, 11));
        device.bound_buffers.insert(7, 11, 0, None, held(0x1000));
        device
            .state
            .content
            .preconstruction_writes
            .note_write(7, 11);
        device.state.content.pending_writebacks.ensure_gva_resource(
            reims_vgpu_core::GvaResourceKey {
                task_id: 7,
                texture_ref: 11,
            },
            0x1000,
            0x1000,
            None,
        );

        let redefined = device.define_task_transition(7, 0x8000, 2);
        assert_eq!(redefined.bound_buffers_retired, 1);
        assert_eq!(redefined.gva_resources_retired, 1);
        assert_eq!(
            redefined.semantic.kind,
            crate::model::TaskDefinitionKind::RedefinedNewRoot
        );
        assert!(device.bound_buffers.is_empty());
        assert_eq!(device.state.content.preconstruction_writes.tracked(), 0);
        assert!(!device.state.fixtures.objects.contains(&(7, 11)));

        assert!(device.insert_object(7, 12));
        device.bound_buffers.insert(7, 12, 0, None, held(0x2000));
        let deleted = device.delete_task_transition(7);
        assert_eq!(deleted.bound_buffers_retired, 1);
        assert!(deleted.semantic.is_some());
        assert!(!device.state.tasks.is_active(7));
        assert!(device.bound_buffers.is_empty());
    }

    #[test]
    fn resource_naming_transitions_retire_only_their_owned_materializations() {
        let mut device = Device::new(DeviceId(4), crate::model::PAGE_SHIFT_X86);
        device.state.define_task(8, 0x4000, 1);
        assert!(device.insert_object(8, 20));
        assert!(device.insert_object(8, 21));
        device.bound_buffers.insert(8, 20, 0, None, held(0x1000));
        device.bound_buffers.insert(8, 21, 0, None, held(0x3000));
        device.state.content.pending_writebacks.ensure_gva_resource(
            reims_vgpu_core::GvaResourceKey {
                task_id: 8,
                texture_ref: 20,
            },
            0x1000,
            0x1000,
            None,
        );

        let deleted = device.delete_object_transition(8, 20);
        assert!(deleted.semantic_removed);
        assert_eq!(deleted.bound_buffers_retired, 1);
        assert!(deleted.gva_resource_retired);
        assert!(device.bound_buffers.get(8, 20, 0, None).is_none());
        assert!(device.bound_buffers.get(8, 21, 0, None).is_some());

        device
            .state
            .content
            .preconstruction_writes
            .note_write(8, 21);
        let replaced = device.replace_object_list(8, 9, 32);
        assert!(replaced.applied);
        assert_eq!(replaced.bound_buffers_retired, 1);
        assert!(device.bound_buffers.is_empty());
        assert_eq!(device.state.content.preconstruction_writes.tracked(), 0);
    }
}
