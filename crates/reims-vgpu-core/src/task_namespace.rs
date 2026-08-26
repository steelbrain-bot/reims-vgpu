//! One semantic-generation owner for task-local object identities.
//!
//! Names are resolved before transaction admission and are not cloned with the
//! transaction runtime. Equal integers in independent API namespaces remain
//! different types; explicit release is the only within-generation reuse
//! boundary.

use crate::{ConditionNamespaceError, ConditionNamespaceOwner, NamespaceError, ReferenceNamespace};
use reims_vgpu_protocol::{
    ComputePipelineObject, DepthStencilObject, EventObject, FenceObject, FunctionObject,
    HeapObject, IndirectCommandBufferObject, RasterizationRateMapObject, RenderPipelineObject,
    ResourceId, SamplerObject, SerializerRef, TaskId,
};

#[derive(Clone, Default)]
pub struct TaskNamespaceOwner {
    conditions: ConditionNamespaceOwner,
    samplers: ReferenceNamespace<SamplerObject>,
    depth_stencil: ReferenceNamespace<DepthStencilObject>,
    render_pipelines: ReferenceNamespace<RenderPipelineObject>,
    compute_pipelines: ReferenceNamespace<ComputePipelineObject>,
    functions: ReferenceNamespace<FunctionObject>,
    heaps: ReferenceNamespace<HeapObject>,
    rasterization_rate_maps: ReferenceNamespace<RasterizationRateMapObject>,
    indirect_command_buffers: ReferenceNamespace<IndirectCommandBufferObject>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleasedTaskNamespace {
    pub events: usize,
    pub fences: usize,
    pub samplers: usize,
    pub depth_stencil: usize,
    pub render_pipelines: usize,
    pub compute_pipelines: usize,
    pub functions: usize,
    pub heaps: usize,
    pub rasterization_rate_maps: usize,
    pub indirect_command_buffers: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskNamespaceSnapshot {
    pub conditions: crate::TaskConditionIdentities,
    pub samplers: Box<[TaskNamespaceEntry<SamplerObject>]>,
    pub depth_stencil: Box<[TaskNamespaceEntry<DepthStencilObject>]>,
    pub render_pipelines: Box<[TaskNamespaceEntry<RenderPipelineObject>]>,
    pub compute_pipelines: Box<[TaskNamespaceEntry<ComputePipelineObject>]>,
    pub functions: Box<[TaskNamespaceEntry<FunctionObject>]>,
    pub heaps: Box<[TaskNamespaceEntry<HeapObject>]>,
    pub rasterization_rate_maps: Box<[TaskNamespaceEntry<RasterizationRateMapObject>]>,
    pub indirect_command_buffers: Box<[TaskNamespaceEntry<IndirectCommandBufferObject>]>,
}

pub type TaskNamespaceEntry<M> = (SerializerRef<M>, ResourceId<M>);

macro_rules! namespace_methods {
    ($publish:ident, $resolve:ident, $release:ident, $field:ident, $marker:ty) => {
        pub fn $publish(
            &mut self,
            task: TaskId,
            reference: SerializerRef<$marker>,
        ) -> Result<ResourceId<$marker>, NamespaceError> {
            self.$field.publish(task, reference)
        }

        pub fn $resolve(
            &self,
            task: TaskId,
            reference: SerializerRef<$marker>,
        ) -> Option<ResourceId<$marker>> {
            self.$field.resolve(task, reference)
        }

        pub fn $release(&mut self, task: TaskId, reference: SerializerRef<$marker>) -> bool {
            self.$field.release(task, reference)
        }
    };
}

impl TaskNamespaceOwner {
    pub fn snapshot_task(&self, task: TaskId) -> TaskNamespaceSnapshot {
        TaskNamespaceSnapshot {
            conditions: self.conditions.task_identities(task),
            samplers: self.samplers.live_for_task(task),
            depth_stencil: self.depth_stencil.live_for_task(task),
            render_pipelines: self.render_pipelines.live_for_task(task),
            compute_pipelines: self.compute_pipelines.live_for_task(task),
            functions: self.functions.live_for_task(task),
            heaps: self.heaps.live_for_task(task),
            rasterization_rate_maps: self.rasterization_rate_maps.live_for_task(task),
            indirect_command_buffers: self.indirect_command_buffers.live_for_task(task),
        }
    }

    pub fn publish_event(
        &mut self,
        task: TaskId,
        reference: SerializerRef<EventObject>,
    ) -> Result<ResourceId<EventObject>, ConditionNamespaceError> {
        self.conditions.publish_event(task, reference)
    }

    pub fn publish_fence(
        &mut self,
        task: TaskId,
        reference: SerializerRef<FenceObject>,
    ) -> Result<ResourceId<FenceObject>, ConditionNamespaceError> {
        self.conditions.publish_fence(task, reference)
    }

    pub fn resolve_event(
        &self,
        task: TaskId,
        reference: SerializerRef<EventObject>,
    ) -> Option<ResourceId<EventObject>> {
        self.conditions.resolve_event(task, reference)
    }

    pub fn resolve_fence(
        &self,
        task: TaskId,
        reference: SerializerRef<FenceObject>,
    ) -> Option<ResourceId<FenceObject>> {
        self.conditions.resolve_fence(task, reference)
    }

    pub fn resolve_fence_operation(
        &mut self,
        task: TaskId,
        reference: SerializerRef<FenceObject>,
        kind: crate::FenceOperationKind,
        scope: crate::FenceScope,
    ) -> Result<crate::FenceOperation, ConditionNamespaceError> {
        self.conditions
            .resolve_fence_operation(task, reference, kind, scope)
    }

    pub fn resolve_event_operation(
        &mut self,
        task: TaskId,
        reference: SerializerRef<EventObject>,
        kind: crate::EventOperationKind,
        value: u64,
    ) -> Result<crate::EventOperation, ConditionNamespaceError> {
        self.conditions
            .resolve_event_operation(task, reference, kind, value)
    }

    pub fn release_event(&mut self, task: TaskId, reference: SerializerRef<EventObject>) -> bool {
        self.conditions.release_event(task, reference)
    }

    pub fn release_fence(&mut self, task: TaskId, reference: SerializerRef<FenceObject>) -> bool {
        self.conditions.release_fence(task, reference)
    }

    namespace_methods!(
        publish_sampler,
        resolve_sampler,
        release_sampler,
        samplers,
        SamplerObject
    );
    namespace_methods!(
        publish_depth_stencil,
        resolve_depth_stencil,
        release_depth_stencil,
        depth_stencil,
        DepthStencilObject
    );

    namespace_methods!(
        publish_render_pipeline,
        resolve_render_pipeline,
        release_render_pipeline,
        render_pipelines,
        RenderPipelineObject
    );
    namespace_methods!(
        publish_compute_pipeline,
        resolve_compute_pipeline,
        release_compute_pipeline,
        compute_pipelines,
        ComputePipelineObject
    );
    namespace_methods!(
        publish_function,
        resolve_function,
        release_function,
        functions,
        FunctionObject
    );
    namespace_methods!(publish_heap, resolve_heap, release_heap, heaps, HeapObject);
    namespace_methods!(
        publish_rasterization_rate_map,
        resolve_rasterization_rate_map,
        release_rasterization_rate_map,
        rasterization_rate_maps,
        RasterizationRateMapObject
    );
    namespace_methods!(
        publish_indirect_command_buffer,
        resolve_indirect_command_buffer,
        release_indirect_command_buffer,
        indirect_command_buffers,
        IndirectCommandBufferObject
    );

    pub fn release_task_conditions(&mut self, task: TaskId) -> crate::ReleasedConditionIdentities {
        self.conditions.release_task(task)
    }

    pub fn release_task(&mut self, task: TaskId) -> ReleasedTaskNamespace {
        let conditions = self.conditions.release_task(task);
        ReleasedTaskNamespace {
            events: conditions.events,
            fences: conditions.fences,
            samplers: self.samplers.release_task(task),
            depth_stencil: self.depth_stencil.release_task(task),
            render_pipelines: self.render_pipelines.release_task(task),
            compute_pipelines: self.compute_pipelines.release_task(task),
            functions: self.functions.release_task(task),
            heaps: self.heaps.release_task(task),
            rasterization_rate_maps: self.rasterization_rate_maps.release_task(task),
            indirect_command_buffers: self.indirect_command_buffers.release_task(task),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_namespaces_are_typed_generational_and_task_owned() {
        let mut owner = TaskNamespaceOwner::default();
        let task = TaskId::new(3);
        let render_ref = SerializerRef::new(9);
        let compute_ref = SerializerRef::new(9);
        let sampler_ref = SerializerRef::new(9);
        let depth_ref = SerializerRef::new(9);
        let function_ref = SerializerRef::new(9);
        let heap_ref = SerializerRef::new(9);
        let rate_map_ref = SerializerRef::new(9);
        let icb_ref = SerializerRef::new(9);
        let event_ref = SerializerRef::new(9);
        let fence_ref = SerializerRef::new(9);
        let render = owner.publish_render_pipeline(task, render_ref).unwrap();
        let compute = owner.publish_compute_pipeline(task, compute_ref).unwrap();
        let sampler = owner.publish_sampler(task, sampler_ref).unwrap();
        let depth = owner.publish_depth_stencil(task, depth_ref).unwrap();
        let function = owner.publish_function(task, function_ref).unwrap();
        let heap = owner.publish_heap(task, heap_ref).unwrap();
        let rate_map = owner
            .publish_rasterization_rate_map(task, rate_map_ref)
            .unwrap();
        let icb = owner
            .publish_indirect_command_buffer(task, icb_ref)
            .unwrap();
        let event = owner.publish_event(task, event_ref).unwrap();
        let fence = owner.publish_fence(task, fence_ref).unwrap();
        assert_eq!(
            owner.resolve_render_pipeline(task, render_ref),
            Some(render)
        );
        assert_eq!(
            owner.resolve_compute_pipeline(task, compute_ref),
            Some(compute)
        );
        assert_eq!(owner.resolve_sampler(task, sampler_ref), Some(sampler));
        assert_eq!(owner.resolve_depth_stencil(task, depth_ref), Some(depth));
        assert_eq!(owner.resolve_function(task, function_ref), Some(function));
        assert_eq!(owner.resolve_heap(task, heap_ref), Some(heap));
        assert_eq!(
            owner.resolve_rasterization_rate_map(task, rate_map_ref),
            Some(rate_map)
        );
        assert_eq!(
            owner.resolve_indirect_command_buffer(task, icb_ref),
            Some(icb)
        );

        assert!(owner.release_render_pipeline(task, render_ref));
        let reused = owner.publish_render_pipeline(task, render_ref).unwrap();
        assert_eq!(reused.index(), render.index());
        assert_ne!(reused.generation(), render.generation());

        let snapshot = owner.snapshot_task(task);
        assert_eq!(snapshot.conditions.events.as_ref(), [(event_ref, event)]);
        assert_eq!(snapshot.conditions.fences.as_ref(), [(fence_ref, fence)]);
        assert_eq!(snapshot.render_pipelines.as_ref(), [(render_ref, reused)]);
        assert_eq!(
            snapshot.compute_pipelines.as_ref(),
            [(compute_ref, compute)]
        );
        assert_eq!(snapshot.samplers.as_ref(), [(sampler_ref, sampler)]);
        assert_eq!(snapshot.depth_stencil.as_ref(), [(depth_ref, depth)]);
        assert_eq!(snapshot.functions.as_ref(), [(function_ref, function)]);
        assert_eq!(snapshot.heaps.as_ref(), [(heap_ref, heap)]);
        assert_eq!(
            snapshot.rasterization_rate_maps.as_ref(),
            [(rate_map_ref, rate_map)]
        );
        assert_eq!(snapshot.indirect_command_buffers.as_ref(), [(icb_ref, icb)]);

        let released = owner.release_task(task);
        assert_eq!(released.events, 1);
        assert_eq!(released.fences, 1);
        assert_eq!(released.render_pipelines, 1);
        assert_eq!(released.compute_pipelines, 1);
        assert_eq!(released.samplers, 1);
        assert_eq!(released.depth_stencil, 1);
        assert_eq!(released.functions, 1);
        assert_eq!(released.heaps, 1);
        assert_eq!(released.rasterization_rate_maps, 1);
        assert_eq!(released.indirect_command_buffers, 1);
        assert_eq!(owner.resolve_render_pipeline(task, render_ref), None);
        assert_eq!(owner.resolve_compute_pipeline(task, compute_ref), None);
        assert_eq!(owner.resolve_sampler(task, sampler_ref), None);
        assert_eq!(owner.resolve_depth_stencil(task, depth_ref), None);
        assert_eq!(owner.resolve_function(task, function_ref), None);
        assert_eq!(owner.resolve_heap(task, heap_ref), None);
        assert_eq!(
            owner.resolve_rasterization_rate_map(task, rate_map_ref),
            None
        );
        assert_eq!(owner.resolve_indirect_command_buffer(task, icb_ref), None);
    }
}
