//! Typed replacement boundary for the task address-space lifetime.

#![allow(dead_code)]

use crate::model::{
    DEFINE_TASK_DIRECTORY_PFN, DEFINE_TASK_ID_SHIFT, DEFINE_TASK_LEN, DEFINE_TASK_LENGTH,
    DEFINE_TASK_RAW_ID, DELETE_TASK_ID, DELETE_TASK_LEN, SET_OBJECT_LIST_COUNT,
    SET_OBJECT_LIST_LEN, SET_OBJECT_LIST_PFN, SET_OBJECT_LIST_TASK_ID,
};
use reims_vgpu_core::endian::{ld32, ld64};
use reims_vgpu_protocol::TaskId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementTaskDefinition {
    pub raw_id: u32,
    pub task: TaskId,
    pub kernel_task: bool,
    pub length: u64,
    pub directory_pfn: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTaskDefinitionDecodeError {
    Short { plen: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementObjectList {
    pub task: TaskId,
    pub pfn: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementObjectListDecodeError {
    Short { plen: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodedReplacementTaskDelete {
    pub task: TaskId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTaskDeleteDecodeError {
    Short { plen: usize },
}

pub(crate) fn decode_replacement_task_definition(
    payload: &[u8],
) -> Result<DecodedReplacementTaskDefinition, ReplacementTaskDefinitionDecodeError> {
    if payload.len() < DEFINE_TASK_LEN {
        return Err(ReplacementTaskDefinitionDecodeError::Short {
            plen: payload.len(),
        });
    }
    let raw_id = ld32(&payload[DEFINE_TASK_RAW_ID..]);
    Ok(DecodedReplacementTaskDefinition {
        raw_id,
        task: TaskId::new(raw_id >> DEFINE_TASK_ID_SHIFT),
        kernel_task: raw_id & 1 != 0,
        length: ld64(&payload[DEFINE_TASK_LENGTH..]),
        directory_pfn: ld32(&payload[DEFINE_TASK_DIRECTORY_PFN..]),
    })
}

pub(crate) fn apply_replacement_task_definition<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    definition: DecodedReplacementTaskDefinition,
) -> Result<
    crate::runtime::replacement_session::ReplacementTaskDefinition,
    crate::runtime::replacement_session::ReplacementTaskRetirementError,
> {
    runtime.define_task(definition.task, definition.length, definition.directory_pfn)
}

pub(crate) fn decode_replacement_object_list(
    payload: &[u8],
) -> Result<DecodedReplacementObjectList, ReplacementObjectListDecodeError> {
    if payload.len() < SET_OBJECT_LIST_LEN {
        return Err(ReplacementObjectListDecodeError::Short {
            plen: payload.len(),
        });
    }
    Ok(DecodedReplacementObjectList {
        task: TaskId::new(ld32(&payload[SET_OBJECT_LIST_TASK_ID..])),
        pfn: ld32(&payload[SET_OBJECT_LIST_PFN..]),
        count: ld32(&payload[SET_OBJECT_LIST_COUNT..]),
    })
}

pub(crate) fn apply_replacement_object_list<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    publication: DecodedReplacementObjectList,
) -> Result<
    crate::runtime::replacement_session::ReplacementObjectListPublication,
    crate::runtime::replacement_session::ReplacementObjectListPublicationError,
> {
    runtime.publish_object_list(publication.task, publication.pfn, publication.count)
}

pub(crate) fn decode_replacement_task_delete(
    payload: &[u8],
) -> Result<DecodedReplacementTaskDelete, ReplacementTaskDeleteDecodeError> {
    if payload.len() < DELETE_TASK_LEN {
        return Err(ReplacementTaskDeleteDecodeError::Short {
            plen: payload.len(),
        });
    }
    Ok(DecodedReplacementTaskDelete {
        task: TaskId::new(ld32(&payload[DELETE_TASK_ID..])),
    })
}

pub(crate) fn apply_replacement_task_delete<Semantic: Clone>(
    runtime: &mut crate::runtime::replacement_session::ReplacementRuntimeSession<Semantic>,
    deletion: DecodedReplacementTaskDelete,
) -> Result<
    crate::runtime::replacement_session::ReplacementTaskRetirement,
    ReplacementTaskDeleteApplyError,
> {
    if !runtime.tasks().is_active(deletion.task.get()) {
        return Err(ReplacementTaskDeleteApplyError::TaskUnavailable(
            deletion.task,
        ));
    }
    runtime
        .retire_task(deletion.task)
        .map_err(ReplacementTaskDeleteApplyError::Retirement)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplacementTaskDeleteApplyError {
    TaskUnavailable(TaskId),
    Retirement(crate::runtime::replacement_session::ReplacementTaskRetirementError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::{st32, st64};

    #[test]
    fn kernel_task_word_retains_flag_and_defines_slot_zero() {
        let mut payload = [0u8; DEFINE_TASK_LEN];
        st32(&mut payload[DEFINE_TASK_RAW_ID..], 1);
        st64(&mut payload[DEFINE_TASK_LENGTH..], 0x1_0000_0000);
        st32(&mut payload[DEFINE_TASK_DIRECTORY_PFN..], 19);
        assert_eq!(
            decode_replacement_task_definition(&payload),
            Ok(DecodedReplacementTaskDefinition {
                raw_id: 1,
                task: TaskId::new(0),
                kernel_task: true,
                length: 0x1_0000_0000,
                directory_pfn: 19,
            })
        );
    }

    #[test]
    fn short_task_definition_refuses_before_reading_any_field() {
        assert_eq!(
            decode_replacement_task_definition(&[0; DEFINE_TASK_LEN - 1]),
            Err(ReplacementTaskDefinitionDecodeError::Short {
                plen: DEFINE_TASK_LEN - 1,
            })
        );
    }

    #[test]
    fn object_list_retains_the_exact_task_pfn_and_count() {
        let mut payload = [0u8; SET_OBJECT_LIST_LEN];
        st32(&mut payload[SET_OBJECT_LIST_TASK_ID..], u32::MAX);
        st32(&mut payload[SET_OBJECT_LIST_PFN..], 0xfeed_beef);
        st32(&mut payload[SET_OBJECT_LIST_COUNT..], 0x8000_0001);
        assert_eq!(
            decode_replacement_object_list(&payload),
            Ok(DecodedReplacementObjectList {
                task: TaskId::new(u32::MAX),
                pfn: 0xfeed_beef,
                count: 0x8000_0001,
            })
        );
        assert_eq!(
            decode_replacement_object_list(&payload[..SET_OBJECT_LIST_LEN - 1]),
            Err(ReplacementObjectListDecodeError::Short {
                plen: SET_OBJECT_LIST_LEN - 1,
            })
        );
    }

    #[test]
    fn task_delete_retains_the_unshifted_full_width_slot() {
        assert_eq!(
            decode_replacement_task_delete(&u32::MAX.to_le_bytes()),
            Ok(DecodedReplacementTaskDelete {
                task: TaskId::new(u32::MAX),
            })
        );
        assert_eq!(
            decode_replacement_task_delete(&[0; DELETE_TASK_LEN - 1]),
            Err(ReplacementTaskDeleteDecodeError::Short {
                plen: DELETE_TASK_LEN - 1,
            })
        );
    }
}
