//! Capability-driven representation planning for managed resources.
//!
//! Storage mode is applied before host topology. The remaining inputs are
//! direct answers about whether the exact decoded representation can use guest
//! backing, use an import as a transfer endpoint, or use host-visible memory.
//! Topology selects the working memory class only after those semantic and
//! representation questions are answered.

use reims_vgpu_protocol::StorageMode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostMemoryTopology {
    Unified,
    Discrete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkingMemoryClass {
    HostVisible,
    DeviceLocal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationCapabilities {
    /// Guest backing exactly satisfies format, tiling, stride, planes, and
    /// ownership for the required representation.
    pub direct_guest_backing: bool,
    /// Guest backing can be imported and used as a transfer endpoint for this
    /// exact representation, even though it cannot be the working object.
    pub imported_transfer: bool,
    /// The required working representation is valid in host-visible memory.
    pub host_visible_working: bool,
    pub device_local_working: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationRoute {
    RenderPassMemoryless,
    NativeWorking { memory: WorkingMemoryClass },
    DirectGuestAlias,
    HostStagingEndpoint,
    ImportedGuestTransfer { working: WorkingMemoryClass },
    HostVisibleWorking,
    HostStagingTransfer { working: WorkingMemoryClass },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationRefusal {
    UndeclaredStorageMode,
    NoValidWorkingMemory,
}

pub fn plan_representation(
    storage_mode: StorageMode,
    topology: HostMemoryTopology,
    capabilities: RepresentationCapabilities,
) -> Result<RepresentationRoute, RepresentationRefusal> {
    match storage_mode {
        StorageMode::Undeclared => return Err(RepresentationRefusal::UndeclaredStorageMode),
        StorageMode::Memoryless => return Ok(RepresentationRoute::RenderPassMemoryless),
        StorageMode::Private => {
            return working_memory(topology, capabilities)
                .map(|memory| RepresentationRoute::NativeWorking { memory });
        }
        StorageMode::Shared | StorageMode::Managed => {}
    }

    if capabilities.direct_guest_backing {
        return Ok(RepresentationRoute::DirectGuestAlias);
    }
    if capabilities.imported_transfer {
        return working_memory(topology, capabilities)
            .map(|working| RepresentationRoute::ImportedGuestTransfer { working });
    }
    if capabilities.host_visible_working {
        return Ok(RepresentationRoute::HostVisibleWorking);
    }
    working_memory(topology, capabilities)
        .map(|working| RepresentationRoute::HostStagingTransfer { working })
}

fn working_memory(
    topology: HostMemoryTopology,
    capabilities: RepresentationCapabilities,
) -> Result<WorkingMemoryClass, RepresentationRefusal> {
    match topology {
        HostMemoryTopology::Unified if capabilities.host_visible_working => {
            Ok(WorkingMemoryClass::HostVisible)
        }
        _ if capabilities.device_local_working => Ok(WorkingMemoryClass::DeviceLocal),
        _ if capabilities.host_visible_working => Ok(WorkingMemoryClass::HostVisible),
        _ => Err(RepresentationRefusal::NoValidWorkingMemory),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(imported_transfer: bool, host_visible: bool) -> RepresentationCapabilities {
        RepresentationCapabilities {
            direct_guest_backing: false,
            imported_transfer,
            host_visible_working: host_visible,
            device_local_working: true,
        }
    }

    #[test]
    fn four_topology_import_cells_choose_only_permitted_routes() {
        let cases = [
            (
                HostMemoryTopology::Unified,
                caps(true, true),
                RepresentationRoute::ImportedGuestTransfer {
                    working: WorkingMemoryClass::HostVisible,
                },
            ),
            (
                HostMemoryTopology::Unified,
                caps(false, true),
                RepresentationRoute::HostVisibleWorking,
            ),
            (
                HostMemoryTopology::Discrete,
                caps(true, false),
                RepresentationRoute::ImportedGuestTransfer {
                    working: WorkingMemoryClass::DeviceLocal,
                },
            ),
            (
                HostMemoryTopology::Discrete,
                caps(false, false),
                RepresentationRoute::HostStagingTransfer {
                    working: WorkingMemoryClass::DeviceLocal,
                },
            ),
        ];
        for (topology, capabilities, expected) in cases {
            assert_eq!(
                plan_representation(StorageMode::Managed, topology, capabilities),
                Ok(expected)
            );
        }
    }

    #[test]
    fn exact_direct_compatibility_wins_in_every_topology() {
        for topology in [HostMemoryTopology::Unified, HostMemoryTopology::Discrete] {
            let capabilities = RepresentationCapabilities {
                direct_guest_backing: true,
                imported_transfer: true,
                host_visible_working: false,
                device_local_working: false,
            };
            assert_eq!(
                plan_representation(StorageMode::Shared, topology, capabilities),
                Ok(RepresentationRoute::DirectGuestAlias)
            );
        }
    }

    #[test]
    fn storage_modes_are_resolved_before_topology_policy() {
        let no_memory = RepresentationCapabilities {
            direct_guest_backing: false,
            imported_transfer: false,
            host_visible_working: false,
            device_local_working: false,
        };
        for topology in [HostMemoryTopology::Unified, HostMemoryTopology::Discrete] {
            assert_eq!(
                plan_representation(StorageMode::Memoryless, topology, no_memory),
                Ok(RepresentationRoute::RenderPassMemoryless)
            );
            assert_eq!(
                plan_representation(StorageMode::Undeclared, topology, no_memory),
                Err(RepresentationRefusal::UndeclaredStorageMode)
            );
        }
    }

    #[test]
    fn import_does_not_imply_direct_image_compatibility() {
        assert_eq!(
            plan_representation(
                StorageMode::Shared,
                HostMemoryTopology::Unified,
                caps(true, true)
            ),
            Ok(RepresentationRoute::ImportedGuestTransfer {
                working: WorkingMemoryClass::HostVisible
            })
        );
    }
}
