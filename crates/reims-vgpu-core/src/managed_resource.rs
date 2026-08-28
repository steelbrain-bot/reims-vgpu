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

/// How a guest CPU write reaches the object a route builds.
///
/// The routes differ in where the guest's bytes live relative to the object
/// that executes over them, and that difference is the whole content of the
/// question "what has to happen before this object holds what the guest just
/// wrote". Answering it anywhere but here means answering it per caller, and
/// a caller that fell through to "nothing has to happen" planned no upload at
/// all --- so the object stayed empty and every later bind of it refused
/// `StaleExecutionRepresentation` with nothing that could ever repair it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestWriteStaging {
    /// The object *is* the guest's allocation, imported. The store the guest
    /// made is already in it and there is nothing to plan.
    AlreadyHeld,
    /// One allocation the host can write and the GPU can read. The upload is a
    /// transfer from the guest representation straight into it.
    Transfer,
    /// Working memory the host cannot reach. The guest's bytes land in a
    /// staging buffer first and a GPU transfer moves them on.
    StageThenTransfer,
    /// Storage the guest cannot CPU write at all --- private working memory
    /// and render-pass memoryless attachments. A host-valid clear naming one
    /// is a statement this device has no contract for.
    Unwritable,
}

impl RepresentationRoute {
    /// See [`GuestWriteStaging`]. Exhaustive by construction: a route added
    /// without deciding this does not compile.
    pub const fn guest_write_staging(self) -> GuestWriteStaging {
        match self {
            Self::DirectGuestAlias => GuestWriteStaging::AlreadyHeld,
            Self::HostVisibleWorking | Self::HostStagingEndpoint => GuestWriteStaging::Transfer,
            // The import is the guest's bytes, so nothing stages; the transfer
            // carries them from the import into working memory.
            Self::ImportedGuestTransfer { .. } => GuestWriteStaging::Transfer,
            Self::HostStagingTransfer { .. } => GuestWriteStaging::StageThenTransfer,
            Self::NativeWorking { .. } | Self::RenderPassMemoryless => {
                GuestWriteStaging::Unwritable
            }
        }
    }
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

    /// Every route says what a guest write has to do to reach its object.
    ///
    /// The classification used to be a `match` inside the validity path with a
    /// catch-all arm, and two routes fell into it: `HostVisibleWorking` and
    /// `HostStagingEndpoint`, both of which are one allocation a transfer
    /// fills. Falling through planned no upload, so the object held nothing
    /// and every later bind of it refused `StaleExecutionRepresentation` on
    /// every retry --- 123 748 of them on one driven macos-13 boot, holding
    /// the head of its channel from 77 s in until the boot was killed.
    ///
    /// This asserts the whole table rather than the two that were wrong,
    /// because the failure was a route nobody had decided about and the next
    /// one added is the next such route.
    #[test]
    fn every_route_says_how_a_guest_write_reaches_it() {
        let working = WorkingMemoryClass::DeviceLocal;
        for (route, staging) in [
            (
                RepresentationRoute::DirectGuestAlias,
                GuestWriteStaging::AlreadyHeld,
            ),
            (
                RepresentationRoute::HostVisibleWorking,
                GuestWriteStaging::Transfer,
            ),
            (
                RepresentationRoute::HostStagingEndpoint,
                GuestWriteStaging::Transfer,
            ),
            (
                RepresentationRoute::ImportedGuestTransfer { working },
                GuestWriteStaging::Transfer,
            ),
            (
                RepresentationRoute::HostStagingTransfer { working },
                GuestWriteStaging::StageThenTransfer,
            ),
            (
                RepresentationRoute::NativeWorking { memory: working },
                GuestWriteStaging::Unwritable,
            ),
            (
                RepresentationRoute::RenderPassMemoryless,
                GuestWriteStaging::Unwritable,
            ),
        ] {
            assert_eq!(
                route.guest_write_staging(),
                staging,
                "{route:?} must say how a guest write reaches it"
            );
        }
    }
}
