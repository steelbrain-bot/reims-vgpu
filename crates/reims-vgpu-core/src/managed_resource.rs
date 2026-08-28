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
/// wrote". It was answered in two places from the same enum --- once where a
/// validity transition picks its upload destination and once where the
/// lifecycle owner validates the pair --- and each carried a catch-all, so
/// neither said what it meant and nothing made them agree. Both now ask here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestWriteStaging {
    /// The object *is* the guest's allocation, imported. The store the guest
    /// made is already in it and there is nothing to plan.
    AlreadyHeld,
    /// Guest memory is imported and readable by the GPU, so nothing stages and
    /// a transfer carries the bytes from the import into working memory. The
    /// backing must carry the alias for this to be reachable, which is what
    /// the lifecycle owner checks against `GUEST_REPRESENTATION`.
    Transfer,
    /// The host cannot reach working memory. The guest's bytes land in a
    /// staging buffer first and a GPU transfer moves them on.
    StageThenTransfer,
    /// This device plans no upload for the route.
    ///
    /// Two different reasons share the arm and neither is a defect here.
    /// Private working memory and memoryless attachments are storage the guest
    /// cannot CPU write at all. `HostVisibleWorking` and `HostStagingEndpoint`
    /// are storage it can, and the device has no upload built for them --- a
    /// guest write against one leaves the object holding nothing, which the
    /// bind reports as `StaleExecutionRepresentation` with `route=` naming
    /// which of them it was. That is the honest reading: a route this device
    /// does not serve, said out loud, rather than a catch-all that read as
    /// "nothing needed".
    NoUploadRoute,
}

impl RepresentationRoute {
    /// See [`GuestWriteStaging`]. Exhaustive by construction: a route added
    /// without deciding this does not compile.
    pub const fn guest_write_staging(self) -> GuestWriteStaging {
        match self {
            Self::DirectGuestAlias => GuestWriteStaging::AlreadyHeld,
            Self::ImportedGuestTransfer { .. } => GuestWriteStaging::Transfer,
            Self::HostStagingTransfer { .. } => GuestWriteStaging::StageThenTransfer,
            Self::HostVisibleWorking
            | Self::HostStagingEndpoint
            | Self::NativeWorking { .. }
            | Self::RenderPassMemoryless => GuestWriteStaging::NoUploadRoute,
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
                GuestWriteStaging::NoUploadRoute,
            ),
            (
                RepresentationRoute::HostStagingEndpoint,
                GuestWriteStaging::NoUploadRoute,
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
                GuestWriteStaging::NoUploadRoute,
            ),
            (
                RepresentationRoute::RenderPassMemoryless,
                GuestWriteStaging::NoUploadRoute,
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
