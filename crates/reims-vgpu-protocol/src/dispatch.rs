//! Semantic compute and mesh dispatch geometry decoded from guest records.

pub const MTL_DISPATCH_TYPE_SERIAL: u32 = 0;
pub const MTL_DISPATCH_TYPE_CONCURRENT: u32 = 1;

#[must_use]
pub fn is_declared_dispatch_type(raw: u32) -> bool {
    matches!(raw, MTL_DISPATCH_TYPE_SERIAL | MTL_DISPATCH_TYPE_CONCURRENT)
}

#[must_use]
pub fn workgroup_counts(
    grid: [u32; 3],
    group: [u32; 3],
    grid_is_threads: bool,
) -> Option<[u32; 3]> {
    if grid.iter().chain(&group).any(|&dimension| dimension == 0) {
        return None;
    }
    if !grid_is_threads {
        return Some(grid);
    }
    Some([
        grid[0].div_ceil(group[0]),
        grid[1].div_ceil(group[1]),
        grid[2].div_ceil(group[2]),
    ])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshDrawDims {
    pub grid: [u32; 3],
    pub object_tg: [u32; 3],
    pub mesh_tg: [u32; 3],
    pub object_tg_defaulted: bool,
}

#[must_use]
pub fn mesh_draw_dims(
    grid: [u32; 3],
    object_tg: [u32; 3],
    mesh_tg: [u32; 3],
) -> Option<MeshDrawDims> {
    if grid.iter().chain(&mesh_tg).any(|&dimension| dimension == 0) {
        return None;
    }
    let object_tg_defaulted = object_tg.contains(&0);
    Some(MeshDrawDims {
        grid,
        object_tg: object_tg.map(|dimension| if dimension == 0 { 1 } else { dimension }),
        mesh_tg,
        object_tg_defaulted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_types_and_workgroups_are_total() {
        assert!(is_declared_dispatch_type(0));
        assert!(is_declared_dispatch_type(1));
        assert!(!(2..=64).any(is_declared_dispatch_type));
        assert_eq!(
            workgroup_counts([17, 1, 1], [8, 1, 1], true),
            Some([3, 1, 1])
        );
        assert_eq!(
            workgroup_counts([7, 3, 1], [8, 8, 1], false),
            Some([7, 3, 1])
        );
    }

    #[test]
    fn zero_in_any_required_dimension_means_no_dispatch() {
        for index in 0..6 {
            let mut grid = [4, 4, 4];
            let mut group = [2, 2, 2];
            if index < 3 {
                grid[index] = 0;
            } else {
                group[index - 3] = 0;
            }
            assert_eq!(workgroup_counts(grid, group, true), None);
        }
    }

    #[test]
    fn mesh_defaults_only_the_optional_object_group() {
        let dimensions =
            mesh_draw_dims([7, 3, 1], [8, 0, 0], [32, 1, 1]).expect("valid mesh dimensions");
        assert_eq!(dimensions.object_tg, [8, 1, 1]);
        assert!(dimensions.object_tg_defaulted);
        for index in 0..6 {
            let mut grid = [4, 4, 4];
            let mut mesh = [2, 2, 2];
            if index < 3 {
                grid[index] = 0;
            } else {
                mesh[index - 3] = 0;
            }
            assert_eq!(mesh_draw_dims(grid, [1, 1, 1], mesh), None);
        }
    }
}
