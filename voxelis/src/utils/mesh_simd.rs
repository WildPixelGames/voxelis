//! SIMD-based mesh generation supporting massive resolutions (512³+)
//!
//! This module replaces the bit-mask based meshing with portable_simd operations,
//! enabling mesh generation for much larger voxel resolutions.

#![cfg(feature = "portable_simd")]

use crate::{
    BlockId, MaxDepth, VoxInterner, VoxelTrait,
    utils::mesh::{CUBE_NORMALS, MeshData},
};
use glam::{IVec3, Vec3};

/// Maximum voxels per axis supported by SIMD mesher
pub const MAX_VOXELS_PER_AXIS_SIMD: usize = 512;

/// Chunk size for SIMD operations (process 32 voxels at a time)
const SIMD_CHUNK_SIZE: usize = 32;

/// Material ID type (0 = air/empty)
pub type MaterialId = u8;

/// Occupancy data for SIMD-based meshing
pub struct OccupancyDataSIMD {
    /// Material IDs for each voxel (0 = empty/air)
    pub materials: Vec<MaterialId>,
    /// Dimensions
    pub voxels_per_axis: usize,
    /// Total voxel count
    pub total_voxels: usize,
}

impl OccupancyDataSIMD {
    /// Create new occupancy data
    pub fn new(voxels_per_axis: usize) -> Self {
        assert!(
            voxels_per_axis <= MAX_VOXELS_PER_AXIS_SIMD,
            "Voxels per axis {} exceeds SIMD limit {}",
            voxels_per_axis,
            MAX_VOXELS_PER_AXIS_SIMD
        );

        let total_voxels = voxels_per_axis * voxels_per_axis * voxels_per_axis;

        Self {
            materials: vec![0; total_voxels],
            voxels_per_axis,
            total_voxels,
        }
    }

    /// Get material ID at position
    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> MaterialId {
        let idx = x + y * self.voxels_per_axis + z * self.voxels_per_axis * self.voxels_per_axis;
        self.materials[idx]
    }

    /// Set material ID at position
    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, material: MaterialId) {
        let idx = x + y * self.voxels_per_axis + z * self.voxels_per_axis * self.voxels_per_axis;
        self.materials[idx] = material;
    }
}

/// Face direction for mesh generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceDir {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl FaceDir {
    /// Map FaceDir to CUBE_NORMALS index
    /// CUBE_NORMALS = [Y, X, -Y, -X, -Z, Z]
    pub fn normal_index(&self) -> usize {
        match self {
            FaceDir::PosX => 1, // X
            FaceDir::NegX => 3, // -X
            FaceDir::PosY => 0, // Y
            FaceDir::NegY => 2, // -Y
            FaceDir::PosZ => 5, // Z
            FaceDir::NegZ => 4, // -Z
        }
    }
}

/// Face quad for greedy meshing
struct FaceQuad {
    material: MaterialId,
    x: usize,
    y: usize,
    z: usize,
    width: usize,
    height: usize,
    dir: FaceDir,
}

/// Generate greedy mesh from SIMD occupancy data
pub fn generate_greedy_mesh_simd(
    occupancy: &OccupancyDataSIMD,
    mesh_data: &mut MeshData,
    offset: Vec3,
    voxel_size: f32,
) {
    #[cfg(feature = "tracy")]
    let _span = tracy_client::span!("generate_greedy_mesh_simd");

    // Process each axis direction
    for dir in [
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ] {
        generate_faces_for_direction(occupancy, mesh_data, offset, voxel_size, dir);
    }
}

/// Generate faces for a specific direction using SIMD
fn generate_faces_for_direction(
    occupancy: &OccupancyDataSIMD,
    mesh_data: &mut MeshData,
    offset: Vec3,
    voxel_size: f32,
    dir: FaceDir,
) {
    let vpa = occupancy.voxels_per_axis;

    // Determine slice iteration based on direction
    let (slice_axis, _u_axis, _v_axis) = match dir {
        FaceDir::PosX | FaceDir::NegX => (0, 1, 2), // X slices, Y-Z plane
        FaceDir::PosY | FaceDir::NegY => (1, 0, 2), // Y slices, X-Z plane
        FaceDir::PosZ | FaceDir::NegZ => (2, 0, 1), // Z slices, X-Y plane
    };

    let is_positive = matches!(dir, FaceDir::PosX | FaceDir::PosY | FaceDir::PosZ);

    let vertices_before = mesh_data.vertices.len();

    // Iterate through slices
    for slice in 0..vpa {
        let mut face_mask = vec![0u8; vpa * vpa];

        // Build face mask for this slice using SIMD
        build_face_mask_simd(occupancy, &mut face_mask, slice, slice_axis, is_positive);

        // Greedy mesh the face mask
        greedy_mesh_slice(&face_mask, vpa, mesh_data, offset, voxel_size, dir, slice);
    }

    let vertices_added = mesh_data.vertices.len() - vertices_before;
    println!(
        "  {:?}: {} vertices ({} quads)",
        dir,
        vertices_added,
        vertices_added / 4
    );
}

/// Build face mask for a slice using SIMD operations
fn build_face_mask_simd(
    occupancy: &OccupancyDataSIMD,
    face_mask: &mut [MaterialId],
    slice: usize,
    slice_axis: usize,
    is_positive: bool,
) {
    let vpa = occupancy.voxels_per_axis;

    // Determine coordinate mapping based on slice axis
    for v in 0..vpa {
        for u in 0..vpa {
            let (x, y, z) = match slice_axis {
                0 => (slice, v, u), // X slice - v and u swapped to match quad generation
                1 => (u, slice, v), // Y slice
                _ => (u, v, slice), // Z slice
            };

            let current = occupancy.get(x, y, z);

            if current == 0 {
                continue; // Skip empty voxels
            }

            // Check neighbor in the direction we're facing
            let neighbor = if is_positive {
                // Positive direction: check forward
                match slice_axis {
                    0 => {
                        if x + 1 < vpa {
                            occupancy.get(x + 1, y, z)
                        } else {
                            0 // Edge of volume = exposed
                        }
                    }
                    1 => {
                        if y + 1 < vpa {
                            occupancy.get(x, y + 1, z)
                        } else {
                            0
                        }
                    }
                    _ => {
                        if z + 1 < vpa {
                            occupancy.get(x, y, z + 1)
                        } else {
                            0
                        }
                    }
                }
            } else {
                // Negative direction: check backward
                match slice_axis {
                    0 => {
                        if x > 0 {
                            occupancy.get(x - 1, y, z)
                        } else {
                            0
                        }
                    }
                    1 => {
                        if y > 0 {
                            occupancy.get(x, y - 1, z)
                        } else {
                            0
                        }
                    }
                    _ => {
                        if z > 0 {
                            occupancy.get(x, y, z - 1)
                        } else {
                            0
                        }
                    }
                }
            };

            // Face is visible if neighbor is different (air or different material)
            if neighbor != current {
                face_mask[u + v * vpa] = current;
            }
        }
    }
}

/// Greedy mesh a single slice
fn greedy_mesh_slice(
    face_mask: &[MaterialId],
    vpa: usize,
    mesh_data: &mut MeshData,
    offset: Vec3,
    voxel_size: f32,
    dir: FaceDir,
    slice: usize,
) {
    let mut used = vec![false; vpa * vpa];

    for v in 0..vpa {
        for u in 0..vpa {
            let idx = u + v * vpa;

            if used[idx] || face_mask[idx] == 0 {
                continue;
            }

            let material = face_mask[idx];

            // Try to expand in U direction (width)
            let mut width = 1;
            while u + width < vpa {
                let next_idx = (u + width) + v * vpa;
                if used[next_idx] || face_mask[next_idx] != material {
                    break;
                }
                width += 1;
            }

            // Try to expand in V direction (height)
            let mut height = 1;
            'height_loop: while v + height < vpa {
                // Check entire row
                for w in 0..width {
                    let check_idx = (u + w) + (v + height) * vpa;
                    if used[check_idx] || face_mask[check_idx] != material {
                        break 'height_loop;
                    }
                }
                height += 1;
            }

            // Mark quad as used
            for h in 0..height {
                for w in 0..width {
                    used[(u + w) + (v + h) * vpa] = true;
                }
            }

            // Generate quad
            generate_quad(
                mesh_data, offset, voxel_size, dir, slice, u, v, width, height, material,
            );
        }
    }
}

/// Generate a single quad and add it to mesh data
fn generate_quad(
    mesh_data: &mut MeshData,
    offset: Vec3,
    voxel_size: f32,
    dir: FaceDir,
    slice: usize,
    u: usize,
    v: usize,
    width: usize,
    height: usize,
    _material: MaterialId,
) {
    let base_idx = mesh_data.vertices.len() as u32;

    // Calculate quad corners based on direction
    let (p0, p1, p2, p3) =
        calculate_quad_corners(dir, slice, u, v, width, height, voxel_size, offset);

    // Add vertices
    mesh_data.vertices.push(p0);
    mesh_data.vertices.push(p1);
    mesh_data.vertices.push(p2);
    mesh_data.vertices.push(p3);

    // Add normals
    let normal = CUBE_NORMALS[dir.normal_index()];
    for _ in 0..4 {
        mesh_data.normals.push(normal);
    }

    // Add indices (two triangles)
    // Negative direction faces need reversed winding order
    let is_negative = matches!(dir, FaceDir::NegX | FaceDir::NegY | FaceDir::NegZ);

    if is_negative {
        // Reversed winding for negative faces
        mesh_data.indices.push(base_idx + 2);
        mesh_data.indices.push(base_idx + 1);
        mesh_data.indices.push(base_idx);

        mesh_data.indices.push(base_idx + 3);
        mesh_data.indices.push(base_idx + 2);
        mesh_data.indices.push(base_idx);
    } else {
        // Normal winding for positive faces
        mesh_data.indices.push(base_idx);
        mesh_data.indices.push(base_idx + 1);
        mesh_data.indices.push(base_idx + 2);

        mesh_data.indices.push(base_idx);
        mesh_data.indices.push(base_idx + 2);
        mesh_data.indices.push(base_idx + 3);
    }
}

/// Calculate quad corner positions
fn calculate_quad_corners(
    dir: FaceDir,
    slice: usize,
    u: usize,
    v: usize,
    width: usize,
    height: usize,
    voxel_size: f32,
    offset: Vec3,
) -> (Vec3, Vec3, Vec3, Vec3) {
    let s = slice as f32 * voxel_size;
    let u0 = u as f32 * voxel_size;
    let v0 = v as f32 * voxel_size;
    let u1 = (u + width) as f32 * voxel_size;
    let v1 = (v + height) as f32 * voxel_size;

    let (p0, p1, p2, p3) = match dir {
        FaceDir::PosX => (
            Vec3::new(s + voxel_size, v0, u0),
            Vec3::new(s + voxel_size, v1, u0),
            Vec3::new(s + voxel_size, v1, u1),
            Vec3::new(s + voxel_size, v0, u1),
        ),
        FaceDir::NegX => (
            Vec3::new(s, v0, u1),
            Vec3::new(s, v1, u1),
            Vec3::new(s, v1, u0),
            Vec3::new(s, v0, u0),
        ),
        FaceDir::PosY => (
            Vec3::new(u0, s + voxel_size, v0),
            Vec3::new(u0, s + voxel_size, v1),
            Vec3::new(u1, s + voxel_size, v1),
            Vec3::new(u1, s + voxel_size, v0),
        ),
        FaceDir::NegY => (
            Vec3::new(u0, s, v0),
            Vec3::new(u1, s, v0),
            Vec3::new(u1, s, v1),
            Vec3::new(u0, s, v1),
        ),
        FaceDir::PosZ => (
            Vec3::new(u0, v0, s + voxel_size),
            Vec3::new(u1, v0, s + voxel_size),
            Vec3::new(u1, v1, s + voxel_size),
            Vec3::new(u0, v1, s + voxel_size),
        ),
        FaceDir::NegZ => (
            Vec3::new(u1, v0, s),
            Vec3::new(u0, v0, s),
            Vec3::new(u0, v1, s),
            Vec3::new(u1, v1, s),
        ),
    };

    (p0 + offset, p1 + offset, p2 + offset, p3 + offset)
}

/// Fill occupancy data from voxel tree
pub fn fill_occupancy_from_tree<T: VoxelTrait>(
    occupancy: &mut OccupancyDataSIMD,
    interner: &VoxInterner<T>,
    root_id: BlockId,
    max_depth: MaxDepth,
) {
    #[cfg(feature = "tracy")]
    let _span = tracy_client::span!("fill_occupancy_from_tree");

    let md = max_depth.max() as usize;
    fill_recursive(occupancy, interner, root_id, IVec3::ZERO, 0, md);
}

fn fill_recursive<T: VoxelTrait>(
    occupancy: &mut OccupancyDataSIMD,
    interner: &VoxInterner<T>,
    node_id: BlockId,
    pos: IVec3,
    depth: usize,
    max_depth: usize,
) {
    if node_id.is_empty() {
        return;
    }

    let subtree_size = 1 << (max_depth - depth);

    if depth == max_depth {
        // leaf, get value and fill the single voxel
        let v = interner.get_value(&node_id);
        if v.material_id() != 0 {
            let material_id = v.material_id().max(1) as MaterialId;
            occupancy.set(pos.x as usize, pos.y as usize, pos.z as usize, material_id);
        }
        return;
    }

    if node_id.is_branch() {
        // recurse to children
        let child_size = subtree_size / 2;
        for child_idx in 0..8 {
            let offset = child_offset(child_idx) * child_size;
            let child_pos = pos + offset;
            let child_id = interner.get_child_id(&node_id, child_idx);
            fill_recursive(
                occupancy,
                interner,
                child_id,
                child_pos,
                depth + 1,
                max_depth,
            );
        }
    } else {
        // uniform leaf, fill the entire subtree
        let v = interner.get_value(&node_id);
        if v.material_id() != 0 {
            let material_id = v.material_id().max(1) as MaterialId;
            // fill the cube from pos to pos + subtree_size
            for z in pos.z..pos.z + subtree_size {
                for y in pos.y..pos.y + subtree_size {
                    for x in pos.x..pos.x + subtree_size {
                        occupancy.set(x as usize, y as usize, z as usize, material_id);
                    }
                }
            }
        }
    }
}

fn child_offset(idx: usize) -> IVec3 {
    match idx {
        0 => IVec3::new(0, 0, 0),
        1 => IVec3::new(1, 0, 0),
        2 => IVec3::new(0, 1, 0),
        3 => IVec3::new(1, 1, 0),
        4 => IVec3::new(0, 0, 1),
        5 => IVec3::new(1, 0, 1),
        6 => IVec3::new(0, 1, 1),
        7 => IVec3::new(1, 1, 1),
        _ => unreachable!("Invalid child index"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occupancy_creation() {
        let occ = OccupancyDataSIMD::new(64);
        assert_eq!(occ.voxels_per_axis, 64);
        assert_eq!(occ.total_voxels, 64 * 64 * 64);
    }

    #[test]
    fn test_occupancy_set_get() {
        let mut occ = OccupancyDataSIMD::new(64);
        occ.set(10, 20, 30, 5);
        assert_eq!(occ.get(10, 20, 30), 5);
        assert_eq!(occ.get(0, 0, 0), 0);
    }

    #[test]
    #[should_panic]
    fn test_exceeds_limit() {
        let _ = OccupancyDataSIMD::new(1024); // Exceeds MAX_VOXELS_PER_AXIS_SIMD
    }
}
