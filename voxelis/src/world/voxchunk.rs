#[cfg(feature = "vtm")]
use std::io::{BufReader, Read, Write};

#[cfg(feature = "vtm")]
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
#[cfg(feature = "vtm")]
use rustc_hash::FxHashMap;

use glam::{IVec3, UVec3, Vec3};
use wide::f32x8;

#[cfg(feature = "vtm")]
use crate::io::{
    consts::VTC_MAGIC,
    varint::{decode_varint_u32_from_reader, encode_varint},
};

use crate::{
    spatial::{
        RayCastHit, VoxOpsBatch, VoxOpsBulkWrite, VoxOpsChunkConfig, VoxOpsConfig, VoxOpsDirty,
        VoxOpsMesh, VoxOpsRayIntersection, VoxOpsRead, VoxOpsSpatial3D, VoxOpsState, VoxOpsWrite,
        VoxTree,
    },
    utils::{
        common::to_vec,
        mesh::{self, MeshData, OccupancyDataBuilder},
    },
};

#[cfg(feature = "trace_greedy_timings")]
use crate::utils::mesh::GreedyTimings;

use crate::{Batch, BlockId, Lod, MaxDepth, VoxInterner, VoxelTrait};

pub struct VoxChunk<T: VoxelTrait> {
    data: VoxTree<T>,
    position: IVec3,
    chunk_size: f32,
}

impl<T: VoxelTrait> VoxChunk<T> {
    pub fn with_position(chunk_size: f32, max_depth: MaxDepth, x: i32, y: i32, z: i32) -> Self {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxChunk::with_position");

        Self {
            data: VoxTree::new(max_depth),
            position: IVec3::new(x, y, z),
            chunk_size,
        }
    }

    pub fn set_position(&mut self, x: i32, y: i32, z: i32) {
        self.position = IVec3::new(x, y, z);
    }

    pub fn get_root_id(&self) -> BlockId {
        self.data.get_root_id()
    }
}

impl<T: VoxelTrait> VoxOpsRead<T> for VoxChunk<T> {
    #[inline(always)]
    fn get(&self, interner: &VoxInterner<T>, position: IVec3) -> Option<T> {
        self.data.get(interner, position)
    }
}

impl<T: VoxelTrait> VoxOpsWrite<T> for VoxChunk<T> {
    #[inline(always)]
    fn set(&mut self, interner: &mut VoxInterner<T>, position: IVec3, voxel: T) -> bool {
        self.data.set(interner, position, voxel)
    }
}

impl<T: VoxelTrait> VoxOpsBulkWrite<T> for VoxChunk<T> {
    #[inline(always)]
    fn fill(&mut self, interner: &mut VoxInterner<T>, value: T) {
        self.data.fill(interner, value)
    }

    #[inline(always)]
    fn clear(&mut self, interner: &mut VoxInterner<T>) {
        self.data.clear(interner)
    }
}

impl<T: VoxelTrait> VoxOpsBatch<T> for VoxChunk<T> {
    #[inline(always)]
    fn create_batch(&self) -> Batch<T> {
        self.data.create_batch()
    }

    #[inline(always)]
    fn apply_batch(&mut self, interner: &mut VoxInterner<T>, batch: &Batch<T>) -> bool {
        self.data.apply_batch(interner, batch)
    }
}

impl<T: VoxelTrait> VoxOpsConfig for VoxChunk<T> {
    #[inline(always)]
    fn max_depth(&self, lod: Lod) -> MaxDepth {
        self.data.max_depth(lod)
    }

    #[inline(always)]
    fn voxels_per_axis(&self, lod: Lod) -> u32 {
        self.data.voxels_per_axis(lod)
    }
}

impl<T: VoxelTrait> VoxOpsState for VoxChunk<T> {
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    fn is_leaf(&self) -> bool {
        self.data.is_leaf()
    }
}

impl<T: VoxelTrait> VoxOpsDirty for VoxChunk<T> {
    #[inline(always)]
    fn is_dirty(&self) -> bool {
        self.data.is_dirty()
    }

    #[inline(always)]
    fn mark_dirty(&mut self) {
        self.data.mark_dirty();
    }

    #[inline(always)]
    fn clear_dirty(&mut self) {
        self.data.clear_dirty()
    }
}

impl<T: VoxelTrait> VoxOpsChunkConfig for VoxChunk<T> {
    #[inline(always)]
    fn chunk_dimensions(&self) -> UVec3 {
        UVec3::splat(1)
    }

    #[inline(always)]
    fn chunk_size(&self) -> f32 {
        self.chunk_size
    }

    #[inline(always)]
    fn voxel_size(&self, lod: Lod) -> f32 {
        self.chunk_size / self.data.voxels_per_axis(lod) as f32
    }
}

impl<T: VoxelTrait> VoxOpsSpatial3D for VoxChunk<T> {
    #[inline(always)]
    fn position_3d(&self) -> IVec3 {
        self.position
    }

    #[inline(always)]
    fn world_position_3d(&self) -> Vec3 {
        self.position.as_vec3() * Vec3::splat(self.chunk_size)
    }

    #[inline(always)]
    fn world_center_position_3d(&self) -> Vec3 {
        let half_size = self.chunk_size / 2.0;
        self.world_position_3d() + Vec3::splat(half_size)
    }

    #[inline(always)]
    fn world_size_3d(&self) -> Vec3 {
        Vec3::splat(self.chunk_size)
    }
}

impl<T: VoxelTrait> VoxOpsMesh<T> for VoxChunk<T> {
    fn generate_naive_mesh_arrays(
        &self,
        interner: &VoxInterner<T>,
        mesh_data: &mut MeshData,
        offset: Vec3,
        lod: Lod,
    ) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxChunk::generate_naive_mesh_arrays");

        let chunk_size = self.chunk_size;

        if self.data.is_leaf() {
            let chunk_v0 = mesh::CUBE_VERTS[0] * chunk_size + offset;
            let chunk_v1 = mesh::CUBE_VERTS[1] * chunk_size + offset;
            let chunk_v2 = mesh::CUBE_VERTS[2] * chunk_size + offset;
            let chunk_v3 = mesh::CUBE_VERTS[3] * chunk_size + offset;
            let chunk_v4 = mesh::CUBE_VERTS[4] * chunk_size + offset;
            let chunk_v5 = mesh::CUBE_VERTS[5] * chunk_size + offset;
            let chunk_v6 = mesh::CUBE_VERTS[6] * chunk_size + offset;
            let chunk_v7 = mesh::CUBE_VERTS[7] * chunk_size + offset;

            mesh::add_quad(
                mesh_data,
                [chunk_v0, chunk_v2, chunk_v3, chunk_v1],
                &mesh::VEC_UP,
            );
            mesh::add_quad(
                mesh_data,
                [chunk_v2, chunk_v5, chunk_v6, chunk_v1],
                &mesh::VEC_RIGHT,
            );
            mesh::add_quad(
                mesh_data,
                [chunk_v7, chunk_v5, chunk_v4, chunk_v6],
                &mesh::VEC_DOWN,
            );
            mesh::add_quad(
                mesh_data,
                [chunk_v0, chunk_v7, chunk_v4, chunk_v3],
                &mesh::VEC_LEFT,
            );
            mesh::add_quad(
                mesh_data,
                [chunk_v3, chunk_v6, chunk_v7, chunk_v2],
                &mesh::VEC_BACK,
            );
            mesh::add_quad(
                mesh_data,
                [chunk_v1, chunk_v4, chunk_v5, chunk_v0],
                &mesh::VEC_FORWARD,
            );

            return;
        }

        let max_depth = self.max_depth(lod);
        let voxels_per_axis = self.voxels_per_axis(lod);
        let voxel_size = self.voxel_size(lod);

        let voxel_size_vec3 = Vec3::splat(voxel_size);
        let shift_y = 1 << (2 * max_depth.as_usize());
        let shift_z = 1 << max_depth.as_usize();

        let chunk_v0 = mesh::CUBE_VERTS[0] * voxel_size_vec3 + offset;
        let chunk_v1 = mesh::CUBE_VERTS[1] * voxel_size_vec3 + offset;
        let chunk_v2 = mesh::CUBE_VERTS[2] * voxel_size_vec3 + offset;
        let chunk_v3 = mesh::CUBE_VERTS[3] * voxel_size_vec3 + offset;
        let chunk_v4 = mesh::CUBE_VERTS[4] * voxel_size_vec3 + offset;
        let chunk_v5 = mesh::CUBE_VERTS[5] * voxel_size_vec3 + offset;
        let chunk_v6 = mesh::CUBE_VERTS[6] * voxel_size_vec3 + offset;
        let chunk_v7 = mesh::CUBE_VERTS[7] * voxel_size_vec3 + offset;

        let chunk_v_x = f32x8::from([
            chunk_v0.x, chunk_v1.x, chunk_v2.x, chunk_v3.x, chunk_v4.x, chunk_v5.x, chunk_v6.x,
            chunk_v7.x,
        ]);
        let chunk_v_y = f32x8::from([
            chunk_v0.y, chunk_v1.y, chunk_v2.y, chunk_v3.y, chunk_v4.y, chunk_v5.y, chunk_v6.y,
            chunk_v7.y,
        ]);
        let chunk_v_z = f32x8::from([
            chunk_v0.z, chunk_v1.z, chunk_v2.z, chunk_v3.z, chunk_v4.z, chunk_v5.z, chunk_v6.z,
            chunk_v7.z,
        ]);

        let data = to_vec(interner, &self.data.get_root_id(), max_depth);

        for y in 0..voxels_per_axis {
            let base_index_y = y as usize * shift_y;
            let v_y = f32x8::splat(y as f32 * voxel_size_vec3.y) + chunk_v_y;
            let v_y_array = v_y.to_array();

            for z in 0..voxels_per_axis {
                let base_index_z = base_index_y + z as usize * shift_z;
                let v_z = f32x8::splat(z as f32 * voxel_size_vec3.z) + chunk_v_z;
                let v_z_array = v_z.to_array();

                for x in 0..voxels_per_axis {
                    let index = base_index_z + x as usize;

                    if unsafe { *data.get_unchecked(index) } == T::default() {
                        continue;
                    }

                    let has_top = y + 1 >= voxels_per_axis
                        || unsafe { *data.get_unchecked(index + shift_y) } == T::default();
                    let has_bottom =
                        y == 0 || unsafe { *data.get_unchecked(index - shift_y) } == T::default();
                    let has_front = z + 1 >= voxels_per_axis
                        || unsafe { *data.get_unchecked(index + shift_z) } == T::default();
                    let has_back =
                        z == 0 || unsafe { *data.get_unchecked(index - shift_z) } == T::default();
                    let has_right = x + 1 >= voxels_per_axis
                        || unsafe { *data.get_unchecked(index + 1) } == T::default();
                    let has_left =
                        x == 0 || unsafe { *data.get_unchecked(index - 1) } == T::default();

                    if !(has_top || has_bottom || has_left || has_right || has_back || has_front) {
                        continue;
                    }

                    let v_x = f32x8::splat(x as f32 * voxel_size_vec3.x) + chunk_v_x;
                    let v_x_array = v_x.to_array();

                    let v0 = Vec3::new(v_x_array[0], v_y_array[0], v_z_array[0]);
                    let v1 = Vec3::new(v_x_array[1], v_y_array[1], v_z_array[1]);
                    let v2 = Vec3::new(v_x_array[2], v_y_array[2], v_z_array[2]);
                    let v3 = Vec3::new(v_x_array[3], v_y_array[3], v_z_array[3]);
                    let v4 = Vec3::new(v_x_array[4], v_y_array[4], v_z_array[4]);
                    let v5 = Vec3::new(v_x_array[5], v_y_array[5], v_z_array[5]);
                    let v6 = Vec3::new(v_x_array[6], v_y_array[6], v_z_array[6]);
                    let v7 = Vec3::new(v_x_array[7], v_y_array[7], v_z_array[7]);

                    if has_top {
                        mesh::add_quad(mesh_data, [v0, v2, v3, v1], &mesh::VEC_UP);
                    }
                    if has_right {
                        mesh::add_quad(mesh_data, [v2, v5, v6, v1], &mesh::VEC_RIGHT);
                    }
                    if has_bottom {
                        mesh::add_quad(mesh_data, [v7, v5, v4, v6], &mesh::VEC_DOWN);
                    }
                    if has_left {
                        mesh::add_quad(mesh_data, [v0, v7, v4, v3], &mesh::VEC_LEFT);
                    }
                    if has_front {
                        mesh::add_quad(mesh_data, [v3, v6, v7, v2], &mesh::VEC_BACK);
                    }
                    if has_back {
                        mesh::add_quad(mesh_data, [v1, v4, v5, v0], &mesh::VEC_FORWARD);
                    }
                }
            }
        }
    }

    fn generate_greedy_mesh_arrays(
        &self,
        interner: &VoxInterner<T>,
        mesh_data: &mut MeshData,
        offset: Vec3,
        lod: Lod,
    ) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("chunk_generate_greedy_mesh_arrays");

        let voxel_size = self.voxel_size(lod);

        let mut builder = OccupancyDataBuilder::default();

        let max_depth = self.max_depth(lod);

        #[cfg(feature = "trace_greedy_timings")]
        let mut timings = GreedyTimings::default();

        mesh::generate_occupancy_masks(
            interner,
            &mut builder,
            &self.data.get_root_id(),
            max_depth,
            UVec3::ZERO,
            #[cfg(feature = "trace_greedy_timings")]
            &mut timings,
        );

        let occupancy_data = builder.build();

        mesh::generate_greedy_mesh_arrays(
            &occupancy_data,
            mesh_data,
            max_depth,
            offset,
            voxel_size,
            #[cfg(feature = "trace_greedy_timings")]
            &mut timings,
        );
    }
}

/// Minimal Ray vs AABB intersection testing.
fn ray_aabb_intersection(
    aabb_min: Vec3,
    aabb_max: Vec3,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<Vec3> {
    debug_assert!(aabb_min.x < aabb_max.x, "empty aabb on x");
    debug_assert!(aabb_min.y < aabb_max.y, "empty aabb on y");
    debug_assert!(aabb_min.z < aabb_max.z, "empty aabb on z");
    debug_assert!(ray_direction.is_normalized(), "unnormalized direction");

    // Look at each pair of near and far planes to find the furthest near and nearest far planes.
    let mut t_near = 0f32;
    let mut t_far = f32::MAX;
    for i in 0..3 {
        // Find t of intercept for both planes on this axis
        let t_min = (aabb_min[i] - ray_origin[i]) / ray_direction[i];
        let t_max = (aabb_max[i] - ray_origin[i]) / ray_direction[i];

        // Sort t_min and t_max because we might be on either side of the aabb
        let (s_min, s_max) = if t_min <= t_max {
            (t_min, t_max)
        } else {
            (t_max, t_min)
        };

        // Replace t_near and t_far with farthest t_min and nearest t_max
        t_near = t_near.max(s_min);
        t_far = t_far.min(s_max);
    }

    // If we intersected a near before far, then we intersected the aabb.
    if t_near < t_far {
        Some(ray_origin + ray_direction * t_near)
    } else {
        None
    }
}

impl<T: VoxelTrait> VoxOpsRayIntersection<T> for VoxChunk<T> {
    fn ray_cast(
        &self,
        lod: Lod,
        interner: &VoxInterner<T>,
        origin: Vec3,
        direction: Vec3,
    ) -> Option<RayCastHit<T>> {
        // This function implements "Amanatides, John & Woo, Andrew. (1987). A Fast Voxel Traversal
        // Algorithm for Ray Tracing. Proceedings of EuroGraphics. 87.", accounting for non-one
        // voxel sizes and utilizing a real vector type for clarity.
        debug_assert!(origin.is_finite(), "NaN ray origin");
        debug_assert!(direction.is_finite(), "NaN ray direction");
        debug_assert!(direction.is_normalized(), "ray direction not normalized");

        // Fast-forward the ray into the chunk.
        let Some(origin) = ray_aabb_intersection(
            self.world_position_3d(),
            self.world_position_3d() + self.world_size_3d(),
            origin,
            direction,
        ) else {
            // Ray does not intersect the chunk
            return None;
        };

        // Constant properties of the chunk at this LOD
        let voxels_per_side = self.data.voxels_per_axis(lod) as i32;
        let voxel_size = self.chunk_size / voxels_per_side as f32;

        // Position of the origin relative to the origin of the chunk.
        // Note that the fast-forward intersection above may lose precision,
        // so we need to loosen our assertions as well.
        let rel_origin = origin - self.world_position_3d();
        debug_assert!(rel_origin.x >= -0.001, "ray origin outside chunk");
        debug_assert!(rel_origin.y >= -0.001, "ray origin outside chunk");
        debug_assert!(rel_origin.z >= -0.001, "ray origin outside chunk");
        let high = self.chunk_size + 0.001;
        debug_assert!(rel_origin.x <= high, "ray origin outside chunk");
        debug_assert!(rel_origin.y <= high, "ray origin outside chunk");
        debug_assert!(rel_origin.z <= high, "ray origin outside chunk");

        // Compute the voxel we are currently in by dividing by the voxel size and
        // casting down. E.g. for a voxel size of 1, this floors towards 0.
        //
        // Note1: the AABB fast-forward we did above is inclusive of the high sides. If the
        //        origin is on the high box edge, this will truncate to voxels_per_side --
        //        e.g. out of bounds -- so we have to `min` this case into the proper range.
        // Note2: Fully external origins are still guarded against via the ray-aabb intersect
        //        algorithm and the debug assertions above.
        let voxel_fractional = rel_origin / voxel_size;
        let mut voxel_pos = voxel_fractional
            .as_ivec3()
            .min(IVec3::splat(voxels_per_side - 1));
        debug_assert!(voxel_pos.x >= 0 && voxel_pos.x < voxels_per_side);
        debug_assert!(voxel_pos.y >= 0 && voxel_pos.y < voxels_per_side);
        debug_assert!(voxel_pos.z >= 0 && voxel_pos.z < voxels_per_side);

        // Check if the initial position is within a voxel.
        if let Some(value) = self.get(interner, voxel_pos) {
            return Some(RayCastHit {
                value,
                voxel_position: voxel_pos,
                world_position: origin,
                world_normal: None,
            });
        }

        // Initialize our iteration variables. Renamed from Amanatides and Woo for clarity.
        let mut curr_pos = origin;
        let step_sizes = voxel_size / direction.abs();
        let mut step_dir = IVec3::ONE;
        let mut next_dist = Vec3::MAX;
        {
            // Compute the fractional position within voxel_pos so that we can compute `next_dist`.
            let fract = voxel_fractional - voxel_pos.as_vec3();
            for i in 0..3 {
                if direction[i] < 0.0 {
                    step_dir[i] = -1;
                    next_dist[i] = -fract[i] / direction[i];
                } else if direction[i] > 0.0 {
                    step_dir[i] = 1;
                    next_dist[i] = (1.0 - fract[i]) / direction[i];
                }
                // Note: the else case is unnecessary because of the initialization to f32::MAX.
                // This will avoid it being picked as our minimal axis in the iteration below.
            }
            next_dist *= voxel_size;
        }

        // Iterate through voxels to find the first that is filled
        loop {
            let i = next_dist.min_position();

            // Move to the next voxel in the ray direction
            voxel_pos[i] += step_dir[i];
            curr_pos += direction * next_dist[i];
            next_dist -= next_dist[i];
            next_dist[i] += step_sizes[i];

            // If we hit the edge, there was no intersection
            if voxel_pos[i] < 0 || voxel_pos[i] >= voxels_per_side {
                return None;
            }

            // If the new voxel pos is non-none, then we have intersected a voxel.
            if let Some(value) = self.get(interner, voxel_pos) {
                let mut world_normal = Vec3::ZERO;
                world_normal[i] = -step_dir[i] as f32;
                return Some(RayCastHit {
                    value,
                    voxel_position: voxel_pos,
                    world_position: curr_pos,
                    world_normal: Some(world_normal),
                });
            }
        }
    }
}

#[cfg(feature = "vtm")]
pub fn serialize_chunk<T: VoxelTrait>(
    chunk: &VoxChunk<T>,
    id_map: &FxHashMap<u32, u32>,
    data: &mut Vec<u8>,
) {
    #[cfg(feature = "tracy")]
    let _span = tracy_client::span!("serialize_chunk");

    let mut writer = std::io::BufWriter::new(data);

    writer.write_all(&VTC_MAGIC).unwrap();

    let position = chunk.position_3d();

    writer.write_i32::<BigEndian>(position.x).unwrap();
    writer.write_i32::<BigEndian>(position.y).unwrap();
    writer.write_i32::<BigEndian>(position.z).unwrap();

    let root_id = chunk.get_root_id();
    let new_id = *id_map.get(&root_id.index()).unwrap();
    let new_id_bytes = encode_varint(new_id as usize);

    writer.write_all(&new_id_bytes).unwrap();
}

#[cfg(feature = "vtm")]
pub fn deserialize_chunk<T: VoxelTrait>(
    interner: &mut VoxInterner<T>,
    leaf_patterns: &FxHashMap<u32, (BlockId, T)>,
    patterns: &FxHashMap<u32, (BlockId, [u32; 8], T)>,
    reader: &mut BufReader<&[u8]>,
    chunk_size: f32,
    max_depth: MaxDepth,
) -> VoxChunk<T> {
    #[cfg(feature = "tracy")]
    let _span = tracy_client::span!("deserialize_chunk");

    let mut magic = [0; VTC_MAGIC.len()];
    reader.read_exact(&mut magic).unwrap();
    assert_eq!(magic, VTC_MAGIC);

    // println!("Magic: {:?}", std::str::from_utf8(&magic).unwrap());

    let x = reader.read_i32::<BigEndian>().unwrap();
    let y = reader.read_i32::<BigEndian>().unwrap();
    let z = reader.read_i32::<BigEndian>().unwrap();

    let mut chunk = VoxChunk::with_position(chunk_size, max_depth, x, y, z);

    let root_id = decode_varint_u32_from_reader(reader).unwrap();
    if let Some((block_id, _, _)) = patterns.get(&root_id) {
        chunk.data.set_root_id(interner, *block_id);
    } else {
        let (block_id, _) = leaf_patterns.get(&root_id).unwrap();
        chunk.data.set_root_id(interner, *block_id);
    }

    chunk
}

#[cfg(test)]
mod test {
    use super::*;
    use glam::{EulerRot, Quat};

    // Initialize one max-size chunk at origin, with 0.1 voxel size,
    // with all the voxels at z=0 set to 1.
    fn create_test_chunk(depth: MaxDepth, pos: IVec3) -> (VoxChunk<u8>, VoxInterner<u8>) {
        let mut interner = VoxInterner::<u8>::with_memory_budget(1024 * 1024 * 1);
        let mut batch = Batch::<u8>::new(depth);
        let mut chunk = VoxChunk::with_position(6.4, depth, pos.x, pos.y, pos.z);
        for i in 0..1i32 << depth.max() {
            for j in 0..1i32 << depth.max() {
                batch.set(&mut interner, IVec3::new(i, j, 0), 1u8);
            }
        }
        chunk.apply_batch(&mut interner, &batch);
        (chunk, interner)
    }

    #[test]
    fn test_voxel_intersection_basic() {
        for depth in 2..=6 {
            for lod in 0..=depth {
                let depth = MaxDepth::new(depth);
                let lod = Lod::new(lod);
                let center = ((1i32 << depth.max()) / 2) >> lod.lod();
                let z_intersect = 6.4 / (1i32 << depth.max() >> lod.lod()) as f32;

                let (chunk, interner) = create_test_chunk(depth, IVec3::ZERO);

                let intersect = chunk
                    .ray_cast(lod, &interner, Vec3::splat(3.2), -Vec3::Z)
                    .expect("missed all voxels");

                assert_eq!(intersect.value, 1);
                assert_eq!(intersect.voxel_position, IVec3::new(center, center, 0));

                // For LOD==MaxDepth, the hit will be inside the one voxel,
                // so will not have a normal and the hit position will be the origin.
                if let Some(n) = intersect.world_normal {
                    assert_eq!(intersect.world_position.x, 3.2);
                    assert_eq!(intersect.world_position.y, 3.2);
                    assert!(
                        intersect.world_position.z >= z_intersect - 0.0001
                            && intersect.world_position.z <= z_intersect + 0.0001
                    );
                    assert_eq!(n, Vec3::Z);
                } else {
                    assert_eq!(intersect.world_position, Vec3::splat(3.2));
                }
            }
        }
    }

    #[test]
    fn test_voxel_intersection_inside_at_origin() {
        for depth in 2..=6 {
            for lod in 0..=depth {
                let depth = MaxDepth::new(depth);
                let lod = Lod::new(lod);
                let z_intersect = 6.4 / (1i32 << depth.max() >> lod.lod()) as f32;

                let (chunk, interner) = create_test_chunk(depth, IVec3::ZERO);

                // Throw out rays at 1 degree intervals. Any intersections should happen at 0.1
                // on the z axis and within the chunk on x and y.
                for theta in -90..=90 {
                    for phi in 0..360 {
                        let dir =
                            Quat::from_euler(EulerRot::YZX, theta as f32, phi as f32, 0.) * Vec3::X;
                        if let Some(intersect) =
                            chunk.ray_cast(lod, &interner, Vec3::splat(0.52), dir)
                        {
                            assert_eq!(intersect.value, 1u8);
                            if let Some(n) = intersect.world_normal {
                                assert!(
                                    intersect.world_position.z >= z_intersect - 0.0001
                                        && intersect.world_position.z <= z_intersect + 0.0001
                                );
                                assert!(intersect.world_position.x >= 0.);
                                assert!(intersect.world_position.x <= 6.4);
                                assert!(intersect.world_position.y >= 0.);
                                assert!(intersect.world_position.y <= 6.4);
                                assert_eq!(n, Vec3::Z);
                            } else {
                                assert_eq!(intersect.world_position, Vec3::splat(0.52));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_voxel_intersection_inside_off_origin() {
        for depth in 2..=6 {
            for lod in 0..=depth {
                let depth = MaxDepth::new(depth);
                let lod = Lod::new(lod);
                let z_intersect = 6.4 + 6.4 / (1i32 << depth.max() >> lod.lod()) as f32;

                let (chunk, interner) = create_test_chunk(depth, IVec3::splat(1));

                // Throw out rays at 1 degree intervals. Any intersections should happen at 0.1 on the z
                // axis and within the chunk on x and y.
                for theta in -90..=90 {
                    for phi in 0..360 {
                        let dir =
                            Quat::from_euler(EulerRot::YZX, theta as f32, phi as f32, 0.) * Vec3::X;
                        if let Some(intersect) =
                            chunk.ray_cast(lod, &interner, Vec3::splat(1.5 * 6.4), dir)
                        {
                            assert_eq!(intersect.value, 1u8);
                            if let Some(n) = intersect.world_normal {
                                assert!(
                                    intersect.world_position.z >= z_intersect - 0.0001
                                        && intersect.world_position.z <= z_intersect + 0.0001
                                );
                                assert!(intersect.world_position.x >= 6.4);
                                assert!(intersect.world_position.x <= 2.0 * 6.4);
                                assert!(intersect.world_position.y >= 6.4);
                                assert!(intersect.world_position.y <= 2.0 * 6.4);
                                assert_eq!(n, Vec3::Z);
                            } else {
                                assert_eq!(intersect.world_position, Vec3::splat(1.5 * 6.4));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_voxel_intersection_outside_at_origin() {
        for depth in 2..=6 {
            for lod in 0..=depth {
                let depth = MaxDepth::new(depth);
                let lod = Lod::new(lod);
                let z_intersect = 6.4 / (1i32 << depth.max() >> lod.lod()) as f32;

                let (chunk, interner) = create_test_chunk(depth, IVec3::ZERO);

                // Throw out rays at 1 degree intervals. Any intersections should happen at 0.1 on the z
                // axis and within the chunk on x and y.
                for theta in -90..=90 {
                    for phi in 0..360 {
                        let dir =
                            Quat::from_euler(EulerRot::YZX, theta as f32, phi as f32, 0.) * Vec3::X;
                        if let Some(intersect) =
                            chunk.ray_cast(lod, &interner, Vec3::splat(10.), dir)
                        {
                            // Note: intersection may be on the high sides of the filled section as well,
                            // if casting from outside.
                            assert!(
                                (intersect.world_position.z >= z_intersect - 0.0001
                                    && intersect.world_position.z <= z_intersect + 0.0001)
                                    || intersect.world_position.x > 6.35
                                    || intersect.world_position.y > 6.35
                            );
                            assert!(intersect.world_position.x >= 0.);
                            assert!(intersect.world_position.x <= 6.4001);
                            assert!(intersect.world_position.y >= 0.);
                            assert!(intersect.world_position.y <= 6.4001);
                            if let Some(world_normal) = intersect.world_normal {
                                if intersect.world_position.z > 0.0999 {
                                    assert!(world_normal.x == 0.);
                                    assert!(world_normal.y == 0.);
                                    assert!(world_normal.z == 1.);
                                } else {
                                    assert!(world_normal.x > 0.0 || world_normal.y > 0.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
