use bevy::{
    log::{info, info_span},
    math::Vec3,
};
use std::{mem::transmute, simd::prelude::*};

pub struct VoxelChunk(pub [i8; 32768]);

impl VoxelChunk {
    pub const CHUNK_SIZE: usize = 32;
    const CHUNK_SIZE_MINUS_ONE: usize = VoxelChunk::CHUNK_SIZE - 1;
    pub const X_SHIFT: usize = 10;
    pub const Y_SHIFT: usize = 5;
    pub const Z_SHIFT: usize = 0;
}

pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
}

pub const EDGE_TOP: u8 = 0b0000_0001;
pub const EDGE_BOTTOM: u8 = 0b0000_0010;
pub const EDGE_LEFT: u8 = 0b0000_0100;
pub const EDGE_RIGHT: u8 = 0b0000_1000;
pub const EDGE_FRONT: u8 = 0b0001_0000;
pub const EDGE_BACK: u8 = 0b0010_0000;

pub struct MesherJob<'a> {
    edge_table: Vec<u16>,
    pub volume: &'a [i8],
    pub buffer: Vec<i32>,
    pub indices: Vec<i32>,
    pub vertices: Vec<Vertex>,
    pub recalculate_normals: bool,
    pub edge_mask: u8, // Add edge_mask field
}

#[inline(always)]
fn test_mix_ones_zeros(input: u32x4, masks: u32x4) -> bool {
    !(input & masks == Simd::splat(1) & masks || input & masks == Simd::splat(0) & masks)
}

impl<'a> MesherJob<'a> {
    pub fn new(volume: &'a [i8], edge_mask: u8) -> Self {
        let buffer_size = (VoxelChunk::CHUNK_SIZE + 1) * (VoxelChunk::CHUNK_SIZE + 1) * 2;
        let mut buffer = vec![];
        buffer.resize(buffer_size, 0);

        Self {
            volume,
            edge_mask,
            recalculate_normals: false,
            edge_table: Vec::with_capacity(256),
            indices: Vec::with_capacity(100),
            vertices: Vec::with_capacity(100),
            buffer,
        }
        .calculate_edge_table()
    }

    pub fn update_edge_mask(&mut self, new_mask: u8) {
        self.edge_mask = new_mask;
    }

    fn calculate_edge_table(mut self) -> Self {
        //let calc_edge_table = info_span!("calc_edge_table", name = "calc_edge_table").entered();
        // Initialize cube_edges with zeros
        let mut cube_edges = [0i32; 24];
        let mut k = 0;

        // Build the cube edges lookup table
        for i in 0..8 {
            let mut j = 1;
            while j <= 4 {
                let p = i ^ j;
                if i <= p {
                    cube_edges[k] = i;
                    cube_edges[k + 1] = p;
                    k += 2;
                }
                j <<= 1; // This is what we were missing - increment by powers of 2
            }
        }

        // Clear and pre-allocate the edge table
        self.edge_table.clear();
        self.edge_table.resize(256, 0);

        // Calculate the edge table
        for i in 0..256 {
            let mut em = 0u16;
            for j in (0..24).step_by(2) {
                let a = (i & (1 << cube_edges[j])) != 0;
                let b = (i & (1 << cube_edges[j + 1])) != 0;
                if a != b {
                    em |= 1 << (j >> 1);
                }
            }
            self.edge_table[i] = em;
        }

        self
    }

    pub fn execute(&mut self) {
        let start_time = std::time::Instant::now();
        //let _create_span = info_span!("execute", name = "execute").entered();
        debug_assert!(
            VoxelChunk::CHUNK_SIZE == 32,
            "ChunkSize must be equal to 32!"
        );

        self.indices.clear();
        self.vertices.clear();

        self.process_voxels();

        info!("Elapsed time: {:?}", start_time.elapsed());

        // if self.recalculate_normals {
        //     self.compute_normals();
        // }
    }

    fn process_voxels(&mut self) {
        //let _process_span = info_span!("process_voxels", name = "process_voxels").entered();
        //TODO: remove need to initialize this guy.
        let mut samples01: [i8; 64] = [0; 64];
        let mut samples23: [i8; 64] = [0; 64];

        let volume_ptr = self.volume.as_ptr();

        let mut mask0 = 0;
        let mut mask1 = 0;
        let mut mask2 = 0;
        let mut mask3 = 0;

        for x in 0..VoxelChunk::CHUNK_SIZE_MINUS_ONE {
            (mask2, mask3) = extract_sign_bits_and_sample(&self.volume, &mut samples23, x, 0);

            for y in 0..VoxelChunk::CHUNK_SIZE_MINUS_ONE {
                // Samples arrays are reused in Y loop, so swap those:
                // samples01 should contain voxels at current Y coordinate.
                // samples23 should contain voxels at Y + 1 coordinate.
                // So they can be reused while iterating over Y.
                std::mem::swap(&mut samples01, &mut samples23);

                //let _mask_compute = info_span!("mask_compute", name = "mask_compute").entered();

                mask0 = mask2;
                mask1 = mask3;

                (mask2, mask3) =
                    extract_sign_bits_and_sample(&self.volume, &mut samples23, x, y + 1);

                let mut masks: u32x4 = Simd::from_array([mask0, mask1, mask2, mask3]);

                let zero_ones = test_mix_ones_zeros(masks, Simd::splat(u32::MAX));
                if !zero_ones {
                    continue;
                }

                let masks_f32: f32x4 = unsafe { std::mem::transmute(masks) };
                let movemask = masks_f32.is_sign_negative().to_bitmask();
                let mut corner_mask = (movemask as u32) << 4;

                //drop(_mask_compute);

                let mut samples = [0.0; 8];

                for z in 0..VoxelChunk::CHUNK_SIZE_MINUS_ONE {
                    //let _zspan = info_span!("z_loop", name = "z_loop").entered();
                    corner_mask = corner_mask >> 4;

                    masks <<= 1;

                    let masks_f32: f32x4 = unsafe { std::mem::transmute(masks) };
                    let movemask = masks_f32.is_sign_negative().to_bitmask();
                    corner_mask |= (movemask as u32) << 4;

                    if corner_mask == 0 || corner_mask == 255 {
                        continue;
                    }

                    let edge_mask = self.edge_table[corner_mask as usize] as u8; // Convert to u8

                    unsafe {
                        let zz = z + z;
                        *samples.get_unchecked_mut(0) = *samples01.get_unchecked(zz + 0) as f32;
                        *samples.get_unchecked_mut(1) = *samples01.get_unchecked(zz + 1) as f32;
                        *samples.get_unchecked_mut(2) = *samples23.get_unchecked(zz + 0) as f32;
                        *samples.get_unchecked_mut(3) = *samples23.get_unchecked(zz + 1) as f32;
                        *samples.get_unchecked_mut(4) = *samples01.get_unchecked(zz + 2) as f32;
                        *samples.get_unchecked_mut(5) = *samples01.get_unchecked(zz + 3) as f32;
                        *samples.get_unchecked_mut(6) = *samples23.get_unchecked(zz + 2) as f32;
                        *samples.get_unchecked_mut(7) = *samples23.get_unchecked(zz + 3) as f32;
                    }

                    let pos = [x, y, z];
                    let flip = (corner_mask & 1) != 0;

                    // Pass the edge_mask to mesh_samples
                    self.mesh_samples(pos, &samples, edge_mask, flip);
                }
            }
        }
    }

    fn mesh_samples(&mut self, pos: [usize; 3], samples: &[f32; 8], edge_mask: u8, flip: bool) {
        const R0: i32 = ((VoxelChunk::CHUNK_SIZE + 1) * (VoxelChunk::CHUNK_SIZE + 1)) as i32;
        let mut r = [R0, (VoxelChunk::CHUNK_SIZE + 1) as i32, 1];

        let mut buffer_index = (pos[2] + (VoxelChunk::CHUNK_SIZE + 1) * pos[1]) as i32;

        if pos[0] % 2 == 0 {
            buffer_index +=
                (1 + (VoxelChunk::CHUNK_SIZE + 1) * (VoxelChunk::CHUNK_SIZE + 2)) as i32;
        } else {
            r[0] = -R0;
            buffer_index += (VoxelChunk::CHUNK_SIZE + 2) as i32;
        }

        unsafe {
            *self.buffer.get_unchecked_mut(buffer_index as usize) = self.vertices.len() as i32;

            let vertex_pos = self.get_vertex_position_from_samples(samples, edge_mask as u16);
            let mut position = Vec3::new(
                pos[0] as f32 + vertex_pos.x,
                pos[1] as f32 + vertex_pos.y,
                pos[2] as f32 + vertex_pos.z,
            );

            self.vertices.push(Vertex {
                pos: position,
                normal: if self.recalculate_normals {
                    Vec3::ZERO
                } else {
                    self.get_vertex_normal_from_samples(samples)
                },
            });

            for i in 0..3 {
                if (edge_mask & (1 << i)) == 0 {
                    continue;
                }

                let iu = (i + 1) % 3;
                let iv = (i + 2) % 3;

                if pos[iu] == 0 || pos[iv] == 0 {
                    continue;
                }

                let du = r[iu];
                let dv = r[iv];

                let idx_base = buffer_index;
                let idx_dudv = buffer_index - du - dv;
                let idx_du = buffer_index - du;
                let idx_dv = buffer_index - dv;

                let start_idx = self.indices.len();
                self.indices.resize(start_idx + 6, 0);

                let indices_slice = self.indices.get_unchecked_mut(start_idx..start_idx + 6);
                if flip {
                    indices_slice[0] = *self.buffer.get_unchecked(idx_base as usize);
                    indices_slice[1] = *self.buffer.get_unchecked(idx_dudv as usize);
                    indices_slice[2] = *self.buffer.get_unchecked(idx_du as usize);
                    indices_slice[3] = *self.buffer.get_unchecked(idx_base as usize);
                    indices_slice[4] = *self.buffer.get_unchecked(idx_dv as usize);
                    indices_slice[5] = *self.buffer.get_unchecked(idx_dudv as usize);
                } else {
                    indices_slice[0] = *self.buffer.get_unchecked(idx_base as usize);
                    indices_slice[1] = *self.buffer.get_unchecked(idx_dudv as usize);
                    indices_slice[2] = *self.buffer.get_unchecked(idx_dv as usize);
                    indices_slice[3] = *self.buffer.get_unchecked(idx_base as usize);
                    indices_slice[4] = *self.buffer.get_unchecked(idx_du as usize);
                    indices_slice[5] = *self.buffer.get_unchecked(idx_dudv as usize);
                }
            }
        }
    }

    fn get_vertex_position_from_samples(&self, samples: &[f32; 8], edge_mask: u16) -> Vec3 {
        //let _vertex_pos = info_span!("get_vertex_pos", name = "get_vertex_pos").entered();
        let mut vert_pos = Vec3::ZERO;
        let mut edge_crossings = 0;

        unsafe {
            if (edge_mask & 0x001) != 0 {
                let s0 = *samples.get_unchecked(0);
                let s1 = *samples.get_unchecked(1);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(t, 0.0, 0.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x002) != 0 {
                let s0 = *samples.get_unchecked(0);
                let s1 = *samples.get_unchecked(2);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(0.0, t, 0.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x004) != 0 {
                let s0 = *samples.get_unchecked(0);
                let s1 = *samples.get_unchecked(4);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(0.0, 0.0, t);
                edge_crossings += 1;
            }
            if (edge_mask & 0x008) != 0 {
                let s0 = *samples.get_unchecked(1);
                let s1 = *samples.get_unchecked(3);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(1.0, t, 0.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x010) != 0 {
                let s0 = *samples.get_unchecked(1);
                let s1 = *samples.get_unchecked(5);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(1.0, 0.0, t);
                edge_crossings += 1;
            }
            if (edge_mask & 0x020) != 0 {
                let s0 = *samples.get_unchecked(2);
                let s1 = *samples.get_unchecked(3);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(t, 1.0, 0.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x040) != 0 {
                let s0 = *samples.get_unchecked(2);
                let s1 = *samples.get_unchecked(6);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(0.0, 1.0, t);
                edge_crossings += 1;
            }
            if (edge_mask & 0x080) != 0 {
                let s0 = *samples.get_unchecked(3);
                let s1 = *samples.get_unchecked(7);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(1.0, 1.0, t);
                edge_crossings += 1;
            }
            if (edge_mask & 0x100) != 0 {
                let s0 = *samples.get_unchecked(4);
                let s1 = *samples.get_unchecked(5);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(t, 0.0, 1.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x200) != 0 {
                let s0 = *samples.get_unchecked(4);
                let s1 = *samples.get_unchecked(6);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(0.0, t, 1.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x400) != 0 {
                let s0 = *samples.get_unchecked(5);
                let s1 = *samples.get_unchecked(7);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(1.0, t, 1.0);
                edge_crossings += 1;
            }
            if (edge_mask & 0x800) != 0 {
                let s0 = *samples.get_unchecked(6);
                let s1 = *samples.get_unchecked(7);
                let t = s0 / (s0 - s1);
                vert_pos += Vec3::new(t, 1.0, 1.0);
                edge_crossings += 1;
            }
        }

        vert_pos / edge_crossings as f32
    }

    #[inline(always)]
    fn get_vertex_normal_from_samples(&self, samples: &[f32; 8]) -> Vec3 {
        let z = (samples[4] - samples[0])
            + (samples[5] - samples[1])
            + (samples[6] - samples[2])
            + (samples[7] - samples[3]);

        let y = (samples[2] - samples[0])
            + (samples[3] - samples[1])
            + (samples[6] - samples[4])
            + (samples[7] - samples[5]);

        let x = (samples[1] - samples[0])
            + (samples[3] - samples[2])
            + (samples[5] - samples[4])
            + (samples[7] - samples[6]);

        Vec3::new(x, y, z) * -0.002
    }
}

#[inline(always)]
fn extract_sign_bits_and_sample(
    volume: &[i8],
    samples23: &mut [i8; 64],
    x: usize,
    y: usize,
) -> (u32, u32) {
    //let _span = info_span!("extract_bits", name = "extract_bits").entered();
    let offset = (x << VoxelChunk::X_SHIFT) + (y << VoxelChunk::Y_SHIFT);
    let chunk =
        unsafe { std::slice::from_raw_parts(volume.as_ptr().add(offset), volume.len() - offset) };

    unsafe {
        let lo2: i8x16 = Simd::from_slice(&chunk.get_unchecked(0..16));
        let hi2: i8x16 = Simd::from_slice(&chunk.get_unchecked(16..32));
        let lo3: i8x16 = Simd::from_slice(&chunk.get_unchecked(1024..1040));
        let hi3: i8x16 = Simd::from_slice(&chunk.get_unchecked(1040..1056));

        let (lo2_lo3_low, lo2_lo3_high) = lo2.interleave(lo3);
        let (hi2_hi3_low, hi2_hi3_high) = hi2.interleave(hi3);

        lo2_lo3_low.copy_to_slice(samples23.get_unchecked_mut(0..16));
        lo2_lo3_high.copy_to_slice(samples23.get_unchecked_mut(16..32));
        hi2_hi3_low.copy_to_slice(samples23.get_unchecked_mut(32..48));
        hi2_hi3_high.copy_to_slice(samples23.get_unchecked_mut(48..64));

        let lo2_rev = lo2.reverse();
        let hi2_rev = hi2.reverse();
        let lo3_rev = lo3.reverse();
        let hi3_rev = hi3.reverse();

        // Directly use the sign bits without negation
        let mask2 = ((lo2_rev.is_negative().to_bitmask() as u32) << 16)
            | (hi2_rev.is_negative().to_bitmask() as u32);
        let mask3 = ((lo3_rev.is_negative().to_bitmask() as u32) << 16)
            | (hi3_rev.is_negative().to_bitmask() as u32);

        (mask2, mask3)
    }
}
