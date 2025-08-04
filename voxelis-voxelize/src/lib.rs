use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(feature = "memory_stats")]
use std::{fmt::Write, sync::Mutex};

use crossbeam::channel::{Receiver, Sender, bounded};
use glam::{DVec3, IVec3};
#[cfg(feature = "memory_stats")]
use indicatif::ProgressState;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use voxelis_math::triangle_cube_intersection;

use voxelis::{
    Batch, Lod, MaxDepth,
    io::Obj,
    spatial::{VoxOpsBatch, VoxOpsConfig, VoxOpsState, VoxOpsWrite},
    world::VoxModel,
};

#[cfg(feature = "memory_stats")]
use voxelis::interner::InternerStats;

#[cfg(feature = "memory_stats")]
const PROGRESS_TEMPLATE: &str = "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos}/{human_len} ({eta_precise:.0} {stats})";

#[cfg(not(feature = "memory_stats"))]
const PROGRESS_TEMPLATE: &str = "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos}/{human_len} ({eta_precise:.0})";

fn convert_voxel_world_to_chunk_position(
    voxel_world_position: IVec3,
    chunk_world_size: f32,
) -> IVec3 {
    voxel_world_position.as_vec3().floor().as_ivec3() * (chunk_world_size as i32)
}

pub struct ByteSize(pub usize);

impl std::fmt::Display for ByteSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 < 1024 {
            write!(f, "{} B", self.0)
        } else if self.0 < 1024 * 1024 {
            write!(f, "{:.3} KB", self.0 as f64 / 1024.0)
        } else if self.0 < 1024 * 1024 * 1024 {
            write!(f, "{:.3} MB", self.0 as f64 / (1024.0 * 1024.0))
        } else {
            write!(f, "{:.3} GB", self.0 as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

pub struct Voxelizer {
    pub mesh: Obj,
    pub model: VoxModel<i32>,
}

impl Voxelizer {
    pub fn empty(
        max_depth: MaxDepth,
        chunk_world_size: f32,
        mesh: Obj,
        memory_budget: usize,
    ) -> Self {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::empty");

        Self {
            mesh,
            model: VoxModel::empty(max_depth, chunk_world_size, memory_budget),
        }
    }

    pub fn new(
        max_depth: MaxDepth,
        chunk_world_size: f32,
        mesh: Obj,
        memory_budget: usize,
    ) -> Self {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::new");

        let world_bounds_x = (mesh.size.x.ceil() as i32) + 1;
        let world_bounds_y = (mesh.size.y.ceil() as i32) + 1;
        let world_bounds_z = (mesh.size.z.ceil() as i32) + 1;

        let world_bounds = IVec3::new(world_bounds_x, world_bounds_y, world_bounds_z);

        Self {
            mesh,
            model: VoxModel::with_dimensions(
                max_depth,
                chunk_world_size,
                world_bounds,
                memory_budget,
            ),
        }
    }

    pub fn clear(&mut self) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::clear");

        self.model.clear();
    }

    pub fn build_face_to_chunk_map(&mut self) -> FxHashMap<IVec3, Vec<IVec3>> {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::build_face_to_chunk_map");

        let mut chunk_face_map: FxHashMap<IVec3, Vec<IVec3>> = FxHashMap::default();

        let mesh_min = self.mesh.aabb.0;

        let voxels_per_axis = self.model.voxels_per_axis(Lod::new(0));
        let voxel_size: f64 = self.model.chunk_world_size as f64 / voxels_per_axis as f64;
        let inv_voxel_size: f64 = 1.0 / voxel_size;

        for face in &self.mesh.faces {
            let v1 = self.mesh.vertices[(face.x - 1) as usize] - mesh_min;
            let v2 = self.mesh.vertices[(face.y - 1) as usize] - mesh_min;
            let v3 = self.mesh.vertices[(face.z - 1) as usize] - mesh_min;

            let min = v1.min(v2).min(v3);
            let max = v1.max(v2).max(v3);

            let world_min_voxel = (min * inv_voxel_size).floor().as_ivec3();
            let world_max_voxel = (max * inv_voxel_size).ceil().as_ivec3();

            // Determine which chunks this face overlaps
            let min_chunk = world_min_voxel / voxels_per_axis as i32;
            let max_chunk = world_max_voxel / voxels_per_axis as i32;

            for chunk_y in min_chunk.y..=max_chunk.y {
                for chunk_z in min_chunk.z..=max_chunk.z {
                    for chunk_x in min_chunk.x..=max_chunk.x {
                        let chunk_position = IVec3::new(chunk_x, chunk_y, chunk_z);

                        chunk_face_map
                            .entry(chunk_position)
                            .or_default()
                            .push(*face);
                    }
                }
            }
        }

        chunk_face_map
    }

    fn voxelize_chunk(
        chunk_position: IVec3,
        depth: MaxDepth,
        chunk_world_size: f64,
        voxel_size: f64,
        voxels_per_axis: usize,
        mesh_min: DVec3,
        faces: &[IVec3],
        vertices: &[DVec3],
    ) -> Option<Batch<i32>> {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::voxelize_chunk");

        let epsilon = voxel_size * 1e-7;
        let splat = DVec3::splat(epsilon);

        let mut batch = Batch::new(depth);

        let chunk_world_position = chunk_position.as_dvec3() * chunk_world_size;

        // Compute the chunk's world bounding box
        let chunk_world_min = chunk_world_position;
        let chunk_world_max = chunk_world_min + DVec3::splat(chunk_world_size);

        for face in faces.iter() {
            let v1 = vertices[(face.x - 1) as usize] - mesh_min;
            let v2 = vertices[(face.y - 1) as usize] - mesh_min;
            let v3 = vertices[(face.z - 1) as usize] - mesh_min;

            // Compute the face's bounding box in world coordinates
            let face_min = v1.min(v2).min(v3);
            let face_max = v1.max(v2).max(v3);

            // Compute the overlapping region between the face and the chunk
            let overlap_min = face_min.max(chunk_world_min) - splat;
            let overlap_max = face_max.min(chunk_world_max) + splat;

            // Check if there is any overlap
            if overlap_min.x >= overlap_max.x
                || overlap_min.y >= overlap_max.y
                || overlap_min.z >= overlap_max.z
            {
                // No overlap with the current chunk, skip to the next face
                continue;
            }

            // Map the overlapping region to voxel indices within the chunk
            let local_min_voxel = ((overlap_min - chunk_world_min) / voxel_size)
                .floor()
                .as_ivec3();
            let local_max_voxel = ((overlap_max - chunk_world_min) / voxel_size)
                .ceil()
                .as_ivec3();

            // Clamp voxel indices to valid range [0, VOXELS_PER_AXIS - 1]
            let local_min_voxel =
                local_min_voxel.clamp(IVec3::ZERO, IVec3::splat(voxels_per_axis as i32 - 1));
            let local_max_voxel =
                local_max_voxel.clamp(IVec3::ZERO, IVec3::splat(voxels_per_axis as i32 - 1));

            // Iterate over the voxels within the overlapping region
            for y in local_min_voxel.y..=local_max_voxel.y {
                for z in local_min_voxel.z..=local_max_voxel.z {
                    for x in local_min_voxel.x..=local_max_voxel.x {
                        // Compute world position of the voxel
                        let world_voxel_position =
                            chunk_world_min + DVec3::new(x as f64, y as f64, z as f64) * voxel_size;

                        // Expand voxel bounds slightly by epsilon (if needed)
                        let world_min_position = world_voxel_position - splat;
                        let world_max_position =
                            world_voxel_position + DVec3::splat(voxel_size) + splat;

                        // Perform the intersection test
                        if triangle_cube_intersection(
                            (v1, v2, v3),
                            (world_min_position, world_max_position),
                        ) {
                            batch.just_set(IVec3::new(x, y, z), 1);
                        }
                    }
                }
            }
        }

        if batch.has_patches() {
            Some(batch)
        } else {
            None
        }
    }

    pub fn voxelize_mesh(&mut self, chunk_face_map: FxHashMap<IVec3, Vec<IVec3>>) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::voxelize_mesh");

        let (tx, rx): (Sender<(IVec3, Batch<i32>)>, Receiver<(IVec3, Batch<i32>)>) = bounded(1024);

        let lod = Lod::new(0);

        let depth = self.model.max_depth(lod);
        let voxels_per_axis = self.model.voxels_per_axis(lod) as usize;
        let voxel_size = self.model.chunk_world_size as f64 / voxels_per_axis as f64;
        let chunk_world_size = self.model.chunk_world_size as f64;
        let mesh_min = self.mesh.aabb.0;
        let vertices = self.mesh.vertices.clone();

        let chunk_positions = chunk_face_map.keys().cloned().collect::<Vec<_>>();

        let chunks_to_process = chunk_positions.len();
        println!(" Chunks to process: {chunks_to_process}");

        let early_quit_no_faces = Arc::new(AtomicUsize::new(0));
        let early_quit_empty_faces = Arc::new(AtomicUsize::new(0));
        let early_quit_empty_batch = Arc::new(AtomicUsize::new(0));
        let processed_chunks = Arc::new(AtomicUsize::new(0));

        let early_quit_no_faces_clone = early_quit_no_faces.clone();
        let early_quit_empty_faces_clone = early_quit_empty_faces.clone();
        let early_quit_empty_batch_clone = early_quit_empty_batch.clone();
        let processed_chunks_clone = processed_chunks.clone();

        let stop_signal: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let stop_signal_clone = stop_signal.clone();

        let handle = std::thread::spawn(move || {
            println!("Voxelizing {} chunks in parallel", chunk_positions.len());

            chunk_positions.par_iter().for_each(|chunk_position| {
                if stop_signal_clone.load(Ordering::Relaxed) {
                    return;
                }

                let Some(faces) = chunk_face_map.get(chunk_position) else {
                    early_quit_no_faces_clone.fetch_add(1, Ordering::SeqCst);
                    return;
                };

                if faces.is_empty() {
                    early_quit_empty_faces_clone.fetch_add(1, Ordering::SeqCst);
                    return;
                }

                let Some(batch) = Self::voxelize_chunk(
                    *chunk_position,
                    depth,
                    chunk_world_size,
                    voxel_size,
                    voxels_per_axis,
                    mesh_min,
                    faces,
                    &vertices,
                ) else {
                    early_quit_empty_batch_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    return;
                };

                if batch.has_patches() {
                    processed_chunks_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    tx.send((*chunk_position, batch))
                        .expect("Failed to send batch to main thread");
                }
            });

            drop(tx);
        });

        let interner_arc = self.model.get_interner();
        let mut interner = interner_arc.write();

        println!("Applying batches to chunks");

        let bar = ProgressBar::new(chunks_to_process as u64);

        #[cfg(feature = "memory_stats")]
        let total_memory = interner.stats().requested_budget;
        #[cfg(feature = "memory_stats")]
        let interner_stats = Arc::new(Mutex::new(InternerStats::default()));
        #[cfg(feature = "memory_stats")]
        let interner_stats_clone = interner_stats.clone();

        let style = ProgressStyle::with_template(PROGRESS_TEMPLATE).unwrap();

        #[cfg(feature = "memory_stats")]
        let style = style.with_key("stats", move |_state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{}", {
                let stats = interner_stats_clone.lock().unwrap();
                let used_memory = stats.alive_nodes * stats.node_size;

                format!("{} / {}", ByteSize(used_memory), ByteSize(total_memory),)
            })
            .unwrap()
        });

        bar.set_style(style.progress_chars("#>-"));

        bar.enable_steady_tick(Duration::from_millis(16));

        for (chunk_position, batch) in rx.iter() {
            self.model
                .get_or_create_chunk(chunk_position)
                .apply_batch(&mut interner, &batch);

            #[cfg(feature = "memory_stats")]
            {
                let stats = interner.stats();
                interner_stats.lock().unwrap().clone_from(&stats);
            }

            bar.inc(1);
        }

        bar.finish();

        println!(
            "Early quits: no faces: {}, empty faces: {}, empty batch: {}",
            early_quit_no_faces.load(std::sync::atomic::Ordering::SeqCst),
            early_quit_empty_faces.load(std::sync::atomic::Ordering::SeqCst),
            early_quit_empty_batch.load(std::sync::atomic::Ordering::SeqCst)
        );
        println!(
            "Processed chunks: {}",
            processed_chunks.load(std::sync::atomic::Ordering::SeqCst)
        );

        handle.join().unwrap();
    }

    pub fn simple_voxelize(&mut self) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::simple_voxelize");

        let voxels_per_axis = self.model.voxels_per_axis(Lod::new(0));
        let voxel_size: f64 = 1.0 / voxels_per_axis as f64;
        let inv_voxel_size: f64 = 1.0 / voxel_size;

        let now = Instant::now();

        let mesh_min = self.mesh.aabb.0;

        let interner = self.model.get_interner();
        let mut interner = interner.write();

        for face in self.mesh.faces.iter() {
            for vertex_index in [face.x, face.y, face.z] {
                let vertex = self.mesh.vertices[(vertex_index - 1) as usize] - mesh_min;
                let voxel = (vertex * inv_voxel_size).floor().as_ivec3();
                let local_voxel = voxel % voxels_per_axis as i32;

                let chunk_position =
                    convert_voxel_world_to_chunk_position(voxel, self.model.chunk_world_size);
                let chunk = &mut self.model.get_or_create_chunk(chunk_position);

                chunk.set(&mut interner, local_voxel, 1);
            }
        }

        println!("Simple voxelize took: {:?}", now.elapsed());
    }

    pub fn voxelize(&mut self) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("Voxelizer::voxelize");

        println!("Voxelize started");

        let face_to_chunk_map_time = Instant::now();

        println!("Building face-to-chunk mapping");

        // Build face-to-chunk mapping
        let chunk_face_map = self.build_face_to_chunk_map();

        let face_to_chunk_map_time = face_to_chunk_map_time.elapsed();

        let voxelize_time = Instant::now();

        println!("Voxelizing mesh");

        self.voxelize_mesh(chunk_face_map);

        let voxelize_time = voxelize_time.elapsed();

        let empty_chunks = self
            .model
            .chunks
            .iter()
            .filter(|(_, chunk)| chunk.is_empty())
            .count();

        let total = face_to_chunk_map_time + voxelize_time;

        #[cfg(feature = "memory_stats")]
        {
            let interner = self.model.interner_stats();
            println!("Interner stats: {interner:#?}");
        }

        println!(
            "Done, {} chunks, empty: {empty_chunks}, face-to-chunk: {face_to_chunk_map_time:?}, voxelized: {voxelize_time:?}, total: {total:?}",
            self.model.chunks.len(),
        );
    }
}
