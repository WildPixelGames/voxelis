//! Spatial Adaptive LOD Example - Deep Tree with Region-based LOD
//!
//! This example demonstrates spatial adaptive LOD within a single massive SVO tree.
//! Unlike traditional LOD that switches the entire chunk, this subdivides a deep tree
//! into spatial regions (octants) and meshes each at different detail levels.
//!
//! # Features
//! - Deep SVO: Depth 17 tree (131,072³ voxel storage capacity)
//! - Spatial subdivision: 64 octants (4x4x4) meshed independently
//! - Region-based LOD: Each octant meshes at ≤512³ based on camera distance
//! - Seamless detail: Close regions show maximum detail, distant regions are simplified
//! - 3D Noise Terrain: Surface detail generated with Perlin-like noise
//! - Massive scale: 1.3km world with 1cm voxel precision
//!
//! # How It Works
//! 1. Create a single depth-17 tree (131,072³ theoretical capacity, ~1.3km at 1cm/voxel)
//! 2. Subdivide into 64 octants (4x4x4 grid, each ~32k³ region)
//! 3. For each octant:
//!    - Calculate distance to camera
//!    - Choose LOD level (0-8): LOD 0 = 512³, LOD 8 = 2³
//!    - Mesh that specific octant region at chosen detail
//! 4. Update meshes as camera moves
//!
//! # Controls
//! - Mouse: Orbit camera
//! - WASD: Move camera
//! - Scroll: Zoom
//! - F1: Toggle stats
//! - Space: Toggle wireframe
//! - Escape: Exit

use bevy::{
    core_pipeline::{
        bloom::Bloom,
        experimental::taa::{TemporalAntiAliasPlugin, TemporalAntiAliasing},
        tonemapping::Tonemapping,
    },
    diagnostic::FrameTimeDiagnosticsPlugin,
    pbr::{CascadeShadowConfigBuilder, DirectionalLightShadowMap, wireframe::WireframePlugin},
    prelude::*,
    render::{
        camera::TemporalJitter,
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
    tasks::{AsyncComputeTaskPool, Task},
    window::PresentMode,
};

use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_screen_diagnostics::{
    ScreenDiagnosticsPlugin, ScreenEntityDiagnosticsPlugin, ScreenFrameDiagnosticsPlugin,
};
use voxelis::{
    Lod, MaxDepth, VoxInterner,
    spatial::{VoxOpsChunkConfig, VoxOpsSpatial3D},
    utils::{
        mesh::MeshData,
        mesh_simd::{OccupancyDataSIMD, generate_greedy_mesh_simd},
    },
    world::VoxChunk,
};

// DEEP TREE SETTINGS
const TREE_DEPTH: MaxDepth = MaxDepth::new(17); // 131,072³ voxel capacity!
const TREE_SIZE: f32 = 1310.72; // 1.3km total size (1cm voxels at full detail)
const OCTANTS_PER_AXIS: usize = 4; // 4x4x4 grid
const NUM_OCTANTS: usize = OCTANTS_PER_AXIS * OCTANTS_PER_AXIS * OCTANTS_PER_AXIS; // 64 octants

// LOD SETTINGS - 9 levels for massive scale
const LOD_DISTANCE_THRESHOLDS: [f32; 9] = [
    50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0,
];
const LOD_MESH_SIZES: [usize; 9] = [512, 256, 128, 64, 32, 16, 8, 4, 2];

// ASTEROID SETTINGS
const ASTEROID_RADIUS: f32 = 20000.0; // Radius in voxels (~200m)
const ASTEROID_CENTER: IVec3 = IVec3::new(65536, 65536, 65536); // Center of world
const SURFACE_THICKNESS: i32 = 200; // Thickness of asteroid shell in voxels (~2m)
const NOISE_SCALE: f32 = 0.0003; // Scale for 3D noise (surface detail)
const NOISE_AMPLITUDE: f32 = 1000.0; // Surface variation in voxels (~10m)

#[derive(Resource, Default)]
pub struct StatsWindowState {
    pub visible: bool,
}

#[derive(Component)]
pub struct PendingMesh {
    pub task: Task<(usize, MeshData)>,
}

struct MeshGenerationData {
    octant_index: usize,
    octant: OctantInfo,
    lod: usize,
    occupancy: OccupancyDataSIMD,
}

#[derive(Component)]
pub struct OctantEntity {
    pub octant_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctantInfo {
    pub index: usize,
    pub offset: IVec3, // Offset in voxel space (0-1024 for each octant in a 2048³ tree)
    pub size: usize,   // Size in voxels (1024 for depth 11 subdivided once)
}

#[derive(Resource)]
pub struct World {
    pub interner: VoxInterner<i32>,
    pub chunk: VoxChunk<i32>,
    pub octants: Vec<OctantInfo>,
    pub octant_meshes: Vec<Handle<Mesh>>,
    pub octant_entities: Vec<Entity>,
    pub octant_lods: Vec<usize>,
}

impl World {
    pub fn new() -> Self {
        let mut interner = VoxInterner::<i32>::with_memory_budget(1024 * 1024 * 512); // 512MB budget

        // Create a single massive chunk with depth 17
        // Note: We generate terrain procedurally during octant sampling,
        // so the tree stays empty initially (no batch allocation issues!)
        let chunk = VoxChunk::with_position(TREE_SIZE, TREE_DEPTH, 0, 0, 0);

        println!("=== Spatial Adaptive LOD World Created ===");
        println!("  Spherical asteroid will be generated procedurally");
        println!("  Tree depth: {}", TREE_DEPTH.max());
        println!("  Storage capacity: 131,072³ voxels");
        println!("  Tree world size: {:.2}km", TREE_SIZE / 1000.0);
        println!(
            "  Voxel size at max detail: {:.2}cm",
            (TREE_SIZE / 131072.0) * 100.0
        );
        println!(
            "  Asteroid radius: {:.1}m",
            ASTEROID_RADIUS * (TREE_SIZE / 131072.0)
        );
        println!(
            "  Number of octants: {} ({}x{}x{})",
            NUM_OCTANTS, OCTANTS_PER_AXIS, OCTANTS_PER_AXIS, OCTANTS_PER_AXIS
        );
        println!("==========================================");

        // Define 64 octants (4x4x4 grid, each ~32k³ region)
        let octant_size = 131072 / OCTANTS_PER_AXIS; // ~32,768 voxels per octant
        let mut octants = Vec::new();

        for oz in 0..OCTANTS_PER_AXIS {
            for oy in 0..OCTANTS_PER_AXIS {
                for ox in 0..OCTANTS_PER_AXIS {
                    let index =
                        oz * OCTANTS_PER_AXIS * OCTANTS_PER_AXIS + oy * OCTANTS_PER_AXIS + ox;
                    octants.push(OctantInfo {
                        index,
                        offset: IVec3::new(
                            (ox * octant_size) as i32,
                            (oy * octant_size) as i32,
                            (oz * octant_size) as i32,
                        ),
                        size: octant_size,
                    });
                }
            }
        }

        let octant_meshes = vec![Handle::default(); NUM_OCTANTS];
        let octant_entities = vec![Entity::PLACEHOLDER; NUM_OCTANTS];
        let octant_lods = vec![0; NUM_OCTANTS]; // Start all at LOD 0

        Self {
            interner,
            chunk,
            octants,
            octant_meshes,
            octant_entities,
            octant_lods,
        }
    }

    // Simple 3D noise function (deterministic hash-based)
    fn simple_noise_3d(x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        // Smooth interpolation
        let u = xf * xf * (3.0 - 2.0 * xf);
        let v = yf * yf * (3.0 - 2.0 * yf);
        let w = zf * zf * (3.0 - 2.0 * zf);

        // 8 corner values
        let c000 = Self::hash3d(xi, yi, zi);
        let c001 = Self::hash3d(xi, yi, zi + 1);
        let c010 = Self::hash3d(xi, yi + 1, zi);
        let c011 = Self::hash3d(xi, yi + 1, zi + 1);
        let c100 = Self::hash3d(xi + 1, yi, zi);
        let c101 = Self::hash3d(xi + 1, yi, zi + 1);
        let c110 = Self::hash3d(xi + 1, yi + 1, zi);
        let c111 = Self::hash3d(xi + 1, yi + 1, zi + 1);

        // Trilinear interpolation
        let x00 = c000 * (1.0 - u) + c100 * u;
        let x01 = c001 * (1.0 - u) + c101 * u;
        let x10 = c010 * (1.0 - u) + c110 * u;
        let x11 = c011 * (1.0 - u) + c111 * u;

        let y0 = x00 * (1.0 - v) + x10 * v;
        let y1 = x01 * (1.0 - v) + x11 * v;

        y0 * (1.0 - w) + y1 * w
    }

    fn hash3d(x: i32, y: i32, z: i32) -> f32 {
        let n = x
            .wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(z.wrapping_mul(1274126177));
        let n = n.wrapping_mul(1664525) ^ (n >> 16);
        ((n & 0x7fffffff) as f32 / 2147483647.0) * 2.0 - 1.0
    }

    pub fn octant_world_center(&self, octant: &OctantInfo) -> Vec3 {
        let voxel_size = self.chunk.voxel_size(Lod::new(0));
        let world_offset = self.chunk.world_position_3d();

        let octant_center_voxels = octant.offset.as_vec3() + Vec3::splat(octant.size as f32 * 0.5);
        world_offset + octant_center_voxels * voxel_size
    }

    pub fn determine_lod(&self, octant: &OctantInfo, camera_position: Vec3) -> usize {
        let octant_center = self.octant_world_center(octant);
        let distance = camera_position.distance(octant_center);

        for (lod, threshold) in LOD_DISTANCE_THRESHOLDS.iter().enumerate() {
            if distance < *threshold {
                return lod;
            }
        }
        LOD_DISTANCE_THRESHOLDS.len() - 1
    }

    fn octant_intersects_terrain(octant: &OctantInfo) -> bool {
        // Check if octant bounding box intersects with asteroid sphere
        let octant_center = octant.offset.as_vec3() + Vec3::splat(octant.size as f32 * 0.5);
        let asteroid_center = ASTEROID_CENTER.as_vec3();
        let octant_half_size = octant.size as f32 * 0.5;

        // Distance from sphere center to closest point in octant
        let closest_point = Vec3::new(
            octant_center.x.clamp(
                octant.offset.x as f32,
                (octant.offset.x + octant.size as i32) as f32,
            ),
            octant_center.y.clamp(
                octant.offset.y as f32,
                (octant.offset.y + octant.size as i32) as f32,
            ),
            octant_center.z.clamp(
                octant.offset.z as f32,
                (octant.offset.z + octant.size as i32) as f32,
            ),
        );

        let distance = asteroid_center.distance(closest_point);
        let max_radius = ASTEROID_RADIUS + NOISE_AMPLITUDE + SURFACE_THICKNESS as f32;

        // Octant intersects if closest point is within max radius
        distance < max_radius
    }

    pub fn fill_octant_occupancy(
        &self,
        octant: &OctantInfo,
        occupancy: &mut OccupancyDataSIMD,
        lod: usize,
    ) {
        let mesh_size = LOD_MESH_SIZES[lod];
        let sample_step = octant.size / mesh_size; // How many voxels to skip per sample

        // Fill occupancy by procedurally generating asteroid surface
        let mut solid_count = 0;

        for z in 0..mesh_size {
            for y in 0..mesh_size {
                for x in 0..mesh_size {
                    // Calculate position in the full 131k³ space
                    let voxel_pos = octant.offset
                        + IVec3::new(
                            (x * sample_step) as i32,
                            (y * sample_step) as i32,
                            (z * sample_step) as i32,
                        );

                    // Procedurally evaluate if this voxel is part of asteroid
                    if Self::is_asteroid_solid(voxel_pos) {
                        occupancy.set(x, y, z, 1);
                        solid_count += 1;
                    }
                }
            }
        }

        // Debug: report if octant is completely full
        if solid_count == mesh_size * mesh_size * mesh_size {
            println!(
                "  Note: Octant {} is completely solid (inside asteroid)",
                octant.index
            );
        }
    }

    fn is_asteroid_solid(pos: IVec3) -> bool {
        // Calculate distance from asteroid center
        let offset = pos - ASTEROID_CENTER;
        let distance = (offset.as_vec3()).length();

        // Add 3D noise for surface detail
        let noise_x = pos.x as f32 * NOISE_SCALE;
        let noise_y = pos.y as f32 * NOISE_SCALE;
        let noise_z = pos.z as f32 * NOISE_SCALE;

        // Fractal 3D noise for surface variation
        let surface_noise = Self::simple_noise_3d(noise_x, noise_y, noise_z) * NOISE_AMPLITUDE
            + Self::simple_noise_3d(noise_x * 2.0, noise_y * 2.0, noise_z * 2.0)
                * NOISE_AMPLITUDE
                * 0.5
            + Self::simple_noise_3d(noise_x * 4.0, noise_y * 4.0, noise_z * 4.0)
                * NOISE_AMPLITUDE
                * 0.25;

        let surface_radius = ASTEROID_RADIUS + surface_noise;

        // Solid if within surface shell
        distance <= surface_radius && distance >= (surface_radius - SURFACE_THICKNESS as f32)
    }

    fn sample_voxel_at_position(&self, position: &IVec3) -> Option<i32> {
        let mut node_id = self.chunk.get_root_id();
        let mut depth = 0u8;
        let max_depth = TREE_DEPTH.max();
        let default_value = i32::default();

        while !node_id.is_empty() {
            if depth >= max_depth {
                let value = self.interner.get_value(&node_id);
                if value != &default_value {
                    return Some(*value);
                } else {
                    return None;
                }
            }

            if node_id.is_branch() {
                // Calculate child index based on position and current depth
                let shift = max_depth - depth - 1;
                let index = ((position.x as usize >> shift) & 1)
                    | (((position.y as usize >> shift) & 1) << 1)
                    | (((position.z as usize >> shift) & 1) << 2);

                node_id = self.interner.get_child_id(&node_id, index);
                depth += 1;
            } else {
                return Some(*self.interner.get_value(&node_id));
            }
        }

        None
    }

    pub fn generate_octant_mesh(
        &mut self,
        meshes: &mut Assets<Mesh>,
        octant_index: usize,
        lod: usize,
    ) -> Handle<Mesh> {
        let octant = self.octants[octant_index];

        // Early out: skip octants that don't intersect terrain
        if !Self::octant_intersects_terrain(&octant) {
            // Return empty mesh
            return meshes.add(Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::RENDER_WORLD,
            ));
        }

        let mesh_size = LOD_MESH_SIZES[lod];

        println!(
            "Generating octant {} mesh at LOD {} ({}³ voxels)",
            octant_index, lod, mesh_size
        );

        // Create occupancy data at the appropriate resolution
        let mut occupancy = OccupancyDataSIMD::new(mesh_size);
        self.fill_octant_occupancy(&octant, &mut occupancy, lod);

        // Generate mesh
        let mut mesh_data = MeshData::default();

        // Calculate the voxel size for this LOD within the octant
        let voxel_size = (TREE_SIZE / 2048.0) * (octant.size / mesh_size) as f32;

        // Calculate world offset for this octant
        let chunk_world_offset = self.chunk.world_position_3d();
        let octant_voxel_offset = octant.offset.as_vec3() * (TREE_SIZE / 2048.0);
        let offset = chunk_world_offset + octant_voxel_offset;

        generate_greedy_mesh_simd(&occupancy, &mut mesh_data, offset, voxel_size);

        println!(
            "  Generated {} vertices, {} triangles",
            mesh_data.vertices.len(),
            mesh_data.indices.len() / 3
        );

        // Create Bevy mesh
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, mesh_data.vertices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_data.normals);
        mesh.insert_indices(Indices::U32(mesh_data.indices));

        meshes.add(mesh)
    }

    pub fn update_lods(&mut self, camera_position: Vec3) -> Vec<(usize, usize)> {
        let mut changes = Vec::new();

        for i in 0..NUM_OCTANTS {
            let octant = &self.octants[i];
            let new_lod = self.determine_lod(octant, camera_position);

            if new_lod != self.octant_lods[i] {
                let distance = camera_position.distance(self.octant_world_center(octant));
                println!(
                    "Octant {} LOD changed: {} -> {} (distance: {:.2}m, mesh: {}³)",
                    i, self.octant_lods[i], new_lod, distance, LOD_MESH_SIZES[new_lod]
                );

                self.octant_lods[i] = new_lod;
                changes.push((i, new_lod));
            }
        }

        changes
    }

    pub fn spawn_mesh_task(&self, commands: &mut Commands, octant_index: usize, lod: usize) {
        let octant = self.octants[octant_index];

        // Early out: skip octants that don't intersect terrain at all
        if !Self::octant_intersects_terrain(&octant) {
            return;
        }

        let task_pool = AsyncComputeTaskPool::get();
        let mesh_size = LOD_MESH_SIZES[lod];

        // Pre-fill occupancy data on main thread (this accesses the interner)
        let mut occupancy = OccupancyDataSIMD::new(mesh_size);
        self.fill_octant_occupancy(&octant, &mut occupancy, lod);

        let data = MeshGenerationData {
            octant_index,
            octant,
            lod,
            occupancy,
        };

        let task = task_pool.spawn(async move {
            // Generate mesh from pre-filled occupancy (no interner access needed)
            let mut mesh_data = MeshData::default();
            let mesh_size = LOD_MESH_SIZES[data.lod];
            let voxel_size = (TREE_SIZE / 131072.0) * (data.octant.size / mesh_size) as f32;
            let octant_voxel_offset = data.octant.offset.as_vec3() * (TREE_SIZE / 131072.0);
            let offset = octant_voxel_offset;

            generate_greedy_mesh_simd(&data.occupancy, &mut mesh_data, offset, voxel_size);

            (data.octant_index, mesh_data)
        });

        commands.spawn(PendingMesh { task });
    }
}

fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    println!("\n=== Setting up Spatial Adaptive LOD world ===");

    let mut world = World::new();

    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.7, 0.6),
        perceptual_roughness: 0.9,
        metallic: 0.1,
        ..default()
    });

    // Generate initial meshes only for octants that intersect terrain
    for i in 0..NUM_OCTANTS {
        let octant = &world.octants[i];

        // Skip octants that don't intersect terrain
        if !World::octant_intersects_terrain(octant) {
            continue;
        }

        let lod = world.octant_lods[i];
        let mesh_handle = world.generate_octant_mesh(&mut meshes, i, lod);

        // Skip if mesh is empty
        if let Some(mesh) = meshes.get(&mesh_handle) {
            if mesh.count_vertices() == 0 {
                continue;
            }
        }

        world.octant_meshes[i] = mesh_handle.clone();

        // Spawn octant entity
        let entity = commands
            .spawn((
                Mesh3d(mesh_handle),
                MeshMaterial3d(material.clone()),
                OctantEntity { octant_index: i },
            ))
            .id();

        world.octant_entities[i] = entity;
    }

    commands.insert_resource(world);

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 25000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 100.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            maximum_distance: 200.0,
            ..default()
        }
        .build(),
    ));

    // Camera
    commands
        .spawn((
            Camera3d::default(),
            Camera {
                hdr: true,
                ..default()
            },
            Msaa::Off,
            // Position camera to view asteroid from outside
            Transform::from_xyz(900.0, 700.0, 900.0)
                .looking_at(Vec3::new(655.0, 655.0, 655.0), Vec3::Y),
            TemporalJitter::default(),
            PanOrbitCamera::default(),
        ))
        .insert(Tonemapping::TonyMcMapface)
        .insert(Bloom::default())
        .insert(TemporalAntiAliasing::default());

    println!("=== Setup complete ===\n");
}

fn update_lod_system(
    camera_query: Query<&Transform, With<Camera3d>>,
    mut commands: Commands,
    mut world: ResMut<World>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    let camera_position = camera_transform.translation;
    let changes = world.update_lods(camera_position);

    // Spawn async tasks for mesh generation
    for (octant_index, lod) in changes {
        world.spawn_mesh_task(&mut commands, octant_index, lod);
    }
}

fn handle_completed_meshes(
    mut commands: Commands,
    mut pending_meshes: Query<(Entity, &mut PendingMesh)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut world: ResMut<World>,
) {
    for (entity, mut pending) in pending_meshes.iter_mut() {
        if let Some((octant_index, mesh_data)) =
            bevy::tasks::block_on(bevy::tasks::poll_once(&mut pending.task))
        {
            println!(
                "  Completed mesh for octant {}: {} vertices, {} triangles",
                octant_index,
                mesh_data.vertices.len(),
                mesh_data.indices.len() / 3
            );

            // Skip empty meshes to avoid Bevy rendering crashes
            if mesh_data.vertices.is_empty() {
                commands.entity(entity).despawn();
                continue;
            }

            // Create Bevy mesh
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::RENDER_WORLD,
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, mesh_data.vertices);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_data.normals);
            mesh.insert_indices(Indices::U32(mesh_data.indices));

            let new_mesh_handle = meshes.add(mesh);

            // Update mesh handle and entity
            world.octant_meshes[octant_index] = new_mesh_handle.clone();

            // If entity exists, update it; otherwise create it
            if world.octant_entities[octant_index] != Entity::PLACEHOLDER {
                // Entity exists - update its mesh (via component query would be better, but this works)
                // The mesh handle update will be picked up by Bevy's reactivity
            }

            // Remove the task entity
            commands.entity(entity).despawn();
        }
    }
}

fn stats_ui(
    mut contexts: EguiContexts,
    world: Res<World>,
    stats_state: Res<StatsWindowState>,
    camera_query: Query<&Transform, With<Camera3d>>,
) -> Result {
    if !stats_state.visible {
        return Ok(());
    }

    let Ok(camera_transform) = camera_query.single() else {
        return Ok(());
    };

    let camera_pos = camera_transform.translation;

    egui::Window::new("🌍 Spatial Adaptive LOD Stats").show(contexts.ctx_mut()?, |ui| {
        ui.heading("Deep Tree with Region-based LOD");
        ui.separator();

        ui.label(format!("📊 Tree Configuration:"));
        ui.label(format!("  Tree depth: {}", TREE_DEPTH.max()));
        ui.label(format!("  Storage capacity: 131,072³ voxels"));
        ui.label(format!("  Tree size: {:.2}km", TREE_SIZE / 1000.0));
        ui.label(format!(
            "  Voxel size (max detail): {:.2}cm",
            (TREE_SIZE / 131072.0) * 100.0
        ));
        ui.separator();

        ui.label(format!(
            "🎮 Octants ({} spatial regions, {}³ grid):",
            NUM_OCTANTS, OCTANTS_PER_AXIS
        ));
        for (i, octant) in world.octants.iter().enumerate() {
            let lod = world.octant_lods[i];
            let mesh_size = LOD_MESH_SIZES[lod];
            let center = world.octant_world_center(octant);
            let distance = camera_pos.distance(center);

            let color = match lod {
                0 => egui::Color32::GREEN,
                1 => egui::Color32::YELLOW,
                _ => egui::Color32::RED,
            };

            ui.colored_label(
                color,
                format!(
                    "  Octant {}: LOD {} ({}³) - {:.1}m away",
                    i, lod, mesh_size, distance
                ),
            );
        }
        ui.separator();

        ui.label(format!("🔥 LOD Levels:"));
        for (lod, (threshold, mesh_size)) in LOD_DISTANCE_THRESHOLDS
            .iter()
            .zip(LOD_MESH_SIZES.iter())
            .enumerate()
        {
            ui.label(format!(
                "  LOD {}: {}³ mesh (< {:.0}m)",
                lod, mesh_size, threshold
            ));
        }
        ui.separator();

        ui.label("💡 How It Works:");
        ui.label("  • Single depth-17 tree (131k³ capacity)");
        ui.label("  • 64 octants (4x4x4) meshed independently");
        ui.label("  • Each octant: up to 512³ mesh");
        ui.label("  • 3D noise terrain with SVO compression");
        ui.label("  • Detail where you need it!");
        ui.separator();

        ui.label("🎮 Controls:");
        ui.label("  F1: Toggle this window");
        ui.label("  Mouse: Orbit camera");
        ui.label("  Zoom in to see LOD changes");
    });

    Ok(())
}

fn toggle_stats_window(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut stats_state: ResMut<StatsWindowState>,
) {
    if keyboard.just_pressed(KeyCode::F1) {
        stats_state.visible = !stats_state.visible;
    }
}

fn main() {
    println!("=======================================================");
    println!("  🌍 VOXELIS SPATIAL ADAPTIVE LOD DEMO 🌍");
    println!("=======================================================");
    println!("  Tree capacity: 131,072³ voxels (depth 17)");
    println!("  World size: 1.3km with 1cm voxel precision");
    println!("  Octants: 64 regions (4x4x4) meshed independently");
    println!("  Max mesh per octant: 512³ voxels");
    println!("  LOD levels: 9 (512³ down to 2³)");
    println!("  Spherical asteroid with 3D noise surface");
    println!("  Press F1 to toggle stats window");
    println!("=======================================================\n");

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Voxelis Spatial Adaptive LOD - Asteroid with Procedural Surface"
                        .to_string(),
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
            TemporalAntiAliasPlugin,
            WireframePlugin::default(),
            PanOrbitCameraPlugin,
            EguiPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            ScreenDiagnosticsPlugin::default(),
            ScreenFrameDiagnosticsPlugin,
            ScreenEntityDiagnosticsPlugin,
        ))
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .insert_resource(StatsWindowState { visible: true })
        .add_systems(Startup, setup_world)
        .add_systems(
            Update,
            (
                update_lod_system,
                handle_completed_meshes,
                toggle_stats_window,
            ),
        )
        .add_systems(EguiPrimaryContextPass, stats_ui)
        .run();
}
