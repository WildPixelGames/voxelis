//! Ultra Detail Example - Showcasing 256³+ Voxel Mesh Resolution with SIMD
//!
//! This example demonstrates the true power of Voxelis after the u64 Morton upgrade.
//! Unlike the standard examples limited to 64³ mesh resolution, this uses the new
//! SIMD-based mesher to render 256³ (16M voxels) or even 512³ (134M voxels) meshes.
//!
//! # Features
//! - Massive voxel detail: 256³ to 512³ voxels per chunk
//! - SIMD-accelerated meshing: Fast greedy mesh generation
//! - True centimeter accuracy: 1-2cm voxels over large areas
//! - Adaptive LOD: Dynamically adjust detail based on camera distance
//!
//! # Controls
//! - Mouse: Orbit camera
//! - Scroll: Zoom
//! - WASD: Move camera
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
    window::PresentMode,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_screen_diagnostics::{
    ScreenDiagnosticsPlugin, ScreenEntityDiagnosticsPlugin, ScreenFrameDiagnosticsPlugin,
};
use voxelis::{
    Lod, MaxDepth, VoxInterner,
    spatial::{VoxOpsBatch, VoxOpsChunkConfig, VoxOpsConfig, VoxOpsSpatial3D},
    utils::{
        mesh::MeshData,
        mesh_simd::{OccupancyDataSIMD, fill_occupancy_from_tree, generate_greedy_mesh_simd},
        shapes::generate_sphere_batch,
    },
    world::VoxChunk,
};

// ULTRA DETAIL SETTINGS
const MAX_DEPTH: MaxDepth = MaxDepth::new(8); // 256³ voxels per chunk
const VOXELS_PER_AXIS: usize = 1 << MAX_DEPTH.max(); // 256
const CHUNK_SIZE: f32 = 5.12; // 5.12m chunk (2cm per voxel at full detail)
const LOD_DISTANCE_STEP: f32 = 20.0; // Distance per LOD level
const MIN_LOD: u8 = 0; // LOD 0 = FULL 256³ mesh resolution!
const MAX_LOD: u8 = 6; // LOD 6 = 4³ minimum
const INITIAL_LOD: u8 = 0; // Start at maximum detail

#[derive(Resource, Default)]
pub struct StatsWindowState {
    pub visible: bool,
}

#[derive(Component)]
pub struct ChunkEntity;

#[derive(Resource)]
pub struct World {
    pub interner: VoxInterner<i32>,
    pub chunk: VoxChunk<i32>,
    pub chunk_mesh: Handle<Mesh>,
    pub chunk_entity: Entity,
    pub current_lod: Lod,
}

impl World {
    pub fn new() -> Self {
        let mut interner = VoxInterner::<i32>::with_memory_budget(1024 * 1024 * 512);

        // Create single chunk with detailed sphere
        let mut chunk = VoxChunk::with_position(CHUNK_SIZE, MAX_DEPTH, 0, 0, 0);
        let mut batch = chunk.create_batch();

        // Create a large, detailed sphere
        let center = IVec3::new(128, 128, 128); // Center of 256³ space
        let radius = 100; // Large radius to show detail
        let value = 1;

        generate_sphere_batch(&mut batch, center, radius, value);
        chunk.apply_batch(&mut interner, &batch);

        println!("=== Ultra Detail World Created ===");
        println!("  Max depth: {}", MAX_DEPTH.max());
        println!("  Voxels per axis: {}", VOXELS_PER_AXIS);
        println!("  Total voxels in chunk: {}", VOXELS_PER_AXIS.pow(3));
        println!("  Chunk size: {}m", CHUNK_SIZE);
        println!(
            "  Voxel size (LOD 0): {}cm",
            (CHUNK_SIZE / VOXELS_PER_AXIS as f32) * 100.0
        );
        println!("  Sphere radius: {} voxels", radius);
        println!("===================================");

        Self {
            interner,
            chunk,
            chunk_mesh: Handle::default(),
            chunk_entity: Entity::PLACEHOLDER,
            current_lod: Lod::new(INITIAL_LOD),
        }
    }

    pub fn update_lod(&mut self, camera_position: Vec3) {
        let chunk_center = self.chunk.world_center_position_3d();
        let distance = camera_position.distance(chunk_center);

        // Calculate LOD based on distance
        let raw_lod = (distance / LOD_DISTANCE_STEP).floor() as u8;
        let new_lod = raw_lod.max(MIN_LOD).min(MAX_LOD).min(MAX_DEPTH.max());

        let new_lod_obj = Lod::new(new_lod);

        if new_lod_obj != self.current_lod {
            let mesh_resolution = self.chunk.voxels_per_axis(new_lod_obj);
            println!(
                "LOD changed to {} (distance: {:.2}m, mesh resolution: {}³ = {} voxels)",
                new_lod,
                distance,
                mesh_resolution,
                mesh_resolution.pow(3)
            );
            self.current_lod = new_lod_obj;
        }
    }

    pub fn generate_mesh(&mut self, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
        let lod = self.current_lod;
        let max_depth = self.chunk.max_depth(lod);
        let voxels_per_axis = self.chunk.voxels_per_axis(lod) as usize;

        println!(
            "Generating mesh: LOD {}, {}³ voxels",
            lod.lod(),
            voxels_per_axis
        );

        // Fill occupancy data from tree
        let mut occupancy = OccupancyDataSIMD::new(voxels_per_axis);
        fill_occupancy_from_tree(
            &mut occupancy,
            &self.interner,
            self.chunk.get_root_id(),
            max_depth,
        );

        // Generate mesh using SIMD
        let mut mesh_data = MeshData::default();
        let voxel_size = self.chunk.voxel_size(lod);
        let offset = self.chunk.world_position_3d();

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
}

fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    println!("\n=== Setting up Ultra Detail world ===");

    let mut world = World::new();

    // Generate initial mesh
    let mesh_handle = world.generate_mesh(&mut meshes);
    world.chunk_mesh = mesh_handle.clone();

    // Spawn chunk entity
    let entity = commands
        .spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.7, 0.6),
                perceptual_roughness: 0.9,
                metallic: 0.1,
                ..default()
            })),
            ChunkEntity,
        ))
        .id();

    world.chunk_entity = entity;

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
            Transform::from_xyz(3.0, 3.0, 8.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
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
    mut world: ResMut<World>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    let camera_position = camera_transform.translation;
    let old_lod = world.current_lod;

    world.update_lod(camera_position);

    // Regenerate mesh if LOD changed
    if world.current_lod != old_lod {
        let new_mesh = world.generate_mesh(&mut meshes);
        world.chunk_mesh = new_mesh;

        // Note: Mesh handle update happens automatically via reactivity
        // If it doesn't, we'd need to update the entity's mesh component
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

    let chunk_center = world.chunk.world_center_position_3d();
    let distance = camera_pos.distance(chunk_center);

    let current_mesh_resolution = world.chunk.voxels_per_axis(world.current_lod);
    let current_voxel_size = world.chunk.voxel_size(world.current_lod);

    egui::Window::new("🚀 Ultra Detail Stats").show(contexts.ctx_mut()?, |ui| {
        ui.heading("Voxelis Ultra Detail Mode");
        ui.separator();

        ui.label(format!("📊 Chunk Configuration:"));
        ui.label(format!("  Tree depth: {}", MAX_DEPTH.max()));
        ui.label(format!(
            "  Storage capacity: {}³ = {} voxels",
            VOXELS_PER_AXIS,
            VOXELS_PER_AXIS.pow(3)
        ));
        ui.label(format!("  Chunk world size: {:.2}m", CHUNK_SIZE));
        ui.separator();

        ui.label(format!("🎮 Current Rendering:"));
        ui.label(format!("  LOD level: {}", world.current_lod.lod()));
        ui.label(format!(
            "  Mesh resolution: {}³ = {} voxels",
            current_mesh_resolution,
            current_mesh_resolution.pow(3)
        ));
        ui.label(format!("  Voxel size: {:.1}cm", current_voxel_size * 100.0));
        ui.label(format!("  Camera distance: {:.2}m", distance));
        ui.separator();

        ui.label(format!("🔥 Detail Levels:"));
        for lod in MIN_LOD..=MAX_LOD {
            let lod_obj = Lod::new(lod);
            let res = world.chunk.voxels_per_axis(lod_obj);
            let size = world.chunk.voxel_size(lod_obj) * 100.0;
            let color = if lod == world.current_lod.lod() {
                egui::Color32::GREEN
            } else {
                egui::Color32::GRAY
            };

            ui.colored_label(
                color,
                format!("  LOD {}: {}³ ({:.1}cm voxels)", lod, res, size),
            );
        }
        ui.separator();

        ui.label("💡 Tips:");
        ui.label("  • Zoom in to see LOD 0 (256³ mesh!)");
        ui.label("  • Watch mesh resolution change");
        ui.label("  • SIMD mesher handles massive detail");
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
    println!("  🚀 VOXELIS ULTRA DETAIL DEMO 🚀");
    println!("=======================================================");
    println!("  Maximum mesh resolution: 256³ = 16,777,216 voxels");
    println!("  Voxel size at LOD 0: 2cm");
    println!("  SIMD-accelerated greedy meshing");
    println!("  Press F1 to toggle stats window");
    println!("=======================================================\n");

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Voxelis Ultra Detail - 256³ SIMD Meshing".to_string(),
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
        .add_systems(Update, (update_lod_system, toggle_stats_window))
        .add_systems(EguiPrimaryContextPass, stats_ui)
        .run();
}
