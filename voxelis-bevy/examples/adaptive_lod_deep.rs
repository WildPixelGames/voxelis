use bevy::{
    core_pipeline::{
        Skybox,
        bloom::Bloom,
        experimental::taa::{TemporalAntiAliasPlugin, TemporalAntiAliasing},
        tonemapping::Tonemapping,
    },
    diagnostic::FrameTimeDiagnosticsPlugin,
    pbr::{
        CascadeShadowConfigBuilder, DirectionalLightShadowMap, ScreenSpaceAmbientOcclusion,
        ScreenSpaceAmbientOcclusionQualityLevel, VolumetricFog,
        wireframe::{WireframeConfig, WireframePlugin},
    },
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
    spatial::{VoxOpsBatch, VoxOpsConfig, VoxOpsMesh, VoxOpsSpatial3D},
    utils::{mesh::MeshData, shapes::generate_sphere_batch},
    world::VoxChunk,
};

const MAX_DEPTH: MaxDepth = MaxDepth::new(8); // 256³ voxels per chunk - TRUE SVO scaling!
const VOXELS_PER_AXIS: usize = 1 << MAX_DEPTH.max();
const CHUNK_SIZE: f32 = 25.6; // 25.6m chunk for 256³ voxels (10cm per voxel)
const GRID_SIZE: i32 = 0; // Single chunk only
const LOD_DISTANCE_STEP: f32 = 80.0; // Distance per LOD level increase
const MIN_LOD: u8 = 2; // Min LOD 2 ensures mesh never exceeds 64³ (current mesh limit)
const INITIAL_LOD: u8 = 2; // Start with LOD 2 (64³ mesh resolution)
const MAX_LOD: u8 = 6; // Max LOD 6 = 4³ mesh resolution minimum

#[derive(Resource, Default)]
pub struct StatsWindowState {
    pub visible: bool,
    pub show_details: bool,
}

#[derive(Component)]
pub struct ChunkEntity {
    pub chunk_index: usize,
}

#[derive(Resource)]
pub struct World {
    pub interner: VoxInterner<i32>,
    pub chunks: Vec<VoxChunk<i32>>,
    pub chunk_meshes: Vec<Handle<Mesh>>,
    pub chunk_entities: Vec<Entity>,
    pub chunk_lods: Vec<Lod>,
}

#[derive(Resource, Default)]
pub struct LodSettings {
    pub max_lod: usize,
}

impl World {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let mut interner = VoxInterner::<i32>::with_memory_budget(1024 * 1024 * 256);

        let mut chunks = Vec::new();
        let mut chunk_meshes = Vec::new();
        let mut chunk_entities = Vec::new();
        let mut chunk_lods = Vec::new();

        // Create a single chunk with a reasonably sized sphere
        let mut chunk = VoxChunk::with_position(CHUNK_SIZE, MAX_DEPTH, 0, 0, 0);
        let mut batch = chunk.create_batch();

        // Create a sphere that fits well within the 256³ voxel space
        let center = IVec3::new(128, 128, 128); // Center of 256³ space
        let radius = 80; // Large sphere for 256³ space - shows SVO compression benefits!
        let value = 1;

        generate_sphere_batch(&mut batch, center, radius, value);
        chunk.apply_batch(&mut interner, &batch);

        // Debug: Verify sphere generation
        println!("=== Sphere Generation Stats ===");
        println!("  Depth: {}", MAX_DEPTH.max());
        println!("  Voxels per axis: {}", 1 << MAX_DEPTH.max());
        println!("  Center: {:?}", center);
        println!("  Radius: {}", radius);
        println!(
            "  Expected voxels: ~{}",
            (4.0 / 3.0 * std::f32::consts::PI * (radius * radius * radius) as f32) as i32
        );
        println!("===============================");

        chunks.push(chunk);
        chunk_meshes.push(Handle::default());
        chunk_entities.push(Entity::PLACEHOLDER);
        chunk_lods.push(Lod::new(INITIAL_LOD)); // Start with LOD 1 (128³ mesh resolution)

        Self {
            interner,
            chunks,
            chunk_meshes,
            chunk_entities,
            chunk_lods,
        }
    }

    pub fn generate_mesh_arrays_from_chunk(
        &self,
        mesh_data: &mut MeshData,
        chunk: &VoxChunk<i32>,
        lod: Lod,
    ) {
        let offset = Vec3::new(0.0, 0.0, 0.0);
        chunk.generate_greedy_mesh_arrays(&self.interner, mesh_data, offset, lod);
    }

    pub fn generate_mesh(
        &mut self,
        meshes: &mut ResMut<Assets<Mesh>>,
        chunk_index: usize,
        lod: Lod,
    ) {
        let mut mesh_data = MeshData::default();

        self.generate_mesh_arrays_from_chunk(&mut mesh_data, &self.chunks[chunk_index], lod);

        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(Indices::U32(mesh_data.indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_data.vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_data.normals);

        self.chunk_meshes[chunk_index] = meshes.add(mesh);
    }

    pub fn regenerate_mesh(
        &mut self,
        meshes: &mut ResMut<Assets<Mesh>>,
        chunk_index: usize,
        lod: Lod,
    ) {
        let mut mesh_data = MeshData::default();

        self.generate_mesh_arrays_from_chunk(&mut mesh_data, &self.chunks[chunk_index], lod);

        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(Indices::U32(mesh_data.indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_data.vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_data.normals);

        meshes.insert(&mut self.chunk_meshes[chunk_index], mesh);
    }

    pub fn update_lod(&mut self, camera_position: Vec3, meshes: &mut ResMut<Assets<Mesh>>) {
        let mut updates = Vec::new();
        for i in 0..self.chunks.len() {
            let chunk_center = self.chunks[i].world_center_position_3d();
            let distance = camera_position.distance(chunk_center);

            // Calculate LOD level based on distance
            let raw_lod_level = (distance / LOD_DISTANCE_STEP).floor() as usize;

            // Clamp LOD level to safe range
            // - Must be >= MIN_LOD (ensure mesh resolution <= 64³, current implementation limit)
            // - Must be <= MAX_LOD (prevent going to dangerously low mesh resolutions)
            // - Must be <= MAX_DEPTH (can't LOD beyond the tree depth)
            let new_lod_level = raw_lod_level
                .max(MIN_LOD as usize)
                .min(MAX_LOD as usize)
                .min(MAX_DEPTH.as_usize());

            // Calculate resulting mesh resolution
            let mesh_resolution = MAX_DEPTH.as_usize().saturating_sub(new_lod_level);
            let voxels_per_axis = 1 << mesh_resolution;

            // Safety check: ensure mesh resolution is reasonable (at least 4³)
            if voxels_per_axis < 4 {
                continue;
            }

            if new_lod_level != self.chunk_lods[i].lod() as usize {
                updates.push((i, new_lod_level, distance, voxels_per_axis));
            }
        }

        for (i, new_lod_level, distance, voxels_per_axis) in updates {
            self.chunk_lods[i] = Lod::new(new_lod_level as u8);
            self.regenerate_mesh(meshes, i, self.chunk_lods[i]);
            println!(
                "Chunk {} LOD changed to {} (distance: {:.2}, mesh resolution: {}³)",
                i, new_lod_level, distance, voxels_per_axis
            );
        }
    }
}

fn lod_ui(mut contexts: EguiContexts, lod_settings: Res<LodSettings>, world: Res<World>) -> Result {
    egui::Window::new("Deep Adaptive LOD Stats").show(contexts.ctx_mut()?, |ui| {
        ui.label(format!("Total chunks: {}", world.chunks.len()));
        ui.label(format!("Max depth: {}", MAX_DEPTH.max()));
        ui.label(format!(
            "Voxels per chunk: {}³ = {} voxels",
            1 << MAX_DEPTH.max(),
            (1u64 << (MAX_DEPTH.max() * 3))
        ));
        ui.label(format!("Voxel size: 20cm (planet-scale ready!)"));
        ui.label(format!("Chunk world size: {:.1}m", CHUNK_SIZE));
        ui.label(format!("Max LOD: {}", lod_settings.max_lod));
        ui.separator();

        let mut lod_counts = std::collections::HashMap::new();
        for lod in &world.chunk_lods {
            *lod_counts.entry(lod.lod()).or_insert(0) += 1;
        }

        ui.label("LOD Distribution:");
        for (lod, count) in lod_counts {
            let resolution = 1 << (MAX_DEPTH.max().saturating_sub(lod));
            ui.label(format!(
                "  LOD {}: {} chunks ({}³ resolution)",
                lod, count, resolution
            ));
        }
    });

    Ok(())
}

fn setup_world(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut world: ResMut<World>,
) {
    let mut cascade_shadow_config_builder = CascadeShadowConfigBuilder::default();
    cascade_shadow_config_builder.first_cascade_far_bound = 50.0;
    cascade_shadow_config_builder.minimum_distance = 0.1;
    cascade_shadow_config_builder.maximum_distance = 100_000.0;

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            illuminance: 6_000.,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(-0.15, -0.05, 0.25), Vec3::Y),
    ));

    commands
        .spawn((
            Camera3d::default(),
            Camera {
                hdr: true,
                ..default()
            },
            Msaa::Off,
            Transform::from_xyz(10.0, 8.0, 15.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
            TemporalJitter::default(),
            PanOrbitCamera::default(),
        ))
        .insert(Tonemapping::TonyMcMapface)
        .insert(Bloom::default())
        .insert(VolumetricFog {
            ambient_intensity: 0.1,
            ..default()
        })
        .insert(ScreenSpaceAmbientOcclusion {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
            ..default()
        })
        .insert(TemporalAntiAliasing::default());

    let material = materials.add(StandardMaterial {
        base_color: Color::from(bevy::color::palettes::tailwind::AMBER_700),
        perceptual_roughness: 1.0,
        reflectance: 0.0,
        thickness: 0.1,
        ior: 1.46,
        ..default()
    });

    for i in 0..world.chunks.len() {
        let lod = world.chunk_lods[i];
        let world_position = world.chunks[i].world_position_3d();
        world.generate_mesh(&mut meshes, i, lod);

        let entity = commands
            .spawn((
                Mesh3d(world.chunk_meshes[i].clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(world_position),
                ChunkEntity { chunk_index: i },
            ))
            .insert(Name::new(format!("Chunk_{}", i)))
            .id();

        world.chunk_entities[i] = entity;
    }
}

fn update_lod_system(
    camera_query: Query<&Transform, With<Camera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut world: ResMut<World>,
) {
    if let Ok(camera_transform) = camera_query.single() {
        let camera_position = camera_transform.translation;
        world.update_lod(camera_position, &mut meshes);
    }
}

fn toggle_wireframe(
    mut wireframe_config: ResMut<WireframeConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        wireframe_config.global = !wireframe_config.global;
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Adaptive LOD Deep - 256³ Voxels, 128³ Mesh (u128 unlocked!)"
                        .to_string(),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            TemporalAntiAliasPlugin,
            EguiPlugin::default(),
            PanOrbitCameraPlugin,
            WireframePlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            ScreenDiagnosticsPlugin::default(),
            ScreenFrameDiagnosticsPlugin,
            ScreenEntityDiagnosticsPlugin,
        ))
        .insert_resource(ClearColor(Color::from(
            bevy::color::palettes::tailwind::BLUE_300,
        )))
        .insert_resource(DirectionalLightShadowMap { size: 8192 })
        .insert_resource(World::new())
        .init_resource::<LodSettings>()
        .add_systems(Startup, setup_world)
        .add_systems(Update, (toggle_wireframe, update_lod_system))
        .add_systems(EguiPrimaryContextPass, lod_ui)
        .run();
}
