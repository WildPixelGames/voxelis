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

const MAX_DEPTH: MaxDepth = MaxDepth::new(6);
const VOXELS_PER_AXIS: usize = 1 << MAX_DEPTH.max();
const CHUNK_SIZE: f32 = 1.28;
const GRID_SIZE: i32 = 5; // 5x5x5 grid of chunks
const LOD_DISTANCE_STEP: f32 = 5.0; // Distance per LOD level increase

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

        for x in -GRID_SIZE / 2..=GRID_SIZE / 2 {
            for y in -GRID_SIZE / 2..=GRID_SIZE / 2 {
                for z in -GRID_SIZE / 2..=GRID_SIZE / 2 {
                    let mut chunk = VoxChunk::with_position(CHUNK_SIZE, MAX_DEPTH, x, y, z);

                    let mut batch = chunk.create_batch();

                    let voxels_per_axis = chunk.voxels_per_axis(Lod::new(0));

                    let half_size = voxels_per_axis as i32 / 2;
                    let center = IVec3::new(half_size, half_size, half_size);
                    let radius = (voxels_per_axis / 2) as i32;
                    let value = 1;

                    generate_sphere_batch(&mut batch, center, radius, value);

                    chunk.apply_batch(&mut interner, &batch);

                    chunks.push(chunk);
                    chunk_meshes.push(Handle::default());
                    chunk_entities.push(Entity::PLACEHOLDER);
                    chunk_lods.push(Lod::new(0));
                }
            }
        }

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
            let new_lod_level = (distance / LOD_DISTANCE_STEP).floor() as usize;
            let new_lod_level = new_lod_level.min(MAX_DEPTH.as_usize());

            if new_lod_level != self.chunk_lods[i].lod() as usize {
                updates.push((i, new_lod_level, distance));
            }
        }
        for (i, new_lod_level, distance) in updates {
            self.chunk_lods[i] = Lod::new(new_lod_level as u8);
            self.regenerate_mesh(meshes, i, self.chunk_lods[i]);
            println!(
                "Chunk {} LOD changed to {} (distance: {:.2})",
                i, new_lod_level, distance
            );
        }
    }
}

fn lod_ui(mut contexts: EguiContexts, lod_settings: Res<LodSettings>, world: Res<World>) -> Result {
    egui::Window::new("Adaptive LOD Stats").show(contexts.ctx_mut()?, |ui| {
        ui.label(format!("Total chunks: {}", world.chunks.len()));
        ui.label(format!("Max LOD: {}", lod_settings.max_lod));
        ui.separator();
        for (i, lod) in world.chunk_lods.iter().enumerate() {
            ui.label(format!("Chunk {}: LOD {}", i, lod.lod() as usize));
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
            Transform::from_xyz(2.2716377, 1.2876732, 3.9676127)
                .looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
            TemporalJitter::default(),
            PanOrbitCamera::default(),
        ))
        .insert(Tonemapping::TonyMcMapface)
        .insert(Bloom::default())
        .insert(Skybox {
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            brightness: 1000.0,
            ..default()
        })
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
                    title: "Adaptive LOD".to_string(),
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
