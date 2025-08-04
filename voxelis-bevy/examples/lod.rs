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
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use voxelis::{
    Lod, MaxDepth, VoxInterner,
    spatial::{VoxOpsMesh, VoxOpsSpatial3D},
    utils::mesh::MeshData,
    world::VoxChunk,
};

const MAX_DEPTH: MaxDepth = MaxDepth::new(6);
const VOXELS_PER_AXIS: usize = 1 << MAX_DEPTH.max();
const CHUNK_SIZE: f32 = 1.28;

#[derive(Resource, Default)]
pub struct StatsWindowState {
    pub visible: bool,
    pub show_details: bool,
}

#[derive(Resource)]
pub struct World {
    pub interner: VoxInterner<i32>,
    pub chunk: VoxChunk<i32>,
    pub mesh: Handle<Mesh>,
}

#[derive(Resource, Default)]
pub struct LodSettings {
    pub level: usize,
}

impl World {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let interner = VoxInterner::<i32>::with_memory_budget(1024 * 1024 * 256);

        let chunk = VoxChunk::with_position(CHUNK_SIZE, MAX_DEPTH, 0, 0, 0);

        Self {
            interner,
            chunk,
            mesh: Handle::default(),
        }
    }

    pub fn generate_mesh_arrays(&self, mesh_data: &mut MeshData, lod: Lod) {
        let offset = Vec3::new(0.0, 0.0, 0.0);
        self.chunk
            .generate_naive_mesh_arrays(&self.interner, mesh_data, offset, lod);
        println!(
            "vertices: {} normals: {} indices: {}",
            humanize_bytes::humanize_quantity!(mesh_data.vertices.len()),
            humanize_bytes::humanize_quantity!(mesh_data.normals.len()),
            humanize_bytes::humanize_quantity!(mesh_data.indices.len())
        );
    }

    pub fn generate_mesh(&mut self, meshes: &mut ResMut<Assets<Mesh>>, lod: Lod) {
        let mut mesh_data = MeshData::default();

        self.generate_mesh_arrays(&mut mesh_data, lod);

        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(Indices::U32(mesh_data.indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_data.vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_data.normals);

        self.mesh = meshes.add(mesh);
    }

    pub fn regenerate_mesh(&mut self, meshes: &mut ResMut<Assets<Mesh>>, lod: Lod) {
        let mut mesh_data = MeshData::default();

        self.generate_mesh_arrays(&mut mesh_data, lod);

        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(Indices::U32(mesh_data.indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_data.vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_data.normals);

        meshes.insert(&mut self.mesh, mesh);
    }
}

fn lod_ui(
    mut contexts: EguiContexts,
    mut lod_settings: ResMut<LodSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut world: ResMut<World>,
) {
    egui::Window::new("LOD Settings").show(contexts.ctx_mut(), |ui| {
        let response = ui.add(
            egui::Slider::new(&mut lod_settings.level, 0..=MAX_DEPTH.as_usize()).text("LOD Level"),
        );

        if response.changed() {
            world.regenerate_mesh(&mut meshes, Lod::new(lod_settings.level as u8));
            println!(
                "LOD Level: {} voxels per axis: {}",
                lod_settings.level,
                1 << (MAX_DEPTH.as_usize() - lod_settings.level)
            );
        }
    });
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
            // illuminance: 3_000., //light_consts::lux::OVERCAST_DAY,
            illuminance: 6_000., //light_consts::lux::OVERCAST_DAY,
            // illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
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

    world.generate_mesh(&mut meshes, Lod::new(0));

    let world_position = world.chunk.world_position_3d();
    let world_position = Vec3::new(world_position.x, 0.0, world_position.y);

    commands
        .spawn((
            Mesh3d(world.mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(world_position),
        ))
        .insert(Name::new("Chunk".to_string()));
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
                    title: "LODyssey".to_string(),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            TemporalAntiAliasPlugin,
            EguiPlugin,
            PanOrbitCameraPlugin,
            WireframePlugin,
            FrameTimeDiagnosticsPlugin,
            // ScreenDiagnosticsPlugin::default(),
            // ScreenFrameDiagnosticsPlugin,
            // ScreenEntityDiagnosticsPlugin,
        ))
        .insert_resource(ClearColor(Color::from(
            bevy::color::palettes::tailwind::BLUE_300,
        )))
        .insert_resource(DirectionalLightShadowMap { size: 8192 })
        .insert_resource(World::new())
        .init_resource::<LodSettings>()
        .add_systems(Startup, setup_world)
        .add_systems(Update, toggle_wireframe)
        .add_systems(Update, lod_ui)
        .run();
}
