use std::path::Path;
use std::time::Instant;

use bevy::color::palettes;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::core_pipeline::Skybox;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::pbr::{
    ScreenSpaceAmbientOcclusionBundle, ScreenSpaceAmbientOcclusionQualityLevel,
    ScreenSpaceAmbientOcclusionSettings, VolumetricFogSettings,
};
use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*, window::PresentMode};
use bevy_egui::EguiPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_screen_diagnostics::{
    ScreenDiagnosticsPlugin, ScreenEntityDiagnosticsPlugin, ScreenFrameDiagnosticsPlugin,
};
use rayon::prelude::*;

use voxelis::io::import::import_model_from_vtm;
use voxelis::model::Model;

struct GamePlugin;

#[derive(Component)]
struct Chunk;

#[derive(Resource)]
pub struct ModelResource(pub Model);

#[derive(Eq, PartialEq)]
pub enum MaterialType {
    Checkered,
    Gradient,
}

#[derive(Resource)]
pub struct ModelSettings {
    pub material_type: MaterialType,
}

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        app.add_systems(Update, toggle_wireframe);
    }
}

fn generate_chunk_color_gradient(index: usize, total: usize) -> Color {
    // Normalize the index to generate distinct hues
    let hue = (index as f32 / total as f32) * 360.0;

    // Convert HSL to RGB
    let rgb = hsl_to_rgb(hue, 0.7, 0.5); // Saturation: 0.7, Lightness: 0.5
    Color::srgba(rgb.0, rgb.1, rgb.2, 1.0)
}

fn hsl_to_rgb(hue: f32, saturation: f32, lightness: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let m = lightness - c / 2.0;

    let (r, g, b) = match hue {
        0.0..=60.0 => (c, x, 0.0),
        60.0..=120.0 => (x, c, 0.0),
        120.0..=180.0 => (0.0, c, x),
        180.0..=240.0 => (0.0, x, c),
        240.0..=300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r + m, g + m, b + m)
}

fn generate_chunk_color_checkered(x: i32, y: i32, z: i32) -> Color {
    // Determine if the chunk should be black based on the sum of its coordinates
    let is_black = (x + y + z) % 2 == 0;

    if is_black {
        Color::from(palettes::css::SILVER)
    } else {
        Color::WHITE
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut model: ResMut<ModelResource>,
    model_settings: Res<ModelSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let model = &mut model.0;

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            // illuminance: 3_000., //light_consts::lux::OVERCAST_DAY,
            illuminance: 6_000., //light_consts::lux::OVERCAST_DAY,
            // illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_at(Vec3::new(-0.15, -0.05, 0.25), Vec3::Y),
        ..default()
    });

    commands
        .spawn((
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                // transform: Transform::from_xyz(0.0, 7., 14.0)
                transform: Transform::from_xyz(2.2716377, 1.2876732, 3.9676127)
                    // transform: Transform::from_xyz(-1.9573995, 1.9533201, -1.9587312)
                    .looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
                ..default()
            },
            PanOrbitCamera::default(),
        ))
        .insert(Tonemapping::TonyMcMapface)
        .insert(BloomSettings::default())
        .insert(Skybox {
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            brightness: 1000.0,
        })
        .insert(VolumetricFogSettings {
            ambient_intensity: 0.1,
            ..default()
        })
        .insert(ScreenSpaceAmbientOcclusionBundle::default())
        .insert(ScreenSpaceAmbientOcclusionSettings {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
        });

    commands.spawn(
        TextBundle::from_section("Press space to toggle wireframes", TextStyle::default())
            .with_style(Style {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                ..default()
            }),
    );

    let ground_material = materials.add(StandardMaterial {
        base_color: Color::from(palettes::css::GRAY),
        perceptual_roughness: 0.7,
        reflectance: 0.4,
        ..default()
    });

    let now = Instant::now();

    println!("Generating meshes...");

    let chunks_meshes: Vec<Option<Mesh>> = model
        .chunks
        .par_iter()
        .map(|chunk| chunk.generate_mesh())
        .collect();

    for (i, chunk_mesh) in chunks_meshes.iter().enumerate() {
        if chunk_mesh.is_none() {
            continue;
        }

        let chunk_mesh = chunk_mesh.as_ref().unwrap();

        let mesh = meshes.add(chunk_mesh.clone());

        let chunk_position = model.chunks[i].get_position();

        let base_color = if model_settings.material_type == MaterialType::Checkered {
            generate_chunk_color_checkered(chunk_position.x, chunk_position.y, chunk_position.z)
        } else {
            generate_chunk_color_gradient(i, model.chunks.len())
        };

        let mesh_material = materials.add(StandardMaterial {
            base_color,
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        });

        commands
            .spawn(PbrBundle {
                mesh,
                material: mesh_material.clone(),
                transform: Transform::from_translation(chunk_position.as_vec3()),
                ..default()
            })
            .insert(Chunk)
            .insert(Name::new(
                format!(
                    "Chunk {}x{}x{}",
                    chunk_position.x, chunk_position.y, chunk_position.z
                )
                .to_string(),
            ));
    }

    println!("\nGenerating meshes took {:?}", now.elapsed());

    commands.spawn(PbrBundle {
        mesh: meshes.add(
            Plane3d::default()
                .mesh()
                .size(250.0, 250.0)
                .subdivisions(32),
        ),
        material: ground_material,
        ..default()
    });
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
    if std::env::args().len() < 2 {
        println!("Usage: vtm-viewer <vtm-file>");
        std::process::exit(1);
    }

    let input = std::env::args().nth(1).unwrap();
    let input = Path::new(&input);

    println!("Opening VTM model {}", input.display());
    let model = import_model_from_vtm(&input);

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "VoxTreeModel Viewer".to_string(),
                    present_mode: PresentMode::Immediate,
                    ..default()
                }),
                ..default()
            }),
            EguiPlugin,
            GamePlugin,
            PanOrbitCameraPlugin,
            WireframePlugin,
            FrameTimeDiagnosticsPlugin,
            ScreenDiagnosticsPlugin::default(),
            ScreenFrameDiagnosticsPlugin,
            ScreenEntityDiagnosticsPlugin,
        ))
        .insert_resource(ClearColor(Color::Srgba(Srgba {
            red: 0.02,
            green: 0.02,
            blue: 0.02,
            alpha: 1.0,
        })))
        .insert_resource(ModelResource(model))
        .insert_resource(ModelSettings {
            material_type: MaterialType::Checkered,
        })
        .insert_resource(Msaa::Off)
        .run();

    println!("Exiting...");
}
