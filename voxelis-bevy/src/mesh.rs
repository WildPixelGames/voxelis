use bevy::{
    prelude::Mesh,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};
use voxelis::chunk::{Chunk, Vec3};

pub fn generate_mesh(chunk: &Chunk) -> Option<Mesh> {
    if chunk.is_empty() {
        return None;
    }

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    let data = chunk.to_vec(0);

    Chunk::generate_mesh_arrays(&data, &mut vertices, &mut normals, &mut indices, Vec3::ZERO);

    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(Indices::U32(indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals),
    )
}

pub fn generate_greedy_mesh(chunk: &Chunk) -> Option<Mesh> {
    if chunk.is_empty() {
        return None;
    }

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    let data = chunk.to_vec(0);

    Chunk::generate_greedy_mesh_arrays(
        &data,
        &mut vertices,
        &mut normals,
        &mut indices,
        Vec3::ZERO,
    );

    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(Indices::U32(indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals),
    )
}
