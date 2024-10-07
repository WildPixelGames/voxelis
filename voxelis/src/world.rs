use bevy::math::IVec3;
use rayon::prelude::*;

use crate::chunk::Chunk;

#[derive(Default)]
pub struct World {
    pub chunks_size: IVec3,
    pub chunks_len: usize,
    pub chunks: Vec<Chunk>,
}

impl World {
    pub fn new() -> Self {
        let chunks_size = IVec3::new(32, 32, 32);
        let chunks_len = chunks_size.x as usize * chunks_size.y as usize * chunks_size.z as usize;
        let chunks = Vec::with_capacity(chunks_len);

        Self {
            chunks_size,
            chunks_len,
            chunks,
        }
    }

    pub fn with_size(size: IVec3) -> Self {
        let chunks_len = size.x as usize * size.y as usize * size.z as usize;
        let chunks = Vec::with_capacity(chunks_len);

        Self {
            chunks_size: size,
            chunks_len,
            chunks,
        }
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    pub fn resize(&mut self, size: IVec3) {
        self.chunks_size = size;
        self.chunks_len = size.x as usize * size.y as usize * size.z as usize;
        self.chunks = Vec::with_capacity(self.chunks_len);
    }

    pub fn update_lods(&mut self) {
        self.chunks.par_iter_mut().for_each(|chunk| {
            if !chunk.is_empty() {
                chunk.update_lods();
            }
        });
    }

    pub fn serialize(&self, data: &mut Vec<u8>) {
        for chunk in self.chunks.iter() {
            chunk.serialize(data);
        }
    }
}
