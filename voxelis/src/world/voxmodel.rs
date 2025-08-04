use std::{
    collections::HashMap,
    io::{BufReader, Write},
    sync::Arc,
};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use glam::{IVec3, UVec3};
use parking_lot::RwLock;

use rustc_hash::FxHashMap;

#[cfg(feature = "memory_stats")]
use crate::interner::InternerStats;

use crate::{
    BlockId, Lod, MaxDepth, VoxInterner, VoxelTrait,
    interner::EMPTY_CHILD,
    io::varint::{decode_varint_u32_from_reader, encode_varint_u32},
    spatial::{VoxOpsChunkConfig, VoxOpsChunkLocalContainer, VoxOpsConfig, VoxOpsSpatial3D},
    world::{
        VoxChunk,
        voxchunk::{deserialize_chunk, serialize_chunk},
    },
};

pub struct VoxModel<T: VoxelTrait> {
    pub max_depth: MaxDepth,
    pub chunk_world_size: f32,
    pub world_bounds: IVec3,
    pub chunks: HashMap<IVec3, VoxChunk<T>>,
    pub interner: Arc<RwLock<VoxInterner<T>>>,
}

fn initialize_chunks<T: VoxelTrait>(
    max_depth: MaxDepth,
    chunk_world_size: f32,
    bounds: IVec3,
) -> HashMap<IVec3, VoxChunk<T>> {
    #[cfg(feature = "tracy")]
    let _span = tracy_client::span!("initialize_chunks");

    let estimated_capacity = bounds.x as usize * bounds.y as usize * bounds.z as usize;

    let mut chunks = HashMap::with_capacity(estimated_capacity);

    for y in 0..bounds.y {
        for z in 0..bounds.z {
            for x in 0..bounds.x {
                chunks.insert(
                    IVec3::new(x, y, z),
                    VoxChunk::with_position(chunk_world_size, max_depth, x, y, z),
                );
            }
        }
    }

    chunks
}

impl<T: VoxelTrait> VoxModel<T> {
    pub fn empty(max_depth: MaxDepth, chunk_world_size: f32, memory_budget: usize) -> Self {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::empty");

        let interner = Arc::new(RwLock::new(VoxInterner::with_memory_budget(memory_budget)));

        Self {
            max_depth,
            chunk_world_size,
            world_bounds: IVec3::ZERO,
            chunks: HashMap::default(),
            interner,
        }
    }

    pub fn new(max_depth: MaxDepth, chunk_world_size: f32, memory_budget: usize) -> Self {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::new");

        let interner = Arc::new(RwLock::new(VoxInterner::with_memory_budget(memory_budget)));
        let world_bounds = IVec3::new(32, 32, 32);
        let chunks = initialize_chunks(max_depth, chunk_world_size, world_bounds);

        Self {
            max_depth,
            chunk_world_size,
            world_bounds,
            chunks,
            interner,
        }
    }

    pub fn with_dimensions(
        max_depth: MaxDepth,
        chunk_world_size: f32,
        world_bounds: IVec3,
        memory_budget: usize,
    ) -> Self {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::with_dimensions");

        println!(
            "Creating model with bounds {world_bounds:?}, chunk: {chunk_world_size}m depth: {max_depth}"
        );
        let interner = Arc::new(RwLock::new(VoxInterner::with_memory_budget(memory_budget)));
        let chunks = initialize_chunks(max_depth, chunk_world_size, world_bounds);

        Self {
            max_depth,
            chunk_world_size,
            world_bounds,
            chunks,
            interner,
        }
    }

    pub fn get_or_create_chunk(&mut self, position: IVec3) -> &mut VoxChunk<T> {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::get_or_create_chunk");

        self.chunks.entry(position).or_insert_with(|| {
            self.world_bounds.x = position.x.max(self.world_bounds.x);
            self.world_bounds.y = position.y.max(self.world_bounds.y);
            self.world_bounds.z = position.z.max(self.world_bounds.z);

            VoxChunk::with_position(
                self.chunk_world_size,
                self.max_depth,
                position.x,
                position.y,
                position.z,
            )
        })
    }

    pub fn get_interner(&self) -> Arc<RwLock<VoxInterner<T>>> {
        self.interner.clone()
    }

    pub fn clear(&mut self) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::clear");

        self.world_bounds = IVec3::ZERO;
        self.chunks.clear();
    }

    pub fn resize(&mut self, bounds: IVec3) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::resize");

        self.chunks.clear();

        self.world_bounds = bounds;
        self.chunks = initialize_chunks(self.max_depth, self.chunk_world_size, self.world_bounds);
    }

    pub fn get_bounds_size(&self) -> usize {
        self.world_bounds.x as usize * self.world_bounds.y as usize * self.world_bounds.z as usize
    }

    pub fn is_position_in_bounds(&self, position: IVec3) -> bool {
        position.x >= 0
            && position.x < self.world_bounds.x
            && position.y >= 0
            && position.y < self.world_bounds.y
            && position.z >= 0
            && position.z < self.world_bounds.z
    }

    #[cfg(feature = "memory_stats")]
    pub fn interner_stats(&self) -> InternerStats {
        self.interner.read().stats()
    }

    pub fn serialize(&self, data: &mut Vec<u8>) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::serialize");

        const BUFFER_SIZE: usize = 256;

        let interner = self.interner.read();

        let leaf_patterns = interner.leaf_patterns();
        let branch_patterns = interner.branch_patterns();

        let mut id_map: FxHashMap<u32, u32> = FxHashMap::default();
        id_map.insert(0, 0);

        let mut leaf_patterns = leaf_patterns.values().map(|id| *id).collect::<Vec<_>>();
        let mut branch_patterns = branch_patterns.values().copied().collect::<Vec<_>>();

        let mut next_id = 1;

        leaf_patterns.sort_by_key(|id| id.index());
        branch_patterns.sort_by_key(|id| id.index());

        leaf_patterns.iter().for_each(|id| {
            id_map.insert(id.index(), next_id);
            next_id += 1;
        });

        branch_patterns.iter().for_each(|id| {
            let index = id.index();
            if index == 0 {
                return;
            }

            id_map.insert(id.index(), next_id);
            next_id += 1;
        });

        let leaf_size = leaf_patterns.len();
        assert!(leaf_size <= u32::MAX as usize);
        let branch_size = branch_patterns.len();
        assert!(branch_size <= u32::MAX as usize);
        let size = leaf_size + branch_size;
        assert!(size <= u32::MAX as usize);

        let leaf_size = leaf_size as u32;
        let branch_size = branch_size as u32;

        let mut writer = std::io::BufWriter::new(data);

        writer.write_u32::<BigEndian>(leaf_size).unwrap();
        for id in leaf_patterns.iter() {
            let new_id = *id_map.get(&id.index()).unwrap();
            // println!(" leaf id: {} -> {new_id}", id.index());
            let new_id_bytes = encode_varint_u32(new_id);
            // writer.write_u32::<BigEndian>(new_id).unwrap();
            writer.write_all(&new_id_bytes).unwrap();
            let value = interner.get_value(id);
            value.write_as_be(&mut writer).unwrap();
        }

        writer.write_u32::<BigEndian>(branch_size - 1).unwrap();
        for id in branch_patterns.iter() {
            if id.index() == 0 {
                continue;
            }

            let new_id = *id_map.get(&id.index()).unwrap();
            // println!("branch id: {} -> {new_id}", id.index());
            let new_id_bytes = encode_varint_u32(new_id);
            // writer.write_u32::<BigEndian>(new_id).unwrap();
            writer.write_all(&new_id_bytes).unwrap();
            writer.write_u8(id.mask()).unwrap();
            let branch = interner.get_children_ref(id);
            for child in branch.iter() {
                if child.is_empty() {
                    // println!(" empty child");
                    continue;
                }
                let new_id = *id_map.get(&child.index()).unwrap();
                // println!("  child id: {} -> {new_id}", child.index());
                let new_id_bytes = encode_varint_u32(new_id);
                // writer.write_u32::<BigEndian>(new_id).unwrap();
                writer.write_all(&new_id_bytes).unwrap();
            }
            let branch_lod_value = interner.get_value(id);
            branch_lod_value.write_as_be(&mut writer).unwrap();
        }

        let chunks_data: Vec<Vec<u8>> = self
            .chunks
            .iter() // .par_iter() needs Send + Sync for VoxelTrait
            .map(|(_, chunk)| {
                let mut buffer = Vec::with_capacity(BUFFER_SIZE);
                serialize_chunk(chunk, &id_map, &mut buffer);
                buffer
            })
            .collect();

        let actual_chunks_len = self.chunks.len();
        writer
            .write_u32::<BigEndian>(actual_chunks_len as u32)
            .unwrap();

        for chunk_data in chunks_data.iter() {
            writer.write_all(chunk_data).unwrap();
        }
    }

    pub fn deserialize(&mut self, data: &[u8]) {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::deserialize");

        println!("Deserializing chunks...");

        let now = std::time::Instant::now();

        let mut reader = BufReader::new(data);

        let leaf_size = reader.read_u32::<BigEndian>().unwrap();
        // let mut leaf_patterns: HashMap<u32, (BlockId, i32)> =
        //     HashMap<K, V, FxBuildHasher>(leaf_size as usize);
        let mut leaf_patterns: FxHashMap<u32, (BlockId, T)> = FxHashMap::default();

        let mut interner = self.interner.write();

        for _ in 0..leaf_size {
            let id = decode_varint_u32_from_reader(&mut reader).unwrap();
            let value = T::read_from_be(&mut reader).unwrap();

            let block_id = interner.deserialize_leaf(id, value);
            leaf_patterns.insert(id, (block_id, value));

            println!(" leaf id: {block_id:?} -> {value}");
        }

        let branch_size = reader.read_u32::<BigEndian>().unwrap();
        let mut branch_patterns: FxHashMap<u32, (BlockId, [u32; 8], T)> =
        // FxHashMap::with_capacity(branch_size as usize);
            FxHashMap::default();

        branch_patterns.insert(0, (BlockId::EMPTY, [0u32; 8], T::default()));

        for _ in 0..branch_size {
            let id = decode_varint_u32_from_reader(&mut reader).unwrap();
            assert_ne!(id, 0);

            let mask = reader.read_u8().unwrap();
            // println!("id: {} mask: {:08b}", id, mask);
            let mut types: u8 = 0;
            let mut children = [0u32; 8];
            for child_id in 0..8 {
                if mask & (1 << child_id) == 0 {
                    // println!(" skipping child {child_id}");
                    continue;
                }
                // println!(" reading child {child_id}");
                children[child_id] = decode_varint_u32_from_reader(&mut reader).unwrap();
                if leaf_patterns.contains_key(&children[child_id]) {
                    types |= 1 << child_id;
                }
            }
            let lod_value = T::read_from_be(&mut reader).unwrap();

            let block_id = interner.preallocate_branch_id(id, types, mask);

            branch_patterns.insert(id, (block_id, children, lod_value));
            // println!(
            //     " branch: mask: {mask:08b} types: {types:08b} id: {id:08X} [{block_id:?}] -> {children:08X?}"
            // );
            assert_ne!(mask, 0);
        }

        branch_patterns
            .iter()
            .for_each(|(id, (block_id, children, lod_value))| {
                if *id == 0 {
                    return;
                }

                let types = block_id.types();
                let mask = block_id.mask();

                let mut branch = EMPTY_CHILD;
                for child_idx in 0..8 {
                    if mask & (1 << child_idx) == 0 {
                        continue;
                    }

                    let child_id = children[child_idx];
                    if types & (1 << child_idx) != 0 {
                        let (leaf_id, _) = leaf_patterns.get(&child_id).unwrap();
                        branch[child_idx] = *leaf_id;
                    } else {
                        branch[child_idx] = branch_patterns.get(&child_id).unwrap().0;
                    }
                }

                // println!("branch: {block_id:?} -> {branch:?}");
                interner.deserialize_branch(*block_id, branch, types, mask, *lod_value);
            });

        // drop(interner);

        let mut branch_ids = branch_patterns
            .iter()
            .map(|(_, (block_id, _, _))| *block_id)
            .collect::<Vec<_>>();
        branch_ids.sort_by_key(|id| id.index());

        // for branch_id in branch_ids.iter() {
        //     println!("Branch id: {branch_id:?}");
        //     interner.dump_node(*branch_id, 0, "  ");
        // }

        let actual_chunks_len = reader.read_u32::<BigEndian>().unwrap();

        for _ in 0..actual_chunks_len {
            let chunk = deserialize_chunk(
                &mut interner,
                &leaf_patterns,
                &branch_patterns,
                &mut reader,
                self.chunk_world_size,
                self.max_depth,
            );

            self.chunks.insert(chunk.position_3d(), chunk);
        }

        let elapsed = now.elapsed();
        println!("Deserializing chunks took {elapsed:?}");
    }
}

impl<T: VoxelTrait> VoxOpsConfig for VoxModel<T> {
    fn max_depth(&self, lod: Lod) -> MaxDepth {
        self.max_depth.for_lod(lod)
    }

    fn voxels_per_axis(&self, lod: Lod) -> u32 {
        1 << self.max_depth.for_lod(lod).max()
    }
}

impl<T: VoxelTrait> VoxOpsChunkConfig for VoxModel<T> {
    fn chunk_dimensions(&self) -> UVec3 {
        self.world_bounds.as_uvec3() + UVec3::ONE
    }

    fn chunk_size(&self) -> f32 {
        self.chunk_world_size
    }

    fn voxel_size(&self, lod: Lod) -> f32 {
        1.0 / self.voxels_per_axis(lod) as f32 * self.chunk_world_size
    }
}

impl<T: VoxelTrait> VoxOpsChunkLocalContainer<T> for VoxModel<T> {
    fn has_local_chunk(&self, position: UVec3) -> bool {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::has_local_chunk");

        let position = position.as_ivec3();
        self.chunks.contains_key(&position)
    }

    fn local_chunk(&self, position: UVec3) -> Option<&VoxChunk<T>> {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::chunk");

        let position = position.as_ivec3();
        self.chunks.get(&position)
    }

    fn local_chunk_mut(&mut self, position: UVec3) -> Option<&mut VoxChunk<T>> {
        #[cfg(feature = "tracy")]
        let _span = tracy_client::span!("VoxModel::chunk_mut");

        let position = position.as_ivec3();
        self.chunks.get_mut(&position)
    }
}
