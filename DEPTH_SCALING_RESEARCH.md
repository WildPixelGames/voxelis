<file_path>
voxelis/DEPTH_SCALING_RESEARCH.md
</file_path>

# VoxTree Depth Scaling Research & Implementation

## Executive Summary

**The VoxTree architecture CAN be scaled to much deeper trees**, but the current mesh generation system limits practical depth to 64³ voxels per chunk. We successfully demonstrated depth scaling from the original limit of 6 to 10, though mesh generation caps effective usage at depth 6.

## Current Architecture Analysis

### Depth Limits Discovered

1. **Hardcoded Limit**: `MAX_ALLOWED_DEPTH = 7` in `voxelis/src/interner/consts.rs`
   - This was increased to 11, allowing theoretical depths up to 10 (1024³ voxels)

2. **PATH_MASKS Constraint**: 2D array in `voxelis/src/spatial/voxtree.rs` 
   - Previously hardcoded for depths 0-6
   - Extended to support depths 0-10
   - Uses u32 (32 bits), theoretically allowing ~10 levels

3. **Mesh Generation Bottleneck**: `MAX_VOXELS_PER_AXIS = 64` in `voxelis/src/utils/mesh.rs`
   - **Critical limitation**: Mesh generation only supports 64³ voxel resolution
   - Any LOD level producing >64³ mesh resolution will crash with index out of bounds

### Theoretical vs. Practical Limits

| Depth | Voxels per Chunk | Status |
|-------|------------------|--------|
| 6     | 64³ = 262,144   | ✅ Working (mesh compatible) |
| 7     | 128³ = 2M       | ✅ Tree creation works |
| 8     | 256³ = 16M      | ✅ Tree creation works |
| 9     | 512³ = 134M     | ✅ Tree creation works |
| 10    | 1024³ = 1B      | ✅ Tree creation works |

## Implementation Changes Made

### 1. Increased Maximum Depth
```rust
// voxelis/src/interner/consts.rs
pub const MAX_ALLOWED_DEPTH: usize = 11; // Was 7
```

### 2. Extended PATH_MASKS Array
- Added support for depths 7-10 in the PATH_MASKS lookup table
- Each additional depth level requires 3 more bits in the bitmask

### 3. Created Deep LOD Example
- `adaptive_lod_deep.rs`: Demonstrates scaling with larger chunk sizes
- Shows LOD working from 64³ down to 1³ mesh resolution
- Uses 20.48 unit chunk size vs original 1.28

## Key Findings

### ✅ What Works
- **Tree Creation**: Depths up to 10 work for data storage
- **LOD System**: Distance-based LOD switching functions correctly
- **Memory Efficiency**: SVO-DAG compression keeps memory usage reasonable
- **Performance**: Deep trees perform well for sparse data

### ❌ Current Limitations
- **Mesh Generation**: Hardcoded to 64³ voxel mesh resolution
- **Memory Usage**: Deep trees with dense data become memory-intensive
- **Generation Time**: Creating large spheres in deep trees is slow

### 🔧 Required for Full Scaling
1. **Dynamic Mesh Generation**: Replace hardcoded 64³ limits with dynamic sizing
2. **Memory Management**: Optimize for sparse vs dense voxel distributions  
3. **Generation Algorithms**: Make sphere/batch generation depth-aware

## Recommendations

### Immediate (Working Now)
- Use depth 6 with larger chunk sizes for bigger worlds
- Implement proper LOD streaming for large terrains
- Optimize for sparse voxel data (air is free in SVO)

### Future Development
- Rewrite mesh generation for dynamic voxel resolutions
- Add multi-resolution mesh generation (hybrid LOD)
- Implement chunk streaming with background mesh generation

## Performance Characteristics

With depth 6 (64³ voxels) and 20x larger chunks:
- **World Scale**: 20.48 units per chunk vs 1.28 originally  
- **LOD Range**: From 64³ to 1³ mesh resolution
- **Memory**: Efficient SVO compression for sparse data
- **Rendering**: Smooth LOD transitions based on distance

The architecture is fundamentally sound for massive scaling - the mesh generation just needs modernization to unlock the full potential.