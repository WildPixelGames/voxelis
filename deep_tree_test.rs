use voxelis::{
    MaxDepth, VoxInterner,
    spatial::VoxTree,
};

#[test]
fn test_deep_tree_creation() {
    println!("Testing VoxTree depth scaling capability...");

    // Test various depths to show scaling works
    let test_depths = [6, 7, 8, 9, 10];

    for &depth in &test_depths {
        println!("\n--- Testing Depth {} ---", depth);

        let mut interner = VoxInterner::<u8>::with_memory_budget(1024 * 1024 * 256);
        let mut tree = VoxTree::new(MaxDepth::new(depth));

        let voxels_per_axis = 1usize << depth;
        let total_voxels = voxels_per_axis * voxels_per_axis * voxels_per_axis;

        println!("  Depth: {}", depth);
        println!("  Voxels per axis: {}", voxels_per_axis);
        println!("  Total voxels: {} ({:.2e})", total_voxels, total_voxels as f64);

        // Test basic voxel operations
        let test_pos = glam::IVec3::new(1, 2, 3);
        let test_value = 42u8;

        assert!(tree.set(&mut interner, test_pos, test_value));
        assert_eq!(tree.get(&interner, test_pos), Some(test_value));

        println!("  ✓ Basic set/get operations work");
        println!("  ✓ Tree created successfully with depth {}", depth);
    }

    println!("\n🎉 All depth levels work! VoxTree successfully scaled from depth 6 to 10.");
    println!("   This represents a 1024x increase in voxel capacity per tree.");
}

#[test]
fn test_depth_10_extreme() {
    println!("\n--- Extreme Test: Depth 10 (1024³ voxels) ---");

    let mut interner = VoxInterner::<u8>::with_memory_budget(1024 * 1024 * 1024); // 1GB budget
    let mut tree = VoxTree::new(MaxDepth::new(10));

    // Test a few sparse voxels in the massive space
    let positions = [
        glam::IVec3::new(0, 0, 0),
        glam::IVec3::new(100, 200, 300),
        glam::IVec3::new(500, 500, 500),
        glam::IVec3::new(1023, 1023, 1023), // Max coordinate
    ];

    for (i, pos) in positions.iter().enumerate() {
        let value = (i + 1) as u8;
        assert!(tree.set(&mut interner, *pos, value));
        assert_eq!(tree.get(&interner, *pos), Some(value));
    }

    println!("  ✓ Successfully set/get voxels in 1024³ space");
    println!("  ✓ Extreme depth 10 works with sparse data");

    // Note: Dense operations would be memory-intensive, but sparse works great
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_deep_tree_tests() {
        test_deep_tree_creation();
        test_depth_10_extreme();
    }
}
