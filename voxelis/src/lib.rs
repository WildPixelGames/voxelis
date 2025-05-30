// #![warn(missing_docs)]
// #![warn(rustdoc::missing_crate_level_docs)]
// #![warn(clippy::all)]
// #![warn(clippy::pedantic)]
// #![allow(clippy::module_name_repetitions)]
#![warn(clippy::cargo)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_not_else)]

pub mod core;
pub mod interner;
pub mod io;
pub mod spatial;
pub mod utils;
pub mod world;

pub use core::{Batch, BlockId, Lod, MaxDepth, TraversalDepth, VoxelTrait};
pub use interner::VoxInterner;
