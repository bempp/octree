//! A Rust based Octree library
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod constants;
pub mod geometry;
pub mod morton;
pub mod octree;
pub mod parallel_octree;
pub mod parsort;
pub mod types;
