//! A Rust based Octree library
//!
//! This library provides a single [Octree](crate::octree::Octree) data structure and utility routines
//! to work with Morton keys used for indexing Octrees.
//!
//! An Octree is a tree data structure in which each internal node has exactly eight children.
//! Octrees are most often used to partition a three-dimensional space by recursively subdividing it into
//! eight octants. Octrees are the 3D analog of quadtrees.
//!
//! The library supports Octrees on single nodes and distributed across multiple nodes via MPI.
//! Each node inside on Octree is indexed through a [MortonKey](crate::morton::MortonKey). A Morton key is a
//! 64 bit integer value that uniquely encodes the position of each node in an Octree.
//!
//! The library is designed to only provide the Octree data structure itself. It is up to the user to build algorithms
//! around the data structure.
//!
//! The Octrees provides by this library are adaptive and 2:1 balanced by default. This means that no neighbour of a node can be more than
//! one level below or above the level of the node. This is a common property of Octrees used in scientific computing.
//! The adaptivity ensures that refined where it is needed.
//!
//! A heuristic strategic ensures that the tree is approximately load-balanced. The implementation of the load-balancing and the 2:1
//! balancing are modeled after the paper *[Bottom-Up Construction and 2:1 Balance Refinement of Linear Octrees in Parallel](https://epubs.siam.org/doi/10.1137/070681727)*
//! by Sundar et. al. The underlying implementation relies on parallel sorting of Morton keys, which is done using a simple bucket sort algorithm
//! in this library.
//!
//! ## Using the library.
//!
//! A new Octree is generated from a list of points using the [Octree::new](crate::octree::Octree::new) function.
//! ```
//! use bempp_octree::{Octree, generate_random_points};
//! use rand::prelude::*;
//! use rand_chacha::ChaCha8Rng;
//! use mpi::traits::Communicator;
//!
//! let universe = mpi::initialize().unwrap();
//!
//! let comm = universe.world();
//! let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);
//! let npoints = 10000;
//! let max_level = 15;
//! let max_leaf_points = 50;
//!
//! let points = generate_random_points(npoints, &mut rng, &comm);
//! let octree = Octree::new(&points, max_level, max_leaf_points, &comm);
//! ```
//! In this code we first initialize MPI and generate random points in the unit cube.
//! We then create an Octree with a maximum level of 15 and a maximum of 50 points per leaf node.
//! Note that when the code is run with in `debug` mode a number of expensive assertion checks are
//! performed during tree construction which cost noticeable time for larger trees and involve
//! communication across MPI nodes. These checks are disabled in `release` mode.
//!
//! An octree is constructed by definining and load balancing a coarse tree and then refining
//! the coarse tree as long as the number of points per leaf node exceeds the maximum. Once sufficiently
//! refined the tree is 2:1 balanced again. The `octree` data structure stores the coarse tree and the
//! fine leaf nodes of the octree.
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod constants;
pub mod geometry;
pub mod morton;
pub mod octree;
pub mod parsort;
//pub mod serial;
pub mod tools;
pub mod types;

pub use crate::octree::Octree;
pub use crate::tools::generate_random_points;
