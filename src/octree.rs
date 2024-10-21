//! Definition of Octree.
pub mod parallel;
use std::collections::HashMap;

use mpi::traits::CommunicatorCollectives;
pub use parallel::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    constants::DEEPEST_LEVEL,
    geometry::{PhysicalBox, Point},
    morton::MortonKey,
};

/// Stores what type of key it is.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum KeyType {
    /// A local leaf.
    LocalLeaf,
    /// A local interior key.
    LocalInterior,
    /// A global key.
    Global,
    /// A ghost key from a specific process.
    Ghost(usize),
}

/// A general structure for octrees.
pub struct Octree<'o, C> {
    points: Vec<Point>,
    point_keys: Vec<MortonKey>,
    coarse_tree: Vec<MortonKey>,
    leaf_tree: Vec<MortonKey>,
    coarse_tree_bounds: Vec<MortonKey>,
    all_keys: HashMap<MortonKey, KeyType>,
    leaf_keys_to_points: HashMap<MortonKey, Vec<usize>>,
    bounding_box: PhysicalBox,
    comm: &'o C,
}

impl<'o, C: CommunicatorCollectives> Octree<'o, C> {
    /// Create a new distributed Octree.
    pub fn new(points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C) -> Self {
        // We need a random number generator for sorting. For simplicity we use a ChaCha8 random number generator
        // seeded with the rank of the process.
        let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

        // First compute the Morton keys of the points.
        let (point_keys, bounding_box) = points_to_morton(points, DEEPEST_LEVEL as usize, comm);

        // Generate the coarse tree

        let (coarse_tree, leaf_tree) = {
            // Linearize the keys.
            let linear_keys = linearize(&point_keys, &mut rng, comm);

            // Compute the first version of the coarse tree without load balancing.
            // We want to ensure that it is 2:1 balanced.
            let coarse_tree = compute_coarse_tree(&linear_keys, comm);

            let coarse_tree = balance(&coarse_tree, &mut rng, comm);
            debug_assert!(is_complete_linear_tree(&coarse_tree, comm));

            // We now compute the weights for the initial coarse tree.

            let weights = compute_coarse_tree_weights(&linear_keys, &coarse_tree, comm);

            // We now load balance the initial coarse tree. This forms our final coarse tree
            // that is used from now on.

            let coarse_tree = load_balance(&coarse_tree, &weights, comm);
            // We also want to redistribute the fine keys with respect to the load balanced coarse trees.

            let fine_keys =
                redistribute_with_respect_to_coarse_tree(&linear_keys, &coarse_tree, comm);

            // We now create the refined tree by recursing the coarse tree until we are at max level
            // or the fine tree keys per coarse tree box is small enough.
            let refined_tree =
                create_local_tree(&fine_keys, &coarse_tree, max_level, max_leaf_points);

            // We now need to 2:1 balance the refined tree and then redistribute again with respect to the coarse tree.

            let refined_tree = redistribute_with_respect_to_coarse_tree(
                &balance(&refined_tree, &mut rng, comm),
                &coarse_tree,
                comm,
            );

            (coarse_tree, refined_tree)

            // redistribute the balanced tree according to coarse tree
        };

        let (points, point_keys) = redistribute_points_with_respect_to_coarse_tree(
            points,
            &point_keys,
            &coarse_tree,
            comm,
        );

        let coarse_tree_bounds = get_tree_bins(&coarse_tree, comm);

        // Duplicate the coarse tree across all nodes

        // let coarse_tree = gather_to_all(&coarse_tree, comm);

        let all_keys = generate_all_keys(&leaf_tree, &coarse_tree, &coarse_tree_bounds, comm);

        let leaf_keys_to_points = assign_points_to_leaf_keys(&point_keys, &leaf_tree);

        Self {
            points: points.to_vec(),
            point_keys,
            coarse_tree,
            leaf_tree,
            coarse_tree_bounds,
            all_keys,
            leaf_keys_to_points,
            bounding_box,
            comm,
        }
    }

    /// Return the keys associated with the redistributed points.
    pub fn point_keys(&self) -> &Vec<MortonKey> {
        &self.point_keys
    }

    /// Return the bounding box.
    pub fn bounding_box(&self) -> &PhysicalBox {
        &self.bounding_box
    }

    /// Return the associated coarse tree.
    pub fn coarse_tree(&self) -> &Vec<MortonKey> {
        &self.coarse_tree
    }

    /// Return the points.
    ///
    /// Points are distributed across the nodes as part of the tree generation.
    /// This function returns the redistributed points.
    pub fn points(&self) -> &Vec<Point> {
        &self.points
    }

    /// Return the leaf tree.
    pub fn leaf_tree(&self) -> &Vec<MortonKey> {
        &self.leaf_tree
    }

    /// Return the map from leaf keys to point indices
    pub fn leafs_to_point_indices(&self) -> &HashMap<MortonKey, Vec<usize>> {
        &self.leaf_keys_to_points
    }

    /// Get the coarse tree bounds.
    ///
    /// This returns an array of size the number of ranks,
    /// where each element consists of the smallest Morton key in
    /// the corresponding rank.
    pub fn coarse_tree_bounds(&self) -> &Vec<MortonKey> {
        &self.coarse_tree_bounds
    }

    /// Return the communicator.
    pub fn comm(&self) -> &C {
        self.comm
    }

    /// Return a map of all keys.
    pub fn all_keys(&self) -> &HashMap<MortonKey, KeyType> {
        &self.all_keys
    }
}
