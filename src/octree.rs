pub mod parallel;
use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use mpi::traits::{CommunicatorCollectives, Equivalence};
pub use parallel::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    constants::DEEPEST_LEVEL,
    geometry::{PhysicalBox, Point},
    morton::MortonKey,
    tools::gather_to_all,
};

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum KeyStatus {
    LocalLeaf,
    LocalInterior,
    Global,
    Ghost(usize),
}

/// A general structure for octrees.
pub struct Octree<'o, C> {
    points: Vec<Point>,
    point_keys: Vec<MortonKey>,
    coarse_tree: Vec<MortonKey>,
    leaf_tree: Vec<MortonKey>,
    coarse_tree_bounds: Vec<MortonKey>,
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
            let coarse_tree = compute_coarse_tree(&linear_keys, comm);
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

        Self {
            points: points.to_vec(),
            point_keys,
            coarse_tree,
            leaf_tree,
            coarse_tree_bounds,
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

    /// Generate all leaf and interior keys.
    pub fn generate_all_keys(&self) -> HashMap<MortonKey, KeyStatus> {
        let rank = self.comm().rank() as usize;
        let size = self.comm().size() as usize;

        let mut all_keys = HashMap::<MortonKey, KeyStatus>::new();
        let mut leaf_keys: HashSet<MortonKey> =
            HashSet::from_iter(self.leaf_tree().iter().copied());

        let mut global_keys = HashSet::<MortonKey>::new();

        // First deal with the parents of the coarse tree. These are different
        // as they may exist on multiple nodes, so receive a different label.

        for &key in self.coarse_tree() {
            let mut parent = key.parent();
            while parent.level() > 0 && !all_keys.contains_key(&parent) {
                global_keys.insert(parent);
                parent = parent.parent();
            }
        }

        // We now send around the parents of the coarse tree to every node. These will
        // be global keys.

        let global_keys = gather_to_all(&global_keys.iter().copied().collect_vec(), self.comm());

        // We can now insert the global keys into `all_keys` with the `Global` label.
        // There may be duplicates in the `global_keys` array. So need to check for that.

        for &key in &global_keys {
            if !all_keys.contains_key(&key) {
                all_keys.insert(key, KeyStatus::Global);
            }
        }

        // We now deal with the fine leafs and their ancestors.
        // The leafs of the coarse tree will also be either part
        // of the fine tree leafs or will be interior keys. In either
        // case the following loop catches them.

        for leaf in leaf_keys {
            debug_assert!(!all_keys.contains_key(&leaf));
            all_keys.insert(leaf, KeyStatus::LocalLeaf);
            let mut parent = leaf.parent();
            while parent.level() > 0 && !all_keys.contains_key(&parent) {
                all_keys.insert(parent, KeyStatus::LocalInterior);
                parent = parent.parent();
            }
        }

        // This maps from rank to the keys that we want to send to the ranks
        let mut rank_key = HashMap::<usize, Vec<MortonKey>>::new();
        for index in 0..size - 1 {
            rank_key.insert(index, Vec::<MortonKey>::new());
        }

        for (&key, &status) in all_keys.iter() {
            // We need not send around global keys to neighbors.
            if status == KeyStatus::Global {
                continue;
            }
            for &neighbor in key.neighbours().iter().filter(|&&key| key.is_valid()) {
                // If the neighbour is a global key then continue.
                if let Some(&value) = all_keys.get(&neighbor) {
                    if value == KeyStatus::Global {
                        continue;
                    }
                }
                // Get rank of the neighbour
                let neighbor_rank = get_key_index(&self.coarse_tree_bounds(), neighbor);
                rank_key
                    .entry(neighbor_rank)
                    .and_modify(|keys| keys.push(key));
            }
        }

        // We now know which key needs to be sent to which rank.
        // Turn to array, get the counts and send around.

        let (arr, counts) = {
            let mut arr = Vec::<MortonKey>::new();
            let mut counts = Vec::<usize>::new();
            for index in 0..size - 1 {
                let value = rank_key.get(&index).unwrap();
                arr.extend(value.iter());
                counts.push(value.len());
            }
            (arr, counts)
        };

        all_keys
    }
}
