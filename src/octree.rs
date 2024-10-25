//! Definition of Octree.
mod implementation;
use std::collections::HashMap;

pub(crate) use implementation::*;
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Root},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    constants::DEEPEST_LEVEL,
    geometry::{PhysicalBox, Point},
    morton::MortonKey,
    tools::gather_to_root,
};

/// Stores the type of the key relative to the octree.
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
    coarse_tree_leafs: Vec<MortonKey>,
    leaf_keys: Vec<MortonKey>,
    coarse_tree_bounds: Vec<MortonKey>,
    all_keys: HashMap<MortonKey, KeyType>,
    neighbours: HashMap<MortonKey, Vec<MortonKey>>,
    leaf_keys_to_local_point_indices: HashMap<MortonKey, Vec<usize>>,
    bounding_box: PhysicalBox,
    comm: &'o C,
}

impl<'o, C: CommunicatorCollectives> Octree<'o, C> {
    /// Create a new distributed Octree.
    ///
    /// # Arguments
    /// - `max_level`: The maximum level of the tree. The maximum level is 16.
    /// - `max_leaf_points`: The maximum number of points per leaf.
    /// - `comm`: The communicator.
    ///
    /// # Returns
    /// A new Octree.
    ///
    /// # Note
    /// The points are redistributed during construction of the octree. The tree stores
    /// the redistributed points and the corresponding Morton keys.
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
        let neighbours = compute_neighbours(&all_keys);

        let leaf_keys_to_points = assign_points_to_leaf_keys(&point_keys, &leaf_tree);

        Self {
            points: points.to_vec(),
            point_keys,
            coarse_tree_leafs: coarse_tree,
            leaf_keys: leaf_tree,
            coarse_tree_bounds,
            all_keys,
            neighbours,
            leaf_keys_to_local_point_indices: leaf_keys_to_points,
            bounding_box,
            comm,
        }
    }

    /// Return the Morton keys associated with points.
    pub fn point_keys(&self) -> &Vec<MortonKey> {
        &self.point_keys
    }

    /// Return the bounding box.
    ///
    /// The bounding box is computed globally for the distributed octree.
    pub fn bounding_box(&self) -> &PhysicalBox {
        &self.bounding_box
    }

    /// Return the coarse tree leafs.
    pub fn coarse_tree_leafs(&self) -> &Vec<MortonKey> {
        &self.coarse_tree_leafs
    }

    /// Return the points.
    ///
    /// Points are distributed across the nodes as part of the tree generation.
    /// This function returns the redistributed points.
    pub fn points(&self) -> &Vec<Point> {
        &self.points
    }

    /// Return the leaf nodes.
    pub fn leaf_keys(&self) -> &Vec<MortonKey> {
        &self.leaf_keys
    }

    /// Return the map from leaf keys to local point indices.
    ///
    /// This allows to find the points associated with a given key.
    /// # Example
    /// ```ignore
    /// let leaf_map = octree.leaf_keys_to_local_point_indices();
    /// let indices = leaf_map.get(&key);
    /// let points_for_key = indices.iter().map(|&i| octree.points()[i]).collect::<Vec<_>>();
    /// ```
    /// Each point in `points_for_key` is contained in the leaf box defined by `key`.
    pub fn leaf_keys_to_local_point_indices(&self) -> &HashMap<MortonKey, Vec<usize>> {
        &self.leaf_keys_to_local_point_indices
    }

    /// Get the coarse tree bounds.
    ///
    /// This returns an array of size the number of ranks,
    /// where each element consists of the smallest Morton key in
    /// the corresponding rank.
    ///
    /// If a Morton key is on rank i with i not the last rank then
    /// ```text
    /// coarse_tree_bounds[i] <= key < coarse_tree_bounds[i+1]
    /// ```
    /// where as if i is the last rank then
    /// ```text
    /// coarse_tree_bounds[i] <= key
    /// ```
    /// This allows to find the rank of a given Morton key.
    pub fn coarse_tree_bounds(&self) -> &Vec<MortonKey> {
        &self.coarse_tree_bounds
    }

    /// Return the communicator.
    pub fn comm(&self) -> &C {
        self.comm
    }

    /// Return a map of all leaf and interior keys.
    ///
    /// The map assigns each key a [KeyType] identifier. It is one of:
    /// - [KeyType::LocalLeaf] for leaf keys
    /// - [KeyType::LocalInterior] for interior keys
    /// - [KeyType::Global] for global keys
    /// - [KeyType::Ghost], a typed enum for keys that are adjacent to keys
    ///   on the current rank but live on a different rank.
    ///
    /// Leaf keys have no children. Interior keys have children within the local rank.
    /// Global keys are keys that are not uniquely assigned to a rank but exist on all ranks.
    /// The global keys are those that are close to the root of the tree. By construction these
    /// are the ancestors of the coarse tree leafs, where as the coarse tree leafs themselves are
    /// the first level of keys distributed across ranks. Ghost keys are keys that are not local to
    /// the current rank but lie along the interface to the current rank. Their identifiers store the value
    /// of the rank that they originate from.
    pub fn all_keys(&self) -> &HashMap<MortonKey, KeyType> {
        &self.all_keys
    }

    /// Get the neighbour map.
    ///
    /// Returns a hash map that contains as keys all the keys obtained from [Octree::all_keys] except
    /// those that are of type [KeyType::Ghost]. The values are the neighbours of the key.
    pub fn neighbour_map(&self) -> &HashMap<MortonKey, Vec<MortonKey>> {
        &self.neighbours
    }
}

/// Test if an array of keys are the leafs of a complete linear and balanced tree.
pub fn is_complete_linear_and_balanced<C: CommunicatorCollectives>(
    arr: &[MortonKey],
    comm: &C,
) -> bool {
    // Send the tree to the root node and check there that it is balanced.

    let mut balanced = false;

    if let Some(arr) = gather_to_root(arr, comm) {
        balanced = MortonKey::is_complete_linear_and_balanced(&arr);
    }

    comm.process_at_rank(0).broadcast_into(&mut balanced);

    balanced
}

/// Compute the global bounding box across all points on all processes.
pub fn compute_global_bounding_box<C: CommunicatorCollectives>(
    points: &[Point],
    comm: &C,
) -> PhysicalBox {
    // Make sure that the points array is a multiple of 3.

    // Now compute the minimum and maximum across each dimension.

    let mut xmin = f64::MAX;
    let mut xmax = f64::MIN;

    let mut ymin = f64::MAX;
    let mut ymax = f64::MIN;

    let mut zmin = f64::MAX;
    let mut zmax = f64::MIN;

    for point in points {
        let x = point.coords()[0];
        let y = point.coords()[1];
        let z = point.coords()[2];

        xmin = f64::min(xmin, x);
        xmax = f64::max(xmax, x);

        ymin = f64::min(ymin, y);
        ymax = f64::max(ymax, y);

        zmin = f64::min(zmin, z);
        zmax = f64::max(zmax, z);
    }

    let mut global_xmin = 0.0;
    let mut global_xmax = 0.0;

    let mut global_ymin = 0.0;
    let mut global_ymax = 0.0;

    let mut global_zmin = 0.0;
    let mut global_zmax = 0.0;

    comm.all_reduce_into(&xmin, &mut global_xmin, SystemOperation::min());
    comm.all_reduce_into(&xmax, &mut global_xmax, SystemOperation::max());

    comm.all_reduce_into(&ymin, &mut global_ymin, SystemOperation::min());
    comm.all_reduce_into(&ymax, &mut global_ymax, SystemOperation::max());

    comm.all_reduce_into(&zmin, &mut global_zmin, SystemOperation::min());
    comm.all_reduce_into(&zmax, &mut global_zmax, SystemOperation::max());

    let xdiam = global_xmax - global_xmin;
    let ydiam = global_ymax - global_ymin;
    let zdiam = global_zmax - global_zmin;

    let xmean = global_xmin + 0.5 * xdiam;
    let ymean = global_ymin + 0.5 * ydiam;
    let zmean = global_zmin + 0.5 * zdiam;

    // We increase diameters by box size on deepest level
    // and use the maximum diameter to compute a
    // cubic bounding box.

    let deepest_box_diam = 1.0 / (1 << DEEPEST_LEVEL) as f64;

    let max_diam = [xdiam, ydiam, zdiam].into_iter().reduce(f64::max).unwrap();

    let max_diam = max_diam * (1.0 + deepest_box_diam);

    PhysicalBox::new([
        xmean - 0.5 * max_diam,
        ymean - 0.5 * max_diam,
        zmean - 0.5 * max_diam,
        xmean + 0.5 * max_diam,
        ymean + 0.5 * max_diam,
        zmean + 0.5 * max_diam,
    ])
}
