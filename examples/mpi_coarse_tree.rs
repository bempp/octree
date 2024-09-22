//! Test the computation of a global bounding box across MPI ranks.

use bempp_octree::{
    constants::DEEPEST_LEVEL,
    octree::{
        complete_tree, compute_coarse_tree, compute_coarse_tree_weights, is_complete_linear_tree,
        linearize, load_balance, points_to_morton, redistribute_with_respect_to_coarse_tree,
    },
    tools::global_size,
};
use mpi::{collective::SystemOperation, traits::*};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub fn main() {
    // Initialise MPI
    let universe = mpi::initialize().unwrap();

    // Get the world communicator
    let comm = universe.world();

    // Initialise a seeded Rng.
    let mut rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    // Create `npoints` per rank.
    let npoints = 10000;

    // Generate random points.

    let mut points = Vec::<f64>::with_capacity(3 * npoints);

    for _ in 0..3 * npoints {
        points.push(rng.gen());
    }

    // Compute the Morton keys on the deepest level
    let (keys, _) = points_to_morton(&points, DEEPEST_LEVEL as usize, &comm);

    // linearize the keys
    let linear_keys = linearize(&keys, &mut rng, &comm);

    // Generate the coarse tree
    let coarse_tree = compute_coarse_tree(&linear_keys, &comm);
    assert!(is_complete_linear_tree(&coarse_tree, &comm));

    // We now compute the weights for the coarse tree.

    let weights = compute_coarse_tree_weights(&linear_keys, &coarse_tree, &comm);

    // Assert that the global sum of the weights is identical to the number of linearized keys.

    let mut global_weight: usize = 0;

    comm.all_reduce_into(
        &(weights.iter().sum::<usize>()),
        &mut global_weight,
        SystemOperation::sum(),
    );

    assert_eq!(global_weight, global_size(&linear_keys, &comm));

    // Now load balance the coarse tree

    let balanced_keys = load_balance(&coarse_tree, &weights, &comm);

    // Compute the weights of the balanced keys

    let balanced_weights = compute_coarse_tree_weights(&linear_keys, &balanced_keys, &comm);

    let mut global_balanced_weight: usize = 0;
    comm.all_reduce_into(
        &(balanced_weights.iter().sum::<usize>()),
        &mut global_balanced_weight,
        SystemOperation::sum(),
    );

    // The global weight of the non-balanced keys should be identical
    // to the global weigth of the balanced keys.

    assert_eq!(global_weight, global_balanced_weight);

    // Now compute the new fine keys.

    let redistributed_fine_keys =
        redistribute_with_respect_to_coarse_tree(&linear_keys, &balanced_keys, &comm);

    assert_eq!(
        global_size(&redistributed_fine_keys, &comm),
        global_size(&linear_keys, &comm)
    );

    if comm.rank() == 0 {
        println!("Coarse tree successfully created and weights computed.");
    }
}
