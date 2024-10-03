//! Test the computation of a global bounding box across MPI ranks.

use bempp_octree::{
    constants::DEEPEST_LEVEL,
    morton::MortonKey,
    octree::{
        balance, compute_coarse_tree, compute_coarse_tree_weights, create_local_tree,
        is_complete_linear_tree, linearize, load_balance, points_to_morton,
        redistribute_points_with_respect_to_coarse_tree, redistribute_with_respect_to_coarse_tree,
    },
    tools::{communicate_back, generate_random_points, global_size, is_sorted_array},
};
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives},
};
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

    let points = generate_random_points(npoints, &mut rng, &comm);

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

    let load_balanced_coarse_keys = load_balance(&coarse_tree, &weights, &comm);

    // Compute the weights of the balanced keys

    let load_balanced_weights =
        compute_coarse_tree_weights(&linear_keys, &load_balanced_coarse_keys, &comm);

    let mut global_balanced_weight: usize = 0;
    comm.all_reduce_into(
        &(load_balanced_weights.iter().sum::<usize>()),
        &mut global_balanced_weight,
        SystemOperation::sum(),
    );

    // The global weight of the non-balanced keys should be identical
    // to the global weigth of the balanced keys.

    assert_eq!(global_weight, global_balanced_weight);

    // Now compute the new fine keys.

    let load_balanced_fine_keys =
        redistribute_with_respect_to_coarse_tree(&linear_keys, &load_balanced_coarse_keys, &comm);

    assert_eq!(
        global_size(&load_balanced_fine_keys, &comm),
        global_size(&linear_keys, &comm)
    );

    let refined_tree =
        create_local_tree(&load_balanced_fine_keys, &load_balanced_coarse_keys, 6, 100);

    assert!(is_complete_linear_tree(&refined_tree, &comm));

    // Now balance the tree.

    let balanced_tree = balance(&refined_tree, &mut rng, &comm);

    // redistribute the balanced tree according to coarse tree

    let balanced_tree =
        redistribute_with_respect_to_coarse_tree(&balanced_tree, &load_balanced_coarse_keys, &comm);

    assert!(is_complete_linear_tree(&balanced_tree, &comm));

    // Redistribute original keys and points with respect to balanced coarse tree.

    let (balanced_points, balanced_keys) = redistribute_points_with_respect_to_coarse_tree(
        &points,
        &keys,
        &load_balanced_coarse_keys,
        &comm,
    );

    let upper_bound;

    if let Some(next_key) = communicate_back(&load_balanced_coarse_keys, &comm) {
        upper_bound = next_key;
    } else {
        upper_bound = MortonKey::upper_bound();
    }

    assert!(load_balanced_coarse_keys.first().unwrap() <= balanced_keys.first().unwrap());
    assert!(*balanced_keys.last().unwrap() < upper_bound);
    assert!(is_sorted_array(&balanced_keys, &comm));

    println!(
        "Rank {} has {} balanced points.",
        comm.rank(),
        balanced_points.len(),
    );

    if comm.rank() == 0 {
        println!("Coarse tree successfully created and weights computed.");
    }
}
