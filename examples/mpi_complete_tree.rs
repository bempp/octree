//! Test the computation of a complete octree.

use bempp_octree::{
    morton::MortonKey,
    octree::{is_complete_linear_and_balanced, KeyType, Octree},
    tools::{gather_to_all, generate_random_points},
};
use itertools::Itertools;
use mpi::traits::Communicator;
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

    let tree = Octree::new(&points, 15, 50, &comm);

    // We now check that each node of the tree has all its neighbors available.

    let leaf_tree = tree.leaf_tree();
    let all_keys = tree.all_keys();

    assert!(is_complete_linear_and_balanced(leaf_tree, &comm));
    for &key in leaf_tree {
        // We only check interior keys. Leaf keys may not have a neighbor
        // on the same level.
        let mut parent = key.parent();
        while parent.level() > 0 {
            // Check that the key itself is there.
            assert!(all_keys.contains_key(&key));
            // Check that all its neighbours are there.
            for neighbor in parent.neighbours().iter().filter(|&key| key.is_valid()) {
                assert!(all_keys.contains_key(neighbor));
            }
            parent = parent.parent();
            // Check that the parent is there.
            assert!(all_keys.contains_key(&parent));
        }
    }

    // At the end check that the root of the tree is also contained.
    assert!(all_keys.contains_key(&MortonKey::root()));

    // Count the number of ghosts on each rank
    // Count the number of global keys on each rank.

    // Assert that all ghosts are from a different rank and count them.

    let nghosts = all_keys
        .iter()
        .filter_map(|(_, &value)| {
            if let KeyType::Ghost(rank) = value {
                assert!(rank != comm.size() as usize);
                Some(rank)
            } else {
                None
            }
        })
        .count();

    if comm.size() == 0 {
        assert_eq!(nghosts, 0);
    } else {
        assert!(nghosts > 0);
    }

    let nglobal = all_keys
        .iter()
        .filter(|(_, &value)| matches!(value, KeyType::Global))
        .count();

    // Assert that all globals across all ranks have the same count.

    let nglobals = gather_to_all(std::slice::from_ref(&nglobal), &comm);

    assert_eq!(nglobals.iter().unique().count(), 1);

    if comm.rank() == 0 {
        println!("Distributed tree is complete and linear.");
    }
}
