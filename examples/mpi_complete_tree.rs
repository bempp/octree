//! Test the computation of a global bounding box across MPI ranks.

use bempp_octree::{
    constants::DEEPEST_LEVEL,
    octree::{complete_tree, is_complete_linear_tree, linearize, points_to_morton},
    tools::generate_random_points,
};
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
    let npoints = 10;

    // Generate random points.

    let points = generate_random_points(npoints, &mut rng, &comm);

    // Compute the Morton keys on the deepest level
    let (keys, _) = points_to_morton(&points, DEEPEST_LEVEL as usize, &comm);

    let linear_keys = linearize(&keys, &mut rng, &comm);

    // Generate a complete tree
    let distributed_complete_tree = complete_tree(&linear_keys, &comm);

    assert!(is_complete_linear_tree(&distributed_complete_tree, &comm));

    if comm.rank() == 0 {
        println!("Distributed tree is complete and linear.");
    }
}
