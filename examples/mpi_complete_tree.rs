//! Test the computation of a global bounding box across MPI ranks.

use bempp_octree::{
    constants::DEEPEST_LEVEL,
    geometry::PhysicalBox,
    octree::{
        complete_tree, compute_global_bounding_box, is_complete_linear_tree, points_to_morton,
    },
    tools::gather_to_root,
};
use mpi::traits::*;
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

    let mut points = Vec::<f64>::with_capacity(3 * npoints);

    for _ in 0..3 * npoints {
        points.push(rng.gen());
    }

    // Compute the Morton keys on the deepest level
    let (keys, _) = points_to_morton(&points, DEEPEST_LEVEL as usize, &comm);

    assert!(!is_complete_linear_tree(&keys, &comm));

    // Generate a complete tree
    let distributed_complete_tree = complete_tree(&keys, &mut rng, &comm);

    let is_complete_linear = is_complete_linear_tree(&distributed_complete_tree, &comm);

    assert!(is_complete_linear);

    if comm.rank() == 0 {
        println!("Distributed tree is complete and linear.");
    }
}
