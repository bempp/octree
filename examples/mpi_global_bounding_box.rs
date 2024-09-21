//! Test the computation of a global bounding box across MPI ranks.

use bempp_octree::{
    geometry::PhysicalBox, octree::compute_global_bounding_box, tools::gather_to_root,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub fn main() {
    // Initialise MPI
    let universe = mpi::initialize().unwrap();

    // Get the world communicator
    let comm = universe.world();

    // Initialise a seeded Rng.
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Create `npoints` per rank.
    let npoints = 10;

    // Generate random points.

    let mut points = Vec::<f64>::with_capacity(3 * npoints);

    for _ in 0..3 * npoints {
        points.push(rng.gen());
    }

    // Compute the distributed bounding box.

    let bounding_box = compute_global_bounding_box(&points, &comm);

    // Copy all points to root and compare local bounding box there.

    if let Some(points_root) = gather_to_root(&points, &comm) {
        // Compute the bounding box on root.

        let expected = PhysicalBox::from_points(&points_root);
        assert_eq!(expected.coordinates(), bounding_box.coordinates());
    }
}
