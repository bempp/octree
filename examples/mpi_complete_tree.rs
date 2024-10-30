//! Demonstrate the instantiation of a complete octree using MPI.

use std::time::Instant;

use bempp_octree::{generate_random_points, Octree};
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
    let npoints = 1000000;

    // Generate random points on the positive octant of the unit sphere.

    let mut points = generate_random_points(npoints, &mut rng, &comm);
    // Make sure that the points live on the unit sphere.
    for point in points.iter_mut() {
        let len = point.coords()[0] * point.coords()[0]
            + point.coords()[1] * point.coords()[1]
            + point.coords()[2] * point.coords()[2];
        let len = len.sqrt();
        point.coords_mut()[0] /= len;
        point.coords_mut()[1] /= len;
        point.coords_mut()[2] /= len;
    }

    let start = Instant::now();
    // The following code will create a complete octree with a maximum level of 16.
    let octree = Octree::new(&points, 16, 50, &comm);
    let duration = start.elapsed();

    let global_number_of_points = octree.global_number_of_points();
    let global_max_level = octree.global_max_level();

    // We now check that each node of the tree has all its neighbors available.

    if comm.rank() == 0 {
        println!(
            "Setup octree with {} points and maximum level {} in {} ms",
            global_number_of_points,
            global_max_level,
            duration.as_millis()
        );
    }
}
