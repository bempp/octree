//! Testing the hyksort component.
use bempp_octree::{
    parsort::parsort,
    tools::{generate_random_keys, is_sorted_array},
};
use mpi::traits::Communicator;
use rand::prelude::*;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let n_per_rank = 1000;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let keys = generate_random_keys(n_per_rank, &mut rng);

    let arr = parsort(&keys, &world, &mut rng);

    assert!(is_sorted_array(&arr, &world));

    if world.rank() == 0 {
        println!("Array is sorted.");
    }
}
