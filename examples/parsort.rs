//! Testing the hyksort component.
use bempp_octree::parsort::{array_to_root, parsort};
use itertools::Itertools;
use mpi::traits::Communicator;
use rand::prelude::*;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as u64;
    let n_per_rank = 1000;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let mut arr = Vec::<u64>::new();

    for _ in 0..n_per_rank {
        arr.push(rng.gen());
    }

    // let splitters = get_splitters(&arr, &world, &mut rng);

    // let bin_displs = get_bin_displacements(&arr, &splitters);

    let arr = parsort(&arr, &world, &mut rng);
    let arr = array_to_root(&arr, &world);

    if rank == 0 {
        let arr = arr.unwrap();

        for (elem1, elem2) in arr.iter().tuple_windows() {
            assert!(elem1 <= elem2);
        }
        println!("Sorted {} elements.", arr.len());
        println!("Finished.");
    }
}
