//! Testing the hyksort component.
use bempp_octree::parsort::{get_global_min_max, to_unique_item};
use mpi;
use mpi::traits::{Communicator, Root};
use rand::prelude::*;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as u64;
    let size = world.size() as u64;
    let root = world.process_at_rank(0);
    let n_per_rank = 5;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let mut arr = Vec::<u64>::new();

    for index in 0..n_per_rank {
        arr.push(n_per_rank * rank + index as u64);
    }

    // let splitters = get_splitters(&arr, &world, &mut rng);

    // let bin_displs = get_bin_displacements(&arr, &splitters);

    let arr = to_unique_item(&arr, rank as usize);

    let (min, max) = get_global_min_max(&arr, &world);

    if rank == 2 {
        println!("Min: {}", min);
        println!("Max: {}", max);

        // println!("Splitters: {:#?}", splitters);

        // println!("Bin displacements: {:#?}", bin_displs);
    }

    // if rank == 0 {
    //     let mut new_arr = vec![0 as u64; (n_per_rank * size) as usize];
    //     root.gather_into_root(&arr, &mut new_arr);

    //     // Print all elements on root
    //     for elem in &new_arr {
    //         println!("Elem: {}", elem);
    //     }
    // } else {
    //     root.gather_into(&arr);
    // }
}
