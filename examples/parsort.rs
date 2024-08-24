//! Testing the hyksort component.
use bempp_octree::parsort::{get_buckets, get_counts, get_global_min_max, to_unique_item};
use mpi;
use mpi::traits::{Communicator, Root};
use rand::prelude::*;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as u64;
    let size = world.size() as u64;
    let root = world.process_at_rank(0);
    let n_per_rank = 20;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let mut arr = Vec::<u64>::new();

    for index in 0..n_per_rank {
        arr.push(n_per_rank * rank + index as u64);
    }

    // let splitters = get_splitters(&arr, &world, &mut rng);

    // let bin_displs = get_bin_displacements(&arr, &splitters);

    let mut arr = to_unique_item(&arr, rank as usize);
    arr.sort_unstable();

    let buckets = get_buckets(&arr, &world, &mut rng);

    let counts = get_counts(&arr, &buckets);

    if rank == 2 {
        for (index, item) in buckets.iter().enumerate() {
            println!("Bucket {}, {}", index, item);
        }

        println!("Counts: {:#?}", counts);

        for elem in arr {
            println!("Value: {}", elem.value);
        }
    }
}
