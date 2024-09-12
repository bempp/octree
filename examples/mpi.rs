//! Testing the hyksort component.
use bempp_octree::morton::MortonKey;
use bempp_octree::parallel_octree::partition;
use bempp_octree::parsort::{array_to_root, parsort};
use itertools::Itertools;
use mpi::traits::Communicator;
use rand::prelude::*;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as u64;
    let n_per_rank = 10;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let mut arr = Vec::<MortonKey>::new();
    let mut weights = Vec::<usize>::new();

    for index in n_per_rank * rank..n_per_rank * (rank + 1) {
        arr.push(MortonKey::from_index_and_level([index as usize, 0, 0], 10));
    }

    let arr = parsort(&arr, &world, &mut rng);

    for index in 0..arr.len() {
        weights.push((rank * n_per_rank) as usize + index);
    }

    // let t = n_per_rank * rank as usize;
    // let mut index_sum = if rank == 0 { 0 } else { (t * (t - 1)) / 2 };
    // for index in n_per_rank * (rank as usize)..(n_per_rank * (1 + rank as usize)) {
    //     arr.push(MortonKey::from_index_and_level([0, 0, 0], 0));
    //     weights.push(index_sum);
    //     index_sum += index;
    //     // weights.push(rng.gen_range(1..20));
    // }

    let partitioned = partition(&arr, &weights, &world);

    println!("Rank: {}, Len: {}", rank, partitioned.len());

    let arr = array_to_root(&partitioned, &world);

    if rank == 0 {
        let arr = arr.unwrap();

        for (elem1, elem2) in arr.iter().tuple_windows() {
            assert!(elem1 <= elem2);
        }
        println!("{} elements are sorted.", arr.len());
        println!("Finished.");
    }
}
