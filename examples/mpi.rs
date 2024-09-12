//! Testing the hyksort component.
use bempp_octree::morton::MortonKey;
use bempp_octree::parallel_octree::{linearize, partition};
use bempp_octree::parsort::{array_to_root, parsort};
use itertools::Itertools;
use mpi::traits::*;
use rand::prelude::*;

pub fn assert_linearized<C: CommunicatorCollectives>(arr: &Vec<MortonKey>, comm: &C) {
    // Check that the keys are still linearized.
    let arr = array_to_root(&arr, comm);

    if comm.rank() == 0 {
        let arr = arr.unwrap();
        for (&elem1, &elem2) in arr.iter().tuple_windows() {
            assert!(!elem1.is_ancestor(elem2));
        }
        println!("{} keys are linearized.", &arr.len());
    }
}

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as u64;
    let max_level = 6;

    // Each process gets its own rng
    let mut rng = rand::rngs::StdRng::seed_from_u64(rank as u64);

    // We first create a non-uniform tree on rank 0.

    let mut keys = Vec::<MortonKey>::new();

    pub fn add_level<R: Rng>(
        keys: &mut Vec<MortonKey>,
        current: MortonKey,
        rng: &mut R,
        max_level: usize,
    ) {
        keys.push(current);

        if current.level() >= max_level {
            return;
        }

        let mut children = current.children();

        // This makes sure that the tree is not sorted.
        children.shuffle(rng);

        for child in children {
            if rng.gen_bool(0.9) {
                add_level(keys, child, rng, max_level);
            }
        }
    }

    add_level(&mut keys, MortonKey::root(), &mut rng, max_level);

    println!("Number of keys on rank {}: {}", rank, keys.len());

    // We now linearize the keys.

    if rank == 0 {
        println!("Linearizing keys.");
    }
    let sorted_keys = linearize(&keys, &mut rng, &world);

    println!(
        "Number of linearized keys on rank {}: {}",
        rank,
        sorted_keys.len()
    );

    // Now check that the tree is properly linearized.

    assert_linearized(&sorted_keys, &world);

    // We now partition the keys equally across the processes. We give
    // each leaf equal weights here.

    let weights = vec![1 as usize; sorted_keys.len()];

    if rank == 0 {
        println!("Partitioning keys.");
    }

    let sorted_keys = partition(&sorted_keys, &weights, &world);

    println!(
        "After partitioning have {} keys on rank {}",
        sorted_keys.len(),
        rank
    );

    assert_linearized(&sorted_keys, &world);
}
