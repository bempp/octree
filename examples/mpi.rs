//! Testing the hyksort component.
use bempp_octree::morton::MortonKey;
use bempp_octree::parallel_octree::{block_partition, is_sorted_array, linearize, partition};
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

pub fn generate_random_tree<R: Rng>(max_level: usize, rng: &mut R) -> Vec<MortonKey> {
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

    let mut keys = Vec::<MortonKey>::new();
    add_level(&mut keys, MortonKey::root(), rng, max_level);

    keys
}

pub fn test_linearize<R: Rng, C: CommunicatorCollectives>(rng: &mut R, comm: &C) {
    let max_level = 6;
    let keys = generate_random_tree(max_level, rng);
    let rank = comm.rank();

    // We now linearize the keys.

    if rank == 0 {
        println!("Linearizing keys.");
    }
    let sorted_keys = linearize(&keys, rng, comm);

    // Now check that the tree is properly linearized.

    assert_linearized(&sorted_keys, comm);
    if rank == 0 {
        println!("Linearization successful.");
    }

    // Now form the coarse tree
}

pub fn test_coarse_partition<R: Rng, C: CommunicatorCollectives>(rng: &mut R, comm: &C) {
    let max_level = 6;
    let keys = generate_random_tree(max_level, rng);
    let rank = comm.rank();

    let arr = array_to_root(&keys, comm);

    if rank == 0 {
        let arr = arr.unwrap();
        println!("Fine tree has {} elements", arr.len());
    }

    // We now linearize the keys.

    let keys = linearize(&keys, rng, comm);

    let coarse_tree = block_partition(&keys, rng, comm);

    println!(
        "Coarse tree on rank {} has {} keys.",
        rank,
        coarse_tree.len()
    );

    let arr = array_to_root(&coarse_tree, comm);

    if rank == 0 {
        let arr = arr.unwrap();
        println!("Coarse tree has {} keys", arr.len());
        assert!(MortonKey::is_complete_linear_octree(&arr));
        println!("Coarse tree is sorted and complete.");
    }
}

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let rank = comm.rank() as u64;
    // Each process gets its own rng
    let mut rng = rand::rngs::StdRng::seed_from_u64(rank as u64);
    test_linearize(&mut rng, &comm);
    test_coarse_partition(&mut rng, &comm);
}
