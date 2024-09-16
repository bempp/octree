//! Testing the hyksort component.
use bempp_octree::constants::{DEEPEST_LEVEL, LEVEL_SIZE};
use bempp_octree::morton::MortonKey;
use bempp_octree::parallel_octree::{block_partition, is_sorted_array, linearize, partition};
use bempp_octree::parsort::{array_to_root, parsort};
use itertools::{izip, Itertools};
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

pub fn generate_random_keys<R: Rng>(nkeys: usize, rng: &mut R) -> Vec<MortonKey> {
    let mut result = Vec::<MortonKey>::with_capacity(nkeys);

    let xindices = rand::seq::index::sample(rng, LEVEL_SIZE as usize, nkeys);
    let yindices = rand::seq::index::sample(rng, LEVEL_SIZE as usize, nkeys);
    let zindices = rand::seq::index::sample(rng, LEVEL_SIZE as usize, nkeys);

    for (xval, yval, zval) in izip!(xindices.iter(), yindices.iter(), zindices.iter()) {
        result.push(MortonKey::from_index_and_level(
            [xval, yval, zval],
            DEEPEST_LEVEL as usize,
        ));
    }

    result
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
    let rank = comm.rank();
    let keys = if rank == 0 {
        generate_random_keys(50, rng)
    } else {
        generate_random_keys(1000, rng)
    };

    // We now linearize the keys.

    let mut keys = linearize(&keys, rng, comm);

    // We move most keys over from rank 0 to rank 2 to check how the partitioning works.

    let nsend = 400;
    // Send the last 200 keys from rank 0 to rank 1.

    if rank == 0 {
        let send_keys = &keys[keys.len() - nsend..keys.len()];
        comm.process_at_rank(1).send(send_keys);
        keys = keys[0..keys.len() - nsend].to_vec();
    }

    if rank == 1 {
        let mut recv_keys = vec![MortonKey::default(); nsend];
        comm.process_at_rank(0).receive_into(&mut recv_keys);
        recv_keys.extend(keys.iter());
        keys = recv_keys;
    }

    println!("Rank {} has {} keys. ", rank, keys.len());

    let partitioned_tree = block_partition(&keys, rng, comm);

    println!(
        "Partitioned tree on rank {} has {} keys.",
        rank,
        partitioned_tree.0.len()
    );

    let arr = array_to_root(&partitioned_tree.0, comm);

    if rank == 0 {
        let arr = arr.unwrap();
        for (elem1, elem2) in arr.iter().tuple_windows() {
            assert!(*elem1 <= *elem2);
        }
        println!("Keys are sorted.");
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
