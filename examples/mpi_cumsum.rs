//! Test the computation of a global bounding box across MPI ranks.

use bempp_octree::tools::{gather_to_root, global_inclusive_cumsum};
use itertools::{izip, Itertools};
use mpi::traits::*;
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
    let nelems = 10;

    // Generate random numbers

    let mut elems = Vec::<usize>::with_capacity(nelems);

    for _ in 0..nelems {
        elems.push(rng.gen_range(0..100));
    }

    // Compute the cumulative sum.

    let global_cum_sum = global_inclusive_cumsum(&elems, &comm);

    // Copy array to root and compare with inclusive scan there.

    if let (Some(cum_sum_root), Some(original_array)) = (
        gather_to_root(&global_cum_sum, &comm),
        gather_to_root(&elems, &comm),
    ) {
        // Scan on root

        let expected_cum_sum = original_array
            .iter()
            .scan(0, |state, x| {
                *state = *x + *state;
                Some(*state)
            })
            .collect_vec();

        // Check that the first element is not modified (inclusive cumsum)
        assert_eq!(
            original_array.first().unwrap(),
            cum_sum_root.first().unwrap()
        );

        for (actual, expected) in izip!(cum_sum_root.iter(), expected_cum_sum.iter()) {
            assert_eq!(*actual, *expected);
        }

        println!("Cumulative sum computed.");
    }
}
