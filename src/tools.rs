//! Utility routines.

use itertools::{izip, Itertools};
use mpi::{
    collective::{SystemOperation, UserOperation},
    datatype::{Partition, PartitionMut},
    point_to_point as p2p,
    traits::{CommunicatorCollectives, Destination, Equivalence, Root, Source},
};
use num::traits::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    constants::{DEEPEST_LEVEL, LEVEL_SIZE},
    geometry::Point,
    morton::MortonKey,
};

/// Gather array to all processes
pub fn gather_to_all<T: Equivalence, C: CommunicatorCollectives>(arr: &[T], comm: &C) -> Vec<T> {
    // First we need to broadcast the individual sizes on each process.

    let size = comm.size();

    let local_len = arr.len() as i32;

    let mut sizes = vec![0; size as usize];

    comm.all_gather_into(&local_len, &mut sizes);

    let recv_len = sizes.iter().sum::<i32>() as usize;

    // Now we have the size of each local contribution.
    // let mut recvbuffer =
    //     vec![T: Default; counts_from_processor.iter().sum::<i32>() as usize];
    let mut recvbuffer = Vec::<T>::with_capacity(recv_len);
    let buf: &mut [T] = unsafe { std::mem::transmute(recvbuffer.spare_capacity_mut()) };

    let recv_displs: Vec<i32> = sizes
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    let mut receiv_partition = PartitionMut::new(buf, sizes, &recv_displs[..]);

    comm.all_gather_varcount_into(arr, &mut receiv_partition);

    unsafe { recvbuffer.set_len(recv_len) };

    recvbuffer
}
/// Array to root

/// Gather distributed array to the root rank.
///
/// The result is a `Vec<T>` on root and `None` on all other ranks.
pub fn gather_to_root<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<Vec<T>> {
    let n = arr.len() as i32;
    let rank = comm.rank();
    let size = comm.size();
    let root_process = comm.process_at_rank(0);

    // We first communicate the length of the array to root.

    if rank == 0 {
        // We are at root.

        let mut counts = vec![0_i32; size as usize];
        root_process.gather_into_root(&n, &mut counts);

        // We now have all ranks at root. Can now a varcount gather to get
        // the array elements.

        let nelements = counts.iter().sum::<i32>();
        let mut new_arr = Vec::<T>::with_capacity(nelements as usize);
        let new_arr_buf: &mut [T] = unsafe { std::mem::transmute(new_arr.spare_capacity_mut()) };

        let displs = displacements(counts.as_slice());

        let mut partition = PartitionMut::new(new_arr_buf, counts, &displs[..]);

        root_process.gather_varcount_into_root(arr, &mut partition);

        unsafe { new_arr.set_len(nelements as usize) };
        Some(new_arr)
    } else {
        root_process.gather_into(&n);
        root_process.gather_varcount_into(arr);
        None
    }
}

/// Get global size of a distributed array.
///
/// Computes the size and broadcoasts it to all ranks.
pub fn global_size<T, C: CommunicatorCollectives>(arr: &[T], comm: &C) -> usize {
    let local_size = arr.len();
    let mut global_size = 0;

    comm.all_reduce_into(&local_size, &mut global_size, SystemOperation::sum());

    global_size
}

/// Get the maximum value across all ranks
pub fn global_max<T: Equivalence + Copy + Ord, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> T {
    let local_max = arr.iter().max().unwrap();

    // Just need to initialize global_max with something.
    let mut global_max = *local_max;

    comm.all_reduce_into(
        local_max,
        &mut global_max,
        &UserOperation::commutative(|x, y| {
            let x: &[T] = x.downcast().unwrap();
            let y: &mut [T] = y.downcast().unwrap();
            for (&x_i, y_i) in x.iter().zip(y) {
                *y_i = x_i.max(*y_i);
            }
        }),
    );

    global_max
}

/// Get the minimum value across all ranks
pub fn global_min<T: Equivalence + Copy + Ord, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> T {
    let local_min = *arr.iter().min().unwrap();

    // Just need to initialize global_min with something.
    let mut global_min = local_min;

    comm.all_reduce_into(
        &local_min,
        &mut global_min,
        &UserOperation::commutative(|x, y| {
            let x: &[T] = x.downcast().unwrap();
            let y: &mut [T] = y.downcast().unwrap();
            for (&x_i, y_i) in x.iter().zip(y) {
                *y_i = x_i.min(*y_i);
            }
        }),
    );

    global_min
}

/// Communicate the first element of each local array back to the previous rank.
pub fn communicate_back<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<T> {
    let rank = comm.rank();
    let size = comm.size();

    if rank == size - 1 {
        comm.process_at_rank(rank - 1).send(arr.first().unwrap());
        None
    } else {
        let (new_last, _status) = if rank > 0 {
            p2p::send_receive(
                arr.first().unwrap(),
                &comm.process_at_rank(rank - 1),
                &comm.process_at_rank(rank + 1),
            )
        } else {
            comm.process_at_rank(1).receive::<T>()
        };
        Some(new_last)
    }
}

/// Check if an array is sorted.
pub fn is_sorted_array<T: Equivalence + PartialOrd, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> bool {
    let mut sorted = true;
    for (elem1, elem2) in arr.iter().tuple_windows() {
        if elem1 > elem2 {
            sorted = false;
        }
    }

    if comm.size() == 1 {
        return sorted;
    }

    if let Some(next_first) = communicate_back(arr, comm) {
        sorted = *arr.last().unwrap() <= next_first;
    }

    let mut global_sorted: bool = false;
    comm.all_reduce_into(&sorted, &mut global_sorted, SystemOperation::logical_and());

    global_sorted
}

/// Redistribute an array via an all_to_all_varcount operation.
pub fn redistribute<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    counts: &[i32],
    comm: &C,
) -> Vec<T> {
    assert_eq!(counts.len(), comm.size() as usize);

    // First send the counts around via an alltoall operation.

    let mut recv_counts = vec![0; counts.len()];

    comm.all_to_all_into(counts, &mut recv_counts);

    // We have the recv_counts. Allocate space and setup the partitions.

    let nelems = recv_counts.iter().sum::<i32>() as usize;

    let mut output = Vec::<T>::with_capacity(nelems);
    let out_buf: &mut [T] = unsafe { std::mem::transmute(output.spare_capacity_mut()) };

    let send_partition = Partition::new(arr, counts, displacements(counts));
    let mut recv_partition =
        PartitionMut::new(out_buf, &recv_counts[..], displacements(&recv_counts));

    comm.all_to_all_varcount_into(&send_partition, &mut recv_partition);

    unsafe { output.set_len(nelems) };

    output
}

/// Perform a global inclusive cumulative sum operation.
///
/// For the array `[1, 3, 5, 7]` the output will be `[1, 4, 9, 16]`.
pub fn global_inclusive_cumsum<T: Equivalence + Zero + Copy, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Vec<T> {
    let mut scan: Vec<T> = arr
        .iter()
        .scan(<T as Zero>::zero(), |state, x| {
            *state = *x + *state;
            Some(*state)
        })
        .collect_vec();
    let scan_last = *scan.last().unwrap();
    let mut scan_result = T::zero();
    comm.exclusive_scan_into(&scan_last, &mut scan_result, SystemOperation::sum());
    for elem in &mut scan {
        *elem = *elem + scan_result;
    }

    scan
}

/// Distribute a sorted sequence into bins.
///
/// For an array with n elements to be distributed into p bins,
/// the array `bins` has p elements. The bins are defined by half-open intervals
/// of the form [b_j, b_{j+1})). The final bin is the half-open interval [b_{p-1}, \infty).
/// It is assumed that the bins and the elements are both sorted sequences and that
/// every element has an associated bin.
/// The function returns a p element array with the counts of how many elements go to each bin.
/// Since the sequence is sorted this fully defines what element goes into which bin.
pub fn sort_to_bins<T: Ord>(sorted_keys: &[T], bins: &[T]) -> Vec<usize> {
    let nbins = bins.len();

    // Make sure that the smallest element of the sorted keys fits into the bins.
    assert!(bins.first().unwrap() <= sorted_keys.first().unwrap());

    // Deal with the special case that there is only one bin.
    // This means that all elements are in the one bin.
    if nbins == 1 {
        return vec![sorted_keys.len(); 1];
    }

    let mut bin_counts = vec![0; nbins];

    // This iterates over each possible bin and returns also the associated rank.
    // The last bin position is not iterated over since for an array with p elements
    // there are p-1 tuple windows.
    let mut bin_iter = izip!(
        bin_counts.iter_mut(),
        bins.iter().tuple_windows::<(&T, &T)>(),
    );

    // We take the first element of the bin iterator. There will always be at least one since
    // there are at least two bins (an actual one, and the last half infinite one)
    let mut r: &mut usize;
    let mut bin_start: &T;
    let mut bin_end: &T;
    (r, (bin_start, bin_end)) = bin_iter.next().unwrap();

    let mut count = 0;
    'outer: for key in sorted_keys.iter() {
        if bin_start <= key && key < bin_end {
            *r += 1;
            count += 1;
        } else {
            // Move the bin forward until it fits. There will always be a fitting bin.
            loop {
                if let Some((rn, (bsn, ben))) = bin_iter.next() {
                    if bsn <= key && key < ben {
                        // We have found the next fitting bin for our current element.
                        // Can register it and go back to the outer for loop.
                        *rn += 1;
                        r = rn;
                        bin_start = bsn;
                        bin_end = ben;
                        count += 1;
                        break;
                    }
                } else {
                    // We have no more fitting bin. So break the outer loop.
                    break 'outer;
                }
            }
        }
    }

    // We now have everything but the last bin. Just bunch the remaining elements to
    // the last count.
    *bin_counts.last_mut().unwrap() = sorted_keys.len() - count;

    bin_counts
}

/// Redistribute locally sorted keys with respect to bins.
///
/// - The array `sorted_keys` is assumed to be sorted within each process. It needs not be globally sorted.
/// - If there are `r` ranks in the communicator, the size of `bins` must be `r`.
/// - The bins are defined through half-open intervals (bin[0], bin[1]), .... This defines r-1 bins. The
///   last bin is the half-open interval [bin[r-1], \infty).
/// - All array elements must be larger or equal bin[0]. This means that each element can be sorted into a bin.
pub fn redistribute_by_bins<T: Equivalence + Ord, C: CommunicatorCollectives>(
    sorted_keys: &[T],
    bins: &[T],
    comm: &C,
) -> Vec<T> {
    let counts = sort_to_bins(sorted_keys, bins);
    let counts = counts.iter().map(|elem| *elem as i32).collect_vec();
    redistribute(sorted_keys, &counts, comm)
}

/// Generate random keys for testing.
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

/// Generate random points for testing.
pub fn generate_random_points<R: Rng, C: CommunicatorCollectives>(
    npoints: usize,
    rng: &mut R,
    comm: &C,
) -> Vec<Point> {
    let mut points = Vec::<Point>::with_capacity(npoints);
    let rank = comm.rank() as usize;

    for index in 0..npoints {
        points.push(Point::new(
            [rng.gen(), rng.gen(), rng.gen()],
            npoints * rank + index,
        ));
    }

    points
}

/// Get a seeded rng
pub fn seeded_rng(seed: usize) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed as u64)
}

/// Compute displacements from a vector of counts.
///
/// This is useful for global MPI varcount operations. Let
/// count [ 3, 4, 5]. Then the corresponding displacements are
// [0, 3, 7]. Note that the last element `5` is ignored.
pub fn displacements(counts: &[i32]) -> Vec<i32> {
    counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::sort_to_bins;

    #[test]
    fn test_sort_to_bins() {
        let elems = (0..100).collect_vec();
        let bins = [0, 17, 55];

        let counts = sort_to_bins(&elems, &bins);

        assert_eq!(counts[0], 17);
        assert_eq!(counts[1], 38);
        assert_eq!(counts[2], 45);
    }
}
