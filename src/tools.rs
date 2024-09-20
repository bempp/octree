//! Utility routines.

use itertools::Itertools;
use mpi::{
    collective::SystemOperation,
    datatype::{Partition, PartitionMut},
    point_to_point as p2p,
    traits::{
        CommunicatorCollectives, Destination, Equivalence, PartitionedBuffer, PartitionedBufferMut,
        Root, Source,
    },
};

/// Gather array to all processes
pub fn gather_to_all<T: Equivalence, C: CommunicatorCollectives>(arr: &[T], comm: &C) -> Vec<T> {
    // First we need to broadcast the individual sizes on each process.

    let size = comm.size();

    let local_len = arr.len() as i32;

    let mut sizes = vec![0 as i32; size as usize];

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

/// Communicate the first element of each local array back to the previous rank.
pub fn communicate_back<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    comm: &C,
) -> Option<T> {
    let rank = comm.rank();
    let size = comm.size();

    if rank == size - 1 {
        comm.process_at_rank(rank - 1).send(arr.first().unwrap());
        return None;
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

    let mut recv_counts = vec![0 as i32; counts.len()];

    comm.all_to_all_into(&counts[..], &mut recv_counts);

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
