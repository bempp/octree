//! Utility routines.

use mpi::{
    collective::SystemOperation,
    datatype::PartitionMut,
    traits::{CommunicatorCollectives, Equivalence, Root},
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

/// Check if an array is sorted.
pub fn is_sorted_array<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[MortonKey],
    comm: &C,
) -> Option<bool> {
    let arr = gather_to_root(arr, comm);
    if comm.rank() == 0 {
        let arr = arr.unwrap();
        for (&elem1, &elem2) in arr.iter().tuple_windows() {
            if elem1 > elem2 {
                return Some(false);
            }
        }
        Some(true)
    } else {
        None
    }
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
