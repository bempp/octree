//! Implementation of a parallel samplesort.

use std::fmt::Display;

use itertools::Itertools;
use mpi::traits::{Equivalence, Root};
use mpi::{
    datatype::{Partition, PartitionMut},
    traits::CommunicatorCollectives,
};
use rand::{seq::SliceRandom, Rng};

const OVERSAMPLING: usize = 8;

/// An internal struct. We convert every array element
/// into this struct. The idea is that this is guaranteed to be unique
/// as it encodes not only the element but also its rank and index.
#[derive(Equivalence, Eq, PartialEq, PartialOrd, Ord, Copy, Clone, Default)]
#[repr(C)]
struct UniqueItem {
    pub value: u64,
    pub rank: usize,
    pub index: usize,
}

impl Display for UniqueItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(value: {}, rank: {}, index: {})",
            self.value, self.rank, self.index
        )
    }
}

impl UniqueItem {
    const MIN: UniqueItem = Self {
        value: 0,
        rank: 0,
        index: 0,
    };

    const MAX: UniqueItem = Self {
        value: u64::MAX,
        rank: usize::MAX,
        index: usize::MAX,
    };

    pub fn new(value: u64, rank: usize, index: usize) -> Self {
        Self { value, rank, index }
    }
}

fn to_unique_item(arr: &[u64], rank: usize) -> Vec<UniqueItem> {
    arr.iter()
        .enumerate()
        .map(|(index, &item)| UniqueItem::new(item, rank, index))
        .collect()
}

fn get_buckets<C, R>(arr: &[UniqueItem], comm: &C, rng: &mut R) -> Vec<UniqueItem>
where
    C: CommunicatorCollectives,
    R: Rng + ?Sized,
{
    let size = comm.size() as usize;

    // In the first step we pick `oversampling * nprocs` splitters.

    let oversampling = if arr.len() < OVERSAMPLING {
        arr.len()
    } else {
        OVERSAMPLING
    };

    // We are choosing unique splitters that neither contain
    // zero nor u64::max.

    let splitters = arr
        .choose_multiple(rng, oversampling)
        .copied()
        .collect::<Vec<_>>();

    // We use an all_gatherv so that each process receives all splitters.
    // For that we first communicate how many splitters each process has
    // and then we send the splitters themselves.

    let nsplitters = splitters.len();
    let mut splitters_per_rank = vec![0_usize; size];

    comm.all_gather_into(&nsplitters, &mut splitters_per_rank);

    // We now know how many splitters each process has. We now create space
    // for the splitters and send them all around.

    let n_all_splitters = splitters_per_rank.iter().sum();

    let mut all_splitters = vec![Default::default(); n_all_splitters];
    let splitters_per_rank = splitters_per_rank.iter().map(|&x| x as i32).collect_vec();

    let displs: Vec<i32> = splitters_per_rank
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    let mut partition = PartitionMut::new(&mut all_splitters[..], splitters_per_rank, &displs[..]);
    comm.all_gather_varcount_into(&splitters, &mut partition);

    // We now have all splitters available on each process.
    // We can now sort the splitters. Every process will then have the same list of sorted splitters.

    all_splitters.sort_unstable();

    // We now insert the smallest and largest possible element if they are not already
    // in the splitter collection.

    if *all_splitters.first().unwrap() != UniqueItem::MIN {
        all_splitters.insert(0, UniqueItem::MIN)
    }

    if *all_splitters.last().unwrap() != UniqueItem::MAX {
        all_splitters.push(UniqueItem::MAX);
    }

    // We now define p buckets (p is number of processors) and we return
    // a p + 1 element array containing the first element of each bucket
    // concluded with the largest possible element.

    all_splitters = split(&all_splitters, size)
        .map(|slice| slice.first().unwrap())
        .copied()
        .collect::<Vec<_>>();
    all_splitters.push(UniqueItem::MAX);

    all_splitters
}

fn get_counts(arr: &[UniqueItem], buckets: &[UniqueItem]) -> Vec<usize> {
    // The following array will store the counts for each bucket.

    let mut counts = vec![0_usize; buckets.len() - 1];

    // We are iterating through the array. Whenever an element is larger or equal than
    // the current splitter we store the current position in `bin_displs` and advance `splitter_iter`
    // by 1.

    // In the following iterator we skip the first bin displacement position as this must be the default
    // zero (start of the bins).

    // Note that bucket iterator has as many elements as counts as the tuple_windows has length
    // 1 smaller than the original array length.
    let mut bucket_iter = buckets.iter().tuple_windows::<(_, _)>();

    // We skip the first element as this is always zero.
    let mut count_iter = counts.iter_mut();

    let mut count: usize = 0;
    let mut current_count = count_iter.next().unwrap();

    let (mut first, mut last) = bucket_iter.next().unwrap();

    for elem in arr {
        // The test after the or sorts out the case that our set includes the maximum possible
        // item and we are in the last bucket. The biggest item should be counted as belonging
        // to the bucket.
        if (first <= elem && elem < last) || (*last == UniqueItem::MAX && *elem == UniqueItem::MAX)
        {
            // Element is in the right bucket.
            count += 1;
            continue;
        } else {
            // Element is not in the right bucket.
            // Store counts and find the correct bucket.
            *current_count = count;
            loop {
                (first, last) = bucket_iter.next().unwrap();
                current_count = count_iter.next().unwrap();
                if (first <= elem && elem < last)
                    || (*last == UniqueItem::MAX && *elem == UniqueItem::MAX)
                {
                    break;
                }
            }
            // Now have the right bucket. Reset count and continue.
            count = 1;
        }
    }

    // Need to store the count for the last bucket in the iterator.
    // This is always necessary as last iterator is half open interval.
    // So we don't go into the else part of the for loop.

    *current_count = count;

    // We don't need to fill the remaining counts entries with zero
    // since the array is already initialized with zero.

    counts
}

/// Parallel sort
pub fn parsort<C: CommunicatorCollectives, R: Rng + ?Sized>(
    arr: &[u64],
    comm: &C,
    rng: &mut R,
) -> Vec<u64> {
    let size = comm.size() as usize;
    let rank = comm.rank() as usize;
    // If we only have a single rank simply sort the local array and return

    let mut arr = arr.to_vec();

    if size == 1 {
        arr.sort_unstable();
        return arr;
    }

    // We first convert the array into unique elements by adding information
    // about index and rank. This guarantees that we don't have duplicates in
    // our sorting set.

    let mut arr = to_unique_item(&arr, rank);

    // We now sort the local array.

    arr.sort_unstable();

    // Let us now get the buckets.

    let buckets = get_buckets(&arr, comm, rng);

    // We now compute how many elements of our array go into each bucket.

    let counts = get_counts(&arr, &buckets)
        .iter()
        .map(|&elem| elem as i32)
        .collect::<Vec<_>>();

    // We now do an all_to_allv to communicate the array elements to the right processors.

    // First we need to communicate how many elements everybody gets from each processor.

    let mut counts_from_processor = vec![0_i32; size];

    comm.all_to_all_into(&counts, &mut counts_from_processor);

    // Each processor now knows how much he gets from all the others.

    // We can now send around the actual elements with an alltoallv.
    let send_displs: Vec<i32> = counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    let send_partition = Partition::new(&arr, counts, &send_displs[..]);

    let mut recvbuffer =
        vec![UniqueItem::default(); counts_from_processor.iter().sum::<i32>() as usize];

    let recv_displs: Vec<i32> = counts_from_processor
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    let mut receiv_partition =
        PartitionMut::new(&mut recvbuffer[..], counts_from_processor, &recv_displs[..]);
    comm.all_to_all_varcount_into(&send_partition, &mut receiv_partition);

    // We now have everything in the receive buffer. Now sort the local elements and return

    recvbuffer.sort_unstable();
    recvbuffer.iter().map(|&elem| elem.value).collect_vec()
}

// The following is a simple iterator that splits a slice into n
// chunks. It is from https://users.rust-lang.org/t/how-to-split-a-slice-into-n-chunks/40008/3

fn split<T>(slice: &[T], n: usize) -> impl Iterator<Item = &[T]> {
    let len = slice.len() / n;
    let rem = slice.len() % n;
    Split { slice, len, rem }
}

struct Split<'a, T> {
    slice: &'a [T],
    len: usize,
    rem: usize,
}

impl<'a, T> Iterator for Split<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }
        let mut len = self.len;
        if self.rem > 0 {
            len += 1;
            self.rem -= 1;
        }
        let (chunk, rest) = self.slice.split_at(len);
        self.slice = rest;
        Some(chunk)
    }
}

/// Array to root
pub fn array_to_root<T: Equivalence + Default + Copy + Clone, C: CommunicatorCollectives>(
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

        let mut ranks = vec![0_i32; size as usize];
        root_process.gather_into_root(&n, &mut ranks);

        // We now have all ranks at root. Can now a varcount gather to get
        // the array elements.

        let nelements = ranks.iter().sum::<i32>();

        let mut new_arr = vec![<T as Default>::default(); nelements as usize];

        let displs: Vec<i32> = ranks
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();

        let mut partition = PartitionMut::new(&mut new_arr[..], ranks, &displs[..]);

        root_process.gather_varcount_into_root(arr, &mut partition);
        Some(new_arr)
    } else {
        root_process.gather_into(&n);
        root_process.gather_varcount_into(arr);
        None
    }
}
