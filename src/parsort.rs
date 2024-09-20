//! Implementation of a parallel samplesort.

use std::fmt::Display;
use std::mem::offset_of;

use itertools::Itertools;
use mpi::datatype::{UncommittedDatatypeRef, UncommittedUserDatatype, UserDatatype};
use mpi::traits::Equivalence;
use mpi::{
    datatype::{Partition, PartitionMut},
    traits::CommunicatorCollectives,
};
use rand::{seq::SliceRandom, Rng};

use crate::tools::{displacements, gather_to_all};

const OVERSAMPLING: usize = 8;

/// Sortable trait that each type fed into parsort needs to satisfy.
pub trait ParallelSortable:
    MinValue
    + MaxValue
    + Equivalence
    + Copy
    + Clone
    + Default
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Display
    + Sized
{
}

impl<
        T: MinValue
            + MaxValue
            + Equivalence
            + Copy
            + Clone
            + Default
            + PartialEq
            + Eq
            + PartialOrd
            + Ord
            + Display
            + Sized,
    > ParallelSortable for T
{
}

/// An internal struct. We convert every array element
/// into this struct. The idea is that this is guaranteed to be unique
/// as it encodes not only the element but also its rank and index.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct UniqueItem<T: ParallelSortable> {
    pub value: T,
    pub rank: usize,
    pub index: usize,
}

unsafe impl<T: ParallelSortable> Equivalence for UniqueItem<T> {
    type Out = UserDatatype;

    // Depending on the MPI implementation the below offset needs
    // to be an i64 or isize. If it is an i64 Clippy warns about
    // a useless conversion. But this warning is MPI implementation
    // dependent. So switch off here.

    #[allow(clippy::useless_conversion)]
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured::<UncommittedDatatypeRef>(
            &[1, 1, 1],
            &[
                (offset_of!(UniqueItem<T>, value) as i64)
                    .try_into()
                    .unwrap(),
                (offset_of!(UniqueItem<T>, rank) as i64).try_into().unwrap(),
                (offset_of!(UniqueItem<T>, index) as i64)
                    .try_into()
                    .unwrap(),
            ],
            &[
                UncommittedUserDatatype::contiguous(1, &<T as Equivalence>::equivalent_datatype())
                    .as_ref(),
                usize::equivalent_datatype().into(),
                usize::equivalent_datatype().into(),
            ],
        )
    }
}

/// Return the minimum possible value of a type.
pub trait MinValue {
    /// Return the min value.
    fn min_value() -> Self;
}

/// Return the maximum possible value of a type.
pub trait MaxValue {
    /// Return the max value.
    fn max_value() -> Self;
}

impl<T: ParallelSortable> Display for UniqueItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(value: {}, rank: {}, index: {})",
            self.value, self.rank, self.index
        )
    }
}

impl<T: ParallelSortable> MinValue for UniqueItem<T> {
    fn min_value() -> Self {
        UniqueItem::new(<T as MinValue>::min_value(), 0, 0)
    }
}

impl<T: ParallelSortable> MaxValue for UniqueItem<T> {
    fn max_value() -> Self {
        UniqueItem::new(<T as MaxValue>::max_value(), 0, 0)
    }
}

impl<T: ParallelSortable> UniqueItem<T> {
    pub fn new(value: T, rank: usize, index: usize) -> Self {
        Self { value, rank, index }
    }
}

fn to_unique_item<T: ParallelSortable>(arr: &[T], rank: usize) -> Vec<UniqueItem<T>> {
    arr.iter()
        .enumerate()
        .map(|(index, &item)| UniqueItem::new(item, rank, index))
        .collect()
}

fn get_buckets<T, C, R>(arr: &[UniqueItem<T>], comm: &C, rng: &mut R) -> Vec<UniqueItem<T>>
where
    T: ParallelSortable,
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

    // We gather the splitters into all ranks so that each rank has all splitters.

    let mut all_splitters = gather_to_all(&splitters, comm);

    // We now have all splitters available on each process.
    // We can now sort the splitters. Every process will then have the same list of sorted splitters.

    all_splitters.sort_unstable();

    // We now insert the smallest and largest possible element if they are not already
    // in the splitter collection.

    if *all_splitters.first().unwrap() != UniqueItem::min_value() {
        all_splitters.insert(0, UniqueItem::min_value())
    }

    if *all_splitters.last().unwrap() != UniqueItem::max_value() {
        all_splitters.push(UniqueItem::max_value());
    }

    // We now define p buckets (p is number of processors) and we return
    // a p + 1 element array containing the first element of each bucket
    // concluded with the largest possible element.

    all_splitters = split(&all_splitters, size)
        .map(|slice| slice.first().unwrap())
        .copied()
        .collect::<Vec<_>>();
    all_splitters.push(UniqueItem::max_value());

    all_splitters
}

fn get_counts<T: ParallelSortable>(arr: &[UniqueItem<T>], buckets: &[UniqueItem<T>]) -> Vec<usize> {
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
        if (first <= elem && elem < last)
            || (*last == UniqueItem::max_value() && *elem == UniqueItem::max_value())
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
                    || (*last == UniqueItem::max_value() && *elem == UniqueItem::max_value())
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
pub fn parsort<T: ParallelSortable, C: CommunicatorCollectives, R: Rng + ?Sized>(
    arr: &[T],
    comm: &C,
    rng: &mut R,
) -> Vec<T> {
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

    let send_displs = displacements(&counts);

    let send_partition = Partition::new(&arr, counts, &send_displs[..]);

    let mut recvbuffer =
        vec![UniqueItem::default(); counts_from_processor.iter().sum::<i32>() as usize];

    let recv_displs = displacements(&counts_from_processor);

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

macro_rules! impl_min_max_value {
    ($type:ty) => {
        impl MinValue for $type {
            fn min_value() -> Self {
                <$type>::MIN
            }
        }

        impl MaxValue for $type {
            fn max_value() -> Self {
                <$type>::MAX
            }
        }
    };
}

impl_min_max_value!(usize);
impl_min_max_value!(i8);
impl_min_max_value!(i32);
impl_min_max_value!(i64);
impl_min_max_value!(u8);
impl_min_max_value!(u32);
impl_min_max_value!(u64);
