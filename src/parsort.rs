//! Implementation of a parallel samplesort.

use std::fmt::Display;
use std::mem::offset_of;

use itertools::{izip, Itertools};
use mpi::collective::{SystemOperation, UserOperation};
use mpi::datatype::{UncommittedDatatypeRef, UncommittedUserDatatype, UserDatatype};
use mpi::traits::Equivalence;
use mpi::{datatype::PartitionMut, traits::CommunicatorCollectives};
use rand::{seq::SliceRandom, Rng};

const OVERSAMPLING: usize = 2;

// An internal struct. We convert every array element
// into this struct. The idea is that this is guaranteed to be unique.

#[derive(Equivalence, Eq, PartialEq, PartialOrd, Ord, Copy, Clone, Default)]
#[repr(C)]
pub struct UniqueItem {
    value: u64,
    rank: usize,
    index: usize,
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

// unsafe impl Equivalence for UniqueItem {
//     type Out = UserDatatype;

//     fn equivalent_datatype() -> Self::Out {
//         UserDatatype::structured::<UncommittedDatatypeRef>(
//             &[1, 1, 1],
//             &[
//                 offset_of!(UniqueItem, value) as isize,
//                 offset_of!(UniqueItem, rank) as isize,
//                 offset_of!(UniqueItem, index) as isize,
//             ],
//             &[
//                 u64::equivalent_datatype().into(),
//                 usize::equivalent_datatype().into(),
//                 usize::equivalent_datatype().into(),
//             ],
//         )
//     }
// }

pub fn to_unique_item(arr: &[u64], rank: usize) -> Vec<UniqueItem> {
    arr.iter()
        .enumerate()
        .map(|(index, &item)| UniqueItem::new(item, rank, index))
        .collect()
}

pub fn get_global_min_max<C>(arr: &[UniqueItem], comm: &C) -> (UniqueItem, UniqueItem)
where
    C: CommunicatorCollectives,
{
    let (min, max) = if arr.len() > 0 {
        (*arr.iter().min().unwrap(), *arr.iter().max().unwrap())
    } else {
        // Return these choices guarantee that an empty local array
        // does not interfere in the global min-max calculation.
        (UniqueItem::MAX, UniqueItem::MIN)
    };

    // Use an allreduce to get the min and max across all processors

    let mut global_min: UniqueItem = Default::default();
    let mut global_max: UniqueItem = Default::default();
    comm.all_reduce_into(
        &min,
        &mut global_min,
        &UserOperation::commutative(|x, y| {
            let x: &[UniqueItem] = x.downcast().unwrap();
            let y: &mut [UniqueItem] = y.downcast().unwrap();
            for (&x_i, y_i) in izip!(x.iter(), y.iter_mut()) {
                *y_i = x_i.min(*y_i);
            }
        }),
    );

    comm.all_reduce_into(
        &max,
        &mut global_max,
        &UserOperation::commutative(|x, y| {
            let x: &[UniqueItem] = x.downcast().unwrap();
            let y: &mut [UniqueItem] = y.downcast().unwrap();
            for (&x_i, y_i) in izip!(x.iter(), y.iter_mut()) {
                *y_i = x_i.max(*y_i);
            }
        }),
    );

    (global_min, global_max)
}

// pub fn get_splitters<C, R>(arr: &[u64], comm: &C, rng: &mut R) -> Vec<u64>
// where
//     C: CommunicatorCollectives,
//     R: Rng + ?Sized,
// {
//     let size = comm.size() as usize;

//     // In the first step we pick `oversampling * nprocs` splitters.

//     let oversampling = if arr.len() < OVERSAMPLING {
//         arr.len()
//     } else {
//         OVERSAMPLING
//     };

//     // We are choosing unique splitters that neither contain
//     // zero nor u64::max.

//     let splitters = arr
//         .choose_multiple(rng, oversampling)
//         .unique()
//         .copied()
//         .filter(|&elem| elem != 0 && elem != u64::MAX)
//         .collect::<Vec<_>>();

//     // We use an all_gatherv so that each process receives all splitters.
//     // For that we first communicate how many splitters each process has
//     // and then we send the splitters themselves.

//     let nsplitters = splitters.len();
//     let mut splitters_per_rank = vec![0 as usize; size];

//     comm.all_gather_into(&nsplitters, &mut splitters_per_rank);

//     // We now know how many splitters each process has. We now create space
//     // for the splitters and send them all around.

//     let n_all_splitters = splitters_per_rank.iter().sum();

//     let mut all_splitters = vec![0 as u64; n_all_splitters];
//     let splitters_per_rank = splitters_per_rank.iter().map(|&x| x as i32).collect_vec();

//     let displs: Vec<i32> = splitters_per_rank
//         .iter()
//         .scan(0, |acc, &x| {
//             let tmp = *acc;
//             *acc += x;
//             Some(tmp)
//         })
//         .collect();

//     let mut partition = PartitionMut::new(&mut all_splitters[..], splitters_per_rank, &displs[..]);
//     comm.all_gather_varcount_into(&splitters, &mut partition);

//     // We now have all splitters available on each process.
//     // We can now sort the splitters. Every process will then have the same list of sorted splitters.

//     all_splitters.sort_unstable();

//     // We add the global min and the global max if they are not already
//     // contained.

//     all_splitters
// }

// pub fn get_bin_displacements(arr: &[u64], splitters: &[u64]) -> Vec<usize> {
//     // The following array will store the bin displacements.
//     // The first bin displacement position is 0. The last bin displacement
//     // position is the length of the array.

//     let mut bin_displs = vec![0 as usize; 2 + splitters.len()];

//     // We are iterating through the array. Whenever an element is larger or equal than
//     // the current splitter we store the current position in `bin_displs` and advance `splitter_iter`
//     // by 1.

//     // In the following iterator we skip the first bin displacement position as this must be the default
//     // zero (start of the bins).

//     let mut splitter_iter = izip!(splitters.iter(), bin_displs.iter_mut().skip(1));

//     if let Some((mut splitter, mut bin_entry)) = splitter_iter.next() {
//         for (index, &arr_item) in arr.iter().enumerate() {
//             if arr_item >= *splitter {
//                 *bin_entry = index;
//                 match splitter_iter.next() {
//                     Some((next_splitter, next_entry)) => {
//                         splitter = next_splitter;
//                         bin_entry = next_entry;
//                     }
//                     None => break,
//                 }
//             }
//         }
//     }

//     // At the end all unused splitters are assigned as bin position the end
//     // of the array. This is fine since the MPI all_to_allv is not actually
//     // accessing and transmitting data for those.

//     while let Some((_, next_entry)) = splitter_iter.next() {
//         *next_entry = arr.len();
//     }

//     // Don't forget to set the last bin displacement position to length
//     // of the array.

//     *bin_displs.last_mut().unwrap() = arr.len();
//     bin_displs
// }

pub fn parsort<C: CommunicatorCollectives, R: Rng + ?Sized>(
    arr: &mut [u64],
    comm: &C,
    rng: &mut R,
) {
    let size = comm.size() as usize;
    let rank = comm.rank() as usize;
    // If we only have a single rank simply sort the local array and return

    if size == 1 {
        arr.sort_unstable();
        return;
    }

    // let splitters = get_splitters(arr, comm, rng);

    // Each process now has the same splitters. We now sort our local data into the bins
    // from the splitters.

    // To do this simple we sort the local array and then store the bin boundaries.

    arr.sort_unstable();

    // let bin_displs = get_bin_displacements(arr, &splitters);
}
