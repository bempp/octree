//! Parallel Octree structure

use std::collections::HashMap;

use crate::{
    constants::{DEEPEST_LEVEL, NSIBLINGS},
    geometry::PhysicalBox,
    morton::MortonKey,
    parsort::parsort,
};

use mpi::{
    datatype::{Partition, PartitionMut},
    point_to_point as p2p,
    traits::{Root, Source},
};

use itertools::{izip, Itertools};
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Destination},
};
use rand::Rng;

// /// A weighted Mortonkey contains weights to enable load balancing.
// #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Equivalence)]
// pub struct WeightedMortonKey {
//     /// The actual MortonKey.
//     pub key: MortonKey,
//     /// The weight of the key, typically the number of points in the corresponding octant.
//     pub weight: usize,
// }

// impl WeightedMortonKey {
//     /// Get a new weighted Morton key
//     pub fn new(key: MortonKey, weight: usize) -> Self {
//         Self { key, weight }
//     }
// }

// impl MinValue for WeightedMortonKey {
//     fn min_value() -> Self {
//         WeightedMortonKey {
//             key: MortonKey::from_index_and_level([0, 0, 0], 0),
//             weight: 0,
//         }
//     }
// }

// impl MaxValue for WeightedMortonKey {
//     fn max_value() -> Self {
//         WeightedMortonKey {
//             key: MortonKey::deepest_last(),
//             weight: usize::MAX,
//         }
//     }
// }

// impl Default for WeightedMortonKey {
//     fn default() -> Self {
//         WeightedMortonKey::new(Default::default(), 0)
//     }
// }

// impl Display for WeightedMortonKey {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "(Key: {}, Weight: {}", self.key, self.weight)
//     }
// }

/// Compute the global bounding box across all points on all processes.
pub fn compute_global_bounding_box<C: CommunicatorCollectives>(
    points: &[f64],
    comm: &C,
) -> PhysicalBox {
    // Make sure that the points array is a multiple of 3.
    assert_eq!(points.len() % 3, 0);
    let points: &[[f64; 3]] = bytemuck::cast_slice(points);

    // Now compute the minimum and maximum across each dimension.

    let mut xmin = f64::MAX;
    let mut xmax = f64::MIN;

    let mut ymin = f64::MAX;
    let mut ymax = f64::MIN;

    let mut zmin = f64::MAX;
    let mut zmax = f64::MIN;

    for point in points {
        let x = point[0];
        let y = point[1];
        let z = point[2];

        xmin = f64::min(xmin, x);
        xmax = f64::max(xmax, x);

        ymin = f64::min(ymin, y);
        ymax = f64::max(ymax, y);

        zmin = f64::min(zmin, z);
        zmax = f64::max(zmax, z);
    }

    let mut global_xmin = 0.0;
    let mut global_xmax = 0.0;

    let mut global_ymin = 0.0;
    let mut global_ymax = 0.0;

    let mut global_zmin = 0.0;
    let mut global_zmax = 0.0;

    comm.all_reduce_into(&xmin, &mut global_xmin, SystemOperation::min());
    comm.all_reduce_into(&xmax, &mut global_xmax, SystemOperation::max());

    comm.all_reduce_into(&ymin, &mut global_ymin, SystemOperation::min());
    comm.all_reduce_into(&ymax, &mut global_ymax, SystemOperation::max());

    comm.all_reduce_into(&zmin, &mut global_zmin, SystemOperation::min());
    comm.all_reduce_into(&zmax, &mut global_zmax, SystemOperation::max());

    let xdiam = global_xmax - global_xmin;
    let ydiam = global_ymax - global_ymin;
    let zdiam = global_zmax - global_zmin;

    let xmean = global_xmin + 0.5 * xdiam;
    let ymean = global_ymin + 0.5 * ydiam;
    let zmean = global_zmin + 0.5 * zdiam;

    // We increase diameters by box size on deepest level
    // and use the maximum diameter to compute a
    // cubic bounding box.

    let deepest_box_diam = 1.0 / (1 << DEEPEST_LEVEL) as f64;

    let max_diam = [xdiam, ydiam, zdiam].into_iter().reduce(f64::max).unwrap();

    let max_diam = max_diam * (1.0 + deepest_box_diam);

    PhysicalBox::new([
        xmean - 0.5 * max_diam,
        ymean - 0.5 * max_diam,
        zmean - 0.5 * max_diam,
        xmean + 0.5 * max_diam,
        ymean + 0.5 * max_diam,
        zmean + 0.5 * max_diam,
    ])
}

/// Convert points to Morton keys on specified level.
pub fn points_to_morton<C: CommunicatorCollectives>(
    points: &[f64],
    max_level: usize,
    comm: &C,
) -> (Vec<MortonKey>, PhysicalBox) {
    // Make sure that the points array is a multiple of 3.
    assert_eq!(points.len() % 3, 0);

    // Make sure that max level never exceeds DEEPEST_LEVEL
    let max_level = if max_level > DEEPEST_LEVEL as usize {
        DEEPEST_LEVEL as usize
    } else {
        max_level
    };

    // Compute the physical bounding box.

    let bounding_box = compute_global_bounding_box(points, comm);

    // Bunch the points in arrays of 3.

    let points: &[[f64; 3]] = bytemuck::cast_slice(points);

    let keys = points
        .iter()
        .map(|&point| MortonKey::from_physical_point(point, &bounding_box, max_level))
        .collect_vec();

    // Now want to get weighted Morton keys. We use a HashMap.

    let mut value_counts = HashMap::<MortonKey, usize>::new();

    for key in &keys {
        *value_counts.entry(*key).or_insert(0) += 1;
    }

    // let weighted_keys = value_counts
    //     .iter()
    //     .map(|(&key, &weight)| WeightedMortonKey::new(key, weight))
    //     .collect_vec();

    (keys, bounding_box)
}

/// Block partition of tree
pub fn block_partition<R: Rng, C: CommunicatorCollectives>(
    sorted_keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    let mut completed_region = sorted_keys
        .first()
        .unwrap()
        .fill_between_keys(*sorted_keys.last().unwrap());

    completed_region.insert(0, *sorted_keys.first().unwrap());
    completed_region.push(*sorted_keys.last().unwrap());

    // Get the smallest level members of the completed region.

    let min_level = completed_region
        .iter()
        .map(|elem| elem.level())
        .min()
        .unwrap();

    let largest_boxes = completed_region
        .iter()
        .filter(|elem| elem.level() == min_level)
        .copied()
        .collect_vec();

    let coarse_tree = complete_tree(&largest_boxes, rng, comm);
    coarse_tree
}

/// Linearize a set of weighted Morton keys.
pub fn linearize<R: Rng, C: CommunicatorCollectives>(
    keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    let size = comm.size();
    let rank = comm.rank();

    // If we only have one process we use the standard serial linearization.

    if size == 1 {
        return MortonKey::linearize(keys);
    }

    // We are first sorting the keys. Then in a linear process across all processors we
    // go through the arrays and delete ancestors of nodes.

    let sorted_keys = parsort(&keys, comm, rng);

    // Each process needs to send its first element to the previous process. Each process
    // then goes through its own list and retains elements that are not ancestors of the
    // next element.

    let mut result = Vec::<MortonKey>::new();

    if rank == size - 1 {
        comm.process_at_rank(rank - 1)
            .send(sorted_keys.first().unwrap());

        for (&m1, &m2) in sorted_keys.iter().tuple_windows() {
            // m1 is also ancestor of m2 if they are identical.
            if m1.is_ancestor(m2) {
                continue;
            } else {
                result.push(m1);
            }
        }

        result.push(*sorted_keys.last().unwrap());
    } else {
        let (other, _status) = if rank > 0 {
            p2p::send_receive(
                sorted_keys.first().unwrap(),
                &comm.process_at_rank(rank - 1),
                &comm.process_at_rank(rank + 1),
            )
        } else {
            comm.any_process().receive::<MortonKey>()
        };
        for (&m1, &m2) in sorted_keys.iter().tuple_windows() {
            // m1 is also ancestor of m2 if they are identical.
            if m1.is_ancestor(m2) {
                continue;
            } else {
                result.push(m1);
            }
        }

        let last = *sorted_keys.last().unwrap();

        if !last.is_ancestor(other) {
            result.push(last)
        }
    }

    result
}

/// Balance a sorted list of Morton keys across processors given an array of corresponding weights.
pub fn partition<C: CommunicatorCollectives>(
    sorted_keys: &[MortonKey],
    weights: &[usize],
    comm: &C,
) -> Vec<MortonKey> {
    assert_eq!(sorted_keys.len(), weights.len());

    let size = comm.size();
    let rank = comm.rank();

    // If we only have one process we simply return.

    if size == 1 {
        return sorted_keys.to_vec();
    }

    // First scan the weight.
    // We scan the local arrays, then use a global scan operation on the last element
    // of each array to get the global sums and then we update the array of each rank
    // with the sum from the previous ranks.

    let mut scan: Vec<usize> = weights
        .iter()
        .scan(0, |state, x| {
            *state += *x;
            Some(*state)
        })
        .collect_vec();
    let scan_last = *scan.last().unwrap();
    let mut scan_result: usize = 0;
    comm.exclusive_scan_into(&scan_last, &mut scan_result, SystemOperation::sum());
    for elem in &mut scan {
        *elem += scan_result;
    }

    let mut total_weight = if rank == size - 1 {
        *scan.last().unwrap()
    } else {
        0
    };

    // Scan the weight (form cumulative sums) and broadcast the total weight (last entry on last process)
    // to all other processes.

    comm.process_at_rank(size - 1)
        .broadcast_into(&mut total_weight);

    let w = total_weight / (size as usize);
    let k = total_weight % (size as usize);

    let mut hash_map = HashMap::<usize, Vec<MortonKey>>::new();

    // Sort the elements into bins according to which process they should be sent.

    for p in 1..=size as usize {
        let q = if p <= k as usize {
            izip!(sorted_keys, &scan)
                .filter_map(|(&key, &s)| {
                    if ((p - 1) * (1 + w) <= s && s < p * (w + 1))
                        || (p == size as usize && (p - 1) * (1 + w) <= s)
                    {
                        Some(key)
                    } else {
                        None
                    }
                })
                .collect_vec()
        } else {
            izip!(sorted_keys, &scan)
                .filter_map(|(&key, &s)| {
                    if ((p - 1) * w + k <= s && s < p * w + k)
                        || (p == size as usize && (p - 1) * w + k <= s)
                    {
                        Some(key)
                    } else {
                        None
                    }
                })
                .collect_vec()
        };
        hash_map.insert(p - 1, q);
    }

    // Now distribute the data with an all to all v.
    // We create a vector of how many elements to send to each process and
    // then send the actual data.

    let mut counts = vec![0 as i32; size as usize];
    let mut counts_from_processor = vec![0 as i32; size as usize];

    let mut all_elements = Vec::<MortonKey>::new();
    for (index, c) in counts.iter_mut().enumerate() {
        let elements = hash_map.get(&index).unwrap();
        *c = elements.len() as i32;
        all_elements.extend(elements.iter())
    }

    // Send around the number of elements for each process
    comm.all_to_all_into(&counts, &mut counts_from_processor);

    // We have the number of elements for each process now. Now send around
    // the actual elements.

    // We can now send around the actual elements with an alltoallv.
    let send_displs: Vec<i32> = counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp as i32)
        })
        .collect();

    let send_partition = Partition::new(&all_elements, &counts[..], &send_displs[..]);

    let mut recvbuffer =
        vec![MortonKey::default(); counts_from_processor.iter().sum::<i32>() as usize];

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

    recvbuffer.sort_unstable();
    recvbuffer
}

/// Given a distributed set of keys, generate a complete linear Octree.
pub fn complete_tree<R: Rng, C: CommunicatorCollectives>(
    keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    let mut linearized_keys = linearize(keys, rng, comm);

    let size = comm.size();
    let rank = comm.rank();

    if size == 1 {
        return MortonKey::complete_tree(linearized_keys.as_slice());
    }

    // Now insert on the first and last process the first and last child of the
    // finest ancestor of first/last box on deepest level

    // Send first element to previous rank and insert into local keys.
    // On the first process we also need to insert the first child of the finest
    // ancestor of the deepest first key and first element. Correspondingly on the last process
    // we need to insert the last child of the finest ancester of the deepest last key and last element.

    if rank == size - 1 {
        // On last process send first element to previous processes and insert last
        // possible box from region into list.
        comm.process_at_rank(rank - 1)
            .send(linearized_keys.first().unwrap());
        let last_key = *linearized_keys.last().unwrap();
        let deepest_last = MortonKey::deepest_last();
        if !last_key.is_ancestor(deepest_last) {
            let ancestor = deepest_last.finest_common_ancestor(last_key);
            linearized_keys.push(ancestor.children()[NSIBLINGS - 1]);
        }
    } else {
        let (other, _status) = if rank > 0 {
            // On intermediate process receive from the next process
            // and send first element to previous process.
            p2p::send_receive(
                linearized_keys.first().unwrap(),
                &comm.process_at_rank(rank - 1),
                &comm.process_at_rank(rank + 1),
            )
        } else {
            // On first process insert at the beginning the first possible
            // box in the region and receive the key from next process.
            let first_key = *linearized_keys.first().unwrap();
            let deepest_first = MortonKey::deepest_first();
            if !first_key.is_ancestor(deepest_first) {
                let ancestor = deepest_first.finest_common_ancestor(first_key);
                linearized_keys.push(ancestor.children()[0]);
            }

            comm.process_at_rank(1).receive::<MortonKey>()
        };
        // If we are not at the last process we need to introduce the received key
        // into our list.
        linearized_keys.push(other);
    };

    // Now complete the regions defined by the keys on each process.

    let mut result = Vec::<MortonKey>::new();

    for (&key1, &key2) in linearized_keys.iter().tuple_windows() {
        result.push(key1);
        result.extend_from_slice(key1.fill_between_keys(key2).as_slice());
    }

    if rank == size - 1 {
        result.push(*linearized_keys.last().unwrap());
    }

    result
}
