//! Parallel Octree structure

use std::collections::HashMap;

use crate::{
    constants::{DEEPEST_LEVEL, NSIBLINGS},
    geometry::PhysicalBox,
    morton::MortonKey,
    parsort::parsort,
    tools::{communicate_back, gather_to_all, global_inclusive_cumsum, redistribute},
};

use mpi::{
    point_to_point as p2p,
    traits::{Root, Source},
};

use itertools::{izip, Itertools};
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Destination},
};
use rand::Rng;

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

    (keys, bounding_box)
}

/// Block partition of tree.
///
/// Returns a tuple `(partitioned_keys, coarse_keys)` of the partitioned
/// keys and the associated coarse keys.
/// A necessary condition for the block partitioning is that
// all sorted keys are on the same level.
pub fn block_partition<R: Rng, C: CommunicatorCollectives>(
    sorted_keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> (Vec<MortonKey>, Vec<MortonKey>) {
    let rank = comm.rank();
    if comm.size() == 1 {
        // On a single node block partitioning should not do anything.
        return (sorted_keys.to_vec(), vec![MortonKey::root()]);
    }

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

    // Each process selects its largest boxes. These are used to create
    // a coarse tree.

    let largest_boxes = completed_region
        .iter()
        .filter(|elem| elem.level() == min_level)
        .copied()
        .collect_vec();

    let coarse_tree = complete_tree(&largest_boxes, rng, comm);

    // We want to partition the coarse tree. But we need the correct weights. The idea
    // is that we use the number of original leafs that intersect with the coarse tree
    // as leafs. In order to compute this we send the coarse tree around to all processes
    // so that each process computes for each coarse tree element how many of its keys
    // intersect with each node of the coarse tree. We then sum up the local weight for each
    // coarse tree node across all nodes to get the weight.

    let global_coarse_tree = gather_to_all(&coarse_tree, comm);

    // We also want to send around a corresponding array of ranks so that for each global coarse tree key
    // we have the rank of where it originates from.

    let coarse_tree_ranks = gather_to_all(&vec![rank as usize; coarse_tree.len()], comm);

    // We now compute the local weights.
    let mut local_weights = vec![0 as usize; global_coarse_tree.len()];

    // In the following loop we want to be a bit smart. We do not iterate through all the local elements.
    // We know that our keys are sorted and also that the coarse tree keys are sorted. So we find the region
    // of our sorted keys that overlaps with the coarse tree region.

    // Let's find the start of our region. The start of our region is a coarse key that is an ancestor
    // of our current key. This works because the coarse tree has levels at most as high as the sorted keys.

    let first_key = *sorted_keys.first().unwrap();

    let first_coarse_index = global_coarse_tree
        .iter()
        .take_while(|coarse_key| !coarse_key.is_ancestor(first_key))
        .count();

    // Now we need to find the end index of our region. For this again we find the index of our coarse tree that
    // is an ancestor of our last key.
    let last_key = *sorted_keys.last().unwrap();

    let last_coarse_index = global_coarse_tree
        .iter()
        .take_while(|coarse_key| !coarse_key.is_ancestor(last_key))
        .count();

    // We now only need to iterate through between the first and last coarse index in the coarse tree.
    // In the way we have computed the indices. The last coarse index is inclusive (it is the ancestor of our last key).

    for (w, &global_coarse_key) in izip!(
        local_weights[first_coarse_index..=last_coarse_index].iter_mut(),
        global_coarse_tree[first_coarse_index..=last_coarse_index].iter()
    ) {
        *w += sorted_keys
            .iter()
            .filter(|&&key| global_coarse_key.is_ancestor(key))
            .count();
    }

    // We now need to sum up the weights across all processes.

    let mut weights = vec![0 as usize; global_coarse_tree.len()];

    comm.all_reduce_into(&local_weights, &mut weights, SystemOperation::sum());

    // Each process now has all weights. However, we only need the ones for the current process.
    // So we just filter the rest out.

    let weights = izip!(coarse_tree_ranks, weights)
        .filter_map(|(r, weight)| {
            if r == rank as usize {
                Some(weight)
            } else {
                None
            }
        })
        .collect_vec();

    let coarse_tree = partition(&coarse_tree, &weights, comm);

    (
        redistribute_with_respect_to_coarse_tree(&sorted_keys, &coarse_tree, comm),
        coarse_tree,
    )

    // We now need to redistribute the global tree according to the coarse tree.
}

/// Redistribute sorted keys with respect to a linear coarse tree.
pub fn redistribute_with_respect_to_coarse_tree<C: CommunicatorCollectives>(
    sorted_keys: &[MortonKey],
    coarse_tree: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    let size = comm.size();

    if size == 1 {
        return sorted_keys.to_vec();
    }

    // We want to globally redistribute keys so that the keys on each process are descendents
    // of the local coarse tree keys.

    // We are using here the fact that the coarse tree is complete and sorted.
    // We are sending around to each process the first local index. This
    // defines bins in which we sort our keys. The keys are then sent around to the correct
    // processes via an alltoallv operation.

    let my_first = coarse_tree.first().unwrap();

    let mut global_bins = gather_to_all(std::slice::from_ref(my_first), comm);

    // We now have the first index from each process. We also want
    // an upper bound for the last index of the tree to make the sorting into
    // bins easier.
    global_bins.push(MortonKey::upper_bound());

    // We now have our bins. We go through our keys and store how
    // many keys are assigned to each rank. We are using here that
    // our keys and the coarse tree are both sorted.

    // This will store for each rank how many keys will be assigned to it.

    let rank_counts = sort_to_bins(sorted_keys, &global_bins)
        .iter()
        .map(|&elem| elem as i32)
        .collect_vec();

    // We now have the counts for each rank. Let's redistribute accordingly and return.

    redistribute(&sorted_keys, &rank_counts, comm)
}

/// Create bins from sorted keys.
pub fn sort_to_bins(sorted_keys: &[MortonKey], bins: &[MortonKey]) -> Vec<usize> {
    let mut bin_counts = vec![0 as usize; bins.len() - 1];

    // This iterates over each possible bin and returns also the associated rank.
    let mut bin_iter = izip!(
        bin_counts.iter_mut(),
        bins.iter().tuple_windows::<(&MortonKey, &MortonKey)>(),
    );

    // We take the first element of the bin iterator. There will always be at least one.
    let mut r: &mut usize;
    let mut bin_start: &MortonKey;
    let mut bin_end: &MortonKey;
    (r, (bin_start, bin_end)) = bin_iter.next().unwrap();

    for &key in sorted_keys.iter() {
        if *bin_start <= key && key < *bin_end {
            *r += 1;
        } else {
            // Move the bin forward until it fits. There will always be a fitting bin.
            while let Some((rn, (bsn, ben))) = bin_iter.next() {
                if *bsn <= key && key < *ben {
                    *rn += 1;
                    r = rn;
                    bin_start = bsn;
                    bin_end = ben;
                    break;
                }
            }
        }
    }

    bin_counts
}

/// Return a complete tree generated from local keys and associated coarse keys.
///
/// The coarse keys are refined until the maximum level is reached or until each coarse key
/// is the ancestor of at most `max_keys` fine keys.
pub fn create_local_tree(
    sorted_fine_keys: &[MortonKey],
    coarse_keys: &[MortonKey],
    mut max_level: usize,
    max_keys: usize,
) -> Vec<MortonKey> {
    if max_level > DEEPEST_LEVEL as usize {
        max_level = DEEPEST_LEVEL as usize;
    }

    // We split the sorted fine keys into subslices so that each subslice
    // is associated with a coarse slice. For this we need to add an upper bound
    // coarse keys to ensure that we have suitable bins.

    let mut bins = coarse_keys.to_vec();
    bins.push(MortonKey::upper_bound());

    let counts = sort_to_bins(&sorted_fine_keys, &bins);

    // We now know how many fine keys are associated with each coarse block. We iterate
    // through and locally refine for each block that requires it.

    let mut remainder = sorted_fine_keys;
    let mut new_coarse_keys = Vec::<MortonKey>::new();

    for (&count, &coarse_key) in izip!(counts.iter(), coarse_keys.iter()) {
        let current;
        (current, remainder) = remainder.split_at(count);
        if coarse_key.level() < max_level && current.len() > max_keys {
            // We need to refine the current split.
            new_coarse_keys.extend_from_slice(
                create_local_tree(
                    current,
                    coarse_key.children().as_slice(),
                    max_level,
                    max_keys,
                )
                .as_slice(),
            );
        } else {
            new_coarse_keys.push(coarse_key)
        }
    }

    coarse_keys.to_vec()
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

    let next_key = communicate_back(&sorted_keys, comm);

    // Treat the local keys
    for (&m1, &m2) in sorted_keys.iter().tuple_windows() {
        // m1 is also ancestor of m2 if they are identical.
        if m1.is_ancestor(m2) {
            continue;
        } else {
            result.push(m1);
        }
    }

    // If we are at the last process simply push the last key.
    // Otherwise check whether it might be the ancestor of `next_key`,
    // the first key on the next process. If yes, don't push it. Otherwise do.

    if rank == size - 1 {
        result.push(*sorted_keys.last().unwrap());
    } else {
        let last = *sorted_keys.last().unwrap();
        if !last.is_ancestor(next_key.unwrap()) {
            result.push(last);
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

    let scan = global_inclusive_cumsum(&weights, comm);

    // Now broadcast the total weight to all processes.

    let mut total_weight = if rank == size - 1 {
        *scan.last().unwrap()
    } else {
        0
    };

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

    let mut all_elements = Vec::<MortonKey>::new();
    for (index, c) in counts.iter_mut().enumerate() {
        let elements = hash_map.get(&index).unwrap();
        *c = elements.len() as i32;
        all_elements.extend(elements.iter())
    }

    let mut recvbuffer = redistribute(&all_elements, &counts, comm);

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
                linearized_keys.insert(0, ancestor.children()[0]);
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

/// Return true on all ranks if distributed tree is complete. Otherwise, return false.
pub fn is_complete_linear_tree<C: CommunicatorCollectives>(arr: &[MortonKey], comm: &C) -> bool {
    // First check that the local tree on each node is complete.

    let mut complete_linear = true;
    for (key1, key2) in arr.iter().tuple_windows() {
        // Make sure that the keys are sorted and not duplicated.
        if key1 >= key2 {
            complete_linear = false;
            break;
        }
        // The next key should be an ancestor of the next non-descendent key.
        if let Some(expected_next) = key1.next_non_descendent_key() {
            if !key2.is_ancestor(expected_next) {
                complete_linear = false;
                break;
            }
        } else {
            // Only for the very last key there should not be a next non-descendent key.
            complete_linear = false;
        }
    }

    // We now check the interfaces.

    if let Some(next_first) = communicate_back(arr, comm) {
        // We are on any but the last rank
        let last_key = arr.last().unwrap();

        // Check that the keys are sorted and not duplicated.
        if *last_key >= next_first {
            complete_linear = false;
        }

        // Check that the next key is an encestor of the next non-descendent.
        if let Some(expected_next) = last_key.next_non_descendent_key() {
            if !next_first.is_ancestor(expected_next) {
                complete_linear = false;
            }
        } else {
            complete_linear = false;
        }
    } else {
        // We are on the last rank
        // Check that the last key is ancestor of deepest last.
        if !arr.last().unwrap().is_ancestor(MortonKey::deepest_last()) {
            complete_linear = false;
        }
    }

    // Now check that at the first rank we include the deepest first.

    if comm.rank() == 0 {
        if !arr.first().unwrap().is_ancestor(MortonKey::deepest_first()) {
            complete_linear = false;
        }
    }

    // Now communicate everything together.

    let mut result = false;
    comm.all_reduce_into(
        &complete_linear,
        &mut result,
        SystemOperation::logical_and(),
    );

    result
}
