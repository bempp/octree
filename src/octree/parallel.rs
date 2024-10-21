//! Parallel Octree structure

use std::collections::{HashMap, HashSet};

use crate::{
    constants::{DEEPEST_LEVEL, NSIBLINGS},
    geometry::{PhysicalBox, Point},
    morton::MortonKey,
    parsort::parsort,
    tools::{
        communicate_back, gather_to_all, gather_to_root, global_inclusive_cumsum, redistribute,
        sort_to_bins,
    },
};

use mpi::traits::{Equivalence, Root};

use itertools::{izip, Itertools};
use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use rand::Rng;

use super::KeyType;

/// Compute the global bounding box across all points on all processes.
pub fn compute_global_bounding_box<C: CommunicatorCollectives>(
    points: &[Point],
    comm: &C,
) -> PhysicalBox {
    // Make sure that the points array is a multiple of 3.

    // Now compute the minimum and maximum across each dimension.

    let mut xmin = f64::MAX;
    let mut xmax = f64::MIN;

    let mut ymin = f64::MAX;
    let mut ymax = f64::MIN;

    let mut zmin = f64::MAX;
    let mut zmax = f64::MIN;

    for point in points {
        let x = point.coords()[0];
        let y = point.coords()[1];
        let z = point.coords()[2];

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
    points: &[Point],
    max_level: usize,
    comm: &C,
) -> (Vec<MortonKey>, PhysicalBox) {
    // Make sure that max level never exceeds DEEPEST_LEVEL
    let max_level = if max_level > DEEPEST_LEVEL as usize {
        DEEPEST_LEVEL as usize
    } else {
        max_level
    };

    // Compute the physical bounding box.

    let bounding_box = compute_global_bounding_box(points, comm);

    // Bunch the points in arrays of 3.

    let keys = points
        .iter()
        .map(|&point| MortonKey::from_physical_point(point, &bounding_box, max_level))
        .collect_vec();

    (keys, bounding_box)
}

/// Take a linear sequence of Morton keys and compute a complete linear associated coarse tree.
/// The returned coarse tree is load balanced according to the number of linear keys in each coarse block.
pub fn compute_coarse_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    let size = comm.size();

    debug_assert!(is_linear_tree(linear_keys, comm));

    // On a single node a complete coarse tree is simply the root.
    if size == 1 {
        return vec![MortonKey::root()];
    }

    let mut completed_region = linear_keys
        .first()
        .unwrap()
        .fill_between_keys(*linear_keys.last().unwrap());

    completed_region.insert(0, *linear_keys.first().unwrap());
    completed_region.push(*linear_keys.last().unwrap());

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

    debug_assert!(is_linear_tree(&largest_boxes, comm));

    complete_tree(&largest_boxes, comm)
}

/// Compute the weights of each coarse tree block as the number of linear keys associated with each coarse block.
pub fn compute_coarse_tree_weights<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree: &[MortonKey],
    comm: &C,
) -> Vec<usize> {
    let rank = comm.rank();
    // We want to partition the coarse tree. But we need the correct weights. The idea
    // is that we use the number of original leafs that intersect with the coarse tree
    // as leafs. In order to compute this we send the coarse tree around to all processes
    // so that each process computes for each coarse tree element how many of its keys
    // intersect with each node of the coarse tree. We then sum up the local weight for each
    // coarse tree node across all nodes to get the weight.

    let global_coarse_tree = gather_to_all(coarse_tree, comm);

    // We also want to send around a corresponding array of ranks so that for each global coarse tree key
    // we have the rank of where it originates from.

    let coarse_tree_ranks = gather_to_all(&vec![rank as usize; coarse_tree.len()], comm);

    // We now compute the local weights.
    let mut local_weight_contribution = vec![0; global_coarse_tree.len()];

    // In the following loop we want to be a bit smart. We do not iterate through all the local elements.
    // We know that our keys are sorted and also that the coarse tree keys are sorted. So we find the region
    // of our sorted keys that overlaps with the coarse tree region.

    // Let's find the start of our region. The start of our region is a coarse key that is an ancestor
    // of our current key. This works because the coarse tree has levels at most as high as the sorted keys.

    let first_key = *linear_keys.first().unwrap();

    let first_coarse_index = global_coarse_tree
        .iter()
        .take_while(|coarse_key| !coarse_key.is_ancestor(first_key))
        .count();

    // Now we need to find the end index of our region. For this again we find the index of our coarse tree that
    // is an ancestor of our last key.
    let last_key = *linear_keys.last().unwrap();

    let last_coarse_index = global_coarse_tree
        .iter()
        .take_while(|coarse_key| !coarse_key.is_ancestor(last_key))
        .count();

    // We now only need to iterate through between the first and last coarse index in the coarse tree.
    // In the way we have computed the indices. The last coarse index is inclusive (it is the ancestor of our last key).

    for (w, &global_coarse_key) in izip!(
        local_weight_contribution[first_coarse_index..=last_coarse_index].iter_mut(),
        global_coarse_tree[first_coarse_index..=last_coarse_index].iter()
    ) {
        *w += linear_keys
            .iter()
            .filter(|&&key| global_coarse_key.is_ancestor(key))
            .count();
    }

    // We now need to sum up the weights across all processes.

    let mut global_weights = vec![0; global_coarse_tree.len()];

    comm.all_reduce_into(
        &local_weight_contribution,
        &mut global_weights,
        SystemOperation::sum(),
    );

    // Each process now has all weights. However, we only need the ones for the current process.
    // So we just filter the rest out.

    izip!(coarse_tree_ranks, global_weights)
        .filter_map(|(r, weight)| {
            if r == rank as usize {
                Some(weight)
            } else {
                None
            }
        })
        .collect_vec()
}

/// Redistribute sorted keys with respect to a linear coarse tree.
pub fn redistribute_with_respect_to_coarse_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    coarse_tree: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    let size = comm.size();

    if size == 1 {
        return linear_keys.to_vec();
    }

    // We want to globally redistribute keys so that the keys on each process are descendents
    // of the local coarse tree keys.

    // We are using here the fact that the coarse tree is complete and sorted.
    // We are sending around to each process the first local index. This
    // defines bins in which we sort our keys. The keys are then sent around to the correct
    // processes via an alltoallv operation.

    let my_first = coarse_tree.first().unwrap();

    let global_bins = gather_to_all(std::slice::from_ref(my_first), comm);

    // We now have our bins. We go through our keys and store how
    // many keys are assigned to each rank. We are using here that
    // our keys and the coarse tree are both sorted.

    // This will store for each rank how many keys will be assigned to it.

    let rank_counts = sort_to_bins(linear_keys, &global_bins)
        .iter()
        .map(|&elem| elem as i32)
        .collect_vec();

    // We now have the counts for each rank. Let's redistribute accordingly and return.

    let result = redistribute(linear_keys, &rank_counts, comm);

    #[cfg(debug_assertions)]
    {
        // Check through that the first and last key of result are descendents
        // of the first and last coarse bloack.
        debug_assert!(coarse_tree
            .first()
            .unwrap()
            .is_ancestor(*result.first().unwrap()));
        debug_assert!(coarse_tree
            .last()
            .unwrap()
            .is_ancestor(*result.last().unwrap()));
    }

    result
}

/// Return a complete tree generated from local keys and associated coarse keys.
///
/// The coarse keys are refined until the maximum level is reached or until each coarse key
/// is the ancestor of at most `max_keys` fine keys.
/// It is assumed that the level of the fine keys is at least as large as `max_level`.
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
    // is associated with a coarse slice.

    let bins = coarse_keys.to_vec();

    let counts = sort_to_bins(sorted_fine_keys, &bins);

    // We now know how many fine keys are associated with each coarse block. We iterate
    // through and locally refine for each block that requires it.

    let mut remainder = sorted_fine_keys;
    let mut refined_keys = Vec::<MortonKey>::new();

    for (&count, &coarse_key) in izip!(counts.iter(), coarse_keys.iter()) {
        let current;
        (current, remainder) = remainder.split_at(count);
        if coarse_key.level() < max_level && current.len() > max_keys {
            // We need to refine the current split.
            refined_keys.extend_from_slice(
                create_local_tree(
                    current,
                    coarse_key.children().as_slice(),
                    max_level,
                    max_keys,
                )
                .as_slice(),
            );
        } else {
            refined_keys.push(coarse_key)
        }
    }

    refined_keys
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

    let sorted_keys = parsort(keys, comm, rng);

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

    debug_assert!(is_linear_tree(&result, comm));

    result
}

/// Balance a sorted list of Morton keys across processors given an array of corresponding weights.
pub fn load_balance<C: CommunicatorCollectives>(
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

    let scan = global_inclusive_cumsum(weights, comm);

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

    // Sort the elements into bins according to which process they should be sent.
    // We do not need to sort the Morton keys themselves into bins but the scanned weights.
    // The corresponding counts are the right counts for the Morton keys.

    let mut bins = Vec::<usize>::with_capacity(size as usize);

    for p in 1..=size as usize {
        if p <= k {
            bins.push((p - 1) * (1 + w));
        } else {
            bins.push((p - 1) * w + k);
        }
    }

    let counts = sort_to_bins(&scan, &bins)
        .iter()
        .map(|elem| *elem as i32)
        .collect_vec();

    // Now distribute the data with an all to all v.
    // We create a vector of how many elements to send to each process and
    // then send the actual data.

    let mut recvbuffer = redistribute(sorted_keys, &counts, comm);

    recvbuffer.sort_unstable();
    recvbuffer
}

/// Given a distributed set of linear keys, generate a complete tree.
pub fn complete_tree<C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    let mut linear_keys = linear_keys.to_vec();

    debug_assert!(is_linear_tree(&linear_keys, comm));

    let size = comm.size();
    let rank = comm.rank();

    if size == 1 {
        return MortonKey::complete_tree(linear_keys.as_slice());
    }

    // Now insert on the first and last process the first and last child of the
    // finest ancestor of first/last box on deepest level

    // Send first element to previous rank and insert into local keys.
    // On the first process we also need to insert the first child of the finest
    // ancestor of the deepest first key and first element. Correspondingly on the last process
    // we need to insert the last child of the finest ancester of the deepest last key and last element.

    let next_key = communicate_back(&linear_keys, comm);

    if rank < size - 1 {
        linear_keys.push(next_key.unwrap());
    }

    // Now fix the first key on the first rank.

    if rank == 0 {
        let first_key = linear_keys.first().unwrap();
        let deepest_first = MortonKey::deepest_first();
        if !first_key.is_ancestor(deepest_first) {
            let ancestor = deepest_first.finest_common_ancestor(*first_key);
            linear_keys.insert(0, ancestor.children()[0]);
        }
    }

    if rank == size - 1 {
        let last_key = linear_keys.last().unwrap();
        let deepest_last = MortonKey::deepest_last();
        if !last_key.is_ancestor(deepest_last) {
            let ancestor = deepest_last.finest_common_ancestor(*last_key);
            linear_keys.push(ancestor.children()[NSIBLINGS - 1]);
        }
    }

    // Now complete the regions defined by the keys on each process.

    let mut result = Vec::<MortonKey>::new();

    for (&key1, &key2) in linear_keys.iter().tuple_windows() {
        result.push(key1);
        result.extend_from_slice(key1.fill_between_keys(key2).as_slice());
    }

    if rank == size - 1 {
        result.push(*linear_keys.last().unwrap());
    }

    debug_assert!(is_complete_linear_tree(&result, comm));

    result
}

/// Balance a distributed tree.
pub fn balance<R: Rng, C: CommunicatorCollectives>(
    linear_keys: &[MortonKey],
    rng: &mut R,
    comm: &C,
) -> Vec<MortonKey> {
    // Treat the case that the length of the keys is one and is only the root.
    // This would lead to an empty output below as we only iterate up to level 1.

    if linear_keys.len() == 1 && *linear_keys.first().unwrap() == MortonKey::root() {
        return vec![MortonKey::root()];
    }

    let deepest_level = deepest_level(linear_keys, comm);

    // Start with keys at deepest level
    let mut work_list = linear_keys
        .iter()
        .copied()
        .filter(|&key| key.level() == deepest_level)
        .collect_vec();

    let mut result = Vec::<MortonKey>::new();

    // Now go through and make sure that for each key siblings and neighbours of parents are added

    for level in (1..=deepest_level).rev() {
        let mut parents = HashSet::<MortonKey>::new();
        let mut new_work_list = Vec::<MortonKey>::new();
        // We filter the work list by level and also make sure that
        // only one sibling of each of the parents children is added to
        // our current level list.
        for key in work_list.iter() {
            let parent = key.parent();
            if !parents.contains(&parent) {
                parents.insert(parent);
                result.extend_from_slice(key.siblings().as_slice());
                new_work_list.extend_from_slice(
                    parent
                        .neighbours()
                        .iter()
                        .copied()
                        .filter(|&key| key.is_valid())
                        .collect_vec()
                        .as_slice(),
                );
            }
        }
        new_work_list.extend(
            linear_keys
                .iter()
                .copied()
                .filter(|&key| key.level() == level - 1),
        );

        work_list = new_work_list;
    }

    let result = linearize(&result, rng, comm);

    debug_assert!(is_complete_linear_and_balanced(&result, comm));
    result
}

/// Return true if the keys are linear.
pub fn is_linear_tree<C: CommunicatorCollectives>(arr: &[MortonKey], comm: &C) -> bool {
    let mut is_linear = true;

    for (&key1, &key2) in arr.iter().tuple_windows() {
        if key1 >= key2 || key1.is_ancestor(key2) {
            is_linear = false;
            break;
        }
    }

    if comm.size() == 1 {
        return is_linear;
    }

    // Now check the interfaces

    if let Some(next_key) = communicate_back(arr, comm) {
        let last = *arr.last().unwrap();
        if last >= next_key || last.is_ancestor(next_key) {
            is_linear = false;
        }
    }

    let mut global_is_linear = false;

    comm.all_reduce_into(
        &is_linear,
        &mut global_is_linear,
        SystemOperation::logical_and(),
    );

    global_is_linear
}

/// Redistribute points with respect to a given coarse tree
pub fn redistribute_points_with_respect_to_coarse_tree<C: CommunicatorCollectives>(
    points: &[Point],
    morton_keys_for_points: &[MortonKey],
    coarse_tree: &[MortonKey],
    comm: &C,
) -> (Vec<Point>, Vec<MortonKey>) {
    if comm.size() == 1 {
        return (points.to_vec(), morton_keys_for_points.to_vec());
    }

    pub fn argsort<T: Ord + Copy>(arr: &[T]) -> Vec<usize> {
        let mut sort_indices = (0..arr.len()).collect_vec();
        sort_indices.sort_unstable_by_key(|&index| arr[index]);
        sort_indices
    }

    pub fn reorder<T: Copy>(arr: &[T], permutation: &[usize]) -> Vec<T> {
        let mut reordered = Vec::<T>::with_capacity(arr.len());
        for &index in permutation.iter() {
            reordered.push(arr[index])
        }
        reordered
    }

    assert_eq!(points.len(), morton_keys_for_points.len());

    let size = comm.size();

    if size == 1 {
        return (points.to_vec(), morton_keys_for_points.to_vec());
    }

    let sort_indices = argsort(morton_keys_for_points);
    let sorted_keys = reorder(morton_keys_for_points, &sort_indices);
    let sorted_points = reorder(points, &sort_indices);

    // Now get the bins

    let my_first = coarse_tree.first().unwrap();

    let global_bins = gather_to_all(std::slice::from_ref(my_first), comm);

    // We now sort the morton indices into the bins.

    // This will store for each rank how many keys will be assigned to it.

    let counts = sort_to_bins(&sorted_keys, &global_bins)
        .iter()
        .map(|&elem| elem as i32)
        .collect_vec();

    // We now redistribute the points and the corresponding keys.

    let (distributed_points, distributed_keys) = (
        redistribute(&sorted_points, &counts, comm),
        redistribute(&sorted_keys, &counts, comm),
    );

    // Now sort the distributed points and keys internally again.

    let sort_indices = argsort(&distributed_keys);
    let sorted_keys = reorder(&distributed_keys, &sort_indices);
    let sorted_points = reorder(&distributed_points, &sort_indices);

    (sorted_points, sorted_keys)
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

    if comm.rank() == 0 && !arr.first().unwrap().is_ancestor(MortonKey::deepest_first()) {
        complete_linear = false;
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

/// Return the deepest level of a distributed list of Morton keys.
pub fn deepest_level<C: CommunicatorCollectives>(keys: &[MortonKey], comm: &C) -> usize {
    let local_deepest_level = keys.iter().map(|elem| elem.level()).max().unwrap();

    if comm.size() == 1 {
        return local_deepest_level;
    }

    let mut global_deepest_level: usize = 0;

    comm.all_reduce_into(
        &local_deepest_level,
        &mut global_deepest_level,
        SystemOperation::max(),
    );

    global_deepest_level
}

/// Check if tree is balanced.
pub fn is_complete_linear_and_balanced<C: CommunicatorCollectives>(
    arr: &[MortonKey],
    comm: &C,
) -> bool {
    // Send the tree to the root node and check there that it is balanced.

    let mut balanced = false;

    if let Some(arr) = gather_to_root(arr, comm) {
        balanced = MortonKey::is_complete_linear_and_balanced(&arr);
    }

    comm.process_at_rank(0).broadcast_into(&mut balanced);

    balanced
}

/// For a complete linear bin get on each process the first key of all processes.
///
/// This information can be used to query on which process a key is living.
pub fn get_tree_bins<C: CommunicatorCollectives>(
    complete_linear_tree: &[MortonKey],
    comm: &C,
) -> Vec<MortonKey> {
    gather_to_all(
        std::slice::from_ref(complete_linear_tree.first().unwrap()),
        comm,
    )
}

/// For a sorted array return either position of the key or positioin directly before search key.
pub fn get_key_index(arr: &[MortonKey], key: MortonKey) -> usize {
    // Does a binary search of the key. If the key is found with Ok(..)
    // the exact index is returned of the found key. If the key is not found
    // the closest larger index is returned. So we subtract one to get the closest
    // smaller index.

    match arr.binary_search(&key) {
        Ok(index) => index,
        Err(index) => index - 1,
    }
}

/// Generate a map that associates each leaf with the corresponding point indices.
pub fn assign_points_to_leaf_keys(
    point_keys: &[MortonKey],
    leaf_keys: &[MortonKey],
) -> HashMap<MortonKey, Vec<usize>> {
    let mut point_map = HashMap::<MortonKey, Vec<usize>>::new();

    for (index, point_key) in point_keys.iter().enumerate() {
        let leaf_key_index = get_key_index(leaf_keys, *point_key);

        let leaf_key = leaf_keys[leaf_key_index];
        debug_assert!(leaf_key.is_ancestor(*point_key));

        point_map
            .entry(leaf_key)
            .or_insert(Vec::<usize>::new())
            .push(index);
    }

    point_map
}

/// Check if a key is associated with the current rank.
///
/// Note that the key does not need to exist as leaf. It just needs
/// to be descendent of a coarse key on the current rank.
pub fn key_on_current_rank(
    key: MortonKey,
    coarse_tree_bounds: &[MortonKey],
    rank: usize,
    size: usize,
) -> bool {
    if rank == size - 1 {
        key >= *coarse_tree_bounds.last().unwrap()
    } else {
        coarse_tree_bounds[rank] <= key && key < coarse_tree_bounds[rank + 1]
    }
}

/// Generate all leaf and interior keys.
pub fn generate_all_keys<C: CommunicatorCollectives>(
    leaf_tree: &[MortonKey],
    coarse_tree: &[MortonKey],
    coarse_tree_bounds: &[MortonKey],
    comm: &C,
) -> HashMap<MortonKey, KeyType> {
    /// This struct combines rank and key information for sending ghosts to neighbors.
    #[derive(Copy, Clone, Equivalence)]
    struct KeyWithRank {
        key: MortonKey,
        rank: usize,
    }

    let rank = comm.rank() as usize;
    let size = comm.size() as usize;

    let mut all_keys = HashMap::<MortonKey, KeyType>::new();
    let leaf_keys: HashSet<MortonKey> = HashSet::from_iter(leaf_tree.iter().copied());

    // If size == 1 we simply create locally the keys, so don't need to treat the global keys.

    if size > 1 {
        let mut global_keys = HashSet::<MortonKey>::new();

        // First deal with the parents of the coarse tree. These are different
        // as they may exist on multiple nodes, so receive a different label.

        for &key in coarse_tree {
            let mut parent = key.parent();
            while parent.level() > 0 && !all_keys.contains_key(&parent) {
                global_keys.insert(parent);
                parent = parent.parent();
            }
        }

        // We now send around the parents of the coarse tree to every node. These will
        // be global keys.

        let global_keys = gather_to_all(&global_keys.iter().copied().collect_vec(), comm);

        // We can now insert the global keys into `all_keys` with the `Global` label.

        for &key in &global_keys {
            all_keys.entry(key).or_insert(KeyType::Global);
        }
    }

    // We now deal with the fine leafs and their ancestors.
    // The leafs of the coarse tree will also be either part
    // of the fine tree leafs or will be interior keys. In either
    // case the following loop catches them.

    for leaf in leaf_keys {
        debug_assert!(!all_keys.contains_key(&leaf));
        all_keys.insert(leaf, KeyType::LocalLeaf);
        let mut parent = leaf.parent();
        while parent.level() > 0 && !all_keys.contains_key(&parent) {
            all_keys.insert(parent, KeyType::LocalInterior);
            parent = parent.parent();
        }
    }

    // Need to explicitly add the root at the end.
    all_keys.entry(MortonKey::root()).or_insert(KeyType::Global);

    // We only need to deal with ghosts if the size is larger than 1.

    if size > 1 {
        // This maps from rank to the keys that we want to send to the ranks

        let mut rank_send_ghost = HashMap::<usize, Vec<KeyWithRank>>::new();
        for index in 0..size {
            rank_send_ghost.insert(index, Vec::<KeyWithRank>::new());
        }

        let mut send_to_all = Vec::<KeyWithRank>::new();

        for (&key, &status) in all_keys.iter() {
            // We need not send around global keys to neighbors.
            if status == KeyType::Global {
                continue;
            }
            for &neighbor in key.neighbours().iter().filter(|&&key| key.is_valid()) {
                // If the neighbour is a global key then continue.
                if all_keys
                    .get(&neighbor)
                    .is_some_and(|&value| value == KeyType::Global)
                {
                    // Global keys exist on all nodes, so need to send their neighbors to all nodes.
                    send_to_all.push(KeyWithRank { key, rank });
                } else {
                    // Get rank of the neighbour
                    let neighbor_rank = get_key_index(coarse_tree_bounds, neighbor);
                    rank_send_ghost
                        .entry(neighbor_rank)
                        .and_modify(|keys| keys.push(KeyWithRank { key, rank }));
                }
            }
        }

        let send_ghost_to_all = gather_to_all(&send_to_all, comm);
        // We now know which key needs to be sent to which rank.
        // Turn to array, get the counts and send around.

        let (arr, counts) = {
            let mut arr = Vec::<KeyWithRank>::new();
            let mut counts = Vec::<i32>::new();
            for index in 0..size {
                let keys = rank_send_ghost.get(&index).unwrap();
                arr.extend(keys.iter());
                counts.push(keys.len() as i32);
            }
            (arr, counts)
        };

        // These are all the keys that are neighbors to our keys. We now go through
        // and store those that do not live on our tree as into `all_keys` with a label
        // of `Ghost`.
        let mut ghost_keys = redistribute(&arr, &counts, comm);
        // Add the neighbors of any global key.
        ghost_keys.extend(send_ghost_to_all.iter());

        for key in &ghost_keys {
            if key.rank == rank {
                // Don't need to add the keys that are already on the rank.
                continue;
            }
            all_keys.insert(key.key, KeyType::Ghost(key.rank));
        }
    }

    all_keys
}

#[cfg(test)]
mod test {
    use crate::{
        octree::get_key_index,
        tools::{generate_random_keys, seeded_rng},
    };

    #[test]
    fn test_get_key_rank() {
        let mut rng = seeded_rng(0);

        let mut keys = generate_random_keys(50, &mut rng);

        keys.sort_unstable();

        let mid = keys[25];

        assert_eq!(25, get_key_index(&keys, mid));

        // Now remove the mid index and do the same again.

        keys.remove(25);

        // The result should be 24.

        assert_eq!(24, get_key_index(&keys, mid));
    }
}
