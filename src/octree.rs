//! Definition of a linear octree
//!
//! A linear octree is a sorted collection of Morton keys.

use crate::{
    constants::{DEEPEST_LEVEL, LEVEL_SIZE, NSIBLINGS},
    morton::MortonKey,
};
use itertools::Itertools;

pub struct Octree {
    keys: Vec<MortonKey>,
}

pub fn remove_overlaps<Iter: Iterator<Item = MortonKey>>(keys: &[MortonKey]) -> Vec<MortonKey> {
    let mut new_keys = Vec::<MortonKey>::new();
    if keys.is_empty() {
        new_keys
    } else {
        for (m1, m2) in keys.iter().tuple_windows() {
            if m1 == m2 || m1.is_ancestor(*m2) {
                continue;
            }
            new_keys.push(*m1)
        }
        new_keys.push(*keys.last().unwrap());
        new_keys
    }
}

pub fn fill_between_two_keys(key1: MortonKey, key2: MortonKey) -> Vec<MortonKey> {
    // If one is the ancestor of the other the region between them is already filled.
    if key1.is_ancestor(key2) || key2.is_ancestor(key1) {
        return Vec::<MortonKey>::new();
    }

    // The finest common ancestor is always closer to the root than either key
    // if key1 is not an ancestor of key2 or vice versa.
    let ancestor = key1.finest_common_ancestor(key2);
    let children = ancestor.children();

    let mut result = Vec::<MortonKey>::new();

    let mut work_set = Vec::<MortonKey>::from_iter(children.iter().copied());

    while let Some(item) = work_set.pop() {
        // If the item is either key we don't want it in the result.
        if item == key1 || item == key2 {
            continue;
        }
        // We want items that are strictly between the two keys and are not ancestors of either.
        // We do not check specifically if item is an ancestor of key1 as then it would be smaller than key1.
        else if key1 < item && item < key2 && !item.is_ancestor(key2) {
            result.push(item);
        } else {
            // If the item is an ancestor of key1 or key2 just refine to the children and try again.
            // Note we already exclude that item is identical to key1 or key2.
            // So if item is an ancestor of either its children cannot have a level larger than key1 or key2.
            if item.is_ancestor(key1) || item.is_ancestor(key2) {
                let children = item.children();
                work_set.extend(children.iter());
            }
        }
    }

    result.sort_unstable();
    result
}

pub fn complete_region<Iter: Iterator<Item = MortonKey>>(keys: &[MortonKey]) -> Vec<MortonKey> {
    // First make sure that the input sequence is sorted.
    let mut keys = keys.to_vec();
    keys.sort_unstable();

    let mut result = Vec::<MortonKey>::new();

    // Special case of empty keys.
    if keys.len() == 0 {
        result.push(MortonKey::from_index_and_level([0, 0, 0], 0));
        return result;
    }

    // If a single element is given then just return the result if it is the root of the tree.
    if keys.len() == 1 {
        if result[0] == MortonKey::from_index_and_level([0, 0, 0], 0) {
            return result;
        }
    }

    let deepest_first = MortonKey::from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize);
    let deepest_last = MortonKey::from_index_and_level(
        [
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
        ],
        DEEPEST_LEVEL as usize,
    );

    // If the first key is not an ancestor of the deepest possible first element in the
    // tree get the finest ancestor between the two and use the first child of that.

    let first_key = *keys.first().unwrap();
    let last_key = *keys.last().unwrap();

    if !first_key.is_ancestor(deepest_first) {
        let ancestor = deepest_first.finest_common_ancestor(first_key);
        keys.insert(0, ancestor.children()[0]);
    }

    if !last_key.is_ancestor(deepest_last) {
        let ancestor = deepest_last.finest_common_ancestor(last_key);
        keys.push(ancestor.children()[NSIBLINGS - 1]);
    }

    // Now just iterate over the keys by tuples of two and fill the region between two keys.

    for (&key1, &key2) in keys.iter().tuple_windows() {
        result.push(key1);
        result.extend_from_slice(fill_between_two_keys(key1, key2).as_slice());
    }

    // Push the final key
    result.push(*keys.last().unwrap());
    // We do not sort the keys. They are already sorted.
    result
}
