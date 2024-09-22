//! Routines for working with Morton indices.

use crate::constants::{
    BYTE_DISPLACEMENT, BYTE_MASK, DEEPEST_LEVEL, DIRECTIONS, LEVEL_DISPLACEMENT, LEVEL_MASK,
    LEVEL_SIZE, NINE_BIT_MASK, NSIBLINGS, X_LOOKUP_DECODE, X_LOOKUP_ENCODE, Y_LOOKUP_DECODE,
    Y_LOOKUP_ENCODE, Z_LOOKUP_DECODE, Z_LOOKUP_ENCODE,
};
use crate::geometry::PhysicalBox;
use itertools::izip;
use itertools::Itertools;
use mpi::traits::Equivalence;
use std::collections::HashSet;

/// A morton key
///
/// This is a distinct type to distinguish from u64
/// numbers.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Equivalence)]
pub struct MortonKey {
    value: u64,
}

impl Default for MortonKey {
    fn default() -> Self {
        MortonKey::invalid_key()
    }
}

impl MortonKey {
    /// Create a new Morton key. Users should use `[MortonKey::from_index_and_level].`
    fn new(value: u64) -> Self {
        let key = Self { value };
        // Make sure that Morton keys are valid (only active in debug mode)
        debug_assert!(key.is_well_formed());
        key
    }

    /// A key that is not valid or well formed but guaranteed to be larger than any valid key.
    ///
    /// This is useful when a guaranteed upper bound is needed.
    pub fn upper_bound() -> Self {
        Self { value: u64::MAX }
    }

    /// Check if a key is invalid.
    pub fn invalid_key() -> Self {
        Self { value: 1 << 63 }
    }

    /// Check if key is valid.
    ///
    /// A key is not valid if its highest bit is 1.
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        // If the highest bit is 1 the key is by definition not valid.
        self.value >> 63 != 1
    }

    /// Create a root key
    #[inline(always)]
    pub fn root() -> MortonKey {
        Self { value: 0 }
    }

    /// Return the first deepest key.
    #[inline(always)]
    pub fn deepest_first() -> Self {
        MortonKey::from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize)
    }

    /// Return the last deepest key.
    #[inline(always)]
    pub fn deepest_last() -> Self {
        MortonKey::from_index_and_level(
            [
                LEVEL_SIZE as usize - 1,
                LEVEL_SIZE as usize - 1,
                LEVEL_SIZE as usize - 1,
            ],
            DEEPEST_LEVEL as usize,
        )
    }

    /// Return the associated physical box with respect to a bounding box.
    #[inline(always)]
    pub fn physical_box(&self, bounding_box: &PhysicalBox) -> PhysicalBox {
        let (level, [x, y, z]) = self.decode();
        let xind = x as f64;
        let yind = y as f64;
        let zind = z as f64;

        let [xmin, ymin, zmin, xmax, ymax, zmax] = bounding_box.coordinates();
        let level_size = (1 << level) as f64;
        let xlength = (xmax - xmin) / level_size;
        let ylength = (ymax - ymin) / level_size;
        let zlength = (zmax - zmin) / level_size;

        PhysicalBox::new([
            xmin + xind * xlength,
            ymin + yind * ylength,
            zmin + zind * zlength,
            xmin + (1.0 + xind) * xlength,
            ymin + (1.0 + yind) * ylength,
            zmin + (1.0 + zind) * zlength,
        ])
    }

    /// Return key in a given direction.
    ///
    /// Returns an invalid key if there is no valid key in that direction.
    pub fn key_in_direction(&self, direction: [i64; 3]) -> MortonKey {
        let (level, [x, y, z]) = self.decode();
        let level_size = 1 << level;

        let new_index = [
            x as i64 + direction[0],
            y as i64 + direction[1],
            z as i64 + direction[2],
        ];

        if 0 <= new_index[0]
            && new_index[0] < level_size
            && 0 <= new_index[1]
            && new_index[1] < level_size
            && 0 <= new_index[2]
            && new_index[2] < level_size
        {
            MortonKey::from_index_and_level(
                [
                    new_index[0] as usize,
                    new_index[1] as usize,
                    new_index[2] as usize,
                ],
                level,
            )
        } else {
            MortonKey::invalid_key()
        }
    }

    /// A key is ill-formed if it has non-zero bits and positions that should be zero by the given level.
    #[inline(always)]
    pub fn is_well_formed(&self) -> bool {
        let level = self.value & LEVEL_MASK;
        let key = self.value >> LEVEL_DISPLACEMENT;
        // Check that all the bits below the level of the key are zero.
        // Need to first create a suitable bitmask that has
        // all bits set to one at the last DEEPEST_LEVEL - level bits.
        let shift = 3 * (DEEPEST_LEVEL - level);
        // The mask has now bits set to one at the last `level_diff` bits
        let mask: u64 = (1 << shift) - 1;
        // Is zero if and only if all the bits of the key at the `level_diff` bits are zero.
        (mask & key) == 0
    }

    /// Map a physical point within a bounding box to a Morton key on a given level.
    /// It is assumed that points are strictly contained within the bounding box.
    pub fn from_physical_point(point: [f64; 3], bounding_box: &PhysicalBox, level: usize) -> Self {
        let level_size = 1 << level;
        let reference = bounding_box.physical_to_reference(point);
        let x = (reference[0] * level_size as f64) as usize;
        let y = (reference[1] * level_size as f64) as usize;
        let z = (reference[2] * level_size as f64) as usize;

        MortonKey::from_index_and_level([x, y, z], level)
    }

    /// Create a new key by providing the [x, y, z] index and a level.
    pub fn from_index_and_level(index: [usize; 3], level: usize) -> MortonKey {
        let level = level as u64;
        debug_assert!(level <= DEEPEST_LEVEL);

        debug_assert!(index[0] < (1 << level));
        debug_assert!(index[1] < (1 << level));
        debug_assert!(index[2] < (1 << level));

        // If we are not on the deepest level we need to shift the box.
        // The box with x-index one on DEEPEST_LEVEL-1 has index two on
        // DEEPEST_LEVEL.

        let level_diff = DEEPEST_LEVEL - level;

        let x = (index[0] as u64) << level_diff;
        let y = (index[1] as u64) << level_diff;
        let z = (index[2] as u64) << level_diff;

        let key: u64 = X_LOOKUP_ENCODE[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
            | Y_LOOKUP_ENCODE[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
            | Z_LOOKUP_ENCODE[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

        let key = (key << 24)
            | X_LOOKUP_ENCODE[(x & BYTE_MASK) as usize]
            | Y_LOOKUP_ENCODE[(y & BYTE_MASK) as usize]
            | Z_LOOKUP_ENCODE[(z & BYTE_MASK) as usize];

        let key = key << LEVEL_DISPLACEMENT;
        Self { value: key | level }
    }

    /// Return the level of a key.
    #[inline(always)]
    pub fn level(&self) -> usize {
        (self.value & LEVEL_MASK) as usize
    }

    /// Decode a key and return a tuple of the form (level, [x, y, z]), where the latter is the index vector.
    pub fn decode(&self) -> (usize, [usize; 3]) {
        fn decode_key_helper(key: u64, lookup_table: &[u64; 512]) -> u64 {
            const N_LOOPS: u64 = 6; // 48 bits for the keys. Process in pairs of 9. So 6 passes enough.
            let mut coord: u64 = 0;

            for index in 0..N_LOOPS {
                coord |=
                    lookup_table[((key >> (index * 9)) & NINE_BIT_MASK) as usize] << (3 * index);
            }

            coord
        }

        let level = self.level();
        let level_diff = DEEPEST_LEVEL - level as u64;

        let key = self.value >> LEVEL_DISPLACEMENT;

        let x = decode_key_helper(key, &X_LOOKUP_DECODE);
        let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
        let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

        let x = x >> level_diff;
        let y = y >> level_diff;
        let z = z >> level_diff;

        (level, [x as usize, y as usize, z as usize])
    }

    /// Return the parent of a key.
    #[inline(always)]
    pub fn parent(&self) -> Self {
        let level = self.level();
        assert!(level > 0);

        // We set the bits at our current level to zero and subtract 1 at the end to reduce the
        // level by one.

        let bit_displacement = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as u64);
        let mask = !(7 << bit_displacement);

        Self {
            value: (self.value & mask) - 1,
        }
    }

    /// Return ancestor of key on specified level
    ///
    /// Return None if level > self.level().
    /// Return the key itself if level == self.level().
    pub fn ancestor_at_level(&self, level: usize) -> Option<Self> {
        let my_level = self.level();

        if my_level < level {
            return None;
        }

        if my_level == level {
            return Some(*self);
        }

        let key = self.value >> LEVEL_DISPLACEMENT;

        let bit_displacement = 3 * (DEEPEST_LEVEL - level as u64);
        // Sets the last bits to zero and shifts back
        let key = (key >> bit_displacement) << (bit_displacement + LEVEL_DISPLACEMENT);

        Some(Self {
            value: key | level as u64,
        })
    }

    /// Check if key is ancestor of `other`. If keys are identical also returns true.
    #[inline(always)]
    pub fn is_ancestor(&self, other: MortonKey) -> bool {
        let my_level = self.level();
        let other_level = other.level();

        if !self.is_valid() || !other.is_valid() {
            return false;
        }

        if *self == other {
            true
        } else if my_level > other_level {
            false
        } else {
            // We shift both keys out to 3 * DEEPEST_LEVEL - my_level
            // This gives identical bit sequences if my_key is an ancestor of other_key
            let my_key = self.value >> (LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as u64));
            let other_key =
                other.value >> (LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as u64));

            my_key == other_key
        }
    }

    /// Return the finest common ancestor of two keys.
    ///
    /// If the keys are identical return the key itself.
    pub fn finest_common_ancestor(&self, other: MortonKey) -> MortonKey {
        if *self == other {
            return *self;
        }

        let my_level = self.level();
        let other_level = other.level();

        // Want to bring both keys to the minimum of the two levels.
        let level = my_level.min(other_level);

        // Remove the level information and bring second key to the same level as first key
        // After the following operation the least significant bits are associated with `first_level`.

        let mut first_key = self.value >> (LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as u64));
        let mut second_key =
            other.value >> (LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as u64));

        // Now move both keys up until they are identical.
        // At the same time we reduce the first level.

        let mut count = 0;

        while first_key != second_key {
            count += 1;
            first_key >>= 3;
            second_key >>= 3;
        }

        // We now return the ancestor at the given level.

        let new_level = level - count;

        first_key <<= 3 * (DEEPEST_LEVEL - new_level as u64) + LEVEL_DISPLACEMENT;

        MortonKey {
            value: first_key | new_level as u64,
        }
    }

    /// Return true if key is equal to the root key.
    #[inline(always)]
    pub fn is_root(&self) -> bool {
        self.value == 0
    }

    /// Return the 8 children of a key.
    #[inline(always)]
    pub fn children(&self) -> [MortonKey; 8] {
        let level = self.level() as u64;
        assert!(level != DEEPEST_LEVEL);

        let child_level = 1 + level;

        let shift = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - child_level);
        let key = self.value;
        [
            MortonKey::new(1 + (key | (0 << shift))),
            MortonKey::new(1 + (key | (1 << shift))),
            MortonKey::new(1 + (key | (2 << shift))),
            MortonKey::new(1 + (key | (3 << shift))),
            MortonKey::new(1 + (key | (4 << shift))),
            MortonKey::new(1 + (key | (5 << shift))),
            MortonKey::new(1 + (key | (6 << shift))),
            MortonKey::new(1 + (key | (7 << shift))),
        ]
    }

    /// Return the 8 siblings of a key.
    ///
    /// The key itself is part of the siblings.
    pub fn siblings(&self) -> [MortonKey; 8] {
        assert!(!self.is_root());
        self.parent().children()
    }

    /// Return the neighbours of a key.
    ///
    /// The key itself is not part of the neighbours.
    /// If along a certain direction there is no neighbour then
    ///  an invalid key is stored.
    pub fn neighbours(&self) -> [MortonKey; 26] {
        let mut result = [MortonKey::default(); 26];

        let (level, [x, y, z]) = self.decode();
        let level_size = 1 << level;

        for (direction, res) in izip!(DIRECTIONS, result.iter_mut()) {
            let new_index = [
                x as i64 + direction[0],
                y as i64 + direction[1],
                z as i64 + direction[2],
            ];
            if 0 <= new_index[0]
                && new_index[0] < level_size
                && 0 <= new_index[1]
                && new_index[1] < level_size
                && 0 <= new_index[2]
                && new_index[2] < level_size
            {
                *res = MortonKey::from_index_and_level(
                    [
                        new_index[0] as usize,
                        new_index[1] as usize,
                        new_index[2] as usize,
                    ],
                    level,
                );
            }
        }
        result
    }

    /// Return the index of the key as a child of the parent, i.e. 0, 1, ..., 7.
    #[inline(always)]
    pub fn child_index(&self) -> usize {
        if *self == MortonKey::root() {
            return 0;
        }
        let level = self.level() as u64;

        let shift = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level);

        ((self.value >> shift) % 8) as usize
    }

    /// Return the finest descendent that is opposite to the joint corner with the siblings.
    pub fn finest_outer_descendent(&self) -> MortonKey {
        // First find out which child the current key is.

        let level = self.level() as u64;

        if level == DEEPEST_LEVEL {
            return *self;
        }

        let mut child_level = 1 + level;
        let mut key = *self;
        let outer_index = self.child_index() as u64;

        while child_level <= DEEPEST_LEVEL {
            let shift = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - child_level);
            key = MortonKey::new(1 + (key.value | outer_index << shift));
            child_level += 1;
        }

        key
    }

    /// Return the next possible Morton key on the deepest level that is not a descendent of the current key.
    ///
    /// If the key is already the last possible key then return None.
    pub fn next_non_descendent_key(&self) -> Option<MortonKey> {
        // If we are an ancestor of deepest_last we return None as then there
        // is next key.

        if self.is_ancestor(MortonKey::deepest_last()) {
            return None;
        }

        let level = self.level() as u64;

        let level_diff = DEEPEST_LEVEL - level;
        let shift = LEVEL_DISPLACEMENT + 3 * level_diff;

        // Need to know which sibling we are.
        let child_index = ((self.value >> shift) % 8) as usize;
        // If we are between 0 and 6 take the next sibling and go to deepest level.
        if child_index < 7 {
            Some(MortonKey::new(self.value + (1 << shift) + level_diff))
        } else {
            // If we are the last child go to the parent and take next key from there.
            self.parent().next_non_descendent_key()
        }
    }

    /// Linearize by sorting and removing overlaps.
    pub fn linearize(keys: &[MortonKey]) -> Vec<MortonKey> {
        let mut new_keys = Vec::<MortonKey>::new();
        if keys.is_empty() {
            new_keys
        } else {
            let mut keys = keys.to_vec();
            keys.sort_unstable();
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

    /// Fill the region between two keys with a minimal number of keys.
    pub fn fill_between_keys(&self, key2: MortonKey) -> Vec<MortonKey> {
        // Make sure that key1 is smaller or equal key2
        let (key1, key2) = if *self < key2 {
            (*self, key2)
        } else {
            (key2, *self)
        };

        // If key1 is ancestor of key2 return empty list. Note that
        // is_ancestor is true if key1 is equal to key2.
        if key1.is_ancestor(key2) {
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

    /// Complete a tree ensuring that the given keys are part of the leafs.
    ///
    /// The given keys must not overlap.
    pub fn complete_tree(keys: &[MortonKey]) -> Vec<MortonKey> {
        // First make sure that the input sequence is sorted.
        let mut keys = keys.to_vec();
        keys.sort_unstable();

        let mut result = Vec::<MortonKey>::new();

        // Special case of empty keys.
        if keys.is_empty() {
            result.push(MortonKey::from_index_and_level([0, 0, 0], 0));
            return result;
        }

        // If just the root is given return that.
        if keys.len() == 1 && *keys.first().unwrap() == MortonKey::root() {
            return keys.to_vec();
        }

        let deepest_first = MortonKey::deepest_first();
        let deepest_last = MortonKey::deepest_last();

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
            result.extend_from_slice(key1.fill_between_keys(key2).as_slice());
        }

        // Push the final key
        result.push(*keys.last().unwrap());
        // We do not sort the keys. They are already sorted.
        result
    }

    /// Get all interior keys for an Octree represented by a list of Morton keys
    ///
    /// Adds the root at level 0 if the root is not a leaf if the octree.
    /// If `keys` only contains the root of the tree then the returned set is empty.
    pub fn get_interior_keys(keys: &[MortonKey]) -> HashSet<MortonKey> {
        let mut interior_keys = HashSet::<MortonKey>::new();

        let keys = MortonKey::linearize(keys);

        for &key in &keys {
            if key.level() > 0 {
                let mut p = key.parent();
                while p.level() > 0 && !interior_keys.contains(&p) {
                    interior_keys.insert(p);
                    p = p.parent();
                }
            }
        }

        if !keys.contains(&MortonKey::root()) {
            interior_keys.insert(MortonKey::root());
        }

        interior_keys
    }

    /// Return interior and leaf keys.
    pub fn get_interior_and_leaf_keys(keys: &[MortonKey]) -> HashSet<MortonKey> {
        let mut all_keys = MortonKey::get_interior_keys(keys);
        all_keys.extend(keys.iter());
        all_keys
    }

    /// Check if a list of Morton keys represent a complete linear tree.
    pub fn is_complete_linear_octree(keys: &[MortonKey]) -> bool {
        // First check if the list is sorted.
        for (key1, key2) in keys.iter().tuple_windows() {
            if key1 > key2 {
                return false;
            }
        }

        // Now check that all interior keys have 8 children.

        let interior_keys = MortonKey::get_interior_keys(keys);
        let mut all_keys = HashSet::<MortonKey>::from_iter(interior_keys.iter().copied());
        all_keys.extend(keys.iter());

        for key in interior_keys {
            let children = key.children();
            for child in children {
                if !all_keys.contains(&child) {
                    return false;
                }
            }
        }

        true
    }

    /// Balance a list of Morton keys with respect to a root key.
    pub fn balance(keys: &[MortonKey], root: MortonKey) -> Vec<MortonKey> {
        let keys = keys
            .iter()
            .copied()
            .filter(|&key| root.is_ancestor(key))
            .collect_vec();

        if keys.is_empty() {
            return Vec::<MortonKey>::new();
        }

        let deepest_level = keys.iter().map(|key| key.level()).max().unwrap();
        let root_level = root.level();

        if deepest_level == root_level {
            return vec![root];
        }

        // Start with keys at deepest level
        let mut work_list = keys
            .iter()
            .copied()
            .filter(|&key| key.level() == deepest_level)
            .collect_vec();

        let mut result = Vec::<MortonKey>::new();

        // Now go through and make sure that for each key siblings and neighbours of parents are added

        for level in ((1 + root_level)..=deepest_level).rev() {
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
                            .filter(|&key| root.is_ancestor(key))
                            .collect_vec()
                            .as_slice(),
                    );
                }
            }
            new_work_list.extend(keys.iter().copied().filter(|&key| key.level() == level - 1));

            work_list = new_work_list;
            // Now extend the work list with the
        }

        MortonKey::linearize(result.as_slice())
    }

    /// Returns true if an Octree is linear, complete, and, balanced.
    pub fn is_complete_linear_and_balanced(keys: &[MortonKey]) -> bool {
        // First check that it is complete and linear.

        if !MortonKey::is_complete_linear_octree(keys) {
            return false;
        }

        // Now check that it is balanced.
        // We add for each key the neighbors of the parents. If
        // we then linearize and the set of keys is identicial the octree
        // was balanced. Otherwise, some key are replaced in the linearisation
        // through desendents on a deeper level and the two lists are not
        // identical.

        let mut new_keys = keys.to_vec();
        for key in keys {
            new_keys.extend(
                key.parent()
                    .neighbours()
                    .iter()
                    .copied()
                    .filter(|&key| key.is_valid()),
            );
        }

        let new_keys = MortonKey::linearize(&new_keys);

        if new_keys.len() != keys.len() {
            return false;
        } else {
            for (&key1, key2) in izip!(keys, new_keys) {
                if key1 != key2 {
                    return false;
                }
            }
        }

        true
    }
}

impl std::fmt::Display for MortonKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (level, [x, y, z]) = self.decode();
        write!(
            f,
            "(level: {}, x: {}, y: {}, z: {}, value: {})",
            level, x, y, z, self.value
        )
    }
}

impl std::fmt::Debug for MortonKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (level, index) = self.decode();
        f.debug_struct("MortonKey")
            .field("level", &level)
            .field("x", &index[0])
            .field("y", &index[1])
            .field("z", &index[2])
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_z_decode_table() {
        for (index, &actual) in Z_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = (index & 1) as u64;
            expected |= (((index >> 3) & 1) << 1) as u64;
            expected |= (((index >> 6) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_y_decode_table() {
        for (index, &actual) in Y_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = ((index >> 1) & 1) as u64;
            expected |= (((index >> 4) & 1) << 1) as u64;
            expected |= (((index >> 7) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_x_decode_table() {
        for (index, &actual) in X_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = ((index >> 2) & 1) as u64;
            expected |= (((index >> 5) & 1) << 1) as u64;
            expected |= (((index >> 8) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_z_encode_table() {
        for (mut index, actual) in Z_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_y_encode_table() {
        for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 1)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_x_encode_table() {
        for (mut index, actual) in X_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 2)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_encoding_decoding() {
        let index: [usize; 3] = [
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
        ];

        let key = MortonKey::from_index_and_level(index, DEEPEST_LEVEL as usize);

        let (level, actual) = key.decode();

        assert_eq!(level, DEEPEST_LEVEL as usize);
        assert_eq!(index, actual);
    }

    #[test]
    fn test_parent() {
        let index = [15, 39, 45];
        let key = MortonKey::from_index_and_level(index, 9);
        let parent = key.parent();

        let expected_index = [7, 19, 22];
        let (actual_level, actual_index) = parent.decode();
        assert_eq!(actual_level, 8);
        assert_eq!(actual_index, expected_index);
    }

    #[test]
    fn test_ancestor() {
        let index = [15, 39, 45];
        let key = MortonKey::from_index_and_level(index, 9);
        assert!(key.is_ancestor(key));
        let ancestor = key.parent().parent();
        assert!(ancestor.is_ancestor(key));
    }

    #[test]
    fn test_ancestor_at_level() {
        let index = [15, 39, 45];
        let key = MortonKey::from_index_and_level(index, 9);
        assert!(key.is_ancestor(key));
        let ancestor = key.parent().parent();
        assert!(key.ancestor_at_level(10).is_none());
        assert_eq!(key.ancestor_at_level(9).unwrap(), key);
        assert_eq!(ancestor, key.ancestor_at_level(7).unwrap());
    }

    #[test]
    fn test_finest_ancestor() {
        let index = [15, 39, 45];

        let key = MortonKey::from_index_and_level(index, 9);
        // The finest ancestor with itself is the key itself.
        assert_eq!(key.finest_common_ancestor(key), key);
        // Finest ancestor with ancestor two levels up is the ancestor.
        let ancestor = key.parent().parent();
        assert_eq!(key.finest_common_ancestor(ancestor), ancestor);

        // Finest ancestor  of the following keys should be the root of the tree.

        let key1 = MortonKey::from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize - 1);
        let key2 = MortonKey::from_index_and_level(
            [
                LEVEL_SIZE as usize - 1,
                LEVEL_SIZE as usize - 1,
                LEVEL_SIZE as usize - 1,
            ],
            DEEPEST_LEVEL as usize,
        );

        assert_eq!(
            key1.finest_common_ancestor(key2),
            MortonKey::from_index_and_level([0, 0, 0], 0)
        );

        // The finest ancestor of these two keys should be at level 1.

        let key1 = MortonKey::from_index_and_level([0, 0, 62], 6);
        let key2 = MortonKey::from_index_and_level([0, 0, 63], 6);
        let expected = MortonKey::from_index_and_level([0, 0, 31], 5);

        assert_eq!(key1.finest_common_ancestor(key2), expected);
    }

    #[test]
    fn test_children() {
        let key = MortonKey::from_index_and_level([4, 9, 8], 4);
        let children = key.children();

        // Check that all the children are different.

        let children_set =
            std::collections::HashSet::<MortonKey>::from_iter(children.iter().copied());
        assert_eq!(children_set.len(), 8);

        // Check that all children are on the correct level and that their parent is our key.

        for child in children {
            assert_eq!(child.level(), 5);
            assert_eq!(child.parent(), key);
        }
    }

    #[test]
    fn test_print() {
        let key = MortonKey::from_index_and_level([0, 2, 4], 3);
        // let key = MortonKey::new(13194139533315);

        println!("Key: {}", key);
    }

    #[test]
    fn test_fill_between_keys() {
        // Do various checks
        fn sanity_checks(key1: MortonKey, key2: MortonKey, mut keys: Vec<MortonKey>) {
            // Check that keys are strictly sorted and that no key is ancestor of the next key.

            let min_level = key1.level().min(key2.level());

            keys.insert(0, key1);
            keys.push(key2);

            for (k1, k2) in keys.iter().tuple_windows() {
                assert!(k1 < k2);
                assert!(!key1.is_ancestor(key2));
            }

            // Check that level not higher than min_level

            for k in keys.iter() {
                assert!(k.level() <= min_level);
            }
        }

        // Correct result for keys on one level
        let key1 = MortonKey::from_index_and_level([0, 1, 0], 4);
        let key2 = MortonKey::from_index_and_level([8, 4, 13], 4);
        let keys = key1.fill_between_keys(key2);
        assert!(!keys.is_empty());

        sanity_checks(key1, key2, keys);

        // Correct result for passing same key twice

        let keys = key1.fill_between_keys(key1);
        assert!(keys.is_empty());

        // Two consecutive keys should also be empty.

        let children = key2.children();

        let keys = children[1].fill_between_keys(children[2]);
        assert!(keys.is_empty());
    }

    #[test]
    pub fn test_complete_region() {
        // Do various checks
        fn sanity_checks(keys: &[MortonKey], complete_region: &[MortonKey]) {
            // Check that keys are strictly sorted and that no key is ancestor of the next key.

            if !keys.is_empty() {
                // Min level of input keys.
                let max_level = keys.iter().max_by_key(|item| item.level()).unwrap().level();
                for k in complete_region.iter() {
                    assert!(k.level() <= max_level);
                }
            }

            // Check that completed region has sorted keys and no overlaps.
            for (&k1, &k2) in complete_region.iter().tuple_windows() {
                assert!(k1 < k2);
                assert!(!k1.is_ancestor(k2));
            }

            // Check that level not higher than min_level

            // Check that first key is ancestor of first in deepest level
            // and that last key is ancestor of last in deepest level.
            let deepest_first = MortonKey::from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize);
            let deepest_last = MortonKey::from_index_and_level(
                [
                    LEVEL_SIZE as usize - 1,
                    LEVEL_SIZE as usize - 1,
                    LEVEL_SIZE as usize - 1,
                ],
                DEEPEST_LEVEL as usize,
            );

            assert!(complete_region.first().unwrap().is_ancestor(deepest_first));
            assert!(complete_region.last().unwrap().is_ancestor(deepest_last));
        }

        // Create 3 Morton keys around which to complete region.

        let key1 = MortonKey::from_index_and_level([17, 30, 55], 10);
        let key2 = MortonKey::from_index_and_level([17, 540, 55], 10);
        let key3 = MortonKey::from_index_and_level([17, 30, 799], 11);

        let keys = [key1, key2, key3];

        let complete_region = MortonKey::complete_tree(keys.as_slice());

        sanity_checks(keys.as_slice(), complete_region.as_slice());

        // For an empty slice the complete region method should just add the root of the tree.
        let keys = Vec::<MortonKey>::new();
        let complete_region = MortonKey::complete_tree(keys.as_slice());
        assert_eq!(complete_region.len(), 1);

        sanity_checks(keys.as_slice(), complete_region.as_slice());

        // Choose a region where the first and last key are ancestors of deepest first and deepest last.

        let keys = [MortonKey::deepest_first(), MortonKey::deepest_last()];

        let complete_region = MortonKey::complete_tree(keys.as_slice());

        sanity_checks(keys.as_slice(), complete_region.as_slice());
    }

    #[test]
    pub fn test_neighbour_directions_unique() {
        let neighbour_set: HashSet<[i64; 3]> = HashSet::from_iter(DIRECTIONS.iter().copied());
        assert_eq!(neighbour_set.len(), 26);
    }

    #[test]
    pub fn test_invalid_keys() {
        let invalid_key = MortonKey::invalid_key();

        // Make sure that an invalid key is invalid.
        assert!(!invalid_key.is_valid());

        // Make sure that an invalid key is not ill-formed.
        assert!(invalid_key.is_well_formed());
    }

    #[test]
    pub fn test_neighbours() {
        // Check that root only has invalid neighbors.
        let neighbours = MortonKey::root().neighbours();
        for key in neighbours {
            assert!(!key.is_valid());
        }

        // Now check inside a tree that all neighbours exist and that their distance to key corresponds
        // to the corresponding directions vector.

        let index = [33, 798, 56];
        let level = 11;
        let key = MortonKey::from_index_and_level(index, level);
        let neighbours = key.neighbours();
        for (dir, key) in izip!(DIRECTIONS, neighbours) {
            assert!(key.is_valid());
            let (level, key_index) = key.decode();
            assert_eq!(key.level(), level);
            let direction: [i64; 3] = [
                key_index[0] as i64 - index[0] as i64,
                key_index[1] as i64 - index[1] as i64,
                key_index[2] as i64 - index[2] as i64,
            ];
            assert_eq!(direction, dir);
        }
    }

    #[test]
    pub fn test_key_in_direction() {
        // Now check inside a tree that all neighbours exist and that their distance to key corresponds
        // to the corresponding directions vector.

        let index = [33, 798, 56];
        let dir = [2, 5, -3];
        let level = 11;
        let key = MortonKey::from_index_and_level(index, level);
        let new_key = key.key_in_direction(dir);

        let (new_level, new_index) = new_key.decode();
        assert_eq!(new_level, level);
        assert_eq!(new_index[0] as i64, index[0] as i64 + dir[0]);
        assert_eq!(new_index[1] as i64, index[1] as i64 + dir[1]);
        assert_eq!(new_index[2] as i64, index[2] as i64 + dir[2]);

        // Now test a direction that gives an invalid key.

        let dir = [-34, 798, 56];
        let new_key = key.key_in_direction(dir);
        assert!(!new_key.is_valid());
    }

    #[test]
    pub fn test_balanced() {
        // Balance the second level of a tree.

        let balanced = MortonKey::balance(
            [MortonKey::from_index_and_level([0, 1, 0], 2)].as_slice(),
            MortonKey::root(),
        );

        assert!(MortonKey::is_complete_linear_octree(&balanced));

        // Try a few keys on deeper levels

        let key1 = MortonKey::from_index_and_level([17, 35, 48], 9);
        let key2 = MortonKey::from_index_and_level([355, 25, 67], 9);
        let key3 = MortonKey::from_index_and_level([0, 0, 0], 8);

        // Just make sure one is not ancestor of the other. Does not matter for routine.
        // But want to avoid for unit test checks.
        assert!(!key3.is_ancestor(key1));

        let balanced = MortonKey::balance([key1, key2, key3].as_slice(), MortonKey::root());

        assert!(MortonKey::is_complete_linear_octree(&balanced));

        // Let us now check balancing with respec to a single given key.

        // We start with all keys on level 1. We replace the first key
        // by its descendents two levels down and linearize. The
        // resulting octree is complete and linear but not balanced.
        // However, the subtree under [0, 0, 0,] on level 1 is balanced.

        let mut keys = vec![
            MortonKey::from_index_and_level([0, 0, 0], 1),
            MortonKey::from_index_and_level([0, 0, 1], 1),
            MortonKey::from_index_and_level([0, 1, 0], 1),
            MortonKey::from_index_and_level([1, 0, 0], 1),
            MortonKey::from_index_and_level([0, 1, 1], 1),
            MortonKey::from_index_and_level([1, 1, 0], 1),
            MortonKey::from_index_and_level([1, 0, 1], 1),
            MortonKey::from_index_and_level([1, 1, 1], 1),
        ];

        // We recurse twice to get 64 children on level 3 from the first box.
        let seed = MortonKey::from_index_and_level([0, 0, 0], 1);
        let children = seed.children();
        let mut descendents = Vec::<MortonKey>::new();
        for child in children {
            descendents.extend(child.children());
        }

        // We add all those children to the tree and linearize.
        keys.extend(descendents.iter());

        let keys = MortonKey::linearize(&keys);

        let subtree_balanced =
            MortonKey::balance(&keys, MortonKey::from_index_and_level([0, 0, 0], 1));

        // Check that this balanced subtree has 64 elements.

        assert_eq!(subtree_balanced.len(), 64);

        // Check that each of the subtree elements lives on level 3.

        for &key in &subtree_balanced {
            assert_eq!(key.level(), 3);
        }
    }

    #[test]
    pub fn test_is_complete_linear_and_balanced() {
        // First we create an unbalanced Octree.
        // We start with all keys at level 1 and then recurse one of the keys
        // two times and linearize.

        let mut keys = vec![
            MortonKey::from_index_and_level([0, 0, 0], 1),
            MortonKey::from_index_and_level([0, 0, 1], 1),
            MortonKey::from_index_and_level([0, 1, 0], 1),
            MortonKey::from_index_and_level([1, 0, 0], 1),
            MortonKey::from_index_and_level([0, 1, 1], 1),
            MortonKey::from_index_and_level([1, 1, 0], 1),
            MortonKey::from_index_and_level([1, 0, 1], 1),
            MortonKey::from_index_and_level([1, 1, 1], 1),
        ];

        // We recurse twice to get 64 children on level 3 from the first box.
        let seed = MortonKey::from_index_and_level([0, 0, 0], 1);
        let children = seed.children();
        let mut descendents = Vec::<MortonKey>::new();
        for child in children {
            descendents.extend(child.children());
        }

        // We add all those children to the tree and linearize.
        keys.extend(descendents.iter());

        let keys = MortonKey::linearize(&keys);

        // This tree should be complete and linear.
        assert!(MortonKey::is_complete_linear_octree(&keys));

        // However, it should not be balanced.
        assert!(!MortonKey::is_complete_linear_and_balanced(&keys));

        // Now balance it.
        let keys = MortonKey::balance(&keys, MortonKey::root());
        // Now the balancing check should be true
        assert!(MortonKey::is_complete_linear_and_balanced(&keys));

        // The balanced tree should have 120 keys. It should have
        // 64 keys on level 3 and then all the other 7 boxes on level 1
        // should have been replaced by their refinement on level 2. Hence,
        // we have 64 + 56 = 120 keys.
        assert_eq!(keys.len(), 120);
    }

    #[test]
    pub fn test_from_physical_point() {
        let bounding_box = PhysicalBox::new([-2.0, -3.0, -1.0, 4.0, 5.0, 6.0]);

        let point = [1.5, -2.5, 5.0];
        let level = 10;

        let key = MortonKey::from_physical_point(point, &bounding_box, level);

        let physical_box = key.physical_box(&bounding_box);

        let coords = physical_box.coordinates();

        assert!(coords[0] <= point[0] && point[0] < coords[3]);
        assert!(coords[1] <= point[1] && point[1] < coords[4]);
        assert!(coords[2] <= point[2] && point[2] < coords[5]);

        // Now compute the box.
    }

    #[test]
    pub fn test_child_index() {
        let key = MortonKey::from_index_and_level([1, 501, 718], 10);

        let children = key.children();

        for (index, child) in children.iter().enumerate() {
            assert_eq!(index, child.child_index());
        }
    }

    #[test]
    pub fn test_finest_outer_descendent() {
        let key = MortonKey::from_index_and_level([0, 0, 0], 1);

        let finest_outer_descendent = key.finest_outer_descendent();

        assert_eq!(
            finest_outer_descendent,
            MortonKey::from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize)
        );

        let key = MortonKey::from_index_and_level([1, 1, 0], 1);
        let finest_outer_descendent = key.finest_outer_descendent();

        assert_eq!(
            finest_outer_descendent,
            MortonKey::from_index_and_level(
                [LEVEL_SIZE as usize - 1, LEVEL_SIZE as usize - 1, 0],
                DEEPEST_LEVEL as usize
            )
        );
    }

    #[test]
    pub fn test_next_nondescendent_key() {
        let key = MortonKey::from_index_and_level([25, 17, 6], 5);

        let children = key.children();

        // Check the next nondescendent key for the first six children

        for (child, next_child) in children.iter().tuple_windows() {
            let next_key = child.next_non_descendent_key().unwrap();
            assert_eq!(next_key.level(), DEEPEST_LEVEL as usize);
            assert!(!child.is_ancestor(next_key));
            assert!(next_child.is_ancestor(next_key));
        }

        // Now check the next nondescendent key from the last child.

        let next_child = children.last().unwrap().next_non_descendent_key();

        // Check that the next nondescendent key from the parent is the same as that of the last child.

        assert_eq!(key.next_non_descendent_key(), next_child);

        // Check that it is not a descendent of the parent and that its level is correct.

        assert_eq!(next_child.unwrap().level(), DEEPEST_LEVEL as usize);
        assert!(!key.is_ancestor(next_child.unwrap()));

        // Finally make sure that an ancestor of deepest last returns None.

        assert!(MortonKey::deepest_last()
            .next_non_descendent_key()
            .is_none());
    }
}
