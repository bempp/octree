//! Routines for working with Morton indices.

use std::collections::HashSet;

use crate::constants::*;
use itertools::izip;
use itertools::Itertools;

// Creating a distinct type for Morton indices
// to distinguish from u64

// numbers.

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    // Return ancestor of key on specified level
    //
    // Return None if level > self.level().
    // Return the key itself if level == self.level().
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

        if *self == other {
            true
        } else if my_level > other_level {
            false
        } else {
            // We shift both keys out to 3 * DEEPEST_LEVEL - my_level
            // This gives identical bit sequences if my_key is an ancestor of other_key
            let my_key = self.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as u64);
            let other_key =
                other.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as u64);

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

        let mut first_key = self.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as u64);
        let mut second_key = other.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as u64);

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

    /// Complete a region ensuring that the given keys are part of the leafs.
    pub fn complete_region(keys: &[MortonKey]) -> Vec<MortonKey> {
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
            result.extend_from_slice(key1.fill_between_keys(key2).as_slice());
        }

        // Push the final key
        result.push(*keys.last().unwrap());
        // We do not sort the keys. They are already sorted.
        result
    }

    pub fn balance(keys: &[MortonKey], root: MortonKey) -> Vec<MortonKey> {
        let mut work_set: HashSet<MortonKey> = keys.iter().cloned().collect();
        let deepest_level = keys.iter().map(|key| key.level()).max().unwrap();
        let root_level = root.level();
        #[cfg(debug_assertions)]
        {
            for key in keys {
                assert!(key.level() >= root_level);
            }
        }

        // Linearize the keys
        let keys = MortonKey::linearize(keys);

        // Now go through and make sure that for each key siblings and neighbours of parents are added
        todo!();
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

        // Correct result for two keys at deepest level
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

        let complete_region = MortonKey::complete_region(keys.as_slice());

        sanity_checks(keys.as_slice(), complete_region.as_slice());

        // For an empty slice the complete region method should just add the root of the tree.
        let keys = Vec::<MortonKey>::new();
        let complete_region = MortonKey::complete_region(keys.as_slice());
        assert_eq!(complete_region.len(), 1);

        sanity_checks(keys.as_slice(), complete_region.as_slice());

        // Choose a region where the first and last key are ancestors of deepest first and deepest last.

        let keys = [MortonKey::deepest_first(), MortonKey::deepest_last()];

        let complete_region = MortonKey::complete_region(keys.as_slice());

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
}
