//! Routines for working with Morton indices.

use crate::constants::*;

// Creating a distinct type for Morton indices
// to distinguish from i64
// numbers.

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MortonKey {
    value: i64,
}

impl MortonKey {
    pub fn new(value: i64) -> Self {
        Self { value }
    }

    pub fn from_index_and_level(index: [usize; 3], level: usize) -> MortonKey {
        let level = level as i64;
        assert!(level <= DEEPEST_LEVEL);

        // If we are not on the deepest level we need to shift the box.
        // The box with x-index one on DEEPEST_LEVEL-1 has index two on
        // DEEPEST_LEVEL.

        let level_diff = DEEPEST_LEVEL - level;

        let x = (index[0] as i64) << level_diff;
        let y = (index[1] as i64) << level_diff;
        let z = (index[2] as i64) << level_diff;

        let key: i64 = X_LOOKUP_ENCODE[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
            | Y_LOOKUP_ENCODE[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
            | Z_LOOKUP_ENCODE[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

        let key = (key << 24)
            | X_LOOKUP_ENCODE[(x & BYTE_MASK) as usize]
            | Y_LOOKUP_ENCODE[(y & BYTE_MASK) as usize]
            | Z_LOOKUP_ENCODE[(z & BYTE_MASK) as usize];

        let key = key << LEVEL_DISPLACEMENT;
        Self { value: key | level }
    }

    pub fn level(&self) -> usize {
        (self.value & LEVEL_MASK) as usize
    }

    pub fn decode(&self) -> (usize, [usize; 3]) {
        fn decode_key_helper(key: i64, lookup_table: &[i64; 512]) -> i64 {
            const N_LOOPS: i64 = 6; // 48 bits for the keys. Process in pairs of 9. So 6 passes enough.
            let mut coord: i64 = 0;

            for index in 0..N_LOOPS {
                coord |=
                    lookup_table[((key >> (index * 9)) & NINE_BIT_MASK) as usize] << (3 * index);
            }

            coord
        }

        let level = self.level();
        let level_diff = DEEPEST_LEVEL - level as i64;

        let key = self.value >> LEVEL_DISPLACEMENT;

        let x = decode_key_helper(key, &X_LOOKUP_DECODE);
        let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
        let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

        let x = x >> level_diff;
        let y = y >> level_diff;
        let z = z >> level_diff;

        (level, [x as usize, y as usize, z as usize])
    }

    pub fn parent(&self) -> Self {
        let level = self.level();
        let parent_level = level - 1;
        let key = self.value >> LEVEL_DISPLACEMENT;

        let bit_displacement = 3 * (DEEPEST_LEVEL - parent_level as i64);
        // Sets the last bits to zero and shifts back
        let key = (key >> bit_displacement) << (bit_displacement + LEVEL_DISPLACEMENT);

        Self {
            value: key | parent_level as i64,
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

        let bit_displacement = 3 * (DEEPEST_LEVEL - level as i64);
        // Sets the last bits to zero and shifts back
        let key = (key >> bit_displacement) << (bit_displacement + LEVEL_DISPLACEMENT);

        Some(Self {
            value: key | level as i64,
        })
    }

    pub fn is_ancestor(&self, other: MortonKey) -> bool {
        let my_level = self.level();
        let other_level = other.level();

        if my_level > other_level {
            false
        } else {
            // We shift both keys out to 3 * DEEPEST_LEVEL - my_level
            // This gives identical bit sequences if my_key is an ancestor of other_key
            let my_key = self.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as i64);
            let other_key =
                other.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as i64);

            my_key == other_key
        }
    }

    // Return the finest common ancestor of two keys.
    // If the keys are identical return the key itself.
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

        let mut first_key = self.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as i64);
        let mut second_key = other.value >> LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as i64);

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

        first_key <<= 3 * (DEEPEST_LEVEL - new_level as i64) + LEVEL_DISPLACEMENT;

        MortonKey {
            value: first_key | new_level as i64,
        }
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
            let mut expected: i64 = (index & 1) as i64;
            expected |= (((index >> 3) & 1) << 1) as i64;
            expected |= (((index >> 6) & 1) << 2) as i64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_y_decode_table() {
        for (index, &actual) in Y_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: i64 = ((index >> 1) & 1) as i64;
            expected |= (((index >> 4) & 1) << 1) as i64;
            expected |= (((index >> 7) & 1) << 2) as i64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_x_decode_table() {
        for (index, &actual) in X_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: i64 = ((index >> 2) & 1) as i64;
            expected |= (((index >> 5) & 1) << 1) as i64;
            expected |= (((index >> 8) & 1) << 2) as i64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_z_encode_table() {
        for (mut index, actual) in Z_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: i64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift)) as i64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_y_encode_table() {
        for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: i64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 1)) as i64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_x_encode_table() {
        for (mut index, actual) in X_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: i64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 2)) as i64;
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
        println!("Ancestor {:#?}", ancestor);
        assert!(ancestor.is_ancestor(key));
    }

    #[test]
    fn test_ancestor_at_level() {
        let index = [15, 39, 45];
        let key = MortonKey::from_index_and_level(index, 9);
        assert!(key.is_ancestor(key));
        let ancestor = key.parent().parent();
        println!("Ancestor {:#?}", ancestor);
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
    fn test_print() {
        let key = MortonKey::from_index_and_level([1, 3, 5], 3);
        let parent = key.parent();

        println!("{}", parent);
    }
}
