//! Routines for working with Morton indices.

use crate::constants::*;

// Creating a distinct type for Morton indices
// to distinguish from u64 numbers.

pub struct MortonKey {
    value: u64,
}

impl MortonKey {
    pub fn new(value: u64) -> Self {
        Self { value }
    }

    pub fn from_index_and_level(index: [usize; 3], level: usize) -> MortonKey {
        let level = level as u64;
        assert!(level <= DEEPEST_LEVEL);

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

    pub fn level(&self) -> usize {
        (self.value & LEVEL_MASK) as usize
    }

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

    pub fn parent(&self) -> Self {
        let level = self.level();
        let parent_level = level - 1;
        let key = self.value >> LEVEL_DISPLACEMENT;

        let bit_displacement = 3 * (DEEPEST_LEVEL - parent_level as u64);
        // Sets the last bits to zero and shifts back
        let key = (key >> bit_displacement) << (bit_displacement + LEVEL_DISPLACEMENT);

        Self {
            value: key | parent_level as u64,
        }
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
    fn test_debug_print() {
        let key = MortonKey::from_index_and_level([1, 3, 5], 3);
        let parent = key.parent();

        println!("{:#?}", parent);
    }
}
