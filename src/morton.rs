//! Routines for working with Morton indices.

use crate::constants::*;

pub struct Morton {
    index: [u64; 3],
    key: u64,
}

pub fn encode_morton(index: [u64; 3], level: u64) -> u64 {
    assert!(level <= DEEPEST_LEVEL);

    // If we are not on the deepest level we need to shift the box.
    // The box with x-index one on DEEPEST_LEVEL-1 has index two on
    // DEEPEST_LEVEL.

    let level_diff = DEEPEST_LEVEL - level;

    let x = index[0] << level_diff;
    let y = index[1] << level_diff;
    let z = index[2] << level_diff;

    let key: u64 = X_LOOKUP_ENCODE[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

    let key = (key << 24)
        | X_LOOKUP_ENCODE[(x & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[(y & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[(z & BYTE_MASK) as usize];

    let key = key << LEVEL_DISPLACEMENT;
    key | level
}

pub fn decode_key(morton: u64) -> [u64; 3] {
    fn decode_key_helper(key: u64, lookup_table: &[u64; 512]) -> u64 {
        const N_LOOPS: u64 = 6; // 48 bits for the keys. Process in pairs of 9. So 6 passes enough.
        let mut coord: u64 = 0;

        for index in 0..N_LOOPS {
            coord |= lookup_table[((key >> (index * 9)) & NINE_BIT_MASK) as usize] << (3 * index);
        }

        coord
    }

    let key = morton >> LEVEL_DISPLACEMENT;

    let x = decode_key_helper(key, &X_LOOKUP_DECODE);
    let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
    let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

    [x, y, z]
}
