//! Definition of a linear octree
//!
//! A linear octree is a sorted collection of Morton keys.

use crate::morton::MortonKey;
use itertools::Itertools;

pub struct Octree {
    keys: Vec<MortonKey>,
}

impl Octree {
    pub fn new<Iter: Iterator<Item = MortonKey>>(keys: Iter) -> Self {
        Self {
            keys: keys.collect(),
        }
    }

    pub fn sort(&mut self) {
        self.keys.sort_unstable();
    }

    pub fn iter(&self) -> std::slice::Iter<'_, MortonKey> {
        self.keys.iter()
    }

    pub fn remove_overlaps<Iter: Iterator<Item = MortonKey>>(&mut self) {
        let mut new_keys = Vec::<MortonKey>::new();
        for (m1, m2) in self.keys.iter().tuple_windows() {
            if m1 == m2 || m1.is_ancestor(*m2) {
                continue;
            }
            new_keys.push(*m1)
        }
        new_keys.push(*self.keys.last().unwrap());
        self.keys = new_keys;
    }
}
