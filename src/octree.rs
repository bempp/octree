//! Definition of a linear octree

use std::collections::HashMap;

use bytemuck;

use crate::{constants::DEEPEST_LEVEL, geometry::PhysicalBox, morton::MortonKey};

pub struct Octree {
    leaf_keys: Vec<MortonKey>,
    points: Vec<[f64; 3]>,
    point_to_level_keys: [Vec<MortonKey>; 1 + DEEPEST_LEVEL as usize],
    bounding_box: PhysicalBox,
    key_counts: HashMap<MortonKey, usize>,
}

impl Octree {
    pub fn from_points(points: &[f64], max_level: usize, max_points_per_box: usize) -> Self {
        // Make sure that the points array is a multiple of 3.
        assert_eq!(points.len() % 3, 0);

        // Make sure that max level never exceeds DEEPEST_LEVEL
        let max_level = if max_level > DEEPEST_LEVEL as usize {
            DEEPEST_LEVEL as usize
        } else {
            max_level
        };

        // Compute the physical bounding box.
        let bounding_box = PhysicalBox::from_points(points);

        // Bunch the points in arrays of 3.

        let points: &[[f64; 3]] = bytemuck::cast_slice(points);
        let npoints = points.len();

        // We create a vector of keys for each point on each level. We compute the
        // keys on the deepest level and fill the other levels by going from
        // parent to parent.

        let mut point_to_level_keys: [Vec<MortonKey>; 1 + DEEPEST_LEVEL as usize] =
            Default::default();
        point_to_level_keys[DEEPEST_LEVEL as usize] = points
            .iter()
            .map(|&point| {
                MortonKey::from_physical_point(point, &bounding_box, DEEPEST_LEVEL as usize)
            })
            .collect::<Vec<_>>();

        for index in (1..=DEEPEST_LEVEL as usize).rev() {
            let mut new_vec = Vec::<MortonKey>::with_capacity(npoints);
            for &key in &point_to_level_keys[index] {
                new_vec.push(key.parent());
            }
            point_to_level_keys[index - 1] = new_vec;
        }

        // We now have to create level keys. We are starting at the root and recursing
        // down until each box has fewer than max_points_per_box keys.

        // First we compute the counts of each key on each level. For that we create
        // for each level a Hashmap for the keys and then add up.

        let mut key_counts: HashMap<MortonKey, usize> = Default::default();

        for index in 0..=DEEPEST_LEVEL as usize {
            for key in &point_to_level_keys[index] {
                *key_counts.entry(*key).or_default() += 1;
            }
        }

        // We can now easily create an adaptive tree by subdividing. We do this by
        // a recursive function.

        let mut leaf_keys = Vec::<MortonKey>::new();

        fn recurse_keys(
            key: MortonKey,
            key_counts: &HashMap<MortonKey, usize>,
            leaf_keys: &mut Vec<MortonKey>,
            max_points_per_box: usize,
            max_level: usize,
        ) {
            let level = key.level();
            // A key may have not be associated with points. This happens if one of the children on
            // the previous level has no points in its physical box. However, we want to create a
            // complete tree. So we still add this one empty child.
            if let Some(&count) = key_counts.get(&key) {
                if count > max_points_per_box && level < max_level {
                    for child in key.children() {
                        recurse_keys(child, key_counts, leaf_keys, max_points_per_box, max_level);
                    }
                } else {
                    leaf_keys.push(key)
                }
            } else {
                leaf_keys.push(key)
            }
        }

        // Now execute the recursion starting from root

        recurse_keys(
            MortonKey::root(),
            &mut key_counts,
            &mut leaf_keys,
            max_points_per_box,
            max_level,
        );

        // The leaf keys are now a complete linear tree. But they are not yet balanced.
        // In the final step we balance the leafs.

        let leaf_keys = MortonKey::balance(&leaf_keys, MortonKey::root());

        Self {
            leaf_keys,
            points: points.to_vec(),
            point_to_level_keys,
            bounding_box,
            key_counts,
        }
    }

    pub fn leaf_keys(&self) -> &Vec<MortonKey> {
        &self.leaf_keys
    }

    pub fn points(&self) -> &Vec<[f64; 3]> {
        &self.points
    }

    pub fn point_to_level_keys(&self) -> &[Vec<MortonKey>; 1 + DEEPEST_LEVEL as usize] {
        &self.point_to_level_keys
    }

    pub fn bounding_box(&self) -> &PhysicalBox {
        &self.bounding_box
    }

    pub fn number_of_points_in_key(&self, key: MortonKey) -> usize {
        if let Some(&count) = self.key_counts.get(&key) {
            count
        } else {
            0
        }
    }
}
