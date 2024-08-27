//! Definition of a linear octree

use crate::{
    constants::{DEEPEST_LEVEL, NLEVELS},
    geometry::PhysicalBox,
    morton::MortonKey,
};
use bytemuck;
use std::collections::HashMap;
use vtkio;

/// A neighbour
pub struct Neighbour {
    /// Direction
    pub direction: [i64; 3],
    /// Level
    pub level: usize,
    /// Morton key
    pub key: MortonKey,
}

/// An octree
pub struct Octree {
    leaf_keys: Vec<MortonKey>,
    points: Vec<[f64; 3]>,
    point_to_level_keys: [Vec<MortonKey>; NLEVELS],
    bounding_box: PhysicalBox,
    key_counts: HashMap<MortonKey, usize>,
    max_leaf_level: usize,
    max_points_in_leaf: usize,
}

impl Octree {
    /// Create octress from points
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

        let mut point_to_level_keys: [Vec<MortonKey>; NLEVELS] = Default::default();
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

        for keys in &point_to_level_keys {
            for key in keys {
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
            &key_counts,
            &mut leaf_keys,
            max_points_per_box,
            max_level,
        );

        // The leaf keys are now a complete linear tree. But they are not yet balanced.
        // In the final step we balance the leafs.

        let leaf_keys = MortonKey::balance(&leaf_keys, MortonKey::root());

        let mut max_leaf_level = 0;
        let mut max_points_in_leaf = 0;

        for key in &leaf_keys {
            max_leaf_level = max_leaf_level.max(key.level());
            max_points_in_leaf =
                max_points_in_leaf.max(if let Some(&count) = key_counts.get(key) {
                    count
                } else {
                    0
                });
        }

        Self {
            leaf_keys,
            points: points.to_vec(),
            point_to_level_keys,
            bounding_box,
            key_counts,
            max_leaf_level,
            max_points_in_leaf,
        }
    }

    /// Leaf keys
    pub fn leaf_keys(&self) -> &Vec<MortonKey> {
        &self.leaf_keys
    }

    /// Points
    pub fn points(&self) -> &Vec<[f64; 3]> {
        &self.points
    }

    /// Get level keys for each point
    pub fn point_to_level_keys(&self) -> &[Vec<MortonKey>; NLEVELS] {
        &self.point_to_level_keys
    }

    /// Bounding box
    pub fn bounding_box(&self) -> &PhysicalBox {
        &self.bounding_box
    }

    /// Maximum leaf level
    pub fn maximum_leaf_level(&self) -> usize {
        self.max_leaf_level
    }

    /// Maximum number of points in a leaf box
    pub fn max_points_in_leaf_box(&self) -> usize {
        self.max_points_in_leaf
    }

    /// Number of points in the box indexed by a key
    pub fn number_of_points_in_key(&self, key: MortonKey) -> usize {
        if let Some(&count) = self.key_counts.get(&key) {
            count
        } else {
            0
        }
    }

    /// Export the tree to vtk
    pub fn export_to_vtk(&self, file_path: &str) {
        use vtkio::model::{
            Attributes, ByteOrder, CellType, Cells, DataSet, IOBuffer, UnstructuredGridPiece,
            Version, VertexNumbers,
        };

        // Each box has 8 corners with 3 coordinates each, hence 24 floats per key.
        let mut points = Vec::<f64>::new();
        // 8 coords per box, hence 8 * nkeys values in connectivity.
        let mut connectivity = Vec::<u64>::new();
        // Store the vtk offset for each box.
        let mut offsets = Vec::<u64>::new();

        let bounding_box = self.bounding_box();

        // Go through the keys and add coordinates and connectivity.
        // Box coordinates are already in the right order, so connectivity
        // just counts up. We don't mind doubly counted vertices from two boxes.
        let mut point_count = 0;
        let mut key_count = 0;

        for key in self.leaf_keys().iter() {
            // We only want to export non-empty boxes.
            if self.number_of_points_in_key(*key) == 0 {
                continue;
            }
            let coords = key.physical_box(bounding_box).corners();

            key_count += 1;
            offsets.push(8 * key_count);

            for coord in &coords {
                points.push(coord[0]);
                points.push(coord[1]);
                points.push(coord[2]);

                connectivity.push(point_count);
                point_count += 1;
            }
        }

        let vtk_file = vtkio::Vtk {
            version: Version::new((1, 0)),
            title: String::new(),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(points),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity,
                        offsets,
                    },
                    types: vec![CellType::Hexahedron; key_count as usize],
                },
                data: Attributes {
                    point: vec![],
                    cell: vec![],
                },
            }),
        };

        vtk_file.export_ascii(file_path).unwrap();
    }

    // We can now create the vtk object.
}

#[cfg(test)]
mod test {
    use super::Octree;
    use rand::prelude::*;

    fn get_points_on_sphere(npoints: usize) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let mut points = Vec::<f64>::with_capacity(3 * npoints);
        for _ in 0..(npoints) {
            let x: f64 = normal.sample(&mut rng);
            let y: f64 = normal.sample(&mut rng);
            let z: f64 = normal.sample(&mut rng);

            let norm = (x * x + y * y + z * z).sqrt();

            points.push(x / norm);
            points.push(y / norm);
            points.push(z / norm);
        }

        points
    }

    #[test]
    fn test_octree() {
        use std::time::Instant;

        let npoints = 1000000;
        let points = get_points_on_sphere(npoints);
        let max_level = 7;
        let max_points_per_box = 100;

        let start = Instant::now();
        let octree = Octree::from_points(&points, max_level, max_points_per_box);
        let duration = start.elapsed();

        println!("Creation time: {}", duration.as_millis());
        println!("Number of leaf keys: {}", octree.leaf_keys().len());
        println!("Bounding box: {}", octree.bounding_box());
    }

    #[test]
    fn test_export() {
        let fname = "_test_sphere.vtk";
        let npoints = 1000000;
        let points = get_points_on_sphere(npoints);
        let max_level = 7;
        let max_points_per_box = 100;

        let octree = Octree::from_points(&points, max_level, max_points_per_box);

        octree.export_to_vtk(&fname);
        println!("Maximum leaf level: {}", octree.maximum_leaf_level());
        println!(
            "Maximum number of points in leaf box: {}",
            octree.max_points_in_leaf_box()
        );
    }
}
