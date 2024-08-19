//! Definition of a linear octree

use std::collections::HashMap;
use vtkio;

use bytemuck;

use crate::{
    constants::{DEEPEST_LEVEL, NLEVELS},
    geometry::PhysicalBox,
    morton::MortonKey,
};

pub struct Octree {
    leaf_keys: Vec<MortonKey>,
    points: Vec<[f64; 3]>,
    point_to_level_keys: [Vec<MortonKey>; NLEVELS],
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

    pub fn point_to_level_keys(&self) -> &[Vec<MortonKey>; NLEVELS] {
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

    /// Export the tree to vtk
    pub fn export_to_vtk(&self, file_path: &str) {
        use vtkio::model::*;

        let nkeys = self.leaf_keys().len();
        // Each box has 8 corners with 3 coordinates each, hence 24 floats per key.
        let mut points = Vec::<f64>::with_capacity(24 * nkeys);
        // 8 coords per box, hence 8 * nkeys values in connectivity.
        let mut connectivity = Vec::<u64>::with_capacity(8 * nkeys);
        // Store the vtk offset for each box.
        let mut offsets = Vec::<u64>::with_capacity(nkeys);

        let bounding_box = self.bounding_box();

        // Go through the keys and add coordinates and connectivity.
        // Box coordinates are already in the right order, so connectivity
        // just counts up. We don't mind doubly counted vertices from two boxes.
        let mut count = 0;

        for (key_index, key) in self.leaf_keys().iter().enumerate() {
            let coords = key.physical_box(&bounding_box).corners();

            offsets.push(8 * (1 + key_index) as u64);

            for coord in &coords {
                points.push(coord[0]);
                points.push(coord[1]);
                points.push(coord[2]);

                connectivity.push(count);
                count += 1;
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
                    types: vec![CellType::Hexahedron; nkeys],
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

    use rand::prelude::*;

    use super::Octree;

    fn get_random_points(npoints: usize) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let mut points = Vec::<f64>::with_capacity(3 * npoints);
        for _ in 0..(3 * npoints) {
            points.push(rng.gen());
        }

        points
    }

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
    pub fn test_octree() {
        let npoints = 1000;
        let points = get_random_points(npoints);
        let max_level = 3;
        let max_points_per_box = 20;

        let octree = Octree::from_points(&points, max_level, max_points_per_box);

        println!("Number of leaf keys: {}", octree.leaf_keys().len());
        println!("Bounding box: {}", octree.bounding_box());
    }

    #[test]
    fn test_export() {
        let fname = "sphere.vtk";
        let npoints = 1000000;
        let points = get_points_on_sphere(npoints);
        let max_level = 6;
        let max_points_per_box = 100;

        let octree = Octree::from_points(&points, max_level, max_points_per_box);

        octree.export_to_vtk(&fname);
    }
}
