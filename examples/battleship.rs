//! Compute an Octree for a medieval battleship

use std::time::Instant;

use bempp_octree::octree::Octree;
use vtkio::model::*;

pub fn main() {
    let data: &[u8] = include_bytes!("battleship.vtk"); // Or just include_bytes!

    let mut vtk_file = Vtk::parse_legacy_le(data).unwrap();

    vtk_file.load_all_pieces().unwrap();

    if let DataSet::UnstructuredGrid { pieces, .. } = vtk_file.data {
        let data = pieces[0].load_piece_data(None).unwrap();
        let num_points = data.num_points();
        println!("Number of points {}", num_points);

        let points: Vec<f64> = data.points.into_vec().unwrap();
        println!("Generating octree.");
        let start = Instant::now();
        let octree = Octree::from_points(&points, 16, 50);
        let duration = start.elapsed();
        println!("Generated Octree in {} ms", duration.as_millis());
        println!(
            "Maximum number of points per box: {}",
            octree.max_points_in_leaf_box()
        );
        println!("Maximum level: {}", octree.maximum_leaf_level());

        octree.export_to_vtk("octree.vtk");
    }
}
