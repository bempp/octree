//! Geometry information

use bytemuck;

// A bounding box describes geometry in which an Octree lives.
pub struct PhysicalBox {
    coords: [f64; 6],
}

impl PhysicalBox {
    /// Create a new bounding box.
    ///
    /// The coordinates are given by `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub fn new(coords: [f64; 6]) -> Self {
        Self { coords }
    }

    /// Give a slice of points. Compute an associated bounding box.
    pub fn from_points(points: &[f64]) -> PhysicalBox {
        assert_eq!(points.len() % 3, 0);

        let points: &[[f64; 3]] = bytemuck::cast_slice(points);

        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;

        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;

        let mut zmin = f64::MAX;
        let mut zmax = f64::MIN;

        for point in points {
            let x = point[0];
            let y = point[1];
            let z = point[2];

            xmin = f64::min(xmin, x);
            xmax = f64::max(xmax, x);

            ymin = f64::min(ymin, y);
            ymax = f64::max(ymax, y);

            zmin = f64::min(zmin, z);
            zmax = f64::max(zmax, z);
        }

        // We want the bounding box to be slightly bigger
        // than the actual point set to avoid issues
        // at the edge of the bounding box.

        xmin *= 1.0 + f64::EPSILON;
        xmax *= 1.0 + f64::EPSILON;
        ymin *= 1.0 + f64::EPSILON;
        ymax *= 1.0 + f64::EPSILON;
        zmin *= 1.0 + f64::EPSILON;
        zmax *= 1.0 + f64::EPSILON;

        PhysicalBox {
            coords: [xmin, ymin, zmin, xmax, ymax, zmax],
        }
    }

    /// Return coordinates
    pub fn coordinates(&self) -> [f64; 6] {
        self.coords
    }

    // Map a point from the reference box [0, 1]^3 to the bounding box.
    pub fn reference_to_physical(&self, point: [f64; 3]) -> [f64; 3] {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        [
            xmin + (xmax - xmin) * point[0],
            ymin + (ymax - ymin) * point[1],
            zmin + (zmax - zmin) * point[2],
        ]
    }

    // Map a point from the physical domain to the reference box.
    pub fn physical_to_reference(&self, point: [f64; 3]) -> [f64; 3] {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        [
            (point[0] - xmin) / (xmax - xmin),
            (point[1] - ymin) / (ymax - ymin),
            (point[2] - zmin) / (zmax - zmin),
        ]
    }
}

impl std::fmt::Display for PhysicalBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        write!(
            f,
            "(xmin: {}, ymin: {}, zmin: {}, xmax: {}, ymax: {}, zmax: {})",
            xmin, ymin, zmin, xmax, ymax, zmax
        )
    }
}
