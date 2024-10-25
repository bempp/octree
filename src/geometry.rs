//! Geometry information

use mpi::traits::Equivalence;

use crate::constants::DEEPEST_LEVEL;

/// Definition of a point.
#[derive(Clone, Copy, Equivalence)]
pub struct Point {
    coords: [f64; 3],
    global_id: usize,
}

impl Point {
    /// Create a new point from coordinates and global id.
    pub fn new(coords: [f64; 3], global_id: usize) -> Self {
        Self { coords, global_id }
    }

    /// Return the coordintes of a point.
    pub fn coords(&self) -> [f64; 3] {
        self.coords
    }

    /// Return a mutable pointer to the coordinates.
    pub fn coords_mut(&mut self) -> &mut [f64; 3] {
        &mut self.coords
    }

    /// Return the global id of the point.
    pub fn global_id(&self) -> usize {
        self.global_id
    }
}

/// A bounding box describes geometry in which an Octree lives.
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
    pub fn from_points(points: &[Point]) -> PhysicalBox {
        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;

        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;

        let mut zmin = f64::MAX;
        let mut zmax = f64::MIN;

        for point in points {
            let x = point.coords()[0];
            let y = point.coords()[1];
            let z = point.coords()[2];

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

        let xdiam = xmax - xmin;
        let ydiam = ymax - ymin;
        let zdiam = zmax - zmin;

        let xmean = xmin + 0.5 * xdiam;
        let ymean = ymin + 0.5 * ydiam;
        let zmean = zmin + 0.5 * zdiam;

        // We increase diameters by box size on deepest level
        // and use the maximum diameter to compute a
        // cubic bounding box.

        let deepest_box_diam = 1.0 / (1 << DEEPEST_LEVEL) as f64;

        let max_diam = [xdiam, ydiam, zdiam].into_iter().reduce(f64::max).unwrap();

        let max_diam = max_diam * (1.0 + deepest_box_diam);

        PhysicalBox {
            coords: [
                xmean - 0.5 * max_diam,
                ymean - 0.5 * max_diam,
                zmean - 0.5 * max_diam,
                xmean + 0.5 * max_diam,
                ymean + 0.5 * max_diam,
                zmean + 0.5 * max_diam,
            ],
        }
    }

    /// Return coordinates
    pub fn coordinates(&self) -> [f64; 6] {
        self.coords
    }

    /// Map a point from the reference box [0, 1]^3 to the bounding box.
    pub fn reference_to_physical(&self, point: [f64; 3]) -> [f64; 3] {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        [
            xmin + (xmax - xmin) * point[0],
            ymin + (ymax - ymin) * point[1],
            zmin + (zmax - zmin) * point[2],
        ]
    }

    /// Map a point from the physical domain to the reference box.
    pub fn physical_to_reference(&self, point: [f64; 3]) -> [f64; 3] {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        [
            (point[0] - xmin) / (xmax - xmin),
            (point[1] - ymin) / (ymax - ymin),
            (point[2] - zmin) / (zmax - zmin),
        ]
    }

    /// Return an ordered list of corners of the box.
    ///
    /// The ordering of the corners on the unit cube is
    /// [0, 0, 0]
    /// [1, 0, 0]
    /// [1, 1, 0]
    /// [0, 1, 0]
    /// [0, 0, 1]
    /// [1, 0, 1]
    /// [1, 1, 1]
    /// [0, 1, 1]
    pub fn corners(&self) -> [[f64; 3]; 8] {
        let reference_points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        [
            self.reference_to_physical(reference_points[0]),
            self.reference_to_physical(reference_points[1]),
            self.reference_to_physical(reference_points[2]),
            self.reference_to_physical(reference_points[3]),
            self.reference_to_physical(reference_points[4]),
            self.reference_to_physical(reference_points[5]),
            self.reference_to_physical(reference_points[6]),
            self.reference_to_physical(reference_points[7]),
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
