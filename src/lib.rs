mod sc2_unit;

use crate::sc2_unit::SC2Unit;
use pyo3::prelude::*;

#[pymodule]
fn rust_helpers(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "get_neighbors")]
    fn get_neighbors(_py: Python, raw_point: (f32, f32)) -> Vec<(i32, i32)> {
        // Return the 8 neighboring coordinates of point
        let mut neighbors: Vec<(i32, i32)> = Vec::with_capacity(8);
        /* Uses 0 to 3 and -1 because I'm not sure if Rust allows
        signed ints as iteration values */
        let point: [i32; 2] = [raw_point.0 as i32, raw_point.1 as i32];
        for i in 0..3 {
            for j in 0..3 {
                // skips what would be +0, +0 (the initial point)
                if i == 1 && j == 1 {
                    continue;
                }
                neighbors.push((point[0] + i - 1, point[1] + j - 1));
            }
        }
        return neighbors;
    }

    #[pyfn(m, "closest_unit_index_to")]
    fn closest_unit_index_to(
        _py: Python,
        units: Vec<SC2Unit>,
        target_position: (f32, f32),
    ) -> usize {
        let mut closest_index: usize = 0;
        let mut closest_distance: f32 = 9999.9;
        for i in 0..units.len() {
            let dist: f32 = get_squared_distance(units[i].position, target_position);
            if dist < closest_distance {
                closest_index = i;
                closest_distance = dist;
            }
        }
        return closest_index;
    }

    #[pyfn(m, "closest_position_index_to")]
    fn closest_position_index_to(
        _py: Python,
        positions: Vec<(f32, f32)>,
        target_position: (f32, f32),
    ) -> usize {
        let mut closest_index: usize = 0;
        let mut closest_distance: f32 = 9999.9;
        for i in 0..positions.len() {
            let dist: f32 = get_squared_distance(positions[i], target_position);
            if dist < closest_distance {
                closest_index = i;
                closest_distance = dist;
            }
        }
        return closest_index;
    }

    fn cdist(xa: Vec<(f32, f32)>, xb: Vec<(f32, f32)>) -> Vec<Vec<f32>> {
        // Form a matrix containing the pairwise distances between the points in the input matrices
        let mut output_array = Vec::new();
        for i in 0..xa.len() {
            let mut curr_row = Vec::new();
            for j in 0..xb.len() {
                curr_row.push(euclidean_distance(xa[i], xb[j]));
            }
            output_array.push(curr_row);
        }
        return output_array;
    }

    fn get_squared_distance(p1: (f32, f32), p2: (f32, f32)) -> f32 {
        return f32::powf(p1.0 - p2.0, 2.0) + f32::powf(p1.1 - p2.1, 2.0);
    }

    fn euclidean_distance(p1: (f32, f32), p2: (f32, f32)) -> f32 {
        return get_squared_distance(p1, p2).sqrt();
    }

    Ok(())
}
