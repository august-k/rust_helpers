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

    #[pyfn(m, "cdist")]
    fn cdist(_py: Python, xa: Vec<Vec<f32>>, xb: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // Form a matrix containing the pairwise distances between the points given
        // This is for calling from Python, Rust functions should use "reference_cdist"
        let mut output_array = Vec::new();

        for i in 0..xa.len() {
            let mut curr_row = Vec::new();
            for j in 0..xb.len() {
                curr_row.push(euclidean_distance(&xa[i], &xb[j]));
            }
            output_array.push(curr_row);
        }

        return output_array;
    }

    #[pyfn(m, "find_center_mass")]
    fn find_center_mass(
        _py: Python,
        units: Vec<SC2Unit>,
        distance: f32,
        default_position: Vec<f32>,
    ) -> (isize, Vec<f32>) {
        // Given a list of Unit objects (so probably a Units object from python-sc2),
        // find the unit that has the most units within <distance> of it

        let mut max_units_found = 0;
        let mut center_position = &default_position;

        // get the positions of all the units
        let mut positions: Vec<Vec<f32>> = Vec::with_capacity(units.len());
        for unit in units.iter() {
            positions.push(vec![unit.position.0, unit.position.1]);
        }

        // get the distance of each unit to each unit
        let distances = reference_cdist(&positions, &positions);
        for i in 0..distances.len() {
            let mut units_found = 0;
            for j in 0..distances[i].len() {
                if distances[i][j] < distance {
                    units_found += 1;
                }
            }
            if units_found > max_units_found {
                max_units_found = units_found;
                center_position = &positions[i];
            }
        }
        return (max_units_found, center_position.to_vec());
    }

    fn reference_cdist(xa: &Vec<Vec<f32>>, xb: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // Form a matrix containing the pairwise distances between the points given
        // This is for calling internally, Pythont functions should use "cdist"
        // For Rust purposes, it makes more sense to have the input vectors be references
        // but I'm not sure how that works with pyo3, so there's a Python version
        let mut output_array = Vec::new();

        for i in 0..xa.len() {
            let mut curr_row = Vec::new();
            for j in 0..xb.len() {
                curr_row.push(euclidean_distance(&xa[i], &xb[j]));
            }
            output_array.push(curr_row);
        }
        return output_array;
    }

    fn get_squared_distance(p1: (f32, f32), p2: (f32, f32)) -> f32 {
        return f32::powf(p1.0 - p2.0, 2.0) + f32::powf(p1.1 - p2.1, 2.0);
    }

    fn euclidean_distance(p1: &Vec<f32>, p2: &Vec<f32>) -> f32 {
        return get_squared_distance((p1[0], p1[1]), (p2[0], p2[1])).sqrt();
    }

    Ok(())
}
