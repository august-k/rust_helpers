// Bounding Circle code adapted from somewhere

mod sc2_unit;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::sc2_unit::SC2Unit;
use pyo3::prelude::*;

const MULTIPLICATIVE_EPSILON: f32 = 1.00000000000001;
const NONEXISTENTCIRCLE: (f32, f32, f32) = (f32::NAN, f32::NAN, f32::NAN);
const TERRAIN_HEIGHT_TOLERANCE: u8 = 11;

extern crate ndarray;
use ndarray::Array2;
use numpy::{Ix2, PyArray, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

#[pymodule]
fn rust_helpers(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "get_neighbors")]
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
        neighbors
    }

    #[pyfn(m)]
    #[pyo3(name = "closest_unit_index_to")]
    fn closest_unit_index_to(
        _py: Python,
        units: Vec<SC2Unit>,
        target_position: (f32, f32),
    ) -> usize {
        let mut closest_index: usize = 0;
        let mut closest_distance: f32 = 9999.9;
        for (i, unit) in units.iter().enumerate() {
            let dist: f32 = get_squared_distance(unit.position, target_position);
            if dist < closest_distance {
                closest_index = i;
                closest_distance = dist;
            }
        }
        closest_index
    }

    #[pyfn(m)]
    #[pyo3(name = "closest_position_index_to")]
    fn closest_position_index_to(
        _py: Python,
        positions: Vec<(f32, f32)>,
        target_position: (f32, f32),
    ) -> usize {
        let mut closest_index: usize = 0;
        let mut closest_distance: f32 = 9999.9;
        for (i, position) in positions.iter().enumerate() {
            let dist: f32 = get_squared_distance(*position, target_position);
            if dist < closest_distance {
                closest_index = i;
                closest_distance = dist;
            }
        }
        closest_index
    }

    #[pyfn(m)]
    #[pyo3(name = "cdist")]
    fn cdist(_py: Python, xa: Vec<Vec<f32>>, xb: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // Form a matrix containing the pairwise distances between the points given
        // This is for calling from Python, Rust functions should use "reference_cdist"
        let mut output_array = Vec::new();

        for a_val in &xa {
            let mut curr_row = Vec::new();
            for b_val in &xb {
                curr_row.push(euclidean_distance(a_val, b_val));
            }
            output_array.push(curr_row);
        }

        output_array
    }

    #[pyfn(m)]
    #[pyo3(name = "find_center_mass")]
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
        let mut units_found: isize;
        for i in 0..distances.len() {
            units_found = 0;
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
        (max_units_found, center_position.to_vec())
    }

    #[pyfn(m)]
    #[pyo3(name = "surround_complete")]
    fn surround_complete(
        _py: Python,
        units: Vec<SC2Unit>,
        our_center: Vec<f32>,
        enemy_center: Vec<f32>,
        offset: f32,
        _ratio: f32,
    ) -> bool {
        /*
        Determine whether we have enough of our surrounding units
        on either side of the target enemy. This is done by drawing a line through
        the potentially offset enemy center perpendicular to the line segment connecting our center and their
        center and then seeing the spread of our units on either side of the line.

        The slope of a line tangent to a circle is -x/y (as calculated by the derivative of
        x**2 + y**2 = R**2). Thefore the tangent line's slope at (x1, y1) is -x1/y1.

        If r is the distance from the origin to (x1, y1) and theta is the angle between the
        vectors (1, 0) and (x1, y1), x1 and y1 can be expressed as r * cos(theta) and r * sin(theta),
        respectively. Therefore the slope of the line is -cos(theta) / sin(theta).

        If the offset is 0, the line will go through the enemy center. Otherwise the line will undergo
        a translation of `offset` in the direction away from our units. This point is then used for
        the inequality based on the upcoming equation.

        We can write the equation of the line in point slope form as:
        y - enemy_y = -cos(theta) / sin(theta) * (x - enemy_x)

        To avoid potentially dividing by zero, this can be written as:
        sin(theta) * (y - enemy_y) = -cos(theta) * (x - enemy_x)

        While this is worse for drawing a line, we don't actually care about the line- we just want
        to separate units based on it. If the two sides of the equation are equal, the point is on the line,
        otherwise it's on one of the sides of the line. Since which side doesn't matter (the only relevant
        information is how many units are on either side), we can divide units into two categories by
        plugging in their position into the final equation and comparing the two sides.

        This gives us the split of units and we determine surround status based on the ratio of units
        on either side.

        ratio is currently unused due to the final else block, but I'm leaving it in for now
        */

        // start getting the angle by applying a translation that moves the enemy to the origin
        let our_adjusted_position: Vec<f32> = vec![
            our_center[0] - enemy_center[0],
            our_center[1] - enemy_center[1],
        ];

        // use atan2 to get the angle
        let angle_to_origin: f32 = our_adjusted_position[1].atan2(our_adjusted_position[0]);

        // We need sine and cosine for the inequality
        let sincos: (f32, f32) = angle_to_origin.sin_cos();

        // Check which side of the line our units are on. Positive and negative don't actually matter,
        // we just need to be consistent. This may be harder to visualize, but it led to fewer
        // headaches
        let mut side_one: f32 = 0.0;
        let mut side_two: f32 = 0.0;

        // Adjust the angle so that it's pointing away from our units and apply the offset
        // The radius of a Zergling is added to the offset so that the entire Zergling is behind the line
        let adjusted_angle: f32 = angle_to_origin + std::f32::consts::PI;
        let enemy_x: f32 = enemy_center[0] + (offset + 0.375) * adjusted_angle.cos();
        let enemy_y: f32 = enemy_center[1] + (offset + 0.375) * adjusted_angle.sin();

        for unit in units.iter() {
            if get_squared_distance(unit.position, (enemy_x, enemy_y)) >= 300.0 {
                continue;
            }
            let y = sincos.0 * (unit.position.1 - enemy_y);
            let x = sincos.1 * -(unit.position.0 - enemy_x);
            if y >= x {
                side_one += 1.0;
            } else {
                side_two += 1.0;
            }
        }
        /*
        Not sure what happened with this block, but it's currently an unnecessary check as
        if it's false, the function returns true anyway due to the final else. Since only
        the sides are checked, it can return the final expression and be equivalent to the
        commented out code.

        if side_one == 0.0 || side_two == 0.0 {
            false
        } else if side_one / side_two <= (1.0 / ratio) && side_one / side_two <= ratio {
            true
        } else {
            true
        }
        */
        !(side_one == 0.0 || side_two == 0.0)
    }

    fn reference_cdist(xa: &[Vec<f32>], xb: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Form a matrix containing the pairwise distances between the points given
        // This is for calling internally, Python functions should use "cdist"
        // For Rust purposes, it makes more sense to have the input vectors be references
        // but I'm not sure how that works with pyo3, so there's a Python version
        let mut output_array = Vec::new();

        for a_val in xa {
            let mut curr_row = Vec::new();
            for b_val in xb {
                curr_row.push(euclidean_distance(a_val, b_val));
            }
            output_array.push(curr_row);
        }
        output_array
    }

    fn get_squared_distance(p1: (f32, f32), p2: (f32, f32)) -> f32 {
        f32::powf(p1.0 - p2.0, 2.0) + f32::powf(p1.1 - p2.1, 2.0)
    }

    fn euclidean_distance(p1: &[f32], p2: &[f32]) -> f32 {
        get_squared_distance((p1[0], p1[1]), (p2[0], p2[1])).sqrt()
    }

    fn hypot(x: f32, y: f32) -> f32 {
        (f32::powf(x, 2.0) + f32::powf(y, 2.0)).sqrt()
    }

    #[pyfn(m)]
    #[pyo3(name = "make_bounding_circle")]
    fn make_bounding_circle(_py: Python, points: Vec<Vec<f32>>) -> (f32, f32, f32) {
        // Return (x_coordinate, y_coordinate, radius) of the circle bounding the points given
        let mut ref_points: Vec<&Vec<f32>> = Vec::new();
        // Needed &Vec and this was the only way I could think to do it
        for p in points.iter() {
            ref_points.push(p);
        }
        make_circle(ref_points)
    }

    fn make_circle(points: Vec<&Vec<f32>>) -> (f32, f32, f32) {
        let mut shuffled = points;
        let mut rng = thread_rng();
        shuffled.shuffle(&mut rng);

        let mut c = (f32::NAN, f32::NAN, f32::NAN);
        for (i, p) in shuffled.iter().cloned().enumerate() {
            if c == NONEXISTENTCIRCLE || !is_in_circle(&c, p) {
                c = _make_circle_one_point(shuffled[i + 1..].to_vec(), p.to_vec());
            }
        }

        c
    }

    fn _make_circle_one_point(points: Vec<&Vec<f32>>, p: Vec<f32>) -> (f32, f32, f32) {
        let mut c = (p[0], p[1], 0.0f32);
        for (i, q) in points.iter().enumerate() {
            if !is_in_circle(&c, q) {
                if c.2 == 0.0 {
                    c = make_diameter(&p, q);
                } else {
                    c = make_circle_two_points(points[i + 1..].to_vec(), p.to_vec(), q.to_vec());
                }
            }
        }
        c
    }

    fn make_circle_two_points(points: Vec<&Vec<f32>>, p: Vec<f32>, q: Vec<f32>) -> (f32, f32, f32) {
        let circle = make_diameter(&p, &q);
        let mut left: (f32, f32, f32) = (f32::NAN, f32::NAN, f32::NAN);
        let mut right: (f32, f32, f32) = (f32::NAN, f32::NAN, f32::NAN);
        let px: f32 = p[0];
        let py: f32 = p[1];
        let qx: f32 = q[0];
        let qy: f32 = q[1];

        // For each point not in the two-point circle
        for r in points.iter().cloned() {
            if is_in_circle(&circle, r) {
                continue;
            }
            // Form a circumcircle and classify it on left or right side
            let cross = cross_product(&px, &py, &qx, &qy, &r[0], &r[1]);
            let c = make_circumcircle(&p, &q, r);
            if c == NONEXISTENTCIRCLE {
                continue;
            } else if cross > 0.0 && left == NONEXISTENTCIRCLE
                || cross_product(&px, &py, &qx, &qy, &c.0, &c.1)
                    > cross_product(&px, &py, &qx, &qy, &left.0, &left.1)
            {
                left = c;
            } else if cross > 0.0 && right == NONEXISTENTCIRCLE
                || cross_product(&px, &py, &qx, &qy, &c.0, &c.1)
                    > cross_product(&px, &py, &qx, &qy, &right.0, &right.1)
            {
                right = c;
            }
        }
        // Not sure how the left.2 <= right.2 check works, but this is functional so I'm not changing it
        if left == NONEXISTENTCIRCLE && right == NONEXISTENTCIRCLE {
            circle
        } else if right == NONEXISTENTCIRCLE || left.2 <= right.2 {
            left
        } else {
            right
        }
    }

    fn make_diameter(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        // Return the average of a and b and the radius of the circle centered on the average point that includes
        // both a and b
        let cx = (a[0] + b[0]) / 2.0;
        let cy = (a[1] + b[1]) / 2.0;
        let r0 = hypot(cx - a[0], cy - a[1]);
        let r1 = hypot(cx - b[0], cy - b[1]);
        (cx, cy, get_max(&[r0, r1]))
    }

    fn make_circumcircle(point_a: &[f32], point_b: &[f32], point_c: &[f32]) -> (f32, f32, f32) {
        // Mathematical algorithm from Wikipedia: Circumscribed circle
        let x_coords: Vec<f32> = vec![point_a[0], point_b[0], point_c[0]];
        let y_coords: Vec<f32> = vec![point_a[1], point_b[1], point_c[1]];
        let ox: f32 = (get_min(&x_coords) + get_max(&x_coords)) / 2.0;
        let oy: f32 = (get_min(&y_coords) + get_max(&y_coords)) / 2.0;

        let ax: f32 = point_a[0] - ox;
        let ay: f32 = point_a[1] - oy;
        let bx: f32 = point_b[0] - ox;
        let by: f32 = point_b[1] - oy;
        let cx: f32 = point_c[0] - ox;
        let cy: f32 = point_c[1] - oy;

        let det: f32 = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0;
        if det == 0.0 {
            return (f32::NAN, f32::NAN, f32::NAN);
        }

        let x_val: f32 = ox
            + ((ax * ax + ay * ay) * (by - cy)
                + (bx * bx + by * by) * (cy - ay)
                + (cx * cx + cy * cy) * (ay - by))
                / det;
        let y_val: f32 = oy
            + ((ax * ax + ay * ay) * (cx - bx)
                + (bx * bx + by * by) * (ax - cx)
                + (cx * cx + cy * cy) * (bx - ax))
                / det;

        let ra: f32 = hypot(x_val - point_a[0], y_val - point_a[0]);
        let rb: f32 = hypot(x_val - point_b[0], y_val - point_b[1]);
        let rc: f32 = hypot(x_val - point_c[0], y_val - point_c[1]);

        (x_val, y_val, get_max(&[ra, rb, rc]))
    }

    fn is_in_circle(c: &(f32, f32, f32), p: &[f32]) -> bool {
        c != &NONEXISTENTCIRCLE && hypot(p[0] - c.0, p[1] - c.1) <= c.2 * MULTIPLICATIVE_EPSILON
    }

    fn cross_product(x0: &f32, y0: &f32, x1: &f32, y1: &f32, x2: &f32, y2: &f32) -> f32 {
        // Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
        (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    }

    fn get_max(vector: &[f32]) -> f32 {
        // Rust doesn't have an easy way of getting the maximum value of a Vec<f32> so I'm using this.
        let mut max = vector[0];
        for val in vector.iter() {
            if val > &max {
                max = *val;
            }
        }
        max
    }

    fn get_min(vector: &[f32]) -> f32 {
        // Rust doesn't have an easy way of getting the minimum value of a Vec<f32> so I'm using this.
        let mut min = vector[0];
        for val in vector.iter() {
            if val < &min {
                min = *val;
            }
        }
        min
    }

    fn get_positions_center(positions: Vec<(f32, f32)>) -> (f32, f32) {
        // Given a list of positions, get the mean position.
        let mut tot_x: f32 = 0.0;
        let mut tot_y: f32 = 0.0;
        for coords in positions.iter() {
            tot_x += coords.0;
            tot_y += coords.1;
        }
        let num_positions: f32 = positions.len() as f32;
        (tot_x / num_positions, tot_y / num_positions)
    }

    fn get_units_center(units: &[SC2Unit]) -> (f32, f32) {
        // Given a list of units, get the mean position.
        let mut position_vector = Vec::new();
        for unit in units.iter() {
            position_vector.push(unit.position);
        }
        get_positions_center(position_vector)
    }

    #[pyfn(m)]
    #[pyo3(name = "adjust_combat_formation")]
    fn adjust_combat_formation(
        _py: Python,
        our_units: Vec<SC2Unit>,
        enemies: Vec<SC2Unit>,
        fodder_tags: Vec<u64>,
        core_unit_multiplier: f32,
        fodder_unit_multiplier: f32,
        retreat_angle: f32,
    ) -> HashMap<u64, (f32, f32)> {
        // Takes our units, enemy units, and our fodder units
        // Returns a (Python) list of tags and positions, where each position is where the unit with that tag should move

        // This will be the dictionary of tag to new position
        let mut tag_to_position: HashMap<u64, (f32, f32)> = HashMap::new();

        // If there are no fodder tags or enemies, none of the units will need their positions adjusted
        if fodder_tags.is_empty() || enemies.is_empty() {
            return tag_to_position;
        }

        // Separate the core units from the fodder units
        let mut core_units: Vec<SC2Unit> = Vec::new();
        let mut fodder_units: Vec<SC2Unit> = Vec::new();

        for unit in our_units.iter() {
            if fodder_tags.contains(&unit.tag) {
                fodder_units.push(*unit);
            } else {
                core_units.push(*unit);
            }
        }

        // Find the center of the units so we can get properly align retreat paths
        let our_center: (f32, f32) = get_units_center(&our_units);
        let enemy_center: (f32, f32) = get_units_center(&enemies);
        // We'll need the enemy center as a vector of vectors later for cdist
        let enemy_center_vector: Vec<Vec<f32>> = vec![vec![enemy_center.0, enemy_center.1]];

        // start getting the angle by applying a translation that moves the enemy to the origin
        let our_adjusted_position: Vec<f32> =
            vec![our_center.0 - enemy_center.0, our_center.1 - enemy_center.1];

        // use atan2 to get the angle
        let angle_to_origin: f32 = our_adjusted_position[1].atan2(our_adjusted_position[0]);

        // We need sine and cosine so that we can give the correct retreat position
        let sincos: (f32, f32) = angle_to_origin.sin_cos();

        let fodder_x_offset: f32 = sincos.1 * fodder_unit_multiplier;
        let fodder_y_offset: f32 = sincos.0 * fodder_unit_multiplier;

        // Rotate offsets by +/- retreat angle degrees so that core units move diagonally backwards
        let core_left_rotate = rotate_by_angle((sincos.1, sincos.0), retreat_angle);
        let core_right_rotate = rotate_by_angle((sincos.1, sincos.0), -retreat_angle);

        let core_left_x_offset = core_left_rotate.1 * core_unit_multiplier;
        let core_left_y_offset = core_left_rotate.0 * core_unit_multiplier;
        let core_right_x_offset = core_right_rotate.1 * core_unit_multiplier;
        let core_right_y_offset = core_right_rotate.0 * core_unit_multiplier;

        // Get the distances from our fodder units to the enemy center
        let mut fodder_positions: Vec<Vec<f32>> = Vec::new();
        for unit in fodder_units.iter() {
            fodder_positions.push(vec![unit.position.0, unit.position.1]);
        }

        let fodder_distances: Vec<Vec<f32>> =
            reference_cdist(&fodder_positions, &enemy_center_vector);

        // Get the distances from our core units to the enemy center
        let mut core_positions: Vec<Vec<f32>> = Vec::new();
        for unit in core_units.iter() {
            core_positions.push(vec![unit.position.0, unit.position.1]);
        }

        let core_distances: Vec<Vec<f32>> = reference_cdist(&core_positions, &enemy_center_vector);

        // Mean positions will be used for determining whether units needs to be moved
        let core_mean: (f32, f32) = get_units_center(&core_units);
        let core_mean_distance: f32 =
            euclidean_distance(&[core_mean.0, core_mean.1], &enemy_center_vector[0]);

        let fodder_mean: (f32, f32) = get_units_center(&fodder_units);
        let fodder_mean_distance: f32 =
            euclidean_distance(&[fodder_mean.0, fodder_mean.1], &enemy_center_vector[0]);

        // Identify if a core unit is closer to the enemy than the fodder mean. If it is, back up diagonally.
        for index in 0..core_units.len() {
            if core_distances[index][0] < fodder_mean_distance {
                let adjusted_unit_position: (f32, f32) = (
                    core_units[index].position.0 - enemy_center.0,
                    core_units[index].position.1 - enemy_center.1,
                );
                let angle_to_enemy: f32 = adjusted_unit_position.1.atan2(adjusted_unit_position.0);
                let unit_sincos: (f32, f32) = angle_to_enemy.sin_cos();
                if unit_sincos.1 > 0.0 {
                    let new_position: (f32, f32) = (
                        core_units[index].position.0 + core_right_x_offset,
                        core_units[index].position.1 + core_right_y_offset,
                    );
                    tag_to_position.insert(core_units[index].tag, new_position);
                } else {
                    let new_position: (f32, f32) = (
                        core_units[index].position.0 + core_left_x_offset,
                        core_units[index].position.1 + core_left_y_offset,
                    );
                    tag_to_position.insert(core_units[index].tag, new_position);
                }
            }
        }

        // Identify if a fodder unit is further from the enemy than core mean. If it is, move forward.
        for index in 0..fodder_units.len() {
            if fodder_distances[index][0] > core_mean_distance {
                let new_position: (f32, f32) = (
                    fodder_units[index].position.0 - fodder_x_offset,
                    fodder_units[index].position.1 - fodder_y_offset,
                );
                tag_to_position.insert(fodder_units[index].tag, new_position);
            }
        }

        // Return the units that need to be moved as a (Python) Dict of tag to new position
        tag_to_position
    }

    #[pyfn(m)]
    #[pyo3(name = "adjust_moving_formation")]
    fn adjust_moving_formation(
        _py: Python,
        our_units: Vec<SC2Unit>,
        target: (f32, f32),
        fodder_tags: Vec<u64>,
        core_unit_multiplier: f32,
        retreat_angle: f32,
    ) -> HashMap<u64, (f32, f32)> {
        // Make sure core units are behind the fodder units by not moving them.

        // Create hashmap to be used as dictionary
        let mut core_unit_repositioning: HashMap<u64, (f32, f32)> = HashMap::new();

        // We need the enemy center as a vector later
        let move_target = vec![vec![target.0, target.1]];

        // If there are no fodder tags, none of the units will need their positions adjusted
        if fodder_tags.is_empty() {
            return core_unit_repositioning;
        }

        // Find the center of the units so we can get properly align retreat paths
        let our_center: (f32, f32) = get_units_center(&our_units);

        // start getting the angle by applying a translation that moves the enemy to the origin
        let our_adjusted_position: Vec<f32> =
            vec![our_center.0 - target.0, our_center.1 - target.1];

        // use atan2 to get the angle
        let angle_to_origin: f32 = our_adjusted_position[1].atan2(our_adjusted_position[0]);

        // We need sine and cosine so that we can give the correct retreat position
        let sincos: (f32, f32) = angle_to_origin.sin_cos();

        // Rotate offsets by +/- retreat angle degrees so that core units move diagonally backwards
        let core_left_rotate = rotate_by_angle((sincos.1, sincos.0), retreat_angle);
        let core_right_rotate = rotate_by_angle((sincos.1, sincos.0), -retreat_angle);

        let core_left_x_offset = core_left_rotate.1 * core_unit_multiplier;
        let core_left_y_offset = core_left_rotate.0 * core_unit_multiplier;
        let core_right_x_offset = core_right_rotate.1 * core_unit_multiplier;
        let core_right_y_offset = core_right_rotate.0 * core_unit_multiplier;

        // Separate the core units from the fodder units
        let mut core_units: Vec<SC2Unit> = Vec::new();
        let mut fodder_units: Vec<SC2Unit> = Vec::new();

        for unit in our_units.iter() {
            if fodder_tags.contains(&unit.tag) {
                fodder_units.push(*unit);
            } else {
                core_units.push(*unit);
            }
        }

        // Get the distances of all core units to the move target
        let mut core_positions: Vec<Vec<f32>> = Vec::new();
        for unit in core_units.iter() {
            core_positions.push(vec![unit.position.0, unit.position.1]);
        }

        let core_distances: Vec<Vec<f32>> = reference_cdist(&core_positions, &move_target);

        // Determine which core units need to move based on the mean fodder distance
        let fodder_mean: (f32, f32) = get_units_center(&fodder_units);
        let fodder_mean_distance: f32 =
            euclidean_distance(&[fodder_mean.0, fodder_mean.1], &[target.0, target.1]);

        // Identify if a core unit is closer to the enemy than the fodder mean. If it is, back up diagonally.
        for index in 0..core_units.len() {
            if core_distances[index][0] < fodder_mean_distance {
                let adjusted_unit_position: (f32, f32) = (
                    core_units[index].position.0 - target.0,
                    core_units[index].position.1 - target.1,
                );
                let angle_to_target: f32 = adjusted_unit_position.1.atan2(adjusted_unit_position.0);
                let unit_sincos: (f32, f32) = angle_to_target.sin_cos();
                if unit_sincos.1 > 0.0 {
                    // If cosine of angle is greater than 0, the unit is to the right of the line so move right diagonally
                    let new_position: (f32, f32) = (
                        core_units[index].position.0 + core_right_x_offset,
                        core_units[index].position.1 + core_right_y_offset,
                    );
                    core_unit_repositioning.insert(core_units[index].tag, new_position);
                } else {
                    // Otherwise, go left diagonally
                    let new_position: (f32, f32) = (
                        core_units[index].position.0 + core_left_x_offset,
                        core_units[index].position.1 + core_left_y_offset,
                    );
                    core_unit_repositioning.insert(core_units[index].tag, new_position);
                }
            }
        }

        core_unit_repositioning
    }

    #[pyfn(m)]
    #[pyo3(name = "get_positions_closer_than")]
    fn get_positions_closer_than(
        _py: Python,
        search_positions: Vec<(f32, f32)>,
        start_position: (f32, f32),
        distance: f32,
    ) -> Vec<(f32, f32)> {
        let mut close_positions: Vec<(f32, f32)> = Vec::new();
        let squared_distance: f32 = f32::powf(distance, 2.0);
        for pos in search_positions.iter() {
            let distance_to_start: f32 = get_squared_distance(*pos, start_position);
            if distance_to_start <= squared_distance {
                close_positions.push(*pos);
            }
        }

        close_positions
    }

    #[pyfn(m)]
    #[pyo3(name = "get_usize_positions_closer_than")]
    fn get_usize_positions_closer_than(
        _py: Python,
        search_positions: Vec<(usize, usize)>,
        start_position: (f32, f32),
        distance: f32,
    ) -> Vec<(usize, usize)> {
        let mut close_positions: Vec<(usize, usize)> = Vec::new();
        let squared_distance: f32 = f32::powf(distance, 2.0);
        for pos in search_positions.iter() {
            let usize_pos = (pos.0 as f32, pos.1 as f32);
            let distance_to_start: f32 = get_squared_distance(usize_pos, start_position);
            if distance_to_start <= squared_distance {
                close_positions.push(*pos);
            }
        }

        close_positions
    }

    fn rotate_by_angle(point: (f32, f32), angle: f32) -> (f32, f32) {
        let sincos = angle.sin_cos();
        let new_x = point.0 * sincos.1 - point.1 * sincos.0;
        let new_y = point.0 * sincos.0 + point.1 * sincos.1;
        (new_x, new_y)
    }

    #[pyfn(m)]
    #[pyo3(name = "get_spore_forest_positions")]
    fn get_spore_forest_positions(
        _py: Python,
        rows_columns: (isize, isize),
        center_point: (f32, f32),
        spacing: f32,
        creep: PyReadonlyArray2<u8>,
        placement: PyReadonlyArray2<u8>,
        vision: PyReadonlyArray2<u8>,
        units_to_avoid: Vec<SC2Unit>,
    ) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
        /*
        Create a diamondish grid where each center point is `spacing` away from its neighbors.
        The intersection points for circles with radius R and centers at the origin and (R, 0)
        is (R * cos(PI/3), +/- R * sin(PI/3)), or (R * .5, +/- R * .866). Therefore, each spore
        in a row should be R away from its neighbor in that row and the rows should be R * .866 apart.
        This results in the spores being equidistant to their neighbors in rows and columns.

        However, this causes some problems as far as SC2 is concerned. Buildings have to fully fill
        the tiles they occupy, so Spore Crawlers (2x2 buildings) can only be placed at integer
        coordinates. Therefore, we round the calculated positions to get something close enough to the
        initial grid that still gives us better starting coordinates for finding final placements.
        */
        let mut raw_spore_positions: Vec<(i32, i32)> = Vec::new();

        let row_count = rows_columns.0;
        let column_count = rows_columns.1;

        // We want the center spore to be close to the center point. By subtracting half of the amount
        // of rows/columns (rounded down) from the iteration variable, we can offset the values to span
        // from -half to +half. For instance, [0, 1, 2, 3, 4] becomes [-2, -1, 0, 1, 2]. This clearly
        // works best with odd values, but it's "good enough" for even values.
        let row_iterator_shift: f32 = (row_count / 2) as f32;
        let column_iterator_shift: f32 = (column_count / 2) as f32;

        // We want to place spores in a diamond shape, so every other spore in a column should be moved
        // in between the two spores in the rows above and below it
        let column_offset = spacing * 0.5;
        // How far each row should be from the neighboring rows
        // We don't have to calculate this for columns because we want to move each one over by `spacing`
        let row_distance = spacing * 0.866;

        for j in 0..row_count {
            for i in 0..column_count {
                // Separated as x and y for legibility. Back/forward/up/down are relative to the center
                let x = (spacing * (i as f32 - column_iterator_shift)     // How far back or forwards to put the spore
                    + (column_offset * ((j + 1) % 2) as f32)    // Whether this spore needs to be moved over
                    + center_point.0)
                    .round(); // The starting point should be the center

                let y = (row_distance * (j as f32 - row_iterator_shift) + center_point.1).round(); // Same process as above, minus offset
                raw_spore_positions.push((x as i32, y as i32));
            }
        }

        // Now that we have the raw positions for each spore, figure out where we can actually put them.
        // SC2 requires that the point and the 3 points "below" the placement be pathable and placeable.
        // Currently this is handled in Python, so just return what we have.

        let mut valid_positions: Vec<(i32, i32)> = Vec::new();
        let mut invalid_positions: Vec<(i32, i32)> = Vec::new();

        for point in raw_spore_positions.iter() {
            if is_valid_spore_position(point, &units_to_avoid, &vision, &placement, &creep) {
                valid_positions.push(*point)
            } else {
                invalid_positions.push(*point);
            }
        }

        for base_point in &invalid_positions {
            for neighbor in int_neighbors8(*base_point) {
                if is_valid_spore_position(&neighbor, &units_to_avoid, &vision, &placement, &creep)
                {
                    valid_positions.push(neighbor);
                    break;
                }
            }
        }

        (valid_positions, invalid_positions)
    }

    fn int_neighbors8(base_point: (i32, i32)) -> Vec<(i32, i32)> {
        let mut neighbors: Vec<(i32, i32)> = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if i == 1 && j == 1 {
                    continue;
                }
                neighbors.push((base_point.0 + i - 1, base_point.1 + j - 1));
            }
        }
        neighbors
    }

    fn u_neighbors_x(base_point: &(usize, usize), search_area: usize) -> Vec<(usize, usize)> {
        let mut neighbors: Vec<(usize, usize)> = Vec::new();
        let iterator_max: usize = 2 * search_area + 1;
        for i in 0..iterator_max {
            for j in 0..iterator_max {
                if i == search_area && j == search_area {
                    continue;
                }
                neighbors.push((
                    base_point.0 + i - search_area,
                    base_point.1 + j - search_area,
                ));
            }
        }
        neighbors
    }

    fn f32_neighbors_x(base_point: (f32, f32), search_range: i32) -> Vec<(i32, i32)> {
        let mut neighbors: Vec<(i32, i32)> = Vec::new();
        let iterator_max: i32 = 2 * search_range + 1;
        let point: (i32, i32) = (base_point.0 as i32, base_point.1 as i32);
        for i in 0..iterator_max {
            for j in 0..iterator_max {
                if i == search_range && j == search_range {
                    continue;
                }
                neighbors.push((point.0 + i - search_range, point.1 + j - search_range));
            }
        }
        neighbors
    }

    fn is_valid_spore_position(
        raw_point: &(i32, i32),
        units_to_avoid: &[SC2Unit],
        vision_grid: &PyReadonlyArray2<u8>,
        placement_grid: &PyReadonlyArray2<u8>,
        creep_grid: &PyReadonlyArray2<u8>,
    ) -> bool {
        let int_point: (usize, usize) = (raw_point.0 as usize, raw_point.1 as usize);
        // Make sure the point is far enough away from a buildings/units, else there's no need to do the other checks
        let mut units_to_avoid_positions: Vec<Vec<f32>> = Vec::new();
        for unit in units_to_avoid.iter() {
            units_to_avoid_positions.push(vec![unit.position.0, unit.position.1]);
        }
        let avoid_distances: Vec<Vec<f32>> = reference_cdist(
            &units_to_avoid_positions,
            &[vec![raw_point.0 as f32, raw_point.1 as f32]],
        );

        for dist in &avoid_distances {
            if dist[0] < 1.5 {
                return false;
            }
        }

        let all_points = [
            int_point,
            (int_point.0 - 1, int_point.1 - 1),
            (int_point.0, int_point.1 - 1),
            (int_point.0 - 1, int_point.1),
        ];
        for point in &all_points {
            // Make sure we can see every point
            if let Some(vision_value) = vision_grid.get([point.1, point.0]) {
                if *vision_value != 2 {
                    return false;
                }
            }

            // Make sure the points are all placeable
            if let Some(placement_value) = placement_grid.get([point.1, point.0]) {
                if *placement_value == 0 {
                    return false;
                }
            }
            // Make sure we have creep at every point
            if let Some(creep_value) = creep_grid.get([point.1, point.0]) {
                if *creep_value == 0 {
                    return false;
                }
            }
        }
        // The necessary points are far enough from buildings/units, in vision, placeable, and
        // have creep
        true
    }

    #[pyfn(m)]
    #[pyo3(name = "is_valid_2x2_position")]
    fn is_valid_2x2_position(
        _py: Python,
        raw_point: (i32, i32),
        units_to_avoid: Vec<SC2Unit>,
        vision_grid: PyReadonlyArray2<u8>,
        placement_grid: PyReadonlyArray2<u8>,
        creep_grid: PyReadonlyArray2<u8>,
    ) -> bool {
        let int_point: (usize, usize) = (raw_point.0 as usize, raw_point.1 as usize);
        // Make sure the point is far enough away from a buildings/units, else there's no need to do the other checks
        let mut units_to_avoid_positions: Vec<Vec<f32>> = Vec::new();
        for unit in units_to_avoid.iter() {
            units_to_avoid_positions.push(vec![unit.position.0, unit.position.1]);
        }
        let avoid_distances: Vec<Vec<f32>> = reference_cdist(
            &units_to_avoid_positions,
            &[vec![raw_point.0 as f32, raw_point.1 as f32]],
        );

        for dist in &avoid_distances {
            if dist[0] < 1.5 {
                return false;
            }
        }

        let all_points = [
            int_point,
            (int_point.0 - 1, int_point.1 - 1),
            (int_point.0, int_point.1 - 1),
            (int_point.0 - 1, int_point.1),
        ];
        for point in &all_points {
            // Make sure we can see every point
            if let Some(vision_value) = vision_grid.get([point.1, point.0]) {
                if *vision_value != 2 {
                    return false;
                }
            }

            // Make sure the points are all placeable
            if let Some(placement_value) = placement_grid.get([point.1, point.0]) {
                if *placement_value == 0 {
                    return false;
                }
            }
            // Make sure we have creep at every point
            if let Some(creep_value) = creep_grid.get([point.1, point.0]) {
                if *creep_value == 0 {
                    return false;
                }
            }
        }
        // The necessary points are far enough from buildings/units, in vision, placeable, and
        // have creep
        true
    }

    #[pyfn(m)]
    #[pyo3(name = "get_blocked_spore_position")]
    fn get_blocked_spore_position(
        _py: Python,
        unit: SC2Unit,
        distances: PyReadonlyArray1<f64>,
        creep: PyReadonlyArray2<u8>,
        placement: PyReadonlyArray2<u8>,
        vision: PyReadonlyArray2<u8>,
        units_to_avoid: Vec<SC2Unit>,
    ) -> (i32, i32) {
        let mut valid_positions: Vec<(i32, i32)> = Vec::new();
        if let Ok(full_dist_vec) = distances.to_vec() {
            let mut dist_vec: Vec<f32> = Vec::new();
            for val in &full_dist_vec {
                if *val != 0.0 {
                    dist_vec.push(*val as f32);
                }
            }
            let min_distance = get_min(&dist_vec);
            if min_distance < 1.5 {
                for pos in &f32_neighbors_x(unit.position, 3) {
                    if is_valid_spore_position(pos, &units_to_avoid, &vision, &placement, &creep) {
                        valid_positions.push((pos.0 as i32, pos.1 as i32));
                    }
                }
            }
        }

        if !valid_positions.is_empty() {
            let mut shuffled = valid_positions;
            let mut rng = thread_rng();
            shuffled.shuffle(&mut rng);
            return shuffled[0];
        }
        // Check for return value of (0, 0) as this meeans no position was found
        (0, 0)
    }

    #[pyfn(m)]
    #[pyo3(name = "find_ling_drop_point")]
    fn find_ling_drop_point(
        _py: Python,
        terrain_map: PyReadonlyArray2<u8>,
        pathing_map: PyReadonlyArray2<u8>,
        rounded_enemy_start: (usize, usize),
        away_from_point: (f32, f32),
        base_perimeter_points: Vec<(usize, usize)>,
        search_area: usize,
    ) -> (usize, usize) {
        let mut valid_points: Vec<(usize, usize)> = Vec::new();
        if let Some(start_height) = terrain_map.get([rounded_enemy_start.1, rounded_enemy_start.0])
        {
            for perimeter in &base_perimeter_points {
                'point_loop: for point in &u_neighbors_x(perimeter, search_area) {
                    if let Some(point_height) = terrain_map.get([point.1, point.0]) {
                        if point_height != start_height {
                            if let Some(point_is_pathable) = pathing_map.get([point.1, point.0]) {
                                if *point_is_pathable == 1 {
                                    valid_points.push(*point);
                                    break 'point_loop;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut dist: f32;
        let mut max_dist: f32 = -1.0;
        let mut best_point: (usize, usize) = (0, 0); // will definitely be set to something else
        let away_from: &[f32] = &[away_from_point.0, away_from_point.1];
        for point in &valid_points {
            dist = euclidean_distance(&[point.0 as f32, point.1 as f32], away_from);
            if dist > max_dist {
                max_dist = dist;
                best_point = *point;
            }
        }
        best_point
    }

    #[pyfn(m)]
    #[pyo3(name = "all_points_have_creep")]
    fn all_points_have_creep(
        _py: Python,
        points: Vec<(usize, usize)>,
        creep: PyReadonlyArray2<u8>,
    ) -> bool {
        for point in &points {
            if let Some(creep_value) = creep.get([point.1, point.0]) {
                if *creep_value == 0 {
                    return false;
                }
            }
        }
        true
    }

    #[pyfn(m)]
    #[pyo3(name = "terrain_flood_fill")]
    fn terrain_flood_fill(
        _py: Python,
        start_point: (usize, usize),
        terrain_grid: PyReadonlyArray2<u8>,
        pathing_grid: PyReadonlyArray2<u8>,
        max_distance: f32,
        choke_points: HashSet<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let mut filled_points: Vec<(usize, usize)> = Vec::new();
        let terrain_height: u8;
        // Only continue if we can get a height for the starting point
        if let Some(terrain_value) = terrain_grid.get([start_point.0, start_point.1]) {
            terrain_height = *terrain_value;
        } else {
            return filled_points;
        }
        // Only continue if the starting point is pathable
        if let Some(pathing_value) = pathing_grid.get([start_point.0, start_point.1]) {
            if pathing_value != &1 {
                return filled_points;
            }
        } else {
            return filled_points;
        }
        grid_flood_fill(
            start_point,
            &terrain_grid,
            &pathing_grid,
            terrain_height,
            &mut filled_points,
            start_point,
            max_distance,
            &choke_points,
            true,
        );
        filled_points
    }

    fn grid_flood_fill(
        point: (usize, usize),
        terrain_grid: &PyReadonlyArray2<u8>,
        pathing_grid: &PyReadonlyArray2<u8>,
        target_val: u8,
        current_vec: &mut Vec<(usize, usize)>,
        start_point: (usize, usize),
        max_distance: f32,
        choke_points: &HashSet<(usize, usize)>,
        initial: bool,
    ) {
        // Check that we haven't already added this point. If we've checked the point but not added it,
        // it's fast enough to check again that I'm ignoring it. May revisit in the future.
        if current_vec.contains(&point) {
            return;
        }
        // Check that this point isn't too far away from the start
        if euclidean_distance(
            &[point.0 as f32, point.1 as f32],
            &[start_point.0 as f32, start_point.1 as f32],
        ) > max_distance
        {
            return;
        }
        // Only keep going if we're able to get a value for this point
        let grid_result: u8;
        if let Some(grid_value) = terrain_grid.get([point.0, point.1]) {
            grid_result = *grid_value;
        } else {
            return;
        }
        // Only keep going if the point is pathable
        if let Some(pathing_value) = pathing_grid.get([point.0, point.1]) {
            if pathing_value != &1 {
                return;
            }
        } else {
            return;
        }
        // Only check more points if:
        //      - the grid value of the point is within the tolerance for non-ramp height variation
        //      - the point is not inside a choke
        // Values have to be u8, make sure we don't overflow
        let min_bound: u8;
        let max_bound: u8;
        if let Some(min_height) = target_val.checked_sub(TERRAIN_HEIGHT_TOLERANCE) {
            min_bound = min_height;
        } else {
            min_bound = 0;
        }
        if let Some(max_height) = target_val.checked_add(TERRAIN_HEIGHT_TOLERANCE) {
            max_bound = max_height;
        } else {
            max_bound = 255;
        }
        if (min_bound < grid_result && grid_result < max_bound)
            && ((!choke_points.contains(&point)) || initial)
        {
            // Add the point to the valid points and check a new set of points
            current_vec.push(point);
            grid_flood_fill(
                (point.0 + 1, point.1),
                terrain_grid,
                pathing_grid,
                target_val,
                current_vec,
                start_point,
                max_distance,
                choke_points,
                false,
            );
            grid_flood_fill(
                (point.0 - 1, point.1),
                terrain_grid,
                pathing_grid,
                target_val,
                current_vec,
                start_point,
                max_distance,
                choke_points,
                false,
            );
            grid_flood_fill(
                (point.0, point.1 + 1),
                terrain_grid,
                pathing_grid,
                target_val,
                current_vec,
                start_point,
                max_distance,
                choke_points,
                false,
            );
            grid_flood_fill(
                (point.0, point.1 - 1),
                terrain_grid,
                pathing_grid,
                target_val,
                current_vec,
                start_point,
                max_distance,
                choke_points,
                false,
            );
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "number_of_chokes_pathed_through")]
    fn number_of_chokes_pathed_through(
        _py: Python,
        path: Vec<(usize, usize)>,
        chokes_list: Vec<HashSet<(usize, usize)>>,
    ) -> isize {
        // Given a path and a list of the points comprising chokes, determine how many chokes
        // the path goes through
        let mut num_chokes_pathed_through: isize = 0;
        for ch in chokes_list {
            'inner: for p in &path {
                if ch.contains(p) {
                    // At least one point in the path goes through this choke, no need to check more points
                    num_chokes_pathed_through += 1;
                    break 'inner;
                }
            }
        }

        num_chokes_pathed_through
    }

    // #[pyfn(m)]
    // #[pyo3(name = "create_region_for_convolution")]
    // fn create_region_for_convolution(
    //     _py: Python,
    //     creep: PyReadonlyArray2<u8>,
    //     placement: PyReadonlyArray2<u8>,
    //     vision: PyReadonlyArray2<u8>,
    //     points: Vec<(usize, usize)>,
    // ) -> PyArray<usize, Ix2> {
    //     // Only use the maximum values so it'll be easier to put them into the array
    //     let mut x_max = 0;
    //     let mut y_max = 0;
    //     for p in points.iter() {
    //         if p.0 > x_max {
    //             x_max = p.0
    //         }
    //         if p.1 > y_max {
    //             y_max = p.1
    //         }
    //     }
    //     let mut conv_array = Array2::<usize>::zeros((x_max, y_max));
    //     let gil = Python::acquire_gil();
    //     let pyarray = PyArray::arange(gil.python(), 0, x_max * y_max, 1)
    //         .reshape([x_max, y_max])
    //         .unwrap();
    //     let mut point: (usize, usize);
    //     for p in points.iter() {
    //         point = (p.1, p.0);
    //         if let Some(vision_value) = vision.get(point) {
    //             if vision_value == &2u8 {
    //                 if let Some(creep_value) = creep.get(point) {
    //                     if creep_value == &1u8 {
    //                         if let Some(placement_value) = placement.get(point) {
    //                             if placement_value != &0u8 {
    //                                 conv_array[[p.0, p.1]] = 1;
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     pyarray
    // }

    Ok(())
}
