// Bounding Circle code adapted from somewhere

mod sc2_unit;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

use crate::sc2_unit::SC2Unit;
use pyo3::prelude::*;

const MULTIPLICATIVE_EPSILON: f32 = 1.00000000000001;
const NONEXISTENTCIRCLE: (f32, f32, f32) = (f32::NAN, f32::NAN, f32::NAN);

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
        return neighbors;
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
        for i in 0..units.len() {
            let dist: f32 = get_squared_distance(units[i].position, target_position);
            if dist < closest_distance {
                closest_index = i;
                closest_distance = dist;
            }
        }
        return closest_index;
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
        for i in 0..positions.len() {
            let dist: f32 = get_squared_distance(positions[i], target_position);
            if dist < closest_distance {
                closest_index = i;
                closest_distance = dist;
            }
        }
        return closest_index;
    }

    #[pyfn(m)]
    #[pyo3(name = "cdist")]
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
        return (max_units_found, center_position.to_vec());
    }

    #[pyfn(m)]
    #[pyo3(name = "surround_complete")]
    fn surround_complete(
        _py: Python,
        units: Vec<SC2Unit>,
        our_center: Vec<f32>,
        enemy_center: Vec<f32>,
        offset: f32,
        ratio: f32,
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
        if side_one == 0.0 || side_two == 0.0 {
            return false;
        } else if side_one / side_two <= (1.0 / ratio) && side_one / side_two <= ratio {
            return true;
        } else {
            return true;
        }
    }

    fn reference_cdist(xa: &Vec<Vec<f32>>, xb: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        // Form a matrix containing the pairwise distances between the points given
        // This is for calling internally, Python functions should use "cdist"
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

    fn hypot(x: f32, y: f32) -> f32 {
        return (f32::powf(x, 2.0) + f32::powf(y, 2.0)).sqrt();
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
        return make_circle(ref_points);
    }

    fn make_circle(points: Vec<&Vec<f32>>) -> (f32, f32, f32) {
        let mut shuffled = points;
        let mut rng = thread_rng();
        shuffled.shuffle(&mut rng);

        let mut c = (f32::NAN, f32::NAN, f32::NAN);
        for (i, p) in shuffled.iter().cloned().enumerate() {
            if c == NONEXISTENTCIRCLE || !is_in_circle(&c, &p) {
                c = _make_circle_one_point(shuffled[i + 1..].to_vec(), p.to_vec());
            }
        }

        return c;
    }

    fn _make_circle_one_point(points: Vec<&Vec<f32>>, p: Vec<f32>) -> (f32, f32, f32) {
        let mut c = (p[0], p[1], 0.0f32);
        for (i, q) in points.iter().enumerate() {
            if !is_in_circle(&c, q) {
                if c.2 == 0.0 {
                    c = make_diameter(&p, &q);
                } else {
                    c = make_circle_two_points(points[i + 1..].to_vec(), p.to_vec(), q.to_vec());
                }
            }
        }
        return c;
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
            let c = make_circumcircle(&p, &q, &r);
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
        if left == NONEXISTENTCIRCLE && right == NONEXISTENTCIRCLE {
            return circle;
        } else if left == NONEXISTENTCIRCLE {
            return right;
        } else if right == NONEXISTENTCIRCLE {
            return left;
        } else {
            if left.2 <= right.2 {
                return left;
            } else {
                return right;
            }
        }
    }

    fn make_diameter(a: &Vec<f32>, b: &Vec<f32>) -> (f32, f32, f32) {
        // Return the average of a and b and the radius of the circle centered on the average point that includes
        // both a and b
        let cx = (a[0] + b[0]) / 2.0;
        let cy = (a[1] + b[1]) / 2.0;
        let r0 = hypot(cx - a[0], cy - a[1]);
        let r1 = hypot(cx - b[0], cy - b[1]);
        return (cx, cy, get_max(&vec![r0, r1]));
    }

    fn make_circumcircle(a: &Vec<f32>, b: &Vec<f32>, c: &Vec<f32>) -> (f32, f32, f32) {
        // Mathematical algorithm from Wikipedia: Circumscribed circle
        let x_coords: Vec<f32> = vec![a[0], b[0], c[0]];
        let y_coords: Vec<f32> = vec![a[1], b[1], c[1]];
        let ox: f32 = (get_min(&x_coords) + get_max(&x_coords)) / 2.0;
        let oy: f32 = (get_min(&y_coords) + get_max(&y_coords)) / 2.0;

        let ax: f32 = a[0] - ox;
        let ay: f32 = a[1] - oy;
        let bx: f32 = b[0] - ox;
        let by: f32 = b[1] - oy;
        let cx: f32 = c[0] - ox;
        let cy: f32 = c[1] - oy;

        let d: f32 = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0;
        if d == 0.0 {
            return (f32::NAN, f32::NAN, f32::NAN);
        }

        let x: f32 = ox
            + ((ax * ax + ay * ay) * (by - cy)
                + (bx * bx + by * by) * (cy - ay)
                + (cx * cx + cy * cy) * (ay - by))
                / d;
        let y: f32 = oy
            + ((ax * ax + ay * ay) * (cx - bx)
                + (bx * bx + by * by) * (ax - cx)
                + (cx * cx + cy * cy) * (bx - ax))
                / d;

        let ra: f32 = hypot(x - a[0], y - a[0]);
        let rb: f32 = hypot(x - b[0], y - b[1]);
        let rc: f32 = hypot(x - c[0], y - c[1]);

        return (x, y, get_max(&vec![ra, rb, rc]));
    }

    fn is_in_circle(c: &(f32, f32, f32), p: &Vec<f32>) -> bool {
        return c != &NONEXISTENTCIRCLE
            && hypot(p[0] - c.0, p[1] - c.1) <= c.2 * MULTIPLICATIVE_EPSILON;
    }

    fn cross_product(x0: &f32, y0: &f32, x1: &f32, y1: &f32, x2: &f32, y2: &f32) -> f32 {
        // Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
        return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    }

    fn get_max(vector: &Vec<f32>) -> f32 {
        // Rust doesn't have an easy way of getting the maximum value of a Vec<f32> so I'm using this.
        let mut max = vector[0];
        for val in vector.iter() {
            if val > &max {
                max = *val;
            }
        }
        return max;
    }

    fn get_min(vector: &Vec<f32>) -> f32 {
        // Rust doesn't have an easy way of getting the minimum value of a Vec<f32> so I'm using this.
        let mut min = vector[0];
        for val in vector.iter() {
            if val < &min {
                min = *val;
            }
        }
        return min;
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
        return (tot_x / num_positions, tot_y / num_positions);
    }

    fn get_units_center(units: &Vec<SC2Unit>) -> (f32, f32) {
        // Given a list of units, get the mean position.
        let mut position_vector = Vec::new();
        for unit in units.iter() {
            position_vector.push(unit.position);
        }
        return get_positions_center(position_vector);
    }

    #[pyfn(m)]
    #[pyo3(name = "adjust_combat_formation")]
    fn adjust_combat_formation(
        _py: Python,
        our_units: Vec<SC2Unit>,
        enemies: Vec<SC2Unit>,
        mut fodder_tags: Vec<u64>,
        core_unit_multiplier: f32,
        fodder_unit_multiplier: f32,
        retreat_angle: f32,
        _caster_tags: Vec<u64>,
    ) -> HashMap<u64, (f32, f32)> {
        // Takes our units, enemy units, our fodder units, and our casters
        // Returns a (Python) list of tags and positions, where each position is where the unit with that tag should move

        // This will be the dictionary of tag to new position
        let mut tag_to_position: HashMap<u64, (f32, f32)> = HashMap::new();

        // Identify fodder units. If fodder tags are given, those are the fodder units. If they aren't, the fodder units
        // are the units with a health percentage higher than the average.
        if fodder_tags.len() == 0 {
            let mut total_health_percent: f32 = 0.0;
            for unit in our_units.iter() {
                total_health_percent += unit.health_percentage
            }
            let avg_health: f32 = total_health_percent / our_units.len() as f32;
            for unit in our_units.iter() {
                if unit.health_percentage > avg_health {
                    fodder_tags.push(unit.tag)
                }
            }
        }

        // If there are no fodder tags or enemies, none of the units will need their positions adjusted
        if fodder_tags.len() == 0 || enemies.len() == 0 {
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
            euclidean_distance(&vec![core_mean.0, core_mean.1], &enemy_center_vector[0]);

        let fodder_mean: (f32, f32) = get_units_center(&fodder_units);
        let fodder_mean_distance: f32 =
            euclidean_distance(&vec![fodder_mean.0, fodder_mean.1], &enemy_center_vector[0]);

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
        return tag_to_position;
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
        if fodder_tags.len() == 0 {
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
        let fodder_mean_distance: f32 = euclidean_distance(
            &vec![fodder_mean.0, fodder_mean.1],
            &vec![target.0, target.1],
        );

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

        return core_unit_repositioning;
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

        return close_positions;
    }

    fn rotate_by_angle(point: (f32, f32), angle: f32) -> (f32, f32) {
        let sincos = angle.sin_cos();
        let new_x = point.0 * sincos.1 - point.1 * sincos.0;
        let new_y = point.0 * sincos.0 + point.1 * sincos.1;
        return (new_x, new_y);
    }

    Ok(())
}
