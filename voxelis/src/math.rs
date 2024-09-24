use bevy::math::Vec3;

pub(crate) fn triangle_cube_intersection(triangle: (Vec3, Vec3, Vec3), cube: (Vec3, Vec3)) -> bool {
    let (tv0, tv1, tv2) = triangle;
    let (cube_min, cube_max) = cube;

    // Calculate the bounding box of the triangle
    let tri_min = tv0.min(tv1).min(tv2);
    let tri_max = tv0.max(tv1).max(tv2);

    // Check if the bounding boxes overlap or touch
    let epsilon = 1e-5;
    if tri_max.x < cube_min.x - epsilon
        || tri_min.x > cube_max.x + epsilon
        || tri_max.y < cube_min.y - epsilon
        || tri_min.y > cube_max.y + epsilon
        || tri_max.z < cube_min.z - epsilon
        || tri_min.z > cube_max.z + epsilon
    {
        return false;
    }

    // Check if the triangle's plane intersects the cube
    let normal = (tv1 - tv0).cross(tv2 - tv0);
    let d = -normal.dot(tv0);

    let p1 = cube_min;
    let p2 = Vec3::new(cube_max.x, cube_min.y, cube_min.z);
    let p3 = Vec3::new(cube_max.x, cube_max.y, cube_min.z);
    let p4 = Vec3::new(cube_min.x, cube_max.y, cube_min.z);
    let p5 = Vec3::new(cube_min.x, cube_min.y, cube_max.z);
    let p6 = Vec3::new(cube_max.x, cube_min.y, cube_max.z);
    let p7 = cube_max;
    let p8 = Vec3::new(cube_min.x, cube_max.y, cube_max.z);

    let cube_points = [p1, p2, p3, p4, p5, p6, p7, p8];
    let sign = (normal.dot(cube_points[0]) + d).signum();

    for p in &cube_points[1..] {
        let new_sign = (normal.dot(*p) + d).signum();
        if new_sign != sign {
            return true; // The plane intersects the cube
        }
    }

    // Check if any of the triangle's vertices are inside or on the cube
    if point_in_or_on_cube(tv0, cube)
        || point_in_or_on_cube(tv1, cube)
        || point_in_or_on_cube(tv2, cube)
    {
        return true;
    }

    // Check if any of the cube's vertices are inside or on the triangle
    let cube_vertices = [
        Vec3::new(cube_min.x, cube_min.y, cube_min.z),
        Vec3::new(cube_max.x, cube_min.y, cube_min.z),
        Vec3::new(cube_max.x, cube_max.y, cube_min.z),
        Vec3::new(cube_min.x, cube_max.y, cube_min.z),
        Vec3::new(cube_min.x, cube_min.y, cube_max.z),
        Vec3::new(cube_max.x, cube_min.y, cube_max.z),
        Vec3::new(cube_max.x, cube_max.y, cube_max.z),
        Vec3::new(cube_min.x, cube_max.y, cube_max.z),
    ];

    for &vertex in &cube_vertices {
        if point_in_or_on_triangle(vertex, triangle) {
            return true;
        }
    }

    // Check if any of the triangle's edges intersect with any of the cube's faces
    let triangle_edges = [(tv0, tv1), (tv1, tv2), (tv2, tv0)];
    let cube_faces = [
        (
            cube_vertices[0],
            cube_vertices[1],
            cube_vertices[2],
            cube_vertices[3],
        ), // Front
        (
            cube_vertices[4],
            cube_vertices[5],
            cube_vertices[6],
            cube_vertices[7],
        ), // Back
        (
            cube_vertices[0],
            cube_vertices[1],
            cube_vertices[5],
            cube_vertices[4],
        ), // Bottom
        (
            cube_vertices[2],
            cube_vertices[3],
            cube_vertices[7],
            cube_vertices[6],
        ), // Top
        (
            cube_vertices[0],
            cube_vertices[3],
            cube_vertices[7],
            cube_vertices[4],
        ), // Left
        (
            cube_vertices[1],
            cube_vertices[2],
            cube_vertices[6],
            cube_vertices[5],
        ), // Right
    ];

    for &edge in &triangle_edges {
        for &face in &cube_faces {
            if edge_quad_intersection(edge, face) {
                return true;
            }
        }
    }

    false
}

fn point_in_or_on_cube(point: Vec3, cube: (Vec3, Vec3)) -> bool {
    let (cube_min, cube_max) = cube;
    let epsilon = 1e-5;

    point.x >= cube_min.x - epsilon
        && point.x <= cube_max.x + epsilon
        && point.y >= cube_min.y - epsilon
        && point.y <= cube_max.y + epsilon
        && point.z >= cube_min.z - epsilon
        && point.z <= cube_max.z + epsilon
}

fn point_in_or_on_triangle(point: Vec3, triangle: (Vec3, Vec3, Vec3)) -> bool {
    let (a, b, c) = triangle;
    let epsilon = 1e-5;

    // Check if the point is in the same plane as the triangle
    let normal = (b - a).cross(c - a);
    let distance_to_plane = normal.dot(point - a);
    if distance_to_plane.abs() > epsilon {
        return false;
    }

    let v0 = b - a;
    let v1 = c - a;
    let v2 = point - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    u >= -epsilon && v >= -epsilon && w >= -epsilon && (u + v + w).abs() <= 1.0 + epsilon
}

fn edge_quad_intersection(edge: (Vec3, Vec3), quad: (Vec3, Vec3, Vec3, Vec3)) -> bool {
    let (e1, e2) = edge;
    let (q1, q2, q3, q4) = quad;
    let epsilon = 1e-5;

    // Check if the edge intersects with either triangle of the quad
    if triangle_edge_intersection(edge, (q1, q2, q3))
        || triangle_edge_intersection(edge, (q1, q3, q4))
    {
        return true;
    }

    // Compute average normal for the potentially non-planar quad
    let normal1 = (q2 - q1).cross(q3 - q1).normalize();
    let normal2 = (q3 - q2).cross(q4 - q2).normalize();
    let normal3 = (q4 - q3).cross(q1 - q3).normalize();
    let normal4 = (q1 - q4).cross(q2 - q4).normalize();
    let avg_normal = (normal1 + normal2 + normal3 + normal4).normalize();

    let edge_vec = e2 - e1;
    let center = (q1 + q2 + q3 + q4) * 0.25;

    // Check if the edge is parallel to the quad's average plane
    if avg_normal.dot(edge_vec).abs() < epsilon {
        // Check if the edge is close to the quad's plane
        let dist_to_plane = avg_normal.dot(e1 - center).abs();
        if dist_to_plane < epsilon {
            // Edge is coplanar with quad, check if it intersects the quad's boundaries
            let quad_edges = [(q1, q2), (q2, q3), (q3, q4), (q4, q1)];
            for &quad_edge in &quad_edges {
                if line_segment_overlap(edge, quad_edge) {
                    return true;
                }
            }

            // Check if the edge is fully contained within the quad
            let edge_midpoint = (e1 + e2) * 0.5;
            if point_in_quad(edge_midpoint, quad) {
                return true;
            }
        }
    } else {
        // Edge is not parallel to the quad's plane, check for intersection
        let t = avg_normal.dot(center - e1) / avg_normal.dot(edge_vec);
        if t >= 0.0 && t <= 1.0 {
            let intersection = e1 + edge_vec * t;
            if point_in_quad(intersection, quad) {
                return true;
            }
        }
    }

    false
}

fn point_in_quad(point: Vec3, quad: (Vec3, Vec3, Vec3, Vec3)) -> bool {
    let (a, b, c, d) = quad;
    let epsilon = 1e-5;

    // Compute an average normal for the potentially non-planar quad
    let normal1 = (b - a).cross(c - a).normalize();
    let normal2 = (c - b).cross(d - b).normalize();
    let normal3 = (d - c).cross(a - c).normalize();
    let normal4 = (a - d).cross(b - d).normalize();
    let avg_normal = (normal1 + normal2 + normal3 + normal4).normalize();

    // Compute the center of the quad
    let center = (a + b + c + d) * 0.25;

    // Project the point onto the average plane of the quad
    let dist_to_plane = avg_normal.dot(point - center);
    let projected_point = point - avg_normal * dist_to_plane;

    // Check if the point is too far from the quad's plane
    if dist_to_plane.abs() > epsilon {
        return false;
    }

    // Function to check if a point is on the right side of an edge
    let is_on_right_side = |v1: Vec3, v2: Vec3| -> bool {
        let edge = v2 - v1;
        let to_point = projected_point - v1;
        avg_normal.dot(edge.cross(to_point)) >= -epsilon
    };

    // Check if the projected point is on the right side of all edges
    is_on_right_side(a, b)
        && is_on_right_side(b, c)
        && is_on_right_side(c, d)
        && is_on_right_side(d, a)
}

fn triangle_edge_intersection(edge: (Vec3, Vec3), triangle: (Vec3, Vec3, Vec3)) -> bool {
    let (e1, e2) = edge;
    let (t1, t2, t3) = triangle;
    let epsilon = 1e-5;

    // Check if any of the triangle vertices lie on the edge
    if point_on_line_segment(t1, edge)
        || point_on_line_segment(t2, edge)
        || point_on_line_segment(t3, edge)
    {
        return true;
    }

    // Check if either endpoint of the edge is inside or on the triangle
    if point_in_or_on_triangle(e1, triangle) || point_in_or_on_triangle(e2, triangle) {
        return true;
    }

    // Compute the triangle's normal
    let normal = (t2 - t1).cross(t3 - t1);

    // If the normal is zero, the triangle is degenerate (a line or point)
    if normal.length_squared() < epsilon {
        // Check if the degenerate triangle overlaps with the edge
        return line_segment_overlap(edge, (t1, t2))
            || line_segment_overlap(edge, (t2, t3))
            || line_segment_overlap(edge, (t3, t1));
    }

    let normal = normal.normalize();
    let d = -normal.dot(t1);

    // Compute the intersection point
    let edge_vec = e2 - e1;
    let denom = normal.dot(edge_vec);

    // If denom is zero, the edge is parallel to the triangle's plane
    if denom.abs() < epsilon {
        // The edge is parallel to the triangle's plane
        // Check if it's coplanar
        if (normal.dot(e1) + d).abs() < epsilon {
            // The edge is coplanar with the triangle
            // Check if it intersects with any of the triangle's edges
            return line_segment_overlap(edge, (t1, t2))
                || line_segment_overlap(edge, (t2, t3))
                || line_segment_overlap(edge, (t3, t1));
        }
        return false;
    }

    let t = -(normal.dot(e1) + d) / denom;

    // Check if the intersection point is on the edge
    if !(0.0..=1.0).contains(&t) {
        return false;
    }

    // Compute the intersection point
    let intersection = e1 + edge_vec * t;

    // Check if the intersection point is inside or on the triangle
    point_in_or_on_triangle(intersection, triangle)
}

fn point_on_line_segment(point: Vec3, segment: (Vec3, Vec3)) -> bool {
    let (a, b) = segment;
    let epsilon = 1e-5;

    let ab = b - a;
    let ap = point - a;

    // Check if the point is collinear with the line segment
    if ab.cross(ap).length_squared() > epsilon * epsilon {
        return false;
    }

    // Check if the point is within the bounds of the line segment
    let t = ap.dot(ab) / ab.length_squared();
    -epsilon <= t && t <= 1.0 + epsilon
}

fn line_segment_overlap(seg1: (Vec3, Vec3), seg2: (Vec3, Vec3)) -> bool {
    let (a, b) = seg1;
    let (c, d) = seg2;
    let epsilon = 1e-6; // Smaller epsilon for more precision

    // Check if the segments are parallel
    let dir1 = (b - a).normalize();
    let dir2 = (d - c).normalize();
    if dir1.cross(dir2).length_squared() < epsilon * epsilon {
        // Segments are parallel, check for collinearity
        let ac = c - a;
        if ac.cross(dir1).length_squared() > epsilon * epsilon {
            return false; // Parallel but not collinear
        }
        // Check for overlap on the same line
        let t1 = ac.dot(dir1);
        let t2 = (d - a).dot(dir1);
        let s2 = (b - a).length();
        return (0.0..=s2).contains(&t1) || (0.0..=s2).contains(&t2) || (t1 <= 0.0 && t2 >= s2);
    }

    // Not parallel, check for intersection
    let n = dir1.cross(dir2);
    let ac = c - a;
    let t = ac.cross(dir2).dot(n) / n.length_squared();
    let u = ac.cross(dir1).dot(n) / n.length_squared();

    (-epsilon..=1.0 + epsilon).contains(&t) && (-epsilon..=1.0 + epsilon).contains(&u)
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_point_in_or_on_cube {
        use super::*;

        #[test]
        fn test_inside_cube() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

            // Inside the cube
            assert!(point_in_or_on_cube(Vec3::new(0.5, 0.5, 0.5), cube));
        }

        #[test]
        fn test_on_edges() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

            // On the edges
            assert!(point_in_or_on_cube(Vec3::new(0.0, 0.5, 0.5), cube));
            assert!(point_in_or_on_cube(Vec3::new(1.0, 0.5, 0.5), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.5, 0.0, 0.5), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.5, 1.0, 0.5), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.5, 0.5, 0.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.5, 0.5, 1.0), cube));
        }

        #[test]
        fn test_on_faces() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

            // On the faces
            assert!(point_in_or_on_cube(Vec3::new(0.0, 0.0, 0.5), cube));
            assert!(point_in_or_on_cube(Vec3::new(1.0, 1.0, 0.5), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.5, 0.0, 0.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.5, 1.0, 1.0), cube));
        }

        #[test]
        fn test_at_vertices() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

            // At the vertices
            assert!(point_in_or_on_cube(Vec3::new(0.0, 0.0, 0.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(1.0, 0.0, 0.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.0, 1.0, 0.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.0, 0.0, 1.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(1.0, 1.0, 0.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(1.0, 0.0, 1.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(0.0, 1.0, 1.0), cube));
            assert!(point_in_or_on_cube(Vec3::new(1.0, 1.0, 1.0), cube));
        }

        #[test]
        fn test_outside_cube() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

            // Outside the cube
            assert!(!point_in_or_on_cube(Vec3::new(-0.1, 0.5, 0.5), cube));
            assert!(!point_in_or_on_cube(Vec3::new(1.1, 0.5, 0.5), cube));
            assert!(!point_in_or_on_cube(Vec3::new(0.5, -0.1, 0.5), cube));
            assert!(!point_in_or_on_cube(Vec3::new(0.5, 1.1, 0.5), cube));
            assert!(!point_in_or_on_cube(Vec3::new(0.5, 0.5, -0.1), cube));
            assert!(!point_in_or_on_cube(Vec3::new(0.5, 0.5, 1.1), cube));
        }

        #[test]
        fn test_very_close_to_cube_but_outside() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let epsilon = 1e-5;

            // Very close to the cube but outside
            assert!(!point_in_or_on_cube(
                Vec3::new(1.0 + epsilon, 0.5, 0.5),
                cube
            ));
            assert!(!point_in_or_on_cube(
                Vec3::new(0.5, 1.0 + epsilon, 0.5),
                cube
            ));
            assert!(!point_in_or_on_cube(
                Vec3::new(0.5, 0.5, 1.0 + epsilon),
                cube
            ));
        }

        #[test]
        fn test_at_center_of_cube() {
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

            // At the center of the cube
            assert!(point_in_or_on_cube(Vec3::new(0.5, 0.5, 0.5), cube));
        }
    }

    mod test_point_in_or_on_triangle {
        use super::*;

        #[test]
        fn test_inside_triangle() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // Inside the triangle
            assert!(point_in_or_on_triangle(
                Vec3::new(0.25, 0.25, 0.0),
                triangle
            ));
        }

        #[test]
        fn test_on_edges() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // On the edges
            assert!(point_in_or_on_triangle(Vec3::new(0.5, 0.0, 0.0), triangle));
            assert!(point_in_or_on_triangle(Vec3::new(0.0, 0.5, 0.0), triangle));
            assert!(point_in_or_on_triangle(Vec3::new(0.5, 0.5, 0.0), triangle));
        }

        #[test]
        fn test_on_vertices() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // On the vertices
            assert!(point_in_or_on_triangle(Vec3::new(0.0, 0.0, 0.0), triangle));
            assert!(point_in_or_on_triangle(Vec3::new(1.0, 0.0, 0.0), triangle));
            assert!(point_in_or_on_triangle(Vec3::new(0.0, 1.0, 0.0), triangle));
        }

        #[test]
        fn test_outside_triangle_same_plane() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // Outside the triangle but in the same plane
            assert!(!point_in_or_on_triangle(Vec3::new(1.0, 1.0, 0.0), triangle));
            assert!(!point_in_or_on_triangle(
                Vec3::new(-0.1, 0.5, 0.0),
                triangle
            ));
            assert!(!point_in_or_on_triangle(
                Vec3::new(0.5, -0.1, 0.0),
                triangle
            ));
        }

        #[test]
        fn test_outside_triangle_not_same_plane() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // Outside the triangle and not in the same plane
            assert!(!point_in_or_on_triangle(
                Vec3::new(0.25, 0.25, 1.0),
                triangle
            ));
            assert!(!point_in_or_on_triangle(Vec3::new(0.5, 0.5, 1.0), triangle));
            assert!(!point_in_or_on_triangle(Vec3::new(1.0, 1.0, 1.0), triangle));
        }

        #[test]
        fn test_very_close_to_triangle_but_outside() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            let epsilon = 1e-5;

            // Very close to the triangle but outside
            assert!(!point_in_or_on_triangle(
                Vec3::new(1.0 + epsilon, 0.0, 0.0),
                triangle
            ));
            assert!(!point_in_or_on_triangle(
                Vec3::new(0.0, 1.0 + epsilon, 0.0),
                triangle
            ));
            assert!(!point_in_or_on_triangle(
                Vec3::new(-epsilon, -epsilon, 0.0),
                triangle
            ));
        }

        #[test]
        fn test_at_centroid_of_triangle() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // At the centroid of the triangle
            let centroid = (triangle.0 + triangle.1 + triangle.2) / 3.0;
            assert!(point_in_or_on_triangle(centroid, triangle));
        }
    }

    mod test_edge_quad_intersection {
        use super::*;

        #[test]
        fn test_edge_completely_inside_quad() {
            let edge = (Vec3::new(0.25, 0.25, 0.0), Vec3::new(0.75, 0.25, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_completely_outside_quad() {
            let edge = (Vec3::new(1.5, 1.5, 0.0), Vec3::new(2.0, 1.5, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_intersecting_quad_edge() {
            let edge = (Vec3::new(0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_intersecting_quad_vertex() {
            let edge = (Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.0, 0.0, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_intersecting_quad_not_touching_vertices_or_edges() {
            let edge = (Vec3::new(0.5, 0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_parallel_to_quad_outside() {
            let edge = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_coinciding_with_quad_edge() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_coinciding_with_quad_vertex() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_parallel_to_quad_inside() {
            let edge = (Vec3::new(0.25, 0.25, 0.0), Vec3::new(0.75, 0.25, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_coinciding_with_quad_face() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_intersecting_quad_at_multiple_points() {
            let edge = (Vec3::new(-0.5, 0.5, 0.0), Vec3::new(1.5, 0.5, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(edge_quad_intersection(edge, quad));
        }

        #[test]
        fn test_edge_very_close_to_quad_but_outside() {
            let edge = (Vec3::new(1.0 + 1e-6, 0.5, 0.0), Vec3::new(2.0, 0.5, 0.0));
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!edge_quad_intersection(edge, quad));
        }
    }

    mod test_point_in_quad {
        use super::*;

        #[test]
        fn test_point_inside_quad() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(point_in_quad(Vec3::new(0.5, 0.5, 0.0), quad));
        }

        #[test]
        fn test_point_on_quad_edge() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(point_in_quad(Vec3::new(0.5, 0.0, 0.0), quad));
            assert!(point_in_quad(Vec3::new(1.0, 0.5, 0.0), quad));
            assert!(point_in_quad(Vec3::new(0.5, 1.0, 0.0), quad));
            assert!(point_in_quad(Vec3::new(0.0, 0.5, 0.0), quad));
        }

        #[test]
        fn test_point_on_quad_vertex() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(point_in_quad(Vec3::new(0.0, 0.0, 0.0), quad));
            assert!(point_in_quad(Vec3::new(1.0, 0.0, 0.0), quad));
            assert!(point_in_quad(Vec3::new(1.0, 1.0, 0.0), quad));
            assert!(point_in_quad(Vec3::new(0.0, 1.0, 0.0), quad));
        }

        #[test]
        fn test_point_outside_quad() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!point_in_quad(Vec3::new(-0.5, 0.5, 0.0), quad));
            assert!(!point_in_quad(Vec3::new(1.5, 0.5, 0.0), quad));
            assert!(!point_in_quad(Vec3::new(0.5, -0.5, 0.0), quad));
            assert!(!point_in_quad(Vec3::new(0.5, 1.5, 0.0), quad));
        }

        #[test]
        fn test_point_in_non_planar_quad() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(point_in_quad(Vec3::new(0.5, 0.5, 0.25), quad));
        }

        #[test]
        fn test_point_outside_non_planar_quad() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!point_in_quad(Vec3::new(0.5, 0.5, 1.0), quad));
        }

        #[test]
        fn test_point_on_vertex() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(point_in_quad(Vec3::new(0.5, 0.0, 0.0), quad));
        }

        #[test]
        fn test_point_outside_vertex() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!point_in_quad(Vec3::new(-0.5, -0.5, 0.0), quad));
        }

        #[test]
        fn test_point_outside_edge_not_touching_vertex() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!point_in_quad(Vec3::new(0.5, -0.5, 0.0), quad));
        }

        #[test]
        fn test_point_far_from_quad() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(!point_in_quad(Vec3::new(10.0, 10.0, 10.0), quad));
        }

        #[test]
        fn test_point_on_diagonal() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            assert!(point_in_quad(Vec3::new(0.5, 0.5, 0.0), quad));
        }

        #[test]
        fn test_point_on_boundary_not_edge_or_vertex() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // Point on the boundary but not on the edges or vertices
            assert!(point_in_quad(Vec3::new(0.5, 0.5, 0.0), quad));
        }

        #[test]
        fn test_point_at_center_of_quad() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // Point exactly at the center of the quad
            assert!(point_in_quad(Vec3::new(0.5, 0.5, 0.0), quad));
        }

        #[test]
        fn test_point_very_close_to_quad_but_outside() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            // Point very close to the quad but outside (testing epsilon)
            assert!(!point_in_quad(Vec3::new(1.0 + 1e-4, 0.5, 0.0), quad));
        }

        #[test]
        fn test_point_very_close_to_quad_but_inside() {
            let quad = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            let point = Vec3::new(1.0 - 1e-6, 0.5, 0.0);
            assert!(point_in_quad(point, quad));
        }
    }

    mod test_triangle_edge_intersection {
        use super::*;

        #[test]
        fn test_triangle_completely_inside_edge() {
            let triangle = (
                Vec3::new(0.25, 0.25, 0.0),
                Vec3::new(0.75, 0.25, 0.0),
                Vec3::new(0.5, 0.75, 0.0),
            );
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_completely_outside_edge() {
            let triangle = (
                Vec3::new(1.5, 1.5, 0.0),
                Vec3::new(2.0, 1.5, 0.0),
                Vec3::new(1.75, 2.0, 0.0),
            );
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(!triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_intersecting_edge_vertex() {
            let triangle = (
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            );
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_intersecting_edge_not_touching_vertices() {
            let triangle = (
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(1.5, 0.0, 0.0),
            );
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_parallel_to_edge_outside() {
            let triangle = (
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.5, 1.0, 1.0),
            );
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(!triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_coinciding_with_edge() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 0.0, 0.0),
            );
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_edge_coinciding_with_edge() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
            );
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_edge_very_close_to_edge_but_not_intersecting() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
            let triangle = (
                Vec3::new(0.0, 0.0, 0.1),
                Vec3::new(1.0, 1.0, 0.1),
                Vec3::new(0.5, 0.5, 0.1),
            );
            assert!(!triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_edge_intersecting_at_multiple_points() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 0.0));
            let triangle = (
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(1.5, 0.5, 0.0),
                Vec3::new(0.5, 1.5, 0.0),
            );
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_edge_intersecting_at_single_point() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
            let triangle = (
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(1.5, 0.5, 0.0),
                Vec3::new(0.5, 1.5, 0.0),
            );
            assert!(triangle_edge_intersection(edge, triangle));
        }

        #[test]
        fn test_triangle_edge_parallel_to_edge_but_inside() {
            let edge = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 0.0));
            let triangle = (
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(1.5, 0.5, 0.0),
                Vec3::new(0.5, 1.5, 0.0),
            );
            assert!(triangle_edge_intersection(edge, triangle));
        }
    }

    mod test_point_on_line_segment {
        use super::*;

        #[test]
        fn test_point_on_segment() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(point_on_line_segment(Vec3::new(0.5, 0.5, 0.5), segment));
        }

        #[test]
        fn test_point_at_segment_start() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(point_on_line_segment(Vec3::new(0.0, 0.0, 0.0), segment));
        }

        #[test]
        fn test_point_at_segment_end() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(point_on_line_segment(Vec3::new(1.0, 1.0, 1.0), segment));
        }

        #[test]
        fn test_point_not_on_segment() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!point_on_line_segment(Vec3::new(0.5, 0.5, 0.0), segment));
        }

        #[test]
        fn test_point_beyond_segment_start() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!point_on_line_segment(Vec3::new(-0.1, -0.1, -0.1), segment));
        }

        #[test]
        fn test_point_beyond_segment_end() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!point_on_line_segment(Vec3::new(1.1, 1.1, 1.1), segment));
        }

        #[test]
        fn test_point_on_horizontal_segment() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(point_on_line_segment(Vec3::new(0.5, 0.0, 0.0), segment));
        }

        #[test]
        fn test_point_on_vertical_segment() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
            assert!(point_on_line_segment(Vec3::new(0.0, 0.5, 0.0), segment));
        }

        #[test]
        fn test_point_near_segment_within_epsilon() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(point_on_line_segment(
                Vec3::new(0.5, 0.5, 0.500001),
                segment
            ));
        }

        #[test]
        fn test_point_on_boundary() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(point_on_line_segment(Vec3::new(0.5, 0.5, 0.5), segment));
        }

        #[test]
        fn test_point_far_beyond_segment_end() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!point_on_line_segment(Vec3::new(1.5, 1.5, 1.5), segment));
        }

        #[test]
        fn test_point_far_beyond_segment_start() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!point_on_line_segment(Vec3::new(-1.5, -1.5, -1.5), segment));
        }

        #[test]
        fn test_point_at_midpoint_of_segment() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let point = Vec3::new(0.5, 0.5, 0.5);
            assert!(point_on_line_segment(point, segment));
        }

        #[test]
        fn test_point_very_close_to_segment_but_outside_epsilon() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let point = Vec3::new(0.5, 0.5, 0.5001); // Slightly outside epsilon
            assert!(!point_on_line_segment(point, segment));
        }

        #[test]
        fn test_point_exactly_at_epsilon_boundary() {
            let segment = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let point = Vec3::new(0.5, 0.5, 0.50001); // Exactly at epsilon boundary
            assert!(point_on_line_segment(point, segment));
        }
    }

    mod test_line_segment_overlap {
        use super::*;

        #[test]
        fn test_segments_overlap() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_touch_at_endpoint() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_do_not_overlap() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_parallel_no_overlap() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let seg2 = (Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_on_same_line_no_overlap() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let seg2 = (Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_overlap_partially() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
            let seg2 = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segment_contained_within_other() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
            let seg2 = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_barely_touching() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let seg2 = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_barely_missing() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let seg2 = (Vec3::new(1.000001, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_perpendicular_intersecting() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
            let seg2 = (Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_perpendicular_not_intersecting() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let seg2 = (Vec3::new(2.0, -1.0, 0.0), Vec3::new(2.0, 1.0, 0.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_skew() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 1.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_parallel_overlap_endpoint() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
            let seg2 = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_same_start_point() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_same_end_point() {
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(0.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_one_point() {
            let seg1 = (Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
            let seg2 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
            assert!(line_segment_overlap(seg1, seg2));
        }

        #[test]
        fn test_segments_parallel_very_close() {
            let epsilon = 1e-7;
            let seg1 = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let seg2 = (Vec3::new(0.0, epsilon, 0.0), Vec3::new(1.0, epsilon, 0.0));
            assert!(!line_segment_overlap(seg1, seg2));
        }
    }

    mod test_triangle_cube_intersection {
        use super::*;

        #[test]
        fn test_triangle_completely_inside_cube() {
            let triangle = (
                Vec3::new(0.25, 0.25, 0.25),
                Vec3::new(0.75, 0.25, 0.25),
                Vec3::new(0.25, 0.75, 0.25),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_completely_outside_cube() {
            let triangle = (
                Vec3::new(1.5, 1.5, 1.5),
                Vec3::new(2.5, 1.5, 1.5),
                Vec3::new(1.5, 2.5, 1.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_vertex_inside_cube() {
            let triangle = (
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(1.5, 0.5, 0.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_edge_intersects_cube() {
            let triangle = (
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(0.5, 1.5, 0.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_face_intersects_cube() {
            let triangle = (
                Vec3::new(0.5, 0.5, -0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(0.5, 1.5, 0.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_cube_above_triangle_no_intersection() {
            let triangle = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            let cube = (Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
            assert!(!triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_crosses_cube() {
            // Test case 1: Current triangle crossing the cube
            let triangle = (
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(1.5, 0.5, 0.5),
                Vec3::new(0.5, 2.0, 0.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));

            // Test case 2: Triangle completely outside and above the cube
            let triangle = (
                Vec3::new(0.5, 1.5, 1.5),
                Vec3::new(1.5, 1.5, 1.5),
                Vec3::new(1.0, 2.0, 1.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!triangle_cube_intersection(triangle, cube));

            // Test case 3: Triangle with one vertex inside the cube
            let triangle = (
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(2.0, 2.0, 2.0),
                Vec3::new(1.5, 1.5, 2.0),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));

            // Test case 4: Triangle parallel to one face of the cube but fully outside
            let triangle = (
                Vec3::new(1.5, 0.5, 0.5),
                Vec3::new(2.5, 0.5, 0.5),
                Vec3::new(2.0, 1.5, 0.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!triangle_cube_intersection(triangle, cube));

            // Test case 5: Triangle fully inside the cube
            let triangle = (
                Vec3::new(0.2, 0.2, 0.2),
                Vec3::new(0.8, 0.2, 0.2),
                Vec3::new(0.5, 0.8, 0.2),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_no_intersection_triangle_behind_cube() {
            let triangle = (
                Vec3::new(-1.5, -1.5, -1.5),
                Vec3::new(-1.0, -1.5, -1.5),
                Vec3::new(-1.5, -1.0, -1.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_parallel_to_cube_face_no_intersection() {
            let triangle = (
                Vec3::new(0.0, 0.0, 1.5),
                Vec3::new(1.0, 0.0, 1.5),
                Vec3::new(0.0, 1.0, 1.5),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_touching_cube_face() {
            let triangle = (
                Vec3::new(0.5, 0.5, 1.0),
                Vec3::new(0.75, 0.25, 1.0),
                Vec3::new(0.25, 0.75, 1.0),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(triangle_cube_intersection(triangle, cube));
        }

        #[test]
        fn test_triangle_overlapping_cube_face_no_intersection() {
            let triangle = (
                Vec3::new(0.5, 0.5, 2.0),
                Vec3::new(1.5, 0.5, 2.0),
                Vec3::new(0.5, 1.5, 2.0),
            );
            let cube = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
            assert!(!triangle_cube_intersection(triangle, cube));
        }
    }
}
