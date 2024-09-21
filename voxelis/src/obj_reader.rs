use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use bevy::math::{IVec3, Vec3};

pub struct Obj {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<IVec3>,
    pub aabb: (Vec3, Vec3),
    pub size: Vec3,
}

impl Obj {
    pub fn parse(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;

        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for line in reader.lines() {
            let line = line.unwrap();
            let tokens: Vec<&str> = line.split_whitespace().collect();
            match tokens[0] {
                "v" => {
                    let x: f32 = tokens[1].parse().unwrap();
                    let y: f32 = tokens[2].parse().unwrap();
                    let z: f32 = tokens[3].parse().unwrap();

                    let vertex = Vec3::new(x, y, z);

                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    min_z = min_z.min(z);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    max_z = max_z.max(z);

                    vertices.push(vertex);
                }
                "f" => {
                    let v1: i32 = tokens[1].parse().unwrap();
                    let v2: i32 = tokens[2].parse().unwrap();
                    let v3: i32 = tokens[3].parse().unwrap();

                    let face = IVec3::new(v1, v2, v3);

                    faces.push(face);
                }
                _ => {}
            }
        }

        let aabb = (
            Vec3::new(min_x, min_y, min_z),
            Vec3::new(max_x, max_y, max_z),
        );
        let size = Vec3::new(max_x - min_x, max_y - min_y, max_z - min_z);

        println!("Parsed obj file: {}", path);
        println!("Vertices: {}", vertices.len());
        println!("Faces: {}", faces.len());
        println!("Size: {:?}", size);
        println!("AABB: {:?}, {:?}", aabb.0, aabb.1);

        Self {
            vertices,
            faces,
            aabb,
            size,
        }
    }
}
