use fixed::Fixed;
use glam::{BVec2, BVec3, BVec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};
use nalgebra::{Point2, Point3, Point4};
use crate::black_box;
use crate::glam_point::*;
use super::int;

#[test]
fn test_vec_point_order_and_round_trip() {
    let p2: Point2<Fixed> = black_box(Vec2 { x: int(1), y: int(-2) }).into();
    assert!(p2 == Point2 { x: int(1), y: int(-2) });
    let p3: Point3<Fixed> = black_box(Vec3 { x: int(1), y: int(-2), z: int(3) }).into();
    assert!(p3 == Point3 { x: int(1), y: int(-2), z: int(3) });
    let p4: Point4<Fixed> = black_box(Vec4 { x: int(1), y: int(-2), z: int(3), w: int(-4) }).into();
    assert!(p4 == Point4 { x: int(1), y: int(-2), z: int(3), w: int(-4) });
    let g2: Vec2 = p2.into();
    let g3: Vec3 = p3.into();
    let g4: Vec4 = p4.into();
    assert!(g2 == Vec2 { x: int(1), y: int(-2) });
    assert!(g3 == Vec3 { x: int(1), y: int(-2), z: int(3) });
    assert!(g4 == Vec4 { x: int(1), y: int(-2), z: int(3), w: int(-4) });
}

#[test]
fn test_ivec_point_order_and_round_trip() {
    let p2: Point2<i32> = black_box(IVec2 { x: -1, y: 2 }).into();
    let p3: Point3<i32> = black_box(IVec3 { x: -1, y: 2, z: -3 }).into();
    let p4: Point4<i32> = black_box(IVec4 { x: -1, y: 2, z: -3, w: 4 }).into();
    assert!(p2 == Point2 { x: -1, y: 2 });
    assert!(p3 == Point3 { x: -1, y: 2, z: -3 });
    assert!(p4 == Point4 { x: -1, y: 2, z: -3, w: 4 });
    let g2: IVec2 = p2.into();
    let g3: IVec3 = p3.into();
    let g4: IVec4 = p4.into();
    assert!(g2 == IVec2 { x: -1, y: 2 });
    assert!(g3 == IVec3 { x: -1, y: 2, z: -3 });
    assert!(g4 == IVec4 { x: -1, y: 2, z: -3, w: 4 });
}

#[test]
fn test_uvec_point_order_and_round_trip() {
    let p2: Point2<u32> = black_box(UVec2 { x: 1, y: 2 }).into();
    let p3: Point3<u32> = black_box(UVec3 { x: 1, y: 2, z: 3 }).into();
    let p4: Point4<u32> = black_box(UVec4 { x: 1, y: 2, z: 3, w: 4 }).into();
    assert!(p2 == Point2 { x: 1, y: 2 });
    assert!(p3 == Point3 { x: 1, y: 2, z: 3 });
    assert!(p4 == Point4 { x: 1, y: 2, z: 3, w: 4 });
    let g2: UVec2 = p2.into();
    let g3: UVec3 = p3.into();
    let g4: UVec4 = p4.into();
    assert!(g2 == UVec2 { x: 1, y: 2 });
    assert!(g3 == UVec3 { x: 1, y: 2, z: 3 });
    assert!(g4 == UVec4 { x: 1, y: 2, z: 3, w: 4 });
}

#[test]
fn test_bvec_point_order_and_round_trip() {
    let p2: Point2<bool> = black_box(BVec2 { x: true, y: false }).into();
    let p3: Point3<bool> = black_box(BVec3 { x: false, y: true, z: false }).into();
    let p4: Point4<bool> = black_box(BVec4 { x: false, y: false, z: true, w: true }).into();
    assert!(p2 == Point2 { x: true, y: false });
    assert!(p3 == Point3 { x: false, y: true, z: false });
    assert!(p4 == Point4 { x: false, y: false, z: true, w: true });
    let g2: BVec2 = p2.into();
    let g3: BVec3 = p3.into();
    let g4: BVec4 = p4.into();
    assert!(g2 == BVec2 { x: true, y: false });
    assert!(g3 == BVec3 { x: false, y: true, z: false });
    assert!(g4 == BVec4 { x: false, y: false, z: true, w: true });
}
