use fixed::Fixed;
use glam::{Vec2, Vec3, Vec4};
use nalgebra::{Translation2, Translation3, Translation4, Vector2, Vector3, Vector4};
use crate::black_box;
use crate::glam_translation::*;
use super::int;

#[test]
fn test_vec_translation_order() {
    let t2: Translation2<Fixed> = black_box(Vec2 { x: int(1), y: int(2) }).into();
    assert!(t2.vector == Vector2 { x: int(1), y: int(2) });
    let t3: Translation3<Fixed> = black_box(Vec3 { x: int(1), y: int(2), z: int(3) }).into();
    assert!(t3.vector == Vector3 { x: int(1), y: int(2), z: int(3) });
    let t4: Translation4<Fixed> = black_box(Vec4 { x: int(1), y: int(2), z: int(3), w: int(4) })
        .into();
    assert!(t4.vector == Vector4 { x: int(1), y: int(2), z: int(3), w: int(4) });
}

#[test]
fn test_translation_into_vec_round_trip() {
    let t2 = black_box(Translation2 { vector: Vector2 { x: int(-1), y: int(2) } });
    let g2: Vec2 = t2.into();
    assert!(g2 == Vec2 { x: int(-1), y: int(2) });
    let back: Translation2<Fixed> = g2.into();
    assert!(back == t2);
    let t3 = black_box(Translation3 { vector: Vector3 { x: int(-1), y: int(2), z: int(-3) } });
    let g3: Vec3 = t3.into();
    assert!(g3 == Vec3 { x: int(-1), y: int(2), z: int(-3) });
    let back: Translation3<Fixed> = g3.into();
    assert!(back == t3);
    let t4 = black_box(
        Translation4 { vector: Vector4 { x: int(1), y: int(-2), z: int(3), w: int(-4) } },
    );
    let g4: Vec4 = t4.into();
    assert!(g4 == Vec4 { x: int(1), y: int(-2), z: int(3), w: int(-4) });
    let back: Translation4<Fixed> = g4.into();
    assert!(back == t4);
}
