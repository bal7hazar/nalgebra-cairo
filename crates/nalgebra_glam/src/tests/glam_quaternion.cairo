use fixed::Fixed;
use glam::Quat;
use nalgebra::{Quaternion, UnitQuaternion};
use crate::black_box;
use crate::glam_quaternion::*;
use super::{int, near};

#[test]
fn test_quat_into_quaternion_order() {
    // glam: (x, y, z, w); nalgebra: `Quaternion::new(w, i, j, k)`, fields `i, j, k, w`.
    let q: Quaternion<Fixed> = black_box(Quat { x: int(1), y: int(2), z: int(3), w: int(4) })
        .into();
    assert!(q == Quaternion { i: int(1), j: int(2), k: int(3), w: int(4) });
}

#[test]
fn test_quaternion_into_quat_round_trip() {
    let q = black_box(Quaternion { i: int(1), j: int(-2), k: int(3), w: int(-4) });
    let g: Quat = q.into();
    assert!(g == Quat { x: int(1), y: int(-2), z: int(3), w: int(-4) });
    let back: Quaternion<Fixed> = g.into();
    assert!(back == q);
}

#[test]
fn test_quat_into_unit_quaternion_normalizes() {
    // (0, 0, 0, 2) -> the identity, exactly.
    let u: UnitQuaternion<Fixed> = black_box(Quat { x: int(0), y: int(0), z: int(0), w: int(2) })
        .into();
    assert!(u.quaternion == Quaternion { i: int(0), j: int(0), k: int(0), w: int(1) });
    // (1, 2, 2, 4) / 5: one correctly rounded division per component, in glam's order.
    let u: UnitQuaternion<Fixed> = black_box(Quat { x: int(1), y: int(2), z: int(2), w: int(4) })
        .into();
    assert!(near(u.quaternion.i, Fixed { raw: 858993459 }, 1));
    assert!(near(u.quaternion.j, Fixed { raw: 1717986918 }, 1));
    assert!(near(u.quaternion.k, Fixed { raw: 1717986918 }, 1));
    assert!(near(u.quaternion.w, Fixed { raw: 3435973837 }, 1));
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_zero_quat_into_unit_quaternion_panics() {
    let _u: UnitQuaternion<Fixed> = black_box(Quat { x: int(0), y: int(0), z: int(0), w: int(0) })
        .into();
}

#[test]
fn test_unit_quaternion_into_quat_order() {
    let u = black_box(
        UnitQuaternion { quaternion: Quaternion { i: int(1), j: int(2), k: int(3), w: int(4) } },
    );
    let g: Quat = u.into();
    assert!(g == Quat { x: int(1), y: int(2), z: int(3), w: int(4) });
}
