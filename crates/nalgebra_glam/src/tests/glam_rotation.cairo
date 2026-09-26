use fixed::Fixed;
use glam::{Mat2, Quat, Vec2};
use nalgebra::{Matrix2, Matrix3, Rotation2, Rotation3};
use crate::black_box;
use crate::glam_rotation::*;
use super::{FRAC_1_SQRT_2_RAW, int, near};

#[test]
fn test_rotation2_into_mat2_keeps_the_columns() {
    let r = black_box(
        Rotation2 { matrix: Matrix2 { m11: int(0), m21: int(1), m12: int(-1), m22: int(0) } },
    );
    let m: Mat2 = r.into();
    assert!(
        m == Mat2 { x_axis: Vec2 { x: int(0), y: int(1) }, y_axis: Vec2 { x: int(-1), y: int(0) } },
    );
}

#[test]
fn test_mat2_into_rotation2_normalizes_the_first_column() {
    // A scaled 90-degree rotation: (0, 3) / 3 = (0, 1), matrix [[re, -im], [im, re]].
    let r: Rotation2<Fixed> = black_box(
        Mat2 { x_axis: Vec2 { x: int(0), y: int(3) }, y_axis: Vec2 { x: int(-3), y: int(0) } },
    )
        .into();
    assert!(r.matrix == Matrix2 { m11: int(0), m21: int(1), m12: int(-1), m22: int(0) });
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_zero_mat2_into_rotation2_panics() {
    let _r: Rotation2<Fixed> = black_box(
        Mat2 { x_axis: Vec2 { x: int(0), y: int(0) }, y_axis: Vec2 { x: int(0), y: int(0) } },
    )
        .into();
}

fn quarter_turn_about_z() -> Quat {
    Quat {
        x: int(0),
        y: int(0),
        z: Fixed { raw: FRAC_1_SQRT_2_RAW },
        w: Fixed { raw: FRAC_1_SQRT_2_RAW },
    }
}

#[test]
fn test_quat_into_rotation3_quarter_turn() {
    let r: Rotation3<Fixed> = black_box(quarter_turn_about_z()).into();
    let m: Matrix3<Fixed> = r.matrix;
    // [[0, -1, 0], [1, 0, 0], [0, 0, 1]] within the rounding of the half angle.
    assert!(near(m.m11, int(0), 4) && near(m.m22, int(0), 4) && near(m.m33, int(1), 4));
    assert!(near(m.m21, int(1), 4) && near(m.m12, int(-1), 4));
    assert!(near(m.m31, int(0), 4) && near(m.m13, int(0), 4));
    assert!(near(m.m32, int(0), 4) && near(m.m23, int(0), 4));
}

#[test]
fn test_quat_into_rotation3_normalizes() {
    // (0, 0, 0, 2) is the identity rotation, exactly.
    let r: Rotation3<Fixed> = black_box(Quat { x: int(0), y: int(0), z: int(0), w: int(2) }).into();
    assert!(
        r
            .matrix == Matrix3 {
                m11: int(1),
                m21: int(0),
                m31: int(0),
                m12: int(0),
                m22: int(1),
                m32: int(0),
                m13: int(0),
                m23: int(0),
                m33: int(1),
            },
    );
}

#[test]
fn test_rotation3_into_quat_round_trip() {
    let r: Rotation3<Fixed> = black_box(quarter_turn_about_z()).into();
    let q: Quat = black_box(r).into();
    assert!(near(q.x, int(0), 4) && near(q.y, int(0), 4));
    assert!(near(q.z, Fixed { raw: FRAC_1_SQRT_2_RAW }, 4));
    assert!(near(q.w, Fixed { raw: FRAC_1_SQRT_2_RAW }, 4));
}

#[test]
fn test_rotation3_half_turn_into_quat_order() {
    // Half a turn about x: the quaternion is (x, y, z, w) = (±1, 0, 0, 0) (the x slot, not w).
    let r = black_box(
        Rotation3 {
            matrix: Matrix3 {
                m11: int(1),
                m21: int(0),
                m31: int(0),
                m12: int(0),
                m22: int(-1),
                m32: int(0),
                m13: int(0),
                m23: int(0),
                m33: int(-1),
            },
        },
    );
    let q: Quat = r.into();
    assert!(q.x.raw == int(1).raw || q.x.raw == int(-1).raw);
    assert!(q.y == int(0) && q.z == int(0) && q.w == int(0));
}
