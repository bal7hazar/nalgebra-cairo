use fixed::Fixed;
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use nalgebra::{
    Isometry2, Isometry3, Quaternion, Translation2, Translation3, UnitComplex, UnitQuaternion,
    Vector2, Vector3,
};
use crate::black_box;
use crate::glam_isometry::*;
use super::{FRAC_1_SQRT_2_RAW, int, near};

fn h() -> Fixed {
    Fixed { raw: FRAC_1_SQRT_2_RAW }
}

/// Rotation by 90 degrees then translation by (3, 4), column-major.
fn rigid3() -> Mat3 {
    Mat3 {
        x_axis: Vec3 { x: int(0), y: int(1), z: int(0) },
        y_axis: Vec3 { x: int(-1), y: int(0), z: int(0) },
        z_axis: Vec3 { x: int(3), y: int(4), z: int(1) },
    }
}

/// Quarter turn about z then translation by (1, 2, 3), column-major.
fn rigid4() -> Mat4 {
    Mat4 {
        x_axis: Vec4 { x: int(0), y: int(1), z: int(0), w: int(0) },
        y_axis: Vec4 { x: int(-1), y: int(0), z: int(0), w: int(0) },
        z_axis: Vec4 { x: int(0), y: int(0), z: int(1), w: int(0) },
        w_axis: Vec4 { x: int(1), y: int(2), z: int(3), w: int(1) },
    }
}

#[test]
fn test_vec2_into_isometry2_is_a_translation() {
    let i: Isometry2<Fixed> = black_box(Vec2 { x: int(3), y: int(4) }).into();
    assert!(i.rotation == UnitComplex { re: int(1), im: int(0) });
    assert!(i.translation == Translation2 { vector: Vector2 { x: int(3), y: int(4) } });
}

#[test]
fn test_vec2_angle_into_isometry2() {
    let i: Isometry2<Fixed> = black_box((Vec2 { x: int(3), y: int(4) }, int(0))).into();
    assert!(i.rotation == UnitComplex { re: int(1), im: int(0) });
    assert!(i.translation == Translation2 { vector: Vector2 { x: int(3), y: int(4) } });
    // A quarter turn: (cos, sin) = (0, 1) within the rounding of pi / 2.
    let q: Isometry2<Fixed> = black_box((Vec2 { x: int(3), y: int(4) }, Fixed { raw: 6746518852 }))
        .into();
    assert!(near(q.rotation.re, int(0), 4) && near(q.rotation.im, int(1), 4));
}

#[test]
fn test_isometry2_into_vec2_angle() {
    let i = black_box(
        Isometry2 {
            rotation: UnitComplex { re: int(0), im: int(1) },
            translation: Translation2 { vector: Vector2 { x: int(3), y: int(4) } },
        },
    );
    let (t, angle): (Vec2, Fixed) = i.into();
    assert!(t == Vec2 { x: int(3), y: int(4) });
    // atan2(1, 0) = pi / 2 (about 12 ulp).
    assert!(near(angle, Fixed { raw: 6746518852 }, 16));
}

#[test]
fn test_isometry2_into_mat3() {
    let i = black_box(
        Isometry2 {
            rotation: UnitComplex { re: int(0), im: int(1) },
            translation: Translation2 { vector: Vector2 { x: int(3), y: int(4) } },
        },
    );
    let m: Mat3 = i.into();
    assert!(m == rigid3());
}

#[test]
fn test_mat3_try_into_isometry2() {
    let i: Isometry2<Fixed> = black_box(rigid3()).try_into().unwrap();
    assert!(i.rotation == UnitComplex { re: int(0), im: int(1) });
    assert!(i.translation == Translation2 { vector: Vector2 { x: int(3), y: int(4) } });
    let back: Mat3 = i.into();
    assert!(back == rigid3());
}

#[test]
fn test_mat3_try_into_isometry2_tolerance_is_100_ulp() {
    // A perturbation of `d` ulp on the diagonal makes the block orthogonal within about `2 d`.
    let mut ok = Mat3 {
        x_axis: Vec3 { x: Fixed { raw: int(1).raw + 30 }, y: int(0), z: int(0) },
        y_axis: Vec3 { x: int(0), y: int(1), z: int(0) },
        z_axis: Vec3 { x: int(0), y: int(0), z: int(1) },
    };
    let i: Option<Isometry2<Fixed>> = black_box(ok).try_into();
    assert!(i.is_some());
    ok.x_axis.x = Fixed { raw: int(1).raw + 100 };
    let i: Option<Isometry2<Fixed>> = black_box(ok).try_into();
    assert!(i.is_none());
}

#[test]
fn test_mat3_try_into_isometry2_rejects() {
    // A nonzero bottom row.
    let mut m = rigid3();
    m.x_axis.z = int(1);
    let i: Option<Isometry2<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    let mut m = rigid3();
    m.y_axis.z = Fixed { raw: 1 };
    let i: Option<Isometry2<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // The corner is not one.
    let mut m = rigid3();
    m.z_axis.z = int(2);
    let i: Option<Isometry2<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // A scaling.
    let mut m = rigid3();
    m.x_axis.y = int(2);
    m.y_axis.x = int(-2);
    let i: Option<Isometry2<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // A reflection (determinant -1).
    let mut m = rigid3();
    m.x_axis = Vec3 { x: int(1), y: int(0), z: int(0) };
    m.y_axis = Vec3 { x: int(0), y: int(-1), z: int(0) };
    let i: Option<Isometry2<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // A shear.
    let mut m = rigid3();
    m.x_axis = Vec3 { x: int(1), y: int(0), z: int(0) };
    m.y_axis = Vec3 { x: int(1), y: int(1), z: int(0) };
    let i: Option<Isometry2<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
}

#[test]
fn test_vec3_into_isometry3_is_a_translation() {
    let i: Isometry3<Fixed> = black_box(Vec3 { x: int(1), y: int(2), z: int(3) }).into();
    assert!(
        i
            .rotation == UnitQuaternion {
                quaternion: Quaternion { i: int(0), j: int(0), k: int(0), w: int(1) },
            },
    );
    assert!(i.translation == Translation3 { vector: Vector3 { x: int(1), y: int(2), z: int(3) } });
}

#[test]
fn test_quat_into_isometry3_is_a_normalized_rotation() {
    let i: Isometry3<Fixed> = black_box(Quat { x: int(0), y: int(0), z: int(0), w: int(2) }).into();
    assert!(i.rotation.quaternion == Quaternion { i: int(0), j: int(0), k: int(0), w: int(1) });
    assert!(i.translation == Translation3 { vector: Vector3 { x: int(0), y: int(0), z: int(0) } });
}

#[test]
fn test_vec3_quat_round_trip_isometry3() {
    let i: Isometry3<Fixed> = black_box(
        (
            Vec3 { x: int(1), y: int(2), z: int(3) },
            Quat { x: int(0), y: int(2), z: int(0), w: int(0) },
        ),
    )
        .into();
    // glam order (x, y, z, w) = (0, 2, 0, 0) normalizes to j = 1.
    assert!(i.rotation.quaternion == Quaternion { i: int(0), j: int(1), k: int(0), w: int(0) });
    assert!(i.translation.vector == Vector3 { x: int(1), y: int(2), z: int(3) });
    let (t, q): (Vec3, Quat) = i.into();
    assert!(t == Vec3 { x: int(1), y: int(2), z: int(3) });
    assert!(q == Quat { x: int(0), y: int(1), z: int(0), w: int(0) });
}

#[test]
fn test_isometry3_into_mat4() {
    let i = black_box(
        Isometry3 {
            rotation: UnitQuaternion {
                quaternion: Quaternion { i: int(0), j: int(0), k: h(), w: h() },
            },
            translation: Translation3 { vector: Vector3 { x: int(1), y: int(2), z: int(3) } },
        },
    );
    let m: Mat4 = i.into();
    let e = rigid4();
    assert!(near(m.x_axis.x, e.x_axis.x, 4) && near(m.x_axis.y, e.x_axis.y, 4));
    assert!(near(m.y_axis.x, e.y_axis.x, 4) && near(m.y_axis.y, e.y_axis.y, 4));
    assert!(near(m.z_axis.z, e.z_axis.z, 4) && near(m.x_axis.z, int(0), 4));
    assert!(m.w_axis == e.w_axis);
    assert!(m.x_axis.w == int(0) && m.y_axis.w == int(0) && m.z_axis.w == int(0));
}

#[test]
fn test_mat4_try_into_isometry3() {
    let i: Isometry3<Fixed> = black_box(rigid4()).try_into().unwrap();
    let q = i.rotation.quaternion;
    assert!(near(q.i, int(0), 4) && near(q.j, int(0), 4));
    assert!(near(q.k, h(), 4) && near(q.w, h(), 4));
    assert!(i.translation.vector == Vector3 { x: int(1), y: int(2), z: int(3) });
}

#[test]
fn test_mat4_try_into_isometry3_rejects() {
    let mut m = rigid4();
    m.x_axis.w = Fixed { raw: 1 };
    let i: Option<Isometry3<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    let mut m = rigid4();
    m.w_axis.w = int(2);
    let i: Option<Isometry3<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // A scaling of the z axis.
    let mut m = rigid4();
    m.z_axis.z = int(2);
    let i: Option<Isometry3<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // A reflection (determinant -1).
    let mut m = rigid4();
    m.z_axis.z = int(-1);
    let i: Option<Isometry3<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
    // A shear.
    let mut m = rigid4();
    m.z_axis.x = int(1);
    let i: Option<Isometry3<Fixed>> = black_box(m).try_into();
    assert!(i.is_none());
}
