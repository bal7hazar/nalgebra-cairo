//! Gas benchmarks (steps criterion, WP 8.6-P19): the main conversions against a baseline that
//! builds the same operands and asserts a component, and the alternatives that lost (upstream's
//! literal formulations) against the formulation of the library. Operands go through `black_box`
//! in every variant of a group, the baseline included, so `net` is the conversion alone.

use fixed::{Fixed, ONE, ONE_RAW, ZERO};
use glam::{Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use nalgebra::{
    Isometry2, Isometry2AngleTrait, Isometry3, Matrix3, Matrix3Trait, Quaternion, Rotation3,
    Similarity3, Translation2, Translation3, UnitComplex, UnitQuaternion, UnitQuaternionTrait,
    UnitVector3, Vector2, Vector3, Vector3Trait,
};
use crate::black_box;
use crate::glam_isometry::*;
use crate::glam_matrix::*;
use crate::glam_quaternion::*;
use crate::glam_rotation::*;
use crate::glam_similarity::*;

fn int(v: i64) -> Fixed {
    Fixed { raw: v * ONE_RAW }
}

fn vec3() -> Vec3 {
    black_box(Vec3 { x: int(1), y: int(2), z: int(3) })
}

fn vector3() -> Vector3<Fixed> {
    black_box(Vector3 { x: int(1), y: int(2), z: int(3) })
}

fn mat4() -> Mat4 {
    black_box(
        Mat4 {
            x_axis: Vec4 { x: int(11), y: int(21), z: int(31), w: int(41) },
            y_axis: Vec4 { x: int(12), y: int(22), z: int(32), w: int(42) },
            z_axis: Vec4 { x: int(13), y: int(23), z: int(33), w: int(43) },
            w_axis: Vec4 { x: int(14), y: int(24), z: int(34), w: int(44) },
        },
    )
}

fn quat() -> Quat {
    black_box(Quat { x: int(1), y: int(2), z: int(2), w: int(4) })
}

/// Quarter turn about z (the sine and cosine of `π / 4`).
fn turn_z() -> Quat {
    black_box(Quat { x: ZERO, y: ZERO, z: Fixed { raw: 3037000500 }, w: Fixed { raw: 3037000500 } })
}

fn unit_quaternion() -> UnitQuaternion<Fixed> {
    black_box(UnitQuaternion { quaternion: Quaternion { i: ZERO, j: ZERO, k: ONE, w: ZERO } })
}

fn isometry3() -> Isometry3<Fixed> {
    black_box(
        Isometry3 {
            rotation: UnitQuaternion {
                quaternion: Quaternion {
                    i: ZERO, j: ZERO, k: Fixed { raw: 3037000500 }, w: Fixed { raw: 3037000500 },
                },
            },
            translation: Translation3 { vector: Vector3 { x: int(1), y: int(2), z: int(3) } },
        },
    )
}

fn isometry2() -> Isometry2<Fixed> {
    black_box(
        Isometry2 {
            rotation: UnitComplex { re: ZERO, im: ONE },
            translation: Translation2 { vector: Vector2 { x: int(3), y: int(4) } },
        },
    )
}

/// Quarter turn about z then translation by (1, 2, 3).
fn rigid4() -> Mat4 {
    black_box(
        Mat4 {
            x_axis: Vec4 { x: ZERO, y: ONE, z: ZERO, w: ZERO },
            y_axis: Vec4 { x: -ONE, y: ZERO, z: ZERO, w: ZERO },
            z_axis: Vec4 { x: ZERO, y: ZERO, z: ONE, w: ZERO },
            w_axis: Vec4 { x: int(1), y: int(2), z: int(3), w: ONE },
        },
    )
}

/// Quarter turn then translation by (3, 4).
fn rigid3() -> Mat3 {
    black_box(
        Mat3 {
            x_axis: Vec3 { x: ZERO, y: ONE, z: ZERO },
            y_axis: Vec3 { x: -ONE, y: ZERO, z: ZERO },
            z_axis: Vec3 { x: int(3), y: int(4), z: ONE },
        },
    )
}

/// Quarter turn, scaling by 2, translation by (1, 2, 3).
fn sim4() -> Mat4 {
    black_box(
        Mat4 {
            x_axis: Vec4 { x: ZERO, y: int(2), z: ZERO, w: ZERO },
            y_axis: Vec4 { x: -int(2), y: ZERO, z: ZERO, w: ZERO },
            z_axis: Vec4 { x: ZERO, y: ZERO, z: int(2), w: ZERO },
            w_axis: Vec4 { x: int(1), y: int(2), z: int(3), w: ONE },
        },
    )
}

/// Quarter turn, scaling by 3, translation by (3, 4).
fn sim3() -> Mat3 {
    black_box(
        Mat3 {
            x_axis: Vec3 { x: ZERO, y: int(3), z: ZERO },
            y_axis: Vec3 { x: -int(3), y: ZERO, z: ZERO },
            z_axis: Vec3 { x: int(3), y: int(4), z: ONE },
        },
    )
}

// --- vectors and matrices (moves only) ----------------------------------------------------------

#[test]
#[inline(never)]
fn bench_vec3_to_vector3__baseline() {
    let v = vec3();
    assert!(v.z == int(3));
}

#[test]
#[inline(never)]
fn bench_vec3_to_vector3__into() {
    let v = vec3();
    let r: Vector3<Fixed> = v.into();
    assert!(r.z == int(3));
}

#[test]
#[inline(never)]
fn bench_vector3_to_vec3__baseline() {
    let v = vector3();
    assert!(v.z == int(3));
}

#[test]
#[inline(never)]
fn bench_vector3_to_vec3__into() {
    let v = vector3();
    let r: Vec3 = v.into();
    assert!(r.z == int(3));
}

#[test]
#[inline(never)]
fn bench_mat4_to_matrix4__baseline() {
    let m = mat4();
    assert!(m.w_axis.w == int(44));
}

#[test]
#[inline(never)]
fn bench_mat4_to_matrix4__into() {
    let m = mat4();
    let r: nalgebra::Matrix4<Fixed> = m.into();
    assert!(r.m44 == int(44));
}

#[test]
#[inline(never)]
fn bench_matrix4_to_mat4__baseline() {
    let n: nalgebra::Matrix4<Fixed> = mat4().into();
    let n = black_box(n);
    assert!(n.m44 == int(44));
}

#[test]
#[inline(never)]
fn bench_matrix4_to_mat4__into() {
    let m = mat4();
    let n: nalgebra::Matrix4<Fixed> = m.into();
    let n = black_box(n);
    let r: Mat4 = n.into();
    assert!(r.w_axis.w == int(44));
}

// --- normalizing conversions
// ----------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_vec3_to_unit_vector3__baseline() {
    let v = vec3();
    assert!(v.z == int(3));
}

#[test]
#[inline(never)]
fn bench_vec3_to_unit_vector3__try_into() {
    let v = vec3();
    let r: Option<UnitVector3<Fixed>> = v.try_into();
    assert!(r.is_some());
}

#[test]
#[inline(never)]
fn bench_quat_to_unit_quaternion__baseline() {
    let q = quat();
    assert!(q.w == int(4));
}

#[test]
#[inline(never)]
fn bench_quat_to_unit_quaternion__into() {
    let q = quat();
    let r: UnitQuaternion<Fixed> = q.into();
    assert!(r.quaternion.w.raw > 0);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_quat__baseline() {
    let q = unit_quaternion();
    assert!(q.quaternion.k == ONE);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_quat__into() {
    let q = unit_quaternion();
    let r: Quat = q.into();
    assert!(r.z == ONE);
}

#[test]
#[inline(never)]
fn bench_mat2_to_rotation2__baseline() {
    let m = black_box(
        Mat2 { x_axis: Vec2 { x: ZERO, y: int(3) }, y_axis: Vec2 { x: -int(3), y: ZERO } },
    );
    assert!(m.x_axis.y == int(3));
}

#[test]
#[inline(never)]
fn bench_mat2_to_rotation2__into() {
    let m = black_box(
        Mat2 { x_axis: Vec2 { x: ZERO, y: int(3) }, y_axis: Vec2 { x: -int(3), y: ZERO } },
    );
    let r: nalgebra::Rotation2<Fixed> = m.into();
    assert!(r.matrix.m21 == ONE);
}

// --- rotations
// --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_quat_to_rotation3__baseline() {
    let q = turn_z();
    assert!(q.w.raw > 0);
}

#[test]
#[inline(never)]
fn bench_quat_to_rotation3__into() {
    let q = turn_z();
    let r: Rotation3<Fixed> = q.into();
    assert!(r.matrix.m33.raw > 0);
}

#[test]
#[inline(never)]
fn bench_rotation3_to_quat__baseline() {
    let q: Rotation3<Fixed> = turn_z().into();
    let q = black_box(q);
    assert!(q.matrix.m33.raw > 0);
}

#[test]
#[inline(never)]
fn bench_rotation3_to_quat__into() {
    let r: Rotation3<Fixed> = turn_z().into();
    let r = black_box(r);
    let q: Quat = r.into();
    assert!(q.w.raw > 0);
}

// --- isometries and similarities -> matrices, tuples
// -----------------------------------------------

#[test]
#[inline(never)]
fn bench_isometry3_to_mat4__baseline() {
    let i = isometry3();
    assert!(i.translation.vector.z == int(3));
}

#[test]
#[inline(never)]
fn bench_isometry3_to_mat4__into() {
    let i = isometry3();
    let m: Mat4 = i.into();
    assert!(m.w_axis.z == int(3));
}

#[test]
#[inline(never)]
fn bench_isometry2_to_mat3__baseline() {
    let i = isometry2();
    assert!(i.translation.vector.y == int(4));
}

#[test]
#[inline(never)]
fn bench_isometry2_to_mat3__into() {
    let i = isometry2();
    let m: Mat3 = i.into();
    assert!(m.z_axis.y == int(4));
}

#[test]
#[inline(never)]
fn bench_similarity3_to_mat4__baseline() {
    let s = black_box(Similarity3 { isometry: isometry3(), scaling: int(2) });
    assert!(s.scaling == int(2));
}

#[test]
#[inline(never)]
fn bench_similarity3_to_mat4__into() {
    let s = black_box(Similarity3 { isometry: isometry3(), scaling: int(2) });
    let m: Mat4 = s.into();
    assert!(m.w_axis.z == int(3));
}

#[test]
#[inline(never)]
fn bench_isometry3_to_vec3_quat__baseline() {
    let i = isometry3();
    assert!(i.translation.vector.z == int(3));
}

#[test]
#[inline(never)]
fn bench_isometry3_to_vec3_quat__into() {
    let i = isometry3();
    let (t, _q): (Vec3, Quat) = i.into();
    assert!(t.z == int(3));
}

#[test]
#[inline(never)]
fn bench_vec3_quat_to_isometry3__baseline() {
    let v = vec3();
    let q = turn_z();
    assert!(v.z == int(3) && q.w.raw > 0);
}

#[test]
#[inline(never)]
fn bench_vec3_quat_to_isometry3__into() {
    let v = vec3();
    let q = turn_z();
    let i: Isometry3<Fixed> = (v, q).into();
    assert!(i.translation.vector.z == int(3));
}

#[test]
#[inline(never)]
fn bench_isometry2_to_vec2_angle__baseline() {
    let i = isometry2();
    assert!(i.translation.vector.y == int(4));
}

#[test]
#[inline(never)]
fn bench_isometry2_to_vec2_angle__into() {
    let i = isometry2();
    let (_t, a): (Vec2, Fixed) = i.into();
    assert!(a.raw > 0);
}

#[test]
#[inline(never)]
fn bench_vec2_angle_to_isometry2__baseline() {
    let v = black_box(Vec2 { x: int(3), y: int(4) });
    let a = black_box(Fixed { raw: 6746518852 });
    assert!(v.y == int(4) && a.raw > 0);
}

#[test]
#[inline(never)]
fn bench_vec2_angle_to_isometry2__into() {
    let v = black_box(Vec2 { x: int(3), y: int(4) });
    let a = black_box(Fixed { raw: 6746518852 });
    let i: Isometry2<Fixed> = (v, a).into();
    assert!(i.translation.vector.y == int(4));
}

/// Upstream's `Isometry2::new(tra, 0.0)`: a `sin_cos` of zero (the loser).
fn vec2_to_isometry2_angle_zero(v: Vec2) -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(Vector2 { x: v.x, y: v.y }, ZERO)
}

#[test]
#[inline(never)]
fn bench_vec2_to_isometry2__baseline() {
    let v = black_box(Vec2 { x: int(3), y: int(4) });
    assert!(v.y == int(4));
}

#[test]
#[inline(never)]
fn bench_vec2_to_isometry2__identity() {
    let v = black_box(Vec2 { x: int(3), y: int(4) });
    let i: Isometry2<Fixed> = v.into();
    assert!(i.rotation.re == ONE);
}

#[test]
#[inline(never)]
fn bench_vec2_to_isometry2__angle_zero() {
    let v = black_box(Vec2 { x: int(3), y: int(4) });
    let i = vec2_to_isometry2_angle_zero(v);
    assert!(i.rotation.re == ONE);
}

// --- checked conversions from matrices
// ------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_mat4_to_isometry3__baseline() {
    let m = rigid4();
    assert!(m.w_axis.w == ONE);
}

#[test]
#[inline(never)]
fn bench_mat4_to_isometry3__try_into() {
    let m = rigid4();
    let i: Option<Isometry3<Fixed>> = m.try_into();
    assert!(i.is_some());
}

#[test]
#[inline(never)]
fn bench_mat3_to_isometry2__baseline() {
    let m = rigid3();
    assert!(m.z_axis.z == ONE);
}

#[test]
#[inline(never)]
fn bench_mat3_to_isometry2__try_into() {
    let m = rigid3();
    let i: Option<Isometry2<Fixed>> = m.try_into();
    assert!(i.is_some());
}

/// Upstream's order of the checks (`is_in_subset`): the orthogonality of the linear block first,
/// the bottom row after, everything in one function (a loser: Cairo charges the most expensive
/// path of the straight-line code of a function, so the rejected matrix pays the orthogonality
/// test and the rotation extraction).
fn mat4_to_isometry3_orthogonal_first(m: Mat4) -> Option<Isometry3<Fixed>> {
    let (x, y, z, w) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis);
    let rot = Matrix3 {
        m11: x.x, m21: x.y, m31: x.z, m12: y.x, m22: y.y, m32: y.z, m13: z.x, m23: z.y, m33: z.z,
    };
    if !Matrix3Trait::is_special_orthogonal(rot, 100) {
        return Option::None;
    }
    if x.w != ZERO || y.w != ZERO || z.w != ZERO || w.w != ONE {
        return Option::None;
    }
    Option::Some(
        Isometry3 {
            rotation: UnitQuaternionTrait::from_rotation_matrix(Rotation3 { matrix: rot }),
            translation: Translation3 { vector: Vector3 { x: w.x, y: w.y, z: w.z } },
        },
    )
}

/// The library's checks in one function, bottom row first (a loser: same cost as the other
/// single-function variant, the early `return` does not save the code below it).
fn mat4_to_isometry3_single_function(m: Mat4) -> Option<Isometry3<Fixed>> {
    let (x, y, z, w) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis);
    if x.w != ZERO || y.w != ZERO || z.w != ZERO || w.w != ONE {
        return Option::None;
    }
    let rot = Matrix3 {
        m11: x.x, m21: x.y, m31: x.z, m12: y.x, m22: y.y, m32: y.z, m13: z.x, m23: z.y, m33: z.z,
    };
    if !Matrix3Trait::is_special_orthogonal(rot, 100) {
        return Option::None;
    }
    Option::Some(
        Isometry3 {
            rotation: UnitQuaternionTrait::from_rotation_matrix(Rotation3 { matrix: rot }),
            translation: Translation3 { vector: Vector3 { x: w.x, y: w.y, z: w.z } },
        },
    )
}

/// A matrix with a bad bottom row (a projection): rejected by the cheap comparison.
fn bad_bottom4() -> Mat4 {
    let mut m = rigid4();
    m.y_axis.w = ONE;
    black_box(m)
}

#[test]
#[inline(never)]
fn bench_mat4_to_isometry3_reject__baseline() {
    let m = bad_bottom4();
    assert!(m.y_axis.w == ONE);
}

#[test]
#[inline(never)]
fn bench_mat4_to_isometry3_reject__bottom_first() {
    let m = bad_bottom4();
    let i: Option<Isometry3<Fixed>> = m.try_into();
    assert!(i.is_none());
}

#[test]
#[inline(never)]
fn bench_mat4_to_isometry3_reject__single_function() {
    let m = bad_bottom4();
    let i = mat4_to_isometry3_single_function(m);
    assert!(i.is_none());
}

#[test]
#[inline(never)]
fn bench_mat4_to_isometry3_reject__orthogonal_first() {
    let m = bad_bottom4();
    let i = mat4_to_isometry3_orthogonal_first(m);
    assert!(i.is_none());
}

#[test]
#[inline(never)]
fn bench_mat3_to_similarity2__baseline() {
    let m = sim3();
    assert!(m.z_axis.z == ONE);
}

#[test]
#[inline(never)]
fn bench_mat3_to_similarity2__try_into() {
    let m = sim3();
    let s: Option<nalgebra::Similarity2<Fixed>> = m.try_into();
    assert!(s.is_some());
}

/// Upstream's two passes (`is_in_subset` normalizes the columns to test them, then
/// `from_superset_unchecked` normalizes them again): the loser, one pass is the library.
fn mat4_to_similarity3_two_pass(m: Mat4) -> Option<Similarity3<Fixed>> {
    let (x, y, z, w) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis);
    let mut a = Vector3 { x: x.x, y: x.y, z: x.z };
    let mut b = Vector3 { x: y.x, y: y.y, z: y.z };
    let mut c = Vector3 { x: z.x, y: z.y, z: z.z };
    if Vector3Trait::try_normalize_mut(ref a, ZERO).is_none()
        || Vector3Trait::try_normalize_mut(ref b, ZERO).is_none()
        || Vector3Trait::try_normalize_mut(ref c, ZERO).is_none() {
        return Option::None;
    }
    if x.w != ZERO || y.w != ZERO || z.w != ZERO || w.w != ONE {
        return Option::None;
    }
    let mut a = Vector3 { x: x.x, y: x.y, z: x.z };
    let mut b = Vector3 { x: y.x, y: y.y, z: y.z };
    let mut c = Vector3 { x: z.x, y: z.y, z: z.z };
    let na = Vector3Trait::normalize_mut(ref a);
    let nb = Vector3Trait::normalize_mut(ref b);
    let nc = Vector3Trait::normalize_mut(ref c);
    let mut m = Matrix3 {
        m11: a.x, m21: a.y, m31: a.z, m12: b.x, m22: b.y, m32: b.z, m13: c.x, m23: c.y, m33: c.z,
    };
    let mut scaling = (na + nb + nc) / Fixed { raw: 0x300000000 };
    if Matrix3Trait::determinant(m) < ZERO {
        m = -m;
        scaling = -scaling;
    }
    Option::Some(
        Similarity3 {
            isometry: Isometry3 {
                rotation: UnitQuaternionTrait::from_rotation_matrix(Rotation3 { matrix: m }),
                translation: Translation3 { vector: Vector3 { x: w.x, y: w.y, z: w.z } },
            },
            scaling,
        },
    )
}

#[test]
#[inline(never)]
fn bench_mat4_to_similarity3__baseline() {
    let m = sim4();
    assert!(m.w_axis.w == ONE);
}

#[test]
#[inline(never)]
fn bench_mat4_to_similarity3__one_pass() {
    let m = sim4();
    let s: Option<Similarity3<Fixed>> = m.try_into();
    assert!(s.is_some());
}

#[test]
#[inline(never)]
fn bench_mat4_to_similarity3__two_pass() {
    let m = sim4();
    let s = mat4_to_similarity3_two_pass(m);
    assert!(s.is_some());
}
