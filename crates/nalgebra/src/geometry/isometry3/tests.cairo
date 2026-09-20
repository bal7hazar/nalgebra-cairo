//! Unit tests of `Isometry3`: exact cases (identity, pure translations, half turns, the observer
//! frames), the identities a rigid-body transform must satisfy (`iso · iso⁻¹ = id`,
//! `inv_mul(a, b) = a⁻¹ · b`, `to_homogeneous` acts like `transform_point`, append / prepend
//! against the composition), and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on
//! the same raw inputs).
//!
//! Stated tolerances: a pure translation is carried BIT FOR BIT (`floor(x + t) = floor(x) + t` for
//! an integral `t` in raw units); everything that goes through the quaternion is floored once per
//! component and inherits the tolerance of the oracle (`ceil(2 + 3·A)`, the sensitivity policy of
//! `tools/oracle`, which for the `large` distribution reaches a few 10^5 ulp because the inputs
//! themselves are of the order of 10^4). `inv_mul(a, b)` and `a.inverse() * b` agree within a few
//! ulp but not bit for bit (the second rounds one more intermediate): see `benches.cairo`.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 5 cases per distribution, the
//! whole `isometry3` suite) with
//! `cargo run --release -- emit-cairo isometry3 --from vectors --max-per-dist 5 --out
//! <oracle.cairo>`.

use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::point3::{Point3, Point3Trait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::quaternion::Quaternion;
use crate::geometry::translation3::{Translation3, Translation3Trait};
use crate::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use super::{Isometry3, Isometry3AngleTrait, Isometry3Trait, oracle};

/// The raw value of 1.
const ONE_RAW: i64 = 0x100000000;

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn int(v: i64) -> Fixed {
    Fixed { raw: v * ONE_RAW }
}

fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

fn p3(t: (i64, i64, i64)) -> Point3<Fixed> {
    let (x, y, z) = t;
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// A unit quaternion from raw components, in the oracle's `(w, i, j, k)` order.
fn uq(t: (i64, i64, i64, i64)) -> UnitQuaternion<Fixed> {
    let (w, i, j, k) = t;
    UnitQuaternion { quaternion: Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) } }
}

fn m4(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [[m11, m12, m13, m14], [m21, m22, m23, m24], [m31, m32, m33, m34], [m41, m42, m43, m44]] =
        rows;
    Matrix4 {
        m11: fx(m11),
        m21: fx(m21),
        m31: fx(m31),
        m41: fx(m41),
        m12: fx(m12),
        m22: fx(m22),
        m32: fx(m32),
        m42: fx(m42),
        m13: fx(m13),
        m23: fx(m23),
        m33: fx(m33),
        m43: fx(m43),
        m14: fx(m14),
        m24: fx(m24),
        m34: fx(m34),
        m44: fx(m44),
    }
}

/// An oracle isometry `((tx, ty, tz), (w, i, j, k))`.
fn iso(t: ((i64, i64, i64), (i64, i64, i64, i64))) -> Isometry3<Fixed> {
    let (tr, rot) = t;
    Isometry3 { rotation: uq(rot), translation: Translation3 { vector: v3(tr) } }
}

fn id() -> Isometry3<Fixed> {
    Isometry3Trait::<Fixed>::identity()
}

/// `new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))`.
fn a() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3((0x180000000, -0x240000000, 0x3c0000000)), v3((0x40000000, -0x30000000, 0x20000000)),
    )
}

/// `new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))`.
fn b() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3((-0xc0000000, 0x80000000, 0x140000000)), v3((-0x80000000, 0x60000000, 0xe0000000)),
    )
}

/// The half turn about `y`, exactly representable: `(w, i, j, k) = (0, 0, 1, 0)`.
fn half_turn_y() -> UnitQuaternion<Fixed> {
    uq((0, 0, ONE_RAW, 0))
}

// --- construction

#[test]
fn test_identity_is_exact() {
    let i = id();
    assert!(i.rotation == UnitQuaternionTrait::<Fixed>::identity());
    assert!(i.translation == Translation3Trait::<Fixed>::identity());
    assert!(i.transform_point(p3((0x123, -0x456, 0x789))) == p3((0x123, -0x456, 0x789)));
}

#[test]
fn test_from_parts_and_the_constructors_agree() {
    let t = Translation3Trait::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000));
    let r: UnitQuaternion<Fixed> = UnitQuaternionAngleTrait::from_scaled_axis(
        v3((0x40000000, -0x30000000, 0x20000000)),
    );
    let i = Isometry3Trait::from_parts(t, r);
    assert!(i == a());
    assert!(i.translation == t && i.rotation == r);
    let pure_t: Isometry3<Fixed> = t.into();
    let pure_r: Isometry3<Fixed> = r.into();
    assert!(Isometry3Trait::translation(t.vector.x, t.vector.y, t.vector.z) == pure_t);
    assert!(Isometry3Trait::from_translation(t) == pure_t);
    assert!(Isometry3Trait::from_rotation(r) == pure_r);
    assert!(
        Isometry3AngleTrait::<Fixed>::rotation(v3((0x40000000, -0x30000000, 0x20000000))) == pure_r,
    );
}

#[test]
fn test_pure_translation_moves_points_exactly() {
    let t = Isometry3Trait::translation(fx(0x140000000), fx(-0x60000000), fx(0x280000000));
    assert!(t.rotation == UnitQuaternionTrait::<Fixed>::identity());
    // The rotation is the exact identity, so the point comes out bit for bit shifted.
    let p = p3((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(t.transform_point(p) == p3((-0x140000000, 0x360000000, 0x340000000)));
    assert!(t.transform_vector(v3((7, -9, 11))) == v3((7, -9, 11)));
}

#[test]
fn test_half_turn_is_exact() {
    let i = Isometry3Trait::from_parts(
        Translation3Trait::new(int(1), Real::ZERO, Real::ZERO), half_turn_y(),
    );
    // A half turn about `y` maps `(x, y, z)` to `(-x, y, -z)`, then `(1, 0, 0)` is added.
    assert!(
        i
            .transform_point(
                p3((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
            ) == p3((0, 2 * ONE_RAW, -3 * ONE_RAW)),
    );
    assert!(
        i
            .transform_vector(
                v3((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
            ) == v3((-ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW)),
    );
    assert!(
        i
            .inverse_transform_point(
                p3((0, 2 * ONE_RAW, -3 * ONE_RAW)),
            ) == p3((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
    );
}

// --- inverse, composition, inv_mul

#[test]
fn test_inverse_composes_to_the_identity() {
    let i = a();
    // The translation is of magnitude 4.5, so the 1-2 ulp of the quaternion kernels move it by
    // about 10 ulp: 32 ulp covers both compositions.
    assert!((i * i.inverse()).abs_diff_eq(id(), 32));
    assert!((i.inverse() * i).abs_diff_eq(id(), 32));
    assert!(i.inverse().inverse().abs_diff_eq(i, 32));
}

#[test]
fn test_inverse_of_a_pure_translation_is_exact() {
    let t = Isometry3Trait::translation(fx(0x140000000), fx(-0x60000000), fx(0x280000000));
    let e = Isometry3Trait::translation(fx(-0x140000000), fx(0x60000000), fx(-0x280000000));
    assert!(t.inverse() == e);
}

#[test]
fn test_inverse_transform_point_undoes_transform_point() {
    let (i, p) = (a(), p3((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!(i.inverse_transform_point(i.transform_point(p)).abs_diff_eq(p, 8));
    assert!(i.inverse_transform_point(p).abs_diff_eq(i.inverse().transform_point(p), 8));
}

#[test]
fn test_inverse_transform_vector_is_the_conjugate_rotation() {
    let (i, v) = (a(), v3((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!(i.inverse_transform_vector(v) == i.rotation.inverse_transform_vector(v));
    assert!(i.inverse_transform_vector(i.transform_vector(v)).abs_diff_eq(v, 8));
}

#[test]
fn test_mul_is_the_composition_of_the_actions() {
    let (x, y, p) = (a(), b(), p3((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!((x * y).transform_point(p).abs_diff_eq(x.transform_point(y.transform_point(p)), 32));
    assert!(x * id() == x);
    assert!(id() * x == x);
}

#[test]
fn test_inv_mul_is_the_inverse_times_other_within_four_ulp() {
    let (x, y) = (a(), b());
    // Same transform, but `inv_mul` rounds one intermediate less (see `benches.cairo`).
    assert!(x.inv_mul(y).abs_diff_eq(x.inverse() * y, 4));
    assert!(x.inv_mul(x).abs_diff_eq(id(), 4));
    let p = p3((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(
        x
            .inv_mul(y)
            .transform_point(p)
            .abs_diff_eq(x.inverse_transform_point(y.transform_point(p)), 8),
    );
}

// --- append / prepend

#[test]
fn test_append_and_prepend_translation_match_the_composition() {
    let (i, t) = (a(), Translation3Trait::new(fx(0x140000000), fx(-0x60000000), fx(0x280000000)));
    let ti: Isometry3<Fixed> = t.into();
    assert!(i.append_translation(t) == ti * i);
    assert!(i.prepend_translation(t) == i * ti);
}

#[test]
fn test_append_and_prepend_rotation_match_the_composition() {
    let (i, r) = (a(), half_turn_y());
    let ri: Isometry3<Fixed> = r.into();
    assert!(i.append_rotation(r) == ri * i);
    assert!(i.prepend_rotation(r) == i * ri);
}

#[test]
fn test_append_rotation_wrt_point_fixes_that_point() {
    let (i, r, p) = (a(), half_turn_y(), p3((0x180000000, -0x80000000, 0x40000000)));
    let j = i.append_rotation_wrt_point(r, p);
    let shift: Isometry3<Fixed> = Translation3Trait::new(p.x, p.y, p.z).into();
    let back: Isometry3<Fixed> = Translation3Trait::new(-p.x, -p.y, -p.z).into();
    let ri: Isometry3<Fixed> = r.into();
    assert!(j.abs_diff_eq(shift * ri * back * i, 4));
    assert!(j.rotation == r * i.rotation);
}

#[test]
fn test_append_rotation_wrt_center_keeps_the_translation() {
    let (i, r) = (a(), half_turn_y());
    let j = i.append_rotation_wrt_center(r);
    assert!(j.translation == i.translation);
    assert!(j.rotation == r * i.rotation);
    let c = Point3 {
        x: i.translation.vector.x, y: i.translation.vector.y, z: i.translation.vector.z,
    };
    assert!(j == i.append_rotation_wrt_point(r, c));
}

// --- observer frames

#[test]
fn test_face_towards_looking_down_z_is_the_identity_rotation() {
    let eye = p3((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    let target = p3((ONE_RAW, 2 * ONE_RAW, 4 * ONE_RAW));
    let up = v3((0, ONE_RAW, 0));
    let i = Isometry3Trait::face_towards(eye, target, up);
    assert!(i.rotation.abs_diff_eq(UnitQuaternionTrait::<Fixed>::identity(), 1));
    assert!(i.translation.vector == v3((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)));
    // The observer's local `z` axis points at the target.
    assert!(i.transform_vector(v3((0, 0, ONE_RAW))).abs_diff_eq(v3((0, 0, ONE_RAW)), 2));
}

#[test]
fn test_look_at_rh_is_the_inverse_frame() {
    let eye = p3((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    let target = p3((ONE_RAW, 2 * ONE_RAW, 4 * ONE_RAW));
    let up = v3((0, ONE_RAW, 0));
    let i = Isometry3Trait::look_at_rh(eye, target, up);
    // A half turn about `y`: the view direction is mapped onto the NEGATIVE `z` axis.
    assert!(i.rotation.abs_diff_eq(half_turn_y(), 1));
    assert!(i.translation.vector.abs_diff_eq(v3((ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW)), 1));
    // The eye goes to the origin of the camera frame and the target onto `-z`.
    assert!(i.transform_point(eye).abs_diff_eq(p3((0, 0, 0)), 2));
    assert!(i.transform_point(target).abs_diff_eq(p3((0, 0, -ONE_RAW)), 2));
}

// --- homogeneous form

#[test]
fn test_to_homogeneous_layout() {
    let i = a();
    let m = i.to_homogeneous();
    let r = i.rotation.to_rotation_matrix().matrix;
    assert!(m.m11 == r.m11 && m.m22 == r.m22 && m.m33 == r.m33);
    assert!(m.m12 == r.m12 && m.m13 == r.m13 && m.m21 == r.m21);
    assert!(m.m14 == i.translation.vector.x && m.m24 == i.translation.vector.y);
    assert!(m.m34 == i.translation.vector.z);
    assert!(m.m41 == Real::ZERO && m.m42 == Real::ZERO && m.m43 == Real::ZERO);
    assert!(m.m44 == Real::ONE);
    assert!(id().to_homogeneous() == Matrix4Trait::<Fixed>::identity());
}

/// `M · (p, 1)` and `transform_point` agree to the rounding of the matrix form (the matrix
/// entries are floored once each, then the product floors again; the quaternion form floors once).
#[test]
fn test_to_homogeneous_acts_like_transform_point() {
    let (i, p) = (a(), p3((-0x280000000, 0x3c0000000, 0xc0000000)));
    let h = i.to_homogeneous().mul_vec(p.to_homogeneous());
    let got = i.transform_point(p);
    assert!(Real::abs_diff_eq(h.x, got.x, 16));
    assert!(Real::abs_diff_eq(h.y, got.y, 16));
    assert!(Real::abs_diff_eq(h.z, got.z, 16));
    assert!(h.w == Real::ONE);
}

// --- renormalisation, comparison, interpolation

#[test]
fn test_renormalize_restores_the_rotation_and_keeps_the_translation() {
    let q = a().rotation;
    let drifted = Isometry3Trait::from_parts(
        a().translation,
        UnitQuaternion {
            quaternion: Quaternion {
                i: q.quaternion.i * int(4),
                j: q.quaternion.j * int(4),
                k: q.quaternion.k * int(4),
                w: q.quaternion.w * int(4),
            },
        },
    );
    let r = drifted.renormalize();
    assert!(r.translation == drifted.translation);
    assert!(r.rotation.abs_diff_eq(q, 4));
    // `renormalize_fast` is a single Newton step: valid near 1, so apply it to a small drift.
    let small = Isometry3Trait::from_parts(
        a().translation,
        UnitQuaternion {
            quaternion: Quaternion {
                i: q.quaternion.i + fx(3),
                j: q.quaternion.j - fx(2),
                k: q.quaternion.k,
                w: q.quaternion.w,
            },
        },
    );
    assert!(small.renormalize_fast().rotation.abs_diff_eq(small.renormalize().rotation, 4));
    assert!(small.renormalize_fast().translation == small.translation);
}

#[test]
fn test_abs_diff_eq_covers_both_parts() {
    let i = a();
    let j = Isometry3Trait::from_parts(
        Translation3 {
            vector: Vector3 {
                x: i.translation.vector.x + fx(2),
                y: i.translation.vector.y - fx(1),
                z: i.translation.vector.z,
            },
        },
        UnitQuaternion {
            quaternion: Quaternion {
                i: i.rotation.quaternion.i + fx(1),
                j: i.rotation.quaternion.j,
                k: i.rotation.quaternion.k,
                w: i.rotation.quaternion.w,
            },
        },
    );
    assert!(i.abs_diff_eq(j, 2));
    assert!(!i.abs_diff_eq(j, 1));
    assert!(i.abs_diff_eq(i, 0));
}

#[test]
fn test_lerp_nlerp_endpoints_and_midpoint() {
    let (x, y) = (a(), b());
    assert!(x.lerp_nlerp(y, Real::ZERO).abs_diff_eq(x, 4));
    assert!(x.lerp_nlerp(y, Real::ONE).abs_diff_eq(y, 4));
    let h = x.lerp_nlerp(y, Real::HALF);
    assert!(
        h
            .translation
            .vector
            .x == Real::lerp(x.translation.vector.x, y.translation.vector.x, Real::HALF),
    );
    let q = h.rotation.quaternion;
    assert!(Real::abs_diff_eq(Real::norm_squared4(q.i, q.j, q.k, q.w), Real::ONE, 4));
}

#[test]
fn test_lerp_slerp_endpoints_and_agreement_with_lerp_nlerp() {
    let (x, y) = (a(), b());
    assert!(x.lerp_slerp(y, Real::ZERO).abs_diff_eq(x, 4));
    assert!(x.lerp_slerp(y, Real::ONE).abs_diff_eq(y, 16));
    let (s, n) = (x.lerp_slerp(y, Real::HALF), x.lerp_nlerp(y, Real::HALF));
    assert!(s.translation == n.translation);
    // Same path, different parametrisation: at the midpoint both are the half-way rotation.
    assert!(s.rotation.abs_diff_eq(n.rotation, 4096));
}

#[test]
fn test_try_lerp_slerp_rejects_nearly_aligned_rotations() {
    let x = a();
    // The same pose: the interpolation direction is undefined, so a positive epsilon rejects it.
    assert!(x.try_lerp_slerp(x, Real::HALF, Real::<Fixed>::from_ratio(1, 512)).is_none());
    // With `epsilon = 0` the result always exists, and equals `lerp_slerp`.
    let (y, t) = (b(), Real::<Fixed>::from_ratio(1, 4));
    let got = x.try_lerp_slerp(y, t, Real::ZERO);
    assert!(got.is_some());
    assert!(got.unwrap().abs_diff_eq(x.lerp_slerp(y, t), 0));
}

// --- oracle vectors (upstream nalgebra 0.35)

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::isometry3_mul_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!((iso(x) * iso(y)).abs_diff_eq(iso(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::isometry3_inverse_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(iso(x).inverse().abs_diff_eq(iso(expected), tol));
    }
}

#[test]
fn test_inv_mul_oracle() {
    let mut cases = oracle::isometry3_inv_mul_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!(iso(x).inv_mul(iso(y)).abs_diff_eq(iso(expected), tol));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::isometry3_transform_point_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(iso(x).transform_point(p3(p)).abs_diff_eq(p3(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::isometry3_transform_vector_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(iso(x).transform_vector(v3(v)).abs_diff_eq(v3(expected), tol));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::isometry3_inverse_transform_point_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(iso(x).inverse_transform_point(p3(p)).abs_diff_eq(p3(expected), tol));
    }
}

#[test]
fn test_inverse_transform_vector_oracle() {
    let mut cases = oracle::isometry3_inverse_transform_vector_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(iso(x).inverse_transform_vector(v3(v)).abs_diff_eq(v3(expected), tol));
    }
}

#[test]
fn test_to_homogeneous_oracle() {
    let mut cases = oracle::isometry3_to_homogeneous_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(iso(x).to_homogeneous().abs_diff_eq(m4(expected), tol));
    }
}

#[test]
fn test_lerp_slerp_oracle() {
    let mut cases = oracle::isometry3_lerp_slerp_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, y, t, expected, tol) = *case;
        assert!(iso(x).lerp_slerp(iso(y), fx(t)).abs_diff_eq(iso(expected), tol));
    }
}
