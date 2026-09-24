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

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::MatrixMul;
use crate::base::matrix4::Matrix4Trait;
use crate::base::matrix_test_utils::{ONE_RAW, fx, int, iso3t, m4, p3t, uqt, v3t};
use crate::base::point3::{Point3, Point3Trait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::isometry3::Isometry3InternalTrait;
use crate::geometry::quaternion::Quaternion;
use crate::geometry::translation3::{Translation3, Translation3Trait};
use crate::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use super::{Isometry3, Isometry3AngleTrait, Isometry3Trait, oracle};

/// The raw value of 1.

fn id() -> Isometry3<Fixed> {
    Isometry3Trait::<Fixed>::identity()
}

/// `new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))`.
fn a() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3t((0x180000000, -0x240000000, 0x3c0000000)), v3t((0x40000000, -0x30000000, 0x20000000)),
    )
}

/// `new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))`.
fn b() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3t((-0xc0000000, 0x80000000, 0x140000000)), v3t((-0x80000000, 0x60000000, 0xe0000000)),
    )
}

/// The half turn about `y`, exactly representable: `(w, i, j, k) = (0, 0, 1, 0)`.
fn half_turn_y() -> UnitQuaternion<Fixed> {
    uqt((0, 0, ONE_RAW, 0))
}

// --- construction

#[test]
fn test_identity_is_exact() {
    let i = id();
    assert!(i.rotation == UnitQuaternionTrait::<Fixed>::identity());
    assert!(i.translation == Translation3Trait::<Fixed>::identity());
    assert!(i.transform_point(p3t((0x123, -0x456, 0x789))) == p3t((0x123, -0x456, 0x789)));
}

#[test]
fn test_from_parts_and_the_constructors_agree() {
    let t = Translation3Trait::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000));
    let r: UnitQuaternion<Fixed> = UnitQuaternionAngleTrait::from_scaled_axis(
        v3t((0x40000000, -0x30000000, 0x20000000)),
    );
    let i = Isometry3Trait::from_parts(t, r);
    assert!(i == a());
    assert!(i.translation == t && i.rotation == r);
    let pure_t: Isometry3<Fixed> = t.into();
    let pure_r = Isometry3Trait::from_parts(Translation3Trait::identity(), r);
    assert!(Isometry3Trait::translation(t.vector.x, t.vector.y, t.vector.z) == pure_t);
    assert!(
        Isometry3AngleTrait::<
            Fixed,
        >::rotation(v3t((0x40000000, -0x30000000, 0x20000000))) == pure_r,
    );
}

#[test]
fn test_pure_translation_moves_points_exactly() {
    let t = Isometry3Trait::translation(fx(0x140000000), fx(-0x60000000), fx(0x280000000));
    assert!(t.rotation == UnitQuaternionTrait::<Fixed>::identity());
    // The rotation is the exact identity, so the point comes out bit for bit shifted.
    let p = p3t((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(t.transform_point(p) == p3t((-0x140000000, 0x360000000, 0x340000000)));
    assert!(t.transform_vector(v3t((7, -9, 11))) == v3t((7, -9, 11)));
}

#[test]
fn test_half_turn_is_exact() {
    let i = Isometry3Trait::from_parts(
        Translation3Trait::new(int(1), Real::zero(), Real::zero()), half_turn_y(),
    );
    // A half turn about `y` maps `(x, y, z)` to `(-x, y, -z)`, then `(1, 0, 0)` is added.
    assert!(
        i
            .transform_point(
                p3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
            ) == p3t((0, 2 * ONE_RAW, -3 * ONE_RAW)),
    );
    assert!(
        i
            .transform_vector(
                v3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
            ) == v3t((-ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW)),
    );
    assert!(
        i
            .inverse_transform_point(
                p3t((0, 2 * ONE_RAW, -3 * ONE_RAW)),
            ) == p3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
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
    let (i, p) = (a(), p3t((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!(i.inverse_transform_point(i.transform_point(p)).abs_diff_eq(p, 8));
    assert!(i.inverse_transform_point(p).abs_diff_eq(i.inverse().transform_point(p), 8));
}

#[test]
fn test_inverse_transform_vector_is_the_conjugate_rotation() {
    let (i, v) = (a(), v3t((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!(i.inverse_transform_vector(v) == i.rotation.inverse_transform_vector(v));
    assert!(i.inverse_transform_vector(i.transform_vector(v)).abs_diff_eq(v, 8));
}

#[test]
fn test_mul_is_the_composition_of_the_actions() {
    let (x, y, p) = (a(), b(), p3t((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!((x * y).transform_point(p).abs_diff_eq(x.transform_point(y.transform_point(p)), 32));
    assert!(x * id() == x);
    assert!(id() * x == x);
}

#[test]
fn test_inv_mul_is_the_inverse_times_other_within_nine_ulp() {
    let (x, y) = (a(), b());
    // Same transform, but `inv_mul` rounds one intermediate less (see `benches.cairo`).
    assert!(x.inv_mul(y).abs_diff_eq(x.inverse() * y, 4));
    assert!(x.inv_mul(x).abs_diff_eq(id(), 4));
    let p = p3t((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(
        x
            .inv_mul(y)
            .transform_point(p)
            .abs_diff_eq(x.inverse_transform_point(y.transform_point(p)), 9),
    );
}

// --- append / prepend

#[test]
fn test_append_and_prepend_translation_match_the_composition() {
    let (i, t) = (a(), Translation3Trait::new(fx(0x140000000), fx(-0x60000000), fx(0x280000000)));
    let ti: Isometry3<Fixed> = t.into();
    assert!({
        let mut m = i;
        m.append_translation_mut(t);
        m
    } == ti * i);
    assert!(i.mul_translation(t) == i * ti);
}

#[test]
fn test_append_and_prepend_rotation_match_the_composition() {
    let (i, r) = (a(), half_turn_y());
    let ri = Isometry3Trait::from_parts(Translation3Trait::identity(), r);
    assert!({
        let mut m = i;
        m.append_rotation_mut(r);
        m
    } == ri * i);
    assert!(i.mul_unit_quaternion(r) == i * ri);
}

#[test]
fn test_append_rotation_wrt_point_fixes_that_point() {
    let (i, r, p) = (a(), half_turn_y(), p3t((0x180000000, -0x80000000, 0x40000000)));
    let j = {
        let mut m = i;
        m.append_rotation_wrt_point_mut(r, p);
        m
    };
    let shift: Isometry3<Fixed> = Translation3Trait::new(p.x, p.y, p.z).into();
    let back: Isometry3<Fixed> = Translation3Trait::new(-p.x, -p.y, -p.z).into();
    let ri = Isometry3Trait::from_parts(Translation3Trait::identity(), r);
    assert!(j.abs_diff_eq(shift * ri * back * i, 4));
    assert!(j.rotation == r * i.rotation);
}

#[test]
fn test_append_rotation_wrt_center_keeps_the_translation() {
    let (i, r) = (a(), half_turn_y());
    let j = {
        let mut m = i;
        m.append_rotation_wrt_center_mut(r);
        m
    };
    assert!(j.translation == i.translation);
    assert!(j.rotation == r * i.rotation);
    let c = Point3 {
        x: i.translation.vector.x, y: i.translation.vector.y, z: i.translation.vector.z,
    };
    assert!(j == {
        let mut m = i;
        m.append_rotation_wrt_point_mut(r, c);
        m
    });
}

// --- observer frames

#[test]
fn test_face_towards_looking_down_z_is_the_identity_rotation() {
    let eye = p3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    let target = p3t((ONE_RAW, 2 * ONE_RAW, 4 * ONE_RAW));
    let up = v3t((0, ONE_RAW, 0));
    let i = Isometry3Trait::face_towards(eye, target, up);
    assert!(i.rotation.abs_diff_eq(UnitQuaternionTrait::<Fixed>::identity(), 1));
    assert!(i.translation.vector == v3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)));
    // The observer's local `z` axis points at the target.
    assert!(i.transform_vector(v3t((0, 0, ONE_RAW))).abs_diff_eq(v3t((0, 0, ONE_RAW)), 2));
}

#[test]
fn test_look_at_rh_is_the_inverse_frame() {
    let eye = p3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    let target = p3t((ONE_RAW, 2 * ONE_RAW, 4 * ONE_RAW));
    let up = v3t((0, ONE_RAW, 0));
    let i = Isometry3Trait::look_at_rh(eye, target, up);
    // A half turn about `y`: the view direction is mapped onto the NEGATIVE `z` axis.
    assert!(i.rotation.abs_diff_eq(half_turn_y(), 1));
    assert!(i.translation.vector.abs_diff_eq(v3t((ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW)), 1));
    // The eye goes to the origin of the camera frame and the target onto `-z`.
    assert!(i.transform_point(eye).abs_diff_eq(p3t((0, 0, 0)), 2));
    assert!(i.transform_point(target).abs_diff_eq(p3t((0, 0, -ONE_RAW)), 2));
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
    assert!(m.m41 == Real::zero() && m.m42 == Real::zero() && m.m43 == Real::zero());
    assert!(m.m44 == Real::one());
    assert!(id().to_homogeneous() == Matrix4Trait::<Fixed>::identity());
}

/// `M · (p, 1)` and `transform_point` agree to the rounding of the matrix form (the matrix
/// entries are floored once each, then the product floors again; the quaternion form floors once).
#[test]
fn test_to_homogeneous_acts_like_transform_point() {
    let (i, p) = (a(), p3t((-0x280000000, 0x3c0000000, 0xc0000000)));
    let h = i.to_homogeneous().mul_mat(p.to_homogeneous());
    let got = i.transform_point(p);
    assert!(Real::abs_diff_eq(h.x, got.x, 16));
    assert!(Real::abs_diff_eq(h.y, got.y, 16));
    assert!(Real::abs_diff_eq(h.z, got.z, 16));
    assert!(h.w == Real::one());
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
    assert!(x.lerp_nlerp(y, Real::zero()).abs_diff_eq(x, 4));
    assert!(x.lerp_nlerp(y, Real::one()).abs_diff_eq(y, 4));
    let h = x.lerp_nlerp(y, Real::HALF);
    assert!(
        h
            .translation
            .vector
            .x == Real::lerp(x.translation.vector.x, y.translation.vector.x, Real::HALF),
    );
    let q = h.rotation.quaternion;
    assert!(Real::abs_diff_eq(Real::norm_squared4(q.i, q.j, q.k, q.w), Real::one(), 4));
}

#[test]
fn test_lerp_slerp_endpoints_and_agreement_with_lerp_nlerp() {
    let (x, y) = (a(), b());
    assert!(x.lerp_slerp(y, Real::zero()).abs_diff_eq(x, 4));
    assert!(x.lerp_slerp(y, Real::one()).abs_diff_eq(y, 16));
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
    let got = x.try_lerp_slerp(y, t, Real::zero());
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
        assert!((iso3t(x) * iso3t(y)).abs_diff_eq(iso3t(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::isometry3_inverse_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(iso3t(x).inverse().abs_diff_eq(iso3t(expected), tol));
    }
}

#[test]
fn test_inv_mul_oracle() {
    let mut cases = oracle::isometry3_inv_mul_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!(iso3t(x).inv_mul(iso3t(y)).abs_diff_eq(iso3t(expected), tol));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::isometry3_transform_point_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(iso3t(x).transform_point(p3t(p)).abs_diff_eq(p3t(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::isometry3_transform_vector_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(iso3t(x).transform_vector(v3t(v)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::isometry3_inverse_transform_point_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(iso3t(x).inverse_transform_point(p3t(p)).abs_diff_eq(p3t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_vector_oracle() {
    let mut cases = oracle::isometry3_inverse_transform_vector_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(iso3t(x).inverse_transform_vector(v3t(v)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_to_homogeneous_oracle() {
    let mut cases = oracle::isometry3_to_homogeneous_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(iso3t(x).to_homogeneous().abs_diff_eq(m4(expected), tol));
    }
}

#[test]
fn test_lerp_slerp_oracle() {
    let mut cases = oracle::isometry3_lerp_slerp_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (x, y, t, expected, tol) = *case;
        assert!(iso3t(x).lerp_slerp(iso3t(y), fx(t)).abs_diff_eq(iso3t(expected), tol));
    }
}
