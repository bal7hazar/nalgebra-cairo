//! Unit tests of the WP 8.4-P08 completion of `UnitQuaternion`: the division, the heterogeneous
//! operators (rotations, translations, isometries, similarities), the observer frames,
//! `rotation_between_axis`, `from_matrix_eps`, `mean_of`, `ln` / `exp`, the constructors and the
//! trait impls: exact cases, identities, panics, the measurements behind `FROM_MATRIX_MAX_ITER`
//! and `MEAN_OF_SQUARINGS`, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on
//! the same raw inputs, tolerance in ulp).
//!
//! `oracle_ext.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo unit_quaternion_completion --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/unit_quaternion/oracle_ext.cairo
//! ```
//!
//! Moved from `crates/nalgebra/src/geometry/unit_quaternion/tests_ext.cairo` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix3::Matrix3Trait;
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::base::vector3::{Vector3, Vector3Trait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::quaternion::{Quaternion, QuaternionTrait};
use nalgebra::geometry::rotation3::Rotation3;
use nalgebra::geometry::similarity3::{Similarity3, Similarity3Trait};
use nalgebra::geometry::translation3::Translation3;
use nalgebra::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use nalgebra_tests_utils::{
    ONE_RAW, excess, fx, int, m3, max_ulp_diff_q, max_ulp_diff_v3, qt, r3, u3t, ulp_diff, uqt, v3i,
    v3t,
};
use simba::scalar::Real;
use crate::unit_quaternion::oracle_ext;

const HALF_RAW: i64 = 0x80000000;

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

fn iso(t: ((i64, i64, i64), (i64, i64, i64, i64))) -> Isometry3<Fixed> {
    let (v, r) = t;
    Isometry3 { rotation: uqt(r), translation: Translation3 { vector: v3t(v) } }
}

fn sim(t: ((i64, i64, i64), (i64, i64, i64, i64), i64)) -> Similarity3<Fixed> {
    let (v, r, s) = t;
    Similarity3 { isometry: iso((v, r)), scaling: fx(s) }
}

fn iso_err(a: Isometry3<Fixed>, b: Isometry3<Fixed>) -> u128 {
    core::cmp::max(
        max_ulp_diff_q(a.rotation.quaternion, b.rotation.quaternion),
        max_ulp_diff_v3(a.translation.vector, b.translation.vector),
    )
}

fn sim_err(a: Similarity3<Fixed>, b: Similarity3<Fixed>) -> u128 {
    core::cmp::max(iso_err(a.isometry, b.isometry), ulp_diff(a.scaling, b.scaling))
}

/// `min(|a - b|, |a + b|)` component-wise: a distance between two rotations up to the sign.
fn err_up_to_sign(a: Quaternion<Fixed>, b: Quaternion<Fixed>) -> u128 {
    core::cmp::min(max_ulp_diff_q(a, b), max_ulp_diff_q(a, -b))
}

// --- oracle vectors: products and quotients

#[test]
fn test_div_oracle_is_bit_exact() {
    let mut cases = oracle_ext::unit_quaternion_div_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, e, _) = *case;
        let got = uqt(a) / uqt(b);
        assert!(got == uqt(e));
        // Bit for bit `a * b.inverse()`.
        assert!(got == uqt(a) * uqt(b).inverse());
    }
}

#[test]
fn test_mul_rotation_oracle() {
    let mut cases = oracle_ext::unit_quaternion_mul_rotation_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, r, e, tol) = *case;
        let err = max_ulp_diff_q(uqt(q).mul_rotation(r3(r)).quaternion, qt(e));
        worst = core::cmp::max(worst, report("mul_rotation", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_div_rotation_oracle() {
    let mut cases = oracle_ext::unit_quaternion_div_rotation_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, r, e, tol) = *case;
        let err = max_ulp_diff_q(uqt(q).div_rotation(r3(r)).quaternion, qt(e));
        worst = core::cmp::max(worst, report("div_rotation", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mul_translation_oracle() {
    let mut cases = oracle_ext::unit_quaternion_mul_translation_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, t, e, tol) = *case;
        let got = uqt(q).mul_translation(Translation3 { vector: v3t(t) });
        worst = core::cmp::max(worst, report("mul_translation", n, iso_err(got, iso(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mul_isometry_oracle() {
    let mut cases = oracle_ext::unit_quaternion_mul_isometry_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, i, e, tol) = *case;
        let got = uqt(q).mul_isometry(iso(i));
        worst = core::cmp::max(worst, report("mul_isometry", n, iso_err(got, iso(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_div_isometry_oracle() {
    let mut cases = oracle_ext::unit_quaternion_div_isometry_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, i, e, tol) = *case;
        let got = uqt(q).div_isometry(iso(i));
        worst = core::cmp::max(worst, report("div_isometry", n, iso_err(got, iso(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mul_similarity_oracle() {
    let mut cases = oracle_ext::unit_quaternion_mul_similarity_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, s, e, tol) = *case;
        let got = uqt(q).mul_similarity(sim(s));
        worst = core::cmp::max(worst, report("mul_similarity", n, sim_err(got, sim(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_div_similarity_oracle() {
    let mut cases = oracle_ext::unit_quaternion_div_similarity_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, s, e, tol) = *case;
        let got = uqt(q).div_similarity(sim(s));
        worst = core::cmp::max(worst, report("div_similarity", n, sim_err(got, sim(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

/// The fused `div_isometry` agrees with upstream's literal `q * iso.inverse()` to a few ulp.
#[test]
fn test_div_isometry_alt_inverse_then_mul_agrees() {
    let mut cases = oracle_ext::unit_quaternion_div_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (q, i, _, tol) = *case;
        let fused = uqt(q).div_isometry(iso(i));
        let literal = uqt(q).mul_isometry(iso(i).inverse());
        assert!(fused.rotation == literal.rotation);
        assert!(iso_err(fused, literal) <= tol.into());
    }
}

// --- oracle vectors: construction

#[test]
fn test_rotation_between_axis_oracle() {
    let mut cases = oracle_ext::unit_quaternion_rotation_between_axis_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = UnitQuaternionTrait::rotation_between_axis(u3t(a), u3t(b)).unwrap();
        worst =
            core::cmp::max(
                worst,
                report("rotation_between_axis", n, max_ulp_diff_q(got.quaternion, qt(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scaled_rotation_between_axis_oracle() {
    let mut cases = oracle_ext::unit_quaternion_scaled_rotation_between_axis_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, s, e, tol) = *case;
        let got = UnitQuaternionAngleTrait::scaled_rotation_between_axis(u3t(a), u3t(b), fx(s))
            .unwrap();
        worst =
            core::cmp::max(
                worst,
                report(
                    "scaled_rotation_between_axis", n, max_ulp_diff_q(got.quaternion, qt(e)), tol,
                ),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_face_towards_oracle() {
    let mut cases = oracle_ext::unit_quaternion_face_towards_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (d, u, e, tol) = *case;
        let got = UnitQuaternionTrait::face_towards(v3t(d), v3t(u));
        worst =
            core::cmp::max(
                worst, report("face_towards", n, max_ulp_diff_q(got.quaternion, qt(e)), tol),
            );
        assert!(UnitQuaternionTrait::new_observer_frames(v3t(d), v3t(u)) == got);
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_look_at_rh_oracle() {
    let mut cases = oracle_ext::unit_quaternion_look_at_rh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (d, u, e, tol) = *case;
        let got = UnitQuaternionTrait::look_at_rh(v3t(d), v3t(u));
        worst =
            core::cmp::max(
                worst, report("look_at_rh", n, max_ulp_diff_q(got.quaternion, qt(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_look_at_lh_oracle() {
    let mut cases = oracle_ext::unit_quaternion_look_at_lh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (d, u, e, tol) = *case;
        let got = UnitQuaternionTrait::look_at_lh(v3t(d), v3t(u));
        worst =
            core::cmp::max(
                worst, report("look_at_lh", n, max_ulp_diff_q(got.quaternion, qt(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_from_matrix_oracle() {
    let mut cases = oracle_ext::unit_quaternion_from_matrix_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let got = UnitQuaternionAngleTrait::from_matrix(m3(m));
        worst =
            core::cmp::max(
                worst, report("from_matrix", n, max_ulp_diff_q(got.quaternion, qt(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

/// A scaled rotation has the same rotation part.
#[test]
fn test_from_matrix_ignores_a_uniform_scaling() {
    let q = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let m = q.to_rotation_matrix().matrix;
    let m3x = Matrix3Trait::from_columns(
        Vector3Trait::scale(Vector3 { x: m.m11, y: m.m21, z: m.m31 }, int(3)),
        Vector3Trait::scale(Vector3 { x: m.m12, y: m.m22, z: m.m32 }, int(3)),
        Vector3Trait::scale(Vector3 { x: m.m13, y: m.m23, z: m.m33 }, int(3)),
    );
    let got = UnitQuaternionAngleTrait::from_matrix(m3x);
    assert!(err_up_to_sign(got.quaternion, q.quaternion) <= 16);
}

// --- mean_of

#[test]
fn test_mean_of_oracle() {
    let mut cases = oracle_ext::unit_quaternion_mean_of_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let ((a, b, c), e, tol) = *case;
        let got = UnitQuaternionTrait::mean_of(array![uqt(a), uqt(b), uqt(c)].span());
        worst =
            core::cmp::max(worst, report("mean_of", n, max_ulp_diff_q(got.quaternion, qt(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mean_of_exact_cases() {
    let q = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    // The mean of one rotation is that rotation (up to the sign), of q and -q too.
    // Each squaring rounds: about one ulp per squaring on a rank-one input.
    let one = UnitQuaternionTrait::mean_of(array![q].span());
    assert!(err_up_to_sign(one.quaternion, q.quaternion) <= 16);
    let both = UnitQuaternionTrait::mean_of(array![q, -q].span());
    assert!(err_up_to_sign(both.quaternion, q.quaternion) <= 16);
    // The mean of identities is the identity (upstream: the half turn about z).
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(UnitQuaternionTrait::mean_of(array![id, id, id].span()) == id);
    // Upstream's doc example: rolls of -0.1, 0 and 0.1 average to a zero roll.
    let r = UnitQuaternionAngleTrait::from_euler_angles(fx(429496730), fx(0), fx(0));
    let mean = UnitQuaternionTrait::mean_of(array![r.inverse(), id, r].span());
    assert!(mean.quaternion.abs_diff_eq(id.quaternion, 2));
}

#[test]
#[should_panic(expected: ('nalgebra: mean of nothing',))]
fn test_mean_of_empty_panics() {
    let _ = UnitQuaternionTrait::<Fixed>::mean_of(array![].span());
}

// --- ln, exp

#[test]
fn test_ln_oracle() {
    let mut cases = oracle_ext::unit_quaternion_ln_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        worst = core::cmp::max(worst, report("ln", n, max_ulp_diff_q(uqt(q).ln(), qt(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_exp_oracle() {
    let mut cases = oracle_ext::unit_quaternion_exp_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        worst = core::cmp::max(worst, report("exp", n, max_ulp_diff_q(uqt(q).exp(), qt(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

/// `ln` is `(0, scaled_axis)`: the FULL angle, like upstream; zero for the identity.
#[test]
fn test_ln_is_the_scaled_axis() {
    let q = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    assert!(q.ln() == QuaternionTrait::from_imag(q.scaled_axis()));
    assert!(UnitQuaternionTrait::<Fixed>::identity().ln() == core::num::traits::Zero::zero());
}

// --- constructors and conversions

#[test]
fn test_new_is_from_scaled_axis() {
    let w = v3t((0x40000000, -0x30000000, 0x20000000));
    assert!(UnitQuaternionAngleTrait::new(w) == UnitQuaternionAngleTrait::from_scaled_axis(w));
    assert!(
        UnitQuaternionAngleTrait::new_eps(
            w, Real::zero(),
        ) == UnitQuaternionAngleTrait::from_scaled_axis(w),
    );
    assert!(
        UnitQuaternionAngleTrait::from_scaled_axis_eps(
            w, Real::zero(),
        ) == UnitQuaternionAngleTrait::new(w),
    );
    // Below the threshold: the identity.
    assert!(UnitQuaternionAngleTrait::new_eps(w, int(1)) == UnitQuaternionTrait::identity());
}

#[test]
fn test_from_quaternion_normalizes() {
    let q = QuaternionTrait::new(int(1), int(2), int(-3), int(4));
    assert!(UnitQuaternionTrait::from_quaternion(q) == UnitQuaternionTrait::new_normalize(q));
}

#[test]
fn test_from_basis_unchecked_is_from_rotation_matrix() {
    let q = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let m = q.to_rotation_matrix().matrix;
    let basis = [
        Vector3 { x: m.m11, y: m.m21, z: m.m31 }, Vector3 { x: m.m12, y: m.m22, z: m.m32 },
        Vector3 { x: m.m13, y: m.m23, z: m.m33 },
    ];
    assert!(
        UnitQuaternionTrait::from_basis_unchecked(
            basis,
        ) == UnitQuaternionTrait::from_rotation_matrix(Rotation3 { matrix: m }),
    );
}

#[test]
fn test_lerp_is_the_quaternion_lerp() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let b = uqt((-1829744033, 968283947, 3750500405, -308145672));
    assert!(a.lerp(b, fx(HALF_RAW)) == a.quaternion.lerp(b.quaternion, fx(HALF_RAW)));
}

#[test]
fn test_to_euler_angles_is_euler_angles() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    assert!(a.to_euler_angles() == a.euler_angles());
}

#[test]
fn test_unit_vector_transforms() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let v = u3t((1393471396, -2090207096, 3483678492));
    assert!(a.transform_unit_vector(v).value == a.transform_vector(v.value));
    assert!(a.inverse_transform_unit_vector(v).value == a.inverse_transform_vector(v.value));
}

#[test]
fn test_rotation_between_axis_degenerate_cases() {
    let x = u3t((ONE_RAW, 0, 0));
    let mx = u3t((-ONE_RAW, 0, 0));
    assert!(
        UnitQuaternionTrait::rotation_between_axis(x, x) == Some(UnitQuaternionTrait::identity()),
    );
    assert!(UnitQuaternionTrait::<Fixed>::rotation_between_axis(x, mx).is_none());
    assert!(
        UnitQuaternionAngleTrait::<Fixed>::scaled_rotation_between_axis(x, mx, int(1)).is_none(),
    );
}

#[test]
fn test_look_at_frames() {
    // look_at_lh maps dir to +z, look_at_rh to -z.
    let dir = v3i(1, 2, -2);
    let up = v3i(0, 1, 0);
    let lh = UnitQuaternionTrait::look_at_lh(dir, up);
    let rh = UnitQuaternionTrait::look_at_rh(dir, up);
    assert!(lh.transform_vector(dir).abs_diff_eq(v3i(0, 0, 3), 16));
    assert!(rh.transform_vector(dir).abs_diff_eq(v3i(0, 0, -3), 16));
}

// --- trait impls and comparisons

#[test]
fn test_default_and_one_are_the_identity() {
    let d: UnitQuaternion<Fixed> = Default::default();
    let o: UnitQuaternion<Fixed> = One::one();
    assert!(d == UnitQuaternionTrait::identity() && o == d);
    assert!(o.is_one() && !(-o).is_one());
}

#[test]
fn test_into_isometry_and_similarity() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let i: Isometry3<Fixed> = a.into();
    assert!(i.rotation == a && i.translation.vector == Vector3Trait::zeros());
    let s: Similarity3<Fixed> = a.into();
    assert!(s.isometry == i && s.scaling == Real::one());
}

#[test]
fn test_cast_is_the_identity_on_fixed() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let c: UnitQuaternion<Fixed> = a.cast();
    assert!(c == a);
}

#[test]
fn test_relative_and_ulps_eq() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let b = uqt((-1509276475, -2563574020, -2263667719, 2114881862));
    assert!(a.relative_eq(b, 2, Real::zero()) && !a.relative_eq(b, 1, Real::zero()));
    assert!(a.ulps_eq(b, 0, 2) && !a.ulps_eq(b, 0, 1));
    assert!(!a.relative_eq(-a, 0, Real::zero()));
}

#[test]
fn test_heterogeneous_operators_compose_like_the_matrices() {
    let a = uqt((-1509276477, -2563574020, -2263667719, 2114881862));
    let b = uqt((-1829744033, 968283947, 3750500405, -308145672));
    let r = b.to_rotation_matrix();
    assert!(a.mul_rotation(r).quaternion.abs_diff_eq((a * b).quaternion, 4));
    assert!(a.div_rotation(r).quaternion.abs_diff_eq((a / b).quaternion, 4));
    let v = v3i(1, -2, 3);
    let i = Isometry3Trait::from_parts(Translation3 { vector: v }, b);
    // (a * iso) * p = a * (iso * p), and (a / iso) * iso = a.
    let p = Point3 { x: int(2), y: int(0), z: int(-1) };
    assert!(
        a
            .mul_isometry(i)
            .transform_point(p)
            .abs_diff_eq(a.transform_point(i.transform_point(p)), 16),
    );
    let back = a.div_isometry(i) * i;
    assert!(back.rotation.quaternion.abs_diff_eq(a.quaternion, 4));
    assert!(back.translation.vector.abs_diff_eq(Vector3Trait::zeros(), 16));
    let s = Similarity3Trait::from_isometry(i, int(2));
    let sback = a.div_similarity(s) * s;
    assert!(sback.isometry.rotation.quaternion.abs_diff_eq(a.quaternion, 4));
    assert!(sback.scaling == Real::one());
    assert!(a.mul_similarity(s).scaling == int(2));
    let t = a.mul_translation(Translation3 { vector: v });
    assert!(
        t
            .transform_point(p)
            .abs_diff_eq(a.transform_point(Point3 { x: int(3), y: int(-2), z: int(2) }), 16),
    );
}
