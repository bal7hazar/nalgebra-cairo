//! Unit tests of the WP 8.4-P09b completion of `Isometry2/3` / `Similarity2/3` (divisions,
//! `rotation_wrt_point`, `look_at_lh`, the mixed products, the approximate comparisons, `cast`,
//! `Default` / `One`, the conversions, the compound assignments), of the `Rotation2/3` /
//! `Translation2/3` operators whose outputs are the rotation-matrix poses, and of the WP's
//! fidelity fixes: the oracle vectors of `tools/oracle` (suite `pose_completion`) and exact cases.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::geometry::isometry2::{Isometry2, Isometry2AngleTrait, Isometry2Trait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3AngleTrait, Isometry3Trait};
use nalgebra::geometry::isometry_matrix2::{IsometryMatrix2, IsometryMatrix2Trait};
use nalgebra::geometry::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use nalgebra::geometry::quaternion::{QuaternionTrait, QuaternionTranscendentalTrait};
use nalgebra::geometry::rotation2::{Rotation2, Rotation2Trait};
use nalgebra::geometry::rotation3::{Rotation3, Rotation3Trait};
use nalgebra::geometry::similarity2::{Similarity2, Similarity2AngleTrait, Similarity2Trait};
use nalgebra::geometry::similarity3::{Similarity3, Similarity3AngleTrait, Similarity3Trait};
use nalgebra::geometry::similarity_matrix2::SimilarityMatrix2Trait;
use nalgebra::geometry::similarity_matrix3::SimilarityMatrix3Trait;
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::{Translation3, Translation3Trait};
use nalgebra::geometry::unit_complex::{UnitComplexAngleTrait, UnitComplexTrait};
use nalgebra::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use nalgebra_tests_utils::{
    ONE_RAW, fx, iso2t, iso3t, isom2t, isom3t, m2, max_ulp_diff2, max_ulp_diff_q, p2t, p3t, qt, r2,
    r3, sim2t, sim3t, simm2t, simm3t, u2, u3, uct, uqt, v2t, v3t,
};
use simba::scalar::Real;
use crate::common::{
    iso2_err, iso3_err, isom2_err, isom3_err, report, sim2_err, sim3_err, simm2_err, simm3_err,
};
use crate::oracle;

/// `new((1.5, -2.25), 0.4 rad)`.
fn a2() -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(v2t((0x180000000, -0x240000000)), fx(0x66666666))
}

/// `new((-0.75, 0.5), -1/6 rad)`.
fn b2() -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(v2t((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa))
}

/// `new((1.5, -2.25, 0.75), (0.25, -0.1875, 0.125))`.
fn a3() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3t((0x180000000, -0x240000000, 0xc0000000)), v3t((0x40000000, -0x30000000, 0x20000000)),
    )
}

/// `new((-0.75, 0.5, 2), (-0.4375, 0.125, 0.15625))`.
fn b3() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3t((-0xc0000000, 0x80000000, 0x200000000)), v3t((-0x70000000, 0x20000000, 0x28000000)),
    )
}

// --- oracle vectors: Isometry2 / Similarity2

#[test]
fn test_isometry2_similarity2_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry2_div_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = iso2t(a) / iso2t(b);
        worst = core::cmp::max(worst, report("iso2 div", n, iso2_err(got, e), tol));
        assert!(got == iso2t(a) * iso2t(b).inverse());
        n += 1;
    }
    let mut cases = oracle::isometry2_div_unit_complex_cases();
    while let Some(case) = cases.pop_front() {
        let (a, r, e, tol) = *case;
        let got = iso2t(a).div_unit_complex(uct(r));
        worst = core::cmp::max(worst, report("iso2 div_uc", n, iso2_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry2_rotation_wrt_point_cases();
    while let Some(case) = cases.pop_front() {
        let (r, p, e, tol) = *case;
        let got = Isometry2Trait::rotation_wrt_point(uct(r), p2t(p));
        worst = core::cmp::max(worst, report("iso2 rotation_wrt_point", n, iso2_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry2_mul_similarity_cases();
    while let Some(case) = cases.pop_front() {
        let (a, s, e, tol) = *case;
        let got = iso2t(a).mul_similarity(sim2t(s));
        worst = core::cmp::max(worst, report("iso2 mul_sim", n, sim2_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::similarity2_div_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (s, a, e, tol) = *case;
        let got = sim2t(s).div_isometry(iso2t(a));
        worst = core::cmp::max(worst, report("sim2 div_iso", n, sim2_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::similarity2_div_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = sim2t(a) / sim2t(b);
        worst = core::cmp::max(worst, report("sim2 div", n, sim2_err(got, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- oracle vectors: Isometry3 / Similarity3

#[test]
fn test_isometry3_similarity3_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry3_div_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = iso3t(a) / iso3t(b);
        worst = core::cmp::max(worst, report("iso3 div", n, iso3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry3_div_unit_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (a, r, e, tol) = *case;
        let got = iso3t(a).div_unit_quaternion(uqt(r));
        worst = core::cmp::max(worst, report("iso3 div_uq", n, iso3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry3_rotation_wrt_point_cases();
    while let Some(case) = cases.pop_front() {
        let (r, p, e, tol) = *case;
        let got = Isometry3Trait::rotation_wrt_point(uqt(r), p3t(p));
        worst = core::cmp::max(worst, report("iso3 rotation_wrt_point", n, iso3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry3_look_at_lh_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, e, tol) = *case;
        let got = Isometry3Trait::look_at_lh(p3t(eye), p3t(target), v3t(up));
        worst = core::cmp::max(worst, report("iso3 look_at_lh", n, iso3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry3_div_similarity_cases();
    while let Some(case) = cases.pop_front() {
        let (a, s, e, tol) = *case;
        let got = iso3t(a).div_similarity(sim3t(s));
        worst = core::cmp::max(worst, report("iso3 div_sim", n, sim3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::similarity3_mul_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (s, a, e, tol) = *case;
        let got = sim3t(s).mul_isometry(iso3t(a));
        worst = core::cmp::max(worst, report("sim3 mul_iso", n, sim3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::similarity3_div_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = sim3t(a) / sim3t(b);
        worst = core::cmp::max(worst, report("sim3 div", n, sim3_err(got, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- oracle vectors: the rotation operators with rotation-matrix poses

#[test]
fn test_rotation_pose_operators_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::rotation2_mul_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (r, at, ar, et, er, tol) = *case;
        let got = r2(r).mul_isometry(isom2t(at, ar));
        worst = core::cmp::max(worst, report("r2 mul_iso", n, isom2_err(got, et, er), tol));
        n += 1;
    }
    let mut cases = oracle::rotation2_div_similarity_cases();
    while let Some(case) = cases.pop_front() {
        let (r, st, sr, ss, et, er, es, tol) = *case;
        let s = simm2t(st, sr, ss);
        let got = r2(r).div_similarity(s);
        worst = core::cmp::max(worst, report("r2 div_sim", n, simm2_err(got, et, er, es), tol));
        // Upstream's formula, bit for bit.
        assert!(got == r2(r).mul_similarity(s.inverse()));
        n += 1;
    }
    let mut cases = oracle::rotation3_mul_translation_cases();
    while let Some(case) = cases.pop_front() {
        let (r, t, et, er, tol) = *case;
        let got = r3(r).mul_translation(Translation3 { vector: v3t(t) });
        worst = core::cmp::max(worst, report("r3 mul_t", n, isom3_err(got, et, er), tol));
        n += 1;
    }
    let mut cases = oracle::rotation3_div_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (r, at, ar, et, er, tol) = *case;
        let a = isom3t(at, ar);
        let got = r3(r).div_isometry(a);
        worst = core::cmp::max(worst, report("r3 div_iso", n, isom3_err(got, et, er), tol));
        assert!(got == r3(r).mul_isometry(a.inverse()));
        n += 1;
    }
    let mut cases = oracle::rotation3_mul_similarity_cases();
    while let Some(case) = cases.pop_front() {
        let (r, st, sr, ss, et, er, es, tol) = *case;
        let got = r3(r).mul_similarity(simm3t(st, sr, ss));
        worst = core::cmp::max(worst, report("r3 mul_sim", n, simm3_err(got, et, er, es), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- oracle vectors: the fidelity fixes

#[test]
fn test_exp_sinh_cosh_of_real_quaternions_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::quaternion_exp_real_cases();
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let err = max_ulp_diff_q(qt(q).exp(), qt(e));
        worst = core::cmp::max(worst, report("exp real", n, err, tol));
        assert!(qt(q).exp() == QuaternionTrait::identity());
        n += 1;
    }
    let mut cases = oracle::quaternion_sinh_real_cases();
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        worst =
            core::cmp::max(worst, report("sinh real", n, max_ulp_diff_q(qt(q).sinh(), qt(e)), tol));
        n += 1;
    }
    let mut cases = oracle::quaternion_cosh_real_cases();
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        worst =
            core::cmp::max(worst, report("cosh real", n, max_ulp_diff_q(qt(q).cosh(), qt(e)), tol));
        n += 1;
    }
    let mut cases = oracle::quaternion_exp_nearly_real_cases();
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        worst =
            core::cmp::max(
                worst, report("exp nearly real", n, max_ulp_diff_q(qt(q).exp(), qt(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_rotation2_renormalize_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::rotation2_renormalize_cases();
    while let Some(case) = cases.pop_front() {
        let (r, drift, e, tol) = *case;
        let m = m2(r) + m2(drift);
        let mut got = Rotation2Trait::from_matrix_unchecked(m);
        got.renormalize();
        worst =
            core::cmp::max(worst, report("renormalize", n, max_ulp_diff2(got.matrix, m2(e)), tol));
        // Upstream's formula: the rotation closest to the matrix.
        assert!(got == Rotation2Trait::from_matrix(m));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
#[should_panic(expected: 'nalgebra: zero column (NaN)')]
fn test_rotation2_renormalize_of_a_zero_first_column_panics() {
    let mut r = r2([[0, ONE_RAW], [0, ONE_RAW]]);
    r.renormalize();
}

#[test]
fn test_rotation2_renormalize_uses_both_columns() {
    // Only the second column drifted: normalising the first column alone would ignore the drift,
    // upstream's closest rotation splits it between the columns.
    let mut r = r2([[ONE_RAW, 0x10000], [0, ONE_RAW]]);
    r.renormalize();
    assert!(r.matrix.m21 != fx(0));
    assert!(r == Rotation2Trait::from_matrix(m2([[ONE_RAW, 0x10000], [0, ONE_RAW]])));
}

// --- exact cases: Isometry2 / Isometry3

#[test]
fn test_isometry_default_one_and_conversions() {
    assert!(Default::<Isometry2<Fixed>>::default() == Isometry2Trait::identity());
    assert!(One::<Isometry2<Fixed>>::one() == Isometry2Trait::identity());
    assert!(One::<Isometry2<Fixed>>::one().is_one() && a2().is_non_one());
    assert!(Default::<Isometry3<Fixed>>::default() == Isometry3Trait::identity());
    assert!(One::<Isometry3<Fixed>>::one() == Isometry3Trait::identity());
    assert!(One::<Isometry3<Fixed>>::one().is_one() && a3().is_non_one());
    let v = v2t((0x180000000, -0x240000000));
    let pure = Isometry2Trait::translation(v.x, v.y);
    let from_v: Isometry2<Fixed> = v.into();
    let from_p: Isometry2<Fixed> = Point2 { x: v.x, y: v.y }.into();
    let from_a: Isometry2<Fixed> = [v.x, v.y].into();
    assert!(from_v == pure && from_p == pure && from_a == pure);
    let w = v3t((1, -2, 3));
    let pure3 = Isometry3Trait::translation(w.x, w.y, w.z);
    let from_w: Isometry3<Fixed> = w.into();
    let from_q: Isometry3<Fixed> = Point3 { x: w.x, y: w.y, z: w.z }.into();
    let from_b: Isometry3<Fixed> = [w.x, w.y, w.z].into();
    assert!(from_w == pure3 && from_q == pure3 && from_b == pure3);
    let s: Similarity2<Fixed> = a2().into();
    assert!(s.isometry == a2() && s.scaling == Real::one());
    let s3: Similarity3<Fixed> = a3().into();
    assert!(s3.isometry == a3() && s3.scaling == Real::one());
}

#[test]
fn test_isometry_operator_forms() {
    let (a, b) = (a2(), b2());
    let r = b.rotation;
    assert!(a.div_unit_complex(r) == Isometry2Trait::from_parts(a.translation, a.rotation / r));
    assert!(a.div_unit_complex(r).mul_unit_complex(r).abs_diff_eq(a, 4));
    let p = p2t((0x100000000, 0x200000000));
    let w = Isometry2Trait::rotation_wrt_point(r, p);
    assert!(
        w.transform_point(p).x.abs_diff_eq(p.x, 2) && w.transform_point(p).y.abs_diff_eq(p.y, 2),
    );
    let mut z = Isometry2Trait::identity();
    z.append_rotation_wrt_point_mut(r, p);
    assert!(z == w);
    assert!(a.to_matrix() == a.to_homogeneous());
    let u = u2(ONE_RAW, 0);
    assert!(a.transform_unit_vector(u).value == a.rotation.transform_vector(u.value));
    assert!(
        a.inverse_transform_unit_vector(u).value == a.rotation.inverse_transform_vector(u.value),
    );
    let s = Similarity2AngleTrait::new(v2t((1, 2)), fx(0x10000000), fx(0x200000000));
    assert!(
        a.mul_similarity(s).isometry == a * s.isometry && a.mul_similarity(s).scaling == s.scaling,
    );
    assert!(a.div_similarity(s) == a.mul_similarity(s.inverse()));
    // 3D.
    let (c, d) = (a3(), b3());
    let q = d.rotation;
    assert!(c.div_unit_quaternion(q) == Isometry3Trait::from_parts(c.translation, c.rotation / q));
    let p3 = p3t((0x100000000, 0x200000000, -0x80000000));
    let w3 = Isometry3Trait::rotation_wrt_point(q, p3);
    let fixed_point = w3.transform_point(p3);
    assert!(fixed_point.x.abs_diff_eq(p3.x, 4) && fixed_point.z.abs_diff_eq(p3.z, 4));
    let mut z3 = Isometry3Trait::identity();
    z3.append_rotation_wrt_point_mut(q, p3);
    assert!(z3.abs_diff_eq(w3, 2));
    assert!(c.to_matrix() == c.to_homogeneous());
    let u3v = u3(0, ONE_RAW, 0);
    assert!(c.transform_unit_vector(u3v).value == c.rotation.transform_vector(u3v.value));
    assert!(
        c
            .inverse_transform_unit_vector(u3v)
            .value == c
            .rotation
            .inverse_transform_vector(u3v.value),
    );
    let s3 = Similarity3AngleTrait::new(v3t((1, 2, 3)), v3t((0x10000000, 0, 0)), fx(0x200000000));
    assert!(c.mul_similarity(s3).isometry == c * s3.isometry);
    assert!(c.div_similarity(s3) == c.mul_similarity(s3.inverse()));
}

#[test]
fn test_isometry3_look_at_lh_and_observer_frame() {
    let (eye, target, up) = (
        p3t((ONE_RAW, 2 * ONE_RAW, 0)), p3t((0, 0, -ONE_RAW)), v3t((0, ONE_RAW, 0)),
    );
    let f = Isometry3Trait::face_towards(eye, target, up);
    assert!(Isometry3Trait::new_observer_frame(eye, target, up) == f);
    let lh = Isometry3Trait::look_at_lh(eye, target, up);
    // `look_at_lh` is the inverse of the observer frame: `target - eye` goes to the +z axis.
    assert!((lh * f).abs_diff_eq(Isometry3Trait::identity(), 16));
    let t = lh.transform_point(target);
    assert!(t.x.abs_diff_eq(fx(0), 16) && t.y.abs_diff_eq(fx(0), 16) && !t.z.is_sign_negative());
    // The same frame as the rotation-matrix isometry, to the conversion.
    let m: IsometryMatrix3<Fixed> = lh.into();
    assert!(m.abs_diff_eq(IsometryMatrix3Trait::look_at_lh(eye, target, up), 32));
}

#[test]
fn test_isometry_compound_assignments() {
    let (a, b) = (a2(), b2());
    let t = Translation2Trait::new(fx(0x10000000), fx(-0x20000000));
    let mut x = a;
    x *= t;
    assert!(x == a.mul_translation(t));
    let mut y = a;
    y *= b;
    assert!(y == a * b);
    let mut z = a;
    z /= b;
    assert!(z == a / b && z == a * b.inverse());
    let (c, d) = (a3(), b3());
    let t3 = Translation3Trait::new(fx(1), fx(2), fx(3));
    let mut u = c;
    u *= t3;
    assert!(u == c.mul_translation(t3));
    let mut v = c;
    v *= d;
    assert!(v == c * d);
    let mut w = c;
    w /= d;
    assert!(w == c / d && w == c * d.inverse());
}

#[test]
fn test_isometry_approximate_comparisons_and_cast() {
    let (a, b) = (a2(), b2());
    assert!(a.relative_eq(a, 0, Real::zero()) && a.ulps_eq(a, 0, 0));
    assert!(!a.relative_eq(b, 4, Real::zero()) && !a.ulps_eq(b, 4, 4));
    let mut c = a;
    c.translation.vector.y = c.translation.vector.y - fx(3);
    assert!(!a.ulps_eq(c, 2, 2) && a.ulps_eq(c, 0, 3));
    let cast: Isometry2<Fixed> = a.cast();
    assert!(cast == a);
    // 3D: the rotations are compared up to their sign, like upstream.
    let d = a3();
    let mut e = d;
    e.rotation = UnitQuaternion { quaternion: -d.rotation.quaternion };
    assert!(d.relative_eq(e, 0, Real::zero()) && d.ulps_eq(e, 0, 0));
    assert!(!d.ulps_eq(b3(), 4, 4));
    let cast3: Isometry3<Fixed> = d.cast();
    assert!(cast3 == d);
}

// --- exact cases: Similarity2 / Similarity3

#[test]
fn test_similarity_completion() {
    let s = Similarity2AngleTrait::new(v2t((0x180000000, 1)), fx(0x66666666), fx(0x180000000));
    assert!(Default::<Similarity2<Fixed>>::default() == Similarity2Trait::identity());
    assert!(One::<Similarity2<Fixed>>::one() == Similarity2Trait::identity() && s.is_non_one());
    let r = b2().rotation;
    let p = p2t((0x100000000, -0x80000000));
    let w = Similarity2Trait::rotation_wrt_point(r, p, fx(0x200000000));
    assert!(w.isometry == Isometry2Trait::rotation_wrt_point(r, p) && w.scaling == fx(0x200000000));
    assert!(s.div_unit_complex(r).isometry == s.isometry.div_unit_complex(r));
    let iso = b2();
    assert!(s.mul_isometry(iso) == s * Similarity2Trait::from_isometry(iso, fx(ONE_RAW)));
    assert!(s.div_isometry(iso) == s.mul_isometry(iso.inverse()));
    let t = Translation2Trait::new(fx(0x10000000), fx(-0x20000000));
    let mut x = s;
    x *= t;
    assert!(x == s.mul_translation(t));
    let mut y = s;
    y *= iso;
    assert!(y == s.mul_isometry(iso));
    let mut z = s;
    z *= s;
    assert!(z == s * s);
    let mut u = s;
    u /= iso;
    assert!(u == s.div_isometry(iso));
    let mut v = s;
    v /= s;
    assert!(v == s / s && v.abs_diff_eq(Similarity2Trait::identity(), 8));
    assert!(s.relative_eq(s, 0, Real::zero()) && s.ulps_eq(s, 0, 0) && !s.ulps_eq(v, 4, 4));
    let cast: Similarity2<Fixed> = s.cast();
    assert!(cast == s);
    // 3D.
    let s3 = Similarity3AngleTrait::new(
        v3t((1, 2, 3)), v3t((0x10000000, 0, 0x8000000)), fx(0x200000000),
    );
    assert!(Default::<Similarity3<Fixed>>::default() == Similarity3Trait::identity());
    assert!(One::<Similarity3<Fixed>>::one().is_one() && s3.is_non_one());
    let q = b3().rotation;
    let p3 = p3t((0x100000000, -0x80000000, 0x40000000));
    let w3 = Similarity3Trait::rotation_wrt_point(q, p3, fx(0x80000000));
    assert!(w3.isometry == Isometry3Trait::rotation_wrt_point(q, p3));
    assert!(s3.div_unit_quaternion(q).isometry == s3.isometry.div_unit_quaternion(q));
    let iso3 = b3();
    assert!(s3.div_isometry(iso3) == s3.mul_isometry(iso3.inverse()));
    let mut x3 = s3;
    x3 *= Translation3Trait::new(fx(1), fx(2), fx(3));
    assert!(x3 == s3.mul_translation(Translation3Trait::new(fx(1), fx(2), fx(3))));
    let mut y3 = s3;
    y3 *= iso3;
    assert!(y3 == s3.mul_isometry(iso3));
    let mut z3 = s3;
    z3 *= s3;
    assert!(z3 == s3 * s3);
    let mut u3v = s3;
    u3v /= iso3;
    assert!(u3v == s3.div_isometry(iso3));
    let mut v3v = s3;
    v3v /= s3;
    assert!(v3v == s3 / s3);
    assert!(s3.relative_eq(s3, 0, Real::zero()) && s3.ulps_eq(s3, 0, 0));
    let cast3: Similarity3<Fixed> = s3.cast();
    assert!(cast3 == s3);
}

#[test]
fn test_similarity3_observer_frames() {
    let (eye, target, up) = (
        p3t((ONE_RAW, 2 * ONE_RAW, 0)), p3t((0, 0, -ONE_RAW)), v3t((0, ONE_RAW, 0)),
    );
    let s = fx(0x200000000);
    let f = Similarity3Trait::face_towards(eye, target, up, s);
    assert!(f.isometry == Isometry3Trait::face_towards(eye, target, up) && f.scaling == s);
    assert!(Similarity3Trait::new_observer_frames(eye, target, up, s) == f);
    let rh = Similarity3Trait::look_at_rh(eye, target, up, s);
    assert!(rh.isometry == Isometry3Trait::look_at_rh(eye, target, up) && rh.scaling == s);
    let lh = Similarity3Trait::look_at_lh(eye, target, up, s);
    assert!(lh.isometry == Isometry3Trait::look_at_lh(eye, target, up) && lh.scaling == s);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_similarity_rotation_wrt_point_zero_scale_panics() {
    let _ = Similarity2Trait::rotation_wrt_point(b2().rotation, p2t((0, 0)), fx(0));
}

// --- exact cases: the rotation / translation operators

#[test]
fn test_rotation_translation_operators() {
    let r: Rotation2<Fixed> = UnitComplexAngleTrait::new(fx(0x40000000)).to_rotation_matrix();
    let t = Translation2Trait::new(fx(0x100000000), fx(-0x80000000));
    let rt = r.mul_translation(t);
    assert!(rt.rotation == r && rt.translation.vector == r.transform_vector(t.vector));
    let tr = t.mul_rotation(r);
    assert!(tr == IsometryMatrix2Trait::from_parts(t, r));
    // `t * r` then applying: rotate first, then translate.
    let iso: IsometryMatrix2<Fixed> = IsometryMatrix2Trait::from_parts(t, r);
    assert!(
        r.mul_isometry(iso) == IsometryMatrix2Trait::from_parts(Translation2Trait::identity(), r)
            * iso,
    );
    assert!(r.div_isometry(iso) == r.mul_isometry(iso.inverse()));
    let s = SimilarityMatrix2Trait::from_isometry(iso, fx(0x200000000));
    assert!(
        r.mul_similarity(s).isometry == r.mul_isometry(iso)
            && r.mul_similarity(s).scaling == s.scaling,
    );
    assert!(r.div_similarity(s) == r.mul_similarity(s.inverse()));
    // 3D.
    let r3: Rotation3<Fixed> = UnitQuaternionAngleTrait::from_scaled_axis(
        v3t((0x10000000, 0x20000000, 0)),
    )
        .to_rotation_matrix();
    let t3 = Translation3Trait::new(fx(1), fx(-2), fx(3));
    let rt3 = r3.mul_translation(t3);
    assert!(rt3.rotation == r3 && rt3.translation.vector == r3.transform_vector(t3.vector));
    assert!(t3.mul_rotation(r3) == IsometryMatrix3Trait::from_parts(t3, r3));
    let iso3 = IsometryMatrix3Trait::from_parts(t3, r3);
    assert!(
        r3.mul_isometry(iso3) == IsometryMatrix3Trait::from_parts(Translation3Trait::identity(), r3)
            * iso3,
    );
    let s3 = SimilarityMatrix3Trait::from_isometry(iso3, fx(0x80000000));
    assert!(r3.div_similarity(s3) == r3.mul_similarity(s3.inverse()));
}

// --- the other fidelity cases

#[test]
fn test_unit_quaternion_exp_of_the_identity_is_the_identity() {
    // Upstream: `UnitQuaternion::exp` is the quaternion `exp`, the identity for a real quaternion.
    let q = UnitQuaternionTrait::<Fixed>::identity();
    assert!(q.exp() == QuaternionTrait::identity());
    let minus: UnitQuaternion<Fixed> = UnitQuaternion { quaternion: -QuaternionTrait::identity() };
    assert!(minus.exp() == QuaternionTrait::identity());
}
