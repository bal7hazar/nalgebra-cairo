//! Unit tests of `SimilarityMatrix2` / `SimilarityMatrix3`: the oracle vectors of `tools/oracle`
//! (suite `pose_completion`), exact cases (identity, `Default`, `One`, the conversions, the zero
//! scale panics), the identities with the isometry part, the operator forms and the compound
//! assignments.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::isometry_matrix2::{IsometryMatrix2AngleTrait, IsometryMatrix2Trait};
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3Trait;
use nalgebra::geometry::rotation2::{Rotation2AngleTrait, Rotation2Trait};
use nalgebra::geometry::rotation3::Rotation3AngleTrait;
use nalgebra::geometry::similarity2::{Similarity2, Similarity2Trait};
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::similarity_matrix2::{
    SimilarityMatrix2, SimilarityMatrix2AngleTrait, SimilarityMatrix2Trait,
};
use nalgebra::geometry::similarity_matrix3::{
    SimilarityMatrix3, SimilarityMatrix3AngleTrait, SimilarityMatrix3Trait,
};
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::Translation3Trait;
use nalgebra_tests_utils::{
    ONE_RAW, fx, max_ulp_diff_v2, max_ulp_diff_v3, p2t, p3t, simm2t, simm3t, v2t, v3t,
};
use simba::scalar::Real;
use crate::common::{report, simm2_err, simm3_err};
use crate::oracle;

/// `new((1.5, -2.25), 0.4 rad, 1.5)`.
fn a2() -> SimilarityMatrix2<Fixed> {
    SimilarityMatrix2AngleTrait::new(
        v2t((0x180000000, -0x240000000)), fx(0x66666666), fx(0x180000000),
    )
}

/// `new((-0.75, 0.5), -1/6 rad, 0.75)`.
fn b2() -> SimilarityMatrix2<Fixed> {
    SimilarityMatrix2AngleTrait::new(
        v2t((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa), fx(0xc0000000),
    )
}

/// `new((1.5, -2.25, 0.75), (0.25, -0.1875, 0.125), 2)`.
fn a3() -> SimilarityMatrix3<Fixed> {
    SimilarityMatrix3AngleTrait::new(
        v3t((0x180000000, -0x240000000, 0xc0000000)),
        v3t((0x40000000, -0x30000000, 0x20000000)),
        fx(0x200000000),
    )
}

/// `new((-0.75, 0.5, 2), (-0.4375, 0.125, 0.15625), 0.5)`.
fn b3() -> SimilarityMatrix3<Fixed> {
    SimilarityMatrix3AngleTrait::new(
        v3t((-0xc0000000, 0x80000000, 0x200000000)),
        v3t((-0x70000000, 0x20000000, 0x28000000)),
        fx(0x80000000),
    )
}

// --- oracle vectors

#[test]
fn test_similarity_matrix2_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::similarity_matrix2_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, bt, br, b_s, et, er, e_s, tol) = *case;
        let (a, b) = (simm2t(at, ar, a_s), simm2t(bt, br, b_s));
        worst = core::cmp::max(worst, report("mul", n, simm2_err(a * b, et, er, e_s), tol));
        n += 1;
    }
    let mut cases = oracle::similarity_matrix2_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, et, er, e_s, tol) = *case;
        let got = simm2t(at, ar, a_s).inverse();
        worst = core::cmp::max(worst, report("inverse", n, simm2_err(got, et, er, e_s), tol));
        n += 1;
    }
    let mut cases = oracle::similarity_matrix2_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, p, e, tol) = *case;
        let got = simm2t(at, ar, a_s).transform_point(p2t(p));
        let err = max_ulp_diff_v2(Vector2 { x: got.x, y: got.y }, v2t(e));
        worst = core::cmp::max(worst, report("transform_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::similarity_matrix2_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, p, e, tol) = *case;
        let got = simm2t(at, ar, a_s).inverse_transform_point(p2t(p));
        let err = max_ulp_diff_v2(Vector2 { x: got.x, y: got.y }, v2t(e));
        worst = core::cmp::max(worst, report("inverse_transform_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_similarity_matrix3_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::similarity_matrix3_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, bt, br, b_s, et, er, e_s, tol) = *case;
        let (a, b) = (simm3t(at, ar, a_s), simm3t(bt, br, b_s));
        worst = core::cmp::max(worst, report("mul", n, simm3_err(a * b, et, er, e_s), tol));
        n += 1;
    }
    let mut cases = oracle::similarity_matrix3_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, et, er, e_s, tol) = *case;
        let got = simm3t(at, ar, a_s).inverse();
        worst = core::cmp::max(worst, report("inverse", n, simm3_err(got, et, er, e_s), tol));
        n += 1;
    }
    let mut cases = oracle::similarity_matrix3_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, p, e, tol) = *case;
        let got = simm3t(at, ar, a_s).transform_point(p3t(p));
        let err = max_ulp_diff_v3(Vector3 { x: got.x, y: got.y, z: got.z }, v3t(e));
        worst = core::cmp::max(worst, report("transform_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::similarity_matrix3_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, a_s, p, e, tol) = *case;
        let got = simm3t(at, ar, a_s).inverse_transform_point(p3t(p));
        let err = max_ulp_diff_v3(Vector3 { x: got.x, y: got.y, z: got.z }, v3t(e));
        worst = core::cmp::max(worst, report("inverse_transform_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- exact cases

#[test]
fn test_identity_default_one() {
    let i2 = SimilarityMatrix2Trait::<Fixed>::identity();
    assert!(i2.isometry == IsometryMatrix2Trait::identity() && i2.scaling == Real::one());
    assert!(Default::<SimilarityMatrix2<Fixed>>::default() == i2);
    assert!(One::<SimilarityMatrix2<Fixed>>::one() == i2 && i2.is_one() && a2().is_non_one());
    let i3 = SimilarityMatrix3Trait::<Fixed>::identity();
    assert!(i3.isometry == IsometryMatrix3Trait::identity() && i3.scaling == Real::one());
    assert!(Default::<SimilarityMatrix3<Fixed>>::default() == i3);
    assert!(One::<SimilarityMatrix3<Fixed>>::one() == i3 && i3.is_one() && a3().is_non_one());
}

#[test]
fn test_constructors_and_scaling() {
    let iso = IsometryMatrix2AngleTrait::new(v2t((0x180000000, -0x240000000)), fx(0x66666666));
    assert!(a2() == SimilarityMatrix2Trait::from_isometry(iso, fx(0x180000000)));
    assert!(
        a2() == SimilarityMatrix2Trait::from_parts(
            iso.translation, Rotation2AngleTrait::new(fx(0x66666666)), fx(0x180000000),
        ),
    );
    assert!(a2().scaling() == fx(0x180000000));
    let mut s = a2();
    s.set_scaling(fx(ONE_RAW));
    assert!(s.isometry == iso && s.scaling == fx(ONE_RAW));
    let pre = a2().prepend_scaling(fx(0x200000000));
    assert!(pre.isometry == iso && pre.scaling == fx(0x300000000));
    let app = a2().append_scaling(fx(0x200000000));
    assert!(app.scaling == fx(0x300000000) && app.isometry.rotation == iso.rotation);
    assert!(app.isometry.translation.vector == v2t((0x300000000, -0x480000000)));
    let sc = SimilarityMatrix3Trait::<Fixed>::from_scaling(fx(0x200000000));
    assert!(sc.isometry == IsometryMatrix3Trait::identity() && sc.scaling == fx(0x200000000));
    let r = Rotation2AngleTrait::<Fixed>::new(fx(0x40000000));
    let p = p2t((0x100000000, -0x80000000));
    let w = SimilarityMatrix2Trait::rotation_wrt_point(r, p, fx(ONE_RAW));
    assert!(w.isometry == IsometryMatrix2Trait::rotation_wrt_point(r, p));
    let r3 = Rotation3AngleTrait::<Fixed>::new(v3t((0x40000000, 0, 0x10000000)));
    let p3 = p3t((0x100000000, -0x80000000, 0x40000000));
    let w3 = SimilarityMatrix3Trait::rotation_wrt_point(r3, p3, fx(0x80000000));
    assert!(w3.isometry == IsometryMatrix3Trait::rotation_wrt_point(r3, p3));
    assert!(w3.scaling == fx(0x80000000));
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_zero_scale_panics() {
    let _ = SimilarityMatrix2Trait::<Fixed>::from_scaling(fx(0));
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_zero_scale_panics_3d() {
    let _ = a3().prepend_scaling(fx(0));
}

#[test]
fn test_conversions() {
    let s: Similarity2<Fixed> = a2().into();
    let back: SimilarityMatrix2<Fixed> = s.into();
    assert!(back == a2());
    let s3: Similarity3<Fixed> = a3().into();
    let back3: SimilarityMatrix3<Fixed> = s3.into();
    assert!(back3.abs_diff_eq(a3(), 8));
    // The same transforms as the unit-complex similarity (the same kernels).
    let p = p2t((0x123456789, -0x98765432));
    assert!(a2().transform_point(p) == s.transform_point(p));
    assert!(a2().inverse_transform_point(p) == s.inverse_transform_point(p));
}

#[test]
fn test_transforms_and_homogeneous() {
    let a = a2();
    let v = v2t((0x100000000, 0x80000000));
    let r = a.isometry.rotation.transform_vector(v);
    assert!(a.transform_vector(v) == Vector2 { x: r.x * a.scaling, y: r.y * a.scaling });
    let h = a.to_homogeneous();
    assert!(h.m11 == a.isometry.rotation.matrix.m11 * a.scaling);
    assert!(h.m12 == a.isometry.rotation.matrix.m12 * a.scaling);
    assert!(h.m13 == a.isometry.translation.vector.x && h.m33 == fx(ONE_RAW));
    let b = a3();
    let h3: Matrix4<Fixed> = b.to_homogeneous();
    assert!(h3.m23 == b.isometry.rotation.matrix.m23 * b.scaling);
    assert!(h3.m34 == b.isometry.translation.vector.z && h3.m44 == fx(ONE_RAW));
    // The inverse transforms undo the transforms.
    let p = p3t((0x100000000, -0x200000000, 0x80000000));
    let q = b.inverse_transform_point(b.transform_point(p));
    assert!(
        max_ulp_diff_v3(
            Vector3 { x: q.x, y: q.y, z: q.z }, v3t((0x100000000, -0x200000000, 0x80000000)),
        ) <= 8,
    );
    let w = b.inverse_transform_vector(b.transform_vector(v3t((ONE_RAW, 0, -ONE_RAW))));
    assert!(max_ulp_diff_v3(w, v3t((ONE_RAW, 0, -ONE_RAW))) <= 8);
}

#[test]
fn test_operator_forms_and_assignments() {
    let (a, b) = (a2(), b2());
    assert!((a * a.inverse()).abs_diff_eq(SimilarityMatrix2Trait::identity(), 8));
    assert!(a / b == a * b.inverse());
    let iso = b.isometry;
    assert!(a.mul_isometry(iso) == a * SimilarityMatrix2Trait::from_isometry(iso, fx(ONE_RAW)));
    assert!(a.div_isometry(iso) == a.mul_isometry(iso.inverse()));
    let r = iso.rotation;
    assert!(a.mul_rotation(r).isometry == a.isometry.mul_rotation(r));
    assert!(a.div_rotation(r).isometry == a.isometry.div_rotation(r));
    let t = Translation2Trait::new(fx(0x10000000), fx(-0x20000000));
    let mut x = a;
    x *= t;
    assert!(x == a.mul_translation(t));
    let mut y = a;
    y *= iso;
    assert!(y == a.mul_isometry(iso));
    let mut z = a;
    z *= b;
    assert!(z == a * b);
    let mut u = a;
    u /= iso;
    assert!(u == a.div_isometry(iso));
    let mut w = a;
    w /= b;
    assert!(w == a / b);
    // Appends act on the isometry part only.
    let mut ap = a;
    ap.append_rotation_mut(r);
    let mut ai = a.isometry;
    ai.append_rotation_mut(r);
    assert!(ap.isometry == ai && ap.scaling == a.scaling);
    let mut at = a;
    at.append_translation_mut(t);
    assert!(at.isometry.translation.vector.x == a.isometry.translation.vector.x + t.vector.x);
    let mut ac = a;
    ac.append_rotation_wrt_center_mut(r);
    assert!(ac.isometry.translation == a.isometry.translation);
    let mut apt = a;
    apt.append_rotation_wrt_point_mut(r, p2t((0, 0)));
    assert!(apt == ap);
    // 3D.
    let (c, d) = (a3(), b3());
    let t3 = Translation3Trait::new(fx(1), fx(2), fx(3));
    let mut x3 = c;
    x3 *= t3;
    assert!(x3 == c.mul_translation(t3));
    let mut y3 = c;
    y3 *= d.isometry;
    assert!(y3 == c.mul_isometry(d.isometry));
    let mut z3 = c;
    z3 /= d;
    assert!(z3 == c / d);
    let mut u3 = c;
    u3 /= d.isometry;
    assert!(u3 == c.div_isometry(d.isometry));
    let mut w3 = c;
    w3 *= d;
    assert!(w3 == c * d);
    let r3 = d.isometry.rotation;
    assert!(c.mul_rotation(r3).isometry == c.isometry.mul_rotation(r3));
    assert!(c.div_rotation(r3).isometry == c.isometry.div_rotation(r3));
}

#[test]
fn test_observer_frames() {
    let (eye, target, up) = (
        p3t((ONE_RAW, 2 * ONE_RAW, 0)), p3t((0, 0, -ONE_RAW)), v3t((0, ONE_RAW, 0)),
    );
    let s = fx(0x200000000);
    let f = SimilarityMatrix3Trait::face_towards(eye, target, up, s);
    assert!(f.isometry == IsometryMatrix3Trait::face_towards(eye, target, up) && f.scaling == s);
    assert!(SimilarityMatrix3Trait::new_observer_frames(eye, target, up, s) == f);
    let rh = SimilarityMatrix3Trait::look_at_rh(eye, target, up, s);
    assert!(rh.isometry == IsometryMatrix3Trait::look_at_rh(eye, target, up));
    let lh = SimilarityMatrix3Trait::look_at_lh(eye, target, up, s);
    assert!(lh.isometry == IsometryMatrix3Trait::look_at_lh(eye, target, up));
}

#[test]
fn test_approximate_comparisons_and_cast() {
    let (a, b) = (a2(), b2());
    assert!(a.relative_eq(a, 0, Real::zero()) && a.ulps_eq(a, 0, 0));
    assert!(!a.relative_eq(b, 4, Real::zero()) && !a.ulps_eq(b, 4, 4));
    let mut c = a;
    c.scaling = c.scaling + fx(3);
    assert!(!a.ulps_eq(c, 2, 2) && a.ulps_eq(c, 0, 3) && a.abs_diff_eq(c, 3));
    let cast: SimilarityMatrix2<Fixed> = a.cast();
    assert!(cast == a);
    let cast3: SimilarityMatrix3<Fixed> = a3().cast();
    assert!(cast3 == a3() && a3().relative_eq(a3(), 0, Real::zero()) && !a3().ulps_eq(b3(), 4, 4));
}
