//! Unit tests of `IsometryMatrix2` / `IsometryMatrix3`: the oracle vectors of `tools/oracle`
//! (suite `pose_completion`, upstream nalgebra 0.35 on the same raw inputs, tolerance in ulp),
//! exact cases (identity, `Default`, `One`, pure translations, the conversions), the identities a
//! rigid-body transform must satisfy, the bit-for-bit agreement with `Isometry2` where the
//! kernels are the same, the operator forms and the compound assignments.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix3::Matrix3;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::isometry2::{Isometry2, Isometry2AngleTrait, Isometry2Trait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::isometry_matrix2::{
    IsometryMatrix2, IsometryMatrix2AngleTrait, IsometryMatrix2Trait,
};
use nalgebra::geometry::isometry_matrix3::{
    IsometryMatrix3, IsometryMatrix3AngleTrait, IsometryMatrix3Trait,
};
use nalgebra::geometry::rotation2::{Rotation2AngleTrait, Rotation2Trait};
use nalgebra::geometry::rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
use nalgebra::geometry::similarity_matrix2::{SimilarityMatrix2, SimilarityMatrix2Trait};
use nalgebra::geometry::similarity_matrix3::{SimilarityMatrix3, SimilarityMatrix3Trait};
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::Translation3Trait;
use nalgebra_tests_utils::{
    ONE_RAW, fx, isom2t, isom3t, max_ulp_diff_v2, max_ulp_diff_v3, p2t, p3t, u2, u3, v2t, v3t,
};
use simba::scalar::Real;
use crate::common::{isom2_err, isom3_err, report};
use crate::oracle;

/// `new((1.5, -2.25), 0.4 rad)`.
fn a2() -> IsometryMatrix2<Fixed> {
    IsometryMatrix2AngleTrait::new(v2t((0x180000000, -0x240000000)), fx(0x66666666))
}

/// `new((-0.75, 0.5), -1/6 rad)`.
fn b2() -> IsometryMatrix2<Fixed> {
    IsometryMatrix2AngleTrait::new(v2t((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa))
}

/// `new((1.5, -2.25, 0.75), (0.25, -0.1875, 0.125))`.
fn a3() -> IsometryMatrix3<Fixed> {
    IsometryMatrix3AngleTrait::new(
        v3t((0x180000000, -0x240000000, 0xc0000000)), v3t((0x40000000, -0x30000000, 0x20000000)),
    )
}

/// `new((-0.75, 0.5, 2), (-0.4375, 0.125, 0.15625))`.
fn b3() -> IsometryMatrix3<Fixed> {
    IsometryMatrix3AngleTrait::new(
        v3t((-0xc0000000, 0x80000000, 0x200000000)), v3t((-0x70000000, 0x20000000, 0x28000000)),
    )
}

// --- oracle vectors, 2D

#[test]
fn test_isometry_matrix2_products_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix2_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, et, er, tol) = *case;
        let (a, b) = (isom2t(at, ar), isom2t(bt, br));
        worst = core::cmp::max(worst, report("mul", n, isom2_err(a * b, et, er), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_matrix2_inv_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, et, er, tol) = *case;
        let (a, b) = (isom2t(at, ar), isom2t(bt, br));
        worst = core::cmp::max(worst, report("inv_mul", n, isom2_err(a.inv_mul(b), et, er), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_matrix2_div_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, et, er, tol) = *case;
        let (a, b) = (isom2t(at, ar), isom2t(bt, br));
        worst = core::cmp::max(worst, report("div", n, isom2_err(a / b, et, er), tol));
        // Upstream's formula, bit for bit.
        assert!(a / b == a * b.inverse());
        n += 1;
    }
    let mut cases = oracle::isometry_matrix2_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, et, er, tol) = *case;
        let a = isom2t(at, ar);
        worst = core::cmp::max(worst, report("inverse", n, isom2_err(a.inverse(), et, er), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_isometry_matrix2_transforms_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix2_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, p, e, tol) = *case;
        let a = isom2t(at, ar);
        let got = a.transform_point(p2t(p));
        let err = max_ulp_diff_v2(Vector2 { x: got.x, y: got.y }, v2t(e));
        worst = core::cmp::max(worst, report("transform_point", n, err, tol));
        // The fused kernel is bit for bit the rotation followed by the addition.
        let r = a.rotation.transform_vector(v2t(p));
        assert!(got == Point2 { x: r.x + a.translation.vector.x, y: r.y + a.translation.vector.y });
        n += 1;
    }
    let mut cases = oracle::isometry_matrix2_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, p, e, tol) = *case;
        let got = isom2t(at, ar).inverse_transform_point(p2t(p));
        let err = max_ulp_diff_v2(Vector2 { x: got.x, y: got.y }, v2t(e));
        worst = core::cmp::max(worst, report("inverse_transform_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_isometry_matrix2_lerp_slerp_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix2_lerp_slerp_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, t, et, er, tol) = *case;
        let (a, b) = (isom2t(at, ar), isom2t(bt, br));
        worst =
            core::cmp::max(
                worst, report("lerp_slerp", n, isom2_err(a.lerp_slerp(b, fx(t)), et, er), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

// --- oracle vectors, 3D

#[test]
fn test_isometry_matrix3_products_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix3_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, et, er, tol) = *case;
        let (a, b) = (isom3t(at, ar), isom3t(bt, br));
        worst = core::cmp::max(worst, report("mul", n, isom3_err(a * b, et, er), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_matrix3_inv_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, et, er, tol) = *case;
        let (a, b) = (isom3t(at, ar), isom3t(bt, br));
        worst = core::cmp::max(worst, report("inv_mul", n, isom3_err(a.inv_mul(b), et, er), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_matrix3_div_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, et, er, tol) = *case;
        let (a, b) = (isom3t(at, ar), isom3t(bt, br));
        worst = core::cmp::max(worst, report("div", n, isom3_err(a / b, et, er), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_matrix3_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, et, er, tol) = *case;
        let a = isom3t(at, ar);
        worst = core::cmp::max(worst, report("inverse", n, isom3_err(a.inverse(), et, er), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_isometry_matrix3_transforms_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix3_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, p, e, tol) = *case;
        let a = isom3t(at, ar);
        let got = a.transform_point(p3t(p));
        let err = max_ulp_diff_v3(Vector3 { x: got.x, y: got.y, z: got.z }, v3t(e));
        worst = core::cmp::max(worst, report("transform_point", n, err, tol));
        let r = a.rotation.transform_vector(v3t(p));
        let t = a.translation.vector;
        assert!(got == Point3 { x: r.x + t.x, y: r.y + t.y, z: r.z + t.z });
        n += 1;
    }
    let mut cases = oracle::isometry_matrix3_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, p, e, tol) = *case;
        let got = isom3t(at, ar).inverse_transform_point(p3t(p));
        let err = max_ulp_diff_v3(Vector3 { x: got.x, y: got.y, z: got.z }, v3t(e));
        worst = core::cmp::max(worst, report("inverse_transform_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_isometry_matrix3_frames_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix3_face_towards_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, et, er, tol) = *case;
        let got = IsometryMatrix3Trait::face_towards(p3t(eye), p3t(target), v3t(up));
        worst = core::cmp::max(worst, report("face_towards", n, isom3_err(got, et, er), tol));
        assert!(IsometryMatrix3Trait::new_observer_frame(p3t(eye), p3t(target), v3t(up)) == got);
        n += 1;
    }
    let mut cases = oracle::isometry_matrix3_look_at_rh_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, et, er, tol) = *case;
        let got = IsometryMatrix3Trait::look_at_rh(p3t(eye), p3t(target), v3t(up));
        worst = core::cmp::max(worst, report("look_at_rh", n, isom3_err(got, et, er), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_matrix3_look_at_lh_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, et, er, tol) = *case;
        let got = IsometryMatrix3Trait::look_at_lh(p3t(eye), p3t(target), v3t(up));
        worst = core::cmp::max(worst, report("look_at_lh", n, isom3_err(got, et, er), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_isometry_matrix3_lerp_slerp_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::isometry_matrix3_lerp_slerp_cases();
    while let Some(case) = cases.pop_front() {
        let (at, ar, bt, br, t, et, er, tol) = *case;
        let (a, b) = (isom3t(at, ar), isom3t(bt, br));
        let got = a.lerp_slerp(b, fx(t));
        worst = core::cmp::max(worst, report("lerp_slerp", n, isom3_err(got, et, er), tol));
        assert!(a.try_lerp_slerp(b, fx(t), Real::zero()) == Some(got));
        n += 1;
    }
    assert!(worst == 0);
}

// --- exact cases

#[test]
fn test_identity_default_one() {
    let i2 = IsometryMatrix2Trait::<Fixed>::identity();
    assert!(i2.rotation == Rotation2Trait::identity());
    assert!(i2.translation == Translation2Trait::identity());
    assert!(Default::<IsometryMatrix2<Fixed>>::default() == i2);
    assert!(One::<IsometryMatrix2<Fixed>>::one() == i2);
    assert!(i2.is_one() && !a2().is_one() && a2().is_non_one());
    let i3 = IsometryMatrix3Trait::<Fixed>::identity();
    assert!(i3.rotation == Rotation3Trait::identity());
    assert!(i3.translation == Translation3Trait::identity());
    assert!(Default::<IsometryMatrix3<Fixed>>::default() == i3);
    assert!(One::<IsometryMatrix3<Fixed>>::one() == i3);
    assert!(i3.is_one() && a3().is_non_one());
}

#[test]
fn test_constructors_agree() {
    let r = Rotation2AngleTrait::<Fixed>::new(fx(0x66666666));
    let t = Translation2Trait::new(fx(0x180000000), fx(-0x240000000));
    assert!(IsometryMatrix2Trait::from_parts(t, r) == a2());
    assert!(
        IsometryMatrix2AngleTrait::<
            Fixed,
        >::rotation(
            fx(0x66666666),
        ) == IsometryMatrix2Trait::from_parts(Translation2Trait::identity(), r),
    );
    assert!(
        IsometryMatrix2Trait::translation(
            fx(0x180000000), fx(-0x240000000),
        ) == IsometryMatrix2Trait::from_parts(t, Rotation2Trait::identity()),
    );
    // Upstream builds `translation` with `Rotation2::new(0)`: the identity exactly.
    assert!(Rotation2AngleTrait::<Fixed>::new(Real::zero()) == Rotation2Trait::identity());
    let w = v3t((0x40000000, -0x30000000, 0x20000000));
    let r3: Rotation3<Fixed> = Rotation3AngleTrait::new(w);
    assert!(IsometryMatrix3AngleTrait::<Fixed>::rotation(w).rotation == r3);
    assert!(
        IsometryMatrix3Trait::translation(fx(1), fx(2), fx(3)).translation.vector == v3t((1, 2, 3)),
    );
}

#[test]
fn test_conversions() {
    let t = Translation2Trait::new(fx(0x180000000), fx(-0x240000000));
    let from_t: IsometryMatrix2<Fixed> = t.into();
    let from_v: IsometryMatrix2<Fixed> = t.vector.into();
    let from_p: IsometryMatrix2<Fixed> = Point2 { x: t.vector.x, y: t.vector.y }.into();
    let from_a: IsometryMatrix2<Fixed> = [t.vector.x, t.vector.y].into();
    let pure = IsometryMatrix2Trait::from_parts(t, Rotation2Trait::identity());
    assert!(from_t == pure && from_v == pure && from_p == pure && from_a == pure);
    let t3 = Translation3Trait::new(fx(1), fx(-2), fx(3));
    let from_t3: IsometryMatrix3<Fixed> = t3.into();
    let from_v3: IsometryMatrix3<Fixed> = t3.vector.into();
    let from_p3: IsometryMatrix3<Fixed> = Point3 { x: fx(1), y: fx(-2), z: fx(3) }.into();
    let from_a3: IsometryMatrix3<Fixed> = [fx(1), fx(-2), fx(3)].into();
    let pure3 = IsometryMatrix3Trait::from_parts(t3, Rotation3Trait::identity());
    assert!(from_t3 == pure3 && from_v3 == pure3 && from_p3 == pure3 && from_a3 == pure3);
    // Isometry2 <-> IsometryMatrix2 is exact both ways.
    let iso: Isometry2<Fixed> = a2().into();
    let back: IsometryMatrix2<Fixed> = iso.into();
    assert!(back == a2());
    let u: Isometry2<Fixed> = Isometry2AngleTrait::new(v2t((3, -4)), fx(0x12345678));
    let m: IsometryMatrix2<Fixed> = u.into();
    let u_back: Isometry2<Fixed> = m.into();
    assert!(u_back == u);
    // Isometry3 <-> IsometryMatrix3 through Shepperd's method: a few ulp.
    let q: Isometry3<Fixed> = a3().into();
    let m3: IsometryMatrix3<Fixed> = q.into();
    assert!(m3.abs_diff_eq(a3(), 8));
    // Into a similarity of scaling 1.
    let s: SimilarityMatrix2<Fixed> = a2().into();
    assert!(s.isometry == a2() && s.scaling == Real::one());
    let s3: SimilarityMatrix3<Fixed> = a3().into();
    assert!(s3.isometry == a3() && s3.scaling == Real::one());
}

#[test]
fn test_matrix_and_complex_forms_transform_alike() {
    // Same kernels: the transforms of an IsometryMatrix2 and of its Isometry2 are bit-identical.
    let m = a2();
    let u: Isometry2<Fixed> = m.into();
    let p = p2t((0x123456789, -0x98765432));
    assert!(m.transform_point(p) == u.transform_point(p));
    assert!(m.inverse_transform_point(p) == u.inverse_transform_point(p));
    let v = v2t((-0x55555555, 0x1abcdef01));
    assert!(m.transform_vector(v) == u.transform_vector(v));
    assert!(m.inverse_transform_vector(v) == u.inverse_transform_vector(v));
}

#[test]
fn test_homogeneous_forms() {
    let a = a2();
    let h = a.to_homogeneous();
    let m = a.rotation.matrix;
    assert!(
        h == Matrix3 {
            m11: m.m11,
            m21: m.m21,
            m31: fx(0),
            m12: m.m12,
            m22: m.m22,
            m32: fx(0),
            m13: a.translation.vector.x,
            m23: a.translation.vector.y,
            m33: fx(ONE_RAW),
        },
    );
    assert!(a.to_matrix() == h);
    let b = a3();
    let h3: Matrix4<Fixed> = b.to_homogeneous();
    assert!(h3.m14 == b.translation.vector.x && h3.m34 == b.translation.vector.z);
    assert!(h3.m23 == b.rotation.matrix.m23 && h3.m44 == fx(ONE_RAW) && h3.m41 == fx(0));
    assert!(b.to_matrix() == h3);
}

#[test]
fn test_inverse_and_composition_identities() {
    let (a, b) = (a2(), b2());
    assert!((a * a.inverse()).abs_diff_eq(IsometryMatrix2Trait::identity(), 4));
    assert!(a.inv_mul(b).abs_diff_eq(a.inverse() * b, 4));
    let (c, d) = (a3(), b3());
    assert!((c * c.inverse()).abs_diff_eq(IsometryMatrix3Trait::identity(), 8));
    assert!(c.inv_mul(d).abs_diff_eq(c.inverse() * d, 8));
    // The inverse transforms undo the transforms.
    let p = p3t((0x100000000, -0x200000000, 0x80000000));
    let q = c.inverse_transform_point(c.transform_point(p));
    assert!(
        max_ulp_diff_v3(
            Vector3 { x: q.x, y: q.y, z: q.z }, v3t((0x100000000, -0x200000000, 0x80000000)),
        ) <= 8,
    );
}

#[test]
fn test_unit_vectors() {
    let v = u2(ONE_RAW, 0);
    assert!(
        a2().transform_unit_vector(v) == Unit { value: a2().rotation.transform_vector(v.value) },
    );
    assert!(
        a2()
            .inverse_transform_unit_vector(
                v,
            ) == Unit { value: a2().rotation.inverse_transform_vector(v.value) },
    );
    let w = u3(0, 0, ONE_RAW);
    assert!(a3().transform_unit_vector(w).value == a3().rotation.transform_vector(w.value));
    assert!(
        a3()
            .inverse_transform_unit_vector(w)
            .value == a3()
            .rotation
            .inverse_transform_vector(w.value),
    );
}

#[test]
fn test_append_and_operator_forms() {
    let (a, r) = (a2(), b2().rotation);
    // Appending a rotation = composing with it on the left.
    let mut x = a;
    x.append_rotation_mut(r);
    let rot_iso = IsometryMatrix2Trait::from_parts(Translation2Trait::identity(), r);
    assert!(x == rot_iso * a);
    // About the centre: only the rotation changes.
    let mut y = a;
    y.append_rotation_wrt_center_mut(r);
    assert!(y.translation == a.translation && y.rotation == r * a.rotation);
    // About a point: the point stays fixed.
    let p = p2t((0x100000000, 0x200000000));
    let mut z = IsometryMatrix2Trait::identity();
    z.append_rotation_wrt_point_mut(r, p);
    assert!(
        max_ulp_diff_v2(
            Vector2 { x: z.transform_point(p).x, y: z.transform_point(p).y },
            v2t((0x100000000, 0x200000000)),
        ) <= 2,
    );
    assert!(z == IsometryMatrix2Trait::rotation_wrt_point(r, p));
    // Translations.
    let t = Translation2Trait::new(fx(0x10000000), fx(-0x20000000));
    let mut w = a;
    w.append_translation_mut(t);
    assert!(w.rotation == a.rotation);
    assert!(w.translation.vector.x == a.translation.vector.x + t.vector.x);
    let tr: IsometryMatrix2<Fixed> = t.into();
    assert!(a.mul_translation(t) == a * tr);
    // Rotations on the right.
    assert!(a.mul_rotation(r) == IsometryMatrix2Trait::from_parts(a.translation, a.rotation * r));
    assert!(a.div_rotation(r) == IsometryMatrix2Trait::from_parts(a.translation, a.rotation / r));
}

#[test]
fn test_compound_assignments() {
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
    assert!(z == a / b);
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
    assert!(w == c / d);
}

#[test]
fn test_mixed_products_with_similarities() {
    let s = SimilarityMatrix2Trait::from_isometry(b2(), fx(0x180000000));
    let p = a2().mul_similarity(s);
    assert!(p.isometry == a2() * b2() && p.scaling == s.scaling);
    assert!(a2().div_similarity(s) == a2().mul_similarity(s.inverse()));
    let s3 = SimilarityMatrix3Trait::from_isometry(b3(), fx(0x80000000));
    let p3 = a3().mul_similarity(s3);
    assert!(p3.isometry == a3() * b3() && p3.scaling == s3.scaling);
    assert!(a3().div_similarity(s3) == a3().mul_similarity(s3.inverse()));
}

#[test]
fn test_approximate_comparisons_and_cast() {
    let (a, b) = (a2(), b2());
    assert!(a.relative_eq(a, 0, Real::zero()) && a.ulps_eq(a, 0, 0));
    assert!(!a.relative_eq(b, 4, Real::zero()) && !a.ulps_eq(b, 4, 4));
    let mut c = a;
    c.translation.vector.x = c.translation.vector.x + fx(3);
    assert!(!a.ulps_eq(c, 2, 2) && a.ulps_eq(c, 0, 3) && a.relative_eq(c, 3, Real::zero()));
    let cast: IsometryMatrix2<Fixed> = a.cast();
    assert!(cast == a);
    let cast3: IsometryMatrix3<Fixed> = a3().cast();
    assert!(cast3 == a3());
    assert!(a3().relative_eq(a3(), 0, Real::zero()) && a3().ulps_eq(a3(), 0, 0));
    assert!(!a3().ulps_eq(b3(), 4, 4));
}

#[test]
fn test_try_lerp_slerp_refuses_aligned_rotations() {
    let a = a3();
    let mut b = a;
    b.translation.vector.x = fx(0);
    assert!(a.try_lerp_slerp(b, fx(0x80000000), fx(0x100000)).is_none());
}

#[test]
fn test_observer_frames_are_consistent() {
    let (eye, target, up) = (
        p3t((ONE_RAW, 2 * ONE_RAW, 0)), p3t((0, 0, -ONE_RAW)), v3t((0, ONE_RAW, 0)),
    );
    let f = IsometryMatrix3Trait::face_towards(eye, target, up);
    let rh = IsometryMatrix3Trait::look_at_rh(eye, target, up);
    let lh = IsometryMatrix3Trait::look_at_lh(eye, target, up);
    // `look_at_lh` is the inverse of `face_towards`; `look_at_rh` maps `target` onto the -z axis.
    assert!((lh * f).abs_diff_eq(IsometryMatrix3Trait::identity(), 16));
    let t = rh.transform_point(target);
    assert!(t.x.abs_diff_eq(fx(0), 16) && t.y.abs_diff_eq(fx(0), 16));
    assert!(t.z.is_sign_negative());
    // Same frames as the unit-quaternion isometries, to the conversion.
    let q = Isometry3Trait::face_towards(eye, target, up);
    let m: IsometryMatrix3<Fixed> = q.into();
    assert!(m.abs_diff_eq(f, 16));
}
