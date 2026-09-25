//! Unit tests of `Affine2` / `Affine3` (WP 8.4-P11a): the oracle vectors of `tools/oracle` (suite
//! `transform`), exact cases, the block inverse and its invariant, the bit-identity of the
//! structured products with the full homogeneous products, and the panics.

use fixed::Fixed;
use nalgebra::base::matrix3::Matrix3Trait;
use nalgebra::base::matrix4::Matrix4Trait;
use nalgebra::geometry::affine2::{Affine2, Affine2Trait};
use nalgebra::geometry::affine3::{Affine3, Affine3Trait};
use nalgebra::geometry::isometry2::Isometry2Trait;
use nalgebra::geometry::isometry3::Isometry3Trait;
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3Trait;
use nalgebra::geometry::rotation2::Rotation2Trait;
use nalgebra::geometry::rotation3::Rotation3Trait;
use nalgebra::geometry::similarity2::Similarity2Trait;
use nalgebra::geometry::similarity3::Similarity3Trait;
use nalgebra::geometry::similarity_matrix3::SimilarityMatrix3Trait;
use nalgebra::geometry::transform::TransformMul;
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::Translation3Trait;
use nalgebra::geometry::unit_complex::UnitComplexTrait;
use nalgebra::geometry::unit_quaternion::UnitQuaternionTrait;
use nalgebra_tests_utils::{
    ONE_RAW, aff2t, aff3t, iso2t, iso3t, isom3t, m3, m4, max_ulp_diff3, max_ulp_diff4,
    max_ulp_diff_v2, max_ulp_diff_v3, p2t, p3t, sim2t, sim3t, simm3t, t2, t3, uct, uqt, v2t, v3i,
    v3t,
};
use crate::common::{p2_err, p3_err, p3i, report};
use crate::oracle;

/// A generic affine map: linear block `[[2, 1, 0], [0, 1, -1], [1, 0, 3]]` (determinant 7),
/// translation `(1, -2, 3)`.
fn a3() -> Affine3<Fixed> {
    aff3t(
        [[2 * ONE_RAW, ONE_RAW, 0], [0, ONE_RAW, -ONE_RAW], [ONE_RAW, 0, 3 * ONE_RAW]],
        (ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW),
    )
}

/// A generic 2D affine map: linear block `[[2, 1], [-1, 3]]` (determinant 7), translation
/// `(1.5, -2)`.
fn a2() -> Affine2<Fixed> {
    aff2t([[2 * ONE_RAW, ONE_RAW], [-ONE_RAW, 3 * ONE_RAW]], (3 * ONE_RAW / 2, -2 * ONE_RAW))
}

/// `(0.6, 0.8)`, `(tx, ty) = (0.25, -1.5)`.
fn iso2() -> nalgebra::geometry::isometry2::Isometry2<Fixed> {
    iso2t(((ONE_RAW / 4, -3 * ONE_RAW / 2), (2576980378, 3435973837)))
}

/// The rotation of 1 radian about `(1, 2, 2) / 3`.
fn q() -> nalgebra::geometry::unit_quaternion::UnitQuaternion<Fixed> {
    uqt((3769188403, 686372336, 1372744673, 1372744673))
}

// --- oracle vectors

#[test]
fn test_affine_oracle_try_inverse() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::affine2_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, e, tol) = *case;
        let got = aff2t(l, t).try_inverse().unwrap().into_inner();
        worst =
            core::cmp::max(worst, report("affine2_try_inverse", n, max_ulp_diff3(got, m3(e)), tol));
        n += 1;
    }
    let mut cases = oracle::affine3_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, e, tol) = *case;
        let got = aff3t(l, t).try_inverse().unwrap().into_inner();
        worst =
            core::cmp::max(worst, report("affine3_try_inverse", n, max_ulp_diff4(got, m4(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_affine_oracle_transforms() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::affine2_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, p, e, tol) = *case;
        let err = p2_err(aff2t(l, t).transform_point(p2t(p)), e);
        worst = core::cmp::max(worst, report("affine2_transform_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine2_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, v, e, tol) = *case;
        let err = max_ulp_diff_v2(aff2t(l, t).transform_vector(v2t(v)), v2t(e));
        worst = core::cmp::max(worst, report("affine2_transform_vector", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, p, e, tol) = *case;
        let err = p3_err(aff3t(l, t).transform_point(p3t(p)), e);
        worst = core::cmp::max(worst, report("affine3_transform_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, v, e, tol) = *case;
        let err = max_ulp_diff_v3(aff3t(l, t).transform_vector(v3t(v)), v3t(e));
        worst = core::cmp::max(worst, report("affine3_transform_vector", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, p, e, tol) = *case;
        let err = p3_err(aff3t(l, t).inverse_transform_point(p3t(p)), e);
        worst = core::cmp::max(worst, report("affine3_inverse_transform_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_inverse_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, v, e, tol) = *case;
        let err = max_ulp_diff_v3(aff3t(l, t).inverse_transform_vector(v3t(v)), v3t(e));
        worst = core::cmp::max(worst, report("affine3_inverse_transform_vector", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_affine_oracle_products() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::affine2_mul_isometry2_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, iso, e, tol) = *case;
        let got = aff2t(l, t).mul_isometry(iso2t(iso)).into_inner();
        worst =
            core::cmp::max(
                worst, report("affine2_mul_isometry2", n, max_ulp_diff3(got, m3(e)), tol),
            );
        n += 1;
    }
    let mut cases = oracle::isometry2_mul_affine2_cases();
    while let Some(case) = cases.pop_front() {
        let (iso, l, t, e, tol) = *case;
        let got: Affine2<Fixed> = iso2t(iso).mul_transform(aff2t(l, t));
        let err = max_ulp_diff3(got.into_inner(), m3(e));
        worst = core::cmp::max(worst, report("isometry2_mul_affine2", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_mul_isometry3_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, iso, e, tol) = *case;
        let got = aff3t(l, t).mul_isometry(iso3t(iso)).into_inner();
        worst =
            core::cmp::max(
                worst, report("affine3_mul_isometry3", n, max_ulp_diff4(got, m4(e)), tol),
            );
        n += 1;
    }
    let mut cases = oracle::isometry3_mul_affine3_cases();
    while let Some(case) = cases.pop_front() {
        let (iso, l, t, e, tol) = *case;
        let got: Affine3<Fixed> = iso3t(iso).mul_transform(aff3t(l, t));
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("isometry3_mul_affine3", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_mul_unit_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, r, e, tol) = *case;
        let got = aff3t(l, t).mul_unit_quaternion(uqt(r)).into_inner();
        let err = max_ulp_diff4(got, m4(e));
        worst = core::cmp::max(worst, report("affine3_mul_unit_quaternion", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::similarity3_mul_affine3_cases();
    while let Some(case) = cases.pop_front() {
        let (sim, l, t, e, tol) = *case;
        let (st, sr, ss) = sim;
        let got: Affine3<Fixed> = sim3t((st, sr, ss)).mul_transform(aff3t(l, t));
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("similarity3_mul_affine3", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- exact cases and invariants

#[test]
fn test_affine3_identity_and_accessors() {
    let i: Affine3<Fixed> = Affine3Trait::identity();
    assert!(i.into_inner() == Matrix4Trait::identity());
    assert!(i.matrix() == i.to_homogeneous() && i.unwrap() == i.into_inner());
    let m = a3().into_inner();
    let a: Affine3<Fixed> = Affine3Trait::from_matrix_unchecked(m);
    assert!(a.into_inner() == m && a == a3());
    assert!(i.try_inverse().unwrap() == i);
}

#[test]
fn test_affine3_inverse_exact() {
    // A pure translation and a scaling by 2: exact inverses.
    let a = aff3t(
        [[2 * ONE_RAW, 0, 0], [0, 2 * ONE_RAW, 0], [0, 0, 2 * ONE_RAW]],
        (ONE_RAW, -2 * ONE_RAW, 4 * ONE_RAW),
    );
    let e = aff3t(
        [[ONE_RAW / 2, 0, 0], [0, ONE_RAW / 2, 0], [0, 0, ONE_RAW / 2]],
        (-ONE_RAW / 2, ONE_RAW, -2 * ONE_RAW),
    );
    assert!(a.inverse() == e);
    let mut b = a;
    b.inverse_mut();
    assert!(b == e);
    let mut c = a;
    assert!(c.try_inverse_mut() && c == e);
    assert!((a * a.inverse()) == Affine3Trait::identity());
}

#[test]
fn test_affine3_inverse_keeps_last_row_exact() {
    let inv = a3().inverse().into_inner();
    assert!(inv.m41 == fx0() && inv.m42 == fx0() && inv.m43 == fx0());
    assert!(inv.m44 == Matrix4Trait::<Fixed>::identity().m44);
    // `a * a⁻¹` within a few ulp of the identity.
    let p = (a3() * a3().inverse()).into_inner();
    assert!(max_ulp_diff4(p, Matrix4Trait::identity()) <= 4);
}

fn fx0() -> Fixed {
    nalgebra_tests_utils::fx(0)
}

#[test]
fn test_affine3_singular() {
    let s = aff3t(
        [[ONE_RAW, 2 * ONE_RAW, 0], [2 * ONE_RAW, 4 * ONE_RAW, 0], [0, 0, ONE_RAW]], (1, 2, 3),
    );
    assert!(s.try_inverse().is_none());
    let mut t = s;
    assert!(!t.try_inverse_mut());
    assert!(t == s);
    // Upstream's `inverse_mut` ignores the failure: no panic, unchanged.
    t.inverse_mut();
    assert!(t == s);
}

#[test]
#[should_panic(expected: 'nalgebra: not invertible')]
fn test_affine3_inverse_singular_panics() {
    let s = aff3t(
        [[ONE_RAW, 2 * ONE_RAW, 0], [2 * ONE_RAW, 4 * ONE_RAW, 0], [0, 0, ONE_RAW]], (1, 2, 3),
    );
    let _i = s.inverse();
}

#[test]
#[should_panic(expected: 'nalgebra: not invertible')]
fn test_affine2_inverse_singular_panics() {
    let s = aff2t([[ONE_RAW, 2 * ONE_RAW], [2 * ONE_RAW, 4 * ONE_RAW]], (1, 2));
    let _p = s.inverse_transform_point(p2t((0, 0)));
}

#[test]
fn test_affine3_transform_ignores_last_row() {
    // Upstream's `TAffine` has no normalizer: a (malformed) last row is not read.
    let mut m = a3().into_inner();
    m.m41 = m.m11;
    m.m44 = m.m12 + m.m12;
    let b: Affine3<Fixed> = Affine3Trait::from_matrix_unchecked(m);
    let p = p3i(1, 2, 3);
    assert!(b.transform_point(p) == a3().transform_point(p));
    assert!(b.transform_vector(v3i(1, 2, 3)) == a3().transform_vector(v3i(1, 2, 3)));
    // (2 + 2 + 0 + 1, 0 + 2 - 3 - 2, 1 + 0 + 9 + 3) = (5, -3, 13)
    assert!(a3().transform_point(p) == p3i(5, -3, 13));
    assert!(a3().transform_vector(v3i(1, 2, 3)) == v3i(4, -1, 10));
}

#[test]
fn test_affine3_inverse_transform_round_trip() {
    // The inverse of a determinant-7 block is inexact: a few ulp.
    assert!(
        p3_err(
            a3().inverse_transform_point(p3i(5, -3, 13)), (ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW),
        ) <= 8,
    );
    let v = a3().inverse_transform_vector(v3i(4, -1, 10));
    assert!(max_ulp_diff_v3(v, v3i(1, 2, 3)) <= 8);
}

#[test]
fn test_affine2_basics() {
    let a = a2();
    // (2 + 2 + 1.5, -1 + 6 - 2) = (5.5, 3)
    assert!(a.transform_point(p2t((ONE_RAW, 2 * ONE_RAW))) == p2t((11 * ONE_RAW / 2, 3 * ONE_RAW)));
    assert!(a.transform_vector(v2t((ONE_RAW, 2 * ONE_RAW))) == v2t((4 * ONE_RAW, 5 * ONE_RAW)));
    let back = a.inverse_transform_point(p2t((11 * ONE_RAW / 2, 3 * ONE_RAW)));
    assert!(p2_err(back, (ONE_RAW, 2 * ONE_RAW)) <= 2);
    // The inverse of a determinant-7 block is inexact: a few ulp.
    let v = a.inverse_transform_vector(v2t((4 * ONE_RAW, 5 * ONE_RAW)));
    assert!(max_ulp_diff_v2(v, v2t((ONE_RAW, 2 * ONE_RAW))) <= 4);
    let i: Affine2<Fixed> = Affine2Trait::identity();
    assert!(i.into_inner() == Matrix3Trait::identity());
    assert!(max_ulp_diff3((a * a.inverse()).into_inner(), Matrix3Trait::identity()) <= 4);
}

// --- the structured products are bit for bit the homogeneous products

#[test]
fn test_affine3_structured_products_bit_identical() {
    let a = a3();
    let m = a.into_inner();
    let r = q().to_rotation_matrix();
    let t = t3(ONE_RAW / 3, -5 * ONE_RAW, 7 * ONE_RAW / 4);
    let iso = iso3t(
        ((ONE_RAW, 2 * ONE_RAW, -ONE_RAW), (3769188403, 686372336, 1372744673, 1372744673)),
    );
    let sim = sim3t(
        (
            (ONE_RAW, 2 * ONE_RAW, -ONE_RAW),
            (3769188403, 686372336, 1372744673, 1372744673),
            3 * ONE_RAW / 2,
        ),
    );
    assert!(a.mul_rotation(r).into_inner() == m * r.to_homogeneous());
    assert!(a.div_rotation(r).into_inner() == m * r.inverse().to_homogeneous());
    assert!(a.mul_unit_quaternion(q()).into_inner() == m * q().to_homogeneous());
    assert!(a.div_unit_quaternion(q()).into_inner() == m * q().inverse().to_homogeneous());
    assert!(a.mul_translation(t).into_inner() == m * t.to_homogeneous());
    assert!(a.div_translation(t).into_inner() == m * t.inverse().to_homogeneous());
    assert!(a.mul_isometry(iso).into_inner() == m * iso.to_homogeneous());
    assert!(a.mul_similarity(sim).into_inner() == m * sim.to_homogeneous());
    let lhs: Affine3<Fixed> = r.mul_transform(a);
    assert!(lhs.into_inner() == r.to_homogeneous() * m);
    let lhs: Affine3<Fixed> = q().mul_transform(a);
    assert!(lhs.into_inner() == q().to_homogeneous() * m);
    let lhs: Affine3<Fixed> = t.mul_transform(a);
    assert!(lhs.into_inner() == t.to_homogeneous() * m);
    let lhs: Affine3<Fixed> = iso.mul_transform(a);
    assert!(lhs.into_inner() == iso.to_homogeneous() * m);
    let lhs: Affine3<Fixed> = sim.mul_transform(a);
    assert!(lhs.into_inner() == sim.to_homogeneous() * m);
    let isom = isom3t((ONE_RAW, 0, -ONE_RAW), [[0, -ONE_RAW, 0], [ONE_RAW, 0, 0], [0, 0, ONE_RAW]]);
    let lhs: Affine3<Fixed> = isom.mul_transform(a);
    assert!(lhs.into_inner() == isom.to_homogeneous() * m);
    let simm = simm3t(
        (ONE_RAW, 0, -ONE_RAW), [[0, -ONE_RAW, 0], [ONE_RAW, 0, 0], [0, 0, ONE_RAW]], 2 * ONE_RAW,
    );
    let lhs: Affine3<Fixed> = simm.mul_transform(a);
    assert!(lhs.into_inner() == simm.to_homogeneous() * m);
    // The rotation-matrix isometries through `Affine3::from` (same bits as a structured product).
    let via: Affine3<Fixed> = isom.into();
    let prod: Affine3<Fixed> = a.mul_transform(via);
    assert!(prod.into_inner() == m * isom.to_homogeneous());
}

#[test]
fn test_affine2_structured_products_bit_identical() {
    let a = a2();
    let m = a.into_inner();
    let c = uct((2576980378, 3435973837));
    let r = c.to_rotation_matrix();
    let t = t2(ONE_RAW / 3, -5 * ONE_RAW);
    let sim = sim2t(((ONE_RAW, 2 * ONE_RAW), (2576980378, 3435973837), 3 * ONE_RAW / 2));
    assert!(a.mul_rotation(r).into_inner() == m * r.to_homogeneous());
    assert!(a.div_rotation(r).into_inner() == m * r.inverse().to_homogeneous());
    assert!(a.mul_unit_complex(c).into_inner() == m * c.to_homogeneous());
    assert!(a.mul_translation(t).into_inner() == m * t.to_homogeneous());
    assert!(a.div_translation(t).into_inner() == m * t.inverse().to_homogeneous());
    assert!(a.mul_isometry(iso2()).into_inner() == m * iso2().to_homogeneous());
    assert!(a.mul_similarity(sim).into_inner() == m * sim.to_homogeneous());
    let lhs: Affine2<Fixed> = r.mul_transform(a);
    assert!(lhs.into_inner() == r.to_homogeneous() * m);
    let lhs: Affine2<Fixed> = c.mul_transform(a);
    assert!(lhs.into_inner() == c.to_homogeneous() * m);
    let lhs: Affine2<Fixed> = t.mul_transform(a);
    assert!(lhs.into_inner() == t.to_homogeneous() * m);
    let lhs: Affine2<Fixed> = sim.mul_transform(a);
    assert!(lhs.into_inner() == sim.to_homogeneous() * m);
}
