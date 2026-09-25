//! Unit tests of `Projective2/3` and of the general `Transform2/3` (WP 8.4-P11a): the oracle
//! vectors of `tools/oracle` (suite `transform`), the homogeneous division of the transforms, the
//! products and quotients between categories, the singular cases and the panics.

use fixed::Fixed;
use nalgebra::base::cg::{Matrix3CgTrait, Matrix4CgTrait};
use nalgebra::base::matrix3::Matrix3Trait;
use nalgebra::base::matrix4::Matrix4Trait;
use nalgebra::geometry::projective2::{Projective2, Projective2Trait};
use nalgebra::geometry::projective3::{Projective3, Projective3Trait};
use nalgebra::geometry::transform::{TransformDiv, TransformMul};
use nalgebra::geometry::transform2::{Transform2, Transform2Trait};
use nalgebra::geometry::transform3::{Transform3, Transform3Trait};
use nalgebra_tests_utils::{
    ONE_RAW, aff3t, m3, m4, m4i, max_ulp_diff3, max_ulp_diff4, max_ulp_diff_v3, p2t, p3t, proj2,
    proj3, v2t, v3t,
};
use crate::common::{p3_err, p3i, report};
use crate::oracle;

/// A projective map with a non-trivial last row: `[[2, 0, 1, 0], [0, 1, 0, -1], [1, 0, 3, 0],
/// [0, 1, 0, 2]]` (determinant 10).
fn p3() -> Projective3<Fixed> {
    Projective3Trait::from_matrix_unchecked(
        m4i([[2, 0, 1, 0], [0, 1, 0, -1], [1, 0, 3, 0], [0, 1, 0, 2]]),
    )
}

/// A singular general transform (rank 3).
fn singular3() -> Transform3<Fixed> {
    Transform3Trait::from_matrix_unchecked(
        m4i([[1, 2, 0, 0], [2, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )
}

// --- oracle vectors

#[test]
fn test_projective_oracle_try_inverse() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::projective2_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let got = proj2(m).try_inverse().unwrap().into_inner();
        worst =
            core::cmp::max(
                worst, report("projective2_try_inverse", n, max_ulp_diff3(got, m3(e)), tol),
            );
        // The general category inverts the same way.
        let g: Transform2<Fixed> = Transform2Trait::from_matrix_unchecked(m3(m));
        assert!(g.try_inverse().unwrap().into_inner() == got);
        n += 1;
    }
    let mut cases = oracle::projective3_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let got = proj3(m).try_inverse().unwrap().into_inner();
        worst =
            core::cmp::max(
                worst, report("projective3_try_inverse", n, max_ulp_diff4(got, m4(e)), tol),
            );
        let g: Transform3<Fixed> = Transform3Trait::from_matrix_unchecked(m4(m));
        assert!(g.try_inverse().unwrap().into_inner() == got);
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_projective_oracle_transforms() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::projective3_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (m, p, e, tol) = *case;
        let got = proj3(m).transform_point(p3t(p));
        worst =
            core::cmp::max(worst, report("projective3_transform_point", n, p3_err(got, e), tol));
        let g: Transform3<Fixed> = Transform3Trait::from_matrix_unchecked(m4(m));
        assert!(g.transform_point(p3t(p)) == got);
        n += 1;
    }
    let mut cases = oracle::projective3_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (m, v, e, tol) = *case;
        let got = proj3(m).transform_vector(v3t(v));
        let err = max_ulp_diff_v3(got, v3t(e));
        worst = core::cmp::max(worst, report("projective3_transform_vector", n, err, tol));
        let g: Transform3<Fixed> = Transform3Trait::from_matrix_unchecked(m4(m));
        assert!(g.transform_vector(v3t(v)) == got);
        n += 1;
    }
    let mut cases = oracle::projective3_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (m, p, e, tol) = *case;
        let got = proj3(m).inverse_transform_point(p3t(p));
        let err = p3_err(got, e);
        worst = core::cmp::max(worst, report("projective3_inverse_transform_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_projective_oracle_products() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::projective3_mul_affine3_cases();
    while let Some(case) = cases.pop_front() {
        let (m, l, t, e, tol) = *case;
        let got: Projective3<Fixed> = proj3(m).mul_transform(aff3t(l, t));
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("projective3_mul_affine3", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_div_projective3_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, m, e, tol) = *case;
        let got: Projective3<Fixed> = aff3t(l, t).div_transform(proj3(m));
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("affine3_div_projective3", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- exact cases

#[test]
fn test_projective3_transform_point_divides() {
    // (2 + 3, 2 - 1, 1 + 9) / (2 + 2) for p = (1, 2, 3): the normalizer m[3, :] . (p, 1) = 4.
    let got = p3().transform_point(p3i(1, 2, 3));
    assert!(got == p3t((5 * ONE_RAW / 4, ONE_RAW / 4, 10 * ONE_RAW / 4)));
    assert!(got == Matrix4CgTrait::transform_point(p3().into_inner(), p3i(1, 2, 3)));
    // A zero normalizer skips the division (upstream's branch): m[3, :] . (0, -2, 0, 1) = 0.
    assert!(p3().transform_point(p3i(0, -2, 0)) == p3i(0, -3, 0));
    // transform_vector: n = m[3, :3] . v = 2 for v = (0, 2, 0): m[:3, :3] * (v / 2).
    assert!(p3().transform_vector(v3t((0, 2 * ONE_RAW, 0))) == v3t((0, ONE_RAW, 0)));
}

#[test]
fn test_projective3_inverse() {
    let inv = p3().inverse();
    assert!(max_ulp_diff4((p3() * inv).into_inner(), Matrix4Trait::identity()) <= 2);
    assert!(inv == p3().try_inverse().unwrap());
    let mut q = p3();
    q.inverse_mut();
    assert!(q == inv);
    let mut r = p3();
    assert!(r.try_inverse_mut() && r == inv);
    let back = p3().inverse_transform_point(p3().transform_point(p3i(1, 2, 3)));
    // The inverse of a determinant-10 matrix is inexact: 8 ulp here.
    assert!(p3_err(back, (ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)) <= 16);
    let v = v3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    // Upstream's definition (the projective vector map is not inverted by the inverse's).
    assert!(p3().inverse_transform_vector(v) == inv.transform_vector(v));
    // `p / p` is `p * p⁻¹`.
    assert!((p3() / p3()).into_inner() == (p3() * inv).into_inner());
}

#[test]
fn test_transform3_singular() {
    let s = singular3();
    assert!(s.try_inverse().is_none());
    let mut t = s;
    assert!(!t.try_inverse_mut() && t == s);
    // A general transform still transforms (no inverse needed).
    assert!(s.transform_point(p3i(1, 1, 1)) == p3i(3, 6, 1));
}

#[test]
#[should_panic(expected: 'nalgebra: not invertible')]
fn test_projective3_inverse_singular_panics() {
    let _i = Projective3Trait::from_matrix_unchecked(singular3().into_inner()).inverse();
}

#[test]
#[should_panic(expected: 'nalgebra: not invertible')]
fn test_projective2_div_singular_panics() {
    let s = proj2([[ONE_RAW, 2 * ONE_RAW, 0], [2 * ONE_RAW, 4 * ONE_RAW, 0], [0, 0, ONE_RAW]]);
    let _q = proj2([[ONE_RAW, 0, 0], [0, ONE_RAW, 0], [0, 0, ONE_RAW]]) / s;
}

#[test]
#[should_panic(expected: 'nalgebra: not invertible')]
fn test_transform3_div_transform_singular_panics() {
    let s: Projective3<Fixed> = Projective3Trait::from_matrix_unchecked(singular3().into_inner());
    let _q: Transform3<Fixed> = singular3().div_transform(s);
}

// --- 2D

#[test]
fn test_projective2_transforms() {
    let p = proj2([[2 * ONE_RAW, 0, ONE_RAW], [0, ONE_RAW, 0], [0, ONE_RAW, 2 * ONE_RAW]]);
    // q = (2 + 1, 2) = (3, 2), n = 2 + 2 = 4 for pt = (1, 2).
    assert!(p.transform_point(p2t((ONE_RAW, 2 * ONE_RAW))) == p2t((3 * ONE_RAW / 4, ONE_RAW / 2)));
    assert!(
        p
            .transform_point(
                p2t((ONE_RAW, 2 * ONE_RAW)),
            ) == Matrix3CgTrait::transform_point(p.into_inner(), p2t((ONE_RAW, 2 * ONE_RAW))),
    );
    assert!(p.transform_vector(v2t((0, 2 * ONE_RAW))) == v2t((0, ONE_RAW)));
    let back = p.inverse_transform_point(p.transform_point(p2t((ONE_RAW, 2 * ONE_RAW))));
    assert!(back == p2t((ONE_RAW, 2 * ONE_RAW)));
    assert!(max_ulp_diff3((p * p.inverse()).into_inner(), Matrix3Trait::identity()) <= 2);
    let t: Transform2<Fixed> = Transform2Trait::from_matrix_unchecked(p.into_inner());
    assert!((t * t).into_inner() == (p * p).into_inner());
    let q: Projective2<Fixed> = Projective2Trait::identity();
    assert!(q.inverse() == q && q.inverse_transform_vector(v2t((ONE_RAW, 0))) == v2t((ONE_RAW, 0)));
}
