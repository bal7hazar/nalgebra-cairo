//! Unit tests of `Orthographic3` and `Matrix4::new_orthographic` (WP 8.4-P11b): the oracle
//! vectors of `tools/oracle` (suite `projections`), exact cases, identities and the panics of
//! upstream's assertions.

use fixed::Fixed;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::point3::Point3Trait;
use nalgebra::geometry::orthographic3::{
    Matrix4OrthographicTrait, Orthographic3, Orthographic3AngleTrait, Orthographic3Trait,
};
use nalgebra_tests_utils::{
    ONE_RAW, fx, int, m4, max_ulp_diff4, max_ulp_diff_v3, ortho6t, p3t, ulp_diff, v3i, v3t,
};
use simba::scalar::Real;
use crate::common::{ortho_err, p3i, report};
use crate::oracle;

/// `Orthographic3::new(-2, 2, -1, 3, 1, 5)`: every entry is exact (`m11 = m22 = 1/2`, `m24 =
/// -1/2`, `m33 = -1/2`, `m34 = -3/2`).
fn o() -> Orthographic3<Fixed> {
    Orthographic3Trait::new(int(-2), int(2), int(-1), int(3), int(1), int(5))
}

/// The matrix of `o()`.
fn o_matrix() -> Matrix4<Fixed> {
    let h = ONE_RAW / 2;
    m4([[h, 0, 0, 0], [0, h, 0, -h], [0, 0, -h, -3 * h], [0, 0, 0, ONE_RAW]])
}

// --- oracle vectors

#[test]
fn test_orthographic3_oracle_new_accessors() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::orthographic3_new_cases();
    while let Some(case) = cases.pop_front() {
        let (l, r, b, t, zn, zf, e, tol) = *case;
        let got: Orthographic3<Fixed> = Orthographic3Trait::new(
            fx(l), fx(r), fx(b), fx(t), fx(zn), fx(zf),
        );
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("new", n, err, tol));
        let m: Matrix4<Fixed> = Matrix4OrthographicTrait::new_orthographic(
            fx(l), fx(r), fx(b), fx(t), fx(zn), fx(zf),
        );
        assert!(m == got.into_inner());
        n += 1;
    }
    let mut cases = oracle::orthographic3_from_fov_cases();
    while let Some(case) = cases.pop_front() {
        let (aspect, vfov, zn, zf, e, tol) = *case;
        let got: Orthographic3<Fixed> = Orthographic3AngleTrait::from_fov(
            fx(aspect), fx(vfov), fx(zn), fx(zf),
        );
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("from_fov", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_accessors_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        let o = ortho6t(x);
        let (l, r, b, t, zn, zf) = e;
        let mut err = ulp_diff(o.left(), fx(l));
        err = core::cmp::max(err, ulp_diff(o.right(), fx(r)));
        err = core::cmp::max(err, ulp_diff(o.bottom(), fx(b)));
        err = core::cmp::max(err, ulp_diff(o.top(), fx(t)));
        err = core::cmp::max(err, ulp_diff(o.znear(), fx(zn)));
        err = core::cmp::max(err, ulp_diff(o.zfar(), fx(zf)));
        worst = core::cmp::max(worst, report("accessors", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        let err = max_ulp_diff4(ortho6t(x).inverse(), m4(e));
        worst = core::cmp::max(worst, report("inverse", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_orthographic3_oracle_projections() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::orthographic3_project_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, pt, e, tol) = *case;
        let got = ortho6t(x).project_point(p3t(pt));
        let err = max_ulp_diff_v3(got.coords(), v3t(e));
        worst = core::cmp::max(worst, report("project_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_project_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let err = max_ulp_diff_v3(ortho6t(x).project_vector(v3t(v)), v3t(e));
        worst = core::cmp::max(worst, report("project_vector", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_unproject_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, pt, e, tol) = *case;
        let got = ortho6t(x).unproject_point(p3t(pt));
        let err = max_ulp_diff_v3(got.coords(), v3t(e));
        worst = core::cmp::max(worst, report("unproject_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_orthographic3_oracle_setters() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::orthographic3_set_left_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_left(fx(v));
        worst = core::cmp::max(worst, report("set_left", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_right_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_right(fx(v));
        worst = core::cmp::max(worst, report("set_right", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_bottom_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_bottom(fx(v));
        worst = core::cmp::max(worst, report("set_bottom", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_top_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_top(fx(v));
        worst = core::cmp::max(worst, report("set_top", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_znear_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_znear(fx(v));
        worst = core::cmp::max(worst, report("set_znear", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_zfar_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_zfar(fx(v));
        worst = core::cmp::max(worst, report("set_zfar", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_left_and_right_cases();
    while let Some(case) = cases.pop_front() {
        let (x, lo, hi, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_left_and_right(fx(lo), fx(hi));
        worst = core::cmp::max(worst, report("set_left_and_right", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_bottom_and_top_cases();
    while let Some(case) = cases.pop_front() {
        let (x, lo, hi, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_bottom_and_top(fx(lo), fx(hi));
        worst = core::cmp::max(worst, report("set_bottom_and_top", n, ortho_err(o, e), tol));
        n += 1;
    }
    let mut cases = oracle::orthographic3_set_znear_and_zfar_cases();
    while let Some(case) = cases.pop_front() {
        let (x, lo, hi, e, tol) = *case;
        let mut o = ortho6t(x);
        o.set_znear_and_zfar(fx(lo), fx(hi));
        worst = core::cmp::max(worst, report("set_znear_and_zfar", n, ortho_err(o, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- construction and conversions

#[test]
fn test_new_exact() {
    assert!(o().into_inner() == o_matrix());
}

#[test]
fn test_new_orthographic_is_new_into_inner() {
    let m: Matrix4<Fixed> = Matrix4OrthographicTrait::new_orthographic(
        int(-2), int(2), int(-1), int(3), int(1), int(5),
    );
    assert!(m == o_matrix());
}

#[test]
fn test_matrix_accessors_agree() {
    let q = o();
    let m = o_matrix();
    assert!(q.as_matrix() == m);
    assert!(q.to_homogeneous() == m);
    assert!(q.unwrap() == m);
    let via_into: Matrix4<Fixed> = q.into();
    assert!(via_into == m);
    assert!(Orthographic3Trait::from_matrix_unchecked(m) == q);
}

#[test]
fn test_partial_eq_and_serde() {
    let a = o();
    let mut b = o();
    assert!(a == b);
    b.set_top(int(4));
    assert!(a != b);
    let mut out: Array<felt252> = array![];
    a.serialize(ref out);
    assert!(out.len() == 16);
    let mut span = out.span();
    let back: Orthographic3<Fixed> = Serde::deserialize(ref span).unwrap();
    assert!(back == a);
}

#[test]
fn test_from_fov_square() {
    // aspect 1, vfov pi/2, znear 2, zfar 4: width = 4 · tan(pi/4) = 4 (within a few ulp), a square
    // of half-width 2.
    let q: Orthographic3<Fixed> = Orthographic3AngleTrait::from_fov(
        int(1), Real::frac_pi_2(), int(2), int(4),
    );
    assert!(ulp_diff(q.left(), int(-2)) <= 8 && ulp_diff(q.right(), int(2)) <= 8);
    assert!(ulp_diff(q.bottom(), int(-2)) <= 8 && ulp_diff(q.top(), int(2)) <= 8);
    assert!(q.znear() == int(2) && q.zfar() == int(4));
    // The negated operand is floored: `right - left` is the width exactly.
    assert!(q.right() - q.left() == q.top() - q.bottom());
}

// --- accessors

#[test]
fn test_accessors_exact() {
    let q = o();
    assert!(q.left() == int(-2) && q.right() == int(2));
    assert!(q.bottom() == int(-1) && q.top() == int(3));
    assert!(q.znear() == int(1) && q.zfar() == int(5));
}

// --- setters

#[test]
fn test_setters_keep_the_other_bound() {
    let mut q = o();
    q.set_left(int(-6));
    assert!(q.left() == int(-6) && q.right() == int(2));
    q.set_right(int(10));
    assert!(q.left() == int(-6) && q.right() == int(10));
    q.set_bottom(int(-5));
    assert!(q.bottom() == int(-5) && q.top() == int(3));
    q.set_top(int(11));
    assert!(q.bottom() == int(-5) && q.top() == int(11));
    q.set_znear(int(-3));
    assert!(q.znear() == int(-3) && q.zfar() == int(5));
    q.set_zfar(int(13));
    assert!(q.znear() == int(-3) && q.zfar() == int(13));
}

#[test]
fn test_set_pairs_exact() {
    let mut q = o();
    q.set_left_and_right(int(-1), int(3));
    q.set_bottom_and_top(int(0), int(8));
    q.set_znear_and_zfar(int(2), int(6));
    let expected: Orthographic3<Fixed> = Orthographic3Trait::new(
        int(-1), int(3), int(0), int(8), int(2), int(6),
    );
    assert!(q == expected);
}

// --- projections

#[test]
fn test_project_point_exact() {
    let q = o();
    // The corners of the view box map to the corners of the unit cube.
    assert!(q.project_point(p3i(-2, -1, -1)) == p3i(-1, -1, -1));
    assert!(q.project_point(p3i(2, 3, -5)) == p3i(1, 1, 1));
    assert!(q.project_point(p3i(0, 1, -3)) == p3i(0, 0, 0));
}

#[test]
fn test_project_vector_exact() {
    assert!(o().project_vector(v3i(2, 4, -2)) == v3i(1, 2, 1));
}

#[test]
fn test_unproject_inverts_project() {
    let q = o();
    let pt = p3i(3, -7, 11);
    assert!(q.unproject_point(q.project_point(pt)) == pt);
    // An odd raw coordinate loses its last bit in the floored product by 1/2.
    let pt = p3t((1234567890, -987654321, -9876543210));
    let back = q.unproject_point(q.project_point(pt));
    assert!(max_ulp_diff_v3(back.coords(), pt.coords()) <= 2);
}

#[test]
fn test_inverse_exact() {
    let q = o();
    let inv = m4(
        [
            [2 * ONE_RAW, 0, 0, 0], [0, 2 * ONE_RAW, 0, ONE_RAW],
            [0, 0, -2 * ONE_RAW, -3 * ONE_RAW], [0, 0, 0, ONE_RAW],
        ],
    );
    assert!(q.inverse() == inv);
    let id = m4([[ONE_RAW, 0, 0, 0], [0, ONE_RAW, 0, 0], [0, 0, ONE_RAW, 0], [0, 0, 0, ONE_RAW]]);
    assert!(q.inverse() * q.into_inner() == id);
}

// --- panics (upstream's assertions)

#[test]
#[should_panic(expected: 'nalgebra: left == right')]
fn test_new_left_equals_right_panics() {
    let _o: Orthographic3<Fixed> = Orthographic3Trait::new(
        int(1), int(1), int(-1), int(3), int(1), int(5),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: bottom == top')]
fn test_new_bottom_equals_top_panics() {
    let _o: Orthographic3<Fixed> = Orthographic3Trait::new(
        int(-2), int(2), int(3), int(3), int(1), int(5),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: superimposed planes')]
fn test_new_superimposed_planes_panics() {
    let _o: Orthographic3<Fixed> = Orthographic3Trait::new(
        int(-2), int(2), int(-1), int(3), int(5), int(5),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: left == right')]
fn test_set_left_to_right_panics() {
    let mut q = o();
    q.set_left(int(2));
}

#[test]
#[should_panic(expected: 'nalgebra: bottom == top')]
fn test_set_top_to_bottom_panics() {
    let mut q = o();
    q.set_top(int(-1));
}

#[test]
#[should_panic(expected: 'nalgebra: superimposed planes')]
fn test_set_zfar_to_znear_panics() {
    let mut q = o();
    q.set_zfar(int(1));
}

#[test]
#[should_panic(expected: 'nalgebra: far plane == near')]
fn test_from_fov_far_equals_near_panics() {
    let _o: Orthographic3<Fixed> = Orthographic3AngleTrait::from_fov(
        int(1), Real::frac_pi_2(), int(4), int(4),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: zero aspect ratio')]
fn test_from_fov_zero_aspect_panics() {
    let _o: Orthographic3<Fixed> = Orthographic3AngleTrait::from_fov(
        fx(-1), Real::frac_pi_2(), int(1), int(4),
    );
}
