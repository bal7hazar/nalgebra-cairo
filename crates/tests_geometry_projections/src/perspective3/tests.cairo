//! Unit tests of `Perspective3` and `Matrix4::new_perspective` (WP 8.4-P11b): the oracle vectors
//! of `tools/oracle` (suite `projections`), exact cases, identities and the panics of upstream's
//! assertions.

use fixed::Fixed;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::point3::Point3Trait;
use nalgebra::geometry::perspective3::{
    Matrix4PerspectiveTrait, Perspective3, Perspective3AngleTrait, Perspective3Trait,
};
use nalgebra_tests_utils::{
    ONE_RAW, fx, int, m4, max_ulp_diff4, max_ulp_diff_v3, p3t, pers4t, ulp_diff, v3i, v3t,
};
use simba::scalar::Real;
use crate::common::{p3i, pers_err, report};
use crate::oracle;

/// `Perspective3::new(2, pi/2, 1, 3)`: `m33 = -2`, `m34 = -3` exactly.
fn p() -> Perspective3<Fixed> {
    Perspective3AngleTrait::new(int(2), Real::frac_pi_2(), int(1), int(3))
}

// --- oracle vectors

#[test]
fn test_perspective3_oracle_new_accessors() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::perspective3_new_cases();
    while let Some(case) = cases.pop_front() {
        let (aspect, fovy, znear, zfar, e, tol) = *case;
        let got: Perspective3<Fixed> = Perspective3AngleTrait::new(
            fx(aspect), fx(fovy), fx(znear), fx(zfar),
        );
        let err = max_ulp_diff4(got.into_inner(), m4(e));
        worst = core::cmp::max(worst, report("new", n, err, tol));
        let m: Matrix4<Fixed> = Matrix4PerspectiveTrait::new_perspective(
            fx(aspect), fx(fovy), fx(znear), fx(zfar),
        );
        assert!(m == got.into_inner());
        n += 1;
    }
    let mut cases = oracle::perspective3_accessors_cases();
    while let Some(case) = cases.pop_front() {
        let (x, aspect, fovy, znear, zfar, tol) = *case;
        let p = pers4t(x);
        let mut err = ulp_diff(p.aspect(), fx(aspect));
        err = core::cmp::max(err, ulp_diff(p.fovy(), fx(fovy)));
        err = core::cmp::max(err, ulp_diff(p.znear(), fx(znear)));
        err = core::cmp::max(err, ulp_diff(p.zfar(), fx(zfar)));
        worst = core::cmp::max(worst, report("accessors", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        let err = max_ulp_diff4(pers4t(x).inverse(), m4(e));
        worst = core::cmp::max(worst, report("inverse", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_perspective3_oracle_projections() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::perspective3_project_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, pt, e, tol) = *case;
        let got = pers4t(x).project_point(p3t(pt));
        let err = max_ulp_diff_v3(got.coords(), v3t(e));
        worst = core::cmp::max(worst, report("project_point", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_project_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let err = max_ulp_diff_v3(pers4t(x).project_vector(v3t(v)), v3t(e));
        worst = core::cmp::max(worst, report("project_vector", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_unproject_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, pt, e, tol) = *case;
        let got = pers4t(x).unproject_point(p3t(pt));
        let err = max_ulp_diff_v3(got.coords(), v3t(e));
        worst = core::cmp::max(worst, report("unproject_point", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_perspective3_oracle_setters() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::perspective3_set_aspect_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut p = pers4t(x);
        p.set_aspect(fx(v));
        worst = core::cmp::max(worst, report("set_aspect", n, pers_err(p, e), tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_set_fovy_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut p = pers4t(x);
        p.set_fovy(fx(v));
        worst = core::cmp::max(worst, report("set_fovy", n, pers_err(p, e), tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_set_znear_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut p = pers4t(x);
        p.set_znear(fx(v));
        worst = core::cmp::max(worst, report("set_znear", n, pers_err(p, e), tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_set_zfar_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let mut p = pers4t(x);
        p.set_zfar(fx(v));
        worst = core::cmp::max(worst, report("set_zfar", n, pers_err(p, e), tol));
        n += 1;
    }
    let mut cases = oracle::perspective3_set_znear_and_zfar_cases();
    while let Some(case) = cases.pop_front() {
        let (x, znear, zfar, e, tol) = *case;
        let mut p = pers4t(x);
        p.set_znear_and_zfar(fx(znear), fx(zfar));
        worst = core::cmp::max(worst, report("set_znear_and_zfar", n, pers_err(p, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- construction and conversions

#[test]
fn test_new_exact_depth_entries() {
    let m = p().into_inner();
    // `tan(pi/4)` is 1 within an ulp or two: `m22` is 1 within as much, `m11 = m22 / 2`.
    assert!(ulp_diff(m.m22, int(1)) <= 2);
    assert!(m.m11 == Real::div(m.m22, int(2)));
    assert!(m.m33 == int(-2) && m.m34 == int(-3));
    assert!(m.m43 == int(-1) && m.m44 == int(0));
    let zero = int(0);
    assert!(m.m21 == zero && m.m31 == zero && m.m41 == zero);
    assert!(m.m12 == zero && m.m32 == zero && m.m42 == zero);
    assert!(m.m13 == zero && m.m23 == zero);
    assert!(m.m14 == zero && m.m24 == zero);
}

#[test]
fn test_new_perspective_is_new_into_inner() {
    let m: Matrix4<Fixed> = Matrix4PerspectiveTrait::new_perspective(
        int(2), Real::frac_pi_2(), int(1), int(3),
    );
    assert!(m == p().into_inner());
}

#[test]
fn test_matrix_accessors_agree() {
    let q = p();
    let m = q.into_inner();
    assert!(q.as_matrix() == m);
    assert!(q.to_homogeneous() == m);
    assert!(q.unwrap() == m);
    let via_into: Matrix4<Fixed> = q.into();
    assert!(via_into == m);
    assert!(Perspective3Trait::from_matrix_unchecked(m) == q);
}

#[test]
fn test_partial_eq_and_serde() {
    let a = p();
    let mut b = p();
    assert!(a == b);
    b.set_znear(int(2));
    assert!(a != b);
    let mut out: Array<felt252> = array![];
    a.serialize(ref out);
    assert!(out.len() == 16);
    let mut span = out.span();
    let back: Perspective3<Fixed> = Serde::deserialize(ref span).unwrap();
    assert!(back == a);
}

// --- accessors

#[test]
fn test_znear_zfar_exact() {
    let q = p();
    assert!(q.znear() == int(1));
    assert!(q.zfar() == int(3));
}

#[test]
fn test_aspect_fovy_roundtrip() {
    let q = p();
    assert!(ulp_diff(q.aspect(), int(2)) <= 2);
    let half_pi: Fixed = Real::frac_pi_2();
    assert!(ulp_diff(q.fovy(), half_pi) <= 4);
}

// --- setters

#[test]
fn test_set_znear_and_zfar_exact() {
    let mut q = p();
    q.set_znear_and_zfar(int(1), int(5));
    // m33 = 6 / -4, m34 = 10 / -4.
    let m = q.into_inner();
    assert!(m.m33 == fx(-3 * ONE_RAW / 2) && m.m34 == fx(-5 * ONE_RAW / 2));
    assert!(q.znear() == int(1) && q.zfar() == int(5));
}

#[test]
fn test_set_znear_keeps_zfar() {
    let mut q = p();
    q.set_znear(int(2));
    assert!(q.znear() == int(2) && q.zfar() == int(3));
    let mut r = p();
    r.set_zfar(int(9));
    assert!(r.znear() == int(1) && r.zfar() == int(9));
}

#[test]
fn test_set_aspect_keeps_fovy() {
    let mut q = p();
    let m22 = q.into_inner().m22;
    q.set_aspect(int(4));
    assert!(q.into_inner().m22 == m22);
    assert!(q.into_inner().m11 == Real::div(m22, int(4)));
}

#[test]
fn test_set_fovy_keeps_aspect() {
    let mut q = p();
    let third_pi: Fixed = Real::frac_pi_3();
    q.set_fovy(third_pi);
    // tan(pi/6) = 1/sqrt(3): m22 = sqrt(3).
    assert!(ulp_diff(q.into_inner().m22, Real::sqrt(int(3))) <= 4);
    assert!(ulp_diff(q.aspect(), int(2)) <= 2);
    assert!(ulp_diff(q.fovy(), third_pi) <= 4);
}

// --- projections

#[test]
fn test_project_point_exact() {
    // m11 = 1/2, m22 = 1 (set exactly), znear 1, zfar 3: (2, 4, -2) -> (0.5, 2, 0.5).
    let q = pers4t((ONE_RAW / 2, ONE_RAW, -2 * ONE_RAW, -3 * ONE_RAW));
    let got = q.project_point(p3i(2, 4, -2));
    assert!(got == p3t((ONE_RAW / 2, 2 * ONE_RAW, ONE_RAW / 2)));
    // The near plane maps to -1, the far plane to 1.
    assert!(q.project_point(p3i(0, 0, -1)).z == int(-1));
    assert!(q.project_point(p3i(0, 0, -3)).z == int(1));
}

#[test]
fn test_project_vector_exact() {
    let q = pers4t((ONE_RAW / 2, ONE_RAW, -2 * ONE_RAW, -3 * ONE_RAW));
    // Upstream's third component is `m33` itself.
    assert!(q.project_vector(v3i(2, 4, -2)) == v3t((ONE_RAW / 2, 2 * ONE_RAW, -2 * ONE_RAW)));
}

#[test]
fn test_unproject_inverts_project() {
    let q = pers4t((ONE_RAW / 2, ONE_RAW, -2 * ONE_RAW, -3 * ONE_RAW));
    let pt = p3i(2, 4, -2);
    assert!(q.unproject_point(q.project_point(pt)) == pt);
    let pt = p3t((1234567890, -987654321, -9876543210));
    let back = q.unproject_point(q.project_point(pt));
    assert!(max_ulp_diff_v3(back.coords(), pt.coords()) <= 8);
}

#[test]
fn test_inverse_times_matrix_is_identity() {
    let q = pers4t((ONE_RAW / 2, ONE_RAW, -2 * ONE_RAW, -3 * ONE_RAW));
    let i = q.inverse() * q.into_inner();
    let id = m4([[ONE_RAW, 0, 0, 0], [0, ONE_RAW, 0, 0], [0, 0, ONE_RAW, 0], [0, 0, 0, ONE_RAW]]);
    assert!(max_ulp_diff4(i, id) <= 2);
}

// --- panics (upstream's assertions)

#[test]
#[should_panic(expected: 'nalgebra: superimposed planes')]
fn test_new_superimposed_planes_panics() {
    let _p: Perspective3<Fixed> = Perspective3AngleTrait::new(
        int(1), Real::frac_pi_2(), int(2), int(2),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: superimposed planes')]
fn test_new_planes_within_one_ulp_panics() {
    let _p: Perspective3<Fixed> = Perspective3AngleTrait::new(
        int(1), Real::frac_pi_2(), fx(2 * ONE_RAW), fx(2 * ONE_RAW + 1),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: zero aspect ratio')]
fn test_new_zero_aspect_panics() {
    let _p: Perspective3<Fixed> = Perspective3AngleTrait::new(
        fx(1), Real::frac_pi_2(), int(1), int(3),
    );
}

#[test]
#[should_panic(expected: 'nalgebra: zero aspect ratio')]
fn test_set_aspect_zero_panics() {
    let mut q = p();
    q.set_aspect(int(0));
}

#[test]
#[should_panic]
fn test_project_point_zero_depth_panics() {
    let _q = p().project_point(p3i(1, 1, 0));
}
