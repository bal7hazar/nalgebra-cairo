//! Gas benchmarks of `Perspective3` (WP 8.4-P11b) (`bench_<group>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementations (`alt_*`, AGENTS.md rule 8).
//!
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::point3::Point3;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::perspective3::{Perspective3, Perspective3AngleTrait, Perspective3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, p3t, pers4t, v3t};
use simba::scalar::Real;

/// `(m11, m22, m33, m34)` ≈ `(1.3, 1.733, -1.025, -0.2025)`.
fn q() -> Perspective3<Fixed> {
    pers4t((5583457485, 7443676160, -4402341478, -869730079))
}

/// `(1.5, -0.75, -3.2)`.
fn pt() -> Point3<Fixed> {
    p3t((6442450944, -3221225472, -13743895347))
}

// --- alternative implementations

/// Upstream's literal `project_point`: `inverse_denom = -1 / z`, then floored products by it
/// (two roundings per coordinate).
#[inline(always)]
fn alt_project_point_recip_mul(q: Perspective3<Fixed>, p: Point3<Fixed>) -> Point3<Fixed> {
    let m = q.into_inner();
    let inv = Real::div(fx(-0x100000000), p.z);
    Point3 { x: m.m11 * p.x * inv, y: m.m22 * p.y * inv, z: (m.m33 * p.z + m.m34) * inv }
}

/// `project_point` with three separate quotients (bit-identical to the prepared divisor).
#[inline(always)]
fn alt_project_point_three_div(q: Perspective3<Fixed>, p: Point3<Fixed>) -> Point3<Fixed> {
    let m = q.into_inner();
    let d = -p.z;
    Point3 {
        x: Real::div(m.m11 * p.x, d),
        y: Real::div(m.m22 * p.y, d),
        z: Real::div(Real::mul_add(m.m33, p.z, m.m34), d),
    }
}

/// `project_vector` with a prepared divisor (`div3`, the third quotient unused).
#[inline(always)]
fn alt_project_vector_div3(q: Perspective3<Fixed>, v: Vector3<Fixed>) -> Vector3<Fixed> {
    let m = q.into_inner();
    let (x, y, _) = Real::div3(m.m11 * v.x, m.m22 * v.y, fx(0), -v.z);
    Vector3 { x, y, z: m.m33 }
}

/// Upstream's literal `znear`: `m34 / (2 · ratio) - m34 / 2`, `ratio = (1 - m33) / (-m33 - 1)`.
#[inline(always)]
fn alt_znear_upstream(q: Perspective3<Fixed>) -> Fixed {
    let m = q.into_inner();
    let one = fx(0x100000000);
    let ratio = Real::div(one - m.m33, -m.m33 - one);
    Real::div(m.m34, ratio + ratio) - Real::div(m.m34, fx(0x200000000))
}


#[test]
fn test_project_point_alts_agree() {
    let (f, t) = (q().project_point(pt()), alt_project_point_three_div(q(), pt()));
    assert!(f == t);
    // Upstream's literal form rounds twice: here 1 ulp below on `x`.
    let l = alt_project_point_recip_mul(q(), pt());
    assert!(l.x != f.x && (f.x - l.x).raw == 1 && l.y == f.y && l.z == f.z);
}

#[test]
fn test_project_vector_alt_agrees() {
    let v = v3t((6442450944, -3221225472, -13743895347));
    assert!(q().project_vector(v) == alt_project_vector_div3(q(), v));
}

#[test]
fn test_znear_alt_upstream_within_one_ulp() {
    assert!((alt_znear_upstream(q()) - q().znear()).raw == 1);
}

// --- benchmarks

#[test]
#[inline(never)]
fn bench_perspective3_new__baseline() {
    let _aspect: Fixed = black_box(fx(7635497415));
    let _fovy: Fixed = black_box(fx(4497880316));
    let _zn: Fixed = black_box(fx(429496730));
    let _zf: Fixed = black_box(fx(429496729600));
    let e: (Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(4184268427), fx(7438699426), fx(-4303565829), fx(-859853313)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_new__new() {
    let aspect: Fixed = black_box(fx(7635497415));
    let fovy: Fixed = black_box(fx(4497880316));
    let zn: Fixed = black_box(fx(429496730));
    let zf: Fixed = black_box(fx(429496729600));
    let e: (Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(4184268427), fx(7438699426), fx(-4303565829), fx(-859853313)),
    );
    let p: Perspective3<Fixed> = Perspective3AngleTrait::new(aspect, fovy, zn, zf);
    let m = p.into_inner();
    let (a, b, c, d) = e;
    assert!(m.m11 == a && m.m22 == b && m.m33 == c && m.m34 == d);
}

#[test]
#[inline(never)]
fn bench_perspective3_project_point__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let _p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((2617245696, -1744611600, 4130550828)));
    assert!(e == e);
}

/// The prepared divisor (winner).
#[test]
#[inline(never)]
fn bench_perspective3_project_point__div3() {
    let q: Perspective3<Fixed> = black_box(q());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((2617245696, -1744611600, 4130550828)));
    assert!(q.project_point(p) == e);
}

/// Three separate quotients (same bits).
#[test]
#[inline(never)]
fn bench_perspective3_project_point__alt_three_div() {
    let q: Perspective3<Fixed> = black_box(q());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((2617245696, -1744611600, 4130550828)));
    assert!(alt_project_point_three_div(q, p) == e);
}

/// Upstream's literal form (two roundings).
#[test]
#[inline(never)]
fn bench_perspective3_project_point__alt_recip_mul() {
    let q: Perspective3<Fixed> = black_box(q());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3t((2617245695, -1744611600, 4130550828)));
    assert!(alt_project_point_recip_mul(q, p) == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_project_vector__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let _v: Vector3<Fixed> = black_box(v3t((6442450944, -3221225472, -13743895347)));
    let e: Vector3<Fixed> = black_box(v3t((2617245696, -1744611600, -4402341478)));
    assert!(e == e);
}

/// Two separate quotients (winner).
#[test]
#[inline(never)]
fn bench_perspective3_project_vector__two_div() {
    let q: Perspective3<Fixed> = black_box(q());
    let v: Vector3<Fixed> = black_box(v3t((6442450944, -3221225472, -13743895347)));
    let e: Vector3<Fixed> = black_box(v3t((2617245696, -1744611600, -4402341478)));
    assert!(q.project_vector(v) == e);
}

/// A prepared divisor for two quotients.
#[test]
#[inline(never)]
fn bench_perspective3_project_vector__alt_div3() {
    let q: Perspective3<Fixed> = black_box(q());
    let v: Vector3<Fixed> = black_box(v3t((6442450944, -3221225472, -13743895347)));
    let e: Vector3<Fixed> = black_box(v3t((2617245696, -1744611600, -4402341478)));
    assert!(alt_project_vector_div3(q, v) == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_unproject_point__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let _p: Point3<Fixed> = black_box(p3t((1717986918, -858993459, 3435973837)));
    let e: Point3<Fixed> = black_box(p3t((1189374467, -446071379, -3865467020)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_unproject_point__unproject() {
    let q: Perspective3<Fixed> = black_box(q());
    let p: Point3<Fixed> = black_box(p3t((1717986918, -858993459, 3435973837)));
    let e: Point3<Fixed> = black_box(p3t((1189374467, -446071379, -3865467020)));
    assert!(q.unproject_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_inverse__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let e: (Fixed, Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(3303820997), fx(2478176599), fx(-4294967296), fx(-21209734513), fx(21739977874)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_inverse__inverse() {
    let q: Perspective3<Fixed> = black_box(q());
    let e: (Fixed, Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(3303820997), fx(2478176599), fx(-4294967296), fx(-21209734513), fx(21739977874)),
    );
    let i = q.inverse();
    let (a, b, c, d, f) = e;
    assert!(i.m11 == a && i.m22 == b && i.m34 == c && i.m43 == d && i.m44 == f);
}

#[test]
#[inline(never)]
fn bench_perspective3_znear__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let e: Fixed = black_box(fx(429496335));
    assert!(e == e);
}

/// One quotient (winner).
#[test]
#[inline(never)]
fn bench_perspective3_znear__one_div() {
    let q: Perspective3<Fixed> = black_box(q());
    let e: Fixed = black_box(fx(429496335));
    assert!(q.znear() == e);
}

/// Upstream's literal form (four roundings).
#[test]
#[inline(never)]
fn bench_perspective3_znear__alt_upstream() {
    let q: Perspective3<Fixed> = black_box(q());
    let e: Fixed = black_box(fx(429496336));
    assert!(alt_znear_upstream(q) == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_fovy__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let e: Fixed = black_box(fx(4495392996));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_fovy__fovy() {
    let q: Perspective3<Fixed> = black_box(q());
    let e: Fixed = black_box(fx(4495392996));
    assert!(q.fovy() == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_set_fovy__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let _fovy: Fixed = black_box(fx(4497880316));
    let e: (Fixed, Fixed) = black_box((fx(5579724466), fx(7438699426)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_set_fovy__set_fovy() {
    let mut q: Perspective3<Fixed> = black_box(q());
    let fovy: Fixed = black_box(fx(4497880316));
    let e: (Fixed, Fixed) = black_box((fx(5579724466), fx(7438699426)));
    q.set_fovy(fovy);
    let (a, b) = e;
    assert!(q.into_inner().m11 == a && q.into_inner().m22 == b);
}

#[test]
#[inline(never)]
fn bench_perspective3_set_znear_and_zfar__baseline() {
    let _q: Perspective3<Fixed> = black_box(q());
    let _zn: Fixed = black_box(fx(429496730));
    let _zf: Fixed = black_box(fx(429496729600));
    let e: (Fixed, Fixed) = black_box((fx(-4303565829), fx(-859853313)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_perspective3_set_znear_and_zfar__set() {
    let mut q: Perspective3<Fixed> = black_box(q());
    let zn: Fixed = black_box(fx(429496730));
    let zf: Fixed = black_box(fx(429496729600));
    let e: (Fixed, Fixed) = black_box((fx(-4303565829), fx(-859853313)));
    q.set_znear_and_zfar(zn, zf);
    let (a, b) = e;
    assert!(q.into_inner().m33 == a && q.into_inner().m34 == b);
}
