//! Probes of `Isometry3`: `mul` (compose), `transform_point`, `inverse`, `inv_mul`. Inputs and
//! expected values are those of the gas benches of `nalgebra_tests_geometry::isometry3`.

use fixed::Fixed;
use nalgebra_geometry3::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra_testing::black_box;
use nalgebra_types3::base::point3::Point3;
use crate::builders::{iso3, p3};

/// `new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))`.
fn a() -> Isometry3<Fixed> {
    iso3((6442450944, -9663676416, 16106127360), (4234293283, 534340439, -400755330, 267170219))
}

/// `new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))`.
fn b() -> Isometry3<Fixed> {
    iso3((-3221225472, 2147483648, 5368709120), (3689020097, -1022754606, 767065954, 1789820560))
}

/// `(-2.5, 3.75, 0.75)`.
fn p() -> Point3<Fixed> {
    p3((-0x280000000, 0x3c0000000, 0xc0000000))
}

fn ab() -> Isometry3<Fixed> {
    iso3((2084353234, -9298899151, 21074521374), (3724384862, -764072791, 125720322, 1994013199))
}

fn a_inv() -> Isometry3<Fixed> {
    iso3((-8531988124, 6465531078, -16724271021), (4234293283, -534340439, 400755330, -267170219))
}

fn a_inv_mul_b() -> Isometry3<Fixed> {
    iso3(
        (-10387824139, 10254455918, -11624179020),
        (3549427458, -1252539985, 1386739257, 1535059155),
    )
}

#[test]
#[inline(never)]
fn probe_isometry3_mul__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(ab());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_mul__op() {
    let (x, y) = black_box((a(), b()));
    let e = black_box(ab());
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_transform_point__baseline() {
    let _i = black_box((a(), p()));
    let e = black_box(p3((-6917091138, 3923952211, 20793852411)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_transform_point__op() {
    let (x, q) = black_box((a(), p()));
    let e = black_box(p3((-6917091138, 3923952211, 20793852411)));
    assert!(x.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_inverse__baseline() {
    let _i = black_box(a());
    let e = black_box(a_inv());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_inverse__op() {
    let x = black_box(a());
    let e = black_box(a_inv());
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_inv_mul__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(a_inv_mul_b());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_isometry3_inv_mul__op() {
    let (x, y) = black_box((a(), b()));
    let e = black_box(a_inv_mul_b());
    assert!(x.inv_mul(y) == e);
}
