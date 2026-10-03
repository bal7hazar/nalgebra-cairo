//! Probes of `Vector3`: `dot`, `cross`, `norm`, `normalize`. Inputs and expected values are those
//! of the gas benches of `nalgebra_tests_base::vector3` (bit-exact integer model of the kernels).
//!
//! Adding a probe: an `<op>_in()` and an `<op>_out()` function, and the pair of tests below (see
//! docs/STEPS.md).

use fixed::Fixed;
use nalgebra_static3::base::vector3::Vector3Trait;
use nalgebra_testing::black_box;
use nalgebra_types3::base::vector3::Vector3;
use crate::builders::{fx, v3};

fn a() -> Vector3<Fixed> {
    v3((0x180000000, -0x240000000, 0x3c0000000))
}

fn b() -> Vector3<Fixed> {
    v3((-0x480000000, 0x40000000, 0x200000000))
}

#[test]
#[inline(never)]
fn probe_vector3_dot__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(fx(0x30000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_vector3_dot__op() {
    let (x, y) = black_box((a(), b()));
    let e = black_box(fx(0x30000000));
    assert!(x.dot(y) == e);
}

#[test]
#[inline(never)]
fn probe_vector3_cross__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(v3((-0x570000000, -0x13e0000000, -0x9c0000000)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_vector3_cross__op() {
    let (x, y) = black_box((a(), b()));
    let e = black_box(v3((-0x570000000, -0x13e0000000, -0x9c0000000)));
    assert!(x.cross(y) == e);
}

#[test]
#[inline(never)]
fn probe_vector3_norm__baseline() {
    let _i = black_box(a());
    let e = black_box(fx(19856967406));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_vector3_norm__op() {
    let x = black_box(a());
    let e = black_box(fx(19856967406));
    assert!(x.norm() == e);
}

#[test]
#[inline(never)]
fn probe_vector3_normalize__baseline() {
    let _i = black_box(a());
    let e = black_box(v3((1393471397, -2090207095, 3483678492)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_vector3_normalize__op() {
    let x = black_box(a());
    let e = black_box(v3((1393471397, -2090207095, 3483678492)));
    assert!(x.normalize() == e);
}
