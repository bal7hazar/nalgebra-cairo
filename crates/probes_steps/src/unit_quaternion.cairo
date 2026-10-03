//! Probes of `UnitQuaternion`: `mul` (compose), `transform_vector`, `from_axis_angle`, `inverse`,
//! `slerp`. Inputs and expected values are those of the gas benches of
//! `nalgebra_tests_geometry::unit_quaternion` (checked against upstream nalgebra there).

use fixed::Fixed;
use nalgebra_core::base::unit::Unit;
use nalgebra_geometry3::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use nalgebra_testing::black_box;
use nalgebra_types3::base::vector3::Vector3;
use crate::builders::{fx, uq, v3};

/// A unit quaternion of negative real part.
fn a() -> UnitQuaternion<Fixed> {
    uq((-1509276477, -2563574020, -2263667719, 2114881862))
}

fn b() -> UnitQuaternion<Fixed> {
    uq((-1829744033, 968283947, 3750500405, -308145672))
}

/// `(1.5, -2.25, 3.75)`.
fn v() -> Vector3<Fixed> {
    v3((0x180000000, -0x240000000, 0x3c0000000))
}

/// `(1.5, -2.25, 3.75)` normalized.
fn axis() -> Unit<Vector3<Fixed>> {
    Unit { value: v3((1393471396, -2090207096, 3483678492)) }
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_mul__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(uq((3349370227, -932498320, -60712365, -2520956970)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_mul__op() {
    let (q, r) = black_box((a(), b()));
    let e = black_box(uq((3349370227, -932498320, -60712365, -2520956970)));
    assert!(q * r == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_transform_vector__baseline() {
    let _i = black_box((a(), v()));
    let e = black_box(v3((-13186805408, -11384226564, -9529255125)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_transform_vector__op() {
    let (q, x) = black_box((a(), v()));
    let e = black_box(v3((-13186805408, -11384226564, -9529255125)));
    assert!(q.transform_vector(x) == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_from_axis_angle__baseline() {
    let _i = black_box((axis(), fx(0x180000000)));
    let e = black_box(uq((3142579763, 949844114, -1424766174, 2374610287)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_from_axis_angle__op() {
    let (u, t) = black_box((axis(), fx(0x180000000)));
    let e = black_box(uq((3142579763, 949844114, -1424766174, 2374610287)));
    assert!(UnitQuaternionAngleTrait::from_axis_angle(u, t) == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_inverse__baseline() {
    let _i = black_box(a());
    let e = black_box(uq((-1509276477, 2563574020, 2263667719, -2114881862)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_inverse__op() {
    let q = black_box(a());
    let e = black_box(uq((-1509276477, 2563574020, 2263667719, -2114881862)));
    assert!(q.inverse() == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_slerp__baseline() {
    let _i = black_box((a(), b(), fx(0x40000000)));
    let e = black_box(uq((-685896195, -2393123539, -2985529558, 1826434634)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_unit_quaternion_slerp__op() {
    let (q, r, t) = black_box((a(), b(), fx(0x40000000)));
    let e = black_box(uq((-685896195, -2393123539, -2985529558, 1826434634)));
    assert!(q.slerp(r, t) == e);
}
