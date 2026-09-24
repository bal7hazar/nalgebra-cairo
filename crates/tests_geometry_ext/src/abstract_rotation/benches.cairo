//! Gas benchmarks of `AbstractRotation` (`bench_abstract_rotation_<op>__<variant>`): the trait
//! only forwards, so a call through it must cost what the direct call costs.

use fixed::Fixed;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::abstract_rotation::AbstractRotation;
use nalgebra::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{uq, v3};

fn q() -> UnitQuaternion<Fixed> {
    uq(0x80000000, 0x80000000, 0x80000000, 0x80000000)
}

fn v() -> Vector3<Fixed> {
    v3(0x100000000, 0x200000000, -0x300000000)
}

#[test]
#[inline(never)]
fn bench_abstract_rotation_transform_vector__baseline() {
    let (r, x) = (black_box(q()), black_box(v()));
    let e = black_box(UnitQuaternionTrait::transform_vector(q(), v()));
    assert!(r == r && x == x && e == e);
}

#[test]
#[inline(never)]
fn bench_abstract_rotation_transform_vector__direct() {
    let (r, x) = (black_box(q()), black_box(v()));
    let e = black_box(UnitQuaternionTrait::transform_vector(q(), v()));
    assert!(UnitQuaternionTrait::transform_vector(r, x) == e);
}

#[test]
#[inline(never)]
fn bench_abstract_rotation_transform_vector__trait() {
    let (r, x) = (black_box(q()), black_box(v()));
    let e = black_box(UnitQuaternionTrait::transform_vector(q(), v()));
    assert!(AbstractRotation::transform_vector(r, x) == e);
}
