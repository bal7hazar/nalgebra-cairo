//! WP 8.4-R: the `Unit` constructors that complete `UnitDualQuaternion` (`new_and_get`,
//! `try_new_and_get`, `unwrap`) with their `None` and panic cases.

use fixed::Fixed;
use nalgebra::geometry::dual_quaternion::DualQuaternion;
use nalgebra::geometry::quaternion::QuaternionTrait;
use nalgebra::geometry::unit_dual_quaternion::UnitDualQuaternionTrait;
use nalgebra_tests_utils::{int, qi};

fn dq() -> DualQuaternion<Fixed> {
    DualQuaternion { real: qi(1, 2, -3, 4), dual: qi(0, 1, 5, -2) }
}

#[test]
fn test_unit_dual_quaternion_new_and_get_try_new_and_get_unwrap() {
    let v = dq();
    let (u, n) = UnitDualQuaternionTrait::<Fixed>::new_and_get(v);
    assert!(u == UnitDualQuaternionTrait::new_normalize(v) && n == v.real.norm());
    assert!(u.unwrap() == u.into_inner() && u.unwrap() == u.dual_quaternion);
    assert!(UnitDualQuaternionTrait::<Fixed>::try_new_and_get(v, int(1)) == Some((u, n)));
    assert!(UnitDualQuaternionTrait::<Fixed>::try_new_and_get(v, int(10)).is_none());
    // `min_norm` is exclusive: a norm equal to it gives `None`.
    assert!(UnitDualQuaternionTrait::<Fixed>::try_new_and_get(v, n).is_none());
    assert!(UnitDualQuaternionTrait::<Fixed>::try_new(v, n).is_none());
    // Exact: real part of norm 5 along i, dual part halved twice over 5.
    let e = DualQuaternion { real: qi(0, 5, 0, 0), dual: qi(0, 0, 10, 0) };
    let (u, n) = UnitDualQuaternionTrait::<Fixed>::new_and_get(e);
    assert!(n == int(5));
    assert!(u.dual_quaternion.real == qi(0, 1, 0, 0) && u.dual_quaternion.dual == qi(0, 0, 2, 0));
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_unit_dual_quaternion_new_and_get_zero_real_part_panics() {
    let e = DualQuaternion { real: qi(0, 0, 0, 0), dual: qi(0, 1, 0, 0) };
    let _ = UnitDualQuaternionTrait::<Fixed>::new_and_get(e);
}
