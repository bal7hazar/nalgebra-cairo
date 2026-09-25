//! Gas benchmarks of the WP 8.4-R in-place forms of `Quaternion` that have a body of their own
//! (`bench_quaternion_<op>__<variant>`, net = raw - `baseline` of the group): each `alt_by_value`
//! is the by-value form the in-place one must match bit for bit and cost about the same as. The
//! other in-place forms (`inverse_mut`, `conjugate_mut`, `*=`, `/=` of the rotations, the
//! scaling mutators of the similarities) are `self = self.op(..)`.

use fixed::Fixed;
use nalgebra::geometry::quaternion::{Quaternion, QuaternionTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, int, q};

/// `1 + 2i - 3j + 4k`, of squared norm 30.
fn a() -> Quaternion<Fixed> {
    q(ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW, 4 * ONE_RAW)
}

/// `-2 + i + 5j - k`, of squared norm 31.
fn b() -> Quaternion<Fixed> {
    q(-2 * ONE_RAW, ONE_RAW, 5 * ONE_RAW, -ONE_RAW)
}

#[test]
#[inline(never)]
fn bench_quaternion_normalize_mut__baseline() {
    let (_a, _b) = (black_box(a()), black_box(b()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_quaternion_normalize_mut__normalize_mut() {
    let mut x = black_box(a());
    let _n = x.normalize_mut();
    assert!(x.w != int(0));
}

#[test]
#[inline(never)]
fn bench_quaternion_normalize_mut__alt_by_value() {
    let x = black_box(a());
    let y = x.normalize();
    assert!(y.w != int(0));
}

#[test]
#[inline(never)]
fn bench_quaternion_try_inverse_mut__baseline() {
    let (_a, _b) = (black_box(a()), black_box(b()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_quaternion_try_inverse_mut__try_inverse_mut() {
    let mut x = black_box(a());
    assert!(x.try_inverse_mut());
}

#[test]
#[inline(never)]
fn bench_quaternion_try_inverse_mut__alt_by_value() {
    let x = black_box(a());
    assert!(x.try_inverse().is_some());
}

#[test]
#[inline(never)]
fn bench_quaternion_mul_assign__baseline() {
    let (_a, _b) = (black_box(a()), black_box(b()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_quaternion_mul_assign__mul_assign() {
    let mut x = black_box(a());
    x *= black_box(b());
    assert!(x.w != int(0));
}

#[test]
#[inline(never)]
fn bench_quaternion_mul_assign__alt_by_value() {
    let x = black_box(a()) * black_box(b());
    assert!(x.w != int(0));
}

#[test]
#[inline(never)]
fn bench_quaternion_div_assign__baseline() {
    let (_a, _b) = (black_box(a()), black_box(b()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_quaternion_div_assign__div_assign() {
    let mut x = black_box(a());
    x /= black_box(int(3));
    assert!(x.w != int(0));
}

#[test]
#[inline(never)]
fn bench_quaternion_div_assign__alt_by_value() {
    let x = black_box(a()).unscale(black_box(int(3)));
    assert!(x.w != int(0));
}
