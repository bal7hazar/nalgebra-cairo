//! Gas benchmarks of `DualQuaternion` (WP 8.4-P12) (`bench_<group>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementation that lost (`alt_*`, AGENTS.md
//! rule 8): upstream's literal dual part `a.real * b.dual + a.dual * b.real` (two rounded Hamilton
//! products and a checked sum) against the fused eight-product accumulation of `Mul`.
//!
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::geometry::dual_quaternion::{DualQuaternion, DualQuaternionTrait};
use nalgebra::geometry::quaternion::QuaternionTrait;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, fx};
use crate::common::dqt;

/// `(1 + 2i - 3j + 0.5k) + ε(-1.5 + 0.25i + 4j - 2k)` times `0x5a5a5a5a / 2^32`.
fn x() -> DualQuaternion<Fixed> {
    dqt(
        (
            (1515870810, 3031741620, -4547612430, 757935405),
            (-2273806215, 378967702, 6063483240, -3031741620),
        ),
    )
}

/// `(-2 + 0.5i + j - 1k) + ε(3 - 1i + 0.75j + 2k)` times `0x3c3c3c3c / 2^32`.
fn y() -> DualQuaternion<Fixed> {
    dqt(
        (
            (-2021161080, 505290270, 1010580540, -1010580540),
            (3031741620, -1010580540, 757935405, 2021161080),
        ),
    )
}

// --- alternative implementation (loser)

/// The LOSER of `Mul`: upstream's literal dual part, two Hamilton products rounded separately and
/// a checked sum (up to 2 ulp from the exact floor the fused kernel returns).
#[inline(always)]
fn alt_mul(a: DualQuaternion<Fixed>, b: DualQuaternion<Fixed>) -> DualQuaternion<Fixed> {
    DualQuaternion { real: a.real * b.real, dual: a.real * b.dual + a.dual * b.real }
}

#[test]
fn test_mul_alt_two_products_agrees() {
    let (f, l) = (x() * y(), alt_mul(x(), y()));
    assert!(f.real == l.real);
    assert!(f.dual.abs_diff_eq(l.dual, 2));
    // The fused kernel is the exact floor: here the literal form is 1 ulp below it.
    assert!(f.dual != l.dual);
}

// --- benchmarks

#[test]
#[inline(never)]
fn bench_dual_quaternion_mul__baseline() {
    let _a: DualQuaternion<Fixed> = black_box(x());
    let _b: DualQuaternion<Fixed> = black_box(y());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (178337742, -356675485, 3299248232, 535013226),
                (1114610889, -1649624117, -8203536147, 2050884036),
            ),
        ),
    );
    assert!(e == e);
}

/// The fused product (winner).
#[test]
#[inline(never)]
fn bench_dual_quaternion_mul__fused() {
    let a: DualQuaternion<Fixed> = black_box(x());
    let b: DualQuaternion<Fixed> = black_box(y());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (178337742, -356675485, 3299248232, 535013226),
                (1114610889, -1649624117, -8203536147, 2050884036),
            ),
        ),
    );
    assert!(a * b == e);
}

/// Upstream's literal form (loser).
#[test]
#[inline(never)]
fn bench_dual_quaternion_mul__alt_two_products() {
    let a: DualQuaternion<Fixed> = black_box(x());
    let b: DualQuaternion<Fixed> = black_box(y());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (178337742, -356675485, 3299248232, 535013226),
                (1114610888, -1649624117, -8203536147, 2050884036),
            ),
        ),
    );
    assert!(alt_mul(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_dual_quaternion_normalize__baseline() {
    let _a: DualQuaternion<Fixed> = black_box(x());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (1137764631, 2275529263, -3413293894, 568882316),
                (-1706646947, 284441157, 4551058525, -2275529263),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_dual_quaternion_normalize__normalize() {
    let a: DualQuaternion<Fixed> = black_box(x());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (1137764631, 2275529263, -3413293894, 568882316),
                (-1706646947, 284441157, 4551058525, -2275529263),
            ),
        ),
    );
    assert!(a.normalize() == e);
}

#[test]
#[inline(never)]
fn bench_dual_quaternion_try_inverse__baseline() {
    let _a: DualQuaternion<Fixed> = black_box(x());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (853970106, -1707940212, 2561910318, -426985053),
                (397021189, -3569445224, 1618048622, 868952037),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_dual_quaternion_try_inverse__try_inverse() {
    let a: DualQuaternion<Fixed> = black_box(x());
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (853970106, -1707940212, 2561910318, -426985053),
                (397021189, -3569445224, 1618048622, 868952037),
            ),
        ),
    );
    assert!(a.try_inverse().unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_dual_quaternion_lerp__baseline() {
    let _a: DualQuaternion<Fixed> = black_box(x());
    let _b: DualQuaternion<Fixed> = black_box(y());
    let _t: Fixed = black_box(fx(ONE_RAW / 3));
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (336860180, 2189591170, -2694881441, 168430090),
                (-505290271, -84215046, 4294967295, -1347440721),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_dual_quaternion_lerp__lerp() {
    let a: DualQuaternion<Fixed> = black_box(x());
    let b: DualQuaternion<Fixed> = black_box(y());
    let t: Fixed = black_box(fx(ONE_RAW / 3));
    let e: DualQuaternion<Fixed> = black_box(
        dqt(
            (
                (336860180, 2189591170, -2694881441, 168430090),
                (-505290271, -84215046, 4294967295, -1347440721),
            ),
        ),
    );
    assert!(a.lerp(b, t) == e);
}
