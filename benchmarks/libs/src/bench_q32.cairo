//! Scalar Q32.32: cubit `f64::Fixed` (sign-magnitude `u64` + `bool`), orion `FP32x32` (the very
//! same cubit type, reached through orion's generic `FixedTrait<T, MAG>`), reference (`i64`).
//!
//! Every test pays the same fixed overhead (`inputs()` + `check*()`, both representations), so the
//! group baseline is exact for every variant. Inputs: a = 3.5, b = -1.25.

use cubit::f64::types::fixed::{Fixed, FixedTrait};
use harness::black_box;
use orion::numbers::fixed_point::core::FixedTrait as OrionFixedTrait;
// orion's operator impls for FP32x32 (= cubit's Fixed) collide with cubit's own as soon as they are
// imported (E2313 multiple implementations), hence the explicit `fp32x32::FP32x32Mul::mul` calls.
use orion::numbers::fixed_point::implementations::fp32x32::core as fp32x32;
use orion::numbers::{FP32x32, FP32x32Impl};
use crate::reference::{q32, q32_bounded};

const A_MAG: u64 = 15032385536; // 3.5
const B_MAG: u64 = 5368709120; // 1.25
const A: i64 = 15032385536;
const B: i64 = -5368709120;

#[derive(Copy, Drop)]
struct Inputs {
    ca: Fixed,
    cb: Fixed,
    cc: Fixed, // |b|, for same-sign comparisons
    ra: i64,
    rb: i64,
    rc: i64,
    yes: bool,
}

#[inline(never)]
fn inputs() -> Inputs {
    Inputs {
        ca: black_box(FixedTrait::new(A_MAG, false)),
        cb: black_box(FixedTrait::new(B_MAG, true)),
        cc: black_box(FixedTrait::new(B_MAG, false)),
        ra: black_box(A),
        rb: black_box(B),
        rc: black_box(-B),
        yes: black_box(true),
    }
}

#[inline(never)]
fn check(c: Fixed, c_mag: u64, c_sign: bool, r: i64, r_expected: i64) {
    assert!(c == FixedTrait::new(c_mag, c_sign));
    assert!(r == r_expected);
}

#[inline(never)]
fn check_bool(c: bool, r: bool) {
    assert!(c);
    assert!(r);
}

// ---- add --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_add_q32__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q32__cubit() {
    let i = inputs();
    check(i.ca + i.cb, 9663676416, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q32__cubit_same_sign() {
    let i = inputs();
    check(i.ca + i.cc, 20401094656, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q32__orion_fp32x32() {
    let i = inputs();
    let (a, b): (FP32x32, FP32x32) = (i.ca, i.cb);
    check(fp32x32::FP32x32Add::add(a, b), 9663676416, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q32__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra + i.rb, 9663676416);
}

// ---- sub --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_sub_q32__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sub_q32__cubit() {
    let i = inputs();
    check(i.ca - i.cb, 20401094656, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sub_q32__cubit_same_sign() {
    let i = inputs();
    check(i.ca - i.cc, 9663676416, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sub_q32__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra - i.rb, 20401094656);
}

// ---- mul --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_mul_q32__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q32__cubit() {
    let i = inputs();
    check(i.ca * i.cb, 18790481920, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q32__orion_fp32x32() {
    let i = inputs();
    let (a, b): (FP32x32, FP32x32) = (i.ca, i.cb);
    check(fp32x32::FP32x32Mul::mul(a, b), 18790481920, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q32__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, q32::mul(i.ra, i.rb), -18790481920);
}

/// The BoundedInt biased-floor kernel from `benchmarks/primitives` (the one nalgebra.cairo ships).
#[test]
#[inline(never)]
fn bench_mul_q32__reference_bounded() {
    let i = inputs();
    check(i.ca, A_MAG, false, q32_bounded::mul(i.ra, i.rb), -18790481920);
}

// ---- div --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_div_q32__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q32__cubit() {
    let i = inputs();
    check(i.ca / i.cb, 12025908428, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q32__orion_fp32x32() {
    let i = inputs();
    let (a, b): (FP32x32, FP32x32) = (i.ca, i.cb);
    check(fp32x32::FP32x32Div::div(a, b), 12025908428, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q32__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, q32::div(i.ra, i.rb), -12025908428);
}

// ---- sqrt -------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_sqrt_q32__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sqrt_q32__cubit() {
    let i = inputs();
    check(FixedTrait::sqrt(i.ca), 8035148054, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sqrt_q32__orion_fp32x32() {
    let i = inputs();
    check(OrionFixedTrait::<FP32x32, u64>::sqrt(i.ca), 8035148054, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sqrt_q32__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, q32::sqrt(i.ra), 8035148054);
}

// ---- neg --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_neg_q32__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_neg_q32__cubit() {
    let i = inputs();
    check(-i.ca, A_MAG, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_neg_q32__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, -i.ra, -A);
}

// ---- comparisons ------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_lt_q32__baseline() {
    let i = inputs();
    check_bool(i.yes, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q32__cubit() {
    let i = inputs();
    check_bool(i.cb < i.ca, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q32__cubit_same_sign() {
    let i = inputs();
    check_bool(i.cc < i.ca, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q32__reference() {
    let i = inputs();
    check_bool(i.yes, i.rb < i.ra);
}

#[test]
#[inline(never)]
fn bench_lt_q32__reference_same_sign() {
    let i = inputs();
    check_bool(i.yes, i.rc < i.ra);
}

#[test]
#[inline(never)]
fn bench_eq_q32__baseline() {
    let i = inputs();
    check_bool(i.yes, i.yes);
}

#[test]
#[inline(never)]
fn bench_eq_q32__cubit() {
    let i = inputs();
    check_bool(i.ca != i.cc, i.yes);
}

#[test]
#[inline(never)]
fn bench_eq_q32__reference() {
    let i = inputs();
    check_bool(i.yes, i.ra != i.rc);
}

// ---- findings ---------------------------------------------------------------------------------

/// Sign-magnitude has two zeros and cubit does not normalise: 0 * -1.25 is "negative zero", which
/// compares different from zero.
#[test]
fn finding_cubit_negative_zero() {
    let i = inputs();
    let zero: Fixed = black_box(FixedTrait::ZERO());
    let product = zero * i.cb;
    assert!(product.mag == 0 && product.sign);
    assert!(product != FixedTrait::ZERO());
}
