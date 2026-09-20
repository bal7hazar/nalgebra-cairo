//! Scalar Q64.64: cubit `f128::Fixed` (sign-magnitude `u128` + `bool`; orion's `FP64x64` is the
//! same type), reference (`i128`, two `u128` limbs for the wide product).
//!
//! Same fixed overhead in every test (see bench_q32). Inputs: a = 3.5, b = -1.25.

use cubit::f128::types::fixed::{Fixed, FixedTrait};
use harness::black_box;
use orion::numbers::fixed_point::implementations::fp64x64::core as fp64x64;
use orion::numbers::{FP64x64, FP64x64Impl};
use crate::reference::q64;

const A_MAG: u128 = 64563604257983430656; // 3.5
const B_MAG: u128 = 23058430092136939520; // 1.25
const A: i128 = 64563604257983430656;
const B: i128 = -23058430092136939520;

#[derive(Copy, Drop)]
struct Inputs {
    ca: Fixed,
    cb: Fixed,
    ra: i128,
    rb: i128,
    yes: bool,
}

#[inline(never)]
fn inputs() -> Inputs {
    Inputs {
        ca: black_box(FixedTrait::new(A_MAG, false)),
        cb: black_box(FixedTrait::new(B_MAG, true)),
        ra: black_box(A),
        rb: black_box(B),
        yes: black_box(true),
    }
}

#[inline(never)]
fn check(c: Fixed, c_mag: u128, c_sign: bool, r: i128, r_expected: i128) {
    assert!(c == FixedTrait::new(c_mag, c_sign));
    assert!(r == r_expected);
}

#[inline(never)]
fn check_bool(c: bool, r: bool) {
    assert!(c);
    assert!(r);
}

#[test]
#[inline(never)]
fn bench_add_q64__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q64__cubit() {
    let i = inputs();
    check(i.ca + i.cb, 41505174165846491136, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q64__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra + i.rb, 41505174165846491136);
}

#[test]
#[inline(never)]
fn bench_mul_q64__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q64__cubit() {
    let i = inputs();
    check(i.ca * i.cb, 80704505322479288320, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q64__orion_fp64x64() {
    let i = inputs();
    let (a, b): (FP64x64, FP64x64) = (i.ca, i.cb);
    check(fp64x64::FP64x64Mul::mul(a, b), 80704505322479288320, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q64__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, q64::mul(i.ra, i.rb), -80704505322479288320);
}

#[test]
#[inline(never)]
fn bench_div_q64__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q64__cubit() {
    let i = inputs();
    check(i.ca / i.cb, 51650883406386744524, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q64__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, q64::div(i.ra, i.rb), -51650883406386744524);
}

#[test]
#[inline(never)]
fn bench_sqrt_q64__baseline() {
    let i = inputs();
    check(i.ca, A_MAG, false, i.ra, A);
}

/// cubit computes `sqrt(mag) * 2^64 / 2^32`: only the top 32 fractional bits are meaningful
/// (expected value below is 1.8708286933 with the low 32 bits at zero).
#[test]
#[inline(never)]
fn bench_sqrt_q64__cubit() {
    let i = inputs();
    check(FixedTrait::sqrt(i.ca), 34510698110448041984, false, i.ra, A);
}

/// Full 64 fractional bits through a u256 square root.
#[test]
#[inline(never)]
fn bench_sqrt_q64__reference() {
    let i = inputs();
    check(i.ca, A_MAG, false, q64::sqrt(i.ra), 34510698112661885445);
}

#[test]
#[inline(never)]
fn bench_lt_q64__baseline() {
    let i = inputs();
    check_bool(i.yes, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q64__cubit() {
    let i = inputs();
    check_bool(i.cb < i.ca, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q64__reference() {
    let i = inputs();
    check_bool(i.yes, i.rb < i.ra);
}
