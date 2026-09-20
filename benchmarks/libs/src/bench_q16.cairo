//! Scalar Q16.16: orion `FP16x16` (orion's own sign-magnitude `u32` + `bool` port of cubit),
//! reference (`i32` widening to `i64`).
//!
//! Same fixed overhead in every test (see bench_q32). Inputs: a = 3.5, b = -1.25.

use harness::black_box;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::{FP16x16, FP16x16Impl};
use crate::reference::q16;

const A_MAG: u32 = 229376; // 3.5
const B_MAG: u32 = 81920; // 1.25
const A: i32 = 229376;
const B: i32 = -81920;

#[derive(Copy, Drop)]
struct Inputs {
    oa: FP16x16,
    ob: FP16x16,
    ra: i32,
    rb: i32,
    yes: bool,
}

#[inline(never)]
fn inputs() -> Inputs {
    Inputs {
        oa: black_box(FixedTrait::new(A_MAG, false)),
        ob: black_box(FixedTrait::new(B_MAG, true)),
        ra: black_box(A),
        rb: black_box(B),
        yes: black_box(true),
    }
}

#[inline(never)]
fn check(o: FP16x16, o_mag: u32, o_sign: bool, r: i32, r_expected: i32) {
    assert!(o == FixedTrait::new(o_mag, o_sign));
    assert!(r == r_expected);
}

#[inline(never)]
fn check_bool(c: bool, r: bool) {
    assert!(c);
    assert!(r);
}

#[test]
#[inline(never)]
fn bench_add_q16__baseline() {
    let i = inputs();
    check(i.oa, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q16__orion_fp16x16() {
    let i = inputs();
    check(i.oa + i.ob, 147456, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_add_q16__reference() {
    let i = inputs();
    check(i.oa, A_MAG, false, i.ra + i.rb, 147456);
}

#[test]
#[inline(never)]
fn bench_mul_q16__baseline() {
    let i = inputs();
    check(i.oa, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q16__orion_fp16x16() {
    let i = inputs();
    check(i.oa * i.ob, 286720, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_mul_q16__reference() {
    let i = inputs();
    check(i.oa, A_MAG, false, q16::mul(i.ra, i.rb), -286720);
}

#[test]
#[inline(never)]
fn bench_div_q16__baseline() {
    let i = inputs();
    check(i.oa, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q16__orion_fp16x16() {
    let i = inputs();
    check(i.oa / i.ob, 183500, true, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_div_q16__reference() {
    let i = inputs();
    check(i.oa, A_MAG, false, q16::div(i.ra, i.rb), -183500);
}

#[test]
#[inline(never)]
fn bench_sqrt_q16__baseline() {
    let i = inputs();
    check(i.oa, A_MAG, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_sqrt_q16__orion_fp16x16() {
    let i = inputs();
    check(i.oa.sqrt(), 122606, false, i.ra, A);
}

#[test]
#[inline(never)]
fn bench_lt_q16__baseline() {
    let i = inputs();
    check_bool(i.yes, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q16__orion_fp16x16() {
    let i = inputs();
    check_bool(i.ob < i.oa, i.yes);
}

#[test]
#[inline(never)]
fn bench_lt_q16__reference() {
    let i = inputs();
    check_bool(i.yes, i.rb < i.ra);
}
