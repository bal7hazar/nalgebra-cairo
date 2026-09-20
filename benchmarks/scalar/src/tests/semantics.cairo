//! Hand-written semantic checks (not benchmarks): rounding modes, overflow policy, and the range
//! advantage of fused (single rescale) forms.
use crate::bi64::{self, BI64};
use crate::felt_fixed::FeltQ32;
use crate::generic;
use crate::i64q32::I64Q32;
use crate::signmag64::SignMag64;

const ONE: i64 = 0x100000000;
const HALF: i64 = 0x80000000;

#[test]
fn test_rounding_modes_on_negative_inexact_product() {
    // (-1 ulp) * 0.5 = -0.5 ulp: floor -> -1 ulp, truncation -> 0.
    assert!((BI64 { raw: -1 } * BI64 { raw: HALF }).raw == -1);
    assert!((FeltQ32 { raw: -1 } * FeltQ32 { raw: HALF.into() }).raw == -1);
    assert!((I64Q32 { raw: -1 } * I64Q32 { raw: HALF }).raw == 0);
    assert!(
        (SignMag64 { mag: 1, sign: true } * SignMag64 { mag: 0x80000000, sign: false }).mag == 0,
    );
}

#[test]
fn test_floor_division() {
    // -1 / 3 = -0.333..: floor -> -1431655766, truncation -> -1431655765.
    assert!((BI64 { raw: -ONE } / BI64 { raw: 3 * ONE }).raw == -1431655766);
    assert!((BI64 { raw: ONE } / BI64 { raw: -3 * ONE }).raw == -1431655766);
    assert!((BI64 { raw: -ONE } / BI64 { raw: -3 * ONE }).raw == 1431655765);
    assert!((I64Q32 { raw: -ONE } / I64Q32 { raw: 3 * ONE }).raw == -1431655765);
    // exact negative quotient: no adjustment.
    assert!((BI64 { raw: -6 * ONE } / BI64 { raw: 3 * ONE }).raw == -2 * ONE);
}

#[test]
#[should_panic(expected: 'BI64 overflow')]
fn test_bi64_mul_overflow_panics() {
    let _ = BI64 { raw: 60000 * ONE } * BI64 { raw: 60000 * ONE };
}

#[test]
#[should_panic(expected: 'I64Q32 mul overflow')]
fn test_i64q32_mul_overflow_panics() {
    let _ = I64Q32 { raw: 60000 * ONE } * I64Q32 { raw: 60000 * ONE };
}

#[test]
#[should_panic(expected: 'FeltQ32 overflow')]
fn test_felt_mul_overflow_panics() {
    let v: felt252 = (60000 * ONE).into();
    let _ = FeltQ32 { raw: v } * FeltQ32 { raw: v };
}

#[test]
#[should_panic]
fn test_bi64_div_by_zero_panics() {
    let _ = BI64 { raw: ONE } / BI64 { raw: 0 };
}

#[test]
fn test_fused_dot_survives_intermediate_overflow() {
    // 60000^2 = 3.6e9 does not fit Q32.32 (max 2.1e9) but the fused sum does.
    let a = BI64 { raw: 60000 * ONE };
    let z = BI64 { raw: 0 };
    assert!(bi64::dot3(a, a, z, a, -a, z).raw == 0);
}

#[test]
#[should_panic(expected: 'BI64 overflow')]
fn test_unfused_dot_overflows() {
    let a = BI64 { raw: 60000 * ONE };
    let z = BI64 { raw: 0 };
    let _ = generic::dot3(a, a, z, a, -a, z);
}

#[test]
fn test_fused_length_of_world_scale_vector() {
    // |(1e6, 1e6, 1e6)| = 1732050.80756...: the squares (1e12) are far outside Q32.32.
    let a = BI64 { raw: 1000000 * ONE };
    let len = bi64::length3(a, -a, a);
    let want: i64 = 7439101573518717; // floor(sqrt(3) * 1e6 * 2^32)
    assert!(len.raw == want);
}

#[test]
fn test_bi64_int_conversions() {
    assert!(bi64::to_int(BI64 { raw: -1 }) == -1);
    assert!(bi64::to_int(BI64 { raw: 0x7fffffffffffffff }) == 0x7fffffff);
    assert!(bi64::to_int(BI64 { raw: -0x7fffffffffffffff - 1 }) == -0x80000000);
    assert!(bi64::from_int(-0x80000000).raw == -0x7fffffffffffffff - 1);
    assert!(bi64::sqrt(BI64 { raw: 2 * ONE }).raw == 6074000999);
}
