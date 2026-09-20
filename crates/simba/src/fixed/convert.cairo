//! Conversions between `Fixed` and integers.
//!
//! `Into` for the integer types that always fit (`i8`, `i16`, `i32`, `u8`, `u16`), `TryInto`
//! (returning `None` out of range) for `i64`, `i128`, `u32`, `u64`, `u128`.

use super::kernels;
use super::types::Fixed;

/// Wraps a raw Q32.32 value (`value * 2^32`). Free.
#[inline(always)]
pub fn from_raw(raw: i64) -> Fixed {
    Fixed { raw }
}

/// The integer `v` as a `Fixed`: exact, cannot fail, no range check.
#[inline(always)]
pub fn from_int(v: i32) -> Fixed {
    Fixed { raw: kernels::from_i32(v) }
}

/// `floor(num / den)` for two INTEGERS (`from_ratio(1, 3)` is one third, floored): one rounding.
/// Panics with `errors::DIVISION_BY_ZERO` / `errors::OVERFLOW`. Meant for constants such as
/// `dt = from_ratio(1, 60)`.
#[inline(always)]
pub fn from_ratio(num: i64, den: i64) -> Fixed {
    // floor(num * 2^32 / den) is the raw division kernel applied to the integers themselves.
    Fixed { raw: kernels::div(num, den) }
}

/// `floor(a)` as an integer (toward -inf: `to_int(-0.5) = -1`). Branch-free, cannot fail.
/// Compose with `round` / `trunc` / `ceil` for the other rounding modes.
#[inline(always)]
pub fn to_int(a: Fixed) -> i32 {
    kernels::to_i32(a.raw)
}

pub impl I8IntoFixed of Into<i8, Fixed> {
    #[inline(always)]
    fn into(self: i8) -> Fixed {
        Fixed { raw: kernels::from_i8(self) }
    }
}

pub impl I16IntoFixed of Into<i16, Fixed> {
    #[inline(always)]
    fn into(self: i16) -> Fixed {
        Fixed { raw: kernels::from_i16(self) }
    }
}

pub impl I32IntoFixed of Into<i32, Fixed> {
    #[inline(always)]
    fn into(self: i32) -> Fixed {
        Fixed { raw: kernels::from_i32(self) }
    }
}

pub impl U8IntoFixed of Into<u8, Fixed> {
    #[inline(always)]
    fn into(self: u8) -> Fixed {
        Fixed { raw: kernels::from_u8(self) }
    }
}

pub impl U16IntoFixed of Into<u16, Fixed> {
    #[inline(always)]
    fn into(self: u16) -> Fixed {
        Fixed { raw: kernels::from_u16(self) }
    }
}

#[inline(always)]
fn wrap(raw: Option<i64>) -> Option<Fixed> {
    match raw {
        Some(raw) => Some(Fixed { raw }),
        None => None,
    }
}

pub impl I64TryIntoFixed of TryInto<i64, Fixed> {
    #[inline(always)]
    fn try_into(self: i64) -> Option<Fixed> {
        wrap(kernels::try_from_i64(self))
    }
}

pub impl I128TryIntoFixed of TryInto<i128, Fixed> {
    #[inline(always)]
    fn try_into(self: i128) -> Option<Fixed> {
        wrap(kernels::try_from_i128(self))
    }
}

pub impl U32TryIntoFixed of TryInto<u32, Fixed> {
    #[inline(always)]
    fn try_into(self: u32) -> Option<Fixed> {
        wrap(kernels::try_from_u32(self))
    }
}

pub impl U64TryIntoFixed of TryInto<u64, Fixed> {
    #[inline(always)]
    fn try_into(self: u64) -> Option<Fixed> {
        wrap(kernels::try_from_u64(self))
    }
}

pub impl U128TryIntoFixed of TryInto<u128, Fixed> {
    #[inline(always)]
    fn try_into(self: u128) -> Option<Fixed> {
        wrap(kernels::try_from_u128(self))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::types::{Fixed, MAX, MIN, NEG_ONE, ONE, ZERO};
    use super::{from_int, from_ratio, from_raw, to_int};

    /// Largest integer value, 2^31 - 1.
    const MAX_INT: Fixed = Fixed { raw: 0x7fffffff00000000 };

    #[test]
    fn test_from_raw_wraps() {
        assert!(from_raw(0x100000000) == ONE);
        assert!(from_raw(-0x8000000000000000) == MIN);
    }

    #[test]
    fn test_from_int_values() {
        assert!(from_int(0) == ZERO);
        assert!(from_int(1) == ONE);
        assert!(from_int(-1) == NEG_ONE);
        assert!(from_int(-3) == fx(-0x300000000));
        assert!(from_int(0x7fffffff) == MAX_INT);
        assert!(from_int(-0x80000000) == MIN);
    }

    #[test]
    fn test_to_int_floors() {
        assert!(to_int(fx(0x280000000)) == 2);
        assert!(to_int(fx(-0x280000000)) == -3);
        assert!(to_int(fx(1)) == 0);
        assert!(to_int(fx(-1)) == -1);
        assert!(to_int(ZERO) == 0);
        assert!(to_int(fx(-0x300000000)) == -3);
        assert!(to_int(MAX) == 0x7fffffff);
        assert!(to_int(MIN) == -0x80000000);
    }

    #[test]
    fn test_to_int_from_int_roundtrip() {
        assert!(to_int(from_int(123456789)) == 123456789);
        assert!(to_int(from_int(-123456789)) == -123456789);
    }

    #[test]
    fn test_from_ratio_rounds_toward_negative_infinity() {
        assert!(from_ratio(1, 3) == fx(1431655765));
        assert!(from_ratio(-1, 3) == fx(-1431655766));
        assert!(from_ratio(1, -3) == fx(-1431655766));
        assert!(from_ratio(-1, -3) == fx(1431655765));
        assert!(from_ratio(1, 60) == fx(71582788));
        assert!(from_ratio(-1, 60) == fx(-71582789));
        assert!(from_ratio(22, 7) == fx(13498468644));
    }

    #[test]
    fn test_from_ratio_exact_and_boundaries() {
        assert!(from_ratio(7, 2) == fx(0x380000000));
        assert!(from_ratio(0, 5) == ZERO);
        assert!(from_ratio(0x7fffffff, 1) == MAX_INT);
        assert!(from_ratio(-0x80000000, 1) == MIN);
        assert!(from_ratio(0x7fffffffffffffff, 0x7fffffffffffffff) == ONE);
        assert!(from_ratio(-0x8000000000000000, 0x7fffffffffffffff) == fx(-0x100000001));
        assert!(from_ratio(-0x8000000000000000, 0x100000000) == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_from_ratio_overflow_panics() {
        from_ratio(black_box(0x80000000), 1);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_from_ratio_underflow_panics() {
        from_ratio(black_box(-0x80000001), 1);
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_from_ratio_zero_denominator_panics() {
        from_ratio(black_box(1), 0);
    }

    #[test]
    fn test_into_small_integers() {
        let x: Fixed = 0x7f_i8.into();
        assert!(x == fx(0x7f00000000));
        let x: Fixed = (-0x80_i8).into();
        assert!(x == fx(-0x8000000000));
        let x: Fixed = 0x7fff_i16.into();
        assert!(x == fx(0x7fff00000000));
        let x: Fixed = (-0x8000_i16).into();
        assert!(x == fx(-0x800000000000));
        let x: Fixed = 0x7fffffff_i32.into();
        assert!(x == MAX_INT);
        let x: Fixed = (-0x80000000_i32).into();
        assert!(x == MIN);
        let x: Fixed = 0xff_u8.into();
        assert!(x == fx(0xff00000000));
        let x: Fixed = 0xffff_u16.into();
        assert!(x == fx(0xffff00000000));
        let x: Fixed = 0_u8.into();
        assert!(x == ZERO);
    }

    #[test]
    fn test_try_into_in_range() {
        assert!(0x7fffffff_i64.try_into() == Some(MAX_INT));
        assert!((-0x80000000_i64).try_into() == Some(MIN));
        assert!(0x7fffffff_i128.try_into() == Some(MAX_INT));
        assert!((-0x80000000_i128).try_into() == Some(MIN));
        assert!(0x7fffffff_u32.try_into() == Some(MAX_INT));
        assert!(0x7fffffff_u64.try_into() == Some(MAX_INT));
        assert!(0x7fffffff_u128.try_into() == Some(MAX_INT));
        assert!(0_u32.try_into() == Some(ZERO));
        assert!((-1_i64).try_into() == Some(NEG_ONE));
    }

    #[test]
    fn test_try_into_out_of_range_is_none() {
        let none: Option<Fixed> = None;
        assert!(0x80000000_i64.try_into() == none);
        assert!((-0x80000001_i64).try_into() == none);
        assert!(0x7fffffffffffffff_i64.try_into() == none);
        assert!(0x80000000_i128.try_into() == none);
        assert!((-0x80000001_i128).try_into() == none);
        assert!(0x80000000_u32.try_into() == none);
        assert!(0xffffffff_u32.try_into() == none);
        assert!(0x80000000_u64.try_into() == none);
        assert!(0xffffffffffffffff_u64.try_into() == none);
        assert!(0x80000000_u128.try_into() == none);
    }

    // --- gas benchmarks (net = raw - baseline of the group) --------------------------------------
    // Losing candidates: `bench_<op>__alt_*` in `fixed::kernels::alternatives`.

    #[test]
    #[inline(never)]
    fn bench_from_int__baseline() {
        let _a = black_box(-0x3_i32);
        let e = black_box(fx(-0x300000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_int__fixed_n() {
        let a = black_box(-0x3_i32);
        let e = black_box(fx(-0x300000000));
        assert!(from_int(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_to_int__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(0x3_i32);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_to_int__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(0x3_i32);
        assert!(to_int(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_to_int__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(-0x3_i32);
        assert!(to_int(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_ratio__baseline() {
        let _a = black_box(0x1_i64);
        let _b = black_box(0x3c_i64);
        let e = black_box(fx(0x4444444));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_ratio__fixed_pp() {
        let a = black_box(0x1_i64);
        let b = black_box(0x3c_i64);
        let e = black_box(fx(0x4444444));
        assert!(from_ratio(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_ratio__fixed_np() {
        let a = black_box(-0x1_i64);
        let b = black_box(0x3c_i64);
        let e = black_box(fx(-0x4444445));
        assert!(from_ratio(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_into__baseline() {
        let _a = black_box(-0x7_i8);
        let e = black_box(fx(-0x700000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_into__fixed_i8() {
        let a = black_box(-0x7_i8);
        let e = black_box(fx(-0x700000000));
        assert!(Into::<_, Fixed>::into(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_into__fixed_i16() {
        let a = black_box(-0x7_i16);
        let e = black_box(fx(-0x700000000));
        assert!(Into::<_, Fixed>::into(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_into__fixed_i32() {
        let a = black_box(-0x7_i32);
        let e = black_box(fx(-0x700000000));
        assert!(Into::<_, Fixed>::into(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_into__fixed_u8() {
        let a = black_box(0x7_u8);
        let e = black_box(fx(0x700000000));
        assert!(Into::<_, Fixed>::into(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_into__fixed_u16() {
        let a = black_box(0x7_u16);
        let e = black_box(fx(0x700000000));
        assert!(Into::<_, Fixed>::into(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_try_into__baseline() {
        let _a = black_box(-0x7_i64);
        let e = black_box(fx(-0x700000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_try_into__fixed_i64() {
        let a = black_box(-0x7_i64);
        let e = black_box(fx(-0x700000000));
        assert!(TryInto::<_, Fixed>::try_into(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_try_into__fixed_i128() {
        let a = black_box(-0x7_i128);
        let e = black_box(fx(-0x700000000));
        assert!(TryInto::<_, Fixed>::try_into(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_try_into__fixed_u32() {
        let a = black_box(0x7_u32);
        let e = black_box(fx(0x700000000));
        assert!(TryInto::<_, Fixed>::try_into(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_try_into__fixed_u64() {
        let a = black_box(0x7_u64);
        let e = black_box(fx(0x700000000));
        assert!(TryInto::<_, Fixed>::try_into(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_try_into__fixed_u128() {
        let a = black_box(0x7_u128);
        let e = black_box(fx(0x700000000));
        assert!(TryInto::<_, Fixed>::try_into(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_raw__baseline() {
        let _a = black_box(0x380000000_i64);
        let e = black_box(fx(0x380000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_raw__fixed() {
        let a = black_box(0x380000000_i64);
        let e = black_box(fx(0x380000000));
        assert!(from_raw(a) == e);
    }
}
