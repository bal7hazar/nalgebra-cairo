//! E. `I32Q16`: newtype over native `i32`, Q16.16, plain corelib arithmetic only.
//!
//! - add/sub/neg: native `i32` ops (range-checked, panic on overflow).
//! - mul: widening `i32 -> i64` multiply, corelib signed `/ 2^16`, checked narrowing to `i32`.
//! - div: `(a * 2^16) / b` on `i64`.
//! Rounding: truncation toward zero (corelib signed DivRem semantics). Overflow: panics.

use core::num::traits::{Sqrt, WideMul};

pub const ONE: i32 = 0x10000;
const ONE_WIDE: i64 = 0x10000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct I32Q16 {
    pub raw: i32,
}

pub impl I32Q16Add of Add<I32Q16> {
    #[inline(always)]
    fn add(lhs: I32Q16, rhs: I32Q16) -> I32Q16 {
        I32Q16 { raw: lhs.raw + rhs.raw }
    }
}

pub impl I32Q16Sub of Sub<I32Q16> {
    #[inline(always)]
    fn sub(lhs: I32Q16, rhs: I32Q16) -> I32Q16 {
        I32Q16 { raw: lhs.raw - rhs.raw }
    }
}

pub impl I32Q16Neg of Neg<I32Q16> {
    #[inline(always)]
    fn neg(a: I32Q16) -> I32Q16 {
        I32Q16 { raw: -a.raw }
    }
}

pub impl I32Q16Mul of Mul<I32Q16> {
    #[inline(always)]
    fn mul(lhs: I32Q16, rhs: I32Q16) -> I32Q16 {
        let wide: i64 = lhs.raw.wide_mul(rhs.raw);
        I32Q16 { raw: (wide / ONE_WIDE).try_into().expect('I32Q16 mul overflow') }
    }
}

pub impl I32Q16Div of Div<I32Q16> {
    #[inline(always)]
    fn div(lhs: I32Q16, rhs: I32Q16) -> I32Q16 {
        let wide: i64 = lhs.raw.wide_mul(ONE);
        I32Q16 { raw: (wide / rhs.raw.into()).try_into().expect('I32Q16 div overflow') }
    }
}

pub impl I32Q16PartialOrd of PartialOrd<I32Q16> {
    #[inline(always)]
    fn lt(lhs: I32Q16, rhs: I32Q16) -> bool {
        lhs.raw < rhs.raw
    }
}

#[inline(always)]
pub fn abs(a: I32Q16) -> I32Q16 {
    if a.raw < 0 {
        I32Q16 { raw: -a.raw }
    } else {
        a
    }
}

#[inline(always)]
pub fn from_int(v: i16) -> I32Q16 {
    let v: i32 = v.into();
    I32Q16 { raw: v * ONE }
}

/// Floor to integer.
#[inline(always)]
pub fn to_int(a: I32Q16) -> i16 {
    let (q, r) = DivRem::div_rem(a.raw, 0x10000);
    let q: i16 = q.try_into().expect('I32Q16 to_int overflow');
    if r < 0 {
        q - 1
    } else {
        q
    }
}

/// sqrt through the corelib integer square root of the widened value.
#[inline(always)]
pub fn sqrt(a: I32Q16) -> I32Q16 {
    let m: u32 = a.raw.try_into().expect('sqrt of negative');
    let wide: u64 = m.wide_mul(0x10000);
    let r: u32 = wide.sqrt();
    I32Q16 { raw: r.try_into().unwrap() }
}
