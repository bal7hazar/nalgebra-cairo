//! B. `I64Q32`: newtype over native `i64`, Q32.32, plain corelib arithmetic only.
//!
//! - add/sub/neg: native `i64` ops (range-checked, panic on overflow).
//! - mul: widening `i64 -> i128` multiply, corelib signed `/ 2^32`, checked narrowing to `i64`.
//! - div: `(a * 2^32) / b` on `i128`.
//! Rounding: truncation toward zero (corelib signed DivRem semantics). Overflow: panics.

use core::num::traits::{Sqrt, WideMul};

pub const ONE: i64 = 0x100000000;
const ONE_WIDE: i128 = 0x100000000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct I64Q32 {
    pub raw: i64,
}

pub impl I64Q32Add of Add<I64Q32> {
    #[inline(always)]
    fn add(lhs: I64Q32, rhs: I64Q32) -> I64Q32 {
        I64Q32 { raw: lhs.raw + rhs.raw }
    }
}

pub impl I64Q32Sub of Sub<I64Q32> {
    #[inline(always)]
    fn sub(lhs: I64Q32, rhs: I64Q32) -> I64Q32 {
        I64Q32 { raw: lhs.raw - rhs.raw }
    }
}

pub impl I64Q32Neg of Neg<I64Q32> {
    #[inline(always)]
    fn neg(a: I64Q32) -> I64Q32 {
        I64Q32 { raw: -a.raw }
    }
}

pub impl I64Q32Mul of Mul<I64Q32> {
    #[inline(always)]
    fn mul(lhs: I64Q32, rhs: I64Q32) -> I64Q32 {
        let wide: i128 = lhs.raw.wide_mul(rhs.raw);
        I64Q32 { raw: (wide / ONE_WIDE).try_into().expect('I64Q32 mul overflow') }
    }
}

pub impl I64Q32Div of Div<I64Q32> {
    #[inline(always)]
    fn div(lhs: I64Q32, rhs: I64Q32) -> I64Q32 {
        let wide: i128 = lhs.raw.wide_mul(ONE);
        I64Q32 { raw: (wide / rhs.raw.into()).try_into().expect('I64Q32 div overflow') }
    }
}

pub impl I64Q32PartialOrd of PartialOrd<I64Q32> {
    #[inline(always)]
    fn lt(lhs: I64Q32, rhs: I64Q32) -> bool {
        lhs.raw < rhs.raw
    }
}

#[inline(always)]
pub fn abs(a: I64Q32) -> I64Q32 {
    if a.raw < 0 {
        I64Q32 { raw: -a.raw }
    } else {
        a
    }
}

#[inline(always)]
pub fn from_int(v: i32) -> I64Q32 {
    let v: i64 = v.into();
    I64Q32 { raw: v * ONE }
}

/// Floor to integer.
#[inline(always)]
pub fn to_int(a: I64Q32) -> i32 {
    let (q, r) = DivRem::div_rem(a.raw, 0x100000000);
    let q: i32 = q.try_into().expect('I64Q32 to_int overflow');
    if r < 0 {
        q - 1
    } else {
        q
    }
}

/// sqrt through the corelib integer square root of the widened value.
#[inline(always)]
pub fn sqrt(a: I64Q32) -> I64Q32 {
    let m: u64 = a.raw.try_into().expect('sqrt of negative');
    let wide: u128 = m.wide_mul(0x100000000);
    let r: u64 = wide.sqrt();
    I64Q32 { raw: r.try_into().unwrap() }
}
