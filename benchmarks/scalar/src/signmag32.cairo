//! A. `SignMag32`: sign + magnitude Q16.16 (orion `FP16x16` style).
//!
//! Rounding: truncation toward zero (natural for magnitudes). Overflow: panics (u32 checked ops).
//! Note: `-0` is normalized on add/sub/neg like cubit does, but not on mul/div (cubit does not
//! either), so derived equality can see `-0 != 0` after a multiplication by zero.

use core::num::traits::{Sqrt, WideMul};

pub const ONE: u32 = 0x10000;
const ONE_U64: u64 = 0x10000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct SignMag32 {
    pub mag: u32,
    pub sign: bool,
}

pub impl SignMag32Add of Add<SignMag32> {
    #[inline(always)]
    fn add(lhs: SignMag32, rhs: SignMag32) -> SignMag32 {
        if lhs.sign == rhs.sign {
            return SignMag32 { mag: lhs.mag + rhs.mag, sign: lhs.sign };
        }
        if lhs.mag == rhs.mag {
            return SignMag32 { mag: 0, sign: false };
        }
        if lhs.mag > rhs.mag {
            SignMag32 { mag: lhs.mag - rhs.mag, sign: lhs.sign }
        } else {
            SignMag32 { mag: rhs.mag - lhs.mag, sign: rhs.sign }
        }
    }
}

pub impl SignMag32Neg of Neg<SignMag32> {
    #[inline(always)]
    fn neg(a: SignMag32) -> SignMag32 {
        if a.mag == 0 {
            a
        } else {
            SignMag32 { mag: a.mag, sign: !a.sign }
        }
    }
}

pub impl SignMag32Sub of Sub<SignMag32> {
    #[inline(always)]
    fn sub(lhs: SignMag32, rhs: SignMag32) -> SignMag32 {
        lhs + (-rhs)
    }
}

pub impl SignMag32Mul of Mul<SignMag32> {
    #[inline(always)]
    fn mul(lhs: SignMag32, rhs: SignMag32) -> SignMag32 {
        let wide: u64 = lhs.mag.wide_mul(rhs.mag);
        SignMag32 {
            mag: (wide / ONE_U64).try_into().expect('SignMag32 mul overflow'),
            sign: lhs.sign ^ rhs.sign,
        }
    }
}

pub impl SignMag32Div of Div<SignMag32> {
    #[inline(always)]
    fn div(lhs: SignMag32, rhs: SignMag32) -> SignMag32 {
        let wide: u64 = lhs.mag.wide_mul(ONE);
        SignMag32 {
            mag: (wide / rhs.mag.into()).try_into().expect('SignMag32 div overflow'),
            sign: lhs.sign ^ rhs.sign,
        }
    }
}

pub impl SignMag32PartialOrd of PartialOrd<SignMag32> {
    #[inline(always)]
    fn lt(lhs: SignMag32, rhs: SignMag32) -> bool {
        if lhs.sign != rhs.sign {
            lhs.sign
        } else {
            lhs.mag != rhs.mag && ((lhs.mag < rhs.mag) ^ lhs.sign)
        }
    }
}

#[inline(always)]
pub fn abs(a: SignMag32) -> SignMag32 {
    SignMag32 { mag: a.mag, sign: false }
}

#[inline(always)]
pub fn from_int(v: i16) -> SignMag32 {
    if v < 0 {
        let m: u16 = (-v).try_into().unwrap();
        SignMag32 { mag: m.into() * ONE, sign: true }
    } else {
        let m: u16 = v.try_into().unwrap();
        SignMag32 { mag: m.into() * ONE, sign: false }
    }
}

/// Floor to integer.
#[inline(always)]
pub fn to_int(a: SignMag32) -> i16 {
    let (q, r) = DivRem::div_rem(a.mag, 0x10000);
    let q: u16 = q.try_into().unwrap();
    let q: i16 = q.try_into().expect('SignMag32 to_int overflow');
    if a.sign {
        if r == 0 {
            -q
        } else {
            -q - 1
        }
    } else {
        q
    }
}

/// sqrt through the corelib integer square root of the widened magnitude.
#[inline(always)]
pub fn sqrt(a: SignMag32) -> SignMag32 {
    assert(!a.sign, 'sqrt of negative');
    let wide: u64 = a.mag.wide_mul(ONE);
    SignMag32 { mag: wide.sqrt(), sign: false }
}
