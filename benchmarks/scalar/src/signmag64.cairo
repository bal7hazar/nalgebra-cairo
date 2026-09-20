//! A. `SignMag64`: sign + magnitude Q32.32 (cubit `f64` style).
//!
//! Rounding: truncation toward zero (natural for magnitudes). Overflow: panics (u64 checked ops).
//! Note: `-0` is normalized on add/sub/neg like cubit does, but not on mul/div (cubit does not
//! either), so derived equality can see `-0 != 0` after a multiplication by zero.

use core::num::traits::{Sqrt, WideMul};

pub const ONE: u64 = 0x100000000;
const ONE_U128: u128 = 0x100000000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct SignMag64 {
    pub mag: u64,
    pub sign: bool,
}

pub impl SignMag64Add of Add<SignMag64> {
    #[inline(always)]
    fn add(lhs: SignMag64, rhs: SignMag64) -> SignMag64 {
        if lhs.sign == rhs.sign {
            return SignMag64 { mag: lhs.mag + rhs.mag, sign: lhs.sign };
        }
        if lhs.mag == rhs.mag {
            return SignMag64 { mag: 0, sign: false };
        }
        if lhs.mag > rhs.mag {
            SignMag64 { mag: lhs.mag - rhs.mag, sign: lhs.sign }
        } else {
            SignMag64 { mag: rhs.mag - lhs.mag, sign: rhs.sign }
        }
    }
}

pub impl SignMag64Neg of Neg<SignMag64> {
    #[inline(always)]
    fn neg(a: SignMag64) -> SignMag64 {
        if a.mag == 0 {
            a
        } else {
            SignMag64 { mag: a.mag, sign: !a.sign }
        }
    }
}

pub impl SignMag64Sub of Sub<SignMag64> {
    #[inline(always)]
    fn sub(lhs: SignMag64, rhs: SignMag64) -> SignMag64 {
        lhs + (-rhs)
    }
}

pub impl SignMag64Mul of Mul<SignMag64> {
    #[inline(always)]
    fn mul(lhs: SignMag64, rhs: SignMag64) -> SignMag64 {
        let wide: u128 = lhs.mag.wide_mul(rhs.mag);
        SignMag64 {
            mag: (wide / ONE_U128).try_into().expect('SignMag64 mul overflow'),
            sign: lhs.sign ^ rhs.sign,
        }
    }
}

pub impl SignMag64Div of Div<SignMag64> {
    #[inline(always)]
    fn div(lhs: SignMag64, rhs: SignMag64) -> SignMag64 {
        let wide: u128 = lhs.mag.wide_mul(ONE);
        SignMag64 {
            mag: (wide / rhs.mag.into()).try_into().expect('SignMag64 div overflow'),
            sign: lhs.sign ^ rhs.sign,
        }
    }
}

pub impl SignMag64PartialOrd of PartialOrd<SignMag64> {
    #[inline(always)]
    fn lt(lhs: SignMag64, rhs: SignMag64) -> bool {
        if lhs.sign != rhs.sign {
            lhs.sign
        } else {
            lhs.mag != rhs.mag && ((lhs.mag < rhs.mag) ^ lhs.sign)
        }
    }
}

#[inline(always)]
pub fn abs(a: SignMag64) -> SignMag64 {
    SignMag64 { mag: a.mag, sign: false }
}

#[inline(always)]
pub fn from_int(v: i32) -> SignMag64 {
    if v < 0 {
        let m: u32 = (-v).try_into().unwrap();
        SignMag64 { mag: m.into() * ONE, sign: true }
    } else {
        let m: u32 = v.try_into().unwrap();
        SignMag64 { mag: m.into() * ONE, sign: false }
    }
}

/// Floor to integer.
#[inline(always)]
pub fn to_int(a: SignMag64) -> i32 {
    let (q, r) = DivRem::div_rem(a.mag, 0x100000000);
    let q: u32 = q.try_into().unwrap();
    let q: i32 = q.try_into().expect('SignMag64 to_int overflow');
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
pub fn sqrt(a: SignMag64) -> SignMag64 {
    assert(!a.sign, 'sqrt of negative');
    let wide: u128 = a.mag.wide_mul(ONE);
    SignMag64 { mag: wide.sqrt(), sign: false }
}
