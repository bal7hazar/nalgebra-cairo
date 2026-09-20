//! D. `FeltQ32`: Q32.32 stored in a `felt252`, negative numbers as `P - x`.
//!
//! Invariant: |raw| < 2^63 (as a signed felt). add/sub/neg are raw felt ops: 1 step, NO range
//! check, NO overflow detection (the invariant can silently break on add overflow; it is
//! re-established by every `mul`/`rescale`, which is checked).
//! mul = felt mul + `rescale`: add 2^127 offset, range-check into u128 (recovers the sign and
//! bounds the value), div_rem by 2^32, remove offset. Rounding: floor.
//! Products can be kept unscaled ("lazy"): `dot3`, `cross3`, `lerp`, `mul_add` rescale once.
//!
//! Q64.64 in a felt252 is NOT possible this way: products reach 2^254 > P ~ 2^251.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, BoundedInt, DivRemHelper, SubHelper, UnitInt, downcast, upcast,
};
use core::num::traits::{Sqrt, WideMul};
use crate::bi64;

const TWO32: felt252 = 0x100000000;
const TWO127: felt252 = 0x80000000000000000000000000000000;
const TWO95: felt252 = 0x800000000000000000000000;
const TWO64: felt252 = 0x10000000000000000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct FeltQ32 {
    pub raw: felt252,
}

type Quot = BoundedInt<0, 0xffffffffffffffffffffffff>;
// [2^95 - 2^63, 2^95 + 2^63 - 1]: quotients whose un-offset value fits an i64.
type QuotOk = BoundedInt<0x7fffffff8000000000000000, 0x800000007fffffffffffffff>;
impl DivU128Two32 of DivRemHelper<u128, UnitInt<TWO32>> {
    type DivT = Quot;
    type RemT = BoundedInt<0, 0xffffffff>;
}

/// floor(p / 2^32) for an unscaled value |p| < 2^127; result checked to fit the i64 range.
#[inline(always)]
pub fn rescale(p: felt252) -> felt252 {
    let off: u128 = (p + TWO127).try_into().expect('FeltQ32 overflow');
    let (q, _r) = bounded_int::div_rem::<u128, UnitInt<TWO32>>(off, 0x100000000);
    let q: QuotOk = downcast(q).expect('FeltQ32 overflow');
    upcast::<QuotOk, felt252>(q) - TWO95
}

/// Same without the final range check: results may exceed the Q32.32 range (up to 2^94) and the
/// error is only caught by a later rescale if a product leaves (-2^127, 2^127).
#[inline(always)]
pub fn rescale_soft(p: felt252) -> felt252 {
    let off: u128 = (p + TWO127).try_into().expect('FeltQ32 overflow');
    let (q, _r) = bounded_int::div_rem::<u128, UnitInt<TWO32>>(off, 0x100000000);
    upcast::<Quot, felt252>(q) - TWO95
}

/// Same with plain corelib ops only (no BoundedInt).
#[inline(always)]
pub fn rescale_plain(p: felt252) -> felt252 {
    let off: u128 = (p + TWO127).try_into().expect('FeltQ32 overflow');
    let r: felt252 = (off / 0x100000000_u128).into() - TWO95;
    let _check: i64 = r.try_into().expect('FeltQ32 overflow');
    r
}

pub impl FeltQ32Add of Add<FeltQ32> {
    #[inline(always)]
    fn add(lhs: FeltQ32, rhs: FeltQ32) -> FeltQ32 {
        FeltQ32 { raw: lhs.raw + rhs.raw }
    }
}

pub impl FeltQ32Sub of Sub<FeltQ32> {
    #[inline(always)]
    fn sub(lhs: FeltQ32, rhs: FeltQ32) -> FeltQ32 {
        FeltQ32 { raw: lhs.raw - rhs.raw }
    }
}

pub impl FeltQ32Neg of Neg<FeltQ32> {
    #[inline(always)]
    fn neg(a: FeltQ32) -> FeltQ32 {
        FeltQ32 { raw: -a.raw }
    }
}

pub impl FeltQ32Mul of Mul<FeltQ32> {
    #[inline(always)]
    fn mul(lhs: FeltQ32, rhs: FeltQ32) -> FeltQ32 {
        FeltQ32 { raw: rescale(lhs.raw * rhs.raw) }
    }
}

#[inline(always)]
pub fn mul_soft(a: FeltQ32, b: FeltQ32) -> FeltQ32 {
    FeltQ32 { raw: rescale_soft(a.raw * b.raw) }
}

#[inline(always)]
pub fn mul_plain(a: FeltQ32, b: FeltQ32) -> FeltQ32 {
    FeltQ32 { raw: rescale_plain(a.raw * b.raw) }
}

pub impl FeltQ32Div of Div<FeltQ32> {
    /// Floor division: both operands must be brought back to i64 (2 conversions) to find signs.
    #[inline(always)]
    fn div(lhs: FeltQ32, rhs: FeltQ32) -> FeltQ32 {
        let a = bi64::BI64 { raw: lhs.raw.try_into().expect('FeltQ32 range') };
        let b = bi64::BI64 { raw: rhs.raw.try_into().expect('FeltQ32 range') };
        FeltQ32 { raw: (a / b).raw.into() }
    }
}

pub impl FeltQ32PartialOrd of PartialOrd<FeltQ32> {
    /// Trusts the invariant: lhs - rhs in (-2^64, 2^64).
    #[inline(always)]
    fn lt(lhs: FeltQ32, rhs: FeltQ32) -> bool {
        let d: Option<u64> = (lhs.raw - rhs.raw + TWO64).try_into();
        d.is_some()
    }
}

#[inline(always)]
pub fn abs(a: FeltQ32) -> FeltQ32 {
    let d: Option<u64> = (a.raw + TWO64).try_into();
    if d.is_some() {
        FeltQ32 { raw: -a.raw }
    } else {
        a
    }
}

#[inline(always)]
pub fn from_int(v: i32) -> FeltQ32 {
    FeltQ32 { raw: v.into() * TWO32 }
}

type U32Like = BoundedInt<0, 0xffffffff>;
type I32Like = BoundedInt<-0x80000000, 0x7fffffff>;
impl DivU64Two32 of DivRemHelper<u64, UnitInt<TWO32>> {
    type DivT = U32Like;
    type RemT = U32Like;
}
impl SubU32Two31 of SubHelper<U32Like, UnitInt<0x80000000>> {
    type Result = I32Like;
}

/// Floor to integer.
#[inline(always)]
pub fn to_int(a: FeltQ32) -> i32 {
    let off: u64 = (a.raw + 0x8000000000000000).try_into().expect('FeltQ32 range');
    let (q, _r) = bounded_int::div_rem::<u64, UnitInt<TWO32>>(off, 0x100000000);
    upcast(bounded_int::sub::<U32Like, UnitInt<0x80000000>>(q, 0x80000000))
}

#[inline(always)]
pub fn sqrt(a: FeltQ32) -> FeltQ32 {
    let m: u64 = a.raw.try_into().expect('sqrt of negative');
    let wide: u128 = m.wide_mul(0x100000000);
    let r: u64 = wide.sqrt();
    FeltQ32 { raw: r.into() }
}

// --- lazy forms: products stay unscaled in the field, ONE rescale ------------------------------

#[inline(always)]
pub fn dot3(
    ax: FeltQ32, ay: FeltQ32, az: FeltQ32, bx: FeltQ32, by: FeltQ32, bz: FeltQ32,
) -> FeltQ32 {
    FeltQ32 { raw: rescale(ax.raw * bx.raw + ay.raw * by.raw + az.raw * bz.raw) }
}

#[inline(always)]
pub fn cross3(
    ax: FeltQ32, ay: FeltQ32, az: FeltQ32, bx: FeltQ32, by: FeltQ32, bz: FeltQ32,
) -> (FeltQ32, FeltQ32, FeltQ32) {
    (
        FeltQ32 { raw: rescale(ay.raw * bz.raw - az.raw * by.raw) },
        FeltQ32 { raw: rescale(az.raw * bx.raw - ax.raw * bz.raw) },
        FeltQ32 { raw: rescale(ax.raw * by.raw - ay.raw * bx.raw) },
    )
}

#[inline(always)]
pub fn mul_add(a: FeltQ32, b: FeltQ32, c: FeltQ32) -> FeltQ32 {
    FeltQ32 { raw: rescale(a.raw * b.raw + c.raw * TWO32) }
}

#[inline(always)]
pub fn lerp(a: FeltQ32, b: FeltQ32, t: FeltQ32) -> FeltQ32 {
    FeltQ32 { raw: rescale((b.raw - a.raw) * t.raw + a.raw * TWO32) }
}
