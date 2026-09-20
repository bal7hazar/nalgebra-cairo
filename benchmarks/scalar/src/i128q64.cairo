//! C. `I128Q64`: newtype over native `i128`, Q64.64.
//!
//! mul needs a 256-bit intermediate: magnitudes -> `u128` wide mul -> `u256`, then `>> 64` done
//! with math (high * 2^64 + low / 2^64). div uses a `u256 / u256` division.
//! Rounding: truncation toward zero. Overflow: panics.

use core::num::traits::{Sqrt, WideMul};

pub const ONE: i128 = 0x10000000000000000;
const ONE_U128: u128 = 0x10000000000000000;
const ONE_FELT: felt252 = 0x10000000000000000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct I128Q64 {
    pub raw: i128,
}

#[inline(always)]
fn abs_sign(v: i128) -> (u128, bool) {
    if v < 0 {
        ((-v).try_into().unwrap(), true)
    } else {
        (v.try_into().unwrap(), false)
    }
}

#[inline(always)]
fn with_sign(mag: u128, sign: bool) -> i128 {
    let m: i128 = mag.try_into().expect('I128Q64 overflow');
    if sign {
        -m
    } else {
        m
    }
}

pub impl I128Q64Add of Add<I128Q64> {
    #[inline(always)]
    fn add(lhs: I128Q64, rhs: I128Q64) -> I128Q64 {
        I128Q64 { raw: lhs.raw + rhs.raw }
    }
}

pub impl I128Q64Sub of Sub<I128Q64> {
    #[inline(always)]
    fn sub(lhs: I128Q64, rhs: I128Q64) -> I128Q64 {
        I128Q64 { raw: lhs.raw - rhs.raw }
    }
}

pub impl I128Q64Neg of Neg<I128Q64> {
    #[inline(always)]
    fn neg(a: I128Q64) -> I128Q64 {
        I128Q64 { raw: -a.raw }
    }
}

pub impl I128Q64Mul of Mul<I128Q64> {
    #[inline(always)]
    fn mul(lhs: I128Q64, rhs: I128Q64) -> I128Q64 {
        let (am, a_neg) = abs_sign(lhs.raw);
        let (bm, b_neg) = abs_sign(rhs.raw);
        let p: u256 = am.wide_mul(bm);
        // mag = high * 2^64 + low / 2^64, recombined in felt252 (cheaper than a checked u128 mul).
        let mag: felt252 = p.high.into() * ONE_FELT + (p.low / ONE_U128).into();
        let m: i128 = mag.try_into().expect('I128Q64 mul overflow');
        I128Q64 { raw: if a_neg ^ b_neg {
            -m
        } else {
            m
        } }
    }
}

pub impl I128Q64Div of Div<I128Q64> {
    #[inline(always)]
    fn div(lhs: I128Q64, rhs: I128Q64) -> I128Q64 {
        let (am, a_neg) = abs_sign(lhs.raw);
        let (bm, b_neg) = abs_sign(rhs.raw);
        let n: u256 = am.wide_mul(ONE_U128);
        let q: u256 = n / bm.into();
        assert(q.high == 0, 'I128Q64 div overflow');
        I128Q64 { raw: with_sign(q.low, a_neg ^ b_neg) }
    }
}

pub impl I128Q64PartialOrd of PartialOrd<I128Q64> {
    #[inline(always)]
    fn lt(lhs: I128Q64, rhs: I128Q64) -> bool {
        lhs.raw < rhs.raw
    }
}

#[inline(always)]
pub fn abs(a: I128Q64) -> I128Q64 {
    if a.raw < 0 {
        I128Q64 { raw: -a.raw }
    } else {
        a
    }
}

#[inline(always)]
pub fn from_int(v: i64) -> I128Q64 {
    let f: felt252 = v.into() * ONE_FELT;
    I128Q64 { raw: f.try_into().unwrap() }
}

/// Floor to integer.
#[inline(always)]
pub fn to_int(a: I128Q64) -> i64 {
    let (q, r) = DivRem::div_rem(a.raw, 0x10000000000000000);
    let q: i64 = q.try_into().expect('I128Q64 to_int overflow');
    if r < 0 {
        q - 1
    } else {
        q
    }
}

/// sqrt through the corelib `u256` integer square root of the widened value.
#[inline(always)]
pub fn sqrt(a: I128Q64) -> I128Q64 {
    let m: u128 = a.raw.try_into().expect('sqrt of negative');
    let wide: u256 = m.wide_mul(ONE_U128);
    let r: u128 = wide.sqrt();
    I128Q64 { raw: r.try_into().unwrap() }
}

// --- BoundedInt-powered Q64.64 multiply ---------------------------------------------------------
//
// a = ah * 2^64 + al (ah signed 64 bits, al unsigned 64 bits) obtained with ONE div_rem of the
// offset value a + 2^127; then floor(a * b / 2^64) = ah*bh*2^64 + (ah*bl + al*bh) + floor(al*bl /
// 2^64)
// is accumulated in the BoundedInt domain (max ~2^190, no wrap) and range-checked once by the
// final felt252 -> i128 conversion. Branch-free; rounding: floor.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, upcast,
};

const TWO64_F: felt252 = 0x10000000000000000;
const TWO127_F: felt252 = 0x80000000000000000000000000000000;
const TWO63_F: felt252 = 0x8000000000000000;
type U64Like = BoundedInt<0x0, 0xffffffffffffffff>;
type I64Like = BoundedInt<-0x8000000000000000, 0x7fffffffffffffff>;
type HH = BoundedInt<-0x3fffffffffffffff8000000000000000, 0x40000000000000000000000000000000>;
type HL = BoundedInt<-0x7fffffffffffffff8000000000000000, 0x7ffffffffffffffe8000000000000001>;
type Mid = BoundedInt<-0xffffffffffffffff0000000000000000, 0xfffffffffffffffd0000000000000002>;
type LL = BoundedInt<0x0, 0xfffffffffffffffe0000000000000001>;
type LLQ = BoundedInt<0x0, 0xfffffffffffffffe>;
type HH64 =
    BoundedInt<
        -0x3fffffffffffffff80000000000000000000000000000000,
        0x400000000000000000000000000000000000000000000000,
    >;
type S1 =
    BoundedInt<
        -0x40000000000000007fffffffffffffff0000000000000000,
        0x4000000000000000fffffffffffffffd0000000000000002,
    >;
type S2 =
    BoundedInt<
        -0x40000000000000007fffffffffffffff0000000000000000,
        0x4000000000000000fffffffffffffffe0000000000000000,
    >;
type U128Like = BoundedInt<0, 0xffffffffffffffffffffffffffffffff>;
impl AddI128Off of AddHelper<i128, UnitInt<TWO127_F>> {
    type Result = U128Like;
}
impl DivU128Two64 of DivRemHelper<U128Like, UnitInt<TWO64_F>> {
    type DivT = U64Like;
    type RemT = U64Like;
}
impl SubU64Two63 of SubHelper<U64Like, UnitInt<TWO63_F>> {
    type Result = I64Like;
}
impl MulHH of MulHelper<I64Like, I64Like> {
    type Result = HH;
}
impl MulHL of MulHelper<I64Like, U64Like> {
    type Result = HL;
}
impl MulLL of MulHelper<U64Like, U64Like> {
    type Result = LL;
}
impl AddHLHL of AddHelper<HL, HL> {
    type Result = Mid;
}
impl DivLLTwo64 of DivRemHelper<LL, UnitInt<TWO64_F>> {
    type DivT = LLQ;
    type RemT = U64Like;
}
impl MulHHTwo64 of MulHelper<HH, UnitInt<TWO64_F>> {
    type Result = HH64;
}
impl AddHH64Mid of AddHelper<HH64, Mid> {
    type Result = S1;
}
impl AddS1LLQ of AddHelper<S1, LLQ> {
    type Result = S2;
}

#[inline(always)]
fn split(v: i128) -> (I64Like, U64Like) {
    let off: U128Like = bounded_int::add::<
        i128, UnitInt<TWO127_F>,
    >(v, 0x80000000000000000000000000000000);
    let (hi, lo) = bounded_int::div_rem::<U128Like, UnitInt<TWO64_F>>(off, 0x10000000000000000);
    (bounded_int::sub::<U64Like, UnitInt<TWO63_F>>(hi, 0x8000000000000000), lo)
}

#[inline(always)]
pub fn mul_bounded(a: I128Q64, b: I128Q64) -> I128Q64 {
    let (ah, al) = split(a.raw);
    let (bh, bl) = split(b.raw);
    let hh: HH64 = bounded_int::mul::<
        HH, UnitInt<TWO64_F>,
    >(bounded_int::mul::<I64Like, I64Like>(ah, bh), 0x10000000000000000);
    let mid: Mid = bounded_int::add::<
        HL, HL,
    >(bounded_int::mul::<I64Like, U64Like>(ah, bl), bounded_int::mul::<I64Like, U64Like>(bh, al));
    let (llq, _) = bounded_int::div_rem::<
        LL, UnitInt<TWO64_F>,
    >(bounded_int::mul::<U64Like, U64Like>(al, bl), 0x10000000000000000);
    let s: S2 = bounded_int::add::<S1, LLQ>(bounded_int::add::<HH64, Mid>(hh, mid), llq);
    let s: felt252 = upcast(s);
    I128Q64 { raw: s.try_into().expect('I128Q64 mul overflow') }
}

/// Lazy dot product: the three 2^128-scaled partial sums are accumulated first, ONE final division.
#[inline(always)]
pub fn dot3_bounded(
    ax: I128Q64, ay: I128Q64, az: I128Q64, bx: I128Q64, by: I128Q64, bz: I128Q64,
) -> I128Q64 {
    // Not fully fused (the low*low carries are divided per product) but the expensive checked
    // conversion back to i128 happens once.
    let s = mul_wide(ax, bx) + mul_wide(ay, by) + mul_wide(az, bz);
    I128Q64 { raw: s.try_into().expect('I128Q64 dot overflow') }
}

#[inline(always)]
fn mul_wide(a: I128Q64, b: I128Q64) -> felt252 {
    let (ah, al) = split(a.raw);
    let (bh, bl) = split(b.raw);
    let hh: HH64 = bounded_int::mul::<
        HH, UnitInt<TWO64_F>,
    >(bounded_int::mul::<I64Like, I64Like>(ah, bh), 0x10000000000000000);
    let mid: Mid = bounded_int::add::<
        HL, HL,
    >(bounded_int::mul::<I64Like, U64Like>(ah, bl), bounded_int::mul::<I64Like, U64Like>(bh, al));
    let (llq, _) = bounded_int::div_rem::<
        LL, UnitInt<TWO64_F>,
    >(bounded_int::mul::<U64Like, U64Like>(al, bl), 0x10000000000000000);
    upcast(bounded_int::add::<S1, LLQ>(bounded_int::add::<HH64, Mid>(hh, mid), llq))
}
