//! Implementation candidates that LOST the gas comparison (or that implement a rejected
//! semantic), kept with their benchmarks so that the evidence stays in the repository
//! (AGENTS.md, efficiency rule 8). Test-only: nothing here is exported.
//!
//! The benchmarks join the group of the exported operation (`bench_<op>__alt_<candidate>`), whose
//! baseline and winner (`fixed*` variants) live next to the public function. Raw `i64` inputs cost
//! exactly the same as `Fixed` ones (`bench_mul__alt_kernel_raw_pn` is the control).

use core::num::traits::{CheckedAdd, Sqrt};
use crate::errors;
#[feature("bounded-int-utils")]
use super::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, downcast, upcast,
};
use super::{
    AddI64Two63, AddProdProd, AddSum2Prod, AddSum3Prod, ConstrainDiffI640, ConstrainHelper, DiffI64,
    DivTwo64NonNeg, DivU64Two32, I64Neg, I64NegNegated, I64Negated, I64NonNeg, I64Sym, I64SymNeg,
    MulI64I64, MulI64MinusOne, OptionRev, Prod, SubI64I64, SubU64Two63, Sum2, Sum3, Sum4, TWO32,
    TWO63, TWO95, U32Like, U64Like, U96Like,
};

const ONE: i64 = 0x100000000;

// --- add / sub / neg ---------------------------------------------------------------------------

/// Native checked operator: same cost, but panics with the corelib message ('i64_add Overflow').
#[inline(always)]
pub fn add_native(a: i64, b: i64) -> i64 {
    a + b
}

#[inline(always)]
pub fn add_checked(a: i64, b: i64) -> i64 {
    a.checked_add(b).expect(errors::OVERFLOW)
}

#[inline(always)]
pub fn sub_native(a: i64, b: i64) -> i64 {
    a - b
}

#[inline(always)]
pub fn neg_native(a: i64) -> i64 {
    -a
}

/// Negation as a bounded multiplication + two-sided range check.
#[inline(always)]
pub fn neg_downcast(a: i64) -> i64 {
    let n: I64Negated = bounded_int::mul::<i64, UnitInt<-1>>(a, -1);
    downcast(n).expect(errors::OVERFLOW)
}

#[inline(always)]
pub fn abs_native(a: i64) -> i64 {
    if a < 0 {
        -a
    } else {
        a
    }
}

/// The first exported version: `MIN` trimmed first (on both paths), then the sign split.
#[inline(always)]
pub fn abs_trim_first(a: i64) -> i64 {
    match bounded_int::trim_min::<i64>(a) {
        OptionRev::None => core::panic_with_felt252(errors::OVERFLOW),
        OptionRev::Some(a) => match bounded_int::constrain::<I64Sym, 0>(a) {
            Ok(lt0) => upcast(bounded_int::mul::<I64SymNeg, UnitInt<-1>>(lt0, -1)),
            Err(ge0) => upcast(ge0),
        },
    }
}
impl ConstrainI64Sym0 of ConstrainHelper<I64Sym, 0> {
    type LowT = I64SymNeg;
    type HighT = I64NonNeg;
}

/// Sign split, then a two-sided range check of the negated value (instead of trimming `MIN`).
#[inline(always)]
pub fn abs_downcast(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(lt0) => {
            let n: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(lt0, -1);
            downcast(n).expect(errors::OVERFLOW)
        },
        Err(ge0) => upcast(ge0),
    }
}

// --- comparisons -------------------------------------------------------------------------------

/// `a < b` as the sign of the exact difference: a tie with the native comparison (and +400 gas
/// when written `constrain(..).is_ok()` instead of an explicit `match`).
#[allow(manual_is_ok)]
#[inline(always)]
pub fn lt_constrain(a: i64, b: i64) -> bool {
    let d: DiffI64 = bounded_int::sub::<i64, i64>(a, b);
    match bounded_int::constrain::<DiffI64, 0>(d) {
        Ok(_) => true,
        Err(_) => false,
    }
}

#[inline(always)]
pub fn min_constrain(a: i64, b: i64) -> i64 {
    let d: DiffI64 = bounded_int::sub::<i64, i64>(a, b);
    match bounded_int::constrain::<DiffI64, 0>(d) {
        Ok(_) => a,
        Err(_) => b,
    }
}

/// signum with two native comparisons.
#[inline(always)]
pub fn signum_native(a: i64) -> i64 {
    if a > 0 {
        ONE
    } else if a < 0 {
        -ONE
    } else {
        0
    }
}

#[inline(always)]
pub fn is_negative_native(a: i64) -> bool {
    a < 0
}

#[inline(always)]
pub fn is_positive_native(a: i64) -> bool {
    a > 0
}

/// `constrain(..).is_ok()` instead of an explicit `match`: +500 gas.
#[inline(always)]
pub fn is_negative_is_ok(a: i64) -> bool {
    bounded_int::constrain::<i64, 0>(a).is_ok()
}

#[inline(always)]
pub fn abs_diff_le_native(a: i64, b: i64, ulps: u64) -> bool {
    let d: i128 = a.into() - b.into();
    let d: i128 = if d < 0 {
        -d
    } else {
        d
    };
    d <= ulps.into()
}

// --- mul ---------------------------------------------------------------------------------------

type Wide129 =
    BoundedInt<-0x200000000000000000000000000000000, 0x200000000000000000000000000000000>;
type Wide129Off = BoundedInt<0, 0x400000000000000000000000000000000>;
type Quot129Off = BoundedInt<0, 0x4000000000000000000000000>;
type Quot129 = BoundedInt<-0x2000000000000000000000000, 0x2000000000000000000000000>;
impl AddWide129Off of AddHelper<Wide129, UnitInt<0x200000000000000000000000000000000>> {
    type Result = Wide129Off;
}
impl DivWide129Off of DivRemHelper<Wide129Off, UnitInt<TWO32>> {
    type DivT = Quot129Off;
    type RemT = U32Like;
}
impl SubQuot129Off of SubHelper<Quot129Off, UnitInt<0x2000000000000000000000000>> {
    type Result = Quot129;
}

/// The prototype's rescale (`benchmarks/scalar`): typed accumulator (up to 8 products), bias,
/// divide, THEN check the quotient. +100 gas vs checking the biased value before the division.
#[inline(always)]
pub fn rescale_postcheck(w: Wide129) -> i64 {
    let off: Wide129Off = bounded_int::add(w, 0x200000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem(off, 0x100000000);
    let q: Quot129 = bounded_int::sub(q, 0x2000000000000000000000000);
    downcast(q).expect(errors::OVERFLOW)
}

#[inline(always)]
pub fn mul_postcheck(a: i64, b: i64) -> i64 {
    rescale_postcheck(upcast(bounded_int::mul::<i64, i64>(a, b)))
}

impl DivU128Two32 of DivRemHelper<u128, UnitInt<TWO32>> {
    type DivT = U96Like;
    type RemT = U32Like;
}

/// Bias, felt252 -> u128, divide, then a one-sided check of the quotient.
#[inline(always)]
pub fn mul_via_u128(a: i64, b: i64) -> i64 {
    let p: felt252 = upcast(bounded_int::mul::<i64, i64>(a, b));
    let p: u128 = (p + TWO95).try_into().expect(errors::OVERFLOW);
    let (q, _r) = bounded_int::div_rem::<u128, UnitInt<TWO32>>(p, 0x100000000);
    let q: U64Like = downcast(q).expect(errors::OVERFLOW);
    upcast(bounded_int::sub::<U64Like, UnitInt<TWO63>>(q, 0x8000000000000000))
}

/// Corelib path (DESIGN D2 fallback): i128 product, signed division. TRUNCATES toward zero.
#[inline(always)]
pub fn mul_native_trunc(a: i64, b: i64) -> i64 {
    let p: i128 = a.into() * b.into();
    (p / 0x100000000).try_into().expect(errors::OVERFLOW)
}

// --- div / rem / recip ---------------------------------------------------------------------------

type NumAbs = BoundedInt<0, 0x800000000000000000000000>;
type NegQuot = BoundedInt<-0x7fffffffffffffff, 0>;
impl MulNonNegTwo32 of MulHelper<I64NonNeg, UnitInt<TWO32>> {
    type Result = BoundedInt<0, 0x7fffffffffffffff00000000>;
}
impl MulNegNegatedTwo32 of MulHelper<I64NegNegated, UnitInt<TWO32>> {
    type Result = BoundedInt<0x100000000, 0x800000000000000000000000>;
}
impl DivNumAbsNonNeg of DivRemHelper<NumAbs, I64NonNeg> {
    type DivT = NumAbs;
    type RemT = BoundedInt<0, 0x7ffffffffffffffe>;
}
impl DivNumAbsNegNegated of DivRemHelper<NumAbs, I64NegNegated> {
    type DivT = NumAbs;
    type RemT = I64NonNeg;
}
impl SubNegQuotOne of SubHelper<NegQuot, UnitInt<1>> {
    type Result = I64Neg;
}

#[inline(always)]
fn num_abs(a: i64) -> (NumAbs, bool) {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(lt0) => {
            let m: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(lt0, -1);
            (upcast(bounded_int::mul::<I64NegNegated, UnitInt<TWO32>>(m, 0x100000000)), true)
        },
        Err(ge0) => (
            upcast(bounded_int::mul::<I64NonNeg, UnitInt<TWO32>>(ge0, 0x100000000)), false,
        ),
    }
}

#[inline(always)]
fn div_finish(q: NumAbs, inexact: bool, neg: bool, floor: bool) -> i64 {
    let q: I64NonNeg = downcast(q).expect(errors::OVERFLOW);
    if neg {
        let m: NegQuot = bounded_int::mul::<I64NonNeg, UnitInt<-1>>(q, -1);
        if floor && inexact {
            upcast(bounded_int::sub::<NegQuot, UnitInt<1>>(m, 1))
        } else {
            upcast(m)
        }
    } else {
        upcast(q)
    }
}

#[inline(always)]
fn div_peel(a: i64, b: i64, floor: bool) -> i64 {
    let d: NonZero<i64> = b.try_into().expect(errors::DIVISION_BY_ZERO);
    let (n, n_neg) = num_abs(a);
    match bounded_int::constrain::<NonZero<i64>, 0>(d) {
        Ok(d_lt0) => {
            let d = bounded_int::mul::<NonZero<I64Neg>, NonZero<UnitInt<-1>>>(d_lt0, -1);
            let (q, r) = bounded_int::div_rem::<NumAbs, I64NegNegated>(n, d);
            div_finish(q, upcast::<_, felt252>(r) != 0, !n_neg, floor)
        },
        Err(d_ge0) => {
            let (q, r) = bounded_int::div_rem::<NumAbs, I64NonNeg>(n, d_ge0);
            div_finish(q, upcast::<_, felt252>(r) != 0, n_neg, floor)
        },
    }
}

/// The prototype's floor division: both signs peeled off, unsigned division, floor fix-up from
/// the remainder.
#[inline(always)]
pub fn div_floor_peel(a: i64, b: i64) -> i64 {
    div_peel(a, b, true)
}

/// Same with TRUNCATION toward zero (no fix-up): the cheapest truncating division found.
#[inline(always)]
pub fn div_trunc_peel(a: i64, b: i64) -> i64 {
    div_peel(a, b, false)
}

/// Corelib path: i128 signed division. TRUNCATES toward zero.
#[inline(always)]
pub fn div_native_trunc(a: i64, b: i64) -> i64 {
    assert(b != 0, errors::DIVISION_BY_ZERO);
    let n: i128 = a.into() * 0x100000000_i128;
    (n / b.into()).try_into().expect(errors::OVERFLOW)
}

/// Corelib `i64` remainder: TRUNCATED (sign of the dividend).
#[inline(always)]
pub fn rem_native_trunc(a: i64, b: i64) -> i64 {
    assert(b != 0, errors::DIVISION_BY_ZERO);
    a % b
}

/// Reciprocal through the generic division kernel.
#[inline(always)]
pub fn recip_div(a: i64) -> i64 {
    super::div(ONE, a)
}

// --- rounding ------------------------------------------------------------------------------------

type FloorSubRem = BoundedInt<-0x80000000ffffffff, 0x7fffffffffffffff>;
impl SubI64U32 of SubHelper<i64, U32Like> {
    type Result = FloorSubRem;
}

/// floor as `a - (a mod 2^32)`: the subtraction needs a range check the quotient form avoids.
#[inline(always)]
pub fn floor_sub_rem(a: i64) -> i64 {
    let off: U64Like = bounded_int::add::<i64, UnitInt<TWO63>>(a, 0x8000000000000000);
    let (_q, r) = bounded_int::div_rem::<U64Like, UnitInt<TWO32>>(off, 0x100000000);
    let f: FloorSubRem = bounded_int::sub::<i64, U32Like>(a, r);
    downcast(f).expect(errors::OVERFLOW)
}

/// floor with the corelib signed `DivRem` (truncated) and a sign fix-up.
#[inline(always)]
pub fn floor_native(a: i64) -> i64 {
    let (_q, r) = DivRem::div_rem(a, 0x100000000);
    if r < 0 {
        a - r - ONE
    } else {
        a - r
    }
}

/// ceil as `-floor(-a)`.
#[inline(always)]
pub fn ceil_neg_floor(a: i64) -> i64 {
    super::neg(super::floor(super::neg(a)))
}

/// ceil as `floor(a)`, plus one when the fraction is not zero.
#[inline(always)]
pub fn ceil_branch(a: i64) -> i64 {
    let off: U64Like = bounded_int::add::<i64, UnitInt<TWO63>>(a, 0x8000000000000000);
    let (_q, r) = bounded_int::div_rem::<U64Like, UnitInt<TWO32>>(off, 0x100000000);
    if upcast::<_, felt252>(r) == 0 {
        a
    } else {
        super::add(super::floor(a), ONE)
    }
}

type RoundUpOff = BoundedInt<0x80000000, 0x1000000007fffffff>;
type RoundUpQuot = BoundedInt<0, 0x100000000>;
type RoundUpShl = BoundedInt<0, 0x10000000000000000>;
type RoundUpResult = BoundedInt<-0x8000000000000000, 0x8000000000000000>;
impl AddI64RoundUpBias of AddHelper<i64, UnitInt<0x8000000080000000>> {
    type Result = RoundUpOff;
}
impl DivRoundUpOff of DivRemHelper<RoundUpOff, UnitInt<TWO32>> {
    type DivT = RoundUpQuot;
    type RemT = U32Like;
}
impl MulRoundUpQuot of MulHelper<RoundUpQuot, UnitInt<TWO32>> {
    type Result = RoundUpShl;
}
impl SubRoundUpShl of SubHelper<RoundUpShl, UnitInt<TWO63>> {
    type Result = RoundUpResult;
}

/// REJECTED SEMANTIC: round half UP (`floor(a + 1/2)`, `round(-2.5) = -2`). Branch-free, but
/// upstream `f64::round` rounds ties away from zero.
#[inline(always)]
pub fn round_half_up(a: i64) -> i64 {
    let off: RoundUpOff = bounded_int::add::<
        i64, UnitInt<0x8000000080000000>,
    >(a, 0x8000000080000000);
    let (q, _r) = bounded_int::div_rem::<RoundUpOff, UnitInt<TWO32>>(off, 0x100000000);
    let shl: RoundUpShl = bounded_int::mul::<RoundUpQuot, UnitInt<TWO32>>(q, 0x100000000);
    let r: RoundUpResult = bounded_int::sub::<RoundUpShl, UnitInt<TWO63>>(shl, 0x8000000000000000);
    downcast(r).expect(errors::OVERFLOW)
}

/// trunc as floor, plus one for negative non-integers.
#[inline(always)]
pub fn trunc_floor_fix(a: i64) -> i64 {
    let off: U64Like = bounded_int::add::<i64, UnitInt<TWO63>>(a, 0x8000000000000000);
    let (_q, r) = bounded_int::div_rem::<U64Like, UnitInt<TWO32>>(off, 0x100000000);
    let r: i64 = upcast(r);
    if a < 0 && r != 0 {
        a - r + ONE
    } else {
        a - r
    }
}

/// fract as `a - trunc(a)` with the checked subtraction.
#[inline(always)]
pub fn fract_sub_trunc(a: i64) -> i64 {
    super::sub(a, super::trunc(a))
}

/// Corelib path: signed `DivRem` (truncated) + floor fix-up + narrowing.
#[inline(always)]
pub fn to_i32_native(a: i64) -> i32 {
    let (q, r) = DivRem::div_rem(a, 0x100000000);
    let q = if r < 0 {
        q - 1
    } else {
        q
    };
    q.try_into().unwrap()
}

/// Corelib path: widening + checked `i64` multiplication.
#[inline(always)]
pub fn from_i32_native(v: i32) -> i64 {
    v.into() * ONE
}

// --- sqrt ----------------------------------------------------------------------------------------

/// The prototype's inverse square root: `2^64 / floor(sqrt(a * 2^32))`: two roundings (the
/// result can be 1 ulp above the exactly floored value) and an overflow check.
#[inline(always)]
pub fn inv_sqrt_two_step(a: i64) -> i64 {
    let s: NonZero<i64> = super::sqrt(a).try_into().expect(errors::DIVISION_BY_ZERO);
    match bounded_int::constrain::<NonZero<i64>, 0>(s) {
        Ok(_) => core::panic_with_felt252(errors::SQRT_OF_NEGATIVE),
        Err(s) => {
            let (q, _r) = bounded_int::div_rem::<
                UnitInt<0x10000000000000000>, I64NonNeg,
            >(0x10000000000000000, s);
            downcast(q).expect(errors::OVERFLOW)
        },
    }
}

/// `recip(sqrt(a))` with the exported kernels.
#[inline(always)]
pub fn inv_sqrt_recip(a: i64) -> i64 {
    super::recip(super::sqrt(a))
}

/// The prototype's sqrt: sign split with `constrain` (+100 gas vs the `i64 -> u64` range check).
#[inline(always)]
pub fn sqrt_constrain(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(_) => core::panic_with_felt252(errors::SQRT_OF_NEGATIVE),
        Err(a) => {
            let wide: u128 = upcast(bounded_int::mul::<I64NonNeg, UnitInt<TWO32>>(a, 0x100000000));
            let r: u64 = wide.sqrt();
            downcast(r).expect(errors::OVERFLOW)
        },
    }
}

/// Exact inverse square root with `NonZero` + `constrain` instead of the `i64 -> u64` range check.
#[inline(always)]
pub fn inv_sqrt_constrain(a: i64) -> i64 {
    let a: NonZero<i64> = a.try_into().expect(errors::DIVISION_BY_ZERO);
    match bounded_int::constrain::<NonZero<i64>, 0>(a) {
        Ok(_) => core::panic_with_felt252(errors::SQRT_OF_NEGATIVE),
        Err(a) => {
            let (q, _r) = bounded_int::div_rem::<
                UnitInt<0x1000000000000000000000000>, I64NonNeg,
            >(0x1000000000000000000000000, a);
            let q: u128 = upcast(q);
            let r: u64 = q.sqrt();
            downcast(r).expect(errors::OVERFLOW)
        },
    }
}
impl DivTwo96NonNeg of DivRemHelper<UnitInt<0x1000000000000000000000000>, I64NonNeg> {
    type DivT = BoundedInt<0x200000000, 0x1000000000000000000000000>;
    type RemT = BoundedInt<0, 0x7ffffffffffffffe>;
}

/// Corelib sqrt: `i64 -> u64` conversion and `u64` wide mul. A tie with the exported kernel.
#[inline(always)]
pub fn sqrt_native(a: i64) -> i64 {
    let m: u64 = a.try_into().expect(errors::SQRT_OF_NEGATIVE);
    let wide: u128 = core::num::traits::WideMul::wide_mul(m, 0x100000000_u64);
    let r: u64 = wide.sqrt();
    r.try_into().expect(errors::OVERFLOW)
}

// --- accumulation of 8 products
// --------------------------------------------------------------------

type Sum5 = BoundedInt<-0x13ffffffffffffffd8000000000000000, 0x140000000000000000000000000000000>;
type Sum6 = BoundedInt<-0x17ffffffffffffffd0000000000000000, 0x180000000000000000000000000000000>;
type Sum7 = BoundedInt<-0x1bffffffffffffffc8000000000000000, 0x1c0000000000000000000000000000000>;
type Sum8 = BoundedInt<-0x1fffffffffffffffc0000000000000000, 0x200000000000000000000000000000000>;
impl AddSum4Prod of AddHelper<Sum4, Prod> {
    type Result = Sum5;
}
impl AddSum5Prod of AddHelper<Sum5, Prod> {
    type Result = Sum6;
}
impl AddSum6Prod of AddHelper<Sum6, Prod> {
    type Result = Sum7;
}
impl AddSum7Prod of AddHelper<Sum7, Prod> {
    type Result = Sum8;
}

#[inline(always)]
fn sum8_typed(a: [i64; 8], b: [i64; 8]) -> Sum8 {
    let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
    let [b0, b1, b2, b3, b4, b5, b6, b7] = b;
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(a0, b0), bounded_int::mul(a1, b1));
    let s: Sum3 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a2, b2));
    let s: Sum4 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a3, b3));
    let s: Sum5 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a4, b4));
    let s: Sum6 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a5, b5));
    let s: Sum7 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a6, b6));
    bounded_int::add(s, bounded_int::mul::<i64, i64>(a7, b7))
}

/// Accumulator candidate 1: typed `BoundedInt` chain (one type per length), then the same
/// pre-checked `rescale` as the exported kernels. Same gas as the felt252 `Wide`, but the
/// accumulator type changes at every step, which cannot be exposed as ONE public type.
#[inline(always)]
pub fn sum_prod8_typed(a: [i64; 8], b: [i64; 8]) -> i64 {
    super::rescale(upcast(sum8_typed(a, b)))
}

/// Accumulator candidate 2: typed chain + the prototype's post-checked typed rescale.
#[inline(always)]
pub fn sum_prod8_typed_postcheck(a: [i64; 8], b: [i64; 8]) -> i64 {
    rescale_postcheck(upcast(sum8_typed(a, b)))
}

/// Accumulator candidate 3: two fused 4-term kernels and a checked addition (TWO roundings:
/// not bit-identical to the exact sum).
#[inline(always)]
pub fn sum_prod8_chunked(a: [i64; 8], b: [i64; 8]) -> i64 {
    let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
    let [b0, b1, b2, b3, b4, b5, b6, b7] = b;
    super::add(
        super::sum_prod4(a0, b0, a1, b1, a2, b2, a3, b3),
        super::sum_prod4(a4, b4, a5, b5, a6, b6, a7, b7),
    )
}

/// Accumulator candidate 4: corelib `i128` accumulation of `i64` wide products (checked adds,
/// cannot hold 4 full-range products), signed division. TRUNCATES.
#[inline(always)]
pub fn sum_prod8_native_trunc(a: [i64; 8], b: [i64; 8]) -> i64 {
    let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
    let [b0, b1, b2, b3, b4, b5, b6, b7] = b;
    let s: i128 = a0.into() * b0.into()
        + a1.into() * b1.into()
        + a2.into() * b2.into()
        + a3.into() * b3.into()
        + a4.into() * b4.into()
        + a5.into() * b5.into()
        + a6.into() * b6.into()
        + a7.into() * b7.into();
    (s / 0x100000000).try_into().expect(errors::OVERFLOW)
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use super::*;

    const HALF: i64 = 0x80000000;
    const THREE: i64 = 0x300000000;
    const MIN: i64 = -0x8000000000000000;
    const MAX: i64 = 0x7fffffffffffffff;

    // The candidates implementing the exported semantic must agree with the exported kernels;
    // the others document how the rejected semantic differs.

    #[test]
    fn test_additive_candidates_agree() {
        assert!(add_native(MAX, MIN) == -1);
        assert!(add_checked(MAX, MIN) == -1);
        assert!(sub_native(-1, MAX) == MIN);
        assert!(neg_native(MAX) == MIN + 1);
        assert!(neg_downcast(MAX) == MIN + 1);
        assert!(neg_downcast(MIN + 1) == MAX);
        assert!(abs_native(MIN + 1) == MAX);
        assert!(abs_downcast(MIN + 1) == MAX);
        assert!(abs_downcast(MAX) == MAX);
        assert!(abs_trim_first(MIN + 1) == MAX);
        assert!(abs_trim_first(MAX) == MAX);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_neg_downcast_min_panics() {
        neg_downcast(black_box(MIN));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_abs_downcast_min_panics() {
        abs_downcast(black_box(MIN));
    }

    #[test]
    fn test_comparison_candidates_agree() {
        assert!(lt_constrain(MIN, MAX));
        assert!(!lt_constrain(MAX, MIN));
        assert!(!lt_constrain(ONE, ONE));
        assert!(min_constrain(MIN, MAX) == MIN);
        assert!(min_constrain(MAX, MIN) == MIN);
        assert!(signum_native(MIN) == -ONE);
        assert!(signum_native(0) == 0);
        assert!(signum_native(1) == ONE);
        assert!(is_negative_native(-1) && !is_negative_native(0));
        assert!(is_negative_is_ok(-1) && !is_negative_is_ok(0));
        assert!(is_positive_native(1) && !is_positive_native(0));
        assert!(abs_diff_le_native(MIN, MAX, 0xffffffffffffffff));
        assert!(!abs_diff_le_native(MAX, MIN, 0xfffffffffffffffe));
    }

    #[test]
    fn test_mul_candidates_agree_on_floor() {
        assert!(mul_postcheck(-1, HALF) == -1);
        assert!(mul_via_u128(-1, HALF) == -1);
        assert!(mul_postcheck(MIN, ONE) == MIN);
        assert!(mul_via_u128(MIN, ONE) == MIN);
        assert!(mul_postcheck(MAX, -ONE) == -MAX);
        assert!(mul_via_u128(MAX, -ONE) == -MAX);
    }

    #[test]
    fn test_mul_native_truncates() {
        assert!(mul_native_trunc(-1, HALF) == 0);
        assert!(super::super::mul(-1, HALF) == -1);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_postcheck_overflow_panics() {
        mul_postcheck(black_box(MIN), MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_via_u128_overflow_panics() {
        mul_via_u128(black_box(MIN), MIN);
    }

    #[test]
    fn test_div_candidates() {
        assert!(div_floor_peel(-ONE, THREE) == -1431655766);
        assert!(div_floor_peel(ONE, -THREE) == -1431655766);
        assert!(div_floor_peel(-ONE, -THREE) == 1431655765);
        assert!(div_floor_peel(-2 * THREE, THREE) == -2 * ONE);
        assert!(div_floor_peel(MIN + 1, -ONE) == MAX);
        // Truncation: one ulp above the floor for negative inexact quotients.
        assert!(div_trunc_peel(-ONE, THREE) == -1431655765);
        assert!(div_native_trunc(-ONE, THREE) == -1431655765);
        assert!(div_trunc_peel(ONE, THREE) == 1431655765);
        assert!(rem_native_trunc(-7 * ONE, THREE) == -ONE);
        assert!(super::super::rem(-7 * ONE, THREE) == 2 * ONE);
        assert!(recip_div(THREE) == 1431655765);
        assert!(recip_div(-THREE) == -1431655766);
        assert!(recip_div(-2) == MIN);
    }

    #[test]
    fn test_rounding_candidates_agree() {
        let values: [i64; 8] = [
            0x280000000, -0x280000000, 0x300000000, -0x300000000, 1, -1, 0, MIN + 1,
        ];
        for v in values.span() {
            let v = *v;
            assert!(floor_sub_rem(v) == super::super::floor(v));
            assert!(floor_native(v) == super::super::floor(v));
            assert!(ceil_neg_floor(v) == super::super::ceil(v));
            assert!(ceil_branch(v) == super::super::ceil(v));
            assert!(trunc_floor_fix(v) == super::super::trunc(v));
            assert!(fract_sub_trunc(v) == super::super::fract(v));
            assert!(to_i32_native(v) == super::super::to_i32(v));
        }
        assert!(from_i32_native(-0x80000000) == MIN);
    }

    #[test]
    fn test_round_half_up_differs_on_negative_ties() {
        assert!(round_half_up(0x280000000) == 0x300000000);
        assert!(round_half_up(-0x280000000) == -0x200000000);
        assert!(super::super::round(-0x280000000) == -0x300000000);
        assert!(round_half_up(-0x2c0000000) == -0x300000000);
    }

    #[test]
    fn test_sqrt_candidates() {
        assert!(sqrt_native(2 * ONE) == 6074000999);
        assert!(sqrt_native(MAX) == super::super::sqrt(MAX));
        assert!(sqrt_constrain(MAX) == super::super::sqrt(MAX));
        assert!(inv_sqrt_constrain(4294) == 4295451025709);
        assert!(inv_sqrt_two_step(4 * ONE) == HALF);
        assert!(inv_sqrt_recip(4 * ONE) == HALF);
        // Two roundings: not the exactly floored value (4295451025709).
        assert!(inv_sqrt_two_step(4294) == 4295451646614);
        assert!(super::super::inv_sqrt(4294) == 4295451025709);
    }

    #[test]
    fn test_sum_prod8_candidates() {
        let a: [i64; 8] = [1, 1, 1, 1, 1, 1, 1, 1];
        let b: [i64; 8] = [0x20000000; 8];
        // 8 * (1/8 ulp) = 1 ulp exactly.
        assert!(sum_prod8_typed(a, b) == 1);
        assert!(sum_prod8_typed_postcheck(a, b) == 1);
        // Two roundings: each half floors 0.5 ulp to 0.
        assert!(sum_prod8_chunked(a, b) == 0);
        let a: [i64; 8] = [MIN, MIN, MIN, MIN, MIN, MIN, MIN, MIN];
        let b: [i64; 8] = [MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX];
        // 4 * (2^126 - 2^63 * (2^63 - 1)) = 2^65 unscaled = 2^33 raw.
        assert!(sum_prod8_typed(a, b) == 0x200000000);
        assert!(sum_prod8_typed_postcheck(a, b) == 0x200000000);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod8_typed_largest_accumulation_panics() {
        let a: [i64; 8] = black_box([MIN; 8]);
        sum_prod8_typed(a, a);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_sum_prod8_typed_postcheck_largest_accumulation_panics() {
        let a: [i64; 8] = black_box([MIN; 8]);
        sum_prod8_typed_postcheck(a, a);
    }

    // --- gas benchmarks: variants of the groups whose baseline lives with the exported op --------

    #[test]
    #[inline(never)]
    fn bench_add__alt_native_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(0x140000000_i64);
        assert!(add_native(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_add__alt_checked_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(0x140000000_i64);
        assert!(add_checked(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sub__alt_native_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(0x5c0000000_i64);
        assert!(sub_native(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(-0x380000000_i64);
        assert!(neg_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(0x240000000_i64);
        assert!(neg_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__alt_downcast_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(-0x380000000_i64);
        assert!(neg_downcast(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_neg__alt_downcast_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(0x240000000_i64);
        assert!(neg_downcast(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__alt_kernel_raw_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(-0x7e0000000_i64);
        assert!(super::super::mul(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__alt_postcheck_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(-0x7e0000000_i64);
        assert!(mul_postcheck(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__alt_via_u128_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(-0x7e0000000_i64);
        assert!(mul_via_u128(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__alt_native_trunc_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(-0x7e0000000_i64);
        assert!(mul_native_trunc(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_mul__alt_native_trunc_pp() {
        let a = black_box(0x380000000_i64);
        let b = black_box(0x240000000_i64);
        let e = black_box(0x7e0000000_i64);
        assert!(mul_native_trunc(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_floor_peel_pp() {
        let a = black_box(0x100000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(0x55555555_i64);
        assert!(div_floor_peel(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_floor_peel_np() {
        let a = black_box(-0x100000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(-0x55555556_i64);
        assert!(div_floor_peel(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_floor_peel_pn() {
        let a = black_box(0x100000000_i64);
        let b = black_box(-0x300000000_i64);
        let e = black_box(-0x55555556_i64);
        assert!(div_floor_peel(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_trunc_peel_pp() {
        let a = black_box(0x100000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(0x55555555_i64);
        assert!(div_trunc_peel(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_trunc_peel_np() {
        let a = black_box(-0x100000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(-0x55555555_i64);
        assert!(div_trunc_peel(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_native_trunc_pp() {
        let a = black_box(0x100000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(0x55555555_i64);
        assert!(div_native_trunc(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_div__alt_native_trunc_np() {
        let a = black_box(-0x100000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(-0x55555555_i64);
        assert!(div_native_trunc(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__alt_native_trunc_pp() {
        let a = black_box(0x700000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(0x100000000_i64);
        assert!(rem_native_trunc(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_rem__alt_native_trunc_np() {
        let a = black_box(-0x700000000_i64);
        let b = black_box(0x300000000_i64);
        let e = black_box(-0x100000000_i64);
        assert!(rem_native_trunc(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lt__alt_constrain_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(false);
        assert!(lt_constrain(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lt__alt_constrain_np() {
        let a = black_box(-0x240000000_i64);
        let b = black_box(0x380000000_i64);
        let e = black_box(true);
        assert!(lt_constrain(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x380000000_i64);
        assert!(abs_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(0x240000000_i64);
        assert!(abs_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__alt_downcast_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x380000000_i64);
        assert!(abs_downcast(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__alt_downcast_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(0x240000000_i64);
        assert!(abs_downcast(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x100000000_i64);
        assert!(signum_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x100000000_i64);
        assert!(signum_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__alt_native_z() {
        let a = black_box(0x0_i64);
        let e = black_box(0x0_i64);
        assert!(signum_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_min__alt_constrain_pn() {
        let a = black_box(0x380000000_i64);
        let b = black_box(-0x240000000_i64);
        let e = black_box(-0x240000000_i64);
        assert!(min_constrain(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_min__alt_constrain_np() {
        let a = black_box(-0x240000000_i64);
        let b = black_box(0x380000000_i64);
        let e = black_box(-0x240000000_i64);
        assert!(min_constrain(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__alt_sub_rem_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x300000000_i64);
        assert!(floor_sub_rem(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x300000000_i64);
        assert!(floor_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__alt_sub_rem_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x300000000_i64);
        assert!(floor_sub_rem(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x300000000_i64);
        assert!(floor_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__alt_neg_floor_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x400000000_i64);
        assert!(ceil_neg_floor(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__alt_branch_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x400000000_i64);
        assert!(ceil_branch(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__alt_neg_floor_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x200000000_i64);
        assert!(ceil_neg_floor(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__alt_branch_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x200000000_i64);
        assert!(ceil_branch(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_round__alt_half_up_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x400000000_i64);
        assert!(round_half_up(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_round__alt_half_up_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x200000000_i64);
        assert!(round_half_up(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_trunc__alt_floor_fix_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x300000000_i64);
        assert!(trunc_floor_fix(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_trunc__alt_floor_fix_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x200000000_i64);
        assert!(trunc_floor_fix(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_fract__alt_sub_trunc_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x80000000_i64);
        assert!(fract_sub_trunc(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_fract__alt_sub_trunc_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x40000000_i64);
        assert!(fract_sub_trunc(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_recip__alt_div_p() {
        let a = black_box(0x300000000_i64);
        let e = black_box(0x55555555_i64);
        assert!(recip_div(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_recip__alt_div_n() {
        let a = black_box(-0x300000000_i64);
        let e = black_box(-0x55555556_i64);
        assert!(recip_div(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt__alt_native_p() {
        let a = black_box(0x200000000_i64);
        let e = black_box(0x16a09e667_i64);
        assert!(sqrt_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_inv_sqrt__alt_two_step_p() {
        let a = black_box(0x200000000_i64);
        let e = black_box(0xb504f334_i64);
        assert!(inv_sqrt_two_step(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_inv_sqrt__alt_recip_p() {
        let a = black_box(0x200000000_i64);
        let e = black_box(0xb504f334_i64);
        assert!(inv_sqrt_recip(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_diff_eq__alt_native_lt() {
        let a = black_box(0x380000000_i64);
        let b = black_box(0x380000003_i64);
        let c = black_box(0x3_u64);
        let e = black_box(true);
        assert!(abs_diff_le_native(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_diff_eq__alt_native_gt() {
        let a = black_box(0x380000004_i64);
        let b = black_box(0x380000000_i64);
        let c = black_box(0x3_u64);
        let e = black_box(false);
        assert!(abs_diff_le_native(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_from_int__alt_native_n() {
        let a = black_box(-0x3_i32);
        let e = black_box(-0x300000000_i64);
        assert!(from_i32_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_to_int__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x3_i32);
        assert!(to_i32_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_to_int__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(-0x3_i32);
        assert!(to_i32_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__alt_typed() {
        let a0 = black_box(0x180000000_i64);
        let b0 = black_box(-0x480000000_i64);
        let a1 = black_box(-0x240000000_i64);
        let b1 = black_box(0x40000000_i64);
        let a2 = black_box(0x3c0000000_i64);
        let b2 = black_box(0x200000000_i64);
        let a3 = black_box(-0x80000000_i64);
        let b3 = black_box(0x720000000_i64);
        let a4 = black_box(0x20000000_i64);
        let b4 = black_box(-0x300000000_i64);
        let a5 = black_box(0x680000000_i64);
        let b5 = black_box(0x1c0000000_i64);
        let a6 = black_box(-0x740000000_i64);
        let b6 = black_box(0x80000000_i64);
        let a7 = black_box(0x200000000_i64);
        let b7 = black_box(-0x880000000_i64);
        let e = black_box(-0xd00000000_i64);
        assert!(
            sum_prod8_typed(
                [a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7],
            ) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__alt_typed_postcheck() {
        let a0 = black_box(0x180000000_i64);
        let b0 = black_box(-0x480000000_i64);
        let a1 = black_box(-0x240000000_i64);
        let b1 = black_box(0x40000000_i64);
        let a2 = black_box(0x3c0000000_i64);
        let b2 = black_box(0x200000000_i64);
        let a3 = black_box(-0x80000000_i64);
        let b3 = black_box(0x720000000_i64);
        let a4 = black_box(0x20000000_i64);
        let b4 = black_box(-0x300000000_i64);
        let a5 = black_box(0x680000000_i64);
        let b5 = black_box(0x1c0000000_i64);
        let a6 = black_box(-0x740000000_i64);
        let b6 = black_box(0x80000000_i64);
        let a7 = black_box(0x200000000_i64);
        let b7 = black_box(-0x880000000_i64);
        let e = black_box(-0xd00000000_i64);
        assert!(
            sum_prod8_typed_postcheck(
                [a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7],
            ) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__alt_native_trunc() {
        let a0 = black_box(0x180000000_i64);
        let b0 = black_box(-0x480000000_i64);
        let a1 = black_box(-0x240000000_i64);
        let b1 = black_box(0x40000000_i64);
        let a2 = black_box(0x3c0000000_i64);
        let b2 = black_box(0x200000000_i64);
        let a3 = black_box(-0x80000000_i64);
        let b3 = black_box(0x720000000_i64);
        let a4 = black_box(0x20000000_i64);
        let b4 = black_box(-0x300000000_i64);
        let a5 = black_box(0x680000000_i64);
        let b5 = black_box(0x1c0000000_i64);
        let a6 = black_box(-0x740000000_i64);
        let b6 = black_box(0x80000000_i64);
        let a7 = black_box(0x200000000_i64);
        let b7 = black_box(-0x880000000_i64);
        let e = black_box(-0xd00000000_i64);
        assert!(
            sum_prod8_native_trunc(
                [a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7],
            ) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_sum_prod8__alt_chunked() {
        let a0 = black_box(0x180000000_i64);
        let b0 = black_box(-0x480000000_i64);
        let a1 = black_box(-0x240000000_i64);
        let b1 = black_box(0x40000000_i64);
        let a2 = black_box(0x3c0000000_i64);
        let b2 = black_box(0x200000000_i64);
        let a3 = black_box(-0x80000000_i64);
        let b3 = black_box(0x720000000_i64);
        let a4 = black_box(0x20000000_i64);
        let b4 = black_box(-0x300000000_i64);
        let a5 = black_box(0x680000000_i64);
        let b5 = black_box(0x1c0000000_i64);
        let a6 = black_box(-0x740000000_i64);
        let b6 = black_box(0x80000000_i64);
        let a7 = black_box(0x200000000_i64);
        let b7 = black_box(-0x880000000_i64);
        let e = black_box(-0xd00000000_i64);
        assert!(
            sum_prod8_chunked(
                [a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7],
            ) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_abs__alt_trim_first_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(0x380000000_i64);
        assert!(abs_trim_first(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__alt_trim_first_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(0x240000000_i64);
        assert!(abs_trim_first(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__alt_is_ok_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(false);
        assert!(is_negative_is_ok(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__alt_is_ok_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(true);
        assert!(is_negative_is_ok(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(false);
        assert!(is_negative_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(true);
        assert!(is_negative_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_positive__alt_native_p() {
        let a = black_box(0x380000000_i64);
        let e = black_box(true);
        assert!(is_positive_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_positive__alt_native_n() {
        let a = black_box(-0x240000000_i64);
        let e = black_box(false);
        assert!(is_positive_native(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt__alt_constrain_p() {
        let a = black_box(0x200000000_i64);
        let e = black_box(6074000999_i64);
        assert!(sqrt_constrain(a) == e);
    }


    #[test]
    #[inline(never)]
    fn bench_inv_sqrt__alt_constrain_p() {
        let a = black_box(0x200000000_i64);
        let e = black_box(3037000499_i64);
        assert!(inv_sqrt_constrain(a) == e);
    }
}
