//! Raw Q32.32 kernels on `i64`, powered by `core::internal::bounded_int`.
//!
//! This is the ONLY module of the workspace allowed to use the unstable `bounded-int-utils`
//! feature (DESIGN D2). Everything here takes and returns raw `i64` values (`value * 2^32`); the
//! typed API lives in the sibling modules (`ops`, `convert`, `math`, `fused`, `wide`).
//!
//! Principles (measured in `benchmarks/scalar` and in `kernels::alternatives`):
//! - products are computed *unscaled* (scale 2^64) in typed `BoundedInt` intervals: one felt
//!   multiplication, no range check, and the type proves that the intermediate cannot wrap;
//! - one `rescale` per output: `floor(x / 2^32)` as a biased unsigned `div_rem` by the constant
//!   2^32. The bias check (`x + 2^95` must fit 96 bits) is the only overflow check of a kernel;
//! - floor rounding is branch-free and sign-agnostic; sign splits (`constrain`) are only used
//!   where the operation is sign-dependent (division by a variable, truncation, abs, signum);
//! - accumulators are typed `BoundedInt`s up to 4 products (no wrap, proven by the compiler) and
//!   a `felt252` beyond (`simba::fixed::wide::Wide`, no wrap by construction): both cost the same.
//!
//! If `bounded-int-utils` ever becomes unavailable, this file is the only one to rewrite (on
//! corelib `i64_wide_mul` + unsigned `DivRem`), without touching the public API or stored data.

use core::internal::OptionRev;
#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, ConstrainHelper, DivRemHelper, MulHelper, SubHelper, TrimMinHelper,
    UnitInt, downcast, upcast,
};
use core::num::traits::Sqrt;
use crate::errors;

#[cfg(test)]
mod alternatives;
pub mod poly;
#[cfg(test)]
pub mod poly_alternatives;
pub mod transcendental;

// `core::zeroable::IsZeroResult` is crate-private in the corelib: the libfunc is re-declared
// with a local result enum.
enum IsZero<T> {
    Zero,
    NonZero: NonZero<T>,
}
extern fn bounded_int_is_zero<T>(value: T) -> IsZero<T> implicits() nopanic;

// ---------------------------------------------------------------------------------------------
// Interval types. The Sierra `bounded_int_*` libfuncs require the EXACT result interval of each
// operation (`[min, max]` of the operation over the operand intervals; for `div_rem`:
// `DivT = [lhs.min / rhs.max, lhs.max / max(rhs.min, 1)]`, `RemT = [0, rhs.max - 1]`). Every
// bound below was computed with Python integers, never by hand; a wrong bound is a compile error.
// ---------------------------------------------------------------------------------------------

const TWO31: felt252 = 0x80000000;
const TWO32: felt252 = 0x100000000;
const NEG_TWO32: felt252 = -0x100000000;
const TWO63: felt252 = 0x8000000000000000;
const TWO64: felt252 = 0x10000000000000000;
const TWO64_M1: felt252 = 0xffffffffffffffff;
const TWO95: felt252 = 0x800000000000000000000000;
const TWO96: felt252 = 0x1000000000000000000000000;
/// 2^63 + 2^32 - 1: bias of `ceil`.
const CEIL_BIAS: felt252 = 0x80000000ffffffff;

type U32Like = BoundedInt<0, 0xffffffff>;
type U64Like = BoundedInt<0, 0xffffffffffffffff>;
type U96Like = BoundedInt<0, 0xffffffffffffffffffffffff>;
type U127Like = BoundedInt<0, 0x7fffffffffffffffffffffffffffffff>;
type I32Like = BoundedInt<-0x80000000, 0x7fffffff>;
type I64Like = BoundedInt<-0x8000000000000000, 0x7fffffffffffffff>;
/// `i64` without `MIN`: the values that can be negated.
type I64Sym = BoundedInt<-0x7fffffffffffffff, 0x7fffffffffffffff>;
type I64Neg = BoundedInt<-0x8000000000000000, -1>;
type I64NonNeg = BoundedInt<0, 0x7fffffffffffffff>;
type I64Pos = BoundedInt<1, 0x7fffffffffffffff>;
/// `-x` for `x: I64Neg`.
type I64NegNegated = BoundedInt<1, 0x8000000000000000>;
/// `-x` for `x: i64`.
type I64Negated = BoundedInt<-0x7fffffffffffffff, 0x8000000000000000>;
type I64SymNeg = BoundedInt<-0x7fffffffffffffff, -1>;

/// `i64 * i64`.
type Prod = BoundedInt<-0x3fffffffffffffff8000000000000000, 0x40000000000000000000000000000000>;
type Sum2 = BoundedInt<-0x7fffffffffffffff0000000000000000, 0x80000000000000000000000000000000>;
type Sum3 = BoundedInt<-0xbffffffffffffffe8000000000000000, 0xc0000000000000000000000000000000>;
type Sum4 = BoundedInt<-0xfffffffffffffffe0000000000000000, 0x100000000000000000000000000000000>;
/// `Prod - Prod`.
type Diff = BoundedInt<-0x7fffffffffffffff8000000000000000, 0x7fffffffffffffff8000000000000000>;
/// `i64 * 2^32`.
type Shl32 = BoundedInt<-0x800000000000000000000000, 0x7fffffffffffffff00000000>;
/// `i64 * -2^32`.
type NegShl32 = BoundedInt<-0x7fffffffffffffff00000000, 0x800000000000000000000000>;
type MulAdd = BoundedInt<-0x400000007fffffff8000000000000000, 0x400000007fffffffffffffff00000000>;
type MulSub = BoundedInt<-0x400000007fffffff7fffffff00000000, 0x40000000800000000000000000000000>;
/// `i64 + i64`.
type SumI64 = BoundedInt<-0x10000000000000000, 0xfffffffffffffffe>;
/// `i64 - i64`.
type DiffI64 = BoundedInt<-0xffffffffffffffff, 0xffffffffffffffff>;
type DiffI64Neg = BoundedInt<-0xffffffffffffffff, -1>;
/// `DiffI64 * i64`.
type LerpProd = BoundedInt<-0x7fffffffffffffff8000000000000000, 0x7fffffffffffffff8000000000000000>;
type Lerp = BoundedInt<-0x800000007fffffff8000000000000000, 0x800000007fffffff7fffffff00000000>;

impl MulI64I64 of MulHelper<i64, i64> {
    type Result = Prod;
}
impl AddProdProd of AddHelper<Prod, Prod> {
    type Result = Sum2;
}
impl AddSum2Prod of AddHelper<Sum2, Prod> {
    type Result = Sum3;
}
impl AddSum3Prod of AddHelper<Sum3, Prod> {
    type Result = Sum4;
}
impl SubProdProd of SubHelper<Prod, Prod> {
    type Result = Diff;
}
impl MulI64Two32 of MulHelper<i64, UnitInt<TWO32>> {
    type Result = Shl32;
}
impl MulI64NegTwo32 of MulHelper<i64, UnitInt<NEG_TWO32>> {
    type Result = NegShl32;
}
impl AddProdShl32 of AddHelper<Prod, Shl32> {
    type Result = MulAdd;
}
impl SubProdShl32 of SubHelper<Prod, Shl32> {
    type Result = MulSub;
}
impl AddI64I64 of AddHelper<i64, i64> {
    type Result = SumI64;
}
impl SubI64I64 of SubHelper<i64, i64> {
    type Result = DiffI64;
}
impl MulDiffI64I64 of MulHelper<DiffI64, i64> {
    type Result = LerpProd;
}
impl AddLerpProdShl32 of AddHelper<LerpProd, Shl32> {
    type Result = Lerp;
}
impl MulI64MinusOne of MulHelper<i64, UnitInt<-1>> {
    type Result = I64Negated;
}
impl MulDiffI64NegMinusOne of MulHelper<DiffI64Neg, UnitInt<-1>> {
    type Result = BoundedInt<1, 0xffffffffffffffff>;
}
impl TrimMinI64NonNeg of TrimMinHelper<I64NonNeg> {
    type Target = I64Pos;
}
impl TrimMinI64Neg of TrimMinHelper<I64Neg> {
    type Target = I64SymNeg;
}
impl ConstrainI641 of ConstrainHelper<i64, 1> {
    type LowT = BoundedInt<-0x8000000000000000, 0>;
    type HighT = I64Pos;
}
impl ConstrainDiffI640 of ConstrainHelper<DiffI64, 0> {
    type LowT = DiffI64Neg;
    type HighT = U64Like;
}

// ---------------------------------------------------------------------------------------------
// Rescale: the single rounding + overflow check of every multiplicative kernel.
// ---------------------------------------------------------------------------------------------

impl DivU96Two32 of DivRemHelper<U96Like, UnitInt<TWO32>> {
    type DivT = U64Like;
    type RemT = U32Like;
}
impl SubU64Two63 of SubHelper<U64Like, UnitInt<TWO63>> {
    type Result = I64Like;
}

/// `floor(wide / 2^32)` for an unscaled value (scale 2^64) held in a `felt252`.
///
/// Precondition (guaranteed by every caller, by typing or by construction of `Wide`): `wide`
/// represents an integer of magnitude `< 2^250`, negatives as `P - |x|`. The result fits `i64`
/// iff `wide` is in `[-2^95, 2^95)`, i.e. iff `wide + 2^95` is in `[0, 2^96)`: that single range
/// check is the overflow check. Panics with `errors::OVERFLOW` otherwise.
#[inline(always)]
pub fn rescale(wide: felt252) -> i64 {
    let biased: U96Like = downcast(wide + TWO95).expect(errors::OVERFLOW);
    let (q, _r) = bounded_int::div_rem::<U96Like, UnitInt<TWO32>>(biased, 0x100000000);
    upcast(bounded_int::sub::<U64Like, UnitInt<TWO63>>(q, 0x8000000000000000))
}

/// Unscaled product `a * b` (scale 2^64) as a felt: `|a * b| <= 2^126`, no range check.
#[inline(always)]
pub fn wide_prod(a: i64, b: i64) -> felt252 {
    upcast(bounded_int::mul::<i64, i64>(a, b))
}

/// `c * 2^32`: a Q32.32 value aligned with unscaled products, `|c * 2^32| <= 2^95`.
#[inline(always)]
pub fn wide_from(c: i64) -> felt252 {
    upcast(bounded_int::mul::<i64, UnitInt<TWO32>>(c, 0x100000000))
}

const TWO127: felt252 = 0x80000000000000000000000000000000;

impl DivU128Two64 of DivRemHelper<u128, UnitInt<TWO64>> {
    type DivT = U64Like;
    type RemT = U64Like;
}

/// `floor(wide * s / 2^64)`: an unscaled accumulator (scale 2^64) times a Q32.32 raw (scale
/// 2^32), rescaled ONCE to a Q32.32 raw. The terminal kernel of `Wide::mul_scalar`, for exact
/// triple products `(a * b - c * d) * e` with a single rounding.
///
/// One felt multiplication, then the `rescale` idiom one level wider: the result fits `i64` iff
/// `wide * s` is in `[-2^127, 2^127)`, i.e. iff `wide * s + 2^127` is in `[0, 2^128)`; that single
/// range check (the downcast to `u128`) is the overflow check, followed by one biased unsigned
/// `div_rem` by the constant 2^64 (floor, sign-agnostic, no branch). Panics with
/// `errors::OVERFLOW` otherwise.
///
/// Precondition (guaranteed by `Wide`): `wide` represents an integer of magnitude `<= 2^187`
/// (`2^61` accumulated terms of `<= 2^126`), negatives as `P - |x|`. Then `|wide * s| <= 2^250`
/// is also exact modulo `P > 2^251`, and `wide * s + 2^127` can only land in `[0, 2^128)` when the
/// exact product is in `[-2^127, 2^127)` (a negative product aliases to `>= P - 2^250 + 2^127`,
/// a positive one out of range is `>= 2^128`). Measured against the candidates in
/// `kernels::alternatives` (`bench_wide_mul_scalar__alt_*`).
#[inline(always)]
pub fn wide_mul_rescale(wide: felt252, s: i64) -> i64 {
    let biased: u128 = (wide * s.into() + TWO127).try_into().expect(errors::OVERFLOW);
    let (q, _r) = bounded_int::div_rem::<u128, UnitInt<TWO64>>(biased, 0x10000000000000000);
    upcast(bounded_int::sub::<U64Like, UnitInt<TWO63>>(q, 0x8000000000000000))
}

// ---------------------------------------------------------------------------------------------
// Additive kernels.
// ---------------------------------------------------------------------------------------------

/// `a + b`, panics with `errors::OVERFLOW`.
#[inline(always)]
pub fn add(a: i64, b: i64) -> i64 {
    let s: SumI64 = bounded_int::add::<i64, i64>(a, b);
    downcast(s).expect(errors::OVERFLOW)
}

/// `a - b`, panics with `errors::OVERFLOW`.
#[inline(always)]
pub fn sub(a: i64, b: i64) -> i64 {
    let s: DiffI64 = bounded_int::sub::<i64, i64>(a, b);
    downcast(s).expect(errors::OVERFLOW)
}

/// `-a`, panics with `errors::OVERFLOW` for `MIN`. No range check: `MIN` is trimmed by equality.
#[inline(always)]
pub fn neg(a: i64) -> i64 {
    match bounded_int::trim_min::<i64>(a) {
        OptionRev::None => core::panic_with_felt252(errors::OVERFLOW),
        OptionRev::Some(a) => upcast(bounded_int::mul::<I64Sym, UnitInt<-1>>(a, -1)),
    }
}

/// `|a|`, panics with `errors::OVERFLOW` for `MIN`. One range check (the sign split); `MIN` is
/// trimmed by equality on the negative side only.
#[inline(always)]
pub fn abs(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(lt0) => match bounded_int::trim_min::<I64Neg>(lt0) {
            OptionRev::None => core::panic_with_felt252(errors::OVERFLOW),
            OptionRev::Some(lt0) => upcast(bounded_int::mul::<I64SymNeg, UnitInt<-1>>(lt0, -1)),
        },
        Err(ge0) => upcast(ge0),
    }
}

/// `-2^32`, `0` or `2^32` according to the sign of `a`: one range check and a zero test.
#[inline(always)]
pub fn signum(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(_) => -0x100000000,
        Err(ge0) => match bounded_int_is_zero(ge0) {
            IsZero::Zero => 0,
            IsZero::NonZero(_) => 0x100000000,
        },
    }
}

/// `a < 0`. An explicit `match` on the sign split: 200 gas cheaper than the native comparison,
/// and 500 cheaper than `.is_ok()` (an out-of-line call on a snapshot), hence the `allow`
/// (measured: `bench_is_negative__*`).
#[allow(manual_is_ok)]
#[inline(always)]
pub fn is_negative(a: i64) -> bool {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// `a > 0`, see `is_negative`.
#[allow(manual_is_err)]
#[inline(always)]
pub fn is_positive(a: i64) -> bool {
    match bounded_int::constrain::<i64, 1>(a) {
        Ok(_) => false,
        Err(_) => true,
    }
}

/// `|a - b| <= ulps` on the exact (non-overflowing) difference.
#[inline(always)]
pub fn abs_diff_le(a: i64, b: i64, ulps: u64) -> bool {
    let d: DiffI64 = bounded_int::sub::<i64, i64>(a, b);
    let mag: u64 = match bounded_int::constrain::<DiffI64, 0>(d) {
        Ok(lt0) => upcast(bounded_int::mul::<DiffI64Neg, UnitInt<-1>>(lt0, -1)),
        Err(ge0) => upcast(ge0),
    };
    mag <= ulps
}

// ---------------------------------------------------------------------------------------------
// Multiplicative and fused kernels: exact unscaled sum, one floor, one overflow check.
// ---------------------------------------------------------------------------------------------

/// `floor(a * b / 2^32)`.
#[inline(always)]
pub fn mul(a: i64, b: i64) -> i64 {
    rescale(wide_prod(a, b))
}

/// `floor((a0 * b0 + a1 * b1) / 2^32)`.
#[inline(always)]
pub fn sum_prod2(a0: i64, b0: i64, a1: i64, b1: i64) -> i64 {
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(a0, b0), bounded_int::mul(a1, b1));
    rescale(upcast(s))
}

/// `floor((a0 * b0 + a1 * b1 + a2 * b2) / 2^32)`.
#[inline(always)]
pub fn sum_prod3(a0: i64, b0: i64, a1: i64, b1: i64, a2: i64, b2: i64) -> i64 {
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(a0, b0), bounded_int::mul(a1, b1));
    let s: Sum3 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a2, b2));
    rescale(upcast(s))
}

/// `floor((a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3) / 2^32)`.
#[inline(always)]
pub fn sum_prod4(a0: i64, b0: i64, a1: i64, b1: i64, a2: i64, b2: i64, a3: i64, b3: i64) -> i64 {
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(a0, b0), bounded_int::mul(a1, b1));
    let s: Sum3 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a2, b2));
    let s: Sum4 = bounded_int::add(s, bounded_int::mul::<i64, i64>(a3, b3));
    rescale(upcast(s))
}

/// `floor((a * b - c * d) / 2^32)`.
#[inline(always)]
pub fn diff_prod(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let s: Diff = bounded_int::sub(bounded_int::mul::<i64, i64>(a, b), bounded_int::mul(c, d));
    rescale(upcast(s))
}

/// `floor((a * b + c * 2^32) / 2^32)`.
#[inline(always)]
pub fn mul_add(a: i64, b: i64, c: i64) -> i64 {
    let c: Shl32 = bounded_int::mul::<i64, UnitInt<TWO32>>(c, 0x100000000);
    let s: MulAdd = bounded_int::add(bounded_int::mul::<i64, i64>(a, b), c);
    rescale(upcast(s))
}

/// `floor((a * b - c * 2^32) / 2^32)`.
#[inline(always)]
pub fn mul_sub(a: i64, b: i64, c: i64) -> i64 {
    let c: Shl32 = bounded_int::mul::<i64, UnitInt<TWO32>>(c, 0x100000000);
    let s: MulSub = bounded_int::sub(bounded_int::mul::<i64, i64>(a, b), c);
    rescale(upcast(s))
}

/// `floor(((b - a) * t + a * 2^32) / 2^32)`: the difference and the product are exact.
#[inline(always)]
pub fn lerp(a: i64, b: i64, t: i64) -> i64 {
    let d: DiffI64 = bounded_int::sub::<i64, i64>(b, a);
    let p: LerpProd = bounded_int::mul::<DiffI64, i64>(d, t);
    let a: Shl32 = bounded_int::mul::<i64, UnitInt<TWO32>>(a, 0x100000000);
    let s: Lerp = bounded_int::add(p, a);
    rescale(upcast(s))
}

// ---------------------------------------------------------------------------------------------
// Division family. Floor division by a variable: bias the numerator by `2^63 * |b|` so that it is
// non-negative exactly when the quotient is `>= -2^63`, divide unsigned, remove the bias. One
// sign split (the divisor), no inspection of the remainder.
// ---------------------------------------------------------------------------------------------

type PosShl63 = BoundedInt<0, 0x3fffffffffffffff8000000000000000>;
type NegShl63 = BoundedInt<0x8000000000000000, 0x40000000000000000000000000000000>;
type DivNumPos = BoundedInt<-0x800000000000000000000000, 0x400000007fffffff7fffffff00000000>;
type DivNumNeg = BoundedInt<-0x7fffffff7fffffff00000000, 0x40000000800000000000000000000000>;
type StrictPosShl63 = BoundedInt<0x8000000000000000, 0x3fffffffffffffff8000000000000000>;
type RemNumPos = BoundedInt<0, 0x3fffffffffffffffffffffffffffffff>;
type RemNumNeg = BoundedInt<1, 0x40000000000000008000000000000000>;
type RemQuotNeg = BoundedInt<0, 0x40000000000000008000000000000000>;
type RemPos = BoundedInt<0, 0x7ffffffffffffffe>;

impl MulNonNegTwo63 of MulHelper<I64NonNeg, UnitInt<TWO63>> {
    type Result = PosShl63;
}
impl MulNegNegatedTwo63 of MulHelper<I64NegNegated, UnitInt<TWO63>> {
    type Result = NegShl63;
}
impl MulPosTwo63 of MulHelper<I64Pos, UnitInt<TWO63>> {
    type Result = StrictPosShl63;
}
impl AddShl32PosShl63 of AddHelper<Shl32, PosShl63> {
    type Result = DivNumPos;
}
impl AddNegShl32NegShl63 of AddHelper<NegShl32, NegShl63> {
    type Result = DivNumNeg;
}
impl AddI64StrictPosShl63 of AddHelper<i64, StrictPosShl63> {
    type Result = RemNumPos;
}
impl AddNegatedNegShl63 of AddHelper<I64Negated, NegShl63> {
    type Result = RemNumNeg;
}
impl DivU127NonNeg of DivRemHelper<U127Like, I64NonNeg> {
    type DivT = U127Like;
    type RemT = RemPos;
}
impl DivU127NegNegated of DivRemHelper<U127Like, I64NegNegated> {
    type DivT = U127Like;
    type RemT = I64NonNeg;
}
impl DivRemNumPos of DivRemHelper<RemNumPos, I64NonNeg> {
    type DivT = RemNumPos;
    type RemT = RemPos;
}
impl DivRemNumNeg of DivRemHelper<RemNumNeg, I64NegNegated> {
    type DivT = RemQuotNeg;
    type RemT = I64NonNeg;
}

/// `floor(a * 2^32 / b)` in exact arithmetic (floor division, like `mul`).
/// Panics with `errors::DIVISION_BY_ZERO` / `errors::OVERFLOW`.
#[inline(always)]
pub fn div(a: i64, b: i64) -> i64 {
    let b: NonZero<i64> = b.try_into().expect(errors::DIVISION_BY_ZERO);
    let q: U127Like = match bounded_int::constrain::<NonZero<i64>, 0>(b) {
        Ok(b) => {
            // a / b = (-a) / |b|
            let b: NonZero<I64NegNegated> = bounded_int::mul::<
                NonZero<I64Neg>, NonZero<UnitInt<-1>>,
            >(b, -1);
            let n: DivNumNeg = bounded_int::add(
                bounded_int::mul::<i64, UnitInt<NEG_TWO32>>(a, -0x100000000),
                bounded_int::mul::<I64NegNegated, UnitInt<TWO63>>(b.into(), 0x8000000000000000),
            );
            let n: U127Like = downcast(n).expect(errors::OVERFLOW);
            let (q, _r) = bounded_int::div_rem::<U127Like, I64NegNegated>(n, b);
            q
        },
        Err(b) => {
            let n: DivNumPos = bounded_int::add(
                bounded_int::mul::<i64, UnitInt<TWO32>>(a, 0x100000000),
                bounded_int::mul::<I64NonNeg, UnitInt<TWO63>>(b.into(), 0x8000000000000000),
            );
            let n: U127Like = downcast(n).expect(errors::OVERFLOW);
            let (q, _r) = bounded_int::div_rem::<U127Like, I64NonNeg>(n, b);
            q
        },
    };
    let q: U64Like = downcast(q).expect(errors::OVERFLOW);
    upcast(bounded_int::sub::<U64Like, UnitInt<TWO63>>(q, 0x8000000000000000))
}

/// Floored modulo `a - b * floor(a / b)`: zero or the sign of the divisor, `|result| < |b|`.
/// Exact (no rounding), cannot overflow. Panics with `errors::DIVISION_BY_ZERO`.
#[inline(always)]
pub fn rem(a: i64, b: i64) -> i64 {
    let b: NonZero<i64> = b.try_into().expect(errors::DIVISION_BY_ZERO);
    match bounded_int::constrain::<NonZero<i64>, 0>(b) {
        Ok(b) => {
            // a mod b = -((-a) mod |b|)
            let b: NonZero<I64NegNegated> = bounded_int::mul::<
                NonZero<I64Neg>, NonZero<UnitInt<-1>>,
            >(b, -1);
            let n: RemNumNeg = bounded_int::add(
                bounded_int::mul::<i64, UnitInt<-1>>(a, -1),
                bounded_int::mul::<I64NegNegated, UnitInt<TWO63>>(b.into(), 0x8000000000000000),
            );
            let (_q, r) = bounded_int::div_rem::<RemNumNeg, I64NegNegated>(n, b);
            upcast(bounded_int::mul::<I64NonNeg, UnitInt<-1>>(r, -1))
        },
        Err(b) => {
            // The type of a non-negative `NonZero` still contains 0: trim it (no range check)
            // so that the biased numerator is provably non-negative.
            match bounded_int::trim_min::<I64NonNeg>(b.into()) {
                OptionRev::None => core::panic_with_felt252(errors::DIVISION_BY_ZERO),
                OptionRev::Some(b_pos) => {
                    let n: RemNumPos = bounded_int::add(
                        a, bounded_int::mul::<I64Pos, UnitInt<TWO63>>(b_pos, 0x8000000000000000),
                    );
                    let (_q, r) = bounded_int::div_rem::<RemNumPos, I64NonNeg>(n, b);
                    upcast(r)
                },
            }
        },
    }
}

type RecipPosQuot = BoundedInt<2, 0x10000000000000000>;
type RecipNegQuot = BoundedInt<1, 0xffffffffffffffff>;
type RecipNegNegated = BoundedInt<-0xffffffffffffffff, -1>;
type RecipNegResult = BoundedInt<-0x10000000000000000, -2>;
impl DivTwo64NonNeg of DivRemHelper<UnitInt<TWO64>, I64NonNeg> {
    type DivT = RecipPosQuot;
    type RemT = RemPos;
}
impl DivTwo64M1NegNegated of DivRemHelper<UnitInt<TWO64_M1>, I64NegNegated> {
    type DivT = RecipNegQuot;
    type RemT = I64NonNeg;
}
impl MulRecipNegQuotMinusOne of MulHelper<RecipNegQuot, UnitInt<-1>> {
    type Result = RecipNegNegated;
}
impl SubRecipNegOne of SubHelper<RecipNegNegated, UnitInt<1>> {
    type Result = RecipNegResult;
}

/// `floor(2^64 / a)`: the reciprocal, floored. Constant numerator: no bias, no pre-check.
/// For `a < 0`: `floor(-2^64 / |a|) = -floor((2^64 - 1) / |a|) - 1`.
/// Panics with `errors::DIVISION_BY_ZERO` / `errors::OVERFLOW` (`|a| < 2` raw units, except -2).
#[inline(always)]
pub fn recip(a: i64) -> i64 {
    let a: NonZero<i64> = a.try_into().expect(errors::DIVISION_BY_ZERO);
    match bounded_int::constrain::<NonZero<i64>, 0>(a) {
        Ok(a) => {
            let a: NonZero<I64NegNegated> = bounded_int::mul::<
                NonZero<I64Neg>, NonZero<UnitInt<-1>>,
            >(a, -1);
            let (q, _r) = bounded_int::div_rem::<
                UnitInt<TWO64_M1>, I64NegNegated,
            >(0xffffffffffffffff, a);
            let q: RecipNegNegated = bounded_int::mul::<RecipNegQuot, UnitInt<-1>>(q, -1);
            let q: RecipNegResult = bounded_int::sub(q, 1);
            downcast(q).expect(errors::OVERFLOW)
        },
        Err(a) => {
            let (q, _r) = bounded_int::div_rem::<UnitInt<TWO64>, I64NonNeg>(0x10000000000000000, a);
            downcast(q).expect(errors::OVERFLOW)
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Square roots: corelib `u128_sqrt` (the prover guesses, the program verifies): exact floor.
// ---------------------------------------------------------------------------------------------

type U64Shl32 = BoundedInt<0, 0xffffffffffffffff00000000>;
type InvSqrtQuot = BoundedInt<0x100000000, 0x1000000000000000000000000>;
impl MulU64Two32 of MulHelper<u64, UnitInt<TWO32>> {
    type Result = U64Shl32;
}
impl DivTwo96U64 of DivRemHelper<UnitInt<TWO96>, u64> {
    type DivT = InvSqrtQuot;
    type RemT = BoundedInt<0, 0xfffffffffffffffe>;
}

/// `floor(sqrt(wide))` of an unscaled (scale 2^64) non-negative value: the result has scale 2^32.
/// Same precondition as `rescale`. Panics with `errors::SQRT_OF_NEGATIVE` / `errors::OVERFLOW`.
#[inline(always)]
pub fn wide_sqrt(wide: felt252) -> i64 {
    // Not `expect`: the message is only computed on the failure path.
    let Some(s): Option<u128> = wide.try_into() else {
        core::panic_with_felt252(wide_sqrt_error(wide))
    };
    let r: u64 = s.sqrt();
    downcast(r).expect(errors::OVERFLOW)
}

/// Failure path of `wide_sqrt` only: `|wide| < 2^250`, so a negative value is a felt above 2^250.
fn wide_sqrt_error(wide: felt252) -> felt252 {
    let wide: u256 = wide.into();
    if wide.high >= 0x4000000000000000000000000000000 {
        errors::SQRT_OF_NEGATIVE
    } else {
        errors::OVERFLOW
    }
}

/// `floor(sqrt(a * 2^32))`, exactly floored. Panics with `errors::SQRT_OF_NEGATIVE`.
/// The sign check is the `i64 -> u64` range check; the widening shift is a free bounded mul.
#[inline(always)]
pub fn sqrt(a: i64) -> i64 {
    let a: u64 = downcast(a).expect(errors::SQRT_OF_NEGATIVE);
    let wide: u128 = upcast(bounded_int::mul::<u64, UnitInt<TWO32>>(a, 0x100000000));
    let r: u64 = wide.sqrt();
    // r < 2^48: the conversion cannot fail.
    downcast(r).expect(errors::OVERFLOW)
}

/// `floor(2^48 / sqrt(a)) = isqrt(floor(2^96 / a))`: the inverse square root, exactly floored
/// (`floor(sqrt(floor(x))) = floor(sqrt(x))`). Cannot overflow (`<= 2^48`).
/// Panics with `errors::SQRT_OF_NEGATIVE` / `errors::DIVISION_BY_ZERO`.
#[inline(always)]
pub fn inv_sqrt(a: i64) -> i64 {
    let a: u64 = downcast(a).expect(errors::SQRT_OF_NEGATIVE);
    let a: NonZero<u64> = a.try_into().expect(errors::DIVISION_BY_ZERO);
    let (q, _r) = bounded_int::div_rem::<UnitInt<TWO96>, u64>(0x1000000000000000000000000, a);
    let q: u128 = upcast(q);
    let r: u64 = q.sqrt();
    // r <= 2^48: the conversion cannot fail.
    downcast(r).expect(errors::OVERFLOW)
}

/// `floor(sqrt(x^2 + y^2))` on the unscaled sum of squares: no rescale, no intermediate overflow.
#[inline(always)]
pub fn norm2(x: i64, y: i64) -> i64 {
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(x, x), bounded_int::mul(y, y));
    wide_sqrt(upcast(s))
}

/// `floor(sqrt(x^2 + y^2 + z^2))`, see `norm2`.
#[inline(always)]
pub fn norm3(x: i64, y: i64, z: i64) -> i64 {
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(x, x), bounded_int::mul(y, y));
    let s: Sum3 = bounded_int::add(s, bounded_int::mul::<i64, i64>(z, z));
    wide_sqrt(upcast(s))
}

/// `floor(sqrt(x^2 + y^2 + z^2 + w^2))`, see `norm2`.
#[inline(always)]
pub fn norm4(x: i64, y: i64, z: i64, w: i64) -> i64 {
    let s: Sum2 = bounded_int::add(bounded_int::mul::<i64, i64>(x, x), bounded_int::mul(y, y));
    let s: Sum3 = bounded_int::add(s, bounded_int::mul::<i64, i64>(z, z));
    let s: Sum4 = bounded_int::add(s, bounded_int::mul::<i64, i64>(w, w));
    wide_sqrt(upcast(s))
}

// ---------------------------------------------------------------------------------------------
// Rounding to integers: biased unsigned `div_rem` by 2^32.
// ---------------------------------------------------------------------------------------------

type FloorShl = BoundedInt<0, 0xffffffff00000000>;
type FloorResult = BoundedInt<-0x8000000000000000, 0x7fffffff00000000>;
type CeilOff = BoundedInt<0xffffffff, 0x100000000fffffffe>;
type CeilQuot = BoundedInt<0, 0x100000000>;
type CeilShl = BoundedInt<0, 0x10000000000000000>;
type CeilResult = BoundedInt<-0x8000000000000000, 0x8000000000000000>;
type RoundPosOff = BoundedInt<0x80000000, 0x800000007fffffff>;
type RoundNegOff = BoundedInt<0x80000001, 0x8000000080000000>;
type RoundQuot = BoundedInt<0, 0x80000000>;
type RoundShl = BoundedInt<0, 0x8000000000000000>;
type RoundNegResult = BoundedInt<-0x8000000000000000, 0>;
type TruncPosQuot = BoundedInt<0, 0x7fffffff>;
type TruncPosResult = BoundedInt<0, 0x7fffffff00000000>;
type FractNegResult = BoundedInt<-0xffffffff, 0>;

impl AddI64Two63 of AddHelper<i64, UnitInt<TWO63>> {
    type Result = U64Like;
}
impl DivU64Two32 of DivRemHelper<U64Like, UnitInt<TWO32>> {
    type DivT = U32Like;
    type RemT = U32Like;
}
impl SubU32Two31 of SubHelper<U32Like, UnitInt<TWO31>> {
    type Result = I32Like;
}
impl MulU32Two32 of MulHelper<U32Like, UnitInt<TWO32>> {
    type Result = FloorShl;
}
impl SubFloorShlTwo63 of SubHelper<FloorShl, UnitInt<TWO63>> {
    type Result = FloorResult;
}
impl AddI64CeilBias of AddHelper<i64, UnitInt<CEIL_BIAS>> {
    type Result = CeilOff;
}
impl DivCeilOffTwo32 of DivRemHelper<CeilOff, UnitInt<TWO32>> {
    type DivT = CeilQuot;
    type RemT = U32Like;
}
impl MulCeilQuotTwo32 of MulHelper<CeilQuot, UnitInt<TWO32>> {
    type Result = CeilShl;
}
impl SubCeilShlTwo63 of SubHelper<CeilShl, UnitInt<TWO63>> {
    type Result = CeilResult;
}
impl AddNonNegTwo31 of AddHelper<I64NonNeg, UnitInt<TWO31>> {
    type Result = RoundPosOff;
}
impl AddNegNegatedTwo31 of AddHelper<I64NegNegated, UnitInt<TWO31>> {
    type Result = RoundNegOff;
}
impl DivRoundPosOffTwo32 of DivRemHelper<RoundPosOff, UnitInt<TWO32>> {
    type DivT = RoundQuot;
    type RemT = U32Like;
}
impl DivRoundNegOffTwo32 of DivRemHelper<RoundNegOff, UnitInt<TWO32>> {
    type DivT = RoundQuot;
    type RemT = U32Like;
}
impl MulRoundQuotTwo32 of MulHelper<RoundQuot, UnitInt<TWO32>> {
    type Result = RoundShl;
}
impl MulRoundQuotNegTwo32 of MulHelper<RoundQuot, UnitInt<NEG_TWO32>> {
    type Result = RoundNegResult;
}
impl DivNonNegTwo32 of DivRemHelper<I64NonNeg, UnitInt<TWO32>> {
    type DivT = TruncPosQuot;
    type RemT = U32Like;
}
impl DivNegNegatedTwo32 of DivRemHelper<I64NegNegated, UnitInt<TWO32>> {
    type DivT = RoundQuot;
    type RemT = U32Like;
}
impl MulU32MinusOne of MulHelper<U32Like, UnitInt<-1>> {
    type Result = FractNegResult;
}
impl MulTruncPosQuotTwo32 of MulHelper<TruncPosQuot, UnitInt<TWO32>> {
    type Result = TruncPosResult;
}

/// `floor(a / 2^32)` as an `i32`: branch-free, cannot fail.
#[inline(always)]
pub fn to_i32(a: i64) -> i32 {
    let off: U64Like = bounded_int::add::<i64, UnitInt<TWO63>>(a, 0x8000000000000000);
    let (q, _r) = bounded_int::div_rem::<U64Like, UnitInt<TWO32>>(off, 0x100000000);
    upcast(bounded_int::sub::<U32Like, UnitInt<TWO31>>(q, 0x80000000))
}

/// `v * 2^32`: exact, no range check.
#[inline(always)]
pub fn from_i32(v: i32) -> i64 {
    // i32 * 2^32 is exactly [-2^63, (2^31 - 1) * 2^32], inside i64.
    upcast(bounded_int::mul::<i32, UnitInt<TWO32>>(v, 0x100000000))
}

type I32Shl32 = BoundedInt<-0x8000000000000000, 0x7fffffff00000000>;
impl MulI32Two32 of MulHelper<i32, UnitInt<TWO32>> {
    type Result = I32Shl32;
}

/// Largest multiple of 2^32 `<= a`: branch-free, cannot fail.
#[inline(always)]
pub fn floor(a: i64) -> i64 {
    let off: U64Like = bounded_int::add::<i64, UnitInt<TWO63>>(a, 0x8000000000000000);
    let (q, _r) = bounded_int::div_rem::<U64Like, UnitInt<TWO32>>(off, 0x100000000);
    let shl: FloorShl = bounded_int::mul::<U32Like, UnitInt<TWO32>>(q, 0x100000000);
    upcast(bounded_int::sub::<FloorShl, UnitInt<TWO63>>(shl, 0x8000000000000000))
}

/// Smallest multiple of 2^32 `>= a`: branch-free. Panics with `errors::OVERFLOW` when
/// `a > 2^63 - 2^32`.
#[inline(always)]
pub fn ceil(a: i64) -> i64 {
    let off: CeilOff = bounded_int::add::<i64, UnitInt<CEIL_BIAS>>(a, 0x80000000ffffffff);
    let (q, _r) = bounded_int::div_rem::<CeilOff, UnitInt<TWO32>>(off, 0x100000000);
    let shl: CeilShl = bounded_int::mul::<CeilQuot, UnitInt<TWO32>>(q, 0x100000000);
    let r: CeilResult = bounded_int::sub::<CeilShl, UnitInt<TWO63>>(shl, 0x8000000000000000);
    downcast(r).expect(errors::OVERFLOW)
}

/// Nearest multiple of 2^32, ties away from zero. Panics with `errors::OVERFLOW` when
/// `a >= 2^63 - 2^31`.
#[inline(always)]
pub fn round(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(a) => {
            let m: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(a, -1);
            let off: RoundNegOff = bounded_int::add::<I64NegNegated, UnitInt<TWO31>>(m, 0x80000000);
            let (q, _r) = bounded_int::div_rem::<RoundNegOff, UnitInt<TWO32>>(off, 0x100000000);
            upcast(bounded_int::mul::<RoundQuot, UnitInt<NEG_TWO32>>(q, -0x100000000))
        },
        Err(a) => {
            let off: RoundPosOff = bounded_int::add::<I64NonNeg, UnitInt<TWO31>>(a, 0x80000000);
            let (q, _r) = bounded_int::div_rem::<RoundPosOff, UnitInt<TWO32>>(off, 0x100000000);
            let r: RoundShl = bounded_int::mul::<RoundQuot, UnitInt<TWO32>>(q, 0x100000000);
            downcast(r).expect(errors::OVERFLOW)
        },
    }
}

/// Multiple of 2^32 nearest to `a` toward zero. Cannot fail.
#[inline(always)]
pub fn trunc(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(a) => {
            let m: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(a, -1);
            let (q, _r) = bounded_int::div_rem::<I64NegNegated, UnitInt<TWO32>>(m, 0x100000000);
            upcast(bounded_int::mul::<RoundQuot, UnitInt<NEG_TWO32>>(q, -0x100000000))
        },
        Err(a) => {
            let (q, _r) = bounded_int::div_rem::<I64NonNeg, UnitInt<TWO32>>(a, 0x100000000);
            upcast(bounded_int::mul::<TruncPosQuot, UnitInt<TWO32>>(q, 0x100000000))
        },
    }
}

/// `a - trunc(a)`: the fractional part, with the sign of `a`. Cannot fail.
#[inline(always)]
pub fn fract(a: i64) -> i64 {
    match bounded_int::constrain::<i64, 0>(a) {
        Ok(a) => {
            let m: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(a, -1);
            let (_q, r) = bounded_int::div_rem::<I64NegNegated, UnitInt<TWO32>>(m, 0x100000000);
            upcast(bounded_int::mul::<U32Like, UnitInt<-1>>(r, -1))
        },
        Err(a) => {
            let (_q, r) = bounded_int::div_rem::<I64NonNeg, UnitInt<TWO32>>(a, 0x100000000);
            upcast(r)
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Integer conversions.
// ---------------------------------------------------------------------------------------------

/// `v * 2^32` for an `i8`: exact, no range check.
#[inline(always)]
pub fn from_i8(v: i8) -> i64 {
    from_i32(upcast(v))
}

/// `v * 2^32` for an `i16`: exact, no range check.
#[inline(always)]
pub fn from_i16(v: i16) -> i64 {
    from_i32(upcast(v))
}

/// `v * 2^32` for an `u8`: exact, no range check.
#[inline(always)]
pub fn from_u8(v: u8) -> i64 {
    from_i32(upcast(v))
}

/// `v * 2^32` for an `u16`: exact, no range check.
#[inline(always)]
pub fn from_u16(v: u16) -> i64 {
    from_i32(upcast(v))
}

/// `v * 2^32` for an `i64` (one range check: `v` must fit `i32`).
#[inline(always)]
pub fn try_from_i64(v: i64) -> Option<i64> {
    match downcast::<i64, i32>(v) {
        Some(v) => Some(from_i32(v)),
        None => None,
    }
}

/// `v * 2^32` for an `i128` (one range check: `v` must fit `i32`).
#[inline(always)]
pub fn try_from_i128(v: i128) -> Option<i64> {
    match downcast::<i128, i32>(v) {
        Some(v) => Some(from_i32(v)),
        None => None,
    }
}

/// `v * 2^32` for an `u32` (one range check: `v` must fit `i32`).
#[inline(always)]
pub fn try_from_u32(v: u32) -> Option<i64> {
    match downcast::<u32, i32>(v) {
        Some(v) => Some(from_i32(v)),
        None => None,
    }
}

/// `v * 2^32` for an `u64` (one range check: `v` must fit `i32`).
#[inline(always)]
pub fn try_from_u64(v: u64) -> Option<i64> {
    match downcast::<u64, i32>(v) {
        Some(v) => Some(from_i32(v)),
        None => None,
    }
}

/// `v * 2^32` for an `u128` (one range check: `v` must fit `i32`).
#[inline(always)]
pub fn try_from_u128(v: u128) -> Option<i64> {
    match downcast::<u128, i32>(v) {
        Some(v) => Some(from_i32(v)),
        None => None,
    }
}
