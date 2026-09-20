//! G. `BI64`: Q32.32 on native `i64` storage, arithmetic powered by `core::internal::bounded_int`.
//!
//! Same storage as `I64Q32` (`raw: i64`), but multiplications are done in the `BoundedInt` domain:
//!   p = a * b                      bounded_int_mul: one felt mul, NO range check
//!   p + 2^128                      bounded_int_add with a constant: value is now non-negative
//!   (q, _) = div_rem(_, 2^32)      unsigned div by a constant (3 range checks)
//!   q - 2^96                       undo the offset => floor(p / 2^32)
//!   downcast to i64                the only overflow check
//! Branch-free, sign-agnostic (same cost for ++, +-, --).
//!
//! Products stay *unscaled* (scale 2^64) in the `Wide` accumulator `BoundedInt<-2^128, 2^128>`:
//! up to 4 products can be summed for free and rescaled ONCE (dot products, cross products,
//! matrix rows, lerp, mul_add).
//!
//! Rounding: floor (toward -inf). Overflow: panics (final downcast), intermediates cannot wrap.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, ConstrainHelper, DivRemHelper, MulHelper, SubHelper, UnitInt,
    downcast, upcast,
};
use core::num::traits::Sqrt;

pub const ONE: i64 = 0x100000000;

#[derive(Copy, Drop, PartialEq, Debug)]
pub struct BI64 {
    pub raw: i64,
}

// ---------------------------------------------------------------------------------------------
// Types (exact interval arithmetic, as required by the Sierra bounded_int libfuncs).
// ---------------------------------------------------------------------------------------------

const TWO32: felt252 = 0x100000000;
const TWO128: felt252 = 0x100000000000000000000000000000000;
const TWO96: felt252 = 0x1000000000000000000000000;

pub type Prod = BoundedInt<-0x3fffffffffffffff8000000000000000, 0x40000000000000000000000000000000>;
pub type Sum2 = BoundedInt<-0x7fffffffffffffff0000000000000000, 0x80000000000000000000000000000000>;
pub type Sum3 = BoundedInt<-0xbffffffffffffffe8000000000000000, 0xc0000000000000000000000000000000>;
pub type Diff = BoundedInt<-0x7fffffffffffffff8000000000000000, 0x7fffffffffffffff8000000000000000>;
pub type Shl32 = BoundedInt<-0x800000000000000000000000, 0x7fffffffffffffff00000000>;
pub type MulAdd =
    BoundedInt<-0x400000007fffffff8000000000000000, 0x400000007fffffffffffffff00000000>;
pub type DiffI64 = BoundedInt<-0xffffffffffffffff, 0xffffffffffffffff>;
pub type LerpProd =
    BoundedInt<-0x7fffffffffffffff8000000000000000, 0x7fffffffffffffff8000000000000000>;
pub type Lerp = BoundedInt<-0x800000007fffffff8000000000000000, 0x800000007fffffff7fffffff00000000>;

/// Unscaled (2^64 scale) accumulator: holds any sum of up to 4 products.
pub type Wide =
    BoundedInt<-0x100000000000000000000000000000000, 0x100000000000000000000000000000000>;
type WideOff = BoundedInt<0, 0x200000000000000000000000000000000>;
type QuotOff = BoundedInt<0, 0x2000000000000000000000000>;
type Quot = BoundedInt<-0x1000000000000000000000000, 0x1000000000000000000000000>;

impl MulI64 of MulHelper<i64, i64> {
    type Result = Prod;
}
impl AddProdProd of AddHelper<Prod, Prod> {
    type Result = Sum2;
}
impl AddSum2Prod of AddHelper<Sum2, Prod> {
    type Result = Sum3;
}
impl SubProdProd of SubHelper<Prod, Prod> {
    type Result = Diff;
}
impl MulI64Two32 of MulHelper<i64, UnitInt<TWO32>> {
    type Result = Shl32;
}
impl AddProdShl32 of AddHelper<Prod, Shl32> {
    type Result = MulAdd;
}
impl SubI64I64 of SubHelper<i64, i64> {
    type Result = DiffI64;
}
impl MulDiffI64 of MulHelper<DiffI64, i64> {
    type Result = LerpProd;
}
impl AddLerp of AddHelper<LerpProd, Shl32> {
    type Result = Lerp;
}
impl AddWideOff of AddHelper<Wide, UnitInt<TWO128>> {
    type Result = WideOff;
}
impl DivWideOff of DivRemHelper<WideOff, UnitInt<TWO32>> {
    type DivT = QuotOff;
    type RemT = BoundedInt<0, 0xffffffff>;
}
impl SubQuotOff of SubHelper<QuotOff, UnitInt<TWO96>> {
    type Result = Quot;
}

/// floor(w / 2^32), checked to fit in `i64`.
#[inline(always)]
pub fn rescale(w: Wide) -> i64 {
    let off: WideOff = bounded_int::add::<
        Wide, UnitInt<TWO128>,
    >(w, 0x100000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem::<WideOff, UnitInt<TWO32>>(off, 0x100000000);
    let q: Quot = bounded_int::sub::<QuotOff, UnitInt<TWO96>>(q, 0x1000000000000000000000000);
    downcast(q).expect('BI64 overflow')
}

/// Unscaled product (scale 2^64), free of range checks.
#[inline(always)]
pub fn wmul(a: BI64, b: BI64) -> Prod {
    bounded_int::mul::<i64, i64>(a.raw, b.raw)
}

pub impl BI64Mul of Mul<BI64> {
    #[inline(always)]
    fn mul(lhs: BI64, rhs: BI64) -> BI64 {
        BI64 { raw: rescale(upcast(wmul(lhs, rhs))) }
    }
}

pub impl BI64Add of Add<BI64> {
    #[inline(always)]
    fn add(lhs: BI64, rhs: BI64) -> BI64 {
        BI64 { raw: lhs.raw + rhs.raw }
    }
}

pub impl BI64Sub of Sub<BI64> {
    #[inline(always)]
    fn sub(lhs: BI64, rhs: BI64) -> BI64 {
        BI64 { raw: lhs.raw - rhs.raw }
    }
}

pub impl BI64Neg of Neg<BI64> {
    #[inline(always)]
    fn neg(a: BI64) -> BI64 {
        BI64 { raw: -a.raw }
    }
}

pub impl BI64PartialOrd of PartialOrd<BI64> {
    #[inline(always)]
    fn lt(lhs: BI64, rhs: BI64) -> bool {
        lhs.raw < rhs.raw
    }
}

// --- add/sub through BoundedInt (alternative to the native checked i64 ops) -------------------

type SumI64 = BoundedInt<-0x10000000000000000, 0xfffffffffffffffe>;
impl AddI64I64 of AddHelper<i64, i64> {
    type Result = SumI64;
}

#[inline(always)]
pub fn add_bounded(a: BI64, b: BI64) -> BI64 {
    let s: SumI64 = bounded_int::add::<i64, i64>(a.raw, b.raw);
    BI64 { raw: downcast(s).expect('BI64 add overflow') }
}

#[inline(always)]
pub fn sub_bounded(a: BI64, b: BI64) -> BI64 {
    let s: DiffI64 = bounded_int::sub::<i64, i64>(a.raw, b.raw);
    BI64 { raw: downcast(s).expect('BI64 sub overflow') }
}

// --- neg / abs ---------------------------------------------------------------------------------

type I64Neg = BoundedInt<-0x8000000000000000, -1>;
type I64NonNeg = BoundedInt<0, 0x7fffffffffffffff>;
type I64NegNegated = BoundedInt<1, 0x8000000000000000>;
type I64Negated = BoundedInt<-0x7fffffffffffffff, 0x8000000000000000>;

impl MulI64Minus1 of MulHelper<i64, UnitInt<-1>> {
    type Result = I64Negated;
}

#[inline(always)]
pub fn neg_bounded(a: BI64) -> BI64 {
    let n: I64Negated = bounded_int::mul::<i64, UnitInt<-1>>(a.raw, -1);
    BI64 { raw: downcast(n).expect('BI64 neg overflow') }
}

#[inline(always)]
pub fn abs(a: BI64) -> BI64 {
    if a.raw < 0 {
        BI64 { raw: -a.raw }
    } else {
        a
    }
}

#[inline(always)]
pub fn abs_bounded(a: BI64) -> BI64 {
    match bounded_int::constrain::<i64, 0>(a.raw) {
        Ok(lt0) => {
            let n: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(lt0, -1);
            BI64 { raw: downcast(n).expect('BI64 abs overflow') }
        },
        Err(_) => a,
    }
}

// --- fused forms: one rescale for several products ---------------------------------------------

#[inline(always)]
pub fn dot3(ax: BI64, ay: BI64, az: BI64, bx: BI64, by: BI64, bz: BI64) -> BI64 {
    let s: Sum2 = bounded_int::add::<Prod, Prod>(wmul(ax, bx), wmul(ay, by));
    let s: Sum3 = bounded_int::add::<Sum2, Prod>(s, wmul(az, bz));
    BI64 { raw: rescale(upcast(s)) }
}

#[inline(always)]
fn diff_of_products(a: BI64, b: BI64, c: BI64, d: BI64) -> BI64 {
    let s: Diff = bounded_int::sub::<Prod, Prod>(wmul(a, b), wmul(c, d));
    BI64 { raw: rescale(upcast(s)) }
}

#[inline(always)]
pub fn cross3(ax: BI64, ay: BI64, az: BI64, bx: BI64, by: BI64, bz: BI64) -> (BI64, BI64, BI64) {
    (
        diff_of_products(ay, bz, az, by),
        diff_of_products(az, bx, ax, bz),
        diff_of_products(ax, by, ay, bx),
    )
}

/// a * b + c with a single rescale.
#[inline(always)]
pub fn mul_add(a: BI64, b: BI64, c: BI64) -> BI64 {
    let c: Shl32 = bounded_int::mul::<i64, UnitInt<TWO32>>(c.raw, 0x100000000);
    let s: MulAdd = bounded_int::add::<Prod, Shl32>(wmul(a, b), c);
    BI64 { raw: rescale(upcast(s)) }
}

/// a + (b - a) * t with a single rescale and no intermediate overflow check.
#[inline(always)]
pub fn lerp(a: BI64, b: BI64, t: BI64) -> BI64 {
    let d: DiffI64 = bounded_int::sub::<i64, i64>(b.raw, a.raw);
    let p: LerpProd = bounded_int::mul::<DiffI64, i64>(d, t.raw);
    let a: Shl32 = bounded_int::mul::<i64, UnitInt<TWO32>>(a.raw, 0x100000000);
    let s: Lerp = bounded_int::add::<LerpProd, Shl32>(p, a);
    BI64 { raw: rescale(upcast(s)) }
}

// --- int conversions ----------------------------------------------------------------------------

type I32Shl32 = BoundedInt<-0x8000000000000000, 0x7fffffff00000000>;
impl MulI32Two32 of MulHelper<i32, UnitInt<TWO32>> {
    type Result = I32Shl32;
}

/// Exact, no range check at all.
#[inline(always)]
pub fn from_int(v: i32) -> BI64 {
    let r: I32Shl32 = bounded_int::mul::<i32, UnitInt<TWO32>>(v, 0x100000000);
    BI64 { raw: upcast(r) }
}

const TWO63: felt252 = 0x8000000000000000;
const TWO31: felt252 = 0x80000000;
type U64Like = BoundedInt<0, 0xffffffffffffffff>;
type U32Like = BoundedInt<0, 0xffffffff>;
type I32Like = BoundedInt<-0x80000000, 0x7fffffff>;
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

/// Floor to integer: no branch, cannot fail.
#[inline(always)]
pub fn to_int(a: BI64) -> i32 {
    let off: U64Like = bounded_int::add::<i64, UnitInt<TWO63>>(a.raw, 0x8000000000000000);
    let (q, _r) = bounded_int::div_rem::<U64Like, UnitInt<TWO32>>(off, 0x100000000);
    upcast(bounded_int::sub::<U32Like, UnitInt<TWO31>>(q, 0x80000000))
}

// --- division -----------------------------------------------------------------------------------
//
// The offset trick does not work with a variable divisor (the quotient type would span more than
// 2^128 and cannot be downcast), so signs are peeled off with `constrain` like corelib's signed
// DivRem does, with a floor adjustment.

type NumAbs = BoundedInt<0, 0x800000000000000000000000>; // |a| * 2^32 <= 2^95
type RemNonNeg = BoundedInt<0, 0x7ffffffffffffffe>;
type NegQuot = BoundedInt<-0x7fffffffffffffff, 0>;

impl MulNegNegatedTwo32 of MulHelper<I64NegNegated, UnitInt<TWO32>> {
    type Result = BoundedInt<0x100000000, 0x800000000000000000000000>;
}
impl DivNumAbsNonNeg of DivRemHelper<NumAbs, I64NonNeg> {
    type DivT = NumAbs;
    type RemT = RemNonNeg;
}
impl DivNumAbsNegNegated of DivRemHelper<NumAbs, I64NegNegated> {
    type DivT = NumAbs;
    type RemT = I64NonNeg;
}
impl SubNegQuotOne of SubHelper<NegQuot, UnitInt<1>> {
    type Result = I64Neg;
}

#[inline(always)]
fn div_finish(q: NumAbs, inexact: bool, neg: bool) -> i64 {
    let q: I64NonNeg = downcast(q).expect('BI64 div overflow');
    if neg {
        let m: NegQuot = bounded_int::mul::<I64NonNeg, UnitInt<-1>>(q, -1);
        if inexact {
            upcast(bounded_int::sub::<NegQuot, UnitInt<1>>(m, 1))
        } else {
            upcast(m)
        }
    } else {
        upcast(q)
    }
}

pub impl BI64Div of Div<BI64> {
    /// Floor division.
    #[inline(always)]
    fn div(lhs: BI64, rhs: BI64) -> BI64 {
        let d: NonZero<i64> = rhs.raw.try_into().expect('BI64 div by zero');
        let (n, n_neg): (NumAbs, bool) = match bounded_int::constrain::<i64, 0>(lhs.raw) {
            Ok(lt0) => {
                let m: I64NegNegated = bounded_int::mul::<I64Neg, UnitInt<-1>>(lt0, -1);
                (upcast(bounded_int::mul::<I64NegNegated, UnitInt<TWO32>>(m, 0x100000000)), true)
            },
            Err(ge0) => (
                upcast(bounded_int::mul::<I64NonNeg, UnitInt<TWO32>>(ge0, 0x100000000)), false,
            ),
        };
        let raw = match bounded_int::constrain::<NonZero<i64>, 0>(d) {
            Ok(d_lt0) => {
                let d = bounded_int::mul::<NonZero<I64Neg>, NonZero<UnitInt<-1>>>(d_lt0, -1);
                let (q, r) = bounded_int::div_rem::<NumAbs, I64NegNegated>(n, d);
                div_finish(q, upcast::<_, felt252>(r) != 0, !n_neg)
            },
            Err(d_ge0) => {
                let (q, r) = bounded_int::div_rem::<NumAbs, I64NonNeg>(n, d_ge0);
                div_finish(q, upcast::<_, felt252>(r) != 0, n_neg)
            },
        };
        BI64 { raw }
    }
}

// --- sqrt ---------------------------------------------------------------------------------------

type NonNegShl32 = BoundedInt<0, 0x7fffffffffffffff00000000>;
impl MulNonNegTwo32 of MulHelper<I64NonNeg, UnitInt<TWO32>> {
    type Result = NonNegShl32;
}

/// sqrt through the corelib `u128` integer square root; the widening shift is a free bounded mul.
#[inline(always)]
pub fn sqrt(a: BI64) -> BI64 {
    match bounded_int::constrain::<i64, 0>(a.raw) {
        Ok(_) => core::panic_with_felt252('sqrt of negative'),
        Err(ge0) => {
            let wide: NonNegShl32 = bounded_int::mul::<I64NonNeg, UnitInt<TWO32>>(ge0, 0x100000000);
            let wide: u128 = upcast(wide);
            let r: u64 = wide.sqrt();
            BI64 { raw: r.try_into().unwrap() }
        },
    }
}

/// |v| for a 3-vector: the unscaled sum of squares (scale 2^64) is exactly what the integer
/// sqrt wants: NO rescale at all, and no overflow for |coordinates| up to 2^31.
#[inline(always)]
pub fn length3(x: BI64, y: BI64, z: BI64) -> BI64 {
    let s: Sum2 = bounded_int::add::<Prod, Prod>(wmul(x, x), wmul(y, y));
    let s: Sum3 = bounded_int::add::<Sum2, Prod>(s, wmul(z, z));
    // Squares are non-negative but the type system does not know it (and `downcast` rejects
    // source ranges wider than 2^128): go through felt252 -> u128.
    let s: felt252 = upcast(s);
    let s: u128 = s.try_into().expect('BI64 length overflow');
    let r: u64 = s.sqrt();
    BI64 { raw: r.try_into().expect('BI64 length overflow') }
}

// --- alternative rescale strategies (measured in `mul_i64`) --------------------------------------

type ProdOff95 =
    BoundedInt<-0x3fffffff7fffffff8000000000000000, 0x40000000800000000000000000000000>;
type U96Like = BoundedInt<0, 0xffffffffffffffffffffffff>;
type I64Like = BoundedInt<-0x8000000000000000, 0x7fffffffffffffff>;
impl AddProdTwo95 of AddHelper<Prod, UnitInt<0x800000000000000000000000>> {
    type Result = ProdOff95;
}
impl DivU96Two32 of DivRemHelper<U96Like, UnitInt<TWO32>> {
    type DivT = U64Like;
    type RemT = U32Like;
}
impl SubU64Two63 of SubHelper<U64Like, UnitInt<TWO63>> {
    type Result = I64Like;
}

/// Overflow check BEFORE the division: (p + 2^95) must be in [0, 2^96).
#[inline(always)]
pub fn mul_precheck(a: BI64, b: BI64) -> BI64 {
    let p: ProdOff95 = bounded_int::add::<
        Prod, UnitInt<0x800000000000000000000000>,
    >(wmul(a, b), 0x800000000000000000000000);
    let p: U96Like = downcast(p).expect('BI64 mul overflow');
    let (q, _r) = bounded_int::div_rem::<U96Like, UnitInt<TWO32>>(p, 0x100000000);
    BI64 { raw: upcast(bounded_int::sub::<U64Like, UnitInt<TWO63>>(q, 0x8000000000000000)) }
}

impl DivU128Two32 of DivRemHelper<u128, UnitInt<TWO32>> {
    type DivT = U96Like;
    type RemT = U32Like;
}

/// felt252 -> u128 conversion as the sign/bound recovery, one-sided downcast after the division.
#[inline(always)]
pub fn mul_via_u128(a: BI64, b: BI64) -> BI64 {
    let p: felt252 = upcast(wmul(a, b));
    let p: u128 = (p + 0x800000000000000000000000).try_into().expect('BI64 mul overflow');
    let (q, _r) = bounded_int::div_rem::<u128, UnitInt<TWO32>>(p, 0x100000000);
    let q: U64Like = downcast(q).expect('BI64 mul overflow');
    BI64 { raw: upcast(bounded_int::sub::<U64Like, UnitInt<TWO63>>(q, 0x8000000000000000)) }
}

// --- truncation toward zero (for comparison with the branch-free floor) -------------------------

type ProdNeg = BoundedInt<-0x3fffffffffffffff8000000000000000, -1>;
type ProdNonNeg = BoundedInt<0, 0x40000000000000000000000000000000>;
type ProdNegAbs = BoundedInt<1, 0x3fffffffffffffff8000000000000000>;
type QuotNegAbs = BoundedInt<0, 0x3fffffffffffffff80000000>;
type QuotNonNeg = BoundedInt<0, 0x400000000000000000000000>;
type QuotNegated = BoundedInt<-0x3fffffffffffffff80000000, 0>;
impl ConstrainProd0 of ConstrainHelper<Prod, 0> {
    type LowT = ProdNeg;
    type HighT = ProdNonNeg;
}
impl MulProdNegMinus1 of MulHelper<ProdNeg, UnitInt<-1>> {
    type Result = ProdNegAbs;
}
impl DivProdNegAbs of DivRemHelper<ProdNegAbs, UnitInt<TWO32>> {
    type DivT = QuotNegAbs;
    type RemT = U32Like;
}
impl DivProdNonNeg of DivRemHelper<ProdNonNeg, UnitInt<TWO32>> {
    type DivT = QuotNonNeg;
    type RemT = U32Like;
}
impl MulQuotNegAbsMinus1 of MulHelper<QuotNegAbs, UnitInt<-1>> {
    type Result = QuotNegated;
}

/// Truncating multiply: one sign split on the product, then an unsigned div_rem.
#[inline(always)]
pub fn mul_trunc(a: BI64, b: BI64) -> BI64 {
    let q: Quot = match bounded_int::constrain::<Prod, 0>(wmul(a, b)) {
        Ok(neg) => {
            let m: ProdNegAbs = bounded_int::mul::<ProdNeg, UnitInt<-1>>(neg, -1);
            let (q, _r) = bounded_int::div_rem::<ProdNegAbs, UnitInt<TWO32>>(m, 0x100000000);
            upcast(bounded_int::mul::<QuotNegAbs, UnitInt<-1>>(q, -1))
        },
        Err(pos) => {
            let (q, _r) = bounded_int::div_rem::<ProdNonNeg, UnitInt<TWO32>>(pos, 0x100000000);
            upcast(q)
        },
    };
    BI64 { raw: downcast(q).expect('BI64 mul overflow') }
}
