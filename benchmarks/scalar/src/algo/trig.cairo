//! sin / cos / atan2 / acos alternatives for `BI64` (Q32.32). Angles in radians.
//!
//! Range reduction is done "by math": one bounded div_rem by pi/2 (after a positive offset that
//! is a multiple of 2*pi) gives the quadrant and the remainder in [0, pi/2) exactly.
//! Coefficients: Chebyshev (near-minimax) fits computed offline with mpmath (see gen_algos.py).

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, downcast, upcast,
};
use crate::algo::poly_typed::{self, Acos9Out, Atan15Out, Atan19Out, RFull, TUnit};
use crate::algo::{cordic, lut};
use crate::bi64::{self, BI64, mul_add};

pub const HALF_PI: i64 = 0x1921fb544;
pub const PI: i64 = 0x3243f6a89;

// ---------------------------------------------------------------------------------------------
// Range reduction
// ---------------------------------------------------------------------------------------------

const HALF_PI_F: felt252 = 0x1921fb544;
// OFFSET = HALF_PI * 1367130552 (a multiple of 4 quarter turns) >= 2^63.
const OFFSET: felt252 = 0x800000013f628ce0;
type XOff = BoundedInt<0x13f628ce0, 0x1000000013f628cdf>;
type Quarter = BoundedInt<0, 0xa2f9836f>;
type QuarterP1 = BoundedInt<1, 0xa2f98370>;
type Rem = BoundedInt<0, 0x1921fb543>;
type Quad = BoundedInt<0, 3>;
impl AddI64Offset of AddHelper<i64, UnitInt<OFFSET>> {
    type Result = XOff;
}
impl DivXOffHalfPi of DivRemHelper<XOff, UnitInt<HALF_PI_F>> {
    type DivT = Quarter;
    type RemT = Rem;
}
impl DivQuarter4 of DivRemHelper<Quarter, UnitInt<4>> {
    type DivT = BoundedInt<0, 0x28be60db>;
    type RemT = Quad;
}
impl AddQuarter1 of AddHelper<Quarter, UnitInt<1>> {
    type Result = QuarterP1;
}
impl DivQuarterP14 of DivRemHelper<QuarterP1, UnitInt<4>> {
    type DivT = BoundedInt<0, 0x28be60dc>;
    type RemT = Quad;
}
impl SubHalfPiRem of SubHelper<UnitInt<HALF_PI_F>, Rem> {
    type Result = BoundedInt<1, 0x1921fb544>;
}

/// Returns (r, negate) such that sin(x) = (-1)^negate * sin(r), r in [0, pi/2].
#[inline(always)]
fn reduce_sin(x: BI64) -> (BI64, bool) {
    let off: XOff = bounded_int::add::<i64, UnitInt<OFFSET>>(x.raw, 0x800000013f628ce0);
    let (q, r) = bounded_int::div_rem::<XOff, UnitInt<HALF_PI_F>>(off, 0x1921fb544);
    let (_, n) = bounded_int::div_rem::<Quarter, UnitInt<4>>(q, 4);
    fold(n, r)
}

/// Same for cos(x) = sin(x + pi/2): the quarter-turn index is shifted by one.
#[inline(always)]
fn reduce_cos(x: BI64) -> (BI64, bool) {
    let off: XOff = bounded_int::add::<i64, UnitInt<OFFSET>>(x.raw, 0x800000013f628ce0);
    let (q, r) = bounded_int::div_rem::<XOff, UnitInt<HALF_PI_F>>(off, 0x1921fb544);
    let q: QuarterP1 = bounded_int::add::<Quarter, UnitInt<1>>(q, 1);
    let (_, n) = bounded_int::div_rem::<QuarterP1, UnitInt<4>>(q, 4);
    fold(n, r)
}

#[inline(always)]
fn fold_typed(n: Quad, r: Rem) -> (RFull, bool) {
    let n: felt252 = upcast(n);
    if n == 0 {
        (upcast(r), false)
    } else if n == 1 {
        (upcast(bounded_int::sub::<UnitInt<HALF_PI_F>, Rem>(0x1921fb544, r)), false)
    } else if n == 2 {
        (upcast(r), true)
    } else {
        (upcast(bounded_int::sub::<UnitInt<HALF_PI_F>, Rem>(0x1921fb544, r)), true)
    }
}

#[inline(always)]
fn fold(n: Quad, r: Rem) -> (BI64, bool) {
    let (r, neg) = fold_typed(n, r);
    (BI64 { raw: upcast(r) }, neg)
}

#[inline(always)]
fn reduce_sin_typed(x: BI64) -> (RFull, bool) {
    let off: XOff = bounded_int::add::<i64, UnitInt<OFFSET>>(x.raw, 0x800000013f628ce0);
    let (q, r) = bounded_int::div_rem::<XOff, UnitInt<HALF_PI_F>>(off, 0x1921fb544);
    let (_, n) = bounded_int::div_rem::<Quarter, UnitInt<4>>(q, 4);
    fold_typed(n, r)
}

#[inline(always)]
fn reduce_cos_typed(x: BI64) -> (RFull, bool) {
    let off: XOff = bounded_int::add::<i64, UnitInt<OFFSET>>(x.raw, 0x800000013f628ce0);
    let (q, r) = bounded_int::div_rem::<XOff, UnitInt<HALF_PI_F>>(off, 0x1921fb544);
    let q: QuarterP1 = bounded_int::add::<Quarter, UnitInt<1>>(q, 1);
    let (_, n) = bounded_int::div_rem::<QuarterP1, UnitInt<4>>(q, 4);
    fold_typed(n, r)
}

/// Typed Horner kernels (generated, see `poly_typed.cairo`): no overflow check, 2^-64 internal
/// precision. The result is in [-1, 1] by construction so the negation cannot overflow either.
pub fn sin_poly9_typed(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin_typed(x);
    apply_sign(BI64 { raw: upcast(poly_typed::sin_q9(r)) }, neg)
}

pub fn sin_poly11_typed(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin_typed(x);
    apply_sign(BI64 { raw: upcast(poly_typed::sin_q11(r)) }, neg)
}

pub fn cos_poly11_typed(x: BI64) -> BI64 {
    let (r, neg) = reduce_cos_typed(x);
    apply_sign(BI64 { raw: upcast(poly_typed::sin_q11(r)) }, neg)
}

#[inline(always)]
fn apply_sign(v: BI64, negate: bool) -> BI64 {
    if negate {
        BI64 { raw: -v.raw }
    } else {
        v
    }
}

// ---------------------------------------------------------------------------------------------
// sin: odd polynomial on [0, pi/2], Horner in z = r^2 with fused mul_add (one rescale per step)
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn c(raw: i64) -> BI64 {
    BI64 { raw }
}

/// Degree 7, max error ~1.2e-6.
#[inline(always)]
fn sin_q_poly7(r: BI64) -> BI64 {
    let z = r * r;
    let acc = mul_add(z, c(-795537), c(35704885));
    let acc = mul_add(z, acc, c(-715785355));
    let acc = mul_add(z, acc, c(4294964019));
    r * acc
}

/// Degree 9, max error ~7e-9.
#[inline(always)]
fn sin_q_poly9(r: BI64) -> BI64 {
    let z = r * r;
    let acc = mul_add(z, c(11189), c(-850791));
    let acc = mul_add(z, acc, c(35790178));
    let acc = mul_add(z, acc, c(-715827508));
    let acc = mul_add(z, acc, c(4294967278));
    r * acc
}

/// Degree 11, polynomial error 3e-11 (below the Q32.32 resolution).
#[inline(always)]
fn sin_q_poly11(r: BI64) -> BI64 {
    let z = r * r;
    let acc = mul_add(z, c(-103), c(11822));
    let acc = mul_add(z, acc, c(-852158));
    let acc = mul_add(z, acc, c(35791384));
    let acc = mul_add(z, acc, c(-715827881));
    let acc = mul_add(z, acc, c(4294967296));
    r * acc
}

/// Degree 9 with plain operators (a rescale AND a checked add per Horner step).
#[inline(always)]
fn sin_q_poly9_unfused(r: BI64) -> BI64 {
    let z = r * r;
    let acc = z * c(11189) + c(-850791);
    let acc = z * acc + c(35790178);
    let acc = z * acc + c(-715827508);
    let acc = z * acc + c(4294967278);
    r * acc
}

pub fn sin_poly7(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin(x);
    apply_sign(sin_q_poly7(r), neg)
}

pub fn sin_poly9(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin(x);
    apply_sign(sin_q_poly9(r), neg)
}

pub fn cos_poly9(x: BI64) -> BI64 {
    let (r, neg) = reduce_cos(x);
    apply_sign(sin_q_poly9(r), neg)
}

pub fn sin_poly11(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin(x);
    apply_sign(sin_q_poly11(r), neg)
}

pub fn sin_poly9_unfused(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin(x);
    apply_sign(sin_q_poly9_unfused(r), neg)
}

/// cubit-style Taylor: 1 - z/(2*3) * (1 - z/(4*5) * (...)) evaluated in a loop with integer
/// divisions, 7 terms, on the reduced angle.
pub fn sin_taylor_loop(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin(x);
    let z = r * r;
    let mut acc = BI64 { raw: bi64::ONE };
    let mut i: i64 = 7;
    while i != 0 {
        let d = (2 * i) * (2 * i + 1);
        let term = z * acc;
        acc = BI64 { raw: bi64::ONE - term.raw / d };
        i -= 1;
    }
    apply_sign(r * acc, neg)
}

// ---------------------------------------------------------------------------------------------
// sin: lookup table over the quarter turn + linear interpolation
// ---------------------------------------------------------------------------------------------

// v = floor(r * floor(2^64 / HALF_PI) / 2^32) in [0, 2^32): top bits = index, low bits = fraction.
const INV_HALF_PI: felt252 = 0xa2f9836e; // floor(2^64 / HALF_PI)
type RScaled = BoundedInt<0, 0xffffffff5a85af38>;
type U32Like = BoundedInt<0, 0xffffffff>;
impl MulRInv of MulHelper<RFull, UnitInt<INV_HALF_PI>> {
    type Result = RScaled;
}
impl DivRScaled of DivRemHelper<RScaled, UnitInt<0x100000000>> {
    type DivT = U32Like;
    type RemT = U32Like;
}
impl DivIdx64 of DivRemHelper<U32Like, UnitInt<0x4000000>> {
    type DivT = BoundedInt<0, 63>;
    type RemT = BoundedInt<0, 0x3ffffff>;
}
impl DivIdx256 of DivRemHelper<U32Like, UnitInt<0x1000000>> {
    type DivT = BoundedInt<0, 255>;
    type RemT = BoundedInt<0, 0xffffff>;
}

#[inline(always)]
fn turn_fraction(r: RFull) -> U32Like {
    let v: RScaled = bounded_int::mul::<RFull, UnitInt<INV_HALF_PI>>(r, 0xa2f9836e);
    let (v, _) = bounded_int::div_rem::<RScaled, UnitInt<0x100000000>>(v, 0x100000000);
    v
}

/// lo + (hi - lo) * frac / 2^bits on unsigned values (sin is increasing on the quarter turn).
#[inline(always)]
fn interpolate(lo: u64, hi: u64, frac: u64, scale: u64) -> BI64 {
    let v: u64 = lo + (hi - lo) * frac / scale;
    BI64 { raw: v.try_into().unwrap() }
}

/// 64 intervals, table as a `match`. Max error ~7.5e-5.
pub fn sin_lut64_match(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin_typed(x);
    let (idx, frac) = bounded_int::div_rem::<
        U32Like, UnitInt<0x4000000>,
    >(turn_fraction(r), 0x4000000);
    let idx: felt252 = upcast(idx);
    let frac: u64 = upcast(frac);
    apply_sign(interpolate(lut::sin64_match(idx), lut::sin64_match(idx + 1), frac, 0x4000000), neg)
}

/// 64 intervals, table as a const array. Max error ~7.5e-5.
pub fn sin_lut64_array(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin_typed(x);
    let (idx, frac) = bounded_int::div_rem::<
        U32Like, UnitInt<0x4000000>,
    >(turn_fraction(r), 0x4000000);
    let idx: u32 = upcast(idx);
    let frac: u64 = upcast(frac);
    let table = lut::SIN64.span();
    apply_sign(interpolate(*table[idx], *table[idx + 1], frac, 0x4000000), neg)
}

/// 256 intervals, table as a const array. Max error ~4.7e-6.
pub fn sin_lut256_array(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin_typed(x);
    let (idx, frac) = bounded_int::div_rem::<
        U32Like, UnitInt<0x1000000>,
    >(turn_fraction(r), 0x1000000);
    let idx: u32 = upcast(idx);
    let frac: u64 = upcast(frac);
    let table = lut::SIN256.span();
    apply_sign(interpolate(*table[idx], *table[idx + 1], frac, 0x1000000), neg)
}

/// CORDIC, 20 unrolled iterations (generated). Max error ~4e-6.
pub fn sin_cordic20(x: BI64) -> BI64 {
    let (r, neg) = reduce_sin(x);
    apply_sign(BI64 { raw: cordic::sin_q(r.raw) }, neg)
}

// ---------------------------------------------------------------------------------------------
// atan2
// ---------------------------------------------------------------------------------------------

/// atan on [0, 1], odd degree 15 (8 coefficients). Max error ~6.4e-8.
#[inline(always)]
fn atan_unit_poly15(t: BI64) -> BI64 {
    let z = t * t;
    let acc = mul_add(z, c(-19584157), c(102136550));
    let acc = mul_add(z, acc, c(-252671866));
    let acc = mul_add(z, acc, c(423864544));
    let acc = mul_add(z, acc, c(-601436734));
    let acc = mul_add(z, acc, c(857574481));
    let acc = mul_add(z, acc, c(-1431590453));
    let acc = mul_add(z, acc, c(4294966789));
    t * acc
}

/// atan on [0, tan(pi/8)], odd degree 9 (5 coefficients). Max error ~6.8e-9.
#[inline(always)]
fn atan_small_poly9(t: BI64) -> BI64 {
    let z = t * t;
    let acc = mul_add(z, c(342579125), c(-594788126));
    let acc = mul_add(z, acc, c(857880307));
    let acc = mul_add(z, acc, c(-1431632248));
    let acc = mul_add(z, acc, c(4294967216));
    t * acc
}

const TAN_PI_8: i64 = 0x6a09e667; // tan(pi/8) = sqrt(2) - 1
const QUARTER_PI: i64 = 0xc90fdaa2;

/// atan on [0, 1] with a second reduction: atan(t) = pi/4 + atan((t - 1) / (t + 1)) for
/// t > tan(pi/8); costs one more division but needs only a degree 9 polynomial.
#[inline(always)]
fn atan_unit_reduced(t: BI64) -> BI64 {
    if t.raw <= TAN_PI_8 {
        atan_small_poly9(t)
    } else {
        // (1 - t) / (1 + t) in [0, tan(pi/8)], atan(t) = pi/4 - atan(that).
        let u = BI64 { raw: bi64::ONE - t.raw } / BI64 { raw: bi64::ONE + t.raw };
        BI64 { raw: QUARTER_PI - atan_small_poly9(u).raw }
    }
}

#[inline(always)]
fn atan2_unfold(a: BI64, swap: bool, x_neg: bool, y_neg: bool) -> BI64 {
    let a = if swap {
        HALF_PI - a.raw
    } else {
        a.raw
    };
    let a = if x_neg {
        PI - a
    } else {
        a
    };
    BI64 { raw: if y_neg {
        -a
    } else {
        a
    } }
}

pub fn atan2_poly15(y: BI64, x: BI64) -> BI64 {
    let ax = bi64::abs(x);
    let ay = bi64::abs(y);
    if ay.raw <= ax.raw {
        if ax.raw == 0 {
            return ax;
        }
        atan2_unfold(atan_unit_poly15(ay / ax), false, x.raw < 0, y.raw < 0)
    } else {
        atan2_unfold(atan_unit_poly15(ax / ay), true, x.raw < 0, y.raw < 0)
    }
}

pub fn atan2_reduced9(y: BI64, x: BI64) -> BI64 {
    let ax = bi64::abs(x);
    let ay = bi64::abs(y);
    if ay.raw <= ax.raw {
        if ax.raw == 0 {
            return ax;
        }
        atan2_unfold(atan_unit_reduced(ay / ax), false, x.raw < 0, y.raw < 0)
    } else {
        atan2_unfold(atan_unit_reduced(ax / ay), true, x.raw < 0, y.raw < 0)
    }
}

// ---------------------------------------------------------------------------------------------
// acos
// ---------------------------------------------------------------------------------------------

/// acos(x) = sqrt(1 - x) * P7(x) on [0, 1] (Abramowitz-Stegun 4.4.46 form, refitted), mirrored
/// for negative x. Max error ~2.9e-8.
pub fn acos_poly7(x: BI64) -> BI64 {
    let ax = bi64::abs(x);
    let s = bi64::sqrt(BI64 { raw: bi64::ONE - ax.raw });
    let acc = mul_add(ax, c(-5204374), c(27880872));
    let acc = mul_add(ax, acc, c(-72331769));
    let acc = mul_add(ax, acc, c(131950511));
    let acc = mul_add(ax, acc, c(-215239293));
    let acc = mul_add(ax, acc, c(382118315));
    let acc = mul_add(ax, acc, c(-921692060));
    let acc = mul_add(ax, acc, c(6746518730));
    let r = s * acc;
    if x.raw < 0 {
        BI64 { raw: PI - r.raw }
    } else {
        r
    }
}

/// acos(x) = atan2(sqrt(1 - x^2), x).
pub fn acos_via_atan2(x: BI64) -> BI64 {
    let s = bi64::sqrt(mul_add(BI64 { raw: -x.raw }, x, BI64 { raw: bi64::ONE }));
    atan2_reduced9(s, x)
}

// ---------------------------------------------------------------------------------------------
// Typed atan2 / acos: BoundedInt end to end (generated kernels in `poly_typed.cairo`).
// The only range checks left are the sign splits, the unsigned division and one downcast.
// ---------------------------------------------------------------------------------------------

type I64Neg = BoundedInt<-0x8000000000000000, -1>;
type Abs = BoundedInt<0, 0x8000000000000000>;
type AbsShl32 = BoundedInt<0, 0x800000000000000000000000>;
const PI_F: felt252 = 0x3243f6a89;
// Angles at the successive unfolding stages (all far inside i64, so `upcast` is enough).
type Ang1 = BoundedInt<-0x400000000, 0x400000000>;
type Ang2 = BoundedInt<-0x800000000, 0x800000000>;

impl MulAbsTwo32 of MulHelper<Abs, UnitInt<0x100000000>> {
    type Result = AbsShl32;
}
impl DivAbsShl32Abs of DivRemHelper<AbsShl32, Abs> {
    type DivT = AbsShl32;
    type RemT = BoundedInt<0, 0x7fffffffffffffff>;
}
impl SubHalfPiAng1 of SubHelper<UnitInt<HALF_PI_F>, Ang1> {
    type Result = BoundedInt<-0x26de04abc, 0x5921fb544>;
}
impl SubPiAng2 of SubHelper<UnitInt<PI_F>, Ang2> {
    type Result = BoundedInt<-0x4dbc09577, 0xb243f6a89>;
}
type Ang3 = BoundedInt<-0x1000000000, 0x1000000000>;
impl MulAng3Minus1 of MulHelper<Ang3, UnitInt<-1>> {
    type Result = Ang3;
}

// `core::zeroable::IsZeroResult` is crate-private: re-declare the libfunc with a local enum.
enum IsZero<T> {
    Zero,
    NonZero: NonZero<T>,
}
extern fn bounded_int_is_zero<T>(value: T) -> IsZero<T> implicits() nopanic;

#[inline(always)]
fn abs_typed(v: i64) -> (Abs, bool) {
    match bounded_int::constrain::<i64, 0>(v) {
        Ok(lt0) => (upcast(bounded_int::mul::<I64Neg, UnitInt<-1>>(lt0, -1)), true),
        Err(ge0) => (upcast(ge0), false),
    }
}

/// min / max in [0, 1] with one unsigned bounded division, or None when both are zero.
#[inline(always)]
fn ratio(num: Abs, den: Abs) -> Option<TUnit> {
    match bounded_int_is_zero(den) {
        IsZero::Zero => None,
        IsZero::NonZero(den) => {
            let n: AbsShl32 = bounded_int::mul::<Abs, UnitInt<0x100000000>>(num, 0x100000000);
            let (q, _) = bounded_int::div_rem::<AbsShl32, Abs>(n, den);
            Some(downcast(q).unwrap())
        },
    }
}

#[inline(always)]
fn unfold_typed(a: Ang1, swap: bool, x_neg: bool, y_neg: bool) -> BI64 {
    let a: Ang2 = if swap {
        upcast(bounded_int::sub::<UnitInt<HALF_PI_F>, Ang1>(0x1921fb544, a))
    } else {
        upcast(a)
    };
    let a: Ang3 = if x_neg {
        upcast(bounded_int::sub::<UnitInt<PI_F>, Ang2>(0x3243f6a89, a))
    } else {
        upcast(a)
    };
    BI64 {
        raw: if y_neg {
            upcast(bounded_int::mul::<Ang3, UnitInt<-1>>(a, -1))
        } else {
            upcast(a)
        },
    }
}

pub fn atan2_poly15_typed(y: BI64, x: BI64) -> BI64 {
    let (ax, x_neg) = abs_typed(x.raw);
    let (ay, y_neg) = abs_typed(y.raw);
    let ax_u: u64 = upcast(ax);
    let ay_u: u64 = upcast(ay);
    let swap = ay_u > ax_u;
    let t = if swap {
        ratio(ax, ay)
    } else {
        ratio(ay, ax)
    };
    match t {
        None => BI64 { raw: 0 },
        Some(t) => {
            let a: Atan15Out = poly_typed::atan_unit15(t);
            unfold_typed(upcast(a), swap, x_neg, y_neg)
        },
    }
}

pub fn atan2_poly19_typed(y: BI64, x: BI64) -> BI64 {
    let (ax, x_neg) = abs_typed(x.raw);
    let (ay, y_neg) = abs_typed(y.raw);
    let ax_u: u64 = upcast(ax);
    let ay_u: u64 = upcast(ay);
    let swap = ay_u > ax_u;
    let t = if swap {
        ratio(ax, ay)
    } else {
        ratio(ay, ax)
    };
    match t {
        None => BI64 { raw: 0 },
        Some(t) => {
            let a: Atan19Out = poly_typed::atan_unit19(t);
            unfold_typed(upcast(a), swap, x_neg, y_neg)
        },
    }
}

impl SubOneTUnit of SubHelper<UnitInt<0x100000000>, TUnit> {
    type Result = TUnit;
}
impl MulTUnitTwo32 of MulHelper<TUnit, UnitInt<0x100000000>> {
    type Result = BoundedInt<0, 0x10000000000000000>;
}

/// acos(x) = sqrt(1 - |x|) * P9(|x|), mirrored for x < 0; panics if |x| > 1.
pub fn acos_poly9_typed(x: BI64) -> BI64 {
    let (ax, x_neg) = abs_typed(x.raw);
    let ax: TUnit = downcast(ax).expect('acos domain');
    let om: TUnit = bounded_int::sub::<UnitInt<0x100000000>, TUnit>(0x100000000, ax);
    let wide: u128 = upcast(bounded_int::mul::<TUnit, UnitInt<0x100000000>>(om, 0x100000000));
    let s: TUnit = downcast(core::num::traits::Sqrt::sqrt(wide)).unwrap();
    let r: Acos9Out = poly_typed::acos_kernel9(ax, s);
    let r: Ang1 = upcast(r);
    BI64 {
        raw: if x_neg {
            let r: Ang2 = upcast(r);
            upcast(bounded_int::sub::<UnitInt<PI_F>, Ang2>(0x3243f6a89, r))
        } else {
            upcast(r)
        },
    }
}
