//! Raw Q32.32 transcendental kernels: signs, reflections, range folding and domain checks.
//!
//! The polynomial evaluations themselves are generated (`kernels::poly`, DESIGN D6); this module
//! is the hand-written half that turns them into total functions on `i64` raw values. Like the
//! rest of `kernels`, it takes and returns raw `i64` (`value * 2^32`); the typed API and the
//! numeric specification live in `simba::fixed::transcendental`.
//!
//! Range reduction (`sin`, `cos`, `sin_cos`, `tan`): one `div_rem` of `x * 2^29 + OFF` by `pi/4`
//! **at scale 2^61** gives an octant index `n` in `0..8` and a remainder `r` in `[0, pi/4)` with
//! 61 fractional bits. The three bits of `n` are exactly the three symmetries of the circle:
//!
//! | bit | value | meaning |
//! |---|---|---|
//! | `low` (`n & 1`) | odd octant | evaluate at `pi/4 - r` and exchange sine and cosine |
//! | `q` (`n & 2`) | odd quadrant | exchange sine and cosine, negate the cosine |
//! | `high` (`n & 4`) | lower half | negate both |
//!
//! so `sin` evaluates exactly one polynomial (`sin_octant` or `cos_octant`), `sin_cos` one shared
//! pair, and no branch needs a comparison against a reduction constant.
//!
//! `atan2` folds the plane onto `[0, 1]` with one unsigned division (`min / max` at scale 2^61),
//! then unfolds with `pi/2 - a`, `pi - a` and `-a`. `asin` / `acos` use `asin(s) = s * R(s^2)` on
//! `|x| <= 1/2` and `acos(x) = 2 asin(sqrt((1 - x) / 2))` above, which keeps both functions exact
//! at 0, 1/2 and 1 and accurate near `|x| = 1` (where `asin(x) = pi/2 - acos(x)` would cancel).

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, BoundedInt, ConstrainHelper, DivRemHelper, MulHelper, SubHelper, UnitInt, downcast,
    upcast,
};
use core::num::traits::Sqrt;
use crate::errors;
use super::{IsZero, bounded_int_is_zero, poly};

/// `pi/2` in Q32.32, floored: the same constant as `simba::fixed::types::FRAC_PI_2`.
const FRAC_PI_2_F: felt252 = 0x1921fb544;
/// `pi` in Q32.32, floored (exactly `2 * FRAC_PI_2_F`): `simba::fixed::types::PI`.
const PI_F: felt252 = 0x3243f6a88;
const ONE_F: felt252 = 0x100000000;
const TWO29: felt252 = 0x20000000;
const TWO89: felt252 = 0x20000000000000000000000;
const TWO93: felt252 = 0x200000000000000000000000;

// ---------------------------------------------------------------------------------------------
// Shared intervals. `Ang2` is any angle in `[0, pi/2]`, `Ang3` in `[0, pi]`, `Ang4` in `[-pi, pi]`
// (`upcast` to `i64` is then free); `Abs` is `|v|` for any `i64`.
// ---------------------------------------------------------------------------------------------

type Bit = BoundedInt<0, 1>;
type Quad = BoundedInt<0, 3>;
type Ang2 = BoundedInt<0, 0x1921fb544>;
type Ang3 = BoundedInt<0, 0x3243f6a88>;
type Ang4 = BoundedInt<-0x3243f6a88, 0x3243f6a88>;
type Abs = BoundedInt<0, 0x8000000000000000>;
type I64Neg = BoundedInt<-0x8000000000000000, -1>;
type I64NonNeg = BoundedInt<0, 0x7fffffffffffffff>;
/// `|x| <= 1` in Q32.32: the domain of `asin` / `acos`.
type Unit = BoundedInt<0, 0x100000000>;
type UnitSmall = BoundedInt<0, 0x80000000>;
type UnitBig = BoundedInt<0x80000001, 0x100000000>;
type OneMinusBig = BoundedInt<0, 0x7fffffff>;
type AtanBig = BoundedInt<0x100000001, 0x8000000000000000>;
type MagNeg = BoundedInt<-0x100000000, 0>;

impl DivOctantFour of DivRemHelper<poly::Octant, UnitInt<4>> {
    type DivT = Bit;
    type RemT = Quad;
}
impl DivQuadTwo of DivRemHelper<Quad, UnitInt<2>> {
    type DivT = Bit;
    type RemT = Bit;
}
// The corelib provides `MulMinus1` for the `i64` halves only; the other intervals need their own.
impl MulMagMinusOne of MulHelper<poly::Mag32, UnitInt<-1>> {
    type Result = MagNeg;
}
impl MulAng2MinusOne of MulHelper<Ang2, UnitInt<-1>> {
    type Result = BoundedInt<-0x1921fb544, 0>;
}
impl MulAng3MinusOne of MulHelper<Ang3, UnitInt<-1>> {
    type Result = BoundedInt<-0x3243f6a88, 0>;
}
impl SubHalfPiAngQp of SubHelper<UnitInt<FRAC_PI_2_F>, poly::AngQp> {
    type Result = BoundedInt<0x921fb544, 0x1921fb544>;
}
impl SubHalfPiAngPi6 of SubHelper<UnitInt<FRAC_PI_2_F>, poly::AngPi6> {
    type Result = BoundedInt<0x10b1fb544, 0x1921fb544>;
}
impl SubHalfPiAngPi3 of SubHelper<UnitInt<FRAC_PI_2_F>, poly::AngPi3> {
    type Result = BoundedInt<0x841fb544, 0x1921fb544>;
}
impl SubPiAng2 of SubHelper<UnitInt<PI_F>, Ang2> {
    type Result = BoundedInt<0x1921fb544, 0x3243f6a88>;
}
impl ConstrainUnitHalf of ConstrainHelper<Unit, 0x80000001> {
    type LowT = UnitSmall;
    type HighT = UnitBig;
}
impl ConstrainAbsOne of ConstrainHelper<Abs, 0x100000001> {
    type LowT = Unit;
    type HighT = AtanBig;
}
impl SubOneUnitBig of SubHelper<UnitInt<ONE_F>, UnitBig> {
    type Result = OneMinusBig;
}
impl MulUnitSmallTwo29 of MulHelper<UnitSmall, UnitInt<TWO29>> {
    type Result = poly::AsinArg;
}
impl MulUnitTwo29 of MulHelper<Unit, UnitInt<TWO29>> {
    type Result = poly::Ratio;
}
impl MulOneMinusBigTwo89 of MulHelper<OneMinusBig, UnitInt<TWO89>> {
    type Result = BoundedInt<0, 0xfffffffe0000000000000000000000>;
}
/// `|v| * 2^61`: the numerator of the `atan2` ratio.
type AbsShlA = BoundedInt<0, 0x10000000000000000000000000000000>;
impl MulAbsTwoA of MulHelper<Abs, UnitInt<0x2000000000000000>> {
    type Result = AbsShlA;
}
impl DivAbsShlAbs of DivRemHelper<AbsShlA, Abs> {
    type DivT = AbsShlA;
    type RemT = BoundedInt<0, 0x7fffffffffffffff>;
}
impl DivTwo93Abs of DivRemHelper<UnitInt<TWO93>, Abs> {
    type DivT = BoundedInt<0x40000000, TWO93>;
    type RemT = BoundedInt<0, 0x7fffffffffffffff>;
}

// ---------------------------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------------------------

/// `b != 0` for a one-bit `BoundedInt` (a felt comparison, no range check).
#[inline(always)]
fn is_set(b: Bit) -> bool {
    let b: felt252 = upcast(b);
    b != 0
}

/// The three symmetry bits of an octant index: `(high, q, low)` = `(n & 4, n & 2, n & 1)`.
#[inline(always)]
fn split_octant(n: poly::Octant) -> (bool, bool, bool) {
    let (high, m) = bounded_int::div_rem::<poly::Octant, UnitInt<4>>(n, 4);
    let (q, low) = bounded_int::div_rem::<Quad, UnitInt<2>>(m, 2);
    (is_set(high), is_set(q), is_set(low))
}

/// The argument of the octant polynomials: `r` on even octants, `pi/4 - r` on odd ones.
#[inline(always)]
fn octant_arg(low: bool, r: poly::OctRem) -> poly::OctArg {
    if low {
        poly::octant_complement(r)
    } else {
        upcast(r)
    }
}

/// A magnitude in `[0, 1]` with a sign: never overflows (`|result| <= 2^32`).
#[inline(always)]
fn signed(m: poly::Mag32, negative: bool) -> i64 {
    if negative {
        upcast(bounded_int::mul::<poly::Mag32, UnitInt<-1>>(m, -1))
    } else {
        upcast(m)
    }
}

/// `(|v|, v < 0)`: one range check (the sign split), no panic for `MIN`.
#[inline(always)]
fn abs_split(v: i64) -> (Abs, bool) {
    match bounded_int::constrain::<i64, 0>(v) {
        Ok(lt0) => (upcast(bounded_int::mul::<I64Neg, UnitInt<-1>>(lt0, -1)), true),
        Err(ge0) => (upcast(ge0), false),
    }
}

// ---------------------------------------------------------------------------------------------
// sin / cos / sin_cos / tan
// ---------------------------------------------------------------------------------------------

/// `sin` (or `cos`) from an octant index and its remainder: exactly one polynomial.
#[inline(always)]
fn sin_of_octant(n: poly::Octant, r: poly::OctRem) -> i64 {
    let (high, q, low) = split_octant(n);
    let arg = octant_arg(low, r);
    // The sine of the octant is the cosine of the reduced argument iff `q` and `low` differ.
    let mag = if q != low {
        poly::cos_octant(arg)
    } else {
        poly::sin_octant(arg)
    };
    signed(mag, high)
}

/// `sin(x)`: octant reduction then one octant polynomial. Total, cannot panic.
pub fn sin(x: i64) -> i64 {
    let (n, r) = poly::reduce_octant(x);
    sin_of_octant(n, r)
}

/// `cos(x) = sin(x + pi/2)`: the same reduction with the octant shifted by two.
pub fn cos(x: i64) -> i64 {
    let (n, r) = poly::reduce_octant_cos(x);
    sin_of_octant(n, r)
}

/// `(sin x, cos x)`: one reduction, one shared `r^2`. Bit for bit `(sin(x), cos(x))`.
pub fn sin_cos(x: i64) -> (i64, i64) {
    let (n, r) = poly::reduce_octant(x);
    let (high, q, low) = split_octant(n);
    let (a, b) = poly::sin_cos_octant(octant_arg(low, r));
    let (s0, c0) = if low {
        (b, a)
    } else {
        (a, b)
    };
    let (sm, cm) = if q {
        (c0, s0)
    } else {
        (s0, c0)
    };
    (signed(sm, high), signed(cm, q != high))
}

/// `tan(x) = sin(x) / cos(x)` computed from a sine and a cosine with 48 fractional bits, so the
/// quotient keeps its accuracy near the poles. Panics with `errors::OVERFLOW` when `|tan|` does
/// not fit Q32.32, and with `errors::DIVISION_BY_ZERO` if the reduced cosine is exactly zero.
pub fn tan(x: i64) -> i64 {
    let (n, r) = poly::reduce_octant(x);
    let (_high, q, low) = split_octant(n);
    let (a, b) = poly::sin_cos_octant_wide(octant_arg(low, r));
    let (s0, c0) = if low {
        (b, a)
    } else {
        (a, b)
    };
    let (sm, cm) = if q {
        (c0, s0)
    } else {
        (s0, c0)
    };
    // sign(tan) = sign(sin) XOR sign(cos) = high XOR (q XOR high) = q.
    let cm = match bounded_int_is_zero(cm) {
        IsZero::Zero => core::panic_with_felt252(errors::DIVISION_BY_ZERO),
        IsZero::NonZero(cm) => cm,
    };
    let mag: I64NonNeg = downcast(poly::tan_ratio(sm, cm)).expect(errors::OVERFLOW);
    if q {
        upcast(bounded_int::mul::<I64NonNeg, UnitInt<-1>>(mag, -1))
    } else {
        upcast(mag)
    }
}

// ---------------------------------------------------------------------------------------------
// atan / atan2
// ---------------------------------------------------------------------------------------------

/// `min / max` at scale 2^61, given `num <= den` and `den != 0`.
#[inline(always)]
fn ratio(num: Abs, den: NonZero<Abs>) -> poly::Ratio {
    let n = bounded_int::mul::<Abs, UnitInt<0x2000000000000000>>(num, 0x2000000000000000);
    let (q, _r) = bounded_int::div_rem::<AbsShlA, Abs>(n, den);
    // `num <= den` by construction, so `q <= 2^61`: the check cannot fire.
    downcast(q).expect(errors::OVERFLOW)
}

/// `a` in `[0, pi/4]` unfolded to the right quadrant: `pi/2 - a`, then `pi - a`, then `-a`.
#[inline(always)]
fn unfold(a: poly::AngQp, swap: bool, x_negative: bool, y_negative: bool) -> i64 {
    let a: Ang2 = if swap {
        upcast(bounded_int::sub::<UnitInt<FRAC_PI_2_F>, poly::AngQp>(0x1921fb544, a))
    } else {
        upcast(a)
    };
    let a: Ang3 = if x_negative {
        upcast(bounded_int::sub::<UnitInt<PI_F>, Ang2>(0x3243f6a88, a))
    } else {
        upcast(a)
    };
    let a: Ang4 = if y_negative {
        upcast(bounded_int::mul::<Ang3, UnitInt<-1>>(a, -1))
    } else {
        upcast(a)
    };
    upcast(a)
}

/// `atan2(y, x)` in `(-pi, pi]`. `atan2(0, 0)` is `0`. Total, cannot panic.
pub fn atan2(y: i64, x: i64) -> i64 {
    let (ax, x_negative) = abs_split(x);
    let (ay, y_negative) = abs_split(y);
    let axu: u64 = upcast(ax);
    let ayu: u64 = upcast(ay);
    let swap = ayu > axu;
    let (num, den) = if swap {
        (ax, ay)
    } else {
        (ay, ax)
    };
    match bounded_int_is_zero(den) {
        // Both are zero: the angle is undefined, `0` is returned (like `f64::atan2(0.0, 0.0)`).
        IsZero::Zero => 0,
        IsZero::NonZero(den) => unfold(
            poly::atan_unit(ratio(num, den)), swap, x_negative, y_negative,
        ),
    }
}

/// `atan(x)` in `(-pi/2, pi/2)`. Bit for bit `atan2(x, 2^32)`. Total, cannot panic.
///
/// `|x| <= 1` needs no division at all: the ratio is `x * 2^29`, exactly what `atan2` computes.
pub fn atan(x: i64) -> i64 {
    let (ax, negative) = abs_split(x);
    let (t, swap) = match bounded_int::constrain::<Abs, 0x100000001>(ax) {
        // `|x| * 2^29` is exactly what `atan2(x, 1)` divides out: no division at all.
        Ok(small) => (upcast(bounded_int::mul::<Unit, UnitInt<TWO29>>(small, 0x20000000)), false),
        Err(_big) => {
            // `|x| > 1`, so the zero test (on the wider `Abs` type, which `is_zero` accepts)
            // cannot fire; the quotient `2^93 / |x|` is `(2^32 << 61) / |x|`, as in `atan2`.
            let den = match bounded_int_is_zero(ax) {
                IsZero::Zero => core::panic_with_felt252(errors::DIVISION_BY_ZERO),
                IsZero::NonZero(v) => v,
            };
            let (q, _r) = bounded_int::div_rem::<
                UnitInt<TWO93>, Abs,
            >(0x200000000000000000000000, den);
            (downcast(q).expect(errors::OVERFLOW), true)
        },
    };
    let a: Ang2 = if swap {
        upcast(
            bounded_int::sub::<UnitInt<FRAC_PI_2_F>, poly::AngQp>(0x1921fb544, poly::atan_unit(t)),
        )
    } else {
        upcast(poly::atan_unit(t))
    };
    if negative {
        upcast(bounded_int::mul::<Ang2, UnitInt<-1>>(a, -1))
    } else {
        upcast(a)
    }
}

// ---------------------------------------------------------------------------------------------
// asin / acos
// ---------------------------------------------------------------------------------------------

/// `|x|` as a Q32.32 value in `[0, 1]`. Panics with `errors::DOMAIN` outside.
#[inline(always)]
fn unit_arg(x: i64) -> (Unit, bool) {
    let (ax, negative) = abs_split(x);
    (downcast(ax).expect(errors::DOMAIN), negative)
}

/// `sqrt((1 - x) / 2)` at scale 2^61, exactly floored, for `x` in `(1/2, 1]`.
#[inline(always)]
fn half_angle(x: UnitBig) -> poly::AsinArg {
    let d = bounded_int::sub::<UnitInt<ONE_F>, UnitBig>(0x100000000, x);
    let w: u128 = upcast(
        bounded_int::mul::<OneMinusBig, UnitInt<TWO89>>(d, 0x20000000000000000000000),
    );
    // `w <= (2^31 - 1) * 2^89 < 2^120`, so the root is below 2^60: the check cannot fire.
    downcast(w.sqrt()).expect(errors::OVERFLOW)
}

/// `asin(x)` in `[-pi/2, pi/2]`. Panics with `errors::DOMAIN` for `|x| > 1`.
pub fn asin(x: i64) -> i64 {
    let (ax, negative) = unit_arg(x);
    let m: Ang2 = match bounded_int::constrain::<Unit, 0x80000001>(ax) {
        Ok(small) => upcast(
            poly::asin_unit(bounded_int::mul::<UnitSmall, UnitInt<TWO29>>(small, 0x20000000)),
        ),
        Err(big) => upcast(
            bounded_int::sub::<
                UnitInt<FRAC_PI_2_F>, poly::AngPi3,
            >(0x1921fb544, poly::asin_double(half_angle(big))),
        ),
    };
    if negative {
        upcast(bounded_int::mul::<Ang2, UnitInt<-1>>(m, -1))
    } else {
        upcast(m)
    }
}

/// `acos(x)` in `[0, pi]`. Panics with `errors::DOMAIN` for `|x| > 1`.
pub fn acos(x: i64) -> i64 {
    let (ax, negative) = unit_arg(x);
    let m: Ang2 = match bounded_int::constrain::<Unit, 0x80000001>(ax) {
        Ok(small) => upcast(
            bounded_int::sub::<
                UnitInt<FRAC_PI_2_F>, poly::AngPi6,
            >(
                0x1921fb544,
                poly::asin_unit(bounded_int::mul::<UnitSmall, UnitInt<TWO29>>(small, 0x20000000)),
            ),
        ),
        Err(big) => upcast(poly::asin_double(half_angle(big))),
    };
    if negative {
        upcast(bounded_int::sub::<UnitInt<PI_F>, Ang2>(0x3243f6a88, m))
    } else {
        upcast(m)
    }
}

/// The same functions on the polynomial degrees that lost the accuracy / gas comparison
/// (`kernels::poly_alternatives`, `polygen.py --report`), kept as evidence (AGENTS.md rule 8).
/// Only the polynomial call differs; the folding is the code above.
#[cfg(test)]
pub mod alternatives {
    #[feature("bounded-int-utils")]
    use core::internal::bounded_int::{self, UnitInt, downcast, upcast};
    use crate::errors;
    use super::super::{IsZero, bounded_int_is_zero, poly, poly_alternatives as alt};
    // The interval impls of the parent module are brought in explicitly: impl resolution does
    // not look through the enclosing module.
    use super::{
        Abs, Ang2, ConstrainUnitHalf, DivTwo93Abs, MulAng2MinusOne, MulUnitSmallTwo29,
        SubHalfPiAngPi3, SubHalfPiAngQp, TWO29, TWO93, Unit, UnitSmall, abs_split, octant_arg,
        ratio, signed, split_octant, unfold, unit_arg,
    };

    /// `sin` with the degree 7 / 6 pair (fit error 466 ulp).
    pub fn sin_deg7(x: i64) -> i64 {
        let (n, r) = poly::reduce_octant(x);
        let (high, q, low) = split_octant(n);
        let arg = octant_arg(low, r);
        let mag = if q != low {
            alt::cos_octant_6(arg)
        } else {
            alt::sin_octant_7(arg)
        };
        signed(mag, high)
    }

    /// `sin` with the shipped degree 11 / 10 pair, but reached through this module like the other
    /// candidates: the three `bench_sin__deg*_alt` variants are then directly comparable (the
    /// extra call out of `poly_alternatives` costs 100 gas on all three).
    pub fn sin_deg11(x: i64) -> i64 {
        let (n, r) = poly::reduce_octant(x);
        let (high, q, low) = split_octant(n);
        let arg = octant_arg(low, r);
        let mag = if q != low {
            poly::cos_octant(arg)
        } else {
            poly::sin_octant(arg)
        };
        signed(mag, high)
    }

    /// `sin` with the degree 9 / 8 pair (1.31 ulp end to end, exactly the same gas).
    pub fn sin_deg9(x: i64) -> i64 {
        let (n, r) = poly::reduce_octant(x);
        let (high, q, low) = split_octant(n);
        let arg = octant_arg(low, r);
        let mag = if q != low {
            alt::cos_octant_8(arg)
        } else {
            alt::sin_octant_9(arg)
        };
        signed(mag, high)
    }

    /// `sin` through two separate polynomials instead of the shared `sin_cos_octant`.
    pub fn sin_cos_two_calls(x: i64) -> (i64, i64) {
        (super::sin(x), super::cos(x))
    }

    fn atan2_with(y: i64, x: i64, degree: felt252) -> i64 {
        let (ax, x_negative) = abs_split(x);
        let (ay, y_negative) = abs_split(y);
        let axu: u64 = upcast(ax);
        let ayu: u64 = upcast(ay);
        let swap = ayu > axu;
        let (num, den) = if swap {
            (ax, ay)
        } else {
            (ay, ax)
        };
        match bounded_int_is_zero(den) {
            IsZero::Zero => 0,
            IsZero::NonZero(den) => {
                let t = ratio(num, den);
                let a = if degree == 17 {
                    alt::atan_unit_17(t)
                } else if degree == 21 {
                    alt::atan_unit_21(t)
                } else {
                    alt::atan_unit_29(t)
                };
                unfold(a, swap, x_negative, y_negative)
            },
        }
    }

    /// `atan2` with the degree 17 polynomial (fit error 234 ulp).
    pub fn atan2_deg17(y: i64, x: i64) -> i64 {
        atan2_with(y, x, 17)
    }

    /// `atan2` with the degree 21 polynomial (fit error 6.4 ulp).
    pub fn atan2_deg21(y: i64, x: i64) -> i64 {
        atan2_with(y, x, 21)
    }

    /// `atan2` with the degree 29 polynomial (fit error 1.03 ulp).
    pub fn atan2_deg29(y: i64, x: i64) -> i64 {
        atan2_with(y, x, 29)
    }

    /// `atan` through `atan2(x, 1)`: the generic path, with its division.
    pub fn atan_via_atan2(x: i64) -> i64 {
        super::atan2(x, 0x100000000)
    }

    fn asin_with(x: i64, degree: felt252) -> i64 {
        let (ax, negative) = unit_arg(x);
        let m: Ang2 = match bounded_int::constrain::<Unit, 0x80000001>(ax) {
            Ok(small) => {
                let s = bounded_int::mul::<UnitSmall, UnitInt<TWO29>>(small, 0x20000000);
                upcast(if degree == 11 {
                    alt::asin_unit_11(s)
                } else {
                    alt::asin_unit_19(s)
                })
            },
            Err(big) => {
                let s = super::half_angle(big);
                let d = if degree == 11 {
                    alt::asin_double_11(s)
                } else {
                    alt::asin_double_19(s)
                };
                upcast(bounded_int::sub::<UnitInt<0x1921fb544>, poly::AngPi3>(0x1921fb544, d))
            },
        };
        if negative {
            upcast(bounded_int::mul::<Ang2, UnitInt<-1>>(m, -1))
        } else {
            upcast(m)
        }
    }

    /// `asin` with the degree 11 polynomial (fit error 311 ulp).
    pub fn asin_deg11(x: i64) -> i64 {
        asin_with(x, 11)
    }

    /// `asin` with the degree 19 polynomial (fit error 0.004 ulp).
    pub fn asin_deg19(x: i64) -> i64 {
        asin_with(x, 19)
    }

    /// `tan` from the Q32.32 sine and cosine instead of their 48-bit versions: one division less
    /// precise near the poles (kept to document the choice).
    pub fn tan_from_q32(x: i64) -> i64 {
        let (n, r) = poly::reduce_octant(x);
        let (_high, q, low) = split_octant(n);
        let (a, b) = poly::sin_cos_octant(octant_arg(low, r));
        let (s0, c0) = if low {
            (b, a)
        } else {
            (a, b)
        };
        let (sm, cm) = if q {
            (c0, s0)
        } else {
            (s0, c0)
        };
        let s: i64 = upcast(sm);
        let c: i64 = upcast(cm);
        let m = super::super::div(s, c);
        if q {
            -m
        } else {
            m
        }
    }

    fn unreachable_nonzero(v: Abs) -> NonZero<Abs> {
        match bounded_int_is_zero(v) {
            IsZero::Zero => core::panic_with_felt252(errors::DIVISION_BY_ZERO),
            IsZero::NonZero(v) => v,
        }
    }

    /// `atan` restricted to `abs(x) >= 1` (`pi/2 - atan(1/x)`): the lower bound on what `atan`
    /// could cost without its `abs(x) <= 1` shortcut. Sierra gas is the static maximum over the
    /// branches, so this measures the price of the extra `constrain`, not of the division.
    pub fn atan_reciprocal_only(x: i64) -> i64 {
        let (ax, negative) = abs_split(x);
        let den = unreachable_nonzero(ax);
        let (q, _r) = bounded_int::div_rem::<UnitInt<TWO93>, Abs>(0x200000000000000000000000, den);
        let t = downcast(q).expect(errors::OVERFLOW);
        let a: Ang2 = upcast(
            bounded_int::sub::<UnitInt<0x1921fb544>, poly::AngQp>(0x1921fb544, poly::atan_unit(t)),
        );
        if negative {
            upcast(bounded_int::mul::<Ang2, UnitInt<-1>>(a, -1))
        } else {
            upcast(a)
        }
    }
}
