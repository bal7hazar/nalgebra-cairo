//! Scalar abstraction (DESIGN D3): what `nalgebra` needs from a number.
//!
//! `nalgebra` types are generic over `T` with `+Real<T>` (plus the corelib operator traits they
//! use: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `PartialOrd`, `PartialEq`, `Copy`, `Drop`), so another
//! scalar (glam.cairo's, a future Q64.64, a plain integer in tests) plugs in by implementing this
//! trait. The trait exposes FUSED kernels, not just operators: generic code must route every sum
//! of products through `sum_prod*` / `diff_prod` / `mul_add` / the wide accumulator, so that a
//! fixed-point scalar rounds once per output and pays one overflow check.
//!
//! Counterpart of upstream `simba::scalar::RealField` (+ `ComplexField` helpers). Transcendental
//! functions live in the separate `Transcendental` trait.

use crate::fixed::types::Fixed;
use crate::fixed::wide::{Wide, WideTrait};
use crate::fixed::{convert, fused, math, transcendental, types};

/// Real scalar: constants, conversions, helpers, fused kernels and wide accumulation.
///
/// Numeric contract for implementors: deterministic, a single rounding per output (toward -inf
/// for fixed point), panics instead of wrapping.
pub trait Real<T> {
    /// Exact accumulator of unscaled products, see `wide_zero`.
    type Wide;

    /// 0.
    const ZERO: T;
    /// 1.
    const ONE: T;
    /// -1.
    const NEG_ONE: T;
    /// 2.
    const TWO: T;
    /// 1/2.
    const HALF: T;
    /// Smallest positive value (1 ulp). Not a float-style machine epsilon.
    const EPSILON: T;
    /// Smallest value.
    const MIN: T;
    /// Largest value.
    const MAX: T;
    /// π.
    const PI: T;
    /// 2π.
    const TAU: T;
    /// π/2.
    const FRAC_PI_2: T;
    /// π/3.
    const FRAC_PI_3: T;
    /// π/4.
    const FRAC_PI_4: T;
    /// π/6.
    const FRAC_PI_6: T;
    /// 1/π.
    const FRAC_1_PI: T;
    /// Euler's number.
    const E: T;
    /// ln 2.
    const LN_2: T;
    /// ln 10.
    const LN_10: T;
    /// √2.
    const SQRT_2: T;
    /// 1/√2.
    const FRAC_1_SQRT_2: T;

    // --- conversions ---------------------------------------------------------------------------

    /// The integer `v`, exactly.
    fn from_int(v: i32) -> T;
    /// `num / den` for two integers, rounded once. Panics on a zero denominator or overflow.
    fn from_ratio(num: i64, den: i64) -> T;
    /// `floor(self)` as an integer.
    fn to_int(self: T) -> i32;

    // --- helpers -------------------------------------------------------------------------------

    /// `|self|`.
    fn abs(self: T) -> T;
    /// `-1`, `0` or `1` according to the sign.
    fn signum(self: T) -> T;
    /// `self < 0`.
    fn is_negative(self: T) -> bool;
    /// `self > 0`.
    fn is_positive(self: T) -> bool;
    /// The smaller of two values.
    fn min(self: T, other: T) -> T;
    /// The larger of two values.
    fn max(self: T, other: T) -> T;
    /// `self` restricted to `[lo, hi]` (`lo <= hi` is not checked).
    fn clamp(self: T, lo: T, hi: T) -> T;
    /// Largest integer `<= self`.
    fn floor(self: T) -> T;
    /// Smallest integer `>= self`.
    fn ceil(self: T) -> T;
    /// Nearest integer, ties away from zero.
    fn round(self: T) -> T;
    /// Integer part, toward zero.
    fn trunc(self: T) -> T;
    /// `self - trunc(self)`.
    fn fract(self: T) -> T;
    /// `1 / self`, cheaper than the division operator.
    fn recip(self: T) -> T;
    /// Square root. Panics on a negative input.
    fn sqrt(self: T) -> T;
    /// `1 / sqrt(self)`, rounded once. Panics on a non-positive input.
    fn inv_sqrt(self: T) -> T;
    /// `|self - other| <= ulps` smallest units (raw units for fixed point).
    fn abs_diff_eq(self: T, other: T, ulps: u64) -> bool;

    // --- division: the ONLY division entry points of generic code -------------------------------

    /// `floor(a / b)` on the exact quotient (toward -inf, whatever the signs; for Q32.32 the
    /// quotient `a * 2^32 / b` is floored once). Panics on a zero divisor ('simba: division by
    /// zero' for the fixed-point scalars) and when the quotient is out of range ('simba:
    /// overflow').
    ///
    /// Generic code (nalgebra) must divide through `Real::div` / `Real::rem`, never through the
    /// corelib `/` / `%` operators: a scalar's own operators may round differently (glam.cairo's
    /// `fixed::Fixed` truncates toward zero), the trait pins the semantics. Upstream: `Div for
    /// f64`.
    fn div(a: T, b: T) -> T;
    /// FLOORED modulo `a - b * floor(a / b)`, exact: zero or the sign of the DIVISOR, and
    /// `|rem(a, b)| < |b|`. Cannot overflow. Panics on a zero divisor ('simba: division by zero'
    /// for the fixed-point scalars). Deviation from upstream: Rust's float `%` is the TRUNCATED
    /// remainder (sign of the dividend). Upstream: `Rem for f64`.
    fn rem(a: T, b: T) -> T;

    // --- fused kernels: one rounding, one overflow check ----------------------------------------

    /// `self^2`.
    fn sqr(self: T) -> T;
    /// `a0*b0 + a1*b1` (dot2, complex products).
    fn sum_prod2(a0: T, b0: T, a1: T, b1: T) -> T;
    /// `a0*b0 + a1*b1 + a2*b2` (dot3, matrix rows).
    fn sum_prod3(a0: T, b0: T, a1: T, b1: T, a2: T, b2: T) -> T;
    /// `a0*b0 + a1*b1 + a2*b2 + a3*b3` (dot4, quaternion products).
    fn sum_prod4(a0: T, b0: T, a1: T, b1: T, a2: T, b2: T, a3: T, b3: T) -> T;
    /// `a*b - c*d` (cross product components, 2x2 determinants).
    fn diff_prod(a: T, b: T, c: T, d: T) -> T;
    /// `a*b + c` (Horner steps, integration).
    fn mul_add(a: T, b: T, c: T) -> T;
    /// `a*b - c`.
    fn mul_sub(a: T, b: T, c: T) -> T;
    /// `a + (b - a) * t`, `t` not clamped.
    fn lerp(a: T, b: T, t: T) -> T;
    /// `x^2 + y^2`.
    fn norm_squared2(x: T, y: T) -> T;
    /// `x^2 + y^2 + z^2`.
    fn norm_squared3(x: T, y: T, z: T) -> T;
    /// `x^2 + y^2 + z^2 + w^2`.
    fn norm_squared4(x: T, y: T, z: T, w: T) -> T;
    /// `sqrt(x^2 + y^2)` without intermediate overflow.
    fn norm2(x: T, y: T) -> T;
    /// `sqrt(x^2 + y^2 + z^2)` without intermediate overflow.
    fn norm3(x: T, y: T, z: T) -> T;
    /// `sqrt(x^2 + y^2 + z^2 + w^2)` without intermediate overflow.
    fn norm4(x: T, y: T, z: T, w: T) -> T;

    // --- wide accumulation: sums of any number of products ---------------------------------------

    /// The empty exact accumulator. Accumulate with `wide_add_prod` / `wide_sub_prod` /
    /// `wide_add` / `wide_sub`, finish with `wide_rescale` (or `wide_sqrt`).
    fn wide_zero() -> Self::Wide;
    /// `w + a*b`, exact.
    fn wide_add_prod(w: Self::Wide, a: T, b: T) -> Self::Wide;
    /// `w - a*b`, exact.
    fn wide_sub_prod(w: Self::Wide, a: T, b: T) -> Self::Wide;
    /// `w + c`, exact.
    fn wide_add(w: Self::Wide, c: T) -> Self::Wide;
    /// `w - c`, exact.
    fn wide_sub(w: Self::Wide, c: T) -> Self::Wide;
    /// The accumulated value as a scalar: the single rounding / overflow check.
    fn wide_rescale(w: Self::Wide) -> T;
    /// Square root of the accumulated value (norm of a long vector).
    fn wide_sqrt(w: Self::Wide) -> T;
    /// The accumulated value times `s`, as a scalar: the single rounding / overflow check of an
    /// exact triple product `(a*b - c*d) * e` (cofactor expansions of 4x4 / 6x6 determinants),
    /// where `diff_prod(a, b, c, d) * e` or `wide_rescale(w) * s` would round twice. Terminal:
    /// consumes the accumulator.
    fn wide_mul_scalar(w: Self::Wide, s: T) -> T;
}

/// Transcendental functions (generated typed Horner polynomials, DESIGN D6). Declared separately
/// from `Real` so that a scalar can be a `Real` without them. Angles are in radians.
///
/// The numeric specification of every function (error in ulp, exact values, symmetries, domain
/// and panics) is documented on the implementation, `simba::fixed::transcendental`.
///
/// `exp2` / `log2` / `powi` are **not** part of this trait: nothing in nalgebra's static surface
/// needs them, `exp2(x) = exp(x * LN_2)` and `log2(x) = ln(x) * (1 / LN_2)` compose from what is
/// here, and `powi` with a runtime exponent needs a loop (AGENTS.md rule 1).
pub trait Transcendental<T> {
    /// Sine.
    fn sin(self: T) -> T;
    /// Cosine.
    fn cos(self: T) -> T;
    /// `(sin, cos)` with a shared range reduction.
    fn sin_cos(self: T) -> (T, T);
    /// Tangent.
    fn tan(self: T) -> T;
    /// Arcsine, in `[-π/2, π/2]`.
    fn asin(self: T) -> T;
    /// Arccosine, in `[0, π]`.
    fn acos(self: T) -> T;
    /// Arctangent, in `[-π/2, π/2]`.
    fn atan(self: T) -> T;
    /// Four-quadrant arctangent of `y / x`, in `(-π, π]`.
    fn atan2(y: T, x: T) -> T;
    /// Exponential.
    fn exp(self: T) -> T;
    /// Natural logarithm.
    fn ln(self: T) -> T;
}

/// `Transcendental` for the Q32.32 `Fixed`: `#[inline(always)]` forwards to the free functions of
/// `simba::fixed::transcendental`, where the numeric specification lives.
pub impl FixedTranscendental of Transcendental<Fixed> {
    #[inline(always)]
    fn sin(self: Fixed) -> Fixed {
        transcendental::sin(self)
    }
    #[inline(always)]
    fn cos(self: Fixed) -> Fixed {
        transcendental::cos(self)
    }
    #[inline(always)]
    fn sin_cos(self: Fixed) -> (Fixed, Fixed) {
        transcendental::sin_cos(self)
    }
    #[inline(always)]
    fn tan(self: Fixed) -> Fixed {
        transcendental::tan(self)
    }
    #[inline(always)]
    fn asin(self: Fixed) -> Fixed {
        transcendental::asin(self)
    }
    #[inline(always)]
    fn acos(self: Fixed) -> Fixed {
        transcendental::acos(self)
    }
    #[inline(always)]
    fn atan(self: Fixed) -> Fixed {
        transcendental::atan(self)
    }
    #[inline(always)]
    fn atan2(y: Fixed, x: Fixed) -> Fixed {
        transcendental::atan2(y, x)
    }
    #[inline(always)]
    fn exp(self: Fixed) -> Fixed {
        transcendental::exp(self)
    }
    #[inline(always)]
    fn ln(self: Fixed) -> Fixed {
        transcendental::ln(self)
    }
}

/// `Real` for the Q32.32 `Fixed`: every method is an `#[inline(always)]` forward to the free
/// functions of `simba::fixed::{convert, math, fused, wide}` (documented there), zero-cost.
pub impl FixedReal of Real<Fixed> {
    type Wide = Wide;

    const ZERO: Fixed = types::ZERO;
    const ONE: Fixed = types::ONE;
    const NEG_ONE: Fixed = types::NEG_ONE;
    const TWO: Fixed = types::TWO;
    const HALF: Fixed = types::HALF;
    const EPSILON: Fixed = types::EPSILON;
    const MIN: Fixed = types::MIN;
    const MAX: Fixed = types::MAX;
    const PI: Fixed = types::PI;
    const TAU: Fixed = types::TAU;
    const FRAC_PI_2: Fixed = types::FRAC_PI_2;
    const FRAC_PI_3: Fixed = types::FRAC_PI_3;
    const FRAC_PI_4: Fixed = types::FRAC_PI_4;
    const FRAC_PI_6: Fixed = types::FRAC_PI_6;
    const FRAC_1_PI: Fixed = types::FRAC_1_PI;
    const E: Fixed = types::E;
    const LN_2: Fixed = types::LN_2;
    const LN_10: Fixed = types::LN_10;
    const SQRT_2: Fixed = types::SQRT_2;
    const FRAC_1_SQRT_2: Fixed = types::FRAC_1_SQRT_2;

    #[inline(always)]
    fn from_int(v: i32) -> Fixed {
        convert::from_int(v)
    }
    #[inline(always)]
    fn from_ratio(num: i64, den: i64) -> Fixed {
        convert::from_ratio(num, den)
    }
    #[inline(always)]
    fn to_int(self: Fixed) -> i32 {
        convert::to_int(self)
    }

    #[inline(always)]
    fn abs(self: Fixed) -> Fixed {
        math::abs(self)
    }
    #[inline(always)]
    fn signum(self: Fixed) -> Fixed {
        math::signum(self)
    }
    #[inline(always)]
    fn is_negative(self: Fixed) -> bool {
        math::is_negative(self)
    }
    #[inline(always)]
    fn is_positive(self: Fixed) -> bool {
        math::is_positive(self)
    }
    #[inline(always)]
    fn min(self: Fixed, other: Fixed) -> Fixed {
        math::min(self, other)
    }
    #[inline(always)]
    fn max(self: Fixed, other: Fixed) -> Fixed {
        math::max(self, other)
    }
    #[inline(always)]
    fn clamp(self: Fixed, lo: Fixed, hi: Fixed) -> Fixed {
        math::clamp(self, lo, hi)
    }
    #[inline(always)]
    fn floor(self: Fixed) -> Fixed {
        math::floor(self)
    }
    #[inline(always)]
    fn ceil(self: Fixed) -> Fixed {
        math::ceil(self)
    }
    #[inline(always)]
    fn round(self: Fixed) -> Fixed {
        math::round(self)
    }
    #[inline(always)]
    fn trunc(self: Fixed) -> Fixed {
        math::trunc(self)
    }
    #[inline(always)]
    fn fract(self: Fixed) -> Fixed {
        math::fract(self)
    }
    #[inline(always)]
    fn recip(self: Fixed) -> Fixed {
        math::recip(self)
    }
    #[inline(always)]
    fn sqrt(self: Fixed) -> Fixed {
        math::sqrt(self)
    }
    #[inline(always)]
    fn inv_sqrt(self: Fixed) -> Fixed {
        math::inv_sqrt(self)
    }
    #[inline(always)]
    fn abs_diff_eq(self: Fixed, other: Fixed, ulps: u64) -> bool {
        math::abs_diff_eq(self, other, ulps)
    }

    /// The `/` operator of `Fixed` (`ops::FixedDiv`, floor): zero-cost.
    #[inline(always)]
    fn div(a: Fixed, b: Fixed) -> Fixed {
        a / b
    }
    /// The `%` operator of `Fixed` (`ops::FixedRem`, floored modulo): zero-cost.
    #[inline(always)]
    fn rem(a: Fixed, b: Fixed) -> Fixed {
        a % b
    }

    #[inline(always)]
    fn sqr(self: Fixed) -> Fixed {
        fused::sqr(self)
    }
    #[inline(always)]
    fn sum_prod2(a0: Fixed, b0: Fixed, a1: Fixed, b1: Fixed) -> Fixed {
        fused::sum_prod2(a0, b0, a1, b1)
    }
    #[inline(always)]
    fn sum_prod3(a0: Fixed, b0: Fixed, a1: Fixed, b1: Fixed, a2: Fixed, b2: Fixed) -> Fixed {
        fused::sum_prod3(a0, b0, a1, b1, a2, b2)
    }
    #[inline(always)]
    fn sum_prod4(
        a0: Fixed, b0: Fixed, a1: Fixed, b1: Fixed, a2: Fixed, b2: Fixed, a3: Fixed, b3: Fixed,
    ) -> Fixed {
        fused::sum_prod4(a0, b0, a1, b1, a2, b2, a3, b3)
    }
    #[inline(always)]
    fn diff_prod(a: Fixed, b: Fixed, c: Fixed, d: Fixed) -> Fixed {
        fused::diff_prod(a, b, c, d)
    }
    #[inline(always)]
    fn mul_add(a: Fixed, b: Fixed, c: Fixed) -> Fixed {
        fused::mul_add(a, b, c)
    }
    #[inline(always)]
    fn mul_sub(a: Fixed, b: Fixed, c: Fixed) -> Fixed {
        fused::mul_sub(a, b, c)
    }
    #[inline(always)]
    fn lerp(a: Fixed, b: Fixed, t: Fixed) -> Fixed {
        fused::lerp(a, b, t)
    }
    #[inline(always)]
    fn norm_squared2(x: Fixed, y: Fixed) -> Fixed {
        fused::norm_squared2(x, y)
    }
    #[inline(always)]
    fn norm_squared3(x: Fixed, y: Fixed, z: Fixed) -> Fixed {
        fused::norm_squared3(x, y, z)
    }
    #[inline(always)]
    fn norm_squared4(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Fixed {
        fused::norm_squared4(x, y, z, w)
    }
    #[inline(always)]
    fn norm2(x: Fixed, y: Fixed) -> Fixed {
        fused::norm2(x, y)
    }
    #[inline(always)]
    fn norm3(x: Fixed, y: Fixed, z: Fixed) -> Fixed {
        fused::norm3(x, y, z)
    }
    #[inline(always)]
    fn norm4(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Fixed {
        fused::norm4(x, y, z, w)
    }

    #[inline(always)]
    fn wide_zero() -> Wide {
        WideTrait::zero()
    }
    #[inline(always)]
    fn wide_add_prod(w: Wide, a: Fixed, b: Fixed) -> Wide {
        w.add_prod(a, b)
    }
    #[inline(always)]
    fn wide_sub_prod(w: Wide, a: Fixed, b: Fixed) -> Wide {
        w.sub_prod(a, b)
    }
    #[inline(always)]
    fn wide_add(w: Wide, c: Fixed) -> Wide {
        w.add(c)
    }
    #[inline(always)]
    fn wide_sub(w: Wide, c: Fixed) -> Wide {
        w.sub(c)
    }
    #[inline(always)]
    fn wide_rescale(w: Wide) -> Fixed {
        w.rescale()
    }
    #[inline(always)]
    fn wide_sqrt(w: Wide) -> Fixed {
        w.sqrt()
    }
    #[inline(always)]
    fn wide_mul_scalar(w: Wide, s: Fixed) -> Fixed {
        w.mul_scalar(s)
    }
}

#[cfg(test)]
mod tests {
    use core::num::traits::Sqrt;
    use nalgebra_testing::black_box;
    use crate::errors;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::types::{self, Fixed};
    use crate::fixed::wide::WideTrait;
    use super::Real;

    // --- a plain integer scalar: proof that `Real` is implementable without fixed point --------

    /// `Real` on integers (`i64`, exact products in an `i128` accumulator). Test-only.
    impl I64Real of Real<i64> {
        type Wide = i128;

        const ZERO: i64 = 0;
        const ONE: i64 = 1;
        const NEG_ONE: i64 = -1;
        const TWO: i64 = 2;
        const HALF: i64 = 0;
        const EPSILON: i64 = 1;
        const MIN: i64 = -0x8000000000000000;
        const MAX: i64 = 0x7fffffffffffffff;
        const PI: i64 = 3;
        const TAU: i64 = 6;
        const FRAC_PI_2: i64 = 1;
        const FRAC_PI_3: i64 = 1;
        const FRAC_PI_4: i64 = 0;
        const FRAC_PI_6: i64 = 0;
        const FRAC_1_PI: i64 = 0;
        const E: i64 = 2;
        const LN_2: i64 = 0;
        const LN_10: i64 = 2;
        const SQRT_2: i64 = 1;
        const FRAC_1_SQRT_2: i64 = 0;

        fn from_int(v: i32) -> i64 {
            v.into()
        }
        fn from_ratio(num: i64, den: i64) -> i64 {
            num / den
        }
        fn to_int(self: i64) -> i32 {
            self.try_into().unwrap()
        }
        fn abs(self: i64) -> i64 {
            if self < 0 {
                -self
            } else {
                self
            }
        }
        fn signum(self: i64) -> i64 {
            if self > 0 {
                1
            } else if self < 0 {
                -1
            } else {
                0
            }
        }
        fn is_negative(self: i64) -> bool {
            self < 0
        }
        fn is_positive(self: i64) -> bool {
            self > 0
        }
        fn min(self: i64, other: i64) -> i64 {
            core::cmp::min(self, other)
        }
        fn max(self: i64, other: i64) -> i64 {
            core::cmp::max(self, other)
        }
        fn clamp(self: i64, lo: i64, hi: i64) -> i64 {
            core::cmp::max(lo, core::cmp::min(self, hi))
        }
        fn floor(self: i64) -> i64 {
            self
        }
        fn ceil(self: i64) -> i64 {
            self
        }
        fn round(self: i64) -> i64 {
            self
        }
        fn trunc(self: i64) -> i64 {
            self
        }
        fn fract(self: i64) -> i64 {
            0
        }
        fn recip(self: i64) -> i64 {
            1 / self
        }
        fn sqrt(self: i64) -> i64 {
            let v: u64 = self.try_into().expect('sqrt of negative');
            let r: u32 = v.sqrt();
            r.into()
        }
        fn inv_sqrt(self: i64) -> i64 {
            1 / Self::sqrt(self)
        }
        /// Floor division (the corelib `/` on `i64` truncates), with simba's panic messages.
        fn div(a: i64, b: i64) -> i64 {
            let (q, _) = floor_div_rem(a, b);
            q.try_into().expect(errors::OVERFLOW)
        }
        /// Floored modulo (sign of the divisor), with simba's panic message.
        fn rem(a: i64, b: i64) -> i64 {
            let (_, r) = floor_div_rem(a, b);
            r.try_into().unwrap()
        }
        fn abs_diff_eq(self: i64, other: i64, ulps: u64) -> bool {
            let d: i128 = self.into() - other.into();
            let d: i128 = if d < 0 {
                -d
            } else {
                d
            };
            d <= ulps.into()
        }
        fn sqr(self: i64) -> i64 {
            self * self
        }
        fn sum_prod2(a0: i64, b0: i64, a1: i64, b1: i64) -> i64 {
            Self::wide_rescale(Self::wide_add_prod(Self::wide_add_prod(0, a0, b0), a1, b1))
        }
        fn sum_prod3(a0: i64, b0: i64, a1: i64, b1: i64, a2: i64, b2: i64) -> i64 {
            let w = Self::wide_add_prod(Self::wide_add_prod(0, a0, b0), a1, b1);
            Self::wide_rescale(Self::wide_add_prod(w, a2, b2))
        }
        fn sum_prod4(
            a0: i64, b0: i64, a1: i64, b1: i64, a2: i64, b2: i64, a3: i64, b3: i64,
        ) -> i64 {
            let w = Self::wide_add_prod(Self::wide_add_prod(0, a0, b0), a1, b1);
            let w = Self::wide_add_prod(Self::wide_add_prod(w, a2, b2), a3, b3);
            Self::wide_rescale(w)
        }
        fn diff_prod(a: i64, b: i64, c: i64, d: i64) -> i64 {
            Self::wide_rescale(Self::wide_sub_prod(Self::wide_add_prod(0, a, b), c, d))
        }
        fn mul_add(a: i64, b: i64, c: i64) -> i64 {
            Self::wide_rescale(Self::wide_add(Self::wide_add_prod(0, a, b), c))
        }
        fn mul_sub(a: i64, b: i64, c: i64) -> i64 {
            Self::wide_rescale(Self::wide_sub(Self::wide_add_prod(0, a, b), c))
        }
        fn lerp(a: i64, b: i64, t: i64) -> i64 {
            Self::wide_rescale(a.into() + (b.into() - a.into()) * t.into())
        }
        fn norm_squared2(x: i64, y: i64) -> i64 {
            Self::sum_prod2(x, x, y, y)
        }
        fn norm_squared3(x: i64, y: i64, z: i64) -> i64 {
            Self::sum_prod3(x, x, y, y, z, z)
        }
        fn norm_squared4(x: i64, y: i64, z: i64, w: i64) -> i64 {
            Self::sum_prod4(x, x, y, y, z, z, w, w)
        }
        fn norm2(x: i64, y: i64) -> i64 {
            Self::wide_sqrt(Self::wide_add_prod(Self::wide_add_prod(0, x, x), y, y))
        }
        fn norm3(x: i64, y: i64, z: i64) -> i64 {
            let w = Self::wide_add_prod(Self::wide_add_prod(0, x, x), y, y);
            Self::wide_sqrt(Self::wide_add_prod(w, z, z))
        }
        fn norm4(x: i64, y: i64, z: i64, w: i64) -> i64 {
            let acc = Self::wide_add_prod(Self::wide_add_prod(0, x, x), y, y);
            Self::wide_sqrt(Self::wide_add_prod(Self::wide_add_prod(acc, z, z), w, w))
        }
        fn wide_zero() -> i128 {
            0
        }
        fn wide_add_prod(w: i128, a: i64, b: i64) -> i128 {
            w + a.into() * b.into()
        }
        fn wide_sub_prod(w: i128, a: i64, b: i64) -> i128 {
            w - a.into() * b.into()
        }
        fn wide_add(w: i128, c: i64) -> i128 {
            w + c.into()
        }
        fn wide_sub(w: i128, c: i64) -> i128 {
            w - c.into()
        }
        fn wide_rescale(w: i128) -> i64 {
            w.try_into().expect('overflow')
        }
        fn wide_sqrt(w: i128) -> i64 {
            let v: u128 = w.try_into().expect('sqrt of negative');
            let r: u64 = v.sqrt();
            r.try_into().expect('overflow')
        }
        /// Exact integer product, checked. Restriction: the `i128` product itself must not
        /// overflow (`|w * s| < 2^127`), otherwise the corelib panics with 'i128_mul Overflow'
        /// before the 'overflow' check of the result; the tests stay far inside that range.
        fn wide_mul_scalar(w: i128, s: i64) -> i64 {
            Self::wide_rescale(w * s.into())
        }
    }

    /// Floored `(quotient, remainder)` of two `i64`, in `i128` (so that `MIN / -1` is caught by
    /// the caller's range check rather than by the corelib).
    fn floor_div_rem(a: i64, b: i64) -> (i128, i128) {
        if b == 0 {
            core::panic_with_felt252(errors::DIVISION_BY_ZERO);
        }
        let (a, b): (i128, i128) = (a.into(), b.into());
        let (q, r) = (a / b, a % b);
        if r != 0 && (r < 0) != (b < 0) {
            (q - 1, r + b)
        } else {
            (q, r)
        }
    }

    // --- generic code written against `Real` only (what `nalgebra` does)
    // ---------------------------

    #[derive(Copy, Drop)]
    struct V3<T> {
        x: T,
        y: T,
        z: T,
    }

    #[generate_trait]
    impl V3Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>> of V3Trait<T> {
        #[inline(always)]
        fn dot(self: V3<T>, o: V3<T>) -> T {
            R::sum_prod3(self.x, o.x, self.y, o.y, self.z, o.z)
        }
        #[inline(always)]
        fn cross(self: V3<T>, o: V3<T>) -> V3<T> {
            V3 {
                x: R::diff_prod(self.y, o.z, self.z, o.y),
                y: R::diff_prod(self.z, o.x, self.x, o.z),
                z: R::diff_prod(self.x, o.y, self.y, o.x),
            }
        }
        #[inline(always)]
        fn norm(self: V3<T>) -> T {
            R::norm3(self.x, self.y, self.z)
        }
        /// Same dot product through the explicit accumulator.
        #[inline(always)]
        fn dot_wide(self: V3<T>, o: V3<T>) -> T {
            let w = R::wide_add_prod(R::wide_zero(), self.x, o.x);
            let w = R::wide_add_prod(w, self.y, o.y);
            R::wide_rescale(R::wide_add_prod(w, self.z, o.z))
        }
        #[inline(always)]
        fn unit_x() -> V3<T> {
            V3 { x: R::ONE, y: R::ZERO, z: R::ZERO }
        }
    }

    #[test]
    fn test_real_generic_code_on_integers() {
        let a = V3 { x: 1_i64, y: -2, z: 2 };
        let b = V3 { x: 4_i64, y: 0, z: -3 };
        assert!(a.dot(b) == -2);
        assert!(a.dot_wide(b) == -2);
        assert!(a.norm() == 3);
        let c = a.cross(b);
        assert!(c.x == 6 && c.y == 11 && c.z == 8);
        assert!(V3Trait::<i64>::unit_x().dot(b) == 4);
    }

    #[test]
    fn test_real_generic_code_on_fixed() {
        let a = V3 { x: fx(0x100000000), y: fx(-0x200000000), z: fx(0x200000000) };
        let b = V3 { x: fx(0x400000000), y: fx(0), z: fx(-0x300000000) };
        assert!(a.dot(b) == fx(-0x200000000));
        assert!(a.dot_wide(b) == fx(-0x200000000));
        assert!(a.norm() == fx(0x300000000));
        let c = a.cross(b);
        assert!(c.x == fx(0x600000000) && c.y == fx(0xb00000000) && c.z == fx(0x800000000));
        assert!(V3Trait::<Fixed>::unit_x().dot(b) == fx(0x400000000));
    }

    #[test]
    fn test_real_constants_forward_to_fixed() {
        assert!(Real::<Fixed>::ZERO == types::ZERO);
        assert!(Real::<Fixed>::ONE == types::ONE);
        assert!(Real::<Fixed>::NEG_ONE == types::NEG_ONE);
        assert!(Real::<Fixed>::TWO == types::TWO);
        assert!(Real::<Fixed>::HALF == types::HALF);
        assert!(Real::<Fixed>::EPSILON == types::EPSILON);
        assert!(Real::<Fixed>::MIN == types::MIN);
        assert!(Real::<Fixed>::MAX == types::MAX);
        assert!(Real::<Fixed>::PI == types::PI);
        assert!(Real::<Fixed>::TAU == types::TAU);
        assert!(Real::<Fixed>::FRAC_PI_2 == types::FRAC_PI_2);
        assert!(Real::<Fixed>::FRAC_PI_3 == types::FRAC_PI_3);
        assert!(Real::<Fixed>::FRAC_PI_4 == types::FRAC_PI_4);
        assert!(Real::<Fixed>::FRAC_PI_6 == types::FRAC_PI_6);
        assert!(Real::<Fixed>::FRAC_1_PI == types::FRAC_1_PI);
        assert!(Real::<Fixed>::E == types::E);
        assert!(Real::<Fixed>::LN_2 == types::LN_2);
        assert!(Real::<Fixed>::LN_10 == types::LN_10);
        assert!(Real::<Fixed>::SQRT_2 == types::SQRT_2);
        assert!(Real::<Fixed>::FRAC_1_SQRT_2 == types::FRAC_1_SQRT_2);
    }

    #[test]
    fn test_real_method_syntax_on_fixed() {
        let x = fx(-0x280000000); // -2.5
        assert!(x.abs() == fx(0x280000000));
        assert!(x.signum() == types::NEG_ONE);
        assert!(x.is_negative() && !x.is_positive());
        assert!(x.min(types::ONE) == x);
        assert!(x.max(types::ONE) == types::ONE);
        assert!(x.clamp(types::NEG_ONE, types::ONE) == types::NEG_ONE);
        assert!(x.floor() == fx(-0x300000000));
        assert!(x.ceil() == fx(-0x200000000));
        assert!(x.round() == fx(-0x300000000));
        assert!(x.trunc() == fx(-0x200000000));
        assert!(x.fract() == fx(-0x80000000));
        assert!(x.to_int() == -3);
        assert!(x.sqr() == fx(0x640000000));
        assert!(types::TWO.recip() == types::HALF);
        assert!(types::TWO.sqrt() == types::SQRT_2);
        assert!(fx(0x400000000).inv_sqrt() == types::HALF);
        assert!(x.abs_diff_eq(fx(-0x280000003), 3));
        assert!(Real::<Fixed>::from_int(-3) == fx(-0x300000000));
        assert!(Real::<Fixed>::from_ratio(1, 3) == fx(1431655765));
    }

    #[test]
    fn test_real_kernels_forward_to_fixed() {
        let (a, b, c, d) = (fx(0x180000000), fx(-0x480000000), fx(-0x240000000), fx(0x40000000));
        assert!(Real::sum_prod2(a, b, c, d) == fx(-0x750000000));
        assert!(Real::sum_prod3(a, b, c, d, a, a) == fx(-0x510000000));
        assert!(Real::sum_prod4(a, b, c, d, a, a, d, d) == fx(-0x500000000));
        assert!(Real::diff_prod(a, b, c, d) == fx(-0x630000000));
        assert!(Real::mul_add(a, b, c) == fx(-0x900000000));
        assert!(Real::mul_sub(a, b, c) == fx(-0x480000000));
        assert!(Real::lerp(a, b, d) == fx(0));
        assert!(Real::norm_squared2(a, c) == fx(0x750000000));
        assert!(Real::norm_squared3(a, c, d) == fx(0x760000000));
        assert!(Real::norm_squared4(a, c, d, d) == fx(0x770000000));
        let (three, four) = (fx(0x300000000), fx(0x400000000));
        assert!(Real::norm2(three, four) == fx(0x500000000));
        assert!(Real::norm3(three, four, fx(0)) == fx(0x500000000));
        assert!(Real::norm4(three, four, fx(0), fx(0)) == fx(0x500000000));
        let w = Real::<Fixed>::wide_zero();
        let w = Real::wide_sub_prod(Real::wide_add_prod(w, a, b), c, d);
        let w = Real::wide_sub(Real::wide_add(w, a), d);
        assert!(Real::wide_rescale(w) == fx(-0x4f0000000));
        assert!(Real::wide_sqrt(WideTrait::from_prod(four, four)) == four);
        let w = Real::wide_sub_prod(Real::wide_add_prod(Real::<Fixed>::wide_zero(), a, b), c, d);
        assert!(Real::wide_mul_scalar(w, three) == fx(-0x1290000000));
    }

    #[test]
    fn test_real_wide_mul_scalar_on_integers() {
        // (3 * 4 - 2 * 5) * -7 = -14
        let w = Real::wide_sub_prod(Real::wide_add_prod(Real::<i64>::wide_zero(), 3, 4), 2, 5);
        assert!(Real::wide_mul_scalar(w, -7_i64) == -14);
        // An accumulator beyond i64, brought back by a zero / unit scalar.
        let big = Real::wide_add_prod(Real::<i64>::wide_zero(), 0x7fffffffffffffff, 4);
        assert!(Real::wide_mul_scalar(big, 0_i64) == 0);
        let back = Real::wide_sub_prod(big, 0x7fffffffffffffff, 4);
        assert!(Real::wide_mul_scalar(Real::wide_add(back, 5), -1_i64) == -5);
    }

    #[test]
    #[should_panic(expected: 'overflow')]
    fn test_real_wide_mul_scalar_on_integers_overflow_panics() {
        let w = Real::wide_add_prod(Real::<i64>::wide_zero(), black_box(0x100000000), 0x100000000);
        Real::wide_mul_scalar(w, 0x80000000_i64);
    }

    // --- division: `Real::div` / `Real::rem` -----------------------------------------------------

    /// `Real::<Fixed>::div` is the `/` operator: equal to it and to the model on the first block of
    /// generated division vectors (rows copied from `fixed::tests_generated::test_div_vectors_0`,
    /// generated by `tools/fixed_model/gen_vectors.py`; the other blocks already pin `/`).
    #[test]
    fn test_div_matches_operator_on_generated_vectors() {
        let rows: [[i64; 3]; 64] = [
            [32748, -10, -14065158900941], [357029614, 118101551, 12983999810],
            [1757481675, 1087208693956608, 6942], [-33198888, -1, 142588138223566848],
            [-2147483648, -287904979026926, 32036],
            [651836042556193, -42144965, -66428207619413451],
            [-1933725011923, 441214281292, -18823700950],
            [551147283665, -16818786101, -140744970797], [57359134, 1, 246355604656881664],
            [-180782904, 411697558, -1885988016], [20344387, -1545958322353, -56521],
            [190756, 774913031929856, 1], [-5, 535049906072071, -1], [64, 32471957574, 8],
            [1386359608573952, 9954760599, 598142880494647], [-2, 460860, -18639],
            [248, 28009411214338556, 0], [-1608, 13, -531254416306],
            [1192246414147584, -4294967295, -1192246414425176], [2, 53905974118, 0],
            [-6678847291, -3511858246, 8168163029], [143110090, 2, 307326578138808320],
            [-6103008, -16548399440809, 1583], [-15, 13, -4955731496], [5, 661023384141824, 0],
            [-235718486788, 2381838871548302, -425052], [-4, -1076306306813, 0],
            [-3123956561, 1712770975, -7833675057], [-7, -61358341811, 0], [-75, -3, 107374182400],
            [-175704271606, 8377320803177, -90081796], [-1873877812, 310634762265655952, -26],
            [-7530581, -612434, 52811566821],
            [9223372036854775806, -139064598593536, -284861004581892],
            [1072455481294848, 411049845063680, 11205845894], [7, 459310245085184, 0],
            [-274237243837, 903349079, -1303859184679],
            [-1041490914574336, 11904302, -375760747432138641],
            [15071122727, -15485697007, -4179984873], [2, -328195873751, -1],
            [1730687136694272, -204855427, -36285319654771850], [0, 2147483648, 0],
            [2, -3156, -2721780], [-37191791, 2111390890328064, -76],
            [-240955473771, 9223372036854775806, -113], [-60169840792, 107550559349, -2402846624],
            [-7, 1315432518647808, -1], [-320727646609, -460732065724, 2989839118],
            [-903532461, 467930269474, -8293207], [-951824882335744, -9223372036854775807, 443228],
            [-37, 2185033551905399400, -1], [-31381612, 4236361559738, -31816],
            [3911278317860, 1383905034764288, 12138703], [6366, -189710, -144123989],
            [-517816967, 6466225116556, -343943], [-1551287829, -10, 666273049223784038],
            [-11340536, 5, -9741446247822132], [-366867629136, -3543257103716659, 444699],
            [1983344231612418091, -2244094642356224, -3795917716974], [2, 7, 1227133513],
            [-13791105, -279313009759, 212064], [0, 2853417650662, 0], [-8, 302268928514598, -1],
            [-686, 1, -2946347565056],
        ];
        for row in rows.span() {
            let [a, b, e] = *row;
            assert!(Real::div(fx(a), fx(b)) == fx(e), "div {:?}", row.span());
            assert!(Real::div(fx(a), fx(b)) == fx(a) / fx(b), "div op {:?}", row.span());
        }
    }

    /// Same for `Real::<Fixed>::rem` and `%` (rows of `test_rem_vectors_0`).
    #[test]
    fn test_rem_matches_operator_on_generated_vectors() {
        let rows: [[i64; 3]; 64] = [
            [-16, -2501646725348, -16], [-10, 2, 0], [-8342650, -4278822, -4063828],
            [-3658646788, 9223372036854775806, 9223372033196129018],
            [-9223372036854775807, 3804260017, 2836792287], [-4, 312460408163019, 312460408163015],
            [1275897338142, 4881753055914, 1275897338142],
            [-13, 3481142685384786162, 3481142685384786149],
            [-144727830, -8912155573027, -144727830], [7, 872043591989, 7],
            [-1660326989, -11374308, -11052329], [-48201694560, 5701637482, 3113042778],
            [7, -60923511, -60923504], [2849400036, -4611686018427387904, -4611686015577987868],
            [88, -212453093281, -212453093193], [-9, -9, 0], [368096280865, -16, -15],
            [-2, -1343212367118336, -2], [-3, -1310555583283200, -3],
            [-1, 1087511489150976, 1087511489150975], [6767009598, -4294967295, -1822924992],
            [12407501211, -6302157, -1445922], [33450532594, 45013037559, 33450532594],
            [-3846803849827, -1535953319493632, -3846803849827],
            [3923387595819, -1611354636811662, -1607431249215843], [11102584172, -1, 0],
            [15135519, -1, 0], [4294967296, -2, 0], [1461220254, -1, 0], [2, 8557459, 2],
            [915143602, 2, 0], [4294967297, 9223372036854775806, 4294967297], [-12, 107, 95],
            [-2, -395514, -2], [322310632527, 780083360768581, 322310632527],
            [-41764202519451156, -4294967295, -672413286], [-1224342704750592, 80035004, 52124772],
            [-1389430510190592, -4112739684, -3099046452], [2037942303, 61810594, 60003295],
            [-58291520, 4611686018427387903, 4611686018369096383], [-1, 52483347, 52483346],
            [-587761611, -1, 0], [104286639658, 435749808533, 104286639658],
            [2, -8875010012989156440, -8875010012989156438], [-40123234293, 303, 255],
            [-848621510469, -20226660719283, -848621510469], [9223372036854775807, 5, 2],
            [-2021812904919040, 1065091759865856, 108370614812672], [7161536018, -11, -1],
            [-1044941920796672, -2006857828794368, -1044941920796672],
            [713723072, 8549233215805853819, 713723072],
            [1900549947570965, -1294170283048960, -687790618526955],
            [-1, 2714089720231, 2714089720230], [-4611686018427387904, -17545480, -3882984],
            [5847939, -48416680, -42568741], [-10, -4294967295, -10], [-13757917, -7, -5],
            [-22751944196498, 3985103638814045, 3962351694617547],
            [-124757752594433742, -1302846116986880, -987371480680142],
            [-19165792, 4406967773, 4387801981], [276814937194496, -4294967297, -64451],
            [1859361539007420, -25988122070130776, -24128760531123356], [0, -40124577, 0],
            [-55861983681, -16, -1],
        ];
        for row in rows.span() {
            let [a, b, e] = *row;
            assert!(Real::rem(fx(a), fx(b)) == fx(e), "rem {:?}", row.span());
            assert!(Real::rem(fx(a), fx(b)) == fx(a) % fx(b), "rem op {:?}", row.span());
        }
    }

    #[test]
    fn test_div_floors_negative_inexact() {
        // -1/3: floor -1431655766 (truncation would give -1431655765), whatever the signs.
        let three = fx(0x300000000);
        assert!(Real::div(types::NEG_ONE, three) == fx(-1431655766));
        assert!(Real::div(types::ONE, -three) == fx(-1431655766));
        assert!(Real::div(types::ONE, three) == fx(1431655765));
        assert!(Real::div(types::NEG_ONE, -three) == fx(1431655765));
        assert!(Real::div(fx(-0x700000000), types::TWO) == fx(-0x380000000));
    }

    #[test]
    fn test_rem_is_floored_modulo() {
        let (seven, three) = (fx(0x700000000), fx(0x300000000));
        assert!(Real::rem(seven, three) == types::ONE);
        assert!(Real::rem(-seven, three) == types::TWO);
        assert!(Real::rem(seven, -three) == -types::TWO);
        assert!(Real::rem(-seven, -three) == types::NEG_ONE);
        assert!(Real::rem(fx(-5), types::ONE) == fx(4294967291));
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_div_by_zero_panics() {
        let _ = Real::div(black_box(types::ONE), types::ZERO);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_overflow_panics() {
        let _ = Real::div(black_box(types::MAX), types::HALF);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_min_by_neg_one_panics() {
        let _ = Real::div(black_box(types::MIN), types::NEG_ONE);
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_rem_by_zero_panics() {
        let _ = Real::rem(black_box(types::ONE), types::ZERO);
    }

    #[test]
    fn test_div_rem_on_integers() {
        assert!(Real::div(7_i64, 2) == 3 && Real::rem(7_i64, 2) == 1);
        assert!(Real::div(-7_i64, 2) == -4 && Real::rem(-7_i64, 2) == 1);
        assert!(Real::div(7_i64, -2) == -4 && Real::rem(7_i64, -2) == -1);
        assert!(Real::div(-7_i64, -2) == 3 && Real::rem(-7_i64, -2) == -1);
        assert!(Real::div(-6_i64, 3) == -2 && Real::rem(-6_i64, 3) == 0);
        assert!(Real::div(-0x8000000000000000_i64, 1) == -0x8000000000000000);
        assert!(Real::rem(-0x8000000000000000_i64, -1) == 0);
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_div_on_integers_by_zero_panics() {
        let _ = Real::div(black_box(1_i64), 0);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_div_on_integers_overflow_panics() {
        let _ = Real::div(black_box(-0x8000000000000000_i64), -1);
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_rem_on_integers_by_zero_panics() {
        let _ = Real::rem(black_box(1_i64), 0);
    }

    // --- gas benchmarks: generic code over `Real` is zero-cost
    // ---------------------------------------

    #[test]
    #[inline(never)]
    fn bench_real_dot3__baseline() {
        let _a = black_box(V3 { x: fx(0x180000000), y: fx(-0x240000000), z: fx(0x3c0000000) });
        let _b = black_box(V3 { x: fx(-0x480000000), y: fx(0x40000000), z: fx(0x200000000) });
        let e = black_box(fx(0x30000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_dot3__generic() {
        let a = black_box(V3 { x: fx(0x180000000), y: fx(-0x240000000), z: fx(0x3c0000000) });
        let b = black_box(V3 { x: fx(-0x480000000), y: fx(0x40000000), z: fx(0x200000000) });
        let e = black_box(fx(0x30000000));
        assert!(a.dot(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_dot3__generic_wide() {
        let a = black_box(V3 { x: fx(0x180000000), y: fx(-0x240000000), z: fx(0x3c0000000) });
        let b = black_box(V3 { x: fx(-0x480000000), y: fx(0x40000000), z: fx(0x200000000) });
        let e = black_box(fx(0x30000000));
        assert!(a.dot_wide(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_dot3__direct() {
        let a = black_box(V3 { x: fx(0x180000000), y: fx(-0x240000000), z: fx(0x3c0000000) });
        let b = black_box(V3 { x: fx(-0x480000000), y: fx(0x40000000), z: fx(0x200000000) });
        let e = black_box(fx(0x30000000));
        assert!(crate::fixed::fused::sum_prod3(a.x, b.x, a.y, b.y, a.z, b.z) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_cross3__baseline() {
        let _a = black_box(V3 { x: fx(0x180000000), y: fx(-0x240000000), z: fx(0x3c0000000) });
        let _b = black_box(V3 { x: fx(-0x480000000), y: fx(0x40000000), z: fx(0x200000000) });
        let e = black_box(fx(-0x570000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_cross3__generic() {
        let a = black_box(V3 { x: fx(0x180000000), y: fx(-0x240000000), z: fx(0x3c0000000) });
        let b = black_box(V3 { x: fx(-0x480000000), y: fx(0x40000000), z: fx(0x200000000) });
        let e = black_box(fx(-0x570000000));
        let c = a.cross(b);
        assert!(c.x == e);
        assert!(c.y == fx(-0x13e0000000));
        assert!(c.z == fx(-0x9c0000000));
    }

    #[test]
    #[inline(never)]
    fn bench_real_norm3__baseline() {
        let _a = black_box(V3 { x: fx(0x100000000), y: fx(-0x200000000), z: fx(0x200000000) });
        let e = black_box(fx(0x300000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_norm3__generic() {
        let a = black_box(V3 { x: fx(0x100000000), y: fx(-0x200000000), z: fx(0x200000000) });
        let e = black_box(fx(0x300000000));
        assert!(a.norm() == e);
    }
    #[test]
    #[inline(never)]
    fn bench_real_div__baseline() {
        let _a = black_box(fx(-0x100000000));
        let _b = black_box(fx(0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_div__fixed() {
        let a = black_box(fx(-0x100000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(Real::div(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_div__alt_operator() {
        let a = black_box(fx(-0x100000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(a / b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_rem__baseline() {
        let _a = black_box(fx(-0x700000000));
        let _b = black_box(fx(0x300000000));
        let e = black_box(fx(0x200000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_rem__fixed() {
        let a = black_box(fx(-0x700000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(0x200000000));
        assert!(Real::rem(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_real_rem__alt_operator() {
        let a = black_box(fx(-0x700000000));
        let b = black_box(fx(0x300000000));
        let e = black_box(fx(0x200000000));
        assert!(a % b == e);
    }
}
