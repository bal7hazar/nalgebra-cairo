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
}
