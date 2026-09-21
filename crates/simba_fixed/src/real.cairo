//! `simba::scalar::Real` for glam.cairo's `fixed::Fixed`.

use fixed::Fixed as Glam;
use simba::fixed::wide::Wide;
use simba::fixed::{convert as sconvert, fused, math, types};
use simba::scalar::Real;
use crate::convert::{from_simba, to_simba};

// The constants are simba's (floored), NOT glam.cairo's (rounded to nearest): see the module doc
// of `crate` and `crate::conformance`. Written as raw literals because a `const` initialiser
// cannot read a field of another `const`; `test_real_constants_are_simbas` pins every one of them.

/// π, floored — `simba::fixed::types::PI`, one ulp below `fixed::PI`.
const PI: Glam = Glam { raw: 13493037704 };
/// 2π, floored — equal to `fixed::TAU`.
const TAU: Glam = Glam { raw: 26986075409 };
/// π/2, floored — equal to `fixed::FRAC_PI_2`.
const FRAC_PI_2: Glam = Glam { raw: 6746518852 };
/// π/3, floored — one ulp below `fixed::FRAC_PI_3`.
const FRAC_PI_3: Glam = Glam { raw: 4497679234 };
/// π/4, floored — equal to `fixed::FRAC_PI_4`.
const FRAC_PI_4: Glam = Glam { raw: 3373259426 };
/// π/6, floored — equal to `fixed::FRAC_PI_6`.
const FRAC_PI_6: Glam = Glam { raw: 2248839617 };
/// 1/π, floored — equal to `fixed::FRAC_1_PI`.
const FRAC_1_PI: Glam = Glam { raw: 1367130551 };
/// e, floored — one ulp below `fixed::E`.
const E: Glam = Glam { raw: 11674931554 };
/// ln 2, floored — one ulp below `fixed::LN_2`.
const LN_2: Glam = Glam { raw: 2977044471 };
/// ln 10, floored — one ulp below `fixed::LN_10`.
const LN_10: Glam = Glam { raw: 9889527670 };
/// √2, floored — one ulp below `fixed::SQRT_2`.
const SQRT_2: Glam = Glam { raw: 6074000999 };
/// 1/√2, floored — one ulp below `fixed::FRAC_1_SQRT_2`.
const FRAC_1_SQRT_2: Glam = Glam { raw: 3037000499 };

/// `Real` for glam.cairo's Q32.32 scalar: every method relabels its arguments as
/// `simba::fixed::Fixed` (free, `crate::convert`) and calls simba's own kernel, so
/// `nalgebra::Vector3<fixed::Fixed>` and `nalgebra::Vector3<simba::fixed::Fixed>` compute
/// bit-identical results and share simba's numeric contract: floor rounding once per output
/// scalar, `'simba: ...'` panic messages on overflow / division by zero / domain errors.
///
/// Nothing is reimplemented here: where `fixed` has a native equivalent (`FixedTrait::sqrt`,
/// `fixed::wide::dot3`, ...) it is left alone and the equivalence is asserted instead, in
/// `crate::conformance`, which is also where the handful of documented differences live.
pub impl FixedReal of Real<Glam> {
    type Wide = Wide;

    const ZERO: Glam = Glam { raw: 0 };
    const ONE: Glam = Glam { raw: 0x100000000 };
    const NEG_ONE: Glam = Glam { raw: -0x100000000 };
    const TWO: Glam = Glam { raw: 0x200000000 };
    const HALF: Glam = Glam { raw: 0x80000000 };
    const EPSILON: Glam = Glam { raw: 1 };
    const MIN: Glam = Glam { raw: -0x8000000000000000 };
    const MAX: Glam = Glam { raw: 0x7fffffffffffffff };
    const PI: Glam = PI;
    const TAU: Glam = TAU;
    const FRAC_PI_2: Glam = FRAC_PI_2;
    const FRAC_PI_3: Glam = FRAC_PI_3;
    const FRAC_PI_4: Glam = FRAC_PI_4;
    const FRAC_PI_6: Glam = FRAC_PI_6;
    const FRAC_1_PI: Glam = FRAC_1_PI;
    const E: Glam = E;
    const LN_2: Glam = LN_2;
    const LN_10: Glam = LN_10;
    const SQRT_2: Glam = SQRT_2;
    const FRAC_1_SQRT_2: Glam = FRAC_1_SQRT_2;

    #[inline(always)]
    fn from_int(v: i32) -> Glam {
        from_simba(sconvert::from_int(v))
    }
    #[inline(always)]
    fn from_ratio(num: i64, den: i64) -> Glam {
        from_simba(sconvert::from_ratio(num, den))
    }
    #[inline(always)]
    fn to_int(self: Glam) -> i32 {
        sconvert::to_int(to_simba(self))
    }

    #[inline(always)]
    fn abs(self: Glam) -> Glam {
        from_simba(math::abs(to_simba(self)))
    }
    #[inline(always)]
    fn signum(self: Glam) -> Glam {
        from_simba(math::signum(to_simba(self)))
    }
    #[inline(always)]
    fn is_negative(self: Glam) -> bool {
        math::is_negative(to_simba(self))
    }
    #[inline(always)]
    fn is_positive(self: Glam) -> bool {
        math::is_positive(to_simba(self))
    }
    #[inline(always)]
    fn min(self: Glam, other: Glam) -> Glam {
        from_simba(math::min(to_simba(self), to_simba(other)))
    }
    #[inline(always)]
    fn max(self: Glam, other: Glam) -> Glam {
        from_simba(math::max(to_simba(self), to_simba(other)))
    }
    #[inline(always)]
    fn clamp(self: Glam, lo: Glam, hi: Glam) -> Glam {
        from_simba(math::clamp(to_simba(self), to_simba(lo), to_simba(hi)))
    }
    #[inline(always)]
    fn floor(self: Glam) -> Glam {
        from_simba(math::floor(to_simba(self)))
    }
    #[inline(always)]
    fn ceil(self: Glam) -> Glam {
        from_simba(math::ceil(to_simba(self)))
    }
    #[inline(always)]
    fn round(self: Glam) -> Glam {
        from_simba(math::round(to_simba(self)))
    }
    #[inline(always)]
    fn trunc(self: Glam) -> Glam {
        from_simba(math::trunc(to_simba(self)))
    }
    #[inline(always)]
    fn fract(self: Glam) -> Glam {
        from_simba(math::fract(to_simba(self)))
    }
    #[inline(always)]
    fn recip(self: Glam) -> Glam {
        from_simba(math::recip(to_simba(self)))
    }
    #[inline(always)]
    fn sqrt(self: Glam) -> Glam {
        from_simba(math::sqrt(to_simba(self)))
    }
    #[inline(always)]
    fn inv_sqrt(self: Glam) -> Glam {
        from_simba(math::inv_sqrt(to_simba(self)))
    }
    #[inline(always)]
    fn abs_diff_eq(self: Glam, other: Glam, ulps: u64) -> bool {
        math::abs_diff_eq(to_simba(self), to_simba(other), ulps)
    }

    /// SIMBA's floor division (`simba::fixed::ops::FixedDiv`), not glam.cairo's truncating `/`:
    /// this is what makes nalgebra's `normalize` / `unscale` / `new_normalize`, which divide
    /// through `Real::div`, bit-identical with both scalars (`'simba: ...'` panic messages too).
    #[inline(always)]
    fn div(a: Glam, b: Glam) -> Glam {
        from_simba(to_simba(a) / to_simba(b))
    }
    /// SIMBA's floored modulo (sign of the divisor), not glam.cairo's truncated `%`.
    #[inline(always)]
    fn rem(a: Glam, b: Glam) -> Glam {
        from_simba(to_simba(a) % to_simba(b))
    }

    #[inline(always)]
    fn sqr(self: Glam) -> Glam {
        from_simba(fused::sqr(to_simba(self)))
    }
    #[inline(always)]
    fn sum_prod2(a0: Glam, b0: Glam, a1: Glam, b1: Glam) -> Glam {
        from_simba(fused::sum_prod2(to_simba(a0), to_simba(b0), to_simba(a1), to_simba(b1)))
    }
    #[inline(always)]
    fn sum_prod3(a0: Glam, b0: Glam, a1: Glam, b1: Glam, a2: Glam, b2: Glam) -> Glam {
        from_simba(
            fused::sum_prod3(
                to_simba(a0), to_simba(b0), to_simba(a1), to_simba(b1), to_simba(a2), to_simba(b2),
            ),
        )
    }
    #[inline(always)]
    fn sum_prod4(
        a0: Glam, b0: Glam, a1: Glam, b1: Glam, a2: Glam, b2: Glam, a3: Glam, b3: Glam,
    ) -> Glam {
        from_simba(
            fused::sum_prod4(
                to_simba(a0),
                to_simba(b0),
                to_simba(a1),
                to_simba(b1),
                to_simba(a2),
                to_simba(b2),
                to_simba(a3),
                to_simba(b3),
            ),
        )
    }
    #[inline(always)]
    fn diff_prod(a: Glam, b: Glam, c: Glam, d: Glam) -> Glam {
        from_simba(fused::diff_prod(to_simba(a), to_simba(b), to_simba(c), to_simba(d)))
    }
    #[inline(always)]
    fn mul_add(a: Glam, b: Glam, c: Glam) -> Glam {
        from_simba(fused::mul_add(to_simba(a), to_simba(b), to_simba(c)))
    }
    #[inline(always)]
    fn mul_sub(a: Glam, b: Glam, c: Glam) -> Glam {
        from_simba(fused::mul_sub(to_simba(a), to_simba(b), to_simba(c)))
    }
    #[inline(always)]
    fn lerp(a: Glam, b: Glam, t: Glam) -> Glam {
        from_simba(fused::lerp(to_simba(a), to_simba(b), to_simba(t)))
    }
    #[inline(always)]
    fn norm_squared2(x: Glam, y: Glam) -> Glam {
        from_simba(fused::norm_squared2(to_simba(x), to_simba(y)))
    }
    #[inline(always)]
    fn norm_squared3(x: Glam, y: Glam, z: Glam) -> Glam {
        from_simba(fused::norm_squared3(to_simba(x), to_simba(y), to_simba(z)))
    }
    #[inline(always)]
    fn norm_squared4(x: Glam, y: Glam, z: Glam, w: Glam) -> Glam {
        from_simba(fused::norm_squared4(to_simba(x), to_simba(y), to_simba(z), to_simba(w)))
    }
    #[inline(always)]
    fn norm2(x: Glam, y: Glam) -> Glam {
        from_simba(fused::norm2(to_simba(x), to_simba(y)))
    }
    #[inline(always)]
    fn norm3(x: Glam, y: Glam, z: Glam) -> Glam {
        from_simba(fused::norm3(to_simba(x), to_simba(y), to_simba(z)))
    }
    #[inline(always)]
    fn norm4(x: Glam, y: Glam, z: Glam, w: Glam) -> Glam {
        from_simba(fused::norm4(to_simba(x), to_simba(y), to_simba(z), to_simba(w)))
    }

    #[inline(always)]
    fn wide_zero() -> Wide {
        Real::<types::Fixed>::wide_zero()
    }
    #[inline(always)]
    fn wide_add_prod(w: Wide, a: Glam, b: Glam) -> Wide {
        Real::<types::Fixed>::wide_add_prod(w, to_simba(a), to_simba(b))
    }
    #[inline(always)]
    fn wide_sub_prod(w: Wide, a: Glam, b: Glam) -> Wide {
        Real::<types::Fixed>::wide_sub_prod(w, to_simba(a), to_simba(b))
    }
    #[inline(always)]
    fn wide_add(w: Wide, c: Glam) -> Wide {
        Real::<types::Fixed>::wide_add(w, to_simba(c))
    }
    #[inline(always)]
    fn wide_sub(w: Wide, c: Glam) -> Wide {
        Real::<types::Fixed>::wide_sub(w, to_simba(c))
    }
    #[inline(always)]
    fn wide_rescale(w: Wide) -> Glam {
        from_simba(Real::<types::Fixed>::wide_rescale(w))
    }
    #[inline(always)]
    fn wide_sqrt(w: Wide) -> Glam {
        from_simba(Real::<types::Fixed>::wide_sqrt(w))
    }
    #[inline(always)]
    fn wide_mul_scalar(w: Wide, s: Glam) -> Glam {
        from_simba(Real::<types::Fixed>::wide_mul_scalar(w, to_simba(s)))
    }
}
