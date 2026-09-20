//! Scalar helpers of `Fixed`: sign, ordering, rounding to integers, reciprocal and square roots.
//!
//! Free functions; the method syntax (`x.abs()`) comes from `simba::scalar::Real`.

use super::kernels;
use super::types::Fixed;

/// `|a|`. Panics with `errors::OVERFLOW` for `MIN`. Upstream: `f64::abs`.
#[inline(always)]
pub fn abs(a: Fixed) -> Fixed {
    Fixed { raw: kernels::abs(a.raw) }
}

/// `-1`, `0` or `1` according to the sign of `a`. Upstream: `f64::signum`, except that zero maps
/// to zero (there is no signed zero here).
#[inline(always)]
pub fn signum(a: Fixed) -> Fixed {
    Fixed { raw: kernels::signum(a.raw) }
}

/// `a < 0`. Upstream: `f64::is_sign_negative` (without signed zero).
#[inline(always)]
pub fn is_negative(a: Fixed) -> bool {
    kernels::is_negative(a.raw)
}

/// `a > 0` (strictly). Upstream: `f64::is_sign_positive`, except for zero.
#[inline(always)]
pub fn is_positive(a: Fixed) -> bool {
    kernels::is_positive(a.raw)
}

/// The smaller of two values. Upstream: `f64::min`.
#[inline(always)]
pub fn min(a: Fixed, b: Fixed) -> Fixed {
    if a.raw < b.raw {
        a
    } else {
        b
    }
}

/// The larger of two values. Upstream: `f64::max`.
#[inline(always)]
pub fn max(a: Fixed, b: Fixed) -> Fixed {
    if a.raw > b.raw {
        a
    } else {
        b
    }
}

/// `x` restricted to `[lo, hi]`. Like upstream `simba`/`nalgebra::clamp`, `lo <= hi` is NOT
/// checked: `lo` wins when `x <= lo`.
#[inline(always)]
pub fn clamp(x: Fixed, lo: Fixed, hi: Fixed) -> Fixed {
    if x.raw > lo.raw {
        if x.raw < hi.raw {
            x
        } else {
            hi
        }
    } else {
        lo
    }
}

/// Largest integer `<= a`. Branch-free, cannot fail. Upstream: `f64::floor`.
#[inline(always)]
pub fn floor(a: Fixed) -> Fixed {
    Fixed { raw: kernels::floor(a.raw) }
}

/// Smallest integer `>= a`. Panics with `errors::OVERFLOW` when `a > 2^31 - 1`.
/// Upstream: `f64::ceil`.
#[inline(always)]
pub fn ceil(a: Fixed) -> Fixed {
    Fixed { raw: kernels::ceil(a.raw) }
}

/// Nearest integer, ties AWAY from zero (`round(-2.5) = -3`), like upstream `f64::round`.
/// Panics with `errors::OVERFLOW` when `a >= 2^31 - 1/2`.
#[inline(always)]
pub fn round(a: Fixed) -> Fixed {
    Fixed { raw: kernels::round(a.raw) }
}

/// Integer part, toward zero (`trunc(-2.5) = -2`). Cannot fail. Upstream: `f64::trunc`.
#[inline(always)]
pub fn trunc(a: Fixed) -> Fixed {
    Fixed { raw: kernels::trunc(a.raw) }
}

/// Fractional part `a - trunc(a)`, with the sign of `a` (`fract(-2.25) = -0.25`). Cannot fail.
/// Upstream: `f64::fract`. For the non-negative variant `a - floor(a)` use `a % ONE`.
#[inline(always)]
pub fn fract(a: Fixed) -> Fixed {
    Fixed { raw: kernels::fract(a.raw) }
}

/// `floor(1 / a)` on the exact reciprocal. Panics with `errors::DIVISION_BY_ZERO`, and with
/// `errors::OVERFLOW` when `|a| <= 2^-31` (except `a = -2^-31`). Cheaper than `ONE / a`.
/// Upstream: `f64::recip`.
#[inline(always)]
pub fn recip(a: Fixed) -> Fixed {
    Fixed { raw: kernels::recip(a.raw) }
}

/// `floor(sqrt(a))` on the exact square root (corelib integer square root: exactly floored).
/// Panics with `errors::SQRT_OF_NEGATIVE`. Upstream: `f64::sqrt` (which returns NaN instead).
#[inline(always)]
pub fn sqrt(a: Fixed) -> Fixed {
    Fixed { raw: kernels::sqrt(a.raw) }
}

/// `floor(1 / sqrt(a))` on the exact value (a single rounding, unlike `recip(sqrt(a))`).
/// Cannot overflow (at most 2^16). Panics with `errors::SQRT_OF_NEGATIVE` /
/// `errors::DIVISION_BY_ZERO`. No direct upstream equivalent (`1.0 / x.sqrt()`).
#[inline(always)]
pub fn inv_sqrt(a: Fixed) -> Fixed {
    Fixed { raw: kernels::inv_sqrt(a.raw) }
}

/// `|a - b| <= ulps` raw units, on the exact difference (cannot overflow). This is the
/// approximate equality of the project: tolerances are counted in ulp (2^-32), not as float
/// epsilons. Upstream: `approx::AbsDiffEq::abs_diff_eq`.
#[inline(always)]
pub fn abs_diff_eq(a: Fixed, b: Fixed, ulps: u64) -> bool {
    kernels::abs_diff_le(a.raw, b.raw, ulps)
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::types::{EPSILON, HALF, MAX, MIN, NEG_ONE, ONE, SQRT_2, TWO, ZERO};
    use super::{
        abs, abs_diff_eq, ceil, clamp, floor, fract, inv_sqrt, is_negative, is_positive, max, min,
        recip, round, signum, sqrt, trunc,
    };

    // 2.5, -2.5, 2.25, -2.25, 2.75, -2.75 and the integers around them.
    const P2_5: i64 = 0x280000000;
    const P2_25: i64 = 0x240000000;
    const P2_75: i64 = 0x2c0000000;
    const P2: i64 = 0x200000000;
    const P3: i64 = 0x300000000;
    /// Largest integer value, 2^31 - 1.
    const MAX_INT: i64 = 0x7fffffff00000000;

    // --- abs / signum / sign tests -------------------------------------------------------------

    #[test]
    fn test_abs_values() {
        assert!(abs(fx(P2_5)) == fx(P2_5));
        assert!(abs(fx(-P2_5)) == fx(P2_5));
        assert!(abs(ZERO) == ZERO);
        assert!(abs(-EPSILON) == EPSILON);
        assert!(abs(MAX) == MAX);
        assert!(abs(MIN + EPSILON) == MAX);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_abs_min_panics() {
        abs(black_box(MIN));
    }

    #[test]
    fn test_signum_values() {
        assert!(signum(fx(P2_5)) == ONE);
        assert!(signum(fx(-P2_5)) == NEG_ONE);
        assert!(signum(ZERO) == ZERO);
        assert!(signum(EPSILON) == ONE);
        assert!(signum(-EPSILON) == NEG_ONE);
        assert!(signum(MAX) == ONE);
        assert!(signum(MIN) == NEG_ONE);
    }

    #[test]
    fn test_is_negative_is_positive_values() {
        assert!(is_negative(-EPSILON));
        assert!(is_negative(MIN));
        assert!(!is_negative(ZERO));
        assert!(!is_negative(EPSILON));
        assert!(is_positive(EPSILON));
        assert!(is_positive(MAX));
        assert!(!is_positive(ZERO));
        assert!(!is_positive(-EPSILON));
    }

    // --- min / max / clamp ---------------------------------------------------------------------

    #[test]
    fn test_min_max_sign_combinations() {
        assert!(min(ONE, TWO) == ONE);
        assert!(min(TWO, ONE) == ONE);
        assert!(min(NEG_ONE, ONE) == NEG_ONE);
        assert!(min(NEG_ONE, -TWO) == -TWO);
        assert!(min(ONE, ONE) == ONE);
        assert!(min(MIN, MAX) == MIN);
        assert!(max(ONE, TWO) == TWO);
        assert!(max(TWO, ONE) == TWO);
        assert!(max(NEG_ONE, ONE) == ONE);
        assert!(max(NEG_ONE, -TWO) == NEG_ONE);
        assert!(max(ONE, ONE) == ONE);
        assert!(max(MIN, MAX) == MAX);
    }

    #[test]
    fn test_clamp_values() {
        assert!(clamp(HALF, ZERO, ONE) == HALF);
        assert!(clamp(-HALF, ZERO, ONE) == ZERO);
        assert!(clamp(TWO, ZERO, ONE) == ONE);
        assert!(clamp(ZERO, ZERO, ONE) == ZERO);
        assert!(clamp(ONE, ZERO, ONE) == ONE);
        assert!(clamp(MIN, NEG_ONE, ONE) == NEG_ONE);
        assert!(clamp(MAX, NEG_ONE, ONE) == ONE);
        assert!(clamp(ONE, MIN, MAX) == ONE);
    }

    #[test]
    fn test_clamp_inverted_bounds_follow_upstream() {
        // lo > hi is not checked (upstream `na::clamp`): `lo` wins below, `hi` above.
        assert!(clamp(ZERO, ONE, NEG_ONE) == ONE);
        assert!(clamp(TWO, ONE, NEG_ONE) == NEG_ONE);
    }

    // --- floor / ceil / round / trunc / fract ----------------------------------------------------

    #[test]
    fn test_floor_values() {
        assert!(floor(fx(P2_5)) == fx(P2));
        assert!(floor(fx(-P2_5)) == fx(-P3));
        assert!(floor(fx(P3)) == fx(P3));
        assert!(floor(fx(-P3)) == fx(-P3));
        assert!(floor(EPSILON) == ZERO);
        assert!(floor(-EPSILON) == NEG_ONE);
        assert!(floor(ZERO) == ZERO);
        assert!(floor(MAX) == fx(MAX_INT));
        assert!(floor(MIN) == MIN);
        assert!(floor(MIN + EPSILON) == MIN);
    }

    #[test]
    fn test_ceil_values() {
        assert!(ceil(fx(P2_5)) == fx(P3));
        assert!(ceil(fx(-P2_5)) == fx(-P2));
        assert!(ceil(fx(P3)) == fx(P3));
        assert!(ceil(fx(-P3)) == fx(-P3));
        assert!(ceil(EPSILON) == ONE);
        assert!(ceil(-EPSILON) == ZERO);
        assert!(ceil(ZERO) == ZERO);
        assert!(ceil(fx(MAX_INT)) == fx(MAX_INT));
        assert!(ceil(MIN) == MIN);
        assert!(ceil(MIN + EPSILON) == fx(-MAX_INT));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_ceil_max_panics() {
        ceil(black_box(MAX));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_ceil_above_max_int_panics() {
        ceil(black_box(fx(MAX_INT + 1)));
    }

    #[test]
    fn test_round_ties_away_from_zero() {
        assert!(round(fx(P2_5)) == fx(P3));
        assert!(round(fx(-P2_5)) == fx(-P3));
        assert!(round(HALF) == ONE);
        assert!(round(-HALF) == NEG_ONE);
        assert!(round(fx(0x180000000)) == TWO);
        assert!(round(fx(-0x180000000)) == -TWO);
    }

    #[test]
    fn test_round_values() {
        assert!(round(fx(P2_25)) == fx(P2));
        assert!(round(fx(-P2_25)) == fx(-P2));
        assert!(round(fx(P2_75)) == fx(P3));
        assert!(round(fx(-P2_75)) == fx(-P3));
        assert!(round(HALF - EPSILON) == ZERO);
        assert!(round(-HALF + EPSILON) == ZERO);
        assert!(round(EPSILON) == ZERO);
        assert!(round(-EPSILON) == ZERO);
        assert!(round(ZERO) == ZERO);
        assert!(round(fx(P3)) == fx(P3));
        assert!(round(MIN) == MIN);
        assert!(round(MIN + HALF) == MIN);
        assert!(round(MIN + HALF - EPSILON) == MIN);
        assert!(round(MAX - HALF) == fx(MAX_INT));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_round_max_panics() {
        round(black_box(MAX));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_round_max_int_plus_half_panics() {
        round(black_box(MAX - HALF + EPSILON));
    }

    #[test]
    fn test_trunc_values() {
        assert!(trunc(fx(P2_5)) == fx(P2));
        assert!(trunc(fx(-P2_5)) == fx(-P2));
        assert!(trunc(fx(P3)) == fx(P3));
        assert!(trunc(fx(-P3)) == fx(-P3));
        assert!(trunc(EPSILON) == ZERO);
        assert!(trunc(-EPSILON) == ZERO);
        assert!(trunc(ZERO) == ZERO);
        assert!(trunc(MAX) == fx(MAX_INT));
        assert!(trunc(MIN) == MIN);
        assert!(trunc(MIN + EPSILON) == fx(-MAX_INT));
    }

    #[test]
    fn test_fract_has_sign_of_input() {
        assert!(fract(fx(P2_5)) == HALF);
        assert!(fract(fx(-P2_5)) == -HALF);
        assert!(fract(fx(-P2_25)) == fx(-0x40000000));
        assert!(fract(fx(P3)) == ZERO);
        assert!(fract(fx(-P3)) == ZERO);
        assert!(fract(EPSILON) == EPSILON);
        assert!(fract(-EPSILON) == -EPSILON);
        assert!(fract(ZERO) == ZERO);
        assert!(fract(MAX) == fx(0xffffffff));
        assert!(fract(MIN) == ZERO);
        assert!(fract(MIN + EPSILON) == fx(-0xffffffff));
    }

    #[test]
    fn test_trunc_plus_fract_is_identity() {
        let x = fx(-0x2c0000001);
        assert!(trunc(x) + fract(x) == x);
        let x = fx(0x7fffffff12345678);
        assert!(trunc(x) + fract(x) == x);
    }

    // --- recip -----------------------------------------------------------------------------------

    #[test]
    fn test_recip_values() {
        assert!(recip(TWO) == HALF);
        assert!(recip(-TWO) == -HALF);
        assert!(recip(HALF) == TWO);
        assert!(recip(ONE) == ONE);
        assert!(recip(NEG_ONE) == NEG_ONE);
        assert!(recip(MAX) == fx(2));
        assert!(recip(MIN) == fx(-2));
    }

    #[test]
    fn test_recip_rounds_toward_negative_infinity() {
        assert!(recip(fx(P3)) == fx(1431655765));
        assert!(recip(fx(-P3)) == fx(-1431655766));
        assert!(recip(fx(3)) == fx(6148914691236517205));
        assert!(recip(fx(-3)) == fx(-6148914691236517206));
    }

    #[test]
    fn test_recip_smallest_inputs() {
        // 1 / 2^-30 = 2^30 fits; 1 / -2^-31 = -2^31 = MIN fits; 1 / 2^-31 = 2^31 does not.
        assert!(recip(fx(4)) == fx(0x4000000000000000));
        assert!(recip(fx(-2)) == MIN);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_recip_two_ulp_panics() {
        recip(black_box(fx(2)));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_recip_epsilon_panics() {
        recip(black_box(EPSILON));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_recip_neg_epsilon_panics() {
        recip(black_box(-EPSILON));
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_recip_zero_panics() {
        recip(black_box(ZERO));
    }

    // --- sqrt / inv_sqrt
    // ---------------------------------------------------------------------------

    #[test]
    fn test_sqrt_values() {
        assert!(sqrt(ZERO) == ZERO);
        assert!(sqrt(ONE) == ONE);
        assert!(sqrt(fx(0x400000000)) == TWO);
        assert!(sqrt(fx(0x40000000)) == HALF);
        assert!(sqrt(TWO) == SQRT_2);
        assert!(sqrt(EPSILON) == fx(65536));
        assert!(sqrt(fx(3)) == fx(113511));
        assert!(sqrt(MAX) == fx(199032864766430));
    }

    #[test]
    #[should_panic(expected: 'simba: sqrt of negative')]
    fn test_sqrt_negative_panics() {
        sqrt(black_box(-EPSILON));
    }

    #[test]
    #[should_panic(expected: 'simba: sqrt of negative')]
    fn test_sqrt_min_panics() {
        sqrt(black_box(MIN));
    }

    #[test]
    fn test_inv_sqrt_values() {
        assert!(inv_sqrt(ONE) == ONE);
        assert!(inv_sqrt(fx(0x400000000)) == HALF);
        assert!(inv_sqrt(fx(0x40000000)) == TWO);
        assert!(inv_sqrt(TWO) == fx(3037000499));
        assert!(inv_sqrt(EPSILON) == fx(0x1000000000000));
        assert!(inv_sqrt(fx(2)) == fx(199032864766430));
        assert!(inv_sqrt(fx(3)) == fx(162509653574040));
        assert!(inv_sqrt(MAX) == fx(92681));
    }

    #[test]
    fn test_inv_sqrt_is_exactly_floored() {
        // x = 4294 ulp (about 1e-6): 1/sqrt(x) = 1000.11.. The composition recip(sqrt(x)) rounds
        // twice and amplifies the rounding of the small root: it is off by 620,905 ulp (1.4e-4).
        assert!(inv_sqrt(fx(4294)) == fx(4295451025709));
        assert!(recip(sqrt(fx(4294))) == fx(4295451646614));
    }

    #[test]
    #[should_panic(expected: 'simba: division by zero')]
    fn test_inv_sqrt_zero_panics() {
        inv_sqrt(black_box(ZERO));
    }

    #[test]
    #[should_panic(expected: 'simba: sqrt of negative')]
    fn test_inv_sqrt_negative_panics() {
        inv_sqrt(black_box(NEG_ONE));
    }

    // --- abs_diff_eq
    // -------------------------------------------------------------------------------

    #[test]
    fn test_abs_diff_eq_tolerance_in_ulps() {
        assert!(abs_diff_eq(ONE, ONE, 0));
        assert!(abs_diff_eq(ONE, ONE + EPSILON, 1));
        assert!(abs_diff_eq(ONE + EPSILON, ONE, 1));
        assert!(!abs_diff_eq(ONE, ONE + EPSILON, 0));
        assert!(!abs_diff_eq(ONE, fx(0x100000003), 2));
        assert!(abs_diff_eq(fx(-0x100000003), NEG_ONE, 3));
        assert!(abs_diff_eq(NEG_ONE, ONE, 0x200000000));
        assert!(!abs_diff_eq(NEG_ONE, ONE, 0x1ffffffff));
    }

    #[test]
    fn test_abs_diff_eq_full_range_does_not_overflow() {
        assert!(abs_diff_eq(MIN, MAX, 0xffffffffffffffff));
        assert!(!abs_diff_eq(MIN, MAX, 0xfffffffffffffffe));
        assert!(!abs_diff_eq(MAX, MIN, 0xfffffffffffffffe));
        assert!(abs_diff_eq(MAX, MIN, 0xffffffffffffffff));
    }

    // --- gas benchmarks (net = raw - baseline of the group) --------------------------------------
    // Losing candidates: `bench_<op>__alt_*` in `fixed::kernels::alternatives`.

    #[test]
    #[inline(never)]
    fn bench_abs__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x380000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x380000000));
        assert!(abs(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(0x240000000));
        assert!(abs(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x100000000));
        assert!(signum(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x100000000));
        assert!(signum(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_signum__fixed_z() {
        let a = black_box(fx(0x0));
        let e = black_box(fx(0x0));
        assert!(signum(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(false);
        assert!(is_negative(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_negative__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(true);
        assert!(is_negative(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_positive__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_positive__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(true);
        assert!(is_positive(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_is_positive__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(false);
        assert!(is_positive(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_min__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x240000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_min__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x240000000));
        assert!(min(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_min__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(fx(-0x240000000));
        assert!(min(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_max__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x380000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_max__fixed_pn() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let e = black_box(fx(0x380000000));
        assert!(max(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_max__fixed_np() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(0x380000000));
        let e = black_box(fx(0x380000000));
        assert!(max(a, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_clamp__baseline() {
        let _a = black_box(fx(0x180000000));
        let _b = black_box(fx(-0x240000000));
        let _c = black_box(fx(0x380000000));
        let e = black_box(fx(0x180000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_clamp__fixed_inside() {
        let a = black_box(fx(0x180000000));
        let b = black_box(fx(-0x240000000));
        let c = black_box(fx(0x380000000));
        let e = black_box(fx(0x180000000));
        assert!(clamp(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_clamp__fixed_below() {
        let a = black_box(fx(-0x240000000));
        let b = black_box(fx(-0xc0000000));
        let c = black_box(fx(0x380000000));
        let e = black_box(fx(-0xc0000000));
        assert!(clamp(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_clamp__fixed_above() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(-0x240000000));
        let c = black_box(fx(0x180000000));
        let e = black_box(fx(0x180000000));
        assert!(clamp(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x300000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x300000000));
        assert!(floor(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_floor__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x300000000));
        assert!(floor(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x400000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x400000000));
        assert!(ceil(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ceil__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x200000000));
        assert!(ceil(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_round__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x400000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_round__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x400000000));
        assert!(round(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_round__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x200000000));
        assert!(round(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_trunc__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x300000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_trunc__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x300000000));
        assert!(trunc(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_trunc__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x200000000));
        assert!(trunc(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_fract__baseline() {
        let _a = black_box(fx(0x380000000));
        let e = black_box(fx(0x80000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_fract__fixed_p() {
        let a = black_box(fx(0x380000000));
        let e = black_box(fx(0x80000000));
        assert!(fract(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_fract__fixed_n() {
        let a = black_box(fx(-0x240000000));
        let e = black_box(fx(-0x40000000));
        assert!(fract(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_recip__baseline() {
        let _a = black_box(fx(0x300000000));
        let e = black_box(fx(0x55555555));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_recip__fixed_p() {
        let a = black_box(fx(0x300000000));
        let e = black_box(fx(0x55555555));
        assert!(recip(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_recip__fixed_n() {
        let a = black_box(fx(-0x300000000));
        let e = black_box(fx(-0x55555556));
        assert!(recip(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt__baseline() {
        let _a = black_box(fx(0x200000000));
        let e = black_box(fx(0x16a09e667));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sqrt__fixed_p() {
        let a = black_box(fx(0x200000000));
        let e = black_box(fx(0x16a09e667));
        assert!(sqrt(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_inv_sqrt__baseline() {
        let _a = black_box(fx(0x200000000));
        let e = black_box(fx(0xb504f333));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_inv_sqrt__fixed_p() {
        let a = black_box(fx(0x200000000));
        let e = black_box(fx(0xb504f333));
        assert!(inv_sqrt(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_diff_eq__baseline() {
        let _a = black_box(fx(0x380000000));
        let _b = black_box(fx(0x380000003));
        let _c = black_box(0x3_u64);
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_diff_eq__fixed_lt() {
        let a = black_box(fx(0x380000000));
        let b = black_box(fx(0x380000003));
        let c = black_box(0x3_u64);
        let e = black_box(true);
        assert!(abs_diff_eq(a, b, c) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_abs_diff_eq__fixed_gt() {
        let a = black_box(fx(0x380000004));
        let b = black_box(fx(0x380000000));
        let c = black_box(0x3_u64);
        let e = black_box(false);
        assert!(abs_diff_eq(a, b, c) == e);
    }
}
