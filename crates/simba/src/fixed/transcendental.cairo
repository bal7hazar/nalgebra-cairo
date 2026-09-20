//! Transcendental functions of `Fixed` (DESIGN D6): trigonometry, its inverses, `exp` and `ln`.
//!
//! Free functions; the method syntax (`x.sin()`) comes from `simba::scalar::Transcendental`.
//! Angles are in radians. The implementation is the pair `kernels::poly` (generated typed Horner
//! kernels, `tools/polygen`) + `kernels::transcendental` (hand-written range folding, signs and
//! domain checks).
//!
//! # Numeric specification
//!
//! Every function is *deterministic* and *exact to the last bit forever*: the same raw input
//! gives the same raw output (AGENTS.md numeric rules). Internally the reduced argument carries
//! 61 fractional bits and the polynomial accumulators 64, so the only rounding of a result is
//! the final one, **to nearest, ties up on the magnitude** — unlike the floor rounding of the
//! algebraic kernels (`mul`, `div`, `sum_prod*`), which is why the errors below stay near half
//! an ulp.
//!
//! Errors are measured by the bit-exact Python model (`tools/fixed_model`) against mpmath at 200
//! bits, over a dense sweep of the domain (40k points), a random sweep (20k points) and the edge
//! cases; 1 ulp is `2^-32 = 2.33e-10`.
//!
//! | function | polynomial | domain | max error (ulp) |
//! |---|---|---|---:|
//! | `sin`, `cos`, `sin_cos` | degree 11 / 10 on `[0, pi/4]` | `abs(x) <= 2^26` rad | **0.56** |
//! | `sin`, `cos`, `sin_cos` | the same | `abs(x) < 2^31` rad | **2.53** |
//! | `tan` | the two above, at 2^-48 | `abs(tan x) <= MAX` | `1 + 2 tan^2` |
//! | `atan`, `atan2` | degree 25 on `[0, 1]` | all | **1.12** |
//! | `asin` | degree 15 on `[0, 1/2]` | `[-1, 1]` | **0.96** |
//! | `acos` | the same | `[-1, 1]` | **1.20** |
//! | `exp` | degree 8 on `[0, ln 2)` | `x <= 21.487` | **0.47 relative** |
//! | `ln` | degree 11 on `[0, 0.172]` | `x > 0` | **0.50** |
//!
//! ## Exact values
//!
//! `sin(0) = 0`, `cos(0) = ONE`, `tan(0) = 0`, `exp(0) = ONE`, `ln(ONE) = 0`, `asin(0) = 0`,
//! `atan(0) = 0`, `asin(HALF) = FRAC_PI_6`, `asin(ONE) = FRAC_PI_2`, `acos(ONE) = 0`,
//! `acos(0) = FRAC_PI_2`, `acos(-ONE) = PI`, `atan(ONE) = FRAC_PI_4`, `tan(FRAC_PI_4) = ONE`,
//! `sin(FRAC_PI_6) = HALF`, and `sin(x) = atan(x) = asin(x) = x` for `abs(x) <= 2^-11`: every
//! fit is written `1 + z * G(z)` with the leading 1 exact.
//!
//! `atan2` on the axes: `atan2(0, x > 0) = 0`, `atan2(0, x < 0) = PI`, `atan2(y > 0, 0) =
//! FRAC_PI_2`, `atan2(y < 0, 0) = -FRAC_PI_2`. **`atan2(0, 0) = 0`**: the angle is undefined
//! there, and zero is returned rather than a panic, like `f64::atan2`.
//!
//! ## Symmetries (exact, not approximate)
//!
//! - `sin(-x) == -sin(x)`, `cos(-x) == cos(x)`, `tan(-x) == -tan(x)`, `asin(-x) == -asin(x)`,
//!   `atan(-x) == -atan(x)` and `atan2(-y, x) == -atan2(y, x)` hold **bit for bit**: the sign is
//!   split off before the polynomial and applied to the magnitude afterwards.
//! - `asin(x) + acos(x) == FRAC_PI_2` and `acos(-x) == PI - acos(x)` hold **bit for bit** on the
//!   whole domain: the two share one kernel value per branch, and `PI` is exactly `2*FRAC_PI_2`.
//! - `sin_cos(x) == (sin(x), cos(x))` bit for bit: same reduction, same polynomial programs,
//!   with `r^2` computed once.
//! - `atan(x) == atan2(x, ONE)` bit for bit.
//! - `tan(x)` is **not** exactly `sin(x) / cos(x)`: it divides a sine and a cosine that keep 48
//!   fractional bits, which is far more accurate than dividing the two rounded Q32.32 values.
//!
//! ## Range reduction and large arguments
//!
//! `sin`, `cos`, `sin_cos` and `tan` reduce `x` modulo `pi/4` with a **2^61-scale** constant
//! (`kernels::poly::QUARTER_PI`), not with the Q32.32 `TAU`. The reduction
//! `q * QUARTER_PI + r = x * 2^29 + OFF` is exact, so the only error is the quantisation of
//! `QUARTER_PI` itself (0.13 of a unit at scale 2^61): reducing the largest representable angle,
//! `abs(x) < 2^31` rad (2.7e9 octants), drifts by at most **2.9 raw units**, and the measured
//! worst case over the whole range is 2.53 ulp, against 0.53 for physics-scale angles. A Q32.32
//! `TAU` would drift by `abs(x) / 8` ulp instead, i.e. 2.7e8 ulp at the same argument — what the
//! oracle tolerates (`tools/oracle`, `4 + abs(x) / 8`) and what this implementation beats by
//! eight orders of magnitude. No argument is ever rejected for being large.
//!
//! ## Chosen degrees
//!
//! Measured accuracy / gas trade-off (`polygen.py --report` and the `bench_*` groups below).
//! Degree 11 / 10 costs **the same** as degree 9 / 8 for `sin` / `cos` — the extra Horner step is
//! absorbed by the lazy rescale, both programs use two `div_rem`s — while halving the error, so
//! it is the default; degree 7 / 6 saves 1,410 gas but is 466 ulp off. `atan` stops at degree 25
//! (1.12 ulp): degree 29 costs 600 gas more for 0.09 ulp, degree 21 saves 2,000 gas but is
//! 6.4 ulp off. `asin` stops at degree 15, `exp` at degree 8 and `ln` at degree 11 for the same
//! reason: one more coefficient buys less than 0.3 ulp.
//!
//! ## Panics
//!
//! | function | condition | error |
//! |---|---|---|
//! | `asin`, `acos` | `abs(x) > 1` | `errors::DOMAIN` |
//! | `ln` | `x <= 0` | `errors::DOMAIN` |
//! | `tan` | `abs(tan x)` does not fit Q32.32 | `errors::OVERFLOW` |
//! | `exp` | `x > 21.487` | `errors::OVERFLOW` |
//!
//! `tan` also carries an `errors::DIVISION_BY_ZERO` for a reduced cosine of exactly zero, which
//! no `Fixed` can reach (`pi/2` is not representable at scale 2^61); it is there so the kernel
//! is total rather than relying on that argument at run time.
//!
//! `exp` **underflows to `ZERO`** below `x = -22.9` instead of panicking: a positive result
//! rounding to zero is not an error. `sin`, `cos`, `sin_cos`, `atan` and `atan2` are **total**:
//! they cannot panic for any `Fixed`.

use super::kernels::transcendental as kernels;
use super::types::Fixed;

/// `sin(x)`, `x` in radians. Total (no panic), result in `[-1, 1]`.
///
/// Octant range reduction at scale 2^61 then one degree-11 polynomial. Max error 0.56 ulp for
/// `abs(x) <= 2^26` rad, 2.53 ulp over the whole range. `sin(0) = 0` and `sin(-x) = -sin(x)`
/// exactly, and `sin(x) = x` for `abs(x) <= 2^-11`. Upstream: `f64::sin`.
#[inline(always)]
pub fn sin(x: Fixed) -> Fixed {
    Fixed { raw: kernels::sin(x.raw) }
}

/// `cos(x)`, `x` in radians. Total (no panic), result in `[-1, 1]`.
///
/// Same reduction as `sin` with the octant shifted by two, then one degree-10 polynomial.
/// Max error 0.56 ulp for `abs(x) <= 2^26` rad. `cos(0) = ONE` and `cos(-x) = cos(x)` exactly.
/// Upstream: `f64::cos`.
#[inline(always)]
pub fn cos(x: Fixed) -> Fixed {
    Fixed { raw: kernels::cos(x.raw) }
}

/// `(sin(x), cos(x))` with a shared range reduction and a shared `r^2`: about 30 % cheaper than
/// the two calls, and bit for bit equal to them. Total. Upstream: `f64::sin_cos`.
#[inline(always)]
pub fn sin_cos(x: Fixed) -> (Fixed, Fixed) {
    let (s, c) = kernels::sin_cos(x.raw);
    (Fixed { raw: s }, Fixed { raw: c })
}

/// `tan(x)`, `x` in radians.
///
/// Computed as the ratio of a sine and a cosine that keep 48 fractional bits, rounded to nearest
/// once. The error is about `1 + 2 * tan(x)^2` ulp (the pole amplifies the argument's own
/// quantisation, as for any fixed-point implementation). `tan(0) = 0` and `tan(-x) = -tan(x)`
/// exactly.
///
/// Panics with `errors::OVERFLOW` when `abs(tan x)` does not fit Q32.32 (`x` within 4.7e-10 of an
/// odd multiple of `pi/2`). Upstream: `f64::tan`.
#[inline(always)]
pub fn tan(x: Fixed) -> Fixed {
    Fixed { raw: kernels::tan(x.raw) }
}

/// `asin(x)` in `[-FRAC_PI_2, FRAC_PI_2]`.
///
/// `x * R(x^2)` (degree 15) for `abs(x) <= 1/2`, `FRAC_PI_2 - 2 asin(sqrt((1 - abs(x)) / 2))`
/// above. Max error 0.96 ulp. `asin(0) = 0`, `asin(HALF) = FRAC_PI_6`, `asin(±ONE) = ±FRAC_PI_2`
/// and `asin(-x) = -asin(x)` exactly.
///
/// Panics with `errors::DOMAIN` for `abs(x) > 1`. Upstream: `f64::asin` (which returns NaN
/// instead).
#[inline(always)]
pub fn asin(x: Fixed) -> Fixed {
    Fixed { raw: kernels::asin(x.raw) }
}

/// `acos(x)` in `[0, PI]`.
///
/// `FRAC_PI_2 - asin(x)` for `abs(x) <= 1/2`, `2 asin(sqrt((1 - abs(x)) / 2))` above — the second
/// form keeps the accuracy near `abs(x) = 1`, where `FRAC_PI_2 - asin(x)` would cancel.
/// Max error 1.20 ulp. `acos(0) = FRAC_PI_2`, `acos(ONE) = 0`, `acos(-ONE) = PI` and
/// `acos(-x) = PI - acos(x)` exactly.
///
/// Panics with `errors::DOMAIN` for `abs(x) > 1`. Upstream: `f64::acos`.
#[inline(always)]
pub fn acos(x: Fixed) -> Fixed {
    Fixed { raw: kernels::acos(x.raw) }
}

/// `atan(x)` in `(-FRAC_PI_2, FRAC_PI_2)`. Total (no panic).
///
/// Degree-25 odd polynomial on `[0, 1]`; `abs(x) <= 1` needs no division at all. Max error
/// 1.12 ulp. `atan(0) = 0`, `atan(-x) = -atan(x)` and `atan(x) = atan2(x, ONE)` exactly.
/// Upstream: `f64::atan`.
#[inline(always)]
pub fn atan(x: Fixed) -> Fixed {
    Fixed { raw: kernels::atan(x.raw) }
}

/// `exp(x) = e^x`, result in `[0, MAX]`.
///
/// `x = q * ln2 + f` at scale 2^61, then `e^f` (degree-8 polynomial on `[0, ln 2)`) scaled by
/// `2^q`, the power of two coming from a generated compare tree. The error is **relative**:
/// 0.47 ulp of the result, over the whole representable range. `exp(0) = ONE` exactly.
///
/// Returns `ZERO` when the result is below half a raw unit (`x < -22.9`, an underflow, not an
/// error) and panics with `errors::OVERFLOW` when it does not fit Q32.32 (`x > 21.487`).
/// Upstream: `f64::exp`.
#[inline(always)]
pub fn exp(x: Fixed) -> Fixed {
    Fixed { raw: kernels::exp(x.raw) }
}

/// `ln(x)`, the natural logarithm, in `[-22.18, 21.49]`.
///
/// `x = m * 2^(e - 62)` normalised by a generated tree of six typed comparisons, then
/// `(e - 32) ln2 + ln(m)` with `ln(m) = 2 atanh((m - 1) / (m + 1))` (degree-11 odd polynomial)
/// split at `sqrt 2` so the argument of `atanh` stays below 0.172. Max error 0.50 ulp, which is
/// *absolute*: `ln` of a tiny value is as accurate as `ln` of a big one. `ln(ONE) = 0` exactly,
/// and `ln(2^k) = k * LN_2` to within a rounding.
///
/// Panics with `errors::DOMAIN` for `x <= 0`. Upstream: `f64::ln` (which returns NaN / -inf).
#[inline(always)]
pub fn ln(x: Fixed) -> Fixed {
    Fixed { raw: kernels::ln(x.raw) }
}

/// `atan2(y, x)`, the angle of the point `(x, y)`, in `(-PI, PI]`. Total (no panic).
///
/// One unsigned division `min / max` at scale 2^61 then the degree-25 polynomial of `atan`.
/// Max error 1.12 ulp, whatever the magnitudes (the ratio is scale free, so `atan2` works for
/// world-scale vectors as well as for tiny ones). Exact values:
///
/// | case | result |
/// |---|---|
/// | `y = 0`, `x > 0` | `0` |
/// | `y = 0`, `x < 0` | `PI` (the `Fixed` constant, floored) |
/// | `y > 0`, `x = 0` | `FRAC_PI_2` |
/// | `y < 0`, `x = 0` | `-FRAC_PI_2` |
/// | `y = 0`, `x = 0` | `0` — the angle is undefined; `0` is returned, like `f64::atan2` |
///
/// `atan2(-y, x) = -atan2(y, x)` holds bit for bit. Upstream: `f64::atan2` (same argument order).
#[inline(always)]
pub fn atan2(y: Fixed, x: Fixed) -> Fixed {
    Fixed { raw: kernels::atan2(y.raw, x.raw) }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use crate::fixed::convert::from_raw as fx;
    use crate::fixed::kernels::transcendental::alternatives as alt;
    use crate::fixed::math::abs_diff_eq;
    use crate::fixed::types::{
        E, EPSILON, FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, HALF, LN_10, LN_2,
        MAX, MIN, ONE, PI, TAU, TWO, ZERO,
    };
    use crate::fixed::{fused, oracle_scalar as oracle};
    use super::{acos, asin, atan, atan2, cos, exp, ln, sin, sin_cos, tan};

    /// Benchmark points: 2.5 rad, 0.75, 5.0, -0.3, 0.9.
    const BX: i64 = 0x280000000;
    const B075: i64 = 0xc0000000;
    const B5: i64 = 0x500000000;
    const BM03: i64 = -0x4ccccccd;
    const B09: i64 = 0xe6666666;
    const BY2: i64 = -0x140000000;
    const BX2: i64 = -0xc0000000;

    // --- oracle vectors (tools/oracle: upstream nalgebra in f64 through libm) -------------------

    #[test]
    fn test_sin_oracle_vectors() {
        for case in oracle::scalar_sin_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(sin(fx(x)), fx(e), tol), "sin {}", x);
        }
    }

    #[test]
    fn test_cos_oracle_vectors() {
        for case in oracle::scalar_cos_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(cos(fx(x)), fx(e), tol), "cos {}", x);
        }
    }

    #[test]
    fn test_tan_oracle_vectors() {
        for case in oracle::scalar_tan_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(tan(fx(x)), fx(e), tol), "tan {}", x);
        }
    }

    #[test]
    fn test_atan_oracle_vectors() {
        for case in oracle::scalar_atan_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(atan(fx(x)), fx(e), tol), "atan {}", x);
        }
    }

    #[test]
    fn test_atan2_oracle_vectors() {
        for case in oracle::scalar_atan2_cases() {
            let (y, x, e, tol) = *case;
            assert!(abs_diff_eq(atan2(fx(y), fx(x)), fx(e), tol), "atan2 {} {}", y, x);
        }
    }

    #[test]
    fn test_asin_oracle_vectors() {
        for case in oracle::scalar_asin_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(asin(fx(x)), fx(e), tol), "asin {}", x);
        }
    }

    #[test]
    fn test_acos_oracle_vectors() {
        for case in oracle::scalar_acos_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(acos(fx(x)), fx(e), tol), "acos {}", x);
        }
    }

    /// The oracle tolerances are generous (they allow a Q32.32 range reduction); this pins the
    /// accuracy actually delivered: 2 ulp over every oracle case.
    #[test]
    fn test_oracle_vectors_within_two_ulp() {
        for case in oracle::scalar_sin_cases() {
            let (x, e, _) = *case;
            assert!(abs_diff_eq(sin(fx(x)), fx(e), 2), "sin {}", x);
        }
        for case in oracle::scalar_cos_cases() {
            let (x, e, _) = *case;
            assert!(abs_diff_eq(cos(fx(x)), fx(e), 2), "cos {}", x);
        }
        for case in oracle::scalar_atan2_cases() {
            let (y, x, e, _) = *case;
            assert!(abs_diff_eq(atan2(fx(y), fx(x)), fx(e), 2), "atan2 {} {}", y, x);
        }
        for case in oracle::scalar_asin_cases() {
            let (x, e, _) = *case;
            assert!(abs_diff_eq(asin(fx(x)), fx(e), 2), "asin {}", x);
        }
        for case in oracle::scalar_acos_cases() {
            let (x, e, _) = *case;
            assert!(abs_diff_eq(acos(fx(x)), fx(e), 2), "acos {}", x);
        }
    }

    // --- exact values ---------------------------------------------------------------------------

    #[test]
    fn test_sin_cos_exact_at_zero() {
        assert!(sin(ZERO) == ZERO);
        assert!(cos(ZERO) == ONE);
        assert!(sin_cos(ZERO) == (ZERO, ONE));
        assert!(tan(ZERO) == ZERO);
        // Below the first non-linear term sin is the identity and cos is exactly 1.
        assert!(sin(EPSILON) == EPSILON);
        assert!(sin(-EPSILON) == -EPSILON);
        assert!(cos(EPSILON) == ONE);
    }

    #[test]
    fn test_sin_cos_at_quarter_turns() {
        // The `Fixed` constants are floored, so these are the sine and cosine of an angle a
        // fraction of an ulp away from the exact multiple of pi/2.
        assert!(sin_cos(FRAC_PI_2) == (ONE, ZERO));
        assert!(sin_cos(PI) == (fx(1), -ONE));
        assert!(sin_cos(PI + FRAC_PI_2) == (-ONE, fx(-1)));
        // TAU is `floor(2*pi * 2^32)`, one raw unit above `PI + PI`: the two differ in the sine.
        assert!(sin_cos(TAU) == (ZERO, ONE));
        assert!(sin_cos(PI + PI) == (fx(-1), ONE));
        assert!(sin_cos(-FRAC_PI_2) == (-ONE, ZERO));
    }

    #[test]
    fn test_sin_of_pi_over_six_is_exactly_half() {
        assert!(sin(FRAC_PI_6) == HALF);
        assert!(cos(FRAC_PI_3) == fx(0x80000001));
        assert!(abs_diff_eq(cos(FRAC_PI_4), FRAC_1_SQRT_2, 2));
        assert!(abs_diff_eq(sin(FRAC_PI_4), FRAC_1_SQRT_2, 2));
    }

    #[test]
    fn test_sin_odd_and_cos_even_exactly() {
        let xs: [i64; 8] = [
            1, -1, BX, -BX, 0x1234567, 0x7fffffffffffffff, -0x7fffffffffffffff, 0x7fffffff00000000,
        ];
        for x in xs.span() {
            let (p, n) = (fx(*x), fx(-*x));
            assert!(sin(n) == -sin(p), "sin odd {}", *x);
            assert!(cos(n) == cos(p), "cos even {}", *x);
            assert!(tan(n) == -tan(p), "tan odd {}", *x);
        }
    }

    #[test]
    fn test_sin_cos_matches_sin_and_cos_bit_for_bit() {
        let xs: [i64; 9] = [
            0, 1, -1, BX, -BX, 0x1234567, 0x123456789, 0x7fffffffffffffff, -0x7fffffffffffffff,
        ];
        for x in xs.span() {
            let x = fx(*x);
            assert!(sin_cos(x) == (sin(x), cos(x)), "sin_cos {}", x.raw);
        }
    }

    #[test]
    fn test_sin_cos_pythagorean_identity() {
        let xs: [i64; 6] = [
            0x40000000, BX, -0x1234567, 0x123456789, 0x1ffffffff, 0x7fffffffffffffff,
        ];
        for x in xs.span() {
            let (s, c) = sin_cos(fx(*x));
            // One fused kernel, one floor: the sum is ONE or ONE + 1 raw unit.
            assert!(abs_diff_eq(fused::sum_prod2(s, s, c, c), ONE, 2), "identity {}", *x);
        }
    }

    #[test]
    fn test_sin_large_arguments_stay_accurate() {
        // The whole `Fixed` range is accepted; the drift of the 2^61 reduction constant stays
        // below 3 ulp (`tools/polygen/polygen.py --report`).
        assert!(sin(MAX) == fx(-0xf8a7c8a0));
        assert!(sin(MIN) == fx(0xf8a7c8a0));
        let (s, c) = sin_cos(MAX);
        assert!(abs_diff_eq(fused::sum_prod2(s, s, c, c), ONE, 2));
        let (s, c) = sin_cos(fx(0x7ffffffffffffffe));
        assert!(abs_diff_eq(fused::sum_prod2(s, s, c, c), ONE, 2));
    }

    // --- tan -------------------------------------------------------------------------------------

    #[test]
    fn test_tan_values() {
        assert!(tan(FRAC_PI_4) == ONE);
        assert!(tan(fx(BX)) == fx(-0xbf3cda70));
        assert!(abs_diff_eq(tan(FRAC_PI_6), fx(0x93cd3a2c), 4));
        assert!(tan(PI) == fx(-1));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_tan_near_pole_overflows() {
        // pi/2 rounds to a value 0.27 ulp below the pole: the tangent is 1.6e10 > 2^31.
        tan(black_box(FRAC_PI_2));
    }

    // --- atan / atan2
    // ------------------------------------------------------------------------------

    #[test]
    fn test_atan2_exact_on_the_axes() {
        assert!(atan2(ZERO, ONE) == ZERO);
        assert!(atan2(ZERO, MAX) == ZERO);
        assert!(atan2(ZERO, -ONE) == PI);
        assert!(atan2(ONE, ZERO) == FRAC_PI_2);
        assert!(atan2(-ONE, ZERO) == -FRAC_PI_2);
        assert!(atan2(ONE, ONE) == FRAC_PI_4);
        assert!(atan2(-ONE, ONE) == -FRAC_PI_4);
        assert!(atan2(MAX, MAX) == FRAC_PI_4);
    }

    /// The angle of `(0, 0)` is undefined; `0` is returned (documented, like `f64::atan2`).
    #[test]
    fn test_atan2_of_zero_zero_is_zero() {
        assert!(atan2(ZERO, ZERO) == ZERO);
    }

    #[test]
    fn test_atan2_is_odd_in_y_exactly() {
        let ys: [i64; 5] = [0x100000000, 0x300000000, 1, 0x7fffffffffffffff, 0x123456789];
        let xs: [i64; 5] = [0x300000000, 0x100000000, -1, -0x7fffffffffffffff, -0x9876543];
        let mut i: u32 = 0;
        while i != 5 {
            let (y, x) = (*ys.span()[i], *xs.span()[i]);
            assert!(atan2(fx(-y), fx(x)) == -atan2(fx(y), fx(x)), "odd {} {}", y, x);
            i += 1;
        }
    }

    #[test]
    fn test_atan_matches_atan2_with_one() {
        let xs: [i64; 8] = [0, 1, -1, B075, -B075, B5, 0x7fffffffffffffff, -0x7fffffffffffffff];
        for x in xs.span() {
            assert!(atan(fx(*x)) == atan2(fx(*x), ONE), "atan {}", *x);
        }
    }

    #[test]
    fn test_atan_values() {
        assert!(atan(ZERO) == ZERO);
        assert!(atan(ONE) == FRAC_PI_4);
        assert!(atan(-ONE) == -FRAC_PI_4);
        assert!(atan(EPSILON) == EPSILON);
        // atan is bounded by pi/2 and reaches it to within 1 ulp at the end of the range.
        assert!(atan(MAX) < FRAC_PI_2);
        assert!(atan(MAX) > FRAC_PI_2 - fx(4));
        assert!(atan(MIN) > -FRAC_PI_2);
    }

    /// `atan2(sin x, cos x)` is the identity on `(-pi, pi]`, up to the accumulated roundings.
    #[test]
    fn test_atan2_of_sin_cos_round_trips() {
        let xs: [i64; 7] = [0, 0x40000000, BX, -BX, 0x300000000, -0x300000000, 0x3243f6a87];
        for x in xs.span() {
            let (s, c) = sin_cos(fx(*x));
            assert!(abs_diff_eq(atan2(s, c), fx(*x), 4), "round trip {}", *x);
        }
    }

    // --- asin / acos
    // -------------------------------------------------------------------------------

    #[test]
    fn test_asin_acos_exact_at_the_endpoints() {
        assert!(asin(ZERO) == ZERO);
        assert!(asin(ONE) == FRAC_PI_2);
        assert!(asin(-ONE) == -FRAC_PI_2);
        assert!(asin(HALF) == FRAC_PI_6);
        assert!(asin(EPSILON) == EPSILON);
        assert!(acos(ZERO) == FRAC_PI_2);
        assert!(acos(ONE) == ZERO);
        assert!(acos(-ONE) == PI);
        assert!(abs_diff_eq(acos(HALF), FRAC_PI_3, 2));
    }

    /// `asin(x) + acos(x) == pi/2` and `acos(-x) == pi - acos(x)` hold bit for bit.
    #[test]
    fn test_asin_acos_complement_exactly() {
        let xs: [i64; 9] = [
            0, 1, -1, 0x80000000, -0x80000000, BM03, B09, 0x100000000, -0x100000000,
        ];
        for x in xs.span() {
            let x = fx(*x);
            assert!(asin(x) + acos(x) == FRAC_PI_2, "complement {}", x.raw);
            assert!(acos(-x) == PI - acos(x), "acos mirror {}", x.raw);
            assert!(asin(-x) == -asin(x), "asin odd {}", x.raw);
        }
    }

    #[test]
    fn test_asin_of_sin_round_trips() {
        let xs: [i64; 5] = [0, 0x40000000, 0x80000000, -0x80000000, 0x1921fb544];
        for x in xs.span() {
            assert!(abs_diff_eq(asin(sin(fx(*x))), fx(*x), 4), "asin(sin) {}", *x);
        }
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_asin_above_one_panics() {
        asin(black_box(fx(0x100000001)));
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_asin_below_minus_one_panics() {
        asin(black_box(fx(-0x100000001)));
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_acos_above_one_panics() {
        acos(black_box(MAX));
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_acos_of_min_panics() {
        acos(black_box(MIN));
    }

    // --- accuracy of the alternatives (evidence for the chosen degrees)
    // ---------------------------

    #[test]
    fn test_alternative_degrees_bracket_the_default() {
        // Degree 7 is 466 ulp off and degree 9 still 1.3 ulp; degree 11 costs 100 gas more
        // and lands on the rounding floor (0.53 ulp) - see `polygen.py --report`.
        let e = sin(fx(BX));
        assert!(!abs_diff_eq(fx(alt::sin_deg7(BX)), e, 4));
        assert!(abs_diff_eq(fx(alt::sin_deg9(BX)), e, 2));
        let a = atan2(fx(BY2), fx(BX2));
        assert!(!abs_diff_eq(fx(alt::atan2_deg17(BY2, BX2)), a, 4));
        assert!(abs_diff_eq(fx(alt::atan2_deg29(BY2, BX2)), a, 2));
        let s = asin(fx(BM03));
        assert!(!abs_diff_eq(fx(alt::asin_deg11(BM03)), s, 4));
        assert!(abs_diff_eq(fx(alt::asin_deg19(BM03)), s, 2));
    }

    // --- gas benchmarks
    // ---------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_sin__baseline() {
        let _x = black_box(fx(BX));
        let e = black_box(fx(0x9935786e));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin__deg11() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0x9935786e));
        assert!(sin(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin__deg11_negative() {
        let x = black_box(fx(-BX));
        let e = black_box(fx(-0x9935786e));
        assert!(sin(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin__deg11_huge_argument() {
        let x = black_box(MAX);
        let e = black_box(fx(-0xf8a7c8a0));
        assert!(sin(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin__deg7_alt() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0x99357882));
        assert!(fx(alt::sin_deg7(x.raw)) == e);
    }

    /// The shipped degrees through the alternatives module: comparable with `deg7_alt` and
    /// `deg9_alt` (same call shape), and 100 gas above the inlined `deg11` above.
    #[test]
    #[inline(never)]
    fn bench_sin__deg11_alt() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0x9935786e));
        assert!(fx(alt::sin_deg11(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin__deg9_alt() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0x9935786f));
        assert!(fx(alt::sin_deg9(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_cos__baseline() {
        let _x = black_box(fx(BX));
        let e = black_box(fx(-0xcd17bf7c));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_cos__deg10() {
        let x = black_box(fx(BX));
        let e = black_box(fx(-0xcd17bf7c));
        assert!(cos(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_cos__baseline() {
        let _x = black_box(fx(BX));
        let e = black_box(fx(0x9935786e));
        assert!(e == e);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_cos__shared() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0x9935786e));
        let (s, c) = sin_cos(x);
        assert!(s == e);
        assert!(c == fx(-0xcd17bf7c));
    }

    #[test]
    #[inline(never)]
    fn bench_sin_cos__two_calls() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0x9935786e));
        let (s, c) = alt::sin_cos_two_calls(x.raw);
        assert!(fx(s) == e);
        assert!(fx(c) == fx(-0xcd17bf7c));
    }

    #[test]
    #[inline(never)]
    fn bench_tan__baseline() {
        let _x = black_box(fx(BX));
        let e = black_box(fx(-0xbf3cda70));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_tan__wide_ratio() {
        let x = black_box(fx(BX));
        let e = black_box(fx(-0xbf3cda70));
        assert!(tan(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_tan__q32_ratio() {
        let x = black_box(fx(BX));
        // One ulp away from `tan`: the Q32.32 ratio loses the 16 extra bits of the wide one.
        let e = black_box(fx(-0xbf3cda6f));
        assert!(fx(alt::tan_from_q32(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan__baseline() {
        let _x = black_box(fx(B075));
        let e = black_box(fx(0xa4bc7d19));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan__unit_no_division() {
        let x = black_box(fx(B075));
        let e = black_box(fx(0xa4bc7d19));
        assert!(atan(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan__above_one_with_division() {
        let x = black_box(fx(B5));
        let e = black_box(fx(0x15f973152));
        assert!(atan(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan__reciprocal_only() {
        let x = black_box(fx(B5));
        let e = black_box(fx(0x15f973152));
        assert!(fx(alt::atan_reciprocal_only(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan__via_atan2() {
        let x = black_box(fx(B075));
        let e = black_box(fx(0xa4bc7d19));
        assert!(fx(alt::atan_via_atan2(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan2__baseline() {
        let _y = black_box(fx(BY2));
        let _x = black_box(fx(BX2));
        let e = black_box(fx(-0x21c78a3f4));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan2__deg25() {
        let y = black_box(fx(BY2));
        let x = black_box(fx(BX2));
        let e = black_box(fx(-0x21c78a3f4));
        assert!(atan2(y, x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan2__deg17() {
        let y = black_box(fx(BY2));
        let x = black_box(fx(BX2));
        let e = black_box(fx(-0x21c78a3c5));
        assert!(fx(alt::atan2_deg17(y.raw, x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan2__deg21() {
        let y = black_box(fx(BY2));
        let x = black_box(fx(BX2));
        let e = black_box(fx(-0x21c78a3f5));
        assert!(fx(alt::atan2_deg21(y.raw, x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_atan2__deg29() {
        let y = black_box(fx(BY2));
        let x = black_box(fx(BX2));
        let e = black_box(fx(-0x21c78a3f4));
        assert!(fx(alt::atan2_deg29(y.raw, x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_asin__baseline() {
        let _x = black_box(fx(BM03));
        let e = black_box(fx(-0x4e005679));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_asin__deg15_small_x() {
        let x = black_box(fx(BM03));
        let e = black_box(fx(-0x4e005679));
        assert!(asin(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_asin__deg15_large() {
        let x = black_box(fx(B09));
        let e = black_box(fx(0x11ea93705));
        assert!(asin(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_asin__deg11_small() {
        let x = black_box(fx(BM03));
        let e = black_box(fx(-0x4e005680));
        assert!(fx(alt::asin_deg11(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_asin__deg19_small() {
        let x = black_box(fx(BM03));
        let e = black_box(fx(-0x4e005679));
        assert!(fx(alt::asin_deg19(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_acos__baseline() {
        let _x = black_box(fx(BM03));
        let e = black_box(fx(0x1e0200bbd));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_acos__deg15_small() {
        let x = black_box(fx(BM03));
        let e = black_box(fx(0x1e0200bbd));
        assert!(acos(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_acos__deg15_large() {
        let x = black_box(fx(B09));
        let e = black_box(fx(0x73767e3f));
        assert!(acos(x) == e);
    }

    // --- exp / ln
    // ---------------------------------------------------------------------------------

    #[test]
    fn test_exp_oracle_vectors() {
        for case in oracle::scalar_exp_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(exp(fx(x)), fx(e), tol), "exp {}", x);
        }
    }

    #[test]
    fn test_ln_oracle_vectors() {
        for case in oracle::scalar_ln_cases() {
            let (x, e, tol) = *case;
            assert!(abs_diff_eq(ln(fx(x)), fx(e), tol), "ln {}", x);
        }
    }

    #[test]
    fn test_exp_ln_exact_values() {
        assert!(exp(ZERO) == ONE);
        assert!(ln(ONE) == ZERO);
        assert!(exp(ONE) == E);
        assert!(ln(E) == ONE);
        assert!(exp(-ONE) == fx(0x5e2d58d9));
        // LN_2, LN_10 and E are floored constants, `exp` / `ln` round to nearest: 1 ulp apart.
        assert!(abs_diff_eq(ln(TWO), LN_2, 2));
        assert!(abs_diff_eq(exp(LN_2), TWO, 2));
        assert!(abs_diff_eq(ln(fx(0xa00000000)), LN_10, 2));
    }

    /// `ln(2^k)` is `k * ln 2`. The `Fixed` constant `LN_2` is floored (0.81 ulp low), so the
    /// comparison against `k * LN_2` drifts by `0.81 * abs(k)` - the logarithm itself does not.
    #[test]
    fn test_ln_of_powers_of_two() {
        let raws: [i64; 7] = [
            1, 0x10000, 0x80000000, 0x100000000, 0x200000000, 0x1000000000000, 0x4000000000000000,
        ];
        let ks: [i64; 7] = [-32, -16, -1, 0, 1, 16, 30];
        let mut i: u32 = 0;
        while i != 7 {
            let (raw, k) = (*raws.span()[i], *ks.span()[i]);
            let want = LN_2.raw * k;
            let slack: u64 = (if k < 0 {
                -k
            } else {
                k
            } + 1).try_into().unwrap();
            assert!(abs_diff_eq(ln(fx(raw)), fx(want), slack), "ln 2^k {}", k);
            i += 1;
        }
    }

    /// `exp` underflows to zero instead of panicking, and is still positive just above.
    #[test]
    fn test_exp_underflows_to_zero() {
        assert!(exp(fx(-23 * 0x100000000)) == ZERO);
        assert!(exp(MIN) == ZERO);
        assert!(exp(fx(-22 * 0x100000000)) == EPSILON);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_exp_above_range_panics() {
        exp(black_box(fx(22 * 0x100000000)));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_exp_of_max_panics() {
        exp(black_box(MAX));
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_ln_of_zero_panics() {
        ln(black_box(ZERO));
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_ln_of_negative_panics() {
        ln(black_box(-ONE));
    }

    /// `ln(exp(x)) == x` to within the accumulated roundings.
    #[test]
    fn test_ln_of_exp_round_trips() {
        let xs: [i64; 6] = [0, 0x40000000, -0x40000000, BX, -BX, 0x800000000];
        for x in xs.span() {
            assert!(abs_diff_eq(ln(exp(fx(*x))), fx(*x), 8), "ln(exp) {}", *x);
        }
    }

    #[test]
    #[inline(never)]
    fn bench_exp__baseline() {
        let _x = black_box(fx(BX));
        let e = black_box(fx(0xc2eb7ec98));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_exp__deg8() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0xc2eb7ec98));
        assert!(exp(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_exp__deg8_negative() {
        let x = black_box(fx(-BX));
        let e = black_box(fx(0x150385c1));
        assert!(exp(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_exp__deg6_alt() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0xc2eb7ecd3));
        assert!(fx(alt::exp_deg6(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_exp__deg10_alt() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0xc2eb7ec98));
        assert!(fx(alt::exp_deg10(x.raw)) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ln__baseline() {
        let _x = black_box(fx(BX));
        let e = black_box(fx(0xea920787));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ln__deg11() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0xea920787));
        assert!(ln(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ln__deg11_tiny() {
        let x = black_box(EPSILON);
        let e = black_box(fx(-0x162e42fefa));
        assert!(ln(x) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_ln__deg7_alt() {
        let x = black_box(fx(BX));
        let e = black_box(fx(0xea920787));
        assert!(fx(alt::ln_deg7(x.raw)) == e);
    }
}
