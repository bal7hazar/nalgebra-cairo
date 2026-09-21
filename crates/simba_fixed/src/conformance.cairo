//! Conformance of `fixed::Fixed` (glam.cairo) with `simba::fixed::Fixed`, operand by operand.
//!
//! Test-only module. Two scalars with the same representation are not automatically the same
//! number: this suite runs both on the generated operands of `crate::vectors` and asserts, bit for
//! bit, either that they agree or — where they do not — the exact shape of the difference.
//! Nothing is papered over and neither scalar is modified: a disagreement asserted here is a
//! documented property of the pair, reported in `REPORT.md`.
//!
//! Summary of what the suite finds (details on each test):
//!
//! | operation | verdict |
//! |---|---|
//! | `+`, `-`, unary `-`, `*`, comparisons | identical |
//! | `abs`, `floor`, `ceil`, `round`, `trunc`, `fract`, `min`, `max`, `sqrt` | identical |
//! | `wide::dot3` / `sum_prod3`, `mul_add`, `mul_sub` / `diff_prod`, `norm3`, `norm_squared3` |
//! identical |
//! | `exp`, `ln` | identical (glam.cairo has neither; `Transcendental` forwards to simba) |
//! | `Real::div`, `Real::rem`, `Real::recip` (what nalgebra calls) | identical: simba's floor |
//! | glam's own `/` operator | **differs**: glam truncates toward zero, simba floors |
//! | glam's own `%` operator | **differs**: truncated remainder vs simba's floored modulo |
//! | `FixedTrait::recip` | **differs**: same truncate-vs-floor split |
//! | `signum(0)` | **differs**: glam gives `+1`, simba `0` |
//! | 7 of the 20 `Real` constants | **differ** by 1 ulp: glam rounds to nearest, simba floors |
//! | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | **differ** by a few ulp: two different
//! generated polynomials; both inside the oracle tolerances |

use fixed::{Fixed as Glam, FixedTrait, wide as gwide};
use simba::fixed::{Fixed as Simba, fused, math, transcendental as strans, types};
use simba::scalar::{Real, Transcendental};
use crate::convert::to_simba;
use crate::real::FixedReal;
use crate::transcendental::FixedTranscendental;
use crate::{oracle_scalar as oracle, vectors};

/// A raw value as glam's scalar.
#[inline(always)]
fn g(raw: i64) -> Glam {
    Glam { raw }
}

/// A raw value as simba's scalar.
#[inline(always)]
fn s(raw: i64) -> Simba {
    Simba { raw }
}

/// `|a - b| <= ulps`, on the exact difference (simba's `abs_diff_eq`, which cannot overflow).
#[inline(always)]
fn close(a: Glam, b: Simba, ulps: u64) -> bool {
    math::abs_diff_eq(to_simba(a), b, ulps)
}

/// Tolerance between the two sine / cosine implementations for an angle of raw magnitude `|x|`:
/// `4 + |x| / 8` raw units EACH (the oracle's own formula, `crate::oracle_scalar`), so twice that
/// between them. Beyond a few thousand turns the bound exceeds the scalar range and the comparison
/// becomes vacuous; the accuracy of the glam path is then pinned by the oracle tests below, not by
/// this one.
fn angle_tolerance(x: i64) -> u64 {
    let wide: i128 = x.into();
    let magnitude: u128 = if wide < 0 {
        (-wide).try_into().unwrap()
    } else {
        wide.try_into().unwrap()
    };
    let bound = 2 * (5 + magnitude / 8);
    match bound.try_into() {
        Some(value) => value,
        None => 0xffffffffffffffff,
    }
}

// --- operators ---------------------------------------------------------------------------------

/// `+`, `-` and unary `-` are exact `i64` operations on both sides: bit-identical.
#[test]
fn test_conformance_add_sub_neg() {
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        assert!((g(a) + g(b)).raw == (s(a) + s(b)).raw, "add {} {}", a, b);
        assert!((g(a) - g(b)).raw == (s(a) - s(b)).raw, "sub {} {}", a, b);
    }
    for case in vectors::unary_cases() {
        let a = *case;
        assert!((-g(a)).raw == (-s(a)).raw, "neg {}", a);
    }
}

/// `*` floors the exact product on both sides: bit-identical, whatever the signs.
#[test]
fn test_conformance_mul() {
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        assert!((g(a) * g(b)).raw == (s(a) * s(b)).raw, "mul {} {}", a, b);
    }
}

/// DIFFERENCE. `fixed::Fixed` divides toward zero (`bounded::div_trunc`), `simba::fixed::Fixed`
/// floors, consistently with its `*`. They agree exactly when the quotient is exact or
/// non-negative; otherwise glam's is simba's plus one ulp. Exhaustive: the test asserts one of the
/// two, chosen by the sign and the exactness of the quotient, never "close enough".
///
/// Only glam's OWN operator differs: nalgebra's generic code divides through `Real::div`, which
/// is simba's floor on every operand (`test_conformance_real_div_rem_are_simbas`), hence the
/// bit-identical `normalize` / `unscale` of `tests/test_nalgebra_types.cairo`.
#[test]
fn test_conformance_div_truncates_where_simba_floors() {
    let mut differences: u32 = 0;
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        let glam = (g(a) / g(b)).raw;
        let simba = (s(a) / s(b)).raw;
        // Exactness of the SCALED quotient `a * 2^32 / b`, not of `a / b`: `simba` is its floor.
        let exact: bool = simba.into() * b.into() == a.into() * 0x100000000_i128;
        let negative = a != 0 && (a < 0) != (b < 0);
        if exact || !negative {
            assert!(glam == simba, "div agrees {} {}", a, b);
        } else {
            assert!(glam == simba + 1, "div trunc = floor + 1 {} {}", a, b);
            differences += 1;
        }
    }
    // The split is real on this operand set, not a vacuous branch.
    assert!(differences > 0);
}

/// DIFFERENCE. `%` is the TRUNCATED remainder in glam.cairo (sign of the dividend, like Rust's
/// `%`) and the FLOORED modulo in simba (sign of the divisor). They agree when the remainder is
/// zero or the operands have the same sign; otherwise they differ by exactly the divisor.
#[test]
fn test_conformance_rem_truncated_vs_floored() {
    let mut differences: u32 = 0;
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        let glam = (g(a) % g(b)).raw;
        let simba = (s(a) % s(b)).raw;
        if simba == 0 || (a < 0) == (b < 0) {
            assert!(glam == simba, "rem agrees {} {}", a, b);
        } else {
            assert!(glam == simba - b, "rem trunc = floor - divisor {} {}", a, b);
            differences += 1;
        }
    }
    assert!(differences > 0);
}

/// `Real::<fixed::Fixed>::div` / `rem` are SIMBA's floor division and floored modulo, bit for
/// bit, on every operand pair — including the negative inexact quotients where glam's own `/`
/// and `%` differ. This is what closes the `normalize` / `unscale` / `new_normalize` gap.
#[test]
fn test_conformance_real_div_rem_are_simbas() {
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        assert!(Real::div(g(a), g(b)).raw == (s(a) / s(b)).raw, "Real::div {} {}", a, b);
        assert!(Real::rem(g(a), g(b)).raw == (s(a) % s(b)).raw, "Real::rem {} {}", a, b);
    }
}

/// The four comparisons are raw `i64` comparisons on both sides.
#[test]
fn test_conformance_comparisons() {
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        assert!((g(a) < g(b)) == (s(a) < s(b)), "lt {} {}", a, b);
        assert!((g(a) <= g(b)) == (s(a) <= s(b)), "le {} {}", a, b);
        assert!((g(a) > g(b)) == (s(a) > s(b)), "gt {} {}", a, b);
        assert!((g(a) >= g(b)) == (s(a) >= s(b)), "ge {} {}", a, b);
        assert!((g(a) == g(b)) == (s(a) == s(b)), "eq {} {}", a, b);
    }
}

// --- helpers -----------------------------------------------------------------------------------

/// `abs`, `floor`, `ceil`, `round`, `trunc`, `fract`, `min` and `max` are the same kernels on both
/// sides (ties away from zero for `round`, sign of the operand for `fract`).
#[test]
fn test_conformance_rounding_and_ordering() {
    for case in vectors::unary_cases() {
        let a = *case;
        assert!(FixedTrait::abs(g(a)).raw == math::abs(s(a)).raw, "abs {}", a);
        assert!(FixedTrait::floor(g(a)).raw == math::floor(s(a)).raw, "floor {}", a);
        assert!(FixedTrait::ceil(g(a)).raw == math::ceil(s(a)).raw, "ceil {}", a);
        assert!(FixedTrait::round(g(a)).raw == math::round(s(a)).raw, "round {}", a);
        assert!(FixedTrait::trunc(g(a)).raw == math::trunc(s(a)).raw, "trunc {}", a);
        assert!(FixedTrait::fract(g(a)).raw == math::fract(s(a)).raw, "fract {}", a);
        assert!(FixedTrait::to_int(g(a)) == Real::to_int(g(a)), "to_int {}", a);
    }
    for case in vectors::binary_cases() {
        let (a, b) = *case;
        assert!(FixedTrait::min(g(a), g(b)).raw == math::min(s(a), s(b)).raw, "min {} {}", a, b);
        assert!(FixedTrait::max(g(a), g(b)).raw == math::max(s(a), s(b)).raw, "max {} {}", a, b);
    }
}

/// `sqrt` is the exact integer square root of `raw * 2^32` on both sides (floor, since the
/// argument is non-negative): bit-identical.
#[test]
fn test_conformance_sqrt() {
    for case in vectors::sqrt_cases() {
        let a = *case;
        assert!(FixedTrait::sqrt(g(a)).raw == math::sqrt(s(a)).raw, "sqrt {}", a);
        assert!(Real::sqrt(g(a)).raw == math::sqrt(s(a)).raw, "Real::sqrt {}", a);
    }
}

/// DIFFERENCE. `recip` inherits the split of `/`: `fixed::FixedTrait::recip` truncates
/// `2^64 / raw`, `simba::fixed::math::recip` floors it. `Real::<fixed::Fixed>::recip` is simba's,
/// so generic nalgebra code (`normalize`, `try_inverse`, ...) is unaffected — only a caller
/// writing `FixedTrait::recip` directly sees the difference.
#[test]
fn test_conformance_recip_truncates_where_simba_floors() {
    let mut differences: u32 = 0;
    for case in vectors::recip_cases() {
        let a = *case;
        let glam = FixedTrait::recip(g(a)).raw;
        let simba = math::recip(s(a)).raw;
        assert!(Real::recip(g(a)).raw == simba, "Real::recip is simba's {}", a);
        if glam == simba {
            continue;
        }
        assert!(a < 0 && glam == simba + 1, "recip trunc = floor + 1 {}", a);
        differences += 1;
    }
    assert!(differences > 0);
}

/// DIFFERENCE. `fixed::FixedTrait::signum` returns `+1` for zero (glam follows `f32::signum` on
/// `+0.0`), `simba::fixed::math::signum` returns `0` (there is no signed zero, and a zero signum
/// is what nalgebra's generic code expects). `Real::<fixed::Fixed>::signum` is simba's.
#[test]
fn test_conformance_signum_differs_at_zero() {
    assert!(FixedTrait::signum(g(0)).raw == 0x100000000);
    assert!(math::signum(s(0)).raw == 0);
    assert!(Real::signum(g(0)).raw == 0);
    for case in vectors::unary_cases() {
        let a = *case;
        if a == 0 {
            continue;
        }
        assert!(FixedTrait::signum(g(a)).raw == math::signum(s(a)).raw, "signum {}", a);
        assert!(Real::signum(g(a)).raw == math::signum(s(a)).raw, "Real::signum {}", a);
    }
}

// --- fused kernels -----------------------------------------------------------------------------

/// `fixed::wide::mul_add` and `simba::fixed::fused::mul_add` are the same single-rescale kernel;
/// so are `FixedTrait::mul_add` (argument order `self * a + b`) and `Real::mul_add`.
#[test]
fn test_conformance_mul_add() {
    for case in vectors::mul_add_cases() {
        let (a, b, c) = *case;
        let expected = fused::mul_add(s(a), s(b), s(c)).raw;
        assert!(
            gwide::mul_add(g(a), g(b), g(c)).raw == expected, "wide::mul_add {} {} {}", a, b, c,
        );
        assert!(FixedTrait::mul_add(g(a), g(b), g(c)).raw == expected, "mul_add {} {} {}", a, b, c);
        assert!(Real::mul_add(g(a), g(b), g(c)).raw == expected, "Real::mul_add {} {} {}", a, b, c);
        assert!(
            Real::mul_sub(g(a), g(b), g(c)).raw == fused::mul_sub(s(a), s(b), s(c)).raw,
            "mul_sub {} {} {}",
            a,
            b,
            c,
        );
    }
}

/// `fixed::wide::dot3` / `mul_sub` / `norm3` / `norm3_squared` against simba's `sum_prod3` /
/// `diff_prod` / `norm3` / `norm_squared3`: the same unscaled accumulation and the same single
/// rescale, hence bit-identical on every vector pair.
#[test]
fn test_conformance_vec3_kernels() {
    for case in vectors::vec3_cases() {
        let (ax, ay, az, bx, by, bz) = *case;
        assert!(
            gwide::dot3(g(ax), g(bx), g(ay), g(by), g(az), g(bz))
                .raw == fused::sum_prod3(s(ax), s(bx), s(ay), s(by), s(az), s(bz))
                .raw,
            "dot3 {} {} {}",
            ax,
            ay,
            az,
        );
        assert!(
            gwide::mul_sub(g(ay), g(bz), g(az), g(by))
                .raw == fused::diff_prod(s(ay), s(bz), s(az), s(by))
                .raw,
            "cross x {} {}",
            ay,
            az,
        );
        assert!(
            gwide::norm3(g(ax), g(ay), g(az)).raw == fused::norm3(s(ax), s(ay), s(az)).raw,
            "norm3 {} {} {}",
            ax,
            ay,
            az,
        );
        assert!(
            gwide::norm3_squared(g(ax), g(ay), g(az))
                .raw == fused::norm_squared3(s(ax), s(ay), s(az))
                .raw,
            "norm3_squared {} {} {}",
            ax,
            ay,
            az,
        );
        // Same figures through the trait, which is what nalgebra calls.
        assert!(
            Real::sum_prod3(g(ax), g(bx), g(ay), g(by), g(az), g(bz))
                .raw == fused::sum_prod3(s(ax), s(bx), s(ay), s(by), s(az), s(bz))
                .raw,
            "Real::sum_prod3",
        );
        assert!(
            Real::norm3(g(ax), g(ay), g(az)).raw == fused::norm3(s(ax), s(ay), s(az)).raw,
            "Real::norm3",
        );
    }
}

/// The `Wide` accumulator of `Real<fixed::Fixed>` IS simba's (`type Wide = simba::fixed::Wide`),
/// so a long sum of products accumulated through the trait rescales to simba's value.
#[test]
fn test_conformance_wide_accumulator() {
    for case in vectors::vec3_cases() {
        let (ax, ay, az, bx, by, bz) = *case;
        let w = Real::wide_add_prod(Real::<Glam>::wide_zero(), g(ax), g(bx));
        let w = Real::wide_add_prod(w, g(ay), g(by));
        let w = Real::wide_add_prod(w, g(az), g(bz));
        let rescaled: Glam = Real::wide_rescale(w);
        let expected = fused::sum_prod3(s(ax), s(bx), s(ay), s(by), s(az), s(bz));
        assert!(rescaled.raw == expected.raw, "wide sum {} {} {}", ax, ay, az);
    }
}

// --- constants ---------------------------------------------------------------------------------

/// `Real::<fixed::Fixed>` exposes SIMBA's constants, so a formula written against `Real::PI` gives
/// the same result with either scalar.
#[test]
fn test_conformance_real_constants_are_simbas() {
    assert!(Real::<Glam>::ZERO.raw == types::ZERO.raw);
    assert!(Real::<Glam>::ONE.raw == types::ONE.raw);
    assert!(Real::<Glam>::NEG_ONE.raw == types::NEG_ONE.raw);
    assert!(Real::<Glam>::TWO.raw == types::TWO.raw);
    assert!(Real::<Glam>::HALF.raw == types::HALF.raw);
    assert!(Real::<Glam>::EPSILON.raw == types::EPSILON.raw);
    assert!(Real::<Glam>::MIN.raw == types::MIN.raw);
    assert!(Real::<Glam>::MAX.raw == types::MAX.raw);
    assert!(Real::<Glam>::PI.raw == types::PI.raw);
    assert!(Real::<Glam>::TAU.raw == types::TAU.raw);
    assert!(Real::<Glam>::FRAC_PI_2.raw == types::FRAC_PI_2.raw);
    assert!(Real::<Glam>::FRAC_PI_3.raw == types::FRAC_PI_3.raw);
    assert!(Real::<Glam>::FRAC_PI_4.raw == types::FRAC_PI_4.raw);
    assert!(Real::<Glam>::FRAC_PI_6.raw == types::FRAC_PI_6.raw);
    assert!(Real::<Glam>::FRAC_1_PI.raw == types::FRAC_1_PI.raw);
    assert!(Real::<Glam>::E.raw == types::E.raw);
    assert!(Real::<Glam>::LN_2.raw == types::LN_2.raw);
    assert!(Real::<Glam>::LN_10.raw == types::LN_10.raw);
    assert!(Real::<Glam>::SQRT_2.raw == types::SQRT_2.raw);
    assert!(Real::<Glam>::FRAC_1_SQRT_2.raw == types::FRAC_1_SQRT_2.raw);
}

/// DIFFERENCE. glam.cairo's own constants round to NEAREST, simba's floor: seven of them are one
/// ulp above `Real`'s. Asserted, not tolerated: a change on either side must break this test.
#[test]
fn test_conformance_glam_constants_round_to_nearest() {
    // Exactly representable, or floor and nearest coincide: identical.
    assert!(fixed::ZERO.raw == Real::<Glam>::ZERO.raw);
    assert!(fixed::ONE.raw == Real::<Glam>::ONE.raw);
    assert!(fixed::NEG_ONE.raw == Real::<Glam>::NEG_ONE.raw);
    assert!(fixed::TWO.raw == Real::<Glam>::TWO.raw);
    assert!(fixed::HALF.raw == Real::<Glam>::HALF.raw);
    assert!(fixed::EPSILON.raw == Real::<Glam>::EPSILON.raw);
    assert!(fixed::MIN.raw == Real::<Glam>::MIN.raw);
    assert!(fixed::MAX.raw == Real::<Glam>::MAX.raw);
    assert!(fixed::TAU.raw == Real::<Glam>::TAU.raw);
    assert!(fixed::FRAC_PI_2.raw == Real::<Glam>::FRAC_PI_2.raw);
    assert!(fixed::FRAC_PI_4.raw == Real::<Glam>::FRAC_PI_4.raw);
    assert!(fixed::FRAC_PI_6.raw == Real::<Glam>::FRAC_PI_6.raw);
    assert!(fixed::FRAC_1_PI.raw == Real::<Glam>::FRAC_1_PI.raw);
    // Rounded up by glam, floored by simba: one ulp apart.
    assert!(fixed::PI.raw == Real::<Glam>::PI.raw + 1);
    assert!(fixed::FRAC_PI_3.raw == Real::<Glam>::FRAC_PI_3.raw + 1);
    assert!(fixed::E.raw == Real::<Glam>::E.raw + 1);
    assert!(fixed::LN_2.raw == Real::<Glam>::LN_2.raw + 1);
    assert!(fixed::LN_10.raw == Real::<Glam>::LN_10.raw + 1);
    assert!(fixed::SQRT_2.raw == Real::<Glam>::SQRT_2.raw + 1);
    assert!(fixed::FRAC_1_SQRT_2.raw == Real::<Glam>::FRAC_1_SQRT_2.raw + 1);
}

// --- transcendentals ---------------------------------------------------------------------------

/// `exp` and `ln` are simba's, bit for bit: glam.cairo has neither.
#[test]
fn test_conformance_exp_ln_are_simbas() {
    for case in oracle::scalar_exp_cases() {
        let (x, e, tol) = *case;
        assert!(Transcendental::exp(g(x)).raw == strans::exp(s(x)).raw, "exp {}", x);
        assert!(close(Transcendental::exp(g(x)), s(e), tol), "exp oracle {}", x);
    }
    for case in oracle::scalar_ln_cases() {
        let (x, e, tol) = *case;
        assert!(Transcendental::ln(g(x)).raw == strans::ln(s(x)).raw, "ln {}", x);
        assert!(close(Transcendental::ln(g(x)), s(e), tol), "ln oracle {}", x);
    }
}

/// The glam trigonometry reached through `Transcendental<fixed::Fixed>` stays inside the oracle
/// tolerances — the real accuracy statement of this package, independent of simba.
#[test]
fn test_conformance_trig_oracle_vectors() {
    for case in oracle::scalar_sin_cases() {
        let (x, e, tol) = *case;
        assert!(close(Transcendental::sin(g(x)), s(e), tol), "sin {}", x);
    }
    for case in oracle::scalar_cos_cases() {
        let (x, e, tol) = *case;
        assert!(close(Transcendental::cos(g(x)), s(e), tol), "cos {}", x);
    }
    for case in oracle::scalar_tan_cases() {
        let (x, e, tol) = *case;
        assert!(close(Transcendental::tan(g(x)), s(e), tol), "tan {}", x);
    }
    for case in oracle::scalar_atan_cases() {
        let (x, e, tol) = *case;
        assert!(close(Transcendental::atan(g(x)), s(e), tol), "atan {}", x);
    }
    for case in oracle::scalar_asin_cases() {
        let (x, e, tol) = *case;
        assert!(close(Transcendental::asin(g(x)), s(e), tol), "asin {}", x);
    }
    for case in oracle::scalar_acos_cases() {
        let (x, e, tol) = *case;
        assert!(close(Transcendental::acos(g(x)), s(e), tol), "acos {}", x);
    }
    for case in oracle::scalar_atan2_cases() {
        let (y, x, e, tol) = *case;
        assert!(close(Transcendental::atan2(g(y), g(x)), s(e), tol), "atan2 {} {}", y, x);
    }
}

/// The `Real` square roots and reciprocals reached through the glam scalar are simba's, and meet
/// the oracle bit for bit where the oracle is exact.
#[test]
fn test_conformance_real_oracle_vectors() {
    for case in oracle::scalar_sqrt_cases() {
        let (x, e, tol) = *case;
        assert!(Real::sqrt(g(x)).raw == math::sqrt(s(x)).raw, "sqrt {}", x);
        assert!(close(Real::sqrt(g(x)), s(e), tol), "sqrt oracle {}", x);
    }
    for case in oracle::scalar_inv_sqrt_cases() {
        let (x, e, tol) = *case;
        assert!(Real::inv_sqrt(g(x)).raw == math::inv_sqrt(s(x)).raw, "inv_sqrt {}", x);
        assert!(close(Real::inv_sqrt(g(x)), s(e), tol), "inv_sqrt oracle {}", x);
    }
    for case in oracle::scalar_recip_cases() {
        let (x, e, tol) = *case;
        assert!(Real::recip(g(x)).raw == math::recip(s(x)).raw, "recip {}", x);
        assert!(close(Real::recip(g(x)), s(e), tol), "recip oracle {}", x);
    }
}

/// DIFFERENCE. glam.cairo and simba evaluate different generated minimax polynomials, so their
/// trigonometric results differ by a few ulp. The gap is bounded here by the sum of the two
/// oracle tolerances; the exact values are not interchangeable.
#[test]
fn test_conformance_trig_differs_from_simba_within_tolerance() {
    let mut differences: u32 = 0;
    for case in vectors::angle_cases() {
        let x = *case;
        let tol = angle_tolerance(x);
        let (gs, gc) = Transcendental::sin_cos(g(x));
        let (ss, sc) = strans::sin_cos(s(x));
        assert!(close(gs, ss, tol), "sin {}", x);
        assert!(close(gc, sc, tol), "cos {}", x);
        // `sin_cos` shares the range reduction but must agree with the separate calls.
        assert!(Transcendental::sin(g(x)).raw == gs.raw, "sin_cos sin {}", x);
        assert!(Transcendental::cos(g(x)).raw == gc.raw, "sin_cos cos {}", x);
        if gs.raw != ss.raw || gc.raw != sc.raw {
            differences += 1;
        }
    }
    for case in vectors::unit_cases() {
        let x = *case;
        assert!(close(Transcendental::asin(g(x)), strans::asin(s(x)), 8), "asin {}", x);
        assert!(close(Transcendental::acos(g(x)), strans::acos(s(x)), 8), "acos {}", x);
    }
    for case in vectors::atan2_cases() {
        let (y, x) = *case;
        assert!(close(Transcendental::atan2(g(y), g(x)), strans::atan2(s(y), s(x)), 8), "atan2");
        assert!(close(Transcendental::atan(g(y)), strans::atan(s(y)), 8), "atan {}", y);
    }
    // The two trigonometries really are distinct: this is a finding, not a formality.
    assert!(differences > 0);
}

/// The exact points both implementations guarantee agree, so a quarter turn is a quarter turn in
/// either scalar.
#[test]
fn test_conformance_trig_exact_points() {
    assert!(Transcendental::sin(Real::<Glam>::ZERO).raw == 0);
    assert!(Transcendental::cos(Real::<Glam>::ZERO).raw == Real::<Glam>::ONE.raw);
    assert!(Transcendental::sin(Real::<Glam>::FRAC_PI_2).raw == Real::<Glam>::ONE.raw);
    assert!(Transcendental::asin(Real::<Glam>::ONE).raw == Real::<Glam>::FRAC_PI_2.raw);
    assert!(Transcendental::acos(Real::<Glam>::ONE).raw == 0);
    assert!(Transcendental::atan(Real::<Glam>::ONE).raw == Real::<Glam>::FRAC_PI_4.raw);
    assert!(
        Transcendental::atan2(Real::<Glam>::ONE, Real::<Glam>::ZERO)
            .raw == Real::<Glam>::FRAC_PI_2
            .raw,
    );
    assert!(Transcendental::exp(Real::<Glam>::ZERO).raw == Real::<Glam>::ONE.raw);
    assert!(Transcendental::ln(Real::<Glam>::ONE).raw == 0);
}

// --- panics ------------------------------------------------------------------------------------

/// Overflow of a `Real` kernel panics with simba's stable message even though the scalar is
/// glam.cairo's: the kernel is simba's.
#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_conformance_kernel_overflow_panics_as_simba() {
    let max = nalgebra_testing::black_box(Real::<Glam>::MAX);
    let _ = Real::sum_prod2(max, max, max, max);
}

/// ... and so does a division by zero reached through `Real::recip`.
#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_conformance_recip_of_zero_panics_as_simba() {
    let _ = Real::recip(nalgebra_testing::black_box(Real::<Glam>::ZERO));
}

/// ... and so does a division by zero reached through `Real::div` (glam's own `/` would panic
/// with `'Fixed: division by zero'`).
#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_conformance_real_div_by_zero_panics_as_simba() {
    let _ = Real::div(Real::<Glam>::ONE, nalgebra_testing::black_box(Real::<Glam>::ZERO));
}

/// ... and an out-of-range quotient.
#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_conformance_real_div_overflow_panics_as_simba() {
    let _ = Real::div(nalgebra_testing::black_box(Real::<Glam>::MAX), Real::<Glam>::HALF);
}

/// ... and a negative square root.
#[test]
#[should_panic(expected: 'simba: sqrt of negative')]
fn test_conformance_sqrt_of_negative_panics_as_simba() {
    let _ = Real::sqrt(nalgebra_testing::black_box(Real::<Glam>::NEG_ONE));
}

/// DIFFERENCE. The trigonometric domain check is glam.cairo's, so `acos` outside `[-1, 1]` panics
/// with `'Fixed: acos domain'` and not with simba's `'simba: out of domain'`.
#[test]
#[should_panic(expected: 'Fixed: acos domain')]
fn test_conformance_acos_domain_panics_as_glam() {
    let _ = Transcendental::acos(nalgebra_testing::black_box(Real::<Glam>::TWO));
}

/// DIFFERENCE. Likewise `asin`.
#[test]
#[should_panic(expected: 'Fixed: asin domain')]
fn test_conformance_asin_domain_panics_as_glam() {
    let _ = Transcendental::asin(nalgebra_testing::black_box(Real::<Glam>::TWO));
}
