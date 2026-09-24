//! Unit tests of the WP 8.4-P08 completion of `Quaternion`: the algebra (`squared`, `half`,
//! `inner`, `outer`, divisions, projections, `sqrt`), the transcendental functions, the polar
//! decomposition, the approximate comparisons and the trait impls (`Index`, `One`, arrays):
//! exact cases, identities, panics, and the oracle vectors of `tools/oracle` (upstream nalgebra
//! 0.35 on the same raw inputs, tolerance in ulp).
//!
//! `oracle_ext.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo quaternion_functions --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/quaternion/oracle_ext.cairo
//! ```
//!
//! Every oracle test checks EVERY case and prints the ones that leave the tolerance (with the
//! excess) before failing, so that a regression shows by how much.
//!
//! Moved from `crates/nalgebra/src/geometry/quaternion/tests_ext.cairo` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use core::num::traits::{One, Zero};
use fixed::Fixed;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::quaternion::{Quaternion, QuaternionTrait, QuaternionTranscendentalTrait};
use nalgebra_tests_utils::{
    ONE_RAW, excess, fx, int, max_ulp_diff_q, max_ulp_diff_v3, qi, qt, u3t, ulp_diff,
};
use simba::scalar::Real;
use crate::quaternion::oracle_ext;

const HALF_RAW: i64 = 0x80000000;
const QUARTER_RAW: i64 = 0x40000000;

/// `1 + 2i - 3j + 4k`.
fn a() -> Quaternion<Fixed> {
    qi(1, 2, -3, 4)
}

/// `-2 + i + 5j - k`.
fn b() -> Quaternion<Fixed> {
    qi(-2, 1, 5, -1)
}

/// `0.5 + 0.25i - 0.5j + 0.75k`, a small general quaternion (|q| ≈ 1.03).
fn s() -> Quaternion<Fixed> {
    qt((HALF_RAW, QUARTER_RAW, -HALF_RAW, 3 * QUARTER_RAW))
}

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

/// The excess of `got` over the tolerance around `expected` (printed when positive).
fn check_q(
    op: ByteArray, index: usize, got: Quaternion<Fixed>, expected: (i64, i64, i64, i64), tol: u64,
) -> u128 {
    report(op, index, max_ulp_diff_q(got, qt(expected)), tol)
}

// --- oracle vectors

#[test]
fn test_exp_oracle() {
    let mut cases = oracle_ext::quaternion_exp_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("exp", n, q.exp(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_ln_oracle() {
    let mut cases = oracle_ext::quaternion_ln_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("ln", n, q.ln(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_powf_oracle() {
    let mut cases = oracle_ext::quaternion_powf_cases();
    let mut worst = 0;
    let mut index = 0;
    while let Some(case) = cases.pop_front() {
        let (q, n, expected, tol) = *case;
        let err = max_ulp_diff_q(qt(q).powf(fx(n)), qt(expected));
        worst = core::cmp::max(worst, report("powf", index, err, tol));
        index += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_sqrt_oracle() {
    let mut cases = oracle_ext::quaternion_sqrt_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("sqrt", n, q.sqrt(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_squared_oracle_is_bit_exact() {
    let mut cases = oracle_ext::quaternion_squared_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("squared", n, q.squared(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_half_oracle() {
    let mut cases = oracle_ext::quaternion_half_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("half", n, q.half(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_inner_oracle_is_bit_exact() {
    let mut cases = oracle_ext::quaternion_inner_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("inner", n, a.inner(b), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_outer_oracle_is_bit_exact() {
    let mut cases = oracle_ext::quaternion_outer_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("outer", n, a.outer(b), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_right_div_oracle() {
    let mut cases = oracle_ext::quaternion_right_div_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("right_div", n, a.right_div(b).unwrap(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_left_div_oracle() {
    let mut cases = oracle_ext::quaternion_left_div_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("left_div", n, a.left_div(b).unwrap(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_project_oracle() {
    let mut cases = oracle_ext::quaternion_project_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("project", n, a.project(b).unwrap(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reject_oracle() {
    let mut cases = oracle_ext::quaternion_reject_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (a, b) = (qt(a), qt(b));
        worst = core::cmp::max(worst, check_q("reject", n, a.reject(b).unwrap(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_polar_decomposition_oracle() {
    let mut cases = oracle_ext::quaternion_polar_decomposition_cases();
    let mut worst = 0;
    let mut index = 0;
    while let Some(case) = cases.pop_front() {
        let (q, n, angle, axis, tol) = *case;
        let (gn, gangle, gaxis) = qt(q).polar_decomposition();
        let err = core::cmp::max(ulp_diff(gn, fx(n)), ulp_diff(gangle, fx(angle)));
        let err = core::cmp::max(err, max_ulp_diff_v3(gaxis.unwrap().value, u3t(axis).value));
        worst = core::cmp::max(worst, report("polar_decomposition", index, err, tol));
        index += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_from_polar_decomposition_oracle() {
    let mut cases = oracle_ext::quaternion_from_polar_decomposition_cases();
    let mut worst = 0;
    let mut index = 0;
    while let Some(case) = cases.pop_front() {
        let (scale, theta, axis, expected, tol) = *case;
        let got = QuaternionTranscendentalTrait::from_polar_decomposition(
            fx(scale), fx(theta), u3t(axis),
        );
        let err = max_ulp_diff_q(got, qt(expected));
        worst = core::cmp::max(worst, report("from_polar_decomposition", index, err, tol));
        index += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_cos_oracle() {
    let mut cases = oracle_ext::quaternion_cos_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("cos", n, q.cos(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_sin_oracle() {
    let mut cases = oracle_ext::quaternion_sin_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("sin", n, q.sin(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_tan_oracle() {
    let mut cases = oracle_ext::quaternion_tan_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("tan", n, q.tan(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_sinh_oracle() {
    let mut cases = oracle_ext::quaternion_sinh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("sinh", n, q.sinh(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_cosh_oracle() {
    let mut cases = oracle_ext::quaternion_cosh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("cosh", n, q.cosh(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_tanh_oracle() {
    let mut cases = oracle_ext::quaternion_tanh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("tanh", n, q.tanh(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_acos_oracle() {
    let mut cases = oracle_ext::quaternion_acos_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("acos", n, q.acos(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_asin_oracle() {
    let mut cases = oracle_ext::quaternion_asin_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("asin", n, q.asin(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_atan_oracle() {
    let mut cases = oracle_ext::quaternion_atan_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("atan", n, q.atan(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_asinh_oracle() {
    let mut cases = oracle_ext::quaternion_asinh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("asinh", n, q.asinh(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_acosh_oracle() {
    let mut cases = oracle_ext::quaternion_acosh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("acosh", n, q.acosh(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_atanh_oracle() {
    let mut cases = oracle_ext::quaternion_atanh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let q = qt(q);
        worst = core::cmp::max(worst, check_q("atanh", n, q.atanh(), e, tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- algebra: exact cases and identities

#[test]
fn test_magnitude_is_norm() {
    assert!(a().magnitude() == a().norm());
    assert!(a().magnitude_squared() == int(30));
}

#[test]
fn test_is_pure_and_pure() {
    assert!(!a().is_pure());
    assert!(a().pure() == qi(0, 2, -3, 4));
    assert!(a().pure().is_pure());
}

#[test]
fn test_cast_is_the_identity_on_fixed() {
    let c: Quaternion<Fixed> = a().cast();
    assert!(c == a());
}

/// `squared` is the Hamilton product `q * q` bit for bit (the cross product vanishes exactly).
#[test]
fn test_squared_matches_the_hamilton_product() {
    assert!(a().squared() == a() * a());
    assert!(s().squared() == s() * s());
    // (1 + 2i - 3j + 4k)² = (1 - 29, 4i - 6j + 8k).
    assert!(a().squared() == qi(-28, 4, -6, 8));
    let mut cases = oracle_ext::quaternion_sqrt_cases();
    while let Some(case) = cases.pop_front() {
        let (q, _, _) = *case;
        if max_ulp_diff_q(qt(q), Zero::zero()) < 0x100000000000 {
            assert!(qt(q).squared() == qt(q) * qt(q));
        }
    }
}

/// `half` rounds to nearest, ties to even: `3 ulp / 2 = 2 ulp`, `5 ulp / 2 = 2 ulp`.
#[test]
fn test_half_rounds_to_nearest_even() {
    assert!(qi(2, -4, 6, 1).half() == qt((ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW, HALF_RAW)));
    assert!(qt((3, 5, -3, -5)).half() == qt((2, 2, -2, -2)));
}

/// The loser: multiplying by `1/2` floors, which differs from upstream's `/ 2` on odd raws.
#[test]
fn test_half_alt_scale_half_floors_odd_raws() {
    let q = qt((3, 5, -3, -5));
    assert!(q.scale(Real::HALF) == qt((1, 2, -2, -3)));
    assert!(q.scale(Real::HALF) != q.half());
}

/// `inner + outer = a * b`: exact for the real part, within 1 ulp for the imaginary one (three
/// floored kernels against one).
#[test]
fn test_inner_plus_outer_is_the_product() {
    let (x, y) = (a(), b());
    assert!((x.inner(y) + x.outer(y)).abs_diff_eq(x * y, 1));
    assert!(x.inner(y) == y.inner(x));
    assert!(x.outer(y) == -y.outer(x));
    assert!(x.outer(x) == Zero::zero());
    // Integer inputs: every product is exact.
    assert!(x.inner(y) + x.outer(y) == x * y);
}

#[test]
fn test_divisions_invert_the_product() {
    let (x, y) = (a(), b());
    assert!((x * y).right_div(y).unwrap().abs_diff_eq(x, 2));
    assert!((y * x).left_div(y).unwrap().abs_diff_eq(x, 2));
    assert!(x.right_div(x).unwrap().abs_diff_eq(QuaternionTrait::identity(), 1));
    assert!(x.left_div(x).unwrap().abs_diff_eq(QuaternionTrait::identity(), 1));
}

#[test]
fn test_divisions_by_zero_are_none() {
    assert!(a().right_div(Zero::zero()).is_none());
    assert!(a().left_div(Zero::zero()).is_none());
    assert!(a().project(Zero::zero()).is_none());
    assert!(a().reject(Zero::zero()).is_none());
}

/// `project + reject = self` (their sum is `(inner + outer) · b⁻¹ = a`).
#[test]
fn test_project_plus_reject_is_self() {
    let (x, y) = (a(), b());
    assert!((x.project(y).unwrap() + x.reject(y).unwrap()).abs_diff_eq(x, 4));
}

// --- sqrt

#[test]
fn test_sqrt_exact_cases() {
    // sqrt(2j) = 1 + j exactly.
    assert!(qi(0, 0, 2, 0).sqrt() == qi(1, 0, 1, 0));
    assert!(qi(0, 0, 2, 0).sqrt().squared().abs_diff_eq(qi(0, 0, 2, 0), 2));
    // sqrt(-2k) = 1 - k exactly.
    assert!(qi(0, 0, 0, -2).sqrt() == qi(1, 0, 0, -1));
}

/// Upstream's `powf(1/2)` is `NaN` on every real quaternion (its `ln` normalises the zero
/// imaginary part): the root panics there (PLAN M8 fidelity rules).
#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_sqrt_of_a_positive_real_quaternion_panics() {
    let _ = qi(4, 0, 0, 0).sqrt();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_sqrt_of_a_negative_real_quaternion_panics() {
    let _ = qi(-4, 0, 0, 0).sqrt();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_sqrt_of_zero_panics() {
    let _ = Zero::<Quaternion<Fixed>>::zero().sqrt();
}

/// `sqrt(q)² = q` on the oracle inputs, in both branches (`w >= 0` and `w < 0`).
#[test]
fn test_sqrt_squares_back() {
    let mut cases = oracle_ext::quaternion_sqrt_cases();
    let mut n: usize = 0;
    while let Some(case) = cases.pop_front() {
        let (q, _, _) = *case;
        // The small and unit cases: the error of the square grows with |q|.
        if n < 16 {
            assert!(qt(q).sqrt().squared().abs_diff_eq(qt(q), 8));
        }
        n += 1;
    }
}

/// The loser: upstream's `powf(1/2) = exp(ln(q) / 2)` gives the same principal root, to the
/// tolerance of the oracle, at nine times the cost (`bench_quaternion_sqrt__alt_powf`).
#[test]
fn test_sqrt_alt_powf_agrees() {
    let mut cases = oracle_ext::quaternion_sqrt_cases();
    let mut n: usize = 0;
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        if n < 16 {
            let alt = qt(q).powf(Real::HALF);
            assert!(check_q("sqrt_alt_powf", n, alt, e, tol + 64) == 0);
        }
        n += 1;
    }
}

// --- exp, ln, powf

#[test]
fn test_exp_exact_cases() {
    // e^0 = 1 exactly, and a pure quaternion below the threshold is the identity, like upstream.
    assert!(Zero::<Quaternion<Fixed>>::zero().exp() == QuaternionTrait::identity());
    assert!(qt((0, 1, 0, 0)).exp() == QuaternionTrait::identity());
    // exp(iπ/2) = i, to the accuracy of sin_cos.
    let q = qt((0, 6746518852, 0, 0));
    assert!(q.exp().abs_diff_eq(qi(0, 1, 0, 0), 4));
}

/// Deviation from upstream: the real quaternion `(1, 0, 0, 0)` has the exponential `(e, 0, 0,
/// 0)` (upstream returns the identity whenever `|v| <= eps`).
#[test]
fn test_exp_of_a_real_quaternion_is_real() {
    let e = qi(1, 0, 0, 0).exp();
    assert!(e.w == Real::e() || e.w.abs_diff_eq(Real::e(), 3));
    assert!(e.pure() == Zero::zero());
    // With a large threshold the imaginary part is ignored.
    assert!(qt((0, HALF_RAW, 0, 0)).exp_eps(fx(ONE_RAW)) == QuaternionTrait::identity());
}

#[test]
fn test_ln_exact_cases() {
    // ln(i) = iπ/2.
    assert!(
        qi(0, 1, 0, 0)
            .ln()
            .abs_diff_eq(Quaternion { i: Real::frac_pi_2(), j: fx(0), k: fx(0), w: fx(0) }, 4),
    );
}

/// Upstream normalises the zero imaginary part of a real quaternion (`NaN`): `ln` panics on
/// every real quaternion, the identity and `-1` included (PLAN M8 fidelity rules).
#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_ln_of_zero_panics() {
    let _ = Zero::<Quaternion<Fixed>>::zero().ln();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_ln_of_the_identity_panics() {
    let _ = QuaternionTrait::<Fixed>::identity().ln();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_ln_of_a_negative_real_panics() {
    let _ = qi(-1, 0, 0, 0).ln();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_powf_of_a_real_quaternion_panics() {
    let _ = qi(2, 0, 0, 0).powf(Real::TWO);
}

#[test]
#[should_panic(expected: ('Fixed: exp overflow',))]
fn test_exp_overflow_panics() {
    let _ = qi(22, 0, 1, 0).exp();
}

/// `exp(ln(q)) = q` and `ln(exp(q)) = q` (for |v| < π).
#[test]
fn test_exp_ln_round_trips() {
    assert!(s().ln().exp().abs_diff_eq(s(), 16));
    assert!(s().exp().ln().abs_diff_eq(s(), 16));
    assert!(a().ln().exp().abs_diff_eq(a(), 256));
}

#[test]
fn test_powf_identities() {
    assert!(s().powf(Real::one()).abs_diff_eq(s(), 16));
    assert!(s().powf(Real::zero()) == QuaternionTrait::identity());
    assert!(s().powf(Real::TWO).abs_diff_eq(s().squared(), 32));
    assert!(s().powf(Real::NEG_ONE).abs_diff_eq(s().try_inverse().unwrap(), 32));
}

/// The loser: upstream's `acos(w / |q|)` loses half the bits near the real axis, where
/// `atan2(|v|, w)` does not.
#[test]
fn test_ln_alt_acos_loses_precision_near_the_real_axis() {
    // q = 1 + 2^-20 i: ln q = (2^-41 ≈ 0, 2^-20 i).
    let q = qt((ONE_RAW, 0x1000, 0, 0));
    let exact = qt((0, 0x1000, 0, 0));
    assert!(q.ln().abs_diff_eq(exact, 1));
    let alt = alt_ln_acos(q);
    assert!(!alt.abs_diff_eq(exact, 64));
}

/// Upstream's `ln`: `θ = acos(w / |q|)`.
pub(crate) fn alt_ln_acos(q: Quaternion<Fixed>) -> Quaternion<Fixed> {
    let n = q.norm();
    let theta = simba::scalar::Transcendental::acos(Real::div(q.w, n));
    let v = q.vector();
    let nv = Real::norm3(v.x, v.y, v.z);
    let (x, y, z) = Real::div3(v.x, v.y, v.z, nv);
    Quaternion { i: x * theta, j: y * theta, k: z * theta, w: simba::scalar::Transcendental::ln(n) }
}

// --- polar decomposition

#[test]
fn test_polar_decomposition_round_trip() {
    let (n, theta, axis) = s().polar_decomposition();
    assert!(n == s().norm());
    let back = QuaternionTranscendentalTrait::from_polar_decomposition(n, theta, axis.unwrap());
    assert!(back.abs_diff_eq(s(), 8));
}

#[test]
fn test_polar_decomposition_degenerate_cases() {
    let (n, theta, axis) = Zero::<Quaternion<Fixed>>::zero().polar_decomposition();
    assert!(n == Real::zero() && theta == Real::zero() && axis.is_none());
    let (n, theta, axis) = qi(-3, 0, 0, 0).polar_decomposition();
    assert!(n == int(3) && theta == Real::zero() && axis.is_none());
    // A pure quaternion: half angle π/2.
    let (n, theta, axis) = qi(0, 0, 2, 0).polar_decomposition();
    assert!(n == int(2) && theta.abs_diff_eq(Real::frac_pi_2(), 4));
    assert!(axis.unwrap() == Unit { value: Vector3 { x: fx(0), y: int(1), z: fx(0) } });
}

// --- trigonometric and hyperbolic functions

/// `sin² + cos² = 1` and `cosh² - sinh² = 1` (q commutes with itself).
#[test]
fn test_trigonometric_identities() {
    let one = QuaternionTrait::<Fixed>::identity();
    let (sn, cs) = (s().sin(), s().cos());
    assert!((sn.squared() + cs.squared()).abs_diff_eq(one, 16));
    let (sh, ch) = (s().sinh(), s().cosh());
    assert!((ch.squared() - sh.squared()).abs_diff_eq(one, 16));
    assert!(s().tan().abs_diff_eq(sn.right_div(cs).unwrap(), 0));
    assert!(s().tanh().abs_diff_eq(sh.right_div(ch).unwrap(), 0));
}

/// The inverse functions invert on a small quaternion.
#[test]
fn test_inverse_functions_invert() {
    let q = qt((0x20000000, 0x30000000, -0x10000000, 0x18000000));
    assert!(q.sin().asin().abs_diff_eq(q, 128));
    assert!(q.cos().acos().abs_diff_eq(q, 128));
    assert!(q.tan().atan().abs_diff_eq(q, 128));
    assert!(q.sinh().asinh().abs_diff_eq(q, 128));
    assert!(q.tanh().atanh().abs_diff_eq(q, 128));
}

/// `sinh` / `cosh` of a real quaternion are the scalar functions.
#[test]
fn test_hyperbolic_of_a_real_quaternion() {
    let sh = qi(1, 0, 0, 0).sinh();
    let ch = qi(1, 0, 0, 0).cosh();
    // sinh 1 = 1.1752011936, cosh 1 = 1.5430806348.
    assert!(sh.abs_diff_eq(qt((5047450692, 0, 0, 0)), 4));
    assert!(ch.abs_diff_eq(qt((6627480861, 0, 0, 0)), 4));
    assert!(Zero::<Quaternion<Fixed>>::zero().sin() == Zero::zero());
    assert!(Zero::<Quaternion<Fixed>>::zero().cos() == QuaternionTrait::identity());
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_acos_of_a_real_quaternion_panics() {
    let _ = qi(2, 0, 0, 0).acos();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_asin_of_a_real_quaternion_panics() {
    let _ = qi(-1, 0, 0, 0).asin();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_atan_of_a_real_quaternion_panics() {
    let _ = qi(1, 0, 0, 0).atan();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_asinh_of_a_real_quaternion_panics() {
    let _ = qi(1, 0, 0, 0).asinh();
}

#[test]
#[should_panic(expected: ('nalgebra: real quaternion (NaN)',))]
fn test_atanh_of_one_panics() {
    let _ = QuaternionTrait::<Fixed>::identity().atanh();
}

// --- approximate comparisons

#[test]
fn test_relative_eq() {
    let x = qi(1000, 0, -1000, 1);
    let y = Quaternion {
        i: fx(0), j: fx(-1000 * ONE_RAW - 100), k: fx(ONE_RAW), w: fx(1000 * ONE_RAW + 100),
    };
    // 100 ulp apart on components of magnitude 1000: relatively 2.3e-11.
    assert!(!x.relative_eq(y, 99, Real::zero()));
    assert!(x.relative_eq(y, 100, Real::zero()));
    assert!(x.relative_eq(y, 0, fx(1)));
    assert!(x.relative_eq(x, 0, Real::zero()));
    // The double cover, like upstream: `-q` is accepted, component-wise against `-other`.
    assert!(x.relative_eq(-x, 0, Real::zero()));
    assert!(x.relative_eq(-y, 100, Real::zero()));
    assert!(!x.relative_eq(-y, 99, Real::zero()));
    // Half of the components flipped is neither `q` nor `-q`.
    let z = Quaternion { i: x.i, j: x.j, k: -x.k, w: -x.w };
    assert!(!x.relative_eq(z, 0, fx(ONE_RAW / 2)));
}

#[test]
fn test_ulps_eq() {
    let x = qt((5, -5, 0, 1));
    let y = qt((7, -3, 0, 1));
    assert!(x.ulps_eq(y, 2, 0));
    assert!(!x.ulps_eq(y, 1, 1));
    assert!(x.ulps_eq(y, 0, 2));
    // The ulp budget does not cross zero; the absolute one does.
    let (p, m) = (qt((1, 0, 0, 0)), qt((-1, 0, 0, 0)));
    assert!(p.ulps_eq(m, 2, 0));
    // ... but `-q` compares equal (the double cover, like upstream).
    assert!(p.ulps_eq(m, 0, 0));
    assert!(x.ulps_eq(-y, 0, 2));
    assert!(!x.ulps_eq(-y, 1, 1));
    let z = qt((5, -5, 0, -1));
    assert!(!x.ulps_eq(z, 1, 1));
}

// --- trait impls

#[test]
fn test_index_is_the_storage_order() {
    let mut x = a();
    assert!(x[0] == int(2) && x[1] == int(-3) && x[2] == int(4) && x[3] == int(1));
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_index_out_of_bounds_panics() {
    let mut x = a();
    let _ = x[4];
}

#[test]
fn test_one() {
    let one: Quaternion<Fixed> = One::one();
    assert!(one == QuaternionTrait::identity());
    assert!(one.is_one() && !one.is_non_one());
    assert!(!a().is_one());
}

#[test]
fn test_from_array_is_the_storage_order() {
    let x: Quaternion<Fixed> = [int(2), int(-3), int(4), int(1)].into();
    assert!(x == a());
}

// --- the losers of `benches_ext.cairo`

/// Upstream's `(a * b + b * a) / 2` rounds two Hamilton products and halves: within 1 ulp of the
/// reduced form, for twice the products.
#[test]
fn test_inner_alt_products_agrees() {
    let t = qt((-0x40000000, 0x80000000, 0x20000000, -0x60000000));
    assert!(super::benches_ext::alt_inner_products(s(), t).abs_diff_eq(s().inner(t), 1));
    assert!(super::benches_ext::alt_inner_products(a(), b()) == a().inner(b()));
}

/// Upstream's `(exp(q) - exp(-q)) / 2` agrees with the closed form to a few ulp.
#[test]
fn test_sinh_alt_exp_difference_agrees() {
    assert!(super::benches_ext::alt_sinh_exp_difference(s()).abs_diff_eq(s().sinh(), 8));
    let mut cases = oracle_ext::quaternion_sinh_cases();
    while let Some(case) = cases.pop_front() {
        let (q, _, tol) = *case;
        let alt = super::benches_ext::alt_sinh_exp_difference(qt(q));
        assert!(alt.abs_diff_eq(qt(q).sinh(), tol));
    }
}

/// `right_div` divides the fused product once; upstream's `a * b⁻¹` multiplies by the rounded
/// inverse. They agree to a few ulp on moderate inputs.
#[test]
fn test_right_div_alt_inverse_then_mul_agrees() {
    let t = qt((-0x40000000, 0x80000000, 0x20000000, -0x60000000));
    let alt = s() * t.try_inverse().unwrap();
    assert!(alt.abs_diff_eq(s().right_div(t).unwrap(), 4));
}

/// Why `right_div` keeps the fused form although upstream's `a * b⁻¹` is cheaper on the bench
/// inputs (27 630 against 34 020 gas): the rounded inverse carries half an ulp per component,
/// which the product then multiplies by `|a|`. With `b = 1 + i + j` (`|b|² = 3` exact, `b⁻¹ =
/// (1 - i - j) / 3` not representable) and `a` of norm ~2 300, `3 · (a / b)` must be the integer
/// quaternion `a · conj(b)`: the fused form is within 2 ulp of it, the loser thousands of ulp.
#[test]
fn test_right_div_alt_inverse_then_mul_loses_bits_on_large_quotients() {
    let x = qi(1000, 2000, -500, 300);
    let y = qi(1, 1, 1, 0);
    let exact = x * y.conjugate();
    let fused = x.right_div(y).unwrap().scale(int(3));
    let alt = (x * y.try_inverse().unwrap()).scale(int(3));
    let fused_err = max_ulp_diff_q(fused, exact);
    let alt_err = max_ulp_diff_q(alt, exact);
    println!("right_div: fused {fused_err} ulp, inverse then mul {alt_err} ulp");
    assert!(fused_err <= 2);
    assert!(alt_err > 1000);
}
