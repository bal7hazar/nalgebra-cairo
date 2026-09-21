//! Unit tests of `Lu6`: an exactly representable factorisation, the identities (`P A = L U`, `A
//! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
//! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
//! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use crate::base::matrix6::{Matrix6, Matrix6Trait};
use crate::base::matrix_test_utils::{
    abs_raw, fx, int, m6, max_abs_m6, max_abs_v6, max_ulp_diff6, max_ulp_diff_v6, oracle_tol,
    ulp_diff, v6it, v6t,
};
use crate::base::vector6::Vector6;
use crate::linalg::lu::{Perm6, PermTrait, oracle_lu6 as oracle};
use super::benches::{new_no_pivot, solve_recip, try_inverse_recip, try_inverse_solve_columns};
use super::{Lu6, Lu6Trait, Matrix6LuTrait};

/// The oracle's first `unit` 6x6 case whose factorisation actually swaps rows (all the benchmarks
/// share it, so their inputs exercise the permutation).
fn a_bench() -> Matrix6<Fixed> {
    m6(
        [
            [-2527254097, 4325213708, -1213189640, 4008510819, 2405075077, 466428599],
            [-700750028, -3536112306, 1731465811, 244560054, 2632174365, 1874080084],
            [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
            [-617580144, 580405440, -3475076872, -924247759, -985644408, 938584824],
            [-1076219818, -1905997652, -1595683778, 2067806730, -3245944919, 3717734456],
            [-308280913, -120179493, 2005545225, -1090342151, 932840990, 2332476429],
        ],
    )
}

/// Its right-hand side.
fn b_bench() -> Vector6<Fixed> {
    v6t((3579353502, 7767965432, 6840630971, -6086016248, -4735434050, -2809324573))
}

/// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay for
/// `new`.
fn f_bench() -> Lu6<Fixed> {
    Lu6 {
        lu: m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [2625460419, 4958875604, -941745570, 5107545028, 4074451472, 218945968],
                [641578630, 636814517, -3269112063, -1412974282, -1181820813, 845644872],
                [1118040539, -1417102436, 2352776611, 4995058989, -543304442, 3221342200],
                [727980405, -2910510776, -1535245700, 3014081670, 5814961885, -4536658],
                [320260371, -37142392, -2667692283, -1538893104, 179474863, 3983829884],
            ],
        ),
        p: Perm6 { p1: 3, p2: 3, p3: 4, p4: 5, p5: 5 },
    }
}

/// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
/// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle, which
/// partial pivoting undoes exactly because `|l_ik| < 1`.
fn a_exact() -> Matrix6<Fixed> {
    m6(
        [
            [-3221225472, 6442450944, -10737418240, 9663676416, -25769803776, 21474836480],
            [9663676416, -7516192768, -10737418240, 8589934592, -24696061952, 22548578304],
            [-9663676416, 11811160064, 34359738368, -6442450944, 0, 12884901888],
            [-3221225472, 6442450944, 15032385536, -4294967296, -30064771072, 31138512896],
            [-3221225472, 7516192768, 19327352832, 8589934592, -22548578304, 16106127360],
            [-12884901888, 12884901888, 8589934592, -17179869184, -4294967296, 12884901888],
        ],
    )
}

/// An exactly singular integer matrix (one row is an integer combination of the others, and every
/// multiplier of the elimination is dyadic, so the last pivot is exactly zero).
fn a_singular() -> Matrix6<Fixed> {
    m6(
        [
            [4294967296, 0, -25769803776, -25769803776, 21474836480, 25769803776],
            [17179869184, 0, -8589934592, 0, 4294967296, 4294967296],
            [-8589934592, -8589934592, -12884901888, 25769803776, 25769803776, -17179869184],
            [-17179869184, 0, -4294967296, 25769803776, -17179869184, 8589934592],
            [0, -8589934592, 17179869184, 103079215104, -30064771072, -12884901888],
            [12884901888, 0, 17179869184, 25769803776, -21474836480, 17179869184],
        ],
    )
}

/// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
/// rejection path measures the rejection and not `new`.
fn f_singular() -> Lu6<Fixed> {
    Lu6 {
        lu: m6(
            [
                [17179869184, 0, -8589934592, 0, 4294967296, 4294967296],
                [-2147483648, -8589934592, -17179869184, 25769803776, 27917287424, -15032385536],
                [0, 4294967296, 34359738368, 77309411328, -57982058496, 2147483648],
                [-4294967296, 0, -1610612736, 54760833024, -34628173824, 13690208256],
                [1073741824, 0, -2952790016, 2147483648, -2147483648, 19327352832],
                [3221225472, 0, 2952790016, -2147483648, 4294967296, 0],
            ],
        ),
        p: Perm6 { p1: 2, p2: 3, p3: 5, p4: 4, p5: 5 },
    }
}

/// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped (one
/// division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the UNPIVOTED
/// factorisation.
fn solve_failures(variant: u8) -> (u32, u128) {
    let mut cases = oracle::lu6_solve_cases();
    let mut failures = 0;
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let f = if variant == 2 {
            new_no_pivot(m6(a))
        } else {
            Lu6Trait::new(m6(a))
        };
        let got = if variant == 1 {
            solve_recip(f, v6t(b))
        } else {
            f.solve(v6t(b))
        };
        let e = v6t(expected);
        let err = max_ulp_diff_v6(got.unwrap(), e);
        if err > oracle_tol(max_abs_v6(e), tol) {
            failures += 1;
        }
        worst = core::cmp::max(worst, err);
    }
    (failures, worst)
}

#[test]
fn test_new_is_exact_on_a_dyadic_matrix() {
    let f = Lu6Trait::new(a_exact());
    let lu = f.lu;
    assert!(
        lu == m6(
            [
                [-12884901888, 12884901888, 8589934592, -17179869184, -4294967296, 12884901888],
                [1073741824, 4294967296, 17179869184, 12884901888, -21474836480, 12884901888],
                [1073741824, 3221225472, -25769803776, 4294967296, -8589934592, 8589934592],
                [-3221225472, 2147483648, 2147483648, -12884901888, -12884901888, 21474836480],
                [3221225472, 2147483648, -3221225472, -1073741824, 4294967296, 8589934592],
                [1073741824, 3221225472, 0, 3221225472, -3221225472, 8589934592],
            ],
        ),
    );
    assert!(f.p() == Perm6 { p1: 6, p2: 5, p3: 6, p4: 5, p5: 6 });
    assert!(f.permute_rows(a_exact()) == f.l() * f.u());
    assert!(f.determinant() == fx(463856467968));
}

#[test]
fn test_new_reconstruction_oracle() {
    // `P A == L U` within 40 raw units on the lu6 vectors.
    let mut cases = oracle::lu6_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        let f = Lu6Trait::new(m6(a));
        worst = core::cmp::max(worst, max_ulp_diff6(f.permute_rows(m6(a)), f.l() * f.u()));
    }
    assert!(worst == 40, "reconstruction error {worst}");
}

#[test]
fn test_new_multipliers_are_bounded_by_one() {
    // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
    let mut cases = oracle::lu6_solve_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        assert!(max_abs_m6(Lu6Trait::new(m6(a)).l()) <= 0x100000000);
    }
}

#[test]
fn test_factors_permutation_and_accessors() {
    let f = Lu6Trait::new(a_exact());
    let l = f.l();
    assert!(
        l == m6(
            [
                [4294967296, 0, 0, 0, 0, 0], [1073741824, 4294967296, 0, 0, 0, 0],
                [1073741824, 3221225472, 4294967296, 0, 0, 0],
                [-3221225472, 2147483648, 2147483648, 4294967296, 0, 0],
                [3221225472, 2147483648, -3221225472, -1073741824, 4294967296, 0],
                [1073741824, 3221225472, 0, 3221225472, -3221225472, 4294967296],
            ],
        ),
    );
    let u = f.u();
    assert!(
        u == m6(
            [
                [-12884901888, 12884901888, 8589934592, -17179869184, -4294967296, 12884901888],
                [0, 4294967296, 17179869184, 12884901888, -21474836480, 12884901888],
                [0, 0, -25769803776, 4294967296, -8589934592, 8589934592],
                [0, 0, 0, -12884901888, -12884901888, 21474836480],
                [0, 0, 0, 0, 4294967296, 8589934592], [0, 0, 0, 0, 0, 8589934592],
            ],
        ),
    );
    let pa = f.permute_rows(a_exact());
    assert!(
        pa == m6(
            [
                [-12884901888, 12884901888, 8589934592, -17179869184, -4294967296, 12884901888],
                [-3221225472, 7516192768, 19327352832, 8589934592, -22548578304, 16106127360],
                [-3221225472, 6442450944, -10737418240, 9663676416, -25769803776, 21474836480],
                [9663676416, -7516192768, -10737418240, 8589934592, -24696061952, 22548578304],
                [-9663676416, 11811160064, 34359738368, -6442450944, 0, 12884901888],
                [-3221225472, 6442450944, 15032385536, -4294967296, -30064771072, 31138512896],
            ],
        ),
    );
    assert!(f.permute(v6it((1, 2, 3, 4, 5, 6))) == v6it((6, 5, 1, 2, 3, 4)));
    // the identity factors without a single swap
    let id = Lu6Trait::new(Matrix6Trait::<Fixed>::identity());
    assert!(id.p() == PermTrait::identity6());
    assert!(id.permute(b_bench()) == b_bench());
    assert!(id.permute_rows(a_bench()) == a_bench());
    assert!(id.l() == Matrix6Trait::<Fixed>::identity());
    assert!(id.u() == Matrix6Trait::<Fixed>::identity());
}

#[test]
fn test_singular_is_rejected() {
    let f = Lu6Trait::new(a_singular());
    assert!(!f.is_invertible());
    assert!(f.solve(b_bench()).is_none());
    assert!(f.try_inverse().is_none());
    assert!(f.determinant() == int(0));
    let z = Lu6Trait::new(Matrix6Trait::<Fixed>::zeros());
    assert!(!z.is_invertible());
    assert!(z.try_inverse().is_none());
    assert!(z.determinant() == int(0));
    assert!(z.p() == PermTrait::identity6());
    assert!(Lu6Trait::new(a_bench()).is_invertible());
}

#[test]
fn test_solve_oracle() {
    let mut cases = oracle::lu6_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Lu6Trait::new(m6(a)).solve(v6t(b)).unwrap();
        let e = v6t(expected);
        let err = max_ulp_diff_v6(x, e);
        assert!(err <= oracle_tol(max_abs_v6(e), tol), "solve error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 8238);
}

#[test]
fn test_solve_near_singular_oracle() {
    // Condition number 1e2..1e4: the oracle flags these as behaviour, not
    // precision, tests. The tolerances are large by construction.
    let mut cases = oracle::lu6_solve_near_singular_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Lu6Trait::new(m6(a)).solve(v6t(b)).unwrap();
        let e = v6t(expected);
        let err = max_ulp_diff_v6(x, e);
        assert!(err <= oracle_tol(max_abs_v6(e), tol), "solve error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 460827);
}

#[test]
fn test_try_inverse_oracle() {
    let mut cases = oracle::lu6_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let inv = Lu6Trait::new(m6(a)).try_inverse().unwrap();
        let e = m6(expected);
        let err = max_ulp_diff6(inv, e);
        assert!(err <= oracle_tol(max_abs_m6(e), tol), "inverse error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 1013);
}

#[test]
fn test_try_inverse_product_is_identity() {
    let mut cases = oracle::lu6_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let inv = Lu6Trait::new(m6(a)).try_inverse().unwrap();
        // Worst residual over the oracle: 100 ulp.
        assert!((m6(a) * inv).is_identity(100));
        assert!((inv * m6(a)).is_identity(100));
    }
}

#[test]
fn test_try_inverse_candidates() {
    // The static unit right-hand sides give BIT-IDENTICAL results to 6 calls
    // to `solve`; the saving is pure gas.
    let mut cases = oracle::lu6_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let f = Lu6Trait::new(m6(a));
        assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
        worst =
            core::cmp::max(
                worst, max_ulp_diff6(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
            );
    }
    // ... and the reciprocal variant drifts by at most this many ulp from it.
    assert!(worst == 3);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_try_inverse_overflow_panics() {
    // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
    let _ = black_box(Matrix6Trait::from_diagonal_element(fx(1))).lu().try_inverse();
}

#[test]
fn test_determinant_oracle() {
    let mut cases = oracle::lu6_determinant_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let det = Lu6Trait::new(m6(a)).determinant();
        let err = ulp_diff(det, fx(expected));
        assert!(err <= oracle_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 93);
}

#[test]
fn test_determinant_exact_and_sign() {
    assert!(Lu6Trait::new(Matrix6Trait::<Fixed>::identity()).determinant() == int(1));
    let d = Matrix6Trait::from_diagonal(v6it((2, -4, 3, -1, 5, -2)));
    assert!(Lu6Trait::new(d).determinant() == int(-240));
    // Swapping two rows flips the sign exactly (the factorisation is exact here,
    // and the sign is applied to the first pivot before any rounding).
    let s = m6(
        [
            [9663676416, -7516192768, -10737418240, 8589934592, -24696061952, 22548578304],
            [-3221225472, 6442450944, -10737418240, 9663676416, -25769803776, 21474836480],
            [-9663676416, 11811160064, 34359738368, -6442450944, 0, 12884901888],
            [-3221225472, 6442450944, 15032385536, -4294967296, -30064771072, 31138512896],
            [-3221225472, 7516192768, 19327352832, 8589934592, -22548578304, 16106127360],
            [-12884901888, 12884901888, 8589934592, -17179869184, -4294967296, 12884901888],
        ],
    );
    assert!(Lu6Trait::new(s).determinant() == -fx(463856467968));
}

#[test]
fn test_solve_candidates_error() {
    // Oracle, 9 well-conditioned matrices: (cases above the LU tolerance,
    // worst error in ulp) of the shipped divisions, of one reciprocal per
    // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
    // The unpivoted figure is NOT a win: these matrices are random and
    // well-conditioned, so their leading entries happen to be usable pivots.
    // `test_no_pivot_candidate_is_wrong` shows the structural failure.
    assert!(solve_failures(0) == (0, 8238));
    assert!(solve_failures(1) == (1, 8237));
    assert!(solve_failures(2) == (0, 7754));
}

#[test]
fn test_no_pivot_candidate_is_wrong() {
    // A permuted identity: the unpivoted elimination finds a zero at `a11` and
    // gives up, where partial pivoting factors it exactly.
    let swapped = m6(
        [
            [0, 4294967296, 0, 0, 0, 0], [4294967296, 0, 0, 0, 0, 0], [0, 0, 4294967296, 0, 0, 0],
            [0, 0, 0, 4294967296, 0, 0], [0, 0, 0, 0, 4294967296, 0], [0, 0, 0, 0, 0, 4294967296],
        ],
    );
    assert!(!new_no_pivot(swapped).is_invertible());
    let f = Lu6Trait::new(swapped);
    assert!(f.is_invertible());
    assert!(f.determinant() == int(-1));
}
