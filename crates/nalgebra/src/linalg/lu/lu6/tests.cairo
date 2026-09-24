//! Unit tests of `Lu6`: an exactly representable factorisation, the identities (`P A = L U`, `A
//! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
//! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
//! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_linalg/src/lu/lu6/tests.cairo`.

use fixed::Fixed;
use crate::base::matrix6::{Matrix6, Matrix6Trait};
use crate::base::matrix_test_utils::{
    fx, m6, max_abs_v6, max_ulp_diff6, max_ulp_diff_v6, oracle_tol, v6it, v6t,
};
use crate::base::vector6::Vector6;
use crate::linalg::lu::{Perm6, Perm6Trait, oracle_lu6 as oracle};
use super::benches::{new_no_pivot, solve_recip};
use super::{Lu6InternalTrait, Lu6Trait};

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
    assert!(worst == 31, "reconstruction error {worst}");
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
    assert!(id.p() == Perm6Trait::identity());
    assert!(id.permute(b_bench()) == b_bench());
    assert!(id.permute_rows(a_bench()) == a_bench());
    assert!(id.l() == Matrix6Trait::<Fixed>::identity());
    assert!(id.u() == Matrix6Trait::<Fixed>::identity());
}

#[test]
fn test_solve_candidates_error() {
    // Oracle, 9 well-conditioned matrices: (cases above the LU tolerance,
    // worst error in ulp) of the shipped divisions, of one reciprocal per
    // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
    // The unpivoted figure is NOT a win: these matrices are random and
    // well-conditioned, so their leading entries happen to be usable pivots.
    // `test_no_pivot_candidate_is_wrong` shows the structural failure.
    assert!(solve_failures(0) == (0, 8212));
    assert!(solve_failures(1) == (1, 8213));
    assert!(solve_failures(2) == (0, 7906));
}
