//! `Lu2`: the LU factorisation with partial pivoting of a `Matrix2` (upstream
//! `nalgebra::linalg::LU` on a 2x2 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 1 row transpositions chosen by partial pivoting. Both factors share one `Matrix2` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm2`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 1 `abs` and comparisons plus 1 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu2_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg4::internal::linalg::lu::lu2::Lu2InternalTrait;
pub use nalgebra_linalg4::linalg::lu::lu2::*;
#[cfg(test)]
use super::Perm2PartialEq;

/// Test-only field-wise equality (upstream `Lu2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Lu2PartialEq<T, +PartialEq<T>> of PartialEq<Lu2<T>> {
    fn eq(lhs: @Lu2<T>, rhs: @Lu2<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Lu2`: an exactly representable factorisation, the identities (`P A = L U`, `A
    //! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
    //! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
    //!
    //! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
    //! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.
    //!
    //! Gas benchmarks of `Lu2` (`bench_lu2_<op>__<variant>`, net = raw - the `baseline` of the
    //! group), and the alternative implementations that lost, kept as evidence together with the
    //! tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong
    //! on a matrix as ordinary as a permuted identity.
    //!
    //! - `alt_recip`: one reciprocal per pivot instead of one correctly rounded division per output
    //! scalar. DEARER for `solve`, where a single right-hand side does not amortise the reciprocal,
    //! cheaper for `try_inverse`, where 2 columns share it, and a second rounding per output in
    //! both.
    //!
    //! - `alt_solve_columns`: the inverse as 2 calls to `solve`. Bit-identical, dearer.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/lu/lu2/tests.cairo`.

    use fixed::Fixed;
    use simba::scalar::Real;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        fx, m2, max_abs_v2, max_ulp_diff2, max_ulp_diff_v2, oracle_tol, v2it, v2t,
    };
    use crate::base::vector2::Vector2;
    use crate::linalg::lu::{Perm2, Perm2PartialEq, Perm2Trait, oracle_lu2 as oracle};
    use crate::testing::black_box;
    use super::{Lu2, Lu2InternalTrait, Lu2Trait};

    /// The oracle's first `unit` 2x2 case whose factorisation actually swaps rows, so every
    /// benchmark exercises the permutation.
    fn a_bench() -> Matrix2<Fixed> {
        m2([[-277028774, 4364373136], [3058251633, 1932838604]])
    }

    /// Its right-hand side.
    fn b_bench() -> Vector2<Fixed> {
        v2t((-5886581674, -6536196560))
    }

    /// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay
    /// for `new`.
    fn f_bench() -> Lu2<Fixed> {
        Lu2 { lu: m2([[3058251633, 1932838604], [-389055470, 4539457456]]), p: Perm2 { p1: 2 } }
    }

    /// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
    /// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle,
    /// which partial pivoting undoes exactly because `|l_ik| < 1`.
    fn a_exact() -> Matrix2<Fixed> {
        m2([[-3221225472, 16106127360], [12884901888, 4294967296]])
    }

    /// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
    /// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it
    /// is wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
    /// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what
    /// keeps `U` from growing. Upstream has no unpivoted variant either.
    fn new_no_pivot(matrix: Matrix2<Fixed>) -> Lu2<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        if a11 != Real::zero() {
            let l = a21 / a11;
            let nl = -l;
            a22 = Real::mul_add(nl, a12, a22);
            a21 = l;
        }
        Lu2 { lu: Matrix2 { m11: a11, m21: a21, m12: a12, m22: a22 }, p: Perm2 { p1: 1 } }
    }

    /// `solve` with ONE reciprocal per pivot and 2 multiplications instead of 2 correctly rounded
    /// divisions.
    /// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
    /// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to
    /// amortise it over, so it costs 16 760 against 14 240 gas; and rounding `1 / u_ii` before
    /// using it puts 4 of the oracle cases outside the tolerance against 0
    /// (`test_solve_candidates_error`).
    fn solve_recip(f: Lu2<Fixed>, b: Vector2<Fixed>) -> Option<Vector2<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let pb = f.permute(b);
        let y1 = pb.x;
        let y2 = Real::mul_add(-f.lu.m21, y1, pb.y);
        let x2 = y2 * r2;
        let x1 = Real::mul_add(-f.lu.m12, x2, y1) * r1;
        Some(Vector2 { x: x1, y: x2 })
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped
    /// (one division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the
    /// UNPIVOTED factorisation.
    fn solve_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle::lu2_solve_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let f = if variant == 2 {
                new_no_pivot(m2(a))
            } else {
                Lu2Trait::new(m2(a))
            };
            let got = if variant == 1 {
                solve_recip(f, v2t(b))
            } else {
                f.solve(v2t(b))
            };
            let e = v2t(expected);
            let err = max_ulp_diff_v2(got.unwrap(), e);
            if err > oracle_tol(max_abs_v2(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    #[test]
    fn test_new_is_exact_on_a_dyadic_matrix() {
        let f = Lu2Trait::new(a_exact());
        let lu = f.lu;
        assert!(lu == m2([[12884901888, 4294967296], [-1073741824, 17179869184]]));
        assert!(f.p() == Perm2 { p1: 2 });
        assert!(f.permute_rows(a_exact()) == f.l() * f.u());
        assert!(f.determinant() == fx(-51539607552));
    }

    #[test]
    fn test_new_reconstruction_oracle() {
        // `P A == L U` within 34 raw units on the lu2 vectors.
        let mut cases = oracle::lu2_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Lu2Trait::new(m2(a));
            worst = core::cmp::max(worst, max_ulp_diff2(f.permute_rows(m2(a)), f.l() * f.u()));
        }
        assert!(worst == 19, "reconstruction error {worst}");
    }

    #[test]
    fn test_factors_permutation_and_accessors() {
        let f = Lu2Trait::new(a_exact());
        let l = f.l();
        assert!(l == m2([[4294967296, 0], [-1073741824, 4294967296]]));
        let u = f.u();
        assert!(u == m2([[12884901888, 4294967296], [0, 17179869184]]));
        let pa = f.permute_rows(a_exact());
        assert!(pa == m2([[12884901888, 4294967296], [-3221225472, 16106127360]]));
        assert!(f.permute(v2it((1, 2))) == v2it((2, 1)));
        // the identity factors without a single swap
        let id = Lu2Trait::new(Matrix2Trait::<Fixed>::identity());
        assert!(id.p() == Perm2Trait::identity());
        assert!(id.permute(b_bench()) == b_bench());
        assert!(id.permute_rows(a_bench()) == a_bench());
        assert!(id.l() == Matrix2Trait::<Fixed>::identity());
        assert!(id.u() == Matrix2Trait::<Fixed>::identity());
    }

    #[test]
    fn test_solve_candidates_error() {
        // Oracle, 18 well-conditioned matrices: (cases above the LU tolerance,
        // worst error in ulp) of the shipped divisions, of one reciprocal per
        // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
        // The unpivoted figure is NOT a win: these matrices are random and
        // well-conditioned, so their leading entries happen to be usable pivots.
        // `test_no_pivot_candidate_is_wrong` shows the structural failure.
        assert!(solve_failures(0) == (0, 122));
        assert!(solve_failures(1) == (2, 396));
        assert!(solve_failures(2) == (0, 79));
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_permute__transpositions() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(v2t((-6536196560, -5886581674)));
        assert!(f.permute(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_permute_rows__transpositions() {
        let f = black_box(f_bench());
        let a = black_box(a_bench());
        let e = black_box(m2([[3058251633, 1932838604], [-277028774, 4364373136]]));
        assert!(f.permute_rows(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_solve__alt_recip() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v2t((-5305313723, -6129723448))));
        assert!(solve_recip(f, b) == e);
    }
}
