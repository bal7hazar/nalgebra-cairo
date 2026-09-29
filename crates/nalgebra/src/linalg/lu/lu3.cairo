//! `Lu3`: the LU factorisation with partial pivoting of a `Matrix3` (upstream
//! `nalgebra::linalg::LU` on a 3x3 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 2 row transpositions chosen by partial pivoting. Both factors share one `Matrix3` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm3`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 3 `abs` and comparisons plus 2 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu3_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

pub use nalgebra_linalg4::linalg::lu::lu3::*;
#[cfg(test)]
use super::Perm3PartialEq;

/// Test-only field-wise equality (upstream `Lu3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Lu3PartialEq<T, +PartialEq<T>> of PartialEq<Lu3<T>> {
    fn eq(lhs: @Lu3<T>, rhs: @Lu3<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Lu3`: an exactly representable factorisation, the identities (`P A = L U`, `A
    //! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
    //! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
    //!
    //! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
    //! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.
    //!
    //! Gas benchmarks of `Lu3` (`bench_lu3_<op>__<variant>`, net = raw - the `baseline` of the
    //! group), and the alternative implementations that lost, kept as evidence together with the
    //! tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong
    //! on a matrix as ordinary as a permuted identity.
    //!
    //! - `alt_recip`: one reciprocal per pivot instead of one correctly rounded division per output
    //! scalar. DEARER for `solve`, where a single right-hand side does not amortise the reciprocal,
    //! cheaper for `try_inverse`, where 3 columns share it, and a second rounding per output in
    //! both.
    //!
    //! - `alt_solve_columns`: the inverse as 3 calls to `solve`. Bit-identical, dearer.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/lu/lu3/tests.cairo`.

    use fixed::Fixed;
    use simba::scalar::Real;
    use crate::base::matrix3::{Matrix3, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        fx, m3, max_abs_v3, max_ulp_diff3, max_ulp_diff_v3, oracle_tol, v3it, v3t,
    };
    use crate::base::vector3::Vector3;
    use crate::linalg::lu::{Perm3, Perm3PartialEq, Perm3Trait, oracle_lu3 as oracle};
    use crate::testing::black_box;
    use super::{Lu3, Lu3InternalTrait, Lu3Trait};

    /// The oracle's first `unit` 3x3 case whose factorisation actually swaps rows, so every
    /// benchmark exercises the permutation.
    fn a_bench() -> Matrix3<Fixed> {
        m3(
            [
                [-2414118097, 417657389, 4595817171], [6298117444, -2725477524, -1338161353],
                [2614037897, 2690897069, 3051067169],
            ],
        )
    }

    /// Its right-hand side.
    fn b_bench() -> Vector3<Fixed> {
        v3t((7757329492, -3622378744, 4147284176))
    }

    /// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay
    /// for `new`.
    fn f_bench() -> Lu3<Fixed> {
        Lu3 {
            lu: m3(
                [
                    [6298117444, -2725477524, -1338161353], [1782629075, 3822108354, 3606471941],
                    [-1646294845, -704614973, 4674552576],
                ],
            ),
            p: Perm3 { p1: 2, p2: 3 },
        }
    }

    /// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
    /// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle,
    /// which partial pivoting undoes exactly because `|l_ik| < 1`.
    fn a_exact() -> Matrix3<Fixed> {
        m3(
            [
                [-3221225472, -13958643712, -2147483648], [-9663676416, 13958643712, 20401094656],
                [-12884901888, 12884901888, 8589934592],
            ],
        )
    }

    /// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
    /// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it
    /// is wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
    /// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what
    /// keeps `U` from growing. Upstream has no unpivoted variant either.
    fn new_no_pivot(matrix: Matrix3<Fixed>) -> Lu3<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        if a11 != Real::zero() {
            let l = a21 / a11;
            let nl = -l;
            a22 = Real::mul_add(nl, a12, a22);
            a23 = Real::mul_add(nl, a13, a23);
            a21 = l;
            let l = a31 / a11;
            let nl = -l;
            a32 = Real::mul_add(nl, a12, a32);
            a33 = Real::mul_add(nl, a13, a33);
            a31 = l;
        }
        if a22 != Real::zero() {
            let l = a32 / a22;
            let nl = -l;
            a33 = Real::mul_add(nl, a23, a33);
            a32 = l;
        }
        Lu3 {
            lu: Matrix3 {
                m11: a11,
                m21: a21,
                m31: a31,
                m12: a12,
                m22: a22,
                m32: a32,
                m13: a13,
                m23: a23,
                m33: a33,
            },
            p: Perm3 { p1: 1, p2: 2 },
        }
    }

    /// `solve` with ONE reciprocal per pivot and 3 multiplications instead of 3 correctly rounded
    /// divisions.
    /// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
    /// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to
    /// amortise it over, so it costs 27 560 against 23 560 gas; and rounding `1 / u_ii` before
    /// using it puts 3 of the oracle cases outside the tolerance against 0
    /// (`test_solve_candidates_error`).
    fn solve_recip(f: Lu3<Fixed>, b: Vector3<Fixed>) -> Option<Vector3<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let r3 = Real::recip(f.lu.m33);
        let pb = f.permute(b);
        let y1 = pb.x;
        let y2 = Real::mul_add(-f.lu.m21, y1, pb.y);
        let y3 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), pb.z), f.lu.m31, y1),
                f.lu.m32,
                y2,
            ),
        );
        let x3 = y3 * r3;
        let x2 = Real::mul_add(-f.lu.m23, x3, y2) * r2;
        let x1 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y1), f.lu.m12, x2),
                f.lu.m13,
                x3,
            ),
        )
            * r1;
        Some(Vector3 { x: x1, y: x2, z: x3 })
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped
    /// (one division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the
    /// UNPIVOTED factorisation.
    fn solve_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle::lu3_solve_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let f = if variant == 2 {
                new_no_pivot(m3(a))
            } else {
                Lu3Trait::new(m3(a))
            };
            let got = if variant == 1 {
                solve_recip(f, v3t(b))
            } else {
                f.solve(v3t(b))
            };
            let e = v3t(expected);
            let err = max_ulp_diff_v3(got.unwrap(), e);
            if err > oracle_tol(max_abs_v3(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    #[test]
    fn test_new_is_exact_on_a_dyadic_matrix() {
        let f = Lu3Trait::new(a_exact());
        let lu = f.lu;
        assert!(
            lu == m3(
                [
                    [-12884901888, 12884901888, 8589934592],
                    [1073741824, -17179869184, -4294967296], [3221225472, -1073741824, 12884901888],
                ],
            ),
        );
        assert!(f.p() == Perm3 { p1: 3, p2: 3 });
        assert!(f.permute_rows(a_exact()) == f.l() * f.u());
        assert!(f.determinant() == fx(154618822656));
    }

    #[test]
    fn test_new_reconstruction_oracle() {
        // `P A == L U` within 16 raw units on the lu3 vectors.
        let mut cases = oracle::lu3_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Lu3Trait::new(m3(a));
            worst = core::cmp::max(worst, max_ulp_diff3(f.permute_rows(m3(a)), f.l() * f.u()));
        }
        assert!(worst == 10, "reconstruction error {worst}");
    }

    #[test]
    fn test_factors_permutation_and_accessors() {
        let f = Lu3Trait::new(a_exact());
        let l = f.l();
        assert!(
            l == m3(
                [
                    [4294967296, 0, 0], [1073741824, 4294967296, 0],
                    [3221225472, -1073741824, 4294967296],
                ],
            ),
        );
        let u = f.u();
        assert!(
            u == m3(
                [
                    [-12884901888, 12884901888, 8589934592], [0, -17179869184, -4294967296],
                    [0, 0, 12884901888],
                ],
            ),
        );
        let pa = f.permute_rows(a_exact());
        assert!(
            pa == m3(
                [
                    [-12884901888, 12884901888, 8589934592],
                    [-3221225472, -13958643712, -2147483648],
                    [-9663676416, 13958643712, 20401094656],
                ],
            ),
        );
        assert!(f.permute(v3it((1, 2, 3))) == v3it((3, 1, 2)));
        // the identity factors without a single swap
        let id = Lu3Trait::new(Matrix3Trait::<Fixed>::identity());
        assert!(id.p() == Perm3Trait::identity());
        assert!(id.permute(b_bench()) == b_bench());
        assert!(id.permute_rows(a_bench()) == a_bench());
        assert!(id.l() == Matrix3Trait::<Fixed>::identity());
        assert!(id.u() == Matrix3Trait::<Fixed>::identity());
    }

    #[test]
    fn test_solve_candidates_error() {
        // Oracle, 18 well-conditioned matrices: (cases above the LU tolerance,
        // worst error in ulp) of the shipped divisions, of one reciprocal per
        // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
        // The unpivoted figure is NOT a win: these matrices are random and
        // well-conditioned, so their leading entries happen to be usable pivots.
        // `test_no_pivot_candidate_is_wrong` shows the structural failure.
        assert!(solve_failures(0) == (0, 436));
        assert!(solve_failures(1) == (3, 374));
        assert!(solve_failures(2) == (0, 447));
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_permute__transpositions() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(v3t((-3622378744, 4147284176, 7757329492)));
        assert!(f.permute(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_permute_rows__transpositions() {
        let f = black_box(f_bench());
        let a = black_box(a_bench());
        let e = black_box(
            m3(
                [
                    [6298117444, -2725477524, -1338161353], [2614037897, 2690897069, 3051067169],
                    [-2414118097, 417657389, 4595817171],
                ],
            ),
        );
        assert!(f.permute_rows(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_solve__alt_recip() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v3t((-1035334030, 24604680, 6703439320))));
        assert!(solve_recip(f, b) == e);
    }
}
