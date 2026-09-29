//! `Lu4`: the LU factorisation with partial pivoting of a `Matrix4` (upstream
//! `nalgebra::linalg::LU` on a 4x4 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 3 row transpositions chosen by partial pivoting. Both factors share one `Matrix4` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm4`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 6 `abs` and comparisons plus 3 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu4_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

pub use nalgebra_linalg4::linalg::lu::lu4::*;
#[cfg(test)]
use super::Perm4PartialEq;

/// Test-only field-wise equality (upstream `Lu4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Lu4PartialEq<T, +PartialEq<T>> of PartialEq<Lu4<T>> {
    fn eq(lhs: @Lu4<T>, rhs: @Lu4<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Lu4`: an exactly representable factorisation, the identities (`P A = L U`, `A
    //! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
    //! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
    //!
    //! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
    //! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.
    //!
    //! Gas benchmarks of `Lu4` (`bench_lu4_<op>__<variant>`, net = raw - the `baseline` of the
    //! group), and the alternative implementations that lost, kept as evidence together with the
    //! tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong
    //! on a matrix as ordinary as a permuted identity.
    //!
    //! - `alt_recip`: one reciprocal per pivot instead of one correctly rounded division per output
    //! scalar. DEARER for `solve`, where a single right-hand side does not amortise the reciprocal,
    //! cheaper for `try_inverse`, where 4 columns share it, and a second rounding per output in
    //! both.
    //!
    //! - `alt_solve_columns`: the inverse as 4 calls to `solve`. Bit-identical, dearer.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/lu/lu4/tests.cairo`.

    use fixed::Fixed;
    use simba::scalar::Real;
    use crate::base::matrix4::{Matrix4, Matrix4Trait};
    use crate::base::matrix_test_utils::{
        fx, m4, max_abs_v4, max_ulp_diff4, max_ulp_diff_v4, oracle_tol, v4it, v4t,
    };
    use crate::base::vector4::Vector4;
    use crate::linalg::lu::{Perm4, Perm4PartialEq, Perm4Trait, oracle_lu4 as oracle};
    use crate::testing::black_box;
    use super::{Lu4, Lu4InternalTrait, Lu4Trait};

    /// The oracle's first `unit` 4x4 case whose factorisation actually swaps rows, so every
    /// benchmark exercises the permutation.
    fn a_bench() -> Matrix4<Fixed> {
        m4(
            [
                [-125512283, -3597765021, -1339269412, -1828698387],
                [-3905117829, -101544735, 2917390324, -1089761038],
                [-220744962, 2678614957, -1309551624, -3799912375],
                [-2641031148, 493840096, -2692687512, 1437594188],
            ],
        )
    }

    /// Its right-hand side.
    fn b_bench() -> Vector4<Fixed> {
        v4t((7470372587, 3697509183, 6158142748, 5116133854))
    }

    /// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay
    /// for `new`.
    fn f_bench() -> Lu4<Fixed> {
        Lu4 {
            lu: m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [138042224, -3594501327, -1433035679, -1793672967],
                    [2904686338, -672132918, -4889978813, 1893902051],
                    [242782019, -3207459345, 2235014795, -6063365662],
                ],
            ),
            p: Perm4 { p1: 2, p2: 2, p3: 4 },
        }
    }

    /// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
    /// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle,
    /// which partial pivoting undoes exactly because `|l_ik| < 1`.
    fn a_exact() -> Matrix4<Fixed> {
        m4(
            [
                [0, 3221225472, 7516192768, 9663676416],
                [-3221225472, 5368709120, -8589934592, -31138512896],
                [-12884901888, 12884901888, 8589934592, -17179869184],
                [3221225472, -7516192768, 10737418240, 8589934592],
            ],
        )
    }

    /// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
    /// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it
    /// is wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
    /// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what
    /// keeps `U` from growing. Upstream has no unpivoted variant either.
    fn new_no_pivot(matrix: Matrix4<Fixed>) -> Lu4<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a14 = matrix.m14;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a24 = matrix.m24;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        let mut a34 = matrix.m34;
        let mut a41 = matrix.m41;
        let mut a42 = matrix.m42;
        let mut a43 = matrix.m43;
        let mut a44 = matrix.m44;
        if a11 != Real::zero() {
            let l = a21 / a11;
            let nl = -l;
            a22 = Real::mul_add(nl, a12, a22);
            a23 = Real::mul_add(nl, a13, a23);
            a24 = Real::mul_add(nl, a14, a24);
            a21 = l;
            let l = a31 / a11;
            let nl = -l;
            a32 = Real::mul_add(nl, a12, a32);
            a33 = Real::mul_add(nl, a13, a33);
            a34 = Real::mul_add(nl, a14, a34);
            a31 = l;
            let l = a41 / a11;
            let nl = -l;
            a42 = Real::mul_add(nl, a12, a42);
            a43 = Real::mul_add(nl, a13, a43);
            a44 = Real::mul_add(nl, a14, a44);
            a41 = l;
        }
        if a22 != Real::zero() {
            let l = a32 / a22;
            let nl = -l;
            a33 = Real::mul_add(nl, a23, a33);
            a34 = Real::mul_add(nl, a24, a34);
            a32 = l;
            let l = a42 / a22;
            let nl = -l;
            a43 = Real::mul_add(nl, a23, a43);
            a44 = Real::mul_add(nl, a24, a44);
            a42 = l;
        }
        if a33 != Real::zero() {
            let l = a43 / a33;
            let nl = -l;
            a44 = Real::mul_add(nl, a34, a44);
            a43 = l;
        }
        Lu4 {
            lu: Matrix4 {
                m11: a11,
                m21: a21,
                m31: a31,
                m41: a41,
                m12: a12,
                m22: a22,
                m32: a32,
                m42: a42,
                m13: a13,
                m23: a23,
                m33: a33,
                m43: a43,
                m14: a14,
                m24: a24,
                m34: a34,
                m44: a44,
            },
            p: Perm4 { p1: 1, p2: 2, p3: 3 },
        }
    }

    /// `solve` with ONE reciprocal per pivot and 4 multiplications instead of 4 correctly rounded
    /// divisions.
    /// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
    /// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to
    /// amortise it over, so it costs 39 660 against 34 180 gas; and rounding `1 / u_ii` before
    /// using it puts 4 of the oracle cases outside the tolerance against 0
    /// (`test_solve_candidates_error`).
    fn solve_recip(f: Lu4<Fixed>, b: Vector4<Fixed>) -> Option<Vector4<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let r3 = Real::recip(f.lu.m33);
        let r4 = Real::recip(f.lu.m44);
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
        let y4 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), pb.w), f.lu.m41, y1,
                    ),
                    f.lu.m42,
                    y2,
                ),
                f.lu.m43,
                y3,
            ),
        );
        let x4 = y4 * r4;
        let x3 = Real::mul_add(-f.lu.m34, x4, y3) * r3;
        let x2 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y2), f.lu.m23, x3),
                f.lu.m24,
                x4,
            ),
        )
            * r2;
        let x1 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), y1), f.lu.m12, x2,
                    ),
                    f.lu.m13,
                    x3,
                ),
                f.lu.m14,
                x4,
            ),
        )
            * r1;
        Some(Vector4 { x: x1, y: x2, z: x3, w: x4 })
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped
    /// (one division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the
    /// UNPIVOTED factorisation.
    fn solve_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle::lu4_solve_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let f = if variant == 2 {
                new_no_pivot(m4(a))
            } else {
                Lu4Trait::new(m4(a))
            };
            let got = if variant == 1 {
                solve_recip(f, v4t(b))
            } else {
                f.solve(v4t(b))
            };
            let e = v4t(expected);
            let err = max_ulp_diff_v4(got.unwrap(), e);
            if err > oracle_tol(max_abs_v4(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    #[test]
    fn test_new_is_exact_on_a_dyadic_matrix() {
        let f = Lu4Trait::new(a_exact());
        let lu = f.lu;
        assert!(
            lu == m4(
                [
                    [-12884901888, 12884901888, 8589934592, -17179869184],
                    [-1073741824, -4294967296, 12884901888, 4294967296],
                    [0, -3221225472, 17179869184, 12884901888],
                    [1073741824, -2147483648, -1073741824, -21474836480],
                ],
            ),
        );
        assert!(f.p() == Perm4 { p1: 3, p2: 4, p3: 3 });
        assert!(f.permute_rows(a_exact()) == f.l() * f.u());
        assert!(f.determinant() == fx(-257698037760));
    }

    #[test]
    fn test_new_reconstruction_oracle() {
        // `P A == L U` within 44 raw units on the lu4 vectors.
        let mut cases = oracle::lu4_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Lu4Trait::new(m4(a));
            worst = core::cmp::max(worst, max_ulp_diff4(f.permute_rows(m4(a)), f.l() * f.u()));
        }
        assert!(worst == 25, "reconstruction error {worst}");
    }

    #[test]
    fn test_factors_permutation_and_accessors() {
        let f = Lu4Trait::new(a_exact());
        let l = f.l();
        assert!(
            l == m4(
                [
                    [4294967296, 0, 0, 0], [-1073741824, 4294967296, 0, 0],
                    [0, -3221225472, 4294967296, 0],
                    [1073741824, -2147483648, -1073741824, 4294967296],
                ],
            ),
        );
        let u = f.u();
        assert!(
            u == m4(
                [
                    [-12884901888, 12884901888, 8589934592, -17179869184],
                    [0, -4294967296, 12884901888, 4294967296], [0, 0, 17179869184, 12884901888],
                    [0, 0, 0, -21474836480],
                ],
            ),
        );
        let pa = f.permute_rows(a_exact());
        assert!(
            pa == m4(
                [
                    [-12884901888, 12884901888, 8589934592, -17179869184],
                    [3221225472, -7516192768, 10737418240, 8589934592],
                    [0, 3221225472, 7516192768, 9663676416],
                    [-3221225472, 5368709120, -8589934592, -31138512896],
                ],
            ),
        );
        assert!(f.permute(v4it((1, 2, 3, 4))) == v4it((3, 4, 1, 2)));
        // the identity factors without a single swap
        let id = Lu4Trait::new(Matrix4Trait::<Fixed>::identity());
        assert!(id.p() == Perm4Trait::identity());
        assert!(id.permute(b_bench()) == b_bench());
        assert!(id.permute_rows(a_bench()) == a_bench());
        assert!(id.l() == Matrix4Trait::<Fixed>::identity());
        assert!(id.u() == Matrix4Trait::<Fixed>::identity());
    }

    #[test]
    fn test_solve_candidates_error() {
        // Oracle, 15 well-conditioned matrices: (cases above the LU tolerance,
        // worst error in ulp) of the shipped divisions, of one reciprocal per
        // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
        // The unpivoted figure is NOT a win: these matrices are random and
        // well-conditioned, so their leading entries happen to be usable pivots.
        // `test_no_pivot_candidate_is_wrong` shows the structural failure.
        assert!(solve_failures(0) == (0, 1451));
        assert!(solve_failures(1) == (3, 1450));
        assert!(solve_failures(2) == (0, 678));
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_permute__transpositions() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(v4t((3697509183, 7470372587, 5116133854, 6158142748)));
        assert!(f.permute(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_permute_rows__transpositions() {
        let f = black_box(f_bench());
        let a = black_box(a_bench());
        let e = black_box(
            m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [-125512283, -3597765021, -1339269412, -1828698387],
                    [-2641031148, 493840096, -2692687512, 1437594188],
                    [-220744962, 2678614957, -1309551624, -3799912375],
                ],
            ),
        );
        assert!(f.permute_rows(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_solve__alt_recip() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v4t((-6526739453, -3077920152, -5908376560, -6714764225))));
        assert!(solve_recip(f, b) == e);
    }
}
