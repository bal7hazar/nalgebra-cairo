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
//! Moved from `crates/nalgebra/src/linalg/lu/lu2.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix2::{Matrix2, Matrix2Trait};
use nalgebra::base::vector2::Vector2;
use nalgebra::linalg::lu::lu2::{Lu2, Lu2Trait, Matrix2LuTrait};
use nalgebra::linalg::lu::{Perm2, Perm2Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Lu2PartialEq, Perm2PartialEq, abs_raw, fx, int, m2, max_abs_m2, max_abs_v2, max_ulp_diff2,
    max_ulp_diff_v2, oracle_tol, ulp_diff, v2it, v2t,
};
use simba::scalar::Real;
use crate::lu::oracle_lu2 as oracle;

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

/// An exactly singular integer matrix (one row is an integer combination of the others, and
/// every multiplier of the elimination is dyadic, so the last pivot is exactly zero).
fn a_singular() -> Matrix2<Fixed> {
    m2([[4294967296, 0], [-8589934592, 0]])
}

/// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
/// rejection path measures the rejection and not `new`.
fn f_singular() -> Lu2<Fixed> {
    Lu2 { lu: m2([[-8589934592, 0], [-2147483648, 0]]), p: Perm2 { p1: 2 } }
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

/// `try_inverse` as 2 full calls to `solve` on the unit vectors — the obvious route, and the
/// one upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
/// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation
/// 2 times and multiplies by the leading zeros of each unit vector.
fn try_inverse_solve_columns(f: Lu2<Fixed>) -> Option<Matrix2<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let c1 = f.solve(Vector2 { x: int(1), y: int(0) }).unwrap();
    let c2 = f.solve(Vector2 { x: int(0), y: int(1) }).unwrap();
    Some(Matrix2 { m11: c1.x, m21: c1.y, m12: c2.x, m22: c2.y })
}

/// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
/// substitution. Here the reciprocal IS amortised (2 columns share it), so this is the
/// candidate the gas argument favours; it is not shipped because it rounds `1 / u_ii` before
/// using it, which is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2
/// forbid (the same call `Vector2::unscale` makes).
fn try_inverse_recip(f: Lu2<Fixed>) -> Option<Matrix2<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let r1 = Real::recip(f.lu.m11);
    let r2 = Real::recip(f.lu.m22);
    let y21 = -f.lu.m21;
    let x21 = y21 * r2;
    let x11 = Real::mul_add(-f.lu.m12, x21, int(1)) * r1;
    let x22 = int(1) * r2;
    let x12 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x22))
        * r1;
    let mut c11 = x11;
    let mut c12 = x12;
    let mut c21 = x21;
    let mut c22 = x22;
    if f.p.p1 == 2 {
        let t = c11;
        c11 = c12;
        c12 = t;
        let t = c21;
        c21 = c22;
        c22 = t;
    }
    Some(Matrix2 { m11: c11, m21: c21, m12: c12, m22: c22 })
}

#[test]
fn test_new_multipliers_are_bounded_by_one() {
    // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
    let mut cases = oracle::lu2_solve_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        assert!(max_abs_m2(Lu2Trait::new(m2(a)).l()) <= 0x100000000);
    }
}

#[test]
fn test_singular_is_rejected() {
    let f = Lu2Trait::new(a_singular());
    assert!(!f.is_invertible());
    assert!(f.solve(b_bench()).is_none());
    assert!(f.try_inverse().is_none());
    assert!(f.determinant() == int(0));
    let z = Lu2Trait::new(Matrix2Trait::<Fixed>::zeros());
    assert!(!z.is_invertible());
    assert!(z.try_inverse().is_none());
    assert!(z.determinant() == int(0));
    assert!(z.p() == Perm2Trait::identity());
    assert!(Lu2Trait::new(a_bench()).is_invertible());
}

#[test]
fn test_solve_oracle() {
    let mut cases = oracle::lu2_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Lu2Trait::new(m2(a)).solve(v2t(b)).unwrap();
        let e = v2t(expected);
        let err = max_ulp_diff_v2(x, e);
        assert!(err <= oracle_tol(max_abs_v2(e), tol), "solve error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 122);
}

#[test]
fn test_try_inverse_oracle() {
    let mut cases = oracle::lu2_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let inv = Lu2Trait::new(m2(a)).try_inverse().unwrap();
        let e = m2(expected);
        let err = max_ulp_diff2(inv, e);
        assert!(err <= oracle_tol(max_abs_m2(e), tol), "inverse error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 394);
}

#[test]
fn test_try_inverse_product_is_identity() {
    let mut cases = oracle::lu2_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let inv = Lu2Trait::new(m2(a)).try_inverse().unwrap();
        // Worst residual over the oracle: 61 ulp.
        assert!((m2(a) * inv).is_identity(61));
        assert!((inv * m2(a)).is_identity(61));
    }
}

#[test]
fn test_try_inverse_candidates() {
    // The static unit right-hand sides give BIT-IDENTICAL results to 2 calls
    // to `solve`; the saving is pure gas.
    let mut cases = oracle::lu2_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let f = Lu2Trait::new(m2(a));
        assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
        worst =
            core::cmp::max(
                worst, max_ulp_diff2(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
            );
    }
    // ... and the reciprocal variant drifts by at most this many ulp from it.
    assert!(worst == 3);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_try_inverse_overflow_panics() {
    // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
    let _ = black_box(Matrix2Trait::from_diagonal_element(fx(1))).lu().try_inverse();
}

#[test]
fn test_determinant_oracle() {
    let mut cases = oracle::lu2_determinant_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let det = Lu2Trait::new(m2(a)).determinant();
        let err = ulp_diff(det, fx(expected));
        assert!(err <= oracle_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 43);
}

#[test]
fn test_determinant_exact_and_sign() {
    assert!(Lu2Trait::new(Matrix2Trait::<Fixed>::identity()).determinant() == int(1));
    let d = Matrix2Trait::from_diagonal(v2it((2, -4)));
    assert!(Lu2Trait::new(d).determinant() == int(-8));
    // Swapping two rows flips the sign exactly (the factorisation is exact here,
    // and the sign is applied to the first pivot before any rounding).
    let s = m2([[12884901888, 4294967296], [-3221225472, 16106127360]]);
    assert!(Lu2Trait::new(s).determinant() == -fx(-51539607552));
}

#[test]
fn test_no_pivot_candidate_is_wrong() {
    // A permuted identity: the unpivoted elimination finds a zero at `a11` and
    // gives up, where partial pivoting factors it exactly.
    let swapped = m2([[0, 4294967296], [4294967296, 0]]);
    assert!(!new_no_pivot(swapped).is_invertible());
    let f = Lu2Trait::new(swapped);
    assert!(f.is_invertible());
    assert!(f.determinant() == int(-1));
}

// --- gas benchmarks --------------------------------------------------------

#[test]
#[inline(never)]
fn bench_lu2_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(m2([[3058251633, 1932838604], [-389055470, 4539457456]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_new__pivot() {
    let a = black_box(a_bench());
    let e = black_box(m2([[3058251633, 1932838604], [-389055469, 4539457456]]));
    assert!(Lu2Trait::new(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu2_new__alt_no_pivot() {
    let a = black_box(a_bench());
    let e = black_box(m2([[-277028774, 4364373136], [-47414174914, 50113217405]]));
    assert!(new_no_pivot(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu2_factors__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(m2([[4294967296, 0], [-389055470, 4294967296]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_factors__l() {
    let f = black_box(f_bench());
    let e = black_box(m2([[4294967296, 0], [-389055470, 4294967296]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_lu2_factors__u() {
    let f = black_box(f_bench());
    let e = black_box(m2([[3058251633, 1932838604], [0, 4539457456]]));
    assert!(f.u() == e);
}

#[test]
#[inline(never)]
fn bench_lu2_p__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(Perm2 { p1: 2 });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_p__field() {
    let f = black_box(f_bench());
    let e = black_box(Perm2 { p1: 2 });
    assert!(f.p() == e);
}

#[test]
#[inline(never)]
fn bench_lu2_permute__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(v2t((-6536196560, -5886581674)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_permute_rows__baseline() {
    let _f = black_box(f_bench());
    let _a = black_box(a_bench());
    let e = black_box(m2([[3058251633, 1932838604], [-277028774, 4364373136]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_is_invertible__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_is_invertible__pivots() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.is_invertible() == e);
}

#[test]
#[inline(never)]
fn bench_lu2_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(Some(v2t((-5305313723, -6129723447))));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_solve__substitution() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(Some(v2t((-5305313723, -6129723447))));
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_lu2_solve_singular__baseline() {
    let _s = black_box(f_singular());
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_solve_singular__none() {
    let s = black_box(f_singular());
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(s.solve(b).is_none() == e);
}

#[test]
#[inline(never)]
fn bench_lu2_try_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645105, 368101372]])));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_try_inverse__columns() {
    let f = black_box(f_bench());
    let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645106, 368101373]])));
    assert!(f.try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_lu2_try_inverse__alt_solve_columns() {
    let f = black_box(f_bench());
    let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645106, 368101373]])));
    assert!(try_inverse_solve_columns(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu2_try_inverse__alt_recip() {
    let f = black_box(f_bench());
    let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645106, 368101372]])));
    assert!(try_inverse_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu2_determinant__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(fx(-3232342000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu2_determinant__pivots() {
    let f = black_box(f_bench());
    let e = black_box(fx(-3232342000));
    assert!(f.determinant() == e);
}
