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
//! Moved from `crates/nalgebra/src/linalg/lu/lu3.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix3::{Matrix3, Matrix3Trait};
use nalgebra::base::vector3::Vector3;
use nalgebra::linalg::lu::lu3::{Lu3, Lu3Trait, Matrix3LuTrait};
use nalgebra::linalg::lu::{Perm3, Perm3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Lu3PartialEq, Perm3PartialEq, abs_raw, fx, int, m3, max_abs_m3, max_abs_v3, max_ulp_diff3,
    max_ulp_diff_v3, oracle_tol, ulp_diff, v3it, v3t,
};
use simba::scalar::Real;
use crate::lu::{oracle_lu3 as oracle, oracle_matrix3_compare as compare};

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

/// An exactly singular integer matrix (one row is an integer combination of the others, and
/// every multiplier of the elimination is dyadic, so the last pivot is exactly zero).
fn a_singular() -> Matrix3<Fixed> {
    m3(
        [
            [-25769803776, -12884901888, -25769803776], [12884901888, 12884901888, 0],
            [51539607552, 38654705664, 25769803776],
        ],
    )
}

/// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
/// rejection path measures the rejection and not `new`.
fn f_singular() -> Lu3<Fixed> {
    Lu3 {
        lu: m3(
            [
                [51539607552, 38654705664, 25769803776], [-2147483648, 6442450944, -12884901888],
                [1073741824, 2147483648, 0],
            ],
        ),
        p: Perm3 { p1: 3, p2: 3 },
    }
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

/// `try_inverse` as 3 full calls to `solve` on the unit vectors — the obvious route, and the
/// one upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
/// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation
/// 3 times and multiplies by the leading zeros of each unit vector.
fn try_inverse_solve_columns(f: Lu3<Fixed>) -> Option<Matrix3<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let c1 = f.solve(Vector3 { x: int(1), y: int(0), z: int(0) }).unwrap();
    let c2 = f.solve(Vector3 { x: int(0), y: int(1), z: int(0) }).unwrap();
    let c3 = f.solve(Vector3 { x: int(0), y: int(0), z: int(1) }).unwrap();
    Some(
        Matrix3 {
            m11: c1.x,
            m21: c1.y,
            m31: c1.z,
            m12: c2.x,
            m22: c2.y,
            m32: c2.z,
            m13: c3.x,
            m23: c3.y,
            m33: c3.z,
        },
    )
}

/// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
/// substitution. Here the reciprocal IS amortised (3 columns share it), so this is the
/// candidate the gas argument favours; it is not shipped because it rounds `1 / u_ii` before
/// using it, which is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2
/// forbid (the same call `Vector3::unscale` makes).
fn try_inverse_recip(f: Lu3<Fixed>) -> Option<Matrix3<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let r1 = Real::recip(f.lu.m11);
    let r2 = Real::recip(f.lu.m22);
    let r3 = Real::recip(f.lu.m33);
    let y21 = -f.lu.m21;
    let y31 = Real::wide_rescale(
        Real::wide_sub_prod(Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m31), f.lu.m32, y21),
    );
    let x31 = y31 * r3;
    let x21 = Real::mul_add(-f.lu.m23, x31, y21) * r2;
    let x11 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m12, x21),
            f.lu.m13,
            x31,
        ),
    )
        * r1;
    let y32 = -f.lu.m32;
    let x32 = y32 * r3;
    let x22 = Real::mul_add(-f.lu.m23, x32, int(1)) * r2;
    let x12 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x22), f.lu.m13, x32,
        ),
    )
        * r1;
    let x33 = int(1) * r3;
    let x23 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m23, x33))
        * r2;
    let x13 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x23), f.lu.m13, x33,
        ),
    )
        * r1;
    let mut c11 = x11;
    let mut c12 = x12;
    let mut c13 = x13;
    let mut c21 = x21;
    let mut c22 = x22;
    let mut c23 = x23;
    let mut c31 = x31;
    let mut c32 = x32;
    let mut c33 = x33;
    if f.p.p2 == 3 {
        let t = c12;
        c12 = c13;
        c13 = t;
        let t = c22;
        c22 = c23;
        c23 = t;
        let t = c32;
        c32 = c33;
        c33 = t;
    }
    if f.p.p1 == 2 {
        let t = c11;
        c11 = c12;
        c12 = t;
        let t = c21;
        c21 = c22;
        c22 = t;
        let t = c31;
        c31 = c32;
        c32 = t;
    } else if f.p.p1 == 3 {
        let t = c11;
        c11 = c13;
        c13 = t;
        let t = c21;
        c21 = c23;
        c23 = t;
        let t = c31;
        c31 = c33;
        c33 = t;
    }
    Some(
        Matrix3 {
            m11: c11,
            m21: c21,
            m31: c31,
            m12: c12,
            m22: c22,
            m32: c32,
            m13: c13,
            m23: c23,
            m33: c33,
        },
    )
}

/// `(cases above the LU tolerance, worst error in ulp)` of the two 3x3 inverses on the
/// `matrix3`
/// oracle vectors: 0 = `Lu3::try_inverse`, 1 = `Matrix3::try_inverse` (cofactors with the
/// integer pre-scaling).
fn matrix3_inverse_failures(variant: u8) -> (u32, u128) {
    let mut cases = compare::matrix3_try_inverse_cases();
    let mut failures = 0;
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = if variant == 0 {
            Lu3Trait::new(m3(a)).try_inverse()
        } else {
            m3(a).try_inverse()
        };
        let e = m3(expected);
        let err = max_ulp_diff3(got.unwrap(), e);
        if err > oracle_tol(max_abs_m3(e), tol) {
            failures += 1;
        }
        worst = core::cmp::max(worst, err);
    }
    (failures, worst)
}

/// The same for the determinant: 0 = product of the LU pivots, 1 = `Matrix3::determinant`
/// (cofactor expansion on fused kernels).
fn matrix3_determinant_failures(variant: u8) -> (u32, u128) {
    let mut cases = compare::matrix3_determinant_cases();
    let mut failures = 0;
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = if variant == 0 {
            Lu3Trait::new(m3(a)).determinant()
        } else {
            m3(a).determinant()
        };
        let err = ulp_diff(got, fx(expected));
        if err > oracle_tol(abs_raw(fx(expected)), tol) {
            failures += 1;
        }
        worst = core::cmp::max(worst, err);
    }
    (failures, worst)
}

#[test]
fn test_new_multipliers_are_bounded_by_one() {
    // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
    let mut cases = oracle::lu3_solve_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        assert!(max_abs_m3(Lu3Trait::new(m3(a)).l()) <= 0x100000000);
    }
}

#[test]
fn test_singular_is_rejected() {
    let f = Lu3Trait::new(a_singular());
    assert!(!f.is_invertible());
    assert!(f.solve(b_bench()).is_none());
    assert!(f.try_inverse().is_none());
    assert!(f.determinant() == int(0));
    let z = Lu3Trait::new(Matrix3Trait::<Fixed>::zeros());
    assert!(!z.is_invertible());
    assert!(z.try_inverse().is_none());
    assert!(z.determinant() == int(0));
    assert!(z.p() == Perm3Trait::identity());
    assert!(Lu3Trait::new(a_bench()).is_invertible());
}

#[test]
fn test_solve_oracle() {
    let mut cases = oracle::lu3_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Lu3Trait::new(m3(a)).solve(v3t(b)).unwrap();
        let e = v3t(expected);
        let err = max_ulp_diff_v3(x, e);
        assert!(err <= oracle_tol(max_abs_v3(e), tol), "solve error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 436);
}

#[test]
fn test_solve_near_singular_oracle() {
    // Condition number 1e2..1e4: the oracle flags these as behaviour, not
    // precision, tests. The tolerances are large by construction.
    let mut cases = oracle::lu3_solve_near_singular_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Lu3Trait::new(m3(a)).solve(v3t(b)).unwrap();
        let e = v3t(expected);
        let err = max_ulp_diff_v3(x, e);
        assert!(err <= oracle_tol(max_abs_v3(e), tol), "solve error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 7603197);
}

#[test]
fn test_try_inverse_oracle() {
    let mut cases = oracle::lu3_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let inv = Lu3Trait::new(m3(a)).try_inverse().unwrap();
        let e = m3(expected);
        let err = max_ulp_diff3(inv, e);
        assert!(err <= oracle_tol(max_abs_m3(e), tol), "inverse error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 1639);
}

#[test]
fn test_try_inverse_product_is_identity() {
    let mut cases = oracle::lu3_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let inv = Lu3Trait::new(m3(a)).try_inverse().unwrap();
        // Worst residual over the oracle: 142 ulp.
        assert!((m3(a) * inv).is_identity(142));
        assert!((inv * m3(a)).is_identity(142));
    }
}

#[test]
fn test_try_inverse_candidates() {
    // The static unit right-hand sides give BIT-IDENTICAL results to 3 calls
    // to `solve`; the saving is pure gas.
    let mut cases = oracle::lu3_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let f = Lu3Trait::new(m3(a));
        assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
        worst =
            core::cmp::max(
                worst, max_ulp_diff3(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
            );
    }
    // ... and the reciprocal variant drifts by at most this many ulp from it.
    assert!(worst == 14);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_try_inverse_overflow_panics() {
    // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
    let _ = black_box(Matrix3Trait::from_diagonal_element(fx(1))).lu().try_inverse();
}

#[test]
fn test_determinant_oracle() {
    let mut cases = oracle::lu3_determinant_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let det = Lu3Trait::new(m3(a)).determinant();
        let err = ulp_diff(det, fx(expected));
        assert!(err <= oracle_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 326241);
}

#[test]
fn test_determinant_exact_and_sign() {
    assert!(Lu3Trait::new(Matrix3Trait::<Fixed>::identity()).determinant() == int(1));
    let d = Matrix3Trait::from_diagonal(v3it((2, -4, 3)));
    assert!(Lu3Trait::new(d).determinant() == int(-24));
    // Swapping two rows flips the sign exactly (the factorisation is exact here,
    // and the sign is applied to the first pivot before any rounding).
    let s = m3(
        [
            [-9663676416, 13958643712, 20401094656], [-3221225472, -13958643712, -2147483648],
            [-12884901888, 12884901888, 8589934592],
        ],
    );
    assert!(Lu3Trait::new(s).determinant() == -fx(154618822656));
}

#[test]
fn test_no_pivot_candidate_is_wrong() {
    // A permuted identity: the unpivoted elimination finds a zero at `a11` and
    // gives up, where partial pivoting factors it exactly.
    let swapped = m3([[0, 4294967296, 0], [4294967296, 0, 0], [0, 0, 4294967296]]);
    assert!(!new_no_pivot(swapped).is_invertible());
    let f = Lu3Trait::new(swapped);
    assert!(f.is_invertible());
    assert!(f.determinant() == int(-1));
}

#[test]
fn test_try_inverse_versus_matrix3_cofactors() {
    // Same 18 `matrix3_try_inverse` / `matrix3_determinant` oracle cases, same
    // tolerance predicate. The closed form wins on accuracy for a 3x3: its
    // determinant is a sum of exact 2x2 minors, where the LU determinant is a
    // product of pivots that already carry the rounding of the elimination.
    assert!(matrix3_inverse_failures(0) == (0, 64));
    assert!(matrix3_inverse_failures(1) == (0, 27));
    assert!(matrix3_determinant_failures(0) == (2, 27731535));
    assert!(matrix3_determinant_failures(1) == (0, 556));
}

// --- gas benchmarks --------------------------------------------------------

#[test]
#[inline(never)]
fn bench_lu3_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(
        m3(
            [
                [6298117444, -2725477524, -1338161353], [1782629075, 3822108354, 3606471941],
                [-1646294845, -704614973, 4674552576],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_new__pivot() {
    let a = black_box(a_bench());
    let e = black_box(
        m3(
            [
                [6298117444, -2725477524, -1338161353], [1782629076, 3822108355, 3606471942],
                [-1646294844, -704614971, 4674552574],
            ],
        ),
    );
    assert!(Lu3Trait::new(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu3_new__alt_no_pivot() {
    let a = black_box(a_bench());
    let e = black_box(
        m3(
            [
                [-2414118097, 417657389, 4595817171], [-11205006284, -1635864183, 10651722790],
                [-4650645423, -8252330158, 28493647233],
            ],
        ),
    );
    assert!(new_no_pivot(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu3_factors__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(
        m3(
            [
                [4294967296, 0, 0], [1782629075, 4294967296, 0],
                [-1646294845, -704614973, 4294967296],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_factors__l() {
    let f = black_box(f_bench());
    let e = black_box(
        m3(
            [
                [4294967296, 0, 0], [1782629075, 4294967296, 0],
                [-1646294845, -704614973, 4294967296],
            ],
        ),
    );
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_factors__u() {
    let f = black_box(f_bench());
    let e = black_box(
        m3(
            [
                [6298117444, -2725477524, -1338161353], [0, 3822108354, 3606471941],
                [0, 0, 4674552576],
            ],
        ),
    );
    assert!(f.u() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_p__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(Perm3 { p1: 2, p2: 3 });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_p__field() {
    let f = black_box(f_bench());
    let e = black_box(Perm3 { p1: 2, p2: 3 });
    assert!(f.p() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_permute__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(v3t((-3622378744, 4147284176, 7757329492)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_permute_rows__baseline() {
    let _f = black_box(f_bench());
    let _a = black_box(a_bench());
    let e = black_box(
        m3(
            [
                [6298117444, -2725477524, -1338161353], [2614037897, 2690897069, 3051067169],
                [-2414118097, 417657389, 4595817171],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_is_invertible__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_is_invertible__pivots() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.is_invertible() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(Some(v3t((-1035334030, 24604680, 6703439320))));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_solve__substitution() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(Some(v3t((-1035334029, 24604680, 6703439320))));
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_lu3_solve_singular__baseline() {
    let _s = black_box(f_singular());
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_solve_singular__none() {
    let s = black_box(f_singular());
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(s.solve(b).is_none() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_try_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904016, 1818436374, 1961768289], [-3723567548, -3176899586, 4215453383],
                    [3946205283, 1243908434, 647398487],
                ],
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_try_inverse__columns() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904015, 1818436373, 1961768289], [-3723567548, -3176899587, 4215453383],
                    [3946205284, 1243908435, 647398487],
                ],
            ),
        ),
    );
    assert!(f.try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_try_inverse__alt_solve_columns() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904015, 1818436373, 1961768289], [-3723567548, -3176899587, 4215453383],
                    [3946205284, 1243908435, 647398487],
                ],
            ),
        ),
    );
    assert!(try_inverse_solve_columns(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu3_try_inverse__alt_recip() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904016, 1818436374, 1961768288], [-3723567548, -3176899586, 4215453382],
                    [3946205284, 1243908434, 647398487],
                ],
            ),
        ),
    );
    assert!(try_inverse_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu3_determinant__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(fx(6100059567));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_determinant__pivots() {
    let f = black_box(f_bench());
    let e = black_box(fx(6100059567));
    assert!(f.determinant() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_vs_matrix3_inverse__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904016, 1818436374, 1961768289], [-3723567548, -3176899586, 4215453383],
                    [3946205283, 1243908434, 647398487],
                ],
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_vs_matrix3_inverse__lu() {
    let a = black_box(a_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904016, 1818436373, 1961768289], [-3723567549, -3176899587, 4215453384],
                    [3946205285, 1243908435, 647398485],
                ],
            ),
        ),
    );
    assert!(a.lu().try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_vs_matrix3_inverse__cofactors() {
    let a = black_box(a_bench());
    let e = black_box(
        Some(
            m3(
                [
                    [-772904015, 1818436374, 1961768289], [-3723567548, -3176899587, 4215453384],
                    [3946205284, 1243908435, 647398485],
                ],
            ),
        ),
    );
    assert!(a.try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_vs_matrix3_determinant__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(fx(6100059567));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu3_vs_matrix3_determinant__lu() {
    let a = black_box(a_bench());
    let e = black_box(fx(6100059565));
    assert!(a.lu().determinant() == e);
}

#[test]
#[inline(never)]
fn bench_lu3_vs_matrix3_determinant__cofactors() {
    let a = black_box(a_bench());
    let e = black_box(fx(6100059569));
    assert!(a.determinant() == e);
}
