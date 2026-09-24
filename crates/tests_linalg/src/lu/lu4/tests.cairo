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
//! Moved from `crates/nalgebra/src/linalg/lu/lu4.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix4::{Matrix4, Matrix4Trait};
use nalgebra::base::vector4::Vector4;
use nalgebra::linalg::lu::lu4::{Lu4, Lu4Trait, Matrix4LuTrait};
use nalgebra::linalg::lu::{Perm4, Perm4Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Lu4PartialEq, Perm4PartialEq, abs_raw, fx, int, m4, max_abs_m4, max_abs_v4, max_ulp_diff4,
    max_ulp_diff_v4, oracle_tol, ulp_diff, v4it, v4t,
};
use simba::scalar::Real;
use crate::lu::oracle_lu4 as oracle;

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

/// An exactly singular integer matrix (one row is an integer combination of the others, and
/// every multiplier of the elimination is dyadic, so the last pivot is exactly zero).
fn a_singular() -> Matrix4<Fixed> {
    m4(
        [
            [17179869184, 8589934592, 25769803776, 85899345920], [0, 0, 0, -21474836480],
            [-12884901888, -21474836480, -12884901888, 4294967296],
            [4294967296, 17179869184, 0, -25769803776],
        ],
    )
}

/// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
/// rejection path measures the rejection and not `new`.
fn f_singular() -> Lu4<Fixed> {
    Lu4 {
        lu: m4(
            [
                [17179869184, 8589934592, 25769803776, 85899345920],
                [-3221225472, -15032385536, 6442450944, 68719476736], [0, 0, 0, -21474836480],
                [1073741824, -4294967296, 0, 21474836480],
            ],
        ),
        p: Perm4 { p1: 1, p2: 3, p3: 3 },
    }
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

/// `try_inverse` as 4 full calls to `solve` on the unit vectors — the obvious route, and the
/// one upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
/// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation
/// 4 times and multiplies by the leading zeros of each unit vector.
fn try_inverse_solve_columns(f: Lu4<Fixed>) -> Option<Matrix4<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let c1 = f.solve(Vector4 { x: int(1), y: int(0), z: int(0), w: int(0) }).unwrap();
    let c2 = f.solve(Vector4 { x: int(0), y: int(1), z: int(0), w: int(0) }).unwrap();
    let c3 = f.solve(Vector4 { x: int(0), y: int(0), z: int(1), w: int(0) }).unwrap();
    let c4 = f.solve(Vector4 { x: int(0), y: int(0), z: int(0), w: int(1) }).unwrap();
    Some(
        Matrix4 {
            m11: c1.x,
            m21: c1.y,
            m31: c1.z,
            m41: c1.w,
            m12: c2.x,
            m22: c2.y,
            m32: c2.z,
            m42: c2.w,
            m13: c3.x,
            m23: c3.y,
            m33: c3.z,
            m43: c3.w,
            m14: c4.x,
            m24: c4.y,
            m34: c4.z,
            m44: c4.w,
        },
    )
}

/// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
/// substitution. Here the reciprocal IS amortised (4 columns share it), so this is the
/// candidate the gas argument favours; it is not shipped because it rounds `1 / u_ii` before
/// using it, which is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2
/// forbid (the same call `Vector4::unscale` makes).
fn try_inverse_recip(f: Lu4<Fixed>) -> Option<Matrix4<Fixed>> {
    if !f.is_invertible() {
        return None;
    }
    let r1 = Real::recip(f.lu.m11);
    let r2 = Real::recip(f.lu.m22);
    let r3 = Real::recip(f.lu.m33);
    let r4 = Real::recip(f.lu.m44);
    let y21 = -f.lu.m21;
    let y31 = Real::wide_rescale(
        Real::wide_sub_prod(Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m31), f.lu.m32, y21),
    );
    let y41 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m41), f.lu.m42, y21,
            ),
            f.lu.m43,
            y31,
        ),
    );
    let x41 = y41 * r4;
    let x31 = Real::mul_add(-f.lu.m34, x41, y31) * r3;
    let x21 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y21), f.lu.m23, x31),
            f.lu.m24,
            x41,
        ),
    )
        * r2;
    let x11 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m12, x21,
                ),
                f.lu.m13,
                x31,
            ),
            f.lu.m14,
            x41,
        ),
    )
        * r1;
    let y32 = -f.lu.m32;
    let y42 = Real::wide_rescale(
        Real::wide_sub_prod(Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m42), f.lu.m43, y32),
    );
    let x42 = y42 * r4;
    let x32 = Real::mul_add(-f.lu.m34, x42, y32) * r3;
    let x22 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m23, x32),
            f.lu.m24,
            x42,
        ),
    )
        * r2;
    let x12 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x22), f.lu.m13, x32,
            ),
            f.lu.m14,
            x42,
        ),
    )
        * r1;
    let y43 = -f.lu.m43;
    let x43 = y43 * r4;
    let x33 = Real::mul_add(-f.lu.m34, x43, int(1)) * r3;
    let x23 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m23, x33), f.lu.m24, x43,
        ),
    )
        * r2;
    let x13 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x23), f.lu.m13, x33,
            ),
            f.lu.m14,
            x43,
        ),
    )
        * r1;
    let x44 = int(1) * r4;
    let x34 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m34, x44))
        * r3;
    let x24 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m23, x34), f.lu.m24, x44,
        ),
    )
        * r2;
    let x14 = Real::wide_rescale(
        Real::wide_sub_prod(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x24), f.lu.m13, x34,
            ),
            f.lu.m14,
            x44,
        ),
    )
        * r1;
    let mut c11 = x11;
    let mut c12 = x12;
    let mut c13 = x13;
    let mut c14 = x14;
    let mut c21 = x21;
    let mut c22 = x22;
    let mut c23 = x23;
    let mut c24 = x24;
    let mut c31 = x31;
    let mut c32 = x32;
    let mut c33 = x33;
    let mut c34 = x34;
    let mut c41 = x41;
    let mut c42 = x42;
    let mut c43 = x43;
    let mut c44 = x44;
    if f.p.p3 == 4 {
        let t = c13;
        c13 = c14;
        c14 = t;
        let t = c23;
        c23 = c24;
        c24 = t;
        let t = c33;
        c33 = c34;
        c34 = t;
        let t = c43;
        c43 = c44;
        c44 = t;
    }
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
        let t = c42;
        c42 = c43;
        c43 = t;
    } else if f.p.p2 == 4 {
        let t = c12;
        c12 = c14;
        c14 = t;
        let t = c22;
        c22 = c24;
        c24 = t;
        let t = c32;
        c32 = c34;
        c34 = t;
        let t = c42;
        c42 = c44;
        c44 = t;
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
        let t = c41;
        c41 = c42;
        c42 = t;
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
        let t = c41;
        c41 = c43;
        c43 = t;
    } else if f.p.p1 == 4 {
        let t = c11;
        c11 = c14;
        c14 = t;
        let t = c21;
        c21 = c24;
        c24 = t;
        let t = c31;
        c31 = c34;
        c34 = t;
        let t = c41;
        c41 = c44;
        c44 = t;
    }
    Some(
        Matrix4 {
            m11: c11,
            m21: c21,
            m31: c31,
            m41: c41,
            m12: c12,
            m22: c22,
            m32: c32,
            m42: c42,
            m13: c13,
            m23: c23,
            m33: c33,
            m43: c43,
            m14: c14,
            m24: c24,
            m34: c34,
            m44: c44,
        },
    )
}

#[test]
fn test_new_multipliers_are_bounded_by_one() {
    // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
    let mut cases = oracle::lu4_solve_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        assert!(max_abs_m4(Lu4Trait::new(m4(a)).l()) <= 0x100000000);
    }
}

#[test]
fn test_singular_is_rejected() {
    let f = Lu4Trait::new(a_singular());
    assert!(!f.is_invertible());
    assert!(f.solve(b_bench()).is_none());
    assert!(f.try_inverse().is_none());
    assert!(f.determinant() == int(0));
    let z = Lu4Trait::new(Matrix4Trait::<Fixed>::zeros());
    assert!(!z.is_invertible());
    assert!(z.try_inverse().is_none());
    assert!(z.determinant() == int(0));
    assert!(z.p() == Perm4Trait::identity());
    assert!(Lu4Trait::new(a_bench()).is_invertible());
}

#[test]
fn test_solve_oracle() {
    let mut cases = oracle::lu4_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Lu4Trait::new(m4(a)).solve(v4t(b)).unwrap();
        let e = v4t(expected);
        let err = max_ulp_diff_v4(x, e);
        assert!(err <= oracle_tol(max_abs_v4(e), tol), "solve error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 1451);
}

#[test]
fn test_try_inverse_oracle() {
    let mut cases = oracle::lu4_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let inv = Lu4Trait::new(m4(a)).try_inverse().unwrap();
        let e = m4(expected);
        let err = max_ulp_diff4(inv, e);
        assert!(err <= oracle_tol(max_abs_m4(e), tol), "inverse error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 2619);
}

#[test]
fn test_try_inverse_product_is_identity() {
    let mut cases = oracle::lu4_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let inv = Lu4Trait::new(m4(a)).try_inverse().unwrap();
        // Worst residual over the oracle: 117 ulp.
        assert!((m4(a) * inv).is_identity(117));
        assert!((inv * m4(a)).is_identity(117));
    }
}

#[test]
fn test_try_inverse_candidates() {
    // The static unit right-hand sides give BIT-IDENTICAL results to 4 calls
    // to `solve`; the saving is pure gas.
    let mut cases = oracle::lu4_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let f = Lu4Trait::new(m4(a));
        assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
        worst =
            core::cmp::max(
                worst, max_ulp_diff4(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
            );
    }
    // ... and the reciprocal variant drifts by at most this many ulp from it.
    assert!(worst == 7);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_try_inverse_overflow_panics() {
    // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
    let _ = black_box(Matrix4Trait::from_diagonal_element(fx(1))).lu().try_inverse();
}

#[test]
fn test_determinant_oracle() {
    let mut cases = oracle::lu4_determinant_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let det = Lu4Trait::new(m4(a)).determinant();
        let err = ulp_diff(det, fx(expected));
        assert!(err <= oracle_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
        worst = core::cmp::max(worst, err);
    }
    assert!(worst == 17456);
}

#[test]
fn test_determinant_exact_and_sign() {
    assert!(Lu4Trait::new(Matrix4Trait::<Fixed>::identity()).determinant() == int(1));
    let d = Matrix4Trait::from_diagonal(v4it((2, -4, 3, -1)));
    assert!(Lu4Trait::new(d).determinant() == int(24));
    // Swapping two rows flips the sign exactly (the factorisation is exact here,
    // and the sign is applied to the first pivot before any rounding).
    let s = m4(
        [
            [-3221225472, 5368709120, -8589934592, -31138512896],
            [0, 3221225472, 7516192768, 9663676416],
            [-12884901888, 12884901888, 8589934592, -17179869184],
            [3221225472, -7516192768, 10737418240, 8589934592],
        ],
    );
    assert!(Lu4Trait::new(s).determinant() == -fx(-257698037760));
}

#[test]
fn test_no_pivot_candidate_is_wrong() {
    // A permuted identity: the unpivoted elimination finds a zero at `a11` and
    // gives up, where partial pivoting factors it exactly.
    let swapped = m4(
        [
            [0, 4294967296, 0, 0], [4294967296, 0, 0, 0], [0, 0, 4294967296, 0],
            [0, 0, 0, 4294967296],
        ],
    );
    assert!(!new_no_pivot(swapped).is_invertible());
    let f = Lu4Trait::new(swapped);
    assert!(f.is_invertible());
    assert!(f.determinant() == int(-1));
}

// --- gas benchmarks --------------------------------------------------------

#[test]
#[inline(never)]
fn bench_lu4_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(
        m4(
            [
                [-3905117829, -101544735, 2917390324, -1089761038],
                [138042224, -3594501327, -1433035679, -1793672967],
                [2904686338, -672132918, -4889978813, 1893902051],
                [242782019, -3207459345, 2235014795, -6063365662],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_new__pivot() {
    let a = black_box(a_bench());
    let e = black_box(
        m4(
            [
                [-3905117829, -101544735, 2917390324, -1089761038],
                [138042224, -3594501327, -1433035679, -1793672967],
                [2904686339, -672132918, -4889978813, 1893902052],
                [242782019, -3207459345, 2235014795, -6063365663],
            ],
        ),
    );
    assert!(Lu4Trait::new(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu4_new__alt_no_pivot() {
    let a = black_box(a_bench());
    let e = black_box(
        m4(
            [
                [-125512283, -3597765021, -1339269412, -1828698387],
                [133631171083, 111837271070, 44586657535, 55807321098],
                [7553781749, 345871238, -2544646850, -5077817040],
                [90374759643, 2926289000, 8253522131, 11651805292],
            ],
        ),
    );
    assert!(new_no_pivot(a).lu == e);
}

#[test]
#[inline(never)]
fn bench_lu4_factors__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(
        m4(
            [
                [4294967296, 0, 0, 0], [138042224, 4294967296, 0, 0],
                [2904686338, -672132918, 4294967296, 0],
                [242782019, -3207459345, 2235014795, 4294967296],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_factors__l() {
    let f = black_box(f_bench());
    let e = black_box(
        m4(
            [
                [4294967296, 0, 0, 0], [138042224, 4294967296, 0, 0],
                [2904686338, -672132918, 4294967296, 0],
                [242782019, -3207459345, 2235014795, 4294967296],
            ],
        ),
    );
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_lu4_factors__u() {
    let f = black_box(f_bench());
    let e = black_box(
        m4(
            [
                [-3905117829, -101544735, 2917390324, -1089761038],
                [0, -3594501327, -1433035679, -1793672967], [0, 0, -4889978813, 1893902051],
                [0, 0, 0, -6063365662],
            ],
        ),
    );
    assert!(f.u() == e);
}

#[test]
#[inline(never)]
fn bench_lu4_p__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(Perm4 { p1: 2, p2: 2, p3: 4 });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_p__field() {
    let f = black_box(f_bench());
    let e = black_box(Perm4 { p1: 2, p2: 2, p3: 4 });
    assert!(f.p() == e);
}

#[test]
#[inline(never)]
fn bench_lu4_permute__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(v4t((3697509183, 7470372587, 5116133854, 6158142748)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_permute_rows__baseline() {
    let _f = black_box(f_bench());
    let _a = black_box(a_bench());
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
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_is_invertible__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_is_invertible__pivots() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.is_invertible() == e);
}

#[test]
#[inline(never)]
fn bench_lu4_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(Some(v4t((-6526739453, -3077920152, -5908376560, -6714764226))));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_solve__substitution() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(Some(v4t((-6526739453, -3077920151, -5908376560, -6714764226))));
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_lu4_solve_singular__baseline() {
    let _s = black_box(f_singular());
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_solve_singular__none() {
    let s = black_box(f_singular());
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(s.solve(b).is_none() == e);
}

#[test]
#[inline(never)]
fn bench_lu4_try_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(
        Some(
            m4(
                [
                    [-368910955, -2803986567, -82972189, -2814138558],
                    [-3573914248, -315013115, 1987894369, 469481705],
                    [-1374341491, 2247339518, -1178301687, -3159192093],
                    [-2024239780, -833661343, -3042327497, 1583166179],
                ],
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_try_inverse__columns() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m4(
                [
                    [-368910954, -2803986567, -82972188, -2814138557],
                    [-3573914249, -315013115, 1987894368, 469481704],
                    [-1374341490, 2247339518, -1178301686, -3159192092],
                    [-2024239779, -833661342, -3042327496, 1583166180],
                ],
            ),
        ),
    );
    assert!(f.try_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_lu4_try_inverse__alt_solve_columns() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m4(
                [
                    [-368910954, -2803986567, -82972188, -2814138557],
                    [-3573914249, -315013115, 1987894368, 469481704],
                    [-1374341490, 2247339518, -1178301686, -3159192092],
                    [-2024239779, -833661342, -3042327496, 1583166180],
                ],
            ),
        ),
    );
    assert!(try_inverse_solve_columns(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu4_try_inverse__alt_recip() {
    let f = black_box(f_bench());
    let e = black_box(
        Some(
            m4(
                [
                    [-368910955, -2803986567, -82972190, -2814138558],
                    [-3573914248, -315013115, 1987894369, 469481705],
                    [-1374341491, 2247339518, -1178301687, -3159192093],
                    [-2024239780, -833661342, -3042327496, 1583166179],
                ],
            ),
        ),
    );
    assert!(try_inverse_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_lu4_determinant__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(fx(5253079147));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_lu4_determinant__pivots() {
    let f = black_box(f_bench());
    let e = black_box(fx(5253079147));
    assert!(f.determinant() == e);
}
