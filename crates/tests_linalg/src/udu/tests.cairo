//! Correctness of `Udu{2,3,4,6}` (upstream `UDU`): upstream's own `udu{2,3}_u_d` oracle vectors,
//! the reconstruction `u·diag(d)·uᵀ ≈ p` over the `ldlt{n}_l_d` inputs (the same bounds as
//! the `LDLᵀ` kernel's, of which this is the reversal), exact integer cases, the upper triangle
//! as the only input, and the zero-pivot criterion.
//!
//! Moved from `crates/nalgebra/src/linalg/udu/tests.cairo` (WP 8.1c, test-only package).

use fixed::Fixed;
use nalgebra::base::matrix2::Matrix2Trait;
use nalgebra::base::matrix3::Matrix3Trait;
use nalgebra::base::matrix4::{Matrix4, Matrix4Trait};
use nalgebra::base::matrix6::{Matrix6, Matrix6Trait};
use nalgebra::linalg::udu::{Udu2Trait, Udu3Trait, Udu4Trait, Udu6Trait};
use nalgebra_tests_utils::{
    m2, m2i, m3, m3i, m4, m4i, m6, m6i, max_ulp_diff2, max_ulp_diff3, max_ulp_diff4, max_ulp_diff6,
    max_ulp_diff_v2, max_ulp_diff_v3, v2it, v2t, v3it, v3t, v4it, v6it,
};
use crate::oracle_udu;

/// Upstream's own `udu2_u_d` vectors: `u` and `d` within the oracle tolerance.
#[test]
fn test_udu2_oracle() {
    let mut cases = oracle_udu::udu2_u_d_cases();
    let mut worst = 0;
    let mut fail = 0;
    while let Some(case) = cases.pop_front() {
        let (a, u, d, tol) = *case;
        let f = Udu2Trait::new(m2(a)).unwrap();
        let e = core::cmp::max(max_ulp_diff2(f.u, m2(u)), max_ulp_diff_v2(f.d, v2t(d)));
        if e > tol.into() {
            fail += 1;
        }
        worst = core::cmp::max(worst, e);
    }
    assert!(fail == 0 && worst <= 8, "udu2 fail {} worst {}", fail, worst);
}

/// Upstream's own `udu3_u_d` vectors: `u` and `d` within the oracle tolerance.
#[test]
fn test_udu3_oracle() {
    let mut cases = oracle_udu::udu3_u_d_cases();
    let mut worst = 0;
    let mut fail = 0;
    while let Some(case) = cases.pop_front() {
        let (a, u, d, tol) = *case;
        let f = Udu3Trait::new(m3(a)).unwrap();
        let e = core::cmp::max(max_ulp_diff3(f.u, m3(u)), max_ulp_diff_v3(f.d, v3t(d)));
        if e > tol.into() {
            fail += 1;
        }
        worst = core::cmp::max(worst, e);
    }
    assert!(fail == 0 && worst <= 8, "udu3 fail {} worst {}", fail, worst);
}

/// Reconstruction `u·diag(d)·uᵀ ≈ p` over the `ldlt2_l_d` inputs (the backward error, bounded
/// like the `LDLᵀ` kernel's reconstruction since the factors are its reversal).
#[test]
fn test_udu2_reconstruction() {
    let mut cases = oracle_udu::ldlt2_l_d_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _l, _d, _tol) = *case;
        let f = Udu2Trait::new(m2(a)).unwrap();
        let r = f.u * f.d_matrix() * f.u.transpose();
        worst = core::cmp::max(worst, max_ulp_diff2(r, m2(a)));
    }
    assert!(worst <= 30, "udu2_rec worst {}", worst);
}

/// Reconstruction `u·diag(d)·uᵀ ≈ p` over the `ldlt3_l_d` inputs (the backward error, bounded
/// like the `LDLᵀ` kernel's reconstruction since the factors are its reversal).
#[test]
fn test_udu3_reconstruction() {
    let mut cases = oracle_udu::ldlt3_l_d_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _l, _d, _tol) = *case;
        let f = Udu3Trait::new(m3(a)).unwrap();
        let r = f.u * f.d_matrix() * f.u.transpose();
        worst = core::cmp::max(worst, max_ulp_diff3(r, m3(a)));
    }
    assert!(worst <= 16, "udu3_rec worst {}", worst);
}

/// Reconstruction `u·diag(d)·uᵀ ≈ p` over the `ldlt4_l_d` inputs (the backward error, bounded
/// like the `LDLᵀ` kernel's reconstruction since the factors are its reversal).
#[test]
fn test_udu4_reconstruction() {
    let mut cases = oracle_udu::ldlt4_l_d_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _l, _d, _tol) = *case;
        let f = Udu4Trait::new(m4(a)).unwrap();
        let r = f.u * f.d_matrix() * f.u.transpose();
        worst = core::cmp::max(worst, max_ulp_diff4(r, m4(a)));
    }
    assert!(worst <= 75, "udu4_rec worst {}", worst);
}

/// Reconstruction `u·diag(d)·uᵀ ≈ p` over the `ldlt6_l_d` inputs (the backward error, bounded
/// like the `LDLᵀ` kernel's reconstruction since the factors are its reversal).
#[test]
fn test_udu6_reconstruction() {
    let mut cases = oracle_udu::ldlt6_l_d_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _l, _d, _tol) = *case;
        let f = Udu6Trait::new(m6(a)).unwrap();
        let r = f.u * f.d_matrix() * f.u.transpose();
        worst = core::cmp::max(worst, max_ulp_diff6(r, m6(a)));
    }
    assert!(worst <= 25, "udu6_rec worst {}", worst);
}

/// `p_ij = 2 + 1 - max(i, j)` (the reversed `min` matrix): `u` is the all-ones unit upper
/// triangular matrix and `d = (1, .., 1)`, exactly.
#[test]
fn test_udu2_reversed_min_matrix_is_exact() {
    let f = Udu2Trait::new(m2i([[2, 1], [1, 1]])).unwrap();
    assert!(f.u == m2i([[1, 1], [0, 1]]));
    assert!(f.d == v2it((1, 1)));
    assert!(f.d_matrix() == Matrix2Trait::identity());
}

/// `p_ij = 3 + 1 - max(i, j)` (the reversed `min` matrix): `u` is the all-ones unit upper
/// triangular matrix and `d = (1, .., 1)`, exactly.
#[test]
fn test_udu3_reversed_min_matrix_is_exact() {
    let f = Udu3Trait::new(m3i([[3, 2, 1], [2, 2, 1], [1, 1, 1]])).unwrap();
    assert!(f.u == m3i([[1, 1, 1], [0, 1, 1], [0, 0, 1]]));
    assert!(f.d == v3it((1, 1, 1)));
    assert!(f.d_matrix() == Matrix3Trait::identity());
}

/// `p_ij = 4 + 1 - max(i, j)` (the reversed `min` matrix): `u` is the all-ones unit upper
/// triangular matrix and `d = (1, .., 1)`, exactly.
#[test]
fn test_udu4_reversed_min_matrix_is_exact() {
    let f = Udu4Trait::new(m4i([[4, 3, 2, 1], [3, 3, 2, 1], [2, 2, 2, 1], [1, 1, 1, 1]])).unwrap();
    assert!(f.u == m4i([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]));
    assert!(f.d == v4it((1, 1, 1, 1)));
    assert!(f.d_matrix() == Matrix4Trait::identity());
}

/// `p_ij = 6 + 1 - max(i, j)` (the reversed `min` matrix): `u` is the all-ones unit upper
/// triangular matrix and `d = (1, .., 1)`, exactly.
#[test]
fn test_udu6_reversed_min_matrix_is_exact() {
    let f = Udu6Trait::new(
        m6i(
            [
                [6, 5, 4, 3, 2, 1], [5, 5, 4, 3, 2, 1], [4, 4, 4, 3, 2, 1], [3, 3, 3, 3, 2, 1],
                [2, 2, 2, 2, 2, 1], [1, 1, 1, 1, 1, 1],
            ],
        ),
    )
        .unwrap();
    assert!(
        f
            .u == m6i(
                [
                    [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1],
                ],
            ),
    );
    assert!(f.d == v6it((1, 1, 1, 1, 1, 1)));
    assert!(f.d_matrix() == Matrix6Trait::identity());
}

/// Only the upper triangle is read, like upstream: the lower one may hold anything.
#[test]
fn test_udu_reads_the_upper_triangle() {
    let f = Udu2Trait::new(m2i([[3, 2], [-7, 2]])).unwrap();
    assert!(f.u == Udu2Trait::new(m2i([[3, 2], [2, 2]])).unwrap().u);
    let f = Udu3Trait::new(m3i([[3, 2, 1], [9, 2, 1], [9, 9, 1]])).unwrap();
    assert!(f.u == m3i([[1, 1, 1], [0, 1, 1], [0, 0, 1]]) && f.d == v3it((1, 1, 1)));
}

/// `None` iff a pivot is exactly zero (`d_n = p_nn` first, like upstream); an indefinite matrix
/// is accepted, `d` then has mixed signs.
#[test]
fn test_udu_zero_pivot_and_indefinite_input() {
    assert!(Udu2Trait::new(m2i([[1, 0], [0, 0]])).is_none());
    assert!(Udu3Trait::new(m3i([[1, 1, 0], [1, 1, 0], [0, 0, 1]])).is_none());
    let f = Udu2Trait::new(m2i([[-1, 0], [0, 1]])).unwrap();
    assert!(f.d == v2it((-1, 1)) && f.u == Matrix2Trait::identity());
    let z4: Matrix4<Fixed> = Matrix4Trait::zeros();
    assert!(Udu4Trait::new(z4).is_none());
    let z6: Matrix6<Fixed> = Matrix6Trait::zeros();
    assert!(Udu6Trait::new(z6).is_none());
}
