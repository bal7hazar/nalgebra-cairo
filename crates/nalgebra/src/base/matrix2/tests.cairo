//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/matrix2/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, int, m2, m2i, max_ulp_diff2, s2, sym2_upper, v2i, v2t};
use crate::base::sym_matrix2::SymMatrix2Trait;
use crate::base::{oracle_matrix2, oracle_matrix2_inverse};
use super::{Matrix2, Matrix2InternalTrait, Matrix2Trait};

// --- losing candidates of the determinant / inverse study (kept as evidence) -----------------

/// `adjugate / determinant` without the integer pre-scaling of small matrices.
fn try_inverse_div(m: Matrix2<Fixed>) -> Option<Matrix2<Fixed>> {
    let adj = m.adjugate();
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    Some(Matrix2 { m11: adj.m11 / det, m21: adj.m21 / det, m12: adj.m12 / det, m22: adj.m22 / det })
}

/// Upstream's `adjugate / determinant` (the formula of `try_inverse_div`) through ONE prepared
/// divisor (`Real::div4`): bit-identical to `try_inverse_div`, so it fails the same oracle
/// cases; kept to price upstream's formula at its cheapest (WP 7.2).
fn try_inverse_div_n(m: Matrix2<Fixed>) -> Option<Matrix2<Fixed>> {
    let adj = m.adjugate();
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    let (m11, m21, m12, m22) = Real::div4(adj.m11, adj.m21, adj.m12, adj.m22, det);
    Some(Matrix2 { m11, m21, m12, m22 })
}

/// `adjugate * (1 / determinant)`: one reciprocal, 4 multiplications.
fn try_inverse_recip(m: Matrix2<Fixed>) -> Option<Matrix2<Fixed>> {
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    Some(m.adjugate().scale(det.recip()))
}

/// `(cases above the oracle tolerance, worst error in ulp)` of an inverse candidate.
fn inverse_failures(variant: u8) -> (u32, u128) {
    let mut cases = oracle_matrix2_inverse::matrix2_try_inverse_cases();
    let mut failures = 0;
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = match variant {
            0 => m2(a).try_inverse(),
            1 => try_inverse_div(m2(a)),
            3 => try_inverse_div_n(m2(a)),
            _ => try_inverse_recip(m2(a)),
        };
        let err = max_ulp_diff2(got.unwrap(), m2(expected));
        if err > tol.into() {
            failures += 1;
        }
        worst = core::cmp::max(worst, err);
    }
    (failures, worst)
}

#[test]
fn test_columns_diagonal() {
    let m = m2i([[1, 2], [3, 4]]);
    assert!(m.column1() == v2i(1, 3));
    assert!(m.column2() == v2i(2, 4));
    assert!(m.diagonal() == v2i(1, 4));
    assert!(Matrix2Trait::from_columns(m.column1(), m.column2()) == m);
}

#[test]
fn test_from_outer_oracle() {
    let mut cases = oracle_matrix2::matrix2_outer_cases();
    while let Some(case) = cases.pop_front() {
        let (u, v, expected, _) = *case;
        assert!(Matrix2InternalTrait::from_outer(v2t(u), v2t(v)) == m2(expected));
    }
}

#[test]
fn test_mul_transpose_matches_generic_product() {
    // small, unit and medium cases (`large` squares do not fit).
    let mut cases = oracle_matrix2::matrix2_mul_cases().slice(0, 12);
    while let Some(case) = cases.pop_front() {
        let (a, b, _, _) = *case;
        assert!(m2(a).mul_transpose().to_matrix() == m2(a) * m2(a).transpose());
        assert!(m2(b).mul_transpose().to_matrix() == m2(b) * m2(b).transpose());
    }
    let one = 0x100000000;
    assert!(m2i([[1, 2], [3, 4]]).mul_transpose() == s2((5 * one, 11 * one, 25 * one)));
}

#[test]
fn test_adjugate_identity() {
    let m = m2i([[2, -1], [5, 3]]);
    let det = m.determinant();
    assert!(det == int(11));
    assert!(m * m.adjugate() == Matrix2Trait::from_diagonal_element(det));
    assert!(m.adjugate() * m == Matrix2Trait::from_diagonal_element(det));
}

#[test]
fn test_try_inverse_candidates_error() {
    // Oracle, 30 well-conditioned matrices (10 small, 10 unit, 10 medium):
    // (cases above the oracle tolerance, worst error in ulp) of the shipped algorithm, of
    // `adjugate / det` without pre-scaling and of `adjugate * (1 / det)`.
    assert!(inverse_failures(0) == (0, 8));
    assert!(inverse_failures(1) == (1, 1112));
    // Upstream's formula through one prepared divisor: the same bits as `try_inverse_div`.
    assert!(inverse_failures(3) == (1, 1112));
    assert!(inverse_failures(2) == (4, 1111));
}

#[test]
fn test_try_inverse_small_scale() {
    // 2^-12 * I: without pre-scaling the determinant (2^-24: 8 significant bits).
    let m = Matrix2Trait::from_diagonal_element(fx(0x100000));
    let expected = Matrix2Trait::from_diagonal_element(int(4096));
    assert!(m.determinant() == fx(0x100));
    assert!(try_inverse_div(m).unwrap() == expected);
    // Relative error below 2^-33 (1 ulp on 4096).
    let inv = m.try_inverse().unwrap();
    assert!(inv.abs_diff_eq(expected, 1));
    assert!(!inv.abs_diff_eq(expected, 0));
}

#[test]
#[inline(never)]
fn bench_matrix2_from_outer__products() {
    let u = black_box(v2t((2347498971, -7037259012)));
    let v = black_box(v2t((-7543252641, 4885438966)));
    let e = black_box(m2([[-4122913306, 2670232892], [12359540590, -8004740671]]));
    assert!(Matrix2InternalTrait::from_outer(u, v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_column__second() {
    let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
    let e = black_box(v2t((2839048663, 5944454799)));
    assert!(a.column2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_mul_transpose__baseline() {
    let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
    let e = black_box(s2((9163632458, -5767163102, 21130337440)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_mul_transpose__structured() {
    let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
    let e = black_box(s2((9163632458, -5767163102, 21130337440)));
    assert!(a.mul_transpose() == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_mul_transpose__generic() {
    let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
    let e = black_box(s2((9163632458, -5767163102, 21130337440)));
    assert!(sym2_upper(a * a.transpose()) == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_adjugate__cofactors() {
    let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
    let e = black_box(m2([[5944454799, -2839048663], [7444297509, 5594399379]]));
    assert!(a.adjugate() == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_try_inverse__alt_div() {
    let a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
    let e = black_box(m2([[-3112122076, 4554249820], [873639280, 8427202879]]));
    assert!(try_inverse_div(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_try_inverse__alt_div_n() {
    let a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
    let e = black_box(m2([[-3112122076, 4554249820], [873639280, 8427202879]]));
    assert!(try_inverse_div_n(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix2_try_inverse__alt_recip() {
    let a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
    let e = black_box(m2([[-3112122077, 4554249819], [873639280, 8427202878]]));
    assert!(try_inverse_recip(a).unwrap() == e);
}
