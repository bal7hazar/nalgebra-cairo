//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/matrix3/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, int, m3, m3i, max_ulp_diff3, s3, sym3_upper, v3i, v3t};
use crate::base::sym_matrix3::SymMatrix3Trait;
use crate::base::{oracle_matrix3, oracle_matrix3_inverse};
use super::{Matrix3, Matrix3InternalTrait, Matrix3Trait};

/// `adjugate / determinant` without the integer pre-scaling of small matrices.
fn try_inverse_div(m: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
    let adj = m.adjugate();
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    Some(
        Matrix3 {
            m11: adj.m11 / det,
            m21: adj.m21 / det,
            m31: adj.m31 / det,
            m12: adj.m12 / det,
            m22: adj.m22 / det,
            m32: adj.m32 / det,
            m13: adj.m13 / det,
            m23: adj.m23 / det,
            m33: adj.m33 / det,
        },
    )
}

/// Upstream's `adjugate / determinant` (the formula of `try_inverse_div`) through ONE prepared
/// divisor (`Real::div9`): bit-identical to `try_inverse_div`, so it fails the same oracle
/// cases; kept to price upstream's formula at its cheapest (WP 7.2).
fn try_inverse_div_n(m: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
    let adj = m.adjugate();
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    let (m11, m21, m31, m12, m22, m32, m13, m23, m33) = Real::div9(
        adj.m11, adj.m21, adj.m31, adj.m12, adj.m22, adj.m32, adj.m13, adj.m23, adj.m33, det,
    );
    Some(Matrix3 { m11, m21, m31, m12, m22, m32, m13, m23, m33 })
}

/// `adjugate * (1 / determinant)`: one reciprocal, 9 multiplications.
fn try_inverse_recip(m: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    Some(m.adjugate().scale(det.recip()))
}

/// `(cases above the oracle tolerance, worst error in ulp)` of an inverse candidate.
fn inverse_failures(variant: u8) -> (u32, u128) {
    let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_cases();
    let mut failures = 0;
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = match variant {
            0 => m3(a).try_inverse(),
            1 => try_inverse_div(m3(a)),
            3 => try_inverse_div_n(m3(a)),
            _ => try_inverse_recip(m3(a)),
        };
        let err = max_ulp_diff3(got.unwrap(), m3(expected));
        if err > tol.into() {
            failures += 1;
        }
        worst = core::cmp::max(worst, err);
    }
    (failures, worst)
}

#[test]
fn test_columns_rows_diagonal() {
    let m = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    assert!(m.column1() == v3i(1, 4, 7));
    assert!(m.column2() == v3i(2, 5, 8));
    assert!(m.column3() == v3i(3, 6, 9));
    assert!(m.row1() == v3i(1, 2, 3));
    assert!(m.row2() == v3i(4, 5, 6));
    assert!(m.row3() == v3i(7, 8, 9));
    assert!(m.diagonal() == v3i(1, 5, 9));
    assert!(Matrix3Trait::from_columns(m.column1(), m.column2(), m.column3()) == m);
    assert!(Matrix3Trait::from_rows(m.row1(), m.row2(), m.row3()) == m);
}

#[test]
fn test_from_outer_oracle() {
    let mut cases = oracle_matrix3::matrix3_outer_cases();
    while let Some(case) = cases.pop_front() {
        let (u, v, expected, _) = *case;
        assert!(Matrix3InternalTrait::from_outer(v3t(u), v3t(v)) == m3(expected));
    }
}

#[test]
fn test_cross_matrix_kernels_match_materialised_products() {
    let mut cases = oracle_matrix3::matrix3_mul_vec_cases();
    while let Some(case) = cases.pop_front() {
        let (a, v, _, _) = *case;
        let cm = Matrix3Trait::cross_matrix(v3t(v));
        assert!(Matrix3InternalTrait::cross_matrix_mul(v3t(v), m3(a)) == cm * m3(a));
    }
    // [x]× * I = [x]× = I * [x]×
    let x = v3i(1, 2, 3);
    let expected = m3i([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]);
    assert!(Matrix3InternalTrait::cross_matrix_mul(x, Matrix3Trait::identity()) == expected);
}

#[test]
fn test_mul_transpose_matches_generic_product() {
    // small, unit and medium cases (`large` squares do not fit).
    let mut cases = oracle_matrix3::matrix3_mul_cases().slice(0, 12);
    while let Some(case) = cases.pop_front() {
        let (a, b, _, _) = *case;
        assert!(m3(a).mul_transpose().to_matrix() == m3(a) * m3(a).transpose());
        assert!(m3(b).mul_transpose().to_matrix() == m3(b) * m3(b).transpose());
    }
    let one = 0x100000000;
    assert!(
        m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            .mul_transpose() == s3((14 * one, 32 * one, 50 * one, 77 * one, 122 * one, 194 * one)),
    );
}

#[test]
fn test_adjugate_identity() {
    let m = m3i([[2, -1, 3], [0, 4, 1], [5, 2, -2]]);
    let det = m.determinant();
    assert!(det == int(-85));
    assert!(m * m.adjugate() == Matrix3Trait::from_diagonal_element(det));
    assert!(m.adjugate() * m == Matrix3Trait::from_diagonal_element(det));
}

#[test]
fn test_try_inverse_candidates_error() {
    // Oracle, 30 well-conditioned matrices (10 small, 10 unit, 10 medium):
    // (cases above the oracle tolerance, worst error in ulp) of the shipped algorithm, of
    // `adjugate / det` without pre-scaling and of `adjugate * (1 / det)`.
    assert!(inverse_failures(0) == (0, 27));
    assert!(inverse_failures(1) == (2, 84960));
    // Upstream's formula through one prepared divisor: the same bits as `try_inverse_div`.
    assert!(inverse_failures(3) == (2, 84960));
    assert!(inverse_failures(2) == (9, 84960));
}

#[test]
fn test_try_inverse_small_scale() {
    // 2^-12 * I: without pre-scaling the determinant (below the resolution).
    let m = Matrix3Trait::from_diagonal_element(fx(0x100000));
    let expected = Matrix3Trait::from_diagonal_element(int(4096));
    assert!(m.determinant() == int(0));
    assert!(try_inverse_div(m).is_none());
    // Relative error below 2^-33 (1496 ulp on 4096).
    let inv = m.try_inverse().unwrap();
    assert!(inv.abs_diff_eq(expected, 1496));
    assert!(!inv.abs_diff_eq(expected, 1495));
}

#[test]
#[inline(never)]
fn bench_matrix3_from_outer__products() {
    let u = black_box(v3t((-2161644290, 7569327059, 4908735083)));
    let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
    let e = black_box(
        m3(
            [
                [-3232315749, -3121528358, -1169986858], [11318446411, 10930507472, 4096887363],
                [7340052101, 7088472341, 2656845790],
            ],
        ),
    );
    assert!(Matrix3InternalTrait::from_outer(u, v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_cross_matrix_mul__structured() {
    let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
    let a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(
        m3(
            [
                [7502480414, -13936724843, 3936909644], [-8807047541, 7998733495, -10972227499],
                [2770170478, 17162262662, 18397458151],
            ],
        ),
    );
    assert!(Matrix3InternalTrait::cross_matrix_mul(v, a) == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_column__second() {
    let a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(v3t((-3562322882, 8037214559, -6638673079)));
    assert!(a.column2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_row__second() {
    let a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(v3t((-6195210852, 8037214559, 5406550886)));
    assert!(a.row2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_mul_transpose__baseline() {
    let _a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(
        s3((30999109664, -3635869986, -7971837166, 30782131443, -10584905494, 17442936779)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_mul_transpose__structured() {
    let a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(
        s3((30999109664, -3635869986, -7971837166, 30782131443, -10584905494, 17442936779)),
    );
    assert!(a.mul_transpose() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_mul_transpose__generic() {
    let a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(
        s3((30999109664, -3635869986, -7971837166, 30782131443, -10584905494, 17442936779)),
    );
    assert!(sym3_upper(a * a.transpose()) == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_adjugate__cofactors() {
    let a = black_box(
        m3(
            [
                [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                [2873393302, -6638673079, 4752733287],
            ],
        ),
    );
    let e = black_box(
        m3(
            [
                [17250669418, 14980862226, 8880080234], [10472566806, -4443699415, 20791662074],
                [4198844782, -15264095938, -20732826170],
            ],
        ),
    );
    assert!(a.adjugate() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_try_inverse__alt_div() {
    let a = black_box(
        m3(
            [
                [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                [1713407532, -2388220103, 1097729905],
            ],
        ),
    );
    let e = black_box(
        m3(
            [
                [2746796817, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                [-2670352862, -3252013208, -2561850149],
            ],
        ),
    );
    assert!(try_inverse_div(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_try_inverse__alt_div_n() {
    let a = black_box(
        m3(
            [
                [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                [1713407532, -2388220103, 1097729905],
            ],
        ),
    );
    let e = black_box(
        m3(
            [
                [2746796817, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                [-2670352862, -3252013208, -2561850149],
            ],
        ),
    );
    assert!(try_inverse_div_n(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix3_try_inverse__alt_recip() {
    let a = black_box(
        m3(
            [
                [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
                [1713407532, -2388220103, 1097729905],
            ],
        ),
    );
    let e = black_box(
        m3(
            [
                [2746796816, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
                [-2670352862, -3252013209, -2561850150],
            ],
        ),
    );
    assert!(try_inverse_recip(a).unwrap() == e);
}
