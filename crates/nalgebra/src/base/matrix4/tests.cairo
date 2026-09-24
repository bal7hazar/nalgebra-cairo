//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/matrix4/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, int, m4, m4i, max_ulp_diff4, v4i, v4t};
use crate::base::{oracle_matrix4, oracle_matrix4_inverse};
use super::{Matrix4, Matrix4InternalTrait, Matrix4Trait};

/// `adjugate / determinant` without the integer pre-scaling of small matrices.
fn try_inverse_div(m: Matrix4<Fixed>) -> Option<Matrix4<Fixed>> {
    let adj = m.adjugate();
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    Some(
        Matrix4 {
            m11: adj.m11 / det,
            m21: adj.m21 / det,
            m31: adj.m31 / det,
            m41: adj.m41 / det,
            m12: adj.m12 / det,
            m22: adj.m22 / det,
            m32: adj.m32 / det,
            m42: adj.m42 / det,
            m13: adj.m13 / det,
            m23: adj.m23 / det,
            m33: adj.m33 / det,
            m43: adj.m43 / det,
            m14: adj.m14 / det,
            m24: adj.m24 / det,
            m34: adj.m34 / det,
            m44: adj.m44 / det,
        },
    )
}

/// Upstream's `adjugate / determinant` (the formula of `try_inverse_div`) through ONE prepared
/// divisor (`Real::div16`): bit-identical to `try_inverse_div`, so it fails the same oracle
/// cases; kept to price upstream's formula at its cheapest (WP 7.2).
fn try_inverse_div_n(m: Matrix4<Fixed>) -> Option<Matrix4<Fixed>> {
    let adj = m.adjugate();
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    let (m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44) =
        Real::div16(
        adj.m11,
        adj.m21,
        adj.m31,
        adj.m41,
        adj.m12,
        adj.m22,
        adj.m32,
        adj.m42,
        adj.m13,
        adj.m23,
        adj.m33,
        adj.m43,
        adj.m14,
        adj.m24,
        adj.m34,
        adj.m44,
        det,
    );
    Some(Matrix4 { m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44 })
}

/// `adjugate * (1 / determinant)`: one reciprocal, 16 multiplications.
fn try_inverse_recip(m: Matrix4<Fixed>) -> Option<Matrix4<Fixed>> {
    let det = m.determinant();
    if det == Real::zero() {
        return None;
    }
    Some(m.adjugate().scale(det.recip()))
}

/// `(cases above the oracle tolerance, worst error in ulp)` of an inverse candidate.
fn inverse_failures(variant: u8) -> (u32, u128) {
    let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_cases();
    let mut failures = 0;
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = match variant {
            0 => m4(a).try_inverse(),
            1 => try_inverse_div(m4(a)),
            3 => try_inverse_div_n(m4(a)),
            _ => try_inverse_recip(m4(a)),
        };
        let err = max_ulp_diff4(got.unwrap(), m4(expected));
        if err > tol.into() {
            failures += 1;
        }
        worst = core::cmp::max(worst, err);
    }
    (failures, worst)
}

#[test]
fn test_columns_rows_diagonal() {
    let m = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    assert!(m.column1() == v4i(1, 5, 9, 13));
    assert!(m.column2() == v4i(2, 6, 10, 14));
    assert!(m.column3() == v4i(3, 7, 11, 15));
    assert!(m.column4() == v4i(4, 8, 12, 16));
    assert!(m.row1() == v4i(1, 2, 3, 4));
    assert!(m.row2() == v4i(5, 6, 7, 8));
    assert!(m.row3() == v4i(9, 10, 11, 12));
    assert!(m.row4() == v4i(13, 14, 15, 16));
    assert!(m.diagonal() == v4i(1, 6, 11, 16));
    assert!(Matrix4Trait::from_columns(m.column1(), m.column2(), m.column3(), m.column4()) == m);
    assert!(Matrix4Trait::from_rows(m.row1(), m.row2(), m.row3(), m.row4()) == m);
}

#[test]
fn test_from_outer_oracle() {
    let mut cases = oracle_matrix4::matrix4_outer_cases();
    while let Some(case) = cases.pop_front() {
        let (u, v, expected, _) = *case;
        assert!(Matrix4InternalTrait::from_outer(v4t(u), v4t(v)) == m4(expected));
    }
}

#[test]
fn test_adjugate_identity() {
    let m = m4i([[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]]);
    let det = m.determinant();
    assert!(det == int(30));
    assert!(m * m.adjugate() == Matrix4Trait::from_diagonal_element(det));
    assert!(m.adjugate() * m == Matrix4Trait::from_diagonal_element(det));
}

#[test]
fn test_try_inverse_candidates_error() {
    // Oracle, 30 well-conditioned matrices (10 small, 10 unit, 10 medium):
    // (cases above the oracle tolerance, worst error in ulp) of the shipped algorithm, of
    // `adjugate / det` without pre-scaling and of `adjugate * (1 / det)`.
    assert!(inverse_failures(0) == (0, 50));
    assert!(inverse_failures(1) == (6, 56713));
    // Upstream's formula through one prepared divisor: the same bits as `try_inverse_div`.
    assert!(inverse_failures(3) == (6, 56713));
    assert!(inverse_failures(2) == (10, 56713));
}

#[test]
fn test_try_inverse_small_scale() {
    // 2^-12 * I: without pre-scaling the determinant (below the resolution).
    let m = Matrix4Trait::from_diagonal_element(fx(0x100000));
    let expected = Matrix4Trait::from_diagonal_element(int(4096));
    assert!(m.determinant() == int(0));
    assert!(try_inverse_div(m).is_none());
    // Relative error below 2^-33 (0 ulp on 4096).
    let inv = m.try_inverse().unwrap();
    assert!(inv.abs_diff_eq(expected, 0));
    assert!(inv == expected);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_outer__products() {
    let u = black_box(v4t((-6975932915, 4075315203, -4507549853, -6978514818)));
    let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(
        m4(
            [
                [-7717017906, -4144979160, 6637697997, 5109044688],
                [4508254419, 2421482085, -3877719568, -2984685740],
                [-4986407317, -2678308469, 4288996898, 3301246430],
                [-7719874096, -4146513282, 6640154714, 5110935626],
            ],
        ),
    );
    assert!(Matrix4InternalTrait::from_outer(u, v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_column__second() {
    let a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
    assert!(a.column2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_row__second() {
    let a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
    assert!(a.row2() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_adjugate__cofactors() {
    let a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [34509485500, 5764173809, -2916270880, 16393887163],
                [-26230675187, -13185528909, 11404010475, 1286032516],
                [22774124647, -13617633838, -34850938698, 17104616577],
                [-16252316022, 32232042512, 27605003334, 6789980084],
            ],
        ),
    );
    assert!(a.adjugate() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse__alt_div() {
    let a = black_box(
        m4(
            [
                [-888332506, 134474598, -3038885559, -4458555824],
                [638823398, 2509477802, -5314665052, 2648272285],
                [-2414734434, -1408008892, -595561448, 2626308541],
                [-585391816, 4129884628, -2943537464, 820202461],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-1580935658, 3651218852, -5141402076, -3920011882],
                [-1134111644, -2131795104, -1434037612, 5310029222],
                [-1976110117, -3309244046, -601828060, 1869987305],
                [-2509710474, 1463757663, 1391329344, -333367901],
            ],
        ),
    );
    assert!(try_inverse_div(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse__alt_div_n() {
    let a = black_box(
        m4(
            [
                [-888332506, 134474598, -3038885559, -4458555824],
                [638823398, 2509477802, -5314665052, 2648272285],
                [-2414734434, -1408008892, -595561448, 2626308541],
                [-585391816, 4129884628, -2943537464, 820202461],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-1580935658, 3651218852, -5141402076, -3920011882],
                [-1134111644, -2131795104, -1434037612, 5310029222],
                [-1976110117, -3309244046, -601828060, 1869987305],
                [-2509710474, 1463757663, 1391329344, -333367901],
            ],
        ),
    );
    assert!(try_inverse_div_n(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse__alt_recip() {
    let a = black_box(
        m4(
            [
                [-888332506, 134474598, -3038885559, -4458555824],
                [638823398, 2509477802, -5314665052, 2648272285],
                [-2414734434, -1408008892, -595561448, 2626308541],
                [-585391816, 4129884628, -2943537464, 820202461],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-1580935658, 3651218852, -5141402076, -3920011882],
                [-1134111645, -2131795105, -1434037613, 5310029221],
                [-1976110118, -3309244046, -601828060, 1869987304],
                [-2509710474, 1463757663, 1391329344, -333367901],
            ],
        ),
    );
    assert!(try_inverse_recip(a).unwrap() == e);
}
