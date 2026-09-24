use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, int, m4, m4i, max_ulp_diff4, ulp_diff, v4i, v4t};
use crate::base::{oracle_matrix4, oracle_matrix4_inverse};
use super::{Matrix4, Matrix4InternalTrait, Matrix4Trait};

// --- losing candidates of the determinant / inverse study (kept as evidence) -----------------

/// 3x3 determinant of the given rows (cofactor expansion along the first one).
fn det3(r1: (Fixed, Fixed, Fixed), r2: (Fixed, Fixed, Fixed), r3: (Fixed, Fixed, Fixed)) -> Fixed {
    let ((a, b, c), (d, e, f), (g, h, i)) = (r1, r2, r3);
    Real::sum_prod3(
        a,
        Real::diff_prod(e, i, f, h),
        b,
        Real::diff_prod(f, g, d, i),
        c,
        Real::diff_prod(d, h, e, g),
    )
}

/// Determinant by cofactor expansion along the first row (four 3x3 determinants): three
/// roundings and 12 `diff_prod` + 4 `sum_prod3` + 4 products, against two roundings and 12
/// `diff_prod` + 6 products for the 2x2 minors.
fn determinant_row_cofactors(m: Matrix4<Fixed>) -> Fixed {
    let c1 = det3((m.m22, m.m23, m.m24), (m.m32, m.m33, m.m34), (m.m42, m.m43, m.m44));
    let c2 = det3((m.m21, m.m23, m.m24), (m.m31, m.m33, m.m34), (m.m41, m.m43, m.m44));
    let c3 = det3((m.m21, m.m22, m.m24), (m.m31, m.m32, m.m34), (m.m41, m.m42, m.m44));
    let c4 = det3((m.m21, m.m22, m.m23), (m.m31, m.m32, m.m33), (m.m41, m.m42, m.m43));
    let w = Real::wide_add_prod(Real::<Fixed>::wide_zero(), m.m11, c1);
    let w = Real::wide_sub_prod(w, m.m12, c2);
    let w = Real::wide_add_prod(w, m.m13, c3);
    Real::wide_rescale(Real::wide_sub_prod(w, m.m14, c4))
}

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

// --- constructors and accessors --------------------------------------------------------------

#[test]
fn test_new_is_row_major() {
    let m = Matrix4Trait::new(
        int(1),
        int(2),
        int(3),
        int(4),
        int(5),
        int(6),
        int(7),
        int(8),
        int(9),
        int(10),
        int(11),
        int(12),
        int(13),
        int(14),
        int(15),
        int(16),
    );
    assert!(m == m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]));
    assert!(m.m12 == int(2) && m.m21 == int(5));
}

#[test]
fn test_serde_is_column_major() {
    let m = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    let mut out = array![];
    m.serialize(ref out);
    let one: felt252 = 0x100000000;
    assert!(
        out == array![
            1 * one, 5 * one, 9 * one, 13 * one, 2 * one, 6 * one, 10 * one, 14 * one, 3 * one,
            7 * one, 11 * one, 15 * one, 4 * one, 8 * one, 12 * one, 16 * one,
        ],
    );
}

#[test]
fn test_zeros_identity() {
    assert!(
        Matrix4Trait::<
            Fixed,
        >::zeros() == m4i([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    );
    assert!(
        Matrix4Trait::<
            Fixed,
        >::identity() == m4i([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    );
    assert!(Matrix4Trait::<Fixed>::zeros() == Default::default());
}

#[test]
fn test_from_diagonal() {
    assert!(
        Matrix4Trait::from_diagonal(
            v4i(2, -3, 4, -5),
        ) == m4i([[2, 0, 0, 0], [0, -3, 0, 0], [0, 0, 4, 0], [0, 0, 0, -5]]),
    );
    assert!(
        Matrix4Trait::from_diagonal_element(
            int(7),
        ) == m4i([[7, 0, 0, 0], [0, 7, 0, 0], [0, 0, 7, 0], [0, 0, 0, 7]]),
    );
}

#[test]
fn test_from_columns_from_rows() {
    let (r1, r2, r3, r4) = (
        v4i(1, 2, 3, 4), v4i(5, 6, 7, 8), v4i(9, 10, 11, 12), v4i(13, 14, 15, 16),
    );
    assert!(
        Matrix4Trait::from_rows(
            r1, r2, r3, r4,
        ) == m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
    );
    assert!(
        Matrix4Trait::from_columns(
            r1, r2, r3, r4,
        ) == m4i([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]),
    );
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

// --- exact operations
// --------------------------------------------------------------------------

#[test]
fn test_add_oracle() {
    let mut cases = oracle_matrix4::matrix4_add_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _) = *case;
        assert!(m4(a) + m4(b) == m4(expected));
        let mut acc = m4(a);
        acc += m4(b);
        assert!(acc == m4(expected));
    }
}

#[test]
fn test_sub_oracle() {
    let mut cases = oracle_matrix4::matrix4_sub_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _) = *case;
        assert!(m4(a) - m4(b) == m4(expected));
        let mut acc = m4(a);
        acc -= m4(b);
        assert!(acc == m4(expected));
    }
}

#[test]
fn test_neg() {
    let m = m4i([[1, -2, 3, -4], [-5, 6, -7, 8], [9, -10, 11, -12], [-13, 14, -15, 0]]);
    assert!(-m == m4i([[-1, 2, -3, 4], [5, -6, 7, -8], [-9, 10, -11, 12], [13, -14, 15, 0]]));
    assert!(m + (-m) == Matrix4Trait::zeros());
}

#[test]
#[should_panic(expected: 'i64_add Overflow')]
fn test_add_overflow_panics() {
    let m = black_box(Matrix4Trait::from_diagonal_element(Real::<Fixed>::max_value().unwrap()));
    let _ = m + m;
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_neg_min_panics() {
    let _ = -black_box(Matrix4Trait::from_diagonal_element(Real::<Fixed>::min_value().unwrap()));
}

#[test]
fn test_transpose_oracle() {
    let mut cases = oracle_matrix4::matrix4_transpose_cases();
    while let Some(case) = cases.pop_front() {
        let (a, expected, _) = *case;
        assert!(m4(a).transpose() == m4(expected));
        assert!(m4(a).transpose().transpose() == m4(a));
    }
}

#[test]
fn test_trace_oracle() {
    let mut cases = oracle_matrix4::matrix4_trace_cases();
    while let Some(case) = cases.pop_front() {
        let (a, expected, _) = *case;
        assert!(m4(a).trace() == fx(expected));
    }
}

#[test]
fn test_abs() {
    let m = m4i([[1, -2, 3, -4], [-5, 6, -7, 8], [9, -10, 11, -12], [-13, 14, -15, 0]]);
    assert!(m.abs() == m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]));
}

// --- products
// ----------------------------------------------------------------------------------

#[test]
fn test_scale_oracle() {
    let mut cases = oracle_matrix4::matrix4_scale_cases();
    while let Some(case) = cases.pop_front() {
        let (a, k, expected, _) = *case;
        assert!(m4(a).scale(fx(k)) == m4(expected));
    }
}

#[test]
fn test_component_mul_exact() {
    let a = m4i([[1, -2, 3, -4], [-4, 5, -6, 7], [7, -8, 0, 1], [2, 2, -2, -2]]);
    let b = m4i([[2, 2, 2, 2], [3, 3, 3, 3], [-1, -1, -1, -1], [5, 5, 5, 5]]);
    assert!(
        a
            .component_mul(
                b,
            ) == m4i([[2, -4, 6, -8], [-12, 15, -18, 21], [-7, 8, 0, -1], [10, 10, -10, -10]]),
    );
    // 0.5 ulp floors to 0, -0.5 ulp to -1 ulp, 1.5 ulp to 1 ulp, -1.5 ulp to -2 ulp.
    let h = Matrix4Trait::from_diagonal_element(Real::<Fixed>::HALF);
    let e = Matrix4Trait::from_diagonal(v4t((1, -1, 3, -3)));
    assert!(e.component_mul(h) == Matrix4Trait::from_diagonal(v4t((0, -1, 1, -2))));
}

#[test]
fn test_mul_oracle() {
    let mut cases = oracle_matrix4::matrix4_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _) = *case;
        assert!(m4(a) * m4(b) == m4(expected));
        let mut acc = m4(a);
        acc *= m4(b);
        assert!(acc == m4(expected));
    }
}

#[test]
fn test_mul_exact_and_identity() {
    let a = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    let b = m4i([[-1, 0, 2, 1], [3, 1, 0, 0], [0, -2, 1, 2], [1, 0, 0, -1]]);
    assert!(a * b == m4i([[9, -4, 5, 3], [21, -8, 17, 11], [33, -12, 29, 19], [45, -16, 41, 27]]));
    assert!(a * Matrix4Trait::identity() == a);
    assert!(Matrix4Trait::identity() * a == a);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_mul_overflow_panics() {
    let m = black_box(Matrix4Trait::from_diagonal_element(int(65536)));
    let _ = m * m;
}

#[test]
fn test_mul_vec_oracle() {
    let mut cases = oracle_matrix4::matrix4_mul_vec_cases();
    while let Some(case) = cases.pop_front() {
        let (a, v, expected, _) = *case;
        assert!(m4(a).mul_vec(v4t(v)) == v4t(expected));
        assert!(m4(a).transpose().tr_mul_vec(v4t(v)) == v4t(expected));
    }
}

#[test]
fn test_mul_vec_tr_mul_vec_exact() {
    let a = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    assert!(a.mul_vec(v4i(1, 0, -1, 2)) == v4i(6, 14, 22, 30));
    assert!(a.tr_mul_vec(v4i(1, 0, -1, 2)) == v4i(18, 20, 22, 24));
}

#[test]
fn test_tr_mul_oracle() {
    let mut cases = oracle_matrix4::matrix4_tr_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _) = *case;
        assert!(m4(a).tr_mul(m4(b)) == m4(expected));
        assert!(m4(a).transpose() * m4(b) == m4(expected));
    }
}

#[test]
fn test_from_outer_oracle() {
    let mut cases = oracle_matrix4::matrix4_outer_cases();
    while let Some(case) = cases.pop_front() {
        let (u, v, expected, _) = *case;
        assert!(Matrix4InternalTrait::from_outer(v4t(u), v4t(v)) == m4(expected));
    }
}

// --- norms
// -------------------------------------------------------------------------------------

#[test]
fn test_norm_exact() {
    let m = m4i([[2, -2, 2, -2], [-2, 2, -2, 2], [2, -2, 2, -2], [-2, 2, -2, 2]]);
    assert!(m.norm_squared() == int(64));
    assert!(m.norm() == int(8));
    assert!(Matrix4Trait::<Fixed>::zeros().norm() == int(0));
    // The squared norm (4 * 2^58) does not fit, the norm does.
    assert!(Matrix4Trait::from_diagonal_element(int(0x20000000)).norm() >= int(0x20000000));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_norm_squared_overflow_panics() {
    black_box(Matrix4Trait::from_diagonal_element(int(0x20000000))).norm_squared();
}

// --- determinant and inverse
// -------------------------------------------------------------------

#[test]
fn test_determinant_exact() {
    assert!(
        m4i([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, -4, 0], [0, 0, 0, 5]]).determinant() == int(-120),
    );
    assert!(
        m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
            .determinant() == int(0),
    );
    assert!(
        m4i([[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]]).determinant() == int(30),
    );
    assert!(Matrix4Trait::<Fixed>::identity().determinant() == int(1));
}

#[test]
fn test_determinant_oracle() {
    let mut cases = oracle_matrix4_inverse::matrix4_determinant_cases();
    let mut worst = 0;
    let mut worst_alt = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let err = ulp_diff(m4(a).determinant(), fx(expected));
        assert!(err <= tol.into(), "determinant error {err} > {tol}");
        worst = core::cmp::max(worst, err);
        worst_alt =
            core::cmp::max(worst_alt, ulp_diff(determinant_row_cofactors(m4(a)), fx(expected)));
    }
    // Worst error over the oracle (exact expectation, |a_ij| up to 1e3): 18993 ulp, against
    // 35351 for the row cofactors.
    assert!(worst == 18993, "worst {worst}");
    assert!(worst_alt == 35351, "worst alt {worst_alt}");
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_determinant_overflow_panics() {
    black_box(Matrix4Trait::from_diagonal_element(int(256))).determinant();
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
fn test_try_inverse_exact() {
    let m = m4i([[1, 1, 0, 2], [2, 3, 3, 4], [0, 3, 10, 1], [1, 1, 2, 5]]);
    let inv = m.try_inverse().unwrap();
    assert!(inv == m4i([[86, -40, 13, -5], [-59, 28, -9, 3], [19, -9, 3, -1], [-13, 6, -2, 1]]));
    assert!(m * inv == Matrix4Trait::identity());
    let d = Matrix4Trait::from_diagonal(v4i(2, -4, 8, -16)).try_inverse().unwrap();
    assert!(
        d == Matrix4Trait::from_diagonal(v4t((0x80000000, -0x40000000, 0x20000000, -0x10000000))),
    );
    assert!(Matrix4Trait::<Fixed>::identity().try_inverse().unwrap() == Matrix4Trait::identity());
}

#[test]
fn test_try_inverse_oracle() {
    let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let err = max_ulp_diff4(m4(a).try_inverse().unwrap(), m4(expected));
        assert!(err <= tol.into(), "inverse error {err} > {tol}");
    }
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
fn test_try_inverse_product_is_identity() {
    let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let inv = m4(a).try_inverse().unwrap();
        // Worst residual over the oracle: 71 ulp.
        assert!((m4(a) * inv).is_identity(71));
        assert!((inv * m4(a)).is_identity(71));
    }
}

#[test]
fn test_try_inverse_singular_oracle() {
    let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_singular_cases();
    while let Some(case) = cases.pop_front() {
        let (a, is_some, _) = *case;
        assert!(m4(a).try_inverse().is_some() == is_some);
    }
    assert!(Matrix4Trait::<Fixed>::zeros().try_inverse().is_none());
    assert!(
        m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
            .try_inverse()
            .is_none(),
    );
}

#[test]
fn test_try_inverse_near_singular_oracle() {
    let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_near_singular_cases();
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let inv = m4(a).try_inverse().unwrap();
        assert!(max_ulp_diff4(inv, m4(expected)) <= tol.into());
    }
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
#[should_panic(expected: 'Fixed: overflow')]
fn test_try_inverse_tiny_norm_panics() {
    let _ = black_box(Matrix4Trait::from_diagonal_element(fx(1))).try_inverse();
}

// --- approximate equality
// ----------------------------------------------------------------------

#[test]
fn test_is_identity_abs_diff_eq() {
    let id = Matrix4Trait::<Fixed>::identity();
    let mut m = id;
    m.m21 = fx(3);
    m.m44 = fx(0x100000000 - 2);
    assert!(m.is_identity(3) && !m.is_identity(2));
    assert!(m.abs_diff_eq(id, 3) && !m.abs_diff_eq(id, 2));
    assert!(id.is_identity(0) && id.abs_diff_eq(id, 0));
}

// --- gas benchmarks
// ----------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_matrix4_new__baseline() {
    let _a = black_box(
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
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_new__struct() {
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
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(
        Matrix4Trait::new(
            a.m11,
            a.m12,
            a.m13,
            a.m14,
            a.m21,
            a.m22,
            a.m23,
            a.m24,
            a.m31,
            a.m32,
            a.m33,
            a.m34,
            a.m41,
            a.m42,
            a.m43,
            a.m44,
        ) == e,
    );
}

#[test]
#[inline(never)]
fn bench_matrix4_zeros__baseline() {
    let e = black_box(m4([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_zeros__const() {
    let e = black_box(m4([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]));
    assert!(Matrix4Trait::<Fixed>::zeros() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_identity__baseline() {
    let e = black_box(
        m4(
            [
                [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                [0, 0, 0, 4294967296],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_identity__const() {
    let e = black_box(
        m4(
            [
                [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                [0, 0, 0, 4294967296],
            ],
        ),
    );
    assert!(Matrix4Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_diagonal__baseline() {
    let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(
        m4(
            [
                [4751241150, 0, 0, 0], [0, 2551995575, 0, 0], [0, 0, -4086721614, 0],
                [0, 0, 0, -3145554884],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_diagonal__struct() {
    let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(
        m4(
            [
                [4751241150, 0, 0, 0], [0, 2551995575, 0, 0], [0, 0, -4086721614, 0],
                [0, 0, 0, -3145554884],
            ],
        ),
    );
    assert!(Matrix4Trait::from_diagonal(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_diagonal_element__baseline() {
    let _k = black_box(fx(-7516192768));
    let e = black_box(
        m4(
            [
                [-7516192768, 0, 0, 0], [0, -7516192768, 0, 0], [0, 0, -7516192768, 0],
                [0, 0, 0, -7516192768],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_diagonal_element__struct() {
    let k = black_box(fx(-7516192768));
    let e = black_box(
        m4(
            [
                [-7516192768, 0, 0, 0], [0, -7516192768, 0, 0], [0, 0, -7516192768, 0],
                [0, 0, 0, -7516192768],
            ],
        ),
    );
    assert!(Matrix4Trait::from_diagonal_element(k) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_columns__baseline() {
    let _c1 = black_box(v4t((6065401010, -4721681308, 7990534196, 4445892797)));
    let _c2 = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
    let _c3 = black_box(v4t((-4169889720, 2695648994, -7212805291, 6546834464)));
    let _c4 = black_box(v4t((-3730684931, 6053150518, -2184064506, 5336335505)));
    let e = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_columns__struct() {
    let c1 = black_box(v4t((6065401010, -4721681308, 7990534196, 4445892797)));
    let c2 = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
    let c3 = black_box(v4t((-4169889720, 2695648994, -7212805291, 6546834464)));
    let c4 = black_box(v4t((-3730684931, 6053150518, -2184064506, 5336335505)));
    let e = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(Matrix4Trait::from_columns(c1, c2, c3, c4) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_rows__baseline() {
    let _r1 = black_box(v4t((6065401010, -2161577173, -4169889720, -3730684931)));
    let _r2 = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
    let _r3 = black_box(v4t((7990534196, 5603363342, -7212805291, -2184064506)));
    let _r4 = black_box(v4t((4445892797, 8226859758, 6546834464, 5336335505)));
    let e = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_rows__struct() {
    let r1 = black_box(v4t((6065401010, -2161577173, -4169889720, -3730684931)));
    let r2 = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
    let r3 = black_box(v4t((7990534196, 5603363342, -7212805291, -2184064506)));
    let r4 = black_box(v4t((4445892797, 8226859758, 6546834464, 5336335505)));
    let e = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(Matrix4Trait::from_rows(r1, r2, r3, r4) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_from_outer__baseline() {
    let _u = black_box(v4t((-6975932915, 4075315203, -4507549853, -6978514818)));
    let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
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
    assert!(e == e);
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
fn bench_matrix4_column__baseline() {
    let _a = black_box(
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
    assert!(e == e);
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
fn bench_matrix4_row__baseline() {
    let _a = black_box(
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
    assert!(e == e);
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
fn bench_matrix4_diagonal__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(v4t((6065401010, -7621971610, -7212805291, 5336335505)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_diagonal__struct() {
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
    let e = black_box(v4t((6065401010, -7621971610, -7212805291, 5336335505)));
    assert!(a.diagonal() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_transpose__baseline() {
    let _a = black_box(
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
                [6065401010, -4721681308, 7990534196, 4445892797],
                [-2161577173, -7621971610, 5603363342, 8226859758],
                [-4169889720, 2695648994, -7212805291, 6546834464],
                [-3730684931, 6053150518, -2184064506, 5336335505],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_transpose__struct() {
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
                [6065401010, -4721681308, 7990534196, 4445892797],
                [-2161577173, -7621971610, 5603363342, 8226859758],
                [-4169889720, 2695648994, -7212805291, 6546834464],
                [-3730684931, 6053150518, -2184064506, 5336335505],
            ],
        ),
    );
    assert!(a.transpose() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_trace__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(fx(-3433040386));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_trace__sum() {
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
    let e = black_box(fx(-3433040386));
    assert!(a.trace() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_abs__baseline() {
    let _a = black_box(
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
                [6065401010, 2161577173, 4169889720, 3730684931],
                [4721681308, 7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, 7212805291, 2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_abs__componentwise() {
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
                [6065401010, 2161577173, 4169889720, 3730684931],
                [4721681308, 7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, 7212805291, 2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    assert!(a.abs() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_add__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [12489641950, -10675106085, -11071906975, 4456957075],
                [-1054211790, -4741980083, -5387795612, 3049489065],
                [10821163348, -585825044, -14658492792, 4310724722],
                [12527228599, 1650094630, -435296038, 843506602],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_add__operator() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [12489641950, -10675106085, -11071906975, 4456957075],
                [-1054211790, -4741980083, -5387795612, 3049489065],
                [10821163348, -585825044, -14658492792, 4310724722],
                [12527228599, 1650094630, -435296038, 843506602],
            ],
        ),
    );
    assert!(a + b == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_add__assign() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [12489641950, -10675106085, -11071906975, 4456957075],
                [-1054211790, -4741980083, -5387795612, 3049489065],
                [10821163348, -585825044, -14658492792, 4310724722],
                [12527228599, 1650094630, -435296038, 843506602],
            ],
        ),
    );
    let mut r = a;
    r += b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_sub__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-358839930, 6351951739, 2732127535, -11918326937],
                [-8389150826, -10501963137, 10779093600, 9056811971],
                [5159905044, 11792551728, 232882210, -8678853734],
                [-3635443005, 14803624886, 13528964966, 9829164408],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_sub__operator() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-358839930, 6351951739, 2732127535, -11918326937],
                [-8389150826, -10501963137, 10779093600, 9056811971],
                [5159905044, 11792551728, 232882210, -8678853734],
                [-3635443005, 14803624886, 13528964966, 9829164408],
            ],
        ),
    );
    assert!(a - b == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_sub__assign() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-358839930, 6351951739, 2732127535, -11918326937],
                [-8389150826, -10501963137, 10779093600, 9056811971],
                [5159905044, 11792551728, 232882210, -8678853734],
                [-3635443005, 14803624886, 13528964966, 9829164408],
            ],
        ),
    );
    let mut r = a;
    r -= b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_neg__baseline() {
    let _a = black_box(
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
                [-6065401010, 2161577173, 4169889720, 3730684931],
                [4721681308, 7621971610, -2695648994, -6053150518],
                [-7990534196, -5603363342, 7212805291, 2184064506],
                [-4445892797, -8226859758, -6546834464, -5336335505],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_neg__operator() {
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
                [-6065401010, 2161577173, 4169889720, 3730684931],
                [4721681308, 7621971610, -2695648994, -6053150518],
                [-7990534196, -5603363342, 7212805291, 2184064506],
                [-4445892797, -8226859758, -6546834464, -5336335505],
            ],
        ),
    );
    assert!(-a == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_scale__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _k = black_box(fx(-7516192768));
    let e = black_box(
        m4(
            [
                [-10614451768, 3782760052, 7297307010, 6528698629],
                [8262942289, 13338450317, -4717385740, -10593013407],
                [-13983434843, -9805885849, 12622409259, 3822112885],
                [-7780312395, -14397004577, -11456960312, -9338587134],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_scale__products() {
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
    let k = black_box(fx(-7516192768));
    let e = black_box(
        m4(
            [
                [-10614451768, 3782760052, 7297307010, 6528698629],
                [8262942289, 13338450317, -4717385740, -10593013407],
                [-13983434843, -9805885849, 12622409259, 3822112885],
                [-7780312395, -14397004577, -11456960312, -9338587134],
            ],
        ),
    );
    assert!(a.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_component_mul__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [9072385143, 4284700788, 6701017450, -7111931372],
                [-4031840310, -5110915205, -5073409835, -4233237096],
                [5266219152, -8074629894, 12504005386, -3302711674],
                [8365314601, -12597563763, -10642887234, -5582171119],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_component_mul__products() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [9072385143, 4284700788, 6701017450, -7111931372],
                [-4031840310, -5110915205, -5073409835, -4233237096],
                [5266219152, -8074629894, 12504005386, -3302711674],
                [8365314601, -12597563763, -10642887234, -5582171119],
            ],
        ),
    );
    assert!(a.component_mul(b) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_mul_vec__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(v4t((12125377576, -16750294840, 20631480820, -331180081)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_mul_vec__fused() {
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
    let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(v4t((12125377576, -16750294840, 20631480820, -331180081)));
    assert!(a.mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_tr_mul_vec__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(v4t((-6954980908, -18276934794, -942863330, -2360400514)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_tr_mul_vec__fused() {
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
    let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(v4t((-6954980908, -18276934794, -942863330, -2360400514)));
    assert!(a.tr_mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_tr_mul_vec__transpose_mul_vec() {
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
    let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
    let e = black_box(v4t((-6954980908, -18276934794, -942863330, -2360400514)));
    assert!(a.transpose().mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_mul__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-2541171855, -1750704652, 7614798780, 10671269236],
                [-404809204, -8905090396, 7419396255, -5926404020],
                [7873481502, 1656702967, -7332201226, 2691523401],
                [28030393544, -20901758122, -42652634060, 7039807887],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_mul__fused() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-2541171855, -1750704652, 7614798780, 10671269236],
                [-404809204, -8905090396, 7419396255, -5926404020],
                [7873481502, 1656702967, -7332201226, 2691523401],
                [28030393544, -20901758122, -42652634060, 7039807887],
            ],
        ),
    );
    assert!(a * b == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_mul__assign() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [-2541171855, -1750704652, 7614798780, 10671269236],
                [-404809204, -8905090396, 7419396255, -5926404020],
                [7873481502, 1656702967, -7332201226, 2691523401],
                [28030393544, -20901758122, -42652634060, 7039807887],
            ],
        ),
    );
    let mut r = a;
    r *= b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_tr_mul__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [18672078587, -33511520172, -21940301505, 22297227754],
                [9430856166, -21498408073, -5269160636, 1077166020],
                [3629416479, 10442068122, 3488725768, -27589927991],
                [8189903724, 6429869131, -10286035073, -20230051259],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_tr_mul__fused() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [18672078587, -33511520172, -21940301505, 22297227754],
                [9430856166, -21498408073, -5269160636, 1077166020],
                [3629416479, 10442068122, 3488725768, -27589927991],
                [8189903724, 6429869131, -10286035073, -20230051259],
            ],
        ),
    );
    assert!(a.tr_mul(b) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_tr_mul__transpose_mul() {
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
    let b = black_box(
        m4(
            [
                [6424240940, -8513528912, -6902017255, 8187642006],
                [3667469518, 2879991527, -8083444606, -3003661453],
                [2830629152, -6189188386, -7445687501, 6494789228],
                [8081335802, -6576765128, -6982130502, -4492828903],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [18672078587, -33511520172, -21940301505, 22297227754],
                [9430856166, -21498408073, -5269160636, 1077166020],
                [3629416479, 10442068122, 3488725768, -27589927991],
                [8189903724, 6429869131, -10286035073, -20230051259],
            ],
        ),
    );
    assert!(a.transpose() * b == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_norm_squared__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(fx(118252144586));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_norm_squared__wide() {
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
    let e = black_box(fx(118252144586));
    assert!(a.norm_squared() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_norm__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(fx(22536394868));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_norm__wide() {
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
    let e = black_box(fx(22536394868));
    assert!(a.norm() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_determinant__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(fx(53942245227));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_determinant__minors2x2() {
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
    let e = black_box(fx(53942245227));
    assert!(a.determinant() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_determinant__alt_row_cofactors() {
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
    let e = black_box(fx(53942245230));
    assert!(determinant_row_cofactors(a) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_adjugate__baseline() {
    let _a = black_box(
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
    assert!(e == e);
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
fn bench_matrix4_try_inverse__baseline() {
    let _a = black_box(
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
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse__prescaled_det_ge_half() {
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
    assert!(a.try_inverse().unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse__prescaled_norm_gt_one() {
    let a = black_box(
        m4(
            [
                [123994628, -2146559718, 1588261118, -1698136391],
                [-1258299206, 94776244, 262037008, -1219409418],
                [908609721, -228549883, -1944374345, -1297968465],
                [-1509590524, -2094158678, -5668301786, 461699291],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [1007590045, -8113460287, 5589193161, -2009983769],
                [-6224010726, 4566957182, 2947460227, -2543902749],
                [1941005617, -48662219, -3082533347, -1655358742],
                [-1106373953, -6410886765, -6200760652, 1520649381],
            ],
        ),
    );
    assert!(a.try_inverse().unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse__prescaled_small() {
    let a = black_box(
        m4(
            [
                [445290890, 496684497, -552302081, 121652393],
                [-769872670, -183641367, -1342383347, 1104390896],
                [143659673, -109440825, 146151883, 495732633],
                [-235861466, 489393595, 274460440, 427305163],
            ],
        ),
    );
    let e = black_box(
        m4(
            [
                [15582438276, -6677561308, 22505666527, -13287450710],
                [12081009333, -2389516454, -15753444996, 21012555187],
                [-9746308220, -6782176897, 9363645371, 9440493449],
                [1024804345, 3407105854, 24450685987, 5706213630],
            ],
        ),
    );
    assert!(a.try_inverse().unwrap() == e);
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

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse_singular__baseline() {
    let _a = black_box(
        m4(
            [
                [-8589934592, 4294967296, 12884901888, 21474836480],
                [17179869184, 8589934592, -85899345920, -34359738368],
                [8589934592, 0, -12884901888, 12884901888],
                [8589934592, -8589934592, 17179869184, 8589934592],
            ],
        ),
    );
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_try_inverse_singular__none() {
    let a = black_box(
        m4(
            [
                [-8589934592, 4294967296, 12884901888, 21474836480],
                [17179869184, 8589934592, -85899345920, -34359738368],
                [8589934592, 0, -12884901888, 12884901888],
                [8589934592, -8589934592, 17179869184, 8589934592],
            ],
        ),
    );
    let e = black_box(true);
    assert!(a.try_inverse().is_none() == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_is_identity__baseline() {
    let _a = black_box(
        m4(
            [
                [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                [0, 0, 0, 4294967296],
            ],
        ),
    );
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_is_identity__all_compared() {
    let a = black_box(
        m4(
            [
                [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                [0, 0, 0, 4294967296],
            ],
        ),
    );
    let e = black_box(true);
    assert!(a.is_identity(2) == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_abs_diff_eq__baseline() {
    let _a = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let _b = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_matrix4_abs_diff_eq__all_compared() {
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
    let b = black_box(
        m4(
            [
                [6065401010, -2161577173, -4169889720, -3730684931],
                [-4721681308, -7621971610, 2695648994, 6053150518],
                [7990534196, 5603363342, -7212805291, -2184064506],
                [4445892797, 8226859758, 6546834464, 5336335505],
            ],
        ),
    );
    let e = black_box(true);
    assert!(a.abs_diff_eq(b, 2) == e);
}
