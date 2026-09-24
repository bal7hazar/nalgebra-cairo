#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix_test_utils::{
        fx, int, m3, m3i, max_ulp_diff3, s3, sym3_upper, ulp_diff, v3i, v3t,
    };
    use crate::base::sym_matrix3::SymMatrix3Trait;
    use crate::base::{oracle_matrix3, oracle_matrix3_inverse};
    use super::{Matrix3, Matrix3InternalTrait, Matrix3Trait};

    // --- losing candidates of the determinant / inverse study (kept as evidence) -----------------

    /// Determinant as six triple products, each rounded twice, summed exactly: more gas AND less
    /// precision than the cofactor expansion.
    ///
    /// This is as close to "exact cofactors in a wide accumulator" as `Real` allows: a determinant
    /// term is a product of THREE scalars, and the accumulator only takes products of two
    /// (`wide_add_prod(w, a, b)`), so an unrounded minor cannot be multiplied by `m1j`. A fully
    /// exact 3x3 determinant would need a 96-bit accumulator op in `simba` (`Wide * Fixed`), which
    /// does not exist; see the report of this work package.
    fn determinant_triple_products(m: Matrix3<Fixed>) -> Fixed {
        let w = Real::wide_add_prod(Real::<Fixed>::wide_zero(), m.m11 * m.m22, m.m33);
        let w = Real::wide_add_prod(w, m.m12 * m.m23, m.m31);
        let w = Real::wide_add_prod(w, m.m13 * m.m21, m.m32);
        let w = Real::wide_sub_prod(w, m.m13 * m.m22, m.m31);
        let w = Real::wide_sub_prod(w, m.m12 * m.m21, m.m33);
        Real::wide_rescale(Real::wide_sub_prod(w, m.m11 * m.m23, m.m32))
    }

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

    // --- constructors and accessors --------------------------------------------------------------

    #[test]
    fn test_new_is_row_major() {
        let m = Matrix3Trait::new(
            int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9),
        );
        assert!(m == m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
        assert!(m.m12 == int(2) && m.m21 == int(4));
    }

    #[test]
    fn test_serde_is_column_major() {
        let m = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mut out = array![];
        m.serialize(ref out);
        let one: felt252 = 0x100000000;
        assert!(
            out == array![
                1 * one, 4 * one, 7 * one, 2 * one, 5 * one, 8 * one, 3 * one, 6 * one, 9 * one,
            ],
        );
    }

    #[test]
    fn test_zeros_identity() {
        assert!(Matrix3Trait::<Fixed>::zeros() == m3i([[0, 0, 0], [0, 0, 0], [0, 0, 0]]));
        assert!(Matrix3Trait::<Fixed>::identity() == m3i([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
        assert!(Matrix3Trait::<Fixed>::zeros() == Default::default());
    }

    #[test]
    fn test_from_diagonal() {
        assert!(
            Matrix3Trait::from_diagonal(v3i(2, -3, 4)) == m3i([[2, 0, 0], [0, -3, 0], [0, 0, 4]]),
        );
        assert!(
            Matrix3Trait::from_diagonal_element(int(7)) == m3i([[7, 0, 0], [0, 7, 0], [0, 0, 7]]),
        );
    }

    #[test]
    fn test_from_columns_from_rows() {
        let (r1, r2, r3) = (v3i(1, 2, 3), v3i(4, 5, 6), v3i(7, 8, 9));
        assert!(Matrix3Trait::from_rows(r1, r2, r3) == m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
        assert!(Matrix3Trait::from_columns(r1, r2, r3) == m3i([[1, 4, 7], [2, 5, 8], [3, 6, 9]]));
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

    // --- exact operations
    // --------------------------------------------------------------------------

    #[test]
    fn test_add_oracle() {
        let mut cases = oracle_matrix3::matrix3_add_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a) + m3(b) == m3(expected));
            let mut acc = m3(a);
            acc += m3(b);
            assert!(acc == m3(expected));
        }
    }

    #[test]
    fn test_sub_oracle() {
        let mut cases = oracle_matrix3::matrix3_sub_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a) - m3(b) == m3(expected));
            let mut acc = m3(a);
            acc -= m3(b);
            assert!(acc == m3(expected));
        }
    }

    #[test]
    fn test_neg() {
        let m = m3i([[1, -2, 3], [-4, 5, -6], [7, -8, 0]]);
        assert!(-m == m3i([[-1, 2, -3], [4, -5, 6], [-7, 8, 0]]));
        assert!(m + (-m) == Matrix3Trait::zeros());
    }

    #[test]
    #[should_panic(expected: 'i64_add Overflow')]
    fn test_add_overflow_panics() {
        let m = black_box(Matrix3Trait::from_diagonal_element(Real::<Fixed>::max_value().unwrap()));
        let _ = m + m;
    }

    #[test]
    #[should_panic(expected: 'i64_neg Underflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(
            Matrix3Trait::from_diagonal_element(Real::<Fixed>::min_value().unwrap()),
        );
    }

    #[test]
    fn test_transpose_oracle() {
        let mut cases = oracle_matrix3::matrix3_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m3(a).transpose() == m3(expected));
            assert!(m3(a).transpose().transpose() == m3(a));
        }
    }

    #[test]
    fn test_trace_oracle() {
        let mut cases = oracle_matrix3::matrix3_trace_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m3(a).trace() == fx(expected));
        }
    }

    #[test]
    fn test_abs() {
        let m = m3i([[1, -2, 3], [-4, 5, -6], [7, -8, 0]]);
        assert!(m.abs() == m3i([[1, 2, 3], [4, 5, 6], [7, 8, 0]]));
    }

    // --- products
    // ----------------------------------------------------------------------------------

    #[test]
    fn test_scale_oracle() {
        let mut cases = oracle_matrix3::matrix3_scale_cases();
        while let Some(case) = cases.pop_front() {
            let (a, k, expected, _) = *case;
            assert!(m3(a).scale(fx(k)) == m3(expected));
        }
    }

    #[test]
    fn test_component_mul_exact() {
        let a = m3i([[1, -2, 3], [-4, 5, -6], [7, -8, 0]]);
        let b = m3i([[2, 2, 2], [3, 3, 3], [-1, -1, -1]]);
        assert!(a.component_mul(b) == m3i([[2, -4, 6], [-12, 15, -18], [-7, 8, 0]]));
        // 0.5 ulp floors to 0, -0.5 ulp to -1 ulp, 1.5 ulp to 1 ulp, -1.5 ulp to -2 ulp.
        let h = Matrix3Trait::from_diagonal_element(Real::<Fixed>::HALF);
        let e = Matrix3Trait::from_diagonal(v3t((1, -1, 3)));
        assert!(e.component_mul(h) == Matrix3Trait::from_diagonal(v3t((0, -1, 1))));
    }

    #[test]
    fn test_mul_oracle() {
        let mut cases = oracle_matrix3::matrix3_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a) * m3(b) == m3(expected));
            let mut acc = m3(a);
            acc *= m3(b);
            assert!(acc == m3(expected));
        }
    }

    #[test]
    fn test_mul_exact_and_identity() {
        let a = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let b = m3i([[-1, 0, 2], [3, 1, 0], [0, -2, 1]]);
        assert!(a * b == m3i([[5, -4, 5], [11, -7, 14], [17, -10, 23]]));
        assert!(a * Matrix3Trait::identity() == a);
        assert!(Matrix3Trait::identity() * a == a);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_mul_overflow_panics() {
        let m = black_box(Matrix3Trait::from_diagonal_element(int(65536)));
        let _ = m * m;
    }

    #[test]
    fn test_mul_vec_oracle() {
        let mut cases = oracle_matrix3::matrix3_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, expected, _) = *case;
            assert!(m3(a).mul_vec(v3t(v)) == v3t(expected));
            assert!(m3(a).transpose().tr_mul_vec(v3t(v)) == v3t(expected));
        }
    }

    #[test]
    fn test_mul_vec_tr_mul_vec_exact() {
        let a = m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        assert!(a.mul_vec(v3i(1, 0, -1)) == v3i(-2, -2, -2));
        assert!(a.tr_mul_vec(v3i(1, 0, -1)) == v3i(-6, -6, -6));
    }

    #[test]
    fn test_tr_mul_oracle() {
        let mut cases = oracle_matrix3::matrix3_tr_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m3(a).tr_mul(m3(b)) == m3(expected));
            assert!(m3(a).transpose() * m3(b) == m3(expected));
        }
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
    fn test_cross_matrix_oracle() {
        let mut cases = oracle_matrix3::matrix3_cross_matrix_cases();
        while let Some(case) = cases.pop_front() {
            let (v, expected, _) = *case;
            assert!(Matrix3Trait::cross_matrix(v3t(v)) == m3(expected));
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
                .mul_transpose() == s3(
                    (14 * one, 32 * one, 50 * one, 77 * one, 122 * one, 194 * one),
                ),
        );
    }

    // --- norms
    // -------------------------------------------------------------------------------------

    #[test]
    fn test_norm_exact() {
        let m = m3i([[2, -2, 2], [-2, 2, -2], [2, -2, 2]]);
        assert!(m.norm_squared() == int(36));
        assert!(m.norm() == int(6));
        assert!(Matrix3Trait::<Fixed>::zeros().norm() == int(0));
        // The squared norm (3 * 2^58) does not fit, the norm does.
        assert!(Matrix3Trait::from_diagonal_element(int(0x20000000)).norm() >= int(0x20000000));
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_norm_squared_overflow_panics() {
        black_box(Matrix3Trait::from_diagonal_element(int(0x20000000))).norm_squared();
    }

    // --- determinant and inverse
    // -------------------------------------------------------------------

    #[test]
    fn test_determinant_exact() {
        assert!(m3i([[1, 2, 3], [0, 1, 4], [5, 6, 0]]).determinant() == int(1));
        assert!(m3i([[2, 0, 0], [0, 3, 0], [0, 0, -4]]).determinant() == int(-24));
        assert!(m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).determinant() == int(0));
        assert!(Matrix3Trait::<Fixed>::identity().determinant() == int(1));
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_determinant_cases();
        let mut worst = 0;
        let mut worst_alt = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = ulp_diff(m3(a).determinant(), fx(expected));
            assert!(err <= tol.into(), "determinant error {err} > {tol}");
            worst = core::cmp::max(worst, err);
            worst_alt =
                core::cmp::max(
                    worst_alt, ulp_diff(determinant_triple_products(m3(a)), fx(expected)),
                );
        }
        // Worst error over the oracle (exact expectation, |a_ij| up to 1e3): 556 ulp, against
        // 612 for the triple products.
        assert!(worst == 556, "worst {worst}");
        assert!(worst_alt == 612, "worst alt {worst_alt}");
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_determinant_overflow_panics() {
        black_box(Matrix3Trait::from_diagonal_element(int(2048))).determinant();
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
    fn test_try_inverse_exact() {
        let m = m3i([[1, 2, 3], [0, 1, 4], [5, 6, 0]]);
        let inv = m.try_inverse().unwrap();
        assert!(inv == m3i([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]));
        assert!(m * inv == Matrix3Trait::identity());
        let d = Matrix3Trait::from_diagonal(v3i(2, -4, 8)).try_inverse().unwrap();
        assert!(d == Matrix3Trait::from_diagonal(v3t((0x80000000, -0x40000000, 0x20000000))));
        assert!(
            Matrix3Trait::<Fixed>::identity().try_inverse().unwrap() == Matrix3Trait::identity(),
        );
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = max_ulp_diff3(m3(a).try_inverse().unwrap(), m3(expected));
            assert!(err <= tol.into(), "inverse error {err} > {tol}");
        }
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
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = m3(a).try_inverse().unwrap();
            // Worst residual over the oracle: 104 ulp.
            assert!((m3(a) * inv).is_identity(104));
            assert!((inv * m3(a)).is_identity(104));
        }
    }

    #[test]
    fn test_try_inverse_singular_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, is_some, _) = *case;
            assert!(m3(a).try_inverse().is_some() == is_some);
        }
        assert!(Matrix3Trait::<Fixed>::zeros().try_inverse().is_none());
        assert!(m3i([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).try_inverse().is_none());
    }

    #[test]
    fn test_try_inverse_near_singular_oracle() {
        let mut cases = oracle_matrix3_inverse::matrix3_try_inverse_near_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = m3(a).try_inverse().unwrap();
            assert!(max_ulp_diff3(inv, m3(expected)) <= tol.into());
        }
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
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_tiny_norm_panics() {
        let _ = black_box(Matrix3Trait::from_diagonal_element(fx(1))).try_inverse();
    }

    // --- approximate equality
    // ----------------------------------------------------------------------

    #[test]
    fn test_is_identity_abs_diff_eq() {
        let id = Matrix3Trait::<Fixed>::identity();
        let mut m = id;
        m.m21 = fx(3);
        m.m33 = fx(0x100000000 - 2);
        assert!(m.is_identity(3) && !m.is_identity(2));
        assert!(m.abs_diff_eq(id, 3) && !m.abs_diff_eq(id, 2));
        assert!(id.is_identity(0) && id.abs_diff_eq(id, 0));
    }

    // --- gas benchmarks
    // ----------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_matrix3_new__baseline() {
        let _a = black_box(
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
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_new__struct() {
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
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(
            Matrix3Trait::new(a.m11, a.m12, a.m13, a.m21, a.m22, a.m23, a.m31, a.m32, a.m33) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_zeros__baseline() {
        let e = black_box(m3([[0, 0, 0], [0, 0, 0], [0, 0, 0]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_zeros__const() {
        let e = black_box(m3([[0, 0, 0], [0, 0, 0], [0, 0, 0]]));
        assert!(Matrix3Trait::<Fixed>::zeros() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_identity__baseline() {
        let e = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_identity__const() {
        let e = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        assert!(Matrix3Trait::<Fixed>::identity() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(m3([[6422282562, 0, 0], [0, 6202159288, 0], [0, 0, 2324644860]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal__struct() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(m3([[6422282562, 0, 0], [0, 6202159288, 0], [0, 0, 2324644860]]));
        assert!(Matrix3Trait::from_diagonal(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal_element__baseline() {
        let _k = black_box(fx(-7516192768));
        let e = black_box(m3([[-7516192768, 0, 0], [0, -7516192768, 0], [0, 0, -7516192768]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_diagonal_element__struct() {
        let k = black_box(fx(-7516192768));
        let e = black_box(m3([[-7516192768, 0, 0], [0, -7516192768, 0], [0, 0, -7516192768]]));
        assert!(Matrix3Trait::from_diagonal_element(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_columns__baseline() {
        let _c1 = black_box(v3t((-8333418062, -6195210852, 2873393302)));
        let _c2 = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        let _c3 = black_box(v3t((-7141719772, 5406550886, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_columns__struct() {
        let c1 = black_box(v3t((-8333418062, -6195210852, 2873393302)));
        let c2 = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        let c3 = black_box(v3t((-7141719772, 5406550886, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(Matrix3Trait::from_columns(c1, c2, c3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_rows__baseline() {
        let _r1 = black_box(v3t((-8333418062, -3562322882, -7141719772)));
        let _r2 = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        let _r3 = black_box(v3t((2873393302, -6638673079, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_rows__struct() {
        let r1 = black_box(v3t((-8333418062, -3562322882, -7141719772)));
        let r2 = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        let r3 = black_box(v3t((2873393302, -6638673079, 4752733287)));
        let e = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        assert!(Matrix3Trait::from_rows(r1, r2, r3) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_from_outer__baseline() {
        let _u = black_box(v3t((-2161644290, 7569327059, 4908735083)));
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [-3232315749, -3121528358, -1169986858], [11318446411, 10930507472, 4096887363],
                    [7340052101, 7088472341, 2656845790],
                ],
            ),
        );
        assert!(e == e);
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
    fn bench_matrix3_cross_matrix__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [0, -2324644860, 6202159288], [2324644860, 0, -6422282562],
                    [-6202159288, 6422282562, 0],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix__struct() {
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            m3(
                [
                    [0, -2324644860, 6202159288], [2324644860, 0, -6422282562],
                    [-6202159288, 6422282562, 0],
                ],
            ),
        );
        assert!(Matrix3Trait::cross_matrix(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_cross_matrix_mul__baseline() {
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let _a = black_box(
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
        assert!(e == e);
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
    fn bench_matrix3_cross_matrix_mul__materialised() {
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
        assert!(Matrix3Trait::cross_matrix(v) * a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_column__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-3562322882, 8037214559, -6638673079)));
        assert!(e == e);
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
    fn bench_matrix3_row__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-6195210852, 8037214559, 5406550886)));
        assert!(e == e);
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
    fn bench_matrix3_diagonal__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-8333418062, 8037214559, 4752733287)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_diagonal__struct() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(v3t((-8333418062, 8037214559, 4752733287)));
        assert!(a.diagonal() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_transpose__baseline() {
        let _a = black_box(
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
                    [-8333418062, -6195210852, 2873393302], [-3562322882, 8037214559, -6638673079],
                    [-7141719772, 5406550886, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_transpose__struct() {
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
                    [-8333418062, -6195210852, 2873393302], [-3562322882, 8037214559, -6638673079],
                    [-7141719772, 5406550886, 4752733287],
                ],
            ),
        );
        assert!(a.transpose() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_trace__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(4456529784));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_trace__sum() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(4456529784));
        assert!(a.trace() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs__baseline() {
        let _a = black_box(
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
                    [8333418062, 3562322882, 7141719772], [6195210852, 8037214559, 5406550886],
                    [2873393302, 6638673079, 4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs__componentwise() {
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
                    [8333418062, 3562322882, 7141719772], [6195210852, 8037214559, 5406550886],
                    [2873393302, 6638673079, 4752733287],
                ],
            ),
        );
        assert!(a.abs() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_add__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-14736756803, -11784916900, -10803461533],
                    [-1695724392, 11493453460, 12809574443], [-2339912399, -3710838010, 7822493936],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_add__operator() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-14736756803, -11784916900, -10803461533],
                    [-1695724392, 11493453460, 12809574443], [-2339912399, -3710838010, 7822493936],
                ],
            ),
        );
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_add__assign() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-14736756803, -11784916900, -10803461533],
                    [-1695724392, 11493453460, 12809574443], [-2339912399, -3710838010, 7822493936],
                ],
            ),
        );
        let mut r = a;
        r += b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_sub__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-1930079321, 4660271136, -3479978011], [-10694697312, 4580975658, -1996472671],
                    [8086699003, -9566508148, 1682972638],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_sub__operator() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-1930079321, 4660271136, -3479978011], [-10694697312, 4580975658, -1996472671],
                    [8086699003, -9566508148, 1682972638],
                ],
            ),
        );
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_sub__assign() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-1930079321, 4660271136, -3479978011], [-10694697312, 4580975658, -1996472671],
                    [8086699003, -9566508148, 1682972638],
                ],
            ),
        );
        let mut r = a;
        r -= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_neg__baseline() {
        let _a = black_box(
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
                    [8333418062, 3562322882, 7141719772], [6195210852, -8037214559, -5406550886],
                    [-2873393302, 6638673079, -4752733287],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_neg__operator() {
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
                    [8333418062, 3562322882, 7141719772], [6195210852, -8037214559, -5406550886],
                    [-2873393302, 6638673079, -4752733287],
                ],
            ),
        );
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_scale__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _k = black_box(fx(-7516192768));
        let e = black_box(
            m3(
                [
                    [14583481608, 6234065043, 12498009601],
                    [10841618991, -14065125479, -9461464051],
                    [-5028438279, 11617677888, -8317283253],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_scale__products() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let k = black_box(fx(-7516192768));
        let e = black_box(
            m3(
                [
                    [14583481608, 6234065043, 12498009601],
                    [10841618991, -14065125479, -9461464051],
                    [-5028438279, 11617677888, -8317283253],
                ],
            ),
        );
        assert!(a.scale(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_component_mul__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [12424238659, 6819966905, 6088785253], [-6490216439, 6467693861, 9319005434],
                    [-3487774563, -4525515217, 3396941726],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_component_mul__products() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [12424238659, 6819966905, 6088785253], [-6490216439, 6467693861, 9319005434],
                    [-3487774563, -4525515217, 3396941726],
                ],
            ),
        );
        assert!(a.component_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_vec__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-21470622535, 5268724875, -2717586978)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul_vec__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-21470622535, 5268724875, -2717586978)));
        assert!(a.mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul_vec__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-19851986100, 2686233155, -299288788)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul_vec__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-19851986100, 2686233155, -299288788)));
        assert!(a.tr_mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul_vec__transpose_mul_vec() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let v = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(v3t((-19851986100, 2686233155, -299288788)));
        assert!(a.transpose().mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
                    [-17007668927, -7603403072, -10495568584],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
                    [-17007668927, -7603403072, -10495568584],
                ],
            ),
        );
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_mul__assign() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
                    [-17007668927, -7603403072, -10495568584],
                ],
            ),
        );
        let mut r = a;
        r *= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2446247659, 12927457326, -1519880552], [21789113621, 8762145550, 12145577416],
                    [10542595260, 21263261548, 18804732413],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul__fused() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2446247659, 12927457326, -1519880552], [21789113621, 8762145550, 12145577416],
                    [10542595260, 21263261548, 18804732413],
                ],
            ),
        );
        assert!(a.tr_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_tr_mul__transpose_mul() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
                    [-5213305701, 2927835069, 3069760649],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [2446247659, 12927457326, -1519880552], [21789113621, 8762145550, 12145577416],
                    [10542595260, 21263261548, 18804732413],
                ],
            ),
        );
        assert!(a.transpose() * b == e);
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
    fn bench_matrix3_norm_squared__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(79224177887));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm_squared__wide() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(79224177887));
        assert!(a.norm_squared() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(18446280196));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_norm__wide() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(18446280196));
        assert!(a.norm() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_determinant__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(-49139065034));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_determinant__cofactors() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(-49139065034));
        assert!(a.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_determinant__alt_triple_products() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(fx(-49139065036));
        assert!(determinant_triple_products(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_adjugate__baseline() {
        let _a = black_box(
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
        assert!(e == e);
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
    fn bench_matrix3_try_inverse__baseline() {
        let _a = black_box(
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
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__prescaled_det_ge_half() {
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
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__prescaled_norm_gt_one() {
        let a = black_box(
            m3(
                [
                    [-549311473, 47395325, -2269605289], [1632143220, 2059389761, -2812332219],
                    [-1861974849, -80861466, 2028588103],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [-6951842770, -153772323, -7990975570], [-3388673950, 9398089719, 9237754150],
                    [-6515945503, 233474262, 2126960560],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse__prescaled_small() {
        let a = black_box(
            m3(
                [
                    [719963565, -491093223, 202288240], [14792173, 6300332, -1708200869],
                    [-556953536, -690401814, -251454189],
                ],
            ),
        );
        let e = black_box(
            m3(
                [
                    [16495394595, 3675684940, -11699880121],
                    [-13341091189, 955040073, -17220417569], [93636196, -10763579368, -164828975],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
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

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse_singular__baseline() {
        let _a = black_box(
            m3(
                [
                    [38654705664, 0, -4294967296], [-21474836480, -21474836480, 17179869184],
                    [-17179869184, 21474836480, -12884901888],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_try_inverse_singular__none() {
        let a = black_box(
            m3(
                [
                    [38654705664, 0, -4294967296], [-21474836480, -21474836480, 17179869184],
                    [-17179869184, 21474836480, -12884901888],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.try_inverse().is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_is_identity__baseline() {
        let _a = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_is_identity__all_compared() {
        let a = black_box(m3([[4294967296, 0, 0], [0, 4294967296, 0], [0, 0, 4294967296]]));
        let e = black_box(true);
        assert!(a.is_identity(2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs_diff_eq__baseline() {
        let _a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix3_abs_diff_eq__all_compared() {
        let a = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let b = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.abs_diff_eq(b, 2) == e);
    }
}
