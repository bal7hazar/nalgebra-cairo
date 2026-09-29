//! `SymMatrix2`: a symmetric 2x2 matrix stored as its 3 independent components.
//!
//! No upstream nalgebra equivalent; the model is rapier / parry's `SdpMatrix2` (2D effective
//! masses, 2x2 constraint blocks). Structured kernels (`quadform`, `quadform_sym`,
//! `from_outer_self`, `Matrix2::mul_transpose`) compute only the 3 independent components, and are
//! bit-identical to the upper triangle of the generic `Matrix2` expression they replace.

pub use nalgebra_static2::internal::base::sym_matrix2::*;
pub use nalgebra_types2::internal::base::sym_matrix2::*;

#[cfg(test)]
mod tests {
    use crate::base::matrix2::Matrix2Trait;
    use crate::base::matrix_test_utils::{int, m2, m2i, s2, s2i, sym2_upper, v2i, v2t};
    use crate::base::oracle_matrix2;
    use crate::testing::black_box;
    use super::SymMatrix2Trait;

    /// The oracle's `matrix2` cases without the `large` distribution (indices [12..16)), whose
    /// squares do not fit a quadratic form.
    fn quadratic_cases() -> Span<([[i64; 2]; 2], [[i64; 2]; 2], [[i64; 2]; 2], u64)> {
        oracle_matrix2::matrix2_mul_cases().slice(0, 12)
    }

    // --- constructors and accessors ----------------------------------------------------------

    #[test]
    fn test_to_matrix_is_a_round_trip() {
        let mut cases = oracle_matrix2::matrix2_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = sym2_upper(m2(a));
            assert!(sym2_upper(s.to_matrix()) == s);
            // A symmetric matrix is its own transpose.
            assert!(s.to_matrix().transpose() == s.to_matrix());
        }
    }

    // --- exact operations ----------------------------------------------------------------------

    // --- products ------------------------------------------------------------------------------
    #[test]
    fn test_quadform_matches_the_generic_product() {
        let mut cases = quadratic_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            let (r, d) = (m2(a), m2(b).diagonal());
            let expected = (r * Matrix2Trait::from_diagonal(d)) * r.transpose();
            // Only the UPPER triangle: the generic product is not exactly symmetric, its `m21`
            // rounds `r21 * d.x` before multiplying by `r11` where `m12` rounds `r11 * d.x`.
            assert!(SymMatrix2Trait::quadform(r, d) == sym2_upper(expected));
        }
        // r * diag(2, 3) * r^T with r = [[1, 0], [1, 1]] is [[2, 2], [2, 5]].
        assert!(SymMatrix2Trait::quadform(m2i([[1, 0], [1, 1]]), v2i(2, 3)) == s2i((2, 2, 5)));
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_quadform_overflow_panics() {
        let r = black_box(Matrix2Trait::from_diagonal_element(int(65536)));
        let _ = SymMatrix2Trait::quadform(r, black_box(v2i(1, 1)));
    }

    // --- determinant and inverse ---------------------------------------------------------------

    // --- approximate equality ------------------------------------------------------------------
    // --- gas benchmarks
    // ----------------------------------------------------------------------------
    //
    // Inputs are shared with the `matrix2` benchmarks wherever the same operation exists there, so
    // that the structured kernels and the generic `Matrix2` path can be read side by side. `a` is
    // the `unit` SPD case of the oracle's `udu2` vectors.
    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_to_matrix__baseline() {
        let _a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(m2([[1400100946, 252374302], [252374302, 1577459800]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_to_matrix__struct() {
        let a = black_box(s2((1400100946, 252374302, 1577459800)));
        let e = black_box(m2([[1400100946, 252374302], [252374302, 1577459800]]));
        assert!(a.to_matrix() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform__baseline() {
        let _r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _d = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(s2((-10663446698, 21499658486, -13302844783)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform__structured() {
        let r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let d = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(s2((-10663446698, 21499658486, -13302844783)));
        assert!(SymMatrix2Trait::quadform(r, d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix2_quadform__generic() {
        let r = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let d = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(s2((-10663446698, 21499658486, -13302844783)));
        let m = (r * Matrix2Trait::from_diagonal(d)) * r.transpose();
        assert!(sym2_upper(m) == e);
    }
}
