//! `SymMatrix3`: a symmetric 3x3 matrix stored as its 6 independent components.
//!
//! No upstream nalgebra equivalent; the model is rapier / parry's `SdpMatrix3` (inertia tensors,
//! effective masses). Structured kernels (`quadform`, `quadform_sym`, `from_outer_self`,
//! `Matrix3::mul_transpose`, `try_inverse`) compute only the 6 independent components, never
//! materialise a diagonal matrix, and are bit-identical to the upper triangle of the generic
//! `Matrix3` expression they replace.

pub use nalgebra_core::internal::base::sym_matrix3::*;

#[cfg(test)]
mod tests {
    use crate::base::matrix3::Matrix3Trait;
    use crate::base::matrix_test_utils::{int, m3, m3i, s3, s3i, sym3_upper, v3i, v3t};
    use crate::base::oracle_matrix3;
    use crate::testing::black_box;
    use super::SymMatrix3Trait;

    /// The oracle's `matrix3` cases without the `large` distribution (indices [12..16)), whose
    /// squares do not fit a quadratic form.
    fn quadratic_cases() -> Span<([[i64; 3]; 3], [[i64; 3]; 3], [[i64; 3]; 3], u64)> {
        oracle_matrix3::matrix3_mul_cases().slice(0, 12)
    }

    // --- constructors and accessors ----------------------------------------------------------

    #[test]
    fn test_to_matrix_is_a_round_trip() {
        let mut cases = oracle_matrix3::matrix3_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let s = sym3_upper(m3(a));
            assert!(sym3_upper(s.to_matrix()) == s);
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
            let (r, d) = (m3(a), m3(b).diagonal());
            let expected = (r * Matrix3Trait::from_diagonal(d)) * r.transpose();
            // Only the UPPER triangle: the generic product is not exactly symmetric, its `m21`
            // rounds `r21 * d.x` before multiplying by `r11` where `m12` rounds `r11 * d.x`.
            assert!(SymMatrix3Trait::quadform(r, d) == sym3_upper(expected));
        }
        // diag(2, 3, 4) in the frame that swaps x and y.
        let swap = m3i([[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
        assert!(SymMatrix3Trait::quadform(swap, v3i(2, 3, 4)) == s3i((3, 0, 0, 2, 0, 4)));
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_quadform_overflow_panics() {
        let r = black_box(Matrix3Trait::from_diagonal_element(int(65536)));
        let _ = SymMatrix3Trait::quadform(r, black_box(v3i(1, 1, 1)));
    }

    // --- determinant and inverse ---------------------------------------------------------------

    // --- approximate equality ------------------------------------------------------------------
    // --- gas benchmarks
    // ----------------------------------------------------------------------------
    //
    // Inputs are shared with the `matrix3` benchmarks wherever the same operation exists there, so
    // that the structured kernels and the generic `Matrix3` path can be read side by side. `a` is
    // the `unit` SPD case of the oracle's `udu3` vectors (an inertia tensor).
    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_to_matrix__baseline() {
        let _a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            m3(
                [
                    [1841663783, 52948593, 34555536], [52948593, 1861238670, 30362422],
                    [34555536, 30362422, 1130131990],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_to_matrix__struct() {
        let a = black_box(s3((1841663783, 52948593, 34555536, 1861238670, 30362422, 1130131990)));
        let e = black_box(
            m3(
                [
                    [1841663783, 52948593, 34555536], [52948593, 1861238670, 30362422],
                    [34555536, 30362422, 1130131990],
                ],
            ),
        );
        assert!(a.to_matrix() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform__baseline() {
        let _r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let _d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((34871941038, 3481951390, -4662719457, 38764687225, -20898872038, 20538935376)),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform__structured() {
        let r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((34871941038, 3481951390, -4662719457, 38764687225, -20898872038, 20538935376)),
        );
        assert!(SymMatrix3Trait::quadform(r, d) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_sym_matrix3_quadform__generic() {
        let r = black_box(
            m3(
                [
                    [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
                    [2873393302, -6638673079, 4752733287],
                ],
            ),
        );
        let d = black_box(v3t((6422282562, 6202159288, 2324644860)));
        let e = black_box(
            s3((34871941038, 3481951390, -4662719457, 38764687225, -20898872038, 20538935376)),
        );
        let m = (r * Matrix3Trait::from_diagonal(d)) * r.transpose();
        assert!(sym3_upper(m) == e);
    }
}
