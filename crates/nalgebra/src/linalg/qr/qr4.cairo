//! `Qr4`: the QR factorisation of a `Matrix4` (upstream `nalgebra::linalg::QR` on a 4x4 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`. The
//! algorithm is the modified Gram-Schmidt of `qr3`, one column longer; the study of the
//! alternatives (Householder, classical Gram-Schmidt, completed basis) lives there.

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg4::internal::linalg::qr::qr4::Qr4InternalTrait;
pub use nalgebra_linalg4::linalg::qr::qr4::*;

/// Test-only field-wise equality (upstream `Qr4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Qr4PartialEq<T, +PartialEq<T>> of PartialEq<Qr4<T>> {
    fn eq(lhs: @Qr4<T>, rhs: @Qr4<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Qr4`: exact cases, the identities and the oracle vectors of `tools/oracle`.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/qr/qr4/tests.cairo`.

    use fixed::Fixed;
    use nalgebra_static4::internal::base::matrix4::Matrix4InternalTrait;
    use simba::scalar::Real;
    use crate::base::matrix4::{Matrix4, Matrix4Trait};
    use crate::base::matrix_test_utils::{
        amax_m4, int, m4, max_ulp_diff4, orthonormality_error_m4, v4t,
    };
    use crate::base::vector4::{Vector4, Vector4Trait};
    use crate::linalg::qr::oracle_qr4 as oracle;
    use crate::testing::black_box;
    use super::{Qr4, Qr4InternalTrait, Qr4Trait};

    /// An oracle `unit` 4x4 case: the benchmark input.
    fn a_bench() -> Matrix4<Fixed> {
        m4(
            [
                [-151519325, -831704190, -37020683, -258362626],
                [-309967323, -174899491, -990814222, 1188862699],
                [526702394, 89514843, 1516307715, 154933714],
                [1138644662, -23815325, 661466655, 61972869],
            ],
        )
    }

    /// Its right-hand side, from `qr4_solve`.
    fn b_bench() -> Vector4<Fixed> {
        v4t((2614746463, 2804691241, -3498193820, 3365132750))
    }

    /// `a_bench()` already factored.
    fn f_bench() -> Qr4<Fixed> {
        Qr4Trait::new(a_bench())
    }

    /// A rank-3 matrix: the fourth column is the second, so `r44` is exactly zero.
    fn a_rank3() -> Matrix4<Fixed> {
        Matrix4Trait::from_columns(
            Vector4 { x: int(2), y: int(0), z: int(0), w: int(0) },
            Vector4 { x: int(0), y: int(4), z: int(0), w: int(0) },
            Vector4 { x: int(0), y: int(0), z: int(8), w: int(0) },
            Vector4 { x: int(0), y: int(4), z: int(0), w: int(0) },
        )
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_and_diagonal_are_exact() {
        let f = Qr4Trait::new(Matrix4Trait::<Fixed>::identity());
        assert!(f.q() == Matrix4Trait::identity());
        assert!(f.r() == Matrix4Trait::identity());
        assert!(f.determinant() == int(1));
        let d = Matrix4Trait::from_diagonal(
            Vector4 { x: int(2), y: int(-4), z: int(8), w: int(-1) },
        );
        let f = Qr4Trait::new(d);
        assert!(
            f
                .r() == Matrix4Trait::from_diagonal(
                    Vector4 { x: int(2), y: int(4), z: int(8), w: int(1) },
                ),
        );
        assert!(f.q() * f.r() == d);
        assert!(f.determinant() == int(64));
        assert!(orthonormality_error_m4(f.q()) == 0);
    }

    #[test]
    fn test_new_rank_deficient_leaves_a_zero_column() {
        let f = Qr4Trait::new(a_rank3());
        assert!(f.r().m44 == Real::zero());
        assert!(f.q().column4() == Vector4Trait::zeros());
        assert!(f.q() * f.r() == a_rank3());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
    }

    #[test]
    fn test_new_reconstruction_and_orthonormality_oracle() {
        let mut cases = oracle::qr4_q_r_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Qr4Trait::new(m4(a));
            let rec = max_ulp_diff4(f.q() * f.r(), m4(a)) / amax_m4(m4(a));
            let orth = orthonormality_error_m4(f.q());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured: `|A - Q R| <= worst_rec ulp * max(1, max |a_ij|)`, `|QᵀQ - I| <= worst_orth`.
        assert!(worst_rec == 3 && worst_orth == 86, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_determinant__diagonal_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.determinant() != Real::zero()) == e);
    }
}
