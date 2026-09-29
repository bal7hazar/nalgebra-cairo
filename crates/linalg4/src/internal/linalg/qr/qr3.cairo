//! Internal, no stability promise: the crate-private items of `linalg::qr::qr3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
use simba::scalar::Real;
use crate::linalg::qr::qr3::Qr3;

/// Crate-internal kernels of `Qr3<T>` (WP 8.0: the public API is strictly upstream's). The back
/// substitution shared by `solve` and `try_inverse`, and `determinant` (upstream's
/// `QR::determinant` is commented out), which the tests use to check the factors.
#[generate_trait]
pub impl Qr3InternalImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Qr3InternalTrait<T> {
    /// `R^-1 * y` by back substitution, the shared body of `solve` and `try_inverse`. The caller
    /// guarantees a nonzero diagonal. No upstream equivalent (upstream substitutes in place).
    #[inline(always)]
    fn back_substitute(self: Qr3<T>, y: Vector3<T>) -> Vector3<T> {
        let x3 = R::div(y.z, self.r.m33);
        let x2 = R::div(R::mul_add(-self.r.m23, x3, y.y), self.r.m22);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.x), self.r.m12, x2);
        let x1 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m13, x3)), self.r.m11);
        Vector3 { x: x1, y: x2, z: x3 }
    }
    /// The determinant: `det(Q) * r11 * r22 * r33`. Upstream: `QR::determinant`.
    ///
    /// `R` has a non-negative diagonal, so the sign lives entirely in `det(Q) = ±1`, read off
    /// `Matrix3::determinant(q)` — `±1` within the rounding of the factorisation, and its SIGN
    /// is what is used. The sign is applied to the FIRST factor (an exact negation) so that the
    /// product stays a floor chain, exactly as `Lu2::determinant` does; `Real` exposes no
    /// `Wide * T`, so the chain of 3 floored multiplications is the best available.
    ///
    /// Exactly zero for a rank-deficient matrix. Panics with the scalar's overflow error if an
    /// intermediate product does not fit.
    ///
    /// PREFER `Matrix3::determinant` when the factorisation is not needed for something else: the
    /// closed form sums exact minors, where this one multiplies three norms and a sign read off
    /// `det(Q)`, all of which already carry the rounding of the orthogonalisation (measured on
    /// the oracle: the two differ by up to 90 847 ulp on a `medium` matrix, whose determinant is
    /// itself of the order of 10^5, `test_determinant_versus_matrix3_cofactors`).
    fn determinant(self: Qr3<T>) -> T {
        let d = if self.q.determinant().is_sign_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22 * self.r.m33
    }
}
