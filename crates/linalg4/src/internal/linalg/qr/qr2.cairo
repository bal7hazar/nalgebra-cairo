//! Internal, no stability promise: the crate-private items of `linalg::qr::qr2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;
use crate::linalg::qr::qr2::Qr2;

/// Crate-internal kernels of `Qr2<T>` (WP 8.0: the public API is strictly upstream's).
/// `determinant`
/// has no upstream counterpart (upstream's `QR::determinant` is commented out); the tests use it to
/// check the factors.
#[generate_trait]
pub impl Qr2InternalImpl<
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
> of Qr2InternalTrait<T> {
    /// The determinant: `det(Q) * r11 * r22`. Upstream: `QR::determinant`.
    ///
    /// `R` has a non-negative diagonal, so the sign lives entirely in `det(Q) = ±1`, read off
    /// `Matrix2::determinant(q)` — a single `diff_prod` on an orthonormal matrix, hence `±1`
    /// within the rounding of the factorisation, and its SIGN is what is used. The sign is applied
    /// to the FIRST factor (an exact negation) so that the product stays a floor chain, exactly as
    /// `Lu2::determinant` does.
    ///
    /// Exactly zero for a rank-deficient matrix. Panics with the scalar's overflow error if the
    /// product does not fit.
    ///
    /// PREFER `Matrix2::determinant` when the factorisation is not needed for something else: the
    /// closed form is one exactly-rounded `diff_prod`, where this one multiplies two norms that
    /// already carry the rounding of the orthogonalisation.
    fn determinant(self: Qr2<T>) -> T {
        let d = if self.q.determinant().is_sign_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22
    }
}
