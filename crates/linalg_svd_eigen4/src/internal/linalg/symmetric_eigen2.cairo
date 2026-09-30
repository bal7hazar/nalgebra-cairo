//! Internal, no stability promise: the crate-private items of `linalg::symmetric_eigen2` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_static2::base::matrix2::Matrix2Trait;
use nalgebra_static2::base::vector2::Vector2Trait;
use nalgebra_static2::internal::base::sym_matrix2::SymMatrix2Trait;
use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::internal::base::sym_matrix2::SymMatrix2;
use simba::scalar::Real;
use crate::linalg::symmetric_eigen2::SymmetricEigen2;

/// Crate-internal kernels of `SymmetricEigen2<T>` (WP 8.0: the public API is strictly
/// upstream's): the decomposition and the eigenvalues of a `SymMatrix2` (the 3 independent
/// components the SVD's Gram matrix is built as), the symmetric reconstruction and the sign
/// convention of the eigenvectors.
#[generate_trait]
pub impl SymmetricEigen2InternalImpl<
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
> of SymmetricEigen2InternalTrait<T> {
    /// The kernel of `SymmetricEigen2Trait::new` on the 3 independent components of `s`, the
    /// form the SVD builds its Gram matrix in. Documented (cost, accuracy) on `new`.
    fn new_sym(s: SymMatrix2<T>) -> SymmetricEigen2<T> {
        let mean = R::sum_prod2(s.m11, R::HALF, s.m22, R::HALF);
        let d = R::diff_prod(s.m11, R::HALF, s.m22, R::HALF);
        let r = R::norm2(d, s.m12);
        let eigenvalues = Vector2 { x: mean - r, y: mean + r };
        if s.m12 == R::zero() && d == R::zero() {
            // Isotropic: every direction is an eigenvector, and both rows of `S - λ₁ I` vanish.
            return SymmetricEigen2 { eigenvalues, eigenvectors: Matrix2Trait::identity() };
        }
        // Rows of `S - λ₁ I`: `(d + r, m12)` and `(m12, r - d)`. Both `d + r` and `r - d` are
        // non-negative; `d > 0` makes the first one the larger, `d <= 0` the second one.
        let u = if d > R::zero() {
            Vector2 { x: -s.m12, y: d + r }
        } else {
            Vector2 { x: d - r, y: s.m12 }
        };
        let c1 = SymmetricEigen2InternalTrait::canonical_sign(u.normalize());
        SymmetricEigen2 {
            eigenvalues, eigenvectors: Matrix2 { m11: c1.x, m21: c1.y, m12: -c1.y, m22: c1.x },
        }
    }
    /// The eigenvalues of `s` alone, ascending, without the eigenvectors (half the cost: no
    /// normalisation). The kernel of
    /// `Matrix2SymmetricEigenTrait::symmetric_eigenvalues`.
    #[inline(always)]
    fn eigenvalues(s: SymMatrix2<T>) -> Vector2<T> {
        let mean = R::sum_prod2(s.m11, R::HALF, s.m22, R::HALF);
        let r = R::norm2(R::diff_prod(s.m11, R::HALF, s.m22, R::HALF), s.m12);
        Vector2 { x: mean - r, y: mean + r }
    }
    /// `v` with the sign that makes its component of largest absolute value positive (ties go to
    /// `x`); `v` unchanged when it is zero. This is the sign convention of `eigenvectors`,
    /// documented on the struct. No upstream equivalent.
    #[inline(always)]
    fn canonical_sign(v: Vector2<T>) -> Vector2<T> {
        let dominant = if v.x.abs() >= v.y.abs() {
            v.x
        } else {
            v.y
        };
        if dominant.is_sign_negative() {
            Vector2 { x: -v.x, y: -v.y }
        } else {
            v
        }
    }
    /// `V * diag(eigenvalues) * Vᵀ`, the symmetric matrix the decomposition came from, up to the
    /// rounding of the decomposition (measured: **4 ulp per unit of `max |m_ij|`** on the generic
    /// oracle vectors, 15 on the near-isotropic SPD ones, see `new`). Goes through
    /// `SymMatrix2::quadform`, so only the 3 independent components are computed. Panics on
    /// overflow. The kernel of `recompose`, which mirrors it.
    #[inline(always)]
    fn recompose_sym(self: SymmetricEigen2<T>) -> SymMatrix2<T> {
        SymMatrix2Trait::quadform(self.eigenvectors, self.eigenvalues)
    }
}
