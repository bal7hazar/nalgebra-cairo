//! Internal, no stability promise: the crate-private items of `base::matrix2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::internal::base::sym_matrix2::SymMatrix2;
use simba::scalar::Real;

/// Crate-internal kernels of `Matrix2<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `adjugate`, `mul_transpose` into a `SymMatrix2`) and the unrolled column accessors that stand
/// for upstream `column(i)` views, used by the decompositions.
#[generate_trait]
pub impl Matrix2InternalImpl<
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
> of Matrix2InternalTrait<T> {
    /// The outer product `a * bᵀ`: each component is one floored product. Panics with the
    /// scalar's overflow error. Upstream: `a * b.transpose()`.
    #[inline(always)]
    fn from_outer(a: Vector2<T>, b: Vector2<T>) -> Matrix2<T> {
        Matrix2 { m11: a.x * b.x, m21: a.y * b.x, m12: a.x * b.y, m22: a.y * b.y }
    }

    /// First column. Upstream: `column(0)`.
    #[inline(always)]
    fn column1(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m11, y: self.m21 }
    }

    /// Second column. Upstream: `column(1)`.
    #[inline(always)]
    fn column2(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m12, y: self.m22 }
    }

    /// `self * selfᵀ` as a symmetric matrix: 3 `sum_prod2` instead of 4, bit-identical to the
    /// upper triangle of `self * self.transpose()`. Panics on overflow.
    /// Upstream: `self * self.transpose()`.
    fn mul_transpose(self: Matrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 {
            m11: R::norm_squared2(self.m11, self.m12),
            m12: R::sum_prod2(self.m11, self.m21, self.m12, self.m22),
            m22: R::norm_squared2(self.m21, self.m22),
        }
    }

    /// The adjugate (transposed cofactor matrix) `[[m22, -m12], [-m21, m11]]`:
    /// `self * adjugate = determinant * I`. Exact; panics on the scalar's `MIN`.
    /// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
    #[inline(always)]
    fn adjugate(self: Matrix2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.m22, m21: -self.m21, m12: -self.m12, m22: self.m11 }
    }
}
