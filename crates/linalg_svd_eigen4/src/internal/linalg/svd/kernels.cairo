//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::internal::base::sym_matrix2::SymMatrix2;
use nalgebra_core::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen4::Sym4;
use crate::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;
use crate::linalg::symmetric_eigen3::SymmetricEigen3InternalTrait;
use crate::linalg::symmetric_eigen4::SymmetricEigen4InternalTrait;

/// The right singular vectors of a matrix with 2..6 columns (crate-internal).
#[generate_trait]
pub impl SvdRightImpl<
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
> of SvdRightTrait<T> {
    /// The eigenvectors of the 2x2 Gram matrix, as the columns of a `Matrix2` (ascending
    /// eigenvalue order), renormalised exactly like `Svd2::new` does: the first column divided by
    /// its own norm, the second its direct perpendicular.
    fn right2(g: SymMatrix2<T>) -> Matrix2<T> {
        let e = SymmetricEigen2InternalTrait::new_sym(g).eigenvectors;
        let nv = R::norm2(e.m11, e.m21);
        let (x, y) = (R::div(e.m11, nv), R::div(e.m21, nv));
        Matrix2 { m11: x, m21: y, m12: -y, m22: x }
    }
    /// `right3` with `Svd3::new`'s renormalisation (columns 1 and 2 divided by their norms, the
    /// third their cross product).
    fn right3(g: SymMatrix3<T>) -> Matrix3<T> {
        Self::renormalise3(SymmetricEigen3InternalTrait::new_sym(g).eigenvectors)
    }
    /// `right3`, or `None` when the four sweeps of `SymmetricEigen3` did not converge within
    /// `eps` (see `SymmetricEigen3Trait::try_new`).
    fn try_right3(g: SymMatrix3<T>, eps: T) -> Option<Matrix3<T>> {
        match SymmetricEigen3InternalTrait::try_new_sym(g, eps) {
            Some(e) => Some(Self::renormalise3(e.eigenvectors)),
            None => None,
        }
    }
    #[inline(always)]
    fn renormalise3(e: Matrix3<T>) -> Matrix3<T> {
        let n1 = R::norm3(e.m11, e.m21, e.m31);
        let n2 = R::norm3(e.m12, e.m22, e.m32);
        let (a1, a2, a3) = R::div3(e.m11, e.m21, e.m31, n1);
        let (b1, b2, b3) = R::div3(e.m12, e.m22, e.m32, n2);
        Matrix3 {
            m11: a1,
            m21: a2,
            m31: a3,
            m12: b1,
            m22: b2,
            m32: b3,
            m13: R::diff_prod(a2, b3, a3, b2),
            m23: R::diff_prod(a3, b1, a1, b3),
            m33: R::diff_prod(a1, b2, a2, b1),
        }
    }
    /// The eigenvectors of the 4x4 Gram matrix (`SymmetricEigen4`, already renormalised and
    /// signed), as the columns of a `Matrix4`, ascending eigenvalue order.
    #[inline(always)]
    fn right4(g: Sym4<T>) -> Matrix4<T> {
        SymmetricEigen4InternalTrait::new_sym(g).eigenvectors
    }
    /// `right4`, or `None` when the Jacobi sweeps did not converge within `eps`.
    #[inline(always)]
    fn try_right4(g: Sym4<T>, eps: T) -> Option<Matrix4<T>> {
        match SymmetricEigen4InternalTrait::try_new_sym(g, eps) {
            Some(e) => Some(e.eigenvectors),
            None => None,
        }
    }
    /// `1 / s` when `s > eps`, `0` otherwise: upstream's `pseudo_inverse` filter.
    #[inline(always)]
    fn inverted(s: T, eps: T) -> T {
        if s > eps {
            s.recip()
        } else {
            R::zero()
        }
    }
    /// `y / s` when `s > eps`, `0` otherwise: upstream's `solve` filter.
    #[inline(always)]
    fn divided(y: T, s: T, eps: T) -> T {
        if s > eps {
            R::div(y, s)
        } else {
            R::zero()
        }
    }
}
