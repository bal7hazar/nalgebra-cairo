//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen3::SymmetricEigen3InternalTrait;

/// The right singular vectors of a matrix with 3 columns (crate-internal).
#[generate_trait]
pub impl SvdRightImpl3<
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
> of SvdRightTrait3<T> {
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
}
