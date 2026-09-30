//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types4::base::matrix4::Matrix4;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen4::Sym4;
use crate::linalg::symmetric_eigen4::SymmetricEigen4InternalTrait;

/// The right singular vectors of a matrix with 4 columns (crate-internal).
#[generate_trait]
pub impl SvdRightImpl4<
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
> of SvdRightTrait4<T> {
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
}
