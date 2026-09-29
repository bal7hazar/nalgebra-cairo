//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_shapes5::base::matrix5::Matrix5;
use simba::scalar::Real;
use crate::linalg::symmetric_eigen5::{Sym5, SymmetricEigen5InternalTrait};

/// The right singular vectors of a matrix with 5 columns (crate-internal).
#[generate_trait]
pub impl SvdRightImpl5<
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
> of SvdRightTrait5<T> {
    /// The eigenvectors of the 5x5 Gram matrix (`SymmetricEigen5`, already renormalised and
    /// signed), as the columns of a `Matrix5`, ascending eigenvalue order.
    #[inline(always)]
    fn right5(g: Sym5<T>) -> Matrix5<T> {
        SymmetricEigen5InternalTrait::new_sym(g).eigenvectors
    }
    /// `right5`, or `None` when the Jacobi sweeps did not converge within `eps`.
    #[inline(always)]
    fn try_right5(g: Sym5<T>, eps: T) -> Option<Matrix5<T>> {
        match SymmetricEigen5InternalTrait::try_new_sym(g, eps) {
            Some(e) => Some(e.eigenvectors),
            None => None,
        }
    }
}
