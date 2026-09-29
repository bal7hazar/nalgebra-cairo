//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_shapes6::base::matrix6::Matrix6;
use simba::scalar::Real;
use crate::linalg::symmetric_eigen6::{Sym6, SymmetricEigen6InternalTrait};

/// The right singular vectors of a matrix with 6 columns (crate-internal).
#[generate_trait]
pub impl SvdRightImpl6<
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
> of SvdRightTrait6<T> {
    /// The eigenvectors of the 6x6 Gram matrix (`SymmetricEigen6`, already renormalised and
    /// signed), as the columns of a `Matrix6`, ascending eigenvalue order.
    #[inline(always)]
    fn right6(g: Sym6<T>) -> Matrix6<T> {
        SymmetricEigen6InternalTrait::new_sym(g).eigenvectors
    }
    /// `right6`, or `None` when the Jacobi sweeps did not converge within `eps`.
    #[inline(always)]
    fn try_right6(g: Sym6<T>, eps: T) -> Option<Matrix6<T>> {
        match SymmetricEigen6InternalTrait::try_new_sym(g, eps) {
            Some(e) => Some(e.eigenvectors),
            None => None,
        }
    }
}
