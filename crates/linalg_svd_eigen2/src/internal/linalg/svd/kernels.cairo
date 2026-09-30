//! Internal, no stability promise: the crate-private items of `linalg::svd::kernels` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::internal::base::sym_matrix2::SymMatrix2;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;

/// The right singular vectors of a matrix with 2 columns (crate-internal).
#[generate_trait]
pub impl SvdRightImpl2<
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
> of SvdRightTrait2<T> {
    /// The eigenvectors of the 2x2 Gram matrix, as the columns of a `Matrix2` (ascending
    /// eigenvalue order), renormalised exactly like `Svd2::new` does: the first column divided by
    /// its own norm, the second its direct perpendicular.
    fn right2(g: SymMatrix2<T>) -> Matrix2<T> {
        let e = SymmetricEigen2InternalTrait::new_sym(g).eigenvectors;
        let nv = R::norm2(e.m11, e.m21);
        let (x, y) = (R::div(e.m11, nv), R::div(e.m21, nv));
        Matrix2 { m11: x, m21: y, m12: -y, m22: x }
    }
}
