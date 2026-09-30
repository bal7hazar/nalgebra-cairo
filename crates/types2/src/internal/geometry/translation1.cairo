//! Internal, no stability promise: the crate-private items of `geometry::translation1` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::geometry::translation1::Translation1;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2;

/// The homogeneous matrix of a `Translation1`, run by `Translation1Trait::to_homogeneous` and by
/// the conversion `Matrix2FromTranslation1` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix2`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix2FromTranslation1KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix2FromTranslation1KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Translation1<T>) -> Matrix2<T> {
        Matrix2 { m11: R::one(), m21: R::zero(), m12: self.vector.x, m22: R::one() }
    }
}
