//! Internal, no stability promise: the crate-private items of `geometry::translation2` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types2::geometry::translation2::Translation2;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3;

/// The homogeneous matrix of a `Translation2`, run by `Translation2Trait::to_homogeneous` and by
/// the conversion `Matrix3FromTranslation2` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix3`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix3FromTranslation2KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix3FromTranslation2KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Translation2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m13: self.vector.x,
            m23: self.vector.y,
            m33: R::one(),
        }
    }
}
