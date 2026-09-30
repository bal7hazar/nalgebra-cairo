//! Internal, no stability promise: the crate-private items of `geometry::rotation2` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types2::geometry::rotation2::Rotation2;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3;

/// The homogeneous matrix of a `Rotation2`, run by `Rotation2Trait::to_homogeneous` and by
/// the conversion `Matrix3FromRotation2` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix3`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix3FromRotation2KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix3FromRotation2KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Rotation2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.matrix.m11,
            m21: self.matrix.m21,
            m31: R::zero(),
            m12: self.matrix.m12,
            m22: self.matrix.m22,
            m32: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
        }
    }
}
