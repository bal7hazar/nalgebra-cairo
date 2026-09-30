//! Internal, no stability promise: the crate-private items of `geometry::translation3` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::geometry::translation3::Translation3;
use simba::scalar::Real;
use crate::base::matrix4::Matrix4;

/// The homogeneous matrix of a `Translation3`, run by `Translation3Trait::to_homogeneous` and by
/// the conversion `Matrix4FromTranslation3` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix4`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix4FromTranslation3KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix4FromTranslation3KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Translation3<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m42: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: R::zero(),
            m14: self.vector.x,
            m24: self.vector.y,
            m34: self.vector.z,
            m44: R::one(),
        }
    }
}
