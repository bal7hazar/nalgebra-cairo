//! Internal, no stability promise: the crate-private items of `geometry::rotation3` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::geometry::rotation3::Rotation3;
use simba::scalar::Real;
use crate::base::matrix4::Matrix4;

/// The homogeneous matrix of a `Rotation3`, run by `Rotation3Trait::to_homogeneous` and by
/// the conversion `Matrix4FromRotation3` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix4`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix4FromRotation3KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix4FromRotation3KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Rotation3<T>) -> Matrix4<T> {
        let m = self.matrix;
        Matrix4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: R::zero(),
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: R::zero(),
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
        }
    }
}
