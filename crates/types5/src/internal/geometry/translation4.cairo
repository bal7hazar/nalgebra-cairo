//! Internal, no stability promise: the crate-private items of `geometry::translation4` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types4::geometry::translation4::Translation4;
use simba::scalar::Real;
use crate::base::matrix5::Matrix5;

/// The homogeneous matrix of a `Translation4`, run by `Translation4Trait::to_homogeneous` and by
/// the conversion `Matrix5FromTranslation4` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix5`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix5FromTranslation4KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix5FromTranslation4KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Translation4<T>) -> Matrix5<T> {
        Matrix5 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m51: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: R::zero(),
            m53: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
            m54: R::zero(),
            m15: self.vector.x,
            m25: self.vector.y,
            m35: self.vector.z,
            m45: self.vector.w,
            m55: R::one(),
        }
    }
}
