//! Internal, no stability promise: the crate-private items of `geometry::translation5` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types5::geometry::translation5::Translation5;
use simba::scalar::Real;
use crate::base::matrix6::Matrix6;

/// The homogeneous matrix of a `Translation5`, run by `Translation5Trait::to_homogeneous` and by
/// the conversion `Matrix6FromTranslation5` through `#[inline(always)]`: a type-level kernel with
/// the types of `Matrix6`, where the conversion sits (docs/SPLIT.md §18.3). Internal.
#[generate_trait]
pub impl Matrix6FromTranslation5KernelImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Matrix6FromTranslation5KernelTrait<T> {
    #[inline(always)]
    fn to_homogeneous(self: Translation5<T>) -> Matrix6<T> {
        Matrix6 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m51: R::zero(),
            m61: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m62: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: R::zero(),
            m53: R::zero(),
            m63: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
            m54: R::zero(),
            m64: R::zero(),
            m15: R::zero(),
            m25: R::zero(),
            m35: R::zero(),
            m45: R::zero(),
            m55: R::one(),
            m65: R::zero(),
            m16: self.vector.x,
            m26: self.vector.y,
            m36: self.vector.z,
            m46: self.vector.w,
            m56: self.vector.a,
            m66: R::one(),
        }
    }
}
