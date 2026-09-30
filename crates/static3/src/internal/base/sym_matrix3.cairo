//! In `nalgebra_static3`: the impl `SymMatrix3Impl`. This module is split over packages; the other
//! parts are in `nalgebra_types3`.
//!
//! Internal, no stability promise: the crate-private items of `base::sym_matrix3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::Real;

/// Methods of `SymMatrix3<T>` for any `Real` scalar.
#[generate_trait]
pub impl SymMatrix3Impl<
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
> of SymMatrix3Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// `r * diag(d) * rᵀ` (world-space inertia from the principal inertia `d` and the rotation
    /// `r`) without materialising the diagonal matrix: 9 products `r_ik * d_k` then 6
    /// `sum_prod3` (two roundings per component, like `(r * diag(d)) * rᵀ`, to which it is
    /// bit-identical). Panics on overflow. Upstream: `quadform_tr` with a diagonal `mid`.
    fn quadform(r: Matrix3<T>, d: Vector3<T>) -> SymMatrix3<T> {
        let (t11, t12, t13) = (r.m11 * d.x, r.m12 * d.y, r.m13 * d.z);
        let (t21, t22, t23) = (r.m21 * d.x, r.m22 * d.y, r.m23 * d.z);
        let (t31, t32, t33) = (r.m31 * d.x, r.m32 * d.y, r.m33 * d.z);
        SymMatrix3 {
            m11: R::sum_prod3(t11, r.m11, t12, r.m12, t13, r.m13),
            m12: R::sum_prod3(t11, r.m21, t12, r.m22, t13, r.m23),
            m13: R::sum_prod3(t11, r.m31, t12, r.m32, t13, r.m33),
            m22: R::sum_prod3(t21, r.m21, t22, r.m22, t23, r.m23),
            m23: R::sum_prod3(t21, r.m31, t22, r.m32, t23, r.m33),
            m33: R::sum_prod3(t31, r.m31, t32, r.m32, t33, r.m33),
        }
    }

    // --- accessors and conversions -------------------------------------------------------------

    /// The full matrix. parry: `SdpMatrix3::into_matrix`.
    #[inline(always)]
    fn to_matrix(self: SymMatrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.m11,
            m21: self.m12,
            m31: self.m13,
            m12: self.m12,
            m22: self.m22,
            m32: self.m23,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
        }
    }
    // --- products ------------------------------------------------------------------------------

    // --- norms ---------------------------------------------------------------------------------

    // --- determinant and inverse ---------------------------------------------------------------

    // --- approximate equality ------------------------------------------------------------------
}
