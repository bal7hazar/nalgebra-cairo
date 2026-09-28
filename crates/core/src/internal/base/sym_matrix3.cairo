//! `SymMatrix3`: a symmetric 3x3 matrix stored as its 6 independent components.
//!
//! No upstream nalgebra equivalent; the model is rapier / parry's `SdpMatrix3` (inertia tensors,
//! effective masses). Structured kernels (`quadform`, `quadform_sym`, `from_outer_self`,
//! `Matrix3::mul_transpose`, `try_inverse`) compute only the 6 independent components, never
//! materialise a diagonal matrix, and are bit-identical to the upper triangle of the generic
//! `Matrix3` expression they replace.
//!
//! Internal, no stability promise (docs/SPLIT.md §12.3): crate-private in `nalgebra` 0.1.0
//! (`nalgebra::base::sym_matrix3`), used by the decompositions above `nalgebra_core`.

use simba::scalar::Real;
use crate::base::matrix3::Matrix3;
use crate::base::vector3::Vector3;

/// A symmetric 3x3 matrix `[[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]]`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymMatrix3<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m22: T,
    pub m23: T,
    pub m33: T,
}

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
// --- operators -----------------------------------------------------------------------------------
