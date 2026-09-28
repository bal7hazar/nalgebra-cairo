//! `SymMatrix2`: a symmetric 2x2 matrix stored as its 3 independent components.
//!
//! No upstream nalgebra equivalent; the model is rapier / parry's `SdpMatrix2` (2D effective
//! masses, 2x2 constraint blocks). Structured kernels (`quadform`, `quadform_sym`,
//! `from_outer_self`, `Matrix2::mul_transpose`) compute only the 3 independent components, and are
//! bit-identical to the upper triangle of the generic `Matrix2` expression they replace.
//!
//! Internal, no stability promise (docs/SPLIT.md §12.3): crate-private in `nalgebra` 0.1.0
//! (`nalgebra::base::sym_matrix2`), used by the decompositions above `nalgebra_core`.

use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::vector2::Vector2;

/// A symmetric 2x2 matrix `[[m11, m12], [m12, m22]]`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymMatrix2<T> {
    pub m11: T,
    pub m12: T,
    pub m22: T,
}

/// Methods of `SymMatrix2<T>` for any `Real` scalar.
#[generate_trait]
pub impl SymMatrix2Impl<
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
> of SymMatrix2Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// `r * diag(d) * rᵀ` without materialising the diagonal matrix: 4 products `r_ik * d_k`
    /// then 3 `sum_prod2` (two roundings per component, like `(r * diag(d)) * rᵀ`, to which it
    /// is bit-identical). Panics on overflow. Upstream: `quadform_tr` with a diagonal `mid`.
    fn quadform(r: Matrix2<T>, d: Vector2<T>) -> SymMatrix2<T> {
        let (t11, t12) = (r.m11 * d.x, r.m12 * d.y);
        let (t21, t22) = (r.m21 * d.x, r.m22 * d.y);
        SymMatrix2 {
            m11: R::sum_prod2(t11, r.m11, t12, r.m12),
            m12: R::sum_prod2(t11, r.m21, t12, r.m22),
            m22: R::sum_prod2(t21, r.m21, t22, r.m22),
        }
    }

    // --- accessors and conversions -------------------------------------------------------------

    /// The full matrix. parry: `SdpMatrix2::into_matrix`.
    #[inline(always)]
    fn to_matrix(self: SymMatrix2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.m11, m21: self.m12, m12: self.m12, m22: self.m22 }
    }
    // --- products ------------------------------------------------------------------------------

    // --- norms ---------------------------------------------------------------------------------

    // --- determinant and inverse ---------------------------------------------------------------

    // --- approximate equality ------------------------------------------------------------------
}
// --- operators -----------------------------------------------------------------------------------
