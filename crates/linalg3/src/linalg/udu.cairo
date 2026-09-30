//! `UDUᵀ` factorisation `p = u·diag(d)·uᵀ` of a symmetric matrix — `u` unit UPPER
//! triangular, `d` a vector — unrolled for the static sizes 2, 3, 4 and 6 (upstream
//! `nalgebra::linalg::UDU`).
//!
//! The kernel is the crate-internal `LDLᵀ` factorisation of `crate::linalg::ldlt`, applied to the
//! reversed matrix `J·p·J` (`J` the reversal permutation): the two factorisations are the same
//! algorithm read backwards, so `UDU::new(p)` costs the `LDLᵀ` kernel plus moves. The public
//! surface is exactly upstream's: the fields `u` and `d`, `new` and `d_matrix` (WP 8.0: the
//! `LDLᵀ` `l`, `solve`, `inverse` and `determinant`, which upstream's `UDU` does not have, are
//! crate-internal).

use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::Real;
use crate::internal::linalg::ldlt::Ldlt3Trait;

/// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of a symmetric 3x3 matrix: the unit UPPER
/// triangular `u` (ones on the diagonal, zeros below it) and the diagonal `d`. The fields are
/// upstream's (`UDU { u, d }`). Build one with `Udu3Trait::new`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Udu3<T> {
    /// The unit upper triangular factor.
    pub u: Matrix3<T>,
    /// The diagonal of `diag(d)`.
    pub d: Vector3<T>,
}

/// Methods of `Udu3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Udu3Impl<
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
> of Udu3Trait<T> {
    /// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of the symmetric `p`, or `None` when a
    /// pivot is exactly zero. Like upstream, only the UPPER triangle of `p` is read (the entries
    /// at row `i`, column `j` with `i <= j`) and the symmetry of `p` is NOT checked. Upstream:
    /// `UDU::new`.
    ///
    /// Computed as the crate-internal `LDLᵀ` kernel (`crate::linalg::ldlt`) of the reversed
    /// matrix `J·p·J`, whose factors are reversed back: `u = J·l'·J`, `d = J·d'`. This is
    /// upstream's algorithm read backwards, operation for operation (upstream factors the trailing
    /// submatrices, from `d_3 = p_33` up); the reversals are moves only, so the cost and the
    /// rounding are those of the kernel: every numerator is one exact accumulation floored once,
    /// every entry of `u` one correctly rounded division, no square root. SINGULARITY CRITERION:
    /// `None` iff a pivot is EXACTLY zero, like upstream's `is_zero` test. Indefinite matrices are
    /// accepted. Panics on overflow; never wraps.
    fn new(p: Matrix3<T>) -> Option<Udu3<T>> {
        match Ldlt3Trait::new(
            SymMatrix3 { m11: p.m33, m12: p.m23, m13: p.m13, m22: p.m22, m23: p.m12, m33: p.m11 },
        ) {
            Some(f) => Some(
                Udu3 {
                    u: Matrix3 {
                        m11: R::one(),
                        m21: R::zero(),
                        m31: R::zero(),
                        m12: f.l32,
                        m22: R::one(),
                        m32: R::zero(),
                        m13: f.l31,
                        m23: f.l21,
                        m33: R::one(),
                    },
                    d: Vector3 { x: f.d.z, y: f.d.y, z: f.d.x },
                },
            ),
            None => None,
        }
    }

    /// `diag(d)`, the diagonal factor as a matrix. Exact. Upstream: `UDU::d_matrix`.
    #[inline(always)]
    fn d_matrix(self: Udu3<T>) -> Matrix3<T> {
        Matrix3Trait::from_diagonal(self.d)
    }
}

/// `SquareMatrix::udu` on `Matrix3<T>` (upstream `nalgebra::linalg` decomposition entry point).
#[generate_trait]
pub impl Matrix3UduImpl<
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
> of Matrix3UduTrait<T> {
    /// The `UDUᵀ` factorisation of the symmetric `self` (its UPPER triangle), or `None` when a
    /// pivot is exactly zero: `Udu3Trait::new(self)`. Upstream: `SquareMatrix::udu`.
    #[inline(always)]
    fn udu(self: Matrix3<T>) -> Option<Udu3<T>> {
        Udu3Trait::new(self)
    }
}
