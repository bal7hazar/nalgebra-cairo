//! In `nalgebra_linalg2`: the `struct` `Udu2`; its 2 impls, among them `Udu2Impl`,
//! `Matrix2UduImpl`. This module is split over packages; the other parts are in `nalgebra_linalg3`,
//! `nalgebra_linalg4`, `nalgebra_linalg6`.
//!
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

use nalgebra_static2::base::matrix2::Matrix2Trait;
use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::internal::base::sym_matrix2::SymMatrix2;
use simba::scalar::Real;
use crate::internal::linalg::ldlt::Ldlt2Trait;

/// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of a symmetric 2x2 matrix: the unit UPPER
/// triangular `u` (ones on the diagonal, zeros below it) and the diagonal `d`. The fields are
/// upstream's (`UDU { u, d }`). Build one with `Udu2Trait::new`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Udu2<T> {
    /// The unit upper triangular factor.
    pub u: Matrix2<T>,
    /// The diagonal of `diag(d)`.
    pub d: Vector2<T>,
}

/// Methods of `Udu2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Udu2Impl<
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
> of Udu2Trait<T> {
    /// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of the symmetric `p`, or `None` when a
    /// pivot is exactly zero. Like upstream, only the UPPER triangle of `p` is read (the entries
    /// at row `i`, column `j` with `i <= j`) and the symmetry of `p` is NOT checked. Upstream:
    /// `UDU::new`.
    ///
    /// Computed as the crate-internal `LDLᵀ` kernel (`crate::linalg::ldlt`) of the reversed
    /// matrix `J·p·J`, whose factors are reversed back: `u = J·l'·J`, `d = J·d'`. This is
    /// upstream's algorithm read backwards, operation for operation (upstream factors the trailing
    /// submatrices, from `d_2 = p_22` up); the reversals are moves only, so the cost and the
    /// rounding are those of the kernel: every numerator is one exact accumulation floored once,
    /// every entry of `u` one correctly rounded division, no square root. SINGULARITY CRITERION:
    /// `None` iff a pivot is EXACTLY zero, like upstream's `is_zero` test. Indefinite matrices are
    /// accepted. Panics on overflow; never wraps.
    fn new(p: Matrix2<T>) -> Option<Udu2<T>> {
        match Ldlt2Trait::new(SymMatrix2 { m11: p.m22, m12: p.m12, m22: p.m11 }) {
            Some(f) => Some(
                Udu2 {
                    u: Matrix2 { m11: R::one(), m21: R::zero(), m12: f.l21, m22: R::one() },
                    d: Vector2 { x: f.d.y, y: f.d.x },
                },
            ),
            None => None,
        }
    }

    /// `diag(d)`, the diagonal factor as a matrix. Exact. Upstream: `UDU::d_matrix`.
    #[inline(always)]
    fn d_matrix(self: Udu2<T>) -> Matrix2<T> {
        Matrix2Trait::from_diagonal(self.d)
    }
}

/// `SquareMatrix::udu` on `Matrix2<T>` (upstream `nalgebra::linalg` decomposition entry point).
#[generate_trait]
pub impl Matrix2UduImpl<
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
> of Matrix2UduTrait<T> {
    /// The `UDUᵀ` factorisation of the symmetric `self` (its UPPER triangle), or `None` when a
    /// pivot is exactly zero: `Udu2Trait::new(self)`. Upstream: `SquareMatrix::udu`.
    #[inline(always)]
    fn udu(self: Matrix2<T>) -> Option<Udu2<T>> {
        Udu2Trait::new(self)
    }
}
