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

use simba::scalar::Real;
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::matrix6::{Matrix6, Matrix6Trait};
use crate::base::sym_matrix2::SymMatrix2;
use crate::base::sym_matrix3::SymMatrix3;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector6::Vector6;
use crate::linalg::ldlt::{Ldlt2Trait, Ldlt3Trait, Ldlt4Trait, Ldlt6Trait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

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

/// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of a symmetric 4x4 matrix: the unit UPPER
/// triangular `u` (ones on the diagonal, zeros below it) and the diagonal `d`. The fields are
/// upstream's (`UDU { u, d }`). Build one with `Udu4Trait::new`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Udu4<T> {
    /// The unit upper triangular factor.
    pub u: Matrix4<T>,
    /// The diagonal of `diag(d)`.
    pub d: Vector4<T>,
}

/// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of a symmetric 6x6 matrix: the unit UPPER
/// triangular `u` (ones on the diagonal, zeros below it) and the diagonal `d`. The fields are
/// upstream's (`UDU { u, d }`). Build one with `Udu6Trait::new`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Udu6<T> {
    /// The unit upper triangular factor.
    pub u: Matrix6<T>,
    /// The diagonal of `diag(d)`.
    pub d: Vector6<T>,
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

/// Methods of `Udu4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Udu4Impl<
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
> of Udu4Trait<T> {
    /// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of the symmetric `p`, or `None` when a
    /// pivot is exactly zero. Like upstream, only the UPPER triangle of `p` is read (the entries
    /// at row `i`, column `j` with `i <= j`) and the symmetry of `p` is NOT checked. Upstream:
    /// `UDU::new`.
    ///
    /// Computed as the crate-internal `LDLᵀ` kernel (`crate::linalg::ldlt`) of the reversed
    /// matrix `J·p·J`, whose factors are reversed back: `u = J·l'·J`, `d = J·d'`. This is
    /// upstream's algorithm read backwards, operation for operation (upstream factors the trailing
    /// submatrices, from `d_4 = p_44` up); the reversals are moves only, so the cost and the
    /// rounding are those of the kernel: every numerator is one exact accumulation floored once,
    /// every entry of `u` one correctly rounded division, no square root. SINGULARITY CRITERION:
    /// `None` iff a pivot is EXACTLY zero, like upstream's `is_zero` test. Indefinite matrices are
    /// accepted. Panics on overflow; never wraps.
    fn new(p: Matrix4<T>) -> Option<Udu4<T>> {
        match Ldlt4Trait::new(
            Matrix4 {
                m11: p.m44,
                m21: p.m34,
                m31: p.m24,
                m41: p.m14,
                m12: p.m43,
                m22: p.m33,
                m32: p.m23,
                m42: p.m13,
                m13: p.m42,
                m23: p.m32,
                m33: p.m22,
                m43: p.m12,
                m14: p.m41,
                m24: p.m31,
                m34: p.m21,
                m44: p.m11,
            },
        ) {
            Some(f) => Some(
                Udu4 {
                    u: Matrix4 {
                        m11: R::one(),
                        m21: R::zero(),
                        m31: R::zero(),
                        m41: R::zero(),
                        m12: f.l43,
                        m22: R::one(),
                        m32: R::zero(),
                        m42: R::zero(),
                        m13: f.l42,
                        m23: f.l32,
                        m33: R::one(),
                        m43: R::zero(),
                        m14: f.l41,
                        m24: f.l31,
                        m34: f.l21,
                        m44: R::one(),
                    },
                    d: Vector4 { x: f.d.w, y: f.d.z, z: f.d.y, w: f.d.x },
                },
            ),
            None => None,
        }
    }

    /// `diag(d)`, the diagonal factor as a matrix. Exact. Upstream: `UDU::d_matrix`.
    #[inline(always)]
    fn d_matrix(self: Udu4<T>) -> Matrix4<T> {
        Matrix4Trait::from_diagonal(self.d)
    }
}

/// Methods of `Udu6<T>` for any `Real` scalar.
#[generate_trait]
pub impl Udu6Impl<
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
> of Udu6Trait<T> {
    /// The `UDUᵀ` factorisation `p = u * diag(d) * uᵀ` of the symmetric `p`, or `None` when a
    /// pivot is exactly zero. Like upstream, only the UPPER triangle of `p` is read (the entries
    /// at row `i`, column `j` with `i <= j`) and the symmetry of `p` is NOT checked. Upstream:
    /// `UDU::new`.
    ///
    /// Computed as the crate-internal `LDLᵀ` kernel (`crate::linalg::ldlt`) of the reversed
    /// matrix `J·p·J`, whose factors are reversed back: `u = J·l'·J`, `d = J·d'`. This is
    /// upstream's algorithm read backwards, operation for operation (upstream factors the trailing
    /// submatrices, from `d_6 = p_66` up); the reversals are moves only, so the cost and the
    /// rounding are those of the kernel: every numerator is one exact accumulation floored once,
    /// every entry of `u` one correctly rounded division, no square root. SINGULARITY CRITERION:
    /// `None` iff a pivot is EXACTLY zero, like upstream's `is_zero` test. Indefinite matrices are
    /// accepted. Panics on overflow; never wraps.
    fn new(p: Matrix6<T>) -> Option<Udu6<T>> {
        match Ldlt6Trait::new(
            Matrix6 {
                m11: Matrix3 {
                    m11: p.m22.m33,
                    m21: p.m22.m23,
                    m31: p.m22.m13,
                    m12: p.m22.m32,
                    m22: p.m22.m22,
                    m32: p.m22.m12,
                    m13: p.m22.m31,
                    m23: p.m22.m21,
                    m33: p.m22.m11,
                },
                m21: Matrix3 {
                    m11: p.m12.m33,
                    m21: p.m12.m23,
                    m31: p.m12.m13,
                    m12: p.m12.m32,
                    m22: p.m12.m22,
                    m32: p.m12.m12,
                    m13: p.m12.m31,
                    m23: p.m12.m21,
                    m33: p.m12.m11,
                },
                m12: Matrix3 {
                    m11: p.m21.m33,
                    m21: p.m21.m23,
                    m31: p.m21.m13,
                    m12: p.m21.m32,
                    m22: p.m21.m22,
                    m32: p.m21.m12,
                    m13: p.m21.m31,
                    m23: p.m21.m21,
                    m33: p.m21.m11,
                },
                m22: Matrix3 {
                    m11: p.m11.m33,
                    m21: p.m11.m23,
                    m31: p.m11.m13,
                    m12: p.m11.m32,
                    m22: p.m11.m22,
                    m32: p.m11.m12,
                    m13: p.m11.m31,
                    m23: p.m11.m21,
                    m33: p.m11.m11,
                },
            },
        ) {
            Some(f) => Some(
                Udu6 {
                    u: Matrix6 {
                        m11: Matrix3 {
                            m11: R::one(),
                            m21: R::zero(),
                            m31: R::zero(),
                            m12: f.l65,
                            m22: R::one(),
                            m32: R::zero(),
                            m13: f.l64,
                            m23: f.l54,
                            m33: R::one(),
                        },
                        m21: Matrix3 {
                            m11: R::zero(),
                            m21: R::zero(),
                            m31: R::zero(),
                            m12: R::zero(),
                            m22: R::zero(),
                            m32: R::zero(),
                            m13: R::zero(),
                            m23: R::zero(),
                            m33: R::zero(),
                        },
                        m12: Matrix3 {
                            m11: f.l63,
                            m21: f.l53,
                            m31: f.l43,
                            m12: f.l62,
                            m22: f.l52,
                            m32: f.l42,
                            m13: f.l61,
                            m23: f.l51,
                            m33: f.l41,
                        },
                        m22: Matrix3 {
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
                    },
                    d: Vector6 {
                        a: Vector3 { x: f.d.b.z, y: f.d.b.y, z: f.d.b.x },
                        b: Vector3 { x: f.d.a.z, y: f.d.a.y, z: f.d.a.x },
                    },
                },
            ),
            None => None,
        }
    }

    /// `diag(d)`, the diagonal factor as a matrix. Exact. Upstream: `UDU::d_matrix`.
    #[inline(always)]
    fn d_matrix(self: Udu6<T>) -> Matrix6<T> {
        Matrix6Trait::from_diagonal(self.d)
    }
}
