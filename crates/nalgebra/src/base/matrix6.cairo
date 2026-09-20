//! `Matrix6`: a statically sized 6x6 matrix (upstream `nalgebra::Matrix6`), stored as 2x2 blocks
//! of `Matrix3` (DESIGN D4: the spatial-algebra layout used by rapier's multibody and soft-body
//! solvers, where a 6x6 spatial inertia or Jacobian is naturally four 3x3 blocks).
//!
//! Every sum of products goes through a fused `Real` kernel: one rounding (floor) and one overflow
//! check per output scalar. The 6-term rows of `mul_vec`, `tr_mul_vec`, `*` and `tr_mul` use the
//! explicit `Real::Wide` accumulator, so a product of two `Matrix6` is bit-identical to the flat
//! 6x6 product with one rescale per output scalar — NOT the sum of four rounded 3x3 block
//! products (which rounds twice; kept as `bench_matrix6_mul__alt_blocks` with the test showing the
//! drift).
//!
//! Operators (`+`, `-`, unary `-`, `*` and their assigning forms) are implemented in this module,
//! so they need no import; the other operations are methods of `Matrix6Trait`.
//!
//! The determinant, the inverse and the decompositions of a 6x6 are NOT here: they belong to the
//! LU / Cholesky work package (DESIGN D6). No block formula gives them cheaply — `try_inverse`
//! through the Schur complement `(m11 - m12 m22^-1 m21)^-1` needs two 3x3 inversions and four
//! 3x3 products, each rounding again, and is numerically far worse than a pivoted factorisation.

use core::ops::{AddAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::matrix3::{Matrix3, Matrix3Trait};
use super::vector3::Vector3;
use super::vector6::Vector6;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

/// A 6x6 matrix, as a 2x2 grid of 3x3 blocks.
///
/// `mIJ` is the 3x3 BLOCK at block-row `I`, block-column `J`, so the scalar at row `r`, column `c`
/// (1-based, as upstream numbers them) is `m.m<I><J>.m<r-3(I-1)><c-3(J-1)>`: the scalar at row 6,
/// column 2 is `m.m21.m32`. `new` takes the 36 components in ROW-major order like upstream
/// `Matrix6::new`.
///
/// Blocks are declared in block-column-major order and each `Matrix3` declares its fields in
/// column-major order, so `Serde` writes `m11`, `m21`, `m12`, `m22` block by block. That is NOT
/// upstream's flat column-major order: `Matrix6` is a block type here (DESIGN D4), not a 36-scalar
/// buffer, and reading it back as a flat upstream matrix would interleave the blocks wrongly.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix6<T> {
    /// Rows 1-3, columns 1-3.
    pub m11: Matrix3<T>,
    /// Rows 4-6, columns 1-3.
    pub m21: Matrix3<T>,
    /// Rows 1-3, columns 4-6.
    pub m12: Matrix3<T>,
    /// Rows 4-6, columns 4-6.
    pub m22: Matrix3<T>,
}

/// Methods of `Matrix6<T>` for any `Real` scalar.
#[generate_trait]
pub impl Matrix6Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Matrix6Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// The matrix with the given 36 components, in ROW-major order (`m11, m12, .., m16, m21, ..`).
    /// Upstream: `Matrix6::new`.
    #[inline(always)]
    fn new(
        m11: T,
        m12: T,
        m13: T,
        m14: T,
        m15: T,
        m16: T,
        m21: T,
        m22: T,
        m23: T,
        m24: T,
        m25: T,
        m26: T,
        m31: T,
        m32: T,
        m33: T,
        m34: T,
        m35: T,
        m36: T,
        m41: T,
        m42: T,
        m43: T,
        m44: T,
        m45: T,
        m46: T,
        m51: T,
        m52: T,
        m53: T,
        m54: T,
        m55: T,
        m56: T,
        m61: T,
        m62: T,
        m63: T,
        m64: T,
        m65: T,
        m66: T,
    ) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: m11,
                m21: m21,
                m31: m31,
                m12: m12,
                m22: m22,
                m32: m32,
                m13: m13,
                m23: m23,
                m33: m33,
            },
            m21: Matrix3 {
                m11: m41,
                m21: m51,
                m31: m61,
                m12: m42,
                m22: m52,
                m32: m62,
                m13: m43,
                m23: m53,
                m33: m63,
            },
            m12: Matrix3 {
                m11: m14,
                m21: m24,
                m31: m34,
                m12: m15,
                m22: m25,
                m32: m35,
                m13: m16,
                m23: m26,
                m33: m36,
            },
            m22: Matrix3 {
                m11: m44,
                m21: m54,
                m31: m64,
                m12: m45,
                m22: m55,
                m32: m65,
                m13: m46,
                m23: m56,
                m33: m66,
            },
        }
    }

    /// The zero matrix. Upstream: `Matrix6::zeros`.
    #[inline(always)]
    fn zeros() -> Matrix6<T> {
        let z = Matrix3Trait::zeros();
        Matrix6 { m11: z, m21: z, m12: z, m22: z }
    }

    /// The identity matrix. Upstream: `Matrix6::identity`.
    #[inline(always)]
    fn identity() -> Matrix6<T> {
        let z = Matrix3Trait::zeros();
        let i = Matrix3Trait::identity();
        Matrix6 { m11: i, m21: z, m12: z, m22: i }
    }

    /// The matrix made of the four given 3x3 blocks. No upstream equivalent (upstream assigns
    /// `fixed_view_mut::<3, 3>(r, c)`); rapier builds spatial inertia and Jacobian matrices this
    /// way.
    #[inline(always)]
    fn from_blocks(
        m11: Matrix3<T>, m12: Matrix3<T>, m21: Matrix3<T>, m22: Matrix3<T>,
    ) -> Matrix6<T> {
        Matrix6 { m11, m21, m12, m22 }
    }

    /// The diagonal matrix `diag(d)`. Upstream: `Matrix6::from_diagonal`.
    #[inline(always)]
    fn from_diagonal(d: Vector6<T>) -> Matrix6<T> {
        let z = Matrix3Trait::zeros();
        Matrix6 {
            m11: Matrix3Trait::from_diagonal(d.a),
            m21: z,
            m12: z,
            m22: Matrix3Trait::from_diagonal(d.b),
        }
    }

    /// The matrix `e * I`. Upstream: `Matrix6::from_diagonal_element`.
    #[inline(always)]
    fn from_diagonal_element(e: T) -> Matrix6<T> {
        let z = Matrix3Trait::zeros();
        let d = Matrix3Trait::from_diagonal_element(e);
        Matrix6 { m11: d, m21: z, m12: z, m22: d }
    }

    // --- accessors -----------------------------------------------------------------------------

    /// The block of rows 1-3, columns 1-3. Upstream: `fixed_view::<3, 3>(0, 0)`.
    #[inline(always)]
    fn block11(self: Matrix6<T>) -> Matrix3<T> {
        self.m11
    }

    /// The block of rows 1-3, columns 4-6. Upstream: `fixed_view::<3, 3>(0, 3)`.
    #[inline(always)]
    fn block12(self: Matrix6<T>) -> Matrix3<T> {
        self.m12
    }

    /// The block of rows 4-6, columns 1-3. Upstream: `fixed_view::<3, 3>(3, 0)`.
    #[inline(always)]
    fn block21(self: Matrix6<T>) -> Matrix3<T> {
        self.m21
    }

    /// The block of rows 4-6, columns 4-6. Upstream: `fixed_view::<3, 3>(3, 3)`.
    #[inline(always)]
    fn block22(self: Matrix6<T>) -> Matrix3<T> {
        self.m22
    }

    /// The diagonal, as a `Vector6`. Upstream: `diagonal`.
    #[inline(always)]
    fn diagonal(self: Matrix6<T>) -> Vector6<T> {
        Vector6 { a: Matrix3Trait::diagonal(self.m11), b: Matrix3Trait::diagonal(self.m22) }
    }

    // --- exact operations ----------------------------------------------------------------------

    /// The transpose: each block is transposed and the two off-diagonal blocks are swapped. Exact.
    /// Upstream: `transpose`.
    #[inline(always)]
    fn transpose(self: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3Trait::transpose(self.m11),
            m21: Matrix3Trait::transpose(self.m12),
            m12: Matrix3Trait::transpose(self.m21),
            m22: Matrix3Trait::transpose(self.m22),
        }
    }

    /// Sum of the six diagonal components (the traces of the two diagonal blocks). Exact; panics
    /// on overflow. Upstream: `trace`.
    #[inline(always)]
    fn trace(self: Matrix6<T>) -> T {
        Matrix3Trait::trace(self.m11) + Matrix3Trait::trace(self.m22)
    }

    /// Component-wise absolute value. Panics on the scalar's `MIN`. Upstream: `abs`.
    #[inline(always)]
    fn abs(self: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3Trait::abs(self.m11),
            m21: Matrix3Trait::abs(self.m21),
            m12: Matrix3Trait::abs(self.m12),
            m22: Matrix3Trait::abs(self.m22),
        }
    }

    // --- products ------------------------------------------------------------------------------

    /// `self * k`: each component is one floored product. Panics on overflow.
    /// Upstream: `self * k`.
    #[inline(always)]
    fn scale(self: Matrix6<T>, k: T) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3Trait::scale(self.m11, k),
            m21: Matrix3Trait::scale(self.m21, k),
            m12: Matrix3Trait::scale(self.m12, k),
            m22: Matrix3Trait::scale(self.m22, k),
        }
    }

    /// `self * v`: each of the six output components is the exact sum of the six products of a row
    /// by `v`, accumulated in `Real::Wide` and rescaled ONCE. Panics on overflow of a component.
    /// Upstream: `self * v`.
    ///
    /// Summing the two rounded 3x3 block products (`m11 * v.a + m12 * v.b`) rounds twice AND
    /// costs 1.67x more gas (`bench_matrix6_mul_vec__alt_blocks`: 37 770 against 22 560 net).
    fn mul_vec(self: Matrix6<T>, v: Vector6<T>) -> Vector6<T> {
        Vector6 {
            a: Vector3 {
                x: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, v.a.x),
                                        self.m11.m12,
                                        v.a.y,
                                    ),
                                    self.m11.m13,
                                    v.a.z,
                                ),
                                self.m12.m11,
                                v.b.x,
                            ),
                            self.m12.m12,
                            v.b.y,
                        ),
                        self.m12.m13,
                        v.b.z,
                    ),
                ),
                y: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m21, v.a.x),
                                        self.m11.m22,
                                        v.a.y,
                                    ),
                                    self.m11.m23,
                                    v.a.z,
                                ),
                                self.m12.m21,
                                v.b.x,
                            ),
                            self.m12.m22,
                            v.b.y,
                        ),
                        self.m12.m23,
                        v.b.z,
                    ),
                ),
                z: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m31, v.a.x),
                                        self.m11.m32,
                                        v.a.y,
                                    ),
                                    self.m11.m33,
                                    v.a.z,
                                ),
                                self.m12.m31,
                                v.b.x,
                            ),
                            self.m12.m32,
                            v.b.y,
                        ),
                        self.m12.m33,
                        v.b.z,
                    ),
                ),
            },
            b: Vector3 {
                x: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m21.m11, v.a.x),
                                        self.m21.m12,
                                        v.a.y,
                                    ),
                                    self.m21.m13,
                                    v.a.z,
                                ),
                                self.m22.m11,
                                v.b.x,
                            ),
                            self.m22.m12,
                            v.b.y,
                        ),
                        self.m22.m13,
                        v.b.z,
                    ),
                ),
                y: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m21.m21, v.a.x),
                                        self.m21.m22,
                                        v.a.y,
                                    ),
                                    self.m21.m23,
                                    v.a.z,
                                ),
                                self.m22.m21,
                                v.b.x,
                            ),
                            self.m22.m22,
                            v.b.y,
                        ),
                        self.m22.m23,
                        v.b.z,
                    ),
                ),
                z: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m21.m31, v.a.x),
                                        self.m21.m32,
                                        v.a.y,
                                    ),
                                    self.m21.m33,
                                    v.a.z,
                                ),
                                self.m22.m31,
                                v.b.x,
                            ),
                            self.m22.m32,
                            v.b.y,
                        ),
                        self.m22.m33,
                        v.b.z,
                    ),
                ),
            },
        }
    }

    /// `selfᵀ * v` without forming the transpose: one 6-term `Real::Wide` accumulation and ONE
    /// rescale per component. Panics on overflow. Upstream: `self.tr_mul(&v)`.
    ///
    /// Bit-identical to `self.transpose().mul_vec(v)` AND exactly as expensive (measured:
    /// `bench_matrix6_tr_mul_vec__fused` and `__alt_transpose_then_mul_vec` are both 22 560 net).
    /// `transpose` only relabels SSA values, so it is free; this method exists for upstream
    /// parity and readability, not for gas.
    fn tr_mul_vec(self: Matrix6<T>, v: Vector6<T>) -> Vector6<T> {
        Vector6 {
            a: Vector3 {
                x: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, v.a.x),
                                        self.m11.m21,
                                        v.a.y,
                                    ),
                                    self.m11.m31,
                                    v.a.z,
                                ),
                                self.m21.m11,
                                v.b.x,
                            ),
                            self.m21.m21,
                            v.b.y,
                        ),
                        self.m21.m31,
                        v.b.z,
                    ),
                ),
                y: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, v.a.x),
                                        self.m11.m22,
                                        v.a.y,
                                    ),
                                    self.m11.m32,
                                    v.a.z,
                                ),
                                self.m21.m12,
                                v.b.x,
                            ),
                            self.m21.m22,
                            v.b.y,
                        ),
                        self.m21.m32,
                        v.b.z,
                    ),
                ),
                z: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, v.a.x),
                                        self.m11.m23,
                                        v.a.y,
                                    ),
                                    self.m11.m33,
                                    v.a.z,
                                ),
                                self.m21.m13,
                                v.b.x,
                            ),
                            self.m21.m23,
                            v.b.y,
                        ),
                        self.m21.m33,
                        v.b.z,
                    ),
                ),
            },
            b: Vector3 {
                x: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, v.a.x),
                                        self.m12.m21,
                                        v.a.y,
                                    ),
                                    self.m12.m31,
                                    v.a.z,
                                ),
                                self.m22.m11,
                                v.b.x,
                            ),
                            self.m22.m21,
                            v.b.y,
                        ),
                        self.m22.m31,
                        v.b.z,
                    ),
                ),
                y: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, v.a.x),
                                        self.m12.m22,
                                        v.a.y,
                                    ),
                                    self.m12.m32,
                                    v.a.z,
                                ),
                                self.m22.m12,
                                v.b.x,
                            ),
                            self.m22.m22,
                            v.b.y,
                        ),
                        self.m22.m32,
                        v.b.z,
                    ),
                ),
                z: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, v.a.x),
                                        self.m12.m23,
                                        v.a.y,
                                    ),
                                    self.m12.m33,
                                    v.a.z,
                                ),
                                self.m22.m13,
                                v.b.x,
                            ),
                            self.m22.m23,
                            v.b.y,
                        ),
                        self.m22.m33,
                        v.b.z,
                    ),
                ),
            },
        }
    }

    /// `selfᵀ * rhs` without forming the transpose: 36 six-term `Real::Wide` accumulations, one
    /// rescale per output component. Panics on overflow. Upstream: `tr_mul`.
    ///
    /// Bit-identical to `self.transpose() * rhs` AND exactly as expensive (measured:
    /// `bench_matrix6_tr_mul__fused` and `__alt_transpose_then_mul` are both 114 660 net), for the
    /// same reason as `tr_mul_vec`: transposing is free.
    fn tr_mul(self: Matrix6<T>, rhs: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, rhs.m11.m11),
                                        self.m11.m21,
                                        rhs.m11.m21,
                                    ),
                                    self.m11.m31,
                                    rhs.m11.m31,
                                ),
                                self.m21.m11,
                                rhs.m21.m11,
                            ),
                            self.m21.m21,
                            rhs.m21.m21,
                        ),
                        self.m21.m31,
                        rhs.m21.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, rhs.m11.m11),
                                        self.m11.m22,
                                        rhs.m11.m21,
                                    ),
                                    self.m11.m32,
                                    rhs.m11.m31,
                                ),
                                self.m21.m12,
                                rhs.m21.m11,
                            ),
                            self.m21.m22,
                            rhs.m21.m21,
                        ),
                        self.m21.m32,
                        rhs.m21.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, rhs.m11.m11),
                                        self.m11.m23,
                                        rhs.m11.m21,
                                    ),
                                    self.m11.m33,
                                    rhs.m11.m31,
                                ),
                                self.m21.m13,
                                rhs.m21.m11,
                            ),
                            self.m21.m23,
                            rhs.m21.m21,
                        ),
                        self.m21.m33,
                        rhs.m21.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, rhs.m11.m12),
                                        self.m11.m21,
                                        rhs.m11.m22,
                                    ),
                                    self.m11.m31,
                                    rhs.m11.m32,
                                ),
                                self.m21.m11,
                                rhs.m21.m12,
                            ),
                            self.m21.m21,
                            rhs.m21.m22,
                        ),
                        self.m21.m31,
                        rhs.m21.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, rhs.m11.m12),
                                        self.m11.m22,
                                        rhs.m11.m22,
                                    ),
                                    self.m11.m32,
                                    rhs.m11.m32,
                                ),
                                self.m21.m12,
                                rhs.m21.m12,
                            ),
                            self.m21.m22,
                            rhs.m21.m22,
                        ),
                        self.m21.m32,
                        rhs.m21.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, rhs.m11.m12),
                                        self.m11.m23,
                                        rhs.m11.m22,
                                    ),
                                    self.m11.m33,
                                    rhs.m11.m32,
                                ),
                                self.m21.m13,
                                rhs.m21.m12,
                            ),
                            self.m21.m23,
                            rhs.m21.m22,
                        ),
                        self.m21.m33,
                        rhs.m21.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, rhs.m11.m13),
                                        self.m11.m21,
                                        rhs.m11.m23,
                                    ),
                                    self.m11.m31,
                                    rhs.m11.m33,
                                ),
                                self.m21.m11,
                                rhs.m21.m13,
                            ),
                            self.m21.m21,
                            rhs.m21.m23,
                        ),
                        self.m21.m31,
                        rhs.m21.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, rhs.m11.m13),
                                        self.m11.m22,
                                        rhs.m11.m23,
                                    ),
                                    self.m11.m32,
                                    rhs.m11.m33,
                                ),
                                self.m21.m12,
                                rhs.m21.m13,
                            ),
                            self.m21.m22,
                            rhs.m21.m23,
                        ),
                        self.m21.m32,
                        rhs.m21.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, rhs.m11.m13),
                                        self.m11.m23,
                                        rhs.m11.m23,
                                    ),
                                    self.m11.m33,
                                    rhs.m11.m33,
                                ),
                                self.m21.m13,
                                rhs.m21.m13,
                            ),
                            self.m21.m23,
                            rhs.m21.m23,
                        ),
                        self.m21.m33,
                        rhs.m21.m33,
                    ),
                ),
            },
            m21: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, rhs.m11.m11),
                                        self.m12.m21,
                                        rhs.m11.m21,
                                    ),
                                    self.m12.m31,
                                    rhs.m11.m31,
                                ),
                                self.m22.m11,
                                rhs.m21.m11,
                            ),
                            self.m22.m21,
                            rhs.m21.m21,
                        ),
                        self.m22.m31,
                        rhs.m21.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, rhs.m11.m11),
                                        self.m12.m22,
                                        rhs.m11.m21,
                                    ),
                                    self.m12.m32,
                                    rhs.m11.m31,
                                ),
                                self.m22.m12,
                                rhs.m21.m11,
                            ),
                            self.m22.m22,
                            rhs.m21.m21,
                        ),
                        self.m22.m32,
                        rhs.m21.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, rhs.m11.m11),
                                        self.m12.m23,
                                        rhs.m11.m21,
                                    ),
                                    self.m12.m33,
                                    rhs.m11.m31,
                                ),
                                self.m22.m13,
                                rhs.m21.m11,
                            ),
                            self.m22.m23,
                            rhs.m21.m21,
                        ),
                        self.m22.m33,
                        rhs.m21.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, rhs.m11.m12),
                                        self.m12.m21,
                                        rhs.m11.m22,
                                    ),
                                    self.m12.m31,
                                    rhs.m11.m32,
                                ),
                                self.m22.m11,
                                rhs.m21.m12,
                            ),
                            self.m22.m21,
                            rhs.m21.m22,
                        ),
                        self.m22.m31,
                        rhs.m21.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, rhs.m11.m12),
                                        self.m12.m22,
                                        rhs.m11.m22,
                                    ),
                                    self.m12.m32,
                                    rhs.m11.m32,
                                ),
                                self.m22.m12,
                                rhs.m21.m12,
                            ),
                            self.m22.m22,
                            rhs.m21.m22,
                        ),
                        self.m22.m32,
                        rhs.m21.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, rhs.m11.m12),
                                        self.m12.m23,
                                        rhs.m11.m22,
                                    ),
                                    self.m12.m33,
                                    rhs.m11.m32,
                                ),
                                self.m22.m13,
                                rhs.m21.m12,
                            ),
                            self.m22.m23,
                            rhs.m21.m22,
                        ),
                        self.m22.m33,
                        rhs.m21.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, rhs.m11.m13),
                                        self.m12.m21,
                                        rhs.m11.m23,
                                    ),
                                    self.m12.m31,
                                    rhs.m11.m33,
                                ),
                                self.m22.m11,
                                rhs.m21.m13,
                            ),
                            self.m22.m21,
                            rhs.m21.m23,
                        ),
                        self.m22.m31,
                        rhs.m21.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, rhs.m11.m13),
                                        self.m12.m22,
                                        rhs.m11.m23,
                                    ),
                                    self.m12.m32,
                                    rhs.m11.m33,
                                ),
                                self.m22.m12,
                                rhs.m21.m13,
                            ),
                            self.m22.m22,
                            rhs.m21.m23,
                        ),
                        self.m22.m32,
                        rhs.m21.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, rhs.m11.m13),
                                        self.m12.m23,
                                        rhs.m11.m23,
                                    ),
                                    self.m12.m33,
                                    rhs.m11.m33,
                                ),
                                self.m22.m13,
                                rhs.m21.m13,
                            ),
                            self.m22.m23,
                            rhs.m21.m23,
                        ),
                        self.m22.m33,
                        rhs.m21.m33,
                    ),
                ),
            },
            m12: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, rhs.m12.m11),
                                        self.m11.m21,
                                        rhs.m12.m21,
                                    ),
                                    self.m11.m31,
                                    rhs.m12.m31,
                                ),
                                self.m21.m11,
                                rhs.m22.m11,
                            ),
                            self.m21.m21,
                            rhs.m22.m21,
                        ),
                        self.m21.m31,
                        rhs.m22.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, rhs.m12.m11),
                                        self.m11.m22,
                                        rhs.m12.m21,
                                    ),
                                    self.m11.m32,
                                    rhs.m12.m31,
                                ),
                                self.m21.m12,
                                rhs.m22.m11,
                            ),
                            self.m21.m22,
                            rhs.m22.m21,
                        ),
                        self.m21.m32,
                        rhs.m22.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, rhs.m12.m11),
                                        self.m11.m23,
                                        rhs.m12.m21,
                                    ),
                                    self.m11.m33,
                                    rhs.m12.m31,
                                ),
                                self.m21.m13,
                                rhs.m22.m11,
                            ),
                            self.m21.m23,
                            rhs.m22.m21,
                        ),
                        self.m21.m33,
                        rhs.m22.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, rhs.m12.m12),
                                        self.m11.m21,
                                        rhs.m12.m22,
                                    ),
                                    self.m11.m31,
                                    rhs.m12.m32,
                                ),
                                self.m21.m11,
                                rhs.m22.m12,
                            ),
                            self.m21.m21,
                            rhs.m22.m22,
                        ),
                        self.m21.m31,
                        rhs.m22.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, rhs.m12.m12),
                                        self.m11.m22,
                                        rhs.m12.m22,
                                    ),
                                    self.m11.m32,
                                    rhs.m12.m32,
                                ),
                                self.m21.m12,
                                rhs.m22.m12,
                            ),
                            self.m21.m22,
                            rhs.m22.m22,
                        ),
                        self.m21.m32,
                        rhs.m22.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, rhs.m12.m12),
                                        self.m11.m23,
                                        rhs.m12.m22,
                                    ),
                                    self.m11.m33,
                                    rhs.m12.m32,
                                ),
                                self.m21.m13,
                                rhs.m22.m12,
                            ),
                            self.m21.m23,
                            rhs.m22.m22,
                        ),
                        self.m21.m33,
                        rhs.m22.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m11, rhs.m12.m13),
                                        self.m11.m21,
                                        rhs.m12.m23,
                                    ),
                                    self.m11.m31,
                                    rhs.m12.m33,
                                ),
                                self.m21.m11,
                                rhs.m22.m13,
                            ),
                            self.m21.m21,
                            rhs.m22.m23,
                        ),
                        self.m21.m31,
                        rhs.m22.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m12, rhs.m12.m13),
                                        self.m11.m22,
                                        rhs.m12.m23,
                                    ),
                                    self.m11.m32,
                                    rhs.m12.m33,
                                ),
                                self.m21.m12,
                                rhs.m22.m13,
                            ),
                            self.m21.m22,
                            rhs.m22.m23,
                        ),
                        self.m21.m32,
                        rhs.m22.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m11.m13, rhs.m12.m13),
                                        self.m11.m23,
                                        rhs.m12.m23,
                                    ),
                                    self.m11.m33,
                                    rhs.m12.m33,
                                ),
                                self.m21.m13,
                                rhs.m22.m13,
                            ),
                            self.m21.m23,
                            rhs.m22.m23,
                        ),
                        self.m21.m33,
                        rhs.m22.m33,
                    ),
                ),
            },
            m22: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, rhs.m12.m11),
                                        self.m12.m21,
                                        rhs.m12.m21,
                                    ),
                                    self.m12.m31,
                                    rhs.m12.m31,
                                ),
                                self.m22.m11,
                                rhs.m22.m11,
                            ),
                            self.m22.m21,
                            rhs.m22.m21,
                        ),
                        self.m22.m31,
                        rhs.m22.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, rhs.m12.m11),
                                        self.m12.m22,
                                        rhs.m12.m21,
                                    ),
                                    self.m12.m32,
                                    rhs.m12.m31,
                                ),
                                self.m22.m12,
                                rhs.m22.m11,
                            ),
                            self.m22.m22,
                            rhs.m22.m21,
                        ),
                        self.m22.m32,
                        rhs.m22.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, rhs.m12.m11),
                                        self.m12.m23,
                                        rhs.m12.m21,
                                    ),
                                    self.m12.m33,
                                    rhs.m12.m31,
                                ),
                                self.m22.m13,
                                rhs.m22.m11,
                            ),
                            self.m22.m23,
                            rhs.m22.m21,
                        ),
                        self.m22.m33,
                        rhs.m22.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, rhs.m12.m12),
                                        self.m12.m21,
                                        rhs.m12.m22,
                                    ),
                                    self.m12.m31,
                                    rhs.m12.m32,
                                ),
                                self.m22.m11,
                                rhs.m22.m12,
                            ),
                            self.m22.m21,
                            rhs.m22.m22,
                        ),
                        self.m22.m31,
                        rhs.m22.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, rhs.m12.m12),
                                        self.m12.m22,
                                        rhs.m12.m22,
                                    ),
                                    self.m12.m32,
                                    rhs.m12.m32,
                                ),
                                self.m22.m12,
                                rhs.m22.m12,
                            ),
                            self.m22.m22,
                            rhs.m22.m22,
                        ),
                        self.m22.m32,
                        rhs.m22.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, rhs.m12.m12),
                                        self.m12.m23,
                                        rhs.m12.m22,
                                    ),
                                    self.m12.m33,
                                    rhs.m12.m32,
                                ),
                                self.m22.m13,
                                rhs.m22.m12,
                            ),
                            self.m22.m23,
                            rhs.m22.m22,
                        ),
                        self.m22.m33,
                        rhs.m22.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m11, rhs.m12.m13),
                                        self.m12.m21,
                                        rhs.m12.m23,
                                    ),
                                    self.m12.m31,
                                    rhs.m12.m33,
                                ),
                                self.m22.m11,
                                rhs.m22.m13,
                            ),
                            self.m22.m21,
                            rhs.m22.m23,
                        ),
                        self.m22.m31,
                        rhs.m22.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m12, rhs.m12.m13),
                                        self.m12.m22,
                                        rhs.m12.m23,
                                    ),
                                    self.m12.m32,
                                    rhs.m12.m33,
                                ),
                                self.m22.m12,
                                rhs.m22.m13,
                            ),
                            self.m22.m22,
                            rhs.m22.m23,
                        ),
                        self.m22.m32,
                        rhs.m22.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), self.m12.m13, rhs.m12.m13),
                                        self.m12.m23,
                                        rhs.m12.m23,
                                    ),
                                    self.m12.m33,
                                    rhs.m12.m33,
                                ),
                                self.m22.m13,
                                rhs.m22.m13,
                            ),
                            self.m22.m23,
                            rhs.m22.m23,
                        ),
                        self.m22.m33,
                        rhs.m22.m33,
                    ),
                ),
            },
        }
    }

    // --- approximate equality ------------------------------------------------------------------

    /// Whether every component is within `ulps` smallest units of the identity's.
    /// Upstream: `is_identity(eps)`, with the tolerance in raw units instead of a float epsilon.
    fn is_identity(self: Matrix6<T>, ulps: u64) -> bool {
        Matrix3Trait::is_identity(self.m11, ulps)
            && Matrix3Trait::abs_diff_eq(self.m21, Matrix3Trait::zeros(), ulps)
            && Matrix3Trait::abs_diff_eq(self.m12, Matrix3Trait::zeros(), ulps)
            && Matrix3Trait::is_identity(self.m22, ulps)
    }

    /// Whether every component of `self` is within `ulps` smallest units of `other`'s.
    /// Upstream: `abs_diff_eq`, with the tolerance in raw units instead of a float epsilon.
    fn abs_diff_eq(self: Matrix6<T>, other: Matrix6<T>, ulps: u64) -> bool {
        Matrix3Trait::abs_diff_eq(self.m11, other.m11, ulps)
            && Matrix3Trait::abs_diff_eq(self.m21, other.m21, ulps)
            && Matrix3Trait::abs_diff_eq(self.m12, other.m12, ulps)
            && Matrix3Trait::abs_diff_eq(self.m22, other.m22, ulps)
    }
}

// --- operators -----------------------------------------------------------------------------------

/// `a + b`, component-wise (block-wise). Exact; panics on overflow.
pub impl Matrix6Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Matrix6<T>> {
    #[inline(always)]
    fn add(lhs: Matrix6<T>, rhs: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: lhs.m11 + rhs.m11,
            m21: lhs.m21 + rhs.m21,
            m12: lhs.m12 + rhs.m12,
            m22: lhs.m22 + rhs.m22,
        }
    }
}

/// `a - b`, component-wise (block-wise). Exact; panics on overflow.
pub impl Matrix6Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Matrix6<T>> {
    #[inline(always)]
    fn sub(lhs: Matrix6<T>, rhs: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: lhs.m11 - rhs.m11,
            m21: lhs.m21 - rhs.m21,
            m12: lhs.m12 - rhs.m12,
            m22: lhs.m22 - rhs.m22,
        }
    }
}

/// `-a`, component-wise (block-wise). Exact; panics on the scalar's `MIN`.
pub impl Matrix6Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Matrix6<T>> {
    #[inline(always)]
    fn neg(a: Matrix6<T>) -> Matrix6<T> {
        Matrix6 { m11: -a.m11, m21: -a.m21, m12: -a.m12, m22: -a.m22 }
    }
}

/// `a * b` (matrix product): 36 six-term `Real::Wide` accumulations, ONE rounding and one overflow
/// check per output scalar. Panics on overflow.
///
/// Deliberately NOT the block composition `m11 * n11 + m12 * n21`, which rounds each 3x3 product
/// before adding (two roundings per output scalar, a different result, and 2.00x the gas: 229 850
/// against 114 660 net, 72 fused kernels plus 36 additions instead of 36); that candidate is kept
/// as `bench_matrix6_mul__alt_blocks`.
pub impl Matrix6Mul<T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>> of Mul<Matrix6<T>> {
    fn mul(lhs: Matrix6<T>, rhs: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m11, rhs.m11.m11),
                                        lhs.m11.m12,
                                        rhs.m11.m21,
                                    ),
                                    lhs.m11.m13,
                                    rhs.m11.m31,
                                ),
                                lhs.m12.m11,
                                rhs.m21.m11,
                            ),
                            lhs.m12.m12,
                            rhs.m21.m21,
                        ),
                        lhs.m12.m13,
                        rhs.m21.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m21, rhs.m11.m11),
                                        lhs.m11.m22,
                                        rhs.m11.m21,
                                    ),
                                    lhs.m11.m23,
                                    rhs.m11.m31,
                                ),
                                lhs.m12.m21,
                                rhs.m21.m11,
                            ),
                            lhs.m12.m22,
                            rhs.m21.m21,
                        ),
                        lhs.m12.m23,
                        rhs.m21.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m31, rhs.m11.m11),
                                        lhs.m11.m32,
                                        rhs.m11.m21,
                                    ),
                                    lhs.m11.m33,
                                    rhs.m11.m31,
                                ),
                                lhs.m12.m31,
                                rhs.m21.m11,
                            ),
                            lhs.m12.m32,
                            rhs.m21.m21,
                        ),
                        lhs.m12.m33,
                        rhs.m21.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m11, rhs.m11.m12),
                                        lhs.m11.m12,
                                        rhs.m11.m22,
                                    ),
                                    lhs.m11.m13,
                                    rhs.m11.m32,
                                ),
                                lhs.m12.m11,
                                rhs.m21.m12,
                            ),
                            lhs.m12.m12,
                            rhs.m21.m22,
                        ),
                        lhs.m12.m13,
                        rhs.m21.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m21, rhs.m11.m12),
                                        lhs.m11.m22,
                                        rhs.m11.m22,
                                    ),
                                    lhs.m11.m23,
                                    rhs.m11.m32,
                                ),
                                lhs.m12.m21,
                                rhs.m21.m12,
                            ),
                            lhs.m12.m22,
                            rhs.m21.m22,
                        ),
                        lhs.m12.m23,
                        rhs.m21.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m31, rhs.m11.m12),
                                        lhs.m11.m32,
                                        rhs.m11.m22,
                                    ),
                                    lhs.m11.m33,
                                    rhs.m11.m32,
                                ),
                                lhs.m12.m31,
                                rhs.m21.m12,
                            ),
                            lhs.m12.m32,
                            rhs.m21.m22,
                        ),
                        lhs.m12.m33,
                        rhs.m21.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m11, rhs.m11.m13),
                                        lhs.m11.m12,
                                        rhs.m11.m23,
                                    ),
                                    lhs.m11.m13,
                                    rhs.m11.m33,
                                ),
                                lhs.m12.m11,
                                rhs.m21.m13,
                            ),
                            lhs.m12.m12,
                            rhs.m21.m23,
                        ),
                        lhs.m12.m13,
                        rhs.m21.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m21, rhs.m11.m13),
                                        lhs.m11.m22,
                                        rhs.m11.m23,
                                    ),
                                    lhs.m11.m23,
                                    rhs.m11.m33,
                                ),
                                lhs.m12.m21,
                                rhs.m21.m13,
                            ),
                            lhs.m12.m22,
                            rhs.m21.m23,
                        ),
                        lhs.m12.m23,
                        rhs.m21.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m31, rhs.m11.m13),
                                        lhs.m11.m32,
                                        rhs.m11.m23,
                                    ),
                                    lhs.m11.m33,
                                    rhs.m11.m33,
                                ),
                                lhs.m12.m31,
                                rhs.m21.m13,
                            ),
                            lhs.m12.m32,
                            rhs.m21.m23,
                        ),
                        lhs.m12.m33,
                        rhs.m21.m33,
                    ),
                ),
            },
            m21: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m11, rhs.m11.m11),
                                        lhs.m21.m12,
                                        rhs.m11.m21,
                                    ),
                                    lhs.m21.m13,
                                    rhs.m11.m31,
                                ),
                                lhs.m22.m11,
                                rhs.m21.m11,
                            ),
                            lhs.m22.m12,
                            rhs.m21.m21,
                        ),
                        lhs.m22.m13,
                        rhs.m21.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m21, rhs.m11.m11),
                                        lhs.m21.m22,
                                        rhs.m11.m21,
                                    ),
                                    lhs.m21.m23,
                                    rhs.m11.m31,
                                ),
                                lhs.m22.m21,
                                rhs.m21.m11,
                            ),
                            lhs.m22.m22,
                            rhs.m21.m21,
                        ),
                        lhs.m22.m23,
                        rhs.m21.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m31, rhs.m11.m11),
                                        lhs.m21.m32,
                                        rhs.m11.m21,
                                    ),
                                    lhs.m21.m33,
                                    rhs.m11.m31,
                                ),
                                lhs.m22.m31,
                                rhs.m21.m11,
                            ),
                            lhs.m22.m32,
                            rhs.m21.m21,
                        ),
                        lhs.m22.m33,
                        rhs.m21.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m11, rhs.m11.m12),
                                        lhs.m21.m12,
                                        rhs.m11.m22,
                                    ),
                                    lhs.m21.m13,
                                    rhs.m11.m32,
                                ),
                                lhs.m22.m11,
                                rhs.m21.m12,
                            ),
                            lhs.m22.m12,
                            rhs.m21.m22,
                        ),
                        lhs.m22.m13,
                        rhs.m21.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m21, rhs.m11.m12),
                                        lhs.m21.m22,
                                        rhs.m11.m22,
                                    ),
                                    lhs.m21.m23,
                                    rhs.m11.m32,
                                ),
                                lhs.m22.m21,
                                rhs.m21.m12,
                            ),
                            lhs.m22.m22,
                            rhs.m21.m22,
                        ),
                        lhs.m22.m23,
                        rhs.m21.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m31, rhs.m11.m12),
                                        lhs.m21.m32,
                                        rhs.m11.m22,
                                    ),
                                    lhs.m21.m33,
                                    rhs.m11.m32,
                                ),
                                lhs.m22.m31,
                                rhs.m21.m12,
                            ),
                            lhs.m22.m32,
                            rhs.m21.m22,
                        ),
                        lhs.m22.m33,
                        rhs.m21.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m11, rhs.m11.m13),
                                        lhs.m21.m12,
                                        rhs.m11.m23,
                                    ),
                                    lhs.m21.m13,
                                    rhs.m11.m33,
                                ),
                                lhs.m22.m11,
                                rhs.m21.m13,
                            ),
                            lhs.m22.m12,
                            rhs.m21.m23,
                        ),
                        lhs.m22.m13,
                        rhs.m21.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m21, rhs.m11.m13),
                                        lhs.m21.m22,
                                        rhs.m11.m23,
                                    ),
                                    lhs.m21.m23,
                                    rhs.m11.m33,
                                ),
                                lhs.m22.m21,
                                rhs.m21.m13,
                            ),
                            lhs.m22.m22,
                            rhs.m21.m23,
                        ),
                        lhs.m22.m23,
                        rhs.m21.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m31, rhs.m11.m13),
                                        lhs.m21.m32,
                                        rhs.m11.m23,
                                    ),
                                    lhs.m21.m33,
                                    rhs.m11.m33,
                                ),
                                lhs.m22.m31,
                                rhs.m21.m13,
                            ),
                            lhs.m22.m32,
                            rhs.m21.m23,
                        ),
                        lhs.m22.m33,
                        rhs.m21.m33,
                    ),
                ),
            },
            m12: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m11, rhs.m12.m11),
                                        lhs.m11.m12,
                                        rhs.m12.m21,
                                    ),
                                    lhs.m11.m13,
                                    rhs.m12.m31,
                                ),
                                lhs.m12.m11,
                                rhs.m22.m11,
                            ),
                            lhs.m12.m12,
                            rhs.m22.m21,
                        ),
                        lhs.m12.m13,
                        rhs.m22.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m21, rhs.m12.m11),
                                        lhs.m11.m22,
                                        rhs.m12.m21,
                                    ),
                                    lhs.m11.m23,
                                    rhs.m12.m31,
                                ),
                                lhs.m12.m21,
                                rhs.m22.m11,
                            ),
                            lhs.m12.m22,
                            rhs.m22.m21,
                        ),
                        lhs.m12.m23,
                        rhs.m22.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m31, rhs.m12.m11),
                                        lhs.m11.m32,
                                        rhs.m12.m21,
                                    ),
                                    lhs.m11.m33,
                                    rhs.m12.m31,
                                ),
                                lhs.m12.m31,
                                rhs.m22.m11,
                            ),
                            lhs.m12.m32,
                            rhs.m22.m21,
                        ),
                        lhs.m12.m33,
                        rhs.m22.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m11, rhs.m12.m12),
                                        lhs.m11.m12,
                                        rhs.m12.m22,
                                    ),
                                    lhs.m11.m13,
                                    rhs.m12.m32,
                                ),
                                lhs.m12.m11,
                                rhs.m22.m12,
                            ),
                            lhs.m12.m12,
                            rhs.m22.m22,
                        ),
                        lhs.m12.m13,
                        rhs.m22.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m21, rhs.m12.m12),
                                        lhs.m11.m22,
                                        rhs.m12.m22,
                                    ),
                                    lhs.m11.m23,
                                    rhs.m12.m32,
                                ),
                                lhs.m12.m21,
                                rhs.m22.m12,
                            ),
                            lhs.m12.m22,
                            rhs.m22.m22,
                        ),
                        lhs.m12.m23,
                        rhs.m22.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m31, rhs.m12.m12),
                                        lhs.m11.m32,
                                        rhs.m12.m22,
                                    ),
                                    lhs.m11.m33,
                                    rhs.m12.m32,
                                ),
                                lhs.m12.m31,
                                rhs.m22.m12,
                            ),
                            lhs.m12.m32,
                            rhs.m22.m22,
                        ),
                        lhs.m12.m33,
                        rhs.m22.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m11, rhs.m12.m13),
                                        lhs.m11.m12,
                                        rhs.m12.m23,
                                    ),
                                    lhs.m11.m13,
                                    rhs.m12.m33,
                                ),
                                lhs.m12.m11,
                                rhs.m22.m13,
                            ),
                            lhs.m12.m12,
                            rhs.m22.m23,
                        ),
                        lhs.m12.m13,
                        rhs.m22.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m21, rhs.m12.m13),
                                        lhs.m11.m22,
                                        rhs.m12.m23,
                                    ),
                                    lhs.m11.m23,
                                    rhs.m12.m33,
                                ),
                                lhs.m12.m21,
                                rhs.m22.m13,
                            ),
                            lhs.m12.m22,
                            rhs.m22.m23,
                        ),
                        lhs.m12.m23,
                        rhs.m22.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m11.m31, rhs.m12.m13),
                                        lhs.m11.m32,
                                        rhs.m12.m23,
                                    ),
                                    lhs.m11.m33,
                                    rhs.m12.m33,
                                ),
                                lhs.m12.m31,
                                rhs.m22.m13,
                            ),
                            lhs.m12.m32,
                            rhs.m22.m23,
                        ),
                        lhs.m12.m33,
                        rhs.m22.m33,
                    ),
                ),
            },
            m22: Matrix3 {
                m11: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m11, rhs.m12.m11),
                                        lhs.m21.m12,
                                        rhs.m12.m21,
                                    ),
                                    lhs.m21.m13,
                                    rhs.m12.m31,
                                ),
                                lhs.m22.m11,
                                rhs.m22.m11,
                            ),
                            lhs.m22.m12,
                            rhs.m22.m21,
                        ),
                        lhs.m22.m13,
                        rhs.m22.m31,
                    ),
                ),
                m21: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m21, rhs.m12.m11),
                                        lhs.m21.m22,
                                        rhs.m12.m21,
                                    ),
                                    lhs.m21.m23,
                                    rhs.m12.m31,
                                ),
                                lhs.m22.m21,
                                rhs.m22.m11,
                            ),
                            lhs.m22.m22,
                            rhs.m22.m21,
                        ),
                        lhs.m22.m23,
                        rhs.m22.m31,
                    ),
                ),
                m31: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m31, rhs.m12.m11),
                                        lhs.m21.m32,
                                        rhs.m12.m21,
                                    ),
                                    lhs.m21.m33,
                                    rhs.m12.m31,
                                ),
                                lhs.m22.m31,
                                rhs.m22.m11,
                            ),
                            lhs.m22.m32,
                            rhs.m22.m21,
                        ),
                        lhs.m22.m33,
                        rhs.m22.m31,
                    ),
                ),
                m12: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m11, rhs.m12.m12),
                                        lhs.m21.m12,
                                        rhs.m12.m22,
                                    ),
                                    lhs.m21.m13,
                                    rhs.m12.m32,
                                ),
                                lhs.m22.m11,
                                rhs.m22.m12,
                            ),
                            lhs.m22.m12,
                            rhs.m22.m22,
                        ),
                        lhs.m22.m13,
                        rhs.m22.m32,
                    ),
                ),
                m22: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m21, rhs.m12.m12),
                                        lhs.m21.m22,
                                        rhs.m12.m22,
                                    ),
                                    lhs.m21.m23,
                                    rhs.m12.m32,
                                ),
                                lhs.m22.m21,
                                rhs.m22.m12,
                            ),
                            lhs.m22.m22,
                            rhs.m22.m22,
                        ),
                        lhs.m22.m23,
                        rhs.m22.m32,
                    ),
                ),
                m32: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m31, rhs.m12.m12),
                                        lhs.m21.m32,
                                        rhs.m12.m22,
                                    ),
                                    lhs.m21.m33,
                                    rhs.m12.m32,
                                ),
                                lhs.m22.m31,
                                rhs.m22.m12,
                            ),
                            lhs.m22.m32,
                            rhs.m22.m22,
                        ),
                        lhs.m22.m33,
                        rhs.m22.m32,
                    ),
                ),
                m13: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m11, rhs.m12.m13),
                                        lhs.m21.m12,
                                        rhs.m12.m23,
                                    ),
                                    lhs.m21.m13,
                                    rhs.m12.m33,
                                ),
                                lhs.m22.m11,
                                rhs.m22.m13,
                            ),
                            lhs.m22.m12,
                            rhs.m22.m23,
                        ),
                        lhs.m22.m13,
                        rhs.m22.m33,
                    ),
                ),
                m23: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m21, rhs.m12.m13),
                                        lhs.m21.m22,
                                        rhs.m12.m23,
                                    ),
                                    lhs.m21.m23,
                                    rhs.m12.m33,
                                ),
                                lhs.m22.m21,
                                rhs.m22.m13,
                            ),
                            lhs.m22.m22,
                            rhs.m22.m23,
                        ),
                        lhs.m22.m23,
                        rhs.m22.m33,
                    ),
                ),
                m33: R::wide_rescale(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(
                                R::wide_add_prod(
                                    R::wide_add_prod(
                                        R::wide_add_prod(R::wide_zero(), lhs.m21.m31, rhs.m12.m13),
                                        lhs.m21.m32,
                                        rhs.m12.m23,
                                    ),
                                    lhs.m21.m33,
                                    rhs.m12.m33,
                                ),
                                lhs.m22.m31,
                                rhs.m22.m13,
                            ),
                            lhs.m22.m32,
                            rhs.m22.m23,
                        ),
                        lhs.m22.m33,
                        rhs.m22.m33,
                    ),
                ),
            },
        }
    }
}

/// `a += b`.
pub impl Matrix6AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Matrix6<T>, Matrix6<T>> {
    #[inline(always)]
    fn add_assign(ref self: Matrix6<T>, rhs: Matrix6<T>) {
        self = self + rhs;
    }
}

/// `a -= b`.
pub impl Matrix6SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Matrix6<T>, Matrix6<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Matrix6<T>, rhs: Matrix6<T>) {
        self = self - rhs;
    }
}

/// `a *= b` (matrix product, `a = a * b`).
pub impl Matrix6MulAssign<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of MulAssign<Matrix6<T>, Matrix6<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Matrix6<T>, rhs: Matrix6<T>) {
        self = self * rhs;
    }
}
