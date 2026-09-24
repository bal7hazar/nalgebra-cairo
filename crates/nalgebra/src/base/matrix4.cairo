//! `Matrix4`: a statically sized 4x4 matrix (upstream `nalgebra::Matrix4`).
//!
//! Every sum of products goes through a fused `Real` kernel: one rounding (floor) and one overflow
//! check per output scalar. Operators (`+`, `-`, unary `-`, `*` between matrices and their
//! assigning forms) are implemented in this module, so they need no import; the other operations
//! are methods of `Matrix4Trait`.

use core::ops::{AddAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::vector4::Vector4;

/// A 4x4 matrix. `mRC` is the component at row `R`, column `C`.
///
/// Fields are declared in column-major order, so `Serde` matches upstream's storage order, while
/// `new` takes its arguments in row-major order like upstream.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix4<T> {
    pub m11: T,
    pub m21: T,
    pub m31: T,
    pub m41: T,
    pub m12: T,
    pub m22: T,
    pub m32: T,
    pub m42: T,
    pub m13: T,
    pub m23: T,
    pub m33: T,
    pub m43: T,
    pub m14: T,
    pub m24: T,
    pub m34: T,
    pub m44: T,
}

/// Internal kernels of `Matrix4`.
#[generate_trait]
impl Matrix4Kernels<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Matrix4KernelsTrait<T> {
    /// `a*b - c*d + e*f`, accumulated exactly and rounded once.
    #[inline(always)]
    fn pmp(a: T, b: T, c: T, d: T, e: T, f: T) -> T {
        let w = R::wide_add_prod(R::wide_zero(), a, b);
        R::wide_rescale(R::wide_add_prod(R::wide_sub_prod(w, c, d), e, f))
    }

    /// `-a*b + c*d - e*f`, accumulated exactly and rounded once.
    #[inline(always)]
    fn mpm(a: T, b: T, c: T, d: T, e: T, f: T) -> T {
        let w = R::wide_sub_prod(R::wide_zero(), a, b);
        R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(w, c, d), e, f))
    }

    /// `(adjugate, determinant)` from the twelve 2x2 minors of the two upper rows (`s0..s5`)
    /// and of the two lower rows (`c0..c5`): 12 `diff_prod`, then 16 three-term and one six-term
    /// exact accumulations (two roundings per output).
    #[inline(always)]
    fn adjugate_determinant(m: Matrix4<T>) -> (Matrix4<T>, T) {
        let s0 = R::diff_prod(m.m11, m.m22, m.m21, m.m12);
        let s1 = R::diff_prod(m.m11, m.m23, m.m21, m.m13);
        let s2 = R::diff_prod(m.m11, m.m24, m.m21, m.m14);
        let s3 = R::diff_prod(m.m12, m.m23, m.m22, m.m13);
        let s4 = R::diff_prod(m.m12, m.m24, m.m22, m.m14);
        let s5 = R::diff_prod(m.m13, m.m24, m.m23, m.m14);
        let c5 = R::diff_prod(m.m33, m.m44, m.m43, m.m34);
        let c4 = R::diff_prod(m.m32, m.m44, m.m42, m.m34);
        let c3 = R::diff_prod(m.m32, m.m43, m.m42, m.m33);
        let c2 = R::diff_prod(m.m31, m.m44, m.m41, m.m34);
        let c1 = R::diff_prod(m.m31, m.m43, m.m41, m.m33);
        let c0 = R::diff_prod(m.m31, m.m42, m.m41, m.m32);
        let w = R::wide_add_prod(R::wide_zero(), s0, c5);
        let w = R::wide_sub_prod(w, s1, c4);
        let w = R::wide_add_prod(w, s2, c3);
        let w = R::wide_add_prod(w, s3, c2);
        let w = R::wide_sub_prod(w, s4, c1);
        let det = R::wide_rescale(R::wide_add_prod(w, s5, c0));
        let adj = Matrix4 {
            m11: Self::pmp(m.m22, c5, m.m23, c4, m.m24, c3),
            m21: Self::mpm(m.m21, c5, m.m23, c2, m.m24, c1),
            m31: Self::pmp(m.m21, c4, m.m22, c2, m.m24, c0),
            m41: Self::mpm(m.m21, c3, m.m22, c1, m.m23, c0),
            m12: Self::mpm(m.m12, c5, m.m13, c4, m.m14, c3),
            m22: Self::pmp(m.m11, c5, m.m13, c2, m.m14, c1),
            m32: Self::mpm(m.m11, c4, m.m12, c2, m.m14, c0),
            m42: Self::pmp(m.m11, c3, m.m12, c1, m.m13, c0),
            m13: Self::pmp(m.m42, s5, m.m43, s4, m.m44, s3),
            m23: Self::mpm(m.m41, s5, m.m43, s2, m.m44, s1),
            m33: Self::pmp(m.m41, s4, m.m42, s2, m.m44, s0),
            m43: Self::mpm(m.m41, s3, m.m42, s1, m.m43, s0),
            m14: Self::mpm(m.m32, s5, m.m33, s4, m.m34, s3),
            m24: Self::pmp(m.m31, s5, m.m33, s2, m.m34, s1),
            m34: Self::mpm(m.m31, s4, m.m32, s2, m.m34, s0),
            m44: Self::pmp(m.m31, s3, m.m32, s1, m.m33, s0),
        };
        (adj, det)
    }
}

/// Methods of `Matrix4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Matrix4Impl<
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
> of Matrix4Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// The matrix with the given components, in ROW-major order. Upstream: `Matrix4::new`.
    #[inline(always)]
    fn new(
        m11: T,
        m12: T,
        m13: T,
        m14: T,
        m21: T,
        m22: T,
        m23: T,
        m24: T,
        m31: T,
        m32: T,
        m33: T,
        m34: T,
        m41: T,
        m42: T,
        m43: T,
        m44: T,
    ) -> Matrix4<T> {
        Matrix4 { m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44 }
    }

    /// The zero matrix. Upstream: `Matrix4::zeros`.
    #[inline(always)]
    fn zeros() -> Matrix4<T> {
        Matrix4 {
            m11: R::ZERO,
            m21: R::ZERO,
            m31: R::ZERO,
            m41: R::ZERO,
            m12: R::ZERO,
            m22: R::ZERO,
            m32: R::ZERO,
            m42: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ZERO,
            m43: R::ZERO,
            m14: R::ZERO,
            m24: R::ZERO,
            m34: R::ZERO,
            m44: R::ZERO,
        }
    }

    /// The identity matrix. Upstream: `Matrix4::identity`.
    #[inline(always)]
    fn identity() -> Matrix4<T> {
        Matrix4 {
            m11: R::ONE,
            m21: R::ZERO,
            m31: R::ZERO,
            m41: R::ZERO,
            m12: R::ZERO,
            m22: R::ONE,
            m32: R::ZERO,
            m42: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ONE,
            m43: R::ZERO,
            m14: R::ZERO,
            m24: R::ZERO,
            m34: R::ZERO,
            m44: R::ONE,
        }
    }

    /// The diagonal matrix `diag(d)`. Upstream: `Matrix4::from_diagonal`.
    #[inline(always)]
    fn from_diagonal(d: Vector4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: d.x,
            m21: R::ZERO,
            m31: R::ZERO,
            m41: R::ZERO,
            m12: R::ZERO,
            m22: d.y,
            m32: R::ZERO,
            m42: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: d.z,
            m43: R::ZERO,
            m14: R::ZERO,
            m24: R::ZERO,
            m34: R::ZERO,
            m44: d.w,
        }
    }

    /// The matrix `e * I`. Upstream: `Matrix4::from_diagonal_element`.
    #[inline(always)]
    fn from_diagonal_element(e: T) -> Matrix4<T> {
        Matrix4 {
            m11: e,
            m21: R::ZERO,
            m31: R::ZERO,
            m41: R::ZERO,
            m12: R::ZERO,
            m22: e,
            m32: R::ZERO,
            m42: R::ZERO,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: e,
            m43: R::ZERO,
            m14: R::ZERO,
            m24: R::ZERO,
            m34: R::ZERO,
            m44: e,
        }
    }

    /// The matrix whose columns are the given vectors. Upstream: `Matrix4::from_columns`.
    #[inline(always)]
    fn from_columns(c1: Vector4<T>, c2: Vector4<T>, c3: Vector4<T>, c4: Vector4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: c1.x,
            m21: c1.y,
            m31: c1.z,
            m41: c1.w,
            m12: c2.x,
            m22: c2.y,
            m32: c2.z,
            m42: c2.w,
            m13: c3.x,
            m23: c3.y,
            m33: c3.z,
            m43: c3.w,
            m14: c4.x,
            m24: c4.y,
            m34: c4.z,
            m44: c4.w,
        }
    }

    /// The matrix whose rows are the given vectors. Upstream: `Matrix4::from_rows`.
    #[inline(always)]
    fn from_rows(r1: Vector4<T>, r2: Vector4<T>, r3: Vector4<T>, r4: Vector4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: r1.x,
            m21: r2.x,
            m31: r3.x,
            m41: r4.x,
            m12: r1.y,
            m22: r2.y,
            m32: r3.y,
            m42: r4.y,
            m13: r1.z,
            m23: r2.z,
            m33: r3.z,
            m43: r4.z,
            m14: r1.w,
            m24: r2.w,
            m34: r3.w,
            m44: r4.w,
        }
    }

    // --- accessors -----------------------------------------------------------------------------

    /// The diagonal. Upstream: `diagonal`.
    #[inline(always)]
    fn diagonal(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m11, y: self.m22, z: self.m33, w: self.m44 }
    }

    // --- exact operations ----------------------------------------------------------------------

    /// The transpose. Exact. Upstream: `transpose`.
    #[inline(always)]
    fn transpose(self: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: self.m11,
            m21: self.m12,
            m31: self.m13,
            m41: self.m14,
            m12: self.m21,
            m22: self.m22,
            m32: self.m23,
            m42: self.m24,
            m13: self.m31,
            m23: self.m32,
            m33: self.m33,
            m43: self.m34,
            m14: self.m41,
            m24: self.m42,
            m34: self.m43,
            m44: self.m44,
        }
    }

    /// Sum of the diagonal. Exact; panics on overflow. Upstream: `trace`.
    #[inline(always)]
    fn trace(self: Matrix4<T>) -> T {
        self.m11 + self.m22 + self.m33 + self.m44
    }

    /// Component-wise absolute value. Panics on the scalar's `MIN`. Upstream: `abs`.
    #[inline(always)]
    fn abs(self: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::abs(self.m11),
            m21: R::abs(self.m21),
            m31: R::abs(self.m31),
            m41: R::abs(self.m41),
            m12: R::abs(self.m12),
            m22: R::abs(self.m22),
            m32: R::abs(self.m32),
            m42: R::abs(self.m42),
            m13: R::abs(self.m13),
            m23: R::abs(self.m23),
            m33: R::abs(self.m33),
            m43: R::abs(self.m43),
            m14: R::abs(self.m14),
            m24: R::abs(self.m24),
            m34: R::abs(self.m34),
            m44: R::abs(self.m44),
        }
    }

    // --- products ------------------------------------------------------------------------------

    /// `self * k`: each component is one floored product. Panics on overflow.
    /// Upstream: `self * k`.
    #[inline(always)]
    fn scale(self: Matrix4<T>, k: T) -> Matrix4<T> {
        Matrix4 {
            m11: self.m11 * k,
            m21: self.m21 * k,
            m31: self.m31 * k,
            m41: self.m41 * k,
            m12: self.m12 * k,
            m22: self.m22 * k,
            m32: self.m32 * k,
            m42: self.m42 * k,
            m13: self.m13 * k,
            m23: self.m23 * k,
            m33: self.m33 * k,
            m43: self.m43 * k,
            m14: self.m14 * k,
            m24: self.m24 * k,
            m34: self.m34 * k,
            m44: self.m44 * k,
        }
    }

    /// Component-wise (Hadamard) product: each component is one floored product. Panics on
    /// overflow. Upstream: `component_mul`.
    #[inline(always)]
    fn component_mul(self: Matrix4<T>, rhs: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: self.m11 * rhs.m11,
            m21: self.m21 * rhs.m21,
            m31: self.m31 * rhs.m31,
            m41: self.m41 * rhs.m41,
            m12: self.m12 * rhs.m12,
            m22: self.m22 * rhs.m22,
            m32: self.m32 * rhs.m32,
            m42: self.m42 * rhs.m42,
            m13: self.m13 * rhs.m13,
            m23: self.m23 * rhs.m23,
            m33: self.m33 * rhs.m33,
            m43: self.m43 * rhs.m43,
            m14: self.m14 * rhs.m14,
            m24: self.m24 * rhs.m24,
            m34: self.m34 * rhs.m34,
            m44: self.m44 * rhs.m44,
        }
    }

    /// `self * v`: one `sum_prod4` per component (one rounding each). Panics on overflow.
    /// Upstream: `self * v`.
    #[inline(always)]
    fn mul_vec(self: Matrix4<T>, v: Vector4<T>) -> Vector4<T> {
        Vector4 {
            x: R::sum_prod4(self.m11, v.x, self.m12, v.y, self.m13, v.z, self.m14, v.w),
            y: R::sum_prod4(self.m21, v.x, self.m22, v.y, self.m23, v.z, self.m24, v.w),
            z: R::sum_prod4(self.m31, v.x, self.m32, v.y, self.m33, v.z, self.m34, v.w),
            w: R::sum_prod4(self.m41, v.x, self.m42, v.y, self.m43, v.z, self.m44, v.w),
        }
    }

    /// `selfᵀ * v` without forming the transpose: one `sum_prod4` per component. Panics on
    /// overflow. Upstream: `self.tr_mul(&v)`.
    #[inline(always)]
    fn tr_mul_vec(self: Matrix4<T>, v: Vector4<T>) -> Vector4<T> {
        Vector4 {
            x: R::sum_prod4(self.m11, v.x, self.m21, v.y, self.m31, v.z, self.m41, v.w),
            y: R::sum_prod4(self.m12, v.x, self.m22, v.y, self.m32, v.z, self.m42, v.w),
            z: R::sum_prod4(self.m13, v.x, self.m23, v.y, self.m33, v.z, self.m43, v.w),
            w: R::sum_prod4(self.m14, v.x, self.m24, v.y, self.m34, v.z, self.m44, v.w),
        }
    }

    /// `selfᵀ * rhs` without forming the transpose: 16 `sum_prod4`. Panics on overflow.
    /// Upstream: `tr_mul`.
    fn tr_mul(self: Matrix4<T>, rhs: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::sum_prod4(
                self.m11, rhs.m11, self.m21, rhs.m21, self.m31, rhs.m31, self.m41, rhs.m41,
            ),
            m21: R::sum_prod4(
                self.m12, rhs.m11, self.m22, rhs.m21, self.m32, rhs.m31, self.m42, rhs.m41,
            ),
            m31: R::sum_prod4(
                self.m13, rhs.m11, self.m23, rhs.m21, self.m33, rhs.m31, self.m43, rhs.m41,
            ),
            m41: R::sum_prod4(
                self.m14, rhs.m11, self.m24, rhs.m21, self.m34, rhs.m31, self.m44, rhs.m41,
            ),
            m12: R::sum_prod4(
                self.m11, rhs.m12, self.m21, rhs.m22, self.m31, rhs.m32, self.m41, rhs.m42,
            ),
            m22: R::sum_prod4(
                self.m12, rhs.m12, self.m22, rhs.m22, self.m32, rhs.m32, self.m42, rhs.m42,
            ),
            m32: R::sum_prod4(
                self.m13, rhs.m12, self.m23, rhs.m22, self.m33, rhs.m32, self.m43, rhs.m42,
            ),
            m42: R::sum_prod4(
                self.m14, rhs.m12, self.m24, rhs.m22, self.m34, rhs.m32, self.m44, rhs.m42,
            ),
            m13: R::sum_prod4(
                self.m11, rhs.m13, self.m21, rhs.m23, self.m31, rhs.m33, self.m41, rhs.m43,
            ),
            m23: R::sum_prod4(
                self.m12, rhs.m13, self.m22, rhs.m23, self.m32, rhs.m33, self.m42, rhs.m43,
            ),
            m33: R::sum_prod4(
                self.m13, rhs.m13, self.m23, rhs.m23, self.m33, rhs.m33, self.m43, rhs.m43,
            ),
            m43: R::sum_prod4(
                self.m14, rhs.m13, self.m24, rhs.m23, self.m34, rhs.m33, self.m44, rhs.m43,
            ),
            m14: R::sum_prod4(
                self.m11, rhs.m14, self.m21, rhs.m24, self.m31, rhs.m34, self.m41, rhs.m44,
            ),
            m24: R::sum_prod4(
                self.m12, rhs.m14, self.m22, rhs.m24, self.m32, rhs.m34, self.m42, rhs.m44,
            ),
            m34: R::sum_prod4(
                self.m13, rhs.m14, self.m23, rhs.m24, self.m33, rhs.m34, self.m43, rhs.m44,
            ),
            m44: R::sum_prod4(
                self.m14, rhs.m14, self.m24, rhs.m24, self.m34, rhs.m34, self.m44, rhs.m44,
            ),
        }
    }

    // --- norms ---------------------------------------------------------------------------------

    /// Squared Frobenius norm: the 16 squares are accumulated exactly (wide accumulator) and
    /// rounded once. Panics on overflow. Upstream: `norm_squared`.
    #[inline(always)]
    fn norm_squared(self: Matrix4<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.m11, self.m11);
        let w = R::wide_add_prod(w, self.m21, self.m21);
        let w = R::wide_add_prod(w, self.m31, self.m31);
        let w = R::wide_add_prod(w, self.m41, self.m41);
        let w = R::wide_add_prod(w, self.m12, self.m12);
        let w = R::wide_add_prod(w, self.m22, self.m22);
        let w = R::wide_add_prod(w, self.m32, self.m32);
        let w = R::wide_add_prod(w, self.m42, self.m42);
        let w = R::wide_add_prod(w, self.m13, self.m13);
        let w = R::wide_add_prod(w, self.m23, self.m23);
        let w = R::wide_add_prod(w, self.m33, self.m33);
        let w = R::wide_add_prod(w, self.m43, self.m43);
        let w = R::wide_add_prod(w, self.m14, self.m14);
        let w = R::wide_add_prod(w, self.m24, self.m24);
        let w = R::wide_add_prod(w, self.m34, self.m34);
        R::wide_rescale(R::wide_add_prod(w, self.m44, self.m44))
    }

    /// Frobenius norm: square root of the exact, unscaled sum of squares (one rounding, no
    /// intermediate overflow: only the result must fit). Upstream: `norm`.
    #[inline(always)]
    fn norm(self: Matrix4<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.m11, self.m11);
        let w = R::wide_add_prod(w, self.m21, self.m21);
        let w = R::wide_add_prod(w, self.m31, self.m31);
        let w = R::wide_add_prod(w, self.m41, self.m41);
        let w = R::wide_add_prod(w, self.m12, self.m12);
        let w = R::wide_add_prod(w, self.m22, self.m22);
        let w = R::wide_add_prod(w, self.m32, self.m32);
        let w = R::wide_add_prod(w, self.m42, self.m42);
        let w = R::wide_add_prod(w, self.m13, self.m13);
        let w = R::wide_add_prod(w, self.m23, self.m23);
        let w = R::wide_add_prod(w, self.m33, self.m33);
        let w = R::wide_add_prod(w, self.m43, self.m43);
        let w = R::wide_add_prod(w, self.m14, self.m14);
        let w = R::wide_add_prod(w, self.m24, self.m24);
        let w = R::wide_add_prod(w, self.m34, self.m34);
        R::wide_sqrt(R::wide_add_prod(w, self.m44, self.m44))
    }

    // --- determinant and inverse ---------------------------------------------------------------

    /// The determinant, from the 2x2 minors of the two upper rows (`s0..s5`) and of the two lower
    /// rows (`c0..c5`): `s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0`, i.e. 12 `diff_prod` (one
    /// rounding each) and one exact six-term accumulation (second rounding). 25 % cheaper and one
    /// rounding less than the cofactor expansion along a row. Panics on overflow.
    /// Upstream: `determinant`.
    fn determinant(self: Matrix4<T>) -> T {
        let s0 = R::diff_prod(self.m11, self.m22, self.m21, self.m12);
        let s1 = R::diff_prod(self.m11, self.m23, self.m21, self.m13);
        let s2 = R::diff_prod(self.m11, self.m24, self.m21, self.m14);
        let s3 = R::diff_prod(self.m12, self.m23, self.m22, self.m13);
        let s4 = R::diff_prod(self.m12, self.m24, self.m22, self.m14);
        let s5 = R::diff_prod(self.m13, self.m24, self.m23, self.m14);
        let c5 = R::diff_prod(self.m33, self.m44, self.m43, self.m34);
        let c4 = R::diff_prod(self.m32, self.m44, self.m42, self.m34);
        let c3 = R::diff_prod(self.m32, self.m43, self.m42, self.m33);
        let c2 = R::diff_prod(self.m31, self.m44, self.m41, self.m34);
        let c1 = R::diff_prod(self.m31, self.m43, self.m41, self.m33);
        let c0 = R::diff_prod(self.m31, self.m42, self.m41, self.m32);
        let w = R::wide_add_prod(R::wide_zero(), s0, c5);
        let w = R::wide_sub_prod(w, s1, c4);
        let w = R::wide_add_prod(w, s2, c3);
        let w = R::wide_add_prod(w, s3, c2);
        let w = R::wide_sub_prod(w, s4, c1);
        R::wide_rescale(R::wide_add_prod(w, s5, c0))
    }

    /// The inverse, or `None` when the matrix is singular. Upstream: `try_inverse`.
    ///
    /// Singularity criterion, like upstream: the computed determinant is EXACTLY zero (no
    /// epsilon). The zero matrix is singular. A nearly singular matrix is inverted with the
    /// precision its conditioning allows, and panics with the scalar's overflow error when a
    /// component of the inverse does not fit.
    ///
    /// Algorithm: `adjugate / determinant` (see `determinant`), one division per component (a
    /// single reciprocal followed by 16 multiplications is cheaper but loses the low bits of
    /// `1 / det` when `|det| >> 1`). Because a fixed-point determinant of a small matrix has few
    /// significant bits, a matrix of Frobenius norm `f <= 1` is first multiplied by the INTEGER
    /// `k = floor(2 / f)` (exact products), which gives `inverse = adjugate(k * self) * (k /
    /// det(k * self))`. Matrices with `|det| >= 1/2` skip the norm computation (`f <= 1`
    /// implies `|det| <= 1/16`). Panics with the overflow error when `0 < f <= 2^-30`.
    ///
    /// Re-ranked on `fixed` 0.3.0 (WP 7.2): the unscaled branch divides through ONE prepared
    /// divisor (`Real::div16`, bit-identical to per-element division). `adjugate / det` without the
    /// pre-scaling costs 161 670 gas through `Real::div16` (`bench_matrix4_try_inverse__alt_div_n`)
    /// against 196 820, and `adjugate * (1 / det)` — upstream's 4x4 formula (MESA's inverse) —
    /// is cheaper still, but they leave respectively 6 and 10 of the 30 oracle cases outside their
    /// tolerance (`test_try_inverse_candidates_error`), so the pre-scaled algorithm stays. The
    /// charged gas is that of the costliest branch (the pre-scaled one): the three
    /// `bench_matrix4_try_inverse__prescaled_*` benchmarks measure the same figure.
    fn try_inverse(self: Matrix4<T>) -> Option<Matrix4<T>> {
        let (adj, det) = Matrix4Kernels::adjugate_determinant(self);
        if det < R::HALF && det > -R::HALF {
            let f = Self::norm(self);
            if f == R::ZERO {
                return None;
            }
            let k = R::floor(R::div(R::TWO, f));
            if k >= R::TWO {
                let (adj_b, det_b) = Matrix4Kernels::adjugate_determinant(Self::scale(self, k));
                if det_b == R::ZERO {
                    return None;
                }
                return Some(Self::scale(adj_b, R::div(k, det_b)));
            }
            if det == R::ZERO {
                return None;
            }
        }
        let (m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44) =
            R::div16(
            adj.m11,
            adj.m21,
            adj.m31,
            adj.m41,
            adj.m12,
            adj.m22,
            adj.m32,
            adj.m42,
            adj.m13,
            adj.m23,
            adj.m33,
            adj.m43,
            adj.m14,
            adj.m24,
            adj.m34,
            adj.m44,
            det,
        );
        Some(
            Matrix4 {
                m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44,
            },
        )
    }

    // --- approximate equality ------------------------------------------------------------------

    /// Whether every component is within `ulps` smallest units of the identity's.
    /// Upstream: `is_identity(eps)`, with the tolerance in raw units instead of a float epsilon.
    fn is_identity(self: Matrix4<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, R::ONE, ulps)
            && R::abs_diff_eq(self.m21, R::ZERO, ulps)
            && R::abs_diff_eq(self.m31, R::ZERO, ulps)
            && R::abs_diff_eq(self.m41, R::ZERO, ulps)
            && R::abs_diff_eq(self.m12, R::ZERO, ulps)
            && R::abs_diff_eq(self.m22, R::ONE, ulps)
            && R::abs_diff_eq(self.m32, R::ZERO, ulps)
            && R::abs_diff_eq(self.m42, R::ZERO, ulps)
            && R::abs_diff_eq(self.m13, R::ZERO, ulps)
            && R::abs_diff_eq(self.m23, R::ZERO, ulps)
            && R::abs_diff_eq(self.m33, R::ONE, ulps)
            && R::abs_diff_eq(self.m43, R::ZERO, ulps)
            && R::abs_diff_eq(self.m14, R::ZERO, ulps)
            && R::abs_diff_eq(self.m24, R::ZERO, ulps)
            && R::abs_diff_eq(self.m34, R::ZERO, ulps)
            && R::abs_diff_eq(self.m44, R::ONE, ulps)
    }

    /// Whether every component of `self` is within `ulps` smallest units of `other`'s.
    /// Upstream: `abs_diff_eq`, with the tolerance in raw units instead of a float epsilon.
    fn abs_diff_eq(self: Matrix4<T>, other: Matrix4<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, other.m11, ulps)
            && R::abs_diff_eq(self.m21, other.m21, ulps)
            && R::abs_diff_eq(self.m31, other.m31, ulps)
            && R::abs_diff_eq(self.m41, other.m41, ulps)
            && R::abs_diff_eq(self.m12, other.m12, ulps)
            && R::abs_diff_eq(self.m22, other.m22, ulps)
            && R::abs_diff_eq(self.m32, other.m32, ulps)
            && R::abs_diff_eq(self.m42, other.m42, ulps)
            && R::abs_diff_eq(self.m13, other.m13, ulps)
            && R::abs_diff_eq(self.m23, other.m23, ulps)
            && R::abs_diff_eq(self.m33, other.m33, ulps)
            && R::abs_diff_eq(self.m43, other.m43, ulps)
            && R::abs_diff_eq(self.m14, other.m14, ulps)
            && R::abs_diff_eq(self.m24, other.m24, ulps)
            && R::abs_diff_eq(self.m34, other.m34, ulps)
            && R::abs_diff_eq(self.m44, other.m44, ulps)
    }
}

/// Crate-internal kernels of `Matrix4<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `adjugate`) and the unrolled row / column accessors that stand for upstream `row(i)` /
/// `column(i)` views, used by the decompositions.
#[generate_trait]
pub(crate) impl Matrix4InternalImpl<
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
> of Matrix4InternalTrait<T> {
    /// Column 1. Upstream: `column(0)`.
    #[inline(always)]
    fn column1(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m11, y: self.m21, z: self.m31, w: self.m41 }
    }

    /// Column 2. Upstream: `column(1)`.
    #[inline(always)]
    fn column2(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m12, y: self.m22, z: self.m32, w: self.m42 }
    }

    /// Column 3. Upstream: `column(2)`.
    #[inline(always)]
    fn column3(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m13, y: self.m23, z: self.m33, w: self.m43 }
    }

    /// Column 4. Upstream: `column(3)`.
    #[inline(always)]
    fn column4(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m14, y: self.m24, z: self.m34, w: self.m44 }
    }

    /// Row 1, as a (column) vector. Upstream: `row(0).transpose()`.
    #[inline(always)]
    fn row1(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m11, y: self.m12, z: self.m13, w: self.m14 }
    }

    /// Row 2, as a (column) vector. Upstream: `row(1).transpose()`.
    #[inline(always)]
    fn row2(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m21, y: self.m22, z: self.m23, w: self.m24 }
    }

    /// Row 3, as a (column) vector. Upstream: `row(2).transpose()`.
    #[inline(always)]
    fn row3(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m31, y: self.m32, z: self.m33, w: self.m34 }
    }

    /// Row 4, as a (column) vector. Upstream: `row(3).transpose()`.
    #[inline(always)]
    fn row4(self: Matrix4<T>) -> Vector4<T> {
        Vector4 { x: self.m41, y: self.m42, z: self.m43, w: self.m44 }
    }

    /// The outer product `a * bᵀ`: each component is one floored product. Panics with the
    /// scalar's overflow error. Upstream: `a * b.transpose()`.
    #[inline(always)]
    fn from_outer(a: Vector4<T>, b: Vector4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: a.x * b.x,
            m21: a.y * b.x,
            m31: a.z * b.x,
            m41: a.w * b.x,
            m12: a.x * b.y,
            m22: a.y * b.y,
            m32: a.z * b.y,
            m42: a.w * b.y,
            m13: a.x * b.z,
            m23: a.y * b.z,
            m33: a.z * b.z,
            m43: a.w * b.z,
            m14: a.x * b.w,
            m24: a.y * b.w,
            m34: a.z * b.w,
            m44: a.w * b.w,
        }
    }
    /// The adjugate (transposed cofactor matrix): `self * adjugate = determinant * I`. 12
    /// `diff_prod` (2x2 minors) then 16 exact three-term accumulations: two roundings per
    /// component. Panics on overflow.
    /// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
    fn adjugate(self: Matrix4<T>) -> Matrix4<T> {
        let (adj, _) = Matrix4Kernels::adjugate_determinant(self);
        adj
    }
}

// --- operators -----------------------------------------------------------------------------------

/// `a + b`, component-wise. Exact; panics on overflow.
pub impl Matrix4Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Matrix4<T>> {
    #[inline(always)]
    fn add(lhs: Matrix4<T>, rhs: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: lhs.m11 + rhs.m11,
            m21: lhs.m21 + rhs.m21,
            m31: lhs.m31 + rhs.m31,
            m41: lhs.m41 + rhs.m41,
            m12: lhs.m12 + rhs.m12,
            m22: lhs.m22 + rhs.m22,
            m32: lhs.m32 + rhs.m32,
            m42: lhs.m42 + rhs.m42,
            m13: lhs.m13 + rhs.m13,
            m23: lhs.m23 + rhs.m23,
            m33: lhs.m33 + rhs.m33,
            m43: lhs.m43 + rhs.m43,
            m14: lhs.m14 + rhs.m14,
            m24: lhs.m24 + rhs.m24,
            m34: lhs.m34 + rhs.m34,
            m44: lhs.m44 + rhs.m44,
        }
    }
}

/// `a - b`, component-wise. Exact; panics on overflow.
pub impl Matrix4Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Matrix4<T>> {
    #[inline(always)]
    fn sub(lhs: Matrix4<T>, rhs: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: lhs.m11 - rhs.m11,
            m21: lhs.m21 - rhs.m21,
            m31: lhs.m31 - rhs.m31,
            m41: lhs.m41 - rhs.m41,
            m12: lhs.m12 - rhs.m12,
            m22: lhs.m22 - rhs.m22,
            m32: lhs.m32 - rhs.m32,
            m42: lhs.m42 - rhs.m42,
            m13: lhs.m13 - rhs.m13,
            m23: lhs.m23 - rhs.m23,
            m33: lhs.m33 - rhs.m33,
            m43: lhs.m43 - rhs.m43,
            m14: lhs.m14 - rhs.m14,
            m24: lhs.m24 - rhs.m24,
            m34: lhs.m34 - rhs.m34,
            m44: lhs.m44 - rhs.m44,
        }
    }
}

/// `-a`, component-wise. Exact; panics on the scalar's `MIN`.
pub impl Matrix4Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Matrix4<T>> {
    #[inline(always)]
    fn neg(a: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: -a.m11,
            m21: -a.m21,
            m31: -a.m31,
            m41: -a.m41,
            m12: -a.m12,
            m22: -a.m22,
            m32: -a.m32,
            m42: -a.m42,
            m13: -a.m13,
            m23: -a.m23,
            m33: -a.m33,
            m43: -a.m43,
            m14: -a.m14,
            m24: -a.m24,
            m34: -a.m34,
            m44: -a.m44,
        }
    }
}

/// `a * b` (matrix product): 16 `sum_prod4`, one rounding per component. Panics on overflow.
pub impl Matrix4Mul<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Mul<Matrix4<T>> {
    fn mul(lhs: Matrix4<T>, rhs: Matrix4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::sum_prod4(
                lhs.m11, rhs.m11, lhs.m12, rhs.m21, lhs.m13, rhs.m31, lhs.m14, rhs.m41,
            ),
            m21: R::sum_prod4(
                lhs.m21, rhs.m11, lhs.m22, rhs.m21, lhs.m23, rhs.m31, lhs.m24, rhs.m41,
            ),
            m31: R::sum_prod4(
                lhs.m31, rhs.m11, lhs.m32, rhs.m21, lhs.m33, rhs.m31, lhs.m34, rhs.m41,
            ),
            m41: R::sum_prod4(
                lhs.m41, rhs.m11, lhs.m42, rhs.m21, lhs.m43, rhs.m31, lhs.m44, rhs.m41,
            ),
            m12: R::sum_prod4(
                lhs.m11, rhs.m12, lhs.m12, rhs.m22, lhs.m13, rhs.m32, lhs.m14, rhs.m42,
            ),
            m22: R::sum_prod4(
                lhs.m21, rhs.m12, lhs.m22, rhs.m22, lhs.m23, rhs.m32, lhs.m24, rhs.m42,
            ),
            m32: R::sum_prod4(
                lhs.m31, rhs.m12, lhs.m32, rhs.m22, lhs.m33, rhs.m32, lhs.m34, rhs.m42,
            ),
            m42: R::sum_prod4(
                lhs.m41, rhs.m12, lhs.m42, rhs.m22, lhs.m43, rhs.m32, lhs.m44, rhs.m42,
            ),
            m13: R::sum_prod4(
                lhs.m11, rhs.m13, lhs.m12, rhs.m23, lhs.m13, rhs.m33, lhs.m14, rhs.m43,
            ),
            m23: R::sum_prod4(
                lhs.m21, rhs.m13, lhs.m22, rhs.m23, lhs.m23, rhs.m33, lhs.m24, rhs.m43,
            ),
            m33: R::sum_prod4(
                lhs.m31, rhs.m13, lhs.m32, rhs.m23, lhs.m33, rhs.m33, lhs.m34, rhs.m43,
            ),
            m43: R::sum_prod4(
                lhs.m41, rhs.m13, lhs.m42, rhs.m23, lhs.m43, rhs.m33, lhs.m44, rhs.m43,
            ),
            m14: R::sum_prod4(
                lhs.m11, rhs.m14, lhs.m12, rhs.m24, lhs.m13, rhs.m34, lhs.m14, rhs.m44,
            ),
            m24: R::sum_prod4(
                lhs.m21, rhs.m14, lhs.m22, rhs.m24, lhs.m23, rhs.m34, lhs.m24, rhs.m44,
            ),
            m34: R::sum_prod4(
                lhs.m31, rhs.m14, lhs.m32, rhs.m24, lhs.m33, rhs.m34, lhs.m34, rhs.m44,
            ),
            m44: R::sum_prod4(
                lhs.m41, rhs.m14, lhs.m42, rhs.m24, lhs.m43, rhs.m34, lhs.m44, rhs.m44,
            ),
        }
    }
}

/// `a += b`.
pub impl Matrix4AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Matrix4<T>, Matrix4<T>> {
    #[inline(always)]
    fn add_assign(ref self: Matrix4<T>, rhs: Matrix4<T>) {
        self = self + rhs;
    }
}

/// `a -= b`.
pub impl Matrix4SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Matrix4<T>, Matrix4<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Matrix4<T>, rhs: Matrix4<T>) {
        self = self - rhs;
    }
}

/// `a *= b` (matrix product, `a = a * b`).
pub impl Matrix4MulAssign<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Matrix4<T>, Matrix4<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Matrix4<T>, rhs: Matrix4<T>) {
        self = self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix_test_utils::{fx, int, m4, m4i, max_ulp_diff4, ulp_diff, v4i, v4t};
    use crate::base::{oracle_matrix4, oracle_matrix4_inverse};
    use super::{Matrix4, Matrix4InternalTrait, Matrix4Trait};

    // --- losing candidates of the determinant / inverse study (kept as evidence) -----------------

    /// 3x3 determinant of the given rows (cofactor expansion along the first one).
    fn det3(
        r1: (Fixed, Fixed, Fixed), r2: (Fixed, Fixed, Fixed), r3: (Fixed, Fixed, Fixed),
    ) -> Fixed {
        let ((a, b, c), (d, e, f), (g, h, i)) = (r1, r2, r3);
        Real::sum_prod3(
            a,
            Real::diff_prod(e, i, f, h),
            b,
            Real::diff_prod(f, g, d, i),
            c,
            Real::diff_prod(d, h, e, g),
        )
    }

    /// Determinant by cofactor expansion along the first row (four 3x3 determinants): three
    /// roundings and 12 `diff_prod` + 4 `sum_prod3` + 4 products, against two roundings and 12
    /// `diff_prod` + 6 products for the 2x2 minors.
    fn determinant_row_cofactors(m: Matrix4<Fixed>) -> Fixed {
        let c1 = det3((m.m22, m.m23, m.m24), (m.m32, m.m33, m.m34), (m.m42, m.m43, m.m44));
        let c2 = det3((m.m21, m.m23, m.m24), (m.m31, m.m33, m.m34), (m.m41, m.m43, m.m44));
        let c3 = det3((m.m21, m.m22, m.m24), (m.m31, m.m32, m.m34), (m.m41, m.m42, m.m44));
        let c4 = det3((m.m21, m.m22, m.m23), (m.m31, m.m32, m.m33), (m.m41, m.m42, m.m43));
        let w = Real::wide_add_prod(Real::<Fixed>::wide_zero(), m.m11, c1);
        let w = Real::wide_sub_prod(w, m.m12, c2);
        let w = Real::wide_add_prod(w, m.m13, c3);
        Real::wide_rescale(Real::wide_sub_prod(w, m.m14, c4))
    }

    /// `adjugate / determinant` without the integer pre-scaling of small matrices.
    fn try_inverse_div(m: Matrix4<Fixed>) -> Option<Matrix4<Fixed>> {
        let adj = m.adjugate();
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        Some(
            Matrix4 {
                m11: adj.m11 / det,
                m21: adj.m21 / det,
                m31: adj.m31 / det,
                m41: adj.m41 / det,
                m12: adj.m12 / det,
                m22: adj.m22 / det,
                m32: adj.m32 / det,
                m42: adj.m42 / det,
                m13: adj.m13 / det,
                m23: adj.m23 / det,
                m33: adj.m33 / det,
                m43: adj.m43 / det,
                m14: adj.m14 / det,
                m24: adj.m24 / det,
                m34: adj.m34 / det,
                m44: adj.m44 / det,
            },
        )
    }

    /// Upstream's `adjugate / determinant` (the formula of `try_inverse_div`) through ONE prepared
    /// divisor (`Real::div16`): bit-identical to `try_inverse_div`, so it fails the same oracle
    /// cases; kept to price upstream's formula at its cheapest (WP 7.2).
    fn try_inverse_div_n(m: Matrix4<Fixed>) -> Option<Matrix4<Fixed>> {
        let adj = m.adjugate();
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        let (m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44) =
            Real::div16(
            adj.m11,
            adj.m21,
            adj.m31,
            adj.m41,
            adj.m12,
            adj.m22,
            adj.m32,
            adj.m42,
            adj.m13,
            adj.m23,
            adj.m33,
            adj.m43,
            adj.m14,
            adj.m24,
            adj.m34,
            adj.m44,
            det,
        );
        Some(
            Matrix4 {
                m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44,
            },
        )
    }

    /// `adjugate * (1 / determinant)`: one reciprocal, 16 multiplications.
    fn try_inverse_recip(m: Matrix4<Fixed>) -> Option<Matrix4<Fixed>> {
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        Some(m.adjugate().scale(det.recip()))
    }

    /// `(cases above the oracle tolerance, worst error in ulp)` of an inverse candidate.
    fn inverse_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = match variant {
                0 => m4(a).try_inverse(),
                1 => try_inverse_div(m4(a)),
                3 => try_inverse_div_n(m4(a)),
                _ => try_inverse_recip(m4(a)),
            };
            let err = max_ulp_diff4(got.unwrap(), m4(expected));
            if err > tol.into() {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    // --- constructors and accessors --------------------------------------------------------------

    #[test]
    fn test_new_is_row_major() {
        let m = Matrix4Trait::new(
            int(1),
            int(2),
            int(3),
            int(4),
            int(5),
            int(6),
            int(7),
            int(8),
            int(9),
            int(10),
            int(11),
            int(12),
            int(13),
            int(14),
            int(15),
            int(16),
        );
        assert!(m == m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]));
        assert!(m.m12 == int(2) && m.m21 == int(5));
    }

    #[test]
    fn test_serde_is_column_major() {
        let m = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let mut out = array![];
        m.serialize(ref out);
        let one: felt252 = 0x100000000;
        assert!(
            out == array![
                1 * one, 5 * one, 9 * one, 13 * one, 2 * one, 6 * one, 10 * one, 14 * one, 3 * one,
                7 * one, 11 * one, 15 * one, 4 * one, 8 * one, 12 * one, 16 * one,
            ],
        );
    }

    #[test]
    fn test_zeros_identity() {
        assert!(
            Matrix4Trait::<
                Fixed,
            >::zeros() == m4i([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        );
        assert!(
            Matrix4Trait::<
                Fixed,
            >::identity() == m4i([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        );
        assert!(Matrix4Trait::<Fixed>::zeros() == Default::default());
    }

    #[test]
    fn test_from_diagonal() {
        assert!(
            Matrix4Trait::from_diagonal(
                v4i(2, -3, 4, -5),
            ) == m4i([[2, 0, 0, 0], [0, -3, 0, 0], [0, 0, 4, 0], [0, 0, 0, -5]]),
        );
        assert!(
            Matrix4Trait::from_diagonal_element(
                int(7),
            ) == m4i([[7, 0, 0, 0], [0, 7, 0, 0], [0, 0, 7, 0], [0, 0, 0, 7]]),
        );
    }

    #[test]
    fn test_from_columns_from_rows() {
        let (r1, r2, r3, r4) = (
            v4i(1, 2, 3, 4), v4i(5, 6, 7, 8), v4i(9, 10, 11, 12), v4i(13, 14, 15, 16),
        );
        assert!(
            Matrix4Trait::from_rows(
                r1, r2, r3, r4,
            ) == m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
        );
        assert!(
            Matrix4Trait::from_columns(
                r1, r2, r3, r4,
            ) == m4i([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]),
        );
    }

    #[test]
    fn test_columns_rows_diagonal() {
        let m = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        assert!(m.column1() == v4i(1, 5, 9, 13));
        assert!(m.column2() == v4i(2, 6, 10, 14));
        assert!(m.column3() == v4i(3, 7, 11, 15));
        assert!(m.column4() == v4i(4, 8, 12, 16));
        assert!(m.row1() == v4i(1, 2, 3, 4));
        assert!(m.row2() == v4i(5, 6, 7, 8));
        assert!(m.row3() == v4i(9, 10, 11, 12));
        assert!(m.row4() == v4i(13, 14, 15, 16));
        assert!(m.diagonal() == v4i(1, 6, 11, 16));
        assert!(
            Matrix4Trait::from_columns(m.column1(), m.column2(), m.column3(), m.column4()) == m,
        );
        assert!(Matrix4Trait::from_rows(m.row1(), m.row2(), m.row3(), m.row4()) == m);
    }

    // --- exact operations
    // --------------------------------------------------------------------------

    #[test]
    fn test_add_oracle() {
        let mut cases = oracle_matrix4::matrix4_add_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m4(a) + m4(b) == m4(expected));
            let mut acc = m4(a);
            acc += m4(b);
            assert!(acc == m4(expected));
        }
    }

    #[test]
    fn test_sub_oracle() {
        let mut cases = oracle_matrix4::matrix4_sub_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m4(a) - m4(b) == m4(expected));
            let mut acc = m4(a);
            acc -= m4(b);
            assert!(acc == m4(expected));
        }
    }

    #[test]
    fn test_neg() {
        let m = m4i([[1, -2, 3, -4], [-5, 6, -7, 8], [9, -10, 11, -12], [-13, 14, -15, 0]]);
        assert!(-m == m4i([[-1, 2, -3, 4], [5, -6, 7, -8], [-9, 10, -11, 12], [13, -14, 15, 0]]));
        assert!(m + (-m) == Matrix4Trait::zeros());
    }

    #[test]
    #[should_panic(expected: 'i64_add Overflow')]
    fn test_add_overflow_panics() {
        let m = black_box(Matrix4Trait::from_diagonal_element(Real::<Fixed>::MAX));
        let _ = m + m;
    }

    #[test]
    #[should_panic(expected: 'i64_neg Underflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(Matrix4Trait::from_diagonal_element(Real::<Fixed>::MIN));
    }

    #[test]
    fn test_transpose_oracle() {
        let mut cases = oracle_matrix4::matrix4_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m4(a).transpose() == m4(expected));
            assert!(m4(a).transpose().transpose() == m4(a));
        }
    }

    #[test]
    fn test_trace_oracle() {
        let mut cases = oracle_matrix4::matrix4_trace_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m4(a).trace() == fx(expected));
        }
    }

    #[test]
    fn test_abs() {
        let m = m4i([[1, -2, 3, -4], [-5, 6, -7, 8], [9, -10, 11, -12], [-13, 14, -15, 0]]);
        assert!(m.abs() == m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]));
    }

    // --- products
    // ----------------------------------------------------------------------------------

    #[test]
    fn test_scale_oracle() {
        let mut cases = oracle_matrix4::matrix4_scale_cases();
        while let Some(case) = cases.pop_front() {
            let (a, k, expected, _) = *case;
            assert!(m4(a).scale(fx(k)) == m4(expected));
        }
    }

    #[test]
    fn test_component_mul_exact() {
        let a = m4i([[1, -2, 3, -4], [-4, 5, -6, 7], [7, -8, 0, 1], [2, 2, -2, -2]]);
        let b = m4i([[2, 2, 2, 2], [3, 3, 3, 3], [-1, -1, -1, -1], [5, 5, 5, 5]]);
        assert!(
            a
                .component_mul(
                    b,
                ) == m4i([[2, -4, 6, -8], [-12, 15, -18, 21], [-7, 8, 0, -1], [10, 10, -10, -10]]),
        );
        // 0.5 ulp floors to 0, -0.5 ulp to -1 ulp, 1.5 ulp to 1 ulp, -1.5 ulp to -2 ulp.
        let h = Matrix4Trait::from_diagonal_element(Real::<Fixed>::HALF);
        let e = Matrix4Trait::from_diagonal(v4t((1, -1, 3, -3)));
        assert!(e.component_mul(h) == Matrix4Trait::from_diagonal(v4t((0, -1, 1, -2))));
    }

    #[test]
    fn test_mul_oracle() {
        let mut cases = oracle_matrix4::matrix4_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m4(a) * m4(b) == m4(expected));
            let mut acc = m4(a);
            acc *= m4(b);
            assert!(acc == m4(expected));
        }
    }

    #[test]
    fn test_mul_exact_and_identity() {
        let a = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let b = m4i([[-1, 0, 2, 1], [3, 1, 0, 0], [0, -2, 1, 2], [1, 0, 0, -1]]);
        assert!(
            a * b == m4i([[9, -4, 5, 3], [21, -8, 17, 11], [33, -12, 29, 19], [45, -16, 41, 27]]),
        );
        assert!(a * Matrix4Trait::identity() == a);
        assert!(Matrix4Trait::identity() * a == a);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_mul_overflow_panics() {
        let m = black_box(Matrix4Trait::from_diagonal_element(int(65536)));
        let _ = m * m;
    }

    #[test]
    fn test_mul_vec_oracle() {
        let mut cases = oracle_matrix4::matrix4_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, expected, _) = *case;
            assert!(m4(a).mul_vec(v4t(v)) == v4t(expected));
            assert!(m4(a).transpose().tr_mul_vec(v4t(v)) == v4t(expected));
        }
    }

    #[test]
    fn test_mul_vec_tr_mul_vec_exact() {
        let a = m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        assert!(a.mul_vec(v4i(1, 0, -1, 2)) == v4i(6, 14, 22, 30));
        assert!(a.tr_mul_vec(v4i(1, 0, -1, 2)) == v4i(18, 20, 22, 24));
    }

    #[test]
    fn test_tr_mul_oracle() {
        let mut cases = oracle_matrix4::matrix4_tr_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m4(a).tr_mul(m4(b)) == m4(expected));
            assert!(m4(a).transpose() * m4(b) == m4(expected));
        }
    }

    #[test]
    fn test_from_outer_oracle() {
        let mut cases = oracle_matrix4::matrix4_outer_cases();
        while let Some(case) = cases.pop_front() {
            let (u, v, expected, _) = *case;
            assert!(Matrix4InternalTrait::from_outer(v4t(u), v4t(v)) == m4(expected));
        }
    }

    // --- norms
    // -------------------------------------------------------------------------------------

    #[test]
    fn test_norm_exact() {
        let m = m4i([[2, -2, 2, -2], [-2, 2, -2, 2], [2, -2, 2, -2], [-2, 2, -2, 2]]);
        assert!(m.norm_squared() == int(64));
        assert!(m.norm() == int(8));
        assert!(Matrix4Trait::<Fixed>::zeros().norm() == int(0));
        // The squared norm (4 * 2^58) does not fit, the norm does.
        assert!(Matrix4Trait::from_diagonal_element(int(0x20000000)).norm() >= int(0x20000000));
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_norm_squared_overflow_panics() {
        black_box(Matrix4Trait::from_diagonal_element(int(0x20000000))).norm_squared();
    }

    // --- determinant and inverse
    // -------------------------------------------------------------------

    #[test]
    fn test_determinant_exact() {
        assert!(
            m4i([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, -4, 0], [0, 0, 0, 5]])
                .determinant() == int(-120),
        );
        assert!(
            m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
                .determinant() == int(0),
        );
        assert!(
            m4i([[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]])
                .determinant() == int(30),
        );
        assert!(Matrix4Trait::<Fixed>::identity().determinant() == int(1));
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle_matrix4_inverse::matrix4_determinant_cases();
        let mut worst = 0;
        let mut worst_alt = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = ulp_diff(m4(a).determinant(), fx(expected));
            assert!(err <= tol.into(), "determinant error {err} > {tol}");
            worst = core::cmp::max(worst, err);
            worst_alt =
                core::cmp::max(worst_alt, ulp_diff(determinant_row_cofactors(m4(a)), fx(expected)));
        }
        // Worst error over the oracle (exact expectation, |a_ij| up to 1e3): 18993 ulp, against
        // 35351 for the row cofactors.
        assert!(worst == 18993, "worst {worst}");
        assert!(worst_alt == 35351, "worst alt {worst_alt}");
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_determinant_overflow_panics() {
        black_box(Matrix4Trait::from_diagonal_element(int(256))).determinant();
    }

    #[test]
    fn test_adjugate_identity() {
        let m = m4i([[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]]);
        let det = m.determinant();
        assert!(det == int(30));
        assert!(m * m.adjugate() == Matrix4Trait::from_diagonal_element(det));
        assert!(m.adjugate() * m == Matrix4Trait::from_diagonal_element(det));
    }

    #[test]
    fn test_try_inverse_exact() {
        let m = m4i([[1, 1, 0, 2], [2, 3, 3, 4], [0, 3, 10, 1], [1, 1, 2, 5]]);
        let inv = m.try_inverse().unwrap();
        assert!(
            inv == m4i([[86, -40, 13, -5], [-59, 28, -9, 3], [19, -9, 3, -1], [-13, 6, -2, 1]]),
        );
        assert!(m * inv == Matrix4Trait::identity());
        let d = Matrix4Trait::from_diagonal(v4i(2, -4, 8, -16)).try_inverse().unwrap();
        assert!(
            d == Matrix4Trait::from_diagonal(
                v4t((0x80000000, -0x40000000, 0x20000000, -0x10000000)),
            ),
        );
        assert!(
            Matrix4Trait::<Fixed>::identity().try_inverse().unwrap() == Matrix4Trait::identity(),
        );
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = max_ulp_diff4(m4(a).try_inverse().unwrap(), m4(expected));
            assert!(err <= tol.into(), "inverse error {err} > {tol}");
        }
    }

    #[test]
    fn test_try_inverse_candidates_error() {
        // Oracle, 30 well-conditioned matrices (10 small, 10 unit, 10 medium):
        // (cases above the oracle tolerance, worst error in ulp) of the shipped algorithm, of
        // `adjugate / det` without pre-scaling and of `adjugate * (1 / det)`.
        assert!(inverse_failures(0) == (0, 50));
        assert!(inverse_failures(1) == (6, 56713));
        // Upstream's formula through one prepared divisor: the same bits as `try_inverse_div`.
        assert!(inverse_failures(3) == (6, 56713));
        assert!(inverse_failures(2) == (10, 56713));
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = m4(a).try_inverse().unwrap();
            // Worst residual over the oracle: 71 ulp.
            assert!((m4(a) * inv).is_identity(71));
            assert!((inv * m4(a)).is_identity(71));
        }
    }

    #[test]
    fn test_try_inverse_singular_oracle() {
        let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, is_some, _) = *case;
            assert!(m4(a).try_inverse().is_some() == is_some);
        }
        assert!(Matrix4Trait::<Fixed>::zeros().try_inverse().is_none());
        assert!(
            m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
                .try_inverse()
                .is_none(),
        );
    }

    #[test]
    fn test_try_inverse_near_singular_oracle() {
        let mut cases = oracle_matrix4_inverse::matrix4_try_inverse_near_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = m4(a).try_inverse().unwrap();
            assert!(max_ulp_diff4(inv, m4(expected)) <= tol.into());
        }
    }

    #[test]
    fn test_try_inverse_small_scale() {
        // 2^-12 * I: without pre-scaling the determinant (below the resolution).
        let m = Matrix4Trait::from_diagonal_element(fx(0x100000));
        let expected = Matrix4Trait::from_diagonal_element(int(4096));
        assert!(m.determinant() == int(0));
        assert!(try_inverse_div(m).is_none());
        // Relative error below 2^-33 (0 ulp on 4096).
        let inv = m.try_inverse().unwrap();
        assert!(inv.abs_diff_eq(expected, 0));
        assert!(inv == expected);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_tiny_norm_panics() {
        let _ = black_box(Matrix4Trait::from_diagonal_element(fx(1))).try_inverse();
    }

    // --- approximate equality
    // ----------------------------------------------------------------------

    #[test]
    fn test_is_identity_abs_diff_eq() {
        let id = Matrix4Trait::<Fixed>::identity();
        let mut m = id;
        m.m21 = fx(3);
        m.m44 = fx(0x100000000 - 2);
        assert!(m.is_identity(3) && !m.is_identity(2));
        assert!(m.abs_diff_eq(id, 3) && !m.abs_diff_eq(id, 2));
        assert!(id.is_identity(0) && id.abs_diff_eq(id, 0));
    }

    // --- gas benchmarks
    // ----------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_matrix4_new__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_new__struct() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(
            Matrix4Trait::new(
                a.m11,
                a.m12,
                a.m13,
                a.m14,
                a.m21,
                a.m22,
                a.m23,
                a.m24,
                a.m31,
                a.m32,
                a.m33,
                a.m34,
                a.m41,
                a.m42,
                a.m43,
                a.m44,
            ) == e,
        );
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_zeros__baseline() {
        let e = black_box(m4([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_zeros__const() {
        let e = black_box(m4([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]));
        assert!(Matrix4Trait::<Fixed>::zeros() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_identity__baseline() {
        let e = black_box(
            m4(
                [
                    [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                    [0, 0, 0, 4294967296],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_identity__const() {
        let e = black_box(
            m4(
                [
                    [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                    [0, 0, 0, 4294967296],
                ],
            ),
        );
        assert!(Matrix4Trait::<Fixed>::identity() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_diagonal__baseline() {
        let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(
            m4(
                [
                    [4751241150, 0, 0, 0], [0, 2551995575, 0, 0], [0, 0, -4086721614, 0],
                    [0, 0, 0, -3145554884],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_diagonal__struct() {
        let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(
            m4(
                [
                    [4751241150, 0, 0, 0], [0, 2551995575, 0, 0], [0, 0, -4086721614, 0],
                    [0, 0, 0, -3145554884],
                ],
            ),
        );
        assert!(Matrix4Trait::from_diagonal(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_diagonal_element__baseline() {
        let _k = black_box(fx(-7516192768));
        let e = black_box(
            m4(
                [
                    [-7516192768, 0, 0, 0], [0, -7516192768, 0, 0], [0, 0, -7516192768, 0],
                    [0, 0, 0, -7516192768],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_diagonal_element__struct() {
        let k = black_box(fx(-7516192768));
        let e = black_box(
            m4(
                [
                    [-7516192768, 0, 0, 0], [0, -7516192768, 0, 0], [0, 0, -7516192768, 0],
                    [0, 0, 0, -7516192768],
                ],
            ),
        );
        assert!(Matrix4Trait::from_diagonal_element(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_columns__baseline() {
        let _c1 = black_box(v4t((6065401010, -4721681308, 7990534196, 4445892797)));
        let _c2 = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
        let _c3 = black_box(v4t((-4169889720, 2695648994, -7212805291, 6546834464)));
        let _c4 = black_box(v4t((-3730684931, 6053150518, -2184064506, 5336335505)));
        let e = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_columns__struct() {
        let c1 = black_box(v4t((6065401010, -4721681308, 7990534196, 4445892797)));
        let c2 = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
        let c3 = black_box(v4t((-4169889720, 2695648994, -7212805291, 6546834464)));
        let c4 = black_box(v4t((-3730684931, 6053150518, -2184064506, 5336335505)));
        let e = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(Matrix4Trait::from_columns(c1, c2, c3, c4) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_rows__baseline() {
        let _r1 = black_box(v4t((6065401010, -2161577173, -4169889720, -3730684931)));
        let _r2 = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
        let _r3 = black_box(v4t((7990534196, 5603363342, -7212805291, -2184064506)));
        let _r4 = black_box(v4t((4445892797, 8226859758, 6546834464, 5336335505)));
        let e = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_rows__struct() {
        let r1 = black_box(v4t((6065401010, -2161577173, -4169889720, -3730684931)));
        let r2 = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
        let r3 = black_box(v4t((7990534196, 5603363342, -7212805291, -2184064506)));
        let r4 = black_box(v4t((4445892797, 8226859758, 6546834464, 5336335505)));
        let e = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(Matrix4Trait::from_rows(r1, r2, r3, r4) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_outer__baseline() {
        let _u = black_box(v4t((-6975932915, 4075315203, -4507549853, -6978514818)));
        let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(
            m4(
                [
                    [-7717017906, -4144979160, 6637697997, 5109044688],
                    [4508254419, 2421482085, -3877719568, -2984685740],
                    [-4986407317, -2678308469, 4288996898, 3301246430],
                    [-7719874096, -4146513282, 6640154714, 5110935626],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_from_outer__products() {
        let u = black_box(v4t((-6975932915, 4075315203, -4507549853, -6978514818)));
        let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(
            m4(
                [
                    [-7717017906, -4144979160, 6637697997, 5109044688],
                    [4508254419, 2421482085, -3877719568, -2984685740],
                    [-4986407317, -2678308469, 4288996898, 3301246430],
                    [-7719874096, -4146513282, 6640154714, 5110935626],
                ],
            ),
        );
        assert!(Matrix4InternalTrait::from_outer(u, v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_column__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_column__second() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(v4t((-2161577173, -7621971610, 5603363342, 8226859758)));
        assert!(a.column2() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_row__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_row__second() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(v4t((-4721681308, -7621971610, 2695648994, 6053150518)));
        assert!(a.row2() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_diagonal__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(v4t((6065401010, -7621971610, -7212805291, 5336335505)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_diagonal__struct() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(v4t((6065401010, -7621971610, -7212805291, 5336335505)));
        assert!(a.diagonal() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_transpose__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [6065401010, -4721681308, 7990534196, 4445892797],
                    [-2161577173, -7621971610, 5603363342, 8226859758],
                    [-4169889720, 2695648994, -7212805291, 6546834464],
                    [-3730684931, 6053150518, -2184064506, 5336335505],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_transpose__struct() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [6065401010, -4721681308, 7990534196, 4445892797],
                    [-2161577173, -7621971610, 5603363342, 8226859758],
                    [-4169889720, 2695648994, -7212805291, 6546834464],
                    [-3730684931, 6053150518, -2184064506, 5336335505],
                ],
            ),
        );
        assert!(a.transpose() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_trace__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(-3433040386));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_trace__sum() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(-3433040386));
        assert!(a.trace() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_abs__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [6065401010, 2161577173, 4169889720, 3730684931],
                    [4721681308, 7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, 7212805291, 2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_abs__componentwise() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [6065401010, 2161577173, 4169889720, 3730684931],
                    [4721681308, 7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, 7212805291, 2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        assert!(a.abs() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_add__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [12489641950, -10675106085, -11071906975, 4456957075],
                    [-1054211790, -4741980083, -5387795612, 3049489065],
                    [10821163348, -585825044, -14658492792, 4310724722],
                    [12527228599, 1650094630, -435296038, 843506602],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_add__operator() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [12489641950, -10675106085, -11071906975, 4456957075],
                    [-1054211790, -4741980083, -5387795612, 3049489065],
                    [10821163348, -585825044, -14658492792, 4310724722],
                    [12527228599, 1650094630, -435296038, 843506602],
                ],
            ),
        );
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_add__assign() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [12489641950, -10675106085, -11071906975, 4456957075],
                    [-1054211790, -4741980083, -5387795612, 3049489065],
                    [10821163348, -585825044, -14658492792, 4310724722],
                    [12527228599, 1650094630, -435296038, 843506602],
                ],
            ),
        );
        let mut r = a;
        r += b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_sub__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-358839930, 6351951739, 2732127535, -11918326937],
                    [-8389150826, -10501963137, 10779093600, 9056811971],
                    [5159905044, 11792551728, 232882210, -8678853734],
                    [-3635443005, 14803624886, 13528964966, 9829164408],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_sub__operator() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-358839930, 6351951739, 2732127535, -11918326937],
                    [-8389150826, -10501963137, 10779093600, 9056811971],
                    [5159905044, 11792551728, 232882210, -8678853734],
                    [-3635443005, 14803624886, 13528964966, 9829164408],
                ],
            ),
        );
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_sub__assign() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-358839930, 6351951739, 2732127535, -11918326937],
                    [-8389150826, -10501963137, 10779093600, 9056811971],
                    [5159905044, 11792551728, 232882210, -8678853734],
                    [-3635443005, 14803624886, 13528964966, 9829164408],
                ],
            ),
        );
        let mut r = a;
        r -= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_neg__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-6065401010, 2161577173, 4169889720, 3730684931],
                    [4721681308, 7621971610, -2695648994, -6053150518],
                    [-7990534196, -5603363342, 7212805291, 2184064506],
                    [-4445892797, -8226859758, -6546834464, -5336335505],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_neg__operator() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-6065401010, 2161577173, 4169889720, 3730684931],
                    [4721681308, 7621971610, -2695648994, -6053150518],
                    [-7990534196, -5603363342, 7212805291, 2184064506],
                    [-4445892797, -8226859758, -6546834464, -5336335505],
                ],
            ),
        );
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_scale__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _k = black_box(fx(-7516192768));
        let e = black_box(
            m4(
                [
                    [-10614451768, 3782760052, 7297307010, 6528698629],
                    [8262942289, 13338450317, -4717385740, -10593013407],
                    [-13983434843, -9805885849, 12622409259, 3822112885],
                    [-7780312395, -14397004577, -11456960312, -9338587134],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_scale__products() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let k = black_box(fx(-7516192768));
        let e = black_box(
            m4(
                [
                    [-10614451768, 3782760052, 7297307010, 6528698629],
                    [8262942289, 13338450317, -4717385740, -10593013407],
                    [-13983434843, -9805885849, 12622409259, 3822112885],
                    [-7780312395, -14397004577, -11456960312, -9338587134],
                ],
            ),
        );
        assert!(a.scale(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_component_mul__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [9072385143, 4284700788, 6701017450, -7111931372],
                    [-4031840310, -5110915205, -5073409835, -4233237096],
                    [5266219152, -8074629894, 12504005386, -3302711674],
                    [8365314601, -12597563763, -10642887234, -5582171119],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_component_mul__products() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [9072385143, 4284700788, 6701017450, -7111931372],
                    [-4031840310, -5110915205, -5073409835, -4233237096],
                    [5266219152, -8074629894, 12504005386, -3302711674],
                    [8365314601, -12597563763, -10642887234, -5582171119],
                ],
            ),
        );
        assert!(a.component_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_mul_vec__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(v4t((12125377576, -16750294840, 20631480820, -331180081)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_mul_vec__fused() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(v4t((12125377576, -16750294840, 20631480820, -331180081)));
        assert!(a.mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_tr_mul_vec__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(v4t((-6954980908, -18276934794, -942863330, -2360400514)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_tr_mul_vec__fused() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(v4t((-6954980908, -18276934794, -942863330, -2360400514)));
        assert!(a.tr_mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_tr_mul_vec__transpose_mul_vec() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let v = black_box(v4t((4751241150, 2551995575, -4086721614, -3145554884)));
        let e = black_box(v4t((-6954980908, -18276934794, -942863330, -2360400514)));
        assert!(a.transpose().mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_mul__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-2541171855, -1750704652, 7614798780, 10671269236],
                    [-404809204, -8905090396, 7419396255, -5926404020],
                    [7873481502, 1656702967, -7332201226, 2691523401],
                    [28030393544, -20901758122, -42652634060, 7039807887],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_mul__fused() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-2541171855, -1750704652, 7614798780, 10671269236],
                    [-404809204, -8905090396, 7419396255, -5926404020],
                    [7873481502, 1656702967, -7332201226, 2691523401],
                    [28030393544, -20901758122, -42652634060, 7039807887],
                ],
            ),
        );
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_mul__assign() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-2541171855, -1750704652, 7614798780, 10671269236],
                    [-404809204, -8905090396, 7419396255, -5926404020],
                    [7873481502, 1656702967, -7332201226, 2691523401],
                    [28030393544, -20901758122, -42652634060, 7039807887],
                ],
            ),
        );
        let mut r = a;
        r *= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_tr_mul__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [18672078587, -33511520172, -21940301505, 22297227754],
                    [9430856166, -21498408073, -5269160636, 1077166020],
                    [3629416479, 10442068122, 3488725768, -27589927991],
                    [8189903724, 6429869131, -10286035073, -20230051259],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_tr_mul__fused() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [18672078587, -33511520172, -21940301505, 22297227754],
                    [9430856166, -21498408073, -5269160636, 1077166020],
                    [3629416479, 10442068122, 3488725768, -27589927991],
                    [8189903724, 6429869131, -10286035073, -20230051259],
                ],
            ),
        );
        assert!(a.tr_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_tr_mul__transpose_mul() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6424240940, -8513528912, -6902017255, 8187642006],
                    [3667469518, 2879991527, -8083444606, -3003661453],
                    [2830629152, -6189188386, -7445687501, 6494789228],
                    [8081335802, -6576765128, -6982130502, -4492828903],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [18672078587, -33511520172, -21940301505, 22297227754],
                    [9430856166, -21498408073, -5269160636, 1077166020],
                    [3629416479, 10442068122, 3488725768, -27589927991],
                    [8189903724, 6429869131, -10286035073, -20230051259],
                ],
            ),
        );
        assert!(a.transpose() * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_norm_squared__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(118252144586));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_norm_squared__wide() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(118252144586));
        assert!(a.norm_squared() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_norm__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(22536394868));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_norm__wide() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(22536394868));
        assert!(a.norm() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_determinant__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(53942245227));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_determinant__minors2x2() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(53942245227));
        assert!(a.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_determinant__alt_row_cofactors() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(fx(53942245230));
        assert!(determinant_row_cofactors(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_adjugate__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [34509485500, 5764173809, -2916270880, 16393887163],
                    [-26230675187, -13185528909, 11404010475, 1286032516],
                    [22774124647, -13617633838, -34850938698, 17104616577],
                    [-16252316022, 32232042512, 27605003334, 6789980084],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_adjugate__cofactors() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [34509485500, 5764173809, -2916270880, 16393887163],
                    [-26230675187, -13185528909, 11404010475, 1286032516],
                    [22774124647, -13617633838, -34850938698, 17104616577],
                    [-16252316022, 32232042512, 27605003334, 6789980084],
                ],
            ),
        );
        assert!(a.adjugate() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__baseline() {
        let _a = black_box(
            m4(
                [
                    [-888332506, 134474598, -3038885559, -4458555824],
                    [638823398, 2509477802, -5314665052, 2648272285],
                    [-2414734434, -1408008892, -595561448, 2626308541],
                    [-585391816, 4129884628, -2943537464, 820202461],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-1580935658, 3651218852, -5141402076, -3920011882],
                    [-1134111645, -2131795105, -1434037613, 5310029221],
                    [-1976110118, -3309244046, -601828060, 1869987304],
                    [-2509710474, 1463757663, 1391329344, -333367901],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__prescaled_det_ge_half() {
        let a = black_box(
            m4(
                [
                    [-888332506, 134474598, -3038885559, -4458555824],
                    [638823398, 2509477802, -5314665052, 2648272285],
                    [-2414734434, -1408008892, -595561448, 2626308541],
                    [-585391816, 4129884628, -2943537464, 820202461],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-1580935658, 3651218852, -5141402076, -3920011882],
                    [-1134111644, -2131795104, -1434037612, 5310029222],
                    [-1976110117, -3309244046, -601828060, 1869987305],
                    [-2509710474, 1463757663, 1391329344, -333367901],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__prescaled_norm_gt_one() {
        let a = black_box(
            m4(
                [
                    [123994628, -2146559718, 1588261118, -1698136391],
                    [-1258299206, 94776244, 262037008, -1219409418],
                    [908609721, -228549883, -1944374345, -1297968465],
                    [-1509590524, -2094158678, -5668301786, 461699291],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [1007590045, -8113460287, 5589193161, -2009983769],
                    [-6224010726, 4566957182, 2947460227, -2543902749],
                    [1941005617, -48662219, -3082533347, -1655358742],
                    [-1106373953, -6410886765, -6200760652, 1520649381],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__prescaled_small() {
        let a = black_box(
            m4(
                [
                    [445290890, 496684497, -552302081, 121652393],
                    [-769872670, -183641367, -1342383347, 1104390896],
                    [143659673, -109440825, 146151883, 495732633],
                    [-235861466, 489393595, 274460440, 427305163],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [15582438276, -6677561308, 22505666527, -13287450710],
                    [12081009333, -2389516454, -15753444996, 21012555187],
                    [-9746308220, -6782176897, 9363645371, 9440493449],
                    [1024804345, 3407105854, 24450685987, 5706213630],
                ],
            ),
        );
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__alt_div() {
        let a = black_box(
            m4(
                [
                    [-888332506, 134474598, -3038885559, -4458555824],
                    [638823398, 2509477802, -5314665052, 2648272285],
                    [-2414734434, -1408008892, -595561448, 2626308541],
                    [-585391816, 4129884628, -2943537464, 820202461],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-1580935658, 3651218852, -5141402076, -3920011882],
                    [-1134111644, -2131795104, -1434037612, 5310029222],
                    [-1976110117, -3309244046, -601828060, 1869987305],
                    [-2509710474, 1463757663, 1391329344, -333367901],
                ],
            ),
        );
        assert!(try_inverse_div(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__alt_div_n() {
        let a = black_box(
            m4(
                [
                    [-888332506, 134474598, -3038885559, -4458555824],
                    [638823398, 2509477802, -5314665052, 2648272285],
                    [-2414734434, -1408008892, -595561448, 2626308541],
                    [-585391816, 4129884628, -2943537464, 820202461],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-1580935658, 3651218852, -5141402076, -3920011882],
                    [-1134111644, -2131795104, -1434037612, 5310029222],
                    [-1976110117, -3309244046, -601828060, 1869987305],
                    [-2509710474, 1463757663, 1391329344, -333367901],
                ],
            ),
        );
        assert!(try_inverse_div_n(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse__alt_recip() {
        let a = black_box(
            m4(
                [
                    [-888332506, 134474598, -3038885559, -4458555824],
                    [638823398, 2509477802, -5314665052, 2648272285],
                    [-2414734434, -1408008892, -595561448, 2626308541],
                    [-585391816, 4129884628, -2943537464, 820202461],
                ],
            ),
        );
        let e = black_box(
            m4(
                [
                    [-1580935658, 3651218852, -5141402076, -3920011882],
                    [-1134111645, -2131795105, -1434037613, 5310029221],
                    [-1976110118, -3309244046, -601828060, 1869987304],
                    [-2509710474, 1463757663, 1391329344, -333367901],
                ],
            ),
        );
        assert!(try_inverse_recip(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse_singular__baseline() {
        let _a = black_box(
            m4(
                [
                    [-8589934592, 4294967296, 12884901888, 21474836480],
                    [17179869184, 8589934592, -85899345920, -34359738368],
                    [8589934592, 0, -12884901888, 12884901888],
                    [8589934592, -8589934592, 17179869184, 8589934592],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_try_inverse_singular__none() {
        let a = black_box(
            m4(
                [
                    [-8589934592, 4294967296, 12884901888, 21474836480],
                    [17179869184, 8589934592, -85899345920, -34359738368],
                    [8589934592, 0, -12884901888, 12884901888],
                    [8589934592, -8589934592, 17179869184, 8589934592],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.try_inverse().is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_is_identity__baseline() {
        let _a = black_box(
            m4(
                [
                    [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                    [0, 0, 0, 4294967296],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_is_identity__all_compared() {
        let a = black_box(
            m4(
                [
                    [4294967296, 0, 0, 0], [0, 4294967296, 0, 0], [0, 0, 4294967296, 0],
                    [0, 0, 0, 4294967296],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.is_identity(2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_abs_diff_eq__baseline() {
        let _a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let _b = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix4_abs_diff_eq__all_compared() {
        let a = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let b = black_box(
            m4(
                [
                    [6065401010, -2161577173, -4169889720, -3730684931],
                    [-4721681308, -7621971610, 2695648994, 6053150518],
                    [7990534196, 5603363342, -7212805291, -2184064506],
                    [4445892797, 8226859758, 6546834464, 5336335505],
                ],
            ),
        );
        let e = black_box(true);
        assert!(a.abs_diff_eq(b, 2) == e);
    }
}
