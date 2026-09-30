//! Internal, no stability promise: the crate-private items of `base::matrix3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::Real;

/// Crate-internal kernels of `Matrix3<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `cross_matrix_mul` = `v.cross_matrix() * m` without the skew matrix, `adjugate`, `mul_transpose`
/// into a `SymMatrix3`) and the unrolled row / column accessors that stand for upstream `row(i)` /
/// `column(i)` views, used by the decompositions.
#[generate_trait]
pub impl Matrix3InternalImpl<
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
> of Matrix3InternalTrait<T> {
    /// The outer product `a * bᵀ`: each component is one floored product. Panics with the
    /// scalar's overflow error. Upstream: `a * b.transpose()`.
    #[inline(always)]
    fn from_outer(a: Vector3<T>, b: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: a.x * b.x,
            m21: a.y * b.x,
            m31: a.z * b.x,
            m12: a.x * b.y,
            m22: a.y * b.y,
            m32: a.z * b.y,
            m13: a.x * b.z,
            m23: a.y * b.z,
            m33: a.z * b.z,
        }
    }

    /// `[v]× * m` without materialising `[v]×`: column `j` is `v × column_j(m)`, 9 `diff_prod`
    /// (one rounding per component), bit-identical to `cross_matrix(v) * m`. Panics on overflow.
    /// Upstream: `v.cross_matrix() * m`; rapier: `gcross_matrix`.
    fn cross_matrix_mul(v: Vector3<T>, m: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::diff_prod(v.y, m.m31, v.z, m.m21),
            m21: R::diff_prod(v.z, m.m11, v.x, m.m31),
            m31: R::diff_prod(v.x, m.m21, v.y, m.m11),
            m12: R::diff_prod(v.y, m.m32, v.z, m.m22),
            m22: R::diff_prod(v.z, m.m12, v.x, m.m32),
            m32: R::diff_prod(v.x, m.m22, v.y, m.m12),
            m13: R::diff_prod(v.y, m.m33, v.z, m.m23),
            m23: R::diff_prod(v.z, m.m13, v.x, m.m33),
            m33: R::diff_prod(v.x, m.m23, v.y, m.m13),
        }
    }

    /// Column 1. Upstream: `column(0)`.
    #[inline(always)]
    fn column1(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m11, y: self.m21, z: self.m31 }
    }

    /// Column 2. Upstream: `column(1)`.
    #[inline(always)]
    fn column2(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m12, y: self.m22, z: self.m32 }
    }

    /// Column 3. Upstream: `column(2)`.
    #[inline(always)]
    fn column3(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m13, y: self.m23, z: self.m33 }
    }

    /// Row 1, as a (column) vector. Upstream: `row(0).transpose()`.
    #[inline(always)]
    fn row1(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m11, y: self.m12, z: self.m13 }
    }

    /// Row 2, as a (column) vector. Upstream: `row(1).transpose()`.
    #[inline(always)]
    fn row2(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m21, y: self.m22, z: self.m23 }
    }

    /// Row 3, as a (column) vector. Upstream: `row(2).transpose()`.
    #[inline(always)]
    fn row3(self: Matrix3<T>) -> Vector3<T> {
        Vector3 { x: self.m31, y: self.m32, z: self.m33 }
    }

    /// `self * selfᵀ` as a symmetric matrix: 6 fused kernels instead of 9, bit-identical to the
    /// upper triangle of `self * self.transpose()`. Panics on overflow.
    /// Upstream: `self * self.transpose()`.
    fn mul_transpose(self: Matrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::norm_squared3(self.m11, self.m12, self.m13),
            m12: R::sum_prod3(self.m11, self.m21, self.m12, self.m22, self.m13, self.m23),
            m13: R::sum_prod3(self.m11, self.m31, self.m12, self.m32, self.m13, self.m33),
            m22: R::norm_squared3(self.m21, self.m22, self.m23),
            m23: R::sum_prod3(self.m21, self.m31, self.m22, self.m32, self.m23, self.m33),
            m33: R::norm_squared3(self.m31, self.m32, self.m33),
        }
    }

    /// The adjugate (transposed cofactor matrix): `self * adjugate = determinant * I`. 9
    /// `diff_prod`, one rounding per component. Panics on overflow.
    /// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
    #[inline(always)]
    fn adjugate(self: Matrix3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::diff_prod(self.m22, self.m33, self.m23, self.m32),
            m21: R::diff_prod(self.m23, self.m31, self.m21, self.m33),
            m31: R::diff_prod(self.m21, self.m32, self.m22, self.m31),
            m12: R::diff_prod(self.m13, self.m32, self.m12, self.m33),
            m22: R::diff_prod(self.m11, self.m33, self.m13, self.m31),
            m32: R::diff_prod(self.m12, self.m31, self.m11, self.m32),
            m13: R::diff_prod(self.m12, self.m23, self.m13, self.m22),
            m23: R::diff_prod(self.m13, self.m21, self.m11, self.m23),
            m33: R::diff_prod(self.m11, self.m22, self.m12, self.m21),
        }
    }
}
