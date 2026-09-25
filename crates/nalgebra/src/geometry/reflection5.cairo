//! `Reflection5`: a reflection with respect to a hyperplane of the 5-dimensional space (upstream
//! `nalgebra::Reflection5`, i.e. `Reflection<T, Const<5>, ArrayStorage<T, 5, 1>>`), WP 8.4-P10.
//!
//! The hyperplane is orthogonal to `axis` (a unit vector) and lies at `bias` along it: the points
//! `p` with `axis . p = bias`. Written from one template for the sizes 1 to 6.
//!
//! Upstream's `reflect` / `reflect_rows` are generic over the shape of the matrix they update in
//! place (`&mut Matrix<T, R2, C2, S2>`); the Cairo counterparts are the generic traits
//! `Reflection5Columns<M, T>` (`reflect`, `reflect_with_sign`: `M` is any of the six shapes with 5
//! rows, `Vector5` to `Matrix5x6`) and `Reflection5Rows<L, W, T>` (`reflect_rows`,
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 5 columns and `W` the scratch
//! vector of its row count), both taking the matrix by `ref` like Rust's `&mut`. The scratch
//! vector `work` of `reflect_rows` holds `lhs * axis - bias` on return, as upstream's does.
//!
//! Numeric contract (AGENTS.md): every dot product is accumulated exactly and rounded ONCE, the
//! factor `-2 (axis . x - bias)` is rounded once (`wide_mul_scalar`), and every updated entry
//! is one fused `mul_add` / `sum_prod2` (one floor rounding and one overflow check). Overflow
//! panics; nothing wraps silently. Unlike upstream, the bias is subtracted unconditionally
//! (upstream skips it when zero: subtracting zero is exact, so the results are identical).

use simba::scalar::Real;
use crate::base::matrix1::Matrix1;
use crate::base::matrix2x5::Matrix2x5;
use crate::base::matrix3x5::Matrix3x5;
use crate::base::matrix4x5::Matrix4x5;
use crate::base::matrix5::Matrix5;
use crate::base::matrix5x2::Matrix5x2;
use crate::base::matrix5x3::Matrix5x3;
use crate::base::matrix5x4::Matrix5x4;
use crate::base::matrix5x6::Matrix5x6;
use crate::base::matrix6x5::Matrix6x5;
use crate::base::row_vector5::RowVector5;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use crate::base::vector6::Vector6;
use super::point5::Point5;

/// A reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along it.
/// The fields are private, like upstream's: build one with `new` / `new_containing_point`.
///
/// Like upstream's `Reflection`, the type derives no comparison, hash or serialization trait:
/// compare `axis()` and `bias()`. Cairo passes values by value, so `Copy` is implemented by hand
/// below (Cairo-imposed; upstream borrows).
#[derive(Drop)]
pub struct Reflection5<T> {
    axis: Vector5<T>,
    bias: T,
}

impl Reflection5Copy<T, +Copy<T>> of Copy<Reflection5<T>>;

/// Constructors and accessors of `Reflection5<T>` over a `Real` scalar.
#[generate_trait]
pub impl Reflection5Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Reflection5Trait<T> {
    /// The reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along
    /// it (a bias of zero is a plane through the origin). Exact. Upstream: `Reflection::new`.
    #[inline(always)]
    fn new(axis: Unit<Vector5<T>>, bias: T) -> Reflection5<T> {
        Reflection5 { axis: axis.value, bias }
    }

    /// The reflection with respect to the hyperplane orthogonal to `axis` that contains `pt`:
    /// `bias = axis . pt`, accumulated exactly and floored once. Panics on overflow. Upstream:
    /// `Reflection::new_containing_point`.
    #[inline(always)]
    fn new_containing_point(axis: Unit<Vector5<T>>, pt: Point5<T>) -> Reflection5<T> {
        let a = axis.value;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, pt.x);
        let w = R::wide_add_prod(w, a.y, pt.y);
        let w = R::wide_add_prod(w, a.z, pt.z);
        let w = R::wide_add_prod(w, a.w, pt.w);
        let w = R::wide_add_prod(w, a.a, pt.a);
        Reflection5 { axis: a, bias: R::wide_rescale(w) }
    }

    /// The reflection axis. Upstream: `Reflection::axis`.
    #[inline(always)]
    fn axis(self: Reflection5<T>) -> Vector5<T> {
        self.axis
    }

    /// The reflection bias: the position of the plane along the axis. Upstream:
    /// `Reflection::bias`.
    #[inline(always)]
    fn bias(self: Reflection5<T>) -> T {
        self.bias
    }
}

/// `reflect` / `reflect_with_sign` of `Reflection5` on a matrix `M` with 5 rows, updated in place.
pub trait Reflection5Columns<M, T> {
    /// Applies the reflection to the columns of `rhs`: every column `x` becomes `x - 2 (axis . x -
    /// bias) axis`. Panics on overflow. Upstream: `Reflection::reflect`.
    fn reflect(self: Reflection5<T>, ref rhs: M);
    /// Applies the reflection to the columns of `rhs` with a sign: every column `x` becomes
    /// `sign x - 2 sign (axis . x - bias) axis`. Panics on overflow. Upstream:
    /// `Reflection::reflect_with_sign`.
    fn reflect_with_sign(self: Reflection5<T>, ref rhs: M, sign: T);
}

/// `Reflection5Columns` on `Vector5`.
pub impl Reflection5ColumnsVector5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection5Columns<Vector5<T>, T> {
    fn reflect(self: Reflection5<T>, ref rhs: Vector5<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_add_prod(w, a.y, m.y);
        let w = R::wide_add_prod(w, a.z, m.z);
        let w = R::wide_add_prod(w, a.w, m.w);
        let w = R::wide_add_prod(w, a.a, m.a);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs =
            Vector5 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f1, a.y, m.y),
                z: R::mul_add(f1, a.z, m.z),
                w: R::mul_add(f1, a.w, m.w),
                a: R::mul_add(f1, a.a, m.a),
            };
    }

    fn reflect_with_sign(self: Reflection5<T>, ref rhs: Vector5<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_add_prod(w, a.y, m.y);
        let w = R::wide_add_prod(w, a.z, m.z);
        let w = R::wide_add_prod(w, a.w, m.w);
        let w = R::wide_add_prod(w, a.a, m.a);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs =
            Vector5 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f1, a.y, sign, m.y),
                z: R::sum_prod2(f1, a.z, sign, m.z),
                w: R::sum_prod2(f1, a.w, sign, m.w),
                a: R::sum_prod2(f1, a.a, sign, m.a),
            };
    }
}

/// `Reflection5Columns` on `Matrix5x2`.
pub impl Reflection5ColumnsMatrix5x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection5Columns<Matrix5x2<T>, T> {
    fn reflect(self: Reflection5<T>, ref rhs: Matrix5x2<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
            };
    }

    fn reflect_with_sign(self: Reflection5<T>, ref rhs: Matrix5x2<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
            };
    }
}

/// `Reflection5Columns` on `Matrix5x3`.
pub impl Reflection5ColumnsMatrix5x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection5Columns<Matrix5x3<T>, T> {
    fn reflect(self: Reflection5<T>, ref rhs: Matrix5x3<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x3 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
            };
    }

    fn reflect_with_sign(self: Reflection5<T>, ref rhs: Matrix5x3<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x3 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
            };
    }
}

/// `Reflection5Columns` on `Matrix5x4`.
pub impl Reflection5ColumnsMatrix5x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection5Columns<Matrix5x4<T>, T> {
    fn reflect(self: Reflection5<T>, ref rhs: Matrix5x4<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x4 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m34: R::mul_add(f4, a.z, m.m34),
                m44: R::mul_add(f4, a.w, m.m44),
                m54: R::mul_add(f4, a.a, m.m54),
            };
    }

    fn reflect_with_sign(self: Reflection5<T>, ref rhs: Matrix5x4<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x4 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m34: R::sum_prod2(f4, a.z, sign, m.m34),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m54: R::sum_prod2(f4, a.a, sign, m.m54),
            };
    }
}

/// `Reflection5Columns` on `Matrix5`.
pub impl Reflection5ColumnsMatrix5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection5Columns<Matrix5<T>, T> {
    fn reflect(self: Reflection5<T>, ref rhs: Matrix5<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m34: R::mul_add(f4, a.z, m.m34),
                m44: R::mul_add(f4, a.w, m.m44),
                m54: R::mul_add(f4, a.a, m.m54),
                m15: R::mul_add(f5, a.x, m.m15),
                m25: R::mul_add(f5, a.y, m.m25),
                m35: R::mul_add(f5, a.z, m.m35),
                m45: R::mul_add(f5, a.w, m.m45),
                m55: R::mul_add(f5, a.a, m.m55),
            };
    }

    fn reflect_with_sign(self: Reflection5<T>, ref rhs: Matrix5<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m34: R::sum_prod2(f4, a.z, sign, m.m34),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m54: R::sum_prod2(f4, a.a, sign, m.m54),
                m15: R::sum_prod2(f5, a.x, sign, m.m15),
                m25: R::sum_prod2(f5, a.y, sign, m.m25),
                m35: R::sum_prod2(f5, a.z, sign, m.m35),
                m45: R::sum_prod2(f5, a.w, sign, m.m45),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
            };
    }
}

/// `Reflection5Columns` on `Matrix5x6`.
pub impl Reflection5ColumnsMatrix5x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection5Columns<Matrix5x6<T>, T> {
    fn reflect(self: Reflection5<T>, ref rhs: Matrix5x6<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m16);
        let w = R::wide_add_prod(w, a.y, m.m26);
        let w = R::wide_add_prod(w, a.z, m.m36);
        let w = R::wide_add_prod(w, a.w, m.m46);
        let w = R::wide_add_prod(w, a.a, m.m56);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m34: R::mul_add(f4, a.z, m.m34),
                m44: R::mul_add(f4, a.w, m.m44),
                m54: R::mul_add(f4, a.a, m.m54),
                m15: R::mul_add(f5, a.x, m.m15),
                m25: R::mul_add(f5, a.y, m.m25),
                m35: R::mul_add(f5, a.z, m.m35),
                m45: R::mul_add(f5, a.w, m.m45),
                m55: R::mul_add(f5, a.a, m.m55),
                m16: R::mul_add(f6, a.x, m.m16),
                m26: R::mul_add(f6, a.y, m.m26),
                m36: R::mul_add(f6, a.z, m.m36),
                m46: R::mul_add(f6, a.w, m.m46),
                m56: R::mul_add(f6, a.a, m.m56),
            };
    }

    fn reflect_with_sign(self: Reflection5<T>, ref rhs: Matrix5x6<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m16);
        let w = R::wide_add_prod(w, a.y, m.m26);
        let w = R::wide_add_prod(w, a.z, m.m36);
        let w = R::wide_add_prod(w, a.w, m.m46);
        let w = R::wide_add_prod(w, a.a, m.m56);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix5x6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m34: R::sum_prod2(f4, a.z, sign, m.m34),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m54: R::sum_prod2(f4, a.a, sign, m.m54),
                m15: R::sum_prod2(f5, a.x, sign, m.m15),
                m25: R::sum_prod2(f5, a.y, sign, m.m25),
                m35: R::sum_prod2(f5, a.z, sign, m.m35),
                m45: R::sum_prod2(f5, a.w, sign, m.m45),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
                m16: R::sum_prod2(f6, a.x, sign, m.m16),
                m26: R::sum_prod2(f6, a.y, sign, m.m26),
                m36: R::sum_prod2(f6, a.z, sign, m.m36),
                m46: R::sum_prod2(f6, a.w, sign, m.m46),
                m56: R::sum_prod2(f6, a.a, sign, m.m56),
            };
    }
}

/// `reflect_rows` / `reflect_rows_with_sign` of `Reflection5` on a matrix `L` with 5 columns and
/// its scratch vector `W`, updated in place.
pub trait Reflection5Rows<L, W, T> {
    /// Applies the reflection to the rows of `lhs`: `work` becomes `lhs * axis - bias`, then every
    /// row `x` of `lhs` becomes `x - 2 work_i axisᵀ`. Panics on overflow. Upstream:
    /// `Reflection::reflect_rows`.
    fn reflect_rows(self: Reflection5<T>, ref lhs: L, ref work: W);
    /// Applies the reflection to the rows of `lhs` with a sign: `work` becomes `lhs * axis - bias`,
    /// then every row `x` of `lhs` becomes `sign x - 2 sign work_i axisᵀ`. Panics on overflow.
    /// Upstream: `Reflection::reflect_rows_with_sign`.
    fn reflect_rows_with_sign(self: Reflection5<T>, ref lhs: L, ref work: W, sign: T);
}

/// `Reflection5Rows` on `RowVector5` with the scratch vector `Matrix1`.
pub impl Reflection5RowsRowVector5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection5Rows<RowVector5<T>, Matrix1<T>, T> {
    fn reflect_rows(self: Reflection5<T>, ref lhs: RowVector5<T>, ref work: Matrix1<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_add_prod(w, m.y, a.y);
        let w = R::wide_add_prod(w, m.z, a.z);
        let w = R::wide_add_prod(w, m.w, a.w);
        let w = R::wide_add_prod(w, m.a, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        lhs =
            RowVector5 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f1, a.y, m.y),
                z: R::mul_add(f1, a.z, m.z),
                w: R::mul_add(f1, a.w, m.w),
                a: R::mul_add(f1, a.a, m.a),
            };
        work = Matrix1 { x: s1 };
    }

    fn reflect_rows_with_sign(
        self: Reflection5<T>, ref lhs: RowVector5<T>, ref work: Matrix1<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_add_prod(w, m.y, a.y);
        let w = R::wide_add_prod(w, m.z, a.z);
        let w = R::wide_add_prod(w, m.w, a.w);
        let w = R::wide_add_prod(w, m.a, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        lhs =
            RowVector5 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f1, a.y, sign, m.y),
                z: R::sum_prod2(f1, a.z, sign, m.z),
                w: R::sum_prod2(f1, a.w, sign, m.w),
                a: R::sum_prod2(f1, a.a, sign, m.a),
            };
        work = Matrix1 { x: s1 };
    }
}

/// `Reflection5Rows` on `Matrix2x5` with the scratch vector `Vector2`.
pub impl Reflection5RowsMatrix2x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection5Rows<Matrix2x5<T>, Vector2<T>, T> {
    fn reflect_rows(self: Reflection5<T>, ref lhs: Matrix2x5<T>, ref work: Vector2<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        lhs =
            Matrix2x5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
            };
        work = Vector2 { x: s1, y: s2 };
    }

    fn reflect_rows_with_sign(
        self: Reflection5<T>, ref lhs: Matrix2x5<T>, ref work: Vector2<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        lhs =
            Matrix2x5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
            };
        work = Vector2 { x: s1, y: s2 };
    }
}

/// `Reflection5Rows` on `Matrix3x5` with the scratch vector `Vector3`.
pub impl Reflection5RowsMatrix3x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection5Rows<Matrix3x5<T>, Vector3<T>, T> {
    fn reflect_rows(self: Reflection5<T>, ref lhs: Matrix3x5<T>, ref work: Vector3<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        lhs =
            Matrix3x5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }

    fn reflect_rows_with_sign(
        self: Reflection5<T>, ref lhs: Matrix3x5<T>, ref work: Vector3<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        lhs =
            Matrix3x5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }
}

/// `Reflection5Rows` on `Matrix4x5` with the scratch vector `Vector4`.
pub impl Reflection5RowsMatrix4x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection5Rows<Matrix4x5<T>, Vector4<T>, T> {
    fn reflect_rows(self: Reflection5<T>, ref lhs: Matrix4x5<T>, ref work: Vector4<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        lhs =
            Matrix4x5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m43: R::mul_add(f4, a.z, m.m43),
                m44: R::mul_add(f4, a.w, m.m44),
                m45: R::mul_add(f4, a.a, m.m45),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }

    fn reflect_rows_with_sign(
        self: Reflection5<T>, ref lhs: Matrix4x5<T>, ref work: Vector4<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        lhs =
            Matrix4x5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m43: R::sum_prod2(f4, a.z, sign, m.m43),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m45: R::sum_prod2(f4, a.a, sign, m.m45),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }
}

/// `Reflection5Rows` on `Matrix5` with the scratch vector `Vector5`.
pub impl Reflection5RowsMatrix5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection5Rows<Matrix5<T>, Vector5<T>, T> {
    fn reflect_rows(self: Reflection5<T>, ref lhs: Matrix5<T>, ref work: Vector5<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        lhs =
            Matrix5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m43: R::mul_add(f4, a.z, m.m43),
                m44: R::mul_add(f4, a.w, m.m44),
                m45: R::mul_add(f4, a.a, m.m45),
                m51: R::mul_add(f5, a.x, m.m51),
                m52: R::mul_add(f5, a.y, m.m52),
                m53: R::mul_add(f5, a.z, m.m53),
                m54: R::mul_add(f5, a.w, m.m54),
                m55: R::mul_add(f5, a.a, m.m55),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }

    fn reflect_rows_with_sign(
        self: Reflection5<T>, ref lhs: Matrix5<T>, ref work: Vector5<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        lhs =
            Matrix5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m43: R::sum_prod2(f4, a.z, sign, m.m43),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m45: R::sum_prod2(f4, a.a, sign, m.m45),
                m51: R::sum_prod2(f5, a.x, sign, m.m51),
                m52: R::sum_prod2(f5, a.y, sign, m.m52),
                m53: R::sum_prod2(f5, a.z, sign, m.m53),
                m54: R::sum_prod2(f5, a.w, sign, m.m54),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }
}

/// `Reflection5Rows` on `Matrix6x5` with the scratch vector `Vector6`.
pub impl Reflection5RowsMatrix6x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection5Rows<Matrix6x5<T>, Vector6<T>, T> {
    fn reflect_rows(self: Reflection5<T>, ref lhs: Matrix6x5<T>, ref work: Vector6<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m61, a.x);
        let w = R::wide_add_prod(w, m.m62, a.y);
        let w = R::wide_add_prod(w, m.m63, a.z);
        let w = R::wide_add_prod(w, m.m64, a.w);
        let w = R::wide_add_prod(w, m.m65, a.a);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = -(s6 + s6);
        lhs =
            Matrix6x5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m43: R::mul_add(f4, a.z, m.m43),
                m44: R::mul_add(f4, a.w, m.m44),
                m45: R::mul_add(f4, a.a, m.m45),
                m51: R::mul_add(f5, a.x, m.m51),
                m52: R::mul_add(f5, a.y, m.m52),
                m53: R::mul_add(f5, a.z, m.m53),
                m54: R::mul_add(f5, a.w, m.m54),
                m55: R::mul_add(f5, a.a, m.m55),
                m61: R::mul_add(f6, a.x, m.m61),
                m62: R::mul_add(f6, a.y, m.m62),
                m63: R::mul_add(f6, a.z, m.m63),
                m64: R::mul_add(f6, a.w, m.m64),
                m65: R::mul_add(f6, a.a, m.m65),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }

    fn reflect_rows_with_sign(
        self: Reflection5<T>, ref lhs: Matrix6x5<T>, ref work: Vector6<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m61, a.x);
        let w = R::wide_add_prod(w, m.m62, a.y);
        let w = R::wide_add_prod(w, m.m63, a.z);
        let w = R::wide_add_prod(w, m.m64, a.w);
        let w = R::wide_add_prod(w, m.m65, a.a);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = m_two * s6;
        lhs =
            Matrix6x5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m43: R::sum_prod2(f4, a.z, sign, m.m43),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m45: R::sum_prod2(f4, a.a, sign, m.m45),
                m51: R::sum_prod2(f5, a.x, sign, m.m51),
                m52: R::sum_prod2(f5, a.y, sign, m.m52),
                m53: R::sum_prod2(f5, a.z, sign, m.m53),
                m54: R::sum_prod2(f5, a.w, sign, m.m54),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
                m61: R::sum_prod2(f6, a.x, sign, m.m61),
                m62: R::sum_prod2(f6, a.y, sign, m.m62),
                m63: R::sum_prod2(f6, a.z, sign, m.m63),
                m64: R::sum_prod2(f6, a.w, sign, m.m64),
                m65: R::sum_prod2(f6, a.a, sign, m.m65),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }
}
