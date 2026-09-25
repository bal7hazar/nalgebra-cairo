//! `Reflection6`: a reflection with respect to a hyperplane of the 6-dimensional space (upstream
//! `nalgebra::Reflection6`, i.e. `Reflection<T, Const<6>, ArrayStorage<T, 6, 1>>`), WP 8.4-P10.
//!
//! The hyperplane is orthogonal to `axis` (a unit vector) and lies at `bias` along it: the points
//! `p` with `axis . p = bias`. Written from one template for the sizes 1 to 6.
//!
//! Upstream's `reflect` / `reflect_rows` are generic over the shape of the matrix they update in
//! place (`&mut Matrix<T, R2, C2, S2>`); the Cairo counterparts are the generic traits
//! `Reflection6Columns<M, T>` (`reflect`, `reflect_with_sign`: `M` is any of the six shapes with 6
//! rows, `Vector6` to `Matrix6`) and `Reflection6Rows<L, W, T>` (`reflect_rows`,
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 6 columns and `W` the scratch vector
//! of its row count), both taking the matrix by `ref` like Rust's `&mut`. The scratch vector `work`
//! of `reflect_rows` holds `lhs * axis - bias` on return, as upstream's does.
//!
//! Numeric contract (AGENTS.md): every dot product is accumulated exactly and rounded ONCE, the
//! factor `-2 (axis . x - bias)` is rounded once (`wide_mul_scalar`), and every updated entry is
//! one fused `mul_add` / `sum_prod2` (one floor rounding and one overflow check). Overflow panics;
//! nothing wraps silently. Unlike upstream, the bias is subtracted unconditionally (upstream skips
//! it when zero: subtracting zero is exact, so the results are identical).

use simba::scalar::Real;
use crate::base::matrix1::Matrix1;
use crate::base::matrix2x6::Matrix2x6;
use crate::base::matrix3x6::Matrix3x6;
use crate::base::matrix4x6::Matrix4x6;
use crate::base::matrix5x6::Matrix5x6;
use crate::base::matrix6::Matrix6;
use crate::base::matrix6x2::Matrix6x2;
use crate::base::matrix6x3::Matrix6x3;
use crate::base::matrix6x4::Matrix6x4;
use crate::base::matrix6x5::Matrix6x5;
use crate::base::row_vector6::RowVector6;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use crate::base::vector6::Vector6;
use super::point6::Point6;

/// A reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along it.
/// The fields are private, like upstream's: build one with `new` / `new_containing_point`.
///
/// Like upstream's `Reflection`, the type derives no comparison, hash or serialization trait:
/// compare `axis()` and `bias()`. Cairo passes values by value, so `Copy` is implemented by hand
/// below (Cairo-imposed; upstream borrows).
#[derive(Drop)]
pub struct Reflection6<T> {
    axis: Vector6<T>,
    bias: T,
}

impl Reflection6Copy<T, +Copy<T>> of Copy<Reflection6<T>>;

/// Constructors and accessors of `Reflection6<T>` over a `Real` scalar.
#[generate_trait]
pub impl Reflection6Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Reflection6Trait<T> {
    /// The reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along
    /// it (a bias of zero is a plane through the origin). Exact. Upstream: `Reflection::new`.
    #[inline(always)]
    fn new(axis: Unit<Vector6<T>>, bias: T) -> Reflection6<T> {
        Reflection6 { axis: axis.value, bias }
    }

    /// The reflection with respect to the hyperplane orthogonal to `axis` that contains `pt`: `bias
    /// = axis . pt`, accumulated exactly and floored once. Panics on overflow. Upstream:
    /// `Reflection::new_containing_point`.
    #[inline(always)]
    fn new_containing_point(axis: Unit<Vector6<T>>, pt: Point6<T>) -> Reflection6<T> {
        let a = axis.value;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, pt.x);
        let w = R::wide_add_prod(w, a.y, pt.y);
        let w = R::wide_add_prod(w, a.z, pt.z);
        let w = R::wide_add_prod(w, a.w, pt.w);
        let w = R::wide_add_prod(w, a.a, pt.a);
        let w = R::wide_add_prod(w, a.b, pt.b);
        Reflection6 { axis: a, bias: R::wide_rescale(w) }
    }

    /// The reflection axis. Upstream: `Reflection::axis`.
    #[inline(always)]
    fn axis(self: Reflection6<T>) -> Vector6<T> {
        self.axis
    }

    /// The reflection bias: the position of the plane along the axis. Upstream: `Reflection::bias`.
    #[inline(always)]
    fn bias(self: Reflection6<T>) -> T {
        self.bias
    }
}

/// `reflect` / `reflect_with_sign` of `Reflection6` on a matrix `M` with 6 rows, updated in place.
pub trait Reflection6Columns<M, T> {
    /// Applies the reflection to the columns of `rhs`: every column `x` becomes `x - 2 (axis . x -
    /// bias) axis`. Panics on overflow. Upstream: `Reflection::reflect`.
    fn reflect(self: Reflection6<T>, ref rhs: M);
    /// Applies the reflection to the columns of `rhs` with a sign: every column `x` becomes `sign x
    /// - 2 sign (axis . x - bias) axis`. Panics on overflow. Upstream:
    /// `Reflection::reflect_with_sign`.
    fn reflect_with_sign(self: Reflection6<T>, ref rhs: M, sign: T);
}

/// `Reflection6Columns` on `Vector6`.
pub impl Reflection6ColumnsVector6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection6Columns<Vector6<T>, T> {
    fn reflect(self: Reflection6<T>, ref rhs: Vector6<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_add_prod(w, a.y, m.y);
        let w = R::wide_add_prod(w, a.z, m.z);
        let w = R::wide_add_prod(w, a.w, m.w);
        let w = R::wide_add_prod(w, a.a, m.a);
        let w = R::wide_add_prod(w, a.b, m.b);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs =
            Vector6 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f1, a.y, m.y),
                z: R::mul_add(f1, a.z, m.z),
                w: R::mul_add(f1, a.w, m.w),
                a: R::mul_add(f1, a.a, m.a),
                b: R::mul_add(f1, a.b, m.b),
            };
    }

    fn reflect_with_sign(self: Reflection6<T>, ref rhs: Vector6<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_add_prod(w, a.y, m.y);
        let w = R::wide_add_prod(w, a.z, m.z);
        let w = R::wide_add_prod(w, a.w, m.w);
        let w = R::wide_add_prod(w, a.a, m.a);
        let w = R::wide_add_prod(w, a.b, m.b);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs =
            Vector6 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f1, a.y, sign, m.y),
                z: R::sum_prod2(f1, a.z, sign, m.z),
                w: R::sum_prod2(f1, a.w, sign, m.w),
                a: R::sum_prod2(f1, a.a, sign, m.a),
                b: R::sum_prod2(f1, a.b, sign, m.b),
            };
    }
}

/// `Reflection6Columns` on `Matrix6x2`.
pub impl Reflection6ColumnsMatrix6x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection6Columns<Matrix6x2<T>, T> {
    fn reflect(self: Reflection6<T>, ref rhs: Matrix6x2<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m61: R::mul_add(f1, a.b, m.m61),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m62: R::mul_add(f2, a.b, m.m62),
            };
    }

    fn reflect_with_sign(self: Reflection6<T>, ref rhs: Matrix6x2<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m61: R::sum_prod2(f1, a.b, sign, m.m61),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m62: R::sum_prod2(f2, a.b, sign, m.m62),
            };
    }
}

/// `Reflection6Columns` on `Matrix6x3`.
pub impl Reflection6ColumnsMatrix6x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection6Columns<Matrix6x3<T>, T> {
    fn reflect(self: Reflection6<T>, ref rhs: Matrix6x3<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x3 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m61: R::mul_add(f1, a.b, m.m61),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m62: R::mul_add(f2, a.b, m.m62),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m63: R::mul_add(f3, a.b, m.m63),
            };
    }

    fn reflect_with_sign(self: Reflection6<T>, ref rhs: Matrix6x3<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x3 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m61: R::sum_prod2(f1, a.b, sign, m.m61),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m62: R::sum_prod2(f2, a.b, sign, m.m62),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m63: R::sum_prod2(f3, a.b, sign, m.m63),
            };
    }
}

/// `Reflection6Columns` on `Matrix6x4`.
pub impl Reflection6ColumnsMatrix6x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection6Columns<Matrix6x4<T>, T> {
    fn reflect(self: Reflection6<T>, ref rhs: Matrix6x4<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_add_prod(w, a.b, m.m64);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x4 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m61: R::mul_add(f1, a.b, m.m61),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m62: R::mul_add(f2, a.b, m.m62),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m63: R::mul_add(f3, a.b, m.m63),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m34: R::mul_add(f4, a.z, m.m34),
                m44: R::mul_add(f4, a.w, m.m44),
                m54: R::mul_add(f4, a.a, m.m54),
                m64: R::mul_add(f4, a.b, m.m64),
            };
    }

    fn reflect_with_sign(self: Reflection6<T>, ref rhs: Matrix6x4<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_add_prod(w, a.b, m.m64);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x4 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m61: R::sum_prod2(f1, a.b, sign, m.m61),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m62: R::sum_prod2(f2, a.b, sign, m.m62),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m63: R::sum_prod2(f3, a.b, sign, m.m63),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m34: R::sum_prod2(f4, a.z, sign, m.m34),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m54: R::sum_prod2(f4, a.a, sign, m.m54),
                m64: R::sum_prod2(f4, a.b, sign, m.m64),
            };
    }
}

/// `Reflection6Columns` on `Matrix6x5`.
pub impl Reflection6ColumnsMatrix6x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection6Columns<Matrix6x5<T>, T> {
    fn reflect(self: Reflection6<T>, ref rhs: Matrix6x5<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_add_prod(w, a.b, m.m64);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_add_prod(w, a.b, m.m65);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m61: R::mul_add(f1, a.b, m.m61),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m62: R::mul_add(f2, a.b, m.m62),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m63: R::mul_add(f3, a.b, m.m63),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m34: R::mul_add(f4, a.z, m.m34),
                m44: R::mul_add(f4, a.w, m.m44),
                m54: R::mul_add(f4, a.a, m.m54),
                m64: R::mul_add(f4, a.b, m.m64),
                m15: R::mul_add(f5, a.x, m.m15),
                m25: R::mul_add(f5, a.y, m.m25),
                m35: R::mul_add(f5, a.z, m.m35),
                m45: R::mul_add(f5, a.w, m.m45),
                m55: R::mul_add(f5, a.a, m.m55),
                m65: R::mul_add(f5, a.b, m.m65),
            };
    }

    fn reflect_with_sign(self: Reflection6<T>, ref rhs: Matrix6x5<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_add_prod(w, a.b, m.m64);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_add_prod(w, a.b, m.m65);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6x5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m61: R::sum_prod2(f1, a.b, sign, m.m61),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m62: R::sum_prod2(f2, a.b, sign, m.m62),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m63: R::sum_prod2(f3, a.b, sign, m.m63),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m34: R::sum_prod2(f4, a.z, sign, m.m34),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m54: R::sum_prod2(f4, a.a, sign, m.m54),
                m64: R::sum_prod2(f4, a.b, sign, m.m64),
                m15: R::sum_prod2(f5, a.x, sign, m.m15),
                m25: R::sum_prod2(f5, a.y, sign, m.m25),
                m35: R::sum_prod2(f5, a.z, sign, m.m35),
                m45: R::sum_prod2(f5, a.w, sign, m.m45),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
                m65: R::sum_prod2(f5, a.b, sign, m.m65),
            };
    }
}

/// `Reflection6Columns` on `Matrix6`.
pub impl Reflection6ColumnsMatrix6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection6Columns<Matrix6<T>, T> {
    fn reflect(self: Reflection6<T>, ref rhs: Matrix6<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_add_prod(w, a.b, m.m64);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_add_prod(w, a.b, m.m65);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m16);
        let w = R::wide_add_prod(w, a.y, m.m26);
        let w = R::wide_add_prod(w, a.z, m.m36);
        let w = R::wide_add_prod(w, a.w, m.m46);
        let w = R::wide_add_prod(w, a.a, m.m56);
        let w = R::wide_add_prod(w, a.b, m.m66);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m31: R::mul_add(f1, a.z, m.m31),
                m41: R::mul_add(f1, a.w, m.m41),
                m51: R::mul_add(f1, a.a, m.m51),
                m61: R::mul_add(f1, a.b, m.m61),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m32: R::mul_add(f2, a.z, m.m32),
                m42: R::mul_add(f2, a.w, m.m42),
                m52: R::mul_add(f2, a.a, m.m52),
                m62: R::mul_add(f2, a.b, m.m62),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m33: R::mul_add(f3, a.z, m.m33),
                m43: R::mul_add(f3, a.w, m.m43),
                m53: R::mul_add(f3, a.a, m.m53),
                m63: R::mul_add(f3, a.b, m.m63),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m34: R::mul_add(f4, a.z, m.m34),
                m44: R::mul_add(f4, a.w, m.m44),
                m54: R::mul_add(f4, a.a, m.m54),
                m64: R::mul_add(f4, a.b, m.m64),
                m15: R::mul_add(f5, a.x, m.m15),
                m25: R::mul_add(f5, a.y, m.m25),
                m35: R::mul_add(f5, a.z, m.m35),
                m45: R::mul_add(f5, a.w, m.m45),
                m55: R::mul_add(f5, a.a, m.m55),
                m65: R::mul_add(f5, a.b, m.m65),
                m16: R::mul_add(f6, a.x, m.m16),
                m26: R::mul_add(f6, a.y, m.m26),
                m36: R::mul_add(f6, a.z, m.m36),
                m46: R::mul_add(f6, a.w, m.m46),
                m56: R::mul_add(f6, a.a, m.m56),
                m66: R::mul_add(f6, a.b, m.m66),
            };
    }

    fn reflect_with_sign(self: Reflection6<T>, ref rhs: Matrix6<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_add_prod(w, a.z, m.m31);
        let w = R::wide_add_prod(w, a.w, m.m41);
        let w = R::wide_add_prod(w, a.a, m.m51);
        let w = R::wide_add_prod(w, a.b, m.m61);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_add_prod(w, a.z, m.m32);
        let w = R::wide_add_prod(w, a.w, m.m42);
        let w = R::wide_add_prod(w, a.a, m.m52);
        let w = R::wide_add_prod(w, a.b, m.m62);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_add_prod(w, a.z, m.m33);
        let w = R::wide_add_prod(w, a.w, m.m43);
        let w = R::wide_add_prod(w, a.a, m.m53);
        let w = R::wide_add_prod(w, a.b, m.m63);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_add_prod(w, a.z, m.m34);
        let w = R::wide_add_prod(w, a.w, m.m44);
        let w = R::wide_add_prod(w, a.a, m.m54);
        let w = R::wide_add_prod(w, a.b, m.m64);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_add_prod(w, a.z, m.m35);
        let w = R::wide_add_prod(w, a.w, m.m45);
        let w = R::wide_add_prod(w, a.a, m.m55);
        let w = R::wide_add_prod(w, a.b, m.m65);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m16);
        let w = R::wide_add_prod(w, a.y, m.m26);
        let w = R::wide_add_prod(w, a.z, m.m36);
        let w = R::wide_add_prod(w, a.w, m.m46);
        let w = R::wide_add_prod(w, a.a, m.m56);
        let w = R::wide_add_prod(w, a.b, m.m66);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m31: R::sum_prod2(f1, a.z, sign, m.m31),
                m41: R::sum_prod2(f1, a.w, sign, m.m41),
                m51: R::sum_prod2(f1, a.a, sign, m.m51),
                m61: R::sum_prod2(f1, a.b, sign, m.m61),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m32: R::sum_prod2(f2, a.z, sign, m.m32),
                m42: R::sum_prod2(f2, a.w, sign, m.m42),
                m52: R::sum_prod2(f2, a.a, sign, m.m52),
                m62: R::sum_prod2(f2, a.b, sign, m.m62),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m43: R::sum_prod2(f3, a.w, sign, m.m43),
                m53: R::sum_prod2(f3, a.a, sign, m.m53),
                m63: R::sum_prod2(f3, a.b, sign, m.m63),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m34: R::sum_prod2(f4, a.z, sign, m.m34),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m54: R::sum_prod2(f4, a.a, sign, m.m54),
                m64: R::sum_prod2(f4, a.b, sign, m.m64),
                m15: R::sum_prod2(f5, a.x, sign, m.m15),
                m25: R::sum_prod2(f5, a.y, sign, m.m25),
                m35: R::sum_prod2(f5, a.z, sign, m.m35),
                m45: R::sum_prod2(f5, a.w, sign, m.m45),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
                m65: R::sum_prod2(f5, a.b, sign, m.m65),
                m16: R::sum_prod2(f6, a.x, sign, m.m16),
                m26: R::sum_prod2(f6, a.y, sign, m.m26),
                m36: R::sum_prod2(f6, a.z, sign, m.m36),
                m46: R::sum_prod2(f6, a.w, sign, m.m46),
                m56: R::sum_prod2(f6, a.a, sign, m.m56),
                m66: R::sum_prod2(f6, a.b, sign, m.m66),
            };
    }
}

/// `reflect_rows` / `reflect_rows_with_sign` of `Reflection6` on a matrix `L` with 6 columns and
/// its scratch vector `W`, updated in place.
pub trait Reflection6Rows<L, W, T> {
    /// Applies the reflection to the rows of `lhs`: `work` becomes `lhs * axis - bias`, then every
    /// row `x` of `lhs` becomes `x - 2 work_i axisᵀ`. Panics on overflow. Upstream:
    /// `Reflection::reflect_rows`.
    fn reflect_rows(self: Reflection6<T>, ref lhs: L, ref work: W);
    /// Applies the reflection to the rows of `lhs` with a sign: `work` becomes `lhs * axis - bias`,
    /// then every row `x` of `lhs` becomes `sign x - 2 sign work_i axisᵀ`. Panics on overflow.
    /// Upstream: `Reflection::reflect_rows_with_sign`.
    fn reflect_rows_with_sign(self: Reflection6<T>, ref lhs: L, ref work: W, sign: T);
}

/// `Reflection6Rows` on `RowVector6` with the scratch vector `Matrix1`.
pub impl Reflection6RowsRowVector6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection6Rows<RowVector6<T>, Matrix1<T>, T> {
    fn reflect_rows(self: Reflection6<T>, ref lhs: RowVector6<T>, ref work: Matrix1<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_add_prod(w, m.y, a.y);
        let w = R::wide_add_prod(w, m.z, a.z);
        let w = R::wide_add_prod(w, m.w, a.w);
        let w = R::wide_add_prod(w, m.a, a.a);
        let w = R::wide_add_prod(w, m.b, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        lhs =
            RowVector6 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f1, a.y, m.y),
                z: R::mul_add(f1, a.z, m.z),
                w: R::mul_add(f1, a.w, m.w),
                a: R::mul_add(f1, a.a, m.a),
                b: R::mul_add(f1, a.b, m.b),
            };
        work = Matrix1 { x: s1 };
    }

    fn reflect_rows_with_sign(
        self: Reflection6<T>, ref lhs: RowVector6<T>, ref work: Matrix1<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_add_prod(w, m.y, a.y);
        let w = R::wide_add_prod(w, m.z, a.z);
        let w = R::wide_add_prod(w, m.w, a.w);
        let w = R::wide_add_prod(w, m.a, a.a);
        let w = R::wide_add_prod(w, m.b, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        lhs =
            RowVector6 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f1, a.y, sign, m.y),
                z: R::sum_prod2(f1, a.z, sign, m.z),
                w: R::sum_prod2(f1, a.w, sign, m.w),
                a: R::sum_prod2(f1, a.a, sign, m.a),
                b: R::sum_prod2(f1, a.b, sign, m.b),
            };
        work = Matrix1 { x: s1 };
    }
}

/// `Reflection6Rows` on `Matrix2x6` with the scratch vector `Vector2`.
pub impl Reflection6RowsMatrix2x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection6Rows<Matrix2x6<T>, Vector2<T>, T> {
    fn reflect_rows(self: Reflection6<T>, ref lhs: Matrix2x6<T>, ref work: Vector2<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        lhs =
            Matrix2x6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m16: R::mul_add(f1, a.b, m.m16),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m26: R::mul_add(f2, a.b, m.m26),
            };
        work = Vector2 { x: s1, y: s2 };
    }

    fn reflect_rows_with_sign(
        self: Reflection6<T>, ref lhs: Matrix2x6<T>, ref work: Vector2<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        lhs =
            Matrix2x6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m16: R::sum_prod2(f1, a.b, sign, m.m16),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m26: R::sum_prod2(f2, a.b, sign, m.m26),
            };
        work = Vector2 { x: s1, y: s2 };
    }
}

/// `Reflection6Rows` on `Matrix3x6` with the scratch vector `Vector3`.
pub impl Reflection6RowsMatrix3x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection6Rows<Matrix3x6<T>, Vector3<T>, T> {
    fn reflect_rows(self: Reflection6<T>, ref lhs: Matrix3x6<T>, ref work: Vector3<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        lhs =
            Matrix3x6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m16: R::mul_add(f1, a.b, m.m16),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m26: R::mul_add(f2, a.b, m.m26),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m36: R::mul_add(f3, a.b, m.m36),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }

    fn reflect_rows_with_sign(
        self: Reflection6<T>, ref lhs: Matrix3x6<T>, ref work: Vector3<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        lhs =
            Matrix3x6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m16: R::sum_prod2(f1, a.b, sign, m.m16),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m26: R::sum_prod2(f2, a.b, sign, m.m26),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m36: R::sum_prod2(f3, a.b, sign, m.m36),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }
}

/// `Reflection6Rows` on `Matrix4x6` with the scratch vector `Vector4`.
pub impl Reflection6RowsMatrix4x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection6Rows<Matrix4x6<T>, Vector4<T>, T> {
    fn reflect_rows(self: Reflection6<T>, ref lhs: Matrix4x6<T>, ref work: Vector4<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_add_prod(w, m.m46, a.b);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        lhs =
            Matrix4x6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m16: R::mul_add(f1, a.b, m.m16),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m26: R::mul_add(f2, a.b, m.m26),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m36: R::mul_add(f3, a.b, m.m36),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m43: R::mul_add(f4, a.z, m.m43),
                m44: R::mul_add(f4, a.w, m.m44),
                m45: R::mul_add(f4, a.a, m.m45),
                m46: R::mul_add(f4, a.b, m.m46),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }

    fn reflect_rows_with_sign(
        self: Reflection6<T>, ref lhs: Matrix4x6<T>, ref work: Vector4<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_add_prod(w, m.m46, a.b);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        lhs =
            Matrix4x6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m16: R::sum_prod2(f1, a.b, sign, m.m16),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m26: R::sum_prod2(f2, a.b, sign, m.m26),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m36: R::sum_prod2(f3, a.b, sign, m.m36),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m43: R::sum_prod2(f4, a.z, sign, m.m43),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m45: R::sum_prod2(f4, a.a, sign, m.m45),
                m46: R::sum_prod2(f4, a.b, sign, m.m46),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }
}

/// `Reflection6Rows` on `Matrix5x6` with the scratch vector `Vector5`.
pub impl Reflection6RowsMatrix5x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection6Rows<Matrix5x6<T>, Vector5<T>, T> {
    fn reflect_rows(self: Reflection6<T>, ref lhs: Matrix5x6<T>, ref work: Vector5<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_add_prod(w, m.m46, a.b);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_add_prod(w, m.m56, a.b);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        lhs =
            Matrix5x6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m16: R::mul_add(f1, a.b, m.m16),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m26: R::mul_add(f2, a.b, m.m26),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m36: R::mul_add(f3, a.b, m.m36),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m43: R::mul_add(f4, a.z, m.m43),
                m44: R::mul_add(f4, a.w, m.m44),
                m45: R::mul_add(f4, a.a, m.m45),
                m46: R::mul_add(f4, a.b, m.m46),
                m51: R::mul_add(f5, a.x, m.m51),
                m52: R::mul_add(f5, a.y, m.m52),
                m53: R::mul_add(f5, a.z, m.m53),
                m54: R::mul_add(f5, a.w, m.m54),
                m55: R::mul_add(f5, a.a, m.m55),
                m56: R::mul_add(f5, a.b, m.m56),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }

    fn reflect_rows_with_sign(
        self: Reflection6<T>, ref lhs: Matrix5x6<T>, ref work: Vector5<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_add_prod(w, m.m46, a.b);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_add_prod(w, m.m56, a.b);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        lhs =
            Matrix5x6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m16: R::sum_prod2(f1, a.b, sign, m.m16),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m26: R::sum_prod2(f2, a.b, sign, m.m26),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m36: R::sum_prod2(f3, a.b, sign, m.m36),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m43: R::sum_prod2(f4, a.z, sign, m.m43),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m45: R::sum_prod2(f4, a.a, sign, m.m45),
                m46: R::sum_prod2(f4, a.b, sign, m.m46),
                m51: R::sum_prod2(f5, a.x, sign, m.m51),
                m52: R::sum_prod2(f5, a.y, sign, m.m52),
                m53: R::sum_prod2(f5, a.z, sign, m.m53),
                m54: R::sum_prod2(f5, a.w, sign, m.m54),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
                m56: R::sum_prod2(f5, a.b, sign, m.m56),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }
}

/// `Reflection6Rows` on `Matrix6` with the scratch vector `Vector6`.
pub impl Reflection6RowsMatrix6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection6Rows<Matrix6<T>, Vector6<T>, T> {
    fn reflect_rows(self: Reflection6<T>, ref lhs: Matrix6<T>, ref work: Vector6<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_add_prod(w, m.m46, a.b);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_add_prod(w, m.m56, a.b);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m61, a.x);
        let w = R::wide_add_prod(w, m.m62, a.y);
        let w = R::wide_add_prod(w, m.m63, a.z);
        let w = R::wide_add_prod(w, m.m64, a.w);
        let w = R::wide_add_prod(w, m.m65, a.a);
        let w = R::wide_add_prod(w, m.m66, a.b);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = -(s6 + s6);
        lhs =
            Matrix6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m13: R::mul_add(f1, a.z, m.m13),
                m14: R::mul_add(f1, a.w, m.m14),
                m15: R::mul_add(f1, a.a, m.m15),
                m16: R::mul_add(f1, a.b, m.m16),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m23: R::mul_add(f2, a.z, m.m23),
                m24: R::mul_add(f2, a.w, m.m24),
                m25: R::mul_add(f2, a.a, m.m25),
                m26: R::mul_add(f2, a.b, m.m26),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m33: R::mul_add(f3, a.z, m.m33),
                m34: R::mul_add(f3, a.w, m.m34),
                m35: R::mul_add(f3, a.a, m.m35),
                m36: R::mul_add(f3, a.b, m.m36),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m43: R::mul_add(f4, a.z, m.m43),
                m44: R::mul_add(f4, a.w, m.m44),
                m45: R::mul_add(f4, a.a, m.m45),
                m46: R::mul_add(f4, a.b, m.m46),
                m51: R::mul_add(f5, a.x, m.m51),
                m52: R::mul_add(f5, a.y, m.m52),
                m53: R::mul_add(f5, a.z, m.m53),
                m54: R::mul_add(f5, a.w, m.m54),
                m55: R::mul_add(f5, a.a, m.m55),
                m56: R::mul_add(f5, a.b, m.m56),
                m61: R::mul_add(f6, a.x, m.m61),
                m62: R::mul_add(f6, a.y, m.m62),
                m63: R::mul_add(f6, a.z, m.m63),
                m64: R::mul_add(f6, a.w, m.m64),
                m65: R::mul_add(f6, a.a, m.m65),
                m66: R::mul_add(f6, a.b, m.m66),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }

    fn reflect_rows_with_sign(
        self: Reflection6<T>, ref lhs: Matrix6<T>, ref work: Vector6<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_add_prod(w, m.m13, a.z);
        let w = R::wide_add_prod(w, m.m14, a.w);
        let w = R::wide_add_prod(w, m.m15, a.a);
        let w = R::wide_add_prod(w, m.m16, a.b);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_add_prod(w, m.m23, a.z);
        let w = R::wide_add_prod(w, m.m24, a.w);
        let w = R::wide_add_prod(w, m.m25, a.a);
        let w = R::wide_add_prod(w, m.m26, a.b);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_add_prod(w, m.m33, a.z);
        let w = R::wide_add_prod(w, m.m34, a.w);
        let w = R::wide_add_prod(w, m.m35, a.a);
        let w = R::wide_add_prod(w, m.m36, a.b);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_add_prod(w, m.m43, a.z);
        let w = R::wide_add_prod(w, m.m44, a.w);
        let w = R::wide_add_prod(w, m.m45, a.a);
        let w = R::wide_add_prod(w, m.m46, a.b);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_add_prod(w, m.m53, a.z);
        let w = R::wide_add_prod(w, m.m54, a.w);
        let w = R::wide_add_prod(w, m.m55, a.a);
        let w = R::wide_add_prod(w, m.m56, a.b);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m61, a.x);
        let w = R::wide_add_prod(w, m.m62, a.y);
        let w = R::wide_add_prod(w, m.m63, a.z);
        let w = R::wide_add_prod(w, m.m64, a.w);
        let w = R::wide_add_prod(w, m.m65, a.a);
        let w = R::wide_add_prod(w, m.m66, a.b);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = m_two * s6;
        lhs =
            Matrix6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m13: R::sum_prod2(f1, a.z, sign, m.m13),
                m14: R::sum_prod2(f1, a.w, sign, m.m14),
                m15: R::sum_prod2(f1, a.a, sign, m.m15),
                m16: R::sum_prod2(f1, a.b, sign, m.m16),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m23: R::sum_prod2(f2, a.z, sign, m.m23),
                m24: R::sum_prod2(f2, a.w, sign, m.m24),
                m25: R::sum_prod2(f2, a.a, sign, m.m25),
                m26: R::sum_prod2(f2, a.b, sign, m.m26),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m33: R::sum_prod2(f3, a.z, sign, m.m33),
                m34: R::sum_prod2(f3, a.w, sign, m.m34),
                m35: R::sum_prod2(f3, a.a, sign, m.m35),
                m36: R::sum_prod2(f3, a.b, sign, m.m36),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m43: R::sum_prod2(f4, a.z, sign, m.m43),
                m44: R::sum_prod2(f4, a.w, sign, m.m44),
                m45: R::sum_prod2(f4, a.a, sign, m.m45),
                m46: R::sum_prod2(f4, a.b, sign, m.m46),
                m51: R::sum_prod2(f5, a.x, sign, m.m51),
                m52: R::sum_prod2(f5, a.y, sign, m.m52),
                m53: R::sum_prod2(f5, a.z, sign, m.m53),
                m54: R::sum_prod2(f5, a.w, sign, m.m54),
                m55: R::sum_prod2(f5, a.a, sign, m.m55),
                m56: R::sum_prod2(f5, a.b, sign, m.m56),
                m61: R::sum_prod2(f6, a.x, sign, m.m61),
                m62: R::sum_prod2(f6, a.y, sign, m.m62),
                m63: R::sum_prod2(f6, a.z, sign, m.m63),
                m64: R::sum_prod2(f6, a.w, sign, m.m64),
                m65: R::sum_prod2(f6, a.a, sign, m.m65),
                m66: R::sum_prod2(f6, a.b, sign, m.m66),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }
}
