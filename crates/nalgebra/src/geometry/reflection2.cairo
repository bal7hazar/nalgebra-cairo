//! `Reflection2`: a reflection with respect to a hyperplane of the 2-dimensional space (upstream
//! `nalgebra::Reflection2`, i.e. `Reflection<T, Const<2>, ArrayStorage<T, 2, 1>>`), WP 8.4-P10.
//!
//! The hyperplane is orthogonal to `axis` (a unit vector) and lies at `bias` along it: the points
//! `p` with `axis . p = bias`. Written from one template for the sizes 1 to 6.
//!
//! Upstream's `reflect` / `reflect_rows` are generic over the shape of the matrix they update in
//! place (`&mut Matrix<T, R2, C2, S2>`); the Cairo counterparts are the generic traits
//! `Reflection2Columns<M, T>` (`reflect`, `reflect_with_sign`: `M` is any of the six shapes with 2
//! rows, `Vector2` to `Matrix2x6`) and `Reflection2Rows<L, W, T>` (`reflect_rows`,
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 2 columns and `W` the scratch vector
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
use crate::base::matrix2::Matrix2;
use crate::base::matrix2x3::Matrix2x3;
use crate::base::matrix2x4::Matrix2x4;
use crate::base::matrix2x5::Matrix2x5;
use crate::base::matrix2x6::Matrix2x6;
use crate::base::matrix3x2::Matrix3x2;
use crate::base::matrix4x2::Matrix4x2;
use crate::base::matrix5x2::Matrix5x2;
use crate::base::matrix6x2::Matrix6x2;
use crate::base::point2::Point2;
use crate::base::row_vector2::RowVector2;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use crate::base::vector6::Vector6;

/// A reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along it.
/// The fields are private, like upstream's: build one with `new` / `new_containing_point`.
///
/// Like upstream's `Reflection`, the type derives no comparison, hash or serialization trait:
/// compare `axis()` and `bias()`. Cairo passes values by value, so `Copy` is implemented by hand
/// below (Cairo-imposed; upstream borrows).
#[derive(Drop)]
pub struct Reflection2<T> {
    axis: Vector2<T>,
    bias: T,
}

impl Reflection2Copy<T, +Copy<T>> of Copy<Reflection2<T>>;

/// Constructors and accessors of `Reflection2<T>` over a `Real` scalar.
#[generate_trait]
pub impl Reflection2Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Reflection2Trait<T> {
    /// The reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along
    /// it (a bias of zero is a plane through the origin). Exact. Upstream: `Reflection::new`.
    #[inline(always)]
    fn new(axis: Unit<Vector2<T>>, bias: T) -> Reflection2<T> {
        Reflection2 { axis: axis.value, bias }
    }

    /// The reflection with respect to the hyperplane orthogonal to `axis` that contains `pt`: `bias
    /// = axis . pt`, accumulated exactly and floored once. Panics on overflow. Upstream:
    /// `Reflection::new_containing_point`.
    #[inline(always)]
    fn new_containing_point(axis: Unit<Vector2<T>>, pt: Point2<T>) -> Reflection2<T> {
        let a = axis.value;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, pt.x);
        let w = R::wide_add_prod(w, a.y, pt.y);
        Reflection2 { axis: a, bias: R::wide_rescale(w) }
    }

    /// The reflection axis. Upstream: `Reflection::axis`.
    #[inline(always)]
    fn axis(self: Reflection2<T>) -> Vector2<T> {
        self.axis
    }

    /// The reflection bias: the position of the plane along the axis. Upstream: `Reflection::bias`.
    #[inline(always)]
    fn bias(self: Reflection2<T>) -> T {
        self.bias
    }
}

/// `reflect` / `reflect_with_sign` of `Reflection2` on a matrix `M` with 2 rows, updated in place.
pub trait Reflection2Columns<M, T> {
    /// Applies the reflection to the columns of `rhs`: every column `x` becomes `x - 2 (axis . x -
    /// bias) axis`. Panics on overflow. Upstream: `Reflection::reflect`.
    fn reflect(self: Reflection2<T>, ref rhs: M);
    /// Applies the reflection to the columns of `rhs` with a sign: every column `x` becomes `sign x
    /// - 2 sign (axis . x - bias) axis`. Panics on overflow. Upstream:
    /// `Reflection::reflect_with_sign`.
    fn reflect_with_sign(self: Reflection2<T>, ref rhs: M, sign: T);
}

/// `Reflection2Columns` on `Vector2`.
pub impl Reflection2ColumnsVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection2Columns<Vector2<T>, T> {
    fn reflect(self: Reflection2<T>, ref rhs: Vector2<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_add_prod(w, a.y, m.y);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs = Vector2 { x: R::mul_add(f1, a.x, m.x), y: R::mul_add(f1, a.y, m.y) };
    }

    fn reflect_with_sign(self: Reflection2<T>, ref rhs: Vector2<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_add_prod(w, a.y, m.y);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs = Vector2 { x: R::sum_prod2(f1, a.x, sign, m.x), y: R::sum_prod2(f1, a.y, sign, m.y) };
    }
}

/// `Reflection2Columns` on `Matrix2`.
pub impl Reflection2ColumnsMatrix2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection2Columns<Matrix2<T>, T> {
    fn reflect(self: Reflection2<T>, ref rhs: Matrix2<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
            };
    }

    fn reflect_with_sign(self: Reflection2<T>, ref rhs: Matrix2<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
            };
    }
}

/// `Reflection2Columns` on `Matrix2x3`.
pub impl Reflection2ColumnsMatrix2x3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection2Columns<Matrix2x3<T>, T> {
    fn reflect(self: Reflection2<T>, ref rhs: Matrix2x3<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x3 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
            };
    }

    fn reflect_with_sign(self: Reflection2<T>, ref rhs: Matrix2x3<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x3 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
            };
    }
}

/// `Reflection2Columns` on `Matrix2x4`.
pub impl Reflection2ColumnsMatrix2x4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection2Columns<Matrix2x4<T>, T> {
    fn reflect(self: Reflection2<T>, ref rhs: Matrix2x4<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x4 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
            };
    }

    fn reflect_with_sign(self: Reflection2<T>, ref rhs: Matrix2x4<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x4 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
            };
    }
}

/// `Reflection2Columns` on `Matrix2x5`.
pub impl Reflection2ColumnsMatrix2x5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection2Columns<Matrix2x5<T>, T> {
    fn reflect(self: Reflection2<T>, ref rhs: Matrix2x5<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x5 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m15: R::mul_add(f5, a.x, m.m15),
                m25: R::mul_add(f5, a.y, m.m25),
            };
    }

    fn reflect_with_sign(self: Reflection2<T>, ref rhs: Matrix2x5<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x5 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m15: R::sum_prod2(f5, a.x, sign, m.m15),
                m25: R::sum_prod2(f5, a.y, sign, m.m25),
            };
    }
}

/// `Reflection2Columns` on `Matrix2x6`.
pub impl Reflection2ColumnsMatrix2x6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection2Columns<Matrix2x6<T>, T> {
    fn reflect(self: Reflection2<T>, ref rhs: Matrix2x6<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m16);
        let w = R::wide_add_prod(w, a.y, m.m26);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x6 {
                m11: R::mul_add(f1, a.x, m.m11),
                m21: R::mul_add(f1, a.y, m.m21),
                m12: R::mul_add(f2, a.x, m.m12),
                m22: R::mul_add(f2, a.y, m.m22),
                m13: R::mul_add(f3, a.x, m.m13),
                m23: R::mul_add(f3, a.y, m.m23),
                m14: R::mul_add(f4, a.x, m.m14),
                m24: R::mul_add(f4, a.y, m.m24),
                m15: R::mul_add(f5, a.x, m.m15),
                m25: R::mul_add(f5, a.y, m.m25),
                m16: R::mul_add(f6, a.x, m.m16),
                m26: R::mul_add(f6, a.y, m.m26),
            };
    }

    fn reflect_with_sign(self: Reflection2<T>, ref rhs: Matrix2x6<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m11);
        let w = R::wide_add_prod(w, a.y, m.m21);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m12);
        let w = R::wide_add_prod(w, a.y, m.m22);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m13);
        let w = R::wide_add_prod(w, a.y, m.m23);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m14);
        let w = R::wide_add_prod(w, a.y, m.m24);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m15);
        let w = R::wide_add_prod(w, a.y, m.m25);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.m16);
        let w = R::wide_add_prod(w, a.y, m.m26);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            Matrix2x6 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m21: R::sum_prod2(f1, a.y, sign, m.m21),
                m12: R::sum_prod2(f2, a.x, sign, m.m12),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m13: R::sum_prod2(f3, a.x, sign, m.m13),
                m23: R::sum_prod2(f3, a.y, sign, m.m23),
                m14: R::sum_prod2(f4, a.x, sign, m.m14),
                m24: R::sum_prod2(f4, a.y, sign, m.m24),
                m15: R::sum_prod2(f5, a.x, sign, m.m15),
                m25: R::sum_prod2(f5, a.y, sign, m.m25),
                m16: R::sum_prod2(f6, a.x, sign, m.m16),
                m26: R::sum_prod2(f6, a.y, sign, m.m26),
            };
    }
}

/// `reflect_rows` / `reflect_rows_with_sign` of `Reflection2` on a matrix `L` with 2 columns and
/// its scratch vector `W`, updated in place.
pub trait Reflection2Rows<L, W, T> {
    /// Applies the reflection to the rows of `lhs`: `work` becomes `lhs * axis - bias`, then every
    /// row `x` of `lhs` becomes `x - 2 work_i axisᵀ`. Panics on overflow. Upstream:
    /// `Reflection::reflect_rows`.
    fn reflect_rows(self: Reflection2<T>, ref lhs: L, ref work: W);
    /// Applies the reflection to the rows of `lhs` with a sign: `work` becomes `lhs * axis - bias`,
    /// then every row `x` of `lhs` becomes `sign x - 2 sign work_i axisᵀ`. Panics on overflow.
    /// Upstream: `Reflection::reflect_rows_with_sign`.
    fn reflect_rows_with_sign(self: Reflection2<T>, ref lhs: L, ref work: W, sign: T);
}

/// `Reflection2Rows` on `RowVector2` with the scratch vector `Matrix1`.
pub impl Reflection2RowsRowVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection2Rows<RowVector2<T>, Matrix1<T>, T> {
    fn reflect_rows(self: Reflection2<T>, ref lhs: RowVector2<T>, ref work: Matrix1<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_add_prod(w, m.y, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        lhs = RowVector2 { x: R::mul_add(f1, a.x, m.x), y: R::mul_add(f1, a.y, m.y) };
        work = Matrix1 { x: s1 };
    }

    fn reflect_rows_with_sign(
        self: Reflection2<T>, ref lhs: RowVector2<T>, ref work: Matrix1<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_add_prod(w, m.y, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        lhs =
            RowVector2 { x: R::sum_prod2(f1, a.x, sign, m.x), y: R::sum_prod2(f1, a.y, sign, m.y) };
        work = Matrix1 { x: s1 };
    }
}

/// `Reflection2Rows` on `Matrix2` with the scratch vector `Vector2`.
pub impl Reflection2RowsMatrix2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection2Rows<Matrix2<T>, Vector2<T>, T> {
    fn reflect_rows(self: Reflection2<T>, ref lhs: Matrix2<T>, ref work: Vector2<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        lhs =
            Matrix2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
            };
        work = Vector2 { x: s1, y: s2 };
    }

    fn reflect_rows_with_sign(
        self: Reflection2<T>, ref lhs: Matrix2<T>, ref work: Vector2<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        lhs =
            Matrix2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
            };
        work = Vector2 { x: s1, y: s2 };
    }
}

/// `Reflection2Rows` on `Matrix3x2` with the scratch vector `Vector3`.
pub impl Reflection2RowsMatrix3x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection2Rows<Matrix3x2<T>, Vector3<T>, T> {
    fn reflect_rows(self: Reflection2<T>, ref lhs: Matrix3x2<T>, ref work: Vector3<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        lhs =
            Matrix3x2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }

    fn reflect_rows_with_sign(
        self: Reflection2<T>, ref lhs: Matrix3x2<T>, ref work: Vector3<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        lhs =
            Matrix3x2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }
}

/// `Reflection2Rows` on `Matrix4x2` with the scratch vector `Vector4`.
pub impl Reflection2RowsMatrix4x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection2Rows<Matrix4x2<T>, Vector4<T>, T> {
    fn reflect_rows(self: Reflection2<T>, ref lhs: Matrix4x2<T>, ref work: Vector4<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        lhs =
            Matrix4x2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }

    fn reflect_rows_with_sign(
        self: Reflection2<T>, ref lhs: Matrix4x2<T>, ref work: Vector4<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        lhs =
            Matrix4x2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }
}

/// `Reflection2Rows` on `Matrix5x2` with the scratch vector `Vector5`.
pub impl Reflection2RowsMatrix5x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection2Rows<Matrix5x2<T>, Vector5<T>, T> {
    fn reflect_rows(self: Reflection2<T>, ref lhs: Matrix5x2<T>, ref work: Vector5<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        lhs =
            Matrix5x2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m51: R::mul_add(f5, a.x, m.m51),
                m52: R::mul_add(f5, a.y, m.m52),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }

    fn reflect_rows_with_sign(
        self: Reflection2<T>, ref lhs: Matrix5x2<T>, ref work: Vector5<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        lhs =
            Matrix5x2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m51: R::sum_prod2(f5, a.x, sign, m.m51),
                m52: R::sum_prod2(f5, a.y, sign, m.m52),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }
}

/// `Reflection2Rows` on `Matrix6x2` with the scratch vector `Vector6`.
pub impl Reflection2RowsMatrix6x2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection2Rows<Matrix6x2<T>, Vector6<T>, T> {
    fn reflect_rows(self: Reflection2<T>, ref lhs: Matrix6x2<T>, ref work: Vector6<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m61, a.x);
        let w = R::wide_add_prod(w, m.m62, a.y);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = -(s6 + s6);
        lhs =
            Matrix6x2 {
                m11: R::mul_add(f1, a.x, m.m11),
                m12: R::mul_add(f1, a.y, m.m12),
                m21: R::mul_add(f2, a.x, m.m21),
                m22: R::mul_add(f2, a.y, m.m22),
                m31: R::mul_add(f3, a.x, m.m31),
                m32: R::mul_add(f3, a.y, m.m32),
                m41: R::mul_add(f4, a.x, m.m41),
                m42: R::mul_add(f4, a.y, m.m42),
                m51: R::mul_add(f5, a.x, m.m51),
                m52: R::mul_add(f5, a.y, m.m52),
                m61: R::mul_add(f6, a.x, m.m61),
                m62: R::mul_add(f6, a.y, m.m62),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }

    fn reflect_rows_with_sign(
        self: Reflection2<T>, ref lhs: Matrix6x2<T>, ref work: Vector6<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m11, a.x);
        let w = R::wide_add_prod(w, m.m12, a.y);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m21, a.x);
        let w = R::wide_add_prod(w, m.m22, a.y);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m31, a.x);
        let w = R::wide_add_prod(w, m.m32, a.y);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m41, a.x);
        let w = R::wide_add_prod(w, m.m42, a.y);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m51, a.x);
        let w = R::wide_add_prod(w, m.m52, a.y);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.m61, a.x);
        let w = R::wide_add_prod(w, m.m62, a.y);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = m_two * s6;
        lhs =
            Matrix6x2 {
                m11: R::sum_prod2(f1, a.x, sign, m.m11),
                m12: R::sum_prod2(f1, a.y, sign, m.m12),
                m21: R::sum_prod2(f2, a.x, sign, m.m21),
                m22: R::sum_prod2(f2, a.y, sign, m.m22),
                m31: R::sum_prod2(f3, a.x, sign, m.m31),
                m32: R::sum_prod2(f3, a.y, sign, m.m32),
                m41: R::sum_prod2(f4, a.x, sign, m.m41),
                m42: R::sum_prod2(f4, a.y, sign, m.m42),
                m51: R::sum_prod2(f5, a.x, sign, m.m51),
                m52: R::sum_prod2(f5, a.y, sign, m.m52),
                m61: R::sum_prod2(f6, a.x, sign, m.m61),
                m62: R::sum_prod2(f6, a.y, sign, m.m62),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }
}
