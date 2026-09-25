//! `Reflection1`: a reflection with respect to a hyperplane of the 1-dimensional space (upstream
//! `nalgebra::Reflection1`, i.e. `Reflection<T, Const<1>, ArrayStorage<T, 1, 1>>`), WP 8.4-P10.
//!
//! The hyperplane is orthogonal to `axis` (a unit vector) and lies at `bias` along it: the points
//! `p` with `axis . p = bias`. Written from one template for the sizes 1 to 6.
//!
//! Upstream's `reflect` / `reflect_rows` are generic over the shape of the matrix they update in
//! place (`&mut Matrix<T, R2, C2, S2>`); the Cairo counterparts are the generic traits
//! `Reflection1Columns<M, T>` (`reflect`, `reflect_with_sign`: `M` is any of the six shapes with 1
//! rows, `Matrix1` to `RowVector6`) and `Reflection1Rows<L, W, T>` (`reflect_rows`,
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 1 columns and `W` the scratch
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
use crate::base::row_vector2::RowVector2;
use crate::base::row_vector3::RowVector3;
use crate::base::row_vector4::RowVector4;
use crate::base::row_vector5::RowVector5;
use crate::base::row_vector6::RowVector6;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use crate::base::vector6::Vector6;
use super::point1::Point1;

/// A reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along it.
/// The fields are private, like upstream's: build one with `new` / `new_containing_point`.
///
/// Like upstream's `Reflection`, the type derives no comparison, hash or serialization trait:
/// compare `axis()` and `bias()`. Cairo passes values by value, so `Copy` is implemented by hand
/// below (Cairo-imposed; upstream borrows).
#[derive(Drop)]
pub struct Reflection1<T> {
    axis: Matrix1<T>,
    bias: T,
}

impl Reflection1Copy<T, +Copy<T>> of Copy<Reflection1<T>>;

/// Constructors and accessors of `Reflection1<T>` over a `Real` scalar.
#[generate_trait]
pub impl Reflection1Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Reflection1Trait<T> {
    /// The reflection with respect to the hyperplane orthogonal to `axis` at position `bias` along
    /// it (a bias of zero is a plane through the origin). Exact. Upstream: `Reflection::new`.
    #[inline(always)]
    fn new(axis: Unit<Matrix1<T>>, bias: T) -> Reflection1<T> {
        Reflection1 { axis: axis.value, bias }
    }

    /// The reflection with respect to the hyperplane orthogonal to `axis` that contains `pt`:
    /// `bias = axis . pt`, accumulated exactly and floored once. Panics on overflow. Upstream:
    /// `Reflection::new_containing_point`.
    #[inline(always)]
    fn new_containing_point(axis: Unit<Matrix1<T>>, pt: Point1<T>) -> Reflection1<T> {
        let a = axis.value;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, pt.x);
        Reflection1 { axis: a, bias: R::wide_rescale(w) }
    }

    /// The reflection axis. Upstream: `Reflection::axis`.
    #[inline(always)]
    fn axis(self: Reflection1<T>) -> Matrix1<T> {
        self.axis
    }

    /// The reflection bias: the position of the plane along the axis. Upstream:
    /// `Reflection::bias`.
    #[inline(always)]
    fn bias(self: Reflection1<T>) -> T {
        self.bias
    }
}

/// `reflect` / `reflect_with_sign` of `Reflection1` on a matrix `M` with 1 rows, updated in place.
pub trait Reflection1Columns<M, T> {
    /// Applies the reflection to the columns of `rhs`: every column `x` becomes `x - 2 (axis . x -
    /// bias) axis`. Panics on overflow. Upstream: `Reflection::reflect`.
    fn reflect(self: Reflection1<T>, ref rhs: M);
    /// Applies the reflection to the columns of `rhs` with a sign: every column `x` becomes
    /// `sign x - 2 sign (axis . x - bias) axis`. Panics on overflow. Upstream:
    /// `Reflection::reflect_with_sign`.
    fn reflect_with_sign(self: Reflection1<T>, ref rhs: M, sign: T);
}

/// `Reflection1Columns` on `Matrix1`.
pub impl Reflection1ColumnsMatrix1<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection1Columns<Matrix1<T>, T> {
    fn reflect(self: Reflection1<T>, ref rhs: Matrix1<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs = Matrix1 { x: R::mul_add(f1, a.x, m.x) };
    }

    fn reflect_with_sign(self: Reflection1<T>, ref rhs: Matrix1<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        rhs = Matrix1 { x: R::sum_prod2(f1, a.x, sign, m.x) };
    }
}

/// `Reflection1Columns` on `RowVector2`.
pub impl Reflection1ColumnsRowVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection1Columns<RowVector2<T>, T> {
    fn reflect(self: Reflection1<T>, ref rhs: RowVector2<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs = RowVector2 { x: R::mul_add(f1, a.x, m.x), y: R::mul_add(f2, a.x, m.y) };
    }

    fn reflect_with_sign(self: Reflection1<T>, ref rhs: RowVector2<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector2 { x: R::sum_prod2(f1, a.x, sign, m.x), y: R::sum_prod2(f2, a.x, sign, m.y) };
    }
}

/// `Reflection1Columns` on `RowVector3`.
pub impl Reflection1ColumnsRowVector3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection1Columns<RowVector3<T>, T> {
    fn reflect(self: Reflection1<T>, ref rhs: RowVector3<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector3 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
            };
    }

    fn reflect_with_sign(self: Reflection1<T>, ref rhs: RowVector3<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector3 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
            };
    }
}

/// `Reflection1Columns` on `RowVector4`.
pub impl Reflection1ColumnsRowVector4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection1Columns<RowVector4<T>, T> {
    fn reflect(self: Reflection1<T>, ref rhs: RowVector4<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.w);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector4 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
                w: R::mul_add(f4, a.x, m.w),
            };
    }

    fn reflect_with_sign(self: Reflection1<T>, ref rhs: RowVector4<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.w);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector4 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
                w: R::sum_prod2(f4, a.x, sign, m.w),
            };
    }
}

/// `Reflection1Columns` on `RowVector5`.
pub impl Reflection1ColumnsRowVector5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection1Columns<RowVector5<T>, T> {
    fn reflect(self: Reflection1<T>, ref rhs: RowVector5<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.w);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.a);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector5 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
                w: R::mul_add(f4, a.x, m.w),
                a: R::mul_add(f5, a.x, m.a),
            };
    }

    fn reflect_with_sign(self: Reflection1<T>, ref rhs: RowVector5<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.w);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.a);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector5 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
                w: R::sum_prod2(f4, a.x, sign, m.w),
                a: R::sum_prod2(f5, a.x, sign, m.a),
            };
    }
}

/// `Reflection1Columns` on `RowVector6`.
pub impl Reflection1ColumnsRowVector6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Neg<T>,
> of Reflection1Columns<RowVector6<T>, T> {
    fn reflect(self: Reflection1<T>, ref rhs: RowVector6<T>) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -R::TWO;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.w);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.a);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.b);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector6 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
                w: R::mul_add(f4, a.x, m.w),
                a: R::mul_add(f5, a.x, m.a),
                b: R::mul_add(f6, a.x, m.b),
            };
    }

    fn reflect_with_sign(self: Reflection1<T>, ref rhs: RowVector6<T>, sign: T) {
        let (a, b, m) = (self.axis, self.bias, rhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.x);
        let w = R::wide_sub(w, b);
        let f1 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.y);
        let w = R::wide_sub(w, b);
        let f2 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.z);
        let w = R::wide_sub(w, b);
        let f3 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.w);
        let w = R::wide_sub(w, b);
        let f4 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.a);
        let w = R::wide_sub(w, b);
        let f5 = R::wide_mul_scalar(w, m_two);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, a.x, m.b);
        let w = R::wide_sub(w, b);
        let f6 = R::wide_mul_scalar(w, m_two);
        rhs =
            RowVector6 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
                w: R::sum_prod2(f4, a.x, sign, m.w),
                a: R::sum_prod2(f5, a.x, sign, m.a),
                b: R::sum_prod2(f6, a.x, sign, m.b),
            };
    }
}

/// `reflect_rows` / `reflect_rows_with_sign` of `Reflection1` on a matrix `L` with 1 columns and
/// its scratch vector `W`, updated in place.
pub trait Reflection1Rows<L, W, T> {
    /// Applies the reflection to the rows of `lhs`: `work` becomes `lhs * axis - bias`, then every
    /// row `x` of `lhs` becomes `x - 2 work_i axisᵀ`. Panics on overflow. Upstream:
    /// `Reflection::reflect_rows`.
    fn reflect_rows(self: Reflection1<T>, ref lhs: L, ref work: W);
    /// Applies the reflection to the rows of `lhs` with a sign: `work` becomes `lhs * axis - bias`,
    /// then every row `x` of `lhs` becomes `sign x - 2 sign work_i axisᵀ`. Panics on overflow.
    /// Upstream: `Reflection::reflect_rows_with_sign`.
    fn reflect_rows_with_sign(self: Reflection1<T>, ref lhs: L, ref work: W, sign: T);
}

/// `Reflection1Rows` on `Matrix1` with the scratch vector `Matrix1`.
pub impl Reflection1RowsMatrix1<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection1Rows<Matrix1<T>, Matrix1<T>, T> {
    fn reflect_rows(self: Reflection1<T>, ref lhs: Matrix1<T>, ref work: Matrix1<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        lhs = Matrix1 { x: R::mul_add(f1, a.x, m.x) };
        work = Matrix1 { x: s1 };
    }

    fn reflect_rows_with_sign(
        self: Reflection1<T>, ref lhs: Matrix1<T>, ref work: Matrix1<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        lhs = Matrix1 { x: R::sum_prod2(f1, a.x, sign, m.x) };
        work = Matrix1 { x: s1 };
    }
}

/// `Reflection1Rows` on `Vector2` with the scratch vector `Vector2`.
pub impl Reflection1RowsVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection1Rows<Vector2<T>, Vector2<T>, T> {
    fn reflect_rows(self: Reflection1<T>, ref lhs: Vector2<T>, ref work: Vector2<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        lhs = Vector2 { x: R::mul_add(f1, a.x, m.x), y: R::mul_add(f2, a.x, m.y) };
        work = Vector2 { x: s1, y: s2 };
    }

    fn reflect_rows_with_sign(
        self: Reflection1<T>, ref lhs: Vector2<T>, ref work: Vector2<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        lhs = Vector2 { x: R::sum_prod2(f1, a.x, sign, m.x), y: R::sum_prod2(f2, a.x, sign, m.y) };
        work = Vector2 { x: s1, y: s2 };
    }
}

/// `Reflection1Rows` on `Vector3` with the scratch vector `Vector3`.
pub impl Reflection1RowsVector3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection1Rows<Vector3<T>, Vector3<T>, T> {
    fn reflect_rows(self: Reflection1<T>, ref lhs: Vector3<T>, ref work: Vector3<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        lhs =
            Vector3 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }

    fn reflect_rows_with_sign(
        self: Reflection1<T>, ref lhs: Vector3<T>, ref work: Vector3<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        lhs =
            Vector3 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
            };
        work = Vector3 { x: s1, y: s2, z: s3 };
    }
}

/// `Reflection1Rows` on `Vector4` with the scratch vector `Vector4`.
pub impl Reflection1RowsVector4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection1Rows<Vector4<T>, Vector4<T>, T> {
    fn reflect_rows(self: Reflection1<T>, ref lhs: Vector4<T>, ref work: Vector4<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.w, a.x);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        lhs =
            Vector4 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
                w: R::mul_add(f4, a.x, m.w),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }

    fn reflect_rows_with_sign(
        self: Reflection1<T>, ref lhs: Vector4<T>, ref work: Vector4<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.w, a.x);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        lhs =
            Vector4 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
                w: R::sum_prod2(f4, a.x, sign, m.w),
            };
        work = Vector4 { x: s1, y: s2, z: s3, w: s4 };
    }
}

/// `Reflection1Rows` on `Vector5` with the scratch vector `Vector5`.
pub impl Reflection1RowsVector5<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection1Rows<Vector5<T>, Vector5<T>, T> {
    fn reflect_rows(self: Reflection1<T>, ref lhs: Vector5<T>, ref work: Vector5<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.w, a.x);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.a, a.x);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        lhs =
            Vector5 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
                w: R::mul_add(f4, a.x, m.w),
                a: R::mul_add(f5, a.x, m.a),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }

    fn reflect_rows_with_sign(
        self: Reflection1<T>, ref lhs: Vector5<T>, ref work: Vector5<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.w, a.x);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.a, a.x);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        lhs =
            Vector5 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
                w: R::sum_prod2(f4, a.x, sign, m.w),
                a: R::sum_prod2(f5, a.x, sign, m.a),
            };
        work = Vector5 { x: s1, y: s2, z: s3, w: s4, a: s5 };
    }
}

/// `Reflection1Rows` on `Vector6` with the scratch vector `Vector6`.
pub impl Reflection1RowsVector6<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Mul<T>, +Neg<T>,
> of Reflection1Rows<Vector6<T>, Vector6<T>, T> {
    fn reflect_rows(self: Reflection1<T>, ref lhs: Vector6<T>, ref work: Vector6<T>) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = -(s1 + s1);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = -(s2 + s2);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = -(s3 + s3);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.w, a.x);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = -(s4 + s4);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.a, a.x);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = -(s5 + s5);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.b, a.x);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = -(s6 + s6);
        lhs =
            Vector6 {
                x: R::mul_add(f1, a.x, m.x),
                y: R::mul_add(f2, a.x, m.y),
                z: R::mul_add(f3, a.x, m.z),
                w: R::mul_add(f4, a.x, m.w),
                a: R::mul_add(f5, a.x, m.a),
                b: R::mul_add(f6, a.x, m.b),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }

    fn reflect_rows_with_sign(
        self: Reflection1<T>, ref lhs: Vector6<T>, ref work: Vector6<T>, sign: T,
    ) {
        let (a, b, m) = (self.axis, self.bias, lhs);
        let m_two = -(sign + sign);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.x, a.x);
        let w = R::wide_sub(w, b);
        let s1 = R::wide_rescale(w);
        let f1 = m_two * s1;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.y, a.x);
        let w = R::wide_sub(w, b);
        let s2 = R::wide_rescale(w);
        let f2 = m_two * s2;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.z, a.x);
        let w = R::wide_sub(w, b);
        let s3 = R::wide_rescale(w);
        let f3 = m_two * s3;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.w, a.x);
        let w = R::wide_sub(w, b);
        let s4 = R::wide_rescale(w);
        let f4 = m_two * s4;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.a, a.x);
        let w = R::wide_sub(w, b);
        let s5 = R::wide_rescale(w);
        let f5 = m_two * s5;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, m.b, a.x);
        let w = R::wide_sub(w, b);
        let s6 = R::wide_rescale(w);
        let f6 = m_two * s6;
        lhs =
            Vector6 {
                x: R::sum_prod2(f1, a.x, sign, m.x),
                y: R::sum_prod2(f2, a.x, sign, m.y),
                z: R::sum_prod2(f3, a.x, sign, m.z),
                w: R::sum_prod2(f4, a.x, sign, m.w),
                a: R::sum_prod2(f5, a.x, sign, m.a),
                b: R::sum_prod2(f6, a.x, sign, m.b),
            };
        work = Vector6 { x: s1, y: s2, z: s3, w: s4, a: s5, b: s6 };
    }
}
