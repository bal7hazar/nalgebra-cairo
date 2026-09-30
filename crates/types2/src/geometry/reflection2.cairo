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

use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::base::unit::Unit;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::point2::Point2;
use crate::base::row_vector2::RowVector2;
use crate::base::vector2::Vector2;

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
