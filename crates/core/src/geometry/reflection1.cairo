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
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 1 columns and `W` the scratch vector
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
use crate::base::unit::Unit;
use crate::geometry::point1::Point1;

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

    /// The reflection with respect to the hyperplane orthogonal to `axis` that contains `pt`: `bias
    /// = axis . pt`, accumulated exactly and floored once. Panics on overflow. Upstream:
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

    /// The reflection bias: the position of the plane along the axis. Upstream: `Reflection::bias`.
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
    /// Applies the reflection to the columns of `rhs` with a sign: every column `x` becomes `sign x
    /// - 2 sign (axis . x - bias) axis`. Panics on overflow. Upstream:
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
