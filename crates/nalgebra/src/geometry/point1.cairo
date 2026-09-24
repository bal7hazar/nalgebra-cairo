//! `Point1`: a 1-dimensional point (upstream `nalgebra::Point1`, i.e. `Point<T, 1>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Matrix1`
//! (`x`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.

use core::num::traits::Bounded;
use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use simba::scalar::Real;
use crate::base::matrix1::Matrix1;
use crate::base::vector2::Vector2;
use super::point::errors as point_errors;
use super::quaternion::ApproxEqTrait;

/// A point in the 1-dimensional space. The coordinates are fields, like `Point2` / `Point3`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point1<T> {
    pub x: T,
}

/// Operations of `Point1<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Point1Impl<
    T,
    impl R: Real<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<T>,
    +Drop<T>,
> of Point1Trait<T> {
    /// The point of coordinates `(x)`. Exact. Upstream: `Point1::new`.
    #[inline(always)]
    fn new(x: T) -> Point1<T> {
        Point1 { x }
    }

    /// The origin. Exact. Upstream: `origin`.
    #[inline(always)]
    fn origin() -> Point1<T> {
        Point1 { x: R::zero() }
    }

    /// The point at the position `v`. Exact. Upstream: `from_coordinates` (deprecated upstream for
    /// `From`), also `v.into()`.
    #[inline(always)]
    fn from_coordinates(v: Matrix1<T>) -> Point1<T> {
        Point1 { x: v.x }
    }

    /// The position vector. Upstream: the `coords` field.
    #[inline(always)]
    fn coords(self: Point1<T>) -> Matrix1<T> {
        Matrix1 { x: self.x }
    }

    /// The homogeneous coordinates `(x, 1)`. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Point1<T>) -> Vector2<T> {
        Vector2 { x: self.x, y: R::one() }
    }

    /// The point of homogeneous coordinates `v`: the first 1 divided by the last (correctly
    /// rounded divisions), or `None` when the last is zero. Upstream: `from_homogeneous`.
    #[inline(always)]
    fn from_homogeneous(v: Vector2<T>) -> Option<Point1<T>> {
        if v.y == R::zero() {
            None
        } else {
            let x = R::div(v.x, v.y);
            Some(Point1 { x })
        }
    }

    /// `self - rhs`, a vector. Exact; panics on overflow. Upstream: `Sub<Point>`.
    #[inline(always)]
    fn sub_point(self: Point1<T>, rhs: Point1<T>) -> Matrix1<T> {
        Matrix1 { x: self.x - rhs.x }
    }

    /// `self + v`. Exact; panics on overflow. Upstream: `Add<Vector>`.
    #[inline(always)]
    fn add_vector(self: Point1<T>, v: Matrix1<T>) -> Point1<T> {
        Point1 { x: self.x + v.x }
    }

    /// `self - v`. Exact; panics on overflow. Upstream: `Sub<Vector>`.
    #[inline(always)]
    fn sub_vector(self: Point1<T>, v: Matrix1<T>) -> Point1<T> {
        Point1 { x: self.x - v.x }
    }

    /// `self * k`: every coordinate floored once. Upstream: `Mul<T>` (and `k * p`).
    #[inline(always)]
    fn scale(self: Point1<T>, k: T) -> Point1<T> {
        Point1 { x: self.x * k }
    }

    /// `self / k`: correctly rounded divisions. Panics with `Fixed: division by zero` for
    /// `k = 0`. Upstream: `Div<T>`.
    #[inline(always)]
    fn unscale(self: Point1<T>, k: T) -> Point1<T> {
        let x = R::div(self.x, k);
        Point1 { x }
    }

    /// The coordinate-wise minimum. Upstream: `inf`.
    #[inline(always)]
    fn inf(self: Point1<T>, other: Point1<T>) -> Point1<T> {
        Point1 { x: R::min(self.x, other.x) }
    }

    /// The coordinate-wise maximum. Upstream: `sup`.
    #[inline(always)]
    fn sup(self: Point1<T>, other: Point1<T>) -> Point1<T> {
        Point1 { x: R::max(self.x, other.x) }
    }

    /// `(inf, sup)`. Upstream: `inf_sup`.
    #[inline(always)]
    fn inf_sup(self: Point1<T>, other: Point1<T>) -> (Point1<T>, Point1<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    /// `true` when every coordinate is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Point1<T>, other: Point1<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps)
    }

    /// `self + (rhs - self) · t`, one fused kernel per coordinate (`Real::lerp`). Upstream:
    /// `lerp`.
    #[inline(always)]
    fn lerp(self: Point1<T>, rhs: Point1<T>, t: T) -> Point1<T> {
        Point1 { x: R::lerp(self.x, rhs.x, t) }
    }

    /// `true` when every coordinate is `relative_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Point1<T>, other: Point1<T>, epsilon: u64, max_relative: T) -> bool {
        ApproxEqTrait::relative_eq(self.x, other.x, epsilon, max_relative)
    }

    /// `true` when every coordinate is `ulps_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Point1<T>, other: Point1<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.x, other.x, epsilon, max_ulps)
    }

    /// The same point with every coordinate converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Point<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Point1<T>) -> Point1<U> {
        Point1 { x: self.x.into() }
    }

    /// The point of coordinates `s[0], .., s[0]`. Panics with `nalgebra: wrong slice length`
    /// unless `s` holds exactly 1 elements (upstream's `from_row_slice` assertion). Upstream:
    /// `from_slice`.
    fn from_slice(s: Span<T>) -> Point1<T> {
        if s.len() != 1 {
            core::panic_with_felt252(point_errors::WRONG_SLICE_LENGTH);
        }
        Point1 { x: *s[0] }
    }

    /// The number of coordinates, 1. Upstream: `len`.
    #[inline(always)]
    fn len(self: Point1<T>) -> usize {
        1
    }

    /// `false`: a point has 1 coordinate. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: Point1<T>) -> bool {
        false
    }

    /// Deprecated upstream: the distance between two consecutive coordinates in storage, 1.
    /// Upstream: `stride`.
    #[inline(always)]
    fn stride(self: Point1<T>) -> usize {
        1
    }

    /// The point whose coordinates are all the scalar's `MIN` (core `Bounded`). Upstream:
    /// `Bounded::min_value` (num's spelling; Cairo's `Bounded` holds constants).
    fn min_value<+Bounded<T>>() -> Point1<T> {
        Point1 { x: Bounded::MIN }
    }

    /// The point whose coordinates are all the scalar's `MAX` (core `Bounded`). Upstream:
    /// `Bounded::max_value`.
    fn max_value<+Bounded<T>>() -> Point1<T> {
        Point1 { x: Bounded::MAX }
    }
}

/// `-p`, coordinate-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl Point1Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Point1<T>> {
    #[inline(always)]
    fn neg(a: Point1<T>) -> Point1<T> {
        Point1 { x: -a.x }
    }
}

/// `p += v`. Exact; panics on overflow. Upstream: `AddAssign<Vector>`.
pub impl Point1AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Point1<T>, Matrix1<T>> {
    #[inline(always)]
    fn add_assign(ref self: Point1<T>, rhs: Matrix1<T>) {
        self = Point1 { x: self.x + rhs.x };
    }
}

/// `p -= v`. Exact; panics on overflow. Upstream: `SubAssign<Vector>`.
pub impl Point1SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Point1<T>, Matrix1<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Point1<T>, rhs: Matrix1<T>) {
        self = Point1 { x: self.x - rhs.x };
    }
}

/// `p *= k`: `scale` in place. Upstream: `MulAssign<T>`.
pub impl Point1MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Point1<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Point1<T>, rhs: T) {
        self = Point1 { x: self.x * rhs };
    }
}

/// `p /= k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Point1DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Point1<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Point1<T>, rhs: T) {
        let x = R::div(self.x, rhs);
        self = Point1 { x };
    }
}

/// `v.into()`: the point at the position `v`. Upstream: `From<Matrix1> for Point1`.
pub impl Point1FromVector<T> of Into<Matrix1<T>, Point1<T>> {
    #[inline(always)]
    fn into(self: Matrix1<T>) -> Point1<T> {
        let Matrix1 { x } = self;
        Point1 { x }
    }
}

/// The position vector of the point. Upstream: the `coords` field.
pub impl Point1IntoVector<T> of Into<Point1<T>, Matrix1<T>> {
    #[inline(always)]
    fn into(self: Point1<T>) -> Matrix1<T> {
        let Point1 { x } = self;
        Matrix1 { x }
    }
}

/// `[x].into()`. Upstream: `From<[T; 1]>`.
pub impl Point1FromArray<T> of Into<[T; 1], Point1<T>> {
    #[inline(always)]
    fn into(self: [T; 1]) -> Point1<T> {
        let [x] = self;
        Point1 { x }
    }
}

/// The coordinates as an array `[x]`. Upstream: `Into<[T; 1]>`.
pub impl Point1IntoArray<T> of Into<Point1<T>, [T; 1]> {
    #[inline(always)]
    fn into(self: Point1<T>) -> [T; 1] {
        let Point1 { x } = self;
        [x]
    }
}

/// `p[i]`: the coordinate `i` (`x`). Panics with `nalgebra: index out of bounds` for
/// `i > 0`. Upstream: `Index<usize> for Point`.
pub impl Point1Index<T, +Copy<T>, +Drop<T>> of Index<Point1<T>, usize> {
    type Target = T;

    fn index(ref self: Point1<T>, index: usize) -> T {
        match index {
            0 => self.x,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point1PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point1<T>> {
    #[inline(always)]
    fn lt(lhs: Point1<T>, rhs: Point1<T>) -> bool {
        lhs.x < rhs.x
    }

    #[inline(always)]
    fn le(lhs: Point1<T>, rhs: Point1<T>) -> bool {
        lhs.x <= rhs.x
    }

    #[inline(always)]
    fn gt(lhs: Point1<T>, rhs: Point1<T>) -> bool {
        lhs.x > rhs.x
    }

    #[inline(always)]
    fn ge(lhs: Point1<T>, rhs: Point1<T>) -> bool {
        lhs.x >= rhs.x
    }
}
