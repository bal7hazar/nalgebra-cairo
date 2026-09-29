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

use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use simba::scalar::Real;
use crate::base::matrix1::Matrix1;
use crate::geometry::point::errors as point_errors;

/// A point in the 1-dimensional space. The coordinates are fields, like `Point2` / `Point3`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point1<T> {
    pub x: T,
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
