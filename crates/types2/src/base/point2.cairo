//! `Point2`: a 2-dimensional point (upstream `nalgebra::Point2`, which lives in `geometry`).
//!
//! A point is an affine position, as opposed to a `Vector2` displacement: points cannot be added
//! together, and `p - q` is a vector. Corelib's `Add` / `Sub` are homogeneous (`T + T -> T`), so
//! the heterogeneous operations are named methods:
//!
//! | upstream | here |
//! |---|---|
//! | `p - q` (a vector) | `p.sub_point(q)` |
//! | `p + v`, `p - v` | `p.add_vector(v)`, `p.sub_vector(v)` (and `p += v`, `p -= v`) |
//! | `p * k`, `p / k` | `p.scale(k)`, `p.unscale(k)` (and `p *= k`, `p /= k`) |
//! | `-p`, `Point::from(v)`, `p.coords` | `-p`, `v.into()` / `from_coordinates`, `p.coords()` |
//!
//! - `Point2Trait` / `Point2Impl`: constructors, conversions, affine operations and
//!   interpolation, generic over a `simba::scalar::Real` scalar (the distances and the midpoint
//!   are upstream's crate-root free functions, crate-internal kernels here until they are ported);
//! - operators and conversions from / to `[T; 2]` and `Vector2<T>`: their impls live in
//!   this module, where the compiler finds them without any import.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use nalgebra_core::geometry::point::errors as point_errors;
use simba::scalar::Real;
use crate::base::vector2::Vector2;

/// A 2-dimensional point.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

/// `-p`, coordinate-wise (the point mirrored through the origin). Exact; panics on overflow
/// (`-MIN`). Upstream: `Neg`.
pub impl Point2Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Point2<T>> {
    #[inline(always)]
    fn neg(a: Point2<T>) -> Point2<T> {
        Point2 { x: -a.x, y: -a.y }
    }
}

/// `p += v`: translation by a vector, in place. Exact; panics on overflow. Upstream:
/// `AddAssign<Vector>`.
pub impl Point2AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Point2<T>, Vector2<T>> {
    #[inline(always)]
    fn add_assign(ref self: Point2<T>, rhs: Vector2<T>) {
        self = Point2 { x: self.x + rhs.x, y: self.y + rhs.y };
    }
}

/// `p -= v`: translation by `-v`, in place. Exact; panics on overflow. Upstream:
/// `SubAssign<Vector>`.
pub impl Point2SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Point2<T>, Vector2<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Point2<T>, rhs: Vector2<T>) {
        self = Point2 { x: self.x - rhs.x, y: self.y - rhs.y };
    }
}

/// `p *= k` for a scalar `k`: `scale` in place. Upstream: `MulAssign<T>`.
pub impl Point2MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Point2<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Point2<T>, rhs: T) {
        self = Point2 { x: self.x * rhs, y: self.y * rhs };
    }
}

/// `p /= k` for a scalar `k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Point2DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Point2<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Point2<T>, rhs: T) {
        self = Point2 { x: R::div(self.x, rhs), y: R::div(self.y, rhs) };
    }
}

/// `v.into()`: the point at the position `v`. Upstream: `From<Vector2> for Point2`.
pub impl Point2FromVector<T> of Into<Vector2<T>, Point2<T>> {
    #[inline(always)]
    fn into(self: Vector2<T>) -> Point2<T> {
        let Vector2 { x, y } = self;
        Point2 { x, y }
    }
}

/// The position vector of the point. Upstream: the `coords` field.
pub impl Point2IntoVector<T> of Into<Point2<T>, Vector2<T>> {
    #[inline(always)]
    fn into(self: Point2<T>) -> Vector2<T> {
        let Point2 { x, y } = self;
        Vector2 { x, y }
    }
}

/// `[x, y].into()`. Upstream: `From<[T; 2]>`.
pub impl Point2FromArray<T> of Into<[T; 2], Point2<T>> {
    #[inline(always)]
    fn into(self: [T; 2]) -> Point2<T> {
        let [x, y] = self;
        Point2 { x, y }
    }
}

/// The coordinates as a fixed-size array `[x, y]`. Upstream: `Into<[T; 2]>`.
pub impl Point2IntoArray<T> of Into<Point2<T>, [T; 2]> {
    #[inline(always)]
    fn into(self: Point2<T>) -> [T; 2] {
        let Point2 { x, y } = self;
        [x, y]
    }
}

/// `p[i]`: the coordinate `i` (`x, y`). Panics with `nalgebra: index out of bounds` for
/// `i > 1`. Upstream: `Index<usize> for Point`.
pub impl Point2Index<T, +Copy<T>, +Drop<T>> of Index<Point2<T>, usize> {
    type Target = T;

    fn index(ref self: Point2<T>, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point2PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point2<T>> {
    #[inline(always)]
    fn lt(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x < rhs.x && lhs.y < rhs.y
    }

    #[inline(always)]
    fn le(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x <= rhs.x && lhs.y <= rhs.y
    }

    #[inline(always)]
    fn gt(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x > rhs.x && lhs.y > rhs.y
    }

    #[inline(always)]
    fn ge(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x >= rhs.x && lhs.y >= rhs.y
    }
}
