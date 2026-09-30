//! In `nalgebra_types6`: the `struct` `Point6`; its 11 impls, among them `Point6Neg`,
//! `Point6AddAssign`, `Point6SubAssign` and 8 more. This module is split over packages; the other
//! parts are in `nalgebra_geometry6`.
//!
//! `Point6`: a 6-dimensional point (upstream `nalgebra::Point6`, i.e. `Point<T, 6>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Vector6`
//! (`x, y, z, w, a, b`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//! There is no `to_homogeneous` / `from_homogeneous`: the homogeneous coordinates of a 6D point
//! form a 7-vector, and the static shapes stop at 6 (DESIGN D4). Out of scope for 0.1.0 by owner
//! ruling (issue #41).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.

use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use nalgebra_core::geometry::point::errors as point_errors;
use simba::scalar::Real;
use crate::base::vector6::Vector6;

/// A point in the 6-dimensional space. The coordinates are fields, like `Point2` / `Point3`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point6<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
    pub a: T,
    pub b: T,
}

/// `-p`, coordinate-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl Point6Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Point6<T>> {
    #[inline(always)]
    fn neg(a: Point6<T>) -> Point6<T> {
        Point6 { x: -a.x, y: -a.y, z: -a.z, w: -a.w, a: -a.a, b: -a.b }
    }
}

/// `p += v`. Exact; panics on overflow. Upstream: `AddAssign<Vector>`.
pub impl Point6AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Point6<T>, Vector6<T>> {
    #[inline(always)]
    fn add_assign(ref self: Point6<T>, rhs: Vector6<T>) {
        self =
            Point6 {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
                w: self.w + rhs.w,
                a: self.a + rhs.a,
                b: self.b + rhs.b,
            };
    }
}

/// `p -= v`. Exact; panics on overflow. Upstream: `SubAssign<Vector>`.
pub impl Point6SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Point6<T>, Vector6<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Point6<T>, rhs: Vector6<T>) {
        self =
            Point6 {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
                w: self.w - rhs.w,
                a: self.a - rhs.a,
                b: self.b - rhs.b,
            };
    }
}

/// `p *= k`: `scale` in place. Upstream: `MulAssign<T>`.
pub impl Point6MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Point6<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Point6<T>, rhs: T) {
        self =
            Point6 {
                x: self.x * rhs,
                y: self.y * rhs,
                z: self.z * rhs,
                w: self.w * rhs,
                a: self.a * rhs,
                b: self.b * rhs,
            };
    }
}

/// `p /= k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Point6DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Point6<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Point6<T>, rhs: T) {
        let (x, y, z, w, a, b) = R::div6(self.x, self.y, self.z, self.w, self.a, self.b, rhs);
        self = Point6 { x, y, z, w, a, b };
    }
}

/// `v.into()`: the point at the position `v`. Upstream: `From<Vector6> for Point6`.
pub impl Point6FromVector<T> of Into<Vector6<T>, Point6<T>> {
    #[inline(always)]
    fn into(self: Vector6<T>) -> Point6<T> {
        let Vector6 { x, y, z, w, a, b } = self;
        Point6 { x, y, z, w, a, b }
    }
}

/// The position vector of the point. Upstream: the `coords` field.
pub impl Point6IntoVector<T> of Into<Point6<T>, Vector6<T>> {
    #[inline(always)]
    fn into(self: Point6<T>) -> Vector6<T> {
        let Point6 { x, y, z, w, a, b } = self;
        Vector6 { x, y, z, w, a, b }
    }
}

/// `[x, y, z, w, a, b].into()`. Upstream: `From<[T; 6]>`.
pub impl Point6FromArray<T> of Into<[T; 6], Point6<T>> {
    #[inline(always)]
    fn into(self: [T; 6]) -> Point6<T> {
        let [x, y, z, w, a, b] = self;
        Point6 { x, y, z, w, a, b }
    }
}

/// The coordinates as an array `[x, y, z, w, a, b]`. Upstream: `Into<[T; 6]>`.
pub impl Point6IntoArray<T> of Into<Point6<T>, [T; 6]> {
    #[inline(always)]
    fn into(self: Point6<T>) -> [T; 6] {
        let Point6 { x, y, z, w, a, b } = self;
        [x, y, z, w, a, b]
    }
}

/// `p[i]`: the coordinate `i` (`x, y, z, w, a, b`). Panics with `nalgebra: index out of bounds` for
/// `i > 5`. Upstream: `Index<usize> for Point`.
pub impl Point6Index<T, +Copy<T>, +Drop<T>> of Index<Point6<T>, usize> {
    type Target = T;

    fn index(ref self: Point6<T>, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            4 => self.a,
            5 => self.b,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point6PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point6<T>> {
    #[inline(always)]
    fn lt(lhs: Point6<T>, rhs: Point6<T>) -> bool {
        lhs.x < rhs.x
            && lhs.y < rhs.y
            && lhs.z < rhs.z
            && lhs.w < rhs.w
            && lhs.a < rhs.a
            && lhs.b < rhs.b
    }

    #[inline(always)]
    fn le(lhs: Point6<T>, rhs: Point6<T>) -> bool {
        lhs.x <= rhs.x
            && lhs.y <= rhs.y
            && lhs.z <= rhs.z
            && lhs.w <= rhs.w
            && lhs.a <= rhs.a
            && lhs.b <= rhs.b
    }

    #[inline(always)]
    fn gt(lhs: Point6<T>, rhs: Point6<T>) -> bool {
        lhs.x > rhs.x
            && lhs.y > rhs.y
            && lhs.z > rhs.z
            && lhs.w > rhs.w
            && lhs.a > rhs.a
            && lhs.b > rhs.b
    }

    #[inline(always)]
    fn ge(lhs: Point6<T>, rhs: Point6<T>) -> bool {
        lhs.x >= rhs.x
            && lhs.y >= rhs.y
            && lhs.z >= rhs.z
            && lhs.w >= rhs.w
            && lhs.a >= rhs.a
            && lhs.b >= rhs.b
    }
}
