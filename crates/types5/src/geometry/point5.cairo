//! In `nalgebra_types5`: the `struct` `Point5`; its 11 impls, among them `Point5Neg`,
//! `Point5AddAssign`, `Point5SubAssign` and 8 more. This module is split over packages; the other
//! parts are in `nalgebra_geometry5`.
//!
//! `Point5`: a 5-dimensional point (upstream `nalgebra::Point5`, i.e. `Point<T, 5>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Vector5`
//! (`x, y, z, w, a`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.

use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use nalgebra_core::geometry::point::errors as point_errors;
use simba::scalar::Real;
use crate::base::vector5::Vector5;

/// A point in the 5-dimensional space. The coordinates are fields, like `Point2` / `Point3`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point5<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
    pub a: T,
}

/// `-p`, coordinate-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl Point5Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Point5<T>> {
    #[inline(always)]
    fn neg(a: Point5<T>) -> Point5<T> {
        Point5 { x: -a.x, y: -a.y, z: -a.z, w: -a.w, a: -a.a }
    }
}

/// `p += v`. Exact; panics on overflow. Upstream: `AddAssign<Vector>`.
pub impl Point5AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Point5<T>, Vector5<T>> {
    #[inline(always)]
    fn add_assign(ref self: Point5<T>, rhs: Vector5<T>) {
        self =
            Point5 {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
                w: self.w + rhs.w,
                a: self.a + rhs.a,
            };
    }
}

/// `p -= v`. Exact; panics on overflow. Upstream: `SubAssign<Vector>`.
pub impl Point5SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Point5<T>, Vector5<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Point5<T>, rhs: Vector5<T>) {
        self =
            Point5 {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
                w: self.w - rhs.w,
                a: self.a - rhs.a,
            };
    }
}

/// `p *= k`: `scale` in place. Upstream: `MulAssign<T>`.
pub impl Point5MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Point5<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Point5<T>, rhs: T) {
        self =
            Point5 {
                x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs, a: self.a * rhs,
            };
    }
}

/// `p /= k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Point5DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Point5<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Point5<T>, rhs: T) {
        let (x, y, z, w, a) = R::div5(self.x, self.y, self.z, self.w, self.a, rhs);
        self = Point5 { x, y, z, w, a };
    }
}

/// `v.into()`: the point at the position `v`. Upstream: `From<Vector5> for Point5`.
pub impl Point5FromVector<T> of Into<Vector5<T>, Point5<T>> {
    #[inline(always)]
    fn into(self: Vector5<T>) -> Point5<T> {
        let Vector5 { x, y, z, w, a } = self;
        Point5 { x, y, z, w, a }
    }
}

/// The position vector of the point. Upstream: the `coords` field.
pub impl Point5IntoVector<T> of Into<Point5<T>, Vector5<T>> {
    #[inline(always)]
    fn into(self: Point5<T>) -> Vector5<T> {
        let Point5 { x, y, z, w, a } = self;
        Vector5 { x, y, z, w, a }
    }
}

/// `[x, y, z, w, a].into()`. Upstream: `From<[T; 5]>`.
pub impl Point5FromArray<T> of Into<[T; 5], Point5<T>> {
    #[inline(always)]
    fn into(self: [T; 5]) -> Point5<T> {
        let [x, y, z, w, a] = self;
        Point5 { x, y, z, w, a }
    }
}

/// The coordinates as an array `[x, y, z, w, a]`. Upstream: `Into<[T; 5]>`.
pub impl Point5IntoArray<T> of Into<Point5<T>, [T; 5]> {
    #[inline(always)]
    fn into(self: Point5<T>) -> [T; 5] {
        let Point5 { x, y, z, w, a } = self;
        [x, y, z, w, a]
    }
}

/// `p[i]`: the coordinate `i` (`x, y, z, w, a`). Panics with `nalgebra: index out of bounds` for
/// `i > 4`. Upstream: `Index<usize> for Point`.
pub impl Point5Index<T, +Copy<T>, +Drop<T>> of Index<Point5<T>, usize> {
    type Target = T;

    fn index(ref self: Point5<T>, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            4 => self.a,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point5PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point5<T>> {
    #[inline(always)]
    fn lt(lhs: Point5<T>, rhs: Point5<T>) -> bool {
        lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w && lhs.a < rhs.a
    }

    #[inline(always)]
    fn le(lhs: Point5<T>, rhs: Point5<T>) -> bool {
        lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w && lhs.a <= rhs.a
    }

    #[inline(always)]
    fn gt(lhs: Point5<T>, rhs: Point5<T>) -> bool {
        lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w && lhs.a > rhs.a
    }

    #[inline(always)]
    fn ge(lhs: Point5<T>, rhs: Point5<T>) -> bool {
        lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w && lhs.a >= rhs.a
    }
}
