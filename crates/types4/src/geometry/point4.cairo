//! `Point4`: a 4-dimensional point (upstream `nalgebra::Point4`, i.e. `Point<T, 4>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Vector4`
//! (`x, y, z, w`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.

use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use nalgebra_core::geometry::point::errors as point_errors;
use simba::scalar::Real;
use crate::base::vector4::Vector4;

/// A point in the 4-dimensional space. The coordinates are fields, like `Point2` / `Point3`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

/// `-p`, coordinate-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl Point4Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Point4<T>> {
    #[inline(always)]
    fn neg(a: Point4<T>) -> Point4<T> {
        Point4 { x: -a.x, y: -a.y, z: -a.z, w: -a.w }
    }
}

/// `p += v`. Exact; panics on overflow. Upstream: `AddAssign<Vector>`.
pub impl Point4AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Point4<T>, Vector4<T>> {
    #[inline(always)]
    fn add_assign(ref self: Point4<T>, rhs: Vector4<T>) {
        self =
            Point4 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w };
    }
}

/// `p -= v`. Exact; panics on overflow. Upstream: `SubAssign<Vector>`.
pub impl Point4SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Point4<T>, Vector4<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Point4<T>, rhs: Vector4<T>) {
        self =
            Point4 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w };
    }
}

/// `p *= k`: `scale` in place. Upstream: `MulAssign<T>`.
pub impl Point4MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Point4<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Point4<T>, rhs: T) {
        self = Point4 { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs };
    }
}

/// `p /= k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Point4DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Point4<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Point4<T>, rhs: T) {
        let (x, y, z, w) = R::div4(self.x, self.y, self.z, self.w, rhs);
        self = Point4 { x, y, z, w };
    }
}

/// `v.into()`: the point at the position `v`. Upstream: `From<Vector4> for Point4`.
pub impl Point4FromVector<T> of Into<Vector4<T>, Point4<T>> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> Point4<T> {
        let Vector4 { x, y, z, w } = self;
        Point4 { x, y, z, w }
    }
}

/// The position vector of the point. Upstream: the `coords` field.
pub impl Point4IntoVector<T> of Into<Point4<T>, Vector4<T>> {
    #[inline(always)]
    fn into(self: Point4<T>) -> Vector4<T> {
        let Point4 { x, y, z, w } = self;
        Vector4 { x, y, z, w }
    }
}

/// `[x, y, z, w].into()`. Upstream: `From<[T; 4]>`.
pub impl Point4FromArray<T> of Into<[T; 4], Point4<T>> {
    #[inline(always)]
    fn into(self: [T; 4]) -> Point4<T> {
        let [x, y, z, w] = self;
        Point4 { x, y, z, w }
    }
}

/// The coordinates as an array `[x, y, z, w]`. Upstream: `Into<[T; 4]>`.
pub impl Point4IntoArray<T> of Into<Point4<T>, [T; 4]> {
    #[inline(always)]
    fn into(self: Point4<T>) -> [T; 4] {
        let Point4 { x, y, z, w } = self;
        [x, y, z, w]
    }
}

/// `p[i]`: the coordinate `i` (`x, y, z, w`). Panics with `nalgebra: index out of bounds` for
/// `i > 3`. Upstream: `Index<usize> for Point`.
pub impl Point4Index<T, +Copy<T>, +Drop<T>> of Index<Point4<T>, usize> {
    type Target = T;

    fn index(ref self: Point4<T>, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point4PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point4<T>> {
    #[inline(always)]
    fn lt(lhs: Point4<T>, rhs: Point4<T>) -> bool {
        lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w
    }

    #[inline(always)]
    fn le(lhs: Point4<T>, rhs: Point4<T>) -> bool {
        lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w
    }

    #[inline(always)]
    fn gt(lhs: Point4<T>, rhs: Point4<T>) -> bool {
        lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w
    }

    #[inline(always)]
    fn ge(lhs: Point4<T>, rhs: Point4<T>) -> bool {
        lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w
    }
}
