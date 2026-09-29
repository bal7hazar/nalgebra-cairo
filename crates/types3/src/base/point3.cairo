//! `Point3`: a 3-dimensional point (upstream `nalgebra::Point3`, which lives in `geometry`).
//!
//! A point is an affine position, as opposed to a `Vector3` displacement: points cannot be added
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
//! - `Point3Trait` / `Point3Impl`: constructors, conversions, affine operations and
//!   interpolation, generic over a `simba::scalar::Real` scalar (the distances and the midpoint
//!   are upstream's crate-root free functions, crate-internal kernels here until they are ported);
//! - operators and conversions from / to `[T; 3]` and `Vector3<T>`: their impls live
//!   in this module, where the compiler finds them without any import.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use nalgebra_core::geometry::point::errors as point_errors;
use simba::scalar::Real;
use crate::base::vector3::Vector3;

/// A 3-dimensional point.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// `-p`, coordinate-wise (the point mirrored through the origin). Exact; panics on overflow
/// (`-MIN`). Upstream: `Neg`.
pub impl Point3Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Point3<T>> {
    #[inline(always)]
    fn neg(a: Point3<T>) -> Point3<T> {
        Point3 { x: -a.x, y: -a.y, z: -a.z }
    }
}

/// `p += v`: translation by a vector, in place. Exact; panics on overflow. Upstream:
/// `AddAssign<Vector>`.
pub impl Point3AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Point3<T>, Vector3<T>> {
    #[inline(always)]
    fn add_assign(ref self: Point3<T>, rhs: Vector3<T>) {
        self = Point3 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z };
    }
}

/// `p -= v`: translation by `-v`, in place. Exact; panics on overflow. Upstream:
/// `SubAssign<Vector>`.
pub impl Point3SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Point3<T>, Vector3<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Point3<T>, rhs: Vector3<T>) {
        self = Point3 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z };
    }
}

/// `p *= k` for a scalar `k`: `scale` in place. Upstream: `MulAssign<T>`.
pub impl Point3MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Point3<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Point3<T>, rhs: T) {
        self = Point3 { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs };
    }
}

/// `p /= k` for a scalar `k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Point3DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Point3<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Point3<T>, rhs: T) {
        let (x, y, z) = R::div3(self.x, self.y, self.z, rhs);
        self = Point3 { x, y, z };
    }
}

/// `v.into()`: the point at the position `v`. Upstream: `From<Vector3> for Point3`.
pub impl Point3FromVector<T> of Into<Vector3<T>, Point3<T>> {
    #[inline(always)]
    fn into(self: Vector3<T>) -> Point3<T> {
        let Vector3 { x, y, z } = self;
        Point3 { x, y, z }
    }
}

/// The position vector of the point. Upstream: the `coords` field.
pub impl Point3IntoVector<T> of Into<Point3<T>, Vector3<T>> {
    #[inline(always)]
    fn into(self: Point3<T>) -> Vector3<T> {
        let Point3 { x, y, z } = self;
        Vector3 { x, y, z }
    }
}

/// `[x, y, z].into()`. Upstream: `From<[T; 3]>`.
pub impl Point3FromArray<T> of Into<[T; 3], Point3<T>> {
    #[inline(always)]
    fn into(self: [T; 3]) -> Point3<T> {
        let [x, y, z] = self;
        Point3 { x, y, z }
    }
}

/// The coordinates as a fixed-size array `[x, y, z]`. Upstream: `Into<[T; 3]>`.
pub impl Point3IntoArray<T> of Into<Point3<T>, [T; 3]> {
    #[inline(always)]
    fn into(self: Point3<T>) -> [T; 3] {
        let Point3 { x, y, z } = self;
        [x, y, z]
    }
}

/// `p[i]`: the coordinate `i` (`x, y, z`). Panics with `nalgebra: index out of bounds` for
/// `i > 2`. Upstream: `Index<usize> for Point`.
pub impl Point3Index<T, +Copy<T>, +Drop<T>> of Index<Point3<T>, usize> {
    type Target = T;

    fn index(ref self: Point3<T>, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point3PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point3<T>> {
    #[inline(always)]
    fn lt(lhs: Point3<T>, rhs: Point3<T>) -> bool {
        lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z
    }

    #[inline(always)]
    fn le(lhs: Point3<T>, rhs: Point3<T>) -> bool {
        lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z
    }

    #[inline(always)]
    fn gt(lhs: Point3<T>, rhs: Point3<T>) -> bool {
        lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z
    }

    #[inline(always)]
    fn ge(lhs: Point3<T>, rhs: Point3<T>) -> bool {
        lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z
    }
}
