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

use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::point2::Point2;
use super::vector3::Vector3;
use super::vector4::Vector4;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A 3-dimensional point.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// Operations of `Point3<T>` over a `Real` scalar. By value, unrolled, no loop.
pub trait Point3Trait<T> {
    /// The point `(x, y, z)`. Upstream: `Point3::new`.
    fn new(x: T, y: T, z: T) -> Point3<T>;
    /// The origin `(0, 0, 0)`. Upstream: `Point3::origin`.
    fn origin() -> Point3<T>;
    /// The point at the position `v`. Upstream: `Point3::from(v)` (`From<Vector3>`), also
    /// available as `v.into()`.
    fn from_coordinates(v: Vector3<T>) -> Point3<T>;
    /// The position vector of the point (its coordinates). Upstream: the `coords` field.
    fn coords(self: Point3<T>) -> Vector3<T>;
    /// The first two coordinates, as a point. Upstream: the `xy` swizzle.
    fn xy(self: Point3<T>) -> Point2<T>;
    /// `(x, y, z, 1)`: homogeneous coordinates of a point (as opposed to a vector, whose last
    /// component is `0`). Upstream: `to_homogeneous`.
    fn to_homogeneous(self: Point3<T>) -> Vector4<T>;
    /// The point of homogeneous coordinates `v`: `(x / w, y / w, z / w)`, each component the
    /// correctly rounded quotient, or `None` when `w = 0`. Panics on overflow of a quotient.
    /// Upstream: `from_homogeneous`.
    ///
    /// One division per component on purpose, like upstream (`v.xyz() / v.w`, see
    /// `Vector3Trait::unscale`), through one prepared divisor (`Real::div3`): multiplying by the
    /// rounded reciprocal of `w` would cost up to `|x|` ulp instead of 1 for 17 % less gas on
    /// `fixed` 0.3.0 (8 090 against 9 710, `bench_point3_from_homogeneous__alt_recip`,
    /// `test_from_homogeneous_alt_recip_is_less_accurate`).
    fn from_homogeneous(v: Vector4<T>) -> Option<Point3<T>>;
    /// `self - rhs`: the displacement vector from `rhs` to `self`. Exact; panics on overflow.
    /// Upstream: `Sub<Point> for Point` (`p - q`).
    fn sub_point(self: Point3<T>, rhs: Point3<T>) -> Vector3<T>;
    /// `self + v`: the point translated by `v`. Exact; panics on overflow. Upstream:
    /// `Add<Vector> for Point` (`p + v`).
    fn add_vector(self: Point3<T>, v: Vector3<T>) -> Point3<T>;
    /// `self - v`: the point translated by `-v`. Exact; panics on overflow. Upstream:
    /// `Sub<Vector> for Point` (`p - v`).
    fn sub_vector(self: Point3<T>, v: Vector3<T>) -> Point3<T>;
    /// `self * k`, each coordinate floored once. Panics on overflow. Upstream: `Mul<T> for
    /// Point` (`p * k`).
    fn scale(self: Point3<T>, k: T) -> Point3<T>;
    /// `self / k`, each coordinate being the correctly rounded quotient. Panics on a zero `k` and
    /// on overflow. Upstream: `Div<T> for Point` (`p / k`).
    fn unscale(self: Point3<T>, k: T) -> Point3<T>;
    /// Coordinate-wise minimum (infimum). Exact. Upstream: `inf`.
    fn inf(self: Point3<T>, other: Point3<T>) -> Point3<T>;
    /// Coordinate-wise maximum (supremum). Exact. Upstream: `sup`.
    fn sup(self: Point3<T>, other: Point3<T>) -> Point3<T>;
    /// `(self.inf(other), self.sup(other))`. Exact. Upstream: `inf_sup`.
    fn inf_sup(self: Point3<T>, other: Point3<T>) -> (Point3<T>, Point3<T>);
    /// `true` when every coordinate is within `ulps` smallest units (raw units for fixed point)
    /// of the matching coordinate of `other`; cannot overflow. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    fn abs_diff_eq(self: Point3<T>, other: Point3<T>, ulps: u64) -> bool;
    /// `self + (rhs - self) * t` per coordinate (`Real::lerp`: exact difference and product, one
    /// floor rounding). `t` is not clamped; `t = 0` gives `self` and `t = 1` gives `rhs`
    /// exactly. Panics on overflow of the result. Upstream: `lerp`.
    fn lerp(self: Point3<T>, rhs: Point3<T>, t: T) -> Point3<T>;
}

pub impl Point3Impl<
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
> of Point3Trait<T> {
    #[inline(always)]
    fn new(x: T, y: T, z: T) -> Point3<T> {
        Point3 { x, y, z }
    }

    #[inline(always)]
    fn origin() -> Point3<T> {
        Point3 { x: R::zero(), y: R::zero(), z: R::zero() }
    }

    #[inline(always)]
    fn from_coordinates(v: Vector3<T>) -> Point3<T> {
        Point3 { x: v.x, y: v.y, z: v.z }
    }

    #[inline(always)]
    fn coords(self: Point3<T>) -> Vector3<T> {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }

    #[inline(always)]
    fn xy(self: Point3<T>) -> Point2<T> {
        Point2 { x: self.x, y: self.y }
    }

    #[inline(always)]
    fn to_homogeneous(self: Point3<T>) -> Vector4<T> {
        Vector4 { x: self.x, y: self.y, z: self.z, w: R::one() }
    }

    #[inline(always)]
    fn from_homogeneous(v: Vector4<T>) -> Option<Point3<T>> {
        if v.w == R::zero() {
            None
        } else {
            let (x, y, z) = R::div3(v.x, v.y, v.z, v.w);
            Some(Point3 { x, y, z })
        }
    }

    #[inline(always)]
    fn sub_point(self: Point3<T>, rhs: Point3<T>) -> Vector3<T> {
        Vector3 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }

    #[inline(always)]
    fn add_vector(self: Point3<T>, v: Vector3<T>) -> Point3<T> {
        Point3 { x: self.x + v.x, y: self.y + v.y, z: self.z + v.z }
    }

    #[inline(always)]
    fn sub_vector(self: Point3<T>, v: Vector3<T>) -> Point3<T> {
        Point3 { x: self.x - v.x, y: self.y - v.y, z: self.z - v.z }
    }

    #[inline(always)]
    fn scale(self: Point3<T>, k: T) -> Point3<T> {
        Point3 { x: self.x * k, y: self.y * k, z: self.z * k }
    }

    #[inline(always)]
    fn unscale(self: Point3<T>, k: T) -> Point3<T> {
        let (x, y, z) = R::div3(self.x, self.y, self.z, k);
        Point3 { x, y, z }
    }

    #[inline(always)]
    fn inf(self: Point3<T>, other: Point3<T>) -> Point3<T> {
        Point3 {
            x: R::min(self.x, other.x), y: R::min(self.y, other.y), z: R::min(self.z, other.z),
        }
    }

    #[inline(always)]
    fn sup(self: Point3<T>, other: Point3<T>) -> Point3<T> {
        Point3 {
            x: R::max(self.x, other.x), y: R::max(self.y, other.y), z: R::max(self.z, other.z),
        }
    }

    #[inline(always)]
    fn inf_sup(self: Point3<T>, other: Point3<T>) -> (Point3<T>, Point3<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    #[inline(always)]
    fn abs_diff_eq(self: Point3<T>, other: Point3<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps)
            && R::abs_diff_eq(self.y, other.y, ulps)
            && R::abs_diff_eq(self.z, other.z, ulps)
    }

    #[inline(always)]
    fn lerp(self: Point3<T>, rhs: Point3<T>, t: T) -> Point3<T> {
        Point3 {
            x: R::lerp(self.x, rhs.x, t),
            y: R::lerp(self.y, rhs.y, t),
            z: R::lerp(self.z, rhs.z, t),
        }
    }
}

/// Kernels of upstream's crate-root free functions `nalgebra::center`, `nalgebra::distance_squared`
/// and `nalgebra::distance` (docs/API_PARITY.md, P21): crate-internal until those free functions
/// are ported, since upstream has no such METHOD on `Point` (WP 8.0: the public API is strictly
/// upstream's).
pub(crate) trait Point3InternalTrait<T> {
    /// The midpoint of `self` and `rhs`: the exact floor of `(self + rhs) / 2` per coordinate (a
    /// fused `sum_prod2` with the constant `1/2`, so the sum cannot overflow). Upstream:
    /// `nalgebra::center` (free function, `(p + q) * 0.5`).
    ///
    /// Measured (Sierra gas net of the baseline): 6 050, against 6 350 for `lerp(rhs, 1/2)`,
    /// 7 670 for upstream's `(p + q) * 1/2` and 8 570 for `(p + q) / 2` (both of which also
    /// overflow on the sum). The candidates are kept as benchmarks
    /// (`bench_point3_center__alt_*`).
    fn center(self: Point3<T>, rhs: Point3<T>) -> Point3<T>;
    /// Squared distance to `rhs`, fused (floored once). Panics when a coordinate difference
    /// overflows, and on overflow of the result: above a distance of about 46 340 (Q32.32) only
    /// `distance` works. Upstream: `nalgebra::distance_squared` (free function).
    fn distance_squared(self: Point3<T>, rhs: Point3<T>) -> T;
    /// Distance to `rhs` (`Real::norm3` of the coordinate differences: square root of the
    /// UNSCALED exact sum of squares, floored once). No intermediate overflow: only the result
    /// and the differences must fit. Upstream: `nalgebra::distance` (free function).
    fn distance(self: Point3<T>, rhs: Point3<T>) -> T;
}

pub(crate) impl Point3InternalImpl<
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
> of Point3InternalTrait<T> {
    #[inline(always)]
    fn center(self: Point3<T>, rhs: Point3<T>) -> Point3<T> {
        Point3 {
            x: R::sum_prod2(self.x, R::HALF, rhs.x, R::HALF),
            y: R::sum_prod2(self.y, R::HALF, rhs.y, R::HALF),
            z: R::sum_prod2(self.z, R::HALF, rhs.z, R::HALF),
        }
    }
    #[inline(always)]
    fn distance_squared(self: Point3<T>, rhs: Point3<T>) -> T {
        R::norm_squared3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
    #[inline(always)]
    fn distance(self: Point3<T>, rhs: Point3<T>) -> T {
        R::norm3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
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
