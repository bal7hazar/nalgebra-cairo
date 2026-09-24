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

use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::vector2::Vector2;
use super::vector3::Vector3;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A 2-dimensional point.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

/// Operations of `Point2<T>` over a `Real` scalar. By value, unrolled, no loop.
pub trait Point2Trait<T> {
    /// The point `(x, y)`. Upstream: `Point2::new`.
    fn new(x: T, y: T) -> Point2<T>;
    /// The origin `(0, 0)`. Upstream: `Point2::origin`.
    fn origin() -> Point2<T>;
    /// The point at the position `v`. Upstream: `Point2::from(v)` (`From<Vector2>`), also
    /// available as `v.into()`.
    fn from_coordinates(v: Vector2<T>) -> Point2<T>;
    /// The position vector of the point (its coordinates). Upstream: the `coords` field.
    fn coords(self: Point2<T>) -> Vector2<T>;
    /// `(x, y, 1)`: homogeneous coordinates of a point (as opposed to a vector, whose last
    /// component is `0`). Upstream: `to_homogeneous`.
    fn to_homogeneous(self: Point2<T>) -> Vector3<T>;
    /// The point of homogeneous coordinates `v`: `(x / w, y / w)`, each component the exactly
    /// correctly rounded quotient, or `None` when `w = 0`. Panics on overflow of a quotient.
    /// Upstream: `from_homogeneous`.
    ///
    /// One division per component on purpose, like upstream (`v.xy() / v.z`, see
    /// `Vector2Trait::unscale`): multiplying by the rounded reciprocal of `w` would cost up to
    /// `|x|` ulp instead of 1 for 6 % less gas on `fixed` 0.3.0 (6 410 against 6 830,
    /// `bench_point2_from_homogeneous__alt_recip`,
    /// `test_from_homogeneous_alt_recip_is_less_accurate`).
    fn from_homogeneous(v: Vector3<T>) -> Option<Point2<T>>;
    /// `self - rhs`: the displacement vector from `rhs` to `self`. Exact; panics on overflow.
    /// Upstream: `Sub<Point> for Point` (`p - q`).
    fn sub_point(self: Point2<T>, rhs: Point2<T>) -> Vector2<T>;
    /// `self + v`: the point translated by `v`. Exact; panics on overflow. Upstream:
    /// `Add<Vector> for Point` (`p + v`).
    fn add_vector(self: Point2<T>, v: Vector2<T>) -> Point2<T>;
    /// `self - v`: the point translated by `-v`. Exact; panics on overflow. Upstream:
    /// `Sub<Vector> for Point` (`p - v`).
    fn sub_vector(self: Point2<T>, v: Vector2<T>) -> Point2<T>;
    /// `self * k`, each coordinate floored once. Panics on overflow. Upstream: `Mul<T> for
    /// Point` (`p * k`).
    fn scale(self: Point2<T>, k: T) -> Point2<T>;
    /// `self / k`, each coordinate being the correctly rounded quotient. Panics on a zero `k` and
    /// on overflow. Upstream: `Div<T> for Point` (`p / k`).
    fn unscale(self: Point2<T>, k: T) -> Point2<T>;
    /// Coordinate-wise minimum (infimum). Exact. Upstream: `inf`.
    fn inf(self: Point2<T>, other: Point2<T>) -> Point2<T>;
    /// Coordinate-wise maximum (supremum). Exact. Upstream: `sup`.
    fn sup(self: Point2<T>, other: Point2<T>) -> Point2<T>;
    /// `(self.inf(other), self.sup(other))`. Exact. Upstream: `inf_sup`.
    fn inf_sup(self: Point2<T>, other: Point2<T>) -> (Point2<T>, Point2<T>);
    /// `true` when every coordinate is within `ulps` smallest units (raw units for fixed point)
    /// of the matching coordinate of `other`; cannot overflow. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    fn abs_diff_eq(self: Point2<T>, other: Point2<T>, ulps: u64) -> bool;
    /// `self + (rhs - self) * t` per coordinate (`Real::lerp`: exact difference and product, one
    /// floor rounding). `t` is not clamped; `t = 0` gives `self` and `t = 1` gives `rhs`
    /// exactly. Panics on overflow of the result. Upstream: `lerp`.
    fn lerp(self: Point2<T>, rhs: Point2<T>, t: T) -> Point2<T>;
}

pub impl Point2Impl<
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
> of Point2Trait<T> {
    #[inline(always)]
    fn new(x: T, y: T) -> Point2<T> {
        Point2 { x, y }
    }

    #[inline(always)]
    fn origin() -> Point2<T> {
        Point2 { x: R::zero(), y: R::zero() }
    }

    #[inline(always)]
    fn from_coordinates(v: Vector2<T>) -> Point2<T> {
        Point2 { x: v.x, y: v.y }
    }

    #[inline(always)]
    fn coords(self: Point2<T>) -> Vector2<T> {
        Vector2 { x: self.x, y: self.y }
    }

    #[inline(always)]
    fn to_homogeneous(self: Point2<T>) -> Vector3<T> {
        Vector3 { x: self.x, y: self.y, z: R::one() }
    }

    #[inline(always)]
    fn from_homogeneous(v: Vector3<T>) -> Option<Point2<T>> {
        if v.z == R::zero() {
            None
        } else {
            Some(Point2 { x: R::div(v.x, v.z), y: R::div(v.y, v.z) })
        }
    }

    #[inline(always)]
    fn sub_point(self: Point2<T>, rhs: Point2<T>) -> Vector2<T> {
        Vector2 { x: self.x - rhs.x, y: self.y - rhs.y }
    }

    #[inline(always)]
    fn add_vector(self: Point2<T>, v: Vector2<T>) -> Point2<T> {
        Point2 { x: self.x + v.x, y: self.y + v.y }
    }

    #[inline(always)]
    fn sub_vector(self: Point2<T>, v: Vector2<T>) -> Point2<T> {
        Point2 { x: self.x - v.x, y: self.y - v.y }
    }

    #[inline(always)]
    fn scale(self: Point2<T>, k: T) -> Point2<T> {
        Point2 { x: self.x * k, y: self.y * k }
    }

    #[inline(always)]
    fn unscale(self: Point2<T>, k: T) -> Point2<T> {
        Point2 { x: R::div(self.x, k), y: R::div(self.y, k) }
    }

    #[inline(always)]
    fn inf(self: Point2<T>, other: Point2<T>) -> Point2<T> {
        Point2 { x: R::min(self.x, other.x), y: R::min(self.y, other.y) }
    }

    #[inline(always)]
    fn sup(self: Point2<T>, other: Point2<T>) -> Point2<T> {
        Point2 { x: R::max(self.x, other.x), y: R::max(self.y, other.y) }
    }

    #[inline(always)]
    fn inf_sup(self: Point2<T>, other: Point2<T>) -> (Point2<T>, Point2<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    #[inline(always)]
    fn abs_diff_eq(self: Point2<T>, other: Point2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps) && R::abs_diff_eq(self.y, other.y, ulps)
    }

    #[inline(always)]
    fn lerp(self: Point2<T>, rhs: Point2<T>, t: T) -> Point2<T> {
        Point2 { x: R::lerp(self.x, rhs.x, t), y: R::lerp(self.y, rhs.y, t) }
    }
}

/// Kernels of upstream's crate-root free functions `nalgebra::center`, `nalgebra::distance_squared`
/// and `nalgebra::distance` (docs/API_PARITY.md, P21): crate-internal until those free functions
/// are ported, since upstream has no such METHOD on `Point` (WP 8.0: the public API is strictly
/// upstream's).
pub(crate) trait Point2InternalTrait<T> {
    /// The midpoint of `self` and `rhs`: the exact floor of `(self + rhs) / 2` per coordinate (a
    /// fused `sum_prod2` with the constant `1/2`, so the sum cannot overflow). Upstream:
    /// `nalgebra::center` (free function, `(p + q) * 0.5`).
    ///
    /// Measured (Sierra gas net of the baseline): 4 000, against 4 200 for `lerp(rhs, 1/2)`,
    /// 5 080 for upstream's `(p + q) * 1/2` and 5 680 for `(p + q) / 2` (both of which also
    /// overflow on the sum). The candidates are kept as benchmarks
    /// (`bench_point2_center__alt_*`).
    fn center(self: Point2<T>, rhs: Point2<T>) -> Point2<T>;
    /// Squared distance to `rhs`, fused (floored once). Panics when a coordinate difference
    /// overflows, and on overflow of the result: above a distance of about 46 340 (Q32.32) only
    /// `distance` works. Upstream: `nalgebra::distance_squared` (free function).
    fn distance_squared(self: Point2<T>, rhs: Point2<T>) -> T;
    /// Distance to `rhs` (`Real::norm2` of the coordinate differences: square root of the
    /// UNSCALED exact sum of squares, floored once). No intermediate overflow: only the result
    /// and the differences must fit. Upstream: `nalgebra::distance` (free function).
    fn distance(self: Point2<T>, rhs: Point2<T>) -> T;
}

pub(crate) impl Point2InternalImpl<
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
> of Point2InternalTrait<T> {
    #[inline(always)]
    fn center(self: Point2<T>, rhs: Point2<T>) -> Point2<T> {
        Point2 {
            x: R::sum_prod2(self.x, R::HALF, rhs.x, R::HALF),
            y: R::sum_prod2(self.y, R::HALF, rhs.y, R::HALF),
        }
    }
    #[inline(always)]
    fn distance_squared(self: Point2<T>, rhs: Point2<T>) -> T {
        R::norm_squared2(self.x - rhs.x, self.y - rhs.y)
    }
    #[inline(always)]
    fn distance(self: Point2<T>, rhs: Point2<T>) -> T {
        R::norm2(self.x - rhs.x, self.y - rhs.y)
    }
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
