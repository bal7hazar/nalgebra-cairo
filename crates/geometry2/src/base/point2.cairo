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

use nalgebra_types2::base::point2::Point2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types3::base::vector3::Vector3;
use simba::scalar::Real;

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
