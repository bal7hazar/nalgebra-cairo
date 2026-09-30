//! In `nalgebra_geometry5`: the impl `Point5Impl`. This module is split over packages; the other
//! parts are in `nalgebra_types5`.
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

use core::num::traits::Bounded;
use nalgebra_core::geometry::point::errors as point_errors;
use nalgebra_core::internal::geometry::quaternion::ApproxEqTrait;
use nalgebra_types5::base::vector5::Vector5;
use nalgebra_types5::geometry::point5::Point5;
use nalgebra_types6::base::vector6::Vector6;
use simba::scalar::Real;

/// Operations of `Point5<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Point5Impl<
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
> of Point5Trait<T> {
    /// The point of coordinates `(x, y, z, w, a)`. Exact. Upstream: `Point5::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T) -> Point5<T> {
        Point5 { x, y, z, w, a }
    }

    /// The origin. Exact. Upstream: `origin`.
    #[inline(always)]
    fn origin() -> Point5<T> {
        Point5 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero() }
    }

    /// The point at the position `v`. Exact. Upstream: `from_coordinates` (deprecated upstream for
    /// `From`), also `v.into()`.
    #[inline(always)]
    fn from_coordinates(v: Vector5<T>) -> Point5<T> {
        Point5 { x: v.x, y: v.y, z: v.z, w: v.w, a: v.a }
    }

    /// The position vector. Upstream: the `coords` field.
    #[inline(always)]
    fn coords(self: Point5<T>) -> Vector5<T> {
        Vector5 { x: self.x, y: self.y, z: self.z, w: self.w, a: self.a }
    }

    /// The homogeneous coordinates `(x, y, z, w, a, 1)`. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Point5<T>) -> Vector6<T> {
        Vector6 { x: self.x, y: self.y, z: self.z, w: self.w, a: self.a, b: R::one() }
    }

    /// The point of homogeneous coordinates `v`: the first 5 divided by the last (correctly
    /// rounded divisions), or `None` when the last is zero. Upstream: `from_homogeneous`.
    #[inline(always)]
    fn from_homogeneous(v: Vector6<T>) -> Option<Point5<T>> {
        if v.b == R::zero() {
            None
        } else {
            let (x, y, z, w, a) = R::div5(v.x, v.y, v.z, v.w, v.a, v.b);
            Some(Point5 { x, y, z, w, a })
        }
    }

    /// `self - rhs`, a vector. Exact; panics on overflow. Upstream: `Sub<Point>`.
    #[inline(always)]
    fn sub_point(self: Point5<T>, rhs: Point5<T>) -> Vector5<T> {
        Vector5 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
            a: self.a - rhs.a,
        }
    }

    /// `self + v`. Exact; panics on overflow. Upstream: `Add<Vector>`.
    #[inline(always)]
    fn add_vector(self: Point5<T>, v: Vector5<T>) -> Point5<T> {
        Point5 {
            x: self.x + v.x, y: self.y + v.y, z: self.z + v.z, w: self.w + v.w, a: self.a + v.a,
        }
    }

    /// `self - v`. Exact; panics on overflow. Upstream: `Sub<Vector>`.
    #[inline(always)]
    fn sub_vector(self: Point5<T>, v: Vector5<T>) -> Point5<T> {
        Point5 {
            x: self.x - v.x, y: self.y - v.y, z: self.z - v.z, w: self.w - v.w, a: self.a - v.a,
        }
    }

    /// `self * k`: every coordinate floored once. Upstream: `Mul<T>` (and `k * p`).
    #[inline(always)]
    fn scale(self: Point5<T>, k: T) -> Point5<T> {
        Point5 { x: self.x * k, y: self.y * k, z: self.z * k, w: self.w * k, a: self.a * k }
    }

    /// `self / k`: correctly rounded divisions. Panics with `Fixed: division by zero` for
    /// `k = 0`. Upstream: `Div<T>`.
    #[inline(always)]
    fn unscale(self: Point5<T>, k: T) -> Point5<T> {
        let (x, y, z, w, a) = R::div5(self.x, self.y, self.z, self.w, self.a, k);
        Point5 { x, y, z, w, a }
    }

    /// The coordinate-wise minimum. Upstream: `inf`.
    #[inline(always)]
    fn inf(self: Point5<T>, other: Point5<T>) -> Point5<T> {
        Point5 {
            x: R::min(self.x, other.x),
            y: R::min(self.y, other.y),
            z: R::min(self.z, other.z),
            w: R::min(self.w, other.w),
            a: R::min(self.a, other.a),
        }
    }

    /// The coordinate-wise maximum. Upstream: `sup`.
    #[inline(always)]
    fn sup(self: Point5<T>, other: Point5<T>) -> Point5<T> {
        Point5 {
            x: R::max(self.x, other.x),
            y: R::max(self.y, other.y),
            z: R::max(self.z, other.z),
            w: R::max(self.w, other.w),
            a: R::max(self.a, other.a),
        }
    }

    /// `(inf, sup)`. Upstream: `inf_sup`.
    #[inline(always)]
    fn inf_sup(self: Point5<T>, other: Point5<T>) -> (Point5<T>, Point5<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    /// `true` when every coordinate is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Point5<T>, other: Point5<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps)
            && R::abs_diff_eq(self.y, other.y, ulps)
            && R::abs_diff_eq(self.z, other.z, ulps)
            && R::abs_diff_eq(self.w, other.w, ulps)
            && R::abs_diff_eq(self.a, other.a, ulps)
    }

    /// `self + (rhs - self) · t`, one fused kernel per coordinate (`Real::lerp`). Upstream:
    /// `lerp`.
    #[inline(always)]
    fn lerp(self: Point5<T>, rhs: Point5<T>, t: T) -> Point5<T> {
        Point5 {
            x: R::lerp(self.x, rhs.x, t),
            y: R::lerp(self.y, rhs.y, t),
            z: R::lerp(self.z, rhs.z, t),
            w: R::lerp(self.w, rhs.w, t),
            a: R::lerp(self.a, rhs.a, t),
        }
    }

    /// `true` when every coordinate is `relative_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Point5<T>, other: Point5<T>, epsilon: u64, max_relative: T) -> bool {
        ApproxEqTrait::relative_eq(self.x, other.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.y, other.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.z, other.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.w, other.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.a, other.a, epsilon, max_relative)
    }

    /// `true` when every coordinate is `ulps_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Point5<T>, other: Point5<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.x, other.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.y, other.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.z, other.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.w, other.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.a, other.a, epsilon, max_ulps)
    }

    /// The same point with every coordinate converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Point<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Point5<T>) -> Point5<U> {
        Point5 {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into(),
            w: self.w.into(),
            a: self.a.into(),
        }
    }

    /// The point of `f(c)` for every coordinate `c`, in order (`x, y, ..`). `f` is any closure or
    /// `Fn` value; Cairo closures take their arguments by value. Upstream: `map`.
    #[inline]
    fn map<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Drop<Func::Output>>(
        self: Point5<T>, f: F,
    ) -> Point5<Func::Output> {
        Point5 { x: f(self.x), y: f(self.y), z: f(self.z), w: f(self.w), a: f(self.a) }
    }

    /// Replaces every coordinate `c` by `f(c)`, in order. Upstream's closure is `FnMut(&mut T)`,
    /// writing through the reference; a Cairo closure cannot, so it RETURNS the new coordinate (its
    /// output converts `Into<T>`). Same bits as `map` followed by the conversion. Upstream:
    /// `apply`.
    #[inline]
    fn apply<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Into<Func::Output, T>>(
        ref self: Point5<T>, f: F,
    ) {
        self =
            Point5 {
                x: f(self.x).into(),
                y: f(self.y).into(),
                z: f(self.z).into(),
                w: f(self.w).into(),
                a: f(self.a).into(),
            };
    }

    /// The point of coordinates `s[0], .., s[4]`. Panics with `nalgebra: wrong slice length`
    /// unless `s` holds exactly 5 elements (upstream's `from_row_slice` assertion). Upstream:
    /// `from_slice`.
    fn from_slice(s: Span<T>) -> Point5<T> {
        if s.len() != 5 {
            core::panic_with_felt252(point_errors::WRONG_SLICE_LENGTH);
        }
        Point5 { x: *s[0], y: *s[1], z: *s[2], w: *s[3], a: *s[4] }
    }

    /// The number of coordinates, 5. Upstream: `len`.
    #[inline(always)]
    fn len(self: Point5<T>) -> usize {
        5
    }

    /// `false`: a point has 5 coordinates. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: Point5<T>) -> bool {
        false
    }

    /// Deprecated upstream: the distance between two consecutive coordinates in storage, 1.
    /// Upstream: `stride`.
    #[inline(always)]
    fn stride(self: Point5<T>) -> usize {
        1
    }

    /// The point whose coordinates are all the scalar's `MIN` (core `Bounded`). Upstream:
    /// `Bounded::min_value` (num's spelling; Cairo's `Bounded` holds constants).
    fn min_value<+Bounded<T>>() -> Point5<T> {
        Point5 {
            x: Bounded::MIN, y: Bounded::MIN, z: Bounded::MIN, w: Bounded::MIN, a: Bounded::MIN,
        }
    }

    /// The point whose coordinates are all the scalar's `MAX` (core `Bounded`). Upstream:
    /// `Bounded::max_value`.
    fn max_value<+Bounded<T>>() -> Point5<T> {
        Point5 {
            x: Bounded::MAX, y: Bounded::MAX, z: Bounded::MAX, w: Bounded::MAX, a: Bounded::MAX,
        }
    }
}
