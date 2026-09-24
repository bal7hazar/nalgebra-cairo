//! `Point6`: a 6-dimensional point (upstream `nalgebra::Point6`, i.e. `Point<T, 6>`), WP 8.4-P09a.
//!
//! The same API as `Point2` / `Point3` (`crate::base::point2`, which predate the move of points to
//! `geometry`) with the WP 8.4-P09a completion, written from one template for the sizes 1, 4, 5
//! and 6. The coordinates are the named fields of the matching vector shape `Vector6`
//! (`x, y, z, w, a, b`), which is what `coords()` returns; see `crate::base::point3` for the table
//! of the heterogeneous operators (`sub_point`, `add_vector`, `scale`...).
//! There is no `to_homogeneous` / `from_homogeneous`: the homogeneous coordinates of a 6D point
//! form a 7-vector, and the static shapes stop at 6 (DESIGN D4).
//!
//! Numeric contract (AGENTS.md): every operation is exact except the divisions (`unscale`,
//! `from_homogeneous`: correctly rounded) and `lerp` (one fused kernel per coordinate); nothing
//! wraps silently.

use core::num::traits::Bounded;
use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use simba::scalar::Real;
use crate::base::vector6::Vector6;
use super::point::errors as point_errors;
use super::quaternion::ApproxEqTrait;

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

/// Operations of `Point6<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Point6Impl<
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
> of Point6Trait<T> {
    /// The point of coordinates `(x, y, z, w, a, b)`. Exact. Upstream: `Point6::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> Point6<T> {
        Point6 { x, y, z, w, a, b }
    }

    /// The origin. Exact. Upstream: `origin`.
    #[inline(always)]
    fn origin() -> Point6<T> {
        Point6 {
            x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero(), b: R::zero(),
        }
    }

    /// The point at the position `v`. Exact. Upstream: `from_coordinates` (deprecated upstream for
    /// `From`), also `v.into()`.
    #[inline(always)]
    fn from_coordinates(v: Vector6<T>) -> Point6<T> {
        Point6 { x: v.x, y: v.y, z: v.z, w: v.w, a: v.a, b: v.b }
    }

    /// The position vector. Upstream: the `coords` field.
    #[inline(always)]
    fn coords(self: Point6<T>) -> Vector6<T> {
        Vector6 { x: self.x, y: self.y, z: self.z, w: self.w, a: self.a, b: self.b }
    }

    /// `self - rhs`, a vector. Exact; panics on overflow. Upstream: `Sub<Point>`.
    #[inline(always)]
    fn sub_point(self: Point6<T>, rhs: Point6<T>) -> Vector6<T> {
        Vector6 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }

    /// `self + v`. Exact; panics on overflow. Upstream: `Add<Vector>`.
    #[inline(always)]
    fn add_vector(self: Point6<T>, v: Vector6<T>) -> Point6<T> {
        Point6 {
            x: self.x + v.x,
            y: self.y + v.y,
            z: self.z + v.z,
            w: self.w + v.w,
            a: self.a + v.a,
            b: self.b + v.b,
        }
    }

    /// `self - v`. Exact; panics on overflow. Upstream: `Sub<Vector>`.
    #[inline(always)]
    fn sub_vector(self: Point6<T>, v: Vector6<T>) -> Point6<T> {
        Point6 {
            x: self.x - v.x,
            y: self.y - v.y,
            z: self.z - v.z,
            w: self.w - v.w,
            a: self.a - v.a,
            b: self.b - v.b,
        }
    }

    /// `self * k`: every coordinate floored once. Upstream: `Mul<T>` (and `k * p`).
    #[inline(always)]
    fn scale(self: Point6<T>, k: T) -> Point6<T> {
        Point6 {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
            w: self.w * k,
            a: self.a * k,
            b: self.b * k,
        }
    }

    /// `self / k`: correctly rounded divisions. Panics with `Fixed: division by zero` for
    /// `k = 0`. Upstream: `Div<T>`.
    #[inline(always)]
    fn unscale(self: Point6<T>, k: T) -> Point6<T> {
        let (x, y, z, w, a, b) = R::div6(self.x, self.y, self.z, self.w, self.a, self.b, k);
        Point6 { x, y, z, w, a, b }
    }

    /// The coordinate-wise minimum. Upstream: `inf`.
    #[inline(always)]
    fn inf(self: Point6<T>, other: Point6<T>) -> Point6<T> {
        Point6 {
            x: R::min(self.x, other.x),
            y: R::min(self.y, other.y),
            z: R::min(self.z, other.z),
            w: R::min(self.w, other.w),
            a: R::min(self.a, other.a),
            b: R::min(self.b, other.b),
        }
    }

    /// The coordinate-wise maximum. Upstream: `sup`.
    #[inline(always)]
    fn sup(self: Point6<T>, other: Point6<T>) -> Point6<T> {
        Point6 {
            x: R::max(self.x, other.x),
            y: R::max(self.y, other.y),
            z: R::max(self.z, other.z),
            w: R::max(self.w, other.w),
            a: R::max(self.a, other.a),
            b: R::max(self.b, other.b),
        }
    }

    /// `(inf, sup)`. Upstream: `inf_sup`.
    #[inline(always)]
    fn inf_sup(self: Point6<T>, other: Point6<T>) -> (Point6<T>, Point6<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    /// `true` when every coordinate is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Point6<T>, other: Point6<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps)
            && R::abs_diff_eq(self.y, other.y, ulps)
            && R::abs_diff_eq(self.z, other.z, ulps)
            && R::abs_diff_eq(self.w, other.w, ulps)
            && R::abs_diff_eq(self.a, other.a, ulps)
            && R::abs_diff_eq(self.b, other.b, ulps)
    }

    /// `self + (rhs - self) · t`, one fused kernel per coordinate (`Real::lerp`). Upstream:
    /// `lerp`.
    #[inline(always)]
    fn lerp(self: Point6<T>, rhs: Point6<T>, t: T) -> Point6<T> {
        Point6 {
            x: R::lerp(self.x, rhs.x, t),
            y: R::lerp(self.y, rhs.y, t),
            z: R::lerp(self.z, rhs.z, t),
            w: R::lerp(self.w, rhs.w, t),
            a: R::lerp(self.a, rhs.a, t),
            b: R::lerp(self.b, rhs.b, t),
        }
    }

    /// `true` when every coordinate is `relative_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Point6<T>, other: Point6<T>, epsilon: u64, max_relative: T) -> bool {
        ApproxEqTrait::relative_eq(self.x, other.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.y, other.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.z, other.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.w, other.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.a, other.a, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.b, other.b, epsilon, max_relative)
    }

    /// `true` when every coordinate is `ulps_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Point6<T>, other: Point6<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.x, other.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.y, other.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.z, other.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.w, other.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.a, other.a, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.b, other.b, epsilon, max_ulps)
    }

    /// The same point with every coordinate converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Point<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Point6<T>) -> Point6<U> {
        Point6 {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into(),
            w: self.w.into(),
            a: self.a.into(),
            b: self.b.into(),
        }
    }

    /// The point of coordinates `s[0], .., s[5]`. Panics with `nalgebra: wrong slice length`
    /// unless `s` holds exactly 6 elements (upstream's `from_row_slice` assertion). Upstream:
    /// `from_slice`.
    fn from_slice(s: Span<T>) -> Point6<T> {
        if s.len() != 6 {
            core::panic_with_felt252(point_errors::WRONG_SLICE_LENGTH);
        }
        Point6 { x: *s[0], y: *s[1], z: *s[2], w: *s[3], a: *s[4], b: *s[5] }
    }

    /// The number of coordinates, 6. Upstream: `len`.
    #[inline(always)]
    fn len(self: Point6<T>) -> usize {
        6
    }

    /// `false`: a point has 6 coordinates. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: Point6<T>) -> bool {
        false
    }

    /// Deprecated upstream: the distance between two consecutive coordinates in storage, 1.
    /// Upstream: `stride`.
    #[inline(always)]
    fn stride(self: Point6<T>) -> usize {
        1
    }

    /// The point whose coordinates are all the scalar's `MIN` (core `Bounded`). Upstream:
    /// `Bounded::min_value` (num's spelling; Cairo's `Bounded` holds constants).
    fn min_value<+Bounded<T>>() -> Point6<T> {
        Point6 {
            x: Bounded::MIN,
            y: Bounded::MIN,
            z: Bounded::MIN,
            w: Bounded::MIN,
            a: Bounded::MIN,
            b: Bounded::MIN,
        }
    }

    /// The point whose coordinates are all the scalar's `MAX` (core `Bounded`). Upstream:
    /// `Bounded::max_value`.
    fn max_value<+Bounded<T>>() -> Point6<T> {
        Point6 {
            x: Bounded::MAX,
            y: Bounded::MAX,
            z: Bounded::MAX,
            w: Bounded::MAX,
            a: Bounded::MAX,
            b: Bounded::MAX,
        }
    }
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
