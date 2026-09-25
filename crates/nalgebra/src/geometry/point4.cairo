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

use core::num::traits::Bounded;
use core::ops::{AddAssign, DivAssign, Index, MulAssign, SubAssign};
use simba::scalar::Real;
use crate::base::vector4::Vector4;
use crate::base::vector5::Vector5;
use super::point::errors as point_errors;
use super::quaternion::ApproxEqTrait;

/// A point in the 4-dimensional space. The coordinates are fields, like `Point2` / `Point3`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Point4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

/// Operations of `Point4<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Point4Impl<
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
> of Point4Trait<T> {
    /// The point of coordinates `(x, y, z, w)`. Exact. Upstream: `Point4::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T) -> Point4<T> {
        Point4 { x, y, z, w }
    }

    /// The origin. Exact. Upstream: `origin`.
    #[inline(always)]
    fn origin() -> Point4<T> {
        Point4 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero() }
    }

    /// The point at the position `v`. Exact. Upstream: `from_coordinates` (deprecated upstream for
    /// `From`), also `v.into()`.
    #[inline(always)]
    fn from_coordinates(v: Vector4<T>) -> Point4<T> {
        Point4 { x: v.x, y: v.y, z: v.z, w: v.w }
    }

    /// The position vector. Upstream: the `coords` field.
    #[inline(always)]
    fn coords(self: Point4<T>) -> Vector4<T> {
        Vector4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }

    /// The homogeneous coordinates `(x, y, z, w, 1)`. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Point4<T>) -> Vector5<T> {
        Vector5 { x: self.x, y: self.y, z: self.z, w: self.w, a: R::one() }
    }

    /// The point of homogeneous coordinates `v`: the first 4 divided by the last (correctly
    /// rounded divisions), or `None` when the last is zero. Upstream: `from_homogeneous`.
    #[inline(always)]
    fn from_homogeneous(v: Vector5<T>) -> Option<Point4<T>> {
        if v.a == R::zero() {
            None
        } else {
            let (x, y, z, w) = R::div4(v.x, v.y, v.z, v.w, v.a);
            Some(Point4 { x, y, z, w })
        }
    }

    /// `self - rhs`, a vector. Exact; panics on overflow. Upstream: `Sub<Point>`.
    #[inline(always)]
    fn sub_point(self: Point4<T>, rhs: Point4<T>) -> Vector4<T> {
        Vector4 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
    }

    /// `self + v`. Exact; panics on overflow. Upstream: `Add<Vector>`.
    #[inline(always)]
    fn add_vector(self: Point4<T>, v: Vector4<T>) -> Point4<T> {
        Point4 { x: self.x + v.x, y: self.y + v.y, z: self.z + v.z, w: self.w + v.w }
    }

    /// `self - v`. Exact; panics on overflow. Upstream: `Sub<Vector>`.
    #[inline(always)]
    fn sub_vector(self: Point4<T>, v: Vector4<T>) -> Point4<T> {
        Point4 { x: self.x - v.x, y: self.y - v.y, z: self.z - v.z, w: self.w - v.w }
    }

    /// `self * k`: every coordinate floored once. Upstream: `Mul<T>` (and `k * p`).
    #[inline(always)]
    fn scale(self: Point4<T>, k: T) -> Point4<T> {
        Point4 { x: self.x * k, y: self.y * k, z: self.z * k, w: self.w * k }
    }

    /// `self / k`: correctly rounded divisions. Panics with `Fixed: division by zero` for
    /// `k = 0`. Upstream: `Div<T>`.
    #[inline(always)]
    fn unscale(self: Point4<T>, k: T) -> Point4<T> {
        let (x, y, z, w) = R::div4(self.x, self.y, self.z, self.w, k);
        Point4 { x, y, z, w }
    }

    /// The coordinate-wise minimum. Upstream: `inf`.
    #[inline(always)]
    fn inf(self: Point4<T>, other: Point4<T>) -> Point4<T> {
        Point4 {
            x: R::min(self.x, other.x),
            y: R::min(self.y, other.y),
            z: R::min(self.z, other.z),
            w: R::min(self.w, other.w),
        }
    }

    /// The coordinate-wise maximum. Upstream: `sup`.
    #[inline(always)]
    fn sup(self: Point4<T>, other: Point4<T>) -> Point4<T> {
        Point4 {
            x: R::max(self.x, other.x),
            y: R::max(self.y, other.y),
            z: R::max(self.z, other.z),
            w: R::max(self.w, other.w),
        }
    }

    /// `(inf, sup)`. Upstream: `inf_sup`.
    #[inline(always)]
    fn inf_sup(self: Point4<T>, other: Point4<T>) -> (Point4<T>, Point4<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    /// `true` when every coordinate is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Point4<T>, other: Point4<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps)
            && R::abs_diff_eq(self.y, other.y, ulps)
            && R::abs_diff_eq(self.z, other.z, ulps)
            && R::abs_diff_eq(self.w, other.w, ulps)
    }

    /// `self + (rhs - self) · t`, one fused kernel per coordinate (`Real::lerp`). Upstream:
    /// `lerp`.
    #[inline(always)]
    fn lerp(self: Point4<T>, rhs: Point4<T>, t: T) -> Point4<T> {
        Point4 {
            x: R::lerp(self.x, rhs.x, t),
            y: R::lerp(self.y, rhs.y, t),
            z: R::lerp(self.z, rhs.z, t),
            w: R::lerp(self.w, rhs.w, t),
        }
    }

    /// `true` when every coordinate is `relative_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Point4<T>, other: Point4<T>, epsilon: u64, max_relative: T) -> bool {
        ApproxEqTrait::relative_eq(self.x, other.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.y, other.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.z, other.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.w, other.w, epsilon, max_relative)
    }

    /// `true` when every coordinate is `ulps_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Point4<T>, other: Point4<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.x, other.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.y, other.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.z, other.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.w, other.w, epsilon, max_ulps)
    }

    /// The same point with every coordinate converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Point<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Point4<T>) -> Point4<U> {
        Point4 { x: self.x.into(), y: self.y.into(), z: self.z.into(), w: self.w.into() }
    }

    /// The point of `f(c)` for every coordinate `c`, in order (`x, y, ..`). `f` is any closure or
    /// `Fn` value; Cairo closures take their arguments by value. Upstream: `map`.
    #[inline]
    fn map<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Drop<Func::Output>>(
        self: Point4<T>, f: F,
    ) -> Point4<Func::Output> {
        Point4 { x: f(self.x), y: f(self.y), z: f(self.z), w: f(self.w) }
    }

    /// Replaces every coordinate `c` by `f(c)`, in order. Upstream's closure is `FnMut(&mut T)`,
    /// writing through the reference; a Cairo closure cannot, so it RETURNS the new coordinate (its
    /// output converts `Into<T>`). Same bits as `map` followed by the conversion. Upstream:
    /// `apply`.
    #[inline]
    fn apply<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Into<Func::Output, T>>(
        ref self: Point4<T>, f: F,
    ) {
        self =
            Point4 {
                x: f(self.x).into(), y: f(self.y).into(), z: f(self.z).into(), w: f(self.w).into(),
            };
    }

    /// The point of coordinates `s[0], .., s[3]`. Panics with `nalgebra: wrong slice length`
    /// unless `s` holds exactly 4 elements (upstream's `from_row_slice` assertion). Upstream:
    /// `from_slice`.
    fn from_slice(s: Span<T>) -> Point4<T> {
        if s.len() != 4 {
            core::panic_with_felt252(point_errors::WRONG_SLICE_LENGTH);
        }
        Point4 { x: *s[0], y: *s[1], z: *s[2], w: *s[3] }
    }

    /// The number of coordinates, 4. Upstream: `len`.
    #[inline(always)]
    fn len(self: Point4<T>) -> usize {
        4
    }

    /// `false`: a point has 4 coordinates. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: Point4<T>) -> bool {
        false
    }

    /// Deprecated upstream: the distance between two consecutive coordinates in storage, 1.
    /// Upstream: `stride`.
    #[inline(always)]
    fn stride(self: Point4<T>) -> usize {
        1
    }

    /// The point whose coordinates are all the scalar's `MIN` (core `Bounded`). Upstream:
    /// `Bounded::min_value` (num's spelling; Cairo's `Bounded` holds constants).
    fn min_value<+Bounded<T>>() -> Point4<T> {
        Point4 { x: Bounded::MIN, y: Bounded::MIN, z: Bounded::MIN, w: Bounded::MIN }
    }

    /// The point whose coordinates are all the scalar's `MAX` (core `Bounded`). Upstream:
    /// `Bounded::max_value`.
    fn max_value<+Bounded<T>>() -> Point4<T> {
        Point4 { x: Bounded::MAX, y: Bounded::MAX, z: Bounded::MAX, w: Bounded::MAX }
    }
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
