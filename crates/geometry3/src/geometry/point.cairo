//! Points (upstream `nalgebra::Point`): the panic messages shared by every point type. The
//! completion of `Point2` / `Point3` (`Point2ExtTrait` / `Point3ExtTrait`) lives in the facade
//! `nalgebra` (`geometry::point`); `Point2Index` / `Point2PartialOrd` (and the `Point3` ones) in
//! the modules of their types (`base::point2` / `base::point3`), where Cairo finds them.

use core::num::traits::Bounded;
use nalgebra_core::geometry::point::errors as point_errors;
use nalgebra_core::internal::geometry::quaternion::ApproxEqTrait;
use nalgebra_types3::base::point3::Point3;
use simba::scalar::Real;

/// The WP 8.4-P09a completion of `Point3<T>` (`crate::base::point3` is outside this package's
/// module; its operators and conversions live there). By value, unrolled, no loop.
#[generate_trait]
pub impl Point3ExtImpl<
    T, impl R: Real<T>, +Sub<T>, +Mul<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Point3ExtTrait<T> {
    /// `true` when every coordinate is `relative_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Point3<T>, other: Point3<T>, epsilon: u64, max_relative: T) -> bool {
        ApproxEqTrait::relative_eq(self.x, other.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.y, other.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.z, other.z, epsilon, max_relative)
    }

    /// `true` when every coordinate is `ulps_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Point3<T>, other: Point3<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.x, other.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.y, other.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.z, other.z, epsilon, max_ulps)
    }

    /// The same point with every coordinate converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Point<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Point3<T>) -> Point3<U> {
        Point3 { x: self.x.into(), y: self.y.into(), z: self.z.into() }
    }

    /// The point of `f(c)` for every coordinate `c`, in order (`x, y, ..`). `f` is any closure or
    /// `Fn` value; Cairo closures take their arguments by value. Upstream: `map`.
    #[inline]
    fn map<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Drop<Func::Output>>(
        self: Point3<T>, f: F,
    ) -> Point3<Func::Output> {
        Point3 { x: f(self.x), y: f(self.y), z: f(self.z) }
    }

    /// Replaces every coordinate `c` by `f(c)`, in order. Upstream's closure is `FnMut(&mut T)`,
    /// writing through the reference; a Cairo closure cannot, so it RETURNS the new coordinate (its
    /// output converts `Into<T>`). Same bits as `map` followed by the conversion. Upstream:
    /// `apply`.
    #[inline]
    fn apply<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Into<Func::Output, T>>(
        ref self: Point3<T>, f: F,
    ) {
        self = Point3 { x: f(self.x).into(), y: f(self.y).into(), z: f(self.z).into() };
    }

    /// The point of coordinates `s[0], .., s[2]`. Panics with `nalgebra: wrong slice length`
    /// unless `s` holds exactly 3 elements (upstream's `from_row_slice` assertion). Upstream:
    /// `from_slice`.
    fn from_slice(s: Span<T>) -> Point3<T> {
        if s.len() != 3 {
            core::panic_with_felt252(point_errors::WRONG_SLICE_LENGTH);
        }
        Point3 { x: *s[0], y: *s[1], z: *s[2] }
    }

    /// The number of coordinates, 3. Upstream: `len`.
    #[inline(always)]
    fn len(self: Point3<T>) -> usize {
        3
    }

    /// `false`: a point has 3 coordinates. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: Point3<T>) -> bool {
        false
    }

    /// Deprecated upstream: the distance between two consecutive coordinates in storage, 1.
    /// Upstream: `stride`.
    #[inline(always)]
    fn stride(self: Point3<T>) -> usize {
        1
    }

    /// The point whose coordinates are all the scalar's `MIN` (core `Bounded`). Upstream:
    /// `Bounded::min_value` (num's spelling; Cairo's `Bounded` holds constants).
    fn min_value<+Bounded<T>>() -> Point3<T> {
        Point3 { x: Bounded::MIN, y: Bounded::MIN, z: Bounded::MIN }
    }

    /// The point whose coordinates are all the scalar's `MAX` (core `Bounded`). Upstream:
    /// `Bounded::max_value`.
    fn max_value<+Bounded<T>>() -> Point3<T> {
        Point3 { x: Bounded::MAX, y: Bounded::MAX, z: Bounded::MAX }
    }
}
