//! Points (upstream `nalgebra::Point`, `geometry/point*.rs`): the panic messages shared by every
//! point type, and the WP 8.4-P09a completion of `Point2` / `Point3`.
//!
//! `Point2` / `Point3` live in `crate::base` (they predate the geometry module); `Point1`,
//! `Point4`, `Point5` and `Point6` live in `geometry::point1` ... `geometry::point6`, with the
//! whole API in one trait each. The completion of `Point2` / `Point3` is here, in
//! `Point2ExtTrait` / `Point3ExtTrait`: the approximate comparisons (`relative_eq`, `ulps_eq`),
//! `cast`, `from_slice`, `len` / `is_empty` / `stride`, `min_value` / `max_value` (upstream's
//! `Bounded`), and the impls `p[i]` (`Index<usize>`) and the component-wise partial order
//! (`PartialOrd`). Cairo finds an impl of a core trait in the module of its type or through an
//! import, so `Point2Index` / `Point2PartialOrd` (and the `Point3` ones) must be imported where
//! they are used.

use core::num::traits::Bounded;
use core::ops::Index;
use simba::scalar::Real;
use crate::base::point2::Point2;
use crate::base::point3::Point3;
use super::point::errors as point_errors;
use super::quaternion::ApproxEqTrait;

/// Panic messages of the point types (stable API).
pub mod errors {
    /// `p[i]` with `i` at least the dimension.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
    /// `from_slice` of a span whose length is not the dimension.
    pub const WRONG_SLICE_LENGTH: felt252 = 'nalgebra: wrong slice length';
}

/// The WP 8.4-P09a completion of `Point2<T>` (`crate::base::point2` is outside this package's
/// module; its operators and conversions live there). By value, unrolled, no loop.
#[generate_trait]
pub impl Point2ExtImpl<
    T, impl R: Real<T>, +Sub<T>, +Mul<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Point2ExtTrait<T> {
    /// `true` when every coordinate is `relative_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_relative` times the larger magnitude; see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Point2<T>, other: Point2<T>, epsilon: u64, max_relative: T) -> bool {
        ApproxEqTrait::relative_eq(self.x, other.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.y, other.y, epsilon, max_relative)
    }

    /// `true` when every coordinate is `ulps_eq` to the matching coordinate of `other` (within
    /// `epsilon` ulp, or `max_ulps` ulp without crossing zero; see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Point2<T>, other: Point2<T>, epsilon: u64, max_ulps: u32) -> bool {
        ApproxEqTrait::ulps_eq(self.x, other.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.y, other.y, epsilon, max_ulps)
    }

    /// The same point with every coordinate converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Point<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Point2<T>) -> Point2<U> {
        Point2 { x: self.x.into(), y: self.y.into() }
    }

    /// The point of `f(c)` for every coordinate `c`, in order (`x, y, ..`). `f` is any closure or
    /// `Fn` value; Cairo closures take their arguments by value. Upstream: `map`.
    #[inline]
    fn map<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Drop<Func::Output>>(
        self: Point2<T>, f: F,
    ) -> Point2<Func::Output> {
        Point2 { x: f(self.x), y: f(self.y) }
    }

    /// Replaces every coordinate `c` by `f(c)`, in order. Upstream's closure is `FnMut(&mut T)`,
    /// writing through the reference; a Cairo closure cannot, so it RETURNS the new coordinate (its
    /// output converts `Into<T>`). Same bits as `map` followed by the conversion. Upstream:
    /// `apply`.
    #[inline]
    fn apply<F, +Drop<F>, impl Func: core::ops::Fn<F, (T,)>, +Into<Func::Output, T>>(
        ref self: Point2<T>, f: F,
    ) {
        self = Point2 { x: f(self.x).into(), y: f(self.y).into() };
    }

    /// The point of coordinates `s[0], .., s[1]`. Panics with `nalgebra: wrong slice length`
    /// unless `s` holds exactly 2 elements (upstream's `from_row_slice` assertion). Upstream:
    /// `from_slice`.
    fn from_slice(s: Span<T>) -> Point2<T> {
        if s.len() != 2 {
            core::panic_with_felt252(point_errors::WRONG_SLICE_LENGTH);
        }
        Point2 { x: *s[0], y: *s[1] }
    }

    /// The number of coordinates, 2. Upstream: `len`.
    #[inline(always)]
    fn len(self: Point2<T>) -> usize {
        2
    }

    /// `false`: a point has 2 coordinates. Upstream: `is_empty`.
    #[inline(always)]
    fn is_empty(self: Point2<T>) -> bool {
        false
    }

    /// Deprecated upstream: the distance between two consecutive coordinates in storage, 1.
    /// Upstream: `stride`.
    #[inline(always)]
    fn stride(self: Point2<T>) -> usize {
        1
    }

    /// The point whose coordinates are all the scalar's `MIN` (core `Bounded`). Upstream:
    /// `Bounded::min_value` (num's spelling; Cairo's `Bounded` holds constants).
    fn min_value<+Bounded<T>>() -> Point2<T> {
        Point2 { x: Bounded::MIN, y: Bounded::MIN }
    }

    /// The point whose coordinates are all the scalar's `MAX` (core `Bounded`). Upstream:
    /// `Bounded::max_value`.
    fn max_value<+Bounded<T>>() -> Point2<T> {
        Point2 { x: Bounded::MAX, y: Bounded::MAX }
    }
}

/// `p[i]`: the coordinate `i` (`x, y`). Panics with `nalgebra: index out of bounds` for
/// `i > 1`. Upstream: `Index<usize> for Point`.
pub impl Point2Index<T, +Copy<T>, +Drop<T>> of Index<Point2<T>, usize> {
    type Target = T;

    fn index(ref self: Point2<T>, index: usize) -> T {
        match index {
            0 => self.x,
            1 => self.y,
            _ => core::panic_with_felt252(point_errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// The component-wise partial order of upstream: `p < q` (resp. `<=`, `>`, `>=`) when EVERY
/// coordinate of `p` is `<` (resp. ...) the matching coordinate of `q`; two points may be
/// incomparable (`!(p <= q) && !(q <= p)`). Upstream: `PartialOrd for Point` (the `Matrix` one).
pub impl Point2PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<Point2<T>> {
    #[inline(always)]
    fn lt(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x < rhs.x && lhs.y < rhs.y
    }

    #[inline(always)]
    fn le(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x <= rhs.x && lhs.y <= rhs.y
    }

    #[inline(always)]
    fn gt(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x > rhs.x && lhs.y > rhs.y
    }

    #[inline(always)]
    fn ge(lhs: Point2<T>, rhs: Point2<T>) -> bool {
        lhs.x >= rhs.x && lhs.y >= rhs.y
    }
}

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
