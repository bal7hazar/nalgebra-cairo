//! Internal, no stability promise: the crate-private items of `base::point2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;
use crate::base::point2::Point2;

/// Kernels of upstream's crate-root free functions `nalgebra::center`, `nalgebra::distance_squared`
/// and `nalgebra::distance` (docs/API_PARITY.md, P21): crate-internal until those free functions
/// are ported, since upstream has no such METHOD on `Point` (WP 8.0: the public API is strictly
/// upstream's).
pub trait Point2InternalTrait<T> {
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

pub impl Point2InternalImpl<
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
