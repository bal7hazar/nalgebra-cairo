//! Internal, no stability promise: the crate-private items of `base::point3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types3::base::point3::Point3;
use simba::scalar::Real;

/// Kernels of upstream's crate-root free functions `nalgebra::center`, `nalgebra::distance_squared`
/// and `nalgebra::distance` (docs/API_PARITY.md, P21): crate-internal until those free functions
/// are ported, since upstream has no such METHOD on `Point` (WP 8.0: the public API is strictly
/// upstream's).
pub trait Point3InternalTrait<T> {
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

pub impl Point3InternalImpl<
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
