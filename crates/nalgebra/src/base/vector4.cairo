//! `Vector4`: a statically sized 4-dimensional column vector (upstream `nalgebra::Vector4`).
//!
//! - `Vector4Trait` / `Vector4Impl`: constructors, component-wise operations, reductions, products,
//!   norms and interpolation, generic over a `simba::scalar::Real` scalar;
//! - `Vector4AngleTrait` / `Vector4AngleImpl`: `angle`, which additionally needs
//!   `simba::scalar::Transcendental`;
//! - operators `+`, `-`, unary `-`, `+=`, `-=` between vectors, `*=` and `/=` by a scalar, and
//!   conversions from / to `(T, T, T, T)` and `[T; 4]`: their impls live in this module, where the
//!   compiler finds them without any import.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use simba::scalar::{Real, Transcendental};
use super::vector2::Vector2;
use super::vector3::Vector3;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A 4-dimensional column vector.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Vector4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

/// Operations of `Vector4<T>` over a `Real` scalar. By value, unrolled, no loop.
pub trait Vector4Trait<T> {
    /// The vector `(x, y, z, w)`. Upstream: `Vector4::new`.
    fn new(x: T, y: T, z: T, w: T) -> Vector4<T>;
    /// The zero vector. Upstream: `Vector4::zeros`.
    fn zeros() -> Vector4<T>;
    /// The vector whose components all equal `elem`. Upstream: `Vector4::repeat`.
    fn repeat(elem: T) -> Vector4<T>;
    /// Alias of `repeat`. Upstream: `Vector4::from_element`.
    fn from_element(elem: T) -> Vector4<T>;
    /// The unit axis `(1, 0, 0, 0)`. Upstream: `Vector4::x` (`x_axis` returns a `Unit`, which is
    /// not ported yet).
    fn x() -> Vector4<T>;
    /// The unit axis `(0, 1, 0, 0)`. Upstream: `Vector4::y` (`y_axis` returns a `Unit`, which is
    /// not ported yet).
    fn y() -> Vector4<T>;
    /// The unit axis `(0, 0, 1, 0)`. Upstream: `Vector4::z` (`z_axis` returns a `Unit`, which is
    /// not ported yet).
    fn z() -> Vector4<T>;
    /// The unit axis `(0, 0, 0, 1)`. Upstream: `Vector4::w` (`w_axis` returns a `Unit`, which is
    /// not ported yet).
    fn w() -> Vector4<T>;
    /// The first two components. Upstream: the `xy` swizzle (`fixed_rows::<2>(0)`).
    fn xy(self: Vector4<T>) -> Vector2<T>;
    /// The first three components. Upstream: the `xyz` swizzle (`fixed_rows::<3>(0)`).
    fn xyz(self: Vector4<T>) -> Vector3<T>;
    /// `self * k`, each component floored once. Panics on overflow. Upstream: `scale`
    /// (`self * k`).
    fn scale(self: Vector4<T>, k: T) -> Vector4<T>;
    /// `self / k`, each component being the correctly rounded quotient. Panics on a zero `k` and
    /// on overflow. Upstream: `unscale` (`self / k`).
    ///
    /// One division per component on purpose: `scale(k.recip())` is cheaper but rounds `1 / k`
    /// first, which costs up to `|self|` ulp instead of 1 (measured by
    /// `bench_vector4_unscale__alt_recip` and `test_unscale_alt_recip_is_less_accurate`).
    /// Callers dividing many vectors by the same value should store its reciprocal and `scale`.
    fn unscale(self: Vector4<T>, k: T) -> Vector4<T>;
    /// Component-wise product, each component floored once. Panics on overflow. Upstream:
    /// `component_mul`.
    fn component_mul(self: Vector4<T>, rhs: Vector4<T>) -> Vector4<T>;
    /// Component-wise quotient, each component rounded to nearest. Panics on a zero component of
    /// `rhs` and on overflow. Upstream: `component_div`.
    fn component_div(self: Vector4<T>, rhs: Vector4<T>) -> Vector4<T>;
    /// Component-wise absolute value. Exact; panics on overflow (`|MIN|`). Upstream: `abs`.
    fn abs(self: Vector4<T>) -> Vector4<T>;
    /// Component-wise minimum (infimum). Exact. Upstream: `inf`.
    fn inf(self: Vector4<T>, other: Vector4<T>) -> Vector4<T>;
    /// Component-wise maximum (supremum). Exact. Upstream: `sup`.
    fn sup(self: Vector4<T>, other: Vector4<T>) -> Vector4<T>;
    /// `(self.inf(other), self.sup(other))`. Exact. Upstream: `inf_sup`.
    fn inf_sup(self: Vector4<T>, other: Vector4<T>) -> (Vector4<T>, Vector4<T>);
    /// The smallest component. Exact. Upstream: `min`.
    fn min(self: Vector4<T>) -> T;
    /// The largest component. Exact. Upstream: `max`.
    fn max(self: Vector4<T>) -> T;
    /// The smallest absolute value of a component. Panics on overflow (`|MIN|`). Upstream:
    /// `amin`.
    fn amin(self: Vector4<T>) -> T;
    /// The largest absolute value of a component (infinity norm). Panics on overflow (`|MIN|`).
    /// Upstream: `amax`.
    fn amax(self: Vector4<T>) -> T;
    /// Index (0, 1, 2, 3) of the smallest component, the first one on ties. Upstream: `imin`.
    fn imin(self: Vector4<T>) -> usize;
    /// Index (0, 1, 2, 3) of the largest component, the first one on ties. Upstream: `imax`.
    fn imax(self: Vector4<T>) -> usize;
    /// Index of the component with the smallest absolute value, the first one on ties. Panics on
    /// overflow (`|MIN|`). Upstream: `iamin`.
    fn iamin(self: Vector4<T>) -> usize;
    /// Index of the component with the largest absolute value, the first one on ties. Panics on
    /// overflow (`|MIN|`). Upstream: `iamax`.
    fn iamax(self: Vector4<T>) -> usize;
    /// Sum of the components. Exact; panics on overflow. Upstream: `sum`.
    fn sum(self: Vector4<T>) -> T;
    /// `true` when every component is zero. Upstream: `Zero::is_zero`.
    fn is_zero(self: Vector4<T>) -> bool;
    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// the matching component of `other`; cannot overflow. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    fn abs_diff_eq(self: Vector4<T>, other: Vector4<T>, ulps: u64) -> bool;
    /// Dot product, fused (`Real::sum_prod4`): the exact sum of products is floored once. Only
    /// the result must fit: panics on overflow. Upstream: `dot`.
    fn dot(self: Vector4<T>, rhs: Vector4<T>) -> T;
    /// Squared Euclidean norm, fused (floored once). Panics on overflow: above a norm of about
    /// 46 340 (Q32.32) only `norm` works. Upstream: `norm_squared`.
    fn norm_squared(self: Vector4<T>) -> T;
    /// Euclidean norm (`Real::norm4`): square root of the UNSCALED exact sum of squares, floored
    /// once. No intermediate overflow: only the result must fit, so the norm of
    /// `(1e6, 1e6, 1e6, 1e6)` is fine. Upstream: `norm`.
    fn norm(self: Vector4<T>) -> T;
    /// Alias of `norm_squared`. Upstream: `magnitude_squared`.
    fn magnitude_squared(self: Vector4<T>) -> T;
    /// Alias of `norm`. Upstream: `magnitude`.
    fn magnitude(self: Vector4<T>) -> T;
    /// `(self - rhs).norm()`. Panics when a component difference or the result overflows.
    /// Upstream: `metric_distance`.
    fn metric_distance(self: Vector4<T>, rhs: Vector4<T>) -> T;
    /// `self / self.norm()`: the floored norm, then one correctly rounded division per component
    /// (`unscale`). The error is about `1 + 1 / norm` ulp per component whatever the magnitude of
    /// `self`, from a few ulp up to the longest vector whose norm fits. Panics with a division by
    /// zero when the norm is zero, and on overflow when the norm does not fit. Upstream:
    /// `normalize`.
    ///
    /// The cheaper candidates are kept as benchmarks (`bench_vector4_normalize__alt_*`): one
    /// reciprocal of the norm then one product per component is off by about `norm` ulp and
    /// overflows for norms up to `2^-31`; `recip(sqrt(norm_squared))` also overflows for norms
    /// above 46 340 and has no precision left for short vectors.
    fn normalize(self: Vector4<T>) -> Vector4<T>;
    /// `Some(self.normalize())`, or `None` when the norm is `<= min_norm`. With `min_norm >= 0`
    /// it never divides by zero. Upstream: `try_normalize`.
    fn try_normalize(self: Vector4<T>, min_norm: T) -> Option<Vector4<T>>;
    /// `self` when its norm is `<= max`, otherwise `self.scale(max / norm)` like upstream. The
    /// ratio is floored, so the capped norm is short of `max` by up to about `norm / max` ulp and
    /// exceeds it by at most the final rounding of the components. `max` is expected to be
    /// `>= 0`. Panics only when the norm does not fit. Upstream: `cap_magnitude`.
    ///
    /// The more accurate and dearer `normalize().scale(max)` is kept as a benchmark
    /// (`bench_vector4_cap_magnitude__alt_normalize`).
    fn cap_magnitude(self: Vector4<T>, max: T) -> Vector4<T>;
    /// `self + (rhs - self) * t` per component (`Real::lerp`: exact difference and product, one
    /// floor rounding). `t` is not clamped; `t = 0` gives `self` and `t = 1` gives `rhs` exactly.
    /// Panics on overflow of the result. Upstream: `lerp` (`self * (1 - t) + rhs * t`).
    fn lerp(self: Vector4<T>, rhs: Vector4<T>, t: T) -> Vector4<T>;
}

/// `angle` needs inverse trigonometry, hence its own trait: scalars may implement `Real` only.
pub trait Vector4AngleTrait<T> {
    /// The smallest angle between two vectors, in `[0, π]` (up to the rounding of `atan2`); `0`
    /// when one of them is zero.
    ///
    /// Computed as `2 * atan2(|u - v|, |u + v|)` on the normalized vectors `u`, `v` (Kahan):
    /// unlike upstream's `acos(dot / (|a| * |b|))`, it cannot overflow on long vectors and stays
    /// accurate for nearly parallel ones. Panics when a norm does not fit. Upstream: `angle`.
    ///
    /// The robustness costs 57 530 gas against 23 910 for upstream's form, kept as
    /// `bench_vector4_angle__alt_acos`: that one returns exactly 0 for two directions 2^-20 rad
    /// apart (its cosine floors to 1) and panics on vectors whose norms multiply out of range.
    fn angle(self: Vector4<T>, other: Vector4<T>) -> T;
}

pub impl Vector4Impl<
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
> of Vector4Trait<T> {
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T) -> Vector4<T> {
        Vector4 { x, y, z, w }
    }

    #[inline(always)]
    fn zeros() -> Vector4<T> {
        Vector4 { x: R::ZERO, y: R::ZERO, z: R::ZERO, w: R::ZERO }
    }

    #[inline(always)]
    fn repeat(elem: T) -> Vector4<T> {
        Vector4 { x: elem, y: elem, z: elem, w: elem }
    }

    #[inline(always)]
    fn from_element(elem: T) -> Vector4<T> {
        Vector4 { x: elem, y: elem, z: elem, w: elem }
    }

    #[inline(always)]
    fn x() -> Vector4<T> {
        Vector4 { x: R::ONE, y: R::ZERO, z: R::ZERO, w: R::ZERO }
    }

    #[inline(always)]
    fn y() -> Vector4<T> {
        Vector4 { x: R::ZERO, y: R::ONE, z: R::ZERO, w: R::ZERO }
    }

    #[inline(always)]
    fn z() -> Vector4<T> {
        Vector4 { x: R::ZERO, y: R::ZERO, z: R::ONE, w: R::ZERO }
    }

    #[inline(always)]
    fn w() -> Vector4<T> {
        Vector4 { x: R::ZERO, y: R::ZERO, z: R::ZERO, w: R::ONE }
    }

    #[inline(always)]
    fn xy(self: Vector4<T>) -> Vector2<T> {
        Vector2 { x: self.x, y: self.y }
    }

    #[inline(always)]
    fn xyz(self: Vector4<T>) -> Vector3<T> {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }

    #[inline(always)]
    fn scale(self: Vector4<T>, k: T) -> Vector4<T> {
        Vector4 { x: self.x * k, y: self.y * k, z: self.z * k, w: self.w * k }
    }

    #[inline(always)]
    fn unscale(self: Vector4<T>, k: T) -> Vector4<T> {
        Vector4 {
            x: R::div(self.x, k), y: R::div(self.y, k), z: R::div(self.z, k), w: R::div(self.w, k),
        }
    }

    #[inline(always)]
    fn component_mul(self: Vector4<T>, rhs: Vector4<T>) -> Vector4<T> {
        Vector4 { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z, w: self.w * rhs.w }
    }

    #[inline(always)]
    fn component_div(self: Vector4<T>, rhs: Vector4<T>) -> Vector4<T> {
        Vector4 {
            x: R::div(self.x, rhs.x),
            y: R::div(self.y, rhs.y),
            z: R::div(self.z, rhs.z),
            w: R::div(self.w, rhs.w),
        }
    }

    #[inline(always)]
    fn abs(self: Vector4<T>) -> Vector4<T> {
        Vector4 { x: R::abs(self.x), y: R::abs(self.y), z: R::abs(self.z), w: R::abs(self.w) }
    }

    #[inline(always)]
    fn inf(self: Vector4<T>, other: Vector4<T>) -> Vector4<T> {
        Vector4 {
            x: R::min(self.x, other.x),
            y: R::min(self.y, other.y),
            z: R::min(self.z, other.z),
            w: R::min(self.w, other.w),
        }
    }

    #[inline(always)]
    fn sup(self: Vector4<T>, other: Vector4<T>) -> Vector4<T> {
        Vector4 {
            x: R::max(self.x, other.x),
            y: R::max(self.y, other.y),
            z: R::max(self.z, other.z),
            w: R::max(self.w, other.w),
        }
    }

    #[inline(always)]
    fn inf_sup(self: Vector4<T>, other: Vector4<T>) -> (Vector4<T>, Vector4<T>) {
        (Self::inf(self, other), Self::sup(self, other))
    }

    #[inline(always)]
    fn min(self: Vector4<T>) -> T {
        R::min(R::min(R::min(self.x, self.y), self.z), self.w)
    }

    #[inline(always)]
    fn max(self: Vector4<T>) -> T {
        R::max(R::max(R::max(self.x, self.y), self.z), self.w)
    }

    #[inline(always)]
    fn amin(self: Vector4<T>) -> T {
        R::min(R::min(R::min(R::abs(self.x), R::abs(self.y)), R::abs(self.z)), R::abs(self.w))
    }

    #[inline(always)]
    fn amax(self: Vector4<T>) -> T {
        R::max(R::max(R::max(R::abs(self.x), R::abs(self.y)), R::abs(self.z)), R::abs(self.w))
    }

    #[inline(always)]
    fn imin(self: Vector4<T>) -> usize {
        let (i, m): (usize, T) = if self.x <= self.y {
            (0, self.x)
        } else {
            (1, self.y)
        };
        let (j, p): (usize, T) = if self.z <= self.w {
            (2, self.z)
        } else {
            (3, self.w)
        };
        if m <= p {
            i
        } else {
            j
        }
    }

    #[inline(always)]
    fn imax(self: Vector4<T>) -> usize {
        let (i, m): (usize, T) = if self.x >= self.y {
            (0, self.x)
        } else {
            (1, self.y)
        };
        let (j, p): (usize, T) = if self.z >= self.w {
            (2, self.z)
        } else {
            (3, self.w)
        };
        if m >= p {
            i
        } else {
            j
        }
    }

    #[inline(always)]
    fn iamin(self: Vector4<T>) -> usize {
        Self::imin(Self::abs(self))
    }

    #[inline(always)]
    fn iamax(self: Vector4<T>) -> usize {
        Self::imax(Self::abs(self))
    }

    #[inline(always)]
    fn sum(self: Vector4<T>) -> T {
        self.x + self.y + self.z + self.w
    }

    #[inline(always)]
    fn is_zero(self: Vector4<T>) -> bool {
        self.x == R::ZERO && self.y == R::ZERO && self.z == R::ZERO && self.w == R::ZERO
    }

    #[inline(always)]
    fn abs_diff_eq(self: Vector4<T>, other: Vector4<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, other.x, ulps)
            && R::abs_diff_eq(self.y, other.y, ulps)
            && R::abs_diff_eq(self.z, other.z, ulps)
            && R::abs_diff_eq(self.w, other.w, ulps)
    }

    #[inline(always)]
    fn dot(self: Vector4<T>, rhs: Vector4<T>) -> T {
        R::sum_prod4(self.x, rhs.x, self.y, rhs.y, self.z, rhs.z, self.w, rhs.w)
    }

    #[inline(always)]
    fn norm_squared(self: Vector4<T>) -> T {
        R::norm_squared4(self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    fn norm(self: Vector4<T>) -> T {
        R::norm4(self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    fn magnitude_squared(self: Vector4<T>) -> T {
        R::norm_squared4(self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    fn magnitude(self: Vector4<T>) -> T {
        R::norm4(self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    fn metric_distance(self: Vector4<T>, rhs: Vector4<T>) -> T {
        R::norm4(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
    }

    #[inline(always)]
    fn normalize(self: Vector4<T>) -> Vector4<T> {
        Self::unscale(self, R::norm4(self.x, self.y, self.z, self.w))
    }

    #[inline(always)]
    fn try_normalize(self: Vector4<T>, min_norm: T) -> Option<Vector4<T>> {
        let n = R::norm4(self.x, self.y, self.z, self.w);
        if n <= min_norm {
            None
        } else {
            Some(Self::unscale(self, n))
        }
    }

    #[inline(always)]
    fn cap_magnitude(self: Vector4<T>, max: T) -> Vector4<T> {
        let n = R::norm4(self.x, self.y, self.z, self.w);
        if n <= max {
            self
        } else {
            Self::scale(self, R::div(max, n))
        }
    }

    #[inline(always)]
    fn lerp(self: Vector4<T>, rhs: Vector4<T>, t: T) -> Vector4<T> {
        Vector4 {
            x: R::lerp(self.x, rhs.x, t),
            y: R::lerp(self.y, rhs.y, t),
            z: R::lerp(self.z, rhs.z, t),
            w: R::lerp(self.w, rhs.w, t),
        }
    }
}

pub impl Vector4AngleImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Add<T>,
    +Sub<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
> of Vector4AngleTrait<T> {
    fn angle(self: Vector4<T>, other: Vector4<T>) -> T {
        let n1 = R::norm4(self.x, self.y, self.z, self.w);
        let n2 = R::norm4(other.x, other.y, other.z, other.w);
        if n1 == R::ZERO || n2 == R::ZERO {
            return R::ZERO;
        }
        let u = Vector4 {
            x: R::div(self.x, n1),
            y: R::div(self.y, n1),
            z: R::div(self.z, n1),
            w: R::div(self.w, n1),
        };
        let v = Vector4 {
            x: R::div(other.x, n2),
            y: R::div(other.y, n2),
            z: R::div(other.z, n2),
            w: R::div(other.w, n2),
        };
        let d = R::norm4(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
        let s = R::norm4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
        let half = Tr::atan2(d, s);
        half + half
    }
}

/// `lhs + rhs`, component-wise. Exact; panics on overflow. Upstream: `Add`.
pub impl Vector4Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Vector4<T>> {
    #[inline(always)]
    fn add(lhs: Vector4<T>, rhs: Vector4<T>) -> Vector4<T> {
        Vector4 { x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z, w: lhs.w + rhs.w }
    }
}

/// `lhs - rhs`, component-wise. Exact; panics on overflow. Upstream: `Sub`.
pub impl Vector4Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Vector4<T>> {
    #[inline(always)]
    fn sub(lhs: Vector4<T>, rhs: Vector4<T>) -> Vector4<T> {
        Vector4 { x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z, w: lhs.w - rhs.w }
    }
}

/// `-a`, component-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl Vector4Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Vector4<T>> {
    #[inline(always)]
    fn neg(a: Vector4<T>) -> Vector4<T> {
        Vector4 { x: -a.x, y: -a.y, z: -a.z, w: -a.w }
    }
}

/// `self += rhs`. Exact; panics on overflow. Upstream: `AddAssign`.
pub impl Vector4AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Vector4<T>, Vector4<T>> {
    #[inline(always)]
    fn add_assign(ref self: Vector4<T>, rhs: Vector4<T>) {
        self =
            Vector4 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w };
    }
}

/// `self -= rhs`. Exact; panics on overflow. Upstream: `SubAssign`.
pub impl Vector4SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Vector4<T>, Vector4<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Vector4<T>, rhs: Vector4<T>) {
        self =
            Vector4 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w };
    }
}

/// `self *= k` for a scalar `k`: `scale` in place (corelib's binary `*` is homogeneous, so `v * k`
/// is the named method `scale`). Upstream: `MulAssign<T>`.
pub impl Vector4MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Vector4<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Vector4<T>, rhs: T) {
        self = Vector4 { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs };
    }
}

/// `self /= k` for a scalar `k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Vector4DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Vector4<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Vector4<T>, rhs: T) {
        self =
            Vector4 {
                x: R::div(self.x, rhs),
                y: R::div(self.y, rhs),
                z: R::div(self.z, rhs),
                w: R::div(self.w, rhs),
            };
    }
}

/// `(x, y, z, w).into()`. Upstream: `From<(T, T, T, T)>`-style construction (`From<[T; 4]>`).
pub impl Vector4FromTuple<T> of Into<(T, T, T, T), Vector4<T>> {
    #[inline(always)]
    fn into(self: (T, T, T, T)) -> Vector4<T> {
        let (x, y, z, w) = self;
        Vector4 { x, y, z, w }
    }
}

/// The components as a tuple `(x, y, z, w)`.
pub impl Vector4IntoTuple<T> of Into<Vector4<T>, (T, T, T, T)> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> (T, T, T, T) {
        let Vector4 { x, y, z, w } = self;
        (x, y, z, w)
    }
}

/// `[x, y, z, w].into()`. Upstream: `From<[T; 4]>`.
pub impl Vector4FromArray<T> of Into<[T; 4], Vector4<T>> {
    #[inline(always)]
    fn into(self: [T; 4]) -> Vector4<T> {
        let [x, y, z, w] = self;
        Vector4 { x, y, z, w }
    }
}

/// The components as a fixed-size array `[x, y, z, w]`. Upstream: `Into<[T; 4]>`.
pub impl Vector4IntoArray<T> of Into<Vector4<T>, [T; 4]> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> [T; 4] {
        let Vector4 { x, y, z, w } = self;
        [x, y, z, w]
    }
}
